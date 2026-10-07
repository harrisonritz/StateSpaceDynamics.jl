#=============================================================================
noise_prototype.jl — should the costate innovation inherit the plant noise?

Step 0 before touching `src/`: three exact likelihoods for the same data, the
costate hidden throughout and only `y_t = C x_t + d + r_t` observed.

  M0  what `LQRStateModel` fits today. Hamiltonian forward map, a constant
      full-rank N(0, Σ_mix) on the mixed residual, a free 2n initial prior per
      condition, the soft terminal `λ_T = Q x_T`, scored as p(y | f = 0).
  M1  option (a). Same Hamiltonian mean and terminal; the forward innovation is
      the causal agent's, `[ε; P_{t+1} ε + ν]`, so its covariance is
      `L_{t+1} blkdiag(Σ, Ω) L_{t+1}ᵀ` with `L = [I 0; P I]` — shared plant noise
      plus costate-specific noise, full rank, time-varying through the Riccati
      sweep. `λ_1 = P_1 x_1 + ν_1`.
  M2  option (b). The causal closed loop on x alone,
      `x_{t+1} = Φ_t x_t + noise`, `Cov = Σ + W_{t+1} S Ω S W_{t+1}ᵀ` — plant noise
      plus control slack (an agent acting on a perturbed costate).

and three generators:

  G1  causal LQR agent, plant noise only     (`simulate_lqr`, the default)
  G2  causal LQR agent + costate slack        (`simulate_lqr(; costate_slack)`)
  G3  draws from M1 itself                    (terminal-conditioned, dense)

G1 and G2 are exact draws from M2 (G1 at Ω = 0), G3 from M1. M1 at Ω = 0 is M2
at Ω = 0 — checked in `--selftest` — so the models nest where they should.

What M1 is, in the (x, δ = λ − P x) coordinates (derivation in the companion
`noise_prototype.md`): `x_{t+1} = Φ_t x_t − S A⁻ᵀ δ_t + ε_t`,
`δ_{t+1} = Φ_t⁻ᵀ δ_t + ν_t`, pinned by `δ_T ≈ 0`. Its costate noise is a
*persistent, anticipatory* control error — a backward-stable process that
vanishes at the deadline — where M2's slack is white.

Everything is self-contained, as `biological.jl` is: nothing is imported from
StateSpaceDynamics. The observation model (C, d, R) is fixed at the truth in
every fit so the comparison is about the dynamics alone; it also pins the
similarity gauge, so plant matrices are compared directly. `tr(S)` is fixed at
its true value (the S/Q scale gauge).

Needs Optim, LineSearches and ForwardDiff on the load path:

  julia --project=<env with Optim, LineSearches, ForwardDiff> -t 4 \
        docs/dev/lqr/noise_prototype.jl [--selftest] [--quick] [--only=a,b]

Sections: selftest, timing, fisher, profile, em, fit. The fit runs as independent processes:

  for i in 1 2 3 4; do julia --project=<env> noise_prototype.jl --only=fit \
      --shard=$i/4 --out=fit_$i.jls & done; wait
  julia --project=<env> noise_prototype.jl --merge=fit_1.jls,fit_2.jls,fit_3.jls,fit_4.jls

(add `--plant=known` to every command for the known-plant variant).
=============================================================================#

using LinearAlgebra, Random, Statistics, Printf
using Optim, LineSearches, ForwardDiff

BLAS.set_num_threads(1)

const ONLY = let
    a = filter(s -> startswith(s, "--only="), ARGS)
    isempty(a) ? nothing : Set(split(split(a[1], "=")[2], ","))
end
const QUICK = "--quick" in ARGS
# `--plant=known` fixes A at the truth in every fit: what is left is the cost, the
# noise and the initial state, which separates "the noise model trades cost
# against plant" from "the noise model biases the cost".
const KNOWN_A = "--plant=known" in ARGS
const SELFTEST = "--selftest" in ARGS
want(s) = ONLY === nothing || s in ONLY
section(t) = (println(); println("="^78); println(t); println("="^78))

# =========================================================================
# Truth
# =========================================================================

const NX = 2          # plant dimension
const NY = 6          # observation dimension
const TT = 30         # trial length
const NTRAIN = QUICK ? 40 : 100   # trials per condition
const NTEST = 100
const NSEEDS = QUICK ? 1 : 4

function truth(; slack2=0.0, Ωg3=nothing)
    A = [1.0 0.1; 0.0 0.9]
    S = 0.05 * Matrix(1.0I, NX, NX)
    Qs = [Diagonal([1.0, 0.2]) |> Matrix, Diagonal([5.0, 0.4]) |> Matrix]
    Σ = 0.01 * Matrix(1.0I, NX, NX)
    Ω = Ωg3 === nothing ? slack2 * Matrix(1.0I, NX, NX) : Ωg3
    μ1 = [1.0, 0.0]
    V1 = 0.1 * Matrix(1.0I, NX, NX)
    C = randn(MersenneTwister(20261007), NY, NX) ./ sqrt(NX)
    d = zeros(NY)
    R = 0.05 * Matrix(1.0I, NY, NY)
    Σf = 1e-3 * Matrix(1.0I, NX, NX)
    return (; A, S, Qs, Σ, Ω, μ1, V1, C, d, R, Σf)
end

const TRUE_G1 = truth()
const TRUE_G2 = truth(; slack2=2.0)                    # S Ω S = Σ/2
const TRUE_G3 = truth(; Ωg3=0.25 * Matrix(1.0I, NX, NX))

# =========================================================================
# Control algebra (generic in the element type, for ForwardDiff)
# =========================================================================

sym(M) = (M + M') / 2
eye(T, k) = Matrix{T}(I, k, k)
blkdiag(X, Y) = [X zeros(eltype(X), size(X, 1), size(Y, 2));
                 zeros(eltype(X), size(Y, 1), size(X, 2)) Y]

"""Finite-horizon Riccati sweep with terminal cost `Qf`: `P[t]` for t = 1..T and
`W[t] = (I + S P[t])⁻¹` (W[1] is computed but only W[2:T] enter the dynamics)."""
function riccati(A, S, Q, Qf, T)
    Tp = promote_type(eltype(A), eltype(S), eltype(Q), eltype(Qf))
    P = Vector{Matrix{Tp}}(undef, T)
    W = Vector{Matrix{Tp}}(undef, T)
    P[T] = Matrix{Tp}(Qf)
    Id = eye(Tp, size(A, 1))
    for t in T:-1:2
        W[t] = inv(Id + S * P[t])
        P[t - 1] = sym(Q + A' * P[t] * W[t] * A)
    end
    W[1] = inv(Id + S * P[1])
    return P, W
end

"Forward Hamiltonian map, the `M_t` of `LQRStateModel`'s docstring."
function hamiltonian(A, S, Q)
    Ait = inv(A)'
    return [A + S * Ait * Q  -S * Ait;
            -Ait * Q          Ait]
end

"Mixed → forward noise map `G = [I  S A⁻ᵀ; 0  −A⁻ᵀ]`."
function gmix(A, S)
    Ait = inv(A)'
    k = size(A, 1)
    return [eye(eltype(Ait), k) S * Ait; zeros(eltype(Ait), k, k) -Ait]
end

ltrigraph(P) = (k = size(P, 1); [eye(eltype(P), k) zeros(eltype(P), k, k); P eye(eltype(P), k)])

# =========================================================================
# The three models as linear-Gaussian chains ("specs")
# =========================================================================
#
# A spec is z_1 ~ N(m1, V1), z_{t+1} = F[t] z_t + w_t, w_t ~ N(0, Qn[t]),
# y_t = H z_t + d + r_t, and optionally a terminal pseudo-observation
# 0 = Hf z_T + e_f, e_f ~ N(0, Σf), scored as p(y | f = 0).

function spec_M0(par, k, obs, T)
    Q = par.Qs[k]
    M = hamiltonian(par.A, par.S, Q)
    G = gmix(par.A, par.S)
    Qn = sym(G * par.Σmix * G')
    Tp = eltype(M)
    return (F=fill(M, T - 1), Qn=fill(Qn, T - 1), m1=par.m0[k], V1=par.V0[k],
            H=[obs.C zeros(size(obs.C, 1), size(Q, 1))], d=obs.d, R=obs.R,
            Hf=[-Q eye(Tp, size(Q, 1))], Σf=par.Σf)
end

function spec_M1(par, k, obs, T)
    Q = par.Qs[k]
    P, _ = riccati(par.A, par.S, Q, Q, T)
    M = hamiltonian(par.A, par.S, Q)
    B = blkdiag(par.Σ, par.Ω)
    Qn = [sym(ltrigraph(P[t + 1]) * B * ltrigraph(P[t + 1])') for t in 1:(T - 1)]
    L1 = ltrigraph(P[1])
    Tp = eltype(M)
    return (F=fill(M, T - 1), Qn=Qn, m1=vcat(par.μ1, P[1] * par.μ1),
            V1=sym(L1 * blkdiag(par.V1, par.Ω) * L1'),
            H=[obs.C zeros(size(obs.C, 1), size(Q, 1))], d=obs.d, R=obs.R,
            Hf=[-Q eye(Tp, size(Q, 1))], Σf=par.Σf)
end

function spec_M2(par, k, obs, T)
    Q = par.Qs[k]
    _, W = riccati(par.A, par.S, Q, Q, T)
    F = [W[t + 1] * par.A for t in 1:(T - 1)]
    SΩS = par.S * par.Ω * par.S'
    Qn = [sym(par.Σ + W[t + 1] * SΩS * W[t + 1]') for t in 1:(T - 1)]
    return (F=F, Qn=Qn, m1=par.μ1, V1=par.V1, H=obs.C, d=obs.d, R=obs.R,
            Hf=nothing, Σf=nothing)
end

"""M2 with *implicit* noise timing — what `:hold` mode assumes: the plant noise
enters before the controller's `W_{t+1}` (`simulate_lqr(; noise_timing = :implicit)`),
so the innovation is `W_{t+1}(Σ + S Ω S)W_{t+1}ᵀ`. In the plant-row/manifold-row
coordinates that makes the noise the constant `blkdiag(Σ, Ω)`, which is what lets
`:hold` profile it in closed form. Causal data are misspecified for it."""
function spec_M2i(par, k, obs, T)
    Q = par.Qs[k]
    _, W = riccati(par.A, par.S, Q, Q, T)
    F = [W[t + 1] * par.A for t in 1:(T - 1)]
    B = par.Σ + par.S * par.Ω * par.S'
    Qn = [sym(W[t + 1] * B * W[t + 1]') for t in 1:(T - 1)]
    return (F=F, Qn=Qn, m1=par.μ1, V1=par.V1, H=obs.C, d=obs.d, R=obs.R,
            Hf=nothing, Σf=nothing)
end

const SPECS = (M0=spec_M0, M1=spec_M1, M2=spec_M2, M2i=spec_M2i)

# =========================================================================
# Likelihoods
# =========================================================================

"Σ_i log N(E[:, i]; 0, LLᵀ) for a Cholesky factor `ch`."
function gauss_ll(ch, E)
    Z = ch.L \ E
    k, N = size(E, 1), size(E, 2)
    return -(N * (2 * sum(log, diag(ch.U)) + k * log(2π)) + sum(abs2, Z)) / 2
end

"""Exact log p(y | f = 0) summed over the trials in `Y` (NY × T × N). Trials
share the spec, so the covariance recursion runs once and only the means are
per trial."""
function kf_loglik(sp, Y)
    _, T, N = size(Y)
    m = repeat(sp.m1, 1, N)
    V = sp.V1
    ll = zero(eltype(V))
    for t in 1:T
        if t > 1
            m = sp.F[t - 1] * m
            V = sym(sp.F[t - 1] * V * sp.F[t - 1]' + sp.Qn[t - 1])
        end
        ch = cholesky(Symmetric(sym(sp.H * V * sp.H' + sp.R)))
        E = Y[:, t, :] .- (sp.H * m .+ sp.d)
        ll += gauss_ll(ch, E)
        K = transpose(ch \ (sp.H * V))
        m = m + K * E
        V = sym(V - K * sp.H * V)
    end
    sp.Hf === nothing && return ll
    # numerator: the terminal pseudo-observation f = 0 on the filtered z_T
    chf = cholesky(Symmetric(sym(sp.Hf * V * sp.Hf' + sp.Σf)))
    ll += gauss_ll(chf, -(sp.Hf * m))
    # normalizer: N × log p(f = 0) from the prior chain alone
    mp, Vp = sp.m1, sp.V1
    for t in 1:(T - 1)
        mp = sp.F[t] * mp
        Vp = sym(sp.F[t] * Vp * sp.F[t]' + sp.Qn[t])
    end
    chp = cholesky(Symmetric(sym(sp.Hf * Vp * sp.Hf' + sp.Σf)))
    ll -= N * gauss_ll(chp, reshape(-(sp.Hf * mp), :, 1))
    return ll
end

"Joint mean and covariance of vec(z_{1:T}) under a spec (no terminal)."
function dense_z(sp, T)
    D = length(sp.m1)
    Tp = eltype(sp.V1)
    μ = Vector{Vector{Tp}}(undef, T)
    V = Vector{Matrix{Tp}}(undef, T)
    μ[1], V[1] = sp.m1, sp.V1
    for t in 2:T
        μ[t] = sp.F[t - 1] * μ[t - 1]
        V[t] = sp.F[t - 1] * V[t - 1] * sp.F[t - 1]' + sp.Qn[t - 1]
    end
    Σz = zeros(Tp, D * T, D * T)
    blk(t) = ((t - 1) * D + 1):(t * D)
    for t in 1:T
        Φ = Matrix{Tp}(I, D, D)
        for s in t:T
            s > t && (Φ = sp.F[s - 1] * Φ)
            Σz[blk(s), blk(t)] = Φ * V[t]
            Σz[blk(t), blk(s)] = (Φ * V[t])'
        end
    end
    return vcat(μ...), Σz, blk
end

"Brute-force log p(y | f = 0) for one trial (NY × T): the reference for `kf_loglik`."
function dense_loglik(sp, y)
    T = size(y, 2)
    μz, Σz, blk = dense_z(sp, T)
    Hb = kron(Matrix(1.0I, T, T), sp.H)
    μy = Hb * μz .+ repeat(sp.d, T)
    Cyy = Hb * Σz * Hb' + kron(Matrix(1.0I, T, T), sp.R)
    if sp.Hf !== nothing
        μf = sp.Hf * μz[blk(T)]
        Cff = sp.Hf * Σz[blk(T), blk(T)] * sp.Hf' + sp.Σf
        Cyf = Hb * Σz[:, blk(T)] * sp.Hf'
        μy = μy + Cyf * (Cff \ (-μf))
        Cyy = Cyy - Cyf * (Cff \ Cyf')
    end
    ch = cholesky(Symmetric(Cyy))
    return gauss_ll(ch, reshape(vec(y) - μy, :, 1))
end

# =========================================================================
# Generators
# =========================================================================

"The causal agent of `simulate_lqr` (`noise_timing = :causal`), with optional
isotropic costate slack of variance `tr.Ω[1,1]`."
function simulate_causal(rng, tr, k, N, T)
    Q = tr.Qs[k]
    P, W = riccati(tr.A, tr.S, Q, Q, T)
    LΣ = cholesky(tr.Σ).L
    LV = cholesky(tr.V1).L
    LR = cholesky(tr.R).L
    slack = sqrt(tr.Ω[1, 1])
    Y = zeros(NY, T, N)
    for i in 1:N
        x = tr.μ1 + LV * randn(rng, NX)
        ν = [slack .* randn(rng, NX) for _ in 1:T]
        for t in 1:T
            Y[:, t, i] = tr.C * x + tr.d + LR * randn(rng, NY)
            t == T && break
            x = W[t + 1] * (tr.A * x - tr.S * ν[t + 1]) + LΣ * randn(rng, NX)
        end
    end
    return Y
end

"Exact draws from M1 given f = 0, through the dense joint."
function simulate_M1(rng, tr, k, N, T)
    sp = spec_M1(tr, k, tr, T)
    μz, Σz, blk = dense_z(sp, T)
    Czf = Σz[:, blk(T)] * sp.Hf'
    Cff = sp.Hf * Σz[blk(T), blk(T)] * sp.Hf' + sp.Σf
    μc = μz + Czf * (Cff \ (-(sp.Hf * μz[blk(T)])))
    Σc = Symmetric(Σz - Czf * (Cff \ Czf'))
    Lc = cholesky(Σc + 1e-10 * I).L
    LR = cholesky(tr.R).L
    D = 2NX
    Y = zeros(NY, T, N)
    for i in 1:N
        z = μc + Lc * randn(rng, D * T)
        for t in 1:T
            Y[:, t, i] = tr.C * z[blk(t)][1:NX] + tr.d + LR * randn(rng, NY)
        end
    end
    return Y
end

const GENS = (
    G1=(truth=TRUE_G1, sim=simulate_causal, model=:M2),
    G2=(truth=TRUE_G2, sim=simulate_causal, model=:M2),
    G3=(truth=TRUE_G3, sim=simulate_M1, model=:M1),
)

# =========================================================================
# Parameterization
# =========================================================================

ntri(k) = k * (k + 1) ÷ 2
function ltri(v, k)
    L = zeros(eltype(v), k, k)
    idx = 0
    for j in 1:k, i in j:k
        idx += 1
        L[i, j] = i == j ? exp(v[idx]) : v[idx]
    end
    return L
end
pdm(v, k) = (L = ltri(v, k); L * L')
function unpdm(M)
    k = size(M, 1)
    L = cholesky(Symmetric(Matrix(M))).L
    return [i == j ? log(L[i, j]) : L[i, j] for j in 1:k for i in j:k]
end

const S_TRACE = tr(TRUE_G1.S)

function layout(model)
    blocks = Pair{Symbol,Int}[:S => ntri(NX), :Q1 => ntri(NX), :Q2 => ntri(NX)]
    KNOWN_A || pushfirst!(blocks, :A => NX^2)
    if model === :M0
        append!(blocks, [:Σmix => ntri(2NX), :m01 => 2NX, :m02 => 2NX,
                         :V01 => ntri(2NX), :V02 => ntri(2NX), :Σf => ntri(NX)])
    else
        append!(blocks, [:Σ => ntri(NX), :Ω => ntri(NX), :μ1 => NX, :V1 => ntri(NX)])
        model === :M1 && push!(blocks, :Σf => ntri(NX))
    end
    r = Dict{Symbol,UnitRange{Int}}()
    o = 0
    for (k, w) in blocks
        r[k] = (o + 1):(o + w)
        o += w
    end
    return r, o
end

function unpack(model, θ)
    lay, _ = layout(model)
    g(k) = θ[lay[k]]
    A = KNOWN_A ? TRUE_G1.A : reshape(g(:A), NX, NX)
    St = pdm(g(:S), NX)
    S = (S_TRACE / tr(St)) * St
    Qs = [pdm(g(:Q1), NX), pdm(g(:Q2), NX)]
    if model === :M0
        return (; A, S, Qs, Σmix=pdm(g(:Σmix), 2NX), m0=[g(:m01), g(:m02)],
                V0=[pdm(g(:V01), 2NX), pdm(g(:V02), 2NX)], Σf=pdm(g(:Σf), NX))
    end
    base = (; A, S, Qs, Σ=pdm(g(:Σ), NX), Ω=pdm(g(:Ω), NX), μ1=g(:μ1), V1=pdm(g(:V1), NX))
    return model === :M1 ? merge(base, (; Σf=pdm(g(:Σf), NX))) : base
end

"Pack a parameter NamedTuple (as `unpack` returns) into θ."
function pack(model, par)
    lay, np = layout(model)
    θ = zeros(np)
    KNOWN_A || (θ[lay[:A]] = vec(par.A))
    θ[lay[:S]] = unpdm(par.S)
    θ[lay[:Q1]] = unpdm(par.Qs[1])
    θ[lay[:Q2]] = unpdm(par.Qs[2])
    if model === :M0
        θ[lay[:Σmix]] = unpdm(par.Σmix)
        θ[lay[:m01]], θ[lay[:m02]] = par.m0
        θ[lay[:V01]], θ[lay[:V02]] = unpdm(par.V0[1]), unpdm(par.V0[2])
        θ[lay[:Σf]] = unpdm(par.Σf)
    else
        θ[lay[:Σ]], θ[lay[:Ω]] = unpdm(par.Σ), unpdm(par.Ω)
        θ[lay[:μ1]], θ[lay[:V1]] = par.μ1, unpdm(par.V1)
        model === :M1 && (θ[lay[:Σf]] = unpdm(par.Σf))
    end
    return θ
end

"""Starting points. `:truth` puts the shared blocks at the generating values and
M0's mixed noise at the time-average of the causal agent's mixed residual
covariance (plus slack, plus a ridge). `:naive` knows nothing about the plant."""
function init_params(model, how, tr, data, obs)
    floorΩ = 1e-4 * Matrix(1.0I, NX, NX)
    if how === :truth
        A, S, Qs, Σ, V1, μ1 = tr.A, tr.S, tr.Qs, tr.Σ, tr.V1, tr.μ1
        Ω = tr.Ω + floorΩ
    else
        A = KNOWN_A ? TRUE_G1.A : 0.95 * Matrix(1.0I, NX, NX)
        S = (S_TRACE / NX) * Matrix(1.0I, NX, NX)
        Qs = [Matrix(1.0I, NX, NX), Matrix(1.0I, NX, NX)]
        Σ = 0.03 * Matrix(1.0I, NX, NX)
        Ω = 0.1 * Matrix(1.0I, NX, NX)
        Cp = pinv(obs.C)
        x1 = hcat([Cp * (Y[:, 1, :] .- obs.d) for Y in data]...)
        μ1 = vec(mean(x1; dims=2))
        V1 = Matrix(Symmetric(cov(x1'))) + 0.01I
    end
    if model === :M0
        Σmix = zeros(2NX, 2NX)
        m0 = Vector{Vector{Float64}}(undef, 2)
        V0 = Vector{Matrix{Float64}}(undef, 2)
        for k in 1:2
            P, _ = riccati(A, S, Qs[k], Qs[k], TT)
            for t in 1:(TT - 1)
                Km = [Matrix(1.0I, NX, NX) + S * P[t + 1]; -A' * P[t + 1]]
                Σmix += Km * Σ * Km' / (2 * (TT - 1))
            end
            m0[k] = vcat(μ1, P[1] * μ1)
            L1 = ltrigraph(P[1])
            V0[k] = sym(L1 * blkdiag(V1, Ω) * L1') + 1e-3I
        end
        Σmix += blkdiag(zeros(NX, NX), Ω + A' * Ω * A) + 1e-3 * LinearAlgebra.tr(Σmix) * I
        return pack(model, (; A, S, Qs, Σmix, m0, V0, Σf=tr.Σf))
    end
    return pack(model, (; A, S, Qs, Σ, Ω, μ1, V1, Σf=tr.Σf))
end

total_loglik(model, par, data, obs) =
    sum(kf_loglik(SPECS[model](par, k, obs, TT), data[k]) for k in eachindex(data))

# =========================================================================
# Fitting
# =========================================================================

function fit(model, θ0, data, obs; iters=QUICK ? 300 : 2000)
    nbins = sum(size(Y, 2) * size(Y, 3) for Y in data)
    function f(θ)
        v = try
            -total_loglik(model, unpack(model, θ), data, obs) / nbins
        catch e
            e isa InterruptException && rethrow()
            convert(eltype(θ), Inf)
        end
        return isfinite(v) ? v : convert(eltype(θ), Inf)
    end
    cfg = ForwardDiff.GradientConfig(f, θ0)
    g!(G, θ) = ForwardDiff.gradient!(G, f, θ, cfg)
    t0 = time()
    res = Optim.optimize(f, g!, θ0, LBFGS(; linesearch=BackTracking()),
                         Optim.Options(; iterations=iters, g_abstol=1e-6, f_reltol=1e-11))
    return (θ=Optim.minimizer(res), nll=Optim.minimum(res), iters=Optim.iterations(res),
            converged=Optim.converged(res), secs=time() - t0)
end

# =========================================================================
# Metrics (all invariant to the gauges left open)
# =========================================================================

#=
Cost is compared entrywise. C is fixed at the truth, so x is in fixed coordinates
and the only gauge left is the S/Q scale, which `tr(S)` pins: Q_k[1,1] is the
position cost and Q_k[2,2] the velocity cost, both directly comparable.
(Sorted eigenvalues are not: when a poorly identified cost runs off, it changes
rank, and the "large eigenvalue" error then reports the wrong direction.)
=#

"|log ratio| of the (position, velocity) diagonal cost, worse of k = 1, 2."
function cost_err(par, tr)
    e = [maximum(abs(log(max(par.Qs[k][i, i], 1e-300) / tr.Qs[k][i, i])) for k in 1:2) for i in 1:NX]
    return e[1], e[2]
end

"|log ratio| of the cost contrast Q₂[i,i]/Q₁[i,i], (position, velocity); true 5, 2."
function contrast_err(par, tr)
    r(p, i) = p.Qs[2][i, i] / p.Qs[1][i, i]
    e = [abs(log(max(r(par, i), 1e-300) / r(tr, i))) for i in 1:NX]
    return e[1], e[2]
end

"mean_t,k ‖Φ_t − Φ_t*‖_F / ‖Φ_t*‖_F (C is fixed, so the gauge is pinned)."
function gain_err(par, tr)
    e = 0.0
    for k in 1:2
        _, W = riccati(par.A, par.S, par.Qs[k], par.Qs[k], TT)
        _, Wt = riccati(tr.A, tr.S, tr.Qs[k], tr.Qs[k], TT)
        for t in 1:(TT - 1)
            Φ, Φt = W[t + 1] * par.A, Wt[t + 1] * tr.A
            e += norm(Φ - Φt) / norm(Φt) / (2 * (TT - 1))
        end
    end
    return e
end

"""Slack share of the closed-loop innovation, `tr(W S Ω S Wᵀ) / tr(Σ + W S Ω S Wᵀ)`
averaged over t and k — what the data can see of Ω. (Ω in a direction the
controller drives to zero is invisible: eig(Σ⁻¹ S Ω S) can be in the thousands
while this is not.) For M1 it is a nominal number: there ν is a persistent
costate error, not white slack."""
function slack_share(par)
    haskey(par, :Ω) || return NaN
    s = 0.0
    for k in 1:2
        _, W = riccati(par.A, par.S, par.Qs[k], par.Qs[k], TT)
        for t in 2:TT
            V = W[t] * par.S * par.Ω * par.S' * W[t]'
            s += tr(V) / (tr(par.Σ) + tr(V)) / (2 * (TT - 1))
        end
    end
    return s
end
sigma_err(par, tr) = haskey(par, :Σ) ? norm(par.Σ - tr.Σ) / norm(tr.Σ) : NaN

# =========================================================================
# Selftest
# =========================================================================

bigify(nt) = map(v -> v isa AbstractArray ?
                      (eltype(v) <: AbstractArray ? [big.(x) for x in v] : big.(v)) : v, nt)

function selftest()
    setprecision(BigFloat, 256)
    section("selftest: Kalman vs dense brute force; nesting; generators")
    rng = MersenneTwister(1)
    obs = TRUE_G1
    ok = true
    for model in (:M0, :M1, :M2)
        _, np = layout(model)
        θ = init_params(model, :truth, TRUE_G2, nothing, obs) .+ 0.05 .* randn(rng, np)
        par = unpack(model, θ)
        y = simulate_causal(rng, TRUE_G2, 2, 1, TT)
        for k in 1:2
            # the reference runs in 256-bit: the Hamiltonian chain's prior covariance
            # grows like ρ(M)^{2T} and the dense conditioning loses ~1e-4 nats of it
            # in Float64 (the filter does not)
            sp = SPECS[model](par, k, obs, TT)
            spb = SPECS[model](bigify(par), k, bigify(obs), TT)
            a, b = kf_loglik(sp, y), Float64(dense_loglik(spb, big.(y[:, :, 1])))
            pass = abs(a - b) < 1e-6 * max(1, abs(b))
            ok &= pass
            @printf("  %s cond %d   Kalman %.8f   dense %.8f   %s\n", model, k, a, b,
                    pass ? "ok" : "MISMATCH")
        end
    end
    # M1 at Ω = 0 is M2 at Ω = 0 (the terminal factor becomes independent of y)
    p1 = merge(TRUE_G1, (; Ω=zeros(NX, NX)))
    y = simulate_causal(rng, TRUE_G1, 1, 20, TT)
    a = kf_loglik(spec_M1(p1, 1, p1, TT), y)
    b = kf_loglik(spec_M2(p1, 1, p1, TT), y)
    pass = abs(a - b) < 1e-6 * abs(b)
    ok &= pass
    @printf("  M1(Ω=0) %.8f   M2(Ω=0) %.8f   %s\n", a, b, pass ? "ok" : "MISMATCH")
    # generators match their models' one-step moments: empirical y-covariance at
    # the last bin vs the model's marginal
    for (name, g) in pairs(GENS)
        Y = g.sim(MersenneTwister(2), g.truth, 2, 4000, TT)
        sp = SPECS[g.model](g.truth, 2, g.truth, TT)
        μz, Σz, blk = dense_z(sp, TT)
        if sp.Hf !== nothing
            Czf = Σz[:, blk(TT)] * sp.Hf'
            Cff = sp.Hf * Σz[blk(TT), blk(TT)] * sp.Hf' + sp.Σf
            μz = μz + Czf * (Cff \ (-(sp.Hf * μz[blk(TT)])))
            Σz = Σz - Czf * (Cff \ Czf')
        end
        for t in (2, TT ÷ 2, TT)
            Cm = sp.H * Σz[blk(t), blk(t)] * sp.H' + sp.R
            Ce = cov(Y[:, t, :]')
            rel = norm(Ce - Cm) / norm(Cm)
            mrel = norm(vec(mean(Y[:, t, :]; dims=2)) - sp.H * μz[blk(t)]) / sqrt(tr(Cm))
            pass = rel < 0.1 && mrel < 0.1
            ok &= pass
            @printf("  %s t=%2d   cov rel err %.3f   mean err/sd %.3f   %s\n", name, t, rel,
                    mrel, pass ? "ok" : "CHECK")
        end
    end
    # ForwardDiff gradient vs central differences on the objective
    for model in (:M0, :M1, :M2)
        data = [simulate_causal(rng, TRUE_G2, k, 10, TT) for k in 1:2]
        θ = init_params(model, :truth, TRUE_G2, data, obs)
        f(θ) = -total_loglik(model, unpack(model, θ), data, obs)
        g = ForwardDiff.gradient(f, θ)
        h = 1e-6
        gd = [(f(θ + h * e) - f(θ - h * e)) / 2h for e in eachcol(Matrix(1.0I, length(θ), length(θ)))]
        rel = norm(g - gd) / norm(gd)
        pass = rel < 1e-5
        ok &= pass
        @printf("  %s gradient rel err %.2e   %s\n", model, rel, pass ? "ok" : "MISMATCH")
    end
    println(ok ? "\nselftest passed" : "\nselftest FAILED")
    return ok
end

# =========================================================================
# Timing
# =========================================================================

function timing()
    section("timing: one objective and one gradient over 2 × $NTRAIN trials × $TT bins")
    rng = MersenneTwister(3)
    obs = TRUE_G1
    data = [simulate_causal(rng, TRUE_G2, k, NTRAIN, TT) for k in 1:2]
    @printf("%-4s %6s %14s %14s\n", "", "params", "value (ms)", "gradient (ms)")
    for model in (:M0, :M1, :M2)
        θ = init_params(model, :truth, TRUE_G2, data, obs)
        f(θ) = -total_loglik(model, unpack(model, θ), data, obs)
        cfg = ForwardDiff.GradientConfig(f, θ)
        G = similar(θ)
        f(θ); ForwardDiff.gradient!(G, f, θ, cfg)
        tv = minimum(@elapsed(f(θ)) for _ in 1:20)
        tg = minimum(@elapsed(ForwardDiff.gradient!(G, f, θ, cfg)) for _ in 1:5)
        @printf("%-4s %6d %14.3f %14.3f\n", model, length(θ), 1e3tv, 1e3tg)
    end
end

# =========================================================================
# Identifiability: expected Fisher information under the true model
# =========================================================================

"""Delta-method standard errors of the cost summaries for M2 fitted to G1 at the
truth, from the exact Gaussian Fisher information of vec(y_{1:T})
(`Jμᵀ C⁻¹ Jμ + ½ tr(C⁻¹ ∂C C⁻¹ ∂C)`), for 100 trials per condition. Ω is held at
its true value 0 (a boundary). This is the floor any noise model inherits: a
cost summary with a standard error near 1 cannot discriminate models."""
function fisher()
    section("fisher: what this design can identify (M2 at the G1 truth, $(KNOWN_A ? "A known" : "A free"))")
    tr0 = TRUE_G1
    lay, np = layout(:M2)
    θ0 = pack(:M2, merge(tr0, (; Ω=1e-3 * Matrix(1.0I, NX, NX))))
    keep = setdiff(1:np, lay[:Ω])
    full(φ) = (θ = convert(Vector{eltype(φ)}, θ0); θ[keep] = φ; θ)
    function ymoments(φ, k)
        par = merge(unpack(:M2, full(φ)), (; Ω=zeros(NX, NX)))
        sp = spec_M2(par, k, tr0, TT)
        μz, Σz, _ = dense_z(sp, TT)
        Hb = kron(Matrix(1.0I, TT, TT), sp.H)
        return Hb * μz, Hb * Σz * Hb' + kron(Matrix(1.0I, TT, TT), sp.R)
    end
    φ0 = θ0[keep]
    F = zeros(length(φ0), length(φ0))
    for k in 1:2
        _, C = ymoments(φ0, k)
        Jμ = ForwardDiff.jacobian(φ -> ymoments(φ, k)[1], φ0)
        JC = ForwardDiff.jacobian(φ -> vec(ymoments(φ, k)[2]), φ0)
        Ci = inv(Symmetric(C))
        F .+= Jμ' * Ci * Jμ
        D = [Ci * reshape(JC[:, i], size(C)) for i in eachindex(φ0)]
        for i in eachindex(φ0), j in eachindex(φ0)
            F[i, j] += tr(D[i] * D[j]) / 2
        end
    end
    F .*= 100
    # tr(S) is fixed, so the raw scale of S's factor is an exact null direction;
    # every summary below is invariant along it, which makes the pseudo-inverse exact.
    Σφ = pinv(Symmetric(F); rtol=1e-10)
    g(φ) = (p = unpack(:M2, full(φ));
            vcat([log(p.Qs[k][i, i]) for k in 1:2 for i in 1:NX],
                 [log(p.Qs[2][i, i] / p.Qs[1][i, i]) for i in 1:NX]))
    J = ForwardDiff.jacobian(g, φ0)
    se = sqrt.(diag(J * Σφ * J'))
    @printf("  SE of log Q₁[i,i]          position %.3f   velocity %.3f\n", se[1], se[2])
    @printf("  SE of log Q₂[i,i]          position %.3f   velocity %.3f\n", se[3], se[4])
    @printf("  SE of log Q₂[i,i]/Q₁[i,i]  position %.3f   velocity %.3f\n", se[5], se[6])
    ev = sort(eigvals(Symmetric(F)))
    @printf("  Fisher spectrum: null %.1e, then %.3g, %.3g, … , %.3g\n", ev[1], ev[2], ev[3], ev[end])
end

# =========================================================================
# Fit comparison
# =========================================================================
#
# Work is split into tasks (gen, seed, model), each fitting from both starts and
# keeping the better training objective. `--shard=i/n` runs every n-th task in
# its own process (Julia's GC contends badly across threads on this allocation-
# heavy dual-number code), `--out=file` serializes the rows with the fitted θ,
# and `--merge=f1,f2,…` reads shards back and prints the tables.

using Serialization

arg(name) = (a = filter(s -> startswith(s, "--$name="), ARGS); isempty(a) ? nothing : split(a[1], "=")[2])

function make_data(gen, seed)
    g = GENS[gen]
    rng = MersenneTwister(1000 * seed + Int(gen === :G2) + 2 * Int(gen === :G3))
    train = [g.sim(rng, g.truth, k, NTRAIN, TT) for k in 1:2]
    test = [g.sim(rng, g.truth, k, NTEST, TT) for k in 1:2]
    return train, test
end

function run_task(gen, seed, model)
    g = GENS[gen]
    tr = g.truth
    train, test = make_data(gen, seed)
    fits = [(how, fit(model, init_params(model, how, tr, train, tr), train, tr))
            for how in (:truth, :naive)]
    best = argmin(r -> r[2].nll, fits)[2]
    return (gen=gen, seed=seed, model=model, known_A=KNOWN_A, θ=best.θ,
            nll=[r[2].nll for r in fits], iters=[r[2].iters for r in fits],
            conv=[r[2].converged for r in fits], secs=[r[2].secs for r in fits])
end

"Everything reported is recomputed from the stored θ, so metrics can change
without refitting."
function score(row)
    g = GENS[row.gen]
    tr = g.truth
    train, test = make_data(row.gen, row.seed)
    par = unpack(row.model, row.θ)
    ntest, ntrain = 2 * NTEST * TT, 2 * NTRAIN * TT
    dtest = (total_loglik(row.model, par, test, tr) - total_loglik(g.model, tr, test, tr)) / ntest
    dtrain = (total_loglik(row.model, par, train, tr) - total_loglik(g.model, tr, train, tr)) / ntrain
    clo, chi = cost_err(par, tr)
    qlo, qhi = contrast_err(par, tr)
    return merge(row, (; np=length(row.θ), dtest, dtrain, cost_lo=clo, cost_hi=chi,
                       contr_lo=qlo, contr_hi=qhi, gain=gain_err(par, tr),
                       sigma=sigma_err(par, tr), slack=slack_share(par),
                       slack_true=slack_share(merge(tr, (; Ω=GENS[row.gen].truth.Ω))),
                       spread=abs(row.nll[1] - row.nll[2])))
end

const MODELS = let m = arg("models")
    m === nothing ? (:M0, :M1, :M2) : Tuple(Symbol.(split(m, ",")))
end
const TASKS = [(gen, seed, model) for model in MODELS
               for gen in (:G1, :G2, :G3) for seed in 1:NSEEDS]

function fitshard()
    sh = arg("shard")
    i, n = sh === nothing ? (1, 1) : parse.(Int, split(sh, "/"))
    out = arg("out")
    mine = TASKS[i:n:end]
    section("fit ($(KNOWN_A ? "A known" : "A free")), shard $i/$n: $(length(mine)) tasks")
    rows = Any[]
    for t in mine
        r = run_task(t...)
        push!(rows, r)
        @printf("  done %s s%d %s  nll %s  iters %s  conv %s  %.0fs\n", r.gen, r.seed, r.model,
                string(round.(r.nll; digits=5)), string(r.iters), string(r.conv), sum(r.secs))
        flush(stdout)
        out === nothing || serialize(out, rows)
    end
    return rows
end

function report(rows)
    rows = [score(r) for r in rows]
    section("results ($(rows[1].known_A ? "A known" : "A free")), $(maximum(r.seed for r in rows)) seeds, $(NTRAIN) train / $(NTEST) test trials per condition, T = $TT")
    println("""
    dtest    held-out log-lik per bin minus the generating model's (nats; 0 = as good as truth)
    dtrain   same on the training data
    Qpos/vel |log ratio| of the position / velocity cost Q_k[i,i] (worse k)
    ctr p/v  |log ratio| of the contrast Q₂[i,i]/Q₁[i,i]         (true 5, 2)
    gain     mean relative error of the closed-loop maps Φ_t
    Σerr     relative error of the plant noise Σ
    slack    slack share of the closed-loop innovation (true in brackets; nominal for M1)
    spread   |nll(truth start) − nll(naive start)| per bin""")
    @printf("\n%-3s %-3s %3s %-18s %8s %6s %6s %6s %6s %6s %6s %-14s %8s %5s %s\n", "gen", "mod",
            "np", "dtest (mean±sd)", "dtrain", "Qpos", "Qvel", "ctr p", "ctr v", "gain",
            "Σerr", "slack [true]", "spread", "secs", "conv")
    for gen in (:G1, :G2, :G3), model in (:M0, :M1, :M2, :M2i)
        rs = filter(r -> r.gen === gen && r.model === model, rows)
        isempty(rs) && continue
        med(f) = median(f.(rs))
        @printf("%-3s %-3s %3d %+8.4f±%-9.4f %+8.4f %6.3f %6.2f %6.3f %6.2f %6.3f %6.3f %6.3f [%.3f] %8.1e %5.0f %d/%d\n",
                gen, model, rs[1].np, mean(r.dtest for r in rs), std(r.dtest for r in rs),
                med(r -> r.dtrain), med(r -> r.cost_lo), med(r -> r.cost_hi),
                med(r -> r.contr_lo), med(r -> r.contr_hi), med(r -> r.gain),
                med(r -> r.sigma), med(r -> r.slack), rs[1].slack_true, med(r -> r.spread),
                med(r -> sum(r.secs)), count(r -> any(r.conv), rs), length(rs))
    end
    println("\npaired held-out differences (same test set), mean ± sd over seeds, nats/bin:")
    for gen in (:G1, :G2, :G3)
        d(a, b) = [r1.dtest - r2.dtest for r1 in rows, r2 in rows
                   if r1.gen === gen && r2.gen === gen && r1.seed == r2.seed &&
                      r1.model === a && r2.model === b]
        parts = String[]
        for (a, b) in ((:M1, :M0), (:M2, :M0), (:M1, :M2), (:M2i, :M2))
            x = d(a, b)
            isempty(x) || push!(parts, @sprintf("%s−%s %+.4f±%.4f", a, b, mean(x), std(x)))
        end
        println("  ", gen, "  ", join(parts, "   "))
    end
    println("\nper seed:")
    for r in sort(rows; by=r -> (r.gen, r.model, r.seed))
        @printf("  %s s%d %s  dtest %+.4f  Q pos/vel %.3f/%.2f  ctr %.3f/%.2f  gain %.3f  slack %.3f  iters %s\n",
                r.gen, r.seed, r.model, r.dtest, r.cost_lo, r.cost_hi, r.contr_lo, r.contr_hi,
                r.gain, r.slack, string(r.iters))
    end
end


# =========================================================================
# EM for M2, and where the time goes
# =========================================================================
#
# The fits above maximize log p(y) directly, so every objective evaluation reruns
# the Riccati sweep *and* the Kalman filter over all N trials, and ForwardDiff
# differentiates through both. That is the simplest exact thing, not the cheap
# one. EM separates them: the smoother runs once per iteration and reduces the
# data to per-transition sufficient statistics; the M-step then needs only the
# Riccati sweep, at a cost independent of N.

"""RTS smoother for a spec without terminal factor, all N trials sharing it (so
the covariance recursions run once). Returns log p(y) and the sufficient
statistics summed over trials, kept *per transition* because the M2 transition
varies with t: `S00[t] = Σᵢ E[x_t x_tᵀ]`, `S10[t] = Σᵢ E[x_{t+1} x_tᵀ]`,
`S11[t] = Σᵢ E[x_{t+1} x_{t+1}ᵀ]`, and the initial-state sums `s1`, `S1`."""
function rts_stats(sp, Y)
    _, T, N = size(Y)
    mf = Vector{Matrix{Float64}}(undef, T)
    Vf = Vector{Matrix{Float64}}(undef, T)
    Vp = Vector{Matrix{Float64}}(undef, T)
    m = repeat(sp.m1, 1, N)
    V = sp.V1
    ll = 0.0
    for t in 1:T
        if t > 1
            m = sp.F[t - 1] * m
            V = sym(sp.F[t - 1] * V * sp.F[t - 1]' + sp.Qn[t - 1])
            Vp[t] = V
        end
        ch = cholesky(Symmetric(sym(sp.H * V * sp.H' + sp.R)))
        E = Y[:, t, :] .- (sp.H * m .+ sp.d)
        ll += gauss_ll(ch, E)
        K = transpose(ch \ (sp.H * V))
        m = m + K * E
        V = sym(V - K * sp.H * V)
        mf[t], Vf[t] = m, V
    end
    ms, Vs = mf[T], Vf[T]
    S00 = Vector{Matrix{Float64}}(undef, T - 1)
    S10 = similar(S00)
    S11 = similar(S00)
    for t in (T - 1):-1:1
        J = transpose(cholesky(Symmetric(Vp[t + 1])) \ (sp.F[t] * Vf[t]))
        msn = mf[t] + J * (ms - sp.F[t] * mf[t])
        Vsn = sym(Vf[t] + J * (Vs - Vp[t + 1]) * J')
        S11[t] = N * Vs + ms * ms'
        S10[t] = N * Vs * J' + ms * msn'
        S00[t] = N * Vsn + msn * msn'
        ms, Vs = msn, Vsn
    end
    return (; S00, S10, S11, s1=vec(sum(ms; dims=2)), S1=N * Vs + ms * ms', N, ll)
end

"""Expected complete-data log-likelihood of the M2 dynamics and initial state
(the emission terms are constant: C, d, R are fixed). Reads only the statistics,
so one evaluation costs one Riccati sweep per condition — no trial loop."""
function mstep_objective(θ, stats, model)
    par = unpack(:M2, θ)
    q = zero(eltype(θ))
    for k in eachindex(stats)
        st = stats[k]
        sp = SPECS[model](par, k, TRUE_G1, TT)
        for t in 1:(TT - 1)
            Φ = sp.F[t]
            ch = cholesky(Symmetric(sp.Qn[t]))
            M = st.S11[t] - Φ * st.S10[t]' - st.S10[t] * Φ' + Φ * st.S00[t] * Φ'
            q -= (st.N * 2 * sum(log, diag(ch.U)) + tr(ch \ M)) / 2
        end
        chV = cholesky(Symmetric(par.V1))
        M1 = st.S1 - par.μ1 * st.s1' - st.s1 * par.μ1' + st.N * par.μ1 * par.μ1'
        q -= (st.N * 2 * sum(log, diag(chV.U)) + tr(chV \ M1)) / 2
    end
    return q
end

"""Generalized EM: smoother → statistics → a warm-started L-BFGS M-step on the
statistics alone. Each M-step only has to improve Q(θ | θ′), so log p(y) cannot
decrease; the trace checks that."""
function em_fit(model, θ0, data; iters=500, inner=25, tol=1e-10)
    θ = copy(θ0)
    trace = Float64[]
    tE = tM = 0.0
    for it in 1:iters
        tE += @elapsed stats = [rts_stats(SPECS[model](unpack(:M2, θ), k, TRUE_G1, TT), data[k])
                                for k in eachindex(data)]
        push!(trace, sum(s.ll for s in stats))
        it > 1 && abs(trace[end] - trace[end - 1]) < tol * abs(trace[end]) && break
        f(θ) = (v = try
                    -mstep_objective(θ, stats, model)
                catch e
                    e isa InterruptException && rethrow()
                    Inf
                end;
                isfinite(v) ? v : convert(eltype(θ), Inf))
        cfg = ForwardDiff.GradientConfig(f, θ)
        g!(G, θ) = ForwardDiff.gradient!(G, f, θ, cfg)
        tM += @elapsed res = Optim.optimize(f, g!, θ, LBFGS(; linesearch=BackTracking()),
                                            Optim.Options(; iterations=inner))
        Optim.minimum(res) <= f(θ) && (θ = Optim.minimizer(res))
    end
    return (; θ, trace, tE, tM)
end

"A random n-dimensional stand-in for timing at larger plant dimension."
function scaled_problem(n, T; seed=1)
    rng = MersenneTwister(seed)
    A = 0.95I + 0.05 .* randn(rng, n, n) ./ sqrt(n)
    B = randn(rng, n, n) ./ sqrt(n)
    S = 0.05 .* (B * B') + 1e-3I
    Q = Matrix(1.0I, n, n)
    p = 2n + 2
    C = randn(rng, p, n) ./ sqrt(n)
    par = (; A, S, Qs=[Q], Σ=0.01 * Matrix(1.0I, n, n), Ω=0.01 * Matrix(1.0I, n, n),
           μ1=ones(n), V1=0.1 * Matrix(1.0I, n, n), Σmix=0.01 * Matrix(1.0I, 2n, 2n),
           m0=[zeros(2n)], V0=[Matrix(1.0I, 2n, 2n)], Σf=1e-3 * Matrix(1.0I, n, n))
    obs = (; C, d=zeros(p), R=0.05 * Matrix(1.0I, p, p))
    return par, obs, p
end

besttime(f; reps=5) = (f(); minimum(@elapsed(f()) for _ in 1:reps))
mb(f) = (f(); (@allocated f()) / 2^20)

function profile()
    section("profile: Riccati vs Kalman vs gradients (one thread, Float64 unless noted)")
    println("""
    Two Riccati recursions are in play and they are different objects:
      control  P_t = Q + Aᵀ P_{t+1}(I + S P_{t+1})⁻¹ A, backward, from the parameters
               alone — gives Φ_t and the noise; shared by every trial of a condition
      filter   predicted/filtered covariances of the Kalman filter, forward, also
               parameter-only (C, R, Φ_t, Q_t) — shared by every trial of a length
    Only the filter's *means* touch the data: O(N T (d p + p²)) per pass.""")

    println("\n(1) control Riccati sweep, one condition")
    @printf("  %4s %5s %12s %10s\n", "n", "T", "time (μs)", "alloc (KB)")
    for n in (2, 4, 8, 16), T in (30, 100)
        par, _, _ = scaled_problem(n, T)
        f() = riccati(par.A, par.S, par.Qs[1], par.Qs[1], T)
        @printf("  %4d %5d %12.1f %10.1f\n", n, T, 1e6 * besttime(f; reps=500), 1024 * mb(f))
    end

    println("\n(2) exact log p(y), one condition, T = 30: closed loop on x (M2, d = n) vs")
    println("    Hamiltonian chain on [x; λ] with terminal normalizer (M0/M1, d = 2n)")
    @printf("  %4s %6s %14s %10s %14s %10s\n", "n", "N", "M2 (ms)", "MB", "M0 (ms)", "MB")
    for n in (2, 8), N in (10, 100, 1000)
        par, obs, p = scaled_problem(n, 30)
        Y = randn(MersenneTwister(2), p, 30, N)
        f2() = kf_loglik(spec_M2(par, 1, obs, 30), Y)
        f0() = kf_loglik(spec_M0(par, 1, obs, 30), Y)
        @printf("  %4d %6d %14.3f %10.2f %14.3f %10.2f\n", n, N, 1e3 * besttime(f2),
                mb(f2), 1e3 * besttime(f0), mb(f0))
    end

    println("\n(3) direct fitting: ForwardDiff gradient of log p(y) through Riccati + filter,")
    println("    M2, n = 2, both conditions (24 parameters) — scales with N")
    @printf("  %6s %14s %10s\n", "N", "gradient (ms)", "MB")
    for N in (10, 100, 1000)
        data = [simulate_causal(MersenneTwister(3), TRUE_G1, k, N, TT) for k in 1:2]
        θ = init_params(:M2, :truth, TRUE_G1, data, TRUE_G1)
        f(θ) = -total_loglik(:M2, unpack(:M2, θ), data, TRUE_G1)
        cfg = ForwardDiff.GradientConfig(f, θ)
        G = similar(θ)
        g() = ForwardDiff.gradient!(G, f, θ, cfg)
        @printf("  %6d %14.2f %10.1f\n", N, 1e3 * besttime(g; reps=3), mb(g))
    end

    println("\n(4) EM pieces, M2, n = 2, both conditions")
    @printf("  %6s %16s %10s %18s %18s\n", "N", "E-step (ms)", "MB", "M-step value (ms)", "M-step grad (ms)")
    for N in (10, 100, 1000)
        data = [simulate_causal(MersenneTwister(3), TRUE_G1, k, N, TT) for k in 1:2]
        θ = init_params(:M2, :truth, TRUE_G1, data, TRUE_G1)
        par = unpack(:M2, θ)
        e() = [rts_stats(spec_M2(par, k, TRUE_G1, TT), data[k]) for k in 1:2]
        stats = e()
        fq(θ) = -mstep_objective(θ, stats, :M2)
        cfg = ForwardDiff.GradientConfig(fq, θ)
        G = similar(θ)
        @printf("  %6d %16.3f %10.2f %18.3f %18.3f\n", N, 1e3 * besttime(e), mb(e),
                1e3 * besttime(() -> fq(θ); reps=20),
                1e3 * besttime(() -> ForwardDiff.gradient!(G, fq, θ, cfg); reps=5))
    end
    println("  statistics kept: 3 n×n matrices per transition per condition + initial sums")
    @printf("  = %.1f KB at n = 2, T = 30, 2 conditions — independent of N\n",
            (3 * (TT - 1) * NX^2 + NX + NX^2) * 2 * 8 / 1024)
end

function emcheck()
    section("em: generalized EM for M2 vs the direct fit, G1 seed 1, $(NTRAIN) trials/condition")
    train, _ = make_data(:G1, 1)
    θ0 = init_params(:M2, :naive, TRUE_G1, train, TRUE_G1)
    nbins = 2 * NTRAIN * TT
    em = em_fit(:M2, θ0, train)
    drops = count(<(-1e-8), diff(em.trace))
    t_direct = @elapsed dr = fit(:M2, θ0, train, TRUE_G1)
    @printf("  EM:     %4d iterations, log p(y)/bin %.6f, E-step %.1fs, M-step %.1fs, decreases: %d\n",
            length(em.trace), em.trace[end] / nbins, em.tE, em.tM, drops)
    @printf("  direct: %4d iterations, log p(y)/bin %.6f, %.1fs\n", dr.iters, -dr.nll, t_direct)
    pe, pd = unpack(:M2, em.θ), unpack(:M2, dr.θ)
    @printf("  contrast (position) |log err|: EM %.3f  direct %.3f;  gain err: EM %.3f  direct %.3f\n",
            contrast_err(pe, TRUE_G1)[1], contrast_err(pd, TRUE_G1)[1],
            gain_err(pe, TRUE_G1), gain_err(pd, TRUE_G1))
    first = findfirst(>=(em.trace[end] - 1e-4 * nbins), em.trace)
    @printf("  EM reaches within 1e-4 nats/bin of its end after %d iterations\n", first)
end

# =========================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    @printf("threads: %d   QUICK: %s   A known: %s\n", Threads.nthreads(), QUICK, KNOWN_A)
    if SELFTEST
        exit(selftest() ? 0 : 1)
    end
    mg = arg("merge")
    if mg !== nothing
        report(vcat([deserialize(String(f)) for f in split(mg, ",")]...))
        exit(0)
    end
    want("selftest") && (selftest() || exit(1))
    want("timing") && timing()
    want("fisher") && fisher()
    want("profile") && profile()
    want("em") && emcheck()
    want("fit") && report(fitshard())
end
