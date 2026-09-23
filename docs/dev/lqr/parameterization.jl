#=============================================================================
Which parameterization? — the measurements behind `parameterization.md`.

This file is deliberately **self-contained**: it imports nothing from
`StateSpaceDynamics`, because its job is to compare the mixed-coordinate
(Hamiltonian) inverse-LQR model in `src/lds/` against alternatives, and a
comparison that reused the incumbent's own kernels would inherit the incumbent's
choices. Everything here is a few dozen lines of LinearAlgebra.

    julia --project=docs -t auto docs/dev/lqr/parameterization.jl
    julia --project=docs -t auto docs/dev/lqr/parameterization.jl --only=gauge,terminal
    julia --project=docs -t auto docs/dev/lqr/parameterization.jl --only=fit

Sections (`--only=` selects a comma-separated subset):

  geometry   reciprocal eigenvalue pairs; the symplectic defect; M-invariance of
             graph(P); the cost-to-go contraction identity
  gauge      the exact (S, Q) scale invariance, and the same ray with R pinned
  profile    the exact marginal log-likelihood along the cost-scale ray
  terminal   the Riccati Jacobian Ψ, checked against finite differences, and the
             backward decay of terminal-cost information
  reference  where a reference perturbation lands, in time and in observed rows
  shaping    the exact equivalence class of costs: potentials on range(B)^⊥
  subspace   the control subspace from the closed loop, and what optimality adds;
             the inverse least squares, its invariants and its conditioning
  sweeps     the feedforward against a direct QP solve, and the reverse-mode
             gradient of both sweeps against central differences
  switching  γ at the generating parameters, delay-then-reach, matched to slds.jl
  fit        end-to-end recovery, swept over the initial cost scale (slow)

The mixed-coordinate numbers quoted for contrast are `README.md`'s and
`slds.jl`'s own measurements, not re-runs here.
=============================================================================#

using LinearAlgebra, Random, Statistics, Printf
using Optim, LineSearches

const ONLY = let
    a = filter(s -> startswith(s, "--only="), ARGS)
    isempty(a) ? nothing : Set(split(split(a[1], "=")[2], ","))
end
# `--quick` shrinks the only expensive section (`fit`) to a smoke test: it checks
# that the code path runs and the table is well formed, not that anything is
# recovered. Every number in `parameterization.md` comes from a full run.
const QUICK = "--quick" in ARGS
want(s) = ONLY === nothing || s in ONLY
section(t) = (println(); println("="^78); println(t); println("="^78))

# =========================================================================
# The control problem: a damped 2-D point mass reaching to a ring of targets.
# =========================================================================

"x = (p1, p2, v1, v2), u = force. A is invertible, as `LQRStateModel` requires."
function plant(; dt = 0.05, damp = 0.5, mass = 1.0)
    A = [I(2) dt*I(2); zeros(2, 2) (1 - dt*damp)*I(2)]
    B = [zeros(2, 2); (dt/mass)*I(2)]
    return Matrix{Float64}(A), Matrix{Float64}(B)
end

"""
Running and terminal costs, both PSD with genuine off-diagonal structure so that
an entrywise correlation is a meaningful score. The overall size is set so the
agent reaches the target in `T` steps against `R = I`; well below it the agent
undershoots and the task is not a reach.
"""
function true_costs()
    Lrun  = 10 * [20.0 0 0 0; 3.0 18.0 0 0; 2.0 -1.0 6.0 0; -1.5 2.0 1.0 5.0]
    Lterm = 10 * [120.0 0 0 0; 15.0 110.0 0 0; 10.0 -8.0 40.0 0; -6.0 12.0 5.0 35.0]
    sym(M) = Matrix((M + M') / 2)
    return [sym(Lrun * Lrun'), sym(Lterm * Lterm')]
end

struct Setup
    A::Matrix{Float64}; B::Matrix{Float64}; R::Matrix{Float64}; C::Matrix{Float64}
    sched::Vector{Int}; refs::Vector{Vector{Float64}}
    T::Int; n::Int; m::Int; p::Int
    P0::Matrix{Float64}; mu0::Vector{Float64}
end

"`R = I` is the gauge: it is what makes the cost scale a parameter and not an orbit."
function setup(; T = 30, ntarget = 8, radius = 0.12, observe = :position)
    A, B = plant(); n, m = size(A, 1), size(B, 2)
    C = Matrix{Float64}(observe === :position ? [I(2) zeros(2, 2)] : I(n))
    refs = [[radius*cos(2pi*(j-1)/ntarget), radius*sin(2pi*(j-1)/ntarget), 0.0, 0.0]
            for j in 1:ntarget]
    return Setup(A, B, Matrix{Float64}(I, m, m), C, vcat(fill(1, T-1), 2), refs,
                 T, n, m, size(C, 1), Matrix(1e-6*I, n, n), zeros(n))
end

"""
Backward Riccati sweep for

  min  Σ_{t<T} ½[(x_t-r)'Q_{k(t)}(x_t-r) + u'Ru] + ½(x_T-r)'Q_{k(T)}(x_T-r).

Returns `(K, P, Fs)`: the feedback gains, the cost-to-go matrices, and the
factorizations of `R + B'P_{t+1}B` for the affine sweep to reuse.

**None of this depends on the reference.** The feedback half of an LQR tracking
policy is target-independent; only the feedforward is not. So this sweep runs
once per parameter evaluation and [`affine_sweep`](@ref) runs once per target,
which is an `8×` saving at eight targets and is also the honest way to write it.
"""
function riccati_gain(A::Matrix{Float64}, B::Matrix{Float64}, R::Matrix{Float64},
                      Qs::Vector{Matrix{Float64}}, sched::Vector{Int})
    T = length(sched)
    P = Vector{Matrix{Float64}}(undef, T)
    K = Vector{Matrix{Float64}}(undef, T-1)
    Fs = Vector{Cholesky{Float64,Matrix{Float64}}}(undef, T-1)
    P[T] = Qs[sched[T]]
    @inbounds for t in (T-1):-1:1
        Q = Qs[sched[t]]
        Fs[t] = cholesky(Symmetric(R + B' * P[t+1] * B))
        K[t]  = Fs[t] \ (B' * P[t+1] * A)
        Acl   = A - B * K[t]
        Pt    = Q + K[t]' * R * K[t] + Acl' * P[t+1] * Acl
        P[t]  = (Pt + Pt') / 2
    end
    return K, P, Fs
end

"The feedforward `kff` for one reference, given the sweep above. `O(T n²)`."
function affine_sweep(A::Matrix{Float64}, B::Matrix{Float64}, R::Matrix{Float64},
                      Qs::Vector{Matrix{Float64}}, sched::Vector{Int},
                      K, P, Fs, r::Vector{Float64})
    T = length(sched)
    b = Vector{Vector{Float64}}(undef, T)
    kf = Vector{Vector{Float64}}(undef, T-1)
    b[T] = -Qs[sched[T]] * r
    @inbounds for t in (T-1):-1:1
        kf[t] = -(Fs[t] \ (B' * b[t+1]))
        Acl   = A - B * K[t]
        # λ_t = P_t x_t + b_t; the P B k and K'R k terms cancel exactly (Φ'P B = K'R),
        # leaving b_t = −Q_t r + Φ_tᵀ b_{t+1}. Checked against a direct QP solve.
        b[t]  = -Qs[sched[t]] * r + Acl' * b[t+1]
    end
    return kf
end

"`(K, kff, P)` for one reference — the convenience wrapper the diagnostics use."
function riccati(A, B, R, Qs, sched::Vector{Int}, r::Vector{Float64})
    K, P, Fs = riccati_gain(A, B, R, Qs, sched)
    return K, affine_sweep(A, B, R, Qs, sched, K, P, Fs, r), P
end

"Closed-loop forward roll with plant noise `Lx` and control noise `Lu` (factors)."
function rollout(rng, A, B, K, kf, x1, Lx, Lu)
    T = length(K) + 1; n, m = size(A, 1), size(B, 2)
    X = Matrix{Float64}(undef, n, T); X[:, 1] = x1
    @inbounds for t in 1:(T-1)
        u = -K[t] * X[:, t] + kf[t] + Lu * randn(rng, m)
        X[:, t+1] = A * X[:, t] + B * u + Lx * randn(rng, n)
    end
    return X
end

relrmse(f, r) = sqrt(mean(abs2, f .- r)) / sqrt(mean(abs2, r))
function pearson(f, r)
    fc = f .- mean(f); rc = r .- mean(r)
    d = sqrt(sum(abs2, fc) * sum(abs2, rc))
    return d < 1e-14 ? NaN : sum(fc .* rc) / d
end
utri(M) = [M[i, j] for j in 1:size(M, 2) for i in 1:j]
"Effective number of timesteps carrying a perturbation: (Σ‖d‖²)² / Σ‖d_t‖⁴."
partic(D) = sum(abs2, D)^2 / sum(t -> sum(abs2, D[:, t])^2, 1:size(D, 2))

# =========================================================================
# Exact marginal likelihood of the closed-loop chain.
# =========================================================================

"""
    loglik(st, Qs, refs, Σ_w, σ²_y, Ys, tg; A, B) -> Float64

`log p(y_{1:T} | θ)` for the closed-loop time-varying LDS

    x_{t+1} = (A - B K_t) x_t + B kff_t + w_t,   y_t = C x_t + ν_t,

marginalised exactly by Kalman filtering. The covariance/gain recursion depends
on neither the target nor the data, so it is built once per parameter evaluation
and shared across every trial — which is what makes a finite-difference gradient
over ~40 parameters affordable.
"""
function loglik(st::Setup, Qs, refs, Sig_w, sig_y2, Ys, tg; A = st.A, B = st.B)
    T, n, p = st.T, st.n, st.p
    K, P, Fs = riccati_gain(A, B, st.R, Qs, st.sched)
    Acl  = [A - B * K[t] for t in 1:(T-1)]
    inps = [[B * kf[t] for t in 1:(T-1)]
            for kf in (affine_sweep(A, B, st.R, Qs, st.sched, K, P, Fs, r) for r in refs)]

    Sy = sig_y2 * Matrix{Float64}(I, p, p)
    Gain = Vector{Matrix{Float64}}(undef, T)
    Fo = Vector{Cholesky{Float64,Matrix{Float64}}}(undef, T)
    logdet_sum = 0.0; Pc = st.P0
    for t in 1:T
        Fo[t] = cholesky(Symmetric(st.C * Pc * st.C' + Sy))
        logdet_sum += logdet(Fo[t])
        Gain[t] = (Pc * st.C') / Fo[t]
        Pf = Pc - Gain[t] * st.C * Pc; Pf = (Pf + Pf') / 2
        t == T && break
        Pc = Acl[t] * Pf * Acl[t]' + Sig_w; Pc = (Pc + Pc') / 2
    end

    quad = 0.0; xh = Vector{Float64}(undef, n)
    for i in eachindex(Ys)
        Y = Ys[i]; inp = inps[tg[i]]; copyto!(xh, st.mu0)
        for t in 1:T
            e = @views Y[:, t] - st.C * xh
            quad += dot(e, Fo[t] \ e)
            xh .+= Gain[t] * e
            t == T && break
            xh .= Acl[t] * xh .+ inp[t]
        end
    end
    N = length(Ys)
    return -0.5 * (N * T * p * log(2pi) + N * logdet_sum + quad)
end

function gen(rng, st::Setup, Qs; ntrial = 400, sig_x = 0.004, sig_u = 0.25, sig_y = 0.006)
    K, _, _ = riccati(st.A, st.B, st.R, Qs, st.sched, zeros(st.n))
    kffs = [riccati(st.A, st.B, st.R, Qs, st.sched, r)[2] for r in st.refs]
    Lx = sig_x * Matrix{Float64}(I, st.n, st.n); Lu = sig_u * Matrix{Float64}(I, st.m, st.m)
    L0 = cholesky(st.P0).L
    Ys = Vector{Matrix{Float64}}(undef, ntrial); tg = Vector{Int}(undef, ntrial)
    for i in 1:ntrial
        j = mod1(i, length(st.refs)); tg[i] = j
        X = rollout(rng, st.A, st.B, K, kffs[j], st.mu0 + L0*randn(rng, st.n), Lx, Lu)
        Ys[i] = st.C * X .+ sig_y * randn(rng, st.p, st.T)
    end
    return Ys, tg
end

# =========================================================================
function section_geometry()
    section("geometry — the symplectic flow, its stable manifold, and the contraction")
    st = setup(); Qs = true_costs()
    K, _, P = riccati(st.A, st.B, st.R, Qs, st.sched, zeros(st.n))
    Phi = [st.A - st.B * K[t] for t in 1:(st.T-1)]

    err = maximum(t -> norm(Phi[t]' * P[t+1] * Phi[t] -
                            (P[t] - (Qs[st.sched[t]] + K[t]' * st.R * K[t]))) / norm(P[t]),
                  1:(st.T-1))
    println("Contraction identity  Φ_t' P_{t+1} Φ_t = P_t - (Q_t + K_t' R K_t)")
    @printf("  max relative residual over t : %.1e\n", err)
    @printf("  max_t ρ(Φ_t)                 : %.4f  (individual steps need not be < 1;\n",
            maximum(t -> maximum(abs, eigvals(Phi[t])), 1:(st.T-1)))
    println("                                  the product contracts in the P-metric)")

    S = st.B * inv(st.R) * st.B'; Ai = inv(st.A)'
    M = [st.A + S*Ai*Qs[1]  -S*Ai; -Ai*Qs[1]  Ai]
    J = [zeros(st.n, st.n) I(st.n); -I(st.n) zeros(st.n, st.n)]
    ab = sort(abs.(eigvals(M)))
    println()
    println("The mixed-coordinate forward transition M (the model's own generative form)")
    println("  |μ|                    : ", join(round.(ab, digits=4), ", "))
    println("  reciprocal-pair products: ",
            join([round(ab[i] * ab[2*st.n + 1 - i], digits=8) for i in 1:st.n], ", "))
    @printf("  symplectic defect ‖M'JM - J‖/‖J‖ : %.1e\n", norm(M'*J*M - J)/norm(J))
    @printf("  growth over T = %d steps          : %.3g   <- why `rand` is unusable\n",
            st.T, maximum(ab)^st.T)

    V = [Matrix{Float64}(I, st.n, st.n); P[1]]      # the graph {λ = P x}
    W = M * V
    @printf("\n  ‖(I - proj)(M · graph(P))‖ / ‖M · graph(P)‖ : %.1e\n", norm(W - V*(V\W))/norm(W))
    println("  => graph(P) is M-invariant: it IS the stable manifold. The closed-loop")
    println("     model parameterises that manifold; the mixed-coordinate model")
    println("     parameterises the whole 2n-dim flow and conditions its way back.")
end

function section_gauge()
    section("gauge — the cost scale is an exact flat direction only when S is free")
    st = setup(); Qs = true_costs()
    mp(Q, r, R = st.R) = (kk = riccati(st.A, st.B, R, Q, st.sched, r);
                          rollout(MersenneTwister(1), st.A, st.B, kk[1], kk[2],
                                  zeros(st.n), zeros(st.n, st.n), zeros(st.m, st.m)))
    g1(Q, R = st.R) = riccati(st.A, st.B, R, Q, st.sched, zeros(st.n))[1][1]
    X0 = mp(Qs, st.refs[1]); K0 = g1(Qs)

    println("Q -> cQ with R = I held fixed:")
    @printf("%-8s %-16s %-14s %-19s %s\n", "c", "d‖K_1‖/‖K_1‖", "ρ(A-BK_1)",
            "d mean path (rel)", "endpoint |x_T - r|")
    for c in [0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0]
        Qc = [c*Q for Q in Qs]; K1 = g1(Qc); X = mp(Qc, st.refs[1])
        @printf("%-8s %-16.4f %-14.4f %-19.4f %.5f\n", c, norm(K1-K0)/norm(K0),
                maximum(abs, eigvals(st.A - st.B*K1)), norm(X-X0)/norm(X0),
                norm(X[1:2, end] - st.refs[1][1:2]))
    end
    println("\nThe same c, with S = B R^-1 B' free to absorb it (S -> S/c, i.e. R -> cR):")
    println("  this is `rescale_costate!`'s (λ,S,Q,…) -> (cλ,c⁻¹S,cQ,…)")
    for c in [0.25, 1.0, 4.0]
        Qc = [c*Q for Q in Qs]; K1 = g1(Qc, c*st.R); X = mp(Qc, st.refs[1], c*st.R)
        @printf("  c = %-6s d‖K_1‖/‖K_1‖ = %-10.3g d mean path = %.3g\n",
                c, norm(K1-K0)/norm(K0), norm(X-X0)/norm(X0))
    end
    println("  => exactly flat. Whenever S and Σ are both fitted the cost scale is a")
    println("     gauge orbit, so no objective and no amount of data can locate it.")
end

function section_profile()
    section("profile — the exact marginal likelihood along the cost-scale ray")
    st = setup(); Qs = true_costs()
    sx, su, sy = 0.004, 0.25, 0.006
    Sw = sx^2*Matrix(I, st.n, st.n) + su^2*(st.B*st.B')
    Ys, tg = gen(MersenneTwister(20240922), st, Qs; ntrial = 400,
                 sig_x = sx, sig_u = su, sig_y = sy)
    per = 1/(length(Ys)*st.T)
    base = per*loglik(st, Qs, st.refs, Sw, sy^2, Ys, tg)
    println("400 trials, T = $(st.T), position-only emission; all else at the truth.")
    @printf("\n%-10s %-16s %s\n", "c", "loglik/step", "deficit vs c = 1")
    cs = [0.0625, 0.125, 0.25, 0.5, 0.71, 1.0, 1.41, 2.0, 4.0, 8.0, 16.0]
    lls = Float64[]
    for c in cs
        ll = per*loglik(st, [c*Q for Q in Qs], st.refs, Sw, sy^2, Ys, tg)
        push!(lls, ll); @printf("%-10s %-16.6f %+.6f\n", c, ll, ll - base)
    end
    println("\nargmax over the ray: c = ", cs[argmax(lls)])
    println("=> an interior maximum at the true scale. Contrast README.md, where the")
    println("   held-out score is monotone in q0 and prefers the smallest value offered.")
end

function section_terminal()
    section("terminal — the Riccati Jacobian, and how far back a cost is felt")
    st = setup(); Qs = true_costs()
    K, _, P = riccati(st.A, st.B, st.R, Qs, st.sched, zeros(st.n))
    Phi = [st.A - st.B*K[t] for t in 1:(st.T-1)]
    # Ψ_{t,s} = Φ_{s-1} Φ_{s-2} ⋯ Φ_t  (order matters; the Φ do not commute)
    Psi(t, s) = (Z = Matrix{Float64}(I, st.n, st.n); for j in (s-1):-1:t; Z = Z*Phi[j]; end; Z)

    println("dP_t = dQ_t + Φ_t' dP_{t+1} Φ_t   (every dK term cancels by optimality)")
    println("  =>  dP_t = Σ_{s≥t} Ψ_{t,s}' dQ_s Ψ_{t,s}   —  closed form, no AD, no")
    println("      implicit solve; one extra backward sweep per gradient.")
    dQ = zeros(st.n, st.n); dQ[1, 1] = 1.0; dQ = (dQ + dQ')/2
    Pp = riccati(st.A, st.B, st.R, [Qs[1], Qs[2] + dQ], st.sched, zeros(st.n))[3]
    fdmax = maximum(t -> norm(Psi(t, st.T)'*dQ*Psi(t, st.T) - (Pp[t] - P[t])) /
                         max(norm(Pp[t] - P[t]), 1e-30), 1:(st.T-6))
    @printf("  finite-difference check, max relative error over t : %.1e\n", fdmax)

    println("\nBackward decay of terminal-cost information, ‖Ψ_{t,T}‖²:")
    @printf("%-6s %-20s %s\n", "t", "‖Ψ_{t,T}‖²", "relative to t = T-1")
    b = norm(Psi(st.T-1, st.T))^2
    for t in [st.T-1, st.T-3, st.T-6, st.T-10, st.T-15, st.T-20, 1]
        v = norm(Psi(t, st.T))^2
        @printf("%-6d %-20.3g %.3g\n", t, v, v/b)
    end

    mp(Q) = (kk = riccati(st.A, st.B, st.R, Q, st.sched, st.refs[1]);
             rollout(MersenneTwister(1), st.A, st.B, kk[1], kk[2],
                     zeros(st.n), zeros(st.n, st.n), zeros(st.m, st.m)))
    X0 = mp(Qs); D = mp([Qs[1], 1.2*Qs[2]]) - X0
    println("\nWhere a 20 % change in Q_term lands (relative change in the mean state):")
    println("  t:     ", join([lpad(t, 7) for t in 1:3:st.T]))
    println("  rel d: ", join([lpad(round(norm(D[:,t])/max(norm(X0[:,t]),1e-12), digits=4), 7)
                               for t in 1:3:st.T]))
    @printf("  participation ratio %.2f of %d;  share in observed rows %.1f %%\n",
            partic(D), st.T, 100*sum(abs2, D[1:2, :])/sum(abs2, D))
    println("  (mixed-coordinate: Q_term enters ONE factor per trial, ratio 1.00 of $(st.T))")

    println("\nAgainst the running/terminal balance — a property of the task, not the fit:")
    @printf("%-14s %-22s %s\n", "Q_run scale", "participation ratio", "share in observed rows")
    for s in [0.0, 0.01, 0.1, 1.0, 10.0]
        Q = [s*Qs[1], Qs[2]]
        Y0 = mp(Q); Dd = mp([s*Qs[1], 1.2*Qs[2]]) - Y0
        @printf("%-14s %-22.2f %.1f %%\n", s, partic(Dd),
                100*sum(abs2, Dd[1:2, :])/sum(abs2, Dd))
    end
    println("=> the closed loop does NOT rescue the terminal cost. It is forgotten")
    println("   backwards at the closed-loop rate; fix it in the design, not the model.")
end

function section_reference()
    section("reference — the feedforward puts it in the observed coordinates")
    st = setup(); Qs = true_costs()
    mp(r) = (kk = riccati(st.A, st.B, st.R, Qs, st.sched, r);
             rollout(MersenneTwister(1), st.A, st.B, kk[1], kk[2],
                     zeros(st.n), zeros(st.n, st.n), zeros(st.m, st.m)))
    X0 = mp(st.refs[1]); rp = copy(st.refs[1]); rp[1:2] *= 1.1
    D = mp(rp) - X0
    println("One target scaled by 1.1 (relative change in the mean state):")
    println("  t:     ", join([lpad(t, 7) for t in 1:3:st.T]))
    println("  rel d: ", join([lpad(round(norm(D[:,t])/max(norm(X0[:,t]),1e-12), digits=4), 7)
                               for t in 1:3:st.T]))
    @printf("  participation ratio %.2f of %d;  share in observed rows %.1f %%\n",
            partic(D), st.T, 100*sum(abs2, D[1:2, :])/sum(abs2, D))
    println("=> identified from the state mean at essentially every timestep. In the")
    println("   mixed-coordinate model Gref enters only the costate half of the affine")
    println("   term, hence README.md's trade: Gref 1.041/0.01 in the best cost row.")
end

# =========================================================================
# switching: γ at the generating parameters, matched to slds.jl
# =========================================================================
logsumexp2(a, b) = (m = max(a, b); m == -Inf ? -Inf : m + log(exp(a-m) + exp(b-m)))

function section_switching()
    section("switching — γ at the generating parameters, delay-then-reach")
    n, T, NTRIAL, STAY = 2, 30, 120, 0.93
    Ap = [0.96 0.05; -0.04 0.96]                       # model.jl `plant(2)`
    Ssym = [0.05 0.01; 0.01 0.05]
    B = Matrix(sqrt(Symmetric(Ssym)))                  # B R⁻¹ B' = S at R = I
    Rc = Matrix{Float64}(I, n, n)
    Qrun = [0.20 0.03; 0.03 0.20]                      # model.jl `cost_bank(2)`
    Qterm = 3.0 * Matrix{Float64}(I, n, n)
    Sig_l = 0.02 * Matrix{Float64}(I, n, n)            # slds.jl `state_noise`
    Af, Sig_f = 0.90*Matrix{Float64}(I,n,n), 0.15*Matrix{Float64}(I,n,n)  # `free_drift`
    Cobs, Robs = Matrix{Float64}(I,n,n), 0.05*Matrix{Float64}(I,n,n)
    P0 = 0.2 * Matrix{Float64}(I, n, n)
    sched = vcat(fill(1, T-1), 2)
    K, kff, _ = riccati(Ap, B, Rc, [Qrun, Qterm], sched, zeros(n))
    Acl = [Ap - B*K[t] for t in 1:(T-1)]
    @printf("free drift ρ = %.3f   LQR closed loop ρ(Φ_1) = %.4f  (regimes told apart by\n",
            0.9, maximum(abs, eigvals(Acl[1])))
    println("  innovation size and feedforward, not by decay rate — not a rigged contrast)")
    trans(k, t) = k == 1 ? (Acl[t], B*kff[t], Sig_l) : (Af, zeros(n), Sig_f)

    rng = MersenneTwister(7)
    Lf, Ll, L0 = cholesky(Sig_f).L, cholesky(Sig_l).L, cholesky(P0).L
    Lo = cholesky(Robs).L
    Ys = Vector{Matrix{Float64}}(undef, NTRIAL); Zs = Vector{Vector{Int}}(undef, NTRIAL)
    sws = Vector{Int}(undef, NTRIAL)
    for i in 1:NTRIAL
        tsw = rand(rng, (T÷3):(2T÷3)); sws[i] = tsw
        X = Matrix{Float64}(undef, n, T); X[:, 1] = L0*randn(rng, n); z = fill(2, T)
        for t in 1:(T-1)
            if t + 1 <= tsw
                X[:, t+1] = Af*X[:, t] + Lf*randn(rng, n); z[t+1] = 2
            else
                X[:, t+1] = Acl[t]*X[:, t] + B*kff[t] + Ll*randn(rng, n); z[t+1] = 1
            end
        end
        Zs[i] = z; Ys[i] = Cobs*X .+ Lo*randn(rng, n, T)
    end

    # Structured variational E-step: responsibility-weighted Gaussian chain <-> HMM,
    # the same plug-in scheme the harness scores γ with.
    Ri, iP0 = inv(Robs), inv(P0)
    Pinv = [inv(Sig_l), inv(Sig_f)]
    ld = [logdet(2pi*Sig_l), logdet(2pi*Sig_f)]
    logA = log.([STAY 1-STAY; 1-STAY STAY]); logpi = log.([1e-6, 1-1e-6])
    rows(t) = ((t-1)*n+1):(t*n)
    tp = zeros(2); nk = zeros(2); mass = Float64[]; onset = Float64[]
    for i in 1:NTRIAL
        Y = Ys[i]; g = fill(0.5, 2, T); M1 = Matrix{Float64}(undef, n, T); Cv = nothing
        for _ in 1:60
            L = zeros(n*T, n*T); h = zeros(n*T)
            L[rows(1), rows(1)] += iP0
            for t in 1:T
                L[rows(t), rows(t)] += Cobs'*Ri*Cobs; h[rows(t)] += Cobs'*Ri*Y[:, t]
            end
            for t in 1:(T-1), k in 1:2
                w = g[k, t+1]; w < 1e-12 && continue
                Ak, bk, _ = trans(k, t); Pk = Pinv[k]
                L[rows(t), rows(t)]     += w*(Ak'*Pk*Ak)
                L[rows(t+1), rows(t+1)] += w*Pk
                L[rows(t), rows(t+1)]   -= w*(Ak'*Pk)
                L[rows(t+1), rows(t)]   -= w*(Pk*Ak)
                h[rows(t)]              -= w*(Ak'*Pk*bk)
                h[rows(t+1)]            += w*(Pk*bk)
            end
            F = cholesky(Symmetric(L)); mu = F \ h; Cv = inv(F)
            for t in 1:T; M1[:, t] = mu[rows(t)]; end
            lp = zeros(2, T)
            for t in 1:(T-1), k in 1:2
                Ak, bk, _ = trans(k, t); Pk = Pinv[k]
                r = M1[:, t+1] - Ak*M1[:, t] - bk
                Vtt, Vt1, Vc = Cv[rows(t), rows(t)], Cv[rows(t+1), rows(t+1)], Cv[rows(t), rows(t+1)]
                lp[k, t+1] = -0.5*(ld[k] + tr(Pk*(Vt1 - Ak*Vc - Vc'*Ak' + Ak*Vtt*Ak' + r*r')))
            end
            la = fill(-Inf, 2, T); lb = zeros(2, T); la[:, 1] = logpi
            for t in 2:T, k in 1:2
                la[k, t] = lp[k, t] + logsumexp2(la[1,t-1]+logA[1,k], la[2,t-1]+logA[2,k])
            end
            for t in (T-1):-1:1, k in 1:2
                lb[k, t] = logsumexp2(logA[k,1]+lp[1,t+1]+lb[1,t+1],
                                      logA[k,2]+lp[2,t+1]+lb[2,t+1])
            end
            gn = la .+ lb
            for t in 1:T
                mm = maximum(gn[:, t]); e = exp.(gn[:, t] .- mm); g[:, t] = e ./ sum(e)
            end
        end
        z = Zs[i]
        for t in 2:T
            k = argmax(g[:, t]); nk[z[t]] += 1; tp[z[t]] += (k == z[t])
            push!(mass, g[z[t], t])
        end
        hit = findfirst(t -> g[1, t] > 0.5, 2:T)
        push!(onset, hit === nothing ? T - sws[i] : abs(hit - sws[i]))
    end
    println()
    println("γ at the GENERATING parameters, closed-loop parameterisation:")
    @printf("  balanced accuracy   %.3f\n", 0.5*(tp[1]/nk[1] + tp[2]/nk[2]))
    @printf("  mean posterior mass %.3f\n", mean(mass))
    @printf("  mean onset error    %.2f timesteps\n", mean(onset))
    println()
    println("For contrast, README.md / slds.jl measured γ at the generating parameters")
    println("in the mixed-coordinate model as a function of the costate innovation —")
    println("a parameter the closed-loop model does not have:")
    println("  Σ_λλ = 1e-4  -> 0.52   (the single-system optimum)")
    println("  Σ_λλ = 1e-2  -> 0.74")
    println("  Σ_λλ = 2e-2  -> 0.81-0.83,  onset 5.5-6.3   (the switching default)")
    println("The margin is within configuration differences. The point is that the")
    println("column of Σ_λλ values does not exist, so γ and the cost cannot conflict.")
end

# =========================================================================
function section_shaping()
    section("shaping — an exact equivalence class of costs")
    st = setup(); Qs = true_costs(); A, B = st.A, st.B
    N = nullspace(Matrix(B'))                  # basis of range(B)^⊥
    X = [3.0e4 -1.1e4; -1.1e4 5.0e4]           # an arbitrary symmetric potential
    M = N*X*N'
    println("For N spanning range(B)^⊥, Nᵀ Φ_t = Nᵀ(A − B K_t) = Nᵀ A for every t: control")
    println("cannot move the unactuated components in one step. So the potential x'Mx with")
    println("M = N X Nᵀ can be added to the cost-to-go at every step without changing any")
    println("decision — potential-based reward shaping, restricted to that subspace:")
    println("    Q_term → Q_term + M,     every running Q_k → Q_k + M − AᵀMA")
    @printf("  dimension of the class: (n − m)(n − m + 1)/2 = %d here; ‖BᵀM‖ = %.1e\n",
            size(N, 2)*(size(N, 2) + 1) ÷ 2, norm(B'*M))
    Qsh = [Qs[1] + M - A'*M*A, Qs[2] + M]
    for (j, r) in enumerate(st.refs[1:3])
        K0, P0, F0 = riccati_gain(A, B, st.R, Qs, st.sched)
        K1, P1, F1 = riccati_gain(A, B, st.R, Qsh, st.sched)
        k0 = affine_sweep(A, B, st.R, Qs, st.sched, K0, P0, F0, r)
        k1 = affine_sweep(A, B, st.R, Qsh, st.sched, K1, P1, F1, r)
        @printf("  target %d: max ‖ΔK_t‖/‖K_t‖ = %.1e, max ‖Δk_t‖/‖k_t‖ = %.1e, max ‖ΔP_t − M‖/‖M‖ = %.1e\n",
                j, maximum(t -> norm(K1[t]-K0[t])/norm(K0[t]), 1:st.T-1),
                maximum(t -> norm(k1[t]-k0[t])/max(norm(k0[t]), 1e-300), 1:st.T-1),
                maximum(t -> norm(P1[t]-P0[t]-M)/norm(M), 1:st.T))
    end
    @printf("  (the shift is not small: ‖ΔQ_term‖/‖Q_term‖ = %.2f, ‖ΔQ_run‖/‖Q_run‖ = %.2f)\n",
            norm(M)/norm(Qs[2]), norm(M - A'*M*A)/norm(Qs[1]))
    println()
    println("Gains are unchanged, so every feedback signature — trial-to-trial variability,")
    println("responses to UNEXPECTED perturbations, max-ent control noise (which sees P only")
    println("through BᵀPB) — is blind to it. The feedforward is unchanged for any reference")
    println("with Nᵀ(A r − r) = 0, i.e. a target at rest. What breaks it is a reference or an")
    println("ANTICIPATED disturbance d_t with Nᵀ(A r + d_t − r) ≠ 0:")
    rv = [0.12, 0.0, 0.3, 0.0]                  # a target specified with a velocity
    K0, P0, F0 = riccati_gain(A, B, st.R, Qs, st.sched)
    K1, P1, F1 = riccati_gain(A, B, st.R, Qsh, st.sched)
    k0 = affine_sweep(A, B, st.R, Qs, st.sched, K0, P0, F0, rv)
    k1 = affine_sweep(A, B, st.R, Qsh, st.sched, K1, P1, F1, rv)
    @printf("  target with velocity: ‖Nᵀ(A r − r)‖ = %.3g, max ‖Δk_t‖/‖k_t‖ = %.2e\n",
            norm(N'*(A*rv - rv)), maximum(t -> norm(k1[t]-k0[t])/norm(k0[t]), 1:st.T-1))
    println()
    println("A single cost regime has no such freedom (it would need M = M − AᵀMA, so AᵀMA = 0,")
    println("so M = 0). The class needs a terminal cost distinct from the running one — which")
    println("is exactly the configuration in which README.md finds the terminal cost never")
    println("recovered. That finding is this equivalence class, not a shortage of data.")
end

# =========================================================================
function section_subspace()
    section("subspace — what the closed loop identifies, and what optimality adds")
    st = setup(); Qs = true_costs(); A, B, R = st.A, st.B, st.R
    K, P, _ = riccati_gain(A, B, R, Qs, st.sched)
    Phi = [A - B*K[t] for t in 1:(st.T-1)]
    Dm = hcat([Phi[t] - Phi[end] for t in 1:(st.T-2)]...)
    sv = svdvals(Dm)
    println("Φ_t − Φ_s = −B (K_t − K_s), so the time variation of the closed loop spans range(B).")
    println("  singular values of [Φ_t − Φ_{T−1}]_t : ", join(round.(sv, sigdigits = 3), ", "))
    U = svd(Dm).U[:, 1:size(B, 2)]
    @printf("  principal angles between the recovered subspace and range(B): %s rad\n",
            join(round.(acos.(clamp.(svdvals(U'*Matrix(qr(B).Q)[:, 1:size(B, 2)]), -1, 1)), sigdigits = 2), ", "))
    @printf("  max_t ‖(I − BB⁺)(Φ_t − A)‖ = %.1e — the unactuated part of A is read directly\n",
            maximum(t -> norm((I - B*pinv(B))*(Phi[t] - A)), 1:(st.T-1)))
    println("  => the NUMBER of effective control channels and their directions are identified")
    println("     from a closed-loop fit with no optimality assumption, provided the gains vary")
    println("     over the trial (a finite horizon, or several cost regimes).")
    println()
    println("What remains: A − B K_t = (A + B D) − B (K_t + D) for any constant D. Only the")
    println("optimality structure can separate intrinsic dynamics from a constant feedback.")
    println("Inverse-optimality least squares for Q at plant A + εBD (linear in Q):")
    println("    R K_t = Bᵀ P_{t+1}(Q) Φ_t,    P_t = Q_t + K_tᵀ R K_t + Φ_tᵀ P_{t+1} Φ_t")
    n = st.n; K_reg = length(Qs); nq = n*(n+1) ÷ 2
    unvec(q) = (Qv = [zeros(n, n) for _ in 1:K_reg]; j = 1;
                for k in 1:K_reg, c in 1:n, r in c:n; Qv[k][r, c] = q[j]; Qv[k][c, r] = q[j]; j += 1 end; Qv)
    function resid(q, Kd, tset)
        Qv = unvec(q); Pt = Vector{Matrix{Float64}}(undef, st.T); Pt[st.T] = Qv[st.sched[st.T]]
        for t in (st.T-1):-1:1
            Pt[t] = Qv[st.sched[t]] + Kd[t]'*R*Kd[t] + Phi[t]'*Pt[t+1]*Phi[t]
        end
        return vcat([vec(R*Kd[t] - B'*Pt[t+1]*Phi[t]) for t in tset]...)
    end
    function ls(Kd, tset)
        c = resid(zeros(nq*K_reg), Kd, tset)
        Mj = hcat([resid(Matrix{Float64}(I, nq*K_reg, nq*K_reg)[:, j], Kd, tset) - c for j in 1:nq*K_reg]...)
        q = -(Mj \ c)
        return norm(Mj*q + c)/norm(vcat([vec(R*Kd[t]) for t in tset]...)), Mj
    end
    D0 = randn(MersenneTwister(5), size(B, 2), n); D0 .*= 0.1*norm(K[1])/norm(D0)
    for (label, tset) in (("all 29 steps (gains vary)", 1:(st.T-1)),
                          ("t ≤ 12 only (gains near stationary)", 1:12))
        @printf("  %-38s", label)
        for ε in (0.0, 0.1, 1.0)
            rr, _ = ls([K[t] + ε*D0 for t in 1:(st.T-1)], tset)
            @printf("  ε=%-4s residual %-9.2e", ε, rr)
        end
        println()
    end
    println("  => a constant feedback offset IS rejected by optimality, but ~8x more weakly when")
    println("     the gains are near stationary. Identify A from uncontrolled epochs if you can.")
    _, Mg = ls(K, 1:(st.T-1))
    scl = [norm(Mg[:, j]) for j in 1:size(Mg, 2)]
    svn = svdvals(Mg ./ scl')
    @printf("\n  rank of the stationarity equations in Q: %d of %d; three smallest normalized σ: %s\n",
            count(>(1e-8), svn), nq*K_reg, join(round.(svn[end-2:end], sigdigits = 2), ", "))
    println("  => the missing three are exactly the shaping class above: the least squares")
    println("     recovers Q only modulo it, so it must be gauge-fixed before it can initialize.")
    c0 = resid(zeros(nq*K_reg), K, 1:(st.T-1))
    Qh = unvec(-(Mg \ c0))
    inv_run(Qr, Qt) = Qr - Qt + A'*Qt*A
    @printf("\n  raw least-squares cost: ‖Q̂_run − Q_run‖/‖Q_run‖ = %.2f, ‖Q̂_term − Q_term‖/‖Q_term‖ = %.2f\n",
            norm(Qh[1] - Qs[1])/norm(Qs[1]), norm(Qh[2] - Qs[2])/norm(Qs[2]))
    @printf("  shaping invariants:     ‖Bᵀ(Q̂_term − Q_term)‖/‖BᵀQ_term‖ = %.1e, ‖Q̃̂ − Q̃‖/‖Q̃‖ = %.1e\n",
            norm(B'*(Qh[2] - Qs[2]))/norm(B'*Qs[2]),
            norm(inv_run(Qh[1], Qh[2]) - inv_run(Qs[1], Qs[2]))/norm(inv_run(Qs[1], Qs[2])))
    println("    (Q̃ = Q_run − Q_term + AᵀQ_term A; both are unchanged by every shaping move)")
    println("  conditioning — relative noise on the gains vs error in the invariant Q̃:")
    for sd in (1e-3, 1e-2, 5e-2)
        rng = MersenneTwister(9); errs = Float64[]
        for _ in 1:20
            Kn = [K[t] + sd*norm(K[t])*randn(rng, size(K[t])...)/sqrt(length(K[t])) for t in 1:(st.T-1)]
            Phin = [A - B*Kn[t] for t in 1:(st.T-1)]
            function resn(q)
                Qv = unvec(q); Pt = Vector{Matrix{Float64}}(undef, st.T); Pt[st.T] = Qv[st.sched[st.T]]
                for t in (st.T-1):-1:1
                    Pt[t] = Qv[st.sched[t]] + Kn[t]'*R*Kn[t] + Phin[t]'*Pt[t+1]*Phin[t]
                end
                return vcat([vec(R*Kn[t] - B'*Pt[t+1]*Phin[t]) for t in 1:(st.T-1)]...)
            end
            cn = resn(zeros(nq*K_reg))
            Mn = hcat([resn(Matrix{Float64}(I, nq*K_reg, nq*K_reg)[:, j]) - cn for j in 1:nq*K_reg]...)
            Qn = unvec(-(Mn \ cn))
            push!(errs, norm(inv_run(Qn[1], Qn[2]) - inv_run(Qs[1], Qs[2]))/norm(inv_run(Qs[1], Qs[2])))
        end
        @printf("    gain noise %.0e  →  median invariant error %.3g\n", sd, sort(errs)[10])
    end
    println("  => an initializer that needs a precise structure fit, not an estimator.")
end

# =========================================================================
function section_sweeps()
    section("sweeps — the feedforward against a QP, and the reverse-mode gradient")
    st = setup(); Qs = true_costs(); A, B, R = st.A, st.B, st.R
    r = st.refs[1]; n, m, T = st.n, st.m, st.T
    K, P, Fs = riccati_gain(A, B, R, Qs, st.sched); k = affine_sweep(A, B, R, Qs, st.sched, K, P, Fs, r)
    cost(X, U) = sum(0.5*(X[:, t] - r)'*Qs[st.sched[t]]*(X[:, t] - r) + 0.5*U[:, t]'*R*U[:, t] for t in 1:(T-1)) +
                 0.5*(X[:, T] - r)'*Qs[st.sched[T]]*(X[:, T] - r)
    X = zeros(n, T); U = zeros(m, T-1)
    for t in 1:(T-1); U[:, t] = -K[t]*X[:, t] + k[t]; X[:, t+1] = A*X[:, t] + B*U[:, t]; end
    # direct QP over the open-loop controls from x₁ = 0 (noiseless ⇒ the same optimum)
    Gx = [zeros(n, m*(T-1)) for _ in 1:T]
    for t in 2:T; Gx[t] = A*Gx[t-1]; Gx[t][:, (t-2)*m+1:(t-1)*m] += B; end
    H = kron(Matrix{Float64}(I, T-1, T-1), R); g = zeros(m*(T-1))
    for t in 1:T; Q = Qs[st.sched[t]]; H += Gx[t]'*Q*Gx[t]; g += Gx[t]'*Q*(-r); end
    uq = -(Symmetric(H) \ g); Xq = hcat([Gx[t]*uq for t in 1:T]...)
    @printf("  sweep: cost %.6g, |x_T − r| = %.3g    QP: cost %.6g, |x_T − r| = %.3g\n",
            cost(X, U), norm(X[1:2, end] - r[1:2]), cost(Xq, reshape(uq, m, T-1)), norm(Xq[1:2, end] - r[1:2]))
    println("  (the recursion b_t = −Q_t r + Φ_tᵀ b_{t+1} is the QP optimum; an earlier version")
    println("   of this script carried a spurious 2KᵀRk term and cost 16 % more)")

    # reverse-mode gradient of both sweeps, on a random problem with three regimes
    rng = MersenneTwister(4); n2, m2, T2 = 4, 2, 12
    A2 = I + 0.1*randn(rng, n2, n2); B2 = randn(rng, n2, m2)
    R2 = Matrix(Symmetric(I + 0.2*(x = randn(rng, m2, m2); x*x')))
    Q2 = [(L = randn(rng, n2, n2); Matrix(Symmetric(L*L'))) for _ in 1:3]
    sc2 = vcat(fill(1, 6), fill(2, T2 - 7), 3); r2 = randn(rng, n2)
    W = [randn(rng, n2, n2) for _ in 1:(T2-1)]; w = [randn(rng, m2) for _ in 1:(T2-1)]
    V = randn(rng, n2, n2); v = randn(rng, n2)
    function fwd(Aq, Bq, Rq, Qq, rq)
        Kq, Pq, Fq = riccati_gain(Aq, Bq, Rq, Qq, sc2)
        bq = Vector{Vector{Float64}}(undef, T2); bq[T2] = -Qq[sc2[T2]]*rq
        for t in (T2-1):-1:1; bq[t] = -Qq[sc2[t]]*rq + (Aq - Bq*Kq[t])'*bq[t+1]; end
        kq = affine_sweep(Aq, Bq, Rq, Qq, sc2, Kq, Pq, Fq, rq)
        return (; K = Kq, P = Pq, G = [Matrix(Fq[t]) for t in 1:(T2-1)], b = bq, k = kq,
                Phi = [Aq - Bq*Kq[t] for t in 1:(T2-1)])
    end
    loss(f) = sum(dot(W[t], f.Phi[t]) + dot(w[t], f.k[t]) for t in 1:(T2-1)) + dot(V, f.P[1]) + dot(v, f.b[1])
    symm(X) = (X + X')/2
    f = fwd(A2, B2, R2, Q2, r2)
    Ab = zeros(n2, n2); Bb = zeros(n2, m2); Rb = zeros(m2, m2); Qb = [zeros(n2, n2) for _ in 1:3]; rb = zeros(n2)
    Pb = [zeros(n2, n2) for _ in 1:T2]; bb = [zeros(n2) for _ in 1:T2]; Pb[1] = symm(V); bb[1] = copy(v)
    for t in 1:(T2-1)
        P1 = f.P[t+1]; Ph = f.Phi[t]; Kt = f.K[t]; G = f.G[t]; kq = sc2[t]
        Qb[kq] .-= symm(bb[t]*r2'); rb .-= Q2[kq]*bb[t]
        Phib = W[t] + f.b[t+1]*bb[t]'; bb[t+1] .+= Ph*bb[t]
        z = G \ w[t]; bb[t+1] .-= B2*z; Bb .-= f.b[t+1]*z'; Gb = -symm(z*f.k[t]')
        Qb[kq] .+= Pb[t]; Rb .+= Kt*Pb[t]*Kt'
        Ab .+= 2*P1*Ph*Pb[t]; Bb .-= 2*P1*Ph*Pb[t]*Kt'; Pb[t+1] .+= Ph*Pb[t]*Ph'
        Ab .+= Phib; Bb .-= Phib*Kt'; Z = G \ (-B2'*Phib)
        Bb .+= P1*Ph*Z' - P1*B2*Z*Kt'; Pb[t+1] .+= symm(B2*Z*Ph'); Ab .+= P1*B2*Z; Rb .-= symm(Z*Kt')
        Rb .+= Gb; Bb .+= 2*P1*B2*Gb; Pb[t+1] .+= B2*Gb*B2'
    end
    kT = sc2[T2]; Qb[kT] .+= Pb[T2]; Qb[kT] .-= symm(bb[T2]*r2'); rb .-= Q2[kT]*bb[T2]
    h = 1e-6
    fd(pert) = (loss(pert(h)) - loss(pert(-h)))/(2h)
    worst(G, pert) = maximum(i -> abs(fd(s_ -> pert(i, s_)) - G[i])/max(1.0, abs(fd(s_ -> pert(i, s_)))),
                             CartesianIndices(G))
    symworst(G, M, mk) = maximum(filter(i -> i[1] <= i[2], collect(CartesianIndices(G)))) do i
        j = CartesianIndex(i[2], i[1]); ga = j == i ? G[i] : G[i] + G[j]
        d = fd(s_ -> (Mp = copy(M); Mp[i] += s_; j != i && (Mp[j] += s_); mk(Mp)))
        abs(d - ga)/max(1.0, abs(d))
    end
    println("\n  reverse-mode gradient vs central differences (n=4, m=2, T=12, three regimes):")
    @printf("    A %.1e   B %.1e   R %.1e", worst(Ab, (i, s_) -> (Ap = copy(A2); Ap[i] += s_; fwd(Ap, B2, R2, Q2, r2))),
            worst(Bb, (i, s_) -> (Bp = copy(B2); Bp[i] += s_; fwd(A2, Bp, R2, Q2, r2))),
            symworst(symm(Rb), R2, Rp -> fwd(A2, B2, Rp, Q2, r2)))
    for q in 1:3
        @printf("   Q[%d] %.1e", q, symworst(symm(Qb[q]), Q2[q], Qp -> (Qv = copy(Q2); Qv[q] = Qp; fwd(A2, B2, R2, Qv, r2))))
    end
    @printf("   r %.1e\n", worst(rb, (i, s_) -> (rp = copy(r2); rp[i] += s_; fwd(A2, B2, R2, Q2, rp))))
    println("  => the M-step gradient needs no AD and no implicit solve.")
end

# =========================================================================
# fit: end-to-end recovery, swept over the initial cost scale
# =========================================================================
ntri(n) = n*(n+1) ÷ 2
lower_tri!(L, v) = (k = 1; for j in 1:size(L,1), i in j:size(L,1); L[i,j] = v[k]; k += 1 end; L)

#=
Everything the objective needs, in one concretely typed struct.

This is not tidiness. When `unpack` and the objective are closures capturing a
dozen locals of a section function, Julia cannot infer their return types, every
small matrix operation dispatches dynamically, and the fit runs about an order
of magnitude slower — measured at 3.5e9 allocations before this was hoisted out.
=#
struct FitSpec
    st::Setup
    Ys::Vector{Matrix{Float64}}
    tg::Vector{Int}
    per::Float64
    ntg::Int
    n::Int
    k::Int
    np::Int
end

function fit_unpack(sp::FitSpec, v::Vector{Float64})
    n, k, ntg = sp.n, sp.k, sp.ntg
    L1 = lower_tri!(zeros(n, n), view(v, 1:k))
    L2 = lower_tri!(zeros(n, n), view(v, (k+1):2k))
    Qs = [Matrix(Symmetric(L1*L1')), Matrix(Symmetric(L2*L2'))]
    off = 2k
    refs = [[v[off+2j-1], v[off+2j], 0.0, 0.0] for j in 1:ntg]
    off += 2*ntg
    ex, eu, ey = exp(v[off+1]), exp(v[off+2]), exp(v[off+3])
    Sw = ex^2*Matrix{Float64}(I, n, n) + eu^2*(sp.st.B*sp.st.B')
    return Qs, refs, Sw, ey^2
end

function fit_nll(sp::FitSpec, v::Vector{Float64})
    Qs, refs, Sw, sy2 = fit_unpack(sp, v)
    try
        return -sp.per * loglik(sp.st, Qs, refs, Sw, sy2, sp.Ys, sp.tg)
    catch
        return 1e8
    end
end

"A cold start: isotropic costs at scale `q0`, references at zero."
function fit_start(sp::FitSpec, q0::Float64)
    n, k, ntg = sp.n, sp.k, sp.ntg
    v = zeros(sp.np); d = sqrt(q0)*10
    for o in (0, k)
        j = 1
        for c in 1:n, r in c:n; v[o+j] = (r == c ? d : 0.0); j += 1 end
    end
    v[2k+2*ntg+1] = log(0.01); v[2k+2*ntg+2] = log(0.5); v[2k+2*ntg+3] = log(0.01)
    return v
end

function section_fit()
    section("fit — end-to-end recovery, swept over the initial cost scale")
    st = setup(); Qs = true_costs()
    sx, su, sy = 0.004, 0.25, 0.006
    ntrial, iters = QUICK ? (100, 40) : (400, 400)
    Ys, tg = gen(MersenneTwister(20240922), st, Qs; ntrial = ntrial,
                 sig_x = sx, sig_u = su, sig_y = sy)
    n, k = st.n, ntri(st.n); ntg = length(st.refs); np = 2k + 2*ntg + 3
    sp = FitSpec(st, Ys, tg, 1/(length(Ys)*st.T), ntg, n, k, np)
    QUICK && println("--quick: $(ntrial) trials, $(iters) iterations — a smoke test, not a result.")

    packL(L) = (v = zeros(k); j = 1; for c in 1:n, r in c:n; v[j] = L[r, c]; j += 1 end; v)
    vtruth = vcat(packL(cholesky(Symmetric(Qs[1])).L), packL(cholesky(Symmetric(Qs[2])).L),
                  vcat([r[1:2] for r in st.refs]...), [log(sx), log(su), log(sy)])
    cl(Q, A = st.A) = A - st.B*riccati(A, st.B, st.R, Q, st.sched, zeros(n))[1][1]
    rt = vcat([r[1:2] for r in st.refs]...)
    per = sp.per

    @printf("%d parameters fitted (known plant: 2 cost matrices, %d references, 3 noise scalars)\n",
            np, ntg)
    @printf("truth: tr(Qrun) = %.1f, tr(Qterm) = %.1f, nll/step = %.6f\n",
            tr(Qs[1]), tr(Qs[2]),
            -per*loglik(st, Qs, st.refs, sx^2*Matrix(I,n,n) + su^2*(st.B*st.B'), sy^2, Ys, tg))
    flush(stdout)
    f = v -> fit_nll(sp, v)
    sc(x) = @sprintf("%.3g", x)
    pair(a, b) = string(sc(a), "/", round(b, digits=3))
    @printf("\n%-22s %-16s %-16s %-10s %-14s %-12s %s\n", "start", "Qrun rmse/corr",
            "Qterm rmse/corr", "scale err", "Gref rmse/corr", "closed-loop", "nll/step")
    grid = QUICK ? [0.05, 100.0] : [0.05, 1.0, 100.0, 10_000.0]
    runs = vcat([("q0 = $(q0)", fit_start(sp, q0)) for q0 in grid],
                [("warm start at truth", copy(vtruth))])
    for (label, v0) in runs
        res = Optim.optimize(f, v0, LBFGS(linesearch = BackTracking()),
                             Optim.Options(iterations = iters, g_abstol = 1e-10,
                                           f_reltol = 1e-14))
        Q, refs, _, _ = fit_unpack(sp, Optim.minimizer(res))
        rf = vcat([r[1:2] for r in refs]...)
        @printf("%-22s %-16s %-16s %-10s %-14s %-12.4f %.6f\n", label,
                pair(relrmse(utri(Q[1]), utri(Qs[1])), pearson(utri(Q[1]), utri(Qs[1]))),
                pair(relrmse(utri(Q[2]), utri(Qs[2])), pearson(utri(Q[2]), utri(Qs[2]))),
                sc(abs(tr(Q[1])/tr(Qs[1]) - 1)),
                pair(relrmse(rf, rt), pearson(rf, rt)),
                relrmse(cl(Q), cl(Qs)), Optim.minimum(res))
        flush(stdout)
    end
    println()
    println("Four readings.")
    println(" 1. The warm start at the truth STAYS there: the objective moves by ~1e-3 per")
    println("    step and every block holds, so the truth is a stationary point. README.md")
    println("    reports the opposite for the mixed-coordinate fit (Qc 0.394 from the truth).")
    println(" 2. Every cold start recovers the reference and the cost's SHAPE (correlation")
    println("    0.93-0.98), under an emission that never sees velocity.")
    println(" 3. The cost's SCALE is set by the start: far starts end tens of times too")
    println("    large, because over-scaling costs almost no likelihood (--only=profile).")
    println("    Take the scale from a prior or from behavioural variability, or report only")
    println("    scale-free quantities.")
    println(" 4. The likelihood separates the converged (warm) fit from every cold start by")
    println("    100+ nats, but does NOT rank the unconverged cold starts by accuracy: they")
    println("    are unconverged along the shallow direction, not alternative optima.")
    println("Q_term stays near 1.0 from cold starts and holds only at the warm start, where")
    println("it simply does not move: its unactuated block is the flat shaping class")
    println("(--only=shaping).")
end

for (name, f) in (("geometry", section_geometry), ("gauge", section_gauge),
                  ("profile", section_profile), ("terminal", section_terminal),
                  ("reference", section_reference), ("shaping", section_shaping),
                  ("subspace", section_subspace), ("sweeps", section_sweeps),
                  ("switching", section_switching),
                  ("fit", section_fit))
    want(name) && f()
end
println()
