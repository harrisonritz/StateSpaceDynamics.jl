#=============================================================================
Workloads for benchmarking the inverse-LQR EM fit.

Each workload is a *fit-ready* model plus ragged data, built deterministically
from a `StableRNG`, so two runs (two revisions, two thread counts) see the same
problem bit for bit. The shapes follow the smoulder-reward suite in
`docs/dev/lqr/smoulder.jl` (ring targets, reward-dependent running/terminal
costs, Poisson channels with low rates), with one deliberate change: trial
lengths are drawn from a log-normal and are therefore *ragged*, nearly all
distinct at small trial counts and spread over a wide range at large ones.

Workloads
---------
* `:plqr`       Poisson LQR, `Qc` grouped by reward (`depends_on`), terminal
                factor pinned to the terminal cost, `condition_terminal = true`
                (the default — the exact `log Z` normalizer and its probe).
* `:plqr_joint` Same model with `condition_terminal = false`. The difference
                between the two is the price of terminal conditioning.
* `:slqr`       Two-state switching model (LQR + `:free`), Poisson, `Qc` grouped
                by reward, `C`/`d` tied, terminal conditioning on.

Tiers (`n` plant dim → `2n` latent, `N` trials, median length, `p` channels)
-----
* `:tiny`      n=2,  N=24,   T≈20,  p=12   — smoke test / CI
* `:small`     n=4,  N=96,   T≈50,  p=40   — laptop, seconds per iteration
* `:medium`    n=8,  N=300,  T≈80,  p=100
* `:smoulder`  n=12, N=1000, T≈100, p=150  — the real problem size
=============================================================================#

using StateSpaceDynamics
using StateSpaceDynamics: SLDS, LinearDynamicalSystem, PoissonObservationModel
using LinearAlgebra
using Random
using StableRNGs
using Distributions: Poisson

const TIERS = Dict(
    :tiny => (n=2, ntrials=24, tmedian=20, tmin=8, tmax=45, obs_dim=12),
    :small => (n=4, ntrials=96, tmedian=50, tmin=20, tmax=110, obs_dim=40),
    :medium => (n=8, ntrials=300, tmedian=80, tmin=30, tmax=180, obs_dim=100),
    :smoulder => (n=12, ntrials=1000, tmedian=100, tmin=40, tmax=250, obs_dim=150),
)

const WORKLOADS = (:plqr, :plqr_joint, :slqr)

"""
    ragged_lengths(rng, N; tmedian, tmin, tmax, spread=0.35) -> Vector{Int}

Log-normal trial lengths around `tmedian`, clamped to `[tmin, tmax]`. With
`spread = 0.35` about two thirds of trials fall within ±40% of the median and
the right tail reaches ~2× — event-aligned windows with a long tail. Pass
`spread = 0` for equal-length trials (the non-ragged control).
"""
function ragged_lengths(rng, N; tmedian, tmin, tmax, spread=0.35)
    spread == 0 && return fill(tmedian, N)
    return [clamp(round(Int, tmedian * exp(spread * randn(rng))), tmin, tmax) for _ in 1:N]
end

# --- structural pieces (copied from docs/dev/lqr/model.jl to stay self-contained)

function _plant(n::Int)
    A = Matrix(0.96I, n, n)
    for i in 1:(n - 1)
        A[i, i + 1] = 0.05
        A[i + 1, i] = -0.04
    end
    S = Matrix(0.05I, n, n) + fill(0.01, n, n) - Diagonal(fill(0.01, n))
    return A, S
end

function _ring_map(n::Int, nref::Int; radius::Float64=1.5)
    G = zeros(n, nref)
    for j in 1:nref
        θ = 2π * (j - 1) / nref
        G[1, j] = radius * cos(θ)
        G[2, j] = radius * sin(θ)
    end
    return G
end

function _mixed_noise(n::Int; state::Float64=0.02, costate::Float64=1e-4)
    Σ = zeros(2n, 2n)
    Σ[1:n, 1:n] .= Matrix(state * I, n, n)
    Σ[(n + 1):(2n), (n + 1):(2n)] .= Matrix(costate * I, n, n)
    return Σ
end

function _reward_costs(n::Int, nrewards::Int)
    base = Matrix(0.20I, n, n)
    for i in 1:(n - 1)
        base[i, i + 1] = base[i + 1, i] = 0.025
    end
    scales = collect(range(0.6, 1.4; length=nrewards))
    return [[scales[r] .* base, scales[r] .* Matrix(3.0I, n, n)] for r in 1:nrewards]
end

function _free_drift(n::Int; decay=0.90, costate_decay=0.6, noise=0.15)
    M = zeros(2n, 2n)
    M[1:n, 1:n] .= Matrix(decay * I, n, n)
    M[(n + 1):(2n), (n + 1):(2n)] .= Matrix(costate_decay * I, n, n)
    return M, Matrix(noise * I, 2n, 2n)
end

function _poisson_counts(rng, C, d, z)
    η = C * z .+ d
    y = Matrix{Float64}(undef, size(η))
    for j in eachindex(η)
        y[j] = rand(rng, Poisson(exp(clamp(η[j], -12.0, 8.0))))
    end
    return y
end

"""
The generating LQR state model: one running cost everywhere, the terminal
factor pinned to cost 2 (`terminal_regime = 2`), which is what a ragged dataset
needs — `schedule[T_i]` would otherwise put the terminal cost on exactly one
trial length.
"""
function _lqr_state(
    n,
    tmax,
    ntargets;
    A,
    S,
    Qc,
    Σ,
    Gref,
    condition_terminal=true,
    fit_flags=LQRFitFlags(),
    mstep_iters=100,
)
    return LQRStateModel(
        A,
        S,
        Qc,
        Σ;
        schedule=fill(1, tmax),
        terminal=true,
        terminal_regime=2,
        condition_terminal=condition_terminal,
        Σf=Matrix(0.02I, n, n),
        P0=Matrix(0.2I, 2n, 2n),
        Bu=zeros(2n, ntargets),
        Gref=Gref,
        observe_costate=false,
        fit_flags=fit_flags,
        mstep_iters=mstep_iters,
    )
end

"""Emission initialization: count-PCA as the smoulder suite does (deterministic)."""
function _count_pca(ys, n::Int)
    p = size(first(ys), 1)
    total = sum(size(y, 2) for y in ys)
    μ = zeros(p)
    for y in ys
        μ .+= vec(sum(y; dims=2))
    end
    μ ./= total
    d = log.(max.(μ, 0.05))
    S = zeros(p, p)
    for y in ys
        z = sqrt.(y .+ 3 / 8)
        z .-= vec(sum(z; dims=2)) ./ size(z, 2)
        mul!(S, z, z', 1.0, 1.0)
    end
    E = eigen(Symmetric(S))
    ord = sortperm(E.values; rev=true)[1:n]
    return 0.25 .* E.vectors[:, ord], d
end

"""
    build_workload(name; tier=:small, seed=1, spread=0.35, nrewards=3,
                   ntargets=8, mstep_iters=100) -> NamedTuple

Returns `(; name, tier, model, y, ux, fit_kwargs, lengths, meta)`. `model` is a
fresh fit-ready model; `deepcopy` it before every timed `fit!`, since `fit!`
mutates. `fit_kwargs` are the keyword arguments the workload is meant to be fitted
with (minus `max_iter`, which the caller controls).
"""
function build_workload(
    name::Symbol;
    tier::Symbol=:small,
    seed::Int=1,
    spread=0.35,
    nrewards::Int=3,
    ntargets::Int=8,
    mstep_iters::Int=100,
    ntrials::Union{Nothing,Int}=nothing,
)
    name in WORKLOADS || throw(ArgumentError("unknown workload $name; one of $WORKLOADS"))
    cfg = TIERS[tier]
    n, p = cfg.n, cfg.obs_dim
    N = something(ntrials, cfg.ntrials)
    rng = StableRNG(seed)
    lengths = ragged_lengths(
        rng, N; tmedian=cfg.tmedian, tmin=cfg.tmin, tmax=cfg.tmax, spread=spread
    )
    tmax = maximum(lengths)

    rewards = [mod1(i, nrewards) for i in 1:N]
    targets = [mod1(i, ntargets) for i in 1:N]
    ux = [
        begin
            u = zeros(ntargets, lengths[i])
            u[targets[i], :] .= 1
            u
        end for i in 1:N
    ]

    # --- truth + data
    A, S = _plant(n)
    costs = _reward_costs(n, nrewards)
    truth = _lqr_state(
        n,
        tmax,
        ntargets;
        A=A,
        S=S,
        Qc=deepcopy(costs[1]),
        Σ=_mixed_noise(n),
        Gref=_ring_map(n, ntargets),
    )
    set_depends_on!(truth, (Qc=rewards,))
    for r in 1:nrewards
        v = group_variant(truth, :Qc, r)
        for k in eachindex(v.Qc)
            v.Qc[k] .= costs[r][k]
        end
        refresh!(v)
    end
    refresh!(truth)
    C = zeros(p, 2n)
    C[:, 1:n] .= 0.35 .* randn(rng, p, n) ./ sqrt(n)
    d = log.(exp.(range(log(0.15), log(1.2); length=p)))

    ys = Vector{Matrix{Float64}}(undef, N)
    Mfree, Qfree = _free_drift(n; decay=0.96, costate_decay=0.7, noise=0.08)
    Lfree = cholesky(Symmetric(Qfree)).L
    for i in 1:N
        v = group_variant(truth, :Qc, rewards[i])
        if name === :slqr
            # free (delay) epoch, then the optimal reach to the end of the trial
            Ti = lengths[i]
            onset = clamp(round(Int, Ti * (0.35 + 0.20 * mod(i, 7) / 6)), 3, Ti - 2)
            z = zeros(2n, Ti)
            z[:, 1] .= 0.2 .* randn(rng, 2n)
            for t in 2:onset
                z[:, t] .= Mfree * z[:, t - 1] .+ Lfree * randn(rng, 2n)
            end
            len = Ti - onset + 1
            sched = fill(1, len)
            seg = LQRStateModel(
                copy(v.A),
                copy(v.S),
                [copy(Q) for Q in v.Qc],
                copy(v.Σ);
                schedule=sched,
                terminal=true,
                terminal_regime=2,
                Σf=copy(v.Σf),
                P0=copy(v.P0),
                h=copy(v.h),
                Bu=copy(v.Bu),
                Gref=copy(v.Gref),
                observe_costate=false,
            )
            z[:, onset:Ti] .= simulate_lqr(
                rng,
                seg,
                len;
                x1=Vector(z[1:n, onset]),
                costate_slack=0.05,
                process_noise=true,
                ux=ux[i][:, onset:Ti],
            )
            ys[i] = _poisson_counts(rng, C, d, z)
        else
            z = simulate_lqr(
                rng, v, lengths[i]; costate_slack=0.05, process_noise=true, ux=ux[i]
            )
            ys[i] = _poisson_counts(rng, C, d, z)
        end
    end

    # --- fit-ready model (smoulder_fit_model defaults: PCA emission, generic start)
    q0 = 0.2
    flags = LQRFitFlags(; A=true, S=true, h=false, Bu=false, Gref=true)
    sm = _lqr_state(
        n,
        tmax,
        ntargets;
        A=Matrix(0.92I, n, n),
        S=Matrix(0.05I, n, n),
        Qc=[Matrix(q0 * I, n, n), Matrix(3q0 * I, n, n)],
        Σ=_mixed_noise(n; state=0.05, costate=name === :slqr ? 2e-2 : 1e-4),
        Gref=0.35 .* _ring_map(n, ntargets),
        condition_terminal=name !== :plqr_joint,
        fit_flags=flags,
        mstep_iters=mstep_iters,
    )
    set_depends_on!(sm, (Qc=rewards,))
    for r in 1:nrewards
        v = group_variant(sm, :Qc, r)
        v.Qc[1] .= Matrix(q0 * I, n, n)
        v.Qc[2] .= Matrix(3q0 * I, n, n)
        refresh!(v)
    end
    Cs, d0 = _count_pca(ys, n)
    C0 = zeros(p, 2n)
    C0[:, 1:n] .= Cs

    model, fit_kwargs = if name === :slqr
        M, Q = _free_drift(n; decay=0.88, costate_decay=0.8, noise=0.12)
        free = free_state_model(
            M,
            Q;
            P0=Matrix(0.2I, 2n, 2n),
            Bu=zeros(2n, ntargets),
            observe_costate=false,
            fit_flags=LQRFitFlags(; h=false, Bu=false),
        )
        set_depends_on!(free, (Qc=rewards,))
        l1 = LinearDynamicalSystem(sm, PoissonObservationModel(copy(C0), copy(d0)))
        l2 = LinearDynamicalSystem(free, PoissonObservationModel(copy(C0), copy(d0)))
        slds = SLDS(; A=[0.95 0.05; 0.05 0.95], πₖ=[0.1, 0.9], LDSs=[l1, l2])
        slds,
        (; ux=ux, smoothing_iters=2, tied_params=(:C, :d), progress=false, tol_kw=nothing)
    else
        lds = LinearDynamicalSystem(sm, PoissonObservationModel(C0, d0))
        lds, (; ux=ux, progress=false, tol=-1.0)
    end
    fit_kwargs = Base.structdiff(fit_kwargs, NamedTuple{(:tol_kw,)})

    meta = (;
        n,
        latent_dim=2n,
        ntrials=N,
        obs_dim=p,
        nrewards,
        ntargets,
        tmin=minimum(lengths),
        tmedian=cfg.tmedian,
        tmax,
        spread,
        ndistinct_lengths=length(unique(lengths)),
        ndistinct_designs=length(unique(zip(lengths, targets))),
        total_bins=sum(lengths),
        mstep_iters,
    )
    return (; name, tier, model, y=ys, ux, fit_kwargs, lengths, meta)
end

"""
    run_fit!(model, w; max_iter, rng_seed=7)

Fit `model` on workload `w` for exactly `max_iter` EM iterations (convergence
disabled) and return the ELBO trace. The SLDS fit is stochastic in its
initialization/sampling only through `rng`, which is seeded here.
"""
function run_fit!(model, w; max_iter::Int, rng_seed::Int=7)
    if model isa SLDS
        return fit!(model, w.y; max_iter=max_iter, rng=StableRNG(rng_seed), w.fit_kwargs...)
    else
        return fit!(model, w.y; max_iter=max_iter, w.fit_kwargs...)
    end
end
