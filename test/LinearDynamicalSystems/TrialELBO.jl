#=============================================================================
`trial_elbos` — the ELBO split by trial.

The contract is a single equation, and it is what every test here checks in one
form or another:

    sum(trial_elbos(m, y)) + log p(θ) == elbo(m, y)

The interesting failure mode is a *constant*. The per-trial `Q_state!` kernel is
written as an M-step objective, so it drops the `-½ D log 2π` per timestep that
cannot move an argmax, while the aggregated one carries it. A regression there
is invisible in any single number and shows up only against `elbo` — which is
why the equality is asserted per family rather than the vector merely being
checked for the right length and finiteness.

Also pinned: the Gaussian case agrees with the exact `loglikelihood` (its
smoother is exact, so the bound is tight), ragged trial lengths and inputs are
carried through, the prior term is exactly the gap when priors are set, and a
grouped model is rejected rather than silently scored against one cell's
parameters.
=============================================================================#

const TE_D, TE_N, TE_T, TE_NTR = 3, 6, 20, 5

function _te_sm(; ux_dim=0)
    return GaussianStateModel(;
        A=0.9 * Matrix{Float64}(I, TE_D, TE_D),
        Q=0.05 * Matrix{Float64}(I, TE_D, TE_D),
        b=zeros(TE_D),
        x0=zeros(TE_D),
        P0=0.1 * Matrix{Float64}(I, TE_D, TE_D),
        B=zeros(TE_D, ux_dim),
    )
end

function _te_gom(seed; uy_dim=0)
    return GaussianObservationModel(;
        C=randn(StableRNG(seed), TE_N, TE_D),
        R=0.2 * Matrix{Float64}(I, TE_N, TE_N),
        d=zeros(TE_N),
        D=zeros(TE_N, uy_dim),
    )
end

function _te_pom(seed; uy_dim=0)
    return PoissonObservationModel(;
        C=0.4 .* randn(StableRNG(seed), TE_N, TE_D),
        d=fill(-1.0, TE_N),
        D=zeros(TE_N, uy_dim),
    )
end

_te_lds(seed) = LinearDynamicalSystem(_te_sm(), _te_gom(seed))
_te_plds(seed) = LinearDynamicalSystem(_te_sm(), _te_pom(seed))

function _te_slds(; K=2)
    return SLDS(;
        A=[0.9 0.1; 0.1 0.9],
        πₖ=[0.5, 0.5],
        LDSs=[LinearDynamicalSystem(_te_sm(), _te_pom(10 + k)) for k in 1:K],
    )
end

"""
    _te_sum_matches(model, y; prior=0.0, kwargs...)

Assert the split reproduces `elbo` once the parameter log-prior is added back,
and that the vector has one finite entry per trial.
"""
function _te_sum_matches(model, y, ntrials; prior=0.0, kwargs...)
    per_trial = trial_elbos(model, y; kwargs...)
    @test length(per_trial) == ntrials
    @test all(isfinite, per_trial)
    @test isapprox(sum(per_trial) + prior, elbo(model, y; kwargs...); rtol=1e-10)
    return per_trial
end

"""
    test_trial_elbos_sum_to_elbo()

The core identity, across every emission family: Gaussian, Poisson, a composite
with both, and the SLDS.
"""
function test_trial_elbos_sum_to_elbo()
    lds = _te_lds(1)
    _, yg = rand(StableRNG(1), lds, fill(TE_T, TE_NTR))
    per_trial = _te_sum_matches(lds, yg, TE_NTR)
    # The Gaussian smoother is exact, so the bound is tight.
    @test isapprox(sum(per_trial), loglikelihood(lds, yg); rtol=1e-10)

    plds = _te_plds(2)
    _, yp = rand(StableRNG(2), plds, fill(TE_T, TE_NTR))
    _te_sum_matches(plds, yp, TE_NTR)

    comp = LinearDynamicalSystem(_te_sm(), (spk=_te_pom(3), kin=_te_gom(4)))
    _, yc_raw = rand(StableRNG(3), comp, fill(TE_T, TE_NTR))
    yc = (spk=[t.spk for t in yc_raw], kin=[t.kin for t in yc_raw])
    _te_sum_matches(comp, yc, TE_NTR)

    slds = _te_slds()
    _, _, ys = rand(StableRNG(4), slds, fill(TE_T, TE_NTR))
    return _te_sum_matches(slds, ys, TE_NTR; smoothing_iters=150)
end

"""
    test_trial_elbos_ragged_and_inputs()

Ragged trial lengths and non-empty `ux` / `uy`. The per-timestep constant the
split has to add back scales with each trial's own length, so a ragged set is
what separates "added the constant" from "added `ntrials × T` of it".
"""
function test_trial_elbos_ragged_and_inputs()
    tsteps = [12, 20, 31, 8]
    ntr = length(tsteps)

    lds = LinearDynamicalSystem(_te_sm(; ux_dim=2), _te_gom(5; uy_dim=1))
    rng = StableRNG(5)
    ux = [randn(rng, 2, t) for t in tsteps]
    uy = [randn(rng, 1, t) for t in tsteps]
    _, yg = rand(rng, lds, tsteps; ux=ux, uy=uy)
    _te_sum_matches(lds, yg, ntr; ux=ux, uy=uy)

    plds = LinearDynamicalSystem(_te_sm(; ux_dim=2), _te_pom(6; uy_dim=1))
    _, yp = rand(StableRNG(6), plds, tsteps; ux=ux, uy=uy)
    return _te_sum_matches(plds, yp, ntr; ux=ux, uy=uy)
end

"""
    test_trial_elbos_single_trial()

A single-trial matrix `y` still returns a one-element `Vector`, not a scalar —
the field exists to be indexed by trial, and an unwrapped scalar would be
indistinguishable from `elbo`.
"""
function test_trial_elbos_single_trial()
    lds = _te_lds(7)
    _, y = rand(StableRNG(7), lds, [TE_T])
    per_trial = trial_elbos(lds, y[1])
    @test per_trial isa Vector{Float64}
    @test length(per_trial) == 1
    @test isapprox(per_trial[1], elbo(lds, y[1]); rtol=1e-10)

    slds = _te_slds()
    _, _, ys = rand(StableRNG(8), slds, [TE_T])
    post = smooth(slds, ys[1]; smoothing_iters=150)
    @test post.trial_elbo isa Vector{Float64}
    @test length(post.trial_elbo) == 1
    @test isapprox(sum(post.trial_elbo), post.elbo; rtol=1e-10)
end

"""
    test_trial_elbos_prior_excluded()

With priors set, the gap between the sum and `elbo` is exactly the parameter
log-prior — it is left out of the per-trial vector rather than smeared across
it, so a trial's number does not depend on how many trials came along with it.
"""
function test_trial_elbos_prior_excluded()
    lds = _te_lds(9)
    lds.state_model.Q_prior = IWPrior(; Ψ=Matrix(0.05 * I(TE_D)), ν=TE_D + 3.0)
    _, y = rand(StableRNG(9), lds, fill(TE_T, TE_NTR))

    prior = SSD._state_prior_logdensity(lds, SSD.SmoothWorkspace(Float64, TE_D, TE_N, TE_T))
    @test prior != 0
    _te_sum_matches(lds, y, TE_NTR; prior=prior)

    # Same posterior, so the per-trial terms are untouched by the prior.
    bare = _te_lds(9)
    @test isapprox(trial_elbos(lds, y), trial_elbos(bare, y); rtol=1e-10)
end

"""
    test_trial_elbos_rejects_grouping()

A grouped model fits one parameter set per cell of trials; the per-trial vector
is under the single set the model carries, so scoring one against the other
would be silently wrong. It errors instead.
"""
function test_trial_elbos_rejects_grouping()
    lds = _te_lds(11)
    _, y = rand(StableRNG(11), lds, fill(TE_T, TE_NTR))
    labels = [:a, :a, :b, :b, :b]
    lds.obs_model.depends_on = (C=labels, d=labels)
    @test_throws ErrorException trial_elbos(lds, y)
end

#=============================================================================
LQR latents.

The same contract, reached by a different route: the LQR state Q-term
has no per-trial kernel, so the split re-aggregates each trial's statistics and
calls the aggregated `Q_state!` on them. That is exact only because `Q_state!`
is additive over trials at fixed parameters — the residual scatter is linear in
the aggregated blocks, and `N` / `N_f` are plain counts. If that ever stops
holding, these tests are what catches it.

Data comes from `simulate_lqr` rather than `rand`: a symplectic transition has
reciprocal eigenvalue pairs, so the model's own forward roll diverges over any
useful horizon and would leave nothing but overflow to compare.
=============================================================================#

const TE_HN, TE_HT = 2, 18

function _te_lqr_sm(;
    ux_dim=0, terminal=false, K=1, tsteps=TE_HT, onset=1, observe_costate=false
)
    n = TE_HN
    return LQRStateModel(
        [1.0 0.1; -0.05 0.95],
        0.3 * Matrix{Float64}(I, n, n),
        [(0.5 + 0.4k) * Matrix{Float64}(I, n, n) for k in 1:K],
        0.05 * Matrix{Float64}(I, 2n, 2n);
        schedule=if K == 1
            Int[]
        else
            SSD.cost_schedule(tsteps; terminal=terminal, onset=onset, nregimes=K)
        end,
        terminal=terminal,
        Bu=zeros(2n, ux_dim),
        Gref=zeros(n, ux_dim),
        P0=0.2 * Matrix{Float64}(I, 2n, 2n),
        observe_costate=observe_costate,
    )
end

function _te_lqr_gom(seed; costate=false)
    return GaussianObservationModel(;
        C=if costate
            randn(StableRNG(seed), TE_N, 2TE_HN)
        else
            hcat(randn(StableRNG(seed), TE_N, TE_HN), zeros(TE_N, TE_HN))
        end,
        R=0.2 * Matrix{Float64}(I, TE_N, TE_N),
        d=zeros(TE_N),
        D=zeros(TE_N, 0),
    )
end

function _te_lqr_pom(seed)
    return PoissonObservationModel(;
        C=hcat(0.4 .* randn(StableRNG(seed), TE_N, TE_HN), zeros(TE_N, TE_HN)),
        d=fill(-0.5, TE_N),
        D=zeros(TE_N, 0),
    )
end

"""
    _te_lqr_data(rng, lds, lengths; ux=nothing) -> y

Latent paths on the stable manifold via [`simulate_lqr`](@ref), then one draw
from the emission per trial, shaped the way `Data` wants it (a composite's
per-trial `NamedTuple`s are transposed into a `NamedTuple` of vectors).
"""
function _te_lqr_data(rng, lds, lengths; ux=nothing)
    sm = lds.state_model
    params = SSD._extract_obs_params(lds.obs_model)
    ys = map(enumerate(lengths)) do (i, len)
        z = simulate_lqr(rng, sm, len; costate_slack=0.05, ux=ux === nothing ? nothing : ux[i])
        y = SSD._alloc_obs(lds, len)
        SSD._sample_lqr_obs!(
            rng,
            y,
            z,
            lds.obs_model,
            params,
            SSD._check_uy(nothing, lds.uy_dim, len, lds.obs_model),
        )
        y
    end
    eltype(ys) <: NamedTuple || return ys
    ks = keys(first(ys))
    return NamedTuple{ks}(([y[k] for y in ys] for k in ks))
end

"""
    test_trial_elbos_lqr()

The identity across the shapes an LQR model comes in: a Gaussian and a
Poisson emission at one cost regime, a scheduled multi-regime cost with the
terminal factor on, and a composite mixing both emissions. The Gaussian case
also pins the vector to the exact `loglikelihood`, whose smoother is exact on
`z = [x; λ]` — and which, with a terminal factor, is `log p(y, y_term = 0)`.
"""
function test_trial_elbos_lqr()
    lds = LinearDynamicalSystem(_te_lqr_sm(), _te_lqr_gom(21))
    y = _te_lqr_data(StableRNG(21), lds, fill(TE_HT, TE_NTR))
    per_trial = _te_sum_matches(lds, y, TE_NTR)
    @test isapprox(sum(per_trial), loglikelihood(lds, y); rtol=1e-10)

    plds = LinearDynamicalSystem(_te_lqr_sm(), _te_lqr_pom(22))
    _te_sum_matches(plds, _te_lqr_data(StableRNG(22), plds, fill(TE_HT, TE_NTR)), TE_NTR)

    # Three cost regimes on a schedule, with the terminal costate factor active:
    # the transition blocks are per regime and the terminal block is per trial,
    # so a trial's share of either going astray shows up here.
    sched = LinearDynamicalSystem(
        _te_lqr_sm(; terminal=true, K=3, onset=8), _te_lqr_pom(23)
    )
    _te_sum_matches(sched, _te_lqr_data(StableRNG(23), sched, fill(TE_HT, TE_NTR)), TE_NTR)

    comp = LinearDynamicalSystem(
        _te_lqr_sm(; terminal=true, K=2), (spk=_te_lqr_pom(24), kin=_te_lqr_gom(25))
    )
    return _te_sum_matches(
        comp, _te_lqr_data(StableRNG(24), comp, fill(TE_HT, TE_NTR)), TE_NTR
    )
end

"""
    test_trial_elbos_lqr_inputs()

Ragged trial lengths under a schedule that covers the longest, a costate the
emission is allowed to read, and a tracking reference driven by `ux` — the three
places a per-trial split could pick up the wrong trial's timesteps, regime or
input column.
"""
function test_trial_elbos_lqr_inputs()
    lengths = [TE_HT, TE_HT - 5, TE_HT - 2]

    ragged = LinearDynamicalSystem(_te_lqr_sm(; K=2, onset=9), _te_lqr_pom(26))
    _te_sum_matches(ragged, _te_lqr_data(StableRNG(26), ragged, lengths), length(lengths))

    costate = LinearDynamicalSystem(
        _te_lqr_sm(; observe_costate=true), _te_lqr_gom(27; costate=true)
    )
    @test !iszero(view(costate.obs_model.C, :, (TE_HN + 1):(2TE_HN)))
    _te_sum_matches(costate, _te_lqr_data(StableRNG(27), costate, lengths), length(lengths))

    # Tracking: `Gref` maps the input to the reference the cost is measured
    # against, so it enters both the per-regime transition and the terminal
    # residual. Both are per trial through `ux`.
    sm = _te_lqr_sm(; ux_dim=TE_HN, terminal=true, K=3, onset=8)
    sm.Gref .= Matrix{Float64}(I, TE_HN, TE_HN)
    refresh!(sm)
    track = LinearDynamicalSystem(sm, _te_lqr_gom(28))
    ux = [randn(StableRNG(100 + i), TE_HN, len) for (i, len) in enumerate(lengths)]
    y = _te_lqr_data(StableRNG(28), track, lengths; ux=ux)
    return _te_sum_matches(track, y, length(lengths); ux=ux)
end

"""
    test_trial_elbos_lqr_rejects_grouping()

As for every other state model: a grouped LQR model carries one
structural estimate per cell, and scoring trials under a single set would be
silently wrong.
"""
function test_trial_elbos_lqr_rejects_grouping()
    lds = LinearDynamicalSystem(_te_lqr_sm(), _te_lqr_gom(29))
    y = _te_lqr_data(StableRNG(29), lds, fill(TE_HT, TE_NTR))
    labels = [:a, :a, :b, :b, :b]
    set_depends_on!(lds.state_model, (structure=labels,))
    @test_throws ErrorException trial_elbos(lds, y)
end
