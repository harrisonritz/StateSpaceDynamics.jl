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
