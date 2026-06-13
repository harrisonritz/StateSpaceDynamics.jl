# Tests for the stitched (per-session) observation models: Gaussian, Poisson,
# and switching (SLDS). A stitched model shares the latent state model across
# groups while each group carries its own emission (possibly different obs_dim).

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

# Stable shared Gaussian state model.
function _stitched_state_model(rng, D)
    A = 0.8 * Matrix{Float64}(I, D, D) .+ 0.05 .* randn(rng, D, D)
    Q = Matrix{Float64}(0.05 * I(D))
    b = 0.05 .* randn(rng, D)
    x0 = zeros(Float64, D)
    P0 = Matrix{Float64}(0.1 * I(D))
    return GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
end

# Stitched Gaussian LDS with `length(ps)` groups of emission dims `ps`.
function _make_stitched_gaussian(rng, D, ps; group_ids=collect(1:length(ps)))
    sm = _stitched_state_model(rng, D)
    models = [
        GaussianObservationModel(;
            C=0.5 .* randn(rng, p, D),
            R=Matrix{Float64}(0.2 * I(p)),
            d=0.1 .* randn(rng, p),
        ) for p in ps
    ]
    om = GaussianObservationModelStitched(models, group_ids)
    return LinearDynamicalSystem(sm, om)
end

function _make_stitched_poisson(rng, D, ps; group_ids=collect(1:length(ps)))
    sm = _stitched_state_model(rng, D)
    models = [
        PoissonObservationModel(; C=0.3 .* randn(rng, p, D), d=-0.5 .+ 0.1 .* randn(rng, p))
        for p in ps
    ]
    om = PoissonObservationModelStitched(models, group_ids)
    return LinearDynamicalSystem(sm, om)
end

# Stitched SLDS: K discrete states, each with a stitched emission over `ps`.
function _make_stitched_slds_component(rng, D, ps, obs)
    sm = _stitched_state_model(rng, D)
    if obs === :gaussian
        models = [
            GaussianObservationModel(;
                C=0.5 .* randn(rng, p, D),
                R=Matrix{Float64}(0.2 * I(p)),
                d=0.1 .* randn(rng, p),
            ) for p in ps
        ]
        om = GaussianObservationModelStitched(models)
    else
        models = [
            PoissonObservationModel(;
                C=0.3 .* randn(rng, p, D), d=-0.5 .+ 0.1 .* randn(rng, p)
            ) for p in ps
        ]
        om = PoissonObservationModelStitched(models)
    end
    return LinearDynamicalSystem(sm, om)
end

function _make_stitched_slds(rng, D, ps, K; obs=:gaussian)
    ldss = [_make_stitched_slds_component(rng, D, ps, obs) for _ in 1:K]
    A = [0.9 0.1; 0.1 0.9]
    πₖ = [0.5, 0.5]
    return SLDS(; A=A, πₖ=πₖ, LDSs=ldss)
end

# ---------------------------------------------------------------------------
# Constructors & traits
# ---------------------------------------------------------------------------

function test_stitched_constructors()
    rng = StableRNG(1)
    D = 2
    ps = [3, 4]

    # Default integer group ids.
    models = [
        GaussianObservationModel(;
            C=randn(rng, p, D), R=Matrix{Float64}(I(p)), d=zeros(p)
        ) for p in ps
    ]
    om = GaussianObservationModelStitched(models)
    @test om.group_ids == [1, 2]
    @test length(om.models) == 2

    # Custom string group ids.
    om2 = GaussianObservationModelStitched(models, ["sessA", "sessB"])
    @test om2.group_ids == ["sessA", "sessB"]

    lds = LinearDynamicalSystem(_stitched_state_model(rng, D), om)
    @test lds.latent_dim == D
    @test lds.obs_dim == maximum(ps)          # max channel count
    @test length(lds.fit_bool) == 6           # Gaussian layout

    # Poisson stitched: fit_bool length 5.
    pmodels = [PoissonObservationModel(; C=randn(rng, p, D), d=zeros(p)) for p in ps]
    pom = PoissonObservationModelStitched(pmodels)
    plds = LinearDynamicalSystem(_stitched_state_model(rng, D), pom)
    @test length(plds.fit_bool) == 5
    @test StateSpaceDynamics._is_poisson_like(pom)
    @test !StateSpaceDynamics._is_poisson_like(om)

    # Constructor argument validation.
    @test_throws StateSpaceDynamics.DimensionMismatchError GaussianObservationModelStitched(
        models, [1, 2, 3]
    )
    @test_throws ArgumentError GaussianObservationModelStitched(models, [1, 1])
    @test_throws StateSpaceDynamics.DimensionMismatchError PoissonObservationModelStitched(
        pmodels, [1]
    )
    @test_throws ArgumentError PoissonObservationModelStitched(pmodels, ["a", "a"])
    return nothing
end

function test_stitched_validation()
    rng = StableRNG(2)
    D = 2
    lds = _make_stitched_gaussian(rng, D, [3, 4])
    @test validate_LDS(lds) === nothing

    # Duplicate group ids (mutate post-construction to bypass the ctor guard).
    bad = _make_stitched_gaussian(rng, D, [3, 4])
    bad.obs_model.group_ids = [1, 1]
    @test_throws ArgumentError validate_LDS(bad)

    # Mismatched latent dim in one group's C.
    bad_models = [
        GaussianObservationModel(; C=randn(rng, 3, D + 1), R=Matrix{Float64}(I(3)), d=zeros(3)),
    ]
    bad_om = GaussianObservationModelStitched(bad_models)
    @test_throws StateSpaceDynamics.DimensionMismatchError StateSpaceDynamics._validate_stitched_groups(
        bad_om, D
    )

    # Empty stitched model.
    empty_om = GaussianObservationModelStitched(
        GaussianObservationModel{Float64,Matrix{Float64},Vector{Float64}}[], Int[]
    )
    @test_throws ArgumentError StateSpaceDynamics._validate_stitched_groups(empty_om, D)
    return nothing
end

function test_stitched_show()
    rng = StableRNG(3)
    glds = _make_stitched_gaussian(rng, 2, [3, 4])
    plds = _make_stitched_poisson(rng, 2, [3, 4])
    @test occursin("Stitched Gaussian", sprint(show, glds.obs_model))
    @test occursin("Stitched Poisson", sprint(show, plds.obs_model))
    @test occursin("Linear Dynamical System", sprint(show, glds))
    return nothing
end

function test_stitched_resolve_obs_group()
    rng = StableRNG(4)
    om = _make_stitched_gaussian(rng, 2, [3, 4]; group_ids=["a", "b"]).obs_model
    @test StateSpaceDynamics._resolve_obs_group(om, ["b", "a", "b"]) == [2, 1, 2]
    @test_throws ArgumentError StateSpaceDynamics._resolve_obs_group(om, ["a", "z"])
    return nothing
end

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

function test_stitched_gaussian_rand()
    rng = StableRNG(10)
    D, ps = 2, [3, 4]
    lds = _make_stitched_gaussian(rng, D, ps)
    obs_group = [1, 2, 1, 2]
    Ts = [20, 25, 22, 18]
    x, y = StateSpaceDynamics.rand(StableRNG(11), lds, Ts; obs_group=obs_group)
    @test length(x) == 4 && length(y) == 4
    for i in eachindex(y)
        @test size(x[i]) == (D, Ts[i])
        @test size(y[i]) == (ps[obs_group[i]], Ts[i])
    end

    # obs_group length mismatch.
    @test_throws StateSpaceDynamics.DimensionMismatchError StateSpaceDynamics.rand(
        lds, [10, 10]; obs_group=[1]
    )
    return nothing
end

function test_stitched_poisson_rand()
    rng = StableRNG(12)
    D, ps = 2, [3, 4]
    lds = _make_stitched_poisson(rng, D, ps)
    obs_group = [2, 1]
    x, y = StateSpaceDynamics.rand(StableRNG(13), lds, [15, 15]; obs_group=obs_group)
    for i in eachindex(y)
        @test size(y[i]) == (ps[obs_group[i]], 15)
        @test all(y[i] .>= 0)
        @test all(y[i] .== round.(y[i]))     # integer counts
    end
    return nothing
end

function test_stitched_no_controls_error()
    rng = StableRNG(14)
    D = 2
    sm = GaussianStateModel(;
        A=0.8 * Matrix{Float64}(I, D, D),
        Q=Matrix{Float64}(0.05 * I(D)),
        b=zeros(D),
        x0=zeros(D),
        P0=Matrix{Float64}(0.1 * I(D)),
        B=randn(rng, D, 1),                     # nonzero dynamics input
    )
    models = [GaussianObservationModel(; C=randn(rng, 3, D), R=Matrix{Float64}(I(3)), d=zeros(3))]
    lds = LinearDynamicalSystem(sm, GaussianObservationModelStitched(models))
    @test_throws ArgumentError StateSpaceDynamics.rand(lds, [10]; obs_group=[1])
    return nothing
end

# ---------------------------------------------------------------------------
# Gaussian fit / smooth / loglikelihood
# ---------------------------------------------------------------------------

# Strong correctness check: with a single group the stitched path must
# reproduce the plain multi-trial Gaussian EM (identical algorithm).
function test_stitched_gaussian_reduces_to_plain()
    rng = StableRNG(20)
    D, p = 2, 3
    sm = _stitched_state_model(rng, D)
    om = GaussianObservationModel(;
        C=0.5 .* randn(rng, p, D), R=Matrix{Float64}(0.2 * I(p)), d=0.1 .* randn(rng, p)
    )
    true_lds = LinearDynamicalSystem(sm, om)
    ntrials = 4
    Ts = fill(30, ntrials)
    _, y = StateSpaceDynamics.rand(StableRNG(21), true_lds, Ts)

    # Two independent init models with identical parameters.
    init_state() = GaussianStateModel(;
        A=0.7 * Matrix{Float64}(I, D, D),
        Q=Matrix{Float64}(0.1 * I(D)),
        b=zeros(D),
        x0=zeros(D),
        P0=Matrix{Float64}(0.2 * I(D)),
    )
    init_obs() = GaussianObservationModel(;
        C=0.2 .* Matrix{Float64}(I, p, D) .+ 0.0, R=Matrix{Float64}(0.5 * I(p)), d=zeros(p)
    )

    plain = LinearDynamicalSystem(init_state(), init_obs())
    stitched = LinearDynamicalSystem(
        init_state(), GaussianObservationModelStitched([init_obs()])
    )

    elbo_plain = fit!(plain, y; max_iter=30, progress=false)
    elbo_stitch = fit!(
        stitched, y; obs_group=ones(Int, ntrials), max_iter=30, progress=false
    )

    # Same algorithm for G=1 ⇒ matching converged ELBO and recovered params.
    @test isapprox(elbo_plain[end], elbo_stitch[end]; rtol=1e-4, atol=1e-4)
    @test isapprox(plain.state_model.A, stitched.state_model.A; rtol=1e-3, atol=1e-3)
    @test isapprox(plain.state_model.Q, stitched.state_model.Q; rtol=1e-3, atol=1e-3)
    @test isapprox(
        plain.obs_model.C, stitched.obs_model.models[1].C; rtol=1e-3, atol=1e-3
    )
    @test isapprox(
        plain.obs_model.R, stitched.obs_model.models[1].R; rtol=1e-3, atol=1e-3
    )
    return nothing
end

function test_stitched_gaussian_fit_multigroup()
    rng = StableRNG(22)
    D, ps = 2, [3, 4]
    true_lds = _make_stitched_gaussian(rng, D, ps)
    ntrials = 8
    obs_group = repeat([1, 2]; outer=ntrials ÷ 2)
    Ts = fill(40, ntrials)
    _, y = StateSpaceDynamics.rand(StableRNG(23), true_lds, Ts; obs_group=obs_group)

    fit_lds = _make_stitched_gaussian(StableRNG(99), D, ps)
    elbos = fit!(fit_lds, y; obs_group=obs_group, max_iter=40, progress=false)
    @test all(isfinite, elbos)
    # ELBO should improve overall (EM lower-bound increases).
    @test elbos[end] > elbos[1]

    # Emission dims preserved per group.
    @test size(fit_lds.obs_model.models[1].C) == (ps[1], D)
    @test size(fit_lds.obs_model.models[2].C) == (ps[2], D)

    # smooth + loglikelihood entry points.
    xs, Ps = StateSpaceDynamics.smooth(fit_lds, y; obs_group=obs_group)
    for i in eachindex(y)
        @test size(xs[i]) == (D, Ts[i])
        @test size(Ps[i]) == (D, D, Ts[i])
    end
    ll = StateSpaceDynamics.loglikelihood(fit_lds, y; obs_group=obs_group)
    @test isfinite(ll)

    # Channel-count mismatch is rejected.
    bad_y = [randn(ps[2] + 1, Ts[1]) for _ in 1:ntrials]
    @test_throws StateSpaceDynamics.DimensionMismatchError fit!(
        fit_lds, bad_y; obs_group=obs_group, max_iter=1, progress=false
    )
    return nothing
end

function test_stitched_gaussian_ragged_and_strings()
    rng = StableRNG(24)
    D, ps = 2, [3, 4]
    true_lds = _make_stitched_gaussian(rng, D, ps; group_ids=["x", "y"])
    obs_group = ["x", "y", "y", "x", "y"]
    Ts = [20, 25, 18, 30, 22]               # ragged lengths
    _, y = StateSpaceDynamics.rand(StableRNG(25), true_lds, Ts; obs_group=obs_group)

    fit_lds = _make_stitched_gaussian(StableRNG(98), D, ps; group_ids=["x", "y"])
    elbos = fit!(fit_lds, y; obs_group=obs_group, max_iter=25, progress=false)
    @test all(isfinite, elbos)
    @test elbos[end] > elbos[1]
    return nothing
end

# ---------------------------------------------------------------------------
# Poisson fit / smooth
# ---------------------------------------------------------------------------

function test_stitched_poisson_fit_multigroup()
    rng = StableRNG(30)
    D, ps = 2, [3, 4]
    true_lds = _make_stitched_poisson(rng, D, ps)
    ntrials = 6
    obs_group = repeat([1, 2]; outer=ntrials ÷ 2)
    Ts = fill(30, ntrials)
    _, y = StateSpaceDynamics.rand(StableRNG(31), true_lds, Ts; obs_group=obs_group)

    fit_lds = _make_stitched_poisson(StableRNG(97), D, ps)
    elbos = fit!(
        fit_lds, y; obs_group=obs_group, max_iter=15, progress=false, newton_max_iter=10
    )
    @test all(isfinite, elbos)
    @test elbos[end] > elbos[1]
    @test size(fit_lds.obs_model.models[1].C) == (ps[1], D)

    xs, Ps = StateSpaceDynamics.smooth(fit_lds, y; obs_group=obs_group)
    for i in eachindex(y)
        @test size(xs[i]) == (D, Ts[i])
        @test size(Ps[i]) == (D, D, Ts[i])
    end
    return nothing
end

# ---------------------------------------------------------------------------
# SLDS (switching) stitched
# ---------------------------------------------------------------------------

function test_stitched_slds_rand()
    rng = StableRNG(40)
    D, ps, K = 2, [3, 4], 2
    slds = _make_stitched_slds(rng, D, ps, K)
    obs_group = [1, 2, 1]
    Ts = [20, 22, 18]
    z, x, y = StateSpaceDynamics.rand(StableRNG(41), slds, Ts; obs_group=obs_group)
    for i in eachindex(y)
        @test length(z[i]) == Ts[i]
        @test size(x[i]) == (D, Ts[i])
        @test size(y[i]) == (ps[obs_group[i]], Ts[i])
        @test all(1 .<= z[i] .<= K)
    end
    return nothing
end

function test_stitched_slds_fit_gaussian()
    rng = StableRNG(42)
    D, ps, K = 2, [3, 4], 2
    true_slds = _make_stitched_slds(rng, D, ps, K)
    ntrials = 4
    obs_group = [1, 2, 1, 2]
    Ts = fill(30, ntrials)
    _, _, y = StateSpaceDynamics.rand(StableRNG(43), true_slds, Ts; obs_group=obs_group)

    fit_slds = _make_stitched_slds(StableRNG(96), D, ps, K)
    elbos = fit!(fit_slds, y; obs_group=obs_group, max_iter=8, progress=false)
    @test length(elbos) == 8
    @test all(isfinite, elbos)
    # Emission dims preserved per (state, group).
    @test size(fit_slds.LDSs[1].obs_model.models[2].C) == (ps[2], D)
    # Transition matrix stays row-stochastic.
    for i in 1:K
        @test isapprox(sum(fit_slds.A[i, :]), 1.0; atol=1e-8)
    end
    return nothing
end

function test_stitched_slds_fit_poisson()
    rng = StableRNG(44)
    D, ps, K = 2, [3, 3], 2
    true_slds = _make_stitched_slds(rng, D, ps, K; obs=:poisson)
    ntrials = 4
    obs_group = [1, 2, 1, 2]
    Ts = fill(25, ntrials)
    _, _, y = StateSpaceDynamics.rand(StableRNG(45), true_slds, Ts; obs_group=obs_group)

    fit_slds = _make_stitched_slds(StableRNG(95), D, ps, K; obs=:poisson)
    elbos = fit!(fit_slds, y; obs_group=obs_group, max_iter=5, progress=false)
    @test length(elbos) == 5
    @test all(isfinite, elbos)
    return nothing
end
