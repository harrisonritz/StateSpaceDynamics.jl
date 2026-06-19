# Tests for the generalized trial-varying parameter models: any of the six
# estimation blocks (x0, P0, [A b], Q, [C d], R) may be fit per trial group, with
# different blocks keyed on different labels supplied via labels::Dict.

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

function _tv_stable_A(rng, D)
    return 0.8 * Matrix{Float64}(I, D, D) .+ 0.05 .* randn(rng, D, D)
end

# All-invariant trial-varying Gaussian model with given plain parameters.
function _tv_invariant_gaussian(A, b, Q, x0, P0, C, d, R)
    sm = TrialVaryingGaussianStateModel(; A=A, b=b, Q=Q, x0=x0, P0=P0)
    om = TrialVaryingGaussianObservationModel(; C=C, d=d, R=R)
    return LinearDynamicalSystem(sm, om)
end

# Trial-varying Gaussian: Q varies by :cond (2 groups), [C d] vary by :session
# (2 groups), everything else invariant.
function _make_tv_gaussian(rng, D, p)
    A = _tv_stable_A(rng, D)
    b = zeros(D)
    x0 = zeros(D)
    P0 = Matrix{Float64}(0.1 * I(D))
    Q1 = Matrix{Float64}(0.05 * I(D))
    Q2 = Matrix{Float64}(0.15 * I(D))
    C1 = 0.5 .* randn(rng, p, D)
    C2 = 0.5 .* randn(rng, p, D)
    d1 = 0.1 .* randn(rng, p)
    d2 = 0.1 .* randn(rng, p)
    R = Matrix{Float64}(0.2 * I(p))

    sm = TrialVaryingGaussianStateModel(;
        A=A, b=b, x0=x0, P0=P0, Q=GroupedParam([Q1, Q2], :cond, [1, 2])
    )
    om = TrialVaryingGaussianObservationModel(;
        C=GroupedParam([C1, C2], :session, ["s1", "s2"]),
        d=GroupedParam([d1, d2], :session, ["s1", "s2"]),
        R=R,
    )
    return LinearDynamicalSystem(sm, om)
end

function _make_tv_poisson(rng, D, p)
    A = _tv_stable_A(rng, D)
    b = zeros(D)
    x0 = zeros(D)
    P0 = Matrix{Float64}(0.1 * I(D))
    Q1 = Matrix{Float64}(0.05 * I(D))
    Q2 = Matrix{Float64}(0.1 * I(D))
    C1 = 0.3 .* randn(rng, p, D)
    C2 = 0.3 .* randn(rng, p, D)
    d1 = -0.5 .+ 0.1 .* randn(rng, p)
    d2 = -0.5 .+ 0.1 .* randn(rng, p)

    sm = TrialVaryingGaussianStateModel(;
        A=A, b=b, x0=x0, P0=P0, Q=GroupedParam([Q1, Q2], :cond, [1, 2])
    )
    om = TrialVaryingPoissonObservationModel(;
        C=GroupedParam([C1, C2], :session, ["s1", "s2"]),
        d=GroupedParam([d1, d2], :session, ["s1", "s2"]),
    )
    return LinearDynamicalSystem(sm, om)
end

# ---------------------------------------------------------------------------
# GroupedParam + constructors + validation
# ---------------------------------------------------------------------------

function test_tv_grouped_param()
    Q1 = Matrix{Float64}(I(2))
    Q2 = Matrix{Float64}(2I(2))

    inv = GroupedParam(Q1)
    @test StateSpaceDynamics.is_invariant(inv)
    @test StateSpaceDynamics.ngroups(inv) == 1

    g = GroupedParam([Q1, Q2], :cond, [1, 2])
    @test !StateSpaceDynamics.is_invariant(g)
    @test StateSpaceDynamics.ngroups(g) == 2
    @test g.label == :cond

    @test_throws StateSpaceDynamics.DimensionMismatchError GroupedParam([Q1, Q2], :c, [1])
    @test_throws ArgumentError GroupedParam([Q1, Q2], :c, [1, 1])
    return nothing
end

function test_tv_constructors_and_validation()
    rng = StableRNG(1)
    D, p = 2, 3
    lds = _make_tv_gaussian(rng, D, p)
    @test validate_LDS(lds) === nothing
    @test lds.latent_dim == D
    @test lds.obs_dim == p
    @test length(lds.fit_bool) == 6

    plds = _make_tv_poisson(rng, D, p)
    @test validate_LDS(plds) === nothing
    @test length(plds.fit_bool) == 5
    @test StateSpaceDynamics._is_poisson_like(plds.obs_model)

    # Mismatched groupings within a mean block are rejected by validate_LDS.
    badC = GroupedParam([randn(rng, p, D), randn(rng, p, D)], :session, [1, 2])
    badd = GroupedParam([randn(rng, p), randn(rng, p)], :other, [1, 2])  # different label
    bad_om = TrialVaryingGaussianObservationModel(;
        C=badC, d=badd, R=Matrix{Float64}(I(p))
    )
    sm = lds.state_model
    @test_throws ArgumentError LinearDynamicalSystem(sm, bad_om)

    # Non-SPD covariance group is rejected.
    bad_sm = TrialVaryingGaussianStateModel(;
        A=_tv_stable_A(rng, D), b=zeros(D), x0=zeros(D), P0=Matrix{Float64}(I(D)),
        Q=GroupedParam([Matrix{Float64}(I(D)), zeros(D, D)], :cond, [1, 2]),
    )
    @test_throws StateSpaceDynamics.NotPositiveDefiniteError LinearDynamicalSystem(
        bad_sm, TrialVaryingGaussianObservationModel(; C=randn(rng, p, D), d=zeros(p), R=Matrix{Float64}(I(p)))
    )
    return nothing
end

function test_tv_show()
    rng = StableRNG(2)
    lds = _make_tv_gaussian(rng, 2, 3)
    s = sprint(show, lds.state_model)
    @test occursin("Trial-Varying Gaussian State Model", s)
    @test occursin("by :cond", s)
    @test occursin("Trial-Varying Gaussian Observation", sprint(show, lds.obs_model))
    @test occursin("Trial-Varying Poisson", sprint(show, _make_tv_poisson(rng, 2, 3).obs_model))
    return nothing
end

# ---------------------------------------------------------------------------
# Sampling + error paths
# ---------------------------------------------------------------------------

function test_tv_rand_and_errors()
    rng = StableRNG(10)
    D, p = 2, 3
    lds = _make_tv_gaussian(rng, D, p)
    ntrials = 6
    labels = Dict(
        :cond => [1, 2, 1, 2, 1, 2], :session => ["s1", "s2", "s2", "s1", "s1", "s2"]
    )
    Ts = fill(25, ntrials)
    x, y = StateSpaceDynamics.rand(StableRNG(11), lds, Ts; labels=labels)
    @test length(y) == ntrials
    for t in 1:ntrials
        @test size(x[t]) == (D, Ts[t])
        @test size(y[t]) == (p, Ts[t])
    end

    # Missing label key.
    @test_throws ArgumentError StateSpaceDynamics.rand(
        lds, Ts; labels=Dict(:cond => labels[:cond])
    )
    # Label value not registered in group_ids.
    bad_labels = Dict(:cond => [1, 2, 3, 1, 2, 1], :session => labels[:session])
    @test_throws ArgumentError StateSpaceDynamics.rand(lds, Ts; labels=bad_labels)
    return nothing
end

# ---------------------------------------------------------------------------
# Reduction to the plain model when all blocks are invariant
# ---------------------------------------------------------------------------

function test_tv_reduces_to_plain()
    rng = StableRNG(20)
    D, p = 2, 3
    A = _tv_stable_A(rng, D)
    Qt = Matrix{Float64}(0.05 * I(D))
    Ct = 0.5 .* randn(rng, p, D)
    Rt = Matrix{Float64}(0.2 * I(p))
    true_lds = LinearDynamicalSystem(
        GaussianStateModel(; A=A, Q=Qt, b=zeros(D), x0=zeros(D), P0=Matrix{Float64}(0.1I(D))),
        GaussianObservationModel(; C=Ct, R=Rt, d=0.1 .* randn(rng, p)),
    )
    ntrials = 5
    Ts = fill(30, ntrials)
    _, y = StateSpaceDynamics.rand(StableRNG(21), true_lds, Ts)

    init_state() = GaussianStateModel(;
        A=0.7 * Matrix{Float64}(I, D, D), Q=Matrix{Float64}(0.1I(D)), b=zeros(D),
        x0=zeros(D), P0=Matrix{Float64}(0.2I(D)),
    )
    init_C() = Matrix{Float64}(0.2 * I, p, D)
    init_obs() = GaussianObservationModel(; C=init_C(), R=Matrix{Float64}(0.5I(p)), d=zeros(p))

    plain = LinearDynamicalSystem(init_state(), init_obs())
    tv = _tv_invariant_gaussian(
        0.7 * Matrix{Float64}(I, D, D), zeros(D), Matrix{Float64}(0.1I(D)), zeros(D),
        Matrix{Float64}(0.2I(D)), init_C(), zeros(p), Matrix{Float64}(0.5I(p)),
    )

    e_plain = fit!(plain, y; max_iter=30, progress=false)
    e_tv = fit!(tv, y; labels=Dict{Symbol,Vector}(), max_iter=30, progress=false)

    @test isapprox(e_plain[end], e_tv[end]; rtol=1e-4, atol=1e-4)
    @test isapprox(plain.state_model.A, tv.state_model.A.values[1]; rtol=1e-3, atol=1e-3)
    @test isapprox(plain.state_model.Q, tv.state_model.Q.values[1]; rtol=1e-3, atol=1e-3)
    @test isapprox(plain.obs_model.C, tv.obs_model.C.values[1]; rtol=1e-3, atol=1e-3)
    @test isapprox(plain.obs_model.R, tv.obs_model.R.values[1]; rtol=1e-3, atol=1e-3)
    return nothing
end

# ---------------------------------------------------------------------------
# Multi-block trial-varying fit (different labels per block)
# ---------------------------------------------------------------------------

function test_tv_gaussian_fit_multiblock()
    rng = StableRNG(22)
    D, p = 2, 3
    true_lds = _make_tv_gaussian(rng, D, p)
    ntrials = 10
    cond = repeat([1, 2]; outer=ntrials ÷ 2)
    session = repeat(["s1", "s2"]; inner=ntrials ÷ 2)
    labels = Dict(:cond => cond, :session => session)
    Ts = fill(40, ntrials)
    _, y = StateSpaceDynamics.rand(StableRNG(23), true_lds, Ts; labels=labels)

    fit_lds = _make_tv_gaussian(StableRNG(99), D, p)
    elbos = fit!(fit_lds, y; labels=labels, max_iter=40, progress=false)
    @test all(isfinite, elbos)
    @test elbos[end] > elbos[1]

    # Group structure preserved.
    @test StateSpaceDynamics.ngroups(fit_lds.state_model.Q) == 2
    @test StateSpaceDynamics.ngroups(fit_lds.obs_model.C) == 2
    # Q groups should differ (they were generated from different true Qs).
    @test !isapprox(fit_lds.state_model.Q.values[1], fit_lds.state_model.Q.values[2])

    xs, Ps = StateSpaceDynamics.smooth(fit_lds, y; labels=labels)
    for t in 1:ntrials
        @test size(xs[t]) == (D, Ts[t])
        @test size(Ps[t]) == (D, D, Ts[t])
    end
    @test isfinite(StateSpaceDynamics.loglikelihood(fit_lds, y; labels=labels))

    # Channel mismatch and labels-length mismatch are rejected.
    @test_throws StateSpaceDynamics.DimensionMismatchError fit!(
        fit_lds, [randn(p + 1, Ts[1]) for _ in 1:ntrials]; labels=labels, max_iter=1,
        progress=false,
    )
    @test_throws StateSpaceDynamics.DimensionMismatchError fit!(
        fit_lds, y; labels=Dict(:cond => cond[1:2], :session => session), max_iter=1,
        progress=false,
    )
    return nothing
end

# A trial-varying model with the emission block grouped by session and all else
# invariant — the fixed-dim analogue of stitching, exercised through the general
# path.
function test_tv_emission_only()
    rng = StableRNG(24)
    D, p = 2, 4
    A = _tv_stable_A(rng, D)
    sm = TrialVaryingGaussianStateModel(;
        A=A, b=zeros(D), x0=zeros(D), P0=Matrix{Float64}(0.1I(D)), Q=Matrix{Float64}(0.05I(D))
    )
    C1 = 0.5 .* randn(rng, p, D)
    C2 = 0.5 .* randn(rng, p, D)
    om = TrialVaryingGaussianObservationModel(;
        C=GroupedParam([C1, C2], :session, [1, 2]),
        d=GroupedParam([zeros(p), zeros(p)], :session, [1, 2]),
        R=Matrix{Float64}(0.2I(p)),
    )
    true_lds = LinearDynamicalSystem(sm, om)
    ntrials = 8
    session = repeat([1, 2]; outer=ntrials ÷ 2)
    labels = Dict(:session => session)
    Ts = fill(35, ntrials)
    _, y = StateSpaceDynamics.rand(StableRNG(25), true_lds, Ts; labels=labels)

    fit_lds = LinearDynamicalSystem(
        TrialVaryingGaussianStateModel(;
            A=0.7Matrix{Float64}(I, D, D), b=zeros(D), x0=zeros(D),
            P0=Matrix{Float64}(0.2I(D)), Q=Matrix{Float64}(0.1I(D)),
        ),
        TrialVaryingGaussianObservationModel(;
            C=GroupedParam([0.2 .* randn(StableRNG(7), p, D), 0.2 .* randn(StableRNG(8), p, D)], :session, [1, 2]),
            d=GroupedParam([zeros(p), zeros(p)], :session, [1, 2]),
            R=Matrix{Float64}(0.5I(p)),
        ),
    )
    elbos = fit!(fit_lds, y; labels=labels, max_iter=40, progress=false)
    @test all(isfinite, elbos)
    @test elbos[end] > elbos[1]
    # Q is shared (invariant) → single group.
    @test StateSpaceDynamics.ngroups(fit_lds.state_model.Q) == 1
    return nothing
end

# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

function test_tv_poisson_fit()
    rng = StableRNG(30)
    D, p = 2, 3
    true_lds = _make_tv_poisson(rng, D, p)
    ntrials = 6
    cond = repeat([1, 2]; outer=ntrials ÷ 2)
    session = repeat(["s1", "s2"]; inner=ntrials ÷ 2)
    labels = Dict(:cond => cond, :session => session)
    Ts = fill(30, ntrials)
    x, y = StateSpaceDynamics.rand(StableRNG(31), true_lds, Ts; labels=labels)
    for t in 1:ntrials
        @test all(y[t] .>= 0) && all(y[t] .== round.(y[t]))
    end

    fit_lds = _make_tv_poisson(StableRNG(98), D, p)
    elbos = fit!(
        fit_lds, y; labels=labels, max_iter=15, progress=false, newton_max_iter=10
    )
    @test all(isfinite, elbos)
    @test elbos[end] > elbos[1]
    @test StateSpaceDynamics.ngroups(fit_lds.obs_model.C) == 2

    xs, Ps = StateSpaceDynamics.smooth(fit_lds, y; labels=labels)
    for t in 1:ntrials
        @test size(xs[t]) == (D, Ts[t])
    end
    return nothing
end
