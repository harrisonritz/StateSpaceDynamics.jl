# Tests for parameter-level `Indexed` (Static/Varying) parameters. Any parameter
# of the standard Gaussian/Poisson/SLDS models may be Static or Varying; group
# assignment is supplied via `labels=Dict(:label=>per_trial_vector)`.

const SSD = StateSpaceDynamics

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

_stable_A(rng, D) = 0.8 * Matrix{Float64}(I, D, D) .+ 0.05 .* randn(rng, D, D)
_spd(rng, n, s=0.2) = Matrix{Float64}(s * I(n))

function _plain_state(rng, D)
    return GaussianStateModel(;
        A=_stable_A(rng, D), Q=_spd(rng, D, 0.05), b=zeros(D), x0=zeros(D), P0=_spd(rng, D, 0.1)
    )
end

# ---------------------------------------------------------------------------
# Indexed unit behavior
# ---------------------------------------------------------------------------

function test_indexed_basics()
    M1 = [1.0 0.0; 0.0 1.0]
    M2 = [2.0 0.0; 0.0 2.0]

    s = Static(M1)
    @test SSD.at(s, 1) === M1
    @test SSD.at(s, 7) === M1
    @test SSD.nvals(s) == 1
    @test SSD.is_indexed(s) && !SSD.is_varying(s)

    v = Varying([M1, M2], :cond, [1, 2])
    @test SSD.at(v, 1) === M1
    @test SSD.at(v, 2) === M2
    @test SSD.nvals(v) == 2
    @test SSD.is_varying(v)
    @test SSD.param_label(v) == :cond

    # plain values are implicitly static
    @test SSD.at(M1, 3) === M1
    @test SSD.nvals(M1) == 1
    @test !SSD.is_indexed(M1)

    # default group ids
    @test Varying([M1, M2], :s).group_ids == [1, 2]

    # constructor validation
    @test_throws ArgumentError Varying([M1, M2], :s, [1])       # length mismatch
    @test_throws ArgumentError Varying([M1, M2], :s, [1, 1])    # non-unique
    @test_throws ArgumentError Varying(Matrix{Float64}[], :s, Int[])  # empty
    return nothing
end

# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------

function test_indexed_validation()
    rng = StableRNG(1)
    D, p = 2, 3
    sm = _plain_state(rng, D)
    om = GaussianObservationModel(;
        C=Varying([0.5 .* randn(rng, p, D), 0.5 .* randn(rng, p, D)], :session, [1, 2]),
        d=Varying([zeros(p), zeros(p)], :session, [1, 2]),
        R=Varying([_spd(rng, p), _spd(rng, p)], :session, [1, 2]),
    )
    lds = LinearDynamicalSystem(sm, om)
    @test validate_LDS(lds) === nothing
    @test lds.latent_dim == D
    @test lds.obs_dim == p

    # Block-consistency: A Varying but b plain must error.
    bad_sm = GaussianStateModel(;
        A=Varying([_stable_A(rng, D), _stable_A(rng, D)], :cond, [1, 2]),
        Q=_spd(rng, D, 0.05), b=zeros(D), x0=zeros(D), P0=_spd(rng, D, 0.1),
    )
    @test_throws ArgumentError LinearDynamicalSystem(bad_sm, GaussianObservationModel(; C=randn(rng, p, D), R=_spd(rng, p), d=zeros(p)))

    # Emission [C d] must share indexing.
    bad_om = GaussianObservationModel(;
        C=Varying([randn(rng, p, D), randn(rng, p, D)], :session, [1, 2]),
        d=zeros(p), R=_spd(rng, p),
    )
    @test_throws ArgumentError LinearDynamicalSystem(_plain_state(rng, D), bad_om)

    # Varying obs_dim requires R to track C.
    bad_dims = GaussianObservationModel(;
        C=Varying([randn(rng, 3, D), randn(rng, 4, D)], :session, [1, 2]),
        d=Varying([zeros(3), zeros(4)], :session, [1, 2]),
        R=_spd(rng, 3),   # plain R can't cover both obs_dims
    )
    @test_throws ArgumentError LinearDynamicalSystem(_plain_state(rng, D), bad_dims)

    # Non-SPD group covariance is rejected.
    bad_cov = GaussianStateModel(;
        A=_stable_A(rng, D), b=zeros(D), x0=zeros(D), P0=_spd(rng, D, 0.1),
        Q=Varying([_spd(rng, D, 0.05), zeros(D, D)], :cond, [1, 2]),
    )
    @test_throws SSD.NotPositiveDefiniteError LinearDynamicalSystem(
        bad_cov, GaussianObservationModel(; C=randn(rng, p, D), R=_spd(rng, p), d=zeros(p))
    )
    return nothing
end

function test_indexed_show()
    rng = StableRNG(2)
    D, p = 2, 3
    om = GaussianObservationModel(;
        C=Varying([randn(rng, p, D), randn(rng, p, D)], :session, [1, 2]),
        d=Varying([zeros(p), zeros(p)], :session, [1, 2]),
        R=Varying([_spd(rng, p), _spd(rng, p)], :session, [1, 2]),
    )
    lds = LinearDynamicalSystem(_plain_state(rng, D), om)
    s = sprint(show, lds)
    @test occursin("Varying", s)
    @test occursin("session", s)
    return nothing
end

# ---------------------------------------------------------------------------
# All-Static reduces to the plain fast path
# ---------------------------------------------------------------------------

function test_indexed_static_matches_plain()
    rng = StableRNG(20)
    D, p = 2, 3
    true_lds = LinearDynamicalSystem(
        GaussianStateModel(; A=_stable_A(rng, D), Q=_spd(rng, D, 0.05), b=zeros(D), x0=zeros(D), P0=_spd(rng, D, 0.1)),
        GaussianObservationModel(; C=0.5 .* randn(rng, p, D), R=_spd(rng, p), d=0.1 .* randn(rng, p)),
    )
    ntrials = 5
    Ts = fill(30, ntrials)
    _, y = StateSpaceDynamics.rand(StableRNG(21), true_lds, Ts)

    init_state() = GaussianStateModel(; A=0.7Matrix{Float64}(I, D, D), Q=_spd(rng, D, 0.1), b=zeros(D), x0=zeros(D), P0=_spd(rng, D, 0.2))
    init_C() = Matrix{Float64}(0.2 * I, p, D)

    plain = LinearDynamicalSystem(init_state(), GaussianObservationModel(; C=init_C(), R=_spd(rng, p, 0.5), d=zeros(p)))
    # Wrap one parameter in Static -> routes through the indexed path (one regime).
    stat_sm = GaussianStateModel(; A=Static(0.7Matrix{Float64}(I, D, D)), Q=_spd(rng, D, 0.1), b=zeros(D), x0=zeros(D), P0=_spd(rng, D, 0.2))
    stat = LinearDynamicalSystem(stat_sm, GaussianObservationModel(; C=init_C(), R=_spd(rng, p, 0.5), d=zeros(p)))

    e_plain = fit!(plain, y; max_iter=30, progress=false)
    e_stat = fit!(stat, y; max_iter=30, progress=false)

    @test isapprox(e_plain[end], e_stat[end]; rtol=1e-4, atol=1e-4)
    @test isapprox(plain.state_model.A, SSD.at(stat.state_model.A, 1); rtol=1e-3, atol=1e-3)
    @test isapprox(plain.obs_model.C, stat.obs_model.C; rtol=1e-3, atol=1e-3)
    @test isapprox(plain.obs_model.R, stat.obs_model.R; rtol=1e-3, atol=1e-3)
    return nothing
end

# ---------------------------------------------------------------------------
# Multi-block Gaussian fit (different labels per block)
# ---------------------------------------------------------------------------

function _make_indexed_gaussian(rng, D, p)
    sm = GaussianStateModel(;
        A=_stable_A(rng, D), b=zeros(D), x0=zeros(D), P0=_spd(rng, D, 0.1),
        Q=Varying([_spd(rng, D, 0.05), _spd(rng, D, 0.15)], :cond, [1, 2]),
    )
    om = GaussianObservationModel(;
        C=Varying([0.5 .* randn(rng, p, D), 0.5 .* randn(rng, p, D)], :session, ["s1", "s2"]),
        d=Varying([0.1 .* randn(rng, p), 0.1 .* randn(rng, p)], :session, ["s1", "s2"]),
        R=_spd(rng, p, 0.2),
    )
    return LinearDynamicalSystem(sm, om)
end

function test_indexed_gaussian_fit_multiblock()
    rng = StableRNG(22)
    D, p = 2, 3
    true_lds = _make_indexed_gaussian(rng, D, p)
    ntrials = 10
    cond = repeat([1, 2]; outer=ntrials ÷ 2)
    session = repeat(["s1", "s2"]; inner=ntrials ÷ 2)
    labels = Dict(:cond => cond, :session => session)
    Ts = fill(40, ntrials)
    _, y = StateSpaceDynamics.rand(StableRNG(23), true_lds, Ts; labels=labels)

    fit_lds = _make_indexed_gaussian(StableRNG(99), D, p)
    elbos = fit!(fit_lds, y; labels=labels, max_iter=40, progress=false)
    @test all(isfinite, elbos)
    @test elbos[end] > elbos[1]
    @test SSD.nvals(fit_lds.state_model.Q) == 2
    @test SSD.nvals(fit_lds.obs_model.C) == 2
    @test !isapprox(SSD.at(fit_lds.state_model.Q, 1), SSD.at(fit_lds.state_model.Q, 2))

    xs, Ps = StateSpaceDynamics.smooth(fit_lds, y; labels=labels)
    for t in 1:ntrials
        @test size(xs[t]) == (D, Ts[t])
        @test size(Ps[t]) == (D, D, Ts[t])
    end
    @test isfinite(StateSpaceDynamics.loglikelihood(fit_lds, y; labels=labels))

    # error paths
    @test_throws ArgumentError fit!(fit_lds, y; max_iter=1, progress=false)  # missing labels
    @test_throws ArgumentError fit!(
        fit_lds, y; labels=Dict(:cond => cond, :session => fill("zzz", ntrials)),
        max_iter=1, progress=false,
    )  # unknown label value
    @test_throws SSD.DimensionMismatchError fit!(
        fit_lds, y; labels=Dict(:cond => cond[1:3], :session => session), max_iter=1,
        progress=false,
    )  # label length mismatch
    return nothing
end

# ---------------------------------------------------------------------------
# Varying obs_dim (stitching with different channel counts)
# ---------------------------------------------------------------------------

function _make_varying_obsdim(rng, D, ps)
    sm = _plain_state(rng, D)
    om = GaussianObservationModel(;
        C=Varying([0.5 .* randn(rng, pg, D) for pg in ps], :session, collect(1:length(ps))),
        d=Varying([0.1 .* randn(rng, pg) for pg in ps], :session, collect(1:length(ps))),
        R=Varying([_spd(rng, pg) for pg in ps], :session, collect(1:length(ps))),
    )
    return LinearDynamicalSystem(sm, om)
end

function test_indexed_varying_obsdim()
    rng = StableRNG(30)
    D, ps = 2, [3, 4]
    true_lds = _make_varying_obsdim(rng, D, ps)
    ntrials = 8
    session = repeat([1, 2]; outer=ntrials ÷ 2)
    labels = Dict(:session => session)
    Ts = fill(35, ntrials)
    x, y = StateSpaceDynamics.rand(StableRNG(31), true_lds, Ts; labels=labels)
    for t in 1:ntrials
        @test size(y[t]) == (ps[session[t]], Ts[t])
    end

    fit_lds = _make_varying_obsdim(StableRNG(98), D, ps)
    elbos = fit!(fit_lds, y; labels=labels, max_iter=40, progress=false)
    @test all(isfinite, elbos)
    @test elbos[end] > elbos[1]
    @test size(SSD.at(fit_lds.obs_model.C, 1)) == (ps[1], D)
    @test size(SSD.at(fit_lds.obs_model.C, 2)) == (ps[2], D)

    xs, Ps = StateSpaceDynamics.smooth(fit_lds, y; labels=labels)
    for t in 1:ntrials
        @test size(xs[t]) == (D, Ts[t])
    end
    @test isfinite(StateSpaceDynamics.loglikelihood(fit_lds, y; labels=labels))
    return nothing
end

# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

function _make_indexed_poisson(rng, D, p)
    sm = GaussianStateModel(;
        A=_stable_A(rng, D), b=zeros(D), x0=zeros(D), P0=_spd(rng, D, 0.1),
        Q=Varying([_spd(rng, D, 0.05), _spd(rng, D, 0.1)], :cond, [1, 2]),
    )
    om = PoissonObservationModel(;
        C=Varying([0.3 .* randn(rng, p, D), 0.3 .* randn(rng, p, D)], :session, [1, 2]),
        d=Varying([-0.5 .+ 0.1 .* randn(rng, p), -0.5 .+ 0.1 .* randn(rng, p)], :session, [1, 2]),
    )
    return LinearDynamicalSystem(sm, om)
end

function test_indexed_poisson_fit()
    rng = StableRNG(40)
    D, p = 2, 3
    true_lds = _make_indexed_poisson(rng, D, p)
    ntrials = 6
    cond = repeat([1, 2]; outer=ntrials ÷ 2)
    session = repeat([1, 2]; inner=ntrials ÷ 2)
    labels = Dict(:cond => cond, :session => session)
    Ts = fill(30, ntrials)
    _, y = StateSpaceDynamics.rand(StableRNG(41), true_lds, Ts; labels=labels)
    for t in 1:ntrials
        @test all(y[t] .>= 0) && all(y[t] .== round.(y[t]))
    end

    fit_lds = _make_indexed_poisson(StableRNG(97), D, p)
    elbos = fit!(fit_lds, y; labels=labels, max_iter=15, progress=false, newton_max_iter=10)
    @test all(isfinite, elbos)
    @test elbos[end] > elbos[1]
    @test SSD.nvals(fit_lds.obs_model.C) == 2
    return nothing
end

# ---------------------------------------------------------------------------
# SLDS
# ---------------------------------------------------------------------------

function _make_indexed_slds(rng, D, p, K)
    ldss = map(1:K) do _
        sm = _plain_state(rng, D)
        om = GaussianObservationModel(;
            C=Varying([0.5 .* randn(rng, p, D), 0.5 .* randn(rng, p, D)], :session, [1, 2]),
            d=Varying([0.1 .* randn(rng, p), 0.1 .* randn(rng, p)], :session, [1, 2]),
            R=Varying([_spd(rng, p), _spd(rng, p)], :session, [1, 2]),
        )
        LinearDynamicalSystem(sm, om)
    end
    A = [0.9 0.1; 0.1 0.9]
    return SLDS(; A=A, πₖ=[0.5, 0.5], LDSs=ldss)
end

function test_indexed_slds_fit()
    rng = StableRNG(50)
    D, p, K = 2, 3, 2
    true_slds = _make_indexed_slds(rng, D, p, K)
    ntrials = 4
    session = [1, 2, 1, 2]
    labels = Dict(:session => session)
    Ts = fill(30, ntrials)
    z, x, y = StateSpaceDynamics.rand(StableRNG(51), true_slds, Ts; labels=labels)
    for t in 1:ntrials
        @test length(z[t]) == Ts[t]
        @test size(y[t]) == (p, Ts[t])
    end

    fit_slds = _make_indexed_slds(StableRNG(96), D, p, K)
    elbos = fit!(fit_slds, y; labels=labels, max_iter=8, progress=false)
    @test length(elbos) == 8
    @test all(isfinite, elbos)
    @test SSD.nvals(fit_slds.LDSs[1].obs_model.C) == 2
    for i in 1:K
        @test isapprox(sum(fit_slds.A[i, :]), 1.0; atol=1e-8)
    end
    return nothing
end
