#=============================================================================
Held-out (test-set) ELBO scoring during `fit!`, and early stopping on it.

The properties worth pinning down are:

* **non-breaking** — without `y_test` every `fit!` returns exactly the
  `Vector{T}` it always did, and *with* it the training half of the returned
  `FitTrace` is elementwise identical, so scoring cannot perturb the fit;
* **alignment** — the held-out value recorded at iteration `k` is the ELBO at
  the parameters the training ELBO at iteration `k` was computed from, checked
  against a standalone `elbo` call on a model stopped one iteration earlier;
* **determinism** — repeating a fit reproduces the held-out trace exactly,
  which is what makes stopping on it well-posed (this is the premise the
  feature rests on, so it is asserted for every family, including the SLDS
  whose *fit* E-step is Monte-Carlo while its `elbo` is not);
* **stopping semantics** — `patience` consecutive non-improving scores, and
  `restore_best` leaving the model at the best-scoring parameters.
=============================================================================#

const HO_D, HO_N, HO_T, HO_NTR = 2, 4, 40, 6

function _ho_sm(a)
    return GaussianStateModel(
        a * Matrix{Float64}(I, HO_D, HO_D),
        0.1 * Matrix{Float64}(I, HO_D, HO_D),
        zeros(HO_D),
        zeros(HO_D),
        Matrix{Float64}(I, HO_D, HO_D),
    )
end

function _ho_om(seed)
    return GaussianObservationModel(
        randn(StableRNG(seed), HO_N, HO_D),
        0.2 * Matrix{Float64}(I, HO_N, HO_N),
        zeros(HO_N),
    )
end

_ho_lds(a, seed) = LinearDynamicalSystem(_ho_sm(a), _ho_om(seed))

function _ho_pom(seed)
    return PoissonObservationModel(
        0.4 .* randn(StableRNG(seed), HO_N, HO_D), fill(-0.5, HO_N)
    )
end

_ho_plds(a, seed) = LinearDynamicalSystem(_ho_sm(a), _ho_pom(seed))

"""Split a sampled dataset into train / test trials."""
function _ho_split(y)
    n = length(y)
    ntest = max(1, n ÷ 3)
    return y[1:(n - ntest)], y[(n - ntest + 1):n]
end

"""Gaussian train/test data from a known model."""
function _ho_gaussian_data(; seed=7)
    _, y = rand(StableRNG(seed), _ho_lds(0.9, 2), fill(HO_T, HO_NTR))
    return _ho_split(y)
end

#=
A mild plant (`ρ(M)` near 1), matching `LQRLDS.jl`'s fixture. A random
symplectic transition is unstable by construction, so sampling one over a long
horizon diverges — these tests need a model whose forward chain is usable.
=#
function _ho_lqr_lds(seed; poisson::Bool=false)
    n = 2
    d = 2n
    A = [0.96 0.07; -0.05 0.93]
    sm = LQRStateModel(
        A,
        [0.06 0.01; 0.01 0.05],
        [[0.25 0.04; 0.04 0.18]],
        Matrix(Diagonal(fill(0.03, d)));
        schedule=Int[],
        terminal=false,
        P0=Matrix(0.25I, d, d),
        x0=zeros(d),
        observe_costate=false,
    )
    C = (poisson ? 0.3 : 1.0) .* randn(StableRNG(seed), HO_N, d)
    C[:, (n + 1):d] .= 0
    om = if poisson
        PoissonObservationModel(C, fill(-0.5, HO_N))
    else
        GaussianObservationModel(C, Matrix(0.08I, HO_N, HO_N), zeros(HO_N))
    end
    return LinearDynamicalSystem(sm, om)
end

"""
    test_holdout_non_breaking()

Without `y_test` the return value is the same `Vector{T}` as always; with it,
the returned `FitTrace` carries an elementwise-identical training trace and
supports every `AbstractVector` operation callers already use on it.
"""
function test_holdout_non_breaking()
    y_tr, y_te = _ho_gaussian_data()

    plain = fit!(_ho_lds(0.5, 3), y_tr; max_iter=15, progress=false)
    @test plain isa Vector{Float64}

    traced = fit!(_ho_lds(0.5, 3), y_tr; y_test=y_te, max_iter=15, progress=false)
    @test traced isa SSD.FitTrace{Float64}
    @test traced isa AbstractVector{Float64}

    # Scoring the held-out set must not perturb the fit at all.
    @test collect(traced) == plain
    @test length(traced) == length(plain)
    @test traced[end] == plain[end]
    @test traced[3] == plain[3]
    @test diff(traced) == diff(plain)
    @test minimum(diff(traced)) >= -1e-6      # still monotone, as a Gaussian EM must be

    # The held-out half.
    @test length(traced.test) == 15
    @test traced.test_iters == collect(1:15)
    @test traced.train == plain
    @test 1 <= traced.best_iter <= 15
    @test !traced.stopped_early

    # `show` should not error.
    @test occursin("FitTrace", sprint(show, MIME"text/plain"(), traced))
    return nothing
end

"""
    test_holdout_matches_standalone_elbo()

The value recorded at iteration `k` is the held-out ELBO at the parameters the
*training* ELBO at iteration `k` was computed from — i.e. before that
iteration's M-step. Checked against an independent `elbo` call on a model
fitted for `k-1` iterations.
"""
function test_holdout_matches_standalone_elbo()
    y_tr, y_te = _ho_gaussian_data()
    tr = fit!(_ho_lds(0.5, 3), y_tr; y_test=y_te, max_iter=12, progress=false)

    for k in (1, 5, 12)
        ref = _ho_lds(0.5, 3)
        k > 1 && fit!(ref, y_tr; max_iter=k - 1, progress=false)
        @test isapprox(tr.test[k], elbo(ref, y_te); rtol=1e-10)
    end

    # Iteration 1 scores the untouched initial parameters.
    @test isapprox(tr.test[1], elbo(_ho_lds(0.5, 3), y_te); rtol=1e-10)
    return nothing
end

"""
    test_holdout_determinism()

Repeating a fit reproduces the held-out trace bit for bit, for every family.
This is the premise early stopping rests on — including for the SLDS, whose
fit E-step is Monte-Carlo but whose `elbo` is deterministic coordinate ascent.
"""
function test_holdout_determinism()
    y_tr, y_te = _ho_gaussian_data()

    a = fit!(_ho_lds(0.5, 3), y_tr; y_test=y_te, max_iter=8, progress=false)
    b = fit!(_ho_lds(0.5, 3), y_tr; y_test=y_te, max_iter=8, progress=false)
    @test a.test == b.test

    _, yp = rand(StableRNG(12), _ho_plds(0.9, 11), fill(HO_T, HO_NTR))
    yp_tr, yp_te = _ho_split(yp)
    pa = fit!(_ho_plds(0.5, 13), yp_tr; y_test=yp_te, max_iter=5, progress=false)
    pb = fit!(_ho_plds(0.5, 13), yp_tr; y_test=yp_te, max_iter=5, progress=false)
    @test pa.test == pb.test

    function mk_slds()
        return SLDS(;
            A=[0.9 0.1; 0.1 0.9],
            πₖ=[0.5, 0.5],
            LDSs=[_ho_lds(0.5 + 0.15k, 20 + k) for k in 1:2],
        )
    end
    sa = fit!(
        mk_slds(),
        y_tr;
        y_test=y_te,
        max_iter=4,
        progress=false,
        rng=StableRNG(1),
        test_kwargs=(smoothing_iters=20,),
    )
    sb = fit!(
        mk_slds(),
        y_tr;
        y_test=y_te,
        max_iter=4,
        progress=false,
        rng=StableRNG(1),
        test_kwargs=(smoothing_iters=20,),
    )
    @test sa.test == sb.test
    return nothing
end

"""
    test_holdout_test_every()

`test_every` spaces the checks, always including iteration 1.
"""
function test_holdout_test_every()
    y_tr, y_te = _ho_gaussian_data()

    tr = fit!(
        _ho_lds(0.5, 3),
        y_tr;
        y_test=y_te,
        test_every=5,
        max_iter=20,
        tol=0.0,
        progress=false,
    )
    @test tr.test_iters == [1, 6, 11, 16]
    @test length(tr.test) == 4
    @test length(tr.train) == 20

    tr3 = fit!(
        _ho_lds(0.5, 3),
        y_tr;
        y_test=y_te,
        test_every=3,
        max_iter=10,
        tol=0.0,
        progress=false,
    )
    @test tr3.test_iters == [1, 4, 7, 10]

    # The scored values agree with a per-iteration run at the same iterations.
    every1 = fit!(_ho_lds(0.5, 3), y_tr; y_test=y_te, max_iter=10, tol=0.0, progress=false)
    @test tr3.test ≈ every1.test[tr3.test_iters]

    @test_throws ArgumentError fit!(
        _ho_lds(0.5, 3), y_tr; y_test=y_te, test_every=0, progress=false
    )
    return nothing
end

"""
    test_holdout_early_stopping()

`early_stopping` is off by default. When on, the fit stops after `patience`
consecutive non-improving scores, and a larger `patience` runs at least as long.
"""
function test_holdout_early_stopping()
    y_tr, y_te = _ho_gaussian_data()

    # Off by default: passing `y_test` alone only records.
    open_run = fit!(
        _ho_lds(0.5, 3), y_tr; y_test=y_te, max_iter=40, tol=0.0, progress=false
    )
    @test !open_run.stopped_early
    @test length(open_run) == 40
    # The held-out curve does turn over on this data, so there is something to stop on.
    @test open_run.best_iter < 40

    stopped = fit!(
        _ho_lds(0.5, 3),
        y_tr;
        y_test=y_te,
        early_stopping=true,
        max_iter=300,
        tol=0.0,
        progress=false,
    )
    @test stopped.stopped_early
    @test length(stopped) < 300
    # patience=1 stops *at* the first non-improving score, so the best is the
    # scored iteration just before it.
    @test stopped.best_iter == stopped.test_iters[end - 1]
    @test stopped.best_iter == open_run.best_iter
    @test maximum(stopped.test) == stopped.test[end - 1]

    # The training trace up to the stop matches the un-stopped run.
    @test stopped.train ≈ open_run.train[1:length(stopped)]

    patient = fit!(
        _ho_lds(0.5, 3),
        y_tr;
        y_test=y_te,
        early_stopping=true,
        patience=3,
        max_iter=300,
        tol=0.0,
        progress=false,
    )
    @test patient.stopped_early
    @test length(patient) >= length(stopped)
    @test patient.best_iter == stopped.best_iter

    @test_throws ArgumentError fit!(
        _ho_lds(0.5, 3), y_tr; y_test=y_te, patience=0, progress=false
    )
    @test_throws ArgumentError fit!(
        _ho_lds(0.5, 3), y_tr; y_test=y_te, min_delta=-1.0, progress=false
    )
    return nothing
end

"""
    test_holdout_restore_best()

On an early stop `restore_best=true` leaves the model at the parameters that
scored best — verified by re-scoring the returned model — and `false` leaves it
at the iterate it stopped on. A fit that runs to completion is never rolled
back, whatever `restore_best` says.
"""
function test_holdout_restore_best()
    y_tr, y_te = _ho_gaussian_data()

    restored = _ho_lds(0.5, 3)
    tr = fit!(
        restored,
        y_tr;
        y_test=y_te,
        early_stopping=true,
        max_iter=300,
        tol=0.0,
        progress=false,
    )
    @test tr.stopped_early
    @test isapprox(elbo(restored, y_te), maximum(tr.test); rtol=1e-8)

    kept = _ho_lds(0.5, 3)
    tr2 = fit!(
        kept,
        y_tr;
        y_test=y_te,
        early_stopping=true,
        restore_best=false,
        max_iter=300,
        tol=0.0,
        progress=false,
    )
    @test tr2.test == tr.test                      # same trajectory either way
    @test isapprox(elbo(kept, y_te), tr2.test[end]; rtol=1e-8)
    # The kept iterate is the one that triggered the stop, so it scores worse.
    @test elbo(kept, y_te) < elbo(restored, y_te)

    #=
    Running to completion must not roll back: `restore_best` describes what an
    early stop does, not a silent "return the best model" mode.
    =#
    full = _ho_lds(0.5, 3)
    tr3 = fit!(full, y_tr; y_test=y_te, max_iter=40, tol=0.0, progress=false)
    @test !tr3.stopped_early
    @test tr3.best_iter < 40
    # Identical to the same fit with no held-out data at all: nothing was rolled back.
    reference = _ho_lds(0.5, 3)
    fit!(reference, y_tr; max_iter=40, tol=0.0, progress=false)
    @test elbo(full, y_te) == elbo(reference, y_te)
    # And it is past the peak, so a rollback would have been observable.
    @test elbo(full, y_te) < maximum(tr3.test)
    return nothing
end

"""
    test_holdout_all_families()

Every `fit!` entry point accepts held-out data, returns a `FitTrace` whose
training half is unchanged, and early-stops with the model restored.
"""
function test_holdout_all_families()
    @testset "Gaussian LDS" begin
        y_tr, y_te = _ho_gaussian_data()
        plain = fit!(_ho_lds(0.5, 3), y_tr; max_iter=6, progress=false)
        tr = fit!(_ho_lds(0.5, 3), y_tr; y_test=y_te, max_iter=6, progress=false)
        @test plain isa Vector{Float64}
        @test tr isa SSD.FitTrace
        @test collect(tr) == plain
    end

    @testset "Poisson LDS" begin
        _, yp = rand(StableRNG(12), _ho_plds(0.9, 11), fill(HO_T, HO_NTR))
        yp_tr, yp_te = _ho_split(yp)
        plain = fit!(_ho_plds(0.5, 13), yp_tr; max_iter=6, progress=false)
        tr = fit!(_ho_plds(0.5, 13), yp_tr; y_test=yp_te, max_iter=6, progress=false)
        @test plain isa Vector{Float64}
        @test tr isa SSD.FitTrace
        @test collect(tr) == plain
        @test length(tr.test) == 6

        # `test_kwargs` reaches the Poisson `elbo`'s Newton controls.
        cheap = fit!(
            _ho_plds(0.5, 13),
            yp_tr;
            y_test=yp_te,
            max_iter=4,
            progress=false,
            test_kwargs=(newton_max_iter=5, newton_tol=1e-4),
        )
        @test length(cheap.test) == 4
    end

    @testset "SLDS" begin
        y_tr, y_te = _ho_gaussian_data()
        mk() = SLDS(;
            A=[0.9 0.1; 0.1 0.9],
            πₖ=[0.5, 0.5],
            LDSs=[_ho_lds(0.5 + 0.15k, 20 + k) for k in 1:2],
        )
        plain = fit!(mk(), y_tr; max_iter=4, progress=false, rng=StableRNG(1))
        tr = fit!(
            mk(),
            y_tr;
            y_test=y_te,
            max_iter=4,
            progress=false,
            rng=StableRNG(1),
            test_kwargs=(smoothing_iters=20,),
        )
        @test plain isa Vector{Float64}
        @test tr isa SSD.FitTrace
        @test collect(tr) == plain
        @test length(tr.test) == 4
    end

    @testset "LQR (Gaussian emission)" begin
        _, yh = rand(StableRNG(32), _ho_lqr_lds(30), fill(14, HO_NTR))
        yh_tr, yh_te = _ho_split(yh)
        plain = fit!(_ho_lqr_lds(40), yh_tr; max_iter=6, progress=false)
        tr = fit!(_ho_lqr_lds(40), yh_tr; y_test=yh_te, max_iter=6, progress=false)
        @test plain isa Vector{Float64}
        @test tr isa SSD.FitTrace
        @test collect(tr) == plain

        stopped_model = _ho_lqr_lds(40)
        st = fit!(
            stopped_model,
            yh_tr;
            y_test=yh_te,
            early_stopping=true,
            max_iter=200,
            tol=0.0,
            progress=false,
        )
        if st.stopped_early
            @test isapprox(elbo(stopped_model, yh_te), maximum(st.test); rtol=1e-8)
        end
    end

    @testset "LQR (Poisson emission)" begin
        _, yhp = rand(StableRNG(50), _ho_lqr_lds(51; poisson=true), fill(14, HO_NTR))
        yhp_tr, yhp_te = _ho_split(yhp)
        plain = fit!(_ho_lqr_lds(52; poisson=true), yhp_tr; max_iter=5, progress=false)
        tr = fit!(
            _ho_lqr_lds(52; poisson=true), yhp_tr; y_test=yhp_te, max_iter=5, progress=false
        )
        @test plain isa Vector{Float64}
        @test tr isa SSD.FitTrace
        @test collect(tr) == plain
        @test length(tr.test) == 5
    end
    return nothing
end

"""
    test_holdout_composite_emission()

A `CompositeObservationModel` is the one model object in the package that is
*immutable*, so restoring the best parameters has to reach through it into its
(mutable) members rather than assigning its fields. Exercised on both an
all-Gaussian and a mixed Gaussian/Poisson composite.
"""
function test_holdout_composite_emission()
    latent = 2
    function sm()
        return GaussianStateModel(
            0.9 * [cos(0.2) -sin(0.2); sin(0.2) cos(0.2)],
            Matrix(0.05I, latent, latent),
            zeros(latent),
            zeros(latent),
            Matrix(0.2I, latent, latent),
        )
    end
    function gauss(p, seed)
        return GaussianObservationModel(
            randn(StableRNG(seed), p, latent), Matrix(Diagonal(fill(0.2, p))), zeros(p)
        )
    end
    function pois(p, seed)
        return PoissonObservationModel(
            0.4 .* randn(StableRNG(seed), p, latent), fill(log(1.2), p)
        )
    end

    for (name, mk_truth, mk_fit) in (
        (
            "all-Gaussian",
            () -> LinearDynamicalSystem(sm(), (a=gauss(3, 71), b=gauss(4, 72))),
            () -> LinearDynamicalSystem(sm(), (a=gauss(3, 73), b=gauss(4, 74))),
        ),
        (
            "mixed",
            () -> LinearDynamicalSystem(sm(), (kin=gauss(3, 75), spk=pois(4, 76))),
            () -> LinearDynamicalSystem(sm(), (kin=gauss(3, 77), spk=pois(4, 78))),
        ),
    )
        @testset "$name composite" begin
            _, ys = rand(StableRNG(79), mk_truth(), fill(30, 6))
            # `rand` yields one NamedTuple per trial; `fit!` wants one vector
            # of trials per member.
            ks = keys(ys[1])
            trials = NamedTuple{ks}(([t[k] for t in ys] for k in ks))
            split = map(_ho_split, trials)
            y_tr = NamedTuple{ks}((split[k][1] for k in ks))
            y_te = NamedTuple{ks}((split[k][2] for k in ks))

            plain = fit!(mk_fit(), y_tr; max_iter=6, progress=false)
            tr = fit!(mk_fit(), y_tr; y_test=y_te, max_iter=6, progress=false)
            @test plain isa Vector{Float64}
            @test tr isa SSD.FitTrace
            @test collect(tr) == plain
            @test length(tr.test) == 6

            #=
            The restore path: a composite is immutable, so this is what would
            throw if `_copy_model!` tried to assign its fields directly.
            =#
            m = mk_fit()
            st = fit!(
                m,
                y_tr;
                y_test=y_te,
                early_stopping=true,
                max_iter=300,
                tol=0.0,
                progress=false,
            )
            if st.stopped_early
                @test isapprox(elbo(m, y_te), maximum(st.test); rtol=1e-8)
            end
        end
    end
    return nothing
end

"""
    test_holdout_grouped()

A `depends_on` fit takes held-out data too, with the test set's own labels. The
per-cell models are views onto the parent's `variants`, so restoring the best
parameters has to reach them — checked by re-scoring the returned model.
"""
function test_holdout_grouped()
    y_tr, y_te = _ho_gaussian_data()
    ntr, nte = length(y_tr), length(y_te)
    labels_tr = [isodd(i) ? :a : :b for i in 1:ntr]
    labels_te = [isodd(i) ? :a : :b for i in 1:nte]

    function mk()
        lds = _ho_lds(0.5, 3)
        set_depends_on!(lds.obs_model, (C=labels_tr, d=labels_tr, R=labels_tr))
        return lds
    end

    plain = fit!(mk(), y_tr; max_iter=6, progress=false)
    tr = fit!(
        mk(),
        y_tr;
        y_test=y_te,
        depends_on_test=(C=labels_te, d=labels_te, R=labels_te),
        max_iter=6,
        progress=false,
    )
    @test plain isa Vector{Float64}
    @test tr isa SSD.FitTrace
    @test collect(tr) == plain
    @test length(tr.test) == 6

    # Early stop + restore has to write through to the grouped variants.
    m = mk()
    st = fit!(
        m,
        y_tr;
        y_test=y_te,
        depends_on_test=(C=labels_te, d=labels_te, R=labels_te),
        early_stopping=true,
        max_iter=300,
        tol=0.0,
        progress=false,
    )
    if st.stopped_early
        scored = elbo(m, y_te; depends_on=(C=labels_te, d=labels_te, R=labels_te))
        @test isapprox(scored, maximum(st.test); rtol=1e-8)
    end
    return nothing
end
