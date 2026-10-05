#=============================================================================
Regressions for a second external review of LDS-family fitting:

* N1 — the sufficient-statistics aggregator decided whether every trial shared
  one covariance by comparing the first two trials' storage, so a ragged or
  mixed-offset dataset whose first two trials matched was mis-weighted: an ELBO
  above the exact likelihood, and fits that changed with trial order;
* N2 — the exact Poisson moment correction computed `0 × Inf` for a finite
  expectation;
* N3 — one-bin trials reached the state kernels and failed there;
* N4 — the last trace entry scored the parameters before the final M-step,
  not the model returned;

and for the switch from a covariance-weighted ridge to the normalized
matrix-normal–inverse-Wishart prior.
=============================================================================#

function _r2_gaussian_lds(; P0=1.3)
    sm = GaussianStateModel([0.8;;], [0.2;;], [0.0], [0.0], [P0;;])
    om = GaussianObservationModel([1.1;;], [0.4;;], [0.0])
    return LinearDynamicalSystem(sm, om)
end

function _r2_poisson_lds()
    return LinearDynamicalSystem(
        _r2_gaussian_lds().state_model, PoissonObservationModel([0.3;;], [0.0])
    )
end

"""
N1: with exact Gaussian inference and no priors, the aggregated ELBO is the
marginal likelihood whatever the trial order and however the covariance buckets
fall, and the fitted parameters do not depend on the order of identical trials.
"""
function test_r2_ragged_aggregation_order_invariant()
    rng = StableRNG(11)
    y = [randn(rng, 1, t) for t in (5, 5, 12, 7, 12, 5)]
    for order in ([1, 2, 3, 4, 5, 6], [3, 1, 2, 5, 4, 6], [6, 4, 3, 5, 1, 2])
        m = _r2_gaussian_lds()
        exact = loglikelihood(m, y[order])
        @test elbo(m, y[order]) ≈ exact rtol = 1e-12
        @test sum(trial_elbos(m, y[order])) ≈ exact rtol = 1e-12
    end

    ys = [
        randn(StableRNG(12), 1, 5), randn(StableRNG(13), 1, 5), randn(StableRNG(14), 1, 12)
    ]
    fits = map(([1, 2, 3], [1, 3, 2], [3, 1, 2], [2, 3, 1])) do order
        m = _r2_gaussian_lds()
        fit!(m, ys[order]; max_iter=5, progress=false)
        [m.state_model.A[1], m.state_model.Q[1], m.obs_model.R[1], m.state_model.P0[1]]
    end
    for f in fits[2:end]
        @test f ≈ fits[1] rtol = 1e-10
    end

    # The grouping itself: storage shared within a bucket, never across.
    m = _r2_gaussian_lds()
    data = StateSpaceDynamics.Data(m, y)
    tfs = StateSpaceDynamics.initialize_FilterSmooth(m, data.tsteps)
    pool = [
        StateSpaceDynamics.SmoothWorkspace(Float64, 1, 1, 12) for
        _ in 1:Threads.maxthreadid()
    ]
    StateSpaceDynamics.smooth!(m, tfs, data, pool)
    group_of, groups = StateSpaceDynamics._shared_cov_groups(tfs)
    for (g, (rep, count)) in enumerate(groups)
        members = findall(==(g), group_of)
        @test length(members) == count
        @test rep in members
        @test allequal(data.tsteps[members])
    end
    @test all(
        group_of[a] != group_of[b] || group_of[a] == 0 for a in 1:6 for
        b in 1:6 if data.tsteps[a] != data.tsteps[b]
    )
    return nothing
end

"""
N1 on the inverse-LQR scoring path: equal-length trials at different schedule
offsets fall in different covariance buckets, so `[a, a, b]` must score what the
per-trial ELBOs sum to, in every order.
"""
function test_r2_lqr_offset_aggregation()
    sm = LQRStateModel(
        [0.9;;],
        [0.4;;],
        [[0.02;;], [2.0;;]],
        0.1 * Matrix{Float64}(I, 2, 2);
        schedule=vcat(fill(1, 8), fill(2, 8)),
        terminal=true,
        observe_costate=true,
        Σf=[0.05;;],
        P0=0.25 * Matrix{Float64}(I, 2, 2),
    )
    om = GaussianObservationModel(
        Matrix{Float64}(I, 2, 2), 0.03 * Matrix{Float64}(I, 2, 2), zeros(2)
    )
    m = LinearDynamicalSystem(sm, om)
    # Identical trials, so every order of the offsets is the same dataset.
    y1 = randn(StableRNG(1), 2, 5) .* 0.1
    y = [copy(y1) for _ in 1:3]
    scores = map(([0, 0, 8], [0, 8, 0], [8, 0, 0])) do offsets
        e = elbo(m, y; cost_offset=offsets)
        @test e ≈ sum(trial_elbos(m, y; cost_offset=offsets)) rtol = 1e-8
        e
    end
    @test scores[1] ≈ scores[2] rtol = 1e-10
    @test scores[2] ≈ scores[3] rtol = 1e-10
    return nothing
end

"""
N2: `exp(μ)(exp(ρ) − 1 − ρ)` to full precision across the range, including a
finite moment whose factors under- and overflow, and the public one-state
switching score on the case that used to return `NaN`.
"""
function test_r2_lognormal_excess_stable()
    excess = StateSpaceDynamics._lognormal_excess
    ref(μ, ρ) = Float64(exp(big(μ)) * (exp(big(ρ)) - 1 - big(ρ)))
    for (μ, ρ) in (
        (0.0, 1e-12),
        (0.3, 1e-6),
        (-1.0, 0.01),
        (2.0, 0.49),
        (2.0, 0.5),
        (-3.0, 1.7),
        (-50.0, 40.0),
        (-800.0, 800.0),
        (-700.0, 705.0),
        (1.0, -1e-9),
        (1.0, -0.3),
        (1.0, -2.0),
    )
        @test excess(μ, ρ) ≈ ref(μ, ρ) rtol = 1e-13 atol = 1e-300
    end
    @test excess(-800.0, 800.0) ≈ 1.0 rtol = 1e-12

    om = PoissonObservationModel([40.0;;], [0.0])
    @test StateSpaceDynamics._emission_moment_excess(
        om, [-20.0;;], zeros(1, 1), 1, nothing, [1.0;;]
    ) ≈ -1.0 rtol = 1e-12
    sm = GaussianStateModel([0.95;;], [0.01;;], [-1.0], [-20.0], [1.0;;])
    lds = LinearDynamicalSystem(sm, om)
    slds = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[deepcopy(lds)])
    y = zeros(1, 3)
    out = smooth(slds, y; smoothing_iters=5, return_cov=true)
    @test isfinite(out.elbo)
    @test out.elbo ≈ elbo(lds, y) atol = 1e-6
    return nothing
end

"""
N3: a trial without a transition is refused with an `ArgumentError` at every
entry point and for every emission family, before anything is fitted.
"""
function test_r2_short_trials_refused()
    for make in (_r2_gaussian_lds, _r2_poisson_lds)
        for y in (zeros(1, 1), [randn(StableRNG(3), 1, 6), zeros(1, 1)], zeros(1, 0))
            m = make()
            A0 = copy(m.state_model.A)
            @test_throws ArgumentError smooth(m, y)
            @test_throws ArgumentError elbo(m, y)
            @test_throws ArgumentError fit!(m, y; max_iter=3, progress=false)
            @test m.state_model.A == A0
        end
    end
    @test_throws ArgumentError loglikelihood(_r2_gaussian_lds(), zeros(1, 1))
    slds = SLDS(;
        A=[0.9 0.1; 0.1 0.9], πₖ=[0.5, 0.5], LDSs=[_r2_gaussian_lds() for _ in 1:2]
    )
    @test_throws ArgumentError elbo(slds, zeros(1, 1))
    @test_throws ArgumentError fit!(slds, zeros(1, 1); max_iter=2, progress=false)
    comp = LinearDynamicalSystem(
        _r2_gaussian_lds().state_model,
        CompositeObservationModel((
            a=GaussianObservationModel([1.1;;], [0.4;;], [0.0]),
            b=PoissonObservationModel([0.3;;], [0.0]),
        )),
    )
    @test_throws ArgumentError elbo(comp, (a=zeros(1, 1), b=zeros(1, 1)))
    @test isfinite(elbo(_r2_gaussian_lds(), randn(StableRNG(4), 1, 2)))
    return nothing
end

"""
N4: every driver returns a scored iterate. The last trace entry is the ELBO of
the returned model; `max_iter` counts scored iterates, so `max_iter = 1` scores
the initial model and leaves it unchanged; and after an early stop that restores
the best parameters, `returned_iter` names the entry that scores them.
"""
function test_r2_trace_scores_returned_model()
    y = randn(StableRNG(25), 1, 15)
    yp = Float64.(rand(StableRNG(25), 0:5, 1, 15))
    for (make, data) in ((_r2_gaussian_lds, y), (_r2_poisson_lds, yp))
        for max_iter in (1, 2, 6)
            m = make()
            trace = fit!(m, data; max_iter=max_iter, progress=false)
            @test length(trace) == max_iter
            @test trace[end] ≈ elbo(m, data) rtol = 1e-10
        end
        m = make()
        A0 = copy(m.state_model.A)
        fit!(m, data; max_iter=1, progress=false)
        @test m.state_model.A == A0
        # Convergence ends the fit at a scored iterate too.
        m = make()
        trace = fit!(m, data; max_iter=300, tol=1e-2, progress=false)
        @test trace[end] ≈ elbo(m, data) rtol = 1e-10
    end

    # A grouped fit.
    ys = [randn(StableRNG(k), 1, 8) for k in 1:4]
    m = _r2_gaussian_lds()
    set_depends_on!(m.obs_model, (C=[1, 1, 2, 2], d=[1, 1, 2, 2]))
    trace = fit!(m, ys; max_iter=4, progress=false)
    @test trace[end] ≈ elbo(m, ys) rtol = 1e-10

    # Early stopping with restore: the returned model is the best iterate.
    rng = StableRNG(31)
    truth = _r2_gaussian_lds()
    _, ytr = rand(rng, truth, fill(6, 3))
    _, yte = rand(rng, truth, fill(6, 3))
    m = LinearDynamicalSystem(
        GaussianStateModel([0.1;;], [1.0;;], [0.0], [0.0], [1.0;;]),
        GaussianObservationModel([0.5;;], [1.0;;], [0.0]),
    )
    trace = fit!(
        m,
        ytr;
        y_test=yte,
        max_iter=200,
        tol=0.0,
        early_stopping=true,
        patience=1,
        min_delta=1e3,
        progress=false,
    )
    @test trace.stopped_early
    @test trace.returned_iter == trace.best_iter
    @test trace.train[trace.returned_iter] ≈ elbo(m, ytr) rtol = 1e-10
    @test trace.test[findfirst(==(trace.returned_iter), trace.test_iters)] ≈ elbo(m, yte) rtol =
        1e-10
    plain = fit!(_r2_gaussian_lds(), ytr; y_test=yte, max_iter=3, progress=false)
    @test plain.returned_iter == length(plain)
    return nothing
end

"""
The matrix-normal term is the normalized `log MN(W; M₀, Σ, Λ⁻¹)` up to a
`(W, Σ)`-free constant: differences in `Σ` match the Distributions.jl density.
"""
function test_r2_mn_logprior_normalized()
    rng = StableRNG(5)
    d, q = 3, 4
    M₀ = randn(rng, d, q)
    L = randn(rng, q, q)
    Λ = Matrix(Symmetric(L * L' + q * I))
    prior = MNPrior(; M₀=M₀, Λ=Λ)
    W = randn(rng, d, q)
    Σs = [Matrix(Symmetric(G * G' + I)) for G in (randn(rng, d, d), randn(rng, d, d))]
    mn(Σ) = StateSpaceDynamics.mn_logprior_term(W, Σ, prior)
    V = Matrix(Symmetric(inv(Λ)))
    exact(Σ) = logpdf(MatrixNormal(M₀, Σ, V), W)
    @test mn(Σs[1]) - mn(Σs[2]) ≈ exact(Σs[1]) - exact(Σs[2]) rtol = 1e-10
    return nothing
end

"""
Each covariance update is the stationary point of the MAP objective it
reports: at fixed sufficient statistics, the ELBO's parameter terms (likelihood
plus normalized MNIW / NIW log-priors) fall when `Q`, `R` or `P0` is moved off
the M-step's value in any direction. Under the old ridge convention the missing
`-(q/2) log det Σ` left each a `q/N` shrinkage away from this point.
"""
function test_r2_mniw_covariance_is_map()
    rng = StableRNG(8)
    D, P = 2, 3
    A = [0.9 0.1; -0.05 0.85]
    truth = LinearDynamicalSystem(
        GaussianStateModel(A, Matrix(0.1I, D, D), zeros(D), zeros(D), Matrix(0.5I, D, D)),
        GaussianObservationModel(randn(rng, P, D), Matrix(0.2I, P, P), zeros(P)),
    )
    _, y = rand(rng, truth, fill(6, 4))
    sm = GaussianStateModel(;
        A=Matrix(0.5I, D, D),
        Q=Matrix(0.3I, D, D),
        b=zeros(D),
        x0=zeros(D),
        P0=Matrix(1.0I, D, D),
        AB_prior=MNPrior(;
            M₀=hcat(Matrix(1.0I, D, D), zeros(D)), Λ=Matrix(2.0I, D + 1, D + 1)
        ),
        Q_prior=IWPrior(; Ψ=Matrix(0.05I, D, D), ν=D + 2.0),
        x0_prior=x0_mean_prior(zeros(D); κ₀=3.0),
        P0_prior=IWPrior(; Ψ=Matrix(0.2I, D, D), ν=D + 1.0),
    )
    om = GaussianObservationModel(;
        C=randn(rng, P, D),
        R=Matrix(0.5I, P, P),
        d=zeros(P),
        CD_prior=MNPrior(; M₀=zeros(P, D + 1), Λ=Matrix(1.5I, D + 1, D + 1)),
    )
    lds = LinearDynamicalSystem(sm, om)
    data = StateSpaceDynamics.Data(lds, y)
    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, data.tsteps)
    pool = [
        StateSpaceDynamics.SmoothWorkspace(Float64, D, P, 6) for
        _ in 1:Threads.maxthreadid()
    ]
    ws = pool[1]
    suf = StateSpaceDynamics._initialize_td_sufficient_statistics(Float64, lds, data.tsteps)
    StateSpaceDynamics._td_init_const_blocks!(ws, lds, data)
    StateSpaceDynamics.estep!(lds, suf, tfs, data, pool)
    StateSpaceDynamics.mstep!(lds, suf, ws)

    function objective()
        StateSpaceDynamics.compute_smooth_constants!(ws, lds)
        return StateSpaceDynamics.elbo!(lds, suf, ws, 0.0)
    end
    f0 = objective()
    for Σ in (lds.state_model.Q, lds.obs_model.R, lds.state_model.P0)
        Σ0 = copy(Σ)
        n = size(Σ, 1)
        for _ in 1:3
            G = randn(rng, n, n)
            Δ = (G + G') ./ 2
            for ε in (1e-4, -1e-4)
                Σ .= Σ0 .+ ε .* Δ
                @test objective() < f0
            end
        end
        Σ .= Σ0
    end
    @test objective() ≈ f0
    return nothing
end

"""
The null model's covariance is the normalized MNIW MAP too: a perturbation of
`R` in any direction lowers likelihood plus log-priors.
"""
function test_r2_null_model_mniw_map()
    rng = StableRNG(9)
    p, k, n = 2, 3, 25
    X = randn(rng, k, n)
    Y = randn(rng, p, k) * X .+ 0.3 .* randn(rng, p, n)
    W_prior = MNPrior(; M₀=zeros(p, k), Λ=Matrix(2.0I, k, k))
    R_prior = IWPrior(; Ψ=Matrix(0.3I, p, p), ν=p + 1.0)
    W, R = StateSpaceDynamics._null_fit_regression(Y, X, W_prior, R_prior)
    function g(Rm)
        E = Y .- W * X
        F = cholesky(Symmetric(Rm))
        return -0.5 * (n * logdet(F) + tr(F \ (E * E'))) +
               StateSpaceDynamics.mn_logprior_term(W, Rm, W_prior) +
               StateSpaceDynamics.iw_logprior_term(Rm, R_prior)
    end
    g0 = g(R)
    for _ in 1:4
        G = randn(rng, p, p)
        Δ = (G + G') ./ 2
        @test g(R .+ 1e-4 .* Δ) < g0
        @test g(R .- 1e-4 .* Δ) < g0
    end
    return nothing
end

"""
A partial tie fits the shared columns once, under the first regime's prior, and
counts that prior once in the covariance update and the ELBO as well, so the EM
trace with priors stays monotone and a whole-matrix prior is not counted per
regime.
"""
function test_r2_partial_tie_prior_counted_once()
    ab = StateSpaceDynamics._owned_prior_cols([1, 2], 4, 3, 3)
    @test ab[1] === nothing
    @test ab[2] == [3, 4] && ab[3] == [3, 4]
    @test StateSpaceDynamics._owned_prior_cols(Int[], 4, 3, 3) === nothing
    @test StateSpaceDynamics._owned_prior_cols(collect(1:4), 4, 3, 3) === nothing

    rng = StableRNG(17)
    D, P = 2, 3
    function regime(seed)
        r = StableRNG(seed)
        sm = GaussianStateModel(;
            A=Matrix(0.8I, D, D) .+ 0.05 .* randn(r, D, D),
            Q=Matrix(0.2I, D, D),
            b=0.1 .* randn(r, D),
            x0=zeros(D),
            P0=Matrix(0.5I, D, D),
            AB_prior=MNPrior(; M₀=zeros(D, D + 1), Λ=Matrix(1.0I, D + 1, D + 1)),
            Q_prior=IWPrior(; Ψ=Matrix(0.05I, D, D), ν=D + 2.0),
        )
        om = GaussianObservationModel(;
            C=randn(r, P, D),
            R=Matrix(0.3I, P, P),
            d=zeros(P),
            CD_prior=MNPrior(; M₀=zeros(P, D + 1), Λ=Matrix(1.0I, D + 1, D + 1)),
        )
        return LinearDynamicalSystem(sm, om)
    end
    slds = SLDS(; A=[0.9 0.1; 0.1 0.9], πₖ=[0.5, 0.5], LDSs=[regime(1), regime(2)])
    _, _, y = rand(rng, slds, fill(30, 4))
    trace = fit!(
        slds,
        y;
        max_iter=6,
        tied_params=[:A, :C],
        smoothing_iters=3,
        num_samples=2,
        rng=StableRNG(2),
        progress=false,
    )
    @test all(isfinite, trace)
    @test slds.LDSs[1].state_model.A == slds.LDSs[2].state_model.A
    @test slds.LDSs[1].obs_model.C == slds.LDSs[2].obs_model.C
    # The reported prior counts the shared columns once: removing regime 2's
    # free-block terms and regime 1's whole term leaves nothing.
    tied = [:A, :C]
    total = StateSpaceDynamics._slds_prior_logdensity(slds, tied)
    function mn(lds, which)
        sm, om = lds.state_model, lds.obs_model
        Wab = hcat(sm.A, sm.b, sm.B)
        Wcd = hcat(om.C, om.d, om.D)
        cols = which === :all ? (1:(D + 1)) : [D + 1]
        return StateSpaceDynamics.mn_logprior_term(
            Wab[:, cols], sm.Q, StateSpaceDynamics._restrict_mn_prior(sm.AB_prior, cols)
        ) + StateSpaceDynamics.mn_logprior_term(
            Wcd[:, cols], om.R, StateSpaceDynamics._restrict_mn_prior(om.CD_prior, cols)
        )
    end
    function iw(lds)
        return StateSpaceDynamics.iw_logprior_term(
            lds.state_model.Q, lds.state_model.Q_prior
        )
    end
    expected =
        mn(slds.LDSs[1], :all) +
        mn(slds.LDSs[2], :free) +
        iw(slds.LDSs[1]) +
        iw(slds.LDSs[2])
    @test total ≈ expected rtol = 1e-10
    return nothing
end
