#=============================================================================
Inverse-LQR latents.

The strategy throughout is to check the new code against something that does
not share its implementation:

* the *structure* against the defining identities (`MᵀJM = J`, the mixed/forward
  round trip, `Q^fwd = GΣGᵀ`);
* the *E-step* against an explicit `2n`-dimensional Gaussian LDS carrying the
  materialised forward parameters — which must agree bit for bit;
* the *ELBO* against the exact Laplace normalizer `ℓ(ẑ) + (dT/2)log 2π −
  ½ logdet H`, computed from the per-timestep kernels and a dense Hessian, i.e.
  code the mixed-coordinate `Q_state!` never touches;
* the *M-step objective and gradient* against an independent re-derivation,
  differentiated by ForwardDiff.
=============================================================================#

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

"""A mild plant: `ρ(M)` close to 1, so the forward chain is usable over the
horizons the tests sample at."""
function lqr_fixture(
    rng;
    n::Int=2,
    p::Int=4,
    terminal::Bool=false,
    nregimes::Int=1,
    tsteps::Int=14,
    ux_dim::Int=0,
    observe_costate::Bool=false,
    onset::Int=1,
    condition_terminal::Bool=true,
)
    d = 2n
    A = n == 2 ? [0.96 0.07; -0.05 0.93] : Matrix(0.95I, n, n) + 0.02 .* randn(rng, n, n)
    Sm = n == 2 ? [0.06 0.01; 0.01 0.05] : Matrix(0.05I, n, n)
    pool = [
        n == 2 ? [0.25 0.04; 0.04 0.18] : Matrix(0.2I, n, n),
        n == 2 ? [0.80 0.00; 0.00 0.60] : Matrix(0.6I, n, n),
        n == 2 ? [1.40 0.10; 0.10 1.10] : Matrix(1.1I, n, n),
    ]
    qcs = [copy(pool[k]) for k in 1:nregimes]
    sched = if nregimes == 1 && !terminal
        Int[]
    else
        cost_schedule(tsteps; terminal=terminal, onset=onset, nregimes=nregimes)
    end
    Σ = Matrix(Diagonal(fill(0.03, d)))
    Bu = ux_dim > 0 ? randn(rng, d, ux_dim) : nothing
    sm = LQRStateModel(
        A,
        Sm,
        qcs,
        Σ;
        schedule=sched,
        terminal=terminal,
        condition_terminal=condition_terminal,
        Σf=Matrix(0.04I, n, n),
        hf=terminal ? randn(rng, n) .* 0.05 : nothing,
        h=randn(rng, d) .* 0.05,
        Bu=Bu,
        P0=Matrix(0.25I, d, d),
        x0=randn(rng, d) .* 0.1,
        observe_costate=observe_costate,
    )
    C = randn(rng, p, d)
    observe_costate || (C[:, (n + 1):d] .= 0)
    om = GaussianObservationModel(C, Matrix(0.08I, p, p), randn(rng, p) .* 0.1)
    return sm, LinearDynamicalSystem(sm, om)
end

"""The explicit `2n` Gaussian LDS carrying this model's materialised forward
parameters. Only valid as a reference when there is one cost regime and no
terminal factor — that is exactly when the LQR model reduces to an
ordinary linear-Gaussian chain."""
function lqr_reference_lds(lds)
    sm = lds.state_model
    gsm = GaussianStateModel(
        Matrix(symplectic_matrix(sm)),
        Matrix(sm.cache.Qfwd),
        copy(sm.cache.bfwd),
        copy(sm.x0),
        Matrix(sm.P0),
    )
    # The reference is only valid at one cost regime, so regime 1's forward input
    # matrix is the whole story.
    if size(sm.cache.Bfwd[1], 2) > 0
        gsm.B = copy(sm.cache.Bfwd[1])
    end
    om = lds.obs_model
    gom = GaussianObservationModel(copy(om.C), copy(om.R), copy(om.d))
    return LinearDynamicalSystem(gsm, gom)
end

"""Exact marginal `log p(y, yᵗᵉʳᵐ = 0)` by the Laplace normalizer, built from the
per-timestep kernels and a dense Hessian — independent of `Q_state!`."""
function lqr_exact_marginal(lds, y; ux=nothing, uy=nothing)
    T = Float64
    d = lds.latent_dim
    tsteps = size(y, 2)
    data = SSD.Data(
        lds, [y]; ux=(ux === nothing ? nothing : [ux]), uy=(uy === nothing ? nothing : [uy])
    )
    SSD._prepare_lqr!(lds, data.tsteps)
    tfs = SSD.initialize_FilterSmooth(lds, data.tsteps)
    pool = SSD._lqr_sws_pool(lds, data)
    SSD.smooth!(lds, tfs, data, pool)
    ẑ = tfs[1].x_smooth
    sws = pool[1]
    SSD.compute_smooth_constants!(sws, lds)
    ll = zeros(T, tsteps)
    SSD.joint_loglikelihood!(
        ll, sws, sws.consts, lds, ẑ, y, data.ux[1], SSD._trial(data.uy, 1)
    )
    SSD.hessian!(sws, lds, ẑ, y, SSD._trial(data.uy, 1))
    btd = sws.btd
    H = Matrix(
        block_tridgm(
            [Matrix(-btd.H_diag[t]) for t in 1:tsteps],
            [Matrix(-btd.H_super[i]) for i in 1:(tsteps - 1)],
            [Matrix(-btd.H_sub[i]) for i in 1:(tsteps - 1)],
        ),
    )
    return sum(ll) + 0.5 * d * tsteps * log(2π) - 0.5 * logdet(Symmetric(H))
end

"""Independent `log p(terminal = 0 | inputs)` by *forward* moment propagation.

The implementation under test integrates backwards in square-root form, so this
reference shares no code path with it: it rolls the unconditioned chain's mean
and covariance forward to `T` and evaluates one dense Gaussian density there.
Correct for horizons short enough that the forward covariance stays finite,
which is the regime this reference is used in.
"""
function lqr_terminal_logz_reference(sm, ux, tsteps, ::Type{V}=Float64) where {V}
    n = SSD._plant_dim(sm)
    c = sm.cache
    mu = Vector{V}(sm.x0)
    P = Matrix{V}(sm.P0)
    Q = Matrix{V}(c.Qfwd)
    u = Matrix{V}(ux)
    for t in 1:(tsteps - 1)
        k = SSD._regime(sm, t)
        b = Vector{V}(c.bfwd)
        size(c.Bfwd[k], 2) > 0 && (b += Matrix{V}(c.Bfwd[k]) * u[:, t])
        M = Matrix{V}(c.M[k])
        mu = M * mu + b
        P = M * P * transpose(M) + Q
    end
    kf = SSD._terminal_regime(sm, tsteps)
    Lf = Matrix{V}(c.Lf[kf])
    target = Vector{V}(sm.hf)
    size(c.Ftrm[kf], 2) > 0 && (target -= Matrix{V}(c.Ftrm[kf]) * u[:, tsteps])
    S = Symmetric(Lf * P * transpose(Lf) + Matrix{V}(sm.Σf))
    r = target - Lf * mu
    return -V(0.5) * (n * log(V(2) * V(pi)) + logdet(S) + dot(r, S \ r))
end

"""Independent re-derivation of the structural M-step objective, generic in the
number type so ForwardDiff can differentiate it."""
function lqr_ref_objective(θ::AbstractVector{V}, hs, sm, profile::Bool) where {V}
    pk = SSD._LQRPack(sm)
    n, d, m, K = pk.n, pk.d, pk.m, pk.K
    grab(r, fallback) = isempty(r) ? V.(fallback) : θ[r]
    blk(b) = SSD._lqr_blk(pk, b, 1)
    A = reshape(grab(blk(SSD._LQR_BLOCK_A), vec(sm.A)), n, n)
    psd(r, fallback) =
        if isempty(r)
            V.(fallback)
        else
            L = zeros(V, n, n)
            q = first(r)
            for j in 1:n, i in j:n
                L[i, j] = i == j ? exp(θ[q]) : θ[q]
                q += 1
            end
            L * transpose(L)
        end
    Sm = psd(blk(SSD._LQR_BLOCK_S), sm.S)
    Qs = map(1:K) do k
        return psd(SSD._lqr_blk_q(pk, 1, k), sm.Qc[k])
    end
    h = grab(blk(SSD._LQR_BLOCK_H), sm.h)
    Bu = V.(sm.Bu)
    br = blk(SSD._LQR_BLOCK_B)
    if !isempty(br)
        Bu[pk.brows, pk.bcols] = reshape(θ[br], length(pk.brows), length(pk.bcols))
    end
    Gr = m > 0 ? reshape(grab(blk(SSD._LQR_BLOCK_G), vec(sm.Gref)), n, m) : zeros(V, n, 0)
    hf = grab(blk(SSD._LQR_BLOCK_F), sm.hf)

    R = V.(hs.Yv)
    for k in 1:K
        E = [A -Sm; Qs[k] transpose(A)]
        # Input block: B_u - [0; Q_k G_r], the tracking term tied to this cost.
        Bk = m > 0 ? Bu - vcat(zeros(V, n, m), Qs[k] * Gr) : Bu
        Th = hcat(E, reshape(h, d, 1), Bk)
        TX = Th * transpose(hs.Xv[k])
        R = R - TX - transpose(TX) + Th * hs.Zw[k] * transpose(Th)
    end
    R = (R + transpose(R)) / 2
    N = sum(hs.nk)
    obj = -N * log(abs(det(A)))
    if profile
        if sm.Σ_prior === nothing
            obj += 0.5 * N * logdet(R)
        else
            prior = sm.Σ_prior
            R = R + V.(prior.Ψ)
            Neff = N + prior.ν + d + 1
            obj += 0.5 * Neff * logdet(R)
        end
    else
        obj += 0.5 * dot(inv(Symmetric(V.(sm.Σ))), R)
    end
    if sm.terminal
        Rf = zeros(V, n, n)
        for k in 1:K
            Psi = hcat(-Qs[k], Matrix{V}(I, n, n), reshape(-hf, n, 1))
            m > 0 && (Psi = hcat(Psi, Qs[k] * Gr))
            Rf .+= Psi * V.(hs.Omega[k]) * transpose(Psi)
        end
        Rf = (Rf + transpose(Rf)) / 2
        Nf = sum(hs.term_n)
        obj += if profile
            0.5 * Nf * logdet(Rf)
        else
            0.5 * dot(inv(Symmetric(V.(sm.Σf))), Rf)
        end
    end
    if sm.Qc_prior !== nothing
        for (k, Q) in enumerate(Qs)
            prior = SSD._qc_prior(sm, k)
            prior === nothing && continue
            w = prior.ν + n + 1
            obj += 0.5 * (w * logdet(Q) + tr(V.(prior.Ψ) * inv(Q)))
        end
    end
    return obj
end

"""Run one E-step and return the aggregated statistics, with the mixed blocks
filled — the state the M-step tests need."""
function lqr_estep_stats(lds, ys; ux=nothing)
    data = SSD.Data(lds, ys; ux=ux)
    SSD._prepare_lqr!(lds, data.tsteps)
    tfs = SSD.initialize_FilterSmooth(lds, data.tsteps)
    pool = SSD._lqr_sws_pool(lds, data)
    hs = SSD._initialize_td_sufficient_statistics(Float64, lds, data.tsteps)
    SSD._td_init_const_blocks!(pool[1], lds, data)
    SSD.estep!(lds, hs, tfs, data, pool)
    SSD._fill_mixed_blocks!(hs, lds.state_model)
    return hs, tfs, data, pool
end

# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

function test_lqr_structure()
    rng = StableRNG(1)
    sm, lds = lqr_fixture(rng; nregimes=1)
    n = size(sm.A, 1)

    @test lds.latent_dim == 2n
    @test SSD._plant_dim(sm) == n

    M = symplectic_matrix(sm)
    J = symplectic_form(n)
    @test isapprox(transpose(M) * J * M, J; atol=1e-12)
    @test symplectic_defect(sm) < 1e-12
    @test isapprox(det(M), 1.0; atol=1e-10)

    # The mixed matrix is what the M-step estimates; check its blocks.
    E = lqr_matrix(sm)
    @test E[1:n, 1:n] ≈ sm.A
    @test E[1:n, (n + 1):(2n)] ≈ -sm.S
    @test E[(n + 1):(2n), 1:n] ≈ sm.Qc[1]
    @test E[(n + 1):(2n), (n + 1):(2n)] ≈ transpose(sm.A)

    # `𝓔` maps [x_t; λ_{t+1}] → [x_{t+1}; λ_t]; `M` maps [x_t; λ_t] →
    # [x_{t+1}; λ_{t+1}]. Both must describe the same solution triple.
    x_t = randn(rng, n)
    lam_next = randn(rng, n)
    v = E * [x_t; lam_next]
    @test M * [x_t; v[(n + 1):(2n)]] ≈ [v[1:n]; lam_next]

    # Noise map and the Jacobian identity the M-step depends on.
    G = sm.cache.G
    @test Matrix(sm.cache.Qfwd) ≈ G * Matrix(sm.Σ) * transpose(G)
    @test logdet(Matrix(sm.cache.Qfwd)) ≈ logdet(Matrix(sm.Σ)) - 2 * log(abs(det(sm.A)))
    @test sm.cache.logabsdetA ≈ log(abs(det(sm.A)))
    @test sm.cache.bfwd ≈ G * sm.h

    # G's inverse has the closed form [I S; 0 −Aᵀ].
    Ginv = [Matrix(I, n, n) Matrix(sm.S); zeros(n, n) -transpose(Matrix(sm.A))]
    @test G * Ginv ≈ Matrix(I, 2n, 2n) atol = 1e-10
    return nothing
end

function test_lqr_regimes_and_schedule()
    rng = StableRNG(2)
    sm, _ = lqr_fixture(rng; terminal=true, nregimes=3, tsteps=14, onset=8)
    @test SSD._nregimes(sm) == 3
    @test length(sm.schedule) == 14
    @test sm.schedule[1] == 1 && sm.schedule[7] == 1
    @test sm.schedule[8] == 2 && sm.schedule[13] == 2
    @test sm.schedule[14] == 3
    # Every regime gets its own symplectic transition, and each is symplectic.
    for k in 1:3
        @test symplectic_defect(sm, k) < 1e-12
        @test symplectic_matrix(sm, k)[(size(sm.A, 1) + 1):end, 1:size(sm.A, 1)] ≈
            -sm.cache.AinvT * sm.Qc[k]
    end
    @test symplectic_matrix(sm, 1) != symplectic_matrix(sm, 2)

    # The noise map is regime-independent: only `Q_k` varies across regimes.
    @test cost_schedule(6) == fill(1, 6)
    @test cost_schedule(6; terminal=true) == [1, 1, 1, 1, 1, 2]
    @test cost_schedule(6; terminal=true, onset=4) == [1, 1, 1, 2, 2, 3]
    @test_throws ArgumentError cost_schedule(1)
    @test_throws ArgumentError cost_schedule(6; onset=9)
    @test_throws ArgumentError cost_schedule(6; terminal=true, nregimes=1)

    # Runs partition the transitions of a trial exactly once, in order.
    runs = SSD._regime_runs(sm, 14)
    @test sum(t1 - t0 + 1 for (_, t0, t1) in runs) == 13
    @test first(runs)[2] == 1 && last(runs)[3] == 13
    for (k, t0, t1) in runs, t in t0:t1
        @test SSD._regime(sm, t) == k
    end
    return nothing
end

function test_lqr_construction_errors()
    rng = StableRNG(3)
    n = 2
    d = 2n
    A = [0.96 0.07; -0.05 0.93]
    Sm = [0.06 0.01; 0.01 0.05]
    Qc = [0.25 0.04; 0.04 0.18]
    Σ = Matrix(0.03I, d, d)

    # A singular plant has no forward symplectic representation at all.
    @test_throws SSD.NumericalStabilityError LQRStateModel([1.0 1.0; 1.0 1.0], Sm, Qc, Σ)
    @test_throws SSD.NotSymmetricError LQRStateModel(A, [1.0 0.5; 0.2 1.0], Qc, Σ)
    @test_throws SSD.NotSymmetricError LQRStateModel(A, Sm, [1.0 0.5; 0.2 1.0], Σ)
    # More than one cost matrix needs a schedule saying which timesteps use which.
    @test_throws ArgumentError LQRStateModel(A, Sm, [Qc, Qc], Σ)
    @test_throws ArgumentError LQRStateModel(A, Sm, Qc, Σ; schedule=[1, 2, 1])
    @test_throws SSD.DimensionMismatchError LQRStateModel(A, Sm, Qc, Matrix(0.03I, 3, 3))
    @test_throws SSD.DimensionMismatchError LQRStateModel(A, Sm, Qc, Σ; h=zeros(3))
    @test_throws ArgumentError LQRStateModel(A, Sm, Qc, Σ; mstep_iters=0)

    # Cost priors may be shared or specified in Qc/schedule order, including a
    # separate terminal prior and unregularized epochs.
    run_prior = IWPrior(Matrix(0.4I, n, n), 8.0)
    terminal_prior = IWPrior(Matrix(6.0I, n, n), 12.0)
    epoch_sm = LQRStateModel(
        A,
        Sm,
        [copy(Qc), 2 .* Qc],
        Σ;
        schedule=cost_schedule(3; terminal=true),
        terminal=true,
        Qc_prior=[run_prior, terminal_prior],
    )
    @test SSD._qc_prior(epoch_sm, 1) === run_prior
    @test SSD._qc_prior(epoch_sm, 2) === terminal_prior
    epoch_sm.Qc_prior = [nothing, terminal_prior]
    @test SSD._qc_prior(epoch_sm, 1) === nothing
    @test SSD._qc_prior(epoch_sm, 2) === terminal_prior
    @test_throws SSD.DimensionMismatchError LQRStateModel(
        A,
        Sm,
        [copy(Qc), 2 .* Qc],
        Σ;
        schedule=cost_schedule(3; terminal=true),
        terminal=true,
        Qc_prior=[run_prior],
    )
    @test_throws SSD.DimensionMismatchError LQRStateModel(
        A, Sm, Qc, Σ; Qc_prior=IWPrior(Matrix(1.0I, n + 1, n + 1), 8.0)
    )
    @test_throws ArgumentError LQRStateModel(
        A,
        Sm,
        [copy(Qc), 2 .* Qc],
        Σ;
        schedule=cost_schedule(3; terminal=true),
        terminal=true,
        Qc_prior=[run_prior, :bad],
    )

    # A non-PD Σ is caught when the model goes into an LDS.
    sm_bad = LQRStateModel(A, Sm, Qc, Σ)
    sm_bad.Σ = Matrix(-0.1I, d, d)
    om = GaussianObservationModel(randn(rng, 3, d), Matrix(0.1I, 3, 3), zeros(3))
    @test_throws SSD.NotPositiveDefiniteError LinearDynamicalSystem(sm_bad, om)

    #= A structural piece may be grouped on its own — `(A = …,)` means "a plant
    per group, everything else shared" — but a name the model does not own is
    refused rather than silently ignored. =#
    sm_dep = LQRStateModel(A, Sm, Qc, Σ)
    sm_dep.depends_on = (A=[1, 1, 2],)
    @test LinearDynamicalSystem(sm_dep, om) isa LinearDynamicalSystem
    sm_bad_dep = LQRStateModel(A, Sm, Qc, Σ)
    sm_bad_dep.depends_on = (Σ=[1, 1, 2],)
    @test_throws Exception LinearDynamicalSystem(sm_bad_dep, om)

    # A schedule that does not cover the longest trial is caught at fit entry.
    sm_short = LQRStateModel(A, Sm, [Qc, Qc], Σ; schedule=cost_schedule(5))
    lds_short = LinearDynamicalSystem(
        sm_short, GaussianObservationModel(randn(rng, 3, d), Matrix(0.1I, 3, 3), zeros(3))
    )
    @test_throws SSD.DimensionMismatchError elbo(lds_short, randn(3, 9))
    return nothing
end

function test_lqr_refresh_and_utilities()
    rng = StableRNG(4)
    sm, _ = lqr_fixture(rng; nregimes=1)
    n = size(sm.A, 1)

    # The cache tracks the fields: mutate, refresh, and the transition follows.
    M_before = symplectic_matrix(sm)
    sm.Qc[1] = sm.Qc[1] .* 2
    refresh!(sm)
    @test symplectic_matrix(sm) != M_before
    @test symplectic_defect(sm) < 1e-12

    lp = lqr_parameters(sm)
    @test lp.A ≈ sm.A
    @test lp.S ≈ sm.S
    @test lp.Qc[1] ≈ sm.Qc[1]

    # The Riccati fixed point satisfies its own defining equation.
    P = riccati_solution(sm)
    @test P ≈ transpose(P) atol = 1e-10
    resid = sm.Qc[1] + transpose(sm.A) * P * ((I + sm.S * P) \ sm.A) - P
    @test maximum(abs, resid) < 1e-9
    # `λ = Px` is a fixed point of the LQR flow: the symplectic map sends
    # the graph of P to itself.
    M = symplectic_matrix(sm)
    x = randn(rng, n)
    z_next = M * [x; P * x]
    @test z_next[(n + 1):(2n)] ≈ P * z_next[1:n] atol = 1e-8
    @test closed_loop_dynamics(sm; P=P) ≈ (I + sm.S * P) \ sm.A
    @test maximum(abs, eigvals(closed_loop_dynamics(sm; P=P))) < 1
    return nothing
end

function test_lqr_rescale_costate()
    rng = StableRNG(5)
    sm, lds = lqr_fixture(rng; nregimes=1, tsteps=12)
    y = randn(rng, lds.obs_dim, 12) .* 0.4
    before = elbo(lds, y)
    SQ_before = sm.S * sm.Qc[1]

    # The costate scale is not identified: rescaling leaves the fit untouched.
    rescale_costate!(sm, 2.5)
    @test elbo(lds, y) ≈ before atol = 1e-8
    @test sm.S * sm.Qc[1] ≈ SQ_before atol = 1e-10
    # The sign is unidentified too, and `:trace` canonicalization fixes both.
    rescale_costate!(sm, -1.0)
    @test elbo(lds, y) ≈ before atol = 1e-8
    rescale_costate!(sm; target=:trace)
    @test tr(sm.Qc[1]) ≈ size(sm.A, 1) atol = 1e-10
    @test elbo(lds, y) ≈ before atol = 1e-8
    rescale_costate!(sm; target=:opnorm)
    @test maximum(abs, eigvals(Symmetric(sm.Qc[1]))) ≈ 1 atol = 1e-10
    @test elbo(lds, y) ≈ before atol = 1e-8
    @test_throws ArgumentError rescale_costate!(sm, 0)
    @test_throws ArgumentError rescale_costate!(sm; target=:nonsense)

    #=
    A grouped model. One `c` serves every group — the transformation rescales the
    costate, which they all share — and every array is transformed exactly once,
    which a piece *shared* between groups makes a real question: visiting the
    groups in turn would divide `S` by `c` once per group.
    =#
    smG, ldsG = lqr_fixture(StableRNG(6); nregimes=1, tsteps=12)
    labels = [1, 1, 2, 2]
    set_depends_on!(smG, (Qc=labels,))
    ldsG2 = LinearDynamicalSystem(smG, ldsG.obs_model)
    ys = [randn(StableRNG(7), ldsG2.obs_dim, 12) .* 0.4 for _ in 1:4]
    fit!(ldsG2, ys; max_iter=6, progress=false)
    g1 = group_variant(smG, :Qc, 1)
    g2 = group_variant(smG, :Qc, 2)
    @test g1.S === g2.S                              # shared by reference
    S0, Q1_0, Q2_0 = copy(g1.S), copy(g1.Qc[1]), copy(g2.Qc[1])
    elbo_before = elbo(ldsG2, ys)

    c = 3.0
    rescale_costate!(smG, c)
    @test g1.S ≈ S0 ./ c                             # once, not once per group
    @test g1.Qc[1] ≈ Q1_0 .* c
    @test g2.Qc[1] ≈ Q2_0 .* c
    @test elbo(ldsG2, ys) ≈ elbo_before atol = 1e-7
    return nothing
end

"""
A ragged model of the shape real reaching data takes: one running cost on every
transition, the terminal factor pinned to its own cost (`terminal_regime`), a
one-hot target input, and trial lengths that are all different.
"""
function _pinned_ragged_lqr(rng; n::Int=2, m::Int=3, tmax::Int=40)
    d = 2n
    sm = LQRStateModel(
        Matrix(0.95I, n, n) + 0.02 .* randn(rng, n, n),
        Matrix(0.05I, n, n),
        [Matrix(0.2I, n, n), Matrix(0.9I, n, n)],
        Matrix(0.03I, d, d);
        schedule=fill(1, tmax),
        terminal=true,
        terminal_regime=2,
        Σf=Matrix(0.04I, n, n),
        hf=randn(rng, n) .* 0.05,
        h=randn(rng, d) .* 0.05,
        Bu=randn(rng, d, m),
        Gref=randn(rng, n, m) .* 0.3,
        P0=Matrix(0.25I, d, d),
        x0=randn(rng, d) .* 0.1,
    )
    SSD.refresh!(sm)
    return sm
end

function test_lqr_terminal_normalizer_pinned_ragged()
    # The shared-step path: every horizon's backward recursion is a prefix of
    # the longest one's, so it is computed once.
    rng = StableRNG(20260924)
    sm = _pinned_ragged_lqr(rng)
    designs = [
        (ux=randn(rng, 3, t), count=c) for
        (t, c) in ((11, 2), (14, 1), (11, 3), (40, 2), (14, 4), (2, 1), (27, 5), (39, 1))
    ]
    reference = sum(d.count * SSD._lqr_terminal_logz(sm, d.ux) for d in designs)
    @test SSD._lqr_terminal_logz_sum(sm, designs) ≈ reference rtol = 1e-12
    # A single design, and designs of one horizon only.
    for subset in (designs[4:4], designs[[1, 3]])
        ref = sum(d.count * SSD._lqr_terminal_logz(sm, d.ux) for d in subset)
        @test SSD._lqr_terminal_logz_sum(sm, subset) ≈ ref rtol = 1e-12
    end
    return nothing
end

"""
The terminal-conditioning probe's statistics: the dedicated aggregator against
the two generic weighted aggregators it replaces, and its serial and parallel
chunk schedules against each other bit for bit.
"""
function test_lqr_probe_aggregate_matches_weighted()
    rng = StableRNG(20260925)
    sm = _pinned_ragged_lqr(rng)
    lengths = [7, 12, 12, 19, 25, 25, 25, 31, 40]
    n = SSD.plant_dim(sm)
    lds = LinearDynamicalSystem(
        sm, GaussianObservationModel(randn(rng, 3, 2n), Matrix(0.5I, 3, 3), zeros(3))
    )
    suf = SSD._initialize_td_sufficient_statistics(Float64, lds, lengths)
    for (i, t) in enumerate(lengths)
        u = zeros(3, t)
        u[mod1(i, 3), :] .= 1
        push!(suf.terminal_inputs, u)
        push!(suf.terminal_counts, Float64(1 + i % 4))
    end
    probe = SSD._lqr_terminal_probe(sm, suf)
    SSD._lqr_sync_probe!(probe, sm)
    SSD.smooth!(probe.lds, probe.tfs, probe.data, probe.pool)

    old = deepcopy(probe.hs)
    SSD._aggregate_td_suff_stats_weighted!(
        old.base, probe.tfs, probe.lds, probe.data, probe.weights, probe.pool[1]
    )
    SSD._aggregate_lqr_stats_weighted!(old, probe.tfs, probe.lds, probe.data, probe.weights)
    counts = [Float64(d.count) for d in probe.designs]
    serial = deepcopy(probe.hs)
    SSD._lqr_probe_aggregate!(serial, probe.tfs, probe.lds, probe.data, counts)
    parallel = deepcopy(probe.hs)
    SSD._lqr_probe_aggregate!(
        parallel, probe.tfs, probe.lds, probe.data, counts; serial_below=0
    )

    for k in eachindex(old.zz)
        @test serial.zz[k] ≈ old.zz[k] rtol = 1e-12
        @test serial.zy[k] ≈ old.zy[k] rtol = 1e-12
        @test serial.yy[k] ≈ old.yy[k] rtol = 1e-12
        @test serial.term_zz[k] ≈ old.term_zz[k] rtol = 1e-12
    end
    @test serial.nk ≈ old.nk rtol = 1e-14
    @test serial.term_n ≈ old.term_n rtol = 1e-14
    @test serial.base.init_n ≈ old.base.init_n rtol = 1e-14
    @test serial.base.init_xy ≈ old.base.init_xy rtol = 1e-12
    @test serial.base.init_yy[] ≈ old.base.init_yy[] rtol = 1e-12
    for f in (:zz, :zy, :yy, :term_zz, :nk, :term_n)
        @test getfield(serial, f) == getfield(parallel, f)
    end
    @test serial.base.init_yy[] == parallel.base.init_yy[]
    return nothing
end

"""The probe is built once per fit, and rebuilt when the designs change."""
function test_lqr_probe_cache()
    rng = StableRNG(20260926)
    sm = _pinned_ragged_lqr(rng)
    n = SSD.plant_dim(sm)
    lds = LinearDynamicalSystem(
        sm, GaussianObservationModel(randn(rng, 3, 2n), Matrix(0.5I, 3, 3), zeros(3))
    )
    suf = SSD._initialize_td_sufficient_statistics(Float64, lds, [9, 14])
    for t in (9, 14)
        push!(suf.terminal_inputs, ones(3, t))
        push!(suf.terminal_counts, 1.0)
    end
    p1 = SSD._lqr_terminal_probe_cached(sm, suf)
    # A new E-step records the same designs in fresh arrays: still the same probe.
    empty!(suf.terminal_inputs)
    empty!(suf.terminal_counts)
    for t in (9, 14)
        push!(suf.terminal_inputs, ones(3, t))
        push!(suf.terminal_counts, 1.0)
    end
    @test SSD._lqr_terminal_probe_cached(sm, suf) === p1
    suf.terminal_counts[2] = 2.0
    p2 = SSD._lqr_terminal_probe_cached(sm, suf)
    @test p2 !== p1
    @test p2.designs[2].count == 2.0
    return nothing
end

"""
Ragged smoothing of an LQR model with a terminal factor and inputs — the
terminal-conditioning probe's case — gives the same bits for every workspace
pool size, and again on a second call that reuses the covariance storage.
"""
function test_lqr_ragged_smooth_pool_invariance()
    rng = StableRNG(20260927)
    sm = _pinned_ragged_lqr(rng)
    n = SSD.plant_dim(sm)
    lds = LinearDynamicalSystem(
        sm, GaussianObservationModel(zeros(1, 2n), ones(1, 1), zeros(1))
    )
    lengths = [5, 9, 9, 14, 22, 22, 31, 40, 17]
    uxs = [randn(rng, 3, t) for t in lengths]
    ys = [zeros(1, t) for t in lengths]
    data = SSD.Data(lds, ys; ux=uxs)
    function run(npool)
        tfs = SSD.initialize_FilterSmooth(lds, lengths)
        pool = [
            SSD.SmoothWorkspace(Float64, 2n, 1, maximum(lengths); ux_dim=3) for _ in 1:npool
        ]
        SSD.smooth!(lds, tfs, data, pool)
        first_pass = deepcopy(tfs)
        SSD.smooth!(lds, tfs, data, pool)
        return first_pass, tfs
    end
    _, ref = run(1)
    for npool in (2, 3, 7)
        first_pass, second = run(npool)
        for tfs in (first_pass, second), i in eachindex(lengths)
            @test tfs[i].x_smooth == ref[i].x_smooth
            @test tfs[i].p_smooth == ref[i].p_smooth
            @test tfs[i].p_smooth_tt1[:, :, 2:end] == ref[i].p_smooth_tt1[:, :, 2:end]
            @test tfs[i].entropy == ref[i].entropy
        end
    end
    return nothing
end

"""`_lqr_terminal_logz` against an independent forward-moment reference.

Also pins the relationship the whole conditional objective rests on: with
`condition_terminal` set, the reported score is the joint score minus this
normalizer, exactly.
"""
function test_lqr_terminal_normalizer()
    rng = StableRNG(97)
    # `cost_schedule` derives the regime count: onset > 1 adds a pre-onset epoch,
    # and the terminal factor always adds one of its own.
    for (nregimes, ux_dim, onset, tsteps) in
        ((2, 0, 1, 2), (2, 0, 1, 7), (3, 2, 5, 11), (3, 3, 4, 20))
        sm, lds = lqr_fixture(
            rng;
            terminal=true,
            nregimes=nregimes,
            tsteps=tsteps,
            ux_dim=ux_dim,
            onset=onset,
            condition_terminal=false,
        )
        ux = ux_dim > 0 ? randn(rng, ux_dim, tsteps) : zeros(0, tsteps)
        # A nonzero reference gain exercises the `Ftrm` term of the target.
        if ux_dim > 0
            sm.Gref .= randn(rng, size(sm.Gref)...) .* 0.3
            SSD.refresh!(sm)
        end
        SSD._prepare_lqr!(lds, [tsteps])
        @test SSD._lqr_terminal_logz(sm, ux) ≈ lqr_terminal_logz_reference(sm, ux, tsteps) atol =
            1e-9

        # The conditional score is the joint score minus exactly that number.
        y = randn(rng, lds.obs_dim, tsteps) .* 0.4
        uxarg = ux_dim > 0 ? ux : nothing
        joint = elbo(lds, y; ux=uxarg)
        sm.condition_terminal = true
        @test elbo(lds, y; ux=uxarg) ≈ joint - SSD._lqr_terminal_logz(sm, ux) atol = 1e-9
        sm.condition_terminal = false
    end

    #=
    The reason for integrating backwards at all. A symplectic transition has
    reciprocal eigenvalues, so the forward covariance of a long chain overflows
    Float64 while the terminal density itself stays perfectly ordinary. The
    reference above cannot reach this case; the implementation must.
    =#
    n = 1
    sm = LQRStateModel(
        fill(0.35, 1, 1),                 # ‖A⁻¹‖ ≫ 1: the costate block explodes
        fill(0.05, 1, 1),
        [fill(0.2, 1, 1)],
        Matrix(0.03I, 2, 2);
        terminal=true,
        condition_terminal=false,
        Σf=Matrix(0.04I, n, n),
        P0=Matrix(0.25I, 2, 2),
    )
    SSD.refresh!(sm)
    long = 200
    @test maximum(abs, SSD.symplectic_matrix(sm)) > 2      # genuinely explosive
    value = SSD._lqr_terminal_logz(sm, zeros(0, long))
    @test isfinite(value)
    @test SSD._lqr_terminal_logz_sum(sm, [(ux=zeros(0, long), count=2)]) ≈ 2value rtol =
        1e-10
    #=
    Float64 forward propagation of the same chain reaches ~1e182 by the endpoint
    and its symmetric eigenvalues straddle zero there, so the dense reference
    above cannot referee this case — `logdet` on it does not merely lose
    precision, it reports a negative determinant.
    =#
    P = Matrix{Float64}(sm.P0)
    M = symplectic_matrix(sm)
    for _ in 2:long
        P = M * P * transpose(M) + Matrix(sm.cache.Qfwd)
    end
    @test maximum(abs, P) > 1e150
    @test minimum(eigvals(Symmetric(P))) < 0
    # At 512 bits it does not, and it agrees with the square-root integration.
    reference = setprecision(BigFloat, 512) do
        return lqr_terminal_logz_reference(sm, zeros(0, long), long, BigFloat)
    end
    @test isfinite(reference)
    @test value ≈ Float64(reference) rtol = 1e-9
    return nothing
end

function test_lqr_terminal_normalizer_shared_horizons()
    rng = StableRNG(20260923)
    sm, lds = lqr_fixture(rng; terminal=true, nregimes=3, tsteps=18, ux_dim=2, onset=6)
    sm.Gref .= randn(rng, size(sm.Gref)...) .* 0.2
    SSD.refresh!(sm)
    SSD._prepare_lqr!(lds, [11, 14, 18])
    designs = [
        (ux=randn(rng, 2, t), count=count) for
        (t, count) in ((11, 2), (14, 1), (11, 3), (18, 2), (14, 4))
    ]
    reference = sum(d.count * SSD._lqr_terminal_logz(sm, d.ux) for d in designs)
    @test SSD._lqr_terminal_logz_sum(sm, designs) ≈ reference atol = 1e-9
    return nothing
end

"""Dimension and costate gauge: what the joint score confounds and the
conditional score does not.

The joint score `log p(y, terminal = 0)` moves when the model gains a plant
dimension the emission never reads, and moves again under the inverse-optimal-
control rescaling that leaves the plant posterior alone. Neither is a change in
how well the model explains `y`, which is why a dimension sweep run on that
score is not a comparison of fits. Conditioning removes both exactly.
"""
function test_lqr_dimension_and_terminal_score()
    y = reshape([0.2, -0.1, 0.3, 0.0, -0.2, 0.1], 1, :)
    T = size(y, 2)
    function model(n, terminal; condition=true)
        sm = LQRStateModel(
            Matrix(0.95I, n, n),
            Matrix(0.05I, n, n),
            [Matrix(0.2I, n, n)],
            Matrix(0.03I, 2n, 2n);
            terminal=terminal,
            condition_terminal=condition,
            Σf=Matrix(0.04I, n, n),
            P0=Matrix(0.25I, 2n, 2n),
        )
        C = zeros(1, 2n)
        C[1, 1] = 1.0
        return LinearDynamicalSystem(
            sm, GaussianObservationModel(C, fill(0.08, 1, 1), zeros(1))
        )
    end
    # Without a terminal factor the unused pair already integrates out.
    @test elbo(model(2, false), y) ≈ elbo(model(1, false), y) atol = 1e-8

    # --- the joint score, which is what the dimension sweep was scored on ---
    small = model(1, true; condition=false)
    large = model(2, true; condition=false)
    # Independently propagate the unobserved pair's prior to the endpoint.
    sm = small.state_model
    M, Q = symplectic_matrix(sm), sm.cache.Qfwd
    P = copy(sm.P0)
    for _ in 2:T
        P = M * P * M' + Q
    end
    H = hcat(-sm.Qc[1], ones(1, 1))
    variance = only(H * P * H' + sm.Σf)
    terminal_logdensity = -0.5 * log(2π * variance)
    @test elbo(large, y) - elbo(small, y) ≈ terminal_logdensity atol = 1e-8

    ys = [y, y]
    before = elbo(large, ys)
    x_before, _ = smooth(large, ys)
    rescale_costate!(large.state_model, 2.5)
    @test elbo(large, ys) - before ≈ -length(ys) * 2 * log(2.5) atol = 1e-8
    x_after, _ = smooth(large, ys)
    @test x_after[1][1:2, :] ≈ x_before[1][1:2, :] atol = 1e-8

    # --- the conditional score: both confounds vanish, to machine precision ---
    csmall, clarge = model(1, true), model(2, true)
    @test elbo(clarge, y) ≈ elbo(csmall, y) atol = 1e-10
    @test elbo(clarge, ys) ≈ elbo(csmall, ys) atol = 1e-10

    #=
    `rescale_costate!` already sends `Σf → c²Σf` and `hf → c·hf`, so the whitened
    terminal residual — and hence the conditioning event — is untouched. The
    invariance is therefore exact, not approximate.
    =#
    gauged = elbo(clarge, ys)
    rescale_costate!(clarge.state_model, 2.5)
    @test elbo(clarge, ys) ≈ gauged atol = 1e-10
    xg_after, _ = smooth(clarge, ys)
    @test xg_after[1][1:2, :] ≈ x_before[1][1:2, :] atol = 1e-8

    # A negative gauge is a symmetry too, and conditioning is blind to it.
    rescale_costate!(clarge.state_model, -0.4)
    @test elbo(clarge, ys) ≈ gauged atol = 1e-10
    return nothing
end

# ---------------------------------------------------------------------------
# E-step: equivalence to an explicit 2n Gaussian LDS
# ---------------------------------------------------------------------------

function test_lqr_reduces_to_gaussian_lds()
    rng = StableRNG(11)
    for ux_dim in (0, 2)
        sm, lds = lqr_fixture(rng; nregimes=1, terminal=false, ux_dim=ux_dim, tsteps=18)
        ref = lqr_reference_lds(lds)
        tsteps = 18
        y = randn(rng, lds.obs_dim, tsteps) .* 0.4
        ux = ux_dim > 0 ? randn(rng, ux_dim, tsteps) : nothing

        xh, ph = smooth(lds, y; ux=ux)
        xg, pg = smooth(ref, y; ux=ux)
        #=
        Exact equality, not approximate: with one cost regime and no terminal
        factor the LQR kernels evaluate the very same arithmetic on the
        very same materialised matrices, so any difference at all would mean a
        different formula, not rounding.
        =#
        @test xh == xg
        @test ph == pg

        @test elbo(lds, y; ux=ux) ≈ elbo(ref, y; ux=ux) atol = 1e-9
        # And against the independent Kalman-filter marginal likelihood.
        @test loglikelihood(lds, y; ux=ux) ≈ loglikelihood(ref, y; ux=ux) atol = 1e-8
    end
    return nothing
end

function test_lqr_multitrial_equivalence()
    rng = StableRNG(12)
    sm, lds = lqr_fixture(rng; nregimes=1, terminal=false, tsteps=16)
    ref = lqr_reference_lds(lds)
    ys = [randn(rng, lds.obs_dim, 16) .* 0.4 for _ in 1:5]
    # Equal-length trials take the shared-covariance + batched mean fast path.
    @test elbo(lds, ys) ≈ elbo(ref, ys) atol = 1e-8
    xh, _ = smooth(lds, ys)
    xg, _ = smooth(ref, ys)
    @test maximum(maximum.(abs, xh .- xg)) < 1e-10

    # Ragged trials must agree with the explicit Gaussian model too.
    yr = [randn(rng, lds.obs_dim, t) .* 0.4 for t in (11, 16, 13)]
    @test elbo(lds, yr) ≈ elbo(ref, yr) atol = 1e-8
    return nothing
end

function test_lqr_ragged_shared_cov_matches_per_trial()
    rng = StableRNG(20260924)
    sm, lds = lqr_fixture(rng; terminal=true, nregimes=3, tsteps=16, ux_dim=2, onset=5)
    lengths = [11, 16, 11, 13, 16]
    ys = [randn(rng, lds.obs_dim, t) .* 0.4 for t in lengths]
    uxs = [randn(rng, 2, t) .* 0.2 for t in lengths]
    data = SSD.Data(lds, ys; ux=uxs)
    SSD._prepare_lqr!(lds, data.tsteps)
    tfs = SSD.initialize_FilterSmooth(lds, data.tsteps)
    pool = SSD._lqr_sws_pool(lds, data)
    SSD.smooth!(lds, tfs, data, pool)
    @test tfs[1].p_smooth === tfs[3].p_smooth
    @test tfs[2].p_smooth === tfs[5].p_smooth
    for i in eachindex(lengths)
        ref = SSD.initialize_FilterSmooth(lds, [lengths[i]])[1]
        SSD.smooth!(lds, ref, ys[i], pool[1], uxs[i], data.uy[i])
        @test tfs[i].x_smooth ≈ ref.x_smooth atol = 1e-9
        @test tfs[i].p_smooth ≈ ref.p_smooth atol = 1e-9
        @test tfs[i].p_smooth_tt1 ≈ ref.p_smooth_tt1 atol = 1e-9
        @test tfs[i].entropy ≈ ref.entropy atol = 1e-9
    end
    return nothing
end

function test_lqr_batched_gradient_matches_per_trial()
    rng = StableRNG(13)
    # The batched mean pass is the equal-length fast path; check it against the
    # per-trial kernel on a model exercising regimes, a terminal factor and inputs.
    sm, lds = lqr_fixture(rng; terminal=true, nregimes=3, tsteps=15, onset=9, ux_dim=2)
    ntrials = 4
    ys = [randn(rng, lds.obs_dim, 15) .* 0.4 for _ in 1:ntrials]
    uxs = [randn(rng, 2, 15) for _ in 1:ntrials]
    data = SSD.Data(lds, ys; ux=uxs)
    SSD._prepare_lqr!(lds, data.tsteps)

    ws = SSD.SmoothWorkspace(
        Float64, lds.latent_dim, lds.obs_dim, 15; ux_dim=2, ntrials=ntrials
    )
    SSD.compute_smooth_constants!(ws, lds)
    SSD._populate_batched_data!(ws, data)
    bat = ws.batched
    xs = [randn(rng, lds.latent_dim, 15) .* 0.3 for _ in 1:ntrials]
    for n in 1:ntrials
        bat.x_mat[:, :, n] .= xs[n]
    end
    SSD.gradient_batched!(ws, lds, bat.x_mat, bat.y, bat.ux, bat.uy)

    ws1 = SSD.SmoothWorkspace(Float64, lds.latent_dim, lds.obs_dim, 15; ux_dim=2)
    SSD.compute_smooth_constants!(ws1, lds)
    for n in 1:ntrials
        g1 = SSD.gradient!(ws1, lds, xs[n], ys[n], uxs[n], data.uy[n])
        @test maximum(abs, g1 .- bat.grad_buf[:, :, n]) < 1e-10
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Kernels against finite differences / the exact normalizer
# ---------------------------------------------------------------------------

function test_lqr_gradient_and_hessian()
    rng = StableRNG(21)
    for (terminal, nregimes, ux_dim) in ((false, 1, 0), (true, 2, 0), (true, 3, 2))
        tsteps = 12
        sm, lds = lqr_fixture(
            rng;
            terminal=terminal,
            nregimes=nregimes,
            tsteps=tsteps,
            ux_dim=ux_dim,
            onset=(nregimes == 3 ? 7 : 1),
        )
        y = randn(rng, lds.obs_dim, tsteps) .* 0.4
        ux = ux_dim > 0 ? randn(rng, ux_dim, tsteps) : zeros(0, tsteps)
        uy = zeros(0, tsteps)
        x = randn(rng, lds.latent_dim, tsteps) .* 0.3

        ws = SSD.SmoothWorkspace(
            Float64, lds.latent_dim, lds.obs_dim, tsteps; ux_dim=ux_dim
        )
        SSD.compute_smooth_constants!(ws, lds)

        function objective(xv)
            xm = reshape(xv, lds.latent_dim, tsteps)
            ll = zeros(eltype(xv), tsteps)
            cc = SSD.SmoothConstants(eltype(xv), lds.latent_dim, lds.obs_dim)
            SSD.compute_smooth_constants!(cc, lds)
            wsx = SSD.SmoothWorkspace(
                eltype(xv), lds.latent_dim, lds.obs_dim, tsteps; ux_dim=ux_dim
            )
            SSD.joint_loglikelihood!(ll, wsx, cc, lds, xm, y, ux, uy)
            return sum(ll)
        end

        g = SSD.gradient!(ws, lds, x, y, ux, uy)
        g_fd = ForwardDiff.gradient(objective, vec(x))
        @test maximum(abs, vec(Matrix(g)) .- g_fd) < 1e-7

        SSD.hessian!(ws, lds, x, y, uy)
        btd = ws.btd
        H = Matrix(
            block_tridgm(
                [Matrix(btd.H_diag[t]) for t in 1:tsteps],
                [Matrix(btd.H_super[i]) for i in 1:(tsteps - 1)],
                [Matrix(btd.H_sub[i]) for i in 1:(tsteps - 1)],
            ),
        )
        H_fd = ForwardDiff.hessian(objective, vec(x))
        @test maximum(abs, H .- H_fd) < 1e-6
        # The negated Hessian is the posterior precision; it must be PD, which is
        # what makes the block-tridiagonal SPD solver applicable.
        @test isposdef(Symmetric(-H))
    end
    return nothing
end

function test_lqr_elbo_matches_exact_marginal()
    rng = StableRNG(22)
    for (terminal, nregimes, ux_dim, onset) in
        ((false, 1, 0, 1), (true, 2, 0, 1), (true, 3, 2, 8), (false, 2, 2, 6))
        tsteps = 14
        sm, lds = lqr_fixture(
            rng;
            terminal=terminal,
            nregimes=nregimes,
            tsteps=tsteps,
            ux_dim=ux_dim,
            onset=onset,
            #= The reference is the *joint* marginal `log p(y, terminal = 0)`:
            it comes from the same Laplace/Gaussian factorization the model
            writes down, and knows nothing of the terminal normalizer. Compare
            against the joint score here, and against the conditional one
            below. =#
            condition_terminal=false,
        )
        y = randn(rng, lds.obs_dim, tsteps) .* 0.4
        ux = ux_dim > 0 ? randn(rng, ux_dim, tsteps) : nothing
        exact = lqr_exact_marginal(lds, y; ux=ux)
        #=
        `elbo` reaches this through the mixed-coordinate `Q_state!` — aggregated
        sufficient statistics, the rearrangement into (w, v), and the
        `−2log|det A|` Jacobian — while the reference goes through the
        per-timestep kernels and a dense log-determinant. Agreement means both
        the rearrangement and the Jacobian term are right.
        =#
        @test elbo(lds, y; ux=ux) ≈ exact atol = 1e-7
        @test loglikelihood(lds, y; ux=ux) ≈ exact atol = 1e-7

        #=
        And with conditioning on, the same two entry points report the exact
        conditional marginal `log p(y | terminal = 0)` — the joint reference
        above less a normalizer computed by forward moment propagation, which
        shares no code with the backward square-root integration under test.
        =#
        terminal || continue
        sm.condition_terminal = true
        u = ux === nothing ? zeros(0, tsteps) : ux
        conditional = exact - lqr_terminal_logz_reference(sm, u, tsteps)
        @test elbo(lds, y; ux=ux) ≈ conditional atol = 1e-7
        @test loglikelihood(lds, y; ux=ux) ≈ conditional atol = 1e-7
        sm.condition_terminal = false
    end
    return nothing
end

function test_lqr_sufficient_statistics()
    rng = StableRNG(23)
    tsteps = 13
    sm, lds = lqr_fixture(rng; terminal=true, nregimes=3, tsteps=tsteps, onset=8, ux_dim=2)
    ntrials = 3
    ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:ntrials]
    uxs = [randn(rng, 2, tsteps) for _ in 1:ntrials]
    hs, tfs, data, _ = lqr_estep_stats(lds, ys; ux=uxs)

    d = lds.latent_dim
    K = SSD._nregimes(sm)
    reg = d + 1 + 2

    # Brute-force the same aggregates from the smoother output, per timestep.
    zz = [zeros(reg, reg) for _ in 1:K]
    zy = [zeros(reg, d) for _ in 1:K]
    yy = [zeros(d, d) for _ in 1:K]
    nk = zeros(K)
    term = [zeros(reg, reg) for _ in 1:K] # terminal regressor is [z_T; 1; u_T]
    for n in 1:ntrials
        fs = tfs[n]
        x = fs.x_smooth
        P = fs.p_smooth
        Ptt1 = fs.p_smooth_tt1
        for t in 1:(tsteps - 1)
            k = SSD._regime(sm, t)
            zt = [x[:, t]; 1.0; uxs[n][:, t]]
            Ezz = zt * zt'
            Ezz[1:d, 1:d] .+= P[:, :, t]
            zz[k] .+= Ezz
            Ezy = zt * x[:, t + 1]'
            Ezy[1:d, :] .+= Ptt1[:, :, t + 1]'
            zy[k] .+= Ezy
            yy[k] .+= x[:, t + 1] * x[:, t + 1]' .+ P[:, :, t + 1]
            nk[k] += 1
        end
        zt = [x[:, tsteps]; 1.0; uxs[n][:, tsteps]]
        Ez = zt * zt'
        Ez[1:d, 1:d] .+= P[:, :, tsteps]
        term[SSD._regime(sm, tsteps)] .+= Ez
    end
    for k in 1:K
        @test maximum(abs, hs.zz[k] .- zz[k]) < 1e-10
        @test maximum(abs, hs.zy[k] .- zy[k]) < 1e-10
        @test maximum(abs, hs.yy[k] .- yy[k]) < 1e-10
        @test hs.nk[k] ≈ nk[k]
    end
    @test all(maximum(abs, hs.term_zz[k] .- term[k]) < 1e-10 for k in 1:K)
    @test sum(hs.term_n) ≈ ntrials
    @test hs.term_n[SSD._regime(sm, tsteps)] ≈ ntrials

    # The mixed blocks must be the second moments of (w, v) built the direct way.
    n = SSD._plant_dim(sm)
    Sww = [zeros(d, d) for _ in 1:K]
    Svw = [zeros(d, d) for _ in 1:K]
    Svv = zeros(d, d)
    for nn in 1:ntrials
        fs = tfs[nn]
        x = fs.x_smooth
        P = fs.p_smooth
        Ptt1 = fs.p_smooth_tt1
        for t in 1:(tsteps - 1)
            k = SSD._regime(sm, t)
            # Joint second moment of (z_t, z_{t+1}) with the smoother covariances.
            Ezz_t = x[:, t] * x[:, t]' .+ P[:, :, t]
            Ezz_n = x[:, t + 1] * x[:, t + 1]' .+ P[:, :, t + 1]
            Ecross = x[:, t] * x[:, t + 1]' .+ Ptt1[:, :, t + 1]'   # E[z_t z_{t+1}ᵀ]
            Jt = [Ezz_t Ecross; Ecross' Ezz_n]                       # 2d × 2d
            # w = [x_t; λ_{t+1}], v = [x_{t+1}; λ_t] as selections of [z_t; z_{t+1}]
            Sel_w = zeros(d, 2d)
            Sel_v = zeros(d, 2d)
            Sel_w[1:n, 1:n] .= I(n)
            Sel_w[(n + 1):d, (d + n + 1):(2d)] .= I(n)
            Sel_v[1:n, (d + 1):(d + n)] .= I(n)
            Sel_v[(n + 1):d, (n + 1):d] .= I(n)
            Sww[k] .+= Sel_w * Jt * Sel_w'
            Svw[k] .+= Sel_v * Jt * Sel_w'
            Svv .+= Sel_v * Jt * Sel_v'
        end
    end
    for k in 1:K
        @test maximum(abs, hs.Zw[k][1:d, 1:d] .- Sww[k]) < 1e-9
        @test maximum(abs, hs.Xv[k][:, 1:d] .- Svw[k]) < 1e-9
    end
    @test maximum(abs, hs.Yv .- Svv) < 1e-9
    return nothing
end

# ---------------------------------------------------------------------------
# M-step
# ---------------------------------------------------------------------------

function test_lqr_mstep_objective_and_gradient()
    rng = StableRNG(31)
    tsteps = 15
    sm, lds = lqr_fixture(rng; terminal=true, nregimes=3, tsteps=tsteps, onset=9, ux_dim=2)
    ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:4]
    uxs = [randn(rng, 2, tsteps) for _ in 1:4]
    hs, _, _, _ = lqr_estep_stats(lds, ys; ux=uxs)
    n = size(sm.A, 1)
    d = 2n
    per_epoch_prior = SSD._normalize_qc_prior(
        Float64,
        [IWPrior(Matrix(0.2I, n, n), 7.0), nothing, IWPrior(Matrix(1.2I, n, n), 15.0)],
        length(sm.Qc),
        n,
    )

    for profile in (true, false)
        for (sigma_prior, qc_prior) in (
            (nothing, nothing),
            (IWPrior(Matrix(0.3I, d, d), 12.0), nothing),
            (nothing, IWPrior(Matrix(0.4I, n, n), 9.0)),
            (IWPrior(Matrix(0.3I, d, d), 12.0), IWPrior(Matrix(0.4I, n, n), 9.0)),
            (IWPrior(Matrix(0.3I, d, d), 12.0), per_epoch_prior),
        )
            sm.Σ_prior = sigma_prior
            sm.Qc_prior = qc_prior
            ctx = SSD._LQRMStepCtx(hs, sm, profile)
            θ = zeros(ctx.pack.np)
            SSD._lqr_pack!(θ, ctx)
            g = similar(θ)
            f = SSD._lqr_fg!(g, θ, ctx)
            @test f ≈ lqr_ref_objective(θ, hs, sm, profile) atol = 1e-8
            g_fd = ForwardDiff.gradient(t -> lqr_ref_objective(t, hs, sm, profile), θ)
            @test maximum(abs, g .- g_fd) / max(1.0, maximum(abs, g_fd)) < 1e-8
        end
    end
    sm.Σ_prior = nothing
    sm.Qc_prior = nothing

    # Arbitrary optimizer iterates must remain valid costs/control authority.
    # Also check the gradient away from the initial square-root factors.
    ctx = SSD._LQRMStepCtx(hs, sm, true)
    θ = zeros(ctx.pack.np)
    SSD._lqr_pack!(θ, ctx)
    SSD._lqr_unpack!(ctx, θ)
    @test ctx.S[1] ≈ sm.S
    @test all(ctx.Qc[1][k] ≈ sm.Qc[k] for k in eachindex(sm.Qc))
    θ .+= 0.1 .* randn(rng, length(θ))
    g = similar(θ)
    SSD._lqr_fg!(g, θ, ctx)
    g_fd = ForwardDiff.gradient(t -> lqr_ref_objective(t, hs, sm, true), θ)
    @test maximum(abs, g .- g_fd) / max(1.0, maximum(abs, g_fd)) < 1e-8
    @test minimum(eigvals(Symmetric(ctx.S[1]))) >= -1e-12
    for Q in ctx.Qc[1]
        @test minimum(eigvals(Symmetric(Q))) >= -1e-12
    end
    # Regression: the reported real-data fit escaped to a negative-definite S.
    sm.S .*= -1
    @test_throws ArgumentError SSD._lqr_pack!(θ, ctx)
    sm.S .*= -1
    sm.Qc[1] .*= -1
    @test_throws ArgumentError SSD._lqr_pack!(θ, ctx)
    return nothing
end

function test_lqr_mstep_freezing()
    rng = StableRNG(32)
    tsteps = 14
    sm, lds = lqr_fixture(rng; terminal=true, nregimes=2, tsteps=tsteps, ux_dim=2)
    ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:4]
    uxs = [randn(rng, 2, tsteps) for _ in 1:4]
    hs, _, _, _ = lqr_estep_stats(lds, ys; ux=uxs)

    n = SSD._plant_dim(sm)
    full = SSD._LQRPack(sm).np

    sm.fit_flags = LQRFitFlags(; A=false, S=false)
    frozen = SSD._LQRPack(sm)
    # Freezing shrinks the problem rather than projecting its solution.
    @test frozen.np == full - n * n - n * (n + 1) ÷ 2
    @test isempty(SSD._lqr_blk(frozen, SSD._LQR_BLOCK_A, 1)) &&
        isempty(SSD._lqr_blk(frozen, SSD._LQR_BLOCK_S, 1))
    ctx = SSD._LQRMStepCtx(hs, sm, true)
    θ = zeros(ctx.pack.np)
    SSD._lqr_pack!(θ, ctx)
    g = similar(θ)
    SSD._lqr_fg!(g, θ, ctx)
    g_fd = ForwardDiff.gradient(t -> lqr_ref_objective(t, hs, sm, true), θ)
    @test maximum(abs, g .- g_fd) < 1e-8

    # A frozen parameter comes back unchanged from a whole fit.
    A0 = copy(sm.A)
    S0 = copy(sm.S)
    Q0 = copy(sm.Qc[1])
    fit!(lds, ys; ux=uxs, max_iter=4, progress=false)
    @test sm.A == A0
    @test sm.S == S0
    @test sm.Qc[1] != Q0

    # Everything frozen: the structural M-step is a no-op.
    #=
    `terminal` gates the terminal factor's own offset `hf`; the terminal cost
    matrix itself is one of the `Qc` and moves with that flag. Freeze both to get
    an empty structural problem.
    =#
    sm.fit_flags = LQRFitFlags(;
        A=false, S=false, Qc=false, h=false, Bu=false, Gref=false, terminal=false
    )
    @test SSD._LQRPack(sm).np == 0
    before = deepcopy(sm.Qc)
    fit!(lds, ys; ux=uxs, max_iter=3, progress=false)
    @test all(sm.Qc[k] == before[k] for k in eachindex(before))
    return nothing
end

function test_lqr_mstep_preserves_structure()
    rng = StableRNG(33)
    tsteps = 14
    sm, lds = lqr_fixture(rng; terminal=true, nregimes=2, tsteps=tsteps)
    ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:6]
    fit!(lds, ys; max_iter=10, progress=false)
    # The point of the whole exercise: after fitting, the transition is still
    # symplectic and the cost matrices still symmetric.
    @test sm.S ≈ transpose(sm.S) atol = 1e-12
    @test minimum(eigvals(Symmetric(sm.S))) >= -1e-12
    for k in 1:SSD._nregimes(sm)
        @test sm.Qc[k] ≈ transpose(sm.Qc[k]) atol = 1e-12
        @test minimum(eigvals(Symmetric(sm.Qc[k]))) >= -1e-12
        @test symplectic_defect(sm, k) < 1e-9
    end
    @test issymmetric(Symmetric(sm.Σ))
    @test isposdef(Symmetric(Matrix(sm.Σ)))
    @test isposdef(Symmetric(Matrix(sm.Σf)))
    return nothing
end

function test_lqr_em_monotone()
    rng = StableRNG(34)
    for (terminal, nregimes, ux_dim, ntrials, onset) in
        ((false, 1, 0, 1, 1), (false, 1, 0, 6, 1), (true, 2, 0, 5, 1), (true, 3, 2, 4, 8))
        tsteps = 14
        sm, lds = lqr_fixture(
            rng;
            terminal=terminal,
            nregimes=nregimes,
            tsteps=tsteps,
            ux_dim=ux_dim,
            onset=onset,
        )
        ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:ntrials]
        uxs = ux_dim > 0 ? [randn(rng, ux_dim, tsteps) for _ in 1:ntrials] : nothing
        els = fit!(lds, ys; ux=uxs, max_iter=25, progress=false)
        @test length(els) >= 2
        # A generalized M-step: the ELBO may not increase much, but it must never
        # decrease.
        @test minimum(diff(els)) > -1e-8
        @test els[end] > els[1]
        @test all(isfinite, els)
        # The returned parameters are the ones scored by the final trace entry.
        @test els[end] ≈ elbo(lds, ys; ux=uxs) atol = 1e-7
    end
    return nothing
end

"""The terminal-conditioned M-step's gradient, against central differences.

Worth checking by hand because it is assembled from three pieces that cannot be
read off one expression: `_lqr_fg!` on the data statistics, the same routine on
the probe's terminal-conditioned prior moments (Fisher's identity for
`∂ log Z / ∂θ`), and closed forms for the `Σ`, `Σf`, `x0` and `P0` blocks that
the structural objective holds fixed. A sign error in any one of them still
leaves a plausible-looking objective.
"""
function test_lqr_conditional_mstep_gradient()
    rng = StableRNG(88)
    tsteps, ux_dim = 9, 2
    sm, lds = lqr_fixture(
        rng;
        terminal=true,
        nregimes=3,
        tsteps=tsteps,
        ux_dim=ux_dim,
        onset=5,
        condition_terminal=true,
    )
    sm.Gref .= randn(rng, size(sm.Gref)...) .* 0.3
    refresh!(sm)
    ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:4]
    # Two designs, each shared by two trials: exercises the dedup and the counts.
    pair = [randn(rng, ux_dim, tsteps) for _ in 1:2]
    uxs = [pair[1], pair[2], pair[1], pair[2]]
    hs, _, _, _ = lqr_estep_stats(lds, ys; ux=uxs)
    @test length(SSD._lqr_terminal_designs(hs)) == 2
    @test hs.terminal_counts == [2.0, 2.0]

    slots = [ones(Int, 1) for _ in 1:4]
    problem = SSD._lqr_conditional_problem([lds], [hs], slots)
    θ = copy(problem.theta)
    @test !isempty(θ)
    analytic = zeros(length(θ))
    @test isfinite(problem.evaluate!(analytic, copy(θ)))

    fd = similar(analytic)
    for i in eachindex(θ)
        step = 1e-6 * max(1.0, abs(θ[i]))
        up, down = copy(θ), copy(θ)
        up[i] += step
        down[i] -= step
        fd[i] =
            (problem.evaluate!(nothing, up) - problem.evaluate!(nothing, down)) / (2 * step)
    end
    problem.write!(θ)          # leave the model where it started
    @test maximum(abs, analytic .- fd) / max(1.0, maximum(abs, fd)) < 1e-7
    return nothing
end

"""The shaping class the `LQRStateModel` docstring describes is an exact symmetry
of the likelihood, and a genuine change of cost.

With control entering through one channel of three, any symmetric `M` with
`S M = 0` gives `Q_run → Q_run + M − AᵀMA`, `Q_term → Q_term + M`, which with the
induced change of costate (`Σ`, `h`, `x0`, `P0` transformed to match) leaves the
closed loop and the ELBO unchanged — joint, terminal-conditioned, and with no
terminal factor at all — while the cost moves by an amount no fit can see. This
is why a fitted `Qc` is reported modulo the class.
"""
function test_lqr_shaping_symmetry()
    rng = StableRNG(9)
    n, p, tsteps = 3, 5, 25
    A = [0.97 0.08 0.0; -0.06 0.95 0.04; 0.02 0.0 0.93]
    b = [1.0, 0.3, 0.0]
    S = 0.08 .* (b * b')                     # rank 1, so the class has dimension 3
    Qrun = [0.3 0.05 0.0; 0.05 0.25 0.02; 0.0 0.02 0.2]
    Qterm = Matrix(1.5I, n, n)
    Σ = Matrix(Diagonal(fill(0.03, 2n))) .+ 0.005
    C = randn(rng, p, 2n)
    C[:, (n + 1):end] .= 0
    # null(S), written out: `nullspace` picks its basis differently across versions.
    N = hcat(normalize([-0.3, 1.0, 0.0]), [0.0, 0.0, 1.0])
    @test norm(S * N) < 1e-15
    M = N * [0.2 0.05; 0.05 0.1] * N'
    @test norm(S * M) < 1e-14
    Id = Matrix(1.0I, n, n)
    T = [Id zeros(n, n); M Id]                # λ → λ + M x
    L = [Id zeros(n, n); -A'*M Id]          # what that does to the innovation
    h = 0.05 .* randn(rng, 2n)
    x0 = 0.1 .* randn(rng, 2n)
    P0 = Matrix(0.3I, 2n, 2n)
    sym(X) = Matrix(Symmetric((X + X') / 2))
    function model(Qs, Σm, hm, x0m, P0m; terminal, condition)
        sm = LQRStateModel(
            copy(A),
            copy(S),
            sym.(Qs),
            sym(Σm);
            schedule=terminal ? cost_schedule(tsteps; terminal=true) : Int[],
            terminal=terminal,
            condition_terminal=condition,
            Σf=Matrix(0.02I, n, n),
            hf=zeros(n),
            h=hm,
            x0=x0m,
            P0=sym(P0m),
        )
        return LinearDynamicalSystem(
            sm, GaussianObservationModel(copy(C), Matrix(0.1I, p, p), zeros(p))
        )
    end
    ys = [0.5 .* randn(StableRNG(40 + i), p, tsteps) for i in 1:3]
    for (terminal, condition) in ((true, true), (true, false), (false, false))
        before = terminal ? [Qrun, Qterm] : [Qrun]
        after = [Qrun + M - A' * M * A]
        terminal && push!(after, Qterm + M)
        base = model(before, Σ, h, x0, P0; terminal=terminal, condition=condition)
        shaped = model(
            after,
            L * Σ * L',
            L * h,
            T * x0,
            T * P0 * T';
            terminal=terminal,
            condition=condition,
        )
        # The move is real: ‖M − AᵀMA‖ ≈ 0.029 on the running cost (A is near I),
        # ‖M‖ ≈ 0.23 on the terminal one — against an ELBO held to 1e-12.
        @test norm(after[1] .- before[1]) > 0.02
        terminal && @test norm(after[2] .- before[2]) > 0.2
        @test maximum(
            abs,
            closed_loop_dynamics(base.state_model) .-
            closed_loop_dynamics(shaped.state_model),
        ) < 1e-12
        @test elbo(shaped, ys) ≈ elbo(base, ys) rtol = 1e-12
    end
    return nothing
end

"""The terminal-conditioned M-step moves only to a point both of its estimates
call an improvement, and falls back to steepest descent when the optimizer's
proposal climbs.

A synthetic problem stands in for the real one. Its surrogate is unbounded
below along `θ₂`, as the switching surrogate is wherever the probe's scatter
exceeds the data's; its score shares the surrogate's gradient at `θ′ = 0` (as
the real score does, at a stationary probe posterior) but rises along `θ₂`. An
optimizer run on that surrogate ends far out along `θ₂`, where every halving
back toward `θ′` still climbs the score.
"""
function test_lqr_conditional_acceptance()
    written = Ref(Float64[])
    function problem(surrogate, score; rescores=true)
        theta = [0.0, 0.0]
        evaluate!(_, θ) = surrogate(θ)
        write!(θ) = (written[] = copy(θ); nothing)
        score!(θ) = score(θ)
        return (; theta, evaluate!, write!, score!, rescores)
    end
    s(θ) = θ[1] - 10 * θ[2]^2
    J(θ) = θ[1] + θ[2]^2
    ∇ = [1.0, 0.0]          # both s and J, at θ′
    at_start = (0.0, 0.0)   # (surrogate, score) at θ′

    # Every halving of the runaway proposal lowers s and raises J: the
    # fallback's unit steepest-descent step is what gets taken.
    @test SSD._lqr_accept_conditional!(problem(s, J), [0.0, 50.0], at_start, ∇)
    @test written[] == [-1.0, 0.0]

    # A proposal both estimates call an improvement is taken whole.
    @test SSD._lqr_accept_conditional!(problem(s, J), [-0.5, 0.1], at_start, ∇)
    @test written[] == [-0.5, 0.1]

    # No proposal at all (a failed line search hands back θ′): straight to the fallback.
    @test SSD._lqr_accept_conditional!(problem(s, J), [0.0, 0.0], at_start, ∇)
    @test written[] == [-1.0, 0.0]

    #=
    A score improvement the surrogate contradicts is refused. Both replace
    `log Z` with a lower bound, so the objective is at least the larger of the
    two, and here the surrogate's is the larger everywhere but θ′. At a
    stationary θ′ there is no fallback either, and the M-step stays put.
    =#
    bowl(θ) = sum(abs2, θ)
    @test !SSD._lqr_accept_conditional!(
        problem(bowl, θ -> -1.0 - abs(θ[1])), [1.0, 1.0], at_start, [0.0, 0.0]
    )
    @test written[] == [0.0, 0.0]

    # A non-finite candidate is a rejected one, on either estimate.
    wall(θ) = θ[1] < -0.3 ? Inf : s(θ)
    @test SSD._lqr_accept_conditional!(problem(wall, J), [0.0, 50.0], at_start, ∇)
    @test written[] == [-0.25, 0.0]
    @test SSD._lqr_accept_conditional!(
        problem(s, θ -> θ[1] < -0.3 ? NaN : J(θ)), [0.0, 50.0], at_start, ∇
    )
    @test written[] == [-0.25, 0.0]

    # With an exact normalizer the two coincide, and the score is never computed.
    exact = problem(s, _ -> error("scored under an exact normalizer"); rescores=false)
    @test SSD._lqr_accept_conditional!(exact, [-0.5, 0.0], at_start, ∇)
    @test written[] == [-0.5, 0.0]
    return nothing
end

"""A numerical failure at one point is a rejected step, not a dead fit.

The normalizer's probe is smoothed on a model carrying no observations, which
is the one place in the package where the Laplace smoother has nothing but the
prior and the terminal factor to work with. A line search walking into a
parameter set that chain cannot be factored at must come back as `Inf`, and the
exception has to be recognised through the task wrapper `tforeach` puts around
it — a predicate matching only the leaf types would let an ordinary rejected
step take the whole fit down.
"""
function test_lqr_rejectable_failures()
    @test SSD._lqr_rejectable(PosDefException(1))
    @test SSD._lqr_rejectable(SingularException(1))
    @test SSD._lqr_rejectable(LAPACKException(1604))
    @test !SSD._lqr_rejectable(ArgumentError("a real bug"))
    @test !SSD._lqr_rejectable(MethodError(sin, (1, 2)))

    # Through the wrapper the threaded smoother raises.
    wrapped = try
        fetch(Threads.@spawn throw(LAPACKException(1604)))
    catch err
        err
    end
    @test wrapped isa TaskFailedException
    @test SSD._lqr_rejectable(wrapped)

    bug = try
        fetch(Threads.@spawn throw(ArgumentError("a real bug")))
    catch err
        err
    end
    @test !SSD._lqr_rejectable(bug)

    # A composite is rejectable only when every member is.
    @test SSD._lqr_rejectable(CompositeException([PosDefException(1), LAPACKException(3)]))
    @test !SSD._lqr_rejectable(
        CompositeException([PosDefException(1), ErrorException("x")])
    )
    @test !SSD._lqr_rejectable(CompositeException([]))
    return nothing
end

function test_lqr_noise_update_closed_form()
    rng = StableRNG(35)
    tsteps = 14
    sm, lds = lqr_fixture(rng; terminal=true, nregimes=2, tsteps=tsteps)
    ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:5]
    hs, _, _, pool = lqr_estep_stats(lds, ys)
    ctx = SSD._LQRMStepCtx(hs, sm, true)
    SSD._lqr_structure_mstep!(ctx, true, sm.mstep_iters)
    SSD._lqr_noise_mstep!(ctx)
    # Σ = R/N and Σf = R_f/N_f exactly, which is what profiling them out assumed.
    @test sm.Σ ≈ ctx.R[1] ./ ctx.N_q[1] atol = 1e-12
    @test sm.Σf ≈ ctx.Rf[1] ./ ctx.Nf_q[1] atol = 1e-12
    @test isposdef(Symmetric(Matrix(sm.Σ)))

    # With an inverse-Wishart prior, the profiled optimizer and the covariance
    # update must use the same posterior mode.
    sm2, lds2 = lqr_fixture(rng; terminal=true, nregimes=2, tsteps=tsteps)
    prior = IWPrior(Matrix(0.6I, lds2.latent_dim, lds2.latent_dim), 17.0)
    sm2.Σ_prior = prior
    hs2, _, _, _ = lqr_estep_stats(lds2, ys)
    ctx2 = SSD._LQRMStepCtx(hs2, sm2, true)
    SSD._lqr_structure_mstep!(ctx2, true, sm2.mstep_iters)
    SSD._lqr_noise_mstep!(ctx2)
    expected = (ctx2.R[1] .+ prior.Ψ) ./ (prior.ν + ctx2.N_q[1] + lds2.latent_dim + 1)
    @test sm2.Σ ≈ expected atol = 1e-12
    return nothing
end

function test_lqr_fixed_costate_sigma()
    rng = StableRNG(135)
    sm, lds = lqr_fixture(rng; terminal=true, nregimes=2, tsteps=10)
    n = size(sm.A, 1)
    v = 1e-3
    sm.fixed_costate_sigma = v
    sm.Σ[(n + 1):(2n), (n + 1):(2n)] .= v .* I(n)
    refresh!(sm)
    ys = [randn(rng, lds.obs_dim, 10) .* 0.4 for _ in 1:3]
    hs, _, _, _ = lqr_estep_stats(lds, ys)
    ctx = SSD._LQRMStepCtx(hs, sm, true)
    @test !ctx.profile
    SSD._lqr_structure_mstep!(ctx, true, sm.mstep_iters)
    SSD._lqr_noise_mstep!(ctx)
    @test sm.Σ[1:n, 1:n] ≈ ctx.R[1][1:n, 1:n] ./ ctx.N_q[1] atol = 1e-12
    @test sm.Σ[(n + 1):(2n), (n + 1):(2n)] ≈ v .* I(n)
    @test sm.Σ[1:n, (n + 1):(2n)] == zeros(n, n)
    @test sm.Σ[(n + 1):(2n), 1:n] == zeros(n, n)

    slots = [ones(Int, 1) for _ in 1:4]
    problem = SSD._lqr_conditional_problem([lds], [hs], slots)
    θ = copy(problem.theta)
    @test isfinite(problem.evaluate!(zeros(length(θ)), θ))
    @test sm.Σ[(n + 1):(2n), (n + 1):(2n)] ≈ v .* I(n)
    @test sm.Σ[1:n, (n + 1):(2n)] == zeros(n, n)

    @test_throws ArgumentError LQRStateModel(
        sm.A,
        sm.S,
        sm.Qc,
        sm.Σ;
        fixed_costate_sigma=v,
        Σ_prior=IWPrior(Matrix(0.1I, 2n, 2n), 8.0),
    )
    return nothing
end

function test_lqr_recovers_parameters()
    rng = StableRNG(36)
    #=
    Self-consistency: with one cost regime and no terminal factor the model is a
    proper directed chain, so `rand` really does draw from it and the MLE should
    land near the generating parameters. (With a terminal factor `rand` draws
    from the *conditioned* path distribution while the objective is the joint —
    a selection effect, documented on the type, so recovery is checked here on
    the configuration where the question is well posed.)
    =#
    n = 2
    d = 2n
    p = 6
    tsteps = 20
    ntrials = 120
    A = [0.97 0.05; -0.04 0.95]
    Sm = [0.05 0.01; 0.01 0.04]
    Qc = [0.20 0.03; 0.03 0.15]
    Σ = Matrix(Diagonal(fill(0.02, d)))
    sm = LQRStateModel(A, Sm, Qc, Σ; P0=Matrix(0.2I, d, d))
    C = randn(rng, p, d)
    C[:, (n + 1):d] .= 0
    R = Matrix(0.05I, p, p)
    lds = LinearDynamicalSystem(sm, GaussianObservationModel(C, copy(R), zeros(p)))
    _, ys = rand(rng, lds, fill(tsteps, ntrials))

    # Inverse LQR proper: the plant is known, the cost is what we are after.
    sm0 = LQRStateModel(
        copy(A),
        copy(Sm),
        Matrix(0.4I, n, n),
        Matrix(0.05I, d, d);
        P0=Matrix(0.2I, d, d),
        fit_flags=LQRFitFlags(; A=false, S=false),
    )
    lds0 = LinearDynamicalSystem(sm0, GaussianObservationModel(copy(C), copy(R), zeros(p)))
    els = fit!(lds0, ys; max_iter=250, tol=1e-10, progress=false)
    @test minimum(diff(els)) > -1e-8

    # Compare in the canonical scale: the cost is identified only up to a
    # nonzero scalar (the classical inverse-optimal-control invariance).
    truth = deepcopy(sm)
    rescale_costate!(truth; target=:trace)
    rescale_costate!(sm0; target=:trace)
    @test maximum(abs, sm0.Qc[1] .- truth.Qc[1]) / maximum(abs, truth.Qc[1]) < 0.25
    # The fit must at least match the generating parameters' own ELBO.
    @test els[end] >= elbo(lds, ys) - 1e-6
    return nothing
end

# ---------------------------------------------------------------------------
# Emission models, costate readout, sampling, printing
# ---------------------------------------------------------------------------

function test_lqr_costate_readout_mask()
    rng = StableRNG(41)
    tsteps = 14
    sm, lds = lqr_fixture(rng; nregimes=1, tsteps=tsteps)
    n = SSD._plant_dim(sm)
    d = lds.latent_dim
    @test SSD._costate_range(lds) == (n + 1):d

    ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:4]
    fit!(lds, ys; max_iter=6, progress=false)
    # `observe_costate = false` means C's costate columns are zero, and the
    # masked emission M-step keeps them there.
    @test all(iszero, lds.obs_model.C[:, (n + 1):d])
    @test !all(iszero, lds.obs_model.C[:, 1:n])

    # A nonzero costate readout supplied by hand is zeroed at entry, with a warning.
    lds.obs_model.C[:, (n + 1):d] .= 0.3
    @test_logs (:warn, r"costate columns") match_mode = :any elbo(lds, ys)
    @test all(iszero, lds.obs_model.C[:, (n + 1):d])

    # Opting in leaves the emission free.
    sm2, lds2 = lqr_fixture(rng; nregimes=1, tsteps=tsteps, observe_costate=true)
    @test SSD._costate_range(lds2) === nothing
    fit!(lds2, ys; max_iter=6, progress=false)
    @test !all(iszero, lds2.obs_model.C[:, (n + 1):d])

    # A full matrix-normal prior must be restricted to the observable columns.
    # A nonzero masked prior mean and dense cross-column precision previously
    # recreated costate readout after the first M-step.
    sm3, lds3 = lqr_fixture(rng; nregimes=1, tsteps=tsteps)
    regdim = d + 1
    M₀ = randn(rng, lds3.obs_dim, regdim)
    M₀[:, (n + 1):d] .= 5
    Λ = Matrix(1.0I, regdim, regdim) .+ 0.1 .* ones(regdim, regdim)
    lds3.obs_model.CD_prior = MNPrior(M₀, Λ)
    fit!(lds3, ys; max_iter=4, progress=false)
    @test all(iszero, lds3.obs_model.C[:, (n + 1):d])
    return nothing
end

function test_lqr_singular_fitted_psd_rejected()
    rng = StableRNG(410)
    sm, lds = lqr_fixture(rng; nregimes=1, tsteps=10)
    ys = [randn(rng, lds.obs_dim, 10) for _ in 1:2]
    hs, _, _, _ = lqr_estep_stats(lds, ys)

    fill!(sm.Qc[1], 0)
    ctx = SSD._LQRMStepCtx(hs, sm, true)
    θ = zeros(ctx.pack.np)
    @test_throws ArgumentError SSD._lqr_pack!(θ, ctx)

    # Exact singular costs are still legal when they are intentionally frozen.
    sm.fit_flags = LQRFitFlags(; Qc=false)
    frozen = SSD._LQRMStepCtx(hs, sm, true)
    @test SSD._lqr_pack!(zeros(frozen.pack.np), frozen) isa Vector
    return nothing
end

function test_lqr_masked_fit_matches_reduced_model()
    rng = StableRNG(42)
    #=
    Pinning C's costate columns at zero must give exactly the emission a model
    with those columns absent would fit: the mask decouples them in the normal
    equations rather than projecting after the solve.
    =#
    tsteps = 14
    sm, lds = lqr_fixture(rng; nregimes=1, tsteps=tsteps)
    n = SSD._plant_dim(sm)
    d = lds.latent_dim
    ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:4]
    hs, tfs, data, pool = lqr_estep_stats(lds, ys)
    SSD.update_C_d!(lds, hs.base, pool[1])
    C_masked = copy(lds.obs_model.C)
    @test all(iszero, C_masked[:, (n + 1):d])

    # Reference: solve the state-only regression by hand from the same statistics.
    Gram = Matrix(hs.base.obs_xx[])
    cross = copy(hs.base.obs_xy)
    keep = vcat(1:n, d + 1)
    C_ref = transpose(Gram[keep, keep] \ cross[keep, :])
    @test maximum(abs, C_masked[:, 1:n] .- C_ref[:, 1:n]) < 1e-8
    @test maximum(abs, lds.obs_model.d .- C_ref[:, n + 1]) < 1e-8
    return nothing
end

function test_lqr_poisson_emission()
    rng = StableRNG(43)
    tsteps = 14
    n = 2
    d = 2n
    p = 5
    sm, _ = lqr_fixture(rng; nregimes=2, terminal=true, tsteps=tsteps)
    C = randn(rng, p, d) .* 0.3
    C[:, (n + 1):d] .= 0
    plds = LinearDynamicalSystem(sm, PoissonObservationModel(C, fill(1.0, p)))
    @test length(plds.fit_bool) == 5

    ys = [Float64.(rand(rng, 0:4, p, tsteps)) for _ in 1:4]
    els = fit!(plds, ys; max_iter=12, progress=false)
    @test all(isfinite, els)
    @test els[end] > els[1]
    # The Poisson emission M-step is a constrained maximization, not a
    # projection, so the costate columns never move off zero.
    @test all(iszero, plds.obs_model.C[:, (n + 1):d])
    @test symplectic_defect(sm, 1) < 1e-9

    xs, ps = smooth(plds, ys)
    @test length(xs) == 4
    @test size(xs[1]) == (d, tsteps)
    @test all(isfinite, xs[1])
    @test isfinite(elbo(plds, ys))
    return nothing
end

function test_lqr_composite_emission()
    rng = StableRNG(44)
    tsteps = 13
    n = 2
    d = 2n
    sm, _ = lqr_fixture(rng; nregimes=1, tsteps=tsteps)
    Ck = randn(rng, 3, d)
    Ck[:, (n + 1):d] .= 0
    Cs = randn(rng, 4, d) .* 0.3
    Cs[:, (n + 1):d] .= 0
    lds = LinearDynamicalSystem(
        sm,
        (
            kin=GaussianObservationModel(Ck, Matrix(0.1I, 3, 3), zeros(3)),
            spk=PoissonObservationModel(Cs, fill(1.0, 4)),
        ),
    )
    @test lds.obs_dim == 7
    # A composite's observations are keyed by member, each value that member's
    # own vector of per-trial matrices.
    ys = (
        kin=[randn(rng, 3, tsteps) .* 0.4 for _ in 1:3],
        spk=[Float64.(rand(rng, 0:4, 4, tsteps)) for _ in 1:3],
    )
    els = fit!(lds, ys; max_iter=8, progress=false)
    @test all(isfinite, els)
    @test els[end] > els[1]
    @test all(iszero, lds.obs_model.kin.C[:, (n + 1):d])
    @test all(iszero, lds.obs_model.spk.C[:, (n + 1):d])

    # An all-Gaussian composite takes the quadratic path, with its own ELBO
    # method; check it against the equivalent stacked single emission.
    sm2, _ = lqr_fixture(rng; nregimes=1, tsteps=tsteps)
    C1 = randn(rng, 3, d)
    C1[:, (n + 1):d] .= 0
    C2 = randn(rng, 2, d)
    C2[:, (n + 1):d] .= 0
    comp = LinearDynamicalSystem(
        sm2,
        (
            a=GaussianObservationModel(C1, Matrix(0.1I, 3, 3), zeros(3)),
            b=GaussianObservationModel(C2, Matrix(0.2I, 2, 2), zeros(2)),
        ),
    )
    ya = randn(rng, 3, tsteps) .* 0.4
    yb = randn(rng, 2, tsteps) .* 0.4
    sm3, _ = lqr_fixture(rng; nregimes=1, tsteps=tsteps)
    sm3.A = copy(sm2.A)
    sm3.S = copy(sm2.S)
    sm3.Qc[1] = copy(sm2.Qc[1])
    sm3.Σ = copy(sm2.Σ)
    sm3.h = copy(sm2.h)
    sm3.x0 = copy(sm2.x0)
    sm3.P0 = copy(sm2.P0)
    refresh!(sm3)
    stacked = LinearDynamicalSystem(
        sm3,
        GaussianObservationModel(
            vcat(C1, C2), Matrix(Diagonal(vcat(fill(0.1, 3), fill(0.2, 2)))), zeros(5)
        ),
    )
    @test elbo(comp, (a=ya, b=yb)) ≈ elbo(stacked, vcat(ya, yb)) atol = 1e-8
    return nothing
end

function test_lqr_sampling()
    rng = StableRNG(45)
    tsteps = 12
    n = 2
    d = 2n

    # Without a terminal factor the chain is a proper directed model, so the
    # forward roll is the model's own distribution.
    sm, lds = lqr_fixture(rng; nregimes=1, tsteps=tsteps)
    z, y = rand(StableRNG(1), lds, tsteps)
    @test size(z) == (d, tsteps)
    @test size(y) == (lds.obs_dim, tsteps)
    z2, y2 = rand(StableRNG(1), lds, tsteps)
    @test z == z2 && y == y2                       # reproducible from a seed

    zs, ys = rand(StableRNG(2), lds, fill(tsteps, 4))
    @test length(zs) == 4 && size(zs[1]) == (d, tsteps)
    zr, yr = rand(StableRNG(3), lds, [9, 12, 11])
    @test size.(zr, 2) == [9, 12, 11]

    # With a terminal factor the path is drawn from the *conditioned* joint, so
    # sampled paths satisfy the terminal condition to within Σf.
    smt, ldst = lqr_fixture(rng; terminal=true, nregimes=2, tsteps=tsteps)
    zt, _ = rand(StableRNG(4), ldst, tsteps)
    resid = zt[(n + 1):d, end] .- smt.Qc[smt.schedule[end]] * zt[1:n, end] .- smt.hf
    @test norm(resid) < 5 * sqrt(maximum(diag(smt.Σf))) * sqrt(n)

    # ... and the conditioning is what keeps the path bounded: the same model
    # rolled forward unconditionally grows with the horizon.
    @test maximum(abs, zt) < 20

    @test_throws ArgumentError rand(StableRNG(5), ldst, tsteps; depends_on=(A=[1],))
    return nothing
end

function test_lqr_simulate_lqr()
    rng = StableRNG(46)
    tsteps = 25
    n = 2
    sm, _ = lqr_fixture(rng; terminal=true, nregimes=2, tsteps=tsteps)
    sm.h .= 0
    sm.hf .= 0
    refresh!(sm)

    # The noiseless rollout must satisfy the LQR recursion exactly.
    z = simulate_lqr(rng, sm, tsteps; process_noise=false, x1=[0.5, -0.3])
    @test size(z) == (2n, tsteps)
    for t in 1:(tsteps - 1)
        E = lqr_matrix(sm, SSD._regime(sm, t))
        w = [z[1:n, t]; z[(n + 1):(2n), t + 1]]
        v = [z[1:n, t + 1]; z[(n + 1):(2n), t]]
        @test maximum(abs, E * w .- v) < 1e-9
    end
    # ... including the terminal boundary condition.
    @test maximum(abs, z[(n + 1):(2n), end] .- sm.Qc[sm.schedule[end]] * z[1:n, end]) <
        1e-10

    # The Riccati sweep is what the rollout follows: λ_t = P_t x_t.
    P, g, W = lqr_riccati_sequence(sm, tsteps)
    for t in 1:tsteps
        @test maximum(abs, z[(n + 1):(2n), t] .- (P[t] * z[1:n, t] .+ g[t])) < 1e-9
        @test P[t] ≈ transpose(P[t]) atol = 1e-10
    end
    @test P[end] ≈ sm.Qc[sm.schedule[end]]

    # It stays bounded where the forward roll would not.
    zn = simulate_lqr(rng, sm, tsteps; costate_slack=0.02)
    @test maximum(abs, zn) < 50
    @test all(isfinite, zn)

    # Far from the terminal step the Riccati solution has reached its fixed point.
    sm1, _ = lqr_fixture(rng; nregimes=1, tsteps=tsteps)
    P1, _, _ = lqr_riccati_sequence(sm1, 200)
    @test maximum(abs, P1[1] .- riccati_solution(sm1)) < 1e-8
    return nothing
end

function test_lqr_tracking_control()
    rng = StableRNG(60)
    n = 2
    d = 2n
    m = 2
    tsteps = 25
    A = [0.97 0.05; -0.04 0.95]
    Sm = [0.20 0.02; 0.02 0.18]
    Qcs = [[0.6 0.05; 0.05 0.5], [3.0 0.0; 0.0 2.5]]
    Σ = Matrix(Diagonal(fill(0.01, d)))
    sched = cost_schedule(tsteps; terminal=true)
    Gref = Matrix(1.0I, n, m)          # the reference *is* the input
    sm = LQRStateModel(
        A,
        Sm,
        Qcs,
        Σ;
        schedule=sched,
        terminal=true,
        Bu=zeros(d, m),
        Gref=Gref,
        Σf=Matrix(1e-6I, n, n),
        P0=Matrix(0.2I, d, d),
    )
    target = [1.5, -0.8]
    ux = repeat(target, 1, tsteps)
    z = simulate_lqr(rng, sm, tsteps; process_noise=false, x1=zeros(n), ux=ux)

    #=
    The tracking rollout must satisfy the *inhomogeneous* LQR recursion
    exactly, with the affine term `[d_t; −Q_t r_t]`. That the costate half
    carries this regime's own cost is the whole point of `Gref`.
    =#
    for t in 1:(tsteps - 1)
        k = SSD._regime(sm, t)
        E = lqr_matrix(sm, k)
        w = [z[1:n, t]; z[(n + 1):d, t + 1]]
        v = [z[1:n, t + 1]; z[(n + 1):d, t]]
        affine = [zeros(n); -Qcs[k] * (Gref * ux[:, t])]
        @test maximum(abs, E * w .+ affine .- v) < 1e-10
    end
    # And the tracking terminal condition λ_T = Q_f (x_T − r_T).
    kT = sched[end]
    @test maximum(abs, z[(n + 1):d, end] .- Qcs[kT] * (z[1:n, end] .- target)) < 1e-10

    # A heavier terminal cost pulls the endpoint onto the target.
    sm_heavy = LQRStateModel(
        A,
        Sm,
        [copy(Qcs[1]), 200.0 * Matrix(I, n, n)],
        Σ;
        schedule=sched,
        terminal=true,
        Bu=zeros(d, m),
        Gref=copy(Gref),
        Σf=Matrix(1e-6I, n, n),
        P0=Matrix(0.2I, d, d),
    )
    z_heavy = simulate_lqr(rng, sm_heavy, tsteps; process_noise=false, x1=zeros(n), ux=ux)
    @test norm(z_heavy[1:n, end] .- target) < norm(z[1:n, end] .- target) / 10

    # With no reference the model is the regulation problem: the input cannot
    # move the costate at all.
    sm_reg = LQRStateModel(
        A,
        Sm,
        Qcs,
        Σ;
        schedule=sched,
        terminal=true,
        Bu=zeros(d, m),
        Σf=Matrix(1e-6I, n, n),
        P0=Matrix(0.2I, d, d),
    )
    @test all(iszero, sm_reg.Gref)
    z_reg = simulate_lqr(rng, sm_reg, tsteps; process_noise=false, x1=zeros(n), ux=ux)
    @test maximum(abs, z_reg) < 1e-12       # starts at 0, no reference to chase
    return nothing
end

function test_lqr_tracking_mstep()
    rng = StableRNG(61)
    n = 2
    m = 2
    tsteps = 16
    sm, lds = lqr_fixture(rng; terminal=true, nregimes=3, tsteps=tsteps, ux_dim=m, onset=9)
    sm.Gref .= randn(rng, n, m) .* 0.5
    refresh!(sm)
    ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:4]
    uxs = [randn(rng, m, tsteps) for _ in 1:4]
    hs, _, _, _ = lqr_estep_stats(lds, ys; ux=uxs)

    #=
    `Gref` enters the objective bilinearly with `Q_k` — through the input block
    `−Q_k G_r` and again through the terminal factor's `+Q_f G_r` — so both
    gradients get cross terms. Check them against the independent reference.
    =#
    for profile in (true, false)
        ctx = SSD._LQRMStepCtx(hs, sm, profile)
        θ = zeros(ctx.pack.np)
        SSD._lqr_pack!(θ, ctx)
        g = similar(θ)
        f = SSD._lqr_fg!(g, θ, ctx)
        @test f ≈ lqr_ref_objective(θ, hs, sm, profile) atol = 1e-8
        gref = ForwardDiff.gradient(t -> lqr_ref_objective(t, hs, sm, profile), θ)
        @test maximum(abs, g .- gref) / max(1.0, maximum(abs, gref)) < 1e-8
    end

    # `Gref` is a packed block, and freezing it removes it from the problem.
    full = SSD._LQRPack(sm).np
    sm.fit_flags = LQRFitFlags(; Gref=false)
    @test SSD._LQRPack(sm).np == full - n * m
    G0 = copy(sm.Gref)
    els = fit!(lds, ys; ux=uxs, max_iter=6, progress=false)
    @test sm.Gref == G0
    @test minimum(diff(els)) > -1e-8

    # Unfrozen, it moves and EM stays monotone.
    sm.fit_flags = LQRFitFlags()
    els2 = fit!(lds, ys; ux=uxs, max_iter=10, progress=false)
    @test sm.Gref != G0
    @test minimum(diff(els2)) > -1e-8

    #=
    The ELBO still matches the exact Laplace normalizer with tracking on — less
    the terminal normalizer, which `Gref` also enters (through `Ftrm`, the
    terminal residual's reference term), so this checks the tracking gain on
    both sides of the conditional score at once.
    =#
    exact = lqr_exact_marginal(lds, ys[1]; ux=uxs[1])
    logz = lqr_terminal_logz_reference(sm, uxs[1], tsteps)
    @test elbo(lds, ys[1]; ux=uxs[1]) ≈ exact - logz atol = 1e-7
    sm.condition_terminal = false
    @test elbo(lds, ys[1]; ux=uxs[1]) ≈ exact atol = 1e-7
    sm.condition_terminal = true
    return nothing
end

function test_lqr_gref_columns()
    rng = StableRNG(62)
    n = 2
    m = 4
    tsteps = 16
    ntrials = 20

    function fixture(cols)
        r = StableRNG(62)
        sm, lds = lqr_fixture(r; terminal=true, nregimes=2, tsteps=tsteps, ux_dim=m)
        sm.Gref .= 0
        sm.Gref[:, 2:3] .= randn(r, n, 2) .* 0.5
        sm.fit_flags = LQRFitFlags(; Gref_cols=cols)
        refresh!(sm)
        ys = [randn(r, lds.obs_dim, tsteps) .* 0.4 for _ in 1:ntrials]
        uxs = [randn(r, m, tsteps) for _ in 1:ntrials]
        return sm, lds, ys, uxs
    end

    # Only the named columns are packed, and only they move.
    sm, lds, ys, uxs = fixture([2, 3])
    @test SSD._LQRPack(sm).np == SSD._LQRPack(sm, LQRFitFlags()).np - n * (m - 2)
    G0 = copy(sm.Gref)
    els = fit!(lds, ys; ux=uxs, max_iter=8, progress=false)
    @test minimum(diff(els)) > -1e-8
    @test sm.Gref[:, [1, 4]] == G0[:, [1, 4]]
    @test !(sm.Gref[:, 2:3] ≈ G0[:, 2:3])

    #=
    Naming every column must be the *same* fit as naming none: the narrowing is
    a restriction of the packed problem, not a different objective.
    =#
    smA, ldsA, ysA, uxsA = fixture(collect(1:m))
    elsA = fit!(ldsA, ysA; ux=uxsA, max_iter=8, progress=false)
    smB, ldsB, ysB, uxsB = fixture(nothing)
    elsB = fit!(ldsB, ysB; ux=uxsB, max_iter=8, progress=false)
    @test maximum(abs, elsA .- elsB) < 1e-10
    @test smA.Gref ≈ smB.Gref

    #=
    And the restricted model is nested inside the free one — at the optimum.
    After a fixed number of generalized-EM steps neither run is there (the free
    one is still gaining ~4e-2 per iteration here), and the free problem has more
    coordinates to move, so it can be behind on the path while being ahead at
    convergence; by iteration 40 the sign flips on its own. Bound the violation
    by how far from a stationary point the free run still is.
    =#
    @test last(els) <= last(elsB) + last(diff(elsB))

    # Freezing wins over narrowing, and the flags validate against the width.
    smC, ldsC, ysC, uxsC = fixture([2, 3])
    smC.fit_flags = LQRFitFlags(; Gref=false, Gref_cols=[2, 3])
    GC = copy(smC.Gref)
    fit!(ldsC, ysC; ux=uxsC, max_iter=4, progress=false)
    @test smC.Gref == GC
    @test_throws ArgumentError LQRFitFlags(; Gref_cols=Int[])
    @test_throws ArgumentError LQRStateModel(
        Matrix(0.9I, n, n),
        Matrix(0.2I, n, n),
        [Matrix(1.0I, n, n)],
        Matrix(0.1I, 2n, 2n);
        Bu=zeros(2n, m),
        fit_flags=LQRFitFlags(; Gref_cols=[m + 1]),
    )

    # Flags compare by value, which an SLDS's "same flags" check relies on.
    @test LQRFitFlags(; Gref_cols=[2, 3]) == LQRFitFlags(; Gref_cols=[2, 3])
    @test LQRFitFlags(; Gref_cols=[2, 3]) != LQRFitFlags(; Gref_cols=[2])
    return nothing
end

function test_lqr_depends_on()
    rng = StableRNG(70)
    tsteps = 16
    ntrials = 12
    sm, lds = lqr_fixture(rng; nregimes=1, tsteps=tsteps)
    ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:ntrials]
    session = repeat([1, 2]; inner=ntrials ÷ 2)
    one_group = fill(1, ntrials)

    #=
    A grouping with a single group must reproduce the ungrouped fit exactly —
    the sharpest check that the grouped path is the same estimator, since the
    cells pool back into one unit.
    =#
    smA, ldsA = lqr_fixture(StableRNG(70); nregimes=1, tsteps=tsteps)
    set_depends_on!(smA, (structure=one_group, noise=one_group))
    ldsA2 = LinearDynamicalSystem(smA, ldsA.obs_model)
    elsA = fit!(ldsA2, ys; max_iter=8, progress=false)
    smB, ldsB = lqr_fixture(StableRNG(70); nregimes=1, tsteps=tsteps)
    elsB = fit!(ldsB, ys; max_iter=8, progress=false)
    @test maximum(abs, elsA .- elsB) < 1e-8

    # Stitching: one shared LQR structure, per-session emission.
    smC, ldsC = lqr_fixture(StableRNG(71); nregimes=1, tsteps=tsteps)
    set_depends_on!(ldsC.obs_model, (C=session, d=session, D=session, R=session))
    ldsC2 = LinearDynamicalSystem(smC, ldsC.obs_model)
    elsC = fit!(ldsC2, ys; max_iter=10, progress=false)
    @test minimum(diff(elsC)) > -1e-8
    @test !(
        group_parameter(ldsC2.obs_model, :C, 1) ≈ group_parameter(ldsC2.obs_model, :C, 2)
    )
    @test isfinite(elbo(ldsC2, ys))

    #=
    State-side grouping. `:structure` is one user-facing name for the whole
    joint block, since its pieces move together; `:noise` likewise. Naming a
    piece is rejected rather than silently taken for the block.
    =#
    for dep in (
        (structure=session,),
        (noise=session,),
        (structure=session, noise=session),
        (x0=session, P0=session),
    )
        smD, ldsD = lqr_fixture(StableRNG(72); nregimes=2, terminal=true, tsteps=tsteps)
        set_depends_on!(smD, dep)
        ldsD2 = LinearDynamicalSystem(smD, ldsD.obs_model)
        els = fit!(ldsD2, ys; max_iter=8, progress=false)
        @test minimum(diff(els)) > -1e-8
        @test all(isfinite, els)
    end

    #=
    `set_depends_on!` only records the declaration; it is resolved — and so
    rejected — when the model goes into a `LinearDynamicalSystem`.
    =#
    for bad in ((Σ=session,), (Mfree=session,), (nonsense=session,))
        smE, ldsE = lqr_fixture(StableRNG(73); nregimes=1, tsteps=tsteps)
        set_depends_on!(smE, bad)
        @test_throws ArgumentError LinearDynamicalSystem(smE, ldsE.obs_model)
    end

    #=
    A piece of the structural block on its own. The cells are the same ones
    `:structure` would give — the pieces are still solved jointly — but only the
    named piece gets a copy per group, so `(Qc = session,)` is "one plant, a cost
    per session" rather than "a different arm per session".
    =#
    smP, ldsP = lqr_fixture(StableRNG(74); nregimes=2, terminal=true, tsteps=tsteps)
    set_depends_on!(smP, (Qc=session,))
    ldsP2 = LinearDynamicalSystem(smP, ldsP.obs_model)
    elsP = fit!(ldsP2, ys; max_iter=10, progress=false)
    @test minimum(diff(elsP)) > -1e-8
    p1 = group_parameter(smP, :structure, 1)
    p2 = group_parameter(smP, :structure, 2)
    @test !(p1.Qc[1] ≈ p2.Qc[1])              # the named piece varies
    @test p1.A === p2.A                       # the rest is one shared array
    @test p1.S === p2.S
    @test p1.h === p2.h && p1.Bu === p2.Bu && p1.Gref === p2.Gref
    @test group_labels(smP, :Qc) == Any[1, 2]
    for v in (p1, p2)
        @test symplectic_defect(v) < 1e-9
    end

    # Naming several pieces is legal; naming them inconsistently is not.
    smQ, ldsQ = lqr_fixture(StableRNG(75); nregimes=1, tsteps=tsteps)
    set_depends_on!(smQ, (Qc=session, h=session))
    ldsQ2 = LinearDynamicalSystem(smQ, ldsQ.obs_model)
    @test minimum(diff(fit!(ldsQ2, ys; max_iter=6, progress=false))) > -1e-8
    q1 = group_parameter(smQ, :structure, 1)
    q2 = group_parameter(smQ, :structure, 2)
    @test q1.A === q2.A && !(q1.h === q2.h)
    smR, ldsR = lqr_fixture(StableRNG(75); nregimes=1, tsteps=tsteps)
    set_depends_on!(smR, (Qc=session, h=one_group))
    @test_throws ArgumentError LinearDynamicalSystem(smR, ldsR.obs_model)

    # A single group still reproduces the ungrouped fit exactly, piecewise too.
    smS, ldsS = lqr_fixture(StableRNG(76); nregimes=1, tsteps=tsteps)
    set_depends_on!(smS, (Qc=one_group,))
    elsS = fit!(LinearDynamicalSystem(smS, ldsS.obs_model), ys; max_iter=8, progress=false)
    smT, ldsT = lqr_fixture(StableRNG(76); nregimes=1, tsteps=tsteps)
    elsT = fit!(ldsT, ys; max_iter=8, progress=false)
    @test maximum(abs, elsS .- elsT) < 1e-8

    # Grouped structure really does diverge, while a shared group stays shared.
    smF, ldsF = lqr_fixture(StableRNG(74); nregimes=1, tsteps=tsteps)
    set_depends_on!(smF, (structure=session,))
    ldsF2 = LinearDynamicalSystem(smF, ldsF.obs_model)
    fit!(ldsF2, ys; max_iter=10, progress=false)
    g1 = group_parameter(smF, :structure, 1)
    g2 = group_parameter(smF, :structure, 2)
    @test !(g1.Qc[1] ≈ g2.Qc[1])
    @test g1.Σ === g2.Σ                       # noise not grouped: shared by reference
    @test group_labels(smF, :structure) == Any[1, 2]
    for v in (g1, g2)
        @test v.S ≈ transpose(v.S) atol = 1e-12
        @test symplectic_defect(v) < 1e-9
    end

    xs, _ = smooth(ldsF2, ys)
    @test length(xs) == ntrials
    @test size(xs[1]) == (lds.latent_dim, tsteps)
    return nothing
end

function test_lqr_show()
    rng = StableRNG(47)
    sm, lds = lqr_fixture(rng; terminal=true, nregimes=2, tsteps=12)
    str = sprint(show, sm)
    @test occursin("LQR", str)
    @test occursin("B R⁻¹ Bᵀ", str)
    @test occursin("Cost schedule", str)
    @test occursin("observe_costate = false", str)
    @test occursin("symplectic defect", str)
    # A wide model prints shapes instead of contents.
    big, _ = lqr_fixture(rng; n=6, p=3, tsteps=12)
    @test occursin("size(A)", sprint(show, big))
    @test occursin("LQR", sprint(show, lds))
    return nothing
end

function test_lqr_priors_and_fit_bool()
    rng = StableRNG(48)
    tsteps = 14
    sm, lds = lqr_fixture(rng; nregimes=1, tsteps=tsteps)
    ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:4]

    # fit_bool's four state slots: [x0, P0, structure, noise].
    lds_frozen = LinearDynamicalSystem(
        sm, lds.obs_model; fit_bool=(x0=false, P0=false, A=false, Q=false)
    )
    x0_0 = copy(sm.x0)
    P0_0 = copy(sm.P0)
    Σ0 = copy(sm.Σ)
    Q0 = copy(sm.Qc[1])
    fit!(lds_frozen, ys; max_iter=4, progress=false)
    @test sm.x0 == x0_0
    @test sm.P0 == P0_0
    @test sm.Σ == Σ0
    @test sm.Qc[1] == Q0     # slot 3 gates the whole structural update

    # With the noise frozen the objective is the fixed-Σ form; still monotone.
    sm2, lds2 = lqr_fixture(rng; nregimes=1, tsteps=tsteps)
    lds2_noQ = LinearDynamicalSystem(sm2, lds2.obs_model; fit_bool=(Q=false,))
    Σ2 = copy(sm2.Σ)
    els = fit!(lds2_noQ, ys; max_iter=12, progress=false)
    @test sm2.Σ == Σ2
    @test minimum(diff(els)) > -1e-8

    # An initial-state prior contributes to the ELBO and shrinks P0.
    sm3, lds3 = lqr_fixture(rng; nregimes=1, tsteps=tsteps)
    d = lds3.latent_dim
    bare = elbo(lds3, ys)
    sm3.P0_prior = IWPrior(Matrix(1.0I, d, d), Float64(d + 8))
    @test elbo(lds3, ys) != bare
    fit!(lds3, ys; max_iter=6, progress=false)
    @test isposdef(Symmetric(Matrix(sm3.P0)))

    # Grouped models may vary noise and structure independently. Count each
    # distinct covariance/cost array once, rather than counting both on the
    # noise slot (which duplicates this shared cost).
    sm4, lds4 = lqr_fixture(rng; nregimes=1, tsteps=tsteps)
    sm5 = deepcopy(sm4)
    sm5.Qc = sm4.Qc
    sigma_prior = IWPrior(Matrix(0.7I, d, d), 13.0)
    qc_prior = IWPrior(Matrix(0.5I, d ÷ 2, d ÷ 2), 11.0)
    for smi in (sm4, sm5)
        smi.Σ_prior = sigma_prior
        smi.Qc_prior = qc_prior
    end
    lds5 = LinearDynamicalSystem(sm5, deepcopy(lds4.obs_model))
    slots = [ones(Int, 2) for _ in 1:6]
    slots[SSD._G_Q] = [1, 2]
    got = SSD._grouped_state_prior_logdensity([lds4, lds5], slots, Float64)
    expected =
        SSD.iw_logprior_term(sm4.Σ, sigma_prior) +
        SSD.iw_logprior_term(sm5.Σ, sigma_prior) +
        SSD.iw_logprior_term(sm4.Qc[1], qc_prior)
    @test got ≈ expected
    return nothing
end

function test_lqr_ragged_with_schedule()
    rng = StableRNG(50)
    #=
    Trials of unequal length under a schedule: the schedule is indexed by
    within-trial timestep, so a short trial uses a prefix of it and its terminal
    factor reads `schedule[T_n]` — the entry for *its own* last step, not the
    longest trial's.
    =#
    tsteps = 18
    sm, lds = lqr_fixture(rng; terminal=true, nregimes=3, tsteps=tsteps, onset=11)
    lengths = [12, 18, 15]
    ys = [randn(rng, lds.obs_dim, t) .* 0.4 for t in lengths]

    els = fit!(lds, ys; max_iter=10, progress=false)
    @test minimum(diff(els)) > -1e-8
    @test all(isfinite, els)

    # The per-regime transition counts must match the schedule, trial by trial.
    hs, _, _, _ = lqr_estep_stats(lds, ys)
    expected = zeros(3)
    for t_n in lengths, t in 1:(t_n - 1)
        expected[SSD._regime(sm, t)] += 1
    end
    @test hs.nk ≈ expected
    expected_terminal = zeros(3)
    for t_n in lengths
        expected_terminal[SSD._regime(sm, t_n)] += 1
    end
    @test hs.term_n ≈ expected_terminal
    @test sum(hs.nk) ≈ sum(lengths) - length(lengths)

    xs, _ = smooth(lds, ys)
    @test size.(xs, 2) == lengths

    # A trial longer than the schedule is refused rather than read out of bounds.
    too_long = [randn(rng, lds.obs_dim, tsteps + 1) .* 0.4]
    @test_throws SSD.DimensionMismatchError elbo(lds, too_long)
    return nothing
end

function test_lqr_single_trial_and_edge_cases()
    rng = StableRNG(49)
    sm, lds = lqr_fixture(rng; nregimes=1, tsteps=10)
    y = randn(rng, lds.obs_dim, 10) .* 0.4
    els = fit!(lds, y; max_iter=8, progress=false)
    @test minimum(diff(els)) > -1e-8

    # A 3-D observation array is accepted like everywhere else.
    y3 = randn(rng, lds.obs_dim, 10, 3) .* 0.4
    @test isfinite(elbo(lds, y3))
    xs, ps = smooth(lds, y3)
    @test length(xs) == 3

    # A model with more cost regimes than the schedule reaches warns at build.
    A = [0.96 0.07; -0.05 0.93]
    Sm = [0.06 0.01; 0.01 0.05]
    Qc = [0.25 0.04; 0.04 0.18]
    @test_logs (:warn, r"never used") match_mode = :any LQRStateModel(
        A, Sm, [Qc, Qc], Matrix(0.03I, 4, 4); schedule=fill(1, 8)
    )

    # `depends_on` is refused at every entry point rather than silently ignored.
    @test_throws ArgumentError fit!(lds, y; depends_on=(A=[1],), progress=false)
    @test_throws ArgumentError elbo(lds, y; depends_on=(A=[1],))
    @test_throws ArgumentError smooth(lds, y; depends_on=(A=[1],))
    return nothing
end

# ---------------------------------------------------------------------------
# `:free` mode
# ---------------------------------------------------------------------------

"""A `:free` state model and the `GaussianStateModel` it must reproduce, wired
to one shared emission so the only difference is the state half."""
function free_pair(rng; d::Int=4, p::Int=3, ux_dim::Int=0)
    M = T_STABLE(rng, d)
    Σ = Matrix(0.15I, d, d)
    x0 = randn(rng, d)
    P0 = Matrix(0.5I, d, d)
    C = randn(rng, p, d)
    R = Matrix(0.2I, p, p)
    dv = randn(rng, p)
    B = ux_dim > 0 ? 0.3 .* randn(rng, d, ux_dim) : zeros(d, 0)
    b = randn(rng, d)

    gsm = GaussianStateModel(;
        A=copy(M),
        Q=copy(Σ),
        b=copy(b),
        x0=copy(x0),
        P0=copy(P0),
        B=copy(B),
        Q_prior=nothing,
        P0_prior=nothing,
        AB_prior=nothing,
        x0_prior=nothing,
    )
    fsm = free_state_model(
        copy(M), copy(Σ); h=copy(b), Bu=copy(B), x0=copy(x0), P0=copy(P0)
    )
    mkobs() = GaussianObservationModel(copy(C), copy(R), copy(dv))
    return (LinearDynamicalSystem(gsm, mkobs()), LinearDynamicalSystem(fsm, mkobs()))
end

"""A mildly contractive `d × d` matrix."""
T_STABLE(rng, d) = 0.85 * Matrix(1.0I, d, d) + 0.05 * randn(rng, d, d)

function test_lqr_free_construction()
    M = 0.9 * Matrix(1.0I, 4, 4)
    Σ = Matrix(0.1I, 4, 4)
    sm = free_state_model(M, Σ)

    @test sm.mode === :free
    @test SSD._is_free(sm)
    @test plant_dim(sm) == 2
    @test SSD._state_latent_dim(sm) == 4
    # One "regime" even though `Qc` is empty — the free transition is the only one.
    @test SSD._nregimes(sm) == 1
    @test isempty(sm.A) && isempty(sm.S) && isempty(sm.Qc)
    @test !sm.terminal

    # The cache is filled the free way: the transition *is* `Mfree`, `G = I`.
    @test sm.cache.M[1] == M
    @test Matrix(sm.cache.Qfwd) ≈ Σ
    @test sm.cache.G ≈ Matrix(1.0I, 4, 4)
    @test sm.cache.logabsdetA == 0.0     # G = I carries no Jacobian

    # Rejections.
    @test_throws DimensionMismatchError free_state_model(randn(4, 3), Σ)
    @test_throws ArgumentError free_state_model(randn(3, 3), Matrix(0.1I, 3, 3))
    @test_throws DimensionMismatchError free_state_model(M, Matrix(0.1I, 3, 3))
    @test_throws DimensionMismatchError free_state_model(M, Σ; h=zeros(3))
    @test_throws ArgumentError free_state_model(M, Σ; mstep_iters=0)

    # An LQR readout has no answer in `:free` mode and must say so rather than
    # returning something built from empty matrices.
    @test_throws ArgumentError lqr_parameters(sm)
    @test_throws ArgumentError riccati_solution(sm)
    @test_throws ArgumentError closed_loop_dynamics(sm)
    @test_throws ArgumentError lqr_matrix(sm)
    @test_throws ArgumentError symplectic_defect(sm)
    @test_throws ArgumentError rescale_costate!(sm, 2.0)
    @test_throws ArgumentError simulate_lqr(sm, 5)
    @test_throws ArgumentError lqr_riccati_sequence(sm, 5)
    return nothing
end

function test_lqr_total_dim_constructor()
    for D in (2, 4, 6)
        @test plant_dim(LQRStateModel(D)) == D ÷ 2
        @test SSD._state_latent_dim(LQRStateModel(D)) == D
        @test plant_dim(LQRStateModel(D; mode=:free)) == D ÷ 2
        @test SSD._state_latent_dim(LQRStateModel(D; mode=:free)) == D
    end
    @test LQRStateModel(4).mode === :lqr
    @test LQRStateModel(4; mode=:free).mode === :free

    # The latent is the state-costate pair, so an odd total is a user error.
    @test_throws ArgumentError LQRStateModel(5)
    @test_throws ArgumentError LQRStateModel(3; mode=:free)
    @test_throws ArgumentError LQRStateModel(0)
    @test_throws ArgumentError LQRStateModel(-2)
    @test_throws ArgumentError LQRStateModel(4; mode=:nonsense)

    # Keywords reach the matrix constructor they stand in for.
    sm = LQRStateModel(4; terminal=true, observe_costate=true)
    @test sm.terminal && sm.observe_costate
    return nothing
end

"""`:free` mode is a plain linear-Gaussian state model wearing the LQR
type, so it must agree with `GaussianStateModel` exactly — not approximately.
Any divergence means the free path has invented structure of its own."""
function test_lqr_free_matches_gaussian_lds()
    for ux_dim in (0, 2)
        rng = StableRNG(4242 + ux_dim)
        ldsG, ldsF = free_pair(rng; ux_dim=ux_dim)
        tsteps, ntrials = 40, 5
        uxs = if ux_dim > 0
            [randn(StableRNG(70 + i), ux_dim, tsteps) for i in 1:ntrials]
        else
            nothing
        end
        ys = [
            rand(StableRNG(90 + i), ldsG, tsteps; ux=(uxs === nothing ? nothing : uxs[i]))[2]
            for i in 1:ntrials
        ]

        @test loglikelihood(ldsG, ys; ux=uxs) ≈ loglikelihood(ldsF, ys; ux=uxs) atol = 1e-9

        sG = smooth(ldsG, ys; ux=uxs)
        sF = smooth(ldsF, ys; ux=uxs)
        @test maximum(maximum(abs, a .- b) for (a, b) in zip(sG[1], sF[1])) < 1e-10

        # The M-step too: both updates are exact maximizers, so a full EM run
        # must track iterate for iterate, not merely end up nearby.
        eG = fit!(ldsG, ys; ux=uxs, max_iter=20, tol=1e-14)
        eF = fit!(ldsF, ys; ux=uxs, max_iter=20, tol=1e-14)
        eGv = eG isa Tuple ? eG[1] : eG
        eFv = eF isa Tuple ? eF[1] : eF
        @test length(eGv) == length(eFv)
        @test maximum(abs, eGv .- eFv) < 1e-8
        @test minimum(diff(eFv)) > -1e-8

        @test maximum(abs, ldsG.state_model.A .- ldsF.state_model.Mfree) < 1e-10
        @test maximum(abs, ldsG.state_model.Q .- ldsF.state_model.Σ) < 1e-10
        @test maximum(abs, ldsG.state_model.b .- ldsF.state_model.h) < 1e-10
        ux_dim > 0 && @test maximum(abs, ldsG.state_model.B .- ldsF.state_model.Bu) < 1e-10
    end

    # The free-state covariance uses the same inverse-Wishart MAP update as the
    # constrained LQR path. Merely storing the prior on the model is not enough:
    # omitting it here lets a switching state's covariance collapse even when the
    # caller explicitly supplied pseudo-transitions to prevent that boundary.
    rng = StableRNG(4343)
    _, lds = free_pair(rng)
    ys = [randn(StableRNG(120 + i), lds.obs_dim, 20) .* 0.4 for i in 1:4]
    hs, _, _, _ = lqr_estep_stats(lds, ys)
    sm = lds.state_model
    d = lds.latent_dim
    prior = IWPrior(Matrix(0.7I, d, d), 13.0)
    sm.Σ_prior = prior
    Theta = SSD._free_theta_pooled([lds], [hs], [1])
    R = SSD._free_residual_scatter(Theta, hs)
    expected = (R + prior.Ψ) / (prior.ν + hs.nk[1] + d + 1)
    # Its diagnostics are `@debug`, not a warning on every M-step.
    @test_logs min_level = Base.CoreLogging.Warn SSD._free_state_mstep!(lds, hs)
    @test sm.Σ ≈ expected atol = 1e-10
    return nothing
end

"""Freezing a column group of the free regression must hold exactly that group,
and still improve the ELBO for the ones left free."""
function test_lqr_free_fit_flags()
    rng = StableRNG(99)
    _, lds = free_pair(rng; ux_dim=2)
    sm = lds.state_model
    ys = [randn(StableRNG(31 + i), lds.obs_dim, 30) .* 0.5 for i in 1:4]
    uxs = [randn(StableRNG(41 + i), 2, 30) for i in 1:4]

    sm.fit_flags = LQRFitFlags(; A=true, h=false, Bu=false)
    h0, Bu0 = copy(sm.h), copy(sm.Bu)
    elbos = fit!(lds, ys; ux=uxs, max_iter=8, tol=1e-12)
    elbos = elbos isa Tuple ? elbos[1] : elbos

    @test sm.h == h0            # frozen exactly, not merely close
    @test sm.Bu == Bu0
    @test minimum(diff(elbos)) > -1e-8

    # Freezing the whole structural block leaves the transition untouched.
    M0 = copy(sm.Mfree)
    lds.fit_bool[3] = false
    fit!(lds, ys; ux=uxs, max_iter=3, tol=1e-12)
    @test sm.Mfree == M0
    return nothing
end

function test_lqr_free_show()
    sm = free_state_model(0.9 * Matrix(1.0I, 4, 4), Matrix(0.1I, 4, 4))
    str = sprint(show, sm)
    @test occursin(":free", str)
    @test occursin("Latent dim = 4", str)
    @test !occursin("symplectic defect", str)   # no such thing here
    @test !occursin("Qc[", str)

    big = free_state_model(0.9 * Matrix(1.0I, 12, 12), Matrix(0.1I, 12, 12))
    @test occursin("size(M) = (12, 12)", sprint(show, big))
    return nothing
end

# ---------------------------------------------------------------------------
# Responsibility-weighted statistics (the SLDS M-step's input)
# ---------------------------------------------------------------------------

"""Aggregate `lds`'s statistics under per-trial weights `w`."""
function lqr_weighted_stats(lds, tfs, data, w)
    hs = SSD._initialize_td_sufficient_statistics(Float64, lds, data.tsteps)
    SSD._aggregate_lqr_stats_weighted!(hs, tfs, lds, data, w)
    return hs
end

"""Largest absolute disagreement between two statistic sets, over every block."""
function lqr_stats_gap(a, b, K)
    g = maximum([maximum(abs, a.zz[k] .- b.zz[k]) for k in 1:K])
    g = max(g, maximum([maximum(abs, a.zy[k] .- b.zy[k]) for k in 1:K]))
    g = max(g, maximum([maximum(abs, a.yy[k] .- b.yy[k]) for k in 1:K]))
    g = max(g, maximum([abs(a.nk[k] - b.nk[k]) for k in 1:K]))
    g = max(g, maximum(maximum(abs, a.term_zz[k] .- b.term_zz[k]) for k in 1:K))
    return max(g, maximum(abs, a.term_n .- b.term_n))
end

"""The weighted aggregator is the SLDS M-step's only source of statistics, and a
one-index slip in the weight-to-timestep convention would bias every fit while
still looking plausible. Three properties pin it down: it must reduce to the
plain aggregator at unit weight, be additive in the weights, and vanish at zero.
"""
function test_lqr_weighted_stats()
    for (terminal, ux_dim, nregimes, onset) in (
        (false, 0, 1, 1),
        (false, 0, 2, 6),
        (true, 0, 2, 1),
        (true, 2, 2, 1),
        (true, 2, 3, 6),
    )
        rng = StableRNG(5150)
        tsteps, ntrials = 18, 4
        sm, lds = lqr_fixture(
            rng;
            terminal=terminal,
            tsteps=tsteps,
            ux_dim=ux_dim,
            nregimes=nregimes,
            onset=onset,
        )
        ys = [randn(StableRNG(7 + i), lds.obs_dim, tsteps) .* 0.4 for i in 1:ntrials]
        uxs = if ux_dim > 0
            [randn(StableRNG(17 + i), ux_dim, tsteps) for i in 1:ntrials]
        else
            nothing
        end
        plain, tfs, data, _ = lqr_estep_stats(lds, ys; ux=uxs)
        K = SSD._nregimes(sm)

        # 1. Unit weight is the plain aggregate — the convention check. If the
        #    transition out of `t` were weighted by `γ(t)` instead of `γ(t+1)`,
        #    this would still pass, so 2. below is the one that pins the index.
        ones_w = [ones(tsteps) for _ in 1:ntrials]
        @test lqr_stats_gap(plain, lqr_weighted_stats(lds, tfs, data, ones_w), K) < 1e-10

        # 2. Additive in the weights, with weights that vary within a trial so a
        #    shifted index changes the answer.
        w1 = [[0.1 + 0.8 * abs(sin(0.7t + i)) for t in 1:tsteps] for i in 1:ntrials]
        w2 = [[0.05 + 0.5 * abs(cos(0.4t - i)) for t in 1:tsteps] for i in 1:ntrials]
        s1 = lqr_weighted_stats(lds, tfs, data, w1)
        s2 = lqr_weighted_stats(lds, tfs, data, w2)
        ssum = lqr_weighted_stats(lds, tfs, data, [w1[i] .+ w2[i] for i in 1:ntrials])
        for k in 1:K
            @test maximum(abs, ssum.zz[k] .- (s1.zz[k] .+ s2.zz[k])) < 1e-10
            @test maximum(abs, ssum.zy[k] .- (s1.zy[k] .+ s2.zy[k])) < 1e-10
            @test maximum(abs, ssum.yy[k] .- (s1.yy[k] .+ s2.yy[k])) < 1e-10
            @test ssum.nk[k] ≈ s1.nk[k] + s2.nk[k]
        end
        for k in 1:K
            @test maximum(abs, ssum.term_zz[k] .- (s1.term_zz[k] .+ s2.term_zz[k])) < 1e-10
        end
        @test ssum.term_n ≈ s1.term_n + s2.term_n

        # 3. A partition of unity sums back to the plain aggregate: this is what
        #    makes a K-state SLDS with `Σₖ γₖ(t) = 1` pool to the ungrouped fit.
        wa = [[0.3 + 0.4 * abs(sin(1.1t + i)) for t in 1:tsteps] for i in 1:ntrials]
        wb = [1 .- wa[i] for i in 1:ntrials]
        sa = lqr_weighted_stats(lds, tfs, data, wa)
        sb = lqr_weighted_stats(lds, tfs, data, wb)
        for k in 1:K
            @test maximum(abs, plain.zz[k] .- (sa.zz[k] .+ sb.zz[k])) < 1e-10
            @test maximum(abs, plain.zy[k] .- (sa.zy[k] .+ sb.zy[k])) < 1e-10
            @test plain.nk[k] ≈ sa.nk[k] + sb.nk[k]
        end
        @test plain.term_n ≈ sa.term_n + sb.term_n

        # 4. Zero weight contributes nothing at all.
        zed = lqr_weighted_stats(lds, tfs, data, [zeros(tsteps) for _ in 1:ntrials])
        for k in 1:K
            @test all(iszero, zed.zz[k])
            @test all(iszero, zed.zy[k])
            @test all(iszero, zed.yy[k])
            @test zed.nk[k] == 0
        end
        @test all(iszero, zed.term_n)
    end
    return nothing
end

function test_lqr_plant_only_inputs()
    rng = StableRNG(91)
    sm, lds = lqr_fixture(rng; terminal=true, nregimes=2, tsteps=14, ux_dim=3)
    sm.Bu[3:4, :] .= 0
    sm.fit_flags = LQRFitFlags(; Gref=false, Bu_rows=1:2, Bu_cols=[1, 3])
    refresh!(sm)
    ys = [randn(rng, lds.obs_dim, 14) for _ in 1:4]
    us = [randn(rng, 3, 14) for _ in 1:4]
    hs, _, _, _ = lqr_estep_stats(lds, ys; ux=us)
    frozen = copy(sm.Bu[:, 2])
    for profile in (true, false)
        ctx = SSD._LQRMStepCtx(hs, sm, profile)
        θ = zeros(ctx.pack.np)
        SSD._lqr_pack!(θ, ctx)
        @test length(SSD._lqr_blk(ctx.pack, SSD._LQR_BLOCK_B, 1)) == 4
        g = similar(θ)
        SSD._lqr_fg!(g, θ, ctx)
        reference = ForwardDiff.gradient(t -> lqr_ref_objective(t, hs, sm, profile), θ)
        @test g ≈ reference rtol = 1e-8 atol = 1e-8
    end
    fit!(lds, ys; ux=us, max_iter=3, progress=false)
    @test iszero(sm.Bu[3:4, :])
    @test sm.Bu[:, 2] == frozen
    @test all(iszero(B[3:4, :]) for B in sm.cache.Bfwd)
    @test LQRFitFlags(; Bu_rows=[2, 1]) == LQRFitFlags(; Bu_rows=1:2)
    @test hash(LQRFitFlags(; Bu_rows=[2, 1])) == hash(LQRFitFlags(; Bu_rows=1:2))
    @test LQRFitFlags(; Bu_rows=1:2) != LQRFitFlags()
    @test_throws ArgumentError LQRFitFlags(; Bu_rows=[0])
    @test_throws ArgumentError LQRStateModel(
        sm.A, sm.S, sm.Qc, sm.Σ; schedule=sm.schedule, fit_flags=LQRFitFlags(; Bu_rows=[5])
    )
    return nothing
end

function test_lqr_terminal_regime_pin()
    rng = StableRNG(51)
    #=
    `terminal_regime` pins which cost the terminal factor is written against.

    It exists for ragged data. The schedule is one vector indexed by within-trial
    timestep, so `schedule[end] == nregimes` marks the last step of the *longest*
    trial only; every shorter trial ends under whatever running cost its own
    length lands on, and the dedicated terminal cost is then fitted from the
    maximal-length trials alone. Pinning puts every trial's last step on the same
    cost whatever its length, and must leave the transitions exactly as they were.
    =#
    tsteps = 18
    lengths = [12, 18, 15]

    sm, lds = lqr_fixture(rng; terminal=true, nregimes=3, tsteps=tsteps, onset=11)
    @test sm.terminal_regime == 0                       # off unless asked for

    # The default: only the 18-bin trial reaches the terminal regime.
    @test SSD._terminal_regime(sm, 18) == 3
    @test SSD._terminal_regime(sm, 12) == SSD._regime(sm, 12) != 3

    pinned, pinned_lds = lqr_fixture(
        rng; terminal=true, nregimes=3, tsteps=tsteps, onset=11
    )
    pinned.terminal_regime = 3
    refresh!(pinned)
    for t_n in lengths
        @test SSD._terminal_regime(pinned, t_n) == 3    # whatever the length
    end
    # Transitions are not the terminal factor and must be untouched by the pin.
    @test all(SSD._regime(pinned, t) == SSD._regime(sm, t) for t in 1:(tsteps - 1))

    #=
    The claim that matters: which `Qc` the M-step's terminal sufficient
    statistics are attributed to. By default they are split across regimes by
    trial length; pinned, all of them land on regime 3.
    =#
    ys = [randn(rng, lds.obs_dim, t) .* 0.4 for t in lengths]

    hs_default, _, _, _ = lqr_estep_stats(lds, ys)
    expected = zeros(3)
    for t_n in lengths
        expected[SSD._regime(sm, t_n)] += 1
    end
    @test hs_default.term_n ≈ expected
    @test expected[3] == 1                              # one of three trials

    hs_pinned, _, _, _ = lqr_estep_stats(pinned_lds, ys)
    @test hs_pinned.term_n ≈ [0.0, 0.0, 3.0]            # all of them
    @test sum(hs_pinned.term_n) ≈ sum(hs_default.term_n)
    # Transition statistics are the same either way.
    @test hs_pinned.nk ≈ hs_default.nk

    # And the Riccati sweep writes the pinned cost at the boundary.
    P, _, _ = lqr_riccati_sequence(pinned, 12)
    @test P[end] ≈ pinned.Qc[3]

    # EM still runs, and still climbs.
    els = fit!(pinned_lds, ys; max_iter=8, progress=false)
    @test minimum(diff(els)) > -1e-8
    @test all(isfinite, els)
    return nothing
end

function test_lqr_terminal_regime_errors()
    rng = StableRNG(52)
    A = [0.96 0.07; -0.05 0.93]
    Sm = [0.06 0.01; 0.01 0.05]
    Qc = [Matrix(0.2I, 2, 2), Matrix(0.6I, 2, 2), Matrix(1.1I, 2, 2)]
    Σ = Matrix(0.03I, 4, 4)
    sched = cost_schedule(10; terminal=true, onset=5, nregimes=3)

    # Pinning a cost for a terminal factor that does not exist.
    @test_throws ArgumentError LQRStateModel(
        A, Sm, Qc, Σ; schedule=sched, terminal=false, terminal_regime=2
    )
    # Pinning a cost index that does not exist.
    @test_throws ArgumentError LQRStateModel(
        A, Sm, Qc, Σ; schedule=sched, terminal=true, terminal_regime=4
    )
    @test_throws ArgumentError LQRStateModel(
        A, Sm, Qc, Σ; schedule=sched, terminal=true, terminal_regime=-1
    )
    # Zero is always allowed: it is the default, "follow the schedule".
    sm = LQRStateModel(A, Sm, Qc, Σ; schedule=sched, terminal=true, terminal_regime=0)
    @test sm.terminal_regime == 0

    #=
    A regime reached only as a pinned terminal *is* used, so the "never used by
    the schedule" warning must not fire for it — that warning would be telling
    the user to drop exactly the cost they asked to fit. The schedule here points
    every transition at regime 1, so regime 2 is reached only as the pin.
    =#
    two = [Matrix(0.2I, 2, 2), Matrix(0.6I, 2, 2)]
    running_only = ones(Int, 10)
    pinned = @test_logs LQRStateModel(
        A, Sm, two, Σ; schedule=running_only, terminal=true, terminal_regime=2
    )
    @test pinned.terminal_regime == 2
    @test SSD._terminal_regime(pinned, 7) == 2
    # Without the pin the same model warns, because then nothing reaches Qc[2].
    @test_logs (:warn, r"Qc\[2\] is never used") LQRStateModel(
        A, Sm, two, Σ; schedule=running_only, terminal=true
    )
    return nothing
end

function test_lqr_ragged_riccati_terminal()
    rng = StableRNG(92)
    sm, _ = lqr_fixture(rng; terminal=true, nregimes=3, tsteps=18, onset=10, ux_dim=2)
    sm.Gref .= 0.2 .* randn(rng, 2, 2)
    refresh!(sm)
    us = randn(rng, 2, 12)
    P, g, _ = lqr_riccati_sequence(sm, 12; ux=us)
    kT = SSD._regime(sm, 12)
    @test P[end] ≈ sm.Qc[kT]
    @test g[end] ≈ sm.hf - sm.Qc[kT] * sm.Gref * us[:, end]
    z = simulate_lqr(rng, sm, 12; ux=us, process_noise=false, x1=zeros(2))
    residual = zeros(2)
    SSD._terminal_residual!(residual, sm, z, us)
    @test norm(residual) < 1e-10
    return nothing
end
