#=============================================================================
Infinite-horizon ("hold") inverse-LQR latents: `LQRStateModel` in `:hold` mode.

As for the finite-horizon form, every check is against something that does not
share the implementation under test:

* the DARE solver against a brute-force backward Riccati recursion, and the
  stationary feedforward against the finite-horizon sweep run long enough to
  forget its endpoint;
* the forward cache against a direct substitution through the plant row and the
  manifold row, and the ELBO against an explicit Gaussian LDS carrying the
  materialized forward parameters;
* the structural gradient — DARE adjoint included — against central differences
  of the packed objective, alone and jointly with a finite-horizon unit;
* EM against data the model generates, alone and in a switching model whose
  control and hold states share one plant.
=============================================================================#

const HOLD_A = [0.96 0.07; -0.05 0.93]
const HOLD_S = [0.3 0.05; 0.05 0.25]
const HOLD_Q = [0.8 0.1; 0.1 0.6]

"""A hold model with every affine piece populated, and an LDS around it."""
function hold_fixture(
    rng; m::Int=2, p::Int=4, observe_costate::Bool=false, fit_flags=LQRFitFlags()
)
    n = 2
    d = 2n
    sm = hold_state_model(
        copy(HOLD_A),
        copy(HOLD_S),
        copy(HOLD_Q),
        Matrix(Diagonal(fill(0.03, d)));
        h=0.05 .* randn(rng, d),
        Bu=0.3 .* randn(rng, d, m),
        Gref=0.5 .* randn(rng, n, m),
        x0=0.1 .* randn(rng, d),
        P0=Matrix(0.25I, d, d),
        observe_costate=observe_costate,
        fit_flags=fit_flags,
    )
    C = randn(rng, p, d)
    observe_costate || (C[:, (n + 1):d] .= 0)
    om = GaussianObservationModel(C, Matrix(0.08I, p, p), 0.1 .* randn(rng, p))
    return sm, LinearDynamicalSystem(sm, om)
end

"""P by the plain backward Riccati recursion, iterated until it stops moving."""
function hold_bruteforce_dare(A, S, Q)
    P = copy(Q)
    for _ in 1:100_000
        Pn = Q + A' * P * ((I + S * P) \ A)
        Pn = (Pn + Pn') / 2
        maximum(abs, Pn - P) < 1e-15 * max(1, maximum(abs, P)) && return Pn
        P = Pn
    end
    return P
end

"""One E-step's statistics with the forward blocks filled, as the M-step sees them."""
function hold_estep_stats(lds, ys; ux=nothing)
    data = SSD.Data(lds, ys; ux=ux)
    SSD._prepare_lqr!(lds, data.tsteps)
    tfs = SSD.initialize_FilterSmooth(lds, data.tsteps)
    pool = SSD._lqr_sws_pool(lds, data)
    hs = SSD._initialize_td_sufficient_statistics(Float64, lds, data.tsteps)
    SSD._td_init_const_blocks!(pool[1], lds, data)
    SSD.estep!(lds, hs, tfs, data, pool)
    SSD._fill_mixed_blocks!(hs, lds.state_model)
    return hs
end

"""Central differences of the packed objective, the ground truth for its gradient."""
function hold_fd_gradient(ctx, θ; h=1e-6)
    g = similar(θ)
    for i in eachindex(θ)
        tp = copy(θ)
        tm = copy(θ)
        tp[i] += h
        tm[i] -= h
        g[i] = (SSD._lqr_fg!(nothing, tp, ctx) - SSD._lqr_fg!(nothing, tm, ctx)) / (2h)
    end
    return g
end

function hold_gradient_error(ctx, rng; jitter=0.05)
    θ = zeros(ctx.pack.np)
    SSD._lqr_pack!(θ, ctx)
    θ .+= jitter .* randn(rng, length(θ))
    g = similar(θ)
    f = SSD._lqr_fg!(g, θ, ctx)
    g_fd = hold_fd_gradient(ctx, θ)
    return f, maximum(abs, g .- g_fd) / max(1.0, maximum(abs, g_fd))
end

"""The state the held regulator settles to under a constant input `ũ = [1; u]`:
`x* = (I − A_cl)⁻¹ W (F − S G) ũ`. Invariant to the costate gauge, and what the
data identify when the emission reads only the state."""
function hold_setpoint(sm)
    H = SSD._hold_unit_at(sm)
    return (I - H.Acl) \ (H.W * (H.F - sm.S * H.Gm))
end

# ---------------------------------------------------------------------------
# (1) DARE
# ---------------------------------------------------------------------------

function test_hold_dare()
    rng = StableRNG(101)
    sm, _ = hold_fixture(rng)
    P = riccati_solution(sm)
    Pref = hold_bruteforce_dare(HOLD_A, HOLD_S, HOLD_Q)
    @test maximum(abs, P .- Pref) < 1e-10
    @test P ≈ P'
    @test isposdef(Symmetric(P))
    # It satisfies the equation, and its closed loop is strictly stable.
    W = inv(I + HOLD_S * P)
    @test maximum(abs, P .- (HOLD_Q .+ HOLD_A' * P * W * HOLD_A)) < 1e-12
    Acl = closed_loop_dynamics(sm)
    @test Acl ≈ W * HOLD_A
    @test maximum(abs, eigvals(Acl)) < 1

    # The finite-horizon model's fixed-point solver agrees.
    lqr = LQRStateModel(copy(HOLD_A), copy(HOLD_S), copy(HOLD_Q), Matrix(0.03I, 4, 4))
    @test maximum(abs, riccati_solution(lqr) .- P) < 1e-9

    # A larger, less benign plant: an open-loop unstable mode.
    n = 4
    A = Matrix(0.9I, n, n) .+ 0.2 .* randn(rng, n, n)
    A[1, 1] = 1.3
    B = randn(rng, n, 2)
    S = B * B' + 1e-3I
    Q = Matrix(0.5I, n, n)
    big = hold_state_model(A, S, Q, Matrix(0.05I, 2n, 2n))
    @test maximum(abs, riccati_solution(big) .- hold_bruteforce_dare(A, S, Q)) < 1e-8
    @test maximum(abs, eigvals(closed_loop_dynamics(big))) < 1

    # No stabilizing solution: an unstable mode the control cannot reach.
    A_bad = [0.9 0.0; 0.0 1.5]
    S_bad = [1.0 0.0; 0.0 0.0]
    @test_throws SSD.NumericalStabilityError hold_state_model(
        A_bad, S_bad, Matrix(1.0I, 2, 2), Matrix(0.05I, 4, 4)
    )
    H = SSD._HoldUnit(Float64, 2, 0)
    @test !SSD._dare_doubling!(H.P, A_bad, S_bad, Matrix(1.0I, 2, 2), H)

    # The adjoint Stein solve.
    Y = similar(Acl)
    C = [1.0 0.3; 0.3 2.0]
    SSD._stein_adjoint!(Y, Acl, C, SSD._HoldUnit(Float64, 2, 0))
    @test maximum(abs, Y .- Acl * Y * Acl' .- C) < 1e-12
    return nothing
end

# ---------------------------------------------------------------------------
# (2) The cache
# ---------------------------------------------------------------------------

function test_hold_cache()
    rng = StableRNG(102)
    m = 2
    sm, lds = hold_fixture(rng; m=m)
    n, d = 2, 4
    c = sm.cache
    P = hold_bruteforce_dare(sm.A, sm.S, sm.Qc[1])
    W = inv(I + sm.S * P)
    Acl = W * sm.A

    #=
    The stationary feedforward against the finite-horizon sweep of the same
    control problem: run long enough under a constant input, its start forgets
    the free endpoint, so `P_1 → P` and `g_1 → G [1; u]`.
    =#
    u = randn(rng, m)
    lqr = LQRStateModel(
        copy(sm.A),
        copy(sm.S),
        copy(sm.Qc[1]),
        Matrix(0.03I, d, d);
        h=copy(sm.h),
        Bu=copy(sm.Bu),
        Gref=copy(sm.Gref),
    )
    Tlong = 400
    Pseq, gseq, _ = lqr_riccati_sequence(lqr, Tlong; ux=repeat(u, 1, Tlong))
    Gm = lqr_parameters(sm).feedforward
    @test maximum(abs, Pseq[1] .- P) < 1e-9
    @test maximum(abs, gseq[1] .- Gm * [1; u]) < 1e-8

    #=
    One transition by direct substitution through the two rows,
        x' = A x − S λ' + F ũ + ε_x,      λ' = P x' + G ũ + ε_λ,
    against the cache's forward map. The noise map is read off the same
    substitution applied to unit innovations.
    =#
    F = hcat(sm.h[1:n], sm.Bu[1:n, :])
    function step(z, ũ, ε)
        x = z[1:n]
        xn = (I + sm.S * P) \ (sm.A * x + F * ũ + ε[1:n] - sm.S * (Gm * ũ + ε[(n + 1):d]))
        λn = P * xn + Gm * ũ + ε[(n + 1):d]
        return vcat(xn, λn)
    end
    z = randn(rng, d)
    ũ = [1; u]
    ε = 0.1 .* randn(rng, d)
    J = reduce(hcat, [step(zeros(d), zeros(1 + m), e) for e in eachcol(Matrix(1.0I, d, d))])
    forward = c.M[1] * z .+ c.bfwd .+ c.Bfwd[1] * u .+ J * ε
    @test maximum(abs, step(z, ũ, ε) .- forward) < 1e-12
    @test maximum(abs, Matrix(c.Qfwd) .- J * sm.Σ * J') < 1e-12
    @test maximum(abs, c.G .- J) < 1e-12
    @test c.M[1] ≈ [Acl zeros(n, n); P*Acl zeros(n, n)]
    @test maximum(abs, eigvals(c.M[1])) < 1
    @test c.logabsdetA ≈ logdet(I + sm.S * P)

    # The ELBO is the exact marginal of the forward chain the cache describes.
    gsm = GaussianStateModel(
        Matrix(c.M[1]), Matrix(c.Qfwd), copy(c.bfwd), copy(sm.x0), Matrix(sm.P0)
    )
    gsm.B = copy(c.Bfwd[1])
    om = lds.obs_model
    ref = LinearDynamicalSystem(
        gsm, GaussianObservationModel(copy(om.C), copy(om.R), copy(om.d))
    )
    ys = [randn(rng, lds.obs_dim, 12) .* 0.4 for _ in 1:3]
    uxs = [randn(rng, m, 12) for _ in 1:3]
    @test elbo(lds, ys; ux=uxs) ≈ elbo(ref, ys; ux=uxs) atol = 1e-8
    @test loglikelihood(lds, ys; ux=uxs) ≈ loglikelihood(ref, ys; ux=uxs) atol = 1e-8

    # A hold model's forward flow is stable: long draws stay bounded, and quietly.
    zs, _ = @test_logs min_level = Base.CoreLogging.Warn rand(
        StableRNG(1), lds, 2000; ux=randn(rng, m, 2000)
    )
    @test maximum(abs, zs) < 50

    #=
    `simulate_lqr` follows the stationary policy: with no noise and no slack the
    costate sits exactly on the manifold `λ_t = P x_t + G ũ_{t−1}`.
    =#
    ux = randn(rng, m, 30)
    zsim = simulate_lqr(StableRNG(2), sm, 30; process_noise=false, ux=ux)
    for t in 2:30
        @test zsim[(n + 1):d, t] ≈ P * zsim[1:n, t] .+ Gm * [1; ux[:, t - 1]]
    end
    return nothing
end

# ---------------------------------------------------------------------------
# (3) The structural objective and its gradient
# ---------------------------------------------------------------------------

function test_hold_mstep_gradient()
    rng = StableRNG(103)
    tsteps, m = 15, 2
    sm, lds = hold_fixture(rng; m=m)
    ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:4]
    uxs = [randn(rng, m, tsteps) for _ in 1:4]
    hs = hold_estep_stats(lds, ys; ux=uxs)
    n, d = 2, 4
    N = sum(hs.nk)

    #=
    The value, independently: `R_h = Lh R^z Lhᵀ`, so the profiled objective is
    `(N/2) log det R^z` over the forward residual scatter, and the plain one
    `½ tr((Q^fwd)⁻¹ R^z) − N log det(I + S P)` — both from the cache alone.
    =#
    c = sm.cache
    Th = hcat(c.M[1], c.bfwd, c.Bfwd[1])
    Rz = hs.yy[1] - Th * hs.zy[1] - hs.zy[1]' * Th' + Th * hs.zz[1] * Th'
    for profile in (true, false)
        ctx = SSD._LQRMStepCtx(hs, sm, profile)
        θ = zeros(ctx.pack.np)
        SSD._lqr_pack!(θ, ctx)
        f = SSD._lqr_fg!(nothing, θ, ctx)
        ref = if profile
            0.5 * N * logdet(Symmetric(Rz))
        else
            0.5 * tr(Matrix(c.Qfwd) \ Rz) - N * c.logabsdetA
        end
        @test f ≈ ref rtol = 1e-10
    end

    # The gradient, with and without priors, away from the packed point.
    for profile in (true, false),
        (sigma_prior, qc_prior) in (
            (nothing, nothing),
            (IWPrior(Matrix(0.3I, d, d), 12.0), IWPrior(Matrix(0.4I, n, n), 9.0)),
        )

        profile || sigma_prior === nothing || continue
        sm.Σ_prior = sigma_prior
        sm.Qc_prior = qc_prior
        ctx = SSD._LQRMStepCtx(hs, sm, profile)
        f, err = hold_gradient_error(ctx, rng)
        @test isfinite(f)
        @test err < 1e-7
    end
    sm.Σ_prior = nothing
    sm.Qc_prior = nothing

    # Freezing composes: a frozen block is simply absent from the gradient.
    sm.fit_flags = LQRFitFlags(; S=false, Gref_cols=[2], Bu_rows=1:n)
    ctx = SSD._LQRMStepCtx(hs, sm, true)
    _, err = hold_gradient_error(ctx, rng)
    @test err < 1e-7
    sm.fit_flags = LQRFitFlags()

    #=
    A point with no stabilizing DARE solution is infeasible, not an error. With
    `S` frozen singular, pushing the uncontrolled mode of `A` past the unit circle
    leaves nothing to stabilize it.
    =#
    smx = hold_state_model(
        [0.9 0.0; 0.0 0.9],
        [1.0 0.0; 0.0 0.0],
        Matrix(1.0I, 2, 2),
        Matrix(0.05I, 4, 4);
        fit_flags=LQRFitFlags(; S=false),
    )
    Cx = randn(rng, 3, 4)
    Cx[:, 3:4] .= 0
    ldsx = LinearDynamicalSystem(
        smx, GaussianObservationModel(Cx, Matrix(0.1I, 3, 3), zeros(3))
    )
    hsx = hold_estep_stats(ldsx, [randn(rng, 3, 10) for _ in 1:2])
    ctx = SSD._LQRMStepCtx(hsx, smx, true)
    θ = zeros(ctx.pack.np)
    SSD._lqr_pack!(θ, ctx)
    @test isfinite(SSD._lqr_fg!(nothing, θ, ctx))
    θ[SSD._lqr_blk(ctx.pack, SSD._LQR_BLOCK_A, 1)[4]] = 1.5
    g = similar(θ)
    @test SSD._lqr_fg!(g, θ, ctx) == Inf
    @test all(iszero, g)
    return nothing
end

"""A hold unit and a finite-horizon unit in one context, as an `SLDS` builds it."""
function test_hold_joint_gradient()
    rng = StableRNG(104)
    tsteps, m, p = 15, 2, 4
    smh, ldsh = hold_fixture(rng; m=m, p=p)
    n, d = 2, 4
    #= The finite-horizon member carries a running and a terminal cost, so the
    two cost copies differ in length — the case the per-copy regime count is
    for. =#
    sml = LQRStateModel(
        HOLD_A .+ 0.01,
        1.1 .* HOLD_S,
        [copy(HOLD_Q), 1.3 .* HOLD_Q],
        Matrix(Diagonal(fill(0.03, d)));
        schedule=cost_schedule(tsteps; terminal=true),
        terminal=true,
        condition_terminal=false,
        h=0.05 .* randn(rng, d),
        Bu=0.3 .* randn(rng, d, m),
        Gref=0.5 .* randn(rng, n, m),
        Σf=Matrix(0.04I, n, n),
        hf=0.05 .* randn(rng, n),
        P0=Matrix(0.25I, d, d),
    )
    ldsl = LinearDynamicalSystem(sml, ldsh.obs_model)
    ys = [randn(rng, p, tsteps) .* 0.4 for _ in 1:4]
    uxs = [randn(rng, m, tsteps) for _ in 1:4]
    hsh = hold_estep_stats(ldsh, ys; ux=uxs)
    hsl = hold_estep_stats(ldsl, ys; ux=uxs)

    for tied in ([:A, :S], [:A, :S, :Gref, :h, :Bu], [:A, :S, :Qc]),
        noise in ([1, 2], [1, 1]),
        profile in (true, false)

        ctx = SSD._LQRMStepCtx(
            [hsl, hsh], [sml, smh], SSD._lqr_block_slots(tied, 2), noise, profile
        )
        # One plant: the hold unit's `A` is the finite-horizon unit's copy.
        @test ctx.pack.nv[SSD._LQR_BLOCK_A] == 1
        @test ctx.pack.Kq == (:Qc in tied ? [2] : [2, 1])
        f, err = hold_gradient_error(ctx, rng; jitter=0.03)
        @test isfinite(f)
        @test err < 1e-7
    end

    # Write-back reaches both models and leaves the hold cost its own.
    ctx = SSD._LQRMStepCtx(
        [hsl, hsh], [sml, smh], SSD._lqr_block_slots([:A, :S], 2), [1, 2], true
    )
    SSD._lqr_structure_mstep!(ctx, true, 20)
    @test sml.A == smh.A
    @test sml.S == smh.S
    @test sml.Qc[1] != smh.Qc[1]
    @test length(smh.Qc) == 1
    return nothing
end

# ---------------------------------------------------------------------------
# (4) EM on data the model generates
# ---------------------------------------------------------------------------

function test_hold_em_recovery()
    rng = StableRNG(105)
    n, d, m, p = 2, 4, 2, 6
    tsteps, ntrials = 30, 16
    Gref = [1.0 0.2; -0.3 1.0]
    flags = LQRFitFlags(; Bu=false, h=false)
    truth_sm = hold_state_model(
        copy(HOLD_A),
        copy(HOLD_S),
        copy(HOLD_Q),
        Matrix(Diagonal([0.02, 0.02, 0.01, 0.01]));
        Bu=zeros(d, m),
        Gref=Gref,
        P0=Matrix(0.25I, d, d),
        fit_flags=flags,
    )
    C = randn(rng, p, d)
    C[:, (n + 1):d] .= 0
    truth = LinearDynamicalSystem(
        truth_sm, GaussianObservationModel(C, Matrix(0.02I, p, p), zeros(p))
    )
    # A constant target per trial: the regime a hold model describes.
    uxs = [repeat(randn(rng, m), 1, tsteps) for _ in 1:ntrials]
    _, ys = rand(rng, truth, fill(tsteps, ntrials); ux=uxs)

    sm = hold_state_model(
        HOLD_A .+ 0.03 .* randn(rng, n, n),
        1.3 .* HOLD_S,
        0.7 .* HOLD_Q,
        Matrix(0.05I, d, d);
        Bu=zeros(d, m),
        Gref=0.6 .* Gref,
        P0=Matrix(0.25I, d, d),
        fit_flags=flags,
    )
    lds = LinearDynamicalSystem(
        sm, GaussianObservationModel(copy(C), Matrix(0.05I, p, p), zeros(p))
    )
    els = fit!(lds, ys; ux=uxs, max_iter=50, progress=false)
    @test all(isfinite, els)
    @test minimum(diff(els)) > -1e-8
    @test els[end] > els[1]
    @test els[end] ≈ elbo(lds, ys; ux=uxs) atol = 1e-7
    @test sm.mode === :hold
    @test maximum(abs, eigvals(closed_loop_dynamics(sm))) < 1
    #=
    What the data identify when only the state is read: the closed loop and the
    state the regulator holds for each target. `S` and `P` separately — and so
    `Gref` — are not, any more than in the finite-horizon form.
    =#
    @test maximum(abs, closed_loop_dynamics(sm) .- closed_loop_dynamics(truth_sm)) < 0.06
    @test maximum(abs, hold_setpoint(sm) .- hold_setpoint(truth_sm)) < 0.15
    return nothing
end

# ---------------------------------------------------------------------------
# (5) Switching: a control state and a hold state sharing one plant
# ---------------------------------------------------------------------------

"""A control state that regulates to the origin and a hold state that holds the
plant at `target`, sharing one plant and one readout."""
function hold_slds(;
    condition_terminal::Bool=false, terminal::Bool=false, p::Int=4, target=[1.5, -1.0]
)
    n, d = 2, 4
    C = randn(StableRNG(11), p, d)
    C[:, (n + 1):d] .= 0
    obs() = GaussianObservationModel(copy(C), Matrix(0.1I, p, p), zeros(p))
    lqr = LQRStateModel(
        copy(HOLD_A),
        copy(HOLD_S),
        [0.25 0.04; 0.04 0.18],
        Matrix(0.05I, d, d);
        terminal=terminal,
        condition_terminal=condition_terminal,
        Σf=Matrix(0.02I, n, n),
        P0=Matrix(0.3I, d, d),
    )
    # A tracking cost on a constant target: h_λ = −Q_h r.
    hold = hold_state_model(
        copy(HOLD_A),
        copy(HOLD_S),
        copy(HOLD_Q),
        Matrix(0.05I, d, d);
        h=vcat(zeros(n), -HOLD_Q * target),
        P0=Matrix(0.3I, d, d),
    )
    return SLDS(;
        A=[0.95 0.05; 0.05 0.95],
        πₖ=[0.5, 0.5],
        LDSs=[LinearDynamicalSystem(lqr, obs()), LinearDynamicalSystem(hold, obs())],
    )
end

"""Reach-then-hold trials: `t1` steps of the control state's optimal trajectory
from a random start, then `t2` of the hold state's from where it ended."""
function hold_slds_data(slds, ntrials, t1, t2; seed=900)
    a, b = slds.LDSs[1].state_model, slds.LDSs[2].state_model
    om = slds.LDSs[1].obs_model
    rng = StableRNG(seed)
    return map(1:ntrials) do _
        z1 = simulate_lqr(rng, a, t1; x1=2 .* randn(rng, 2), costate_slack=0.05)
        z2 = simulate_lqr(rng, b, t2 + 1; x1=z1[1:2, end], costate_slack=0.05)
        z = hcat(z1, z2[:, 2:end])
        return om.C * z .+ 0.2 .* randn(rng, size(om.C, 1), size(z, 2))
    end
end

function test_hold_slds()
    t1, t2 = 18, 12
    ys = hold_slds_data(hold_slds(), 6, t1, t2)
    slds = hold_slds()
    @test validate_SLDS(slds) === nothing
    a, b = slds.LDSs[1].state_model, slds.LDSs[2].state_model
    els = fit!(
        slds,
        ys;
        max_iter=10,
        progress=false,
        rng=StableRNG(7),
        num_samples=4,
        tied_params=[:A, :S, :C, :d, :R],
    )
    els = els isa Tuple ? els[1] : els
    @test all(isfinite, els)
    # Monotone up to the E-step's Monte Carlo noise.
    @test minimum(diff(els)) > -1e-3
    @test els[end] > els[1]
    @test a.mode === :lqr && b.mode === :hold
    @test a.A == b.A                         # one plant, bit for bit
    @test a.S == b.S
    @test a.A != HOLD_A                      # and it was fitted
    @test a.Qc[1] != b.Qc[1]                 # each state its own cost
    @test symplectic_defect(a) < 1e-10
    @test maximum(abs, eigvals(closed_loop_dynamics(b))) < 1
    @test all(iszero, slds.LDSs[2].obs_model.C[:, 3:4])

    #=
    Both states stay in use, so the shared plant really is fitted from both:
    the hold state owns the held segment, and the control state a good part of
    the reach (the boundary between the two is genuinely blurred — a regulator
    near its goal is a hold).
    =#
    γ = smooth(slds, ys).γ
    hold_share = sum(g -> g[2, :], γ) ./ length(γ)
    @test sum(1 .- hold_share[1:t1]) / t1 > 0.35
    @test sum(hold_share[(t1 + 2):end]) / (t2 - 1) > 0.75

    # A terminal factor on the control state is fine unconditioned...
    @test validate_SLDS(hold_slds(; terminal=true)) === nothing
    # ...but terminal conditioning is not implemented with a hold state, and says so.
    cond = hold_slds(; terminal=true, condition_terminal=true)
    @test_throws ArgumentError validate_SLDS(cond)
    @test_throws ArgumentError fit!(cond, ys; max_iter=2, progress=false, rng=StableRNG(7))
    return nothing
end

# ---------------------------------------------------------------------------
# (6) Construction, validation and the readouts
# ---------------------------------------------------------------------------

function test_hold_construction()
    rng = StableRNG(106)
    A, S, Q = copy(HOLD_A), copy(HOLD_S), copy(HOLD_Q)
    Σ = Matrix(0.05I, 4, 4)

    sm = hold_state_model(A, S, Q, Σ)
    @test sm.mode === :hold
    @test !sm.terminal
    @test isempty(sm.schedule)
    @test length(sm.Qc) == 1
    @test SSD._nregimes(sm) == 1
    @test plant_dim(sm) == 2
    @test hold_state_model(A, S, [Q], Σ).Qc[1] == Q

    # By total latent dimension.
    sm4 = LQRStateModel(4; mode=:hold)
    @test sm4.mode === :hold && plant_dim(sm4) == 2
    @test_throws ArgumentError LQRStateModel(4; mode=:bogus)
    @test_throws ArgumentError LQRStateModel(4; mode=:hold, terminal=true)

    # What a hold model cannot be.
    @test_throws ArgumentError hold_state_model(A, S, Q, Σ; terminal=true)
    @test_throws ArgumentError hold_state_model(A, S, Q, Σ; schedule=[1, 1, 1])
    @test_throws ArgumentError hold_state_model(A, S, [Q, Q], Σ)
    @test_throws SSD.NotSymmetricError hold_state_model(A, S, [1.0 0.5; 0.0 1.0], Σ)
    @test_throws SSD.DimensionMismatchError hold_state_model(A, S, Q, Matrix(0.05I, 3, 3))
    @test_throws SSD.DimensionMismatchError hold_state_model(A, S, Q, Σ; h=zeros(3))
    @test_throws SSD.DimensionMismatchError hold_state_model(
        A, S, Q, Σ; Bu=zeros(4, 2), Gref=zeros(3, 2)
    )
    @test_throws SSD.DimensionMismatchError hold_state_model(
        A, S, Q, Σ; Bu=zeros(4, 2), Gref=zeros(2, 3)
    )
    @test_throws ArgumentError hold_state_model(A, S, Q, Σ; mstep_iters=0)
    # A singular plant is fine here: the hold transition never inverts `A`.
    @test hold_state_model([0.5 0.0; 0.0 0.0], S, Q, Σ).mode === :hold

    # Validation catches a hand-edited model.
    om = GaussianObservationModel(randn(rng, 3, 4), Matrix(0.1I, 3, 3), zeros(3))
    lds = LinearDynamicalSystem(hold_state_model(A, S, Q, Σ), om)
    lds.state_model.terminal = true
    @test_throws ArgumentError validate_LDS(lds)
    lds.state_model.terminal = false
    push!(lds.state_model.Qc, copy(Q))
    @test_throws ArgumentError validate_LDS(lds)

    # Readouts.
    params = lqr_parameters(sm)
    @test params.P ≈ riccati_solution(sm)
    @test params.A_cl ≈ closed_loop_dynamics(sm)
    @test size(params.feedforward) == (2, 1)
    @test_throws ArgumentError riccati_solution(sm; k=2)
    @test_throws ArgumentError symplectic_defect(sm)
    io = IOBuffer()
    show(io, sm)
    out = String(take!(io))
    @test occursin(":hold", out)
    @test occursin("ρ(A_cl)", out)
    return nothing
end

"""The costate gauge `λ → cλ` is an exact symmetry of a hold model too."""
function test_hold_gauge()
    rng = StableRNG(107)
    m = 2
    sm, lds = hold_fixture(rng; m=m)
    ys = [randn(rng, lds.obs_dim, 12) .* 0.4 for _ in 1:3]
    uxs = [randn(rng, m, 12) for _ in 1:3]
    before = elbo(lds, ys; ux=uxs)
    P = riccati_solution(sm)
    Acl = closed_loop_dynamics(sm)
    rescale_costate!(sm, 2.5)
    @test elbo(lds, ys; ux=uxs) ≈ before atol = 1e-8
    @test riccati_solution(sm) ≈ 2.5 .* P
    @test closed_loop_dynamics(sm) ≈ Acl
    rescale_costate!(sm; target=:trace)
    @test tr(sm.Qc[1]) ≈ 2
    @test elbo(lds, ys; ux=uxs) ≈ before atol = 1e-8
    return nothing
end

"""`depends_on` grouping runs hold units through the same joint context."""
function test_hold_depends_on()
    rng = StableRNG(108)
    tsteps, ntrials, m = 14, 8, 2
    sm, lds = hold_fixture(rng; m=m)
    ys = [randn(rng, lds.obs_dim, tsteps) .* 0.4 for _ in 1:ntrials]
    uxs = [randn(rng, m, tsteps) for _ in 1:ntrials]
    session = repeat([1, 2]; inner=ntrials ÷ 2)
    set_depends_on!(sm, (Qc=session,))
    grouped = LinearDynamicalSystem(sm, lds.obs_model)
    els = fit!(grouped, ys; ux=uxs, max_iter=6, progress=false)
    @test all(isfinite, els)
    @test minimum(diff(els)) > -1e-8
    v1, v2 = group_variant(sm, :Qc, 1), group_variant(sm, :Qc, 2)
    @test v1.mode === :hold && v2.mode === :hold
    @test v1.A === v2.A                      # the plant is shared
    @test v1.Qc[1] != v2.Qc[1]               # the cost is not
    return nothing
end
