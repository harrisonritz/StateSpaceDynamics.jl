"""
Tests for the Kalman/RTS E-step backend for Gaussian LDS.

The Kalman path is enabled with `kalman_filter=true`. It should:
- match the block-tridiagonal (Newton) smoother to numerical tolerance
  when no inputs are present,
- share covariance storage across trials (reference identity),
- accept and learn `B`, `B0` input matrices via the extended M-step,
- reject invalid configurations (Poisson obs, missing `u` when `B` is set).
"""

function _make_toy_lds(; kalman_filter::Bool, D::Int=3, p::Int=5, seed::Int=7,
                      B=nothing, B0=nothing)
    Random.seed!(seed)
    sm = GaussianStateModel(;
        A=0.8*Matrix{Float64}(I, D, D),
        Q=0.2*Matrix{Float64}(I, D, D),
        x0=zeros(D),
        P0=Matrix{Float64}(I, D, D),
        b=zeros(D),
        B=B,
        B0=B0,
    )
    om = GaussianObservationModel(;
        C=randn(p, D),
        R=0.5*Matrix{Float64}(I, p, p),
        d=zeros(p),
    )
    return LinearDynamicalSystem(sm, om; kalman_filter=kalman_filter)
end

function _simulate_lds(lds::LinearDynamicalSystem, T::Int, N::Int; seed::Int=42,
                      u=nothing, u0=nothing)
    Random.seed!(seed)
    D = lds.latent_dim
    p = lds.obs_dim
    A = lds.state_model.A
    b = lds.state_model.b
    Q = lds.state_model.Q
    C = lds.obs_model.C
    d = lds.obs_model.d
    R = lds.obs_model.R
    P0 = lds.state_model.P0
    x0 = lds.state_model.x0
    B = lds.state_model.B
    B0 = lds.state_model.B0

    Lq = cholesky(Q).L
    Lr = cholesky(R).L
    Lp0 = cholesky(P0).L
    y = zeros(p, T, N)
    for n in 1:N
        x0_eff = B0 === nothing ? copy(x0) : x0 .+ B0 * u0[:, n]
        x = x0_eff .+ Lp0 * randn(D)
        y[:, 1, n] = C * x + d + Lr * randn(p)
        for t in 2:T
            bu = B === nothing ? zero(b) : B * u[:, t-1, n]
            x = A * x + b + bu + Lq * randn(D)
            y[:, t, n] = C * x + d + Lr * randn(p)
        end
    end
    return y
end

function test_kalman_smooth_agrees_with_newton()
    D, p, T, N = 3, 5, 40, 4
    lds_kf = _make_toy_lds(; kalman_filter=true)
    lds_bt = _make_toy_lds(; kalman_filter=false)
    y = _simulate_lds(lds_kf, T, N)

    elbos_kf = fit!(lds_kf, y; max_iter=1, progress=false)
    elbos_bt = fit!(lds_bt, y; max_iter=1, progress=false)

    # Single-iteration ELBO should match across backends modulo tol_PD floor.
    @test abs(elbos_kf[1] - elbos_bt[1]) < 1e-3
end

function test_kalman_fit_matches_newton()
    D, p, T, N = 3, 5, 40, 4
    lds_kf = _make_toy_lds(; kalman_filter=true)
    lds_bt = _make_toy_lds(; kalman_filter=false)
    y = _simulate_lds(lds_kf, T, N)

    elbos_kf = fit!(lds_kf, y; max_iter=20, progress=false)
    elbos_bt = fit!(lds_bt, y; max_iter=20, progress=false)

    # Both paths must be monotone-ish (small tol_PD slop).
    @test all(diff(elbos_kf) .>= -1e-4)
    @test all(diff(elbos_bt) .>= -1e-4)
    # Final ELBOs agree to a loose tolerance (floor-induced drift).
    @test abs(elbos_kf[end] - elbos_bt[end]) < 1e-2
    # Learned parameters agree to a similarly loose tolerance.
    @test maximum(abs.(lds_kf.state_model.A .- lds_bt.state_model.A)) < 1e-3
    @test maximum(abs.(lds_kf.obs_model.C .- lds_bt.obs_model.C)) < 1e-3
end

function test_kalman_covariance_shared_across_trials()
    D, p, T, N = 3, 5, 20, 3
    lds = _make_toy_lds(; kalman_filter=true)
    y = _simulate_lds(lds, T, N)

    tsteps, ntrials = size(y, 2), size(y, 3)
    kws = StateSpaceDynamics.KalmanWorkspace(lds, tsteps, ntrials)
    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, tsteps, ntrials)
    StateSpaceDynamics.smooth!(lds, tfs, y, kws)

    # All trials alias the same underlying covariance storage.
    @test tfs[1].p_smooth === tfs[2].p_smooth
    @test tfs[1].p_smooth === tfs[3].p_smooth
    @test tfs[1].p_smooth_tt1 === tfs[2].p_smooth_tt1
    @test tfs[1].entropy == tfs[2].entropy
end

function test_kalman_with_B_input_equivalent_to_bias()
    D, p, T, N = 3, 4, 30, 2
    u_dim = D
    # B = I, u = ones → effective bias of 1 at every step.
    B = Matrix{Float64}(I, D, u_dim)
    Random.seed!(11)
    sm = GaussianStateModel(;
        A=0.7*Matrix{Float64}(I, D, D),
        Q=0.3*Matrix{Float64}(I, D, D),
        x0=zeros(D),
        P0=Matrix{Float64}(I, D, D),
        b=zeros(D),
        B=B,
    )
    om = GaussianObservationModel(;
        C=randn(p, D),
        R=0.4*Matrix{Float64}(I, p, p),
        d=zeros(p),
    )
    lds_B = LinearDynamicalSystem(sm, om; kalman_filter=true)

    u = ones(u_dim, T, N)
    y = _simulate_lds(lds_B, T, N; u=u)

    # Equivalent LDS where bias b=1 replaces B·u=1.
    Random.seed!(11)
    sm_b = GaussianStateModel(;
        A=lds_B.state_model.A, Q=lds_B.state_model.Q,
        x0=lds_B.state_model.x0, P0=lds_B.state_model.P0,
        b=ones(D),
    )
    om_b = GaussianObservationModel(;
        C=lds_B.obs_model.C, R=lds_B.obs_model.R, d=lds_B.obs_model.d,
    )
    lds_b = LinearDynamicalSystem(sm_b, om_b; kalman_filter=true)

    kws_B = StateSpaceDynamics.KalmanWorkspace(lds_B, T, N)
    kws_b = StateSpaceDynamics.KalmanWorkspace(lds_b, T, N)
    tfs_B = StateSpaceDynamics.initialize_FilterSmooth(lds_B, T, N)
    tfs_b = StateSpaceDynamics.initialize_FilterSmooth(lds_b, T, N)

    StateSpaceDynamics.smooth!(lds_B, tfs_B, y, kws_B; u=u)
    StateSpaceDynamics.smooth!(lds_b, tfs_b, y, kws_b)

    @test maximum(abs.(tfs_B[1].x_smooth .- tfs_b[1].x_smooth)) < 1e-8
    @test maximum(abs.(tfs_B[2].x_smooth .- tfs_b[2].x_smooth)) < 1e-8
end

function test_kalman_rejects_poisson_obs()
    D, p = 2, 3
    Random.seed!(5)
    sm = GaussianStateModel(;
        A=0.7*Matrix{Float64}(I, D, D),
        Q=0.2*Matrix{Float64}(I, D, D),
        x0=zeros(D), P0=Matrix{Float64}(I, D, D), b=zeros(D),
    )
    om = PoissonObservationModel(; C=randn(p, D), log_d=zeros(p))
    @test_throws Exception LinearDynamicalSystem(sm, om; kalman_filter=true)
end

function test_kalman_missing_u_errors()
    D, p, T, N = 2, 3, 15, 2
    B = Matrix{Float64}(I, D, D)
    Random.seed!(3)
    sm = GaussianStateModel(;
        A=0.6*Matrix{Float64}(I, D, D),
        Q=0.2*Matrix{Float64}(I, D, D),
        x0=zeros(D), P0=Matrix{Float64}(I, D, D), b=zeros(D), B=B,
    )
    om = GaussianObservationModel(;
        C=randn(p, D), R=0.3*Matrix{Float64}(I, p, p), d=zeros(p),
    )
    lds = LinearDynamicalSystem(sm, om; kalman_filter=true)
    y = randn(p, T, N)
    kws = StateSpaceDynamics.KalmanWorkspace(lds, T, N)
    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, T, N)
    # B is set but u is omitted → should error.
    @test_throws Exception StateSpaceDynamics.smooth!(lds, tfs, y, kws)
end
