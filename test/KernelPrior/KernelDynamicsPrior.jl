using Test, StateSpaceDynamics, LinearAlgebra, Random
const KDSSD = StateSpaceDynamics

kd_spd(rng, d) = (X = randn(rng, d, d + 3); X * X' / (d + 3) + 0.2I)
kd_nondecreasing(trace) = all(diff(collect(trace)) .>= -1e-8 * abs(last(trace)))

function kd_kernels(rng, d; kwargs...)
    smooth = [exp(-(i - j)^2 / 4) for i in 1:d, j in 1:d]
    return KernelPrior(
        [Kernel(Matrix(1.0I, d, d); name="noise"), Kernel(smooth; name="smooth")]; kwargs...
    )
end

function kd_prior(rng, d; kwargs...)
    return KernelMNPrior(kd_kernels(rng, d), kd_kernels(rng, d); kwargs...)
end

# A prior with its weights set, at kernel dimension `d`.
function kd_ready(rng, d)
    kp = kd_prior(rng, d)
    KDSSD._initialize_prior!(kp.rows, kd_spd(rng, d))
    KDSSD._initialize_prior!(kp.cols, kd_spd(rng, d))
    kp.m = 0.4
    return kp
end

@testset "q(A) and its bound against the dense posterior" begin
    rng = MersenneTwister(31)
    k = 3
    kp = kd_ready(rng, k)
    U, V = KDSSD._mn_covariances(kp)
    Qinv, Sxx, Sxx_c = inv(kd_spd(rng, k)), 20 .* kd_spd(rng, k), 15 .* kd_spd(rng, k)
    Syx_c = randn(rng, k, k) .* 5
    q = KDSSD._mn_posterior!(kp, Qinv, Syx_c, Sxx_c, Sxx)
    prior_precision = kron(inv(V), inv(U))
    mean_precision = kron(Sxx_c, Qinv) + prior_precision
    expected = mean_precision \ (vec(Qinv * Syx_c) + prior_precision * vec(kp.m * I(k)))
    @test vec(q.mean) ≈ expected rtol = 1e-9
    covariance = inv(kron(Sxx, Qinv) + prior_precision)
    factor = kron(q.right, q.left)
    @test factor * Diagonal(vec(q.var)) * factor' ≈ covariance rtol = 1e-8
    # E log p(A) + H[q(A)] by the dense formulas
    centred = vec(q.mean - kp.m * I)
    prior_cov = kron(V, U)
    dense =
        -k^2 / 2 * log(2π) - logdet(prior_cov) / 2 -
        (dot(centred, prior_cov \ centred) + tr(prior_cov \ covariance)) / 2 +
        logdet(2π * ℯ * covariance) / 2
    bound =
        KDSSD._mn_bound(kp) - KDSSD._kernel_logprior(kp.rows) -
        KDSSD._kernel_logprior(kp.cols)
    @test bound ≈ dense rtol = 1e-9
    # the smoother's term and Q's extra scatter
    Δ = KDSSD._loading_excess(q, Qinv)
    draws = [
        q.mean + q.left * (sqrt.(q.var) .* randn(rng, k, k)) * q.right' for _ in 1:40000
    ]
    @test sum(A' * Qinv * A for A in draws) / length(draws) - q.mean' * Qinv * q.mean ≈ Δ rtol =
        0.05
    S = kd_spd(rng, k)
    @test KDSSD._mn_spread(kp, S) ≈
        sum((A - q.mean) * S * (A - q.mean)' for A in draws) / length(draws) rtol = 0.05
end

@testset "the smoother with an uncertain transition" begin
    rng = MersenneTwister(32)
    k, p = 3, 5
    Φ = Matrix(qr(randn(rng, p, k)).Q)[:, 1:k]
    A, Q, P0 = 0.7 .* Matrix(qr(randn(rng, k, k)).Q), kd_spd(rng, k), kd_spd(rng, k)
    b, x0 = 0.1 .* randn(rng, k), randn(rng, k)
    C, R, d = Φ, kd_spd(rng, p), 0.1 .* randn(rng, p)
    kp = kd_ready(rng, k)
    kp.posterior = KDSSD.LoadingPosterior(
        copy(A), Matrix(1.0I, k, k), Matrix(1.0I, k, k), zeros(k, k)
    )
    root = randn(rng, k, k)
    kp.excess = root * root'
    sm = GaussianStateModel(; A, Q, b, x0, P0, A_prior=kp)
    lds = LinearDynamicalSystem(sm, GaussianObservationModel(; C, R, d, C_fixed=true))
    # the dense posterior of one trial: Δ on every state with a successor
    function dense(Y)
        T = size(Y, 2)
        J, h = zeros(k * T, k * T), zeros(k * T)
        blk(t) = ((t - 1) * k + 1):(t * k)
        Qi, Ri, P0i = inv(Q), inv(R), inv(P0)
        for t in 1:T
            J[blk(t), blk(t)] .+= C' * Ri * C
            h[blk(t)] .+= C' * Ri * (Y[:, t] - d)
            if t == 1
                J[blk(t), blk(t)] .+= P0i
                h[blk(t)] .+= P0i * x0
            else
                J[blk(t), blk(t)] .+= Qi
                h[blk(t)] .+= Qi * b
                J[blk(t), blk(t - 1)] .-= Qi * A
                J[blk(t - 1), blk(t)] .-= A' * Qi
            end
            if t < T
                J[blk(t), blk(t)] .+= A' * Qi * A + kp.excess
                h[blk(t)] .-= A' * Qi * b
            end
        end
        Σ = inv(Symmetric(J))
        return reshape(Σ * h, k, T), [Σ[blk(t), blk(t)] for t in 1:T]
    end
    for y in (
        [randn(rng, p, 7) for _ in 1:3],
        [randn(rng, p, n) for n in (6, 9)],
        [randn(rng, p, 8)],
    )
        x, P = smooth(lds, y)
        for (i, Y) in enumerate(y)
            x_ref, P_ref = dense(Y)
            @test x[i] ≈ x_ref rtol = 1e-8
            @test all(P[i][:, :, t] ≈ P_ref[t] for t in axes(Y, 2))
        end
    end
end

@testset "the sandwich projection and A's term in the loading" begin
    rng = MersenneTwister(33)
    r, k = 5, 2
    H = randn(rng, r, k)
    Z = kd_spd(rng, r)
    F = cholesky(Symmetric(Z))
    sandwich = KDSSD._Projection(:sandwich, H)
    K̃, _ = KDSSD._project(F, sandwich)
    @test K̃ ≈ H' * Z * H
    T = randn(rng, k, k) + 2I
    @test first(KDSSD._project(F, KDSSD._Projection(:sandwich, H / T))) ≈ T' \ K̃ / T
    G̃ = kd_spd(rng, k)
    value(Hx) = dot(G̃, first(KDSSD._project(F, KDSSD._Projection(:sandwich, Hx))))
    h = 1e-6
    E(i) = h .* reshape(1:(r * k) .== i, r, k)
    numeric = reshape([(value(H .+ E(i)) - value(H .- E(i))) / 2h for i in 1:(r * k)], r, k)
    @test KDSSD._projection_gradient(sandwich, F, K̃, G̃) ≈ numeric rtol = 1e-6
    # A's term in H, for both projections of the rows
    for projection in (:noise_weighted, :euclidean)
        kp = KernelMNPrior(kd_kernels(rng, r), kd_kernels(rng, r); projection)
        KDSSD._initialize_prior!(kp.rows, kd_spd(rng, r))
        KDSSD._initialize_prior!(kp.cols, kd_spd(rng, r))
        kp.loading = H
        kp.posterior = KDSSD.LoadingPosterior(
            0.5 .* randn(rng, k, k), kd_spd(rng, k), kd_spd(rng, k), rand(rng, k, k)
        )
        FU = cholesky(Symmetric(kernel_covariance(kp.rows)))
        FV = cholesky(Symmetric(kernel_covariance(kp.cols)))
        term(Hx) = first(KDSSD._projected_dynamics_term(Hx, kp, FU, FV))
        _, ∂g = KDSSD._projected_dynamics_term(H, kp, FU, FV)
        numeric = reshape(
            [(term(H .+ E(i)) - term(H .- E(i))) / 2h for i in 1:(r * k)], r, k
        )
        @test ∂g ≈ numeric rtol = 1e-5 atol = 1e-7
        # the term is the bound's prior part at the projected U, V
        U, V = KDSSD._mn_covariances(kp)
        @test term(H) ≈
            -k / 2 * (logdet(U) + logdet(V)) - tr(U \ KDSSD._mn_row_moment(kp, V)) / 2 rtol =
            1e-10
    end
end

# A fixed basis (C = Φ held) or a learned loading (C = Φ H), with the priors given.
function kd_system(rng; k=3, r=k, p=10, noise=0.1, A=nothing)
    Φ = Matrix(qr(randn(rng, p, r)).Q)[:, 1:r]
    H = r == k ? Matrix(1.0I, k, k) : randn(rng, r, k)
    A === nothing && (A = 0.8 .* Matrix(qr(randn(rng, k, k)).Q))
    sm = GaussianStateModel(;
        A, Q=Matrix(0.1I, k, k), b=zeros(k), x0=zeros(k), P0=Matrix(1.0I, k, k)
    )
    om = GaussianObservationModel(; C=Φ * H, R=Matrix(noise * I, p, p), d=zeros(p))
    return Φ, H, LinearDynamicalSystem(sm, om)
end

function kd_model(rng, Φ, H; learned, A_prior=nothing, Q_prior=nothing, R_prior=nothing)
    p, r = size(Φ)
    k = size(H, 2)
    sm = GaussianStateModel(;
        A=Matrix(0.5I, k, k),
        Q=Matrix(0.3I, k, k),
        b=zeros(k),
        x0=zeros(k),
        P0=Matrix(1.0I, k, k),
        A_prior,
        Q_prior,
    )
    om = if learned
        ConstrainedGaussianObservationModel(;
            basis=Φ, H=H .+ 0.3 .* randn(rng, size(H)), R=Matrix(1.0I, p, p), R_prior
        )
    else
        GaussianObservationModel(;
            C=copy(Φ), R=Matrix(1.0I, p, p), d=zeros(p), R_prior, C_fixed=true
        )
    end
    return LinearDynamicalSystem(sm, om)
end

@testset "KernelMNPrior: monotone fits with Q and R priors" begin
    rng = MersenneTwister(34)
    for (learned, r, k) in ((false, 4, 4), (true, 6, 3))
        p = 12
        Φ, H, truth = kd_system(rng; k, r, p)
        y = rand(rng, truth, fill(50, 6))[2]
        y_test = rand(rng, truth, fill(50, 2))[2]
        for (with_q, with_r) in ((false, false), (false, true), (true, false), (true, true))
            lds = kd_model(
                rng,
                Φ,
                H;
                learned,
                A_prior=kd_prior(rng, r),
                Q_prior=with_q ? KernelIWPrior(kd_kernels(rng, r)) : nothing,
                R_prior=with_r ? KernelIWPrior(kd_kernels(rng, p)) : nothing,
            )
            trace = fit!(lds, y; max_iter=15, progress=false, y_test)
            @test all(isfinite, collect(trace))
            @test kd_nondecreasing(trace)
            kp = lds.state_model.A_prior
            q = dynamics_posterior(lds.state_model)
            @test q !== nothing && lds.state_model.A ≈ q.mean
            @test minimum(eigvals(Symmetric(kp.excess))) > -1e-10
            @test isempty(kp.loading) != learned
            learned && @test kp.loading ≈ Φ \ lds.obs_model.C
            @test isfinite(elbo(lds, y))
            @test all(isfinite, (w.weight for w in kernel_weights(kp).cols))
        end
    end
    # fitted trees: the trace may dip when they move, but the fit runs
    Φ, H, truth = kd_system(rng; k=4, r=4)
    y = rand(rng, truth, fill(40, 4))[2]
    prior = KernelMNPrior(kd_kernels(rng, 4; levels=:hierarchical), kd_kernels(rng, 4))
    lds = kd_model(rng, Φ, H; learned=false, A_prior=prior)
    @test all(isfinite, collect(fit!(lds, y; max_iter=8, progress=false)))
end

@testset "KernelMNPrior: a strong sender stands out" begin
    rng = MersenneTwister(35)
    k = 4
    A = 0.5 .* Matrix(1.0I, k, k)
    A[[1, 3, 4], 2] .= 0.35            # latent 2 drives the others
    Φ, H, truth = kd_system(rng; k, A, noise=0.02)
    y = rand(rng, truth, fill(100, 30))[2]
    one_each(i) = Kernel(Matrix(Diagonal(1.0 .* (1:k .== i))) + 1e-3I; name="d$i")
    region_prior() = KernelPrior([one_each(i) for i in 1:k])
    prior = KernelMNPrior(region_prior(), region_prior())
    lds = kd_model(rng, Φ, H; learned=false, A_prior=prior)
    fit!(lds, y; max_iter=40, progress=false)
    senders = Dict(w.name => w.value for w in kernel_weights(prior).cols)
    @test argmax(senders) == "d2"
    @test senders["d2"] > 3 * maximum(v for (n, v) in senders if n != "d2")
end

@testset "KernelMNPrior: refusals" begin
    rng = MersenneTwister(36)
    k, p = 3, 8
    Φ = Matrix(qr(randn(rng, p, k)).Q)[:, 1:k]
    state(; kwargs...) = GaussianStateModel(;
        A=Matrix(0.5I, k, k),
        Q=Matrix(0.3I, k, k),
        b=zeros(k),
        x0=zeros(k),
        P0=Matrix(1.0I, k, k),
        kwargs...,
    )
    fixed = GaussianObservationModel(; C=Φ, R=Matrix(1.0I, p, p), d=zeros(p), C_fixed=true)
    @test_throws ArgumentError LinearDynamicalSystem(
        state(;
            A_prior=kd_prior(rng, k),
            AB_prior=MNPrior(; M₀=zeros(k, k + 1), Λ=Matrix(1.0I, k + 1, k + 1)),
        ),
        fixed,
    )
    @test_throws KDSSD.DimensionMismatchError LinearDynamicalSystem(
        state(; A_prior=kd_prior(rng, k + 1)), fixed
    )
    free_C = GaussianObservationModel(; C=Φ, R=Matrix(1.0I, p, p), d=zeros(p))
    @test_throws ArgumentError LinearDynamicalSystem(
        state(; A_prior=kd_prior(rng, k)), free_C
    )
    with_h_prior = ConstrainedKernelGaussianObservationModel(;
        basis=Φ, prior=kd_kernels(rng, k), H=randn(rng, k, k), R=Matrix(1.0I, p, p)
    )
    @test_throws ArgumentError LinearDynamicalSystem(
        state(; A_prior=kd_prior(rng, k)), with_h_prior
    )
    lds = LinearDynamicalSystem(state(; A_prior=kd_prior(rng, k)), fixed)
    y = [randn(rng, p, 20) for _ in 1:2]
    @test_throws ArgumentError fit!(lds, y; max_iter=2, progress=false, gauge_whiten=true)
    slds = SLDS([0.9 0.1; 0.1 0.9], [0.5, 0.5], [lds, deepcopy(lds)])
    @test_throws ArgumentError fit!(slds, y; max_iter=2, progress=false)
    @test_throws DimensionMismatch KernelMNPrior(kd_kernels(rng, k), kd_kernels(rng, k + 1))
    @test_throws ArgumentError KernelMNPrior(
        kd_kernels(rng, k), kd_kernels(rng, k); mean=:other
    )
end
