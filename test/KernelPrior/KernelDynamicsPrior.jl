using Test, StateSpaceDynamics, LinearAlgebra, Random, Statistics
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
    # beside a kernel prior on the loading, projected through its posterior mean
    @test LinearDynamicalSystem(state(; A_prior=kd_prior(rng, k)), with_h_prior) isa
        LinearDynamicalSystem
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

# ----------------------------------------------------------------------------
# A structured mean (MNMeanBasis)
# ----------------------------------------------------------------------------

# Two networks of `k ÷ 2` units: receiver i's "in" term e_i n_i' (n_i its
# network's other units) and sender j's "out" term n_j e_j'.
function kd_network_basis(k; pooled=true, df=Inf, sd_prior=(0.1, 2.0))
    half = k ÷ 2
    net(i) = i <= half ? (1:half) : ((half + 1):k)
    others(i) = [Float64(j in net(i) && j != i) for j in 1:k]
    e(i) = Float64.(1:k .== i)
    left = hcat([e(i) for i in 1:k]..., [others(j) for j in 1:k]...)
    right = hcat([others(i) for i in 1:k]..., [e(j) for j in 1:k]...)
    return MNMeanBasis(
        left,
        right;
        names=vcat(["in$i" for i in 1:k], ["out$j" for j in 1:k]),
        group=vcat(fill(1, k), fill(2, k)),
        group_names=["in", "out"],
        pooled,
        df,
        sd=0.2,
        sd_prior,
    )
end

@testset "a mean basis: q(A), the GLS and the bound against dense formulas" begin
    rng = MersenneTwister(41)
    k = 4
    kp = KernelMNPrior(
        kd_kernels(rng, k), kd_kernels(rng, k); mean_basis=kd_network_basis(k)
    )
    KDSSD._initialize_prior!(kp.rows, kd_spd(rng, k))
    KDSSD._initialize_prior!(kp.cols, kd_spd(rng, k))
    mb = kp.mean_basis
    kp.m = 0.4
    mb.mean .= 0.1 .* randn(rng, 2k)
    root = randn(rng, 2k, 2k)
    mb.cov = 0.01 .* (root * root' / 2k + 0.1I)
    U, V = KDSSD._mn_covariances(kp)
    Qinv, Sxx, Sxx_c = inv(kd_spd(rng, k)), 20 .* kd_spd(rng, k), 15 .* kd_spd(rng, k)
    Syx_c = randn(rng, k, k) .* 5
    q = KDSSD._mn_posterior!(kp, Qinv, Syx_c, Sxx_c, Sxx)
    terms = [zeros(k, k) for _ in 1:(2k)]
    for (c, p) in enumerate(mb.coef)
        terms[p] .+= mb.left[:, c] * mb.right[:, c]'
    end
    M = kp.m * I(k) + sum(b .* T for (b, T) in zip(mb.mean, terms))
    prior_precision = kron(inv(V), inv(U))
    expected =
        (kron(Sxx_c, Qinv) + prior_precision) \
        (vec(Qinv * Syx_c) + prior_precision * vec(M))
    @test vec(q.mean) ≈ expected rtol = 1e-9
    @test KDSSD._mn_mean(kp) ≈ M
    # the bound over q(A) q(β): β's spread adds tr(prior⁻¹ Cov(vec M))
    covariance = inv(kron(Sxx, Qinv) + prior_precision)
    Tm = hcat(vec.(terms)...)
    prior_cov = kron(V, U)
    centred = vec(q.mean - M)
    dense =
        -k^2 / 2 * log(2π) - logdet(prior_cov) / 2 -
        (
            dot(centred, prior_cov \ centred) +
            tr(prior_cov \ covariance) +
            tr(prior_cov \ (Tm * mb.cov * Tm'))
        ) / 2 + logdet(2π * ℯ * covariance) / 2
    bound =
        KDSSD._mn_bound(kp) - KDSSD._kernel_logprior(kp.rows) -
        KDSSD._kernel_logprior(kp.cols) - KDSSD._mean_bound(mb)
    @test bound ≈ dense rtol = 1e-9
    # one sweep of m and q(β) is the dense generalized least squares
    centre, precision = KDSSD._mean_prior(mb)
    X = hcat(vec(Matrix(1.0I, k, k)), Tm)
    W = inv(prior_cov)
    Λ = X' * W * X + Diagonal(vcat(0.0, precision))
    sol = Λ \ (X' * W * vec(q.mean) + vcat(0.0, precision .* centre))
    KDSSD._mn_update_mean!(kp, U, V; sweeps=1)
    @test kp.m ≈ sol[1] rtol = 1e-8
    # the sweep updates q(λ), μ, τ after q(β); q(β) itself is the GLS's
    @test mb.cov ≈ inv(Λ[2:end, 2:end]) rtol = 1e-8
end

@testset "a mean basis: A's term in the loading and its gradient" begin
    rng = MersenneTwister(42)
    r, k = 6, 3
    H = randn(rng, r, k)
    E(i) = 1e-6 .* reshape(1:(r * k) .== i, r, k)
    for projection in (:noise_weighted, :euclidean)
        kp = KernelMNPrior(
            kd_kernels(rng, r),
            kd_kernels(rng, r);
            projection,
            mean_basis=kd_network_basis(r),
        )
        KDSSD._initialize_prior!(kp.rows, kd_spd(rng, r))
        KDSSD._initialize_prior!(kp.cols, kd_spd(rng, r))
        mb = kp.mean_basis
        mb.mean .= 0.2 .* randn(rng, 2r)
        root = randn(rng, 2r, 2r)
        mb.cov = 0.02 .* (root * root' / 2r + 0.1I)
        kp.m = 0.3
        kp.loading = H
        kp.posterior = KDSSD.LoadingPosterior(
            0.5 .* randn(rng, k, k), kd_spd(rng, k), kd_spd(rng, k), rand(rng, k, k)
        )
        FU = cholesky(Symmetric(kernel_covariance(kp.rows)))
        FV = cholesky(Symmetric(kernel_covariance(kp.cols)))
        term(Hx) = first(KDSSD._projected_dynamics_term(Hx, kp, FU, FV))
        _, ∂g = KDSSD._projected_dynamics_term(H, kp, FU, FV)
        numeric = reshape(
            [(term(H .+ E(i)) - term(H .- E(i))) / 2e-6 for i in 1:(r * k)], r, k
        )
        @test ∂g ≈ numeric rtol = 1e-5 atol = 1e-7
        U, V = KDSSD._mn_covariances(kp)
        @test term(H) ≈
            -k / 2 * (logdet(U) + logdet(V)) - tr(U \ KDSSD._mn_row_moment(kp, V)) / 2 rtol =
            1e-10
    end
end

@testset "a mean basis: a fit finds the network coupling, a t prior the outlier" begin
    rng = MersenneTwister(43)
    k = 6
    A = Matrix(0.5I, k, k)
    for i in 1:k, j in 1:k
        i != j && (i <= 3) == (j <= 3) && (A[i, j] = 0.08)
    end
    A[5, [4, 6]] .= 0.3                # receiver 5 takes far more from its network
    Φ, H, truth = kd_system(rng; k, A, noise=0.02)
    y = rand(rng, truth, fill(100, 40))[2]
    one_each(i) = Kernel(Matrix(Diagonal(1.0 .* (1:k .== i))) + 1e-3I; name="d$i")
    fits = Dict{Float64,Any}()
    for df in (Inf, 3.0)
        prior = KernelMNPrior(
            KernelPrior([one_each(i) for i in 1:k]),
            KernelPrior([one_each(i) for i in 1:k]);
            mean_basis=kd_network_basis(k; df),
        )
        lds = kd_model(rng, Φ, H; learned=false, A_prior=prior)
        trace = fit!(lds, y; max_iter=40, progress=false)
        @test all(isfinite, collect(trace))
        @test kd_nondecreasing(trace)
        fits[df] = (mean_coefficients(prior), KDSSD._mn_mean(prior))
    end
    (gauss, M), (t3, _) = fits[Inf], fits[3.0]
    ins = 1:k
    # in and out are identified only as sums, A_ij ≈ β_in,i + β_out,j: the mean's
    # within-network couplings sit near the shared one, receiver 5's above it
    within = [(i, j) for i in 1:k, j in 1:k if i != j && (i <= 3) == (j <= 3)]
    usual = [M[i, j] for (i, j) in within if i != 5]
    @test abs(median(usual) - 0.08) < 0.04
    @test minimum(M[5, [4, 6]]) > maximum(usual)
    @test all(abs(M[i, j]) < 1e-12 for i in 1:k, j in 1:k if (i <= 3) != (j <= 3))
    @test gauss.mean[5] > maximum(gauss.mean[setdiff(ins, 5)])
    # the t prior shrinks the outlier less, and gives it the smallest λ
    @test t3.mean[5] >= gauss.mean[5] - 1e-3
    @test argmin(t3.lambda[ins]) == 5
    @test all(==(1.0), gauss.lambda)
end

@testset "a mean basis with a learned loading, and A and Q on a masked loading" begin
    rng = MersenneTwister(44)
    # a learned loading: the mean basis in the basis's coordinates, the fit monotone
    Φ, H, truth = kd_system(rng; k=3, r=6, p=12)
    y = rand(rng, truth, fill(50, 6))[2]
    prior = KernelMNPrior(
        kd_kernels(rng, 6), kd_kernels(rng, 6); mean_basis=kd_network_basis(6)
    )
    lds = kd_model(rng, Φ, H; learned=true, A_prior=prior)
    trace = fit!(lds, y; max_iter=15, progress=false)
    @test all(isfinite, collect(trace)) && kd_nondecreasing(trace)
    @test size(KDSSD._mn_mean(prior)) == (3, 3)
    # a masked loading: each latent its own two columns; Q and A see the latents
    k, p = 3, 10
    Φm = randn(rng, p, 2k)
    mask = falses(2k, k)
    for j in 1:k
        mask[(2j - 1):(2j), j] .= true
    end
    H0 = zeros(2k, k)
    H0[mask] .= randn(rng, count(mask))
    sm = GaussianStateModel(;
        A=Matrix(0.5I, k, k),
        Q=Matrix(0.3I, k, k),
        b=zeros(k),
        x0=zeros(k),
        P0=Matrix(1.0I, k, k),
        A_prior=KernelMNPrior(kd_kernels(rng, k), kd_kernels(rng, k);),
        Q_prior=KernelIWPrior(kd_kernels(rng, k)),
    )
    om = ConstrainedGaussianObservationModel(;
        basis=Φm, H=H0, R=Matrix(1.0I, p, p), H_mask=mask
    )
    lds = LinearDynamicalSystem(sm, om)
    truth = LinearDynamicalSystem(
        GaussianStateModel(;
            A=Matrix(0.7I, k, k),
            Q=Matrix(0.1I, k, k),
            b=zeros(k),
            x0=zeros(k),
            P0=Matrix(1.0I, k, k),
        ),
        GaussianObservationModel(; C=Φm * H0, R=Matrix(0.1I, p, p), d=zeros(p)),
    )
    y = rand(rng, truth, fill(50, 6))[2]
    trace = fit!(lds, y; max_iter=15, progress=false)
    @test all(isfinite, collect(trace)) && kd_nondecreasing(trace)
    @test isempty(sm.A_prior.loading) && isempty(sm.Q_prior.loading)
    @test all(iszero, KDSSD._loading_coefficients(lds.obs_model)[.!mask])
    # the wrong size of kernels for a masked loading is refused
    bad = GaussianStateModel(;
        A=Matrix(0.5I, k, k),
        Q=Matrix(0.3I, k, k),
        b=zeros(k),
        x0=zeros(k),
        P0=Matrix(1.0I, k, k),
        Q_prior=KernelIWPrior(kd_kernels(rng, 2k)),
    )
    @test_throws KDSSD.DimensionMismatchError LinearDynamicalSystem(bad, om)
end

# ----------------------------------------------------------------------------
# A kernel prior on B (KernelInputPrior)
# ----------------------------------------------------------------------------

# Two input groups (the first linked to `rows` when given) and a flat input.
function kd_input_prior(rng, k; rows=nothing)
    first_dim = rows === nothing ? k : length(rows)
    return KernelInputPrior(
        [kd_kernels(rng, first_dim), kd_kernels(rng, k)],
        [[1], [2, 3]];
        rows=rows === nothing ? nothing : [rows, nothing],
    )
end

@testset "q(B) and its bound against the dense posterior" begin
    rng = MersenneTwister(51)
    k, m = 4, 4
    for rows in (nothing, [1, 3])
        kb = kd_input_prior(rng, k; rows)
        for p in kb.priors
            KDSSD._initialize_prior!(p, kd_spd(rng, p.dim))
        end
        Qinv, Suu_c = inv(kd_spd(rng, k)), 30 .* kd_spd(rng, m)
        Sru_c = 5 .* randn(rng, k, m)
        KDSSD._input_posterior!(kb, Qinv, Suu_c, Sru_c)
        # dense: the free entries, each group's columns N(0, U_g) on its rows
        free = KDSSD._input_free(kb, k, m)
        Λp = zeros(k * m, k * m)
        for (g, cols) in enumerate(kb.groups), c in cols
            r = KDSSD._input_rows(kb, g, k)
            idx = (c - 1) * k .+ r
            Λp[idx, idx] .+= inv(KDSSD._input_covariance(kb, g))
        end
        Λ = (kron(Suu_c, Qinv) + Λp)[free, free]
        μ = Λ \ vec(Qinv * Sru_c)[free]
        @test vec(kb.mean)[free] ≈ μ rtol = 1e-9
        @test all(iszero, vec(kb.mean)[setdiff(1:(k * m), free)])
        @test kb.cov[free, free] ≈ inv(Λ) rtol = 1e-8
        rows === nothing || @test all(iszero, kb.mean[[2, 4], 1])
        # E log p(B) + H[q(B)], the flat input's entropy alone
        dense = 0.0
        for (g, cols) in enumerate(kb.groups), c in cols
            r = KDSSD._input_rows(kb, g, k)
            U = KDSSD._input_covariance(kb, g)
            idx = (c - 1) * k .+ r
            b, S = kb.mean[r, c], kb.cov[idx, idx]
            dense +=
                -length(r) / 2 * log(2π) - logdet(U) / 2 - (dot(b, U \ b) + tr(U \ S)) / 2
        end
        dense += logdet(2π * ℯ * kb.cov[free, free]) / 2
        bound = KDSSD._input_bound(kb) - sum(KDSSD._kernel_logprior(p) for p in kb.priors)
        @test bound ≈ dense rtol = 1e-9
        # B's spread at a moment S
        S = kd_spd(rng, m)
        draws = [
            begin
                v = zeros(k * m)
                v[free] =
                    cholesky(Symmetric(kb.cov[free, free])).L * randn(rng, length(free))
                reshape(v, k, m)
            end for _ in 1:20000
        ]
        @test KDSSD._input_spread(kb, S) ≈ sum(E * S * E' for E in draws) / length(draws) rtol =
            0.05
    end
end

@testset "B's term in the loading and its gradient" begin
    rng = MersenneTwister(52)
    r, k, m = 6, 3, 3
    kb = KernelInputPrior([kd_kernels(rng, r), kd_kernels(rng, r)], [[1, 2], [3]])
    for p in kb.priors
        KDSSD._initialize_prior!(p, kd_spd(rng, r))
    end
    H = randn(rng, r, k)
    kb.loading = H
    KDSSD._input_posterior!(
        kb, inv(kd_spd(rng, k)), 30 .* kd_spd(rng, m), 5 .* randn(rng, k, m)
    )
    Fs = [cholesky(Symmetric(kernel_covariance(p))) for p in kb.priors]
    g, grad = KDSSD._projected_input_term(H, kb, Fs)
    @test isfinite(g)
    ε = 1e-6
    numeric = similar(H)
    for i in eachindex(H)
        Hp, Hm = copy(H), copy(H)
        Hp[i] += ε
        Hm[i] -= ε
        numeric[i] =
            (
                first(KDSSD._projected_input_term(Hp, kb, Fs)) -
                first(KDSSD._projected_input_term(Hm, kb, Fs))
            ) / (2ε)
    end
    @test grad ≈ numeric rtol = 1e-5
end

function kd_input_system(rng; k=3, r=k, p=10, m=3, B=nothing, noise=0.05)
    Φ = Matrix(qr(randn(rng, p, r)).Q)[:, 1:r]
    H = r == k ? Matrix(1.0I, k, k) : randn(rng, r, k)
    B === nothing && (B = randn(rng, k, m))
    sm = GaussianStateModel(;
        A=0.7 .* Matrix(qr(randn(rng, k, k)).Q),
        Q=Matrix(0.1I, k, k),
        b=zeros(k),
        x0=zeros(k),
        P0=Matrix(1.0I, k, k),
        B,
    )
    om = GaussianObservationModel(; C=Φ * H, R=Matrix(noise * I, p, p), d=zeros(p))
    truth = LinearDynamicalSystem(sm, om)
    u = [randn(rng, m, 60) for _ in 1:8]
    x, y = rand(rng, truth, fill(60, 8); ux=u)
    return Φ, H, truth, y, u
end

@testset "KernelInputPrior: monotone fits, alone and with an A prior" begin
    rng = MersenneTwister(53)
    k, m = 3, 3
    for (learned, r) in ((false, 3), (true, 5))
        Φ, H, truth, y, u = kd_input_system(rng; k, r, m)
        for with_a in (false, true)
            kb = KernelInputPrior([kd_kernels(rng, r), kd_kernels(rng, r)], [[1], [2, 3]])
            sm = GaussianStateModel(;
                A=Matrix(0.5I, k, k),
                Q=Matrix(0.3I, k, k),
                b=zeros(k),
                x0=zeros(k),
                P0=Matrix(1.0I, k, k),
                B=zeros(k, m),
                A_prior=with_a ? kd_prior(rng, r) : nothing,
                B_prior=kb,
            )
            om = if learned
                ConstrainedGaussianObservationModel(;
                    basis=Φ,
                    H=H .+ 0.3 .* randn(rng, size(H)),
                    R=Matrix(1.0I, size(Φ, 1), size(Φ, 1)),
                )
            else
                GaussianObservationModel(;
                    C=copy(Φ),
                    R=Matrix(1.0I, size(Φ, 1), size(Φ, 1)),
                    d=zeros(size(Φ, 1)),
                    C_fixed=true,
                )
            end
            lds = LinearDynamicalSystem(sm, om)
            trace = fit!(lds, y; ux=u, max_iter=15, progress=false)
            @test all(isfinite, collect(trace))
            @test kd_nondecreasing(trace)
            post = input_posterior(lds.state_model)
            @test post !== nothing && lds.state_model.B ≈ post.mean
            @test isempty(kb.loading) != learned
            learned && @test kb.loading ≈ Φ \ lds.obs_model.C
            @test length(kernel_weights(kb)) == 2
            # the fit sees the inputs' footprint (up to the latents' gauge)
            if !learned
                @test cor(vec(lds.state_model.B), vec(truth.state_model.B)) > 0.9
            end
        end
    end
end

@testset "KernelInputPrior: rows link an input to some latents" begin
    rng = MersenneTwister(54)
    k, m = 4, 2
    B = zeros(k, m)
    B[1:2, 1] .= [1.0, -0.8]
    B[:, 2] = randn(rng, k)
    Φ, H, truth, y, u = kd_input_system(rng; k, m, B)
    kb = KernelInputPrior(
        [kd_kernels(rng, 2), kd_kernels(rng, k)], [[1], [2]]; rows=[[1, 2], nothing]
    )
    sm = GaussianStateModel(;
        A=Matrix(0.5I, k, k),
        Q=Matrix(0.3I, k, k),
        b=zeros(k),
        x0=zeros(k),
        P0=Matrix(1.0I, k, k),
        B=zeros(k, m),
        B_prior=kb,
    )
    om = GaussianObservationModel(;
        C=copy(Φ), R=Matrix(1.0I, 10, 10), d=zeros(10), C_fixed=true
    )
    lds = LinearDynamicalSystem(sm, om)
    trace = fit!(lds, y; ux=u, max_iter=20, progress=false)
    @test kd_nondecreasing(trace)
    @test all(iszero, lds.state_model.B[3:4, 1])
    @test lds.state_model.B[1:2, 1] ≈ B[1:2, 1] atol = 0.15
    # rows need the latents' own coordinates; groups index inputs that exist
    learned = ConstrainedGaussianObservationModel(;
        basis=Φ, H=randn(rng, k, k), R=Matrix(1.0I, 10, 10)
    )
    @test_throws ArgumentError LinearDynamicalSystem(
        GaussianStateModel(;
            A=Matrix(0.5I, k, k),
            Q=Matrix(0.3I, k, k),
            b=zeros(k),
            x0=zeros(k),
            P0=Matrix(1.0I, k, k),
            B=zeros(k, m),
            B_prior=deepcopy(kb),
        ),
        learned,
    )
    @test_throws ArgumentError LinearDynamicalSystem(
        GaussianStateModel(;
            A=Matrix(0.5I, k, k),
            Q=Matrix(0.3I, k, k),
            b=zeros(k),
            x0=zeros(k),
            P0=Matrix(1.0I, k, k),
            B=zeros(k, 1),
            B_prior=deepcopy(kb),
        ),
        om,
    )
    @test_throws ArgumentError KernelInputPrior([kd_kernels(rng, k)], [[1], [2]])
    @test_throws ArgumentError KernelInputPrior(
        [kd_kernels(rng, k), kd_kernels(rng, k)], [[1], [1]]
    )
    @test_throws DimensionMismatch KernelInputPrior(
        [kd_kernels(rng, k)], [[1]]; rows=[[1, 2]]
    )
end

@testset "priors on Q, A and B beside a kernel prior on the loading" begin
    rng = MersenneTwister(55)
    k, r, m, p = 3, 5, 2, 12
    Φ, H, truth, y, u = kd_input_system(rng; k, r, m, p)
    sm = GaussianStateModel(;
        A=Matrix(0.5I, k, k),
        Q=Matrix(0.3I, k, k),
        b=zeros(k),
        x0=zeros(k),
        P0=Matrix(1.0I, k, k),
        B=zeros(k, m),
        A_prior=kd_prior(rng, r),
        Q_prior=KernelIWPrior(kd_kernels(rng, r)),
        B_prior=KernelInputPrior([kd_kernels(rng, r)], [[1, 2]]),
    )
    om = ConstrainedKernelGaussianObservationModel(;
        basis=Φ,
        prior=kd_kernels(rng, r),
        H=H .+ 0.3 .* randn(rng, size(H)),
        R=Matrix(1.0I, p, p),
        gauge=:moment,
    )
    lds = LinearDynamicalSystem(sm, om)
    trace = fit!(lds, y; ux=u, max_iter=15, progress=false)
    @test all(isfinite, collect(trace))
    # the plug-in bound may dip as H moves, but the fit gains
    @test last(trace) > first(trace)
    Hm = KDSSD._loading_coefficients(lds.obs_model)
    @test lds.state_model.A_prior.loading ≈ Hm
    @test lds.state_model.Q_prior.loading ≈ Hm
    @test lds.state_model.B_prior.loading ≈ Hm
    # a gauge step moves the projected priors with the latents: their terms hold
    before = (
        KDSSD._mn_bound(lds.state_model.A_prior),
        KDSSD._input_bound(lds.state_model.B_prior),
    )
    Tm = Matrix(1.0I, k, k) + 0.3 .* randn(rng, k, k)
    KDSSD._apply_latent_gauge!(lds, Tm)
    after = (
        KDSSD._mn_bound(lds.state_model.A_prior),
        KDSSD._input_bound(lds.state_model.B_prior),
    )
    @test collect(after) ≈ collect(before) rtol = 1e-8
    @test lds.state_model.A_prior.posterior.mean ≈ lds.state_model.A
    @test lds.state_model.B_prior.mean ≈ lds.state_model.B
end
