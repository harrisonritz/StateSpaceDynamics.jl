using Test, StateSpaceDynamics, LinearAlgebra, Random, SparseArrays
const KCSSD = StateSpaceDynamics

kc_spd(rng, d) = (X = randn(rng, d, d + 3); X * X' / (d + 3) + 0.2I)

# Two sum kernels over `d` dimensions that together span them.
function kc_prior(rng, d; kwargs...)
    smooth = [exp(-(i - j)^2 / 4) for i in 1:d, j in 1:d]
    return KernelPrior(
        [Kernel(Matrix(1.0I, d, d); name="noise"), Kernel(smooth; name="smooth")]; kwargs...
    )
end

# A small scale-correlation structure beside a noise kernel.
function kc_structured_prior(rng, d)
    n = 9
    G = randn(rng, d, n)
    indicator(cols) = sparse(reshape(Float64.(in(cols).(1:n)), :, 1))
    family = CorrelationFamily(
        [indicator(1:4), indicator(5:9)]; type=:parcel, names=["a", "b"]
    )
    structure = ScaleCorrelation(
        G,
        0.5 .+ rand(rng, n),
        [1, 1, 1, 1, 2, 2, 2, 2, 2];
        names=["a", "b"],
        families=[family],
    )
    return KernelPrior([Kernel(Matrix(0.5I, d, d); name="I", type=:nuisance)]; structure)
end

nondecreasing(trace) = all(diff(collect(trace)) .>= -1e-8 * abs(last(trace)))

@testset "inverse-Wishart target: gradient and information" begin
    rng = MersenneTwister(3)
    d, ν = 5, 9.0
    for prior in (kc_prior(rng, d), kc_structured_prior(rng, d))
        nw = length(prior.labels)
        KCSSD._initialize_prior!(prior, kc_spd(rng, d))
        w = prior.weights .+ 0.3 .* randn(rng, nw)
        μ, Λ = randn(rng, nw), Matrix(0.5I, nw, nw)
        target = KCSSD._IWMean(Matrix(inv(kc_spd(rng, d))), ν)
        f = x -> KCSSD._kernel_objective!(nothing, prior, x, target, μ, Λ)
        grad = zeros(nw)
        @test KCSSD._kernel_objective!(grad, prior, w, target, μ, Λ) ≈ f(w)
        h = 1e-6
        numeric = [
            (f(w .+ h .* (1:nw .== i)) - f(w .- h .* (1:nw .== i))) / 2h for i in 1:nw
        ]
        @test grad ≈ numeric rtol = 1e-5 atol = 1e-6
        # the expected curvature (P = E[Σ⁻¹] under the prior, ν/(ν − d − 1) K⁻¹) is
        # that of ν draws
        K = kernel_covariance(prior, w)
        at_prior = KCSSD._IWMean(ν / (ν - d - 1) .* Matrix(inv(K)), ν)
        g = x -> (G = zeros(nw); KCSSD._kernel_objective!(G, prior, x, at_prior, μ, Λ); G)
        hessian = reduce(
            hcat,
            [(g(w .+ h .* (1:nw .== i)) - g(w .- h .* (1:nw .== i))) / 2h for i in 1:nw],
        )
        @test (hessian + hessian') / 2 - Λ ≈ ν .* KCSSD._kernel_information(prior, w) rtol =
            1e-4 atol = 1e-6
    end
end

@testset "KernelIWPrior: the bound at the optimal posterior" begin
    rng = MersenneTwister(4)
    d, ν, N = 4, 8.0, 30.0
    kp = KernelIWPrior(kc_prior(rng, d); dof=ν)
    KCSSD._initialize_prior!(kp.prior, kc_spd(rng, d))
    S = N .* kc_spd(rng, d)
    Ψ = (ν - d - 1) .* kernel_covariance(kp.prior)
    Σ̃ = (Ψ + S) ./ (ν + N)
    kp.count, kp.posterior_dof = N, ν + N
    plug_in = -N / 2 * logdet(Σ̃) - tr(Σ̃ \ S) / 2 - N * d / 2 * log(2π)
    bound = plug_in + KCSSD._kernel_iw_bound(kp, Σ̃) - KCSSD._kernel_logprior(kp.prior)
    lmg(a) = KCSSD._logmvgamma(d, a)
    marginal =
        -N * d / 2 * log(π) + lmg((ν + N) / 2) - lmg(ν / 2) + ν / 2 * logdet(Ψ) -
        (ν + N) / 2 * logdet(Ψ + S)
    @test bound ≈ marginal rtol = 1e-10
    # any other posterior is a lower bound
    other = 1.3 .* Σ̃
    bound_other =
        -N / 2 * logdet(other) - tr(other \ S) / 2 - N * d / 2 * log(2π) +
        KCSSD._kernel_iw_bound(kp, other) - KCSSD._kernel_logprior(kp.prior)
    @test bound_other < marginal
    # dof against Σ's dimension is checked with the model (refusals below)
    @test_throws ArgumentError KernelIWPrior(kc_prior(rng, d); dof=1)
    @test_throws ArgumentError KernelIWPrior(kc_prior(rng, d); projection=:other)
    @test_throws ArgumentError KCSSD._kernel_iw_bound(
        KernelIWPrior(kc_prior(rng, d)), Matrix(1.0I, d, d)
    )
end

# A fixed-basis LDS: the latents in `r` basis coordinates, `C = Φ` held.
function kc_system(rng; r=4, p=10, Q=Matrix(0.1I, r, r), d=zeros(p), noise=0.3)
    Φ = Matrix(qr(randn(rng, p, r)).Q)[:, 1:r]
    A = 0.85 .* Matrix(qr(randn(rng, r, r)).Q)
    sm = GaussianStateModel(; A, Q, b=zeros(r), x0=zeros(r), P0=Matrix(1.0I, r, r))
    om = GaussianObservationModel(; C=Φ, R=Matrix(noise * I, p, p), d)
    return Φ, LinearDynamicalSystem(sm, om)
end

function kc_model(Φ, r, p; Q_prior=nothing, R_prior=nothing, C_fixed=true)
    sm = GaussianStateModel(;
        A=Matrix(0.5I, r, r),
        Q=Matrix(0.3I, r, r),
        b=zeros(r),
        x0=zeros(r),
        P0=Matrix(1.0I, r, r),
        Q_prior,
    )
    om = GaussianObservationModel(;
        C=copy(Φ), R=Matrix(1.0I, p, p), d=zeros(p), R_prior, C_fixed
    )
    return LinearDynamicalSystem(sm, om)
end

@testset "C_fixed holds the loading and fits the offset" begin
    rng = MersenneTwister(5)
    r, p = 3, 8
    d_true = randn(rng, p)
    Φ, truth = kc_system(rng; r, p, d=d_true)
    y = rand(rng, truth, fill(80, 6))[2]
    lds = kc_model(Φ, r, p)
    trace = fit!(lds, y; max_iter=30, progress=false)
    @test lds.obs_model.C == Φ
    # d's part in the basis's span trades with the latents' mean (b, x0); the rest
    # is identified
    outside = I - Φ * Φ'
    @test outside * lds.obs_model.d ≈ outside * d_true atol = 0.1
    @test nondecreasing(trace)
end

@testset "KernelIWPrior on Q and R: monotone fits" begin
    rng = MersenneTwister(6)
    r, p = 4, 10
    Φ, truth = kc_system(rng; r, p)
    y = rand(rng, truth, fill(60, 6))[2]
    y_test = rand(rng, truth, fill(60, 2))[2]
    for (q, rr) in ((true, false), (false, true), (true, true))
        lds = kc_model(
            Φ,
            r,
            p;
            Q_prior=q ? KernelIWPrior(kc_prior(rng, r); dof=r + 4) : nothing,
            R_prior=rr ? KernelIWPrior(kc_prior(rng, p); dof=p + 4) : nothing,
        )
        trace = fit!(lds, y; max_iter=25, progress=false, y_test)
        @test all(isfinite, collect(trace))
        @test nondecreasing(trace)
        for prior in (lds.state_model.Q_prior, lds.obs_model.R_prior)
            prior === nothing && continue
            @test all(isfinite, (row.weight for row in kernel_weights(prior)))
            @test prior.posterior_dof > prior.dof
        end
        @test isfinite(elbo(lds, y))
    end
    # fitted trees: the trace may dip when they move, but the fit runs
    lds = kc_model(
        Φ, r, p; R_prior=KernelIWPrior(kc_prior(rng, p; levels=:hierarchical); dof=p + 4)
    )
    @test all(isfinite, collect(fit!(lds, y; max_iter=10, progress=false)))
end

@testset "KernelIWPrior on Q: the weights follow the truth" begin
    rng = MersenneTwister(7)
    r, p = 5, 12
    smooth = [exp(-(i - j)^2 / 4) for i in 1:r, j in 1:r]
    Q_true = 0.02 .* Matrix(1.0I, r, r) .+ 0.2 .* smooth
    Φ, truth = kc_system(rng; r, p, Q=Q_true, noise=0.05)
    y = rand(rng, truth, fill(100, 20))[2]
    lds = kc_model(Φ, r, p; Q_prior=KernelIWPrior(kc_prior(rng, r); dof=4r))
    fit!(lds, y; max_iter=60, progress=false)
    values = Dict(row.name => row.value for row in kernel_weights(lds.state_model.Q_prior))
    @test values["smooth"] > 5 * values["noise"]
    @test values["smooth"] ≈ 0.2 rtol = 0.5
end

@testset "KernelIWPrior on R with a kernel-prior emission" begin
    rng = MersenneTwister(8)
    r, p, k = 4, 10, 2
    Φ = Matrix(qr(randn(rng, p, r)).Q)[:, 1:r]
    sm() = GaussianStateModel(;
        A=[0.9 -0.2; 0.2 0.9],
        Q=Matrix(0.1I, k, k),
        b=zeros(k),
        x0=zeros(k),
        P0=Matrix(1.0I, k, k),
    )
    truth = LinearDynamicalSystem(
        sm(),
        GaussianObservationModel(;
            C=Φ * randn(rng, r, k), R=Matrix(0.3I, p, p), d=zeros(p)
        ),
    )
    y = rand(rng, truth, fill(50, 6))[2]
    om = ConstrainedKernelGaussianObservationModel(;
        basis=Φ,
        prior=kc_prior(rng, r),
        H=randn(rng, r, k),
        R=Matrix(1.0I, p, p),
        R_prior=KernelIWPrior(kc_prior(rng, p); dof=p + 4),
    )
    lds = LinearDynamicalSystem(sm(), om)
    trace = fit!(lds, y; max_iter=20, progress=false)
    @test nondecreasing(trace)
    @test lds.obs_model.R_prior.posterior_dof > lds.obs_model.R_prior.dof
end

@testset "KernelIWPrior and C_fixed: refusals" begin
    rng = MersenneTwister(9)
    r, p = 3, 6
    Φ = Matrix(qr(randn(rng, p, r)).Q)[:, 1:r]
    @test_throws KCSSD.DimensionMismatchError kc_model(
        Φ, r, p; Q_prior=KernelIWPrior(kc_prior(rng, r + 1))
    )
    @test_throws KCSSD.DimensionMismatchError kc_model(
        Φ, r, p; R_prior=KernelIWPrior(kc_prior(rng, p - 1))
    )
    lds = kc_model(Φ, r, p; Q_prior=KernelIWPrior(kc_prior(rng, r)))
    y = [randn(rng, p, 20) for _ in 1:2]
    @test_throws ArgumentError fit!(lds, y; max_iter=2, progress=false, gauge_whiten=true)
    diagonal = GaussianObservationModel(;
        C=Φ,
        R=Matrix(1.0I, p, p),
        d=zeros(p),
        R_prior=KernelIWPrior(kc_prior(rng, p)),
        R_diagonal=true,
    )
    @test_throws ArgumentError LinearDynamicalSystem(
        kc_model(Φ, r, p).state_model, diagonal
    )
    with_cd = GaussianObservationModel(;
        C=Φ,
        R=Matrix(1.0I, p, p),
        d=zeros(p),
        C_fixed=true,
        CD_prior=MNPrior(; M₀=zeros(p, r + 1), Λ=Matrix(1.0I, r + 1, r + 1)),
    )
    @test_throws ArgumentError LinearDynamicalSystem(kc_model(Φ, r, p).state_model, with_cd)
    with_h_prior = ConstrainedKernelGaussianObservationModel(;
        basis=Φ, prior=kc_prior(rng, r), H=randn(rng, r, r), R=Matrix(1.0I, p, p)
    )
    @test_throws ArgumentError LinearDynamicalSystem(
        kc_model(Φ, r, p; Q_prior=KernelIWPrior(kc_prior(rng, r))).state_model, with_h_prior
    )
    free_C = GaussianObservationModel(; C=Φ, R=Matrix(1.0I, p, p), d=zeros(p))
    @test_throws ArgumentError LinearDynamicalSystem(
        kc_model(Φ, r, p; Q_prior=KernelIWPrior(kc_prior(rng, r))).state_model, free_C
    )
    @test_throws ArgumentError kc_model(
        Φ, r, p; Q_prior=KernelIWPrior(kc_prior(rng, r); dof=r + 1)
    )

    member = kc_model(Φ, r, p; R_prior=KernelIWPrior(kc_prior(rng, p)))
    slds = SLDS([0.9 0.1; 0.1 0.9], [0.5, 0.5], [member, deepcopy(member)])
    @test_throws ArgumentError fit!(slds, y; max_iter=2, progress=false)
end

# A learned loading in a basis: the latents are k of the basis's r coordinates' span.
function kc_learned(rng; r=6, k=3, p=12, Z=nothing, projection=:noise_weighted, noise=0.2)
    Φ = Matrix(qr(randn(rng, p, r)).Q)[:, 1:r]
    H = randn(rng, r, k)
    Z === nothing && (Z = kc_spd(rng, r))
    Q = first(KCSSD._project(cholesky(Symmetric(Z)), KCSSD._Projection(projection, H)))
    A = 0.85 .* Matrix(qr(randn(rng, k, k)).Q)
    sm = GaussianStateModel(; A, Q, b=zeros(k), x0=zeros(k), P0=Matrix(1.0I, k, k))
    om = GaussianObservationModel(; C=Φ * H, R=Matrix(noise * I, p, p), d=zeros(p))
    return Φ, H, LinearDynamicalSystem(sm, om)
end

function kc_learned_model(rng, Φ, H, k; Q_prior=nothing, R_prior=nothing)
    p, r = size(Φ)
    sm = GaussianStateModel(;
        A=Matrix(0.5I, k, k),
        Q=Matrix(0.3I, k, k),
        b=zeros(k),
        x0=zeros(k),
        P0=Matrix(1.0I, k, k),
        Q_prior,
    )
    om = ConstrainedGaussianObservationModel(;
        basis=Φ, H=H .+ 0.3 .* randn(rng, size(H)), R=Matrix(1.0I, p, p), R_prior
    )
    return LinearDynamicalSystem(sm, om)
end

@testset "projected target: gradient, information, coordinates" begin
    rng = MersenneTwister(13)
    r, k, ν = 5, 2, 7.0
    for projection in (:noise_weighted, :euclidean),
        prior in (kc_prior(rng, r), kc_structured_prior(rng, r))

        nw = length(prior.labels)
        KCSSD._initialize_prior!(prior, kc_spd(rng, r))
        w = prior.weights .+ 0.3 .* randn(rng, nw)
        H = randn(rng, r, k)
        μ, Λ = randn(rng, nw), Matrix(0.5I, nw, nw)
        proj = KCSSD._Projection(projection, H)
        target = KCSSD._IWMean(Matrix(inv(kc_spd(rng, k))), ν, proj)
        f = x -> KCSSD._kernel_objective!(nothing, prior, x, target, μ, Λ)
        grad = zeros(nw)
        KCSSD._kernel_objective!(grad, prior, w, target, μ, Λ)
        h = 1e-6
        e(i) = h .* (1:nw .== i)
        numeric = [(f(w .+ e(i)) - f(w .- e(i))) / 2h for i in 1:nw]
        @test grad ≈ numeric rtol = 1e-5 atol = 1e-6
        # expected curvature at P = E[Σ⁻¹] under the prior: ν draws' information
        K̃ = first(KCSSD._project(KCSSD._kernel_cholesky(prior, w), proj))
        at_prior = KCSSD._IWMean(ν / (ν - k - 1) .* inv(K̃), ν, proj)
        g = x -> (G = zeros(nw); KCSSD._kernel_objective!(G, prior, x, at_prior, μ, Λ); G)
        hessian = reduce(hcat, [(g(w .+ e(i)) - g(w .- e(i))) / 2h for i in 1:nw])
        @test (hessian + hessian') / 2 - Λ ≈
            ν .* KCSSD._projected_information(prior, w, proj) rtol = 1e-4 atol = 1e-6
        # the projection moves with the latents: H → H T⁻¹ takes K̃ to T K̃ T'
        T = randn(rng, k, k) + 2I
        moved = KCSSD._Projection(projection, H / T)
        K̃_moved = first(KCSSD._project(KCSSD._kernel_cholesky(prior, w), moved))
        @test K̃_moved ≈ T * K̃ * T' rtol = 1e-8
        # the prior's term in H, and its gradient
        Fz = cholesky(Symmetric(kernel_covariance(prior, w)))
        P = Matrix(inv(kc_spd(rng, k)))
        term(Hx) = first(KCSSD._projected_loading_term(Hx, Fz, P, ν, projection))
        _, ∂g = KCSSD._projected_loading_term(H, Fz, P, ν, projection)
        E(i) = h .* reshape(1:(r * k) .== i, r, k)
        numeric_H = reshape(
            [(term(H .+ E(i)) - term(H .- E(i))) / 2h for i in 1:(r * k)], r, k
        )
        @test ∂g ≈ numeric_H rtol = 1e-5 atol = 1e-6
    end
end

@testset "KernelIWPrior on Q through a learned loading: monotone fits" begin
    rng = MersenneTwister(14)
    r, k, p = 6, 3, 12
    Φ, H, truth = kc_learned(rng; r, k, p)
    y = rand(rng, truth, fill(60, 6))[2]
    y_test = rand(rng, truth, fill(60, 2))[2]
    for projection in (:noise_weighted, :euclidean),
        loading_iters in (20, 0),
        with_r in (false, true)

        lds = kc_learned_model(
            rng,
            Φ,
            H,
            k;
            Q_prior=KernelIWPrior(kc_prior(rng, r); projection, loading_iters),
            R_prior=with_r ? KernelIWPrior(kc_prior(rng, p)) : nothing,
        )
        trace = fit!(lds, y; max_iter=20, progress=false, y_test)
        @test all(isfinite, collect(trace))
        @test nondecreasing(trace)
        prior = lds.state_model.Q_prior
        @test size(prior.loading) == (r, k)
        @test prior.loading ≈ Φ \ lds.obs_model.C
        @test prior.dof == k + 2
        @test isfinite(elbo(lds, y))
    end
    # one loading step: never worse than the closed-form H or the current one
    lds = kc_learned_model(rng, Φ, H, k; Q_prior=KernelIWPrior(kc_prior(rng, r)))
    kp = lds.state_model.Q_prior
    KCSSD._set_loading!(kp, Φ \ lds.obs_model.C)
    KCSSD._initialize_prior!(kp.prior, kc_spd(rng, r))
    kp.dof = k + 2.0
    A1, Sxx_c, rhs = kc_spd(rng, r), kc_spd(rng, k), randn(rng, r, k)
    H_closed = (A1 \ rhs) / Sxx_c
    Fz = cholesky(Symmetric(kernel_covariance(kp.prior)))
    P = Matrix(inv(lds.state_model.Q))
    total(Hx) =
        dot(Hx, rhs) - dot(Hx, A1 * Hx * Sxx_c) / 2 +
        first(KCSSD._projected_loading_term(Hx, Fz, P, kp.dof, kp.projection))
    H_new = KCSSD._polish_loading(lds, kp, nothing, H_closed, A1, rhs, Sxx_c)
    @test total(H_new) >= max(total(H_closed), total(Φ \ lds.obs_model.C))
    @test total(H_new) > total(H_closed)
end

@testset "KernelIWPrior on Q through a learned loading: the weights follow the truth" begin
    rng = MersenneTwister(15)
    r, k, p = 6, 3, 14
    smooth = [exp(-(i - j)^2 / 4) for i in 1:r, j in 1:r]
    Z = 0.01 .* Matrix(1.0I, r, r) .+ 0.3 .* smooth
    Φ, H, truth = kc_learned(rng; r, k, p, Z, noise=0.05)
    y = rand(rng, truth, fill(100, 20))[2]
    lds = kc_learned_model(rng, Φ, H, k; Q_prior=KernelIWPrior(kc_prior(rng, r); dof=4k))
    fit!(lds, y; max_iter=60, progress=false)
    values = Dict(row.name => row.value for row in kernel_weights(lds.state_model.Q_prior))
    @test values["smooth"] > 5 * values["noise"]
end
