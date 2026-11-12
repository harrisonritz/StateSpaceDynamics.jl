using Test, StateSpaceDynamics, LinearAlgebra, Random
const KESSD = StateSpaceDynamics

function ke_state(k; rng=nothing, q=0)
    A = k == 2 ? [0.9 -0.2; 0.2 0.9] : Matrix(0.8I, k, k)
    return GaussianStateModel(;
        A,
        Q=Matrix(0.1I, k, k),
        b=zeros(k),
        x0=zeros(k),
        P0=Matrix(1.0I, k, k),
        B=q == 0 ? zeros(k, 0) : 0.3 .* randn(rng, k, q),
    )
end

ke_basis(rng, p, r) = Matrix(qr(randn(rng, p, r)).Q)[:, 1:r]

# Kernels over `r` dimensions that together span them.
function ke_prior(rng, r; kwargs...)
    return KernelPrior(
        [
            Kernel(Matrix(1.0I, r, r); name="identity"),
            Kernel(randn(rng, r, 2); name="low", factor=true),
        ];
        kwargs...,
    )
end

# Trials from a Gaussian LDS whose loading lies in the span of `Φ`.
function ke_data(rng, Φ, k; ntrials=6, tsteps=40, noise=0.5)
    p, r = size(Φ)
    truth = LinearDynamicalSystem(
        ke_state(k),
        GaussianObservationModel(;
            C=Φ * randn(rng, r, k), R=Matrix(noise * I, p, p), d=zeros(p)
        ),
    )
    return rand(rng, truth, fill(tsteps, ntrials))[2]
end

function ke_model(rng, Φ, k; prior=nothing, kwargs...)
    p, r = size(Φ)
    H = randn(rng, r, k)
    om = if prior === nothing
        ConstrainedGaussianObservationModel(; basis=Φ, H, R=Matrix(1.0I, p, p))
    else
        ConstrainedKernelGaussianObservationModel(;
            basis=Φ, prior, H, R=Matrix(1.0I, p, p), kwargs...
        )
    end
    return LinearDynamicalSystem(ke_state(k), om)
end

nondecreasing(trace) = all(diff(collect(trace)) .>= -1e-8 * abs(last(trace)))

@testset "q(H) against the dense posterior" begin
    rng = MersenneTwister(21)
    r, k = 5, 3
    spd(n) = (X = randn(rng, n, n + 2); X * X' / n + 0.1I)
    Σ, A1, Sxx = spd(r), spd(r), spd(k)
    Sxx_c = Sxx - 0.3 * (v = randn(rng, k); v * v' / sum(abs2, v)) * minimum(eigvals(Sxx))
    rhs = randn(rng, r, k)
    q = KESSD._loading_posterior(Σ, A1, rhs, Sxx_c, Sxx)

    # vec H: precision I ⊗ Σ⁻¹ + Sxx ⊗ A1; the mean with the centred moment.
    cov = inv(kron(Matrix(1.0I, k, k), inv(Σ)) + kron(Sxx, A1))
    mean = reshape((kron(Matrix(1.0I, k, k), inv(Σ)) + kron(Sxx_c, A1)) \ vec(rhs), r, k)
    @test q.mean ≈ mean
    @test kron(q.right, q.left) * Diagonal(vec(q.var)) * kron(q.right, q.left)' ≈ cov

    # Moments, from the dense covariance: E[H S H']_ab = Σ_jl S_jl E[H_aj H_bl].
    S = spd(k)
    second = vec(mean) * vec(mean)' + cov
    block(j, l) = second[((j - 1) * r + 1):(j * r), ((l - 1) * r + 1):(l * r)]
    @test KESSD._loading_moment(q, S) ≈ sum(S[j, l] * block(j, l) for j in 1:k, l in 1:k)
    @test KESSD._loading_moment(q) ≈ sum(block(j, j) for j in 1:k)
    excess = [tr(A1 * (block(l, j) - mean[:, l] * mean[:, j]')) for j in 1:k, l in 1:k]
    @test KESSD._loading_excess(q, A1) ≈ excess

    # E[log p(H)] + H[q] is minus the Gaussian divergence from the prior.
    prior_cov = kron(Matrix(1.0I, k, k), Σ)
    kl =
        (
            tr(prior_cov \ cov) + dot(vec(mean), prior_cov \ vec(mean)) - r * k +
            logdet(prior_cov) - logdet(cov)
        ) / 2
    @test KESSD._loading_bound(q, Σ) ≈ -kl

    # The gauge H → H T⁻¹ keeps q(H) in the family, with the moments it should have.
    T = randn(rng, k, k) + 2I
    moved = KESSD.LoadingPosterior{Float64}(q.mean / T, q.left, inv(T)' * q.right, q.var)
    @test KESSD._loading_moment(moved, S) ≈ KESSD._loading_moment(q, T \ S / T')
    @test KESSD._loading_excess(moved, A1) ≈ inv(T)' * KESSD._loading_excess(q, A1) / T

    # The bound-optimal gauge: no other T gives a higher E[log p] + H[q].
    Tb = KESSD._bound_gauge(q, Σ)
    at(T) = KESSD._loading_bound(
        KESSD.LoadingPosterior{Float64}(q.mean / T, q.left, inv(T)' * q.right, q.var), Σ
    )
    @test at(Tb) >= at(Matrix(1.0I, k, k))
    @test all(at(Tb) >= at(Tb + 0.05 * randn(rng, k, k)) for _ in 1:20)
end

@testset "the smoother with an uncertain loading" begin
    rng = MersenneTwister(22)
    p, k, r = 6, 2, 4
    Φ = ke_basis(rng, p, r)
    lds = ke_model(rng, Φ, k; prior=ke_prior(rng, r))
    om = lds.obs_model
    root = randn(rng, k, k)
    om.excess .= root * root'

    # A plain Gaussian emission with the same posterior: C̃ = C (I + J⁻¹Δ),
    # R̃ = R + C J⁻¹ Δ J⁻¹ C', J = C'R⁻¹C, has precision J + Δ and the same
    # information vector.
    J = om.C' * (om.R \ om.C)
    W = om.C / J
    surrogate = LinearDynamicalSystem(
        ke_state(k),
        GaussianObservationModel(;
            C=om.C + W * om.excess,
            R=Matrix(Symmetric(om.R + W * om.excess * W')),
            d=copy(om.d),
        ),
    )
    for y in (
        [randn(rng, p, 12) for _ in 1:4],                # equal lengths
        [randn(rng, p, n) for n in (7, 12, 9)],          # ragged
        [randn(rng, p, 10)],                             # one trial
    )
        x, P = smooth(lds, y)
        x_ref, P_ref = smooth(surrogate, y)
        @test all(isapprox.(x, x_ref; rtol=1e-8))
        @test all(isapprox.(P, P_ref; rtol=1e-8))
    end
end

@testset "the emission term of the bound" begin
    rng = MersenneTwister(23)
    p, k, r = 6, 2, 4
    Φ = ke_basis(rng, p, r)
    y = ke_data(rng, Φ, k)
    lds = ke_model(rng, Φ, k; prior=ke_prior(rng, r), gauge=:none)
    fit!(lds, y; max_iter=4, progress=false)
    om = lds.obs_model
    q = emission_posterior(om)
    @test q isa LoadingPosterior && om.C ≈ Φ * q.mean
    @test kernel_weights(om) == kernel_weights(om.prior)

    # E_q(H)[tr(R⁻¹ ΦH Sxx H'Φ')] from the dense covariance of vec H.
    Sxx = (X = randn(rng, k, 9); X * X')
    A1 = Φ' * (om.R \ Φ)
    cov = kron(q.right, q.left) * Diagonal(vec(q.var)) * kron(q.right, q.left)'
    @test dot(om.excess, Sxx) ≈ tr(kron(Sxx, A1) * cov)

    # The bound is the Gaussian one at the mean loading, less ½ tr(Δ Sxx), plus
    # the prior's terms: check by evaluating both on the same data.
    plain = LinearDynamicalSystem(
        deepcopy(lds.state_model),
        GaussianObservationModel(; C=copy(om.C), R=copy(om.R), d=copy(om.d)),
    )
    xs, Ps = smooth(lds, y)
    moment = sum(
        sum(Ps[n][:, :, t] for t in axes(xs[n], 2)) + xs[n] * xs[n]' for n in eachindex(xs)
    )
    prior_terms =
        KESSD._loading_bound(q, kernel_covariance(om.prior)) +
        KESSD._kernel_logprior(om.prior)
    # Not the same q(x) (the plain model's smoother ignores Δ), so compare the
    # pieces that do not depend on it instead: the difference of the two bounds
    # at the kernel model's own q(x) is −½ tr(Δ Sxx) + the prior's terms.
    @test KESSD._obs_prior_logdensity(lds, nothing) ≈ prior_terms
    @test dot(om.excess, moment) > 0
    @test elbo(lds, y) <= elbo(plain, y) + prior_terms
end

@testset "constrained emission (the former C_basis)" begin
    rng = MersenneTwister(211)
    p, k, q, r = 6, 2, 1, 3
    A = randn(rng, p, r)
    make(basis; rng=rng) = LinearDynamicalSystem(
        ke_state(k),
        ConstrainedGaussianObservationModel(;
            basis,
            H=basis \ randn(rng, p, k),
            R=(root = randn(rng, p, p); root * root' / p + I),
            D=zeros(p, q),
        ),
    )

    # The M-step against the constrained least squares solved directly:
    # vec([C d D]) = J θ with θ = [vec(H); vec([d D])].
    lds = make(A)
    sw = KESSD.SmoothWorkspace(Float64, k, p, 8; uy_dim=q)
    suf = KESSD._initialize_td_sufficient_statistics(Float64, lds, [8])
    width = k + 1 + q
    Z = randn(rng, width, 3width)
    G, XY = Z * Z', randn(rng, width, p)
    suf.obs_xx[] = KESSD.pd_gram(G; name="basis fixture")
    suf.obs_xy .= XY
    KESSD._update_loading!(lds, suf)
    Ri = inv(lds.obs_model.R)
    J = cat(
        kron(Matrix(1.0I, k, k), A), Matrix(1.0I, p * (1 + q), p * (1 + q)); dims=(1, 2)
    )
    θ = (J' * kron(G, Ri) * J) \ (J' * vec(Ri * XY'))
    @test hcat(lds.obs_model.C, lds.obs_model.d, lds.obs_model.D) ≈ reshape(J * θ, p, width)
    @test all(iszero, lds.obs_model.excess) && emission_posterior(lds.obs_model) === nothing

    y = [randn(rng, p, 30) for _ in 1:5]
    uy = [randn(rng, q, 30) for _ in 1:5]

    # A square, invertible basis constrains nothing: the fit is the unconstrained one.
    square = make(randn(rng, p, p))
    so = square.obs_model
    free = LinearDynamicalSystem(
        ke_state(k),
        GaussianObservationModel(; C=copy(so.C), R=copy(so.R), d=zeros(p), D=zeros(p, q)),
    )
    elbo_free = fit!(free, y; uy, max_iter=6, progress=false)
    elbo_square = fit!(square, y; uy, max_iter=6, progress=false)
    @test collect(elbo_square) ≈ collect(elbo_free)
    @test square.obs_model.C ≈ free.obs_model.C && square.obs_model.R ≈ free.obs_model.R

    # A narrower basis: the ELBO still never decreases and C stays in the span.
    lds = make(A)
    elbos = collect(fit!(lds, y; uy, max_iter=25, progress=false))
    @test nondecreasing(elbos)
    C = lds.obs_model.C
    @test norm(C - A * (A \ C)) <= 1e-10 * norm(C)
    @test last(elbos) < last(collect(fit!(free, y; uy, max_iter=25, progress=false)))
    @test lds.fit_bool == fill(true, 6)

    sm = lds.state_model
    R = Matrix(1.0I, p, p)
    constrained(; kwargs...) = ConstrainedGaussianObservationModel(; R, kwargs...)
    @test_throws ArgumentError LinearDynamicalSystem(
        sm,
        constrained(; basis=A, C=randn(rng, p, k)),                 # outside the span
    )
    @test_throws ArgumentError LinearDynamicalSystem(
        sm,
        constrained(; basis=[A A[:, 1]], C=A[:, 1:k]),              # rank deficient
    )
    @test_throws KESSD.DimensionMismatchError LinearDynamicalSystem(
        sm, constrained(; basis=A[1:4, :], C=A[:, 1:k])
    )
    @test_throws ArgumentError constrained(; basis=A)                   # no start
    @test_throws ArgumentError constrained(; basis=A, C=A[:, 1:k], H=zeros(r, k))
    @test_throws ArgumentError constrained(; basis=A, C=A[:, 1:k], gauge=:none)
    @test_throws ArgumentError fit!(lds, y; uy, depends_on=(C=[1, 1, 2, 2, 2],))
    @test_throws ArgumentError CompositeObservationModel((a=lds.obs_model,))
end

@testset "kernel-prior emissions: fitting" begin
    rng = MersenneTwister(24)
    p, k, r = 8, 2, 4
    Φ = ke_basis(rng, p, r)
    y = ke_data(rng, Φ, k)

    # The trace never decreases, in any gauge, with flat hyperpriors.
    final = Dict{Symbol,Float64}()
    for gauge in (:none, :bound)
        lds = ke_model(
            MersenneTwister(1), Φ, k; prior=ke_prior(MersenneTwister(2), r), gauge
        )
        @test lds.fit_bool == fill(true, 7)
        trace = fit!(lds, y; max_iter=80, progress=false)
        @test nondecreasing(trace)
        final[gauge] = last(trace)
        om = lds.obs_model
        @test norm(om.C - Φ * (Φ \ om.C)) <= 1e-10 * norm(om.C)
        @test isposdef(Symmetric(om.excess))
        @test all(row.sd > 0 for row in kernel_weights(om))
        @test elbo(lds, y) ≈ last(trace)
    end
    # The gauge step only speeds the same ascent.
    @test final[:bound] >= final[:none]

    # The unit-moment gauge is a constraint: its fit runs and stays there.
    lds = ke_model(
        MersenneTwister(1), Φ, k; prior=ke_prior(MersenneTwister(2), r), gauge=:moment
    )
    trace = fit!(lds, y; max_iter=40, tol=0.0, progress=false)
    @test all(isfinite, trace)
    xs = smooth(lds, y)[1]
    # (the gauge was set under the q(x) before the last M-step, so only roughly)
    centred =
        reduce(hcat, xs) .- sum(sum(x; dims=2) for x in xs) ./ sum(size(x, 2) for x in xs)
    @test centred * centred' / size(centred, 2) ≈ I atol = 0.5

    # Held weights: the prior is fixed at the given weights and nothing is fitted
    # to it.
    prior = ke_prior(MersenneTwister(2), r; weights=[0.3, -1.0])
    held = LinearDynamicalSystem(
        ke_state(k),
        ConstrainedKernelGaussianObservationModel(;
            basis=Φ, prior, H=randn(rng, r, k), R=Matrix(1.0I, p, p)
        );
        fit_bool=(weights=false,),
    )
    trace = fit!(held, y; max_iter=40, progress=false)
    @test nondecreasing(trace)
    @test prior.weights == [0.3, -1.0]

    # A very wide prior is no prior on the mean: given the same statistics, the
    # posterior mean of the loading is the constrained maximum-likelihood one.
    # (Its uncertainty stays, so the fits themselves differ.)
    start = randn(rng, r, k)
    wide = LinearDynamicalSystem(
        ke_state(k),
        ConstrainedKernelGaussianObservationModel(;
            basis=Φ,
            prior=KernelPrior([Kernel(Matrix(1.0I, r, r); name="I")]; weights=[log(1e12)]),
            H=copy(start),
            R=Matrix(1.0I, p, p),
        ),
    )
    ml = LinearDynamicalSystem(
        ke_state(k),
        ConstrainedGaussianObservationModel(; basis=Φ, H=copy(start), R=Matrix(1.0I, p, p)),
    )
    suf = KESSD._initialize_td_sufficient_statistics(Float64, ml, [8])
    Z = randn(rng, k + 1, 3k + 3)
    suf.obs_xx[] = KESSD.pd_gram(Z * Z'; name="wide fixture")
    suf.obs_xy .= randn(rng, k + 1, p)
    KESSD._update_loading!(wide, suf)
    KESSD._update_loading!(ml, suf)
    @test wide.obs_model.C ≈ ml.obs_model.C rtol = 1e-8
    @test wide.obs_model.d ≈ ml.obs_model.d rtol = 1e-8
    @test tr(KESSD._refresh_excess!(wide.obs_model)) > 0

    # Without a basis the prior is on C itself.
    direct = LinearDynamicalSystem(
        ke_state(k),
        KernelGaussianObservationModel(;
            prior=ke_prior(MersenneTwister(2), p), C=randn(rng, p, k), R=Matrix(1.0I, p, p)
        ),
    )
    trace = fit!(direct, y; max_iter=40, progress=false)
    @test nondecreasing(trace)
    @test size(emission_posterior(direct.obs_model).mean) == (p, k)

    # Inputs on both sides, ragged trials, a held-out set.
    q = 2
    ragged = [randn(rng, p, n) for n in (20, 31, 25, 20)]
    ux = [randn(rng, q, size(t, 2)) for t in ragged]
    uy = [randn(rng, 1, size(t, 2)) for t in ragged]
    full = LinearDynamicalSystem(
        ke_state(k; rng, q),
        ConstrainedKernelGaussianObservationModel(;
            basis=Φ,
            prior=ke_prior(rng, r),
            H=randn(rng, r, k),
            R=Matrix(1.0I, p, p),
            D=zeros(p, 1),
        ),
    )
    trace = fit!(
        full,
        ragged;
        ux,
        uy,
        max_iter=30,
        progress=false,
        y_test=ragged[1:2],
        ux_test=ux[1:2],
        uy_test=uy[1:2],
    )
    @test nondecreasing(trace)
    @test length(trace.test) == length(trace) && all(isfinite, trace.test)
end

@testset "kernel-prior emissions: the gauge" begin
    rng = MersenneTwister(25)
    p, k, r = 8, 2, 4
    Φ = ke_basis(rng, p, r)
    y = ke_data(rng, Φ, k)

    # Without a prior on the loading the bound does not depend on the gauge,
    # state priors included.
    lds = ke_model(rng, Φ, k)
    sm = lds.state_model
    sm.Q_prior = IWPrior(; Ψ=Matrix(0.5I, k, k), ν=4.0)
    sm.P0_prior = IWPrior(; Ψ=Matrix(1.0I, k, k), ν=4.0)
    sm.AB_prior = MNPrior(; M₀=0.1 .* randn(rng, k, k + 1), Λ=Matrix(2.0I, k + 1, k + 1))
    fit!(lds, y; max_iter=5, progress=false)
    before = elbo(lds, y)
    KESSD._apply_latent_gauge!(lds, randn(rng, k, k) + 2I)
    @test elbo(lds, y) ≈ before

    # With one, the bound-optimal gauge raises it, by what the loading's terms gain.
    lds = ke_model(rng, Φ, k; prior=ke_prior(rng, r), gauge=:none)
    fit!(lds, y; max_iter=5, progress=false)
    om = lds.obs_model
    before, terms = elbo(lds, y), KESSD._obs_prior_logdensity(lds, nothing)
    T = KESSD._bound_gauge(om.posterior, kernel_covariance(om.prior))
    KESSD._apply_latent_gauge!(lds, T)
    gain = KESSD._obs_prior_logdensity(lds, nothing) - terms
    @test gain > 0
    @test elbo(lds, y) - before ≈ gain rtol = 1e-6
    @test KESSD._bound_gauge(om.posterior, kernel_covariance(om.prior)) ≈ I atol = 1e-8

    # A held dynamics parameter pins the gauge: no step is taken.
    pinned = LinearDynamicalSystem(
        ke_state(k),
        ConstrainedKernelGaussianObservationModel(;
            basis=Φ, prior=ke_prior(rng, r), H=randn(rng, r, k), R=Matrix(1.0I, p, p)
        );
        fit_bool=(Q=false,),
    )
    Q = copy(pinned.state_model.Q)
    @test nondecreasing(fit!(pinned, y; max_iter=10, progress=false))
    @test pinned.state_model.Q == Q
end

@testset "kernel-prior emissions: checks" begin
    rng = MersenneTwister(26)
    p, k, r = 6, 2, 3
    Φ = ke_basis(rng, p, r)
    R = Matrix(1.0I, p, p)
    sm = ke_state(k)
    build(; prior=ke_prior(rng, r), kwargs...) = LinearDynamicalSystem(
        sm,
        ConstrainedKernelGaussianObservationModel(;
            basis=Φ, prior, H=randn(rng, r, k), R, kwargs...
        ),
    )
    @test build() isa LinearDynamicalSystem
    @test_throws DimensionMismatch build(; prior=ke_prior(rng, r + 1))
    @test_throws ArgumentError build(; gauge=:whiten)
    @test_throws ArgumentError build(; weight_iters=0)
    @test_throws ArgumentError build(; tree_every=0)
    @test_throws ArgumentError KernelGaussianObservationModel(;
        prior=ke_prior(rng, p), basis=Φ, C=randn(rng, p, k), R
    )
    @test_throws DimensionMismatch LinearDynamicalSystem(
        sm, KernelGaussianObservationModel(; prior=ke_prior(rng, r), C=randn(rng, p, k), R)
    )
    @test_throws ArgumentError kernel_weights(
        ConstrainedGaussianObservationModel(; basis=Φ, H=randn(rng, r, k), R)
    )
    @test_throws ArgumentError LinearDynamicalSystem(
        sm, build().obs_model; fit_bool=(warp=true,)
    )
    @test occursin("Constrained Kernel", sprint(show, build().obs_model))
    @test occursin("kernel weights", sprint(show, build()))
end

@testset "kernel-prior emissions: scoring, sampling, refusals" begin
    rng = MersenneTwister(27)
    p, k, r = 8, 2, 4
    Φ = ke_basis(rng, p, r)
    y = ke_data(rng, Φ, k)
    lds = ke_model(rng, Φ, k; prior=ke_prior(rng, r))
    fit!(lds, y; max_iter=10, progress=false)
    om = lds.obs_model

    # The plug-in likelihood is the Gaussian one at the mean loading.
    plain = LinearDynamicalSystem(
        deepcopy(lds.state_model),
        GaussianObservationModel(; C=copy(om.C), R=copy(om.R), d=copy(om.d)),
    )
    @test loglikelihood(lds, y) ≈ loglikelihood(plain, y)
    x, drawn = rand(MersenneTwister(1), lds, fill(15, 3))
    @test size(drawn[1]) == (p, 15) && all(isfinite, drawn[1])

    switching = SLDS([0.9 0.1; 0.1 0.9], [0.5, 0.5], [lds, deepcopy(lds)])
    @test_throws ArgumentError validate_SLDS(switching)
    @test_throws ArgumentError fit!(switching, y; max_iter=2, progress=false)
end

@testset "kernel-prior emissions: recovery" begin
    rng = MersenneTwister(28)
    p, k, r = 30, 3, 20
    Φ = ke_basis(rng, p, r)
    F = randn(rng, r, 2)
    kernels() = [
        Kernel(F; name="pattern", factor=true), Kernel(Matrix(1.0I, r, r); name="identity")
    ]
    # Loadings drawn from the prior: mostly in the span of F.
    H = F * randn(rng, 2, k) + 0.05 .* randn(rng, r, k)
    truth = LinearDynamicalSystem(
        ke_state(k), GaussianObservationModel(; C=Φ * H, R=Matrix(4.0I, p, p), d=zeros(p))
    )
    y = rand(rng, truth, fill(25, 3))[2]
    y_test = rand(rng, truth, fill(25, 40))[2]
    start = randn(rng, r, k)
    base() = (; basis=Φ, H=copy(start), R=Matrix(1.0I, p, p))

    free = LinearDynamicalSystem(ke_state(k), ConstrainedGaussianObservationModel(; base()...))
    fit!(free, y; max_iter=200, progress=false)
    fits = map((:bound, :none)) do gauge
        model = LinearDynamicalSystem(
            ke_state(k),
            ConstrainedKernelGaussianObservationModel(;
                base()..., prior=KernelPrior(kernels(); prior_sd=3.0), gauge
            ),
        )
        model, collect(fit!(model, y; max_iter=200, progress=false))
    end
    with_prior, trace = fits[1]

    # With few trials the prior helps on held-out data, and its weights say why.
    @test loglikelihood(with_prior, y_test) > loglikelihood(free, y_test)
    weights = kernel_weights(with_prior.obs_model)
    power = [w.value * tr(k.factor * k.factor') for (w, k) in zip(weights, kernels())]
    @test power[1] > power[2]

    # The gauge step reaches the unassisted fit's bound in fewer iterations.
    slow = fits[2][2]
    @test nondecreasing(trace) && nondecreasing(slow)
    @test something(findfirst(>=(last(slow) - 1e-3), trace), typemax(Int)) <= length(slow)
    @test last(trace) >= last(slow) - 1e-3
end

@testset "kernel-prior emissions: structured priors in a fit" begin
    rng = MersenneTwister(29)
    p, k, r, n = 12, 2, 6, 9
    Φ = ke_basis(rng, p, r)
    y = ke_data(rng, Φ, k)
    block = repeat(1:3; inner=3)
    family = CorrelationFamily(
        [reshape(Float64.(block .== b), n, 1) for b in 1:3];
        type=:coherence,
        names=["b1", "b2", "b3"],
    )
    structure = ScaleCorrelation(
        randn(rng, r, n), ones(n), block; names=["b1", "b2", "b3"], families=[family]
    )
    prior = KernelPrior(
        [Kernel(Matrix(1.0I, r, r); name="identity", type=:nuisance)];
        structure,
        levels=(scale=:hierarchical, coherence=:hierarchical),
        correlated=[(:scale, :coherence)],
    )
    lds = LinearDynamicalSystem(
        ke_state(k),
        ConstrainedKernelGaussianObservationModel(;
            basis=Φ, prior, H=randn(rng, r, k), R=Matrix(1.0I, p, p), tree_every=5
        ),
    )
    trace = collect(fit!(lds, y; max_iter=40, tol=0.0, progress=false))
    # Fitted trees can lower the trace a little when they are updated; between
    # updates it does not decrease.
    steps = diff(trace)
    @test all(isfinite, trace)
    @test count(<(-1e-6), steps) <= length(steps) ÷ 5
    @test last(trace) > first(trace)
    table = kernel_weights(lds.obs_model)
    @test length(table) == 7
    @test all(0 < row.value < 1 for row in table if row.link === :logit)
end
