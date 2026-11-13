using Test, StateSpaceDynamics, LinearAlgebra, Random, ForwardDiff, MAT
const KPSSD = StateSpaceDynamics

# Factors of `g` kernels in `r` dimensions; the first and last are full rank.
function kp_factors(rng, r; ranks=(r, 3, 2, r))
    return [randn(rng, r, m) for m in ranks]
end

# The second moment of `n` draws from N(0, Σ).
function kp_moment(rng, Σ, n)
    X = cholesky(Symmetric(Σ)).L * randn(rng, size(Σ, 1), n)
    return X * X' / n
end

# The objective written out from the definition, for any element type.
function kp_reference_objective(w, factors, S, n, μ, Λ)
    Σ = sum(exp(w[g]) * factors[g] * factors[g]' for g in eachindex(factors))
    return n / 2 * (logdet(Σ) + tr(Σ \ S)) + dot(w - μ, Λ * (w - μ)) / 2
end

@testset "Kernel" begin
    rng = MersenneTwister(1)
    F = randn(rng, 5, 2)
    whole = Kernel(F * F'; name="a")
    factored = Kernel(F; name="a", factor=true)
    @test size(whole.factor) == (5, 2)                  # stored at its rank
    @test whole.factor * whole.factor' ≈ F * F'
    @test factored.factor == F
    wide = Kernel(randn(rng, 5, 9); name="wide", factor=true)
    @test size(wide.factor) == (5, 5)                   # no wider than its rank
    @test whole.unit == "a" && whole.type === :kernel && whole.group === nothing
    @test Kernel(F * F'; name=:a, group=3).group == "3"

    @test_throws ArgumentError Kernel(randn(rng, 5, 5); name="asymmetric")
    @test_throws ArgumentError Kernel(-Matrix(1.0I, 3, 3); name="negative")
    @test_throws ArgumentError Kernel(zeros(3, 3); name="zero")
    @test_throws ArgumentError Kernel([1.0 NaN; NaN 1.0]; name="nan")
    @test_throws ArgumentError Kernel(randn(rng, 4, 3); name="not square")
    @test_throws ArgumentError Kernel(F * F'; name="a", prior_sd=0.0)
    @test_throws ArgumentError Kernel(F * F'; name="a", prior_mean=Inf)
end

@testset "KernelPrior checks" begin
    rng = MersenneTwister(2)
    r = 4
    eye(name; kwargs...) = Kernel(Matrix(1.0I, r, r); name, kwargs...)
    low(name; kwargs...) = Kernel(randn(rng, r, 1); name, factor=true, kwargs...)

    @test_throws ArgumentError KernelPrior(Kernel{Float64}[])
    @test_throws DimensionMismatch KernelPrior([
        eye("a"), Kernel(Matrix(1.0I, 3, 3); name="b")
    ])
    @test_throws ArgumentError KernelPrior([eye("a"), eye("a")])
    @test KernelPrior([eye("a"), eye("a"; type=:other)]) isa KernelPrior
    @test_throws ArgumentError KernelPrior([eye("a")]; levels=:pooled)
    @test_throws ArgumentError KernelPrior([eye("a")]; levels=(missing_type=:flat,))
    @test_throws ArgumentError KernelPrior([eye("a")]; levels=:hierarchical)
    @test_throws ArgumentError KernelPrior([eye("a"), eye("b")]; levels=:nested)
    @test_throws ArgumentError KernelPrior(
        [eye("a"; group="g"), eye("b"; group="g")]; levels=:nested
    )
    @test_throws ArgumentError KernelPrior(
        [eye("a"; prior_sd=2.0), eye("b")]; levels=:hierarchical
    )
    @test_throws ArgumentError KernelPrior([eye("a")]; prior_sd=(other=1.0,))
    @test_throws ArgumentError KernelPrior([eye("a")]; sd_prior=(0.5, -1.0))
    @test_throws ArgumentError KernelPrior([eye("a")]; root_sd=0.0)
    @test_throws DimensionMismatch KernelPrior([eye("a")]; weights=[0.0, 1.0])

    # correlated sets
    pair(t) = [eye("u1"; type=t), eye("u2"; type=t), eye("u3"; type=t)]
    both = vcat(pair(:a), pair(:b))
    @test KernelPrior(both; levels=:hierarchical, correlated=[(:a, :b)]) isa KernelPrior
    @test_throws ArgumentError KernelPrior(both; correlated=[(:a, :b)])       # flat
    @test_throws ArgumentError KernelPrior(both; levels=:hierarchical, correlated=[(:a,)])
    @test_throws ArgumentError KernelPrior(
        both; levels=:hierarchical, correlated=[(:a, :c)]
    )
    @test_throws ArgumentError KernelPrior(
        both; levels=(a=:hierarchical, b=:nested), correlated=[(:a, :b)]
    )
    @test_throws ArgumentError KernelPrior(
        vcat(both, pair(:c)); levels=:hierarchical, correlated=[(:a, :b), (:b, :c)]
    )
    disjoint = vcat(pair(:a), [eye("v1"; type=:b), eye("v2"; type=:b)])
    @test_throws ArgumentError KernelPrior(
        disjoint; levels=:hierarchical, correlated=[(:a, :b)]
    )

    # the kernels together must span the space, and the moment must fit
    thin = KernelPrior([low("a"), low("b")])
    @test_throws ArgumentError fit_kernel_weights!(thin, Matrix(1.0I, r, r), 10)
    prior = KernelPrior([eye("a")])
    @test_throws DimensionMismatch fit_kernel_weights!(prior, Matrix(1.0I, 3, 3), 10)
    @test_throws ArgumentError fit_kernel_weights!(prior, Matrix(1.0I, r, r), 0)
    @test_throws ArgumentError kernel_covariance(prior)          # no weights yet
    @test_throws ArgumentError kernel_evidence(prior, Matrix(1.0I, r, r), 10)
end

@testset "kernel objective: gradient and information" begin
    rng = MersenneTwister(3)
    r = 5
    factors = kp_factors(rng, r)
    prior = KernelPrior([
        Kernel(F; name="k$g", factor=true) for (g, F) in enumerate(factors)
    ])
    g = length(factors)
    w = 0.5 .* randn(rng, g)
    S = kp_moment(rng, Matrix(1.0I, r, r), 20)
    n = 7.0
    μ = randn(rng, g)
    root = randn(rng, g, g)
    Λ = root * root' + I

    grad = zeros(g)
    value = KPSSD._kernel_objective!(grad, prior, w, S, n, μ, Λ)
    reference = x -> kp_reference_objective(x, factors, S, n, μ, Λ)
    @test value ≈ reference(w)
    @test grad ≈ ForwardDiff.gradient(reference, w)
    @test KPSSD._kernel_objective!(nothing, prior, w, S, n, μ, Λ) == value
    @test kernel_covariance(prior, w) ≈
        sum(exp(w[i]) * factors[i] * factors[i]' for i in 1:g)

    # The Fisher information is the Hessian of the per-draw objective when the
    # moment is the model's own covariance.
    Σ = kernel_covariance(prior, w)
    expected = ForwardDiff.hessian(
        x -> kp_reference_objective(x, factors, Σ, 1.0, zeros(g), zeros(g, g)), w
    )
    @test KPSSD._kernel_information(prior, w) ≈ expected

    # No positive definite covariance: an infinite objective, not an error.
    @test KPSSD._kernel_objective!(grad, prior, fill(-Inf, g), S, n, μ, Λ) == Inf
end

@testset "kernel weights: flat prior" begin
    rng = MersenneTwister(4)
    r = 6
    factors = kp_factors(rng, r)
    θ = [2.0, 0.5, 1.0, 0.1]
    Σ = sum(θ[g] * factors[g] * factors[g]' for g in 1:4)
    kernels = [Kernel(F; name="k$g", factor=true) for (g, F) in enumerate(factors)]

    # Many draws and a wide prior: the weights that generated them.
    n = 20_000
    S = many = kp_moment(rng, Σ, n)
    prior = KernelPrior(kernels; prior_sd=5.0)
    found = fit_kernel_weights!(prior, S, n)
    @test found.converged && found.rounds == 1
    @test exp.(prior.weights) ≈ θ rtol = 0.1
    table = kernel_weights(prior)
    @test [row.name for row in table] == ["k1", "k2", "k3", "k4"]
    @test [row.value for row in table] == exp.(prior.weights)
    @test all(row.sd > 0 for row in table)
    @test all(abs(log(θ[g]) - table[g].weight) < 4 * table[g].sd for g in 1:4)
    @test kernel_covariance(prior) ≈ Σ rtol = 0.1

    # The fit is a stationary point of the stated objective.
    μ, Λ = KPSSD._weight_prior(prior)
    @test μ ≈ [log(tr(S) / sum(abs2, F)) for F in factors]       # the "alone" centres
    @test Λ ≈ Matrix(I, 4, 4) / 25
    grad = zeros(4)
    value = KPSSD._kernel_objective!(grad, prior, prior.weights, S, Float64(n), μ, Λ)
    @test value ≈ found.objective
    @test norm(grad) < 1e-4 * abs(value)

    # Few draws: a tight prior holds the weights at its centres, a kernel's own
    # prior overrides the default, and the centres are kept by a second fit.
    S = kp_moment(rng, Σ, 4)
    tight = KernelPrior(
        [
            Kernel(factors[1]; name="k1", factor=true, prior_mean=1.0, prior_sd=1e-4),
            kernels[2:4]...,
        ];
        prior_mean=-0.5,
        prior_sd=1e-4,
    )
    fit_kernel_weights!(tight, S, 4)
    @test tight.weights ≈ [1.0, -0.5, -0.5, -0.5] atol = 1e-3
    centres = copy(KPSSD._weight_prior(tight)[1])
    fit_kernel_weights!(tight, 3 .* S, 4)
    @test KPSSD._weight_prior(tight)[1] == centres

    # Starting weights are used, and a fit from the optimum stays there.
    again = KernelPrior(kernels; prior_sd=5.0, weights=prior.weights)
    @test again.weights == prior.weights
    fit_kernel_weights!(again, many, n)
    @test again.weights ≈ prior.weights atol = 1e-5

    # The Laplace evidence prefers the kernels that made the data.
    S = kp_moment(rng, Σ, 200)
    right = KernelPrior(kernels)
    wrong = KernelPrior(kernels[[1, 4]])
    fit_kernel_weights!(right, S, 200)
    fit_kernel_weights!(wrong, S, 200)
    @test kernel_evidence(right, S, 200) > kernel_evidence(wrong, S, 200)
end

# Reference values from dynameg's Python code (test_data/kernel_emissions).
function kp_reference()
    return matread(
        joinpath(@__DIR__, "..", "test_data", "kernel_emissions", "reference.mat")
    )
end

@testset "kernel weights: dynameg fit_kernels" begin
    ref = kp_reference()
    for case in ("flat1", "flat2")
        Z, S, n = ref["$(case)_Z"], ref["$(case)_S"], ref["$(case)_n"]
        sd, centre = vec(ref["$(case)_prior_sd"]), vec(ref["$(case)_prior_mean"])
        kernels = map(1:size(Z, 1)) do g
            # flat1 leaves the centres to the default rule; flat2 gives them
            own = case == "flat2" ? (; prior_mean=centre[g], prior_sd=sd[g]) : (;)
            Kernel(Z[g, :, :]; name="k$g", own...)
        end
        prior = KernelPrior(kernels)
        found = fit_kernel_weights!(prior, S, n; max_iter=2000, g_tol=1e-11, f_reltol=0.0)
        @test found.converged
        @test KPSSD._weight_prior(prior)[1] ≈ centre
        @test exp.(prior.weights) ≈ vec(ref["$(case)_map_theta"]) rtol = 1e-7
        @test prior.weights_cov ≈ ref["$(case)_map_posterior_cov"] rtol = 1e-6
        @test kernel_evidence(prior, S, n) ≈ ref["$(case)_map_free_energy"] rtol = 1e-9
        # dynameg reports 2/n of the minimized quantity
        @test 2 * found.objective / n ≈ ref["$(case)_map_objective"] rtol = 1e-9
        info = KPSSD._kernel_information(prior, prior.weights)
        θ = exp.(prior.weights)
        @test info ≈ θ .* ref["$(case)_map_information"] .* θ' rtol = 1e-6
    end
end
