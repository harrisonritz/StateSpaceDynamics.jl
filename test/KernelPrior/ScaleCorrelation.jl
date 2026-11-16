using Test, StateSpaceDynamics, LinearAlgebra, Random, SparseArrays, ForwardDiff, MAT
const SCSSD = StateSpaceDynamics

function sc_reference()
    return matread(
        joinpath(@__DIR__, "..", "test_data", "kernel_emissions", "reference.mat")
    )
end

sc_sigmoid(a) = 1 / (1 + exp(-a))
sc_logit(p) = log(p / (1 - p))

# The stick-broken weights of the logits `a`, the ancestors given by hand.
function sc_rho(a, ancestors)
    ρ = zeros(eltype(a), length(a))
    for c in eachindex(a)
        ρ[c] = (1 - sum(ρ[ancestors[c]]; init=zero(eltype(a)))) * sc_sigmoid(a[c])
    end
    return ρ
end

# C = I + Σ_c ρ_c (R_c − I_c) from the definition, dense.
function sc_correlation(ρ, factors)
    n = size(first(factors), 1)
    C = Matrix{eltype(ρ)}(I, n, n)
    for (c, F) in enumerate(factors)
        norms = vec(sqrt.(sum(abs2, Matrix(F); dims=2)))
        covered = norms .> 0
        B = Matrix(F) ./ ifelse.(covered, norms, 1.0)
        C += ρ[c] .* (B * B' - Diagonal(covered))
    end
    return C
end

# Σ_sc = G S C S G' from the definition, for any element type of `w`.
function sc_sigma(w, G, v, block, factors, ancestors)
    g = maximum(block)
    θ = exp.(w[1:g])
    scale = sqrt.(θ[block] .* v)
    C = sc_correlation(sc_rho(w[(g + 1):end], ancestors), factors)
    return G * (scale .* C .* scale') * G'
end

# 14 columns in 3 scale blocks that cut across the kernels; three families:
# two networks (columns 13:14 in neither), four parcels (three nested, one
# outside the networks), two finer kernels inside a parcel each.
function sc_problem(rng; r=5)
    n = 14
    G = randn(rng, r, n)
    v = 0.5 .+ rand(rng, n)
    block = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 3]
    factor(cols, m) = (F = zeros(n, m); F[cols, :] .= randn(rng, length(cols), m); F)
    coarse = [factor(1:6, 1), factor(7:12, 1)]
    fine = [factor(1:3, 2), factor(4:6, 2), factor(7:9, 2), factor(13:14, 1)]
    finer = [factor(1:3, 3), factor(7:9, 3)]
    ancestors = [Int[], Int[], [1], [1], [2], Int[], [1, 3], [2, 5]]
    families = [
        CorrelationFamily(coarse; type=:coarse, names=["c1", "c2"]),
        CorrelationFamily(fine; type=:fine, names=["f1", "f2", "f3", "f4"]),
        CorrelationFamily(sparse.(finer); type=:finer, names=["e1", "e3"]),
    ]
    sc = ScaleCorrelation(G, v, block; names=["b1", "b2", "b3"], families)
    return (; G, v, block, factors=vcat(coarse, fine, finer), ancestors, families, sc)
end

@testset "scale-correlation: dynameg ScaleCorrelationPrior" begin
    ref = sc_reference()
    G, v = ref["sc1_G"], vec(ref["sc1_variance"])
    block = Int.(vec(ref["sc1_membership"]))
    names = ["b0", "b1", "b2", "b3"]
    net = CorrelationFamily(
        [ref["sc1_net_F$i"] for i in 0:1]; type=:net, names=["n0", "n1"]
    )
    par = CorrelationFamily([ref["sc1_par_F$i"] for i in 0:3]; type=:par, names)
    sc = ScaleCorrelation(G, v, block; names, families=[net, par])
    @test sc isa ScaleCorrelation{Float64}
    @test sc.ancestors == [Int[], Int[], [1], [1], [2], [2]]
    # every parcel is a whole block: no Z_cb beyond the blocks' own and the networks'
    @test length(sc.Z) == 4

    # the reference is ordered (log θ, a, log ν); the prior (log ν, log θ, a)
    order = vcat(11, 1:10)
    w = vec(ref["sc1_x"])[order]
    prior = KernelPrior(
        [Kernel(Matrix(1.0I, 6, 6); name="I", type=:nuisance)]; structure=sc, weights=w
    )
    @test [l.type for l in prior.labels] == vcat(:nuisance, fill(:scale, 4), :net, :net, fill(:par, 4))
    @test [l.link for l in prior.labels] == vcat(fill(:log, 5), fill(:logit, 6))
    @test kernel_covariance(prior, w) ≈ ref["sc1_covariance"] rtol = 1e-11
    values = [row.value for row in kernel_weights(prior)]
    @test values[1] ≈ ref["sc1_nu"] rtol = 1e-12
    @test values[2:5] ≈ vec(ref["sc1_theta"]) rtol = 1e-12
    @test values[6:11] ≈ vec(ref["sc1_rho"]) rtol = 1e-12
    @test SCSSD._structure_values(sc, w[2:end]) == values[2:end]

    S, n = ref["sc1_S"], ref["sc1_n"]
    grad = zeros(11)
    value = SCSSD._kernel_objective!(grad, prior, w, S, n, zeros(11), zeros(11, 11))
    @test value ≈ ref["sc1_objective"] rtol = 1e-11
    @test grad ≈ vec(ref["sc1_gradient"])[order] rtol = 1e-9
    @test SCSSD._kernel_information(prior, w) ≈ ref["sc1_information"][order, order] rtol =
        1e-9

    # ∂ρ/∂a, through the gradient of a function linear in each derivative
    D = zeros(6, 6, 10)
    SCSSD._derivatives!(D, sc, w[2:end])
    θ, ρ = values[2:5], values[6:11]
    J = ref["sc1_rho_jacobian"]
    Dρ = [sum(inv(J)[c, i] * D[:, :, 4 + c] for c in 1:6) for i in 1:6]   # in ρ
    scale = sqrt.(θ[block] .* v)
    for (c, F) in enumerate(vcat(net.factors, par.factors))
        covered = vec(sum(abs2, F; dims=2)) .> 0
        expected = G * (scale .* (F * F' - Diagonal(covered)) .* scale') * G'
        @test Dρ[c] ≈ expected rtol = 1e-9
    end
end

@testset "scale-correlation: from the definition" begin
    rng = MersenneTwister(11)
    p = sc_problem(rng)
    sc = p.sc
    @test sc.ancestors == p.ancestors
    @test SCSSD._structure_dim(sc) == 5
    labels = SCSSD._structure_labels(sc)
    @test [l.name for l in labels] == ["b1", "b2", "b3", "c1", "c2", "f1", "f2", "f3", "f4", "e1", "e3"]
    @test [l.type for l in labels] == vcat(fill(:scale, 3), fill(:coarse, 2), fill(:fine, 4), fill(:finer, 2))
    @test all(l.unit == l.name && l.group === nothing for l in labels)
    Gw = p.G .* sqrt.(p.v)'
    traces = SCSSD._structure_traces(sc)
    @test traces[1:3] ≈ [sum(abs2, Gw[:, p.block .== b]) for b in 1:3]
    @test all(isnan, traces[4:end])

    # the factors' rows are at unit norm on the covered columns
    for family in p.families, (B, cols) in zip(family.factors, family.covered)
        norms = vec(sqrt.(sum(abs2, B; dims=2)))
        @test norms[cols] ≈ ones(length(cols))
        @test all(iszero, norms[setdiff(1:14, cols)])
    end

    covering = [findall(F -> any(!iszero, F[i, :]), p.factors) for i in 1:14]
    for scale in (1.0, 10.0, 50.0)
        w = scale .* randn(rng, 11)
        Σ = zeros(5, 5)
        SCSSD._add_covariance!(Σ, sc, w)
        expected = sc_sigma(w, p.G, p.v, p.block, p.factors, p.ancestors)
        @test Σ ≈ expected rtol = 1e-10
        @test SCSSD._add_covariance!(copy(Σ), sc, w) ≈ 2 .* expected rtol = 1e-10  # it adds

        values = SCSSD._structure_values(sc, w)
        @test values[1:3] == exp.(w[1:3])
        ρ = values[4:end]
        @test ρ ≈ sc_rho(w[4:end], p.ancestors)
        @test all(0 .<= ρ .<= 1)
        @test all(sum(ρ[cs]; init=0.0) <= 1 + 1e-12 for cs in covering)
        C = sc_correlation(ρ, p.factors)
        @test diag(C) ≈ ones(14)
        @test eigmin(Symmetric(C)) > -1e-12
    end

    # every weight starts at θ = exp(logscale), ρ = 0.1
    start = SCSSD._structure_start(sc, -0.7)
    @test start[1:3] == fill(-0.7, 3)
    @test SCSSD._structure_values(sc, start)[4:end] ≈ fill(0.1, 8)

    # a structure alone is a prior; the element type follows the gain
    alone = KernelPrior(Kernel{Float64}[]; structure=sc)
    w = 0.3 .* randn(rng, 11)
    @test kernel_covariance(alone, w) ≈
        sc_sigma(w, p.G, p.v, p.block, p.factors, p.ancestors)
    single = ScaleCorrelation(
        Float32.(p.G), p.v, p.block; names=["b1", "b2", "b3"], families=p.families
    )
    @test single isa ScaleCorrelation{Float32}
    Σ = zeros(Float32, 5, 5)
    SCSSD._add_covariance!(Σ, single, Float32.(w))
    @test Σ ≈ kernel_covariance(alone, w) rtol = 1e-4

    # no family: the scales alone
    plain = ScaleCorrelation(p.G, p.v, p.block; names=["b1", "b2", "b3"])
    Σ = zeros(5, 5)
    SCSSD._add_covariance!(Σ, plain, w[1:3])
    @test Σ ≈ Gw * Diagonal(exp.(w[1:3])[p.block]) * Gw'
end

@testset "scale-correlation: derivatives" begin
    rng = MersenneTwister(12)
    p = sc_problem(rng)
    sc = p.sc
    reference = x -> sc_sigma(x, p.G, p.v, p.block, p.factors, p.ancestors)
    for scale in (0.5, 3.0)
        w = scale .* randn(rng, 11)
        M = randn(rng, 5, 5)
        M = M + M'
        grad = ones(11)
        SCSSD._add_gradient!(grad, sc, w, M)
        @test grad .- 1 ≈ ForwardDiff.gradient(x -> dot(M, reference(x)), w) rtol = 1e-9
        D = fill(NaN, 5, 5, 11)
        SCSSD._derivatives!(D, sc, w)
        @test reshape(D, 25, 11) ≈ ForwardDiff.jacobian(x -> vec(reference(x)), w) rtol =
            1e-9
    end

    # through the prior, with an ordinary kernel beside the structure
    F = randn(rng, 5, 5)
    prior = KernelPrior([Kernel(F; name="noise", factor=true)]; structure=sc)
    w = 0.5 .* randn(rng, 12)
    X = randn(rng, 5, 30)
    S = X * X' / 30
    μ = randn(rng, 12)
    root = randn(rng, 12, 12)
    Λ = root * root' + I
    function objective(x, S, n, μ, Λ)
        Σ = exp(x[1]) * F * F' + reference(x[2:end])
        return n / 2 * (logdet(Σ) + tr(Σ \ S)) + dot(x - μ, Λ * (x - μ)) / 2
    end
    grad = zeros(12)
    value = SCSSD._kernel_objective!(grad, prior, w, S, 7.0, μ, Λ)
    @test value ≈ objective(w, S, 7.0, μ, Λ)
    @test grad ≈ ForwardDiff.gradient(x -> objective(x, S, 7.0, μ, Λ), w) rtol = 1e-8
    # the information is the Hessian of the per-draw objective at S = Σ
    Σ = kernel_covariance(prior, w)
    expected = ForwardDiff.hessian(x -> objective(x, Σ, 1.0, zeros(12), zeros(12, 12)), w)
    @test SCSSD._kernel_information(prior, w) ≈ expected rtol = 1e-8
end

@testset "scale-correlation: calls split over threads" begin
    # With BLAS on one thread and Julia on several, the covariance, gradient,
    # information and Laplace covariance split their BLAS calls over Julia's
    # threads. A lowered split size makes this small problem split too; the
    # results must match the unsplit ones. (Without threads both paths are one.)
    rng = MersenneTwister(31)
    p = sc_problem(rng)
    prior = KernelPrior(
        [Kernel(randn(rng, 5, 5); name="noise", factor=true)]; structure=p.sc
    )
    w = 0.5 .* randn(rng, 12)
    X = randn(rng, 5, 30)
    S = X * X' / 30
    μ, Λ = zeros(12), Matrix(2.0I, 12, 12)
    function results()
        grad = zeros(12)
        value = SCSSD._kernel_objective!(grad, prior, w, S, 7.0, μ, Λ)
        info = SCSSD._kernel_information(prior, w)
        F = cholesky(Symmetric(7 .* info .+ Λ))
        return (;
            Σ=kernel_covariance(prior, w),
            value,
            grad,
            info,
            inverse=SCSSD._cholesky_inverse(F),
        )
    end
    blas, least = BLAS.get_num_threads(), SCSSD._SPLIT_LEAST[]
    try
        BLAS.set_num_threads(max(blas, 2))
        whole = results()
        BLAS.set_num_threads(1)
        SCSSD._SPLIT_LEAST[] = 1
        @test SCSSD._split_blas() == (Threads.nthreads() > 1)
        split = results()
        for key in keys(whole)
            @test split[key] ≈ whole[key] rtol = 1e-10
        end
        @test split.inverse ≈ inv(cholesky(Symmetric(7 .* whole.info .+ Λ))) rtol = 1e-10
    finally
        BLAS.set_num_threads(blas)
        SCSSD._SPLIT_LEAST[] = least
    end
end

@testset "scale-correlation: checks" begin
    # an untyped empty vector (`families=[]`, or one built from an empty list)
    # is no family
    let rng = MersenneTwister(5), G = randn(rng, 3, 4)
        sc = ScaleCorrelation(G, ones(4), [1, 1, 2, 2]; names=["a", "b"], families=[])
        @test isempty(sc.families) && length(SCSSD._structure_labels(sc)) == 2
        @test_throws ArgumentError ScaleCorrelation(
            G, ones(4), [1, 1, 2, 2]; names=["a", "b"], families=Any[1]
        )
    end
    rng = MersenneTwister(13)
    r, n = 4, 8
    G = randn(rng, r, n)
    v = ones(n)
    block = [1, 1, 1, 1, 2, 2, 2, 2]
    names = ["a", "b"]
    function cover(cols; m=1)
        F = zeros(n, m)
        F[cols, :] .= 1 .+ rand(rng, length(cols), m)
        return F
    end
    family(type, covers; kwargs...) = CorrelationFamily(
        [cover(c) for c in covers];
        type,
        names=["k$i" for i in eachindex(covers)],
        kwargs...,
    )
    build(; G=G, v=v, block=block, names=names, kwargs...) =
        ScaleCorrelation(G, v, block; names, kwargs...)
    message(f) =
        try
            f()
            ""
        catch err
            sprint(showerror, err)
        end

    @test build(; families=[family(:net, [1:4, 5:8]), family(:par, [1:2, 3:4, 7:8])]) isa
        ScaleCorrelation

    # the gain, the variances and the blocks
    @test_throws ArgumentError build(; G=fill(NaN, r, n))
    @test_throws ArgumentError build(; G=zeros(0, n))
    @test_throws DimensionMismatch build(; v=ones(n - 1))
    @test_throws ArgumentError build(; v=vcat(-1.0, ones(n - 1)))
    @test_throws ArgumentError build(; v=vcat(Inf, ones(n - 1)))
    @test_throws DimensionMismatch build(; block=block[1:(n - 1)])
    @test_throws ArgumentError build(; block=block .- 1)
    @test_throws ArgumentError build(; block=block .+ 1)
    @test_throws ArgumentError build(; names=["a", "b", "c"])           # c is empty
    @test occursin("c", message(() -> build(; names=["a", "b", "c"])))
    @test_throws ArgumentError build(; v=vcat(zeros(4), ones(4)))        # a has no trace
    @test occursin("positive trace", message(() -> build(; v=vcat(zeros(4), ones(4)))))
    @test_throws ArgumentError build(; G=hcat(G[:, 1:4], zeros(r, 4)))
    @test_throws ArgumentError build(; block=ones(Int, n), names=["a", "a"])
    @test_throws ArgumentError build(;
        block=[1, 1, 2, 2, 3, 3, 3, 3], names=["a", "b", "a"]
    )
    @test_throws DimensionMismatch build(; groups=["g"])
    @test_throws DimensionMismatch build(; units=["u", "v", "w"])
    @test build(; groups=["g", nothing]).groups == ["g", nothing]

    # a family on its own
    @test_throws ArgumentError CorrelationFamily(Matrix{Float64}[]; type=:net, names=[])
    @test_throws DimensionMismatch CorrelationFamily(
        [cover(1:4)]; type=:net, names=["x", "y"]
    )
    @test_throws ArgumentError family(:net, [1:4, 5:8]; names=["x", "x"])
    @test_throws DimensionMismatch family(:net, [1:4, 5:8]; groups=["g"])
    @test_throws DimensionMismatch family(:net, [1:4, 5:8]; units=["u"])
    @test_throws DimensionMismatch CorrelationFamily(
        [cover(1:4), ones(n + 1, 1)]; type=:net, names=["x", "y"]
    )
    @test_throws ArgumentError CorrelationFamily([fill(Inf, n, 1)]; type=:net, names=["x"])
    @test_throws ArgumentError CorrelationFamily([zeros(n, 2)]; type=:net, names=["x"])
    @test_throws ArgumentError family(:net, [1:4, 4:8])
    @test occursin("overlap", message(() -> family(:net, [1:4, 4:8])))
    # stored zeros of a sparse factor cover nothing
    stored = sparse([1, 2, 5], [1, 1, 1], [1.0, 2.0, 0.0], n, 1)
    @test CorrelationFamily([stored]; type=:net, names=["x"]).covered == [[1, 2]]

    # the families of a structure
    @test_throws DimensionMismatch build(;
        families=[CorrelationFamily([ones(n + 1, 1)]; type=:net, names=["x"])]
    )
    @test_throws ArgumentError build(; families=[family(:net, [1:4]), family(:net, [5:8])])
    @test_throws ArgumentError build(; families=[family(:scale, [1:4])])
    @test build(; type=:depth, families=[family(:scale, [1:4])]) isa ScaleCorrelation
    # nesting: across two kernels, and partly outside
    straddling = [family(:net, [1:4, 5:8]), family(:par, [3:6])]
    @test_throws ArgumentError build(; families=straddling)
    @test occursin("par:k1", message(() -> build(; families=straddling)))
    @test occursin("straddles net:k1, net:k2", message(() -> build(; families=straddling)))
    partial = [family(:net, [1:4]), family(:par, [3:6])]
    @test_throws ArgumentError build(; families=partial)
    @test occursin("partly inside net:k1", message(() -> build(; families=partial)))
    # with a family between that leaves the columns alone
    skipping = [family(:net, [1:4, 5:8]), family(:mid, [1:2]), family(:par, [3:6])]
    @test_throws ArgumentError build(; families=skipping)

    # through the prior: the size, and names shared with a kernel of the type
    sc = build(; families=[family(:net, [1:4, 5:8])])
    eye(d; kwargs...) = Kernel(Matrix(1.0I, d, d); kwargs...)
    @test_throws DimensionMismatch KernelPrior([eye(r + 1; name="I")]; structure=sc)
    @test_throws ArgumentError KernelPrior([eye(r; name="a", type=:scale)]; structure=sc)
    @test_throws ArgumentError KernelPrior([eye(r; name="I", type=:net)]; structure=sc)
end

@testset "scale-correlation: fit" begin
    rng = MersenneTwister(14)
    r, per, g = 10, 6, 6
    n = per * g
    G = randn(rng, r, n)
    v = 0.5 .+ rand(rng, n)
    block = repeat(1:g; inner=per)
    names = ["p$b" for b in 1:g]
    groups = ["n1", "n1", "n1", "n2", "n2", "n2"]
    net = map(1:2) do k
        F = zeros(n, 1)
        F[findall(b -> groups[b] == "n$k", block), :] .= 1
        F
    end
    par = map(1:g) do b
        F = zeros(n, 2)
        F[block .== b, :] .= randn(rng, per, 2)
        F
    end
    families = [
        CorrelationFamily(net; type=:net, names=["n1", "n2"]),
        CorrelationFamily(par; type=:par, names, groups),
    ]
    sc = ScaleCorrelation(G, v, block; names, groups, families)
    noise = [Kernel(Matrix(1.0I, r, r); name="I", type=:nuisance)]

    ν = 0.5
    θ = [1.0, 2.0, 0.5, 1.5, 0.8, 1.2]
    ρ_net = [0.3, 0.2]
    ρ_par = [0.3, 0.4, 0.2, 0.5, 0.3, 0.4]
    rest = 1 .- ρ_net[[1, 1, 1, 2, 2, 2]]
    truth = vcat(log(ν), log.(θ), sc_logit.(ρ_net), sc_logit.(ρ_par ./ rest))
    prior = KernelPrior(noise; structure=sc, prior_sd=5.0)
    Σ = kernel_covariance(prior, truth)
    draws = 200_000
    X = cholesky(Symmetric(Σ)).L * randn(rng, r, draws)
    S = X * X' / draws

    found = fit_kernel_weights!(prior, S, draws; max_iter=1000)
    @test found.converged
    table = kernel_weights(prior)
    @test [row.link for row in table] == vcat(fill(:log, 7), fill(:logit, 8))
    values = [row.value for row in table]
    @test values[1:7] == exp.(prior.weights[1:7])
    @test values[8:9] ≈ sc_sigmoid.(prior.weights[8:9])               # ρ, not exp(a)
    @test values[10:15] ≈
        (1 .- values[[8, 8, 8, 9, 9, 9]]) .* sc_sigmoid.(prior.weights[10:15])
    @test values[1:7] ≈ vcat(ν, θ) rtol = 0.1
    @test values[8:15] ≈ vcat(ρ_net, ρ_par) atol = 0.05
    @test all(row.sd > 0 for row in table)
    @test all(abs(truth[i] - table[i].weight) < 4 * table[i].sd for i in 1:15)
    @test kernel_covariance(prior) ≈ Σ rtol = 0.02

    # the fit is a stationary point of the stated objective
    μ, Λ = SCSSD._weight_prior(prior)
    @test μ[8:15] ≈ fill(sc_logit(0.1), 8)
    grad = zeros(15)
    value = SCSSD._kernel_objective!(grad, prior, prior.weights, S, Float64(draws), μ, Λ)
    @test value ≈ found.objective
    @test norm(grad) < 1e-5 * abs(value)

    # trees over the scales and a family, and the two correlated by unit
    X = cholesky(Symmetric(Σ)).L * randn(rng, r, 500)
    S = X * X' / 500
    for options in (
        (; levels=(scale=:hierarchical, par=:hierarchical)),
        (; levels=(scale=:nested, par=:nested)),
        (; levels=(scale=:hierarchical, par=:hierarchical), correlated=[(:scale, :par)]),
        (; levels=(scale=:nested, par=:nested), correlated=[(:scale, :par)]),
    )
        pooled = KernelPrior(noise; structure=sc, options...)
        found = fit_kernel_weights!(pooled, S, 500; rounds=200)
        @test found.converged
        @test found.rounds < 200
        @test all(isfinite, pooled.weights)
        @test isposdef(Symmetric(pooled.weights_cov))
        @test isposdef(Symmetric(kernel_covariance(pooled)))
        tree = pooled.trees[findfirst(t -> :scale in t.types, pooled.trees)]
        @test tree.levels ==
            (haskey(options, :levels) && options.levels.scale === :nested ? 2 : 1)
        @test length(tree.types) == (haskey(options, :correlated) ? 2 : 1)
        haskey(options, :correlated) && @test tree.idx == hcat(2:7, 10:15)
    end
end

#=== A shared kernel with a weight per unit ===================================#

# An exponential kernel on `m` ordered columns, positive definite.
sc_exp_kernel(m, length) = [exp(-abs(i - j) / length) for i in 1:m, j in 1:m]

# Σ_sc from the definition for a structure with shared families: `layout` lists
# the families in order, `(:factors, F, count)` (one factor per kernel) or
# `(:shared, blocks, count)` (`blocks` of `(cols, owner, M)`). Dense, any
# element type of `w`.
function sc_shared_sigma(w, G, v, block, layout, ancestors)
    g = maximum(block)
    n = length(block)
    θ = exp.(w[1:g])
    ρ = sc_rho(w[(g + 1):end], ancestors)
    C = Matrix{eltype(w)}(I, n, n)
    flat = 0
    for (kind, spec, count) in layout
        if kind === :factors
            C += sc_correlation(ρ[(flat + 1):(flat + count)], spec) - I
        else
            for (cols, owner, M) in spec
                d = sqrt.(ρ[flat .+ owner])
                C[cols, cols] .+= d .* M .* d' .- Diagonal(d .^ 2)
            end
        end
        flat += count
    end
    scale = sqrt.(θ[block] .* v)
    return G * (scale .* C .* scale') * G'
end

# 14 columns in 3 scale blocks. Networks 1:6 and 7:12; a shared kernel over two
# blocks of columns (1:9 and 10:14, so its unit 3 sits in network 2 and unit 5
# in none) with five units; two fine kernels, each inside one unit.
function sc_shared_problem(rng, order; r=5)
    n = 14
    G = randn(rng, r, n)
    v = 0.5 .+ rand(rng, n)
    block = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 3]
    factor(cols, m) = (F = zeros(n, m); F[cols, :] .= randn(rng, length(cols), m); F)
    nets = [factor(1:6, 1), factor(7:12, 1)]
    fine = [factor(1:3, 2), factor(7:9, 3)]
    blocks = [
        (collect(1:9), [1, 1, 1, 2, 2, 2, 3, 3, 3], sc_exp_kernel(9, 2.0)),
        (collect(10:14), [4, 4, 4, 5, 5], sc_exp_kernel(5, 1.5)),
    ]
    units = ["u$i" for i in 1:5]
    make = Dict(
        :coarse => () -> CorrelationFamily(nets; type=:coarse, names=["c1", "c2"]),
        :matern =>
            () -> SCSSD.SharedCorrelationFamily(
                blocks;
                type=:matern,
                names=units,
                groups=["a", "a", "b", "b", nothing],
                label="range 2",
            ),
        :fine => () -> CorrelationFamily(fine; type=:fine, names=["f1", "f3"]),
    )
    ancestors_in = Dict(
        (:coarse, :matern, :fine) =>
            [Int[], Int[], [1], [1], [2], [2], Int[], [1, 3], [2, 5]],
        (:matern, :fine) => [Int[], Int[], Int[], Int[], Int[], [1], [3]],
        (:matern,) => [Int[] for _ in 1:5],
        (:coarse, :matern) => [Int[], Int[], [1], [1], [2], [2], Int[]],
    )
    layouts = Dict(
        :coarse => (:factors, nets, 2),
        :matern => (:shared, blocks, 5),
        :fine => (:factors, fine, 2),
    )
    families = [make[name]() for name in order]
    layout = [layouts[name] for name in order]
    ancestors = ancestors_in[order]
    sc = ScaleCorrelation(G, v, block; names=["b1", "b2", "b3"], families)
    return (; G, v, block, layout, ancestors, families, sc, n_weights=3 + length(ancestors))
end

const SC_SHARED_ORDERS = [
    (:coarse, :matern, :fine), (:matern, :fine), (:matern,), (:coarse, :matern)
]

@testset "scale-correlation: a shared kernel" begin
    rng = MersenneTwister(21)
    for order in SC_SHARED_ORDERS
        p = sc_shared_problem(rng, order)
        sc = p.sc
        nw = p.n_weights
        @test sc.ancestors == p.ancestors
        @test SCSSD._structure_dim(sc) == 5
        labels = SCSSD._structure_labels(sc)
        @test length(labels) == nw
        shared = findfirst(f -> f isa SCSSD.SharedCorrelationFamily, p.families)
        fam = p.families[shared]
        @test fam.covered == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14]]
        start = 3 + sum(length(f.names) for f in p.families[1:(shared - 1)]; init=0)
        mine = labels[(start + 1):(start + 5)]
        @test [l.name for l in mine] == ["u1", "u2", "u3", "u4", "u5"]
        @test all(l.type === :matern && l.link === :logit && l.unit == l.name for l in mine)
        @test [l.group for l in mine] == ["a", "a", "b", "b", nothing]
        @test all(isnan, SCSSD._structure_traces(sc)[4:end])

        for scale in (0.5, 2.0, 50.0)
            w = scale .* randn(rng, nw)
            Σ = zeros(5, 5)
            SCSSD._add_covariance!(Σ, sc, w)
            expected = sc_shared_sigma(w, p.G, p.v, p.block, p.layout, p.ancestors)
            @test Σ ≈ expected rtol = 1e-10
            @test SCSSD._add_covariance!(copy(Σ), sc, w) ≈ 2 .* expected rtol = 1e-10
            @test issymmetric(Σ)
            @test eigmin(Symmetric(Σ)) > -1e-9 * norm(Σ)
        end

        # ∂Σ/∂w and tr(G ∂Σ/∂w) against the dense reference
        reference = x -> sc_shared_sigma(x, p.G, p.v, p.block, p.layout, p.ancestors)
        for scale in (0.5, 3.0)
            w = scale .* randn(rng, nw)
            Gm = randn(rng, 5, 5)
            Gm = Gm + Gm'
            grad = ones(nw)
            SCSSD._add_gradient!(grad, sc, w, Gm)
            @test grad .- 1 ≈ ForwardDiff.gradient(x -> dot(Gm, reference(x)), w) rtol =
                1e-9
            D = fill(NaN, 5, 5, nw)
            SCSSD._derivatives!(D, sc, w)
            @test reshape(D, 25, nw) ≈ ForwardDiff.jacobian(x -> vec(reference(x)), w) rtol =
                1e-9
        end
    end

    # the cached E and F follow the weights: the covariance, then the gradient
    # at the same weights (a hit), then at others (a miss)
    p = sc_shared_problem(rng, (:coarse, :matern, :fine))
    reference = x -> sc_shared_sigma(x, p.G, p.v, p.block, p.layout, p.ancestors)
    w1, w2 = randn(rng, p.n_weights), randn(rng, p.n_weights)
    Gm = Matrix(Symmetric(randn(rng, 5, 5)))
    SCSSD._add_covariance!(zeros(5, 5), p.sc, w1)
    for w in (w1, w2, w1)
        grad = zeros(p.n_weights)
        SCSSD._add_gradient!(grad, p.sc, w, Gm)
        @test grad ≈ ForwardDiff.gradient(x -> dot(Gm, reference(x)), w) rtol = 1e-9
    end

    # a single-precision structure
    single = ScaleCorrelation(
        Float32.(p.G), p.v, p.block; names=["b1", "b2", "b3"], families=p.families
    )
    @test single isa ScaleCorrelation{Float32}
    Σ = zeros(Float32, 5, 5)
    SCSSD._add_covariance!(Σ, single, Float32.(w1))
    @test Σ ≈ reference(w1) rtol = 1e-3
end

@testset "scale-correlation: a shared kernel is checked" begin
    rng = MersenneTwister(22)
    M = sc_exp_kernel(4, 2.0)
    ok = [(1:4, [1, 1, 2, 2], M)]
    shared(blocks; kwargs...) =
        SCSSD.SharedCorrelationFamily(blocks; type=:matern, names=["a", "b"], kwargs...)
    @test shared(ok).covered == [[1, 2], [3, 4]]
    @test shared(ok; units=["x", "y"]).units == ["x", "y"]
    @test_throws ArgumentError shared([])
    @test_throws ArgumentError shared([(1:4, [1, 1, 2, 2], M), (4:5, [1, 2], M[1:2, 1:2])])  # overlap
    @test_throws ArgumentError shared([(1:4, [1, 1, 1, 1], M)])             # unit b: no column
    @test_throws ArgumentError shared([(1:4, [1, 1, 2, 3], M)])             # owner out of range
    @test_throws ArgumentError shared([(1:4, [1, 1, 2, 2], 2 .* M)])        # not a correlation
    @test_throws ArgumentError shared([(1:4, [1, 1, 2, 2], M .+ [0 0 0 0.1; zeros(3, 4)])])
    @test_throws ArgumentError shared([(1:4, [1, 1, 2, 2], fill(NaN, 4, 4))])
    @test_throws ArgumentError shared([(0:3, [1, 1, 2, 2], M)])
    @test_throws ArgumentError shared([([1, 1, 2, 3], [1, 1, 2, 2], M)])    # repeated column
    @test_throws DimensionMismatch shared([(1:4, [1, 1, 2, 2], M[1:3, 1:3])])
    @test_throws DimensionMismatch shared([(1:4, [1, 2, 2], M)])
    @test_throws ArgumentError SCSSD.SharedCorrelationFamily(
        ok; type=:matern, names=["a", "a"]
    )

    G, v, block = randn(rng, 3, 6), ones(6), [1, 1, 1, 2, 2, 2]
    family = shared(ok)
    # a column beyond the gain
    @test_throws DimensionMismatch ScaleCorrelation(
        G[:, 1:3], v[1:3], [1, 1, 2]; names=["s", "t"], families=[family]
    )
    # families need distinct types, whichever kind
    other = CorrelationFamily([ones(6, 1)]; type=:matern, names=["x"])
    @test_throws ArgumentError ScaleCorrelation(
        G, v, block; names=["s", "t"], families=[family, other]
    )
    # units must nest: a later kernel across two units; an earlier kernel holding
    # both units, or cutting across them
    across = CorrelationFamily([[1.0 1 1 1 0 0]']; type=:wide, names=["w"])
    @test_throws ArgumentError ScaleCorrelation(
        G, v, block; names=["s", "t"], families=[family, across]
    )
    @test ScaleCorrelation(
        G, v, block; names=["s", "t"], families=[across, family]
    ).ancestors == [Int[], [1], [1]]                    # both units inside `across`
    straddle = CorrelationFamily([[0.0 1 1 0 0 0]']; type=:mid, names=["m"])
    @test_throws ArgumentError ScaleCorrelation(
        G, v, block; names=["s", "t"], families=[straddle, family]
    )
    inside = CorrelationFamily([[1.0 1 0 0 0 0]']; type=:inner, names=["i"])
    @test ScaleCorrelation(
        G, v, block; names=["s", "t"], families=[family, inside]
    ).ancestors == [Int[], Int[], [1]]
    @test_throws ArgumentError ScaleCorrelation(
        G, v, block; names=["s", "t"], families=[family, 3]
    )
end

@testset "scale-correlation: a shared kernel, fit" begin
    rng = MersenneTwister(23)
    r, per, g = 10, 6, 6
    n = per * g
    G = randn(rng, r, n)
    v = 0.5 .+ rand(rng, n)
    block = repeat(1:g; inner=per)
    names = ["p$b" for b in 1:g]
    groups = ["n1", "n1", "n1", "n2", "n2", "n2"]
    net = map(1:2) do k
        F = zeros(n, 1)
        F[findall(b -> groups[b] == "n$k", block), :] .= 1
        F
    end
    # one dense kernel over the whole cortex (two "hemispheres"), a weight per parcel
    half = n ÷ 2
    blocks = [
        (collect(1:half), block[1:half], sc_exp_kernel(half, 4.0)),
        (collect((half + 1):n), block[(half + 1):n], sc_exp_kernel(n - half, 6.0)),
    ]
    families = [
        CorrelationFamily(net; type=:net, names=["n1", "n2"]),
        SCSSD.SharedCorrelationFamily(blocks; type=:matern, names, groups),
    ]
    sc = ScaleCorrelation(G, v, block; names, groups, families)
    @test sc.ancestors == vcat([Int[], Int[]], [[1], [1], [1], [2], [2], [2]])
    noise = [Kernel(Matrix(1.0I, r, r); name="I", type=:nuisance)]

    ν, θ = 0.5, [1.0, 2.0, 0.5, 1.5, 0.8, 1.2]
    ρ_net = [0.2, 0.1]
    ρ_mat = [0.5, 0.6, 0.4, 0.6, 0.5, 0.7]
    rest = 1 .- ρ_net[[1, 1, 1, 2, 2, 2]]
    truth = vcat(log(ν), log.(θ), sc_logit.(ρ_net), sc_logit.(ρ_mat ./ rest))
    prior = KernelPrior(noise; structure=sc, prior_sd=5.0)
    Σ = kernel_covariance(prior, truth)
    draws = 200_000
    X = cholesky(Symmetric(Σ)).L * randn(rng, r, draws)
    S = X * X' / draws

    found = fit_kernel_weights!(prior, S, draws; max_iter=1000)
    @test found.converged
    table = kernel_weights(prior)
    @test [row.link for row in table] == vcat(fill(:log, 7), fill(:logit, 8))
    @test [row.type for row in table][10:15] == fill(:matern, 6)
    values = [row.value for row in table]
    @test values[1:7] ≈ vcat(ν, θ) rtol = 0.15
    @test values[10:15] ≈ ρ_mat atol = 0.1
    @test all(row.sd > 0 for row in table)
    @test all(abs(truth[i] - table[i].weight) < 4 * table[i].sd for i in 1:15)
    μ, Λ = SCSSD._weight_prior(prior)
    grad = zeros(15)
    value = SCSSD._kernel_objective!(grad, prior, prior.weights, S, Float64(draws), μ, Λ)
    @test value ≈ found.objective
    @test norm(grad) < 1e-5 * abs(value)

    # trees over the family, nested or pooled, alone or correlated with the
    # scales by unit; and unit kernels on its units
    X = cholesky(Symmetric(Σ)).L * randn(rng, r, 500)
    S = X * X' / 500
    for options in (
        (; levels=(scale=:hierarchical, matern=:hierarchical)),
        (; levels=(scale=:nested, matern=:nested)),
        (; levels=(scale=:nested, matern=:nested), correlated=[(:scale, :matern)]),
        (; levels=(scale=:hierarchical, matern=:flat)),
    )
        pooled = KernelPrior(noise; structure=sc, options...)
        found = fit_kernel_weights!(pooled, S, 500; rounds=200)
        @test found.converged
        @test all(isfinite, pooled.weights)
        @test isposdef(Symmetric(pooled.weights_cov))
        @test isposdef(Symmetric(kernel_covariance(pooled)))
        if haskey(options, :correlated)
            tree = pooled.trees[findfirst(t -> :matern in t.types, pooled.trees)]
            @test tree.idx == hcat(2:7, 10:15)
        end
    end
    near = reshape([exp(-abs(i - j) / 2.0) for i in 1:g, j in 1:g], 1, g, g)
    unit_structure = (;
        kernels=["near" => (names, near)],
        kernel_types=["near" => [:matern]],
        covariates=Pair{String,Tuple{Vector{String},Vector{Float64}}}[],
        covariate_types=Pair{String,Vector{Symbol}}[],
        kernel_sd=2.0,
    )
    grouped = KernelPrior(
        noise; structure=sc, levels=(scale=:hierarchical, matern=:nested), unit_structure
    )
    found = fit_kernel_weights!(grouped, S, 500; rounds=200)
    @test found.converged
    @test all(isfinite, grouped.weights)
end

# --------------------------------------------------------------------------
# A spectral family: an eigenmode kernel whose spectrum is fitted
# --------------------------------------------------------------------------

function sc_modes(rng, n_rows, N=7)
    lam = sort(exp.(range(log(1e3), log(2e5); length=N)) .* (0.8 .+ 0.4 .* rand(rng, N)))
    area = 4π * N / lam[end]  # Weyl's: ψ² ≈ 1/area, as the tail assumes
    return randn(rng, n_rows, N) ./ sqrt(area), lam
end

const SC_SPECTRAL_CASES = [
    (SCSSD.SurfaceSpectrum(:matern; alpha=2.0), [log(0.016)], [:log_range], nothing),
    (SCSSD.SurfaceSpectrum(:matern; alpha=3.0), [log(0.016)], Symbol[], :log_range),
    (SCSSD.SurfaceSpectrum(:field; omega=2π .* [5.0, 20.0, 35.0], weight=ones(3)),
     [log(0.02), log(60.0)], [:log_range, :log_gamma], nothing),
    (SCSSD.SurfaceSpectrum(:field; omega=2π .* [5.0, 20.0, 35.0], weight=ones(3)),
     [log(0.02), log(60.0)], [:log_gamma], :log_range),
    (SCSSD.SurfaceSpectrum(:field_matern; omega=2π .* [5.0, 20.0], weight=[1.0, 2.0]),
     [log(0.02), log(60.0)], [:log_gamma], :log_range),
]

@testset "scale-correlation: a spectral family" begin
    rng = MersenneTwister(31)
    r, n = 5, 14
    G = randn(rng, r, n)
    v = rand(rng, n) .+ 0.5
    block = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    for (spec, start, free, per_unit) in SC_SPECTRAL_CASES
        c1, c2 = collect(1:7), collect(8:14)
        (U1, l1), (U2, l2) = sc_modes(rng, 7), sc_modes(rng, 7)
        o1, o2 = [1, 1, 2, 2, 2, 3, 3], [3, 4, 4, 4, 5, 5, 5]
        fam = SCSSD.SpectralCorrelationFamily(
            [(c1, o1, U1, l1), (c2, o2, U2, l2)], spec; type=:surf,
            names=["u1", "u2", "u3", "u4", "u5"], groups=["a", "a", "b", "b", "b"],
            start, free, per_unit,
        )
        sc = ScaleCorrelation(G, v, block; names=["b1", "b2", "b3"], families=[fam])
        labels = SCSSD._structure_labels(sc)
        nw = length(labels)
        n_hyper = length(free) + (per_unit === nothing ? 0 : 5)
        @test nw == 3 + 5 + n_hyper
        @test all(l.link === :identity for l in labels[9:end])
        @test all(startswith(string(l.type), "surf_log_") for l in labels[9:end])
        w = vcat(0.3 .* randn(rng, 8), sc.hyper_start .+ 0.2 .* randn(rng, n_hyper))
        Σ = zeros(r, r)
        SCSSD._add_covariance!(Σ, sc, w)
        # the dense twin: M = D^(−½)(VV' + diag t)D^(−½) on each block
        θ, ρ, _, _ = SCSSD._scale_correlation_weights(sc, w)
        φ = SCSSD._hyper_weights(sc, w)
        C = Matrix(1.0I, n, n)
        for (part, b) in zip(sc.spectral, fam.blocks)
            P, uniform = SCSSD._part_params(part, φ)
            row = uniform ? ones(Int, length(b.owner)) : b.owner
            sq = reduce(vcat, (sqrt.(SCSSD._spectrum(spec, b.lam, P[k, :])[1])' for k in row))
            t = [SCSSD._spectrum_tail(spec, b.lam[end], P[k, :])[1] for k in row]
            V = b.U .* sq
            K = V * V' + Diagonal(t)
            M = K ./ sqrt.(diag(K) * diag(K)')
            @test diag(M) ≈ ones(length(t))
            dd = sqrt.(ρ[b.owner])
            C[b.cols, b.cols] .+= dd .* M .* dd' - Diagonal(dd .^ 2)
        end
        s = sqrt.(θ[block] .* v)
        @test Σ ≈ G * (s .* C .* s') * G' rtol = 1e-10
        # the gradient and the derivatives against finite differences
        A = randn(rng, r, r)
        Msym = (A + A') / 2
        cov(w) = (X = zeros(r, r); SCSSD._add_covariance!(X, sc, w); X)
        gr = zeros(nw)
        SCSSD._add_gradient!(gr, sc, w, Msym)
        e(i) = 1e-6 .* (1:nw .== i)
        fd = [(dot(Msym, cov(w .+ e(i))) - dot(Msym, cov(w .- e(i)))) / 2e-6 for i in 1:nw]
        @test gr ≈ fd rtol = 1e-6
        D = zeros(r, r, nw)
        SCSSD._derivatives!(D, sc, w)
        fdD = cat([(cov(w .+ e(i)) - cov(w .- e(i))) / 2e-6 for i in 1:nw]...; dims=3)
        @test D ≈ fdD rtol = 1e-6
        @test SCSSD._structure_values(sc, w)[(end - n_hyper + 1):end] == w[9:end]
    end
end

@testset "scale-correlation: spectral tails in closed form" begin
    for (kind, p) in ((:matern, [log(0.016)]), (:field, [log(0.02), log(80.0)]),
                      (:field_matern, [log(0.02), log(80.0)]))
        spec = SCSSD.SurfaceSpectrum(kind; omega=2π .* [3.0, 30.0], weight=[1.0, 1.0])
        lam_n = 2.0e5
        t, dt = SCSSD._spectrum_tail(spec, lam_n, p)
        # a trapezoid on a log grid, far enough out
        grid = exp.(range(log(lam_n), log(lam_n * 1e7); length=200_001))
        vals = SCSSD._spectrum(spec, grid, p)[1]
        numeric = sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2 .* diff(grid)) / (4π)
        @test t ≈ numeric rtol = 1e-4
        for j in eachindex(p)
            h = 1e-6 .* (eachindex(p) .== j)
            fd = (SCSSD._spectrum_tail(spec, lam_n, p .+ h)[1] -
                  SCSSD._spectrum_tail(spec, lam_n, p .- h)[1]) / 2e-6
            @test dt[j] ≈ fd rtol = 1e-6
        end
    end
    @test_throws ArgumentError SCSSD.SurfaceSpectrum(:matern; alpha=1.0)
    @test_throws ArgumentError SCSSD.SurfaceSpectrum(:other)
end

@testset "KernelPrior: spectral parameters as typed weights" begin
    rng = MersenneTwister(32)
    r, n = 5, 14
    G = randn(rng, r, n)
    v = rand(rng, n) .+ 0.5
    block = repeat(1:7; inner=2)
    (U, lam) = sc_modes(rng, n)
    fam = SCSSD.SpectralCorrelationFamily(
        [(collect(1:n), repeat(1:7; inner=2), U, lam)], SCSSD.SurfaceSpectrum(:matern);
        type=:surf, names=["u$i" for i in 1:7], groups=["a", "a", "a", "b", "b", "b", "b"],
        start=[log(0.016)], per_unit=:log_range,
    )
    sc = ScaleCorrelation(G, v, block; names=["b$i" for i in 1:7], families=[fam])
    X = randn(rng, r, 50)
    S = X * X' ./ 50
    prior = KernelPrior(Kernel[]; structure=sc, levels=Dict(:surf_log_range => :nested),
                        prior_mean=Dict(:surf_log_range => log(0.016)),
                        type_root_sd=Dict(:surf_log_range => 0.5),
                        type_sd_prior=Dict(:surf_log_range => (0.3, 10.0)))
    found = fit_kernel_weights!(prior, S, 50; rounds=3)
    @test all(isfinite, prior.weights)
    tree = only(t for t in prior.trees if t.types == [:surf_log_range])
    @test tree.root_sd == 0.5 && tree.sd_prior == (0.3, 10.0)
    # an :identity weight without its centre is an error
    bare = KernelPrior(Kernel[]; structure=sc, levels=Dict(:surf_log_range => :nested))
    @test_throws ArgumentError fit_kernel_weights!(bare, S, 50; rounds=1)
end

@testset "scale-correlation: the field's tail past the resonance" begin
    # γ so low that the fast nodes resonate far beyond the cutoff: positive tail
    spec = SCSSD.SurfaceSpectrum(:field; omega=2π .* [10.0, 30.0], weight=[1.0, 1.0])
    p = [log(0.016), log(0.01)]
    t, dt = SCSSD._spectrum_tail(spec, 2.3e5, p)
    r2 = 0.016^2 / 8
    expect = sum(0.5 * atan(2om / 0.01, 1 - om^2 / 0.01^2 + r2 * 2.3e5) / (2om / 0.01)
                 for om in 2π .* [10.0, 30.0]) / r2 / (4π)
    @test t > 0
    @test t ≈ expect rtol = 1e-10
end
