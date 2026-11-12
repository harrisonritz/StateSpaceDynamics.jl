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

@testset "scale-correlation: checks" begin
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
