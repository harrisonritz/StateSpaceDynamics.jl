using Test, StateSpaceDynamics, LinearAlgebra, Random, Statistics, MAT
const KHSSD = StateSpaceDynamics

function kh_reference()
    return matread(
        joinpath(@__DIR__, "..", "test_data", "kernel_emissions", "reference.mat")
    )
end

# Labels of `n` weights of one type, with optional groups.
function kh_labels(type, n; groups=fill(nothing, n), link=:log)
    return [KHSSD.WeightLabel("u$i", type, groups[i], "u$i", link) for i in 1:n]
end

# A tree over `labels`, started as a fit starts it, at the centre `m0`.
function kh_tree(
    labels, levels, m0; correlated=(), sd_prior=(0.5, 4.0), root_sd=3.0, eta=2.0
)
    types = unique(l.type for l in labels)
    trees = KHSSD._build_trees(
        Float64,
        labels,
        Dict(t => levels for t in types),
        correlated;
        sd_prior,
        root_sd,
        lkj_eta=eta,
    )
    tree = only(trees)
    tree.root_centre .= m0
    tree.root .= m0
    tree.node_mean .= m0'
    return tree
end

@testset "weight tree: dynameg Hierarchy" begin
    ref = kh_reference()
    a, v = vec(ref["tree1_a"]), vec(ref["tree1_v"])
    groups = string.(Int.(vec(ref["tree1_groups"])))
    tree = kh_tree(kh_labels(:a, length(a); groups), 2, [ref["tree1_start"]])
    @test KHSSD._update_tree!(tree, a, Matrix(Diagonal(v)); sweeps=20_000, tol=1e-13)
    @test vec(tree.node_mean) ≈ vec(ref["tree1_means"]) rtol = 1e-9
    @test vec(tree.node_cov) ≈ vec(ref["tree1_variances"]) rtol = 1e-9
    @test [tree.sds[1], tree.psi[1]] ≈ vec(ref["tree1_sds"]) rtol = 1e-9
    @test tree.root[1] ≈ ref["tree1_root"] rtol = 1e-9
end

@testset "weight tree: dynameg JointHierarchy" begin
    ref = kh_reference()
    a, V = ref["joint1_a"], ref["joint1_V"]
    units = size(a, 1)
    groups = string.(Int.(vec(ref["joint1_groups"])))
    labels = vcat(kh_labels(:a, units; groups), kh_labels(:b, units; groups))
    tree = kh_tree(labels, 2, vec(ref["joint1_start"]); correlated=[(:a, :b)])
    @test tree.idx == hcat(1:units, (units + 1):(2units))
    w = vcat(a[:, 1], a[:, 2])
    Σw = zeros(2units, 2units)
    for u in 1:units
        rows = [u, units + u]
        Σw[rows, rows] .= V[u, :, :]
    end
    @test KHSSD._update_tree!(tree, w, Σw; sweeps=20_000, tol=1e-12)
    @test tree.node_mean ≈ ref["joint1_means"] rtol = 1e-6
    @test permutedims(tree.node_cov, (3, 1, 2)) ≈ ref["joint1_variances"] rtol = 1e-6
    @test tree.sds ≈ vec(ref["joint1_sds"]) rtol = 1e-6
    @test tree.corr ≈ ref["joint1_corr"] rtol = 1e-6
    @test tree.psi ≈ vec(ref["joint1_psi"]) rtol = 1e-6
    @test tree.root ≈ vec(ref["joint1_root"]) rtol = 1e-6
    # its Ω is at least as good an optimum of the Ω-step as the reference's
    L = cholesky(Symmetric(tree.corr)).L
    nll = KHSSD._omega_objective(
        ref["joint1_S_omega"],
        ref["joint1_n_omega"],
        ref["joint1_lkj_eta"],
        log.(tree.sds),
        L,
    )
    @test nll <= ref["joint1_omega_nll"] + 1e-8
end

@testset "weight tree: closed forms" begin
    rng = MersenneTwister(11)
    a = randn(rng, 9) .+ 2
    exact = zeros(9, 9)

    # One level, no hyperpriors, weights known exactly: the mean, and (the
    # mean's own uncertainty counted) the unbiased variance.
    tree = kh_tree(kh_labels(:a, 9), 1, [0.0]; sd_prior=nothing, root_sd=nothing)
    @test KHSSD._update_tree!(tree, a, exact; sweeps=10_000, tol=1e-13)
    @test tree.node_mean[1] ≈ mean(a)
    @test tree.sds[1] ≈ std(a) rtol = 1e-6
    @test KHSSD._tree_state(tree) == vcat(tree.root, tree.sds, vec(tree.corr), tree.psi)

    # The weights' own variances add to the spread; a root prior pulls the mean.
    spread = kh_tree(kh_labels(:a, 9), 1, [0.0]; sd_prior=nothing, root_sd=nothing)
    KHSSD._update_tree!(spread, a, Matrix(0.5I, 9, 9); sweeps=10_000, tol=1e-13)
    @test spread.sds[1]^2 ≈ tree.sds[1]^2 + 0.5 * 9 / 8 rtol = 1e-6
    pulled = kh_tree(kh_labels(:a, 9), 1, [0.0]; sd_prior=nothing, root_sd=0.1)
    KHSSD._update_tree!(pulled, a, exact)
    @test 0 < pulled.node_mean[1] < mean(a)

    # Two identical weights: without a hyperprior the level collapses to the
    # floor, with one it keeps the pseudo-observations' share.
    same = [1.0, 1.0]
    collapsed = kh_tree(kh_labels(:a, 2), 1, [1.0]; sd_prior=nothing, root_sd=nothing)
    KHSSD._update_tree!(collapsed, same, zeros(2, 2); sweeps=100_000, tol=1e-12)
    @test collapsed.sds[1] ≈ KHSSD._TREE_SD_FLOOR
    kept = kh_tree(kh_labels(:a, 2), 1, [1.0]; root_sd=nothing)
    KHSSD._update_tree!(kept, same, zeros(2, 2); sweeps=10_000, tol=1e-13)
    @test kept.sds[1] > 0.5 * sqrt(4 / 6)

    # A flat tree is not fitted.
    flat = kh_tree(kh_labels(:a, 3), 0, [0.0])
    @test KHSSD._update_tree!(flat, a[1:3], exact[1:3, 1:3])
    @test isempty(KHSSD._tree_state(flat)) && KHSSD._tree_logdensity(flat) == 0
end

@testset "weight tree: the prior it puts on the weights" begin
    rng = MersenneTwister(12)
    # Two correlated types of three units; :b has a fourth unit of its own.
    labels = vcat(kh_labels(:a, 3), kh_labels(:b, 4), kh_labels(:c, 2))
    trees = KHSSD._build_trees(
        Float64,
        labels,
        Dict(:a => 1, :b => 1),
        [(:a, :b)];
        sd_prior=(0.5, 4.0),
        root_sd=3.0,
        lkj_eta=2.0,
    )
    joint, flat = trees
    @test joint.types == [:a, :b] && joint.idx == [1 4; 2 5; 3 6]
    @test joint.leftover == [Int[], [7]]
    @test flat.types == [:c] && flat.levels == 0
    joint.node_mean .= [0.3 -0.7]
    joint.sds .= [0.5, 2.0]
    joint.corr .= [1.0 0.6; 0.6 1.0]
    flat.centre .= [1.0, 2.0]
    flat.sd .= [0.5, 4.0]
    μ, Λ = KHSSD._weight_prior!(zeros(9), zeros(9, 9), trees)
    Ω = [0.25 0.6; 0.6 4.0]
    @test μ == [0.3, 0.3, 0.3, -0.7, -0.7, -0.7, -0.7, 1.0, 2.0]
    @test Λ[[2, 5], [2, 5]] ≈ inv(Ω)
    @test Λ[7, 7] ≈ 1 / 4.0 && Λ[8, 8] ≈ 4.0 && Λ[9, 9] ≈ 1 / 16
    @test Λ[1, 2] == 0 && Λ[1, 5] == 0 && issymmetric(Λ)

    # A stronger LKJ prior shrinks the fitted correlation.
    units = 12
    base = randn(rng, units)
    w = vcat(base .+ 0.3 .* randn(rng, units), base .+ 0.3 .* randn(rng, units))
    pair = vcat(kh_labels(:a, units), kh_labels(:b, units))
    fitted = map((1.0, 2.0, 50.0)) do eta
        tree = kh_tree(pair, 1, [0.0, 0.0]; correlated=[(:a, :b)], eta)
        KHSSD._update_tree!(tree, w, zeros(2units, 2units); sweeps=2000, tol=1e-10)
        @test isfinite(KHSSD._tree_logdensity(tree))
        tree.corr[1, 2]
    end
    @test fitted[1] > fitted[2] > fitted[3] > 0
    @test fitted[1] > 0.7
end

@testset "kernel weights: fitted trees" begin
    rng = MersenneTwister(13)
    # One rank-one kernel per dimension: each weight is then estimated on its
    # own, from `n` draws, with an error of about sqrt(2 / n) in the log.
    r = 40
    axis(i; kwargs...) = Kernel(reshape(Float64.(1:r .== i), r, 1); factor=true, kwargs...)
    moment(θ, n) = Matrix(Diagonal(θ .* [sum(abs2, randn(rng, n)) / n for _ in 1:r]))

    # :hierarchical finds the spread of the weights and shrinks towards the mean.
    truth = -1.0 .+ 0.7 .* randn(rng, r)
    n = 6
    S = moment(exp.(truth), n)
    pooled = KernelPrior([axis(i; name="k$i") for i in 1:r]; levels=:hierarchical)
    found = fit_kernel_weights!(pooled, S, n; rounds=200)
    @test found.converged && found.rounds > 1
    tree = only(pooled.trees)
    @test tree.node_mean[1] ≈ -1.0 atol = 0.4
    @test tree.sds[1] ≈ 0.7 atol = 0.3
    wide = KernelPrior([axis(i; name="k$i") for i in 1:r]; prior_sd=10.0)
    fit_kernel_weights!(wide, S, n)
    @test sum(abs2, pooled.weights .- truth) < sum(abs2, wide.weights .- truth)
    @test all(row.sd > 0 for row in kernel_weights(pooled))
    @test isfinite(KHSSD._kernel_logprior(pooled))

    # With the trees held, a round is the MAP alone and the state does not move.
    state = KHSSD._tree_state(tree)
    fit_kernel_weights!(pooled, S, n; update_trees=false)
    @test KHSSD._tree_state(tree) == state

    # :nested finds each group's mean.
    groups = [i <= r ÷ 2 ? "low" : "high" for i in 1:r]
    centres = [g == "low" ? -2.0 : 1.0 for g in groups]
    truth = centres .+ 0.3 .* randn(rng, r)
    S = moment(exp.(truth), 20)
    nested = KernelPrior(
        [axis(i; name="k$i", group=groups[i]) for i in 1:r]; levels=:nested
    )
    @test fit_kernel_weights!(nested, S, 20; rounds=200).converged
    tree = only(nested.trees)
    @test tree.node_names == ["low", "high"]
    @test vec(tree.node_mean) ≈ [-2.0, 1.0] atol = 0.4
    @test tree.sds[1] < 0.8 && tree.psi[1] > 0.5
    @test -2.0 < tree.root[1] < 1.0

    # Correlated types: the weights of a unit covary across the two types.
    half = r ÷ 2
    shared = 0.8 .* randn(rng, half)
    truth = vcat(shared .+ 0.3 .* randn(rng, half), shared .+ 0.3 .* randn(rng, half))
    S = moment(exp.(truth), 30)
    kernels = vcat(
        [axis(i; name="u$i", type=:a) for i in 1:half],
        [axis(half + i; name="u$i", type=:b) for i in 1:half],
    )
    joint = KernelPrior(kernels; levels=:hierarchical, correlated=[(:a, :b)])
    @test fit_kernel_weights!(joint, S, 30; rounds=200).converged
    @test only(joint.trees).corr[1, 2] > 0.4
    apart = KernelPrior(kernels; levels=:hierarchical)
    fit_kernel_weights!(apart, S, 30; rounds=200)
    @test length(apart.trees) == 2
    @test sum(abs2, joint.weights .- truth) < sum(abs2, apart.weights .- truth)
end
