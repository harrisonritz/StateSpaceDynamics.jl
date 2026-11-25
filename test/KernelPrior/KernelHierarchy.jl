using Test, StateSpaceDynamics, LinearAlgebra, Random, Statistics, MAT, Optim
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

@testset "weight tree: Ω-step at a perfect correlation" begin
    # Units whose weights move exactly together put the Ω-step's optimum on the
    # boundary |ρ| = 1; it must stop there rather than fail on a singular factor.
    v = [1.0, 2.0, -0.5]
    S = 50 .* (v * v') + 1e-12 * I
    σ, R = KHSSD._fit_omega(S, 50.0, 2.0, ones(3), Matrix(1.0I, 3, 3))
    @test all(isfinite, σ) && all(isfinite, R)
    @test minimum(eigvals(Symmetric(R))) > -1e-8
    @test R[1, 2] > 0.99 && R[1, 3] < -0.99
    ratio = σ ./ abs.(v)                # the SDs keep the units' proportions
    @test maximum(ratio) / minimum(ratio) < 1.01
end

# ----------------------------------------------------------------------------
# Unit kernels and covariates
# ----------------------------------------------------------------------------

# A grid of squared-exponential kernels (unit diagonal) over positions `x`.
function kh_grid(x, lengths)
    K = zeros(length(lengths), length(x), length(x))
    for (i, ℓ) in enumerate(lengths)
        K[i, :, :] .= exp.(-(x .- x') .^ 2 ./ (2ℓ^2))
    end
    return K
end

# A tree over `labels` with the unit structure `spec`, started at `m0`.
function kh_unit_tree(labels, levels, m0, spec; correlated=(), kwargs...)
    tree = kh_tree(labels, levels, m0; correlated, kwargs...)
    KHSSD._attach_unit_structure!([tree], labels, spec)
    return tree
end

# Draw the `U × d` weights of a tree from the model: Z μ + x βᵀ + (K_w ⊗ Ω)^½ e.
function kh_draw(rng, Kw, Ω, M)
    U, d = size(M)
    E = cholesky(Symmetric(Kw)).L * randn(rng, U, d) * cholesky(Symmetric(Ω)).U
    return M .+ E
end

@testset "unit structure: none leaves the trees as they were" begin
    rng = MersenneTwister(21)
    labels = kh_labels(:a, 6)
    trees = KHSSD._build_trees(
        Float64, labels, Dict(:a => 1), (); sd_prior=(0.5, 4.0), root_sd=3.0, lkj_eta=2.0
    )
    KHSSD._attach_unit_structure!(trees, labels, (;))
    @test only(trees).unit_structure === nothing
    r = 6
    kernels = [
        Kernel(reshape(Float64.(1:r .== i), r, 1); factor=true, name="u$i") for i in 1:r
    ]
    S = Matrix(Diagonal(exp.(randn(rng, r))))
    plain = KernelPrior(kernels; levels=:hierarchical)
    empty = KernelPrior(kernels; levels=:hierarchical, unit_structure=(;))
    fit_kernel_weights!(plain, S, 10; rounds=20)
    fit_kernel_weights!(empty, S, 10; rounds=20)
    @test plain.weights == empty.weights && plain.weights_cov == empty.weights_cov
    @test KHSSD._tree_state(only(plain.trees)) == KHSSD._tree_state(only(empty.trees))

    # A structure without kernels or covariates (K_w = I, B = 0) is the plain model.
    for (levels, groups) in ((1, fill(nothing, 30)), (2, [string(i % 3) for i in 1:30]))
        a = randn(rng, 30) .+ [i % 3 for i in 1:30]
        V = Matrix(Diagonal(0.1 .+ rand(rng, 30)))
        plain = kh_tree(kh_labels(:a, 30; groups), levels, [0.0])
        structured = kh_tree(kh_labels(:a, 30; groups), levels, [0.0])
        structured.unit_structure = KHSSD.UnitStructure{Float64}(
            String[],
            Vector{Matrix{Float64}}[],
            String[],
            zeros(30, 0),
            zeros(Bool, 0, 1),
            2.0,
            Float64[],
            Int[],
            zeros(0, 1),
        )
        @test KHSSD._update_tree!(plain, a, V; sweeps=5000, tol=1e-12)
        @test KHSSD._update_tree!(structured, a, V; sweeps=5000, tol=1e-12)
        @test KHSSD._tree_state(structured) ≈ KHSSD._tree_state(plain) rtol = 1e-8
        @test structured.node_mean ≈ plain.node_mean rtol = 1e-8
        @test structured.node_cov ≈ plain.node_cov rtol = 1e-8
        μp, Λp = KHSSD._weight_prior!(zeros(30), zeros(30, 30), [plain])
        μs, Λs = KHSSD._weight_prior!(zeros(30), zeros(30, 30), [structured])
        @test μs ≈ μp && Λs ≈ Λp
    end
end

@testset "unit structure: alignment and checks" begin
    labels = vcat(kh_labels(:a, 4), [KHSSD.WeightLabel("none", :a, nothing, "none", :log)])
    K = kh_grid([0.0, 1.0, 2.0], [1.0, 2.0])
    spec = (;
        kernels=Dict("near" => (["u3", "u1", "u2"], K)),
        kernel_types=Dict("near" => (:a,)),
        covariates=Dict("depth" => (["u1", "u2", "u3", "u4"], [1.0, 2.0, 3.0, 6.0])),
        covariate_types=Dict("depth" => [:a]),
    )
    tree = kh_unit_tree(labels, 1, [0.0], spec)
    s = tree.unit_structure
    @test s.kernel_names == ["near"] && s.covariate_names == ["depth"]
    @test length(s.kernels[1]) == 2 && s.width == [2] && s.z == [0.0]
    # u1, u2, u3 at 1, 2, 0; u4 and "none" unknown to the kernel
    @test s.kernels[1][2][1:3, 1:3] ≈ K[2, [2, 3, 1], [2, 3, 1]]
    @test s.kernels[1][1][4:5, :] == [0 0 0 1 0; 0 0 0 0 1]
    # centred and scaled over the units with a value; "none" gets zero
    x = [1.0, 2.0, 3.0, 6.0]
    @test s.X[1:4] ≈ (x .- mean(x)) ./ std(x; corrected=false)
    @test s.X[5] == 0
    @test s.flags == reshape([true], 1, 1) && s.beta == zeros(1, 1)

    @test_throws ArgumentError kh_unit_tree(
        labels,
        1,
        [0.0],
        (;
            covariates=Dict("depth" => (["u1", "u2"], [1.0, 2.0])),
            covariate_types=Dict("depth" => [:a]),
        ),
    )
    @test_throws ArgumentError kh_unit_tree(
        labels, 1, [0.0], (; kernel_types=Dict("far" => [:a]))
    )
    @test_throws ArgumentError kh_unit_tree(labels, 1, [0.0], (; widths=1))
    flat = kh_tree(labels, 0, [0.0])
    @test_throws ArgumentError KHSSD._attach_unit_structure!([flat], labels, spec)
    bad = (;
        kernels=Dict("near" => (["u1", "u2"], [1.0 0.5; 0.4 1.0])),
        kernel_types=Dict("near" => [:a]),
    )
    @test_throws ArgumentError kh_unit_tree(labels, 1, [0.0], bad)

    # The types of a correlated set share their kernels.
    pair = vcat(kh_labels(:a, 3), kh_labels(:b, 3))
    one_sided = (;
        kernels=Dict("near" => (["u1", "u2", "u3"], K[1, :, :])),
        kernel_types=Dict("near" => [:a]),
    )
    @test_throws ArgumentError kh_unit_tree(
        pair, 1, [0.0, 0.0], one_sided; correlated=[(:a, :b)]
    )
    shared = (; kernels=one_sided.kernels, kernel_types=Dict("near" => [:a, :b]))
    joint = kh_unit_tree(pair, 1, [0.0, 0.0], shared; correlated=[(:a, :b)])
    @test joint.unit_structure.kernel_names == ["near"]

    # The w-step gradient.
    rng = MersenneTwister(22)
    s = KHSSD.UnitStructure{Float64}(
        ["a", "b"],
        [
            [kh_grid(randn(rng, 7), [1.0])[1, :, :]],
            [kh_grid(randn(rng, 7), [0.5])[1, :, :]],
        ],
        String[],
        zeros(7, 0),
        zeros(Bool, 0, 2),
        2.0,
        [0.3, -0.4],
        [1, 1],
        zeros(0, 2),
    )
    A = randn(rng, 7, 7)
    F = A * A' + I
    G = zeros(2)
    KHSSD._unit_objective!(G, s, [0.3, -0.4], F, 2)
    h = 1e-6
    numeric = map(1:2) do k
        e = h .* (1:2 .== k)
        (
            KHSSD._unit_objective!(nothing, s, [0.3, -0.4] .+ e, F, 2) -
            KHSSD._unit_objective!(nothing, s, [0.3, -0.4] .- e, F, 2)
        ) / 2h
    end
    @test G ≈ numeric rtol = 1e-5
    # ... and with a Student-t tree's nugget (1/E[λ_u] ≠ 1, so ∂K_w/∂w₀ ≠ I),
    # with and without the profiled scale
    s.nugget .= exp.(randn(rng, 7))
    for profile in (nothing, (14.0, 0.7))
        KHSSD._unit_objective!(G, s, [0.3, -0.4], F, 2; profile)
        numeric = map(1:2) do k
            e = h .* (1:2 .== k)
            (
                KHSSD._unit_objective!(nothing, s, [0.3, -0.4] .+ e, F, 2; profile) -
                KHSSD._unit_objective!(nothing, s, [0.3, -0.4] .- e, F, 2; profile)
            ) / 2h
        end
        @test G ≈ numeric rtol = 1e-5
    end
end

@testset "unit structure: the prior it puts on the weights" begin
    rng = MersenneTwister(23)
    U = 5
    labels = vcat(kh_labels(:a, U), kh_labels(:b, U + 1))
    K = kh_grid(randn(rng, U), [0.7])
    spec = (;
        kernels=[("near" => (["u$i" for i in 1:U], K))],
        kernel_types=["near" => (:a, :b)],
        covariates=["depth" => (["u$i" for i in 1:U], randn(rng, U))],
        covariate_types=["depth" => (:b,)],
    )
    trees = KHSSD._build_trees(
        Float64,
        labels,
        Dict(:a => 1, :b => 1),
        [(:a, :b)];
        sd_prior=(0.5, 4.0),
        root_sd=3.0,
        lkj_eta=2.0,
    )
    KHSSD._attach_unit_structure!(trees, labels, spec)
    tree = only(trees)
    s = tree.unit_structure
    @test s.flags == [false true]
    tree.root_centre .= 0
    tree.node_mean .= [0.3 -0.7]
    tree.sds .= [0.5, 2.0]
    tree.corr .= [1.0 0.6; 0.6 1.0]
    s.z .= [0.8]
    s.beta .= [0.0 0.4]
    nw = length(labels)
    μ, Λ = KHSSD._weight_prior!(zeros(nw), zeros(nw, nw), trees)
    w = KHSSD._unit_weights([0.8])
    Kw = w[1] * I + w[2] * K[1, :, :]
    Ω = KHSSD._tree_omega(tree)
    order = vec(permutedims(tree.idx))
    @test Λ[order, order] ≈ kron(inv(Kw), inv(Ω))
    @test μ[tree.idx[:, 1]] ≈ fill(0.3, U)
    @test μ[tree.idx[:, 2]] ≈ -0.7 .+ 0.4 .* s.X[:, 1]
    # the leftover unit of :b keeps its marginal prior
    @test tree.leftover == [Int[], [nw]]
    @test Λ[nw, nw] ≈ 1 / 4.0 && μ[nw] ≈ -0.7
    @test count(!iszero, Λ[nw, :]) == 1
    s.z .= 0
    base = KHSSD._tree_logdensity(tree)
    s.z .= [0.8]
    @test KHSSD._tree_logdensity(tree) ≈ base - 0.8^2 / 8
end

@testset "unit structure: the update recovers a kernel and a slope" begin
    rng = MersenneTwister(24)
    U = 300
    positions = sort(rand(rng, U))
    lengths = [0.01, 0.03, 0.1, 0.3]
    K = kh_grid(positions, lengths)
    x = randn(rng, U)
    x = (x .- mean(x)) ./ std(x; corrected=false)
    groups = [string(1 + i % 3) for i in 1:U]
    names = ["u$i" for i in 1:U]
    spec = (;
        kernels=Dict("near" => (names, K)),
        kernel_types=Dict("near" => [:a]),
        covariates=Dict("depth" => (names, 3 .* x .+ 1)),
        covariate_types=Dict("depth" => [:a]),
    )
    z, β, σ = 1.5, 0.8, 0.5
    w = KHSSD._unit_weights([z])
    Kw = w[1] * I + w[2] * K[2, :, :]   # the grid starts at its third width
    centres = Dict("1" => -1.0, "2" => 0.0, "3" => 1.0)
    M = reshape([centres[g] for g in groups] .+ β .* x, U, 1)
    η = vec(kh_draw(rng, Kw, fill(σ^2, 1, 1), M))
    tree = kh_unit_tree(kh_labels(:a, U; groups), 2, [0.0], spec)
    converged = KHSSD._update_tree!(tree, η, Matrix(1e-4I, U, U))
    @test converged
    s = tree.unit_structure
    @test s.width == [2]
    @test KHSSD._unit_weights(s.z)[2] ≈ w[2] atol = 0.1
    @test s.beta[1] ≈ β atol = 0.15
    @test tree.sds[1] ≈ σ rtol = 0.2
    # the field's own mean shifts every node alike
    found = vec(tree.node_mean)
    @test found .- mean(found) ≈ [centres[n] for n in tree.node_names] atol = 0.25

    # Without the kernel the units look independent, with a smaller spread.
    plain = kh_unit_tree(
        kh_labels(:a, U; groups),
        2,
        [0.0],
        (; covariates=spec.covariates, covariate_types=spec.covariate_types),
    )
    @test KHSSD._update_tree!(plain, η, Matrix(1e-4I, U, U))
    @test plain.unit_structure.beta[1] ≈ β atol = 0.2
end

@testset "unit structure: a correlated set with a shared kernel" begin
    rng = MersenneTwister(25)
    U = 200
    positions = sort(rand(rng, U))
    K = kh_grid(positions, [0.05])
    x = randn(rng, U)
    x = (x .- mean(x)) ./ std(x; corrected=false)
    names = ["u$i" for i in 1:U]
    spec = (;
        kernels=Dict("near" => (names, K)),
        kernel_types=Dict("near" => [:a, :b]),
        covariates=Dict("depth" => (names, x)),
        covariate_types=Dict("depth" => [:a]),
    )
    z, β = 1.0, [0.6, 0.0]
    sds, ρ = [0.5, 0.8], 0.5
    Ω = sds .* [1 ρ; ρ 1] .* sds'
    w = KHSSD._unit_weights([z])
    Kw = w[1] * I + w[2] * K[1, :, :]
    η = kh_draw(rng, Kw, Ω, [0.5 -1.0] .+ x * β')
    labels = vcat(kh_labels(:a, U), kh_labels(:b, U))
    tree = kh_unit_tree(labels, 1, [0.0, 0.0], spec; correlated=[(:a, :b)])
    V = Matrix(1e-4I, 2U, 2U)
    @test KHSSD._update_tree!(tree, vec(η), V; sweeps=500)
    s = tree.unit_structure
    @test s.flags == reshape([true, false], 1, 2)
    @test s.beta[2] == 0
    @test s.beta[1] ≈ β[1] atol = 0.15
    @test KHSSD._unit_weights(s.z)[2] ≈ w[2] atol = 0.15
    @test tree.sds ≈ sds rtol = 0.25
    @test tree.corr[1, 2] ≈ ρ atol = 0.2
    @test vec(tree.node_mean) ≈ [0.5, -1.0] atol = 0.4

    # Through KernelPrior: the trees get the structure, and a fit runs.
    U = 30
    names = ["u$i" for i in 1:U]
    spec = (;
        kernels=Dict("near" => (names, kh_grid(positions[1:U], [0.05, 0.2]))),
        kernel_types=Dict("near" => [:a, :b]),
        covariates=Dict("depth" => (names, x[1:U])),
        covariate_types=Dict("depth" => [:a]),
    )
    r = 2U
    axis(i; kwargs...) = Kernel(reshape(Float64.(1:r .== i), r, 1); factor=true, kwargs...)
    kernels = vcat(
        [axis(i; name="u$i", type=:a) for i in 1:U],
        [axis(U + i; name="u$i", type=:b) for i in 1:U],
    )
    prior = KernelPrior(
        kernels; levels=:hierarchical, correlated=[(:a, :b)], unit_structure=spec
    )
    @test only(prior.trees).unit_structure.kernel_names == ["near"]
    truth = vec(η[1:U, :])
    S = Matrix(Diagonal(exp.(truth) .* [sum(abs2, randn(rng, 50)) / 50 for _ in 1:r]))
    fit = fit_kernel_weights!(prior, S, 50; rounds=5)
    @test fit.rounds >= 1 && isfinite(KHSSD._kernel_logprior(prior))
    @test all(isfinite, prior.weights) && all(isfinite, only(prior.trees).unit_structure.z)
    @test_throws ArgumentError KernelPrior(kernels; levels=:flat, unit_structure=spec)
    @test_throws ArgumentError KernelPrior(
        kernels;
        levels=:hierarchical,
        correlated=[(:a, :b)],
        unit_structure=(; kernels=spec.kernels, kernel_types=Dict("near" => [:a])),
    )
end

@testset "Student-t trees: an outlying unit is shrunk less" begin
    rng = MersenneTwister(91)
    n = 30
    # weights about a common centre, one far from it
    w = 0.3 .* randn(rng, n)
    w[7] = 3.0
    Σw = Matrix(0.01I, n, n)
    labels = [KHSSD.WeightLabel("u$i", :scale, nothing, "u$i", :log) for i in 1:n]
    function tree(df)
        t = only(
            KHSSD._build_trees(
                Float64, labels, Dict(:scale => 1), ();
                sd_prior=nothing, root_sd=nothing, lkj_eta=2.0,
                type_df=Dict(:scale => df),
            ),
        )
        t.node_mean .= 0.0
        t.root .= 0.0
        t.root_centre .= 0.0
        KHSSD._update_tree!(t, w, Σw)
        return t
    end
    gauss, t3 = tree(Inf), tree(3.0)
    @test all(==(1.0), gauss.lambda)
    @test argmin(t3.lambda) == 7 && t3.lambda[7] < 0.2
    # the outlier widens the Gaussian tree; the t tree's scale follows the rest
    @test t3.sds[1] < gauss.sds[1]
    @test abs(t3.sds[1] - std(w[setdiff(1:n, 7)])) < 0.1
    # the prior precision of each weight is E[λ_u] / σ²
    μ, Λ = KHSSD._weight_prior!(zeros(n), zeros(n, n), [t3])
    @test diag(Λ) ≈ t3.lambda ./ t3.sds[1]^2
    @test isfinite(KHSSD._tree_logdensity(t3))
    # df = Inf is the Gaussian tree exactly
    g2 = tree(Inf)
    @test g2.sds ≈ gauss.sds && g2.node_mean ≈ gauss.node_mean
    @test_throws ArgumentError KHSSD._build_trees(
        Float64, labels, Dict(:scale => 0), ();
        sd_prior=nothing, root_sd=nothing, lkj_eta=2.0, type_df=Dict(:scale => 3.0),
    )
end

@testset "Student-t trees with unit kernels: the nugget follows λ" begin
    rng = MersenneTwister(92)
    n, d = 24, 1
    D = [abs(i - j) for i in 1:n, j in 1:n]
    K = exp.(-D ./ 3)
    smooth = cholesky(Symmetric(K + 1e-9I)).L * randn(rng, n)
    w = 0.5 .* smooth .+ 0.1 .* randn(rng, n)
    w[12] += 3.0
    labels = [KHSSD.WeightLabel("u$i", :scale, nothing, "u$i", :log) for i in 1:n]
    names = ["u$i" for i in 1:n]
    spec = (; kernels=["near" => (names, reshape(K, 1, n, n))], kernel_types=["near" => [:scale]])
    fits = map((Inf, 3.0)) do df
        trees = KHSSD._build_trees(
            Float64, labels, Dict(:scale => 1), ();
            sd_prior=(0.5, 4.0), root_sd=3.0, lkj_eta=2.0, type_df=Dict(:scale => df),
        )
        t = only(trees)
        KHSSD._attach_unit_structure!(trees, labels, spec)
        t.node_mean .= 0.0
        t.root .= 0.0
        t.root_centre .= 0.0
        KHSSD._update_tree!(t, w, Matrix(0.01I, n, n))
        t
    end
    gauss, t3 = fits
    @test all(==(1.0), gauss.unit_structure.nugget)
    @test argmin(t3.lambda) == 12
    @test t3.unit_structure.nugget ≈ 1 ./ t3.lambda
    μ, Λ = KHSSD._weight_prior!(zeros(n), zeros(n, n), [t3])
    @test isposdef(Symmetric(Λ))
end

@testset "Student-t trees with unit kernels: the w-step is taken" begin
    #=
    With a nugget ≠ 1 a gradient that took ∂K_w/∂w₀ = I sent every L-BFGS run
    of the w-step to its 200-iteration cap, above its start, and the run was
    discarded: the logits never left their first-sweep values. The step must
    now be taken, converge well inside the cap, and leave a stationary point.
    =#
    rng = MersenneTwister(93)
    U, d = 50, 20
    x = sort(rand(rng, U))
    near, broad = kh_grid(x, [0.05])[1, :, :], kh_grid(x, [0.3])[1, :, :]
    nugget = exp.(2 .* randn(rng, U))
    w = KHSSD._unit_weights([1.0, -0.5])
    Kw = w[1] * Diagonal(nugget) + w[2] * near + w[3] * broad
    E = cholesky(Symmetric(Kw)).L * randn(rng, U, d)
    F = E * E'
    for profile in (nothing, (d * U + 4.0, 0.7))
        s = KHSSD.UnitStructure{Float64}(
            ["near", "broad"],
            [[near], [broad]],
            String[],
            zeros(U, 0),
            zeros(Bool, 0, d),
            2.0,
            zeros(2),
            [1, 1],
            zeros(0, d),
            copy(nugget),
        )
        f = z -> KHSSD._unit_objective!(nothing, s, z, F, d; profile)
        g! = (G, z) -> (KHSSD._unit_objective!(G, s, z, F, d; profile); G)
        z0 = copy(s.z)
        run = optimize(
            f, g!, z0, LBFGS(), Optim.Options(; iterations=200, g_tol=1e-9, f_reltol=1e-14)
        )
        @test Optim.converged(run) && Optim.iterations(run) < 50
        KHSSD._fit_unit_weights!(s, F, d; profile)
        @test (s.steps, s.capped, s.rejected) == (1, 0, 0)
        @test s.z != z0 && f(s.z) < f(z0)
        # stationary in the objective itself (central differences), not only
        # in the analytic gradient the step followed
        h = 1e-5
        bump(k) = h .* (1:2 .== k)
        slope = [(f(s.z .+ bump(k)) - f(s.z .- bump(k))) / 2h for k in 1:2]
        @test norm(slope) < 1e-4
        @test norm(g!(zeros(2), s.z)) < 1e-6
        # from its own optimum the step stays and is still taken
        z1 = copy(s.z)
        KHSSD._fit_unit_weights!(s, F, d; profile)
        @test (s.steps, s.capped, s.rejected) == (2, 0, 0)
        @test s.z ≈ z1 atol = 1e-6
    end

    # A df = 1 tree: after the first sweep the nugget is 1/E[λ_u], far from
    # ones, and every later sweep's w-step is taken.
    labels = [KHSSD.WeightLabel("u$i", :scale, nothing, "u$i", :log) for i in 1:U]
    names = ["u$i" for i in 1:U]
    spec = (;
        kernels=[
            "near" => (names, kh_grid(x, [0.02, 0.05])),
            "broad" => (names, kh_grid(x, [0.3])),
        ],
        kernel_types=["near" => [:scale], "broad" => [:scale]],
    )
    a = vec(cholesky(Symmetric(Kw)).L * randn(rng, U))
    a[[5, 25, 40]] .+= [4.0, -3.0, 5.0]
    trees = KHSSD._build_trees(
        Float64,
        labels,
        Dict(:scale => 1),
        ();
        sd_prior=(0.5, 4.0),
        root_sd=3.0,
        lkj_eta=2.0,
        type_df=Dict(:scale => 1.0),
    )
    tree = only(trees)
    KHSSD._attach_unit_structure!(trees, labels, spec)
    tree.node_mean .= 0.0
    tree.root .= 0.0
    tree.root_centre .= 0.0
    s = tree.unit_structure
    @test KHSSD._update_tree!(tree, a, Matrix(0.01I, U, U))
    @test s.steps > 2 && s.capped == 0 && s.rejected == 0
    @test maximum(s.nugget) / minimum(s.nugget) > 10
    @test s.nugget ≈ 1 ./ tree.lambda
    @test all(!iszero, s.z)

    # Through KernelPrior: the counts, and no warning from a fit whose steps work.
    r = U
    kernels = map(1:r) do i
        Kernel(reshape(Float64.(1:r .== i), r, 1); factor=true, name="u$i", type=:scale)
    end
    plain = KernelPrior(kernels; levels=:hierarchical)
    @test kernel_unit_steps(plain) == (; steps=0, capped=0, rejected=0)
    prior = KernelPrior(kernels; levels=:hierarchical, df=1.0, unit_structure=spec)
    @test kernel_unit_steps(prior) == (; steps=0, capped=0, rejected=0)
    S = Matrix(Diagonal(exp.(a) .* [sum(abs2, randn(rng, 50)) / 50 for _ in 1:r]))
    @test_logs min_level = Base.CoreLogging.Warn fit_kernel_weights!(prior, S, 50; rounds=5)
    counts = kernel_unit_steps(prior)
    @test counts.steps > 0 && counts.capped == 0 && counts.rejected == 0
    # and a warning when a call adds to either count
    @test_logs (:warn, r"iteration cap") KHSSD._warn_unit_steps(
        (; steps=0, capped=0, rejected=0), (; steps=4, capped=1, rejected=2)
    )
    @test_logs KHSSD._warn_unit_steps(
        (; steps=0, capped=0, rejected=0), (; steps=4, capped=0, rejected=0)
    )
end
