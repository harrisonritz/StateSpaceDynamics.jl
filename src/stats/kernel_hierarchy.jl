#=============================================================================
Hyperpriors on kernel weights: a Gaussian tree per type of weight.

A weight `w` (a log variance weight, or the logit of a correlation weight)
belongs to one `type`. Each type, or each set of types declared correlated,
has one `WeightTree` with 0, 1 or 2 fitted levels:

  levels = 0   w_g ~ N(m_g, s_g²)                         fixed
  levels = 1   η_u ~ N(μ, Ω),      μ ~ N(m₀, r² I)
  levels = 2   η_u ~ N(μ_n(u), Ω), μ_n ~ N(ρ, diag ψ²),  ρ ~ N(m₀, r² I)

`η_u` collects the weights of unit `u` across the tree's `d` types (`d = 1`
unless the types are correlated), `Ω = diag(σ) R diag(σ)` with `R ~ LKJ(η)`,
and every variance has a scaled-inverse-χ² hyperprior entering as `ν₀`
pseudo-observations of `s₀²`. The tree is updated by coordinate ascent given
the Laplace posterior of the weights (`_update_tree!`).
=============================================================================#

# No level of a tree collapses below this SD (natural-log or logit units).
const _TREE_SD_FLOOR = 0.05
# The bound on a canonical partial correlation's unconstrained value in the Ω-step.
const _CPC_BOUND = 10.0

"""
    WeightLabel

What one kernel weight is: its `name` (unique within its `type`), the `type`
whose hyperprior it shares, its optional `group` (the node it hangs from in a
nested tree), its `unit` (matches weights across correlated types) and its
`link` (`:log` for a variance weight `θ = exp(w)`, `:logit` for a correlation
weight).
"""
struct WeightLabel
    name::String
    type::Symbol
    group::Union{Nothing,String}
    unit::String
    link::Symbol
end

_label_text(label::WeightLabel) = string(label.type, ":", label.name)

"""
    WeightTree{T}

**Internal.** The hyperprior of the weights of one type (`d = 1`) or of a set of
correlated types, and its fitted state. `idx[u, j]` is the position in the
weight vector of unit `u`'s weight of type `types[j]`; `node[u]` its node.
See the file header for the model.
"""
mutable struct WeightTree{T<:Real}
    const types::Vector{Symbol}
    const levels::Int
    const idx::Matrix{Int}
    const node::Vector{Int}
    const node_names::Vector{String}
    const leftover::Vector{Vector{Int}}
    # levels = 0: the fixed prior of each weight (NaN centre: set at the start)
    const centre::Vector{T}
    const sd::Vector{T}
    # levels > 0: fitted state
    node_mean::Matrix{T}            # (nodes, d)
    node_cov::Array{T,3}            # (d, d, nodes)
    sds::Vector{T}                  # σ, (d)
    corr::Matrix{T}                 # R, (d, d)
    psi::Vector{T}                  # ψ, (d); levels = 2
    root::Vector{T}                 # ρ, (d); levels = 2
    const root_centre::Vector{T}    # m₀, (d); NaN: set at the start
    const sd_prior::Union{Nothing,Tuple{T,T}}
    const root_sd::Union{Nothing,T}
    const lkj_eta::T
end

_tree_dim(tree::WeightTree) = length(tree.types)
_tree_omega(tree::WeightTree) = tree.sds .* tree.corr .* tree.sds'

function _level_count(level::Symbol)
    level === :flat && return 0
    level === :hierarchical && return 1
    level === :nested && return 2
    return throw(
        ArgumentError("levels must be :flat, :hierarchical or :nested; got :$level")
    )
end

"""
    _build_trees(T, labels, levels, correlated; sd_prior, root_sd, lkj_eta)

One [`WeightTree`](@ref) per correlated set and per remaining type, with every
check on the grouping the trees rely on. `levels` maps a type to its number of
levels (missing: 0); `correlated` is a collection of tuples of types.
"""
function _build_trees(
    ::Type{T},
    labels::Vector{WeightLabel},
    levels::Dict{Symbol,Int},
    correlated;
    sd_prior::Union{Nothing,Tuple{T,T}},
    root_sd::Union{Nothing,T},
    lkj_eta::T,
) where {T<:Real}
    types = unique(label.type for label in labels)
    for type in keys(levels)
        type in types || throw(
            ArgumentError(
                "levels names the type :$type, which no kernel has; the types are " *
                join(string.(":", types), ", "),
            ),
        )
    end
    members = Dict(type => findall(l -> l.type === type, labels) for type in types)
    for type in types
        names = [labels[i].name for i in members[type]]
        allunique(names) ||
            throw(ArgumentError("kernel names must be unique within the type :$type"))
        links = unique(labels[i].link for i in members[type])
        length(links) == 1 ||
            throw(ArgumentError("the type :$type mixes variance and correlation weights"))
    end

    trees = WeightTree{T}[]
    joined = Symbol[]
    for set in correlated
        set = collect(Symbol, set)
        length(set) >= 2 && allunique(set) || throw(
            ArgumentError("a correlated set needs at least two distinct types; got $set"),
        )
        for type in set
            type in types || throw(
                ArgumentError("correlated names the type :$type, which no kernel has")
            )
            type in joined &&
                throw(ArgumentError("the type :$type is in more than one correlated set"))
        end
        depth = unique(get(levels, type, 0) for type in set)
        length(depth) == 1 && depth[1] >= 1 || throw(
            ArgumentError(
                "correlated types need the same levels, :hierarchical or :nested; " *
                "got $(join(string.(":", set), ", "))",
            ),
        )
        append!(joined, set)
        push!(
            trees,
            _joint_tree(T, labels, members, set, depth[1]; sd_prior, root_sd, lkj_eta),
        )
    end
    for type in types
        type in joined && continue
        push!(
            trees,
            _single_tree(
                T, labels, members[type], type, get(levels, type, 0); sd_prior, root_sd
            ),
        )
    end
    return trees
end

# The nodes of a nested tree: one per group label, in order of appearance.
function _tree_nodes(labels::Vector{WeightLabel}, rows::Vector{Int}, type::Symbol)
    missing_group = [labels[i].name for i in rows if labels[i].group === nothing]
    isempty(missing_group) || throw(
        ArgumentError(
            "a :nested type needs a group for every kernel; :$type has none for " *
            join(first(missing_group, 5), ", "),
        ),
    )
    groups = String[something(labels[i].group, "") for i in rows]
    names = unique(groups)
    length(names) >= 2 || throw(
        ArgumentError(
            "a :nested type needs at least two groups; :$type has $(length(names))"
        ),
    )
    return [findfirst(==(g), names)::Int for g in groups], names
end

function _empty_tree(
    ::Type{T},
    types::Vector{Symbol},
    levels::Int,
    idx::Matrix{Int},
    node::Vector{Int},
    node_names::Vector{String},
    leftover::Vector{Vector{Int}};
    sd_prior,
    root_sd,
    lkj_eta::T,
) where {T<:Real}
    d = length(types)
    nodes = length(node_names)
    nflat = levels == 0 ? size(idx, 1) : 0
    return WeightTree{T}(
        types,
        levels,
        idx,
        node,
        node_names,
        leftover,
        fill(T(NaN), nflat),
        ones(T, nflat),
        fill(T(NaN), nodes, d),
        zeros(T, d, d, nodes),
        ones(T, d),
        Matrix{T}(I, d, d),
        ones(T, d),
        fill(T(NaN), d),
        fill(T(NaN), d),
        sd_prior,
        root_sd,
        lkj_eta,
    )
end

function _single_tree(
    ::Type{T},
    labels::Vector{WeightLabel},
    rows::Vector{Int},
    type::Symbol,
    levels::Int;
    sd_prior,
    root_sd,
) where {T<:Real}
    if levels >= 1 && length(rows) < 2
        throw(
            ArgumentError(
                "a :hierarchical or :nested type needs at least two kernels; :$type has " *
                "$(length(rows))",
            ),
        )
    end
    node, names = if levels == 2
        _tree_nodes(labels, rows, type)
    else
        ones(Int, length(rows)), ["all"]
    end
    return _empty_tree(
        T,
        [type],
        levels,
        reshape(copy(rows), :, 1),
        node,
        names,
        [Int[]];
        sd_prior,
        root_sd,
        lkj_eta=one(T),
    )
end

function _joint_tree(
    ::Type{T},
    labels::Vector{WeightLabel},
    members::Dict{Symbol,Vector{Int}},
    set::Vector{Symbol},
    levels::Int;
    sd_prior,
    root_sd,
    lkj_eta::T,
) where {T<:Real}
    lookup = map(set) do type
        units = [labels[i].unit for i in members[type]]
        allunique(units) || throw(
            ArgumentError(
                "a correlated type needs one kernel per unit; :$type repeats one"
            ),
        )
        return Dict(zip(units, members[type]))
    end
    common = [
        u for
        u in (labels[i].unit for i in members[set[1]]) if all(haskey(l, u) for l in lookup)
    ]
    length(common) >= 2 || throw(
        ArgumentError(
            "the correlated types $(join(string.(":", set), ", ")) share " *
            "$(length(common)) units; they need at least two (units match by name)",
        ),
    )
    idx = [lookup[j][u] for u in common, j in eachindex(set)]
    leftover = [setdiff(members[type], view(idx, :, j)) for (j, type) in enumerate(set)]
    node, names = if levels == 2
        # The group of a unit: its first type's that gives one; the others must agree.
        groups = map(eachindex(common)) do u
            found = unique(
                something(labels[i].group) for
                i in view(idx, u, :) if labels[i].group !== nothing
            )
            length(found) == 1 || throw(
                ArgumentError(
                    "the unit $(common[u]) of the correlated types " *
                    "$(join(string.(":", set), ", ")) needs one group; found " *
                    (isempty(found) ? "none" : join(found, ", ")),
                ),
            )
            return found[1]
        end
        names = unique(groups)
        length(names) >= 2 ||
            throw(ArgumentError("a :nested correlated set needs at least two groups"))
        [findfirst(==(g), names)::Int for g in groups], names
    else
        ones(Int, length(common)), ["all"]
    end
    return _empty_tree(
        T, set, levels, idx, node, names, leftover; sd_prior, root_sd, lkj_eta
    )
end

# ----------------------------------------------------------------------------
# The prior the trees put on the weights
# ----------------------------------------------------------------------------

"""
    _weight_prior!(μ, Λ, trees)

The Gaussian prior of the weight vector given the trees' current state: its
mean `μ` and precision `Λ` (dense; diagonal except for the `d × d` blocks of
correlated units).
"""
function _weight_prior!(
    μ::Vector{T}, Λ::Matrix{T}, trees::Vector{WeightTree{T}}
) where {T<:Real}
    fill!(μ, zero(T))
    fill!(Λ, zero(T))
    for tree in trees
        if tree.levels == 0
            for (u, i) in enumerate(view(tree.idx, :, 1))
                μ[i] = tree.centre[u]
                Λ[i, i] = inv(tree.sd[u]^2)
            end
            continue
        end
        Ω = _tree_omega(tree)
        Ωi = inv(Symmetric(Ω))
        for u in axes(tree.idx, 1)
            rows = view(tree.idx, u, :)
            for (a, i) in enumerate(rows)
                μ[i] = tree.node_mean[tree.node[u], a]
                for (b, j) in enumerate(rows)
                    Λ[i, j] = Ωi[a, b]
                end
            end
        end
        # Unmatched weights of a correlated type keep its marginal prior.
        for (a, rows) in enumerate(tree.leftover), i in rows
            if tree.levels == 2
                μ[i] = tree.root[a]
                Λ[i, i] = inv(Ω[a, a] + tree.psi[a]^2)
            else
                μ[i] = tree.node_mean[1, a]
                Λ[i, i] = inv(Ω[a, a])
            end
        end
    end
    return μ, Λ
end

# ----------------------------------------------------------------------------
# The update
# ----------------------------------------------------------------------------

# σ² from the summed expected squared residuals of `n` units, with `ν₀`
# pseudo-observations of `s₀²`.
function _level_variance(total::T, n::Real, sd_prior) where {T<:Real}
    sd_prior === nothing && return total / T(n)
    s0, nu0 = sd_prior
    return (total + nu0 * s0^2) / (T(n) + nu0)
end

# The Cholesky factor of a correlation matrix from its unconstrained canonical
# partial correlations (tanh of `z`, row-major below the diagonal).
function _cpc_cholesky(z::AbstractVector{T}, d::Int) where {T<:Real}
    L = zeros(T, d, d)
    L[1, 1] = one(T)
    k = 0
    for i in 2:d
        rem = one(T)
        for j in 1:(i - 1)
            k += 1
            L[i, j] = tanh(z[k]) * sqrt(rem)
            rem -= L[i, j]^2
        end
        L[i, i] = sqrt(max(rem, zero(T)))
    end
    return L
end

function _cpc_from_corr(R::AbstractMatrix{T}) where {T<:Real}
    d = size(R, 1)
    L = cholesky(Symmetric(Matrix(R))).L
    z = T[]
    clip = one(T) - T(1e-9)
    for i in 2:d
        rem = one(T)
        for j in 1:(i - 1)
            push!(z, atanh(clamp(L[i, j] / sqrt(max(rem, floatmin(T))), -clip, clip)))
            rem -= L[i, j]^2
        end
    end
    return z
end

# −log of the Ω-step objective: n/2 log|Ω| + ½ tr(Ω⁻¹S) − (η − 1) log|R|.
function _omega_objective(
    S::AbstractMatrix{T}, n::T, eta::T, logsd::AbstractVector{T}, L::AbstractMatrix{T}
) where {T<:Real}
    # A correlation at ±1 (a zero pivot) is the boundary, not a solution.
    all(i -> L[i, i] > 0, axes(L, 1)) || return T(Inf)
    scaled = S .* exp.(-logsd) .* exp.(-logsd)'
    half = LowerTriangular(L) \ scaled
    quad = tr(LowerTriangular(L) \ half')
    logdet_half = sum(log(max(L[i, i], floatmin(T))) for i in axes(L, 1))
    return n * sum(logsd) + (n - 2 * (eta - 1)) * logdet_half + quad / 2
end

"""
    _fit_omega(S, n, eta, σ, R) -> (σ, R)

The maximizer of `−n/2 log|Ω| − ½ tr(Ω⁻¹S) + (η − 1) log|R|` over
`Ω = diag(σ) R diag(σ)`, started at `(σ, R)`: `S / n` in closed form when the
LKJ prior is flat (`η = 1`) or there is one dimension, else L-BFGS over
`(log σ, z)`, `z` the canonical partial correlations. Never worse than the
start.
"""
function _fit_omega(
    S::AbstractMatrix{T}, n::T, eta::T, σ::Vector{T}, R::Matrix{T}
) where {T<:Real}
    d = size(S, 1)
    if d == 1 || abs(eta - 1) < 1e-12
        sd = max.(sqrt.(diag(S) ./ n), T(_TREE_SD_FLOOR))
        corr = Matrix(Symmetric(S ./ n ./ (sd .* sd')))
        for i in 1:d
            corr[i, i] = one(T)
        end
        return sd, corr
    end
    # The partial correlations are held in ±tanh(_CPC_BOUND), as dynameg's
    # bounded search does: past it `tanh` rounds to one and the factor is singular.
    cpc(p) = _cpc_cholesky(clamp.(view(p, (d + 1):length(p)), -_CPC_BOUND, _CPC_BOUND), d)
    p0 = vcat(log.(σ), clamp.(_cpc_from_corr(R), -_CPC_BOUND, _CPC_BOUND))
    f = p -> _omega_objective(S, n, eta, view(p, 1:d), cpc(p))
    start = f(p0)
    result = optimize(f, p0, LBFGS(), Optim.Options(; iterations=500, g_tol=1e-10))
    p = Optim.minimizer(result)
    (isfinite(Optim.minimum(result)) && Optim.minimum(result) <= start) || return σ, R
    L = cpc(p)
    return max.(exp.(p[1:d]), T(_TREE_SD_FLOOR)), Matrix(Symmetric(L * L'))
end

"""
    _update_tree!(tree, w, Σw; sweeps=200, tol=1e-8) -> Bool

Coordinate ascent of one tree's state given the Laplace posterior of the
weights (mode `w`, covariance `Σw`): each node's Gaussian posterior, `Ω` from
the expected second moment of the units about their nodes, then `ψ` and the
root. Mean field across levels. A tree without fitted levels is left alone.
Returns whether the sweeps settled.
"""
function _update_tree!(
    tree::WeightTree{T}, w::Vector{T}, Σw::Matrix{T}; sweeps::Int=200, tol::Real=1e-8
) where {T<:Real}
    tree.levels == 0 && return true
    idx = tree.idx
    units, d = size(idx)
    nodes = length(tree.node_names)
    a = w[idx]
    Vsum = zeros(T, d, d)
    for u in 1:units
        rows = view(idx, u, :)
        Vsum .+= view(Σw, rows, rows)
    end
    counts = zeros(T, nodes)
    sums = zeros(T, nodes, d)
    for u in 1:units
        counts[tree.node[u]] += 1
        sums[tree.node[u], :] .+= view(a, u, :)
    end
    m0 = tree.root_centre
    root_precision = tree.root_sd === nothing ? zero(T) : inv(tree.root_sd^2)
    S = zeros(T, d, d)
    for _ in 1:sweeps
        old = vcat(tree.root, tree.sds, vec(tree.corr), tree.psi, vec(tree.node_mean))
        Ωi = inv(Symmetric(_tree_omega(tree)))
        for n in 1:nodes
            # the node's own prior: the root's (levels = 2) or the hyperprior's
            P = counts[n] .* Ωi
            rhs = Ωi * view(sums, n, :)
            if tree.levels == 2
                P += Diagonal(inv.(tree.psi .^ 2))
                rhs .+= tree.root ./ tree.psi .^ 2
            else
                P += root_precision * I
                rhs .+= root_precision .* m0
            end
            cov = inv(Symmetric(P))
            tree.node_cov[:, :, n] .= cov
            tree.node_mean[n, :] .= cov * rhs
        end
        resid = a .- view(tree.node_mean, tree.node, :)
        S .= resid' * resid .+ Vsum
        for n in 1:nodes
            S .+= counts[n] .* view(tree.node_cov, :, :, n)
        end
        n_eff = T(units)
        if tree.sd_prior !== nothing
            s0, nu0 = tree.sd_prior
            S .+= nu0 * s0^2 .* Matrix{T}(I, d, d)
            n_eff += nu0
        end
        tree.sds, tree.corr = _fit_omega(
            Matrix(Symmetric(S)), n_eff, tree.lkj_eta, tree.sds, tree.corr
        )
        if tree.levels == 2
            for j in 1:d
                precision = nodes / tree.psi[j]^2 + root_precision
                tree.root[j] =
                    (
                        sum(view(tree.node_mean, :, j)) / tree.psi[j]^2 +
                        root_precision * m0[j]
                    ) / precision
                root_var = tree.root_sd === nothing ? zero(T) : inv(precision)
                total = zero(T)
                for n in 1:nodes
                    total +=
                        (tree.node_mean[n, j] - tree.root[j])^2 +
                        tree.node_cov[j, j, n] +
                        root_var
                end
                tree.psi[j] = max(
                    sqrt(_level_variance(total, nodes, tree.sd_prior)), T(_TREE_SD_FLOOR)
                )
            end
        end
        new = vcat(tree.root, tree.sds, vec(tree.corr), tree.psi, vec(tree.node_mean))
        maximum(abs, new .- old) < tol && return true
    end
    return false
end

# The state a tree's convergence is judged by.
function _tree_state(tree::WeightTree{T}) where {T<:Real}
    tree.levels == 0 && return T[]
    return vcat(tree.root, tree.sds, vec(tree.corr), tree.psi)
end

"""
    _tree_logdensity(tree) -> T

The log density of a tree's fitted state under its hyperpriors, up to constants:
the nodes about the root, the root, and each variance's pseudo-observations
(with the LKJ term for a correlated set). Zero for a tree without fitted levels.
"""
function _tree_logdensity(tree::WeightTree{T}) where {T<:Real}
    tree.levels == 0 && return zero(T)
    total = zero(T)
    d = _tree_dim(tree)
    m0 = tree.root_centre
    if tree.levels == 2
        for j in 1:d, n in axes(tree.node_mean, 1)
            total -=
                log(tree.psi[j]) +
                (tree.node_mean[n, j] - tree.root[j])^2 / (2tree.psi[j]^2)
        end
    end
    if tree.root_sd !== nothing
        top = tree.levels == 2 ? tree.root : view(tree.node_mean, 1, :)
        total -= sum(abs2, top .- m0) / (2tree.root_sd^2)
    end
    Ω = _tree_omega(tree)
    if tree.sd_prior !== nothing
        s0, nu0 = tree.sd_prior
        total -= nu0 / 2 * logdet(Symmetric(Ω)) + nu0 * s0^2 / 2 * tr(inv(Symmetric(Ω)))
        if tree.levels == 2
            for j in 1:d
                total -= nu0 * log(tree.psi[j]) + nu0 * s0^2 / (2tree.psi[j]^2)
            end
        end
    end
    d > 1 && (total += (tree.lkj_eta - 1) * logdet(Symmetric(tree.corr)))
    return total
end
