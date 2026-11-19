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

A tree with `df = ν < ∞` makes the units' law a multivariate Student t,
`η_u ~ t_ν(μ, Ω)`, as the scale mixture `η_u | λ_u ~ N(μ, Ω/λ_u)`,
`λ_u ~ Gamma(ν/2, ν/2)`, with `q(λ_u) = Gamma((ν + d)/2, (ν + q_u)/2)` (`q_u` the
expected Mahalanobis distance of the unit's deviation): a unit far from the
rest gets a small `E[λ_u]` and is shrunk less, without widening the prior of
the others. With unit kernels the scale mixture is on the nugget alone,
`K_w = w₀ D_λ⁻¹ + Σ w_k K_k` (`D_λ = diag(E[λ])`), so the deviation a unit
shares with its neighbours stays Gaussian.
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
    UnitStructure{T}

**Internal.** Group-level structure across the units of a fitted tree:
optional unit kernels on their covariance and covariates on their mean,

    η = Z μ + X B + E,   Cov(E) = K_w ⊗ Ω (unit-major),
    K_w = w₀ I + Σ_k w_k K_k,   (w₀, w₁, …) = softmax(0, z₁, …),
    z_k ~ N(0, kernel_sd²),

`η` the `units × d` weights, `Z` each unit's node, `X` (`units × p`) the
covariates centred and scaled to unit SD over the tree's units, `B` (`p × d`)
their slopes (fitted where `flags`, else zero). Each kernel is a grid of
widths (`U × U` matrices with unit diagonal, aligned to the tree's units);
`width` is the one in use.
"""
mutable struct UnitStructure{T<:Real}
    const kernel_names::Vector{String}
    const kernels::Vector{Vector{Matrix{T}}}    # kernel → width → (U, U)
    const covariate_names::Vector{String}
    const X::Matrix{T}                          # (U, p)
    const flags::Matrix{Bool}                   # (p, d)
    const kernel_sd::T
    z::Vector{T}                                # (kernels)
    width::Vector{Int}                          # (kernels)
    beta::Matrix{T}                             # (p, d)
    # the identity's share per unit, 1/E[λ_u] of a Student-t tree (else ones)
    nugget::Vector{T}                           # (U)
end

# Without a nugget: ones (a Gaussian tree's).
function UnitStructure{T}(
    names, kernels, cnames, X, flags, kernel_sd, z, width, beta
) where {T<:Real}
    return UnitStructure{T}(
        names, kernels, cnames, X, flags, kernel_sd, z, width, beta, ones(T, size(X, 1))
    )
end

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
    # levels > 0: optional unit kernels and covariates (`_attach_unit_structure!`)
    unit_structure::Union{Nothing,UnitStructure{T}}
    # a Student-t tree (`df < Inf`): q(λ_u) = Gamma((ν + d)/2, rate), E[λ], E[log λ]
    const df::T
    rate::Vector{T}
    lambda::Vector{T}
    loglambda::Vector{T}
end

_is_student(tree::WeightTree) = tree.levels > 0 && isfinite(tree.df)

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
    type_sd_prior::AbstractDict=Dict{Symbol,Any}(),
    type_root_sd::AbstractDict=Dict{Symbol,Any}(),
    type_df::AbstractDict=Dict{Symbol,Any}(),
) where {T<:Real}
    types = unique(label.type for label in labels)
    for type in keys(type_df)
        type in types ||
            throw(ArgumentError("df names the type :$type, which no kernel has"))
    end
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
        dfs = unique(T(get(type_df, type, Inf)) for type in set)
        length(dfs) == 1 ||
            throw(ArgumentError("correlated types need one df; got $(join(dfs, ", "))"))
        push!(
            trees,
            _joint_tree(
                T, labels, members, set, depth[1]; sd_prior, root_sd, lkj_eta, df=dfs[1]
            ),
        )
    end
    for type in types
        type in joined && continue
        push!(
            trees,
            _single_tree(
                T,
                labels,
                members[type],
                type,
                get(levels, type, 0);
                sd_prior=get(type_sd_prior, type, sd_prior),
                root_sd=get(type_root_sd, type, root_sd),
                df=T(get(type_df, type, Inf)),
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
    df::T=T(Inf),
) where {T<:Real}
    d = length(types)
    nodes = length(node_names)
    nflat = levels == 0 ? size(idx, 1) : 0
    df > 0 || throw(ArgumentError("a tree's df must be positive (Inf: Gaussian); got $df"))
    levels == 0 &&
        isfinite(df) &&
        throw(ArgumentError("a Student-t tree (df = $df) needs fitted levels"))
    units = levels == 0 ? 0 : size(idx, 1)
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
        nothing,
        df,
        ones(T, units),
        ones(T, units),
        zeros(T, units),
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
    df::T=T(Inf),
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
        df,
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
    df::T=T(Inf),
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
        T, set, levels, idx, node, names, leftover; sd_prior, root_sd, lkj_eta, df
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
        structure = tree.unit_structure
        if structure === nothing
            for u in axes(tree.idx, 1)
                rows = view(tree.idx, u, :)
                for (a, i) in enumerate(rows)
                    μ[i] = tree.node_mean[tree.node[u], a]
                    for (b, j) in enumerate(rows)
                        Λ[i, j] = tree.lambda[u] * Ωi[a, b]
                    end
                end
            end
        else
            _unit_prior!(μ, Λ, tree, structure, Ωi)
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

# q(λ_u) of a Student-t tree from each unit's expected Mahalanobis distance `q`.
function _student_scales!(tree::WeightTree{T}, q::AbstractVector{T}) where {T<:Real}
    ν, d = tree.df, _tree_dim(tree)
    shape = (ν + d) / 2
    for u in eachindex(q)
        tree.rate[u] = (ν + max(q[u], zero(T))) / 2
        tree.lambda[u] = shape / tree.rate[u]
        tree.loglambda[u] = digamma(shape) - log(tree.rate[u])
    end
    s = tree.unit_structure
    s === nothing || (s.nugget .= inv.(tree.lambda))
    return tree
end

# `E[log p(λ)] + H[q(λ)]`, and the gap `d/2 (E[log λ] − log E[λ])` between the
# variational and the plug-in Gaussian prior of the weights, of a Student-t tree.
function _student_logdensity(tree::WeightTree{T}) where {T<:Real}
    _is_student(tree) || return zero(T)
    ν, d = tree.df, _tree_dim(tree)
    shape = (ν + d) / 2
    total = zero(T)
    for u in eachindex(tree.lambda)
        Eλ, Elog, rate = tree.lambda[u], tree.loglambda[u], tree.rate[u]
        total += ν / 2 * log(ν / 2) - loggamma(ν / 2) + (ν / 2 - 1) * Elog - ν / 2 * Eλ
        total -= shape * log(rate) - loggamma(shape) + (shape - 1) * Elog - rate * Eλ
        total += d / 2 * (Elog - log(Eλ))
    end
    return total
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
    structure = tree.unit_structure
    structure === nothing || return _update_unit_tree!(tree, structure, w, Σw; sweeps, tol)
    idx = tree.idx
    units, d = size(idx)
    nodes = length(tree.node_names)
    a = w[idx]
    V = [Matrix(view(Σw, view(idx, u, :), view(idx, u, :))) for u in 1:units]
    student = _is_student(tree)
    counts = zeros(T, nodes)
    sums = zeros(T, nodes, d)
    Vsum = zeros(T, d, d)
    m0 = tree.root_centre
    root_precision = tree.root_sd === nothing ? zero(T) : inv(tree.root_sd^2)
    S = zeros(T, d, d)
    for _ in 1:sweeps
        old = vcat(
            tree.root, tree.sds, vec(tree.corr), tree.psi, vec(tree.node_mean), tree.lambda
        )
        # the units weighted by E[λ_u] (all one for a Gaussian tree)
        fill!(counts, zero(T))
        fill!(sums, zero(T))
        fill!(Vsum, zero(T))
        for u in 1:units
            λ = tree.lambda[u]
            counts[tree.node[u]] += λ
            sums[tree.node[u], :] .+= λ .* view(a, u, :)
            Vsum .+= λ .* V[u]
        end
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
        S .= resid' * Diagonal(tree.lambda) * resid .+ Vsum
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
        tree.levels == 2 && _update_root!(tree, root_precision)
        if student
            Ωi = inv(Symmetric(_tree_omega(tree)))
            q = [
                dot(Ωi, view(resid, u, :) * view(resid, u, :)' .+ V[u]) +
                dot(Ωi, view(tree.node_cov, :, :, tree.node[u])) for u in 1:units
            ]
            _student_scales!(tree, q)
        end
        new = vcat(
            tree.root, tree.sds, vec(tree.corr), tree.psi, vec(tree.node_mean), tree.lambda
        )
        maximum(abs, new .- old) < tol && return true
    end
    return false
end

# The root and ψ of a nested tree given its nodes' Gaussian posteriors.
function _update_root!(tree::WeightTree{T}, root_precision::T) where {T<:Real}
    nodes = length(tree.node_names)
    m0 = tree.root_centre
    for j in 1:_tree_dim(tree)
        precision = nodes / tree.psi[j]^2 + root_precision
        tree.root[j] =
            (sum(view(tree.node_mean, :, j)) / tree.psi[j]^2 + root_precision * m0[j]) /
            precision
        root_var = tree.root_sd === nothing ? zero(T) : inv(precision)
        total = zero(T)
        for n in 1:nodes
            total +=
                (tree.node_mean[n, j] - tree.root[j])^2 + tree.node_cov[j, j, n] + root_var
        end
        tree.psi[j] = max(
            sqrt(_level_variance(total, nodes, tree.sd_prior)), T(_TREE_SD_FLOOR)
        )
    end
    return tree
end

# The state a tree's convergence is judged by.
function _tree_state(tree::WeightTree{T}) where {T<:Real}
    tree.levels == 0 && return T[]
    state = vcat(tree.root, tree.sds, vec(tree.corr), tree.psi)
    _is_student(tree) && (state = vcat(state, tree.lambda))
    s = tree.unit_structure
    return s === nothing ? state : vcat(state, s.z, vec(s.beta))
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
    s = tree.unit_structure
    s === nothing || (total -= sum(abs2, s.z; init=zero(T)) / (2s.kernel_sd^2))
    return total + _student_logdensity(tree)
end

# ----------------------------------------------------------------------------
# Unit kernels and covariates (`UnitStructure`)
# ----------------------------------------------------------------------------

# The bound on a unit kernel's logit in the w-step.
const _UNIT_Z_BOUND = 10.0
# The w-step's prior term on the logits is Σ z² / (_UNIT_PENALTY kernel_sd²): −2 log of
# their N(0, kernel_sd²) prior, on the scale of the objective (−2 log p).
const _UNIT_PENALTY = 1.0
const _UNIT_STRUCTURE_KEYS = (
    :kernels, :kernel_types, :covariates, :covariate_types, :kernel_sd
)

_spec_get(spec, key::Symbol, default) = haskey(spec, key) ? spec[key] : default

# The entries of a name → value collection, in a stable order: as given, unless
# it is a `Dict` (unordered), whose entries are sorted by name.
function _named_entries(x)
    entries = if x isa AbstractVector
        [string(first(e)) => last(e) for e in x]
    else
        [string(k) => v for (k, v) in pairs(x)]
    end
    x isa Dict && sort!(entries; by=first)
    names = first.(entries)
    allunique(names) || throw(ArgumentError("unit_structure repeats a name"))
    return entries
end

# A unit kernel as its grid of widths, each checked: symmetric, unit diagonal, PSD.
function _kernel_grid(::Type{T}, name::String, value) where {T<:Real}
    names = String[string(n) for n in value[1]]
    K = value[2]
    grid = if ndims(K) == 2
        [Matrix{T}(K)]
    elseif ndims(K) == 3
        [Matrix{T}(K[i, :, :]) for i in axes(K, 1)]
    else
        throw(ArgumentError("the unit kernel $name must be (widths, U, U) or (U, U)"))
    end
    n = length(names)
    allunique(names) || throw(ArgumentError("the unit kernel $name repeats a unit name"))
    isempty(grid) && throw(ArgumentError("the unit kernel $name has no widths"))
    for (j, Kj) in enumerate(grid)
        label = "the unit kernel $name (width $j)"
        size(Kj) == (n, n) ||
            throw(DimensionMismatch("$label is $(size(Kj)); it has $n unit names"))
        all(isfinite, Kj) || throw(ArgumentError("$label has non-finite entries"))
        scale = max(one(T), maximum(abs, Kj))
        maximum(abs, Kj .- Kj'; init=zero(T)) <= sqrt(eps(T)) * scale ||
            throw(ArgumentError("$label is not symmetric"))
        all(i -> abs(Kj[i, i] - 1) <= 1e-6, 1:n) ||
            throw(ArgumentError("$label needs a unit diagonal"))
        issuccess(cholesky(Symmetric(Kj) + sqrt(eps(T)) * I; check=false)) ||
            throw(ArgumentError("$label is not positive semidefinite"))
    end
    return names, grid
end

"""
    _attach_unit_structure!(trees, labels, spec)

Give the fitted trees the unit kernels and covariates of `spec` (see
[`KernelPrior`](@ref)'s `unit_structure`), aligned by unit name to each tree's
units: a unit a kernel does not name gets an identity row; a covariate is
centred and scaled over the units with a value, and a unit without one raises,
except `"none"`, which gets zero (the others' mean).
"""
function _attach_unit_structure!(
    trees::Vector{WeightTree{T}}, labels::Vector{WeightLabel}, spec
) where {T<:Real}
    spec === nothing && return trees
    for key in keys(spec)
        Symbol(key) in _UNIT_STRUCTURE_KEYS || throw(
            ArgumentError(
                "unit_structure has the unknown field $key; it takes " *
                join(_UNIT_STRUCTURE_KEYS, ", "),
            ),
        )
    end
    kernel_sd = T(_spec_get(spec, :kernel_sd, 2.0))
    isfinite(kernel_sd) && kernel_sd > 0 ||
        throw(ArgumentError("unit_structure's kernel_sd must be positive and finite"))
    grids = Dict(
        name => _kernel_grid(T, name, value) for
        (name, value) in _named_entries(_spec_get(spec, :kernels, Pair{String,Any}[]))
    )
    covariates = Dict{String,Tuple{Vector{String},Vector{T}}}()
    for (name, value) in _named_entries(_spec_get(spec, :covariates, Pair{String,Any}[]))
        names = String[string(n) for n in value[1]]
        values = Vector{T}(vec(value[2]))
        length(names) == length(values) || throw(
            DimensionMismatch(
                "the covariate $name has $(length(names)) names and " *
                "$(length(values)) values",
            ),
        )
        allunique(names) || throw(ArgumentError("the covariate $name repeats a unit"))
        all(isfinite, values) ||
            throw(ArgumentError("the covariate $name has non-finite values"))
        covariates[name] = (names, values)
    end
    function types_of(x, what, known)
        map(_named_entries(x)) do (name, members)
            haskey(known, name) || throw(
                ArgumentError("unit_structure gives types for the unknown $what $name")
            )
            return name => Symbol[Symbol(m) for m in members]
        end
    end
    kernel_types = types_of(
        _spec_get(spec, :kernel_types, Pair{String,Any}[]), "kernel", grids
    )
    covariate_types = types_of(
        _spec_get(spec, :covariate_types, Pair{String,Any}[]), "covariate", covariates
    )
    tree_of = Dict(type => tree for tree in trees for type in tree.types)
    for (name, members) in vcat(kernel_types, covariate_types), type in members
        haskey(tree_of, type) || throw(
            ArgumentError(
                "unit_structure names the type :$type ($name), which no kernel has"
            ),
        )
        tree_of[type].levels >= 1 || throw(
            ArgumentError(
                "unit_structure names the type :$type ($name), whose weights are " *
                ":flat; unit kernels and covariates need a fitted tree",
            ),
        )
    end
    for tree in trees
        tree.levels >= 1 || continue
        sets = [
            [k for (k, members) in kernel_types if type in members] for type in tree.types
        ]
        all(==(sets[1]), sets) || throw(
            ArgumentError(
                "the correlated types $(join(string.(":", tree.types), ", ")) need one " *
                "set of unit kernels; they have " *
                join(("[" * join(set, ", ") * "]" for set in sets), ", "),
            ),
        )
        flags = [type in members for (_, members) in covariate_types, type in tree.types]
        used = [any(view(flags, c, :)) for c in axes(flags, 1)]
        kernels = sets[1]
        (isempty(kernels) && !any(used)) && continue
        units = [labels[i].unit for i in view(tree.idx, :, 1)]
        aligned = [_align_kernel(T, k, grids[k]..., units) for k in kernels]
        cnames = [first(covariate_types[c]) for c in eachindex(used) if used[c]]
        X = reduce(
            hcat,
            (_align_covariate(T, c, covariates[c]..., units) for c in cnames);
            init=zeros(T, length(units), 0),
        )
        nk, p, d = length(kernels), length(cnames), _tree_dim(tree)
        tree.unit_structure = UnitStructure{T}(
            kernels,
            aligned,
            cnames,
            X,
            Matrix{Bool}(flags[used, :]),
            kernel_sd,
            zeros(T, nk),
            [length(grid) ÷ 2 + 1 for grid in aligned],
            zeros(T, p, d),
            ones(T, length(units)),
        )
    end
    return trees
end

# A kernel's widths on the tree's units: identity rows for the units it does not name.
function _align_kernel(
    ::Type{T}, name::String, names::Vector{String}, grid::Vector{Matrix{T}}, units
) where {T<:Real}
    position = Dict(n => i for (i, n) in enumerate(names))
    known = [u for u in eachindex(units) if haskey(position, units[u])]
    length(known) >= 2 || throw(
        ArgumentError(
            "the unit kernel $name names $(length(known)) of the tree's units; it " *
            "needs at least two (units match by name)",
        ),
    )
    at = [position[units[u]] for u in known]
    return map(grid) do K
        aligned = Matrix{T}(I, length(units), length(units))
        aligned[known, known] .= view(K, at, at)
        return aligned
    end
end

# A covariate on the tree's units, centred and scaled to unit SD over the units
# with a value; a unit without one raises, except `"none"`, which gets zero.
function _align_covariate(
    ::Type{T}, name::String, names::Vector{String}, values::Vector{T}, units
) where {T<:Real}
    position = Dict(n => i for (i, n) in enumerate(names))
    missing_units = [u for u in units if u != "none" && !haskey(position, u)]
    isempty(missing_units) || throw(
        ArgumentError(
            "the covariate $name has no value for the units " *
            join(first(missing_units, 5), ", "),
        ),
    )
    known = [haskey(position, u) for u in units]
    raw = T[values[position[u]] for u in units[known]]
    centre = sum(raw) / length(raw)
    sd = sqrt(sum(abs2, raw .- centre) / length(raw))
    sd > sqrt(eps(T)) ||
        throw(ArgumentError("the covariate $name is constant over the tree's units"))
    x = zeros(T, length(units))
    x[known] .= (raw .- centre) ./ sd
    return x
end

# softmax(0, z): the weight of the identity, then of each kernel.
function _unit_weights(z::AbstractVector{T}) where {T<:Real}
    top = max(zero(T), maximum(z; init=zero(T)))
    e = exp.(vcat(zero(T), z) .- top)
    return e ./ sum(e)
end

# K_w at the logits `z` and the structure's widths, and the weights.
function _unit_kernel(s::UnitStructure{T}, z::AbstractVector{T}, n::Int) where {T<:Real}
    w = _unit_weights(z)
    K = Matrix{T}(Diagonal(w[1] .* s.nugget))
    for k in eachindex(s.kernels)
        K .+= w[k + 1] .* s.kernels[k][s.width[k]]
    end
    return K, w
end

function _unit_kernel_inverse(s::UnitStructure{T}, n::Int) where {T<:Real}
    K, _ = _unit_kernel(s, s.z, n)
    return _cholesky_inverse(cholesky(Symmetric(K)))
end

# The prior of a structured tree's units: mean Z μ + X B, precision K_w⁻¹ ⊗ Ω⁻¹.
function _unit_prior!(
    μ::Vector{T},
    Λ::Matrix{T},
    tree::WeightTree{T},
    s::UnitStructure{T},
    Ωi::AbstractMatrix{T},
) where {T<:Real}
    idx = tree.idx
    units, d = size(idx)
    Kinv = _unit_kernel_inverse(s, units)
    M = tree.node_mean[tree.node, :] .+ s.X * s.beta
    for v in 1:units, b in 1:d
        j = idx[v, b]
        μ[j] = M[v, b]
        for u in 1:units, a in 1:d
            Λ[idx[u, a], j] = Kinv[u, v] * Ωi[a, b]
        end
    end
    return μ, Λ
end

"""
    _unit_objective!(G, s, z, F, d; profile=nothing) -> T

The w-step objective `d log|K_w| + tr(K_w⁻¹ F) + Σ z²/kernel_sd²` at the
logits `z` (and the structure's widths), and its gradient in `z` into `G`
unless `nothing`.

With `profile = (m, e)` the scale `c` of `Ω` is profiled out instead: `F` is
taken at the current `Ω`, the objective over `(z, cΩ)` is minimal at
`c = (tr(K_w⁻¹F) + e) / m` (`m = d (n + ν₀)`, `e = ν₀ s₀² tr Ω⁻¹`), and the
objective is `d log|K_w| + m log(tr(K_w⁻¹F) + e) + Σ z²/kernel_sd²`.
"""
function _unit_objective!(
    G,
    s::UnitStructure{T},
    z::AbstractVector{T},
    F::Matrix{T},
    d::Int;
    profile::Union{Nothing,Tuple{T,T}}=nothing,
) where {T<:Real}
    n = size(F, 1)
    K, w = _unit_kernel(s, z, n)
    C = cholesky!(Symmetric(K); check=false)
    if !issuccess(C)
        G === nothing || fill!(G, T(NaN))
        return T(Inf)
    end
    Kinv = _cholesky_inverse(C)
    t = dot(Kinv, F)
    penalty = _UNIT_PENALTY * s.kernel_sd^2
    fit, factor = if profile === nothing
        t, one(T)
    else
        m, e = profile
        m * log(t + e), m / (t + e)
    end
    value = d * logdet(C) + fit + sum(abs2, z; init=zero(T)) / penalty
    G === nothing && return value
    M = Kinv * F * Kinv
    slope = zeros(T, length(w))
    slope[1] = d * tr(Kinv) - factor * tr(M)
    for k in eachindex(s.kernels)
        Kk = s.kernels[k][s.width[k]]
        slope[k + 1] = d * dot(Kinv, Kk) - factor * dot(M, Kk)
    end
    mean_slope = dot(slope, w)
    for k in eachindex(z)
        G[k] = w[k + 1] * (slope[k + 1] - mean_slope) + 2z[k] / penalty
    end
    return value
end

"""
    _fit_unit_weights!(s, F, d; profile=nothing) -> T

Move `s.z` to the minimizer of [`_unit_objective!`](@ref) in [−10, 10] by
L-BFGS from the current logits; never worse. Returns `tr(K_w⁻¹ F)` at the
logits it leaves.
"""
function _fit_unit_weights!(
    s::UnitStructure{T}, F::Matrix{T}, d::Int; profile::Union{Nothing,Tuple{T,T}}=nothing
) where {T<:Real}
    if !isempty(s.z)
        f = z -> _unit_objective!(nothing, s, z, F, d; profile)
        g! = (G, z) -> (_unit_objective!(G, s, z, F, d; profile); G)
        fg! = (G, z) -> _unit_objective!(G, s, z, F, d; profile)
        z0 = copy(s.z)
        start = f(z0)
        options = Optim.Options(; iterations=200, g_tol=1e-9, f_reltol=1e-14)
        objective = Optim.OnceDifferentiable(f, g!, fg!, z0)
        bound = T(_UNIT_Z_BOUND)
        result = try
            optimize(objective, z0, LBFGS(), options)
        catch err
            err isa LineSearchException || rethrow()
            nothing
        end
        z = result === nothing ? z0 : Optim.minimizer(result)
        if any(x -> abs(x) > bound, z)
            # The bound binds: L-BFGS in the box from the clamped point.
            inside = clamp.(z, -bound + T(1e-6), bound - T(1e-6))
            boxed = optimize(
                objective,
                fill(-bound, length(z)),
                fill(bound, length(z)),
                inside,
                Optim.Fminbox(LBFGS()),
                options,
            )
            z = clamp.(Optim.minimizer(boxed), -bound, bound)
        end
        value = f(z)
        (all(isfinite, z) && isfinite(value) && value <= start) && (s.z .= z)
    end
    K, _ = _unit_kernel(s, s.z, size(F, 1))
    return dot(_cholesky_inverse(cholesky(Symmetric(K))), F)
end

# For each kernel with several widths, the width with the lowest w-step
# objective at the current logits (the others held).
function _choose_widths!(s::UnitStructure{T}, F::Matrix{T}, d::Int) where {T<:Real}
    for k in eachindex(s.kernels)
        length(s.kernels[k]) > 1 || continue
        best, lowest = s.width[k], T(Inf)
        for j in eachindex(s.kernels[k])
            s.width[k] = j
            value = _unit_objective!(nothing, s, s.z, F, d)
            value < lowest && ((best, lowest) = (j, value))
        end
        s.width[k] = best
    end
    return s
end

# The (d, d) blocks of a unit-major (or node-major) covariance: `B[a, b][u, v]`
# is the covariance of member `a` of unit `u` with member `b` of unit `v`.
function _member_blocks(V::AbstractMatrix{T}, d::Int) where {T<:Real}
    return [V[a:d:end, b:d:end] for a in 1:d, b in 1:d]
end

"""
    _unit_nodes!(tree, s, a, Z, Kinv, Oi) -> (Cμ, ZKZ)

The nodes' joint Gaussian posterior given `B` (step 2): precision
`kron(Z'K⁻¹Z, Ω⁻¹)` plus the nodes' own prior. Writes the means and the
diagonal blocks of the covariance into the tree; returns the full covariance
(node-major) and `Z'K⁻¹Z`.
"""
function _unit_nodes!(
    tree::WeightTree{T},
    s::UnitStructure{T},
    a::Matrix{T},
    Z::Matrix{T},
    Kinv::Matrix{T},
    Oi::Matrix{T},
) where {T<:Real}
    d, nodes = size(a, 2), size(Z, 2)
    ZK = Z' * Kinv
    ZKZ = Matrix(Symmetric(ZK * Z))
    P = kron(ZKZ, Oi)
    linear = vec(permutedims(ZK * (a .- s.X * s.beta) * Oi))
    root_precision = tree.root_sd === nothing ? zero(T) : inv(tree.root_sd^2)
    for n in 1:nodes
        r = ((n - 1) * d + 1):(n * d)
        if tree.levels == 2
            for j in 1:d
                P[r[j], r[j]] += inv(tree.psi[j]^2)
                linear[r[j]] += tree.root[j] / tree.psi[j]^2
            end
        else
            for j in 1:d
                P[r[j], r[j]] += root_precision
                linear[r[j]] += root_precision * tree.root_centre[j]
            end
        end
    end
    Cμ = Matrix(Symmetric(inv(Symmetric(P))))
    tree.node_mean .= permutedims(reshape(Cμ * linear, d, nodes))
    for n in 1:nodes
        r = ((n - 1) * d + 1):(n * d)
        tree.node_cov[:, :, n] .= view(Cμ, r, r)
    end
    return Cμ, ZKZ
end

# The slopes of the flagged covariates given the nodes (step 3): GLS.
function _unit_slopes!(
    s::UnitStructure{T}, resid::Matrix{T}, Kinv::Matrix{T}, Oi::Matrix{T}
) where {T<:Real}
    p, d = size(s.beta)
    p == 0 && return s
    KX = Kinv * s.X
    b = vec(permutedims(KX' * resid * Oi))          # (covariate, member), row-major
    H = kron(s.X' * KX, Oi)
    f = findall(vec(permutedims(s.flags)))
    beta = zeros(T, p * d)
    beta[f] .= Symmetric(H[f, f]) \ b[f]
    s.beta .= permutedims(reshape(beta, d, p))
    return s
end

# F = R Ω⁻¹ R' + Σ_ab Ω⁻¹[a, b] (V_ab + Z Cμ_ab Z') (units × units).
function _unit_moment(
    R::Matrix{T}, Vb::Matrix{Matrix{T}}, Cb::Matrix{Matrix{T}}, Z::Matrix{T}, Oi::Matrix{T}
) where {T<:Real}
    F = R * Oi * R'
    G = zeros(T, size(Z, 2), size(Z, 2))
    for a in axes(Oi, 1), b in axes(Oi, 2)
        F .+= Oi[a, b] .* Vb[a, b]
        G .+= Oi[a, b] .* Cb[a, b]
    end
    F .+= Z * G * Z'
    return Matrix(Symmetric(F))
end

"""
    _update_unit_tree!(tree, s, w, Σw; sweeps, tol) -> Bool

[`_update_tree!`](@ref) for a tree with unit kernels or covariates: given the
mode `w` and the full covariance `Σw` of the weights, coordinate ascent over
the nodes (jointly, step 2), the covariates' slopes (3), `Ω` (4), the kernels'
logits jointly with the scale of `Ω` (5; on the first sweep each kernel's width
is chosen first, at the current logits) and the root and `ψ` (6).
"""
function _update_unit_tree!(
    tree::WeightTree{T},
    s::UnitStructure{T},
    w::Vector{T},
    Σw::Matrix{T};
    sweeps::Int=200,
    tol::Real=1e-8,
) where {T<:Real}
    idx = tree.idx
    units, d = size(idx)
    nodes = length(tree.node_names)
    a = w[idx]
    order = vec(permutedims(idx))
    Vb = _member_blocks(Σw[order, order], d)
    Z = zeros(T, units, nodes)
    for u in 1:units
        Z[u, tree.node[u]] = one(T)
    end
    root_precision = tree.root_sd === nothing ? zero(T) : inv(tree.root_sd^2)
    resid(B) = a .- view(tree.node_mean, tree.node, :) .- s.X * B
    function state()
        return vcat(
            tree.root,
            tree.sds,
            vec(tree.corr),
            tree.psi,
            vec(tree.node_mean),
            s.z,
            vec(s.beta),
            tree.lambda,
        )
    end
    student = _is_student(tree)
    for sweep in 1:sweeps
        old = state()
        Kinv = _unit_kernel_inverse(s, units)
        Oi = Matrix(inv(Symmetric(_tree_omega(tree))))
        Cμ, ZKZ = _unit_nodes!(tree, s, a, Z, Kinv, Oi)
        _unit_slopes!(s, resid(zero(s.beta)), Kinv, Oi)
        R = resid(s.beta)
        Cb = _member_blocks(Cμ, d)
        S = R' * Kinv * R
        for i in 1:d, j in 1:d
            S[i, j] += dot(Kinv, Vb[i, j]) + dot(ZKZ, Cb[i, j])
        end
        n_eff, n_eff_prior = T(units), zero(T)
        if tree.sd_prior !== nothing
            s0, nu0 = tree.sd_prior
            S .+= nu0 * s0^2 .* Matrix{T}(I, d, d)
            n_eff += nu0
            n_eff_prior = nu0 * s0^2
        end
        tree.sds, tree.corr = _fit_omega(
            Matrix(Symmetric(S)), n_eff, tree.lkj_eta, tree.sds, tree.corr
        )
        if !isempty(s.z)
            # The logits jointly with the scale of Ω (the slow direction of
            # alternating the two): the same fixed point, in far fewer sweeps.
            Oi = Matrix(inv(Symmetric(_tree_omega(tree))))
            F = _unit_moment(R, Vb, Cb, Z, Oi)
            sweep == 1 && _choose_widths!(s, F, d)
            e = tree.sd_prior === nothing ? zero(T) : n_eff_prior * tr(Oi)
            m = d * n_eff
            t = _fit_unit_weights!(s, F, d; profile=(m, e))
            tree.sds .= max.(tree.sds .* sqrt((t + e) / m), T(_TREE_SD_FLOOR))
        end
        tree.levels == 2 && _update_root!(tree, root_precision)
        if student
            # the nugget e_n of the deviation e = e_s + e_n, e_n,u ~ N(0, w₀ Ω/λ_u):
            # its second moment given e's, Σ_n Σ⁻¹ E[ee'] Σ⁻¹ Σ_n + Σ_n − Σ_n Σ⁻¹ Σ_n
            Oi = Matrix(inv(Symmetric(_tree_omega(tree))))
            F = _unit_moment(R, Vb, Cb, Z, Oi)
            K, w = _unit_kernel(s, s.z, units)
            Kinv = _cholesky_inverse(cholesky(Symmetric(K)))
            nug = w[1] .* s.nugget
            B = Diagonal(nug) * Kinv
            q = [
                (
                    dot(view(B, u, :), F, view(B, u, :)) +
                    d * (nug[u] - nug[u]^2 * Kinv[u, u])
                ) / w[1] for u in 1:units
            ]
            _student_scales!(tree, q)
        end
        maximum(abs, state() .- old) < tol && return true
    end
    return false
end
