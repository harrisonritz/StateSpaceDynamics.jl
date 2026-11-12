#=============================================================================
Kernel priors: a covariance built from weighted kernels, the hyperprior on the
weights, and their fit by variational Laplace.

    Σ(w) = Σ_g exp(w_g) Z_g  +  structure(w)                  (r × r)
    w ~ N(μ(η), Λ(η)⁻¹)                                       (kernel_hierarchy.jl)

Given the second moment `S` of `n` draws from `N(0, Σ)`, the weights are the
MAP of `−n/2 [log|Σ| + tr(Σ⁻¹S)] − ½ (w − μ)'Λ(w − μ)`, their Laplace
covariance is `(n I(w) + Λ)⁻¹` with `I` the Fisher information per draw, and
the trees `η` are updated from that posterior.

The LDS emission classes that use it are in `lds/kernel_observations.jl`;
nothing here knows about a state-space model.
=============================================================================#

"""
    AbstractKernelStructure{T}

A part of a [`KernelPrior`](@ref)'s covariance that is not linear in its
weights (see [`ScaleCorrelation`](@ref)). **Internal** interface, for a
structure `s` with weights `w` (a view of the prior's weight vector):

- `_structure_dim(s)`: the size `r` of the covariance;
- `_structure_labels(s)`: one [`WeightLabel`](@ref) per weight;
- `_structure_traces(s)`: per weight, the trace of the kernel it scales (`NaN`
  for a correlation weight);
- `_structure_start(s, logscale)`: starting weights, the variance weights at
  `logscale`;
- `_structure_values(s, w)`: the weights on their natural scale;
- `_add_covariance!(Σ, s, w)`: add the structure's covariance to `Σ`;
- `_add_gradient!(g, s, w, G)`: add `tr(G ∂Σ/∂w_i)` to `g[i]`, `G` symmetric;
- `_derivatives!(D, s, w)`: write `∂Σ/∂w_i` into `D[:, :, i]`.
"""
abstract type AbstractKernelStructure{T<:Real} end

# Eigenvalues below this fraction of the largest are round-off.
_kernel_rank_tol(::Type{T}, r::Int) where {T<:Real} = r * eps(T)

"""
    Kernel(K; name, type=:kernel, group=nothing, unit=name, factor=false,
           prior_mean=nothing, prior_sd=nothing)

One kernel of a [`KernelPrior`](@ref): a symmetric positive semidefinite
`r × r` matrix `K`, or with `factor=true` an `r × m` factor `F` of `K = F F'`
(a low-rank kernel by its columns). Either way the kernel is stored as a factor
of its numerical rank.

The labels say how its weight is pooled:

- `name`: unique within its `type`;
- `type`: kernels of one type share a hyperprior (see [`KernelPrior`](@ref));
- `group`: the node the weight hangs from when its type is `:nested`;
- `unit`: matches this kernel with those of other types when the types are
  declared correlated (default: its name).

`prior_mean` and `prior_sd` set this kernel's own Gaussian prior on its log
weight, for a `:flat` type; they override the prior's defaults.
"""
struct Kernel{T<:Real}
    name::String
    type::Symbol
    group::Union{Nothing,String}
    unit::String
    factor::Matrix{T}
    prior_mean::Union{Nothing,T}
    prior_sd::Union{Nothing,T}
end

function Kernel(
    K::AbstractMatrix{<:Real};
    name,
    type::Symbol=:kernel,
    group=nothing,
    unit=name,
    factor::Bool=false,
    prior_mean::Union{Nothing,Real}=nothing,
    prior_sd::Union{Nothing,Real}=nothing,
)
    T = float(eltype(K))
    label = string(type, ":", name)
    all(isfinite, K) || throw(ArgumentError("the kernel $label has non-finite entries"))
    F = factor ? _compress_factor(Matrix{T}(K), label) : _psd_factor(Matrix{T}(K), label)
    sum(abs2, F) > 0 || throw(ArgumentError("the kernel $label is zero"))
    if prior_sd !== nothing && !(isfinite(prior_sd) && prior_sd > 0)
        throw(ArgumentError("prior_sd of the kernel $label must be positive and finite"))
    end
    if prior_mean !== nothing && !isfinite(prior_mean)
        throw(ArgumentError("prior_mean of the kernel $label must be finite"))
    end
    return Kernel{T}(
        string(name),
        type,
        group === nothing ? nothing : string(group),
        string(unit),
        F,
        prior_mean === nothing ? nothing : T(prior_mean),
        prior_sd === nothing ? nothing : T(prior_sd),
    )
end

# A factor of a symmetric PSD matrix, with as many columns as its numerical rank.
function _psd_factor(K::Matrix{T}, label::AbstractString) where {T<:Real}
    size(K, 1) == size(K, 2) || throw(
        ArgumentError(
            "the kernel $label must be square (or a factor, with factor=true); got " *
            "$(size(K))",
        ),
    )
    scale = max(maximum(abs, K), floatmin(T))
    norm(K - K') <= sqrt(eps(T)) * scale * size(K, 1) ||
        throw(ArgumentError("the kernel $label is not symmetric"))
    values, vectors = eigen(Symmetric((K + K') / 2))
    top = max(values[end], zero(T))
    values[1] >= -sqrt(eps(T)) * max(top, floatmin(T)) ||
        throw(ArgumentError("the kernel $label is not positive semidefinite"))
    keep = values .> _kernel_rank_tol(T, size(K, 1)) * top
    return vectors[:, keep] .* sqrt.(values[keep])'
end

# A factor with more columns than rows is replaced by one of its kernel's rank.
function _compress_factor(F::Matrix{T}, label::AbstractString) where {T<:Real}
    size(F, 2) >= 1 ||
        throw(ArgumentError("the factor of the kernel $label has no columns"))
    size(F, 2) <= size(F, 1) && return F
    return _psd_factor(F * F', label)
end

"""
    KernelPrior(kernels; structure=nothing, levels=:flat, correlated=(),
                prior_sd=1.0, prior_mean=nothing, sd_prior=(0.5, 4.0),
                root_sd=3.0, lkj_eta=2.0, weights=nothing)

A zero-mean Gaussian prior whose `r × r` covariance is a weighted sum of
[`Kernel`](@ref)s, `Σ = Σ_g θ_g Z_g`, plus an optional `structure` (a
[`ScaleCorrelation`](@ref)), with a hyperprior on the weights. The weights are
held on an unconstrained scale, `w = log θ` for a variance weight.

# The hyperprior

Every weight belongs to a `type`. `levels` gives each type's model (one symbol
for all types, or a `NamedTuple` / `Dict` by type; a type left out is `:flat`):

- `:flat`: `w_g ~ N(m_g, s_g²)`, fixed. `prior_mean` gives `m` (a number, or by
  type; `nothing`, the default, is the empirical centre `log(tr S / tr Z_g)`,
  the weight at which the kernel alone carries the power of the first second
  moment the prior is fitted to) and `prior_sd` gives `s`; a kernel's own
  `prior_mean` / `prior_sd` override both.
- `:hierarchical`: `w_g ~ N(μ, τ²)` with `μ` and `τ` fitted,
  `μ ~ N(m₀, root_sd²)`.
- `:nested`: `w_g ~ N(μ_group, τ²)`, `μ_group ~ N(m, s²)`,
  `m ~ N(m₀, root_sd²)`, all fitted; needs each kernel's `group`.

`m₀` is `prior_mean` for the type, or by default the weight at which the type's
kernels together carry the power. Each fitted variance has a
scaled-inverse-χ² hyperprior `sd_prior = (s₀, ν₀)`, entering as `ν₀`
pseudo-observations of `s₀²`; `nothing` for either `sd_prior` or `root_sd`
removes that hyperprior (plain empirical Bayes, which lets a level with few
members collapse).

`correlated` is a collection of tuples of types, e.g. `[(:scale, :coherence)]`.
The weights of one `unit` across those types then share a multivariate normal
about their node, with a fitted covariance whose correlation has an
`LKJ(lkj_eta)` prior. The types of a set need the same `levels`
(`:hierarchical` or `:nested`); kernels of a unit that is missing from one of
them keep their type's marginal prior.

`weights` (unconstrained, one per weight, kernels first) sets the starting
weights; by default every variance weight starts where the kernels together
carry the power.

The fitted state is on the prior: `weights`, `weights_cov` (the Laplace
covariance) and the trees. See [`fit_kernel_weights!`](@ref),
[`kernel_weights`](@ref), [`kernel_covariance`](@ref).
"""
mutable struct KernelPrior{T<:Real,S<:Union{Nothing,AbstractKernelStructure{T}}}
    const dim::Int
    const kernels::Vector{Kernel{T}}
    const structure::S
    const labels::Vector{WeightLabel}
    const U::Matrix{T}              # the kernels' factors, side by side
    const owner::Vector{Int}        # the kernel of each column of U
    const cols::Vector{UnitRange{Int}}
    const traces::Vector{T}         # tr Z of each variance weight, NaN otherwise
    const trees::Vector{WeightTree{T}}
    weights::Vector{T}
    weights_cov::Matrix{T}
    initialized::Bool
end

_by_type(value, ::Symbol, default) = value === nothing ? default : value
function _by_type(value::Union{NamedTuple,AbstractDict}, type::Symbol, default)
    return haskey(value, type) ? value[type] : default
end

function KernelPrior(
    kernels::AbstractVector{<:Kernel};
    structure::Union{Nothing,AbstractKernelStructure}=nothing,
    levels::Union{Symbol,NamedTuple,AbstractDict}=:flat,
    correlated=(),
    prior_sd::Union{Real,NamedTuple,AbstractDict}=1.0,
    prior_mean::Union{Nothing,Real,NamedTuple,AbstractDict}=nothing,
    sd_prior::Union{Nothing,Tuple{Real,Real}}=(0.5, 4.0),
    root_sd::Union{Nothing,Real}=3.0,
    lkj_eta::Real=2.0,
    weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    isempty(kernels) &&
        structure === nothing &&
        throw(ArgumentError("a KernelPrior needs at least one kernel or a structure"))
    T = if isempty(kernels)
        _structure_eltype(structure)
    else
        eltype(first(kernels).factor)
    end
    kernels = convert(Vector{Kernel{T}}, collect(kernels))
    r = isempty(kernels) ? _structure_dim(structure) : size(first(kernels).factor, 1)
    for kernel in kernels
        size(kernel.factor, 1) == r || throw(
            DimensionMismatch(
                "the kernel $(kernel.type):$(kernel.name) has $(size(kernel.factor, 1)) " *
                "rows; the others have $r",
            ),
        )
    end
    if structure !== nothing && _structure_dim(structure) != r
        throw(
            DimensionMismatch(
                "the structure has $(_structure_dim(structure)) rows; the kernels have $r"
            ),
        )
    end
    if sd_prior !== nothing && !(all(isfinite, sd_prior) && all(>(0), sd_prior))
        throw(
            ArgumentError(
                "sd_prior must be a pair (s₀, ν₀) of positive numbers, or nothing"
            ),
        )
    end
    if root_sd !== nothing && !(isfinite(root_sd) && root_sd > 0)
        throw(ArgumentError("root_sd must be positive and finite, or nothing"))
    end
    lkj_eta > 0 || throw(ArgumentError("lkj_eta must be positive"))

    labels = [WeightLabel(k.name, k.type, k.group, k.unit, :log) for k in kernels]
    traces = T[sum(abs2, k.factor) for k in kernels]
    if structure !== nothing
        append!(labels, _structure_labels(structure))
        append!(traces, _structure_traces(structure))
    end
    types = unique(label.type for label in labels)
    depth = if levels isa Symbol
        Dict(type => _level_count(levels) for type in types)
    else
        Dict{Symbol,Int}(Symbol(k) => _level_count(v) for (k, v) in pairs(levels))
    end
    trees = _build_trees(
        T,
        labels,
        depth,
        correlated;
        sd_prior=sd_prior === nothing ? nothing : (T(sd_prior[1]), T(sd_prior[2])),
        root_sd=root_sd === nothing ? nothing : T(root_sd),
        lkj_eta=T(lkj_eta),
    )
    for spec in (prior_sd, prior_mean)
        spec isa Union{NamedTuple,AbstractDict} || continue
        for type in keys(spec)
            Symbol(type) in types || throw(
                ArgumentError("a prior is given for the type :$type, which no kernel has"),
            )
        end
    end
    _set_tree_priors!(trees, labels, kernels, prior_mean, prior_sd)

    widths = [size(k.factor, 2) for k in kernels]
    stops = cumsum(widths)
    cols = [(stop - width + 1):stop for (stop, width) in zip(stops, widths)]
    U = isempty(kernels) ? zeros(T, r, 0) : reduce(hcat, (k.factor for k in kernels))
    owner = reduce(vcat, (fill(g, w) for (g, w) in enumerate(widths)); init=Int[])
    nw = length(labels)
    w = fill(T(NaN), nw)
    initialized = false
    if weights !== nothing
        length(weights) == nw ||
            throw(DimensionMismatch("weights needs $nw values; got $(length(weights))"))
        all(isfinite, weights) || throw(ArgumentError("weights must be finite"))
        w .= weights
    end
    return KernelPrior{T,typeof(structure)}(
        r,
        kernels,
        structure,
        labels,
        U,
        owner,
        cols,
        traces,
        trees,
        w,
        fill(T(NaN), nw, nw),
        initialized,
    )
end

_structure_eltype(::AbstractKernelStructure{T}) where {T} = T

# Logits start, and are centred by default, where a correlation weight is this.
const _START_RHO = 0.1
# The default SD of a flat prior on a logit.
const _LOGIT_SD = 2.0

# Write the given centres and SDs into the trees; NaN marks an empirical centre,
# resolved against the first second moment (`_initialize_prior!`).
function _set_tree_priors!(
    trees::Vector{WeightTree{T}},
    labels::Vector{WeightLabel},
    kernels::Vector{Kernel{T}},
    prior_mean,
    prior_sd,
) where {T<:Real}
    for tree in trees
        if tree.levels == 0
            type = tree.types[1]
            for (u, i) in enumerate(view(tree.idx, :, 1))
                default_sd = labels[i].link === :log ? one(T) : T(_LOGIT_SD)
                sd = prior_sd isa Real && labels[i].link === :log ? prior_sd : default_sd
                sd = _by_type(prior_sd isa Real ? nothing : prior_sd, type, sd)
                mean = prior_mean isa Real && labels[i].link === :log ? prior_mean : NaN
                mean = _by_type(prior_mean isa Real ? nothing : prior_mean, type, mean)
                if i <= length(kernels)
                    kernels[i].prior_sd === nothing || (sd = kernels[i].prior_sd)
                    kernels[i].prior_mean === nothing || (mean = kernels[i].prior_mean)
                end
                isfinite(sd) && sd > 0 ||
                    throw(ArgumentError("prior_sd must be positive and finite"))
                tree.sd[u] = sd
                tree.centre[u] = mean
            end
        else
            for (j, type) in enumerate(tree.types)
                rows = findall(l -> l.type === type, labels)
                for i in rows
                    i <= length(kernels) || continue
                    (
                        kernels[i].prior_sd === nothing && kernels[i].prior_mean === nothing
                    ) || throw(
                        ArgumentError(
                            "the kernel $(_label_text(labels[i])) has a prior of its " *
                            "own, but its type is not :flat",
                        ),
                    )
                end
                link = labels[first(rows)].link
                mean = prior_mean isa Real && link === :log ? prior_mean : NaN
                mean = _by_type(prior_mean isa Real ? nothing : prior_mean, type, mean)
                tree.root_centre[j] = mean
            end
        end
    end
    return trees
end

_logit(p::Real) = log(p) - log1p(-p)

"""
    _initialize_prior!(prior, S)

Resolve what the prior takes from the first second moment `S` it is fitted to:
the empirical centres left open, the trees' starting state, and the starting
weights unless given. Done once; later fits keep them, so that the fitted
objective is that of one fixed model.
"""
function _initialize_prior!(prior::KernelPrior{T}, S::AbstractMatrix{T}) where {T<:Real}
    power = tr(S)
    power > 0 || throw(ArgumentError("the second moment has no power"))
    labels, traces = prior.labels, prior.traces
    is_log = [label.link === :log for label in labels]
    total = sum(traces[is_log]; init=zero(T))
    start = total > 0 ? log(power / total) : zero(T)
    for tree in prior.trees
        if tree.levels == 0
            for (u, i) in enumerate(view(tree.idx, :, 1))
                isnan(tree.centre[u]) || continue
                tree.centre[u] = is_log[i] ? log(power / traces[i]) : T(_logit(_START_RHO))
            end
        else
            for (j, type) in enumerate(tree.types)
                if isnan(tree.root_centre[j])
                    rows = findall(l -> l.type === type, labels)
                    tree.root_centre[j] = if is_log[first(rows)]
                        log(power / sum(traces[rows]))
                    else
                        T(_logit(_START_RHO))
                    end
                end
                tree.root[j] = tree.root_centre[j]
                tree.node_mean[:, j] .= tree.root_centre[j]
            end
        end
    end
    if any(isnan, prior.weights)
        ng = length(prior.kernels)
        prior.weights[1:ng] .= start
        s = prior.structure
        s === nothing || (prior.weights[(ng + 1):end] .= _structure_start(s, start))
    end
    prior.initialized = true
    _kernel_cholesky(prior, prior.weights) === nothing && throw(
        ArgumentError(
            "the kernels' covariance is not positive definite at the starting weights: " *
            "together the kernels must span all $(prior.dim) dimensions (add a " *
            "full-rank kernel, such as the identity, or reduce the basis)",
        ),
    )
    return prior
end

# ----------------------------------------------------------------------------
# Covariance, gradient and information
# ----------------------------------------------------------------------------

function _structure_weights(prior::KernelPrior, w::AbstractVector)
    return view(w, (length(prior.kernels) + 1):length(w))
end

"""
    kernel_covariance(prior[, weights]) -> Matrix

The prior's covariance `Σ` at its current weights (or at `weights`,
unconstrained).
"""
function kernel_covariance(
    prior::KernelPrior{T}, w::AbstractVector{<:Real}=prior.weights
) where {T<:Real}
    length(w) == length(prior.labels) ||
        throw(DimensionMismatch("need $(length(prior.labels)) weights; got $(length(w))"))
    any(isnan, w) &&
        throw(ArgumentError("the prior has no weights yet: fit it, or give weights"))
    Σ = zeros(T, prior.dim, prior.dim)
    if !isempty(prior.kernels)
        scaled = prior.U .* exp.(view(w, prior.owner) ./ 2)'
        mul!(Σ, scaled, scaled')
    end
    s = prior.structure
    s === nothing || _add_covariance!(Σ, s, _structure_weights(prior, w))
    return Matrix(Symmetric((Σ .+ Σ') ./ 2))
end

# The Cholesky factor of Σ(w), or `nothing` where it is not positive definite.
function _kernel_cholesky(prior::KernelPrior{T}, w::AbstractVector{T}) where {T<:Real}
    Σ = kernel_covariance(prior, w)
    all(isfinite, Σ) || return nothing
    F = cholesky(Symmetric(Σ); check=false)
    return issuccess(F) ? F : nothing
end

"""
    _kernel_objective!(grad, prior, w, S, n, μ, Λ) -> T

`n/2 [log|Σ(w)| + tr(Σ(w)⁻¹S)] + ½ (w − μ)'Λ(w − μ)`, minus the log posterior
of the weights up to a constant, and its gradient (skipped when `grad` is
`nothing`). `Inf` where `Σ` is not positive definite, so that a line search
backs off.
"""
function _kernel_objective!(
    grad::Union{Nothing,AbstractVector{T}},
    prior::KernelPrior{T},
    w::AbstractVector{T},
    S::AbstractMatrix{T},
    n::T,
    μ::Vector{T},
    Λ::Matrix{T},
) where {T<:Real}
    F = all(isfinite, w) ? _kernel_cholesky(prior, w) : nothing
    if F === nothing
        grad === nothing || fill!(grad, zero(T))
        return T(Inf)
    end
    Σi = inv(F)
    ΣiS = Σi * S
    resid = w .- μ
    Λr = Λ * resid
    value = n / 2 * (logdet(F) + tr(ΣiS)) + dot(resid, Λr) / 2
    grad === nothing && return value
    G = Σi .- ΣiS * Σi
    G .= (G .+ G') ./ 2
    fill!(grad, zero(T))
    if !isempty(prior.kernels)
        GU = G * prior.U
        for (g, cols) in enumerate(prior.cols)
            grad[g] = exp(w[g]) * dot(view(prior.U, :, cols), view(GU, :, cols))
        end
    end
    s = prior.structure
    if s !== nothing
        ng = length(prior.kernels)
        _add_gradient!(view(grad, (ng + 1):length(w)), s, _structure_weights(prior, w), G)
    end
    grad .= (n / 2) .* grad .+ Λr
    return value
end

"""
    _kernel_information(prior, w) -> Matrix

The Fisher information of the weights per draw,
`I_ij = ½ tr(Σ⁻¹ ∂_iΣ Σ⁻¹ ∂_jΣ)`. Forms every whitened derivative, so it takes
`n_weights × r²` memory.
"""
function _kernel_information(prior::KernelPrior{T}, w::AbstractVector{T}) where {T<:Real}
    F = _kernel_cholesky(prior, w)
    F === nothing &&
        throw(ArgumentError("the kernels' covariance is not positive definite"))
    r, nw, ng = prior.dim, length(w), length(prior.kernels)
    D = zeros(T, r, r, nw)
    for (g, cols) in enumerate(prior.cols)
        half = view(prior.U, :, cols)
        mul!(view(D, :, :, g), half, half', exp(w[g]), zero(T))
    end
    s = prior.structure
    s === nothing ||
        _derivatives!(view(D, :, :, (ng + 1):nw), s, _structure_weights(prior, w))
    X = Matrix{T}(undef, r * r, nw)
    L = F.L
    for i in 1:nw
        whitened = L \ view(D, :, :, i) / L'
        X[:, i] .= vec(whitened)
    end
    info = X' * X
    return (info .+ info') ./ 4
end

# The prior's Gaussian on the weights, from its trees.
function _weight_prior(prior::KernelPrior{T}) where {T<:Real}
    nw = length(prior.labels)
    return _weight_prior!(zeros(T, nw), zeros(T, nw, nw), prior.trees)
end

# ----------------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------------

function _check_moment(prior::KernelPrior{T}, S::AbstractMatrix, n::Real) where {T<:Real}
    size(S) == (prior.dim, prior.dim) || throw(
        DimensionMismatch(
            "the second moment must be $(prior.dim) × $(prior.dim); got $(size(S))"
        ),
    )
    all(isfinite, S) || throw(ArgumentError("the second moment has non-finite entries"))
    isfinite(n) && n > 0 || throw(ArgumentError("the number of draws must be positive"))
    return Matrix{T}((S .+ S') ./ 2)
end

"""
    _kernel_map!(prior, S, n, μ, Λ; max_iter, g_tol) -> (value, converged)

Move `prior.weights` to the MAP given the Gaussian prior `(μ, Λ)`, by L-BFGS
from the current weights. The weights are left alone if the search does not
improve on them.
"""
function _kernel_map!(
    prior::KernelPrior{T},
    S::Matrix{T},
    n::T,
    μ::Vector{T},
    Λ::Matrix{T};
    max_iter::Int,
    g_tol::Real,
) where {T<:Real}
    w0 = copy(prior.weights)
    start = _kernel_objective!(nothing, prior, w0, S, n, μ, Λ)
    isfinite(start) || throw(
        ArgumentError(
            "the kernels' covariance is not positive definite at the current weights"
        ),
    )
    f = w -> _kernel_objective!(nothing, prior, w, S, n, μ, Λ)
    g! = (G, w) -> (_kernel_objective!(G, prior, w, S, n, μ, Λ); G)
    options = Optim.Options(; iterations=max_iter, g_tol=g_tol)
    result = try
        optimize(f, g!, w0, LBFGS(; linesearch=HagerZhang()), options)
    catch err
        err isa LineSearchException || rethrow()
        nothing
    end
    result === nothing && return start, false
    w = Optim.minimizer(result)
    value = Optim.minimum(result)
    (all(isfinite, w) && isfinite(value) && value <= start) || return start, false
    prior.weights .= w
    return value, Optim.converged(result)
end

# The Laplace covariance of the weights, (n I + Λ)⁻¹.
function _kernel_laplace!(prior::KernelPrior{T}, n::T, Λ::Matrix{T}) where {T<:Real}
    curvature = n .* _kernel_information(prior, prior.weights) .+ Λ
    F = cholesky(Symmetric(curvature); check=false)
    prior.weights_cov = issuccess(F) ? Matrix(Symmetric(inv(F))) : pinv(curvature)
    return prior.weights_cov
end

_has_fitted_trees(prior::KernelPrior) = any(tree -> tree.levels > 0, prior.trees)

"""
    fit_kernel_weights!(prior, S, n; max_iter=200, rounds=1, update_trees=true,
                        laplace=true, g_tol=1e-6, tree_tol=1e-4)

Fit a [`KernelPrior`](@ref)'s weights to the second moment `S` (`r × r`) of
`n` draws from `N(0, Σ)`, by variational Laplace: the weights are moved to
their MAP given the hyperprior (at most `max_iter` L-BFGS steps from the
current ones), their Laplace covariance is computed, and the trees of the
`:hierarchical` and `:nested` types are updated from it. That is one round;
`rounds` is the most to run, stopping once the trees move by less than
`tree_tol`.

`update_trees=false` holds the trees; `laplace=false` also skips the covariance
(the costly part with many weights) when the trees are held.

The first call resolves the prior's empirical centres against `S` and keeps
them. Returns `(; converged, objective, rounds)`: `objective` is
`n/2 [log|Σ| + tr(Σ⁻¹S)] + ½ (w − μ)'Λ(w − μ)` at the weights, under the trees
they were fitted with.
"""
function fit_kernel_weights!(
    prior::KernelPrior{T},
    S::AbstractMatrix{<:Real},
    n::Real;
    max_iter::Int=200,
    rounds::Int=1,
    update_trees::Bool=true,
    laplace::Bool=true,
    g_tol::Real=1e-6,
    tree_tol::Real=1e-4,
) where {T<:Real}
    S = _check_moment(prior, S, n)
    max_iter >= 1 || throw(ArgumentError("max_iter must be at least 1"))
    prior.initialized || _initialize_prior!(prior, S)
    update_trees = update_trees && _has_fitted_trees(prior)
    nw = length(prior.labels)
    μ, Λ = zeros(T, nw), zeros(T, nw, nw)
    value, converged, done = T(NaN), false, 0
    for round in 1:max(rounds, 1)
        done = round
        _weight_prior!(μ, Λ, prior.trees)
        value, found = _kernel_map!(prior, S, T(n), μ, Λ; max_iter, g_tol)
        (laplace || update_trees) && _kernel_laplace!(prior, T(n), Λ)
        if !update_trees
            converged = found
            break
        end
        before = reduce(vcat, (_tree_state(tree) for tree in prior.trees))
        for tree in prior.trees
            _update_tree!(tree, prior.weights, prior.weights_cov)
        end
        after = reduce(vcat, (_tree_state(tree) for tree in prior.trees))
        if found && maximum(abs, after .- before) < tree_tol
            converged = true
            break
        end
    end
    return (; converged, objective=value, rounds=done)
end

"""
    kernel_evidence(prior, S, n) -> Real

The Laplace approximation of the log evidence of the kernels for `n` draws
with second moment `S`, at the prior's current weights and trees:

    −n/2 [log|Σ| + tr(Σ⁻¹S) + r log 2π] − ½ (w − μ)'Λ(w − μ) + ½ log det(Σ_w Λ)

with `Σ_w = (n I + Λ)⁻¹`. It leaves out the uncertainty of the trees, so it
compares sets of kernels fitted to one `S` under one hyperprior.
"""
function kernel_evidence(
    prior::KernelPrior{T}, S::AbstractMatrix{<:Real}, n::Real
) where {T<:Real}
    S = _check_moment(prior, S, n)
    prior.initialized || throw(ArgumentError("the prior has not been fitted"))
    μ, Λ = _weight_prior(prior)
    w = prior.weights
    value = _kernel_objective!(nothing, prior, w, S, T(n), μ, Λ)
    curvature = T(n) .* _kernel_information(prior, w) .+ Λ
    return -value - T(n) * prior.dim * log(T(2π)) / 2 -
           (logdet(cholesky(Symmetric(curvature))) - logdet(cholesky(Symmetric(Λ)))) / 2
end

"""
    _kernel_logprior(prior) -> T

`log p(w | trees) + log p(trees)` at the current weights, up to constants that
do not change during a fit.
"""
function _kernel_logprior(prior::KernelPrior{T}) where {T<:Real}
    μ, Λ = _weight_prior(prior)
    resid = prior.weights .- μ
    total = -dot(resid, Λ * resid) / 2 + logdet(cholesky(Symmetric(Λ))) / 2
    for tree in prior.trees
        total += _tree_logdensity(tree)
    end
    return total
end

"""
    kernel_weights(prior) -> Vector{NamedTuple}

One row per weight of a fitted [`KernelPrior`](@ref): its `name`, `type`,
`group`, `link`, the unconstrained `weight`, its natural `value` (the variance
weight `θ`, or the correlation weight `ρ`) and the Laplace `sd` of `weight`
(`NaN` before a fit computes it).
"""
function kernel_weights(prior::KernelPrior{T}) where {T<:Real}
    w = prior.weights
    ng = length(prior.kernels)
    values = exp.(w)
    s = prior.structure
    s === nothing ||
        (values[(ng + 1):end] .= _structure_values(s, _structure_weights(prior, w)))
    return map(enumerate(prior.labels)) do (i, label)
        variance = prior.weights_cov[i, i]
        (;
            name=label.name,
            type=label.type,
            group=label.group,
            link=label.link,
            weight=w[i],
            value=values[i],
            sd=variance >= 0 ? sqrt(variance) : T(NaN),
        )
    end
end
