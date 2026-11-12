#=============================================================================
The scale-correlation structure of a kernel prior: a covariance factored into
scales and a correlation matrix, seen through a fixed matrix `G` (r × n).

    Σ_sc = G Γ G',   Γ = S C S,   S = diag(s),   s_i² = θ_{b(i)} v_i
    C = I + Σ_c ρ_c (R_c − I_c),   R_c = B_c B_c'

`θ_b = exp(u_b)`, one per scale block (a partition of the `n` columns). The
correlation kernels come in families ordered coarse to fine, each kernel inside
at most one kernel of every earlier family (its ancestors), and their weights
are stick-broken, `ρ_c = (1 − Σ_{p ∈ ancestors(c)} ρ_p) σ(a_c)`, so that the
weights covering any one column sum to at most one and `C` stays a correlation
matrix.

Nothing `n × n` is formed. With `Gw = G diag(√v)`, `Z_b = Gw_b Gw_b'` and, for
a kernel `c` and a block `b` it touches (`cols`: the columns of `b` that `c`
covers), `W_cb = Gw[:, cols] B_c[cols, :]` and `Z_cb = Gw[:, cols] Gw[:, cols]'`:

    Σ_sc = Σ_b θ_b Z_b + Σ_c ρ_c (A_c A_c' − Σ_b θ_b Z_cb),   A_c = Σ_b √θ_b W_cb

The weights are `[u ; a of family 1 ; a of family 2 ; …]`.
=============================================================================#

# The first few of a list of names, for an error message.
function _first_few(names::AbstractVector{<:AbstractString}; limit::Int=5)
    shown = join(first(names, limit), ", ")::String
    return length(names) > limit ? shown * ", …" : shown
end

# The names, groups and units of `count` weights of the type `type`, checked.
function _weight_labels(type::Symbol, count::Int, names, groups, units)
    labels = String[string(name) for name in names]
    length(labels) == count || throw(
        DimensionMismatch("the type :$type needs $count names; got $(length(labels))")
    )
    seen, repeated = Set{String}(), String[]
    for label in labels
        label in seen && !(label in repeated) && push!(repeated, label)
        push!(seen, label)
    end
    isempty(repeated) || throw(
        ArgumentError(
            "names must be unique within the type :$type; repeated: " *
            _first_few(repeated),
        ),
    )
    nodes = if groups === nothing
        Union{Nothing,String}[nothing for _ in 1:count]
    else
        Union{Nothing,String}[g === nothing ? nothing : string(g) for g in groups]
    end
    length(nodes) == count || throw(
        DimensionMismatch(
            "groups of the type :$type needs $count values; got $(length(nodes))"
        ),
    )
    matched = String[string(unit) for unit in units]
    length(matched) == count || throw(
        DimensionMismatch(
            "units of the type :$type needs $count values; got $(length(matched))"
        ),
    )
    return labels, nodes, matched
end

# A factor with its rows scaled to unit norm, and the rows that are not zero.
function _unit_rows(::Type{T}, factor::AbstractMatrix{<:Real}) where {T<:Real}
    B = dropzeros!(SparseMatrixCSC{T,Int}(sparse(factor)))
    rows, values = rowvals(B), nonzeros(B)
    norms = zeros(T, size(B, 1))
    for (i, value) in zip(rows, values)
        norms[i] = hypot(norms[i], value)
    end
    values ./= view(norms, rows)
    return B, findall(>(0), norms)
end

"""
    CorrelationFamily(factors; type, names, groups=nothing, units=names)

One family of correlation kernels of a [`ScaleCorrelation`](@ref): kernels of
one kind (one per network, say, or one per parcel) whose weights share a
hyperprior.

`factors` holds one `n × m_c` matrix per kernel, sparse or dense, `n` the
number of columns of the structure's gain. Its rows are scaled to unit norm
here, giving `B_c`; the kernel's correlation matrix is `R_c = B_c B_c'` on the
columns whose row is not zero (the columns it covers), and it leaves the others
alone. The kernels of one family must not overlap.

The labels are those of a [`Kernel`](@ref), one per factor:

- `type`: the family's type, under which a [`KernelPrior`](@ref) is told its
  `levels` and whether it is `correlated` with another type;
- `names`: unique within the family;
- `groups`: the node each weight hangs from when the type is `:nested`;
- `units`: match the kernels with the weights of other types (default: the
  names).
"""
struct CorrelationFamily{T<:Real}
    type::Symbol
    names::Vector{String}
    groups::Vector{Union{Nothing,String}}
    units::Vector{String}
    factors::Vector{SparseMatrixCSC{T,Int}}     # B_c, rows at unit norm
    covered::Vector{Vector{Int}}                # the columns each kernel covers
end

function CorrelationFamily(
    factors::AbstractVector{<:AbstractMatrix{<:Real}};
    type::Symbol,
    names,
    groups=nothing,
    units=names,
)
    isempty(factors) && throw(ArgumentError("the correlation family :$type has no kernels"))
    labels, nodes, matched = _weight_labels(type, length(factors), names, groups, units)
    T = float(mapreduce(eltype, promote_type, factors))
    n = size(first(factors), 1)
    unit, covered = SparseMatrixCSC{T,Int}[], Vector{Int}[]
    owner = zeros(Int, n)
    for (c, factor) in enumerate(factors)
        label = string(type, ":", labels[c])
        size(factor, 1) == n || throw(
            DimensionMismatch(
                "the factor of $label has $(size(factor, 1)) rows; the family's first " *
                "has $n",
            ),
        )
        all(isfinite, factor) ||
            throw(ArgumentError("the factor of $label has non-finite entries"))
        B, cols = _unit_rows(T, factor)
        isempty(cols) && throw(ArgumentError("the factor of $label covers no column"))
        others = unique(owner[i] for i in cols if owner[i] > 0)
        isempty(others) || throw(
            ArgumentError(
                "the kernels of the correlation family :$type overlap: $(labels[c]) " *
                "shares columns with " *
                _first_few(labels[others]),
            ),
        )
        owner[cols] .= c
        push!(unit, B)
        push!(covered, cols)
    end
    return CorrelationFamily{T}(type, labels, nodes, matched, unit, covered)
end

# The same family with factors of element type `T`.
function _family_as(::Type{T}, family::CorrelationFamily) where {T<:Real}
    return CorrelationFamily{T}(
        family.type,
        family.names,
        family.groups,
        family.units,
        [SparseMatrixCSC{T,Int}(B) for B in family.factors],
        family.covered,
    )
end

# What a correlation kernel adds on one scale block: the block, `W_cb`, and the
# index of `Z_cb` in the structure's `Z`.
struct _BlockImage{T<:Real}
    block::Int
    image::Matrix{T}
    z::Int
end

"""
    ScaleCorrelation(gain, variance, block; names, type=:scale, groups=nothing,
                     units=names, families=CorrelationFamily[])

A covariance factored into scales and correlations and seen through a fixed
matrix, the `structure` of a [`KernelPrior`](@ref):

    Σ_sc = G Γ G',   Γ = S C S,   S = diag(s),   s_i² = θ_{b(i)} v_i
    C = I + Σ_c ρ_c (R_c − I_c)

- `gain`: `G`, `r × n`, in the coordinates of the prior's covariance (`r`);
- `variance`: `v`, the base variance of each of the `n` columns, non-negative;
- `block`: `b`, the scale block (`1 … g`) of each column. Each block has one
  variance weight `θ_b = exp(u_b)`;
- `names`, `groups`, `units`: the labels of the `g` scale weights, of the type
  `type` (see [`Kernel`](@ref));
- `families`: the [`CorrelationFamily`](@ref)s, ordered coarse to fine.

A correlation kernel `c` has a correlation matrix `R_c` on the columns it
covers (`I_c` is their identity) and a weight `ρ_c`: a column's variance is
`s_i²` whatever correlates it, and two columns covary by `s_i s_j C_ij`. A
kernel must lie inside one kernel of an earlier family, or outside all of them;
those enclosing it are its ancestors. The weights are stick-broken,

    ρ_c = (1 − Σ_{p ∈ ancestors(c)} ρ_p) σ(a_c),   σ(a) = (1 + tanh(a/2)) / 2

with one logit `a_c` per kernel, so that the weights of the kernels covering
any one column sum to at most one and `C` is a correlation matrix for every
`a`.

The structure's weights are `[u ; a]`, the scale blocks and then the kernels of
each family in order; in a `KernelPrior` they follow the weights of its
kernels. [`kernel_weights`](@ref) reports `θ` and `ρ` as their values.

Only `r × r` matrices and `r × m_c` images of the factors are kept, one per
kernel and scale block it touches, so the cost does not grow with `n²`.
"""
struct ScaleCorrelation{T<:Real} <: AbstractKernelStructure{T}
    gain::Matrix{T}
    variance::Vector{T}
    block::Vector{Int}
    type::Symbol
    names::Vector{String}
    groups::Vector{Union{Nothing,String}}
    units::Vector{String}
    families::Vector{CorrelationFamily{T}}
    # per correlation kernel, families in order: the kernels enclosing it
    ancestors::Vector{Vector{Int}}
    # Z_b of each scale block, then the distinct Z_cb
    Z::Vector{Matrix{T}}
    # per correlation kernel: its image on each scale block it touches
    images::Vector{Vector{_BlockImage{T}}}
end

function ScaleCorrelation(
    gain::AbstractMatrix{<:Real},
    variance::AbstractVector{<:Real},
    block::AbstractVector{<:Integer};
    names,
    type::Symbol=:scale,
    groups=nothing,
    units=names,
    families::AbstractVector{<:CorrelationFamily}=CorrelationFamily[],
)
    T = float(eltype(gain))
    r, n = size(gain)
    r >= 1 && n >= 1 || throw(ArgumentError("gain is empty; got $(size(gain))"))
    all(isfinite, gain) || throw(ArgumentError("gain has non-finite entries"))
    length(variance) == n || throw(
        DimensionMismatch(
            "variance needs one value per column of gain ($n); got $(length(variance))"
        ),
    )
    all(v -> isfinite(v) && v >= 0, variance) ||
        throw(ArgumentError("variance must be finite and non-negative"))
    length(block) == n || throw(
        DimensionMismatch(
            "block needs one value per column of gain ($n); got $(length(block))"
        ),
    )
    g = length(names)
    labels, nodes, matched = _weight_labels(type, g, names, groups, units)
    all(b -> 1 <= b <= g, block) || throw(
        ArgumentError(
            "block must hold the scale block of each column, 1 to $g (the number of " *
            "names); got values from $(minimum(block)) to $(maximum(block))",
        ),
    )
    members = [Int[] for _ in 1:g]
    for (i, b) in enumerate(block)
        push!(members[b], i)
    end
    empty = findall(isempty, members)
    isempty(empty) || throw(
        ArgumentError(
            "every scale block needs a column; none for " * _first_few(labels[empty])
        ),
    )
    Gw = Matrix{T}(gain) .* sqrt.(T.(variance))'
    Z = map(members) do cols
        half = Gw[:, cols]
        half * half'
    end
    powerless = findall(z -> !(tr(z) > 0), Z)
    isempty(powerless) || throw(
        ArgumentError(
            "every scale block needs a positive trace (a column with a variance and " *
            "a gain); none for " *
            _first_few(labels[powerless]),
        ),
    )

    fams = CorrelationFamily{T}[_family_as(T, family) for family in families]
    types = Symbol[family.type for family in fams]
    allunique(types) || throw(
        ArgumentError(
            "the correlation families need distinct types; got " *
            _first_few(String[string(":", t) for t in types]),
        ),
    )
    type in types && throw(
        ArgumentError("a correlation family has the type of the scale blocks, :$type")
    )
    for family in fams
        rows = size(first(family.factors), 1)
        rows == n || throw(
            DimensionMismatch(
                "the factors of the correlation family :$(family.type) have $rows " *
                "rows; gain has $n columns",
            ),
        )
    end
    ancestors = _correlation_ancestors(fams, n)

    block = Vector{Int}(block)
    slot = Dict{Tuple{Int,Vector{Int}},Int}((b, members[b]) => b for b in 1:g)
    images = Vector{_BlockImage{T}}[]
    for family in fams, (B, cols) in zip(family.factors, family.covered)
        push!(images, _block_images!(Z, slot, Gw, B, cols, block))
    end
    return ScaleCorrelation{T}(
        Matrix{T}(gain),
        Vector{T}(variance),
        block,
        type,
        labels,
        nodes,
        matched,
        fams,
        ancestors,
        Z,
        images,
    )
end

# For each correlation kernel (families in order), the kernels of the earlier
# families that enclose it. A kernel must lie inside one kernel of an earlier
# family or outside all of them.
function _correlation_ancestors(families::Vector{<:CorrelationFamily}, n::Int)
    ancestors = Vector{Int}[]
    labels = String[]
    owners = Vector{Int}[]      # per family: the kernel covering each column, or 0
    for family in families
        owner = zeros(Int, n)
        for (c, cols) in enumerate(family.covered)
            found = Int[]
            for (e, earlier) in enumerate(owners)
                ids = unique(view(earlier, cols))
                if length(ids) > 1
                    inside = _first_few(labels[filter(>(0), ids)])
                    how = 0 in ids ? "lies partly inside $inside" : "straddles $inside"
                    throw(
                        ArgumentError(
                            "the correlation kernel $(family.type):$(family.names[c]) " *
                            "is not nested in one kernel of the family " *
                            ":$(families[e].type): it $how",
                        ),
                    )
                end
                ids[1] > 0 && push!(found, ids[1])
            end
            push!(ancestors, found)
            owner[cols] .= length(ancestors)
            push!(labels, string(family.type, ":", family.names[c]))
        end
        push!(owners, owner)
    end
    return ancestors
end

# The images of one kernel (unit-row factor `B`, covering `cols`) on the scale
# blocks it touches. A `Z_cb` not yet in `Z` is appended; `slot` finds those
# already there by their block and columns.
function _block_images!(
    Z::Vector{Matrix{T}},
    slot::Dict{Tuple{Int,Vector{Int}},Int},
    Gw::Matrix{T},
    B::SparseMatrixCSC{T,Int},
    cols::Vector{Int},
    block::Vector{Int},
) where {T<:Real}
    touched = sort!(unique(view(block, cols)))
    position = Dict(b => k for (k, b) in enumerate(touched))
    W = [zeros(T, size(Gw, 1), size(B, 2)) for _ in touched]
    rows, values = rowvals(B), nonzeros(B)
    for j in axes(B, 2), k in nzrange(B, j)
        i = rows[k]
        axpy!(values[k], view(Gw, :, i), view(W[position[block[i]]], :, j))
    end
    within = [Int[] for _ in touched]
    for i in cols
        push!(within[position[block[i]]], i)
    end
    return map(enumerate(touched)) do (k, b)
        z = get!(slot, (b, within[k])) do
            half = Gw[:, within[k]]
            push!(Z, half * half')
            length(Z)
        end
        _BlockImage{T}(b, W[k], z)
    end
end

_scale_count(s::ScaleCorrelation) = length(s.names)

_structure_dim(s::ScaleCorrelation) = size(s.gain, 1)

function _structure_labels(s::ScaleCorrelation)
    labels = [
        WeightLabel(s.names[b], s.type, s.groups[b], s.units[b], :log) for
        b in eachindex(s.names)
    ]
    for family in s.families, c in eachindex(family.names)
        push!(
            labels,
            WeightLabel(
                family.names[c], family.type, family.groups[c], family.units[c], :logit
            ),
        )
    end
    return labels
end

function _structure_traces(s::ScaleCorrelation{T}) where {T<:Real}
    g = _scale_count(s)
    return vcat(T[tr(s.Z[b]) for b in 1:g], fill(T(NaN), length(s.ancestors)))
end

# The logits start where every correlation weight is `_START_RHO`.
function _structure_start(s::ScaleCorrelation{T}, logscale::Real) where {T<:Real}
    logits = map(s.ancestors) do enclosing
        rest = max(1 - _START_RHO * length(enclosing), 1e-6)
        T(_logit(_START_RHO / rest))
    end
    return vcat(fill(T(logscale), _scale_count(s)), logits)
end

_sigmoid(a::Real) = (1 + tanh(a / 2)) / 2

# `θ` of the scale blocks, and of the correlation kernels `ρ`, the remainder
# `1 − Σ_ancestors ρ` each breaks and `σ(a)`.
function _scale_correlation_weights(
    s::ScaleCorrelation{T}, w::AbstractVector
) where {T<:Real}
    g, nc = _scale_count(s), length(s.ancestors)
    length(w) == g + nc ||
        throw(DimensionMismatch("the structure has $(g + nc) weights; got $(length(w))"))
    θ = T[exp(w[b]) for b in 1:g]
    σ = T[_sigmoid(w[g + c]) for c in 1:nc]
    ρ, rest = zeros(T, nc), ones(T, nc)
    for (c, enclosing) in enumerate(s.ancestors)
        for p in enclosing
            rest[c] -= ρ[p]
        end
        ρ[c] = rest[c] * σ[c]
    end
    return θ, ρ, rest, σ
end

function _structure_values(s::ScaleCorrelation, w::AbstractVector)
    θ, ρ, _, _ = _scale_correlation_weights(s, w)
    return vcat(θ, ρ)
end

# `A_c = Σ_b √θ_b W_cb` of one kernel.
function _kernel_image(images::Vector{_BlockImage{T}}, rootθ::Vector{T}) where {T<:Real}
    A = zeros(T, size(first(images).image))
    for part in images
        A .+= rootθ[part.block] .* part.image
    end
    return A
end

function _add_covariance!(
    Σ::AbstractMatrix{T}, s::ScaleCorrelation{T}, w::AbstractVector
) where {T<:Real}
    θ, ρ, _, _ = _scale_correlation_weights(s, w)
    rootθ = sqrt.(θ)
    coefficient = zeros(T, length(s.Z))
    coefficient[1:length(θ)] .= θ
    for (c, images) in enumerate(s.images)
        A = _kernel_image(images, rootθ)
        mul!(Σ, A, A', ρ[c], one(T))
        for part in images
            coefficient[part.z] -= ρ[c] * θ[part.block]
        end
    end
    for (z, Z) in enumerate(s.Z)
        Σ .+= coefficient[z] .* Z
    end
    return Σ
end

# The adjoint of the stick-breaking: `out[c] = Σ_d (∂ρ_d/∂a_c) slope[d]`, with
# `slope` overwritten. A kernel's ancestors come before it, so a backward pass
# has every `slope[c]` complete when it is used.
function _stick_adjoint!(
    out::AbstractVector{T},
    s::ScaleCorrelation{T},
    slope::Vector{T},
    rest::Vector{T},
    σ::Vector{T},
) where {T<:Real}
    for c in reverse(eachindex(s.ancestors))
        for p in s.ancestors[c]
            slope[p] -= σ[c] * slope[c]
        end
        out[c] += rest[c] * σ[c] * (1 - σ[c]) * slope[c]
    end
    return out
end

function _add_gradient!(
    grad::AbstractVector{T}, s::ScaleCorrelation{T}, w::AbstractVector, M::AbstractMatrix{T}
) where {T<:Real}
    θ, ρ, rest, σ = _scale_correlation_weights(s, w)
    g = length(θ)
    rootθ = sqrt.(θ)
    traces = T[dot(Z, M) for Z in s.Z]
    slope_θ = traces[1:g]
    slope_ρ = zeros(T, length(ρ))
    for (c, images) in enumerate(s.images)
        A = _kernel_image(images, rootθ)
        MA = M * A
        slope_ρ[c] = dot(A, MA)
        for part in images
            b = part.block
            slope_θ[b] += ρ[c] * (dot(part.image, MA) / rootθ[b] - traces[part.z])
            slope_ρ[c] -= θ[b] * traces[part.z]
        end
    end
    for b in 1:g
        grad[b] += θ[b] * slope_θ[b]
    end
    _stick_adjoint!(view(grad, (g + 1):(g + length(ρ))), s, slope_ρ, rest, σ)
    return grad
end

function _derivatives!(
    D::AbstractArray{T,3}, s::ScaleCorrelation{T}, w::AbstractVector
) where {T<:Real}
    θ, ρ, rest, σ = _scale_correlation_weights(s, w)
    g = length(θ)
    rootθ = sqrt.(θ)
    for b in 1:g
        D[:, :, b] .= s.Z[b]
    end
    # in (θ, ρ)
    for (c, images) in enumerate(s.images)
        A = _kernel_image(images, rootθ)
        Dρ = view(D, :, :, g + c)
        mul!(Dρ, A, A')
        for part in images
            b = part.block
            Zcb = s.Z[part.z]
            WA = part.image * A'
            view(D, :, :, b) .+= ρ[c] .* ((WA .+ WA') ./ (2 * rootθ[b]) .- Zcb)
            Dρ .-= θ[b] .* Zcb
        end
    end
    # to (u, a): as `_stick_adjoint!`, on matrices
    for b in 1:g
        view(D, :, :, b) .*= θ[b]
    end
    for c in reverse(eachindex(s.ancestors))
        Dρ = view(D, :, :, g + c)
        for p in s.ancestors[c]
            view(D, :, :, g + p) .-= σ[c] .* Dρ
        end
        Dρ .*= rest[c] * σ[c] * (1 - σ[c])
    end
    return D
end
