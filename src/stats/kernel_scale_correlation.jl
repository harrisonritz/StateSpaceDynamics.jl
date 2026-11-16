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

A family may instead be one dense kernel shared by its units, each unit with its
own weight (`SharedCorrelationFamily`): a kernel that crosses the units'
borders, such as a surface Matérn kernel with a weight per parcel. On a block
`h` of columns with the dense correlation `M_h`, unit `u(i)` and scale block
`b(i)` of each column, with `c_i = √(θ_{b(i)} ρ_{u(i)})`:

    E_h = diag(c) Gw[:, cols_h]',   F_h = M_h E_h
    Σ_sc += Σ_h E_h' F_h − Σ_{u,b} ρ_u θ_b Z_ub

(the identity part `Z_ub` is a unit's image of zero width, handled with the
factor kernels' `Z_cb`); `E` and `F` are the `n_h² r` part, kept between the
covariance and the gradient at the same weights.
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
    AbstractCorrelationFamily{T}

A family of correlation kernels of a [`ScaleCorrelation`](@ref): a
[`CorrelationFamily`](@ref) (one sparse factor per kernel) or a
`SharedCorrelationFamily` (one dense kernel, a weight per unit). Each has
`type`, `names`, `groups`, `units` (the labels of its weights) and `covered`
(the columns each weight's kernel covers).
"""
abstract type AbstractCorrelationFamily{T<:Real} end

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
struct CorrelationFamily{T<:Real} <: AbstractCorrelationFamily{T}
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

# One block of a shared kernel: the columns, the unit of each, and the dense
# correlation matrix on them.
struct _SharedBlock{T<:Real}
    cols::Vector{Int}
    owner::Vector{Int}
    M::Matrix{T}
end

"""
    SharedCorrelationFamily(blocks; type, names, groups=nothing, units=names, label="")

One dense correlation kernel shared by the units of a family, each unit with
its own weight `ρ_u`: a kernel that crosses the units' borders (a surface
Matérn kernel with a weight per parcel, say), where a
[`CorrelationFamily`](@ref) has one sparse kernel per weight.

`blocks` holds one `(cols, owner, M)` tuple per block of columns (a hemisphere,
say): `cols` the columns of the structure's gain, `owner` the unit (an index
into `names`) of each, and `M` their dense correlation matrix (symmetric, unit
diagonal; positive semidefinite, not checked). The blocks must not share a
column and every unit needs a column. The kernel's part of the correlation
matrix is `D^½ M D^½ − D` on those columns, `D = diag(ρ_{u(i)})`, so that a
column's variance is untouched and the weights covering it keep their
stick-broken sum below one.

`type`, `names` (the units), `groups` and `units` are the labels of the
weights, as for a [`CorrelationFamily`](@ref); `label` names the kernel (its
range, say). A unit's kernel covers its columns, so ancestors and descendants
follow from them as for any family: the units must lie inside one kernel of
every earlier family, and later kernels inside one unit.
"""
struct SharedCorrelationFamily{T<:Real} <: AbstractCorrelationFamily{T}
    type::Symbol
    names::Vector{String}
    groups::Vector{Union{Nothing,String}}
    units::Vector{String}
    blocks::Vector{_SharedBlock{T}}
    covered::Vector{Vector{Int}}                # the columns each unit covers
    label::String
end

function SharedCorrelationFamily(
    blocks::AbstractVector; type::Symbol, names, groups=nothing, units=names, label=""
)
    isempty(blocks) &&
        throw(ArgumentError("the shared correlation family :$type has no block"))
    count = length(names)
    labels, nodes, matched = _weight_labels(type, count, names, groups, units)
    T = float(mapreduce(block -> eltype(block[3]), promote_type, blocks))
    parts = _SharedBlock{T}[]
    seen = Set{Int}()
    for (h, block) in enumerate(blocks)
        length(block) == 3 || throw(
            ArgumentError("a block is a (cols, owner, M) tuple; got $(length(block))")
        )
        cols, owner, M = Vector{Int}(block[1]), Vector{Int}(block[2]), block[3]
        tag = "block $h of the shared correlation family :$type"
        isempty(cols) && throw(ArgumentError("$tag has no column"))
        size(M) == (length(cols), length(cols)) || throw(
            DimensionMismatch(
                "$tag: M is $(size(M)); its $(length(cols)) columns need a square matrix",
            ),
        )
        length(owner) == length(cols) || throw(
            DimensionMismatch(
                "$tag: owner needs one unit per column ($(length(cols))); got " *
                "$(length(owner))",
            ),
        )
        all(>(0), cols) || throw(ArgumentError("$tag: cols are 1-based column indices"))
        all(u -> 1 <= u <= count, owner) || throw(
            ArgumentError(
                "$tag: owner indexes the $count names; got " *
                "$(minimum(owner)) to $(maximum(owner))",
            ),
        )
        all(isfinite, M) || throw(ArgumentError("$tag: M has non-finite entries"))
        maximum(abs, diag(M) .- 1) <= 1e-8 && maximum(abs, M .- M') <= 1e-8 ||
            throw(ArgumentError("$tag: M must be a symmetric correlation matrix"))
        length(unique(cols)) == length(cols) ||
            throw(ArgumentError("$tag: cols repeats a column"))
        any(in(seen), cols) && throw(
            ArgumentError("the blocks of the shared correlation family :$type overlap")
        )
        union!(seen, cols)
        push!(parts, _SharedBlock{T}(cols, owner, Matrix{T}((M .+ M') ./ 2)))
    end
    covered = [sort!(reduce(vcat, [b.cols[b.owner .== u] for b in parts])) for u in 1:count]
    empty = findall(isempty, covered)
    isempty(empty) || throw(
        ArgumentError("every unit needs a column; none for " * _first_few(labels[empty])),
    )
    return SharedCorrelationFamily{T}(
        type, labels, nodes, matched, parts, covered, string(label)
    )
end

function _family_as(::Type{T}, family::SharedCorrelationFamily) where {T<:Real}
    return SharedCorrelationFamily{T}(
        family.type,
        family.names,
        family.groups,
        family.units,
        [_SharedBlock{T}(b.cols, b.owner, Matrix{T}(b.M)) for b in family.blocks],
        family.covered,
        family.label,
    )
end

# The columns a family's kernels reach: for a check against the gain's width.
_family_rows(family::CorrelationFamily) = size(first(family.factors), 1)
function _family_rows(family::SharedCorrelationFamily)
    return maximum(maximum(b.cols) for b in family.blocks)
end

# One block of a spectral kernel: the columns, the unit of each, the modes at
# the columns' dipoles and their eigenvalues.
struct _SpectralBlock{T<:Real}
    cols::Vector{Int}
    owner::Vector{Int}
    U::Matrix{T}
    lam::Vector{T}
end

"""
    SpectralCorrelationFamily(blocks, spectrum; type, names, groups=nothing,
                              units=names, start, free=Symbol[], per_unit=nothing,
                              label="", lower=nothing, upper=nothing)

A surface kernel from the Laplace–Beltrami eigenmodes, shared by the units of a
family as a `SharedCorrelationFamily` is (a weight `ρ_u` per unit, `D^½ M D^½ −
D` on its columns), whose `SurfaceSpectrum` parameters are fitted with
the weights. `blocks` holds one `(cols, owner, U, lam)` per block of columns: the
columns, the unit of each, the modes at the columns' dipoles (`n_h × N`) and
their eigenvalues. The block's correlation is `M = D^(−½)(V V' + diag t)D^(−½)`,
`V_ik = U_ik √s(λ_k; p_u(i))`, `t` Weyl's tail and `d` the diagonal (see
`kernel_spectral.jl`).

`start` holds the spectrum's parameters (log scale, in the order of the kind:
`log_range`, then `log_gamma`); those in `free` are fitted once for the family
and `per_unit` (one parameter) once per unit, as weights of their own after the
logits: of the type `Symbol(type, "_", parameter)`, link `:identity`. A
[`KernelPrior`](@ref) then needs their `prior_mean` (and `prior_sd` when
`:flat`) by type; a per-unit type may be `:nested` in the units' `groups`.
`lower` and `upper` (one per parameter, default unbounded) bound them: past a
bound the spectrum is held at it and the weight has no slope, so its prior
brings the search back. This is dynameg's `SpectralKernel`
(`validate/shared_scale_correlation.py`).
"""
struct SpectralCorrelationFamily{T<:Real} <: AbstractCorrelationFamily{T}
    type::Symbol
    names::Vector{String}
    groups::Vector{Union{Nothing,String}}
    units::Vector{String}
    blocks::Vector{_SpectralBlock{T}}
    covered::Vector{Vector{Int}}
    label::String
    spectrum::SurfaceSpectrum{T}
    start::Vector{T}
    free::Vector{Symbol}
    per_unit::Union{Nothing,Symbol}
    lower::Vector{T}
    upper::Vector{T}
end

function SpectralCorrelationFamily(
    blocks::AbstractVector,
    spectrum::SurfaceSpectrum;
    type::Symbol,
    names,
    groups=nothing,
    units=names,
    start,
    free=Symbol[],
    per_unit=nothing,
    label="",
    lower=nothing,
    upper=nothing,
)
    isempty(blocks) &&
        throw(ArgumentError("the spectral correlation family :$type has no block"))
    count = length(names)
    labels, nodes, matched = _weight_labels(type, count, names, groups, units)
    T = typeof(spectrum.alpha)
    params = _spectrum_params(spectrum)
    length(start) == length(params) || throw(
        DimensionMismatch("start needs $(length(params)) values ($params); got $(length(start))"),
    )
    free = Symbol[Symbol(q) for q in free]
    per_unit = per_unit === nothing ? nothing : Symbol(per_unit)
    for q in (free..., (per_unit === nothing ? () : (per_unit,))...)
        q in params ||
            throw(ArgumentError(":$(spectrum.kind) has the parameters $params; got :$q"))
    end
    allunique(free) && !(per_unit in free) ||
        throw(ArgumentError("a parameter is free twice, or both free and per unit"))
    parts = _SpectralBlock{T}[]
    seen = Set{Int}()
    for (h, block) in enumerate(blocks)
        length(block) == 4 || throw(
            ArgumentError("a block is a (cols, owner, U, lam) tuple; got $(length(block))")
        )
        cols, owner = Vector{Int}(block[1]), Vector{Int}(block[2])
        U, lam = Matrix{T}(block[3]), Vector{T}(block[4])
        tag = "block $h of the spectral correlation family :$type"
        size(U) == (length(cols), length(lam)) && length(owner) == length(cols) ||
            throw(DimensionMismatch("$tag: U, lam, cols and owner do not match"))
        all(>(0), cols) && all(u -> 1 <= u <= count, owner) ||
            throw(ArgumentError("$tag: cols are 1-based, owner indexes the names"))
        all(isfinite, U) && all(>=(0), lam) ||
            throw(ArgumentError("$tag: U must be finite and lam non-negative"))
        any(in(seen), cols) && throw(
            ArgumentError("the blocks of the spectral correlation family :$type overlap")
        )
        union!(seen, cols)
        push!(parts, _SpectralBlock{T}(cols, owner, U, lam))
    end
    covered = [sort!(reduce(vcat, [b.cols[b.owner .== u] for b in parts])) for u in 1:count]
    empty = findall(isempty, covered)
    isempty(empty) || throw(
        ArgumentError("every unit needs a column; none for " * _first_few(labels[empty])),
    )
    return SpectralCorrelationFamily{T}(
        type, labels, nodes, matched, parts, covered, string(label), spectrum,
        Vector{T}(start), free, per_unit,
        lower === nothing ? fill(T(-Inf), length(params)) : Vector{T}(lower),
        upper === nothing ? fill(T(Inf), length(params)) : Vector{T}(upper),
    )
end

function _family_as(::Type{T}, family::SpectralCorrelationFamily) where {T<:Real}
    sp = family.spectrum
    return SpectralCorrelationFamily{T}(
        family.type, family.names, family.groups, family.units,
        [_SpectralBlock{T}(b.cols, b.owner, Matrix{T}(b.U), Vector{T}(b.lam)) for b in family.blocks],
        family.covered, family.label,
        SurfaceSpectrum{T}(sp.kind, T(sp.alpha), Vector{T}(sp.omega), Vector{T}(sp.weight), T(sp.lam_max)),
        Vector{T}(family.start), family.free, family.per_unit,
        Vector{T}(family.lower), Vector{T}(family.upper),
    )
end

function _family_rows(family::SpectralCorrelationFamily)
    return maximum(maximum(b.cols) for b in family.blocks)
end

# The weights a spectral family adds after the logits: `(param index, type,
# names, groups, units)` of its free parameters (one weight each), then of its
# per-unit parameter (one weight per unit).
function _hyper_types(family::SpectralCorrelationFamily)
    params = _spectrum_params(family.spectrum)
    out = Tuple{Int,Symbol,Vector{String},Vector{Union{Nothing,String}},Vector{String}}[]
    for q in family.free
        push!(out, (findfirst(==(q), params)::Int, Symbol(family.type, "_", q),
                    [string(q)], Union{Nothing,String}[nothing], [string(q)]))
    end
    if family.per_unit !== nothing
        q = family.per_unit
        push!(out, (findfirst(==(q), params)::Int, Symbol(family.type, "_", q),
                    family.names, family.groups, family.units))
    end
    return out
end

# What a correlation kernel adds on one scale block: the block, `W_cb`, and the
# index of `Z_cb` in the structure's `Z`.
struct _BlockImage{T<:Real}
    block::Int
    image::Matrix{T}
    z::Int
end

# A block of a shared kernel as a structure uses it: `M`, the rows `W` of
# `Gw[:, cols]'`, the kernel (flat, over all families) and scale block of each
# row, and the rows of each unit and of each scale block.
struct _SharedPart{T<:Real}
    M::Matrix{T}
    W::Matrix{T}
    kernel::Vector{Int}
    scale::Vector{Int}
    by_kernel::Vector{Pair{Int,Vector{Int}}}
    by_scale::Vector{Pair{Int,Vector{Int}}}
end

# A block of a spectral family as a structure uses it: its modes, the rows `W`
# of `Gw[:, cols]'`, the kernel (flat) and scale block of each row, the unit of
# each row and the rows of each unit, and where the family's parameters are:
# `start` (all of them), `free` `(param, slot)` and `unit` `(param, first slot)`
# (`(0, 0)`: none), slots counted in the structure's parameter weights.
struct _SpectralPart{T<:Real}
    spectrum::SurfaceSpectrum{T}
    U::Matrix{T}
    U2::Matrix{T}
    lam::Vector{T}
    W::Matrix{T}
    kernel::Vector{Int}
    scale::Vector{Int}
    owner::Vector{Int}
    n_units::Int
    by_kernel::Vector{Pair{Int,Vector{Int}}}
    by_scale::Vector{Pair{Int,Vector{Int}}}
    by_owner::Vector{Pair{Int,Vector{Int}}}
    start::Vector{T}
    free::Vector{Tuple{Int,Int}}
    unit::Tuple{Int,Int}
    lower::Vector{T}
    upper::Vector{T}
end

# A spectral block's terms at one set of parameters: the unit spectra `S` and
# their roots (one row when no parameter varies by unit), `V = U √s`, the tail
# `t`, the diagonal `d` and `rd = d^(−½)`, `Ẽ = D^(−½) E` and `B = V' Ẽ`.
struct _SpectralAux{T<:Real}
    P::Matrix{T}
    sq::Matrix{T}
    V::Matrix{T}
    t::Vector{T}
    d::Vector{T}
    rd::Vector{T}
    Et::Matrix{T}
    B::Matrix{T}
    uniform::Bool
end

# `E = diag(√(θρ)) W` and `F = M E` of every shared block (and spectral block,
# with its `_SpectralAux`), for the weights `key = [θ ; ρ ; φ]` they were last
# computed at.
mutable struct _SharedTerms{T<:Real}
    key::Vector{T}
    E::Vector{Matrix{T}}
    F::Vector{Matrix{T}}
    SE::Vector{Matrix{T}}
    SF::Vector{Matrix{T}}
    aux::Vector{_SpectralAux{T}}
    const lock::ReentrantLock
end

function _SharedTerms{T}() where {T<:Real}
    return _SharedTerms{T}(
        T[], Matrix{T}[], Matrix{T}[], Matrix{T}[], Matrix{T}[], _SpectralAux{T}[],
        ReentrantLock(),
    )
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
- `families`: the [`CorrelationFamily`](@ref)s, ordered coarse to fine; a
  `SharedCorrelationFamily` (one dense kernel, a weight per unit) may stand
  among them.

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
    families::Vector{AbstractCorrelationFamily{T}}
    # per correlation kernel, families in order: the kernels enclosing it
    ancestors::Vector{Vector{Int}}
    # Z_b of each scale block, then the distinct Z_cb
    Z::Vector{Matrix{T}}
    # per correlation kernel: its image on each scale block it touches
    images::Vector{Vector{_BlockImage{T}}}
    # the Z side by side, each as a column (r² × length(Z)), for one gemv
    Zstack::Matrix{T}
    # per correlation kernel: its columns in the kernels' images side by side
    # (none for a shared family's unit: its image has no width)
    columns::Vector{UnitRange{Int}}
    # the blocks of the shared families, and their `E`, `F` at the last weights
    shared::Vector{_SharedPart{T}}
    # the blocks of the spectral families, and the labels and starting values of
    # their parameter weights (after the logits)
    spectral::Vector{_SpectralPart{T}}
    hyper_labels::Vector{WeightLabel}
    hyper_start::Vector{T}
    terms::_SharedTerms{T}
end

function ScaleCorrelation(
    gain::AbstractMatrix{<:Real},
    variance::AbstractVector{<:Real},
    block::AbstractVector{<:Integer};
    names,
    type::Symbol=:scale,
    groups=nothing,
    units=names,
    families::AbstractVector=CorrelationFamily[],
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
        return half * half'
    end
    powerless = findall(z -> !(tr(z) > 0), Z)
    isempty(powerless) || throw(
        ArgumentError(
            "every scale block needs a positive trace (a column with a variance and " *
            "a gain); none for " *
            _first_few(labels[powerless]),
        ),
    )

    # Any vector of families, so that `families=[]` (a `Vector{Any}`) is no family.
    for family in families
        family isa AbstractCorrelationFamily || throw(
            ArgumentError(
                "families must hold CorrelationFamily or SharedCorrelationFamily " *
                "values; got $(typeof(family))",
            ),
        )
    end
    fams = AbstractCorrelationFamily{T}[_family_as(T, family) for family in families]
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
        rows = _family_rows(family)
        if family isa SharedCorrelationFamily || family isa SpectralCorrelationFamily
            rows <= n || throw(
                DimensionMismatch(
                    "the shared correlation family :$(family.type) has a column " *
                    "$rows; gain has $n columns",
                ),
            )
        else
            rows == n || throw(
                DimensionMismatch(
                    "the factors of the correlation family :$(family.type) have " *
                    "$rows rows; gain has $n columns",
                ),
            )
        end
    end
    ancestors = _correlation_ancestors(fams, n)

    block = Vector{Int}(block)
    slot = Dict{Tuple{Int,Vector{Int}},Int}((b, members[b]) => b for b in 1:g)
    images = Vector{_BlockImage{T}}[]
    shared = _SharedPart{T}[]
    spectral = _SpectralPart{T}[]
    hyper_labels = WeightLabel[]
    hyper_start = T[]
    for family in fams
        before = length(images)
        if family isa CorrelationFamily
            for (B, cols) in zip(family.factors, family.covered)
                push!(images, _block_images!(Z, slot, Gw, B, cols, block))
            end
        elseif family isa SpectralCorrelationFamily
            for cols in family.covered
                push!(images, _block_images!(Z, slot, Gw, nothing, cols, block))
            end
            free, unit = Tuple{Int,Int}[], (0, 0)
            for (j, htype, hnames, hgroups, hunits) in _hyper_types(family)
                first_slot = length(hyper_labels) + 1
                for (name, group, u) in zip(hnames, hgroups, hunits)
                    push!(hyper_labels, WeightLabel(name, htype, group, u, :identity))
                    push!(hyper_start, family.start[j])
                end
                if Symbol(family.type, "_", family.per_unit) === htype
                    unit = (j, first_slot)
                else
                    push!(free, (j, first_slot))
                end
            end
            for part in family.blocks
                push!(spectral, _spectral_part(part, family, before, Gw, block, free, unit))
            end
        else
            for cols in family.covered
                push!(images, _block_images!(Z, slot, Gw, nothing, cols, block))
            end
            for part in family.blocks
                push!(shared, _shared_part(part, before, Gw, block))
            end
        end
    end
    widths = [size(first(parts).image, 2) for parts in images]
    stops = cumsum(widths)
    columns = [(stop - width + 1):stop for (stop, width) in zip(stops, widths)]
    # Filled in place: `reduce(hcat, …)` over a generator copies pairwise,
    # quadratic in the number of `Z` (gigabytes for a parcellation).
    Zstack = Matrix{T}(undef, r * r, length(Z))
    for (j, z) in enumerate(Z)
        Zstack[:, j] .= vec(z)
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
        Zstack,
        columns,
        shared,
        spectral,
        hyper_labels,
        hyper_start,
        _SharedTerms{T}(),
    )
end

# For each correlation kernel (families in order), the kernels of the earlier
# families that enclose it. A kernel must lie inside one kernel of an earlier
# family or outside all of them.
function _correlation_ancestors(
    families::AbstractVector{<:AbstractCorrelationFamily}, n::Int
)
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

# The rows of each distinct value of `ids`, as `value => rows`, by value.
function _group_rows(ids::Vector{Int})
    rows = Dict{Int,Vector{Int}}()
    for (i, id) in enumerate(ids)
        push!(get!(rows, id, Int[]), i)
    end
    return sort!(collect(rows); by=first)
end

# A shared block's `W = Gw[:, cols]'` and the groupings of its rows; `before` is
# the number of kernels before the family's first unit.
function _shared_part(
    part::_SharedBlock{T}, before::Int, Gw::Matrix{T}, block::Vector{Int}
) where {T<:Real}
    kernel = before .+ part.owner
    scale = block[part.cols]
    return _SharedPart{T}(
        part.M,
        Matrix{T}(transpose(Gw[:, part.cols])),
        kernel,
        scale,
        _group_rows(kernel),
        _group_rows(scale),
    )
end

function _spectral_part(
    part::_SpectralBlock{T}, family::SpectralCorrelationFamily{T}, before::Int,
    Gw::Matrix{T}, block::Vector{Int}, free::Vector{Tuple{Int,Int}}, unit::Tuple{Int,Int},
) where {T<:Real}
    kernel = before .+ part.owner
    scale = block[part.cols]
    return _SpectralPart{T}(
        family.spectrum, part.U, part.U .^ 2, part.lam,
        Matrix{T}(transpose(Gw[:, part.cols])), kernel, scale, part.owner,
        length(family.names), _group_rows(kernel), _group_rows(scale),
        _group_rows(part.owner), family.start, free, unit, family.lower, family.upper,
    )
end

# The images of one kernel (unit-row factor `B`, covering `cols`) on the scale
# blocks it touches. A `Z_cb` not yet in `Z` is appended; `slot` finds those
# already there by their block and columns. A shared family's unit has no
# factor (`B = nothing`): its images have no width, only the `Z_cb`.
function _block_images!(
    Z::Vector{Matrix{T}},
    slot::Dict{Tuple{Int,Vector{Int}},Int},
    Gw::Matrix{T},
    B::Union{Nothing,SparseMatrixCSC{T,Int}},
    cols::Vector{Int},
    block::Vector{Int},
) where {T<:Real}
    touched = sort!(unique(view(block, cols)))
    position = Dict(b => k for (k, b) in enumerate(touched))
    W = [zeros(T, size(Gw, 1), B === nothing ? 0 : size(B, 2)) for _ in touched]
    if B !== nothing
        rows, values = rowvals(B), nonzeros(B)
        for j in axes(B, 2), k in nzrange(B, j)
            i = rows[k]
            axpy!(values[k], view(Gw, :, i), view(W[position[block[i]]], :, j))
        end
    end
    within = [Int[] for _ in touched]
    for i in cols
        push!(within[position[block[i]]], i)
    end
    return map(enumerate(touched)) do (k, b)
        z = get!(slot, (b, within[k])) do
            half = Gw[:, within[k]]
            push!(Z, half * half')
            return length(Z)
        end
        return _BlockImage{T}(b, W[k], z)
    end
end

_scale_count(s::ScaleCorrelation) = length(s.names)

_structure_dim(s::ScaleCorrelation) = size(s.gain, 1)

_hyper_count(s::ScaleCorrelation) = length(s.hyper_labels)

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
    return append!(labels, s.hyper_labels)
end

function _structure_traces(s::ScaleCorrelation{T}) where {T<:Real}
    g = _scale_count(s)
    return vcat(
        T[tr(s.Z[b]) for b in 1:g], fill(T(NaN), length(s.ancestors) + _hyper_count(s))
    )
end

# The logits start where every correlation weight is `_START_RHO`.
function _structure_start(s::ScaleCorrelation{T}, logscale::Real) where {T<:Real}
    logits = map(s.ancestors) do enclosing
        rest = max(1 - _START_RHO * length(enclosing), 1e-6)
        return T(_logit(_START_RHO / rest))
    end
    return vcat(fill(T(logscale), _scale_count(s)), logits, s.hyper_start)
end

_sigmoid(a::Real) = (1 + tanh(a / 2)) / 2

# `θ` of the scale blocks, and of the correlation kernels `ρ`, the remainder
# `1 − Σ_ancestors ρ` each breaks and `σ(a)`.
function _scale_correlation_weights(
    s::ScaleCorrelation{T}, w::AbstractVector
) where {T<:Real}
    g, nc = _scale_count(s), length(s.ancestors)
    nw = g + nc + _hyper_count(s)
    length(w) == nw ||
        throw(DimensionMismatch("the structure has $nw weights; got $(length(w))"))
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
    return vcat(θ, ρ, _hyper_weights(s, w))
end

# The spectral parameters: the weights after the logits.
function _hyper_weights(s::ScaleCorrelation{T}, w::AbstractVector) where {T<:Real}
    offset = _scale_count(s) + length(s.ancestors)
    return T[w[i] for i in (offset + 1):length(w)]
end

# `A_c = Σ_b √θ_b W_cb` of one kernel.
function _kernel_image(images::Vector{_BlockImage{T}}, rootθ::Vector{T}) where {T<:Real}
    A = zeros(T, size(first(images).image))
    for part in images
        A .+= rootθ[part.block] .* part.image
    end
    return A
end

#= Splitting large BLAS calls over Julia's threads. Only when BLAS itself runs
on one thread (as when it is pinned to avoid nesting under Julia threads) and
Julia has several: otherwise the call is left whole, to BLAS. =#
_split_blas() = Threads.nthreads() > 1 && BLAS.get_num_threads() == 1

# The shortest block a split call is given (a `Ref` so that tests can lower it).
const _SPLIT_LEAST = Ref(256)

# `n` indices in about one contiguous range per thread, none shorter than `least`.
function _chunks(n::Int; least::Int=_SPLIT_LEAST[])
    parts = clamp(n ÷ least, 1, Threads.nthreads())
    edges = round.(Int, range(0, n; length=parts + 1))
    return [(edges[i] + 1):edges[i + 1] for i in 1:parts]
end

# `out = X v` (`transpose=false`) or `X' v`, by blocks of rows or columns of `X`.
function _split_gemv!(
    out::AbstractVector{T},
    X::AbstractMatrix{T},
    v::AbstractVector{T};
    transpose::Bool=false,
    add::Bool=false,
) where {T<:Real}
    β = add ? one(T) : zero(T)
    if !_split_blas()
        transpose ? mul!(out, X', v, one(T), β) : mul!(out, X, v, one(T), β)
        return out
    end
    tforeach(_chunks(length(out))) do part
        if transpose
            mul!(view(out, part), view(X, :, part)', v, one(T), β)
        else
            mul!(view(out, part), view(X, part, :), v, one(T), β)
        end
    end
    return out
end

# Every kernel's `scale[c] A_c`, side by side (r × total width), in `s.columns`.
function _stacked_images(
    s::ScaleCorrelation{T}, rootθ::Vector{T}, scale::AbstractVector{T}
) where {T<:Real}
    r = size(s.gain, 1)
    width = isempty(s.columns) ? 0 : last(last(s.columns))
    A = zeros(T, r, width)
    fill_kernel = c -> begin
        Ac = view(A, :, s.columns[c])
        for part in s.images[c]
            axpy!(scale[c] * rootθ[part.block], part.image, Ac)
        end
    end
    if _split_blas()
        tforeach(fill_kernel, eachindex(s.images))
    else
        foreach(fill_kernel, eachindex(s.images))
    end
    return A
end

# The lower triangle of `A A'`, by blocks of columns of `A` when split.
function _split_syrk(A::AbstractMatrix{T}) where {T<:Real}
    _split_blas() || return BLAS.syrk('L', 'N', one(T), A)
    return tmapreduce(+, _chunks(size(A, 2))) do part
        return BLAS.syrk('L', 'N', one(T), view(A, :, part))
    end
end

# `M A`, by blocks of columns of `A` when split.
function _split_mul(M::AbstractMatrix{T}, A::AbstractMatrix{T}) where {T<:Real}
    _split_blas() || return M * A
    out = similar(A, size(M, 1), size(A, 2))
    tforeach(_chunks(size(A, 2))) do part
        return mul!(view(out, :, part), M, view(A, :, part))
    end
    return out
end

# `M E`, by blocks of rows of `M` when split (a shared block is `n_h² r`).
function _split_row_mul(M::AbstractMatrix{T}, E::AbstractMatrix{T}) where {T<:Real}
    _split_blas() || return M * E
    out = similar(E, size(M, 1), size(E, 2))
    tforeach(_chunks(size(M, 1))) do part
        return mul!(view(out, part, :), view(M, part, :), E)
    end
    return out
end

# `E` and `F = M E` of every shared block at the weights `θ`, `ρ`: the costly
# part (`n_h² r` a block), kept for the covariance and the gradient that follow
# at the same weights.
function _shared_terms(
    s::ScaleCorrelation{T}, θ::Vector{T}, ρ::Vector{T}, φ::Vector{T}=T[]
) where {T<:Real}
    key = vcat(θ, ρ, φ)
    cache = s.terms
    return lock(cache.lock) do
        if cache.key != key
            root = sqrt.(max.(ρ, zero(T)))
            E = [part.W .* (sqrt.(θ[part.scale]) .* root[part.kernel]) for part in s.shared]
            F = [_split_row_mul(part.M, e) for (part, e) in zip(s.shared, E)]
            SE = [part.W .* (sqrt.(θ[part.scale]) .* root[part.kernel]) for part in s.spectral]
            aux = [_spectral_aux(part, φ, e) for (part, e) in zip(s.spectral, SE)]
            SF = [(_spectral_v(part, a, a.B) .+ a.t .* a.Et) .* a.rd
                  for (part, a) in zip(s.spectral, aux)]
            cache.E, cache.F, cache.key = E, F, key
            cache.SE, cache.SF, cache.aux = SE, SF, aux
        end
        return cache.E, cache.F
    end
end

# The parameter rows of a spectral part's units (one row when none varies by unit).
function _part_params(part::_SpectralPart{T}, φ::AbstractVector) where {T<:Real}
    uniform = part.unit[1] == 0
    P = repeat(reshape(part.start, 1, :), uniform ? 1 : part.n_units, 1)
    for (j, slot) in part.free
        P[:, j] .= clamp(φ[slot], part.lower[j], part.upper[j])
    end
    if !uniform
        j, first_slot = part.unit
        for u in 1:part.n_units
            P[u, j] = clamp(φ[first_slot + u - 1], part.lower[j], part.upper[j])
        end
    end
    return P, uniform
end

# Whether a parameter weight is inside its bounds (outside, the spectrum is held
# at the bound: no slope).
_inside(part::_SpectralPart, j::Int, value::Real) = part.lower[j] < value < part.upper[j]

function _spectral_aux(part::_SpectralPart{T}, φ::AbstractVector, E::Matrix{T}) where {T<:Real}
    P, uniform = _part_params(part, φ)
    N = length(part.lam)
    sq = zeros(T, size(P, 1), N)
    t_u = zeros(T, size(P, 1))
    for r in axes(P, 1)
        values, _ = _spectrum(part.spectrum, part.lam, view(P, r, :))
        sq[r, :] .= sqrt.(values)
        t_u[r] = first(_spectrum_tail(part.spectrum, part.lam[end], view(P, r, :)))
    end
    rd_of(d) = 1 ./ sqrt.(d)
    if uniform  # V = U diag(√s), never formed: products through U
        s0 = vec(sq[1, :])
        t = fill(t_u[1], length(part.owner))
        d = part.U2 * (s0 .^ 2) .+ t
        rd = rd_of(d)
        Et = E .* rd
        return _SpectralAux{T}(P, sq, zeros(T, 0, 0), t, d, rd, Et, s0 .* (part.U' * Et), true)
    end
    V = part.U .* view(sq, part.owner, :)
    t = t_u[part.owner]
    d = vec(sum(abs2, V; dims=2)) .+ t
    rd = rd_of(d)
    Et = E .* rd
    return _SpectralAux{T}(P, sq, V, t, d, rd, Et, V' * Et, false)
end

# `V Y` of a spectral block.
function _spectral_v(part::_SpectralPart, aux::_SpectralAux, Y::AbstractMatrix)
    aux.uniform && return part.U * (vec(aux.sq[1, :]) .* Y)
    return aux.V * Y
end

# Per unit row, `∂√s/∂p` (N × n_p), `∂s/∂p` and `∂t/∂p` of a spectral part.
function _spectral_slopes(part::_SpectralPart{T}, aux::_SpectralAux{T}) where {T<:Real}
    half, dS, dt = Matrix{T}[], Matrix{T}[], Vector{T}[]
    for r in axes(aux.P, 1)
        _, ds = _spectrum(part.spectrum, part.lam, view(aux.P, r, :); grad=true)
        push!(dS, ds)
        push!(half, ds ./ (2 .* max.(view(aux.sq, r, :), floatmin(T))))
        push!(dt, last(_spectrum_tail(part.spectrum, part.lam[end], view(aux.P, r, :))))
    end
    return half, dS, dt
end

# `(param, slot, rows)` of every fitted parameter weight of a spectral part:
# a free one over all its rows, a per-unit one over each unit's rows.
function _hyper_slots(part::_SpectralPart)
    out = Tuple{Int,Int,Vector{Int},Int}[]
    all_rows = collect(1:length(part.owner))
    for (j, slot) in part.free
        push!(out, (j, slot, all_rows, 0))
    end
    if part.unit[1] != 0
        j, first_slot = part.unit
        for (u, rows) in part.by_owner
            push!(out, (j, first_slot + u - 1, rows, u))
        end
    end
    return out
end

# The coefficient of each `Z` in `Σ_sc`: `θ_b` on the blocks, `−Σ ρ_c θ_b` on
# the `Z_cb`.
function _z_coefficients(s::ScaleCorrelation{T}, θ::Vector{T}, ρ::Vector{T}) where {T<:Real}
    coefficient = zeros(T, length(s.Z))
    coefficient[1:length(θ)] .= θ
    for (c, images) in enumerate(s.images)
        for part in images
            coefficient[part.z] -= ρ[c] * θ[part.block]
        end
    end
    return coefficient
end

#= `Σ_sc` is one `syrk` of the stacked images `√ρ_c A_c` and one `gemv` of the
stacked `Z`: a call per kernel would spend most of its time checking and
copying triangles of `r × r` matrices. =#
function _add_covariance!(
    Σ::AbstractMatrix{T}, s::ScaleCorrelation{T}, w::AbstractVector
) where {T<:Real}
    θ, ρ, _, _ = _scale_correlation_weights(s, w)
    φ = _hyper_weights(s, w)
    rootθ = sqrt.(θ)
    if !isempty(s.columns)
        A = _stacked_images(s, rootθ, sqrt.(max.(ρ, zero(T))))
        Σ .+= Symmetric(_split_syrk(A), :L)
    end
    _split_gemv!(vec(Σ), s.Zstack, _z_coefficients(s, θ, ρ); add=true)
    if !isempty(s.shared) || !isempty(s.spectral)
        E, F = _shared_terms(s, θ, ρ, φ)
        X = zeros(T, size(Σ))
        for (e, f) in zip(E, F)
            mul!(X, e', f, one(T), one(T))
        end
        for (e, f) in zip(s.terms.SE, s.terms.SF)
            mul!(X, e', f, one(T), one(T))
        end
        Σ .+= (X .+ X') ./ 2
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
    φ = _hyper_weights(s, w)
    g = length(θ)
    rootθ = sqrt.(θ)
    traces = _split_gemv!(zeros(T, length(s.Z)), s.Zstack, vec(M); transpose=true)
    slope_θ = traces[1:g]
    slope_ρ = zeros(T, length(ρ))
    if !isempty(s.columns)
        A = _stacked_images(s, rootθ, ones(T, length(ρ)))
        MA = _split_mul(M, A)
        for (c, images) in enumerate(s.images)
            cols = s.columns[c]
            MAc = view(MA, :, cols)
            slope_ρ[c] = dot(view(A, :, cols), MAc)
            for part in images
                b = part.block
                slope_θ[b] += ρ[c] * (dot(part.image, MAc) / rootθ[b] - traces[part.z])
                slope_ρ[c] -= θ[b] * traces[part.z]
            end
        end
    end
    if !isempty(s.shared)
        # ∂ tr(G E'ME)/∂E_i = 2 (F G)_i, and E_i ∝ √(θ_b ρ_u): with
        # p_i = (F G)_i · E_i, ∂/∂ρ_u = Σ_{i∈u} p_i/ρ_u and ∂/∂θ_b alike
        E, F = _shared_terms(s, θ, ρ, φ)
        for (part, e, f) in zip(s.shared, E, F)
            p = vec(sum((f * M) .* e; dims=2))
            for (i, value) in enumerate(p)
                slope_ρ[part.kernel[i]] += value / max(ρ[part.kernel[i]], 1e-300)
                slope_θ[part.scale[i]] += value / max(θ[part.scale[i]], 1e-300)
            end
        end
    end
    if !isempty(s.spectral)
        _shared_terms(s, θ, ρ, φ)
        offset = g + length(ρ)
        for (part, e, f, aux) in zip(s.spectral, s.terms.SE, s.terms.SF, s.terms.aux)
            p = vec(sum((f * M) .* e; dims=2))
            for (i, value) in enumerate(p)
                slope_ρ[part.kernel[i]] += value / max(ρ[part.kernel[i]], 1e-300)
                slope_θ[part.scale[i]] += value / max(θ[part.scale[i]], 1e-300)
            end
            slots = _hyper_slots(part)
            isempty(slots) && continue
            # M = D^(−½)(VV' + T)D^(−½): through V and t at fixed d, then d
            EtG = aux.Et * M
            pd = p ./ aux.d
            gt = vec(sum(EtG .* aux.Et; dims=2)) .- pd
            half, _, dt = _spectral_slopes(part, aux)
            if aux.uniform  # free parameters only: Σ_i gV_ik U_ik by BLAS
                s0 = vec(aux.sq[1, :])
                a = 2 .* vec(sum(aux.B .* (part.U' * EtG); dims=2)) .- 2 .* s0 .* (part.U2' * pd)
                for (j, slot, _, _) in slots
                    _inside(part, j, φ[slot]) || continue
                    grad[offset + slot] += dot(a, view(half[1], :, j)) + sum(gt) * dt[1][j]
                end
                continue
            end
            gV = 2 .* (EtG * aux.B') .- 2 .* aux.V .* pd
            gVU = gV .* part.U
            for (j, slot, rows, u) in slots
                _inside(part, j, φ[slot]) || continue
                r = aux.uniform ? 1 : u
                if u == 0 && !aux.uniform  # a free parameter, per-unit spectra
                    for (unit, urows) in part.by_owner
                        a = vec(sum(view(gVU, urows, :); dims=1))
                        grad[offset + slot] += dot(a, view(half[unit], :, j)) +
                                               sum(view(gt, urows)) * dt[unit][j]
                    end
                else
                    a = vec(sum(view(gVU, rows, :); dims=1))
                    grad[offset + slot] += dot(a, view(half[r], :, j)) +
                                           sum(view(gt, rows)) * dt[r][j]
                end
            end
        end
    end
    for b in 1:g
        grad[b] += θ[b] * slope_θ[b]
    end
    _stick_adjoint!(view(grad, (g + 1):(g + length(ρ))), s, slope_ρ, rest, σ)
    return grad
end

# A spectral family's part of `∂Σ/∂(θ, ρ)` (as a shared family's) and its
# `∂Σ/∂φ`: with `Q = ∂V B` and `Δ = ∂d/d` on the rows that move,
# `Q'Ẽ + Ẽ'Q + Ẽ' ∂T Ẽ − ½ (E'ΔF + F'ΔE)`.
function _spectral_derivatives!(
    D::AbstractArray{T,3}, s::ScaleCorrelation{T}, θ::Vector{T}, ρ::Vector{T},
    φ::Vector{T},
) where {T<:Real}
    g = length(θ)
    offset = g + length(ρ)
    for k in (offset + 1):(offset + _hyper_count(s))
        fill!(view(D, :, :, k), zero(T))
    end
    _shared_terms(s, θ, ρ, φ)
    X = zeros(T, size(D, 1), size(D, 2))
    for (part, e, f, aux) in zip(s.spectral, s.terms.SE, s.terms.SF, s.terms.aux)
        for (groups, weights, off) in ((part.by_kernel, ρ, g), (part.by_scale, θ, 0))
            for (id, rows) in groups
                mul!(X, view(e, rows, :)', view(f, rows, :))
                scale = 1 / (2 * max(weights[id], 1e-300))
                view(D, :, :, off + id) .+= scale .* (X .+ X')
            end
        end
        slots = _hyper_slots(part)
        isempty(slots) && continue
        half, dS, dt = _spectral_slopes(part, aux)
        for (j, slot, rows, _) in slots
            _inside(part, j, φ[slot]) || continue
            if aux.uniform  # one spectrum: through BLAS
                dti = fill(dt[1][j], length(rows))
                Q = view(part.U, rows, :) * (view(half[1], :, j) .* aux.B)
                delta = (view(part.U2, rows, :) * view(dS[1], :, j) .+ dti) ./ aux.d[rows]
            else
                row = part.owner[rows]
                H = reduce(vcat, (view(half[r], :, j)' for r in row))
                dSj = reduce(vcat, (view(dS[r], :, j)' for r in row))
                dti = T[dt[r][j] for r in row]
                Q = (view(part.U, rows, :) .* H) * aux.B
                delta = (vec(sum(view(part.U2, rows, :) .* dSj; dims=2)) .+ dti) ./ aux.d[rows]
            end
            Et = aux.Et[rows, :]
            XQ = Q' * Et
            Y = (e[rows, :] .* delta)' * f[rows, :]
            view(D, :, :, offset + slot) .+= XQ .+ XQ' .+ (Et .* dti)' * Et .- (Y .+ Y') ./ 2
        end
    end
    return D
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
        size(A, 2) == 0 ? fill!(Dρ, zero(T)) : mul!(Dρ, A, A')
        for part in images
            b = part.block
            Zcb = s.Z[part.z]
            WA = part.image * A'
            view(D, :, :, b) .+= ρ[c] .* ((WA .+ WA') ./ (2 * rootθ[b]) .- Zcb)
            Dρ .-= θ[b] .* Zcb
        end
    end
    if !isempty(s.shared)
        # ∂(E'ME)/∂w = Σ_{i∈w} (E_i F_i' + F_i E_i') / 2w, for a unit's ρ and a
        # scale block's θ alike
        E, F = _shared_terms(s, θ, ρ, _hyper_weights(s, w))
        X = zeros(T, size(D, 1), size(D, 2))
        for (part, e, f) in zip(s.shared, E, F)
            for (groups, weights, offset) in ((part.by_kernel, ρ, g), (part.by_scale, θ, 0))
                for (id, rows) in groups
                    mul!(X, view(e, rows, :)', view(f, rows, :))
                    scale = 1 / (2 * max(weights[id], 1e-300))
                    view(D, :, :, offset + id) .+= scale .* (X .+ X')
                end
            end
        end
    end
    if !isempty(s.spectral)
        _spectral_derivatives!(D, s, θ, ρ, _hyper_weights(s, w))
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
