# =============================================================================
# Parameter-level indexing: `Indexed{T}` = `Static{T}` or `Varying{T,G}`.
#
# Any numeric parameter of a state/observation model (A, Q, b, x0, P0, B, C, R,
# d, D) may be one of:
#   * a plain array          — trial-invariant (implicitly static);
#   * a `Static`             — trial-invariant (explicit wrapper);
#   * a `Varying`            — one value per *trial group*, keyed on a label.
#
# `at(param, k)` returns the value for group index `k` (returning the underlying
# array *by reference*, so in-place M-step updates propagate back into the
# `Varying`'s storage). Plain values and `Static`s ignore `k`.
#
# A `Varying` carries the label it keys on (e.g. `:session`) and the ordered
# group ids; the per-trial group index is resolved at fit/rand/smooth time from a
# `labels::Dict{Symbol,<:AbstractVector}` (see `_group_indices` in
# `indexed_fit.jl`). This is defined *before* `lds/types.jl` so the model structs'
# field types and the `_lds_*_dim` traits can refer to `Indexed` and `at`.
# =============================================================================

"""
    Indexed{T}

Abstract supertype for a model parameter whose value may depend on a per-trial
group index. Concrete subtypes are [`Static`](@ref) (one shared value) and
[`Varying`](@ref) (one value per group). `T` is the stored value type (e.g.
`Matrix{Float64}` or `Vector{Float64}`).
"""
abstract type Indexed{T} end

"""
    Static{T} <: Indexed{T}

A trial-invariant parameter holding a single value `val::T`. Equivalent to
passing the plain value, but explicit; `at(::Static, k) === val` for any `k`.
"""
struct Static{T} <: Indexed{T}
    val::T
end

"""
    Varying{T,G} <: Indexed{T}

A trial-varying parameter holding one value per group. `vals[g]` is the value for
group `g`; `label` names the trial-label set this parameter keys on; `group_ids[g]`
is the label value identifying group `g` (so a trial whose label value equals
`group_ids[g]` uses `vals[g]`).

# Fields
- `vals::Vector{T}`: one parameter value per group.
- `label::Symbol`: the trial-label key this parameter is grouped by.
- `group_ids::Vector{G}`: ordered group identifiers (integers or strings).
"""
struct Varying{T,G} <: Indexed{T}
    vals::Vector{T}
    label::Symbol
    group_ids::Vector{G}

    function Varying{T,G}(
        vals::Vector{T}, label::Symbol, group_ids::Vector{G}
    ) where {T,G}
        length(vals) == length(group_ids) || throw(
            ArgumentError(
                "Varying: length(vals)=$(length(vals)) ≠ length(group_ids)=$(length(group_ids))",
            ),
        )
        allunique(group_ids) ||
            throw(ArgumentError("Varying group_ids must be unique; got $(group_ids)"))
        isempty(vals) && throw(ArgumentError("Varying must have at least one group"))
        return new{T,G}(vals, label, group_ids)
    end
end

"""
    Varying(vals, label[, group_ids])

Construct a [`Varying`](@ref) parameter. When `group_ids` is omitted it defaults
to `1:length(vals)`.
"""
function Varying(vals::Vector{T}, label::Symbol, group_ids::AbstractVector{G}) where {T,G}
    return Varying{T,G}(vals, label, collect(group_ids))
end
function Varying(vals::Vector{T}, label::Symbol) where {T}
    return Varying{T,Int}(vals, label, collect(1:length(vals)))
end

"""
    at(param, k::Integer)

Value of `param` for group index `k`. `Varying` indexes into `vals` (returning the
stored array by reference); `Static` and plain values ignore `k`.
"""
at(p::Static, ::Integer) = p.val
at(p::Varying, k::Integer) = p.vals[k]
at(x, ::Integer) = x

"""
    nvals(param) -> Int

Number of groups: `length(vals)` for a `Varying`, `1` for a `Static` or a plain
value.
"""
nvals(::Static) = 1
nvals(p::Varying) = length(p.vals)
nvals(::Any) = 1

"""
    is_indexed(param) -> Bool

`true` if `param` is a [`Static`](@ref) or [`Varying`](@ref) wrapper (as opposed
to a plain array).
"""
is_indexed(::Indexed) = true
is_indexed(::Any) = false

"""
    is_varying(param) -> Bool

`true` only for a [`Varying`](@ref) parameter.
"""
is_varying(::Varying) = true
is_varying(::Any) = false

# Scalar element type of a parameter's stored array(s) (`Float64`, `Int`, …),
# used to infer the model's `T` and to size eltype-preserving defaults.
param_eltype(x::AbstractArray) = eltype(x)
param_eltype(p::Static) = eltype(p.val)
param_eltype(p::Varying) = eltype(first(p.vals))

# The label a parameter keys on (`Symbol`); a sentinel for non-varying params so
# code can compare block labels uniformly.
const _INVARIANT_LABEL = Symbol("")
param_label(::Any) = _INVARIANT_LABEL
param_label(p::Varying) = p.label

# The ordered group ids of a parameter (`[1]` for non-varying params).
param_group_ids(::Any) = Any[1]
param_group_ids(p::Varying) = p.group_ids
