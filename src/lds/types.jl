"""
    AbstractStateModel{T<:Real}

Abstract supertype for latent-state models of a [`LinearDynamicalSystem`](@ref)
(e.g. [`GaussianStateModel`](@ref)). A state model defines how the latent state
evolves from one timestep to the next.

Every concrete subtype carries a `depends_on` field declaring which of its
parameters are estimated separately per group of trials, and a companion
`variants` field holding one model object per parameter-group combination.
See `src/lds/parameter_groups.jl` for the full description.
"""
abstract type AbstractStateModel{T<:Real} end

"""
    AbstractObservationModel{T<:Real}

Abstract supertype for observation (emission) models of a
[`LinearDynamicalSystem`](@ref) (e.g. [`GaussianObservationModel`](@ref) or
[`PoissonObservationModel`](@ref)). An observation model defines how observed
data are generated from the latent state.

Every concrete subtype carries a `depends_on` field declaring which of its
parameters are estimated separately per group of trials, and a companion
`variants` field holding one model object per parameter-group combination.
See `src/lds/parameter_groups.jl` for the full description.
"""
abstract type AbstractObservationModel{T<:Real} end

"""
    Data{T<:Real}

**Internal** container for a normalized, validated multi-trial dataset:
per-trial observations `y`, dynamics inputs `ux`, and observation inputs `uy`
(each a vector of `(dim, T_i)` matrices; input matrices have zero rows when
the model takes no inputs), plus the per-trial lengths `tsteps`.

Not part of the public API. Public entry points (`fit!`, `smooth`,
`loglikelihood`) accept plain arrays — a `(obs_dim, T)` matrix, a
`(obs_dim, T, ntrials)` array, or a vector of per-trial matrices — and
construct a `Data` via the validating constructor, which is the single
shape/dimension validation site. Everything downstream of a `Data` may
assume consistent, model-compatible shapes.

For a [`CompositeObservationModel`](@ref), `y` and `uy` are instead
`NamedTuple`s keyed by observation model, each value being that model's own
vector of per-trial matrices. `ux` and `tsteps` are shared, so every state-side
consumer of a `Data` is identical in both cases — which is why `YV` and `UYV`
carry no bound.

See also [`Data(lds, y; ux, uy)`](@ref), the validating constructor (below).
"""
struct Data{T<:Real,YV,UXV<:AbstractVector{<:AbstractMatrix{T}},UYV}
    y::YV
    ux::UXV
    uy::UYV
    tsteps::Vector{Int}
end

"""
    GaussianStateModel{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}

Represents the state model of a Linear Dynamical System with Gaussian noise.

State evolution:
```math
x_1           ~ N(x_0, P_0)
x_{t+1} | x_t ~ N(A x_t + b + B ux_t, Q)
```
where `B·ux_t` is present only when `B` is supplied (i.e., has nonzero columns).

# Fields
- `A::M`: Transition matrix (size `latent_dim × latent_dim`).
- `Q::M`: Process noise covariance matrix.
- `b::V`: Bias vector (length `latent_dim`).
- `x0::V`: Initial state mean (length `latent_dim`).
- `P0::M`: Initial state covariance (size `latent_dim × latent_dim`).
- `B::M`: Optional dynamics input matrix (`latent_dim × ux_dim`).
    When supplied, inputs `ux` must be passed to `fit!`/`smooth!` via a keyword argument.
- `Q_prior::Union{Nothing,IWPrior{T}} = nothing`: Optional Inverse-Wishart prior on `Q`. If set, MAP updates use its mode.
- `P0_prior::Union{Nothing,IWPrior{T}} = nothing`: Optional Inverse-Wishart prior on `P0`. If set, MAP updates use its mode.
- `AB_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing`: Optional matrix-normal prior on
    the stacked dynamics matrix `[A B]`. Pair with `Q_prior` for a full MNIW prior on `(AB, Q)`.
    Prior matrices are stored as plain `Matrix{T}` (decoupled from `A`'s storage type `M`) so
    they match the internal workspaces regardless of how `A` is stored.
- `x0_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing`: Optional matrix-normal prior on the
    initial mean `x0`.
- `depends_on::Union{Nothing,NamedTuple} = nothing`: Optional declaration that some
    parameters are estimated separately per group of trials. Keys are parameter names
    (`:x0`, `:P0`, `:A`/`:b`/`:B`, `:Q`), values are per-trial label vectors; `nothing`
    (the default) means one parameter set shared by every trial.
- `variants::Union{Nothing,Vector{GaussianStateModel{T,M,V}}} = nothing`: Derived storage
    for the per-group parameter sets; populated from `depends_on` by the fitting entry
    points. Parameters that do not vary are shared **by reference** across variants, so
    an M-step write through any variant is visible from all of them.
"""
Base.@kwdef mutable struct GaussianStateModel{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}
} <: AbstractStateModel{T}
    A::M
    Q::M
    b::V
    x0::V
    P0::M
    B::M = zeros(eltype(A), size(A, 1), 0)
    Q_prior::Union{Nothing,IWPrior{T}} = nothing
    P0_prior::Union{Nothing,IWPrior{T}} = nothing
    AB_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing
    x0_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing
    depends_on::Union{Nothing,NamedTuple} = nothing
    variants::Union{Nothing,Vector{GaussianStateModel{T,M,V}}} = nothing
end

"""
    GaussianObservationModel{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}

Represents the observation model of a Linear Dynamical System with Gaussian noise.

# Fields
- `C::M`: Observation matrix of size `(obs_dim × latent_dim)`. Maps latent states into
    observation space.
- `R::M`: Observation noise covariance of size `(obs_dim × obs_dim)`.
- `d::V`: Bias vector of length `(obs_dim)`.
- `R_prior::Union{Nothing, IWPrior{T}} = nothing`: Optional Inverse-Wishart prior for `R`.
- `CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing`: Optional matrix-normal prior on
    the stacked emission matrix `[C D]`. Pair with `R_prior` for a full MNIW prior on `(CD, R)`.
    Prior matrices are stored as plain `Matrix{T}` (decoupled from `C`'s storage type `M`) so
    they match the internal workspaces regardless of how `C` is stored.
- `depends_on::Union{Nothing,NamedTuple} = nothing`: Optional declaration that some
    parameters are estimated separately per group of trials. Keys are parameter names
    (`:C`/`:d`/`:D`, `:R`), values are per-trial label vectors; `nothing` (the default)
    means one parameter set shared by every trial.
- `group_seeds::Union{Nothing,AbstractDict} = nothing`: Optional starting values for
    individual `depends_on` groups, as `label => (C=..., R=..., d=..., D=...)` with any
    subset of those keys. Only the groups a label names are seeded; the rest keep the
    defaults described under `variants`. Set it with [`set_group_seeds!`](@ref), which
    checks the labels.
- `variants::Union{Nothing,Vector{GaussianObservationModel{T,M,V}}} = nothing`: Derived
    storage for the per-group parameter sets; populated from `depends_on` by the fitting
    entry points. Parameters that do not vary are shared **by reference** across variants.
"""
Base.@kwdef mutable struct GaussianObservationModel{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}
} <: AbstractObservationModel{T}
    C::M
    R::M
    d::V
    D::M = zeros(eltype(C), size(C, 1), 0)  # eltype-preserving default
    R_prior::Union{Nothing,IWPrior{T}} = nothing
    CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing
    depends_on::Union{Nothing,NamedTuple} = nothing
    group_seeds::Union{Nothing,AbstractDict} = nothing
    variants::Union{Nothing,Vector{GaussianObservationModel{T,M,V}}} = nothing
end

# Convenience constructors (State)
function GaussianStateModel(
    A::M, Q::M, b::V, x0::V, P0::M
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    return GaussianStateModel{T,M,V}(;
        A=A,
        Q=Q,
        b=b,
        x0=x0,
        P0=P0,
        B=zeros(T, size(A, 1), 0),
        Q_prior=nothing,
        P0_prior=nothing,
        AB_prior=nothing,
    )
end

function GaussianStateModel(A::M, Q::M, B::M, P0::M) where {T<:Real,M<:AbstractMatrix{T}}
    return GaussianStateModel{T,M,Vector{T}}(;
        A=A,
        Q=Q,
        b=zeros(T, size(A, 1)),
        x0=zeros(T, size(A, 1)),
        P0=P0,
        B=B,
        Q_prior=nothing,
        P0_prior=nothing,
        AB_prior=nothing,
    )
end

# Convenience constructors (Observation)

function GaussianObservationModel(
    C::M, R::M, d::V
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    return GaussianObservationModel{T,M,V}(;
        C=C, R=R, d=d, D=zeros(eltype(C), size(C, 1), 0), R_prior=nothing, CD_prior=nothing
    )
end

function GaussianObservationModel(C::M, R::M, D::M) where {T<:Real,M<:AbstractMatrix{T}}
    return GaussianObservationModel{T,M,Vector{T}}(;
        C=C, R=R, d=zeros(T, size(C, 1)), D=D, R_prior=nothing, CD_prior=nothing
    )
end

"""
    PoissonObservationModel{
        T<:Real,
        M<:AbstractMatrix{T},
        V<:AbstractVector{T}
    } <: AbstractObservationModel{T}

Represents the observation model of a Linear Dynamical System with Poisson observations,
with canonical log-link:

```math
λ_t = exp(C x_t + d + D v_t)
```

`d` is the standard Poisson-GLM intercept — the per-channel baseline log-rate,
unconstrained in ℝ; positivity of the rate `λ` is provided by the `exp`. `D v_t`
is an optional observation-input (covariate) term; when `D` has zero columns the
model reduces to the canonical `λ_t = exp(C x_t + d)`.

# Fields
- `C::AbstractMatrix{T}`: Observation matrix of size `(obs_dim × latent_dim)`. Maps latent
    states into observation space.
- `d::AbstractVector{T}`: Per-neuron baseline log-rate (length `obs_dim`). Free in ℝ.
- `D::AbstractMatrix{T} = zeros(..., obs_dim, 0)`: Observation-input matrix of size
    `(obs_dim × uy_dim)` mapping the observation input `v_t` (`uy`) into log-rate space.
    Defaults to a zero-column matrix (no inputs).
- `CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing`: Optional matrix-normal prior on
    the stacked emission matrix `[C d D]` (treated as a single regression of `log λ` on
    `[x; 1; v]`). Prior matrices are stored as plain `Matrix{T}`, decoupled from `C`'s storage
    type `M`. `M₀` and `Λ` have shapes `(obs_dim, latent_dim+1+uy_dim)` and
    `(latent_dim+1+uy_dim, latent_dim+1+uy_dim)` respectively. Unlike the Gaussian path there
    is no IW counterpart since Poisson has no observation-noise covariance — this is an
    MN-only prior contributing `½ tr(([C d D] - M₀) Λ ([C d D] - M₀)')` to the LBFGS objective.
- `depends_on::Union{Nothing,NamedTuple} = nothing`: Optional declaration that `[C d D]`
    is estimated separately per group of trials. Keys are parameter names (`:C`/`:d`/`:D`),
    values are per-trial label vectors; `nothing` (the default) means one parameter set
    shared by every trial.
- `group_seeds::Union{Nothing,AbstractDict} = nothing`: Optional starting values for
    individual `depends_on` groups, as `label => (C=..., d=..., D=...)` with any subset of
    those keys. Only the groups a label names are seeded; the rest keep the defaults
    described under `variants`. Set it with [`set_group_seeds!`](@ref), which checks
    the labels.
- `variants::Union{Nothing,Vector{PoissonObservationModel{T,M,V}}} = nothing`: Derived
    storage for the per-group parameter sets; populated from `depends_on` by the fitting
    entry points.
"""
Base.@kwdef mutable struct PoissonObservationModel{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}
} <: AbstractObservationModel{T}
    C::M
    d::V
    D::M = zeros(eltype(C), size(C, 1), 0)  # eltype-preserving default (no obs inputs)
    CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing
    depends_on::Union{Nothing,NamedTuple} = nothing
    group_seeds::Union{Nothing,AbstractDict} = nothing
    variants::Union{Nothing,Vector{PoissonObservationModel{T,M,V}}} = nothing
end

# 2-arg convenience constructor; matches the Gaussian path's positional form
# so callers don't have to spell out `D` / `CD_prior`.
function PoissonObservationModel(
    C::M, d::V
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    return PoissonObservationModel{T,M,V}(;
        C=C, d=d, D=zeros(eltype(C), size(C, 1), 0), CD_prior=nothing
    )
end

# ============================================================================
# Composite (multi-model) observations
# ============================================================================

"""
    _key_list(names) -> String

`":kin, :spk"` — a list of symbols rendered for an error message. Accepts the
member keys of a composite (via its `NamedTuple` of models) or a plain tuple of
names.
"""
_key_list(names::Tuple{Vararg{Symbol}}) = join((":" * String(n) for n in names), ", ")
_key_list(models::NamedTuple) = _key_list(keys(models))

"""
    _emission_is_quadratic(obs_model) -> Bool

Whether an observation model's log-density is quadratic in the latent state.

`true` means the emission's curvature `∂² log p(y|x) / ∂x²` does not depend on
`x`, so the Newton smoother converges in a single step and the block-tridiagonal
Hessian — hence the smoothed covariance — is data-independent. That is what lets
`smooth!` share one covariance across equal-length trials and run the batched
BLAS-3 mean pass. `false` (Poisson) requires the iterative Laplace smoother.

The composite's value is the `AND` over its members, carried in its `QUAD` type
parameter so the choice of smoother is a compile-time dispatch.
"""
_emission_is_quadratic(::GaussianObservationModel) = true
_emission_is_quadratic(::PoissonObservationModel) = false

# Unrolled over the (heterogeneous, statically-sized) tuple of members so the
# result is a compile-time constant.
_all_quadratic(::Tuple{}) = true
function _all_quadratic(models::Tuple)
    return _emission_is_quadratic(first(models)) && _all_quadratic(Base.tail(models))
end

"""
    CompositeObservationModel{T<:Real,QUAD,NT<:NamedTuple} <: AbstractObservationModel{T}

Several observation models reading out one shared latent state. Build one by
handing a `NamedTuple` of observation models to `LinearDynamicalSystem`:

```julia
lds = LinearDynamicalSystem(
    state_model,
    (kin = GaussianObservationModel(C_kin, R_kin, d_kin),
     spk = PoissonObservationModel(C_spk, d_spk)),
)
```

The members are conditionally independent given the latent path, so
`log p(y | x) = Σ_m log p(y_m | x)` and every emission term (log-density,
gradient, curvature, Q-term, prior) is a sum over members. The dynamics
`[A b B]`, `Q` and the initial state `x0`, `P0` are shared.

Observations and observation inputs are supplied under the same keys:

```julia
fit!(lds, (kin = Ykin, spk = Yspk); uy = (kin = Vkin, spk = Vspk))
```

# Fields
- `models::NT`: the member observation models, keyed by name. Each is an
    ordinary [`GaussianObservationModel`](@ref) / [`PoissonObservationModel`](@ref)
    and keeps its own `obs_dim`, `D`/`uy_dim`, priors, `depends_on`,
    `group_seeds` and `variants` — nothing about a member changes by being put
    in a composite.

# Type parameters
- `QUAD::Bool`: `true` when every member is quadratic in the latent state (see
    [`_emission_is_quadratic`](@ref)). Encoded in the type so the single-step
    versus iterative smoother is chosen by dispatch rather than at run time.

# Parameter access
A member is reached by its key, and a member's parameter by the key-suffixed
name — the same spelling `depends_on`, `fit_bool` and `tied_params` use:

```julia
obs.kin         # the GaussianObservationModel
obs.kin.C       # its emission matrix
obs.C_kin       # the same array
```

Keys may not be observation-parameter names (`:C`, `:d`, `:D`, `:R`) or the
field name `:models`, since either would make the suffixed spelling ambiguous.

See also [`set_depends_on!`](@ref).
"""
struct CompositeObservationModel{T<:Real,QUAD,NT<:NamedTuple} <: AbstractObservationModel{T}
    models::NT
end

# Internal accessor. `getproperty` is overloaded below, so every internal read
# of the member tuple goes through `getfield` to stay on the fast path.
@inline _models(c::CompositeObservationModel) = getfield(c, :models)

"""
    _obs_keys(model) -> Tuple{Vararg{Symbol}}

The observation-model keys of a model: the composite's member names, or `()`
for a single observation model (which has no keys — its parameters are named
without a suffix).
"""
@inline _obs_keys(c::CompositeObservationModel) = keys(_models(c))
@inline _obs_keys(::AbstractObservationModel) = ()

@inline function _emission_is_quadratic(::CompositeObservationModel{T,QUAD}) where {T,QUAD}
    return QUAD
end

_obs_eltype(::AbstractObservationModel{T}) where {T<:Real} = T

#=
Names a member key may not take. `:models` is the struct's own field, and an
observation-parameter name would make `obs.C_kin` ambiguous the moment a member
were called `:C` (is it `models.C.kin`, or `models.kin.C`?).
=#
const _RESERVED_OBS_KEYS = (:models, :C, :d, :D, :R, :depends_on, :group_seeds, :variants)

"""
    CompositeObservationModel(models::NamedTuple)

Bundle several observation models into one. Called for you by
`LinearDynamicalSystem(state_model, models::NamedTuple)`; use it directly only
when you want the composite on its own.

# Throws
- `ArgumentError` on an empty `NamedTuple`, a member that is not an
  `AbstractObservationModel`, a reserved key name, or members with different
  element types.
"""
function CompositeObservationModel(models::NamedTuple)
    isempty(models) && throw(
        ArgumentError(
            "a composite observation model needs at least one member; got an empty " *
            "NamedTuple. Pass the observation model on its own for the single-emission " *
            "case.",
        ),
    )
    for (key, m) in pairs(models)
        m isa AbstractObservationModel || throw(
            ArgumentError(
                "observation model `:$key` is a $(typeof(m)); every member of a " *
                "composite must be an AbstractObservationModel",
            ),
        )
        key in _RESERVED_OBS_KEYS && throw(
            ArgumentError(
                "`:$key` cannot name an observation model: it collides with an " *
                "observation-parameter name or with the composite's own field, which " *
                "would make the suffixed spelling `C_$key` ambiguous. Reserved names " *
                "are $(_key_list(_RESERVED_OBS_KEYS)).",
            ),
        )
    end

    T = _obs_eltype(first(values(models)))
    for (key, m) in pairs(models)
        _obs_eltype(m) === T || throw(
            ArgumentError(
                "observation model `:$key` has element type $(_obs_eltype(m)) but " *
                "`:$(first(keys(models)))` has $T; every member of a composite must " *
                "share one element type",
            ),
        )
    end

    quad = _all_quadratic(values(models))
    return CompositeObservationModel{T,quad,typeof(models)}(models)
end

#=
`c.kin` (a member), `c.C_kin` (a member's parameter) and `c.depends_on` (the
assembled suffixed declaration). Everything else falls through to `getfield`, so
`c.models` still works.

The suffix split resolves against the actual member keys rather than by
splitting on the last `_`, so a member may be called `:eye_pos` without its
parameters becoming unreachable. The longest matching key wins, which makes the
split unambiguous even when one key is a suffix of another.
=#
function Base.getproperty(c::CompositeObservationModel, name::Symbol)
    name === :models && return getfield(c, :models)
    models = getfield(c, :models)
    haskey(models, name) && return models[name]
    name === :depends_on && return _composite_depends_on(c)
    split = _split_obs_name(name, models)
    split === nothing && throw(
        ArgumentError(
            "`$name` is not a member or parameter of this composite observation " *
            "model. Members are $(_key_list(models)); " *
            "a member's parameter is named with the member as a suffix, e.g. " *
            "`C_$(first(keys(models)))`.",
        ),
    )
    return getproperty(models[split[2]], split[1])
end

function Base.propertynames(c::CompositeObservationModel, private::Bool=false)
    models = getfield(c, :models)
    names = Symbol[:models, :depends_on]
    for key in keys(models)
        push!(names, key)
        for param in propertynames(models[key])
            push!(names, _suffixed(param, key))
        end
    end
    return Tuple(names)
end

"""
    _suffixed(param, key) -> Symbol

A member's parameter name in the composite's flat spelling: `(:C, :kin)` →
`:C_kin`. The inverse is [`_split_obs_name`](@ref).
"""
@inline _suffixed(param::Symbol, key::Symbol) = Symbol(param, :_, key)

"""
    _split_obs_name(name, models) -> Tuple{Symbol,Symbol} or nothing

Split a flat parameter name into `(parameter, member_key)`, or `nothing` when it
names no member's parameter. The longest matching key wins, so keys that are
suffixes of one another still resolve.
"""
function _split_obs_name(name::Symbol, models::NamedTuple)
    s = String(name)
    best = nothing
    best_len = 0
    for key in keys(models)
        suffix = "_" * String(key)
        endswith(s, suffix) || continue
        length(suffix) < length(s) || continue
        length(suffix) > best_len || continue
        param = Symbol(s[1:(end - length(suffix))])
        hasproperty(models[key], param) || continue
        best = (param, key)
        best_len = length(suffix)
    end
    return best
end

"""
    _composite_depends_on(c) -> NamedTuple or nothing

The composite's `depends_on` assembled from its members: each member's own
declaration with the member key appended to every parameter name. `nothing` when
no member declares one, which is what keeps `parameter_grouping` on the
ungrouped path.
"""
function _composite_depends_on(c::CompositeObservationModel)
    models = getfield(c, :models)
    entries = Pair{Symbol,Any}[]
    for key in keys(models)
        spec = models[key].depends_on
        spec === nothing && continue
        for param in keys(spec)
            push!(entries, _suffixed(param, key) => getproperty(spec, param))
        end
    end
    isempty(entries) && return nothing
    return NamedTuple(entries)
end

"""
    LinearDynamicalSystem{T<:Real, S<:AbstractStateModel{T}, O<:AbstractObservationModel{T}}

Represents a unified Linear Dynamical System with customizable state and observation models.

# Fields
- `state_model::S`: The state model (e.g., GaussianStateModel)
- `obs_model::O`: The observation model (e.g., GaussianObservationModel or
    PoissonObservationModel)
- `latent_dim::Int`: Dimension of the latent state
- `obs_dim::Int`: Dimension of the observations
- `ux_dim::Int`: Dimension of the dynamics input `ux` (0 when `B` is absent)
- `uy_dim::Int`: Dimension of the observation input `uy` (0 when `D` is absent)
- `fit_bool::Vector{Bool}`: Vector indicating which parameters to fit during optimization.
    The first four entries are the state side, `[x0, P0, A&b&B, Q]`; the rest are
    the observation side, one block per observation model in order — `[C&d&D, R]`
    for a Gaussian emission and `[C&d&D]` for a Poisson one. The M-step fits each
    regression jointly, which is why `A`, `b`, `B` share a slot (and `C`, `d`, `D`
    theirs). So the length is 6 for a Gaussian LDS, 5 for a Poisson one, and
    `4 + \u03a3\u2098 blocks` for a [`CompositeObservationModel`](@ref).

    The constructor also accepts the keyword form and lowers it, which avoids
    counting positions by hand:

    ```julia
    fit_bool = (x0=true, P0=true, A=true, Q=true, C=true, R=false)
    fit_bool = (x0=true, P0=true, A=true, Q=true,          # composite
                kin=(C=true, R=false), spk=(C=true,))
    ```

    Omitted names default to `true`.
"""
Base.@kwdef struct LinearDynamicalSystem{
    T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}
}
    state_model::S
    obs_model::O
    latent_dim::Int
    obs_dim::Int
    ux_dim::Int = 0
    uy_dim::Int = 0
    fit_bool::Vector{Bool}
end

#=
Observation-side shapes and `fit_bool` layout. Each helper has a single-model
method reading the model's own arrays and a composite method summing over
members, so the constructor below is one code path for both.
=#

"""
    _obs_dim(obs_model) -> Int

Total number of observed channels: `size(C, 1)` for a single observation model,
the sum over members for a composite.
"""
_obs_dim(om::AbstractObservationModel) = size(om.C, 1)
_obs_dim(c::CompositeObservationModel) = sum(_obs_dim, values(_models(c)))

"""
    _uy_dim(obs_model) -> Int

Total observation-input dimension: `size(D, 2)`, summed over a composite's
members. Each member keeps its own `D` and its own input sequence, so this total
is only the model-level summary — buffer sizing on the composite path uses each
member's own width via [`_obs_views`](@ref).
"""
function _uy_dim(om::AbstractObservationModel)
    return hasproperty(om, :D) && !isnothing(om.D) ? size(om.D, 2) : 0
end
_uy_dim(c::CompositeObservationModel) = sum(_uy_dim, values(_models(c)))

"""
    _obs_nblocks(obs_model) -> Int

How many `fit_bool` slots an observation model occupies: 2 for a Gaussian
emission (`[C&d&D]` and `R`), 1 for a Poisson one (no noise covariance), and the
sum over members for a composite.
"""
_obs_nblocks(::GaussianObservationModel) = 2
_obs_nblocks(::PoissonObservationModel) = 1
_obs_nblocks(c::CompositeObservationModel) = sum(_obs_nblocks, values(_models(c)))

"""
    _obs_fit_range(obs_model) -> UnitRange{Int}
    _obs_fit_range(composite, key) -> UnitRange{Int}

The slice of `fit_bool` holding one observation model's flags. The state side
always occupies `1:4`, so a single model's slice starts at 5 and a composite
member's starts after every member declared before it.
"""
_obs_fit_range(om::AbstractObservationModel) = 5:(4 + _obs_nblocks(om))

function _obs_fit_range(c::CompositeObservationModel, key::Symbol)
    models = _models(c)
    offset = 4
    for k in keys(models)
        n = _obs_nblocks(models[k])
        k === key && return (offset + 1):(offset + n)
        offset += n
    end
    return throw(
        ArgumentError(
            "`:$key` is not an observation model of this composite; members are " *
            "$(_key_list(models))",
        ),
    )
end

"""
    _default_fit_bool(obs_model) -> Vector{Bool}

All-`true` `fit_bool` of the right length for this observation model.
"""
_default_fit_bool(om::AbstractObservationModel) = fill(true, 4 + _obs_nblocks(om))

# Names the keyword `fit_bool` form accepts, for the error message on a typo.
function _fit_bool_keys(om::AbstractObservationModel)
    return (:x0, :P0, :A, :Q, :C, :R)[1:(4 + _obs_nblocks(om))]
end

function _fit_bool_keys(c::CompositeObservationModel)
    models = _models(c)
    names = Symbol[:x0, :P0, :A, :Q]
    for key in keys(models)
        push!(names, key)
        for param in (:C, :R)[1:_obs_nblocks(models[key])]
            push!(names, _suffixed(param, key))
        end
    end
    return Tuple(names)
end

"""
    _lower_fit_bool(obs_model, spec::NamedTuple) -> Vector{Bool}

Lower the keyword `fit_bool` form to the positional vector. Omitted names stay
`true`; an unknown name is an error rather than a silent no-op, since a typo
there would quietly fit a parameter the caller meant to freeze.
"""
function _lower_fit_bool(om::AbstractObservationModel, spec::NamedTuple)
    valid = _fit_bool_keys(om)
    for key in keys(spec)
        key in valid || throw(
            ArgumentError(
                "fit_bool: `:$key` is not a parameter group of this model; valid names " *
                "are $(_key_list(valid))",
            ),
        )
    end
    fb = _default_fit_bool(om)
    for (i, name) in enumerate((:x0, :P0, :A, :Q))
        haskey(spec, name) && (fb[i] = spec[name])
    end
    _apply_obs_fit_bool!(fb, om, spec)
    return fb
end

function _apply_obs_fit_bool!(fb::Vector{Bool}, om::AbstractObservationModel, spec)
    r = _obs_fit_range(om)
    haskey(spec, :C) && (fb[first(r)] = spec[:C])
    length(r) > 1 && haskey(spec, :R) && (fb[first(r) + 1] = spec[:R])
    return fb
end

#=
A composite member is named either by nesting (`kin = (C = true, R = false)`) or
by the flat suffixed spelling (`C_kin = true`). Both are accepted; the flat form
is applied second so it wins if a caller somehow gives both.
=#
function _apply_obs_fit_bool!(
    fb::Vector{Bool}, c::CompositeObservationModel, spec::NamedTuple
)
    models = _models(c)
    for key in keys(models)
        r = _obs_fit_range(c, key)
        nested = get(spec, key, nothing)
        if nested !== nothing
            nested isa NamedTuple || throw(
                ArgumentError(
                    "fit_bool[:$key] must be a NamedTuple of that model's parameter " *
                    "groups, e.g. `(C = true, R = false)`; got a $(typeof(nested))",
                ),
            )
            _apply_obs_fit_bool!(view(fb, r), models[key], nested)
        end
        haskey(spec, _suffixed(:C, key)) && (fb[first(r)] = spec[_suffixed(:C, key)])
        length(r) > 1 &&
            haskey(spec, _suffixed(:R, key)) &&
            (fb[first(r) + 1] = spec[_suffixed(:R, key)])
    end
    return fb
end

# Nested form writes through a length-1/2 view of the parent vector.
function _apply_obs_fit_bool!(
    fb::AbstractVector{Bool}, om::AbstractObservationModel, spec::NamedTuple
)
    haskey(spec, :C) && (fb[1] = spec[:C])
    length(fb) > 1 && haskey(spec, :R) && (fb[2] = spec[:R])
    return fb
end

"""
    LinearDynamicalSystem(state_model, obs_model; fit_bool=nothing)
    LinearDynamicalSystem(state_model, obs_models::NamedTuple; fit_bool=nothing)

Build a linear dynamical system from a state model and either a single
observation model or a `NamedTuple` of them (wrapped into a
[`CompositeObservationModel`](@ref)). Dimensions are inferred from the parameter
matrices and the result is validated before it is returned.

`fit_bool` accepts the positional vector or the keyword form described under the
type's `fit_bool` field; omitted, every parameter is fitted.
"""
function LinearDynamicalSystem(
    state_model::S, obs_model::O; fit_bool::Union{Vector{Bool},NamedTuple,Nothing}=nothing
) where {T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}}

    # Infer dimensions from matrices
    latent_dim = size(state_model.A, 1)
    obs_dim = _obs_dim(obs_model)
    ux_dim = if hasproperty(state_model, :B) && !isnothing(state_model.B)
        size(state_model.B, 2)
    else
        0
    end
    uy_dim = _uy_dim(obs_model)

    fb = if fit_bool === nothing
        _default_fit_bool(obs_model)
    elseif fit_bool isa NamedTuple
        _lower_fit_bool(obs_model, fit_bool)
    else
        fit_bool
    end

    # Create the LDS
    lds = LinearDynamicalSystem{T,S,O}(
        state_model, obs_model, latent_dim, obs_dim, ux_dim, uy_dim, fb
    )

    # Validate the constructed LDS (throws on error)
    validate_LDS(lds)

    return lds
end

function LinearDynamicalSystem(
    state_model::AbstractStateModel, obs_models::NamedTuple; kwargs...
)
    return LinearDynamicalSystem(
        state_model, CompositeObservationModel(obs_models); kwargs...
    )
end

"""
    SLDS{T,S,O,TM,ISV}

A Switching Linear Dynamical System (SLDS). A hierarchical time-series model of the form:

```math
z_t | z_{t-1} ~ Categorical(A_{z_{t-1}, :})
x_t | x_{t-1}, z_t ~ N(A^{(z_t)} x_{t-1} + b^{(z_t)}, Q^{(z_t)})
y_t | x_t, z_t ~ N(C^{(z_t)} x_t + d^{(z_t)}, R^{(z_t)})
```

# Fields
- `A::TM`: Transition matrix for the discrete states (K x K)
- `πₖ::ISV`: Initial state distribution for the discrete states (K-dimensional vector)
- `LDSs::Vector{LinearDynamicalSystem{T,S,O}}`: Vector of K Linear Dynamical Systems, one for each discrete state
"""
@kwdef mutable struct SLDS{
    T<:Real,
    S<:AbstractStateModel,
    O<:AbstractObservationModel,
    TM<:AbstractMatrix{T},
    ISV<:AbstractVector{T},
}
    A::TM
    πₖ::ISV
    LDSs::Vector{LinearDynamicalSystem{T,S,O}}
end

"""
    SLDSDiscreteLayer{T,TM,TV}

Thin wrapper satisfying the `HiddenMarkovModels.AbstractHMM` interface for the discrete
switching layer of an SLDS.  The `logL` matrix (K×T) is pre-filled with per-state
log-likelihoods before each forward-backward call; `obs_seq` is then just `1:T` (timestep
indices) so that `obs_logdensities!` can look up the correct column.

Fields `A` and `πₖ` are kept as references to the parent SLDS matrices so that in-place
M-step updates are automatically reflected.
"""
mutable struct SLDSDiscreteLayer{T<:Real,TM<:AbstractMatrix{T},TV<:AbstractVector{T}} <:
               HMMs.AbstractHMM
    A::TM            # K×K row-stochastic transition matrix
    πₖ::TV           # K initial-state distribution
    logL::Matrix{T}  # K×T pre-computed log-likelihoods; mutated before each FB pass
end

HMMs.initialization(dl::SLDSDiscreteLayer) = dl.πₖ
HMMs.transition_matrix(dl::SLDSDiscreteLayer) = dl.A

# Override the log-density chokepoint so obs_distributions is never needed.
# obs is the timestep index t (an Int), supplied as obs_seq = 1:T.
function HMMs.obs_logdensities!(
    logb::AbstractVector, dl::SLDSDiscreteLayer, obs::Int, control; kwargs...
)
    logb .= view(dl.logL, :, obs)
    return nothing
end

# Provide eltype without going through obs_distributions
Base.eltype(::SLDSDiscreteLayer{T}, obs, control) where {T} = T

#=
Workaround for JET union-split false positive on views with unbound eltype
(remove when fixed upstream; see: https://github.com/depasquale-lab/StateSpaceDynamics.jl/issues/105)
=#
@inline tview(A::AbstractArray{T}, I...) where {T} = view(A, I...)::SubArray{T}

# ============================================================================
# `Data` construction — the single validation site for the public array API.
# `fit!` / `smooth` / `loglikelihood` accept observations in three shapes
# (single matrix, 3-D array, vector of per-trial matrices) plus optional
# `ux` / `uy` inputs in the same shape family, and canonicalize them here into
# the private `Data` container consumed by the multi-trial backend. The
# per-trial input normalization helpers (`_normalize_multitrial_ux` / `_uy`)
# and `DimensionMismatchError` live in `utils/validation.jl`.
# ============================================================================

"""
    Data(lds, y; ux=nothing, uy=nothing)

Validate observations and inputs against `lds` and canonicalize them into the
internal [`Data`](@ref) container.

`y` may be a `(obs_dim, T)` matrix (single trial), a `(obs_dim, T, ntrials)`
array, or a vector of per-trial `(obs_dim, T_i)` matrices (ragged trial
lengths allowed). `ux` / `uy` accept the same shape family as `y`, or
`nothing` when the model has no `B` / `D` input matrix; absent inputs are
canonicalized to zero-row matrices.

# Throws
- `DimensionMismatchError` when observation or input dimensions disagree with
  the model, or input trial lengths disagree with `y`
- `ArgumentError` when inputs are omitted for a model that requires them
  (`ux_dim > 0` / `uy_dim > 0`)
"""
function Data(
    lds::LinearDynamicalSystem{T},
    y::AbstractVector{<:AbstractMatrix{T}};
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
) where {T<:Real}
    isempty(y) && throw(ArgumentError("y must contain at least one trial"))
    #=
    A group-dependent emission is the stitching case: each session brings its
    own channel count, so rows are checked per parameter version once the trial
    partition is known (`_slot_obs_dims`) rather than against the template's
    `obs_dim` here. Everything else keeps the strict single-`obs_dim` check.
    =#
    if !_has_parameter_dependence(lds.obs_model)
        for (i, yt) in enumerate(y)
            size(yt, 1) == lds.obs_dim ||
                throw(DimensionMismatchError("y[$i] rows", lds.obs_dim, size(yt, 1)))
        end
    end
    tsteps = Int[size(yt, 2) for yt in y]
    ux_seq = _normalize_multitrial_ux(ux, lds.ux_dim, tsteps, T, "ux")
    uy_seq = _normalize_multitrial_uy(uy, lds.uy_dim, tsteps, T, lds.obs_model)
    return Data(y, ux_seq, uy_seq, tsteps)
end

function Data(
    lds::LinearDynamicalSystem{T},
    y::AbstractMatrix{T};
    ux::Union{Nothing,AbstractMatrix{T}}=nothing,
    uy::Union{Nothing,AbstractMatrix{T}}=nothing,
) where {T<:Real}
    return Data(
        lds, [y]; ux=(ux === nothing ? nothing : [ux]), uy=(uy === nothing ? nothing : [uy])
    )
end

function Data(
    lds::LinearDynamicalSystem{T},
    y::AbstractArray{T,3};
    ux::Union{Nothing,AbstractArray{T,3}}=nothing,
    uy::Union{Nothing,AbstractArray{T,3}}=nothing,
) where {T<:Real}
    _trials(A) = [view(A, :, :, n) for n in axes(A, 3)]
    return Data(
        lds,
        _trials(y);
        ux=(ux === nothing ? nothing : _trials(ux)),
        uy=(uy === nothing ? nothing : _trials(uy)),
    )
end
