#=============================================================================
Composite (multi-model) observations

A [`CompositeObservationModel`](@ref) bundles several observation models that
read out one shared latent state. They are conditionally independent given the
latent path, so every emission quantity is a **sum over members**:

    log p(y | x)     = Σ_m log p(y_m | x)
    ∂/∂x  log p(y|x) = Σ_m ∂/∂x log p(y_m | x)
    ∂²/∂x² log p(y|x)= Σ_m ∂²/∂x² log p(y_m | x)

and each member's M-step, prior term and sufficient statistics are its own.

Two devices keep this from duplicating the single-model machinery.

**Per-member LDS views** (`_obs_views`). A view is a `LinearDynamicalSystem`
holding the parent's state model, one member observation model, that member's
own `obs_dim` / `uy_dim`, and the member's slice of `fit_bool`. Parameter arrays
are shared by reference, so every existing single-observation routine —
`Q_obs!`, `update_C_d!`, `update_R!`, `update_observation_model!`,
`_accumulate_obs_scatter!`, the grouped M-step, the prior log-densities — runs
on a view unchanged and writes straight through to the member's parameters.
This is the same trick `_cell_lds` uses for `depends_on` cells.

**Per-member sub-workspaces** (`_obs_workspaces!`). One `SmoothWorkspace` per
member, built with `_cell_workspace` so the O(D²·T) block-tridiagonal and
shared-covariance storage is shared with the parent and only the `obs_dim`-shaped
buffers are allocated per member. The parent workspace of a composite fit is
itself built with `obs_dim = 0`, so the `Σₘ pₘ`-square buffers a single-model
workspace would carry are never allocated at all.

Member loops go through a function barrier (`_obs_views` / the sub-workspace
vector) rather than being unrolled over the member tuple. They run once per
`gradient!` / `joint_loglikelihood!` / `hessian!` / M-step call — never per
timestep, since each member's contribution is accumulated over a whole trial
before the next member runs — so the handful of dynamic dispatches per call is
immaterial next to the per-timestep work behind them.
=============================================================================#

# ============================================================================
# Per-member LDS views
# ============================================================================

"""
    _obs_view(lds, key) -> LinearDynamicalSystem

The parent system restricted to one observation model: same state model, that
member's emission, its own `obs_dim` / `uy_dim`, and a `fit_bool` of the shape a
single-observation model has (`[x0, P0, A&b&B, Q, C&d&D]` plus `R` when the
member is Gaussian).

Parameter arrays are shared by reference, so an M-step run through a view
updates the member itself. `fit_bool` is the one exception — it is copied, since
it is a `Vector{Bool}` rather than a parameter array. Nothing writes to
`fit_bool` during a fit, so the copy is only ever read.
"""
function _obs_view(
    lds::LinearDynamicalSystem{T,S,O}, key::Symbol
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    om = _models(lds.obs_model)[key]
    r = _obs_fit_range(lds.obs_model, key)
    fit_bool = Vector{Bool}(undef, 4 + length(r))
    @views fit_bool[1:4] .= lds.fit_bool[1:4]
    @views fit_bool[5:end] .= lds.fit_bool[r]
    return LinearDynamicalSystem{T,S,typeof(om)}(
        lds.state_model, om, lds.latent_dim, _obs_dim(om), lds.ux_dim, _uy_dim(om), fit_bool
    )
end

"""
    _obs_views(lds) -> NamedTuple

One [`_obs_view`](@ref) per member, keyed as the composite is. Cheap enough to
rebuild per M-step (two struct allocations and a length-6 `Vector{Bool}` each),
so nothing caches it.
"""
function _obs_views(
    lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    keys_ = _obs_keys(lds.obs_model)
    return NamedTuple{keys_}(map(k -> _obs_view(lds, k), keys_))
end

# ============================================================================
# Per-member sub-workspaces
# ============================================================================

"""
    _obs_workspaces!(ws, lds) -> Vector{SmoothWorkspace}

The parent workspace's per-member sub-workspaces, in the composite's key order,
built on first use and reused after.

Each is a [`_cell_workspace`](@ref) over `ws`: the block-tridiagonal storage and
the shared-covariance cache are shared with the parent (members never run the BT
solve — the parent does, once, on the summed curvature), while everything shaped
by `obs_dim` or `uy_dim` is the member's own.

Rebuilt when a member's width or the workspace's trial length no longer fits,
which is what makes a workspace safe to reuse across the cells of a grouped fit.
"""
function _obs_workspaces!(
    ws::SmoothWorkspace{WT}, lds::LinearDynamicalSystem{T,S,O}
) where {WT<:Real,T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    models = _models(lds.obs_model)
    tsteps = length(ws.opt.ll_vec)
    subs = ws.obs
    if subs !== nothing && _obs_workspaces_fit(subs, lds, tsteps)
        return subs
    end
    fresh = SmoothWorkspace{WT}[
        _cell_workspace(
            ws, lds.latent_dim, _obs_dim(m), tsteps; ux_dim=lds.ux_dim, uy_dim=_uy_dim(m)
        ) for m in values(models)
    ]
    ws.obs = fresh
    return fresh
end

# Whether cached sub-workspaces still have the shapes this model and trial
# length need. `obs_temp` is (p, p) and `CD` is (p, D + 1 + uy_dim), so between
# them they pin every observation-side width.
function _obs_workspaces_fit(
    subs::Vector{SmoothWorkspace{WT}}, lds::LinearDynamicalSystem{T,S,O}, tsteps::Int
) where {WT<:Real,T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    models = _models(lds.obs_model)
    length(subs) == length(models) || return false
    for (i, m) in enumerate(values(models))
        sub = subs[i]
        size(sub.elbo.obs_temp, 1) == _obs_dim(m) || return false
        size(sub.reg.CD, 2) == lds.latent_dim + 1 + _uy_dim(m) || return false
        length(sub.opt.ll_vec) >= tsteps || return false
    end
    return true
end

"""
    compute_smooth_constants!(ws::SmoothWorkspace, lds_with_composite_emission)

Cache the constants for a composite emission: the state half once on the parent,
each member's emission half on its own sub-workspace, and then the two aggregate
slots the shared Hessian assembly reads —

- `ws.consts.yt_given_xt = Σₘ (-Cₘ' Rₘ⁻¹ Cₘ)`, the summed emission curvature. A
  Poisson member contributes nothing here (its curvature depends on `x` and is
  added per Newton step), which is exactly the single-model convention.
- `ws.consts.cR = Σₘ cRₘ`, the summed emission normalizer.

Because those are the only two emission slots `_fill_hessian_blocks!` and the
likelihood kernels read off the parent, the whole quadratic fast path works on a
composite without any change to `SmoothConstants`.

Members get only their emission half: the state constants live on the parent,
and nothing that runs on a sub-workspace reads them.
"""
function compute_smooth_constants!(
    ws::SmoothWorkspace{WT}, lds::LinearDynamicalSystem{T,S,O}
) where {WT<:Real,T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    subs = _obs_workspaces!(ws, lds)
    _compute_composite_constants!(
        ws.consts, [sub.consts for sub in subs], lds.state_model, lds.obs_model
    )
    return nothing
end

"""
    _compute_composite_constants!(cc, member_ccs, state_model, obs_model)

Cache a composite emission's constants: the state half on `cc`, each member's
emission half on its own `member_ccs[i]`, and the two aggregate slots the shared
Hessian assembly reads.

Shared by the single-LDS workspace (whose members live on sub-workspaces) and by
an SLDS's per-regime refresh (whose members live on [`ObsScratch`](@ref)).
"""
function _compute_composite_constants!(
    cc::SmoothConstants{WT},
    member_ccs::AbstractVector{SmoothConstants{WT}},
    sm::AbstractGaussianStateModel{T},
    om::CompositeObservationModel{T},
) where {WT<:Real,T<:Real}
    _compute_state_constants!(cc, sm)

    #=
    No aggregate `C'R⁻¹` exists — the gradient needs each member's own residual —
    so the parent's slot is cleared and the per-member ones stay per member.
    =#
    fill!(cc.C_inv_R, zero(WT))
    fill!(cc.yt_given_xt, zero(WT))
    cR = zero(WT)

    for (i, m) in enumerate(values(_models(om)))
        member_cc = member_ccs[i]
        _compute_obs_constants!(member_cc, m)
        cc.yt_given_xt .+= member_cc.yt_given_xt
        cR += member_cc.cR
    end
    cc.cR = cR

    return nothing
end

# ============================================================================
# `Data` construction for a composite emission
#
# Same contract as the single-observation constructors in `types.jl`: this is
# the one place shapes are validated, and everything downstream of a `Data` may
# assume they are consistent. `ux` and `tsteps` are shared across members, so
# only `y` and `uy` become NamedTuples.
# ============================================================================

"""
    _as_trials(y) -> Vector of per-trial matrices

Canonicalize one member's observations from any of the three public shapes — a
`(p, T)` matrix (single trial), a `(p, T, ntrials)` array, or a vector of
per-trial matrices — into the vector-of-matrices form the backend consumes.
"""
_as_trials(y::AbstractVector{<:AbstractMatrix}) = y
_as_trials(y::AbstractMatrix) = [y]
_as_trials(y::AbstractArray{<:Any,3}) = [view(y, :, :, n) for n in axes(y, 3)]

"""
    Data(lds, y::NamedTuple; ux=nothing, uy=nothing)

Validate a composite emission's observations and inputs and canonicalize them
into the internal [`Data`](@ref) container.

`y` carries one entry per member of the [`CompositeObservationModel`](@ref),
under the same keys. Each entry independently takes any of the shapes the
single-observation API accepts — a `(pₘ, T)` matrix, a `(pₘ, T, ntrials)` array,
or a vector of per-trial `(pₘ, T_i)` matrices — but every member must agree on
the trial count and on each trial's length, since they observe one shared latent
path.

`uy` is either a `NamedTuple` under the same keys, giving each member its own
observation-input sequence, or a single array applied to every member (the
convenient case where one set of covariates feeds all the readouts). `ux` is
shared and takes the same shapes as before.

# Throws
- `ArgumentError` when `y`'s keys do not match the model's, when members
  disagree on trial count or trial lengths, or when inputs are omitted for a
  member that needs them
- `DimensionMismatchError` when a member's row count disagrees with its emission
"""
function Data(
    lds::LinearDynamicalSystem{T,S,O}, y::NamedTuple; ux=nothing, uy=nothing
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    models = _models(lds.obs_model)
    obs_keys = keys(models)

    Set(keys(y)) == Set(obs_keys) || throw(
        ArgumentError(
            "observations must be given under exactly this model's observation keys " *
            "($(_key_list(models))); got $(_key_list(keys(y)))",
        ),
    )

    # Reorder to the model's key order so `values(...)` lines up with the members.
    y_seq = NamedTuple{obs_keys}(map(k -> _as_trials(y[k]), obs_keys))

    ref_key = first(obs_keys)
    ref = y_seq[ref_key]
    isempty(ref) && throw(ArgumentError("y[:$ref_key] must contain at least one trial"))
    tsteps = Int[size(yt, 2) for yt in ref]

    for key in obs_keys
        yk = y_seq[key]
        length(yk) == length(tsteps) || throw(
            ArgumentError(
                "y[:$key] has $(length(yk)) trials but y[:$ref_key] has " *
                "$(length(tsteps)); every observation model sees the same trials",
            ),
        )
        for (i, yt) in enumerate(yk)
            size(yt, 2) == tsteps[i] || throw(
                ArgumentError(
                    "y[:$key][$i] has $(size(yt, 2)) timesteps but y[:$ref_key][$i] has " *
                    "$(tsteps[i]); every observation model sees the same latent path",
                ),
            )
        end
        #=
        A group-dependent emission is the stitching case: each session brings its
        own channel count, so rows are checked per parameter version once the
        trial partition is known, not against the template here.
        =#
        _has_parameter_dependence(models[key]) && continue
        p = _obs_dim(models[key])
        for (i, yt) in enumerate(yk)
            size(yt, 1) == p ||
                throw(DimensionMismatchError("y[:$key][$i] rows", p, size(yt, 1)))
        end
    end

    ux_seq = _normalize_multitrial_ux(
        ux === nothing ? nothing : _as_trials(ux), lds.ux_dim, tsteps, T, "ux"
    )

    uy_seq = NamedTuple{obs_keys}(
        map(obs_keys) do key
            supplied = _member_uy(uy, key)
            _normalize_multitrial_ux(
                supplied === nothing ? nothing : _as_trials(supplied),
                _uy_dim(models[key]),
                tsteps,
                T,
                "uy[:$key]",
            )
        end,
    )

    return Data(y_seq, ux_seq, uy_seq, tsteps)
end

#=
One member's observation input: its own entry when `uy` is a NamedTuple, and the
whole thing otherwise — a bare array is the shorthand for "these covariates feed
every readout".
=#
_member_uy(::Nothing, ::Symbol) = nothing
_member_uy(uy::NamedTuple, key::Symbol) = get(uy, key, nothing)
_member_uy(uy, ::Symbol) = uy

#=
Guard rails for mismatched call shapes. Without these, an array handed to a
composite model would fall through to the single-observation constructor and
fail somewhere deep in `_normalize_multitrial_uy`, and a NamedTuple handed to a
single-observation model would be a bare `MethodError`.
=#
function _composite_needs_namedtuple(lds::LinearDynamicalSystem)
    return throw(
        ArgumentError(
            "this model has several observation models, so observations must be a " *
            "NamedTuple keyed by them, e.g. " *
            "`($(first(_obs_keys(lds.obs_model))) = y, ...)`. Members are " *
            "$(_key_list(_models(lds.obs_model))).",
        ),
    )
end

# One guard per shape the single-observation constructor accepts, so each is
# strictly more specific than the method it shadows.
function Data(
    lds::LinearDynamicalSystem{T,S,O},
    ::AbstractMatrix{T};
    ux::Union{Nothing,AbstractMatrix{T}}=nothing,
    uy::Union{Nothing,AbstractMatrix{T}}=nothing,
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    return _composite_needs_namedtuple(lds)
end

function Data(
    lds::LinearDynamicalSystem{T,S,O},
    ::AbstractArray{T,3};
    ux::Union{Nothing,AbstractArray{T,3}}=nothing,
    uy::Union{Nothing,AbstractArray{T,3}}=nothing,
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    return _composite_needs_namedtuple(lds)
end

function Data(
    lds::LinearDynamicalSystem{T,S,O},
    ::AbstractVector{<:AbstractMatrix{T}};
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    return _composite_needs_namedtuple(lds)
end

function Data(
    lds::LinearDynamicalSystem{T,S,O}, ::NamedTuple; kwargs...
) where {T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}}
    return throw(
        ArgumentError(
            "this model has a single $(nameof(typeof(lds.obs_model))), so observations " *
            "must be an array, not a NamedTuple. Build the model with a NamedTuple of " *
            "observation models to fit several emissions at once.",
        ),
    )
end

# ============================================================================
# `depends_on`
# ============================================================================

"""
    set_depends_on!(model, spec) -> model

Declare which parameters are estimated separately per group of trials.

For a state model or a single observation model this is `model.depends_on = spec`
spelled as a function, so the same call works whatever the model is.

For a [`CompositeObservationModel`](@ref) the keys carry the member as a suffix
and are distributed to the members:

```julia
set_depends_on!(obs, (C_kin = session, d_kin = session, R_kin = session))
```

is the same as `obs.kin.depends_on = (C = session, d = session, R = session)`.
This sets the *whole* declaration: a member named by no key has its `depends_on`
cleared, exactly as assigning a fresh `NamedTuple` to a single model's field
replaces whatever was there. `nothing` clears every member.

Reading back, `obs.depends_on` assembles the suffixed view of whatever the
members currently declare.
"""
function set_depends_on!(model::AbstractObservationModel, spec::Union{Nothing,NamedTuple})
    model.depends_on = spec
    return model
end

function set_depends_on!(model::AbstractStateModel, spec::Union{Nothing,NamedTuple})
    model.depends_on = spec
    return model
end

function set_depends_on!(c::CompositeObservationModel, spec::Union{Nothing,NamedTuple})
    models = _models(c)
    if spec === nothing
        for m in values(models)
            m.depends_on = nothing
        end
        return c
    end

    per_member = Dict{Symbol,Vector{Pair{Symbol,Any}}}()
    for key in keys(spec)
        split = _split_obs_name(key, models)
        split === nothing && throw(
            ArgumentError(
                "set_depends_on!: `:$key` does not name any member's parameter. Use the " *
                "member as a suffix, e.g. `C_$(first(keys(models)))`; members are " *
                "$(_key_list(models)).",
            ),
        )
        push!(get!(per_member, split[2], Pair{Symbol,Any}[]), split[1] => spec[key])
    end

    for key in keys(models)
        entries = get(per_member, key, nothing)
        models[key].depends_on = entries === nothing ? nothing : NamedTuple(entries)
    end
    return c
end

"""
    group_labels(composite, name)
    group_parameter(composite, name, label)

The public `depends_on` accessors, spelled with the member as a suffix:

```julia
group_labels(obs, :C_kin)                 # [:session_a, :session_b]
group_parameter(obs, :C_kin, :session_a)  # that session's kinematics loadings
```

Both split the suffix and delegate to the member that owns the parameter, which
is also where its `variants` live.
"""
function group_labels(c::CompositeObservationModel, name::Symbol)
    param, key = _split_obs_param(c, name)
    return group_labels(_models(c)[key], param)
end

function group_parameter(c::CompositeObservationModel, name::Symbol, label)
    param, key = _split_obs_param(c, name)
    return group_parameter(_models(c)[key], param, label)
end

function _split_obs_param(c::CompositeObservationModel, name::Symbol)
    models = _models(c)
    split = _split_obs_name(name, models)
    split === nothing && throw(
        ArgumentError(
            "`:$name` does not name a parameter of any member of this composite " *
            "observation model. Use the member as a suffix, e.g. " *
            "`:C_$(first(keys(models)))`; members are $(_key_list(models)).",
        ),
    )
    return split
end

# ============================================================================
# Sampling glue
# ============================================================================

"""
    _extract_obs_params(composite) -> NamedTuple

Per-member parameter NamedTuples, keyed as the composite is. The sampler walks
these rather than the models themselves so a grouped draw can substitute a
cell's parameters without rebuilding the models.
"""
function _extract_obs_params(om::CompositeObservationModel)
    models = _models(om)
    return NamedTuple{keys(models)}(map(_extract_obs_params, values(models)))
end

# ============================================================================
# Latent-inference kernels
#
# The emission is a sum over members, so each of the three kernels below runs
# the state half once and then adds one member's whole-trial contribution at a
# time. Iterating members outside the timestep loop means the one dynamic
# dispatch per member per call is paid once rather than once per timestep, and
# everything behind the `_accumulate_member_*!` barrier is concretely typed.
#
# The accumulators take their scratch as explicit pieces rather than a
# workspace, and an optional per-timestep weight. That is what lets the single
# LDS (unweighted, scratch from a sub-`SmoothWorkspace`) and the SLDS (weighted
# by a regime's responsibilities `γₖ(t)`, scratch from an `ObsScratch`) share one
# implementation.
# ============================================================================

# One member's slice of a per-trial observation / input bundle. `nothing`
# propagates, which is how "this model takes no observation inputs" is spelled
# everywhere else.
@inline _member_at(::Nothing, ::Int) = nothing
@inline _member_at(bundle::NamedTuple, i::Int) = bundle[i]

@inline _weight_at(::Nothing, ::Int, ::Type{T}) where {T} = one(T)
@inline _weight_at(w::AbstractVector, t::Int, ::Type{T}) where {T} = @inbounds T(w[t])

"""
    joint_loglikelihood!(ll, ws, cc, lds, x, y::NamedTuple[, ux, uy, lognorms])

Per-timestep complete-data log-likelihood for a composite emission:
`ll[t] = Σₘ log p(yₘ,ₜ | xₜ) + log p(xₜ | xₜ₋₁)`.

`y` (and `uy`, when given) are `NamedTuple`s of this trial's per-member
matrices, keyed as the composite is. `lognorms` optionally carries each
member's precomputed data-only normalizer (see [`_emission_lognorm`](@ref)).
"""
function joint_loglikelihood!(
    ll::AbstractVector{T},
    ws::SmoothWorkspace{T},
    cc::SmoothConstants{T},
    lds::LinearDynamicalSystem{T0,S,O},
    x::AbstractMatrix{T},
    y::NamedTuple,
    ux::Union{Nothing,AbstractMatrix}=nothing,
    uy::Union{Nothing,NamedTuple}=nothing,
    lognorms::Union{Nothing,NamedTuple}=nothing,
) where {
    T<:Real,T0<:Real,S<:AbstractGaussianStateModel{T0},O<:CompositeObservationModel{T0}
}
    tsteps = size(x, 2)
    @assert length(ll) == tsteps

    opt = ws.opt
    @inbounds for t in 1:tsteps
        ll[t] = state_loglikelihood!(cc, opt.temp_dx, opt.temp_solve_Q, lds, x, t, ux)
    end

    subs = _obs_workspaces!(ws, lds)
    for (i, om) in enumerate(values(_models(lds.obs_model)))
        sub = subs[i]
        _accumulate_member_loglikelihood!(
            ll,
            sub,
            sub.consts,
            sub.opt.temp_dy,
            sub.opt.temp_solve_R,
            om,
            x,
            y[i],
            _member_at(uy, i),
            tsteps,
            _member_at(lognorms, i),
            nothing,
        )
    end

    return ll
end

"""
    _accumulate_member_loglikelihood!(ll, pb_ws, cc, b1, b2, om, x, y, uy, tsteps,
                                      lognorm, weights)

Add one observation model's per-timestep emission log-density to `ll`, scaled by
`weights[t]` when given.

`pb_ws` owns the batched Poisson scratch (see [`poisson_batch!`](@ref)); `cc`,
`b1` and `b2` are that member's cached constants and two `obs_dim` scratch
vectors.
"""
function _accumulate_member_loglikelihood!(
    ll::AbstractVector{T},
    ::Union{SmoothWorkspace{T},SLDSSmoothWorkspace{T}},
    cc::SmoothConstants{T},
    b1::AbstractVector{T},
    b2::AbstractVector{T},
    om::AbstractObservationModel,
    x::AbstractMatrix{T},
    y_m::AbstractMatrix,
    uy_m::Union{Nothing,AbstractMatrix},
    tsteps::Int,
    ::Union{Nothing,AbstractVector},
    weights::Union{Nothing,AbstractVector},
) where {T<:Real}
    @inbounds for t in 1:tsteps
        ll[t] +=
            _weight_at(weights, t, T) *
            observation_loglikelihood!(cc, b1, b2, om, x, y_m, t, uy_m)
    end
    return ll
end

#=
A Poisson member takes the batched route: `η = C·x + d + D·v` for the whole
trial as one `gemm`, and the data-only `Σᵢ log(y!)` normalizer supplied
precomputed. Inside a Newton line search that normalizer would otherwise be
re-summed over every neuron and bin at every objective evaluation, which is what
`_emission_lognorm` hoists out.
=#
function _accumulate_member_loglikelihood!(
    ll::AbstractVector{T},
    pb_ws::Union{SmoothWorkspace{T},SLDSSmoothWorkspace{T}},
    ::SmoothConstants{T},
    ::AbstractVector{T},
    ::AbstractVector{T},
    om::PoissonObservationModel{T},
    x::AbstractMatrix{T},
    y_m::AbstractMatrix{T},
    uy_m::Union{Nothing,AbstractMatrix},
    tsteps::Int,
    lognorm::Union{Nothing,AbstractVector},
    weights::Union{Nothing,AbstractVector},
) where {T<:Real}
    obs_dim, latent_dim = size(om.C)
    pb = poisson_batch!(pb_ws, latent_dim, obs_dim, tsteps)
    Eta = _poisson_linear_predictor!(pb, om.C, om.d, om.D, x, uy_m, tsteps)
    @inbounds @views for t in 1:tsteps
        η = Eta[:, t]
        norm_t = lognorm === nothing ? _poisson_lognorm_at(y_m, t) : lognorm[t]
        ll[t] += _weight_at(weights, t, T) * (dot(y_m[:, t], η) - sum(exp, η) - norm_t)
    end
    return ll
end

"""
    gradient!(grad, ws, lds, x, y::NamedTuple[, ux, uy])
    gradient!(ws, lds, x, y::NamedTuple[, ux, uy])

Gradient of the complete-data log-likelihood w.r.t. the latent path for a
composite emission: the shared state half from `_state_gradient!`, then each
member's `Σₜ Cₘ'Rₘ⁻¹ rₘ,ₜ`-style term added on top.
"""
function gradient!(
    grad::AbstractMatrix{T},
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::NamedTuple,
    ux::Union{Nothing,AbstractMatrix}=nothing,
    uy::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    tsteps = size(x, 2)
    _state_gradient!(grad, ws, lds, x, ux)

    subs = _obs_workspaces!(ws, lds)
    for (i, om) in enumerate(values(_models(lds.obs_model)))
        sub = subs[i]
        _accumulate_member_gradient!(
            grad,
            sub,
            sub.consts,
            sub.opt.dyt,
            sub.opt.tmp1,
            om,
            x,
            y[i],
            _member_at(uy, i),
            tsteps,
            nothing,
        )
    end

    return grad
end

function gradient!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::NamedTuple,
    ux::Union{Nothing,AbstractMatrix}=nothing,
    uy::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    grad = view(ws.opt.grad_buf, :, 1:size(x, 2))
    return gradient!(grad, ws, lds, x, y, ux, uy)
end

"""
    _accumulate_member_gradient!(grad, pb_ws, cc, obs_buf, tmp, om, x, y, uy,
                                 tsteps, weights)

Add one observation model's `∂ log p(yₜ|xₜ)/∂xₜ` to `grad` for a whole trial,
scaled by `weights[t]` when given.
"""
function _accumulate_member_gradient!(
    grad::AbstractMatrix{T},
    ::Union{SmoothWorkspace{T},SLDSSmoothWorkspace{T}},
    cc::SmoothConstants{T},
    obs_buf::AbstractVector{T},
    tmp::AbstractVector{T},
    om::AbstractObservationModel,
    x::AbstractMatrix{T},
    y_m::AbstractMatrix,
    uy_m::Union{Nothing,AbstractMatrix},
    tsteps::Int,
    weights::Union{Nothing,AbstractVector},
) where {T<:Real}
    @inbounds for t in 1:tsteps
        observation_gradient!(tmp, cc, obs_buf, om, x, y_m, t, uy_m)
        α = _weight_at(weights, t, T)
        @simd for i in eachindex(tmp)
            grad[i, t] += α * tmp[i]
        end
    end
    return grad
end

#=
`Cₘ'(yₜ − λₜ)` for the whole trial as two `gemm`s rather than a `tsteps`-long
loop of BLAS-2 calls, with the weights folded into the residual before the
second — which is where a spike-train fit spends most of its emission time.
=#
function _accumulate_member_gradient!(
    grad::AbstractMatrix{T},
    pb_ws::Union{SmoothWorkspace{T},SLDSSmoothWorkspace{T}},
    ::SmoothConstants{T},
    ::AbstractVector{T},
    ::AbstractVector{T},
    om::PoissonObservationModel{T},
    x::AbstractMatrix{T},
    y_m::AbstractMatrix{T},
    uy_m::Union{Nothing,AbstractMatrix},
    tsteps::Int,
    weights::Union{Nothing,AbstractVector},
) where {T<:Real}
    obs_dim, latent_dim = size(om.C)
    pb = poisson_batch!(pb_ws, latent_dim, obs_dim, tsteps)
    Eta = _poisson_linear_predictor!(pb, om.C, om.d, om.D, x, uy_m, tsteps)

    # Overwrite η in place with the (weighted) residual w(t)·(yₜ − exp(ηₜ)).
    @inbounds for t in 1:tsteps
        wt = _weight_at(weights, t, T)
        col = view(Eta, :, t)
        yt = view(y_m, :, t)
        @simd for i in 1:obs_dim
            col[i] = wt * (yt[i] - exp(col[i]))
        end
    end

    mul!(view(grad, :, 1:tsteps), transpose(om.C), Eta, one(T), one(T))
    return grad
end

"""
    hessian!(sws, lds, x, y::NamedTuple[, uy])

Block-tridiagonal Hessian of the complete-data log-likelihood for a composite
emission.

When every member is Gaussian the summed emission curvature `Σₘ -Cₘ'Rₘ⁻¹Cₘ` is
constant in `x` and already cached on `sws.consts.yt_given_xt` by
`compute_smooth_constants!`, so this is exactly the single-model assembly — one
axpy per timestep, no member loop at all.
"""
function hessian!(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::NamedTuple,
    uy::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T,true}}
    _fill_hessian_blocks!(sws, lds.state_model, size(x, 2))
    return nothing
end

#=
With a non-quadratic member present the curvature depends on the iterate, so it
is rebuilt per member per Newton step. A Gaussian member still contributes its
cached constant template; only the members that need to look at `x` do.
=#
function hessian!(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::NamedTuple,
    uy::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T,false}}
    tsteps = size(x, 2)
    _state_hessian_blocks!(sws.btd, sws.consts, lds.state_model, tsteps)

    subs = _obs_workspaces!(sws, lds)
    for (i, om) in enumerate(values(_models(lds.obs_model)))
        sub = subs[i]
        _accumulate_member_hessian!(
            sws.btd,
            sub,
            sub.consts,
            sub.elbo.rho_obs,
            sub.elbo.h_obs,
            om,
            x,
            y[i],
            _member_at(uy, i),
            tsteps,
            nothing,
        )
    end

    return nothing
end

"""
    _accumulate_member_hessian!(btd, pb_ws, cc, b1, b2, om, x, y, uy, tsteps, weights)

Add one observation model's emission curvature to the diagonal Hessian blocks,
scaled by `weights[t]` when given.
"""
function _accumulate_member_hessian!(
    btd::BlockTridiagonalWorkspace{T},
    ::Union{SmoothWorkspace{T},SLDSSmoothWorkspace{T}},
    cc::SmoothConstants{T},
    b1::AbstractVector{T},
    b2::AbstractVector{T},
    om::AbstractObservationModel,
    x::AbstractMatrix{T},
    y_m::AbstractMatrix,
    uy_m::Union{Nothing,AbstractMatrix},
    tsteps::Int,
    weights::Union{Nothing,AbstractVector},
) where {T<:Real}
    for t in 1:tsteps
        observation_hessian!(
            btd.H_diag[t], cc, b1, b2, om, x, y_m, t, _weight_at(weights, t, T), uy_m
        )
    end
    return nothing
end

#=
A Poisson member routes to the batched emission curvature — the same `gemm`
whole-trial form the single-model Poisson `hessian!` uses. Accumulates into the
diagonal blocks, so members (and, for an SLDS, regimes) compose.
=#
function _accumulate_member_hessian!(
    btd::BlockTridiagonalWorkspace{T},
    pb_ws::Union{SmoothWorkspace{T},SLDSSmoothWorkspace{T}},
    ::SmoothConstants{T},
    ::AbstractVector{T},
    ::AbstractVector{T},
    om::PoissonObservationModel{T},
    x::AbstractMatrix{T},
    y_m::AbstractMatrix,
    uy_m::Union{Nothing,AbstractMatrix},
    tsteps::Int,
    weights::Union{Nothing,AbstractVector},
) where {T<:Real}
    obs_dim, latent_dim = size(om.C)
    pb = poisson_batch!(pb_ws, latent_dim, obs_dim, tsteps)
    _poisson_emission_hessian!(btd, pb, om, x, uy_m, tsteps, weights)
    return nothing
end

# ============================================================================
# Shapes shared by the fit drivers
#
# The drivers in `fit_LDS.jl` / `fit_PLDS.jl` are written against a single
# observation model. These few accessors are what let the same code serve a
# composite: they read the one number or slice the driver actually needs,
# rather than assuming `y` is a matrix and `lds.obs_dim` sizes the buffers.
# ============================================================================

"""
    _ws_obs_dim(lds) -> Int
    _ws_uy_dim(lds) -> Int

Observation widths a `SmoothWorkspace` for `lds` should be sized at.

For a single observation model these are the model's own. For a composite they
are **zero**: the parent workspace does only state-side work, and every
emission-shaped buffer lives on a per-member sub-workspace instead. That is what
keeps a composite fit from allocating the `(Σₘ pₘ)²` buffers a naive stacking
would need.
"""
_ws_obs_dim(lds::LinearDynamicalSystem) = lds.obs_dim
_ws_uy_dim(lds::LinearDynamicalSystem) = lds.uy_dim

function _ws_obs_dim(
    ::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    return 0
end

function _ws_uy_dim(
    ::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    return 0
end

"""
    _ntsteps(y) -> Int

Trial length of one trial's observations, whether that is a single matrix or a
`NamedTuple` of per-member matrices (every member sees the same latent path, so
any member gives the answer).
"""
_ntsteps(y::AbstractMatrix) = size(y, 2)
_ntsteps(y::NamedTuple) = size(first(values(y)), 2)

"""
    _trial(y, n)

Trial `n` of a `Data` field: the trial's matrix for a single observation model,
or a `NamedTuple` of the members' matrices for a composite.
"""
_trial(y::AbstractVector{<:AbstractMatrix}, n::Int) = y[n]
_trial(y::NamedTuple, n::Int) = map(v -> v[n], y)

"""
    _zero_uy(lds, tsteps)

The "no observation inputs" value for one trial, shaped for this model: a
`0 × tsteps` matrix, or a `NamedTuple` of them for a composite.
"""
_zero_uy(::LinearDynamicalSystem{T}, tsteps::Int) where {T<:Real} = zeros(T, 0, tsteps)

function _zero_uy(
    lds::LinearDynamicalSystem{T,S,O}, tsteps::Int
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    ks = _obs_keys(lds.obs_model)
    return NamedTuple{ks}(map(_ -> zeros(T, 0, tsteps), ks))
end

"""
    _member_datas(data) -> NamedTuple of Data

One single-observation `Data` per member: that member's observations and
observation inputs, sharing the dynamics inputs and trial lengths by reference.

These are what let the per-member M-step, Q-term and sufficient-statistics
aggregation be the ordinary single-observation routines — a member's view plus
its own `Data` is indistinguishable from a single-emission model.
"""
function _member_datas(data::Data)
    ks = keys(data.y)
    return NamedTuple{ks}(map(k -> Data(data.y[k], data.ux, data.uy[k], data.tsteps), ks))
end

# ============================================================================
# Sufficient statistics
#
# One `SufficientStatistics` per member, each carrying that member's emission
# blocks *and* a copy of the shared state blocks. The state blocks come out
# bitwise identical across members — they depend only on the smoother output and
# `ux`, neither of which varies by member — so the state M-step may read any
# member's, which `_state_suf` picks.
#
# Recomputing them per member costs an O(N·T·D²) pass that the emission pass
# (O(N·T·pₘ·D), with pₘ the channel count) dominates. Paying that buys back the
# whole aggregator unchanged, cov-cache fast path and all, rather than a second
# split-out copy of one of the subtlest functions in the package. The same
# applies to the γ-weighted aggregator an SLDS uses.
# ============================================================================

"""
    _state_suf(suf)

The sufficient-statistics block the state-side M-step and `Q_state!` should
read: the single one, or any member's (they agree).
"""
_state_suf(suf::SufficientStatistics) = suf
_state_suf(suf::NamedTuple) = first(values(suf))

function _initialize_td_sufficient_statistics(
    ::Type{T}, lds::LinearDynamicalSystem{T,S,O}, tsteps_per_trial::AbstractVector{Int}
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    views = _obs_views(lds)
    return map(v -> _initialize_td_sufficient_statistics(T, v, tsteps_per_trial), views)
end

function _td_init_const_blocks!(
    sws::SmoothWorkspace{T}, lds::LinearDynamicalSystem{T,S,O}, data::Data{T}
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    subs = _obs_workspaces!(sws, lds)
    views = _obs_views(lds)
    datas = _member_datas(data)
    for (i, key) in enumerate(_obs_keys(lds.obs_model))
        _td_init_const_blocks!(subs[i], views[key], datas[key])
    end
    return nothing
end

function _aggregate_td_suff_stats!(
    suf::NamedTuple,
    tfs::TrialFilterSmooth{T},
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    subs = _obs_workspaces!(sws, lds)
    views = _obs_views(lds)
    datas = _member_datas(data)
    for (i, key) in enumerate(_obs_keys(lds.obs_model))
        _aggregate_td_suff_stats!(suf[key], tfs, views[key], datas[key], subs[i])
    end
    return suf
end

function _aggregate_td_suff_stats_weighted!(
    suf::NamedTuple,
    tfs::TrialFilterSmooth{T},
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T},
    weights::AbstractVector{<:AbstractVector{T}},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    subs = _obs_workspaces!(sws, lds)
    views = _obs_views(lds)
    datas = _member_datas(data)
    for (i, key) in enumerate(_obs_keys(lds.obs_model))
        _aggregate_td_suff_stats_weighted!(
            suf[key], tfs, views[key], datas[key], weights, subs[i]
        )
    end
    return suf
end

# ============================================================================
# E-step Q-term, M-step and ELBO
# ============================================================================

"""
    Q_obs!(sws, lds, suf::NamedTuple)

Emission Q-term of a composite: the sum over members, each evaluated by the
ordinary single-observation `Q_obs!` on that member's view, sufficient
statistics and sub-workspace.

Defined for an all-Gaussian composite, where every member has a
sufficient-statistic form. A composite containing a Poisson member goes through
the per-trial path in `fit_PLDS.jl` instead.
"""
function Q_obs!(
    sws::SmoothWorkspace{T}, lds::LinearDynamicalSystem{T,S,O}, suf::NamedTuple
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T,true}}
    subs = _obs_workspaces!(sws, lds)
    views = _obs_views(lds)
    total = zero(T)
    for (i, key) in enumerate(_obs_keys(lds.obs_model))
        total += Q_obs!(subs[i], views[key], suf[key])
    end
    return total
end

"""
    _obs_prior_logdensity(lds, sws) -> T

`log p(θ)` for a composite emission: the sum of its members' own prior terms,
each evaluated against that member's parameters and sub-workspace scratch.
"""
function _obs_prior_logdensity(
    lds::LinearDynamicalSystem{T,S,O}, sws::Union{Nothing,SmoothWorkspace{T}}
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    subs = sws === nothing ? nothing : _obs_workspaces!(sws, lds)
    views = _obs_views(lds)
    total = zero(T)
    for (i, key) in enumerate(_obs_keys(lds.obs_model))
        total += _obs_prior_logdensity(views[key], subs === nothing ? nothing : subs[i])
    end
    return total
end

"""
    mstep!(lds, suf::NamedTuple, sws)

M-step for an all-Gaussian composite: the four state updates once from any
member's (identical) state blocks, then each member's `[C d D]` and `R` from its
own statistics. Each member's `fit_bool` flags travel with its view, so freezing
one emission leaves the others free.
"""
function mstep!(
    lds::LinearDynamicalSystem{T,S,O}, suf::NamedTuple, sws::SmoothWorkspace{T}
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T,true}}
    state = _state_suf(suf)
    update_initial_state_mean!(lds, state)
    update_initial_state_covariance!(lds, state, sws)
    update_A_b!(lds, state, sws)
    update_Q!(lds, state, sws)

    subs = _obs_workspaces!(sws, lds)
    views = _obs_views(lds)
    for (i, key) in enumerate(_obs_keys(lds.obs_model))
        update_C_d!(views[key], suf[key], subs[i])
        update_R!(views[key], suf[key], subs[i])
    end
    return nothing
end

"""
    elbo!(lds, suf::NamedTuple, sws, total_entropy)

Total ELBO of an all-Gaussian composite from the aggregated sufficient
statistics: the shared state Q-term, the summed emission Q-terms, the state and
per-member emission log-priors, and the posterior entropy.
"""
function elbo!(
    lds::LinearDynamicalSystem{T,S,O},
    suf::NamedTuple,
    sws::SmoothWorkspace{T},
    total_entropy::T,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T,true}}
    Q_total = Q_state!(sws, lds, _state_suf(suf)) + Q_obs!(sws, lds, suf)
    prior_term = _state_prior_logdensity(lds, sws) + _obs_prior_logdensity(lds, sws)
    return Q_total + prior_term + total_entropy
end

#=
Public-shape return convention for `smooth` under a composite emission: the
member entries all have the same container shape, so any of them decides
whether this was a single-trial (matrix) or multi-trial call.
=#
function _collect_smooth_output(tfs::TrialFilterSmooth, y::NamedTuple)
    return _collect_smooth_output(tfs, first(values(y)))
end

"""
    _mirror_smooth_constants!(sws, source_sws, lds)

Copy the cached smoothing constants from one workspace to another, so a
per-task workspace on the equal-length fast path can reuse the Cholesky
factorizations `_precompute_shared_cov!` already did on the designated
workspace.

For a composite emission the members' constants have to travel too: the summed
curvature on the parent is enough to assemble the Hessian, but the gradient
needs each member's own `Cₘ'Rₘ⁻¹`, which lives on that member's sub-workspace.
"""
function _mirror_smooth_constants!(
    sws::SmoothWorkspace{T}, source_sws::SmoothWorkspace{T}, ::LinearDynamicalSystem{T}
) where {T<:Real}
    _copy_smooth_constants!(sws.consts, source_sws.consts)
    return nothing
end

function _mirror_smooth_constants!(
    sws::SmoothWorkspace{T},
    source_sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    _copy_smooth_constants!(sws.consts, source_sws.consts)
    src = source_sws.obs
    src === nothing && return nothing
    dst = _obs_workspaces!(sws, lds)
    for i in eachindex(dst)
        _copy_smooth_constants!(dst[i].consts, src[i].consts)
    end
    return nothing
end

"""
    _trial_lengths(y) -> Vector{Int}

Per-trial timestep counts of a multi-trial observation container, whether that
is a vector of matrices or a `NamedTuple` of them.
"""
_trial_lengths(y::AbstractVector{<:AbstractMatrix}) = Int[size(yt, 2) for yt in y]
_trial_lengths(y::NamedTuple) = _trial_lengths(first(values(y)))

"""
    _batched_ntrials(lds, ntrials) -> Int

The `ntrials` a fit's designated workspace should be built with, which decides
whether it carries the BLAS-3 mean-pass buffers.

A composite emission gets `1`, i.e. no batched buffers. The batched mean pass
stacks one `(p, T, N)` observation tensor and one `C`/`d`/`D`, which a composite
does not have; it would need per-member tensors and a per-member pass. The
equal-length fast path still applies — the covariance is computed once and
shared across trials — so this only costs a composite the BLAS-2-to-BLAS-3
promotion of the per-trial mean pass, not the shared-covariance saving.
"""
_batched_ntrials(lds::LinearDynamicalSystem, ntrials::Int) =
    _supports_batched(lds) ? ntrials : 1

"""
    _supports_batched(lds) -> Bool

Whether the BLAS-3 batched mean pass applies to this model. False for a
composite emission — see [`_batched_ntrials`](@ref) for why, and for what it
does and does not cost.
"""
_supports_batched(::LinearDynamicalSystem) = true

function _supports_batched(
    ::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    return false
end

"""
    _cell_obs_shape(lds)

The observation-side widths a workspace for this model must be built at. For a
composite it is the per-member `(obs_dim, uy_dim)` pairs, since those are what
the sub-workspaces are shaped by — the parent's own widths are zero either way.

Used to decide whether the cells of a grouped fit can share one workspace pool
or need their own: under stitching each session has its own channel counts.
"""
_cell_obs_shape(lds::LinearDynamicalSystem) = (lds.obs_dim, lds.uy_dim)

function _cell_obs_shape(
    lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    return map(m -> (_obs_dim(m), _uy_dim(m)), values(_models(lds.obs_model)))
end

"""
    joint_loglikelihood(lds, x, y::NamedTuple[, ux, uy])

Complete-data log-likelihood `log p(x, y)` of a composite emission at the given
latent path, summed over timesteps. Allocating convenience wrapper; the
element type is promoted across the latent path and every member's
observations, so a `Float32` iterate against `Float64` data works as it does for
a single observation model.
"""
function joint_loglikelihood(
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{XT},
    y::NamedTuple,
    ux::Union{Nothing,AbstractMatrix}=nothing,
    uy::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,XT<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    tsteps = _ntsteps(y)
    WT = promote_type(T, XT, mapreduce(eltype, promote_type, values(y)))
    ws = SmoothWorkspace(WT, lds.latent_dim, 0, tsteps)
    compute_smooth_constants!(ws, lds)
    ll = view(ws.opt.ll_vec, 1:tsteps)
    return joint_loglikelihood!(ll, ws, ws.consts, lds, x, y, ux, uy)
end

# ============================================================================
# Marginal log-likelihood
#
# The Kalman filter's innovation covariance `S_t = C P_t C' + R` is dense
# whatever the emission's block structure, so unlike the E-step there is nothing
# to gain from keeping the members apart here. An all-Gaussian composite is
# therefore evaluated by building the equivalent single-emission model once —
# `C = [C₁; C₂; …]`, `d = [d₁; d₂; …]`, `R = blockdiag(R₁, R₂, …)`,
# `D = blockdiag(D₁, D₂, …)` — and running the existing filter on it.
# ============================================================================

"""
    _stacked_gaussian_lds(lds) -> LinearDynamicalSystem

The single-emission model equivalent to an all-Gaussian composite: members
stacked down the channel axis, with `R` and `D` block-diagonal so the members
stay conditionally independent given the latent path.

Built fresh (the parameters are copied, not shared), so it is a read-only
snapshot — fitting through it would produce a full `R` rather than the composite's
block-diagonal one.
"""
function _stacked_gaussian_lds(
    lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T,true}}
    models = _models(lds.obs_model)
    D = lds.latent_dim
    p = lds.obs_dim
    uy = lds.uy_dim

    C = zeros(T, p, D)
    d = zeros(T, p)
    Dm = zeros(T, p, uy)
    R = zeros(T, p, p)

    row = 0
    col = 0
    for m in values(models)
        pm = _obs_dim(m)
        um = _uy_dim(m)
        rows = (row + 1):(row + pm)
        copyto!(view(C, rows, :), m.C)
        copyto!(view(d, rows), m.d)
        copyto!(view(R, rows, rows), m.R)
        um > 0 && copyto!(view(Dm, rows, (col + 1):(col + um)), m.D)
        row += pm
        col += um
    end

    om = GaussianObservationModel{T,Matrix{T},Vector{T}}(; C=C, R=R, d=d, D=Dm)
    return LinearDynamicalSystem{T,S,typeof(om)}(
        lds.state_model, om, D, p, lds.ux_dim, uy, fill(true, 6)
    )
end

# Per-trial `vcat` of the members' observations / observation inputs, matching
# the row order `_stacked_gaussian_lds` stacks the parameters in.
function _stack_trials(bundle::NamedTuple, ntrials::Int)
    return [reduce(vcat, (v[n] for v in values(bundle))) for n in 1:ntrials]
end

"""
    loglikelihood(lds, y::NamedTuple; ux=nothing, uy=nothing)

Marginal (observed-data) log-likelihood of an all-Gaussian composite emission,
with the latent states integrated out. Equal to the value the equivalent stacked
single-emission model gives (see [`_stacked_gaussian_lds`](@ref)), which is what
it is computed from.

A composite containing a non-Gaussian member has no tractable marginal, exactly
as a Poisson LDS does not; use `elbo` for a lower bound.

Returns the **total** log-likelihood over every member, trial and timestep.
"""
function StatsAPI.loglikelihood(
    lds::LinearDynamicalSystem{T,SM,OM},
    y::NamedTuple;
    ux=nothing,
    uy=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,SM<:AbstractGaussianStateModel{T},OM<:CompositeObservationModel{T,true}}
    data = Data(lds, y; ux=ux, uy=uy)
    ntrials = length(data.tsteps)

    grp = parameter_grouping(lds, ntrials; depends_on=depends_on, y=data.y)
    grp === nothing || throw(
        ArgumentError(
            "marginal loglikelihood of a composite emission whose parameters depend on " *
            "an ancillary variable is not implemented; score each group separately, or " *
            "use `elbo`, which handles grouped models.",
        ),
    )

    stacked = _stacked_gaussian_lds(lds)
    ys = _stack_trials(data.y, ntrials)
    uys = lds.uy_dim > 0 ? _stack_trials(data.uy, ntrials) : nothing
    return loglikelihood(stacked, ys; ux=data.ux, uy=uys)
end

function StatsAPI.loglikelihood(
    lds::LinearDynamicalSystem{T,SM,OM}, y::NamedTuple; kwargs...
) where {T<:Real,SM<:AbstractGaussianStateModel{T},OM<:CompositeObservationModel{T,false}}
    return error(
        "marginal loglikelihood is not implemented for a composite emission with a " *
        "non-Gaussian member (the marginal log p(y) is intractable, as it is for the " *
        "Poisson LDS). Use `elbo` for a lower bound, or `joint_loglikelihood` for the " *
        "complete-data log-likelihood at a given latent path.",
    )
end

# ============================================================================
# Non-quadratic composites: the Laplace path
#
# A composite with a non-Gaussian member has an emission curvature that depends
# on the latent path, so it goes through the iterative-Newton driver in
# `fit_PLDS.jl`. That driver is emission-agnostic apart from two hooks — the
# hoisted log-normalizer and the line-search objective — which dispatch here.
# ============================================================================

"""
    _emission_lognorm(lds, y::NamedTuple) -> NamedTuple

Each member's data-only log-normalizer for one trial, or `nothing` for a member
that has none. A Gaussian emission folds its normalizer into `cR`, so only the
Poisson members contribute anything to hoist.
"""
function _emission_lognorm(
    lds::LinearDynamicalSystem{T,S,O}, y::NamedTuple
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    models = _models(lds.obs_model)
    norms = map(key -> _member_lognorm(models[key], y[key]), keys(models))
    # Nothing to hoist when no member has a normalizer (every member Gaussian);
    # `nothing` then travels the same way a single Gaussian emission's does.
    all(isnothing, norms) && return nothing
    return NamedTuple{keys(models)}(norms)
end

_member_lognorm(::AbstractObservationModel, ::AbstractMatrix) = nothing
_member_lognorm(::PoissonObservationModel, y_m::AbstractMatrix) = _poisson_lognorm_t(y_m)

function _joint_loglikelihood_total(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::NamedTuple,
    lognorms::NamedTuple,
    ux::Union{Nothing,AbstractMatrix},
    uy::Union{Nothing,NamedTuple},
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    ll = view(sws.opt.ll_vec, 1:_ntsteps(y))
    joint_loglikelihood!(ll, sws, sws.consts, lds, x, y, ux, uy, lognorms)
    return sum(ll)
end

"""
    _member_pools(sws_pool, lds) -> Vector{Vector{SmoothWorkspace}}

Transpose the pool: `_member_pools(pool, lds)[m]` is member `m`'s sub-workspace
from every workspace in `pool`, which is the pool shape the per-trial parallel
routines (`_poisson_q_obs_total`, `update_observation_model!`) expect.
"""
function _member_pools(
    sws_pool::Vector{SmoothWorkspace{T}}, lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    per_ws = [_obs_workspaces!(ws, lds) for ws in sws_pool]
    nmembers = length(first(per_ws))
    return [SmoothWorkspace{T}[subs[m] for subs in per_ws] for m in 1:nmembers]
end

#=
One member's emission Q-term. A Gaussian member has a sufficient-statistic form
and reads it straight off its own `suf`; a Poisson member is irreducibly
non-conjugate and stays the per-trial loop, chunked over its slice of the pool.
=#
function _member_q_obs(
    view_m::LinearDynamicalSystem{T,S,O},
    suf_m::SufficientStatistics{T},
    ::TrialFilterSmooth{T},
    ::Data{T},
    pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:GaussianObservationModel{T}}
    return Q_obs!(pool[1], view_m, suf_m)
end

function _member_q_obs(
    view_m::LinearDynamicalSystem{T,S,O},
    ::SufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    data_m::Data{T},
    pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:PoissonObservationModel{T}}
    return _poisson_q_obs_total(view_m, tfs, data_m, pool)
end

"""
    _composite_q_obs_total(lds, suf, tfs, data, sws_pool) -> T

Emission Q-term of a composite, summed over members by whichever route each
member supports. Used by the non-quadratic ELBO and by its grouped counterpart,
which evaluates the same quantity one cell at a time.
"""
function _composite_q_obs_total(
    lds::LinearDynamicalSystem{T,S,O},
    suf::NamedTuple,
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    views = _obs_views(lds)
    datas = _member_datas(data)
    pools = _member_pools(sws_pool, lds)
    total = zero(T)
    for (i, key) in enumerate(_obs_keys(lds.obs_model))
        total += _member_q_obs(views[key], suf[key], tfs, datas[key], pools[i])
    end
    return total
end

"""
    elbo!(lds, suf::NamedTuple, tfs, data, sws_pool)

Total ELBO of a composite emission with a non-Gaussian member. Mirrors the
Poisson LDS split: the state Q-term from the aggregated sufficient statistics,
each member's emission Q-term by whichever route that member supports, the
state and per-member log-priors, and the posterior entropy from the smoother.
"""
function elbo!(
    lds::LinearDynamicalSystem{T,S,O},
    suf::NamedTuple,
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T,false}}
    total_entropy = zero(T)
    for fs in tfs.FilterSmooths
        total_entropy += fs.entropy
    end

    compute_smooth_constants!(sws_pool[1], lds)
    total = Q_state!(sws_pool[1], lds, _state_suf(suf))
    total += _composite_q_obs_total(lds, suf, tfs, data, sws_pool)

    total += _state_prior_logdensity(lds, sws_pool[1])
    total += _obs_prior_logdensity(lds, sws_pool[1])

    return total + total_entropy
end

#=
One member's emission M-step. Gaussian: the conjugate regression and IW update
from its sufficient statistics. Poisson: the row-wise Newton solver over that
member's trials. Both are the ordinary single-observation routines, reached
through the member's view so they write back to the member's own parameters and
respect its own `fit_bool` flags.
=#
function _member_obs_mstep!(
    view_m::LinearDynamicalSystem{T,S,O},
    suf_m::SufficientStatistics{T},
    ::TrialFilterSmooth{T},
    ::Data{T},
    pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:GaussianObservationModel{T}}
    update_C_d!(view_m, suf_m, pool[1])
    update_R!(view_m, suf_m, pool[1])
    return nothing
end

function _member_obs_mstep!(
    view_m::LinearDynamicalSystem{T,S,O},
    ::SufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    data_m::Data{T},
    pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:PoissonObservationModel{T}}
    update_observation_model!(view_m, tfs, data_m.y, pool; uy=data_m.uy)
    return nothing
end

"""
    mstep!(lds, suf::NamedTuple, tfs, data, sws_pool)

M-step for a composite emission with a non-Gaussian member: the four state
updates once from any member's (identical) state blocks, then each member's own
emission update.
"""
function mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    suf::NamedTuple,
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T,false}}
    sws = sws_pool[1]
    state = _state_suf(suf)
    update_initial_state_mean!(lds, state)
    update_initial_state_covariance!(lds, state, sws)
    update_A_b!(lds, state, sws)
    update_Q!(lds, state, sws)

    views = _obs_views(lds)
    datas = _member_datas(data)
    pools = _member_pools(sws_pool, lds)
    for (i, key) in enumerate(_obs_keys(lds.obs_model))
        _member_obs_mstep!(views[key], suf[key], tfs, datas[key], pools[i])
    end
    return nothing
end

# ============================================================================
# SLDS emission kernels
#
# One regime of an SLDS reaches the same per-member accumulators the single-LDS
# path uses, with that regime's responsibilities `γₖ(t)` as the per-timestep
# weight and its scratch coming from `ObsScratch` rather than a
# sub-`SmoothWorkspace`. Contributions accumulate, so members compose within a
# regime exactly as regimes compose within a trial.
# ============================================================================

function _slds_emission_loglik!(
    out::AbstractVector{T},
    ws::SLDSSmoothWorkspace{T},
    ::SmoothConstants{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::NamedTuple,
    uy::Union{Nothing,NamedTuple},
    tsteps::Int,
    lognorm_t::Union{Nothing,NamedTuple},
    obs_scratch::Union{Nothing,Vector{ObsScratch{T}}},
) where {T<:Real,S<:AbstractStateModel,O<:CompositeObservationModel{T}}
    obs_scratch === nothing && throw(
        ArgumentError(
            "this SLDS workspace carries no per-member emission scratch, so it was not " *
            "built for a composite emission; construct it with " *
            "`SLDSSmoothWorkspace(T, slds, tsteps)` from the SLDS being smoothed",
        ),
    )
    scratch = obs_scratch
    @inbounds for t in 1:tsteps
        out[t] = zero(T)
    end
    for (i, om) in enumerate(values(_models(lds.obs_model)))
        sc = scratch[i]
        _accumulate_member_loglikelihood!(
            out,
            ws,
            sc.consts,
            sc.buf1,
            sc.buf2,
            om,
            x,
            y[i],
            _member_at(uy, i),
            tsteps,
            _member_at(lognorm_t, i),
            nothing,
        )
    end
    return out
end

function _slds_emission_gradient!(
    grad::AbstractMatrix{T},
    ws::SLDSSmoothWorkspace{T},
    ::SmoothConstants{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::NamedTuple,
    weights::AbstractVector{T},
    uy::Union{Nothing,NamedTuple},
    tsteps::Int,
    tmp::AbstractVector{T},
    ::AbstractVector{T},
    obs_scratch::Union{Nothing,Vector{ObsScratch{T}}},
) where {T<:Real,S<:AbstractStateModel,O<:CompositeObservationModel{T}}
    obs_scratch === nothing && throw(
        ArgumentError(
            "this SLDS workspace carries no per-member emission scratch, so it was not " *
            "built for a composite emission; construct it with " *
            "`SLDSSmoothWorkspace(T, slds, tsteps)` from the SLDS being smoothed",
        ),
    )
    scratch = obs_scratch
    for (i, om) in enumerate(values(_models(lds.obs_model)))
        sc = scratch[i]
        _accumulate_member_gradient!(
            grad,
            ws,
            sc.consts,
            sc.buf1,
            tmp,
            om,
            x,
            y[i],
            _member_at(uy, i),
            tsteps,
            weights,
        )
    end
    return nothing
end

function _slds_emission_hessian!(
    ws::SLDSSmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    ::SmoothConstants{T},
    x::AbstractMatrix{T},
    y::NamedTuple,
    weights::AbstractVector{T},
    uy::Union{Nothing,NamedTuple},
    tsteps::Int,
    ::AbstractVector{T},
    ::AbstractVector{T},
    obs_scratch::Union{Nothing,Vector{ObsScratch{T}}},
) where {T<:Real,S<:AbstractStateModel,O<:CompositeObservationModel{T}}
    obs_scratch === nothing && throw(
        ArgumentError(
            "this SLDS workspace carries no per-member emission scratch, so it was not " *
            "built for a composite emission; construct it with " *
            "`SLDSSmoothWorkspace(T, slds, tsteps)` from the SLDS being smoothed",
        ),
    )
    scratch = obs_scratch
    for (i, om) in enumerate(values(_models(lds.obs_model)))
        sc = scratch[i]
        _accumulate_member_hessian!(
            ws.btd,
            ws,
            sc.consts,
            sc.buf1,
            sc.buf2,
            om,
            x,
            y[i],
            _member_at(uy, i),
            tsteps,
            weights,
        )
    end
    return nothing
end

"""
    _emission_curvature_at!(H_obs, ws, cc, lds_k, x, y, t, uy, obs_scratch)

One regime's *unweighted* emission curvature at a single timestep, written into
`H_obs` (which the caller has zeroed). Used by the ELBO's `½ tr(H Σ)` covariance
correction, which needs the regime's curvature in isolation.

Summed over members for a composite emission.
"""
function _emission_curvature_at!(
    H_obs::AbstractMatrix{T},
    ws::SLDSSmoothWorkspace{T},
    cc::SmoothConstants{T},
    lds_k::LinearDynamicalSystem{T},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    t::Int,
    uy::Union{Nothing,AbstractMatrix},
    ::Nothing,
) where {T<:Real}
    observation_hessian!(
        H_obs, cc, ws.opt.dyt, ws.opt.temp_dy, lds_k.obs_model, x, y, t, one(T), uy
    )
    return H_obs
end

function _emission_curvature_at!(
    H_obs::AbstractMatrix{T},
    ::SLDSSmoothWorkspace{T},
    ::SmoothConstants{T},
    lds_k::LinearDynamicalSystem{T},
    x::AbstractMatrix{T},
    y::NamedTuple,
    t::Int,
    uy::Union{Nothing,NamedTuple},
    obs_scratch::Vector{ObsScratch{T}},
) where {T<:Real}
    for (i, om) in enumerate(values(_models(lds_k.obs_model)))
        sc = obs_scratch[i]
        observation_hessian!(
            H_obs, sc.consts, sc.buf1, sc.buf2, om, x, y[i], t, one(T), _member_at(uy, i)
        )
    end
    return H_obs
end

"""
    _ntrials(y) -> Int

Number of trials in a multi-trial observation container, whether that is a
vector of per-trial matrices or a `NamedTuple` of them (the members all see the
same trials).
"""
_ntrials(y::AbstractVector{<:AbstractMatrix}) = length(y)
_ntrials(y::NamedTuple) = length(first(values(y)))

"""
    _poisson_lognorm_one(lds, y::NamedTuple)

Per-member hoisted normalizers for a composite emission — the same thing
[`_emission_lognorm`](@ref) returns, reached through the name the SLDS path uses.
"""
function _poisson_lognorm_one(
    lds::LinearDynamicalSystem{T,S,O}, y::NamedTuple
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    return _emission_lognorm(lds, y)
end

#=
An SLDS whose emission is a composite: the latent path switches regime per
timestep, so each member's observations are drawn from whichever regime's
parameters `zₜ` selects. As on the single-LDS path the path is drawn in full
first — with several emissions there is no one interleaving to pick.
=#
function _sample_continuous_given_discrete!(
    rng,
    x_trial,
    y_trial,
    z_trial,
    state_params,
    obs_params,
    obs_model::CompositeObservationModel,
    ux_trial::AbstractMatrix,
    uy_trial,
)
    tsteps = length(z_trial)

    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    for t in 2:tsteps
        k = z_trial[t]
        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params[k].A * x_trial[:, t - 1] +
                state_params[k].b +
                state_params[k].B * ux_trial[:, t - 1],
                state_params[k].Q,
            ),
        )
    end

    for (i, m) in enumerate(values(_models(obs_model)))
        y_m = y_trial[i]
        uy_m = uy_trial[i]
        @views for t in 1:tsteps
            y_m[:, t] = _draw_obs(
                rng, m, obs_params[z_trial[t]][i], x_trial[:, t], uy_m[:, t]
            )
        end
    end
    return nothing
end
