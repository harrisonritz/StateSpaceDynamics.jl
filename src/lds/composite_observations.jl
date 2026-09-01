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
    ws::SmoothWorkspace{T}, lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    models = _models(lds.obs_model)
    tsteps = length(ws.opt.ll_vec)
    subs = ws.obs
    if subs !== nothing && _obs_workspaces_fit(subs, lds, tsteps)
        return subs
    end
    fresh = SmoothWorkspace{T}[
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
    subs::Vector{SmoothWorkspace{T}}, lds::LinearDynamicalSystem{T,S,O}, tsteps::Int
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
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
) where {WT<:Real,T<:Real,S<:GaussianStateModel{T},O<:CompositeObservationModel{T}}
    subs = _obs_workspaces!(ws, lds)
    cc = ws.consts

    _compute_state_constants!(cc, lds.state_model)

    #=
    No aggregate `C'R⁻¹` exists — the gradient needs each member's own residual —
    so the parent's slot is cleared and the per-member ones live on the subs.
    =#
    fill!(cc.C_inv_R, zero(WT))
    fill!(cc.yt_given_xt, zero(WT))
    cR = zero(WT)

    for (i, om) in enumerate(values(_models(lds.obs_model)))
        sub_cc = subs[i].consts
        _compute_obs_constants!(sub_cc, om)
        cc.yt_given_xt .+= sub_cc.yt_given_xt
        cR += sub_cc.cR
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
function Data(
    lds::LinearDynamicalSystem{T,S,O},
    ::Union{AbstractMatrix,AbstractArray{<:Any,3},AbstractVector{<:AbstractMatrix}};
    kwargs...,
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    return throw(
        ArgumentError(
            "this model has several observation models, so observations must be a " *
            "NamedTuple keyed by them, e.g. " *
            "`($(first(_obs_keys(lds.obs_model))) = y, ...)`. Members are " *
            "$(_key_list(_models(lds.obs_model))).",
        ),
    )
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
