#=============================================================================
Spline-Gaussian emissions in a Switching Linear Dynamical System.

The warp is **shared across regimes**. That is a modelling choice, and the right
one: a per-channel monotone distortion is a property of the measurement — a
saturating sensor, a rectified rate, a skewed marginal — not of which dynamical
regime the system happens to be in. An `SLDS` then discovers regimes of dynamics
*on* a fixed nonlinear manifold, which is what makes the regimes comparable. It
is also the only version with any hope of being identified: a per-regime warp
would be estimated from the fraction of timesteps that regime occupies, and
would soak up the very differences the discrete state is supposed to explain.

Sharing the warp makes the integration almost free. With one `g`,

```math
log p(y_t | x_t, z_t = k) = log N(g(y_t); C_k x_t + d_k, R_k) + Σ_j log g_j'(y_{tj})
```

and the Jacobian term is **the same for every `k`**. So it cancels out of the
discrete posterior entirely — `q(z)` is exactly what the existing forward-backward
pass computes on the embedded observations — and enters the ELBO as a single
additive term. Every SLDS routine therefore runs unchanged on a shadow `SLDS`
whose regimes carry Gaussian emissions sharing this model's arrays, fed the
embedded observations.

Only the warp's own conditional-maximization step is new, and it is the mixture
form the objective in `spline_gaussian_observations.jl` is already written for:

```math
Q(φ) = -½ Σ_{n,t} Σ_k γ_{k,n,t} ‖L_k⁻¹(g_φ(y_{nt}) - μ̂^{(k)}_{nt})‖²
       + Σ_{n,t,j} log g'_{φ_j}(y_{ntj})
```

with `γ` the responsibilities from forward-backward and `μ̂^{(k)}` regime `k`'s
readout of the one shared smoothed path. Since `Σ_k γ_{k,n,t} = 1`, the
log-Jacobian is counted exactly once per sample.
=============================================================================#

"""
    _slds_is_warped(slds) -> Bool

Whether this `SLDS`'s emission carries a learned warp. Regimes share an emission
*type*, so regime 1 decides.
"""
_slds_is_warped(slds::SLDS) = _has_warped_member(slds.LDSs[1].obs_model)

#=
Every warped emission of a model, paired with its member key (`nothing` for a
standalone spline emission). Used to walk the regimes in lockstep.
=#
_warped_members(om::SplineGaussianObservationModel) = ((nothing, om),)
function _warped_members(c::CompositeObservationModel)
    models = _models(c)
    return Tuple(
        (key, models[key]) for
        key in _obs_keys(c) if models[key] isa SplineGaussianObservationModel
    )
end

#=
Resolve a member key against a model or an observation bundle. `nothing` is the
standalone-emission case and a `Symbol` the composite one; both are written
against the abstract emission so they also resolve against a Gaussian shadow.

The two impossible pairings are given methods that throw rather than left to a
`MethodError`: `key` is a `Union{Nothing,Symbol}` field, so every call site
union-splits over all four combinations.
=#
@inline _member_om(om::AbstractObservationModel, ::Nothing) = om
@inline _member_om(c::CompositeObservationModel, key::Symbol) = _models(c)[key]
function _member_om(om::AbstractObservationModel, key::Symbol)
    return throw(
        ArgumentError(
            "`:$key` names a composite member, but this emission is a " *
            "$(nameof(typeof(om))) with no members",
        ),
    )
end

@inline _member_y(y::AbstractVector, ::Nothing) = y
@inline _member_y(y::NamedTuple, key::Symbol) = y[key]
function _member_y(::NamedTuple, ::Nothing)
    return throw(
        ArgumentError(
            "observations are keyed by composite member, but no member key was given"
        ),
    )
end
function _member_y(::AbstractVector, key::Symbol)
    return throw(
        ArgumentError("observations are not keyed, so `:$key` names nothing in them")
    )
end

"""
    _slds_share_warps!(slds) -> slds

Bind every regime's warp (per warped member) to regime 1's object, so the shared
warp the algorithm assumes is also literally one array set. Warns when the
regimes' warps disagree numerically, since collapsing them then discards
whatever was seeded into regimes 2…K.

# Throws
- `ArgumentError` when the regimes' warps differ in shape or knot interval.
"""
function _slds_share_warps!(slds::SLDS{T}) where {T<:Real}
    K = length(slds.LDSs)
    K > 1 || return slds
    om1 = slds.LDSs[1].obs_model
    for (key, _) in _warped_members(om1)
        ref = _member_om(om1, key).warp
        differed = false
        for k in 2:K
            om_k = _member_om(slds.LDSs[k].obs_model, key)
            w = om_k.warp
            w === ref && continue
            (warp_channels(w) == warp_channels(ref) && warp_bins(w) == warp_bins(ref)) ||
                throw(
                    ArgumentError(
                        "regime $k's warp is $(warp_channels(w))×$(warp_bins(w)) but " *
                        "regime 1's is $(warp_channels(ref))×$(warp_bins(ref)); an " *
                        "SLDS shares one warp across regimes, so they must match",
                    ),
                )
            (w.lo == ref.lo && w.hi == ref.hi) || throw(
                ArgumentError(
                    "regime $k's warp has a different knot interval from regime 1's; " *
                    "an SLDS shares one warp across regimes, so build every regime's " *
                    "emission with the same `bounds` (or the same `y`)",
                ),
            )
            differed |= !(w.θw == ref.θw && w.θh == ref.θh && w.θd == ref.θd)
            om_k.warp = ref
        end
        differed && @warn(
            "SLDS regimes carried different warps; they have been collapsed onto " *
                "regime 1's. An SLDS fits one warp shared by all regimes — see the " *
                "`SplineGaussianObservationModel` docs for why.",
            member = key === nothing ? :emission : key,
            maxlog = 1,
        )
    end
    return slds
end

"""
    _SLDSSplineSite{T,YV}

One warped member of an `SLDS`: the shared warp, the per-regime emissions that
read through it, the embedding buffers, the per-regime smoothed emission means,
and the mixture M-step context.
"""
struct _SLDSSplineSite{T<:Real,YV<:AbstractVector{<:AbstractMatrix{T}}}
    key::Union{Nothing,Symbol}
    om::SplineGaussianObservationModel{T}
    z::Vector{Matrix{T}}
    mu::Vector{Vector{Matrix{T}}}
    ctx::_SplineMStepCtx{T,YV}
    θ0::Vector{T}
    θ1::Vector{T}
    logjac::Base.RefValue{T}
    fit_warp::Bool
end

"""
    _SLDSSplineState{T,DT}

Per-fit state for a warped `SLDS`: the shadow `SLDS` the existing machinery runs
on, the shadow `Data` holding the embedded observations, the warped members'
sites, and the responsibility buffers the mixture objective reads.
"""
struct _SLDSSplineState{T<:Real,SL,DT}
    shadow::SL
    sdata::DT
    sites::Vector{_SLDSSplineSite{T,Vector{Matrix{T}}}}
    gamma::Vector{Vector{Vector{T}}}
    logjac::Base.RefValue{T}
end

"""
    _slds_gaussian_shadow(slds) -> SLDS

The `SLDS` with every regime's warped emission replaced by its Gaussian shadow.
The discrete transition matrix, the initial distribution and every regime's
state model and emission arrays are shared by reference, so the existing
discrete and continuous M-steps write straight through to the real model.
"""
function _slds_gaussian_shadow(slds::SLDS{T}) where {T<:Real}
    ldss = [_gaussian_shadow(l) for l in slds.LDSs]
    return SLDS(; A=slds.A, πₖ=slds.πₖ, LDSs=ldss)
end

"""
    _slds_spline_state(slds, data) -> _SLDSSplineState or nothing

Build the fit state for a warped `SLDS`, or `nothing` when the emission carries
no warp. Also performs the initial embedding, so the warm start already sees
`z = g(y)`.
"""
function _slds_spline_state(slds::SLDS{T}, data::Data{T}) where {T<:Real}
    _slds_is_warped(slds) || return nothing
    for l in slds.LDSs
        _reject_spline_lqr(l)
    end
    _reject_spline_grouping(slds.LDSs[1])
    _slds_share_warps!(slds)

    K = length(slds.LDSs)
    shadow = _slds_gaussian_shadow(slds)

    gamma = [[zeros(T, Ti) for Ti in data.tsteps] for _ in 1:K]

    om1 = slds.LDSs[1].obs_model
    sites = _SLDSSplineSite{T,Vector{Matrix{T}}}[]
    zmap = Dict{Symbol,Any}()
    zsingle = nothing
    for (key, om) in _warped_members(om1)
        y_m = collect(Matrix{T}, _member_y(data.y, key))
        p_m = _obs_dim(om)
        z = [Matrix{T}(undef, p_m, Ti) for Ti in data.tsteps]
        mu = [[Matrix{T}(undef, p_m, Ti) for Ti in data.tsteps] for _ in 1:K]
        ctx = _SplineMStepCtx(om, y_m, mu; gamma=gamma)
        nθ = warp_nparams(om.warp)
        fit_warp = _warp_fit_flag(slds.LDSs[1], key)
        push!(
            sites,
            _SLDSSplineSite{T,Vector{Matrix{T}}}(
                key, om, z, mu, ctx, zeros(T, nθ), zeros(T, nθ), Ref(zero(T)), fit_warp
            ),
        )
        key === nothing ? (zsingle = z) : (zmap[key] = z)
    end

    sy = if zsingle !== nothing
        zsingle
    else
        NamedTuple{keys(data.y)}(map(k -> get(zmap, k, data.y[k]), keys(data.y)))
    end
    sdata = Data(sy, data.ux, data.uy, data.tsteps)

    state = _SLDSSplineState{T,typeof(shadow),typeof(sdata)}(
        shadow, sdata, sites, gamma, Ref(zero(T))
    )
    _slds_spline_embed!(state, data)
    return state
end

#=
The warp's `fit_bool` slot: the third of a standalone spline emission's block,
or the third of the member's block in a composite.
=#
_warp_fit_flag(lds::LinearDynamicalSystem, ::Nothing) = lds.fit_bool[7]
function _warp_fit_flag(lds::LinearDynamicalSystem, key::Symbol)
    return lds.fit_bool[_obs_fit_range(lds.obs_model, key)[3]]
end

"""
    _slds_spline_embed!(state, data) -> T

Refresh every site's embedding from the raw observations, and record and return
the total log-Jacobian. Shared by all regimes, so this runs once per E-step
rather than once per regime.
"""
function _slds_spline_embed!(state::_SLDSSplineState{T}, data::Data{T}) where {T<:Real}
    total = zero(T)
    for site in state.sites
        y_m = _member_y(data.y, site.key)
        acc = zero(T)
        for n in eachindex(y_m)
            acc += warp_apply!(site.z[n], site.om.warp, y_m[n])
        end
        site.logjac[] = acc
        total += acc
    end
    state.logjac[] = total
    return total
end

"""
    _slds_spline_logprior(T, state) -> T

The shared warp's ridge log-prior, counted once (not once per regime).
"""
function _slds_spline_logprior(::Type{T}, state::_SLDSSplineState) where {T<:Real}
    total = zero(T)
    for site in state.sites
        total += _spline_logprior(site.om)
    end
    return total
end

"""
    _slds_spline_gamma!(state, fb_storage, seq_ends)

Copy the forward-backward responsibilities into the per-regime, per-trial
buffers the mixture objective indexes. Copied rather than viewed so the context
stays concretely typed and the objective can run in parallel over trials.
"""
function _slds_spline_gamma!(
    state::_SLDSSplineState{T}, fb_storage, seq_ends::AbstractVector{Int}
) where {T<:Real}
    K = length(state.gamma)
    ntrials = length(state.gamma[1])
    γ = fb_storage.γ
    for trial in 1:ntrials
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        for k in 1:K
            dst = state.gamma[k][trial]
            @inbounds for (i, t) in enumerate(t1:t2)
                dst[i] = γ[k, t]
            end
        end
    end
    return state
end

"""
    _slds_project_R!(state)

Apply each warped member's `R_structure` and `R_floor` to every regime.

The SLDS Gaussian M-step fits an unconstrained `R` per regime (or per tied
group) as `R = S / N`, or `(Ψ + S) / (ν + N + p + 1)` under an `IWPrior`. Both
act entrywise on the scatter, so the diagonal of the unconstrained answer *is*
the diagonal-restricted MLE / MAP — projecting afterwards is exact, not an
approximation.
"""
function _slds_project_R!(state::_SLDSSplineState{T}) where {T<:Real}
    for lds in state.shadow.LDSs, site in state.sites
        R = _member_om(lds.obs_model, site.key).R
        if site.om.R_structure === :diagonal
            @inbounds for j in axes(R, 1), i in axes(R, 1)
                i == j || (R[i, j] = zero(T))
            end
        end
        _apply_R_floor!(R, site.om.R_floor)
    end
    return state
end

"""
    _slds_spline_mstep!(state, tfs, fb_storage, seq_ends; spline_iters)

The warp's conditional-maximization step for an `SLDS`: project each regime's
`R`, cache every regime's smoothed emission means, refresh the responsibilities
and the per-regime Cholesky factors, then run one L-BFGS solve per warped
member against the responsibility-weighted mixture objective.
"""
function _slds_spline_mstep!(
    state::_SLDSSplineState{T},
    tfs::TrialFilterSmooth{T},
    fb_storage,
    seq_ends::AbstractVector{Int};
    spline_iters::Int=25,
) where {T<:Real}
    _slds_project_R!(state)
    _slds_spline_gamma!(state, fb_storage, seq_ends)

    K = length(state.shadow.LDSs)
    for site in state.sites
        uy_m = _member_y(state.sdata.uy, site.key)
        for k in 1:K
            gom = _member_om(state.shadow.LDSs[k].obs_model, site.key)
            _spline_emission_means!(site.mu[k], gom, tfs, uy_m)
            site.ctx.Rchol[k] = cholesky(Symmetric(Matrix{T}(gom.R)))
        end
        _spline_warp_mstep!(
            site.om, site.ctx, site.θ0, site.θ1; iters=spline_iters, fit_warp=site.fit_warp
        )
    end
    return state
end

# ============================================================================
# Scoring entry points
# ============================================================================

"""
    _slds_spline_smooth(slds, y, kwargs) -> NamedTuple

`smooth` for a warped `SLDS`: run the ordinary alternating smoother on the
shadow model and the embedded observations, then move the reported ELBO back to
the observation scale by adding the change-of-variables term — total to `.elbo`,
and per trial to `.trial_elbo`, which is where it belongs (it is a property of
the trial's own observations).
"""
function _slds_spline_smooth(
    slds::SLDS{T},
    y,
    ux,
    uy,
    smoothing_iters::Int,
    tol::Real,
    return_cov::Bool,
    progress::Bool,
    npool::Int,
) where {T<:Real}
    data = Data(slds.LDSs[1], y; ux=ux, uy=uy)
    #=
    `_slds_spline_state` returns `nothing` for an unwarped model, and every
    caller reaches here only after `_slds_is_warped`. Assert it rather than
    leaving the rest of the function to work through a `Union{Nothing,…}`.
    =#
    state = _slds_spline_state(slds, data)
    state === nothing &&
        throw(ArgumentError("this SLDS carries no warp; use the ordinary `smooth`"))
    out = smooth(
        state.shadow,
        state.sdata.y;
        ux=state.sdata.ux,
        uy=state.sdata.uy,
        smoothing_iters=smoothing_iters,
        tol=tol,
        return_cov=return_cov,
        progress=progress,
        npool=npool,
    )

    per_trial = _slds_trial_logjac(state, data)
    trial_elbo = out.trial_elbo .+ per_trial
    total = out.elbo + state.logjac[] + _slds_spline_logprior(T, state)
    return (; x=out.x, γ=out.γ, elbo=total, trial_elbo=trial_elbo, p=out.p)
end

#=
Each trial's own log-Jacobian, summed over warped members. `_slds_spline_embed!`
keeps only the total, and a per-trial ELBO has to be exact for
`sum(trial_elbo) + log p(θ) == elbo` to hold.
=#
function _slds_trial_logjac(state::_SLDSSplineState{T}, data::Data{T}) where {T<:Real}
    ntrials = length(data.tsteps)
    out = zeros(T, ntrials)
    for site in state.sites
        y_m = _member_y(data.y, site.key)
        p_m = warp_channels(site.om.warp)
        for n in 1:ntrials
            yn = y_m[n]
            acc = zero(T)
            @inbounds for t in axes(yn, 2), j in 1:p_m
                acc += warp_forward(site.om.warp, j, yn[j, t])[2]
            end
            out[n] += acc
        end
    end
    return out
end

# ============================================================================
# Sampling
# ============================================================================

"""
    _sample_continuous_given_discrete!(rng, x, y, z, state_params, obs_params,
                                       ::SplineGaussianObservationModel, ux, uy)

Draw a warped `SLDS` trial given its regime path. Identical to the Gaussian
sampler except that each emission draw is made in embedding space and pushed
back out through `g⁻¹`. The warp is shared across regimes, so which regime is
active changes `(C, d, R)` but not the map.
"""
function _sample_continuous_given_discrete!(
    rng,
    x_trial,
    y_trial,
    z_trial,
    state_params,
    obs_params,
    ::SplineGaussianObservationModel,
    ux_trial::AbstractMatrix,
    uy_trial,
)
    tsteps = length(z_trial)

    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    @views _draw_spline_obs!(
        y_trial[:, 1],
        rng,
        obs_params[k1],
        obs_params[k1].warp,
        x_trial[:, 1],
        uy_trial[:, 1],
    )

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
        @views _draw_spline_obs!(
            y_trial[:, t],
            rng,
            obs_params[k],
            obs_params[k].warp,
            x_trial[:, t],
            uy_trial[:, t],
        )
    end
    return nothing
end
