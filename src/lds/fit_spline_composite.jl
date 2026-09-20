#=============================================================================
Spline-Gaussian emissions inside a `CompositeObservationModel`.

A composite's members are conditionally independent given the latent path, so
every emission term is a sum over members and a warped member contributes its
own change-of-variables term:

```math
log p(y | x) = Σ_m log p(y_m | x),
    log p(y_m | x) = N(g_m(y_m); C_m x + d_m + D_m v_m, R_m) · ∏_j g'_{m,j}(y_{m,j})
```

The warps do not interact: member `m`'s warp objective sees only `y_m`, `R_m`
and `C_m x̂ + d_m + D_m v_m`, so CM-step 2 is one independent L-BFGS solve per
warped member. The state parameters are shared, as always.

Mechanically this is the same shadow construction as the single-emission driver,
one level up: the composite is rebuilt with each warped member replaced by its
Gaussian shadow (sharing `C`, `R`, `d`, `D` by reference), and the `NamedTuple`
of observations is rebuilt with each warped member's entry replaced by its
embedding `z_m`. Every existing composite routine — `_obs_views`,
`_member_datas`, the per-member sub-workspaces, `Q_obs!`, `mstep!` — then applies
unchanged.

A composite's `QUAD` type parameter cannot encode "contains a warp": it is the
`AND` of a different property. The public entry points in `fit_LDS.jl` therefore
branch on `_has_warped_member` at run time.
=============================================================================#

#=
Member-wise shadow: a warped member becomes its Gaussian shadow, everything else
passes through untouched (and by reference).
=#
_member_shadow(m::SplineGaussianObservationModel) = _gaussian_shadow(m)
_member_shadow(m::AbstractObservationModel) = m

"""
    _gaussian_shadow(c::CompositeObservationModel) -> CompositeObservationModel

The composite with every warped member replaced by its Gaussian shadow. Members
that carry no warp are passed through as the same objects, so an M-step run on
the shadow updates the original composite's members directly.
"""
function _gaussian_shadow(c::CompositeObservationModel)
    models = _models(c)
    return CompositeObservationModel(
        NamedTuple{keys(models)}(map(_member_shadow, values(models)))
    )
end

#=
A warped member declares three `fit_bool` slots and its shadow only two, so the
shadow's vector has to be rebuilt rather than sliced: keep the state block, then
each member's leading `_obs_nblocks(shadow)` flags. The warp flag is not dropped
— it is read separately when building that member's site.
=#
function _shadow_fit_bool(lds::LinearDynamicalSystem)
    c = lds.obs_model
    models = _models(c)
    fb = Bool[lds.fit_bool[i] for i in 1:4]
    for key in _obs_keys(c)
        r = _obs_fit_range(c, key)
        n = _obs_nblocks(_member_shadow(models[key]))
        for i in 0:(n - 1)
            push!(fb, lds.fit_bool[first(r) + i])
        end
    end
    return fb
end

function _gaussian_shadow(
    lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel{T},O<:CompositeObservationModel{T}}
    om = _gaussian_shadow(lds.obs_model)
    return LinearDynamicalSystem{T,S,typeof(om)}(
        lds.state_model,
        om,
        lds.latent_dim,
        lds.obs_dim,
        lds.ux_dim,
        lds.uy_dim,
        _shadow_fit_bool(lds),
    )
end

"""
    _SplineSite{T,YV}

One warped member of a model, with everything its embedding and its warp
CM-step need: the member's key, its spline emission, the embedding buffers
`z_m`, the smoothed emission means `μ̂_m`, the L-BFGS context and parameter
vectors, the member's running log-Jacobian, and whether its warp is being fitted.

A standalone spline emission is the one-site case with `key = nothing`; the
single-emission driver keeps its own (equivalent) buffers rather than routing
through this, but the M-step helpers are shared.
"""
struct _SplineSite{T<:Real,YV<:AbstractVector{<:AbstractMatrix{T}}}
    key::Symbol
    om::SplineGaussianObservationModel{T}
    z::Vector{Matrix{T}}
    mu::Vector{Matrix{T}}
    ctx::_SplineMStepCtx{T,YV}
    θ0::Vector{T}
    θ1::Vector{T}
    logjac::Base.RefValue{T}
    fit_warp::Bool
end

"""
    _spline_sites(lds, data) -> Vector{_SplineSite}

One site per warped member of a composite emission, in the composite's key
order. Allocates each member's embedding and emission-mean buffers at that
member's own channel count.
"""
function _spline_sites(
    lds::LinearDynamicalSystem{T,S,O}, data::Data{T}
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    _reject_spline_lqr(lds)
    c = lds.obs_model
    models = _models(c)
    sites = Any[]
    for key in _obs_keys(c)
        m = models[key]
        m isa SplineGaussianObservationModel || continue
        y_m = data.y[key]
        p_m = _obs_dim(m)
        z = [Matrix{T}(undef, p_m, size(yt, 2)) for yt in y_m]
        mu = [Matrix{T}(undef, p_m, size(yt, 2)) for yt in y_m]
        ctx = _SplineMStepCtx(m, y_m, mu)
        nθ = warp_nparams(m.warp)
        fit_warp = lds.fit_bool[_obs_fit_range(c, key)[3]]
        push!(
            sites,
            _SplineSite{T,typeof(y_m)}(
                key, m, z, mu, ctx, zeros(T, nθ), zeros(T, nθ), Ref(zero(T)), fit_warp
            ),
        )
    end
    return identity.(sites)
end

"""
    _spline_shadow_data(data, sites) -> Data

`data` with every warped member's observations replaced by that member's
embedding buffer. Inputs, trial lengths and unwarped members are shared by
reference, so refreshing a site's `z` refreshes what the shadow `Data` exposes.
"""
function _spline_shadow_data(data::Data{T}, sites::AbstractVector) where {T<:Real}
    y = data.y
    zmap = Dict{Symbol,Any}(site.key => site.z for site in sites)
    znt = NamedTuple{keys(y)}(map(k -> get(zmap, k, y[k]), keys(y)))
    return Data(znt, data.ux, data.uy, data.tsteps)
end

"""
    _spline_embed_sites!(sites, data) -> T

Refresh every site's embedding from the raw observations and return the total
log-Jacobian summed over warped members.
"""
function _spline_embed_sites!(sites::AbstractVector, data::Data{T}) where {T<:Real}
    total = zero(T)
    for site in sites
        y_m = data.y[site.key]
        acc = zero(T)
        for n in eachindex(y_m)
            acc += warp_apply!(site.z[n], site.om.warp, y_m[n])
        end
        site.logjac[] = acc
        total += acc
    end
    return total
end

"""
    _spline_sites_logprior(T, sites) -> T

Summed warp log-priors (the per-member ridge), the composite counterpart of
[`_spline_logprior`](@ref).
"""
function _spline_sites_logprior(::Type{T}, sites::AbstractVector) where {T<:Real}
    total = zero(T)
    for site in sites
        total += _spline_logprior(site.om)
    end
    return total
end

"""
    _spline_composite_mstep!(lds, glds, suf, sws_pool, tfs, sdata, sites; spline_iters)

M-step for a composite containing warped members, shared by the quadratic and
Laplace drivers.

CM-step 1 is the shared state update plus each member's own emission update,
reached through that member's view on the shadow composite: a warped member gets
`update_C_d!` and [`_spline_update_R!`](@ref) (so `R_structure` and `R_floor` are
honoured), and every other member keeps whatever
[`_member_obs_mstep!`](@ref) already does for it — the conjugate Gaussian update,
or the Poisson emission's row-wise Newton solve.

CM-step 2 is one warp solve per warped member. They are independent: member `m`'s
objective sees only `y_m`, `R_m` and `C_m x̂ + d_m + D_m v_m`.
"""
function _spline_composite_mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    glds::LinearDynamicalSystem{T,S,GO},
    suf::NamedTuple,
    sws_pool::Vector{SmoothWorkspace{T}},
    tfs::TrialFilterSmooth{T},
    sdata::Data{T},
    sites::AbstractVector;
    spline_iters::Int=25,
) where {
    T<:Real,
    S<:AbstractGaussianStateModel{T},
    O<:CompositeObservationModel{T},
    GO<:CompositeObservationModel{T},
}
    sws = sws_pool[1]
    state = _state_suf(suf)
    update_initial_state_mean!(glds, state)
    update_initial_state_covariance!(glds, state, sws)
    update_A_b!(glds, state, sws)
    update_Q!(glds, state, sws)

    views = _obs_views(glds)
    datas = _member_datas(sdata)
    pools = _member_pools(sws_pool, glds)
    models = _models(lds.obs_model)

    # CM-step 1, per member.
    for (i, key) in enumerate(_obs_keys(glds.obs_model))
        v = views[key]
        m = models[key]
        if m isa SplineGaussianObservationModel
            update_C_d!(v, suf[key], pools[i][1])
            _spline_update_R!(m, v, suf[key], pools[i][1])
        else
            _member_obs_mstep!(v, suf[key], tfs, datas[key], pools[i])
        end
    end

    # CM-step 2, per warped member.
    for site in sites
        _spline_emission_means!(site.mu, views[site.key].obs_model, tfs, sdata.uy[site.key])
        _spline_warp_mstep!(
            site.om, site.ctx, site.θ0, site.θ1; iters=spline_iters, fit_warp=site.fit_warp
        )
    end
    return nothing
end

"""
    _fit_spline_composite!(lds, data; kwargs...)

ECM loop for a composite emission with at least one warped member. Same shape as
[`_fit_spline!`](@ref), with per-member embeddings and one warp CM-step per
warped member.
"""
function _fit_spline_composite!(
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T};
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
    spline_iters::Int=25,
    monitor=nothing,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T,true}}
    glds = _gaussian_shadow(lds)
    sites = _spline_sites(lds, data)
    sdata = _spline_shadow_data(data, sites)

    tsteps_per_trial = data.tsteps
    T_max = maximum(tsteps_per_trial)
    ntrials_total = length(tsteps_per_trial)
    elbos = Vector{T}(undef, max_iter)

    cov_alias = ntrials_total > 1 && all(t -> t == tsteps_per_trial[1], tsteps_per_trial)
    tfs = initialize_FilterSmooth(
        glds, tsteps_per_trial; cov_alias=cov_alias
    )::TrialFilterSmooth{T}

    pool_size = Threads.maxthreadid()
    sws_pool = [
        SmoothWorkspace(
            T,
            lds.latent_dim,
            _ws_obs_dim(glds),
            T_max;
            ux_dim=lds.ux_dim,
            uy_dim=_ws_uy_dim(glds),
        ) for _ in 1:pool_size
    ]

    # As in `_fit_spline!`: an inverse-LQR state is refused before we get here.
    suf = _initialize_td_sufficient_statistics(T, glds, tsteps_per_trial)::NamedTuple

    prog = if progress
        Progress(
            max_iter;
            desc="Fitting composite spline LDS via ECM...",
            barlen=50,
            showspeed=true,
        )
    else
        nothing
    end

    for iter in 1:max_iter
        logjac = _spline_embed_sites!(sites, data)
        _td_init_const_blocks!(sws_pool[1], glds, sdata)

        estep!(glds, suf, tfs, sdata, sws_pool)

        total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths; init=zero(T))
        elbos[iter] =
            elbo!(glds, suf, sws_pool[1], total_entropy) +
            logjac +
            _spline_sites_logprior(T, sites)

        _holdout_due(monitor, iter) && _holdout_record!(monitor, lds, iter)
        if _holdout_stop(monitor)
            prog !== nothing && finish!(prog)
            resize!(elbos, iter)
            return _fit_result(monitor, elbos, lds)
        end

        converged = iter > 1 && abs(elbos[iter] - elbos[iter - 1]) < tol

        _spline_composite_mstep!(
            lds, glds, suf, sws_pool, tfs, sdata, sites; spline_iters=spline_iters
        )

        prog !== nothing && next!(prog)

        if converged
            prog !== nothing && finish!(prog)
            resize!(elbos, iter)
            return _fit_result(monitor, elbos, lds)
        end
    end

    prog !== nothing && finish!(prog)
    return _fit_result(monitor, elbos, lds)
end

# ============================================================================
# Public entry points, reached from `fit_LDS.jl`'s run-time branch
# ============================================================================

function _spline_composite_smooth(
    lds::LinearDynamicalSystem{T,S,O}, y, ux, uy
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    _reject_spline_grouping(lds)
    data = Data(lds, y; ux=ux, uy=uy)
    glds = _gaussian_shadow(lds)
    sites = _spline_sites(lds, data)
    sdata = _spline_shadow_data(data, sites)
    _spline_embed_sites!(sites, data)
    tfs = _smooth_data(glds, sdata)
    return _collect_smooth_output(tfs, y)
end

function _spline_composite_elbo(
    lds::LinearDynamicalSystem{T,S,O}, y, ux, uy
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    _reject_spline_grouping(lds)
    data = Data(lds, y; ux=ux, uy=uy)
    glds = _gaussian_shadow(lds)
    sites = _spline_sites(lds, data)
    sdata = _spline_shadow_data(data, sites)
    logjac = _spline_embed_sites!(sites, data)

    tfs = initialize_FilterSmooth(glds, data.tsteps)::TrialFilterSmooth{T}
    npool = min(Threads.maxthreadid(), length(data.tsteps))
    sws_pool = [
        SmoothWorkspace(
            T,
            lds.latent_dim,
            _ws_obs_dim(glds),
            maximum(data.tsteps);
            ux_dim=lds.ux_dim,
            uy_dim=_ws_uy_dim(glds),
        ) for _ in 1:npool
    ]
    # As in `_fit_spline!`: an inverse-LQR state is refused before we get here.
    suf = _initialize_td_sufficient_statistics(T, glds, data.tsteps)::NamedTuple
    _td_init_const_blocks!(sws_pool[1], glds, sdata)
    estep!(glds, suf, tfs, sdata, sws_pool)
    total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths; init=zero(T))
    return elbo!(glds, suf, sws_pool[1], total_entropy) +
           logjac +
           _spline_sites_logprior(T, sites)
end

function _spline_composite_loglikelihood(
    lds::LinearDynamicalSystem{T,S,O}, y, ux, uy
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    _reject_spline_grouping(lds)
    data = Data(lds, y; ux=ux, uy=uy)
    glds = _gaussian_shadow(lds)
    sites = _spline_sites(lds, data)
    sdata = _spline_shadow_data(data, sites)
    logjac = _spline_embed_sites!(sites, data)
    return loglikelihood(glds, sdata.y; ux=sdata.ux, uy=sdata.uy) + logjac
end

# ============================================================================
# The Laplace path: a warped member alongside a Poisson one
#
# A composite mixing a spline-Gaussian member with a Poisson member is
# non-quadratic — the Poisson curvature depends on the latent path — so it needs
# the iterative Laplace smoother of `fit_PLDS.jl` rather than the single-step
# Newton one. Nothing about the warp changes: given `g`, the warped member is
# still an ordinary Gaussian readout of the same latent state, so the embedding
# is refreshed exactly as on the quadratic path and the warp objective is the
# same one. Only the E-step and the Q-term routes differ, and both are reached
# through the existing non-quadratic overloads on the shadow system.
# ============================================================================

"""
    _fit_spline_laplace!(lds, data; kwargs...)

ECM loop for a composite with at least one warped member and at least one
non-quadratic member. Mirrors `_fit_laplace!` with the per-member embedding
refresh at the top of each E-step and the warp CM-steps at the foot of each
M-step.
"""
function _fit_spline_laplace!(
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T};
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
    newton_max_iter::Int=20,
    newton_tol::Float64=1e-6,
    spline_iters::Int=25,
    monitor=nothing,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T,false}}
    glds = _gaussian_shadow(lds)
    sites = _spline_sites(lds, data)
    sdata = _spline_shadow_data(data, sites)

    T_max = maximum(data.tsteps)
    tfs = initialize_FilterSmooth(glds, data.tsteps)::TrialFilterSmooth{T}

    npool = Threads.maxthreadid()
    sws_pool = [
        SmoothWorkspace(
            T,
            lds.latent_dim,
            _ws_obs_dim(glds),
            T_max;
            ux_dim=lds.ux_dim,
            uy_dim=_ws_uy_dim(glds),
        ) for _ in 1:npool
    ]

    # As in `_fit_spline!`: an inverse-LQR state is refused before we get here.
    suf = _initialize_td_sufficient_statistics(T, glds, data.tsteps)::NamedTuple
    elbos = Vector{T}(undef, max_iter)

    prog = if progress
        Progress(
            max_iter;
            desc="Fitting composite spline LDS via Laplace ECM...",
            barlen=50,
            showspeed=true,
        )
    else
        nothing
    end

    for iter in 1:max_iter
        logjac = _spline_embed_sites!(sites, data)
        _td_init_const_blocks!(sws_pool[1], glds, sdata)

        estep!(glds, suf, tfs, sdata, sws_pool; max_iter=newton_max_iter, tol=T(newton_tol))

        elbos[iter] =
            elbo!(glds, suf, tfs, sdata, sws_pool) +
            logjac +
            _spline_sites_logprior(T, sites)

        _holdout_due(monitor, iter) && _holdout_record!(monitor, lds, iter)
        if _holdout_stop(monitor)
            prog !== nothing && finish!(prog)
            resize!(elbos, iter)
            return _fit_result(monitor, elbos, lds)
        end

        converged = iter > 1 && abs(elbos[iter] - elbos[iter - 1]) < tol

        _spline_composite_mstep!(
            lds, glds, suf, sws_pool, tfs, sdata, sites; spline_iters=spline_iters
        )

        prog !== nothing && next!(prog)

        if converged
            prog !== nothing && finish!(prog)
            resize!(elbos, iter)
            return _fit_result(monitor, elbos, lds)
        end
    end

    prog !== nothing && finish!(prog)
    return _fit_result(monitor, elbos, lds)
end

function _spline_composite_smooth_laplace(
    lds::LinearDynamicalSystem{T,S,O}, y, ux, uy, newton_max_iter::Int, newton_tol::Float64
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T,false}}
    _reject_spline_grouping(lds)
    data = Data(lds, y; ux=ux, uy=uy)
    glds = _gaussian_shadow(lds)
    sites = _spline_sites(lds, data)
    sdata = _spline_shadow_data(data, sites)
    _spline_embed_sites!(sites, data)

    tfs = initialize_FilterSmooth(glds, data.tsteps)::TrialFilterSmooth{T}
    npool = min(Threads.maxthreadid(), length(data.tsteps))
    sws_pool = [
        SmoothWorkspace(
            T,
            lds.latent_dim,
            _ws_obs_dim(glds),
            maximum(data.tsteps);
            ux_dim=lds.ux_dim,
            uy_dim=_ws_uy_dim(glds),
        ) for _ in 1:npool
    ]
    smooth!(glds, tfs, sdata, sws_pool; max_iter=newton_max_iter, tol=T(newton_tol))
    return _collect_smooth_output(tfs, y)
end

function _spline_composite_elbo_laplace(
    lds::LinearDynamicalSystem{T,S,O}, y, ux, uy, newton_max_iter::Int, newton_tol::Float64
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T,false}}
    _reject_spline_grouping(lds)
    data = Data(lds, y; ux=ux, uy=uy)
    glds = _gaussian_shadow(lds)
    sites = _spline_sites(lds, data)
    sdata = _spline_shadow_data(data, sites)
    logjac = _spline_embed_sites!(sites, data)

    tfs = initialize_FilterSmooth(glds, data.tsteps)::TrialFilterSmooth{T}
    npool = min(Threads.maxthreadid(), length(data.tsteps))
    sws_pool = [
        SmoothWorkspace(
            T,
            lds.latent_dim,
            _ws_obs_dim(glds),
            maximum(data.tsteps);
            ux_dim=lds.ux_dim,
            uy_dim=_ws_uy_dim(glds),
        ) for _ in 1:npool
    ]
    # As in `_fit_spline!`: an inverse-LQR state is refused before we get here.
    suf = _initialize_td_sufficient_statistics(T, glds, data.tsteps)::NamedTuple
    _td_init_const_blocks!(sws_pool[1], glds, sdata)
    estep!(glds, suf, tfs, sdata, sws_pool; max_iter=newton_max_iter, tol=T(newton_tol))
    return elbo!(glds, suf, tfs, sdata, sws_pool) +
           logjac +
           _spline_sites_logprior(T, sites)
end

"""
    _spline_trial_logjac(T, sites, data, ntrials) -> Vector{T}

Each trial's own change-of-variables term, summed over warped members.

The embedding pass keeps only the dataset total, but a per-trial ELBO has to be
exact for `sum(trial_elbos) + log p(θ) == elbo` to hold, so this re-walks the raw
observations trial by trial.
"""
function _spline_trial_logjac(
    ::Type{T}, sites::AbstractVector, data::Data{T}, ntrials::Int
) where {T<:Real}
    out = zeros(T, ntrials)
    for site in sites
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

"""
    _spline_composite_trial_elbos(lds, y, ux, uy; kwargs...) -> Vector

`trial_elbos` for a composite with a warped member: the shadow model's per-trial
contributions on the embedded scale, plus each trial's own log-Jacobian. The
parameter log-prior (IW/MN terms and the warp ridge) belongs to no trial and is
excluded, exactly as for every other emission.
"""
function _spline_composite_trial_elbos(
    lds::LinearDynamicalSystem{T,S,O}, y, ux, uy; kwargs...
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    _reject_spline_grouping(lds)
    data = Data(lds, y; ux=ux, uy=uy)
    glds = _gaussian_shadow(lds)
    sites = _spline_sites(lds, data)
    sdata = _spline_shadow_data(data, sites)
    _spline_embed_sites!(sites, data)
    per_trial = trial_elbos(glds, sdata.y; ux=sdata.ux, uy=sdata.uy, kwargs...)
    return per_trial .+ _spline_trial_logjac(T, sites, data, length(data.tsteps))
end
