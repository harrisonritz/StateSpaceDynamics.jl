#=============================================================================
Drivers for a spline-Gaussian emission: ECM fitting, smoothing, ELBO, marginal
log-likelihood and sampling.

Every routine here is the same two moves. Embed the observations with the
current warp (`z = g(y)`, accumulating `Σ log g'`), then hand the *shadow*
Gaussian system — which shares `C`, `R`, `d`, `D` by reference — the shadow
`Data` holding `z` and let the existing, tuned Gaussian pipeline do the work.
The change of variables enters as one additive term:

```math
log p(y_{1:T}) = log p_z(g(y_{1:T})) + Σ_{t,j} log g_j'(y_{tj})
```

so a log-density on the `y` scale is the Gaussian one plus `Σ log g'`, and the
smoother output is unchanged (the Jacobian does not involve `x`).

Fitting differs from `_fit_tridiag!` in exactly three places, all forced by `z`
moving between iterations rather than being fixed data:

  1. `_td_init_const_blocks!` is re-run every E-step instead of once at entry;
  2. the batched smoother's staged copy of the observations is invalidated every
     E-step;
  3. the M-step gains a second conditional-maximization step for the warp.

The reported ELBO includes `Σ log g'` and the spline's ridge log-prior, without
which the trace would not be comparable across iterations — `z` itself changes,
so the Gaussian part alone is not a likelihood of anything fixed.
=============================================================================#

#=
Shared setup for every entry point: a `Data` over the raw `y`, the shadow
Gaussian system, and the embedding buffers already filled at the current warp.
=#
function _spline_setup(
    lds::LinearDynamicalSystem{T,S,O}, data::Data{T}
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:SplineGaussianObservationModel{T}}
    _reject_spline_lqr(lds)
    glds = _gaussian_shadow(lds)
    emb = SplineEmbedding(data, lds.obs_dim)
    _embed!(emb, lds.obs_model.warp, data)
    return glds, emb
end

#=
An inverse-LQR state model constrains its transition to the symplectic form a
control problem implies, and is fitted by its own structural M-step
(`lqr_mstep.jl`) rather than by the generic `update_A_b!` / `update_Q!` the
spline driver calls. Running the generic updates on it would silently discard
that structure, so refuse instead.
=#
function _reject_spline_lqr(lds::LinearDynamicalSystem)
    (lds.state_model isa LQRStateModel && _has_warped_member(lds.obs_model)) && throw(
        ArgumentError(
            "a SplineGaussianObservationModel is not supported with an " *
            "LQRStateModel: the inverse-LQR state M-step is structural, and the " *
            "spline driver runs the generic Gaussian state updates. Use a " *
            "GaussianStateModel, or a GaussianObservationModel.",
        ),
    )
    return nothing
end

#=
Grouped (`depends_on`) fits route through `parameter_groups.jl`, which builds one
parameter version per cell of trials. A spline emission's warp is part of that
per-cell parameter set, so the cell machinery has to know how to split and
rebuild it; until it does, refuse rather than silently fit one shared warp.
=#
function _reject_spline_grouping(lds::LinearDynamicalSystem)
    _has_parameter_dependence(lds) && throw(
        ArgumentError(
            "`depends_on` parameter grouping is not supported for a " *
            "SplineGaussianObservationModel: each group would need its own warp, " *
            "which the grouped M-step does not yet build. Fit each group " *
            "separately, or use a GaussianObservationModel.",
        ),
    )
    return nothing
end

"""
    smooth(lds, y; ux=nothing, uy=nothing)

Posterior mean and covariance of the latent trajectory under a spline-Gaussian
emission.

The warp is applied to `y` first; the smoother then runs on the embedded
observations, where the model is exactly linear-Gaussian. The returned posterior
is therefore exact, not a Laplace approximation — the nonlinearity lives entirely
in the (invertible, `x`-independent) change of variables.

Returns and accepts exactly what the Gaussian [`smooth`](@ref) does.
"""
function smooth(
    lds::LinearDynamicalSystem{T,S,O},
    y::Observations{T};
    ux=nothing,
    uy=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:SplineGaussianObservationModel{T}}
    depends_on === nothing ||
        throw(ArgumentError("`depends_on` is not supported for a spline emission"))
    _reject_spline_grouping(lds)
    data = Data(lds, y; ux=ux, uy=uy)
    glds, emb = _spline_setup(lds, data)
    tfs = _smooth_data(glds, emb.data)
    return _collect_smooth_output(tfs, y)
end

"""
    elbo(lds, y; ux=nothing, uy=nothing)

Evidence lower bound of a spline-Gaussian `LinearDynamicalSystem` at its current
parameters, on the **observation** scale:

```math
F = E_q[log p(x)] + E_q[log p(z | x)] + Σ_{t,j} log g_j'(y_{tj}) + H[q] + log p(θ)
```

The smoother is exact here, so with no priors this equals the marginal
[`loglikelihood`](@ref). `log p(θ)` collects the IW/MN terms and the spline's
ridge, matching the penalized objective the M-step maximizes.
"""
function elbo(
    lds::LinearDynamicalSystem{T,S,O},
    y::Observations{T};
    ux=nothing,
    uy=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:SplineGaussianObservationModel{T}}
    depends_on === nothing ||
        throw(ArgumentError("`depends_on` is not supported for a spline emission"))
    _reject_spline_grouping(lds)
    data = Data(lds, y; ux=ux, uy=uy)
    glds, emb = _spline_setup(lds, data)

    tfs = initialize_FilterSmooth(glds, data.tsteps)::TrialFilterSmooth{T}
    npool = min(Threads.maxthreadid(), length(data.tsteps))
    sws_pool = [
        SmoothWorkspace(
            T,
            lds.latent_dim,
            lds.obs_dim,
            maximum(data.tsteps);
            ux_dim=lds.ux_dim,
            uy_dim=lds.uy_dim,
        ) for _ in 1:npool
    ]
    suf = _initialize_td_sufficient_statistics(T, glds, data.tsteps)
    _td_init_const_blocks!(sws_pool[1], glds, emb.data)

    estep!(glds, suf, tfs, emb.data, sws_pool)
    total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths; init=zero(T))
    return elbo!(glds, suf, sws_pool[1], total_entropy) +
           emb.logjac[] +
           _spline_logprior(lds.obs_model)
end

"""
    loglikelihood(lds, y; ux=nothing, uy=nothing)

Marginal (observed-data) log-likelihood of a spline-Gaussian
`LinearDynamicalSystem`: the Kalman-filter predictive density of the embedded
observations, plus the change-of-variables term.

```math
log p(y_{1:T}) = Σ_t log p(z_t | z_{1:t-1}) + Σ_{t,j} log g_j'(y_{tj}),  z = g(y)
```

Unlike [`elbo`](@ref) this excludes any parameter log-prior, so the two agree
exactly for a model with no priors and a zero `spline_ridge`.
"""
function StatsAPI.loglikelihood(
    lds::LinearDynamicalSystem{T,SM,OM},
    y::Observations{T};
    ux=nothing,
    uy=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,SM<:GaussianStateModel{T},OM<:SplineGaussianObservationModel{T}}
    depends_on === nothing ||
        throw(ArgumentError("`depends_on` is not supported for a spline emission"))
    _reject_spline_grouping(lds)
    data = Data(lds, y; ux=ux, uy=uy)
    glds, emb = _spline_setup(lds, data)
    return loglikelihood(glds, emb.z; ux=data.ux, uy=data.uy) + emb.logjac[]
end

"""
    fit!(lds, y; max_iter=100, tol=1e-6, spline_iters=25, ...)

Fit a spline-Gaussian Linear Dynamical System by Expectation Conditional
Maximization.

Each iteration is

1. **E-step.** Embed `z = g(y)` with the current warp and smooth. The posterior
   is exact — given `g`, the model in `z` is linear-Gaussian.
2. **CM-step 1.** Closed-form updates of `x0`, `P0`, `[A b B]`, `Q`, `[C d D]`
   and `R` from the aggregated sufficient statistics of `z`, with the warp held
   fixed (the log-Jacobian is then a constant and drops out).
3. **CM-step 2.** L-BFGS on the spline parameters with everything else held
   fixed. That objective is exactly a normalizing-flow MLE against a Gaussian
   base centered on the smoothed prediction `C x̂_t + d + D uy_t`; only the
   smoothed *mean* enters, because the covariance contributes a term constant in
   the spline parameters.

Both conditional maximizations increase the same `Q(θ, q)`, and the warp step is
accepted only when it strictly improves, so the observed-data log-likelihood is
non-decreasing.

# Arguments
- `lds`: model to fit in place.
- `y`: observations — `(obs_dim, T)` matrix, `(obs_dim, T, ntrials)` array, or a
  `Vector{<:AbstractMatrix}` of per-trial matrices (ragged lengths allowed).

# Keywords
- `max_iter::Int=100`, `tol::Float64=1e-6`, `progress::Bool=true`: as for the
  Gaussian [`fit!`](@ref).
- `spline_iters::Int=25`: L-BFGS iterations per warp CM-step. A partial
  maximization is still a valid generalized-EM step, so a small budget is safe;
  raise it if the warp is visibly still moving at convergence. `0` freezes the
  warp, as does `fit_bool = (spline = false,)`.
- `ux` / `uy`: dynamics / observation inputs, in the same shape family as `y`.
- `y_test`, `ux_test`, `uy_test`, `test_every`, `early_stopping`, `patience`,
  `min_delta`, `restore_best`, `test_kwargs`: held-out scoring and early
  stopping, exactly as for the Gaussian [`fit!`](@ref). Held-out ELBOs are on
  the observation scale, so they are comparable with a plain Gaussian LDS fit to
  the same data.

Returns a `Vector{T}` of ELBO values — or a [`FitTrace{T}`](@ref) when `y_test`
is given, which behaves as that same vector.

# Note
`depends_on` parameter grouping is not supported for this emission; see
[`SplineGaussianObservationModel`](@ref).
"""
function fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::Observations{T};
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
    spline_iters::Int=25,
    ux=nothing,
    uy=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
    y_test=nothing,
    ux_test=nothing,
    uy_test=nothing,
    depends_on_test::Union{Nothing,NamedTuple}=nothing,
    test_every::Int=1,
    early_stopping::Bool=false,
    patience::Int=1,
    min_delta::Real=0.0,
    restore_best::Bool=true,
    test_kwargs::NamedTuple=NamedTuple(),
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:SplineGaussianObservationModel{T}}
    depends_on === nothing ||
        throw(ArgumentError("`depends_on` is not supported for a spline emission"))
    depends_on_test === nothing ||
        throw(ArgumentError("`depends_on_test` is not supported for a spline emission"))
    _reject_spline_grouping(lds)

    data = Data(lds, y; ux=ux, uy=uy)
    monitor = _holdout_monitor(
        T,
        y_test;
        ux_test=ux_test,
        uy_test=uy_test,
        test_every=test_every,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        restore_best=restore_best,
        test_kwargs=test_kwargs,
    )
    return _fit_spline!(
        lds,
        data;
        max_iter=max_iter,
        tol=tol,
        progress=progress,
        spline_iters=spline_iters,
        monitor=monitor,
    )
end

"""
    _fit_spline!(lds, data; max_iter, tol, progress, spline_iters, monitor)

ECM loop for a spline-Gaussian emission. Structurally `_fit_tridiag!` with the
embedding refresh at the top of each E-step and the warp CM-step at the foot of
each M-step; see the file header for why those three differences are forced.
"""
function _fit_spline!(
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T};
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
    spline_iters::Int=25,
    monitor=nothing,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:SplineGaussianObservationModel{T}}
    glds, emb = _spline_setup(lds, data)
    tsteps_per_trial = data.tsteps
    T_max = maximum(tsteps_per_trial)
    ntrials_total = length(tsteps_per_trial)
    elbos = Vector{T}(undef, max_iter)

    cov_alias = ntrials_total > 1 && all(t -> t == tsteps_per_trial[1], tsteps_per_trial)
    tfs = initialize_FilterSmooth(
        glds, tsteps_per_trial; cov_alias=cov_alias
    )::TrialFilterSmooth{T}

    pool_size = Threads.maxthreadid()
    sws_pool = Vector{SmoothWorkspace{T}}(undef, pool_size)
    sws_pool[1] = SmoothWorkspace(
        T,
        lds.latent_dim,
        lds.obs_dim,
        T_max;
        ux_dim=lds.ux_dim,
        uy_dim=lds.uy_dim,
        ntrials=_batched_ntrials(glds, ntrials_total),
    )
    for i in 2:pool_size
        sws_pool[i] = SmoothWorkspace(
            T, lds.latent_dim, lds.obs_dim, T_max; ux_dim=lds.ux_dim, uy_dim=lds.uy_dim
        )
    end

    #=
    An inverse-LQR state model would return its own statistics type here,
    which the closed-form emission updates cannot read. `_reject_spline_lqr`
    has already refused that combination; this pins it for the reader and
    for inference alike.
    =#
    suf = _initialize_td_sufficient_statistics(
        T, glds, tsteps_per_trial
    )::SufficientStatistics{T}

    ctx = _SplineMStepCtx(lds.obs_model, data.y, emb.mu)
    nθ = warp_nparams(lds.obs_model.warp)
    θ0 = zeros(T, nθ)
    θ1 = zeros(T, nθ)

    prog = if progress
        Progress(max_iter; desc="Fitting spline LDS via ECM...", barlen=50, showspeed=true)
    else
        nothing
    end

    for iter in 1:max_iter
        #=
        E-step. The warp moved in the previous M-step, so `z` — and with it every
        data-only constant the aggregator caches, and the batched smoother's
        staged copy — has to be rebuilt before smoothing.
        =#
        _embed!(emb, lds.obs_model.warp, data)
        _td_init_const_blocks!(sws_pool[1], glds, emb.data)
        bat = sws_pool[1].batched
        bat === nothing || (bat.data_valid[] = false)

        estep!(glds, suf, tfs, emb.data, sws_pool)

        total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths; init=zero(T))
        elbos[iter] =
            elbo!(glds, suf, sws_pool[1], total_entropy) +
            emb.logjac[] +
            _spline_logprior(lds.obs_model)

        _holdout_due(monitor, iter) && _holdout_record!(monitor, lds, iter)
        if _holdout_stop(monitor)
            prog !== nothing && finish!(prog)
            resize!(elbos, iter)
            return _fit_result(monitor, elbos, lds)
        end

        converged = iter > 1 && abs(elbos[iter] - elbos[iter - 1]) < tol

        # CM-step 1: state parameters and the linear half of the emission.
        _spline_linear_mstep!(lds, glds, suf, sws_pool[1])

        #=
        CM-step 2: the warp, against the emission means implied by the *updated*
        `[C d D]` and whitened by the *updated* `R` — a genuine conditional
        maximization of the same Q, which is what keeps ECM monotone.
        =#
        _spline_emission_means!(emb.mu, glds.obs_model, tfs, emb.data.uy)
        _spline_warp_mstep!(
            lds.obs_model, ctx, θ0, θ1; iters=spline_iters, fit_warp=lds.fit_bool[7]
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
# Sampling
# ============================================================================

#=
Draw one observation: a Gaussian draw in embedding space pushed back out through
`g⁻¹`. The inverse of a rational-quadratic spline is a quadratic root, so this
is exact and costs the same as the forward map.
=#
function _draw_spline_obs!(
    out::AbstractVector{T}, rng::AbstractRNG, p, warp::MonotonicWarp{T}, x_t, uy_t
) where {T<:Real}
    z = rand(rng, MvNormal(p.C * x_t + p.d + p.D * uy_t, p.R))
    @inbounds for j in eachindex(z)
        out[j] = warp_inverse(warp, j, z[j])
    end
    return out
end

function _draw_obs(rng, ::SplineGaussianObservationModel, p, x_t, uy_t)
    z = rand(rng, MvNormal(p.C * x_t + p.d + p.D * uy_t, p.R))
    return [warp_inverse(p.warp, j, z[j]) for j in eachindex(z)]
end

function _sample_trial!(
    rng,
    x_trial,
    y_trial,
    state_params,
    obs_params,
    ::SplineGaussianObservationModel,
    ux_trial::AbstractMatrix,
    uy_trial,
)
    tsteps = size(x_trial, 2)
    warp = obs_params.warp

    x_trial[:, 1] = rand(rng, MvNormal(state_params.x0, state_params.P0))
    @views _draw_spline_obs!(
        y_trial[:, 1], rng, obs_params, warp, x_trial[:, 1], uy_trial[:, 1]
    )

    for t in 2:tsteps
        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params.A * x_trial[:, t - 1] +
                state_params.b +
                state_params.B * ux_trial[:, t - 1],
                state_params.Q,
            ),
        )
        @views _draw_spline_obs!(
            y_trial[:, t], rng, obs_params, warp, x_trial[:, t], uy_trial[:, t]
        )
    end
    return nothing
end

# ============================================================================
# Per-trial ELBO split
# ============================================================================

"""
    trial_elbos(lds, y; ux=nothing, uy=nothing)

Per-trial ELBO contributions of a spline-Gaussian `LinearDynamicalSystem`.

Each entry is that trial's Gaussian contribution on the embedded scale plus its
own change-of-variables term `Σ_{t,j} log g_j'(y_{tj})`, which is a per-trial
quantity. The parameter log-prior (IW/MN terms and the spline ridge) belongs to
no trial and is excluded, so — as for every other emission —

    sum(trial_elbos(lds, y)) + log p(θ) == elbo(lds, y)
"""
function trial_elbos(
    lds::LinearDynamicalSystem{T,S,O}, y::Observations{T}; ux=nothing, uy=nothing
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:SplineGaussianObservationModel{T}}
    _reject_spline_grouping(lds)
    data = Data(lds, y; ux=ux, uy=uy)
    glds, emb = _spline_setup(lds, data)

    per_trial = trial_elbos(glds, emb.z; ux=data.ux, uy=data.uy)
    #=
    Re-walk the raw observations per trial: `_embed!` only keeps the total, and
    the split has to be exact for the contract above to hold.
    =#
    warp = lds.obs_model.warp
    p = lds.obs_dim
    for n in eachindex(data.y)
        yn = data.y[n]
        acc = zero(T)
        @inbounds for t in axes(yn, 2), j in 1:p
            acc += warp_forward(warp, j, yn[j, t])[2]
        end
        per_trial[n] += acc
    end
    return per_trial
end
