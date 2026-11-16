#=============================================================================
Fitting a constrained or kernel-prior Gaussian emission.

The loop is `_fit_tridiag!`'s: E-step, bound, M-step. The E-step is the existing
smoother (its emission precision carries the loading's uncertainty, see
`kernel_observations.jl`); `elbo!` and `mstep!` are this emission's own. What
this file adds is the start (the prior's weights and a first `q(H)`) and the
gauge step.
=============================================================================#

# ----------------------------------------------------------------------------
# The gauge
# ----------------------------------------------------------------------------

# The symmetric positive definite square root, or `nothing` where the matrix is
# not safely positive definite.
function _gauge_root(M::AbstractMatrix{T}; max_condition::Real=1e10) where {T<:Real}
    all(isfinite, M) || return nothing
    values, vectors = eigen(Symmetric(Matrix(M)))
    (values[1] > 0 && values[end] / values[1] <= max_condition) || return nothing
    return Matrix(Symmetric(vectors * Diagonal(sqrt.(values)) * vectors'))
end

"""
    _apply_latent_gauge!(lds, T)

Rewrite the model for the latents `T x`: the dynamics by similarity, their
priors pushed forward (so the state side of the bound does not change), the
loading as `H T⁻¹` with `q(H)` and the emission's precision excess carried
along. The prior on the loading is not pushed forward: it is `N(0, I ⊗ Σ)` in
whichever gauge the model is in, which is what makes the gauge matter.
"""
function _apply_latent_gauge!(lds::_StructuredLDS{R}, T::AbstractMatrix{R}) where {R<:Real}
    sm, om = lds.state_model, lds.obs_model
    k = lds.latent_dim
    Ti = inv(T)
    # The state priors and their constant, through the affine-gauge machinery.
    shell_om = GaussianObservationModel(; C=copy(om.C), R=om.R, d=om.d, D=om.D)
    shell = LinearDynamicalSystem{R,typeof(sm),typeof(shell_om)}(
        sm, shell_om, k, lds.obs_dim, lds.ux_dim, lds.uy_dim, lds.fit_bool[1:6]
    )
    plan = _gauge_prior_plan([shell], T, zeros(R, k))
    shift = _gauge_prior_shift([shell], nothing, logabsdet(T)[1])
    gauge = if sm.gauge === nothing
        AffineGauge{R}(; map=Matrix{R}(I, k, k), offset=zeros(R, k))
    else
        sm.gauge
    end

    A, b, B = (T * sm.A) * Ti, T * sm.b, T * sm.B
    Q, P0 = Matrix(Symmetric(T * sm.Q * T')), Matrix(Symmetric(T * sm.P0 * T'))
    x0, B0 = T * sm.x0, T * sm.B0
    copyto!(sm.A, A)
    copyto!(sm.b, b)
    copyto!(sm.B, B)
    copyto!(sm.Q, Q)
    copyto!(sm.P0, P0)
    copyto!(sm.x0, x0)
    copyto!(sm.B0, B0)
    for (member, field, value) in plan
        member === sm && setproperty!(sm, field, value)
    end
    sm.gauge = gauge
    gauge.logprior_offset += shift
    gauge.map = T * gauge.map
    gauge.offset = T * gauge.offset

    copyto!(om.C, om.C * Ti)
    q = om.posterior
    if q !== nothing
        om.posterior = LoadingPosterior{R}(q.mean * Ti, q.left, Ti' * q.right, q.var)
    end
    om.excess .= Symmetric(Ti' * om.excess * Ti)
    return nothing
end

# Whether the gauge is free to move: every dynamics parameter and the loading
# are being fitted.
_gauge_is_free(lds::_StructuredLDS) = all(view(lds.fit_bool, 1:5))

"""
    _bound_gauge(q, Σ) -> T or nothing

The gauge that maximises the bound over `x → T x`, `H → H T⁻¹` with `q(H)`
carried along: `T'T = E[H'Σ⁻¹H] / r`, as its symmetric square root. Only
`E[log p(H)] + H[q(H)]` depends on `T`,
`−½ tr(T⁻ᵀ E[H'Σ⁻¹H] T⁻¹) − r log|det T|`.
"""
function _bound_gauge(q::LoadingPosterior{T}, Σ::AbstractMatrix{T}) where {T<:Real}
    r = size(q.mean, 1)
    F = cholesky(Symmetric(Matrix(Σ)))
    whitened = F.L \ q.left
    weight = vec(sum(abs2, whitened; dims=1))            # diag(B'Σ⁻¹B)
    G = q.mean' * (F \ q.mean) + q.right * Diagonal(vec(weight' * q.var)) * q.right'
    return _gauge_root(G ./ r)
end

# The gauge of a unit centred second moment of the latents, `T = M^-½`.
function _moment_gauge(lds::_StructuredLDS{T}, suf::SufficientStatistics{T}) where {T}
    k = lds.latent_dim
    G = suf.obs_xx[].mat
    n = suf.obs_n
    sum_x = view(G, 1:k, k + 1)
    M = (G[1:k, 1:k] .- sum_x * sum_x' ./ n) ./ n
    root = _gauge_root(M)
    return root === nothing ? nothing : inv(root)
end

"""
    _kernel_gauge_step!(lds, suf)

The emission's gauge step at the end of an M-step (see
[`KernelGaussianObservationModel`](@ref)): `:bound` moves to the gauge that
maximises the bound, `:moment` to a unit centred second moment of the latents
under the `q(x)` behind `suf`. Skipped when part of the dynamics or the loading
is held, since the gauge then is not free.
"""
function _kernel_gauge_step!(lds::_StructuredLDS{T}, suf::SufficientStatistics{T}) where {T}
    om = lds.obs_model
    prior, q = om.prior, om.posterior
    (prior === nothing || q === nothing || om.gauge === :none) && return nothing
    _gauge_is_free(lds) || return nothing
    Tm = if om.gauge === :bound
        _bound_gauge(q, kernel_covariance(prior))
    else
        _moment_gauge(lds, suf)
    end
    Tm === nothing || _apply_latent_gauge!(lds, Tm)
    return nothing
end

# ----------------------------------------------------------------------------
# The start
# ----------------------------------------------------------------------------

"""
    _start_kernel_fit!(lds, suf, tfs, data, sws_pool; start_rounds=30)

Before the first scored iteration of a kernel-prior fit that has no `q(H)` yet:
move to the unit-moment gauge of the starting model (unless `gauge` is `:none`;
the prior's empirical centres are resolved there, where the loading's second
moment is the emission's share of the data's), fit the prior's weights and trees
to the starting loading, as a post-hoc fit of the kernels would, and compute
the first `q(H)`.
"""
function _start_kernel_fit!(
    lds::_StructuredLDS{T}, suf, tfs, data, sws_pool; start_rounds::Int=30
) where {T<:Real}
    om = lds.obs_model
    prior = om.prior
    (prior === nothing || om.posterior !== nothing) && return nothing
    k = lds.latent_dim
    estep!(lds, suf, tfs, data, sws_pool)
    if om.gauge !== :none && _gauge_is_free(lds)
        Tm = _moment_gauge(lds, suf)
        if Tm !== nothing
            _apply_latent_gauge!(lds, Tm)
            estep!(lds, suf, tfs, data, sws_pool)
        end
    end
    H = _loading_coefficients(om.basis, om.C)
    S = H * H' ./ k
    if lds.fit_bool[7]
        prior.initialized ||
            fit_kernel_weights!(prior, S, k; max_iter=5000, rounds=start_rounds)
    else
        prior.initialized || _initialize_prior!(prior, Matrix(Symmetric(S)))
    end
    if lds.fit_bool[5]
        _update_loading!(lds, suf)
        _refresh_excess!(om)
    end
    return nothing
end

# ----------------------------------------------------------------------------
# fit!
# ----------------------------------------------------------------------------

function _reject_structured_grouping(depends_on, depends_on_test)
    (depends_on === nothing && depends_on_test === nothing) || throw(
        ArgumentError(
            "`depends_on` is not supported for a constrained or kernel-prior emission"
        ),
    )
    return nothing
end

"""
    fit!(lds, y; max_iter=100, tol=1e-6, rtol=0.0, progress=true, start_rounds=30, ...)

Fit a Gaussian LDS whose emission is a
[`ConstrainedGaussianObservationModel`](@ref), a
[`KernelGaussianObservationModel`](@ref) or a
[`ConstrainedKernelGaussianObservationModel`](@ref), by (variational) EM. The
keywords are those of the Gaussian `fit!`, without `depends_on` and the
`gauge_*` options (the emission's own `gauge` field replaces them), plus
`start_rounds`: the most rounds of weights and trees the kernel prior takes on
the starting loading before the first iteration.

Returns the trace of the bound, one value per iteration (a [`FitTrace`](@ref)
with `y_test`). With a kernel prior the held-out score is the same bound on the
held-out trials at the fitted `q(H)` and weights; the terms that do not involve
the trials are in both traces.
"""
function fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::Observations{T};
    max_iter::Int=100,
    tol::Float64=1e-6,
    rtol::Float64=0.0,
    progress::Bool=true,
    start_rounds::Int=30,
    ux0=nothing,
    ux=nothing,
    uy=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
    y_test=nothing,
    ux0_test=nothing,
    ux_test=nothing,
    uy_test=nothing,
    depends_on_test::Union{Nothing,NamedTuple}=nothing,
    test_every::Int=1,
    early_stopping::Bool=false,
    patience::Int=1,
    min_delta::Real=0.0,
    restore_best::Bool=true,
    test_kwargs::NamedTuple=NamedTuple(),
    test_diagnostics::Bool=true,
    test_callback=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:StructuredGaussianObservationModel{T}}
    _reject_structured_grouping(depends_on, depends_on_test)
    _has_parameter_dependence(lds.state_model) && throw(
        ArgumentError(
            "`depends_on` is not supported for a constrained or kernel-prior emission"
        ),
    )
    data = Data(lds, y; ux0=ux0, ux=ux, uy=uy)
    monitor = _holdout_monitor(
        T,
        y_test;
        ux0_test=ux0_test,
        ux_test=ux_test,
        uy_test=uy_test,
        test_every=test_every,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        restore_best=restore_best,
        test_kwargs=test_kwargs,
        test_diagnostics=test_diagnostics,
        test_callback=test_callback,
    )
    return _fit_structured!(lds, data; max_iter, tol, rtol, progress, start_rounds, monitor)
end

function _fit_structured!(
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T};
    max_iter::Int=100,
    tol::Float64=1e-6,
    rtol::Float64=0.0,
    progress::Bool=true,
    start_rounds::Int=30,
    monitor=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:StructuredGaussianObservationModel{T}}
    tsteps_per_trial = data.tsteps
    T_max = maximum(tsteps_per_trial)
    ntrials_total = length(tsteps_per_trial)
    elbos = Vector{T}(undef, max_iter)

    cov_alias = ntrials_total > 1 && all(t -> t == tsteps_per_trial[1], tsteps_per_trial)
    tfs = initialize_FilterSmooth(
        lds, tsteps_per_trial; cov_alias=cov_alias
    )::TrialFilterSmooth{T}
    pool_size = Threads.maxthreadid()
    sws_pool = [
        SmoothWorkspace(
            T, lds.latent_dim, lds.obs_dim, T_max; ux_dim=lds.ux_dim, uy_dim=lds.uy_dim
        ) for _ in 1:pool_size
    ]
    suf = _initialize_td_sufficient_statistics(
        T, lds, tsteps_per_trial
    )::SufficientStatistics{T}
    _td_init_const_blocks!(sws_pool[1], lds, data)

    _start_kernel_fit!(lds, suf, tfs, data, sws_pool; start_rounds)

    prog = if progress
        Progress(max_iter; desc="Fitting kernel LDS via EM...", barlen=50, showspeed=true)
    else
        nothing
    end

    for iter in 1:max_iter
        estep!(lds, suf, tfs, data, sws_pool)
        iter == 1 && _start_covariance_priors!(lds, suf)
        total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths; init=zero(T))
        elbos[iter] = elbo!(lds, suf, sws_pool[1], total_entropy)

        _holdout_due(monitor, iter) &&
            _holdout_record!(monitor, lds, iter; posterior=tfs, train_elbo=elbos[iter])
        if _holdout_stop(monitor)
            prog !== nothing && finish!(prog)
            resize!(elbos, iter)
            return _fit_result(monitor, elbos, lds)
        end

        converged = _em_converged(elbos, iter, tol, rtol)
        #= Return a scored iterate: the trace's last entry is always the score of
        the parameters handed back, so no M-step runs after it. =#
        if converged || iter == max_iter
            prog !== nothing && finish!(prog)
            resize!(elbos, iter)
            return _fit_result(monitor, elbos, lds)
        end

        mstep!(lds, suf, sws_pool[1])
        _kernel_gauge_step!(lds, suf)

        prog !== nothing && next!(prog)
    end

    prog !== nothing && finish!(prog)
    return _fit_result(monitor, elbos, lds)
end

# ----------------------------------------------------------------------------
# The plug-in likelihood
# ----------------------------------------------------------------------------

# The Gaussian LDS at the mean loading, sharing every array.
function _plugin_gaussian(
    lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:GaussianStateModel{T},O<:StructuredGaussianObservationModel{T}}
    om = lds.obs_model
    plain = GaussianObservationModel(;
        C=om.C, R=om.R, d=om.d, D=om.D, R_prior=om.R_prior, R_diagonal=om.R_diagonal
    )
    return LinearDynamicalSystem{T,S,typeof(plain)}(
        lds.state_model,
        plain,
        lds.latent_dim,
        lds.obs_dim,
        lds.ux_dim,
        lds.uy_dim,
        lds.fit_bool[1:6],
    )
end

"""
    loglikelihood(lds, y; ux0=nothing, ux=nothing, uy=nothing)

For a constrained or kernel-prior emission: the marginal log-likelihood of the
Gaussian LDS at the mean loading `obs_model.C`, by the Kalman filter. With a
kernel prior this is a plug-in score, not the model's evidence: it leaves out
the uncertainty of the loading and the prior's terms, which the bound that
`fit!` and [`elbo`](@ref) report includes. It is comparable between models and
with a `GaussianObservationModel` fit, which is what makes it useful on
held-out trials.
"""
function StatsAPI.loglikelihood(
    lds::LinearDynamicalSystem{T,SM,OM},
    y::Observations{T};
    ux0=nothing,
    ux=nothing,
    uy=nothing,
) where {T<:Real,SM<:GaussianStateModel{T},OM<:StructuredGaussianObservationModel{T}}
    return loglikelihood(_plugin_gaussian(lds), y; ux0, ux, uy)
end
