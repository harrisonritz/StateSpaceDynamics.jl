#=============================================================================
Poisson LDS

    Log-Likelihood: joint_loglikelihood!(ws, plds, x, y[, lognorm_t, ux, uy])
                    joint_loglikelihood(plds, x, y)

    Gradient:       gradient!(ws, lds, x, y[, ux, uy])
                    gradient_observation_model!(grad, C, d, D, tfs, y, uy, sws_pool)

    Hessian:        (generic hessian! lives in continuous_latents.jl;
                    emission kernels in poisson_observations.jl)

    Smooth:         smooth(plds, y; ux, uy)         — public, allocating
                    smooth!(lds, fs, y, sws, ux, uy)
                    smooth!(lds, tfs, data, sws_pool)

    ELBO:           elbo(plds, y; ux, uy)           — public, allocating
                    elbo!(plds, suf, tfs, data, sws_pool)

    E-Step:         estep!(lds, suf, tfs, data, sws_pool)

    M-Step:         mstep!(plds, suf, tfs, data, sws_pool)

    Fit:            fit!(plds, y; ux, uy)

    Public entry points take plain arrays and construct a validated `Data`
    (see `utils/validation.jl`); the multi-trial backend consumes `Data`.
    The Poisson path supports both latent (dynamics) inputs `ux` via the
    `B u_{t-1}` term and observation inputs `uy` via the `D v_t` emission term
    (rate `λ = exp(C x + d + D v)`), threaded through the smoother/ELBO/M-step
    exactly as in the Gaussian path. `uy` also enters the Poisson emission
    curvature, since the rate depends on the linear predictor.
=============================================================================#

"""
    _poisson_lognorm_at(y, t)

One timestep of the Poisson emission normalizer, `Σᵢ log(y[i,t]!)`.
"""
@inline function _poisson_lognorm_at(y::AbstractMatrix{T}, t::Int) where {T<:Real}
    acc = zero(T)
    @inbounds for i in axes(y, 1)
        acc += _log_factorial(y[i, t])
    end
    return acc
end

"""
    _poisson_lognorm_t(y)

Per-timestep Poisson emission normalizer `lognorm_t[t] = Σᵢ log(y[i,t]!)`;
constant in the latents. Computed once per trial and handed to `joint_loglikelihood!`
(see [`_log_factorial`](@ref) for why the counts are not all sent to `loggamma`).
"""
function _poisson_lognorm_t(y::AbstractMatrix{T}) where {T<:Real}
    tsteps = size(y, 2)
    out = Vector{T}(undef, tsteps)
    @inbounds for t in 1:tsteps
        out[t] = _poisson_lognorm_at(y, t)
    end
    return out
end

"""
    _poisson_lognorm_all(y)

One `lognorm_t` vector per trial, or `nothing` when the emission is not Poisson.
Built once at fit entry and threaded through the smoother, so the `log(y!)`
constant is never recomputed inside a Newton line search.
"""
function _poisson_lognorm_all(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractVector{<:AbstractMatrix{T}}
) where {T<:Real,S<:AbstractStateModel,O<:PoissonObservationModel{T}}
    return [_poisson_lognorm_t(yt) for yt in y]
end

_poisson_lognorm_all(::LinearDynamicalSystem, ::AbstractVector) = nothing

"""
    _poisson_lognorm_one(lds, y) -> Vector or nothing

One trial's `lognorm_t`, or `nothing` for a non-Poisson emission. The
single-trial counterpart of [`_poisson_lognorm_all`](@ref).
"""
function _poisson_lognorm_one(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}
) where {T<:Real,S<:AbstractStateModel,O<:PoissonObservationModel{T}}
    return _poisson_lognorm_t(y)
end

_poisson_lognorm_one(::LinearDynamicalSystem, ::AbstractMatrix) = nothing

"""
    _emission_lognorm(lds, y)

The emission's data-only log-normalizer for one trial, hoisted out of the Newton
line search where it would otherwise be recomputed at every objective
evaluation.

`Σᵢ log(y!)` for a Poisson emission, and a `NamedTuple` of per-member
normalizers for a composite (`nothing` for a member that has none — a Gaussian
emission folds its normalizer into `cR`).
"""
_emission_lognorm(::LinearDynamicalSystem, y::AbstractMatrix) = _poisson_lognorm_t(y)

"""
    _joint_loglikelihood_total(sws, lds, x, y, lognorm, ux, uy) -> T

Total complete-data log-likelihood `log p(x, y)` at one latent path: the Newton
smoother's line-search objective. Dispatched so a composite emission can reach
its own per-member kernel with each member's precomputed normalizer.
"""
function _joint_loglikelihood_total(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    lognorm::AbstractVector{<:Real},
    ux::Union{Nothing,AbstractMatrix},
    uy::Union{Nothing,AbstractMatrix},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    return sum(joint_loglikelihood!(sws, lds, x, y, lognorm, ux, uy))
end

"""
    joint_loglikelihood!(ws, plds, x, y[, lognorm_t, ux, uy])

Per-timestep complete-data log-likelihood of a Poisson LDS, written into
`ws.opt.ll_vec` (an active-length view is returned — the workspace may be
pool-oversized). Requires `compute_smooth_constants!(ws, plds)` to have been
called. The rate follows the canonical Poisson GLM `λ_t = exp(C x_t + d + D v_t)`.

- `ll[1]` includes: log p(x₁) + log p(y₁ | x₁)
- `ll[t]` for t≥2 includes: log p(x_t | x_{t-1}) + log p(y_t | x_t)

Normalization terms (Gaussian logdet + log(2π) and Poisson `-log(y!)`) are
included, so `sum(ll)` is the exact complete-data log-density `log p(x, y)`.

Pass `ux` (state inputs, `t`-indexed like `x`) to include the `B u_{t-1}`
dynamics term and `uy` (observation inputs, `t`-indexed like `y`) to include the
`D v_t` emission term; `nothing` (default) or a zero-row matrix skips either.
"""
function joint_loglikelihood!(
    ws::SmoothWorkspace{T},
    plds::LinearDynamicalSystem{TM,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{TM},
    lognorm_t::AbstractVector{<:Real}=_poisson_lognorm_t(y),
    ux::Union{Nothing,AbstractMatrix}=nothing,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,TM<:Real,S<:GaussianStateModel{TM},O<:PoissonObservationModel{TM}}
    tsteps = size(y, 2)

    C = plds.obs_model.C
    d = plds.obs_model.d
    D_obs = plds.obs_model.D

    cc = ws.consts
    ll_vec = view(ws.opt.ll_vec, 1:tsteps)
    η = ws.opt.temp_dy                  # length obs_dim
    dx = ws.opt.temp_dx                 # length latent_dim
    tmp = ws.opt.temp_solve_Q           # length latent_dim

    @views for t in 1:tsteps
        # Emission (with -log(y!)): y_t'η_t - sum(exp(η_t)),
        # with η_t = Cx_t + d (+ D v_t)
        mul!(η, C, x[:, t])
        if uy !== nothing
            mul!(η, D_obs, uy[:, t], one(T), one(T))
        end
        @. η = η + d
        ll_vec[t] = dot(y[:, t], η) - sum(exp, η) - lognorm_t[t]

        # Prior (t = 1) / transition (t ≥ 2)
        ll_vec[t] += state_loglikelihood!(cc, dx, tmp, plds, x, t, ux)
    end

    return ll_vec
end

"""
    joint_loglikelihood(plds, x, y)

Per-timestep complete-data log-likelihood of a Poisson LDS for a single trial
(allocating convenience wrapper around `joint_loglikelihood!`).

# Arguments
- `plds::LinearDynamicalSystem`: the Poisson LDS model
- `x::AbstractMatrix`: latent states, (latent_dim × tsteps)
- `y::AbstractMatrix`: observed counts, (obs_dim × tsteps)
"""
function joint_loglikelihood(
    plds::LinearDynamicalSystem{T,S,O}, x::AbstractMatrix{U}, y::AbstractMatrix{T}
) where {U<:Real,T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    R = promote_type(T, U)
    tsteps = size(y, 2)

    ws = SmoothWorkspace(R, plds.latent_dim, plds.obs_dim, tsteps)
    compute_smooth_constants!(ws, plds)
    x_R = convert(AbstractMatrix{R}, x)

    return joint_loglikelihood!(ws, plds, x_R, y)
end

"""
    joint_loglikelihood(plds, x, y)

Multi-trial complete-data log-likelihood for a Poisson LDS. `x` and `y` are vectors
of per-trial matrices.
"""
function joint_loglikelihood(
    plds::LinearDynamicalSystem{T,S,O},
    x::AbstractVector{<:AbstractMatrix{<:Real}},
    y::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    ntrials = length(y)
    chunks = collect(partition(1:ntrials, max(1, cld(ntrials, Threads.nthreads()))))
    return tmapreduce(+, chunks) do chunk
        acc = zero(T)
        for n in chunk
            acc += sum(joint_loglikelihood(plds, x[n], y[n]))
        end
        return acc
    end
end

"""
    loglikelihood(plds, y)

Marginal (observed-data) log-likelihood for a Poisson LDS — **not implemented**.

The marginal `log p(y) = ∫ p(x, y) dx` is intractable for the Poisson observation
model (non-conjugate; there is no closed-form Kalman filter as in the Gaussian case).
Use `joint_loglikelihood(plds, x, y)` for the complete-data log-likelihood given a
trajectory `x`, or the ELBO returned by `fit!` as a lower bound on `log p(y)`.
"""
function StatsAPI.loglikelihood(
    plds::LinearDynamicalSystem{T,S,O}, y
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    return error(
        "marginal loglikelihood is not implemented for the Poisson LDS (the marginal " *
        "log p(y) is intractable). Use joint_loglikelihood(plds, x, y) for the " *
        "complete-data log-likelihood, or the ELBO from fit! as a lower bound.",
    )
end

"""
    gradient_observation_model_single_trial!(grad, C, d, D_obs, E_z, p_smooth, y, uy, weights, ...)

Compute the gradient for a single trial and add it to the accumulated gradient.

Treats `[C d D]` as a single regression matrix `W` of size
`obs_dim × (latent_dim + 1 + uy_dim)` on the augmented latent `z_aug = [x; 1; v]`
(where `v` is the observation input `uy`). The parameter layout is
`vcat(vec(C), d, vec(D)) == vec([C d D])` in column-major order — so callers of
this function don't need to know about the stacked view; the gradient
computation is unified across the three blocks. The MN-prior penalty (added by
`update_observation_model!`) lives over the same stacked `W`.

The rate is `λ = exp(C·E[x] + d + D·v + ρ)` with `ρ = ½ diag(C P C')`. Only the
`C` block carries the variance-correction term `CP = C·P`; the `d` and `D`
blocks have zero correction since `ρ` depends on neither `d` nor `D`.
"""
function gradient_observation_model_single_trial!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    D_obs::AbstractMatrix{T},
    E_z::AbstractMatrix{T},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
    uy::Union{Nothing,AbstractMatrix},
    weights::Union{Nothing,AbstractVector{T}},
    h::AbstractVector{T},
    ρ::AbstractVector{T},
    λ::AbstractVector{T},
    CP::AbstractMatrix{T},
) where {T<:Real}
    obs_dim, latent_dim = size(C)
    uy_dim = size(D_obs, 2)
    Dp1 = latent_dim + 1
    reg_dim = Dp1 + uy_dim
    tsteps = size(y, 2)

    # 2-D view of the gradient buffer as `[C d D]`-shaped W (obs_dim × reg_dim).
    grad_W = reshape(view(grad, 1:(obs_dim * reg_dim)), obs_dim, reg_dim)

    @views for t in 1:tsteps
        weight = isnothing(weights) ? one(T) : weights[t]

        E_z_t = E_z[:, t]
        P_smooth_t = p_smooth[:, :, t]
        y_t = y[:, t]

        mul!(h, C, E_z_t)
        h .+= d
        if uy !== nothing
            mul!(h, D_obs, uy[:, t], one(T), one(T))
        end
        mul!(CP, C, P_smooth_t)

        @views for i in 1:obs_dim
            ρ[i] = T(0.5) * dot(C[i, :], CP[i, :])
            λ[i] = exp(h[i] + ρ[i])
        end

        #=
        Stacked gradient ∂Q/∂W[i, j] = w · (y_t[i] − λ[i]) · z_aug[j]  −  w · λ[i] · CP_aug[i, j]
        where z_aug = [E_z_t; 1; v_t] and CP_aug[:, 1:D] = CP, with the d- and
        v-columns of CP_aug all zero (ρ has no dependence on the d- or D-columns
        of W). Split into three blocks to skip the always-zero CP_aug columns.
        =#
        for j in 1:latent_dim
            for i in 1:obs_dim
                grad_W[i, j] += weight * (y_t[i] * E_z_t[j] - λ[i] * (E_z_t[j] + CP[i, j]))
            end
        end
        # j = latent_dim + 1: z_aug[j] = 1, CP_aug[:, j] = 0 → grad ← grad + w · (y − λ)
        grad_W[:, Dp1] .+= weight .* (y_t .- λ)
        # D block (cols D+2 : D+1+uy_dim): z_aug[j] = v_t[j], CP_aug[:, j] = 0.
        if uy !== nothing
            v_t = uy[:, t]
            for jj in 1:uy_dim
                col = Dp1 + jj
                vj = v_t[jj]
                for i in 1:obs_dim
                    grad_W[i, col] += weight * ((y_t[i] - λ[i]) * vj)
                end
            end
        end
    end
end

"""
    gradient_observation_model!(grad, C, d, D_obs, tfs, y, uy, sws_pool, w)

Compute the gradient of the Q-function with respect to the stacked emission
parameters `[C d D]` using `TrialFilterSmooth`. `uy` is the per-trial vector of
observation-input matrices (or `nothing` for no inputs).
"""
function gradient_observation_model!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    D_obs::AbstractMatrix{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}},
    sws_pool::Vector{SmoothWorkspace{T}},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing;
    tasks_per_thread::Int=2,
) where {T<:Real}
    trials = length(tfs.FilterSmooths)
    npar = length(grad)
    @assert length(sws_pool[1].reg.CD) == npar && length(sws_pool[1].reg.Syz) == npar "Poisson gradient accumulator size $(length(sws_pool[1].reg.CD)) ≠ npar=$npar (expected obs_dim·(latent_dim+1+uy_dim); build the workspace with uy_dim=lds.uy_dim)"

    #=
    Cap ntasks at `length(sws_pool)` so each chunk gets its own
    pre-allocated workspace slot indexed by its position in the chunk
    iteration (not `threadid()`, which can migrate under task
    scheduling).
    =#
    desired = max(1, tasks_per_thread * Threads.nthreads())
    ntasks = min(trials, desired, length(sws_pool))
    chunk_size = max(1, cld(trials, ntasks))
    chunks = collect(partition(1:trials, chunk_size))

    tforeach(eachindex(chunks)) do task_idx
        #=
        Each chunk owns one workspace from the pool. Buffers
        used by `gradient_observation_model_single_trial!`
        (h/ρ/λ/CP) come from this workspace's existing
        `Q_obs!` scratch fields, and the per-chunk `acc`/`tmp`
        gradient accumulators are views into `.reg.CD` / `.reg.Syz`
        (both sized `obs_dim × Dp1 = npar` for Poisson, where
        `uy_dim = 0`).
        =#
        sws = sws_pool[task_idx]
        acc = vec(sws.reg.CD)
        tmp = vec(sws.reg.Syz)
        fill!(acc, zero(T))

        h_buf = sws.elbo.h_obs
        ρ_buf = sws.elbo.rho_obs
        λ_buf = sws.elbo.CEz_obs
        CP_buf = sws.elbo.CP_obs

        for k in chunks[task_idx]
            fill!(tmp, zero(T))

            fs = tfs[k]
            weights = isnothing(w) ? nothing : w[k]
            uy_k = isnothing(uy) ? nothing : uy[k]

            gradient_observation_model_single_trial!(
                tmp,
                C,
                d,
                D_obs,
                fs.x_smooth,
                fs.p_smooth,
                y[k],
                uy_k,
                weights,
                h_buf,
                ρ_buf,
                λ_buf,
                CP_buf,
            )

            @simd for i in 1:npar
                acc[i] += tmp[i]
            end
        end
    end

    # Deterministic reduction on the caller thread (chunk order is fixed).
    # Named `chunk_acc`, not `acc`: sharing the closure's `acc` binding would
    # box it, which OhMyThreads rejects.
    fill!(grad, zero(T))
    for task_idx in eachindex(chunks)
        chunk_acc = vec(sws_pool[task_idx].reg.CD)
        @simd for i in 1:npar
            grad[i] += chunk_acc[i]
        end
    end

    @. grad = -grad
    return grad
end

"""
    mstep!(plds, suf, tfs, data, sws_pool)

Suf-based M-step for a Poisson LDS. The Gaussian-state half (x0, P0, A&b&B, Q)
is updated from the aggregated sufficient statistics in `suf`. The Poisson
emission `[C d D]` is non-conjugate and still goes through the existing LBFGS
routine (`update_observation_model!`), which receives `data.uy` for the `D v`
term.
"""
function mstep!(
    plds::LinearDynamicalSystem{T,S,O},
    suf::SufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:NonQuadraticEmission{T}}
    sws = sws_pool[1]
    update_initial_state_mean!(plds, suf)
    update_initial_state_covariance!(plds, suf, sws)
    update_A_b!(plds, suf, sws)
    update_Q!(plds, suf, sws)
    update_observation_model!(plds, tfs, data.y, sws_pool; uy=data.uy)
    return nothing
end

"""
    _poisson_q_obs_total(plds, tfs, data, sws_pool)

Observation-side Q-term summed over trials. The Poisson emission is
irreducibly non-conjugate — there is no sufficient-statistic form — so this
stays a per-trial loop, chunked across the workspace pool.
"""
function _poisson_q_obs_total(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    y = data.y
    uy = data.uy
    ntrials = length(y)
    ntasks = min(ntrials, length(sws_pool))
    partial = zeros(T, ntasks)
    chunksize = cld(ntrials, ntasks)
    tforeach(1:ntasks) do i
        lo = (i - 1) * chunksize + 1
        hi = min(i * chunksize, ntrials)
        lo > hi && return nothing
        sws = sws_pool[i]
        acc = zero(T)
        for trial in lo:hi
            fs = tfs[trial]
            acc += Q_obs!(sws, plds, fs.x_smooth, fs.p_smooth, y[trial], uy[trial])
        end
        partial[i] = acc
        return nothing
    end
    return sum(partial)
end

"""
    elbo!(plds, suf, tfs, data, sws_pool)

Suf-based Poisson ELBO. Mirrors the Gaussian TD path's split:

* state-side Q-term via `Q_state!(sws, plds, suf)` from the aggregated
  sufficient statistics (O(D³) per E-step, not O(N·T·D²)),
* observation-side Q-term per-trial via the existing Poisson `Q_obs!`,
  which is irreducibly non-conjugate (no aggregator equivalent),
* posterior entropy from `tfs[trial].entropy` (filled by `smooth!`),
* the parameter log-priors, via the shared [`_state_prior_logdensity`](@ref)
  and the Poisson [`_obs_prior_logdensity`](@ref) — the same terms the Gaussian
  path uses on the state side, and the bare MN quadratic on `[C d D]` that
  matches the emission M-step objective.
"""
function elbo!(
    plds::LinearDynamicalSystem{T,S,O},
    suf::SufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    total_entropy = zero(T)
    for fs in tfs.FilterSmooths
        total_entropy += fs.entropy
    end

    compute_smooth_constants!(sws_pool[1], plds)
    Q_state_total = Q_state!(sws_pool[1], plds, suf)
    Q_obs_total = _poisson_q_obs_total(plds, tfs, data, sws_pool)

    prior_term =
        _state_prior_logdensity(plds, sws_pool[1]) +
        _obs_prior_logdensity(plds, sws_pool[1])

    return Q_state_total + Q_obs_total + prior_term + total_entropy
end

"""
    smooth!(lds, fs, y, sws, ux, uy; max_iter=20, tol=1e-6)

Poisson LDS smoothing using iterative Newton with block tridiagonal solve.
Uses `SmoothWorkspace` for pre-allocated buffers. `ux`/`uy` are the per-trial
latent-input `(ux_dim, T_i)` and observation-input `(uy_dim, T_i)` matrices
feeding the `B u_{t-1}` dynamics and `D v_t` emission terms; pass `0×T_i`
matrices when there are no inputs. The Poisson emission curvature depends on the
rate `λ = exp(Cx + d + D v)`, so `uy` also enters the Hessian.

Since the Poisson log-likelihood is non-quadratic, multiple Newton iterations
are required (unlike Gaussian LDS which converges in one step).
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    y::Union{AbstractMatrix{T},NamedTuple},
    sws::SmoothWorkspace{T},
    ux::AbstractMatrix{T},
    uy::Union{AbstractMatrix{T},NamedTuple};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:NonQuadraticEmission{T}}
    tsteps, D = _ntsteps(y), lds.latent_dim
    n_active = D * tsteps
    btd = sws.btd

    compute_smooth_constants!(sws, lds)

    x = fs.x_smooth

    #=
    Warm-start the Newton iteration from the previous EM iteration's smoothed
    mean. If the smoothed mean is all zeros (e.g., first EM iteration), use the
    prior mean x0 instead.
    =#
    if all(x .== 0)
        x .= lds.state_model.x0
    end

    # Active-length views into (possibly) oversized workspace buffers.
    X0 = view(sws.opt.X0, 1:n_active)
    grad_active = view(sws.opt.grad_buf, :, 1:tsteps)
    neg_diag_v = view(btd.neg_diag, 1:tsteps)
    neg_sub_v = view(btd.neg_sub, 1:(tsteps - 1))
    neg_super_v = view(btd.neg_super, 1:(tsteps - 1))

    ls = BackTrackingLS{T}()
    g = grad_active
    p = reshape(X0, D, tsteps)

    #=
    The line-search objective is the exact complete-data log-likelihood;
    hoisting the data-only normalizer makes that free per evaluation. Both are
    dispatched so a composite emission can hand each member its own normalizer
    (`nothing` for the members that have none).
    =#
    lognorm_t = _emission_lognorm(lds, y)
    ϕ!() = _joint_loglikelihood_total(sws, lds, x, y, lognorm_t, ux, uy)

    compute_grad! = (gcur, xcur) -> begin
        gradient!(gcur, sws, lds, xcur, y, ux, uy)
        return nothing
    end

    build_hess! = (xcur) -> begin
        hessian!(sws, lds, xcur, y, uy)
        _negate_blocks!(btd, tsteps)
        return nothing
    end

    solve_dir! =
        (pcur, gcur) -> begin
            gvec = vec(gcur)
            pvec = vec(pcur)
            copyto!(pvec, gvec)
            #=
            Negated Hessian is SPD at the MAP — use the SPD-specialised
            solve, which routes to LAPACK `pbsv` at the latent
            dimensionalities a state-space model is normally fitted at.
            =#
            block_tridiagonal_solve_spd!(
                pvec, neg_sub_v, neg_diag_v, neg_super_v, gvec, btd
            )
            return nothing
        end

    newton_smooth!(
        Val(:max),
        x,
        g,
        p,
        compute_grad!,
        build_hess!,
        solve_dir!,
        ϕ!,
        ls;
        max_iter=max_iter,
        tol=tol,
    )

    hessian!(sws, lds, x, y, uy)
    _negate_blocks!(btd, tsteps)

    logdet_precision = block_tridiagonal_inverse_logdet!(
        fs.p_smooth, fs.p_smooth_tt1, neg_sub_v, neg_diag_v, neg_super_v, btd
    )

    fs.entropy = gaussian_entropy_from_logdet(logdet_precision, n_active)

    @views for i in 1:tsteps
        Symmetrize!(fs.p_smooth[:, :, i])
    end

    return fs
end

# Backward-compatible no-input overload (zero-row ux/uy).
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    y::Union{AbstractMatrix{T},NamedTuple},
    sws::SmoothWorkspace{T};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:NonQuadraticEmission{T}}
    ux = zeros(T, 0, _ntsteps(y))
    uy = _zero_uy(lds, _ntsteps(y))
    return smooth!(lds, fs, y, sws, ux, uy; max_iter=max_iter, tol=tol)
end

"""
    smooth!(lds, tfs, data::Data, sws_pool; max_iter=20, tol=1e-6)

Multi-trial Poisson LDS smoothing. `data` carries the per-trial observations
`data.y`, latent inputs `data.ux`, and observation inputs `data.uy` (zero-row
when absent); each trial's `B u_{t-1}` and `D v_t` terms are threaded through
the single-trial smoother.
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:NonQuadraticEmission{T}}
    y = data.y
    ux = data.ux
    uy = data.uy
    ntrials = length(data.tsteps)

    if ntrials == 1
        smooth!(
            lds,
            tfs[1],
            _trial(y, 1),
            sws_pool[1],
            ux[1],
            _trial(uy, 1);
            max_iter=max_iter,
            tol=tol,
        )
        return tfs
    end

    ntasks = min(ntrials, length(sws_pool))
    chunksize = cld(ntrials, ntasks)

    tforeach(1:ntasks) do i
        lo = (i - 1) * chunksize + 1
        hi = min(i * chunksize, ntrials)
        lo > hi && return nothing
        sws = sws_pool[i]
        for trial in lo:hi
            smooth!(
                lds,
                tfs[trial],
                _trial(y, trial),
                sws,
                ux[trial],
                _trial(uy, trial);
                max_iter=max_iter,
                tol=tol,
            )
        end
    end

    return tfs
end

# Convenience overload: validate + canonicalize raw observations into a `Data`
# (no inputs) before smoothing.
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple},
    sws_pool::Vector{SmoothWorkspace{T}};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:NonQuadraticEmission{T}}
    return smooth!(lds, tfs, Data(lds, y), sws_pool; max_iter=max_iter, tol=tol)
end

"""
    smooth!(lds, tfs, y; max_iter=20, tol=1e-6)

Convenience method that creates a workspace pool sized at `max(T_i)`.
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:NonQuadraticEmission{T}}
    T_max = maximum(_trial_lengths(y))
    npool = Threads.maxthreadid()
    sws_pool = [
        SmoothWorkspace(
            T,
            lds.latent_dim,
            _ws_obs_dim(lds),
            T_max;
            ux_dim=lds.ux_dim,
            uy_dim=_ws_uy_dim(lds),
        ) for _ in 1:npool
    ]
    return smooth!(lds, tfs, y, sws_pool; max_iter=max_iter, tol=tol)
end

"""
    estep!(lds, suf, tfs, data, sws_pool; max_iter=20, tol=1e-6)

Suf-based Poisson E-step. Smooths, aggregates state-side sufficient
statistics into `suf` from each trial's smoother output (`x_smooth`,
`p_smooth`, `p_smooth_tt1`), and returns the suf-based ELBO.
"""
function estep!(
    lds::LinearDynamicalSystem{T,S,O},
    suf::Union{SufficientStatistics{T},NamedTuple},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:NonQuadraticEmission{T}}

    # smooth each trial
    smooth!(lds, tfs, data, sws_pool; max_iter=max_iter, tol=tol)

    # compute the sufficient statistics
    return _aggregate_td_suff_stats!(suf, tfs, lds, data, sws_pool[1])
end

"""
    elbo(plds, y; newton_max_iter=20, newton_tol=1e-6)

Evidence lower bound of a Poisson `LinearDynamicalSystem` at the current
parameters (allocating convenience wrapper around the workspace-based
[`elbo!`](@ref)): runs one Laplace E-step (iterative-Newton smoothing +
sufficient-statistics aggregation) and evaluates the ELBO at the resulting
Gaussian posterior approximation `q(x)`.

This is the same quantity `fit!` reports per iteration — a lower bound on the
(intractable) marginal `log p(y)`, plus any IW/MN prior log-density terms.

# Arguments
- `y`: observed counts — a `(obs_dim, T)` matrix, a `(obs_dim, T, ntrials)`
  array, or a `Vector{<:AbstractMatrix}` of per-trial `(obs_dim, T_i)`
  matrices (ragged lengths allowed).

# Keywords
- `newton_max_iter` / `newton_tol`: Newton-smoother iteration cap and
  convergence tolerance (as in `fit!`).
- `depends_on`: optional `NamedTuple` of per-trial label vectors overriding the
  `depends_on` declared on the models for this call (see the ancillary parameter
  dependency docs). Needed when this dataset's trial count differs from the one
  the labels on the model were written for.

Returns a scalar.
"""
function elbo(
    plds::LinearDynamicalSystem{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    ux=nothing,
    uy=nothing,
    newton_max_iter::Int=20,
    newton_tol::Float64=1e-6,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:NonQuadraticEmission{T}}
    data = Data(plds, y; ux=ux, uy=uy)
    grp = parameter_grouping(plds, length(data.tsteps); depends_on=depends_on, y=data.y)
    if grp !== nothing
        sws_pool = _grouped_sws_pool(plds, data)
        state = _grouped_fit_state(plds, data, grp, sws_pool)
        return _grouped_estep_elbo_poisson!(
            state, grp, sws_pool; max_iter=newton_max_iter, tol=T(newton_tol)
        )
    end
    tfs = initialize_FilterSmooth(plds, data.tsteps)::TrialFilterSmooth{T}
    npool = min(Threads.maxthreadid(), length(data.tsteps))
    sws_pool = [
        SmoothWorkspace(
            T,
            plds.latent_dim,
            _ws_obs_dim(plds),
            maximum(data.tsteps);
            ux_dim=plds.ux_dim,
            uy_dim=_ws_uy_dim(plds),
        ) for _ in 1:npool
    ]
    suf = _initialize_td_sufficient_statistics(T, plds, data.tsteps)
    _td_init_const_blocks!(sws_pool[1], plds, data)

    estep!(plds, suf, tfs, data, sws_pool; max_iter=newton_max_iter, tol=T(newton_tol))

    return elbo!(plds, suf, tfs, data, sws_pool)
end

"""
    smooth(plds, y)

Direct smoothing for a Poisson LDS (allocating convenience wrapper around the
iterative-Newton `smooth!`).

# Arguments
- `plds::LinearDynamicalSystem`: the Poisson LDS model.
- `y`: observed counts — a `(obs_dim, T)` matrix (single trial), a
  `(obs_dim, T, ntrials)` array, or a `Vector{<:AbstractMatrix}` of per-trial
  `(obs_dim, T_i)` matrices (ragged lengths allowed).
- `depends_on`: optional `NamedTuple` of per-trial label vectors overriding the
  `depends_on` declared on the models for this call (see the ancillary parameter
  dependency docs). Needed when this dataset's trial count differs from the one
  the labels on the model were written for.

# Returns
For a single-trial (matrix) `y`:
- `x_smooth::Matrix`: smoothed latent means (latent_dim × tsteps).
- `p_smooth::Array{T,3}`: smoothed latent covariances (latent_dim × latent_dim × tsteps).

For multi-trial `y`: `Vector`s of the above, one entry per trial.
"""
function smooth(
    plds::LinearDynamicalSystem{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    ux=nothing,
    uy=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:NonQuadraticEmission{T}}
    data = Data(plds, y; ux=ux, uy=uy)
    grp = parameter_grouping(plds, length(data.tsteps); depends_on=depends_on, y=data.y)
    grp === nothing || return _grouped_smooth(plds, data, grp, y)
    tfs = initialize_FilterSmooth(plds, data.tsteps)::TrialFilterSmooth{T}
    # Cap the pool at the trial count — workspaces beyond ntrials are never
    # touched and each carries O(D²·T) of block-tridiagonal storage.
    npool = min(Threads.maxthreadid(), length(data.tsteps))
    sws_pool = [
        SmoothWorkspace(
            T,
            plds.latent_dim,
            _ws_obs_dim(plds),
            maximum(data.tsteps);
            ux_dim=plds.ux_dim,
            uy_dim=_ws_uy_dim(plds),
        ) for _ in 1:npool
    ]
    smooth!(plds, tfs, data, sws_pool)
    return _collect_smooth_output(tfs, y)
end

"""
    fit!(plds, y; max_iter=100, tol=1e-6, progress=true, newton_max_iter=20, newton_tol=1e-6)

Fit a Poisson LDS via Laplace-EM.

# Arguments
- `plds`: the Poisson LDS model (modified in place)
- `y`: observations. Three shapes accepted:
    * `AbstractMatrix{T}` of size `(obs_dim, T)` — single trial
    * `AbstractArray{T,3}` of size `(obs_dim, T, ntrials)` — equal-length multi-trial
    * `AbstractVector{<:AbstractMatrix{T}}` — multi-trial, each `(obs_dim, T_i)`,
      ragged trial lengths allowed

# Keywords
- `ux`: latent (dynamics) inputs feeding the `B u_{t-1}` term; same shapes as
  `y` with `ux_dim` rows. Required when `size(state_model.B, 2) > 0`; `nothing`
  (default) means no inputs.
- `uy`: observation inputs feeding the `D v_t` emission term; same shapes as `y`
  with `uy_dim` rows. Required when `size(obs_model.D, 2) > 0`; `nothing`
  (default) means no inputs.
- `max_iter`: maximum EM iterations
- `tol`: convergence tolerance on ELBO change
- `progress`: show progress bar
- `newton_max_iter`: Newton iterations per E-step inner solve
- `newton_tol`: Newton convergence tolerance
- `depends_on`: optional `NamedTuple` of per-trial label vectors overriding the
  `depends_on` declared on the models for this call (see the ancillary parameter
  dependency docs). Needed when this dataset's trial count differs from the one
  the labels on the model were written for.
"""
function fit!(
    plds::LinearDynamicalSystem{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    ux=nothing,
    uy=nothing,
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress=true,
    newton_max_iter::Int=20,
    newton_tol::Float64=1e-6,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:NonQuadraticEmission{T}}
    data = Data(plds, y; ux=ux, uy=uy)
    grp = parameter_grouping(plds, length(data.tsteps); depends_on=depends_on, y=data.y)
    grp === nothing || return _fit_plds_grouped!(
        plds,
        data,
        grp;
        max_iter=max_iter,
        tol=tol,
        progress=progress,
        newton_max_iter=newton_max_iter,
        newton_tol=newton_tol,
    )
    T_max = maximum(data.tsteps)

    tfs = initialize_FilterSmooth(plds, data.tsteps)::TrialFilterSmooth{T}

    npool = Threads.maxthreadid()
    sws_pool = [
        SmoothWorkspace(
            T,
            plds.latent_dim,
            _ws_obs_dim(plds),
            T_max;
            ux_dim=plds.ux_dim,
            uy_dim=_ws_uy_dim(plds),
        ) for _ in 1:npool
    ]

    suf = _initialize_td_sufficient_statistics(T, plds, data.tsteps)
    _td_init_const_blocks!(sws_pool[1], plds, data)

    elbos = Vector{T}(undef, max_iter)

    prog = if progress
        Progress(
            max_iter;
            desc="Fitting Poisson LDS via LaPlaceEM...",
            barlen=50,
            showspeed=true,
        )
    else
        nothing
    end

    for iter in 1:max_iter

        # E-step: smooth each trial, aggregate state-side suff-stats, compute ELBO
        estep!(plds, suf, tfs, data, sws_pool; max_iter=newton_max_iter, tol=T(newton_tol))

        # compute the ELBO
        elbos[iter] = elbo!(plds, suf, tfs, data, sws_pool)

        # M-step: update state-side suff-stats from suf, update Poisson emission via LBFGS
        mstep!(plds, suf, tfs, data, sws_pool)

        # print progress
        prog !== nothing && next!(prog)

        # check convergence
        if iter > 1 && abs(elbos[iter] - elbos[iter - 1]) < tol
            prog !== nothing && finish!(prog)
            resize!(elbos, iter)
            return elbos
        end
    end

    prog !== nothing && finish!(prog)
    return elbos
end

"""
    _grouped_estep_elbo_poisson!(state, grp, sws_pool; max_iter, tol)

One grouped Laplace E-step for a Poisson LDS: smooth and aggregate each cell
with its own parameters, and return the total ELBO. Mirrors
`_grouped_estep_elbo_gaussian!`, but the observation-side Q-term stays a
per-trial loop (the Poisson emission has no sufficient-statistic form).
"""
function _grouped_estep_elbo_poisson!(
    state::GroupedFitState{T,L},
    grp::ParameterGrouping,
    sws_pool::Vector{SmoothWorkspace{T}};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {
    T<:Real,L<:LinearDynamicalSystem{T,<:GaussianStateModel{T},<:NonQuadraticEmission{T}}
}
    total = zero(T)
    for c in 1:(grp.ncells)
        lds_c = state.cell_lds[c]
        suf_c = state.sufs[c]
        tfs_c = state.cell_tfs[c]
        cell_pool = _prepare_cell!(sws_pool, state, c)
        estep!(
            lds_c, suf_c, tfs_c, state.cell_data[c], cell_pool; max_iter=max_iter, tol=tol
        )

        compute_smooth_constants!(cell_pool[1], lds_c)
        total += Q_state!(cell_pool[1], lds_c, _state_suf(suf_c))
        total += _grouped_cell_q_obs(lds_c, suf_c, tfs_c, state.cell_data[c], cell_pool)
        for fs in tfs_c.FilterSmooths
            total += fs.entropy
        end
    end
    total += _grouped_state_prior_logdensity(state.cell_lds, grp.cell_slot, T)
    total += _grouped_obs_prior_logdensity(
        state.cell_lds[1], state.cell_lds, grp.cell_slot, T
    )
    return total
end

"""
    _grouped_cell_q_obs(lds_c, suf_c, tfs_c, data_c, cell_pool) -> T

One cell's emission Q-term on the Laplace path: the per-trial Poisson Q for a
single emission, and the sum over members for a composite.
"""
function _grouped_cell_q_obs(
    lds_c::LinearDynamicalSystem{T,S,O},
    ::SufficientStatistics{T},
    tfs_c::TrialFilterSmooth{T},
    data_c::Data{T},
    cell_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    return _poisson_q_obs_total(lds_c, tfs_c, data_c, cell_pool)
end

function _grouped_cell_q_obs(
    lds_c::LinearDynamicalSystem{T,S,O},
    suf_c::NamedTuple,
    tfs_c::TrialFilterSmooth{T},
    data_c::Data{T},
    cell_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:CompositeObservationModel{T}}
    return _composite_q_obs_total(lds_c, suf_c, tfs_c, data_c, cell_pool)
end

"""
    _grouped_update_observation_model!(state, grp, data, sws_pool)

Poisson emission M-step, once per version of `[C d D]`: the LBFGS solve runs on
the trials of every cell that shares that version, so a `[C d D]` shared across
cells is still fit from all of their data.
"""
function _grouped_update_observation_model!(
    state::GroupedFitState{T,L},
    grp::ParameterGrouping,
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {
    T<:Real,L<:LinearDynamicalSystem{T,<:GaussianStateModel{T},<:PoissonObservationModel{T}}
}
    ord = _obs_slot_ordinals(state.cell_lds[1].obs_model)[1]
    _grouped_poisson_emission!(
        state, grp, data.y, data.uy, sws_pool, grp.cell_slot[ord[1]], nothing
    )
    return nothing
end

#=
A composite on the Laplace path updates each member by whichever route that
member supports: the conjugate regression and IW update for a Gaussian member,
the row-wise Newton solve over pooled trials for a Poisson one. The Gaussian
members are exactly the grouped update the all-Gaussian path runs, so nothing is
duplicated for them.
=#
function _grouped_update_observation_model!(
    state::GroupedFitState{T,L},
    grp::ParameterGrouping,
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {
    T<:Real,
    L<:LinearDynamicalSystem{T,<:GaussianStateModel{T},<:CompositeObservationModel{T}},
}
    om = state.cell_lds[1].obs_model
    ords = _obs_slot_ordinals(om)
    cell_ws1 = [p[1] for p in state.cell_sws]
    for (m, key) in enumerate(_obs_keys(om))
        _grouped_member_obs_mstep!(
            _models(om)[key], state, grp, data, sws_pool, cell_ws1, key, m, ords[m]
        )
    end
    return nothing
end

function _grouped_member_obs_mstep!(
    ::GaussianObservationModel,
    state::GroupedFitState{T},
    grp::ParameterGrouping,
    ::Data{T},
    ::Vector{SmoothWorkspace{T}},
    cell_ws1::AbstractVector,
    key::Symbol,
    m::Int,
    ord::UnitRange{Int},
) where {T<:Real}
    views = _member_unit_views(state.cell_lds, key)
    member_sws = _member_unit_sws(cell_ws1, state.cell_lds, m)
    _grouped_gaussian_obs_mstep!(
        views,
        [s[key] for s in state.sufs],
        grp.cell_slot[ord[1]],
        grp.cell_slot[ord[2]],
        member_sws[1],
        state.bufs[key];
        unit_sws=member_sws,
    )
    return nothing
end

function _grouped_member_obs_mstep!(
    ::PoissonObservationModel,
    state::GroupedFitState{T},
    grp::ParameterGrouping,
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
    ::AbstractVector,
    key::Symbol,
    m::Int,
    ord::UnitRange{Int},
) where {T<:Real}
    _grouped_poisson_emission!(
        state, grp, data.y[key], data.uy[key], sws_pool, grp.cell_slot[ord[1]], (key, m)
    )
    return nothing
end

"""
    _grouped_poisson_emission!(state, grp, y, uy, sws_pool, slots_cd, member)

One LBFGS/Newton emission solve per version of a Poisson `[C d D]`, over the
trials of every cell that shares that version — so a `[C d D]` shared across
cells is still fitted from all of their data.

`member` is `nothing` for a single Poisson emission, or `(key, m)` to run the
same solve for one member of a composite, on that member's views and
sub-workspaces.
"""
function _grouped_poisson_emission!(
    state::GroupedFitState{T},
    grp::ParameterGrouping,
    y::AbstractVector{<:AbstractMatrix{T}},
    uy::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
    slots_cd::AbstractVector{Int},
    member::Union{Nothing,Tuple{Symbol,Int}},
) where {T<:Real}
    for units in _units_by_slot(slots_cd)
        trials = Int[]
        for c in units
            append!(trials, grp.cell_trials[c])
        end
        sort!(trials)
        tfs = TrialFilterSmooth([state.tfs_all[n] for n in trials])

        # solve using the pooled workspace for each cell
        cell_pool = _prepare_cell!(sws_pool, state, units[1])
        lds_u = state.cell_lds[units[1]]
        if member === nothing
            update_observation_model!(lds_u, tfs, y[trials], cell_pool; uy=uy[trials])
        else
            key, m = member
            update_observation_model!(
                _obs_view(lds_u, key),
                tfs,
                y[trials],
                _member_pools(cell_pool, lds_u)[m];
                uy=uy[trials],
            )
        end
    end
    return nothing
end

"""
    _fit_plds_grouped!(plds, data, grp; ...)

Laplace-EM driver for a Poisson LDS whose parameters depend on an ancillary
variable. Same structure as the ungrouped `fit!`: the state side flows through
the pooled sufficient statistics, the emission through one LBFGS solve per
`[C d D]` version.
"""
function _fit_plds_grouped!(
    plds::LinearDynamicalSystem{T,S,O},
    data::Data{T},
    grp::ParameterGrouping;
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress=true,
    newton_max_iter::Int=20,
    newton_tol::Float64=1e-6,
) where {T<:Real,S<:GaussianStateModel{T},O<:NonQuadraticEmission{T}}
    sws_pool = _grouped_sws_pool(plds, data)
    state = _grouped_fit_state(plds, data, grp, sws_pool)
    elbos = Vector{T}(undef, max_iter)

    prog = if progress
        Progress(
            max_iter;
            desc="Fitting grouped Poisson LDS via LaPlaceEM...",
            barlen=50,
            showspeed=true,
        )
    else
        nothing
    end

    for iter in 1:max_iter
        elbos[iter] = _grouped_estep_elbo_poisson!(
            state, grp, sws_pool; max_iter=newton_max_iter, tol=T(newton_tol)
        )

        _grouped_state_mstep!(
            state.cell_lds,
            _state_sufs(state.sufs),
            grp.cell_slot,
            sws_pool[1],
            _state_bufs(state.bufs),
        )
        _grouped_update_observation_model!(state, grp, data, sws_pool)

        prog !== nothing && next!(prog)

        if iter > 1 && abs(elbos[iter] - elbos[iter - 1]) < tol
            prog !== nothing && finish!(prog)
            resize!(elbos, iter)
            return elbos
        end
    end

    prog !== nothing && finish!(prog)
    return elbos
end
