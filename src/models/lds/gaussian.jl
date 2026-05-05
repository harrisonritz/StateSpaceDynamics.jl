function _extract_state_params(state_model::GaussianStateModel{T}) where {T}
    return (
        A=state_model.A,
        Q=state_model.Q,
        b=state_model.b,
        x0=state_model.x0,
        P0=state_model.P0,
    )
end

"""
    initialize_FilterSmooth(model, tsteps::Int)

Initialize a per-trial `FilterSmooth` buffer sized for `tsteps` timesteps.
"""
function initialize_FilterSmooth(
    model::LinearDynamicalSystem{T,S,O}, tsteps::Int
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    D = model.latent_dim
    return FilterSmooth{T}(
        zeros(T, D, tsteps),                              # x_smooth
        zeros(T, D, D, tsteps),                           # p_smooth
        zeros(T, D, D, tsteps),                           # p_smooth_tt1
        zeros(T, D, tsteps),                              # E_z
        zeros(T, D, D, tsteps),                           # E_zz
        zeros(T, D, D, tsteps),                           # E_zz_prev
        zero(T),                                          # entropy
    )
end

"""
    initialize_FilterSmooth(model, tsteps_per_trial::AbstractVector{<:Integer})

Initialize a `TrialFilterSmooth` with one `FilterSmooth` per trial. Trial lengths may
differ.
"""
function initialize_FilterSmooth(
    model::LinearDynamicalSystem{T,S,O}, tsteps_per_trial::AbstractVector{<:Integer}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    filter_smooths = [initialize_FilterSmooth(model, Int(t)) for t in tsteps_per_trial]
    return TrialFilterSmooth(filter_smooths)
end

function _extract_obs_params(obs_model::GaussianObservationModel{T}) where {T}
    return (C=obs_model.C, R=obs_model.R, d=obs_model.d)
end

function _extract_obs_params(obs_model::PoissonObservationModel{T}) where {T}
    return (C=obs_model.C, log_d=obs_model.log_d, d=exp.(obs_model.log_d))
end

function _get_all_params_vec(
    lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}}
    state_params = _extract_state_params(lds.state_model)
    obs_params = _extract_obs_params(lds.obs_model)

    # Convert named tuples to vectors and concatenate
    state_vec = vcat(
        vec(state_params.A),
        vec(state_params.Q),
        vec(state_params.b),
        vec(state_params.x0),
        vec(state_params.P0),
    )

    if lds.obs_model isa GaussianObservationModel
        obs_vec = vcat(vec(obs_params.C), vec(obs_params.R), vec(obs_params.d))
    else # PoissonObservationModel
        obs_vec = vcat(vec(obs_params.C), vec(obs_params.log_d))
    end

    return vcat(state_vec, obs_vec)
end

function _sample_trial!(
    rng, x_trial, y_trial, state_params, obs_params, obs_model::GaussianObservationModel
)
    tsteps = size(x_trial, 2)

    # Initial state
    x_trial[:, 1] = rand(rng, MvNormal(state_params.x0, state_params.P0))
    y_trial[:, 1] = rand(
        rng, MvNormal(obs_params.C * x_trial[:, 1] + obs_params.d, obs_params.R)
    )

    # Subsequent states
    for t in 2:tsteps
        x_trial[:, t] = rand(
            rng,
            MvNormal(state_params.A * x_trial[:, t - 1] + state_params.b, state_params.Q),
        )
        y_trial[:, t] = rand(
            rng, MvNormal(obs_params.C * x_trial[:, t] + obs_params.d, obs_params.R)
        )
    end
end

function _sample_trial!(
    rng, x_trial, y_trial, state_params, obs_params, obs_model::PoissonObservationModel
)
    tsteps = size(x_trial, 2)

    # Initial state
    x_trial[:, 1] = rand(rng, MvNormal(state_params.x0, state_params.P0))
    y_trial[:, 1] = rand.(rng, Poisson.(exp.(obs_params.C * x_trial[:, 1] + obs_params.d)))

    # Subsequent states
    for t in 2:tsteps
        x_trial[:, t] = rand(
            rng,
            MvNormal(state_params.A * x_trial[:, t - 1] + state_params.b, state_params.Q),
        )
        y_trial[:, t] = rand.(
            rng, Poisson.(exp.(obs_params.C * x_trial[:, t] + obs_params.d))
        )
    end
end

"""
    Random.rand([rng,] lds, tsteps::Integer)
    Random.rand([rng,] lds, tsteps_per_trial::AbstractVector{<:Integer})

Sample from a Linear Dynamical System.

- With a scalar `tsteps`, returns one trial as `(x::Matrix, y::Matrix)` of sizes
  `(latent_dim, tsteps)` and `(obs_dim, tsteps)` respectively.
- With a vector of per-trial lengths, returns
  `(x::Vector{Matrix}, y::Vector{Matrix})`. Lengths may differ across trials.
"""
function Random.rand(
    rng::AbstractRNG, lds::LinearDynamicalSystem{T,S,O}, tsteps::Integer
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    state_params = _extract_state_params(lds.state_model)
    obs_params = _extract_obs_params(lds.obs_model)

    x = Matrix{T}(undef, lds.latent_dim, Int(tsteps))
    y = Matrix{T}(undef, lds.obs_dim, Int(tsteps))
    _sample_trial!(rng, x, y, state_params, obs_params, lds.obs_model)
    return x, y
end

function Random.rand(
    rng::AbstractRNG,
    lds::LinearDynamicalSystem{T,S,O},
    tsteps_per_trial::AbstractVector{<:Integer},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    state_params = _extract_state_params(lds.state_model)
    obs_params = _extract_obs_params(lds.obs_model)

    ntrials = length(tsteps_per_trial)
    x = Vector{Matrix{T}}(undef, ntrials)
    y = Vector{Matrix{T}}(undef, ntrials)
    for i in 1:ntrials
        Ti = Int(tsteps_per_trial[i])
        x[i] = Matrix{T}(undef, lds.latent_dim, Ti)
        y[i] = Matrix{T}(undef, lds.obs_dim, Ti)
    end

    if ntrials > 10
        Threads.@threads for trial in 1:ntrials
            _sample_trial!(rng, x[trial], y[trial], state_params, obs_params, lds.obs_model)
        end
    else
        for trial in 1:ntrials
            _sample_trial!(rng, x[trial], y[trial], state_params, obs_params, lds.obs_model)
        end
    end

    return x, y
end

function Random.rand(lds::LinearDynamicalSystem, tsteps::Integer)
    return rand(Random.default_rng(), lds, tsteps)
end

function Random.rand(
    lds::LinearDynamicalSystem, tsteps_per_trial::AbstractVector{<:Integer}
)
    return rand(Random.default_rng(), lds, tsteps_per_trial)
end

"""
    loglikelihood!(ws, x, lds, y)

In-place version of `loglikelihood` that uses pre-computed Cholesky factors from
`ws::SmoothWorkspace` and writes into `ws.ll_vec`. Returns the sum of log-likelihoods.
"""
function loglikelihood!(
    ws::SmoothWorkspace{T},
    x::AbstractMatrix{T},
    lds::LinearDynamicalSystem{T0,S,O},
    y::AbstractMatrix{T0},
) where {T<:Real,T0<:Real,S<:GaussianStateModel{T0},O<:GaussianObservationModel{T0}}
    tsteps = size(y, 2)

    A = lds.state_model.A
    b = lds.state_model.b
    x0 = lds.state_model.x0
    d = lds.obs_model.d

    R_U = UpperTriangular(ws.R_chol_U)
    Q_U = UpperTriangular(ws.Q_chol_U)
    P0_U = UpperTriangular(ws.P0_chol_U)

    ll_vec = ws.ll_vec
    temp_dx = ws.temp_dx
    temp_dy = ws.temp_dy
    temp_solve_Q = ws.temp_solve_Q
    temp_solve_R = ws.temp_solve_R

    latent_dim = lds.latent_dim
    obs_dim = lds.obs_dim

    cP0 = -T(0.5) * (T(latent_dim) * log(T(2π)) + _logdet_from_U(ws.P0_chol_U, latent_dim))
    cQ = -T(0.5) * (T(latent_dim) * log(T(2π)) + _logdet_from_U(ws.Q_chol_U, latent_dim))
    cR = -T(0.5) * (T(obs_dim) * log(T(2π)) + _logdet_from_U(ws.R_chol_U, obs_dim))

    for t in 1:tsteps
        ll_t = zero(T)

        # Initial state (t=1): log p(x1)
        if t == 1
            @views temp_dx .= x[:, 1] .- x0
            ldiv!(temp_solve_Q, P0_U, temp_dx)
            ll_t += cP0 - T(0.5) * sum(abs2, temp_solve_Q)
        end

        # Dynamics (t>1): log p(x_t | x_{t-1})
        if t > 1
            @views mul!(temp_dx, A, x[:, t - 1])
            @views temp_dx .= x[:, t] .- temp_dx .- b
            ldiv!(temp_solve_Q, Q_U, temp_dx)
            ll_t += cQ - T(0.5) * sum(abs2, temp_solve_Q)
        end

        # Emission: log p(y_t | x_t)
        @views mul!(temp_dy, lds.obs_model.C, x[:, t])
        @views temp_dy .= y[:, t] .- temp_dy .- d
        ldiv!(temp_solve_R, R_U, temp_dy)
        ll_t += cR - T(0.5) * sum(abs2, temp_solve_R)

        ll_vec[t] = ll_t
    end

    return ll_vec
end

"""
    loglikelihood!(ws, x, lds, y)

Compute per-timestep complete-data log-likelihood contributions for a Gaussian LDS:

- `ll[1]` includes: log p(x₁) + log p(y₁ | x₁)
- `ll[t]` for t≥2 includes: log p(x_t | x_{t-1}) + log p(y_t | x_t)

Writes into `ws.ll_vec` and returns it.

Notes:
- Normalization terms (logdet + log(2π)) are included. These are constant w.r.t. `x`,
  but **not** constant across SLDS discrete states when `Q`/`R` differ by state.
"""
function loglikelihood!(
    ll::AbstractVector{T},
    ws::SLDSSmoothWorkspace{T},
    cc::LDSConstantCache{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    @assert length(ll) == tsteps

    A = lds.state_model.A
    b = lds.state_model.b
    x0 = lds.state_model.x0
    C = lds.obs_model.C
    d = lds.obs_model.d

    Q_U = UpperTriangular(cc.Q_chol_U)
    P0_U = UpperTriangular(cc.P0_chol_U)
    R_U = UpperTriangular(cc.R_chol_U)

    dxt = ws.dxt
    dyt = ws.dyt
    tmp = ws.tmp1  # latent_dim work vector (used as transition residual)

    for t in 1:tsteps
        ll_t = zero(T)

        # emission: cR - 0.5*||R^{-1/2}(y_t - Cx_t - d)||^2
        @views mul!(dyt, C, x[:, t])
        @views dyt .= y[:, t] .- dyt .- d
        ldiv!(dyt, R_U, dyt)
        ll_t += cc.cR - T(0.5) * sum(abs2, dyt)

        if t == 1
            # prior: cP0 - 0.5*||P0^{-1/2}(x1 - x0)||^2
            @views dxt .= x[:, 1] .- x0
            ldiv!(dxt, P0_U, dxt)
            ll_t += cc.cP0 - T(0.5) * sum(abs2, dxt)
        else
            # transition: cQ - 0.5*||Q^{-1/2}(x_t - A x_{t-1} - b)||^2
            @views mul!(tmp, A, x[:, t - 1])
            @views tmp .= x[:, t] .- tmp .- b
            ldiv!(tmp, Q_U, tmp)
            ll_t += cc.cQ - T(0.5) * sum(abs2, tmp)
        end

        ll[t] = ll_t
    end

    return ll
end

"""
    Gradient!(ws, lds, y, x)

In-place version of `Gradient` that uses pre-computed Cholesky-derived terms from
`ws::SmoothWorkspace` and writes the result into `ws.grad_buf`.
Returns `ws.grad_buf`.
"""
function Gradient!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    latent_dim, tsteps = size(x)
    A = lds.state_model.A
    b = lds.state_model.b
    x0 = lds.state_model.x0

    C_inv_R = ws.C_inv_R
    A_inv_Q = ws.A_inv_Q
    # ws.x_t = -P0^{-1}, ws.xt_given_xt_1 = -Q^{-1}
    # So P0^{-1}*v = -(ws.x_t * v), Q^{-1}*v = -(ws.xt_given_xt_1 * v)
    neg_P0_inv = ws.x_t         # = -P0^{-1}
    neg_Q_inv = ws.xt_given_xt_1  # = -Q^{-1}

    grad = ws.grad_buf
    dxt = ws.dxt
    dxt_next = ws.dxt_next
    dyt = ws.dyt
    tmp1 = ws.tmp1
    tmp2 = ws.tmp2
    tmp3 = ws.tmp3

    # First time step
    dxt .= x[:, 1] .- x0
    mul!(dxt_next, A, view(x, :, 1))
    @views dxt_next .= x[:, 2] .- dxt_next .- b
    mul!(dyt, lds.obs_model.C, view(x, :, 1))
    @views dyt .= y[:, 1] .- dyt .- lds.obs_model.d

    mul!(tmp1, C_inv_R, dyt)
    mul!(tmp2, A_inv_Q, dxt_next)
    # P0^{-1} * dxt = -(neg_P0_inv * dxt)
    mul!(tmp3, neg_P0_inv, dxt)

    # grad[:,1] = tmp1 + tmp2 - (-neg_P0_inv * dxt) = tmp1 + tmp2 + tmp3
    grad[:, 1] .= tmp1 .+ tmp2 .+ tmp3

    # Middle steps
    @views for t in 2:(tsteps - 1)
        mul!(dxt, A, x[:, t - 1])
        dxt .= x[:, t] .- dxt .- b

        mul!(dxt_next, A, x[:, t])
        dxt_next .= x[:, t + 1] .- dxt_next .- b

        mul!(dyt, lds.obs_model.C, x[:, t])
        dyt .= y[:, t] .- dyt .- lds.obs_model.d

        mul!(tmp1, C_inv_R, dyt)
        mul!(tmp2, A_inv_Q, dxt_next)
        # Q^{-1}*dxt = -neg_Q_inv * dxt, so neg_Q_inv*dxt = -Q^{-1}*dxt
        # grad = tmp1 - Q^{-1}*dxt + tmp2 = tmp1 + neg_Q_inv*dxt + tmp2
        mul!(tmp3, neg_Q_inv, dxt)

        grad[:, t] .= tmp1 .+ tmp3 .+ tmp2
    end

    # Last time step
    @views begin
        mul!(dxt, A, x[:, tsteps - 1])
        dxt .= x[:, tsteps] .- dxt .- b
        mul!(dyt, lds.obs_model.C, x[:, tsteps])
        dyt .= y[:, tsteps] .- dyt .- lds.obs_model.d

        mul!(tmp1, C_inv_R, dyt)
        mul!(tmp3, neg_Q_inv, dxt)

        grad[:, tsteps] .= tmp1 .+ tmp3
    end

    return grad
end

"""
    Hessian!(sws, lds, y, x)

Fill `sws.btd.H_diag`, `H_sub`, `H_super` with the log-likelihood Hessian blocks for
the active trial (length derived from `size(y, 2)`). Returns nothing — the sparse form
is **not** built here because the Newton solver consumes blocks directly.
Workspace buffers may be sized for a longer trial; only the first `tsteps` blocks are
written, which keeps this hot path safe for ragged-length fitting.
"""
function Hessian!(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    btd = sws.btd

    for i in 1:(tsteps - 1)
        copyto!(btd.H_sub[i], sws.H_sub_entry)
        copyto!(btd.H_super[i], sws.H_super_entry)
    end

    btd.H_diag[1] .= sws.yt_given_xt .+ sws.xt1_given_xt .+ sws.x_t
    for i in 2:(tsteps - 1)
        btd.H_diag[i] .= sws.yt_given_xt .+ sws.xt_given_xt_1 .+ sws.xt1_given_xt
    end
    btd.H_diag[tsteps] .= sws.yt_given_xt .+ sws.xt_given_xt_1

    return nothing
end

"""
    smooth(lds, y::AbstractMatrix)

Direct smoothing for a single trial.

# Arguments
- `lds::LinearDynamicalSystem`: The model.
- `y::AbstractMatrix`: Observations (obs_dim × tsteps).

# Returns
- `x_smooth::AbstractMatrix`: Smoothed latent means (latent_dim × tsteps).
- `p_smooth::Array{T,3}`: Smoothed latent covariances (latent_dim × latent_dim × tsteps).
"""
function smooth(lds::LinearDynamicalSystem, y::AbstractMatrix{T}) where {T}
    # Type assertion narrows the union for JET; runtime no-op since dispatch on
    # `size(y, 2)::Int` already lands in the FilterSmooth-returning method.
    fs = initialize_FilterSmooth(lds, size(y, 2))::FilterSmooth{T}
    sws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, size(y, 2))
    smooth!(lds, fs, y, sws)
    return fs.x_smooth, fs.p_smooth
end

function smooth(
    lds::LinearDynamicalSystem, y::AbstractVector{<:AbstractMatrix{T}}
) where {T}
    tsteps_per_trial = [size(yt, 2) for yt in y]
    T_max = maximum(tsteps_per_trial)
    tfs = initialize_FilterSmooth(lds, tsteps_per_trial)::TrialFilterSmooth{T}
    sws_pool = [
        SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, T_max) for
        _ in 1:Threads.maxthreadid()
    ]
    smooth!(lds, tfs, y, sws_pool)

    N = length(y)
    xs = Vector{Matrix{T}}(undef, N)
    Ps = Vector{Array{T,3}}(undef, N)
    for n in 1:N
        fs = tfs.FilterSmooths[n]
        xs[n] = copy(fs.x_smooth)
        Ps[n] = copy(fs.p_smooth)
    end
    return xs, Ps
end

"""
    smooth!(lds, fs, y, sws::SmoothWorkspace)

Low-allocation smoothing using `SmoothWorkspace`. Uses a direct single-step
Newton solver (since the Gaussian LDS has a quadratic log-likelihood),
exploiting the block tridiagonal structure of the Hessian for efficient solving.
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    y::AbstractMatrix{T},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps, D = size(y, 2), lds.latent_dim
    n_active = D * tsteps
    btd = sws.btd

    compute_smooth_constants!(sws, lds)

    # Workspace buffers may be sized for a longer trial; take active-length views.
    X0 = view(sws.X₀, 1:n_active)
    grad_vec = view(sws.grad_vec, 1:n_active)
    neg_diag_v = view(btd.neg_diag, 1:tsteps)
    neg_sub_v = view(btd.neg_sub, 1:(tsteps - 1))
    neg_super_v = view(btd.neg_super, 1:(tsteps - 1))

    # Warm-start from previous E[z].
    copyto!(X0, fs.E_z)

    x_mat = reshape(X0, D, tsteps)
    Gradient!(sws, lds, y, x_mat)
    # grad_vec = -gradient (minimize negative log-likelihood)
    for t in 1:tsteps, i in 1:D
        sws.grad_vec[(t - 1) * D + i] = -sws.grad_buf[i, t]
    end

    Hessian!(sws, lds, y, x_mat)
    _negate_blocks!(btd, tsteps)

    # Save x_old in fs.x_smooth before we overwrite sws.X₀ with the Newton step.
    fs.x_smooth .= x_mat

    block_tridiagonal_solve!(X0, neg_sub_v, neg_diag_v, neg_super_v, grad_vec, btd)

    step_mat = reshape(X0, D, tsteps)
    fs.x_smooth .-= step_mat

    logdet_precision = block_tridiagonal_inverse_logdet!(
        fs.p_smooth, fs.p_smooth_tt1, neg_sub_v, neg_diag_v, neg_super_v, btd
    )

    fs.entropy = gaussian_entropy_from_logdet(logdet_precision, n_active)

    @views for i in 1:tsteps
        Symmetrize!(fs.p_smooth[:, :, i])
    end

    return fs
end

"""
    smooth!(lds, tfs, y::AbstractVector{<:AbstractMatrix}, sws_pool)

Low-allocation multi-trial smoothing. Each task in `sws_pool` owns one workspace;
trials are partitioned across tasks via `@spawn` / `fetch` (see
https://julialang.org/blog/2023/07/PSA-dont-use-threadid/).

# Arguments
- `lds::LinearDynamicalSystem`
- `tfs::TrialFilterSmooth`: one `FilterSmooth` per trial, sized at each trial's length
- `y::AbstractVector{<:AbstractMatrix}`: one `obs_dim × T_i` matrix per trial
- `sws_pool::Vector{SmoothWorkspace{T}}`: task-local workspaces, each sized at
  `max(T_i)`
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    ntrials = length(y)

    if ntrials == 1
        smooth!(lds, tfs[1], y[1], sws_pool[1])
        return tfs
    end

    ntasks = min(ntrials, length(sws_pool))
    chunksize = cld(ntrials, ntasks)

    @sync for i in 1:ntasks
        lo = (i - 1) * chunksize + 1
        hi = min(i * chunksize, ntrials)
        lo > hi && continue
        @spawn begin
            sws = sws_pool[i]
            for trial in lo:hi
                smooth!(lds, tfs[trial], y[trial], sws)
            end
        end
    end

    return tfs
end

"""
    Q_state!(ws, lds, E_z, E_zz, E_zz_prev)

State Q-term for an LDS with affine dynamics x_t ~ N(A x_{t-1} + b, Q).
In-place version of `Q_state` that uses pre-allocated buffers from `SmoothWorkspace`.
Uses cached Cholesky factors from `compute_smooth_constants!`.
"""
function Q_state!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    tstep = size(E_z, 2)
    D = lds.latent_dim
    A = lds.state_model.A
    b = lds.state_model.b
    x0 = lds.state_model.x0

    # Use cached Cholesky factors (already computed by compute_smooth_constants!)
    # For Cholesky P0 = U'U, we need to solve P0 \ temp = inv(U) * inv(U') * temp
    Q_U = UpperTriangular(ws.Q_chol_U)
    P0_U = UpperTriangular(ws.P0_chol_U)

    # Compute log determinants from cached Cholesky factors
    log_det_Q = zero(T)
    log_det_P0 = zero(T)
    for j in 1:D
        log_det_Q += 2 * log(Q_U[j, j])
        log_det_P0 += 2 * log(P0_U[j, j])
    end

    # Use workspace buffers
    temp = ws.elbo_temp
    sum_E_zz = ws.elbo_sum_E_zz
    sum_E_zzm1 = ws.elbo_sum_E_zzm1
    sum_E_cross = ws.elbo_sum_E_cross
    sum_mu_t = ws.elbo_sum_mu_t
    sum_mu_tm1 = ws.elbo_sum_mu_tm1
    temp2 = ws.elbo_temp2

    # Initial-state part: temp = E_zz[:,:,1] - E_z[:,1]*x0' - x0*E_z[:,1]' + x0*x0'
    fill!(temp, zero(T))
    @views begin
        temp .+= E_zz[:, :, 1]
        mul!(temp, E_z[:, 1:1], x0', -one(T), one(T))  # temp -= E_z[:,1] * x0'
        mul!(temp, x0, E_z[:, 1:1]', -one(T), one(T))  # temp -= x0 * E_z[:,1]'
        mul!(temp, x0, x0', one(T), one(T))            # temp += x0 * x0'
    end

    # Solve P0 \ temp = inv(P0) * temp = inv(U) * inv(U') * temp
    # First: temp2 = inv(U') * temp (solve U' * temp2 = temp)
    ldiv!(P0_U', temp)  # temp = inv(U') * temp
    ldiv!(P0_U, temp)   # temp = inv(U) * temp = P0 \ original_temp
    Q_val = T(-0.5) * (log_det_P0 + tr(temp))

    # Transition part: accumulate sums over t=2:tstep
    fill!(sum_E_zz, zero(T))
    fill!(sum_E_zzm1, zero(T))
    fill!(sum_E_cross, zero(T))
    fill!(sum_mu_t, zero(T))
    fill!(sum_mu_tm1, zero(T))

    @views for t in 2:tstep
        sum_E_zz .+= E_zz[:, :, t]
        sum_E_zzm1 .+= E_zz[:, :, t - 1]
        sum_E_cross .+= E_zz_prev[:, :, t]
        sum_mu_t .+= E_z[:, t]
        sum_mu_tm1 .+= E_z[:, t - 1]
    end

    # Build temp = sum_E_zz - A*sum_E_cross' - sum_E_cross*A' + A*sum_E_zzm1*A'
    copyto!(temp, sum_E_zz)
    mul!(temp, A, sum_E_cross', -one(T), one(T))       # temp -= A * sum_E_cross'
    mul!(temp, sum_E_cross, A', -one(T), one(T))       # temp -= sum_E_cross * A'
    mul!(temp2, A, sum_E_zzm1)                          # temp2 = A * sum_E_zzm1
    mul!(temp, temp2, A', one(T), one(T))              # temp += temp2 * A'

    # Bias terms (use rank-1 updates to avoid allocations)
    # temp -= sum_mu_t * b'
    mul!(temp, sum_mu_t, b', -one(T), one(T))
    # temp -= b * sum_mu_t'
    mul!(temp, b, sum_mu_t', -one(T), one(T))
    # temp += A * sum_mu_tm1 * b' (use temp2 as intermediate)
    mul!(ws.tmp1, A, sum_mu_tm1)  # tmp1 = A * sum_mu_tm1
    mul!(temp, ws.tmp1, b', one(T), one(T))
    # temp += b * (A * sum_mu_tm1)'  = b * sum_mu_tm1' * A'
    mul!(temp, b, ws.tmp1', one(T), one(T))
    # temp += (tstep - 1) * b * b'
    mul!(temp, b, b', T(tstep - 1), one(T))

    # Solve Q \ temp = inv(Q) * temp
    ldiv!(Q_U', temp)
    ldiv!(Q_U, temp)
    Q_val += T(-0.5) * ((tstep - 1) * log_det_Q + tr(temp))

    return Q_val
end

"""
    Q_obs!(C, d, E_z, E_zz, y)

Single time-step observation component of the Q-function for
y_t ~ 𝓝(C x_t + d, R), before applying R^{-1} and constants.
"""
function Q_obs!(
    result::AbstractMatrix{T},
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    E_z::AbstractVector{T},
    E_zz::AbstractMatrix{T},
    y::AbstractVector{T},
    buffers,
) where {T<:Real}

    # Unpack buffers
    ytil, sum_yy, sum_yz, work1, work2 = buffers

    # Residualize: ytil = y - d (pre-allocated buffer)
    ytil .= y .- d

    # All operations use pre-allocated buffers
    mul!(sum_yy, ytil, ytil')

    # Efficient outer product: sum_yz = ytil * E_z'
    fill!(sum_yz, zero(T))
    BLAS.ger!(one(T), ytil, E_z, sum_yz)

    # Build result using buffers
    copyto!(result, sum_yy)
    mul!(result, C, sum_yz', -one(T), one(T))   # result -= C * sum_yz'
    mul!(work1, sum_yz, C')                      # work1 = sum_yz * C'  
    result .-= work1                             # result -= work1
    mul!(work2, E_zz, C')                        # work2 = E_zz * C'
    mul!(result, C, work2, one(T), one(T))       # result += C * work2

    return result
end

"""
    Q_obs!(ws, lds, E_z, E_zz, y)

Full observation Q-term for Gaussian LDS over all time steps. 
In-place version of `Q_obs` that uses pre-allocated buffers from `SmoothWorkspace`.
Uses cached Cholesky factors from `compute_smooth_constants!`.
"""
function Q_obs!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    obs_dim = lds.obs_dim
    tsteps = size(y, 2)
    C = lds.obs_model.C
    d = lds.obs_model.d

    # Use cached Cholesky factor
    R_U = UpperTriangular(ws.R_chol_U)

    # Compute log determinant from cached Cholesky
    log_det_R = zero(T)
    for j in 1:obs_dim
        log_det_R += 2 * log(R_U[j, j])
    end
    const_term = obs_dim * log(T(2π))

    # Use workspace buffers
    temp = ws.elbo_obs_temp
    work_matrix = ws.elbo_obs_work
    ytil = ws.elbo_ytil
    sum_yy = ws.elbo_sum_yy
    sum_yz = ws.elbo_sum_yz
    work1 = ws.elbo_obs_work1
    work2 = ws.elbo_obs_work2

    fill!(temp, zero(T))

    @views for t in axes(y, 2)
        # Residualize: ytil = y[:,t] - d
        ytil .= y[:, t] .- d

        # sum_yy = ytil * ytil'
        mul!(sum_yy, ytil, ytil')

        # sum_yz = ytil * E_z[:,t]' (outer product)
        fill!(sum_yz, zero(T))
        BLAS.ger!(one(T), ytil, E_z[:, t], sum_yz)

        # Build work_matrix
        copyto!(work_matrix, sum_yy)
        mul!(work_matrix, C, sum_yz', -one(T), one(T))   # work_matrix -= C * sum_yz'
        mul!(work1, sum_yz, C')                           # work1 = sum_yz * C'
        work_matrix .-= work1                             # work_matrix -= work1
        mul!(work2, E_zz[:, :, t], C')                    # work2 = E_zz * C'
        mul!(work_matrix, C, work2, one(T), one(T))       # work_matrix += C * work2

        # Accumulate
        temp .+= work_matrix
    end

    # Solve R \ temp = inv(R) * temp = inv(U) * inv(U') * temp
    ldiv!(R_U', temp)
    ldiv!(R_U, temp)
    return T(-0.5) * (tsteps * (const_term + log_det_R) + tr(temp))
end

"""
    calculate_elbo(lds, tfs, y, sws)

Low-allocation version of `calculate_elbo` using `SmoothWorkspace` buffers.
Uses cached Cholesky factors from `compute_smooth_constants!`.

Note: For single-trial case only. Multi-trial threading would require workspace pool.
"""
function calculate_elbo(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    ntrials = length(y)

    total_entropy = zero(T)
    for fs in tfs.FilterSmooths
        total_entropy += fs.entropy
    end

    ntasks = min(ntrials, length(sws_pool))
    partial = zeros(T, ntasks)
    chunksize = cld(ntrials, ntasks)

    @sync for i in 1:ntasks
        lo = (i - 1) * chunksize + 1
        hi = min(i * chunksize, ntrials)
        lo > hi && continue

        @spawn begin
            sws = sws_pool[i]
            acc = zero(T)
            for trial in lo:hi
                fs = tfs[trial]
                compute_smooth_constants!(sws, lds)
                acc +=
                    Q_state!(sws, lds, fs.E_z, fs.E_zz, fs.E_zz_prev) +
                    Q_obs!(sws, lds, fs.E_z, fs.E_zz, y[trial])
            end
            partial[i] = acc
        end
    end

    Q_total = sum(partial)

    prior_term = zero(T)
    if lds.state_model.Q_prior !== nothing
        prior_term += iw_logprior_term(lds.state_model.Q, lds.state_model.Q_prior)
    end
    if lds.state_model.P0_prior !== nothing
        prior_term += iw_logprior_term(lds.state_model.P0, lds.state_model.P0_prior)
    end
    if lds.obs_model.R_prior !== nothing
        prior_term += iw_logprior_term(lds.obs_model.R, lds.obs_model.R_prior)
    end

    return Q_total + prior_term + total_entropy
end

"""
    sufficient_statistics(x_smooth, p_smooth, p_smooth_t1)

Compute sufficient statistics for the EM algorithm in a Linear Dynamical System.

# Note
- The function computes the expected values for all trials.
- For single-trial data, use inputs with ntrials = 1.
"""
function sufficient_statistics!(fs::FilterSmooth{T}) where {T<:Real}
    latent_dim, tsteps = size(fs.x_smooth)

    # E_z is just a copy of x_smooth
    fs.E_z .= fs.x_smooth

    # Compute E_zz and E_zz_prev in-place
    @views for t in 1:tsteps
        # E_zz[:,:,t] = p_smooth[:,:,t] + x_smooth[:,t] * x_smooth[:,t]'
        mul!(fs.E_zz[:, :, t], fs.x_smooth[:, t:t], fs.x_smooth[:, t:t]')
        fs.E_zz[:, :, t] .+= fs.p_smooth[:, :, t]

        if t > 1
            # E_zz_prev[:,:,t] = p_smooth_tt1[:,:,t] + x_smooth[:,t] * x_smooth[:,t-1]'
            mul!(
                fs.E_zz_prev[:, :, t], fs.x_smooth[:, t:t], fs.x_smooth[:, (t - 1):(t - 1)]'
            )
            fs.E_zz_prev[:, :, t] .+= fs.p_smooth_tt1[:, :, t]
        else
            fs.E_zz_prev[:, :, 1] .= 0
        end
    end
end

function sufficient_statistics!(tfs::TrialFilterSmooth{T}) where {T<:Real}
    ntrials = length(tfs.FilterSmooths)

    if ntrials == 1
        sufficient_statistics!(tfs[1])
    else
        @threads for i in 1:ntrials
            sufficient_statistics!(tfs[i])
        end
    end
end

"""
    estep!(lds, tfs, y, sws_pool)

Perform the E-step of the EM algorithm for a Linear Dynamical System.
Uses `SmoothWorkspace` pool for low-allocation smoothing.

# Note
- This function first smooths the data using `smooth!`, then computes sufficient
    statistics.
- It treats all input as multi-trial, with single-trial being a special case where
    `ntrials = 1`.
"""
function estep!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    smooth!(lds, tfs, y, sws_pool)
    sufficient_statistics!(tfs)
    elbo = calculate_elbo(lds, tfs, y, sws_pool)
    return elbo
end

"""
    update_initial_state_mean!(
                        lds::LinearDynamicalSystem{T,S,O, 
                        tfs::TrialFilterSmooth,
                        w::Union{Nothing,AbstractVector{<:AbstractVector{T}}} = nothing
                    )

Update the initial state mean of the Linear Dynamical System using the average across all
trials.
"""
function update_initial_state_mean!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    if lds.fit_bool[1]
        ntrials = length(tfs.FilterSmooths)
        x0_new = zeros(T, lds.latent_dim)
        total_weight = zero(T)

        for trial in 1:ntrials
            fs = tfs[trial]
            weight = isnothing(w) ? one(T) : w[trial][1]  # Weight at t=1

            x0_new .+= weight .* fs.E_z[:, 1]
            total_weight += weight
        end

        lds.state_model.x0 .= x0_new ./ total_weight
    end
end

"""
    mstep!(lds, tfs, y, sws; w=nothing)

Perform the M-step of the EM algorithm for a Linear Dynamical System.
Uses `SmoothWorkspace` buffers for low-allocation parameter updates.
"""
function mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws::SmoothWorkspace{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    update_initial_state_mean!(lds, tfs, w)
    update_initial_state_covariance!(lds, tfs, sws, w)
    update_A_b!(lds, tfs, sws, w)
    update_Q!(lds, tfs, sws, w)
    update_C_d!(lds, tfs, y, sws, w)
    update_R!(lds, tfs, y, sws, w)
    return nothing
end

"""
    update_initial_state_covariance!(
        lds::LinearDynamicalSystem{T,S,O},
        tfs::TrialFilterSmooth{T},
        sws::SmoothWorkspace{T},
        w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing
    )

Update the initial state covariance of the Linear Dynamical System using the average
across all trials.
"""
function update_initial_state_covariance!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    sws::SmoothWorkspace{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[2] || return nothing

    D = lds.latent_dim
    S0_sum = sws.S0_sum
    fill!(S0_sum, zero(T))
    total_weight = zero(T)

    @views for trial in 1:length(tfs.FilterSmooths)
        fs = tfs[trial]
        wt = isnothing(w) ? one(T) : w[trial][1]
        S0_sum .+= wt .* (fs.E_zz[:, :, 1] - (lds.state_model.x0 * lds.state_model.x0'))
        total_weight += wt
    end

    if lds.state_model.P0_prior === nothing
        copyto!(S0_sum, S0_sum ./ total_weight)
    else
        Ψ, ν = lds.state_model.P0_prior.Ψ, lds.state_model.P0_prior.ν
        copyto!(S0_sum, iw_map(Ψ, ν, S0_sum, total_weight, D))
    end

    copyto!(lds.state_model.P0, Symmetrize!(S0_sum))
    return nothing
end

"""
    update_A_b!(
        lds::LinearDynamicalSystem{T,S,O},
        tfs::TrialFilterSmooth{T},
        sws::SmoothWorkspace{T},
        w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing
    )

Update the transition matrix A and bias b of the Linear Dynamical System.
"""
function update_A_b!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    sws::SmoothWorkspace{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[3] || return nothing
    D = lds.latent_dim
    ntrials = length(tfs)

    Sxz = sws.Sxz
    Szz = sws.Szz_Ab
    AB = sws.AB
    fill!(Sxz, zero(T))
    fill!(Szz, zero(T))

    for trial in 1:ntrials
        fs = tfs[trial]
        tsteps = size(fs.E_z, 2)
        weights = isnothing(w) ? nothing : w[trial]

        @views for t in 2:tsteps
            weight = isnothing(weights) ? one(T) : weights[t]

            Sxz[:, 1:D] .+= weight .* fs.E_zz_prev[:, :, t]
            Sxz[:, D + 1] .+= weight .* fs.E_z[:, t]

            Szz[1:D, 1:D] .+= weight .* fs.E_zz[:, :, t - 1]
            Szz[1:D, D + 1] .+= weight .* fs.E_z[:, t - 1]
            Szz[D + 1, 1:D] .+= weight .* fs.E_z[:, t - 1]
            Szz[D + 1, D + 1] += weight
        end
    end

    F = factorize(Szz')
    ldiv!(transpose(AB), F, transpose(Sxz)) # AB' = (Szz') \ (Sxz')

    copyto!(lds.state_model.A, view(AB, :, 1:D))
    copyto!(lds.state_model.b, view(AB, :, D+1))

    return nothing
end

"""
    update_Q!(
        lds::LinearDynamicalSystem{T,S,O},
        tfs::TrialFilterSmooth{T},
        sws::SmoothWorkspace{T},
        w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing
    )

Update the process noise covariance matrix Q of the Linear Dynamical System.
"""
function update_Q!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    sws::SmoothWorkspace{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[4] || return nothing
    ntrials = length(tfs)
    state_dim = lds.latent_dim
    A = lds.state_model.A
    b = lds.state_model.b

    Q_sum = sws.Q_sum
    fill!(Q_sum, zero(T))

    temp1 = sws.temp_Q1
    temp2 = sws.temp_Q2
    temp3 = sws.temp_Q3
    temp4 = sws.temp_Q4
    temp5 = sws.temp_Q5
    innovation_cov = sws.innovation_cov
    bbT = sws.bbT
    mul!(bbT, b, b')

    total_weight = zero(T)

    for trial in 1:ntrials
        fs = tfs[trial]
        tsteps = size(fs.E_zz, 3)
        weights = isnothing(w) ? nothing : w[trial]

        @views for t in 2:tsteps
            weight = isnothing(weights) ? one(T) : weights[t]

            Σt = fs.E_zz[:, :, t]
            Σtm1 = fs.E_zz[:, :, t - 1]
            Σcross = fs.E_zz_prev[:, :, t]
            μt = fs.E_z[:, t]
            μtm1 = fs.E_z[:, t - 1]

            mul!(temp1, Σcross, A')
            mul!(temp2, A, Σcross')
            mul!(temp3, A, Σtm1)
            mul!(temp4, temp3, A')

            @. innovation_cov = Σt - temp1 - temp2 + temp4

            mul!(temp5, A, μtm1)
            # innovation_cov += α * x * y'
            mul!(innovation_cov, reshape(μt, :, 1), reshape(b, 1, :), -one(T), one(T))  # -= μt*b'
            mul!(innovation_cov, reshape(b, :, 1), reshape(μt, 1, :), -one(T), one(T))  # -= b*μt'
            mul!(innovation_cov, reshape(temp5, :, 1), reshape(b, 1, :), one(T), one(T))  # += temp5*b'
            mul!(innovation_cov, reshape(b, :, 1), reshape(temp5, 1, :), one(T), one(T))  # += b*temp5'

            innovation_cov .+= bbT

            Q_sum .+= weight .* innovation_cov
            total_weight += weight
        end
    end

    if lds.state_model.Q_prior === nothing
        copyto!(Q_sum, Q_sum ./ total_weight)
    else
        Ψ, ν = lds.state_model.Q_prior.Ψ, lds.state_model.Q_prior.ν
        copyto!(Q_sum, iw_map(Ψ, ν, Q_sum, total_weight, state_dim))
    end

    copyto!(lds.state_model.Q, Symmetrize!(Q_sum))
    return nothing
end

"""
    update_C_d!(lds, tfs, y, sws, w=nothing)

Update the observation matrix `C` and bias `d`. Expects `y` as a vector of per-trial
observation matrices (each `obs_dim × T_i`); trial lengths may differ.
"""
function update_C_d!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws::SmoothWorkspace{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[5] || return nothing

    ntrials = length(tfs)
    D = lds.latent_dim

    Syz = sws.Syz
    Szz = sws.Szz_Cd
    CD = sws.CD

    fill!(Syz, zero(T))
    fill!(Szz, zero(T))

    work_yz = sws.work_yz
    work_outer = sws.work_outer

    for trial in 1:ntrials
        fs = tfs[trial]
        y_trial = y[trial]
        tsteps = size(y_trial, 2)
        weights = isnothing(w) ? nothing : w[trial]
        @views for t in 1:tsteps
            wt = isnothing(weights) ? one(T) : weights[t]

            μ = fs.E_z[:, t]
            Σ = fs.E_zz[:, :, t]
            yt = y_trial[:, t]

            fill!(work_yz, zero(T))
            BLAS.ger!(wt, yt, μ, work_yz)
            Syz[:, 1:D] .+= work_yz
            Syz[:, D + 1] .+= wt .* yt

            work_outer .= Σ
            work_outer .*= wt
            Szz[1:D, 1:D] .+= work_outer
            Szz[1:D, D + 1] .+= wt .* μ
            Szz[D + 1, 1:D] .+= wt .* μ
            Szz[D + 1, D + 1] += wt
        end
    end

    F = factorize(Szz)
    ldiv!(transpose(CD), F, transpose(Syz))

    copyto!(lds.obs_model.C, view(CD, :, 1:D))
    copyto!(lds.obs_model.d, view(CD, :, D + 1))

    return nothing
end

function update_R!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws::SmoothWorkspace{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[6] || return nothing

    obs_dim = lds.obs_dim
    ntrials = length(tfs)

    R_new = sws.R_sum
    fill!(R_new, zero(T))
    C = lds.obs_model.C
    d = lds.obs_model.d

    total_weight = zero(T)

    innovation = sws.innovation
    Czt = sws.Czt
    temp_matrix = sws.temp_R_matrix
    outer_product = sws.outer_product
    state_uncertainty = sws.state_uncertainty

    for trial in 1:ntrials
        fs = tfs[trial]
        y_trial = y[trial]
        tsteps = size(y_trial, 2)
        weights = isnothing(w) ? nothing : w[trial]

        @views for t in 1:tsteps
            wt = isnothing(weights) ? one(T) : weights[t]

            mul!(Czt, C, fs.E_z[:, t])
            @. innovation = y_trial[:, t] - (Czt + d)

            BLAS.ger!(wt, innovation, innovation, R_new)

            mul!(outer_product, fs.E_z[:, t], fs.E_z[:, t]')
            state_uncertainty .= fs.E_zz[:, :, t]
            state_uncertainty .-= outer_product

            mul!(temp_matrix, C, state_uncertainty)
            mul!(R_new, temp_matrix, C', wt, one(T))

            total_weight += wt
        end
    end

    if lds.obs_model.R_prior === nothing
        R_hat = R_new ./ total_weight
    else
        Ψ, ν = lds.obs_model.R_prior.Ψ, lds.obs_model.R_prior.ν
        R_hat = iw_map(Ψ, ν, R_new, total_weight, obs_dim)
    end

    copyto!(lds.obs_model.R, Symmetrize!(R_hat))
    return nothing
end

"""
    fit!(lds, y; max_iter=100, tol=1e-6, progress=true)

Fit a Gaussian Linear Dynamical System via Expectation-Maximization.

# Arguments
- `lds::LinearDynamicalSystem{T,S,O}`: model to fit in place
- `y`: observations. Two shapes accepted:
    * `AbstractMatrix{T}` of size `(obs_dim, T)` — single trial
    * `AbstractVector{<:AbstractMatrix{T}}` — multi-trial, each `(obs_dim, T_i)`,
      trial lengths may differ

# Keywords
- `max_iter::Int=100`: maximum EM iterations
- `tol::Float64=1e-6`: convergence tolerance on ELBO change
- `progress::Bool=true`: show progress bar

Returns a `Vector{T}` of ELBO values, one per iteration.
"""
function fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
    u0=nothing,
    u=nothing,
    d=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    return _fit!(lds, y, max_iter, tol, progress, u0, u, d, Val(lds.kalman_filter))
end

function _fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y_vec::AbstractVector{<:AbstractMatrix{T}},
    max_iter::Int,
    tol::Float64,
    progress::Bool,
    u0,
    u,
    d,
    ::Val{true},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    y_combined = zeros(T, size(y_vec[1], 1), size(y_vec[1], 2), length(y_vec))
    try
        # combine y vector into matrix
        y_combined = cat(y_vec...; dims=3)
    catch
        throw(
            ArgumentError(
                """
                Failed to combine input vector of matrices into a single matrix. 
                Ensure all matrices have the same number of rows (obs_dim) and that 
                the total number of columns does not exceed memory limits.
                """
            ),
        )
    end

    return _fit_kalman!(
        lds, y_combined; u0=u0, u=u, d=d, max_iter=max_iter, tol=tol, progress=progress
    )
end

function _fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}},
    max_iter::Int,
    tol::Float64,
    progress::Bool,
    u0,
    u,
    d,
    ::Val{false},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    return _fit_tridiag!(lds, y; max_iter=max_iter, tol=tol, progress=progress)
end

function _fit_tridiag!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps_per_trial = [size(yt, 2) for yt in y]
    T_max = maximum(tsteps_per_trial)

    prev_elbo = -T(Inf)
    elbos = Vector{T}()
    sizehint!(elbos, max_iter)

    tfs = initialize_FilterSmooth(lds, tsteps_per_trial)::TrialFilterSmooth{T}

    sws_pool = [
        SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, T_max) for
        _ in 1:Threads.maxthreadid()
    ]

    prog = if progress
        Progress(max_iter; desc="Fitting LDS via EM...", barlen=50, showspeed=true)
    else
        nothing
    end

    for _ in 1:max_iter
        elbo = estep!(lds, tfs, y, sws_pool)
        mstep!(lds, tfs, y, sws_pool[1])
        push!(elbos, elbo)

        prog !== nothing && next!(prog)

        if abs(elbo - prev_elbo) < tol
            prog !== nothing && finish!(prog)
            return elbos
        end
        prev_elbo = elbo
    end

    prog !== nothing && finish!(prog)
    return elbos
end

function fit!(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T}; kwargs...
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    # reshape y from [obs_dim, tsteps, trials] to vector of matrices if needed
    if ndims(y) == 3
        obs_dim, tsteps, ntrials = size(y)
        y_vec = [view(y,:,:,i) for i in 1:ntrials]
        return fit!(lds, y_vec; kwargs...)
    elseif ndims(y) == 2
        # single trial case, wrap in vector
        return fit!(lds, [y]; kwargs...)
    else
        throw(ArgumentError("Input array y must be 2D or 3D."))
    end
end

#=
Legacy wrappers for testing purposes only; use workspace-aware versions instead. Eventually remove.
=#
function update_initial_state_covariance!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    sws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, size(tfs[1].E_z, 2))
    return update_initial_state_covariance!(lds, tfs, sws, w)
end

function update_A_b!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    sws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, size(tfs[1].E_z, 2))
    return update_A_b!(lds, tfs, sws, w)
end

function update_Q!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    sws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, size(tfs[1].E_z, 2))
    return update_Q!(lds, tfs, sws, w)
end

function update_C_d!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    T_max = maximum(size(yt, 2) for yt in y)
    sws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, T_max)
    return update_C_d!(lds, tfs, y, sws, w)
end

function update_R!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    T_max = maximum(size(yt, 2) for yt in y)
    sws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, T_max)
    return update_R!(lds, tfs, y, sws, w)
end

function Gradient(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    ws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, tsteps)
    compute_smooth_constants!(ws, lds)
    return copy(Gradient!(ws, lds, y, x))
end

function Gradient(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    compute_smooth_constants!(ws, lds)
    return copy(Gradient!(ws, lds, y, x))
end

function Hessian(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    ws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, tsteps)
    compute_smooth_constants!(ws, lds)
    Hessian!(ws, lds, y, x)
    block_tridgm!(ws.btd)
    return copy(ws.btd.H_sparse)
end

function Hessian(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    compute_smooth_constants!(ws, lds)
    Hessian!(ws, lds, y, x)
    block_tridgm!(ws.btd)
    return copy(ws.btd.H_sparse)
end

function mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    T_max = maximum(size(yt, 2) for yt in y)
    sws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, T_max)
    return mstep!(lds, tfs, y, sws, w)
end

function estep!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    T_max = maximum(size(yt, 2) for yt in y)
    npool = Threads.maxthreadid()
    sws_pool = [SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, T_max) for _ in 1:npool]
    return estep!(lds, tfs, y, sws_pool)
end

function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    T_max = maximum(size(yt, 2) for yt in y)
    npool = Threads.maxthreadid()
    sws_pool = [SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, T_max) for _ in 1:npool]
    return smooth!(lds, tfs, y, sws_pool)
end

function loglikelihood(
    x::AbstractMatrix{XT}, lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{YT}
) where {T<:Real,YT<:Real,XT<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    WT = promote_type(T, YT, XT)
    ws = SmoothWorkspace(WT, lds.latent_dim, lds.obs_dim, tsteps)
    compute_smooth_constants!(ws, lds)
    return loglikelihood!(ws, x, lds, y)
end

"""
    filter_loglikelihood(lds, y)

One-step-ahead predictive log-likelihood ∑_{t,n} log p(y_t^n | y_{1:t-1}^n) via
the Kalman filter.  Valid for any fitted `LinearDynamicalSystem` with a
`GaussianObservationModel`, regardless of which E-step backend was used to train it.

Returns the **total** log-likelihood.  Divide by `obs_dim * tsteps * ntrials` for a
per-observation score that is comparable across configurations.
"""
function filter_loglikelihood(
    lds::LinearDynamicalSystem{T,SM,OM}, y::AbstractArray{T,3}
) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
    A = lds.state_model.A
    Q = lds.state_model.Q
    b = lds.state_model.b
    x0 = lds.state_model.x0
    P0 = lds.state_model.P0
    C = lds.obs_model.C
    R = lds.obs_model.R
    d = lds.obs_model.d

    obs_dim, tsteps, ntrials = size(y)
    D = lds.latent_dim

    total_ll = zero(T)
    log2πp = T(obs_dim) * log(T(2π))

    # Pre-allocate buffers (reused across trials and time steps)
    x_p = Vector{T}(undef, D)
    x_f = Vector{T}(undef, D)
    P_p = Matrix{T}(undef, D, D)
    P_f = Matrix{T}(undef, D, D)
    tmp_DD = Matrix{T}(undef, D, D)
    innov = Vector{T}(undef, obs_dim)
    Si_e = Vector{T}(undef, obs_dim)
    Smat = Matrix{T}(undef, obs_dim, obs_dim)
    PCt = Matrix{T}(undef, D, obs_dim)
    SiPCt = Matrix{T}(undef, obs_dim, D)

    for n in 1:ntrials
        x_f .= x0
        P_f .= P0

        for t in 1:tsteps
            # Prediction (t=1: prediction == prior)
            if t == 1
                x_p .= x0
                P_p .= P0
            else
                mul!(x_p, A, x_f)
                x_p .+= b
                mul!(tmp_DD, A, P_f)
                mul!(P_p, tmp_DD, A')
                P_p .+= Q
                Symmetrize!(P_p)
                # P_p .= (P_p .+ P_p') .* T(0.5)
            end

            # Innovation: e = y_t - C x_p - d
            mul!(innov, C, x_p)
            @views innov .= y[:, t, n] .- innov .- d

            # Innovation covariance: S = C P_p C' + R
            mul!(PCt, P_p, C')
            mul!(Smat, C, PCt)
            Smat .+= R
            Symmetrize!(Smat)
            # Smat .= (Smat .+ Smat') .* T(0.5)

            # One-step predictive log-likelihood: log N(e; 0, S)
            S_ch = cholesky(Hermitian(Smat))
            Si_e .= innov
            ldiv!(S_ch, Si_e)           # Si_e ← S⁻¹ e
            total_ll -= T(0.5) * (log2πp + logdet(S_ch) + dot(innov, Si_e))

            # Update: x_f = x_p + K e  where K = PCt S⁻¹
            mul!(x_f, PCt, Si_e)        # x_f = PCt (S⁻¹ e)
            x_f .+= x_p

            # P_f = P_p - PCt S⁻¹ PCt'
            SiPCt .= PCt'
            ldiv!(S_ch, SiPCt)          # SiPCt ← S⁻¹ PCt'
            mul!(P_f, PCt, SiPCt)
            P_f .= P_p .- P_f
            Symmetrize!(P_f)
            # P_f .= (P_f .+ P_f') .* T(0.5)
        end
    end

    return total_ll
end

function filter_loglikelihood(
    lds::LinearDynamicalSystem{T,SM,OM}, y::AbstractVector{<:AbstractMatrix{T}}
) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
    y_comb = cat(y...; dims=3)
    return filter_loglikelihood(lds, y_comb)
end
