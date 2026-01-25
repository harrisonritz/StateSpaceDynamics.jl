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
    initialize_FilterSmooth(model, num_obs) 
   
Initialize a `FilterSmooth` object for a given linear dynamical system model and number of observations.
"""
function initialize_FilterSmooth(
    model::LinearDynamicalSystem{T,S,O}, num_obs::Int
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    num_states = model.latent_dim
    return FilterSmooth{T}(
        zeros(T, num_states, num_obs),                    # x_smooth
        zeros(T, num_states, num_states, num_obs),        # p_smooth  
        zeros(T, num_states, num_states, num_obs),        # p_smooth_tt1
        zeros(T, num_states, num_obs),                    # E_z
        zeros(T, num_states, num_states, num_obs),        # E_zz
        zeros(T, num_states, num_states, num_obs),        # E_zz_prev
        zero(T),                                           # entropy
    )
end

function initialize_FilterSmooth(
    model::LinearDynamicalSystem{T,S,O}, tsteps::Int, ntrials::Int
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    filter_smooths = [initialize_FilterSmooth(model, tsteps) for _ in 1:ntrials]
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

function Random.rand(
    rng::AbstractRNG, lds::LinearDynamicalSystem{T,S,O}; tsteps::Int, ntrials::Int=1
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}

    # Extract parameters once using a more systematic approach
    state_params = _extract_state_params(lds.state_model)
    obs_params = _extract_obs_params(lds.obs_model)

    # Pre-allocate based on observation model type
    x = Array{T,3}(undef, lds.latent_dim, tsteps, ntrials)
    y = Array{T,3}(undef, lds.obs_dim, tsteps, ntrials)

    # Sample trials (potentially in parallel for large ntrials)
    if ntrials > 10  # Threshold for parallelization
        Threads.@threads for trial in 1:ntrials
            _sample_trial!(
                rng,
                view(x,:,:,trial),
                view(y,:,:,trial),
                state_params,
                obs_params,
                lds.obs_model,
            )
        end
    else
        for trial in 1:ntrials
            _sample_trial!(
                rng,
                view(x,:,:,trial),
                view(y,:,:,trial),
                state_params,
                obs_params,
                lds.obs_model,
            )
        end
    end

    return x, y
end

"""
    Random.rand(lds::LinearDynamicalSystem; tsteps::Int, ntrials::Int)
    Random.rand(rng::AbstractRNG, lds::LinearDynamicalSystem; tsteps::Int, ntrials::Int)

Sample from a Linear Dynamical System.
"""
function Random.rand(lds::LinearDynamicalSystem; kwargs...)
    return rand(Random.default_rng(), lds; kwargs...)
end

"""
    loglikelihood(
        x::AbstractMatrix{T},
        lds::LinearDynamicalSystem{S,O},
        y::AbstractMatrix{T}
    ) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Calculate the complete-data log-likelihood of a linear dynamical system (LDS) given the
observed data.

# Arguments
- `x::AbstractMatrix{T}`: The state sequence of the LDS.
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System.
- `y::AbstractMatrix{T}`: The observed data.

# Returns
- `ll::Vector{T}`: The complete-data log-likelihood of the LDS at each timestep.
"""
function loglikelihood(
    x::AbstractMatrix{U}, lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}
) where {U<:Real,T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    A, Q, x0, P0 = lds.state_model.A,
    lds.state_model.Q, lds.state_model.x0,
    lds.state_model.P0
    C, R, b, d = lds.obs_model.C, lds.obs_model.R, lds.state_model.b, lds.obs_model.d

    R_chol = cholesky(Symmetric(R)).U
    Q_chol = cholesky(Symmetric(Q)).U
    P0_chol = cholesky(Symmetric(P0)).U

    ll_vec = Vector{eltype(x)}(undef, tsteps)

    # Pre-allocate all temporary arrays
    temp_dx = zeros(eltype(x), size(x, 1))
    temp_dy = zeros(eltype(x), size(y, 1))
    temp_solve_Q = zeros(eltype(x), size(x, 1))
    temp_solve_R = zeros(eltype(x), size(y, 1))

    for t in 1:tsteps
        ll_t = zero(eltype(x))

        # Initial state contribution (only at t=1)
        if t == 1
            dx0 = view(x, :, 1) .- x0
            ll_t += sum(abs2, P0_chol \ dx0)
        end

        # Dynamics contribution (t > 1)
        if t > 1
            mul!(temp_dx, A, view(x, :, t-1), -one(eltype(x)), false)
            temp_dx .+= view(x, :, t) .- b
            ldiv!(temp_solve_Q, Q_chol, temp_dx)
            ll_t += sum(abs2, temp_solve_Q)
        end

        # Emission contribution
        mul!(temp_dy, C, view(x, :, t), -one(eltype(x)), false)
        temp_dy .+= view(y, :, t) .- d
        ldiv!(temp_solve_R, R_chol, temp_dy)
        ll_t += sum(abs2, temp_solve_R)

        ll_vec[t] = -eltype(x)(0.5) * ll_t
    end

    return ll_vec
end

"""
    Gradient(lds, y, x)

Compute the gradient of the log-likelihood with respect to the latent states for a linear
dynamical system.
"""
function Gradient(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    latent_dim, tsteps = size(x)
    obs_dim = size(y, 1)

    A, Q, x0, P0 = lds.state_model.A,
    lds.state_model.Q, lds.state_model.x0,
    lds.state_model.P0
    C, R, b, d = lds.obs_model.C, lds.obs_model.R, lds.state_model.b, lds.obs_model.d

    R_chol = cholesky(Symmetric(R))
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))

    C_inv_R = (R_chol \ C)'      # C' * inv(R)
    A_inv_Q = (Q_chol \ A)'      # A' * inv(Q)

    grad = zeros(T, latent_dim, tsteps)

    # Pre-allocate all temporary arrays for efficiency
    dxt = zeros(T, latent_dim)
    dxt_next = zeros(T, latent_dim)
    dyt = zeros(T, obs_dim)
    tmp1 = zeros(T, latent_dim)  # for C_inv_R * dyt
    tmp2 = zeros(T, latent_dim)  # for A_inv_Q * dxt_next
    tmp3 = zeros(T, latent_dim)  # for Q_chol \ dxt

    # First time step
    dxt .= x[:, 1] .- x0
    mul!(dxt_next, A, x[:, 1])
    dxt_next .= x[:, 2] .- dxt_next .- b
    mul!(dyt, C, x[:, 1])
    dyt .= y[:, 1] .- dyt .- d

    mul!(tmp1, C_inv_R, dyt)
    mul!(tmp2, A_inv_Q, dxt_next)
    ldiv!(tmp3, P0_chol, dxt)

    grad[:, 1] .= tmp1 .+ tmp2 .- tmp3

    # Middle steps
    @views for t in 2:(tsteps - 1)
        # dxt = x[:, t] - A * x[:, t-1] - b
        mul!(dxt, A, x[:, t - 1])
        dxt .= x[:, t] .- dxt .- b

        # dxt_next = x[:, t+1] - A * x[:, t] - b
        mul!(dxt_next, A, x[:, t])
        dxt_next .= x[:, t + 1] .- dxt_next .- b

        # dyt = y[:, t] - C * x[:, t] - d
        mul!(dyt, C, x[:, t])
        dyt .= y[:, t] .- dyt .- d

        # tmp1 = C_inv_R * dyt
        mul!(tmp1, C_inv_R, dyt)

        # tmp2 = A_inv_Q * dxt_next
        mul!(tmp2, A_inv_Q, dxt_next)

        # tmp3 = Q_chol \ dxt
        ldiv!(tmp3, Q_chol, dxt)

        # grad[:, t] = tmp1 - tmp3 + tmp2
        grad[:, t] .= tmp1 .- tmp3 .+ tmp2
    end

    # Last time step
    mul!(dxt, A, x[:, tsteps - 1])
    dxt .= x[:, tsteps] .- dxt .- b
    mul!(dyt, C, x[:, tsteps])
    dyt .= y[:, tsteps] .- dyt .- d

    mul!(tmp1, C_inv_R, dyt)
    ldiv!(tmp3, Q_chol, dxt)

    grad[:, tsteps] .= tmp1 .- tmp3

    return grad
end

"""
    loglikelihood!(ws, x, lds, y)

In-place version of `loglikelihood` that uses pre-computed Cholesky factors from
`ws::SmoothWorkspace` and writes into `ws.ll_vec`. Returns the sum of log-likelihoods.
"""
function loglikelihood!(
    ws::SmoothWorkspace{T},
    x::AbstractMatrix,
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
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

    total_ll = zero(T)

    for t in 1:tsteps
        ll_t = zero(T)

        # Initial state contribution (only at t=1)
        if t == 1
            @views temp_dx .= x[:, 1] .- x0
            ldiv!(temp_solve_Q, P0_U, temp_dx)
            ll_t += sum(abs2, temp_solve_Q)
        end

        # Dynamics contribution (t > 1)
        if t > 1
            @views mul!(temp_dx, A, x[:, t-1])
            @views temp_dx .= x[:, t] .- temp_dx .- b
            ldiv!(temp_solve_Q, Q_U, temp_dx)
            ll_t += sum(abs2, temp_solve_Q)
        end

        # Emission contribution
        @views mul!(temp_dy, lds.obs_model.C, x[:, t])
        @views temp_dy .= y[:, t] .- temp_dy .- d
        ldiv!(temp_solve_R, R_U, temp_dy)
        ll_t += sum(abs2, temp_solve_R)

        ll_vec[t] = -T(0.5) * ll_t
        total_ll += ll_vec[t]
    end

    return total_ll
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
    # Wait: original is grad = C'R^{-1}dyt + A'Q^{-1}dxt_next - P0^{-1}dxt
    # P0^{-1}*dxt = -neg_P0_inv * dxt, and neg_P0_inv*dxt gives -P0^{-1}*dxt
    # So: grad = tmp1 + tmp2 - P0^{-1}*dxt = tmp1 + tmp2 + neg_P0_inv*dxt = tmp1 + tmp2 + tmp3
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
    Hessian(lds, y, x) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Construct the Hessian matrix of the log-likelihood of the LDS model given a set of
observations.

This function is used for the direct optimization of the log-likelihood as advocated by
Paninski et al. (2009). The block tridiagonal structure of the Hessian is exploited to
reduce the number of parameters that need to be computed, and to reduce the memory
requirements. Together with the gradient, this allows for Kalman Smoothing to be performed
by simply solving a linear system of equations:

    ̂xₙ₊₁ = ̂xₙ - H \\ ∇

where ` ̂xₙ` is the current smoothed state estimate, `H` is the Hessian matrix, and `∇` is the
gradient of the log-likelihood.

# Note
- `x` is not used in this function, but is required to match the function signature of other
    Hessian calculations e.g., in PoissonLDS.
"""
function Hessian(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    A, Q, x0, P0 = lds.state_model.A,
    lds.state_model.Q, lds.state_model.x0,
    lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R

    tsteps = size(y, 2)
    state_dim = size(A, 1)

    # Pre-compute Cholesky factorizations
    R_chol = cholesky(Symmetric(R))
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))

    # Pre-allocate all blocks
    H_sub = Vector{Matrix{T}}(undef, tsteps - 1)
    H_super = Vector{Matrix{T}}(undef, tsteps - 1)
    H_diag = Vector{Matrix{T}}(undef, tsteps)

    # Off-diagonal terms
    H_sub_entry = Q_chol \ A
    H_super_entry = Matrix(H_sub_entry')

    # Calculate main diagonal terms
    I_mat = Matrix{T}(I, state_dim, state_dim)
    yt_given_xt = -C' * (R_chol \ C)
    xt_given_xt_1 = -(Q_chol \ I_mat)
    xt1_given_xt = -A' * (Q_chol \ A)
    x_t = -(P0_chol \ I_mat)

    # Build off-diagonals
    for i in 1:(tsteps - 1)
        H_sub[i] = H_sub_entry
        H_super[i] = H_super_entry
    end

    # Build main diagonal
    H_diag[1] = yt_given_xt + xt1_given_xt + x_t

    for i in 2:(tsteps - 1)
        H_diag[i] = yt_given_xt + xt_given_xt_1 + xt1_given_xt
    end

    H_diag[tsteps] = yt_given_xt + xt_given_xt_1
    H = block_tridgm(H_diag, H_super, H_sub)

    return H, H_diag, H_super, H_sub
end

"""
    Hessian!(ws, lds, y, x)

In-place version of `Hessian` that writes blocks into the workspace and
updates the sparse matrix values. Returns the workspace's sparse matrix.
"""
function Hessian!(
    ws::BlockTridiagonalWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    A_mat, Q, x0, P0 = lds.state_model.A,
        lds.state_model.Q, lds.state_model.x0,
        lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R

    tsteps = size(y, 2)
    state_dim = size(A_mat, 1)

    # Pre-compute Cholesky factorizations
    R_chol = cholesky(Symmetric(R))
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))

    # Compute reusable terms
    H_sub_entry = Q_chol \ A_mat
    H_super_entry = Matrix(H_sub_entry')

    I_mat = Matrix{T}(I, state_dim, state_dim)
    yt_given_xt = -C' * (R_chol \ C)
    xt_given_xt_1 = -(Q_chol \ I_mat)
    xt1_given_xt = -A_mat' * (Q_chol \ A_mat)
    x_t = -(P0_chol \ I_mat)

    # Fill blocks in-place
    for i in 1:(tsteps - 1)
        copyto!(ws.H_sub[i], H_sub_entry)
        copyto!(ws.H_super[i], H_super_entry)
    end

    # Main diagonal
    ws.H_diag[1] .= yt_given_xt .+ xt1_given_xt .+ x_t
    for i in 2:(tsteps - 1)
        ws.H_diag[i] .= yt_given_xt .+ xt_given_xt_1 .+ xt1_given_xt
    end
    ws.H_diag[tsteps] .= yt_given_xt .+ xt_given_xt_1

    # Update sparse matrix in-place
    block_tridgm!(ws)

    return ws.H_sparse
end

"""
    Hessian!(sws, lds, y, x)

In-place Hessian using pre-computed block templates from `SmoothWorkspace`.
Avoids all Cholesky and matrix solve allocations by using cached terms.
"""
function Hessian!(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    btd = sws.btd

    # Fill blocks from pre-computed templates (no computation needed)
    for i in 1:(tsteps - 1)
        copyto!(btd.H_sub[i], sws.H_sub_entry)
        copyto!(btd.H_super[i], sws.H_super_entry)
    end

    # Main diagonal from cached templates
    btd.H_diag[1] .= sws.yt_given_xt .+ sws.xt1_given_xt .+ sws.x_t
    for i in 2:(tsteps - 1)
        btd.H_diag[i] .= sws.yt_given_xt .+ sws.xt_given_xt_1 .+ sws.xt1_given_xt
    end
    btd.H_diag[tsteps] .= sws.yt_given_xt .+ sws.xt_given_xt_1

    # Update sparse matrix in-place
    block_tridgm!(btd)

    return btd.H_sparse
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
    fs = initialize_FilterSmooth(lds, size(y, 2))
    smooth!(lds, fs, y)
    return fs.x_smooth, fs.p_smooth
end

function smooth(lds::LinearDynamicalSystem, y::AbstractArray{T,3}) where {T}
    tfs = initialize_FilterSmooth(lds, size(y, 2), size(y, 3))
    smooth!(lds, tfs, y)

    D = lds.latent_dim
    Tt = size(y, 2)
    N = size(y, 3)

    xs = Array{T,3}(undef, D, Tt, N)
    Ps = Array{T,4}(undef, D, D, Tt, N)

    for n in 1:N
        fs = tfs.FilterSmooths[n]
        xs[:, :, n] .= fs.x_smooth
        Ps[:, :, :, n] .= fs.p_smooth
    end
    return xs, Ps
end

function smooth!(
    lds::LinearDynamicalSystem{T,S,O}, fs::FilterSmooth{T}, y::AbstractMatrix{T},
    ws::Union{Nothing,BlockTridiagonalWorkspace{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    tsteps, D = size(y, 2), lds.latent_dim

    # use old fs if it exists, by default is zeros if no iteration of EM has occurred
    X₀ = Vector{T}(vec(fs.E_z))

    function nll(vec_x::AbstractVector{T})
        x = reshape(vec_x, D, tsteps)
        return -sum(loglikelihood(x, lds, y))
    end

    function g!(g::Vector{T}, vec_x::Vector{T})
        x = reshape(vec_x, D, tsteps)
        grad = Gradient(lds, y, x)
        return g .= vec(-grad)
    end

    # Use workspace-aware Hessian if available
    function h!(h::SparseMatrixCSC{T}, vec_x::Vector{T}) where {T<:Real}
        x = reshape(vec_x, D, tsteps)
        if ws !== nothing
            Hessian!(ws, lds, y, x)
            # Same sparsity pattern: directly negate nzval
            h.nzval .= .-ws.H_sparse.nzval
        else
            H, _, _, _ = Hessian(lds, y, x)
            mul!(h, -1.0, H)
        end
        return nothing
    end

    # Initial values setup
    initial_f = nll(X₀)
    inital_g = similar(X₀)
    g!(inital_g, X₀)

    # Use workspace sparse pattern if available
    initial_h = if ws !== nothing
        copy(ws.H_sparse)
    else
        spzeros(T, length(X₀), length(X₀))
    end
    h!(initial_h, X₀)

    td = TwiceDifferentiable(nll, g!, h!, X₀, initial_f, inital_g, initial_h)
    opts = Optim.Options(; g_abstol=1e-8, x_abstol=1e-8, f_abstol=1e-8, iterations=100)

    res = optimize(td, X₀, Newton(; linesearch=LineSearches.BackTracking()), opts)

    fs.x_smooth .= reshape(res.minimizer, D, tsteps)

    # Get the second moments of the latent state path
    if ws !== nothing && lds.latent_dim > 10
        # In-place path: use workspace for Hessian and inverse
        Hessian!(ws, lds, y, fs.x_smooth)
        _negate_blocks!(ws)
        block_tridiagonal_inverse!(
            fs.p_smooth, fs.p_smooth_tt1,
            ws.neg_sub, ws.neg_diag, ws.neg_super, ws,
        )
        fs.entropy = gaussian_entropy(Symmetric(ws.H_sparse))
    elseif lds.latent_dim > 10
        H, main, super, sub = Hessian(lds, y, fs.x_smooth)
        p_smooth_result, p_smooth_tt1_result = block_tridiagonal_inverse(
            -sub, -main, -super
        )
        fs.p_smooth .= p_smooth_result
        fs.p_smooth_tt1[:, :, 2:end] .= p_smooth_tt1_result
        fs.entropy = gaussian_entropy(Symmetric(H))
    else
        H, main, super, sub = Hessian(lds, y, fs.x_smooth)
        p_smooth_result, p_smooth_tt1_result = block_tridiagonal_inverse_static(
            -sub, -main, -super, Val(lds.latent_dim)
        )
        fs.p_smooth .= p_smooth_result
        fs.p_smooth_tt1[:, :, 2:end] .= p_smooth_tt1_result
        fs.entropy = gaussian_entropy(Symmetric(H))
    end

    # Symmetrize
    @views for i in 1:tsteps
        fs.p_smooth[:, :, i] .= 0.5 .* (fs.p_smooth[:, :, i] .+ fs.p_smooth[:, :, i]')
    end

    return fs
end

"""
    smooth!(lds, fs, y, sws::SmoothWorkspace)

Low-allocation smoothing using `SmoothWorkspace`. Uses a direct single-step
Newton solver (since the Gaussian LDS has a quadratic log-likelihood),
exploiting the block tridiagonal structure of the Hessian for efficient solving.
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O}, fs::FilterSmooth{T}, y::AbstractMatrix{T},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    tsteps, D = size(y, 2), lds.latent_dim
    btd = sws.btd

    # Pre-compute all constant terms once for this smooth! call
    compute_smooth_constants!(sws, lds)

    # Initialize X₀ from previous E[z] (warm start)
    copyto!(sws.X₀, 1, fs.E_z, 1, length(sws.X₀))

    # Compute gradient at X₀ (negated for minimization: we minimize -loglik)
    x_mat = reshape(sws.X₀, D, tsteps)
    Gradient!(sws, lds, y, x_mat)
    # grad_vec = -gradient (for minimization of negative log-likelihood)
    copyto!(sws.grad_vec, 1, sws.grad_buf, 1, length(sws.grad_vec))
    sws.grad_vec .*= -one(T)

    # Compute Hessian (constant for Gaussian LDS, only depends on parameters)
    Hessian!(sws, lds, y, x_mat)
    _negate_blocks!(btd)

    # Direct Newton step for minimizing f(x) = -loglik(x)
    # Newton update: x_new = x_old - H_f⁻¹ * ∇f
    # where H_f = -H_loglik (Hessian of negative loglik), ∇f = -∇loglik
    #
    # We have: neg_diag etc = -H_loglik (which is SPD since H_loglik is negative definite)
    # grad_vec = -Gradient!(sws, ...) = -∇loglik = ∇f
    #
    # So we solve: (-H_loglik) * step = ∇f = -∇loglik
    # => step = (-H_loglik)⁻¹ * (-∇loglik)
    # => x_new = x_old - step

    # Save x_old in fs.x_smooth before overwriting sws.X₀
    fs.x_smooth .= x_mat

    # Solve for Newton step using block tridiagonal structure
    # sws.X₀ will be overwritten with the step
    block_tridiagonal_solve!(
        sws.X₀,
        btd.neg_sub,
        btd.neg_diag,
        btd.neg_super,
        sws.grad_vec,
        btd
    )

    # x_new = x_old - step
    step_mat = reshape(sws.X₀, D, tsteps)
    fs.x_smooth .-= step_mat

    # Get the second moments using the block structure
    # The Hessian blocks are already in btd.neg_* (as -H_loglik = precision matrix)
    logdet_precision = block_tridiagonal_inverse_logdet!(
        fs.p_smooth, fs.p_smooth_tt1,
        btd.neg_sub, btd.neg_diag, btd.neg_super, btd,
    )
    # Compute entropy from logdet
    n = D * tsteps
    fs.entropy = gaussian_entropy_from_logdet(logdet_precision, n)

    # Symmetrize covariances
    @views for i in 1:tsteps
        fs.p_smooth[:, :, i] .= T(0.5) .* (fs.p_smooth[:, :, i] .+ fs.p_smooth[:, :, i]')
    end

    return fs
end

"""
    smooth!(lds, tfs, y::AbstractArray{T,3})

Direct smoothing for multiple trials.

# Arguments
- `lds::LinearDynamicalSystem`: The model.
- `tfs::TrialFilterSmooth`: Preallocated container (one per trial).
- `y::Array{T,3}`: Observations (obs_dim × tsteps × ntrials).

# Side effects
- Fills each `FilterSmooth` in `tfs` with `x_smooth`, `p_smooth`, `p_smooth_tt1`, `E_z`, `E_zz`, `E_zz_prev`, `entropy`.

# Returns
- `tfs`: The same `TrialFilterSmooth`, populated.
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3},
    ws_pool::Union{Nothing,Vector{BlockTridiagonalWorkspace{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    ntrials = size(y, 3)

    if ntrials == 1
        ws = ws_pool !== nothing ? ws_pool[1] : nothing
        smooth!(lds, tfs[1], y[:, :, 1], ws)
    else
        @views @threads for trial in 1:ntrials
            ws = ws_pool !== nothing ? ws_pool[Threads.threadid()] : nothing
            smooth!(lds, tfs[trial], y[:, :, trial], ws)
        end
    end

    return tfs
end

function smooth!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    ntrials = size(y, 3)

    if ntrials == 1
        @views smooth!(lds, tfs[1], y[:, :, 1], sws_pool[1])
    else
        @views @threads for trial in 1:ntrials
            sws = sws_pool[Threads.threadid()]
            smooth!(lds, tfs[trial], y[:, :, trial], sws)
        end
    end

    return tfs
end

"""
    Q_state(A, b, Q, P0, x0, E_z, E_zz, E_zz_prev)

State Q-term for an LDS with affine dynamics x_t ~ N(A x_{t-1} + b, Q).
Matches the style of `Q_state` but includes the bias contributions.
"""
function Q_state(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    P0::AbstractMatrix{T},
    x0::AbstractVector{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
) where {T<:Real}
    tstep = size(E_z, 2)
    D = size(A, 1)

    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))
    log_det_Q = logdet(Q_chol)
    log_det_P0 = logdet(P0_chol)

    # initial-state part (unchanged)
    temp = zeros(T, D, D)
    mul!(temp, E_z[:, 1], x0', T(-1), T(0))
    temp .+= @view E_zz[:, :, 1]
    temp .-= x0 * E_z[:, 1]'
    temp .+= x0 * x0'
    Q_val = T(-0.5) * (log_det_P0 + tr(P0_chol \ temp))

    # transition part with bias
    sum_E_zz = zeros(T, D, D)
    sum_E_zzm1 = zeros(T, D, D)
    sum_E_cross = zeros(T, D, D)
    sum_mu_t = zeros(T, D)
    sum_mu_tm1 = zeros(T, D)

    for t in 2:tstep
        sum_E_zz .+= @view E_zz[:, :, t]
        sum_E_zzm1 .+= @view E_zz[:, :, t - 1]
        sum_E_cross .+= @view E_zz_prev[:, :, t]
        sum_mu_t .+= @view E_z[:, t]
        sum_mu_tm1 .+= @view E_z[:, t - 1]
    end

    copyto!(temp, sum_E_zz)
    mul!(temp, A, sum_E_cross', T(-1), T(1))
    temp .-= sum_E_cross * A'
    mul!(temp, A, sum_E_zzm1 * A', T(1), T(1))
    # bias terms
    temp .-= sum_mu_t * b'
    temp .-= b * sum_mu_t'
    temp .+= A * (sum_mu_tm1 * b')    # A μ_{t-1} bᵀ
    temp .+= (b * sum_mu_tm1') * A'   # b μ_{t-1}ᵀ Aᵀ
    temp .+= (tstep - 1) * (b * b')

    Q_val += T(-0.5) * ((tstep - 1) * log_det_Q + tr(Q_chol \ temp))
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
    Q_obs(C, d, R, E_z, E_zz, y)
Full observation Q-term for Gaussian LDS over all time steps.
"""
function Q_obs(
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    R::AbstractMatrix{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real}
    obs_dim = size(C, 1)
    latent_dim = size(E_z, 1)
    tsteps = size(y, 2)

    # Pre-compute constants
    R_chol = cholesky(Symmetric(R))
    log_det_R = logdet(R_chol)
    const_term = obs_dim * log(2π)

    # Pre-allocate ALL buffers once (reuse across all timesteps!)
    temp = zeros(T, obs_dim, obs_dim)
    work_matrix = zeros(T, obs_dim, obs_dim)

    # Buffers for the lower-level Q_obs! (including ytil for bias)
    buffers = (
        ytil=zeros(T, obs_dim),
        sum_yy=zeros(T, obs_dim, obs_dim),
        sum_yz=zeros(T, obs_dim, latent_dim),
        work1=zeros(T, obs_dim, obs_dim),
        work2=zeros(T, latent_dim, obs_dim),
    )

    # Use views in the loop - now with buffer passing
    @views for t in axes(y, 2)
        # Pass buffers to lower-level function (with bias d)
        Q_obs!(work_matrix, C, d, E_z[:, t], E_zz[:, :, t], y[:, t], buffers)

        # Accumulate in-place
        temp .+= work_matrix
    end

    return T(-0.5) * (tsteps * (const_term + log_det_R) + tr(R_chol \ temp))
end

"""
    Q_function(A, b, Q, C, d, R, P0, x0, E_z, E_zz, E_zz_prev, y)

Complete Q-function for Gaussian LDS:
x_t ~ 𝓝(A x_{t-1} + b, Q),  y_t ~ 𝓝(C x_t + d, R).
"""
function Q_function(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    R::AbstractMatrix{T},
    P0::AbstractMatrix{T},
    x0::AbstractVector{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real}
    Q_val_state = Q_state(A, b, Q, P0, x0, E_z, E_zz, E_zz_prev)
    Q_val_obs = Q_obs(C, d, R, E_z, E_zz, y)
    return Q_val_state + Q_val_obs
end

"""
    calculate_elbo(lds, E_z, E_zz, E_zz_prev, p_smooth, y, total_entropy)

Calculate the Evidence Lower Bound (ELBO) for a Linear Dynamical System. 
Adds constant-free IW log-prior terms for `Q` and `P0` when priors are set, 
so the ELBO tracks the MAP objective.
"""
function calculate_elbo(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    ntrials = size(y, 3)
    Q_vals = zeros(T, ntrials)

    # Calculate total entropy from individual FilterSmooth objects
    total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths)

    # Thread over trials
    @threads for trial in 1:ntrials
        fs = tfs[trial]  # Get the FilterSmooth for this trial
        Q_vals[trial] = Q_function(
            lds.state_model.A,
            lds.state_model.b,
            lds.state_model.Q,
            lds.obs_model.C,
            lds.obs_model.d,
            lds.obs_model.R,
            lds.state_model.P0,
            lds.state_model.x0,
            fs.E_z,
            fs.E_zz,
            fs.E_zz_prev,
            view(y,:,:,trial),
        )
    end

    # prior terms (included once)
    prior_term = zero(T)
    if lds.state_model.Q_prior !== nothing
        prior_term += iw_logprior_term(lds.state_model.Q, lds.state_model.Q_prior)
    end
    if lds.state_model.P0_prior !== nothing
        prior_term += iw_logprior_term(lds.state_model.P0, lds.state_model.P0_prior)
    end
    if (lds.obs_model isa GaussianObservationModel) && (lds.obs_model.R_prior !== nothing)
        prior_term += iw_logprior_term(lds.obs_model.R, lds.obs_model.R_prior)
    end

    return sum(Q_vals) + prior_term - total_entropy
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
    estep(lds::LinearDynamicalSystem{T,S,O},tfs::TrialFilterSmooth, y::AbstractArray{T,3})

Perform the E-step of the EM algorithm for a Linear Dynamical System, treating all input as
multi-trial.

# Note
- This function first smooths the data using the `smooth` function, then computes sufficient
    statistics.
- It treats all input as multi-trial, with single-trial being a special case where
    `ntrials = 1`.
"""
function estep!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth, y::AbstractArray{T,3},
    ws_pool::Union{Nothing,Vector{BlockTridiagonalWorkspace{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    smooth!(lds, tfs, y, ws_pool)
    sufficient_statistics!(tfs)
    elbo = calculate_elbo(lds, tfs, y)
    return elbo
end

function estep!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    smooth!(lds, tfs, y, sws_pool)
    sufficient_statistics!(tfs)
    elbo = calculate_elbo(lds, tfs, y)
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
    update_initial_state_covariance!(
        lds::LinearDynamicalSystem{T,S,O},
        E_z::AbstractArray{T,3},
        E_zz::AbstractArray{T,4}
    )

Update the initial state covariance of the Linear Dynamical System using the average across
all trials.
"""
function update_initial_state_covariance!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[2] || return nothing

    D = lds.latent_dim
    S0_sum = zeros(T, D, D)
    total_weight = zero(T)

    for trial in 1:length(tfs.FilterSmooths)
        fs = tfs[trial]
        wt = isnothing(w) ? one(T) : w[trial][1]  # weight at t=1
        S0_sum .+= wt .* (fs.E_zz[:, :, 1] - (lds.state_model.x0 * lds.state_model.x0'))
        total_weight += wt
    end

    P0_hat = if lds.state_model.P0_prior === nothing
        S0_sum ./ total_weight
    else
        Ψ, ν = lds.state_model.P0_prior.Ψ, lds.state_model.P0_prior.ν
        iw_map(Ψ, ν, S0_sum, total_weight, D)
    end

    P0_hat .= 0.5 .* (P0_hat .+ P0_hat')  # symmetrize
    lds.state_model.P0 = P0_hat
    return nothing
end

"""
    update_A!(
        lds::LinearDynamicalSystem{T,S,O},
        E_zz::AbstractArray{T,4},
        E_zz_prev::AbstractArray{T,4}
    )

Update the transition matrix A of the Linear Dynamical System.

"""
function update_A_b!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[3] || return nothing
    D = lds.latent_dim
    ntrials = length(tfs)

    # Accumulate statistics for [A b] jointly
    Sxz = zeros(T, D, D + 1)
    Szz = zeros(T, D + 1, D + 1)

    for trial in 1:ntrials
        fs = tfs[trial]
        tsteps = size(fs.E_z, 2)
        weights = isnothing(w) ? nothing : w[trial]

        @views for t in 2:tsteps
            weight = isnothing(weights) ? one(T) : weights[t]

            # Sxz accumulation
            Sxz[:, 1:D] .+= weight .* fs.E_zz_prev[:, :, t]
            Sxz[:, D + 1] .+= weight .* fs.E_z[:, t]

            # Szz for augmented state z_{t-1} = [x_{t-1}; 1]
            Szz[1:D, 1:D] .+= weight .* fs.E_zz[:, :, t - 1]
            Szz[1:D, D + 1] .+= weight .* fs.E_z[:, t - 1]
            Szz[D + 1, 1:D] .+= weight .* fs.E_z[:, t - 1]
            Szz[D + 1, D + 1] += weight
        end
    end

    # Solve jointly: [A b] = Sxz / Szz
    AB = Sxz / Szz
    lds.state_model.A = AB[:, 1:D]
    lds.state_model.b = AB[:, D + 1]

    return nothing
end

"""
    update_Q!(
        lds::LinearDynamicalSystem{T,S,O},
        tfs::TrialFilterSmooth,
        w::Union{Nothing,AbstractVector{<:AbstractVector{T}}} = nothing
    )

Update the process noise covariance matrix Q of the Linear Dynamical System.

"""
function update_Q!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[4] || return nothing
    ntrials = length(tfs)
    state_dim = lds.latent_dim
    A = lds.state_model.A
    b = lds.state_model.b
    Q_sum = zeros(T, state_dim, state_dim)

    # Pre-allocate working matrices
    temp1 = Matrix{T}(undef, state_dim, state_dim)
    temp2 = Matrix{T}(undef, state_dim, state_dim)
    temp3 = Matrix{T}(undef, state_dim, state_dim)
    temp4 = Matrix{T}(undef, state_dim, state_dim)
    temp5 = Vector{T}(undef, state_dim)
    innovation_cov = Matrix{T}(undef, state_dim, state_dim)

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

            # Compute using pre-allocated temps
            mul!(temp1, Σcross, A')
            mul!(temp2, A, Σcross')
            mul!(temp3, A, Σtm1)
            mul!(temp4, temp3, A')

            @. innovation_cov = Σt - temp1 - temp2 + temp4

            # Add bias terms using in-place rank-1 updates (no temporaries)
            mul!(temp5, A, μtm1)
            mul!(innovation_cov, μt, b', -one(T), one(T))
            mul!(innovation_cov, b, μt', -one(T), one(T))
            mul!(innovation_cov, temp5, b', one(T), one(T))
            mul!(innovation_cov, b, temp5', one(T), one(T))
            mul!(innovation_cov, b, b', one(T), one(T))

            Q_sum .+= weight .* innovation_cov
            total_weight += weight
        end
    end

    if lds.state_model.Q_prior === nothing
        Q_hat = Q_sum ./ total_weight
    else
        Ψ, ν = lds.state_model.Q_prior.Ψ, lds.state_model.Q_prior.ν
        Q_hat = iw_map(Ψ, ν, Q_sum, total_weight, state_dim)
    end

    Q_hat .= 0.5 .* (Q_hat .+ Q_hat')   # symmetrize
    lds.state_model.Q = Q_hat
    return nothing
end

"""
    update_C_d!(
        lds::LinearDynamicalSystem{T,S,O},
        tfs::TrialFilterSmooth{T},
        y::AbstractArray{T,3},
        w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing
    )

Update the observation matrix C and bias d of the Linear Dynamical System.
"""
function update_C_d!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[5] || return nothing

    ntrials = length(tfs)
    tsteps = size(y, 2)
    D = lds.latent_dim
    p = lds.obs_dim

    # Accumulate statistics for [C d] jointly
    Syz = zeros(T, p, D + 1)
    Szz = zeros(T, D + 1, D + 1)

    # Pre-allocate working matrices (reuse across all trials and timesteps!)
    work_yz = Matrix{T}(undef, p, D)  # For y * μ'
    work_outer = Matrix{T}(undef, D, D)  # For weighted Σ

    for trial in 1:ntrials
        fs = tfs[trial]
        weights = isnothing(w) ? nothing : w[trial]
        @views for t in 1:tsteps
            wt = isnothing(weights) ? one(T) : weights[t]

            μ = fs.E_z[:, t]
            Σ = fs.E_zz[:, :, t]  # E[x_t x_tᵀ]
            yt = y[:, t, trial]

            # Syz accumulates y_t * μ'  (outer product), weighted
            fill!(work_yz, zero(T))
            BLAS.ger!(wt, yt, μ, work_yz)   # work_yz += wt * yt * μ'
            Syz[:, 1:D] .+= work_yz
            Syz[:, D + 1] .+= wt .* yt      # bias column

            # Szz accumulates E[[x;1][x;1]ᵀ] weighted
            work_outer .= Σ
            work_outer .*= wt
            Szz[1:D, 1:D] .+= work_outer
            Szz[1:D, D + 1] .+= wt .* μ
            Szz[D + 1, 1:D] .+= wt .* μ
            Szz[D + 1, D + 1] += wt
        end
    end

    # Solve jointly: [C d] = Syz / Szz
    CD = Syz / Szz
    lds.obs_model.C = CD[:, 1:D]
    lds.obs_model.d = CD[:, D + 1]

    return nothing
end

"""
    update_R!(
        lds::LinearDynamicalSystem{T,S,O},
        tfs::TrialFilterSmooth{T},
        y::AbstractArray{T,3},
        w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing
    )

Update the observation noise covariance matrix R of the Linear Dynamical System.
"""
function update_R!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[6] || return nothing

    obs_dim = lds.obs_dim
    tsteps = size(y, 2)
    ntrials = length(tfs)

    R_new = zeros(T, obs_dim, obs_dim)
    C = lds.obs_model.C
    d = lds.obs_model.d

    total_weight = zero(T)

    # Pre-allocate all temporary arrays (reuse across all trials and timesteps!)
    innovation = Vector{T}(undef, obs_dim)
    Czt = Vector{T}(undef, obs_dim)
    temp_matrix = Matrix{T}(undef, obs_dim, lds.latent_dim)
    outer_product = Matrix{T}(undef, lds.latent_dim, lds.latent_dim)
    state_uncertainty = Matrix{T}(undef, lds.latent_dim, lds.latent_dim)

    for trial in 1:ntrials
        fs = tfs[trial]
        weights = isnothing(w) ? nothing : w[trial]

        @views for t in 1:tsteps
            wt = isnothing(weights) ? one(T) : weights[t]

            # Compute innovation = y - (C*z_t + d) using pre-allocated arrays
            mul!(Czt, C, fs.E_z[:, t])
            @. innovation = y[:, t, trial] - (Czt + d)

            # Add weighted innovation outer product
            BLAS.ger!(wt, innovation, innovation, R_new)

            # Compute state_uncertainty = E[zz] - E[z]E[z]' efficiently
            mul!(outer_product, fs.E_z[:, t], fs.E_z[:, t]')
            state_uncertainty .= fs.E_zz[:, :, t]
            state_uncertainty .-= outer_product

            # Add weighted C * state_uncertainty * C'
            mul!(temp_matrix, C, state_uncertainty)
            mul!(R_new, temp_matrix, C', wt, one(T))

            total_weight += wt
        end
    end

    # Apply prior and normalize
    if lds.obs_model.R_prior === nothing
        R_hat = R_new ./ total_weight
    else
        Ψ, ν = lds.obs_model.R_prior.Ψ, lds.obs_model.R_prior.ν
        R_hat = iw_map(Ψ, ν, R_new, total_weight, obs_dim)
    end

    R_hat .= 0.5 .* (R_hat .+ R_hat')
    lds.obs_model.R = R_hat
    return nothing
end

"""
    mstep!(lds, tfs, y, w)

Perform the M-step of the EM algorithm for a Linear Dynamical System with multi-trial data.
"""
function mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    # Get initial parameters using new approach
    old_params = _get_all_params_vec(lds)

    # Update parameters
    update_initial_state_mean!(lds, tfs, w)
    update_initial_state_covariance!(lds, tfs, w)
    update_A_b!(lds, tfs, w)
    update_Q!(lds, tfs, w)
    update_C_d!(lds, tfs, y, w)
    update_R!(lds, tfs, y, w)

    # Get new parameters using new approach
    new_params = _get_all_params_vec(lds)

    # Parameter delta
    norm_change = norm(new_params - old_params)
    return norm_change
end

"""
    mstep!(lds, tfs, y, sws; w=nothing)

Low-allocation M-step using `SmoothWorkspace` buffers.
"""
function mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    sws::SmoothWorkspace{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    old_params = _get_all_params_vec(lds)

    update_initial_state_mean!(lds, tfs, w)
    update_initial_state_covariance!(lds, tfs, sws, w)
    update_A_b!(lds, tfs, sws, w)
    update_Q!(lds, tfs, sws, w)
    update_C_d!(lds, tfs, y, sws, w)
    update_R!(lds, tfs, y, sws, w)

    new_params = _get_all_params_vec(lds)
    norm_change = norm(new_params - old_params)
    return norm_change
end

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

    for trial in 1:length(tfs.FilterSmooths)
        fs = tfs[trial]
        wt = isnothing(w) ? one(T) : w[trial][1]
        S0_sum .+= wt .* (fs.E_zz[:, :, 1] - (lds.state_model.x0 * lds.state_model.x0'))
        total_weight += wt
    end

    P0_hat = if lds.state_model.P0_prior === nothing
        S0_sum ./ total_weight
    else
        Ψ, ν = lds.state_model.P0_prior.Ψ, lds.state_model.P0_prior.ν
        iw_map(Ψ, ν, S0_sum, total_weight, D)
    end

    P0_hat .= 0.5 .* (P0_hat .+ P0_hat')
    lds.state_model.P0 = P0_hat
    return nothing
end

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

    AB = Sxz / Szz
    lds.state_model.A = AB[:, 1:D]
    lds.state_model.b = AB[:, D + 1]

    return nothing
end

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
            mul!(innovation_cov, μt, b', -one(T), one(T))
            mul!(innovation_cov, b, μt', -one(T), one(T))
            mul!(innovation_cov, temp5, b', one(T), one(T))
            mul!(innovation_cov, b, temp5', one(T), one(T))
            mul!(innovation_cov, b, b', one(T), one(T))

            Q_sum .+= weight .* innovation_cov
            total_weight += weight
        end
    end

    if lds.state_model.Q_prior === nothing
        Q_hat = Q_sum ./ total_weight
    else
        Ψ, ν = lds.state_model.Q_prior.Ψ, lds.state_model.Q_prior.ν
        Q_hat = iw_map(Ψ, ν, Q_sum, total_weight, state_dim)
    end

    Q_hat .= 0.5 .* (Q_hat .+ Q_hat')
    lds.state_model.Q = Q_hat
    return nothing
end

function update_C_d!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    sws::SmoothWorkspace{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[5] || return nothing

    ntrials = length(tfs)
    tsteps = size(y, 2)
    D = lds.latent_dim
    p = lds.obs_dim

    Syz = sws.Syz
    Szz = sws.Szz_Cd
    fill!(Syz, zero(T))
    fill!(Szz, zero(T))

    work_yz = sws.work_yz
    work_outer = sws.work_outer

    for trial in 1:ntrials
        fs = tfs[trial]
        weights = isnothing(w) ? nothing : w[trial]
        @views for t in 1:tsteps
            wt = isnothing(weights) ? one(T) : weights[t]

            μ = fs.E_z[:, t]
            Σ = fs.E_zz[:, :, t]
            yt = y[:, t, trial]

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

    CD = Syz / Szz
    lds.obs_model.C = CD[:, 1:D]
    lds.obs_model.d = CD[:, D + 1]

    return nothing
end

function update_R!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    sws::SmoothWorkspace{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[6] || return nothing

    obs_dim = lds.obs_dim
    tsteps = size(y, 2)
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
        weights = isnothing(w) ? nothing : w[trial]

        @views for t in 1:tsteps
            wt = isnothing(weights) ? one(T) : weights[t]

            mul!(Czt, C, fs.E_z[:, t])
            @. innovation = y[:, t, trial] - (Czt + d)

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

    R_hat .= 0.5 .* (R_hat .+ R_hat')
    lds.obs_model.R = R_hat
    return nothing
end

"""
    fit!(lds, y; max_iter::Int=100, tol::Real=1e-6)
    where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Fit a Linear Dynamical System using the Expectation-Maximization (EM) algorithm with Kalman
smoothing over multiple trials

# Arguments
- `lds::LinearDynamicalSystem{T,S,O}`: The Linear Dynamical System to be fitted.
- `y::AbstractArray{T,3}`: Observed data, size(obs_dim, T_steps, n_trials)

# Keyword Arguments
- `max_iter::Int=100`: Maximum number of EM iterations.
- `tol::T=1e-6`: Convergence tolerance for log-likelihood change.

# Returns
- `mls::Vector{T}`: Vector of log-likelihood values for each iteration.
- `param_diff::Vector{T}`: Vector of parameter deltas for each EM iteration.
"""
function fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3};
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress=true,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    if eltype(y) !== T
        error("Observed data must be of type $(T); Got $(eltype(y)))")
    end

    # Initialize log-likelihood
    prev_elbo = -T(Inf)

    # Create a vector to store the log-likelihood values
    elbos = Vector{T}()
    param_diff = Vector{T}()

    sizehint!(elbos, max_iter)  # Pre-allocate for efficiency
    # Create a FilterSmooth object
    tfs = initialize_FilterSmooth(lds, size(y, 2), size(y, 3))

    # Create workspace pool (one per thread)
    sws_pool = if lds.latent_dim > 10
        [SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, size(y, 2)) for _ in 1:Threads.nthreads()]
    else
        nothing
    end

    # Initialize progress bar only if progress=true
    prog = if progress
        if O <: GaussianObservationModel
            Progress(max_iter; desc="Fitting LDS via EM...", barlen=50, showspeed=true)
        elseif O <: PoissonObservationModel
            Progress(
                max_iter;
                desc="Fitting Poisson LDS via LaPlaceEM...",
                barlen=50,
                showspeed=true,
            )
        else
            error("Unknown LDS model type")
        end
    else
        nothing
    end

    # Run EM
    for i in 1:max_iter
        # E-step
        if sws_pool !== nothing
            elbo = estep!(lds, tfs, y, sws_pool)
            # M-step (use first workspace for M-step buffers)
            Δparams = mstep!(lds, tfs, y, sws_pool[1])
        else
            elbo = estep!(lds, tfs, y)
            Δparams = mstep!(lds, tfs, y)
        end
        # Update the log-likelihood vector and parameter difference
        push!(elbos, elbo)
        push!(param_diff, Δparams)

        # Update the progress bar only if it exists
        if progress && prog !== nothing
            next!(prog)
        end

        # Check convergence
        if abs(elbo - prev_elbo) < tol
            if progress && prog !== nothing
                finish!(prog)
            end
            return elbos, param_diff
        end

        prev_elbo = elbo
    end

    # Finish the progress bar if max_iter is reached
    if progress && prog !== nothing
        finish!(prog)
    end

    return elbos, param_diff
end
