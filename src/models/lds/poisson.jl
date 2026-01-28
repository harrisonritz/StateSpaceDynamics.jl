# Poisson LDS implementations - exports handled by types.jl and gaussian.jl

"""
    loglikelihood(
        x::AbstractMatrix{U},
        plds::LinearDynamicalSystem{T,S,O},
        y::AbstractMatrix{T}
    )

Calculate the complete-data log-likelihood of a Poisson Linear Dynamical System model for a
single trial.

# Arguments
- `x::AbstractMatrix{T}`: The latent state variables. Dimensions: (latent_dim, tsteps)
- `lds::LinearDynamicalSystem{T,S,O}`: The Linear Dynamical System model.
- `y::AbstractMatrix{T}`: The observed data. Dimensions: (obs_dim, tsteps)
- `w::Vector{T}`: Weights for each observation in the log-likelihood calculation. Not
    currently used.

# Returns
- `ll::Vector{T}`: The log-likelihood value.

# Ref
- loglikelihood(
    x::AbstractArray{T,3},
    plds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3}
)
"""
function loglikelihood(
    x::AbstractMatrix{U}, plds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}
) where {U<:Real,T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}

    # Result type and setup
    R = promote_type(T, U)
    tsteps = size(y, 2)
    ll = zeros(R, tsteps)

    # Convert the log firing rate to firing rate
    d = exp.(plds.obs_model.log_d)

    # Pre-compute Cholesky factorizations
    P0_chol = cholesky(Symmetric(plds.state_model.P0))
    Q_chol = cholesky(Symmetric(plds.state_model.Q))

    # Get dimensions
    C = plds.obs_model.C
    A = plds.state_model.A
    x0 = plds.state_model.x0
    obs_dim, latent_dim = size(C)
    obs_tmp = Vector{eltype(x)}(undef, obs_dim)

    @views for t in 1:tsteps
        obs_tmp .= C * x[:, t] .+ d
        ll[t] += (dot(y[:, t], obs_tmp) - sum(exp, obs_tmp))
    end

    # Prior term p(x₁) goes to t = 1
    dx1 = @view(x[:, 1]) .- plds.state_model.x0
    ll[1] += -R(0.5) * dot(dx1, P0_chol \ dx1)

    # Transition terms p(xₜ|xₜ₋₁) go to their respective t (t ≥ 2)
    A = plds.state_model.A
    b = plds.state_model.b
    trans_tmp = Vector{eltype(x)}(undef, latent_dim)

    @views for t in 2:tsteps
        trans_tmp .= x[:, t] .- (A * x[:, t - 1] .+ b)
        ll[t] += -R(0.5) * dot(trans_tmp, Q_chol \ trans_tmp)
    end

    return ll
end

"""
    loglikelihood(x::AbstractArray{T,3}, plds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T,3})

Calculate the complete-data log-likelihood of a Poisson Linear Dynamical System model for multiple trials.
"""
function loglikelihood(
    x::AbstractArray{T,3}, plds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Calculate the log-likelihood over all trials
    ll = zeros(T, size(y, 3))

    @threads for n in axes(y, 3)
        ll[n] .= sum(loglikelihood(x[:, :, n], plds, y[:, :, n]))
    end

    return sum(ll)
end

"""
    Gradient(lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T})

Calculate the gradient of the log-likelihood of a Poisson Linear Dynamical System model for a single trial.
"""
function Gradient(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
    w::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if w === nothing
        w = ones(T, size(y, 2))
    end

    # Extract model parameters
    A, Q, b = lds.state_model.A, lds.state_model.Q, lds.state_model.b
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d (non-log space)
    d = exp.(log_d)

    # Get dimensions
    tsteps = size(y, 2)
    latent_dim = lds.latent_dim
    obs_dim = lds.obs_dim

    # Precompute Cholesky factorizations
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))

    # Pre-allocate gradient
    grad = zeros(T, latent_dim, tsteps)

    # Pre-allocate ALL temporary vectors (reused across timesteps)
    Cx_t = Vector{T}(undef, obs_dim)           # C * x[:, t]
    exp_term = Vector{T}(undef, obs_dim)       # exp(C * x[:, t] + d)
    innovation = Vector{T}(undef, obs_dim)     # y[:, t] - exp_term
    common_term = Vector{T}(undef, latent_dim) # C' * innovation

    # Temporary vectors for state dynamics terms
    Ax_t = Vector{T}(undef, latent_dim)        # A * x[:, t]
    Ax_prev = Vector{T}(undef, latent_dim)     # A * x[:, t-1]
    state_diff = Vector{T}(undef, latent_dim)  # Various state differences
    temp_grad = Vector{T}(undef, latent_dim)   # Temporary for accumulating gradient parts

    # Calculate gradient for each time step
    @views for t in 1:tsteps
        # Compute observation term efficiently
        # temp = exp.(C * x[:, t] .+ d)
        mul!(Cx_t, C, x[:, t])                 # Cx_t = C * x[:, t]
        for i in 1:obs_dim
            exp_term[i] = exp(Cx_t[i] + d[i])  # exp_term = exp(C * x[:, t] + d)
        end

        # common_term = C' * (y[:, t] - temp)
        innovation .= y[:, t] .- exp_term      # innovation = y[:, t] - exp_term
        mul!(common_term, C', innovation)      # common_term = C' * innovation

        if t == 1
            # First time step: common_term + A' * inv_Q * (x[:, 2] - A * x[:, t]) - inv_P0 * (x[:, t] - x0)

            # Compute A * x[:, t]
            mul!(Ax_t, A, x[:, t])

            # Compute x[:, 2] - A * x[:, t]
            state_diff .= x[:, 2] .- Ax_t

            # Compute Q \ (x[:, 2] - A * x[:, t])
            copyto!(temp_grad, state_diff)
            ldiv!(Q_chol, temp_grad)

            # Compute A' * Q \ (x[:, 2] - A * x[:, t])
            mul!(grad[:, t], A', temp_grad)

            # Add common_term
            grad[:, t] .+= common_term

            # Subtract P0 \ (x[:, t] - x0)
            state_diff .= x[:, t] .- x0
            copyto!(temp_grad, state_diff)
            ldiv!(P0_chol, temp_grad)
            grad[:, t] .-= temp_grad

        elseif t == tsteps
            # Last time step: common_term - Q \ (x[:, t] - A * x[:, t-1])

            # Compute A * x[:, t-1]
            mul!(Ax_prev, A, x[:, t - 1])

            # Compute x[:, t] - A * x[:, t-1]
            state_diff .= x[:, t] .- Ax_prev

            # Compute Q \ (x[:, t] - A * x[:, t-1])
            copyto!(temp_grad, state_diff)
            ldiv!(Q_chol, temp_grad)

            # grad[:, t] = common_term - Q \ (...)
            grad[:, t] .= common_term .- temp_grad

        else
            # Intermediate time steps:
            # common_term + A' * Q \ (x[:, t+1] - A * x[:, t]) - Q \ (x[:, t] - A * x[:, t-1])

            # First part: A' * Q \ (x[:, t+1] - A * x[:, t])
            mul!(Ax_t, A, x[:, t])                   # Ax_t = A * x[:, t]
            state_diff .= x[:, t + 1] .- Ax_t       # state_diff = x[:, t+1] - A * x[:, t]
            copyto!(temp_grad, state_diff)           # temp_grad = state_diff
            ldiv!(Q_chol, temp_grad)                 # temp_grad = Q \ state_diff
            mul!(grad[:, t], A', temp_grad)          # grad[:, t] = A' * temp_grad

            # Add common_term
            grad[:, t] .+= common_term

            # Second part: - Q \ (x[:, t] - A * x[:, t-1])
            mul!(Ax_prev, A, x[:, t - 1])           # Ax_prev = A * x[:, t-1]
            state_diff .= x[:, t] .- Ax_prev        # state_diff = x[:, t] - A * x[:, t-1]
            copyto!(temp_grad, state_diff)           # temp_grad = state_diff
            ldiv!(Q_chol, temp_grad)                 # temp_grad = Q \ state_diff
            grad[:, t] .-= temp_grad                 # grad[:, t] -= temp_grad
        end
    end

    return grad
end

"""
    Hessian(lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T})

Calculate the Hessian matrix of the log-likelihood for a Poisson Linear Dynamical System.
"""
function Hessian(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
    w::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if w === nothing
        w=ones(T, size(y, 2))
    end

    # Extract model components
    A, Q = lds.state_model.A, lds.state_model.Q
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d i.e. non-log space
    d = exp.(log_d)

    # Pre-compute a few things
    tsteps = size(y, 2)
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))

    # Calculate super and sub diagonals
    H_sub_entry = Q_chol \ A
    H_super_entry = permutedims(H_sub_entry)

    # Fill the super and sub diagonals
    H_sub = [H_sub_entry for _ in 1:(tsteps - 1)]
    H_super = [H_super_entry for _ in 1:(tsteps - 1)]

    λ = zeros(T, size(C, 1))
    z = similar(λ)
    poisson_tmp = Matrix{T}(undef, size(C, 2), size(C, 2))
    H_diag = [Matrix{T}(undef, size(x, 1), size(x, 1)) for _ in 1:tsteps]

    # minnimal allocation Hessian helper function
    function calculate_poisson_hess!(out::Matrix{T}, C::Matrix{T}, λ::Vector{T}) where {T}
        n, p = size(C)
        for j in 1:p, i in 1:p
            acc = zero(T)
            for k in 1:n
                acc += C[k, i] * λ[k] * C[k, j]
            end
            out[i, j] = -acc
        end
    end

    # Pre-computed values for the Hessian
    state_dim = size(A, 1)
    I_mat = Matrix{T}(I, state_dim, state_dim)
    xt_given_xt_1 = -(Q_chol \ I_mat)
    xt1_given_xt = -A' * (Q_chol \ A)
    x_t = -(P0_chol \ I_mat)

    Q_middle = xt1_given_xt + xt_given_xt_1
    Q_first = x_t + xt1_given_xt
    Q_last = xt_given_xt_1

    @views for t in 1:tsteps
        mul!(z, C, x[:, t])  # z = C * x[:, t]
        @. λ = exp(z + d)

        if t == 1
            H_diag[t] .= Q_first
        elseif t == tsteps
            H_diag[t] .= Q_last
        else
            H_diag[t] .= Q_middle
        end

        calculate_poisson_hess!(poisson_tmp, C, λ)
        H_diag[t] .+= poisson_tmp
    end

    H = block_tridgm(H_diag, H_super, H_sub)

    return H, H_diag, H_super, H_sub
end

"""
    Hessian!(ws, lds, y, x; w=nothing)

In-place version of `Hessian` for Poisson LDS that writes blocks into the workspace
and updates the sparse matrix values. Returns the workspace's sparse matrix.
"""
function Hessian!(
    ws::BlockTridiagonalWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
    w::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if w === nothing
        w = ones(T, size(y, 2))
    end

    A, Q = lds.state_model.A, lds.state_model.Q
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    d = exp.(log_d)

    tsteps = size(y, 2)
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))

    H_sub_entry = Q_chol \ A
    H_super_entry = permutedims(H_sub_entry)

    for i in 1:(tsteps - 1)
        copyto!(ws.H_sub[i], H_sub_entry)
        copyto!(ws.H_super[i], H_super_entry)
    end

    state_dim = size(A, 1)
    obs_dim = size(C, 1)
    λ = zeros(T, obs_dim)
    z = similar(λ)
    poisson_tmp = Matrix{T}(undef, state_dim, state_dim)

    function calculate_poisson_hess!(out::Matrix{T}, C::Matrix{T}, λ::Vector{T}) where {T}
        n, p = size(C)
        for j in 1:p, i in 1:p
            acc = zero(T)
            for k in 1:n
                acc += C[k, i] * λ[k] * C[k, j]
            end
            out[i, j] = -acc
        end
    end

    I_mat = Matrix{T}(I, state_dim, state_dim)
    xt_given_xt_1 = -(Q_chol \ I_mat)
    xt1_given_xt = -A' * (Q_chol \ A)
    x_t = -(P0_chol \ I_mat)

    Q_middle = xt1_given_xt + xt_given_xt_1
    Q_first = x_t + xt1_given_xt
    Q_last = xt_given_xt_1

    @views for t in 1:tsteps
        mul!(z, C, x[:, t])
        @. λ = exp(z + d)

        if t == 1
            ws.H_diag[t] .= Q_first
        elseif t == tsteps
            ws.H_diag[t] .= Q_last
        else
            ws.H_diag[t] .= Q_middle
        end

        calculate_poisson_hess!(poisson_tmp, C, λ)
        ws.H_diag[t] .+= poisson_tmp
    end

    block_tridgm!(ws)

    return ws.H_sparse
end

"""
    Q_state(A, Q, P0, x0, E_z, E_zz, E_zz_prev)

Calculates the Q-function for the state model over multiple trials.
"""
function Q_state(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    P0::AbstractMatrix{T},
    x0::AbstractVector{T},
    E_z::AbstractArray{T,3},
    E_zz::AbstractArray{T,4},
    E_zz_prev::AbstractArray{T,4},
) where {T<:Real}
    # Calculate the Q-function for the state model
    vals = zeros(T, size(E_z, 3))

    @views @threads for k in axes(E_z, 3)
        vals[k] = Q_state(
            A, b, Q, P0, x0, E_z[:, :, k], E_zz[:, :, :, k], E_zz_prev[:, :, :, k]
        )
    end

    return sum(vals)
end

"""
    Q_observation_model(C, D, log_d, E_z, E_zz, y)

Calculate the Q-function for the observation model.
"""
function Q_observation_model(
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractArray{U,3},
    P_smooth::AbstractArray{U,4},
    y::AbstractArray{U,3},
) where {T<:Real,U<:Real}
    obs_dim, state_dim = size(C)

    d = exp.(log_d)
    Q_val = zero(T)
    trials = size(E_z, 3)
    tsteps = size(E_z, 2)

    h = Vector{T}(undef, obs_dim)
    ρ = Vector{T}(undef, obs_dim)
    CC = zeros(T, obs_dim, state_dim^2)

    for i in 1:obs_dim
        CC[i, :] .= vec(C[i, :] * C[i, :]')
    end

    @threads for k in 1:trials
        @views for t in 1:tsteps
            Ez_t = view(E_z, :, t, k)
            P_t = view(P_smooth,:,:,t,k)
            y_t = view(y, :, t, k)

            mul!(h, C, Ez_t)          # h = C * E_z[:, t, k]
            h .+= d

            mul!(ρ, CC, reshape(P_t, :))
            ρ .*= T(0.5)
            ŷ = exp.(h .+ ρ)

            Q_val += sum(y_t .* h .- ŷ)
        end
    end

    return Q_val
end

"""
    Q_observation_model(C, log_d, E_z, p_smooth, y, weights)

Calculate the Q-function for the observation model for a single trial with optional per-timestep weights.
"""
function Q_observation_model(
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractMatrix{T},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
    weights::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real}
    obs_dim, state_dim = size(C)
    d = exp.(log_d)
    Q_val = zero(T)
    tsteps = size(y, 2)

    h = Vector{T}(undef, obs_dim)
    ρ = Vector{T}(undef, obs_dim)
    temp_vec = Vector{T}(undef, state_dim)

    @views for t in 1:tsteps
        wt = isnothing(weights) ? one(T) : weights[t]

        Ez_t = E_z[:, t]
        P_t = p_smooth[:, :, t]
        y_t = y[:, t]

        # h = C * Ez_t + d
        mul!(h, C, Ez_t)
        h .+= d

        # Compute ρ[i] = 0.5 * c_i' * P_t * c_i
        for i in 1:obs_dim
            c_i = view(C, i, :)
            mul!(temp_vec, P_t, c_i)
            ρ[i] = T(0.5) * dot(c_i, temp_vec)
        end

        # Compute ŷ = exp(h + ρ) in-place, reusing ρ as ŷ
        @. ρ = exp(h + ρ)

        # Compute weighted sum(y_t .* h .- ŷ)
        for i in 1:obs_dim
            Q_val += wt * (y_t[i] * h[i] - ρ[i])
        end
    end

    return Q_val
end

"""
    Q_observation_model(C, log_d, tfs, y, w)

Calculate the Q-function for the observation model across all trials using TrialFilterSmooth with optional weights.
"""
function Q_observation_model(
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real}
    trials = length(tfs.FilterSmooths)
    Q_vals = zeros(T, trials)

    @threads for k in 1:trials
        fs = tfs[k]
        weights = isnothing(w) ? nothing : w[k]
        Q_vals[k] = Q_observation_model(
            C, log_d, fs.E_z, fs.p_smooth, view(y,:,:,k), weights
        )
    end

    return sum(Q_vals)
end

"""
    Q_function(A, b, Q, C, log_d, x0, P0, E_z, E_zz, E_zz_prev, y)

Calculate the Q-function for a single trial of a Poisson Linear Dynamical System.
"""
function Q_function(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    x0::AbstractVector{T},
    P0::AbstractMatrix{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real}
    state_q = StateSpaceDynamics.Q_state(A, b, Q, P0, x0, E_z, E_zz, E_zz_prev)
    obs_q = Q_observation_model(C, log_d, E_z, p_smooth, y)
    return state_q + obs_q
end

"""
    calculate_elbo(
        plds::LinearDynamicalSystem{T,S,O},
        E_z::AbstractArray{T, 3},
        E_zz::AbstractArray{T, 4},
        E_zz_prev::AbstractArray{T, 4},
        P_smooth::AbstractArray{T, 4},
        y::AbstractArray{T, 3},
        total_entropy::T
    ) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}

Calculate the Evidence Lower Bound (ELBO) for a Poisson Linear Dynamical System (PLDS). Adds constant-free IW log-prior terms 
for `Q` and `P0` when priors are set, so the ELBO tracks the MAP objective.

# Note
Ensure that the dimensions of input arrays match the expected dimensions as described in the
arguments section.
"""
function calculate_elbo(
    plds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Set up parameters
    A, b, Q, x0, p0 = plds.state_model.A,
    plds.state_model.b, plds.state_model.Q, plds.state_model.x0,
    plds.state_model.P0
    C, log_d = plds.obs_model.C, plds.obs_model.log_d

    ntrials = length(tfs.FilterSmooths)
    Q_vals = zeros(T, ntrials)

    # Calculate total entropy from individual FilterSmooth objects
    total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths)

    # Thread over trials (like Gaussian version)
    @threads for trial in 1:ntrials
        fs = tfs[trial]  # Get the FilterSmooth for this trial
        Q_vals[trial] = Q_function(
            A,
            b,
            Q,
            C,
            log_d,
            x0,
            p0,
            fs.E_z,
            fs.E_zz,
            fs.E_zz_prev,
            fs.p_smooth,
            view(y,:,:,trial),
        )
    end

    # IW priors on state covariances (if present)
    prior_term = zero(T)
    if plds.state_model.Q_prior !== nothing
        prior_term += iw_logprior_term(plds.state_model.Q, plds.state_model.Q_prior)
    end

    if plds.state_model.P0_prior !== nothing
        prior_term += iw_logprior_term(plds.state_model.P0, plds.state_model.P0_prior)
    end

    return sum(Q_vals) + prior_term - total_entropy
end

"""
    gradient_observation_model_single_trial!(grad, C, log_d, E_z, p_smooth, y, weights)

Compute the gradient for a single trial and add it to the accumulated gradient.
"""
function gradient_observation_model_single_trial!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractMatrix{T},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
    weights::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real}
    d = exp.(log_d)
    obs_dim, latent_dim = size(C)
    tsteps = size(y, 2)

    # Pre-allocate temporary arrays
    h = Vector{T}(undef, obs_dim)
    ρ = Vector{T}(undef, obs_dim)
    λ = Vector{T}(undef, obs_dim)
    CP = Matrix{T}(undef, obs_dim, latent_dim)

    @views for t in 1:tsteps
        weight = isnothing(weights) ? one(T) : weights[t]

        E_z_t = E_z[:, t]
        P_smooth_t = p_smooth[:, :, t]
        y_t = y[:, t]

        # Compute h = C * z_t + d
        mul!(h, C, E_z_t)
        h .+= d

        # Pre-compute CP = C * P_smooth_t
        mul!(CP, C, P_smooth_t)

        # Compute ρ and λ = exp(h + ρ)
        @views for i in 1:obs_dim
            ρ[i] = T(0.5) * dot(C[i, :], CP[i, :])
            λ[i] = exp(h[i] + ρ[i])
        end

        # Gradient computation with weight
        for j in 1:latent_dim
            for i in 1:obs_dim
                idx = (j - 1) * obs_dim + i
                grad[idx] += weight * (y_t[i] * E_z_t[j] - λ[i] * (E_z_t[j] + CP[i, j]))
            end
        end

        # Update log_d gradient
        @views grad[(end - obs_dim + 1):end] .+= weight .* (y_t .- λ) .* d
    end
end

"""
    gradient_observation_model!(grad, C, log_d, tfs, y, w)

Compute the gradient of the Q-function with respect to the observation model parameters using TrialFilterSmooth.
"""
function gradient_observation_model!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing;
    tasks_per_thread::Int=2,
) where {T<:Real}
    trials = length(tfs.FilterSmooths)
    npar = length(grad)

    # Partition work into chunks and spawn tasks: 
    # see https://julialang.org/blog/2023/07/PSA-dont-use-threadid/ for details.
    chunk_size = max(1, trials ÷ (tasks_per_thread * Threads.nthreads()))
    chunks = partition(1:trials, chunk_size)

    tasks = Task[]
    @sync begin
        for chunk in chunks
            push!(
                tasks,
                Threads.@spawn begin
                    acc = zeros(T, npar)   # task-local accumulator
                    tmp = zeros(T, npar)   # task-local scratch

                    for k in chunk
                        fill!(tmp, zero(T))

                        fs = tfs[k]
                        weights = isnothing(w) ? nothing : w[k]

                        gradient_observation_model_single_trial!(
                            tmp, C, log_d, fs.E_z, fs.p_smooth, view(y,:,:,k), weights
                        )

                        @simd for i in 1:npar
                            acc[i] += tmp[i]
                        end
                    end

                    return acc
                end
            )
        end
    end

    # Deterministic reduction on the caller thread
    fill!(grad, zero(T))
    for t in tasks
        acc = fetch(t)::Vector{T}  # type assertion avoids type instability from fetch
        @simd for i in 1:npar
            grad[i] += acc[i]
        end
    end

    @. grad = -grad
    return grad
end

"""
    update_observation_model!(plds, tfs, y, w)

Update the observation model parameters of a PLDS model.
"""
function update_observation_model!(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if plds.fit_bool[5]
        params = vcat(vec(plds.obs_model.C), plds.obs_model.log_d)

        function f(params::Vector{T})
            C_size = plds.obs_dim * plds.latent_dim
            log_d = params[(end - plds.obs_dim + 1):end]
            C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
            return -Q_observation_model(C, log_d, tfs, y, w)
        end

        function g!(grad::Vector{T}, params::Vector{T})
            C_size = plds.obs_dim * plds.latent_dim
            log_d = params[(end - plds.obs_dim + 1):end]
            C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
            return gradient_observation_model!(grad, C, log_d, tfs, y, w)
        end

        opts = Optim.Options(;
            x_reltol=1e-12, x_abstol=1e-12, g_abstol=1e-12, f_reltol=1e-12, f_abstol=1e-12
        )

        result = optimize(
            f, g!, params, LBFGS(; linesearch=LineSearches.HagerZhang()), opts
        )

        # Update the parameters
        C_size = plds.obs_dim * plds.latent_dim
        plds.obs_model.C = reshape(
            result.minimizer[1:C_size], plds.obs_dim, plds.latent_dim
        )
        plds.obs_model.log_d = result.minimizer[(end - plds.obs_dim + 1):end]
    end

    return nothing
end

"""
    mstep!(plds, tfs, y, w)

Perform the M-step of the EM algorithm for a Poisson Linear Dynamical System with multi-trial data.
"""
function mstep!(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Get old params
    old_params = _get_all_params_vec(plds)

    # Update state parameters
    update_initial_state_mean!(plds, tfs, w)
    update_initial_state_covariance!(plds, tfs, w)
    update_A_b!(plds, tfs, w)
    update_Q!(plds, tfs, w)

    # Update observation parameters
    update_observation_model!(plds, tfs, y, w)

    # Return parameter delta
    new_params = _get_all_params_vec(plds)
    return norm(new_params - old_params)
end


"""
    _fill_hessian_blocks_poisson!(ws, lds, x)

Fill the Hessian block diagonal and off-diagonal entries for Poisson LDS.
Uses pre-computed state model terms from `compute_smooth_constants_poisson!`
and computes the x-dependent Poisson observation term per-timestep.
"""
function _fill_hessian_blocks_poisson!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    tsteps = size(x, 2)
    btd = ws.btd
    C = lds.obs_model.C
    log_d = lds.obs_model.log_d
    d = exp.(log_d)
    obs_dim, latent_dim = size(C)

    # Pre-compute diagonal templates
    Q_middle = ws.xt1_given_xt .+ ws.xt_given_xt_1
    Q_first = ws.x_t .+ ws.xt1_given_xt
    Q_last = ws.xt_given_xt_1

    # Fill sub/super-diagonal blocks (constant for all timesteps)
    for i in 1:(tsteps - 1)
        copyto!(btd.H_sub[i], ws.H_sub_entry)
        copyto!(btd.H_super[i], ws.H_super_entry)
    end

    # Temporaries for Poisson observation term
    λ = Vector{T}(undef, obs_dim)
    z = Vector{T}(undef, obs_dim)

    @views for t in 1:tsteps
        # Start with state model contribution
        if t == 1
            copyto!(btd.H_diag[t], Q_first)
        elseif t == tsteps
            copyto!(btd.H_diag[t], Q_last)
        else
            copyto!(btd.H_diag[t], Q_middle)
        end

        # Add Poisson observation term: -C' * diag(λ) * C
        mul!(z, C, x[:, t])
        @. λ = exp(z + d)

        # Compute -C' * diag(λ) * C and add to diagonal block
        for j in 1:latent_dim, i in 1:latent_dim
            acc = zero(T)
            for k in 1:obs_dim
                acc += C[k, i] * λ[k] * C[k, j]
            end
            btd.H_diag[t][i, j] -= acc
        end
    end

    return nothing
end

"""
    _compute_gradient_poisson!(grad, ws, lds, y, x)

Compute the gradient of the log-likelihood for Poisson LDS.
Fills `grad` (latent_dim × tsteps) matrix with gradient values.
"""
function _compute_gradient_poisson!(
    grad::AbstractMatrix{T},
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    A = lds.state_model.A
    b = lds.state_model.b
    C = lds.obs_model.C
    log_d = lds.obs_model.log_d
    x0 = lds.state_model.x0

    d = exp.(log_d)
    tsteps = size(y, 2)
    latent_dim, obs_dim = lds.latent_dim, lds.obs_dim

    # Cholesky factors from workspace (stored as upper triangular)
    Q_chol_U = UpperTriangular(ws.Q_chol_U)
    P0_chol_U = UpperTriangular(ws.P0_chol_U)

    # Reuse workspace temp vectors
    Cx_t = ws.dyt           # obs_dim
    exp_term = ws.temp_dy   # obs_dim
    state_diff = ws.dxt     # latent_dim
    temp_grad = ws.tmp1     # latent_dim
    common_term = ws.tmp2   # latent_dim

    @views for t in 1:tsteps
        # Observation term: C' * (y - exp(C*x + d))
        mul!(Cx_t, C, x[:, t])
        @. exp_term = exp(Cx_t + d)
        exp_term .= y[:, t] .- exp_term
        mul!(common_term, C', exp_term)

        if t == 1
            # First timestep: common_term + A' * Q⁻¹ * (x₂ - A*x₁ - b) - P0⁻¹ * (x₁ - x0)
            mul!(state_diff, A, x[:, t])
            state_diff .= x[:, 2] .- state_diff .- b

            # Q⁻¹ * state_diff via Cholesky: solve Q_chol_U' * Q_chol_U * z = state_diff
            copyto!(temp_grad, state_diff)
            ldiv!(Q_chol_U', temp_grad)
            ldiv!(Q_chol_U, temp_grad)

            mul!(grad[:, t], A', temp_grad)
            grad[:, t] .+= common_term

            # Subtract P0⁻¹ * (x₁ - x0)
            state_diff .= x[:, t] .- x0
            copyto!(temp_grad, state_diff)
            ldiv!(P0_chol_U', temp_grad)
            ldiv!(P0_chol_U, temp_grad)
            grad[:, t] .-= temp_grad

        elseif t == tsteps
            # Last timestep: common_term - Q⁻¹ * (xₜ - A*xₜ₋₁ - b)
            mul!(state_diff, A, x[:, t - 1])
            state_diff .= x[:, t] .- state_diff .- b

            copyto!(temp_grad, state_diff)
            ldiv!(Q_chol_U', temp_grad)
            ldiv!(Q_chol_U, temp_grad)

            grad[:, t] .= common_term .- temp_grad
        else
            # Middle timesteps
            # common_term + A' * Q⁻¹ * (xₜ₊₁ - A*xₜ - b) - Q⁻¹ * (xₜ - A*xₜ₋₁ - b)

            # Forward term: A' * Q⁻¹ * (xₜ₊₁ - A*xₜ - b)
            mul!(state_diff, A, x[:, t])
            state_diff .= x[:, t + 1] .- state_diff .- b
            copyto!(temp_grad, state_diff)
            ldiv!(Q_chol_U', temp_grad)
            ldiv!(Q_chol_U, temp_grad)
            mul!(grad[:, t], A', temp_grad)
            grad[:, t] .+= common_term

            # Backward term: -Q⁻¹ * (xₜ - A*xₜ₋₁ - b)
            mul!(state_diff, A, x[:, t - 1])
            state_diff .= x[:, t] .- state_diff .- b
            copyto!(temp_grad, state_diff)
            ldiv!(Q_chol_U', temp_grad)
            ldiv!(Q_chol_U, temp_grad)
            grad[:, t] .-= temp_grad
        end
    end

    return grad
end

"""
    smooth!(lds, fs, y, sws; max_iter=20, tol=1e-6)

Poisson LDS smoothing using iterative Newton with block tridiagonal solve.
Uses `SmoothWorkspace` for pre-allocated buffers.

Since the Poisson log-likelihood is non-quadratic, multiple Newton iterations
are required (unlike Gaussian LDS which converges in one step).
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O}, fs::FilterSmooth{T}, y::AbstractMatrix{T},
    sws::SmoothWorkspace{T};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    tsteps, D = size(y, 2), lds.latent_dim
    btd = sws.btd

    # Pre-compute all constant terms for this smooth! call
    compute_smooth_constants_poisson!(sws, lds)

    # Use x_smooth as the working buffer (sufficient_statistics! copies x_smooth -> E_z)
    x = fs.x_smooth

    # Initialize from previous E_z (warm start) or from prior
    if all(fs.E_z .== 0)
        # First call: initialize from prior
        x[:, 1] .= lds.state_model.x0
        for t in 2:tsteps
            mul!(view(x, :, t), lds.state_model.A, view(x, :, t - 1))
            x[:, t] .+= lds.state_model.b
        end
    else
        # Warm start from previous E-step
        copyto!(x, fs.E_z)
    end

    # Newton iteration with backtracking line search (Armijo condition)
    converged = false
    c_armijo = T(1e-4)

    for iter in 1:max_iter
        # Compute gradient
        _compute_gradient_poisson!(sws.grad_buf, sws, lds, y, x)

        # Check gradient-based convergence
        grad_norm = norm(sws.grad_buf)
        if grad_norm < tol
            converged = true
            break
        end

        # Build Hessian blocks and negate for Newton step: (-H) * Δx = g
        _fill_hessian_blocks_poisson!(sws, lds, x)
        _negate_blocks!(btd)

        # Compute Newton direction via block tridiagonal solve
        grad_vec = vec(sws.grad_buf)
        copyto!(sws.X₀, grad_vec)
        block_tridiagonal_solve!(
            sws.X₀, btd.neg_sub, btd.neg_diag, btd.neg_super, grad_vec, btd
        )

        # Backtracking line search with Armijo condition
        Δx = reshape(sws.X₀, D, tsteps)
        current_ll = sum(loglikelihood(x, lds, y))
        directional_deriv = dot(grad_vec, sws.X₀)  # g' * Δx > 0 (ascent direction)
        α = one(T)

        for ls_iter in 1:25
            x .+= α .* Δx
            new_ll = sum(loglikelihood(x, lds, y))
            if isfinite(new_ll) && new_ll ≥ current_ll + c_armijo * α * directional_deriv
                break  # Accept step (x already updated)
            end
            x .-= α .* Δx  # Revert
            α *= T(0.5)
            if ls_iter == 25
                # All line search steps failed; take the smallest step
                x .+= α .* Δx
            end
        end

        # Check convergence on effective step size
        if α * norm(Δx) < tol
            converged = true
            break
        end
    end

    # After convergence, compute posterior covariances
    # -H at the MAP is the precision matrix of the Laplace approximation
    _fill_hessian_blocks_poisson!(sws, lds, x)
    _negate_blocks!(btd)

    # Compute inverse (covariance) and log-determinant
    logdet_precision = block_tridiagonal_inverse_logdet!(
        fs.p_smooth, fs.p_smooth_tt1,
        btd.neg_sub, btd.neg_diag, btd.neg_super, btd
    )

    # Compute entropy from log-determinant
    n_total = D * tsteps
    fs.entropy = gaussian_entropy_from_logdet(logdet_precision, n_total)

    return fs
end

"""
    smooth!(lds, tfs, y, sws_pool; max_iter=20, tol=1e-6)

Multi-trial Poisson LDS smoothing with workspace pool.
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3},
    sws_pool::Vector{SmoothWorkspace{T}};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    ntrials = size(y, 3)

    if ntrials == 1
        @views smooth!(lds, tfs[1], y[:, :, 1], sws_pool[1]; max_iter=max_iter, tol=tol)
    else
        @threads for trial in 1:ntrials
            tid = Threads.threadid()
            @views smooth!(
                lds, tfs[trial], y[:, :, trial], sws_pool[tid];
                max_iter=max_iter, tol=tol
            )
        end
    end

    return tfs
end

"""
    smooth!(lds, tfs, y; max_iter=20, tol=1e-6)

Convenience method for multi-trial Poisson LDS smoothing that creates workspaces internally.
This is less efficient than passing a pre-allocated workspace pool, but useful for testing.
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    tsteps = size(y, 2)
    npool = Threads.maxthreadid()
    sws_pool = [SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, tsteps) for _ in 1:npool]
    return smooth!(lds, tfs, y, sws_pool; max_iter=max_iter, tol=tol)
end

"""
    estep!(lds, tfs, y, sws_pool; max_iter=20, tol=1e-6)

E-step for Poisson LDS: smooth and compute sufficient statistics.
"""
function estep!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3},
    sws_pool::Vector{SmoothWorkspace{T}};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    smooth!(lds, tfs, y, sws_pool; max_iter=max_iter, tol=tol)
    sufficient_statistics!(tfs)
    elbo = calculate_elbo(lds, tfs, y)
    return elbo
end

"""
    estep!(lds, tfs, y; max_iter=20, tol=1e-6)

Convenience method for Poisson LDS E-step that creates workspaces internally.
"""
function estep!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    tsteps = size(y, 2)
    npool = Threads.maxthreadid()
    sws_pool = [SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, tsteps) for _ in 1:npool]
    return estep!(lds, tfs, y, sws_pool; max_iter=max_iter, tol=tol)
end

"""
    fit!(plds, y; max_iter=100, tol=1e-6, progress=true, newton_max_iter=20, newton_tol=1e-6)

Fit a Poisson LDS model using the Laplace-EM algorithm with workspace-based smoothing.

# Arguments
- `plds`: The Poisson Linear Dynamical System model
- `y`: Observed data array (obs_dim × tsteps × ntrials)
- `max_iter`: Maximum EM iterations (default: 100)
- `tol`: Convergence tolerance for ELBO (default: 1e-6)
- `progress`: Show progress bar (default: true)
- `newton_max_iter`: Max Newton iterations per E-step (default: 20)
- `newton_tol`: Newton convergence tolerance (default: 1e-6)
"""
function fit!(
    plds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3};
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress=true,
    newton_max_iter::Int=20,
    newton_tol::Float64=1e-6,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if eltype(y) !== T
        error("Observed data must be of type $(T); Got $(eltype(y)))")
    end

    obs_dim, tsteps, ntrials = size(y)
    latent_dim = plds.latent_dim

    # Initialize TrialFilterSmooth
    tfs = initialize_FilterSmooth(plds, tsteps, ntrials)

    # Create workspace pool (one per thread ID)
    npool = Threads.maxthreadid()
    sws_pool = [SmoothWorkspace(T, latent_dim, obs_dim, tsteps) for _ in 1:npool]

    # Tracking arrays
    elbos = Vector{T}(undef, max_iter)
    param_diffs = Vector{T}(undef, max_iter)

    # Progress bar
    prog = if progress
        Progress(max_iter; desc="Fitting Poisson LDS via LaPlaceEM...", barlen=50, showspeed=true)
    else
        nothing
    end

    # EM iterations
    for iter in 1:max_iter
        # E-step: smooth and compute sufficient statistics
        elbos[iter] = estep!(
            plds, tfs, y, sws_pool;
            max_iter=newton_max_iter, tol=T(newton_tol)
        )

        # M-step: update parameters
        param_diffs[iter] = mstep!(plds, tfs, y)

        # Update progress
        if !isnothing(prog)
            next!(prog; showvalues=[(:ELBO, elbos[iter]), (:Δparams, param_diffs[iter])])
        end

        # Check convergence (after at least 2 iterations)
        if iter > 1 && abs(elbos[iter] - elbos[iter - 1]) < tol
            resize!(elbos, iter)
            resize!(param_diffs, iter)
            break
        end
    end

    if !isnothing(prog)
        finish!(prog)
    end

    return elbos, param_diffs
end
