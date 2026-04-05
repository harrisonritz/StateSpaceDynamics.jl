"""
    _make_slds_fb_storage(dl, tsteps)

Pre-allocate a `HMMs.ForwardBackwardStorage` for one trial given discrete layer `dl` and
sequence length `tsteps`.  The observation sequence is `1:tsteps` (timestep indices).
"""
function _make_slds_fb_storage(dl::SLDSDiscreteLayer{T}, tsteps::Int) where {T}
    obs_seq = 1:tsteps
    ctrl_seq = fill(nothing, tsteps)
    return HMMs.initialize_forward_backward(
        dl, obs_seq, ctrl_seq; seq_ends=(tsteps,), transition_marginals=true
    )
end

"""
    rand(rng::AbstractRNG, slds::SLDS{T,S,O}; tsteps::Int, ntrials::Int=1) where {T<:Real, S<:AbstractStateModel, O<:AbstractObservationModel}

Sample from a Switching Linear Dynamical System (SLDS). Returns a tuple `(z, x, y)` where:
- `z` is a matrix of discrete states of size `(tsteps, ntrials)`
- `x` is a 3D array of continuous latent states of size `(latent_dim, tsteps, ntrials)`
- `y` is a 3D array of observations of size `(obs_dim, tsteps, ntrials)`
"""
function Random.rand(
    rng::AbstractRNG, slds::SLDS{T,S,O}; tsteps::Int, ntrials::Int=1
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    K = length(slds.LDSs)  # Number of discrete states
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim

    # Pre-allocate outputs
    z = Array{Int,2}(undef, tsteps, ntrials)        # Discrete states
    x = Array{T,3}(undef, latent_dim, tsteps, ntrials)  # Continuous states  
    y = Array{T,3}(undef, obs_dim, tsteps, ntrials)     # Observations

    # Pre-extract parameters for all LDS models
    state_params = [_extract_state_params(lds.state_model) for lds in slds.LDSs]
    obs_params = [_extract_obs_params(lds.obs_model) for lds in slds.LDSs]

    # Sample each trial
    for trial in 1:ntrials
        _sample_slds_trial!(
            rng,
            view(z, :, trial),
            view(x,:,:,trial),
            view(y,:,:,trial),
            slds.A,
            slds.πₖ,
            state_params,
            obs_params,
            slds.LDSs[1].obs_model,
        )  # Use first for type dispatch
    end

    return z, x, y
end

# Core SLDS trial sampling logic
function _sample_slds_trial!(
    rng, z_trial, x_trial, y_trial, A, πₖ, state_params, obs_params, obs_model_type
)
    tsteps = length(z_trial)
    K = size(A, 1)

    # Sample discrete state sequence using forward sampling
    z_trial[1] = rand(rng, Categorical(πₖ))
    for t in 2:tsteps
        z_trial[t] = rand(rng, Categorical(A[z_trial[t - 1], :]))
    end

    # Sample continuous states and observations given discrete sequence
    return _sample_continuous_given_discrete!(
        rng, x_trial, y_trial, z_trial, state_params, obs_params, obs_model_type
    )
end

# Sample continuous dynamics given discrete state sequence
function _sample_continuous_given_discrete!(
    rng,
    x_trial,
    y_trial,
    z_trial,
    state_params,
    obs_params,
    obs_model_type::GaussianObservationModel,
)
    tsteps = length(z_trial)

    # Initial state from the selected LDS
    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    y_trial[:, 1] = rand(
        rng, MvNormal(obs_params[k1].C * x_trial[:, 1] + obs_params[k1].d, obs_params[k1].R)
    )

    # Subsequent states - switch dynamics based on discrete state
    for t in 2:tsteps
        k_prev, k_curr = z_trial[t - 1], z_trial[t]

        # Continuous state follows previous discrete state's dynamics
        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params[k_curr].A * x_trial[:, t - 1] + state_params[k_curr].b,
                state_params[k_curr].Q,
            ),
        )

        # Observation follows current discrete state's model
        y_trial[:, t] = rand(
            rng,
            MvNormal(
                obs_params[k_curr].C * x_trial[:, t] + obs_params[k_curr].d,
                obs_params[k_curr].R,
            ),
        )
    end
end

function _sample_continuous_given_discrete!(
    rng,
    x_trial,
    y_trial,
    z_trial,
    state_params,
    obs_params,
    obs_model_type::PoissonObservationModel,
)
    tsteps = length(z_trial)

    # Initial state
    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    y_trial[:, 1] = rand.(
        rng, Poisson.(exp.(obs_params[k1].C * x_trial[:, 1] + obs_params[k1].d))
    )

    # Subsequent states
    for t in 2:tsteps
        k_prev, k_curr = z_trial[t - 1], z_trial[t]

        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params[k_curr].A * x_trial[:, t - 1] + state_params[k_curr].b,
                state_params[k_curr].Q,
            ),
        )

        y_trial[:, t] = rand.(
            rng, Poisson.(exp.(obs_params[k_curr].C * x_trial[:, t] + obs_params[k_curr].d))
        )
    end
end

# Convenience method without explicit RNG
Random.rand(slds::SLDS; kwargs...) = rand(Random.default_rng(), slds; kwargs...)

"""
    loglikelihood(slds::SLDS, x, y, w)
    
Compute weighted complete-data log-likelihood for SLDS.
Returns vector of per-timestep log-likelihoods.
"""
function loglikelihood!(
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    w::AbstractMatrix{T},   # K × T responsibilities/weights
) where {T<:Real}
    fill!(ws.ll_vec, zero(T))

    K = length(slds.LDSs)
    for k in 1:K
        loglikelihood!(ws.ll_tmp, ws, ws.consts[k], slds.LDSs[k], x, y)

        # accumulate: ll_vec[t] += w[k,t] * ll_tmp[t]
        for t in 1:length(ws.ll_vec)
            ws.ll_vec[t] += w[k, t] * ws.ll_tmp[t]
        end
    end

    return ws.ll_vec
end

"""
    Gradient!(ws, slds, y, x, w)

In-place SLDS gradient with a backend compatible shape to LDS.
Semantics match current SLDS.jl: compute per-timestep component gradient and scale by w[k,t].
Writes into `ws.grad_buf` and returns it.
"""
function Gradient!(
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
    w::AbstractMatrix{T},
) where {T<:Real}
    latent_dim, Tsteps = size(x)
    K = length(slds.LDSs)

    grad = ws.grad_buf
    fill!(grad, zero(T))

    dxt = ws.dxt
    dxt_next = ws.dxt_next
    dyt = ws.dyt
    tmp1 = ws.tmp1
    tmp2 = ws.tmp2
    tmp3 = ws.tmp3

    z = ws.z
    λ = ws.λ

    @views for k in 1:K
        lds_k = slds.LDSs[k]
        cc = ws.consts[k]

        A = lds_k.state_model.A
        b = lds_k.state_model.b
        x0 = lds_k.state_model.x0

        A_inv_Q = cc.A_inv_Q        # A'Q^{-1}
        neg_Q_inv = cc.xt_given_xt_1  # -Q^{-1}
        neg_P0_inv = cc.x_t            # -P0^{-1}

        if lds_k.obs_model isa GaussianObservationModel{T}
            C = cc.C
            d = lds_k.obs_model.d
            C_inv_R = cc.C_inv_R        # C'R^{-1}

            if Tsteps == 1
                # emission at t=1
                mul!(dyt, C, x[:, 1])
                @. dyt = y[:, 1] - dyt - d
                mul!(tmp1, C_inv_R, dyt)

                # prior at t=1
                @. dxt = x[:, 1] - x0
                mul!(tmp3, neg_P0_inv, dxt)

                α = w[k, 1]
                @. grad[:, 1] += α * (tmp1 + tmp3)
                continue
            end

            # t = 1 
            # emission (weighted by w[k,1])
            mul!(dyt, C, x[:, 1])
            @. dyt = y[:, 1] - dyt - d
            mul!(tmp1, C_inv_R, dyt)
            α = w[k, 1]
            @. grad[:, 1] += α * tmp1

            # prior (weighted by w[k,1])
            @. dxt = x[:, 1] - x0
            mul!(tmp3, neg_P0_inv, dxt)
            @. grad[:, 1] += α * tmp3

            # outgoing dynamics term comes from factor at time 2, weighted by w[k,2]
            mul!(dxt_next, A, x[:, 1])
            @. dxt_next = x[:, 2] - dxt_next - b
            mul!(tmp2, A_inv_Q, dxt_next)
            β = w[k, 2]
            @. grad[:, 1] += β * tmp2

            # 2 .. T-1
            for t in 2:(Tsteps - 1)
                # emission at t, weighted by w[k,t]
                mul!(dyt, C, x[:, t])
                @. dyt = y[:, t] - dyt - d
                mul!(tmp1, C_inv_R, dyt)
                α = w[k, t]
                @. grad[:, t] += α * tmp1

                # incoming dynamics factor at time t, weighted by w[k,t]
                mul!(dxt, A, x[:, t - 1])
                @. dxt = x[:, t] - dxt - b
                mul!(tmp3, neg_Q_inv, dxt)
                @. grad[:, t] += α * tmp3

                # outgoing dynamics factor at time t+1, weighted by w[k,t+1]
                mul!(dxt_next, A, x[:, t])
                @. dxt_next = x[:, t + 1] - dxt_next - b
                mul!(tmp2, A_inv_Q, dxt_next)
                β = w[k, t + 1]
                @. grad[:, t] += β * tmp2
            end

            # t = T
            # emission at T, weighted by w[k,T]
            mul!(dyt, C, x[:, Tsteps])
            @. dyt = y[:, Tsteps] - dyt - d
            mul!(tmp1, C_inv_R, dyt)
            α = w[k, Tsteps]
            @. grad[:, Tsteps] += α * tmp1

            # incoming dynamics factor at time T, weighted by w[k,T]
            mul!(dxt, A, x[:, Tsteps - 1])
            @. dxt = x[:, Tsteps] - dxt - b
            mul!(tmp3, neg_Q_inv, dxt)
            @. grad[:, Tsteps] += α * tmp3

        elseif lds_k.obs_model isa PoissonObservationModel{T}
            C = cc.C
            d = cc.d  # cached exp.(log_d)

            if Tsteps == 1
                # emission: C'*(y - λ)
                mul!(z, C, x[:, 1])
                @. λ = exp(z + d)
                @. z = y[:, 1] - λ
                mul!(tmp1, C', z)

                # prior
                @. dxt = x[:, 1] - x0
                mul!(tmp3, neg_P0_inv, dxt)

                α = w[k, 1]
                @. grad[:, 1] += α * (tmp1 + tmp3)
                continue
            end

            # t = 1
            mul!(z, C, x[:, 1])
            @. λ = exp(z + d)
            @. z = y[:, 1] - λ
            mul!(tmp1, C', z)
            α = w[k, 1]
            @. grad[:, 1] += α * tmp1

            @. dxt = x[:, 1] - x0
            mul!(tmp3, neg_P0_inv, dxt)
            @. grad[:, 1] += α * tmp3

            # outgoing dynamics factor at time 2 weighted by w[k,2]
            mul!(dxt_next, A, x[:, 1])
            @. dxt_next = x[:, 2] - dxt_next - b
            mul!(tmp2, A_inv_Q, dxt_next)
            β = w[k, 2]
            @. grad[:, 1] += β * tmp2

            # 2 .. T-1
            for t in 2:(Tsteps - 1)
                mul!(z, C, x[:, t])
                @. λ = exp(z + d)
                @. z = y[:, t] - λ
                mul!(tmp1, C', z)
                α = w[k, t]
                @. grad[:, t] += α * tmp1

                mul!(dxt, A, x[:, t - 1])
                @. dxt = x[:, t] - dxt - b
                mul!(tmp3, neg_Q_inv, dxt)
                @. grad[:, t] += α * tmp3

                mul!(dxt_next, A, x[:, t])
                @. dxt_next = x[:, t + 1] - dxt_next - b
                mul!(tmp2, A_inv_Q, dxt_next)
                β = w[k, t + 1]
                @. grad[:, t] += β * tmp2
            end

            # t = T
            mul!(z, C, x[:, Tsteps])
            @. λ = exp(z + d)
            @. z = y[:, Tsteps] - λ
            mul!(tmp1, C', z)
            α = w[k, Tsteps]
            @. grad[:, Tsteps] += α * tmp1

            mul!(dxt, A, x[:, Tsteps - 1])
            @. dxt = x[:, Tsteps] - dxt - b
            mul!(tmp3, neg_Q_inv, dxt)
            @. grad[:, Tsteps] += α * tmp3

        else
            throw(ArgumentError("Unsupported observation model $(typeof(lds_k.obs_model))"))
        end
    end

    return grad
end

"""
    Hessian_blocks!(ws, slds, y, x, w)

Fill `ws.btd.H_diag`, `ws.btd.H_sub`, `ws.btd.H_super` with the weighted Hessian blocks
for the Laplace/Newton step over `x₁:T` matching Zoltowski et al. Appendix B.

Convention matched:
    x_t | x_{t-1}, z_t=k ~ N(A_k x_{t-1} + b_k, Q_k)
so the dynamics factor that couples (x_{t-1}, x_t) is weighted by w[k,t] = q(z_t=k).

Weights:
- emission curvature at time t uses w[k,t]
- dynamics curvature from factor at time t uses w[k,t]
- off-diagonal block coupling (t-1,t) uses w[k,t]
"""
function Hessian_blocks!(
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
    w::AbstractMatrix{T},
) where {T<:Real}
    latent_dim, Tsteps = size(x)
    K = length(slds.LDSs)

    H_diag = ws.btd.H_diag
    H_sub = ws.btd.H_sub
    H_super = ws.btd.H_super

    for t in 1:Tsteps
        fill!(H_diag[t], zero(T))
    end
    for t in 1:(Tsteps - 1)
        fill!(H_sub[t], zero(T))
        fill!(H_super[t], zero(T))
    end

    z = ws.z
    λ = ws.λ

    @views for k in 1:K
        lds_k = slds.LDSs[k]
        cc = ws.consts[k]

        # Cached state-model templates for regime k
        neg_Q_inv = cc.xt_given_xt_1    # -Q^{-1}
        neg_AtQinvA = cc.xt1_given_xt     # -A'Q^{-1}A
        neg_P0_inv = cc.x_t              # -P0^{-1}
        sub_entry = cc.H_sub_entry      #  Q^{-1}A
        super_entry = cc.H_super_entry    # (Q^{-1}A)'

        if Tsteps == 1
            α = w[k, 1]
            @. H_diag[1] += α * neg_P0_inv

            if lds_k.obs_model isa GaussianObservationModel{T}
                @. H_diag[1] += α * cc.yt_given_xt
            elseif lds_k.obs_model isa PoissonObservationModel{T}
                C = cc.C
                d = cc.d  # cached exp.(log_d)

                mul!(z, C, x[:, 1])
                @. λ = exp(z + d)

                for j in 1:latent_dim, i in 1:latent_dim
                    acc = zero(T)
                    for o in eachindex(λ)
                        acc += C[o, i] * λ[o] * C[o, j]
                    end
                    H_diag[1][i, j] -= α * acc
                end
            else
                throw(
                    ArgumentError(
                        "Unsupported observation model $(typeof(lds_k.obs_model))"
                    ),
                )
            end

            continue
        end

        # Dynamics factor at time t couples (x_{t-1}, x_t), weighted by w[k,t].
        # Off-diagonal blocks between t-1 and t therefore use w[k,t].
        for t in 2:Tsteps
            α = w[k, t]
            @. H_sub[t - 1] += α * sub_entry
            @. H_super[t - 1] += α * super_entry
        end

        # Diagonal state-model contributions:
        # - At t=1: prior term weighted by w[k,1], plus "previous-role" from factor at t=2 weighted by w[k,2]
        @. H_diag[1] += w[k, 1] * neg_P0_inv
        @. H_diag[1] += w[k, 2] * neg_AtQinvA

        # - For 2..T-1: current-role from factor at t (neg_Q_inv) weighted by w[k,t]
        #               previous-role from factor at t+1 (neg_AtQinvA) weighted by w[k,t+1]
        for t in 2:(Tsteps - 1)
            @. H_diag[t] += w[k, t] * neg_Q_inv
            @. H_diag[t] += w[k, t + 1] * neg_AtQinvA
        end

        # - At t=T: current-role from factor at T weighted by w[k,T]
        @. H_diag[Tsteps] += w[k, Tsteps] * neg_Q_inv

        # Emission curvature contributions
        if lds_k.obs_model isa GaussianObservationModel{T}
            for t in 1:Tsteps
                @. H_diag[t] += w[k, t] * cc.yt_given_xt   # -C'R^{-1}C
            end

        elseif lds_k.obs_model isa PoissonObservationModel{T}
            C = cc.C
            d = cc.d  # cached exp.(log_d)

            # Add -w[k,t] * C' diag(λ_t) C where λ_t = exp(C x_t + d)
            # This implementation is allocation-free but O(latent^2 * obs) per time. Fix later.
            for t in 1:Tsteps
                α = w[k, t]

                mul!(z, C, x[:, t])
                @. λ = exp(z + d)

                for j in 1:latent_dim, i in 1:latent_dim
                    acc = zero(T)
                    for o in eachindex(λ)
                        acc += C[o, i] * λ[o] * C[o, j]
                    end
                    H_diag[t][i, j] -= α * acc
                end
            end

        else
            throw(ArgumentError("Unsupported observation model $(typeof(lds_k.obs_model))"))
        end
    end

    return nothing
end

"""
    _compute_hessian_blocks(lds, y, x) -> (H_diag, H_super, H_sub)

Compute Hessian blocks for a single LDS. Dispatches on observation model type.
"""
function _compute_hessian_blocks(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    latent_dim = lds.latent_dim

    # Pre-compute Cholesky factorizations
    Q_chol = cholesky(Symmetric(lds.state_model.Q))
    P0_chol = cholesky(Symmetric(lds.state_model.P0))
    R_chol = cholesky(Symmetric(lds.obs_model.R))

    A = lds.state_model.A
    C = lds.obs_model.C

    # Compute block templates (constant for Gaussian)
    H_sub_entry = Q_chol \ A
    H_super_entry = permutedims(H_sub_entry)

    I_mat = Matrix{T}(I, latent_dim, latent_dim)
    yt_given_xt = -C' * (R_chol \ C)
    xt_given_xt_1 = -(Q_chol \ I_mat)
    xt1_given_xt = -A' * (Q_chol \ A)
    x_t = -(P0_chol \ I_mat)

    # Allocate blocks
    H_diag = [Matrix{T}(undef, latent_dim, latent_dim) for _ in 1:tsteps]
    H_sub = [copy(H_sub_entry) for _ in 1:(tsteps - 1)]
    H_super = [copy(H_super_entry) for _ in 1:(tsteps - 1)]

    # Fill diagonal blocks
    H_diag[1] .= yt_given_xt .+ xt1_given_xt .+ x_t
    for t in 2:(tsteps - 1)
        H_diag[t] .= yt_given_xt .+ xt_given_xt_1 .+ xt1_given_xt
    end
    H_diag[tsteps] .= yt_given_xt .+ xt_given_xt_1

    return H_diag, H_super, H_sub
end

function _compute_hessian_blocks(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    tsteps = size(y, 2)
    latent_dim = lds.latent_dim
    obs_dim = lds.obs_dim

    A = lds.state_model.A
    C = lds.obs_model.C
    log_d = lds.obs_model.log_d
    d = exp.(log_d)

    # Pre-compute Cholesky factorizations
    Q_chol = cholesky(Symmetric(lds.state_model.Q))
    P0_chol = cholesky(Symmetric(lds.state_model.P0))

    # Compute block templates for state model
    H_sub_entry = Q_chol \ A
    H_super_entry = permutedims(H_sub_entry)

    I_mat = Matrix{T}(I, latent_dim, latent_dim)
    xt_given_xt_1 = -(Q_chol \ I_mat)
    xt1_given_xt = -A' * (Q_chol \ A)
    x_t = -(P0_chol \ I_mat)

    Q_middle = xt1_given_xt + xt_given_xt_1
    Q_first = x_t + xt1_given_xt
    Q_last = xt_given_xt_1

    # Allocate blocks
    H_diag = [Matrix{T}(undef, latent_dim, latent_dim) for _ in 1:tsteps]
    H_sub = [copy(H_sub_entry) for _ in 1:(tsteps - 1)]
    H_super = [copy(H_super_entry) for _ in 1:(tsteps - 1)]

    # Temporaries for Poisson observation term
    λ = Vector{T}(undef, obs_dim)
    z = Vector{T}(undef, obs_dim)

    @views for t in 1:tsteps
        # Start with state model contribution
        if t == 1
            copyto!(H_diag[t], Q_first)
        elseif t == tsteps
            copyto!(H_diag[t], Q_last)
        else
            copyto!(H_diag[t], Q_middle)
        end

        # Add Poisson observation term: -C' * diag(λ) * C
        mul!(z, C, x[:, t])
        @. λ = exp(z + d)

        for j in 1:latent_dim, i in 1:latent_dim
            acc = zero(T)
            for k in 1:obs_dim
                acc += C[k, i] * λ[k] * C[k, j]
            end
            H_diag[t][i, j] -= acc
        end
    end

    return H_diag, H_super, H_sub
end

function smooth!(
    slds::SLDS{T},
    fs::FilterSmooth{T},
    y::AbstractMatrix{T},
    w::AbstractMatrix{T};
    ws::Union{Nothing,SLDSSmoothWorkspace{T}}=nothing,
    max_iter::Int=20,
    tol::T=T(1e-6),
    linesearch::Union{Nothing,AbstractLineSearch}=BackTrackingLS{T}(),
) where {T<:Real}
    latent_dim = slds.LDSs[1].latent_dim
    tsteps = size(y, 2)

    ws === nothing && (ws = SLDSSmoothWorkspace(T, slds, tsteps))
    btd = ws.btd

    # working buffer
    x = fs.x_smooth

    # init (warm start if available)
    if all(fs.E_z .== 0)
        # crude prior init using first mode's dynamics
        lds1 = slds.LDSs[1]
        x[:, 1] .= lds1.state_model.x0
        for t in 2:tsteps
            mul!(view(x, :, t), lds1.state_model.A, view(x, :, t - 1))
            x[:, t] .+= lds1.state_model.b
        end
    else
        copyto!(x, fs.E_z)
    end

    # gradient buffer and direction buffer
    g = ws.grad_buf                       # D×T
    p = reshape(ws.X₀, latent_dim, tsteps)  # D×T (view)

    # objective at current x (allocation-free)
    ϕ!() = begin
        loglikelihood!(ws, slds, x, y, w)  # fills ws.ll_vec
        return sum(ws.ll_vec)
    end

    # gradient callback: g <- ∇ loglik
    compute_grad! = (gcur, xcur) -> begin
        # xcur is x (we mutate x in-place); Gradient! writes into ws.grad_buf
        Gradient!(ws, slds, y, xcur, w)
        copyto!(gcur, ws.grad_buf)
        return nothing
    end

    # Hessian builder: fill blocks for ∇² loglik, then negate to form (-Hℓ) SPD
    build_hess! = (xcur) -> begin
        Hessian_blocks!(ws, slds, y, xcur, w)  # fills btd.H_* for loglik
        _negate_blocks!(btd)                    # fills btd.neg_* = -(Hℓ blocks)
        return nothing
    end

    # Solve (-Hℓ) p = g  (Newton ascent direction)
    solve_dir! =
        (pcur, gcur) -> begin
            gvec = vec(gcur)
            pvec = vec(pcur)
            copyto!(pvec, gvec)
            block_tridiagonal_solve!(
                pvec, btd.neg_sub, btd.neg_diag, btd.neg_super, gvec, btd
            )
            return nothing
        end

    converged = newton_smooth!(
        Val(:max),
        x,
        g,
        p,
        compute_grad!,
        build_hess!,
        solve_dir!,
        ϕ!,
        linesearch;
        max_iter=max_iter,
        tol=tol,
    )

    # Posterior covariances from Laplace approx:
    # precision = -∇² loglik at MAP
    Hessian_blocks!(ws, slds, y, x, w)
    _negate_blocks!(btd)

    block_tridiagonal_inverse!(
        fs.p_smooth, fs.p_smooth_tt1, btd.neg_sub, btd.neg_diag, btd.neg_super, btd
    )

    @views for t in 1:tsteps
        fs.p_smooth[:, :, t] .= 0.5 .* (fs.p_smooth[:, :, t] .+ fs.p_smooth[:, :, t]')
    end

    return fs
end

# Public API wrapper
function smooth(slds::SLDS, y::AbstractMatrix{T}, w::AbstractMatrix{T}) where {T<:Real}
    fs = initialize_FilterSmooth(slds.LDSs[1], size(y, 2))
    smooth!(slds, fs, y, w)
    return fs.x_smooth, fs.p_smooth
end

"""
    sample_posterior!(x_out, rng, tfs, randn_buf)

In-place version of `sample_posterior` for multi-trial use inside EM loops.
Fills the pre-allocated `x_out` (latent_dim × tsteps × ntrials) with one posterior
sample per trial. `randn_buf` (latent_dim) is a scratch vector reused across timesteps.
"""
function sample_posterior!(
    x_out::AbstractArray{T,3},
    rng::AbstractRNG,
    tfs::TrialFilterSmooth{T},
    randn_buf::Vector{T},
) where {T<:Real}
    min_jitter = T(1e-8)
    tsteps = size(tfs[1].x_smooth, 2)
    ntrials = length(tfs.FilterSmooths)

    for trial in 1:ntrials
        fs = tfs[trial]
        x_trial = view(x_out, :, :, trial)
        for t in 1:tsteps
            chol = nothing
            jitter = zero(T)
            for attempt in 1:5
                try
                    chol = cholesky(Symmetric(fs.p_smooth[:, :, t]) + jitter * I)
                    break
                catch
                    jitter = min_jitter * T(10)^(attempt - 1)
                    if attempt == 5
                        @warn "Covariance not positive definite at t=$t trial=$trial, jitter=$jitter"
                        chol = cholesky(Symmetric(fs.p_smooth[:, :, t]) + jitter * I)
                    end
                end
            end
            randn!(rng, randn_buf)
            lmul!(chol.L, randn_buf)
            x_trial[:, t] .= fs.x_smooth[:, t] .+ randn_buf
        end
    end
    return x_out
end

"""
    sample_posterior(rng::AbstractRNG, fs::FilterSmooth{T}) where {T<:Real}

Sample a trajectory from the posterior over continuous states and compute its entropy.

Returns:
- x_sample: matrix of size (latent_dim, tsteps) representing one sample from
  q(x) = ∏ₜ N(x_t | x_smooth_t, p_smooth_t)
- entropy: H[q(x)] = ∑ₜ H[N(x_t | x_smooth_t, p_smooth_t)]
"""
function sample_posterior(rng::AbstractRNG, fs::FilterSmooth{T}) where {T<:Real}
    latent_dim, tsteps = size(fs.x_smooth)
    x_sample = similar(fs.x_smooth)
    entropy = zero(T)
    min_jitter = T(1e-8)

    for t in 1:tsteps
        μ = fs.x_smooth[:, t]
        Σ = Symmetric(fs.p_smooth[:, :, t])

        # Try Cholesky decomposition with increasing jitter if needed
        chol = nothing
        jitter = zero(T)
        max_attempts = 5

        for attempt in 1:max_attempts
            try
                chol = cholesky(Σ + jitter * I)
                break
            catch
                if attempt == max_attempts
                    # Last resort: use larger jitter
                    jitter = min_jitter * T(10)^(attempt-1)
                    @warn "Covariance matrix not positive definite at t=$t, adding jitter=$jitter"
                    chol = cholesky(Σ + jitter * I)
                else
                    # Increase jitter and try again
                    jitter = min_jitter * T(10)^(attempt-1)
                end
            end
        end

        # Sample using the Cholesky factor
        Σ_chol = chol.L
        x_sample[:, t] = μ + Σ_chol * randn(rng, T, latent_dim)

        # Accumulate entropy using log determinant from Cholesky
        # log|Σ| = 2 * sum(log(diag(L))) where Σ = L*L'
        logdet_Σ = 2 * sum(log, diag(Σ_chol))
        entropy += 0.5 * (latent_dim * (1 + log(2π)) + logdet_Σ)
    end

    return x_sample, entropy
end

# Convenience method
function sample_posterior(fs::FilterSmooth{T}) where {T<:Real}
    return sample_posterior(Random.GLOBAL_RNG, fs)
end

"""
    sample_posterior(rng::AbstractRNG, tfs::TrialFilterSmooth{T}, nsamples::Int=1) where {T<:Real}

Sample trajectories from the posterior for multiple trials and compute entropies.

Returns:
- samples: array of size (latent_dim, tsteps, ntrials, nsamples)
- entropies: matrix of size (ntrials, nsamples) containing entropy for each sample
"""
function sample_posterior(
    rng::AbstractRNG, tfs::TrialFilterSmooth{T}, nsamples::Int=1
) where {T<:Real}
    ntrials = length(tfs.FilterSmooths)
    latent_dim, tsteps = size(tfs[1].x_smooth)

    samples = Array{T,4}(undef, latent_dim, tsteps, ntrials, nsamples)
    entropies = Matrix{T}(undef, ntrials, nsamples)

    for trial in 1:ntrials
        for s in 1:nsamples
            samples[:, :, trial, s], entropies[trial, s] = sample_posterior(rng, tfs[trial])
        end
    end

    return samples, entropies
end

function sample_posterior(tfs::TrialFilterSmooth{T}, nsamples::Int=1) where {T<:Real}
    return sample_posterior(Random.GLOBAL_RNG, tfs, nsamples)
end

"""
    estep!(slds, tfs, fb_storages, dl, y, x_samples)

E-step for SLDS using a single sample from the continuous posterior.
- Fills `dl.logL` with per-state log-likelihoods from sampled continuous states
- Runs forward-backward (external HiddenMarkovModels.jl backend) to get discrete posteriors
- Smooths continuous states given discrete posteriors
- Computes sufficient statistics and returns the stochastic ELBO estimate
"""
function estep!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storages::AbstractVector{<:HMMs.ForwardBackwardStorage},
    dl::SLDSDiscreteLayer{T},
    y::AbstractArray{T,3},
    x_samples::AbstractArray{T,3},  # (latent_dim, tsteps, ntrials)
    slds_ws::SLDSSmoothWorkspace{T},
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    ntrials = size(y, 3)
    K = length(slds.LDSs)
    tsteps = size(y, 2)
    obs_seq = 1:tsteps
    ctrl_seq = fill(nothing, tsteps)

    total_elbo = zero(T)

    for trial in 1:ntrials
        y_trial = view(y, :, :, trial)
        x_sample = view(x_samples, :, :, trial)
        fbs = fb_storages[trial]

        # Fill pre-computed log-likelihoods for the discrete layer
        for k in 1:K
            loglikelihood!(slds_ws.ll_tmp, slds_ws, slds_ws.consts[k], slds.LDSs[k], x_sample, y_trial)
            dl.logL[k, :] .= slds_ws.ll_tmp
        end

        # Forward-backward via external backend
        HMMs.forward_backward!(
            fbs, dl, obs_seq, ctrl_seq; seq_ends=(tsteps,), transition_marginals=true
        )

        # γ[k,t] is already a probability (not log); use directly as weights
        w = fbs.γ  # K×T

        # Smooth continuous states given discrete posteriors
        smooth!(slds, tfs[trial], y_trial, w; ws=slds_ws)
        sufficient_statistics!(tfs[trial])

        trial_elbo = zero(T)
        x_smooth_trial = tfs[trial].x_smooth

        # E_q[log p(y,x|z)] weighted by discrete posteriors
        for k in 1:K
            loglikelihood!(slds_ws.ll_tmp, slds_ws, slds_ws.consts[k], slds.LDSs[k], x_smooth_trial, y_trial)
            for t in 1:tsteps
                trial_elbo += w[k, t] * slds_ws.ll_tmp[t]
            end
        end

        # log p(z_1)
        for k in 1:K
            trial_elbo += w[k, 1] * log(slds.πₖ[k] + T(1e-12))
        end

        # log p(z_t | z_{t-1}): sum_t sum_{i,j} ξ[t][i,j] * log A[i,j]
        for t in 1:(tsteps - 1)
            for i in 1:K, j in 1:K
                trial_elbo += fbs.ξ[t][i, j] * log(slds.A[i, j] + T(1e-12))
            end
        end

        # Subtract entropies
        trial_elbo -= tfs[trial].entropy  # H[q(x)]
        for k in 1:K, t in 1:tsteps
            wkt = w[k, t]
            wkt > 0 && (trial_elbo += wkt * log(wkt + T(1e-12)))  # -H[q(z)] = +E[log q(z)]
        end

        total_elbo += trial_elbo
    end

    return total_elbo
end

"""
    mstep!(slds, tfs, fb_storages, y, sws)

M-step for SLDS.
- Updates discrete parameters (A, πₖ) from aggregated ForwardBackwardStorage statistics
- Updates each LDS using weighted sufficient statistics
"""
function mstep!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storages::AbstractVector{<:HMMs.ForwardBackwardStorage},
    y::AbstractArray{T,3},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    K = length(slds.LDSs)
    ntrials = size(y, 3)
    tsteps = size(y, 2)

    fill!(slds.πₖ, zero(T))
    for trial in 1:ntrials
        slds.πₖ .+= view(fb_storages[trial].γ, :, 1)
    end
    slds.πₖ ./= ntrials

    # A[i,j] = sum_trial sum_{t<T} ξ[t][i,j] / sum_trial sum_{t<T} γ[i,t]
    num = zeros(T, K, K)
    denom = zeros(T, K)
    for trial in 1:ntrials
        fbs = fb_storages[trial]
        for t in 1:(tsteps - 1)
            num .+= fbs.ξ[t]
            denom .+= view(fbs.γ, :, t)
        end
    end
    for i in 1:K
        slds.A[i, :] .= num[i, :] ./ max(denom[i], T(1e-12))
    end

    weights = Vector{AbstractVector{T}}(undef, ntrials)
    for k in 1:K
        for trial in 1:ntrials
            weights[trial] = view(fb_storages[trial].γ, k, :)
        end
        mstep!(slds.LDSs[k], tfs, y, sws, weights)
    end

    return nothing
end

function mstep!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storages::AbstractVector{<:HMMs.ForwardBackwardStorage},
    y::AbstractArray{T,3},
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    sws = SmoothWorkspace(T, slds.LDSs[1].latent_dim, slds.LDSs[1].obs_dim, size(y, 2))
    return mstep!(slds, tfs, fb_storages, y, sws)
end

"""
    fit!(slds::SLDS{T,S,O}, y::AbstractArray{T,3}; max_iter::Int=50, progress::Bool=true)

Fit SLDS using variational Laplace EM with stochastic ELBO estimates.
Runs for exactly `max_iter` iterations.
"""
function fit!(
    slds::SLDS{T,S,O}, y::AbstractArray{T,3}; max_iter::Int=50, progress::Bool=true
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    ntrials = size(y, 3)
    tsteps = size(y, 2)
    K = length(slds.LDSs)
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim

    # Continuous-state smoother storage
    tfs = initialize_FilterSmooth(slds.LDSs[1], tsteps, ntrials)

    # Discrete-layer wrapper (holds references to slds.A and slds.πₖ)
    dl = SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(T, K, tsteps))

    # Pre-allocate ForwardBackwardStorage for each trial (reused each iteration)
    fb_storages = [_make_slds_fb_storage(dl, tsteps) for _ in 1:ntrials]

    # Workspaces — allocated once, reused every iteration
    sws = SmoothWorkspace(T, latent_dim, obs_dim, tsteps)
    slds_ws = SLDSSmoothWorkspace(T, slds, tsteps)
    x_samples = Array{T,3}(undef, latent_dim, tsteps, ntrials)
    randn_buf = Vector{T}(undef, latent_dim)

    prog = progress ? Progress(max_iter; desc="Fitting SLDS via EM...", barlen=50, showspeed=true) : nothing
    elbos = Vector{T}(undef, max_iter)

    # Warm-start: smooth once with uniform discrete weights
    w_uniform = fill(one(T) / K, K, tsteps)
    for trial in 1:ntrials
        smooth!(slds, tfs[trial], y[:, :, trial], w_uniform; ws=slds_ws)
    end

    for iter in 1:max_iter
        sample_posterior!(x_samples, Random.default_rng(), tfs, randn_buf)

        elbo = estep!(slds, tfs, fb_storages, dl, y, x_samples, slds_ws)
        elbos[iter] = elbo

        mstep!(slds, tfs, fb_storages, y, sws)
        refresh_slds_constants!(slds_ws, slds)

        if prog !== nothing
            next!(prog; showvalues=[(:iteration, iter), (:ELBO, elbo)])
        end
    end

    if prog !== nothing
        finish!(prog)
    end
    return elbos
end

# Deprecation wrappers for old API
function Gradient(
    slds::SLDS{T},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
    w::AbstractMatrix{T};
    ws::Union{Nothing,SLDSSmoothWorkspace{T}}=nothing,
) where {T<:Real}
    ws === nothing && (ws = SLDSSmoothWorkspace(T, slds, size(y, 2)))
    Gradient!(ws, slds, y, x, w)
    return copy(ws.grad_buf)
end

function Hessian(
    slds::SLDS{T},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
    w::AbstractMatrix{T};
    ws::Union{Nothing,SLDSSmoothWorkspace{T}}=nothing,
) where {T<:Real}
    latent_dim, tsteps = size(x)

    ws === nothing && (ws = SLDSSmoothWorkspace(T, slds, tsteps))

    Hessian_blocks!(ws, slds, y, x, w)

    H_diag = [zeros(T, latent_dim, latent_dim) for _ in 1:tsteps]
    H_sub = [zeros(T, latent_dim, latent_dim) for _ in 1:(tsteps - 1)]
    H_super = [zeros(T, latent_dim, latent_dim) for _ in 1:(tsteps - 1)]

    for t in 1:tsteps
        copyto!(H_diag[t], ws.btd.H_diag[t])
    end
    for t in 1:(tsteps - 1)
        copyto!(H_sub[t], ws.btd.H_sub[t])
        copyto!(H_super[t], ws.btd.H_super[t])
    end

    return H_diag, H_super, H_sub
end
