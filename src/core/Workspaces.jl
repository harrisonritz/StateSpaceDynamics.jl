# Block Tridiagonal Workspace
"""
    BlockTridiagonalWorkspace{T<:Real}

Pre-allocated workspace for block tridiagonal operations in LDS smoothing.
Holds all temporary buffers needed by `Hessian!`, `block_tridgm!`, and
`block_tridiagonal_inverse!` to avoid repeated allocations during EM iterations.
"""
struct BlockTridiagonalWorkspace{T<:Real}
    block_size::Int
    n_blocks::Int

    # Hessian block storage
    H_diag::Vector{Matrix{T}}
    H_sub::Vector{Matrix{T}}
    H_super::Vector{Matrix{T}}

    # Sparse matrix with fixed sparsity pattern
    H_sparse::SparseMatrixCSC{T,Int}
    # Precomputed map from block (k, i, j) to nzval index for zero-allocation writes
    nzval_map::Vector{Int}

    # block_tridiagonal_inverse buffers
    D::Vector{Matrix{T}}
    E::Vector{Matrix{T}}
    M::Matrix{T}
    term1::Matrix{T}
    term2::Matrix{T}
    S::Matrix{T}
    Ibs::Matrix{T}
    Z::Matrix{T}

    # Negated blocks
    neg_diag::Vector{Matrix{T}}
    neg_sub::Vector{Matrix{T}}
    neg_super::Vector{Matrix{T}}
end

"""
    BlockTridiagonalWorkspace(::Type{T}, block_size::Int, n_blocks::Int)

Construct a preallocated workspace for block tridiagonal operations with the
given block size (latent dimension) and number of blocks (timesteps).
"""
function BlockTridiagonalWorkspace(
    ::Type{T}, block_size::Int, n_blocks::Int
) where {T<:Real}
    H_diag = [zeros(T, block_size, block_size) for _ in 1:n_blocks]
    H_sub = [zeros(T, block_size, block_size) for _ in 1:(n_blocks - 1)]
    H_super = [zeros(T, block_size, block_size) for _ in 1:(n_blocks - 1)]

    H_sparse = _build_block_tridiag_pattern(T, block_size, n_blocks)
    nzval_map = _build_nzval_map(H_sparse, block_size, n_blocks)

    D = [zeros(T, block_size, block_size) for _ in 1:(n_blocks + 1)]
    E = [zeros(T, block_size, block_size) for _ in 1:(n_blocks + 1)]
    M = zeros(T, block_size, block_size)
    term1 = zeros(T, block_size, block_size)
    term2 = zeros(T, block_size, block_size)
    S = zeros(T, block_size, block_size)
    Ibs = Matrix{T}(I, block_size, block_size)
    Z = zeros(T, block_size, block_size)

    neg_diag = [zeros(T, block_size, block_size) for _ in 1:n_blocks]
    neg_sub = [zeros(T, block_size, block_size) for _ in 1:(n_blocks - 1)]
    neg_super = [zeros(T, block_size, block_size) for _ in 1:(n_blocks - 1)]

    return BlockTridiagonalWorkspace{T}(
        block_size,
        n_blocks,
        H_diag,
        H_sub,
        H_super,
        H_sparse,
        nzval_map,
        D,
        E,
        M,
        term1,
        term2,
        S,
        Ibs,
        Z,
        neg_diag,
        neg_sub,
        neg_super,
    )
end

"""
    SmoothWorkspace{T<:Real}

Pre-allocated workspace for the full LDS smoothing + EM pipeline.
Houses a `BlockTridiagonalWorkspace` for block tridiagonal operations, plus all
buffers needed by `loglikelihood!`, `Gradient!`, `Hessian!`, and M-step updates
to avoid repeated allocations during EM iterations.
"""
struct SmoothWorkspace{T<:Real}
    # Sub-workspace for block tridiagonal operations
    btd::BlockTridiagonalWorkspace{T}

    # cholesky buffers
    R_buf::Matrix{T}      # (obs_dim × obs_dim)
    Q_buf::Matrix{T}      # (latent_dim × latent_dim)
    P0_buf::Matrix{T}     # (latent_dim × latent_dim)

    # Cached upper-tri Chol factors
    R_chol_U::Matrix{T}       # (obs_dim × obs_dim)
    Q_chol_U::Matrix{T}       # (latent_dim × latent_dim)
    P0_chol_U::Matrix{T}      # (latent_dim × latent_dim)

    # Solve Outputs
    tmp_RC::Matrix{T}    # obs_dim × latent_dim   (R^{-1} C)
    tmp_QA::Matrix{T}    # latent_dim × latent_dim (Q^{-1} A)

    # Derived terms for Gradient
    C_inv_R::Matrix{T}        # (R_chol \ C)' = C'inv(R), size (latent_dim × obs_dim)
    A_inv_Q::Matrix{T}        # (Q_chol \ A)' = A'inv(Q), size (latent_dim × latent_dim)

    # Derived terms for Hessian block templates
    H_sub_entry::Matrix{T}    # Q_chol \ A, size (latent_dim × latent_dim)
    H_super_entry::Matrix{T}  # H_sub_entry', size (latent_dim × latent_dim)
    yt_given_xt::Matrix{T}    # -C'*(R_chol \ C), size (latent_dim × latent_dim)
    xt_given_xt_1::Matrix{T}  # -(Q_chol \ I), size (latent_dim × latent_dim)
    xt1_given_xt::Matrix{T}   # -A'*(Q_chol \ A), size (latent_dim × latent_dim)
    x_t::Matrix{T}            # -(P0_chol \ I), size (latent_dim × latent_dim)

    # Optimizer buffers (reused across EM iterations) 
    X₀::Vector{T}             # Vectorized latent path (latent_dim * tsteps)
    grad_buf::Matrix{T}       # Gradient output buffer (latent_dim × tsteps)
    grad_vec::Vector{T}       # Vectorized gradient for TwiceDifferentiable (latent_dim * tsteps)
    initial_h::SparseMatrixCSC{T,Int}  # Sparse Hessian (avoid copy each iteration)

    # Gradient temp vectors
    dxt::Vector{T}            # (latent_dim,)
    dxt_next::Vector{T}       # (latent_dim,)
    dyt::Vector{T}            # (obs_dim,)
    tmp1::Vector{T}           # (latent_dim,)
    tmp2::Vector{T}           # (latent_dim,)
    tmp3::Vector{T}           # (latent_dim,)

    # loglikelihood temp vectors
    ll_vec::Vector{T}         # (tsteps,)
    temp_dx::Vector{T}        # (latent_dim,)
    temp_dy::Vector{T}        # (obs_dim,)
    temp_solve_Q::Vector{T}   # (latent_dim,)
    temp_solve_R::Vector{T}   # (obs_dim,)

    # Hessian temp matrix
    I_mat::Matrix{T}          # Identity (latent_dim × latent_dim)

    # M-step buffers
    Sxz::Matrix{T}            # (latent_dim × latent_dim+1) for update_A_b!
    Szz_Ab::Matrix{T}         # (latent_dim+1 × latent_dim+1) for update_A_b!
    AB::Matrix{T}             # (latent_dim × latent_dim+1) for update_A_b!
    Syz::Matrix{T}            # (obs_dim × latent_dim+1) for update_C_d!
    Szz_Cd::Matrix{T}         # (latent_dim+1 × latent_dim+1) for update_C_d!
    Q_sum::Matrix{T}          # (latent_dim × latent_dim)
    R_sum::Matrix{T}          # (obs_dim × obs_dim)
    S0_sum::Matrix{T}         # (latent_dim × latent_dim)

    # update_Q! temps
    temp_Q1::Matrix{T}        # (latent_dim × latent_dim)
    temp_Q2::Matrix{T}        # (latent_dim × latent_dim)
    temp_Q3::Matrix{T}        # (latent_dim × latent_dim)
    temp_Q4::Matrix{T}        # (latent_dim × latent_dim)
    temp_Q5::Vector{T}        # (latent_dim,)
    bbT::Matrix{T}           # (latent_dim × latent_dim)
    innovation_cov::Matrix{T} # (latent_dim × latent_dim)

    # update_R! temps
    innovation::Vector{T}     # (obs_dim,)
    Czt::Vector{T}            # (obs_dim,)
    temp_R_matrix::Matrix{T}  # (obs_dim × latent_dim)
    outer_product::Matrix{T}  # (latent_dim × latent_dim)
    state_uncertainty::Matrix{T} # (latent_dim × latent_dim)

    # update_C_d! temps
    work_yz::Matrix{T}        # (obs_dim × latent_dim)
    work_outer::Matrix{T}     # (latent_dim × latent_dim)
    CD::Matrix{T}             # (obs_dim × latent_dim + 1)

    # ELBO / Q_state buffers
    elbo_temp::Matrix{T}           # (latent_dim × latent_dim) - main accumulator
    elbo_sum_E_zz::Matrix{T}       # (latent_dim × latent_dim)
    elbo_sum_E_zzm1::Matrix{T}     # (latent_dim × latent_dim)
    elbo_sum_E_cross::Matrix{T}    # (latent_dim × latent_dim)
    elbo_sum_mu_t::Vector{T}       # (latent_dim,)
    elbo_sum_mu_tm1::Vector{T}     # (latent_dim,)
    elbo_temp2::Matrix{T}          # (latent_dim × latent_dim) - for A * sum_E_zzm1 * A'

    # Q_obs buffers (Gaussian)
    elbo_obs_temp::Matrix{T}       # (obs_dim × obs_dim) - accumulator
    elbo_obs_work::Matrix{T}       # (obs_dim × obs_dim) - work matrix
    elbo_ytil::Vector{T}           # (obs_dim,) - residualized y
    elbo_sum_yy::Matrix{T}         # (obs_dim × obs_dim)
    elbo_sum_yz::Matrix{T}         # (obs_dim × latent_dim)
    elbo_obs_work1::Matrix{T}      # (obs_dim × obs_dim)
    elbo_obs_work2::Matrix{T}      # (latent_dim × obs_dim)

    # Q_obs buffers (Poisson)
    h_obs::Vector{T}               # (obs_dim,) - h_t = C * E[x_t] + d
    rho_obs::Vector{T}             # (obs_dim,) - variance correction
    CP_obs::Matrix{T}              # (obs_dim × latent_dim) - C * P_t
    CEz_obs::Vector{T}             # (obs_dim,) - C * E[x_t]
end

"""
    SmoothWorkspace(::Type{T}, latent_dim::Int, obs_dim::Int, tsteps::Int)

Construct a preallocated `SmoothWorkspace` for the full LDS EM pipeline.
"""
function SmoothWorkspace(
    ::Type{T}, latent_dim::Int, obs_dim::Int, tsteps::Int
) where {T<:Real}
    btd = BlockTridiagonalWorkspace(T, latent_dim, tsteps)

    # Pre-computed constant terms (will be filled by compute_smooth_constants!)
    R_buf = zeros(T, obs_dim, obs_dim)
    Q_buf = zeros(T, latent_dim, latent_dim)
    P0_buf = zeros(T, latent_dim, latent_dim)

    R_chol_U = zeros(T, obs_dim, obs_dim)
    Q_chol_U = zeros(T, latent_dim, latent_dim)
    P0_chol_U = zeros(T, latent_dim, latent_dim)

    tmp_RC = zeros(T, obs_dim, latent_dim)
    tmp_QA = zeros(T, latent_dim, latent_dim)

    C_inv_R = zeros(T, latent_dim, obs_dim)
    A_inv_Q = zeros(T, latent_dim, latent_dim)
    H_sub_entry = zeros(T, latent_dim, latent_dim)
    H_super_entry = zeros(T, latent_dim, latent_dim)
    yt_given_xt = zeros(T, latent_dim, latent_dim)
    xt_given_xt_1 = zeros(T, latent_dim, latent_dim)
    xt1_given_xt = zeros(T, latent_dim, latent_dim)
    x_t = zeros(T, latent_dim, latent_dim)

    # Optimizer buffers
    X₀ = zeros(T, latent_dim * tsteps)
    grad_buf = zeros(T, latent_dim, tsteps)
    grad_vec = zeros(T, latent_dim * tsteps)
    initial_h = copy(btd.H_sparse)

    # Gradient temp vectors
    dxt = zeros(T, latent_dim)
    dxt_next = zeros(T, latent_dim)
    dyt = zeros(T, obs_dim)
    tmp1 = zeros(T, latent_dim)
    tmp2 = zeros(T, latent_dim)
    tmp3 = zeros(T, latent_dim)

    # loglikelihood temp vectors
    ll_vec = zeros(T, tsteps)
    temp_dx = zeros(T, latent_dim)
    temp_dy = zeros(T, obs_dim)
    temp_solve_Q = zeros(T, latent_dim)
    temp_solve_R = zeros(T, obs_dim)

    # Hessian temp matrix
    I_mat = Matrix{T}(I, latent_dim, latent_dim)

    # M-step buffers
    Sxz = zeros(T, latent_dim, latent_dim + 1)
    Szz_Ab = zeros(T, latent_dim + 1, latent_dim + 1)
    AB = zeros(T, latent_dim, latent_dim + 1)
    Syz = zeros(T, obs_dim, latent_dim + 1)
    Szz_Cd = zeros(T, latent_dim + 1, latent_dim + 1)
    Q_sum = zeros(T, latent_dim, latent_dim)
    R_sum = zeros(T, obs_dim, obs_dim)
    S0_sum = zeros(T, latent_dim, latent_dim)
    temp_Q1 = zeros(T, latent_dim, latent_dim)
    temp_Q2 = zeros(T, latent_dim, latent_dim)
    temp_Q3 = zeros(T, latent_dim, latent_dim)
    temp_Q4 = zeros(T, latent_dim, latent_dim)
    temp_Q5 = zeros(T, latent_dim)
    bbT = zeros(T, latent_dim, latent_dim)
    innovation_cov = zeros(T, latent_dim, latent_dim)
    innovation = zeros(T, obs_dim)
    Czt = zeros(T, obs_dim)
    temp_R_matrix = zeros(T, obs_dim, latent_dim)
    outer_product = zeros(T, latent_dim, latent_dim)
    state_uncertainty = zeros(T, latent_dim, latent_dim)
    work_yz = zeros(T, obs_dim, latent_dim)
    work_outer = zeros(T, latent_dim, latent_dim)
    CD = zeros(T, obs_dim, latent_dim + 1)

    # ELBO / Q_state buffers
    elbo_temp = zeros(T, latent_dim, latent_dim)
    elbo_sum_E_zz = zeros(T, latent_dim, latent_dim)
    elbo_sum_E_zzm1 = zeros(T, latent_dim, latent_dim)
    elbo_sum_E_cross = zeros(T, latent_dim, latent_dim)
    elbo_sum_mu_t = zeros(T, latent_dim)
    elbo_sum_mu_tm1 = zeros(T, latent_dim)
    elbo_temp2 = zeros(T, latent_dim, latent_dim)

    # Q_obs buffers (Gaussian)
    elbo_obs_temp = zeros(T, obs_dim, obs_dim)
    elbo_obs_work = zeros(T, obs_dim, obs_dim)
    elbo_ytil = zeros(T, obs_dim)
    elbo_sum_yy = zeros(T, obs_dim, obs_dim)
    elbo_sum_yz = zeros(T, obs_dim, latent_dim)
    elbo_obs_work1 = zeros(T, obs_dim, obs_dim)
    elbo_obs_work2 = zeros(T, latent_dim, obs_dim)

    # Q_obs buffers (Poisson)
    h_obs = zeros(T, obs_dim)
    rho_obs = zeros(T, obs_dim)
    CP_obs = zeros(T, obs_dim, latent_dim)
    CEz_obs = zeros(T, obs_dim)

    return SmoothWorkspace{T}(
        btd,
        R_buf,
        Q_buf,
        P0_buf,
        R_chol_U,
        Q_chol_U,
        P0_chol_U,
        tmp_RC,
        tmp_QA,
        C_inv_R,
        A_inv_Q,
        H_sub_entry,
        H_super_entry,
        yt_given_xt,
        xt_given_xt_1,
        xt1_given_xt,
        x_t,
        X₀,
        grad_buf,
        grad_vec,
        initial_h,
        dxt,
        dxt_next,
        dyt,
        tmp1,
        tmp2,
        tmp3,
        ll_vec,
        temp_dx,
        temp_dy,
        temp_solve_Q,
        temp_solve_R,
        I_mat,
        Sxz,
        Szz_Ab,
        AB,
        Syz,
        Szz_Cd,
        Q_sum,
        R_sum,
        S0_sum,
        temp_Q1,
        temp_Q2,
        temp_Q3,
        temp_Q4,
        temp_Q5,
        bbT,
        innovation_cov,
        innovation,
        Czt,
        temp_R_matrix,
        outer_product,
        state_uncertainty,
        work_yz,
        work_outer,
        CD,
        elbo_temp,
        elbo_sum_E_zz,
        elbo_sum_E_zzm1,
        elbo_sum_E_cross,
        elbo_sum_mu_t,
        elbo_sum_mu_tm1,
        elbo_temp2,
        elbo_obs_temp,
        elbo_obs_work,
        elbo_ytil,
        elbo_sum_yy,
        elbo_sum_yz,
        elbo_obs_work1,
        elbo_obs_work2,
        h_obs,
        rho_obs,
        CP_obs,
        CEz_obs,
    )
end

"""
    compute_smooth_constants!(ws::SmoothWorkspace{T}, lds)

Pre-compute and cache all Cholesky factors and derived terms that are constant
within a single `smooth!` call (i.e., depend only on model parameters, not on x).
Must be called once at the start of each `smooth!` invocation.

Dispatches on the observation model type:
- Gaussian: computes both state and observation model terms.
- Poisson: only computes state model terms (observation terms are x-dependent).
"""
function compute_smooth_constants!(
    ws::SmoothWorkspace{WT}, lds::LinearDynamicalSystem{T,S,O}
) where {WT<:Real,T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    A = lds.state_model.A
    Q = lds.state_model.Q
    P0 = lds.state_model.P0
    C = lds.obs_model.C
    R = lds.obs_model.R

    # Cholesky in-place
    copyto!(ws.R_buf, R)
    Rchol = cholesky!(Symmetric(ws.R_buf))

    copyto!(ws.Q_buf, Q)
    Qchol = cholesky!(Symmetric(ws.Q_buf))

    copyto!(ws.P0_buf, P0)
    P0chol = cholesky!(Symmetric(ws.P0_buf))

    # store U factors
    copyto!(ws.R_chol_U, Rchol.U)
    copyto!(ws.Q_chol_U, Qchol.U)
    copyto!(ws.P0_chol_U, P0chol.U)

    # tmp_RC = R^{-1} C
    copyto!(ws.tmp_RC, C)
    ldiv!(Rchol, ws.tmp_RC)
    copyto!(ws.C_inv_R, ws.tmp_RC')

    # tmp_QA = Q^{-1} A
    copyto!(ws.tmp_QA, A)
    ldiv!(Qchol, ws.tmp_QA)
    copyto!(ws.A_inv_Q, ws.tmp_QA')
    copyto!(ws.H_sub_entry, ws.tmp_QA)
    copyto!(ws.H_super_entry, ws.tmp_QA')

    # yt_given_xt = -C' * (R^{-1} C)
    mul!(ws.yt_given_xt, C', ws.tmp_RC)
    ws.yt_given_xt .*= -one(T)

    # xt_given_xt_1 = -Q^{-1}
    copyto!(ws.xt_given_xt_1, ws.I_mat)
    ldiv!(Qchol, ws.xt_given_xt_1)
    ws.xt_given_xt_1 .*= -one(T)

    # xt1_given_xt = -A' * (Q^{-1} A)
    mul!(ws.xt1_given_xt, A', ws.tmp_QA)
    ws.xt1_given_xt .*= -one(T)

    # x_t = -P0^{-1}
    copyto!(ws.x_t, ws.I_mat)
    ldiv!(P0chol, ws.x_t)
    ws.x_t .*= -one(T)

    return nothing
end

function compute_smooth_constants!(
    ws::SmoothWorkspace{WT}, lds::LinearDynamicalSystem{T,S,O}
) where {WT<:Real,T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    A = lds.state_model.A
    Q = lds.state_model.Q
    P0 = lds.state_model.P0

    # Compute Cholesky factors for state model only
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))

    copyto!(ws.Q_chol_U, Q_chol.U)
    copyto!(ws.P0_chol_U, P0_chol.U)

    # Gradient terms: A_inv_Q = (Q_chol \ A)'
    tmp_QA = Q_chol \ A   # latent_dim × latent_dim
    copyto!(ws.A_inv_Q, tmp_QA')

    # Hessian block templates for state model
    copyto!(ws.H_sub_entry, tmp_QA)          # Q_chol \ A
    copyto!(ws.H_super_entry, tmp_QA')       # (Q_chol \ A)'

    # xt_given_xt_1 = -(Q_chol \ I) = -Q^{-1}
    copyto!(ws.xt_given_xt_1, ws.I_mat)
    ldiv!(Q_chol, ws.xt_given_xt_1)
    ws.xt_given_xt_1 .*= -one(T)

    # xt1_given_xt = -A' * (Q_chol \ A)
    mul!(ws.xt1_given_xt, A', tmp_QA)
    ws.xt1_given_xt .*= -one(T)

    # x_t = -(P0_chol \ I) = -P0^{-1}
    copyto!(ws.x_t, ws.I_mat)
    ldiv!(P0_chol, ws.x_t)
    ws.x_t .*= -one(T)

    return nothing
end

"""
    LDSConstantCache{T}

Cache of Cholesky-derived constants for a single LDS component used in SLDS smoothing.
This mirrors the *constant* parts of `SmoothWorkspace`, but does not include optimizer buffers
or block-tridiagonal storage.
"""
mutable struct LDSConstantCache{T<:Real}
    # Cholesky buffers (for in-place factorization)
    R_buf::Matrix{T}
    Q_buf::Matrix{T}
    P0_buf::Matrix{T}

    # Cached upper factors
    R_chol_U::Matrix{T}
    Q_chol_U::Matrix{T}
    P0_chol_U::Matrix{T}

    # LL constant terms
    cP0::T
    cQ::T
    cR::T

    # Solve outputs / derived terms
    tmp_RC::Matrix{T}     # R^{-1}C  (obs_dim × latent_dim)
    tmp_QA::Matrix{T}     # Q^{-1}A  (latent_dim × latent_dim)

    C_inv_R::Matrix{T}    # (R^{-1}C)'  (latent_dim × obs_dim)
    A_inv_Q::Matrix{T}    # (Q^{-1}A)'  (latent_dim × latent_dim)

    # Hessian templates (state + Gaussian obs)
    H_sub_entry::Matrix{T}       # Q^{-1}A  (latent_dim × latent_dim)
    H_super_entry::Matrix{T}     # (Q^{-1}A)' (latent_dim × latent_dim)
    yt_given_xt::Matrix{T}       # -C'R^{-1}C  (latent_dim × latent_dim)  (Gaussian only; zero for Poisson)
    xt_given_xt_1::Matrix{T}     # -Q^{-1}
    xt1_given_xt::Matrix{T}      # -A'Q^{-1}A
    x_t::Matrix{T}               # -P0^{-1}

    # Model matrices needed for Poisson fast path
    C::Matrix{T}                 # obs_dim × latent_dim
    log_d::Vector{T}             # obs_dim
    d::Vector{T}                 # exp.(log_d) cached once
end

function LDSConstantCache(::Type{T}, latent_dim::Int, obs_dim::Int) where {T<:Real}
    return LDSConstantCache{T}(
        zeros(T, obs_dim, obs_dim),             # R_buf
        zeros(T, latent_dim, latent_dim),       # Q_buf
        zeros(T, latent_dim, latent_dim),       # P0_buf
        zeros(T, obs_dim, obs_dim),             # R_chol_U
        zeros(T, latent_dim, latent_dim),       # Q_chol_U
        zeros(T, latent_dim, latent_dim),       # P0_chol_U
        zero(T),                                # cP0
        zero(T),                                # cQ
        zero(T),                                # cR
        zeros(T, obs_dim, latent_dim),          # tmp_RC
        zeros(T, latent_dim, latent_dim),       # tmp_QA
        zeros(T, latent_dim, obs_dim),          # C_inv_R  (latent_dim × obs_dim)
        zeros(T, latent_dim, latent_dim),       # A_inv_Q  (latent_dim × latent_dim)
        zeros(T, latent_dim, latent_dim),       # H_sub_entry
        zeros(T, latent_dim, latent_dim),       # H_super_entry
        zeros(T, latent_dim, latent_dim),       # yt_given_xt
        zeros(T, latent_dim, latent_dim),       # xt_given_xt_1
        zeros(T, latent_dim, latent_dim),       # xt1_given_xt
        zeros(T, latent_dim, latent_dim),       # x_t
        zeros(T, obs_dim, latent_dim),          # C
        zeros(T, obs_dim),                      # log_d
        zeros(T, obs_dim),                      # d = exp.(log_d)
    )
end

"""
    compute_slds_constants!(cc, lds)

Fill a single-component cache with Cholesky-derived constants.
For Poisson observation models, `yt_given_xt` is left as zero and Poisson terms are handled per-iteration.
"""
function compute_slds_constants!(
    cc::LDSConstantCache{T}, lds::LinearDynamicalSystem{T,S,O}, I_mat::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    A = lds.state_model.A
    Q = lds.state_model.Q
    P0 = lds.state_model.P0

    C = lds.obs_model.C
    cc.C .= C

    obs_dim, latent_dim = size(C)

    # Store log_d if it exists (Poisson); otherwise leave zeros
    if hasproperty(lds.obs_model, :log_d)
        cc.log_d .= getproperty(lds.obs_model, :log_d)
        @. cc.d = exp(cc.log_d)
    else
        fill!(cc.log_d, zero(T))
        fill!(cc.d, zero(T))
    end

    if lds.obs_model isa GaussianObservationModel{T}
        cc.cR = -T(0.5) * (T(obs_dim) * log(T(2π)) + _logdet_from_U(cc.R_chol_U, obs_dim))
    else
        cc.cR = zero(T)  # unused for Poisson
    end

    # Q, P0 cholesky (in-place)
    copyto!(cc.Q_buf, Q)
    Qchol = cholesky!(Symmetric(cc.Q_buf))
    copyto!(cc.Q_chol_U, Qchol.U)

    copyto!(cc.P0_buf, P0)
    P0chol = cholesky!(Symmetric(cc.P0_buf))
    copyto!(cc.P0_chol_U, P0chol.U)

    # tmp_QA = Q^{-1}A, A_inv_Q = (Q^{-1}A)'
    copyto!(cc.tmp_QA, A)
    ldiv!(Qchol, cc.tmp_QA)
    copyto!(cc.A_inv_Q, cc.tmp_QA')
    copyto!(cc.H_sub_entry, cc.tmp_QA)
    copyto!(cc.H_super_entry, cc.tmp_QA')

    # xt_given_xt_1 = -Q^{-1}
    copyto!(cc.xt_given_xt_1, I_mat)
    ldiv!(Qchol, cc.xt_given_xt_1)
    cc.xt_given_xt_1 .*= -one(T)

    # xt1_given_xt = -A'Q^{-1}A = -A' * (Q^{-1}A)
    mul!(cc.xt1_given_xt, A', cc.tmp_QA)
    cc.xt1_given_xt .*= -one(T)

    # x_t = -P0^{-1}
    copyto!(cc.x_t, I_mat)
    ldiv!(P0chol, cc.x_t)
    cc.x_t .*= -one(T)

    # Observation terms only if Gaussian
    if lds.obs_model isa GaussianObservationModel{T}
        R = lds.obs_model.R
        copyto!(cc.R_buf, R)
        Rchol = cholesky!(Symmetric(cc.R_buf))
        copyto!(cc.R_chol_U, Rchol.U)

        # tmp_RC = R^{-1}C, C_inv_R = (R^{-1}C)'
        copyto!(cc.tmp_RC, C)
        ldiv!(Rchol, cc.tmp_RC)
        copyto!(cc.C_inv_R, cc.tmp_RC')

        # yt_given_xt = -C'R^{-1}C
        mul!(cc.yt_given_xt, C', cc.tmp_RC)
        cc.yt_given_xt .*= -one(T)
    else
        fill!(cc.yt_given_xt, zero(T))
        fill!(cc.C_inv_R, zero(T))  # unused for Poisson
    end

    # Now safe to compute constants (state-specific, needed for SLDS)
    cc.cP0 =
        -T(0.5) * (T(latent_dim) * log(T(2π)) + _logdet_from_U(cc.P0_chol_U, latent_dim))
    cc.cQ = -T(0.5) * (T(latent_dim) * log(T(2π)) + _logdet_from_U(cc.Q_chol_U, latent_dim))

    if lds.obs_model isa GaussianObservationModel{T}
        cc.cR = -T(0.5) * (T(obs_dim) * log(T(2π)) + _logdet_from_U(cc.R_chol_U, obs_dim))
    else
        cc.cR = zero(T)
    end

    return nothing
end

"""
    SLDSSmoothWorkspace{T}

Workspace for SLDS smoothing that matches the LDS backend shape:
- Owns a BlockTridiagonalWorkspace (H blocks + sparse + inverse scratch)
- Owns per-component LDSConstantCache objects
- Owns optimizer buffers (X₀, grad, sparse Hessian template)
- Owns Poisson temporary vectors
"""
struct SLDSSmoothWorkspace{T<:Real}
    btd::BlockTridiagonalWorkspace{T}
    I_mat::Matrix{T}

    consts::Vector{LDSConstantCache{T}}

    # Optim buffers
    X₀::Vector{T}
    grad_buf::Matrix{T}
    grad_vec::Vector{T}
    initial_h::SparseMatrixCSC{T,Int}

    # Poisson temporaries (shared)
    z::Vector{T}   # obs_dim
    λ::Vector{T}   # obs_dim

    # temp vectors
    dxt::Vector{T}        # latent_dim
    dxt_next::Vector{T}   # latent_dim
    dyt::Vector{T}        # obs_dim  (Gaussian only)
    tmp1::Vector{T}       # latent_dim
    tmp2::Vector{T}       # latent_dim
    tmp3::Vector{T}       # latent_dim

    # LL buffers
    ll_vec::Vector{T}   # accumulator (length tsteps)
    ll_tmp::Vector{T}   # per-component scratch (length tsteps)
end

function SLDSSmoothWorkspace(::Type{T}, slds::SLDS, tsteps::Int) where {T<:Real}
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim
    K = length(slds.LDSs)

    btd = BlockTridiagonalWorkspace(T, latent_dim, tsteps)
    I_mat = Matrix{T}(I, latent_dim, latent_dim)

    consts = [LDSConstantCache(T, latent_dim, obs_dim) for _ in 1:K]

    X₀ = zeros(T, latent_dim * tsteps)
    grad_buf = zeros(T, latent_dim, tsteps)
    grad_vec = zeros(T, latent_dim * tsteps)
    initial_h = copy(btd.H_sparse)

    z = zeros(T, obs_dim)
    λ = zeros(T, obs_dim)

    dxt = zeros(T, latent_dim)
    dxt_next = zeros(T, latent_dim)
    dyt = zeros(T, obs_dim)
    tmp1 = zeros(T, latent_dim)
    tmp2 = zeros(T, latent_dim)
    tmp3 = zeros(T, latent_dim)

    ll_vec = zeros(T, tsteps)
    ll_tmp = zeros(T, tsteps)

    ws = SLDSSmoothWorkspace{T}(
        btd,
        I_mat,
        consts,
        X₀,
        grad_buf,
        grad_vec,
        initial_h,
        z,
        λ,
        dxt,
        dxt_next,
        dyt,
        tmp1,
        tmp2,
        tmp3,
        ll_vec,
        ll_tmp,
    )

    # Cache constants once
    for k in 1:K
        compute_slds_constants!(ws.consts[k], slds.LDSs[k], ws.I_mat)
    end

    return ws
end

#=
Refresh the per-regime constant caches after an M-step has updated the LDS parameters.
Must be called before the next E-step so that Cholesky factors, Hessian templates, etc.
reflect the current Q, R, A, P0.
=#
function refresh_slds_constants!(ws::SLDSSmoothWorkspace{T}, slds) where {T}
    for k in eachindex(slds.LDSs)
        compute_slds_constants!(ws.consts[k], slds.LDSs[k], ws.I_mat)
    end
    return nothing
end
