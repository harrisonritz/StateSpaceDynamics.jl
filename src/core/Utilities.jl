# Type checking utilities
"""
    check_same_type(args...)

Utility function to check if n arguments share the same types.
"""
function check_same_type(args...)
    if length(args) ≤ 1
        return true  # trivial case
    end

    first_type = typeof(args[1])
    return all(x -> typeof(x) == first_type, args)
end

# Matrix utilities
"""
    block_tridiagonal_inverse(A, B, C)

Compute the inverse of a block tridiagonal matrix.

# Notes: This implementation is from the paper:
"An Accelerated Lambda Iteration Method for Multilevel Radiative Transfer” Rybicki, G.B.,
and Hummer, D.G., Astronomy and Astrophysics, 245, 171–181 (1991), Appendix B.
"""
function block_tridiagonal_inverse(
    A::Vector{<:AbstractMatrix{T}},
    B::Vector{<:AbstractMatrix{T}},
    C::Vector{<:AbstractMatrix{T}},
) where {T<:Real}
    n = length(B)
    bs = size(B[1], 1)

    # Preallocate D and E blocks (all blocks exist and will be overwritten in-place)
    D = [zeros(T, bs, bs) for _ in 1:(n + 1)]
    E = [zeros(T, bs, bs) for _ in 1:(n + 1)]

    # Outputs
    λii = Array{T}(undef, bs, bs, n)
    λij = Array{T}(undef, bs, bs, n-1)

    Ibs = Matrix{T}(I, bs, bs)
    Z = zeros(T, bs, bs)  # reusable "edge" block

    # Work buffers (reused; LU overwrites its input)
    M = zeros(T, bs, bs)
    term1 = zeros(T, bs, bs)
    term2 = zeros(T, bs, bs)
    S = zeros(T, bs, bs)

    for i in 1:n
        Ai = (i == 1) ? Z : A[i - 1]
        Ci = (i <= length(C)) ? C[i] : Z

        copyto!(M, B[i])                         # M = B[i]
        mul!(M, Ai, D[i], -one(T), one(T))       # M = B[i] - Ai*D[i]
        F = lu!(M)                               # in-place LU on M
        ldiv!(D[i + 1], F, Ci)                     # D[i+1] = F \ Ci (no alloc)
    end

    for i in n:-1:1
        Ci = (i <= length(C)) ? C[i] : Z
        Ai = (i == 1) ? Z : A[i - 1]

        copyto!(M, B[i])                         # M = B[i]
        mul!(M, Ci, E[i + 1], -one(T), one(T))     # M = B[i] - Ci*E[i+1]
        F = lu!(M)
        ldiv!(E[i], F, Ai)                       # E[i] = F \ Ai (no alloc)
    end

    for i in 1:n
        # term1 = I - D[i+1]*E[i+1]
        copyto!(term1, Ibs)
        mul!(term1, D[i + 1], E[i + 1], -one(T), one(T))

        # term2 = B[i] - A[i-1]*D[i]
        Ai = (i == 1) ? Z : A[i - 1]
        copyto!(term2, B[i])
        mul!(term2, Ai, D[i], -one(T), one(T))

        # S = term2 * term1
        mul!(S, term2, term1)
        F = lu!(S)

        @views ldiv!(λii[:, :, i], F, Ibs)        # λii[:,:,i] = F \ I
    end

    for i in 2:n
        @views mul!(λij[:, :, i - 1], E[i], λii[:, :, i - 1])
    end

    # avoid allocating -λij
    for k in eachindex(λij)
        λij[k] = -λij[k]
    end

    return λii, λij
end

"""
    block_tridiagonal_inverse_static(A, B, C)

Compute the inverse of a block tridiagonal matrix using static matrices. See
`block_tridiagonal_inverse` for details.
"""
function block_tridiagonal_inverse_static(
    A::Vector{<:AbstractMatrix{T}},
    B::Vector{<:AbstractMatrix{T}},
    C::Vector{<:AbstractMatrix{T}},
    ::Val{N},
) where {T<:Real,N}
    n = length(B)

    # Pre-allocate working matrices (reuse these)
    M = MMatrix{N,N,T}(undef)  # Mutable static matrix for intermediate calculations
    temp = MMatrix{N,N,T}(undef)
    identity_static = MMatrix{N,N,T}(I)
    zero_static = @SMatrix zeros(N, N)

    # Initialize D and E arrays - use mutable static matrices
    D = Vector{SMatrix{N,N,T}}(undef, n + 1)
    E = Vector{SMatrix{N,N,T}}(undef, n + 1)
    D[1] = zero_static
    E[n + 1] = zero_static

    # Pre-allocate output arrays
    λii = Array{T}(undef, N, N, n)
    λij = Array{T}(undef, N, N, n - 1)

    # Forward sweep for D
    for i in 1:n
        # M = B[i] - A_extended[i] * D[i]
        if i == 1
            M .= B[1]  # A_extended[1] is zeros
        else
            mul!(temp, SMatrix{N,N,T}(A[i - 1]), D[i])  # Convert only when needed
            M .= B[i] .- temp
        end

        # D[i + 1] = inv(M) * C_extended[i]
        if i == n
            D[i + 1] = zero_static  # C_extended[n] is zeros
        else
            M_static = SMatrix{N,N,T}(M)
            C_static = SMatrix{N,N,T}(C[i])
            D[i + 1] = M_static \ C_static
        end
    end

    # Backward sweep for E
    for i in n:-1:1
        # M = B[i] - C_extended[i] * E[i + 1]
        if i == n
            M .= B[n]  # C_extended[n] is zeros
        else
            mul!(temp, SMatrix{N,N,T}(C[i]), E[i + 1])
            M .= B[i] .- temp
        end

        # E[i] = inv(M) * A_extended[i]
        if i == 1
            E[i] = zero_static  # A_extended[1] is zeros
        else
            M_static = SMatrix{N,N,T}(M)
            A_static = SMatrix{N,N,T}(A[i - 1])
            E[i] = M_static \ A_static
        end
    end

    # Compute λii
    for i in 1:n
        # term1 = identity - D[i + 1] * E[i + 1]
        mul!(temp, D[i + 1], E[i + 1])
        term1 = identity_static - SMatrix{N,N,T}(temp)

        # term2 = B[i] - A_extended[i] * D[i]
        if i == 1
            term2 = SMatrix{N,N,T}(B[1])
        else
            mul!(temp, SMatrix{N,N,T}(A[i - 1]), D[i])
            term2 = SMatrix{N,N,T}(B[i]) - SMatrix{N,N,T}(temp)
        end

        # S = term2 * term1
        S = term2 * term1
        λii[:, :, i] = Matrix(S \ identity_static)
    end

    # Compute λij
    for i in 2:n
        result = E[i] * SMatrix{N,N,T}(view(λii,:,:,(i - 1)))
        λij[:, :, i - 1] = Matrix(result)
    end

    return λii, -λij
end

"""
    block_tridgm(
        main_diag::Vector{Matrix{T}},
        upper_diag::Vector{Matrix{T}},
        lower_diag::Vector{Matrix{T}}
    ) where {T<:Real}

Construct a block tridiagonal matrix from three vectors of matrices.

# Throws
- `ErrorException` if the lengths of `upper_diag` and `lower_diag` are not one less than the
    length of `main_diag`.
"""
function block_tridgm(
    main_diag::Vector{<:AbstractMatrix{T}},
    upper_diag::Vector{<:AbstractMatrix{T}},
    lower_diag::Vector{<:AbstractMatrix{T}},
) where {T<:Real}
    n = length(main_diag)
    m = size(main_diag[1], 1)
    N = n * m
    total_nnz = n * m * m + 2 * (n - 1) * m * m

    I = Vector{Int}(undef, total_nnz)
    J = Vector{Int}(undef, total_nnz)
    V = Vector{T}(undef, total_nnz)

    idx = 1

    for k in 1:n
        base = (k - 1) * m
        block = main_diag[k]
        for i in 1:m, j in 1:m
            I[idx] = base + i
            J[idx] = base + j
            V[idx] = block[i, j]
            idx += 1
        end
    end

    for k in 1:(n - 1)
        base_k = (k - 1) * m
        base_kp1 = k * m
        block_up = upper_diag[k]
        block_low = lower_diag[k]
        for i in 1:m, j in 1:m
            I[idx] = base_k + i
            J[idx] = base_kp1 + j
            V[idx] = block_up[i, j]
            idx += 1
        end
        for i in 1:m, j in 1:m
            I[idx] = base_kp1 + i
            J[idx] = base_k + j
            V[idx] = block_low[i, j]
            idx += 1
        end
    end

    return sparse(I, J, V, N, N)
end

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
function BlockTridiagonalWorkspace(::Type{T}, block_size::Int, n_blocks::Int) where {T<:Real}
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
        block_size, n_blocks,
        H_diag, H_sub, H_super,
        H_sparse, nzval_map,
        D, E, M, term1, term2, S, Ibs, Z,
        neg_diag, neg_sub, neg_super,
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

    # Pre-computed constant terms (filled by compute_smooth_constants!)
    # Cholesky upper triangular factors
    R_chol_U::Matrix{T}       # (obs_dim × obs_dim)
    Q_chol_U::Matrix{T}       # (latent_dim × latent_dim)
    P0_chol_U::Matrix{T}      # (latent_dim × latent_dim)

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

    # ELBO / Q_state buffers
    elbo_temp::Matrix{T}           # (latent_dim × latent_dim) - main accumulator
    elbo_sum_E_zz::Matrix{T}       # (latent_dim × latent_dim)
    elbo_sum_E_zzm1::Matrix{T}     # (latent_dim × latent_dim)
    elbo_sum_E_cross::Matrix{T}    # (latent_dim × latent_dim)
    elbo_sum_mu_t::Vector{T}       # (latent_dim,)
    elbo_sum_mu_tm1::Vector{T}     # (latent_dim,)
    elbo_temp2::Matrix{T}          # (latent_dim × latent_dim) - for A * sum_E_zzm1 * A'

    # Q_obs buffers
    elbo_obs_temp::Matrix{T}       # (obs_dim × obs_dim) - accumulator
    elbo_obs_work::Matrix{T}       # (obs_dim × obs_dim) - work matrix
    elbo_ytil::Vector{T}           # (obs_dim,) - residualized y
    elbo_sum_yy::Matrix{T}         # (obs_dim × obs_dim)
    elbo_sum_yz::Matrix{T}         # (obs_dim × latent_dim)
    elbo_obs_work1::Matrix{T}      # (obs_dim × obs_dim)
    elbo_obs_work2::Matrix{T}      # (latent_dim × obs_dim)
end

"""
    SmoothWorkspace(::Type{T}, latent_dim::Int, obs_dim::Int, tsteps::Int)

Construct a preallocated `SmoothWorkspace` for the full LDS EM pipeline.
"""
function SmoothWorkspace(::Type{T}, latent_dim::Int, obs_dim::Int, tsteps::Int) where {T<:Real}
    btd = BlockTridiagonalWorkspace(T, latent_dim, tsteps)

    # Pre-computed constant terms (will be filled by compute_smooth_constants!)
    R_chol_U = zeros(T, obs_dim, obs_dim)
    Q_chol_U = zeros(T, latent_dim, latent_dim)
    P0_chol_U = zeros(T, latent_dim, latent_dim)
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
    innovation_cov = zeros(T, latent_dim, latent_dim)
    innovation = zeros(T, obs_dim)
    Czt = zeros(T, obs_dim)
    temp_R_matrix = zeros(T, obs_dim, latent_dim)
    outer_product = zeros(T, latent_dim, latent_dim)
    state_uncertainty = zeros(T, latent_dim, latent_dim)
    work_yz = zeros(T, obs_dim, latent_dim)
    work_outer = zeros(T, latent_dim, latent_dim)

    # ELBO / Q_state buffers
    elbo_temp = zeros(T, latent_dim, latent_dim)
    elbo_sum_E_zz = zeros(T, latent_dim, latent_dim)
    elbo_sum_E_zzm1 = zeros(T, latent_dim, latent_dim)
    elbo_sum_E_cross = zeros(T, latent_dim, latent_dim)
    elbo_sum_mu_t = zeros(T, latent_dim)
    elbo_sum_mu_tm1 = zeros(T, latent_dim)
    elbo_temp2 = zeros(T, latent_dim, latent_dim)

    # Q_obs buffers
    elbo_obs_temp = zeros(T, obs_dim, obs_dim)
    elbo_obs_work = zeros(T, obs_dim, obs_dim)
    elbo_ytil = zeros(T, obs_dim)
    elbo_sum_yy = zeros(T, obs_dim, obs_dim)
    elbo_sum_yz = zeros(T, obs_dim, latent_dim)
    elbo_obs_work1 = zeros(T, obs_dim, obs_dim)
    elbo_obs_work2 = zeros(T, latent_dim, obs_dim)

    return SmoothWorkspace{T}(
        btd,
        R_chol_U, Q_chol_U, P0_chol_U,
        C_inv_R, A_inv_Q,
        H_sub_entry, H_super_entry,
        yt_given_xt, xt_given_xt_1, xt1_given_xt, x_t,
        X₀, grad_buf, grad_vec, initial_h,
        dxt, dxt_next, dyt, tmp1, tmp2, tmp3,
        ll_vec, temp_dx, temp_dy, temp_solve_Q, temp_solve_R,
        I_mat,
        Sxz, Szz_Ab, Syz, Szz_Cd,
        Q_sum, R_sum, S0_sum,
        temp_Q1, temp_Q2, temp_Q3, temp_Q4, temp_Q5, innovation_cov,
        innovation, Czt, temp_R_matrix, outer_product, state_uncertainty,
        work_yz, work_outer,
        elbo_temp, elbo_sum_E_zz, elbo_sum_E_zzm1, elbo_sum_E_cross,
        elbo_sum_mu_t, elbo_sum_mu_tm1, elbo_temp2,
        elbo_obs_temp, elbo_obs_work, elbo_ytil, elbo_sum_yy, elbo_sum_yz,
        elbo_obs_work1, elbo_obs_work2,
    )
end

"""
    compute_smooth_constants!(ws::SmoothWorkspace{T}, lds)

Pre-compute and cache all Cholesky factors and derived terms that are constant
within a single `smooth!` call (i.e., depend only on model parameters, not on x).
Must be called once at the start of each `smooth!` invocation.

For Gaussian observation models, computes both state and observation model terms.
For Poisson observation models, only computes state model terms (observation
terms are x-dependent and computed per-iteration).
"""
function compute_smooth_constants!(
    ws::SmoothWorkspace{T},
    lds,
) where {T<:Real}
    A = lds.state_model.A
    Q = lds.state_model.Q
    P0 = lds.state_model.P0
    C = lds.obs_model.C
    R = lds.obs_model.R

    # Compute Cholesky factors
    R_chol = cholesky(Symmetric(R))
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))

    copyto!(ws.R_chol_U, R_chol.U)
    copyto!(ws.Q_chol_U, Q_chol.U)
    copyto!(ws.P0_chol_U, P0_chol.U)

    # Gradient terms: C_inv_R = (R_chol \ C)' and A_inv_Q = (Q_chol \ A)'
    # R_chol \ C computes inv(R) * C via Cholesky, then transpose
    tmp_RC = R_chol \ C   # obs_dim × latent_dim → but we want (latent_dim × obs_dim)
    copyto!(ws.C_inv_R, tmp_RC')
    tmp_QA = Q_chol \ A   # latent_dim × latent_dim
    copyto!(ws.A_inv_Q, tmp_QA')

    # Hessian block templates
    copyto!(ws.H_sub_entry, tmp_QA)          # Q_chol \ A
    copyto!(ws.H_super_entry, tmp_QA')       # (Q_chol \ A)'

    # yt_given_xt = -C' * (R_chol \ C)
    mul!(ws.yt_given_xt, C', tmp_RC)
    ws.yt_given_xt .*= -one(T)

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
    compute_smooth_constants_poisson!(ws::SmoothWorkspace{T}, lds)

Pre-compute and cache Cholesky factors and derived terms for Poisson LDS smoothing.
Only computes state model terms since the observation terms are x-dependent
and must be computed per-iteration.

Must be called once at the start of each `smooth!` invocation for Poisson LDS.
"""
function compute_smooth_constants_poisson!(
    ws::SmoothWorkspace{T},
    lds,
) where {T<:Real}
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
    _build_nzval_map(H_sparse, bs, n)

Build a mapping vector that encodes, for each logical block entry in the order
that `block_tridgm!` iterates, the corresponding index into `H_sparse.nzval`.
This allows `block_tridgm!` to write directly to nzval without sparse lookups.
"""
function _build_nzval_map(H_sparse::SparseMatrixCSC, bs::Int, n::Int)
    total_entries = n * bs * bs + 2 * (n - 1) * bs * bs
    nzval_map = Vector{Int}(undef, total_entries)

    colptr = H_sparse.colptr
    rowval = H_sparse.rowval

    idx = 1
    # Diagonal blocks: for k=1:n, j=1:bs, i=1:bs
    for k in 1:n
        base = (k - 1) * bs
        for j in 1:bs
            col = base + j
            # Find row (base + i) in this column's rowval range
            for i in 1:bs
                row = base + i
                # Binary search in rowval[colptr[col]:colptr[col+1]-1]
                lo, hi = colptr[col], colptr[col + 1] - 1
                while lo <= hi
                    mid = (lo + hi) >> 1
                    if rowval[mid] == row
                        nzval_map[idx] = mid
                        break
                    elseif rowval[mid] < row
                        lo = mid + 1
                    else
                        hi = mid - 1
                    end
                end
                idx += 1
            end
        end
    end

    # Off-diagonal blocks: for k=1:n-1, j=1:bs, i=1:bs (upper then lower)
    for k in 1:(n - 1)
        base_k = (k - 1) * bs
        base_kp1 = k * bs
        # Upper block: row in base_k+1:base_k+bs, col in base_kp1+1:base_kp1+bs
        for j in 1:bs
            col = base_kp1 + j
            for i in 1:bs
                row = base_k + i
                lo, hi = colptr[col], colptr[col + 1] - 1
                while lo <= hi
                    mid = (lo + hi) >> 1
                    if rowval[mid] == row
                        nzval_map[idx] = mid
                        break
                    elseif rowval[mid] < row
                        lo = mid + 1
                    else
                        hi = mid - 1
                    end
                end
                idx += 1
            end
        end
        # Lower block: row in base_kp1+1:base_kp1+bs, col in base_k+1:base_k+bs
        for j in 1:bs
            col = base_k + j
            for i in 1:bs
                row = base_kp1 + i
                lo, hi = colptr[col], colptr[col + 1] - 1
                while lo <= hi
                    mid = (lo + hi) >> 1
                    if rowval[mid] == row
                        nzval_map[idx] = mid
                        break
                    elseif rowval[mid] < row
                        lo = mid + 1
                    else
                        hi = mid - 1
                    end
                end
                idx += 1
            end
        end
    end

    return nzval_map
end

"""
    _build_block_tridiag_pattern(::Type{T}, bs::Int, n::Int)

Build a sparse matrix with the block tridiagonal sparsity pattern (values zeroed),
so that subsequent calls can update values in-place.
"""
function _build_block_tridiag_pattern(::Type{T}, bs::Int, n::Int) where {T<:Real}
    N = n * bs
    total_nnz = n * bs * bs + 2 * (n - 1) * bs * bs

    I_vec = Vector{Int}(undef, total_nnz)
    J_vec = Vector{Int}(undef, total_nnz)
    # Use ones so sparse() retains the structural entries (zeros get dropped)
    V_vec = ones(T, total_nnz)

    idx = 1
    for k in 1:n
        base = (k - 1) * bs
        for i in 1:bs, j in 1:bs
            I_vec[idx] = base + i
            J_vec[idx] = base + j
            idx += 1
        end
    end
    for k in 1:(n - 1)
        base_k = (k - 1) * bs
        base_kp1 = k * bs
        for i in 1:bs, j in 1:bs
            I_vec[idx] = base_k + i
            J_vec[idx] = base_kp1 + j
            idx += 1
        end
        for i in 1:bs, j in 1:bs
            I_vec[idx] = base_kp1 + i
            J_vec[idx] = base_k + j
            idx += 1
        end
    end

    H = sparse(I_vec, J_vec, V_vec, N, N)
    fill!(H.nzval, zero(T))  # Zero out values but keep the structure
    return H
end

"""
    block_tridgm!(ws::BlockTridiagonalWorkspace{T})

Update the values of the preallocated sparse matrix `ws.H_sparse` from the
current contents of `ws.H_diag`, `ws.H_sub`, `ws.H_super`.
Uses the precomputed `nzval_map` for direct writes — no sparse lookups or allocations.
"""
function block_tridgm!(ws::BlockTridiagonalWorkspace{T}) where {T<:Real}
    bs = ws.block_size
    n = ws.n_blocks
    nzval = ws.H_sparse.nzval
    map = ws.nzval_map

    idx = 1
    # Diagonal blocks
    for k in 1:n
        block = ws.H_diag[k]
        for j in 1:bs, i in 1:bs
            nzval[map[idx]] = block[i, j]
            idx += 1
        end
    end

    # Off-diagonal blocks (upper then lower for each k)
    for k in 1:(n - 1)
        block_up = ws.H_super[k]
        block_low = ws.H_sub[k]
        for j in 1:bs, i in 1:bs
            nzval[map[idx]] = block_up[i, j]
            idx += 1
        end
        for j in 1:bs, i in 1:bs
            nzval[map[idx]] = block_low[i, j]
            idx += 1
        end
    end

    return ws.H_sparse
end

"""
    block_tridiagonal_inverse!(p_smooth, p_smooth_tt1, A, B, C, ws)

Compute the block tridiagonal inverse, writing diagonal blocks into
`p_smooth[:,:,i]` and off-diagonal blocks into `p_smooth_tt1[:,:,i]`.
Uses preallocated buffers from `ws`.
"""
function block_tridiagonal_inverse!(
    p_smooth::AbstractArray{T,3},
    p_smooth_tt1::AbstractArray{T,3},
    A::Vector{<:AbstractMatrix{T}},
    B::Vector{<:AbstractMatrix{T}},
    C::Vector{<:AbstractMatrix{T}},
    ws::BlockTridiagonalWorkspace{T},
) where {T<:Real}
    n = length(B)

    D = ws.D
    E = ws.E
    M = ws.M
    term1 = ws.term1
    term2 = ws.term2
    S = ws.S
    Ibs = ws.Ibs
    Z = ws.Z

    fill!(D[1], zero(T))
    fill!(E[n + 1], zero(T))

    # Forward sweep for D
    for i in 1:n
        Ai = (i == 1) ? Z : A[i - 1]
        Ci = (i <= length(C)) ? C[i] : Z

        copyto!(M, B[i])
        mul!(M, Ai, D[i], -one(T), one(T))
        F = lu!(M)
        ldiv!(D[i + 1], F, Ci)
    end

    # Backward sweep for E
    for i in n:-1:1
        Ci = (i <= length(C)) ? C[i] : Z
        Ai = (i == 1) ? Z : A[i - 1]

        copyto!(M, B[i])
        mul!(M, Ci, E[i + 1], -one(T), one(T))
        F = lu!(M)
        ldiv!(E[i], F, Ai)
    end

    # Compute diagonal blocks -> p_smooth[:,:,i]
    for i in 1:n
        copyto!(term1, Ibs)
        mul!(term1, D[i + 1], E[i + 1], -one(T), one(T))

        Ai = (i == 1) ? Z : A[i - 1]
        copyto!(term2, B[i])
        mul!(term2, Ai, D[i], -one(T), one(T))

        mul!(S, term2, term1)
        F = lu!(S)

        @views ldiv!(p_smooth[:, :, i], F, Ibs)
    end

    # Compute off-diagonal blocks -> p_smooth_tt1[:,:,i] for i=2:n
    for i in 2:n
        @views mul!(p_smooth_tt1[:, :, i], E[i], p_smooth[:, :, i - 1])
        @views p_smooth_tt1[:, :, i] .*= -one(T)
    end

    return nothing
end

"""
    block_tridiagonal_inverse_logdet!(p_smooth, p_smooth_tt1, A, B, C, ws)

Compute the block tridiagonal inverse and log-determinant simultaneously.
Returns the log-determinant of the precision matrix (i.e., logdet of the input matrix).

This is more efficient than calling block_tridiagonal_inverse! and gaussian_entropy
separately, as it computes logdet during the forward sweep without additional
matrix factorizations.
"""
function block_tridiagonal_inverse_logdet!(
    p_smooth::AbstractArray{T,3},
    p_smooth_tt1::AbstractArray{T,3},
    A::Vector{<:AbstractMatrix{T}},
    B::Vector{<:AbstractMatrix{T}},
    C::Vector{<:AbstractMatrix{T}},
    ws::BlockTridiagonalWorkspace{T},
) where {T<:Real}
    n = length(B)

    D = ws.D
    E = ws.E
    M = ws.M
    term1 = ws.term1
    term2 = ws.term2
    S = ws.S
    Ibs = ws.Ibs
    Z = ws.Z

    fill!(D[1], zero(T))
    fill!(E[n + 1], zero(T))

    # Accumulate log-determinant during forward sweep
    logdet_val = zero(T)

    # Forward sweep for D - accumulate logdet from Schur complement factors
    for i in 1:n
        Ai = (i == 1) ? Z : A[i - 1]
        Ci = (i <= length(C)) ? C[i] : Z

        copyto!(M, B[i])
        mul!(M, Ai, D[i], -one(T), one(T))
        F = lu!(M)

        # Accumulate log|det(M_i)| from LU factors
        # det(LU) = det(L) * det(U) = 1 * prod(diag(U))
        # With pivoting, det = (-1)^p * prod(diag(U)) where p = number of row swaps
        # For logdet, we sum log|diag(U)|
        # Note: F.factors stores L\U in place; diagonal is U's diagonal (L has unit diagonal)
        factors = F.factors
        bs = size(M, 1)
        for j in 1:bs
            logdet_val += log(abs(factors[j, j]))
        end

        ldiv!(D[i + 1], F, Ci)
    end

    # Backward sweep for E
    for i in n:-1:1
        Ci = (i <= length(C)) ? C[i] : Z
        Ai = (i == 1) ? Z : A[i - 1]

        copyto!(M, B[i])
        mul!(M, Ci, E[i + 1], -one(T), one(T))
        F = lu!(M)
        ldiv!(E[i], F, Ai)
    end

    # Compute diagonal blocks -> p_smooth[:,:,i]
    for i in 1:n
        copyto!(term1, Ibs)
        mul!(term1, D[i + 1], E[i + 1], -one(T), one(T))

        Ai = (i == 1) ? Z : A[i - 1]
        copyto!(term2, B[i])
        mul!(term2, Ai, D[i], -one(T), one(T))

        mul!(S, term2, term1)
        F = lu!(S)

        @views ldiv!(p_smooth[:, :, i], F, Ibs)
    end

    # Compute off-diagonal blocks -> p_smooth_tt1[:,:,i] for i=2:n
    for i in 2:n
        @views mul!(p_smooth_tt1[:, :, i], E[i], p_smooth[:, :, i - 1])
        @views p_smooth_tt1[:, :, i] .*= -one(T)
    end

    return logdet_val
end

"""
    gaussian_entropy_from_logdet(logdet_precision::T, n::Int) where {T<:Real}

Compute Gaussian entropy from the log-determinant of the precision matrix.
`n` is the dimensionality (number of variables).
"""
function gaussian_entropy_from_logdet(logdet_precision::T, n::Int) where {T<:Real}
    return T(0.5) * (n * (1 + log(2π)) - logdet_precision)
end

"""
    block_tridiagonal_solve!(x, A, B, C, b, ws)

Solve a block tridiagonal system `H * x = b` where H has:
- Lower off-diagonal blocks A[i] (size bs×bs, i=1:n-1)
- Main diagonal blocks B[i] (size bs×bs, i=1:n)
- Upper off-diagonal blocks C[i] (size bs×bs, i=1:n-1)

Uses block LU decomposition (Thomas algorithm for blocks).
The solution overwrites `x` (which should be length n*bs).

This is much more efficient than sparse LU for block tridiagonal matrices
as it only requires O(n) block factorizations instead of filling in a
potentially large sparse LU.

# Arguments
- `x::AbstractVector{T}`: Output vector (length n*bs), will be overwritten with solution
- `A::Vector{Matrix{T}}`: Lower off-diagonal blocks (length n-1)
- `B::Vector{Matrix{T}}`: Main diagonal blocks (length n)
- `C::Vector{Matrix{T}}`: Upper off-diagonal blocks (length n-1)
- `b::AbstractVector{T}`: Right-hand side vector (length n*bs)
- `ws::BlockTridiagonalWorkspace{T}`: Workspace with temp buffers
"""
function block_tridiagonal_solve!(
    x::AbstractVector{T},
    A::Vector{<:AbstractMatrix{T}},
    B::Vector{<:AbstractMatrix{T}},
    C::Vector{<:AbstractMatrix{T}},
    b::AbstractVector{T},
    ws::BlockTridiagonalWorkspace{T},
) where {T<:Real}
    n = length(B)
    bs = size(B[1], 1)

    # Reuse workspace buffers
    # We need: modified diagonal blocks (stored in D), modified RHS (stored temporarily)
    # D[i] will store the modified upper off-diagonal after forward elimination
    D = ws.D
    M = ws.M  # Temp for LU factorization

    # Forward elimination (modify diagonal and upper off-diagonal, and RHS)
    # Store modified C[i] in D[i+1] and modified b[i] in x[block i] temporarily

    # First block: just copy b₁ to x₁ block and store C₁' in D[2]
    @views copyto!(x[1:bs], b[1:bs])

    # Process first block specially (no A[0])
    copyto!(M, B[1])
    F = lu!(M)
    @views ldiv!(F, x[1:bs])  # x₁ = B₁⁻¹ b₁
    ldiv!(D[2], F, C[1])       # D[2] = B₁⁻¹ C₁

    # Forward sweep for blocks 2 to n
    for i in 2:n
        # Get block indices
        idx_start = (i - 1) * bs + 1
        idx_end = i * bs
        idx_prev_start = (i - 2) * bs + 1
        idx_prev_end = (i - 1) * bs

        # Copy b[i] to x[i] block
        @views copyto!(x[idx_start:idx_end], b[idx_start:idx_end])

        # Modify diagonal: B̃ᵢ = Bᵢ - Aᵢ₋₁ * D[i]
        copyto!(M, B[i])
        mul!(M, A[i - 1], D[i], -one(T), one(T))

        # Modify RHS: b̃ᵢ = bᵢ - Aᵢ₋₁ * x[i-1]
        @views mul!(x[idx_start:idx_end], A[i - 1], x[idx_prev_start:idx_prev_end], -one(T), one(T))

        # Factor modified diagonal
        F = lu!(M)

        # Solve for x[i] block
        @views ldiv!(F, x[idx_start:idx_end])

        # Compute modified upper off-diagonal for next iteration (if not last block)
        if i < n
            ldiv!(D[i + 1], F, C[i])
        end
    end

    # Backward substitution
    for i in (n - 1):-1:1
        idx_start = (i - 1) * bs + 1
        idx_end = i * bs
        idx_next_start = i * bs + 1
        idx_next_end = (i + 1) * bs

        # x[i] = x[i] - D[i+1] * x[i+1]
        @views mul!(x[idx_start:idx_end], D[i + 1], x[idx_next_start:idx_next_end], -one(T), one(T))
    end

    return x
end

"""
    _negate_blocks!(ws::BlockTridiagonalWorkspace{T})

Copy negated H_diag/H_sub/H_super into neg_diag/neg_sub/neg_super in-place.
"""
function _negate_blocks!(ws::BlockTridiagonalWorkspace{T}) where {T<:Real}
    for i in eachindex(ws.H_diag)
        ws.neg_diag[i] .= .-ws.H_diag[i]
    end
    for i in eachindex(ws.H_sub)
        ws.neg_sub[i] .= .-ws.H_sub[i]
        ws.neg_super[i] .= .-ws.H_super[i]
    end
    return nothing
end

# Initialization utilities
"""
    euclidean_distance(a::AbstractVector{Float64}, b::AbstractVector{Float64})

Calculate the Euclidean distance between two points.
"""
function euclidean_distance(a::AbstractVector, b::AbstractVector)
    return sqrt(sum((a .- b) .^ 2))
end

"""
    kmeanspp_initialization(data::AbstractMatrix{T}, k_means::Int) where {T<:Real}

Perform K-means++ initialization for cluster centroids (column-major input).
"""
function kmeanspp_initialization(data::AbstractMatrix{T}, k_means::Int) where {T<:Real}
    D, N = size(data)  # (D, N) data layout
    centroids = zeros(T, D, k_means)
    rand_idx = rand(1:N)
    centroids[:, 1] = data[:, rand_idx]

    for k in 2:k_means
        dists = zeros(N)
        for i in 1:N
            dists[i] = minimum([
                euclidean_distance(@view(data[:, i]), @view(centroids[:, j])) for
                j in 1:(k - 1)
            ])
        end

        probs = dists .^ 2
        probs ./= sum(probs)
        next_idx = StatsBase.sample(1:N, Weights(probs))
        centroids[:, k] = data[:, next_idx]
    end

    return centroids
end

"""
    kmeanspp_initialization(data::AbstractVector{T}, k_means::Int)

K-means++ initialization for vector data.
"""
function kmeanspp_initialization(data::AbstractVector{T}, k_means::Int) where {T<:Real}
    data = reshape(data, 1, :)  # shape (1, N)
    return kmeanspp_initialization(data, k_means)
end

"""
    kmeans_clustering(
        data::AbstractMatrix{T}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6
    ) where {T<:Real}

Perform K-means clustering on column-major data.
"""
function kmeans_clustering(
    data::AbstractMatrix{T}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6
) where {T<:Real}
    D, N = size(data)
    centroids = kmeanspp_initialization(data, k_means)
    labels = zeros(Int, N)

    for iter in 1:max_iters
        for i in 1:N
            x_i = @view data[:, i]
            min_k, min_dist = 1, euclidean_distance(x_i, @view centroids[:, 1])

            for k in 2:k_means
                dist = euclidean_distance(x_i, @view centroids[:, k])
                if dist < min_dist
                    min_dist = dist
                    min_k = k
                end
            end

            labels[i] = min_k
        end

        old_centroids = copy(centroids)
        new_centroids = zeros(T, D, k_means)

        for k in 1:k_means
            inds = findall(labels .== k)

            if isempty(inds)
                new_centroids[:, k] .= data[:, rand(1:N)]
            else
                cluster_points = data[:, inds]
                new_centroids[:, k] .= mean(cluster_points; dims=2)
            end
        end

        centroids .= new_centroids

        if all(
            euclidean_distance(centroids[:, k], old_centroids[:, k]) <= tol for
            k in 1:k_means
        )
            break
        end
    end

    return centroids, labels
end

"""
    kmeans_clustering(
        data::AbstractVector{T}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6
    )

Perform K-means clustering on vector data.
"""
function kmeans_clustering(
    data::AbstractVector{T}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6
) where {T<:Real}
    data = reshape(data, 1, :)  # shape (1, N)
    return kmeans_clustering(data, k_means, max_iters, tol)
end

"""
    logistic(x::Real)

Calculate the logistic function in a numerically stable way.
"""
function logistic(x::Real)
    if x > 0
        return 1 / (1 + exp(-x))
    else
        exp_x = exp(x)

        return exp_x / (1 + exp_x)
    end
end

"""
    make_posdef!(A::AbstractMatrix{T}) where {T<:Real}

Ensure that a matrix is positive definite by adjusting its eigenvalues.
"""
function make_posdef!(A::AbstractMatrix{T}; min_eigval::T=convert(T, 1e-6)) where {T<:Real}
    # Work with the symmetric part
    B = Symmetric((A + A') / 2)

    # Get eigendecomposition
    F = eigen(B)

    # Find negative or small eigenvalues
    neg_eigs = F.values .< min_eigval

    # If already positive definite, return early
    if !any(neg_eigs)
        return A
    end

    # Fix negative eigenvalues
    F.values[neg_eigs] .= min_eigval

    # Reconstruct
    A .= F.vectors * Diagonal(F.values) * F.vectors'

    # Ensure symmetry due to numerical errors
    A .= (A + A') / 2

    return A
end

"""
    stabilize_covariance_matrix(Σ::Matrix{<:Real})

Stabilize a covariance matrix by ensuring it is symmetric and positive definite.
"""
function stabilize_covariance_matrix(Σ::AbstractMatrix{T}) where {T<:Real}
    # check if the covariance is symmetric. If not, make it symmetric
    if !ishermitian(Σ)
        Σ = (Σ + Σ') * 0.5
    end

    # check if matrix is posdef. If not, add a small value to the diagonal (sometimes an
    # emission only models one observation and the covariance matrix is singular)
    if !isposdef(Σ)
        Σ = Σ + 1e-12 * I
    end

    return Σ
end

function valid_Σ(Σ::AbstractMatrix{T}) where {T<:Real}
    return ishermitian(Σ) && isposdef(Σ)
end

# Function for stacking data... in prep for the trialized M_step!()
function stack_tuples(d)
    # Determine the number of tuples and number of elements in each tuple
    num_tuples = length(d)
    num_elements = length(d[1])

    # Initialize an array to store the stacked matrices
    stacked_matrices = Vector{Matrix{Float64}}(undef, num_elements)

    # Stack matrices for each position in the tuple
    for i in 1:num_elements
        # Extract all matrices at the i-th position from each tuple
        matrices_to_stack = [d[j][i] for j in 1:num_tuples]
        # Vertically concatenate the collected matrices
        stacked_matrices[i] = vcat(matrices_to_stack...)
    end

    # Return the stacked matrices as a tuple
    return tuple(stacked_matrices...)
end

"""
    gaussian_entropy(H::Symmetric{T}) where {T<:Real}

Entropy (in nats) of a Gaussian whose log-posterior Hessian at the MAP is `H`
(i.e., `H = ∇² log p` so the precision is `Λ = -H`).
"""
function gaussian_entropy(H::Symmetric{T}) where {T<:Real}
    n = size(H, 1)
    F = cholesky(-H)                    # factorize Λ = -H (SPD)
    logdetΛ = 2 * sum(log, diag(F))     # logdet precision
    return 0.5 * (n * (1 + log(2π)) - logdetΛ)
end

"""
    gaussian_entropy(H::Symmetric{BigFloat, <:SparseMatrix})

Specialized method for BigFloat sparse matrices using logdet.
"""
function gaussian_entropy(H::Symmetric{BigFloat,<:AbstractSparseMatrix})
    n = size(H, 1)
    logdet_H = logdet(-H)

    return 0.5 * (n * (BigFloat(1 + log(BigFloat(2π))) - logdet_H))
end

"""
    random_rotation_matrix(n)

Generate a random rotation matrix of size `n x n`.
"""
function random_rotation_matrix(n::Int, rng::AbstractRNG=Random.default_rng())
    # Generate a random orthogonal matrix using QR decomposition
    Q, _ = qr(randn(rng, n, n))

    return Matrix(Q)
end

"""
    getproperty(model::AutoRegressiveEmission, sym::Symbol)

Get various properties of 'innerGaussianRegression`.
"""
function Base.getproperty(model::AutoRegressiveEmission, sym::Symbol)
    if sym === :β
        return model.innerGaussianRegression.β
    elseif sym === :Σ
        return model.innerGaussianRegression.Σ
    elseif sym === :include_intercept
        return model.innerGaussianRegression.include_intercept
    elseif sym === :λ
        return model.innerGaussianRegression.λ
    else # fallback to getfield
        return getfield(model, sym)
    end
end

"""
    setproperty!(model::AutoRegressiveEmission, sym::Symbol, value)

Assign to properties of an `AutoRegressiveEmission` by forwarding certain symbols
to its `innerGaussianRegression` field:
"""
# define setters for innerGaussianRegression fields
function Base.setproperty!(model::AutoRegressiveEmission, sym::Symbol, value)
    if sym === :β
        model.innerGaussianRegression.β = value
    elseif sym === :Σ
        model.innerGaussianRegression.Σ = value
    elseif sym === :λ
        model.innerGaussianRegression.λ = value
    else # fallback to setfield!
        setfield!(model, sym, value)
    end
end

# Pretty print function that doesn't truncate arrays of model objects

"""
    print_full([io::Union{IO, Base.TTY}, ] obj)

Prints full description of object `obj`, overriding both `io`-based limits as
well as the limits set in the default pretty printing of `StateSpaceDynamics`
objects.
"""
function print_full(io::Union{IO,Base.TTY}, obj)
    println(IOContext(io, :limit => false), obj)

    return nothing
end

print_full(obj) = print_full(stdout, obj)
