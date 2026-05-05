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
    Symmetrize!(A::AbstractMatrix{T}) where {T<:Real}

In-place symmetrization of a square matrix `A` via averaging with its transpose:
A <- 0.5*(A + A').
"""
function Symmetrize!(A::AbstractMatrix{T}) where {T<:Real}
    n, m = size(A)
    @boundscheck n == m || throw(
        DimensionMismatch("Matrix must be square for symmetrization, got $(n)×$(m)")
    )
    half = T(0.5)

    for j in 1:n
        for i in 1:(j - 1)
            avg = (A[i, j] + A[j, i]) * half
            A[i, j] = avg
            A[j, i] = avg
        end
    end

    return A
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
    main_diag::AbstractVector{<:AbstractMatrix{T}},
    upper_diag::AbstractVector{<:AbstractMatrix{T}},
    lower_diag::AbstractVector{<:AbstractMatrix{T}},
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
    A::AbstractVector{<:AbstractMatrix{T}},
    B::AbstractVector{<:AbstractMatrix{T}},
    C::AbstractVector{<:AbstractMatrix{T}},
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
    A::AbstractVector{<:AbstractMatrix{T}},
    B::AbstractVector{<:AbstractMatrix{T}},
    C::AbstractVector{<:AbstractMatrix{T}},
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
    A::AbstractVector{<:AbstractMatrix{T}},
    B::AbstractVector{<:AbstractMatrix{T}},
    C::AbstractVector{<:AbstractMatrix{T}},
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
        @views mul!(
            x[idx_start:idx_end], A[i - 1], x[idx_prev_start:idx_prev_end], -one(T), one(T)
        )

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
        @views mul!(
            x[idx_start:idx_end], D[i + 1], x[idx_next_start:idx_next_end], -one(T), one(T)
        )
    end

    return x
end

"""
    _negate_blocks!(ws::BlockTridiagonalWorkspace{T}, n_active::Int=ws.n_blocks)

Copy negated H_diag/H_sub/H_super into neg_diag/neg_sub/neg_super in-place
for the first `n_active` diagonal blocks (and `n_active - 1` off-diagonal blocks).
"""
function _negate_blocks!(
    ws::BlockTridiagonalWorkspace{T}, n_active::Int=ws.n_blocks
) where {T<:Real}
    for i in 1:n_active
        ws.neg_diag[i] .= .-ws.H_diag[i]
    end
    for i in 1:(n_active - 1)
        ws.neg_sub[i] .= .-ws.H_sub[i]
        ws.neg_super[i] .= .-ws.H_super[i]
    end
    return nothing
end

function valid_Σ(Σ::AbstractMatrix{T}) where {T<:Real}
    return ishermitian(Σ) && isposdef(Σ)
end

"""
    tol_PD(A; tol=1e-6) -> PDMat

Eigen-floor stabilization for covariance matrices used along the Kalman path. All
eigenvalues below `tol * λ_max` are raised to `tol * λ_max`, preserving the overall
scale/conditioning of the matrix; the result is rewrapped as a `PDMat` so downstream
code can reuse the cached Cholesky. The matrix is symmetrized (via `hermitianpart`) if
passed as a plain `Matrix`.

Used by the Kalman filter/smoother to keep predicted and filtered covariances strictly
positive definite in the presence of floating-point noise. Ported from
StateSpaceAnalysis.
"""
function tol_PD(
    A_sym::Union{Symmetric{T},Hermitian{T}}; tol::T=1e-6
)::PDMat{T,Matrix{T}} where {T<:Real}
    # F = eigen!(A_sym)
    # λ_max = F.values[end]
    # λ_r = max.(F.values ./ λ_max, zero(T))
    # λ_new = (λ_max - λ_max * tol) .* λ_r .+ λ_max * tol
    # return PDMat(X_A_Xt(PDiagMat(λ_new), F.vectors))

    F = eigen!(A_sym)
    λ_max = F.values[end]
    scale = λ_max * tol
    slope = λ_max - scale        # = λ_max * (1 - tol)
    for i in eachindex(F.values)
        r = max(F.values[i] / λ_max, zero(T))
        F.values[i] = slope * r + scale
    end
    return PDMat(X_A_Xt(PDiagMat(F.values), F.vectors))
end

tol_PD(A::Matrix; tol::Real=1e-6)::PDMat = tol_PD(hermitianpart(A); tol=tol)
tol_PD(A::PDMat; tol::Real=1e-6)::PDMat = tol_PD(Hermitian(Matrix(A)); tol=tol)
id_PD(A::Matrix; tol::Real=1e-6)::PDMat = PDMat(
    hermitianpart(A + (tol * tr(A) / size(A, 1)) * I)
)

# logdet(Σ) from Cholesky Σ = U'U => logdet(Σ) = 2 * sum(log(diag(U)))
function _logdet_from_U(U::AbstractMatrix{T}, n::Int) where {T}
    s = zero(T)
    for i in 1:n
        s += log(U[i, i])
    end
    return 2s
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
    random_rotation_matrix(n)

Generate a random rotation matrix of size `n x n`.
"""
function random_rotation_matrix(n::Int, rng::AbstractRNG=Random.default_rng())
    # Generate a random orthogonal matrix using QR decomposition
    Q, _ = qr(randn(rng, n, n))

    return Matrix(Q)
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
