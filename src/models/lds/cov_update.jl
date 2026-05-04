using LinearAlgebra
using LinearAlgebra: LAPACK, BlasFloat, axpy!, copytri!
using PDMats

"""
    CovUpdateCache{T}(n)

Preallocated workspace for `info_update!`. Holds three `n × n` matrices:
scratch for the intermediate sum and its Cholesky, plus storage for the
output covariance matrix and its Cholesky factor.
"""
struct CovUpdateCache{T<:BlasFloat}
    M::Matrix{T}              # inv(P0) + CiRC, then its Cholesky, then inv(sum)
    Pmat::Matrix{T}           # output: full symmetric P = inv(inv(P0) + CiRC)
    Pchol_factors::Matrix{T}  # output: Cholesky factor of P
end

function CovUpdateCache{T}(n::Integer) where {T<:BlasFloat}
    CovUpdateCache{T}(Matrix{T}(undef, n, n),
                      Matrix{T}(undef, n, n),
                      Matrix{T}(undef, n, n))
end
CovUpdateCache(n::Integer) = CovUpdateCache{Float64}(n)


"""
    info_update!(cache, P0, CiRC) -> PDMat

Return `P = inv(inv(P0) + CiRC)` as a `PDMat`, using the Cholesky cached
inside `P0` and the scratch buffers in `cache`. Both `P0` and `CiRC` must
be `PDMat{T,Matrix{T}}` of dimension `n`, matching `cache`.

The returned `PDMat` *shares storage* with `cache.Pmat` and
`cache.Pchol_factors`. It is invalidated by the next call with the same
cache — extract or accumulate what you need (e.g. add `P.mat + μ*μ'` into
your running `E[xxᵀ]` sum) before the next invocation.

Cost: one `potri` + one `cholesky!` + one `potri` + one `cholesky!` on
`n × n` matrices, ≈ 4n³/3 flops plus O(n²) for the add. Compare with
the naive `inv(inv(P0) + CiRC)` which goes ≈ 8n³/3 via PDMat → Matrix →
LU-based `inv` → PDMat, losing PD structure in the middle.
"""
function info_update!(cache::CovUpdateCache{T},
                      P0::PDMat{T,Matrix{T}},
                      CiRC::PDMat{T,Matrix{T}}) where {T<:BlasFloat}
    n = size(P0, 1)
    @boundscheck begin
        size(CiRC, 1) == n || throw(DimensionMismatch("P0 and CiRC differ"))
        size(cache.M) == (n, n) ||
            throw(DimensionMismatch("cache sized for n=$(size(cache.M, 1))"))
    end

    M    = cache.M
    Pmat = cache.Pmat
    Pfac = cache.Pchol_factors

    # (1) M ← inv(P0), using P0's cached Cholesky.
    #     potri! takes a Cholesky factor (in the `uplo` triangle) and
    #     overwrites it with the inverse of the original PD matrix
    #     (also in the `uplo` triangle). The other triangle is left
    #     untouched, so we then reflect to get a full symmetric matrix.
    uplo0 = P0.chol.uplo                       # 'U' or 'L'
    copyto!(M, P0.chol.factors)
    LAPACK.potri!(uplo0, M)
    copytri!(M, uplo0)

    # (2) M ← M + CiRC.mat.  (Both operands symmetric; axpy! on the
    #     whole dense backing is fine and is what BLAS is happiest with.)
    axpy!(one(T), CiRC.mat, M)

    # (3) Cholesky of M, in place, upper triangle: M_upper ← U with M = UᵀU.
    cholesky!(Symmetric(M, :U); check = true)

    # (4) potri! on the fresh factor: M's upper triangle now holds
    #     inv(P0⁻¹ + CiRC), i.e. the new covariance P.
    LAPACK.potri!('U', M)

    # (5) Copy to Pmat and symmetrize — this is the `mat` field of the
    #     output PDMat.
    copyto!(Pmat, M)
    copytri!(Pmat, 'U')

    # (6) Fresh Cholesky of Pmat into Pfac — the `chol` field of output.
    #     Unavoidable: there is no closed-form shortcut from chol(M) to
    #     chol(inv(M)) that preserves standard Cholesky triangularity.
    copyto!(Pfac, Pmat)
    Cout = cholesky!(Symmetric(Pfac, :U); check = true)

    return PDMat(Pmat, Cout)
end