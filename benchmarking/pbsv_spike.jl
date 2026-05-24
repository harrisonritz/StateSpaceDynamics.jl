using StateSpaceDynamics
using LinearAlgebra, SparseArrays, BenchmarkTools, Random
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 3.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5

# Manual ccall to dpbsv_ — solves SPD banded system in-place.
function dpbsv!(
    uplo::Char,
    kd::Int,
    AB::Matrix{Float64},
    B::AbstractVecOrMat{Float64},
)
    n = size(AB, 2)
    ldab = stride(AB, 2)
    nrhs = B isa AbstractVector ? 1 : size(B, 2)
    ldb = B isa AbstractVector ? n : stride(B, 2)
    info = Ref{LinearAlgebra.BlasInt}(0)
    ccall(
        (LinearAlgebra.BLAS.@blasfunc(dpbsv_), LinearAlgebra.libblastrampoline),
        Cvoid,
        (Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt},
         Ref{LinearAlgebra.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BlasInt},
         Ptr{Float64}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}),
        uplo, n, kd, nrhs, AB, ldab, B, ldb, info,
    )
    LinearAlgebra.LAPACK.chklapackerror(info[])
    return B
end

# Build a real negated-Hessian for a Poisson LDS at its initial x=0 state.
function make_poisson_hessian(D::Int, T::Int, p::Int)
    rng = MersenneTwister(42)
    A = 0.9 .* StateSpaceDynamics.random_rotation_matrix(D, rng)
    Q = Matrix(0.1 * I(D))
    x0 = zeros(D)
    P0 = Matrix(0.1 * I(D))
    C = 0.3 .* randn(rng, p, D)
    d = log.(0.5 .+ rand(rng, p))
    b = zeros(D)
    sm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
    om = PoissonObservationModel(; C=C, d=d)
    lds = LinearDynamicalSystem(sm, om)
    rng2 = MersenneTwister(123)
    _, y_multi = StateSpaceDynamics.rand(rng2, lds, fill(T, 1))
    y = y_multi[1]
    x = zeros(D, T)
    H, _, _, _ = StateSpaceDynamics.Hessian(lds, y, x)
    return -Matrix(H)   # negate to make SPD (Hessian at MAP is neg-def for max)
end

# Pack upper triangle of an SPD matrix into LAPACK banded format AB.
function pack_banded_upper!(AB, Hdense, kd)
    n = size(Hdense, 1)
    fill!(AB, 0)
    for j in 1:n
        for i in max(1, j - kd):j
            AB[kd + 1 + i - j, j] = Hdense[i, j]
        end
    end
    return AB
end

println(rpad("(D, T, p)", 16),
        rpad("ours_bt (ms)", 14), rpad("pbsv (ms)", 12), rpad("UMFPACK (ms)", 14),
        rpad("pbsv vs ours", 14), "UMF vs ours")
for (D, T, p) in [
    (3,  200,  5),
    (5,  200, 10),
    (8,  200, 16),
    (16, 200, 32),
    (32, 200, 64),
    (50, 200, 80),
    (80, 200, 100),
]
    Hdense_raw = make_poisson_hessian(D, T, p)
    # Make exactly symmetric (sparse → dense round-trip may leave ULP asymmetry)
    Hdense = (Hdense_raw + Hdense_raw') ./ 2
    if !isposdef(Hdense)
        λmin = eigmin(Hdense)
        # Small diagonal jitter to push borderline indefinite to PD.
        Hdense += max(-λmin + 1e-8, 0.0) * I
        @assert isposdef(Hdense) "still not SPD after jitter (D=$D T=$T)"
    end

    Hsparse = sparse(Hdense)

    n = D * T
    kd = 2 * D - 1
    AB = zeros(kd + 1, n)
    pack_banded_upper!(AB, Hdense, kd)

    rhs = randn(n)
    x_ref = Hdense \ rhs
    AB_work = copy(AB); rhs_work = copy(rhs)
    dpbsv!('U', kd, AB_work, rhs_work)
    err = norm(rhs_work - x_ref) / norm(x_ref)
    @assert err < 1e-6 "pbsv result wrong (D=$D T=$T): rel err $err"

    AB_template = copy(AB)
    rhs_template = copy(rhs)
    b_pbsv = @benchmark begin
        copyto!($AB_work, $AB_template)
        copyto!($rhs_work, $rhs_template)
        dpbsv!('U', $kd, $AB_work, $rhs_work)
    end

    b_umf = @benchmark $Hsparse \ $rhs

    # Build a SmoothWorkspace + feed Hessian blocks to our block_tridiagonal_solve!
    sws = StateSpaceDynamics.SmoothWorkspace(Float64, D, p, T; u_dim=0, d_dim=0)
    btd = sws.btd
    # Repack Hdense into block-tridiagonal form
    for k in 1:T
        btd.neg_diag[k] .= Hdense[((k - 1) * D + 1):(k * D), ((k - 1) * D + 1):(k * D)]
    end
    for k in 1:(T - 1)
        btd.neg_super[k] .= Hdense[
            ((k - 1) * D + 1):(k * D), (k * D + 1):((k + 1) * D)
        ]
        btd.neg_sub[k] .= btd.neg_super[k]'  # symmetric
    end
    neg_sub_v = view(btd.neg_sub, 1:(T - 1))
    neg_diag_v = view(btd.neg_diag, 1:T)
    neg_super_v = view(btd.neg_super, 1:(T - 1))
    p_buf = similar(rhs)
    # warm
    StateSpaceDynamics.block_tridiagonal_solve!(
        p_buf, neg_sub_v, neg_diag_v, neg_super_v, rhs, btd
    )
    b_ours = @benchmark StateSpaceDynamics.block_tridiagonal_solve!(
        $p_buf, $neg_sub_v, $neg_diag_v, $neg_super_v, $rhs, $btd
    )

    pbsv_ratio = round(median(b_ours).time / median(b_pbsv).time; digits=2)
    umf_ratio = round(median(b_ours).time / median(b_umf).time; digits=2)
    println(
        rpad("D=$D T=$T p=$p", 16),
        rpad(round(median(b_ours).time / 1e6; digits=3), 14),
        rpad(round(median(b_pbsv).time / 1e6; digits=3), 12),
        rpad(round(median(b_umf).time / 1e6; digits=3), 14),
        rpad(pbsv_ratio, 14),
        umf_ratio,
    )
end
