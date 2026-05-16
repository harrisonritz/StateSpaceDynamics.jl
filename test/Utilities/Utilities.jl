function test_block_tridgm()
    # Test with minimal block sizes
    super = [rand(1, 1) for i in 1:1]
    sub = [rand(1, 1) for i in 1:1]
    main = [rand(1, 1) for i in 1:2]
    A = block_tridgm(main, super, sub)
    @test size(A) == (2, 2)
    @test A[1, 1] == main[1][1, 1]
    @test A[2, 2] == main[2][1, 1]
    @test A[1, 2] == super[1][1, 1]
    @test A[2, 1] == sub[1][1, 1]

    # Test with 2x2 blocks and a larger matrix
    super = [rand(2, 2) for i in 1:9]
    sub = [rand(2, 2) for i in 1:9]
    main = [rand(2, 2) for i in 1:10]
    A = block_tridgm(main, super, sub)
    @test size(A) == (20, 20)

    # Check some blocks in the matrix
    for i in 1:10
        @test A[(2i - 1):(2i), (2i - 1):(2i)] == main[i]
        if i < 10
            @test A[(2i - 1):(2i), (2i + 1):(2i + 2)] == super[i]
            @test A[(2i + 1):(2i + 2), (2i - 1):(2i)] == sub[i]
        end
    end

    # Test with integer blocks
    super = [rand(Int, 2, 2) for i in 1:9]
    sub = [rand(Int, 2, 2) for i in 1:9]
    main = [rand(Int, 2, 2) for i in 1:10]
    A = block_tridgm(main, super, sub)
    @test size(A) == (20, 20)
    for i in 1:10
        @test A[(2i - 1):(2i), (2i - 1):(2i)] == main[i]
        if i < 10
            @test A[(2i - 1):(2i), (2i + 1):(2i + 2)] == super[i]
            @test A[(2i + 1):(2i + 2), (2i - 1):(2i)] == sub[i]
        end
    end
end

function test_gaussian_entropy()
    n = 3
    A = sprandn(n, n, 0.6)
    Λ = Symmetric(A' * A) + 1e-8I

    F = cholesky(Λ)
    Σ = Symmetric(Matrix(F \ Matrix{Float64}(I, n, n)))

    gaus_entropy_dist = entropy(MvNormal(zeros(n), Σ))
    gauss_entropy_ssd = gaussian_entropy(-Λ)
    @test isapprox(gaus_entropy_dist, gauss_entropy_ssd; atol=1e-6)
end

# Build a random, well-conditioned block tridiagonal matrix and return both
# (A, B, C) blocks plus the dense reconstruction H so tests can compare against inv(H).
function _random_block_tridiag(::Type{T}, block_size::Int, n::Int, rng) where {T<:Real}
    A = Matrix{T}[randn(rng, T, block_size, block_size) for _ in 1:(n - 1)]
    B = Matrix{T}[
        T(5.0) * Matrix{T}(I, block_size, block_size) +
        randn(rng, T, block_size, block_size) for _ in 1:n
    ]
    C = Matrix{T}[randn(rng, T, block_size, block_size) for _ in 1:(n - 1)]
    H = Matrix(block_tridgm(B, C, A))
    return A, B, C, H
end

function test_block_tridiagonal_inverse_mutating()
    rng = MersenneTwister(42)

    for T in (Float64, Float32)
        atol = T === Float32 ? 1e-4 : 1e-8
        for block_size in (1, 2, 3), n in (1, 2, 3, 4)
            A, B, C, H = _random_block_tridiag(T, block_size, n, rng)
            Hinv = inv(H)

            ws = StateSpaceDynamics.BlockTridiagonalWorkspace(T, block_size, n)
            p_smooth = zeros(T, block_size, block_size, n)
            p_smooth_tt1 = zeros(T, block_size, block_size, n)

            StateSpaceDynamics.block_tridiagonal_inverse!(
                p_smooth, p_smooth_tt1, A, B, C, ws
            )

            for i in 1:n
                rows = ((i - 1) * block_size + 1):(i * block_size)
                @test isapprox(p_smooth[:, :, i], Hinv[rows, rows]; atol=atol, rtol=0)
            end
            for i in 2:n
                rows_i = ((i - 1) * block_size + 1):(i * block_size)
                rows_im1 = ((i - 2) * block_size + 1):((i - 1) * block_size)
                @test isapprox(
                    p_smooth_tt1[:, :, i], Hinv[rows_i, rows_im1]; atol=atol, rtol=0
                )
            end
        end
    end
end

# Build a random SPD block tridiagonal matrix by forming `H = L Lᵀ` for a
# block-bidiagonal `L`. This matches the precondition of
# `block_tridiagonal_inverse_logdet!`, which factors the SPD Schur
# complements via Cholesky.
function _random_spd_block_tridiag(::Type{T}, block_size::Int, n::Int, rng) where {T<:Real}
    # Block-bidiagonal L: diagonal L_diag and lower off-diagonal L_off.
    L_diag = Matrix{T}[
        T(2.0) * Matrix{T}(I, block_size, block_size) +
        T(0.3) * randn(rng, T, block_size, block_size) for _ in 1:n
    ]
    L_off = Matrix{T}[
        T(0.3) * randn(rng, T, block_size, block_size) for _ in 1:(n - 1)
    ]
    # Assemble L (lower block-bidiagonal) densely, then H = L Lᵀ.
    Ldense = zeros(T, n * block_size, n * block_size)
    for i in 1:n
        rows = ((i - 1) * block_size + 1):(i * block_size)
        Ldense[rows, rows] = L_diag[i]
        if i < n
            rows_next = (i * block_size + 1):((i + 1) * block_size)
            Ldense[rows_next, rows] = L_off[i]
        end
    end
    H = Ldense * Ldense'
    H = (H + H') / 2  # exact symmetry
    # Extract the block tridiagonal pieces.
    A = Matrix{T}[copy(H[(i * block_size + 1):((i + 1) * block_size),
                         ((i - 1) * block_size + 1):(i * block_size)]) for i in 1:(n - 1)]
    B = Matrix{T}[copy(H[((i - 1) * block_size + 1):(i * block_size),
                         ((i - 1) * block_size + 1):(i * block_size)]) for i in 1:n]
    C = Matrix{T}[copy(transpose(A[i])) for i in 1:(n - 1)]
    return A, B, C, H
end

function test_block_tridiagonal_inverse_logdet()
    rng = MersenneTwister(7)
    T = Float64
    block_size = 3
    n = 5

    A, B, C, H = _random_spd_block_tridiag(T, block_size, n, rng)
    expected_logdet = logdet(H)

    ws = StateSpaceDynamics.BlockTridiagonalWorkspace(T, block_size, n)
    p_smooth = zeros(T, block_size, block_size, n)
    p_smooth_tt1 = zeros(T, block_size, block_size, n)

    logdet_val = StateSpaceDynamics.block_tridiagonal_inverse_logdet!(
        p_smooth, p_smooth_tt1, A, B, C, ws
    )

    @test isapprox(logdet_val, expected_logdet; atol=1e-8, rtol=0)

    Hinv = inv(H)
    for i in 1:n
        rows = ((i - 1) * block_size + 1):(i * block_size)
        @test isapprox(p_smooth[:, :, i], Hinv[rows, rows]; atol=1e-8, rtol=0)
    end
end

function test_block_tridiagonal_solve()
    rng = MersenneTwister(11)
    T = Float64
    block_size = 2
    n = 4

    A, B, C, H = _random_block_tridiag(T, block_size, n, rng)
    b = randn(rng, T, n * block_size)
    expected = H \ b

    ws = StateSpaceDynamics.BlockTridiagonalWorkspace(T, block_size, n)
    x = zeros(T, n * block_size)
    StateSpaceDynamics.block_tridiagonal_solve!(x, A, B, C, b, ws)

    @test isapprox(x, expected; atol=1e-8, rtol=0)
end
