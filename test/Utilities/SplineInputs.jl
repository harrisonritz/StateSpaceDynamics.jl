function test_spline_inputs_shape_and_kron_structure()
    T = Float64
    obs_dim, tsteps, ntrials = 2, 40, 4
    num_bases = 8
    P = 3

    rng = MersenneTwister(101)
    trial_pred = randn(rng, T, ntrials, P)
    y = randn(rng, T, obs_dim, tsteps, ntrials)
    u = zeros(T, P * num_bases, tsteps, ntrials)
    d = zeros(T, 1, tsteps, ntrials)

    data = Data(; y=y, u=u, d=d, trial_pred=trial_pred)
    penalty = generate_spline_inputs!(data, num_bases; target=:u, order=4)

    @test size(data.u) == (P * num_bases, tsteps, ntrials)
    @test size(penalty) == (P * num_bases, P * num_bases)
    @test eltype(data.u) === T
    @test eltype(penalty) === T

    K = num_bases
    # Recover B' from the first predictor block of trial 1, then verify all
    # other (predictor, trial) blocks match `trial_pred[n, p] * Bt_recovered`.
    coeff_ref = trial_pred[1, 1]
    @assert abs(coeff_ref) > 1e-6
    Bt_recovered = data.u[1:K, :, 1] ./ coeff_ref

    for n in 1:ntrials, p in 1:P
        rows = ((p - 1) * K + 1):(p * K)
        @test data.u[rows, :, n] ≈ trial_pred[n, p] .* Bt_recovered atol = 1e-12
    end
end

function test_spline_inputs_partition_of_unity()
    T = Float64
    tsteps, ntrials = 25, 1
    num_bases = 6

    y = randn(T, 1, tsteps, ntrials)
    u = zeros(T, num_bases, tsteps, ntrials)
    d = zeros(T, 1, tsteps, ntrials)
    data = Data(; y=y, u=u, d=d)

    _ = generate_spline_inputs!(data, num_bases; target=:u)

    # Cubic B-splines built via averagebasis form a partition of unity at
    # interior evaluation points; columns of B (== rows of Bt == data.u[:,t,1])
    # should sum to 1.
    col_sums = vec(sum(data.u[:, :, 1]; dims=1))
    @test all(abs.(col_sums .- one(T)) .< 1e-10)
end

function test_spline_inputs_default_trial_pred_broadcasts_across_trials()
    T = Float64
    tsteps, ntrials = 30, 5
    num_bases = 7

    y = randn(T, 1, tsteps, ntrials)
    u = zeros(T, num_bases, tsteps, ntrials)
    d = zeros(T, 1, tsteps, ntrials)
    data = Data(; y=y, u=u, d=d)

    _ = generate_spline_inputs!(data, num_bases)

    for n in 2:ntrials
        @test data.u[:, :, n] ≈ data.u[:, :, 1] atol = 1e-14
    end
end

function test_spline_inputs_target_d()
    T = Float64
    tsteps, ntrials = 25, 3
    num_bases = 5

    y = randn(T, 2, tsteps, ntrials)
    u = zeros(T, 1, tsteps, ntrials)
    d = zeros(T, num_bases, tsteps, ntrials)
    data = Data(; y=y, u=u, d=d)

    _ = generate_spline_inputs!(data, num_bases; target=:d)

    @test all(data.u .== 0)
    @test !all(iszero, data.d)
end

function test_spline_inputs_penalty_nullspace()
    T = Float64
    tsteps, ntrials = 30, 2
    num_bases = 10
    P = 2

    y = randn(T, 1, tsteps, ntrials)
    u = zeros(T, P * num_bases, tsteps, ntrials)
    d = zeros(T, 1, tsteps, ntrials)
    trial_pred = ones(T, ntrials, P)
    data = Data(; y=y, u=u, d=d, trial_pred=trial_pred)

    penalty = generate_spline_inputs!(data, num_bases; penalty_order=2)

    @test issymmetric(penalty)
    eigs = eigvals(Symmetric(penalty))
    @test all(eigs .>= -1e-10)
    # For a 2nd-order P-spline penalty, constants and linears are in the null
    # space *of each predictor block*, so total nullity = 2*P.
    @test count(abs.(eigs) .< 1e-8) == 2 * P
end

function test_difference_matrix_nullspace()
    T = Float64
    K = 12
    D = StateSpaceDynamics._difference_matrix(K, 2, T)
    @test size(D) == (K - 2, K)

    @test maximum(abs.(D * ones(T, K))) < 1e-12
    @test maximum(abs.(D * collect(T, 1:K))) < 1e-12
    # Quadratics are not in the 2nd-difference nullspace.
    @test maximum(abs.(D * (collect(T, 1:K) .^ 2))) > 1.0

    D0 = StateSpaceDynamics._difference_matrix(K, 0, T)
    @test D0 == Matrix{T}(I, K, K)
end

function test_spline_inputs_size_mismatch_throws()
    T = Float64
    tsteps, ntrials = 20, 2
    num_bases = 4
    P = 2

    y = randn(T, 1, tsteps, ntrials)
    # Wrong size: should be P*num_bases = 8 rows.
    u = zeros(T, 5, tsteps, ntrials)
    d = zeros(T, 1, tsteps, ntrials)
    data = Data(; y=y, u=u, d=d, trial_pred=ones(T, ntrials, P))

    @test_throws DimensionMismatch generate_spline_inputs!(data, num_bases)
end

function test_spline_inputs_trial_pred_row_mismatch_throws()
    T = Float64
    tsteps, ntrials = 20, 4
    num_bases = 4

    y = randn(T, 1, tsteps, ntrials)
    u = zeros(T, num_bases, tsteps, ntrials)
    d = zeros(T, 1, tsteps, ntrials)
    # trial_pred has 3 rows but data.y has 4 trials.
    data = Data(; y=y, u=u, d=d, trial_pred=ones(T, 3, 1))

    @test_throws DimensionMismatch generate_spline_inputs!(data, num_bases)
end

function test_spline_inputs_manual_knots()
    T = Float64
    tsteps, ntrials = 40, 2
    num_bases = 8
    order = 4
    knots = collect(range(T(1), T(tsteps); length=num_bases - order + 2))

    y = randn(T, 1, tsteps, ntrials)
    u = zeros(T, num_bases, tsteps, ntrials)
    d = zeros(T, 1, tsteps, ntrials)
    data = Data(; y=y, u=u, d=d)

    penalty = generate_spline_inputs!(data, num_bases; knots=knots, order=order)
    @test size(penalty) == (num_bases, num_bases)
    @test size(data.u) == (num_bases, tsteps, ntrials)
    col_sums = vec(sum(data.u[:, :, 1]; dims=1))
    @test all(abs.(col_sums .- one(T)) .< 1e-10)
end

function test_spline_inputs_manual_knots_wrong_length_throws()
    T = Float64
    tsteps, ntrials = 20, 1
    num_bases = 6
    order = 4
    # Required length is num_bases - order + 2 = 4. Pass 5 instead.
    bad_knots = collect(range(T(1), T(tsteps); length=5))

    y = randn(T, 1, tsteps, ntrials)
    u = zeros(T, num_bases, tsteps, ntrials)
    d = zeros(T, 1, tsteps, ntrials)
    data = Data(; y=y, u=u, d=d)

    @test_throws ArgumentError generate_spline_inputs!(
        data, num_bases; knots=bad_knots, order=order
    )
end

function test_spline_inputs_float32_type_preservation()
    T = Float32
    tsteps, ntrials = 20, 2
    num_bases = 5
    P = 2

    y = randn(T, 1, tsteps, ntrials)
    u = zeros(T, P * num_bases, tsteps, ntrials)
    d = zeros(T, 1, tsteps, ntrials)
    trial_pred = randn(T, ntrials, P)
    data = Data(; y=y, u=u, d=d, trial_pred=trial_pred)

    penalty = generate_spline_inputs!(data, num_bases)
    @test eltype(data.u) === T
    @test eltype(penalty) === T
end

function test_spline_inputs_invalid_args()
    T = Float64
    tsteps, ntrials = 20, 1
    y = randn(T, 1, tsteps, ntrials)
    u = zeros(T, 4, tsteps, ntrials)
    d = zeros(T, 1, tsteps, ntrials)
    data = Data(; y=y, u=u, d=d)

    @test_throws ArgumentError generate_spline_inputs!(data, 4; target=:bogus)
    # num_bases < order
    @test_throws ArgumentError generate_spline_inputs!(data, 3; order=4)
    # penalty_order >= num_bases
    @test_throws ArgumentError generate_spline_inputs!(data, 4; penalty_order=4)
    # penalty_order < 0
    @test_throws ArgumentError generate_spline_inputs!(data, 4; penalty_order=-1)
end
