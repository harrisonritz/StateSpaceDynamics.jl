function _null_make_data(
    rng::AbstractRNG,
    obs_dim::Int,
    tsteps::Int,
    ntrials::Int;
    v_dim::Int=0,
    T::Type=Float64,
)
    y = randn(rng, T, obs_dim, tsteps, ntrials)
    u = randn(rng, T, v_dim, tsteps, ntrials)
    d = zeros(T, 0, tsteps, ntrials)
    return Data(; y=y, u=u, d=d)
end

function _null_make_var_data(
    rng::AbstractRNG,
    F::AbstractMatrix{T},
    d_vec::AbstractVector{T},
    R::AbstractMatrix{T},
    μ_0::AbstractVector{T},
    R_0::AbstractMatrix{T},
    tsteps::Int,
    ntrials::Int,
) where {T<:Real}
    obs_dim = length(d_vec)
    y = zeros(T, obs_dim, tsteps, ntrials)
    R_0_chol = cholesky(Symmetric(R_0)).L
    R_chol = cholesky(Symmetric(R)).L
    for n in 1:ntrials
        y[:, 1, n] .= μ_0 .+ R_0_chol * randn(rng, T, obs_dim)
        for t in 2:tsteps
            y[:, t, n] .= F * y[:, t - 1, n] .+ d_vec .+ R_chol * randn(rng, T, obs_dim)
        end
    end
    u = zeros(T, 0, tsteps, ntrials)
    d = zeros(T, 0, tsteps, ntrials)
    return Data(; y=y, u=u, d=d)
end

function test_null_intercept_matches_mvnormal_loglik()
    T = Float64
    obs_dim, tsteps, ntrials = 3, 20, 5
    rng = StableRNG(0xC0FFEE)
    data = _null_make_data(rng, obs_dim, tsteps, ntrials)

    res = test_null(data)

    # Closed-form MLE: d = mean over (t,n), R = empirical covariance (with /n).
    Y = reshape(data.y, obs_dim, tsteps * ntrials)
    n = size(Y, 2)
    d_hat = vec(mean(Y; dims=2))
    Yc = Y .- d_hat
    R_hat = (Yc * Yc') ./ n

    ref_ll = -0.5 * (
        n * obs_dim * log(2π) + n * logdet(R_hat) +
        sum(Yc[:, i]' * (R_hat \ Yc[:, i]) for i in 1:n)
    )

    @test res.intercept.train_ll ≈ ref_ll atol = 1e-8 rtol = 1e-8
    @test res.intercept.params.d ≈ d_hat atol = 1e-10
    @test res.intercept.params.R ≈ R_hat atol = 1e-10
    @test res.intercept.test_ll === nothing
end

function test_null_test_ll_matches_plugin_gaussian()
    T = Float64
    obs_dim, tsteps, ntrials = 2, 15, 4
    rng = StableRNG(7)
    train = _null_make_data(rng, obs_dim, tsteps, ntrials)
    test = _null_make_data(rng, obs_dim,12, 3)

    res = test_null(train; test_data=test)

    d_hat = res.intercept.params.d
    R_hat = res.intercept.params.R
    Y_te = reshape(test.y, obs_dim, 12 * 3)
    Yc = Y_te .- d_hat
    n_te = size(Y_te, 2)
    ref = -0.5 * (
        n_te * obs_dim * log(2π) + n_te * logdet(R_hat) +
        sum(Yc[:, i]' * (R_hat \ Yc[:, i]) for i in 1:n_te)
    )
    @test res.intercept.test_ll ≈ ref atol = 1e-8 rtol = 1e-8
end

function test_null_inputs_collapses_to_intercept_when_no_inputs()
    T = Float64
    obs_dim, tsteps, ntrials = 3, 10, 6
    rng = StableRNG(42)
    data = _null_make_data(rng, obs_dim, tsteps, ntrials; v_dim=0)

    res = test_null(data)

    # With v_dim = 0 the inputs model has the same regressors as the intercept
    # model, and the VAR+inputs model matches the VAR-only model.
    @test res.inputs.train_ll ≈ res.intercept.train_ll atol = 1e-10
    @test res.var_inputs.train_ll ≈ res.var.train_ll atol = 1e-10
end

function test_null_inputs_helps_when_signal_present()
    T = Float64
    obs_dim, tsteps, ntrials = 4, 40, 10
    v_dim = 2
    rng = StableRNG(1234)

    D_true = randn(rng, T, obs_dim, v_dim)
    d_true = randn(rng, T, obs_dim)
    v = randn(rng, T, v_dim, tsteps, ntrials)
    R = Matrix{T}(0.1 * I, obs_dim, obs_dim)
    R_chol = cholesky(Symmetric(R)).L
    y = zeros(T, obs_dim, tsteps, ntrials)
    for n in 1:ntrials, t in 1:tsteps
        y[:, t, n] .= d_true .+ D_true * v[:, t, n] .+ R_chol * randn(rng, T, obs_dim)
    end
    data = Data(; y=y, u=v, d=zeros(T, 0, tsteps, ntrials))

    res = test_null(data)

    # The inputs model should fit strictly better than intercept-only given the
    # true generative process has D ≠ 0; the VAR+inputs model should be at
    # least as good as the VAR-only model.
    @test res.inputs.train_ll > res.intercept.train_ll
    @test res.var_inputs.train_ll > res.var.train_ll
    @test res.inputs.params.D ≈ D_true rtol = 0.15
    @test res.inputs.params.d ≈ d_true rtol = 0.2
end

function test_null_var_recovers_true_F_on_var_data()
    T = Float64
    obs_dim, tsteps, ntrials = 3, 200, 8
    rng = StableRNG(2025)

    F_true = T(0.7) * Matrix{T}(I, obs_dim, obs_dim) .+ T(0.05) .* randn(rng, T, obs_dim, obs_dim)
    d_true = zeros(T, obs_dim)
    R_true = Matrix{T}(0.05 * I, obs_dim, obs_dim)
    μ_0 = zeros(T, obs_dim)
    R_0 = Matrix{T}(0.2 * I, obs_dim, obs_dim)
    R0_prior = IWPrior(Ψ=Matrix{T}(0.1 * I, obs_dim, obs_dim), ν=T(obs_dim + 2))

    data = _null_make_var_data(rng, F_true, d_true, R_true, μ_0, R_0, tsteps, ntrials)
    res = test_null(data; R0_prior=R0_prior)

    @test res.var.params.F ≈ F_true atol = 0.05
    @test res.var.params.R ≈ R_true atol = 0.02
    # VAR model should beat intercept-only on truly autocorrelated data.
    @test res.var.train_ll > res.intercept.train_ll
end

function test_null_R_prior_shifts_LL_by_iw_logprior_term()
    # The training LL must include `iw_logprior_term(R, R_prior)` exactly. So
    # changing only the prior (keeping everything else fixed) shifts the LL by
    # the difference of the two prior terms — verified by fitting twice and
    # checking the delta.
    T = Float64
    obs_dim, tsteps, ntrials = 3, 20, 5
    rng = StableRNG(11)
    data = _null_make_data(rng, obs_dim, tsteps, ntrials)

    # Strong prior pulls R toward the prior mean; switch from "no prior" to
    # "weak prior" and check the LL shift matches the formula. (Strict
    # equality would require *also* the regression to be unchanged; with
    # a weak IW prior and intercept-only model that's true up to the iw_map
    # rescaling of R itself.)
    res_no = test_null(data)
    Ψ = Matrix{T}(0.1 * I, obs_dim, obs_dim)
    ν = T(obs_dim + 3)
    R_prior = IWPrior(Ψ=Ψ, ν=ν)
    res_with = test_null(data; R_prior=R_prior)

    # The IW prior term, by construction, must be the only change in the
    # training LL beyond R's MAP shift. Recompute the data LL by hand with
    # each fit's R and confirm `train_ll - data_only_ll ≈ iw_logprior_term`.
    function data_ll(W, R)
        Y = reshape(data.y, obs_dim, tsteps * ntrials)
        n = size(Y, 2)
        E = Y .- W * fill(one(T), 1, n)
        return -0.5 * (n * obs_dim * log(2π) + n * logdet(R) +
                       sum(E[:, i]' * (R \ E[:, i]) for i in 1:n))
    end

    W_no = reshape(res_no.intercept.params.d, obs_dim, 1)
    W_with = reshape(res_with.intercept.params.d, obs_dim, 1)
    @test res_no.intercept.train_ll ≈ data_ll(W_no, res_no.intercept.params.R) atol = 1e-8
    delta = res_with.intercept.train_ll - data_ll(W_with, res_with.intercept.params.R)
    # The remaining piece must be exactly the IW log-prior term (no MN prior).
    R_with = res_with.intercept.params.R
    expected = -0.5 * ((ν + obs_dim + 1) * logdet(R_with) + tr(R_with \ Ψ))
    @test delta ≈ expected atol = 1e-8
end

function test_null_inputs_override_uses_supplied_array()
    T = Float64
    obs_dim, tsteps, ntrials = 2, 12, 5
    v_dim = 3
    rng = StableRNG(55)

    data = _null_make_data(rng, obs_dim, tsteps, ntrials; v_dim=v_dim)
    # `train_data.u` has v_dim = 3; supply a 1-column override.
    v_override = randn(rng, T, 1, tsteps, ntrials)
    res = test_null(data; train_inputs=v_override)

    # `D` in the inputs model should have 1 column matching the override dim.
    @test size(res.inputs.params.D, 2) == 1
    @test size(res.var_inputs.params.D, 2) == 1
end

function test_null_var_requires_tsteps_ge_2()
    T = Float64
    rng = StableRNG(99)
    data = _null_make_data(rng, 2, 1, 4)
    @test_throws ArgumentError test_null(data)
end

function test_null_input_shape_mismatch_throws()
    T = Float64
    obs_dim, tsteps, ntrials = 2, 8, 3
    rng = StableRNG(123)
    data = _null_make_data(rng, obs_dim, tsteps, ntrials)
    bad_inputs = randn(rng, T, 2, tsteps + 1, ntrials)  # wrong tsteps
    @test_throws SSD.DimensionMismatchError test_null(data; train_inputs=bad_inputs)
end

function test_null_test_data_obs_dim_mismatch_throws()
    T = Float64
    rng = StableRNG(321)
    train = _null_make_data(rng, 3, 10, 4)
    test = _null_make_data(rng, 2, 10, 4)
    @test_throws SSD.DimensionMismatchError test_null(train; test_data=test)
end

function test_null_capacity_ordering_on_var_data()
    # On data generated from a VAR(1) process, training LL should be ordered:
    # intercept ≤ inputs (collapses when no inputs) and intercept ≤ var ≤
    # var_inputs (collapses to var when no inputs).
    T = Float64
    obs_dim, tsteps, ntrials = 3, 100, 6
    rng = StableRNG(9)

    F_true = T(0.6) * Matrix{T}(I, obs_dim, obs_dim)
    d_true = T(0.5) * ones(T, obs_dim)
    R_true = Matrix{T}(0.1 * I, obs_dim, obs_dim)
    μ_0 = zeros(T, obs_dim)
    R_0 = Matrix{T}(0.2 * I, obs_dim, obs_dim)

    data = _null_make_var_data(rng, F_true, d_true, R_true, μ_0, R_0, tsteps, ntrials)
    R0_prior = IWPrior(Ψ=Matrix{T}(0.1 * I, obs_dim, obs_dim), ν=T(obs_dim + 2))
    res = test_null(data; R0_prior=R0_prior)

    @test res.var.train_ll > res.intercept.train_ll
    @test res.var_inputs.train_ll ≈ res.var.train_ll atol = 1e-10
    @test res.inputs.train_ll ≈ res.intercept.train_ll atol = 1e-10
end
