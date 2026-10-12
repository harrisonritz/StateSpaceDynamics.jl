#=============================================================================
Two opt-in Gaussian-emission options.

- `GaussianObservationModel(...; R_diagonal=true)`: the M-step keeps the diagonal
  of the residual scatter, the exact maximiser of the Gaussian objective once the
  off-diagonals are pinned at zero. Since the C/d update does not depend on R,
  one EM step from the same start must give the diagonal of the full-R step.
- `CompositeObservationModel(models; quadratic=false)`: an all-Gaussian composite
  on the trial-by-trial Laplace path. Its likelihood is the batched exact one to
  rounding, and unlike the batched path it fits per-trial cost offsets.

Included after `LQRCostOffsets.jl`, whose fixtures it reuses.
=============================================================================#

function _go_lds(rng; R_diagonal::Bool, p::Int=5, d::Int=2)
    A = [0.9 0.1; -0.1 0.9]
    state = GaussianStateModel(;
        A=A, Q=Matrix(0.1I, d, d), b=zeros(d), x0=zeros(d), P0=Matrix(1.0I, d, d)
    )
    C = randn(StableRNG(7), p, d)
    obs = GaussianObservationModel(;
        C=C, R=Matrix(0.5I, p, p), d=zeros(p), R_diagonal=R_diagonal
    )
    return LinearDynamicalSystem(state, obs)
end

function test_r_diagonal_is_the_diagonal_of_the_full_update()
    rng = StableRNG(11)
    full = _go_lds(rng; R_diagonal=false)
    diag_ = _go_lds(rng; R_diagonal=true)
    ys = [randn(StableRNG(12 + i), 5, 30) .+ randn(StableRNG(40 + i), 5) for i in 1:6]
    fit!(full, ys; max_iter=1, progress=false)
    fit!(diag_, ys; max_iter=1, progress=false)
    R_full, R_diag = full.obs_model.R, diag_.obs_model.R
    @test R_diag ≈ Diagonal(diag(R_full)) atol = 1e-10
    @test all(>(0), diag(R_diag))
    @test full.obs_model.C ≈ diag_.obs_model.C atol = 1e-10
    # It stays diagonal through further iterations, and EM stays monotone.
    els = fit!(diag_, ys; max_iter=5, progress=false)
    @test isdiag(diag_.obs_model.R)
    @test all(diff(els) .>= -1e-8 * maximum(abs, els))
end

function test_r_diagonal_survives_grouping()
    rng = StableRNG(13)
    lds = _go_lds(rng; R_diagonal=true)
    labels = repeat(["a", "b"], 3)
    lds.obs_model.depends_on = (C=labels, d=labels, R=labels)
    ys = [randn(StableRNG(50 + i), 5, 25) for i in 1:6]
    fit!(lds, ys; max_iter=3, progress=false)
    for label in ("a", "b")
        @test isdiag(group_parameter(lds.obs_model, :R, label))
    end
end

function test_composite_quadratic_keyword()
    C = randn(StableRNG(14), 3, 2)
    gauss = (a=GaussianObservationModel(C, Matrix(0.3I, 3, 3), zeros(3)),
             b=GaussianObservationModel(C[1:2, :], Matrix(0.2I, 2, 2), zeros(2)))
    @test SSD._emission_is_quadratic(CompositeObservationModel(gauss))
    @test !SSD._emission_is_quadratic(CompositeObservationModel(gauss; quadratic=false))
    mixed = (a=PoissonObservationModel(C, fill(0.5, 3)), b=gauss.b)
    @test_throws ArgumentError CompositeObservationModel(mixed; quadratic=true)
    @test !SSD._emission_is_quadratic(CompositeObservationModel(mixed))
end

function test_laplace_composite_matches_exact_and_fits_offsets()
    sm, _ = _co_model(StableRNG(15); poisson=false)
    n = SSD.plant_dim(sm)
    members() = (
        spk=GaussianObservationModel(
            [randn(StableRNG(16), 4, n) zeros(4, n)], Matrix(0.4I, 4, 4), zeros(4)
        ),
        kin=GaussianObservationModel(
            [randn(StableRNG(17), 2, n) zeros(2, n)], Matrix(0.1I, 2, 2), zeros(2)
        ),
    )
    ys = (spk=[randn(StableRNG(18 + i), 4, t) for (i, t) in enumerate(CO_LENGTHS)],
          kin=[randn(StableRNG(30 + i), 2, t) for (i, t) in enumerate(CO_LENGTHS)])
    uys = (spk=[zeros(0, t) for t in CO_LENGTHS], kin=[zeros(0, t) for t in CO_LENGTHS])

    exact = LinearDynamicalSystem(deepcopy(sm), members())
    laplace = LinearDynamicalSystem(
        deepcopy(sm), CompositeObservationModel(members(); quadratic=false)
    )
    @test elbo(exact, ys) ≈ elbo(laplace, ys; uy=uys) rtol = 1e-10
    xa, _ = smooth(exact, ys)
    xb, _ = smooth(laplace, ys; uy=uys)
    @test maximum(maximum(abs.(a .- b)) for (a, b) in zip(xa, xb)) < 1e-8

    # The batched exact path refuses per-trial offsets; the Laplace one fits them.
    @test_throws ArgumentError fit!(
        LinearDynamicalSystem(deepcopy(sm), members()), ys;
        max_iter=1, progress=false, cost_offset=CO_OFFSETS,
    )
    els = fit!(laplace, ys; uy=uys, max_iter=3, progress=false, cost_offset=CO_OFFSETS)
    @test all(isfinite, els)
end
