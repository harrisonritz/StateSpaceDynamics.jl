using Test, StateSpaceDynamics, LinearAlgebra, Random

function diagnostic_fixture(; poisson=false, grouped=false)
    labels = ["low", "low", "high", "high"]
    state = GaussianStateModel(;
        A=Matrix(0.85I, 2, 2),
        Q=Matrix(0.15I, 2, 2),
        b=[0.1, -0.1],
        x0=zeros(2),
        P0=Matrix(1.0I, 2, 2),
        depends_on=grouped ? (A=labels,) : nothing,
    )
    C = [0.3 0.1; 0.1 -0.2; -0.2 0.2]
    obs = if poisson
        PoissonObservationModel(; C, d=fill(-1.0, 3))
    else
        GaussianObservationModel(; C, R=Matrix(0.2I, 3, 3), d=fill(-1.0, 3))
    end
    return LinearDynamicalSystem(state, obs)
end

@testset "fit checkpoint diagnostics" begin
    for poisson in (false, true), grouped in (false, true)
        model = diagnostic_fixture(; poisson, grouped)
        rng = MersenneTwister(43)
        y = if poisson
            [Float64.(rand(rng, 0:1, 3, 15)) for _ in 1:4]
        else
            [randn(rng, 3, 15) for _ in 1:4]
        end
        plain, measured = deepcopy(model), deepcopy(model)
        baseline = fit!(
            plain, y; y_test=y, max_iter=3, progress=false, test_diagnostics=false
        )
        records = Any[]
        trace = fit!(
            measured,
            y;
            y_test=y,
            max_iter=3,
            progress=false,
            test_callback=r -> push!(records, r),
        )
        @test baseline.train == trace.train
        @test baseline.test == trace.test
        @test plain.state_model.A == measured.state_model.A
        @test plain.obs_model.C == measured.obs_model.C
        @test isempty(baseline.diagnostics)
        @test length(trace.diagnostics) == 4
        @test records == trace.diagnostics
        @test records[end]["event"] == "returned"
        first = records[1]["diagnostics"]
        @test first["latent"]["n_bins"] == 60
        @test length(first["states"]) == (grouped ? 2 : 1)
        @test all(!haskey(row, "diagnostic_error") for row in first["states"])
        @test first["states"][1]["A_spectral_radius"] ≈ 0.85
        @test first["states"][1]["Q_condition"] ≈ 1
        @test first["observations"][1]["d_maxabs"] ≈ 1
        @test haskey(first["observations"][1], "d_plus_C_mean_norm")
    end
    # Stable eigenvalues can conceal large transient amplification.
    model = diagnostic_fixture()
    model.state_model.A .= [0.9 1000.0; 0 0.9]
    row = parameter_diagnostics(model)["states"][1]
    @test row["A_spectral_radius"] ≈ 0.9
    @test row["A_opnorm"] > 1000
    @test row["I_minus_A_condition"] > 1e7
    model.state_model.Q .= 0
    row = parameter_diagnostics(model)["states"][1]
    @test !row["Q_positive_definite"]
    @test row["Q_condition"] === nothing
    # A failed held-out likelihood still delivers the diagnostic record.
    bad = StateSpaceDynamics._holdout_monitor(
        Float64, zeros(2, 4); test_callback=r -> nothing
    )
    @test_throws Exception StateSpaceDynamics._holdout_record!(bad, diagnostic_fixture(), 1)
    @test length(bad.diagnostics) == 1
    @test haskey(bad.diagnostics[1], "error")
    # Previous positional FitTrace constructor remains supported.
    legacy = StateSpaceDynamics.FitTrace{Float64}([1.0], [2.0], [1], 1, false, 1)
    @test isempty(legacy.diagnostics)
end
