"""Trial-level initial predictors must reach inference, EM, and sampling."""
function test_initial_inputs()
    y = [reshape([1.0, 1.0, 1.0], 1, 3), reshape([3.0, 3.0, 3.0], 1, 3)]
    u0 = [1.0 1.0; 0.0 1.0]
    state = GaussianStateModel(;
        A=reshape([0.9], 1, 1),
        Q=reshape([0.1], 1, 1),
        b=zeros(1),
        x0=[9.0],
        P0=reshape([0.1], 1, 1),
        B0=zeros(1, 2),
    )
    obs = GaussianObservationModel(;
        C=reshape([1.0], 1, 1), R=reshape([0.1], 1, 1), d=zeros(1)
    )
    lds = LinearDynamicalSystem(state, obs)
    @test_throws ArgumentError smooth(lds, y)
    @test_throws SSD.DimensionMismatchError smooth(lds, y; ux0=ones(1, 2))
    fit!(lds, y; ux0=u0, max_iter=2, progress=false)
    @test state.B0[1, 2] > 0.5
    @test sum(trial_elbos(lds, y; ux0=u0)) ≈ elbo(lds, y; ux0=u0) atol = 1e-6

    # The stored x0 is ignored when B0 is present. With a common RNG, changing
    # only one trial's initial input shifts its sampled x1 by B0 times that change.
    x, _ = rand(StableRNG(21), lds, [3, 3]; ux0=u0)
    changed = copy(u0)
    changed[2, 2] += 2.0
    x_changed, _ = rand(StableRNG(21), lds, [3, 3]; ux0=changed)
    @test x_changed[1][:, 1] ≈ x[1][:, 1]
    @test x_changed[2][1, 1] - x[2][1, 1] ≈ 2 * state.B0[1, 2]

    switching = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[deepcopy(lds)])
    @test isfinite(smooth(switching, y; ux0=u0, smoothing_iters=2, progress=false).elbo)
    fit!(switching, y; ux0=u0, max_iter=2, smoothing_iters=1, progress=false)
    @test all(isfinite, switching.LDSs[1].state_model.B0)
end

"""A terminal normalizer must distinguish trials with different initial means."""
function test_lqr_initial_inputs_terminal()
    state = LQRStateModel(
        reshape([0.9], 1, 1),
        reshape([0.1], 1, 1),
        [reshape([0.2], 1, 1)],
        Matrix(0.05I, 2, 2);
        terminal=true,
        condition_terminal=true,
        Σf=reshape([0.1], 1, 1),
        P0=Matrix(0.2I, 2, 2),
        B0=[0.0 2.0; 0.0 0.0],
    )
    obs = GaussianObservationModel([1.0 0.0], reshape([0.1], 1, 1), [0.0])
    lds = LinearDynamicalSystem(state, obs)
    u0 = [1.0 1.0; 0.0 1.0]
    inputs = [zeros(0, 3), zeros(0, 3)]
    values = terminal_normalizer(lds, inputs; ux0=u0)
    @test abs(values[1] - values[2]) > 1e-3
    y = [ones(1, 3), 2 .* ones(1, 3)]
    @test isfinite(elbo(lds, y; ux0=u0))
    @test sum(trial_elbos(lds, y; ux0=u0)) ≈ elbo(lds, y; ux0=u0) atol = 1e-6
end
