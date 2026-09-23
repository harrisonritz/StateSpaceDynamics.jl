#=
The EM stopping rule: `|ΔELBO| < max(tol, rtol · |ELBO|)`, with `rtol = 0` by
default so that the absolute test alone is what every existing call gets.
=#

"""Four trials, each drawn from a seed of its own. (The multi-trial `rand` no
longer depends on the thread layout either, but these stopping points were
tuned on these draws.)"""
function convergence_draws(model, seed)
    return [rand(StableRNG(seed + i), model, 60)[2] for i in 1:4]
end

"""A small Gaussian LDS, its data, and a perturbed copy to fit from."""
function convergence_gaussian()
    rng = StableRNG(311)
    truth = LinearDynamicalSystem(;
        state_model=GaussianStateModel(;
            A=[0.95 0.1; -0.1 0.95],
            Q=Matrix(0.02I, 2, 2),
            b=zeros(2),
            x0=zeros(2),
            P0=Matrix(0.1I, 2, 2),
        ),
        obs_model=GaussianObservationModel(;
            C=randn(rng, 3, 2), d=zeros(3), R=Matrix(0.05I, 3, 3)
        ),
        latent_dim=2,
        obs_dim=3,
        fit_bool=fill(true, 6),
    )
    y = convergence_draws(truth, 311)
    start = deepcopy(truth)
    start.state_model.A .= [0.8 0.0; 0.0 0.8]
    start.obs_model.C .+= 0.3 .* randn(rng, 3, 2)
    return start, y
end

"""A small Poisson LDS: the Laplace-EM loop."""
function convergence_poisson()
    rng = StableRNG(312)
    truth = LinearDynamicalSystem(;
        state_model=GaussianStateModel(;
            A=[0.95 0.1; -0.1 0.95],
            Q=Matrix(0.02I, 2, 2),
            b=zeros(2),
            x0=zeros(2),
            P0=Matrix(0.1I, 2, 2),
        ),
        obs_model=PoissonObservationModel(; C=0.5 .* randn(rng, 3, 2), d=fill(0.5, 3)),
        latent_dim=2,
        obs_dim=3,
        fit_bool=fill(true, 6),
    )
    y = convergence_draws(truth, 312)
    start = deepcopy(truth)
    start.state_model.A .= [0.8 0.0; 0.0 0.8]
    return start, y
end

"""A small inverse-LQR model: the loop that stops without a final M-step."""
function convergence_lqr()
    rng = StableRNG(313)
    sm = LQRStateModel(
        [0.96 0.07; -0.05 0.93],
        [0.06 0.01; 0.01 0.05],
        [0.25 0.04; 0.04 0.18],
        Matrix(0.05I, 4, 4);
        P0=Matrix(0.3I, 4, 4),
    )
    C = randn(rng, 3, 4)
    C[:, 3:4] .= 0
    lds = LinearDynamicalSystem(
        sm, GaussianObservationModel(C, Matrix(0.1I, 3, 3), zeros(3))
    )
    y = [0.5 .* randn(rng, 3, 30) for _ in 1:4]
    return lds, y
end

"""The rule itself: absolute alone at `rtol = 0`, the looser of the two
otherwise, and never at the first iterate."""
function test_em_converged_rule()
    e = [-1000.0, -10.0, -9.99999, -9.999989]     # steps 990, 1e-5, 1e-6
    @test !SSD._em_converged(e, 1, Inf, 0.0)
    @test !SSD._em_converged(e, 3, 1e-6, 0.0)
    @test SSD._em_converged(e, 4, 1e-5, 0.0)
    @test SSD._em_converged(e, 3, 1e-6, 1e-5)      # 1e-5 < 1e-5 · 10 ...
    @test !SSD._em_converged(e, 3, 1e-6, 1e-7)     # ... but not < 1e-7 · 10
    @test SSD._em_converged(e, 4, 1e-5, 1e-12)     # a tiny rtol never tightens `tol`
    @test !SSD._em_converged(e, 2, 1e-6, 1e-2)     # a large step is a large step
    return nothing
end

"""On a real fit of each loop — Gaussian, Laplace (Poisson) and inverse LQR,
whose loop stops without a final M-step — `rtol = 0` reproduces the
absolute-only trace exactly, and a positive `rtol` visits the same iterates but
stops at the first one that meets it, which is earlier."""
function test_em_relative_tolerance()
    for make in (convergence_gaussian, convergence_poisson, convergence_lqr)
        model, y = make()
        function fitted(; kwargs...)
            return collect(fit!(deepcopy(model), y; max_iter=60, progress=false, kwargs...))
        end
        absolute = fitted(; tol=1e-8)
        @test absolute == fitted(; tol=1e-8, rtol=0.0)

        rtol = 1e-3       # meets each of these fits partway through its sixty iterations
        relative = fitted(; tol=1e-8, rtol=rtol)
        n = length(relative)
        @test 2 <= n < length(absolute)
        @test relative == absolute[1:n]
        @test abs(relative[n] - relative[n - 1]) < rtol * abs(relative[n])
        @test all(
            abs(relative[i] - relative[i - 1]) >= rtol * abs(relative[i]) for i in 2:(n - 1)
        )
    end
    return nothing
end
