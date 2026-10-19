function _initial_prior_fixture(q; legacy=false)
    state = GaussianStateModel(;
        A=[0.7 0.1; 0 0.6],
        Q=[0.2 0; 0 0.1],
        b=zeros(2),
        x0=[0.2, -0.1],
        P0=[0.8 0.1; 0.1 0.5],
        B0=legacy ? zeros(2, 0) : zeros(2, q),
    )
    obs = GaussianObservationModel(;
        C=Matrix{Float64}(I, 2, 2), R=Matrix(0.2I, 2, 2), d=zeros(2)
    )
    return LinearDynamicalSystem(state, obs)
end

function _initial_prior_stats(model, U, means, weights)
    suf = SSD._initialize_td_sufficient_statistics(Float64, model, [3])
    suf.init_n = sum(weights)
    suf.init_xy .= reshape(means * weights, 1, :)
    if !isempty(model.state_model.B0)
        suf.init_u0u0 .= U * Diagonal(weights) * U'
        suf.init_u0y .= U * Diagonal(weights) * means'
    end
    suf.init_yy[] = means * Diagonal(weights) * means' + sum(weights) * [0.2 0.03; 0.03 0.1]
    return suf
end

function test_initial_mniw_priors()
    @testset "Initial regression MNIW joint MAP" begin
        # Fractional trial weights, nonzero mean, correlated column precision,
        # and a nonisotropic IW scale exercise the full matrix prior.
        U = [1.0 1 1 1 1; -1 0 1 2 0.5; 0 1 0 1 1]
        means = [0.2 1.0 -0.4 0.6 0.8; -0.1 0.5 0.9 -0.4 0.3]
        weights = [0.2, 0.6, 0.9, 0.3, 0.7]
        M = [0.3 -0.2 0.1; -0.1 0.4 0.2]
        Λ = [2.0 0.2 -0.1; 0.2 1.0 0.1; -0.1 0.1 0.7]
        Ψ, ν = [0.6 0.1; 0.1 0.4], 5.0
        model = _initial_prior_fixture(3)
        sm = model.state_model
        sm.x0_prior = MNPrior(; M₀=M, Λ)
        sm.P0_prior = IWPrior(; Ψ, ν)
        suf = _initial_prior_stats(model, U, means, weights)
        W = Diagonal(weights)
        Λn = Λ + U * W * U'
        expected_B = (M * Λ + means * W * U') / Λn
        Ψn = Ψ + suf.init_yy[] + M * Λ * M' - expected_B * Λn * expected_B'
        expected_P = Matrix(Symmetric(Ψn / (ν + sum(weights) + 2 + 3 + 1)))
        sws = SSD.SmoothWorkspace(Float64, 2, 2, 3)
        SSD.update_initial_state_mean!(model, suf)
        SSD.update_initial_state_covariance!(model, suf, sws)
        @test sm.B0 ≈ expected_B atol = 1e-12
        @test sm.P0 ≈ expected_P atol = 1e-12
        @test sm.x0 == [0.2, -0.1] # unused legacy field is not another intercept

        # Independently normalized vector-Gaussian + IW density catches missing
        # q*logdet(P0), not just the coefficient shrinkage quadratic.
        function density(P)
            return logpdf(MvNormal(vec(M), Symmetric(kron(inv(Λ), P))), vec(sm.B0)) +
                   logpdf(InverseWishart(ν, Ψ), P)
        end
        base = SSD._state_prior_logdensity(model, nothing)
        base_density = density(sm.P0)
        for P in ([0.4 0.02; 0.02 0.7], [0.9 -0.1; -0.1 0.5])
            sm.P0 .= P
            @test SSD._state_prior_logdensity(model, nothing) - base ≈
                density(P) - base_density atol = 1e-10
        end
        sm.P0 .= expected_P

        function objective(B, P)
            scatter =
                suf.init_yy[] - B * suf.init_u0y - suf.init_u0y' * B' +
                B * suf.init_u0u0 * B'
            return -0.5 * (sum(weights) * logdet(P) + tr(P \ scatter)) +
                   logpdf(MvNormal(vec(M), Symmetric(kron(inv(Λ), P))), vec(B)) +
                   logpdf(InverseWishart(ν, Ψ), P)
        end
        eps = 1e-6
        for i in eachindex(sm.B0)
            delta = zeros(size(sm.B0))
            delta[i] = eps
            @test abs(
                (objective(sm.B0 + delta, sm.P0) - objective(sm.B0 - delta, sm.P0)) / (2eps)
            ) < 1e-6
        end
        for (i, j) in ((1, 1), (2, 2), (1, 2))
            delta = zeros(2, 2)
            delta[i, j] = delta[j, i] = eps
            @test abs(
                (objective(sm.B0, sm.P0 + delta) - objective(sm.B0, sm.P0 - delta)) / (2eps)
            ) < 1e-6
        end
    end

    @testset "Shared initial coefficients with distinct P0 require GLS" begin
        for legacy in (false, true)
            q = legacy ? 1 : 2
            models = [_initial_prior_fixture(q; legacy) for _ in 1:3]
            for model in models
                model.state_model.x0_prior = MNPrior(;
                    M₀=fill(0.1, 2, q), Λ=Matrix(0.7I, q, q)
                )
                model.state_model.P0_prior = IWPrior(; Ψ=Matrix(0.2I, 2, 2), ν=4.0)
            end
            for model in models[2:3]
                if legacy
                    model.state_model.x0 = models[1].state_model.x0
                else
                    model.state_model.B0 = models[1].state_model.B0
                end
            end
            models[2].state_model.P0 = models[1].state_model.P0
            models[3].state_model.P0 .= [0.1 -0.02; -0.02 1.5]
            U = legacy ? ones(1, 3) : [1.0 1 1; -1 0 2]
            means = [0.2 1.0 -0.4; -0.1 0.5 0.9]
            sufs = [
                _initial_prior_stats(models[i], U, means .+ i, [0.2, 0.6, 0.9] .* i) for
                i in 1:3
            ]
            bufs = SSD.GroupedSufBuffers(Float64, models[1], [3])
            SSD._grouped_update_x0!(models, sufs, [1, 1, 1], [1, 1, 2], bufs)
            B = SSD._initial_coefficients(models[1].state_model)
            grad = zeros(size(B))
            for i in 1:3
                sm, suf = models[i].state_model, sufs[i]
                XX = legacy ? fill(suf.init_n, 1, 1) : suf.init_u0u0
                XY = legacy ? suf.init_xy : suf.init_u0y
                grad .+= sm.P0 \ (B * XX - XY')
                if i in (1, 3) # one prior per coefficient/covariance pair
                    grad .+= sm.P0 \ ((B - sm.x0_prior.M₀) * sm.x0_prior.Λ)
                end
            end
            @test norm(grad) < 1e-10
            expected = Matrix{Float64}[]
            for inds in ([1, 2], [3])
                S = zeros(2, 2)
                N = sum(sufs[i].init_n for i in inds)
                for i in inds
                    suf = sufs[i]
                    XX = legacy ? fill(suf.init_n, 1, 1) : suf.init_u0u0
                    XY = legacy ? suf.init_xy : suf.init_u0y
                    S .+= suf.init_yy[] - B * XY - XY' * B' + B * XX * B'
                end
                prior = models[first(inds)].state_model.x0_prior
                delta = B - prior.M₀
                S .+= delta * prior.Λ * delta'
                push!(expected, (S + 0.2I) / (4 + N + q + 2 + 1))
            end
            sws = SSD.SmoothWorkspace(Float64, 2, 2, 3)
            SSD._grouped_update_P0!(models, sufs, [1, 1, 2], [1, 1, 1], sws)
            @test models[1].state_model.P0 ≈ expected[1] atol = 1e-12
            @test models[3].state_model.P0 ≈ expected[2] atol = 1e-12
            @test models[1].state_model.P0 === models[2].state_model.P0
        end
    end

    @testset "Constant initial input is the legacy x0 model" begin
        legacy = _initial_prior_fixture(1; legacy=true)
        regression = _initial_prior_fixture(1)
        regression.state_model.B0[:, 1] .= legacy.state_model.x0
        for model in (legacy, regression)
            model.state_model.x0_prior = x0_mean_prior([0.1, -0.2]; κ₀=2.0)
            model.state_model.P0_prior = IWPrior(; Ψ=Matrix(0.5I, 2, 2), ν=5.0)
        end
        y = [randn(StableRNG(i), 2, 5) for i in 1:6]
        a = fit!(legacy, y; max_iter=5, progress=false)
        b = fit!(regression, y; ux0=ones(1, 6), max_iter=5, progress=false)
        @test a ≈ b atol = 1e-9
        @test legacy.state_model.x0 ≈ vec(regression.state_model.B0) atol = 1e-10
        @test legacy.state_model.P0 ≈ regression.state_model.P0 atol = 1e-10
    end
end
