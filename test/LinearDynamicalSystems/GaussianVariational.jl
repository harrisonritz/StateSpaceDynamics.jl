# Independent moment, invariance, failure-policy and integration checks.
# Kept as a callable testset so the full suite and focused runs share coverage.
function _vi_fixture(; grouped=false, composite=false, priors=true)
    k = 2
    p = 4
    sm = GaussianStateModel(;
        A=[0.7 0.1; 0 0.4],
        b=[0.2, -0.1],
        B=[0.1; 0.2;;],
        Q=[0.2 0.03; 0.03 0.3],
        x0=[1.0, -0.5],
        P0=[0.8 0.1; 0.1 0.6],
    )
    if priors
        sm.AB_prior = MNPrior(; M₀=zeros(k, 4), Λ=0.1Matrix{Float64}(I, 4, 4))
        sm.x0_prior = x0_mean_prior(zeros(k); κ₀=0.1)
        sm.Q_prior = IWPrior(; Ψ=0.1Matrix{Float64}(I, k, k), ν=3.0)
        sm.P0_prior = IWPrior(; Ψ=0.1Matrix{Float64}(I, k, k), ν=3.0)
    end
    labs = ["a", "a", "b", "b"]
    om = PoissonObservationModel(;
        C=randn(p, k) * 0.25,
        d=fill(-1.0, p),
        D=randn(p, 1) * 0.1,
        depends_on=grouped ? (; C=labs, d=labs, D=labs) : nothing,
    )
    priors && (om.CD_prior = MNPrior(; M₀=zeros(p, 4), Λ=0.1Matrix{Float64}(I, 4, 4)))
    ys = [Float64.(rand(0:2, p, t)) for t in [8, 10, 9, 11]]
    ux = [randn(1, size(y, 2)) for y in ys]
    uy = [randn(1, size(y, 2)) for y in ys]
    if composite
        kin = GaussianObservationModel(;
            C=[0.1 0.3], d=[0.2], R=reshape([0.2], 1, 1), D=reshape([0.1], 1, 1)
        )
        priors && (kin.CD_prior = MNPrior(; M₀=zeros(1, 4), Λ=0.1Matrix{Float64}(I, 4, 4)))
        om = CompositeObservationModel((spikes=om, kin=kin))
        ys = (spikes=ys, kin=[randn(1, t) for t in [8, 10, 9, 11]])
        uy = (spikes=uy, kin=uy)
    end
    return LinearDynamicalSystem(sm, om), ys, ux, uy
end

function _vi_posterior(model, y, ux, uy; inference=:laplace, compact=false, kwargs...)
    data = SSD.Data(model, y; ux, uy)
    tfs = SSD.initialize_FilterSmooth(model, data.tsteps; compact)
    pool = [
        SSD.SmoothWorkspace(
            Float64,
            model.latent_dim,
            SSD._ws_obs_dim(model),
            maximum(data.tsteps);
            ux_dim=model.ux_dim,
            uy_dim=SSD._ws_uy_dim(model),
        ),
    ]
    SSD.smooth!(
        model,
        tfs,
        data,
        pool;
        inference,
        max_iter=100,
        tol=1e-10,
        variational_max_iter=200,
        variational_tol=1e-7,
        kwargs...,
    )
    return tfs, pool, data
end

# Recover the dense covariance from Markov Gaussian marginal/lag blocks.
# This is independent of the production precision/inverse kernels.
function _vi_dense_covariance(P, lag)
    k, _, nt = size(P)
    S = zeros(k * nt, k * nt)
    for t in 1:nt
        it = ((t - 1) * k + 1):(t * k)
        S[it, it] .= P[:, :, t]
        t == 1 && continue
        ip = ((t - 2) * k + 1):((t - 1) * k)
        past = 1:((t - 1) * k)
        S[it, past] .= (lag[:, :, t] / P[:, :, t - 1]) * S[ip, past]
        S[past, it] .= S[it, past]'
    end
    return S
end

# Build J and h directly from transition/initial/observation equations.
function _vi_dense_gaussian_terms(model, y, u, v)
    sm = model.state_model
    k = model.latent_dim
    nt = size(u, 2)
    J = zeros(k * nt, k * nt)
    h = zeros(k * nt)
    J[1:k, 1:k] .= inv(sm.P0)
    h[1:k] .= sm.P0 \ sm.x0
    Qinv = inv(sm.Q)
    for t in 2:nt
        a = ((t - 2) * k + 1):((t - 1) * k)
        b = ((t - 1) * k + 1):(t * k)
        offset = sm.b + sm.B * u[:, t - 1]
        J[a, a] .+= sm.A' * Qinv * sm.A
        J[b, b] .+= Qinv
        J[b, a] .-= Qinv * sm.A
        J[a, b] .-= sm.A' * Qinv
        h[a] .-= sm.A' * Qinv * offset
        h[b] .+= Qinv * offset
    end
    for member in SSD._vi_members(model.obs_model, y, v)
        member.om isa GaussianObservationModel || continue
        om = member.om
        for t in 1:nt
            it = ((t - 1) * k + 1):(t * k)
            J[it, it] .+= om.C' * (om.R \ om.C)
            h[it] .+= om.C' * (om.R \ (member.y[:, t] - om.d - om.D * member.uy[:, t]))
        end
    end
    return J, h
end

function test_gaussian_variational_and_gauge()
    Random.seed!(181026)
    @testset "Gaussian variational inference and affine gauge" begin
        @testset "MAP trajectory preserved, including grouped/composite priors" begin
            for grouped in (false, true), composite in (false, true)
                model, y, ux, uy = _vi_fixture(; grouped, composite)
                original = deepcopy(model)
                baseline = fit!(
                    original, y; ux, uy, max_iter=5, progress=false, newton_tol=1e-10
                )
                canonical = deepcopy(model)
                normalized = fit!(
                    canonical,
                    y;
                    ux,
                    uy,
                    max_iter=5,
                    progress=false,
                    newton_tol=1e-10,
                    gauge_center=true,
                    gauge_whiten=true,
                )
                @test normalized ≈ baseline atol = 2e-6 rtol = 1e-9
                @test canonical.state_model.Q ≈ I atol = 1e-10
                g = canonical.state_model.gauge
                # Coordinate maps also agree on independently resmoothed means.
                xa, Pa = smooth(original, y; ux, uy, newton_tol=1e-10)
                xb, Pb = smooth(canonical, y; ux, uy, newton_tol=1e-10)
                for n in eachindex(xa)
                    @test xb[n] ≈ g.map * xa[n] .+ g.offset atol = 1e-5
                    @test Pb[n][:, :, 1] ≈ g.map * Pa[n][:, :, 1] * g.map' atol = 1e-6
                end
                @test elbo(canonical, y; ux, uy, newton_tol=1e-10) ≈ normalized[end] atol =
                    1e-6
            end
        end
        @testset "Centering plus Q prior, with whitening disabled" begin
            for method in (:laplace, :mean_corrected, :variational)
                model, y, ux, uy = _vi_fixture(; priors=false)
                model.state_model.Q_prior = IWPrior(; Ψ=0.2Matrix{Float64}(I, 2, 2), ν=7.0)
                original = deepcopy(model)
                canonical = deepcopy(model)
                a = fit!(
                    original,
                    y;
                    ux,
                    uy,
                    max_iter=5,
                    progress=false,
                    inference=method,
                    newton_tol=1e-9,
                    variational_tol=1e-7,
                )
                b = fit!(
                    canonical,
                    y;
                    ux,
                    uy,
                    max_iter=5,
                    progress=false,
                    inference=method,
                    newton_tol=1e-9,
                    variational_tol=1e-7,
                    gauge_center=true,
                    gauge_whiten=false,
                )
                @test a ≈ b atol = 1e-6
                @test canonical.state_model.Q ≈ original.state_model.Q atol = 1e-8
                @test norm(canonical.state_model.Q - I) > 0.1
                @test canonical.state_model.Q_prior.Ψ == 0.2I
                @test canonical.state_model.Q_prior.ν == 7.0
                @test canonical.state_model.gauge.map == I
                @test canonical.state_model.gauge.logprior_offset == 0
                @test elbo(
                    canonical,
                    y;
                    ux,
                    uy,
                    inference=method,
                    newton_tol=1e-9,
                    variational_tol=1e-7,
                ) ≈ b[end] atol = 1e-6
            end
            model, y, ux, uy = _vi_fixture(; priors=false)
            model.state_model.Q_prior = IWPrior(; Ψ=0.2Matrix{Float64}(I, 2, 2), ν=7.0)
            tfs, pool, data = _vi_posterior(model, y, ux, uy)
            before = deepcopy(model.state_model)
            c =
                reduce(+, vec(sum(fs.x_smooth; dims=2)) for fs in tfs.FilterSmooths) /
                sum(data.tsteps)
            SSD._affine_gauge!(model, [model], [tfs], nothing; center=true, whiten=false)
            @test norm(
                reduce(+, vec(sum(fs.x_smooth; dims=2)) for fs in tfs.FilterSmooths)
            ) < 1e-10
            @test model.state_model.x0 ≈ before.x0 - c atol = 1e-12
            @test model.state_model.Q == before.Q
            @test model.state_model.Q_prior.Ψ == before.Q_prior.Ψ
        end
        @testset "Grouped dynamics and reference prior constants" begin
            model, y, ux, uy = _vi_fixture(; priors=true)
            labels = ["a", "a", "b", "b"]
            model.state_model.depends_on = (;
                A=labels, b=labels, B=labels, Q=labels, x0=labels, P0=labels
            )
            for method in (:laplace, :mean_corrected, :variational)
                original = deepcopy(model)
                canonical = deepcopy(model)
                a = fit!(
                    original,
                    y;
                    ux,
                    uy,
                    max_iter=4,
                    progress=false,
                    inference=method,
                    newton_tol=1e-9,
                    variational_tol=1e-7,
                )
                b = fit!(
                    canonical,
                    y;
                    ux,
                    uy,
                    max_iter=4,
                    progress=false,
                    inference=method,
                    newton_tol=1e-9,
                    variational_tol=1e-7,
                    gauge_center=true,
                    gauge_whiten=true,
                )
                @test a ≈ b atol = 1e-5
                @test elbo(
                    canonical,
                    y;
                    ux,
                    uy,
                    inference=method,
                    newton_tol=1e-9,
                    variational_tol=1e-7,
                ) ≈ b[end] atol = 1e-6
            end
            model, y, ux, uy = _vi_fixture(; priors=false)
            model.state_model.depends_on = (; A=labels)
            before = copy(model.state_model.A)
            @test_throws ArgumentError fit!(model, y; ux, uy, max_iter=3, gauge_center=true)
            @test model.state_model.A == before
        end
        @testset "Reward A-only versus B/b-only ties survive whitening" begin
            reward = ["low", "low", "high", "high"]
            session = ["s1", "s2", "s1", "s2"]
            for effect in (:A, :Bb), method in (:laplace, :mean_corrected, :variational)
                model, y, ux, uy = _vi_fixture(; priors=true)
                model.state_model.depends_on =
                    effect == :A ? (; A=reward) : (; B=reward, b=reward)
                model.obs_model.depends_on = (; C=session, d=session, D=session)
                original = deepcopy(model)
                canonical = deepcopy(model)
                controls = (;
                    ux,
                    uy,
                    max_iter=4,
                    progress=false,
                    inference=method,
                    newton_tol=1e-9,
                    variational_tol=1e-7,
                )
                a = fit!(original, y; controls...)
                b = fit!(canonical, y; controls..., gauge_whiten=true)
                @test a ≈ b atol = 1e-5
                @test elbo(
                    canonical,
                    y;
                    ux,
                    uy,
                    inference=method,
                    newton_tol=1e-9,
                    variational_tol=1e-7,
                ) ≈ b[end] atol = 1e-6
                g = canonical.state_model.gauge
                @test all(iszero, g.offset)
                va, vb = canonical.state_model.variants
                @test va.Q === vb.Q
                @test va.Q ≈ I atol = 1e-10
                @test va.x0 === vb.x0
                @test va.P0 === vb.P0
                for field in (:A, :B, :b)
                    varies = effect == :A ? field == :A : field in (:B, :b)
                    @test (getproperty(va, field) === getproperty(vb, field)) == !varies
                end
                for (old, new) in
                    zip(original.state_model.variants, canonical.state_model.variants)
                    @test new.A ≈ g.map * old.A / g.map atol = 1e-7
                    @test new.b ≈ g.map * old.b atol = 1e-7
                    @test new.B ≈ g.map * old.B atol = 1e-7
                end
            end
        end
        @testset "An x0-centered translation breaks A-only intercept ties" begin
            A1 = [0.7 0.1; 0 0.4]
            A2 = [0.4 0; 0.1 0.6]
            b = [0.2, -0.1]
            x0 = [1.0, -0.5]
            b1 = b + (A1 - I) * x0
            b2 = b + (A2 - I) * x0
            @test b1 - b2 ≈ (A1 - A2) * x0
            @test norm(b1 - b2) > 0.1
            # Carrying a common origin anchor preserves the original shared b.
            @test b1 - (A1 - I) * x0 ≈ b
            @test b2 - (A2 - I) * x0 ≈ b
        end
        @testset "Large cancelling intercepts remain scoreable" begin
            make(c) = LinearDynamicalSystem(
                GaussianStateModel(; A=[0.8;;], b=[0.2c], Q=[0.1;;], x0=[c], P0=[0.3;;]),
                PoissonObservationModel(; C=[1.0;;], d=[-c]),
            )
            centered, shifted = make(0.0), make(1000.0)
            y = [reshape(Float64[0, 1, 2, 0, 1, 1], 1, :)]
            for method in (:laplace, :mean_corrected, :variational)
                controls = (; inference=method, newton_tol=1e-9, variational_tol=1e-7)
                @test elbo(centered, y; controls...) ≈ elbo(shifted, y; controls...) atol =
                    1e-7
            end
        end
        @testset "Mean-only vs full VI; compact, ragged and composite" begin
            for composite in (false, true)
                model, y, ux, uy = _vi_fixture(; composite, priors=false)
                reference = elbo(model, y; ux, uy, newton_tol=1e-10)
                scores = Float64[]
                for method in (:mean_corrected, :variational)
                    full, pool, data = _vi_posterior(model, y, ux, uy; inference=method)
                    packed, ppool, _ = _vi_posterior(
                        model, y, ux, uy; inference=method, compact=true
                    )
                    score = elbo(
                        model,
                        y;
                        ux,
                        uy,
                        inference=method,
                        variational_max_iter=200,
                        variational_tol=1e-7,
                        newton_tol=1e-10,
                    )
                    push!(scores, score)
                    @test score >= reference - 1e-8
                    @test sum(
                        trial_elbos(
                            model,
                            y;
                            ux,
                            uy,
                            inference=method,
                            variational_max_iter=200,
                            variational_tol=1e-7,
                            newton_tol=1e-10,
                        ),
                    ) ≈ score atol = 1e-7
                    laplace, _, _ = _vi_posterior(model, y, ux, uy)
                    for n in eachindex(full.FilterSmooths)
                        fs = full[n]
                        fp = packed[n]
                        @test fs.x_smooth ≈ fp.x_smooth atol = 1e-9
                        @test fs.p_smooth ≈
                            SSD.smoothed_covariances(fp, ppool[1].agg.p_smooth_shared) atol =
                            1e-9
                        @test SSD._lag_covariance_sum(fs) ≈ fp.xcov_sum atol = 1e-9
                        @test fs.entropy ≈ fp.entropy atol = 1e-9
                        method === :mean_corrected &&
                            @test fs.p_smooth == laplace[n].p_smooth
                        x = fs.x_smooth
                        k, nt = size(x)
                        yy = SSD._trial(data.y, n)
                        vv = SSD._trial(data.uy, n)
                        J, h = _vi_dense_gaussian_terms(model, yy, data.ux[n], vv)
                        S = _vi_dense_covariance(fs.p_smooth, fs.p_smooth_tt1)
                        grad = h - J * vec(x)
                        precision = copy(J)
                        for term in SSD._vi_members(model.obs_model, yy, vv)
                            term.om isa PoissonObservationModel || continue
                            om = term.om
                            for t in 1:nt
                                it = ((t - 1) * k + 1):(t * k)
                                rate =
                                    exp.(
                                        om.C * x[:, t] +
                                        om.d +
                                        om.D * term.uy[:, t] +
                                        diag(om.C * S[it, it] * om.C') / 2
                                    )
                                grad[it] .+= om.C' * (term.y[:, t] - rate)
                                precision[it, it] .+= om.C' * Diagonal(rate) * om.C
                            end
                        end
                        @test norm(grad) < 1e-4
                        if method === :variational
                            @test inv(S) ≈ precision atol = 1e-5 rtol = 1e-6
                            @test fs.entropy ≈
                                (length(grad) * (1 + log(2pi)) + logdet(Symmetric(S))) / 2 atol =
                                1e-8
                            @test all(diff(fs.inference_stats["accepted_elbos"]) .>= 0)
                            @test fs.inference_stats["converged"]
                        end
                    end
                end
                @test scores[2] >= scores[1] - 1e-8
            end
        end
        @testset "Exact Gaussian special case and VI EM ascent" begin
            model, y, ux, uy = _vi_fixture(; priors=false)
            om = GaussianObservationModel(;
                C=copy(model.obs_model.C),
                d=copy(model.obs_model.d),
                D=copy(model.obs_model.D),
                R=Matrix{Float64}(I, 4, 4),
            )
            exact = LinearDynamicalSystem(deepcopy(model.state_model), om)
            # Force the shared iterative route, matching the production Gaussian
            # neural composite. Its one-step Gaussian solution must remain exact.
            iterative = LinearDynamicalSystem(
                deepcopy(exact.state_model),
                CompositeObservationModel((gaussian=deepcopy(om),); quadratic=false),
            )
            for method in (:laplace, :mean_corrected, :variational)
                @test elbo(
                    iterative, (gaussian=y,); ux, uy=(gaussian=uy,), inference=method
                ) ≈ elbo(exact, y; ux, uy) atol = 1e-7
            end
            # Both exact Gaussian drivers retain their covariance-sharing path.
            for equal_length in (false, true)
                yy = equal_length ? [copy(y[1]) for _ in 1:4] : y
                uu = equal_length ? [copy(ux[1]) for _ in 1:4] : ux
                vv = equal_length ? [copy(uy[1]) for _ in 1:4] : uy
                a = fit!(deepcopy(exact), yy; ux=uu, uy=vv, max_iter=4, progress=false)
                b = fit!(
                    deepcopy(exact),
                    yy;
                    ux=uu,
                    uy=vv,
                    max_iter=4,
                    progress=false,
                    gauge_center=true,
                    gauge_whiten=true,
                )
                @test a ≈ b atol = 1e-7
            end
            trace = fit!(
                model,
                y;
                ux,
                uy,
                max_iter=10,
                progress=false,
                inference=:variational,
                gauge_center=true,
                gauge_whiten=true,
                variational_max_iter=200,
                variational_tol=1e-7,
            )
            @test minimum(diff(trace)) >= -1e-7
        end
        @testset "Unsafe whitening is skipped; incompatible constraints fail early" begin
            model, y, ux, uy = _vi_fixture(; priors=false)
            model.state_model.Q .= Diagonal([1e-12, 1.0])
            originalQ = copy(model.state_model.Q)
            fs, pool, data = _vi_posterior(model, y, ux, uy)
            @test_logs (:warn, r"Skipping unsafe") SSD._affine_gauge!(
                model, [model], [fs], nothing; center=true, whiten=true, max_condition=1e8
            )
            @test model.state_model.Q == originalQ
            @test model.state_model.gauge.whitening_skips == 1
            @test norm(reduce(+, vec(sum(f.x_smooth; dims=2)) for f in fs.FilterSmooths)) <
                1e-8
            @test all(isnan, fs[1].p_smooth) # stale posterior caches cannot be mistaken for valid ones
            model, y, ux, uy = _vi_fixture(; priors=false)
            model.obs_model.d_fixed_zero = true
            fill!(model.obs_model.d, 0)
            @test_throws ArgumentError fit!(model, y; ux, uy, gauge_center=true, max_iter=2)
            @test all(iszero, model.obs_model.d)
            @test_throws ArgumentError elbo(model, y; ux, uy, inference=:unknown)
            @test_throws ArgumentError elbo(
                model, y; ux, uy, inference=:variational, variational_max_iter=0
            )
        end
        @testset "Trial-specific initial means and constrained mean updates" begin
            model, y, ux, uy = _vi_fixture(; priors=false)
            # Nonzero B0 gets its own trial model, including the correct x0.
            model.state_model.B0 = [1.0 0.2; -0.3 0.4]
            initial = [1.0 1.0 1.0 1.0; -0.5 0.0 0.5 1.0]
            for method in (:mean_corrected, :variational)
                score = elbo(model, y; ux, uy, ux0=initial, inference=method)
                @test isfinite(score)
                @test score >= elbo(model, y; ux, uy, ux0=initial) - 1e-7
            end
            @test_throws ArgumentError fit!(
                model, y; ux, uy, ux0=initial, gauge_center=true
            )
            @test all(
                isfinite,
                fit!(
                    model,
                    y;
                    ux,
                    uy,
                    ux0=initial,
                    max_iter=3,
                    progress=false,
                    gauge_whiten=true,
                ),
            )
        end
        @testset "Early-stop snapshot includes gauge and transformed priors" begin
            model, y, ux, uy = _vi_fixture(; grouped=true, composite=true)
            trace = fit!(
                model,
                y;
                ux,
                uy,
                y_test=y,
                ux_test=ux,
                uy_test=uy,
                max_iter=5,
                progress=false,
                gauge_center=true,
                gauge_whiten=true,
                early_stopping=true,
                min_delta=1e9,
                patience=2,
                test_every=1,
            )
            @test trace.stopped_early
            @test trace.returned_iter == 1
            @test model.state_model.gauge === nothing
            @test elbo(model, y; ux, uy) ≈ trace[1] atol = 1e-7
        end
        @testset "Warm-started VI reaches the cold optimum; stopping is coordinate free" begin
            controls = (;
                inference=:variational,
                max_iter=100,
                tol=1e-10,
                variational_max_iter=200,
                variational_tol=1e-7,
            )
            for composite in (false, true)
                model, y, ux, uy = _vi_fixture(; composite, priors=false)
                warm, pool, data = _vi_posterior(model, y, ux, uy; inference=:variational)
                @test all(!fs.inference_stats["warm_start"] for fs in warm.FilterSmooths)
                # New parameters, as after an M-step: the stored posterior is a
                # start, not the answer.
                moved = deepcopy(model)
                spikes = composite ? SSD._models(moved.obs_model).spikes : moved.obs_model
                spikes.C .*= 1.2
                spikes.d .+= 0.15
                moved.state_model.A .*= 0.9
                SSD.smooth!(moved, warm, data, pool; controls...)
                fresh, _, _ = _vi_posterior(moved, y, ux, uy; inference=:variational)
                for n in eachindex(warm.FilterSmooths)
                    stats = warm[n].inference_stats
                    @test stats["warm_start"]
                    @test stats["converged"]
                    @test all(diff(stats["accepted_elbos"]) .>= 0)
                    @test length(warm[n].vi_sites) == 1
                    @test !fresh[n].inference_stats["warm_start"]
                    @test warm[n].x_smooth ≈ fresh[n].x_smooth atol = 1e-6
                    @test warm[n].p_smooth ≈ fresh[n].p_smooth atol = 1e-6
                    @test warm[n].p_smooth_tt1[:, :, 2:end] ≈
                        fresh[n].p_smooth_tt1[:, :, 2:end] atol = 1e-6
                    @test warm[n].entropy ≈ fresh[n].entropy atol = 1e-6
                end
            end

            # Sites are rates, so a gauge-normalized warm start is the exact
            # transform of the plain one.
            model, y, ux, uy = _vi_fixture(; priors=false)
            plain, pool, data = _vi_posterior(model, y, ux, uy; inference=:variational)
            gauged_model = deepcopy(model)
            gauged = deepcopy(plain)
            SSD._affine_gauge!(
                gauged_model, [gauged_model], [gauged], nothing; center=true, whiten=true
            )
            g = gauged_model.state_model.gauge
            SSD.smooth!(model, plain, data, pool; controls...)
            SSD.smooth!(gauged_model, gauged, data, pool; controls...)
            for n in eachindex(plain.FilterSmooths)
                @test gauged[n].inference_stats["warm_start"]
                @test gauged[n].x_smooth ≈ g.map * plain[n].x_smooth .+ g.offset atol = 1e-6
                @test gauged[n].p_smooth[:, :, 1] ≈
                    g.map * plain[n].p_smooth[:, :, 1] * g.map' atol = 1e-6
            end

            # The precision mismatch is invariant under a linear latent map.
            S = [2.0 0.3; 0.3 1.0]
            Δ = [0.5 -0.2; -0.2 0.1]
            T = [1.5 0.2; -0.4 0.7]
            vb = SSD.VIBuffers(Float64, 2, 1)
            a = SSD._vi_precision_mismatch(reshape(S, 2, 2, 1), [zeros(2, 2)], [Δ], vb, 1)
            b = SSD._vi_precision_mismatch(
                reshape(T * S * T', 2, 2, 1), [zeros(2, 2)], [(T' \ Δ) / T], vb, 1
            )
            @test a ≈ tr(S * Δ * S * Δ) rtol = 1e-12
            @test b ≈ a rtol = 1e-10
        end
    end
end
