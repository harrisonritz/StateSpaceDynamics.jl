#=============================================================================
Causal (closed-loop) inverse LQR: `LQRStateModel` in `:causal` mode.

Every check is against something that does not share the implementation under
test:

* the horizon caches against `lqr_riccati_sequence` (the sweep `simulate_lqr`
  uses), with schedules, offsets, terminal costs and gated references;
* the exact log-likelihood against a dense Gaussian over the whole path, built
  here from the generative equations of the causal controller;
* `rand` against the same dense moments, by Monte Carlo;
* the structural gradient against central differences of the packed objective,
  over every option the mode has;
* the noise update against its closed form, and the M-step objective against
  the ELBO's own transition term;
* EM against data the model generates: monotone, invariant where it should be,
  and recovering the controller.
=============================================================================#

const CAUSAL_A = [1.0 0.1; -0.05 0.9]
const CAUSAL_S = [0.06 0.01; 0.01 0.04]

"""
A causal model with every option settable, and a Gaussian LDS around it.
`nreg = 3` gives a delay epoch, a reach epoch and a terminal cost.
"""
function causal_fixture(
    rng;
    nreg::Int=1,
    tmax::Int=12,
    m::Int=0,
    p::Int=5,
    terminal_cost::Bool=true,
    slack::Bool=true,
    plant_noise::Symbol=:dense,
    costate_noise::Symbol=:dense,
    prior::Bool=false,
    fixed::Union{Nothing,Float64}=nothing,
    gate::Union{Nothing,Matrix{Bool}}=nothing,
    observe_costate::Bool=false,
    affine::Bool=true,
)
    n = 2
    d = 2n
    Qs = [[1.0 0.2; 0.2 0.3], [2.0 -0.1; -0.1 0.5], [3.0 0.0; 0.0 1.0]][1:nreg]
    schedule = if nreg == 1
        Int[]
    elseif nreg == 2
        vcat(fill(1, 4), fill(2, tmax - 4))
    else
        vcat(fill(1, 4), fill(2, tmax - 5), [3])
    end
    Σ = plant_noise === :dense ? [0.02 0.005; 0.005 0.01] : Matrix(Diagonal([0.02, 0.01]))
    Ω = if fixed !== nothing
        Matrix(fixed * I, n, n)
    elseif costate_noise === :dense
        [0.5 0.1; 0.1 0.3]
    else
        Matrix(Diagonal([0.5, 0.3]))
    end
    kw = (;
        schedule=schedule,
        terminal_cost=terminal_cost,
        slack_drives_state=slack,
        plant_noise=plant_noise,
        costate_noise=costate_noise,
        h=affine ? 0.05 .* randn(rng, d) : nothing,
        hf=affine ? 0.1 .* randn(rng, n) : nothing,
        x0=[1.0, 0.0, 0.0, 0.0],
        P0=Matrix(0.1I, d, d),
        observe_costate=observe_costate,
    )
    if m > 0
        kw = merge(
            kw, (; Bu=0.1 .* randn(rng, d, m), Gref=randn(rng, n, m), gref_gate=gate)
        )
    end
    if prior
        kw = merge(
            kw,
            (;
                Σ_prior=IWPrior(; Ψ=Matrix(0.05I, d, d), ν=8.0),
                Qc_prior=IWPrior(; Ψ=Matrix(1.0I, n, n), ν=4.0),
            ),
        )
    end
    fixed === nothing || (kw = merge(kw, (; fixed_costate_sigma=fixed)))
    sm = causal_state_model(
        copy(CAUSAL_A), copy(CAUSAL_S), nreg == 1 ? Qs[1] : Qs, Σ, Ω; kw...
    )
    C = randn(rng, p, d)
    observe_costate || (C[:, (n + 1):d] .= 0)
    lds = LinearDynamicalSystem(
        sm, GaussianObservationModel(C, Matrix(0.05I, p, p), 0.1 .* randn(rng, p))
    )
    return sm, lds
end

"""Constant per-trial inputs, as the causal mode requires."""
causal_inputs(rng, m, lengths) = [repeat(randn(rng, m), 1, T) for T in lengths]

"""
The joint mean and covariance of `z_{1:T}` for one trial, from the causal
controller's generative equations and `lqr_riccati_sequence` — nothing from the
horizon cache.
"""
function causal_dense_moments(sm, T::Int, u::AbstractVector, offset::Int)
    n = plant_dim(sm)
    d = 2n
    m = length(u)
    ux = m > 0 ? repeat(u, 1, T) : nothing
    smo = StateSpaceDynamics._with_cost_offset(sm, offset)
    P, g, W = lqr_riccati_sequence(smo, T; ux=ux)
    A, S = sm.A, sm.S
    c = sm.h[1:n] .+ (m > 0 ? sm.Bu[1:n, :] * u : zeros(n))
    Σ = sm.Σ[1:n, 1:n]
    Ω = sm.Σ[(n + 1):d, (n + 1):d]
    slack = sm.causal.slack_drives_state
    F = Vector{Matrix{Float64}}(undef, T - 1)
    b = Vector{Vector{Float64}}(undef, T - 1)
    Qn = Vector{Matrix{Float64}}(undef, T - 1)
    for t in 1:(T - 1)
        s = t + 1
        Φ = W[s] * A
        bx = W[s] * (c .- S * g[s])
        F[t] = [Φ zeros(n, n); P[s]*Φ zeros(n, n)]
        b[t] = vcat(bx, P[s] * bx .+ g[s])
        # x' = Φx + bx + ε − [slack] W S ν,  λ' = P x' + g + ν
        Lx = if slack
            hcat(Matrix(1.0I, n, n), -W[s] * S)
        else
            hcat(Matrix(1.0I, n, n), zeros(n, n))
        end
        L = vcat(Lx, P[s] * Lx .+ hcat(zeros(n, n), Matrix(1.0I, n, n)))
        Qn[t] = L * [Σ zeros(n, n); zeros(n, n) Ω] * L'
    end
    μ = zeros(d * T)
    V = zeros(d * T, d * T)
    blk(t) = ((t - 1) * d + 1):(t * d)
    μt = [copy(sm.x0)]
    Vt = [Matrix(sm.P0)]
    for t in 2:T
        push!(μt, F[t - 1] * μt[end] .+ b[t - 1])
        push!(Vt, F[t - 1] * Vt[end] * F[t - 1]' .+ Qn[t - 1])
    end
    for t in 1:T
        μ[blk(t)] = μt[t]
        Φts = Matrix(1.0I, d, d)
        for s in t:T
            s > t && (Φts = F[s - 1] * Φts)
            V[blk(s), blk(t)] = Φts * Vt[t]
            V[blk(t), blk(s)] = (Φts * Vt[t])'
        end
    end
    return μ, V
end

"""`log p(y)` of one trial under the dense moments."""
function causal_dense_loglik(lds, y::AbstractMatrix, u::AbstractVector, offset::Int)
    sm = lds.state_model
    T = size(y, 2)
    μ, V = causal_dense_moments(sm, T, u, offset)
    om = lds.obs_model
    H = kron(Matrix(1.0I, T, T), om.C)
    my = H * μ .+ repeat(om.d, T)
    Cy = Symmetric(H * V * H' + kron(Matrix(1.0I, T, T), om.R))
    r = vec(y) .- my
    F = cholesky(Cy)
    return -0.5 * (logdet(F) + dot(r, F \ r) + length(r) * log(2π))
end

# ----------------------------------------------------------------------------

function test_causal_construction()
    rng = StableRNG(1)
    sm, lds = causal_fixture(rng; nreg=3)
    @test sm.mode === :causal
    @test !sm.terminal && !sm.condition_terminal
    @test sm.causal == CausalOptions(; terminal_cost=true)
    @test plant_dim(sm) == 2 && lds.latent_dim == 4
    @test iszero(sm.Σ[1:2, 3:4]) && iszero(sm.Σ[3:4, 1:2])
    @test lqr_parameters(sm).terminal

    # The default-shaped constructor, and a singular plant (only applied forward).
    dflt = LQRStateModel(4; mode=:causal)
    @test dflt.mode === :causal && plant_dim(dflt) == 2
    sing = causal_state_model(
        [1.0 0.0; 0.0 0.0],
        Matrix(0.1I, 2, 2),
        Matrix(1.0I, 2, 2),
        Matrix(0.1I, 2, 2),
        Matrix(0.1I, 2, 2),
    )
    @test sing.mode === :causal

    # Display names the mode and its noise.
    shown = sprint(show, sm)
    @test occursin("Causal LQR State Model", shown)
    @test occursin("slack drives state", shown)
    @test !occursin("symplectic defect", shown)

    I2 = Matrix(1.0I, 2, 2)
    @test_throws ArgumentError CausalOptions(; plant_noise=:full)
    @test_throws ArgumentError causal_state_model(
        CAUSAL_A, CAUSAL_S, I2, [0.1 0.01; 0.01 0.1], I2; plant_noise=:diagonal
    )
    @test_throws ArgumentError causal_state_model(
        CAUSAL_A, CAUSAL_S, I2, I2, [0.1 0.01; 0.01 0.1]; costate_noise=:diagonal
    )
    @test_throws ArgumentError causal_state_model(CAUSAL_A, CAUSAL_S, I2, -I2, I2)
    @test_throws StateSpaceDynamics.NotSymmetricError causal_state_model(
        CAUSAL_A, CAUSAL_S, I2, [1.0 0.5; 0.0 1.0], I2
    )
    @test_throws StateSpaceDynamics.DimensionMismatchError causal_state_model(
        CAUSAL_A, CAUSAL_S, I2, Matrix(1.0I, 3, 3), I2
    )
    @test_throws ArgumentError causal_state_model(
        CAUSAL_A, CAUSAL_S, [I2, 2I2], I2, I2; schedule=[1, 1, 2], terminal_regime=2
    )
    @test_throws ArgumentError causal_state_model(
        CAUSAL_A,
        CAUSAL_S,
        [I2, 2I2],
        I2,
        I2;
        schedule=[1, 1, 2],
        terminal_cost=true,
        terminal_regime=3,
    )
    @test_throws ArgumentError causal_state_model(
        CAUSAL_A, CAUSAL_S, I2, I2, I2; fixed_costate_sigma=0.5
    )

    #= A `Σ_prior` acts through its block marginals, IW(Ψ_bb, ν − (2n − q)),
    which are proper (positive pseudo-counts) only for ν > 2n − 1. =#
    weak = IWPrior(; Ψ=Matrix(0.05I, 4, 4), ν=3.0)
    @test_throws ArgumentError causal_state_model(
        CAUSAL_A, CAUSAL_S, I2, I2, I2; Σ_prior=weak
    )
    @test causal_state_model(
        CAUSAL_A, CAUSAL_S, I2, I2, I2; Σ_prior=IWPrior(; Ψ=Matrix(0.05I, 4, 4), ν=3.5)
    ).mode === :causal

    # Symplectic-only readouts refuse; the steady state is still available.
    @test_throws ArgumentError symplectic_defect(sm)
    @test_throws ArgumentError symplectic_matrix(sm)
    @test size(closed_loop_dynamics(sm; k=2)) == (2, 2)
    # No schedule boundaries yet; a reference gate is fine.
    @test_throws ArgumentError set_schedule_boundaries!(sm; entries=[5])
    smg, _ = causal_fixture(StableRNG(2); nreg=2, m=2)
    set_gref_gate!(smg, Bool[1 0; 0 1])
    @test smg.gref_gate == Bool[1 0; 0 1]
    return nothing
end

function test_causal_horizons()
    rng = StableRNG(3)
    for (nreg, term, m, gate) in (
        (1, true, 0, nothing),
        (1, false, 2, nothing),
        (3, true, 2, nothing),
        (2, true, 2, Bool[1 0; 0 1]),
    )
        sm, _ = causal_fixture(rng; nreg=nreg, terminal_cost=term, m=m, gate=gate, tmax=14)
        offsets = nreg == 1 ? [0] : [0, 2]
        for off in offsets, T in (6, 12)
            StateSpaceDynamics._register_causal_horizons!(sm, [T], [off])
            refresh!(sm)
            H = StateSpaceDynamics._causal_horizon(
                StateSpaceDynamics._with_cost_offset(sm, off), T
            )
            u = randn(rng, m)
            P, g, W = lqr_riccati_sequence(
                StateSpaceDynamics._with_cost_offset(sm, off),
                T;
                ux=m > 0 ? repeat(u, 1, T) : nothing,
            )
            ũ = vcat(1.0, u)
            @test maximum(norm(P[t] - H.P[t]) for t in 1:T) < 1e-10
            @test maximum(norm(W[t] - H.W[t]) for t in 1:T) < 1e-10
            @test maximum(norm(g[t] - H.G[t] * ũ) for t in 1:T) < 1e-10
            for t in 1:(T - 1)
                # Q_t⁻¹ from the template is the inverse of L B Lᵀ, Rz = L⁻¹.
                L = inv(H.Rz[t])
                Qt = L * sm.Σ * L'
                @test norm(-H.negQinv[t] * Qt - I) < 1e-9
                @test det(H.Rz[t]) ≈ 1 atol = 1e-12
            end
        end
    end
    # Without a schedule the offset is invisible: one horizon per length.
    sm, _ = causal_fixture(StableRNG(4))
    StateSpaceDynamics._register_causal_horizons!(sm, [8, 8, 9], [0, 3, 5])
    @test Set(sm.cache.causal_keys) ⊇ Set([(0, 8), (0, 9)])
    @test !((3, 8) in sm.cache.causal_keys)
    # An unregistered horizon is reported, not silently invented.
    @test_throws ArgumentError StateSpaceDynamics._causal_horizon(sm, 31)
    return nothing
end

function test_causal_exact_loglikelihood()
    for (slack, nreg, m, observe) in (
        (true, 1, 0, false),
        (false, 1, 0, false),
        (true, 3, 2, false),
        (false, 2, 2, false),
        (true, 1, 1, true),
    )
        rng = StableRNG(10 + nreg + m)
        sm, lds = causal_fixture(
            rng; slack=slack, nreg=nreg, m=m, tmax=10, observe_costate=observe
        )
        lengths = nreg == 3 ? [10, 10, 10] : [10, 8, 9]
        ux = m > 0 ? causal_inputs(rng, m, lengths) : nothing
        _, y = rand(rng, lds, lengths; ux=ux)
        ll = loglikelihood(lds, y; ux=ux)
        dense = sum(
            causal_dense_loglik(lds, y[i], m > 0 ? ux[i][:, 1] : Float64[], 0) for
            i in eachindex(y)
        )
        @test ll ≈ dense rtol = 1e-9
        # With no parameter priors the ELBO at the exact posterior is the likelihood.
        @test elbo(lds, y; ux=ux) ≈ ll rtol = 1e-9
    end

    # Per-trial cost offsets (scored through the per-trial smoother).
    rng = StableRNG(20)
    sm, lds = causal_fixture(rng; nreg=2, m=1, tmax=16)
    offs = [0, 3, 5]
    ux = causal_inputs(rng, 1, [10, 10, 10])
    _, y = rand(rng, lds, [10, 10, 10]; ux=ux, cost_offset=offs)
    ll = loglikelihood(lds, y; ux=ux, cost_offset=offs)
    dense = sum(causal_dense_loglik(lds, y[i], ux[i][:, 1], offs[i]) for i in 1:3)
    @test ll ≈ dense rtol = 1e-9

    #= A readout slack never reaches the state, so an emission that cannot see
    the costate cannot see Ω either. =#
    rng = StableRNG(21)
    sm, lds = causal_fixture(rng; slack=false)
    _, y = rand(rng, lds, [12, 12])
    l1 = loglikelihood(lds, y)
    sm.Σ[3:4, 3:4] .*= 7
    refresh!(sm)
    @test loglikelihood(lds, y) ≈ l1 rtol = 1e-10
    return nothing
end

function test_causal_sampling()
    rng = StableRNG(30)
    sm, lds = causal_fixture(rng; m=1, tmax=8)
    T = 8
    u = [0.7]
    μ, V = causal_dense_moments(sm, T, u, 0)
    N = 6000
    z, _ = rand(rng, lds, fill(T, N); ux=[repeat(u, 1, T) for _ in 1:N])
    Z = reduce(hcat, (vec(zi) for zi in z))
    se = sqrt.(diag(V) ./ N)
    @test maximum(abs.(vec(mean(Z; dims=2)) .- μ) ./ se) < 5
    Vemp = cov(Z; dims=2)
    @test norm(Vemp - V) / norm(V) < 0.06

    # The chain is the stable closed loop: a long draw stays bounded, with no
    # divergence warning.
    sml, ldsl = causal_fixture(StableRNG(31); tmax=400)
    zl, _ = @test_logs rand(StableRNG(32), ldsl, 400)
    @test all(isfinite, zl)
    @test maximum(abs, zl[1:2, :]) < 50
    # A time-varying input is refused.
    @test_throws ArgumentError rand(rng, lds, T; ux=randn(rng, 1, T))
    return nothing
end

"""The packed M-step objective and gradient of a fixture, at a perturbed point."""
function causal_mstep_context(; profile::Bool=true, kw...)
    rng = StableRNG(40)
    sm, lds = causal_fixture(rng; kw...)
    m = size(sm.Bu, 2)
    lengths = fill(12, 16)
    ux = m > 0 ? causal_inputs(rng, m, lengths) : nothing
    _, y = rand(rng, lds, lengths; ux=ux)
    data = StateSpaceDynamics.Data(lds, y; ux=ux)
    StateSpaceDynamics._prepare_lqr!(lds, data)
    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, data.tsteps)
    pool = StateSpaceDynamics._lqr_sws_pool(lds, data)
    hs = StateSpaceDynamics._initialize_td_sufficient_statistics(Float64, lds, data.tsteps)
    StateSpaceDynamics._td_init_const_blocks!(pool[1], lds, data)
    StateSpaceDynamics.estep!(lds, hs, tfs, data, pool)
    ctx = StateSpaceDynamics._LQRMStepCtx(hs, sm, profile)
    return ctx, hs, sm, lds, pool
end

function test_causal_mstep_gradient()
    configs = (
        (;),
        (; slack=false),
        (; plant_noise=:diagonal, costate_noise=:diagonal),
        (; terminal_cost=false),
        (; m=2),
        (; m=2, slack=false),
        (; nreg=3),
        (; nreg=2, m=2, gate=Bool[1 0; 0 1]),
        (; prior=true),
        (; prior=true, plant_noise=:diagonal),
        (; fixed=0.4),
        (; nreg=3, m=2, prior=true, slack=false, costate_noise=:diagonal),
    )
    for cfg in configs, profile in (true, false)
        ctx, = causal_mstep_context(; profile=profile, cfg...)
        np = StateSpaceDynamics._lqr_nparams(ctx)
        θ = zeros(np)
        StateSpaceDynamics._lqr_pack!(θ, ctx)
        θ .+= 0.01 .* randn(StableRNG(41), np)
        g = similar(θ)
        f0 = StateSpaceDynamics._lqr_fg!(g, θ, ctx)
        @test isfinite(f0)
        @test StateSpaceDynamics._lqr_fg!(nothing, θ, ctx) == f0
        h = 1e-6
        gd = map(1:np) do i
            e = zeros(np)
            e[i] = h
            return (
                StateSpaceDynamics._lqr_fg!(nothing, θ .+ e, ctx) -
                StateSpaceDynamics._lqr_fg!(nothing, θ .- e, ctx)
            ) / 2h
        end
        @test norm(g - gd) / norm(gd) < 1e-6
    end
    # An infeasible trial point is reported as such, not thrown.
    ctx, = causal_mstep_context()
    θ = zeros(StateSpaceDynamics._lqr_nparams(ctx))
    StateSpaceDynamics._lqr_pack!(θ, ctx)
    θ[5:7] .= 800.0                         # S's log-diagonal: an overflowing sweep
    @test StateSpaceDynamics._lqr_fg!(similar(θ), θ, ctx) == Inf
    return nothing
end

function test_causal_noise_update()
    # Dense, no prior: R/N per block; the cross blocks stay zero.
    for (pn, cn, prior, fixed) in (
        (:dense, :dense, false, nothing),
        (:diagonal, :diagonal, false, nothing),
        (:dense, :diagonal, true, nothing),
        (:dense, :dense, false, 0.3),
    )
        ctx, hs, sm, lds, pool = causal_mstep_context(;
            plant_noise=pn, costate_noise=cn, prior=prior, fixed=fixed
        )
        StateSpaceDynamics._lqr_structure_mstep!(ctx, false, 1)
        R = copy(ctx.R[1])
        N = ctx.N_q[1]
        StateSpaceDynamics._lqr_noise_mstep!(ctx)
        n = 2
        @test iszero(sm.Σ[1:n, (n + 1):(2n)])
        for (r, form) in ((1:n, pn), ((n + 1):(2n), cn))
            if r == (n + 1):(2n) && fixed !== nothing
                @test sm.Σ[r, r] ≈ fixed * I
                continue
            end
            Ψ = prior ? Matrix(0.05I, 2n, 2n)[r, r] : zeros(n, n)
            q = form === :dense ? n : 1
            Neff = prior ? (8.0 - (2n - q)) + N + q + 1 : N
            expect = (R[r, r] .+ Ψ) ./ Neff
            form === :diagonal && (expect = Diagonal(diag(expect)))
            @test sm.Σ[r, r] ≈ expect rtol = 1e-12
        end
    end

    #=
    The M-step objective and the ELBO's transition term come from separate code
    (the optimizer's sweep at a packed point, the cache's sweep at the model's
    parameters). At the noise maximizer, no prior, they are tied exactly:
    Q_trans = −f − ½Nd log 2π + ½N(Σ_b d_b log N) − ½Nd.
    =#
    for (pn, cn) in ((:dense, :dense), (:diagonal, :diagonal))
        ctx, hs, sm, lds, pool = causal_mstep_context(; plant_noise=pn, costate_noise=cn)
        StateSpaceDynamics._lqr_structure_mstep!(ctx, false, 1)
        StateSpaceDynamics._lqr_noise_mstep!(ctx)
        refresh!(sm)
        θ = zeros(StateSpaceDynamics._lqr_nparams(ctx))
        StateSpaceDynamics._lqr_pack!(θ, ctx)
        f = StateSpaceDynamics._lqr_fg!(nothing, θ, ctx)
        N = ctx.N_q[1]
        d = 4
        Qt = StateSpaceDynamics._causal_Q_transition(sm, hs)
        @test Qt ≈ -f - 0.5N * d * log(2π) + 0.5N * d * log(N) - 0.5N * d rtol = 1e-10
    end
    return nothing
end

function test_causal_em()
    mono(e) = minimum(diff(collect(e)))
    for cfg in (
        (; nreg=3, tmax=14),
        (; slack=false),
        (; plant_noise=:diagonal, costate_noise=:diagonal),
        (; fixed=0.4),
        (; terminal_cost=false),
        (; prior=true),
    )
        rng = StableRNG(50)
        sm, lds = causal_fixture(rng; cfg...)
        lengths = haskey(cfg, :nreg) ? [14, 11, 14, 12, 14, 13] : fill(12, 6)
        _, y = rand(rng, lds, lengths)
        sm.A .+= 0.03
        refresh!(sm)
        els = fit!(lds, y; max_iter=6, progress=false)
        @test mono(els) > -1e-7
        @test iszero(sm.Σ[1:2, 3:4])
    end

    # Inputs with a gate, constant per trial; a time-varying input is refused.
    rng = StableRNG(51)
    sm, lds = causal_fixture(rng; nreg=2, m=2, gate=Bool[1 0; 0 1])
    ux = causal_inputs(rng, 2, fill(12, 6))
    _, y = rand(rng, lds, fill(12, 6); ux=ux)
    @test mono(fit!(lds, y; ux=ux, max_iter=6, progress=false)) > -1e-7
    @test_throws ArgumentError fit!(
        lds, y; ux=[randn(rng, 2, 12) for _ in 1:6], max_iter=2, progress=false
    )
    # The last column is never read by a transition, so it may differ.
    ux_last = [hcat(u[:, 1:(end - 1)], randn(rng, 2)) for u in ux]
    @test loglikelihood(lds, y; ux=ux_last) ≈ loglikelihood(lds, y; ux=ux) rtol = 1e-12

    # Poisson emission with per-trial cost offsets.
    rng = StableRNG(52)
    sm, lds = causal_fixture(rng; nreg=2, tmax=18)
    plds = LinearDynamicalSystem(
        sm, PoissonObservationModel(0.5 .* lds.obs_model.C, fill(0.5, 5))
    )
    offs = [0, 3, 5, 2, 0, 4]
    _, y = rand(rng, plds, fill(12, 6); cost_offset=offs)
    els = fit!(plds, y; cost_offset=offs, max_iter=6, progress=false)
    @test mono(els) > -1e-6

    # Initial-state inputs: each trial its own x0 = B0 u0.
    rng = StableRNG(53)
    sm, lds = causal_fixture(rng)
    sm.B0 = 0.3 .* randn(rng, 4, 2)
    refresh!(sm)
    u0 = randn(rng, 2, 6)
    _, y = rand(rng, lds, fill(12, 6); ux0=u0)
    @test mono(fit!(lds, y; ux0=u0, max_iter=5, progress=false)) > -1e-7

    # Held-out scoring through the same horizons.
    rng = StableRNG(54)
    sm, lds = causal_fixture(rng)
    _, y = rand(rng, lds, fill(12, 8))
    _, yt = rand(rng, lds, fill(10, 3))
    tr_ = fit!(lds, y; y_test=yt, max_iter=4, progress=false)
    @test length(tr_.test) == 4 && all(isfinite, tr_.test)
    return nothing
end

function test_causal_depends_on()
    mono(e) = minimum(diff(collect(e)))
    session = repeat([1, 2]; inner=4)
    one_group = fill(1, 8)
    # One group reproduces the ungrouped fit exactly.
    smA, ldsA = causal_fixture(StableRNG(60))
    _, y = rand(StableRNG(61), ldsA, fill(12, 8))
    smB, ldsB = causal_fixture(StableRNG(60))
    set_depends_on!(smA, (structure=one_group, noise=one_group))
    ldsA2 = LinearDynamicalSystem(smA, ldsA.obs_model)
    eA = fit!(ldsA2, y; max_iter=5, progress=false)
    eB = fit!(ldsB, y; max_iter=5, progress=false)
    @test maximum(abs, collect(eA) .- collect(eB)) < 1e-8

    for dep in ((Qc=session,), (structure=session,), (noise=session,))
        sm, lds = causal_fixture(StableRNG(62); nreg=2)
        _, y = rand(StableRNG(63), lds, fill(12, 8))
        set_depends_on!(sm, dep)
        lds2 = LinearDynamicalSystem(sm, lds.obs_model)
        els = fit!(lds2, y; max_iter=6, progress=false)
        @test mono(els) > -1e-7
        @test all(isfinite, els)
        if haskey(dep, :Qc)
            p1 = group_parameter(sm, :structure, 1)
            p2 = group_parameter(sm, :structure, 2)
            @test p1.A === p2.A && !(p1.Qc[1] ≈ p2.Qc[1])
        end
    end

    # Rescaling the costate of a grouped model leaves its score alone.
    sm, lds = causal_fixture(StableRNG(64); m=1)
    ux = causal_inputs(StableRNG(65), 1, fill(12, 8))
    _, y = rand(StableRNG(66), lds, fill(12, 8); ux=ux)
    set_depends_on!(sm, (Qc=session,))
    lds2 = LinearDynamicalSystem(sm, lds.obs_model)
    fit!(lds2, y; ux=ux, max_iter=3, progress=false)
    before = elbo(lds2, y; ux=ux)
    rescale_costate!(sm, 2.5)
    @test elbo(lds2, y; ux=ux) ≈ before rtol = 1e-9
    return nothing
end

function test_causal_invariances()
    rng = StableRNG(70)
    sm, lds = causal_fixture(rng; m=1)
    ux = causal_inputs(rng, 1, fill(12, 5))
    _, y = rand(rng, lds, fill(12, 5); ux=ux)
    l0 = loglikelihood(lds, y; ux=ux)
    rescale_costate!(sm, 2.5)
    @test loglikelihood(lds, y; ux=ux) ≈ l0 rtol = 1e-11
    rescale_costate!(sm; target=:trace)
    @test loglikelihood(lds, y; ux=ux) ≈ l0 rtol = 1e-11
    @test sum(trial_elbos(lds, y; ux=ux)) ≈ elbo(lds, y; ux=ux) rtol = 1e-11

    # A prior moves the bound by exactly its own log-density.
    smp, ldsp = causal_fixture(StableRNG(71); prior=true)
    _, yp = rand(StableRNG(72), ldsp, fill(12, 4))
    lp = StateSpaceDynamics._lqr_structural_logprior(smp)
    @test isfinite(lp) && lp != 0
    @test elbo(ldsp, yp) ≈ loglikelihood(ldsp, yp) + lp rtol = 1e-10
    return nothing
end

function test_causal_recovery()
    #=
    The controller is what the design identifies (see
    `docs/dev/lqr/noise_prototype.md`): from a perturbed start, EM on the model's
    own data recovers the closed-loop maps and the plant noise. Absolute costs are
    not identified at this size, so they are not tested.
    =#
    rng = StableRNG(80)
    sm, lds = causal_fixture(rng; affine=false, tmax=25)
    Σtrue = copy(sm.Σ[1:2, 1:2])
    _, Wt = lqr_riccati_sequence(sm, 25)[[1, 3]]
    Φtrue = [Wt[t + 1] * sm.A for t in 1:24]
    _, y = rand(rng, lds, fill(25, 150))
    fitm = deepcopy(lds)
    fitm.state_model.A .= 0.95I(2)
    fitm.state_model.Qc[1] .= Matrix(1.0I, 2, 2)
    fitm.state_model.Σ .= Diagonal([0.05, 0.05, 1.0, 1.0])
    refresh!(fitm.state_model)
    fit!(fitm, y; max_iter=40, progress=false)
    _, Wf = lqr_riccati_sequence(fitm.state_model, 25)[[1, 3]]
    Φfit = [Wf[t + 1] * fitm.state_model.A for t in 1:24]
    @test mean(norm(Φfit[t] - Φtrue[t]) / norm(Φtrue[t]) for t in 1:24) < 0.05
    @test norm(fitm.state_model.Σ[1:2, 1:2] - Σtrue) / norm(Σtrue) < 0.5
    return nothing
end

function test_causal_rejections()
    sm, lds = causal_fixture(StableRNG(90))
    slds = SLDS(; A=[0.9 0.1; 0.1 0.9], πₖ=[0.5, 0.5], LDSs=[lds, deepcopy(lds)])
    @test_throws ArgumentError validate_SLDS(slds)
    _, y = rand(StableRNG(91), lds, fill(10, 2))
    @test_throws ArgumentError fit!(slds, y; max_iter=1, progress=false)
    # A hand-edited model that breaks the block structure is refused.
    bad = deepcopy(sm)
    bad.Σ[1, 3] = bad.Σ[3, 1] = 0.01
    @test_throws ArgumentError StateSpaceDynamics._check_causal_structure(bad)
    smf, _ = causal_fixture(StableRNG(92); fixed=0.4)
    smf.Σ[3, 3] = 0.7
    @test_throws ArgumentError StateSpaceDynamics._check_causal_structure(smf)
    return nothing
end
