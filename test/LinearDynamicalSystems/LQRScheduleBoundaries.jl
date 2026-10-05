#=============================================================================
Known epoch boundaries of an inverse-LQR cost schedule (`lqr_switches.jl`):
bridges, entry priors and the per-regime reference gate.

The reference everything is checked against is a dense Gaussian built from the
definitions — forward moments of `z` with the entry transition derived from
`λ' = μ + K (x − r) + e₁`, `x' = A x − S λ' + … + e₂`, the bridge and trial-end
pseudo-observations, and a Gaussian emission. With a Gaussian emission the ELBO
at the exact posterior is `log p(y, goals = 0)`, and the conditional ELBO
`log p(y | goals = 0)`: one comparison covers the kernels, the smoother's
routing, the statistics, the Q-terms and the exact normalizer.
=============================================================================#

function _sb_fixture(; seed=1, n=2, m=2, gate=true, entry=true, bridge=true, terminal=true)
    rng = StableRNG(seed)
    A = Matrix(0.9I, n, n) .+ 0.05 .* randn(rng, n, n)
    S = let B = randn(rng, n, n)
        0.3 .* (B * B') ./ n + 0.1I
    end
    Qs = [
        let B = randn(rng, n, n)
            Matrix(Symmetric(0.2 .* B * B' / n + 0.1I))
        end for _ in 1:(bridge ? 4 : 3)
    ]
    d = 2n
    Σ = let B = randn(rng, d, d)
        Matrix(Symmetric(0.05 .* B * B' / d + 0.05I))
    end
    L = 10
    sched = [1, 1, 1, 1, 2, 2, 2, 2, 2, 3]
    G = randn(rng, n, m)
    Bu = vcat(0.1 .* randn(rng, n, m), zeros(n, m))
    h = 0.05 .* randn(rng, d)
    gatem = gate ? (bridge ? Bool[1 0; 0 1; 0 1; 1 0] : Bool[1 0; 0 1; 0 1]) : nothing
    sm = LQRStateModel(
        A,
        S,
        Qs,
        Σ;
        schedule=sched,
        terminal=terminal,
        condition_terminal=false,
        Σf=Matrix(0.3I, n, n),
        hf=0.1 .* randn(rng, n),
        h=h,
        Bu=Bu,
        Gref=G,
        x0=0.1 .* randn(rng, d),
        P0=Matrix(0.5I, d, d),
        gref_gate=gatem,
        bridges=bridge ? [5 => 4] : Pair{Int,Int}[],
        entries=entry ? [5] : Int[],
    )
    if entry
        ep = sm.switches[1].entry
        ep.μ .= 0.2 .* randn(rng, n)
        ep.K .= 0.3 .* randn(rng, n, n)
        ep.P .= Matrix(Symmetric(0.4I + 0.1 .* (x -> x * x')(randn(rng, n, n))))
        refresh!(sm)
    end
    p = 3
    C = hcat(randn(rng, p, n), zeros(p, n))
    obs = GaussianObservationModel(C, Matrix(0.2I, p, p), 0.1 .* randn(rng, p))
    lds = LinearDynamicalSystem(sm, obs)
    return lds, rng
end

# Dense reference for one trial: forward moments of z, then the Gaussian marginal of
# [y; goals]. Entry transition rebuilt from its definition, not from the cache.
function _sb_dense_trial(lds, y, u, off; goals_only=false)
    sm0 = lds.state_model
    sm = SSD._with_cost_offset(sm0, off)
    c = sm.cache
    n = SSD._plant_dim(sm)
    d = 2n
    T = size(u, 2)
    xr, lr = 1:n, (n + 1):d
    m = zeros(d * T)
    C = zeros(d * T, d * T)
    blk(t) = ((t - 1) * d + 1):(t * d)
    x0 = sm.x0
    m[blk(1)] .= x0
    C[blk(1), blk(1)] .= sm.P0
    for t in 1:(T - 1)
        a = t + off
        s = findfirst(s -> s.pos == a && s.entry !== nothing, sm.switches)
        if s === nothing
            k = SSD._regime(sm, t)
            F, f, Q = c.M[k], c.bfwd .+ c.Bfwd[k] * u[:, t], Matrix(c.Qfwd)
        else
            ep = sm.switches[s].entry
            kp = sm.schedule[a - 1]
            Gp = isempty(sm.gref_gate) ? sm.Gref : sm.Gref .* transpose(sm.gref_gate[kp, :])
            r = Gp * u[:, t]
            # λ' = K x + (μ − K r) + e1 ; x' = A x − S λ' + h_x + Bux u + e2
            Fl = hcat(ep.K, zeros(n, n))
            fl = ep.μ .- ep.K * r
            Fx = hcat(sm.A, zeros(n, n)) .- sm.S * Fl
            fx = sm.h[xr] .+ sm.Bu[xr, :] * u[:, t] .- sm.S * fl
            F = vcat(Fx, Fl)
            f = vcat(fx, fl)
            Gn = [-sm.S Matrix(I, n, n); Matrix(I, n, n) zeros(n, n)]  # (e1, e2)
            Q = Gn * cat(ep.P, sm.Σ[xr, xr]; dims=(1, 2)) * Gn'
        end
        m[blk(t + 1)] .= F * m[blk(t)] .+ f
        for s2 in 1:t
            C[blk(t + 1), blk(s2)] .= F * C[blk(t), blk(s2)]
            C[blk(s2), blk(t + 1)] .= C[blk(t + 1), blk(s2)]'
        end
        C[blk(t + 1), blk(t + 1)] .= F * C[blk(t), blk(t)] * F' .+ Q
    end
    rows = Matrix{Float64}[]
    shift = Vector{Float64}[]
    noise = Matrix{Float64}[]
    vals = Vector{Float64}[]
    function factor!(t, k)
        Gk = if sm.gref_gate === nothing || isempty(sm.gref_gate)
            sm.Gref
        else
            sm.Gref .* transpose(sm.gref_gate[k, :])
        end
        H = zeros(n, d * T)
        H[:, blk(t)] .= hcat(-sm.Qc[k], Matrix(I, n, n))
        push!(rows, H)
        push!(shift, sm.Qc[k] * Gk * u[:, t] .- sm.hf)
        push!(noise, sm.Σf)
        return push!(vals, zeros(n))
    end
    for s in sm.switches
        tb = s.pos - off
        (s.bridge != 0 && 1 <= tb <= T - 1) && factor!(tb, s.bridge)
    end
    sm.terminal &&
        factor!(T, sm.terminal_regime > 0 ? sm.terminal_regime : sm.schedule[T + off])
    if !goals_only
        om = lds.obs_model
        for t in 1:T
            H = zeros(size(om.C, 1), d * T)
            H[:, blk(t)] .= om.C
            push!(rows, H)
            push!(shift, om.d)
            push!(noise, om.R)
            push!(vals, y[:, t])
        end
    end
    H = vcat(rows...)
    h0 = vcat(shift...)
    R = cat(noise...; dims=(1, 2))
    v = vcat(vals...)
    μo = H * m .+ h0
    So = Symmetric(H * C * H' .+ R)
    r = v .- μo
    return -0.5 * (length(v) * log(2π) + logdet(So) + dot(r, So \ r))
end

function test_lqr_switches_dense()
    for (gate, entry, bridge) in
        ((true, true, true), (false, true, false), (true, false, true))
        lds, rng = _sb_fixture(; gate=gate, entry=entry, bridge=bridge)
        specs = [(10, 0), (7, 3), (6, 0), (5, 0), (4, 5), (6, 4), (8, 1)]
        u = [randn(rng, 2, T) for (T, _) in specs]
        y = [randn(rng, 3, T) for (T, _) in specs]
        off = [o for (_, o) in specs]
        joint = sum(_sb_dense_trial(lds, y[i], u[i], off[i]) for i in eachindex(specs))
        goals = sum(
            _sb_dense_trial(lds, y[i], u[i], off[i]; goals_only=true) for
            i in eachindex(specs)
        )
        lds.state_model.condition_terminal = false
        @test elbo(lds, y; ux=u, cost_offset=off) ≈ joint rtol = 1e-8
        lds.state_model.condition_terminal = true
        @test elbo(lds, y; ux=u, cost_offset=off) ≈ joint - goals rtol = 1e-8
        designs = [(ux=u[i], ux0=zeros(0), off=off[i], count=1.0) for i in eachindex(specs)]
        @test SSD._lqr_terminal_logz_sum(lds.state_model, designs) ≈ goals rtol = 1e-9
        # Per trial, too.
        te = trial_elbos(lds, y; ux=u, cost_offset=off)
        for i in eachindex(specs)
            @test te[i] ≈
                _sb_dense_trial(lds, y[i], u[i], off[i]) -
                  _sb_dense_trial(lds, y[i], u[i], off[i]; goals_only=true) rtol = 1e-8
        end
    end
    return nothing
end

"""The gate gives regime `k` the forward input `G (B_u − [0; Q_k G_r D_k])`."""
function test_lqr_gref_gate_cache()
    lds, _ = _sb_fixture()
    sm = lds.state_model
    n = 2
    for k in eachindex(sm.Qc)
        Gk = sm.Gref .* transpose(sm.gref_gate[k, :])
        @test sm.cache.Bfwd[k] ≈ sm.cache.G * (sm.Bu .- vcat(zeros(n, 2), sm.Qc[k] * Gk))
        @test sm.cache.Ftrm[k] ≈ sm.Qc[k] * Gk
    end
    return nothing
end

function test_lqr_switches_em()
    for cond in (false, true), emission in (:gaussian, :poisson)
        truth, rng = _sb_fixture()
        truth.state_model.condition_terminal = cond
        model = if emission === :gaussian
            truth
        else
            C = hcat(0.4 .* randn(rng, 12, 2), zeros(12, 2))
            LinearDynamicalSystem(
                deepcopy(truth.state_model),
                PoissonObservationModel(C, fill(log(2.0), 12)),
            )
        end
        ntr = 40
        lens = emission === :gaussian ? fill(10, ntr) : [rand(rng, 6:10) for _ in 1:ntr]
        off =
            emission === :gaussian ? zeros(Int, ntr) : [rand(rng, 0:(10 - T)) for T in lens]
        u = [randn(rng, 2, T) for T in lens]
        _, y = rand(rng, model, lens; ux=u, cost_offset=off)
        fitm = deepcopy(model)
        sm = fitm.state_model
        sm.A .+= 0.03 .* randn(rng, 2, 2)
        foreach(Q -> Q .*= 1.3, sm.Qc)
        ep = sm.switches[1].entry
        ep.μ .= 0
        ep.K .= 0
        ep.P .= Matrix(1.0I, 2, 2)
        sm.Gref .*= 0.5
        refresh!(sm)
        kw = emission === :gaussian ? (;) : (; cost_offset=off)
        trace = collect(fit!(fitm, y; ux=u, max_iter=6, progress=false, kw...))
        @test all(isfinite, trace)
        @test all(diff(trace) .>= -1e-6 * abs(trace[end]))
        @test !iszero(ep.K)
    end
    return nothing
end

"""With bridges (and the gate) every term of the conditional M-step objective is
in the statistics, so its Fisher-identity gradient is exact."""
function test_lqr_switches_conditional_gradient()
    lds, rng = _sb_fixture(; entry=false)
    lds.state_model.condition_terminal = true
    lens = [10, 10, 8, 6]
    u = [randn(rng, 2, T) for T in lens]
    y = [randn(rng, 3, T) for T in lens]
    data = SSD.Data(lds, y; ux=u)
    SSD._prepare_lqr!(lds, data)
    tfs = SSD.initialize_FilterSmooth(lds, data.tsteps)
    pool = SSD._lqr_sws_pool(lds, data)
    hs = SSD._initialize_td_sufficient_statistics(Float64, lds, data.tsteps)
    SSD._td_init_const_blocks!(pool[1], lds, data)
    SSD.estep!(lds, hs, tfs, data, pool)
    problem = SSD._lqr_conditional_problem([lds], [hs], [ones(Int, 1) for _ in 1:4])
    θ = copy(problem.theta)
    g = zeros(length(θ))
    @test isfinite(problem.evaluate!(g, copy(θ)))
    fd = similar(g)
    for i in eachindex(θ)
        h = 1e-6 * max(1.0, abs(θ[i]))
        up, dn = copy(θ), copy(θ)
        up[i] += h
        dn[i] -= h
        fd[i] = (problem.evaluate!(nothing, up) - problem.evaluate!(nothing, dn)) / (2h)
    end
    problem.write!(θ)
    @test maximum(abs, g .- fd) / max(1.0, maximum(abs, fd)) < 1e-6
    return nothing
end

function test_lqr_switches_grouped()
    for cond in (false, true)
        truth, rng = _sb_fixture()
        truth.state_model.condition_terminal = cond
        C = hcat(0.4 .* randn(rng, 10, 2), zeros(10, 2))
        model = LinearDynamicalSystem(
            deepcopy(truth.state_model), PoissonObservationModel(C, fill(log(2.0), 10))
        )
        ntr = 30
        lens = [rand(rng, 6:10) for _ in 1:ntr]
        off = [rand(rng, 0:(10 - T)) for T in lens]
        u = [randn(rng, 2, T) for T in lens]
        _, y = rand(rng, model, lens; ux=u, cost_offset=off)
        fitm = deepcopy(model)
        fitm.state_model.A .*= 0.97
        refresh!(fitm.state_model)
        set_depends_on!(fitm.state_model, (structure=[isodd(i) ? 1 : 2 for i in 1:ntr],))
        trace = collect(fit!(fitm, y; ux=u, cost_offset=off, max_iter=5, progress=false))
        @test all(isfinite, trace)
        @test all(diff(trace) .>= -1e-6 * abs(trace[end]))
        # The variants share the parent's switches, entry prior included.
        @test all(
            v.switches === fitm.state_model.switches for v in fitm.state_model.variants
        )
    end
    return nothing
end

"""Without a terminal factor the forward roll draws entries from their own
transition: the sample mean follows the dense forward mean."""
function test_lqr_switches_rand_forward()
    lds, rng = _sb_fixture(; terminal=false, bridge=false)
    sm = lds.state_model
    T = 9
    u = randn(rng, 2, T)
    N = 20000
    _, _ = rand(rng, lds, [T]; ux=[u])
    z, _ = rand(rng, lds, fill(T, N); ux=fill(u, N))
    zbar = sum(z) ./ N
    m = copy(sm.x0)
    c = sm.cache
    for t in 1:(T - 1)
        e = SSD._entry_into(sm, t + 1)
        if e == 0
            k = SSD._regime(sm, t)
            m = c.M[k] * m .+ c.bfwd .+ c.Bfwd[k] * u[:, t]
        else
            ec = c.switch[e]
            m = ec.M * m .+ ec.b .+ ec.B * u[:, t]
        end
        @test maximum(abs, zbar[:, t + 1] .- m) < 0.05
    end
    return nothing
end

function test_lqr_switches_validation()
    lds, _ = _sb_fixture()
    sm = lds.state_model
    @test length(sm.switches) == 1 && sm.switches[1].bridge == 4
    @test_throws ArgumentError set_schedule_boundaries!(deepcopy(sm); bridges=[1 => 4])
    @test_throws ArgumentError set_schedule_boundaries!(deepcopy(sm); bridges=[10 => 4])
    @test_throws ArgumentError set_schedule_boundaries!(deepcopy(sm); bridges=[5 => 9])
    @test_throws ArgumentError set_schedule_boundaries!(deepcopy(sm); entries=[5, 5])
    noterm, _ = _sb_fixture(; terminal=false, bridge=false)
    @test_throws ArgumentError set_schedule_boundaries!(
        noterm.state_model; bridges=[5 => 3]
    )
    # A failed call leaves the configuration as it was.
    @test noterm.state_model.switches[1].entry !== nothing
    @test_throws DimensionMismatchError set_gref_gate!(deepcopy(sm), trues(2, 2))
    # Models with switches take the per-trial smoother paths.
    @test SSD._has_switches(lds)
    # Clearing.
    cleared = set_schedule_boundaries!(deepcopy(sm))
    @test isempty(cleared.switches) && isempty(cleared.cache.switch)
    # A switching model takes `set_boundaries!`, not these.
    obs = lds.obs_model
    @test_throws ArgumentError validate_SLDS(
        SLDS(;
            A=[0.9 0.1; 0.1 0.9],
            πₖ=[0.5, 0.5],
            LDSs=[LinearDynamicalSystem(deepcopy(sm), obs) for _ in 1:2],
        ),
    )
    return nothing
end
