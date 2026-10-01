#=============================================================================
Boundaries between an SLDS's discrete states: exit bridges.

A bridged state applies its terminal factor wherever its segment ends — at every
exit `s_t = k → s_{t+1} ≠ k` as well as at the end of a trial. The anchors:

* the discrete layer's time-indexed transition reproduces, exactly, the chain
  posterior of a brute-force enumeration over paths with the bridge potentials
  attached, and the exit weights are that posterior's exit probabilities;
* the smoother's objective, gradient and Hessian agree with each other under
  bridges (finite differences);
* exits add exactly their weighted terminal moments to the M-step statistics;
* a fit through a bridge runs, improves its converged bound, and honours the
  bridge: the costate meets the terminal relation at the exits.

Uses the inverse-LQR switching fixtures of `LQRSLDS.jl` (`hslds_state`,
`hslds_data`, `_trace`).
=============================================================================#

const _SB = StateSpaceDynamics

"""Every path's unnormalised weight, with bridge potentials on exits."""
function _bridge_path_posterior(A, π, logL, Φ, bridge)
    K, T = size(logL)
    paths = Iterators.product(ntuple(_ -> 1:K, T)...)
    γ = zeros(K, T)
    ξ = [zeros(K, K) for _ in 1:(T - 1)]
    total = 0.0
    for s in paths
        w = π[s[1]] * exp(logL[s[1], 1])
        for t in 2:T
            w *= A[s[t - 1], s[t]] * exp(logL[s[t], t])
            bridge[s[t - 1]] && s[t] != s[t - 1] && (w *= exp(Φ[s[t - 1], t - 1]))
        end
        total += w
        for t in 1:T
            γ[s[t], t] += w
        end
        for t in 1:(T - 1)
            ξ[t][s[t], s[t + 1]] += w
        end
    end
    return γ ./ total, [x ./ total for x in ξ]
end

"""
Forward-backward on the bridged discrete layer is exact: `γ` and `ξ` match a
brute-force sum over every path of the chain with the exit potentials attached,
trial by trial (so no potential leaks across a trial boundary), and the exit
weights are that posterior's `q(s_t = k, s_{t+1} ≠ k)`.
"""
function test_bridge_forward_backward_exact()
    rng = StableRNG(41)
    K = 3
    tsteps = [5, 4]
    total = sum(tsteps)
    A = [0.6 0.3 0.1; 0.2 0.5 0.3; 0.25 0.25 0.5]
    π = [0.5, 0.3, 0.2]
    logL = randn(rng, K, total)
    Φ = 2.0 .* randn(rng, K, total)
    bridge = [true, false, true]

    dl = _SB.SLDSDiscreteLayer(copy(A), copy(π), copy(logL))
    dl.bridge = copy(bridge)
    dl.exit_logL = copy(Φ)
    dl.exit_w = zeros(K, total)
    seq_ends = cumsum(tsteps)
    fb = _SB._make_slds_fb_storage(dl, seq_ends)
    _SB.HMMs.forward_backward!(
        fb,
        dl,
        collect(1:total),
        collect(1:total);
        seq_ends=seq_ends,
        transition_marginals=true,
    )
    _SB._slds_exit_weights!(dl.exit_w, fb, seq_ends, dl.bridge)

    for trial in eachindex(tsteps)
        t1, t2 = _SB.HMMs.seq_limits(seq_ends, trial)
        γ, ξ = _bridge_path_posterior(A, π, logL[:, t1:t2], Φ[:, t1:t2], bridge)
        @test fb.γ[:, t1:t2] ≈ γ rtol = 1e-10
        for t in t1:(t2 - 1)
            @test fb.ξ[t] ≈ ξ[t - t1 + 1] rtol = 1e-10
            for k in 1:K
                expected = bridge[k] ? γ[k, t - t1 + 1] - ξ[t - t1 + 1][k, k] : 0.0
                @test dl.exit_w[k, t] ≈ expected atol = 1e-12
            end
        end
        @test all(iszero, dl.exit_w[:, t2])
    end

    # Without bridges the time-indexed transition is the plain chain.
    plain = _SB.SLDSDiscreteLayer(copy(A), copy(π), copy(logL))
    @test _SB.HMMs.transition_matrix(plain, 3) === plain.A
    @test _SB.HMMs.transition_matrix(dl, 1) === dl.A
    return nothing
end

"""A control state (bridged, joint objective) and a `:free` state sharing one latent."""
function _bridge_model(; p::Int=4, n::Int=2, stay::Real=0.9, Qc=[0.4 0.05; 0.05 0.3])
    control = hslds_state(Qc; p=p, terminal=true)
    control.state_model.condition_terminal = false
    d = 2n
    free = LinearDynamicalSystem(
        free_state_model(
            0.9 * Matrix(1.0I, d, d), Matrix(0.05I, d, d); P0=Matrix(0.3I, d, d)
        ),
        deepcopy(control.obs_model),
    )
    free.state_model.observe_costate = false
    A, πₖ = banded_transition(2; stay=stay)
    slds = SLDS(; A=A, πₖ=πₖ, LDSs=[control, free])
    set_boundaries!(slds; bridge_states=[1])
    return slds
end

"""
The bridged smoother is self-consistent: `gradient!` is the derivative of the
summed `joint_loglikelihood!` and `hessian!` the derivative of `gradient!`, with
random responsibilities and exit weights. With zero exit weights the bridge terms
vanish exactly.
"""
function test_bridge_smoother_derivatives()
    rng = StableRNG(43)
    slds = _bridge_model()
    tsteps = 7
    d = slds.LDSs[1].latent_dim
    y = 0.5 .* randn(rng, 4, tsteps)
    x = 0.3 .* randn(rng, d, tsteps)
    w = rand(rng, 2, tsteps)
    w ./= sum(w; dims=1)
    ew = zeros(2, tsteps)
    ew[1, 1:(tsteps - 1)] .= 0.5 .* rand(rng, tsteps - 1)

    ws = _SB.SLDSSmoothWorkspace(Float64, slds, tsteps)
    f(xv) = sum(_SB.joint_loglikelihood!(ws, slds, xv, y, w; ew=ew))
    g = copy(_SB.gradient!(ws, slds, x, y, w; ew=ew))
    h = 1e-6
    gfd = similar(x)
    for i in eachindex(x)
        xp = copy(x)
        xm = copy(x)
        xp[i] += h
        xm[i] -= h
        gfd[i] = (f(xp) - f(xm)) / (2h)
    end
    @test g ≈ gfd rtol = 1e-6 atol = 1e-6

    _SB.hessian!(ws, slds, x, y, w; ew=ew)
    Hd = [copy(ws.btd.H_diag[t]) for t in 1:tsteps]
    for t in 1:tsteps, i in 1:d
        xp = copy(x)
        xm = copy(x)
        xp[i, t] += h
        xm[i, t] -= h
        gp = copy(_SB.gradient!(ws, slds, xp, y, w; ew=ew))
        gm = copy(_SB.gradient!(ws, slds, xm, y, w; ew=ew))
        col = (gp[:, t] .- gm[:, t]) ./ (2h)
        @test Hd[t][:, i] ≈ col rtol = 1e-5 atol = 1e-6
    end

    plain = sum(_SB.joint_loglikelihood!(ws, slds, x, y, w))
    zero_ew = sum(_SB.joint_loglikelihood!(ws, slds, x, y, w; ew=zeros(2, tsteps)))
    @test zero_ew == plain
    @test f(x) != plain
    return nothing
end

"""
Exits add exactly their weighted terminal moments to the bridged state's terminal
statistics, on the bridge's own cost regime, and nothing else changes.
"""
function test_bridge_aggregate_stats()
    rng = StableRNG(47)
    slds = _bridge_model()
    lds = slds.LDSs[1]
    sm = lds.state_model
    tsteps = [6, 8]
    ys = [0.5 .* randn(rng, 4, T) for T in tsteps]
    data = _SB.Data(lds, ys)
    tfs = _SB.initialize_FilterSmooth(lds, tsteps)
    for (i, T) in enumerate(tsteps)
        tfs[i].x_smooth .= 0.4 .* randn(rng, size(tfs[i].x_smooth)...)
        for t in 1:T
            L = 0.2 .* randn(rng, 4, 4)
            tfs[i].p_smooth[:, :, t] .= L * L' + 0.1I
        end
    end
    γ = [rand(rng, T) for T in tsteps]
    e = [[t < T ? 0.3 * rand(rng) : 0.0 for t in 1:T] for T in tsteps]

    plain = _SB._initialize_td_sufficient_statistics(Float64, lds, tsteps)
    _SB._aggregate_lqr_stats_weighted!(plain, tfs, lds, data, γ)
    bridged = _SB._initialize_td_sufficient_statistics(Float64, lds, tsteps)
    _SB._aggregate_lqr_stats_weighted!(bridged, tfs, lds, data, γ; exit_weights=e)

    kb = _SB._bridge_regime(sm)
    d = lds.latent_dim
    expected = copy(plain.term_zz[kb])
    for (i, T) in enumerate(tsteps), t in 1:(T - 1)
        z = tfs[i].x_smooth[:, t]
        expected[1:d, 1:d] .+= e[i][t] .* (z * z' .+ tfs[i].p_smooth[:, :, t])
        expected[1:d, d + 1] .+= e[i][t] .* z
        expected[d + 1, 1:d] .+= e[i][t] .* z
    end
    expected[d + 1, d + 1] = plain.term_n[kb] + sum(sum, e)
    @test bridged.term_n[kb] ≈ plain.term_n[kb] + sum(sum, e) rtol = 1e-12
    @test bridged.term_zz[kb] ≈ expected rtol = 1e-10
    for k in eachindex(plain.zz)
        @test bridged.zz[k] == plain.zz[k]
        @test bridged.yy[k] == plain.yy[k]
    end
    return nothing
end

"""`set_boundaries!` refuses what a bridge cannot be, and clears on no states."""
function test_bridge_validation()
    slds = _bridge_model()
    @test slds.boundaries.bridge == [true, false]
    @test_throws ArgumentError set_boundaries!(deepcopy(slds); bridge_states=[3])
    # A `:free` state has no terminal factor to apply.
    @test_throws ArgumentError set_boundaries!(deepcopy(slds); bridge_states=[2])
    # A control state without a terminal factor.
    bad = deepcopy(slds)
    bad.boundaries = nothing
    bad.LDSs[1].state_model.terminal = false
    @test_throws ArgumentError set_boundaries!(bad; bridge_states=[1])
    # The conditional objective's normalizer does not carry bridges yet.
    cond = deepcopy(slds)
    cond.LDSs[1].state_model.condition_terminal = true
    @test_throws ArgumentError validate_SLDS(cond)
    # Clearing.
    cleared = set_boundaries!(deepcopy(slds); bridge_states=Int[])
    @test cleared.boundaries === nothing
    return nothing
end

"""
End to end: a banded control → free model fitted with a bridge on the control
state runs, keeps its band, improves its converged bound, and the bridge does
what it says — at the bins where the chain leaves the control state, the
posterior costate is closer to the terminal relation than without the bridge.
"""
function test_bridge_fit()
    p, tsteps, ntrials = 4, 30, 6
    ys = hslds_data(p, tsteps, ntrials)
    bridged = _bridge_model(; p=p, stay=0.85)
    plain = deepcopy(bridged)
    plain.boundaries = nothing

    before = elbo(bridged, ys)
    trace = _trace(fit!(bridged, ys; max_iter=8, progress=false, rng=StableRNG(5)))
    after = elbo(bridged, ys)
    @test all(isfinite, trace)
    @test after > before
    @test bridged.A[2, 1] == 0 && bridged.πₖ == [1.0, 0.0]
    @test symplectic_defect(bridged.LDSs[1].state_model) < 1e-10

    # The bridge pulls the exit-bin costate onto λ = Q_f (x − r) + h_f.
    sm = bridged.LDSs[1].state_model
    res_b = smooth(bridged, ys)
    res_p = smooth(plain, ys)
    function exit_residual(res)
        total, weight = 0.0, 0.0
        rf = zeros(_SB._plant_dim(sm))
        for (i, γ) in enumerate(res.γ), t in 1:(tsteps - 1)
            # Mass leaving the control state at t, read off the marginals.
            e = max(0.0, γ[1, t] - γ[1, t + 1])
            _SB._bridge_residual!(rf, sm, res.x[i], t, nothing)
            total += e * sum(abs2, rf)
            weight += e
        end
        return total / max(weight, eps())
    end
    @test exit_residual(res_b) < exit_residual(res_p)
    @test isfinite(res_b.elbo) && res_b.elbo != res_p.elbo
    return nothing
end

"""
The grouped (`depends_on`) path and the multi-sample E-step carry bridges too: a
stitched-emission fit and a `num_samples = 2` fit both run with finite traces and
keep the band, and with every trial in one group the grouped score is the
ungrouped one.
"""
function test_bridge_grouped_and_sampled()
    p, tsteps, ntrials = 4, 24, 4
    ys = hslds_data(p, tsteps, ntrials)

    one_group = _bridge_model(; p=p)
    for lds in one_group.LDSs
        lds.obs_model.depends_on = (C=fill(:a, ntrials), d=fill(:a, ntrials))
    end
    ungrouped = _bridge_model(; p=p)
    @test elbo(one_group, ys) ≈ elbo(ungrouped, ys) rtol = 1e-8

    stitched = _bridge_model(; p=p)
    labels = [:a, :a, :b, :b]
    for lds in stitched.LDSs
        lds.obs_model.depends_on = (C=labels, d=labels, R=labels)
    end
    trace = _trace(fit!(stitched, ys; max_iter=3, progress=false, rng=StableRNG(9)))
    @test all(isfinite, trace)
    @test stitched.A[2, 1] == 0

    sampled = _bridge_model(; p=p)
    trace = _trace(
        fit!(sampled, ys; max_iter=3, num_samples=2, progress=false, rng=StableRNG(9))
    )
    @test all(isfinite, trace)
    @test sampled.A[2, 1] == 0
    return nothing
end
