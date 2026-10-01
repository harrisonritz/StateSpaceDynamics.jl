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

# ============================================================================
# Entry priors
# ============================================================================

"""A banded control → free → control model whose last state has an entry prior
(and both control states bridges)."""
function _entry_model(; p::Int=4, n::Int=2, stays=[0.85, 0.85], middle::Symbol=:free)
    c1 = hslds_state([0.4 0.05; 0.05 0.3]; p=p, terminal=true)
    c1.state_model.condition_terminal = false
    c3 = hslds_state([0.6 0.0; 0.0 0.5]; p=p, terminal=true, seed=11)
    c3.state_model.condition_terminal = false
    d = 2n
    middle_sm = if middle === :hold
        # Control → hold → control: the hold state shares the plant.
        sm1 = c1.state_model
        hold_state_model(
            copy(sm1.A),
            copy(sm1.S),
            [0.5 0.0; 0.0 0.4],
            Matrix(0.05I, d, d);
            P0=Matrix(0.3I, d, d),
        )
    else
        free_state_model(0.9 * Matrix(1.0I, d, d), Matrix(0.05I, d, d); P0=Matrix(0.3I, d, d))
    end
    mid = LinearDynamicalSystem(middle_sm, deepcopy(c1.obs_model))
    mid.state_model.observe_costate = false
    A, πₖ = banded_transition(3; stay=stays)
    slds = SLDS(; A=A, πₖ=πₖ, LDSs=[c1, mid, c3])
    set_boundaries!(slds; bridge_states=[1, 3], entry_states=[3], entry_cov=0.5)
    return slds
end

"""
Forward-backward with entry potentials is exact: brute force over paths, where an
`i → j` transition into an entry-prior state `j` also pays `exp(Φ[i, j, t])`.
"""
function test_entry_forward_backward_exact()
    rng = StableRNG(53)
    K, T = 3, 5
    A = [0.6 0.3 0.1; 0.2 0.5 0.3; 0.25 0.25 0.5]
    π = [0.5, 0.3, 0.2]
    logL = randn(rng, K, T)
    Φ = 2.0 .* randn(rng, K, K, T)
    entry = [false, true, true]

    dl = _SB.SLDSDiscreteLayer(copy(A), copy(π), copy(logL))
    dl.entry = copy(entry)
    dl.entry_logL = copy(Φ)
    dl.entry_w = zeros(K, K, T)
    fb = _SB._make_slds_fb_storage(dl, [T])
    _SB.HMMs.forward_backward!(
        fb, dl, collect(1:T), collect(1:T); seq_ends=[T], transition_marginals=true
    )
    _SB._slds_entry_weights!(dl.entry_w, fb, [T], dl.entry)

    γ = zeros(K, T)
    ξ = [zeros(K, K) for _ in 1:(T - 1)]
    total = 0.0
    for s in Iterators.product(ntuple(_ -> 1:K, T)...)
        w = π[s[1]] * exp(logL[s[1], 1])
        for t in 2:T
            w *= A[s[t - 1], s[t]] * exp(logL[s[t], t])
            entry[s[t]] && s[t] != s[t - 1] && (w *= exp(Φ[s[t - 1], s[t], t]))
        end
        total += w
        for t in 1:T
            γ[s[t], t] += w
        end
        for t in 1:(T - 1)
            ξ[t][s[t], s[t + 1]] += w
        end
    end
    @test fb.γ ≈ γ ./ total rtol = 1e-10
    for t in 1:(T - 1)
        @test fb.ξ[t] ≈ ξ[t] ./ total rtol = 1e-10
        for i in 1:K, j in 1:K
            expected = (entry[j] && i != j) ? ξ[t][i, j] / total : 0.0
            @test dl.entry_w[i, j, t + 1] ≈ expected atol = 1e-12
        end
    end
    return nothing
end

"""
The entry swap's objective, gradient and block-tridiagonal Hessian agree by
finite differences — diagonal *and* off-diagonal blocks — and a zero entry weight
leaves the smoother's objective exactly as it was.
"""
function test_entry_smoother_derivatives(; middle::Symbol=:free)
    rng = StableRNG(59)
    slds = _entry_model(; middle=middle)
    slds.boundaries.entry[3].μ .= [0.2, -0.1]
    slds.boundaries.entry[3].K .= [0.3 0.1; -0.2 0.4]
    tsteps = 6
    d = slds.LDSs[1].latent_dim
    y = 0.5 .* randn(rng, 4, tsteps)
    x = 0.3 .* randn(rng, d, tsteps)
    w = rand(rng, 3, tsteps)
    w ./= sum(w; dims=1)
    pw = zeros(3, 3, tsteps)
    pw[1, 3, 2:end] .= 0.3 .* rand(rng, tsteps - 1)
    pw[2, 3, 2:end] .= 0.3 .* rand(rng, tsteps - 1)

    ws = _SB.SLDSSmoothWorkspace(Float64, slds, tsteps)
    f(xv) = sum(_SB.joint_loglikelihood!(ws, slds, xv, y, w; pw=pw))
    g = copy(_SB.gradient!(ws, slds, x, y, w; pw=pw))
    h = 1e-6
    gfd = similar(x)
    for i in eachindex(x)
        xp, xm = copy(x), copy(x)
        xp[i] += h
        xm[i] -= h
        gfd[i] = (f(xp) - f(xm)) / (2h)
    end
    @test g ≈ gfd rtol = 1e-6 atol = 1e-6

    _SB.hessian!(ws, slds, x, y, w; pw=pw)
    Hd = [copy(ws.btd.H_diag[t]) for t in 1:tsteps]
    Hs = [copy(ws.btd.H_sub[t]) for t in 1:(tsteps - 1)]      # ∂²/∂z_{t+1}∂z_t
    Hp = [copy(ws.btd.H_super[t]) for t in 1:(tsteps - 1)]    # ∂²/∂z_t∂z_{t+1}
    for t in 1:tsteps, i in 1:d
        xp, xm = copy(x), copy(x)
        xp[i, t] += h
        xm[i, t] -= h
        gp = copy(_SB.gradient!(ws, slds, xp, y, w; pw=pw))
        gm = copy(_SB.gradient!(ws, slds, xm, y, w; pw=pw))
        dg = (gp .- gm) ./ (2h)
        @test Hd[t][:, i] ≈ dg[:, t] rtol = 1e-5 atol = 1e-6
        t < tsteps && @test Hs[t][:, i] ≈ dg[:, t + 1] rtol = 1e-5 atol = 1e-6
        t > 1 && @test Hp[t - 1][:, i] ≈ dg[:, t - 1] rtol = 1e-5 atol = 1e-6
    end

    plain = sum(_SB.joint_loglikelihood!(ws, slds, x, y, w))
    @test sum(_SB.joint_loglikelihood!(ws, slds, x, y, w; pw=zeros(3, 3, tsteps))) == plain
    return nothing
end

"""
The entry prior's M-step is the exact weighted regression: entries whose new
costate is an exact affine function of where the state was relative to the
previous reference recover that function, and with `fit_gain = false` only the
offset moves.
"""
function test_entry_prior_mstep_exact()
    rng = StableRNG(61)
    slds = _entry_model()
    lds = slds.LDSs[3]
    n = 2
    d = 2n
    μ_true = [0.5, -0.3]
    K_true = [0.7 0.2; -0.1 0.4]
    tsteps = [5, 5, 5, 5, 5, 5]
    ys = [zeros(4, T) for T in tsteps]
    data = _SB.Data(lds, ys)
    tfs = _SB.initialize_FilterSmooth(lds, tsteps)
    seq_ends = cumsum(tsteps)
    entry_w = zeros(3, 3, last(seq_ends))
    for (i, T) in enumerate(tsteps)
        x = tfs[i].x_smooth
        x .= randn(rng, d, T)
        # Entries at local t = 3, from state 1 (no inputs, so r = 0).
        x[(n + 1):d, 3] .= μ_true .+ K_true * x[1:n, 2]
        t1 = seq_ends[i] - T + 1
        entry_w[1, 3, t1 + 2] = 1.0
        fill!(tfs[i].p_smooth, 0.0)
        fill!(tfs[i].p_smooth_tt1, 0.0)
    end
    st = _SB._entry_stats(_ -> slds, 3, tfs, data, entry_w, seq_ends)
    ep = deepcopy(slds.boundaries.entry[3])
    _SB._update_entry_prior!(ep, st)
    @test ep.μ ≈ μ_true atol = 1e-8
    @test ep.K ≈ K_true atol = 1e-8
    @test all(eigvals(Symmetric(ep.P)) .> 0)
    @test st.N[] ≈ length(tsteps)

    held = deepcopy(slds.boundaries.entry[3])
    held.fit_gain = false
    held.K .= K_true
    _SB._update_entry_prior!(held, st)
    @test held.μ ≈ μ_true atol = 1e-8
    @test held.K == K_true
    return nothing
end

"""
The acceptance guard keeps a structural step that does not lower the objective
with the entries' plant row counted, and undoes one that does.
"""
function test_entry_guard_restores()
    rng = StableRNG(67)
    slds = _entry_model()
    lds = slds.LDSs[3]
    sm = lds.state_model
    n = 2
    d = 2n
    T = 6
    ys = [zeros(4, T) for _ in 1:3]
    data = _SB.Data(lds, ys)
    tfs = _SB.initialize_FilterSmooth(lds, fill(T, 3))
    for i in 1:3
        tfs[i].x_smooth .= 0.3 .* randn(rng, d, T)
        for t in 1:T
            tfs[i].p_smooth[:, :, t] .= 0.01I(d)
        end
    end
    seq_ends = cumsum(fill(T, 3))
    entry_w = zeros(3, 3, last(seq_ends))
    entry_w[1, 3, 3] = entry_w[1, 3, T + 3] = 1.0
    st = [_SB._entry_stats(_ -> slds, 3, tfs, data, entry_w, seq_ends)]
    hs = _SB._initialize_td_sufficient_statistics(Float64, lds, fill(T, 3))
    _SB._aggregate_lqr_stats_weighted!(hs, tfs, lds, data, [ones(T) for _ in 1:3])
    stats = Union{Nothing,Vector{eltype(st)}}[nothing, nothing, st]
    A0 = copy(sm.A)

    # A step that wrecks the plant is undone.
    _SB._slds_entry_guarded([lds], [hs], stats, u -> (3, 1)) do
        sm.A .*= 3.0
        refresh!(sm)
    end
    @test sm.A == A0
    # A step that changes nothing is kept (trivially).
    _SB._slds_entry_guarded([lds], [hs], stats, u -> (3, 1)) do
        nothing
    end
    @test sm.A == A0
    # Without entry statistics the step runs unguarded.
    _SB._slds_entry_guarded([lds], [hs], nothing, u -> (3, 1)) do
        sm.A .*= 1.01
        refresh!(sm)
    end
    @test sm.A ≈ 1.01 .* A0
    return nothing
end

"""`set_boundaries!` refuses an entry prior on anything but a control state."""
function test_entry_validation()
    slds = _entry_model()
    @test slds.boundaries.entry[3] isa _SB.EntryPrior
    @test slds.boundaries.entry[1] === nothing
    @test slds.boundaries.entry[3].P == 0.5I(2)
    @test_throws ArgumentError set_boundaries!(deepcopy(slds); entry_states=[2])
    @test_throws ArgumentError set_boundaries!(deepcopy(slds); entry_states=[4])
    @test_throws ArgumentError set_boundaries!(
        deepcopy(slds); entry_states=[3], entry_cov=0.0
    )
    cond = deepcopy(slds)
    cond.LDSs[1].state_model.condition_terminal = true
    cond.LDSs[3].state_model.condition_terminal = true
    @test_throws ArgumentError validate_SLDS(cond)
    return nothing
end

"""
End to end: control (bridged) → free → control (bridged, entry prior) on a banded
chain fits with finite traces, improves its converged bound, keeps the band, and
moves the entry prior off its initial value; grouped and sampled fits run too.
"""
function test_entry_fit()
    p, tsteps, ntrials = 4, 36, 6
    ys = hslds_data(p, tsteps, ntrials)
    slds = _entry_model(; p=p)
    before = elbo(slds, ys)
    trace = _trace(fit!(slds, ys; max_iter=6, progress=false, rng=StableRNG(5)))
    after = elbo(slds, ys)
    @test all(isfinite, trace)
    @test after > before
    @test slds.A[2, 1] == 0 && slds.A[3, 1] == 0 && slds.A[3, 2] == 0
    ep = slds.boundaries.entry[3]
    @test norm(ep.μ) > 0
    @test isposdef(Symmetric(ep.P))
    for k in (1, 3)
        @test symplectic_defect(slds.LDSs[k].state_model) < 1e-10
    end

    labels = [:a, :a, :a, :b, :b, :b]
    grouped = _entry_model(; p=p)
    for lds in grouped.LDSs
        lds.obs_model.depends_on = (C=labels, d=labels)
    end
    trace = _trace(fit!(grouped, ys; max_iter=3, progress=false, rng=StableRNG(5)))
    @test all(isfinite, trace)
    sampled = _entry_model(; p=p)
    trace = _trace(
        fit!(sampled, ys; max_iter=3, num_samples=2, progress=false, rng=StableRNG(5))
    )
    @test all(isfinite, trace)
    return nothing
end

"""
Control (bridged) → hold → control (bridged, entry prior): the bridge out of state
1 lands in an infinite-horizon hold, the entry into state 3 leaves one and reads
its reference, and the guarded structural step fits the hold state jointly with
the control states. Derivatives agree by finite differences and the fit runs.
"""
function test_entry_with_hold()
    slds = _entry_model(; middle=:hold)
    @test slds.LDSs[2].state_model.mode === :hold
    @test_throws ArgumentError set_boundaries!(deepcopy(slds); entry_states=[2])
    @test_throws ArgumentError set_boundaries!(deepcopy(slds); bridge_states=[2])
    test_entry_smoother_derivatives(; middle=:hold)

    p, tsteps, ntrials = 4, 36, 6
    ys = hslds_data(p, tsteps, ntrials)
    slds = _entry_model(; p=p, middle=:hold)
    before = elbo(slds, ys)
    trace = _trace(
        fit!(slds, ys; max_iter=5, progress=false, rng=StableRNG(5), tied_params=[:A, :S])
    )
    after = elbo(slds, ys)
    @test all(isfinite, trace)
    @test after > before
    @test slds.A[2, 1] == 0 && slds.A[3, 1] == 0 && slds.A[3, 2] == 0
    hold = slds.LDSs[2].state_model
    @test hold.mode === :hold
    @test hold.A ≈ slds.LDSs[1].state_model.A
    @test hold.S ≈ slds.LDSs[3].state_model.S
    @test maximum(abs, eigvals(closed_loop_dynamics(hold))) < 1
    @test isposdef(Symmetric(slds.boundaries.entry[3].P))
    return nothing
end
