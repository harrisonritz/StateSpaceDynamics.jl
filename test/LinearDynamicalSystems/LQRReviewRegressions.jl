#=============================================================================
Regressions for defects found in an external review of the inverse-LQR and
switching inverse-LQR fits. Each test reproduces the failure it guards against
on the smallest model that showed it:

* aliased cost matrices optimized as separate parameters and overwritten on
  writeback, so a public `fit!` decreased its own objective;
* the terminal-conditioned switching M-step skipping the uniform-layout checks
  and moving a cost another state had frozen;
* `loglikelihood` scoring the parent model of a grouped fit instead of its
  variants;
* a whole-structure freeze still packing a singular known block as fitted;
* single-timestep trials failing with an index error inside a kernel;
* the switching score using a second-order Poisson expectation, overstating
  the ELBO of its own Gaussian `q`;
* `simulate_lqr` passing plant noise through the controller;
* the conditioned switching score being reported without its two halves.
=============================================================================#

"""A scalar-plant inverse-LQR model with a state-only Gaussian readout."""
function review_scalar_lqr(;
    terminal::Bool=false, flags=LQRFitFlags(), S::Real=0.1, Q::Real=0.2, iters::Int=4
)
    sm = LQRStateModel(
        reshape([0.9], 1, 1),
        reshape([Float64(S)], 1, 1),
        reshape([Float64(Q)], 1, 1),
        0.05 * Matrix{Float64}(I, 2, 2);
        terminal=terminal,
        Σf=reshape([0.1], 1, 1),
        fit_flags=flags,
        mstep_iters=iters,
    )
    return LinearDynamicalSystem(
        sm,
        GaussianObservationModel(reshape([1.0, 0.0], 1, 2), reshape([0.1], 1, 1), zeros(1)),
    )
end

const REVIEW_QC_ONLY = LQRFitFlags(;
    A=false, S=false, Qc=true, h=false, Bu=false, Gref=false, terminal=false
)
const REVIEW_FROZEN = LQRFitFlags(;
    A=false, S=false, Qc=false, h=false, Bu=false, Gref=false, terminal=false
)

"""
Aliased cost entries are refused, at construction and again at the M-step (the
`Qc` vector is mutable), and the supported way to reuse a cost — several
schedule entries pointing at one index — fits monotonically on the data that
exposed the defect.
"""
function test_lqr_review_qc_alias_rejected()
    Q = reshape([0.2], 1, 1)
    H = 26
    schedule = vcat(fill(1, 20), fill(2, 6))
    Σ = 0.05 * Matrix{Float64}(I, 2, 2)
    @test_throws ArgumentError LQRStateModel(ones(1, 1), zeros(1, 1), [Q, Q], Σ; schedule)

    lam = zeros(H)
    for t in (H - 1):-1:1
        lam[t] = lam[t + 1] + (schedule[t] == 1 ? 0.8 : 0.05)
    end
    y = vcat(ones(1, H), reshape(lam, 1, H))
    function build(Qc)
        sm = LQRStateModel(
            ones(1, 1),
            zeros(1, 1),
            Qc,
            copy(Σ);
            schedule,
            observe_costate=true,
            fit_flags=REVIEW_QC_ONLY,
            mstep_iters=50,
        )
        lds = LinearDynamicalSystem(
            sm,
            GaussianObservationModel(
                Matrix{Float64}(I, 2, 2), 1e-4 * Matrix{Float64}(I, 2, 2), zeros(2)
            ),
        )
        lds.fit_bool .= false
        lds.fit_bool[3] = true
        return lds
    end

    # Two separately fitted costs: the accepted objective never decreases.
    lds = build([copy(Q), copy(Q)])
    trace = fit!(lds, y; max_iter=3, progress=false)
    @test all(diff(trace) .>= -1e-8)
    @test elbo(lds, y) ≈ trace[end] rtol = 1e-10

    # An alias introduced after construction is caught at entry, before any
    # parameter moves, and again by the M-step context if it gets that far.
    lds = build([copy(Q), copy(Q)])
    lds.state_model.Qc[2] = lds.state_model.Qc[1]
    before = copy(lds.state_model.Qc[1])
    @test_throws ArgumentError fit!(lds, y; max_iter=2, progress=false)
    @test lds.state_model.Qc[1] == before
    @test_throws ArgumentError elbo(lds, y)
    hs = SSD._initialize_td_sufficient_statistics(Float64, lds, [H])
    @test_throws ArgumentError SSD._LQRMStepCtx(hs, lds.state_model, false)
    return nothing
end

"""
A frozen block is never packed as a fitted one: a known singular `S` survives a
fit that freezes the whole structural group, on both the unconditioned and the
terminal-conditioned M-step, and still refuses to be fitted when it is free.
"""
function test_lqr_review_whole_freeze_singular()
    y = randn(StableRNG(1), 1, 6)
    for terminal in (false, true)
        lds = review_scalar_lqr(; terminal=terminal, S=0.0)
        lds.fit_bool[3] = false
        Qc0 = copy(lds.state_model.Qc[1])
        trace = fit!(lds, y; max_iter=2, progress=false)
        @test all(isfinite, trace)
        @test lds.state_model.S == zeros(1, 1)
        @test lds.state_model.Qc[1] == Qc0
    end
    lds = review_scalar_lqr(; S=0.0)
    @test_throws ArgumentError fit!(lds, y; max_iter=2, progress=false)
    return nothing
end

"""Trials without a transition are refused with an `ArgumentError` at every
entry point, rather than an index error inside a kernel."""
function test_lqr_review_short_trials()
    for terminal in (false, true)
        lds = review_scalar_lqr(; terminal=terminal)
        @test_throws ArgumentError smooth(lds, zeros(1, 1))
        @test_throws ArgumentError elbo(lds, zeros(1, 1))
        @test_throws ArgumentError fit!(lds, zeros(1, 1); max_iter=2, progress=false)
        ragged = [randn(StableRNG(2), 1, 5), zeros(1, 1)]
        @test_throws ArgumentError elbo(lds, ragged)
    end
    sl = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[review_scalar_lqr()])
    @test_throws ArgumentError elbo(sl, zeros(1, 1))
    @test isfinite(elbo(review_scalar_lqr(), zeros(1, 2)))
    return nothing
end

"""
`loglikelihood` scores a grouped fit under its fitted variants, as `elbo` does:
the two agree without priors, and differ by exactly the grouped prior with one.
"""
function test_lqr_review_grouped_loglikelihood()
    rng = StableRNG(7)
    ys = [randn(rng, 3, 10) for _ in 1:4]
    labels = [1, 1, 2, 2]
    function build(; prior::Bool)
        sm = LQRStateModel(
            reshape([0.9], 1, 1),
            reshape([0.1], 1, 1),
            reshape([0.2], 1, 1),
            0.05 * Matrix{Float64}(I, 2, 2);
            terminal=true,
            Σf=reshape([0.1], 1, 1),
            Σ_prior=prior ? IWPrior(; Ψ=0.1 * Matrix{Float64}(I, 2, 2), ν=4.0) : nothing,
        )
        om = GaussianObservationModel(
            randn(StableRNG(3), 3, 2), Matrix(0.1I, 3, 3), zeros(3)
        )
        lds = LinearDynamicalSystem(sm, om)
        set_depends_on!(lds.obs_model, (C=labels, d=labels))
        fit!(lds, ys; max_iter=3, progress=false)
        return lds
    end

    lds = build(; prior=false)
    @test loglikelihood(lds, ys) ≈ elbo(lds, ys) rtol = 1e-10
    # An explicit label vector is the same grouping as the stored one.
    @test loglikelihood(lds, ys; depends_on=(C=labels, d=labels)) ≈ elbo(lds, ys) rtol =
        1e-10

    lds = build(; prior=true)
    prior = SSD.iw_logprior_term(lds.state_model.Σ, lds.state_model.Σ_prior)
    @test elbo(lds, ys) - loglikelihood(lds, ys) ≈ prior rtol = 1e-8
    return nothing
end

"""
`simulate_lqr` is causal stochastic LQR by default: plant noise lands after the
closed-loop step, so `Var(x₂ | x₁) = Σ_xx`. The legacy `:implicit` timing passes
it through `W = (I + S P₂)⁻¹`, giving `W² Σ_xx`.
"""
function test_lqr_review_simulate_causal_noise()
    sm = LQRStateModel(
        ones(1, 1),
        ones(1, 1),
        [zeros(1, 1), ones(1, 1)],
        Matrix{Float64}(I, 2, 2);
        schedule=[1, 2],
        terminal=true,
    )
    function draws(timing)
        return [
            simulate_lqr(StableRNG(i), sm, 2; x1=[0.0], noise_timing=timing)[1, 2] for
            i in 1:20_000
        ]
    end
    @test var(draws(:causal)) ≈ 1.0 rtol = 0.04
    @test var(draws(:implicit)) ≈ 0.25 rtol = 0.04
    # Without noise the two timings are the same deterministic rollout.
    z1 = simulate_lqr(StableRNG(1), sm, 2; x1=[0.3], process_noise=false)
    z2 = simulate_lqr(
        StableRNG(1), sm, 2; x1=[0.3], process_noise=false, noise_timing=:implicit
    )
    @test z1 == z2
    @test_throws ArgumentError simulate_lqr(sm, 2; noise_timing=:bogus)
    return nothing
end

"""
The terminal-conditioned switching M-step runs the same uniform-layout checks as
the unconditioned one, and they run at `fit!` entry, so a refused configuration
leaves every parameter — state, emission and chain — where it was.
"""
function test_slds_lqr_review_layout_checked_before_conditioning()
    y = randn(StableRNG(2), 1, 8)
    for terminal in (false, true)
        a = review_scalar_lqr(; terminal=terminal, flags=REVIEW_QC_ONLY)
        b = review_scalar_lqr(; terminal=terminal, flags=REVIEW_FROZEN, Q=0.7)
        for l in (a, b)
            l.fit_bool .= false
            l.fit_bool[3] = true
        end
        sl = SLDS(; A=[0.9 0.1; 0.1 0.9], πₖ=[0.5, 0.5], LDSs=[a, b])
        snapshot = deepcopy(sl)
        @test_throws ArgumentError fit!(
            sl, y; max_iter=2, progress=false, num_samples=2, rng=StableRNG(3)
        )
        @test sl.A == snapshot.A
        @test b.state_model.Qc[1] == [0.7;;]
        @test a.state_model.Qc[1] == snapshot.LDSs[1].state_model.Qc[1]
        @test a.obs_model.C == snapshot.LDSs[1].obs_model.C
    end

    # Under conditioning the initial state joins the joint problem, so the
    # states must agree on freezing it too.
    a = review_scalar_lqr(; terminal=true)
    b = review_scalar_lqr(; terminal=true, Q=0.7)
    b.fit_bool[1] = false
    sl = SLDS(; A=[0.9 0.1; 0.1 0.9], πₖ=[0.5, 0.5], LDSs=[a, b])
    @test_throws ArgumentError fit!(
        sl, y; max_iter=2, progress=false, num_samples=2, rng=StableRNG(3)
    )
    return nothing
end

"""
A Poisson emission's expectation under the Gaussian `q` is the exact lognormal
moment in the switching score, so a one-regime switching model scores what the
non-switching model does — which already used the exact moment.
"""
function test_slds_review_poisson_exact_moment()
    sm = LQRStateModel(
        reshape([0.9], 1, 1),
        reshape([0.1], 1, 1),
        reshape([0.2], 1, 1),
        0.5 * Matrix{Float64}(I, 2, 2);
        P0=2.0 * Matrix{Float64}(I, 2, 2),
    )
    lds = LinearDynamicalSystem(
        sm, PoissonObservationModel(reshape([1.5, 0.0], 1, 2), [0.0])
    )
    sl = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[deepcopy(lds)])
    for y in (zeros(1, 4), Float64[0 2 1 0 3])
        @test elbo(sl, y; smoothing_iters=30) ≈ elbo(lds, y) atol = 1e-8
    end

    # The correction itself: second-order term plus excess is the exact moment.
    om = PoissonObservationModel([0.7 -0.4; 0.2 0.9], [0.1, -0.3])
    x = [0.3 -0.2; 0.5 0.1]
    Σt = [0.6 0.1; 0.1 0.4]
    excess = SSD._emission_moment_excess(om, x, zeros(2, 2), 2, nothing, Σt)
    η = om.C * x[:, 2] .+ om.d
    v = [dot(om.C[i, :], Σt * om.C[i, :]) for i in 1:2]
    @test excess ≈ -sum(exp.(η) .* (exp.(v ./ 2) .- 1 .- v ./ 2)) rtol = 1e-12
    @test excess < 0
    @test SSD._emission_moment_excess(
        GaussianObservationModel(om.C, Matrix(0.1I, 2, 2), om.d), x, x, 2, nothing, Σt
    ) == 0
    return nothing
end

"""
Exact log densities of a small switching inverse-LQR model by enumerating all
`K^T` regime paths: the joint `log p(y, terminal = 0)`, the normalizer
`log p(terminal = 0)`, and their difference. Dense and exponential — an oracle
for a handful of steps only. The regime of the transition into `t` is `s_t`.
"""
function review_exact_slqr(sl, y)
    lse(v) = maximum(v) + log(sum(exp.(v .- maximum(v))))
    K, H = length(sl.LDSs), size(y, 2)
    d, p = sl.LDSs[1].latent_dim, size(y, 1)
    n = d ÷ 2
    ix(t) = (d * (t - 1) + 1):(d * t)
    nums, dens = Float64[], Float64[]
    for path in Iterators.product(ntuple(_ -> 1:K, H)...)
        lp = log(sl.πₖ[path[1]]) + sum(log(sl.A[path[t - 1], path[t]]) for t in 2:H)
        m, V = zeros(d * H), zeros(d * H, d * H)
        sm1 = sl.LDSs[path[1]].state_model
        m[ix(1)] .= sm1.x0
        V[ix(1), ix(1)] .= sm1.P0
        for t in 2:H
            sm = sl.LDSs[path[t]].state_model
            M = sm.cache.M[1]
            m[ix(t)] .= M * m[ix(t - 1)] + sm.cache.bfwd
            for j in 1:(t - 1)
                V[ix(t), ix(j)] .= M * V[ix(t - 1), ix(j)]
                V[ix(j), ix(t)] .= V[ix(t), ix(j)]'
            end
            V[ix(t), ix(t)] .= M * V[ix(t - 1), ix(t - 1)] * M' + Matrix(sm.cache.Qfwd)
        end
        L, R, b = zeros(p * H + n, d * H), zeros(p * H + n, p * H + n), zeros(p * H + n)
        for t in 1:H
            om = sl.LDSs[path[t]].obs_model
            iy = (p * (t - 1) + 1):(p * t)
            L[iy, ix(t)] .= om.C
            R[iy, iy] .= om.R
            b[iy] .= om.d
        end
        sm = sl.LDSs[path[H]].state_model
        it = (p * H + 1):(p * H + n)
        L[it, ix(H)] .= sm.cache.Lf[1]
        R[it, it] .= sm.Σf
        b[it] .= -sm.hf
        μ = L * m + b
        Ω = Symmetric(L * V * L' + R)
        push!(nums, lp + logpdf(MvNormal(μ, Ω), vcat(vec(y), zeros(n))))
        push!(dens, lp + logpdf(MvNormal(μ[it], Symmetric(Matrix(Ω)[it, it])), zeros(n)))
    end
    joint, logz = lse(nums), lse(dens)
    return (; joint, logz, conditional=joint - logz)
end

"""
The conditioned switching score is reported with its two halves. Against exact
enumeration, the joint half and the normalizer are each genuine lower bounds,
while their difference — the reported `elbo` — is not a bound on anything, so it
is the halves that carry a guarantee.
"""
function test_slds_lqr_review_conditional_score_halves()
    ls = map([0.02, 2.0]) do q
        sm = LQRStateModel(
            reshape([0.9], 1, 1),
            reshape([0.4], 1, 1),
            reshape([q], 1, 1),
            0.1 * Matrix{Float64}(I, 2, 2);
            terminal=true,
            observe_costate=true,
            Σf=reshape([0.05], 1, 1),
            P0=0.25 * Matrix{Float64}(I, 2, 2),
        )
        LinearDynamicalSystem(
            sm,
            GaussianObservationModel(
                Matrix{Float64}(I, 2, 2), 0.001 * Matrix{Float64}(I, 2, 2), zeros(2)
            ),
        )
    end
    sl = SLDS(; A=[0.8 0.2; 0.2 0.8], πₖ=[0.5, 0.5], LDSs=ls)
    y = zeros(2, 5)
    out = smooth(sl, y; smoothing_iters=100)
    @test out.elbo_joint ≈ out.elbo + out.terminal_logz rtol = 1e-12
    @test out.log_prior == 0
    exact = review_exact_slqr(sl, y)
    @test out.elbo_joint <= exact.joint + 1e-8
    @test out.terminal_logz <= exact.logz + 1e-8
    @test isfinite(out.elbo)

    # One regime: nothing to approximate, so all three are exact.
    sl1 = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[deepcopy(ls[1])])
    out1 = smooth(sl1, y; smoothing_iters=50)
    exact1 = review_exact_slqr(sl1, y)
    @test out1.elbo_joint ≈ exact1.joint atol = 1e-6
    @test out1.terminal_logz ≈ exact1.logz atol = 1e-6
    @test out1.elbo ≈ exact1.conditional atol = 1e-6
    return nothing
end
