#=============================================================================
Switching inverse-LQR: `SLDS` whose discrete states are LQR models.

The SLDS uses one *shared* continuous latent path — `joint_loglikelihood!` forms
each timestep as the responsibility-weighted mixture `Σₖ wₖₜ ℓₜ⁽ᵏ⁾(x)` — so every
discrete state reads and writes the same `2n`-dimensional `z = [x; λ]`, and the
switching is over which cost (and plant) generated the transition.

The anchors, in order of how much they pin down:

* a `K = 1` switching model must reproduce the ungrouped inverse-LQR fit, since
  γ ≡ 1 makes the weighted statistics the plain ones and the discrete layer
  contributes nothing — this is the test that caught the costate readout leaking
  through the switching emission M-step;
* an all-`:free` switching model must reproduce a Gaussian `SLDS` exactly;
* the ELBO must be monotone under the generalized M-step, whose Jacobian term is
  scaled by the *effective* count `n̄ₖ = Σ γₖ(t)` rather than a timestep count.
=============================================================================#

"""One inverse-LQR discrete state: a shared mild plant, its own cost."""
function hslds_state(Qc; p::Int=4, C=nothing, terminal::Bool=false, seed::Int=11)
    A = [0.96 0.07; -0.05 0.93]
    S = [0.06 0.01; 0.01 0.05]
    Σ = Matrix(0.05I, 4, 4)
    Cm = C === nothing ? randn(StableRNG(seed), p, 4) : copy(C)
    Cm[:, 3:4] .= 0
    sm = LQRStateModel(
        copy(A),
        copy(S),
        copy(Qc),
        copy(Σ);
        terminal=terminal,
        Σf=Matrix(0.02I, 2, 2),
        P0=Matrix(0.3I, 4, 4),
    )
    return LinearDynamicalSystem(
        sm, GaussianObservationModel(Cm, Matrix(0.1I, p, p), zeros(p))
    )
end

"""A `K`-state switching inverse-LQR sharing one emission."""
function hslds_model(Qcs; p::Int=4, stay::Float64=0.95, terminal::Bool=false)
    K = length(Qcs)
    C = randn(StableRNG(11), p, 4)
    P = fill((1 - stay) / (K - 1), K, K)
    for k in 1:K
        P[k, k] = stay
    end
    return SLDS(;
        A=P,
        πₖ=fill(1 / K, K),
        LDSs=[hslds_state(Qcs[k]; p=p, C=C, terminal=terminal) for k in 1:K],
    )
end

function hslds_data(p, tsteps, ntrials)
    return [randn(StableRNG(800 + i), p, tsteps) .* 0.5 for i in 1:ntrials]
end

_trace(e) = e isa Tuple ? e[1] : e

"""With one discrete state the responsibilities are identically one, the HMM
contributes nothing, and the weighted statistics are the plain ones — so this
must reproduce the ungrouped fit, parameter for parameter. Anything that leaks
into the switching path but not the plain one shows up here."""
function test_slds_lqr_matches_lds()
    p, tsteps, ntrials = 4, 35, 5
    ys = hslds_data(p, tsteps, ntrials)
    Qc = [0.25 0.04; 0.04 0.18]
    function fit_both(iters)
        lds = hslds_state(Qc; p=p)
        slds = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[hslds_state(Qc; p=p)])
        e_lds = _trace(fit!(lds, ys; max_iter=iters, tol=1e-14))
        e_slds = _trace(fit!(slds, ys; max_iter=iters, progress=false, rng=StableRNG(7)))
        return lds, slds, e_lds, e_slds
    end

    #=
    One M-step (the final iteration is scored, not updated): the sharp check.
    Both paths start from identical statistics, so a leak is an O(1)
    difference here, while agreement is limited only by the structural
    L-BFGS stopping on its own tolerance — measured at ~1e-7.
    =#
    lds, slds, e_lds, e_slds = fit_both(2)
    a, b = lds.state_model, slds.LDSs[1].state_model
    @test maximum(abs, a.Qc[1] .- Qc) > 1e-2      # the M-step moved something
    @test maximum(abs, e_lds .- e_slds) < 1e-5
    for key in (:A, :S, :Σ, :x0, :P0, :h)
        @test maximum(abs, getproperty(a, key) .- getproperty(b, key)) < 1e-6
    end
    @test maximum(abs, a.Qc[1] .- b.Qc[1]) < 1e-6
    for key in (:C, :R, :d)
        @test maximum(
            abs, getproperty(lds.obs_model, key) .- getproperty(slds.LDSs[1].obs_model, key)
        ) < 1e-6
    end

    #=
    Ten iterations: from the second M-step on, the two structural optimizers
    stop at different points of the objective's flattest direction — the cost
    scale, along which `Qc` drifts by ~6% while the loop gain `S P` agrees to
    ~1e-5 — and the traces separate by ~1e-2 nats in a fit gaining ~200. So
    what is compared is what the data identify: the closed loop and the loop
    gain, the noise, the emission and the bound, not the raw cost.
    =#
    lds, slds, e_lds, e_slds = fit_both(10)
    a, b = lds.state_model, slds.LDSs[1].state_model
    @test length(e_lds) == length(e_slds)
    @test maximum(abs, e_lds .- e_slds) < 5e-2
    @test maximum(abs, closed_loop_dynamics(a) .- closed_loop_dynamics(b)) < 1e-3
    @test maximum(abs, a.S * riccati_solution(a) .- b.S * riccati_solution(b)) < 1e-3
    @test maximum(abs, a.A .- b.A) < 1e-3
    @test maximum(abs, a.Σ .- b.Σ) < 1e-3
    @test maximum(abs, lds.obs_model.C .- slds.LDSs[1].obs_model.C) < 1e-3

    # `observe_costate` is off, so the emission may never read the costate half.
    # Without the mask on the switching side these columns fill in silently.
    @test all(iszero, slds.LDSs[1].obs_model.C[:, 3:4])
    return nothing
end

function test_slds_lqr_fixed_costate_sigma()
    slds = hslds_model([[0.25 0.04; 0.04 0.18], [0.9 0.0; 0.0 0.7]])
    v = 1e-3
    for lds in slds.LDSs
        sm = lds.state_model
        sm.fixed_costate_sigma = v
        sm.Σ[3:4, 3:4] .= v .* I(2)
        refresh!(sm)
    end
    ys = hslds_data(4, 12, 2)
    fit!(slds, ys; max_iter=1, progress=false, rng=StableRNG(7))
    for lds in slds.LDSs
        Σ = lds.state_model.Σ
        @test Σ[3:4, 3:4] ≈ v .* I(2)
        @test Σ[1:2, 3:4] == zeros(2, 2)
        @test Σ[3:4, 1:2] == zeros(2, 2)
        @test isposdef(Symmetric(Σ[1:2, 1:2]))
    end
    return nothing
end

"""The fit must improve the bound, and the constrained parameterization must
survive doing so.

Measured with `elbo(slds, y)` rather than the `fit!` trace. The trace is the
bound at whatever `q` the inner variational alternation reached that iteration —
`smoothing_iters` of it, one by default — and for an inverse-LQR state that inner
problem is much harder than usual, because a symplectic transition's forward flow
is unstable and the shared `q(x)` is correspondingly slow to converge. Measured
here the trace sits well below `elbo` at the same parameters (~170 nats on the
tutorial's model, against ~0.5 for a Gaussian `SLDS`), and it can dip from one
iteration to the next while the parameters are still improving. So the trace is
not a monotonicity diagnostic for a switching LQR; the converged bound is."""
function test_slds_lqr_monotone()
    p = 4
    slds = hslds_model([[0.25 0.04; 0.04 0.18], [0.9 0.0; 0.0 0.7]]; p=p)
    ys = hslds_data(p, 40, 5)
    before = elbo(slds, ys)
    elbos = _trace(fit!(slds, ys; max_iter=15, progress=false, rng=StableRNG(7)))
    after = elbo(slds, ys)

    @test length(elbos) > 1
    @test all(isfinite, elbos)
    @test after > before                 # the fit improved the converged bound
    @test elbos[end] > elbos[1]          # and the trace's own endpoints agree
    @test after >= elbos[end] - 1e-6     # a converged `q` is never a worse bound

    # The constrained parameterization survives the switching M-step.
    for lds in slds.LDSs
        @test symplectic_defect(lds.state_model) < 1e-10
        @test issymmetric(lds.state_model.Qc[1])
        @test isposdef(lds.state_model.Σ)
    end
    # Responsibilities really did separate, or the test above proves little:
    # a monotone ELBO under γ ≡ 1/K would not exercise the effective-count path.
    γ = smooth(slds, ys).γ          # one K × T matrix per trial
    @test maximum(maximum(abs, g[1, :] .- 0.5) for g in γ) > 0.1
    return nothing
end

"""An all-`:free` switching model is a Gaussian `SLDS` wearing the LQR
type, which is the whole point of `:free` — so it must match one exactly."""
function test_slds_free_matches_gaussian_slds()
    p, d, tsteps, ntrials = 3, 4, 30, 4
    rng = StableRNG(606)
    Ms = [0.85 * Matrix(1.0I, d, d) + 0.05 * randn(rng, d, d) for _ in 1:2]
    Σs = [Matrix(0.15I, d, d) for _ in 1:2]
    C = randn(rng, p, d)
    P = [0.9 0.1; 0.1 0.9]

    function mkG(k)
        return LinearDynamicalSystem(
            GaussianStateModel(;
                A=copy(Ms[k]),
                Q=copy(Σs[k]),
                b=zeros(d),
                x0=zeros(d),
                P0=Matrix(0.5I, d, d),
                B=zeros(d, 0),
                Q_prior=nothing,
                P0_prior=nothing,
                AB_prior=nothing,
                x0_prior=nothing,
            ),
            GaussianObservationModel(copy(C), Matrix(0.2I, p, p), zeros(p)),
        )
    end
    function mkF(k)
        return LinearDynamicalSystem(
            free_state_model(copy(Ms[k]), copy(Σs[k]); P0=Matrix(0.5I, d, d)),
            GaussianObservationModel(copy(C), Matrix(0.2I, p, p), zeros(p)),
        )
    end
    sG = SLDS(; A=copy(P), πₖ=[0.5, 0.5], LDSs=[mkG(1), mkG(2)])
    sF = SLDS(; A=copy(P), πₖ=[0.5, 0.5], LDSs=[mkF(1), mkF(2)])
    ys = [randn(StableRNG(910 + i), p, tsteps) .* 0.5 for i in 1:ntrials]

    eG = _trace(fit!(sG, ys; max_iter=10, progress=false, rng=StableRNG(7)))
    eF = _trace(fit!(sF, ys; max_iter=10, progress=false, rng=StableRNG(7)))
    @test length(eG) == length(eF)
    @test maximum(abs, eG .- eF) < 1e-7
    for k in 1:2
        @test maximum(abs, sG.LDSs[k].state_model.A .- sF.LDSs[k].state_model.Mfree) < 1e-8
        @test maximum(abs, sG.LDSs[k].state_model.Q .- sF.LDSs[k].state_model.Σ) < 1e-8
    end
    @test maximum(abs, sG.A .- sF.A) < 1e-8
    return nothing
end

"""An inverse-LQR state's `Σ` is shared by `:noise` and its cost by `:structure`
(or `:Qc`); each tie makes that prior one term instead of one per state, as the
constrained M-step fits it. Two identical states with both tied are one model and
score as it does; the initial state's prior counts once whatever is tied."""
function test_slds_lqr_tied_prior_counted_once()
    function with_priors()
        lds = hslds_state([0.25 0.04; 0.04 0.18])
        sm = lds.state_model
        sm.Σ_prior = IWPrior(; Ψ=Matrix(0.1I, 4, 4), ν=8.0)
        sm.Qc_prior = IWPrior(; Ψ=Matrix(0.2I, 2, 2), ν=5.0)
        sm.P0_prior = IWPrior(; Ψ=Matrix(0.3I, 4, 4), ν=7.0)
        return lds
    end
    y = hslds_data(4, 35, 5)
    lds = with_priors()
    slds = SLDS(; A=[0.9 0.1; 0.2 0.8], πₖ=[0.5, 0.5], LDSs=[with_priors(), with_priors()])
    sm = lds.state_model
    σ = SSD.iw_logprior_term(Matrix(sm.Σ), sm.Σ_prior)
    qc = SSD._lqr_structural_logprior(sm) - σ
    e = elbo(lds, y)
    @test elbo(slds, y; tied_params=[:structure, :noise]) ≈ e rtol = 1e-10
    @test elbo(slds, y; tied_params=[:noise]) - e ≈ qc rtol = 1e-8
    @test elbo(slds, y; tied_params=[:structure]) - e ≈ σ rtol = 1e-8
    @test elbo(slds, y; tied_params=[:Qc]) - e ≈ σ rtol = 1e-8
    @test elbo(slds, y) - e ≈ σ + qc rtol = 1e-8
    return nothing
end

"""Mixing an unconstrained state with an inverse-LQR one in a single switching
model — the configuration `:free` mode exists for. Each keeps its own kind of
update: the LQR state stays symplectic, the free one does not have to."""
function test_slds_mixed_free_and_lqr()
    p, tsteps, ntrials = 4, 35, 5
    C = randn(StableRNG(11), p, 4)
    C[:, 3:4] .= 0
    lqr = hslds_state([0.25 0.04; 0.04 0.18]; p=p, C=C)
    free = LinearDynamicalSystem(
        free_state_model(
            0.9 * Matrix(1.0I, 4, 4), Matrix(0.1I, 4, 4); P0=Matrix(0.3I, 4, 4)
        ),
        GaussianObservationModel(copy(C), Matrix(0.1I, p, p), zeros(p)),
    )
    slds = SLDS(; A=[0.9 0.1; 0.1 0.9], πₖ=[0.5, 0.5], LDSs=[lqr, free])
    @test slds.LDSs[1].state_model.mode === :lqr
    @test slds.LDSs[2].state_model.mode === :free
    @test slds.LDSs[1].latent_dim == slds.LDSs[2].latent_dim   # one shared latent path

    #=
    Monotone only up to the E-step's Monte Carlo noise, which one sample leaves
    at a few tenths of a nat: enough to show as a decrease once the fit nears
    its optimum (it does on half the seeds tried). Four samples separate what
    the M-step does from that noise.
    =#
    ys = hslds_data(p, tsteps, ntrials)
    elbos = _trace(
        fit!(slds, ys; max_iter=12, progress=false, rng=StableRNG(7), num_samples=4)
    )
    @test minimum(diff(elbos)) > -1e-6
    @test all(isfinite, elbos)
    # The free state reads what the LQR state reads, which is not its costate.
    @test !slds.LDSs[2].state_model.observe_costate
    @test iszero(slds.LDSs[2].obs_model.C[:, 3:4])

    # Each state kept its own kind of parameterization.
    @test symplectic_defect(slds.LDSs[1].state_model) < 1e-10
    @test isposdef(slds.LDSs[2].state_model.Σ)
    @test size(slds.LDSs[2].state_model.Mfree) == (4, 4)
    return nothing
end

"""`tied` shares a parameter version across discrete states.

`:structure` shares the whole joint block `(A, S, Qc, h, Bu, Gref)` and `:noise`
shares `Σ`. Individual blocks may also be named, which is the configuration a
switching inverse-LQR model is usually for: `[:A, :S]` is one plant with a cost
per discrete state. A partial tie cannot be one constrained optimization here —
each parameter *version* owns a full copy of every block — so it runs as two
alternating passes, shared-free then per-state-free. Each accepts only an
improvement, so the composition still cannot decrease the bound.

Shared parameters must come out bit-identical, not merely close: a tie is one
fitted value copied out, and `≈` would pass on two independent fits that happened
to land nearby."""
function test_slds_lqr_tied()
    p, tsteps, ntrials = 4, 35, 5
    ys = hslds_data(p, tsteps, ntrials)
    costs = [[0.25 0.04; 0.04 0.18], [0.9 0.0; 0.0 0.7]]
    function fit_tied(names)
        slds = hslds_model(costs; p=p)
        before = elbo(slds, ys)
        fit!(slds, ys; max_iter=10, progress=false, rng=StableRNG(7), tied_params=names)
        return (slds, before, elbo(slds, ys))
    end

    # Whole-block tie.
    slds, before, after = fit_tied([:structure])
    a, b = slds.LDSs[1].state_model, slds.LDSs[2].state_model
    @test a.A == b.A
    @test a.S == b.S
    @test a.Qc[1] == b.Qc[1]
    @test a.Σ != b.Σ                     # noise untied
    @test after > before

    # The headline partial tie: one plant, a cost per discrete state.
    slds, before, after = fit_tied([:A, :S])
    a, b = slds.LDSs[1].state_model, slds.LDSs[2].state_model
    @test a.A == b.A
    @test a.S == b.S
    @test a.Qc[1] != b.Qc[1]             # the cost is what switches
    @test after > before
    @test symplectic_defect(a) < 1e-10
    @test symplectic_defect(b) < 1e-10

    # A single named block shares exactly that block.
    slds, _, _ = fit_tied([:A])
    a, b = slds.LDSs[1].state_model, slds.LDSs[2].state_model
    @test a.A == b.A
    @test a.S != b.S
    @test a.Qc[1] != b.Qc[1]

    # Tying the noise as well.
    slds, _, _ = fit_tied([:structure, :noise])
    @test slds.LDSs[1].state_model.Σ == slds.LDSs[2].state_model.Σ

    # Untied, everything is free to differ — otherwise the tests above are vacuous.
    slds, _, _ = fit_tied(Symbol[])
    a, b = slds.LDSs[1].state_model, slds.LDSs[2].state_model
    @test a.A != b.A
    @test a.Qc[1] != b.Qc[1]

    #=
    A partial tie must leave the per-state blocks alone, not merely refit them
    afterwards. With `Qc` frozen there is no second pass to hide a clobber, so a
    shared pass that broadcast the whole structural block would show up here as
    both states ending on state 1's cost.
    =#
    frozen = hslds_model(costs; p=p)
    for lds in frozen.LDSs
        lds.state_model.fit_flags = LQRFitFlags(; Qc=false)
    end
    q1 = copy(frozen.LDSs[1].state_model.Qc[1])
    q2 = copy(frozen.LDSs[2].state_model.Qc[1])
    @test q1 != q2
    fit!(frozen, ys; max_iter=6, progress=false, rng=StableRNG(7), tied_params=[:A, :S])
    @test frozen.LDSs[1].state_model.Qc[1] == q1      # frozen exactly
    @test frozen.LDSs[2].state_model.Qc[1] == q2      # and *not* overwritten by q1
    @test frozen.LDSs[1].state_model.A == frozen.LDSs[2].state_model.A

    #=
    A frozen block is never shared, whatever the tie asks for. Freezing means
    "keep your own value", so under `:structure` — which does ask to share the
    cost — two states with different frozen costs must each keep theirs, and the
    objective must score each against its own rather than against state 1's.
    =#
    frozen_struct = hslds_model(costs; p=p)
    for lds in frozen_struct.LDSs
        lds.state_model.fit_flags = LQRFitFlags(; Qc=false)
    end
    fq1 = copy(frozen_struct.LDSs[1].state_model.Qc[1])
    fq2 = copy(frozen_struct.LDSs[2].state_model.Qc[1])
    @test fq1 != fq2
    before_fs = elbo(frozen_struct, ys)
    fit!(
        frozen_struct,
        ys;
        max_iter=6,
        progress=false,
        rng=StableRNG(7),
        tied_params=[:structure],
    )
    @test frozen_struct.LDSs[1].state_model.Qc[1] == fq1
    @test frozen_struct.LDSs[2].state_model.Qc[1] == fq2
    @test frozen_struct.LDSs[1].state_model.A == frozen_struct.LDSs[2].state_model.A
    @test elbo(frozen_struct, ys) > before_fs

    #=
    Every inverse-LQR state is packed into one parameter layout, which is read
    off the first model's `fit_flags`. States that freeze different blocks would
    have one state's freezes silently applied to the other, so the disagreement
    is refused instead.
    =#
    mixed_flags = hslds_model(costs; p=p)
    mixed_flags.LDSs[2].state_model.fit_flags = LQRFitFlags(; Gref=false)
    @test_throws ArgumentError fit!(
        mixed_flags, ys; max_iter=2, progress=false, rng=StableRNG(7), tied_params=[:A, :S]
    )

    # A name this model has no parameter for is still rejected.
    bad = hslds_model(costs; p=p)
    @test_throws ArgumentError fit!(
        bad, ys; max_iter=2, progress=false, rng=StableRNG(7), tied_params=[:Q]
    )
    @test_throws ArgumentError fit!(
        bad, ys; max_iter=2, progress=false, rng=StableRNG(7), tied_params=[:nonsense]
    )
    return nothing
end

"""The soft terminal condition is an extra factor at `t = T`. Under switching it
is weighted by that state's responsibility at `T` like every other factor —
`joint_loglikelihood!` gets it through the dispatched `state_loglikelihood!`, but
the curvature paths build their state blocks from flat templates that have no
slot for it, so it has to be added there by hand."""
function test_slds_lqr_terminal()
    p, tsteps, ntrials = 4, 30, 5
    slds = hslds_model([[0.25 0.04; 0.04 0.18], [0.9 0.0; 0.0 0.7]]; p=p, terminal=true)
    @test all(lds.state_model.terminal for lds in slds.LDSs)
    #=
    On the joint objective, where the M-step is a genuine minorant and the trace
    is therefore monotone. The conditional default divides by a variational
    normalizer, which is a surrogate rather than a majorant and may dip; that is
    `test_slds_lqr_terminal_conditioning`'s business.
    =#
    for lds in slds.LDSs
        lds.state_model.condition_terminal = false
    end

    ys = hslds_data(p, tsteps, ntrials)
    elbos = _trace(fit!(slds, ys; max_iter=10, progress=false, rng=StableRNG(7)))
    @test minimum(diff(elbos)) > -1e-6
    @test all(isfinite, elbos)

    res = smooth(slds, ys)
    @test all(all(isfinite, x) for x in res.x)

    # The terminal factor pins λ_T toward Q_f x_T, so the endpoint residual is
    # small relative to the costate's own scale — it is a soft constraint that
    # is actually doing something, not an inert extra factor.
    sm = slds.LDSs[1].state_model
    xT = res.x[1][:, end]
    resid = xT[3:4] .- sm.Qc[end] * xT[1:2] .- sm.hf
    @test norm(resid) < 10 * norm(xT[3:4]) + 1e-6
    return nothing
end

"""Terminal conditioning through the switching path.

`log p(terminal = 0)` has no exact form for a switching model, so it is
estimated by running the E-step on a zero-loading copy. `K = 1` is where that
estimate has nothing to approximate — the discrete layer is degenerate and the
continuous smoother is exact — so it is the case that can check the variational
route against the closed form the non-switching model uses.
"""
function test_slds_lqr_terminal_conditioning()
    p, tsteps, ntrials = 4, 20, 4
    ys = hslds_data(p, tsteps, ntrials)
    Qc = [0.25 0.04; 0.04 0.18]

    lds = hslds_state(Qc; p=p, terminal=true)
    slds = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[hslds_state(Qc; p=p, terminal=true)])
    @test SSD._slds_condition_terminal(slds)

    exact = ntrials * SSD._lqr_terminal_logz(lds.state_model, zeros(0, tsteps))
    @test terminal_logz(slds, ys) ≈ exact rtol = 1e-8
    # And the conditional score itself agrees with the non-switching one.
    @test elbo(slds, ys) ≈ elbo(lds, ys) atol = 1e-6
    @test smooth(slds, ys).terminal_logz ≈ exact rtol = 1e-8

    # Joint and conditional differ by exactly that normalizer.
    joint = SSD.terminal_logz(slds, ys)
    conditional = elbo(slds, ys)
    for member in slds.LDSs
        member.state_model.condition_terminal = false
    end
    @test !SSD._slds_condition_terminal(slds)
    @test elbo(slds, ys) ≈ conditional + joint atol = 1e-6
    @test terminal_logz(slds, ys) == 0
    for member in slds.LDSs
        member.state_model.condition_terminal = true
    end

    #=
    The costate gauge moves the joint score by `-N n log|c|` and leaves the
    conditional one alone. The emission here reads no costate, so nothing else
    has to be rescaled alongside it.
    =#
    before = elbo(slds, ys)
    rescale_costate!(slds.LDSs[1].state_model, 2.5)
    @test elbo(slds, ys) ≈ before atol = 1e-6

    # States carrying a terminal factor must agree on whether it is conditioned.
    mixed = hslds_model([Qc, [0.8 0.0; 0.0 0.6]]; p=p, terminal=true)
    mixed.LDSs[2].state_model.condition_terminal = false
    @test_throws ArgumentError SSD._slds_condition_terminal(mixed)

    # K = 2 fits on the conditional objective and improves on it.
    two = hslds_model([Qc, [0.8 0.0; 0.0 0.6]]; p=p, terminal=true)
    els = _trace(fit!(two, ys; max_iter=8, progress=false, rng=StableRNG(7)))
    @test all(isfinite, els)
    @test els[end] > els[1]
    #=
    Not `> -1e-6` as on the joint objective. Both halves of the M-step only
    accept a point that improves the scored objective — the discrete chain is
    checked against `log Z` too (`test_slds_lqr_terminal_chain_step`) — but the
    E-step is Monte Carlo. What the fit may not do is lose ground comparable to
    the progress it makes.
    =#
    @test minimum(diff(els)) > -0.05 * (els[end] - els[1])
    @test terminal_logz(two, ys) < 0
    #=
    The regression this fixture caught: the fixed-posterior surrogate is
    unbounded below wherever the probe's scatter exceeds the data's, and an
    M-step that trusted it drove a state's `Σ` singular within three iterations.
    =#
    for lds in two.LDSs
        @test minimum(eigvals(Symmetric(lds.state_model.Σ))) > 1e-3
    end
    return nothing
end

"""Under terminal conditioning the discrete chain moves `log Ẑ` as well as the
data half, so its update is checked against the conditioned score
`g(A, π) = Σ N log A + Σ n log π − log Ẑ`, the posterior held fixed. The step
never lowers `g`, keeps the chain stochastic, and leaves the probe smoothed at
the chain it kept, which is what lets the state M-step reuse it. The random
posteriors are weak enough that the probe's pull on the chain matters: two keep
the Baum–Welch proposal, and one needs the score's own ascent step."""
function test_slds_lqr_terminal_chain_step()
    p, tsteps, ntrials = 4, 20, 4
    ys = hslds_data(p, tsteps, ntrials)
    Qc = [0.25 0.04; 0.04 0.18]
    HMMs = SSD.HMMs
    proposals = Symbol[]
    for seed in 1:3
        two = hslds_model([Qc, [0.8 0.0; 0.0 0.6]]; p=p, terminal=true)
        data = SSD.Data(two.LDSs[1], ys)
        seq_ends = cumsum(data.tsteps)
        total = last(seq_ends)
        dl = SSD.SLDSDiscreteLayer(two.A, two.πₖ, 0.3 .* randn(StableRNG(seed), 2, total))
        fb = SSD._make_slds_fb_storage(dl, seq_ends)
        HMMs.forward_backward!(
            fb,
            dl,
            collect(1:total),
            fill(nothing, total);
            seq_ends=seq_ends,
            transition_marginals=true,
        )
        N, n = SSD._slds_chain_counts(fb, seq_ends, 2, Float64)
        function g(m)
            return sum(N .* log.(m.A .+ 1e-12)) + sum(n .* log.(m.πₖ .+ 1e-12)) -
                   terminal_logz(m, ys)
        end
        before = g(two)
        A_bw = N ./ sum(N; dims=2)
        probe = SSD._slqr_terminal_probe(two, data.ux)
        current = SSD._slqr_chain_mstep!(two, dl, fb, collect(1:total), seq_ends, probe)
        @test g(two) >= before - 1e-9
        @test all(≈(1), sum(two.A; dims=2)) && sum(two.πₖ) ≈ 1
        @test all(>=(0), two.A) && all(>=(0), two.πₖ)
        current && @test probe.logz ≈ terminal_logz(two, ys) rtol = 1e-10
        kept = two.A ≈ A_bw ? :baum_welch : :gradient
        push!(proposals, current ? kept : :none)
    end
    @test proposals == [:baum_welch, :gradient, :baum_welch]
    return nothing
end

"""Re-smoothing the switching normalizer's probe after its parameters move must
give what a probe built fresh at those parameters gives.

The probe's workspace pool caches each member's smoother constants. A probe
smoothed once, at the parameters it was built with, never notices; the M-step's
acceptance check re-smooths it at every candidate, and a stale pool there
smoothed under the old parameters while scoring under the new — a score whose
slope disagreed with the exact normalizer's everywhere but at the start.
"""
function test_slds_lqr_probe_resmoothing()
    p, tsteps, ntrials = 4, 20, 4
    ys = hslds_data(p, tsteps, ntrials)
    Qc = [0.25 0.04; 0.04 0.18]
    single = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[hslds_state(Qc; p=p, terminal=true)])
    switching = hslds_model([Qc, [0.8 0.0; 0.0 0.6]]; p=p, terminal=true)
    for slds in (single, switching)
        data = SSD.Data(slds.LDSs[1], ys)
        probe = SSD._slqr_terminal_probe(slds, data.ux)
        SSD._slqr_sync_probe!(probe, slds)
        SSD._slqr_probe_estep!(probe)
        backend = SSD._SLQRNormalizer(probe)
        start = probe.logz

        moved = deepcopy(slds)
        for lds in moved.LDSs
            sm = lds.state_model
            sm.Σ .*= 1.3
            sm.Qc[1] .*= 0.8
            refresh!(sm)
        end
        sms = [lds.state_model for lds in moved.LDSs]
        reused = SSD._terminal_score_logz(backend, sms)
        @test abs(reused - start) > 0.1         # the move is not a no-op ...
        @test reused ≈ terminal_logz(moved, ys) rtol = 1e-10   # ... and is seen
        if length(sms) == 1
            exact = ntrials * SSD._lqr_terminal_logz(sms[1], zeros(0, tsteps))
            @test reused ≈ exact rtol = 1e-8
        end
        # And back: the reused probe returns to exactly where it started.
        back = SSD._terminal_score_logz(backend, [lds.state_model for lds in slds.LDSs])
        @test back ≈ start rtol = 1e-12
    end
    return nothing
end

"""The switching surrogate the M-step descends has the scored objective's
gradient at the point it was built.

The surrogate holds the probe's posterior fixed, and at a stationary posterior
that costs nothing to first order (Danskin): its gradient is the gradient of the
freshly re-smoothed score. That is what lets the acceptance step's
steepest-descent fallback promise an improvement, so it is checked here against
central differences of the score itself, for two discrete states. Scoring
re-smooths the probe, so the check also confirms that doing so leaves the
surrogate exactly as it was.
"""
function test_slds_lqr_conditional_score_gradient()
    p, tsteps, ntrials = 4, 20, 4
    ys = hslds_data(p, tsteps, ntrials)
    slds = hslds_model([[0.25 0.04; 0.04 0.18], [0.8 0.0; 0.0 0.6]]; p=p, terminal=true)
    K = length(slds.LDSs)
    data = SSD.Data(slds.LDSs[1], ys)
    # Any valid per-state statistics serve as the data side: the identity under
    # test is about the normalizer, which the data side does not touch.
    sufs = [first(lqr_estep_stats(lds, ys)) for lds in slds.LDSs]
    probe = SSD._slqr_terminal_probe(slds, data.ux)
    SSD._slqr_sync_probe!(probe, slds)
    SSD._slqr_probe_estep!(probe)
    problem = SSD._lqr_conditional_problem(
        slds.LDSs,
        sufs,
        [ones(Int, K), ones(Int, K), collect(1:K), collect(1:K)],
        SSD._lqr_block_slots(Symbol[], K),
        SSD._SLQRNormalizer(probe),
    )
    @test problem.rescores
    θ = copy(problem.theta)
    ∇ = similar(θ)
    @test isfinite(problem.evaluate!(∇, θ))
    before = problem.evaluate!(nothing, θ)

    rng = StableRNG(5)
    for _ in 1:3
        d = normalize(randn(rng, length(θ)))
        ε = 1e-5
        fd = (problem.score!(θ .+ ε .* d) - problem.score!(θ .- ε .* d)) / (2ε)
        @test fd ≈ dot(∇, d) rtol = 1e-6
    end
    @test problem.evaluate!(nothing, θ) == before
    problem.write!(θ)          # leave the model where it started
    return nothing
end

"""An inverse-LQR discrete state may not also carry a deterministic cost
schedule: in a switching model the discrete state *is* the cost epoch, so a
schedule inside one would be a second notion of regime nested in the first.
`terminal` and `observe_costate` describe the trial and the emission rather than
the state, so they have to agree across states."""
function test_slds_lqr_validation()
    p = 4
    ok = hslds_model([[0.25 0.04; 0.04 0.18], [0.9 0.0; 0.0 0.7]]; p=p)
    @test validate_SLDS(ok) === nothing

    # A discrete state with its own cost schedule.
    A = [0.96 0.07; -0.05 0.93]
    S = [0.06 0.01; 0.01 0.05]
    Σ = Matrix(0.05I, 4, 4)
    C = randn(StableRNG(11), p, 4)
    C[:, 3:4] .= 0
    obs() = GaussianObservationModel(copy(C), Matrix(0.1I, p, p), zeros(p))
    scheduled = LQRStateModel(
        copy(A),
        copy(S),
        [[0.25 0.04; 0.04 0.18], [0.9 0.0; 0.0 0.7]],
        copy(Σ);
        schedule=cost_schedule(30; onset=10),
        P0=Matrix(0.3I, 4, 4),
    )
    bad = SLDS(;
        A=[0.9 0.1; 0.1 0.9],
        πₖ=[0.5, 0.5],
        LDSs=[
            LinearDynamicalSystem(scheduled, obs()),
            hslds_state([0.9 0.0; 0.0 0.7]; p=p, C=C),
        ],
    )
    @test_throws ArgumentError validate_SLDS(bad)

    # `terminal` disagreeing across states.
    mixed_terminal = SLDS(;
        A=[0.9 0.1; 0.1 0.9],
        πₖ=[0.5, 0.5],
        LDSs=[
            hslds_state([0.25 0.04; 0.04 0.18]; p=p, C=C, terminal=true),
            hslds_state([0.9 0.0; 0.0 0.7]; p=p, C=C, terminal=false),
        ],
    )
    @test_throws ArgumentError validate_SLDS(mixed_terminal)

    # `observe_costate` disagreeing across states.
    seeing = LQRStateModel(
        copy(A),
        copy(S),
        [0.9 0.0; 0.0 0.7],
        copy(Σ);
        observe_costate=true,
        P0=Matrix(0.3I, 4, 4),
    )
    mixed_readout = SLDS(;
        A=[0.9 0.1; 0.1 0.9],
        πₖ=[0.5, 0.5],
        LDSs=[
            hslds_state([0.25 0.04; 0.04 0.18]; p=p, C=C),
            LinearDynamicalSystem(seeing, obs()),
        ],
    )
    @test_throws ArgumentError validate_SLDS(mixed_readout)

    # A free state alongside an LQR one is the supported mix, not an error.
    free = LinearDynamicalSystem(
        free_state_model(
            0.9 * Matrix(1.0I, 4, 4), Matrix(0.1I, 4, 4); P0=Matrix(0.3I, 4, 4)
        ),
        obs(),
    )
    @test validate_SLDS(
        SLDS(;
            A=[0.9 0.1; 0.1 0.9],
            πₖ=[0.5, 0.5],
            LDSs=[hslds_state([0.25 0.04; 0.04 0.18]; p=p, C=C), free],
        ),
    ) === nothing
    return nothing
end

"""Which cost was active is recoverable from data the model itself generates, and
*not* from exactly-optimal trajectories — a distinction that matters more for
switching than for a single fit.

`simulate_lqr` follows the stable manifold (the Riccati solution) while the
model's density is the forward symplectic chain, which is divergent: this model
is a relaxation of exact optimality rather than a description of it. A single fit
pays some accuracy for that. Switching pays much more, because the
responsibilities *compare* two such densities and the mismatch is larger than the
cost signal — on `simulate_lqr` data the margin comes out favouring the cheaper
cost even on expensive-cost trials.

The assertion is on the gap between the two regimes rather than on either
accuracy alone, so it keeps stating the real claim if the absolute numbers move.
"""
function test_slds_lqr_prior_vs_optimal_data()
    A = [0.96 0.07; -0.05 0.93]
    S = [0.06 0.01; 0.01 0.05]
    Σ = Matrix(0.02I, 4, 4)
    Q_lo = [0.20 0.03; 0.03 0.15]
    Q_hi = [1.20 0.0; 0.0 0.90]
    obs_dim, tsteps, ntrials = 8, 16, 20
    C = randn(StableRNG(1234), obs_dim, 4)
    C[:, 3:4] .= 0

    function gen(Q)
        return LQRStateModel(copy(A), copy(S), copy(Q), copy(Σ); P0=Matrix(0.2I, 4, 4))
    end
    function st(Q)
        return LinearDynamicalSystem(
            gen(Q),
            GaussianObservationModel(
                copy(C), Matrix(0.02I, obs_dim, obs_dim), zeros(obs_dim)
            ),
        )
    end
    lo, hi = st(Q_lo), st(Q_hi)

    # The two control problems really are distinguishable in principle.
    @test maximum(abs, eigvals(closed_loop_dynamics(gen(Q_lo)))) >
        maximum(abs, eigvals(closed_loop_dynamics(gen(Q_hi)))) + 0.05

    margin(y) = loglikelihood(lo, [y]) - loglikelihood(hi, [y])
    function accuracy(ylo, yhi)
        return (count(>(0), margin.(ylo)) + count(<(0), margin.(yhi))) / (2 * ntrials)
    end

    # From each model's own prior: the likelihood separates them sharply.
    prior_acc = accuracy(
        [rand(StableRNG(50 + i), lo, tsteps)[2] for i in 1:ntrials],
        [rand(StableRNG(90 + i), hi, tsteps)[2] for i in 1:ntrials],
    )
    @test prior_acc > 0.9

    # From exactly-optimal trajectories: it does not.
    opt_acc = accuracy(
        [
            C * simulate_lqr(StableRNG(150 + i), gen(Q_lo), tsteps; costate_slack=0.03) .+
            sqrt(0.02) .* randn(StableRNG(300 + i), obs_dim, tsteps) for i in 1:ntrials
        ],
        [
            C * simulate_lqr(StableRNG(190 + i), gen(Q_hi), tsteps; costate_slack=0.03) .+
            sqrt(0.02) .* randn(StableRNG(400 + i), obs_dim, tsteps) for i in 1:ntrials
        ],
    )
    @test prior_acc - opt_acc > 0.25
    return nothing
end

"""A state whose own terminal cost differs from its running cost, and a `:free`
state beside it — the pair a delay-then-reach model uses."""
function hslds_terminal_pair(p, tsteps; terminal_first=true)
    n = 2
    C = randn(StableRNG(11), p, 2n)
    C[:, (n + 1):end] .= 0
    obs() = GaussianObservationModel(copy(C), Matrix(0.1I, p, p), zeros(p))
    lqr = LQRStateModel(
        [0.96 0.07; -0.05 0.93],
        [0.06 0.01; 0.01 0.05],
        [[0.25 0.04; 0.04 0.18], [2.0 0.0; 0.0 1.5]],
        Matrix(0.05I, 2n, 2n);
        schedule=cost_schedule(tsteps; terminal=true),
        terminal=true,
        condition_terminal=false,
        Σf=Matrix(0.02I, n, n),
        P0=Matrix(0.3I, 2n, 2n),
    )
    free = free_state_model(
        0.9 * Matrix(1.0I, 2n, 2n), Matrix(0.05I, 2n, 2n); P0=Matrix(0.3I, 2n, 2n)
    )
    members = [LinearDynamicalSystem(lqr, obs()), LinearDynamicalSystem(free, obs())]
    terminal_first || reverse!(members)
    return SLDS(; A=[0.9 0.1; 0.1 0.9], πₖ=[0.5, 0.5], LDSs=members)
end

"""A `:free` state and an inverse-LQR one sharing an emission family — the
free state as `free_state_model` builds it by default, allowed to read every
latent coordinate, the LQR state with `observe_costate` as given."""
function hslds_mixed_pair(emission; free_first::Bool=false, observe_costate::Bool=false)
    n = 2
    A = [0.95 0.10; -0.05 0.90]
    S = Matrix(0.15I, n, n)
    Qc = Matrix(0.20I, n, n)
    Σ = Matrix(Diagonal(fill(0.02, 2n)))
    lqr = LQRStateModel(
        copy(A),
        copy(S),
        [copy(Qc)],
        copy(Σ);
        terminal=true,
        condition_terminal=false,
        x0=zeros(2n),
        P0=Matrix(0.1I, 2n, 2n),
        observe_costate=observe_costate,
    )
    seed = LQRStateModel(
        copy(A), copy(S), [copy(Qc)], copy(Σ); x0=zeros(2n), P0=Matrix(0.1I, 2n, 2n)
    )
    free = SSD.free_state_model(
        Matrix(SSD.symplectic_matrix(seed, 1)),
        Matrix(seed.cache.Qfwd);
        h=Vector(seed.cache.bfwd),
        x0=zeros(2n),
        P0=Matrix(0.1I, 2n, 2n),
    )
    @assert free.observe_costate
    members = [
        LinearDynamicalSystem(lqr, emission()), LinearDynamicalSystem(free, emission())
    ]
    free_first && reverse!(members)
    return SLDS(; A=[0.9 0.1; 0.1 0.9], πₖ=[0.5, 0.5], LDSs=members)
end

"""In a switching model a `:free` state reads the latent coordinates the
inverse-LQR states read, no more: its `observe_costate` is matched to theirs.

The emission then means the same thing in every mode, and a tied `C` has one
mask to obey. Before, a free state (which reads everything by default) kept its
own mask, and a tied emission obeyed whichever mask its path happened to
consult: a partial tie (`:C` without `:d`) none at all, a whole tie only the
first state's — so listing the free state first let its statistics fill the
costate columns the inverse-LQR state then read. Every path is checked —
Gaussian and Poisson, partial and whole ties, both orders, pooled and grouped,
untied — along with a prior that would pull those columns away from zero, and
the matching in the other direction.
"""
function test_slds_lqr_tied_emission_mask()
    n, tsteps, ntrials = 2, 20, 24
    costate = (n + 1):(2n)
    function gauss()
        return GaussianObservationModel(;
            C=hcat(Matrix(1.0I, n, n), zeros(n, n)), d=zeros(n), R=Matrix(0.05I, n, n)
        )
    end
    function pois()
        return PoissonObservationModel(;
            C=hcat(0.5 .* Matrix(1.0I, n, n), zeros(n, n)), d=fill(0.5, n)
        )
    end
    rng = StableRNG(7)
    y_gauss = [0.5 .* randn(rng, n, tsteps) for _ in 1:ntrials]
    y_pois = [Float64.(rand(StableRNG(100 + i), 0:3, n, tsteps)) for i in 1:ntrials]
    labels = [isodd(i) ? "lo" : "hi" for i in 1:ntrials]
    function fitted(
        emission, y, tied; free_first=false, grouped=false, prior=nothing, observe=false
    )
        slds = hslds_mixed_pair(emission; free_first=free_first, observe_costate=observe)
        for lds in slds.LDSs
            grouped && (lds.state_model.depends_on = (Gref=labels,))
            prior === nothing || (lds.obs_model.CD_prior = prior)
        end
        # One M-step: the final iteration only scores.
        fit!(slds, y; max_iter=2, progress=false, rng=StableRNG(3), tied_params=tied)
        return slds
    end
    reads_costate(slds) = [norm(lds.obs_model.C[:, costate]) for lds in slds.LDSs]

    for (emission, y, ties) in (
        (gauss, y_gauss, ([:A, :S], [:A, :S, :C, :R], [:A, :S, :C, :d, :R])),
        (pois, y_pois, ([:A, :S], [:A, :S, :C, :d])),
    )
        for tied in ties, free_first in (false, true), grouped in (false, true)
            slds = fitted(emission, y, tied; free_first=free_first, grouped=grouped)
            @test all(iszero, reads_costate(slds))
            @test !any(lds.state_model.observe_costate for lds in slds.LDSs)
        end
    end

    #=
    A prior centred away from zero in the costate columns, coupling them to the
    state columns: masking only the data would let it recreate them. Block
    diagonal between `C` and `d`, as a partial tie requires.
    =#
    M₀ = [1.0 0.2 0.5 -0.4; 0.1 0.9 0.3 0.6]
    L = [2.0 0.5 0.4 0.3; 0.0 2.0 0.2 0.1; 0.0 0.0 1.5 0.3; 0.0 0.0 0.0 1.5]
    Λ = zeros(5, 5)
    Λ[1:4, 1:4] .= L' * L
    Λ[5, 5] = 1.0
    prior = MNPrior(; M₀=hcat(M₀, zeros(n)), Λ=Λ)
    for tied in ([:A, :S, :C, :R], [:A, :S, :C, :d, :R]), free_first in (false, true)
        slds = fitted(gauss, y_gauss, tied; free_first=free_first, prior=prior)
        @test all(iszero, reads_costate(slds))
        @test norm(slds.LDSs[1].obs_model.C[:, 1:n] .- M₀[:, 1:n]) > 0.1   # data moved it
    end

    #=
    A whole tie is a labelling-invariant pooled fit: the same `[C d]` in either
    order, and the same through a grouping that splits nothing. So is `R`, which
    is fitted after it from each state's residuals, and so needs every state to
    hold the shared `C` by then. (Not the ELBO: the LQR state's L-BFGS stopping
    point moves it by ~1e-3 nats under a 1e-14 change in its statistics.)
    =#
    tied = [:A, :S, :C, :d, :R]
    lqr_first = fitted(gauss, y_gauss, tied)
    free_first = fitted(gauss, y_gauss, tied; free_first=true)
    grouped = fitted(gauss, y_gauss, tied; grouped=true)
    C = lqr_first.LDSs[1].obs_model.C
    @test norm(C[:, 1:n] .- Matrix(1.0I, n, n)) > 0.05                     # it was fitted
    @test free_first.LDSs[2].obs_model.C ≈ C rtol = 1e-10
    @test grouped.LDSs[1].obs_model.C ≈ C rtol = 1e-10
    for (k, j) in ((1, 2), (2, 1))
        @test free_first.LDSs[j].obs_model.R ≈ lqr_first.LDSs[k].obs_model.R rtol = 1e-10
        @test free_first.LDSs[j].obs_model.d ≈ lqr_first.LDSs[k].obs_model.d rtol = 1e-10
    end
    @test grouped.LDSs[2].obs_model.R ≈ lqr_first.LDSs[2].obs_model.R rtol = 1e-10
    p_lqr = fitted(pois, y_pois, [:A, :S, :C, :d])
    p_free = fitted(pois, y_pois, [:A, :S, :C, :d]; free_first=true)
    @test p_free.LDSs[2].obs_model.C ≈ p_lqr.LDSs[1].obs_model.C rtol = 1e-8

    # The other direction: LQR states that read their costate let the free one read it too.
    reading = fitted(gauss, y_gauss, [:A, :S]; observe=true)
    @test all(lds.state_model.observe_costate for lds in reading.LDSs)
    @test all(>(1e-3), reads_costate(reading))

    #=
    `rand` matches too, so a sample is drawn from the model a fit would see — even
    from a free state constructed with a nonzero costate readout, which is
    zeroed with a warning like any other state that may not read those columns.
    =#
    fresh = hslds_mixed_pair(gauss; free_first=true)
    fresh.LDSs[1].obs_model.C[:, costate] .= 0.3
    @test_logs (:warn, r"costate columns") match_mode = :any rand(StableRNG(1), fresh, 5)
    @test !fresh.LDSs[1].state_model.observe_costate
    @test all(iszero, reads_costate(fresh))

    # A model with no inverse-LQR state leaves its free states alone.
    lone = hslds_mixed_pair(gauss)
    only_free = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[lone.LDSs[2]])
    SSD._match_costate_readout!(only_free)
    @test only_free.LDSs[1].state_model.observe_costate
    return nothing
end

"""The order the discrete states are listed in is a labelling, not a model.

An inverse-LQR state with its own terminal cost carries two cost regimes and a
`:free` state one. The responsibility-weighted statistics used to be allocated
as `K` copies of whatever the *first* state needed, so listing the free state
first under-sized the LQR state's per-regime blocks and the fit died with a
`BoundsError` — the other order worked by accident. Both must fit, and to the
same model: exactly for the free state and the chain, and for the LQR state up
to where its structural optimizer stops in the cost's flat direction, which
flips with the last bit of the data (see `test_slds_lqr_grouped_free_state_pools`).
"""
function test_slds_lqr_state_order()
    p, tsteps = 4, 20
    ys = hslds_data(p, tsteps, 4)
    function fitted(terminal_first)
        slds = hslds_terminal_pair(p, tsteps; terminal_first=terminal_first)
        # One M-step: the final iteration only scores.
        e = _trace(fit!(slds, ys; max_iter=2, progress=false, rng=StableRNG(1)))
        return slds, e
    end
    a, ea = fitted(true)
    b, eb = fitted(false)
    @test maximum(abs, ea .- eb) < 1e-2
    @test maximum(abs, a.A .- b.A[[2, 1], [2, 1]]) < 1e-10
    @test maximum(abs, a.LDSs[2].state_model.Mfree .- b.LDSs[1].state_model.Mfree) < 1e-10

    la, lb = a.LDSs[1].state_model, b.LDSs[2].state_model
    rel(x, y) = norm(x .- y) / norm(x)
    @test norm(la.Qc[2] .- [2.0 0.0; 0.0 1.5]) > 0.1     # the terminal cost was fitted
    @test rel(la.A, lb.A) < 1e-3
    @test rel(la.S, lb.S) < 2e-2
    @test rel(la.Qc[1], lb.Qc[1]) < 2e-2
    @test rel(la.Qc[2], lb.Qc[2]) < 2e-2
    return nothing
end

"""`rand` rolls each inverse-LQR state's own transition cost, and every entry
point refuses a state whose schedule switches costs within a trial.

The sampler used to hand back `cache.M[1]` whatever the schedule said, while the
smoother and M-step honoured it — silently a different model. A separate
terminal cost is not a switch (it is read only by the terminal factor), so a
state whose transitions all follow cost 2 and whose terminal step is written
against cost 1 must roll `M[2]`.
"""
function test_slds_lqr_rand_schedules()
    p, tsteps = 4, 20
    A = [0.96 0.07; -0.05 0.93]
    S = [0.06 0.01; 0.01 0.05]
    Σ = Matrix(0.05I, 4, 4)
    C = randn(StableRNG(11), p, 4)
    C[:, 3:4] .= 0
    obs() = GaussianObservationModel(copy(C), Matrix(0.1I, p, p), zeros(p))
    partner() = hslds_state([0.9 0.0; 0.0 0.7]; p=p, C=C, terminal=true)

    flipped = LQRStateModel(
        copy(A),
        copy(S),
        [[2.0 0.0; 0.0 1.5], [0.25 0.04; 0.04 0.18]],
        copy(Σ);
        schedule=vcat(fill(2, tsteps - 1), 1),
        terminal=true,
        Σf=Matrix(0.02I, 2, 2),
        P0=Matrix(0.3I, 4, 4),
    )
    @test SSD._slds_transition_regimes(flipped) == [2]
    @test SSD._extract_state_params(flipped).A === flipped.cache.M[2]
    @test flipped.cache.M[2] != flipped.cache.M[1]
    ok = SLDS(;
        A=[0.9 0.1; 0.1 0.9],
        πₖ=[0.5, 0.5],
        LDSs=[LinearDynamicalSystem(flipped, obs()), partner()],
    )
    @test validate_SLDS(ok) === nothing
    z, x, y = rand(StableRNG(3), ok, tsteps)
    @test size(x) == (4, tsteps) && all(isfinite, x) && all(isfinite, y)

    # The usual terminal pair passes too, in either order.
    @test validate_SLDS(hslds_terminal_pair(p, tsteps)) === nothing
    @test validate_SLDS(hslds_terminal_pair(p, tsteps; terminal_first=false)) === nothing

    # A cost switch inside a state is refused wherever the model is used.
    switching = LQRStateModel(
        copy(A),
        copy(S),
        [[0.25 0.04; 0.04 0.18], [0.9 0.0; 0.0 0.7]],
        copy(Σ);
        schedule=cost_schedule(tsteps; onset=8),
        P0=Matrix(0.3I, 4, 4),
    )
    @test SSD._slds_transition_regimes(switching) == [1, 2]
    bad = SLDS(;
        A=[0.9 0.1; 0.1 0.9],
        πₖ=[0.5, 0.5],
        LDSs=[
            LinearDynamicalSystem(switching, obs()),
            hslds_state([0.9 0.0; 0.0 0.7]; p=p, C=C),
        ],
    )
    ys = hslds_data(p, tsteps, 2)
    @test_throws ArgumentError rand(StableRNG(3), bad, tsteps)
    @test_throws ArgumentError rand(StableRNG(3), bad, [tsteps, tsteps])
    @test_throws ArgumentError fit!(bad, ys; max_iter=2, progress=false)
    @test_throws ArgumentError smooth(bad, ys)
    return nothing
end

"""`rand` must draw from the model it is given, which for an inverse-LQR state
means the *forward* chain `z_{t+1} = M z_t + G h + ε`, `ε ~ N(0, GΣGᵀ)` — the
density inference uses, not the optimal trajectory `simulate_lqr` follows.

Checked against the parameters directly rather than through a fit: the discrete
path's empirical transitions must match `slds.A`, and regressing `z_{t+1}` on
`z_t` within the timesteps each state was active must return that state's cached
`M`. The transition *into* `t` is drawn under `z_t`, so a pair `(t-1, t)` belongs
to state `z_t` — an off-by-one there would return a blend of the two."""
function test_slds_lqr_rand()
    p, tsteps, ntrials = 4, 40, 60
    # Persistent switching, so each state gets long runs to regress within.
    slds = hslds_model([[0.20 0.03; 0.03 0.15], [1.20 0.0; 0.0 0.90]]; p=p, stay=0.97)
    zs, xs, ys = rand(StableRNG(21), slds, fill(tsteps, ntrials))

    @test length(zs) == ntrials
    @test all(size(x) == (4, tsteps) for x in xs)
    @test all(all(isfinite, x) for x in xs)
    @test all(all(isfinite, y) for y in ys)
    @test Set(reduce(vcat, zs)) == Set([1, 2])          # both states are visited

    # Empirical transition matrix of the sampled discrete path.
    counts = zeros(2, 2)
    for z in zs, t in 2:tsteps
        counts[z[t - 1], z[t]] += 1
    end
    emp = counts ./ sum(counts; dims=2)
    @test maximum(abs, emp .- slds.A) < 0.05

    #=
    Within each state, least squares on the pairs it generated must return that
    state's forward transition. `[z_t; 1]` as the regressor absorbs `G h`.
    =#
    d = 4
    for k in 1:2
        W = zeros(d + 1, d + 1)
        V = zeros(d + 1, d)
        for (z, x) in zip(zs, xs), t in 2:tsteps
            z[t] == k || continue
            w = vcat(x[:, t - 1], 1.0)
            W .+= w * w'
            V .+= w * x[:, t]'
        end
        Θ = (W \ V)'                                    # d × (d+1)
        @test maximum(abs, Θ[:, 1:d] .- slds.LDSs[k].state_model.cache.M[1]) < 0.05
    end

    # A mixed free/LQR model samples too — the free state rolls its own matrix.
    C = randn(StableRNG(11), p, 4)
    C[:, 3:4] .= 0
    mixed = SLDS(;
        A=[0.9 0.1; 0.1 0.9],
        πₖ=[0.5, 0.5],
        LDSs=[
            hslds_state([0.25 0.04; 0.04 0.18]; p=p, C=C),
            LinearDynamicalSystem(
                free_state_model(
                    0.9 * Matrix(1.0I, 4, 4), Matrix(0.1I, 4, 4); P0=Matrix(0.3I, 4, 4)
                ),
                GaussianObservationModel(copy(C), Matrix(0.1I, p, p), zeros(p)),
            ),
        ],
    )
    zm, xm, ym = rand(StableRNG(5), mixed, 25)
    @test size(xm) == (4, 25)
    @test all(isfinite, xm) && all(isfinite, ym)
    return nothing
end

"""Each noise version's inverse must come from the model that *uses* it.

The M-step context caches `Σ⁻¹` per noise version. Looking that model up by the
structural version instead is wrong whenever the two groupings differ — with
`:structure` tied and the noise untied, every version would take state 1's `Σ` —
and it is wrong silently, since the objective stays finite and EM still moves.

The assumption behind that lookup holds for `depends_on`, where cells sharing a
parameter alias one array, and fails for an `SLDS`, whose discrete states hold
separate arrays. This checks the case where the two groupings disagree."""
function test_slds_lqr_noise_version_lookup()
    p = 4
    slds = hslds_model([[0.25 0.04; 0.04 0.18], [0.9 0.0; 0.0 0.7]]; p=p)
    sms = [lds.state_model for lds in slds.LDSs]
    # Distinct noise per state, so taking the wrong one is visible.
    copyto!(sms[1].Σ, Matrix(0.05I, 4, 4))
    copyto!(sms[2].Σ, Matrix(0.40I, 4, 4))
    for sm in sms
        refresh!(sm)
    end

    ys = hslds_data(p, 30, 4)
    _, tfs, data, _ = lqr_estep_stats(
        LinearDynamicalSystem(sms[1], slds.LDSs[1].obs_model), ys
    )
    sufs = map(1:2) do k
        hs = SSD._initialize_td_sufficient_statistics(Float64, slds.LDSs[k], data.tsteps)
        SSD._aggregate_lqr_stats_weighted!(
            hs, tfs, slds.LDSs[k], data, [fill(0.5, 30) for _ in 1:4]
        )
        SSD._fill_mixed_blocks!(hs, sms[k])
        hs
    end

    # `:structure` shares every structural block; the noise stays per state.
    slots = SSD._lqr_block_slots([:structure], 2)
    @test all(all(isone, sl) for sl in slots)          # structure is shared
    ctx = SSD._LQRMStepCtx(sufs, sms, slots, [1, 2], false)
    @test ctx.nq == 2

    # Version `k`'s inverse is state `k`'s, not state 1's twice over.
    for k in 1:2
        @test ctx.Sinv[k] ≈ inv(Matrix(sms[k].Σ)) atol = 1e-10
    end
    @test !(ctx.Sinv[1] ≈ ctx.Sinv[2])
    return nothing
end

"""An empty untied noise version contributes no objective or gradient and must
not make the joint profiled structural objective infinite."""
function test_slds_lqr_zero_count_noise_version()
    p, tsteps = 4, 24
    q1 = [0.25 0.04; 0.04 0.18]
    q2 = [0.9 0.0; 0.0 0.7]
    lds1 = hslds_state(q1; p=p)
    lds2 = hslds_state(q2; p=p)
    ys = hslds_data(p, tsteps, 3)
    _, tfs, data, _ = lqr_estep_stats(lds1, ys)

    function weighted(lds, value)
        hs = SSD._initialize_td_sufficient_statistics(Float64, lds, data.tsteps)
        SSD._aggregate_lqr_stats_weighted!(
            hs, tfs, lds, data, [fill(value, tsteps) for _ in eachindex(ys)]
        )
        SSD._fill_mixed_blocks!(hs, lds.state_model)
        return hs
    end

    full = weighted(lds1, 1.0)
    empty = weighted(lds2, 0.0)
    sms = [lds1.state_model, lds2.state_model]
    slots = SSD._lqr_block_slots(Symbol[], 2)
    ctx = SSD._LQRMStepCtx([full, empty], sms, slots, [1, 2], true)
    @test ctx.active_q == [true, false]

    θ = zeros(ctx.pack.np)
    SSD._lqr_pack!(θ, ctx)
    g = similar(θ)
    f = SSD._lqr_fg!(g, θ, ctx)
    @test isfinite(f)
    for b in 1:(SSD._LQR_BLOCK_N)
        @test all(iszero, g[SSD._lqr_blk(ctx.pack, b, 2)])
    end

    one_ctx = SSD._LQRMStepCtx(full, lds1.state_model, true)
    θ1 = zeros(one_ctx.pack.np)
    SSD._lqr_pack!(θ1, one_ctx)
    @test f ≈ SSD._lqr_fg!(nothing, θ1, one_ctx) atol = 1e-10
    return nothing
end

"""
    test_slds_lqr_grouped()

Switching inverse LQR whose *emission* is grouped: the stitched fit, where one
control problem per discrete state is read out through one emission per session.

The grouped switching M-step is a different code path from the ungrouped one —
it aggregates over `K · ncells` (regime, cell) units rather than `K` states, and
the structural parameters have no conjugate update to fall back on — so it gets
the same anchor the ungrouped path has. With `K = 1` the responsibilities are
identically one and the discrete layer contributes nothing, so a grouped
switching fit must reproduce the grouped *single* inverse-LQR fit, parameter for
parameter. Anything that reaches only one of the two shows up here.

The second half checks the piece that anchor cannot see: with `K > 1` the
structural version of unit `(k, c)` is the pair of its version across regimes and
across cells, and a tie has to collapse the first without collapsing the second.
"""
function test_slds_lqr_grouped()
    p, tsteps, ntrials = 4, 30, 6
    ys = hslds_data(p, tsteps, ntrials)
    Qc = [0.25 0.04; 0.04 0.18]
    labels = [:a, :a, :a, :b, :b, :b]
    group(model) = (C=labels, d=labels, R=labels)

    lds = hslds_state(Qc; p=p)
    lds.obs_model.depends_on = group(lds.obs_model)
    slds = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[hslds_state(Qc; p=p)])
    slds.LDSs[1].obs_model.depends_on = group(slds.LDSs[1].obs_model)

    e_lds = _trace(fit!(lds, ys; max_iter=8, tol=1e-14))
    e_slds = _trace(fit!(slds, ys; max_iter=8, progress=false, rng=StableRNG(7)))

    @test all(isfinite, e_slds)
    @test length(e_lds) == length(e_slds)
    # Grouped LDS and grouped SLDS pool the same moments in a different order.
    # The resulting roundoff is amplified slightly by the flat cost-scale
    # direction, but remains below one part per million of the objective.
    @test maximum(abs, e_lds .- e_slds) < 5e-4

    a, b = lds.state_model, slds.LDSs[1].state_model
    @test maximum(abs, a.A .- b.A) < 5e-6
    @test maximum(abs, a.S .- b.S) < 1e-6
    @test maximum(abs, a.Qc[1] .- b.Qc[1]) < 1e-4
    @test maximum(abs, a.Σ .- b.Σ) < 1e-6
    @test maximum(abs, a.x0 .- b.x0) < 1e-6
    # Every group's emission, not just the template's: the whole point of the
    # grouped path is that each session gets its own readout.
    for label in (:a, :b)
        @test maximum(
            abs,
            group_parameter(lds.obs_model, :C, label) .-
            group_parameter(slds.LDSs[1].obs_model, :C, label),
        ) < 1e-5
    end
    # `observe_costate` is off, so no group's emission may read the costate.
    for label in (:a, :b)
        @test all(iszero, group_parameter(slds.LDSs[1].obs_model, :C, label)[:, 3:4])
    end

    #=
    Two states and two groups: four (regime, cell) units, and the structural
    version of each is the pair of its regime version and its cell version. The
    state side is ungrouped here — only the emission is stitched — so every cell
    shares a structural version and the units collapse back to one per regime.
    That is the case the pipeline actually fits, and the one where the stitched
    and single-session fits have to be the same estimator.
    =#
    two = hslds_model([Qc, [0.9 0.0; 0.0 0.7]]; p=p)
    for lds_k in two.LDSs
        lds_k.obs_model.depends_on = group(lds_k.obs_model)
    end
    before = elbo(two, ys)
    trace = _trace(
        fit!(two, ys; max_iter=10, progress=false, rng=StableRNG(7), tied_params=[:A, :S])
    )
    after = elbo(two, ys)

    @test all(isfinite, trace)
    @test after > before
    # The tie held across discrete states...
    @test maximum(abs, two.LDSs[1].state_model.A .- two.LDSs[2].state_model.A) < 1e-9
    @test maximum(abs, two.LDSs[1].state_model.S .- two.LDSs[2].state_model.S) < 1e-9
    # ...while the untied cost did not collapse with it.
    @test maximum(abs, two.LDSs[1].state_model.Qc[1] .- two.LDSs[2].state_model.Qc[1]) >
        1e-6
    for lds_k in two.LDSs
        @test symplectic_defect(lds_k.state_model) < 1e-10
        @test isposdef(lds_k.state_model.Σ)
    end
    return nothing
end

"""
    test_lqr_pair_slots()

The (regime, cell) version map: units agreeing on both axes share a version,
units differing on either get their own, and the result is renumbered from 1
with no gaps — which is what `_LQRMStepCtx` needs, since it sizes its per-version
storage from the maximum.
"""
function test_lqr_pair_slots()
    # Two regimes x two cells, nothing shared: four versions.
    @test SSD._lqr_pair_slots([1, 1, 2, 2], [1, 2, 1, 2]) == [1, 2, 3, 4]
    # Tied across regimes, grouped across cells: one version per cell.
    @test SSD._lqr_pair_slots([1, 1, 1, 1], [1, 2, 1, 2]) == [1, 2, 1, 2]
    # Untied across regimes, ungrouped across cells: one version per regime.
    @test SSD._lqr_pair_slots([1, 1, 2, 2], [1, 1, 1, 1]) == [1, 1, 2, 2]
    # Both collapsed: a single version.
    @test SSD._lqr_pair_slots([1, 1, 1, 1], [1, 1, 1, 1]) == [1, 1, 1, 1]
    # Dense renumbering even when the inputs are not.
    @test SSD._lqr_pair_slots([3, 3, 7, 7], [2, 5, 2, 5]) == [1, 2, 3, 4]
    return nothing
end

#=
A `:free` discrete state beside a grouped LQR one. The SLDS requires every
regime to declare the same labels, so the free state has to accept the names its
LQR neighbour groups by — and on a free state `(Qc = labels,)` names nothing it
holds, so its dynamics stay ONE array across the groups. That array has to end
the M-step at the estimate pooled over every group, not at whichever group was
updated last: with a declaration that varies nothing either state actually
holds, one M-step must reproduce the ungrouped fit exactly.
=#
function test_slds_lqr_grouped_free_state_pools()
    n, T, N = 2, 20, 24
    A = [0.95 0.10; -0.05 0.90]
    S = Matrix(0.15I, n, n)
    Qc = Matrix(0.20I, n, n)
    Σ = Matrix(Diagonal([0.02, 0.02, 0.02, 0.02]))
    function emission()
        return GaussianObservationModel(;
            C=hcat(Matrix(1.0I, n, n), zeros(n, n)), d=zeros(n), R=Matrix(0.05I, n, n)
        )
    end
    function lqr_lds()
        sm = LQRStateModel(
            copy(A),
            copy(S),
            [copy(Qc)],
            copy(Σ);
            terminal=true,
            #=
            What is under test here is how a `:free` state pools with an
            inverse-LQR one, and terminal conditioning is refused for that
            mixture — the shared initial state would be fitted from the
            inverse-LQR regimes alone. Score the joint objective so the pooling
            is what the test exercises.
            =#
            condition_terminal=false,
            x0=zeros(2n),
            P0=Matrix(0.1I, 2n, 2n),
        )
        return LinearDynamicalSystem(sm, emission())
    end
    function free_lds()
        seed = LQRStateModel(
            copy(A),
            copy(S),
            [copy(Qc)],
            copy(Σ);
            terminal=false,
            x0=zeros(2n),
            P0=Matrix(0.1I, 2n, 2n),
        )
        sm = SSD.free_state_model(
            Matrix(SSD.symplectic_matrix(seed, 1)),
            Matrix(seed.cache.Qfwd);
            h=Vector(seed.cache.bfwd),
            x0=zeros(2n),
            P0=Matrix(0.1I, 2n, 2n),
        )
        return LinearDynamicalSystem(sm, emission())
    end

    rng = StableRNG(7)
    y = [0.5 .* randn(rng, n, T) for _ in 1:N]
    labels = [isodd(i) ? "lo" : "hi" for i in 1:N]
    for i in 1:N
        isodd(i) || (y[i] .*= 3.0)   # groups that differ, so last-wins cannot pass
    end

    function fitted(dep)
        slds = SLDS(; A=[0.9 0.1; 0.1 0.9], πₖ=[0.5, 0.5], LDSs=[lqr_lds(), free_lds()])
        if dep !== nothing
            for lds in slds.LDSs
                lds.state_model.depends_on = dep
            end
        end
        # `max_iter = 2` is exactly one M-step: the last iteration only scores.
        fit!(
            slds,
            y;
            max_iter=2,
            progress=false,
            rng=StableRNG(3),
            tied_params=[:A, :S, :C, :R],
        )
        return slds
    end

    # Both models accept the declaration; this used to throw on the free state.
    plain = fitted(nothing)
    grouped = fitted((Gref=labels,))   # ux_dim = 0: splits nothing either holds
    rel(a, b) = norm(a .- b) / max(norm(a), eps())
    fp, fg = plain.LDSs[2].state_model, grouped.LDSs[2].state_model
    @test rel(fp.Mfree, fg.Mfree) < 1e-10
    @test rel(fp.h, fg.h) < 1e-10
    @test rel(fp.Σ, fg.Σ) < 1e-10
    #=
    The inverse-LQR state cannot be held to the same standard, and not because
    the paths differ: its structural M-step stops at a point in the cost's flat
    direction that flips with the last bit of the data. Scaling `y` by
    `1 + 1e-15` moves one M-step's `A` by ~3e-5 and `Qc` by ~2.5e-3 on one
    Julia version and by nothing on another. A pooling that dropped or
    double-counted a cell moves them by tens of percent.
    =#
    lp, lg = plain.LDSs[1].state_model, grouped.LDSs[1].state_model
    @test rel(lp.A, lg.A) < 1e-3
    @test rel(closed_loop_dynamics(lp), closed_loop_dynamics(lg)) < 1e-3
    @test rel(lp.Qc[1], lg.Qc[1]) < 2e-2

    # A real grouping: the LQR cost splits, the free state's dynamics do not.
    split = fitted((Qc=labels,))
    free = split.LDSs[2].state_model.variants
    @test all(v -> v.Mfree === free[1].Mfree && v.Σ === free[1].Σ, free)
    costs = split.LDSs[1].state_model.variants
    @test !(costs[1].Qc[1] === costs[2].Qc[1])

    # A partial split of the free regression is refused rather than guessed.
    @test_throws ArgumentError fitted((h=labels,))
    return nothing
end
