#=============================================================================
Switching inverse-LQR: `SLDS` whose discrete states are Hamiltonian models.

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
    sm = HamiltonianStateModel(
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
function test_slds_hamiltonian_matches_lds()
    p, tsteps, ntrials = 4, 35, 5
    ys = hslds_data(p, tsteps, ntrials)
    Qc = [0.25 0.04; 0.04 0.18]

    lds = hslds_state(Qc; p=p)
    slds = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[hslds_state(Qc; p=p)])

    e_lds = _trace(fit!(lds, ys; max_iter=10, tol=1e-14))
    e_slds = _trace(fit!(slds, ys; max_iter=10, progress=false, rng=StableRNG(7)))

    @test length(e_lds) == length(e_slds)
    @test maximum(abs, e_lds .- e_slds) < 1e-5

    #=
    The structural step is L-BFGS on a profiled objective and stops on its own
    tolerance, so ten iterations of the two paths agree to ~1e-6 rather than to
    machine precision. The ELBO match above is the tight claim; these confirm the
    agreement is in every parameter, not just the bound.
    =#
    a, b = lds.state_model, slds.LDSs[1].state_model
    @test maximum(abs, a.A .- b.A) < 1e-6
    @test maximum(abs, a.S .- b.S) < 1e-6
    @test maximum(abs, a.Qc[1] .- b.Qc[1]) < 1e-5
    @test maximum(abs, a.Σ .- b.Σ) < 1e-7
    @test maximum(abs, a.x0 .- b.x0) < 1e-6
    @test maximum(abs, lds.obs_model.C .- slds.LDSs[1].obs_model.C) < 1e-6

    # `observe_costate` is off, so the emission may never read the costate half.
    # Without the mask on the switching side these columns fill in silently.
    @test all(iszero, slds.LDSs[1].obs_model.C[:, 3:4])
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
function test_slds_hamiltonian_monotone()
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

"""An all-`:free` switching model is a Gaussian `SLDS` wearing the Hamiltonian
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

    ys = hslds_data(p, tsteps, ntrials)
    elbos = _trace(fit!(slds, ys; max_iter=12, progress=false, rng=StableRNG(7)))
    @test minimum(diff(elbos)) > -1e-6
    @test all(isfinite, elbos)

    # Each state kept its own kind of parameterization.
    @test symplectic_defect(slds.LDSs[1].state_model) < 1e-10
    @test isposdef(slds.LDSs[2].state_model.Σ)
    @test size(slds.LDSs[2].state_model.Mfree) == (4, 4)
    return nothing
end

"""`tied` shares a parameter version across discrete states. An inverse-LQR
model's structural parameters are coordinates of one constrained
parameterization rather than separable regression columns, so `:structure` ties
`(A, S, Qc, h, Bu, Gref)` as a unit, fitted jointly from the states that use it.
Naming an individual one is rejected rather than promoted to the whole block —
`[:A, :S]` is a different and more useful model than a fully shared block, and
answering it with the latter would fit something the caller did not ask for."""
function test_slds_hamiltonian_tied()
    p, tsteps, ntrials = 4, 35, 5
    slds = hslds_model([[0.25 0.04; 0.04 0.18], [0.9 0.0; 0.0 0.7]]; p=p)
    ys = hslds_data(p, tsteps, ntrials)
    elbos = _trace(
        fit!(
            slds,
            ys;
            max_iter=10,
            rng=StableRNG(7),
            progress=false,
            tied_params=[:structure],
        ),
    )

    @test minimum(diff(elbos)) > -1e-6
    a, b = slds.LDSs[1].state_model, slds.LDSs[2].state_model
    # Shared exactly, not merely close: a tie is one fitted value copied out.
    @test a.A == b.A
    @test a.S == b.S
    @test a.Qc[1] == b.Qc[1]

    # Tying the noise as well.
    slds2 = hslds_model([[0.25 0.04; 0.04 0.18], [0.9 0.0; 0.0 0.7]]; p=p)
    fit!(
        slds2,
        ys;
        max_iter=6,
        progress=false,
        rng=StableRNG(7),
        tied_params=[:structure, :noise],
    )
    @test slds2.LDSs[1].state_model.Σ == slds2.LDSs[2].state_model.Σ

    # Naming part of the block is an error, not a silent whole-block tie.
    slds_err = hslds_model([[0.25 0.04; 0.04 0.18], [0.9 0.0; 0.0 0.7]]; p=p)
    for names in ([:A], [:A, :S], [:Qc], [:Q])
        @test_throws ArgumentError fit!(
            slds_err, ys; max_iter=2, progress=false, rng=StableRNG(7), tied_params=names
        )
    end

    # Untied, the two states are free to differ — otherwise the test above is vacuous.
    slds3 = hslds_model([[0.25 0.04; 0.04 0.18], [0.9 0.0; 0.0 0.7]]; p=p)
    fit!(slds3, ys; max_iter=10, progress=false, rng=StableRNG(7))
    @test slds3.LDSs[1].state_model.Qc[1] != slds3.LDSs[2].state_model.Qc[1]
    return nothing
end

"""The soft terminal condition is an extra factor at `t = T`. Under switching it
is weighted by that state's responsibility at `T` like every other factor —
`joint_loglikelihood!` gets it through the dispatched `state_loglikelihood!`, but
the curvature paths build their state blocks from flat templates that have no
slot for it, so it has to be added there by hand."""
function test_slds_hamiltonian_terminal()
    p, tsteps, ntrials = 4, 30, 5
    slds = hslds_model([[0.25 0.04; 0.04 0.18], [0.9 0.0; 0.0 0.7]]; p=p, terminal=true)
    @test all(lds.state_model.terminal for lds in slds.LDSs)

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

"""An inverse-LQR discrete state may not also carry a deterministic cost
schedule: in a switching model the discrete state *is* the cost epoch, so a
schedule inside one would be a second notion of regime nested in the first.
`terminal` and `observe_costate` describe the trial and the emission rather than
the state, so they have to agree across states."""
function test_slds_hamiltonian_validation()
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
    scheduled = HamiltonianStateModel(
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
    seeing = HamiltonianStateModel(
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
function test_slds_hamiltonian_prior_vs_optimal_data()
    A = [0.96 0.07; -0.05 0.93]
    S = [0.06 0.01; 0.01 0.05]
    Σ = Matrix(0.02I, 4, 4)
    Q_lo = [0.20 0.03; 0.03 0.15]
    Q_hi = [1.20 0.0; 0.0 0.90]
    obs_dim, tsteps, ntrials = 8, 16, 20
    C = randn(StableRNG(1234), obs_dim, 4)
    C[:, 3:4] .= 0

    function gen(Q)
        return HamiltonianStateModel(
            copy(A), copy(S), copy(Q), copy(Σ); P0=Matrix(0.2I, 4, 4)
        )
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
