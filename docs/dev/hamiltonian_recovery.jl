#=============================================================================
Inverse-LQR parameter recovery — a standalone validation harness.

Run it:

    julia --project=docs/dev docs/dev/hamiltonian_recovery.jl
    julia --project=. -e 'include("docs/dev/hamiltonian_recovery.jl")'

This is not a test. Tests check that the machinery computes what it claims;
this checks what you can actually *learn* from data, which is a different and
softer question — how close the estimated cost gets, under which conditions,
and how that degrades. It prints a table you read, not an assertion that passes.

The design decision worth knowing: `C = I` by default. An inverse-LQR fit has
two nested identifiability problems — the emission's latent basis, and the cost
given the latents — and mixing them tells you nothing about either. Fixing the
emission to the identity isolates the second, which is the one this model is
for. `--free-C` relaxes it if you want to see the cost of the other.

The cost is identified only up to a nonzero scalar (scaling a cost does not
change the policy it induces), so every comparison is made after
`rescale_costate!(...; target = :trace)`.
=============================================================================#

using StateSpaceDynamics
using LinearAlgebra
using Printf
using Random

const SSD = StateSpaceDynamics

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

"""
    truth_model(; n, terminal, tsteps, ux_dim, observe_costate, tracking)

The generating model. `A` is a mild contraction and `S`, `Qc` are scaled so the
symplectic spectral radius stays near 1 — a Hamiltonian matrix has reciprocal
eigenvalue pairs, so `ρ(M)^T` is how fast the model's own forward chain
diverges, and a sampler-based recovery check needs it modest.
"""
function truth_model(;
    n::Int=2,
    terminal::Bool=false,
    tsteps::Int=20,
    ux_dim::Int=0,
    observe_costate::Bool=false,
    tracking::Bool=false,
)
    d = 2n
    A = Matrix(0.96I, n, n)
    for i in 1:(n - 1)
        A[i, i + 1] = 0.05
        A[i + 1, i] = -0.04
    end
    S = Matrix(0.05I, n, n) + fill(0.01, n, n) - Diagonal(fill(0.01, n))
    Qbase = Matrix(0.20I, n, n) + fill(0.03, n, n) - Diagonal(fill(0.03, n))
    qcs = terminal ? [Qbase, 3.0 .* Matrix(1.0I, n, n)] : [Qbase]
    sched = terminal ? cost_schedule(tsteps; terminal=true) : Int[]
    Σ = Matrix(0.02I, d, d)
    return HamiltonianStateModel(
        A,
        S,
        qcs,
        Σ;
        schedule=sched,
        terminal=terminal,
        Σf=Matrix(0.02I, n, n),
        P0=Matrix(0.2I, d, d),
        Bu=ux_dim > 0 ? zeros(d, ux_dim) : nothing,
        Gref=(tracking && ux_dim > 0) ? Matrix(1.0I, n, ux_dim) : nothing,
        observe_costate=observe_costate,
    )
end

"""
    emission(n, obs_dim; free_C, observe_costate, rng)

`C = [I 0]` by default: the state is observed directly and the costate is not,
so the only thing left to identify is the cost. `free_C` draws a random readout
instead, which folds the latent-basis problem back in.
"""
function emission(n::Int, obs_dim::Int; free_C::Bool, observe_costate::Bool, rng)
    d = 2n
    C = zeros(obs_dim, d)
    if free_C
        C .= randn(rng, obs_dim, d)
        observe_costate || (C[:, (n + 1):d] .= 0)
    else
        obs_dim == n || error("C = I needs obs_dim == n; got $obs_dim vs $n")
        C[1:n, 1:n] .= Matrix(1.0I, n, n)
    end
    return C
end

# ---------------------------------------------------------------------------
# One recovery run
# ---------------------------------------------------------------------------

"""
    recover(; kwargs...) -> NamedTuple

Simulate from a known model, refit from a deliberately wrong start, and report
how close the cost came. Returns the relative error on `Qc[1]` in the canonical
scale, the ELBO the fit reached, the ELBO at the generating parameters, and
whether EM stayed monotone.

`known_plant` is the usual inverse-optimal-control posture: the body is known,
the objective is not. Set it `false` to estimate both.
"""
function recover(;
    n::Int=2,
    tsteps::Int=20,
    ntrials::Int=120,
    terminal::Bool=false,
    tracking::Bool=false,
    free_C::Bool=false,
    observe_costate::Bool=false,
    known_plant::Bool=true,
    obs_noise::Float64=0.05,
    max_iter::Int=250,
    seed::Int=1,
)
    rng = MersenneTwister(seed)
    d = 2n
    ux_dim = tracking ? n : 0
    truth = truth_model(;
        n=n,
        terminal=terminal,
        tsteps=tsteps,
        ux_dim=ux_dim,
        observe_costate=observe_costate,
        tracking=tracking,
    )
    obs_dim = free_C ? max(2n, 6) : n
    C = emission(n, obs_dim; free_C=free_C, observe_costate=observe_costate, rng=rng)
    R = Matrix(obs_noise * I, obs_dim, obs_dim)
    gen = LinearDynamicalSystem(
        truth, GaussianObservationModel(copy(C), copy(R), zeros(obs_dim))
    )

    #=
    Draw from the model itself. Recovery is only a well-posed question against
    the model's own distribution: an exactly-optimal trajectory (`simulate_lqr`)
    has a rank-`n`, time-varying innovation that a full-rank constant `Σ`
    contains only as a limit, so fitting one is a projection rather than
    estimation. See the note this prints at the end.
    =#
    uxs = ux_dim > 0 ? [repeat(randn(rng, n), 1, tsteps) for _ in 1:ntrials] : nothing
    _, ys = rand(rng, gen, fill(tsteps, ntrials); ux=uxs)
    truth_elbo = elbo(gen, ys; ux=uxs)

    # Refit from a wrong cost.
    fit_sm = truth_model(;
        n=n,
        terminal=terminal,
        tsteps=tsteps,
        ux_dim=ux_dim,
        observe_costate=observe_costate,
        tracking=tracking,
    )
    for Q in fit_sm.Qc
        Q .= Matrix(0.4I, n, n)
    end
    fit_sm.Σ .= Matrix(0.05I, d, d)
    fit_sm.fit_flags = HamiltonianFitFlags(; A=!known_plant, S=!known_plant, Gref=false)
    refresh!(fit_sm)
    fit_lds = LinearDynamicalSystem(
        fit_sm, GaussianObservationModel(copy(C), copy(R), zeros(obs_dim))
    )
    #=
    Freeze the emission too when `C = I`: that is the point of fixing it, and a
    fitted `C` would drift the latent basis out from under the comparison.
    =#
    free_C || (fit_lds.fit_bool[5] = false)

    elbos = fit!(fit_lds, ys; ux=uxs, max_iter=max_iter, tol=1e-10, progress=false)

    # Compare in the canonical scale: the cost is identified up to a scalar.
    ref = deepcopy(truth)
    rescale_costate!(ref; target=:trace)
    rescale_costate!(fit_sm; target=:trace)
    err = [
        maximum(abs, fit_sm.Qc[k] .- ref.Qc[k]) / maximum(abs, ref.Qc[k]) for
        k in eachindex(ref.Qc)
    ]
    return (
        err=err,
        elbo=elbos[end],
        truth_elbo=truth_elbo,
        iters=length(elbos),
        monotone=minimum(diff(elbos)) > -1e-8,
        rho=maximum(abs, eigvals(symplectic_matrix(truth))),
        defect=symplectic_defect(fit_sm),
    )
end

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

function header()
    @printf(
        "%-42s %8s %9s %9s %6s %5s %s\n",
        "condition",
        "Qc err",
        "elbo",
        "truth",
        "iters",
        "mono",
        "ρ(M)"
    )
    return println("-"^96)
end

function report(label, r; note::String="")
    @printf(
        "%-42s %8.3f %9.1f %9.1f %6d %5s %.3f %s\n",
        label,
        maximum(r.err),
        r.elbo,
        r.truth_elbo,
        r.iters,
        r.monotone ? "yes" : "NO",
        r.rho,
        note
    )
    return nothing
end

function main(; free_C::Bool=false, quick::Bool=false)
    println()
    println("Inverse-LQR parameter recovery")
    println(
        "C = ",
        if free_C
            "random readout (latent basis also unidentified)"
        else
            "I (state observed directly)"
        end,
    )
    println()
    header()

    ntr = quick ? 40 : 120
    report("baseline, known plant", recover(; ntrials=ntr, free_C=free_C))
    report("  half the trials", recover(; ntrials=ntr ÷ 2, free_C=free_C))
    report("  double the trials", recover(; ntrials=2ntr, free_C=free_C))
    report("  longer trials (T = 40)", recover(; ntrials=ntr, tsteps=40, free_C=free_C))
    report("  noisier observations", recover(; ntrials=ntr, obs_noise=0.25, free_C=free_C))
    report("plant estimated too", recover(; ntrials=ntr, known_plant=false, free_C=free_C))
    report("costate observed", recover(; ntrials=ntr, observe_costate=true, free_C=true))
    #=
    The terminal row is expected to look bad, and it is worth saying why rather
    than letting a reader take it for a defect. The terminal factor is a
    conditioning event: `rand` draws `p(z, y | y_term = 0)` while the fitted
    objective is `p(y, y_term = 0)`, and the two differ by `p(y_term = 0 | θ)`.
    Maximizing the latter on data drawn from the former is a selection effect,
    not an unbiased estimator, so the fit legitimately beats the truth's ELBO by
    a lot. Encode a terminal cost as the last regime of the transition schedule
    when you want an unbiased recovery check.
    =#
    report(
        "terminal cost",
        recover(; ntrials=ntr, terminal=true, free_C=free_C);
        note="<- conditioning event; see note",
    )
    report("tracking a reference", recover(; ntrials=ntr, tracking=true, free_C=free_C))
    if !quick
        report("n = 3", recover(; n=3, ntrials=ntr, free_C=free_C))
        for seed in 2:4
            report("baseline, seed $seed", recover(; ntrials=ntr, seed=seed, free_C=free_C))
        end
    end

    println()
    println(
        """
Reading this table
------------------
`Qc err` is the largest relative entry error on a cost matrix, after
canonical rescaling. The terminal-cost row is the one expected to look bad:
its factor is a conditioning event, so `rand` draws the conditioned path
distribution while the objective is the joint, and maximizing the latter on
the former is a selection effect rather than an estimator. Encode a terminal
cost as the last regime of the transition schedule for an unbiased check. `elbo` versus `truth` says whether the fit found a
better explanation than the generating parameters — a gap of a few nats is
ordinary finite-sample slack, a large one means the model prefers something
other than the truth.

These trials are drawn from the *model*. Behaviour from an exactly optimal
agent (`simulate_lqr`) is a harder case: its costate is a deterministic
function of its state, so its innovation is rank n and time-varying, while
the model's Σ is full rank and constant. Fitting that is a projection onto
the model rather than estimation within it, and the maximum-likelihood cost
need not be the generating one. What sharpens identification, roughly in
order: observing the costate, a terminal condition, behaviour that is noisily
rather than exactly optimal, a cost that changes within the trial, and more
trials.
""",
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; free_C=("--free-C" in ARGS), quick=("--quick" in ARGS))
end
