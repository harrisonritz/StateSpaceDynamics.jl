#=============================================================================
Inverse-LQR parameter recovery — a standalone validation harness.

Run it:

    julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl
    julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --quick
    julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --full --only=design,switching
    julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --smoulder

The `docs` environment needs the local package dev'd in, which is the same thing
CI does before building the docs:

    julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'

This is not a test. Tests check that the machinery computes what it claims; this
checks what you can actually *learn* from data, which is a different and softer
question — how close the estimated cost gets, under which conditions, and how
that degrades. It prints tables you read and writes figures you look at, not an
assertion that passes.

## Flags

    --quick              a ~2-minute smoke test; one seed per cell, so its
                         numbers say whether the code runs, not what is true
    --full               the long tier: more seeds, more trials, wider ladders
    --smoulder           the requested 12-plant-dim, 500-trial, 100-bin,
                         150-neuron Poisson tier; defaults to the three
                         smoulder experiments only
    --only=a,b           run only these experiments (overview, design, scale,
                         procedure, initialization, modelrecovery, goldstandard,
                         priors, switching, reference-audit, smoulder-lqr, smoulder-gref,
                         smoulder-slqr)
    --gen=rand|lqr|both  which generative mode the parametric sweeps use
                         (default: both for `design`, `lqr` elsewhere)
    --no-figures         tables only
    --selftest           check the scoring plumbing and exit
    --free-C             let the emission be a random readout, folding the
                         latent-basis problem back into every row

## The files

    model.jl        the generating models and the data they produce
    scoring.jl      the metrics, parameter blocks and γ alike
    recovery.jl     simulate → refit → score, for one system
    slds.jl         the same for an `SLDS` with one free and one LQR state
    plotting.jl     figures (PNG, into `figures/`, gitignored)
    report.jl       tables
    experiments.jl  the five sweeps
    smoulder.jl     grouped Poisson LQR/SLQR recovery at the task's scale
    reference_audit.jl  labelled-target likelihood and EM convergence diagnostics

## The design decision worth knowing

`C = [I 0]` by default. An inverse-LQR fit has two nested identifiability
problems — the emission's latent basis, and the cost given the latents — and
mixing them tells you nothing about either. Fixing the emission to the identity
isolates the second, which is the one this model is for. `--free-C` relaxes it
if you want to see the cost of the other.

The cost is identified only up to a nonzero scalar (scaling a cost does not
change the policy it induces), so every comparison is made after
`rescale_costate!(...; target = :trace)`.
=============================================================================#

using StateSpaceDynamics
using LinearAlgebra
using Printf
using Random
using Distributions

const SSD = StateSpaceDynamics

for _f in (
    "scoring.jl",
    "model.jl",
    "recovery.jl",
    "slds.jl",
    "compare.jl",
    "report.jl",
    "plotting.jl",
    "experiments.jl",
    "smoulder.jl",
    "reference_audit.jl",
)
    include(joinpath(@__DIR__, _f))
end

const DEFAULT_EXPERIMENTS = (
    "overview",
    "design",
    "scale",
    "procedure",
    "initialization",
    "modelrecovery",
    "goldstandard",
    "priors",
    "switching",
)

const ALL_EXPERIMENTS = (
    DEFAULT_EXPERIMENTS..., "reference-audit", "smoulder-lqr", "smoulder-gref", "smoulder-slqr"
)

const SMOULDER_EXPERIMENTS = ("smoulder-lqr", "smoulder-gref", "smoulder-slqr")

"""
    parse_args(args) -> NamedTuple

The flags, resolved. Unknown flags are an error rather than a shrug: a
misremembered flag that silently runs the default tier is how a two-hour sweep
turns into a two-hour smoke test.
"""
function parse_args(args)
    tier_name = :default
    only = collect(DEFAULT_EXPERIMENTS)
    figures = true
    free_C = false
    gen = :both
    self = false
    only_given = false
    for a in args
        if a == "--quick"
            tier_name = :quick
        elseif a == "--selftest"
            self = true
        elseif a == "--full"
            tier_name = :full
        elseif a == "--smoulder"
            tier_name = :smoulder
        elseif a == "--no-figures"
            figures = false
        elseif a == "--free-C"
            free_C = true
        elseif startswith(a, "--only=")
            only_given = true
            only = split(a[8:end], ',')
            bad = setdiff(only, ALL_EXPERIMENTS)
            isempty(bad) || error(
                "--only names unknown experiments: $(join(bad, ", ")); " *
                "valid: $(join(ALL_EXPERIMENTS, ", "))",
            )
        elseif startswith(a, "--gen=")
            g = a[7:end]
            g in ("rand", "lqr", "both") || error("--gen must be rand, lqr or both; got $g")
            gen = Symbol(g)
        else
            error("unknown flag $a; see the header of $(@__FILE__)")
        end
    end
    tier_name === :smoulder && !only_given && (only = collect(SMOULDER_EXPERIMENTS))
    return (
        tier=tier_name, only=only, figures=figures, free_C=free_C, gen=gen, selftest=self
    )
end

_gens(gen, default) = gen === :both ? default : (gen,)

function main(args=String[])
    opt = parse_args(args)
    opt.selftest && return (selftest(); smoulder_selftest(); reference_audit_selftest(); nothing)
    cfg = tier(opt.tier)
    t0 = time()

    println()
    println("Inverse-LQR parameter recovery")
    println("tier          ", opt.tier, "   (", nseeds(cfg.seeds), " per cell)")
    println("plant / trial  n = ", cfg.n, ", T = ", cfg.tsteps, ", N = ", cfg.ntrials)
    println(
        "emission      ",
        if opt.free_C
            "random readout (latent basis also unidentified)"
        else
            "C = [I 0] (state observed directly)"
        end,
    )
    opt.figures && println("figures       ", figdir())

    if "overview" in opt.only
        experiment_overview(cfg; figures=opt.figures, free_C=opt.free_C)
    end
    if "reference-audit" in opt.only
        opt.free_C && error("reference-audit requires fixed C; omit --free-C")
        experiment_reference_audit(cfg; figures=opt.figures)
    end
    if "design" in opt.only
        for g in _gens(opt.gen, (:rand, :lqr))
            experiment_design(cfg; gen=g, figures=opt.figures, free_C=opt.free_C)
            experiment_reference_identification(
                cfg; gen=g, figures=opt.figures, free_C=opt.free_C
            )
        end
    end
    if "scale" in opt.only
        for g in _gens(opt.gen, (:lqr,))
            experiment_scale(cfg; gen=g, figures=opt.figures, free_C=opt.free_C)
        end
    end
    if "procedure" in opt.only
        for g in _gens(opt.gen, (:lqr,))
            experiment_procedure(cfg; gen=g, figures=opt.figures, free_C=opt.free_C)
        end
    end
    if "initialization" in opt.only
        for g in _gens(opt.gen, (:lqr,))
            experiment_initialization(cfg; gen=g, figures=opt.figures, free_C=opt.free_C)
        end
    end
    if "modelrecovery" in opt.only
        experiment_model_recovery(cfg; figures=opt.figures)
    end
    if "goldstandard" in opt.only
        experiment_gold_standard(cfg; figures=opt.figures)
    end
    if "priors" in opt.only
        experiment_priors(cfg; figures=opt.figures)
    end
    if "switching" in opt.only
        experiment_switching(cfg; figures=opt.figures)
    end
    if "smoulder-lqr" in opt.only
        experiment_smoulder_lqr(cfg; figures=opt.figures)
    end
    if "smoulder-gref" in opt.only
        experiment_smoulder_gref(cfg; figures=opt.figures)
    end
    if "smoulder-slqr" in opt.only
        experiment_smoulder_slqr(cfg; figures=opt.figures)
    end

    @printf("\nelapsed: %.1f min\n", (time() - t0) / 60)
    reading_guide()
    return nothing
end

"""
    reading_guide()

What the columns mean. Kept separate from the findings, which belong in
`README.md` and go stale; this is about the instrument, and does not.
"""
function reading_guide()
    return println("""

    Reading these tables
    --------------------
    Each parameter block gets `rmse/corr`, both computed on its independent entries
    (the upper triangle for a symmetric block) after canonical rescaling:

      rmse  relative RMSE — the RMSE divided by the RMS of the true entries, so
            blocks of different magnitude are on one scale. 0 is exact.
      corr  Pearson correlation across entries. It ignores scale entirely, so it
            answers the separate question of whether the estimate has the right
            *shape*; a high rmse with a high corr means a scale error, the reverse
            means a structural one.

    The tables omit the delay regime's cost, which would read `--` on most rows; the
    `params_*` figures draw it. Read its relative error with the denominator in mind —
    the delay cost is a near-zero regime by construction, so the RMS of its true entries
    is small and a modest absolute error is a large relative one.

    `Σ xx` is the *state* block of the innovation, not the whole of it: the canonical
    rescaling multiplies `Σ`'s costate rows and columns by the cost scale, so a whole-`Σ`
    error is mostly the cost-scale error again, which `S` already reports exactly.

    `Qc run` and `Qc term` are scored separately rather than pooled, because they are
    identified by different things — the running cost by the within-trial transitions,
    the terminal cost by the endpoint condition alone. `A` reads `--` when the plant is
    known, because it is then frozen at the truth and a perfect score would be
    reporting the initialization. `S` is frozen too on those rows, but the canonical
    rescaling divides it by the fitted cost scale — so its rmse there is exactly the
    relative error in `tr(Qc[1])`, and its corr is exactly ±1. A corr of `-1.00` is
    worth stopping on: the costate's *sign* is unidentified along with its scale,
    `:trace` canonicalization pins it by making `tr(Qc[1])` positive, and a `-1`
    therefore says the fit landed on a cost of the opposite sign. `Gref` reads `--`
    when the model has no reference to estimate. `closed-loop` is `(I + S P)⁻¹ A`, the
    steady-state plant under the optimal policy — the one comparison that is invariant
    to the cost scale, and so the best single summary of whether the fit found the same
    control *problem*; it reads `--` when the fitted cost admits no stabilizing Riccati
    solution, which is itself a result.

    Recovery results with a reference also carry a `gauge` audit. Tables print
    its coordinate branch whenever the emission basis moved; the dedicated
    reference-identification and smoulder-Gref sections additionally print
    reference-geometry and design-rank diagnostics. The audit estimates an
    orthogonal state-basis map from the fitted and true emission loadings, then
    apply that same map to `A`, `S`, every reward-specific running/terminal
    `Qc`, the closed loop, and `Gref`. Centred contrasts and pairwise target
    distances remove the separate common-origin gauge; the latter are also
    rotation-invariant. The augmented-design and translation nullities say
    whether that origin is identified at all. `G'G` is rotation-invariant but
    is not invariant to a common target translation.

    `Δelbo` is the fit's ELBO minus the ELBO at the generating parameters: a few nats is
    ordinary finite-sample slack, a large positive number means the model prefers
    something other than the truth. A `Δelbo` that grows roughly *linearly* in the
    number of trials is the signature of misspecification rather than slack — compare
    the rows of experiment 3. `creep` is the mean ELBO gain per iteration over the last
    tenth of the run, and stands in for a converged flag: with `tol = 1e-10` these fits
    essentially never stop on their tolerance, so a flag built on that would read `NO`
    everywhere and say nothing. Below ~1e-3 nats/iteration the answer has stopped
    moving; above ~1e-1 the row is reporting the iteration budget as much as the data.

    In the switching table the tail changes: `γ acc` is the balanced accuracy of the MAP
    discrete state, `(truth)` is the same quantity at the generating parameters — a
    reference, not a bound, since a fit may move the free state and the transition
    matrix to suit the data while that holds every one of them fixed. `onset` is the
    mean absolute switch-time error in timesteps. `stay` is the fitted self-transition
    probability, and the quickest tell for a collapsed fit: everything in one state
    reports a `stay` near 1 and a `γ acc` near 0.5.

    Section 1 draws from the model, where recovery is a well-posed estimation question.
    That includes its terminal row: the terminal factor is a conditioning event, `rand`
    draws the conditioned path distribution, and the package's default objective
    (`condition_terminal = true`) is that same conditional likelihood. Fitting the joint
    instead (`condition_terminal = false`) on those draws is a selection effect rather
    than an estimator. Section 2 draws from the control problem's actual
    optimum, which is the case the model is for and the harder one; at `slack = 0` the
    agent is exactly optimal, its innovation is rank `n` and time-varying while the
    model's `Σ` is full rank and constant, and the row is allowed to fail outright.

    See `docs/dev/lqr/README.md` for what the current numbers say.
    """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
