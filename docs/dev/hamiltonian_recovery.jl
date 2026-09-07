#=============================================================================
Inverse-LQR parameter recovery — a standalone validation harness.

Run it:

    julia --project=docs/dev docs/dev/hamiltonian_recovery.jl
    julia --project=. -e 'include("docs/dev/hamiltonian_recovery.jl")'

This is not a test. Tests check that the machinery computes what it claims;
this checks what you can actually *learn* from data, which is a different and
softer question — how close the estimated cost gets, under which conditions,
and how that degrades. It prints a table you read, not an assertion that passes.

Two generative modes, one per section:

  * **the model itself** (`rand`) — trials drawn from the forward chain the
    smoother assumes. Recovery is a well-posed estimation question here, so
    this section is the one whose numbers should be small.
  * **the LQR optimum** (`simulate_lqr`) — trials from an agent that actually
    solves the control problem, optimally or nearly so. This is the case the
    model is *for*, and it is harder: an exactly optimal trajectory has a
    rank-`n`, time-varying innovation that a full-rank constant `Σ` contains
    only as a limit, so fitting one is a projection rather than estimation. The
    `slack` ladder walks from that degenerate corner back into the model.

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
    truth_model(; n, terminal, tsteps, ux_dim, observe_costate, tracking, drift)

The generating model. `A` is a mild contraction and `S`, `Qc` are scaled so the
symplectic spectral radius stays near 1 — a Hamiltonian matrix has reciprocal
eigenvalue pairs, so `ρ(M)^T` is how fast the model's own forward chain
diverges, and a sampler-based recovery check needs it modest.

`drift` puts a nonzero mixed-coordinate bias `h` in the truth. Without it `h` is
identically zero, and a zero parameter is not something a recovery metric can
say anything about — the correlation is undefined and the relative error has no
denominator. The `h` columns read `--` on every other row for exactly that
reason.
"""
function truth_model(;
    n::Int=4,
    terminal::Bool=false,
    tsteps::Int=20,
    ux_dim::Int=0,
    observe_costate::Bool=false,
    tracking::Bool=false,
    drift::Bool=false,
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
    # Deterministic, so the truth does not depend on where the rng happens to be.
    h = drift ? [fill(0.03, n); collect(range(-0.04, 0.04; length=n))] : nothing
    return HamiltonianStateModel(
        A,
        S,
        qcs,
        Σ;
        schedule=sched,
        terminal=terminal,
        Σf=Matrix(0.02I, n, n),
        P0=Matrix(0.2I, d, d),
        h=h,
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
# Scoring
#
# Two numbers per parameter block, both computed on the *entries*: a relative
# RMSE (divided by the RMS of the true entries, so blocks of different
# magnitude are comparable) and a Pearson correlation (which ignores scale
# entirely, and so answers the different question of whether the estimate has
# the right shape). A block whose truth is ~0, or whose entries are constant,
# has no answer to either question and reports `NaN` rather than a number that
# would look like one.
#
# Statistics.jl is a dependency of the package but not of `docs/`, and these are
# six lines, so they live here rather than constraining how the script is run.
# ---------------------------------------------------------------------------

_mean(v) = sum(v) / length(v)
_var(v) = (m=_mean(v); sum(abs2, v .- m) / length(v))

function _cor(a, b)
    va, vb = _var(a), _var(b)
    (va <= 1e-24 || vb <= 1e-24) && return NaN
    return _mean((a .- _mean(a)) .* (b .- _mean(b))) / sqrt(va * vb)
end

"""
    _entries(M; sym) -> Vector

The independent entries of a block: the upper triangle (diagonal included) of a
symmetric matrix, every entry otherwise. Counting a symmetric matrix's
off-diagonals twice would weight them double in the RMSE and inflate the
correlation's sample size for free.
"""
function _entries(M::AbstractMatrix; sym::Bool=false)
    return sym ? [M[i, j] for j in axes(M, 2) for i in 1:j] : vec(collect(M))
end
_entries(v::AbstractVector; sym::Bool=false) = collect(v)

"""
    score(fit, ref; sym) -> (rmse, corr)

Relative RMSE and Pearson correlation of one parameter block against its truth.
"""
function score(fit, ref; sym::Bool=false)
    f = _entries(fit; sym=sym)
    r = _entries(ref; sym=sym)
    scale = sqrt(_mean(abs2.(r)))
    rmse = scale > 1e-12 ? sqrt(_mean(abs2.(f .- r))) / scale : NaN
    return (rmse=rmse, corr=_cor(f, r))
end

const NOSCORE = (rmse=NaN, corr=NaN)

"""
    score_worst(fits, refs; sym) -> (rmse, corr)

The worst regime, when a block is a vector of matrices: the largest relative
RMSE and the smallest correlation over the cost regimes. Pooling the regimes'
entries instead would flatter the fit — the truth's regimes differ in magnitude
by an order of magnitude, and a correlation across that spread is mostly
measuring which regime an entry came from.
"""
function score_worst(fits, refs; sym::Bool=false)
    ss = [score(fits[k], refs[k]; sym=sym) for k in eachindex(refs)]
    return (rmse=maximum(s.rmse for s in ss), corr=minimum(s.corr for s in ss))
end

"""
    closed_loop_or_nan(sm) -> Matrix or nothing

`(I + S P)⁻¹ A`, the steady-state closed-loop plant of regime 1. This is the
scale-invariant summary of the whole fit — `S → S/c`, `Q → cQ` leaves `S P`
alone — so it is the one comparison that needs no canonicalization to be
meaningful. Returns `nothing` when the Riccati iteration finds no stabilizing
solution, which an intermediate fit is entitled to do.
"""
function closed_loop_or_nan(sm)
    try
        return closed_loop_dynamics(sm; k=1)
    catch err
        err isa NumericalStabilityError || rethrow()
        return nothing
    end
end

"""
    compare(fit_sm, ref; known_plant) -> NamedTuple of (rmse, corr)

Every scored block, in the canonical scale. `A` is reported only when it was
estimated: frozen at the truth it is exactly right, and a column of `0.000/1.00`
is noise in the table rather than a result.

`S` *is* reported even when frozen, because the canonical rescaling divides it
by the fitted cost scale — so on a known-plant row its relative RMSE is exactly
the relative error in `tr(Qc[1])`, and its correlation is exactly 1.
"""
function compare(fit_sm, ref; known_plant::Bool)
    cl_f = closed_loop_or_nan(fit_sm)
    cl_r = closed_loop_or_nan(ref)
    return (
        Qc=score_worst(fit_sm.Qc, ref.Qc; sym=true),
        A=known_plant ? NOSCORE : score(fit_sm.A, ref.A),
        S=score(fit_sm.S, ref.S; sym=true),
        Sig=score(fit_sm.Σ, ref.Σ; sym=true),
        h=score(fit_sm.h, ref.h),
        cl=(cl_f === nothing || cl_r === nothing) ? NOSCORE : score(cl_f, cl_r),
    )
end

# ---------------------------------------------------------------------------
# One recovery run
# ---------------------------------------------------------------------------

"""
    simulate_trials(rng, truth, C, R, tsteps, ntrials; slack, process_noise, uxs)

Observations of an agent that actually solves the control problem: roll the
optimal trajectory out along the stable manifold and read it through the
emission. `slack` is the standard deviation of the perturbation on the agent's
own costate — how far from exactly optimal it is — and the agent *acts* on the
perturbed costate, so the slack moves the state too.
"""
function simulate_trials(
    rng, truth, C, R, tsteps::Int, ntrials::Int; slack::Float64, process_noise::Bool, uxs
)
    obs_dim = size(C, 1)
    L = cholesky(Symmetric(R)).L
    return [
        begin
            z = simulate_lqr(
                rng,
                truth,
                tsteps;
                costate_slack=slack,
                process_noise=process_noise,
                ux=(uxs === nothing ? nothing : uxs[i]),
            )
            C * z .+ L * randn(rng, obs_dim, tsteps)
        end for i in 1:ntrials
    ]
end

"""
    recover(; kwargs...) -> NamedTuple

Simulate from a known model, refit from a deliberately wrong start, and report
how close every parameter block came. Returns the per-block `(rmse, corr)`
scores in the canonical scale, the ELBO the fit reached, the ELBO at the
generating parameters, and whether EM stayed monotone.

`gen` selects the generative mode: `:rand` draws from the model's own forward
chain, `:lqr` rolls out the optimal trajectory with `simulate_lqr` and adds
`costate_slack` of suboptimality. `process_noise` applies to `:lqr` only.

`known_plant` is the usual inverse-optimal-control posture: the body is known,
the objective is not. Set it `false` to estimate both — the plant is then also
started away from the truth, since a warm-started `A` would make its recovery
column say nothing.
"""
function recover(;
    n::Int=4,
    tsteps::Int=20,
    ntrials::Int=120,
    gen::Symbol=:rand,
    costate_slack::Float64=0.1,
    process_noise::Bool=true,
    terminal::Bool=false,
    tracking::Bool=false,
    drift::Bool=false,
    free_C::Bool=false,
    observe_costate::Bool=false,
    known_plant::Bool=true,
    obs_noise::Float64=0.05,
    max_iter::Int=250,
    seed::Int=1,
)
    gen in (:rand, :lqr) || throw(ArgumentError("gen must be :rand or :lqr; got :$gen"))
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
        drift=drift,
    )
    obs_dim = free_C ? max(2n, 6) : n
    C = emission(n, obs_dim; free_C=free_C, observe_costate=observe_costate, rng=rng)
    R = Matrix(obs_noise * I, obs_dim, obs_dim)
    gen_lds = LinearDynamicalSystem(
        truth, GaussianObservationModel(copy(C), copy(R), zeros(obs_dim))
    )

    uxs = ux_dim > 0 ? [repeat(randn(rng, n), 1, tsteps) for _ in 1:ntrials] : nothing
    #=
    `rand` draws from the model itself, against which recovery is a well-posed
    estimation question. `simulate_lqr` draws from the control problem's actual
    optimum, which is the case the model is *for* and a strictly harder one: at
    `costate_slack = 0` the trajectory's innovation is rank `n` and lives on the
    graph of the Riccati map, so the fitted full-rank `Σ` is a projection onto
    the model rather than an estimate within it.
    =#
    ys = if gen === :rand
        last(rand(rng, gen_lds, fill(tsteps, ntrials); ux=uxs))
    else
        simulate_trials(
            rng,
            truth,
            C,
            R,
            tsteps,
            ntrials;
            slack=costate_slack,
            process_noise=process_noise,
            uxs=uxs,
        )
    end
    truth_elbo = elbo(gen_lds, ys; ux=uxs)

    # Refit from a wrong cost.
    fit_sm = truth_model(;
        n=n,
        terminal=terminal,
        tsteps=tsteps,
        ux_dim=ux_dim,
        observe_costate=observe_costate,
        tracking=tracking,
        drift=drift,
    )
    for Q in fit_sm.Qc
        Q .= Matrix(0.4I, n, n)
    end
    fit_sm.Σ .= Matrix(0.05I, d, d)
    fit_sm.h .= 0
    #=
    When the plant is estimated, start it wrong too. Left at the truth it would
    simply stay there — EM has no reason to move a parameter it starts at the
    optimum of — and its recovery column would report the initialization rather
    than the fit.
    =#
    if !known_plant
        fit_sm.A .= 0.90 .* truth.A .+ Matrix(0.03I, n, n)
        fit_sm.S .= 1.5 .* truth.S
    end
    fit_sm.fit_flags = HamiltonianFitFlags(; A=(!known_plant), S=(!known_plant), Gref=false)
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
    return (
        scores=compare(fit_sm, ref; known_plant=known_plant),
        elbo=elbos[end],
        truth_elbo=truth_elbo,
        iters=length(elbos),
        monotone=minimum(diff(elbos)) > -1e-8,
        rho=maximum(abs, eigvals(symplectic_matrix(truth))),
        defect=symplectic_defect(fit_sm),
    )
end

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

const BLOCKS = (:Qc, :A, :S, :Sig, :h, :cl)
const BLOCK_NAMES = ("Qc", "A", "S", "Σ", "h", "closed-loop")
const RULE = 129

_rmse_str(x) = isnan(x) ? "   --" : (x >= 9.9995 ? ">9.99" : @sprintf("%5.3f", x))
_corr_str(x) = isnan(x) ? "   --" : @sprintf("%5.2f", x)

function cell(s)
    isnan(s.rmse) && isnan(s.corr) && return lpad("--", 11)
    return string(_rmse_str(s.rmse), "/", _corr_str(s.corr))
end

function header()
    @printf(
        "%-30s %11s %11s %11s %11s %11s %11s %8s %5s %5s %5s\n",
        "condition",
        BLOCK_NAMES...,
        "Δelbo",
        "iters",
        "mono",
        "ρ(M)"
    )
    @printf(
        "%-30s %11s %11s %11s %11s %11s %11s %8s %5s %5s %5s\n",
        "",
        fill("rmse/corr", 6)...,
        "vs truth",
        "",
        "",
        ""
    )
    return println("-"^RULE)
end

function section(title::String)
    println()
    println(title)
    println("=" ^ length(title))
    return header()
end

function report(label, r; note::String="")
    @printf(
        "%-30s %11s %11s %11s %11s %11s %11s %8.1f %5d %5s %5.2f %s\n",
        label,
        (cell(getfield(r.scores, b)) for b in BLOCKS)...,
        r.elbo - r.truth_elbo,
        r.iters,
        r.monotone ? "yes" : "NO",
        r.rho,
        note
    )
    return nothing
end

"""
    run_row(label, kwargs...; note)

One table row, with failures reported rather than thrown. The degenerate corners
this harness deliberately visits — an exactly optimal agent above all — can
drive the fitted `Σ` singular, and a table that stops at the first such row
tells you less than one that prints it and carries on.
"""
function run_row(label; note::String="", kwargs...)
    try
        report(label, recover(; kwargs...); note=note)
    catch err
        @printf("%-30s  %s\n", label, "FAILED: $(typeof(err))")
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

function model_section(; free_C::Bool, quick::Bool, ntr::Int)
    section("Section 1 — trials drawn from the model (`rand`)")
    run_row("baseline, known plant"; ntrials=ntr, free_C=free_C)
    run_row("  half the trials"; ntrials=ntr ÷ 2, free_C=free_C)
    run_row("  double the trials"; ntrials=2ntr, free_C=free_C)
    run_row("  longer trials (T = 40)"; ntrials=ntr, tsteps=40, free_C=free_C)
    run_row("  noisier observations"; ntrials=ntr, obs_noise=0.25, free_C=free_C)
    run_row("plant estimated too"; ntrials=ntr, known_plant=false, free_C=free_C)
    run_row("costate observed"; ntrials=ntr, observe_costate=true, free_C=true)
    #=
    The terminal row is expected to look bad, and it is worth saying why rather
    than letting a reader take it for a defect. The terminal factor is a
    conditioning event: `rand` draws `p(z, y | y_term = 0)` while the fitted
    objective is `p(y, y_term = 0)`, and the two differ by `p(y_term = 0 | θ)`.
    Maximizing the latter on data drawn from the former is a selection effect,
    not an unbiased estimator, so the fit legitimately beats the truth's ELBO by
    a lot. `simulate_lqr` has no such problem — it satisfies the terminal
    condition by construction — so section 2's terminal row is the honest one.
    =#
    run_row(
        "terminal cost";
        ntrials=ntr,
        terminal=true,
        free_C=free_C,
        note="<- conditioning event; see note",
    )
    run_row("tracking a reference"; ntrials=ntr, tracking=true, free_C=free_C)
    run_row("affine drift (h ≠ 0)"; ntrials=ntr, drift=true, free_C=free_C)
    if !quick
        run_row("n = 3"; n=3, ntrials=ntr, free_C=free_C)
        for seed in 2:4
            run_row("baseline, seed $seed"; ntrials=ntr, seed=seed, free_C=free_C)
        end
    end
    return nothing
end

function lqr_section(; free_C::Bool, quick::Bool, ntr::Int)
    section("Section 2 — trials drawn from the LQR optimum (`simulate_lqr`)")
    #=
    The slack ladder is the axis that matters, and the answer it gives is not
    the comfortable one. At `slack = 0` the agent is exactly optimal and its
    costate is a deterministic function of its state, so the latent innovation
    is rank `n` while the model's `Σ` is full rank and constant: the
    maximum-likelihood cost need not be the generating one. Each rung up makes
    the innovation full rank — but not white. The agent acts on its own
    perturbed costate, so the mixed-coordinate residual is `ν_t − Aᵀν_{t+1}`,
    and the ladder trades a degenerate residual for a serially correlated one
    rather than walking into the model. Read it against the `no process noise`
    and `terminal condition` rows below, which each remove one competing term.
    =#
    for slack in (0.0, 0.02, 0.10, 0.30)
        note = slack == 0.0 ? "<- exactly optimal; outside the model" : ""
        run_row(
            @sprintf("slack = %.2f", slack);
            ntrials=ntr,
            gen=:lqr,
            costate_slack=slack,
            free_C=free_C,
            note=note,
        )
    end
    run_row(
        "  no process noise";
        ntrials=ntr,
        gen=:lqr,
        costate_slack=0.10,
        process_noise=false,
        free_C=free_C,
    )
    run_row(
        "  terminal condition";
        ntrials=ntr,
        gen=:lqr,
        costate_slack=0.10,
        terminal=true,
        free_C=free_C,
    )
    run_row(
        "  longer trials (T = 40)";
        ntrials=ntr,
        tsteps=40,
        gen=:lqr,
        costate_slack=0.10,
        free_C=free_C,
    )
    run_row(
        "  half the trials"; ntrials=ntr ÷ 2, gen=:lqr, costate_slack=0.10, free_C=free_C
    )
    run_row(
        "  double the trials"; ntrials=2ntr, gen=:lqr, costate_slack=0.10, free_C=free_C
    )
    run_row(
        "plant estimated too";
        ntrials=ntr,
        gen=:lqr,
        costate_slack=0.10,
        known_plant=false,
        free_C=free_C,
    )
    if !quick
        run_row(
            "costate observed";
            ntrials=ntr,
            gen=:lqr,
            costate_slack=0.10,
            observe_costate=true,
            free_C=true,
        )
        run_row(
            "affine drift (h ≠ 0)";
            ntrials=ntr,
            gen=:lqr,
            costate_slack=0.10,
            drift=true,
            free_C=free_C,
        )
    end
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

    ntr = quick ? 40 : 120
    model_section(; free_C=free_C, quick=quick, ntr=ntr)
    lqr_section(; free_C=free_C, quick=quick, ntr=ntr)

    println()
    println("""
    Reading these tables
    --------------------
    Each parameter block gets `rmse/corr`, both computed on its independent entries
    (the upper triangle for a symmetric block) after canonical rescaling:

      rmse  relative RMSE — the RMSE divided by the RMS of the true entries, so
            blocks of different magnitude are on one scale. 0 is exact.
      corr  Pearson correlation across entries. It ignores scale entirely, so it
            answers the separate question of whether the estimate has the right
            *shape*; a low rmse with a high corr means a scale error, the reverse
            means a structural one.

    `Qc` is the worst of the cost regimes, not their average. `A` reads `--` when
    the plant is known, because it is then frozen at the truth and a perfect score
    would be reporting the initialization. `S` is frozen too on those rows, but the
    canonical rescaling divides it by the fitted cost scale — so its rmse there is
    exactly the relative error in `tr(Qc[1])`, and its corr is exactly ±1. A corr of
    `-1.00` is worth stopping on: the costate's *sign* is unidentified along with
    its scale, `:trace` canonicalization pins it by making `tr(Qc[1])` positive, and
    a `-1` therefore says the fit landed on a cost of the opposite sign. `h` reads
    `--` wherever the truth's `h` is zero: a zero parameter has no relative error
    and no correlation. `closed-loop` is `(I + S P)⁻¹ A`, the steady-state plant
    under the optimal policy — the one comparison that is invariant to the cost
    scale, and so the best single summary of whether the fit found the same control
    problem; it reads `--` when the fitted cost admits no stabilizing Riccati
    solution, which is itself a result. `Δelbo` is the fit's ELBO minus the ELBO at
    the generating parameters: a few nats is ordinary finite-sample slack, a large
    positive number means the model prefers something other than the truth. A
    `Δelbo` that grows roughly *linearly* in the number of trials is the signature
    of misspecification rather than slack — compare the half/double rows.

    Section 1 draws from the model, where recovery is a well-posed estimation
    question; its terminal row is the exception, and is expected to look bad. The
    terminal factor is a conditioning event, so `rand` draws the conditioned path
    distribution while the objective is the joint, and maximizing the latter on the
    former is a selection effect rather than an estimator.

    Section 2 draws from the control problem's actual optimum, which is the case
    the model is for and the harder one. At `slack = 0` the agent is exactly
    optimal: its costate is a deterministic function of its state, so its innovation
    is rank `n` and time-varying while the model's `Σ` is full rank and constant.
    Fitting that is a projection onto the model rather than estimation within it,
    and the row is allowed to fail outright.

    Do not expect the slack ladder to climb out of that. Slack makes the data
    full-rank, but it does not make it the model's own noise: the agent perturbs its
    costate by `ν_t` and acts on the result, so the residual the mixed-coordinate
    model sees is `ν_t − Aᵀν_{t+1}`, an MA(1) process, where a white innovation is
    what `Σ` assumes. Turning the slack up trades a degenerate residual for a
    serially correlated one, and the fitted cost absorbs the difference. That the
    `no process noise` and `terminal condition` rows do better than the plain ladder
    is the same point from the other side — each removes one of the two competing
    noise sources, or pins the endpoint the free-endpoint Riccati sweep otherwise
    leaves the stationary model unable to represent.

    What sharpens identification, roughly in order: observing the costate, a
    terminal condition, behaviour that is noisily rather than exactly optimal, a
    cost that changes within the trial, and more trials. What does not: more data
    alone, when the residual is misspecified.
    """)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; free_C=("--free-C" in ARGS), quick=("--quick" in ARGS))
end
