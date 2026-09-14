#=============================================================================
The sweeps.

Five experiments, each answering one question, each printing a table and writing
figures:

  1. `overview`   — the descriptive pass: what recovery looks like across the
                    structural and generative conditions, one row each.
  2. `design`     — the factorial the whole harness is pointed at: how much do a
                    terminal condition, a within-trial cost change, and the
                    number of distinct reference targets each buy?
  3. `scale`      — how much of the remaining error is sampling error. Trials and
                    trial length, so a floor can be told from a slope.
  4. `procedure`  — the fitting procedure rather than the model: iterations,
                    restarts, initialization, inner-loop budget.
  5. `switching`  — the `SLDS` with one free and one LQR state, and the two
                    questions it asks separately: the LQR state's parameters,
                    and the discrete path.

Sizes come from `tier`, so the same code runs as a 90-second smoke test or a
two-hour study.
=============================================================================#

"""
    tier(name) -> NamedTuple

Sample sizes per tier. `:quick` is a smoke test and its numbers should not be
read as results — one seed per cell cannot separate a condition from a draw.
`:default` is sized so the differences the sweeps look for are larger than the
seed-to-seed spread; `:full` adds seeds and trials rather than new conditions.
"""
function tier(name::Symbol)
    name === :quick && return (
        n=3,
        tsteps=20,
        ntrials=60,
        seeds=1:1,
        max_iter=120,
        trial_ladder=[25, 100, 400],
        iter_ladder=[50, 200, 800],
        slds=(n=2, tsteps=40, ntrials=60, seeds=1:1, max_iter=30),
    )
    #=
    The switching section dominates `:full`: one `SLDS` fit is two orders of
    magnitude more expensive than one single-system fit, so its seeds and trials
    are held well below the parametric sweeps' rather than scaled with them.
    =#
    name === :full && return (
        n=4,
        tsteps=30,
        ntrials=800,
        seeds=1:5,
        max_iter=600,
        trial_ladder=[25, 50, 100, 200, 400, 800, 1600],
        iter_ladder=[25, 50, 100, 250, 500, 1000, 2000],
        slds=(n=2, tsteps=48, ntrials=250, seeds=1:2, max_iter=70),
    )
    name === :default && return (
        n=3,
        tsteps=25,
        ntrials=400,
        seeds=1:3,
        max_iter=250,
        trial_ladder=[25, 50, 100, 200, 400, 800],
        iter_ladder=[25, 50, 100, 250, 500, 1000],
        slds=(n=2, tsteps=40, ntrials=150, seeds=1:1, max_iter=50),
    )
    return throw(ArgumentError("unknown tier :$name"))
end

#=
The three structural conditions the design experiment contrasts, and the series
order every figure in this file uses for them. Three, not more: past three the
palette cannot keep every pair separable under color-vision deficiency on a
form where any two series can end up adjacent.
=#
const STRUCTURE_SERIES = (
    ("running only", (terminal=false, onset=1)),
    ("+ terminal", (terminal=true, onset=1)),
    ("+ term + delay", (terminal=true, onset=0)),   # onset = 0 → a mid-trial onset
)

"""Resolve the sentinel `onset = 0` to "a delay covering the first third"."""
_onset(o::Int, tsteps::Int) = o == 0 ? max(2, tsteps ÷ 3) : o

structure_kwargs(spec, tsteps) =
    (terminal=spec.terminal, onset=_onset(spec.onset, tsteps))

# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------

function experiment_overview(cfg; figures::Bool=true, free_C::Bool=false)
    n, T, N, mi = cfg.n, cfg.tsteps, cfg.ntrials, cfg.max_iter
    onset = _onset(0, T)
    base = (; n=n, tsteps=T, ntrials=N, max_iter=mi, free_C=free_C)

    section("1 — Trials drawn from the model (`rand`)")
    table_header()
    keep = Dict{String,Any}()
    keep["model baseline"] = run_row(report, "baseline, known plant", recover, (; base...))
    run_row(report, "  noisier observations", recover, (; base..., obs_noise=0.25))
    run_row(report, "  costate observed", recover, (; base..., observe_costate=true))
    run_row(report, "plant estimated too", recover, (; base..., known_plant=false))
    #=
    The terminal row under `rand` is expected to look bad, and it is worth
    saying why rather than letting a reader take it for a defect. The terminal
    factor is a conditioning event: `rand` draws `p(z, y | y_term = 0)` while the
    fitted objective is `p(y, y_term = 0)`, and the two differ by
    `p(y_term = 0 | θ)`. Maximizing the latter on data drawn from the former is a
    selection effect, not an unbiased estimator, so the fit legitimately beats
    the truth's ELBO by a lot. `simulate_lqr` has no such problem — it satisfies
    the terminal condition by construction — so section 2's terminal rows are
    the honest ones.
    =#
    run_row(
        report,
        "terminal cost",
        recover,
        (; base..., terminal=true);
        note="<- conditioning event; see notes",
    )
    run_row(report, "delay epoch (2 regimes)", recover, (; base..., onset=onset))
    run_row(report, "tracking 1 reference", recover, (; base..., nref=1))
    run_row(report, "tracking 4 references", recover, (; base..., nref=4))
    keep["model full"] = run_row(
        report,
        "4 refs + terminal + delay",
        recover,
        (; base..., nref=4, terminal=true, onset=onset),
    )
    #=
    `h` needs its own header rather than a ragged row under the one above: a
    seven-column row printed under a seven-column header with different labels is
    worse than no row at all. It is also the only place `h` is scorable — a truth
    whose drift is identically zero has no relative error and no correlation to
    report, which is why every other row would read `--`.
    =#
    println()
    table_header((:Qc, :Gref, :S, :h, :cl))
    run_row(
        report,
        "affine drift (h ≠ 0)",
        recover,
        (; base..., drift=true);
        blocks=(:Qc, :Gref, :S, :h, :cl),
    )
    run_row(
        report,
        "  with 4 references",
        recover,
        (; base..., drift=true, nref=4);
        blocks=(:Qc, :Gref, :S, :h, :cl),
    )

    section("2 — Trials drawn from the LQR optimum (`simulate_lqr`)")
    table_header()
    lqr = (; base..., gen=:lqr)
    #=
    The slack ladder is the axis that matters, and the answer it gives is not
    the comfortable one. At `slack = 0` the agent is exactly optimal and its
    costate is a deterministic function of its state, so the latent innovation
    is rank `n` while the model's `Σ` is full rank and constant: the
    maximum-likelihood cost need not be the generating one. Each rung up makes
    the innovation full rank — but not white. The agent acts on its own
    perturbed costate, so the mixed-coordinate residual is `ν_t − Aᵀν_{t+1}`,
    and the ladder trades a degenerate residual for a serially correlated one
    rather than walking into the model.
    =#
    for slack in (0.0, 0.02, 0.10, 0.30)
        run_row(
            report,
            @sprintf("slack = %.2f", slack),
            recover,
            (; lqr..., costate_slack=slack);
            note=(slack == 0.0 ? "<- exactly optimal; outside the model" : ""),
        )
    end
    run_row(report, "  no process noise", recover, (; lqr..., process_noise=false))
    keep["agent baseline"] = run_row(
        report, "  terminal cost", recover, (; lqr..., terminal=true)
    )
    run_row(report, "  delay epoch (2 regimes)", recover, (; lqr..., onset=onset))
    run_row(report, "  tracking 1 reference", recover, (; lqr..., nref=1))
    run_row(report, "  tracking 4 references", recover, (; lqr..., nref=4))
    keep["agent full"] = run_row(
        report,
        "4 refs + terminal + delay",
        recover,
        (; lqr..., nref=4, terminal=true, onset=onset),
    )
    run_row(report, "  costate observed", recover, (; lqr..., observe_costate=true))
    run_row(report, "plant estimated too", recover, (; lqr..., known_plant=false))

    figures || return keep
    for (name, res) in keep
        res === nothing && continue
        slug = replace(name, " " => "_")
        matrix_figure(res, "params_$slug"; title="Parameter recovery — $name")
        scatter_figure(res, "entries_$slug"; title="Entrywise recovery — $name")
    end
    traces = [
        (k, keep[k].elbos .- keep[k].truth_elbo) for
        k in sort(collect(keys(keep))) if keep[k] !== nothing
    ]
    isempty(traces) || elbo_figure(
        "elbo_traces"; traces=traces[1:min(3, end)], title="EM traces against the truth"
    )
    return keep
end

# ---------------------------------------------------------------------------
# 2. Design: terminal condition, cost epochs, number of references
# ---------------------------------------------------------------------------

"""
    experiment_design(cfg; ...)

The factorial this harness is pointed at. Along one axis, how many distinct
reference targets the input codes for (`0` is no reference at all); along the
other, whether the trial carries a terminal condition and a within-trial cost
change.

Why this pairing and not two separate sweeps: the model's own documentation says
that with one cost regime and no terminal factor, `B_u`'s costate rows and
`−Q₁ G_r` both map the input into the costate and are not separately identified,
and that several cost regimes or a terminal factor separate them. Freezing `B_u`
removes that particular collision, but a weaker version of it survives — with a
single reference the input never varies, so `−Q₁ G_r u` is a constant and
competes with the affine drift `h`. Two or more targets identify the *contrasts*
between reference vectors; something that breaks the within-trial symmetry
identifies their common level. The factorial is the experiment that says whether
that is true and by how much.
"""
function experiment_design(cfg; gen::Symbol=:lqr, figures::Bool=true, free_C::Bool=false)
    n, T, N, mi = cfg.n, cfg.tsteps, cfg.ntrials, cfg.max_iter
    refs = [0, 1, 2, 4, 8]
    grid = Any[]
    for (si, (slabel, spec)) in enumerate(STRUCTURE_SERIES), r in refs
        push!(
            grid,
            (si, r) => (;
                n=n,
                tsteps=T,
                ntrials=N,
                max_iter=mi,
                gen=gen,
                free_C=free_C,
                nref=r,
                structure_kwargs(spec, T)...,
            ),
        )
    end
    res = cells(recover, grid; seeds=cfg.seeds)

    label = gen === :lqr ? "agent (`simulate_lqr`)" : "model (`rand`)"
    section(
        "2 — Design factorial: terminal condition × reference targets — $label" *
        "\n     (every cell is the median over $(nseeds(cfg.seeds)))",
    )
    table_header()
    for (si, (slabel, _)) in enumerate(STRUCTURE_SERIES), r in refs
        a = aggregate(res[(si, r)])
        a === nothing && continue
        report(rpad("$(r) refs, $slabel", LBLW), a)
    end

    figures || return res
    panels = Any[]
    for (mlabel, path, xs, keyfn) in (
        ("cost  Qc (running)", r -> r.scores.Qc.rmse, refs, (si, r) -> (si, r)),
        ("reference map  Gref", r -> r.scores.Gref.rmse, refs[2:end], (si, r) -> (si, r)),
        ("closed-loop plant", r -> r.scores.cl.rmse, refs, (si, r) -> (si, r)),
    )
        series = Any[]
        for (si, (slabel, _)) in enumerate(STRUCTURE_SERIES)
            ms = [center(metric(res[keyfn(si, r)], path)) for r in xs]
            push!(series, (slabel, [m[1] for m in ms], ([m[2] for m in ms], [m[3] for m in ms])))
        end
        push!(panels, (mlabel, xs, series))
    end
    sweep_figure(
        "design_$(gen)";
        panels=panels,
        xlabel="distinct reference targets",
        xticks=(refs, string.(refs)),
        title="What sharpens the cost — $label ($(nseeds(cfg.seeds)), N = $N)",
    )
    return res
end

"""
    experiment_reference_identification(cfg; ...)

The sharpest single claim this harness can check, isolated.

The reference enters the model only through the costate half of the affine term,
as `−Q_k G_r u_t`. With one target the input never varies within the dataset, so
that term is a *constant* — and the model already has a free constant in the
costate half of `h`. The two are then not separately identified, and no amount of
data separates them. With two or more targets the term varies across trials, and
the contrasts between reference vectors are identified whatever `h` does.

So: `Gref` recovery against the number of targets, once with `h` estimated and
once with `h` frozen at zero. If the account above is right, the two series
should be far apart at one target and on top of each other by four — and the
gap, not the level, is the result.

This runs at a loose `Σ_λλ`, against the harness default, because the default is
what makes the reference unidentifiable in the first place; see the comment on
the grid below.
"""
function experiment_reference_identification(
    cfg; gen::Symbol=:rand, figures::Bool=true, free_C::Bool=false
)
    n, T, N, mi = cfg.n, cfg.tsteps, cfg.ntrials, cfg.max_iter
    refs = [1, 2, 4, 8]
    grid = Any[]
    for fh in (true, false), r in refs
        push!(
            grid,
            (fh, r) => (;
                n=n,
                tsteps=T,
                ntrials=N,
                max_iter=mi,
                gen=gen,
                free_C=free_C,
                nref=r,
                free_h=fh,
                #=
                A *loose* costate innovation here, against the harness default,
                and the reason is the question being asked. The default `1e-4` is
                what makes the cost recoverable, and it does that by taking the
                costate out of play — but the reference lives only in the costate
                half of the affine term, so under the default neither `Gref` nor
                the `h` it competes with is identified and the contrast this
                experiment exists to measure is a comparison of two nulls. The
                confound is a structural claim about the model; measuring it
                needs a regime where the parameter is identifiable at all.
                =#
                sig0_costate=5e-2,
            ),
        )
    end
    res = cells(recover, grid; seeds=cfg.seeds)

    #=
    The truth's `h` is zero here, so freezing the fit's `h` at zero freezes it at
    the *true* value — it concedes nothing about the reference, which is what
    makes this a clean test rather than a trade. The `h` column is omitted for
    the same reason: a zero parameter has no relative error and no correlation.
    =#
    section(
        "2b — Is the reference identified? `Gref` against the affine drift `h`" *
        "\n     (loose Σ_λλ = 5e-2 throughout, not the harness default; see the" *
        " docstring)" *
        "\n     (every cell is the median over $(nseeds(cfg.seeds)))",
    )
    blocks = (:Qc, :Gref, :S, :Sig, :cl)
    table_header(blocks)
    for fh in (true, false), r in refs
        a = aggregate(res[(fh, r)])
        a === nothing && continue
        report(
            rpad("$(r) refs, h $(fh ? "estimated" : "frozen at 0")", LBLW), a; blocks=blocks
        )
    end

    figures || return res
    function ser(path)
        return [
            begin
                ms = [center(metric(res[(fh, r)], path)) for r in refs]
                (
                    fh ? "h estimated" : "h frozen at 0",
                    [m[1] for m in ms],
                    ([m[2] for m in ms], [m[3] for m in ms]),
                )
            end for fh in (true, false)
        ]
    end
    sweep_figure(
        "identify_reference_$(gen)";
        panels=[
            ("reference map  Gref", refs, ser(r -> r.scores.Gref.rmse)),
            ("cost  Qc (running)", refs, ser(r -> r.scores.Qc.rmse)),
            ("closed-loop plant", refs, ser(r -> r.scores.cl.rmse)),
        ],
        xlabel="distinct reference targets",
        xticks=(refs, string.(refs)),
        title="One reference is confounded with the drift; several are not — " *
              "$(gen === :lqr ? "agent" : "model")",
    )
    return res
end

# ---------------------------------------------------------------------------
# 3. Scale: is the residual error sampling error?
# ---------------------------------------------------------------------------

"""
    experiment_scale(cfg; ...)

Trials and trial length, on a log axis, for each structural condition.

The shape is the answer, not the level. An error that falls like `N^{-1/2}` is
sampling error and more data fixes it; an error that flattens is an
identification floor and more data will not. Reading one condition's number
without its slope is how a harness talks itself into believing a model is
identified when it is only under-powered.
"""
function experiment_scale(cfg; gen::Symbol=:lqr, figures::Bool=true, free_C::Bool=false)
    n, T, mi = cfg.n, cfg.tsteps, cfg.max_iter
    Ns = cfg.trial_ladder
    Ts = [T ÷ 2, T, 2T, 4T]
    grid = Any[]
    for (si, (_, spec)) in enumerate(STRUCTURE_SERIES)
        for N in Ns
            push!(
                grid,
                (:N, si, N) => (;
                    n=n,
                    tsteps=T,
                    ntrials=N,
                    max_iter=mi,
                    gen=gen,
                    free_C=free_C,
                    nref=4,
                    structure_kwargs(spec, T)...,
                ),
            )
        end
        for Ti in Ts
            push!(
                grid,
                (:T, si, Ti) => (;
                    n=n,
                    tsteps=Ti,
                    ntrials=cfg.ntrials,
                    max_iter=mi,
                    gen=gen,
                    free_C=free_C,
                    nref=4,
                    structure_kwargs(spec, Ti)...,
                ),
            )
        end
    end
    res = cells(recover, grid; seeds=cfg.seeds)

    section(
        "3 — How the error scales with data — $(gen === :lqr ? "agent" : "model")" *
        "\n     (trials below; the trial-length ladder is in the figure)",
    )
    table_header()
    for (si, (slabel, _)) in enumerate(STRUCTURE_SERIES), N in Ns
        a = aggregate(res[(:N, si, N)])
        a === nothing && continue
        report(rpad("N = $N, $slabel", LBLW), a)
    end

    figures || return res
    for (axis, xs, xlabel, tag) in
        ((:N, Ns, "trials", "trials"), (:T, Ts, "timesteps per trial", "horizon"))
        panels = Any[]
        for (mlabel, path) in (
            ("cost  Qc (running)", r -> r.scores.Qc.rmse),
            ("reference map  Gref", r -> r.scores.Gref.rmse),
            ("closed-loop plant", r -> r.scores.cl.rmse),
        )
            series = Any[]
            for (si, (slabel, _)) in enumerate(STRUCTURE_SERIES)
                ms = [center(metric(res[(axis, si, x)], path)) for x in xs]
                push!(
                    series,
                    (slabel, [m[1] for m in ms], ([m[2] for m in ms], [m[3] for m in ms])),
                )
            end
            push!(panels, (mlabel, xs, series))
        end
        sweep_figure(
            "scale_$(tag)_$(gen)";
            panels=panels,
            xlabel=xlabel,
            logx=true,
            title="Does more data fix it? — " *
                  "$(gen === :lqr ? "agent" : "model") ($(nseeds(cfg.seeds)))",
        )
    end
    return res
end

# ---------------------------------------------------------------------------
# 4. Procedure: the fitting code rather than the model
# ---------------------------------------------------------------------------

"""
    experiment_procedure(cfg; ...)

Everything here holds the generative model fixed and varies how it is fitted.
The question is which of the knobs a user actually controls moves recovery, and
by how much relative to the structural features of experiment 2.

The iteration ladder comes first because the baseline sweeps all stop on
`max_iter` rather than on their tolerance, which means every number they report
is partly a statement about the budget. A curve that is still falling at the
right edge says the fits are iteration-limited; one that flattens says they are
not, and the remaining error is the model's.
"""
function experiment_procedure(cfg; gen::Symbol=:lqr, figures::Bool=true, free_C::Bool=false)
    n, T, N = cfg.n, cfg.tsteps, cfg.ntrials
    onset = _onset(0, T)
    base = (;
        n=n,
        tsteps=T,
        ntrials=N,
        gen=gen,
        free_C=free_C,
        nref=4,
        terminal=true,
        onset=onset,
    )
    iters = cfg.iter_ladder
    grid = [(:iter, m) => (; base..., max_iter=m) for m in iters]
    res = cells(recover, grid; seeds=cfg.seeds)

    mi = cfg.max_iter
    levers = [
        ("baseline (cold start)", (; base..., max_iter=mi)),
        ("4 random restarts", (; base..., max_iter=mi, restarts=4)),
        ("warm start at the truth", (; base..., max_iter=mi, init=:warm)),
        ("inner M-step 25 iters", (; base..., max_iter=mi, mstep_iters=25)),
        ("inner M-step 400 iters", (; base..., max_iter=mi, mstep_iters=400)),
        ("loose tol (1e-6)", (; base..., max_iter=mi, tol=1e-6)),
        ("affine drift h frozen at 0", (; base..., max_iter=mi, free_h=false)),
        ("plant estimated too", (; base..., max_iter=mi, known_plant=false)),
        ("4× the EM budget", (; base..., max_iter=4mi)),
        ("costate observed", (; base..., max_iter=mi, observe_costate=true)),
    ]
    lres = cells(recover, [l[1] => l[2] for l in levers]; seeds=cfg.seeds)

    section(
        "4 — What in the fitting procedure helps — $(gen === :lqr ? "agent" : "model")" *
        "\n     (4 references, terminal cost and a delay epoch throughout;" *
        " median over $(nseeds(cfg.seeds)))",
    )
    table_header()
    for m in iters
        a = aggregate(res[(:iter, m)])
        a === nothing && continue
        report(rpad("max_iter = $m", LBLW), a)
    end
    println()
    for (lab, _) in levers
        a = aggregate(lres[lab])
        a === nothing && continue
        report(rpad(lab, LBLW), a)
    end

    figures || return (iter=res, levers=lres)
    labs = [l[1] for l in levers]
    function one_series(ms)
        return (
            "relative RMSE", [m[1] for m in ms], ([m[2] for m in ms], [m[3] for m in ms])
        )
    end
    lever_means(path) = [center(metric(lres[l], path))[1] for l in labs]
    qc = [center(metric(res[(:iter, m)], r -> r.scores.Qc.rmse)) for m in iters]
    gr = [center(metric(res[(:iter, m)], r -> r.scores.Gref.rmse)) for m in iters]
    cl = [center(metric(res[(:iter, m)], r -> r.scores.cl.rmse)) for m in iters]
    sweep_figure(
        "procedure_iterations_$(gen)";
        panels=[
            ("cost  Qc (running)", iters, [one_series(qc)]),
            ("reference map  Gref", iters, [one_series(gr)]),
            ("closed-loop plant", iters, [one_series(cl)]),
        ],
        xlabel="EM iterations",
        logx=true,
        title="Is the answer iteration-limited? — $(gen === :lqr ? "agent" : "model")",
    )
    dot_figure(
        "procedure_levers_$(gen)";
        labels=labs,
        panels=[
            ("cost  Qc (running)", lever_means(r -> r.scores.Qc.rmse)),
            ("reference map  Gref", lever_means(r -> r.scores.Gref.rmse)),
            ("closed-loop plant", lever_means(r -> r.scores.cl.rmse)),
        ],
        title="Fitting-procedure levers, generative model held fixed",
    )
    return (iter=res, levers=lres)
end

# ---------------------------------------------------------------------------
# 4b. Initialization: where the fit starts
# ---------------------------------------------------------------------------

"""
    experiment_initialization(cfg; ...)

Where to start the fit, searched rather than asserted.

The harness's own default is a heuristic — start `Σ`'s costate block at `1e-4`,
three orders of magnitude below its state block — and a heuristic in a validation
harness is a claim that has not been checked. This experiment checks it and its
neighbourhood.

Why that particular knob is worth a whole experiment: on the optimal path the
costate is a deterministic function of the state, so the innovation an optimal
agent produces has rank `n` and the costate half of `Σ` should be near zero.
Start it isotropic and EM has `n` free directions in which the cost can drift
without costing the bound anything — which is where these fits go wrong, and why
a single number chosen before the first iteration can matter more than the
iteration budget.

Four readings, in increasing cost:

  * a ladder on the costate block, with `Σ` estimated and with `Σ` frozen at the
    start, which separates "a good place to begin" from "a constraint worth
    imposing";
  * a ladder on the state block, the same knob's uninteresting twin, as a
    control — if recovery moved as much along this axis the story would be about
    initialization in general rather than about the costate;
  * a ladder on the initial cost scale `q0`;
  * a two-axis grid over the costate block and `q0`, since the two are the
    settings most likely to interact — a cost started too large and a costate
    innovation started too free are the same mistake seen from two sides.
"""
function experiment_initialization(
    cfg; gen::Symbol=:lqr, figures::Bool=true, free_C::Bool=false
)
    n, T, N, mi = cfg.n, cfg.tsteps, cfg.ntrials, cfg.max_iter
    base = (;
        n=n,
        tsteps=T,
        ntrials=N,
        max_iter=mi,
        gen=gen,
        free_C=free_C,
        nref=4,
        terminal=true,
        onset=_onset(0, T),
    )
    cnoise = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2]
    snoise = [0.005, 0.02, 0.05, 0.2]
    q0s = [0.05, 0.2, 0.4, 1.0, 4.0]
    gridq = [0.1, 0.4, 1.6]
    gridc = [1e-5, 1e-3, 5e-2]

    grid = Any[]
    for fn in (true, false), c in cnoise
        push!(grid, (:cn, fn, c) => (; base..., sig0_costate=c, fit_noise=fn))
    end
    for v in snoise
        push!(grid, (:sn, v) => (; base..., sig0_state=v))
    end
    for q in q0s
        push!(grid, (:q0, q) => (; base..., q0=q))
    end
    res = cells(recover, grid; seeds=cfg.seeds)
    #=
    The grid gets fewer seeds than the ladders on purpose: it is there to show
    the shape of the surface and where its minimum sits, which a couple of seeds
    settle, and it costs `length(gridq) * length(gridc)` fits per seed.
    =#
    gseeds = cfg.seeds[1:min(2, length(cfg.seeds))]
    ggrid = [
        (:g, q, c) => (; base..., q0=q, sig0_costate=c) for q in gridq for c in gridc
    ]
    gres = cells(recover, ggrid; seeds=gseeds)

    #=
    Named combinations, run on *both* generators, because the answer differs
    between them and that difference is the finding. Drawing from the model,
    the costate is a quantity to be explained and a loose `Σ_λλ` is right;
    drawing from an optimal agent, the costate is a deterministic function of the
    state and a tight one is right. The harness's default serves the second,
    which is the case the model is for.
    =#
    combos = [
        ("loose Σ_λλ = 5e-2", (; sig0_costate=5e-2)),
        ("tight Σ_λλ = 1e-4  [default]", (; sig0_costate=1e-4)),
        ("anneal 5e-2 → 1e-4", (; sig0_costate=1e-4, anneal_costate=5e-2)),
        ("tight + costate observed", (; sig0_costate=1e-4, observe_costate=true)),
        ("anneal + costate observed", (; sig0_costate=1e-4, anneal_costate=5e-2, observe_costate=true)),
        #=
        `q0 = 0.2` is the truth's own cost scale. It is in the table as an upper
        bound on what a lucky guess buys, not as a setting anyone can follow: in
        a real fit you do not know the scale, which is the whole point of the
        `q0` ladder above showing that the fitted scale barely moves from it.
        =#
        ("tight + q0 = 0.2 (the truth's scale)", (; sig0_costate=1e-4, q0=0.2)),
    ]
    cgrid = Any[]
    for g in (:rand, :lqr), (lab, kw) in combos
        cgrid = push!(cgrid, (:combo, g, lab) => (; base..., gen=g, kw...))
    end
    cres = cells(recover, cgrid; seeds=cfg.seeds)

    section(
        "4b — Where to start: initial conditions and settings — " *
        "$(gen === :lqr ? "agent" : "model")" *
        "\n     (4 references, terminal cost and a delay epoch throughout;" *
        " median over $(nseeds(cfg.seeds)))",
    )
    table_header()
    for fn in (true, false), c in cnoise
        a = aggregate(res[(:cn, fn, c)])
        a === nothing && continue
        report(
            rpad(@sprintf("Σ_λλ init %.0e, Σ %s", c, fn ? "estimated" : "frozen"), LBLW), a
        )
    end
    println()
    for v in snoise
        a = aggregate(res[(:sn, v)])
        a === nothing && continue
        report(rpad(@sprintf("Σ_xx init %.3f", v), LBLW), a)
    end
    println()
    for q in q0s
        a = aggregate(res[(:q0, q)])
        a === nothing && continue
        report(rpad(@sprintf("cost init q0 = %.2f", q), LBLW), a)
    end

    for g in (:rand, :lqr)
        println()
        println("   combinations, trials drawn from the $(g === :lqr ? "agent" : "model"):")
        table_header()
        for (lab, _) in combos
            a = aggregate(cres[(:combo, g, lab)])
            a === nothing && continue
            report(rpad("  " * lab, LBLW), a)
        end
    end

    figures || return (ladders=res, grid=gres, combos=cres)
    function ser(keyfn, vals, labels, path)
        return [
            begin
                ms = [center(metric(res[keyfn(l, v)], path)) for v in vals]
                (lab, [m[1] for m in ms], ([m[2] for m in ms], [m[3] for m in ms]))
            end for (l, lab) in labels
        ]
    end
    sweep_figure(
        "init_costate_noise_$(gen)";
        panels=[
            (
                "cost  Qc (running)",
                cnoise,
                ser(
                    (l, v) -> (:cn, l, v),
                    cnoise,
                    ((true, "Σ estimated"), (false, "Σ frozen at the start")),
                    r -> r.scores.Qc.rmse,
                ),
            ),
            (
                "reference map  Gref",
                cnoise,
                ser(
                    (l, v) -> (:cn, l, v),
                    cnoise,
                    ((true, "Σ estimated"), (false, "Σ frozen at the start")),
                    r -> r.scores.Gref.rmse,
                ),
            ),
            (
                "closed-loop plant",
                cnoise,
                ser(
                    (l, v) -> (:cn, l, v),
                    cnoise,
                    ((true, "Σ estimated"), (false, "Σ frozen at the start")),
                    r -> r.scores.cl.rmse,
                ),
            ),
        ],
        xlabel="initial costate innovation  Σ_λλ",
        logx=true,
        xticks=(cnoise, [@sprintf("%.0e", c) for c in cnoise]),
        title="The one setting that matters most — $(gen === :lqr ? "agent" : "model")",
    )

    function one(keyfn, vals, path)
        ms = [center(metric(res[keyfn(v)], path)) for v in vals]
        return [("relative RMSE", [m[1] for m in ms], ([m[2] for m in ms], [m[3] for m in ms]))]
    end
    sweep_figure(
        "init_controls_$(gen)";
        panels=[
            ("Σ_xx init → Qc", snoise, one(v -> (:sn, v), snoise, r -> r.scores.Qc.rmse)),
            ("cost init q0 → Qc", q0s, one(v -> (:q0, v), q0s, r -> r.scores.Qc.rmse)),
            ("cost init q0 → Gref", q0s, one(v -> (:q0, v), q0s, r -> r.scores.Gref.rmse)),
        ],
        xlabel="initial value",
        logx=true,
        title="The other two initialization axes, as controls",
    )

    z = [
        center(metric(gres[(:g, q, c)], r -> r.scores.Qc.rmse))[1] for c in gridc,
        q in gridq
    ]
    zcl = [
        center(metric(gres[(:g, q, c)], r -> r.scores.cl.rmse))[1] for c in gridc,
        q in gridq
    ]
    for (tag, zz, what) in (
        ("qc", z, "Qc relative RMSE — the cost's shape"),
        ("closedloop", zcl, "closed-loop relative RMSE — shape and scale together"),
    )
        grid_figure(
            "init_grid_$(tag)_$(gen)";
            xs=[@sprintf("%.2f", q) for q in gridq],
            ys=[@sprintf("%.0e", c) for c in gridc],
            z=zz,
            xlabel="initial cost scale  q0",
            ylabel="initial costate innovation  Σ_λλ",
            title="$what  ($(nseeds(gseeds)); ringed cell is best)",
        )
    end

    labs = [c[1] for c in combos]
    for g in (:rand, :lqr)
        dot_figure(
            "init_combos_$(g)";
            labels=labs,
            panels=[
                (
                    "cost  Qc (running)",
                    [center(metric(cres[(:combo, g, l)], r -> r.scores.Qc.rmse))[1] for l in labs],
                ),
                (
                    "reference map  Gref",
                    [center(metric(cres[(:combo, g, l)], r -> r.scores.Gref.rmse))[1] for l in labs],
                ),
                (
                    "closed-loop plant",
                    [center(metric(cres[(:combo, g, l)], r -> r.scores.cl.rmse))[1] for l in labs],
                ),
            ],
            title="Initialization combinations — trials from the $(g === :lqr ? "agent" : "model")",
        )
    end
    return (ladders=res, grid=gres, combos=cres)
end

# ---------------------------------------------------------------------------
# 4c. Model recovery: LQR against a plain LDS
# ---------------------------------------------------------------------------

"""
    experiment_model_recovery(cfg; ...)

Given data, can you tell which model class produced it?

Every other experiment assumes the class and asks about its parameters. This one
asks whether the assumption is checkable, by generating from three sources and
scoring an LQR candidate and an LDS candidate on held-out trials:

  * `rand` — the LQR model's own forward chain. The LQR is correctly specified
    and the LDS is not, so the comparison must work here or it works nowhere.
  * `lqr` — the optimal trajectory. This is the case the model is *for*, and the
    LQR is misspecified for it: an optimal path's mixed-coordinate residual is
    zero up to slack, not a draw from the forward chain.
  * `lds` — a first-order attractor toward the target, the alternative account of
    goal-directed reaching.

The sweep asks how much data the distinction needs, and whether trial length,
observation noise or the agent's suboptimality make it easier or harder.
"""
function experiment_model_recovery(cfg; figures::Bool=true)
    n, T, N = 2, cfg.tsteps, cfg.ntrials
    #=
    Fewer rungs and a smaller iteration budget than the parameter sweeps use.
    Every cell here fits *two* models, and a held-out score does not need the
    last decimal of convergence the way a parameter comparison does.
    =#
    Ns = filter(<=(2N), cfg.trial_ladder)[1:2:end]
    noises = (0.02, 0.20)
    gens = (:rand, :lqr, :lds)
    base = (;
        n=n,
        tsteps=T,
        nref=8,
        max_iter=min(cfg.max_iter, 150),
        #=
        A *loose* costate innovation, against the parameter sweeps' default. The
        tight one makes the model over-confident about the costate, and
        over-confidence is exactly what a held-out score punishes: at `1e-4` the
        LQR loses to the LDS on every generator, including its own chain.
        =#
        sig0_costate=1e-2,
    )

    grid = Any[]
    for g in gens
        push!(grid, (:base, g) => (; base..., ntrials=N, gen=g))
        for v in Ns
            push!(grid, (:N, g, v) => (; base..., ntrials=v, gen=g))
        end
        for v in noises
            push!(grid, (:noise, g, v) => (; base..., ntrials=N, gen=g, obs_noise=v))
        end
    end
    res = cells(model_recovery, grid; seeds=cfg.seeds)

    section(
        "4c — Model recovery: an LQR against a plain LDS" *
        "\n     (held-out ELBO per timestep, 8 ring targets, no terminal factor;" *
        " median over $(nseeds(cfg.seeds)))",
    )
    @printf(
        "%-34s %10s %10s %10s %9s %8s\n",
        "generated by",
        "truth",
        "LQR fit",
        "LDS fit",
        "LQR−LDS",
        "picked"
    )
    println("-"^86)
    for g in gens
        rs = filter(!isnothing, res[(:base, g)])
        isempty(rs) && continue
        m(f) = center([Float64(f(r)) for r in rs])[1]
        picked = m(r -> r.delta) > 0 ? "LQR" : "LDS"
        mark = Symbol(lowercase(picked)) === (g === :rand ? :lqr : g) ? "  ok" : "  --"
        @printf(
            "%-34s %10.4f %10.4f %10.4f %9.4f %8s%s\n",
            "$(g)   (p: LQR $(rs[1].p_lqr), LDS $(rs[1].p_lds))",
            m(r -> r.truth),
            m(r -> r.lqr),
            m(r -> r.lds),
            m(r -> r.delta),
            picked,
            mark
        )
    end

    figures || return res
    function ser(keyfn, vals)
        return [
            begin
                ms = [center(metric(res[keyfn(g, v)], r -> r.delta)) for v in vals]
                (
                    string(g),
                    [m[1] for m in ms],
                    ([m[2] for m in ms], [m[3] for m in ms]),
                )
            end for g in gens
        ]
    end
    sweep_figure(
        "model_recovery";
        panels=[
            ("LQR − LDS held-out ELBO vs trials", Ns, ser((g, v) -> (:N, g, v), Ns)),
            (
                "vs observation noise",
                collect(noises),
                ser((g, v) -> (:noise, g, v), collect(noises)),
            ),
        ],
        xlabel="",
        logx=true,
        ylog=false,
        zeroline=true,
        ylabel="nats/timestep  (>0 picks LQR)",
        title="Is the model class identifiable? ($(nseeds(cfg.seeds)))",
    )
    return res
end

# ---------------------------------------------------------------------------
# 4d. The gold-standard configuration
# ---------------------------------------------------------------------------

"""
    experiment_gold_standard(cfg; ...)

One realistic configuration, and the best fitting procedure for it.

The configuration is a centre-out reach: a two-dimensional plant, eight targets
equally spaced on a ring, and three cost levels within the trial — a near-zero
delay cost, a running cost from movement onset, and a terminal cost at the
endpoint. Trials come from an agent that actually solves it.

Everything else in this directory varies one knob at a time across many
configurations. This fixes the configuration and searches the procedure, which
is the question someone with data in hand actually has: *given this experiment,
how should I fit it?* The winner is chosen on the closed-loop error, because
that is the scale-invariant summary and the cost matrices' own scale is set by
the initialization rather than by the data.
"""
function experiment_gold_standard(cfg; figures::Bool=true)
    T = max(cfg.tsteps, 30)
    N = cfg.ntrials
    base = (;
        n=2,
        tsteps=T,
        ntrials=N,
        nref=8,
        ring=true,
        terminal=true,
        onset=max(2, T ÷ 3),
        gen=:lqr,
        max_iter=cfg.max_iter,
    )
    procedures = [
        ("default (tight Σ_λλ = 1e-4)", (;)),
        ("loose Σ_λλ = 5e-2", (; sig0_costate=5e-2)),
        ("tight + q0 = 0.2", (; q0=0.2)),
        ("tight + q0 = 0.1", (; q0=0.1)),
        ("tight + 4 restarts", (; restarts=4)),
        ("tight + 4× iterations", (; max_iter=4cfg.max_iter)),
        ("tight + h frozen at 0", (; free_h=false)),
        ("tight + Σ held at the start", (; fit_noise=false)),
        ("anneal 5e-2 → 1e-4", (; sig0_costate=1e-4, anneal_costate=5e-2)),
        ("tight + costate observed", (; observe_costate=true)),
        ("anneal + costate observed", (; sig0_costate=1e-4, anneal_costate=5e-2, observe_costate=true)),
        ("tight, plant estimated too", (; known_plant=false)),
    ]
    res = cells(recover, [(p[1] => (; base..., p[2]...)) for p in procedures]; seeds=cfg.seeds)

    section(
        "4d — Gold standard: 3 cost levels, 8 targets on a ring" *
        "\n     (n = 2, T = $T, N = $N, agent generator;" *
        " median over $(nseeds(cfg.seeds)))",
    )
    blocks = (:Qc, :Qdel, :Qterm, :Gref, :S, :cl)
    table_header(blocks)
    best, best_cl = nothing, Inf
    for (lab, _) in procedures
        a = aggregate(res[lab])
        a === nothing && continue
        report(rpad(lab, LBLW), a; blocks=blocks)
        c = a.scores.cl.rmse
        if isfinite(c) && c < best_cl
            best, best_cl = lab, c
        end
    end
    best === nothing || println("\n   best on closed-loop error: $best  ($(round(best_cl; digits=4)))")

    #=
    Can the cost scale be *selected* rather than guessed? The `q0` ladder says
    the fitted scale barely moves from its start and that the best start is the
    truth's own — which is no use to anyone fitting real data. A held-out score
    is the only quantity here computable without the truth, so this asks whether
    it ranks `q0` the way the closed-loop error does. If it does, "you cannot
    guess the scale" becomes "you can cross-validate it".
    =#
    q0s = [0.05, 0.1, 0.2, 0.4, 1.0, 2.0]
    sel = cells(
        recover,
        [q => (; base..., q0=q, test_frac=0.3) for q in q0s];
        seeds=cfg.seeds,
    )
    println()
    println("   selecting the cost scale by cross-validation:")
    @printf(
        "%-18s %14s %14s %14s\n", "q0", "held-out/step", "closed-loop", "ELBO − truth"
    )
    println("-"^62)
    bycv, bycl = (-Inf, 0.0), (Inf, 0.0)
    for q in q0s
        rs = filter(!isnothing, sel[q])
        isempty(rs) && continue
        m(f) = center([Float64(f(r)) for r in rs])[1]
        ho, cl = m(r -> r.heldout), m(r -> r.scores.cl.rmse)
        @printf("%-18.2f %14.4f %14.4f %14.1f\n", q, ho, cl, m(r -> r.elbo - r.truth_elbo))
        isfinite(ho) && ho > bycv[1] && (bycv = (ho, q))
        isfinite(cl) && cl < bycl[1] && (bycl = (cl, q))
    end
    @printf(
        "   cross-validation picks q0 = %.2f; the closed-loop error is best at q0 = %.2f%s\n",
        bycv[2],
        bycl[2],
        bycv[2] == bycl[2] ? "  — they agree" : "  — they disagree"
    )

    figures || return (procedures=res, scale=sel)
    labs = [p[1] for p in procedures]
    sweep_figure(
        "gold_standard_scale";
        panels=[
            (
                "held-out ELBO per timestep",
                q0s,
                [(
                    "held-out",
                    [center(metric(sel[q], r -> r.heldout))[1] for q in q0s],
                    nothing,
                )],
            ),
            (
                "closed-loop relative RMSE",
                q0s,
                [(
                    "closed-loop",
                    [center(metric(sel[q], r -> r.scores.cl.rmse))[1] for q in q0s],
                    nothing,
                )],
            ),
        ],
        xlabel="initial cost scale  q0",
        logx=true,
        ylog=false,
        ylabel="",
        title="Can the cost scale be cross-validated?",
    )
    dot_figure(
        "gold_standard";
        labels=labs,
        panels=[
            ("closed-loop plant", [center(metric(res[l], r -> r.scores.cl.rmse))[1] for l in labs]),
            ("cost  Qc (running)", [center(metric(res[l], r -> r.scores.Qc.rmse))[1] for l in labs]),
            ("reference map  Gref", [center(metric(res[l], r -> r.scores.Gref.rmse))[1] for l in labs]),
        ],
        title="3 cost levels, 8 ring targets — which fitting procedure wins",
    )
    if best !== nothing
        rs = filter(!isnothing, res[best])
        isempty(rs) || begin
            matrix_figure(rs[1], "gold_standard_params"; title="Gold standard — $best")
            scatter_figure(rs[1], "gold_standard_entries"; title="Gold standard — $best")
        end
    end
    return (procedures=res, scale=sel)
end

# ---------------------------------------------------------------------------
# 4e. Priors
# ---------------------------------------------------------------------------

"""
    experiment_priors(cfg; ...)

What an explicit prior buys over an initialization that happens to stick.

Two findings motivated the priors this measures. The fitted cost scale barely
moves from where it starts, so the analysis already *has* a prior on it — an
infinitely strong one, placed by the initialization and invisible in the output.
And a switching LQR's innovation, left free, inflates until the LQR state is a
second free state, which every working switching row avoids by pinning `Σ` at the
truth — a concession, not a method.

Both are now inverse-Wishart priors in the model rather than harness tricks, so
this asks the three questions that follow:

  * does a prior on `Σ` make the *initialization* stop mattering? If the answer
    is yes, the biggest lever in this directory stops being a lever and becomes a
    modelling choice with a strength attached.
  * does a prior on the cost pin the scale the data cannot?
  * does a prior on `Σ` replace pinning it in the switching fit?

Strengths are in the inverse-Wishart's own units: `ν` is a pseudo-count against
the transitions in the data, so `ν = 100` against 400 trials of 30 timesteps is
a weak prior and `ν = 10000` a stiff one.
"""
function experiment_priors(cfg; figures::Bool=true)
    T = max(cfg.tsteps, 30)
    N = cfg.ntrials
    base = (;
        n=2,
        tsteps=T,
        ntrials=N,
        nref=8,
        ring=true,
        terminal=true,
        onset=max(2, T ÷ 3),
        gen=:lqr,
        max_iter=cfg.max_iter,
    )
    #=
    Strengths are pseudo-counts against the data, so the ladder has to reach the
    data's own weight to say anything. The gold-standard configuration has
    `N × (T − 1)` transitions — about 12,000 here — and a `ν` of 1e3 is under a
    tenth of that, which is why the first pass at this experiment found the cost
    prior "too weak to matter" when it was only too small to matter.
    =#
    νs = [0.0, 1e2, 1e3, 1e4, 1e5]
    νq = [0.0, 1e2, 1e3, 1e4, 1e5]
    q_modes = [0.2, 0.02, 3.0]  # running, delay, terminal

    #=
    The `Σ` prior crossed with the *initialization* it is meant to replace. If the
    prior works, the two starts converge as `ν` grows; if it does not, the loose
    row stays bad however strong the prior.
    =#
    grid = Any[]
    for ν in νs, sc in (1e-4, 5e-2)
        push!(
            grid,
            (:sig, ν, sc) =>
                (; base..., sig0_costate=sc, sigma_prior_strength=ν),
        )
    end
    for ν in νq
        push!(grid, (:qc, ν) => (; base..., qc_prior_strength=ν, qc_prior_scale=q_modes))
    end
    #=
    The same prior aimed at the wrong scale. A prior that pins the cost is only
    useful if being wrong about it costs something a reader can see, and this is
    the row that shows what.
    =#
    for ν in (1e3, 1e4, 1e5)
        push!(grid, (:qcbad, ν) => (; base..., qc_prior_strength=ν, qc_prior_scale=[0.6, 0.02, 3.0]))
    end
    for ν in νq
        push!(
            grid,
            (:both, ν) => (;
                base...,
                sigma_prior_strength=1e4,
                qc_prior_strength=ν,
                qc_prior_scale=q_modes,
            ),
        )
    end
    res = cells(recover, grid; seeds=cfg.seeds)

    section(
        "4e — Priors: what they buy over an initialization that sticks" *
        "\n     (gold-standard configuration; median over $(nseeds(cfg.seeds)))",
    )
    blocks = (:Qc, :Qterm, :Gref, :S, :Sig, :cl)
    table_header(blocks)
    for sc in (1e-4, 5e-2), ν in νs
        a = aggregate(res[(:sig, ν, sc)])
        a === nothing && continue
        report(
            rpad(@sprintf("Σ prior ν=%-6g  init Σ_λλ %.0e", ν, sc), LBLW), a; blocks=blocks
        )
    end
    println()
    for ν in νq
        a = aggregate(res[(:qc, ν)])
        a === nothing && continue
        report(rpad(@sprintf("cost prior ν=%-6g  (epoch modes)", ν), LBLW), a; blocks=blocks)
    end
    println()
    for ν in (1e3, 1e4, 1e5)
        a = aggregate(res[(:qcbad, ν)])
        a === nothing && continue
        report(
            rpad(@sprintf("cost prior ν=%-6g  run=0.6 (wrong)", ν), LBLW), a; blocks=blocks
        )
    end
    println()
    for ν in νq
        a = aggregate(res[(:both, ν)])
        a === nothing && continue
        report(
            rpad(@sprintf("Σ prior 1e4 + cost prior ν=%-6g", ν), LBLW), a; blocks=blocks
        )
    end

    #=
    The switching half. `Σ` is estimated throughout the prior rows. Cross the
    useful `Σ` strength with the cost prior so this section asks about both
    parameter priors, not only whether regularizing `Σ` replaces pinning it.
    =#
    sl = cfg.slds
    sbase = (;
        n=sl.n,
        tsteps=sl.tsteps,
        ntrials=sl.ntrials,
        max_iter=sl.max_iter,
        gen=:epoch,
        terminal=true,
        nref=4,
    )
    sq_modes = [0.2, 3.0]  # running, terminal
    sconds = [
        ("Σ estimated, no prior", (; sbase...)),
        ("Σ estimated, prior ν=1e2", (; sbase..., sigma_prior_strength=1e2, sigma_prior_costate=2e-2)),
        ("Σ estimated, prior ν=1e3", (; sbase..., sigma_prior_strength=1e3, sigma_prior_costate=2e-2)),
        ("Σ estimated, prior ν=1e4", (; sbase..., sigma_prior_strength=1e4, sigma_prior_costate=2e-2)),
        ("Σ estimated, prior ν=1e5", (; sbase..., sigma_prior_strength=1e5, sigma_prior_costate=2e-2)),
        ("Qc prior ν=1e3", (; sbase..., qc_prior_strength=1e3, qc_prior_scale=sq_modes)),
        ("Qc prior ν=1e4", (; sbase..., qc_prior_strength=1e4, qc_prior_scale=sq_modes)),
        ("Qc prior ν=1e5", (; sbase..., qc_prior_strength=1e5, qc_prior_scale=sq_modes)),
        (
            "Σ ν=1e4 + Qc ν=1e3",
            (;
                sbase...,
                sigma_prior_strength=1e4,
                sigma_prior_costate=2e-2,
                qc_prior_strength=1e3,
                qc_prior_scale=sq_modes,
            ),
        ),
        (
            "Σ ν=1e4 + Qc ν=1e4",
            (;
                sbase...,
                sigma_prior_strength=1e4,
                sigma_prior_costate=2e-2,
                qc_prior_strength=1e4,
                qc_prior_scale=sq_modes,
            ),
        ),
        (
            "Σ ν=1e4 + Qc ν=1e5",
            (;
                sbase...,
                sigma_prior_strength=1e4,
                sigma_prior_costate=2e-2,
                qc_prior_strength=1e5,
                qc_prior_scale=sq_modes,
            ),
        ),
        ("Σ pinned (the concession)", (; sbase..., sig0_state=0.02, sig0_costate=2e-2, fit_noise=false)),
    ]
    sres = cells(recover_slds, [c[1] => c[2] for c in sconds]; seeds=sl.seeds)
    println()
    println("   switching: Σ and Qc priors, against pinning Σ")
    table_header(CORE_BLOCKS; tail=SLDS_TAIL)
    for (lab, _) in sconds
        a = aggregate_slds(sres[lab])
        a === nothing && (println(rpad(lab, LBLW), "  FAILED"); continue)
        report_slds(rpad(lab, LBLW), a)
    end

    figures || return (single=res, switching=sres)
    function ser(sc, path)
        ms = [center(metric(res[(:sig, ν, sc)], path)) for ν in νs]
        return (
            sc == 1e-4 ? "init Σ_λλ = 1e-4" : "init Σ_λλ = 5e-2",
            [m[1] for m in ms],
            ([m[2] for m in ms], [m[3] for m in ms]),
        )
    end
    xs = [1.0, 1e2, 1e3, 1e4, 1e5]  # ν = 0 drawn at 1 so a log axis can hold it
    sweep_figure(
        "priors_sigma";
        panels=[
            (
                "cost  Qc (running)",
                xs,
                [ser(1e-4, r -> r.scores.Qc.rmse), ser(5e-2, r -> r.scores.Qc.rmse)],
            ),
            (
                "closed-loop plant",
                xs,
                [ser(1e-4, r -> r.scores.cl.rmse), ser(5e-2, r -> r.scores.cl.rmse)],
            ),
            (
                "innovation  Σ xx",
                xs,
                [ser(1e-4, r -> r.scores.Sig.rmse), ser(5e-2, r -> r.scores.Sig.rmse)],
            ),
        ],
        xlabel="Σ prior strength ν  (1 = no prior)",
        logx=true,
        xticks=(xs, ["none", "1e2", "1e3", "1e4", "1e5"]),
        title="Does a prior on Σ make the initialization stop mattering?",
    )
    slabs = [c[1] for c in sconds]
    have = [l for l in slabs if aggregate_slds(sres[l]) !== nothing]
    isempty(have) || gamma_figure(
        "priors_slds_gamma";
        labels=have,
        fitted=[center(metric(sres[l], r -> r.gamma.acc))[1] for l in have],
        reference=[center(metric(sres[l], r -> r.truth_gamma.acc))[1] for l in have],
        title="Σ and Qc priors in the switching fit ($(sl.ntrials) trials, T = $(sl.tsteps))",
    )
    return (single=res, switching=sres)
end

# ---------------------------------------------------------------------------
# 5. Switching: one free state, one LQR state
# ---------------------------------------------------------------------------

"""
    experiment_switching(cfg; ...)

The `SLDS` section. Every condition reports both halves of the answer — the LQR
state's parameters, and the posterior over the discrete path — because a
switching fit can fail either one alone, and usually does.

The conditions are ordered to isolate one thing at a time: what the default fit
does, what pinning the LQR state's innovation covariance does, what starting at
the truth does, and what observing the costate does. Then the structural axes
(terminal, references, horizon, trials, observation noise) under the setting the
first block picks out.
"""
function experiment_switching(cfg; figures::Bool=true)
    s = cfg.slds
    base = (;
        n=s.n,
        tsteps=s.tsteps,
        ntrials=s.ntrials,
        max_iter=s.max_iter,
        gen=:epoch,
        terminal=true,
        nref=4,
    )
    #=
    "Pinned" means `Σ` held at the truth's own two blocks rather than estimated.
    It is a concession — the fit is told the innovation — and it is in the table
    because without it nothing else in this section is interpretable: an LQR
    discrete state with a free `Σ` inflates until it is a second free state, and
    every row would be reporting that collapse instead of the question asked.
    =#
    pinned = (; sig0_state=0.02, sig0_costate=2e-2, fit_noise=false)

    conds = [
        ("cold start, Σ estimated", base),
        ("cold start, Σ pinned", (; base..., pinned...)),
        ("warm start, Σ estimated", (; base..., init=:warm)),
        ("warm start, Σ pinned", (; base..., pinned..., init=:warm)),
        ("costate observed, Σ estimated", (; base..., observe_costate=true)),
        ("costate observed, Σ pinned", (; base..., pinned..., observe_costate=true)),
        ("Σ pinned, no terminal", (; base..., pinned..., terminal=false)),
        ("Σ pinned, no reference", (; base..., pinned..., nref=0)),
        ("Σ pinned, 1 reference", (; base..., pinned..., nref=1)),
        ("Σ pinned, T = $(2s.tsteps)", (; base..., pinned..., tsteps=2s.tsteps)),
        ("Σ pinned, 4× the trials", (; base..., pinned..., ntrials=4s.ntrials)),
        ("Σ pinned, quiet emission", (; base..., pinned..., obs_noise=0.01)),
        ("Σ pinned, plant estimated", (; base..., pinned..., known_plant=false)),
        #=
        The costate-innovation ladder, pinned so the number under test is the
        one being varied. This is where the switching model disagrees with the
        single-system one: `1e-4` is the single-system optimum and is chance
        here. `γ (truth)` moves with it, so the collapse is a property of the
        model rather than of the fit.
        =#
        ("Σ pinned, Σ_λλ = 1e-4 (1-system optimum)", (; base..., pinned..., sig0_costate=1e-4, costate_noise=1e-4)),
        ("Σ pinned, Σ_λλ = 2.5e-3", (; base..., pinned..., sig0_costate=2.5e-3, costate_noise=2.5e-3)),
        ("Σ pinned, Σ_λλ = 1e-2", (; base..., pinned..., sig0_costate=1e-2, costate_noise=1e-2)),
        #=
        Two-stage fits. The ladder above says `γ` and the cost want opposite
        costate innovations, so the obvious response is to do both in turn. It
        half works: annealing to `1e-3` gives the best `γ` in the table, and
        annealing all the way to the single-system optimum destroys it again
        without buying the cost.
        =#
        ("Σ pinned, anneal → 1e-3", (; base..., pinned..., anneal_costate=1e-3)),
        ("Σ pinned, anneal → 1e-4", (; base..., pinned..., anneal_costate=1e-4)),
        (
            "anneal → 1e-4 + costate observed",
            (; base..., pinned..., anneal_costate=1e-4, observe_costate=true),
        ),
        ("Σ pinned, `rand` generator", (; base..., pinned..., gen=:rand, sscale=0.25)),
        (
            "Σ pinned, `rand`, plant est.",
            (; base..., pinned..., gen=:rand, sscale=0.25, known_plant=false),
        ),
    ]
    res = cells(recover_slds, [c[1] => c[2] for c in conds]; seeds=s.seeds)

    section("5 — Switching: one `:free` state and one LQR state")
    table_header(CORE_BLOCKS; tail=SLDS_TAIL)
    for (lab, _) in conds
        a = aggregate_slds(res[lab])
        if a === nothing
            println(rpad(lab, LBLW), "  FAILED")
            continue
        end
        report_slds(rpad(lab, LBLW), a)
    end

    #=
    Segment-then-fit, which is the section's answer to "what would improve
    this". The joint fit conflates two problems; this runs them in sequence —
    take a segmentation, slice out the LQR-governed timesteps, hand them to the
    single-system machinery. The oracle row is the ceiling: whatever it fails to
    recover is not the switching layer's fault.
    =#
    println()
    println("   segment-then-fit (fixed onset, single-system LQR on the LQR epoch):")
    segblocks = (:Qc, :Qterm, :Gref, :S, :cl)
    table_header(segblocks; tail="    Δelbo iters    creep  kept")
    segres = Dict{Any,Any}()
    for src in (:oracle, :fit), oc in (false, true)
        lab = "  $(src) segmentation$(oc ? " + costate observed" : "")"
        rs = Any[]
        for sd in s.seeds
            r = try
                segment_recover(;
                    n=s.n,
                    tsteps=s.tsteps,
                    ntrials=s.ntrials,
                    nref=4,
                    source=src,
                    observe_costate=oc,
                    max_iter=4s.max_iter,
                    slds_iter=s.max_iter,
                    seed=sd,
                )
            catch err
                err isa InterruptException && rethrow()
                nothing
            end
            push!(rs, r)
        end
        #=
        A run that found no LQR segments carries a `failed` marker rather than
        scores; `aggregate` cannot read it, so it is reported on its own terms.
        =#
        ok = [r for r in rs if r !== nothing && !haskey(r, :failed)]
        segres[(src, oc)] = ok
        if isempty(ok)
            reason = any(r -> r !== nothing && haskey(r, :failed), rs) ?
                "no LQR segments — the stage-one fit put every timestep in the free state" :
                "FAILED"
            println(rpad(lab, LBLW), "  ", reason)
            continue
        end
        a = aggregate(ok)
        a === nothing && (println(rpad(lab, LBLW), "  FAILED"); continue)
        print(rpad(lab, LBLW))
        for b in segblocks
            print(cell(getfield(a.scores, b)))
        end
        kept = center([Float64(r.kept) for r in ok])[1]
        @printf(" %8.1f %5d %8.1e %5.0f\n", a.elbo - a.truth_elbo, a.iters, a.creep, kept)
    end

    figures || return (joint=res, segment=segres)
    labs = [c[1] for c in conds]
    have = [l for l in labs if !isempty(filter(!isnothing, res[l]))]
    gamma_figure(
        "slds_gamma";
        labels=have,
        fitted=[center(metric(res[l], r -> r.gamma.acc))[1] for l in have],
        reference=[center(metric(res[l], r -> r.truth_gamma.acc))[1] for l in have],
        title="Recovering the discrete path ($(s.ntrials) trials, T = $(s.tsteps))",
    )
    shown = ("cold start, Σ estimated", "cold start, Σ pinned", "costate observed, Σ pinned")
    for l in shown
        rs = filter(!isnothing, get(res, l, Any[]))
        isempty(rs) && continue
        slug = replace(replace(l, " " => "_"), "," => "")
        slds_trial_figure(rs[1], "slds_trial_$slug"; title="Switching fit — $l")
    end
    #=
    The parameter figure is drawn for the condition that recovered the discrete
    path best, since a parameter panel from a collapsed fit is a picture of the
    collapse rather than of the model.
    =#
    accs = [center(metric(res[l], r -> r.gamma.acc))[1] for l in have]
    if any(isfinite, accs)
        best = have[argmax(map(a -> isfinite(a) ? a : -Inf, accs))]
        rs = filter(!isnothing, res[best])
        isempty(rs) || matrix_figure(
            rs[1], "slds_params"; title="LQR state's parameters, switching fit — $best"
        )
    end
    #=
    The segment fit beside the joint one, on the same axes: the bar is the
    joint fit's cost error, the rule the segment fit's, so the gap is the price
    of estimating the epochs and the cost at the same time.
    =#
    segqc = [
        (string(src, oc ? " + costate" : ""), aggregate(segres[(src, oc)]))
        for src in (:oracle, :fit) for oc in (false, true)
    ]
    keep = [(l, a) for (l, a) in segqc if a !== nothing]
    isempty(keep) || dot_figure(
        "slds_segment";
        labels=[l for (l, _) in keep],
        panels=[
            ("cost  Qc (running)", [a.scores.Qc.rmse for (_, a) in keep]),
            ("closed-loop plant", [a.scores.cl.rmse for (_, a) in keep]),
        ],
        title="Segment-then-fit: the cost given the epochs",
    )
    return (joint=res, segment=segres)
end
