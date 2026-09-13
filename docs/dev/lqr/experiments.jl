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
        seeds=1:4,
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
    # `Σ` pinned at the truth's scale: see the discussion this experiment prints.
    pinned = (; lqr_sig0=0.02, fit_noise=false)

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

    figures || return res
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
    return res
end
