#=============================================================================
Tables.

Two numbers per parameter block, printed as `rmse/corr`, plus a tail of
diagnostics. A block the row does not estimate — or whose truth is zero, so that
a relative error has no denominator and a correlation no variance — reads `--`
rather than a number that would look like one.

The column set is an argument rather than a constant: a section that has no
reference to estimate should not carry an empty `Gref` column for thirty rows.
=============================================================================#

const BLOCK_LABEL = (
    Qc="Qc run",
    Qdel="Qc delay",
    Qterm="Qc term",
    A="A",
    S="S",
    Gref="Gref",
    Sig="Σ xx",
    h="h",
    cl="closed-loop",
)

"""The default column set: everything a structural sweep varies."""
const CORE_BLOCKS = (:Qc, :Qterm, :Gref, :A, :S, :Sig, :cl)

const LBLW = 34   # label column width
const COLW = 12   # one `rmse/corr` cell

"""`n seeds`, or `1 seed`."""
nseeds(seeds) = string(length(seeds), " seed", length(seeds) == 1 ? "" : "s")

"""
    center(v) -> (med, lo, hi)

The median over the seeds of one cell, and the distances from it to the smallest
and largest seed. `NaN` values — a block with nothing to score — are dropped, and
a cell with no finite value at all reports `NaN` rather than a zero that would
plot as a point.

Median and full range, not mean and standard deviation, and the reason is in the
data. These recovery errors are heavily right-skewed: most seeds of a condition
land together and one occasionally fails outright, at fifty times the others'
error. A mean of `{0.07, 0.07, 0.07, 4.0}` is `1.05`, which on a log axis draws
every seed as if it had failed; a standard deviation of `2.0` around it draws an
error bar that runs below zero and cannot be plotted at all. The median says what
the typical seed did and the range says how bad the worst one was, which are the
two things worth knowing and are not the same number.

Nothing is hidden by this: the range's upper arm *is* the catastrophic seed, drawn
at its true size.
"""
function center(v)
    f = sort!(filter(isfinite, collect(v)))
    isempty(f) && return (NaN, NaN, NaN)
    n = length(f)
    med = isodd(n) ? f[(n + 1) ÷ 2] : (f[n ÷ 2] + f[n ÷ 2 + 1]) / 2
    return (med, med - f[1], f[end] - med)
end

_rmse_str(x) = isnan(x) ? "  --" : (x >= 99.95 ? ">99" : @sprintf("%5.3f", x))
_corr_str(x) = isnan(x) ? " --" : @sprintf("%5.2f", x)

function cell(s)
    (isnan(s.rmse) && isnan(s.corr)) && return lpad("--", COLW)
    return lpad(string(_rmse_str(s.rmse), "/", _corr_str(s.corr)), COLW)
end

function table_header(blocks=CORE_BLOCKS; tail::String="    Δelbo iters    creep  ρ(M)")
    print(rpad("condition", LBLW))
    for b in blocks
        print(lpad(getfield(BLOCK_LABEL, b), COLW))
    end
    println(tail)
    print(rpad("", LBLW))
    for _ in blocks
        print(lpad("rmse/corr", COLW))
    end
    println(" vs truth")
    return println("-"^(LBLW + COLW * length(blocks) + length(tail)))
end

function section(title::String)
    println()
    println(title)
    return println("=" ^ length(title))
end

"""
    report(label, r; blocks, note)

One row of a single-system sweep.

`Δelbo` is the fit's ELBO minus the ELBO at the generating parameters: a few nats
is ordinary finite-sample slack, a large positive number means the model prefers
something other than the truth.

`creep` is the mean ELBO gain per iteration over the last tenth of the run. It
replaces a converged / not-converged flag, which on these fits is useless: with
`tol = 1e-10` the bound keeps rising by tiny amounts for thousands of iterations,
so every row would read `NO` and the column would distinguish nothing. Below
~1e-3 nats/iteration the parameters are flat to three decimals; above ~1e-1 the
row is reporting the iteration budget as much as the data.
"""
function report(label, r; blocks=CORE_BLOCKS, note::String="")
    print(rpad(label, LBLW))
    for b in blocks
        print(cell(getfield(r.scores, b)))
    end
    @printf(
        " %8.1f %5d %8.1e %5.2f %s\n",
        r.elbo - r.truth_elbo,
        r.iters,
        r.creep,
        r.rho,
        note
    )
    report_gauge(r)
    return nothing
end

const SLDS_TAIL = "   γ acc  (truth) γ post  onset  stay  Δelbo"

"""
    report_slds(label, r; blocks, note)

One row of the switching sweep. The parameter columns are the LQR discrete
state's; the tail is the discrete-state answer, which is the separate question
this section exists to ask.

`γ acc` is the balanced accuracy of the MAP state and `(truth)` the same
quantity at the generating parameters — a reference, not a bound. `onset` is the
mean absolute switch-time error in timesteps, `--` when the generator has no
single switch to find. `stay` is the fitted self-transition probability, which
is the quickest tell for a collapsed fit: a model that has put every timestep in
one state reports a `stay` near 1 and a `γ acc` near 0.5.
"""
function report_slds(label, r; blocks=CORE_BLOCKS, note::String="")
    print(rpad(label, LBLW))
    for b in blocks
        print(cell(getfield(r.scores, b)))
    end
    @printf(
        " %7.3f %8.3f %7.3f %6s %5.2f %8.1f %s\n",
        r.gamma.acc,
        r.truth_gamma.acc,
        r.gamma.post,
        isnan(r.onset.mad) ? "--" : @sprintf("%.1f", r.onset.mad),
        r.stay_fit,
        r.elbo - r.truth_elbo,
        note
    )
    report_gauge(r)
    return nothing
end

"""Print the latent-gauge audit when the emission actually moved its basis."""
function report_gauge(r)
    hasproperty(r, :gauge) || return nothing
    g = r.gauge
    g === nothing && return nothing
    raw, proc, lin = g.raw.Gref.rmse, g.procrustes.Gref.rmse, g.linear.Gref.rmse
    isfinite(raw) || return nothing
    moved = g.maps.nonorthogonality > 1e-6 || g.C.procrustes.rmse > 1e-6
    moved || return nothing
    @printf(
        "   gauge: Gref raw/proc/linear %.3f / %.3f / %.3f; G'G %.3f; C proc/linear %.3f / %.3f; nonorth %.3f\n",
        raw, proc, lin, g.Ggram.rmse, g.C.procrustes.rmse,
        g.C.linear.rmse, g.maps.nonorthogonality,
    )
    return nothing
end

"""
    run_row(printer, label, f, kwargs; note) -> result or nothing

One table row, with failures reported rather than thrown. The degenerate corners
this harness deliberately visits — an exactly optimal agent above all — can
drive the fitted `Σ` singular, and a table that stops at the first such row tells
you less than one that prints it and carries on.
"""
function run_row(printer, label, f, kwargs; note::String="", blocks=CORE_BLOCKS)
    try
        r = f(; kwargs...)
        printer(label, r; blocks=blocks, note=note)
        return r
    catch err
        err isa InterruptException && rethrow()
        println(rpad(label, LBLW), "  FAILED: ", typeof(err))
        return nothing
    end
end

"""
    aggregate(rs) -> result-shaped NamedTuple, or `nothing`

Reduce a cell's seeds to something `report` can print.

The tables would otherwise show one seed while the figures beside them show
several summarized, and a reader comparing the two would be comparing different
quantities. Aggregation is blockwise, NaN-aware and by [`center`](@ref)'s median, so a block
that is unscorable in every seed stays unscorable rather than becoming zero, and
one failed seed does not set the whole cell.

The `rmse` of a median fit is not the median of the `rmse`s, and this is the
latter: it is the typical error of a fit, not the error of a typical fit. That is
the one the sweeps want — nobody runs four seeds and averages the parameters.
"""
function aggregate(rs)
    good = filter(!isnothing, rs)
    isempty(good) && return nothing
    blocks = keys(good[1].scores)
    sc = NamedTuple{blocks}(
        map(blocks) do b
            (
                rmse=center([getfield(r.scores, b).rmse for r in good])[1],
                corr=center([getfield(r.scores, b).corr for r in good])[1],
            )
        end,
    )
    m(f) = center([Float64(f(r)) for r in good])[1]
    gauge = hasproperty(good[1], :gauge) ? aggregate_gauge(good) : nothing
    return (
        scores=sc,
        elbo=m(r -> r.elbo),
        truth_elbo=m(r -> r.truth_elbo),
        iters=round(Int, m(r -> r.iters)),
        creep=m(r -> r.creep),
        rho=m(r -> r.rho),
        nseeds=length(good),
        gauge=gauge,
    )
end

function aggregate_gauge(good)
    m(f) = center([Float64(f(r)) for r in good])[1]
    blocks = keys(good[1].gauge.raw)
    branch(which) = NamedTuple{blocks}(map(blocks) do b
        (rmse=m(r -> getproperty(getproperty(r.gauge, which), b).rmse),
         corr=m(r -> getproperty(getproperty(r.gauge, which), b).corr))
    end)
    return (
        raw=branch(:raw), procrustes=branch(:procrustes), linear=branch(:linear),
        Ggram=(rmse=m(r -> r.gauge.Ggram.rmse), corr=m(r -> r.gauge.Ggram.corr)),
        maps=(nonorthogonality=m(r -> r.gauge.maps.nonorthogonality),),
        C=(
            procrustes=(rmse=m(r -> r.gauge.C.procrustes.rmse),
                        corr=m(r -> r.gauge.C.procrustes.corr)),
            linear=(rmse=m(r -> r.gauge.C.linear.rmse),
                    corr=m(r -> r.gauge.C.linear.corr)),
        ),
    )
end

"""
    aggregate_slds(rs) -> result-shaped NamedTuple, or `nothing`

The same for a switching cell, carrying the γ tail `report_slds` prints.
"""
function aggregate_slds(rs)
    good = filter(!isnothing, rs)
    isempty(good) && return nothing
    base = aggregate(rs)
    m(f) = center([Float64(f(r)) for r in good])[1]
    return (
        base...,
        gamma=(
            acc=m(r -> r.gamma.acc), post=m(r -> r.gamma.post), xent=m(r -> r.gamma.xent)
        ),
        truth_gamma=(acc=m(r -> r.truth_gamma.acc),),
        onset=(bias=m(r -> r.onset.bias), mad=m(r -> r.onset.mad)),
        stay_fit=m(r -> r.stay_fit),
    )
end

"""
    cells(f, grid; seeds) -> Dict

Run `f` once per `(cell, seed)` and collect the results. `grid` is a vector of
`(key, kwargs)`; the returned dictionary maps each key to the vector of results
over seeds, with a failed run recorded as `nothing` so a cell that broke is
distinguishable from one that was never run.

Replication over seeds is not decoration. A single recovery run is one draw from
a sampling distribution whose spread, on the harder conditions here, is the same
size as the differences between conditions — so a sweep read off one seed per
cell mostly reports which seed it got.
"""
function cells(f, grid; seeds)
    out = Dict{Any,Vector{Any}}()
    for (key, kw) in grid
        rs = Any[]
        for s in seeds
            r = try
                f(; kw..., seed=s)
            catch err
                err isa InterruptException && rethrow()
                nothing
            end
            push!(rs, r)
        end
        out[key] = rs
    end
    return out
end

"""
    metric(rs, path) -> Vector{Float64}

Pull one scalar out of every seed's result, `NaN` where the run failed. `path`
is applied to the result, e.g. `r -> r.scores.Qc.rmse`.
"""
metric(rs, path) = [r === nothing ? NaN : Float64(path(r)) for r in rs]
