#=============================================================================
Figures.

Everything here writes a PNG into `FIGDIR` and returns the path. Nothing is
displayed: this runs headless, and the point is a folder of figures you can flip
through next to the tables.

## Encoding rules this file holds to

A parameter matrix is *signed* — a cost's off-diagonals, a plant's coupling, a
reference's coordinates all take either sign — so every matrix panel uses a
**diverging** scale: one cool arm, one warm arm, a neutral midpoint at exactly
zero, and symmetric limits. A sequential ramp would put zero somewhere arbitrary
and make "small negative" and "small positive" look like different magnitudes of
the same thing.

Truth and recovery share one color scale, always. Two panels on their own scales
would hide exactly the error the figure exists to show — a fit that is uniformly
half the truth would look identical to it. The residual panel gets its own
(symmetric) limits, and says so in its title, because it is usually an order of
magnitude smaller and would otherwise be a flat gray square.

Categorical series take the palette slots in fixed order and never more than
three, which is the number that stays separable under color-vision deficiency
when every pair can appear side by side. Grid and axes are hairlines a shade off
the surface; marks are thin; labels go on the series, not on every point.
=============================================================================#

using Plots

# Chart chrome and the two diverging arms, light surface.
const INK = "#0b0b0b"
const INK_MUTED = "#898781"
const GRIDLINE = "#e1e0d9"
const SURFACE = "#fcfcfb"
const SERIES = ("#2a78d6", "#eb6834", "#1baf7a")   # slots 1-3, fixed order
const COOL_ARM = ("#0d366b", "#184f95", "#2a78d6", "#6da7ec", "#9ec5f4", "#cde2fb")
const MIDPOINT = "#f0efec"
const WARM_ARM = ("#fbd6d6", "#f4a8a7", "#ec7b7a", "#e34948", "#b32e2d", "#7e1f1e")

"""The diverging ramp: cool → neutral → warm, with zero pinned at the middle."""
diverging() = cgrad(vcat(collect(COOL_ARM), MIDPOINT, collect(WARM_ARM)))

"""
    figdir() -> String

Where figures go. Created on first use; gitignored, because these are
regenerated on every run and a repository is not an image host.
"""
function figdir()
    d = get(ENV, "LQR_FIGDIR", joinpath(@__DIR__, "figures"))
    isdir(d) || mkpath(d)
    return d
end

function save_fig(plt, name::AbstractString)
    path = joinpath(figdir(), endswith(name, ".png") ? name : name * ".png")
    savefig(plt, path)
    return path
end

"""Plot defaults shared by every figure: recessive chrome, thin marks, no box."""
function base_attrs(; kwargs...)
    return (;
        background_color=SURFACE,
        background_color_inside=SURFACE,
        foreground_color_axis=GRIDLINE,
        foreground_color_border=GRIDLINE,
        foreground_color_text=INK,
        foreground_color_guide=INK,
        gridalpha=0.55,
        gridcolor=GRIDLINE,
        gridlinewidth=0.7,
        gridstyle=:solid,
        framestyle=:grid,
        tickfontcolor=INK_MUTED,
        tickfontsize=7,
        guidefontsize=8,
        titlefontsize=9,
        legendfontsize=7,
        kwargs...,
    )
end

# ---------------------------------------------------------------------------
# Parameter matrices: truth against recovery
# ---------------------------------------------------------------------------

"""
    heat(M; clims, title, annotate) -> Plot

One matrix as a diverging heat-table: the fill carries sign and magnitude, and
the value is printed in the cell.

Both, not one or the other. These blocks are small — a handful of entries — and
at that size the fill is what makes the *pattern* legible at a glance while the
number is what makes the panel checkable. A colorbar would cost a third of the
panel's width to say what the printed values already say exactly, and would make
the three panels of a row different widths, which reads as a hierarchy that
isn't there; the shared range goes in the title instead. Above `annotate_max`
cells the numbers would collide, so the fill carries it alone.

Rows run top to bottom, so the picture matches how the matrix is written.
"""
function heat(
    M::AbstractMatrix; clims, title::String, annotate::Bool=true, annotate_max::Int=42
)
    r, c = size(M)
    p = heatmap(
        1:c,
        1:r,
        Matrix{Float64}(M);
        c=diverging(),
        clims=clims,
        yflip=true,
        aspect_ratio=:equal,
        xlims=(0.5, c + 0.5),
        ylims=(0.5, r + 0.5),
        xticks=(1:c, string.(1:c)),
        yticks=(1:r, string.(1:r)),
        title=title,
        colorbar=false,
        base_attrs(; grid=false)...,
    )
    if annotate && r * c <= annotate_max
        lim = max(abs(clims[1]), abs(clims[2]), 1e-12)
        fs = r * c <= 16 ? 7 : 6
        for i in 1:r, j in 1:c
            v = M[i, j]
            # White ink once the fill is dark enough to swallow black text.
            ink = abs(v) / lim > 0.55 ? :white : INK
            annotate!(p, j, i, text(_cellfmt(v, lim), fs, ink, :center))
        end
    end
    return p
end

"""
Two significant figures, with anything negligible against the panel's own range
printed as a bare `0`. A floating-point zero that reads `9e-17` is noise wearing
the clothes of a measurement.
"""
function _cellfmt(v::Real, lim::Real)
    a = abs(v)
    a <= 1e-6 * lim && return "0"
    (a >= 1e4 || a < 1e-3) && return @sprintf("%.0e", v)
    a >= 100 && return @sprintf("%.0f", v)
    a >= 10 && return @sprintf("%.1f", v)
    return @sprintf("%.2f", v)
end

"""
    blocks_of(res) -> Vector{(name, truth, fit, sym)}

Which parameter blocks a result has to show, in a fixed reading order: the plant
first, then the cost regimes, then the reference, then the noise. A block the
model does not carry (no terminal regime, no reference) is simply absent rather
than drawn empty.
"""
function blocks_of(res)
    f, r, idx = res.fit_sm, res.ref_sm, res.idx
    out = Tuple{String,Matrix{Float64},Matrix{Float64},Bool}[]
    # A `:free` state model carries no plant, cost or reference to draw.
    r.mode === :free && return out
    push!(out, ("A  (plant)", Matrix(r.A), Matrix(f.A), false))
    push!(out, ("S  (control)", Matrix(r.S), Matrix(f.S), true))
    push!(out, ("Qc  (running cost)", Matrix(r.Qc[idx.run]), Matrix(f.Qc[idx.run]), true))
    if idx.delay !== nothing
        push!(
            out,
            ("Qc  (delay cost)", Matrix(r.Qc[idx.delay]), Matrix(f.Qc[idx.delay]), true),
        )
    end
    if idx.term !== nothing
        push!(
            out,
            ("Qc  (terminal cost)", Matrix(r.Qc[idx.term]), Matrix(f.Qc[idx.term]), true),
        )
    end
    if size(r.Gref, 2) > 0
        push!(out, ("Gref  (reference map)", Matrix(r.Gref), Matrix(f.Gref), false))
    end
    #=
    The *state* block of the innovation, which is what the tables score and for
    the same reason: `rescale_costate!` multiplies `Σ`'s costate rows and columns
    by the cost scale, so a whole-`Σ` panel is mostly a picture of the cost-scale
    error — which `S` already reports exactly — drawn at a hundred times the size
    of everything else on the row.
    =#
    push!(
        out,
        (
            "Σ xx  (process noise)",
            Matrix(_state_block(r.Σ)),
            Matrix(_state_block(f.Σ)),
            true,
        ),
    )
    return out
end

"""
    matrix_figure(res, name; title) -> path

The figure this harness exists to produce: every parameter block as truth,
recovery and residual, one row each.

The recovery panel's title carries that block's relative RMSE and entry
correlation, so the picture and the number are never read apart. The residual
panel's title carries its own limit, since it is on its own scale.
"""
function matrix_figure(res, name::AbstractString; title::String="")
    bs = blocks_of(res)
    panels = Any[]
    for (bname, tru, fit, sym) in bs
        #=
        The shared scale is the *truth's*, not the pair's. A fit that diverges
        by two orders of magnitude — which is exactly what a degenerate cost
        does — would otherwise set the limits for both panels and render the
        truth as a flat gray square, losing the one thing the figure is for.
        The fit is clipped instead, and its own range is stated in the title so
        nothing is hidden, only moved from the color channel to the text.
        =#
        lim = max(maximum(abs, tru), 1e-9)
        flim = maximum(abs, fit)
        s = score(fit, tru; sym=sym)
        dlim = max(maximum(abs, fit .- tru), 1e-9)
        clipped = flim > lim * 1.02 ? @sprintf(" [fill clipped; |fit| ≤ %.3g]", flim) : ""
        push!(
            panels,
            heat(tru; clims=(-lim, lim), title=@sprintf("%s — truth  (±%.3g)", bname, lim)),
        )
        push!(
            panels,
            heat(
                fit;
                clims=(-lim, lim),
                title=@sprintf("recovered — rmse %.3f, r %.2f%s", s.rmse, s.corr, clipped)
            ),
        )
        push!(
            panels,
            heat(
                fit .- tru;
                clims=(-dlim, dlim),
                title=@sprintf("residual (fit − truth), ±%.3g", dlim)
            ),
        )
    end
    nrow = length(bs)
    plt = plot(
        panels...;
        layout=grid(nrow, 3),
        size=(1080, 250nrow + 40),
        plot_title=title,
        plot_titlefontsize=11,
        left_margin=4Plots.mm,
        bottom_margin=2Plots.mm,
    )
    return save_fig(plt, name)
end

"""
    scatter_figure(res, name; title) -> path

The same comparison read entrywise: recovered against true, one panel per block,
with the identity line. This is the view that separates a *scale* error (points
on a line through the origin, off the diagonal) from a *structural* one (points
off any line), which the two summary numbers can only hint at.
"""
function scatter_figure(res, name::AbstractString; title::String="")
    bs = blocks_of(res)
    panels = Any[]
    for (bname, tru, fit, sym) in bs
        x = _entries(tru; sym=sym)
        y = _entries(fit; sym=sym)
        lim = max(maximum(abs, x), maximum(abs, y), 1e-9) * 1.1
        s = score(fit, tru; sym=sym)
        p = plot(
            [-lim, lim],
            [-lim, lim];
            color=INK_MUTED,
            lw=0.8,
            label="",
            base_attrs(;
                xlims=(-lim, lim),
                ylims=(-lim, lim),
                aspect_ratio=:equal,
                xlabel="truth",
                ylabel="recovered",
                title=@sprintf("%s\nrmse %.3f   r %.2f", bname, s.rmse, s.corr),
            )...,
        )
        scatter!(
            p,
            x,
            y;
            color=SERIES[1],
            markerstrokecolor=SURFACE,
            markerstrokewidth=1.2,
            markersize=4.5,
            label="",
        )
        push!(panels, p)
    end
    ncol = min(3, length(panels))
    nrow = cld(length(panels), ncol)
    plt = plot(
        panels...;
        layout=grid(nrow, ncol),
        size=(330ncol, 330nrow + 40),
        plot_title=title,
        plot_titlefontsize=11,
        left_margin=3Plots.mm,
        bottom_margin=3Plots.mm,
    )
    return save_fig(plt, name)
end

# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

# `center` lives in `report.jl`, which has no plotting dependency.

const RMSE_FLOOR = 1e-4   # a log axis has no room for an exact zero

"""
    log_ticks(lo, hi) -> (positions, labels)

Ticks covering `[lo, hi]`, labelled as plain decimals rather than exponents.

Left to itself a log axis labels `10^-0.5`, which is an exponent where the
reader wants a number. Which mantissa ladder to use depends on how wide the span
is: a panel covering three decades wants decade ticks, one covering a factor of
two wants something finer or it gets a single label and a lot of unexplained
gridlines. So the ladders are tried coarse to fine and the first that yields at
least three ticks wins.
"""
function log_ticks(lo::Real, hi::Real)
    ladders = ([1.0], [1.0, 3.0], [1.0, 2.0, 5.0], [1.0, 1.5, 2.0, 3.0, 5.0, 7.0])
    fmt(v) = v >= 1 ? @sprintf("%g", v) : @sprintf("%.3g", v)
    best = nothing
    for mantissas in ladders
        keep = Float64[]
        for e in -6:6, m in mantissas
            v = m * 10.0^e
            (v >= lo / 1.05 && v <= hi * 1.05) && push!(keep, v)
        end
        isempty(keep) && continue
        best = keep
        length(keep) >= 3 && break
    end
    best === nothing && return :auto
    return (best, fmt.(best))
end

"""
    sweep_figure(name; panels, xlabel, xticks, logx, title) -> path

A grid of line panels, one per metric. `panels` is a vector of

    (title, xs, Vector{(series_label, ys, errs)})

Each series is `(label, centers, errors)`, where `errors` is `nothing` or a
`(below, above)` pair of distances — the asymmetric form, because what these
sweeps have to show is a median with a long upper tail rather than a symmetric
spread. Every panel shares the series order, so a series keeps its color across the
whole figure. At most three series: past that, color-vision separation on an
all-pairs form is not achievable from a fixed palette, and the answer is to
facet rather than to add a fourth hue.

The legend is drawn once, on the first panel. One legend per panel is the same
three labels repeated three times, and at this panel width they collide with
each other and with the titles — so the figure spends a third of its ink saying
nothing new and clips the labels while doing it.

The y axis is relative RMSE on a log scale — these span two or three orders of
magnitude across conditions, and a linear axis would compress everything
interesting into the bottom pixel row. Error bars are clamped to the same floor
as the centers, since a lower arm reaching zero has nowhere to be drawn.
"""
function sweep_figure(
    name::AbstractString;
    panels,
    xlabel::String,
    xticks=nothing,
    logx::Bool=false,
    title::String="",
    ylabel::String="relative RMSE",
    ylog::Bool=true,
)
    ps = Any[]
    for (ptitle, xs, series) in panels
        length(series) <= 3 || error("at most three series per panel; got $(length(series))")
        #=
        Explicit ticks on both axes. The x values here are counts (targets,
        trials, iterations) and belong on the axis as themselves; the y values
        span decades and belong on a 1-3-10 ladder. Plots' own log formatter
        writes exponents for both, which is the wrong unit for a reader.
        =#
        allys = Float64[]
        for (_, ys, errs) in series
            yy = ylog ? max.(Float64.(ys), RMSE_FLOOR) : Float64.(ys)
            append!(allys, yy)
            errs isa Tuple && append!(allys, yy .+ errs[2])
        end
        fin = filter(isfinite, allys)
        lo, hi = isempty(fin) ? (0.1, 1.0) : extrema(fin)
        #=
        Pad the limits past the data, then read the ticks off the *padded* range.
        Fit to the data exactly and a tick landing on the extreme value is drawn
        half outside the axis with its label clipped; read the ticks off the
        unpadded range instead and the padding is left conspicuously unlabelled.
        =#
        yl = if isempty(fin)
            :auto
        elseif ylog
            (max(lo / 1.45, RMSE_FLOOR / 1.45), hi * 1.45)
        else
            pad = 0.08 * max(hi - lo, 1e-9)
            (lo - pad, hi + pad)
        end
        yt = (ylog && yl !== :auto) ? log_ticks(yl[1], yl[2]) : :auto
        xt = xticks !== nothing ? xticks : (logx ? (xs, string.(xs)) : :auto)
        p = plot(;
            base_attrs(;
                xlabel=xlabel,
                ylabel=ylabel,
                title=ptitle,
                xscale=(logx ? :log10 : :identity),
                yscale=(ylog ? :log10 : :identity),
                legend=(length(series) > 1 && isempty(ps) ? :best : false),
            )...,
        )
        for (i, (lbl, ys, errs)) in enumerate(series)
            yy = ylog ? max.(ys, RMSE_FLOOR) : ys
            #=
            Clamp the lower arm so it stops at the axis floor rather than at
            zero, which a log axis cannot draw and which silently drops the whole
            error bar when it happens.
            =#
            ee = if errs === nothing
                nothing
            elseif errs isa Tuple
                (min.(errs[1], yy .- RMSE_FLOOR), errs[2])
            else
                errs
            end
            plot!(
                p,
                xs,
                yy;
                yerror=ee,
                color=SERIES[i],
                lw=2,
                marker=:circle,
                markersize=4,
                markerstrokecolor=SURFACE,
                markerstrokewidth=1.2,
                label=lbl,
            )
        end
        #=
        Ticks and limits go on *after* the series. Set on an empty `plot()` they
        are discarded when the first series arrives and the axis reverts to
        Plots' own choice — which on a log axis is the exponent labels this whole
        helper exists to avoid.
        =#
        plot!(p; xticks=xt, yticks=yt, ylims=yl)
        push!(ps, p)
    end
    ncol = min(3, length(ps))
    nrow = cld(length(ps), ncol)
    plt = plot(
        ps...;
        layout=grid(nrow, ncol),
        size=(460ncol, 400nrow),
        plot_title=title,
        plot_titlefontsize=11,
        left_margin=6Plots.mm,
        right_margin=4Plots.mm,
        top_margin=5Plots.mm,
        bottom_margin=6Plots.mm,
    )
    return save_fig(plt, name)
end

"""
    dot_figure(name; labels, panels, title, xlabel, logx) -> path

One dot per condition, for the axes that are a list of choices rather than a
scale — which fitting procedure, which structural feature.

A dot plot, not a bar chart, and for a specific reason: these values span
decades and belong on a log axis, and a bar on a log axis is a lie — a bar
encodes a *length* measured from zero, and zero is not on the axis, so its
length would depend on where the axis happened to be cut. A dot encodes a
position, which is exactly what a log axis supports. The hairline leader is a
reading aid to the label, not the encoding.

One color for every dot: the conditions have no natural order, so a ramp would
double-encode the value as hue and burn the free channel on information the
position already carries.
"""
function dot_figure(
    name::AbstractString;
    labels,
    panels,
    title::String="",
    xlabel::String="relative RMSE",
    logx::Bool=true,
)
    ps = Any[]
    for (ptitle, vals) in panels
        vv = logx ? max.(Float64.(vals), RMSE_FLOOR) : Float64.(vals)
        fin = filter(isfinite, vv)
        lo, hi = isempty(fin) ? (0.1, 1.0) : extrema(fin)
        xt = logx ? log_ticks(lo, hi) : :auto
        p = plot(;
            base_attrs(;
                yticks=(1:length(labels), labels),
                yflip=true,
                ylims=(0.4, length(labels) + 0.6),
                xlabel=xlabel,
                title=ptitle,
                xscale=(logx ? :log10 : :identity),
                xticks=xt,
                legend=false,
            )...,
        )
        for (i, v) in enumerate(vv)
            isfinite(v) || continue
            plot!(p, [lo / 1.3, v], [i, i]; color=GRIDLINE, lw=1.2, label="")
        end
        scatter!(
            p,
            vv,
            1:length(labels);
            color=SERIES[1],
            markersize=7,
            markerstrokecolor=SURFACE,
            markerstrokewidth=1.5,
            label="",
        )
        push!(ps, p)
    end
    ncol = min(3, length(ps))
    nrow = cld(length(ps), ncol)
    plt = plot(
        ps...;
        layout=grid(nrow, ncol),
        size=(440ncol, 34 * length(labels) * nrow + 150nrow),
        plot_title=title,
        plot_titlefontsize=11,
        left_margin=34Plots.mm,
        bottom_margin=6Plots.mm,
    )
    return save_fig(plt, name)
end

"""
    grid_figure(name; xs, ys, z, xlabel, ylabel, title, zlabel) -> path

A two-axis search over settings, drawn as an annotated heatmap.

The value is a *magnitude* — a relative error, which has no meaningful sign and
no meaningful zero-crossing — so the scale here is **sequential**: one hue, light
to dark, light meaning "small error". The diverging ramp the parameter figures
use would be wrong twice over: it would put its neutral midpoint somewhere
arbitrary and imply that the two arms mean opposite things.

Colour is on a log scale because that is how these errors are spread, but the
printed value is the number itself, so nothing has to be read off the ramp. The
best cell is ringed rather than recoloured — recolouring it would break the
ramp's one job.
"""
function grid_figure(
    name::AbstractString;
    xs,
    ys,
    z,
    xlabel::String,
    ylabel::String,
    title::String="",
)
    zz = Matrix{Float64}(z)
    fin = filter(isfinite, vec(zz))
    lo = isempty(fin) ? RMSE_FLOOR : max(minimum(fin), RMSE_FLOOR)
    hi = isempty(fin) ? 1.0 : max(maximum(fin), lo * 1.01)
    logz = log10.(clamp.(zz, lo, hi))
    ramp = cgrad(["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    p = heatmap(
        1:length(xs),
        1:length(ys),
        logz;
        c=ramp,
        clims=(log10(lo), log10(hi)),
        yflip=false,
        colorbar=false,
        xticks=(1:length(xs), string.(xs)),
        yticks=(1:length(ys), string.(ys)),
        xlims=(0.5, length(xs) + 0.5),
        ylims=(0.5, length(ys) + 0.5),
        base_attrs(; xlabel=xlabel, ylabel=ylabel, title=title, grid=false)...,
    )
    span = log10(hi) - log10(lo)
    for i in eachindex(ys), j in eachindex(xs)
        v = zz[i, j]
        isfinite(v) || continue
        frac = span <= 0 ? 0.0 : (logz[i, j] - log10(lo)) / span
        annotate!(p, j, i, text(_cellfmt(v, hi), 7, frac > 0.55 ? :white : INK, :center))
    end
    if !isempty(fin)
        best = argmin(map(v -> isfinite(v) ? v : Inf, zz))
        scatter!(
            p,
            [best[2]],
            [best[1]];
            marker=:rect,
            markersize=22,
            markeralpha=0,
            markerstrokecolor="#eb6834",
            markerstrokewidth=2.5,
            label="",
        )
    end
    plt = plot(
        p;
        size=(120 * length(xs) + 240, 90 * length(ys) + 190),
        left_margin=6Plots.mm,
        bottom_margin=6Plots.mm,
    )
    return save_fig(plt, name)
end

# ---------------------------------------------------------------------------
# Switching figures
# ---------------------------------------------------------------------------

"""
    slds_trial_figure(res, name; title) -> path

One trial, read top to bottom: what was observed, and what the fit made of it.

The top panel is the observed channels with the true LQR epoch shaded, so the
switch is visible in the data rather than only in the legend. The bottom panel
is `γ(LQR state)` over time for the fit and, as a second series, for the
*generating* parameters — the reference this section reads the fit against. The
true switch is a vertical rule on both.
"""
function slds_trial_figure(res, name::AbstractString; title::String="")
    ex = res.example
    T = size(ex.y, 2)
    t_sw = something(findfirst(==(LQR_STATE), ex.z), T)
    shade = [z == LQR_STATE ? 1.0 : 0.0 for z in ex.z]

    top = plot(;
        base_attrs(;
            ylabel="observed y",
            title="one trial — shaded band is the true LQR epoch",
            legend=:outertopright,
        )...,
    )
    ylo, yhi = extrema(ex.y)
    pad = 0.1 * max(yhi - ylo, 1e-6)
    plot!(
        top,
        1:T,
        fill(yhi + pad, T);
        fillrange=fill(ylo - pad, T),
        fillalpha=(0.0 .+ 0.13 .* shade),
        linealpha=0,
        color=SERIES[1],
        label="",
    )
    #=
    At most three channels get a palette slot. Beyond that the remainder is one
    muted group with one legend entry: a fourth and fifth series drawn in a
    repeated hue would say "these two are the same thing", which is exactly the
    claim a categorical palette is supposed to make and exactly the wrong one
    here. With the costate observed there are `2n` channels and only the first
    `n` are the state, so the group is meaningful rather than a fudge.
    =#
    nch = size(ex.y, 1)
    for i in 1:min(nch, 3)
        plot!(top, 1:T, ex.y[i, :]; color=SERIES[i], lw=1.6, label="channel $i")
    end
    for i in 4:nch
        plot!(
            top,
            1:T,
            ex.y[i, :];
            color=INK_MUTED,
            lw=1.2,
            alpha=0.7,
            label=(i == 4 ? "channels 4–$nch" : ""),
        )
    end
    vline!(top, [t_sw]; color=INK_MUTED, lw=1.2, label="")
    ylims!(top, ylo - pad, yhi + pad)

    bot = plot(;
        base_attrs(;
            xlabel="timestep",
            ylabel="γ(LQR state)",
            ylims=(-0.03, 1.03),
            title="posterior over the discrete state",
            legend=:outertopright,
        )...,
    )
    plot!(bot, 1:T, ex.γ_truth[LQR_STATE, :]; color=SERIES[2], lw=2, label="at the truth")
    plot!(bot, 1:T, ex.γ[LQR_STATE, :]; color=SERIES[1], lw=2, label="fitted")
    vline!(bot, [t_sw]; color=INK_MUTED, lw=1.2, label="true switch")

    plt = plot(
        top,
        bot;
        layout=grid(2, 1; heights=[0.55, 0.45]),
        size=(880, 560),
        plot_title=title,
        plot_titlefontsize=11,
        left_margin=6Plots.mm,
        bottom_margin=5Plots.mm,
    )
    return save_fig(plt, name)
end

"""
    gamma_figure(name; labels, fitted, reference, title) -> path

Balanced γ accuracy per condition, with γ at the generating parameters marked on
the same row. Two encodings rather than two bars: the fit is the bar, the truth
is a rule across it, because they are not two comparable series — one is the
result and the other is the reference it is read against.

`0.5` is where a two-state model that has learned nothing lands, and it is drawn,
because a bar chart starting at zero makes 0.52 look like an achievement.
"""
function gamma_figure(
    name::AbstractString; labels, fitted, reference, title::String="", chance::Float64=0.5
)
    p = bar(
        1:length(labels),
        fitted;
        orientation=:horizontal,
        color=SERIES[1],
        linecolor=SURFACE,
        linewidth=1.5,
        bar_width=0.5,
        label="fitted",
        base_attrs(;
            yticks=(1:length(labels), labels),
            yflip=true,
            ylims=(0.4, length(labels) + 0.6),
            xlabel="balanced accuracy of the MAP state",
            xlims=(0, 1.02),
            xticks=(0:0.25:1, ["0", "0.25", "0.5", "0.75", "1"]),
            title=title,
            legend=:outertop,
            legend_columns=-1,
        )...,
    )
    scatter!(
        p,
        reference,
        1:length(labels);
        marker=:vline,
        markersize=11,
        markerstrokewidth=2.5,
        color=SERIES[2],
        label="at the truth",
    )
    vline!(p, [chance]; color=INK_MUTED, lw=1.2, label="chance")
    plt = plot(
        p;
        size=(800, 38 * length(labels) + 190),
        left_margin=40Plots.mm,
        bottom_margin=6Plots.mm,
    )
    return save_fig(plt, name)
end

"""
    elbo_figure(name; traces, title) -> path

The ELBO trace of each fit against the ELBO at the generating parameters, which
is drawn as a rule at zero — every trace is plotted as `elbo − truth_elbo`, so
"above the line" means the model prefers something other than the truth, and
"still climbing at the right edge" means the iteration budget, not the data, set
the answer.
"""
function elbo_figure(name::AbstractString; traces, title::String="")
    p = plot(;
        base_attrs(;
            xlabel="EM iteration",
            ylabel="ELBO − ELBO(truth)   [nats]",
            title=title,
            legend=:best,
        )...,
    )
    hline!(p, [0.0]; color=INK_MUTED, lw=1.2, label="truth")
    for (i, (lbl, tr)) in enumerate(traces)
        plot!(p, eachindex(tr), collect(tr); color=SERIES[min(i, 3)], lw=2, label=lbl)
    end
    plt = plot(p; size=(760, 430), left_margin=6Plots.mm, bottom_margin=5Plots.mm)
    return save_fig(plt, name)
end
