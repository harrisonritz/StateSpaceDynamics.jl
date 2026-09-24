#=============================================================================
Scaling / comparison table from a `run.jl --csv` file.

    julia --project=benchmark/lqr benchmark/lqr/summarize.jl timings.csv

For each (workload, tier, label) it reports the best wall time per EM iteration
at each thread count, the speedup over the smallest thread count present, and
the parallel efficiency `speedup / (threads / threads_min)`. When several labels
(revisions) share a thread count it also prints their ratio to the first label.
=============================================================================#

using Printf

function read_csv(path)
    lines = readlines(path)
    header = split(lines[1], ",")
    rows = [Dict(zip(header, split(l, ","))) for l in lines[2:end] if !isempty(l)]
    return filter(r -> get(r, "timestamp", "") != "timestamp", rows)
end

function summarize(path; io=stdout)
    rows = read_csv(path)
    groups = Dict{Tuple{String,String,String},Dict{Int,Float64}}()
    for r in rows
        key = (r["workload"], r["tier"], r["label"])
        th = parse(Int, r["threads"])
        t = parse(Float64, r["per_iter_s"])
        d = get!(groups, key, Dict{Int,Float64}())
        d[th] = min(get(d, th, Inf), t)
    end
    for key in sort!(collect(keys(groups)))
        (wl, tier, label) = key
        d = groups[key]
        ths = sort!(collect(keys(d)))
        t0, th0 = d[ths[1]], ths[1]
        @printf(io, "\n%s / %s%s\n", wl, tier, isempty(label) ? "" : " [$label]")
        @printf(io, "  %8s %12s %9s %11s\n", "threads", "s / iter", "speedup", "efficiency")
        for th in ths
            s = t0 / d[th]
            @printf(io, "  %8d %12.3f %8.2fx %10.0f%%\n", th, d[th], s, 100s / (th / th0))
        end
    end
    labels = unique(k[3] for k in keys(groups))
    if length(labels) > 1
        @printf(
            io, "\nlabel ratios (time of first label / time of other; >1 means faster):\n"
        )
        ref = first(sort(labels))
        for (wl, tier, label) in sort!(collect(keys(groups)))
            label == ref && continue
            haskey(groups, (wl, tier, ref)) || continue
            a, b = groups[(wl, tier, ref)], groups[(wl, tier, label)]
            for th in sort!(collect(intersect(keys(a), keys(b))))
                @printf(
                    io,
                    "  %-10s %-8s %4d threads: %s/%s = %.2fx\n",
                    wl,
                    tier,
                    th,
                    ref,
                    label,
                    a[th] / b[th]
                )
            end
        end
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && summarize(ARGS[1])
