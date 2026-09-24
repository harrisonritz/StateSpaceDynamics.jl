#=============================================================================
Compare fit dumps written by `run.jl --dump=...`.

    julia --project=benchmark/lqr benchmark/lqr/compare.jl base.jls new.jls [noise1.jls ...]
          [--rtol=1e-8] [--factor=3]

With only `base` and `new`: the relative difference `‖new − base‖∞ / ‖base‖∞` of
the ELBO trace and every fitted parameter array, PASS if all are below `--rtol`.
That is the right test for a change that should be bitwise identical, or nearly
so, and for well-conditioned fits.

With noise dumps (the base revision refitted from initial values perturbed by
~1e-14; `compare_revs.sh` makes them): the reference ensemble is
`{base, noise…}`, and each parameter's *noise spread* is the largest pairwise
difference within it. `new` is flagged only where its distance to `base` exceeds
`--factor` × that spread.

Why the ensemble. The terminal-conditioned LQR fit amplifies rounding: its
structural M-step is 100 L-BFGS iterations with an accept/halve step on an
objective that is nearly flat along `x0`, `hf` and the costate gauge, so a 1e-14
change in the starting point moves those by O(10%) within two EM iterations,
while `A`, `S`, `Qc` move by 1e-5…1e-3. Any change that reorders floating-point
work — parallel reductions, batched BLAS — perturbs the fit by that much and no
less. A tolerance on the parameters alone cannot separate that from a bug; the
spread of the unchanged code under the same-sized perturbation can. Differences
far inside the spread are rounding; one well outside it is a change in the
algorithm. For the final ELBO, a `new` value above the ensemble's range is a
better optimum reached, not an error.
=============================================================================#

using Serialization
using Printf

function relerr(a, b)
    size(a) == size(b) || return Inf
    scale = max(maximum(abs, a; init=0.0), eps())
    return maximum(abs.(a .- b); init=0.0) / scale
end

function compare_dumps(pbase, pnew, pnoise=String[]; rtol=1e-8, factor=3.0, io=stdout)
    base, new = deserialize(pbase), deserialize(pnew)
    noise = [deserialize(p) for p in pnoise]
    ref = [base; noise]
    @printf(io, "base:  %s (rev %s, %d threads)\n", pbase, base.rev, base.threads)
    @printf(io, "new:   %s (rev %s, %d threads)\n", pnew, new.rev, new.threads)
    isempty(noise) || @printf(io, "noise: %d perturbed refits of base\n", length(noise))

    get_elbo(d) = d.elbo
    get_param(key) = d -> d.params[key]
    keys_all = sort!(collect(union(keys(base.params), keys(new.params))))
    items = [("ELBO trace", get_elbo); [(k, get_param(k)) for k in keys_all]]

    spread(getter) = begin
        s = 0.0
        for i in eachindex(ref), j in (i + 1):length(ref)
            s = max(s, relerr(getter(ref[i]), getter(ref[j])))
        end
        s
    end

    worst = 0.0
    flagged = String[]
    if isempty(noise)
        @printf(io, "  %-24s %12s\n", "", "rel diff")
    else
        @printf(io, "  %-24s %12s %12s  %s\n", "", "new vs base", "noise spread", "verdict")
    end
    for (name, getter) in items
        present = all(
            d -> name == "ELBO trace" || haskey(d.params, name), [base; new; noise]
        )
        if !present
            @printf(io, "  %-24s missing on one side\n", name)
            push!(flagged, name)
            continue
        end
        e = relerr(getter(base), getter(new))
        worst = max(worst, e)
        if isempty(noise)
            e > 0 && @printf(io, "  %-24s %12.3e\n", name, e)
            e > rtol && push!(flagged, name)
        else
            s = spread(getter)
            verdict = e <= max(factor * s, rtol) ? "within noise" : "OUTSIDE NOISE"
            verdict == "OUTSIDE NOISE" && push!(flagged, name)
            @printf(io, "  %-24s %12.3e %12.3e  %s\n", name, e, s, verdict)
        end
    end

    finals = [last(d.elbo) for d in ref]
    lo, hi = extrema(finals)
    fnew = last(new.elbo)
    where_ = if fnew > hi
        "above the ensemble (a better optimum)"
    elseif fnew < lo
        "below the ensemble"
    else
        "inside the ensemble"
    end
    @printf(
        io,
        "  final ELBO: new %.10g; reference ensemble [%.10g, %.10g] → %s\n",
        fnew,
        lo,
        hi,
        where_
    )
    if isempty(flagged)
        @printf(io, "  PASS (worst relative difference %.3e)\n", worst)
    else
        @printf(io, "  FAIL: %s\n", join(flagged, ", "))
    end
    return isempty(flagged)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    function opt(name, default)
        i = findfirst(a -> startswith(a, "--$name="), ARGS)
        return i === nothing ? default : parse(Float64, split(ARGS[i], "="; limit=2)[2])
    end
    files = filter(a -> !startswith(a, "--"), ARGS)
    length(files) >= 2 || error(
        "usage: compare.jl base.jls new.jls [noise.jls ...] [--rtol=1e-8] [--factor=3]"
    )
    ok = compare_dumps(
        files[1], files[2], files[3:end]; rtol=opt("rtol", 1e-8), factor=opt("factor", 3.0)
    )
    exit(ok ? 0 : 1)
end
