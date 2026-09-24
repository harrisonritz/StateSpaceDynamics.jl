#=============================================================================
Benchmark / profile driver for the inverse-LQR EM fit.

    julia --project=benchmark/lqr -t 16 benchmark/lqr/run.jl [options]

Options (all optional):
  --workload=plqr,plqr_joint,slqr   which workloads (default: plqr)
  --tier=small                      tiny | small | medium | smoulder
  --iters=3                         EM iterations per timed fit (convergence off)
  --reps=1                          timed repetitions (min is reported)
  --spread=0.35                     log-normal length spread (0 = equal lengths)
  --ntrials=N                       override the tier's trial count
  --mstep-iters=100                 L-BFGS iterations per structural M-step
  --blas-threads=1                  BLAS threads (1 avoids oversubscription)
  --profile                         phase-attributed sampling profile of one fit
  --csv=path                        append one row per (workload, rep) to a CSV
  --dump=path                       serialize ELBO trace + fitted parameters
                                    (compare two dumps with compare.jl)
  --label=text                      free-text tag stored in the CSV (e.g. a rev)
  --perturb=δ                       scale the initial emission loadings by (1+δ):
                                    a rounding-sized perturbation whose effect on
                                    the dump is the noise floor for compare.jl

Every fit starts from a `deepcopy` of the same fit-ready model and runs exactly
`--iters` EM iterations, so wall times and dumps are comparable across revisions
and thread counts. A warm-up fit on the `:tiny` tier compiles every method first.
=============================================================================#

using Dates
using LinearAlgebra
using Logging
using Printf
using Serialization
using Statistics

include(joinpath(@__DIR__, "workloads.jl"))
include(joinpath(@__DIR__, "phaseprof.jl"))

function parse_args(args)
    opts = Dict{String,String}()
    for a in args
        startswith(a, "--") || error("unrecognised argument $a")
        kv = split(a[3:end], "="; limit=2)
        opts[kv[1]] = length(kv) == 2 ? kv[2] : "true"
    end
    return opts
end

"""The revision of the StateSpaceDynamics actually loaded (it may be a worktree)."""
function git_rev()
    dir = pkgdir(StateSpaceDynamics)
    try
        rev = strip(read(`git -C $dir rev-parse --short HEAD`, String))
        dirty = !isempty(read(`git -C $dir status --porcelain -- src`, String))
        return rev * (dirty ? "+dirty" : "")
    catch
        return "unknown"
    end
end

"""Fitted parameters as plain arrays, keyed by name, for `compare.jl`."""
function snapshot(model)
    out = Dict{String,Array{Float64}}()
    function add_lds!(prefix, lds)
        sm = lds.state_model
        for key in (:A, :S, :Gref, :h, :Bu, :Σ, :Σf, :hf, :x0, :P0)
            hasproperty(sm, key) || continue
            out["$prefix.$key"] = copy(Array{Float64}(getproperty(sm, key)))
        end
        if hasproperty(sm, :Qc)
            labels = try
                group_labels(sm, :Qc)
            catch
                Any[]
            end
            if !isempty(labels)
                for r in labels, (k, Q) in enumerate(group_variant(sm, :Qc, r).Qc)
                    out["$prefix.Qc[$r][$k]"] = copy(Matrix{Float64}(Q))
                end
            else
                for (k, Q) in enumerate(sm.Qc)
                    out["$prefix.Qc[$k]"] = copy(Matrix{Float64}(Q))
                end
            end
        end
        out["$prefix.C"] = copy(Matrix{Float64}(lds.obs_model.C))
        return out["$prefix.d"] = copy(Vector{Float64}(lds.obs_model.d))
    end
    if model isa SLDS
        for (k, lds) in enumerate(model.LDSs)
            add_lds!("lds$k", lds)
        end
        out["slds.A"] = copy(model.A)
        out["slds.π"] = copy(model.πₖ)
    else
        add_lds!("lds", model)
    end
    return out
end

"""
    perturb!(model, δ)

Scale every emission loading `C` by `1 + δ` in place. With `δ ≈ 1e-14` this is a
perturbation of the size floating-point reordering produces, so the spread of
fits under it is the floor below which two revisions cannot be told apart.
"""
function perturb!(model, δ)
    δ == 0 && return model
    ldss = model isa SLDS ? model.LDSs : [model]
    for lds in ldss
        lds.obs_model.C .*= (1 + δ)
    end
    return model
end

const CSV_HEADER =
    "timestamp,label,rev,workload,tier,threads,blas_threads,iters,rep," *
    "wall_s,per_iter_s,latent_dim,ntrials,obs_dim,tmin,tmax," *
    "ndistinct_lengths,ndistinct_designs,total_bins,mstep_iters,final_elbo"

function main(args=ARGS)
    opts = parse_args(args)
    workloads = Symbol.(split(get(opts, "workload", "plqr"), ","))
    tier = Symbol(get(opts, "tier", "small"))
    iters = parse(Int, get(opts, "iters", "3"))
    reps = parse(Int, get(opts, "reps", "1"))
    spread = parse(Float64, get(opts, "spread", "0.35"))
    ntrials = haskey(opts, "ntrials") ? parse(Int, opts["ntrials"]) : nothing
    mstep_iters = parse(Int, get(opts, "mstep-iters", "100"))
    BLAS.set_num_threads(parse(Int, get(opts, "blas-threads", "1")))
    label = get(opts, "label", "")
    δ = parse(Float64, get(opts, "perturb", "0"))
    rev = git_rev()

    global_logger(ConsoleLogger(stderr, Logging.Error))   # the fits warn by design
    println(
        "rev=$rev threads=$(Threads.nthreads()) (+$(Threads.nthreads(:interactive)) ",
        "interactive) BLAS=$(BLAS.get_num_threads()) tier=$tier iters=$iters",
    )

    for name in workloads
        # warm-up: identical types at the tiny size compile every method.
        wt = build_workload(name; tier=:tiny, spread=spread, mstep_iters=mstep_iters)
        run_fit!(deepcopy(wt.model), wt; max_iter=iters)

        w = build_workload(
            name; tier=tier, spread=spread, ntrials=ntrials, mstep_iters=mstep_iters
        )
        m = w.meta
        println(
            "\n== $name  latent=$(m.latent_dim) N=$(m.ntrials) p=$(m.obs_dim) ",
            "T∈[$(m.tmin),$(m.tmax)] distinct lengths=$(m.ndistinct_lengths) ",
            "designs=$(m.ndistinct_designs) bins=$(m.total_bins)",
        )
        GC.gc()
        times = Float64[]
        trace = nothing
        fitted = nothing
        for rep in 1:reps
            model = perturb!(deepcopy(w.model), δ)
            t = @elapsed trace = run_fit!(model, w; max_iter=iters)
            fitted = model
            push!(times, t)
            @printf(
                "  rep %d: %.3f s  (%.3f s/iter)  final ELBO %.10g\n",
                rep,
                t,
                t / iters,
                last(collect(trace))
            )
            if haskey(opts, "csv")
                path = opts["csv"]
                newfile = !isfile(path)
                open(path, "a") do io
                    newfile && println(io, CSV_HEADER)
                    println(
                        io,
                        join(
                            (
                                now(),
                                label,
                                rev,
                                name,
                                tier,
                                Threads.nthreads(),
                                BLAS.get_num_threads(),
                                iters,
                                rep,
                                t,
                                t / iters,
                                m.latent_dim,
                                m.ntrials,
                                m.obs_dim,
                                m.tmin,
                                m.tmax,
                                m.ndistinct_lengths,
                                m.ndistinct_designs,
                                m.total_bins,
                                m.mstep_iters,
                                last(collect(trace)),
                            ),
                            ",",
                        ),
                    )
                end
            end
            GC.gc()
        end
        @printf("  min %.3f s  median %.3f s\n", minimum(times), median(times))

        if haskey(opts, "dump")
            path = replace(opts["dump"], "{workload}" => string(name))
            serialize(
                path,
                (;
                    workload=name,
                    tier,
                    iters,
                    rev,
                    threads=Threads.nthreads(),
                    elbo=collect(Float64, collect(trace)),
                    params=snapshot(fitted),
                ),
            )
            println("  dumped → $path")
        end

        if haskey(opts, "profile")
            rep = phase_profile(() -> run_fit!(deepcopy(w.model), w; max_iter=iters))
            println("\n  phase profile (depth 1):")
            print_phase_report(stdout, rollup(rep; depth=1))
            println("\n  phase profile (full paths):")
            print_phase_report(stdout, rep)
        end
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
