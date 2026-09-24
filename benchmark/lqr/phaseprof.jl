#=============================================================================
Phase attribution for a multi-threaded profile.

A flat profile answers "which function burns CPU". For a multi-core EM fit the
question is different: "which *phase* is wall-clock time spent in, and how many
cores are busy while it runs?". A phase that is 30% of wall time on one busy core
is the bottleneck on 64 cores, even if it is 3% of CPU samples.

Julia's sampler stops every thread at each tick and records one block per
thread, carrying `(threadid, taskid, sleep state)`. So per tick we know:

  * the root task's stack  → which phase the *fit* is in (wall-clock attribution;
    the root task sits in `wait` inside the phase while workers run a `tforeach`),
  * every other thread's stack → whether that core was doing work.

`phase_report` groups ticks by the root task's phase path and reports, for each
path, its share of wall-clock ticks and its mean number of busy threads
("parallelism"). A tick in which the root task is not on any thread — it is
parked in `wait` on a `tforeach` it spawned — is charged to the common context
of the root's previous and next visible paths, extended by the phase label the
busy workers are in; dropping those ticks would hide precisely the parallel
sections. (`Profile.@profile_walltime`, which samples parked tasks directly,
segfaults on multi-threaded runs in Julia 1.13.) A thread counts as busy when it is not flagged sleeping and its
stack is not parked in the scheduler (`poptask`, `wait`, `task_done_hook`, …).

Phases are recognised by function name, root to leaf, from `PHASES` below. The
path is the sequence of matched phase labels (consecutive repeats collapsed), so
`E-step > smooth` and `M-step state > probe > smooth` are separate rows. Names
are matched after stripping closure decoration (`#smooth!##0` → `smooth!`), so a
`tforeach` body counts as its enclosing function.
=============================================================================#

using Profile
using Printf

# (label, function names). Order does not matter; matching is by exact name.
const PHASES = [
    # drivers / E-step
    (
        "E-step",
        (
            :estep!,
            :_grouped_estep_elbo_poisson!,
            :_grouped_estep_elbo_gaussian!,
            :_vem_alternate!,
            :_slds_estep!,
        ),
    ),
    (
        "smooth",
        (
            :smooth!,
            :_slds_smooth_all!,
            :_slds_smooth_cell!,
            :_smooth_ragged_prefix!,
            :_smooth_bucket!,
            :_smooth_mean_only!,
        ),
    ),
    ("newton", (:newton_smooth!,)),
    ("cov (BTD inverse)", (:block_tridiagonal_inverse_logdet!, :_precompute_shared_cov!)),
    (
        "aggregate",
        (
            :_aggregate_td_suff_stats!,
            :_aggregate_td_suff_stats_weighted!,
            :_aggregate_lqr_stats!,
            :_aggregate_lqr_stats_weighted!,
            :_slds_aggregate_weighted!,
        ),
    ),
    ("forward-backward", (:forward_backward!, :_slds_fill_logL!)),
    ("ELBO", (:elbo!, :_poisson_q_obs_total, :Q_state!, :_slds_trial_elbo)),
    # M-step, state side
    ("M-step state", (:_lqr_state_mstep!, :_grouped_state_mstep!, :_slds_state_mstep!)),
    ("conditional M-step", (:_lqr_conditional_mstep!,)),
    ("L-BFGS", (:optimize,)),
    ("accept", (:_lqr_accept_conditional!,)),
    ("probe", (:_lqr_probe_statistics!, :_terminal_probe_stats!, :_slqr_probe_estep!)),
    ("probe aggregate", (:_lqr_probe_aggregate!,)),
    ("log Z", (:_lqr_terminal_logz_sum, :_lqr_terminal_logz, :_slds_terminal_trial_logz)),
    ("structure f/g", (:_lqr_fg!,)),
    (
        "probe setup",
        (:_lqr_terminal_probe, :_slqr_terminal_probe, :_lqr_conditional_problem),
    ),
    # M-step, emission side
    (
        "M-step emission",
        (
            :update_observation_model!,
            :_grouped_update_observation_model!,
            :_tied_poisson_emission!,
            :_lqr_obs_mstep!,
        ),
    ),
]

const _PHASE_OF = Dict{Symbol,String}(f => label for (label, fs) in PHASES for f in fs)

const _IDLE_FRAMES = Set([
    :poptask,
    :wait,
    :task_done_hook,
    :wait_forever,
    :try_yieldto,
    :jl_task_get_next,
    :ijl_task_get_next,
    :yield,
    :sleep,
    :jl_safepoint_wait_gc,
    :ijl_safepoint_wait_gc,
    # parked parallel-GC threads and generic condition waits
    :uv_cond_wait,
    :pthread_cond_wait,
    :jl_parallel_gc_threadfun,
    :jl_concurrent_gc_threadfun,
    :ijl_task_get_next,
])

_strip(name::Symbol) = begin
    s = String(name)
    s = lstrip(s, '#')
    i = findfirst('#', s)
    Symbol(i === nothing ? s : s[1:(i - 1)])
end

struct _Sample
    thread::Int
    task::UInt
    sleeping::Bool
    clock::UInt64         # cycle counter when the block was taken
    ips::Vector{UInt64}   # leaf → root
end

function _parse_samples(data::Vector{UInt64})
    samples = _Sample[]
    start = 1
    i = 1
    n = length(data)
    while i <= n
        if Profile.is_block_end(data, i)
            ips = data[start:(i - 6)]
            push!(
                samples,
                _Sample(
                    Int(data[i - Profile.META_OFFSET_THREADID]),
                    data[i - Profile.META_OFFSET_TASKID],
                    data[i - Profile.META_OFFSET_SLEEPSTATE] - 1 == 1,
                    data[i - Profile.META_OFFSET_CPUCYCLECLOCK],
                    ips,
                ),
            )
            start = i + 1
        end
        i += 1
    end
    return samples
end

"""Group consecutive per-thread blocks into ticks (a tick ends when a thread repeats)."""
function _ticks(samples)
    ticks = Vector{Vector{_Sample}}()
    seen = Set{Int}()
    cur = _Sample[]
    for s in samples
        if s.thread in seen
            push!(ticks, cur)
            cur = _Sample[]
            empty!(seen)
        end
        push!(cur, s)
        push!(seen, s.thread)
    end
    isempty(cur) || push!(ticks, cur)
    return ticks
end

function _frames(s::_Sample, lidict)
    names = Symbol[]
    for ip in reverse(s.ips)          # root → leaf
        fr = get(lidict, ip, nothing)
        fr === nothing && continue
        frs = fr isa Vector ? fr : [fr]
        for j in length(frs):-1:1     # non-inlined root first, inlined leaf last
            push!(names, _strip(frs[j].func))
        end
    end
    return names
end

function _phase_labels(names)
    path = String[]
    for nm in names
        label = get(_PHASE_OF, nm, nothing)
        label === nothing && continue
        (isempty(path) || path[end] != label) && push!(path, label)
    end
    return path
end

function _phase_path(names)
    path = _phase_labels(names)
    return isempty(path) ? "(other)" : join(path, " > ")
end

function _busy(s::_Sample, names)
    s.sleeping && return false
    isempty(names) && return false
    # parked in the scheduler if any of the leaf-most few Julia frames is idle
    for nm in Iterators.take(Iterators.reverse(names), 12)
        nm in _IDLE_FRAMES && return false
    end
    return true
end

"""
    phase_profile(f; delay=0.002) -> report NamedTuple

Run `f()` under the sampling profiler (all threads) and attribute each tick to
the phase the calling task is in. Returns `(; rows, nticks, nthreads, wall)`
where each row is `(path, ticks, frac, parallelism)`.
"""
function phase_profile(f; delay=0.002, nsamples=10^7)
    Profile.clear()
    Profile.init(; n=nsamples, delay=delay)
    root = UInt(pointer_from_objref(current_task()))
    wall = @elapsed Profile.@profile f()
    data = Profile.fetch(; include_meta=true)
    lidict = Profile.getdict(data)
    samples = _parse_samples(data)
    ticks = _ticks(samples)
    namecache = Dict{UInt64,Any}()
    acc = Dict{String,Tuple{Int,Int}}()
    nthreads = maximum(s.thread for s in samples; init=1)
    #=
    Pass 1: the root task's own phase path in the ticks where it is on a thread,
    and the phase label the busy workers are in (the innermost label on their
    stacks — a `tforeach` body is named after the function that spawned it).
    =#
    nt = length(ticks)
    clocks = [minimum(s.clock for s in tick) for tick in ticks]
    tacc = Dict{String,Float64}()
    rootpath = Vector{Union{Nothing,Vector{String}}}(nothing, nt)
    workerlabel = Vector{Union{Nothing,String}}(nothing, nt)
    busycount = zeros(Int, nt)
    for (k, tick) in enumerate(ticks)
        votes = Dict{String,Int}()
        for s in tick
            names = _frames(s, lidict)
            b = _busy(s, names)
            busycount[k] += b
            if s.task == root
                rootpath[k] = _phase_labels(names)
            elseif b
                labels = _phase_labels(names)
                isempty(labels) || (votes[labels[end]] = get(votes, labels[end], 0) + 1)
            end
        end
        isempty(votes) || (workerlabel[k] = first(argmax(last, collect(votes))))
    end
    #=
    Pass 2: a tick where the root is not on any thread — parked in `wait` on a
    `tforeach` it spawned — is inside the common context of the root's previous
    and next visible paths, in the phase the workers are running. Charging it to
    the previous path alone would hand a short parallel section to whatever the
    root did just before it, which at millisecond phase lengths is badly biased.
    =#
    prevpath = [String[] for _ in 1:nt]
    nextpath = [String[] for _ in 1:nt]
    last_seen = String[]
    for k in 1:nt
        rootpath[k] === nothing || (last_seen = rootpath[k])
        prevpath[k] = last_seen
    end
    last_seen = String[]
    for k in nt:-1:1
        rootpath[k] === nothing || (last_seen = rootpath[k])
        nextpath[k] = last_seen
    end
    #=
    A maximal run of consecutive parked ticks is one wait, so it takes one worker
    label: the majority over the run. Ticks at the head and tail of a `tforeach`,
    where the workers are still being woken or already idle, then land in the
    phase the rest of the run was in rather than in the surrounding context.
    =#
    k = 1
    while k <= nt
        if rootpath[k] !== nothing
            k += 1
            continue
        end
        stop = k
        while stop < nt && rootpath[stop + 1] === nothing
            stop += 1
        end
        votes = Dict{String,Int}()
        for j in k:stop
            L = workerlabel[j]
            L === nothing || (votes[L] = get(votes, L, 0) + 1)
        end
        if !isempty(votes)
            L = first(argmax(last, collect(votes)))
            for j in k:stop
                workerlabel[j] = L
            end
        end
        k = stop + 1
    end
    for k in 1:nt
        labels = if rootpath[k] !== nothing
            rootpath[k]
        else
            a, b = prevpath[k], nextpath[k]
            n = 0
            while n < min(length(a), length(b)) && a[n + 1] == b[n + 1]
                n += 1
            end
            ctx = a[1:n]
            L = workerlabel[k]
            if L === nothing
                ctx
            else
                i = findlast(==(L), ctx)
                i === nothing ? vcat(ctx, L) : ctx[1:i]
            end
        end
        path = isempty(labels) ? "(other)" : join(labels, " > ")
        t, b = get(acc, path, (0, 0))
        acc[path] = (t + 1, b + busycount[k])
        dt = k < nt ? Float64(clocks[k + 1] - clocks[k]) : 0.0
        tacc[path] = get(tacc, path, 0.0) + dt
    end
    total = sum(first, values(acc); init=0)
    ttotal = max(sum(values(tacc); init=0.0), eps())
    rows = [
        (
            path=p,
            ticks=t,
            frac=t / max(total, 1),
            tfrac=tacc[p] / ttotal,
            parallelism=b / max(t, 1),
        ) for (p, (t, b)) in acc
    ]
    sort!(rows; by=r -> -r.ticks)
    return (; rows, nticks=total, nthreads, wall)
end

function print_phase_report(io::IO, rep; mincount=0.005)
    @printf(
        io,
        "wall %.2f s, %d ticks attributed, %d threads sampled\n",
        rep.wall,
        rep.nticks,
        rep.nthreads
    )
    @printf(io, "  %6s  %6s  %6s  %s\n", "time%", "tick%", "busy", "phase path (root task)")
    for r in sort(rep.rows; by=r -> -r.tfrac)
        max(r.frac, r.tfrac) < mincount && continue
        @printf(
            io,
            "  %5.1f%%  %5.1f%%  %6.2f  %s\n",
            100r.tfrac,
            100r.frac,
            r.parallelism,
            r.path
        )
    end
    return nothing
end

"""Roll the report up to the first `depth` labels of each path."""
function rollup(rep; depth=1)
    acc = Dict{String,NTuple{3,Float64}}()
    for r in rep.rows
        key = join(first(split(r.path, " > "), depth), " > ")
        t, f, b = get(acc, key, (0.0, 0.0, 0.0))
        acc[key] = (t + r.ticks, f + r.tfrac, b + r.parallelism * r.ticks)
    end
    total = max(rep.nticks, 1)
    rows = [
        (path=p, ticks=Int(t), frac=t / total, tfrac=f, parallelism=b / max(t, 1)) for
        (p, (t, f, b)) in acc
    ]
    return (; rows, rep.nticks, rep.nthreads, rep.wall)
end
