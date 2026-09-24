# Inverse-LQR EM: benchmarking and profiling

A harness for measuring where an inverse-LQR fit spends its time, how that
scales with threads, and whether a change to the code kept its answers. It is
built around the workload the LQR machinery is used for: Poisson counts,
event-aligned trials of **ragged, nearly all-distinct lengths**, costs grouped by
a known trial label (`depends_on`), and a terminal factor with the exact
terminal-conditioned objective (`condition_terminal = true`, the default).

```console
$ julia --project=benchmark/lqr -e 'using Pkg; Pkg.instantiate()'
$ julia --project=benchmark/lqr -t 8 benchmark/lqr/run.jl --workload=plqr --tier=small --iters=3 --profile
```

| file | what it does |
|---|---|
| `workloads.jl` | deterministic, fit-ready workloads (model + ragged data) at four sizes |
| `run.jl` | time fixed-iteration fits; CSV rows; parameter dumps; phase profile |
| `phaseprof.jl` | wall-clock-by-phase profile with per-phase parallelism |
| `compare.jl` | compare two dumps, against a measured rounding-noise floor |
| `compare_revs.sh` | benchmark two git revisions on identical workloads, plus the noise floor |
| `scaling.sh`, `summarize.jl` | thread-scaling sweep and its speedup / efficiency table |

The regression suite in `benchmark/benchmarks.jl` (run on every PR by
AirspeedVelocity) carries a small slice of the same workload, `LQR-ragged`.

## Workloads

| name | model |
|---|---|
| `plqr` | Poisson LQR, `Qc` grouped by reward, terminal factor pinned to its own cost (`terminal_regime = 2`), `condition_terminal = true` |
| `plqr_joint` | the same with `condition_terminal = false` — the difference is the price of terminal conditioning |
| `slqr` | two-state switching model (LQR + `:free`), Poisson, `Qc` grouped by reward, `C`/`d` tied |

| tier | plant `n` (latent `2n`) | trials | median length (range) | channels |
|---|---|---|---|---|
| `tiny` | 2 (4) | 24 | 20 (8–45) | 12 |
| `small` | 4 (8) | 96 | 50 (20–110) | 40 |
| `medium` | 8 (16) | 300 | 80 (30–180) | 100 |
| `smoulder` | 12 (24) | 1000 | 100 (40–250) | 150 |

Lengths are log-normal around the median (`--spread`, default 0.35; `--spread=0`
gives equal lengths as a control). Everything is drawn from a `StableRNG`, so two
revisions or two thread counts see bit-identical inputs. Each timed fit starts
from a `deepcopy` of the same fit-ready model and runs exactly `--iters` EM
iterations (convergence is disabled). A warm-up fit on the `tiny` tier compiles
every method first.

Note that with `max_iter = k` an LQR fit runs `k − 1` M-steps: the last
iteration's E-step scores the final parameters and returns before an M-step.

## Where the time goes, and how to see it

### End to end

```console
$ julia --project=benchmark/lqr -t 16 benchmark/lqr/run.jl \
      --workload=plqr,plqr_joint,slqr --tier=medium --iters=3 --reps=2 --csv=timings.csv
```

`--blas-threads` defaults to 1. The E-step and the probe are parallel over trials
and buckets with Julia tasks; a multi-threaded BLAS inside each of those tasks
only oversubscribes the cores.

### Phase profile

`--profile` runs one more fit under the sampling profiler and prints wall time by
*phase* — `M-step state > conditional M-step > L-BFGS > probe > smooth`, and so on
— with the mean number of busy threads in each. A flat profile answers "which
function burns CPU"; on a many-core node the question that matters is "which
phase is wall time spent in, and how many cores are working while it runs": a
phase at 30% of wall time on one busy core is *the* bottleneck on 64 cores even
if it is 3% of CPU samples.

How it works: the sampler stops every thread at each tick. The root task's stack
names the phase; every other thread's stack says whether that core was busy.
When the root task is parked in `wait` on a `tforeach` it spawned it is on no
thread, and the tick is charged to the common context of its previous and next
visible stacks, extended by the phase the busy workers are in.

**Read it at the `medium` tier or larger.** Sampling ticks are ~4 ms apart (the
cost of suspending every thread), so phases shorter than that — which is most of
them at the `small` tier — are attributed with a bias of up to 2× between
neighbouring phases. Totals per top-level phase and the busy-thread column stay
reliable; for exact per-call costs use the microbenchmarks in the "Results"
section's scripts instead. (`Profile.@profile_walltime`, which would sample the
parked root task directly, segfaults on multi-threaded runs in Julia 1.13.)

### Thread scaling

```console
$ benchmark/lqr/scaling.sh "1 4 8 16 32" --workload=plqr,slqr --tier=smoulder --iters=3
```

prints the per-iteration time, speedup and parallel efficiency per workload.

## Checking that a change kept the answers

```console
$ JULIA_NUM_THREADS=8 benchmark/lqr/compare_revs.sh main -- --workload=plqr,slqr --tier=small --iters=3
```

checks `main` out as a worktree (`.bench-worktrees/`, gitignored), runs the same
workloads on it and on the current checkout, then refits `main` `NOISE` times
(default 3) from initial loadings scaled by `1 ± k·10⁻¹⁴`. `compare.jl` then
judges every fitted parameter of the new revision against the spread of those
refits.

Why not a plain tolerance: **the terminal-conditioned fit amplifies rounding.**
Its structural M-step is up to 100 L-BFGS iterations with an accept-and-halve
step, on an objective that is nearly flat along `x0`, `hf` and the costate gauge.
A 10⁻¹⁵ change in the probe weights moves `x0`/`hf` by ~10% within two EM
iterations of the *unchanged* code, while `A`, `S`, `Qc` move by 10⁻⁵…10⁻³. Any
change that reorders floating-point work — a parallel reduction, a batched BLAS
call — perturbs the fit that much and no less, so "equal to 1e-8" can only be
asked of bitwise-preserving changes. What can be asked of every change is that
it is indistinguishable from the unchanged code's own sensitivity to rounding,
and that is what the noise ensemble measures. A final ELBO above the ensemble's
range is a better optimum, not an error.

Component-level equivalence is tested directly, and much more tightly, in
`test/`: the ragged smoother is bitwise identical for every pool size, the
probe's statistics agree with the generic aggregators to 1e-12, and the shared
terminal normalizer agrees with the per-trial one to 1e-12.

## Results

See [`RESULTS.md`](RESULTS.md) for the measurements behind the changes this
harness was built to find, and what is left.
