# Why the Poisson SLDS fit is slow (and single-threaded)

Analysis of `fit!(::SLDS, y)` for a Poisson emission, versus `fit!(::LinearDynamicalSystem, y)`
for the Gaussian (LDS) and Poisson (PLDS) single-regime cases.

Branch: `claude/poisson-slds-parallel-ab6v7p`

## Summary

Two separate things are going on, and only one of them is threading:

1. **`fit_SLDS.jl` never builds a workspace pool.** `fit!` allocates exactly one
   `SLDSSmoothWorkspace` and threads it through every per-trial loop, which forces those
   loops to be sequential. `fit_LDS.jl` / `fit_PLDS.jl` build
   `sws_pool = [SmoothWorkspace(...) for _ in 1:Threads.maxthreadid()]` and chunk trials
   across it with `tforeach`. The parallelism gap is a *workspace-ownership* gap, not a
   missing `@threads`.

2. **The SLDS Poisson path does not reuse PLDS's Poisson kernels.** It reaches the generic,
   per-timestep `observation_loglikelihood!` instead of PLDS's `joint_loglikelihood!` with a
   precomputed `lognorm_t`. The `-Σ log(y!)` normalizer — a constant in the latents — is
   recomputed via `loggamma` on *every line-search evaluation, per regime, per Newton step,
   per trial, per E-step alternation, per EM iteration*. In a profile of a representative fit
   this single term is ~39% of total runtime.

Item 2 is the larger and cheaper win, and it helps at any thread count including `-t 1`.

## Measurements

`K=3, D=3, N=40, T=200, ntrials=16`, 10 EM iterations, `BLAS.set_num_threads(1)` so that
Julia-level task parallelism is what is being measured.

| model  | 1 thread | 2 threads | 4 threads |
|--------|----------|-----------|-----------|
| LDS    | 0.011 s  | 0.011 s   | 0.012 s   |
| G-SLDS | 0.483 s  | 0.487 s   | 0.481 s   |
| PLDS   | 0.377 s  | 0.329 s   | 0.401 s   |
| P-SLDS | 1.904 s  | 1.992 s   | 2.060 s   |

The Poisson SLDS gets *slower* as threads are added — there is no parallel work to
distribute, only scheduler overhead. (This problem size is small enough that PLDS's own
parallel gain is within noise; see the larger case below.)

A sampling profile of `fit!(::SLDS)` with a Poisson emission, 20 iterations, 4 threads:

```
Total snapshots: 32648.  Utilization: 50% across all threads and tasks.
  16325  Base/task.jl:1216  poptask            <- idle worker threads
   4055  (main task, all real work)
   4050    fit!                     (fit_SLDS.jl:2023)
   2899      estep! / _vem_alternate!
   2778        _slds_smooth_all!    (fit_SLDS.jl:1048)   68% of total
   2625          newton_smooth!
   2022            ϕ!  ->  joint_loglikelihood!  (fit_SLDS.jl:421)
   1659              observation_loglikelihood!  (poisson_observations.jl:405)
   1570                sum(yi -> loggamma(yi+1), yt)     39% of total
```

Only the main task does work; the three worker threads sit in `poptask`.

## 1. There is no SLDS workspace pool

`fit!` (`fit_SLDS.jl:2090`) allocates a single workspace:

```julia
slds_ws = SLDSSmoothWorkspace(T, slds, T_max)
```

Every per-trial routine takes that one object and loops sequentially over trials, because all
trials would otherwise write into the same `btd` / `opt` / `consts` / `ll_tmp` / `poisson`
buffers:

- `_slds_smooth_all!` (`fit_SLDS.jl:1048`) — `for trial in eachindex(y) … smooth!(…; ws=slds_ws)`
- `_slds_fill_logL!` (`fit_SLDS.jl:982`) — `for trial in eachindex(y) … fill_trial!(slds, slds_ws, trial)`
- `elbo!` (`fit_SLDS.jl:1502`) — `for trial in 1:ntrials … _slds_trial_elbo(…, slds_ws, …)`
- `_slds_warmstart!` (`fit_SLDS.jl:2275`)
- the grouped variants (`_slds_smooth_cell!`, `_elbo_grouped!`), which additionally serialize
  cells because each cell workspace shares `base.btd`

The one `tforeach` in the whole file (`fit_SLDS.jl:366`) is in the discrete-layer `fit!`,
accumulating `ξ[t1:t2-1]` into `ξ[t2]` — `O(K²·T)` adds, the cheapest loop in the file.

The other parallel piece, `HMMs.forward_backward!`, *is* threaded over trials
(`HiddenMarkovModels/src/inference/forward_backward.jl:92`) — and it is also `O(K²·T)`.
So the two loops that are threaded are the two cheap ones, and every `O(N·D²·T)` loop is
serial.

Compare `fit_LDS.jl:1020`:

```julia
pool_size = Threads.maxthreadid()
sws_pool = Vector{SmoothWorkspace{T}}(undef, pool_size)
…
```

and `smooth!(lds, tfs, data, sws_pool)` (`fit_LDS.jl:282`), which chunks trials across the
pool with `tforeach`, indexing by chunk position rather than `threadid()`.

## 2. The Poisson emission M-step is handed a one-element pool

`_tied_poisson_emission!` (`fit_SLDS.jl:1708`):

```julia
update_observation_model!(slds.LDSs[k], tfs, data.y, [sws], weights_of(k); uy=data.uy)
#                                                    ^^^^^ one-element pool
```

Inside `update_observation_model!` (`poisson_emission_mstep.jl:506`) the pool length caps the
task count:

```julia
ntasks = max(1, min(ntrials, length(sws_pool)))
```

so `ntasks == 1`, one chunk, and the `tforeach` in `_poisson_mstep_pass!`
(`poisson_emission_mstep.jl:307`) runs a single task. PLDS passes the real pool
(`fit_PLDS.jl:383`). The SLDS then does this `K` times, once per regime, each fully serial.
The grouped path has the same shape at `fit_SLDS.jl:2747`.

## 3. The SLDS Poisson path misses PLDS's optimized kernels

`joint_loglikelihood!(ws::SLDSSmoothWorkspace, slds, x, y, w, ux, uy)` (`fit_SLDS.jl:404`)
calls the **generic** method in `continuous_latents.jl:126`, which loops timesteps calling
`observation_loglikelihood!`. The Poisson method (`poisson_observations.jl:382`) ends with:

```julia
return dot(yt, z) - sum(λ) - sum(yi -> loggamma(yi + one(T)), yt)
```

`Σᵢ log(y[i,t]!)` does not depend on any parameter or on `x`. PLDS already solved this:
`_poisson_lognorm_t(y)` (`fit_PLDS.jl:42`) computes it once per trial, and PLDS's own
`joint_loglikelihood!` (`fit_PLDS.jl:73`) takes it as an argument and also folds the
per-timestep emission into `dot(y, η) - sum(exp, η)`. The SLDS never reaches that method,
because it dispatches on `SmoothWorkspace` and the SLDS carries an `SLDSSmoothWorkspace`.

A prototype of the SLDS `joint_loglikelihood!` using the PLDS-style kernel plus a precomputed
`lognorm_t`, on the same inputs:

```
max |Δ| vs. current            = 1.4e-14
current  joint_loglikelihood!  = 509.7 µs
prototype                      = 197.3 µs   (2.58x)
```

The Newton line search calls `ϕ!` — that kernel — repeatedly per Newton step, which is why it
dominates the profile.

## 4. Structural costs that threading will not remove

Worth calibrating expectations against:

- **The shared-covariance fast path is unavailable to an SLDS.** `smooth!(lds, tfs, data, sws_pool)`
  exploits the Gaussian LDS Hessian being observation-independent, so equal-length trials share
  one covariance pass and one BT factorization, and the mean pass collapses into a batched
  `(D·T)×N` solve (`fit_LDS.jl:295-345`). This is why `LDS` is 0.011 s against `PLDS`'s 0.377 s
  above. In an SLDS the Hessian is weighted by `γₖ(t)`, which differs per trial, so every trial
  pays its own Hessian, its own cov pass, and its own `block_tridiagonal_inverse_logdet!`.
- **Everything is `K` times over.** `gradient!`, `hessian!`, `joint_loglikelihood!` all loop
  `k in 1:K`; the M-step aggregates `K` weighted sufficient statistics and runs `K` LBFGS
  emission solves.
- **The weighted aggregator cannot reuse the constant blocks.** `_aggregate_td_suff_stats_weighted!`
  (`sufficient_statistics.jl:429`) rebuilds the data-side sums every E-step because the weights
  change, and skips the cov-cache fast path — noted in its own docstring at lines 419-423.

So a realistic target for a fully parallel Poisson SLDS is roughly `K × PLDS / nthreads`, not
parity with PLDS.

## 5. Smaller per-iteration overheads

All in `mstep!` (`fit_SLDS.jl:1839`), once per EM iteration:

- `data = Data(slds.LDSs[1], y; ux=ux, uy=uy)` (line 1863) — re-validates every trial's shapes
  each iteration; `fit!` already built the same `Data` at entry.
- `sufs = [_initialize_td_sufficient_statistics(…) for _ in 1:K]` (line 1878) and
  `bufs = GroupedSufBuffers(T, lds1, data.tsteps)` (line 1893) — freshly allocated each iteration.
- `weights_of(k)` (line 1865) builds a new `Vector` of `ntrials` views on each of its `K + K` calls.
- `_update_shared_initial_state!` (line 1955) does `deepcopy(slds.LDSs[1])` — copies `A`, `Q`,
  `C`, `d`, `P0` and all priors — once per iteration, to use as a scratch model.

## Proposed solutions

Ordered by (measured or expected) payoff per unit of risk.

### A. Reuse the PLDS Poisson kernels in the SLDS log-likelihood — *no threading involved*

Add an `SLDSSmoothWorkspace` method (or generalize the PLDS one over
`Union{SmoothWorkspace,SLDSSmoothWorkspace}`, which `continuous_latents.jl:128` already does for
the generic form) so the SLDS Poisson path gets the batched emission term, and thread a
per-trial `lognorm_t` — computed once in `fit!` alongside `tfs` — through `_slds_smooth_all!`
into `smooth!`'s `ϕ!`.

- Measured 2.58× on a kernel that is ~50% of total fit time (`ϕ!` is 2022 of 4055 samples),
  so ≈1.4× end-to-end on its own; agrees with the current result to 1.4e-14.
- No concurrency, no reproducibility question, no memory growth. Lands independently of B-D.
- The same treatment applies to the SLDS `gradient!` (`fit_SLDS.jl:441`), which still calls
  `observation_gradient!` per timestep — forming `η = C·X + d` and `G = C'(Y - Λ)` as two
  `gemm`s per regime is the same trick commit `c5f0a46` applied to the Hessian, and
  `PoissonBatchBuffers` / `_poisson_linear_predictor!` already exist for it.

### B. Give the SLDS a workspace pool and chunk the per-trial loops

Mirror `fit_LDS.jl:1020` exactly:

```julia
npool = min(Threads.maxthreadid(), ntrials)
slds_ws_pool = [SLDSSmoothWorkspace(T, slds, T_max) for _ in 1:npool]
```

then chunk `_slds_smooth_all!`, `_slds_fill_logL!`, and `elbo!` over the pool with `tforeach`,
indexing by chunk position (never `threadid()`).

Correctness notes:

- `_slds_smooth_all!` writes only into `tfs[trial]` — disjoint per trial, so results are
  bit-identical to the serial loop.
- `_slds_fill_logL!` writes disjoint column ranges `dl.logL[:, t1:t2]` — likewise.
- `elbo!` accumulates a scalar, so it needs per-chunk partials reduced in chunk order (the
  `partial = zeros(T, ntasks)` pattern at `fit_PLDS.jl:406`) to stay deterministic.
- **`refresh_slds_constants!` must be applied to every pool entry** after each M-step, not just
  the first (currently one call at `fit_SLDS.jl:2196`).
- **RNG is the one real design decision.** `_slds_smooth_all!` passes `rng` into `smooth!` for
  the joint draw into `x_samples[trial]`. Sharing one RNG across tasks is both a data race and
  non-reproducible. Deriving a per-*trial* RNG (`Xoshiro(hash((seed, trial, iter)))`) keeps the
  fit reproducible *and* independent of thread count — worth doing even though it changes the
  numbers relative to today's stream.
- Memory: `npool × (BlockTridiagonalWorkspace O(D²·T) + NewtonBuffers + K·SmoothConstants +
  PoissonBatchBuffers O(N·T))`. The Poisson batch buffers are the big block; for large `N·T`
  this is the term that decides whether `npool` should be capped below `maxthreadid()`.

### C. Pass a real pool to the Poisson emission M-step

Change `_tied_poisson_emission!` (and `fit_SLDS.jl:2747`) to hand `update_observation_model!` a
genuine `Vector{SmoothWorkspace}` instead of `[sws]`. This immediately re-enables the chunked
`tforeach` inside `_poisson_mstep_pass!` that PLDS already gets — a small, local diff.

Caveat: `update_observation_model!` allocates `curv_bufs` / `ls_bufs` sized `ntasks`, each
`O(N·T_max)`, per call. With `K` regimes that is `K × ntasks` such buffers per M-step; they
should be hoisted into the pool rather than reallocated.

### D. Parallelize the M-step's `K` axis

`for k in 1:K: _aggregate_td_suff_stats_weighted!(sufs[k], …)` (`fit_SLDS.jl:1879`) writes into
disjoint `sufs[k]`, and the `K` LBFGS solves in `_tied_poisson_emission!` write into disjoint
`slds.LDSs[k]`. Both are embarrassingly parallel across `k`, needing only `K` `SmoothWorkspace`s.
This is a second, independent parallel axis, useful when `ntrials < nthreads`. Do not try to
parallelize the `k` loops inside `gradient!` / `hessian!` / `joint_loglikelihood!` — those
accumulate into shared `grad` / `H_diag` / `ll_vec`, so they would need per-`k` accumulators plus
a reduction, for a `K` that is typically 2-5.

### E. Housekeeping in `mstep!`

Hoist `Data`, `sufs`, `bufs`, and the `weights_of` view vectors into `fit!` and reuse them
across iterations; replace the `deepcopy` in `_update_shared_initial_state!` with a scratch LDS
allocated once. Small next to A-D, but free.

## Open questions

These change which of A-E is worth doing first, and how B should be built.

1. **What is the real workload shape** — `ntrials`, `T`, `N` (channels), `K`, `D`, and how
   many EM iterations? Trial-parallelism (B) only pays when `ntrials ≥ nthreads`. If the fits
   are few-trial / many-neuron, A and the batched `gradient!` matter far more than B, and D
   (the `K` axis) becomes the only useful parallel axis.
2. **Is the `depends_on` / stitching path in use?** The grouped code path serializes cells as
   well as trials, and cell workspaces deliberately share `base.btd`
   (`_cell_slds_workspace`, `fit_SLDS.jl:2370`). Parallelizing it means giving each pool slot
   its own `btd`, which changes that sharing scheme.
3. **Does reproducibility across thread counts need to hold?** B changes how the E-step's
   posterior draws consume the RNG. A per-trial derived RNG makes results independent of
   thread count but will not reproduce today's `fit!(…; rng=MersenneTwister(1))` traces. Is
   the existing-trace stability a constraint, or is thread-count-independence preferable?
4. **Memory ceiling per fit?** A pool of `nthreads` SLDS workspaces multiplies the
   `O(D²·T) + O(N·T)` scratch by the thread count. On a 64-core node with large `N·T` this can
   be the binding constraint, and the pool should be capped.
5. **Which entry points matter** — just `fit!`, or also `smooth` / `elbo` / `loglikelihood`
   for post-fit inference? They share `_slds_smooth_all!` and `_slds_fill_logL!`, so B covers
   all of them, but `smooth` allocates its own workspace at `fit_SLDS.jl:875` and would need
   the same treatment.
