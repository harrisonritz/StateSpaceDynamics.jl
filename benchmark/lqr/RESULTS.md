# Inverse-LQR EM: what was slow, what changed, what is left

Measured with the harness in this directory on a 4-core container (Julia 1.13,
`BLAS.set_num_threads(1)`, Julia threads as stated). The workload of interest is
`plqr`: Poisson, `Qc` grouped by reward, ragged trial lengths, and the exact
terminal-conditioned objective. The numbers here are for the `small` and
`medium` tiers; `smoulder` (n = 12, 1000 trials, 150 channels) should be run on
the target node with `scaling.sh`, which is where the multi-core behaviour below
matters most.

## 1. Diagnosis (baseline, `85bedb5`)

`plqr`, `small` tier, 4 threads — phase profile of two EM iterations:

| phase | wall | busy threads |
|---|---|---|
| M-step state › conditional M-step › L-BFGS › exact `log Z` | 38% | 1.0 |
| … › probe › smooth (ragged Gaussian smoother) | 41% | ~1.0 |
| … › probe › aggregate | 15% | 1.0 |
| E-step (Poisson Laplace smoother) | 2% | 3.4 |

Nearly the whole fit was the **terminal-conditioning M-step, on one core**. The
same fit with `condition_terminal = false` (`plqr_joint`) was ~25× faster. Three
things compounded:

1. **Every L-BFGS gradient evaluation re-smooths a "probe"** — a zero-loading
   copy of the model whose smoother gives the terminal-conditioned prior moments
   (Fisher's identity for `∂ log Z/∂θ`). With `mstep_iters = 100` that is tens
   to hundreds of probe smooths per EM iteration, per reward cell.
2. **The probe is ragged, and the ragged Gaussian smoother was serial.** Trials
   are bucketed by length and a bucket's covariance is shared, but buckets ran
   one after another; with nearly all-distinct lengths almost every bucket is a
   singleton, which also took the general single-trial path — a banded `pbsv`
   solve for the mean *and* a second block factorization for the covariance.
3. **Exact `log Z` recomputed the backward square-root recursion per distinct
   horizon**, with an allocating QR per step and a per-design allocating mean
   recursion — ~9000 allocations per call at smoulder size, on every objective
   evaluation.

The **SLQR** workload is different: its cost is the E-step (Laplace smoother and
forward–backward), already parallel. Note that terminal conditioning is **not
active** in a grouped (`depends_on`) switching fit — its M-step takes the joint
path, and scoring a grouped switching fit with `condition_terminal = true` throws
— so the SLQR probe never runs there. That is a modelling gap, not a
performance one, and is left as is.

## 2. Changes

Every change is exact algebra. "Bitwise" means the result is identical to the
baseline; "reorder" means only the order of floating-point additions changed
(observed differences ≤ 6·10⁻¹⁵ relative in the affected quantities). All
parallel reductions use the repository's fixed-chunk scheme
(`numerics/reduction.jl`), so every result is **independent of the thread
count**.

| # | change | where | exactness |
|---|---|---|---|
| 1 | **Ragged smoother shares the forward factorization across lengths.** A length-`h` trial's precision equals the longest trial's on blocks `1…h−1` (the schedule is indexed from `t = 1`); only the last block differs. The forward block-Cholesky sweep (`Mᵢ` factors, `Dᵢ₊₁ = Mᵢ⁻¹Cᵢ`, `Mᵢ⁻¹`, log-det prefix) runs once over `T_max`; each length adds one last-block factor and the backward covariance recursion — ~4 D³ per block instead of ~10 D³. | `fit_LDS.jl`: `_prefix_forward!`, `_bucket_from_prefix!`, `_smooth_ragged_prefix!` | bitwise (same operations in the same order; tested for pool sizes 1–7) |
| 2 | **Length buckets run in parallel** (greedy, longest first, one workspace per task); with few buckets and many trials each, the per-trial mean solves are split instead. Singleton buckets reuse the covariance pass's factors for the mean (no second factorization). Bucket covariance storage is reused across calls instead of reallocated. | `fit_LDS.jl` | bitwise |
| 3 | **Probe statistics aggregated directly.** Weights are per-design constants and covariances are shared per length bucket, so each shared covariance is summed once, scaled by its total count, and the means go through BLAS-3 per regime run; only the moments the conditional M-step reads are formed. Parallel over fixed chunks above a work threshold. | `lqr_terminal.jl`: `_lqr_probe_aggregate!` | reorder |
| 4 | **Exact `log Z` shares its backward steps across horizons.** With one running regime and a pinned terminal regime (the ragged shape), step `j` is the same for every horizon, so the recursion is computed once; the per-design mean recursion runs for all designs at once (active designs are a leading block of columns). In-place QR on preallocated buffers. Falls back to per-horizon steps when the schedule varies. | `lqr_terminal.jl`: `_lqr_terminal_logz_sum`, `_whitening_factor!` | reorder (≤ 2·10⁻¹⁶ vs the per-design reference) |
| 5 | **Reward cells in parallel** for `log Z` and for the probe (each cell's probe owns its workspaces). | `lqr_terminal.jl` | bitwise |
| 6 | **Probe cached across M-steps** (weakly keyed on the fit's statistics; rebuilt if the designs or structure change) instead of rebuilding a model copy and a `maxthreadid`-sized workspace pool every M-step. | `lqr_terminal.jl` | bitwise |
| 7 | **Poisson E-step load-balanced**: trials handed out one at a time, longest first, instead of fixed contiguous chunks — ragged lengths and varying Newton counts no longer leave threads idle behind the slowest chunk. | `fit_PLDS.jl` | bitwise |
| 8 | **E-step aggregations parallel** (base and per-regime LQR statistics), fixed chunks. | `sufficient_statistics.jl`, `lqr_mstep.jl` | reorder |

## 3. Results

### End to end (`compare_revs.sh 85bedb5`, 4 threads, `--iters=3`, i.e. two M-steps)

| workload, tier | baseline s/iter | now s/iter | speedup |
|---|---|---|---|
| `plqr`, small (latent 8, 96 trials, 49 distinct lengths) | 6.86 | 1.10 | **6.3×** |
| `plqr_joint`, small | 0.176 | 0.144 | 1.2× (near timing noise at this size) |

With one M-step (`--iters=2`) the `plqr` ratio is 4.1× (2.7 → 0.66 s/iter); the
second M-step benefits more because the probe is cached from the first.

At the `medium` tier (latent 16, 300 trials, 104 distinct lengths, 100 channels)
the current code spends, over two EM iterations on 4 threads:

| phase | wall | busy threads (of 4) |
|---|---|---|
| M-step › conditional M-step › probe (smooth + aggregate, all cells) | 59% | 3.8 |
| E-step › Poisson Newton smoother | 14% | 4.0 |
| M-step › L-BFGS bookkeeping, `log Z`, `_lqr_fg!` | 8% | 1.7 |
| M-step emission (Poisson row-wise Newton) | 5% | 2.5 |
| probe setup (first M-step only; cached afterwards) | 4% | 1.0 |
| E-step aggregation, other | 3% | 1.0–2 |

— i.e. ~90% of the wall time now runs on all cores, against ~5% before.

### Per-component (microbenchmarks; `small` unless stated)

| component | before | after |
|---|---|---|
| probe smooth, per call (one reward cell) | 8.2 ms | 2.7 ms |
| probe statistics, per call | 5.3 ms (generic weighted aggregators) | 0.49 ms |
| exact `log Z`, per call, smoulder size (n = 12, 333 designs, 126 horizons) | 7.5–9.9 ms, 9060 allocs | 4.4–5.7 ms, 1573 allocs |
| E-step aggregation, medium, 4 threads (base + LQR) | 31 + 25 ms | 20 + 16 ms |

### Accuracy

Against a 2-member rounding-noise ensemble of the baseline (`compare.jl`), every
fitted parameter of `plqr` and `plqr_joint` is within the ensemble's spread, and
the `plqr` final ELBO (−178201.94) is *above* every baseline refit
(−178202.28 … −178202.16). Component-level: smoother bitwise identical (all pool
sizes); probe statistics and E-step aggregates within 6·10⁻¹⁵ relative; `log Z`
within 2·10⁻¹⁶ of the per-design reference. The `Linear Dynamical Systems` test
set (3485 tests, plus the four added here) passes.

### Still to measure on the target node

`smoulder` tier, and thread counts above 4:

```console
$ benchmark/lqr/scaling.sh "1 8 16 32 64" --workload=plqr,slqr --tier=smoulder --iters=3
$ JULIA_NUM_THREADS=32 benchmark/lqr/compare_revs.sh 85bedb5 -- --workload=plqr --tier=medium --iters=3
```

## 4. What is left, and what would need a looser contract

**Exact, not done.**

* *The L-BFGS bookkeeping between probe evaluations* (`_lqr_fg!`, context
  construction, the sequential part of `log Z`) is serial and becomes the
  Amdahl fraction on many cores. The backward step recursion of `log Z` is
  inherently sequential in `t` (O(T_max n³) per evaluation).
* *Grouped E-step cells run one after another*, each with a parallel smooth and
  a barrier; overlapping cell `c`'s aggregation with cell `c+1`'s smoothing, or
  smoothing all cells in one pool, would remove the per-cell barriers.
* *Probe covariance cost is O(Σₕ h·D³) per evaluation* — one backward recursion
  per distinct horizon. Exactly, only the forward half is shareable (done). The
  sums the M-step needs could in principle be accumulated without forming every
  `Σₜ`, but not with less than per-horizon work.

**Would change the results (outside the requested contract).**

* `mstep_iters` (L-BFGS iterations per M-step, default 100) multiplies the probe
  cost directly. It is the largest remaining lever; a fit that converges in EM
  iterations rarely needs a fully converged inner M-step (generalized EM only
  needs improvement), but this changes the trajectory.
* Binning trial lengths (e.g. to multiples of 5 bins) would cut the number of
  distinct horizons, and so the probe's covariance work, ~5×.
* Truncating the probe's covariance recursion where it has converged to its
  stationary value would be exact to rounding in practice but is not exact
  algebra.

**A statistical observation made along the way.** The terminal-conditioned fit
amplifies rounding: a 10⁻¹⁴ perturbation of the initial loadings moves `x0` and
`hf` by O(10%) within two EM iterations, while `A`, `S`, `Qc` move by
10⁻⁵…10⁻³. Those directions are nearly flat in the objective — `x0` and `hf` are
weakly identified in this design. That is why `compare.jl` judges changes against
a measured noise ensemble rather than a fixed tolerance, and it may be worth a
prior on (or freezing of) `x0`/`hf` in real fits.
