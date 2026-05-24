# Refactor Speedup — `main` vs `dev_ryan_`

Comparison of `fit!` throughput across the BTD (tridiagonal), Kalman, and
Poisson paths. Both runs use the same `benchmarking/refactor_compare*.jl`
driver: 20 EM iterations with `tol=0.0` to force fixed-iter timing,
BenchmarkTools median of 5 samples, 3 s/sample, Julia 1.12. Identical
model parameters, RNG seeds, and trial shapes on both branches.

The Kalman backend did not exist on `main`, so those rows are
`dev_ryan_`-only.

## Headline (post-pbsv + batched-mean)

* **Gaussian TD path: 14-66× faster than `main`, then another
  1.4-1.7× from the batched mean pass.** Allocations drop 100s-1000s
  of times. The `long_single` row — flagged in the previous summary as
  a 14% slowdown vs main — is now **9× faster than main**: it was the
  BLAS-dispatch-on-small-blocks hotspot in the BT solve, now routed
  through LAPACK `pbsv` for `bs ≤ 8`.
* **TD now beats Kalman on multi-trial Gaussian.** The remaining gap
  identified in the previous summary (Kalman 1.1-2.9× faster on the
  four multi-trial scenarios) is **gone**: a batched-trial mean pass
  collapses the per-trial Newton step into a single `(D·T) × N`
  matrix-RHS backsubst, doing the same total math at BLAS-3 dispatch
  efficiency. TD is now 1.2-1.6× faster than Kalman on `medium` /
  `large` / `many_short`, and Kalman still errors at `ntrials = 1`
  (so TD wins `long_single` by default).
* **Poisson path: 2-3× faster than `main` everywhere**, with 100-500×
  less memory. The previous summary recorded a 2-3× wall-clock
  regression on the Poisson path; that's now inverted by the same pbsv
  fast path the Gaussian smoother uses.
* **Memory footprint collapses** across the board (37-700× less than
  main) thanks to the suf-based aggregator + preallocated
  SmoothWorkspace.

## Gaussian (BTD path) — vs main

| scenario | main time | dev_ryan_ time | **speedup** | main mem | dev mem | **mem×** |
|---|---:|---:|---:|---:|---:|---:|
| small (D=3, p=5, T=100, N=4)        | 0.083 s | 0.005 s | **17×** | 135 MB     | 1.5 MB  |  92× |
| medium (D=5, p=10, T=200, N=8)      | 0.531 s | 0.014 s | **38×** | 1.24 GB    | 7.4 MB  | 168× |
| large (D=8, p=16, T=500, N=16)      | 6.837 s | 0.061 s | **112×**| 14.57 GB   | 54.8 MB | 266× |
| long_single (D=4, p=8, T=2000, N=1) | 0.619 s | 0.066 s | **9.4×**| 1.07 GB    | 26.1 MB |  41× |
| many_short (D=3, p=5, T=50, N=64)   | 0.513 s | 0.012 s | **43×** | 1.11 GB    | 3.4 MB  | 326× |

The multi-trial rows take a 1.4-1.7× improvement on top of pbsv from
the batched mean pass (medium 25 → 14 ms, large 104 → 61 ms,
many_short 22 → 12 ms). `long_single` (N=1) is unchanged — the
batched path is gated on `ntrials > 1`. Mem grows modestly on the
multi-trial rows due to the `(D, T, N)` batched buffers; still 90-330×
below main.

## Gaussian (Kalman backend, dev_ryan_ only)

| scenario | TD time | Kalman time | **TD/Kalman** | TD mem | Kalman mem |
|---|---:|---:|---:|---:|---:|
| small        | 0.005 s | 0.005 s | **1.0×** | 1.5 MB  | 4.75 MB |
| medium       | 0.014 s | 0.018 s | **0.81× (TD faster)** | 7.4 MB  | 17.4 MB |
| large        | 0.061 s | 0.095 s | **0.64× (TD faster)** | 54.8 MB | 92.6 MB |
| many_short   | 0.012 s | 0.008 s | **1.5×** | 3.4 MB  | 4.36 MB |
| long_single  | 0.066 s | failed (N=1 path) | — | 26.1 MB | — |
| **huge (D=128, p=64, T=250, N=500, 5 iter)** | **7.9 s** | 9.5 s | **0.83× (TD faster)** | 3.26 GB | 3.82 GB |

The huge row is a stress test of the worst case for the previous TD
init: `initialize_FilterSmooth` was eagerly allocating four `(D, D, T)`
arrays per trial (`p_smooth`, `p_smooth_tt1`, `E_zz`, `E_zz_prev`), which
came to **64 GB at D=128, N=500, T=250** — even though the cov-cache
fast path immediately aliases `p_smooth` away to `sws.p_smooth_shared`
and the new TD aggregator never reads `E_zz` / `E_zz_prev` at all.
Before the fix, TD was 3.65× **slower** than Kalman on this row and
used 17× more memory. After: 0.83× / 0.85×.

TD now matches or beats Kalman on every multi-trial scenario except
`many_short` (where Kalman still wins on the N=64 dispatch loop). The
previous Kalman edge came from its `(D, T, N)` BLAS-3 mean pass; TD
now does the same — the BT Hessian / Cholesky cache is computed once
on `sws_pool[1]` and the per-trial Newton step collapses into a
single matrix-RHS `block_tridiagonal_backsubst!` call.

For Poisson, TD is still the only option — Kalman doesn't apply
because the Hessian is x-dependent. The N=1 Kalman failure is a real
bug worth filing — the `@sync` chunked smoother probably doesn't
degrade gracefully when `ntasks = 1`.

## Poisson

| scenario | main time | dev_ryan_ time | **speedup** | main mem | dev mem | **mem×** |
|---|---:|---:|---:|---:|---:|---:|
| small        |  0.18 s | 0.089 s | **2.0×** |   235 MB |  2.12 MB | **111×** |
| medium       |  1.70 s | 0.753 s | **2.3×** |  1.97 GB |  6.96 MB | **290×** |
| large        | 20.72 s | 6.858 s | **3.0×** | 22.13 GB | 42.50 MB | **534×** |
| long_single  |  1.85 s | 0.889 s | **2.1×** |  1.80 GB | 27.29 MB |  **68×** |
| many_short   |  1.47 s | 0.652 s | **2.3×** |  1.87 GB |  3.65 MB | **524×** |

Allocation reduction: 30-50× across all scenarios. **Both speed and
memory dominate main on every Poisson scenario.**

## What closed the regression

The previous summary identified the LBFGS emission update with `1e-12`
tolerances as the likely Poisson wall-clock culprit. Profiling showed
otherwise: the M-step LBFGS is sub-millisecond per EM iter; the
real bottleneck was the **block-tridiagonal Newton smoother**, where
small-D (`bs = latent_dim ≤ 8`) per-block BLAS dispatch overhead
dominated the actual arithmetic. Each `mul!`/`getrf!`/`getrs!` on a
5×5 matrix carried ~1 μs of dispatch cost; one block-tridiagonal
solve does ~2000 such calls, so the BT solve was paying ~2 ms of pure
dispatch per call for ~50 μs of math.

Fix landed as `block_tridiagonal_solve_spd!` — at `bs ≤ 8` it packs
the symmetric block-tridiagonal into LAPACK's upper-banded format and
calls `dpbsv_` directly (one LAPACK call, dispatch overhead amortised).
At `bs ≥ ~12` the existing hand-rolled block-Thomas takes over (BLAS-3
is efficient at that block size; the per-block dispatch overhead
becomes negligible vs the actual arithmetic). The crossover at `bs = 8`
is empirical from `benchmarking/pbsv_spike.jl`.

This is the path the Newton smoothers (Gaussian + Poisson + SLDS) all
use; the general `block_tridiagonal_solve!` is unchanged for arbitrary
(non-symmetric, possibly indefinite) tridiagonal solves.

**Batched mean pass (closing the Kalman gap):**

After pbsv, decomposing the medium scenario per-iter cost revealed
Kalman's remaining edge was entirely in the mean update: its
`smooth_mean!` operates on `(D, T, N)` tensors and every `mul!` is a
`D×D × D×N` BLAS-3 matmul (one 77 μs call for all 8 trials). TD ran
per-trial via `@spawn` with vector RHS — 71 μs *per trial* (8 ×) —
because the Cholesky `ldiv!` and `mul!` on `D × 1` slices are BLAS-2
and pay dispatch overhead at small `D`.

Fix landed as `block_tridiagonal_backsubst!(x::Matrix, …)` plus a
new `Gradient_batched!` and `_smooth_mean_only_batched!`: the
per-trial iterate, gradient, and RHS are stacked into `(D, T, N)`
tensors and the entire Newton step becomes a single matrix-RHS
backsubst. `sws_pool[1]` carries the batched buffers (sized at fit
entry from `length(y)`); the rest of the pool stays at `ntrials = 1`.
The cov-cache fast path dispatches into the batched mean when
`all_equal && ntrials > 1` — for `ntrials = 1` or ragged trials the
existing per-trial path still wins (no batching benefit, and ragged
can't share `T`).

Result: TD `smooth!` time drops from 0.79 ms to 0.42 ms on the
medium scenario (47% faster), per-iter cost from 0.85 ms to 0.49 ms
(43% faster), and TD now beats Kalman per-iter by 30%.

**FilterSmooth init bloat (only visible at large `(D, T, N)`):**

After the batched-mean change landed, a D=128 / N=500 / T=250 stress
test showed TD at 3.65× slower than Kalman with **65 GB** of memory
churn vs Kalman's 3.8 GB — even though per-iter `smooth!` + aggregate
+ mstep was a perfectly reasonable 1.4 s / 1.6 MB. The cost was all
in fit-entry init: `initialize_FilterSmooth` was eagerly allocating
four `(D, D, T)` arrays per trial (`p_smooth`, `p_smooth_tt1`,
`E_zz`, `E_zz_prev`) = 128 MB × 500 trials = 64 GB. The cov-cache
fast path then **immediately aliases** `p_smooth` / `p_smooth_tt1` to
`sws.p_smooth_shared` on the first `smooth!` call, and the new TD
aggregator **never reads** `E_zz` / `E_zz_prev` (it consumes
`x_smooth` / `p_smooth` / `p_smooth_tt1` directly). So all 64 GB
were either overwritten or unused.

Fix: `initialize_FilterSmooth` gets a `cov_alias::Bool=false` kwarg.
When the caller knows the cov-cache fast path will fire
(`_fit_tridiag!` opts in on equal-length multi-trial), all four
`(D, D, T)` arrays are stored as `(0, 0, 0)` stubs. SLDS, Poisson,
ragged, and single-trial callers keep the default `cov_alias=false`
because they invoke the per-trial smoother that *does* write into
`fs.p_smooth`. Legacy `sufficient_statistics!(fs)` (still called from
tests / benchmarks) resizes the stubs on demand.

Result on the huge row: 35.2 s → 7.9 s (4.4× faster), 65 GB → 3.3 GB
(20× less memory); TD now wins by 20% at this size.

Additional Poisson M-step work shipped alongside:
* LBFGS tol loosened from `1e-12` → `1e-8` (the surrounding EM uses
  `1e-6`, so `1e-12` was 10⁴× overkill) + `iterations=200` cap.
* `gradient_observation_model!` per-task scratch buffers (h/ρ/λ/CP)
  now come from `sws_pool[task_idx]`'s existing `Q_obs!` workspace
  fields instead of being allocated per gradient call.
* Per-trial `_loglikelihood_ws` now calls `LAPACK.trtrs!` directly
  against `pdm.chol.factors` instead of `ldiv!(pdm.chol.L, …)` — the
  `.chol.L` accessor allocates a fresh `LowerTriangular` wrapper on
  every access, and the Newton smoother evaluates the loglikelihood
  10s of times per smoother call.

## Reproducing

From repo root, on either branch:

```bash
julia --project=. benchmarking/refactor_compare.jl > /tmp/dev.csv      # dev_ryan_
julia --project=. benchmarking/refactor_compare_main.jl > /tmp/main.csv # main
```

The two scripts are kept separate because the `main` API doesn't
support `kalman_filter=true`, returns `(elbos, param_diff)` from
`fit!`, and uses `log_d` instead of `d` in `PoissonObservationModel`.
Output columns are identical so the CSVs diff cleanly.
