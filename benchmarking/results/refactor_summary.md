# Refactor Speedup — `main` vs `dev_ryan_`

Comparison of `fit!` throughput across the BTD (tridiagonal), Kalman, and
Poisson paths before and after the dev_ryan_ refactor. Both runs use the
same `benchmarking/refactor_compare*.jl` driver: 20 EM iterations with
`tol=0.0` to force fixed-iter timing, BenchmarkTools median of 5 samples,
2 s/sample, Julia 1.12. Identical model parameters, RNG seeds, and
trial shapes on both branches.

The Kalman backend did not exist on `main`, so those rows are
`dev_ryan_`-only.

## Headline

* **Gaussian TD path: 11–65× faster** on multi-trial workloads; allocations
  drop **76–748×**. The bigger the model, the bigger the win — `large`
  (D=8, p=16, T=500, N=16) goes from 6.8 s to 0.10 s.
* **Memory footprint collapses** across the board (37–700× less) thanks to
  the suf-based aggregator + preallocated SmoothWorkspace.
* **New Kalman backend is competitive with TD** on its own terms,
  marginally faster on the small/medium scenarios and slower in
  allocations (the cov-storage Vector{PDMat} pays a fixed cost).
* **Poisson path: state-side wins, observation-side regression.** Memory
  drops 30–100× (suf-based state aggregator working as intended), but
  wall-clock is currently *slower* than `main` — the LBFGS emission
  update with `1e-12` tolerances dominates and was tuned for the legacy
  per-trial loop. See "Poisson follow-up" below.

## Gaussian (BTD path)

| scenario | main time | dev_ryan_ time | **speedup** | main mem | dev mem | **mem×** |
|---|---:|---:|---:|---:|---:|---:|
| small (D=3, p=5, T=100, N=4)        | 0.083 s | 0.007 s | **11.4×** | 135 MB     | 1.05 MB | 129× |
| medium (D=5, p=10, T=200, N=8)      | 0.531 s | 0.024 s | **22.0×** | 1.24 GB    | 5.19 MB | 239× |
| large (D=8, p=16, T=500, N=16)      | 6.837 s | 0.105 s | **65.1×** | 14.57 GB   | 38.5 MB | 378× |
| long_single (D=4, p=8, T=2000, N=1) | 0.619 s | 0.720 s | 0.86× (slower) | 1.07 GB | 28.9 MB | 37× |
| many_short (D=3, p=5, T=50, N=64)   | 0.513 s | 0.024 s | **21.6×** | 1.11 GB    | 1.64 MB | 695× |

| scenario | main allocs | dev_ryan_ allocs | **allocs×** |
|---|---:|---:|---:|
| small        |    440 652 |   5 787 |  76× |
| medium       |  1 582 730 |   9 475 | 167× |
| large        |  7 385 294 |  20 581 | 359× |
| long_single  |  1 822 272 | 153 768 |  12× |
| many_short   |  3 818 823 |   5 105 | 748× |

Notes:
* The `long_single` scenario is a single very long trial (N=1, T=2000).
  Here the multi-trial parallelization and cov-cache fast path don't help,
  and `dev_ryan_` is currently ~14% slower in wall time despite the 37×
  memory reduction. The cov-cache fast path requires N≥2 equal-length
  trials; with N=1 we fall through to the single-trial smoother.
  Investigating this is a follow-up — likely a Newton-inner-iter
  count change.

## Gaussian (Kalman backend, dev_ryan_ only)

| scenario | TD time | Kalman time | TD mem | Kalman mem |
|---|---:|---:|---:|---:|
| small        | 0.007 s | 0.005 s | 1.05 MB | 4.75 MB |
| medium       | 0.024 s | 0.019 s | 5.19 MB | 17.4 MB |
| large        | 0.105 s | 0.097 s | 38.5 MB | 92.6 MB |
| many_short   | 0.024 s | 0.008 s | 1.64 MB | 4.36 MB |
| long_single  | 0.720 s | failed (N=1 path) | 28.9 MB | — |

Kalman edges out TD on the `many_short` (D=3, N=64) case because the
covariance forward-backward pass is computed exactly once across all 64
trials. TD is more allocation-efficient. The N=1 failure is a real bug
worth filing — the @sync chunked smoother probably doesn't degrade
gracefully when ntasks=1.

## Poisson

| scenario | main time | dev_ryan_ time | main mem | dev mem | mem× |
|---|---:|---:|---:|---:|---:|
| small        |  0.18 s |  0.46 s |   235 MB |  6.9 MB | 34× |
| medium       |  1.70 s |  4.68 s |  1.97 GB | 35.4 MB | 55× |
| large        | 20.72 s | 38.47 s | 22.13 GB |  215 MB | 103× |
| long_single  |  1.85 s |  6.43 s |  1.80 GB | 61.0 MB | 30× |
| many_short   |  1.47 s |  3.32 s |  1.87 GB | 47.0 MB | 40× |

Allocation reduction: 7–10× across all scenarios.

### Poisson follow-up

The Poisson wall-clock regression is **not** the suf-based state
aggregator — that's strictly faster (the state-side scatter math is
identical to the Gaussian TD path, which got 11–65×). It's the
**LBFGS emission update** (`update_observation_model!`) using
`x_reltol=x_abstol=g_abstol=f_reltol=f_abstol=1e-12`. Worth profiling
to confirm, but most likely:
* LBFGS now does *more* inner iterations because the smoothed states
  are themselves more accurate (better Newton convergence in `smooth!`),
* the per-iteration cost of `Q_obs!` / `gradient_observation_model!` is
  dominated by `exp.(C*x + d)` regardless of what changed in the state
  side.

Concrete fixes to try, in order of likely payoff:
1. Loosen LBFGS tolerance to `1e-8` (the surrounding EM converges at
   `tol=1e-6`, so `1e-12` per-EM-iter is way overkill).
2. Cap LBFGS iterations per EM step (currently unbounded).
3. Profile `gradient_observation_model_single_trial!` for the hot loop.

## Reproducing

From repo root, on either branch:

```bash
julia --project=. benchmarking/refactor_compare.jl > /tmp/dev.csv      # dev_ryan_
julia --project=. benchmarking/refactor_compare_main.jl > /tmp/main.csv # main
```

The two scripts are kept separate because the `main` API doesn't
support `kalman_filter=true`, returns `(elbos, param_diff)` from `fit!`,
and uses `log_d` instead of `d` in `PoissonObservationModel`. Output
columns are identical so the CSVs diff cleanly.
