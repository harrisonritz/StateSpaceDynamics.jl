# October 8, 2026: non-causal LQR performance audit

The causal additions did **not** reproduce a major performance regression in
non-causal LQR. The costly path was already present in the October 5–6 fits:
terminal conditioning repeatedly smooths a Gaussian prior probe, and per-trial
cost offsets had disabled the shared forward factorization. Each `(length,
offset)` bucket then ran a complete factorization on every gradient evaluation.

## PR attribution

There are two PRs numbered 21 in the local repositories:

- **smoulder-reward #21** (`c55579c`, October 8): causal fit options and figures.
  `lqr_causal` remains false by default. Its backend dependency is a local path,
  so the harness merge alone does not specify a backend revision.
- **StateSpaceDynamics #21** (`752a200`, October 2): hold states and switching
  boundaries. Compare to its first parent `b830dac`.

The causal backend itself merged as **StateSpaceDynamics #26**, `5c9fe2c`,
October 8. Its first parent before the causal changes is `c16012e`.

The offset path came from `764c17b` (October 2, merged in backend #23), paired
with harness `5795629`. Offsets are needed for correct event-aligned costs;
removing them would change which cost a trial reads.

Saved logs `plds-15124823_*.txt`, started October 6, already show the affected
non-causal `earl__go_win-target-750` fits. Their saved fit times divided by 200
scored iterations range from 153 to 254 seconds. These include periodic held-out
scoring and are averages, not timings of an individual E/M step. They predate
the October 8 causal merge. The dataset has 2,739 trials, 21 lengths, 19 starts,
and only two schedule endpoints (170 and 171), an ideal case for backward reuse.

## Controlled comparisons

Julia 1.13, four Julia workers, one BLAS thread. These are local measurements,
not a prediction of wall time on a SLURM node. Load/compilation is warmed before
measurement. The objectives and the 100-iteration inner optimizer cap stay fixed.

For cross-revision EM, observations and initial emission matrices were saved
once and reused. Generating each revision's own data is unsuitable: the simulator
also changed between revisions. Timings below use two repetitions, minimum
where noted, and exactly three scored iterations / two M-steps.

| Comparison | Before | After | Interpretation |
|---|---:|---:|---|
| Conditional EM, medium, `b830dac` vs `5c9fe2c`, same inputs | 13.75 s | 14.17 s | about 3% slower; same final ELBO |
| Joint EM, medium, same comparison | 3.42 s | 3.02 s | no regression; same final ELBO |
| Causal backend, small, `c16012e` vs `5c9fe2c` | identical ELBO traces | identical ELBO traces | causal additions do not change this non-causal fit |

The medium conditional profile puts about 76% of wall time in the state M-step;
the probe remains its main cost. Small unused hold/causal workspace allocations
were measured too: constructing a structural context was about 15–17 microseconds
versus roughly 100 milliseconds for an offset probe. They do not explain the
reported doubling and were left alone.

## Repair

For ordinary non-causal LQR with variable starts and shared endpoints, factor the
precision backward once per endpoint. Every trial ending there shares all
backward Schur complements after its first block. A bucket adds its own
initial-prior block, computes its covariance forward, and solves its means with
the cached backward factors. Covariance sharing still requires both length and
offset to agree. Factors and covariances are rebuilt at each parameter point;
input and initial-state designs are retained.

The optimization applies when at least two workspaces are available and there
are at least twice as many buckets as endpoints. Single-workspace calls, sparse
endpoint sharing, schedule boundaries, and causal dynamics keep their existing
paths. No likelihood term, prior, optimizer cap, or fit default changes.

Against the unmodified `5c9fe2c` checkout:

| Measurement | Before | After | Speedup |
|---|---:|---:|---:|
| Synthetic offset probe, n=8, 300 designs, 19 starts / 2 ends | 114.73 ms | 92.37 ms | 1.24x |
| Full conditional value/gradient at that point | 130.55 ms | 97.52 ms | 1.34x |
| Saved n=8 fit parameters and all 2,739 trial horizons, one probe (34 designs) | 56.14 ms | 15.72 ms | 3.57x |
| Conditional EM with two endpoints, small, median of two repetitions | 8.07 s | 2.90 s | 2.79x |

The saved-parameter probe uses
`earl__go_win-target-750/fits/lqr_k08_fa_s99_c3_by-reward-Qc_frz-S-h_b0_pr0.995_Srank8_rsnone_sig1000-0.0001_ssid_cd0.001`.
It is a component benchmark of one probe, not a full refit of all reward cells.
The complete EM comparison uses the same saved synthetic data and initialization
on both sides. The new final ELBO was within 0.14 nats of the old value and was
higher in both repetitions. Reversing the factorization changes floating-point
order, so optimizer trajectories and inner evaluation counts can change; the
fixed-point component timings separate the runtime saving from that effect.
The full gradient agrees to 2.6e-13 relative in the offset fixture.

The existing benchmark now includes `plqr_offsets` for future regression checks:

```sh
julia --project=benchmark/lqr -t 4 benchmark/lqr/run.jl \
  --workload=plqr,plqr_offsets --tier=small --iters=3 --reps=2 --profile
```

Local raw CSVs, logs, and the saved-parameter probe reproduction are in
`benchmark/lqr/results/2026-10-08-regression-audit/` (gitignored). From the repo:

```sh
julia --project=. -t 4 \
  benchmark/lqr/results/2026-10-08-regression-audit/real_probe.jl
```

## Verification

- Independent single-trial references check means, covariance, lag covariance,
  and entropy for different endpoints, pinned/unpinned terminal regimes, terminal
  on/off, trial-level initial-state inputs, and repeated parameter changes.
- Cost-offset, terminal-gradient, normalizer, probe aggregation, and shared
  covariance checks: 1,985 assertions passed.
- Causal horizon, dense likelihood, M-step gradient, and schedule-boundary
  conditional-gradient checks: another 330 assertions passed.
- The new shared-suffix conditional gradient matches finite differences of the
  exact QR objective, including trial-level initial-state inputs: two assertions
  passed (2,317 total). The new `plqr_offsets` benchmark smoke run and formatting
  checks also passed.

Full production iterations on the target SLURM nodes were not re-run. Existing
jobs have already loaded their code; the repair takes effect in newly started
Julia processes.
