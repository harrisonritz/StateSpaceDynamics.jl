# Inverse-LQR parameter recovery

A standalone validation harness for `LQRStateModel` and for an `SLDS` that mixes
a `:free` discrete state with an LQR one.

This is **not a test**. Tests check that the machinery computes what it claims;
this checks what you can actually *learn* from data — how close the estimated
cost gets, under which conditions, and how that degrades. It prints tables you
read and writes figures you look at, not an assertion that passes.

```console
$ julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --selftest
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --quick        # ~3 min
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl                # ~40 min
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --full         # hours
```

Useful subsets:

```console
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --only=design,switching
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --only=procedure --gen=lqr
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --quick --no-figures
```

Figures land in `docs/dev/lqr/figures/` (gitignored — they are regenerated on
every run).

## The files

| file | what is in it |
|---|---|
| `lqr_recovery.jl` | entry point: flags, tiers, the reading guide |
| `model.jl` | the generating models and the data they produce |
| `scoring.jl` | the metrics — parameter blocks and γ alike |
| `recovery.jl` | simulate → refit → score, for one system; `selftest` |
| `slds.jl` | the same for an `SLDS` with one free and one LQR state |
| `report.jl` | tables, seed aggregation |
| `plotting.jl` | figures |
| `experiments.jl` | the five sweeps |

## What it measures

Every parameter block is scored twice, on its independent entries (the upper
triangle for a symmetric block), after `rescale_costate!(...; target = :trace)`:

* **relative RMSE** — the RMSE divided by the RMS of the true entries, so blocks
  of different magnitude are on one scale. `0` is exact.
* **Pearson correlation** across entries, which ignores scale entirely. A high
  rmse with a high corr is a scale error; the reverse is a structural one.

Plus `closed-loop`, the steady-state plant `(I + S P)⁻¹ A` under the optimal
policy. That one is invariant to the cost scale, so it is the best single
summary of whether the fit found the same control *problem* even when it did not
find the same cost matrices.

The switching section adds the discrete-state question, scored separately:
balanced MAP accuracy, mean posterior mass on the true state, per-timestep
cross-entropy, and the switch-time error in timesteps. Each condition also
reports **γ at the generating parameters**, which is the reference the fit is
read against — on a mixed free/LQR system that reference is nowhere near 1.

`--selftest` checks the two properties the whole comparison rests on: a model
scored against itself is exactly `0 / 1` on every block, and rescaling a model's
costate by an arbitrary factor and re-canonicalizing returns it to the same
place.

## Two generative modes, and why both

* **`rand`** — trials drawn from the forward chain the smoother assumes.
  Recovery is a well-posed estimation question here, so this is the section
  whose numbers should be small.
* **`simulate_lqr`** — trials from an agent that actually solves the control
  problem. This is the case the model is *for*, and it is strictly harder: an
  exactly optimal trajectory has a rank-`n`, time-varying innovation that a
  full-rank constant `Σ` contains only as a limit, so fitting one is a
  projection rather than an estimation. `costate_slack` walks from that
  degenerate corner back toward the model — but not into it, since the agent
  acts on its own perturbed costate and the residual the mixed-coordinate model
  sees is `ν_t − Aᵀν_{t+1}`, an MA(1) process where `Σ` assumes white noise.

The switching section has the same split: `:rand` rolls the `SLDS`'s own chain,
and `:epoch` builds a delay-then-reach trial where the latent drifts under the
free state up to a per-trial switch and the agent then rolls the *optimal*
trajectory to the end of the trial.

## The emission

`C = [I 0]` by default: the state is observed directly and the costate is not.
An inverse-LQR fit has two nested identifiability problems — the emission's
latent basis, and the cost given the latents — and mixing them tells you nothing
about either. `observe_costate` widens the readout to `C = I`; `--free-C` folds
the latent-basis problem back in.

## Figures

| file | what it shows |
|---|---|
| `params_*.png` | every parameter block as truth, recovery and residual — the original-versus-recovered figure |
| `entries_*.png` | the same comparison entrywise, with the identity line |
| `design_rand.png`, `design_lqr.png` | cost / reference / closed-loop error against the number of reference targets, one series per structural condition |
| `identify_reference_*.png` | `Gref` recovery with the affine drift `h` estimated versus frozen |
| `scale_trials_*.png`, `scale_horizon_*.png` | the same errors against trials and trial length, on log axes |
| `procedure_iterations_*.png` | error against the EM budget |
| `procedure_levers_*.png` | the fitting-procedure levers, generative model held fixed |
| `elbo_traces.png` | ELBO traces as `elbo − elbo(truth)` |
| `slds_gamma.png` | discrete-state recovery per condition, against γ at the truth |
| `slds_trial_*.png` | one trial: observations with the true epoch shaded, and `γ(LQR)` for the fit and for the truth |
| `slds_params.png` | the LQR discrete state's parameters, truth versus recovery |

## The experiments

| # | name | question |
|---|---|---|
| 1 | `overview` | what recovery looks like across the structural and generative conditions, one row each |
| 2 | `design` | how much do a terminal condition, a within-trial cost change, and the number of distinct reference targets each buy? (and `2b`: is the reference identified at all?) |
| 3 | `scale` | how much of the remaining error is sampling error — trials and trial length, so a floor can be told from a slope |
| 4 | `procedure` | the fitting procedure rather than the model: iterations, restarts, initialization, inner-loop budget |
| 5 | `switching` | the `SLDS` with one free and one LQR state, and its two separate questions — the LQR state's parameters, and the discrete path |

Every cell of experiments 2–4 is a median over seeds, with the figure's error
bar spanning the full seed range. That is not decoration: these errors are
heavily right-skewed, most seeds of a condition land together and one
occasionally fails outright at fifty times the others' error, so a single seed
per cell would mostly report which seed it got.

## Findings on the package

Three things this harness turned up that are about `src/`, not about the models
it fits. None of them is patched here — this directory is a measuring
instrument, not a fix.

### 1. An `SLDS` mis-sizes its weighted sufficient statistics when the discrete states differ in regime count

`_mstep_slds!` allocates the responsibility-weighted statistics as `K` copies of
whatever shape `slds.LDSs[1]` needs:

```julia
# src/lds/fit_SLDS.jl:2930 (and again at :3473)
[_initialize_td_sufficient_statistics(T, slds.LDSs[1], dat.tsteps) for _ in 1:K]
```

Every discrete state then gets containers sized for the *first* one. A `:free`
state carries one cost regime; an LQR state with a terminal factor carries two.
Put the free state first and the LQR state's per-regime blocks are
under-allocated:

```julia
free = free_state_model(0.9 * Matrix(I, 4, 4), Matrix(0.05I, 4, 4))
lqr  = LQRStateModel(A, S, [Qrun, Qterm], Σ;
                     schedule = cost_schedule(T; terminal = true), terminal = true)
#  LDSs = [free_lds, lqr_lds]  →  BoundsError: 1-element Vector{Matrix{Float64}} at index [2]
#  LDSs = [lqr_lds, free_lds]  →  fine (the free state's blocks are merely oversized)
```

`slds.jl` here works around it by making the LQR state discrete state 1 — hence
`LQR_STATE = 1`, `FREE_STATE = 2`, which is otherwise an odd way to write
"delay, then reach". The general fix is to allocate per discrete state; the
ordering workaround still fails for two LQR states with different regime counts.

### 2. `rand` on an `SLDS` ignores an LQR state's cost schedule

`_extract_state_params(sm::LQRStateModel)` hands the sampler `cache.M[1]` — one
transition per discrete state, on the principle that an `SLDS` member carries one
cost because "the discrete state *is* the epoch". That is exact for a state with
a terminal factor and no onset (every *transition* is regime 1; the terminal
regime is read only by the endpoint factor), which is what this harness uses. It
is silently wrong for a state whose schedule changes mid-trial: the smoother and
M-step honour `schedule`, the sampler does not. Either the sampler should walk
the schedule, or the constructor should refuse a multi-transition schedule on a
member of an `SLDS`.

### 3. `tol` is an absolute ELBO change, and these models never reach it

`converged = iter > 1 && abs(elbos[iter] - elbos[iter-1]) < tol`. On an
inverse-LQR fit the bound creeps upward by small amounts for thousands of
iterations, so with ELBOs of order `1e5` even `tol = 1e-6` is roughly `1e-11`
relative and is never tripped — every fit in every table below runs to
`max_iter`, and the harness reports a per-iteration ELBO *creep* instead of a
converged flag because a flag built on `tol` distinguishes nothing. A relative
criterion (`|Δ| < tol * |elbo|`), or a stopping rule on the parameters rather
than the bound, would make `fit!` say something useful about convergence.

### 4. Nothing regularizes an LQR state's `Σ`

`LQRStateModel` accepts `P0_prior::IWPrior` and `x0_prior::MNPrior`, but there is
no prior or penalty on the innovation covariance. The switching results below
turn on exactly that gap: left free, an LQR discrete state's `Σ` inflates until
the state is a second free state, and the fit collapses onto one regime. An
inverse-Wishart prior on `Σ` — or the ability to constrain it to a scaled
identity — would be the difference between a switching inverse-LQR model that
fits and one that does not.
