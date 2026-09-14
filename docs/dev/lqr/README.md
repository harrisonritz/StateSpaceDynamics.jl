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

## One number to know before running anything

`Σ`'s costate block. The harness defaults it to `1e-4` for a single system and
`2e-2` for the switching one, and those two numbers disagree on purpose — see
"What the numbers say" below. It moves recovery more than anything else in this
directory, including how much data you have.

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
| `init_costate_noise_*.png` | the costate-innovation ladder — the single biggest lever |
| `init_controls_*.png` | the state-block and cost-scale ladders, as controls |
| `init_grid_qc_*.png`, `init_grid_closedloop_*.png` | the two-axis settings search, best cell ringed |
| `init_combos_*.png` | named initialization combinations, one figure per generator |
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
| 4b | `initialization` | where to start: ladders on both blocks of `Σ` and on the initial cost scale, a two-axis grid over them, and a combination table run on both generators |
| 5 | `switching` | the `SLDS` with one free and one LQR state, and its two separate questions — the LQR state's parameters, and the discrete path |

Every cell of experiments 2–4 is a median over seeds, with the figure's error
bar spanning the full seed range. That is not decoration: these errors are
heavily right-skewed, most seeds of a condition land together and one
occasionally fails outright at fifty times the others' error, so a single seed
per cell would mostly report which seed it got.

## What the numbers say

From the default tier: `n = 3`, `T = 25`, `N = 400` trials, 3 seeds, medians.
Rerun it — these will drift, and the point of the harness is that they can be
re-measured rather than remembered.

### The control problem comes back; the cost that induces it comes back less well

Across every condition in section 2, the closed-loop plant `(I + S P)⁻¹ A` is
recovered to 2–4% while the running cost's entries are off by 14–28% and its
overall scale by a factor of about two. The gap is not noise: the closed loop is
invariant to the cost scale and the cost matrices are not. **Report the closed
loop, not the raw cost matrices** — it is the part of the answer the data
actually determine.

### The costate innovation's starting value is the single biggest lever

Bigger than the terminal condition, the references, the iteration budget, the
number of restarts, or the amount of data. On the hardest condition, trials from
an optimal agent:

| initial `Σ_λλ` | `Qc` rmse / corr | closed-loop |
|---|---|---|
| 1e-6 | 0.148 / 1.00 | 0.038 |
| 1e-5 | 0.147 / 1.00 | 0.038 |
| **1e-4** | **0.141 / 1.00** | **0.032** |
| 1e-3 | 0.568 / 0.76 | 0.040 |
| 1e-2 | 1.075 / 0.75 | 0.075 |
| 5e-2 | 1.095 / 0.32 | 0.068 |

There is a cliff between `1e-4` and `1e-3`, and the harness default sits at its
floor. The mechanism is the one the model implies: on the optimal path the
costate is a deterministic function of the state, so a costate innovation
comparable to the state's gives the fitted cost `n` directions to drift along
at no cost to the bound. The same ladder run on the **state** block moves `Qc`
from 0.144 to 0.141 across a 40× range — the effect is specific to the costate,
not to initialization in general.

It also collapses the slack ladder. `costate_slack` from 0 to 0.30 now gives
`Qc` 0.177–0.182 with correlation 1.00, where a loose start gave 0.61–0.90 with
correlations near 0.7. The "an exactly optimal agent is outside the model"
problem is mostly a statement about where the fit was started.

### The cost's overall scale is set by its initialization, not by the data

`S`'s relative error on a known-plant row is exactly the relative error in
`tr(Qc[1])`, and it tracks the initial cost scale `q0`:

| `q0` | 0.05 | 0.20 | 0.40 | 1.00 | 4.00 |
|---|---|---|---|---|---|
| `S` (scale error) | 0.93 | 0.25 | 0.79 | 3.86 | 15.94 |
| closed-loop | 0.067 | 0.018 | 0.032 | 0.110 | 0.263 |

EM moves the cost's *shape* onto the truth (correlation 1.00 nearly everywhere)
and barely moves its *size*. The minimum sits at the truth's own scale, which
you do not know in a real fit — so treat the cost scale as something you have
set, not something you have estimated, and read `S` to see what you set.

### More data does not help; more iterations trade shape for scale

From 25 to 800 trials, `Qc` goes 0.147 → 0.144 and the closed loop does not move
at all. Meanwhile `Δelbo` grows linearly in `N` (115 → 3069 over a 32× increase),
which is the signature of misspecification rather than sampling slack. **The
residual error is an identification floor.**

The iteration ladder is a trade rather than an improvement: from 25 to 1000
iterations `Qc` degrades 0.147 → 0.339 while `S` improves 0.98 → 0.19 and the
closed loop improves 0.037 → 0.026. EM spends its later iterations fixing the
scale at the expense of the shape.

Three procedure levers do **nothing at all**, to the printed digit: four random
restarts, the inner M-step budget (25 vs 400), and loosening `tol` from `1e-10`
to `1e-6`. Restarts fail because the ELBO ranks the cold start best every time —
under misspecification the bound is not a guide to recovery. And a warm start
*at the truth* ends at `Qc` 0.394: EM walks away from the generating parameters.

### One reference target is confounded with the affine drift; two are not

At a loose `Σ_λλ`, where the reference is identifiable at all (`rand`):

| targets | `Gref`, `h` estimated | `Gref`, `h` frozen at 0 |
|---|---|---|
| 1 | 1.001 / 0.81 | **0.178 / 1.00** |
| 2 | 0.098 / 1.00 | 0.099 / 1.00 |
| 4 | 0.077 / 1.00 | 0.078 / 1.00 |
| 8 | 0.076 / 1.00 | 0.078 / 1.00 |

Exactly the structure the model's own documentation implies. With one target the
input never varies, `−Q₁ G_r u` is a constant, and the costate half of `h` is
another one; freezing `h` at its true value recovers the reference 5.6×
better. From two targets on, the contrasts identify it and `h` is irrelevant.
Past four targets nothing more is bought.

### The reference lives in the costate, so it needs the costate

The tight costate innovation that fixes the cost destroys the reference: `Gref`
goes from 0.077/1.00 to ~1.05/0.02. As `Σ_λλ → 0` the smoother can satisfy the
costate recursion for *any* `G_r`, so nothing pins it. Annealing — fit loose,
then tighten — does not rescue it: it reverts to the loose answer. What does
work is observing the costate, and there annealing becomes the best setting
found anywhere (`Gref` 0.022 / 1.00 drawing from the model, 0.104 / 1.00 from
the agent).

**This is a genuine trade, not a tuning failure.** The default serves the cost;
if the reference is what you care about, start loose and observe the costate.

### Terminal conditions and multiple references buy less than expected

Once the costate innovation is set well, the structural features barely move the
cost. Drawing from the agent, `Qc` across the whole factorial sits in
0.136–0.275; the terminal factor moves the closed loop from 0.029 to 0.025 and a
delay epoch to 0.023. The terminal *cost itself* is never recovered — `Qc term`
reads ~1.00 in every condition, because the endpoint condition constrains one
factor at one timestep per trial and that is not enough.

What the terminal factor does buy is the cost *scale*: `S` improves from 0.71
(running only) to 0.57 (+terminal) to 0.37 (+terminal +delay). Within-trial
variation in the cost is what pins its size, which is the same story as `q0`
from the other side.

### Switching: pin `Σ`, and the costate innovation wants the opposite value

With `Σ` pinned at the truth's two blocks, the `SLDS` recovers the epochs about
as well as the generating parameters do — γ balanced accuracy **0.845** against
0.832 at the truth, with a mean onset error of 6.3 timesteps. Estimating `Σ`
instead collapses the fit onto one state (γ **0.135**, `stay` 0.93): an LQR
discrete state with a free innovation inflates until it is a second free state.

Conditions that sharpen γ: a quieter emission (0.957, onset 1.7 steps), longer
trials (0.950), no reference (0.927). Conditions that do not: more trials
(0.878), and — surprisingly — observing the costate (0.641 against 0.980 at the
truth; the fit does not exploit it).

The LQR state's parameters and its discrete path want *different* costate
innovations, and the table shows the inversion directly:

| `Σ_λλ` | γ fitted | γ at truth | `Qc` | closed-loop |
|---|---|---|---|---|
| 1e-4 (single-system optimum) | 0.499 | 0.524 | **0.104 / 1.00** | **0.037** |
| 2.5e-3 | 0.498 | 0.568 | 0.135 / 1.00 | 0.023 |
| 1e-2 | 0.695 | 0.764 | 0.817 / 0.86 | 0.067 |
| 2e-2 (switching default) | **0.845** | 0.832 | 0.853 / 0.93 | 0.075 |

γ is a plug-in scored at the smoothed mean, and the smoothed mean never sits
exactly on the Riccati graph — so a costate innovation tight enough to identify
the cost makes the LQR state's per-timestep likelihood hopeless and every
timestep goes to the free state. Note that γ *at the generating parameters*
moves too, so this is a property of the model, not of the optimizer.

Best parameter recovery in the switching section comes from pinning `Σ` and
observing the costate: `Qc` 0.036/1.00, terminal cost 0.177/0.98, `Gref`
0.331/1.00, closed loop 0.034, and the plant `A` to 0.058 when it is estimated
too — but γ only 0.641.

### If you are fitting one of these to real data

1. Start `Σ`'s costate block small (`1e-4`) and its state block anywhere.
2. Choose `q0` deliberately; the fitted cost scale will not move far from it.
   Read `S`, or better, report the closed loop.
3. Use two or more distinct reference targets, or freeze `h`. One target
   identifies nothing about the reference.
4. Do not spend compute on restarts or on iterations past a few hundred. Spend
   it on observing more of the latent.
5. For a switching model, pin or regularize the LQR state's `Σ`, and expect its
   costate innovation to want a looser value than a single-system fit does.
6. Do not use the ELBO to choose among fits. Under misspecification it prefers
   the wrong one, and a warm start at the truth walks away.

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
