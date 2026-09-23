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
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl                # ~30 min
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --full         # hours
```

Useful subsets:

```console
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --only=goldstandard
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --only=modelrecovery,switching
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --only=procedure --gen=lqr
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --quick --no-figures
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --smoulder
$ julia --project=docs -t auto docs/dev/lqr/lqr_recovery.jl --smoulder --only=smoulder-gref
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
| `slds.jl` | the same for an `SLDS` with one free and one LQR state, plus segment-then-fit |
| `compare.jl` | LQR against a plain LDS: the competitor, the generators, held-out scoring |
| `report.jl` | tables, seed aggregation |
| `plotting.jl` | figures |
| `experiments.jl` | the eight sweeps |
| `smoulder.jl` | grouped Poisson recovery matched to the smoulder-reward task |
| `parameterization.md` | a design review: is the mixed-coordinate model the right parameterization? |
| `parameterization.jl` | the measurements behind it — self-contained, imports nothing from `src/` |
| `biological.md` | the same question for a system that may only be *approximately* control-like |
| `biological.jl` | its measurements — also self-contained |
| `implementation-plan.md` | the plan that follows from both: a closed-loop state model for noisy, approximately optimal neural systems |

`parameterization.md` is the one file here that is not a recovery sweep. It
takes the findings below as given and asks the prior question: whether the
mixed-coordinate (Hamiltonian) latent is the right way to write the inverse
control problem in the first place, and what the alternatives buy. Several of
the harder findings below — the cost scale, the reference/cost trade, the
`Σ_λλ` inversion in the switching fit — turn out to be properties of the
parameterization rather than of the data.

`biological.md` drops the premise that the data came from a controller at all,
which is the situation for neural data, and asks what can still be claimed. Its
central measurement is that `Σ_λλ` does **not** measure suboptimality: an
exactly optimal agent under plant noise already violates the adjoint recursion
by a large, systematic amount, while the Riccati-graph relation it *doesn't*
violate is the one that responds to suboptimality alone. It also evaluates
fixing `S`, constraining `Σ`'s costate block, and an equality-constrained KKT
backend.

## Smoulder-reward recovery suite

`--smoulder` selects an opt-in tier with a 12-dimensional LQR plant (24 latent
state-costate dimensions), 1,000 trials, 100 bins per trial, 150 Poisson
channels, eight centre-out targets, and three known reward labels. It runs only
the three experiments below unless `--only` says otherwise:

| name | question |
|---|---|
| `smoulder-lqr` | Which cost scale, costate-noise initialization, explicit `Σ`/`Qc` prior, emission initialization, and known-plant control best recover grouped costs? |
| `smoulder-gref` | Is poor raw `Gref` recovery merely latent rotation? |
| `smoulder-slqr` | Can a two-state Poisson SLQR recover a generated free/delay → controlled/reach transition while keeping reward known? |

There is deliberately no session variable and no emission grouping in these
experiments. Only `Qc` depends on the known reward label. Since the terminal
cost is the last member of `Qc`, both running and terminal costs get one copy
per reward, while `A`, `S`, `Gref`, `Σ`, and the Poisson emission are shared.

The full suite is expensive: each fit processes 15 million counts, and the
initialization/prior sweep contains multiple fits over each seed. Use
`--quick --only=smoulder-lqr,smoulder-gref,smoulder-slqr` as a smoke test before
submitting `--smoulder` as a threaded batch job.

### Reading the `Gref` audit

When the Poisson loading is fitted, plant coordinates have an orthogonal gauge.
If `x_true = T*x_fit`, consistency requires transforming all control parameters:

```math
A^* = T A T^\top,\quad S^* = T S T^\top,\quad
Q_k^* = T Q_k T^\top,\quad G_{ref}^* = T G_{ref}.
```

The audit estimates `T` by orthogonal Procrustes from the fitted and true
state-loading columns. It reports raw and aligned errors for the complete
parameter set, plus `Gref'Gref`, which preserves target lengths and pairwise
angles without choosing a gauge. Rotating `Gref` alone is not a valid recovery
assessment: it would put the reference in a different coordinate system from
the plant and costs.

It also reports a full least-squares linear alignment. The more general change
of coordinates uses `x_true = T*x_fit` and `lambda_true = T^{-T}lambda_fit`, so
`A`, `S`, and `Gref` transform as above while
`Q_k^* = T^{-T}Q_kT^{-1}`. If Procrustes and the full map agree and the latter's
`T'T` is near identity, the ambiguity is only rotational. If only the full map
works, the fit also contains latent scaling or shear and `Gref'Gref` is not an
invariant assessment.

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

Every ordinary LQR and SLQR recovery result also carries a `gauge` field with
raw, orthogonal-Procrustes, and full-linear versions of all parameter scores.
When a fitted emission moves the latent basis (notably under `--free-C`), table
rows print a compact second line with the three `Gref` errors, the target Gram
error, emission-alignment residuals, and the linear map's non-orthogonality.
Fixed `[I 0]` emissions reduce these checks to the raw score and omit the
redundant extra line.

The switching section adds the discrete-state question, scored separately:
balanced MAP accuracy, mean posterior mass on the true state, per-timestep
cross-entropy, and the switch-time error in timesteps. Each condition also
reports **γ at the generating parameters**, which is the reference the fit is
read against — on a mixed free/LQR system that reference is nowhere near 1.

`--selftest` checks the properties the whole comparison rests on: identity,
costate-scale invariance, exact recovery after a coherent orthogonal rotation,
and exact recovery after a coherent non-orthogonal scale/shear. The last two
transform every affected parameter, not `Gref` alone.

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
| `slds_segment.png` | segment-then-fit: the cost given the epochs, oracle and fitted |
| `model_recovery.png` | LQR − LDS held-out ELBO against trials and noise, with a rule at zero |
| `gold_standard.png` | the twelve procedures on the gold-standard configuration |
| `gold_standard_scale.png` | held-out score and closed-loop error against the initial cost scale |
| `gold_standard_params.png`, `gold_standard_entries.png` | the winning procedure's recovery |
| `priors_sigma.png` | the `Σ` prior's strength ladder, crossed with the initialization it replaces |
| `priors_slds_gamma.png` | `Σ`-only, `Qc`-only and combined priors against pinning `Σ`, in the switching fit |

## The experiments

| # | name | question |
|---|---|---|
| 1 | `overview` | what recovery looks like across the structural and generative conditions, one row each |
| 2 | `design` | how much do a terminal condition, a within-trial cost change, and the number of distinct reference targets each buy? (and `2b`: is the reference identified at all?) |
| 3 | `scale` | how much of the remaining error is sampling error — trials and trial length, so a floor can be told from a slope |
| 4 | `procedure` | the fitting procedure rather than the model: iterations, restarts, initialization, inner-loop budget |
| 4b | `initialization` | where to start: ladders on both blocks of `Σ` and on the initial cost scale, a two-axis grid over them, and a combination table run on both generators |
| 4c | `modelrecovery` | can you tell an LQR from a plain LDS at all? Three generators, two candidates, held-out ELBO |
| 4e | `priors` | what an explicit inverse-Wishart prior on `Σ` or on the cost buys over an initialization that happens to stick |
| 4d | `goldstandard` | one realistic configuration — 3 cost levels, 8 ring targets — searched over twelve fitting procedures, plus whether cross-validation can select the cost scale |
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

### Switching LQR is half identifiable, and the two halves fight

With `Σ` pinned at the truth's blocks the `SLDS` finds the epochs about as well
as the generating parameters do — γ balanced accuracy **0.865** against 0.824 at
the truth, onset error 5.5 timesteps. Estimating `Σ` instead collapses the fit
onto one state (γ **0.144**): an LQR discrete state with a free innovation
inflates until it *is* a second free state.

But the discrete path and the LQR state's parameters want opposite values of the
same number, and no setting gets both:

| `Σ_λλ` | γ fitted | γ at truth | `Qc` | closed-loop |
|---|---|---|---|---|
| 1e-4 (single-system optimum) | 0.498 | 0.523 | **0.104 / 1.00** | **0.038** |
| 2.5e-3 | 0.497 | 0.555 | 0.125 / 1.00 | 0.025 |
| 1e-2 | 0.500 | 0.765 | 0.459 / 0.89 | 0.054 |
| 2e-2 (switching default) | **0.865** | 0.824 | 0.829 / 0.96 | 0.075 |

γ is a plug-in scored at the smoothed posterior mean, and that mean never sits
exactly on the Riccati graph — so a costate innovation tight enough to identify
the cost makes the LQR state's per-timestep likelihood hopeless and every
timestep goes to the free state. Note that γ *at the generating parameters*
moves with it: this is the model, not the optimizer.

Two-stage fitting inside the joint model half works: annealing to `1e-3` gives
the best γ in the table (0.899, onset 4.0) without buying the cost, and
annealing to `1e-4` destroys γ again (0.538).

**What sharpens γ:** a quieter emission (0.958, onset 1.7), longer trials
(0.944), no reference to estimate (0.927). **What does not:** more trials
(0.882), and — against expectation — observing the costate (0.625 against 0.982
at the truth; the fit does not exploit it).

**What would improve it: fit the epochs and the cost in sequence, not jointly.**

| | `Qc` | closed-loop | trials kept |
|---|---|---|---|
| joint, `Σ_λλ` tuned for γ | 0.829 | 0.075 | — |
| oracle segmentation → single-system fit | **0.094** | 0.036 | 150/150 |
| fitted segmentation (loose stage 1) → fit | 0.181 | 0.044 | 142/150 |

Stage one at a loose `Σ_λλ` finds the epochs; the single-system machinery then
identifies the cost on the sliced segments, at a tight one. The fitted
segmentation lands within 2× of the oracle, so **the switching layer is the
bottleneck, not the LQR identification given the epochs** — and the two-stage
procedure gets both answers where no joint setting gets either pair.

Best parameters inside the joint fit come from pinning `Σ`, observing the
costate and annealing: `Qc` 0.037/1.00, terminal cost 0.303/1.00, `Gref`
0.313/1.00, closed loop 0.035 — with γ still only 0.635.

### Is the model class identifiable? Only where the LQR is correctly specified

Held-out ELBO per timestep, eight ring targets, terminal factor off so the two
candidates score the same quantity:

| generated by | truth | LQR fit | LDS fit | LQR − LDS | picked |
|---|---|---|---|---|---|
| the LQR model's own chain | −0.540 | **−0.547** | −0.657 | +0.112 | LQR ✓ |
| the LQR *optimal trajectory* | −0.640 | −0.470 | **−0.433** | −0.034 | LDS ✗ |
| a first-order attractor (LDS) | −0.427 | −0.576 | **−0.429** | −0.145 | LDS ✓ |

The LDS competitor has *more* free parameters (52 against 33), so the LQR's win
on its own chain is not a win on parsimony.

The middle row is the important one. On optimal-trajectory data — the case the
model is *for* — both fits beat the generating model, so neither is well
specified, and the LDS edges it. An optimal path's mixed-coordinate residual is
zero up to slack rather than a draw from the forward chain, and no amount of
data repairs that. **You cannot establish "this behaviour was generated by an
LQR" by model comparison against a linear alternative.** What you can establish
is the weaker claim the closed loop supports: that the fitted controller
reproduces the same state dynamics.

### A gold standard for three cost levels and eight ring targets

A centre-out reach: `n = 2`, eight targets equally spaced on a ring, three cost
levels (near-zero delay, running from onset, terminal at the endpoint), trials
from an agent that solves it. Twelve procedures, ranked on the closed loop:

| procedure | `Qc` run | `Gref` | `S` (scale) | closed-loop |
|---|---|---|---|---|
| **tight `Σ_λλ` + `q0` = 0.2** | **0.050 / 1.00** | 1.041 / 0.01 | 0.206 | **0.011** |
| tight + 4× iterations | 0.067 / 0.99 | 1.048 | 0.361 | 0.016 |
| tight + costate observed | 0.141 / 0.98 | 0.801 / 0.60 | 0.383 | 0.020 |
| default (tight, `q0` = 0.4) | 0.089 / 1.00 | 1.049 / 0.01 | 0.843 | 0.033 |
| tight + 4 restarts | 0.089 / 1.00 | 1.049 | 0.843 | 0.033 (identical) |
| loose `Σ_λλ` | 0.865 / 0.84 | 1.415 / −0.67 | 1.000 | 0.075 |
| anneal + costate observed | 1.000 / 0.50 | **0.102 / 1.00** | 0.582 | 0.053 |

Two blocks are never recovered in any procedure. The **terminal cost** reads
~1.00 throughout: the endpoint condition constrains one factor at one timestep
per trial, and that is not enough. The **delay cost** reads 5–12, because it is a
near-zero regime and its relative error has a near-zero denominator — the fit
puts a real cost where the truth has almost none.

**And the cost scale cannot be cross-validated.** The obvious rescue for "the
fitted scale does not move from `q0`" is to select `q0` by held-out likelihood.
It does not work — the held-out score is monotone in `q0` and prefers the
smallest value on offer, while the closed-loop error has an interior minimum:

| `q0` | 0.05 | 0.10 | 0.20 | 0.40 | 1.00 | 2.00 |
|---|---|---|---|---|---|---|
| held-out / step | **−0.487** | −0.593 | −0.877 | −1.190 | −1.845 | −5.217 |
| closed-loop rmse | 0.066 | 0.052 | **0.011** | 0.033 | 0.111 | 0.188 |

Under misspecification the likelihood rewards hedging, and a smaller cost hedges.
So the scale is genuinely not determined by the data under this observation
model: it has to come from a prior, an external calibration, or a convention —
and until it does, only scale-invariant summaries mean anything.

### Priors do the job the initialization was doing silently

Two of the findings above are really the same complaint: a number that matters is
being set by where the fit starts rather than by anything stated. `LQRStateModel`
now takes `Σ_prior` and `Qc_prior`, both inverse-Wishart, and the question is
what they buy. Strengths below are `ν`, a pseudo-count against the ~12,000
transitions in this configuration — so `ν = 1e2` is a whisper and `ν = 1e5` is
louder than the data.

**A prior on the cost pins the scale the data cannot.** This is the fix for
"cross-validation cannot select `q0`":

| cost prior `ν` (per-epoch modes) | 0 | 1e2 | 1e3 | 1e4 | 1e5 |
|---|---|---|---|---|---|
| `S` (scale error) | 0.842 | 0.845 | 0.785 | 0.139 | **0.010** |
| terminal `Qc` | 1.000 | 0.775 | 0.549 | 0.116 | **0.006** |
| closed-loop | 0.033 | 0.033 | 0.031 | 0.015 | **0.007** |

At full strength it beats the best `q0` anyone could have guessed (0.007 against
0.011), and — unlike `q0` — its strength is a number in the model rather than an
artefact of the starting point.

It is doing real work in both directions, which is the test that it is a prior
and not a fudge. Moving only the running-cost mode to 0.6 against a truth of 0.2
degrades the answer:

| cost prior `ν` (mode at 0.6) | 1e3 | 1e4 | 1e5 |
|---|---|---|---|
| `S` | 0.895 | 1.072 | 1.939 |
| closed-loop | 0.035 | 0.042 | 0.066 |

**A prior on `Σ` replaces pinning it in the switching fit** — the concession
every working switching row rested on. With `Σ` *estimated* throughout:

| | γ | γ at truth | `Σ xx` | onset |
|---|---|---|---|---|
| no prior | 0.144 | 0.824 | 5.748 | 19.2 |
| prior ν=1e2 | 0.500 | 0.824 | 0.810 | 19.8 |
| prior ν=1e3 | 0.774 | 0.824 | 0.344 | 8.2 |
| **prior ν=1e4** | **0.871** | 0.824 | **0.030** | 5.0 |
| prior ν=1e5 | 0.871 | 0.824 | 0.003 | 5.0 |
| `Σ` pinned (the concession) | 0.865 | 0.824 | — | 5.5 |

A prior at `ν = 1e4` matches pinning and slightly beats it, while estimating the
innovation rather than being told it — and it recovers `Σ` itself to 3%. It
saturates by `1e4`. The LQR state's *cost* is no better for it (0.83–0.95
throughout): the prior fixes the collapse, not the cost, which still needs the
two-stage fit.

**The `Qc` prior fixes the continuous control problem in the switching fit, but
not the switching problem.** The new switching cross-check makes that separation
visible (default tier; the switching tier has one seed):

| switching fit | `Qc` run | `Qc` terminal | `S` scale | `Σ xx` | closed-loop | γ | onset |
|---|---:|---:|---:|---:|---:|---:|---:|
| no prior | 0.817 | 0.997 | 0.995 | 5.730 | 0.075 | 0.142 | 19.2 |
| `Σ` prior `ν=1e4` | 0.945 | 2.994 | 0.994 | **0.030** | 0.075 | **0.870** | **5.0** |
| `Qc` prior `ν=1e4` | **0.107** | 0.012 | 0.011 | 6.383 | **0.007** | 0.272 | 19.2 |
| both, `ν=1e4` | **0.107** | **0.009** | **0.009** | **0.014** | **0.007** | 0.500 | 19.8 |

Thus neither prior improves every notion of recovery. `Σ` regularization is
the discrete-state lever; `Qc` regularization is the cost-scale/control-problem
lever. Combining them recovers the continuous parameters but changes the
competition with the free state enough that the epoch fit falls back to chance.
Do not read good parameter recovery as evidence that the discrete path recovered,
or vice versa.

`Qc_prior` now accepts a vector aligned with `Qc`, with `nothing` for any epoch
that should remain unregularized. These rows use modes `[0.2, 0.02, 3.0]` for
running, delay and terminal costs in the single-system fit and `[0.2, 3.0]` for
running and terminal costs in the switching fit. That removes the old shared-mode
artifact: at `ν=1e4`, terminal-cost error falls from about 0.93 to 0.012 in the
switching model, and to 0.009 when combined with the `Σ` prior.

**What a prior on `Σ` does not do is rescue a bad start.** Crossed with the
initialization it was meant to replace:

| | `Qc`, init `Σ_λλ` = 1e-4 | `Qc`, init `Σ_λλ` = 5e-2 | closed-loop, loose init |
|---|---|---|---|
| ν = 0 | 0.089 | 0.865 / 0.84 | 0.075 |
| ν = 1e3 | 0.088 | 0.624 / 1.00 | 0.075 |
| ν = 1e5 | 0.097 | 0.556 / 1.00 | 0.075 |

The prior recovers the cost's *shape* from a loose start (correlation 0.84 → 1.00)
but never the control problem — the closed-loop error sits at 0.075 whatever the
strength, against 0.033 from the tight start. On a single system the
initialization is still the lever; what the prior adds there is `Σ` itself,
recovered from 4.42 to 1.55.

### If you are fitting one of these to real data

1. **Start `Σ`'s costate block small** (`1e-4`); its state block does not matter.
   This is the largest single lever here — a 7× difference in cost recovery, and
   larger than the terminal condition, the references, the iterations or the
   amount of data.
2. **State the cost scale as a `Qc_prior`, not as a starting value.** At a
   strength comparable to the transition count it beats any `q0` you could have
   guessed, and a misspecified one degrades the answer visibly rather than
   silently. Left to the initialization the scale is a prior anyway — just an
   invisible one of infinite strength.
3. **Report the closed loop.** `(I + S P)⁻¹ A` is invariant to the cost scale and
   comes back 3–10× better than the cost matrices do. If you must report the
   cost, report its shape after canonicalization and read `S` for what the scale
   is doing.
4. **Use two or more distinct reference targets, or freeze `h`.** One target
   identifies nothing about the reference; two or more identify its contrasts.
   Past four, nothing more is bought.
5. **The reference needs the costate.** It enters only through the costate half
   of the affine term, so the tight innovation that fixes the cost makes `Gref`
   unidentifiable. If the reference is the question, observe the costate and
   anneal; otherwise do not report `Gref` at all.
6. **Do not spend compute on restarts or on iterations past a few hundred.**
   Neither changes a printed digit. Spend it on observing more of the latent.
7. **Do not use a likelihood to choose among fits.** Under misspecification the
   ELBO prefers the wrong one, a warm start at the truth walks away from it, and
   held-out likelihood cannot select the cost scale — it prefers whichever fit
   hedges most.
8. **For a switching model, choose the prior for the question you care about.**
   A `Σ_prior` with `ν` of order the transition count recovers the epochs and
   `Σ`; a `Qc_prior` recovers the cost scale and closed loop. Combining them did
   not recover the epochs in this experiment, so it is not a free improvement.
   Then fit the
   epochs and the cost in **two stages** — a loose costate innovation to segment,
   the single-system machinery at a tight one on the segments — because no single
   setting recovers both.
9. **Do not claim the data came from an LQR on the strength of a model
   comparison.** Against a plain LDS it only works where the LQR is correctly
   specified, which is not the case the model is for.

## Findings on the package

Things this harness turned up that are about `src/`, not about the models it
fits. The directory is a measuring instrument, so the fixes live in `src/` and
`test/`; each entry says where. #1–#3 were closed by milestone M0 of
[`implementation-plan.md`](implementation-plan.md), #4 earlier, #5–#7 after
it.

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

**Fixed.** The statistics are now allocated per discrete state, at all three
sites (the M-step, the fit's preallocation, and the terminal normalizer's
probe). `test_slds_lqr_state_order` fits the pair above in both orders and gets
the same model; the ordering workaround is no longer needed.

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

**Fixed, by refusing.** `validate_SLDS` already had a rule for this, but nothing
called it, and it was stricter than the model: it refused any state with more
than one `Qc`, which includes the running-plus-terminal state this harness uses.
The rule is now the precise one — a state's *transitions* follow one cost; a
separate terminal cost, read only by the terminal factor, is fine — and it is
enforced wherever the model is used (`fit!`, `smooth`, `elbo` and both `rand`
methods). The sampler rolls that transition cost's `M`, which need not be
`M[1]`. See `test_slds_lqr_rand_schedules`.

### 3. `tol` is an absolute ELBO change, and these models never reach it

`converged = iter > 1 && abs(elbos[iter] - elbos[iter-1]) < tol`. On an
inverse-LQR fit the bound creeps upward by small amounts for thousands of
iterations, so with ELBOs of order `1e5` even `tol = 1e-6` is roughly `1e-11`
relative and is never tripped — every fit in every table below runs to
`max_iter`, and the harness reports a per-iteration ELBO *creep* instead of a
converged flag because a flag built on `tol` distinguishes nothing. A relative
criterion (`|Δ| < tol * |elbo|`), or a stopping rule on the parameters rather
than the bound, would make `fit!` say something useful about convergence.

**Fixed, opt-in.** `fit!` on an `LDS`, a Poisson `LDS` and both inverse-LQR
emissions takes `rtol` alongside `tol`, and stops once
`|Δ| < max(tol, rtol · |ELBO|)`. The default `rtol = 0` keeps the absolute test,
so no existing fit changes; pass something like `rtol = 1e-8` to have a large fit
stop on the same terms as a small one. See `test_em_relative_tolerance`.

### 4. Nothing regularized an LQR state's `Σ` — now something does

`LQRStateModel` accepted `P0_prior::IWPrior` and `x0_prior::MNPrior` but had no
prior on the innovation covariance or on the cost. The switching results turned
on exactly that gap: left free, an LQR discrete state's `Σ` inflates until the
state is a second free state and the fit collapses onto one regime.

**This one is now implemented** rather than reported — `Σ_prior` and `Qc_prior`
on `LQRStateModel`, acting in the profiled objective, its gradient, the noise
M-step and the prior log-density. See "Priors do the job the initialization was
doing silently" above for what they buy.

The M-step tests now differentiate an independent reference objective with
`ForwardDiff` for no, shared, combined, and per-epoch priors in both the profiled and
fixed-noise branches. They also check the closed-form posterior mode for `Σ`,
the prior contribution to the ELBO, and counting when grouped structure and
noise vary independently. That last check exposed and fixed an ELBO bug: the
grouped path had counted both structural priors according to the noise groups,
which under-counted varying `Qc` or over-counted shared `Qc`. The focused prior
suite passes 61/61.

### 5. A tied emission leaks into the costate columns on the pooled SLDS path

In a switching model that ties `C` across a `:free` state and an inverse-LQR
state with `observe_costate = false`, the plain fit let `C`'s costate columns
become non-zero (≈ 0.28 on the diagonal after one M-step in
`test_slds_lqr_grouped_free_state_pools`'s fixture), while the same fit through
the grouped path — with a grouping that splits nothing — kept them at zero. The
two ended 14 nats apart. A `:free` state defaults to `observe_costate = true`,
so its statistics carried the costate columns into the pooled `C`; the grouped
path happened to take its mask from the LQR state.

**Fixed, by making the two modes read the same coordinates.** A `:free` state in
a switching model now takes the inverse-LQR states' `observe_costate`
(`_match_costate_readout!`, run at every entry point: `fit!`, the E-step
helpers and `rand`). With `observe_costate = false` neither mode reads the
costate half directly; the free state still feels it through its dynamics,
which couple the two halves. Standalone, a `:free` state keeps reading
everything. The partial-tie route also had its own leak: its full-width
`CD_prior` could recreate costate coefficients from a non-zero `M₀`, so the
prior is now masked to the free columns too (`_masked_mn_prior`). Every
route — pooled, GLS and partial ties, Gaussian and Poisson, plain and grouped,
either state order — now leaves the costate columns at exactly zero;
`test_slds_lqr_tied_emission_mask` checks each.

### 6. A tied regression's noise update read the other regimes' previous coefficients

Found while checking #5, and not specific to LQR. A whole tie of `[C d D]`
(or `[A b B]`) is fitted onto the tie's first regime and copied onto the rest
at the end of the M-step. But `R` (or `Q`) is fitted in between, from each
regime's residual scatter at *that regime's own* regression matrix, and every
regime but the first still held the previous iterate's. So the noise update mixed
the new shared `C` with stale copies of it, and the fit depended on which regime
was listed first: `R` moved by 3e-3 and the fit ended 0.39 nats apart in the
mixed LQR fixture, and every tied combination of a plain Gaussian `SLDS` differed
between the two orders from the second iteration on.

**Fixed.** `_grouped_update_C_d!` / `_grouped_update_A_b!` now copy the slot's
fitted value onto the slot's other units before returning
(`_share_slot_obs!` / `_share_slot_dyn!`). This is a no-op for the `depends_on`
cells of one `LDS`, which alias one array. `test_SLDS_tied_params_order_invariant`
fits `(:C, :d)`, `(:C, :d, :R)`, `(:A, :b)` and `(:A, :b, :Q)` in both orders and
requires the same trace and parameters; before the fix all 36 of its checks fail.

What is *not* order-invariant, and should not be expected to be: an inverse-LQR
state's structural M-step is an L-BFGS solve along a nearly flat cost-scale
direction, and its stopping point moves the ELBO by ~1e-3 nats under a 1e-14
relative change in the data — the same amount the regime order moves it. That
is the optimizer, not the model; see the concerns under "Remaining concerns".

### 7. A multi-trial `rand` depended on the thread layout

The `LDS` sampler split `rng` into `min(ntrials, Threads.maxthreadid())`
`MersenneTwister` children, one per chunk of trials, so the same seed gave
different data on different thread counts. Julia 1.12+ starts one interactive
thread by default, which put `maxthreadid()` at 2 on 1.13 and 1 on 1.10 in the
same CI configuration. `MersenneTwister`'s integer seeding also changed between
those versions.

**Fixed.** Each trial now draws from a `Xoshiro` of its own, seeded with the
trial's `UInt64` off `rng`, drawn in trial order before anything is sampled.
A trial's data depend on `rng` and its index alone: the same at `maxthreadid()`
1, 2 and 8, and the same on 1.10 and 1.13 (`Xoshiro`'s seeded stream agrees
across them) to the last bit of the BLAS. `test_multitrial_rand_is_per_trial`
pins trial `i` to the single-trial draw from its seed. The `SLDS` and LQR
samplers draw serially from `rng` and were never affected. This changes the
data every multi-trial `rand` call produces for a given seed.

## Remaining concerns

Reported, not fixed: each is a design call or outside this branch's scope.
Most urgent first.

1. **The switching ELBO counts a tied parameter's prior once per regime.**
   `_slds_prior_logdensity` (and `_grouped_slds_prior_logdensity`, per regime
   again) sum every regime's `_state_prior_logdensity` /
   `_obs_prior_logdensity`. The M-step applies a tied group's prior once
   (the pooled or GLS solve, and `_distinct_by_slot` in the noise updates).
   So with a tie and a prior, the reported trace is not the objective being
   maximized. It is off by `(K − 1) · log p(θ_tied)`, which can make it
   non-monotone, and it skews any comparison across `K`. The fix is to count
   once per distinct version, as the grouped cells already do.

2. **Under terminal conditioning, the discrete chain ignores `log Ẑ`.**
   `mstep!` runs the plain Baum–Welch update (`StatsAPI.fit!(dl, …)`) whether
   or not the model conditions. The probe that estimates
   `log Ẑ = log p(terminal = 0)` carries copies of `A` and `πₖ`, so the
   normalizer depends on them. The state M-step (including `x0`/`P0`) is
   fitted against it; `A`/`πₖ` are not. With the probe held fixed, the chain's
   surrogate is `Σ (ξ_data − ξ_probe) log A`. It is not concave wherever the
   probe expects more transitions than the data, so there is no closed form.
   It needs either an accept-if-improves step on the logits, or a documented
   statement that the chain is fitted to the unconditioned half.

3. **An inverse-LQR state's structural M-step is chaotic in its inputs.**
   Measured on the mixed switching fixture: scaling the data by `1 + 1e-14`
   moves the ELBO by 1.6e-3 nats after one structural M-step and by ~5e-3
   after five. It moves `S` by 1–2%. The L-BFGS solve stops at a different
   point along the nearly flat cost-scale direction. As a result, fits are
   not reproducible to that level across thread counts (see 6), BLAS builds
   or regime orders, and tests must compare what one M-step determines. A
   `Qc_prior` is the existing lever; a tighter `g_tol` only moves the problem.

4. **`validate_SLDS` is never called on the fitting path.** `fit!`, `rand` and
   the E-step helpers run only the state-model rules
   (`_validate_slds_state_models`). A model with `πₖ = [0.7, 0.7]` and a
   transition row summing to 1.2 fits without complaint: the first
   iteration is scored under the improper chain, and the M-step then
   renormalizes it silently. The comment at `fit_SLDS.jl`'s `Data`
   construction says the input dimensions are "enforced by `validate_SLDS`";
   they are not. Calling it from `_prepare_slds!` would fix this.

5. **Diagnostic warnings left in the free-state M-step.** `_free_regression`
   and `_free_noise_mstep!` (`lqr_mstep.jl`, from `79e9437`) emit an `@warn`
   on every M-step of every `:free` state. Each pays for a `cond` and two
   `eigvals`, and together they fill the test log. `@debug` would keep them
   available.

6. **Some reductions associate by thread count.** `joint_loglikelihood` for a
   Poisson `LDS` chunks by `Threads.nthreads()`; the Poisson emission M-step
   and gradient by `tasks_per_thread · nthreads()` and the workspace pool.
   The sums are deterministic for a given thread count but differ in the last
   bits across counts. That is harmless for the convex updates, but through
   3 it becomes ~1e-3 nats in an LQR fit. Chunking by trial count instead of
   thread count would make them layout-independent, as `rand` now is.
