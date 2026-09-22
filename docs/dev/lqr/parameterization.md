# Which parameterization? The Hamiltonian model and three alternatives

`README.md` in this directory measures what can be recovered from data by the
mixed-coordinate (Hamiltonian) inverse-LQR model. This document asks the prior
question: **is that the right parameterization of the inverse control problem?**

It is a design review, not a recovery sweep. Every numeric claim below is
produced by `parameterization.jl`, which depends on nothing in `src/` — it
reimplements the alternatives from scratch precisely so that the comparison
does not inherit the incumbent's choices.

```console
$ julia --project=. docs/dev/lqr/parameterization.jl                      # ~1 min
$ julia --project=. docs/dev/lqr/parameterization.jl --only=gauge,terminal
$ julia --project=. docs/dev/lqr/parameterization.jl --only=fit            # ~15 min
```

It runs against the *root* project rather than `docs`, because all it needs is
`LinearAlgebra` and `Optim`, both of which are already dependencies there.

## Verdict

The mixed-coordinate model is an excellent **M-step** and a poor **generative
model**. Writing the stationarity conditions as

```math
\begin{bmatrix} x_{t+1} \\ \lambda_t \end{bmatrix}
  = \mathcal{E}_t \begin{bmatrix} x_t \\ \lambda_{t+1} \end{bmatrix} + h + B_u u_t + \varepsilon_t,
\qquad
\mathcal{E}_t = \begin{bmatrix} A & -S \\ Q_t & A^\top \end{bmatrix}
```

makes the transition **linear in `A`, `S` and every `Qc`**, which is why EM
works at all and why the M-step in `lqr_mstep.jl` is closed-form. That is a real
and non-obvious achievement. But it buys linearity by promoting the Lagrange
multiplier `λ` from a *function of the trajectory* to a *latent variable with
its own process noise*, and the three problems the harness spends most of its
effort on are all consequences of that one choice:

1. the forward chain is unstable by construction, so the model cannot sample
   itself (`rand` diverges like `ρ(M)^T`; `_sample_lqr_path_conditional!` costs
   a dense `(dT)²` factorization and refuses `dT > 4000`; `simulate_lqr`
   bypasses the model entirely);
2. `Σ` must be non-singular on the costate block, while the process the model is
   *for* is exactly singular there;
3. the resulting misspecification is not repairable with data — `Δelbo` grows
   linearly in `N`, a warm start at the truth walks away from it, and on
   optimal-trajectory data a plain LDS wins the model comparison.

**There is an exact alternative.** The `n` unstable directions of the symplectic
flow are removed not by adding noise and conditioning, but by the change of
variables the control problem already provides: `λ_t = P_t x_t + g_t`. Imposing
it instead of approximating it gives an ordinary `n`-dimensional linear-Gaussian
state-space model whose transition is the closed loop `Φ_t = A − B K_t`. This
**closed-loop parameterization** (family **A** below) is, measured against the
harness's own failures:

| harness finding | closed-loop parameterization |
|---|---|
| "the cost scale cannot be cross-validated" (held-out score monotone in `q0`) | with `R` pinned the exact likelihood has an interior maximum at the true scale — but a shallow one, roughly ×0.65 to ×1.9 at 95 % even with everything else known |
| `Gref` 1.04 / corr 0.01 in the best cost-recovering procedure | reference perturbations spread over 15.1 of 30 timesteps, 38 % in observed coordinates |
| γ at truth is a function of `Σ_λλ` (0.52 → 0.83), and γ and the cost want opposite values | no `Σ_λλ` exists; γ at truth 0.875, onset error 3.7 steps, no trade-off |
| "a warm start *at the truth* ends at `Qc` 0.394" — EM walks away from the generating parameters | the truth is a stationary point: 400 L-BFGS iterations move the objective by 9e-4/step and every block holds |
| "do not use a likelihood to choose among fits"; four restarts change nothing | the likelihood separates the converged fit from every cold start by 115–290 nats, but does not rank unconverged fits by accuracy — the advice stands in practice |
| terminal cost never recovered (rel. err ≈ 1.00) | **also not recovered**, and now explained: an exact shaping equivalence on the unactuated subspace (§2.4), a property of the control problem under any parameterization |
| `rand` diverges; sampling needs a dense `(dT)²` solve | forward chain contracts in the cost-to-go metric; sampling is `O(T n²)` |
| latent dimension `2n` (24 for the smoulder plant), plus an exact terminal-normalizer sweep | latent dimension `n` (12), no normalizer — smoother blocks shrink `(2n)³/n³ = 8×`, against one added `O(Tn³)` Riccati sweep per parameter update |

The price is a non-convex M-step, and it is a real price. It is a *cheap*
non-convex step — the Riccati Jacobian is closed form (§3.1), so a gradient
costs one extra backward sweep and no implicit solve — but §2.5 shows cold
starts drifting along the shallow scale direction of §2.1 and ending 65–78× off,
with the closed loop 6–10 % wrong, while a warm start holds to 1 %. The objective
is right and hard to optimize from a bad start. The remedy is a data-driven
initializer (family **C**, and `implementation-plan.md` §6) together with a
scale that comes from a prior or from behavioural variability. Given that the
harness already reports that EM walks away from the truth, trading an
exactly-solved step in the wrong objective for an approximately-solved step in
the right one is still the correct direction — but it moves the difficulty into
initialization rather than removing it.

---

## 1. One defect, three symptoms

### 1.1 The geometry

`M_t` is symplectic, so its spectrum comes in reciprocal pairs `(μ, 1/μ)`:
exactly `n` modes grow and `n` decay. Measured on the configuration in
`parameterization.jl` (a damped 2-D point mass, `n = 4`, `m = 2`):

```
|mu| of the symplectic transition: 0.0833, 0.1039, 0.8511, 0.8511, 1.1749, 1.1749, 9.6225, 12.005
products of reciprocal pairs:      1.0, 1.0, 1.0, 1.0
symplectic defect ||M'JM - J||/||J|| = 2.9e-12
```

A forward roll grows by `12^T` — a factor of `2.4e32` over the 30 timesteps of
this configuration. That is the whole of symptom (1).

The `n` decaying directions are not scattered: they are the **graph of the
Riccati solution**, `{(x, P_t x)}`. Numerically, that graph is `M`-invariant:

```
||(I - proj)(M * graph(P))|| / ||M * graph(P)|| = 8.69e-10
```

Restricted to it, the flow is `Φ_t = A − B K_t`, and it satisfies the exact
identity (verified to `1e-16`)

```math
\Phi_t^\top P_{t+1} \Phi_t = P_t - \big(Q_t + K_t^\top R K_t\big) \preceq P_t,
```

so the closed loop contracts in the cost-to-go metric **even where
`ρ(Φ_t)` is not itself below 1** (in this configuration `max_t ρ(Φ_t) = 0.985`).
This is the correct sense in which "LQR is stable", and it is why the
closed-loop model needs no boundary condition to be well-posed: the terminal
cost is simply the last block of the precision, not a conditioning event.

So the two parameterizations are related as:

> `LQRStateModel` parameterizes the **whole `2n`-dimensional symplectic flow**
> and uses the terminal factor to project the posterior back onto its stable
> manifold. The closed-loop model parameterizes **the manifold itself**.

### 1.2 What `Σ_λλ` actually is

Once that is said, the costate innovation's role is clear. `Σ_λλ` is the width
of a Gaussian tube around the stable manifold — a **relaxation parameter**,
not a physical noise level. Three independent observations already in this
repository say so:

* `simulate_lqr`'s docstring: "the exactly-optimal trajectory has a *rank-`n`*
  innovation, supported on the graph of the Riccati map. A full-rank `Σ` — which
  the smoother requires — contains that only as a limit."
* `README.md`: the costate-innovation ladder "is the single biggest lever …
  bigger than the terminal condition, the references, the iteration budget, the
  number of restarts, or the amount of data", with **a cliff between `1e-4` and
  `1e-3`**. Relaxation parameters behave exactly like that; noise levels do not.
* `slds.jl`'s own comment: "the costate innovation is doing two different jobs.
  In one system it is slack to be removed; in a switching one it is the tolerance
  that makes the LQR state selectable at all."

And the decisive one: **it is not estimable.** `README.md` reports that leaving
`Σ` free in a switching fit collapses the model onto one regime (γ 0.135–0.144),
because "an LQR discrete state with a free innovation inflates until it is a
second free state"; `Σ_prior` at `ν = 1e4` is what fixes that. A parameter that
must be pinned or priored, that has to be set to conflicting values depending
on which question you are asking, and that moves the *reference* score
(γ at the generating parameters) as well as the fit, is a numerical knob wearing
the costume of a covariance.

---

## 2. What that costs, measured

**State the benchmark honestly first.** The generating process throughout this
section is a certainty-equivalent controller buffeted by plant noise and control
noise — what `simulate_lqr` produces, with control noise in place of
`costate_slack`. That process is **inside** the closed-loop class (`Ξ → 0`
recovers the noiseless optimum) and **outside** the mixed-coordinate class,
whose own docstring says so: an exactly optimal trajectory "has a rank-`n`
innovation … a full-rank `Σ` contains that only as a limit."

So this is not a neutral benchmark, and it is not meant to be. The claim is
about *which class contains the behaviour*, and an optimal controller under
noise is the standard model of motor behaviour — it is what `README.md` calls
"the case the model is *for*". Where a measurement is about the parameter map
rather than about fitting (§2.1, §2.2, §2.4) no generator is involved at all.

The plant is a damped 2-D point mass: `n = 4` (position and velocity), `m = 2`,
`T = 30`, eight ring targets, running plus terminal cost, `R = I` as the gauge,
and an emission that reads **position only** — so velocity is unobserved, which
is a strictly harder observation model than the `C = [I 0]` the harness uses by
default.

### 2.1 The cost scale: an exact gauge when `S` is free, a shallow one when it is not

`rescale_costate!`'s docstring states the invariance exactly:
`(λ, S, Q, h, Σ, Σf, hf) → (cλ, c⁻¹S, cQ, …)` leaves the conditional score
**unchanged, exactly**. Verified:

```
   The SAME c, with S = B R^-1 B' free to absorb it (S -> S/c, Q -> cQ):
   c = 0.25    d|K_1|/|K_1| = 0.0     d mean path = 0.0
   c = 4.0     d|K_1|/|K_1| = 0.0     d mean path = 0.0
```

So whenever `S` and `Σ` are both in the fit, the cost scale is an **exact flat
direction of the objective being maximized**. On rows where `S` is fitted, the
harness's `q0` table is reporting where on a gauge orbit the optimizer happened
to stop, which is why held-out likelihood cannot select it. *No amount of data
will.*

Pin the gauge — hold `R` fixed (equivalently `tr S`) — and the direction is no
longer exactly flat:

```
=== The cost-scale direction:  Q -> cQ,  with R = I held fixed ===
c       d|K_1|/|K_1|    rho(A-BK_1)   d mean path (rel)  endpoint |x_T - r|
0.25    0.1719          0.8505        0.0669             0.00043
0.5     0.0719          0.8509        0.0271             0.00042
1.0     0.0             0.8511        0.0                0.00041
2.0     0.0455          0.8512        0.0167             0.00041
4.0     0.0716          0.8512        0.0261             0.00041
```

The gains move by 5–17 % over a 16-fold range. **The observable mean barely
does**: 3–7 %, and the endpoint not at all — an optimal controller with a strong
terminal cost reaches the target whatever the cost's overall size, and what the
scale changes is only the shape of the transient. The closed-loop spectral
radius moves in the fourth digit, so the summary the harness (rightly) prefers
for its scale invariance is also the one that cannot see the scale. Profiling the
**exact** marginal likelihood along the ray, everything else held at the truth,
400 trials of `T = 30` observed in position only:

```
c         loglik/step    deficit vs c=1
0.0625    6.772054       -0.033245
0.25      6.801343       -0.003955
0.5       6.804665       -0.000634
0.71      6.805172       -0.000127
1.00      6.805299        0.000000   <- argmax
1.41      6.805241       -0.000058
2.0       6.805103       -0.000196
4.0       6.804808       -0.000491
16.0      6.804475       -0.000824
```

An interior maximum exactly at the true scale — so pinning `R` does convert an
exact gauge into an identified parameter. But a **shallow** one. Over the 12,000
timesteps of this dataset, halving the cost costs 7.6 nats and doubling it only
2.4; the 95 % profile interval is roughly `c ∈ (0.65, 1.9)`, and that is with
every other parameter known. Over-scaling is especially cheap — a 16-fold
over-scale costs 10 nats — because the closed loop saturates as the cost grows.

> **Correction.** An earlier version of this section reported a deficit of
> 0.72 nats per step at `c = 1/16` and concluded that "the cost scale is strongly
> identified; it was never a data problem". That came from a sign error in the
> feedforward sweep of `parameterization.jl` (the affine recursion carried a
> spurious `2KᵀRk` term), which made the endpoint error depend on the cost scale
> and so manufactured most of the curvature. The package's own
> `lqr_riccati_sequence` was never affected. With the sweep corrected and checked
> against a direct QP solve, the right conclusion is weaker: **the scale is
> identified in principle and poorly in practice**, which is much closer to what
> `README.md` observed than this section first claimed. §2.5 shows the practical
> consequence.

Two things follow. First, on rows where `S` is fitted the scale is a pure gauge
and must be fixed by convention; pinning `R` is the right convention. Second,
even with `R` pinned, the scale should come from somewhere other than the
feedback structure when it matters: a `Qc_prior`, the max-ent tie to behavioural
variability (§3.2), or a design whose references move the transient enough to
constrain it — and when none is available, **report only scale-invariant
quantities**.

Why does the scale drift even on `README.md`'s *known-plant* rows, where `S` is
frozen and the gauge is formally closed? Partly for the reason above — the
curvature is small. And partly because with `S` fixed the Riccati equation
`P = Q + AᵀP(I + SP)⁻¹A` is not homogeneous in `Q`, so the rescaling
`(λ, Q) → (cλ, cQ)` leaves a residual in `E`'s first row,
`x_{t+1} − A x_t + S λ_{t+1}`, which is penalized by `Σ_xx⁻¹` — and `Σ_xx` is
free, so a scale error can be absorbed into inflated state process noise. That
is a mechanism rather than a measurement, but it is consistent with `README.md`
reporting `Σ` recovered at **4.42** relative error without a prior and 1.55 with
one.

### 2.2 The reference lives in the wrong coordinate

In `LQRStateModel` the reference enters only through the costate half of the
affine term, `−Q_k G_r u_t`. `README.md` draws the correct conclusion — "the
reference lives in the costate, so it needs the costate" — and documents the
resulting trade: the tight `Σ_λλ` that identifies the cost takes `Gref` from
0.077/1.00 to ≈1.05/0.02, annealing does not rescue it, and the only fix is to
observe the costate, at which point the cost degrades. The gold-standard table
shows the trade in one place: the best cost procedure has `Gref` 1.041 / corr
**0.01**, and the best `Gref` procedure has `Qc` 1.000 / corr 0.50.

Under the closed loop the reference enters through the **feedforward** `k_t`,
which shifts the state mean directly. Perturbing one target by 10 %:

```
=== Where the REFERENCE leaves its fingerprint (one target -> 1.1 x) ===
   t:            1      4      7     10     13     16     19     22     25     28
   rel d:      0.0    0.1    0.1    0.1    0.1    0.1    0.1    0.1    0.1    0.1
   participation ratio: 15.1 of 30
   share of the perturbation in the OBSERVED (position) rows: 38.1 %
```

A 10 % change in the reference moves the trajectory by 10 % at essentially every
timestep, a third of it in the coordinates the emission actually reads. There is
nothing to trade off: the reference and the cost are identified by the same data
under the same settings.

### 2.3 The switching trade-off is a property of the relaxation

This is the most consequential difference, because the switching model is what
the delay-then-reach design is for.

In an `SLDS` both discrete states must share one latent, so today a `:free`
state carries `n` meaningless costate dimensions (and the per-state
sufficient-statistic sizing bug, `README.md` finding #1, is downstream of that).
The LQR state's per-timestep log-density contains
`−½‖λ̂_t − Q x̂_t − Aᵀλ̂_{t+1}‖²_{Σ_λλ⁻¹}`, and the smoothed `λ̂` never sits
exactly on the Riccati graph — so the penalty diverges as `Σ_λλ → 0`. Hence the
exact inversion the harness measures, in which γ and `Qc` want opposite values
of one number, **at the generating parameters as well as in the fit**.

Remove the costate and the comparison becomes ordinary: two `n`-dimensional
Gaussian transitions on the same variable, `x_{t+1} = A_f x_t + ε` against
`x_{t+1} = Φ_t x_t + B k_t + ε'`. Measured on a configuration matched to
`slds.jl` (`n = 2`, `T = 30`, 120 trials, `stay = 0.93`, onset uniform on
10:20, `obs_noise = 0.05`, free drift `0.9I` / `0.15I`, the same `Qrun`,
`Qterm`, `A` and `S` as `model.jl`), scoring γ **at the generating
parameters** with the same structured-variational plug-in the harness uses:

| parameterization | γ balanced acc. | posterior mass | onset error |
|---|---|---|---|
| mixed-coordinate, `Σ_λλ = 1e-4` (single-system optimum) | 0.52 | — | — |
| mixed-coordinate, `Σ_λλ = 1e-2` | 0.74 | — | — |
| mixed-coordinate, `Σ_λλ = 2e-2` (switching default) | 0.81–0.83 | — | 5.5–6.3 |
| **closed-loop (no such parameter)** | **0.875** | 0.879 | **3.71** |

The mixed-coordinate rows are `README.md`'s and `slds.jl`'s own measurements.
The margin (0.875 against 0.83) is within configuration differences and is not
the point. **The point is that the column of `Σ_λλ` values does not exist.**
There is no number that has to be set one way for the epochs and the other way
for the cost, so the harness's central switching conclusion — "fit the epochs
and the cost in two stages, because no single setting recovers both" — should
not apply. This configuration is not rigged in the closed loop's favour: the
free drift has `ρ = 0.900` and the LQR closed loop `ρ(Φ_1) = 0.896`, so the two
regimes are told apart almost entirely by innovation magnitude and feedforward,
not by decay rate.

### 2.4 What does **not** improve: the terminal cost

`README.md` finds that "`Qc term` reads ~1.00 in every condition". An earlier
version of this section attributed that to information decay — the terminal cost
being forgotten backwards through a contracting Riccati sweep — and recommended
design fixes. **Most of that was wrong.** The dominant cause is an exact
equivalence class of costs, and no design of the kind first recommended touches
it.

**Potential-based shaping on the unactuated subspace.** Let `N` span
`range(B)^⊥`. Because `NᵀB = 0`, `NᵀΦ_t = Nᵀ(A − BK_t) = NᵀA` for every `t`:
control cannot move the unactuated components in one step. So for any symmetric
`X`, with `M = N X Nᵀ`,

```math
Q_{\text{term}} \;\to\; Q_{\text{term}} + M,
\qquad
Q_k \;\to\; Q_k + M - A^\top M A \quad\text{for every running regime } k
```

shifts every cost-to-go by exactly `P_t → P_t + M`, which `Bᵀ` annihilates — so
no gain changes, and for any reference at rest (`Nᵀ(Ar − r) = 0`) no feedforward
changes either. It is the LQ instance of potential-based reward shaping, confined
to the subspace where a quadratic potential stays inside the model family.
Verified by construction (`--only=shaping`):

```
target 1: max ‖ΔK_t‖/‖K_t‖ = 4.3e-16, max ‖Δk_t‖/‖k_t‖ = 1.1e-15, max ‖ΔP_t − M‖/‖M‖ = 1.4e-15
(the shift is not small: ‖ΔQ_term‖/‖Q_term‖ = 0.03, ‖ΔQ_run‖/‖Q_run‖ = 0.08)
```

and independently by rank (`--only=subspace`): the stationarity equations that
determine `Q` from the gains have rank 17 of 20, with three normalized singular
values at `1e-11` against a gap to `1e-3` — the same three at `T = 6`, `10` and
`30`, so it is not decay. The missing three are this class, of dimension
`(n − m)(n − m + 1)/2`. For this plant that is the terminal **position** block —
the endpoint-accuracy term, the one a reaching study most wants.

What that means for the design fixes first proposed here:

* **Observing velocity does not help** — the equivalence leaves the whole
  trajectory distribution unchanged, so no emission can see it.
* **Shortening the horizon does not help** — the class exists at every `T`.
* **Unexpected perturbations do not help** — they probe the feedback gains,
  which are invariant.
* **What breaks it** is a reference, or an *anticipated* disturbance `d_t`, with
  `Nᵀ(A r + d_t − r) ≠ 0` — a target specified with a velocity changes the
  feedforward by 8.8 % under the same shaping. So does a design with a single
  cost regime: the class needs a terminal cost distinct from the running one.

Everything outside the class is weakly identified for the reason the earlier
version gave. Perturbing `Q_term` by 20 %, the part that survives lands on the
mean state thinly (relative change ≤ 0.0013) but not only at the end:

```
   t:            1      4      7     10     13     16     19     22     25     28
   rel d:      0.0    0.0    0.0 0.0001 0.0001 0.0002 0.0003 0.0005 0.0008 0.0013
   participation ratio 7.39 of 30;  share in observed rows 10.5 %
```

and more of the trial carries it when the running cost is weaker (participation
ratio 19.3 at `Q_run = 0`, 6.5 at `10 × Q_run`). Information about `Q_s` still
reaches `P_t` only through the contracting product `Ψ_{t,s} = Φ_{s−1}⋯Φ_t`
(§3.1), whose squared norm falls five orders of magnitude over 28 steps.

So: **report the terminal cost only through shaping invariants** —
`Bᵀ Q_term` and `Q̃_k = Q_k − Q_term + Aᵀ Q_term A`, both unchanged by the class
(§3.4 shows the least squares recovers them to `1e-11` at the truth) — and treat
the unactuated block as a convention unless the design includes a reference or
a predictable disturbance that the passive dynamics cannot follow. This holds for
the Hamiltonian parameterization too, since the class is a property of the
control problem rather than of any model of it.

### 2.5 End to end: the objective is right; the scale is set by the start

Fitting all 39 parameters — two cost matrices, eight reference vectors, three
noise scalars, plant known — by direct gradient ascent on the **exact** marginal
likelihood. 400 trials of `T = 30`, position only, 400 L-BFGS iterations.
The truth sits at `nll/step = −6.805299`.

| start | `Qrun` rmse/corr | `Qterm` rmse/corr | scale err | `Gref` rmse/corr | closed-loop | nll/step |
|---|---|---|---|---|---|---|
| `q0` = 0.05 | 68.7 / 0.934 | 1.00 / −0.109 | 68.8 | 0.0393 / 1.000 | 0.0641 | −6.7818 |
| `q0` = 1.0 | 77.5 / 0.976 | 1.00 / −0.194 | 77.5 | 0.0156 / 1.000 | 0.0754 | −6.7966 |
| `q0` = 100 | **0.333 / 0.979** | 0.992 / 0.668 | **0.289** | 0.0157 / 1.000 | 0.0648 | −6.7908 |
| `q0` = 10 000 | 64.7 / 0.980 | 0.685 / 0.663 | 65.5 | 0.0164 / 1.000 | 0.0964 | −6.7954 |
| warm start at the truth | 0.00374 / 1.000 | 9.2e-5 / 1.000 | 0.0027 | 0.0093 / 1.000 | 0.0089 | **−6.8062** |

(The truth's `tr Qrun` is 8.1e4, and `q0` enters through a Cholesky diagonal, so
`q0 ≈ 200` is the start that matches the truth's scale.)

> **Correction.** An earlier version of this table came from runs with the
> feedforward sign error described in §2.1. It showed the reference landing
> 748× too large from a cold start — a "`Q · Gref` valley" — and read the
> likelihood as ranking all five fits in exactly their order of accuracy. Both
> were artefacts. The corrected readings follow.

**1. The warm start at the truth stays there.** 400 iterations move the objective
by `9e-4` per step and every block holds. The generating parameters are, to
sampling noise, a stationary point — the opposite of `README.md`'s finding that a
warm start at the truth walks to `Qc` 0.394. As before, `Qterm` holding at
`9e-5` shows only that it did not move: §2.4's shaping class is exactly flat.

**2. Every cold start recovers the reference and the cost's shape.** `Gref` to
0.016–0.039 at correlation 1.00, and `Qrun` at correlation 0.93–0.98, from starts
spanning five orders of magnitude, under an emission that never sees velocity.
The reference/cost trade-off of the mixed-coordinate model does not appear.

**3. The cost's scale is set by where the fit starts.** Three of four cold starts
end 65–78× too large; only the start near the right scale stays near it (0.29).
That is §2.1's shallow profile in action: over-scaling costs almost no
likelihood, so nothing pulls the fit back. It is also, in substance, what
`README.md` reported for the mixed-coordinate model — and this section's earlier
claim that pinning `R` fixes it was wrong. Pinning `R` makes the scale
identifiable; it does not make it well identified. Take it from a prior or from
behavioural variability (§3.2), or report only scale-free quantities.

**4. The likelihood picks out the converged fit, and ranks nothing else.** Every
cold start sits 0.010–0.024 nats per step below the warm start — 115 to 290 nats
over the dataset — so none is an alternative optimum; they are unconverged along
the shallow direction. Among them, the likelihood order (`q0` = 1, 10 000, 100,
0.05) is close to the *reverse* of their closed-loop accuracy (0.05, 100, 1,
10 000). So the likelihood is a sound guide to which fit has converged, and no
guide to which unconverged fit is closest. `README.md`'s advice not to rank fits
by likelihood therefore stands in practice here too — not because the objective
is wrong, but because cold starts do not reach its optimum in a practical budget.

The closed loop is 6.4–9.6 % from the truth in every cold start, against 0.9 %
at the warm start: the shallow scale direction costs accuracy in the gains, not
only in the cost's size. The remedy is the initialization in §3.4 and in
`implementation-plan.md` §6, together with a scale that comes from somewhere.


---

## 3. The alternatives

### 3.1 A — the closed-loop (policy-space) parameterization *(recommended)*

Impose `λ_t = P_t x_t + g_t` instead of noising it. Solve the control problem
forward and write the agent as a controller:

```math
u_t = -K_t(\theta)\,x_t + k_t(\theta) + \eta_t, \qquad \eta_t \sim N(0, \Xi_t),
```
```math
x_{t+1} = A x_t + B u_t + d_t + \epsilon_t, \qquad \epsilon_t \sim N(0, \Sigma_x),
\qquad y_t = C x_t + \nu_t,
```

whose state marginal is an ordinary time-varying linear-Gaussian chain

```math
x_{t+1} = \Phi_t x_t + B k_t + d_t + w_t,\qquad
\Phi_t = A - BK_t,\qquad \operatorname{Cov}(w_t) = B \Xi_t B^\top + \Sigma_x,
```

with `K_t`, `P_t` from the backward Riccati recursion and `k_t`, `b_t` from its
companion affine sweep:

```math
K_t = (R + B^\top P_{t+1} B)^{-1} B^\top P_{t+1} A, \qquad
P_t = Q_t + K_t^\top R K_t + \Phi_t^\top P_{t+1} \Phi_t, \qquad P_T = Q_{k_T},
```
```math
k_t = -(R + B^\top P_{t+1}B)^{-1} B^\top b_{t+1},\qquad
b_t = -Q_t r_t + \Phi_t^\top\!\big(P_{t+1} B k_t + b_{t+1}\big) + K_t^\top R k_t .
```

**What it fixes, structurally.** Latent dimension `n` rather than `2n`. The
transition contracts in the `P`-metric, so `rand` is an ordinary forward roll,
the Kalman filter works in covariance form, sampling is `O(Tn²)` with no
`(dT)²` factorization and no horizon cap, and no terminal *conditioning* is
needed — the terminal cost is the last block of the Riccati recursion.
Suboptimality is **control noise** `Ξ_t`, which is where motor noise physically
is and which the exactly-optimal agent attains at `Ξ = 0`: the generating
process is *inside* the model class rather than on a measure-zero limit of it.
No `Σ_λλ`.

**The M-step is not closed form, but its gradient is.** Differentiating the
Riccati recursion, every `dK` term cancels by optimality (the envelope theorem),
leaving a *linear* recursion in `dP`:

```math
dP_t = dQ_t + dA^\top P_{t+1}\Phi_t + \Phi_t^\top P_{t+1}\,dA + \Phi_t^\top\, dP_{t+1}\, \Phi_t ,
```

and hence, for the cost block alone, the closed form

```math
dP_t = \sum_{s \ge t} \Psi_{t,s}^\top \, dQ_s \, \Psi_{t,s},
\qquad \Psi_{t,s} = \Phi_{s-1}\Phi_{s-2}\cdots\Phi_t, \quad \Psi_{t,t} = I .
```

(Verified against finite differences to `4.9e-6`.) So no AD through the Riccati
sweep and no implicit-function solve is required: a reverse-mode gradient costs
one extra backward sweep, `O(Tn³)`, and `Ψ` is the same object that quantifies
how far back a cost regime is identifiable. `dB` and `dR` follow from the same
envelope argument.

**Identifiability of plant against cost.** The data determine `{Φ_t}`, and
`Φ_t = A − BK_t` with `BK_t` confined to `range(B)`. Therefore
`(I − BB⁺)Φ_t = (I − BB⁺)A` — **the projection of `A` onto the orthogonal
complement of `range(B)` is identified outright**, `n − m` combinations of its
rows, with no reliance on regime contrasts (this presumes the latent gauge is
pinned, by a fixed `C` or by the audit). Within `range(B)`,
a *time-varying* `K_t` supplies many equations for the same `A`, so the
finite-horizon transient is an identifying asset. This is the exact opposite of
the mixed-coordinate model, which treats transitions as exchangeable given
their regime label and can therefore learn only from regime contrasts.

**Costs and caveats.** The M-step is non-convex and wants a decent
initialization (§3.4). EM's exact monotonicity is lost — but the harness already
reports that the bound misranks fits, so little is given up; direct gradient
ascent on the *exact* marginal likelihood is available instead, and is what
`parameterization.jl` does. The model commits to a horizon (or to a
receding-horizon rule); `LQRStateModel` is agnostic about that, which is a
genuine advantage of the incumbent. And A is a strictly *smaller* class — only
Riccati-consistent transitions, not every symplectic one — which is a feature
for inference and a limitation for exploratory fitting.

### 3.2 A′ — tie the control noise to the cost (maximum-entropy LQR)

Soft value iteration for the LQ problem at inverse temperature `β` gives a
*stochastic* optimal policy whose mean is the same `−K_t x + k_t` and whose
covariance is not free:

```math
\Xi_t = \big[\beta\,(R + B^\top P_{t+1} B)\big]^{-1}.
```

This is the Gaussian analogue of maximum-entropy IRL and it is a sub-model of
A with one fewer free parameter. Two consequences worth having:

* Scaling `(Q, R) → c(Q, R)` leaves `K_t` unchanged but sends `Ξ_t → c⁻¹Ξ_t`.
  So the cost scale is read off the **magnitude of behavioural variability** —
  a more strongly-costed agent is less variable — rather than from `K` alone.
  (`β` and `c` are confounded, so `β = 1` is the natural normalization and the
  cost carries the scale.)
* `Ξ_t` shrinks as `P_{t+1}` grows, so variability should fall toward the
  endpoint in a Riccati-determined way. That time course is a *signature*, and
  it is what separates control noise `B Ξ_t Bᵀ` (rank `m`, time-varying) from
  plant noise `Σ_x` (constant) — which are otherwise confounded, since both are
  additive innovations at the same point.

Use A′ when behavioural variability is signal rather than nuisance; use plain A
with a free `Ξ` when it is not.

### 3.3 B — control as inference: the trajectory Gaussian

Do not factor the trajectory as a Markov chain at all. Write

```math
p(x_{1:T}, u_{1:T-1} \mid \theta) \propto
\exp\Big( -\tfrac12\|x_1-\mu_0\|^2_{P_0^{-1}}
          -\tfrac12\textstyle\sum_t \|x_{t+1} - A x_t - B u_t - d_t\|^2_{\Sigma_x^{-1}}
          -\tfrac12\textstyle\sum_t \big[(x_t - r_t)^\top Q_t (x_t - r_t) + u_t^\top R u_t\big] \Big).
```

This is Gaussian with a **block-tridiagonal precision that is affine in the
cost**:

```math
\Lambda(\theta) = \Lambda_{\mathrm{dyn}}(A, B, \Sigma_x, P_0)
                  + \operatorname{blkdiag}(Q_1, \dots, Q_T, R, \dots, R).
```

Its attractions are real and specific. The cost enters as a **natural
parameter**, the Gaussian log-partition `½ηᵗΛ⁻¹η − ½log|Λ|` is convex in the
canonical pair `(η, Λ)`, and `(η, Λ)` is affine in `({Q_t}, {Q_t r_t}, R)` — so
the cost M-step is a **concave** problem (an SDP with `Q_t, R ⪰ 0`), globally
solvable rather than merely closed-form. `Λ ≻ 0` holds by construction, so there
is no instability and no terminal conditioning; `block_tridiagonal.jl` already
provides the `O(Tn³)` smoother and the `O(Tn³)` sampler. Note also the exact
relation to the incumbent: **the Hamiltonian equations are the stationarity
conditions of this exponent**, with `λ` the multiplier on the dynamics
constraint. B integrates the multiplier out analytically; `LQRStateModel`
promotes it to a noisy latent. In that precise sense the mixed-coordinate model
is a *relaxation* of B, and `Σ_λλ` is the relaxation width.

The caveat is decisive for this application. Setting
`λ_{t+1} := −Σ_x⁻¹(x_{t+1} − A x_t − B u_t)`, the stationarity conditions read

```math
\lambda_t = Q_t x_t + A^\top \lambda_{t+1}, \qquad
u_t = -R^{-1}B^\top \lambda_{t+1}, \qquad
x_{t+1} = A x_t - (S + \Sigma_x)\,\lambda_{t+1},
```

which is the LQR problem with `S` replaced by **`S + Σ_x`**: the plant noise
acts as a free extra control channel. This is the familiar optimism of
planning-as-inference, and here it means the object of interest is confounded
with the plant noise by an exact additive shift. B is the right choice when the
plant is near-deterministic (a cursor, a manipulandum) or `Σ_x` is calibrated
independently; otherwise A is right, because A models an agent that applies a
feedback law and is buffeted, which is the correct story for motor control.

### 3.4 C — two stages: free closed loop, then LMI inversion

The harness discovered empirically that the closed loop comes back to 2–4 %
while the cost is off by 14–28 %. That is a well-conditioned problem followed by
an ill-conditioned one, and it can be *split* rather than solved jointly.

**Stage 1.** Fit `{Φ_t}`, `{Bk_t}`, `C` and the noises as a free time-varying
(or regime-varying) LDS, with no optimality constraint. Standard machinery;
subspace identification handles the partial-observation gauge without local
optima and makes a good initializer.

**Stage 2.** Given `Φ` and `(A, B)`, inverse optimality is a **linear** system
in `(P, Q, R)`. From `K = (R + BᵗPB)⁻¹BᵗPA`,

```math
R K = B^\top P\,\Phi, \qquad Q = P - K^\top R K - \Phi^\top P \Phi,
\qquad P \succeq 0,\; Q \succeq 0,\; R \succ 0 .
```

So the set of costs for which the observed closed loop is optimal is a
**spectrahedron** — an affine subspace intersected with the PSD cone — and three
things follow that the current approach cannot deliver:

* **The honest answer to a non-identified problem is a set, not a point.**
  Report the cross-section of that cone at a stated normalization (`tr R = 1`),
  rather than a point estimate whose scale came from `q0`. The cone is a cone
  precisely *because* of the scale gauge, which appears here in its natural form.
* **A specification test.** If the spectrahedron is empty, no LQ cost makes the
  estimated closed loop optimal. This is what `README.md` finding #9 was
  reaching for and could not get: the LQR-versus-LDS held-out comparison pits
  two misspecified parametric models against each other, and on
  optimal-trajectory data both beat the generating model. The LMI feasibility
  gap instead asks whether the *nonparametric* closed loop lies in the
  optimality-consistent set. That is a test of optimality, not of parametric
  fit.
* **A data-driven initializer** for A, replacing `q0` — which, per §2.1, is
  currently an invisible prior of infinite strength.

**Measured, and two corrections to the above.** The stationarity equations are
indeed linear in the cost once the gains are fixed — `P_t` is affine in `Q` given
`{K_t, Φ_t}` — so the least squares needs no SDP solver. But they have a null
space, exactly the shaping class of §2.4, so the raw solution is wrong
(`‖Q̂_run − Q_run‖/‖Q_run‖ = 2.8` at the truth, and indefinite) while its shaping
invariants are right to `3e-11`, the error being a pure shaping move (`--only=subspace`). The
problem must be gauge-fixed before it initializes anything. And it is poorly
conditioned: 1 % relative noise in the gains becomes 20 % error in the invariant
`Q_k − Q_term + AᵀQ_termA`, and 5 % becomes 81 %. So stage 2 is an initializer
that needs a precise stage 1, not an estimator in its own right; the likelihood
fit that follows is what weights the equations properly.

The same equations answer a question the closed loop alone cannot: whether a
constant feedback offset could masquerade as intrinsic dynamics
(`A − BK_t = (A + BD) − B(K_t + D)`). It cannot — no cost makes the offset gains
optimal for the offset plant — but the rejection is about 8× weaker when only
near-stationary gains are available (`--only=subspace`). And the control
subspace itself falls out with no optimality assumption at all: the time
variation `Φ_t − Φ_s = −B(K_t − K_s)` spans `range(B)`, recovered here with
singular values `18.3, 11.7, 3e-16, 9e-18` and zero principal angle.

The limitations are the usual two-stage ones: stage-1 uncertainty does not
propagate (bootstrap, or run A from the stage-2 solution), and a stationary `K`
identifies the cost only up to the cone. Where the agent really is
infinite-horizon, the cone *is* the answer.

### 3.5 D — patches to the existing model, in order of value

None of these requires the architecture to change, and the first two address the
downsides directly.

1. **Stop assuming white noise on the costate.** `README.md` already identifies
   the exact misspecification: the agent acts on its own perturbed costate, so
   the residual the mixed-coordinate model sees is `ν_t − Aᵀν_{t+1}` — an MA(1)
   process — "where `Σ` assumes white noise". Put the slack on `ν` and let the
   mixed-coordinate innovation be its MA(1) image. That is a specific banded
   precision, well within `block_tridiagonal.jl`, and it replaces the
   `n(n+1)/2`-parameter `Σ_λλ` with **one interpretable scalar** `σ_ν²`
   measuring how far from optimal the agent is. The dominant lever stops being a
   lever and becomes an estimand.
2. **Fix the gauge; do not prior it.** Since the scale is an *exact* flat
   direction whenever `S` and `Σ` are fitted (§2.1), constrain the M-step —
   `tr S = tr S₀`, or `tr Qc[1] = n`, the normalization
   `rescale_costate!` already implements for scoring. One projection removes a
   flat direction at no cost in bias. `Qc_prior` at `ν = 1e5` currently does this
   job by brute force, at the price of a strength that has to be chosen and a
   mode that can be wrong (the harness measures both: it beats the best `q0`,
   and a mode at 0.6 against a truth of 0.2 degrades everything). Keep
   `Qc_prior` for genuine prior information; use a constraint for the gauge.
3. **The hybrid E-step — the cheapest decisive experiment.** Keep the existing
   mixed-coordinate M-step verbatim and replace only the E-step: smooth the
   `n`-dimensional closed-loop chain, then impute
   `λ̂_t = P_t x̂_t + g_t` with second moments
   `Cov(λ_t, λ_s) = P_t Cov(x_t, x_s) P_sᵗ`, and feed those sufficient
   statistics to `lqr_mstep.jl` as it stands. Re-solve the Riccati equation, and
   iterate. Same M-step code, different second moments; the E-step is stable and
   `8×` cheaper. This is a self-consistency (MM) scheme, not EM, so monotonicity
   is not guaranteed — but if the `Σ_λλ` lever disappears and the cost recovery
   improves, the diagnosis in §1 is confirmed for a day's work, before anything
   is rewritten.
4. **A relative convergence criterion** (`|Δ| < tol·|elbo|`) or a
   parameter-space stopping rule — `README.md` finding #3, still open.
5. **Allocate `SLDS` sufficient statistics per discrete state** — finding #1.
   Moot under A, where the LQR regime is `n`-dimensional like the free one and
   the padded costate block disappears along with the ordering workaround.

---

## 4. Side by side

| | **H** mixed-coordinate (current) | **A** closed loop | **A′** maxent LQR | **B** trajectory Gaussian | **C** two-stage + LMI |
|---|---|---|---|---|---|
| latent dimension | `2n` | `n` | `n` | `n + m` | `n` |
| forward flow | unstable, `n` growing modes | contracts in the `P`-metric | same | n/a (one joint Gaussian) | stable by fiat |
| sampling | `(dT)²` dense solve, capped | `O(Tn²)` forward roll | same | `O(Tn³)` banded | n/a |
| noise on the costate | **required**, non-estimable | none | none | none | none |
| suboptimality modelled as | costate innovation | control noise `Ξ` | `Ξ = [β(R+BᵗPB)]⁻¹` | trajectory temperature | stage-1 residual |
| optimal agent in the class? | no (measure-zero limit) | yes (`Ξ → 0`) | yes | yes | yes |
| cost M-step | **closed form**, linear | non-convex, closed-form gradient | same | **concave** (SDP) | linear + PSD (SDP) |
| cold-start conditioning | scale pinned by `q0`, which is flat | scale drifts along a shallow direction (§2.5); wants a C-style init or a scale prior | shallow, but max-ent ties the scale to variability | same | convex — no initializer needed |
| cost scale | exact gauge; needs a prior | identified (§2.1) | identified, plus variability | identified | reported as a cone |
| reference `Gref` | trades against the cost | identified with the cost | same | same | in stage 1 |
| terminal cost | not identified | unactuated block: exact shaping class; rest: weak | same | same | same, as a cone |
| switching | γ vs cost trade-off in `Σ_λλ` | ordinary regime comparison | same | same | n/a |
| plant vs cost | opaque; regime contrasts only | `(I−BB⁺)A` identified outright | same | `S` confounded with `Σ_x` | explicit |
| horizon assumption | none | required | required | required | none (stationary) |
| model class | any symplectic transition | Riccati-consistent only | + tied noise | + optimistic mode | nonparametric then projected |
| specification test | held-out vs LDS (**fails**) | same weakness | same | same | **LMI feasibility** |

---

## 5. What no parameterization fixes

* **The scale is a gauge.** Some normalization is always required. What differs
  is whether you *state* it (`R = I`, `tr S` fixed, `tr Qc[1] = n`) or discover
  after the fact that `q0` was a prior of infinite strength.
* **The terminal cost's unactuated block.** An exact shaping equivalence of
  dimension `(n − m)(n − m + 1)/2` (§2.4). Report `Bᵀ Q_term` and
  `Q_k − Q_term + Aᵀ Q_term A`, which are invariant; treat the rest as a
  convention unless a reference or anticipated disturbance moves against the
  passive dynamics. What lies outside the class is weakly identified by decay.
* **The latent gauge under a free `C`.** Intrinsic to partial observation; keep
  the Procrustes/linear audit exactly as it is.
* **A stationary agent.** If `K` is constant, only the inverse-optimality cone
  is identified, whatever the parameterization. Report the cone.
* **"Was this an LQR at all?"** Only C's feasibility gap tests this. A
  likelihood comparison against a linear alternative does not, and
  `README.md` #9 should stand under every parameterization here.

## 6. Recommended path

1. **Run D.3, the hybrid E-step.** One day, no new architecture, and it settles
   whether §1's diagnosis is right before anything is rewritten.
2. **Add D.1 and D.2** regardless of what follows — the MA(1) costate slack and
   the gauge constraint are small, local, and remove the two levers that
   currently dominate every table in `README.md`.
3. **Implement A as a first-class state model**, with the closed-form Riccati
   adjoint of §3.1. It reuses the existing Gaussian/Poisson emissions, the
   `SLDS` layer, `depends_on` grouping, the priors, the holdout scoring and the
   gauge audit; what is new is the Riccati sweep, its adjoint, and a
   gradient-based M-step. At `n = 12` the smoother's blocks shrink from `24×24`
   to `12×12` — `8×` — and the terminal normalizer is no longer needed; the
   Riccati and adjoint sweeps cost `O(Tn³)` each and are shared across trials.
4. **Use C to initialize A and to report the identified set**, and adopt its
   LMI feasibility gap as the specification test that replaces LQR-versus-LDS
   model comparison.
5. **Adopt A′** where behavioural variability is signal.
6. **Keep H** for what it is uniquely good at: a horizon-agnostic model over
   *any* symplectic transition, with a closed-form M-step, for exploratory fits
   where Riccati consistency is not yet something you want to assume.

## 7. Reproducing the numbers

`parameterization.jl` is self-contained — it imports nothing from `src/`, so its
verdicts are independent of the code under review. Sections:

| flag | what it measures |
|---|---|
| `--only=geometry` | reciprocal eigenvalue pairs, the symplectic defect, `M`-invariance of `graph(P)`, the `P`-metric contraction identity |
| `--only=gauge` | the exact `(S, Q)` scale invariance, and the same direction with `R` pinned |
| `--only=profile` | the exact marginal likelihood along the cost-scale ray |
| `--only=terminal` | the Riccati Jacobian `Ψ`, its finite-difference check, and the backward decay of terminal-cost information |
| `--only=reference` | where a reference perturbation lands, in time and in observed coordinates |
| `--only=shaping` | the exact shaping class of costs, constructed and checked (§2.4) |
| `--only=subspace` | the control subspace from the closed loop; the constant-feedback offset; the rank of the inverse-optimality equations (§3.4) |
| `--only=switching` | γ at the generating parameters for a matched delay-then-reach design |
| `--only=fit` | end-to-end recovery under the closed-loop parameterization, swept over the initial cost scale |

The mixed-coordinate numbers quoted for comparison are `README.md`'s and
`slds.jl`'s own measurements, not re-runs; where a configuration differs, the
text says so.
