# Implementation plan: inverse control for noisy, approximately optimal neural systems

This is the plan that follows from `parameterization.md` (which parameterization
recovers a controller) and `biological.md` (what can be claimed when the system
may not be optimizing). It is written for the target use case — latent neural
dynamics recorded as spikes or rates, with a fitted emission, many trials and
conditions, and a system that is at best approximately control-like — and it is
specific about where each piece lives in `src/`.

Every claim marked **verified** has a runnable check:

```console
$ julia --project=. docs/dev/lqr/parameterization.jl --only=shaping,subspace,sweeps,terminal,profile
$ julia --project=. docs/dev/lqr/parameterization.jl --only=fit          # ~15 min
$ julia --project=. docs/dev/lqr/biological.jl --only=residuals,gauge,identify,threshold
```

The exception is the package half of F10, which needs `StateSpaceDynamics`
and so cannot live in these deliberately package-free scripts; it is M0's first
test. Claims marked **derived** follow algebraically from verified ones but have
not been run; each has a test assigned to the milestone that first depends on
it.

---

## Summary

1. **Build one new state model, `ClosedLoopLQRStateModel`, and keep the existing
   one.** It parameterizes the stable manifold of the control problem — the
   closed loop `Φ_t = A − B K_t(θ)` on an `n`-dimensional latent — rather than
   the `2n`-dimensional Hamiltonian flow. It implements the
   `AbstractGaussianStateModel` contract in `types.jl`, so the block-tridiagonal
   smoother, the Gaussian **and Poisson** E-steps, every emission model and the
   held-out monitor apply unchanged. What is new is added by dispatch exactly as
   `LQRStateModel` does it: sufficient statistics, the M-step, `fit!`, and the
   `SLDS` state-M-step hook.
2. **Model suboptimality explicitly, as estimands with detection thresholds.**
   Policy noise (isotropic, free, or max-ent) and systematic gain deviations,
   each reported with the smallest effect the design could have detected.
3. **Fix every exact symmetry by an explicit convention — never leave it to
   where the optimizer started.** There are more than the harness has accounted for: besides the latent
   basis and the cost scale, an exact **shaping class** of dimension
   `(n − r)(n − r + 1)/2` in the cost (for a 12-D latent with 3 control channels,
   45 flat directions). Report quantities invariant to all of them.
4. **Fit by generalized EM with a numerical M-step** on per-time-to-go
   sufficient statistics, using an analytic reverse-mode pass through the
   Riccati and feedforward sweeps (**verified** to `≤ 2e-8` for every parameter
   block). The M-step's cost does not grow with the number of trials.
5. **Initialize from a structure-only fit**: estimate the control subspace from
   the time-variation of the closed loop (**verified** exact), then solve a
   linear inverse-optimality least squares for the cost modulo shaping
   (**verified** exact at the truth, but it amplifies gain noise ~20×).
6. **Arbitrate between hypotheses by held-out prediction**, never in-sample
   likelihood, along a ladder from a free LDS to exact optimality, and summarize
   with a gauge-free index of how much of the shared structure optimality
   explains.
7. **Model the delay-then-reach epoch as an exact change-point mixture** rather
   than a sticky HMM.

Milestones M0–M6 are in §10. M0–M3 deliver a usable model for Gaussian and
Poisson data; M4 delivers the approximate-optimality science layer; M5 the epoch
model; M6 is conditional on what M4 finds.

---

## 1. Premises

What the plan rests on. Each finding changes a design decision below.

| # | finding | status | consequence |
|---|---|---|---|
| F1 | The Hamiltonian forward map has `n` unstable modes (`ρ = 12` here); `graph(P_t)` is its stable manifold (`M`-invariant to `8.7e-10`); the closed loop contracts in the cost-to-go metric | verified | parameterize the manifold: latent `n`, stable forward roll, no terminal conditioning |
| F2 | `Σ_λλ` measures plant noise, not suboptimality: an optimal agent under plant noise has adjoint residual `6.66e3`, Riccati residual `0`; adding suboptimality moves them to `6.64e3` and `0.944` | verified | suboptimality must enter through the Riccati-graph relation, i.e. as policy noise or gain deviations |
| F3 | The cost scale is an exact gauge when `S` is fitted, and only **shallowly** identified when `R` is pinned — a 95 % profile interval of roughly ×0.65 to ×1.9 with everything else known | verified | pin `R`; report scale-invariant quantities unless a prior or behavioural variability supplies the scale |
| F4 | An exact **shaping class**: `Q_term → Q_term + M`, `Q_k → Q_k + M − AᵀMA` with `BᵀM = 0` changes no gain, no cost-to-go that `Bᵀ` can see, and no feedforward for targets at rest | verified to `1e-15` | a third gauge; report `BᵀQ_term` and `Q_k − Q_term + AᵀQ_termA` |
| F5 | The control subspace `range(B)` and its rank are recovered exactly from the time variation of the closed loop, with no optimality assumption | verified | estimate `rank(B)` as a result; use it to initialize |
| F6 | `A − BK_t = (A+BD) − B(K_t+D)`: a constant feedback offset mimics intrinsic dynamics. Optimality rejects it, but ~8× more weakly when the gains are near stationary | verified | identify the actuated part of `A` from uncontrolled epochs when available |
| F7 | The inverse-optimality least squares recovers the shaping invariants to `1e-11` at the truth; 1 % gain noise becomes 20 % invariant error | verified | an initializer that needs a precise structure fit, not an estimator |
| F8 | Suboptimality is detectable only if the emission spans `range(B)`, and here only for slack carrying `φ ≳ 0.15` of the innovation variance | verified | a detection-threshold tool, and design guidance |
| F9 | Reverse-mode gradient through the Riccati and feedforward sweeps matches central differences to `≤ 2e-8` for `A`, `B`, `R`, every `Q_k`, and the reference | verified | analytic M-step gradient; no AD dependency in `src/` |
| F10 | The package's feedforward is correct: `simulate_lqr` matches a direct QP solve of the tracking problem to `8.6e-14`. The scripts in this directory carried a sign error in theirs, now fixed and checked against the same QP | verified | use `b_t = −Q_t r + Φ_tᵀ b_{t+1}`; make the package-side QP check M0's first test |
| F11 | From cold starts, the corrected closed-loop fit recovers the reference (0.016–0.039, corr 1.00) and the cost's shape (corr 0.93–0.98), but ends 65–78× off in scale from three of four starts with the closed loop 6–10 % wrong; a warm start holds to 1 %. The likelihood separates the converged fit by 115–290 nats but does not rank unconverged ones by accuracy | verified | initialization (§6) and a scale source are load-bearing, not optional; select among restarts only once they have converged |

---

## 2. The target model

### 2.1 Generative model

For trial `i`, timestep `t = 1, …, T_i`, and design index `τ = τ(i, t)`:

```math
\begin{aligned}
x_1 &\sim \mathcal N(\mu_0, V_0),\\
u_t &= -\big(K_\tau + \Delta_\tau\big)\,x_t + k_\tau(v_i) + \eta_t,
      &\eta_t &\sim \mathcal N(0, \Xi_\tau),\\
x_{t+1} &= A x_t + B u_t + B_d\, d_{i,t} + \varepsilon_t,
      &\varepsilon_t &\sim \mathcal N(0, \Sigma_x),\\
y_{i,t} &\sim p\big(y \mid C x_t + D_y\, w_{i,t}\big)
      && \text{Gaussian, Poisson, or composite.}
\end{aligned}
```

`K_τ` and `k_τ` come from the finite-horizon sweeps for `(A, B, R, {Q_k},
schedule, r_i = G_{ref} v_i)`:

```math
\begin{aligned}
G_t &= R + B^\top P_{t+1} B, &
K_t &= G_t^{-1} B^\top P_{t+1} A, &
\Phi_t &= A - B K_t,\\
P_t &= Q_{k(t)} + K_t^\top R K_t + \Phi_t^\top P_{t+1}\Phi_t, &
P_T &= Q_{k(T)},\\
k_t &= -G_t^{-1} B^\top b_{t+1}, &
b_t &= -Q_{k(t)}\, r + \Phi_t^\top b_{t+1}, &
b_T &= -Q_{k(T)}\, r .
\end{aligned}
```

The state marginal the smoother sees is an ordinary time-varying linear-Gaussian
chain:

```math
x_{t+1} = \Phi^{\Delta}_\tau x_t + f_{i,t} + w_t,\qquad
\Phi^{\Delta}_\tau = A - B(K_\tau + \Delta_\tau),\qquad
f_{i,t} = B k_\tau(v_i) + B_d d_{i,t},\qquad
w_t \sim \mathcal N\!\big(0,\ \Omega_\tau\big),\quad
\Omega_\tau = \Sigma_x + B\,\Xi_\tau B^\top .
```

`R` is fixed to `I_r` (the gauge of F3); `B` is `n × r` with `r` the number of
effective control channels.

### 2.2 The causal convention, and why

`simulate_lqr` uses the Hamiltonian discretization, in which `u_t` responds to
`λ_{t+1} = P_{t+1}x_{t+1} + g_{t+1}` and therefore to the noise realized in the
same step; its innovation is `W_{t+1}(·)W_{t+1}ᵀ` with `W = (I + SP)⁻¹`. The model
above is causal — `u_t` depends on `x_t` — which is standard discrete-time LQG
and is what a sensorimotor delay forces on a biological controller. Noise-free,
the two coincide (`Φ_t = W_{t+1}A` in both); with noise they differ at `O(Δt)`.

The distinction does not disturb F2: in both conventions the adjoint residual is
`N_t` times the state residual with the same `N_t = −AᵀP_{t+1}(I + SP_{t+1})⁻¹`
(**derived**; the anticipatory case is verified). Keep the anticipatory
convention available as a harness generator so recovery can be checked against
both.

### 2.3 The suboptimality menu

Two kinds, which answer different scientific questions.

**Noisy optimality** — the gains are optimal, the execution is not:

| option | `Ξ_τ` | free parameters | reading |
|---|---|---|---|
| `:none` | `0` | — | optimal controller in a noisy plant |
| `:isotropic` | `ξ² I_r` | 1 | isotropic motor/command noise |
| `:free` | `Ξ ⪰ 0` | `r(r+1)/2` | anisotropic command noise |
| `:maxent` | `β⁻¹ G_τ⁻¹` | 1 | bounded rationality; `β` an inverse temperature |

**Systematic suboptimality** — the gains themselves depart from the Riccati
solution:

```math
\Delta_\tau = \sum_{j=1}^{J} w_j(\tau)\,\Delta^{(j)},
\qquad \sum_\tau w_j(\tau) = 0,
\qquad \text{penalty } \tfrac{\lambda_\Delta}{2}\sum_j \|\Delta^{(j)}\|_F^2 .
```

The zero-temporal-mean constraint is forced by F6: a constant gain offset is
indistinguishable from a change of `A`'s actuated rows, so it is assigned to `A`
by convention and `Δ` carries only departures from the Riccati *time course*.
`λ_Δ → ∞` recovers exact optimality; `λ_Δ → 0` a structure-only model. Choose it
by held-out likelihood.

F8 applies to both kinds: they load on `range(B)`, so they are visible only
through an emission that spans it.

### 2.4 Horizons and schedules

Aggregation, caching and identifiability all depend on what `τ` indexes.

* **End-anchored** (terminal cost at trial end, running cost before): `P`, `K`,
  `L` depend only on time-to-go, so **one sweep of length `max_i T_i` serves every
  trial**, whatever its length. This is the default and the fast path.
* **Onset-anchored** (a fixed-duration movement after a cue): index the
  controlled epoch by time since onset; with known onsets the sweep is again
  shared. With unknown onsets, see §8.
* **Horizon as a parameter.** The gains depend on the assumed horizon, and a
  biological controller's horizon is not given. Profile the likelihood over a
  small grid of horizons (a discrete parameter) and report the profile.
* **Stationary** (`T → ∞`, constant `K`): available, but it removes the time
  variation that identifies `range(B)` (F5) and weakens F6 eightfold. Warn when a
  fit's gains are near stationary over the observed window.

### 2.5 Inputs

* **References** `r_i = G_{ref} v_i`, with `v_i` a task input (target identity,
  reward). The feedforward is linear in `r`, `k_t = L_t r`, so one affine sweep
  per column of `G_ref` suffices and each trial's `f_{i,t}` is a matrix–vector
  product. Constant-within-trial references are the v1 case; piecewise-constant
  ones (a target jump) reduce to a few `L` operators.
* **Exogenous plant inputs** `B_d d_{i,t}` — sensory events, stimulation.
  Whether a disturbance is *anticipated* (enters the feedforward) or *unexpected*
  (does not) is a modelling choice with identifiability consequences (§7.6, F4).

### 2.6 Emissions

Everything in `src/lds/*_observations.jl` applies unchanged: Gaussian, Poisson,
and `CompositeObservationModel`. The composite is the most useful for this use
case: **anchor part of the latent in behaviour** — hand position and velocity
read through a fixed emission, neural activity through a fitted one. The
anchored block then carries physical units, which (a) removes most of the latent
gauge, (b) gives the task cost `Q` physical meaning on that block, and (c) makes
F8's requirement — an emission spanning `range(B)` — something that can be
checked rather than hoped for.

---

## 3. Symmetries, conventions and invariants

### 3.1 Every exact symmetry

With `R` pinned, the likelihood is exactly invariant under:

| symmetry | action | dimension | fixed by |
|---|---|---|---|
| latent basis (free emission) | `x → Tx`; `A → TAT⁻¹`, `B → TB`, `Q → T⁻ᵀQT⁻¹`, `Σ_x → TΣ_xTᵀ`, `C → CT⁻¹`, `K → KT⁻¹`, `r → Tr` | `n²` | canonical gauge, §3.2 |
| control basis | `u → Mu`, `B → BM⁻¹`, `Ξ → MΞMᵀ`, `K → MK` with `M ∈ O(r)` | `r(r−1)/2` | canonical gauge |
| shaping (F4) | `Q_term → Q_term + M`, `Q_k → Q_k + M − AᵀMA`, `BᵀM = 0` | `(n−r)(n−r+1)/2` | report invariants; weak gauge penalty in the fit |

and approximately invariant (weakly identified) along:

| direction | why | mitigation |
|---|---|---|
| cost scale | F3: the endpoint does not move with scale; only the transient shape does | report scale-free quantities; scale from max-ent or a prior |
| actuated rows of `A` vs constant feedback | F6 | uncontrolled epochs; zero-mean `Δ` convention |
| inverse-optimality cone (stationary gains) | many costs make one stationary `K` optimal | finite horizons; several regimes |
| terminal cost outside the shaping class | decay through `Ψ_{t,s}` (§2.4 of `parameterization.md`) | report as weakly identified |

With a free emission the latent scale is part of `GL(n)`, and rescaling the
latent by `a` sends `(B, Q) → (aB, a⁻²Q)` — so the **absolute** size of `Q` has
no physical meaning in a neural-only fit, whatever is pinned. Only an anchored
emission (§2.6) or a stated convention gives it units.

### 3.2 A canonical gauge (free emission)

**Derived**, WLOG for `Σ_x ≻ 0`. Choose `T` and `M` so that

1. `B = [I_r; 0]` — control-whitened coordinates, so `S = diag(I_r, 0)`;
2. `Σ_x` is block-diagonal across the actuated / unactuated split (fixes the
   `r × (n−r)` shear, `T₁₂ = −Σ₁₂Σ₂₂⁻¹`);
3. the unactuated block of `Σ_x` is `I_{n−r}` (fixes `T₂₂ = Σ₂₂^{-1/2}`).

The residual group is `O(r) × O(n−r)`, compact and handled by orthogonal
Procrustes. When the unactuated plant noise is small, step 3 is ill-conditioned;
whiten by the unactuated block of the stationary state covariance instead.

Two modes follow. **Neural-only**: fit directly in this parameterization — `B`
fixed at `[I_r; 0]`, `Σ_x = blockdiag(Σ_c, I)` — so the flat `GL(n)` directions
never enter the optimizer. **Anchored**: keep physical coordinates on the
anchored block and apply the canonical gauge to the free block only.

Pinning `B` presumes `r` is known. Estimate it first (§6, step 2) and treat it as
a model-selection variable.

### 3.3 What to report

Every entry is invariant under all three exact symmetries.

| quantity | formula | reading |
|---|---|---|
| closed-loop spectra | `eig(Φ_τ)` for each `τ` | how the controlled dynamics evolve over the trial |
| loop-gain spectrum | `eig(S P_τ)` | cost-to-go in control units; also invariant to the cost scale |
| control rank | `r̂` and its held-out profile | number of effective control channels |
| terminal cost, actuated part | `Bᵀ Q_term` (in the canonical gauge) | the task-relevant terminal cost the data can see |
| shaped running cost | `Q_k − Q_term + Aᵀ Q_term A` | the running cost modulo shaping |
| reference geometry | `G_refᵀ G_ref` (Gram of targets) | target distances and angles without a basis |
| slack share and threshold | `φ̂`, and the smallest detectable `φ` for the design (§7.3) | how suboptimal, and whether that could have been seen |
| optimality index | `𝒪` (§7.2) | how much of the shared structure optimality explains |
| predictions | held-out likelihood, co-smoothing | everything else is judged against these |

Raw `Q`, `A` and `B` are reported only in the canonical gauge, labelled as such.

---

## 4. Package architecture

### 4.1 The contract

`types.jl` states what a state model owes the rest of the package:

> `_state_latent_dim` / `_state_ux_dim`; `state_loglikelihood!`,
> `_state_gradient!`, `_state_hessian_blocks!`; `Q_state!`,
> `_state_prior_logdensity`; `_state_mstep!`. Everything else — the emission
> kernels, the block-tridiagonal solver, the workspaces, and the EM drivers — is
> shared.

The Poisson driver in `fit_PLDS.jl` is written against
`S <: AbstractGaussianStateModel{T}`, so a new state model that honours the
contract gets the Laplace E-step for free. The existing LQR model already shows
the pattern for everything the contract does not cover: it overrides
`_initialize_td_sufficient_statistics`, `mstep!` (one method for
`QuadraticEmission`, one for `NonQuadraticEmission`) and `fit!` by dispatch on
`S <: LQRStateModel`, and prepares a per-dataset cache in `_prepare_lqr!`.

### 4.2 Files and types

| file | contents |
|---|---|
| `src/numerics/riccati.jl` | `riccati_gain!`, `affine_sweep!`, `riccati_adjoint!` on preallocated buffers — shared with the existing model |
| `src/lds/closed_loop_types.jl` | `ClosedLoopLQRStateModel`, `ClosedLoopFitFlags`, `ClosedLoopCache`, constructors, `refresh!` |
| `src/lds/closed_loop_latents.jl` | the eight contract functions, mirroring `lqr_latents.jl` with a per-`τ` lookup |
| `src/lds/closed_loop_mstep.jl` | `ClosedLoopSufficientStatistics`, the aggregator, the GEM objective and gradient |
| `src/lds/fit_closed_loop.jl` | `_prepare_closed_loop!`, `mstep!` and `fit!` methods |
| `src/lds/closed_loop_report.jl` | canonical gauge, invariants, `φ̂`, the detection threshold, `𝒪` |
| `src/lds/changepoint.jl` | the epoch model (M5) |

The model's fields: `A`, `B`, `Qc::Vector` (one per regime, as now), `schedule`
and a horizon spec, `Gref`, `Bd`, `Σx`, a policy-noise spec (`:none |
:isotropic | :free | :maxent` plus its parameters), an optional gain-deviation
basis and penalty, `x0`, `P0`, fit flags, priors, `depends_on` / `variants` as on
every other state model, and a `:free` mode (a plain time-invariant `A` with no
control structure) so the type can serve as the non-controlled member of an
`SLDS` or a change-point model with the same latent dimension.

### 4.3 The cache

Keyed by **design** — a (horizon, schedule) pair — and within a design by `τ`:
`K_τ`, `P_τ`, the Cholesky factor of `G_τ`, `Φ_τ`, the feedforward operators
`L_τ` (`r × n`, one affine sweep per column of `G_ref`), `Ω_τ` and its Cholesky
factor and log-determinant. `_prepare_closed_loop!` collapses the dataset's
trials onto the fewest designs (end-anchored schedules collapse to one), and
`refresh!` recomputes on every parameter change at `O(designs × T × n³)`,
independent of the number of trials.

### 4.4 Sufficient statistics

The key change from `LQRSufficientStatistics`. The mixed-coordinate transition
is constant within a regime, so that type aggregates per regime `k`. The closed
loop changes at every `τ`, so aggregate per `τ` (per design), keeping the same
regressor layout `z = [x_t; 1; v]`:

```math
S^{zz}_\tau = \sum_{(i,t):\,\tau(i,t)=\tau} \mathbb E[z z^\top], \qquad
S^{zy}_\tau = \sum \mathbb E[z\, x_{t+1}^\top], \qquad
S^{yy}_\tau = \sum \mathbb E[x_{t+1}x_{t+1}^\top], \qquad
n_\tau = \#\{(i,t)\}.
```

Storage is `O(T_max (n + 1 + p)²)`; for `n = 12`, eight targets and `T = 100`,
about 44 000 numbers. The efficient accumulation is a BLAS-3 update **across
trials** at each `τ` (gather `x_{i,T_i−τ}` for all live trials into one `n × N_τ`
block), rather than across time within a run as `lqr_mstep.jl` does now. The
initial-state and emission halves stay in the shared `base`, exactly as in
`LQRSufficientStatistics`.

### 4.5 E-step

Gaussian emissions: the block-tridiagonal smoother, with `Φ_τ` and `Ω_τ` looked up
per step. Poisson: the existing Laplace E-step, unchanged. No terminal factor, no
normalizer, no `(dT)²` sampler: `rand` is a stable forward roll.

### 4.6 `fit!`

Mirror `fit_LQR.jl`: `_prepare_closed_loop!(lds, data.tsteps)`, then the EM loop
with the existing held-out monitor, `FitTrace`, `early_stopping` and
`restore_best`. Replace the absolute convergence test
`abs(elbos[iter] − elbos[iter−1]) < tol` with a relative one here from the start.

---

## 5. The M-step

### 5.1 Objective

With `H_τ = [Φ^Δ_τ, c, F_τ]` acting on `z = [x; 1; v]`, where `F_τ = B L_τ G_ref`
and `c` is any constant drift,

```math
\mathcal Q(\theta) = \sum_\tau\Big[-\tfrac{n_\tau}{2}\log\lvert 2\pi\Omega_\tau\rvert
  - \tfrac12\operatorname{tr}\!\big(\Omega_\tau^{-1}\mathcal E_\tau\big)\Big]
  + \mathcal Q_{\text{init}} + \log p(\theta),
\qquad
\mathcal E_\tau = S^{yy}_\tau - H_\tau S^{zy}_\tau - S^{zy\top}_\tau H_\tau^\top
                + H_\tau S^{zz}_\tau H_\tau^\top .
```

Its partial derivatives with respect to the per-`τ` quantities are closed form:

```math
\frac{\partial\mathcal Q}{\partial H_\tau} = \Omega_\tau^{-1}\big(S^{zy\top}_\tau - H_\tau S^{zz}_\tau\big),
\qquad
\frac{\partial\mathcal Q}{\partial \Omega_\tau}
  = \tfrac12\,\Omega_\tau^{-1}\big(\mathcal E_\tau - n_\tau\Omega_\tau\big)\Omega_\tau^{-1}.
```

Slices of `∂𝒬/∂H_τ` are `Φ̄_τ`, `c̄` and `F̄_τ`; `F̄_τ` gives `L̄_τ = BᵀF̄_τG_refᵀ`,
`B̄ += F̄_τ(L_τG_ref)ᵀ`, `Ḡ_ref += (BL_τ)ᵀF̄_τ`; and `L̄_τ` is the direct
feedforward adjoint `k̄_τ`, one column per column of `G_ref`.

### 5.2 Reverse pass through the sweeps

Given `Φ̄_t` and `k̄_t` for `t = 1, …, T−1`, accumulate `Ā, B̄, R̄, Q̄_k, r̄` by one
forward-in-time pass (**verified**, `≤ 2e-8` against central differences for
every block, `--only=sweeps`; `sym(X) = (X + Xᵀ)/2`):

```math
\begin{aligned}
&\text{1. through } b_t:&& \bar Q_{k(t)} \mathrel{-}= \operatorname{sym}(\bar b_t r^\top),\;
  \bar r \mathrel{-}= Q_{k(t)}\bar b_t,\;
  \bar\Phi_t \mathrel{+}= b_{t+1}\bar b_t^\top,\;
  \bar b_{t+1} \mathrel{+}= \Phi_t\bar b_t\\
&\text{2. through } k_t:&& z = G_t^{-1}\bar k_t,\;
  \bar b_{t+1} \mathrel{-}= Bz,\; \bar B \mathrel{-}= b_{t+1}z^\top,\;
  \bar G_t = -\operatorname{sym}(z k_t^\top)\\
&\text{3. through } P_t \text{ (envelope)}:&& \bar Q_{k(t)} \mathrel{+}= \bar P_t,\;
  \bar R \mathrel{+}= K_t\bar P_tK_t^\top,\;
  \bar A \mathrel{+}= 2P_{t+1}\Phi_t\bar P_t,\;
  \bar B \mathrel{-}= 2P_{t+1}\Phi_t\bar P_tK_t^\top,\;
  \bar P_{t+1} \mathrel{+}= \Phi_t\bar P_t\Phi_t^\top\\
&\text{4. through } \Phi_t, K_t:&& \bar A \mathrel{+}= \bar\Phi_t,\;
  \bar B \mathrel{-}= \bar\Phi_tK_t^\top,\;
  Z = -G_t^{-1}B^\top\bar\Phi_t,\\
&&& \bar B \mathrel{+}= P_{t+1}\Phi_tZ^\top - P_{t+1}BZK_t^\top,\;
  \bar P_{t+1} \mathrel{+}= \operatorname{sym}(BZ\Phi_t^\top),\;
  \bar A \mathrel{+}= P_{t+1}BZ,\;
  \bar R \mathrel{-}= \operatorname{sym}(ZK_t^\top)\\
&\text{5. through } G_t:&& \bar R \mathrel{+}= \bar G_t,\;
  \bar B \mathrel{+}= 2P_{t+1}B\bar G_t,\;
  \bar P_{t+1} \mathrel{+}= B\bar G_tB^\top
\end{aligned}
```

and at the end `Q̄_{k(T)} += P̄_T − sym(b̄_T rᵀ)`, `r̄ −= Q_{k(T)}b̄_T`. Step 3 is
where optimality pays: because `K_t` minimizes, every `dK_t` term in `dP_t`
cancels, leaving the linear recursion
`dP_t = dQ_t + K_tᵀdRK_t + (dA − dBK_t)ᵀP_{t+1}Φ_t + Φ_tᵀP_{t+1}(dA − dBK_t) + Φ_tᵀdP_{t+1}Φ_t`.
No implicit solve and no AD are needed; a gradient costs one extra `O(Tn³)` sweep.

The max-ent noise adds one path (**derived**, to be verified in M4): with
`V_t = G_t⁻¹` and `Ω_t = Σ_x + β⁻¹BV_tBᵀ`, given `Ω̄_t`:
`Σ̄_x += Ω̄_t`, `β̄ −= β⁻²⟨Ω̄_t, BV_tBᵀ⟩`, `B̄ += 2β⁻¹Ω̄_tBV_t`, and
`Ḡ_t −= V_t(β⁻¹BᵀΩ̄_tB)V_t` into step 5.

### 5.3 Parameterization

| block | coordinates | notes |
|---|---|---|
| `Q_k` | `exp(s_k) · n · L̃_kL̃_kᵀ / tr(L̃_kL̃_kᵀ)` | log-scale separated from shape, so the shallow scale direction (F3) is one coordinate rather than a diagonal through all of them |
| `A` | unconstrained; flags for "fixed", "unactuated rows only", "tied to an uncontrolled epoch" | F6 |
| `B` | fixed at `[I_r; 0]` in the neural-only canonical gauge; free `n × r` otherwise | §3.2 |
| `Σ_x` | log-Cholesky; `blockdiag(Σ_c, I)` in the canonical gauge | |
| `Ξ` | per §2.3 | |
| `Δ^{(j)}` | unconstrained, ridge `λ_Δ` | zero temporal mean |
| `G_ref` | unconstrained, with the existing `Gref_cols` narrowing | |
| shaping | penalty `(ε/2)‖Nᵀ Q_term N‖²_F`, `ε` small and stated | picks one member of the class; invariants do not depend on `ε` |

Optimize with L-BFGS and backtracking, rejecting any point where a `G_t`, `Ω_τ`
or `Σ_x` factorization fails (the existing rejectable-step machinery). Accept the
M-step only if `𝒬` increases, which keeps the outer loop a generalized EM and
monotone.

### 5.4 Cost

Per `𝒬` evaluation: `O(designs × T × n³)` for the sweeps and adjoints, plus
`O(T × (n + 1 + p)² n)` for the contractions — neither depends on the number of
trials. The E-step dominates, as it does now.

---

## 6. Initialization

The measured failure mode of a cold start is the scale (F3): with the corrected
feedforward, fits started far from the right scale drift along the shallow
direction. The pipeline below gets close enough to the right scale, cost shape
and control subspace that GEM refines rather than searches.

1. **Structure-only fit (level L1 of §7.1).** The same model type with the
   optimality constraint removed: `Φ_τ = Σ_j w_j(τ) Φ^{(j)}` on a small temporal
   basis (a handful of B-splines in `τ`), shared across conditions, with a free
   feedforward per condition. It is an ordinary time-varying LDS, fitted by the
   same E-step.
2. **Control subspace and rank.** `range(B) = span{Φ_τ − Φ_σ}` (F5). Take the
   leading left singular vectors of `[Φ̂_τ − Φ̂_{τ₀}]_τ`; choose `r` by the
   singular-value gap, confirmed by held-out likelihood. This is the first
   scientific output of the pipeline.
3. **Intrinsic dynamics.** `(I − BB⁺)Â = (I − BB⁺)Φ̂_τ` is read directly (F5). For
   the actuated rows, use an uncontrolled epoch if the design has one (a delay
   period modelled with `u = 0`), otherwise profile over the constant offset of F6
   using step 4's residual.
4. **Cost, modulo shaping.** With `K̂_τ = B⁺(Â − Φ̂_τ)` and `P_t` affine in `Q`
   given the gains, solve the linear least squares
   `R K̂_t = Bᵀ P_{t+1}(Q) Φ̂_t` for `{Q_k}` (**verified**: exact invariants at the
   truth), add the shaping gauge penalty, and project each `Q_k` onto the PSD
   cone. Its residual is also the feasibility statistic of §7.5.
5. **Noise** from the L1 residuals; policy noise initialized small.
6. **GEM** from there. Because step 4 amplifies gain error ~20× (F7), the
   pipeline's value depends on step 1's precision: run a few restarts that
   perturb step 4's solution, run each to convergence under the relative test,
   and only then select by **held-out** likelihood. F11 is the reason for the
   order: the likelihood identifies a converged fit reliably and says nothing
   useful about which unconverged fit is closest.

---

## 7. The approximate-optimality layer

This is what turns a controller fit into a statement about a biological system.

### 7.1 The ladder

Every level is a state model on `x` alone, fitted by the same machinery.

| level | transition | hypothesis |
|---|---|---|
| L0 | time-invariant `A`, free per condition feedforward | no control structure |
| L1 | `Φ_τ` on a temporal basis, **shared** across conditions; free feedforward | shared structure, no optimality |
| L2 | `Φ_τ = A − BK_τ(θ)`, feedforward from the affine sweep | exact feedback optimality |
| L3 | L2 + policy noise `Ξ` (`:isotropic`, `:free`, `:maxent`) | noisy optimality |
| L3Δ | L3 + zero-mean gain deviations `Δ_τ`, ridge `λ_Δ` | systematic, time-varying suboptimality |
| L4 | hard adjoint + terminal equalities on `(x, λ)` | open-loop planning (deferred, §10 M6) |

L3Δ is a continuum between L1 (`λ_Δ → 0`) and L3 (`λ_Δ → ∞`), so the ladder is
not only a set of discrete models.

### 7.2 Arbitration

**Held-out, never in-sample.** For biological data every level is out of class,
and in-sample likelihood rewards whatever hedges most (`README.md` #7). Two
held-out scores, both already expressible with the existing held-out monitor:

* held-out **trials**, scored by the marginal likelihood (Gaussian) or its Laplace
  approximation (Poisson);
* **co-smoothing**: hold out a subset of neurons on held-out trials and score the
  prediction of their activity from the rest — the Neural Latents Benchmark
  convention, and the one that most directly tests the latent dynamics.

**The optimality index.** With `ℓ_k` the held-out log likelihood of level `k`,

```math
\mathcal O = \frac{\ell_{L2\text{ or }L3} - \ell_{L0}}{\ell_{L1} - \ell_{L0}} .
```

The share of what shared structure buys that optimality accounts for: near 1,
the Riccati constraint explains the structure; near 0, it does not. Being built
from held-out likelihoods, it is invariant to every gauge in §3. Calibrate it by
parametric bootstrap — refit all levels to data simulated from the fitted L1 and
from the fitted L3 — so a value can be read against its null and its best case.

### 7.3 Suboptimality estimands, with thresholds

Report, for the policy-noise options, the share of innovation variance the slack
carries,

```math
\hat\varphi = \frac{\sum_\tau \operatorname{tr}\big(B\hat\Xi_\tau B^\top\big)}
                   {\sum_\tau \operatorname{tr}\big(\hat\Omega_\tau\big)},
```

and alongside it the **detection threshold** for the actual design: simulate from
the fitted model at a ladder of `φ`, refit the noise parameters only (cheap — the
structure is held fixed), and report the smallest `φ` recovered within a stated
tolerance. `biological.jl --only=threshold` is the prototype; the package
function should take a fitted model and a design and return the curve. A small
`φ̂` means "near-optimal" only if it sits above the threshold.

For gain deviations, report `λ̂_Δ`, the held-out gain of L3Δ over L3, and the
deviations themselves in the canonical gauge.

### 7.4 The minimum-intervention cross-constraint

An optimal feedback controller corrects only task-relevant deviations, so
trial-to-trial variability should be small in the range of the cost and large in
its null space; the closed-loop covariance follows
`Σ_{t+1} = Φ_t Σ_t Φ_tᵀ + Ω_t`. On its own this is not discriminating — L1 fits any
covariance. It becomes discriminating as a **cross-constraint**: the one cost
must simultaneously predict the mean trajectories (through the feedforward), the
gains (through the closed loop) and the anisotropy of the variability. Report the
held-out prediction of the empirical per-`τ` covariance under L3 against L1; a
controller that matches L1 here with far fewer parameters is evidence for
optimality that a free model cannot counterfeit.

Remember F4 when reading it: the gains, and so the variability geometry, are
blind to the shaping class. The test constrains the cost modulo shaping, which is
all the data can constrain.

### 7.5 Inverse-optimality feasibility

Fit L1, then ask whether any quadratic cost makes its gains optimal: the residual
of §6 step 4, normalized, with the PSD violation of its projection. Calibrate by
bootstrap from L3 fits. This is the test `README.md` #9 wanted and a likelihood
comparison against an LDS cannot supply — it asks whether the *structure-only*
closed loop lies in the optimality-consistent set, rather than which of two
misspecified models fits better. Its conditioning (F7) means it needs a precise
L1 fit; report it with its bootstrap spread, never as a bare p-value.

### 7.6 Feedback control or open-loop planning

`biological.md` §1 shows these are different hypotheses about which optimality
relation holds. F4 sharpens what can test them:

* **Unexpected perturbations** probe the feedback gains `K_t`. A feedback
  controller corrects; an open-loop planner accumulates. This distinguishes the
  two, but is blind to the shaping class.
* **Anticipated disturbances and moving references** probe the feedforward. Only
  these can identify the unactuated block of the cost (F4), and only when
  `Nᵀ(A r + d_t − r) ≠ 0` — a target specified with a velocity changed the
  feedforward by 8.8 % under a shaping move that left static targets untouched.

---

## 8. Epoch structure: an exact change-point mixture

A delay-then-reach trial has **one** switch. A sticky HMM spends its
flexibility on paths the task never produces and forces the variational
machinery `README.md` documents. A change-point mixture is exact:

```math
p(y_{1:T}\mid\theta) = \sum_{s \in \mathcal W} \pi(s)\; p(y_{1:T} \mid s, \theta),
```

where for onset `s` the latent evolves under the pre-onset dynamics (a `:free`
member, or the plant `A` with `u = 0` — the **tied-plant** option) for `t < s` and
under the closed loop afterwards. Each component is a time-varying linear-Gaussian
chain, so:

* **Gaussian emissions**: an exact Kalman marginal per component; filter the
  shared pre-onset prefix once and branch, `O(|𝒲| T n³)` per trial worst case.
* **Poisson**: a Laplace approximation per component.
* **E-step**: onset posterior `∝ π(s) p(y | s)`; sufficient statistics are the
  responsibility-weighted component statistics, so the §5 M-step is unchanged.
  `π(s)` has a closed-form update.
* **Horizons**: end-anchored gains are shared across components (time-to-go does
  not depend on `s`); onset-anchored gains are shared across trials with the same
  post-onset duration.

The tied-plant option matters beyond segmentation: if the delay epoch is the same
plant with the controller off, it identifies the actuated rows of `A` directly,
which is the cleanest answer to F6. Compare tied against untied by held-out
likelihood; the difference is itself a finding about whether preparatory dynamics
are the uncontrolled plant.

Keep the `SLDS` for designs that genuinely switch several times; for one switch
per trial, the mixture is exact, cheaper to reason about, and returns a
calibrated onset posterior rather than a plug-in `γ`.

---

## 9. Validation

### 9.1 Unit tests (`test/LinearDynamicalSystems/ClosedLoopLQR.jl`)

* **Sweeps**: feedforward against a direct QP solve of the tracking problem (the
  check that caught F10); `riccati_adjoint!` against `ForwardDiff` (already a test
  dependency) and central differences, every block, several schedules.
* **Kernels**: `state_loglikelihood!`, `_state_gradient!`,
  `_state_hessian_blocks!` against a dense `nT × nT` Gaussian, mirroring the
  existing LQR latent tests.
* **Sufficient statistics**: the per-`τ` BLAS-3 aggregator against a brute-force
  loop; ragged trial lengths; time-to-go sharing gives the same likelihood as
  per-trial sweeps to roundoff.
* **Symmetries as tests**: a random `T ∈ GL(n)` with a fitted emission leaves the
  likelihood and every §3.3 invariant unchanged; a random shaping move leaves the
  likelihood unchanged to `1e-12`; canonicalization is idempotent.
* **GEM**: `𝒬` non-decreasing in every M-step; the ELBO non-decreasing across
  iterations (Gaussian).
* **`rand`**: bounded over long horizons (the property the Hamiltonian form lacks).
* **JET**: the objective and kernels stay on typed paths (the fit section of
  `parameterization.jl` logged 3.5e9 allocations before its closures were hoisted).

### 9.2 Recovery harness

Extend `docs/dev/lqr/` with generators that span the hypothesis space, and fit the
whole ladder to each:

| generator | tests |
|---|---|
| G1 the model itself | self-consistency |
| G2 causal LQG + policy noise | the target case |
| G3 G2 + time-varying gain deviations | L3Δ recovers `Δ`; `𝒪` falls below 1 |
| G4 free LDS | `𝒪 ≈ 0`; feasibility test rejects |
| G5 anticipatory (Hamiltonian) convention | robustness to §2.2 |
| G6 open-loop planner | L3 misfits in the predicted way |
| G7 delayed-feedback controller | robustness to the most likely biological misspecification |

Score only §3.3 invariants, with bootstrap intervals, and report cold-start and
warm-start fits separately.

### 9.3 Acceptance criteria

A milestone is done when its tests pass and, on its harness rows, the invariants
fall within twice their bootstrap standard error, with those numbers written into
the relevant `docs/dev/lqr/*.md` by the script that produced them.

---

## 10. Milestones

Sizes are rough (S ≈ a week, M ≈ two to three, L ≈ a month of focused work) and
are there to order the work, not to promise dates.

### M0 — Groundwork *(S)*

* `src/numerics/riccati.jl`: `riccati_gain!`, `affine_sweep!`,
  `riccati_adjoint!` on preallocated buffers, with the QP and `ForwardDiff`
  tests of §9.1. First test: `simulate_lqr` (noiseless) against a direct QP
  solve of the tracking problem, which it matches to `8.6e-14` today — a guard
  on the feedforward that the recovery scripts once got wrong.
* The open `README.md` findings, cheap and independent of everything else:
  a relative convergence test in `fit_LDS.jl` and `fit_PLDS.jl` (#3);
  per-state sufficient-statistic allocation in `fit_SLDS.jl` (#1); refuse
  multi-transition schedules on `SLDS` members in `_extract_state_params` (#2).
* `LQRStateModel` documentation: `Σ_λλ` does not measure suboptimality (F2), and
  its `Qc` carries the shaping class (F4) exactly as the new model's will.
* Optionally, the `tr S` gauge constraint for the existing model
  (`parameterization.md` §3.5 D.2).

**Done when** the sweep kernels pass their tests and the three findings are
closed.

### M1 — The closed-loop model, Gaussian, known `B` *(M)*

Types, cache, kernels, per-`τ` statistics, the GEM M-step over `A`, `{Q_k}`,
`G_ref`, `Σ_x`; policy noise `:none` and `:isotropic`; end-anchored schedules;
constant references; a stable `rand`.

**Done when** §9.1 passes; a warm start at the truth is stationary to `1e-3` nats
per step; and on generators G1–G2 the §3.3 invariants fall within two bootstrap
standard errors.

### M2 — Neural-data features *(M)*

Poisson through the inherited Laplace E-step; composite emissions with a
behaviour-anchored block; free emission with the canonical gauge of §3.2;
estimated `B` of rank `r`; `depends_on` grouping; ragged trial lengths; the
reporting module of §3.3.

**Done when** Poisson recovery on G2 matches the Gaussian rows within bootstrap
error; the `GL(n)` and shaping invariance tests pass; and a smoulder-scale
problem (`n = 12`, 150 channels, 1000 trials) fits inside a stated time budget.

### M3 — Initialization *(M)*

The structure-only mode (L1), subspace and rank, `A`, the least squares modulo
shaping, and restarts ranked by held-out likelihood.

**Done when** `r` is recovered on G2 across seeds, and fits started from the
pipeline reach the warm-start invariants in at least 80 % of seeds — against
the cold-start baseline in `parameterization.md` §2.5, where they do not.

### M4 — Approximate optimality *(L)*

`:free` and `:maxent` policy noise (verify the `Ω` adjoint of §5.2), gain
deviations (L3Δ), the detection-threshold function, the ladder with held-out and
co-smoothing scores, `𝒪` with bootstrap calibration, the feasibility test, and
the minimum-intervention report.

**Done when** G3 and G4 move `𝒪` and the feasibility statistic in the predicted
directions with calibrated intervals, and the threshold function reproduces
`biological.md` §4.

### M5 — Change-point epochs *(M)*

The mixture E-step, the tied-plant option, the onset posterior.

**Done when** onset posteriors are calibrated (nominal coverage on simulated
onsets), segmentation is at least as good as the `SLDS` on the `slds.jl`
configuration, and the tied-plant model recovers the actuated rows of `A` on a
generator where the delay epoch is the uncontrolled plant.

### M6 — Conditional

Each item has a trigger; none is started without it.

| item | trigger |
|---|---|
| hard-adjoint / open-loop backend (the equality-constrained KKT proposal) | L3 residuals show anticipatory rather than corrective structure, or perturbation data favour accumulation over correction |
| feedback delay by state augmentation | G7 shows L3 materially biased at realistic delays |
| an LQG agent with an internal state estimate (latent `(x, x̂)`) | the population appears to encode a belief rather than the state |
| signal-dependent noise (`Ξ ∝ diag(u²)`) | variability grows with command magnitude; needs a moment-matched E-step |
| an explicit action channel (EMG or kinematics as `u`) | such data exist — it separates `B` from `R` and pins `r` |

---

## 11. Risks

| risk | where it bites | mitigation |
|---|---|---|
| the cost scale is shallow (F3) | cold starts drift along it; raw `Q` is unreliable | log-scale coordinate; scale from max-ent or a stated prior; report scale-free |
| the shaping class (F4) | the unactuated terminal cost is a convention | report invariants; design references that move against the passive dynamics |
| constant feedback vs intrinsic dynamics (F6) | "intrinsic dynamics" claims | uncontrolled epochs; tied-plant option; zero-mean `Δ` |
| slack below threshold (F8) | "near-optimal" claims | detection threshold reported with every `φ̂`; emissions that span `range(B)` |
| non-convex M-step | local optima | initialization pipeline; restarts ranked by held-out likelihood |
| everything is out of class | model selection | the ladder and held-out arbitration only; never in-sample |
| horizon misspecification | gains, and so the cost | profile the horizon; compare anchorings |
| feedback delays | gains fitted without delay are biased | G7 in the harness; augmentation in M6 |
| ill-conditioned inverse least squares (F7) | the initializer and the feasibility test | precise L1 fits; bootstrap spread, not bare statistics |

---

## 12. Design guidance: what data identify what

For planning experiments as much as for reading fits.

| to identify | the data need | not enough |
|---|---|---|
| number and directions of control channels | gains that vary over the trial: a finite horizon, or several cost regimes | stationary behaviour |
| unactuated part of the intrinsic dynamics | any closed-loop data | — |
| actuated part of the intrinsic dynamics | an uncontrolled epoch, or inputs that bypass the controller | optimality alone, when the gains are near stationary (weak) |
| the reference geometry | several targets | one target, which is confounded with the affine drift |
| the cost, modulo shaping and scale | several targets and conditions; precise gains | — |
| the unactuated terminal cost | references or anticipated disturbances that the passive dynamics cannot follow | observing more coordinates; shorter horizons; unexpected perturbations |
| the cost scale | behavioural variability under max-ent, or an external calibration | the feedback structure alone (shallow) |
| suboptimality | an emission spanning `range(B)`, and slack above threshold | position-only readouts of a force-controlled plant |
| feedback versus planning | unexpected perturbations | unperturbed trajectories |

---

## 13. From the equality-constrained KKT proposal

Take now, independent of backend: the measure convention (declare `x_{1:T}` the
free coordinate when any constraint depends on `θ`); the three-way terminal
taxonomy (KKT boundary, soft boundary, outcome selection — the last is a
selection effect that analyzing only successful trials creates, and the
closed-loop model does not model it); the restricted Laplace determinant; LDLᵀ
rather than Cholesky on indefinite KKT systems; the explicit action channel; the
validation plan; the decision criteria.

Defer: the constrained message pass, until M6's trigger. It imposes the adjoint
recursion, which a feedback controller breaks (F2); the relation that survives
needs only the Kalman machinery above. If it is built, build it soft-first with a
fitted slack and use the hard constraint as the limiting test.

---

## Appendix A — the shaping class

**Claim.** Let `N` span `range(B)^⊥` and `M = N X Nᵀ` for any symmetric `X`. Replace
`Q_term → Q_term + M` and every running `Q_k → Q_k + M − AᵀMA`. Then `P_t → P_t + M`
for every `t`, every `K_t` is unchanged, and every feedforward `k_t` is unchanged
for references with `Nᵀ(Ar − r) = 0`.

**Proof.** `NᵀB = 0` gives `BᵀM = 0` and `NᵀΦ_t = NᵀA`, so `Φ_tᵀMΦ_t = AᵀMA` for
every `t`. Induct backwards: `P_T + M` is the new terminal value; if `P_{t+1} + M`
is the new cost-to-go then `G_t` and `K_t` are unchanged because `BᵀM = 0`, so
`Φ_t` is unchanged, and
`P'_t = Q_k + M − AᵀMA + K_tᵀRK_t + Φ_tᵀ(P_{t+1} + M)Φ_t = P_t + M`.
For the feedforward, suppose `b'_{t+1} = b_{t+1} − Mr`; then
`b'_t = −(Q_k + M − AᵀMA)r + Φ_tᵀ(b_{t+1} − Mr) = b_t − Mr + AᵀM(Ar − r)`, which
differs from `b_t − Mr` by `AᵀN X Nᵀ(Ar − r)` — zero for every `X` exactly when
`Nᵀ(Ar − r) = 0`, i.e. for references at rest in the unactuated directions; and `k'_t = k_t` since `Bᵀ(b'_{t+1} − b_{t+1}) =
−BᵀMr = 0`. A known disturbance `d_t` adds `P_{t+1}d_t` to `b_{t+1}` in both
recursions and replaces `Ar − r` by `Ar + d_t − r`. ∎

The class has dimension `(n − r)(n − r + 1)/2`; with a single cost regime it is
empty, because `M = M − AᵀMA` forces `M = 0` for invertible `A`. It is
potential-based reward shaping (Ng, Harada & Russell, 1999) restricted to the
potentials that keep the cost quadratic in the state with no state–action cross
term.

## Appendix B — pointers

Background the plan leans on; each is a starting point rather than a citation of
a specific result used here.

* Optimal feedback control and the minimum-intervention principle: Todorov &
  Jordan (2002), *Nature Neuroscience*.
* Signal-dependent noise in LQG models of movement: Todorov (2005), *Neural
  Computation*.
* Motor-cortex dynamics interpreted as a feedback controller: Kalidindi et al.
  (2021), *eLife*.
* Inverse optimal control with sensorimotor noise models: Schultheis, Straub &
  Rothkopf (2021), *NeurIPS*.
* Inverse LQR as a linear/convex problem, including biological applications:
  Priess et al. (2015), *IEEE Transactions on Automatic Control*;
  Keshavarz, Wang & Boyd (2011), on imputing convex objectives.
* Reward shaping and its invariance: Ng, Harada & Russell (1999), *ICML*.
* Maximum-entropy inverse reinforcement learning: Ziebart et al. (2008), *AAAI*.
* Differentiating through LQR solvers: Amos et al. (2018), *NeurIPS*.
* Control-theoretic models of neural population dynamics: Kao, Sadabadi &
  Hennequin (2021), *Neuron*; Schimel et al. (2022), *ICLR*.
