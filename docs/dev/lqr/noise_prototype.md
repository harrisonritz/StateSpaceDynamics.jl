# Should the costate innovation inherit the plant noise?

The question: a causal LQR agent's costate innovation is a fixed image of its
plant noise, `w^λ = P ε + ν`, so the forward innovation covariance is

```math
W_t = \begin{bmatrix} \Sigma & \Sigma P_{t+1}^\top \\ P_{t+1}\Sigma & P_{t+1}\Sigma P_{t+1}^\top + \Omega \end{bmatrix}
    = L_{t+1}\,\mathrm{blkdiag}(\Sigma, \Omega)\,L_{t+1}^\top,
\qquad L = \begin{bmatrix} I & 0 \\ P & I \end{bmatrix}.
```

Is that worth modelling, instead of `LQRStateModel`'s constant full-rank mixed
`Σ`? If so, how — and what does it cost?

Numbers come from `noise_prototype.jl`, which imports nothing from `src/`
(selftest, Fisher analysis, fits; see its header for the commands).

## Verdict

**Go — but build option (b), not option (a).** Concretely: a finite-horizon
generalization of the existing `:hold` mode, in which the costate sits on the
Riccati graph `λ_{t+1} = P_{t+1} x_{t+1} + ν` and the plant row carries the
plant noise. That is the user's `W_t` exactly, read causally.

1. **The current noise model biases the controller.** On data from a causal
   LQR agent, M0 (today's model) recovers the one cost quantity the design
   identifies — the position-cost contrast between conditions, Fisher SE 0.10 —
   with a log error of 0.7–1.0, i.e. off by a factor of 2–2.6, at 7–10 standard
   errors. Its closed-loop gains are 2–4× worse than either structured model's.
   This is not an optimizer artifact: 8000 further iterations change neither.
2. **Both structured models fix it** — and with half the parameters. They also
   score better on held-out data, by 0.003–0.004 nats/bin on every seed, but be
   clear what that is: mostly parameter count. A correctly specified MLE loses
   about `k / (2 · 6000)` nats/bin out of sample here, 0.0045 for M0's 54
   parameters against 0.002 for M1/M2's 24–27. M0's *misspecification* cost
   beyond that is only 0.0006–0.0017 nats/bin. **The case against M0 is the
   bias, not the fit:** a model can predict nearly as well as the truth and still
   misread the controller by a factor of two.
3. **M1 and M2 are close on each other's data** (≤ 0.0014 nats/bin), each winning
   on its own. Choosing between them is therefore not a fit question but a
   modelling and cost question, and both point the same way:
   - M1's costate noise is **non-causal** (below): the control error at `t`
     depends on costate innovations after `t`, pinned to vanish at the deadline.
     M2's is the causal agent of `simulate_lqr`.
   - M2 needs no terminal factor and no conditional normalizer, and drops into
     the `:hold` machinery that already exists. M1 needs both, under
     time-varying noise.
4. **The noise is time-varying in `z`, constant in the right coordinates.** For
   M2 those are the plant-row/manifold-row coordinates `:hold` already uses; for
   M1 they are `(x, δ = λ − P x)`. Either way `P_t` is one Riccati sweep per
   cost regime and trial length — `O(T n³)`, shared by every trial — and the
   M-step keeps profiling the noise out in closed form.

What does **not** come out of this: the slack. With full-state observations M2
over-reads slack on G2 (share 0.61 vs 0.26) and reports a spurious 0.12 on G1;
plant noise and slack trade off. That is `biological.md` §4's finding again, and
the same caution applies before reading a fitted `Ω` as suboptimality.

---

## 1. The three models

All share the plant `A`, control authority `S = B R⁻¹ Bᵀ`, costs `Q_k` per
condition (terminal cost = running cost), observations `y_t = C x_t + d + r_t`.
The costate is hidden throughout.

| | mean | innovation | boundary | score |
|---|---|---|---|---|
| **M0** (today) | Hamiltonian `M_k` on `z = [x; λ]` | constant `G Σ_mix Gᵀ`, `Σ_mix` full 2n×2n | soft `λ_T = Q x_T`, free 2n prior per condition | `p(y │ f = 0)` |
| **M1** (option a) | Hamiltonian `M_k` | `L_{t+1} blkdiag(Σ, Ω) L_{t+1}ᵀ` | soft `λ_T = Q x_T`; `λ_1 = P_1 x_1 + ν` | `p(y │ f = 0)` |
| **M2** (option b) | closed loop `Φ_t = W_{t+1} A` on `x` | `Σ + W_{t+1} S Ω S W_{t+1}ᵀ` | none needed | `p(y)` |

with `W_{t+1} = (I + S P_{t+1})⁻¹` and `P_t` the finite-horizon Riccati sweep from
`P_T = Q_k`. M0 is checked against the package: the prototype's likelihood
equals `loglikelihood(LinearDynamicalSystem(LQRStateModel(…), …))` to 4e-16
relative, conditional and joint, at two terminal covariances.

Generators: **G1** the causal agent of `simulate_lqr` (plant noise only), **G2**
the same plus `costate_slack` (slack share 0.26), **G3** exact draws from M1.
G1 and G2 are exact draws from M2; M1 at `Ω = 0` *is* M2 at `Ω = 0` (checked).

## 2. What option (a) actually says

Write `δ_t = λ_t − P_t x_t`, the costate's departure from the Riccati graph. Using
`P_t − Q = Aᵀ P_{t+1} W_{t+1} A`, M1's Hamiltonian chain becomes

```math
x_{t+1} = \Phi_t x_t - S A^{-\top}\delta_t + \varepsilon_t,
\qquad
\delta_{t+1} = \Phi_t^{-\top}\delta_t + \nu_t,
\qquad \delta_T \approx 0 ,
```

and since `S A⁻ᵀ Φ_tᵀ = W_{t+1} S`, the plant row is
`x_{t+1} = Φ_t x_t − W_{t+1} S e_t + ε_t` with `e_t = Φ_t^{-⊤} δ_t`. Compare M2:
`x_{t+1} = Φ_t x_t − W_{t+1} S ν_{t+1} + ε_t`. **The two differ only in what drives
the control error**: M2 a fresh `ν` each step; M1 a process `δ` that is
*anti-stable forward* (`Φ⁻ᵀ`) and therefore, once the terminal condition pins
it, *stable backward*: `δ_t = Φ_tᵀ(δ_{t+1} − ν_t)`. M1's control error at `t` is a
discounted sum of costate innovations **after** `t`.

That is a coherent Gaussian model, but not of an agent that re-plans: it is the
two-point boundary value problem with its noise, the "open-loop planning"
hypothesis `biological.md` §5 calls level 4. If that is the hypothesis you want
to test, M1 is the right model of it. As the default noise for a feedback
controller it is the wrong one.

Two practical consequences:

- In `(x, δ)` coordinates M1's noise is the constant `blkdiag(Σ, Ω)` and the
  coordinate change has unit determinant — so its M-step can profile `Σ` and `Ω`
  separately and needs no `log|det A|` term. The P-dependence moves into the
  residual. That is the efficient way to compute it, if it is ever built.
- Its terminal normalizer `log p(f = 0)` grows like `ρ(Φ⁻¹)^{2T}`, exactly as
  M0's does. A float64 dense evaluation of it is already off by 1e-4 nats at
  `T = 30` with `ρ(M) ≈ 1.6` (the Kalman form is exact to 1e-10 against 256-bit);
  the package's square-root integration exists for this reason.

## 3. What the design can identify

Before comparing models, what can *any* model recover? Expected Fisher
information of the true model (M2 on G1), 100 trials per condition, `T = 30`,
`C` fixed (which pins the similarity gauge; `tr S` pins the S/Q scale):

| SE of log … | A free: position | velocity | A known: position | velocity |
|---|---|---|---|---|
| `Q_1[i,i]` | 0.58 | 2.30 | 0.31 | 0.66 |
| `Q_2[i,i]` | 0.61 | 1.43 | 0.31 | 0.60 |
| contrast `Q_2[i,i]/Q_1[i,i]` | **0.096** | 1.11 | **0.084** | 0.51 |

The closed loop is well identified (gains to ~2%); the cost behind it is not,
except for the position contrast. Absolute costs trade off against the plant
(`A` free) and against each other along the late-trial transient, which is where
the finite-horizon Riccati sweep departs from stationarity and where the state
has decayed into the noise. **Velocity-cost errors in the tables below are
uninformative about the noise model** — any model, including the true one, gets
them wrong by a factor of several. Read the position contrast and the gains.

## 4. Results

4 seeds; 100 training and 100 test trials per condition, two conditions,
`T = 30`; `C, d, R` fixed at the truth; each fit is the better of a truth start
and a naive start. `dtest` is held-out log-likelihood per bin minus the
generating model's (0 = as good as the truth).

### A free

| data | model | params | dtest | contrast pos (│log│) | gains | slack share [true] |
|---|---|---|---|---|---|---|
| G1 | M0 | 54 | −0.0062 ± 0.0015 | 0.97 | 8.0 % | — |
| G1 | M1 | 27 | −0.0017 ± 0.0004 | 0.14 | 2.2 % | 0.001 [0] |
| G1 | M2 | 24 | −0.0018 ± 0.0006 | 0.13 | 2.4 % | 0.118 [0] |
| G2 | M0 | 54 | −0.0051 ± 0.0004 | 0.92 | 8.1 % | — |
| G2 | M1 | 27 | −0.0022 ± 0.0004 | 0.19 | 3.0 % | 0.008 [0.26] |
| G2 | M2 | 24 | −0.0016 ± 0.0004 | 0.10 | 2.7 % | 0.613 [0.26] |
| G3 | M0 | 54 | −0.0060 ± 0.0026 | 0.72 | 7.3 % | — |
| G3 | M1 | 27 | −0.0021 ± 0.0011 | 0.08 | 2.9 % | 0.043 [0.042] |
| G3 | M2 | 24 | −0.0036 ± 0.0010 | 0.27 | 4.3 % | 0.100 [0.042] |

Paired held-out differences (same test set, mean ± sd over seeds, nats/bin):

| data | M1 − M0 | M2 − M0 | M1 − M2 |
|---|---|---|---|
| G1 | +0.0044 ± 0.0016 | +0.0044 ± 0.0017 | +0.0000 ± 0.0004 |
| G2 | +0.0029 ± 0.0003 | +0.0035 ± 0.0003 | −0.0006 ± 0.0004 |
| G3 | +0.0038 ± 0.0017 | +0.0024 ± 0.0017 | +0.0014 ± 0.0008 |

KNOWN_A_PLACEHOLDER

### Robustness of the M0 numbers

Every M0 fit stopped at the 2000-iteration cap. Continuing two of them for 8000
more iterations moved the training objective by 2.4e-4 nats/bin and changed
neither the held-out score (−0.0060 → −0.0060; −0.0054 → −0.0056) nor the
contrast error (1.12 → 1.11; 0.95 → 0.96). M0 is crawling along flat directions
(its 20-parameter costate prior and costate noise block), not toward the truth.
The two starts agree to 2–5e-4 nats/bin.

Read `dtest` against capacity. With `k` parameters fitted on 6000 bins, a
correctly specified model is expected to lose about `k/12000` nats/bin out of
sample: 0.0045 for M0, 0.0023 for M1, 0.0020 for M2. Against that:

| excess over capacity (nats/bin) | G1 | G2 | G3 |
|---|---|---|---|
| M0 | −0.0017 | −0.0006 | −0.0015 |
| M1 | +0.0005 | +0.0001 | +0.0002 |
| M2 | +0.0002 | +0.0004 | −0.0016 |

So each structured model is predictively exact on its own data, and M0's
misspecification costs about as much as M2's does on M1 draws — little. What
capacity cannot explain is the contrast bias, a misspecification effect: a
constant mixed covariance has to average a time-varying, rank-`n`, P-dependent
residual (plus, on G2, the slack's lag-one correlation between rows `t` and
`t+1`), and the cost absorbs the error. Predictive scores will not flag it;
recovery on simulated data does.

## 5. Cost of building it

The current M-step profiles a constant `Σ` out (`Σ = R(θ)/N`) and runs L-BFGS on
`g(θ) = (N/2) log det R(θ) − N log|det A|` over statistics summed within each
cost regime (`lqr_mstep.jl`). `:hold` mode already does the same with
`L_h = [I S; −P I]`, `−N log det(I + S P)`, and one adjoint Stein solve through
the DARE.

**M2 (recommended)** = `:hold` with the finite-horizon `P_{t+1}` in place of the
DARE solution:

- noise stays constant in plant/manifold coordinates, so `Σ` is still profiled
  in closed form (full 2n×2n, so plant–manifold correlation is available, and
  the slack block is `Ω`);
- statistics must be kept per time-to-go `τ = T − t` rather than summed, because
  `L_{h,t}` varies with `t`; with a constant schedule `P` depends only on `τ`, so
  trials of different lengths still share them — memory `O(T_max (2n)²)`;
- the Jacobian becomes `Σ_t log det(I + S P_{t+1})`;
- the gradient through `P_t` is a backward–forward adjoint of the Riccati
  recursion instead of a Stein solve — standard and cheap;
- **no terminal factor and no normalizer**: the chain is stable, `p(y)` is the
  score, and `rand` works directly (no more `simulate_lqr` vs `rand` split);
- schedules with per-trial offsets need statistics keyed by the trial's
  schedule window instead of `τ` alone.

Per-iteration compute: one Riccati sweep per (regime, length) — negligible next
to the smoother. In the prototype the M2 gradient is 23 ms against M0's 190 ms,
mostly because it carries less than half the parameters; the E-step can stay on
`z = [x; λ]` (reusing everything) or drop to `x` alone (8× cheaper in `d³`).

**M1** would need all of the above *plus* the terminal factor and the
conditional normalizer `log p(f = 0)` under time-varying noise, with its
gradient; its own coordinates help (unit Jacobian, block-diagonal noise) but the
normalizer is where the work and the numerical risk are.

## 6. Recommendation

1. Implement M2 as a finite-horizon closed-loop mode (`mode = :causal`, say),
   generalizing `:hold`. Keep the 2n state so plant-row/manifold-row noise and
   all the downstream machinery carry over.
2. Keep M0 for what it is good at — the exactly-solvable regression tests and
   historical fits — and stop recommending it for inverse control on behavioral
   data: on causal-agent data its cost estimates are biased at many standard
   errors.
3. Leave M1 unbuilt unless the open-loop-planning hypothesis becomes the one
   under test; §2 is how to build it if so.
4. Before interpreting costs from any model, run the `fisher` section on your
   own design. Here only the position contrast and the closed loop were
   identified; absolute costs were not, by any model.
5. Treat the fitted slack as a nuisance until a threshold sweep on your design
   (`biological.jl --only=threshold`) says otherwise.

Caveats: one plant, one cost pair, `n = 2`, a full-rank `C` held at the truth, no
offsets or inputs, 4 seeds. The structure of the result (M0's bias is a
misspecification effect; M1 vs M2 is a causality choice) does not depend on
those, but the magnitudes do.
