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

**Go — but build option (b), not option (a), and keep its noise causal.**
Concretely: a finite-horizon closed-loop mode in which the costate sits on the
Riccati graph `λ_{t+1} = P_{t+1} x_{t+1} + ν` and the plant noise arrives *after*
the controller acts. That is the user's `W_t` exactly, read causally. It borrows
`:hold`'s structure but **not its noise timing** (§5).

1. **The current noise model biases the controller.** On data from a causal
   LQR agent, M0 (today's model) recovers the one cost quantity the design
   identifies — the position-cost contrast between conditions, Fisher SE 0.10 —
   with a log error of 0.7–1.0, i.e. off by a factor of 2–2.6, at 7–10 standard
   errors. Its closed-loop gains are 2–4× worse than either structured model's.
   This is not an optimizer artifact (8000 further iterations change neither),
   and it survives fixing `A` at the truth, where the error grows to over ten
   standard errors.
2. **Both structured models fix it** — and with half the parameters. They also
   score better on held-out data, by 0.003–0.004 nats/bin on every seed, but be
   clear what that is: mostly parameter count. A correctly specified MLE loses
   about `k / (2 · 6000)` nats/bin out of sample here, 0.0045 for M0's 54
   parameters against 0.002 for M1/M2's 24–27. M0's *misspecification* cost
   beyond that is only 0.0006–0.0017 nats/bin. **The case against M0 is the
   bias, not the fit:** a model can predict nearly as well as the truth and still
   misread the controller by a factor of two.
3. **M1 and M2 are close on each other's data** (≤ 0.0020 nats/bin), each winning
   on its own — distinguishably so on M1 draws once the plant is known, where M2
   also misreads the costs. Choosing between them is therefore not a fit question but a
   modelling and cost question, and both point the same way:
   - M1's costate noise is **non-causal** (below): the control error at `t`
     depends on costate innovations after `t`, pinned to vanish at the deadline.
     M2's is the causal agent of `simulate_lqr`.
   - M2 needs no terminal factor and no conditional normalizer. M1 needs
     both, under time-varying noise.
4. **Noise timing matters as much as noise structure.** M2 with `:hold`'s
   timing (plant noise entering before `W_{t+1}`) is as biased as M0 on causal
   data (contrast error 0.30–0.36, about 4 SE) and loses 0.003–0.005 nats/bin to
   causal M2. So `:hold`'s noise model cannot be reused as is.
5. **The noise is time-varying in `z`, constant in the right coordinates.** For
   causal M2, keeping `λ` as a hidden state, they are
   `ε_t = W_{t+1}(x_{t+1} + S λ_{t+1} − A x_t)` and `ν_{t+1} = λ_{t+1} − P_{t+1}x_{t+1}`:
   independent, covariances `Σ` and `Ω` constant, unit Jacobian. For M1 they are
   `(x, δ = λ − P x)`. Either way `P_t` is one Riccati sweep per cost regime and
   trial length — about 20 μs at `n = 2, T = 30` — and the M-step can profile the
   noise out in closed form, as today.
6. **EM keeps the data out of the gradient loop.** The smoother runs once per EM
   iteration and reduces the data to per-transition statistics (5.5 KB here); the
   M-step's objective and gradient then cost the same at `N = 10` and
   `N = 1000`. Prototype EM reaches the direct maximum-likelihood fit (§6).

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

### A known

Fixing `A` at the truth removes the plant–cost trade-off (and 4 parameters).
Position costs become identified in absolute terms (SE 0.31), and the picture
sharpens rather than changes:

| data | model | params | dtest | Q pos (│log│) | contrast pos (│log│) | gains | slack share [true] |
|---|---|---|---|---|---|---|---|
| G1 | M0 | 50 | −0.0058 ± 0.0013 | 0.78 | 1.06 | 7.2 % | — |
| G1 | M1 | 23 | −0.0017 ± 0.0003 | 0.27 | 0.09 | 2.1 % | 0.002 [0] |
| G1 | M2 | 20 | −0.0013 ± 0.0005 | 0.23 | 0.08 | 1.8 % | 0.115 [0] |
| G2 | M0 | 50 | −0.0042 ± 0.0010 | 0.76 | 0.99 | 6.0 % | — |
| G2 | M1 | 23 | −0.0019 ± 0.0009 | 0.42 | 0.09 | 1.9 % | 0.007 [0.26] |
| G2 | M2 | 20 | −0.0011 ± 0.0005 | 0.34 | 0.09 | 1.9 % | 0.549 [0.26] |
| G3 | M0 | 50 | −0.0056 ± 0.0027 | 0.70 | 0.87 | 6.3 % | — |
| G3 | M1 | 23 | −0.0019 ± 0.0011 | 0.34 | 0.04 | 2.5 % | 0.041 [0.042] |
| G3 | M2 | 20 | −0.0039 ± 0.0011 | 0.41 | 0.19 | 3.3 % | 0.306 [0.042] |

| data | M1 − M0 | M2 − M0 | M1 − M2 |
|---|---|---|---|
| G1 | +0.0042 ± 0.0013 | +0.0045 ± 0.0008 | −0.0003 ± 0.0006 |
| G2 | +0.0023 ± 0.0010 | +0.0030 ± 0.0006 | −0.0008 ± 0.0006 |
| G3 | +0.0037 ± 0.0017 | +0.0018 ± 0.0018 | +0.0020 ± 0.0005 |

- M0's contrast error is now 0.87–1.06 against an SE of 0.084 — over ten
  standard errors on every generator, including its own nearest relative (G3).
  Its absolute position cost is off by a factor of about 2 (SE 0.31); M1 and M2
  sit at about one SE.
- On M1 draws (G3), M2 is now distinguishable (+0.0020 ± 0.0005 nats/bin) and
  visibly misspecified: its velocity cost runs off (log error 4.0) and it reports
  a slack share of 0.31 against 0.04 — it explains persistent costate error as
  white slack. So the causal-vs-planning question *is* testable on data like
  these, by comparing the two fits; it does not have to be assumed.
- M2's plant-noise/slack split stays unreliable (Σ error 0.66 on G2).

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

## 5. Noise timing: causal, not `:hold`'s

`:hold` puts the plant noise in the plant row before the controller's gain,
`x_{t+1} = W(A x_t − S ν + ε)` — what `simulate_lqr` calls `:implicit` timing — so
the innovation is `W(Σ + SΩS)Wᵀ`, constant in plant/manifold coordinates. A
causal agent's noise arrives after the control is chosen:
`x_{t+1} = W(A x_t − S ν) + ε`, innovation `Σ + W S Ω S Wᵀ`. With a stationary
`W` the two are the same model class (`Σ ↦ W Σ Wᵀ`); with the finite-horizon
`W_{t+1}` they are not. Fitting the `:hold`-timed version (M2i) to the same data,
4 seeds:

| | data | contrast pos (│log│) M2 → M2i | gains M2 → M2i | Σ error M2 → M2i | dtest M2i − M2 |
|---|---|---|---|---|---|
| A free | G1 | 0.13 → 0.35 | 2.4 → 3.6 % | 0.18 → 0.54 | −0.0044 ± 0.0005 |
| | G2 | 0.10 → 0.32 | 2.7 → 3.2 % | 0.71 → 0.67 | −0.0033 ± 0.0008 |
| | G3 | 0.27 → 0.08 | 4.3 → 4.6 % | 0.19 → 0.76 | −0.0053 ± 0.0007 |
| A known | G1 | 0.08 → 0.36 | 1.8 → 3.1 % | 0.17 → 0.64 | −0.0050 ± 0.0006 |
| | G2 | 0.09 → 0.30 | 1.9 → 2.7 % | 0.66 → 0.80 | −0.0036 ± 0.0007 |
| | G3 | 0.19 → 0.13 | 3.3 → 3.5 % | 0.45 → 0.83 | −0.0051 ± 0.0007 |

On causal data (G1, G2) the wrong timing costs as much held-out likelihood as
M0 does, with a third of its parameters, and biases the contrast by about 4 SE.
It fits worse than the truth even on training data (`dtrain` −0.001 to −0.005):
plain misspecification, not variance.

Causal timing is therefore required, and it does **not** cost the closed-form
noise update, provided the costate stays in the state. On `z = [x; λ]` the
causal model is

```math
z_{t+1} = \begin{bmatrix} \Phi_t & 0 \\ P_{t+1}\Phi_t & 0 \end{bmatrix} z_t + \text{noise},
\qquad
\begin{bmatrix} \varepsilon_t \\ \nu_{t+1} \end{bmatrix}
 = \underbrace{\begin{bmatrix} W_{t+1} & W_{t+1}S \\ -P_{t+1} & I \end{bmatrix}}_{\det = 1} z_{t+1}
 - \begin{bmatrix} \Phi_t & 0 \\ 0 & 0 \end{bmatrix} z_t
 \sim N\big(0, \mathrm{blkdiag}(\Sigma, \Omega)\big),
```

so `ε_t` is `W_{t+1}` times `:hold`'s own plant-row residual and `ν` its manifold
row. Checked numerically: the `2n` chain gives the same `log p(y)` as the `n`-dim
closed loop to 1e-13, and the map above reproduces `blkdiag(Σ, Ω)` to 5e-16 with
determinant 1. The expected complete-data log-likelihood is then two
independent Gaussian blocks with constant covariance, and both `Σ` and `Ω`
profile out exactly as `Σ` does today:

```math
g(\theta) = \tfrac{N}{2}\log\det \textstyle\sum_t W_{t+1}\,\mathbb{E}[r^p_t r^{p\top}_t]\,W_{t+1}^\top
          + \tfrac{N}{2}\log\det \sum_t \mathbb{E}[r^m_t r^{m\top}_t] ,
```

with `r^p`, `r^m` the plant and manifold rows — no Jacobian term at all. On `x`
alone the innovation `Σ + W S Ω S Wᵀ` is a sum, and no closed form exists: that is
the price of marginalizing `λ`, and the reason to keep it.

## 6. Compute and memory

All single-threaded, Float64, this machine; the prototype's filter allocates
freely, so its absolute numbers are upper bounds on a careful implementation.

**Two different Riccati recursions.** The *control* Riccati sweep
(`P_t = Q + AᵀP_{t+1}(I + S P_{t+1})⁻¹A`, backward) maps the parameters to the
controller `Φ_t` and the noise; it never sees data. The *filter's* covariance
recursion (forward) is also parameter-only, so it is shared by every trial of a
length and condition. Only the filter's means touch the data. So a likelihood
evaluation is one control sweep + one filter covariance pass + a pass over the
means of all `N` trials.

Control sweep, one condition:

| n | T = 30 | T = 100 |
|---|---|---|
| 2 | 22 μs, 66 KB | 78 μs, 220 KB |
| 4 | 42 μs, 121 KB | 131 μs, 403 KB |
| 8 | 61 μs, 281 KB | 205 μs, 943 KB |
| 16 | 180 μs, 804 KB | 553 μs, 2.7 MB |

Exact `log p(y)`, one condition, `T = 30`: closed loop on `x` (d = n) vs the
Hamiltonian chain on `[x; λ]` with its terminal normalizer (d = 2n):

| n | N | M2 on x | M0 on [x; λ] |
|---|---|---|---|
| 2 | 10 | 0.12 ms | 0.12 ms |
| 2 | 100 | 0.31 ms | 0.39 ms |
| 2 | 1000 | 2.4 ms, 7 MB | 2.8 ms, 8.5 MB |
| 8 | 100 | 1.2 ms | 1.5 ms |
| 8 | 1000 | 8.7 ms, 23 MB | 8.5 ms, 29 MB |

The trial-mean pass dominates, so doubling the state costs far less than the
`8×` a `d³` count suggests at these sizes.

**Direct fitting re-filters on every step.** The fits in §4 maximize `log p(y)`
with L-BFGS and ForwardDiff through the sweep *and* the filter, so every
objective and gradient re-runs both over all trials: the M2 gradient (24
parameters, two conditions) is 2.2 ms at `N = 10`, 11 ms at `N = 100`, 80 ms and
300 MB at `N = 1000`. Simple and exact, not cheap.

**EM does not.** The smoother runs once per iteration; after it, the M-step
objective reads only per-transition statistics (three `n × n` matrices per
transition per condition — 5.5 KB here, independent of `N`):

| N | E-step (smoother + stats) | M-step value | M-step gradient |
|---|---|---|---|
| 10 | 0.36 ms | 0.15 ms | 0.5 ms |
| 100 | 0.97 ms | 0.16 ms | 0.5–1.0 ms |
| 1000 | 5.7 ms, 18 MB | 0.14 ms | 0.5 ms |

Checked end to end (`--only=em`, G1, 100 trials per condition, naive start):
generalized EM never decreased `log p(y)` over 500 iterations, came within 1e-4
nats/bin of its final value after 46, and finished 1e-6 nats/bin from the direct
fit, with the same contrast (0.239) and gain errors (1.6 %).

**The package's EM today (M0), for scale** — `fit!` on the same plant, 4 Julia
threads, BLAS single-threaded, per iteration:

| N | T | total | E-step | M-step | allocated |
|---|---|---|---|---|---|
| 30 | 30 | 18 ms | 1.0 ms | 16 ms | 6.6 MB |
| 100 | 30 | 16 ms | 3.5 ms | 18–28 ms* | 6.6 MB |
| 1000 | 30 | 25 ms | 27 ms* | 18 ms | 6.8 MB |
| 100 | 100 | 44 ms | 7.5 ms | 39 ms | 12 MB |

(*E/M split timed separately from the totals; medians, noisy.) The M-step —
an L-BFGS over the structural parameters on summed statistics — dominates up to
a few hundred trials and is flat in `N`; the E-step grows linearly, about 27 μs
per trial at `T = 30`. Causal M2 changes neither shape: its E-step is the same
smoother on the same `2n` state, and its M-step adds one control sweep and its
adjoint per evaluation (tens of μs at these sizes) while keeping statistics per
transition instead of summed (`O(T (2n)²)` memory instead of `O((2n)²)`,
kilobytes).

## 7. Cost of building it

The current M-step profiles a constant `Σ` out (`Σ = R(θ)/N`) and runs L-BFGS on
`g(θ) = (N/2) log det R(θ) − N log|det A|` over statistics summed within each
cost regime (`lqr_mstep.jl`). `:hold` mode does the same with
`L_h = [I S; −P I]`, `−N log det(I + S P)`, and one adjoint Stein solve through
the DARE.

**Causal M2 (recommended)**:

- the residual is `:hold`'s plant row left-multiplied by `W_{t+1}`, and its
  manifold row; `Σ` and `Ω` profile out separately (§5), with no Jacobian term;
- statistics must be kept per transition (or per time-to-go `τ = T − t`, which is
  what `P` depends on under a constant schedule, so trials of different lengths
  still pool) rather than summed;
- the gradient through `P_t` is a backward adjoint of the Riccati recursion
  instead of a Stein solve — standard and cheap;
- **no terminal factor and no normalizer**: the chain is stable, `p(y)` is the
  score, and `rand` works directly (no more `simulate_lqr` vs `rand` split);
- `Ω` must stay positive definite for the `2n` smoother (as `Σ`'s costate block
  must today), and EM slows as `Ω → 0`, the usual price of a nearly
  deterministic latent;
- schedules with per-trial offsets need statistics keyed by the trial's
  schedule window instead of `τ` alone.

**M1** would need the terminal factor and the conditional normalizer
`log p(f = 0)` under time-varying noise, with its gradient; its own coordinates
help (unit Jacobian, block-diagonal noise) but the normalizer is where the work
and the numerical risk are.

## 8. Recommendation

1. Implement causal M2 as a finite-horizon closed-loop mode (`mode = :causal`,
   say) on the `2n` state: `:hold`'s structure with the finite-horizon sweep, the
   causal residual of §5, and per-transition statistics. The `2n` state is not
   just for reuse (emissions, `SLDS` units sharing a latent dimension with
   `:lqr`/`:hold` states, grouping, priors): it is what keeps the noise update
   in closed form.
2. Do not reuse `:hold`'s noise timing for it (§5). For `:hold` itself the
   timing is a reparameterization, not a bias, because its `W` is constant.
3. Keep M0 for what it is good at — the exactly-solvable regression tests and
   historical fits — and stop recommending it for inverse control on behavioral
   data: on causal-agent data its cost estimates are biased at many standard
   errors.
4. Leave M1 unbuilt unless the open-loop-planning hypothesis becomes the one
   under test; §2 is how to build it if so. The known-plant G3 rows say that
   test has power: a feedback model fitted to planning data loses measurably
   and misreads the cost.
5. Before interpreting costs from any model, run the `fisher` section on your
   own design. Here only the position contrast and the closed loop were
   identified; absolute costs were not, by any model.
6. Treat the fitted slack as a nuisance until a threshold sweep on your design
   (`biological.jl --only=threshold`) says otherwise.

Caveats: one plant, one cost pair, `n = 2`, a full-rank `C` held at the truth, no
offsets or inputs, 4 seeds. The structure of the result (M0's bias is a
misspecification effect; noise timing matters once `W` varies; M1 vs M2 is a
causality choice) does not depend on those, but the magnitudes do. The §4 fits
of M2 run on `x` alone; the `2n` form is the same likelihood (§5).
