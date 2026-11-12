# Kernel-prior emissions: implementation plan

Port of the dynameg "constrained emission under a spatial prior" fit
(`dynameg/validate/shared_constrained.py`, `shared_parcel_prior.py`,
`shared_scale_correlation.py`, `src/dynameg/inverse.py::fit_kernels`) into
SSD.jl as new Gaussian observation classes. Working branch:
`claude/lvi-port-ssd-jl-kxfqf9`. dynameg itself is not changed in this pass.

Status legend: `[ ]` open, `[~]` in progress, `[x]` done. Update this file in
the same commit as the work.

## 1. Model

```
y_t = Φ H x_t + d + D u_t + ε_t,   ε_t ~ N(0, R)          C = Φ H
vec H ~ N(0, I_k ⊗ Σ(w)),   Σ(w) = Σ_sum + Σ_sc            (r × r)
w ~ N(μ(η), Λ(η)⁻¹),   η ~ p(η)                            (trees, hyperpriors)
```

- `Φ` (p × r): fixed, full column rank, given by the caller already reduced
  (decided 2026-11-12: no wide `L`, no projection inside SSD.jl).
- `Σ_sum = Σ_g θ_g Z_g`, `θ_g = exp(w_g)`. A kernel is an r × r PSD matrix or
  a factor `F` (r × m, `Z = F F'`), already in `H`'s coordinates.
- `Σ_sc = G S C S G'`, with `G` (r × n) given by the caller (`Φ'L`),
  `S² = θ_{b(i)} v_i` over a partition of the n columns into scale blocks,
  `C = I + Σ_c ρ_c (R_c − I_c)`, correlation families (n × m factors) ordered
  coarse to fine, each kernel nested in one kernel of every earlier family,
  `ρ_c = (1 − Σ_{ancestors} ρ) σ(a_c)` (stick-breaking). As in
  `shared_scale_correlation.py`.

Three exported classes, one parametric struct underneath
(`basis::Union{Nothing,Matrix}`, `prior::Union{Nothing,KernelPrior}`):

| class | basis | prior | notes |
|---|---|---|---|
| `ConstrainedGaussianObservationModel` | Φ | none | `H` point estimate; replaces the `C_basis` field (removed) |
| `KernelGaussianObservationModel` | I | kernels (p × p) | `q(C)` |
| `ConstrainedKernelGaussianObservationModel` | Φ | kernels | `q(H)` |

## 2. Inference (variational Laplace EM)

Mean field `q(x) q(H) δ(w − ŵ)`, Laplace `N(ŵ, Σ_w)` used for the tree
updates and the reported evidence. `A1 = Φ'R⁻¹Φ`; `Sxx`, `Syx` from the
existing `SufficientStatistics` (`obs_xx`, `obs_xy`; regressors `[x; 1; u]`).

- **q(x)**: the existing smoother, with emission precision `Ĉ'R⁻¹Ĉ + Δ`,
  `Δ = E[H'A1H] − Ĥ'A1Ĥ`. Hook: the new type gets its own
  `_compute_obs_constants!` (adds `−Δ` to `yt_given_xt`) and gradient kernels
  (`−Δ x_t`); `joint_loglikelihood!`, `gradient_batched!`,
  `_smooth_mean_only_batched!` signatures widened. No new smoother.
- **q(H)** (exact given `R`, `[d D]` profiled): with `Σ^½`,
  `eig(Σ^½ A1 Σ^½) = P λ P'`, `eig(Sxx) = Qx μ Qx'`:
  `H = Ĥ + B E Qx'`, `B = Σ^½ P`, `Var E_lj = V_lj = 1/(1 + λ_l μ_j)`.
  Mean from the residualized moments (`[1; u]` partialled out of `Sxx`, `Syx`,
  own eigenbasis); covariance from the uncentred `Sxx`. Then
  `[d D] = (Syz − ΦĤ Sxz) Szz⁻¹`.
  Moments: `E[H S H'] = Ĥ S Ĥ' + B diag(V s) B'`, `s = diag(Qx'SQx)`;
  `Δ = Qx diag(g'V) Qx'`, `g = diag(B'A1B)`;
  `E log p(H) + H[q(H)] = rk/2 − ½ tr(Σ⁻¹ĤĤ') − ½ ΣV + ½ Σ log V`.
- **R**: residual scatter at `Ĉ` plus `Φ B diag(Vμ) B'Φ'`, through
  `_accumulate_obs_scatter!` / `_finalize_R!` (keeps `R_prior`, `R_diagonal`).
- **dynamics**: existing `update_*!` state M-step, unchanged.
- **w**: MAP of `−k/2 [log|Σ| + tr(Σ⁻¹S)] − ½ (w−μ)'Λ(w−μ)`, `S = E[HH']/k`,
  by L-BFGS (Optim) with analytic gradient
  `k/2 tr((Σ⁻¹ − Σ⁻¹SΣ⁻¹) ∂_iΣ) + Λ(w−μ)`, warm-started. Laplace covariance
  `(k I(w) + Λ)⁻¹`, `I_ij = ½ tr(Σ⁻¹∂_iΣ Σ⁻¹∂_jΣ)`.
- **η** (trees): coordinate ascent given the Laplace posterior, as
  `Hierarchy.update` / `JointHierarchy.update`. Every `hyper_every` iterations.
- **Gauge**: no projection to a unit second moment. The prior fixes `V = I_k`
  in the model's own gauge and every dynamics parameter is free, so the bound
  is well posed. A parameter-expansion step (PX-VB) speeds the drift along the
  gauge orbit: `T'T = E[H'Σ⁻¹H]/r`, `x → T x`, `Ĥ → Ĥ T⁻¹`,
  `Δ → T⁻ᵀΔT⁻¹`, state side and its priors through a refactored
  `_affine_gauge!` that takes `T`. It never lowers the bound. Empirical prior
  centres are computed once at the start (after one unit-moment gauge) and
  frozen. `gauge_center`/`gauge_whiten` are refused for these classes.
  Option `gauge = :bound` (default, the PX step), `:moment` (dynameg's unit
  centred second moment each iteration; the trace is then not monotone) or
  `:none`.
- **Trace**: `Q_state + E_q(H)[Q_obs] + H[q(x)] + E log p(H) + H[q(H)] +
  log p(ŵ|η) + log p(η) + state/R priors`. Monotone with the trees held; a
  tree update uses `Σ_w`, so it can lower it slightly. The Laplace evidence
  (adds `½ log det 2πΣ_w`) is a diagnostic.
- Held-out score: the `x` part of the bound on the test trials at fixed
  `q(H)`, `ŵ`.

## 3. Hyperprior interface

Each weight belongs to a `type` (a `Symbol`: a kind of sum kernel, the scale
blocks, one correlation family, …). Per kernel: `name`, `type`, optional
`group` (level-1 label), optional `unit` (matches weights across types).
Per type, `levels`:

- `:flat` (0): `w_g ~ N(m_g, s_g²)`, fixed. Default centre: the "alone" rule
  (`log(power / tr Z_g)`), default SD 1.
- `:hierarchical` (1): `w_g ~ N(μ, τ²)`, `μ ~ N(m₀, r²)`.
- `:nested` (2): `w_g ~ N(μ_group, τ²)`, `μ_group ~ N(m, s²)`, `m ~ N(m₀, r²)`.

Level variances `~ Inv-χ²(ν₀, s₀²)`; defaults `(s₀, ν₀) = (0.5, 4)`,
`r = 3`, SD floor 0.05 (the `shared_scale_correlation` defaults). One SD per
level per type.

`correlated = [(:typeA, :typeB, …)]`: the weights of one `unit` across those
types share `N(μ_node, Ω)`, `Ω = diag(σ) R diag(σ)`, `R ~ LKJ(2)`; unmatched
units keep the marginal prior. The joined types must use the same `levels`.

Checks (errors name the offending kernels): unique names; shapes against the
declared space; symmetric PSD / finite factors; positive trace; `Σ` positive
definite at the start; `:nested` needs a `group` for every kernel of the type
and at least two groups; `:hierarchical` at least two kernels; scale blocks
partition the rows, none empty; correlation kernels of a family do not
overlap and nest in earlier families; `correlated` types share units.

Out of scope for this pass (stated, not built): per-latent weights
(`shared_parcel_prior`), Student-t groups, softplus link, the convex-bound
update, rotation of the latents, SLDS / LQR / composite / `depends_on`
(refused with an `ArgumentError`), source-space maps, contrast and region
posteriors (these stay in dynameg).

## 4. Files

- `src/stats/kernel_prior.jl`: `Kernel`, `ScaleCorrelation`, `KernelPrior`,
  covariance / gradient / information, checks.
- `src/stats/kernel_hierarchy.jl`: trees, joint tree, hyperpriors.
- `src/stats/kernel_weights.jl`: MAP + Laplace; exported
  `fit_kernel_weights!` (also the post-hoc lift).
- `src/lds/kernel_observations.jl`: the struct and aliases, `q(H)`, emission
  M-step, bound terms, E-step constants, accessors
  (`emission_posterior`, `kernel_weights`).
- `src/lds/fit_kernel_LDS.jl`: `fit!` / `elbo` / `smooth` / `loglikelihood` /
  `rand` methods, PX gauge step, holdout.
- Tests: `test/KernelPrior/*.jl`, `test/LinearDynamicalSystems/KernelEmissions.jl`,
  fixtures in `test/test_data/kernel_emissions/` (`.mat`, written by
  `make_reference.py` from the dynameg functions; committed).

## 5. Milestones

Each ends with its tests green, `JuliaFormatter` clean, a commit and a push.

- [x] **M0** Plan committed.
- [x] **M1** Sum kernels, flat prior, MAP + Laplace.
  Tests: gradient and information against ForwardDiff; recovery of known
  weights from Wishart draws; reference fixture against `fit_kernels`
  (`method="laplace"`); every check in §3 that applies.
- [ ] **M2** Vertical slice: the struct, `q(H)`, E-step hook, emission M-step,
  bound, `fit!`; migrate `C_basis` into `ConstrainedGaussianObservationModel`.
  Tests: `q(H)` and its moments against a dense `vec H` solve; smoother with
  `Δ` against a dense joint Gaussian (single, batched, ragged); bound equals a
  brute-force value on a tiny problem; monotone trace (weights fixed, weights
  learned); a very wide prior reproduces the constrained ML fit; `Φ = I` and
  a wide prior reproduce `GaussianObservationModel`; the `EmissionBasis` tests
  pass on the new class.
- [x] **M3** Trees: `:hierarchical`, `:nested`, hyperpriors.
  Tests: fixture against `Hierarchy.update`; closed-form two-level Gaussian
  case; simulated recovery of group means; small groups do not collapse.
- [x] **M4** Correlated types (joint tree, LKJ).
  Tests: fixture against `JointHierarchy.update`; recovered sign and size of a
  simulated correlation; unmatched units.
- [ ] **M5** Scale-correlation structure.
  Tests: covariance against the explicit `G S C S G'`; gradient / information
  against ForwardDiff; stick-breaking keeps `C` a correlation matrix; fixture
  against `ScaleCorrelationPrior`; simulated recovery of `ρ`.
- [ ] **M6** PX gauge step, held-out scoring, `rand`, `loglikelihood`
  (plug-in), `show`, accessors, refusals. Tests: gauge step raises the bound
  and leaves the likelihood terms unchanged; fewer iterations with it on;
  end-to-end recovery from the generative model against the unconstrained fit.
- [ ] **M7** Docs: `docs/src/LinearDynamicalSystems.md` section, `api.md`,
  `CHANGELOG.md`; Aqua (ambiguities), JET, full `Pkg.test()`.
- [ ] **Later** dynameg wiring (`run_fit.jl` flags, format), per-latent
  weights, source-side contrast posteriors.

M5 depends only on the `AbstractKernelStructure` interface in
`src/stats/kernel_prior.jl` and can run in parallel with M2.

## 6. Decisions and log

- 2026-11-12: plan written. User decisions: the three classes are as in §1
  and `C_basis` is replaced by the constrained class; inputs are reduced
  (`Φ`, r × r kernels or factors, `G` for scale-correlation) and SSD.jl does
  no projection or source-space maps; both gauges, `:bound` the default; no
  extras (per-latent weights, Student-t, source maps) in this pass.
- The new struct joins `QuadraticEmission`, so the existing smoother, E-step
  and `_fit_tridiag!` loop run on it; it supplies `_compute_obs_constants!`,
  the per-bin emission kernels, `elbo!` and `mstep!`. Rejected: a Gaussian
  shadow with a surrogate `(C̃, R̃)` (needs `Ĉ'R⁻¹Ĉ` inverted; ill-conditioned
  when a loading shrinks).
- 2026-11-12: M1, M3, M4 done together. One `WeightTree` covers every case
  (`d = 1` for a single type, `d > 1` for a correlated set; 0, 1 or 2 levels),
  so the univariate and joint trees are one implementation
  (`src/stats/kernel_hierarchy.jl`). The MAP is L-BFGS, the Laplace covariance
  from the dense whitened derivatives. Agrees with dynameg's `fit_kernels`
  (MAP, covariance, free energy), `Hierarchy.update` and
  `JointHierarchy.update` on the fixtures in
  `test/test_data/kernel_emissions/` (`make_reference.py` regenerates them
  from a dynameg checkout; numpy and scipy only).
- Run single test files with a script that includes them under
  `julia --project=test` after `Pkg.develop(path=".")` there; revert the
  change that makes to `test/Project.toml` before committing.
