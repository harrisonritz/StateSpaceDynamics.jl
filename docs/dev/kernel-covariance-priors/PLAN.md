# Kernel priors on Q and R: implementation plan

Extends the kernel-prior emissions (`docs/dev/kernel-emissions/PLAN.md`) to the
noise covariances. Status legend: `[ ]` open, `[~]` in progress, `[x]` done.

## 1. Decisions (2026-11-13, with the user)

- **Architecture.** A prior on `Q` is only anatomical if the latent coordinates
  are: the intended use is a *fixed basis*, `C = Φ` held (`x` in the basis's
  `r` coordinates, kernels `Φ'L K L'Φ` directly). No projection through a
  fitted loading (its kernels would be the outside ones sandwiched by `H`, with
  only contrasts informative, and would couple `p(Q)` to `q(H)`).
- **R** is gauge-free and takes a kernel prior with any Gaussian emission,
  including the kernel-prior emissions (`L K L'` in sensor space plus a
  nuisance kernel: sensor noise and unmodelled brain background).
- **Inference.** Variational: `q(Q)`, `q(R)` inverse-Wishart, the other
  parameters point estimates. The smoother sees `E[Σ⁻¹]`, so it is unchanged.
- **Optional by parameter.** `Q` and `R` priors are independent. No `P0` kernel
  prior, no `A` prior (deferred: `A = A_sym + A_skew` with the kernel as the
  mean of `A_sym`).

## 2. Model

    Σ ~ IW(ν, (ν − d − 1) K(w)),   E[Σ] = K(w) = kernel_covariance(prior, w)
    w ~ the KernelPrior's hyperprior (types, levels, trees)

`KernelIWPrior(prior; dof=ν)`, a `KernelPrior` and the degrees of freedom
(`ν > d + 1`), in the `Q_prior` of a `GaussianStateModel` or the `R_prior` of a
`GaussianObservationModel` / `StructuredGaussianObservationModel`.

## 3. Inference

With `S` the expected residual scatter and `N` its count (transitions or bins,
plus the width of a paired `AB_prior` / `CD_prior`), exactly as the IW MAP
uses them now:

- `q(Σ) = IW(ν + N, Ψ + S)`, `Ψ = (ν − d − 1) K(w)`. The model stores
  `Σ̃ = E[Σ⁻¹]⁻¹ = (Ψ + S)/(ν + N)` in `Q` / `R`: every consumer of `Q`, `R`
  (smoother, M-steps, `q(H)`) then sees `E[Σ⁻¹]`, which is what VB asks.
- Weights: the MAP of `E_q[log IW(Σ | Ψ(w), ν)] + log p(w)`,
  `= ν/2 log|K| − (ν − d − 1)/2 tr(K E[Σ⁻¹]) + log p(w) + const`, by the
  kernel prior's L-BFGS; Laplace covariance `(ν I(w) + Λ)⁻¹` (the Fisher
  information of `ν` draws); trees every `tree_every` iterations.
- Bound: the point-`Σ̃` terms already in the ELBO, plus
  `−N/2 (E[log|Σ|] − log|Σ̃|)` (the likelihood and MN prior priced `log|Σ̃|`),
  `+ E_q[log p(Σ | w)] − E_q[log q(Σ)] + log p(w, trees)`, replacing the IW
  MAP term.
- Start: after the first E-step, the weights are fitted to the starting `Σ`
  (as `E[Σ⁻¹]⁻¹`) and `ν_q = ν + N` is set.

## 4. Supporting change

`GaussianObservationModel.C_fixed`: hold `C` while `d`, `D` are fitted
(generalizes `_fixed_obs_cols`, whose fixed columns were pinned at zero, to
keep their values). Needed for the fixed basis; useful on its own.

## 5. Scope and refusals

`fit!` of a `GaussianStateModel` LDS with a Gaussian or structured emission.
Refused with an `ArgumentError`: `depends_on`, the affine `gauge_*` options
(they transform `Q`), `R_diagonal` with a kernel `R_prior`, SLDS, composite
emissions, `LQRStateModel`.

## 6. Milestones

- [x] **K1** Objective: a Wishart-scale target for the kernel weights
  (gradient, information), beside the Gaussian-draws one. Tests: ForwardDiff
  gradient and Fisher information, sum kernels and scale–correlation.
- [x] **K2** `KernelIWPrior`: `q(Σ)`, weight update, bound terms, start.
  Tests: bound against a closed form at the optimal `q`; recovery of known
  weights.
- [x] **K3** Wiring: `_finalize_Q!`, `_finalize_R!`, the prior log-densities,
  the start in both fit loops and `elbo`; `C_fixed`; refusals. Tests: monotone
  trace (flat levels) for Q, R, both, with the kernel emission; `C_fixed`
  holds `C` and fits `d`.
- [x] **K4** Docs (LDS page section, API), changelog, JuliaFormatter, Aqua.

## 7. Log

- 2026-11-13: K1–K3 done. The weights' objective takes a target (`_Draws`,
  `_IWMean`); the inverse-Wishart one's expected curvature is `ν` times the
  per-draw information (tested by finite differences, sum kernels and
  scale–correlation). The bound at the optimal `q(Σ)` equals the closed-form
  marginal. `KernelIWPrior` keeps `posterior_dof` (from the training data)
  apart from `count` (the data a bound is scored on), so the held-out bound
  prices its own transitions under the training posterior. `C_fixed` is its own
  branch of the Gaussian emission M-step, not part of `_fixed_obs_cols` (whose
  fixed columns other paths pin at zero). With `C` held, `d`'s part in the
  basis's span trades with the latents' mean.
- Open: an `A` prior (`A_sym` with a kernel mean); learning `dof`; the
  constrained / kernel-prior emissions with `C_fixed`-like holding of `H`.
- 2026-11-13 (2): learned loading. Decisions with the user: models with the
  anatomy in `R`, `Q`, `R + Q` (each with a learned `H`, no `H` prior) and in
  `H`, plus the fixed-basis variants; both projections, noise-weighted
  (`(H'Z⁻¹H)⁻¹`) the default; `H` by its closed form, then a gradient polish
  with the prior's term (best of the old and closed-form `H` as the start, so
  the bound does not decrease); the `A` prior later. Both projections are
  `dK̃ = W dZ W'` (`W = K̃H'Z⁻¹` or `H⁺`): one target (`_IWProjected`), its
  information whitened by `L̃⁻¹W`. `dof` defaults to `d + 2`, resolved at the
  start (a projected `Q`'s `d` is `k`, not the kernels' `r`); it is checked
  against `d` in `_validate_covariance_priors`. Refused: a `Q` prior with a
  kernel prior on `H` (its `q(H)` would enter the projection) or with a free `C`.
- `A` prior, for later: a matrix-normal with kernel row (receiver) and column
  (sender) covariances, `Var(A_ij) = u_i v_j`; weights by flip-flop targets
  (`E[A V⁻¹A']`, `E[A'U⁻¹A]`); `q(A)` given `Q` has the two-Kronecker form of
  `q(H)`; a latent `A` adds `E[A'Q⁻¹A]` to the smoother.
- 2026-11-13 (3): `KernelMNPrior` on `A` (`src/stats/kernel_dynamics_prior.jl`).
  `A ~ MN(mI, U, V)`; `Ã = (A − mI)V^{-½}` has the loading's prior, so `q(A)`
  is `_loading_posterior`'s and `Δ_A = _loading_excess(q, Q⁻¹)`. The smoother
  gets `Δ_A` in `xt1_given_xt` (every Hessian path reads it) and `−Δ_A x_t` in
  `_state_gradient!` (the step is one Newton step from the old iterate, so both
  are needed); the batched mean pass is off for these models. `Q_state!` (both
  forms) subtracts `½ tr(Δ_A Σ_{t<T} E[x x'])`; `update_Q!` adds `A`'s spread.
  Weights: `m` in closed form, then rows given columns and columns given rows
  (`_Draws` targets with `k` draws). The start alternates with the actual
  `q(A)`: alternating with `A` as known collapses the rows at a start of `mI`.
  Projection through `H`: rows by `:noise_weighted` / `:euclidean`, columns by
  the new `:sandwich` (`H'ZH`); `_projection_gradient` serves `Q`, `U`, `V`.
  The projection types were refactored into one `_Projection` for both targets.
  Checked against dense posteriors (q(A), bound, smoother on equal, ragged and
  single trials), finite differences (sandwich, A's term in `H`), monotone
  fits (fixed and learned bases; A, A+R, A+Q, A+Q+R) and a planted sender.

