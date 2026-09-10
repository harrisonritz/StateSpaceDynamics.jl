# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **The inverse optimal control model now uses LQR naming throughout.** Use
  `LQRStateModel`, `LQRFitFlags`, and `lqr_matrix`; source files, internal
  helpers, tests, and tutorials follow the same convention. This is a complete
  API rename without compatibility aliases; model behavior is unchanged.

### Fixed
- **`refresh!` now rebuilds a grouped model's variant caches.** A
  `LQRStateModel` with `depends_on` builds one variant per cell, aliasing
  the parent's arrays for every group that does not vary — so a write to the
  parent *is* a write to theirs. But each variant carries its own derived cache,
  and a grouped model smooths through those, not through the parent's. A
  parameter assigned by hand therefore reached the fields and never the
  transitions actually used. Silent, and large: on a two-session stitched fit,
  `rescale_costate!` — which mutates `Qc` and `S` and then refreshes — moved the
  ELBO by tens of thousands of nats across an operation that is an exact
  symmetry. `refresh!` now refreshes any variants it has, one level deep.
- **`rescale_costate!` refuses a model whose structure is grouped.** Variants
  alias the parent for groups that do not vary, so rescaling the parent serves
  them all — but a model whose *structural* block varies by cell holds separate
  costs, and one global factor would rescale one cell and leave the rest on a
  different costate scale. It says so instead.
- **Composite emissions work with an inverse-LQR state.** Three sites indexed a
  member's sufficient statistics by key — the grouped emission M-step, the
  switching one, and the pooled initial-state update — but an LQR state's
  statistics *wrap* the per-member blocks rather than being them, so a fit
  combining an inverse-LQR state with several emissions threw a `MethodError`
  the moment it reached the M-step. They now unwrap through the accessors that
  already exist for it (`_obs_suf`, `_slds_init_suf`), which are the identity for
  every other state model.

- **The weighted LQR aggregator accepts input views.** It asserted
  `data.ux[trial]::Matrix`, which rejected the ordinary case of a `Data` built
  from an array the caller owns — where the per-trial inputs are views rather
  than copies. `ux` is only ever reached through `tview`, so nothing downstream
  could tell the difference.

### Added
- **Switching inverse LQR under `depends_on`.** The grouped switching M-step now
  handles LQR discrete states, so a switching inverse-LQR model can be
  *stitched*: one control problem per discrete state, read out through one
  emission per session. Previously this path aggregated with the base routine
  rather than the dispatched one and then looked for a conjugate update that an
  inverse-LQR state does not have.

  The units are the `K · ncells` (regime, cell) pairs, and each structural
  block's version at a unit is the pair of its version across regimes (from the
  tie) and across cells (from the grouping), mapped to a dense index by
  `_lqr_pair_slots`. With the state side ungrouped — only the emission stitched,
  which is the usual shape — every cell shares one structural version and this
  reduces exactly to the ungrouped switching M-step, so the stitched fit and the
  single-session one are the same estimator. A `K = 1` grouped switching fit
  reproduces the grouped single fit parameter for parameter, which is the
  anchor the ungrouped path already had.

- **Switching inverse LQR.** An `SLDS` can now switch between inverse-LQR
  dynamics, so the discrete state selects *which control problem* generated the
  behaviour — different plants, different costs, or both.

  ```julia
  slds = SLDS(; A=P, πₖ=π, LDSs=[LinearDynamicalSystem(LQRStateModel(A, S, Q₁, Σ), obs),
                                 LinearDynamicalSystem(LQRStateModel(A, S, Q₂, Σ), obs)])
  fit!(slds, y; tied_params=[:structure])     # one shared control problem
  ```

  * **The latent dimension is uniform, and cannot be otherwise.** The `SLDS`
    uses a structured variational approximation with a *single* continuous latent
    path — each timestep is the responsibility-weighted mixture
    `ℓₜ = Σₖ wₖₜ ℓₜ⁽ᵏ⁾(x)` — so every discrete state reads and writes the same
    `2n`-dimensional `z = [x; λ]`. Differently-sized states are not expressible,
    and would not be wanted even if they were: their per-timestep log densities
    are against different base measures, so the responsibilities would drift
    toward whichever state has more dimensions to spend.
  * **The discrete state replaces the cost schedule.** In a switching model the
    state *is* the cost epoch, inferred rather than specified, so a member with
    more than one `Qc` is rejected — that would be a second notion of regime
    nested inside the first. `terminal` and `observe_costate` describe the trial
    and the emission rather than the state, so they must agree across states.
  * **`:free` mode**, via `free_state_model(M, Σ)`: a state whose `2n × 2n`
    transition is unconstrained rather than symplectic, so a switching model can
    mix plain linear dynamics with LQR dynamics. `SLDS` stores one concrete
    state-model type per discrete state, which is why a "plain dynamics" state
    has to *be* an `LQRStateModel`; `mode` is a runtime field rather than
    a type parameter for the same reason. Its M-step is the ordinary conjugate
    regression, and it reproduces a `GaussianStateModel` to machine precision —
    matching log-likelihood, identical smoothed means, and EM that tracks
    iterate for iterate.
  * **The weighted generalized M-step** is the unweighted one fed
    responsibility-weighted statistics: `_LQRMStepCtx` reads only the statistics,
    and its Jacobian coefficient comes from `Σ nk`, which the weighted aggregator
    fills with the *effective* count `n̄ₖ = Σ γₖ(t)`. Scaling `−n̄ₖ log|det Aₖ|`
    by a raw timestep count instead would stay monotone under balanced
    responsibilities and break once they separate.
  * **Tying**, including partial. `:structure` shares the whole joint block
    `(A, S, Qc, h, Bu, Gref)` across discrete states and `:noise` shares `Σ`, but
    the individual blocks may also be named: `tied_params = [:A, :S]` is one
    plant with a cost per discrete state, which is what a switching inverse-LQR
    model is usually for. Tied states pool their statistics into one version, so
    a tie is fitted jointly rather than fitted once and copied — including a
    partial tie, which is one L-BFGS solve rather than an alternation, because
    the parameter vector is laid out block-major and a shared block simply has
    one copy instead of one per state. That is the difference between one
    objective sweep per L-BFGS iteration and two, and it is worth 1.8-2.1x of
    the wall clock spent in the structural M-step at plant dimension 16 and 24,
    at a higher ELBO at every equal-wall-clock checkpoint. A frozen block is
    never shared whatever the tie asks for, since freezing means "keep your own
    value", and the states sharing a layout must agree on `fit_flags` as they
    already must on `fit_bool`.
  * A `LQRStateModel(latent_dim)` constructor taking the **total**
    dimension — the spelling a switching model wants — throwing on an odd value,
    and `plant_dim` for the other half of the contract.

  A `K = 1` switching model reproduces the ungrouped inverse-LQR fit, and an
  all-`:free` one reproduces a Gaussian `SLDS`; both are tested, and the first is
  what caught the costate readout leaking back through the switching emission
  M-step.

- **`trial_elbos` covers LQR latents.** The per-trial ELBO
  split now accepts an `LQRStateModel` alongside the Gaussian ones, with
  the same contract — `sum(trial_elbos(m, y)) + log p(θ) == elbo(m, y)` — and
  the same support for Gaussian, Poisson and composite emissions, multi-regime
  cost schedules, the terminal factor, `observe_costate`, inputs and ragged
  trial lengths.

  This is what lets an LQR fit be scored per trial: a held-out likelihood
  quoted per trial, a paired bootstrap between models, or a co-smoothing
  conditional `log p(y_out | y_in)` taken as a difference of two ELBOs.

  An LQR `Q_state!` has no per-trial kernel — its transition term lives
  in mixed coordinates and carries the `N log|det A|` Jacobian — so the split
  re-aggregates each trial's statistics and calls the aggregated `Q_state!` on
  them. That is exact, not approximate: at fixed parameters the residual scatter
  is linear in the aggregated blocks and `N` / `N_f` are plain counts, so the
  objective is additive over trials. `_aggregate_lqr_stats!` takes an
  optional `trials` argument to make that reachable.

- **Inverse LQR through LQR latents.** A new state model,
  `LQRStateModel`, whose latent state is the LQR state-costate pair
  `z = [x; λ]` and whose transition is constrained to the symplectic form a
  linear-quadratic control problem implies — so fitting the state-space model
  *is* recovering the plant and the cost function the behaviour is optimal for.

  ```julia
  sm = LQRStateModel(A, S, Qc, Σ)          # S = B R⁻¹ Bᵀ, Qc the state cost
  lds = LinearDynamicalSystem(sm, GaussianObservationModel(C, R, d))
  fit!(lds, y)
  lqr_parameters(sm)                               # (A, S, Qc, schedule, terminal)
  ```

  The user-facing latent dimension doubles: `LQRStateModel(A, ...)` with
  an `n × n` plant gives a `2n`-dimensional latent. Everything else — Gaussian,
  Poisson and composite emissions, single- and multi-trial, ragged and
  equal-length fast paths, inputs, initial-state priors — works as it does for
  any other state model, and the ordinary linear-Gaussian path is unchanged.

  * **The M-step preserves the structure.** Estimation happens in the *mixed*
    coordinates `[x_{t+1}; λ_t] = 𝓔_t [x_t; λ_{t+1}]`, where
    `𝓔_t = [A −S; Q_t Aᵀ]` is linear in the free parameters — while the forward
    symplectic transition `M_t` is rational in them. The rearrangement is free:
    every moment of `(w, v)` is a block of the joint moment of `(z_t, z_{t+1})`
    the smoother already returns, so the E-step is untouched. The change of
    coordinates does carry a Jacobian, `log det Q^fwd = log det Σ − 2 log|det A|`,
    and the profiled objective `½ log det R(θ) − log|det A|` is optimized by
    L-BFGS with analytic gradients, accepted only when it improves — a
    generalized M-step, so the ELBO never decreases.
  * **Time-varying cost.** `Qc` holds `K` cost matrices and a per-timestep
    `schedule` says which each timestep uses; `cost_schedule(T; terminal, onset)`
    builds the common shapes (a running cost, a delay epoch, a terminal cost).
    Only `M_t` varies across regimes — the noise map does not — so the extra cost
    is `K` cached transitions.
  * **Terminal condition.** Optional, as a soft pseudo-observation
    `0 = λ_T − Q_{k_T} x_T − h_f + ε_f`. It is what removes the forward
    transition's unstable directions, since a symplectic matrix has reciprocal
    eigenvalue pairs.
  * **The costate needs process noise**, and this is structural rather than
    numerical: the forward covariance is `G Σ Gᵀ` with `G` invertible, so a
    singular `Σ` leaves the smoother's precision undefined. `Σ` lives in the
    mixed coordinates, which makes its blocks interpretable as plant process
    noise and costate slack.
  * **The emission does not read the costate** by default; `observe_costate=true`
    opts in. The constraint is applied inside the emission M-step (a decoupled
    Gram for a linear solve, a frozen block in the Poisson Newton system), not by
    projecting afterwards.
  * `simulate_lqr` rolls out the optimal trajectory via the backward Riccati
    sweep and the closed-loop map — the way to generate ground truth, since the
    model's own forward flow is unstable by construction.
  * **Reference / tracking control.** The tracking Lagrangian gives an affine
    term `[d_t; -Q_t r_t]`, so the costate half of the input coupling is *not*
    free: it is tied to the same cost matrix that sits in `𝓔_t`'s lower-left
    block, and it varies with the regime because `Q_t` does. A free,
    regime-shared `B_u` cannot represent that. `Gref` supplies it — writing
    `r_t = G_r u_t`, regime `k`'s input matrix is `B_u - [0; Q_k G_r]` and the
    terminal factor picks up `+Q_{k_T} G_r u_T`, so a reach is scored against
    where the target was. Pass the reference as the input and freeze `G_r` at
    `I`, or pass task regressors and estimate it.
  * **`depends_on` grouping.** The state side offers four groups matching the
    `fit_bool` slots: `:x0`, `:P0`, `:structure` (the whole joint block) and
    `:noise`. Observation-side grouping with a shared plant — stitching sessions
    — works too. Groups sharing a noise version pool into that version's
    residual scatter, so grouped fits are joint rather than group-by-group, and
    a single-group fit reproduces the ungrouped one exactly.
  * Utilities: `symplectic_matrix`, `lqr_matrix`, `symplectic_defect`,
    `riccati_solution`, `closed_loop_dynamics`, `lqr_parameters`,
    `lqr_riccati_sequence`, and `rescale_costate!` for the classical
    inverse-optimal-control scale invariance (the cost is identified only up to a
    nonzero scalar).

- Several observation models on one latent state. Hand `LinearDynamicalSystem` a
  `NamedTuple` of observation models instead of one and they all read out the
  same latent process, with observations supplied under the same keys:

  ```julia
  lds = LinearDynamicalSystem(state_model,
                              (kin = GaussianObservationModel(C, R, d),
                               spk = PoissonObservationModel(C, d)))
  fit!(lds, (kin = Ykin, spk = Yspk); uy = (kin = Vkin, spk = Vspk))
  ```

  The members are conditionally independent given the latent path, so every
  emission quantity — log-density, gradient, curvature, Q-term, prior — is a sum
  over them, and each member keeps its own channel count, priors, `depends_on`,
  `group_seeds` and `fit_bool` flags. Any number of members, all of one type or
  mixed. Works for `LinearDynamicalSystem` and `SLDS`.
  * A member is reached by its key and a member's parameter by the key-suffixed
    name — `obs.kin`, `obs.kin.C`, `obs.C_kin`. That suffixed spelling is what
    `depends_on` (`(C_kin = session, d_kin = session, R_kin = session)`),
    `tied_params` (`(:C_spk, :d_spk)`), `fit_bool` and `group_parameter` use,
    so one emission can be per-session, or frozen, or tied across SLDS regimes,
    while another is not
  * `fit_bool` becomes `[x0, P0, A&b&B, Q]` followed by one block per
    observation model (`[C&d&D, R]` for a Gaussian emission, `[C&d&D]` for a
    Poisson one). It also accepts a keyword form that lowers to that layout —
    `fit_bool = (x0=true, …, kin=(C=true, R=false), spk=(C=true,))` — which
    avoids counting positions by hand. Single-emission layouts (length 5 / 6)
    are unchanged
  * Which smoother runs follows from the members' types, at compile time: an
    all-Gaussian composite has an emission curvature that does not depend on the
    latent path, so it takes the single-Newton-step smoother with its
    equal-length shared-covariance fast path, while any non-Gaussian member
    routes to the iterative Laplace smoother. Verified against the equivalent
    stacked single-emission model (`C = [C₁; C₂]`, `R = blockdiag(R₁, R₂)`):
    identical smoothed means and covariances, ELBO and marginal log-likelihood,
    and identical parameters after one EM step
  * What the composite buys over that stacked form is a block-diagonal `R` that
    stays block-diagonal, per-member `obs_dim` (so one member can be stitched
    across sessions of differing channel counts while another is not), and
    members that need not be Gaussian at all
  * `rand` returns observations as a `NamedTuple`; `loglikelihood` is available
    for an all-Gaussian composite and errors for a mixed one, as it does for a
    Poisson LDS
  * Known gap: the BLAS-3 batched mean pass does not apply to a composite — it
    stacks one observation tensor against one `C`/`d`/`D`, which a composite does
    not have. The equal-length fast path still computes the covariance once and
    shares it, so this costs the promotion of the mean pass from BLAS-2 to
    BLAS-3, not the shared-covariance saving

- `fit!(slds, y; tied_params=...)`: share any parameter group across every
  regime instead of fitting one per regime. Takes a `Symbol` or a collection of
  them, named the way `depends_on` and `fit_bool` name parameters — `[A b B]` is
  fit as one regression so any of `:A`/`:b`/`:B` names the whole group, likewise
  `:C`/`:d`/`:D` for `[C d D]`, with `:Q` and `:R` groups of their own.
  `tied_params = (:C, :R)` is the usual reading for neural data, where the
  recording does not change when the dynamics do (and `K` times fewer emission
  parameters); `tied_params = (:A, :Q)` is the mirror image, one set of dynamics
  with switching emissions. `:x0`/`:P0` are accepted and ignored, since an SLDS
  ties its initial state across regimes unconditionally. Works with
  `depends_on`, where the tie is *within* a group — each session keeps its own
  version, shared by every regime — and leaves a frozen group (`fit_bool`)
  untouched. Tied groups are broadcast before the first E-step, so no regime
  ever infers `q(x)`/`q(z)` through a parameter the model does not have
  * Tying a regression alongside its noise covariance (`(:C, :d, :D, :R)`,
    `(:A, :b, :B, :Q)`) is the ordinary M-step on pooled statistics: the shared
    term does not depend on the regime and `Σₖ γₖ(t) = 1`, so the summed
    per-regime weighted objectives collapse to the unit-weight one. Tying only
    the noise is equally cheap — each regime contributes its own residual
    scatter and they are summed before the covariance is formed
  * Tying only the regression (`(:C, :d, :D)` without `:R`) is exact but costs
    more. The residual covariance no longer divides out of `∂/∂W`, so the output
    rows couple and the shared fit becomes a generalized least-squares solve of
    size `p·m` — `O((p·m)³)` against the pooled fit's `O(m³)`. The solver
    (`_tied_gls_regression`) is written against a flat list of units and reduces
    exactly to the pooled `mn_map` when the covariances agree, so it is
    available to any future caller with the same shape
  * Tying *part* of a regression (`:C` without `:d`) is exact too: each regime's
    free columns are projected out of its statistics, the shared block is solved
    by the same GLS over what remains, and the free columns come back by
    back-substitution (`_partial_tied_regression`, Frisch–Waugh–Lovell). Two
    cases have no such reduction and throw: a Poisson `[C d D]`, which is fitted
    by LBFGS rather than from sufficient statistics, and a partial tie alongside
    `depends_on`, which already splits the regression per group of trials. A
    matrix-normal prior is split between the shared and free blocks when its
    column precision `Λ` does not couple them (a diagonal `Λ` always qualifies),
    and throws when it does
- `smooth(slds, y; ux, uy, depends_on, smoothing_iters, tol, return_cov,
  progress)`: the variational posteriors of a fitted SLDS at fixed parameters,
  returned as one `NamedTuple` `(; x, γ, elbo, p)` — the continuous states
  `q(x)`, the discrete responsibilities `γₜ(k) = q(zₜ = k)`, the ELBO at those
  posteriors, and (opt-in via `return_cov`) the smoothed covariances. It
  alternates forward-backward over the switching chain with the Laplace/Kalman
  smoother over the continuous states (Ghahramani & Hinton, 1996) until `γ`
  converges (`tol`) or `smoothing_iters` alternations are spent. Unlike the
  single-Monte-Carlo-sample E-step `fit!` runs during learning, the coupling
  here is deterministic — the discrete layer is scored at the smoothed
  posterior mean — so the result is reproducible with no `rng` to pass. `fit!`
  runs the same alternation but keeps its forward-backward storage private, so
  this is the way to get `q(z)` (regime occupancy, a Viterbi-style `argmax`
  path, a rate averaged over regimes) out of a model, on training or held-out
  data
- `fit!(slds, y; smoothing_iters=n)`: run `n` discrete↔continuous alternations
  per E-step instead of one. The default of 1 is the standard vLEM update;
  larger values hand the M-step a better-converged posterior at proportional
  cost per iteration
- Ancillary parameter dependencies: every
  `AbstractStateModel` and `AbstractObservationModel` now carries a
  `depends_on` field (default `nothing`). Setting it to a `NamedTuple` of
  per-trial label vectors — e.g. `obs_model.depends_on = (C = session, R =
  session)` — declares that those parameters are to be estimated separately for
  each group of trials, while everything else stays pooled. This is the
  "stitching" setup for combining recording sessions that observe different
  neurons in the same animal: shared latent dynamics, session-specific
  emissions
  * Keys are canonicalized to the same groups `fit_bool` uses, since those
    parameters are fit jointly as one regression — `:A`/`:b`/`:B` name one
    group and `:C`/`:d`/`:D` another. Labels may be `Symbol`s, integers or
    strings. Different parameters may use different label vectors; the trial
    partition is their common refinement
  * A malformed declaration (unknown parameter name, aliases of one group
    carrying different labels, label vectors of unequal length) is rejected by
    `validate_LDS` — so by the positional `LinearDynamicalSystem(state_model,
    obs_model)` constructor, not at the first `fit!`
  * Per-group values live in a new `variants` field holding one model object
    per parameter-group combination, with non-varying parameters shared **by
    reference** so a single M-step write covers all of them. They are read back
    with the new exported `group_labels(model, name)` and
    `group_parameter(model, name, label)`
  * `show` reports the declared groups for a model that has any
  * Supported for the **Gaussian LDS**, the **Poisson LDS** and the **SLDS**
    across `fit!`, `smooth`, `elbo`, `loglikelihood` and `rand`, each of which
    also accepts a `depends_on` keyword overriding the model's stored labels so
    a held-out set with a different trial count can be scored without mutating
    the model. All versions of a parameter share the model's prior; each
    version contributes its own log-prior term to the ELBO
  * The efficiency of same-length epochs is preserved *within* each group:
    trials sharing every parameter form a cell, and the smoothed covariance is
    computed once per cell and shared across it (parameters differ between
    cells, so their covariances genuinely differ). The `O(D²·T)` workspace
    storage is allocated once and reused across cells, so a grouped fit's
    memory tracks an ungrouped one's instead of scaling with the number of
    groups
  * For the Poisson emission there is no sufficient-statistic form, so the
    M-step runs one LBFGS solve per version of `[C d D]`, over the trials of
    every cell that shares that version
  * For an `SLDS` every regime must declare the same labels — the trial
    partition is a property of the data, not of a regime — and mismatched
    declarations raise an `ArgumentError`; `x0`/`P0` stay tied across regimes
    as they are for an ungrouped SLDS
  * With `depends_on` unset, every entry point takes its original code path
- Public allocating `elbo(model, y; ...)` for all three models (Gaussian LDS
  with `ux`/`uy` keywords, Poisson LDS with Newton-smoother keywords, SLDS
  with an `rng` keyword since its E-step consumes a posterior sample). Runs
  one E-step and evaluates the ELBO at the resulting posterior, the same
  quantity `fit!` reports per iteration,  without requiring the private
  workspace structs the `elbo!` variants take (#139)
- `loglikelihood(slds, y)` now throws an informative error (the marginal is
  intractable for a switching model; use `elbo`) instead of a raw
  `MethodError`, mirroring the Poisson LDS (#139)
- Composable Normal-Inverse-Wishart prior on the initial latent state via a new
  `GaussianStateModel` field `x0_prior::Union{Nothing,MNPrior}` (the mean half),
  paired with the existing `P0_prior::IWPrior` (the covariance half). The initial
  state `x₁ ~ N(x0, P0)` is an intercept-only regression, so its NIW prior is the
  same `MNPrior` + `IWPrior` composition used for `[A b]`/`Q` and `[C d]`/`R` —
  no bespoke prior type. Construct the mean half with the exported
  `x0_mean_prior(μ₀; κ₀)` helper; the M-step then does
  `x0 = (Σγ·x₁ + κ₀ μ₀) / (Σγ + κ₀)` and folds `κ₀(x0-μ₀)(x0-μ₀)'` into the IW
  scale. With `κ₀ → 0` (and no `P0_prior`) it reduces exactly to the previous MLE
  update

### Changed
- `AbstractStateModel` gained an intermediate supertype,
  `AbstractGaussianStateModel`, for state models with a linear-Gaussian
  transition. `GaussianStateModel` and `LQRStateModel` are its subtypes,
  and the drivers, emission kernels, workspaces and aggregators now dispatch on
  it rather than on `GaussianStateModel`. Existing behaviour is unchanged; a new
  state model plugs in by supplying the state half of the log-density, gradient,
  curvature, ELBO and M-step.
- The emission kernels `observation_loglikelihood!`, `observation_gradient!` and
  `observation_hessian!` take the observation model rather than the enclosing
  `LinearDynamicalSystem`, which is what lets a composite emission call them once
  per member. They are unexported internals; no public API changed.
- `ParameterGrouping.cell_obs` is now one variant-index vector per observation
  model rather than a single one. Also internal.
- **Breaking:** parameter names no longer stand in for the group they are fitted
  with. `depends_on = (C = session, R = session)` was shorthand for grouping the
  whole `[C d D]` regression; it is now an error, and the members must be named
  — `(C = session, d = session, D = session, R = session)`. `:A`/`:b`/`:B` the
  same. A model with no observation input has no `D` to fit, so `(C, d)` is the
  whole group there. The old spelling reads as a claim about `C` alone while
  quietly fitting `d` and `D` per group as well, which is exactly the kind of
  mistake the check now catches. `group_labels` / `group_parameter` /
  `set_group_seeds!` are unaffected: they look a parameter up rather than
  declaring anything, so an individual name is still what they want
- **Poisson emission M-step now solves each neuron's parameters separately by
  Newton**, replacing the single LBFGS over all `obs_dim × (latent_dim + 1 +
  uy_dim)` parameters at once. The Q-function separates over the rows of
  `[C d D]` — no term couples two neurons, the matrix-normal prior included —
  and each row's problem is strictly convex, so the two solvers maximise the
  same unique optimum. Newton gets there in a handful of exact-curvature steps
  from the previous M-step's warm start instead of 25-40 limited-memory ones,
  which on a 200-neuron, 16-latent, 100-bin fit is the difference between ~35
  and ~4 sweeps over the trials. Every row backtracks on its own objective, so
  a badly scaled neuron cannot hold up the rest, and a row whose full Newton
  step promises less than the tolerance stops being solved for — which is what
  keeps a unit some session never recorded (optimum at `d = -∞`) from charging
  the others for its iterations. The previous solver is kept as
  `_update_observation_model_lbfgs!`, and the two are checked against each
  other in the tests
  * Fits are not bit-identical to previous versions: the emission M-step
    reaches a slightly *better* iterate than LBFGS's stopping rule allowed,
    so the EM trajectory differs. The ELBO stays monotone
- The banded LAPACK `pbsv` path in `block_tridiagonal_solve_spd!` now covers
  block sizes up to 32 rather than 8. Measured on a 100-block system it is
  2.1× faster than the general block-Thomas solve at block size 4, 1.5× at 16
  and 1.1× at 32, only losing by 48 — the old cutoff left the whole useful
  latent-dimensionality range on the slower path

- **Breaking:** the previously exported (but unused) `Data` struct is now a
  private, validated container for multi-trial observations + `ux`/`uy` inputs.
  Public entry points (`fit!`, `smooth`, `loglikelihood`) accept plain arrays —
  a `(obs_dim, T)` matrix, a `(obs_dim, T, ntrials)` array, or a vector of
  per-trial matrices, with `ux`/`uy` in the same shape family — and construct
  a `Data` at the boundary, which is the single shape/dimension validation
  site (observation rows are now checked against `obs_dim` up front). The
  multi-trial backend (`estep!`, multi-trial `smooth!`, the sufficient-stats
  aggregators, `_fit_tridiag!`) consumes `Data` instead of threading
  `y`/`ux`/`uy` triples through every signature (#139)
- `fit!(slds, y)` now validates observations through `Data` like the other
  entry points (dimension mismatches throw a clean `DimensionMismatchError`
  upfront instead of failing deep in the smoother) and accepts the
  `(obs_dim, T, ntrials)` array form (#139)
- `smooth` (public, allocating) now accepts `ux`/`uy` keywords on the Gaussian
  path and all three observation shapes on both the Gaussian and Poisson
  paths; multi-trial input returns per-trial vectors, matrix input returns
  matrices as before (#139)
- **Breaking:** `elbo(slds, y)` is now deterministic. It infers `q(x)` and
  `q(z)` by the same coordinate ascent as `smooth(slds, y)` and returns that
  call's `elbo` field, rather than running one Monte-Carlo E-step off a joint
  draw from `q(x)`. It no longer takes an `rng`, and takes `smoothing_iters` /
  `tol` / `progress` instead; the value is a converged bound rather than one
  matching `fit!`'s first noisy trace entry
- **Breaking:** `loglikelihood(slds, y)` returns the ELBO instead of throwing.
  The exact marginal `log p(y)` is still intractable for a switching model
  (it needs a sum over all `K^T` regime sequences), so the returned value is a
  variational lower bound — comparable across models fit to the same data, but
  not a likelihood
- **Breaking:** renamed the control-input arguments `latent_inputs`/`obs_inputs`
  to `ux`/`uy` across the public API (keywords on `fit!`/`rand`, positional on
  `smooth!`/`estep!`) (#139)
- **Breaking:** renamed the `LinearDynamicalSystem` fields
  `state_input_dim`/`obs_input_dim` to `ux_dim`/`uy_dim` to match (#139)
- Multithreading now uses OhMyThreads.jl (`tforeach`/`tmapreduce`) instead of
  `Base.Threads` (`@threads`/`@spawn`); OhMyThreads is a new dependency (#143)
- Deduplicated the per-observation-model complete-data log-likelihood, gradient,
  and Hessian implementations (Gaussian / Poisson / SLDS-weighted) into shared
  kernels in `continuous_latents.jl`. The emission-specific pieces are now single
  dispatch points (`obsloglikelihood!`, `observationgradient!`, and the Hessian
  emission block), the affine transition residual is defined in exactly one
  place, and kernel signatures follow a uniform `f!(out, ws, model, x, y, ...)`
  convention — a new observation model plugs in with one method per kernel
  (#135, #136, #141)
- Reworked the monolithic ~90-field `SmoothedWorkspace` into modular components
  (`BlockTridiagonalWorkspace`, `SmoothConstants`, `NewtonBuffers`,
  `RegressionBuffers`, `ElboBuffers`, `TDAggBuffers`, `BatchedBuffers`) and
  unified the dev-facing function handles across the Gaussian, Poisson, and
  SLDS paths (#144)
- `loglikelihood(lds, y)` now computes the observation-independent half of the
  Kalman filter (innovation covariances and gains) once and shares it across
  trials, uses the positive-definite-by-construction information-form update
  (`info_update!`) from the retired Kalman path, supports ragged trial
  lengths, and accepts `ux`/`uy` input keywords. Models with input matrices
  (`B`/`D` with nonzero columns) now **require** the matching input
  sequences — previously inputs were silently ignored, giving a wrong
  likelihood

### Performance
- `block_tridiagonal_inverse_logdet!` replaces its second (UL) sweep and the
  per-block factorisation that followed with the Kalman-smoother covariance
  recursion `Σᵢ₋₁,ᵢ₋₁ = Mᵢ₋₁⁻¹ - Dᵢ Σᵢ,ᵢ₋₁` read off the forward sweep's
  cached Cholesky factors — one factorisation per block where there were
  three. ~1.65× on the kernel, which is `O(T · D³)` and runs once per trial
  per E-step, so it dominates a fit at larger latent dimensionality (8.9 ms →
  5.4 ms per trial at `latent_dim = 32`, `T = 100`)
- The Poisson emission kernels are batched over a whole trial instead of
  looping over timesteps:
  * `hessian!` forms `C' diag(λₜ) C` for every `t` as one `gemm` over the
    `D(D+1)/2` distinct entries of the symmetric block, which also halves the
    arithmetic (12.2 ms → 0.4 ms per trial at `obs_dim = 200`, `latent_dim =
    16`, `T = 100`)
  * `Q_obs!` forms the linear predictor and the variance correction
    `ρᵢₜ = ½ cᵢ' Pₜ cᵢ` the same way (3.1 ms → 0.5 ms at the same size), and
    short-circuits the `log Γ(y+1)` normaliser at counts of 0 and 1, which at
    typical bin widths is almost all of the data
  * A `PoissonBatchBuffers` field on `SmoothWorkspace` holds the scratch,
    allocated on first use so a Gaussian fit never pays for it
- The SLDS emission Hessian uses the same batched Poisson kernel as the single
  LDS. `hessian!` for an SLDS sums `-γₖ(t)·C' diag(λₜ) C` over regimes, which is
  `O(K · N · D² · T)` and runs on every Newton step of every trial's smooth, on
  every E-step — the dominant cost of a Poisson SLDS fit. The per-regime
  responsibilities fold into the rates before the `gemm`, so the weighted
  curvature costs what the unweighted one does: 12.0 ms → 0.63 ms per
  `hessian!` call at `K = 2`, `obs_dim = 200`, `latent_dim = 16`, `T = 100`, and
  1.26 s → 0.50 s per EM iteration on an 8-trial fit of that size. The
  Gaussian path keeps the per-timestep kernel, whose curvature is a cached
  `O(D²)` axpy per timestep and has nothing to batch
- Together with the Newton M-step, a Poisson EM iteration on a 60-trial,
  200-neuron, 16-latent, 100-bin problem went from 5.4 s to 0.97 s, and the
  fraction of the iteration that runs in parallel rose from about half to
  nearly all of it — the previous LBFGS objective was evaluated on a single
  thread while only its gradient was chunked across the workspace pool

### Removed
- Stale one-off profiling scripts under `benchmark/profiling/` (#144)
- The retired information-form Kalman/RTS smoother EM machinery
  (`src/stats/kalman.jl`: the `_fit_kalman!` driver with its E/M-step, ELBO,
  and sufficient-statistics code, plus the internal `KalmanWorkspace`). It had
  not been a selectable `fit!` backend since v0.4.0; the filter it contributed
  now lives behind `loglikelihood` (see Changed). `marginal_loglikelihood`
  remains as an internal alias of `loglikelihood`
- The internal `tol_PD` / `id_PD` eigen-floor helpers. The filter wraps the
  model covariances as strict `PDMat`s instead. — a genuinely non-PD `Q`/`R`/
  `P0` now fails.

### Fixed

- A noise version's `Σ⁻¹` in the inverse-LQR M-step was looked up by the
  *structural* version rather than by the model that uses it. Wherever the two
  groupings differ — `tied_params = [:structure]` with the noise left free, or a
  `depends_on` whose structural and noise groups are not the same partition —
  every noise version took the first model's `Σ`, so the objective was
  mis-specified while staying finite and letting EM keep moving. The assumption
  behind it holds for `depends_on`, where cells sharing a parameter alias one
  array, and fails for an `SLDS`, whose discrete states hold separate arrays.


- ELBO monotonicity assertions now scale their tolerance to the bound's own
  magnitude rather than using a fixed `1e-6`. The sufficient statistics are
  accumulated in parallel chunks, so summation order — and with it the last few
  digits of every M-step — depends on the thread count. The result stays
  deterministic for a given count (repeated fits agree bit for bit), but the
  Gaussian+Poisson composite fit was monotone on one thread and dipped by ~6e-4
  on two, purely from reassociation, which EM then amplified into a different
  local optimum tens of nats away. The threshold was tighter than parallel
  floating point can promise, not a symptom of a race. `SSDTest.elbo_monotone`
  is the shared predicate; `test_em_monotone` takes `rtol` in place of `tol`.

- A grouped (`depends_on`) SLDS with a **Poisson** emission threw
  `BoundsError` out of the first M-step, so no such fit could run at all. The
  grouped SLDS M-step read `cell_slot[_G_R]` before branching on the emission
  type, and `R` is a group only on the Gaussian side — `_group_names` gives a
  Poisson emission `(:C,)` alone, so its `cell_slot` is one entry shorter and
  that index is off the end. It is now read inside the Gaussian branch, which
  is the only place its value was ever used. Every grouped-SLDS test was
  Gaussian, which is what let it through; there is now a Poisson one covering
  the plain grouped fit and the `tied_params = (:C, :d)` tie
- A grouped (`depends_on`) fit that pooled a regression over units with
  *different* noise versions — e.g. `depends_on = (R = session,)` with one
  emission over all sessions — solved the ordinary pooled normal equations,
  which do not maximize the ELBO when the residual covariance is not shared.
  Those cases now go through the generalized-least-squares solve described
  under Added; a version whose units do share a covariance keeps the cheap
  pooled path, which is the same estimator
- The documentation build failed: `set_group_seeds!` is exported and carries a
  docstring but was not in any `@docs` block, so Documenter raised both
  `missing_docs` and the unresolved `@ref`s pointing at it
- Multi-trial `rand(lds, tsteps_per_trial)` threw
  `Attempted to capture and modify outer local variables` instead of sampling.
  The per-trial parameter vectors were assigned from two branches of an `if`
  and then captured by the `tforeach` sampling closure, so Julia boxed them and
  OhMyThreads rejected the closure outright. They are now built in a helper, so
  each name is assigned once
- `fit!(slds, y; tie_emissions=true)` fitted the shared Gaussian emission from an
  uninitialized workspace, usually throwing `PosDefException` from the Cholesky
  in `_aggregate_td_suff_stats!` and otherwise returning nonsense. The
  unit-weight aggregator seeds its buffers from the data-only constant blocks
  (`Σ y y'`, `Σ y`, the observation count, the `uy` blocks), which only the
  LDS/PLDS `fit!` entry points fill — the SLDS never reaches them, because its
  own M-step goes through the weighted aggregator, which needs no constants.
  Both the plain and the grouped tied-emission updates now fill them first
- SLDS `forward_backward` could produce `NaN`s when a regime received ~no
  responsibility at trial starts: its initial-state effective count `init_n`
  underflowed toward zero, so `x0 = init_xy/init_n` and `P0 = S0/init_n` blew up
  to `±Inf`/`NaN`, poisoning `dl.logL` and the whole chain posterior. Setting the
  new initial-state NIW prior (`x0_prior` + `P0_prior`) makes the update degrade
  to the prior (`x0 → μ₀`, `P0 →` its IW mode) instead of dividing by ~0
- SLDS ELBO computation (incorrect sign, among other errors); correctness is now
  tested via the K=1 SLDS ≡ LDS equivalence (#145)
- SLDS posterior sampling drew from the marginals of `q(x)` (a mean-field
  approximation) instead of the joint smoothed posterior (#145)
- SLDS complete-data log-likelihood was inconsistent with the LDS
  implementations; fixed by deduplicating into the shared kernels, with new
  tests against Distributions.jl (#135)
- Poisson LDS ELBO omitted the matrix-normal prior term on the stacked dynamics
  `[A b B]` (#146)
- PPCA M-step computed the `σ²` update from the stale `W` instead of the freshly
  updated one (#147)
- The backtracking line search's cubic interpolation always stepped to the local
  minimizer of the interpolant, even when maximizing (#147)
- `validate_probvec` used a tolerance that could give wrong results for
  lower-precision element types (e.g. `Float32`) (#147)
- `tol_PD` threw a method error when the `tol` keyword did not match the matrix
  element type (#147)
- The LDS model-selection docs example could select the wrong latent
  dimension (#148)

## [0.4.1] - 2026-07-07

### Added
- Add `tview` helper to fix JET errors
- LineSearches added as a new dep, since `Optim` no longer re-exposes `HagerZhang` (#89)

### Changed
- Tests and code now use `Optim` v2 API (#89)
- Updates `Optim` lower bound `2` (#89)
- `Optim` v2's `LBFGS`+`HagerZhang` line search converges to marginally different values for the Poisson observation M-step (~1e-5 magnitude) than v1 did (#89)
- `Symmetrize!` now returns a `Symmetric` matrix (#130)

### Fixed
- Re-enables JET testing after fixing false positives. (#124)
- Fixes lower bound of Julia to 1.10 (was 1.11) (#89)
- Enforced a stable A matrix in the SLDS tests to fix flakyness in CI testing. (#130)
- Enforces symmetry in certain SLDS statistics causing a non-PD issue. (#130)

## [0.4.0] - 2026-07-03

### Added

- CHANGELOG.md to track version history
- Benchmarking CI workflow to track performance over time
- Centralized exports in StateSpaceDynamics.jl main module
- Custom exception types with improved error messages:
  - `DimensionMismatchError` for dimension validation
  - `NotPositiveDefiniteError` for matrix validation
  - `NotSymmetricError` for symmetry checks
  - `InvalidProbabilityVectorError` for probability vector validation
  - `NumericalStabilityError` for numerical issues
- Matrix-normal priors (`MNPrior`) on the stacked dynamics `[A b B]` and
  emission `[C d D]` matrices, giving a full MNIW MAP when paired with `IWPrior`
- Support for exogenous inputs: a dynamics input matrix `B` (`B·u`) and an
  observation input matrix `D` (`D·v`), with explicit `b` / `d` bias vectors
- Hand-rolled Newton smoother (`newton_smooth!`) with a backtracking line
  search for the non-conjugate (Poisson) observation path
- QuickStart example/tutorial
- Auto-formatting CI workflows (`Format.yml`, `Format-PR.yml`)

### Changed

- Refactored model validation system with descriptive exceptions
- Improved error messages across validation functions
- Consolidated all package exports into main module file
- Refactored block tridiagonal inverse implementation
- Renamed the `PoissonObservationModel` field `log_d` to `d`, adopting the
  canonical log-link `λ = exp(C x + d)`
- Standardized `fit_bool` layout: length 6 for the Gaussian path
  (`[x0, P0, A&b&B, Q, C&d&D, R]`) and length 5 for the Poisson path
  (`[x0, P0, A&b, Q, C&d]`)
- Reorganized the LDS source tree, extracting shared emission-agnostic code out
  of `gaussian.jl` into `common.jl` (parameter extraction / FilterSmooth init),
  `simulate.jl` (sampling), `dynamics.jl` (state M-step and state ELBO term), and
  `suff_stats.jl` (sufficient-statistics aggregation); moved the block-tridiagonal
  kernel into `block_tridiagonal.jl`, control-input validation into the validation
  module, and `Base.show` methods into `show.jl`
- Substantially optimized the multi-trial EM hot path: sufficient-statistics
  aggregation that is O(1) in trial length `T` and trial count `N`, a shared
  smoothed-covariance cache for equal-length trials, and an allocation-minimal
  block-tridiagonal smoother
- Clarified the log-likelihood API: the complete-data `log p(x, y)` given a
  trajectory is now `joint_loglikelihood(x, lds, y)`, while `loglikelihood(lds, y)`
  is the marginal (observed-data) `log p(y); a method of `StatsAPI.loglikelihood`,
  consistent with `loglikelihood(ppca, X)`. The marginal throws for Poisson LDS
  (intractable). Replaces the former `filter_loglikelihood`.

### Removed

- **Refocused the package on Linear Dynamical Systems.** Removed the Hidden
  Markov Model, Mixture Model, and standalone emission/regression model families
  along with their tests, documentation, examples, and benchmarks. Specifically:
  - Hidden Markov Models and GLM-HMMs: `HiddenMarkovModel`, `viterbi`,
    `class_probabilities`, the switching Gaussian/Poisson/Bernoulli regression
    models, and AutoRegressive HMM (ARHMM) support
  - Mixture Models: `GaussianMixtureModel`, `PoissonMixtureModel`
  - Emission / regression models: `EmissionModel`, `GaussianEmission`,
    `RegressionEmission`, `GaussianRegressionEmission`,
    `BernoulliRegressionEmission`, `PoissonRegressionEmission`,
    `AutoRegressionEmission`
- The Kalman/RTS smoother as a selectable E-step backend for `fit!` (the
  `kalman_filter` flag on `LinearDynamicalSystem`). All Gaussian fitting now uses
  the block-tridiagonal MAP path. The Kalman filter implementation is retained
  internally for the marginal log-likelihood `loglikelihood(lds, y)`.

### Fixed

- Formatter issues in test suite
- Documentation consistency across modules
- Double-exponential bug in the Poisson observation rate (previously
  `exp(C x + exp(log_d))`, now `exp(C x + d)`)

## [0.3.0] - 2025-11-12

### Added

- Inverse-Wishart priors for covariance matrices (IWPrior)
- Support for MAP estimation with priors on Q, P0, and R matrices
- PoissonLDS prior functionality
- JET.jl static analysis integration in CI
- Comprehensive test suite for prior-based estimation

### Changed

- Refactored LDS code structure for better maintainability
- Split LDS implementations into separate files (gaussian.jl, poisson.jl, types.jl)
- Improved test organization with shared utilities

### Fixed

- Block tridiagonal inverse numerical stability
- Test runner organization

## [0.2.0] - 2024-06-18

### Added

- Documentation improvements
- Enhanced plotting capabilities in examples
- DOI badge and updated README

### Changed

- Updated documentation structure
- Improved badges and metadata

## [0.1.0] - 2024-04-10

### Added

- Initial release of StateSpaceDynamics.jl
- Core implementations:
  - Linear Dynamical Systems (Gaussian and Poisson observations)
  - Hidden Markov Models (Gaussian, Poisson, ARHMM)
  - Mixture Models (Gaussian, Poisson)
  - Switching Linear Dynamical Systems (SLDS)
  - HMM-GLMs (Gaussian, Poisson, Bernoulli)
- Inference algorithms:
  - Kalman filtering and RTS smoothing
  - Laplace approximation for non-conjugate models
  - EM algorithm for parameter estimation
  - Forward-backward algorithm for HMMs
  - Viterbi algorithm for state sequences
- Utilities:
  - K-means initialization
  - Block tridiagonal matrix operations
  - Covariance matrix stabilization
  - Probabilistic PCA preprocessing
- Validation framework
- Comprehensive test suite
- Documentation and examples
- Benchmarking suite

[Unreleased]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/releases/tag/v0.1.0
