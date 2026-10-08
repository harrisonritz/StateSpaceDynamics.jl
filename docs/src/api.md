# API Reference

This page collects the public API of StateSpaceDynamics.jl in one place. The model
pages ([Linear Dynamical Systems](LinearDynamicalSystems.md) and
[Switching Linear Dynamical Systems](SLDS.md)) intersperse the same docstrings with
theory and usage notes.

```@index
Pages = ["api.md"]
```

## Models

```@docs; canonical = false
LinearDynamicalSystem
SLDS
GaussianStateModel
GaussianObservationModel
PoissonObservationModel
SplineGaussianObservationModel
```

## Monotonic splines

The normalizing-flow layer of a [`SplineGaussianObservationModel`](@ref): a
strictly increasing, analytically invertible rational-quadratic spline per
observation channel. Usable on its own as a parameter transformation.

```@docs; canonical = false
MonotonicWarp
warp_forward
warp_inverse
warp_apply!
warp_unapply!
warp_bounds
warp_channels
warp_bins
warp_nparams
is_identity_warp
refresh_knots!
copy_warp!
```

```@docs
CompositeObservationModel
ProbabilisticPCA
AbstractStateModel
AbstractGaussianStateModel
AbstractObservationModel
```

## Inverse LQR

A state model whose latent is the LQR state–costate pair and whose transition is
constrained to the symplectic form a linear-quadratic control problem implies, so
that fitting recovers the plant and the cost function directly.

```@docs
LQRStateModel
LQRFitFlags
cost_schedule
refresh!
lqr_parameters
lqr_matrix
symplectic_matrix
symplectic_form
symplectic_defect
riccati_solution
lqr_riccati_sequence
closed_loop_dynamics
simulate_lqr
rescale_costate!
free_state_model
hold_state_model
causal_state_model
CausalOptions
plant_dim
```

### Causal (closed-loop) mode

`causal_state_model` fits the finite-horizon **feedback controller** rather than
the two-point boundary-value form: each trial's backward Riccati sweep gives
`P_t` and the feedforward, the costate sits on the Riccati graph
`λ_t = P_t x_t + g_t` up to a costate noise `ν ~ N(0, Ω)`, and the plant noise
`ε ~ N(0, Σ)` arrives after the control is chosen. The innovation is therefore
the one the theory implies — the costate innovation is `P_{t+1}` times the state
innovation plus `ν` — rather than a free mixed-coordinate `Σ`, and the forward
chain is stable, so there is no terminal factor and `rand` samples the model
directly. `slack_drives_state` chooses whether the agent acts on its perturbed
costate (the slack moves the state by `−W_{t+1} S ν`) or `ν` is a pure readout;
`plant_noise` and `costate_noise` choose dense or diagonal `Σ` and `Ω`.

On data from a causal controller this is the generative model, whereas `:lqr`
mode is a misspecified approximation of it whose cost estimates can be biased
(see `docs/dev/lqr/noise_prototype.md` on the `claude/costate-noise-prototype`
branch). Inputs must be constant within a trial, and the mode does not yet
support schedule boundaries or switching models.

### Switching

An [`SLDS`](@ref) whose discrete states are inverse-LQR models switches between
control problems: the state selects which plant and cost generated the
transition. Every discrete state shares one continuous latent path, so they all
carry the same `2n`-dimensional `z = [x; λ]` — the switching is over parameters,
not over dimension. In a switching model the discrete state *is* the cost epoch,
inferred rather than given by `schedule`, so each member carries a single cost.

`free_state_model` supplies a state whose transition is unconstrained rather than
symplectic, which is how a switching model mixes plain linear dynamics with LQR
dynamics under one concrete state-model type.

`hold_state_model` supplies the infinite-horizon counterpart of a control state:
a stationary regulator that holds the state at a reference, with its costate on
the stable manifold of the discrete algebraic Riccati equation. It shares the
plant, control authority, inputs and reference map with the finite-horizon form
and is fitted in the same joint M-step, so `tied_params = [:A, :S]` gives a
"reach" state and a "hold" state one plant with a cost each.

`tied_params = [:structure]` shares the joint block `(A, S, Qc, h, Bu, Gref)`
across discrete states and `[:noise]` shares `Σ`, each fitted jointly from the
states that use it. Individual blocks may also be named — `[:A, :S]` is one plant
with a cost per discrete state, the usual reason to switch at all.

## Priors

```@docs; canonical = false
IWPrior
```

```@docs
MNPrior
```

```@docs
x0_mean_prior
```

```@docs
transition_prior
```

```@docs
set_boundaries!
```

```@docs
set_schedule_boundaries!
set_gref_gate!
LQRSwitch
StateSpaceDynamics.EntryPrior
```

```@docs
banded_transition
banded_transition_prior
median_dwell_stay
```

## Ancillary parameter dependencies

```@docs
group_labels
group_parameter
group_variant
set_group_seeds!
set_depends_on!
```

## Sampling

```@docs; canonical = false
Random.rand(rng::AbstractRNG, lds::LinearDynamicalSystem{T,S,O}, tsteps::Integer) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:AbstractObservationModel{T}}
Random.rand(rng::AbstractRNG, slds::SLDS{T,S,O}, tsteps::Integer) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
```

```@docs
Random.rand(rng::AbstractRNG, lds::LinearDynamicalSystem{T,S,O}, tsteps::Integer) where {T<:Real,S<:LQRStateModel{T},O<:AbstractObservationModel{T}}
Random.rand(rng::AbstractRNG, ppca::ProbabilisticPCA, n::Int)
```

## Smoothing and fitting

```@docs; canonical = false
smooth
fit!(lds::LinearDynamicalSystem{T,S,O}, y::StateSpaceDynamics.CompositeObservations{T}; max_iter::Int=100, tol::Float64=1e-6, progress::Bool=true) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:StateSpaceDynamics.QuadraticEmission{T}}
fit!(slds::SLDS{T,S,O}, y::StateSpaceDynamics.CompositeObservations{T}; max_iter::Int=50, progress::Bool=true) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
```

```@docs
fit!(plds::LinearDynamicalSystem{T,S,O}, y::StateSpaceDynamics.CompositeObservations{T}) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:StateSpaceDynamics.NonQuadraticEmission{T}}
fit!(lds::LinearDynamicalSystem{T,S,O}, y::StateSpaceDynamics.CompositeObservations{T}) where {T<:Real,S<:LQRStateModel{T},O<:StateSpaceDynamics.QuadraticEmission{T}}
fit!(lds::LinearDynamicalSystem{T,S,O}, y::StateSpaceDynamics.CompositeObservations{T}) where {T<:Real,S<:LQRStateModel{T},O<:StateSpaceDynamics.NonQuadraticEmission{T}}
fit!(ppca::ProbabilisticPCA, X::AbstractMatrix{T}, max_iters::Int=100, tol::Float64=1e-6) where {T<:Real}
FitTrace
```

## Likelihoods and ELBO

`elbo` is the public, allocating entry point (all three models); the `elbo!`
variants are the workspace-based internals it wraps.

```@docs
elbo
trial_elbos
terminal_logz
terminal_normalizer
loglikelihood(lds::LinearDynamicalSystem{T,SM,OM}, y::StateSpaceDynamics.Observations{T}) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
loglikelihood(plds::LinearDynamicalSystem{T,S,O}, y) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:PoissonObservationModel{T}}
loglikelihood(lds::LinearDynamicalSystem{T,SM,OM}, y::NamedTuple) where {T<:Real,SM<:AbstractGaussianStateModel{T},OM<:CompositeObservationModel{T,true}}
loglikelihood(lds::LinearDynamicalSystem{T,S,O}, y::StateSpaceDynamics.Observations{T}) where {T<:Real,S<:LQRStateModel{T},O<:GaussianObservationModel{T}}
loglikelihood(slds::SLDS, y)
loglikelihood(ppca::ProbabilisticPCA, X::AbstractMatrix{T}) where {T<:Real}
elbo!
```

## Model comparison

Latent-free baselines to score a fitted model against, plus the StatsAPI
methods built on them.

```@docs
AffineNullModel
fit!(null::StateSpaceDynamics.AffineNullModel, y)
loglikelihood(null::StateSpaceDynamics.AffineNullModel, y)
r2(lds::LinearDynamicalSystem{T,SM,OM}, y, variant::Symbol) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
nullloglikelihood(lds::LinearDynamicalSystem{T,SM,OM}, y) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
nobs(lds::LinearDynamicalSystem{T}, y) where {T<:Real}
```

## Validation

```@docs
validate_LDS
validate_SLDS
validate_probvec
DimensionMismatchError
NotPositiveDefiniteError
NotSymmetricError
InvalidProbabilityVectorError
NumericalStabilityError
```

## Utilities

```@docs
random_rotation_matrix
gaussian_entropy
valid_Σ
block_tridgm
print_full
info_update!
CovUpdateCache
```

## Internals

Documented internal methods, listed here for completeness. These are not part
of the public API and may change between releases.

```@docs
fit!(dl::StateSpaceDynamics.SLDSDiscreteLayer{T}, fb_storage::StateSpaceDynamics.HMMs.ForwardBackwardStorage, obs_seq::AbstractVector) where {T<:Real}
```
