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
CompositeObservationModel
```

```@docs
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
plant_dim
```

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

## Ancillary parameter dependencies

```@docs
group_labels
group_parameter
set_group_seeds!
set_depends_on!
```

## Sampling

```@docs; canonical = false
Random.rand(rng::AbstractRNG, lds::LinearDynamicalSystem{T,S,O}, tsteps::Integer) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
Random.rand(rng::AbstractRNG, slds::SLDS{T,S,O}, tsteps::Integer) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
```

```@docs
Random.rand(rng::AbstractRNG, ppca::ProbabilisticPCA, n::Int)
```

## Smoothing and fitting

```@docs; canonical = false
smooth
fit!(lds::LinearDynamicalSystem{T,S,O}, y::Union{AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple}; max_iter::Int=100, tol::Float64=1e-6, progress::Bool=true) where {T<:Real,S<:GaussianStateModel{T},O<:StateSpaceDynamics.QuadraticEmission{T}}
fit!(slds::SLDS{T,S,O}, y::Union{AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple}; max_iter::Int=50, progress::Bool=true) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
```

```@docs
fit!(plds::LinearDynamicalSystem{T,S,O}, y::Union{AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple}) where {T<:Real,S<:GaussianStateModel{T},O<:StateSpaceDynamics.NonQuadraticEmission{T}}
fit!(ppca::ProbabilisticPCA, X::AbstractMatrix{T}, max_iters::Int=100, tol::Float64=1e-6) where {T<:Real}
```

## Likelihoods and ELBO

`elbo` is the public, allocating entry point (all three models); the `elbo!`
variants are the workspace-based internals it wraps.

```@docs
elbo
loglikelihood(lds::LinearDynamicalSystem{T,SM,OM}, y::Union{AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}}}) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
loglikelihood(plds::LinearDynamicalSystem{T,S,O}, y) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
loglikelihood(slds::SLDS, y)
loglikelihood(ppca::ProbabilisticPCA, X::AbstractMatrix{T}) where {T<:Real}
elbo!(lds::LinearDynamicalSystem{T,S,O}, suf::StateSpaceDynamics.SufficientStatistics{T}, sws::StateSpaceDynamics.SmoothWorkspace{T}, total_entropy::T) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
elbo!(plds::LinearDynamicalSystem{T,S,O}, suf::StateSpaceDynamics.SufficientStatistics{T}, tfs::StateSpaceDynamics.TrialFilterSmooth{T}, data::StateSpaceDynamics.Data{T}, sws_pool::Vector{StateSpaceDynamics.SmoothWorkspace{T}}) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
elbo!(slds::SLDS{T,S,O}, tfs::StateSpaceDynamics.TrialFilterSmooth{T}, fb_storage::StateSpaceDynamics.HMMs.ForwardBackwardStorage, y::AbstractVector{<:AbstractMatrix{T}}, slds_ws::StateSpaceDynamics.SLDSSmoothWorkspace{T}) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
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
