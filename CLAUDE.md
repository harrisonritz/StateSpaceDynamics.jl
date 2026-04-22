# StateSpaceDynamics.jl — Developer Guide

## Overview

StateSpaceDynamics.jl is a Julia package for probabilistic state space models (SSMs), focused on efficient CPU-based EM learning. Primary motivation is neuroscience (trialized data, Poisson/count observations), but models are general-purpose. Version 0.3.0, requires Julia 1.11+.

Core design principle: **analytically derived gradients and Hessians** exploiting block tridiagonal structure for O(T) inference, rather than automatic differentiation.

## Source Layout

```
src/
├── StateSpaceDynamics.jl       # Module entry point — imports and exports
├── core/
│   ├── GlobalTypes.jl          # Abstract type hierarchy + key structs (FilterSmooth, ForwardBackward)
│   ├── Utilities.jl            # Matrix ops, block tridiagonal solvers, k-means, entropy
│   └── Workspaces.jl           # Pre-allocated workspaces (avoid allocations in EM loops)
├── optimization/
│   ├── newton.jl               # Newton's method optimizer (used in MAP estimation)
│   └── linesearch.jl           # Backtracking line search with Armijo condition
├── algorithms/
│   ├── Preprocessing.jl        # ProbabilisticPCA (EM-based)
│   └── Valid.jl                # Validation functions + custom exception types
└── models/
    ├── lds/
    │   ├── types.jl            # GaussianStateModel, GaussianObservationModel, PoissonObservationModel, LinearDynamicalSystem, SLDS
    │   ├── gaussian.jl         # Gaussian LDS: smoothing, gradient, Hessian, EM M-step
    │   ├── poisson.jl          # Poisson LDS: log-likelihood, gradient, Hessian
    │   └── SLDS.jl             # Switching LDS: sampling, inference, likelihood
    ├── EmissionModels.jl       # Gaussian, Poisson, Bernoulli, AR emission models for HMMs
    ├── HiddenMarkovModels.jl   # HMM with pluggable emissions; wraps HiddenMarkovModels.jl
    └── MixtureModels.jl        # GaussianMixtureModel, PoissonMixtureModel
```

## Type Hierarchy

```
DynamicalSystem
├── LinearDynamicalSystem{T, S<:AbstractStateModel, O<:AbstractObservationModel}
└── SLDS{T, S, O, TM, ISV}

AbstractStateModel{T}
└── GaussianStateModel{T, M, V}          # x_t = A*x_{t-1} + b + ε, ε ~ N(0,Q)

AbstractObservationModel{T}
├── GaussianObservationModel{T, M, V}    # y_t = C*x_t + d + η, η ~ N(0,R)
└── PoissonObservationModel{T, M, V}     # y_t ~ Poisson(exp(C*x_t + log_d))

AbstractHMM
├── HiddenMarkovModel{T, V, M, VE}       # Discrete HMM with pluggable EmissionModel
└── SLDSDiscreteLayer{T, TM, TV}         # Wraps SLDS discrete layer for HMM interface

MixtureModel
├── GaussianMixtureModel{T, M, V}
└── PoissonMixtureModel

EmissionModel
├── GaussianEmission
├── RegressionEmission
│   ├── GaussianRegressionEmission
│   ├── PoissonRegressionEmission
│   ├── BernoulliRegressionEmission
│   └── AutoRegressionEmission
```

## Key Algorithms

### Inference (E-step)

**Gaussian LDS:** Direct MAP optimization of complete-data log-likelihood. The negative Hessian is block tridiagonal (Paninski 2010 approach), enabling O(T) exact smoothing equivalent to Kalman/RTS smoother.

**Poisson/non-Gaussian LDS:** Laplace approximation — MAP path via Newton's method, then posterior approximated as Gaussian centered at MAP with covariance = inverse of negative Hessian. Exact same code path as Gaussian when observations are Gaussian.

**SLDS:** Variational Laplace EM (vLEM). Per-trial responsibility weights `w[k,t]` from discrete HMM layer; continuous layer does weighted smoothing per component.

**HMMs:** Delegates to `HiddenMarkovModels.jl` backend for forward-backward, Viterbi, etc.

### Parameter Learning (M-step)

All M-step updates are analytical closed forms:
- LDS: `A`, `Q`, `C`, `R`, `x0`, `P0` updated from expected sufficient statistics (`E_z`, `E_zz`, `E_zz_prev`)
- HMM: transition matrix, emission parameters updated from `γ` (state occupancy) and `ξ` (pairwise occupancy)
- GMM: means, covariances, mixing weights from responsibilities

### Block Tridiagonal Structure

The Hessian of the complete-data log-likelihood is block tridiagonal. Key utilities:
- `block_tridiagonal_inverse(A, B, C)` — Rybicki & Hummer algorithm for exact inverse
- `block_tridgm(main, upper, lower)` — Build sparse representation
- `block_tridiagonal_solve!(x, A, B, C, b, ws)` — Solve block tridiagonal system

## Workspace Pattern

**Critical for performance:** Pre-allocated workspaces avoid GC pressure during EM iterations.

```julia
# Pattern used throughout the codebase
ws = SmoothWorkspace(lds, T, ntrials)      # allocate once
smooth!(lds, fs, y, ws)                    # reuse across EM iterations
```

Key workspace types:
- `BlockTridiagonalWorkspace{T}` — Hessian blocks, sparse nzval map, LU buffers
- `SmoothWorkspace{T}` — Full workspace for LDS smoothing: Cholesky caches, gradient/Hessian buffers, ELBO accumulators
- `LDSConstantCache{T}` — Per-component Cholesky-derived constants (log-det terms) for SLDS
- `SLDSSmoothWorkspace{T}` — SLDS workspace containing per-component `LDSConstantCache` objects

## `fit_bool` Parameter

`LinearDynamicalSystem` takes a `fit_bool::Vector{Bool}` (length 6) controlling which parameters are updated during EM:
```
[fit_A, fit_Q, fit_C, fit_R, fit_x0, fit_P0]
```

## Multi-Trial (Trialized) Data

Most models support multi-trial data as `Vector{Matrix}` where each matrix is `obs_dim × T_i` (trials can have different lengths). `TrialFilterSmooth` wraps a vector of `FilterSmooth` objects, one per trial.

## Conventions

- Functions mutating arguments end in `!` (e.g., `smooth!`, `Gradient!`, `Hessian!`, `fit!`)
- Internal workspace-using functions often prefixed with `_` (e.g., `_loglikelihood_ws`)
- Observations: matrices are `obs_dim × T` (columns = time steps)
- Latent states: matrices are `latent_dim × T`
- `E_z[t]` = `E[x_t]`, `E_zz[t]` = `E[x_t x_t']`, `E_zz_prev[t]` = `E[x_t x_{t-1}']`

## Key Dependencies

| Package | Role |
|---------|------|
| `HiddenMarkovModels.jl` | Backend for HMM forward-backward, Viterbi (recently migrated to this) |
| `LinearAlgebra` | BLAS/LAPACK, Cholesky, eigendecompositions |
| `SparseArrays` | Block tridiagonal Hessian storage |
| `StaticArrays` | `SMatrix` in `block_tridiagonal_inverse_static` for small-dim speedup |
| `ForwardDiff` | Available but analytically derived gradients are preferred |
| `Optim` / `LineSearches` | Newton optimizer infrastructure |
| `Distributions` | Probability distributions |

## Testing

Tests organized under `test/` by model type:
```
test/
├── runtests.jl
├── helper_functions.jl
├── HiddenMarkovModels/
├── LinearDynamicalSystems/
├── MixtureModels/
├── Preprocessing/
├── RegressionModels/
├── Utilities/
├── Validation/
└── test_data/
```

Run tests: `julia --project -e 'using Pkg; Pkg.test()'`

## Active Development Areas (as of 2025)

- HMM backend recently migrated to `HiddenMarkovModels.jl` (commit `6008c1d`)
- SLDS inference via vLEM is implemented; rSLDS is not yet
- Block tridiagonal solver does not yet fully exploit banded structure — future optimization target
- GPU/AD support planned but not implemented
- Remaining model gaps: Binomial/Negative Binomial HMMs, Negative Binomial GLMs, PFLDS, rSLDS

## Common Pitfalls

- Covariance matrices must be symmetric positive definite; use `stabilize_covariance_matrix` or `make_posdef!` when numerics are suspect
- Workspace objects must be recreated if model dimensions change
- `fit_bool` must match the number of free parameters (always length 6 for LDS)
- Laplace approximation quality degrades if MAP optimization does not converge — check Newton solver tolerance
