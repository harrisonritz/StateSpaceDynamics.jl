module StateSpaceDynamics

import HiddenMarkovModels as HMMs

using Distributions
using LinearAlgebra
using PDMats
using Random
using SparseArrays

using Optim: Optim, optimize, LBFGS
using LineSearches: HagerZhang
using ProgressMeter: Progress, next!, finish!
using SpecialFunctions: loggamma
using Statistics: mean
using StatsAPI: StatsAPI
import StatsAPI: loglikelihood, fit!

using OhMyThreads: tforeach, tmapreduce
using Base.Iterators: partition
using Base: show

# Model-agnostic numerical kernels (no package types — reusable primitives).
include("numerics/linalg.jl")
include("numerics/optimization.jl")        # line search + Newton
include("numerics/block_tridiagonal.jl")   # BTD workspace + solver/inverse
include("numerics/cov_update.jl")          # info_update! + CovUpdateCache

# Conjugate priors — defined first because model structs reference IWPrior/MNPrior
# in their field type annotations.
include("stats/priors.jl")

# Model definitions + inference-state containers.
include("lds/types.jl")                             # abstract types, model structs, SLDS
include("lds/workspaces.jl")                        # FilterSmooth / SufficientStatistics / workspaces
include("lds/hamiltonian_types.jl")                 # inverse-LQR state model + derived cache
include("lds/parameter_groups.jl")                  # `depends_on` -> per-group parameter variants
include("utils/show.jl")
include("utils/validation.jl")

# Shared latent inference machinery.
include("stats/preprocessing.jl")           # PPCA (standalone model)
include("stats/sufficient_statistics.jl")
include("stats/simulate.jl")

# latents models (LDS, PLDS, SLDS) + inference machinery (E-step).
include("lds/continuous_latents.jl")                # state-model Q-term + state M-step
include("lds/hamiltonian_latents.jl")               # inverse-LQR E-step kernels

# Observation models + composite / standalone models.
include("lds/gaussian_observations.jl")
include("lds/poisson_observations.jl")
include("lds/poisson_emission_mstep.jl")            # row-wise Newton emission M-step
include("lds/composite_observations.jl")            # several emissions on one latent state

# Grouped (`depends_on`) M-step + ELBO machinery, shared by LDS / PLDS / SLDS.
include("lds/grouped_em.jl")

# Fitting Functions
include("lds/fit_LDS.jl")
include("lds/fit_PLDS.jl")
include("lds/fit_SLDS.jl")

# Inverse-LQR M-step + driver glue. After the drivers, since it specialises
# their `estep!` / `elbo!` / `mstep!` / `fit!` hooks.
include("lds/hamiltonian_mstep.jl")
include("lds/fit_hamiltonian.jl")

# Errors/Exceptions/Validations
export validate_SLDS, validate_LDS, validate_probvec
export DimensionMismatchError, NotPositiveDefiniteError, NotSymmetricError
export InvalidProbabilityVectorError, NumericalStabilityError

# Models and Types
export ProbabilisticPCA, SLDS, LinearDynamicalSystem
export AbstractStateModel, AbstractGaussianStateModel, AbstractObservationModel
export GaussianStateModel, GaussianObservationModel, PoissonObservationModel
export CompositeObservationModel
export IWPrior, MNPrior, x0_mean_prior
export CovUpdateCache

# Inverse LQR (Hamiltonian latents)
export HamiltonianStateModel, HamiltonianFitFlags, cost_schedule, refresh!
export hamiltonian_matrix, symplectic_matrix, symplectic_form, symplectic_defect
export lqr_parameters, riccati_solution, closed_loop_dynamics, rescale_costate!
export simulate_lqr, lqr_riccati_sequence

# Ancillary parameter dependencies (`depends_on`)
export group_labels, group_parameter, set_group_seeds!, set_depends_on!

# Utilities
export block_tridgm
export valid_Σ, gaussian_entropy
export random_rotation_matrix
export print_full
export info_update!
export tview

# Common functions
export rand, smooth, fit!, loglikelihood, elbo, elbo!

end
