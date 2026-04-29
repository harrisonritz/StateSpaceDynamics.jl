module StateSpaceDynamics

import HiddenMarkovModels as HMMs

using ArrayLayouts
using Distributions
using ForwardDiff
using LinearAlgebra
using LineSearches
using Optim
using PDMats
using ProgressMeter
using Random
using SparseArrays
using SpecialFunctions
using StaticArrays
using Statistics
using StatsBase
using StatsFuns

using Base.Threads: @threads, @spawn
using Base.Iterators: partition
using Base: show

# Core types and utilities
include("core/GlobalTypes.jl")
include("models/lds/types.jl")
include("core/Workspaces.jl")
include("core/Utilities.jl")

# Include optimization utilities
include("optimization/linesearch.jl")
include("optimization/newton.jl")

# Linear Dynamical Systems
include("models/lds/kalman.jl")
include("models/lds/gaussian.jl")
include("models/lds/poisson.jl")
include("models/lds/cov_update.jl")
include("models/lds/SLDS.jl")

# Other models
include("models/EmissionModels.jl")
include("models/HiddenMarkovModels.jl")
include("models/MixtureModels.jl")

# Algorithms
include("algorithms/Preprocessing.jl")
include("algorithms/Valid.jl")

# Errors/Exceptions/Validations
export validate_SLDS, validate_LDS, validate_probvec
export DimensionMismatchError, NotPositiveDefiniteError, NotSymmetricError
export InvalidProbabilityVectorError, NumericalStabilityError

# Models and Types
export ProbabilisticPCA,
    SLDS,
    LinearDynamicalSystem,
    GaussianMixtureModel,
    PoissonMixtureModel,
    HiddenMarkovModel,
    Data
export RegressionEmission,
    PoissonRegressionEmission,
    AutoRegressionEmission,
    GaussianEmission,
    GaussianRegressionEmission,
    BernoulliRegressionEmission
export MixtureModel, EmissionModel, DynamicalSystem
export AbstractHMM, AbstractStateModel, AbstractObservationModel
export GaussianStateModel, GaussianObservationModel, PoissonObservationModel
export IWPrior
export CovUpdateCache

# Utilities
export kmeanspp_initialization, kmeans_clustering, fit!, block_tridgm
export block_tridiagonal_inverse, block_tridiagonal_inverse_static
export stabilize_covariance_matrix, valid_Σ, make_posdef!, gaussian_entropy
export random_rotation_matrix
export print_full
export info_update!

# Common functions
export rand, smooth, fit!, loglikelihood, filter_loglikelihood, kmeans_init!, viterbi, class_probabilities

end
