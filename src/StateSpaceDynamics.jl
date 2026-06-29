module StateSpaceDynamics

import HiddenMarkovModels as HMMs

using ArrayLayouts
import BSplines
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
using Statistics
using StatsAPI: StatsAPI
using StatsBase
using StatsFuns

using Base.Threads: @threads, @spawn
using Base.Iterators: partition
using Base: show

# Core types and utilities
include("core/GlobalTypes.jl")
include("core/priors.jl")
include("models/lds/types.jl")
include("core/Workspaces.jl")
include("core/Utilities.jl")

# Include optimization utilities
include("optimization/linesearch.jl")
include("optimization/newton.jl")

# Linear Dynamical Systems
include("models/lds/cov_update.jl")
include("models/lds/kalman.jl")
include("models/lds/gaussian.jl")
include("models/lds/poisson.jl")
include("models/lds/SLDS.jl")

# Input basis library (B-spline, Fourier, raised cosine, polynomial)
include("basis/Bases.jl")
include("basis/BSpline.jl")
include("basis/Fourier.jl")
include("basis/RaisedCosine.jl")
include("basis/Polynomial.jl")

# Algorithms
include("algorithms/Preprocessing.jl")
include("algorithms/Valid.jl")

# Errors/Exceptions/Validations
export validate_SLDS, validate_LDS, validate_probvec
export DimensionMismatchError, NotPositiveDefiniteError, NotSymmetricError
export InvalidProbabilityVectorError, NumericalStabilityError

# Models and Types
export ProbabilisticPCA, SLDS, LinearDynamicalSystem, Data
export AbstractStateModel, AbstractObservationModel
export GaussianStateModel, GaussianObservationModel, PoissonObservationModel
export IWPrior, MNPrior
export CovUpdateCache

# Utilities
export fit!, block_tridgm
export valid_Σ, gaussian_entropy
export random_rotation_matrix
export print_full
export info_update!

# Input bases
export AbstractInputBasis
export BSpline, Fourier, RaisedCosineLinear, RaisedCosineLog, Polynomial
export apply!, get_penalty, n_bases, evaluate_basis

# Common functions
export rand, smooth, fit!, loglikelihood, filter_loglikelihood

end
