# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed

- Refactored model validation system with descriptive exceptions
- Improved error messages across validation functions
- Consolidated all package exports into main module file
- Refactored block tridiagonal inverse implementation

### Fixed

- Formatter issues in test suite
- Documentation consistency across modules

## [0.3.0] - 2025-01-XX

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

## [0.2.0] - 2024-XX-XX

### Added

- Documentation improvements
- Enhanced plotting capabilities in examples
- DOI badge and updated README

### Changed

- Updated documentation structure
- Improved badges and metadata

## [0.1.1] - 2024-XX-XX

### Fixed

- Initial bug fixes and improvements

## [0.1.0] - 2024-XX-XX

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

[Unreleased]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/depasquale-lab/StateSpaceDynamics.jl/releases/tag/v0.1.0
