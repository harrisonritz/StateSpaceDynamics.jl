"""
Abstract type for Dynamical Systems. I.e. LDS, etc.
"""
abstract type AbstractStateModel{T<:Real} end
abstract type AbstractObservationModel{T<:Real} end

"""
    Data{T<:Real}

Container for the observed data passed to the Kalman path: observations `y`,
dynamics (latent) inputs `ux`, and observation inputs `uy`, each `(dim, tsteps, ntrials)`
(input fields may have zero rows when no controls are supplied).
"""
Base.@kwdef struct Data{T<:Real}
    y::Array{T,3}
    ux::Array{T,3}
    uy::Array{T,3}
end

"""
    GaussianStateModel{T<:Real,TA,TQ,Tb,Tx0,TP0,TB} <: AbstractStateModel{T}

Represents the state model of a Linear Dynamical System with Gaussian noise.

State evolution:
```math
x_1           ~ N(x_0, P_0)
x_{t+1} | x_t ~ N(A x_t + b + B ux_t, Q)
```
where `B·ux_t` is present only when `B` is supplied (i.e., has nonzero columns).

Each numeric parameter may be a plain array (trial-invariant, the default and
fast path) or wrapped in an [`Indexed`](@ref) — [`Static`](@ref) (shared) or
[`Varying`](@ref) (one value per trial group). The dynamics-mean block `[A b B]`
must share the same indexing (see `validate_LDS`); `Q`, `x0`, `P0` may each be
indexed independently. `T` is the scalar element type.

# Fields
- `A`: Transition matrix (size `latent_dim × latent_dim`).
- `Q`: Process noise covariance matrix.
- `b`: Bias vector (length `latent_dim`).
- `x0`: Initial state mean (length `latent_dim`).
- `P0`: Initial state covariance (size `latent_dim × latent_dim`).
- `B`: Optional dynamics input matrix (`latent_dim × ux_dim`). When supplied,
    inputs `ux` must be passed to `fit!`/`smooth!` via a keyword argument. Shares
    the `[A b B]` block indexing.
- `Q_prior::Union{Nothing,IWPrior{T}} = nothing`: Optional Inverse-Wishart prior on `Q`. If set, MAP updates use its mode.
- `P0_prior::Union{Nothing,IWPrior{T}} = nothing`: Optional Inverse-Wishart prior on `P0`. If set, MAP updates use its mode.
- `AB_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing`: Optional matrix-normal prior on
    the stacked dynamics matrix `[A B]`. Pair with `Q_prior` for a full MNIW prior on `(AB, Q)`.
    Prior matrices are stored as plain `Matrix{T}`.
"""
mutable struct GaussianStateModel{T<:Real,TA,TQ,Tb,Tx0,TP0,TB} <: AbstractStateModel{T}
    A::TA
    Q::TQ
    b::Tb
    x0::Tx0
    P0::TP0
    B::TB
    Q_prior::Union{Nothing,IWPrior{T}}
    P0_prior::Union{Nothing,IWPrior{T}}
    AB_prior::Union{Nothing,MNPrior{T,Matrix{T}}}
end

"""
    GaussianStateModel(; A, Q, b, x0, P0, B=nothing, Q_prior=nothing,
                       P0_prior=nothing, AB_prior=nothing)

Keyword constructor. The scalar type `T` is inferred from `A` (via
[`param_eltype`](@ref)); each field may be a plain array or an [`Indexed`](@ref)
wrapper. `B` defaults to an eltype-preserving zero-column matrix (no dynamics
input).
"""
function GaussianStateModel(;
    A, Q, b, x0, P0, B=nothing, Q_prior=nothing, P0_prior=nothing, AB_prior=nothing
)
    T = param_eltype(A)
    if B === nothing
        B = zeros(T, size(at(A, 1), 1), 0)
    end
    return GaussianStateModel{
        T,typeof(A),typeof(Q),typeof(b),typeof(x0),typeof(P0),typeof(B)
    }(
        A, Q, b, x0, P0, B, Q_prior, P0_prior, AB_prior
    )
end

"""
    GaussianObservationModel{T<:Real,TC,TR,Td,TD} <: AbstractObservationModel{T}

Represents the observation model of a Linear Dynamical System with Gaussian noise.

Each numeric parameter may be a plain array (trial-invariant, the default) or an
[`Indexed`](@ref) wrapper ([`Static`](@ref)/[`Varying`](@ref)). The emission-mean
block `[C d D]` must share the same indexing; `R` may be indexed independently,
except that when `C` is [`Varying`](@ref) with differing `obs_dim` across groups
(multi-session stitching), `R` and `d` must share `C`'s grouping (see
`validate_LDS`). `T` is the scalar element type.

# Fields
- `C`: Observation matrix of size `(obs_dim × latent_dim)`.
- `R`: Observation noise covariance of size `(obs_dim × obs_dim)`.
- `d`: Bias vector of length `(obs_dim)`.
- `D`: Optional observation input matrix (`obs_dim × uy_dim`); shares the `[C d D]`
    block indexing.
- `R_prior::Union{Nothing, IWPrior{T}} = nothing`: Optional Inverse-Wishart prior for `R`.
- `CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing`: Optional matrix-normal prior on
    the stacked emission matrix `[C D]`. Pair with `R_prior` for a full MNIW prior on `(CD, R)`.
"""
mutable struct GaussianObservationModel{T<:Real,TC,TR,Td,TD} <: AbstractObservationModel{T}
    C::TC
    R::TR
    d::Td
    D::TD
    R_prior::Union{Nothing,IWPrior{T}}
    CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}}
end

"""
    GaussianObservationModel(; C, R, d, D=nothing, R_prior=nothing, CD_prior=nothing)

Keyword constructor. `T` is inferred from `C`; each field may be a plain array or
an [`Indexed`](@ref) wrapper. `D` defaults to an eltype-preserving zero-column
matrix (no observation input).
"""
function GaussianObservationModel(;
    C, R, d, D=nothing, R_prior=nothing, CD_prior=nothing
)
    T = param_eltype(C)
    if D === nothing
        D = zeros(T, size(at(C, 1), 1), 0)
    end
    return GaussianObservationModel{T,typeof(C),typeof(R),typeof(d),typeof(D)}(
        C, R, d, D, R_prior, CD_prior
    )
end

# Positional convenience constructors (State). Plain arrays only; delegate to the
# keyword constructor.
function GaussianStateModel(
    A::M, Q::M, b::V, x0::V, P0::M
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    return GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
end

function GaussianStateModel(A::M, Q::M, B::M, P0::M) where {T<:Real,M<:AbstractMatrix{T}}
    return GaussianStateModel(;
        A=A, Q=Q, b=zeros(T, size(A, 1)), x0=zeros(T, size(A, 1)), P0=P0, B=B
    )
end

# Positional convenience constructors (Observation).
function GaussianObservationModel(
    C::M, R::M, d::V
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    return GaussianObservationModel(; C=C, R=R, d=d)
end

function GaussianObservationModel(C::M, R::M, D::M) where {T<:Real,M<:AbstractMatrix{T}}
    return GaussianObservationModel(; C=C, R=R, d=zeros(T, size(C, 1)), D=D)
end

"""
    PoissonObservationModel{
        T<:Real,
        M<:AbstractMatrix{T},
        V<:AbstractVector{T}
    } <: AbstractObservationModel{T}

Represents the observation model of a Linear Dynamical System with Poisson observations,
with canonical log-link:

```math
λ_t = exp(C x_t + d)
```

`d` is the standard Poisson-GLM intercept — unconstrained in ℝ; positivity of the rate
`λ` is provided by the `exp`. The previously-named `log_d` field was a misnomer that
caused a double-exp bug (`exp(C x + exp(log_d))`); see git log for the fix.

# Fields
- `C::AbstractMatrix{T}`: Observation matrix of size `(obs_dim × latent_dim)`. Maps latent
    states into observation space.
- `d::AbstractVector{T}`: Per-neuron baseline log-rate (length `obs_dim`). Free in ℝ.
- `CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing`: Optional matrix-normal prior on
    the stacked emission matrix `[C d]` (treated as a single regression of `log λ` on
    `[x; 1]`). Prior matrices are stored as plain `Matrix{T}`, decoupled from `C`'s storage
    type `M`. `M₀` and `Λ` have shapes `(obs_dim, latent_dim+1)` and
    `(latent_dim+1, latent_dim+1)` respectively. Unlike the Gaussian path there is no IW
    counterpart since Poisson has no observation-noise covariance — this is an MN-only
    prior contributing `½ tr(([C d] - M₀) Λ ([C d] - M₀)')` to the LBFGS objective.
"""
mutable struct PoissonObservationModel{T<:Real,TC,Td} <: AbstractObservationModel{T}
    C::TC
    d::Td
    CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}}
end

"""
    PoissonObservationModel(; C, d, CD_prior=nothing)

Keyword constructor. `T` is inferred from `C`; `C`/`d` may be plain arrays or
[`Indexed`](@ref) wrappers (sharing the `[C d]` block indexing).
"""
function PoissonObservationModel(; C, d, CD_prior=nothing)
    T = param_eltype(C)
    return PoissonObservationModel{T,typeof(C),typeof(d)}(C, d, CD_prior)
end

# 2-arg positional convenience constructor.
function PoissonObservationModel(
    C::M, d::V
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    return PoissonObservationModel(; C=C, d=d)
end

# ---------------------------------------------------------------------------
# Dimension / observation-family traits. A parameter may be a plain array or an
# `Indexed` wrapper, so dimensions are queried through `at(param, 1)` (a
# representative value). For `Varying` observation params whose per-group
# `obs_dim` differs, this returns the first group's dimension; per-regime code
# sizes workspaces from each group's own value.
# ---------------------------------------------------------------------------
_lds_latent_dim(sm::AbstractStateModel) = size(at(sm.A, 1), 1)
_lds_obs_dim(om::AbstractObservationModel) = size(at(om.C, 1), 1)

_is_poisson_like(::AbstractObservationModel) = false
_is_poisson_like(::PoissonObservationModel) = true

"""
    LinearDynamicalSystem{T<:Real, S<:AbstractStateModel{T}, O<:AbstractObservationModel{T}}

Represents a unified Linear Dynamical System with customizable state and observation models.

# Fields
- `state_model::S`: The state model (e.g., GaussianStateModel)
- `obs_model::O`: The observation model (e.g., GaussianObservationModel or
    PoissonObservationModel)
- `latent_dim::Int`: Dimension of the latent state
- `obs_dim::Int`: Dimension of the observations
- `fit_bool::Vector{Bool}`: Vector indicating which parameters to fit during optimization.
    Length 6 for the Gaussian path (`[x0, P0, A&b&B, Q, C&d&D, R]`); the M-step
    regression fits each row jointly. Length 5 for the Poisson path
    (`[x0, P0, A&b, Q, C&d]`).
"""
Base.@kwdef struct LinearDynamicalSystem{
    T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}
}
    state_model::S
    obs_model::O
    latent_dim::Int
    obs_dim::Int
    state_input_dim::Int = 0
    obs_input_dim::Int = 0
    fit_bool::Vector{Bool}
end

function LinearDynamicalSystem(
    state_model::S, obs_model::O; fit_bool::Union{Vector{Bool},Nothing}=nothing
) where {T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}}

    # Infer dimensions via the `_lds_*_dim` traits and `at`, so plain and
    # `Indexed` (Static/Varying) parameters are handled uniformly. Input
    # dimensions (columns of `B`/`D`) are shared across groups by construction.
    latent_dim = _lds_latent_dim(state_model)
    obs_dim = _lds_obs_dim(obs_model)
    state_input_dim = if hasproperty(state_model, :B) && !isnothing(state_model.B)
        size(at(state_model.B, 1), 2)
    else
        0
    end
    obs_input_dim = if hasproperty(obs_model, :D) && !isnothing(obs_model.D)
        size(at(obs_model.D, 1), 2)
    else
        0
    end

    # Set default fit_bool based on observation model type. The M-step fits
    # [A b B] and [C d D] as joint regressions, so the Gaussian layout is length 6.
    if fit_bool === nothing
        if _is_poisson_like(obs_model)
            # Poisson: [x0, P0, A&b, Q, C&d] (5 parameters)
            fit_bool = [true, true, true, true, true]
        else
            # Gaussian (BTD): [x0, P0, A&b&B, Q, C&d&D, R] (6 parameters)
            fit_bool = [true, true, true, true, true, true]
        end
    end

    # Create the LDS
    lds = LinearDynamicalSystem{T,S,O}(
        state_model,
        obs_model,
        latent_dim,
        obs_dim,
        state_input_dim,
        obs_input_dim,
        fit_bool,
    )

    # Validate the constructed LDS (throws on error)
    validate_LDS(lds)

    return lds
end

"""
    SLDS{T,S,O,TM,ISV}

A Switching Linear Dynamical System (SLDS). A hierarchical time-series model of the form:

```math
z_t | z_{t-1} ~ Categorical(A_{z_{t-1}, :})
x_t | x_{t-1}, z_t ~ N(A^{(z_t)} x_{t-1} + b^{(z_t)}, Q^{(z_t)})
y_t | x_t, z_t ~ N(C^{(z_t)} x_t + d^{(z_t)}, R^{(z_t)})
```

# Fields
- `A::TM`: Transition matrix for the discrete states (K x K)
- `πₖ::ISV`: Initial state distribution for the discrete states (K-dimensional vector)
- `LDSs::Vector{LinearDynamicalSystem{T,S,O}}`: Vector of K Linear Dynamical Systems, one for each discrete state
"""
@kwdef mutable struct SLDS{
    T<:Real,
    S<:AbstractStateModel,
    O<:AbstractObservationModel,
    TM<:AbstractMatrix{T},
    ISV<:AbstractVector{T},
}
    A::TM
    πₖ::ISV
    LDSs::Vector{LinearDynamicalSystem{T,S,O}}
end

"""
    SLDSDiscreteLayer{T,TM,TV}

Thin wrapper satisfying the `HiddenMarkovModels.AbstractHMM` interface for the discrete
switching layer of an SLDS.  The `logL` matrix (K×T) is pre-filled with per-state
log-likelihoods before each forward-backward call; `obs_seq` is then just `1:T` (timestep
indices) so that `obs_logdensities!` can look up the correct column.

Fields `A` and `πₖ` are kept as references to the parent SLDS matrices so that in-place
M-step updates are automatically reflected.
"""
mutable struct SLDSDiscreteLayer{T<:Real,TM<:AbstractMatrix{T},TV<:AbstractVector{T}} <:
               HMMs.AbstractHMM
    A::TM            # K×K row-stochastic transition matrix
    πₖ::TV           # K initial-state distribution
    logL::Matrix{T}  # K×T pre-computed log-likelihoods; mutated before each FB pass
end

HMMs.initialization(dl::SLDSDiscreteLayer) = dl.πₖ
HMMs.transition_matrix(dl::SLDSDiscreteLayer) = dl.A

# Override the log-density chokepoint so obs_distributions is never needed.
# obs is the timestep index t (an Int), supplied as obs_seq = 1:T.
function HMMs.obs_logdensities!(
    logb::AbstractVector, dl::SLDSDiscreteLayer, obs::Int, control; kwargs...
)
    logb .= view(dl.logL, :, obs)
    return nothing
end

# Provide eltype without going through obs_distributions
Base.eltype(::SLDSDiscreteLayer{T}, obs, control) where {T} = T
