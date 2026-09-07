"""
    DimensionMismatchError <: Exception

Custom exception for dimension mismatches in model parameters.

# Fields
- `parameter::String`: Name of the parameter with incorrect dimensions
- `expected::Union{Int,Tuple{Vararg{Int}}}`: Expected dimension(s)
- `got::Union{Int,Tuple{Vararg{Int}}}`: Actual dimension(s)
"""
struct DimensionMismatchError <: Exception
    parameter::String
    expected::Union{Int,Tuple{Vararg{Int}}}
    got::Union{Int,Tuple{Vararg{Int}}}
end

function Base.showerror(io::IO, e::DimensionMismatchError)
    print(io, "DimensionMismatchError: ")
    return print(io, "$(e.parameter) has dimensions $(e.got), expected $(e.expected)")
end

"""
    NotPositiveDefiniteError <: Exception

Custom exception for matrices that should be positive definite but aren't.

# Fields
- `matrix_name::String`: Name of the matrix
- `min_eigenvalue::Float64`: Minimum eigenvalue found
"""
struct NotPositiveDefiniteError <: Exception
    matrix_name::String
    min_eigenvalue::Float64
end

function Base.showerror(io::IO, e::NotPositiveDefiniteError)
    print(io, "NotPositiveDefiniteError: ")
    print(io, "$(e.matrix_name) is not positive definite ")
    print(io, "(minimum eigenvalue: $(e.min_eigenvalue)). ")
    return print(io, "Consider adding regularization or checking for numerical issues.")
end

"""
    NotSymmetricError <: Exception

Custom exception for matrices that should be symmetric but aren't.

# Fields
- `matrix_name::String`: Name of the matrix
- `max_asymmetry::Float64`: Maximum asymmetry measure
"""
struct NotSymmetricError <: Exception
    matrix_name::String
    max_asymmetry::Float64
end

function Base.showerror(io::IO, e::NotSymmetricError)
    print(io, "NotSymmetricError: ")
    print(io, "$(e.matrix_name) is not symmetric ")
    return print(io, "(max asymmetry: $(e.max_asymmetry))")
end

"""
    InvalidProbabilityVectorError <: Exception

Custom exception for invalid probability vectors.

# Fields
- `vector_name::String`: Name of the probability vector
- `sum_value::Float64`: Sum of the vector
- `has_negative::Bool`: Whether the vector contains negative values
- `has_greater_than_one::Bool`: Whether the vector contains values > 1.0
"""
struct InvalidProbabilityVectorError <: Exception
    vector_name::String
    sum_value::Float64
    has_negative::Bool
    has_greater_than_one::Bool
end

function Base.showerror(io::IO, e::InvalidProbabilityVectorError)
    print(io, "InvalidProbabilityVectorError: ")
    print(io, "$(e.vector_name) is not a valid probability vector. ")
    if !isapprox(e.sum_value, 1.0; atol=1e-10)
        print(io, "Sum is $(e.sum_value), not 1.0. ")
    end
    if e.has_negative
        print(io, "Contains negative values. ")
    end
    if e.has_greater_than_one
        print(io, "Contains values > 1.0.")
    end
end

"""
    NumericalStabilityError <: Exception

Custom exception for numerical stability issues.

# Fields
- `parameter::String`: Name of the parameter
- `issue::String`: Description of the numerical issue
"""
struct NumericalStabilityError <: Exception
    parameter::String
    issue::String
end

function Base.showerror(io::IO, e::NumericalStabilityError)
    print(io, "NumericalStabilityError: ")
    return print(io, "$(e.parameter) - $(e.issue)")
end

"""
    _validate_state_model(state_model::HamiltonianStateModel{T}, latent_dim::Int) where T

Validate a [`HamiltonianStateModel`](@ref): the LQR structure (`A` square and
invertible, `S` and every `Qc` symmetric, a schedule that indexes real cost
matrices), the shapes of the mixed-coordinate noise and bias against the doubled
latent dimension `2n`, and positive definiteness of `Σ`, `Σf` and `P0`.

# Throws
- `DimensionMismatchError`, `NotSymmetricError`, `NotPositiveDefiniteError`,
  `NumericalStabilityError` (a singular plant), or `ArgumentError` (a bad
  schedule, or a `depends_on` this model cannot honor)
"""
function _validate_state_model(
    state_model::HamiltonianStateModel{T}, latent_dim::Int
) where {T}
    sm = state_model
    n = _plant_dim(sm)
    if latent_dim != 2n
        throw(DimensionMismatchError("Hamiltonian latent_dim (2n)", 2n, latent_dim))
    end

    #=
    A `:free` model has no plant, cost or costate to check, and no terminal
    factor — only the shapes that both modes share. Its transition is checked
    here instead of by `_check_hamiltonian_structure`, which is about the
    symplectic form.
    =#
    if _is_free(sm)
        if size(sm.Mfree) != (2n, 2n)
            throw(DimensionMismatchError("free transition", (2n, 2n), size(sm.Mfree)))
        end
        if !isempty(sm.A) || !isempty(sm.S) || !isempty(sm.Qc)
            throw(
                ArgumentError(
                    "a `:free` state model carries no plant or cost, but `A`, `S` or " *
                    "`Qc` is non-empty. Build it with `free_state_model`.",
                ),
            )
        end
        sm.terminal && throw(
            ArgumentError(
                "a `:free` state model has no costate to pin, so it cannot carry a " *
                "terminal condition. Use `:lqr` mode for that.",
            ),
        )
    else
        _check_hamiltonian_structure(sm.A, sm.S, sm.Qc, sm.schedule, sm.terminal)
    end

    for (name, Σ, dim) in ((:Σ, sm.Σ, 2n), (:Σf, sm.Σf, n), (:P0, sm.P0, 2n))
        if size(Σ) != (dim, dim)
            throw(DimensionMismatchError("Hamiltonian $name", (dim, dim), size(Σ)))
        end
        if !issymmetric(Σ)
            throw(NotSymmetricError("Hamiltonian $name", maximum(abs.(Σ .- Σ'))))
        end
        if !isposdef(Σ)
            throw(NotPositiveDefiniteError("Hamiltonian $name", minimum(eigvals(Σ))))
        end
    end

    if length(sm.h) != 2n
        throw(DimensionMismatchError("Hamiltonian h", 2n, length(sm.h)))
    end
    if length(sm.x0) != 2n
        throw(DimensionMismatchError("Hamiltonian x0", 2n, length(sm.x0)))
    end
    if length(sm.hf) != n
        throw(DimensionMismatchError("Hamiltonian hf", n, length(sm.hf)))
    end
    if size(sm.Bu, 1) != 2n
        throw(DimensionMismatchError("Hamiltonian Bu rows", 2n, size(sm.Bu, 1)))
    end
    return nothing
end

"""
    _validate_state_model(state_model::GaussianStateModel{T}, latent_dim::Int) where T

Validate GaussianStateModel parameters. Throws exceptions on validation failure.

# Throws
- `DimensionMismatchError`: If dimensions don't match expected values
- `NotSymmetricError`: If covariance matrices aren't symmetric
- `NotPositiveDefiniteError`: If covariance matrices aren't positive definite
"""
function _validate_state_model(
    state_model::GaussianStateModel{T}, latent_dim::Int
) where {T}
    # Check A matrix
    if size(state_model.A) != (latent_dim, latent_dim)
        throw(
            DimensionMismatchError(
                "A matrix", (latent_dim, latent_dim), size(state_model.A)
            ),
        )
    end

    # Check optional B matrix (dynamics input)
    if size(state_model.B, 1) != latent_dim
        throw(DimensionMismatchError("B matrix rows", latent_dim, size(state_model.B, 1)))
    end

    # Check Q matrix (process noise covariance)
    if size(state_model.Q) != (latent_dim, latent_dim)
        throw(
            DimensionMismatchError(
                "Q matrix", (latent_dim, latent_dim), size(state_model.Q)
            ),
        )
    end

    if !issymmetric(state_model.Q)
        max_asym = maximum(abs.(state_model.Q - state_model.Q'))
        throw(NotSymmetricError("Q matrix", max_asym))
    end

    if !isposdef(state_model.Q)
        min_eval = minimum(eigvals(state_model.Q))
        throw(NotPositiveDefiniteError("Q matrix", min_eval))
    end

    # Check bias vector b
    if length(state_model.b) != latent_dim
        throw(DimensionMismatchError("bias vector b", latent_dim, length(state_model.b)))
    end

    # Check initial state x0
    if length(state_model.x0) != latent_dim
        throw(
            DimensionMismatchError("initial state x0", latent_dim, length(state_model.x0))
        )
    end

    # Check P0 matrix (initial covariance)
    if size(state_model.P0) != (latent_dim, latent_dim)
        throw(
            DimensionMismatchError(
                "P0 matrix", (latent_dim, latent_dim), size(state_model.P0)
            ),
        )
    end

    if !issymmetric(state_model.P0)
        max_asym = maximum(abs.(state_model.P0 - state_model.P0'))
        throw(NotSymmetricError("P0 matrix", max_asym))
    end

    if !isposdef(state_model.P0)
        min_eval = minimum(eigvals(state_model.P0))
        throw(NotPositiveDefiniteError("P0 matrix", min_eval))
    end

    return nothing
end

"""
    _validate_obs_model(obs_model::GaussianObservationModel{T}, obs_dim::Int, latent_dim::Int) where T

Validate GaussianObservationModel parameters. Throws exceptions on validation failure.

# Throws
- `DimensionMismatchError`: If dimensions don't match expected values
- `NotSymmetricError`: If R matrix isn't symmetric
- `NotPositiveDefiniteError`: If R matrix isn't positive definite
"""
function _validate_obs_model(
    obs_model::GaussianObservationModel{T}, obs_dim::Int, latent_dim::Int
) where {T}
    # Check C matrix
    if size(obs_model.C) != (obs_dim, latent_dim)
        throw(DimensionMismatchError("C matrix", (obs_dim, latent_dim), size(obs_model.C)))
    end

    # Check R matrix (observation noise covariance)
    if size(obs_model.R) != (obs_dim, obs_dim)
        throw(DimensionMismatchError("R matrix", (obs_dim, obs_dim), size(obs_model.R)))
    end

    # TODO: check D matrix

    if !issymmetric(obs_model.R)
        max_asym = maximum(abs.(obs_model.R - obs_model.R'))
        throw(NotSymmetricError("R matrix", max_asym))
    end

    if !isposdef(obs_model.R)
        min_eval = minimum(eigvals(obs_model.R))
        throw(NotPositiveDefiniteError("R matrix", min_eval))
    end

    # Check bias vector d
    if length(obs_model.d) != obs_dim
        throw(DimensionMismatchError("observation bias d", obs_dim, length(obs_model.d)))
    end

    return nothing
end

"""
    _validate_obs_model(obs_model::PoissonObservationModel{T}, obs_dim::Int, latent_dim::Int) where T

Validate PoissonObservationModel parameters. Throws exceptions on validation failure.

# Throws
- `DimensionMismatchError`: If dimensions don't match expected values
- `NumericalStabilityError`: If `d` values are extremely large/small
"""
function _validate_obs_model(
    obs_model::PoissonObservationModel{T}, obs_dim::Int, latent_dim::Int
) where {T}
    # Check C matrix
    if size(obs_model.C) != (obs_dim, latent_dim)
        throw(DimensionMismatchError("C matrix", (obs_dim, latent_dim), size(obs_model.C)))
    end

    # Check d vector
    if length(obs_model.d) != obs_dim
        throw(DimensionMismatchError("d vector", obs_dim, length(obs_model.d)))
    end

    # Check D matrix (observation-input map): (obs_dim × uy_dim). uy_dim is free,
    # but the row count must match obs_dim.
    if hasproperty(obs_model, :D) && size(obs_model.D, 1) != obs_dim
        throw(DimensionMismatchError("D matrix rows", obs_dim, size(obs_model.D, 1)))
    end

    #=
    Check that d values are reasonable. `d` enters the linear predictor as
    `λ = exp(C x + d + D v)`; |d| above ~50 risks exp overflow/underflow once Cx
    is added on top.
    =#
    if any(x -> abs(x) > 50, obs_model.d)  # exp(50) ≈ 5e21, exp(-50) ≈ 2e-22
        max_val = maximum(abs.(obs_model.d))
        println("WARNING: high d")
        println("\nd:\n $(obs_model.d)")
        println("\nD:\n $(obs_model.D)")
        throw(
            NumericalStabilityError(
                "d vector",
                "contains extremely large/small values (max |d| = $max_val), may cause numerical overflow/underflow",
            ),
        )
    end

    return nothing
end

"""
    _validate_obs_model(obs_model::CompositeObservationModel, obs_dim, latent_dim)

Validate every member of a composite emission against its own channel count.
`obs_dim` is the composite's total and is checked against the members' sum; each
member is then handed its own width, so a per-member shape error names the
member it came from.

# Throws
- `DimensionMismatchError`: if the members' widths do not sum to `obs_dim`, or a
  member's own parameters are inconsistent
- whatever the member's validator throws, with the member named
"""
function _validate_obs_model(
    obs_model::CompositeObservationModel{T}, obs_dim::Int, latent_dim::Int
) where {T}
    models = _models(obs_model)
    total = 0
    for (key, m) in pairs(models)
        p = _obs_dim(m)
        try
            _validate_obs_model(m, p, latent_dim)
        catch err
            err isa Exception || rethrow()
            throw(
                ArgumentError(
                    "observation model `:$key` is invalid: " * sprint(showerror, err)
                ),
            )
        end
        total += p
    end

    if total != obs_dim
        throw(
            DimensionMismatchError("composite obs_dim (sum over members)", total, obs_dim)
        )
    end

    return nothing
end

"""
    validate_LDS(lds::LinearDynamicalSystem{T,S,O}) where {T,S,O}

Validate that all parameters in a LinearDynamicalSystem are dimensionally consistent
and mathematically valid. Throws descriptive exceptions on validation failure.

# Checks performed
- Matrix dimensions are consistent
- Covariance matrices are positive definite and symmetric
- fit_bool has correct length for the observation model type
- Stored dimensions match dimensions inferred from matrices

# Throws
- `DimensionMismatchError`: If dimensions don't match
- `NotPositiveDefiniteError`: If covariance matrices aren't positive definite
- `NotSymmetricError`: If covariance matrices aren't symmetric
- `NumericalStabilityError`: If numerical issues are detected

# Examples
```julia
# This will throw DimensionMismatchError if invalid
validate_LDS(my_lds)

# Can be caught for custom handling
try
    validate_LDS(my_lds)
    println("LDS is valid!")
catch e
    if e isa DimensionMismatchError
        println("Dimension error: ", e)
    end
end
```
"""
function validate_LDS(lds::LinearDynamicalSystem{T,S,O}) where {T,S,O}
    # Check state model dimensions and properties
    _validate_state_model(lds.state_model, lds.latent_dim)

    # Check observation model dimensions and properties
    _validate_obs_model(lds.obs_model, lds.obs_dim, lds.latent_dim)

    #=
    Resolve any ancillary parameter dependencies so a malformed `depends_on`
    (unknown parameter name, conflicting aliases of one jointly-fitted group,
    label vectors of unequal length) is reported at construction rather than at
    the first `fit!`. The resolved value is discarded — the fitting entry points
    re-resolve it against the actual trial count.
    =#
    _resolve_dependence(lds.state_model)
    _resolve_dependence(lds.obs_model)

    #=
    Check fit_bool length: four state groups, then one block per observation
    model (`[C d D]`, plus `R` when that model is Gaussian). Length 6 for a
    Gaussian LDS, 5 for a Poisson one, `4 + Σₘ blocks` for a composite.
    =#
    expected_fit_length = 4 + _obs_nblocks(lds.obs_model)
    if length(lds.fit_bool) != expected_fit_length
        throw(DimensionMismatchError("fit_bool", expected_fit_length, length(lds.fit_bool)))
    end

    # Check consistency between inferred and stored dimensions
    inferred_latent = _state_latent_dim(lds.state_model)
    inferred_obs = _obs_dim(lds.obs_model)

    if lds.latent_dim != inferred_latent
        throw(
            DimensionMismatchError(
                "latent_dim (stored vs inferred from A)", inferred_latent, lds.latent_dim
            ),
        )
    end

    if lds.obs_dim != inferred_obs
        throw(
            DimensionMismatchError(
                "obs_dim (stored vs inferred from the emission)", inferred_obs, lds.obs_dim
            ),
        )
    end

    return nothing
end

"""
    validate_SLDS(slds::SLDS)

Validate SLDS structure. Throws descriptive exceptions on validation failure.

# Checks performed
- Dimensions of A match the length of πₖ and the number of LDSs
- Rows of A and πₖ are valid probability vectors
- Each LDS has the same state dimension and observation dimension
- Each individual LDS is valid

# Throws
- `DimensionMismatchError`: If dimensions are inconsistent
- `InvalidProbabilityVectorError`: If probability vectors are invalid
- Other exceptions from `validate_LDS` for individual LDS validation

# Examples
```julia
# This will throw if invalid
validate_SLDS(my_slds)

# Can be caught for custom handling
try
    validate_SLDS(my_slds)
catch e
    if e isa InvalidProbabilityVectorError
        println("Probability vector error: ", e)
    end
end
```
"""
function validate_SLDS(slds::SLDS)
    k = size(slds.A, 1)
    D = length(slds.πₖ)
    lds_count = length(slds.LDSs)

    # Checks for HMM components
    if k != D
        throw(DimensionMismatchError("size(A, 1) vs length(πₖ)", D, k))
    end

    if k != lds_count
        throw(DimensionMismatchError("size(A, 1) vs number of LDSs", lds_count, k))
    end

    # Validate transition matrix rows and the initial distribution. Delegating to
    # validate_probvec keeps a single source of truth for the type-scaled
    # tolerance (see A12) instead of re-hardcoding 1.0/atol=1e-10 here.
    for i in 1:k
        validate_probvec(@view(slds.A[i, :]); name="A[$i, :]")
    end
    validate_probvec(slds.πₖ; name="πₖ")

    # Checks for LDS models
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim
    ux_dim = slds.LDSs[1].ux_dim
    uy_dim = slds.LDSs[1].uy_dim

    for (i, lds) in enumerate(slds.LDSs)
        if lds.latent_dim != latent_dim
            throw(DimensionMismatchError("LDS[$i].latent_dim", latent_dim, lds.latent_dim))
        end

        if lds.obs_dim != obs_dim
            throw(DimensionMismatchError("LDS[$i].obs_dim", obs_dim, lds.obs_dim))
        end

        # Input dimensions must be uniform across LDsS models so a single `Data`
        if lds.ux_dim != ux_dim
            throw(DimensionMismatchError("LDS[$i].ux_dim", ux_dim, lds.ux_dim))
        end

        if lds.uy_dim != uy_dim
            throw(DimensionMismatchError("LDS[$i].uy_dim", uy_dim, lds.uy_dim))
        end

        #=
        A composite emission needs no key check here: `SLDS.LDSs` is a
        `Vector{LinearDynamicalSystem{T,S,O}}` with one concrete `O`, and the
        member names and order are part of the composite's type — so regimes
        that disagree cannot be put in the same `SLDS` at all.
        =#

        # This will throw if invalid
        validate_LDS(lds)
    end

    return nothing
end

"""
    validate_probvec(v::AbstractVector{T}; name::String="vector") where {T<:Real}

Validate that a vector is a valid probability vector (sums to 1, all non-negative, all ≤ 1).
Throws `InvalidProbabilityVectorError` if validation fails.

# Arguments
- `v`: The vector to validate
- `name`: Optional name for the vector (used in error messages)

# Examples
```julia
v1 = [0.3, 0.5, 0.2]
validate_probvec(v1)  # No error

v2 = [0.3, 0.5, 0.3]
validate_probvec(v2)  # Throws InvalidProbabilityVectorError
```
"""
function validate_probvec(v::AbstractVector{T}; name::String="vector") where {T<:Real}
    sum_val = sum(v)
    has_neg = any(x -> x < 0, v)
    has_gt1 = any(x -> x > 1, v)

    # Type-scaled tolerance:
    atol = sqrt(eps(float(T)))
    if !isapprox(sum_val, one(T); atol=atol) || has_neg || has_gt1
        throw(InvalidProbabilityVectorError(name, sum_val, has_neg, has_gt1))
    end

    return nothing
end

# ============================================================================
# input-sequence normalization helpers. The public `ux`/`uy`
# kwargs accept either `nothing` (no inputs — must match a zero-column `B`/`D`)
# or per-trial matrices. Internally every sampler/smoother/M-step expects an
# `AbstractMatrix{T}` of shape `(ux_dim, T_i)` (possibly `0 × T_i`), so these
# helpers validate the supplied sequences and canonicalize on the way in.
# ============================================================================

function _check_ux(
    cs::Nothing, expected_dim::Int, tsteps::Int, name::AbstractString, ::Type{T}
) where {T}
    expected_dim == 0 || throw(
        ArgumentError(
            "$(name)=nothing is only valid when the corresponding input matrix is " *
            "zero-column; got expected_dim=$(expected_dim). Pass a $(expected_dim)×T " *
            "matrix or shrink the input matrix.",
        ),
    )
    return zeros(T, 0, tsteps)
end

function _check_ux(
    cs::AbstractMatrix{T}, expected_dim::Int, tsteps::Int, name::AbstractString, ::Type{T}
) where {T<:Real}
    size(cs, 1) == expected_dim || throw(
        DimensionMismatchError(
            "$(name) rows vs input-matrix cols", expected_dim, size(cs, 1)
        ),
    )
    size(cs, 2) == tsteps ||
        throw(DimensionMismatchError("$(name) tsteps", tsteps, size(cs, 2)))
    return cs
end

@inline function _check_uy(
    cs, expected_dim::Int, tsteps::Int, ::GaussianObservationModel{T}
) where {T}
    return _check_ux(cs, expected_dim, tsteps, "uy", T)
end

@inline function _check_uy(
    cs, expected_dim::Int, tsteps::Int, ::PoissonObservationModel{T}
) where {T}
    return _check_ux(cs, expected_dim, tsteps, "uy", T)
end

#=
A composite emission takes one input sequence per member, so the canonicalized
value is a NamedTuple of matrices rather than one matrix. A bare input is the
shorthand for "these covariates feed every readout" and is checked against each
member's own `D`; `_member_uy` picks a member's entry apart.
=#
@inline function _check_uy(
    cs, ::Int, tsteps::Int, om::CompositeObservationModel{T}
) where {T}
    models = _models(om)
    return NamedTuple{keys(models)}(
        map(
            key ->
                _check_uy(_member_uy(cs, key), _uy_dim(models[key]), tsteps, models[key]),
            keys(models),
        ),
    )
end

@inline function _normalize_multitrial_uy(
    cs, ::Int, tsteps_per_trial, ::Type{T}, om::CompositeObservationModel
) where {T<:Real}
    models = _models(om)
    return NamedTuple{keys(models)}(
        map(
            key -> _normalize_multitrial_ux(
                _member_uy(cs, key),
                _uy_dim(models[key]),
                tsteps_per_trial,
                T,
                "uy[:$key]",
            ),
            keys(models),
        ),
    )
end

function _normalize_multitrial_ux(
    cs::Nothing, expected_dim::Int, tsteps_per_trial, ::Type{T}, name::AbstractString
) where {T<:Real}
    expected_dim == 0 || throw(
        ArgumentError(
            "$(name)=nothing is only valid when expected_dim == 0; got $(expected_dim)"
        ),
    )
    return [zeros(T, 0, Int(Ti)) for Ti in tsteps_per_trial]
end

function _normalize_multitrial_ux(
    cs::AbstractVector{<:AbstractMatrix{T}},
    expected_dim::Int,
    tsteps_per_trial,
    ::Type{T},
    name::AbstractString,
) where {T<:Real}
    length(cs) == length(tsteps_per_trial) || throw(
        DimensionMismatchError("$(name) ntrials", length(tsteps_per_trial), length(cs))
    )
    for (i, ci) in enumerate(cs)
        size(ci, 1) == expected_dim ||
            throw(DimensionMismatchError("$(name)[$i] rows", expected_dim, size(ci, 1)))
        size(ci, 2) == Int(tsteps_per_trial[i]) || throw(
            DimensionMismatchError(
                "$(name)[$i] tsteps", Int(tsteps_per_trial[i]), size(ci, 2)
            ),
        )
    end
    return cs
end

@inline function _normalize_multitrial_uy(
    cs, expected_dim::Int, tsteps_per_trial, ::Type{T}, ::GaussianObservationModel
) where {T<:Real}
    return _normalize_multitrial_ux(cs, expected_dim, tsteps_per_trial, T, "uy")
end

@inline function _normalize_multitrial_uy(
    cs, expected_dim::Int, tsteps_per_trial, ::Type{T}, ::PoissonObservationModel
) where {T<:Real}
    return _normalize_multitrial_ux(cs, expected_dim, tsteps_per_trial, T, "uy")
end

# The public `Data(lds, y; ux, uy)` constructors that consume the multitrial
# normalization helpers above live next to the `Data` struct in `lds/types.jl`.
