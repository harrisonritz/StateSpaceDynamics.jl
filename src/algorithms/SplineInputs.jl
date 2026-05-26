"""
    generate_spline_inputs!(
        data::Data{T},
        num_bases::Int;
        target::Symbol = :u,
        order::Int = 4,
        knots::Union{Symbol,AbstractVector{<:Real}} = :auto,
        penalty_order::Int = 2,
    ) where {T<:Real} -> Matrix{T}

Construct a time-varying B-spline input basis of shape `(P*num_bases, tsteps, ntrials)`
and write it into the chosen input field of `data` (either `data.u` or `data.d`).

The per-trial input matrix is `kron(trial_pred[n, :], B')` where `B` is the
`(tsteps × num_bases)` matrix obtained by evaluating the basis at the integer
timesteps `1:tsteps`, and `trial_pred` is `data.trial_pred` (or a default
single-column matrix of ones when `data.trial_pred` is empty). The number of
predictors is `P = size(trial_pred, 2)`.

The target field is updated **in place** via `copyto!`; `data` itself is not
replaced. The caller must therefore pre-allocate `data.u` (or `data.d`) with
shape `(P*num_bases, tsteps, ntrials)` before calling — a clear
`DimensionMismatch` is thrown otherwise.

Returns the P-spline regularization matrix `kron(I_P, D'D)` of size
`(P*num_bases) × (P*num_bases)`, where `D` is the `penalty_order`-th
finite-difference matrix on `num_bases` coefficients. The kron with `I_P`
penalises spline-coefficient roughness independently for each predictor block.

# Keywords
- `target::Symbol`: which field to write into, `:u` (dynamics) or `:d`
  (observation). Default `:u`.
- `order::Int`: B-spline order; `4` is cubic. Default `4`.
- `knots`: `:auto` (default) places knots via `BSplines.averagebasis` over
  `num_bases` equally-spaced sites in `1:tsteps`. Pass an
  `AbstractVector{<:Real}` of breakpoints to override; the resulting basis must
  have exactly `num_bases` functions, i.e. `length(knots) == num_bases - order + 2`.
- `penalty_order::Int`: order of the finite-difference penalty; `2` is the
  standard P-spline choice (penalises curvature). Default `2`.
"""
function generate_spline_inputs!(
    data::Data{T},
    num_bases::Int;
    target::Symbol=:u,
    order::Int=4,
    knots::Union{Symbol,AbstractVector{<:Real}}=:auto,
    penalty_order::Int=2,
) where {T<:Real}
    num_bases >= order || throw(
        ArgumentError(
            "num_bases ($num_bases) must be >= order ($order) for a valid B-spline basis.",
        ),
    )
    penalty_order >= 0 ||
        throw(ArgumentError("penalty_order ($penalty_order) must be >= 0."))
    penalty_order < num_bases || throw(
        ArgumentError("penalty_order ($penalty_order) must be < num_bases ($num_bases)."),
    )
    target in (:u, :d) ||
        throw(ArgumentError("target must be :u or :d, got $(repr(target))."))

    tsteps = size(data.y, 2)
    ntrials = size(data.y, 3)

    trial_pred = if isempty(data.trial_pred)
        ones(T, ntrials, 1)
    else
        size(data.trial_pred, 1) == ntrials || throw(
            DimensionMismatch(
                "data.trial_pred has $(size(data.trial_pred, 1)) rows but data.y has " *
                "$ntrials trials. trial_pred must be shape (ntrials, npredictors).",
            ),
        )
        data.trial_pred
    end
    P = size(trial_pred, 2)

    basis = if knots === :auto
        data_sites = collect(range(T(1), T(tsteps); length=num_bases))
        BSplines.averagebasis(order, data_sites)
    elseif knots isa AbstractVector
        BSplines.BSplineBasis(order, collect(T.(knots)))
    else
        throw(ArgumentError("knots must be :auto or an AbstractVector, got $(typeof(knots))."))
    end
    length(basis) == num_bases || throw(
        ArgumentError(
            "constructed basis has length $(length(basis)) but num_bases=$num_bases. " *
            "When passing explicit knots, length(knots) must equal num_bases - order + 2.",
        ),
    )

    ts = collect(T(1):T(tsteps))
    B_raw = BSplines.basismatrix(basis, ts)
    B = eltype(B_raw) === T ? B_raw : convert(Matrix{T}, B_raw)

    K = num_bases
    target_arr = getfield(data, target)
    size(target_arr) == (P * K, tsteps, ntrials) || throw(
        DimensionMismatch(
            "data.$target has shape $(size(target_arr)) but spline inputs require " *
            "$((P * K, tsteps, ntrials)). Pre-allocate data.$target before calling.",
        ),
    )

    # Block layout: for predictor p ∈ 1:P, rows ((p-1)*K + 1):(p*K) of
    # target_arr[:, :, n] hold `trial_pred[n, p] * B'`.
    Bt = transpose(B)
    @inbounds for n in 1:ntrials
        for p in 1:P
            row_start = (p - 1) * K + 1
            row_end = p * K
            coeff = trial_pred[n, p]
            @views target_arr[row_start:row_end, :, n] .= coeff .* Bt
        end
    end

    Dmat = _difference_matrix(K, penalty_order, T)
    DtD = Dmat' * Dmat
    return kron(Matrix{T}(I, P, P), DtD)
end

# d-th finite-difference matrix of size (K - d) × K, obtained by iterating
# `diff` over the K × K identity. `d == 0` returns the identity (ridge penalty).
function _difference_matrix(K::Int, d::Int, ::Type{T}) where {T<:Real}
    D = Matrix{T}(I, K, K)
    for _ in 1:d
        D = diff(D; dims=1)
    end
    return D
end
