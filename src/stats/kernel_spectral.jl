#=============================================================================
Isotropic surface kernels from Laplace–Beltrami eigenmodes, for a
`SpectralCorrelationFamily` of a `ScaleCorrelation` (dynameg's
`validate/shared_surface_spectral.py`, the same formulas).

On a block of columns (a hemisphere) with modes `U` (n_h × N) at the columns'
dipoles and eigenvalues `λ` (N), a spectrum `s(λ; p)` with parameters `p` (log
scale) per unit gives

    V_ik = U_ik √s(λ_k; p_u(i)),   d_i = Σ_k V_ik² + t(p_u(i)),
    M = D^(−½) (V V' + diag t) D^(−½),

`t = (1/4π) ∫_{λ_N}^{λ_max} s dλ` Weyl's tail. Kinds (`:matern`, `:field`,
`:field_matern`), lengths in m, λ in m⁻², ω in rad/s:

- `:matern`: `(κ² + λ)^{−α}`, `κ² = c_α / ℓ²`, `c_α = 8(α − 1)` (α > 1, else 1);
  parameter `log_range`;
- `:field`: `Σ_q w_q / ((a_q + r²λ)² + b_q²)`, `r² = ℓ²/8`, `a_q = 1 − ω_q²/γ²`,
  `b_q = 2ω_q/γ`; parameters `log_range`, `log_gamma`;
- `:field_matern`: `Σ_q w_q (κ_q² + λ)^{−2}`, `κ_q² = (8/ℓ²)(1 + ω_q²/γ²)`.
=============================================================================#

const _SPECTRUM_PARAMS = Dict(
    :matern => [:log_range], :field => [:log_range, :log_gamma],
    :field_matern => [:log_range, :log_gamma],
)

"""
    SurfaceSpectrum(kind; alpha=2, omega=[0], weight=[1], lam_max=Inf)

A spatial spectrum on the surface eigenvalues for a
`SpectralCorrelationFamily`: `kind` `:matern` (smoothness `alpha`),
`:field` (the exact neural field) or `:field_matern` (its Matérn
approximation), the field kinds averaged over the temporal-frequency nodes
`omega` (rad/s) with weights `weight` (normalised here); `lam_max` the upper
cutoff of Weyl's tail (m⁻²). See dynameg's `shared_surface_spectral.py`.
"""
struct SurfaceSpectrum{T<:Real}
    kind::Symbol
    alpha::T
    omega::Vector{T}
    weight::Vector{T}
    lam_max::T
end

function SurfaceSpectrum(
    kind::Symbol; alpha::Real=2.0, omega=[0.0], weight=[1.0], lam_max::Real=Inf
)
    haskey(_SPECTRUM_PARAMS, kind) || throw(
        ArgumentError("kind must be :matern, :field or :field_matern; got :$kind")
    )
    T = float(promote_type(typeof(alpha), eltype(omega), eltype(weight)))
    alpha > 0 || throw(ArgumentError("alpha must be positive"))
    kind === :matern && alpha <= 1 && !isfinite(lam_max) && throw(
        ArgumentError("alpha = $alpha ≤ 1: the Matérn tail diverges, give lam_max")
    )
    length(omega) == length(weight) && !isempty(omega) ||
        throw(DimensionMismatch("omega and weight need one value per node"))
    all(>=(0), omega) && all(>=(0), weight) && sum(weight) > 0 ||
        throw(ArgumentError("omega and weight must be non-negative, weight not zero"))
    lam_max > 0 || throw(ArgumentError("lam_max must be positive"))
    return SurfaceSpectrum{T}(
        kind, T(alpha), Vector{T}(omega), Vector{T}(weight ./ sum(weight)), T(lam_max)
    )
end

_spectrum_params(s::SurfaceSpectrum) = _SPECTRUM_PARAMS[s.kind]
_c_alpha(alpha::Real) = alpha > 1 ? 8 * (alpha - 1) : one(alpha)

# `s` (N) at the eigenvalues `lam` for one parameter row `p`, and with `grad` its
# derivatives `ds` (N × n_params).
function _spectrum(s::SurfaceSpectrum{T}, lam::AbstractVector, p::AbstractVector;
                   grad::Bool=false) where {T<:Real}
    N = length(lam)
    ell = exp(p[1])
    out = zeros(T, N)
    ds = grad ? zeros(T, N, length(p)) : zeros(T, 0, 0)
    if s.kind === :matern
        a = s.alpha
        k2 = _c_alpha(a) / ell^2
        @inbounds for k in 1:N
            base = k2 + lam[k]
            out[k] = base^(-a)
            grad && (ds[k, 1] = 2 * a * k2 * base^(-a - 1))
        end
        return out, ds
    end
    gam = exp(p[2])
    for (om, wq) in zip(s.omega, s.weight)
        if s.kind === :field
            r2 = ell^2 / 8
            aq = 1 - om^2 / gam^2
            bq = 2 * om / gam
            @inbounds for k in 1:N
                X = aq + r2 * lam[k]
                D = X^2 + bq^2
                out[k] += wq / D
                if grad
                    f = wq / D^2
                    ds[k, 1] -= f * (4 * X * r2 * lam[k])
                    ds[k, 2] -= f * (4 * X * om^2 / gam^2 - 2 * bq^2)
                end
            end
        else
            k2 = (8 / ell^2) * (1 + om^2 / gam^2)
            @inbounds for k in 1:N
                base = k2 + lam[k]
                out[k] += wq / base^2
                if grad
                    f = 2 * wq / base^3
                    ds[k, 1] += f * 2 * k2
                    ds[k, 2] += f * 16 * om^2 / (ell^2 * gam^2)
                end
            end
        end
    end
    return out, ds
end

# ∫_lo^hi (κ² + λ)^{−α} dλ and its derivative in κ².
function _matern_tail(k2::Real, alpha::Real, lo::Real, hi::Real)
    if alpha == 1
        value = (isfinite(hi) ? log(k2 + hi) : oftype(k2, Inf)) - log(k2 + lo)
        d = (isfinite(hi) ? 1 / (k2 + hi) : zero(k2)) - 1 / (k2 + lo)
        return value, d
    end
    top = isfinite(hi) ? (k2 + hi)^(1 - alpha) / (alpha - 1) : zero(k2)
    value = (k2 + lo)^(1 - alpha) / (alpha - 1) - top
    d = -(k2 + lo)^(-alpha) + (isfinite(hi) ? (k2 + hi)^(-alpha) : zero(k2))
    return value, d
end

# `F = atan(b, X)/b = ∫_X^∞ dx/(x² + b²)`, `∂F/∂X`, `∂F/∂b` (a series for b ≪ |X|).
function _atan_ratio(X::Real, b::Real)
    if X > 0 && abs(b) < 1e-4 * X  # the series holds below the resonance
        return 1 / X - b^2 / (3 * X^3), -1 / (X^2 + b^2), -2 * b / (3 * X^3)
    end
    F = atan(b, X) / b
    return F, -1 / (X^2 + b^2), X / (b * (X^2 + b^2)) - atan(b, X) / b^2
end

# Weyl's tail `t` and `∂t/∂p` (n_params) for one parameter row.
function _spectrum_tail(s::SurfaceSpectrum{T}, lam_n::Real, p::AbstractVector) where {T<:Real}
    top = s.lam_max
    n_p = length(p)
    dt = zeros(T, n_p)
    top > lam_n || return zero(T), dt
    ell = exp(p[1])
    t = zero(T)
    if s.kind === :matern
        k2 = _c_alpha(s.alpha) / ell^2
        t, dk = _matern_tail(k2, s.alpha, lam_n, top)
        dt[1] = dk * -2 * k2
    else
        gam = exp(p[2])
        for (om, wq) in zip(s.omega, s.weight)
            if s.kind === :field
                r2 = ell^2 / 8
                a = 1 - om^2 / gam^2
                b = 2 * om / gam
                F0, X0, B0 = _atan_ratio(a + r2 * lam_n, b)
                F1, X1, B1 = isfinite(top) ? _atan_ratio(a + r2 * top, b) :
                             (zero(T), zero(T), zero(T))
                tq = (F0 - F1) / r2
                t += wq * tq
                top_l = isfinite(top) ? r2 * top : zero(T)
                dt[1] += wq * ((X0 * 2 * r2 * lam_n - X1 * 2 * top_l) / r2 - 2 * tq)
                dt[2] += wq * ((X0 - X1) * 2 * om^2 / gam^2 - (B0 - B1) * b) / r2
            else
                k2 = (8 / ell^2) * (1 + om^2 / gam^2)
                tq, dk = _matern_tail(k2, 2, lam_n, top)
                t += wq * tq
                dt[1] += wq * dk * -2 * k2
                dt[2] += wq * dk * (-16 * om^2 / (ell^2 * gam^2))
            end
        end
    end
    return t / (4 * T(π)), dt ./ (4 * T(π))
end
