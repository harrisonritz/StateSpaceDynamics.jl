#=============================================================================
Monotonic rational-quadratic splines (RQS).

Element-wise, strictly increasing, analytically invertible maps used as the
single normalizing-flow layer of a `SplineGaussianObservationModel`. Pure
numerics — nothing here knows about latent-state models.

The spline is the one of Gregory & Delbourgo (1982) in the normalizing-flow
parameterization of Durkan et al. (2019), *Neural Spline Flows*. Channel `j`
maps its own interval `[lo[j], hi[j]]` onto itself through `K` bins and is the
identity outside it, so the map is a C¹ bijection of ℝ.

Storage layout (`pX`, `pY`, `dYdX`: knot x-positions, y-positions and
derivatives, each `(K+1) × p`) is the same one `MonotonicSplines.jl` uses for
its `RQSpline`, including that package's continuity requirements
(`pX[1] == pY[1]`, `pX[end] == pY[end]`, `dYdX[1] == dYdX[end] == 1`). Column
`j` can therefore be handed straight to `MonotonicSplines.RQSpline` for
plotting or as an `InverseFunctions`/`ChangesOfVariables` object:

```julia
using MonotonicSplines
f = RQSpline(warp.pX[:, j], warp.pY[:, j], warp.dYdX[:, j])
```

Knots are derived, not free. The free parameters are unconstrained reals — a
softmax over bin widths, a softmax over bin heights, and log-derivatives at the
interior knots — so monotonicity holds for *every* point in parameter space and
the M-step optimizer needs no constraints:

```math
w = W ⊙ softmax(θʷ),  h = W ⊙ softmax(θʰ),  δ_{k+1} = exp(θᵈ_k)
```

with `W = hi - lo`, `pX = lo .+ cumsum(w)` (prepending `lo`), `pY` likewise, and
`δ₁ = δ_{K+1} = 1` pinned by the identity tails. `θ = 0` is exactly the identity
map, which is what makes a spline emission initialize at its linear counterpart.

Pinning the endpoints to the diagonal is not cosmetic: without it the family
`g ↦ α ⊙ g + β` is an exact `2p`-dimensional flat direction of the likelihood
(the `log det R` and log-Jacobian terms cancel), and the warp would be
unidentified against the emission's `C`, `d` and `R`.
=============================================================================#

"""
    MonotonicWarp{T<:Real}

`p` independent monotonic rational-quadratic splines, one per observation
channel, with `K` bins each. See the file header for the parameterization.

# Fields
- `lo::Vector{T}` / `hi::Vector{T}`: per-channel knot interval (length `p`).
    Fixed at construction — they are part of the model definition, not fitted
    parameters, so log-densities stay comparable across EM iterations.
- `θw::Matrix{T}`: bin-width logits (`K × p`).
- `θh::Matrix{T}`: bin-height logits (`K × p`).
- `θd::Matrix{T}`: interior-knot log-derivatives (`(K-1) × p`).
- `pX`, `pY`, `dYdX::Matrix{T}`: derived knots (`(K+1) × p`), refreshed from
    the logits by [`refresh_knots!`](@ref).
- `sw`, `sh::Matrix{T}`: derived bin-width / bin-height *fractions* (`K × p`),
    cached because the parameter-space chain rule needs them.
- `min_bin::T`: floor on a bin fraction, applied as
    `σ ↦ min_bin + (1 - K·min_bin)·softmax(θ)`. Keeps a bin from collapsing and
    the local slope `s = Δy/Δx` from blowing up.

# Notes
Free parameters number `p · (3K - 1)`. `K ≥ 2` is required: `K == 1` admits only
the identity.
"""
struct MonotonicWarp{T<:Real}
    lo::Vector{T}
    hi::Vector{T}
    θw::Matrix{T}
    θh::Matrix{T}
    θd::Matrix{T}
    pX::Matrix{T}
    pY::Matrix{T}
    dYdX::Matrix{T}
    sw::Matrix{T}
    sh::Matrix{T}
    min_bin::T
end

"""
    MonotonicWarp(lo, hi; n_bins=8, min_bin=1e-3)

An identity warp on the per-channel intervals `[lo[j], hi[j]]`, with `n_bins`
bins. All logits start at zero, so `warp(y) == y` and every log-Jacobian is
zero until the M-step moves them.

# Throws
- `ArgumentError` when `lo`/`hi` disagree in length, some `hi[j] <= lo[j]`,
  `n_bins < 2`, or `min_bin` does not leave room for `n_bins` bins.
"""
function MonotonicWarp(
    lo::AbstractVector{T}, hi::AbstractVector{T}; n_bins::Int=8, min_bin::Real=1e-3
) where {T<:Real}
    p = length(lo)
    length(hi) == p || throw(
        ArgumentError("warp bounds disagree: length(lo)=$p, length(hi)=$(length(hi))")
    )
    p > 0 || throw(ArgumentError("a warp needs at least one channel"))
    n_bins >= 2 || throw(
        ArgumentError(
            "a monotonic warp needs n_bins ≥ 2 (n_bins = 1 can only be the identity); " *
            "got $n_bins",
        ),
    )
    mb = T(min_bin)
    (mb >= zero(T) && n_bins * mb < one(T)) || throw(
        ArgumentError(
            "min_bin must satisfy 0 ≤ n_bins·min_bin < 1; got min_bin=$min_bin with " *
            "n_bins=$n_bins",
        ),
    )
    for j in 1:p
        hi[j] > lo[j] || throw(
            ArgumentError(
                "warp bounds must satisfy hi > lo; channel $j has " *
                "lo=$(lo[j]), hi=$(hi[j])",
            ),
        )
    end

    K = n_bins
    warp = MonotonicWarp{T}(
        collect(T, lo),
        collect(T, hi),
        zeros(T, K, p),
        zeros(T, K, p),
        zeros(T, K - 1, p),
        zeros(T, K + 1, p),
        zeros(T, K + 1, p),
        zeros(T, K + 1, p),
        zeros(T, K, p),
        zeros(T, K, p),
        mb,
    )
    refresh_knots!(warp)
    return warp
end

"""
    warp_channels(warp) -> Int

Number of observation channels the warp covers.
"""
@inline warp_channels(warp::MonotonicWarp) = length(warp.lo)

"""
    warp_bins(warp) -> Int

Number of spline bins per channel.
"""
@inline warp_bins(warp::MonotonicWarp) = size(warp.θw, 1)

"""
    warp_nparams(warp) -> Int

Length of the flat parameter vector: `p · (3K - 1)`.
"""
@inline function warp_nparams(warp::MonotonicWarp)
    return warp_channels(warp) * (3 * warp_bins(warp) - 1)
end

"""
    is_identity_warp(warp) -> Bool

Whether every logit is zero, i.e. the warp is exactly the identity map. Used to
short-circuit the embedding pass at the first EM iteration and to let callers
report that a fit never left the linear model.
"""
function is_identity_warp(warp::MonotonicWarp{T}) where {T<:Real}
    return all(iszero, warp.θw) && all(iszero, warp.θh) && all(iszero, warp.θd)
end

#=
Softmax with the bin floor folded in, written into `out`. Subtracting the max
keeps `exp` in range for the wide logits a stalled line search can propose.

The arguments are not tied to one element type in the signature. They always
*are* one at a call site — they are columns of the same warp — but writing it
that way makes the method unresolvable when a caller is analysed with an
abstractly-typed `MonotonicWarp`, because the element type of a `view` into a
`Matrix{T}` field then widens. Inference at the real call site still sees the
concrete types, so nothing is lost.
=#
function _floored_softmax!(out::AbstractVector, θ::AbstractVector, min_bin::Real)
    T = eltype(out)
    K = length(θ)
    m = θ[1]
    @inbounds for k in 2:K
        θ[k] > m && (m = θ[k])
    end
    total = zero(T)
    @inbounds for k in 1:K
        e = exp(θ[k] - m)
        out[k] = e
        total += e
    end
    scale = (one(T) - K * min_bin) / total
    @inbounds for k in 1:K
        out[k] = min_bin + out[k] * scale
    end
    return out
end

"""
    refresh_knots!(warp) -> warp

Rebuild the derived knots (`pX`, `pY`, `dYdX`) and the cached bin fractions
(`sw`, `sh`) from the logits. Every write to `θw` / `θh` / `θd` must be followed
by this call; `warp_unpack!` does it for you.
"""
function refresh_knots!(warp::MonotonicWarp{T}) where {T<:Real}
    K = warp_bins(warp)
    @inbounds for j in 1:warp_channels(warp)
        width = warp.hi[j] - warp.lo[j]

        _floored_softmax!(view(warp.sw, :, j), view(warp.θw, :, j), warp.min_bin)
        _floored_softmax!(view(warp.sh, :, j), view(warp.θh, :, j), warp.min_bin)

        warp.pX[1, j] = warp.lo[j]
        warp.pY[1, j] = warp.lo[j]
        for k in 1:K
            warp.pX[k + 1, j] = warp.pX[k, j] + width * warp.sw[k, j]
            warp.pY[k + 1, j] = warp.pY[k, j] + width * warp.sh[k, j]
        end
        #=
        The fractions sum to 1 analytically, but the running sum above is off by
        an ulp or two. Pin the last knot so `y == hi` lands on the identity
        branch exactly and the tails stay continuous.
        =#
        warp.pX[K + 1, j] = warp.hi[j]
        warp.pY[K + 1, j] = warp.hi[j]

        warp.dYdX[1, j] = one(T)
        warp.dYdX[K + 1, j] = one(T)
        for k in 1:(K - 1)
            warp.dYdX[k + 1, j] = exp(warp.θd[k, j])
        end
    end
    return warp
end

"""
    warp_pack!(θ, warp) -> θ

Flatten the logits into `θ` (length [`warp_nparams`](@ref)), channel-major:
each channel contributes `θw`, then `θh`, then `θd`.
"""
function warp_pack!(θ::AbstractVector{T}, warp::MonotonicWarp{T}) where {T<:Real}
    K = warp_bins(warp)
    idx = 0
    @inbounds for j in 1:warp_channels(warp)
        for k in 1:K
            θ[idx + k] = warp.θw[k, j]
        end
        idx += K
        for k in 1:K
            θ[idx + k] = warp.θh[k, j]
        end
        idx += K
        for k in 1:(K - 1)
            θ[idx + k] = warp.θd[k, j]
        end
        idx += K - 1
    end
    return θ
end

"""
    warp_unpack!(warp, θ) -> warp

Inverse of [`warp_pack!`](@ref): scatter `θ` back into the logits and refresh
the derived knots.
"""
function warp_unpack!(warp::MonotonicWarp{T}, θ::AbstractVector{T}) where {T<:Real}
    K = warp_bins(warp)
    idx = 0
    @inbounds for j in 1:warp_channels(warp)
        for k in 1:K
            warp.θw[k, j] = θ[idx + k]
        end
        idx += K
        for k in 1:K
            warp.θh[k, j] = θ[idx + k]
        end
        idx += K
        for k in 1:(K - 1)
            warp.θd[k, j] = θ[idx + k]
        end
        idx += K - 1
    end
    return refresh_knots!(warp)
end

"""
    warp_bin(warp, j, y) -> Int

Index of the bin containing `y` in channel `j`, or `0` when `y` is on the
identity tail. Knot columns are ascending, so this is a binary search.
"""
@inline function warp_bin(warp::MonotonicWarp{T}, j::Int, y::Real) where {T<:Real}
    (y <= warp.lo[j] || y >= warp.hi[j]) && return 0
    return searchsortedlast(view(warp.pX, :, j), T(y))
end

#=
One bin of the forward map. Returns `(z, ℓ)` where `ℓ = log g'(y)`.

`Dn` cannot vanish: with positive widths, heights and knot derivatives,
`Dn = s + (δₖ + δₖ₊₁ - 2s)·u ≥ s/2 + (δₖ + δₖ₊₁)/4 > 0` because `u = ξ(1-ξ) ≤ ¼`.
That is the algebraic reason this parameterization needs no monotonicity guard.
=#
@inline function _rqs_bin_forward(
    Xk::T, Xk1::T, Yk::T, Yk1::T, δk::T, δk1::T, y::T
) where {T<:Real}
    Δ = Xk1 - Xk
    Γ = Yk1 - Yk
    s = Γ / Δ
    ξ = (y - Xk) / Δ
    ξ̄ = one(T) - ξ
    u = ξ * ξ̄

    a = δk + δk1 - 2 * s
    Dn = s + a * u
    M = s * ξ * ξ + δk * u
    N = Γ * M
    z = Yk + N / Dn

    P = δk1 * ξ * ξ + 2 * s * u + δk * ξ̄ * ξ̄
    ℓ = 2 * log(s) + log(P) - 2 * log(Dn)

    return z, ℓ
end

#=
One bin of the inverse map. `g` is strictly increasing, so inverting it on a bin
is a quadratic root; the `-b - sqrt(...)` form is the numerically stable branch
(Durkan et al., appendix A). Returns the pre-image `y`.
=#
@inline function _rqs_bin_inverse(
    Xk::T, Xk1::T, Yk::T, Yk1::T, δk::T, δk1::T, z::T
) where {T<:Real}
    Δ = Xk1 - Xk
    Γ = Yk1 - Yk
    s = Γ / Δ
    Δz = z - Yk
    a2 = δk1 + δk - 2 * s

    a = Γ * (s - δk) + Δz * a2
    b = Γ * δk - Δz * a2
    c = -s * Δz

    denom = -b - sqrt(b * b - 4 * a * c)
    ξ = 2 * c / denom
    return Xk + ξ * Δ
end

"""
    warp_forward(warp, j, y) -> (z, ℓ)

Apply channel `j`'s spline to the scalar `y`, returning the image `z = g_j(y)`
and `ℓ = log g_j'(y)`. Outside `[lo[j], hi[j]]` this is `(y, 0)`.
"""
@inline function warp_forward(warp::MonotonicWarp{T}, j::Int, y::Real) where {T<:Real}
    yT = T(y)
    k = warp_bin(warp, j, yT)
    k == 0 && return yT, zero(T)
    return _rqs_bin_forward(
        warp.pX[k, j],
        warp.pX[k + 1, j],
        warp.pY[k, j],
        warp.pY[k + 1, j],
        warp.dYdX[k, j],
        warp.dYdX[k + 1, j],
        yT,
    )
end

"""
    warp_inverse(warp, j, z) -> y

Pre-image of `z` under channel `j`'s spline. Exact (a quadratic root), not
iterative.
"""
@inline function warp_inverse(warp::MonotonicWarp{T}, j::Int, z::Real) where {T<:Real}
    zT = T(z)
    (zT <= warp.lo[j] || zT >= warp.hi[j]) && return zT
    k = searchsortedlast(view(warp.pY, :, j), zT)
    return _rqs_bin_inverse(
        warp.pX[k, j],
        warp.pX[k + 1, j],
        warp.pY[k, j],
        warp.pY[k + 1, j],
        warp.dYdX[k, j],
        warp.dYdX[k + 1, j],
        zT,
    )
end

"""
    warp_apply!(z, warp, y) -> T

Embed a whole `(p × T)` observation block: `z[j, t] = g_j(y[j, t])`. Returns the
summed log-Jacobian `Σ_{t,j} log g_j'(y[j,t])`, which is the change-of-variables
term the ELBO needs. `z` may alias `y`.
"""
function warp_apply!(
    z::AbstractMatrix{T}, warp::MonotonicWarp{T}, y::AbstractMatrix{T}
) where {T<:Real}
    p = warp_channels(warp)
    size(y, 1) == p || throw(
        DimensionMismatch("warp covers $p channels but the block has $(size(y, 1)) rows"),
    )
    size(z) == size(y) ||
        throw(DimensionMismatch("warp output $(size(z)) ≠ input $(size(y))"))
    total = zero(T)
    @inbounds for t in axes(y, 2), j in 1:p
        zj, ℓj = warp_forward(warp, j, y[j, t])
        z[j, t] = zj
        total += ℓj
    end
    return total
end

"""
    warp_unapply!(y, warp, z) -> y

Inverse of [`warp_apply!`](@ref) over a whole block: `y[j, t] = g_j⁻¹(z[j, t])`.
Used when sampling, where the latent Gaussian draw is pushed back out to
observation space. `y` may alias `z`.
"""
function warp_unapply!(
    y::AbstractMatrix{T}, warp::MonotonicWarp{T}, z::AbstractMatrix{T}
) where {T<:Real}
    p = warp_channels(warp)
    size(z, 1) == p || throw(
        DimensionMismatch("warp covers $p channels but the block has $(size(z, 1)) rows"),
    )
    @inbounds for t in axes(z, 2), j in 1:p
        y[j, t] = warp_inverse(warp, j, z[j, t])
    end
    return y
end

"""
    WarpGradBuffers{T<:Real}

Scratch for the warp's parameter gradient: knot-space accumulators `gX`, `gY`
(`(K+1) × p`) and `gD` (`(K+1) × p`, with the pinned first/last rows left at
zero), plus the `K`-length work vector the softmax chain rule needs.

Gradients are accumulated in *knot* space — each observation touches only the
two knots bounding its bin, so that pass is `O(1)` per sample — and converted to
logit space once per objective evaluation by
[`warp_knot_grad_to_theta!`](@ref), which costs `O(pK)`.
"""
struct WarpGradBuffers{T<:Real}
    gX::Matrix{T}
    gY::Matrix{T}
    gD::Matrix{T}
    work::Vector{T}
end

function WarpGradBuffers(warp::MonotonicWarp{T}) where {T<:Real}
    K = warp_bins(warp)
    p = warp_channels(warp)
    return WarpGradBuffers{T}(
        zeros(T, K + 1, p), zeros(T, K + 1, p), zeros(T, K + 1, p), zeros(T, K)
    )
end

"""
    reset!(buffers::WarpGradBuffers) -> buffers

Zero the knot-space accumulators before an objective evaluation.
"""
function reset!(buf::WarpGradBuffers{T}) where {T<:Real}
    fill!(buf.gX, zero(T))
    fill!(buf.gY, zero(T))
    fill!(buf.gD, zero(T))
    return buf
end

"""
    warp_accumulate_grad!(buf, warp, j, y, ω, with_logjac=true) -> (z, ℓ)

Evaluate channel `j`'s spline at `y` and accumulate into `buf` the knot-space
gradient of

```math
ω · g_j(y) + log g_j'(y)
```

the per-sample integrand of the spline M-step objective: `ω = -(R⁻¹ r)_j` is the
Gaussian pull toward the smoothed prediction, and the second term is the flow's
change-of-variables reward. Returns the same `(z, ℓ)` as
[`warp_forward`](@ref) so a caller needs only this one pass.

Pass `with_logjac = false` to accumulate the pull alone. A mixture objective
(an `SLDS`, where one warp is scored under every regime) calls this once per
component for the same sample, and the change-of-variables term belongs to the
sample, not to a component — so exactly one of those calls carries it.

A sample on the identity tail contributes nothing — the tail is pinned to the
diagonal and the interval endpoints are fixed, so no parameter moves it.
"""
function warp_accumulate_grad!(
    buf::WarpGradBuffers{T},
    warp::MonotonicWarp{T},
    j::Int,
    y::T,
    ω::T,
    with_logjac::Bool=true,
) where {T<:Real}
    k = warp_bin(warp, j, y)
    k == 0 && return y, zero(T)

    Xk = warp.pX[k, j]
    Xk1 = warp.pX[k + 1, j]
    Yk = warp.pY[k, j]
    Yk1 = warp.pY[k + 1, j]
    δk = warp.dYdX[k, j]
    δk1 = warp.dYdX[k + 1, j]

    Δ = Xk1 - Xk
    Γ = Yk1 - Yk
    s = Γ / Δ
    ξ = (y - Xk) / Δ
    ξ̄ = one(T) - ξ
    u = ξ * ξ̄
    v = one(T) - 2 * ξ                      # ∂u/∂ξ

    a = δk + δk1 - 2 * s
    Dn = s + a * u
    Dn2 = Dn * Dn
    M = s * ξ * ξ + δk * u
    N = Γ * M
    z = Yk + N / Dn

    P = δk1 * ξ * ξ + 2 * s * u + δk * ξ̄ * ξ̄
    ℓ = 2 * log(s) + log(P) - 2 * log(Dn)

    #=
    Partials of `z` and `ℓ` with respect to the intermediates (s, Δ, ξ, δₖ,
    δₖ₊₁, Yₖ), which are then pushed onto the knots below. `y` is data, so it is
    held fixed throughout.
    =#
    z_s = (Δ * (M + s * ξ * ξ)) / Dn - (N * (one(T) - 2 * u)) / Dn2
    z_Δ = (s * M) / Dn
    z_δk = (Δ * s * u) / Dn - (N * u) / Dn2
    z_δk1 = -(N * u) / Dn2
    z_ξ = (Δ * s * (2 * s * ξ + δk * v)) / Dn - (N * a * v) / Dn2

    ℓ_s = 2 / s + (2 * u) / P - (2 * (one(T) - 2 * u)) / Dn
    ℓ_δk = (ξ̄ * ξ̄) / P - (2 * u) / Dn
    ℓ_δk1 = (ξ * ξ) / P - (2 * u) / Dn
    ℓ_ξ = (2 * δk1 * ξ + 2 * s * v - 2 * δk * ξ̄) / P - (2 * a * v) / Dn

    jw = with_logjac ? one(T) : zero(T)
    F_s = ω * z_s + jw * ℓ_s
    F_Δ = ω * z_Δ                            # ℓ does not depend on Δ
    F_ξ = ω * z_ξ + jw * ℓ_ξ
    F_δk = ω * z_δk + jw * ℓ_δk
    F_δk1 = ω * z_δk1 + jw * ℓ_δk1
    F_Yk = ω                                 # ∂z/∂Yₖ = 1 at fixed (s, Δ, ξ)

    #=
    Chain onto the knots via s = Γ/Δ, Δ = Xₖ₊₁ - Xₖ, ξ = (y - Xₖ)/Δ. Two
    invariants worth keeping in mind when reading this: shifting both Y knots
    shifts `z` by the same amount and leaves `ℓ` alone, and shifting both X
    knots only moves `ξ`.
    =#
    s_over_Δ = s / Δ
    @inbounds begin
        buf.gY[k, j] += F_Yk - F_s / Δ
        buf.gY[k + 1, j] += F_s / Δ
        buf.gX[k, j] += F_s * s_over_Δ - F_Δ + F_ξ * (ξ - one(T)) / Δ
        buf.gX[k + 1, j] += -F_s * s_over_Δ + F_Δ - F_ξ * ξ / Δ
        buf.gD[k, j] += F_δk
        buf.gD[k + 1, j] += F_δk1
    end

    return z, ℓ
end

#=
Push a knot-position gradient through `X = lo .+ W·cumsum(σ)` and the floored
softmax, adding the result into `out`.

`∂X_m/∂σ_i = W·[i ≤ m-1]`, so the gradient with respect to the fractions is a
suffix sum of the knot gradients; the floor contributes the constant factor
`1 - K·min_bin`. The softmax Jacobian `σ_i(δ_ia - σ_a)` then annihilates any
constant shift in that vector, which is exactly why the last knot (pinned at
`hi`, and contributing the same amount to every entry) needs no special case.
=#
function _accumulate_logit_grad!(
    out::AbstractVector,
    gknot::AbstractVector,
    σ::AbstractVector,
    work::AbstractVector,
    width::Real,
    min_bin::Real,
)
    # Element types are left off the signature for the reason given on
    # `_floored_softmax!` above.
    T = eltype(σ)
    K = length(σ)
    scale = width * (one(T) - K * min_bin)

    suffix = zero(T)
    @inbounds for i in K:-1:1
        suffix += gknot[i + 1]
        work[i] = scale * suffix
    end

    #=
    `σ` carries the floor, so the softmax probability is
    `π = (σ - min_bin)/(1 - K·min_bin)`; the `1 - K·min_bin` factor of
    `∂σ/∂π` is already folded into `scale`, so `work` holds `∂Q/∂π` and the
    softmax Jacobian `π_a(δ_ab - π_b)` closes the chain. The weighted mean must
    be taken against `π`, not `σ` — they differ by the floor.
    =#
    inv_scale = one(T) / (one(T) - K * min_bin)
    dot_π = zero(T)
    @inbounds for i in 1:K
        dot_π += (σ[i] - min_bin) * inv_scale * work[i]
    end
    @inbounds for a in 1:K
        πa = (σ[a] - min_bin) * inv_scale
        out[a] += πa * (work[a] - dot_π)
    end
    return out
end

"""
    warp_knot_grad_to_theta!(G, warp, buf) -> G

Convert the knot-space gradients accumulated in `buf` into the flat logit-space
gradient `G` (same layout as [`warp_pack!`](@ref)), *adding* into `G` so a
caller can seed it with a prior/penalty term first.
"""
function warp_knot_grad_to_theta!(
    G::AbstractVector{T}, warp::MonotonicWarp{T}, buf::WarpGradBuffers{T}
) where {T<:Real}
    K = warp_bins(warp)
    idx = 0
    @inbounds for j in 1:warp_channels(warp)
        width = warp.hi[j] - warp.lo[j]

        _accumulate_logit_grad!(
            view(G, (idx + 1):(idx + K)),
            view(buf.gX, :, j),
            view(warp.sw, :, j),
            buf.work,
            width,
            warp.min_bin,
        )
        idx += K

        _accumulate_logit_grad!(
            view(G, (idx + 1):(idx + K)),
            view(buf.gY, :, j),
            view(warp.sh, :, j),
            buf.work,
            width,
            warp.min_bin,
        )
        idx += K

        # δ_{k+1} = exp(θᵈ_k), so ∂/∂θᵈ_k = δ_{k+1}·∂/∂δ_{k+1}. The pinned
        # boundary derivatives (rows 1 and K+1 of `gD`) are not parameters.
        for k in 1:(K - 1)
            G[idx + k] += buf.gD[k + 1, j] * warp.dYdX[k + 1, j]
        end
        idx += K - 1
    end
    return G
end

"""
    warp_bounds(y, quantile_margin=0.0) -> (lo, hi)

Per-channel knot bounds inferred from observations: the channel range, widened
by `quantile_margin` times the range on each side. `y` is a matrix or a vector
of per-trial matrices.

A channel that is constant across the dataset gets a unit-wide interval centered
on its value, so the warp stays well-defined (and stays the identity — there is
nothing in that channel to reshape).
"""
function warp_bounds(
    y::AbstractVector{<:AbstractMatrix{T}}, margin::Real=zero(T)
) where {T<:Real}
    isempty(y) && throw(ArgumentError("warp_bounds needs at least one trial"))
    p = size(first(y), 1)
    lo = fill(T(Inf), p)
    hi = fill(T(-Inf), p)
    for yt in y
        size(yt, 1) == p || throw(
            DimensionMismatch("trials disagree on channel count: $p vs $(size(yt, 1))")
        )
        @inbounds for t in axes(yt, 2), j in 1:p
            v = yt[j, t]
            v < lo[j] && (lo[j] = v)
            v > hi[j] && (hi[j] = v)
        end
    end

    m = T(margin)
    @inbounds for j in 1:p
        span = hi[j] - lo[j]
        if !(span > zero(T)) || !isfinite(span)
            center = isfinite(lo[j]) ? lo[j] : zero(T)
            lo[j] = center - T(0.5)
            hi[j] = center + T(0.5)
        else
            pad = m * span
            lo[j] -= pad
            hi[j] += pad
        end
    end
    return lo, hi
end

function warp_bounds(y::AbstractMatrix{T}, margin::Real=zero(T)) where {T<:Real}
    return warp_bounds([y], margin)
end

function warp_bounds(y::AbstractArray{T,3}, margin::Real=zero(T)) where {T<:Real}
    return warp_bounds([view(y, :, :, n) for n in axes(y, 3)], margin)
end

"""
    copy_warp!(dest, src) -> dest

Copy `src`'s parameters and derived knots into `dest`, preserving `dest`'s
object identity. Used when restoring a parameter snapshot: anything holding a
reference to `dest` (an M-step context, a shadow model) keeps pointing at the
live warp.

# Throws
- `DimensionMismatch` when the two warps differ in channel or bin count.
"""
function copy_warp!(dest::MonotonicWarp{T}, src::MonotonicWarp{T}) where {T<:Real}
    (warp_channels(dest) == warp_channels(src) && warp_bins(dest) == warp_bins(src)) ||
        throw(
            DimensionMismatch(
                "warp shapes differ: $(warp_channels(dest))×$(warp_bins(dest)) vs " *
                "$(warp_channels(src))×$(warp_bins(src))",
            ),
        )
    copyto!(dest.lo, src.lo)
    copyto!(dest.hi, src.hi)
    copyto!(dest.θw, src.θw)
    copyto!(dest.θh, src.θh)
    copyto!(dest.θd, src.θd)
    return refresh_knots!(dest)
end
