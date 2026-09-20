#=============================================================================
Spline-Gaussian Observations — nonlinear manifold discovery for an LDS.

    y_t = g⁻¹(z_t),   z_t | x_t ~ N(C x_t + d + D uy_t, R)

with `g` an element-wise monotonic rational-quadratic spline (one per channel;
see `numerics/rqspline.jl`). Because `g` is a diffeomorphism the model is a
*normalizing flow* over the emission:

```math
p(y_t | x_t) = N(g(y_t); C x_t + d + D uy_t, R) · ∏_j g_j'(y_{tj})
```

Manifold discovery. `C` is `p × k` with `k < p` in the interesting case, so `z`
concentrates near a `k`-dimensional affine subspace and `y = g⁻¹(z)` traces a
curved `k`-manifold in observation space. What one layer of element-wise warps
buys, precisely, is a **Gaussian-copula (nonparanormal) LDS**: arbitrary
continuous per-channel marginals — rectification, saturation, skew, heavy tails
— over linear-Gaussian latent dynamics. It does *not* mix channels, so it cannot
bend a manifold sideways; that is the price of having each `g_j` be a directly
interpretable per-channel tuning curve.

Why exact EM works here
-----------------------
Hold `g` fixed. Then `z_{1:T} = g(y_{1:T})` is *data* and the model in `z` is an
ordinary linear-Gaussian SSM, so

  E-step. `q(x_{1:T}) = p(x_{1:T} | z_{1:T})` is the exact posterior from the
  usual smoother — this is not a variational approximation. The Jacobian term
  does not involve `x` at all.

  CM-step 1. With `g` fixed the Jacobian term is constant, so `[C d D]` and `R`
  come from the standard closed-form Gaussian updates run on `z`.

  CM-step 2. With everything else fixed, and writing `μ̂_t = C x̂_t + d + D uy_t`
  for the *smoothed* emission mean, the expectation over `q` separates:

```math
E_q[(g(y_t) - μ_t)' R⁻¹ (g(y_t) - μ_t)]
    = ‖L⁻¹(g(y_t) - μ̂_t)‖²  +  tr(R⁻¹ C Σ_t C')
```

  and the trace is constant in the spline parameters. The warp objective is
  therefore exactly a normalizing-flow MLE against a per-timestep Gaussian base:

```math
Q(φ) = -½ Σ_{n,t} ‖L⁻¹(g_φ(y_{nt}) - μ̂_{nt})‖²  +  Σ_{n,t,j} log g'_{φ_j}(y_{ntj})
```

  Only the smoothed *mean* enters; the smoothed covariance drops out.

Both CM steps maximize the same `Q(θ, q)` with `q` fixed, so this is ECM
(Meng & Rubin 1993) and the observed-data log-likelihood increases monotonically.
The warp step is accepted only when it genuinely improves, which keeps that
guarantee under a stalled line search.

Implementation. `[C d D]` and `R` obey the same algebra as a plain Gaussian
emission, so this file does not re-derive any of it: it builds a
[`GaussianObservationModel`](@ref) *sharing the same arrays by reference*
(`_gaussian_shadow`) and runs the existing, tuned pipeline on a shadow `Data`
holding `z`. In-place M-step writes land back in this model's arrays for free.
=============================================================================#

"""
    SplineGaussianObservationModel{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}

Gaussian emission composed with a learned element-wise monotonic warp — the
observation model for nonlinear manifold discovery in a
[`LinearDynamicalSystem`](@ref). See the file header for the model and the
ECM algorithm.

# Fields
- `C::M`: Emission matrix, `(obs_dim × latent_dim)`. May be tall (`k < p`); that
    is the manifold-discovery regime.
- `R::M`: Residual covariance of the *embedded* observations `z = g(y)`,
    `(obs_dim × obs_dim)`. Shape constrained by `R_structure`.
- `d::V`: Emission bias, length `obs_dim`.
- `warp::MonotonicWarp{T}`: the per-channel splines. `obs_dim` channels.
- `D::M`: Optional observation-input matrix, `(obs_dim × uy_dim)`.
- `R_structure::Symbol = :diagonal`: `:diagonal` or `:full`. With `k < p` a
    diagonal `R` is the identified, factor-analysis-style choice — it forces the
    latent state to carry all cross-channel structure, so `C` spans the manifold
    and `R` is per-channel noise. `:full` reproduces the unconstrained
    [`GaussianObservationModel`](@ref) behaviour, at the cost of letting residual
    correlation compete with the latent subspace.
- `R_floor::T`: numerical floor on `diag(R)`. **Not** a statistical prior — a
    safety net. Like any normalizing flow with a free noise level (and like a
    Gaussian mixture with free component variances), this model's likelihood is
    *unbounded above*: a warp that interpolates the smoothed prediction on one
    channel drives that channel's residual to zero and the density to infinity.
    The floor keeps `R` factorizable if a fit walks into that direction, and a
    warning fires when it binds — which is the signal to add an `R_prior`, cut
    `n_bins`, or raise `spline_ridge`. The convenience constructor defaults it to
    `1e-10 · mean((hi - lo)²)`, far below any real noise scale; the raw
    constructor defaults it to `0` (off).
- `spline_ridge::T = 1e-3`: weight of the ridge `-½·λ·‖φ‖²` on the spline
    logits — a proper log-prior shrinking the warp toward the identity. It also
    removes the softmax shift degeneracy (adding a constant to a channel's width
    logits leaves the map unchanged), which would otherwise be an exactly flat
    direction for the optimizer. Set to `0` only if you have a lot of data per
    channel.
- `R_prior::Union{Nothing,IWPrior{T}} = nothing`: Inverse-Wishart prior on `R`.
    Honoured under both `R_structure`s; with `:diagonal` the MAP is the
    diagonal-restricted one, `R_jj = (Ψ_jj + S_jj) / (ν + n + p + 1)`.
- `CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing`: matrix-normal prior
    on the stacked `[C d D]`, exactly as for a Gaussian emission.
- `depends_on`, `group_seeds`, `variants`: present for interface parity with the
    other emissions, but grouping is **not** supported here — see below.

# Unsupported combinations
Both are refused with a clear error rather than silently approximated:

- `depends_on` parameter grouping. A group would need its own warp to be
  coherent with this package's session stitching (groups may observe different
  channel sets), which the grouped M-step does not build. Fit each group
  separately.
- An [`LQRStateModel`](@ref). Its state M-step is structural (it constrains the
  transition to the symplectic form a control problem implies), and the spline
  driver runs the generic Gaussian state updates, which would discard that.

# Identifiability
The warp's interval endpoints are pinned to the diagonal and the map is the
identity outside `[lo, hi]` (Durkan et al.'s convention, which `MonotonicSplines.jl`
also enforces). Without that pinning `g ↦ α ⊙ g + β` would be an exact
`2·obs_dim`-dimensional flat direction: the `log det R` and log-Jacobian terms
cancel identically when the rescaling is absorbed into `C`, `d` and `R`.

# Initialization
All spline logits start at zero, which is *exactly* the identity map, so a fresh
`SplineGaussianObservationModel` is the corresponding linear-Gaussian emission
and fitting can only improve on it.

# Example
```julia
sm = GaussianStateModel(A, Q, b, x0, P0)
om = SplineGaussianObservationModel(C, R, d; y = Y, n_bins = 8)
lds = LinearDynamicalSystem(sm, om)
elbos = fit!(lds, Y)

# the learned tuning curve of channel j, on the observation scale
zs = [warp_forward(lds.obs_model.warp, j, y)[1] for y in ys]
```

See also [`MonotonicWarp`](@ref), [`GaussianObservationModel`](@ref).
"""
Base.@kwdef mutable struct SplineGaussianObservationModel{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}
} <: AbstractObservationModel{T}
    C::M
    R::M
    d::V
    warp::MonotonicWarp{T}
    D::M = zeros(eltype(C), size(C, 1), 0)
    R_structure::Symbol = :diagonal
    spline_ridge::T = T(1e-3)
    R_floor::T = zero(T)
    R_prior::Union{Nothing,IWPrior{T}} = nothing
    CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing
    depends_on::Union{Nothing,NamedTuple} = nothing
    group_seeds::Union{Nothing,AbstractDict} = nothing
    variants::Union{Nothing,Vector{SplineGaussianObservationModel{T,M,V}}} = nothing
end

"""
    SplineGaussianObservationModel(C, R, d; kwargs...)

Build a spline-Gaussian emission with an identity warp.

The warp's per-channel interval must come from somewhere; supply exactly one of

- `y`: observations in any shape `fit!` accepts, from which the per-channel
  range is taken (widened by `margin`, a fraction of the range), or
- `bounds = (lo, hi)`: explicit per-channel vectors.

The interval is **fixed** for the lifetime of the model — it is part of the
model definition, not a fitted parameter, so log-densities stay comparable
across EM iterations and across datasets.

# Keywords
- `y`, `bounds`, `margin = 0.05`: knot interval, as above.
- `n_bins::Int = 8`: spline bins per channel (`≥ 2`). Each channel then has
  `3·n_bins - 1` free parameters.
- `min_bin::Real = 1e-3`: floor on a bin's width/height fraction.
- `D`, `R_structure`, `spline_ridge`, `R_prior`, `CD_prior`: as on the type.
"""
function SplineGaussianObservationModel(
    C::M,
    R::M,
    d::V;
    y=nothing,
    bounds::Union{Nothing,Tuple{<:AbstractVector,<:AbstractVector}}=nothing,
    margin::Real=0.05,
    n_bins::Int=8,
    min_bin::Real=1e-3,
    D::Union{Nothing,M}=nothing,
    R_structure::Symbol=:diagonal,
    spline_ridge::Real=1e-3,
    R_floor::Union{Nothing,Real}=nothing,
    R_prior::Union{Nothing,IWPrior{T}}=nothing,
    CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}}=nothing,
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    p = size(C, 1)
    lo, hi = _warp_bounds_from(y, bounds, margin, p, T)
    warp = MonotonicWarp(lo, hi; n_bins=n_bins, min_bin=min_bin)
    return SplineGaussianObservationModel{T,M,V}(;
        C=C,
        R=R,
        d=d,
        warp=warp,
        D=(D === nothing ? zeros(eltype(C), p, 0) : D),
        R_structure=_checked_R_structure(R_structure),
        spline_ridge=T(spline_ridge),
        R_floor=(R_floor === nothing ? T(1e-10) * mean(abs2, hi .- lo) : T(R_floor)),
        R_prior=R_prior,
        CD_prior=CD_prior,
    )
end

function _checked_R_structure(s::Symbol)
    s in (:diagonal, :full) ||
        throw(ArgumentError("R_structure must be :diagonal or :full; got :$s"))
    return s
end

function _warp_bounds_from(y, bounds, margin, p::Int, ::Type{T}) where {T<:Real}
    if bounds !== nothing
        y === nothing || throw(
            ArgumentError(
                "pass either `y` (infer the warp interval from data) or `bounds` " *
                "(give it explicitly), not both",
            ),
        )
        lo, hi = bounds
        length(lo) == p && length(hi) == p || throw(
            DimensionMismatchError("warp bounds length", p, min(length(lo), length(hi)))
        )
        return collect(T, lo), collect(T, hi)
    end
    y === nothing && throw(
        ArgumentError(
            "a spline emission needs a knot interval: pass `y = <observations>` to " *
            "take it from the data range, or `bounds = (lo, hi)` to set it directly",
        ),
    )
    return warp_bounds(_warp_bounds_trials(y, T), margin)
end

# Normalize the three public observation shapes into the vector-of-matrices form
# `warp_bounds` walks. Deliberately independent of `Data`, so a model can be
# built before an `lds` exists to validate against.
_warp_bounds_trials(y::AbstractMatrix{T}, ::Type{T}) where {T<:Real} = [y]
function _warp_bounds_trials(y::AbstractArray{T,3}, ::Type{T}) where {T<:Real}
    return [view(y, :, :, n) for n in axes(y, 3)]
end
function _warp_bounds_trials(
    y::AbstractVector{<:AbstractMatrix{T}}, ::Type{T}
) where {T<:Real}
    return y
end
function _warp_bounds_trials(y, ::Type{T}) where {T<:Real}
    return throw(
        ArgumentError(
            "cannot infer warp bounds from a $(typeof(y)); pass observations as a " *
            "matrix, a 3-D array, or a vector of per-trial matrices (or set " *
            "`bounds = (lo, hi)` directly)",
        ),
    )
end

# ============================================================================
# Model-level plumbing
# ============================================================================

#=
Quadratic in the latent state: given `z`, the emission is Gaussian, and the
Jacobian term does not involve `x` at all. So the single-step Newton smoother
and the shared-covariance fast path both apply verbatim.
=#
_emission_is_quadratic(::SplineGaussianObservationModel) = true

#=
Three `fit_bool` slots: `[C d D]`, `R`, and the warp. The warp is genuinely
separable from `[C d D]` — freezing it (`spline = false`) recovers an ordinary
Gaussian LDS fit, which is the natural baseline to compare a spline fit against.
=#
_obs_block_names(::SplineGaussianObservationModel) = (:C, :R, :spline)

"""
    _has_warped_member(obs_model) -> Bool

Whether an emission contains a learned warp, and so needs the embedding pass
before the ordinary Gaussian machinery can run.

A composite's `QUAD` type parameter cannot carry this — it is the `AND` of a
*different* property — so the entry points in `fit_LDS.jl` branch on this at run
time instead of by dispatch.
"""
_has_warped_member(::AbstractObservationModel) = false
_has_warped_member(::SplineGaussianObservationModel) = true

#=
Unrolled over the (heterogeneous, statically-sized) tuple of members so the
result is a compile-time constant, exactly as `_all_quadratic` is: `any` over
`values(models)` would leave it an abstract-tuple reduction.
=#
_any_warped(::Tuple{}) = false
function _any_warped(models::Tuple)
    return _has_warped_member(first(models)) || _any_warped(Base.tail(models))
end

_has_warped_member(c::CompositeObservationModel) = _any_warped(values(_models(c)))

_group_names(::SplineGaussianObservationModel) = (:C, :R, :spline)
_all_param_names(::SplineGaussianObservationModel) = (:C, :d, :D, :R, :warp)
_valid_param_names(::SplineGaussianObservationModel) = ":C, :d, :D, :R, :warp"

function _param_group(::SplineGaussianObservationModel, name::Symbol)
    name in (:C, :d, :D) && return :C
    name === :R && return :R
    name === :warp && return :spline
    return nothing
end

function _group_members(om::SplineGaussianObservationModel, group::Symbol)
    group === :C && return (:C, :d, :D)
    group === :R && return (:R,)
    group === :spline && return (:warp,)
    return ()
end

function _extract_obs_params(om::SplineGaussianObservationModel{T}) where {T}
    return (C=om.C, R=om.R, d=om.d, D=om.D, warp=om.warp)
end

@inline function _check_uy(
    cs, expected_dim::Int, tsteps::Int, ::SplineGaussianObservationModel{T}
) where {T}
    return _check_ux(cs, expected_dim, tsteps, "uy", T)
end

@inline function _normalize_multitrial_uy(
    cs, expected_dim::Int, tsteps_per_trial, ::Type{T}, ::SplineGaussianObservationModel
) where {T<:Real}
    return _normalize_multitrial_ux(cs, expected_dim, tsteps_per_trial, T, "uy")
end

"""
    _validate_obs_model(obs_model::SplineGaussianObservationModel, obs_dim, latent_dim)

Validate a spline-Gaussian emission: the Gaussian half exactly as for a
[`GaussianObservationModel`](@ref), plus that the warp covers every channel and
that `R_structure` is one of the two supported shapes.
"""
function _validate_obs_model(
    obs_model::SplineGaussianObservationModel{T}, obs_dim::Int, latent_dim::Int
) where {T}
    if size(obs_model.C) != (obs_dim, latent_dim)
        throw(DimensionMismatchError("C matrix", (obs_dim, latent_dim), size(obs_model.C)))
    end
    if size(obs_model.R) != (obs_dim, obs_dim)
        throw(DimensionMismatchError("R matrix", (obs_dim, obs_dim), size(obs_model.R)))
    end
    if !issymmetric(obs_model.R)
        max_asym = maximum(abs.(obs_model.R - obs_model.R'))
        throw(NotSymmetricError("R matrix", max_asym))
    end
    if !isposdef(obs_model.R)
        min_eval = minimum(eigvals(obs_model.R))
        throw(NotPositiveDefiniteError("R matrix", min_eval))
    end
    if length(obs_model.d) != obs_dim
        throw(DimensionMismatchError("observation bias d", obs_dim, length(obs_model.d)))
    end
    if size(obs_model.D, 1) != obs_dim
        throw(DimensionMismatchError("D matrix rows", obs_dim, size(obs_model.D, 1)))
    end
    if warp_channels(obs_model.warp) != obs_dim
        throw(
            DimensionMismatchError("warp channels", obs_dim, warp_channels(obs_model.warp))
        )
    end
    _checked_R_structure(obs_model.R_structure)
    obs_model.spline_ridge >= zero(T) || throw(
        ArgumentError("spline_ridge must be non-negative; got $(obs_model.spline_ridge)"),
    )
    obs_model.R_floor >= zero(T) ||
        throw(ArgumentError("R_floor must be non-negative; got $(obs_model.R_floor)"))
    return nothing
end

function Base.show(io::IO, om::SplineGaussianObservationModel; gap="")
    p, k = size(om.C)
    K = warp_bins(om.warp)
    println(io, gap, "Spline-Gaussian Observation Model:")
    println(io, gap, "---------------------------------")
    if p > 3 || k > 3
        println(io, gap, " size(C) = ($p, $k)")
        println(io, gap, " size(R) = ($p, $p)   [$(om.R_structure)]")
        println(io, gap, " size(d) = ($(length(om.d)),)")
        println(io, gap, " size(D) = ($(size(om.D,1)), $(size(om.D,2)))")
    else
        println(io, gap, " C = $(round.(om.C, digits=2))")
        println(io, gap, " R = $(round.(om.R, digits=2))   [$(om.R_structure)]")
        println(io, gap, " d = $(round.(om.d, digits=2))")
        println(io, gap, " D = $(round.(om.D, digits=2))")
    end
    identity_note = is_identity_warp(om.warp) ? "  (currently the identity)" : ""
    println(
        io,
        gap,
        " warp = $p channels × $K bins, $(warp_nparams(om.warp)) params$identity_note",
    )
    println(io, gap, " spline_ridge = $(om.spline_ridge)")
    _show_depends_on(io, om; gap=gap)
    return nothing
end

_obs_fit_labels(::SplineGaussianObservationModel) = ["C (and d, D)", "R", "warp (spline)"]

# ============================================================================
# The Gaussian shadow
# ============================================================================

"""
    _gaussian_shadow(om::SplineGaussianObservationModel) -> GaussianObservationModel

A [`GaussianObservationModel`](@ref) sharing `C`, `R`, `d`, `D` and the priors
**by reference**. Everything the linear half of the model needs — the smoother
constants, `Q_obs!`, `update_C_d!`, the emission kernels — is already written
against that type, and every M-step write into it is an in-place `copyto!`, so
updates through the shadow are visible from the spline model with no copy-back.
"""
function _gaussian_shadow(
    om::SplineGaussianObservationModel{T,M,V}
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    return GaussianObservationModel{T,M,V}(;
        C=om.C, R=om.R, d=om.d, D=om.D, R_prior=om.R_prior, CD_prior=om.CD_prior
    )
end

"""
    _gaussian_shadow(lds) -> LinearDynamicalSystem

The whole system with its emission replaced by the shadow above, sharing the
state model by reference too. `fit_bool` is truncated to the six slots the
Gaussian M-step reads; the seventh (the warp) is handled by
[`_spline_warp_mstep!`](@ref).
"""
function _gaussian_shadow(
    lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel{T},O<:SplineGaussianObservationModel{T}}
    om = _gaussian_shadow(lds.obs_model)
    return LinearDynamicalSystem{T,S,typeof(om)}(
        lds.state_model,
        om,
        lds.latent_dim,
        lds.obs_dim,
        lds.ux_dim,
        lds.uy_dim,
        lds.fit_bool[1:6],
    )
end

# ============================================================================
# Embedding cache
# ============================================================================

"""
    SplineEmbedding{T}

The per-fit embedding buffers: the embedded observations `z = g(y)` wrapped in a
shadow [`Data`](@ref) that every downstream Gaussian routine consumes in place
of the raw one, the running total log-Jacobian, and the smoothed emission means
`μ̂ = C x̂ + d + D uy` the warp M-step reads.

`z` is refreshed at the top of every E-step because the spline moved in the
previous M-step — which is also why the data-only constant blocks
(`_td_init_const_blocks!`) and the batched-smoother staging buffers have to be
rebuilt each iteration rather than once at fit entry.
"""
struct SplineEmbedding{T<:Real,DT}
    z::Vector{Matrix{T}}
    data::DT
    mu::Vector{Matrix{T}}
    logjac::Base.RefValue{T}
end

function SplineEmbedding(data::Data{T}, obs_dim::Int) where {T<:Real}
    y = data.y
    z = [Matrix{T}(undef, obs_dim, size(yt, 2)) for yt in y]
    mu = [Matrix{T}(undef, obs_dim, size(yt, 2)) for yt in y]
    zdata = Data(z, data.ux, data.uy, data.tsteps)
    return SplineEmbedding{T,typeof(zdata)}(z, zdata, mu, Ref(zero(T)))
end

"""
    _embed!(emb, warp, data) -> emb

Write `z = g(y)` into the embedding buffers and record the total log-Jacobian
`Σ_{n,t,j} log g_j'(y_{ntj})`.
"""
function _embed!(
    emb::SplineEmbedding{T}, warp::MonotonicWarp{T}, data::Data{T}
) where {T<:Real}
    total = zero(T)
    y = data.y
    for n in eachindex(y)
        total += warp_apply!(emb.z[n], warp, y[n])
    end
    emb.logjac[] = total
    return emb
end

"""
    _spline_logprior(om) -> T

The warp's log-prior at its current parameters: `-½·λ·‖φ‖²`, the ridge shrinking
the spline toward the identity. Reported in the ELBO for the same reason
[`mn_logprior_term`](@ref) is — the M-step maximizes the penalized objective, so
a trace that omitted the penalty could decrease.
"""
function _spline_logprior(om::SplineGaussianObservationModel{T}) where {T<:Real}
    λ = om.spline_ridge
    iszero(λ) && return zero(T)
    w = om.warp
    ss = sum(abs2, w.θw) + sum(abs2, w.θh) + sum(abs2, w.θd)
    return -T(0.5) * λ * ss
end

# ============================================================================
# M-step: R with an optional diagonal constraint
# ============================================================================

"""
    _finalize_R_diag!(glds, S_res, N)

Diagonal counterpart of [`_finalize_R!`](@ref). Restricting the Gaussian
likelihood to diagonal `R` makes the channels independent given `x`, so the MLE
is just the diagonal of the residual scatter, `R_jj = S_jj / N`.

Under an `IWPrior` the restricted density is
`∏_j R_jj^{-(ν+p+1)/2} exp(-½ Ψ_jj / R_jj)`, an independent inverse-gamma per
channel, whose MAP is `R_jj = (Ψ_jj + S_jj) / (ν + N + p + 1)` — the same
denominator as [`iw_map`](@ref), which is why the two agree when `S` and `Ψ`
happen to be diagonal.
"""
function _finalize_R_diag!(
    glds::LinearDynamicalSystem{T,S,O}, S_res::AbstractMatrix{T}, N::T
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:GaussianObservationModel{T}}
    p = glds.obs_dim
    R = glds.obs_model.R
    prior = glds.obs_model.R_prior
    fill!(R, zero(T))
    if prior === nothing
        @inbounds for j in 1:p
            R[j, j] = S_res[j, j] / N
        end
    else
        Ψ, ν = prior.Ψ, prior.ν
        denom = ν + N + T(p) + one(T)
        @inbounds for j in 1:p
            R[j, j] = (Ψ[j, j] + S_res[j, j]) / denom
        end
    end
    return nothing
end

"""
    _spline_update_R!(om, glds, suf, sws)

`R` M-step honouring `om.R_structure` and `om.R_floor`. The residual scatter
itself is exactly the Gaussian one — accumulated by the shared
[`_accumulate_obs_scatter!`](@ref) from the shadow system `glds`, which shares
`om`'s arrays — and only the final projection differs.

Takes the spline emission rather than the enclosing system so a composite member
reaches it the same way a standalone emission does; `glds` carries the
`fit_bool` flag either way.
"""
function _spline_update_R!(
    om::SplineGaussianObservationModel{T},
    glds::LinearDynamicalSystem{T,S,GO},
    suf::SufficientStatistics{T},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:AbstractGaussianStateModel{T},GO<:GaussianObservationModel{T}}
    glds.fit_bool[6] || return nothing
    S_res = sws.elbo.obs_work
    fill!(S_res, zero(T))
    _accumulate_obs_scatter!(S_res, glds, suf, sws)
    _accumulate_cd_prior_scatter!(S_res, glds, sws)
    if om.R_structure === :diagonal
        _finalize_R_diag!(glds, S_res, T(suf.obs_n))
    else
        _finalize_R!(glds, S_res, T(suf.obs_n))
    end
    _apply_R_floor!(glds.obs_model.R, om.R_floor)
    return nothing
end

"""
    _apply_R_floor!(R, floor) -> R

Raise any diagonal entry of `R` below `floor` up to it. See the `R_floor` field
of [`SplineGaussianObservationModel`](@ref) for why this exists; binding is a
symptom worth acting on, so it warns rather than passing silently.
"""
function _apply_R_floor!(R::AbstractMatrix{T}, floor::T) where {T<:Real}
    floor > zero(T) || return R
    hit = 0
    @inbounds for j in axes(R, 1)
        if R[j, j] < floor
            R[j, j] = floor
            hit += 1
        end
    end
    if hit > 0
        @warn(
            "spline emission: R hit its numerical floor on $hit channel(s) — the warp " *
                "is driving a channel's residual to zero (the flow's unbounded-likelihood " *
                "direction). Set an `R_prior`, reduce `n_bins`, or raise `spline_ridge`.",
            R_floor = floor,
            maxlog = 3,
        )
    end
    return R
end

"""
    _spline_linear_mstep!(lds, glds, suf, sws)

CM-step 1: the state parameters and the linear half of the emission, all in
closed form from the aggregated sufficient statistics of the *embedded*
observations. Identical to [`mstep!`](@ref) for a Gaussian LDS except that `R`
goes through [`_spline_update_R!`](@ref).
"""
function _spline_linear_mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    glds::LinearDynamicalSystem{T,S,GO},
    suf::SufficientStatistics{T},
    sws::SmoothWorkspace{T},
) where {
    T<:Real,
    S<:AbstractGaussianStateModel{T},
    O<:SplineGaussianObservationModel{T},
    GO<:GaussianObservationModel{T},
}
    update_initial_state_mean!(glds, suf)
    update_initial_state_covariance!(glds, suf, sws)
    update_A_b!(glds, suf, sws)
    update_Q!(glds, suf, sws)
    update_C_d!(glds, suf, sws)
    _spline_update_R!(lds.obs_model, glds, suf, sws)
    return nothing
end

# ============================================================================
# M-step: the warp (normalizing-flow conditional maximization)
# ============================================================================

"""
    _SplineMStepCtx{T,YV}

Everything the warp objective needs, allocated once per fit.

The objective is written for a **mixture** of Gaussian components,

```math
Q(φ) = -½ Σ_{n,t} Σ_c γ_{c,n,t} ‖L_c⁻¹(g_φ(y_{nt}) - μ̂^{(c)}_{nt})‖²
       + Σ_{n,t,j} log g'_{φ_j}(y_{ntj}) - ½ λ‖φ‖²
```

because that is what an `SLDS` needs: with the warp shared across regimes (see
`fit_slds_spline.jl`), each timestep's residual is scored against every regime's
`(C_c, d_c, R_c)`, weighted by its responsibility `γ_{c,n,t}`. A plain LDS is the
one-component case with `gamma = nothing`, read as `γ ≡ 1`, and then this is
exactly the single-Gaussian objective of the file header.

Chunked over trials so the objective and its gradient run in parallel: each chunk
owns a [`WarpGradBuffers`](@ref) and its own `obs_dim` scratch, and the shared
`warp` is written only between parallel sections.

# Fields
- `mu::Vector{Vector{Matrix{T}}}`: `mu[c][n]` is component `c`'s smoothed
    emission mean for trial `n`, `(obs_dim × T_n)`.
- `gamma`: `gamma[c][n]` is component `c`'s per-timestep responsibility vector
    for trial `n`; `nothing` for the single-component case.
- `Rchol::Vector{Cholesky{T,Matrix{T}}}`: one factor per component, refreshed at
    the top of each warp CM-step because `R` moved in CM-step 1.
"""
struct _SplineMStepCtx{T<:Real,YV<:AbstractVector{<:AbstractMatrix{T}}}
    warp::MonotonicWarp{T}
    bufs::Vector{WarpGradBuffers{T}}
    zbuf::Vector{Vector{T}}
    rbuf::Vector{Vector{T}}
    wbuf::Vector{Vector{T}}
    partial::Vector{T}
    y::YV
    mu::Vector{Vector{Matrix{T}}}
    gamma::Union{Nothing,Vector{Vector{Vector{T}}}}
    chunks::Vector{UnitRange{Int}}
    Rchol::Vector{Cholesky{T,Matrix{T}}}
    ridge::T
end

"""
    _SplineMStepCtx(om, y, mu; gamma=nothing, ncomponents=length(mu))

Build the context for a warp whose emission is `om`, over observations `y` and
per-component smoothed means `mu`. `mu` may be given as a plain
`Vector{Matrix}` for the single-component case.
"""
function _SplineMStepCtx(
    om::SplineGaussianObservationModel{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    mu::Vector{Vector{Matrix{T}}};
    gamma::Union{Nothing,Vector{Vector{Vector{T}}}}=nothing,
) where {T<:Real}
    p = size(om.C, 1)
    ntrials = length(y)
    nchunks = max(1, min(Threads.maxthreadid(), ntrials))
    chunksize = cld(ntrials, nchunks)
    chunks = UnitRange{Int}[]
    for i in 1:nchunks
        lo = (i - 1) * chunksize + 1
        hi = min(i * chunksize, ntrials)
        lo <= hi && push!(chunks, lo:hi)
    end
    nc = length(chunks)
    R0 = cholesky(Symmetric(Matrix{T}(om.R)))
    return _SplineMStepCtx{T,typeof(y)}(
        om.warp,
        [WarpGradBuffers(om.warp) for _ in 1:nc],
        [zeros(T, p) for _ in 1:nc],
        [zeros(T, p) for _ in 1:nc],
        [zeros(T, p) for _ in 1:nc],
        zeros(T, nc),
        y,
        mu,
        gamma,
        chunks,
        [copy(R0) for _ in 1:length(mu)],
        om.spline_ridge,
    )
end

function _SplineMStepCtx(
    om::SplineGaussianObservationModel{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    mu::Vector{Matrix{T}},
) where {T<:Real}
    return _SplineMStepCtx(om, y, [mu])
end

"""
    _spline_emission_means!(mu, gom, tfs, uy)

Cache `μ̂_{nt} = C x̂_{nt} + d + D uy_{nt}` from the current smoother output.
These are the only thing the warp objective needs from the E-step: the smoothed
covariance contributes `tr(R⁻¹ C Σ_t C')`, which is constant in the spline
parameters (see the file header).
"""
function _spline_emission_means!(
    mu::Vector{Matrix{T}},
    gom::GaussianObservationModel{T},
    tfs::TrialFilterSmooth{T},
    uy::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real}
    C = gom.C
    d = gom.d
    D_obs = gom.D
    for n in eachindex(mu)
        mu_n = mu[n]
        mul!(mu_n, C, tfs[n].x_smooth)
        if size(D_obs, 2) > 0
            mul!(mu_n, D_obs, uy[n], one(T), one(T))
        end
        mu_n .+= d
    end
    return mu
end

#=
Objective and gradient of the warp CM-step, in Optim's minimization convention:

    f(φ) = ½ Σ_{n,t} Σ_c γ_{c,n,t} ‖L_c⁻¹(g_φ(y) - μ̂_c)‖²  -  Σ log g'  +  ½ λ‖φ‖²

`G === nothing` skips the gradient half. The per-sample seed handed to
`warp_accumulate_grad!` is `ω_j = -(R⁻¹ r)_j = ∂Q_data/∂z_j`, so the accumulator
returns `∂Q_data/∂φ` directly and the sign flip happens once at the end.
=#
function _spline_fg!(
    G::Union{Nothing,AbstractVector{T}}, θ::AbstractVector{T}, ctx::_SplineMStepCtx{T}
) where {T<:Real}
    warp_unpack!(ctx.warp, θ)
    want_grad = G !== nothing

    if want_grad
        for buf in ctx.bufs
            reset!(buf)
        end
    end
    fill!(ctx.partial, zero(T))

    nchunks = length(ctx.chunks)
    ncomp = length(ctx.mu)
    gamma = ctx.gamma
    p = length(ctx.zbuf[1])

    tforeach(1:nchunks) do ci
        zb = ctx.zbuf[ci]
        rb = ctx.rbuf[ci]
        wb = ctx.wbuf[ci]
        buf = ctx.bufs[ci]
        acc = zero(T)
        for n in ctx.chunks[ci]
            yn = ctx.y[n]
            @inbounds for t in axes(yn, 2)
                # One embedding per sample, scored under every component.
                for j in 1:p
                    zj, ℓj = warp_forward(ctx.warp, j, yn[j, t])
                    zb[j] = zj
                    acc -= ℓj
                end
                #=
                The change-of-variables term belongs to the sample, not to a
                component, so exactly one accumulate call per sample carries it
                — tracked here rather than keyed to `c == 1`, since a component
                with zero responsibility is skipped entirely.
                =#
                seeded = false
                for c in 1:ncomp
                    w = gamma === nothing ? one(T) : gamma[c][n][t]
                    iszero(w) && continue
                    mu_c = ctx.mu[c][n]
                    for j in 1:p
                        rb[j] = zb[j] - mu_c[j, t]
                    end
                    copyto!(wb, rb)
                    ldiv!(ctx.Rchol[c], wb)           # wb = R_c⁻¹ r
                    acc += T(0.5) * w * dot(rb, wb)
                    if want_grad
                        for j in 1:p
                            warp_accumulate_grad!(
                                buf, ctx.warp, j, yn[j, t], -w * wb[j], !seeded
                            )
                        end
                        seeded = true
                    end
                end
                if want_grad && !seeded
                    # Every component was skipped; the sample still contributes
                    # its log-Jacobian gradient.
                    for j in 1:p
                        warp_accumulate_grad!(buf, ctx.warp, j, yn[j, t], zero(T), true)
                    end
                end
            end
        end
        ctx.partial[ci] = acc
        return nothing
    end

    f = sum(ctx.partial)
    λ = ctx.ridge
    if !iszero(λ)
        f += T(0.5) * λ * sum(abs2, θ)
    end

    if want_grad
        #=
        Reduce the per-chunk knot accumulators into the first, convert once to
        logit space, then flip sign (the accumulator built ∂Q/∂φ, Optim wants
        ∂f/∂φ = -∂Q/∂φ) and add the ridge derivative.
        =#
        base = ctx.bufs[1]
        for ci in 2:nchunks
            base.gX .+= ctx.bufs[ci].gX
            base.gY .+= ctx.bufs[ci].gY
            base.gD .+= ctx.bufs[ci].gD
        end
        fill!(G, zero(T))
        warp_knot_grad_to_theta!(G, ctx.warp, base)
        @. G = -G + λ * θ
    end

    return f
end

"""
    _spline_warp_mstep!(om, ctx, θ0, θ1; iters, fit_warp) -> Bool

CM-step 2: maximize the normalizing-flow objective over the spline parameters
with everything else held fixed, by L-BFGS with exact gradients.

Returns whether a new warp was accepted. A proposal is taken only when it
strictly improves the objective: L-BFGS normally returns one, but a stalled line
search otherwise hands EM a worse `Q` and breaks the monotonicity the ECM
argument provides. On rejection the warp is restored to `θ0`, so the next E-step
still sees a consistent model.
"""
function _spline_warp_mstep!(
    om::SplineGaussianObservationModel{T},
    ctx::_SplineMStepCtx{T},
    θ0::Vector{T},
    θ1::Vector{T};
    iters::Int=25,
    fit_warp::Bool=true,
) where {T<:Real}
    fit_warp || return false
    iters > 0 || return false

    #=
    `R` moved in CM-step 1; the warp objective whitens against the current one.
    A mixture context has one factor per regime and is refreshed by its own
    caller before this runs (see `fit_slds_spline.jl`).
    =#
    if length(ctx.Rchol) == 1
        ctx.Rchol[1] = cholesky(Symmetric(Matrix{T}(om.R)))
    end

    warp_pack!(θ0, ctx.warp)
    f0 = _spline_fg!(nothing, θ0, ctx)
    isfinite(f0) || (warp_unpack!(ctx.warp, θ0); return false)

    f_obj(θ) = _spline_fg!(nothing, θ, ctx)
    function g_obj!(G, θ)
        _spline_fg!(G, θ, ctx)
        return G
    end

    opts = Optim.Options(; x_abstol=1e-10, g_abstol=1e-9, f_reltol=1e-12, iterations=iters)
    result = optimize(f_obj, g_obj!, copy(θ0), LBFGS(; linesearch=HagerZhang()), opts)
    copyto!(θ1, Optim.minimizer(result))
    f1 = _spline_fg!(nothing, θ1, ctx)

    if isfinite(f1) && f1 < f0
        warp_unpack!(ctx.warp, θ1)
        return true
    end
    warp_unpack!(ctx.warp, θ0)
    return false
end

#=
Restoring a snapshot must not rebind the warp: the M-step context and the
shadow emission hold it by reference, and the generic `setfield!` path would
leave them pointing at the discarded object. Copy into it instead.
=#
function _copy_model!(dest::M, src::M) where {M<:SplineGaussianObservationModel}
    for i in 1:fieldcount(M)
        if fieldname(M, i) === :warp
            copy_warp!(getfield(dest, i), getfield(src, i))
        else
            setfield!(dest, i, getfield(src, i))
        end
    end
    return dest
end
