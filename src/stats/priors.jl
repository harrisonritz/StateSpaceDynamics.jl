#=============================================================================
Conjugate priors used across LDS-family models.

Currently:
  * `IWPrior`  — inverse-Wishart prior on a covariance matrix (Q, P0, R, ...).
  * `MNPrior`  — matrix-normal prior on a regression coefficient matrix
                 ([A B], [C D], ...). Pair with an `IWPrior` on the same
                 regression to obtain the full MNIW conjugate prior on
                 (W, Σ).

Each prior ships with a small `*_map` helper that returns the closed-form
MAP update so M-steps stay one-liner-clean.
=============================================================================#

"""
    IWPrior{T<:Real, M<:AbstractMatrix}

Inverse-Wishart prior for a covariance matrix Σ ~ IW(Ψ, ν), with density
p(Σ) ∝ |Σ|^{-(ν + d + 1)/2} exp(-½ tr(Ψ Σ^{-1})) for d = size(Σ,1).

# Fields
- `Ψ::M`: Scale matrix (d×d, SPD).
- `ν::T`: Degrees of freedom (must satisfy `ν > d + 1` for a proper mode).

# Notes
- The MAP update for a posterior IW(Ψ + S, ν + n) is `(Ψ + S) / (ν + n + d + 1)`.
"""
Base.@kwdef struct IWPrior{T<:Real,M<:AbstractMatrix}
    Ψ::M
    ν::T
end

# helpers for new priors on cov matrices
@inline function iw_map(
    Ψ::AbstractMatrix{T}, ν::T, S::AbstractMatrix{T}, n::T, d::Int
) where {T}
    return (Ψ .+ S) ./ (ν + n + d + one(T))
end

# TODO: this should use PD Mats
@inline function iw_logprior_term(Σ::AbstractMatrix{T}, prior::IWPrior{T}) where {T}
    D = size(Σ, 1)
    Ψ, ν = prior.Ψ, prior.ν
    # log|Σ| via Cholesky
    F = cholesky(Symmetric(Σ))
    logdetΣ = 2sum(log, diag(F.U))
    # tr(Ψ Σ^{-1}) via triangular solves
    X = F \ Ψ                 # solves Σ * X = Ψ
    return -T(0.5) * ((ν + D + one(T)) * logdetΣ + tr(X))
end

"""
    MNPrior{T<:Real, M<:AbstractMatrix}

Matrix-normal prior on a regression coefficient matrix W (size `k × p`)
appearing in a linear regression `Y = W X + ε` with row-noise covariance Σ:

```math
W | Σ ~ MN(M₀, Σ ⊗ Λ⁻¹)
```

equivalently `vec(W) ~ N(vec(M₀), Λ⁻¹ ⊗ Σ)`. Σ is the regression's row
covariance (paired with `IWPrior` when both halves of an MNIW prior are
desired); Λ is the column precision and is the only piece that enters the MAP
update for W.

# Fields
- `M₀::M`: prior mean (`k × p`, same shape as W). Use a zero matrix for plain
    ridge; an identity-like matrix for shrinkage toward a random walk on `A`.
- `Λ::M`: column precision (`p × p`, SPD).

# Notes
- The MAP update for W given the regression sufficient statistics `XX = X Xᵀ`
    and `XY = X Yᵀ` is
    ```math
    W = (XYᵀ + M₀ Λ) (XX + Λ)⁻¹
    ```
    Reduces to OLS when `Λ = 0`, and to ordinary ridge regression when
    `M₀ = 0`.
- Mathematically half of an MNIW prior; combine with an `IWPrior` on the same
    regression to recover the full conjugate prior on `(W, Σ)`.
"""
Base.@kwdef struct MNPrior{T<:Real,M<:AbstractMatrix{T}}
    M₀::M
    Λ::M
end

"""
    mn_map(XX, XY, prior) -> Matrix

MAP estimate for the regression coefficient `W` under an `MNPrior`:
returns `W = (XYᵀ + M₀ Λ)(XX + Λ)⁻¹`. Falls back to OLS (`W = XYᵀ XX⁻¹`)
when `prior === nothing`.

`XX` may be a plain `AbstractMatrix` or a `PDMat`; the latter is preferred so
the cached Cholesky of `XX + Λ` is reused (PDMats handles the addition).
"""
@inline function mn_map(
    XX::AbstractMatrix{T}, XY::AbstractMatrix{T}, prior::MNPrior{T}
) where {T}
    return transpose((XX + prior.Λ) \ (XY + prior.Λ * prior.M₀'))
end

@inline function mn_map(XX::AbstractMatrix{T}, XY::AbstractMatrix{T}, ::Nothing) where {T}
    return transpose(XX \ XY)
end

"""
    _restrict_mn_prior(prior, cols)

Restrict a matrix-normal regression prior to a set of free coefficient columns.
Cross-precision terms involving omitted (constrained) columns are deliberately
dropped: the resulting prior is the one induced on the model's actual free
parameter space.
"""
@inline _restrict_mn_prior(::Nothing, ::AbstractVector{Int}) = nothing

@inline function _restrict_mn_prior(prior::MNPrior, cols::AbstractVector{Int})
    return MNPrior(Matrix(prior.M₀[:, cols]), Matrix(prior.Λ[cols, cols]))
end

"""
    mn_logprior_term(W, Σ, prior) -> Real

W-dependent part of the matrix-normal log prior `log p(W | Σ)` evaluated at the
current `(W, Σ)`. Drops Σ- and Λ-only constants that the IW prior + the `mn_map`
M-step already cover, leaving the quadratic term

    -½ tr(Σ^{-1} (W - M₀) Λ (W - M₀)')

This is the contribution the ELBO needs to display the true MAP objective when
an `MNPrior` is set on a regression coefficient (e.g. `[A b B]` or `[C d D]`).
Without it, EM is still maximizing the right thing internally, but the displayed
ELBO can decrease under strong shrinkage — see git log for the fix context.
"""
@inline function mn_logprior_term(
    W::AbstractMatrix{T}, Σ::AbstractMatrix{T}, prior::MNPrior{T}
) where {T<:Real}
    Wm = W .- prior.M₀
    # tr(Σ^{-1} Wm Λ Wm') = tr(Λ Wm' Σ^{-1} Wm); pick the cheaper order based
    # on shape, but both are k × k after the trace, so just go left-to-right.
    M = Wm * prior.Λ * Wm'
    F = cholesky(Symmetric(Σ))
    return -T(0.5) * tr(F \ M)
end

@inline mn_logprior_term(W::AbstractMatrix, ::AbstractMatrix, ::Nothing) = zero(eltype(W))

"""
    x0_mean_prior(μ₀; κ₀=1) -> MNPrior

Convenience constructor for the mean half of a Normal–Inverse–Wishart prior on
the initial latent state.

- pair this `MNPrior` (on `x0`, via `state_model.x0_prior`) with an `IWPrior` on
  `P0` (via `state_model.P0_prior`) to obtain the full NIW conjugate prior:

  ```math
  x0 | P0 ~ N(μ₀, P0 / κ₀),   P0 ~ IW(Ψ, ν)
  ```
"""
function x0_mean_prior(μ₀::AbstractVector{T}; κ₀::Real=one(T)) where {T<:Real}
    κ₀ > zero(T) || throw(ArgumentError("x0_mean_prior: κ₀ must be positive, got $κ₀"))
    D = length(μ₀)
    return MNPrior{T,Matrix{T}}(; M₀=reshape(collect(μ₀), D, 1), Λ=fill(T(κ₀), 1, 1))
end

"""
    transition_prior(K; concentration=1.0, sticky=0.0) -> Matrix

Dirichlet concentrations for an `SLDS`'s transition matrix — pass the result as
`A_prior` — with every row `Dir(concentration, …, concentration + sticky, …)`,
the extra `sticky` on the diagonal:

    α = fill(concentration, K, K) + sticky · I

- `concentration > 1` adds `concentration − 1` pseudo-transitions to every
  entry, so every transition stays reachable. Without it a regime that goes
  unused for a few iterations can have its row driven to something like
  `[1, 1e-63]`, from which EM cannot bring it back.
- `sticky > 0` adds pseudo-self-transitions, which lengthens the expected dwell
  in every state (the sticky-HMM prior, as a MAP rather than a hierarchical
  one).

`concentration = 1, sticky = 0` is the flat prior and changes nothing. Every
entry must be `≥ 1` for the MAP update to be the closed form it is, so
`concentration ≥ 1` and `sticky ≥ 0`.
"""
function transition_prior(K::Integer; concentration::Real=1.0, sticky::Real=0.0)
    K >= 1 || throw(ArgumentError("transition_prior: K must be ≥ 1, got $K"))
    concentration >= 1 || throw(
        ArgumentError(
            "transition_prior: concentration must be ≥ 1 (the MAP update needs " *
            "non-negative pseudo-counts), got $concentration",
        ),
    )
    sticky >= 0 || throw(ArgumentError("transition_prior: sticky must be ≥ 0, got $sticky"))
    T = float(promote_type(typeof(concentration), typeof(sticky)))
    α = fill(T(concentration), K, K)
    for k in 1:K
        α[k, k] += T(sticky)
    end
    return α
end

"""
    median_dwell_stay(median_dwell) -> Real

The self-transition probability `p` of a stage whose dwell time has the given
median, in bins: a geometric dwell leaves the stage within `d` bins with
probability `1 − p^d`, so a 50 % chance of having switched by `d` is

    p = 0.5^(1/d).

This is the natural way to put a task timescale on a stage of a
[`banded_transition`](@ref) chain — "the go cue comes at bin 40 in half the
trials" becomes `median_dwell_stay(40)` for the stage that ends at the go cue.
`median_dwell` need not be an integer, and must exceed zero.
"""
function median_dwell_stay(median_dwell::Real)
    median_dwell > 0 || throw(
        ArgumentError(
            "median_dwell_stay: median_dwell must be positive, got $median_dwell"
        ),
    )
    return 0.5^(1 / float(median_dwell))
end

#=
A stage's stay probabilities, one per non-absorbing stage, from either a scalar
(every stage the same) or a vector of length `K − 1`.
=#
function _banded_stays(K::Integer, stay)
    K >= 1 || throw(ArgumentError("a banded chain needs K ≥ 1 stages, got $K"))
    p = stay isa Real ? fill(float(stay), K - 1) : float.(collect(stay))
    length(p) == K - 1 || throw(
        ArgumentError(
            "a banded chain of $K stages takes $(K - 1) stay probabilities (the last " *
            "stage is absorbing); got $(length(p))",
        ),
    )
    all(x -> 0 < x < 1, p) || throw(
        ArgumentError(
            "stay probabilities must lie strictly between 0 and 1 (each stage has to " *
            "be both enterable and leavable); got $p",
        ),
    )
    return p
end

"""
    banded_transition(K; stay) -> (A, πₖ)

A **banded** (left-to-right) discrete chain of `K` stages: stage `k` either stays,
with probability `stay[k]`, or moves on to stage `k + 1`; it never skips ahead or
returns, and the last stage is absorbing. Every trial starts in stage 1
(`πₖ = e₁`).

`stay` is a scalar (every stage the same) or one probability per non-absorbing
stage (length `K − 1`); [`median_dwell_stay`](@ref) turns a median dwell time in
bins into one. Pass the result as an [`SLDS`](@ref)'s `A` / `πₖ`, and pair it with
[`banded_transition_prior`](@ref).

The band is kept by fitting rather than imposed: an entry that starts at zero has
no expected transitions and no pseudo-counts, so the Baum–Welch / Dirichlet-MAP
update and the terminal-conditioned chain step both leave it at zero, and the
initial distribution stays on stage 1 for the same reason. A banded chain is a
changepoint process with geometric stage durations: each trial runs through the
stages in order, and inference is over where the boundaries fall.

# Example
```jldoctest
julia> A, π = banded_transition(3; stay=[0.75, 0.5]);

julia> A
3×3 Matrix{Float64}:
 0.75  0.25  0.0
 0.0   0.5   0.5
 0.0   0.0   1.0

julia> π
3-element Vector{Float64}:
 1.0
 0.0
 0.0
```
"""
function banded_transition(K::Integer; stay)
    p = _banded_stays(K, stay)
    T = eltype(p)
    A = zeros(T, K, K)
    for k in 1:(K - 1)
        A[k, k] = p[k]
        A[k, k + 1] = one(T) - p[k]
    end
    A[K, K] = one(T)
    πₖ = zeros(T, K)
    πₖ[1] = one(T)
    return A, πₖ
end

"""
    banded_transition_prior(K; stay, strength=10.0) -> Matrix

Dirichlet concentrations for a [`banded_transition`](@ref) chain — pass the result
as `A_prior`. Only the band carries a prior: row `k` puts `strength · stay[k]`
pseudo-counts on staying and `strength · (1 − stay[k])` on moving on,

    α[k, k] = 1 + strength · stay[k],    α[k, k+1] = 1 + strength · (1 − stay[k]),

and every other entry is `1` (no pseudo-counts, so the structural zeros stay
zero). A row of a banded chain has one free parameter, so this is a
`Beta(1 + strength·p, 1 + strength·(1 − p))` prior on each stage's stay
probability, whose **mode is exactly `stay[k]`**: with no data the MAP chain is the
one `stay` describes, and `strength` is how many transitions' worth of evidence
it takes to move it halfway. The last (absorbing) row is left flat.

With `stay = median_dwell_stay.(d)` the prior says "half the trials have left
stage `k` within `d[k]` bins of entering it".

`strength = 0` is the flat prior. It must be `≥ 0`.
"""
function banded_transition_prior(K::Integer; stay, strength::Real=10.0)
    p = _banded_stays(K, stay)
    strength >= 0 ||
        throw(ArgumentError("banded_transition_prior: strength must be ≥ 0, got $strength"))
    T = float(promote_type(eltype(p), typeof(strength)))
    α = ones(T, K, K)
    for k in 1:(K - 1)
        α[k, k] += T(strength) * p[k]
        α[k, k + 1] += T(strength) * (one(T) - p[k])
    end
    return α
end

"""
    dirichlet_logprior_term(p, α) -> Real

`p`-dependent part of `log Dir(p | α)`, summed over the rows of a matrix `p`
(each row its own Dirichlet, as for a transition matrix) or over a single
vector: `Σ (α − 1) log p`.

The `log Γ` normaliser depends on `α` alone and is dropped, as
`iw_logprior_term` / `mn_logprior_term` drop theirs: the ELBO reports the MAP
objective up to a constant in the parameters, and a flat `α ≡ 1` then adds
exactly zero. `p` is floored at `1e-12` like the chain's own `E_q[log p(z)]`
term, so a transition the chain has driven to zero scores as the ELBO already
scores it rather than as `-Inf`; the terminal-conditioned chain step
(`_slqr_chain_mstep!`) scores the prior with that same floor.
"""
function dirichlet_logprior_term(p::AbstractArray{T}, α::AbstractArray) where {T<:Real}
    total = zero(T)
    floor = T(1e-12)
    for i in eachindex(p, α)
        a = T(α[i]) - one(T)
        iszero(a) || (total += a * log(p[i] + floor))
    end
    return total
end
