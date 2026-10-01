#=============================================================================
Infinite-horizon ("hold") LQR — numerical primitives.

A `:hold` [`LQRStateModel`](@ref) is a stationary regulator: the costate sits
(softly) on the stable manifold `λ = P x + g` of the infinite-horizon problem,
where `P` is the stabilizing solution of the discrete algebraic Riccati
equation

    P = Q_h + Aᵀ P (I + S P)⁻¹ A

and `g` the stationary feedforward of a quasi-constant forcing. Everything the
hold model needs from `(A, S, Q_h, h, B_u, G_r)` is a function of that one
solution, so it is computed in one place — here — and read by both the cache
refresh (`_refresh_hold_head!` in `lqr_types.jl`) and the structural M-step
(`lqr_mstep.jl`), which differentiates through it.

    DARE:        _dare_doubling!       structure-preserving doubling
    Adjoint:     _stein_adjoint!       Y − A_cl Y A_clᵀ = C  (Smith doubling)
    Steady state _hold_steady_state!   P, W, A_cl, V, K, g-map, Lh, log det(I+SP)

These live ahead of `lqr_types.jl` because they depend on nothing but matrices,
and the model's own refresh needs them.
=============================================================================#

"""
    _HoldUnit{T}

Steady-state quantities of one hold model (or one hold unit of the structural
M-step), with the scratch the doubling solvers and the gradient chain reuse.

Writing `F = [h_x  B_u,x]` (plant drift and disturbance) and
`E = [h_λ  B_u,λ − Q_h G_r]` (costate forcing, tracking term included), both
`n × (1 + m)` against `ũ = [1; u]`:

- `P`: stabilizing DARE solution; `W = (I + S P)⁻¹`; `Acl = W A`, the closed
  loop, `ρ(Acl) < 1`; `V = (I − Aclᵀ)⁻¹`.
- `K = E + Aclᵀ P F` and `Gm = V K`, the stationary feedforward map
  `g(ũ) = Gm ũ`.
- `Lh = [I  S; −P  I]`, the hold model's "plant-row / manifold-row" design on
  `z_{t+1}`, and `logdetM = log det(I + S P) = log det Lh`, the Jacobian between
  those coordinates and the forward ones.

The remaining fields are scratch; nothing outside the routines below should
read them.
"""
mutable struct _HoldUnit{T<:Real}
    const P::Matrix{T}
    const W::Matrix{T}
    const Acl::Matrix{T}
    const V::Matrix{T}
    const F::Matrix{T}
    const E::Matrix{T}
    const K::Matrix{T}
    const Gm::Matrix{T}
    const Lh::Matrix{T}
    logdetM::T
    # doubling scratch
    const Ak::Matrix{T}
    const Gk::Matrix{T}
    const Hk::Matrix{T}
    const nn1::Matrix{T}
    const nn2::Matrix{T}
    const nn3::Matrix{T}
    # gradient scratch
    const DLh::Matrix{T}       # d × d
    const DTh::Matrix{T}       # d × reg
    const dd::Matrix{T}        # d × d
    const dr::Matrix{T}        # d × reg
    const Pbar::Matrix{T}
    const Aclbar::Matrix{T}
    const Ybar::Matrix{T}
    const Kbar::Matrix{T}      # n × (1 + m)
    const Fbar::Matrix{T}      # n × (1 + m)
    const nr::Matrix{T}        # n × (1 + m)
end

function _HoldUnit(::Type{T}, n::Int, m::Int) where {T<:Real}
    d = 2n
    r = 1 + m
    nn() = zeros(T, n, n)
    return _HoldUnit{T}(
        nn(),
        nn(),
        nn(),
        nn(),
        zeros(T, n, r),
        zeros(T, n, r),
        zeros(T, n, r),
        zeros(T, n, r),
        zeros(T, d, d),
        zero(T),
        nn(),
        nn(),
        nn(),
        nn(),
        nn(),
        nn(),
        zeros(T, d, d),
        zeros(T, d, d + r),
        zeros(T, d, d),
        zeros(T, d, d + r),
        nn(),
        nn(),
        nn(),
        zeros(T, n, r),
        zeros(T, n, r),
        zeros(T, n, r),
    )
end

"""
    _dare_doubling!(P, A, S, Q, H; max_iter=100) -> Bool

The stabilizing solution of `P = Q + Aᵀ P (I + S P)⁻¹ A` by the
structure-preserving doubling algorithm, written into `P`:

    A₀ = A,  G₀ = S,  H₀ = Q,   W_k = (I + G_k H_k)⁻¹,
    A_{k+1} = A_k W_k A_k,
    G_{k+1} = G_k + A_k W_k G_k A_kᵀ,
    H_{k+1} = H_k + A_kᵀ H_k W_k A_k,        H_k → P.

Convergence is quadratic whenever a stabilizing solution exists (`(A, S)`
stabilizable and `(Q, A)` detectable), so the iteration cap is never reached in
practice; where none exists `H_k` diverges or stalls. The converged `P` is then
checked against the equation itself.

Returns `false` rather than throwing on failure — the structural M-step reads
that as an infeasible point (objective `Inf`), and the model's own refresh turns
it into an error. `H` supplies the scratch.
"""
function _dare_doubling!(
    P::AbstractMatrix{T},
    A::AbstractMatrix{T},
    S::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    H::_HoldUnit{T};
    max_iter::Int=100,
) where {T<:Real}
    n = size(A, 1)
    Ak, Gk, Hk = H.Ak, H.Gk, H.Hk
    X, Y, Z = H.nn1, H.nn2, H.nn3
    copyto!(Ak, A)
    copyto!(Gk, S)
    copyto!(Hk, Q)
    converged = false
    tol = T(64) * eps(T)
    for _ in 1:max_iter
        # X = I + G_k H_k, factored once and used for W_k A_k and W_k G_k.
        copyto!(X, I)
        mul!(X, Gk, Hk, one(T), one(T))
        Fk = lu!(X; check=false)
        issuccess(Fk) || return false
        copyto!(Y, Ak)
        ldiv!(Fk, Y)                       # Y = W_k A_k
        copyto!(Z, Gk)
        ldiv!(Fk, Z)                       # Z = W_k G_k
        # H_{k+1} = H_k + A_kᵀ H_k (W_k A_k); `X` is free again after the solves.
        mul!(X, Hk, Y)
        delta = zero(T)
        scale = zero(T)
        mul!(P, transpose(Ak), X)
        @inbounds for j in 1:n, i in 1:n
            delta = max(delta, abs(P[i, j]))
        end
        Hk .+= P
        Symmetrize!(Hk)
        # G_{k+1} = G_k + A_k (W_k G_k) A_kᵀ.
        mul!(X, Ak, Z)
        mul!(Gk, X, transpose(Ak), one(T), one(T))
        Symmetrize!(Gk)
        # A_{k+1} = A_k (W_k A_k).
        mul!(X, Ak, Y)
        copyto!(Ak, X)
        @inbounds for j in 1:n, i in 1:n
            scale = max(scale, abs(Hk[i, j]))
        end
        isfinite(scale) || return false
        if delta <= tol * max(one(T), scale)
            converged = true
            break
        end
    end
    converged || return false
    copyto!(P, Hk)

    #= Check the equation rather than trusting the iteration: a stall that
    happens to satisfy the increment test is not a solution. =#
    copyto!(X, I)
    mul!(X, S, P, one(T), one(T))
    Fm = lu!(X; check=false)
    issuccess(Fm) || return false
    copyto!(Y, A)
    ldiv!(Fm, Y)                           # W A
    mul!(Z, P, Y)
    mul!(X, transpose(A), Z)               # Aᵀ P W A
    resid = zero(T)
    pmax = zero(T)
    @inbounds for j in 1:n, i in 1:n
        resid = max(resid, abs(P[i, j] - Q[i, j] - X[i, j]))
        pmax = max(pmax, abs(P[i, j]))
    end
    return resid <= sqrt(eps(T)) * max(one(T), pmax)
end

"""
    _stein_adjoint!(Y, Acl, C, H; max_iter=100) -> Y

Solve the Stein equation `Y − Acl Y Aclᵀ = C` by Smith doubling,
`Y ← Y + A_k Y A_kᵀ`, `A_{k+1} = A_k²`, which after `k` steps has summed
`Σ_{j < 2^k} Acl^j C Acl^{jᵀ}`. Requires `ρ(Acl) < 1`, which the hold model's
closed loop has by construction.

This is the adjoint of the DARE: differentiating `P = Q + Aᵀ P W A` gives
`dP − Aclᵀ dP Acl = dQ + dAᵀ P Acl + Aclᵀ P dA − Aclᵀ P dS P Acl`, so pulling a
cotangent `P̄` back through it is one solve with the transposed closed loop.
"""
function _stein_adjoint!(
    Y::AbstractMatrix{T},
    Acl::AbstractMatrix{T},
    C::AbstractMatrix{T},
    H::_HoldUnit{T};
    max_iter::Int=100,
) where {T<:Real}
    n = size(Acl, 1)
    Ak, X = H.Ak, H.nn1
    copyto!(Y, C)
    copyto!(Ak, Acl)
    for _ in 1:max_iter
        mul!(X, Ak, Y)
        mul!(Y, X, transpose(Ak), one(T), one(T))
        mul!(X, Ak, Ak)
        copyto!(Ak, X)
        amax = zero(T)
        @inbounds for j in 1:n, i in 1:n
            amax = max(amax, abs(Ak[i, j]))
        end
        amax <= eps(T) && break
    end
    return Y
end

"""
    _hold_steady_state!(H, A, S, Q, h, Bu, Gref) -> Bool

Fill `H` with the hold model's steady state at these parameters: the DARE
solution `P`, `W = (I + S P)⁻¹`, the closed loop `Acl = W A`,
`V = (I − Aclᵀ)⁻¹`, the forcing blocks `F` and `E`, the feedforward map
`Gm = V (E + Aclᵀ P F)`, the design `Lh = [I S; −P I]` and
`log det(I + S P)`.

The feedforward is the stationary costate offset under a constant forcing:
substituting `λ = P x + g` into the plant row
`x' = A x − S λ' + f` and the costate row `λ = Q x + Aᵀ λ' + e` gives the DARE
for `P` and `g = Aclᵀ g + Aclᵀ P f + e` (using `Aᵀ P W = Aclᵀ P`), i.e.
`g = V (e + Aclᵀ P f)`. With `f = F ũ` and `e = E ũ` that is `g = Gm ũ`.

Returns `false` when there is no stabilizing solution, the closed loop is not
strictly stable, or `I + S P` has a non-positive determinant — every one of
which makes the hold model undefined at these parameters.
"""
function _hold_steady_state!(
    H::_HoldUnit{T},
    A::AbstractMatrix{T},
    S::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    h::AbstractVector{T},
    Bu::AbstractMatrix{T},
    Gref::AbstractMatrix{T},
) where {T<:Real}
    n = size(A, 1)
    d = 2n
    m = size(Bu, 2)
    xr, lr = 1:n, (n + 1):d
    _dare_doubling!(H.P, A, S, Q, H) || return false
    P = H.P

    # W = (I + S P)⁻¹ and its log-determinant (positive: S P is similar to a PSD matrix).
    X = H.nn1
    copyto!(X, I)
    mul!(X, S, P, one(T), one(T))
    Fm = lu!(X; check=false)
    issuccess(Fm) || return false
    ld, sgn = logabsdet(Fm)
    (sgn > zero(T) && isfinite(ld)) || return false
    H.logdetM = T(ld)
    copyto!(H.W, I)
    ldiv!(Fm, H.W)
    mul!(H.Acl, H.W, A)
    ρ = maximum(abs, eigvals(H.Acl))
    (isfinite(ρ) && ρ < one(T)) || return false

    # V = (I − Aclᵀ)⁻¹.
    X .= .-transpose(H.Acl)
    @inbounds for i in 1:n
        X[i, i] += one(T)
    end
    Fv = lu!(X; check=false)
    issuccess(Fv) || return false
    copyto!(H.V, I)
    ldiv!(Fv, H.V)

    # F = [h_x  B_u,x],  E = [h_λ  B_u,λ − Q G_r].
    @views begin
        H.F[:, 1] .= h[xr]
        H.E[:, 1] .= h[lr]
        if m > 0
            H.F[:, 2:end] .= Bu[xr, :]
            H.E[:, 2:end] .= Bu[lr, :]
            mul!(H.E[:, 2:end], Q, Gref, -one(T), one(T))
        end
    end
    # K = E + Aclᵀ P F,  Gm = V K.
    mul!(H.nn2, transpose(H.Acl), P)
    copyto!(H.K, H.E)
    mul!(H.K, H.nn2, H.F, one(T), one(T))
    mul!(H.Gm, H.V, H.K)

    # Lh = [I  S; −P  I].
    Lh = H.Lh
    fill!(Lh, zero(T))
    @views begin
        Lh[xr, lr] .= S
        Lh[lr, xr] .= .-P
    end
    @inbounds for i in 1:d
        Lh[i, i] = one(T)
    end
    return true
end
