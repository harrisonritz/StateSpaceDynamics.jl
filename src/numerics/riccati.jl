#=============================================================================
Finite-horizon LQR sweeps in the `(A, B, R)` form, on preallocated buffers,
with the reverse-mode pass an M-step gradient needs.

The problem is

    minimize   Σ_{t=1}^{T} (½ x_tᵀ Q_{k(t)} x_t + q_tᵀ x_t)  +  Σ_{t=1}^{T-1} ½ u_tᵀ R u_t
    subject to x_{t+1} = A x_t + B u_t,

with a cost schedule `k(t)` choosing which `Q` is active at each step. Its value
function is `½ xᵀ P_t x + b_tᵀ x + const` and its optimal policy
`u_t = −K_t x_t + k_t`, where, sweeping backward from `P_T = Q_{k(T)}`, `b_T = q_T`,

    G_t = R + Bᵀ P_{t+1} B,
    K_t = G_t⁻¹ Bᵀ P_{t+1} A,           Φ_t = A − B K_t,
    P_t = Q_{k(t)} + K_tᵀ R K_t + Φ_tᵀ P_{t+1} Φ_t,
    k_t = −G_t⁻¹ Bᵀ b_{t+1},            b_t = q_t + Φ_tᵀ b_{t+1}.

The gain half does not depend on `q`, so it runs once per parameter value and
the affine half once per linear cost — once per reference, for a tracking cost
`½(x − r_t)ᵀ Q (x − r_t)`, which is `q_t = −Q_{k(t)} r_t` (see
[`tracking_cost!`](@ref)).

Two things are easy to get wrong and are pinned by the tests:

- The feedforward recursion is `b_t = q_t + Φ_tᵀ b_{t+1}` and nothing more. The
  `Φ_tᵀ P_{t+1} B k_t` and `K_tᵀ R k_t` terms a direct expansion produces cancel
  exactly, because `Φ_tᵀ P_{t+1} B = K_tᵀ R`. A version carrying one of them
  costs measurably more than the QP optimum.
- `λ_t = P_t x_t + b_t` is the costate of `LQRStateModel` when `S = B R⁻¹ Bᵀ`,
  and `Φ_t = (I + S P_{t+1})⁻¹ A` its closed-loop map: this is the same sweep as
  [`lqr_riccati_sequence`](@ref), written where `B` and `R` are separately
  available rather than only through `S`.
=============================================================================#

"""
    RiccatiSweep{T}(n, m, horizon)

Buffers for one finite-horizon sweep of an `n`-state, `m`-input problem, and the
results the sweeps leave in them:

- after [`riccati_gain!`](@ref): `P[t]` for `t = 1:horizon`, and `K[t]`, `Φ[t]`
  and the factor of `G_t` for `t = 1:horizon-1`;
- after [`affine_sweep!`](@ref): `b[:, t]` for `t = 1:horizon` and `k[:, t]` for
  `t = 1:horizon-1`.

`G[t]` holds the upper Cholesky factor of `R + Bᵀ P_{t+1} B`, overwritten in
place; [`riccati_adjoint!`](@ref) reads it, so the adjoint must follow the sweep
it differentiates without another gain sweep in between. The remaining fields
are scratch.

Element type is generic so that the forward sweep runs on dual numbers too.
"""
struct RiccatiSweep{T<:Real}
    P::Vector{Matrix{T}}
    K::Vector{Matrix{T}}
    Φ::Vector{Matrix{T}}
    G::Vector{Matrix{T}}
    b::Matrix{T}
    k::Matrix{T}
    # scratch, shared by the sweeps and the adjoint
    mn::Matrix{T}
    mn2::Matrix{T}
    nn::Matrix{T}
    nn2::Matrix{T}
    nn3::Matrix{T}
    nm::Matrix{T}
    nm2::Matrix{T}
    mm::Matrix{T}
    mm2::Matrix{T}
    Pbar::Matrix{T}
    Pbar_next::Matrix{T}
    Φbar::Matrix{T}
    bbar::Vector{T}
    bbar_next::Vector{T}
    z::Vector{T}
end

function RiccatiSweep{T}(n::Integer, m::Integer, horizon::Integer) where {T<:Real}
    n >= 1 || throw(ArgumentError("RiccatiSweep needs n ≥ 1, got $n"))
    m >= 1 || throw(ArgumentError("RiccatiSweep needs m ≥ 1, got $m"))
    horizon >= 2 || throw(ArgumentError("RiccatiSweep needs horizon ≥ 2, got $horizon"))
    mats(r, c, len) = [zeros(T, r, c) for _ in 1:len]
    return RiccatiSweep{T}(
        mats(n, n, horizon),
        mats(m, n, horizon - 1),
        mats(n, n, horizon - 1),
        mats(m, m, horizon - 1),
        zeros(T, n, horizon),
        zeros(T, m, horizon - 1),
        zeros(T, m, n),
        zeros(T, m, n),
        zeros(T, n, n),
        zeros(T, n, n),
        zeros(T, n, n),
        zeros(T, n, m),
        zeros(T, n, m),
        zeros(T, m, m),
        zeros(T, m, m),
        zeros(T, n, n),
        zeros(T, n, n),
        zeros(T, n, n),
        zeros(T, n),
        zeros(T, n),
        zeros(T, m),
    )
end

_sweep_horizon(ws::RiccatiSweep) = length(ws.P)
_sweep_dims(ws::RiccatiSweep) = (size(ws.P[1], 1), size(ws.K[1], 1))

"""The Cholesky factorization of `G_t` the gain sweep left in `ws.G[t]`."""
_gain_factor(ws::RiccatiSweep, t::Int) = Cholesky(ws.G[t], 'U', 0)

function _check_sweep(ws::RiccatiSweep, A, B, R, Q, schedule)
    n, m = _sweep_dims(ws)
    horizon = _sweep_horizon(ws)
    size(A) == (n, n) || throw(DimensionMismatch("A must be $n×$n, got $(size(A))"))
    size(B) == (n, m) || throw(DimensionMismatch("B must be $n×$m, got $(size(B))"))
    size(R) == (m, m) || throw(DimensionMismatch("R must be $m×$m, got $(size(R))"))
    for (j, Qj) in enumerate(Q)
        size(Qj) == (n, n) ||
            throw(DimensionMismatch("Q[$j] must be $n×$n, got $(size(Qj))"))
    end
    length(schedule) == horizon || throw(
        DimensionMismatch(
            "schedule must have one entry per step ($horizon), got $(length(schedule))"
        ),
    )
    for k in schedule
        1 <= k <= length(Q) ||
            throw(ArgumentError("schedule entry $k names no cost; there are $(length(Q))"))
    end
    return nothing
end

"""`C += α a bᵀ` for vectors `a`, `b`, without the allocation `a * b'` makes."""
function _outer!(C::AbstractMatrix, α, a::AbstractVector, b::AbstractVector)
    @inbounds for j in eachindex(b)
        βj = α * b[j]
        for i in eachindex(a)
            C[i, j] += a[i] * βj
        end
    end
    return C
end

"""`C += α (X + Xᵀ)/2` for a square `X`."""
function _add_sym!(C::AbstractMatrix, α, X::AbstractMatrix)
    h = α / 2
    @inbounds for j in axes(X, 2), i in axes(X, 1)
        C[i, j] += h * (X[i, j] + X[j, i])
    end
    return C
end

"""
    riccati_gain!(ws, A, B, R, Q, schedule) -> ws

The gain half of the sweep: `P_t`, `K_t`, `Φ_t` and the factor of `G_t`, written
into `ws`. `Q` holds one symmetric cost per regime and `schedule[t]` names the
one active at step `t`, including the terminal step, whose cost is `P_T`.

`R + Bᵀ P_{t+1} B` is factored in place; a `PosDefException` from it is the
parameter point being out of reach, and propagates for the caller's
rejectable-step handling to act on. `P_t` is updated in the Joseph form, which
keeps it symmetric positive semidefinite where the textbook
`Q + Aᵀ P A − Aᵀ P B G⁻¹ Bᵀ P A` loses that to cancellation.
"""
function riccati_gain!(
    ws::RiccatiSweep{T},
    A::AbstractMatrix,
    B::AbstractMatrix,
    R::AbstractMatrix,
    Q::AbstractVector{<:AbstractMatrix},
    schedule::AbstractVector{<:Integer},
) where {T<:Real}
    _check_sweep(ws, A, B, R, Q, schedule)
    horizon = _sweep_horizon(ws)
    copyto!(ws.P[horizon], Q[schedule[horizon]])
    Symmetrize!(ws.P[horizon])
    for t in (horizon - 1):-1:1
        Pn = ws.P[t + 1]
        BtP = ws.mn
        mul!(BtP, transpose(B), Pn)
        G = ws.G[t]
        copyto!(G, R)
        mul!(G, BtP, B, one(T), one(T))
        F = cholesky!(Hermitian(G, :U))
        K = ws.K[t]
        mul!(K, BtP, A)
        ldiv!(F, K)
        Φ = ws.Φ[t]
        copyto!(Φ, A)
        mul!(Φ, B, K, -one(T), one(T))
        Pt = ws.P[t]
        copyto!(Pt, Q[schedule[t]])
        mul!(ws.nn, Pn, Φ)
        mul!(Pt, transpose(Φ), ws.nn, one(T), one(T))
        mul!(ws.mn2, R, K)
        mul!(Pt, transpose(K), ws.mn2, one(T), one(T))
        Symmetrize!(Pt)
    end
    return ws
end

"""
    affine_sweep!(ws, B, q) -> ws

The affine half of the sweep for the linear state cost `q` (`n × horizon`,
column `t` is `q_t`): `b_t` and the feedforward `k_t`, written into `ws.b` and
`ws.k`. Reads the gains and factors [`riccati_gain!`](@ref) left in `ws`.
`O(horizon · n²)`, against the gain sweep's `O(horizon · n³)`.
"""
function affine_sweep!(
    ws::RiccatiSweep{T}, B::AbstractMatrix, q::AbstractMatrix
) where {T<:Real}
    n, m = _sweep_dims(ws)
    horizon = _sweep_horizon(ws)
    size(B) == (n, m) || throw(DimensionMismatch("B must be $n×$m, got $(size(B))"))
    size(q) == (n, horizon) ||
        throw(DimensionMismatch("q must be $n×$horizon, got $(size(q))"))
    b, k = ws.b, ws.k
    copyto!(view(b, :, horizon), view(q, :, horizon))
    for t in (horizon - 1):-1:1
        bn = view(b, :, t + 1)
        kt = view(k, :, t)
        mul!(kt, transpose(B), bn)
        ldiv!(_gain_factor(ws, t), kt)
        kt .*= -one(T)
        bt = view(b, :, t)
        copyto!(bt, view(q, :, t))
        mul!(bt, transpose(ws.Φ[t]), bn, one(T), one(T))
    end
    return ws
end

"""
    tracking_cost!(q, Q, schedule, r) -> q

The linear cost of tracking the reference `r`: `q_t = −Q_{schedule[t]} r_t`, with
`r` either one `n`-vector for the whole horizon or an `n × horizon` matrix of
per-step references. The constant `½ rᵀ Q r` does not move the policy and is
left out.
"""
function tracking_cost!(
    q::AbstractMatrix,
    Q::AbstractVector{<:AbstractMatrix},
    schedule::AbstractVector{<:Integer},
    r::AbstractVecOrMat,
)
    size(q, 2) == length(schedule) || throw(
        DimensionMismatch("q has $(size(q, 2)) columns for $(length(schedule)) steps")
    )
    for t in axes(q, 2)
        rt = r isa AbstractVector ? r : view(r, :, t)
        mul!(view(q, :, t), Q[schedule[t]], rt, -1, false)
    end
    return q
end

"""
    tracking_cost_adjoint!(Qbar, rbar, qbar, Q, schedule, r) -> (Qbar, rbar)

Pull the gradient with respect to `q` (as [`riccati_adjoint!`](@ref) leaves it
in `grad.q`) back through [`tracking_cost!`](@ref) and add it to `Qbar` (one
symmetric matrix per regime) and `rbar` (shaped like `r`: a constant reference
collects every step's contribution).
"""
function tracking_cost_adjoint!(
    Qbar::AbstractVector{<:AbstractMatrix},
    rbar::AbstractVecOrMat,
    qbar::AbstractMatrix,
    Q::AbstractVector{<:AbstractMatrix},
    schedule::AbstractVector{<:Integer},
    r::AbstractVecOrMat,
)
    size(rbar) == size(r) ||
        throw(DimensionMismatch("rbar must match r's size $(size(r)), got $(size(rbar))"))
    for t in axes(qbar, 2)
        k = schedule[t]
        qt = view(qbar, :, t)
        rt = r isa AbstractVector ? r : view(r, :, t)
        rbt = rbar isa AbstractVector ? rbar : view(rbar, :, t)
        mul!(rbt, Q[k], qt, -1, true)
        _outer!(Qbar[k], -0.5, qt, rt)
        _outer!(Qbar[k], -0.5, rt, qt)
    end
    return Qbar, rbar
end

"""
    RiccatiGradient{T}(n, m, nregimes, horizon)

What [`riccati_adjoint!`](@ref) accumulates into: the gradient of a scalar loss
with respect to `A` (`n × n`), `B` (`n × m`), `R` (`m × m`), each regime's `Q`
(`n × n`) and the linear cost `q` (`n × horizon`).

`R` and `Q` are symmetric parameters and their gradients are the symmetric
matrices `R̄`, `Q̄` with `dL = ⟨R̄, dR⟩` for every *symmetric* `dR`. A caller that
parameterizes the upper triangle reads `2R̄[i, j]` off an off-diagonal entry.
"""
struct RiccatiGradient{T<:Real}
    A::Matrix{T}
    B::Matrix{T}
    R::Matrix{T}
    Q::Vector{Matrix{T}}
    q::Matrix{T}
end

function RiccatiGradient{T}(
    n::Integer, m::Integer, nregimes::Integer, horizon::Integer
) where {T<:Real}
    return RiccatiGradient{T}(
        zeros(T, n, n),
        zeros(T, n, m),
        zeros(T, m, m),
        [zeros(T, n, n) for _ in 1:nregimes],
        zeros(T, n, horizon),
    )
end

"""Zero every accumulator of `g`, for reuse across evaluations."""
function _zero_gradient!(g::RiccatiGradient)
    fill!(g.A, 0)
    fill!(g.B, 0)
    fill!(g.R, 0)
    foreach(Qk -> fill!(Qk, 0), g.Q)
    fill!(g.q, 0)
    return g
end

"""
    riccati_adjoint!(grad, ws, A, B, R, Q, schedule, Φbar, kbar; Pbar, bbar) -> grad

Reverse-mode pass through [`riccati_gain!`](@ref) and [`affine_sweep!`](@ref):
given a scalar loss's sensitivities to the sweep's outputs, add its gradient
with respect to `A`, `B`, `R`, every `Q_k` and `q` into `grad`.

The sensitivities, any of which may be `nothing` for zero:
- `Φbar[t]` (`n × n`) and `kbar[:, t]` (`m`), for `t = 1:horizon-1`;
- `Pbar[t]` (`n × n`, its symmetric part is used) and `bbar[:, t]` (`n`), for
  `t = 1:horizon`, as keywords.

`ws` must hold the sweep at these parameters: the gain sweep always, and the
affine sweep whenever `kbar` or `bbar` is given.

One forward-in-time pass, `O(horizon · (n³ + n²m + m³))`, and no implicit
solve: because `K_t` minimizes the stage cost, every `dK_t` term in `dP_t`
cancels (the envelope theorem), leaving the linear recursion
`dP_t = dQ + K_tᵀ dR K_t + (dA − dB K_t)ᵀ P_{t+1} Φ_t + Φ_tᵀ P_{t+1} (dA − dB K_t) + Φ_tᵀ dP_{t+1} Φ_t`.
`K_t` is still a function of the parameters in its own right, and its
sensitivity through `Φbar` and `kbar` is carried separately.
"""
function riccati_adjoint!(
    grad::RiccatiGradient,
    ws::RiccatiSweep{T},
    A::AbstractMatrix,
    B::AbstractMatrix,
    R::AbstractMatrix,
    Q::AbstractVector{<:AbstractMatrix},
    schedule::AbstractVector{<:Integer},
    Φbar::Union{Nothing,AbstractVector{<:AbstractMatrix}},
    kbar::Union{Nothing,AbstractMatrix};
    Pbar::Union{Nothing,AbstractVector{<:AbstractMatrix}}=nothing,
    bbar::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real}
    _check_sweep(ws, A, B, R, Q, schedule)
    n, m = _sweep_dims(ws)
    horizon = _sweep_horizon(ws)
    size(grad.A) == (n, n) && size(grad.B) == (n, m) && size(grad.q) == (n, horizon) ||
        throw(DimensionMismatch("gradient accumulators do not match the sweep"))
    length(grad.Q) >= length(Q) ||
        throw(DimensionMismatch("gradient has $(length(grad.Q)) Q slots for $(length(Q))"))
    Φbar === nothing ||
        length(Φbar) == horizon - 1 ||
        throw(DimensionMismatch("Φbar needs one entry per transition ($(horizon - 1))"))
    kbar === nothing ||
        size(kbar) == (m, horizon - 1) ||
        throw(DimensionMismatch("kbar must be $m×$(horizon - 1), got $(size(kbar))"))
    Pbar === nothing ||
        length(Pbar) == horizon ||
        throw(DimensionMismatch("Pbar needs one entry per step ($horizon)"))
    bbar === nothing ||
        size(bbar) == (n, horizon) ||
        throw(DimensionMismatch("bbar must be $n×$horizon, got $(size(bbar))"))
    affine = kbar !== nothing || bbar !== nothing

    P̄, P̄n = ws.Pbar, ws.Pbar_next
    b̄, b̄n = ws.bbar, ws.bbar_next
    Φ̄, Ḡ = ws.Φbar, ws.mm
    fill!(P̄, 0)
    fill!(b̄, 0)
    Pbar === nothing || _add_sym!(P̄, one(T), Pbar[1])
    bbar === nothing || (b̄ .+= view(bbar, :, 1))

    for t in 1:(horizon - 1)
        P1, Φ, K = ws.P[t + 1], ws.Φ[t], ws.K[t]
        F = _gain_factor(ws, t)
        # Seeds for step t + 1; the pass below adds what step t sends it.
        fill!(P̄n, 0)
        fill!(b̄n, 0)
        Pbar === nothing || _add_sym!(P̄n, one(T), Pbar[t + 1])
        bbar === nothing || (b̄n .+= view(bbar, :, t + 1))
        Φbar === nothing ? fill!(Φ̄, 0) : copyto!(Φ̄, Φbar[t])
        fill!(Ḡ, 0)

        if affine
            bn = view(ws.b, :, t + 1)
            # 1. through b_t = q_t + Φ_tᵀ b_{t+1}
            view(grad.q, :, t) .+= b̄
            _outer!(Φ̄, one(T), bn, b̄)
            mul!(b̄n, Φ, b̄, one(T), one(T))
            # 2. through k_t = −G_t⁻¹ Bᵀ b_{t+1}
            if kbar !== nothing
                z = ws.z
                copyto!(z, view(kbar, :, t))
                ldiv!(F, z)
                mul!(b̄n, B, z, -one(T), one(T))
                _outer!(grad.B, -one(T), bn, z)
                kt = view(ws.k, :, t)
                _outer!(Ḡ, -one(T) / 2, z, kt)
                _outer!(Ḡ, -one(T) / 2, kt, z)
            end
        end

        # 3. through P_t, by the envelope theorem
        grad.Q[schedule[t]] .+= P̄
        mul!(ws.mn2, K, P̄)
        mul!(grad.R, ws.mn2, transpose(K), one(T), one(T))
        PΦ = ws.nn
        mul!(PΦ, P1, Φ)
        mul!(ws.nn2, PΦ, P̄)
        grad.A .+= 2 .* ws.nn2
        mul!(grad.B, ws.nn2, transpose(K), -2 * one(T), one(T))
        mul!(ws.nn2, Φ, P̄)
        mul!(P̄n, ws.nn2, transpose(Φ), one(T), one(T))

        # 4. through Φ_t = A − B K_t and K_t = G_t⁻¹ Bᵀ P_{t+1} A
        grad.A .+= Φ̄
        mul!(grad.B, Φ̄, transpose(K), -one(T), one(T))
        Z = ws.mn
        mul!(Z, transpose(B), Φ̄)
        ldiv!(F, Z)
        Z .*= -one(T)
        mul!(grad.B, PΦ, transpose(Z), one(T), one(T))
        PB = ws.nm
        mul!(PB, P1, B)
        mul!(ws.nn2, PB, Z)
        mul!(grad.B, ws.nn2, transpose(K), -one(T), one(T))
        grad.A .+= ws.nn2
        mul!(ws.nn2, B, Z)
        mul!(ws.nn3, ws.nn2, transpose(Φ))
        _add_sym!(P̄n, one(T), ws.nn3)
        mul!(ws.mm2, Z, transpose(K))
        _add_sym!(grad.R, -one(T), ws.mm2)

        # 5. through G_t = R + Bᵀ P_{t+1} B
        grad.R .+= Ḡ
        mul!(grad.B, PB, Ḡ, 2 * one(T), one(T))
        mul!(ws.nm2, B, Ḡ)
        mul!(P̄n, ws.nm2, transpose(B), one(T), one(T))

        P̄, P̄n = P̄n, P̄
        b̄, b̄n = b̄n, b̄
    end
    grad.Q[schedule[horizon]] .+= P̄
    affine && (view(grad.q, :, horizon) .+= b̄)
    return grad
end
