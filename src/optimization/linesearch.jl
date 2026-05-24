abstract type AbstractLineSearch end

Base.@kwdef struct BackTrackingLS{T} <: AbstractLineSearch
    c1::T = 1e-4
    ρ_hi::T = 0.5
    ρ_lo::T = 0.1
    max_iters::Int = 25
    max_halvings::Int = 50
    order::Int = 3
end

# Armijo check, written so we can handle max/min with Val
@inline function armijo_ok(::Val{:max}, ϕ, ϕ0, α, dϕ0, c1)
    return ϕ >= ϕ0 + c1*α*dϕ0
end
@inline function armijo_ok(::Val{:min}, ϕ, ϕ0, α, dϕ0, c1)
    return ϕ <= ϕ0 + c1*α*dϕ0
end

"""
    backtracking!(sense, ls, x, p, ϕ!, ϕ0, dϕ0)

In-place backtracking along direction `p` from current `x`.
- `sense = Val(:max)` for maximizing ϕ
- `sense = Val(:min)` for minimizing ϕ
- `ϕ!()` must return ϕ(x) using current `x` (and should be allocation-free).
Returns (α, ϕ_new).
"""
function backtracking!(
    sense::Val,
    ls::BackTrackingLS{T},
    x::AbstractArray{T},
    p::AbstractArray{T},
    ϕ!::F,
    ϕ0::T,
    dϕ0::T,
) where {T<:Real,F}
    @assert ls.order == 2 || ls.order == 3

    α1 = one(T)
    α2 = one(T)

    # trial
    @. x = x + α2*p
    ϕx0 = ϕ0
    ϕx1 = ϕ!()

    # phase 1: ensure finite
    h = 0
    while !isfinite(ϕx1) && h < ls.max_halvings
        h += 1
        @. x = x - α2*p     # revert
        α1 = α2
        α2 *= T(0.5)
        @. x = x + α2*p
        ϕx1 = ϕ!()
    end

    # Bail if phase 1 couldn't get to a finite ϕ in `max_halvings` halvings.
    # Falling through into phase 2 with `ϕx1` non-finite poisons the cubic /
    # quadratic interpolation (`NaN - ϕ0 = NaN` in the denominator), which
    # produces `αtmp = NaN`, which then writes `NaN` to every entry of `x`
    # via `@. x = x + α2*p`. The Poisson smoother trips on this on the very
    # first Newton step from `x = 0` when the Hessian gives a large-norm
    # direction that drives `Cx + d` into the `exp` overflow regime —
    # halving α doesn't help because `||p||` is already huge.
    # Revert the most recent step and return zero progress; the outer
    # Newton loop's `α * norm(p) < tol` check handles termination from
    # here.
    if !isfinite(ϕx1)
        @. x = x - α2*p
        return zero(T), ϕ0
    end

    # phase 2: interpolation
    for k in 1:ls.max_iters
        if armijo_ok(sense, ϕx1, ϕ0, α2, dϕ0, ls.c1)
            return α2, ϕx1
        end

        # pick αtmp
        αtmp = α2
        if ls.order == 2 || k == 1
            denom = (ϕx1 - ϕ0 - dϕ0*α2)
            αtmp = -(dϕ0 * α2 * α2) / (2*denom)
        else
            div = one(T) / (α1 * α1 * α2 * α2 * (α2 - α1))
            a = (α1*α1*(ϕx1 - ϕ0 - dϕ0*α2) - α2*α2*(ϕx0 - ϕ0 - dϕ0*α1)) * div
            b = (-α1^3*(ϕx1 - ϕ0 - dϕ0*α2) + α2^3*(ϕx0 - ϕ0 - dϕ0*α1)) * div
            if abs(a) < eps(T)
                αtmp = -dϕ0 / (2*b)
            else
                disc = max(b*b - 3*a*dϕ0, zero(T))
                αtmp = (-b + sqrt(disc)) / (3*a)
            end
        end

        # safeguards
        αtmp = min(αtmp, α2*ls.ρ_hi)
        αtmp = max(αtmp, α2*ls.ρ_lo)

        # update step: revert old α2, apply αtmp
        @. x = x - α2*p
        α1 = α2
        α2 = αtmp
        @. x = x + α2*p

        ϕx0, ϕx1 = ϕx1, ϕ!()
    end

    # if we get here, return best we have (or throw)
    return α2, ϕx1
end
