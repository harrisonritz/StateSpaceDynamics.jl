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
# Improvement check in the sense direction (max wants ϕ > ϕ0; min wants ϕ < ϕ0).
# Strict comparison so a step that doesn't change ϕ at all doesn't count as
# "progress" — it just wastes Newton's budget.
@inline function _improves(::Val{:max}, ϕ, ϕ0)
    return isfinite(ϕ) && ϕ > ϕ0
end
@inline function _improves(::Val{:min}, ϕ, ϕ0)
    return isfinite(ϕ) && ϕ < ϕ0
end

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

    # Phase 1: halve α until ϕ is *finite and monotone* (improves on ϕ0).
    # `dϕ0 = g·p` is the directional derivative — if Newton handed us an
    # ascent/descent direction, calculus guarantees small-enough α gives
    # `ϕ > ϕ0` (max) / `ϕ < ϕ0` (min). The original loop only checked for
    # finiteness, which let phase 2 start from a non-monotone foothold;
    # then if phase 2's interpolation overshot back into the exp-overflow
    # regime and bailed, the outer Newton step regressed.
    h = 0
    while h < ls.max_halvings && !_improves(sense, ϕx1, ϕ0)
        h += 1
        @. x = x - α2*p     # revert
        α1 = α2
        α2 *= T(0.5)
        @. x = x + α2*p
        ϕx1 = ϕ!()
    end

    # Phase 1 couldn't find any monotone finite step — direction is bad
    # (orthogonal-to-gradient roundoff, or `||p||` so huge that even
    # `α = 2^-max_halvings · ||p||` lands at an overflow). Revert to
    # `x_start` and return zero progress; Newton's outer `α*||p|| < tol`
    # check handles termination from here.
    if !_improves(sense, ϕx1, ϕ0)
        @. x = x - α2*p
        return zero(T), ϕ0
    end

    # Track the best step found so we can fall back to it if phase 2's
    # interpolation lands at a non-finite point. `α_best` always satisfies
    # the improvement check.
    α_best = α2
    ϕ_best = ϕx1

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

        # Interp itself produced a non-finite trial step (typically from a
        # zero / NaN denominator). Fall back to the best monotone step
        # found so far — phase 1 guaranteed this exists.
        if !isfinite(αtmp)
            @. x = x - α2*p
            @. x = x + α_best*p
            return α_best, ϕ_best
        end

        # update step: revert old α2, apply αtmp
        @. x = x - α2*p
        α1 = α2
        α2 = αtmp
        @. x = x + α2*p

        ϕx0, ϕx1 = ϕx1, ϕ!()

        # Phase 2 landed at a point where `ϕ!()` overflows even though α
        # shrank (cubic interpolation can pick the upper safeguard
        # `α2*ρ_hi`, leaving α only halved — borderline overflow inputs can
        # still tip). Fall back to the best monotone step.
        if !isfinite(ϕx1)
            @. x = x - α2*p
            @. x = x + α_best*p
            return α_best, ϕ_best
        end

        # Track the best monotone step seen so far for potential fallback.
        if _improves(sense, ϕx1, ϕ_best)
            α_best = α2
            ϕ_best = ϕx1
        end
    end

    # Out of phase-2 iters; return whatever we converged toward. The
    # best-monotone fallback isn't needed here because `ϕx1` is finite by
    # the loop invariant.
    return α2, ϕx1
end
