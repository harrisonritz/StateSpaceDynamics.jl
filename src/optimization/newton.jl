"""
    newton_smooth!(x, compute_grad!, build_hess!, solve_dir!, ϕ!, linesearch; ...)

- `x` is D×T (or vector view)
- `compute_grad!(g, x)` writes gradient into preallocated `g`
- `build_hess!(x)` fills workspace Hessian blocks
- `solve_dir!(p, g)` solves for Newton direction into preallocated `p`
- `ϕ!()` evaluates objective at current `x`
"""
function newton_smooth!(
    sense::Val,
    x,
    g,
    p,
    compute_grad!,
    build_hess!,
    solve_dir!,
    ϕ!,
    ls::Union{Nothing,AbstractLineSearch};
    max_iter::Int=20,
    tol=1e-6,
)
    ϕ_prev = -Inf
    for it in 1:max_iter
        compute_grad!(g, x)
        # DIAGNOSTIC (Poisson PosDef CI flake, 2026-05-24): which iter
        # blows up and which step does it.
        if !all(isfinite, g)
            error("newton_smooth!: non-finite gradient at iter $it " *
                  "(n_nonfinite=$(count(!isfinite, g)) of $(length(g))). " *
                  "x finite? $(all(isfinite, x))  ϕ_prev=$ϕ_prev")
        end
        gn = norm(g)
        if gn < tol
            return true
        end

        build_hess!(x)
        solve_dir!(p, g)  # p is Newton direction
        if !all(isfinite, p)
            error("newton_smooth!: non-finite Newton direction `p` at iter $it " *
                  "(n_nonfinite=$(count(!isfinite, p)) of $(length(p))). " *
                  "gn=$gn  ϕ_prev=$ϕ_prev — likely a singular Hessian " *
                  "in the BT solve.")
        end

        # step selection
        if ls === nothing
            @. x = x + p
            ϕ_prev = ϕ!()
        else
            ϕ0 = ϕ!()
            dϕ0 = dot(vec(g), vec(p))  # should be > 0 for ascent; < 0 for descent
            α, ϕ_new = backtracking!(sense, ls, x, p, ϕ!, ϕ0, dϕ0)
            if !all(isfinite, x)
                error("newton_smooth!: non-finite `x` after backtracking! at " *
                      "iter $it (α=$α, ϕ_new=$ϕ_new, ϕ0=$ϕ0, dϕ0=$dϕ0). " *
                      "Line search couldn't recover to a finite step.")
            end
            # optional convergence checks:
            if abs(ϕ_new - ϕ_prev) < tol || α*norm(p) < tol
                return true
            end
            ϕ_prev = ϕ_new
        end
    end
    return false
end
