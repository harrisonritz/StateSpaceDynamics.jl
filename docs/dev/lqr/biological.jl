#=============================================================================
Inverse control for a system that may not be optimizing — the measurements
behind `biological.md`.

Self-contained, like `parameterization.jl`: it imports nothing from
`StateSpaceDynamics`, so its verdicts about the model in `src/lds/` do not
inherit that model's own kernels.

    julia --project=. docs/dev/lqr/biological.jl
    julia --project=. docs/dev/lqr/biological.jl --only=residuals
    julia --project=. docs/dev/lqr/biological.jl --only=threshold   # slow

Sections:

  residuals  which of the three optimality relations survives which noise
             source — the result that decides which constraint to impose
  gauge      S = I as a change of latent basis: what it costs, what it buys,
             and why it is unavailable when control authority is rank deficient
  identify   is a one-scalar suboptimality parameter estimable at all, for two
             slack models crossed with two emissions
  threshold  how large the slack must be before it is detectable (slow)
=============================================================================#

using LinearAlgebra, Random, Statistics, Printf

const ONLY = let
    a = filter(s -> startswith(s, "--only="), ARGS)
    isempty(a) ? nothing : Set(split(split(a[1], "=")[2], ","))
end
want(s) = ONLY === nothing || s in ONLY
section(t) = (println(); println("="^78); println(t); println("="^78))

# ------------------------------------------------------------------ the problem
"""
A damped 2-D point mass: `x = (p1, p2, v1, v2)`, `u` = force. Note `m = 2 < n = 4`,
so control authority is rank deficient — which is the realistic case, and the
one that decides what `S = I` can and cannot mean.
"""
function plant(; dt = 0.05, damp = 0.5, mass = 1.0)
    A = [I(2) dt*I(2); zeros(2, 2) (1 - dt*damp)*I(2)]
    B = [zeros(2, 2); (dt/mass)*I(2)]
    return Matrix{Float64}(A), Matrix{Float64}(B)
end
function true_costs()
    Lr = 10*[20.0 0 0 0; 3 18 0 0; 2 -1 6 0; -1.5 2 1 5]
    Lt = 10*[120.0 0 0 0; 15 110 0 0; 10 -8 40 0; -6 12 5 35]
    return [Matrix(Symmetric(Lr*Lr')), Matrix(Symmetric(Lt*Lt'))]
end

const A, B = plant()
const n, m = size(A, 1), size(B, 2)
const Rc = Matrix{Float64}(I, m, m)
const S  = B*inv(Rc)*B'
const Qs = true_costs()
const T  = 30
const SCHED = vcat(fill(1, T-1), 2)
const P0 = Matrix(1e-6*I, n, n)
const REFS = [[0.12cos(2pi*(j-1)/8), 0.12sin(2pi*(j-1)/8), 0.0, 0.0] for j in 1:8]

"Backward Riccati and affine sweeps. Returns (K, kff, P, Fs)."
function sweep(r::Vector{Float64}; Aq = A, Sq = S, Qq = Qs)
    P = Vector{Matrix{Float64}}(undef, T); b = Vector{Vector{Float64}}(undef, T)
    K = Vector{Matrix{Float64}}(undef, T-1); kf = Vector{Vector{Float64}}(undef, T-1)
    Fs = Vector{Cholesky{Float64,Matrix{Float64}}}(undef, T-1)
    P[T] = Qq[SCHED[T]]; b[T] = -Qq[SCHED[T]]*r
    for t in (T-1):-1:1
        Fs[t] = cholesky(Symmetric(Rc + B'*P[t+1]*B))
        K[t]  = Fs[t] \ (B'*P[t+1]*Aq)
        kf[t] = -(Fs[t] \ (B'*b[t+1]))
        Acl   = Aq - B*K[t]
        Pt    = Qq[SCHED[t]] + K[t]'*Rc*K[t] + Acl'*P[t+1]*Acl
        P[t]  = (Pt + Pt')/2
        b[t]  = -Qq[SCHED[t]]*r + Acl'*(P[t+1]*B*kf[t] + b[t+1]) + K[t]'*Rc*kf[t]
    end
    return K, kf, P, Fs
end

const K1, _, P1, FS1 = sweep(zeros(n))
const PHI = [A - B*K1[t] for t in 1:(T-1)]
const WM  = [inv(I + S*P1[t]) for t in 1:T]          # (I + S P_t)^{-1}
const FF  = [[B*sweep(r)[2][t] for t in 1:(T-1)] for r in REFS]
const XI  = [inv(FS1[t]) for t in 1:(T-1)]           # (R + B'P_{t+1}B)^{-1}

# =========================================================================
function section_residuals()
    section("residuals — which optimality relation survives which noise source")
    ric(x, l, t) = l[:, t] - P1[t]*x[:, t]
    adj(x, l, t) = l[:, t] - Qs[SCHED[t]]*x[:, t] - A'*l[:, t+1]
    sta(x, l, t) = x[:, t+1] - A*x[:, t] + S*l[:, t+1]
    Nmap(t) = -A'*P1[t+1]*WM[t+1]

    "An agent that re-optimizes every step, with plant noise and costate slack."
    function agent(rng; sx, sn)
        x = zeros(n, T); l = zeros(n, T); x[:, 1] = 0.05*randn(rng, n)
        nu = [sn > 0 ? sn*randn(rng, n) : zeros(n) for _ in 1:T]
        for t in 1:T
            l[:, t] = P1[t]*x[:, t] + nu[t]      # the agent's own (perturbed) costate
            t == T && break
            rhs = A*x[:, t] - S*nu[t+1]
            sx > 0 && (rhs += sx*randn(rng, n))
            x[:, t+1] = WM[t+1]*rhs              # it acts on that costate
            l[:, t+1] = P1[t+1]*x[:, t+1] + nu[t+1]
        end
        return x, l
    end

    println("Mean residual norm per transition, 200 trials, n = $n, T = $T.")
    println("  (i)   ric   = λ_t − P_t x_t                  Riccati graph")
    println("  (ii)  adj   = λ_t − Q_t x_t − Aᵀλ_{t+1}      adjoint recursion")
    println("  (iii) state = x_{t+1} − A x_t + S λ_{t+1}    state equation")
    println("  N_t = −Aᵀ P_{t+1} (I + S P_{t+1})^{-1}")
    println()
    @printf("%-32s %-11s %-12s %-11s %s\n", "generating agent", "(i) ric", "(ii) adj",
            "(iii) state", "‖adj − N·state‖/‖adj‖")
    for (nm, sx, sn) in (("exactly optimal, no noise", 0.0, 0.0),
                         ("plant noise, feedback agent", 0.02, 0.0),
                         ("costate slack only (suboptimal)", 0.0, 0.5),
                         ("both", 0.02, 0.5))
        rng = MersenneTwister(3)
        rr = Float64[]; ar = Float64[]; sr = Float64[]; num = 0.0; den = 0.0
        for _ in 1:200
            x, l = agent(rng; sx = sx, sn = sn)
            for t in 1:(T-1)
                a = adj(x, l, t); s = sta(x, l, t)
                push!(rr, norm(ric(x, l, t))); push!(ar, norm(a)); push!(sr, norm(s))
                num += norm(a - Nmap(t)*s); den += norm(a)
            end
        end
        @printf("%-32s %-11.3g %-12.3g %-11.3g %.3g\n", nm, mean(rr), mean(ar), mean(sr),
                num/(den + 1e-300))
    end
    println()
    println("Read columns (i) and (ii) across the last two rows:")
    println("  the Riccati residual is EXACTLY blind to plant noise and responds only to")
    println("  suboptimality; the adjoint residual responds overwhelmingly to plant noise")
    println("  and is unmoved by suboptimality. The identity adj = N·state confirms why.")
    println("  => Σ_λλ in the mixed-coordinate model is a plant-noise parameter, not a")
    println("     measure of how far from optimal the system is.")
end

# =========================================================================
function section_gauge()
    section("gauge — S = I as a change of latent basis")
    @printf("S = B R⁻¹ Bᵀ has rank %d of %d; eigenvalues %s\n",
            rank(S; atol = 1e-12), n, join(round.(eigvals(Symmetric(S)), sigdigits = 3), ", "))
    println("With m = $m control channels and n = $n latent dimensions S is SINGULAR, so no")
    println("change of basis makes it the identity. Whitening needs full-rank control")
    println("authority. Demonstrate the gauge on a full-rank stand-in:")
    Sf = S + 0.02*Matrix{Float64}(I, n, n)
    Tw = inv(sqrt(Symmetric(Sf)))
    Aw = Tw*A*inv(Tw); Sw = Tw*Sf*Tw'
    Qw = [inv(Tw)'*Q*inv(Tw) for Q in Qs]
    C  = Matrix{Float64}([I(2) zeros(2, 2)]); Cw = C*inv(Tw)
    function closed_loop(Aq, Sq, Qq)
        P = Vector{Matrix{Float64}}(undef, T); P[T] = Qq[SCHED[T]]
        for t in (T-1):-1:1
            Pt = Qq[SCHED[t]] + Aq'*P[t+1]*(inv(I + Sq*P[t+1])*Aq); P[t] = (Pt + Pt')/2
        end
        return [inv(I + Sq*P[t+1])*Aq for t in 1:(T-1)], P
    end
    cl0, Pa = closed_loop(A, Sf, Qs); clw, Pb = closed_loop(Aw, Sw, Qw)
    @printf("  ‖T S Tᵀ − I‖                              = %.2e   (T = S^{-1/2})\n", norm(Sw - I))
    @printf("  max_t ‖T Φ_t T⁻¹ − Φ*_t‖/‖Φ_t‖            = %.2e   (closed loop is similar)\n",
            maximum(t -> norm(Tw*cl0[t]*inv(Tw) - clw[t])/norm(cl0[t]), 1:(T-1)))
    @printf("  max_t ‖C Φ_t − C* Φ*_t T‖                 = %.2e   (predictions identical)\n",
            maximum(t -> norm(C*cl0[t] - Cw*clw[t]*Tw), 1:(T-1)))
    @printf("  ‖T⁻ᵀ P₁ T⁻¹ − P*₁‖/‖P₁‖                   = %.2e   (P transforms as a cost)\n",
            norm(inv(Tw)'*Pa[1]*inv(Tw) - Pb[1])/norm(Pa[1]))
    println()
    println("  => with a fitted emission the latent basis is free, so S = I is a pure gauge")
    println("     choice when S is full rank, and it cuts the gauge group GL(n) → O(n).")
    O = Matrix(qr(randn(MersenneTwister(1), n, n)).Q)
    M = randn(MersenneTwister(2), n, n)
    @printf("  isotropy IS O(n)-invariant:  ‖Oᵀ(σ²I)O − σ²I‖ = %.2e\n", norm(O'*(2.5*I(n))*O - 2.5*I))
    @printf("  isotropy is NOT GL(n)-invariant: ‖Mᵀ(σ²I)M − σ²I‖ = %.3g\n", norm(M'*(2.5*I(n))*M - 2.5*I))
    println("  => an isotropic Σ_λλ only means something once the basis is fixed. The two")
    println("     constraints are complementary: adopt them together or neither.")
end

# =========================================================================
# Two one-scalar slack models. Both load on range(B); they differ in time course.
Om_slack(t, sx, s)  = WM[t+1]*(sx^2*Matrix{Float64}(I, n, n) + s^2*(S*S'))*WM[t+1]'
Om_maxent(t, sx, s) = sx^2*Matrix{Float64}(I, n, n) + s^2*(B*XI[t]*B')

function gen(rng, Om; sx, s, ntrial, sy, C)
    p = size(C, 1); L0 = cholesky(P0).L
    Ys = Vector{Matrix{Float64}}(undef, ntrial); tg = Vector{Int}(undef, ntrial)
    for i in 1:ntrial
        j = mod1(i, 8); tg[i] = j
        X = Matrix{Float64}(undef, n, T); X[:, 1] = L0*randn(rng, n)
        for t in 1:(T-1)
            L = cholesky(Symmetric(Om(t, sx, s) + 1e-14I)).L
            X[:, t+1] = PHI[t]*X[:, t] + FF[j][t] + L*randn(rng, n)
        end
        Ys[i] = C*X .+ sy*randn(rng, p, T)
    end
    return Ys, tg
end

"Exact marginal log likelihood; the process covariance is time varying."
function loglik(Ys, tg, Om, sx, s, sy, C)
    p = size(C, 1); Sy = sy^2*Matrix{Float64}(I, p, p)
    G = Vector{Matrix{Float64}}(undef, T)
    Fo = Vector{Cholesky{Float64,Matrix{Float64}}}(undef, T)
    ld = 0.0; Pc = P0
    for t in 1:T
        Fo[t] = cholesky(Symmetric(C*Pc*C' + Sy)); ld += logdet(Fo[t])
        G[t] = (Pc*C')/Fo[t]; Pf = Pc - G[t]*C*Pc; Pf = (Pf + Pf')/2
        t == T && break
        Pc = PHI[t]*Pf*PHI[t]' + Om(t, sx, s); Pc = (Pc + Pc')/2
    end
    q = 0.0; xh = Vector{Float64}(undef, n)
    for i in eachindex(Ys)
        Y = Ys[i]; ff = FF[tg[i]]; fill!(xh, 0.0)
        for t in 1:T
            e = @views Y[:, t] - C*xh; q += dot(e, Fo[t] \ e); xh .+= G[t]*e
            t == T && break
            xh .= PHI[t]*xh .+ ff[t]
        end
    end
    N = length(Ys)
    return -0.5*(N*T*p*log(2pi) + N*ld + q)
end

function profile(Om, sx0, s0, C, sy; smax, ntrial = 400, ngrid = 34)
    Ys, tg = gen(MersenneTwister(11), Om; sx = sx0, s = s0, ntrial = ntrial, sy = sy, C = C)
    best = (-Inf, 0.0, 0.0)
    for lx in range(log(5e-4), log(6e-2), length = ngrid),
        ls in vcat(-Inf, range(log(smax/200), log(smax), length = ngrid))
        sv = ls == -Inf ? 0.0 : exp(ls)
        v = loglik(Ys, tg, Om, exp(lx), sv, sy, C)
        v > best[1] && (best = (v, exp(lx), sv))
    end
    return best[2], best[3]
end

"Share of the mean innovation variance contributed by the slack term."
function phi(Om, sx, s)
    tot = mean(tr(Om(t, sx, s)) for t in 1:(T-1))
    bas = mean(tr(Om(t, sx, 0.0)) for t in 1:(T-1))
    return (tot - bas)/tot
end

const CPOS = Matrix{Float64}([I(2) zeros(2, 2)])
const CFUL = Matrix{Float64}(I, n, n)

function section_identify()
    section("identify — is a one-scalar suboptimality parameter estimable?")
    println("400 trials, T = $T, observation noise 0.006; σ_ν fitted jointly with σ_x.")
    println("  Riccati slack : Ω_t = W(σ_x²I + σ_ν² S Sᵀ)Wᵀ    — both terms share W_t")
    println("  max-ent noise : Ω_t = σ_x²I + β⁻¹ B(R+BᵀPB)⁻¹Bᵀ — the slack term varies in t")
    println()
    @printf("%-16s %-14s %-11s %-11s %-10s %s\n", "slack model", "emission", "true σ_ν",
            "fit σ_ν", "rel err", "verdict")
    for (nm, Om, smax) in (("Riccati slack", Om_slack, 6.0), ("max-ent noise", Om_maxent, 6.0))
        for (cn, C) in (("position only", CPOS), ("full state", CFUL))
            _, fs = profile(Om, 0.004, 1.0, C, 0.006; smax = smax)
            err = abs(fs - 1.0)
            @printf("%-16s %-14s %-11.3g %-11.4g %-10.3g %s\n", nm, cn, 1.0, fs, err,
                    err < 0.25 ? "RECOVERED" : "not identified")
            flush(stdout)
        end
    end
    println()
    println("Two structural lessons. Every slack model loads on range(B) — here the")
    println("velocity subspace — so an emission that does not span the control directions")
    println("cannot see suboptimality at any sample size. And plant noise competes in the")
    println("same slot, so shape alone is not enough: the max-ent term is estimable because")
    println("P_{t+1} gives it a time course constant plant noise cannot mimic.")
    println()
    println("CAUTION: these rows compare the models at the same NOMINAL scalar, which is")
    println("not the same observable effect (the max-ent term is ~4x larger here). Use")
    println("--only=threshold to locate the detection threshold for your own design.")
end

function section_threshold()
    section("threshold — how large must the slack be before it is detectable?")
    println("φ = share of the innovation variance contributed by the slack term.")
    println()
    @printf("%-16s %-14s %-9s %-10s %-10s %s\n", "model", "emission", "true σ_ν", "true φ",
            "fit σ_ν", "fit φ")
    for (nm, Om, smax, grid) in (("Riccati slack", Om_slack, 60.0, [1.0, 4.0, 12.0, 30.0]),
                                 ("max-ent noise", Om_maxent, 20.0, [0.3, 1.0, 3.0, 8.0]))
        for (cn, C) in (("position only", CPOS), ("full state", CFUL))
            for s0 in grid
                fx, fs = profile(Om, 0.004, s0, C, 0.006; smax = smax)
                @printf("%-16s %-14s %-9.3g %-10.3f %-10.4g %.3f\n", nm, cn, s0,
                        phi(Om, 0.004, s0), fs, phi(Om, fx, fs))
                flush(stdout)
            end
        end
    end
end

for (nm, f) in (("residuals", section_residuals), ("gauge", section_gauge),
                ("identify", section_identify), ("threshold", section_threshold))
    want(nm) && f()
end
println()
