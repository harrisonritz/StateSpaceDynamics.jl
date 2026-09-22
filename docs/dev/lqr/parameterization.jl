#=============================================================================
Which parameterization? — the measurements behind `parameterization.md`.

This file is deliberately **self-contained**: it imports nothing from
`StateSpaceDynamics`, because its job is to compare the mixed-coordinate
(Hamiltonian) inverse-LQR model in `src/lds/` against alternatives, and a
comparison that reused the incumbent's own kernels would inherit the incumbent's
choices. Everything here is a few dozen lines of LinearAlgebra.

    julia --project=docs -t auto docs/dev/lqr/parameterization.jl
    julia --project=docs -t auto docs/dev/lqr/parameterization.jl --only=gauge,terminal
    julia --project=docs -t auto docs/dev/lqr/parameterization.jl --only=fit

Sections (`--only=` selects a comma-separated subset):

  geometry   reciprocal eigenvalue pairs; the symplectic defect; M-invariance of
             graph(P); the cost-to-go contraction identity
  gauge      the exact (S, Q) scale invariance, and the same ray with R pinned
  profile    the exact marginal log-likelihood along the cost-scale ray
  terminal   the Riccati Jacobian Ψ, checked against finite differences, and the
             backward decay of terminal-cost information
  reference  where a reference perturbation lands, in time and in observed rows
  switching  γ at the generating parameters, delay-then-reach, matched to slds.jl
  fit        end-to-end recovery, swept over the initial cost scale (slow)

The mixed-coordinate numbers quoted for contrast are `README.md`'s and
`slds.jl`'s own measurements, not re-runs here.
=============================================================================#

using LinearAlgebra, Random, Statistics, Printf
using Optim, LineSearches

const ONLY = let
    a = filter(s -> startswith(s, "--only="), ARGS)
    isempty(a) ? nothing : Set(split(split(a[1], "=")[2], ","))
end
want(s) = ONLY === nothing || s in ONLY
section(t) = (println(); println("="^78); println(t); println("="^78))

# =========================================================================
# The control problem: a damped 2-D point mass reaching to a ring of targets.
# =========================================================================

"x = (p1, p2, v1, v2), u = force. A is invertible, as `LQRStateModel` requires."
function plant(; dt = 0.05, damp = 0.5, mass = 1.0)
    A = [I(2) dt*I(2); zeros(2, 2) (1 - dt*damp)*I(2)]
    B = [zeros(2, 2); (dt/mass)*I(2)]
    return Matrix{Float64}(A), Matrix{Float64}(B)
end

"""
Running and terminal costs, both PSD with genuine off-diagonal structure so that
an entrywise correlation is a meaningful score. The overall size is set so the
reach lands within ~10 % of the target radius in `T` steps against `R = I` —
below that the agent undershoots and the task is not a reach.
"""
function true_costs()
    Lrun  = 10 * [20.0 0 0 0; 3.0 18.0 0 0; 2.0 -1.0 6.0 0; -1.5 2.0 1.0 5.0]
    Lterm = 10 * [120.0 0 0 0; 15.0 110.0 0 0; 10.0 -8.0 40.0 0; -6.0 12.0 5.0 35.0]
    sym(M) = Matrix((M + M') / 2)
    return [sym(Lrun * Lrun'), sym(Lterm * Lterm')]
end

struct Setup
    A::Matrix{Float64}; B::Matrix{Float64}; R::Matrix{Float64}; C::Matrix{Float64}
    sched::Vector{Int}; refs::Vector{Vector{Float64}}
    T::Int; n::Int; m::Int; p::Int
    P0::Matrix{Float64}; mu0::Vector{Float64}
end

"`R = I` is the gauge: it is what makes the cost scale a parameter and not an orbit."
function setup(; T = 30, ntarget = 8, radius = 0.12, observe = :position)
    A, B = plant(); n, m = size(A, 1), size(B, 2)
    C = Matrix{Float64}(observe === :position ? [I(2) zeros(2, 2)] : I(n))
    refs = [[radius*cos(2pi*(j-1)/ntarget), radius*sin(2pi*(j-1)/ntarget), 0.0, 0.0]
            for j in 1:ntarget]
    return Setup(A, B, Matrix{Float64}(I, m, m), C, vcat(fill(1, T-1), 2), refs,
                 T, n, m, size(C, 1), Matrix(1e-6*I, n, n), zeros(n))
end

"""
Backward Riccati sweep and its companion affine sweep, for

  min  Σ_{t<T} ½[(x_t-r)'Q_{k(t)}(x_t-r) + u'Ru] + ½(x_T-r)'Q_{k(T)}(x_T-r).

Returns `(K, kff, P)` with the optimal policy `u_t = -K_t x_t + kff_t`.
"""
function riccati(A, B, R, Qs, sched::Vector{Int}, r::Vector{Float64})
    T = length(sched); n = size(A, 1)
    P = Vector{Matrix{Float64}}(undef, T); b = Vector{Vector{Float64}}(undef, T)
    K = Vector{Matrix{Float64}}(undef, T-1); kf = Vector{Vector{Float64}}(undef, T-1)
    QT = Qs[sched[T]]; P[T] = QT; b[T] = -QT * r
    @inbounds for t in (T-1):-1:1
        Q = Qs[sched[t]]
        F = cholesky(Symmetric(R + B' * P[t+1] * B))
        K[t]  = F \ (B' * P[t+1] * A)
        kf[t] = -(F \ (B' * b[t+1]))
        Acl   = A - B * K[t]
        P[t]  = Q + K[t]' * R * K[t] + Acl' * P[t+1] * Acl
        P[t]  = (P[t] + P[t]') / 2
        b[t]  = -Q * r + Acl' * (P[t+1] * B * kf[t] + b[t+1]) + K[t]' * R * kf[t]
    end
    return K, kf, P
end

"Closed-loop forward roll with plant noise `Lx` and control noise `Lu` (factors)."
function rollout(rng, A, B, K, kf, x1, Lx, Lu)
    T = length(K) + 1; n, m = size(A, 1), size(B, 2)
    X = Matrix{Float64}(undef, n, T); X[:, 1] = x1
    @inbounds for t in 1:(T-1)
        u = -K[t] * X[:, t] + kf[t] + Lu * randn(rng, m)
        X[:, t+1] = A * X[:, t] + B * u + Lx * randn(rng, n)
    end
    return X
end

relrmse(f, r) = sqrt(mean(abs2, f .- r)) / sqrt(mean(abs2, r))
function pearson(f, r)
    fc = f .- mean(f); rc = r .- mean(r)
    d = sqrt(sum(abs2, fc) * sum(abs2, rc))
    return d < 1e-14 ? NaN : sum(fc .* rc) / d
end
utri(M) = [M[i, j] for j in 1:size(M, 2) for i in 1:j]
"Effective number of timesteps carrying a perturbation: (Σ‖d‖²)² / Σ‖d_t‖⁴."
partic(D) = sum(abs2, D)^2 / sum(t -> sum(abs2, D[:, t])^2, 1:size(D, 2))

# =========================================================================
# Exact marginal likelihood of the closed-loop chain.
# =========================================================================

"""
    loglik(st, Qs, refs, Σ_w, σ²_y, Ys, tg; A, B) -> Float64

`log p(y_{1:T} | θ)` for the closed-loop time-varying LDS

    x_{t+1} = (A - B K_t) x_t + B kff_t + w_t,   y_t = C x_t + ν_t,

marginalised exactly by Kalman filtering. The covariance/gain recursion depends
on neither the target nor the data, so it is built once per parameter evaluation
and shared across every trial — which is what makes a finite-difference gradient
over ~40 parameters affordable.
"""
function loglik(st::Setup, Qs, refs, Sig_w, sig_y2, Ys, tg; A = st.A, B = st.B)
    T, n, p = st.T, st.n, st.p
    K, _, _ = riccati(A, B, st.R, Qs, st.sched, zeros(n))
    Acl  = [A - B * K[t] for t in 1:(T-1)]
    inps = [[B * riccati(A, B, st.R, Qs, st.sched, r)[2][t] for t in 1:(T-1)] for r in refs]

    Sy = sig_y2 * Matrix{Float64}(I, p, p)
    Gain = Vector{Matrix{Float64}}(undef, T)
    Fo = Vector{Cholesky{Float64,Matrix{Float64}}}(undef, T)
    logdet_sum = 0.0; Pc = st.P0
    for t in 1:T
        Fo[t] = cholesky(Symmetric(st.C * Pc * st.C' + Sy))
        logdet_sum += logdet(Fo[t])
        Gain[t] = (Pc * st.C') / Fo[t]
        Pf = Pc - Gain[t] * st.C * Pc; Pf = (Pf + Pf') / 2
        t == T && break
        Pc = Acl[t] * Pf * Acl[t]' + Sig_w; Pc = (Pc + Pc') / 2
    end

    quad = 0.0; xh = Vector{Float64}(undef, n)
    for i in eachindex(Ys)
        Y = Ys[i]; inp = inps[tg[i]]; copyto!(xh, st.mu0)
        for t in 1:T
            e = @views Y[:, t] - st.C * xh
            quad += dot(e, Fo[t] \ e)
            xh .+= Gain[t] * e
            t == T && break
            xh .= Acl[t] * xh .+ inp[t]
        end
    end
    N = length(Ys)
    return -0.5 * (N * T * p * log(2pi) + N * logdet_sum + quad)
end

function gen(rng, st::Setup, Qs; ntrial = 400, sig_x = 0.004, sig_u = 0.25, sig_y = 0.006)
    K, _, _ = riccati(st.A, st.B, st.R, Qs, st.sched, zeros(st.n))
    kffs = [riccati(st.A, st.B, st.R, Qs, st.sched, r)[2] for r in st.refs]
    Lx = sig_x * Matrix{Float64}(I, st.n, st.n); Lu = sig_u * Matrix{Float64}(I, st.m, st.m)
    L0 = cholesky(st.P0).L
    Ys = Vector{Matrix{Float64}}(undef, ntrial); tg = Vector{Int}(undef, ntrial)
    for i in 1:ntrial
        j = mod1(i, length(st.refs)); tg[i] = j
        X = rollout(rng, st.A, st.B, K, kffs[j], st.mu0 + L0*randn(rng, st.n), Lx, Lu)
        Ys[i] = st.C * X .+ sig_y * randn(rng, st.p, st.T)
    end
    return Ys, tg
end

# =========================================================================
section_geometry = function ()
    section("geometry — the symplectic flow, its stable manifold, and the contraction")
    st = setup(); Qs = true_costs()
    K, _, P = riccati(st.A, st.B, st.R, Qs, st.sched, zeros(st.n))
    Phi = [st.A - st.B * K[t] for t in 1:(st.T-1)]

    err = maximum(t -> norm(Phi[t]' * P[t+1] * Phi[t] -
                            (P[t] - (Qs[st.sched[t]] + K[t]' * st.R * K[t]))) / norm(P[t]),
                  1:(st.T-1))
    println("Contraction identity  Φ_t' P_{t+1} Φ_t = P_t - (Q_t + K_t' R K_t)")
    @printf("  max relative residual over t : %.1e\n", err)
    @printf("  max_t ρ(Φ_t)                 : %.4f  (individual steps need not be < 1;\n",
            maximum(t -> maximum(abs, eigvals(Phi[t])), 1:(st.T-1)))
    println("                                  the product contracts in the P-metric)")

    S = st.B * inv(st.R) * st.B'; Ai = inv(st.A)'
    M = [st.A + S*Ai*Qs[1]  -S*Ai; -Ai*Qs[1]  Ai]
    J = [zeros(st.n, st.n) I(st.n); -I(st.n) zeros(st.n, st.n)]
    ab = sort(abs.(eigvals(M)))
    println()
    println("The mixed-coordinate forward transition M (the model's own generative form)")
    println("  |μ|                    : ", join(round.(ab, digits=4), ", "))
    println("  reciprocal-pair products: ",
            join([round(ab[i] * ab[2*st.n + 1 - i], digits=8) for i in 1:st.n], ", "))
    @printf("  symplectic defect ‖M'JM - J‖/‖J‖ : %.1e\n", norm(M'*J*M - J)/norm(J))
    @printf("  growth over T = %d steps          : %.3g   <- why `rand` is unusable\n",
            st.T, maximum(ab)^st.T)

    V = [Matrix{Float64}(I, st.n, st.n); P[1]]      # the graph {λ = P x}
    W = M * V
    @printf("\n  ‖(I - proj)(M · graph(P))‖ / ‖M · graph(P)‖ : %.1e\n", norm(W - V*(V\W))/norm(W))
    println("  => graph(P) is M-invariant: it IS the stable manifold. The closed-loop")
    println("     model parameterises that manifold; the mixed-coordinate model")
    println("     parameterises the whole 2n-dim flow and conditions its way back.")
end

section_gauge = function ()
    section("gauge — the cost scale is an exact flat direction only when S is free")
    st = setup(); Qs = true_costs()
    mp(Q, r, R = st.R) = (kk = riccati(st.A, st.B, R, Q, st.sched, r);
                          rollout(MersenneTwister(1), st.A, st.B, kk[1], kk[2],
                                  zeros(st.n), zeros(st.n, st.n), zeros(st.m, st.m)))
    g1(Q, R = st.R) = riccati(st.A, st.B, R, Q, st.sched, zeros(st.n))[1][1]
    X0 = mp(Qs, st.refs[1]); K0 = g1(Qs)

    println("Q -> cQ with R = I held fixed:")
    @printf("%-8s %-16s %-14s %-19s %s\n", "c", "d‖K_1‖/‖K_1‖", "ρ(A-BK_1)",
            "d mean path (rel)", "endpoint |x_T - r|")
    for c in [0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0]
        Qc = [c*Q for Q in Qs]; K1 = g1(Qc); X = mp(Qc, st.refs[1])
        @printf("%-8s %-16.4f %-14.4f %-19.4f %.5f\n", c, norm(K1-K0)/norm(K0),
                maximum(abs, eigvals(st.A - st.B*K1)), norm(X-X0)/norm(X0),
                norm(X[1:2, end] - st.refs[1][1:2]))
    end
    println("\nThe same c, with S = B R^-1 B' free to absorb it (S -> S/c, i.e. R -> cR):")
    println("  this is `rescale_costate!`'s (λ,S,Q,…) -> (cλ,c⁻¹S,cQ,…)")
    for c in [0.25, 1.0, 4.0]
        Qc = [c*Q for Q in Qs]; K1 = g1(Qc, c*st.R); X = mp(Qc, st.refs[1], c*st.R)
        @printf("  c = %-6s d‖K_1‖/‖K_1‖ = %-10.3g d mean path = %.3g\n",
                c, norm(K1-K0)/norm(K0), norm(X-X0)/norm(X0))
    end
    println("  => exactly flat. Whenever S and Σ are both fitted the cost scale is a")
    println("     gauge orbit, so no objective and no amount of data can locate it.")
end

section_profile = function ()
    section("profile — the exact marginal likelihood along the cost-scale ray")
    st = setup(); Qs = true_costs()
    sx, su, sy = 0.004, 0.25, 0.006
    Sw = sx^2*Matrix(I, st.n, st.n) + su^2*(st.B*st.B')
    Ys, tg = gen(MersenneTwister(20240922), st, Qs; ntrial = 400,
                 sig_x = sx, sig_u = su, sig_y = sy)
    per = 1/(length(Ys)*st.T)
    base = per*loglik(st, Qs, st.refs, Sw, sy^2, Ys, tg)
    println("400 trials, T = $(st.T), position-only emission; all else at the truth.")
    @printf("\n%-10s %-16s %s\n", "c", "loglik/step", "deficit vs c = 1")
    cs = [0.0625, 0.125, 0.25, 0.5, 0.71, 1.0, 1.41, 2.0, 4.0, 8.0, 16.0]
    lls = Float64[]
    for c in cs
        ll = per*loglik(st, [c*Q for Q in Qs], st.refs, Sw, sy^2, Ys, tg)
        push!(lls, ll); @printf("%-10s %-16.6f %+.6f\n", c, ll, ll - base)
    end
    println("\nargmax over the ray: c = ", cs[argmax(lls)])
    println("=> an interior maximum at the true scale. Contrast README.md, where the")
    println("   held-out score is monotone in q0 and prefers the smallest value offered.")
end

section_terminal = function ()
    section("terminal — the Riccati Jacobian, and how far back a cost is felt")
    st = setup(); Qs = true_costs()
    K, _, P = riccati(st.A, st.B, st.R, Qs, st.sched, zeros(st.n))
    Phi = [st.A - st.B*K[t] for t in 1:(st.T-1)]
    # Ψ_{t,s} = Φ_{s-1} Φ_{s-2} ⋯ Φ_t  (order matters; the Φ do not commute)
    Psi(t, s) = (Z = Matrix{Float64}(I, st.n, st.n); for j in (s-1):-1:t; Z = Z*Phi[j]; end; Z)

    println("dP_t = dQ_t + Φ_t' dP_{t+1} Φ_t   (every dK term cancels by optimality)")
    println("  =>  dP_t = Σ_{s≥t} Ψ_{t,s}' dQ_s Ψ_{t,s}   —  closed form, no AD, no")
    println("      implicit solve; one extra backward sweep per gradient.")
    dQ = zeros(st.n, st.n); dQ[1, 1] = 1.0; dQ = (dQ + dQ')/2
    Pp = riccati(st.A, st.B, st.R, [Qs[1], Qs[2] + dQ], st.sched, zeros(st.n))[3]
    fdmax = maximum(t -> norm(Psi(t, st.T)'*dQ*Psi(t, st.T) - (Pp[t] - P[t])) /
                         max(norm(Pp[t] - P[t]), 1e-30), 1:(st.T-6))
    @printf("  finite-difference check, max relative error over t : %.1e\n", fdmax)

    println("\nBackward decay of terminal-cost information, ‖Ψ_{t,T}‖²:")
    @printf("%-6s %-20s %s\n", "t", "‖Ψ_{t,T}‖²", "relative to t = T-1")
    b = norm(Psi(st.T-1, st.T))^2
    for t in [st.T-1, st.T-3, st.T-6, st.T-10, st.T-15, st.T-20, 1]
        v = norm(Psi(t, st.T))^2
        @printf("%-6d %-20.3g %.3g\n", t, v, v/b)
    end

    mp(Q) = (kk = riccati(st.A, st.B, st.R, Q, st.sched, st.refs[1]);
             rollout(MersenneTwister(1), st.A, st.B, kk[1], kk[2],
                     zeros(st.n), zeros(st.n, st.n), zeros(st.m, st.m)))
    X0 = mp(Qs); D = mp([Qs[1], 1.2*Qs[2]]) - X0
    println("\nWhere a 20 % change in Q_term lands (relative change in the mean state):")
    println("  t:     ", join([lpad(t, 7) for t in 1:3:st.T]))
    println("  rel d: ", join([lpad(round(norm(D[:,t])/max(norm(X0[:,t]),1e-12), digits=4), 7)
                               for t in 1:3:st.T]))
    @printf("  participation ratio %.2f of %d;  share in observed rows %.1f %%\n",
            partic(D), st.T, 100*sum(abs2, D[1:2, :])/sum(abs2, D))
    println("  (mixed-coordinate: Q_term enters ONE factor per trial, ratio 1.00 of $(st.T))")

    println("\nAgainst the running/terminal balance — a property of the task, not the fit:")
    @printf("%-14s %-22s %s\n", "Q_run scale", "participation ratio", "share in observed rows")
    for s in [0.0, 0.01, 0.1, 1.0, 10.0]
        Q = [s*Qs[1], Qs[2]]
        Y0 = mp(Q); Dd = mp([s*Qs[1], 1.2*Qs[2]]) - Y0
        @printf("%-14s %-22.2f %.1f %%\n", s, partic(Dd),
                100*sum(abs2, Dd[1:2, :])/sum(abs2, Dd))
    end
    println("=> the closed loop does NOT rescue the terminal cost. It is forgotten")
    println("   backwards at the closed-loop rate; fix it in the design, not the model.")
end

section_reference = function ()
    section("reference — the feedforward puts it in the observed coordinates")
    st = setup(); Qs = true_costs()
    mp(r) = (kk = riccati(st.A, st.B, st.R, Qs, st.sched, r);
             rollout(MersenneTwister(1), st.A, st.B, kk[1], kk[2],
                     zeros(st.n), zeros(st.n, st.n), zeros(st.m, st.m)))
    X0 = mp(st.refs[1]); rp = copy(st.refs[1]); rp[1:2] *= 1.1
    D = mp(rp) - X0
    println("One target scaled by 1.1 (relative change in the mean state):")
    println("  t:     ", join([lpad(t, 7) for t in 1:3:st.T]))
    println("  rel d: ", join([lpad(round(norm(D[:,t])/max(norm(X0[:,t]),1e-12), digits=4), 7)
                               for t in 1:3:st.T]))
    @printf("  participation ratio %.2f of %d;  share in observed rows %.1f %%\n",
            partic(D), st.T, 100*sum(abs2, D[1:2, :])/sum(abs2, D))
    println("=> identified from the state mean at essentially every timestep. In the")
    println("   mixed-coordinate model Gref enters only the costate half of the affine")
    println("   term, hence README.md's trade: Gref 1.041/0.01 in the best cost row.")
end

# =========================================================================
# switching: γ at the generating parameters, matched to slds.jl
# =========================================================================
logsumexp2(a, b) = (m = max(a, b); m == -Inf ? -Inf : m + log(exp(a-m) + exp(b-m)))

section_switching = function ()
    section("switching — γ at the generating parameters, delay-then-reach")
    n, T, NTRIAL, STAY = 2, 30, 120, 0.93
    Ap = [0.96 0.05; -0.04 0.96]                       # model.jl `plant(2)`
    Ssym = [0.05 0.01; 0.01 0.05]
    B = Matrix(sqrt(Symmetric(Ssym)))                  # B R⁻¹ B' = S at R = I
    Rc = Matrix{Float64}(I, n, n)
    Qrun = [0.20 0.03; 0.03 0.20]                      # model.jl `cost_bank(2)`
    Qterm = 3.0 * Matrix{Float64}(I, n, n)
    Sig_l = 0.02 * Matrix{Float64}(I, n, n)            # slds.jl `state_noise`
    Af, Sig_f = 0.90*Matrix{Float64}(I,n,n), 0.15*Matrix{Float64}(I,n,n)  # `free_drift`
    Cobs, Robs = Matrix{Float64}(I,n,n), 0.05*Matrix{Float64}(I,n,n)
    P0 = 0.2 * Matrix{Float64}(I, n, n)
    sched = vcat(fill(1, T-1), 2)
    K, kff, _ = riccati(Ap, B, Rc, [Qrun, Qterm], sched, zeros(n))
    Acl = [Ap - B*K[t] for t in 1:(T-1)]
    @printf("free drift ρ = %.3f   LQR closed loop ρ(Φ_1) = %.4f  (regimes told apart by\n",
            0.9, maximum(abs, eigvals(Acl[1])))
    println("  innovation size and feedforward, not by decay rate — not a rigged contrast)")
    trans(k, t) = k == 1 ? (Acl[t], B*kff[t], Sig_l) : (Af, zeros(n), Sig_f)

    rng = MersenneTwister(7)
    Lf, Ll, L0 = cholesky(Sig_f).L, cholesky(Sig_l).L, cholesky(P0).L
    Lo = cholesky(Robs).L
    Ys = Vector{Matrix{Float64}}(undef, NTRIAL); Zs = Vector{Vector{Int}}(undef, NTRIAL)
    sws = Vector{Int}(undef, NTRIAL)
    for i in 1:NTRIAL
        tsw = rand(rng, (T÷3):(2T÷3)); sws[i] = tsw
        X = Matrix{Float64}(undef, n, T); X[:, 1] = L0*randn(rng, n); z = fill(2, T)
        for t in 1:(T-1)
            if t + 1 <= tsw
                X[:, t+1] = Af*X[:, t] + Lf*randn(rng, n); z[t+1] = 2
            else
                X[:, t+1] = Acl[t]*X[:, t] + B*kff[t] + Ll*randn(rng, n); z[t+1] = 1
            end
        end
        Zs[i] = z; Ys[i] = Cobs*X .+ Lo*randn(rng, n, T)
    end

    # Structured variational E-step: responsibility-weighted Gaussian chain <-> HMM,
    # the same plug-in scheme the harness scores γ with.
    Ri, iP0 = inv(Robs), inv(P0)
    Pinv = [inv(Sig_l), inv(Sig_f)]
    ld = [logdet(2pi*Sig_l), logdet(2pi*Sig_f)]
    logA = log.([STAY 1-STAY; 1-STAY STAY]); logpi = log.([1e-6, 1-1e-6])
    rows(t) = ((t-1)*n+1):(t*n)
    tp = zeros(2); nk = zeros(2); mass = Float64[]; onset = Float64[]
    for i in 1:NTRIAL
        Y = Ys[i]; g = fill(0.5, 2, T); M1 = Matrix{Float64}(undef, n, T); Cv = nothing
        for _ in 1:60
            L = zeros(n*T, n*T); h = zeros(n*T)
            L[rows(1), rows(1)] += iP0
            for t in 1:T
                L[rows(t), rows(t)] += Cobs'*Ri*Cobs; h[rows(t)] += Cobs'*Ri*Y[:, t]
            end
            for t in 1:(T-1), k in 1:2
                w = g[k, t+1]; w < 1e-12 && continue
                Ak, bk, _ = trans(k, t); Pk = Pinv[k]
                L[rows(t), rows(t)]     += w*(Ak'*Pk*Ak)
                L[rows(t+1), rows(t+1)] += w*Pk
                L[rows(t), rows(t+1)]   -= w*(Ak'*Pk)
                L[rows(t+1), rows(t)]   -= w*(Pk*Ak)
                h[rows(t)]              -= w*(Ak'*Pk*bk)
                h[rows(t+1)]            += w*(Pk*bk)
            end
            F = cholesky(Symmetric(L)); mu = F \ h; Cv = inv(F)
            for t in 1:T; M1[:, t] = mu[rows(t)]; end
            lp = zeros(2, T)
            for t in 1:(T-1), k in 1:2
                Ak, bk, _ = trans(k, t); Pk = Pinv[k]
                r = M1[:, t+1] - Ak*M1[:, t] - bk
                Vtt, Vt1, Vc = Cv[rows(t), rows(t)], Cv[rows(t+1), rows(t+1)], Cv[rows(t), rows(t+1)]
                lp[k, t+1] = -0.5*(ld[k] + tr(Pk*(Vt1 - Ak*Vc - Vc'*Ak' + Ak*Vtt*Ak' + r*r')))
            end
            la = fill(-Inf, 2, T); lb = zeros(2, T); la[:, 1] = logpi
            for t in 2:T, k in 1:2
                la[k, t] = lp[k, t] + logsumexp2(la[1,t-1]+logA[1,k], la[2,t-1]+logA[2,k])
            end
            for t in (T-1):-1:1, k in 1:2
                lb[k, t] = logsumexp2(logA[k,1]+lp[1,t+1]+lb[1,t+1],
                                      logA[k,2]+lp[2,t+1]+lb[2,t+1])
            end
            gn = la .+ lb
            for t in 1:T
                mm = maximum(gn[:, t]); e = exp.(gn[:, t] .- mm); g[:, t] = e ./ sum(e)
            end
        end
        z = Zs[i]
        for t in 2:T
            k = argmax(g[:, t]); nk[z[t]] += 1; tp[z[t]] += (k == z[t])
            push!(mass, g[z[t], t])
        end
        hit = findfirst(t -> g[1, t] > 0.5, 2:T)
        push!(onset, hit === nothing ? T - sws[i] : abs(hit - sws[i]))
    end
    println()
    println("γ at the GENERATING parameters, closed-loop parameterisation:")
    @printf("  balanced accuracy   %.3f\n", 0.5*(tp[1]/nk[1] + tp[2]/nk[2]))
    @printf("  mean posterior mass %.3f\n", mean(mass))
    @printf("  mean onset error    %.2f timesteps\n", mean(onset))
    println()
    println("For contrast, README.md / slds.jl measured γ at the generating parameters")
    println("in the mixed-coordinate model as a function of the costate innovation —")
    println("a parameter the closed-loop model does not have:")
    println("  Σ_λλ = 1e-4  -> 0.52   (the single-system optimum)")
    println("  Σ_λλ = 1e-2  -> 0.74")
    println("  Σ_λλ = 2e-2  -> 0.81-0.83,  onset 5.5-6.3   (the switching default)")
    println("The margin is within configuration differences. The point is that the")
    println("column of Σ_λλ values does not exist, so γ and the cost cannot conflict.")
end

# =========================================================================
# fit: end-to-end recovery, swept over the initial cost scale
# =========================================================================
ntri(n) = n*(n+1) ÷ 2
lower_tri!(L, v) = (k = 1; for j in 1:size(L,1), i in j:size(L,1); L[i,j] = v[k]; k += 1 end; L)

section_fit = function ()
    section("fit — end-to-end recovery, swept over the initial cost scale")
    st = setup(); Qs = true_costs()
    sx, su, sy = 0.004, 0.25, 0.006
    Ys, tg = gen(MersenneTwister(20240922), st, Qs; ntrial = 400,
                 sig_x = sx, sig_u = su, sig_y = sy)
    n, k = st.n, ntri(st.n); ntg = length(st.refs); np = 2k + 2*ntg + 3

    function unpack(v)
        L1 = lower_tri!(zeros(n, n), view(v, 1:k)); L2 = lower_tri!(zeros(n, n), view(v, k+1:2k))
        Q = [Matrix(Symmetric(L1*L1')), Matrix(Symmetric(L2*L2'))]
        off = 2k
        refs = [[v[off+2j-1], v[off+2j], 0.0, 0.0] for j in 1:ntg]
        off += 2*ntg
        ex, eu, ey = exp(v[off+1]), exp(v[off+2]), exp(v[off+3])
        return Q, refs, ex^2*Matrix{Float64}(I,n,n) + eu^2*(st.B*st.B'), ey^2
    end
    function start(q0)
        v = zeros(np); d = sqrt(q0)*10
        for o in (0, k); j = 1
            for c in 1:n, r in c:n; v[o+j] = (r == c ? d : 0.0); j += 1 end
        end
        v[2k+2*ntg+1] = log(0.01); v[2k+2*ntg+2] = log(0.5); v[2k+2*ntg+3] = log(0.01)
        return v
    end
    packL(L) = (v = zeros(k); j = 1; for c in 1:n, r in c:n; v[j] = L[r, c]; j += 1 end; v)
    vtruth = vcat(packL(cholesky(Symmetric(Qs[1])).L), packL(cholesky(Symmetric(Qs[2])).L),
                  vcat([r[1:2] for r in st.refs]...), [log(sx), log(su), log(sy)])
    cl(Q, A = st.A) = A - st.B*riccati(A, st.B, st.R, Q, st.sched, zeros(n))[1][1]
    rt = vcat([r[1:2] for r in st.refs]...)
    per = 1/(length(Ys)*st.T)

    @printf("%d parameters fitted (known plant: 2 cost matrices, %d references, 3 noise scalars)\n",
            np, ntg)
    @printf("truth: tr(Qrun) = %.1f, tr(Qterm) = %.1f, nll/step = %.6f\n",
            tr(Qs[1]), tr(Qs[2]),
            -per*loglik(st, Qs, st.refs, sx^2*Matrix(I,n,n) + su^2*(st.B*st.B'), sy^2, Ys, tg))
    f = function (v)
        Q, refs, Sw, sy2 = unpack(v)
        try; return -per*loglik(st, Q, refs, Sw, sy2, Ys, tg); catch; return 1e8 end
    end
    sc(x) = @sprintf("%.3g", x)
    pair(a, b) = string(sc(a), "/", round(b, digits=3))
    @printf("\n%-22s %-16s %-16s %-10s %-14s %-12s %s\n", "start", "Qrun rmse/corr",
            "Qterm rmse/corr", "scale err", "Gref rmse/corr", "closed-loop", "nll/step")
    runs = vcat([("q0 = $(q0)", start(q0)) for q0 in [0.05, 1.0, 100.0, 10_000.0]],
                [("warm start at truth", copy(vtruth))])
    for (label, v0) in runs
        res = Optim.optimize(f, v0, LBFGS(linesearch = BackTracking()),
                             Optim.Options(iterations = 400, g_abstol = 1e-10))
        Q, refs, _, _ = unpack(Optim.minimizer(res))
        rf = vcat([r[1:2] for r in refs]...)
        @printf("%-22s %-16s %-16s %-10s %-14s %-12.4f %.6f\n", label,
                pair(relrmse(utri(Q[1]), utri(Qs[1])), pearson(utri(Q[1]), utri(Qs[1]))),
                pair(relrmse(utri(Q[2]), utri(Qs[2])), pearson(utri(Q[2]), utri(Qs[2]))),
                sc(abs(tr(Q[1])/tr(Qs[1]) - 1)),
                pair(relrmse(rf, rt), pearson(rf, rt)),
                relrmse(cl(Q), cl(Qs)), Optim.minimum(res))
        flush(stdout)
    end
    println()
    println("Read three things here, in order of importance.")
    println(" 1. The warm start at the truth STAYS there: nll moves by 9e-4/step and every")
    println("    block holds. The closed-loop likelihood is correctly specified for this")
    println("    agent, so the truth is (to sampling noise) a stationary point. README.md")
    println("    reports the opposite for the mixed-coordinate fit — a warm start at the")
    println("    truth ends at Qc 0.394, and the ELBO ranks the cold start best.")
    println(" 2. A start near the right scale recovers the cost AND the reference at once")
    println("    (Gref 0.025/1.00), under a position-only emission that never sees velocity.")
    println("    There is no reference/cost trade to make.")
    println(" 3. A start far from the right scale FAILS, and that is an optimizer problem,")
    println("    not an identification one (§profile shows the scale has real curvature).")
    println("    Q_k and Gref enter the feedforward only through the product Q_k Gref, so a")
    println("    badly scaled cost can be traded against a badly scaled reference along a")
    println("    long valley. Initialize from an estimated closed loop (family C), or")
    println("    separate the cost's scale from its shape in the parameterization.")
    println("Q_term is ~1.0 in every cold-start row, as §terminal predicts.")
end

for (name, f) in (("geometry", section_geometry), ("gauge", section_gauge),
                  ("profile", section_profile), ("terminal", section_terminal),
                  ("reference", section_reference), ("switching", section_switching),
                  ("fit", section_fit))
    want(name) && f()
end
println()
