#=
The finite-horizon LQR sweeps of `src/numerics/riccati.jl`: against a direct QP
solve of the control problem, against the package's own `S`-form Riccati
sequence, and — for the reverse pass — against `ForwardDiff` and central
differences, every block, several regimes.
=#

"""A random `(A, B, R, Q, schedule)` problem: a mildly unstable plant, full-rank
actuation, and three cost regimes switching mid-horizon."""
function riccati_problem(rng; n=4, m=2, horizon=12)
    A = Matrix(1.0I, n, n) .+ 0.1 .* randn(rng, n, n)
    B = randn(rng, n, m)
    X = randn(rng, m, m)
    R = Matrix(Symmetric(I + 0.2 .* X * X'))
    Q = [(L = randn(rng, n, n); Matrix(Symmetric(L * L' ./ n))) for _ in 1:3]
    schedule = vcat(fill(1, 6), fill(2, horizon - 7), 3)
    return A, B, R, Q, schedule
end

"""The tracking problem as one QP over the controls, from a fixed `x₁`:
`x_t = a_t + Γ_t u` and a quadratic in `u`. Returns the optimal states and
controls. Independent of every recursion under test."""
function riccati_qp(A, B, R, Q, schedule, r, x1)
    n, m = size(B)
    horizon = length(schedule)
    nu = m * (horizon - 1)
    a = [zeros(n) for _ in 1:horizon]
    Γ = [zeros(n, nu) for _ in 1:horizon]
    a[1] .= x1
    for t in 2:horizon
        a[t] = A * a[t - 1]
        Γ[t] = A * Γ[t - 1]
        Γ[t][:, ((t - 2) * m + 1):((t - 1) * m)] .+= B
    end
    H = kron(Matrix(1.0I, horizon - 1, horizon - 1), R)
    g = zeros(nu)
    for t in 1:horizon
        Qt = Q[schedule[t]]
        rt = r isa AbstractVector ? r : r[:, t]
        H .+= Γ[t]' * Qt * Γ[t]
        g .+= Γ[t]' * Qt * (a[t] .- rt)
    end
    u = -(Symmetric(H) \ g)
    X = reduce(hcat, [a[t] .+ Γ[t] * u for t in 1:horizon])
    return X, reshape(u, m, horizon - 1)
end

"""Roll the sweep's policy `u_t = −K_t x_t + k_t` forward, noiselessly."""
function riccati_rollout(ws, A, B, x1)
    horizon = length(ws.P)
    X = zeros(length(x1), horizon)
    U = zeros(size(B, 2), horizon - 1)
    X[:, 1] .= x1
    for t in 1:(horizon - 1)
        U[:, t] = -ws.K[t] * X[:, t] .+ ws.k[:, t]
        X[:, t + 1] = A * X[:, t] .+ B * U[:, t]
    end
    return X, U
end

"""An inverse-LQR model in the `S = B R⁻¹ Bᵀ` form of the same tracking problem:
a terminal factor (so `P_T = Q_{k(T)}`), a reference `r_t = G_ref u_t` read off
the input, and no other affine term."""
function riccati_lqr_model(A, B, R, Qc, horizon; onset, ux_dim)
    n = size(A, 1)
    S = Matrix(Symmetric(B * (R \ B')))
    return LQRStateModel(
        A,
        S,
        Qc,
        Matrix(0.03I, 2n, 2n);
        schedule=cost_schedule(horizon; terminal=true, onset=onset, nregimes=length(Qc)),
        terminal=true,
        Σf=Matrix(0.04I, n, n),
        hf=zeros(n),
        h=zeros(2n),
        Bu=zeros(2n, ux_dim),
        Gref=randn(StableRNG(3), n, ux_dim),
    )
end

"""The cost schedule an `LQRStateModel` actually uses over `horizon` steps,
terminal step included."""
function riccati_model_schedule(sm, horizon)
    return vcat(
        [SSD._regime(sm, t) for t in 1:(horizon - 1)], SSD._terminal_regime(sm, horizon)
    )
end

"""The first guard: `simulate_lqr` with the noise off is the solution of the
tracking QP, and so is a rollout of the new sweeps.

This is the check that caught a sign error in the feedforward of the recovery
scripts under `docs/dev/lqr/`: a recursion carrying a spurious `2KᵀRk` term
rolls out a plausible-looking trajectory that costs 16% more than the optimum.
A per-step reference exercises the feedforward harder than a constant one.
"""
function test_riccati_simulate_lqr_matches_qp()
    rng = StableRNG(101)
    n, m, horizon, ux_dim = 3, 2, 16, 2
    A = [0.97 0.10 0.0; -0.08 0.95 0.05; 0.0 0.02 1.01]
    B = randn(rng, n, m)
    R = [1.0 0.2; 0.2 0.7]
    # Before the onset, after it, and the terminal cost.
    Qc = [
        Matrix(Diagonal([0.3, 0.2, 0.1])),
        [1.5 0.1 0.0; 0.1 1.2 0.0; 0.0 0.0 0.9],
        Matrix(Diagonal([4.0, 3.0, 2.0])),
    ]
    sm = riccati_lqr_model(A, B, R, Qc, horizon; onset=9, ux_dim=ux_dim)
    ux = randn(rng, ux_dim, horizon)
    r = sm.Gref * ux
    schedule = riccati_model_schedule(sm, horizon)
    @test sort(unique(schedule)) == 1:3       # every regime is exercised
    x1 = [0.4, -0.3, 0.2]

    Xqp, Uqp = riccati_qp(A, B, R, Qc, schedule, r, x1)
    z = simulate_lqr(rng, sm, horizon; x1=x1, process_noise=false, ux=ux)
    @test maximum(abs, z[1:n, :] .- Xqp) < 1e-10

    ws = SSD.RiccatiSweep{Float64}(n, m, horizon)
    SSD.riccati_gain!(ws, A, B, R, Qc, schedule)
    q = SSD.tracking_cost!(zeros(n, horizon), Qc, schedule, r)
    SSD.affine_sweep!(ws, B, q)
    X, U = riccati_rollout(ws, A, B, x1)
    @test maximum(abs, X .- Xqp) < 1e-10
    @test maximum(abs, U .- Uqp) < 1e-10

    # A constant reference is the vector form of the same thing.
    q1 = SSD.tracking_cost!(zeros(n, horizon), Qc, schedule, r[:, 1])
    q2 = SSD.tracking_cost!(zeros(n, horizon), Qc, schedule, repeat(r[:, 1], 1, horizon))
    @test q1 == q2
    return nothing
end

"""The sweeps are the package's `S`-form Riccati sequence, written where `B` and
`R` are separately available: `P_t` and the costate offset agree, the closed
loop is `Φ_t = W_{t+1} A`, and the feedforward enters the state as
`B k_t = −W_{t+1} S g_{t+1}` — the push-through identity
`B G_t⁻¹ Bᵀ = (I + S P_{t+1})⁻¹ S`."""
function test_riccati_matches_lqr_riccati_sequence()
    rng = StableRNG(102)
    n, m, horizon, ux_dim = 3, 2, 14, 1
    A = Matrix(1.0I, n, n) .+ 0.05 .* randn(rng, n, n)
    B = randn(rng, n, m)
    R = [0.8 0.1; 0.1 0.6]
    Qc = [Matrix(0.2I, n, n), [1.1 0.2 0.0; 0.2 0.9 0.1; 0.0 0.1 0.7], Matrix(2.0I, n, n)]
    sm = riccati_lqr_model(A, B, R, Qc, horizon; onset=5, ux_dim=ux_dim)
    ux = randn(rng, ux_dim, horizon)
    schedule = riccati_model_schedule(sm, horizon)

    P, g, W = lqr_riccati_sequence(sm, horizon; ux=ux)
    ws = SSD.RiccatiSweep{Float64}(n, m, horizon)
    SSD.riccati_gain!(ws, A, B, R, Qc, schedule)
    SSD.affine_sweep!(
        ws, B, SSD.tracking_cost!(zeros(n, horizon), Qc, schedule, sm.Gref * ux)
    )
    S = sm.S
    for t in 1:horizon
        @test ws.P[t] ≈ P[t] rtol = 1e-10
        @test ws.b[:, t] ≈ g[t] rtol = 1e-10 atol = 1e-12
        @test issymmetric(ws.P[t])
    end
    for t in 1:(horizon - 1)
        @test ws.Φ[t] ≈ W[t + 1] * A rtol = 1e-10
        @test B * ws.k[:, t] ≈ -W[t + 1] * S * g[t + 1] rtol = 1e-10 atol = 1e-12
    end
    return nothing
end

"""The sensitivities a test loss seeds: every output of both sweeps at every
step, so that each path through the reverse pass is exercised."""
function riccati_seeds(rng, n, m, horizon)
    return (;
        Φbar=[randn(rng, n, n) for _ in 1:(horizon - 1)],
        kbar=randn(rng, m, horizon - 1),
        Pbar=[randn(rng, n, n) for _ in 1:horizon],
        bbar=randn(rng, n, horizon),
    )
end

"""The loss the reverse pass is checked on: `Σ ⟨seed, output⟩` over every output
the seeds name, with the linear cost that of tracking a constant reference."""
function riccati_loss(A, B, R, Q, schedule, r, seeds; affine=true)
    T = promote_type(eltype(A), eltype(B), eltype(R), eltype(r), eltype(Q[1]))
    n, m = size(B)
    horizon = length(schedule)
    ws = SSD.RiccatiSweep{T}(n, m, horizon)
    SSD.riccati_gain!(ws, A, B, R, Q, schedule)
    L = sum(dot(seeds.Φbar[t], ws.Φ[t]) for t in 1:(horizon - 1))
    L += sum(dot(seeds.Pbar[t], ws.P[t]) for t in 1:horizon)
    if affine
        SSD.affine_sweep!(ws, B, SSD.tracking_cost!(zeros(T, n, horizon), Q, schedule, r))
        L += dot(seeds.kbar, ws.k) + dot(seeds.bbar, ws.b)
    end
    return L
end

# Symmetric parameters are packed by their upper triangle.
function riccati_pack(A, B, R, Q, r)
    ut(M) = [M[i, j] for j in axes(M, 2) for i in 1:j]
    return vcat(vec(A), vec(B), ut(R), reduce(vcat, ut.(Q)), r)
end

function riccati_unpack(θ, n, m, nq)
    T = eltype(θ)
    pos = Ref(0)
    take(len) = (v = θ[(pos[] + 1):(pos[] + len)]; pos[] += len; v)
    function sym(k)
        v = take(k * (k + 1) ÷ 2)
        M = zeros(T, k, k)
        c = 0
        for j in 1:k, i in 1:j
            c += 1
            M[i, j] = M[j, i] = v[c]
        end
        return M
    end
    A = reshape(take(n * n), n, n)
    B = reshape(take(n * m), n, m)
    R = sym(m)
    Q = [sym(n) for _ in 1:nq]
    r = take(n)
    return A, B, R, Q, r
end

"""The gradient of a symmetric parameter in upper-triangle coordinates: an
off-diagonal coordinate moves two entries, so it collects both."""
riccati_pack_grad(A, B, R, Q, r) =
    riccati_pack(A, B, 2 .* R .- Diagonal(R), [2 .* Qk .- Diagonal(Qk) for Qk in Q], r)

"""`riccati_adjoint!` is the gradient of the sweeps: exactly (to roundoff)
against `ForwardDiff` through the same kernels run on dual numbers, and against
central differences of the plain kernels along random directions — the second
check does not trust that the kernels differentiate correctly under AD.
Covered twice: with every sensitivity seeded, and with the gain half alone
(no affine sweep at all), which is its own branch."""
function test_riccati_adjoint()
    rng = StableRNG(103)
    n, m, horizon = 4, 2, 12
    A, B, R, Q, schedule = riccati_problem(rng; n=n, m=m, horizon=horizon)
    r = randn(rng, n)
    seeds = riccati_seeds(rng, n, m, horizon)
    nq = length(Q)
    θ = riccati_pack(A, B, R, Q, r)

    for affine in (true, false)
        L = θ -> begin
            Aθ, Bθ, Rθ, Qθ, rθ = riccati_unpack(θ, n, m, nq)
            riccati_loss(Aθ, Bθ, Rθ, Qθ, schedule, rθ, seeds; affine=affine)
        end
        g_ad = ForwardDiff.gradient(L, θ)

        ws = SSD.RiccatiSweep{Float64}(n, m, horizon)
        SSD.riccati_gain!(ws, A, B, R, Q, schedule)
        q = SSD.tracking_cost!(zeros(n, horizon), Q, schedule, r)
        affine && SSD.affine_sweep!(ws, B, q)
        grad = SSD.RiccatiGradient{Float64}(n, m, nq, horizon)
        SSD.riccati_adjoint!(
            grad,
            ws,
            A,
            B,
            R,
            Q,
            schedule,
            seeds.Φbar,
            affine ? seeds.kbar : nothing;
            Pbar=seeds.Pbar,
            bbar=affine ? seeds.bbar : nothing,
        )
        Qbar = deepcopy(grad.Q)
        rbar = zeros(n)
        SSD.tracking_cost_adjoint!(Qbar, rbar, grad.q, Q, schedule, r)
        asym(X) = maximum(abs, X .- X') / maximum(abs, X)
        @test all(Qk -> asym(Qk) < 1e-13, Qbar) && asym(grad.R) < 1e-13
        g = riccati_pack_grad(grad.A, grad.B, grad.R, Qbar, rbar)
        @test maximum(abs, g .- g_ad) / maximum(abs, g_ad) < 1e-11
        affine || @test all(iszero, rbar)

        # Central differences of the Float64 kernels, along random directions.
        for _ in 1:4
            d = normalize(randn(rng, length(θ)))
            h = 1e-6
            fd = (L(θ .+ h .* d) - L(θ .- h .* d)) / (2h)
            @test fd ≈ dot(g, d) rtol = 1e-6
        end
    end
    return nothing
end

"""The adjoint accumulates rather than overwrites — a caller sums it over
designs — and `_zero_gradient!` resets it."""
function test_riccati_adjoint_accumulates()
    rng = StableRNG(104)
    n, m, horizon = 3, 1, 8
    A, B, R, Q, schedule = riccati_problem(rng; n=n, m=m, horizon=horizon)
    seeds = riccati_seeds(rng, n, m, horizon)
    ws = SSD.RiccatiSweep{Float64}(n, m, horizon)
    SSD.riccati_gain!(ws, A, B, R, Q, schedule)
    SSD.affine_sweep!(ws, B, randn(rng, n, horizon))
    grad = SSD.RiccatiGradient{Float64}(n, m, length(Q), horizon)
    function run!()
        return SSD.riccati_adjoint!(
            grad,
            ws,
            A,
            B,
            R,
            Q,
            schedule,
            seeds.Φbar,
            seeds.kbar;
            Pbar=seeds.Pbar,
            bbar=seeds.bbar,
        )
    end
    run!()
    once = deepcopy(grad)
    run!()
    @test grad.A ≈ 2 .* once.A
    @test grad.B ≈ 2 .* once.B
    @test grad.R ≈ 2 .* once.R
    @test all(grad.Q[k] ≈ 2 .* once.Q[k] for k in eachindex(Q))
    @test grad.q ≈ 2 .* once.q
    SSD._zero_gradient!(grad)
    @test all(iszero, grad.A) && all(iszero, grad.B) && all(iszero, grad.R)
    @test all(all(iszero, Qk) for Qk in grad.Q) && all(iszero, grad.q)
    return nothing
end

"""Sweeping a longer horizon costs more arithmetic, not more allocation: the
kernels run on the workspace's buffers."""
function test_riccati_preallocated()
    rng = StableRNG(105)
    n, m = 4, 2
    A, B, R, Q, _ = riccati_problem(rng; n=n, m=m, horizon=12)
    function bytes(horizon)
        schedule = vcat(fill(1, horizon - 1), 3)
        ws = SSD.RiccatiSweep{Float64}(n, m, horizon)
        q = randn(rng, n, horizon)
        seeds = riccati_seeds(rng, n, m, horizon)
        grad = SSD.RiccatiGradient{Float64}(n, m, length(Q), horizon)
        function sweep!()
            SSD.riccati_gain!(ws, A, B, R, Q, schedule)
            SSD.affine_sweep!(ws, B, q)
            return SSD.riccati_adjoint!(
                grad,
                ws,
                A,
                B,
                R,
                Q,
                schedule,
                seeds.Φbar,
                seeds.kbar;
                Pbar=seeds.Pbar,
                bbar=seeds.bbar,
            )
        end
        sweep!()
        return @allocated sweep!()
    end
    @test bytes(200) == bytes(20)
    return nothing
end

function test_riccati_errors()
    rng = StableRNG(106)
    n, m, horizon = 3, 2, 9
    A, B, R, Q, schedule = riccati_problem(rng; n=n, m=m, horizon=horizon)
    @test_throws ArgumentError SSD.RiccatiSweep{Float64}(n, m, 1)
    @test_throws ArgumentError SSD.RiccatiSweep{Float64}(0, m, horizon)
    ws = SSD.RiccatiSweep{Float64}(n, m, horizon)
    @test_throws DimensionMismatch SSD.riccati_gain!(ws, A, B[:, 1:1], R, Q, schedule)
    @test_throws DimensionMismatch SSD.riccati_gain!(ws, A, B, R, Q, schedule[1:(end - 1)])
    @test_throws ArgumentError SSD.riccati_gain!(ws, A, B, R, Q, fill(4, horizon))
    @test_throws DimensionMismatch SSD.affine_sweep!(ws, B, zeros(n, horizon - 1))
    # A control cost that is not positive definite is a point out of reach,
    # reported the way the rejectable-step machinery recognizes.
    @test_throws PosDefException SSD.riccati_gain!(ws, A, B, -R, Q, schedule)
    SSD.riccati_gain!(ws, A, B, R, Q, schedule)
    grad = SSD.RiccatiGradient{Float64}(n, m, length(Q), horizon)
    @test_throws DimensionMismatch SSD.riccati_adjoint!(
        grad, ws, A, B, R, Q, schedule, nothing, zeros(m, horizon)
    )
    return nothing
end
