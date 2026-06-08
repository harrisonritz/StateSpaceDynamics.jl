using StateSpaceDynamics, LinearAlgebra, Random, SparseArrays, BenchmarkTools
using Optim, LineSearches

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 3.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5

function make_poisson_lds(D, p; seed=42)
    rng = MersenneTwister(seed)
    A = 0.9 .* StateSpaceDynamics.random_rotation_matrix(D, rng)
    Q = Matrix(0.1 * I(D))
    x0 = zeros(D)
    P0 = Matrix(0.1 * I(D))
    C = 0.3 .* randn(rng, p, D)
    d = log.(0.5 .+ rand(rng, p))
    b = zeros(D)
    sm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
    om = PoissonObservationModel(; C=C, d=d)
    return LinearDynamicalSystem(sm, om)
end

function bench_one_smooth(D, p, T_t)
    lds = make_poisson_lds(D, p)
    rng = MersenneTwister(123)
    _, y_multi = StateSpaceDynamics.rand(rng, lds, fill(T_t, 1))
    y = y_multi[1]

    # ---- Hand-rolled ----
    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, [T_t])
    sws = StateSpaceDynamics.SmoothWorkspace(Float64, D, p, T_t; u_dim=0, d_dim=0)
    for _ in 1:3
        StateSpaceDynamics.smooth!(lds, tfs[1], y, sws)
    end
    b_ours = @benchmark StateSpaceDynamics.smooth!($lds, $(tfs[1]), $y, $sws)

    # ---- Optim Newton with sparse Hessian ----
    function nll(vec_x::AbstractVector{Float64})
        x = reshape(vec_x, D, T_t)
        return -sum(StateSpaceDynamics.loglikelihood(x, lds, y))
    end
    function g!(g::Vector{Float64}, vec_x::Vector{Float64})
        x = reshape(vec_x, D, T_t)
        grad = StateSpaceDynamics.Gradient(lds, y, x)
        g .= vec(-grad)
        return g
    end
    function h!(h::SparseMatrixCSC{Float64,Int}, vec_x::Vector{Float64})
        x = reshape(vec_x, D, T_t)
        H, _, _, _ = StateSpaceDynamics.Hessian(lds, y, x)
        h.nzval .= -H.nzval
        return nothing
    end

    X₀ = zeros(D * T_t)
    initial_f = nll(X₀)
    initial_g = similar(X₀)
    g!(initial_g, X₀)
    _H, _, _, _ = StateSpaceDynamics.Hessian(lds, y, reshape(X₀, D, T_t))
    initial_h = SparseMatrixCSC{Float64,Int}(
        _H.m, _H.n, _H.colptr, _H.rowval, zeros(length(_H.nzval))
    )
    h!(initial_h, X₀)
    td = TwiceDifferentiable(nll, g!, h!, X₀, initial_f, initial_g, initial_h)
    opts = Optim.Options(; g_abstol=1e-6, x_abstol=1e-6, f_abstol=1e-6, iterations=100)
    for _ in 1:2
        optimize(td, copy(X₀), Newton(; linesearch=LineSearches.BackTracking()), opts)
    end
    b_optim = @benchmark optimize(
        $td, copy($X₀), Newton(; linesearch=LineSearches.BackTracking()), $opts
    )

    return (
        ours_ms=round(median(b_ours).time / 1e6; digits=2),
        ours_kb=round(b_ours.memory / 1024; digits=2),
        optim_ms=round(median(b_optim).time / 1e6; digits=2),
        optim_kb=round(b_optim.memory / 1024; digits=1),
    )
end

println(rpad("D", 5), rpad("T", 5), rpad("p", 5),
        rpad("ours_ms", 10), rpad("ours_KB", 10),
        rpad("optim_ms", 10), rpad("optim_KB", 12),
        rpad("speedup", 10))

# Try D values spanning small → typical → large; keep T_t and p reasonable.
for (D, p, T_t) in [
    (3,  5,  200),
    (5,  10, 200),
    (8,  16, 200),
    (16, 32, 200),
    (32, 64, 200),
    (50, 80, 200),
    (80, 100, 200),
]
    r = bench_one_smooth(D, p, T_t)
    println(
        rpad(D, 5), rpad(T_t, 5), rpad(p, 5),
        rpad(r.ours_ms, 10), rpad(r.ours_kb, 10),
        rpad(r.optim_ms, 10), rpad(r.optim_kb, 12),
        rpad(round(r.ours_ms / r.optim_ms; digits=2), 10),
    )
end
