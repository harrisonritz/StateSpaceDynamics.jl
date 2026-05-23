# Same benchmark as `refactor_compare.jl` but adapted to the pre-refactor `main`
# API. Differences from dev_ryan_:
#   * `rand(lds; tsteps, ntrials)` returns 3D arrays (D, T, N) directly
#   * `fit!(lds, y::3D)` (no Vector-of-Matrix path)
#   * `fit!` returns `(elbos, param_diff)` tuple
#   * `PoissonObservationModel(; C, log_d=...)` (was renamed to `d`)
#   * No `kalman_filter` keyword (Kalman path didn't exist on main)
#
# Output schema is identical so the two CSVs diff cleanly.

using StateSpaceDynamics
using BenchmarkTools
using LinearAlgebra
using Random
using Printf

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 2.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5

const SCENARIOS = [
    ("small",          3,  5,  100,  4),
    ("medium",         5, 10,  200,  8),
    ("large",          8, 16,  500, 16),
    ("long_single",    4,  8, 2000,  1),
    ("many_short",     3,  5,   50, 64),
]

const MAX_EM = 20

function make_gaussian_lds(D, p; seed=42)
    rng = MersenneTwister(seed)
    A = StateSpaceDynamics.random_rotation_matrix(D, rng)
    Q = (M = randn(rng, D, D); M*M' + 1e-3 * I)
    x0 = randn(rng, D)
    P0 = (M = randn(rng, D, D); M*M' + 1e-3 * I)
    C = randn(rng, p, D)
    R = (M = randn(rng, p, p); M*M' + 1e-3 * I)
    b = randn(rng, D)
    d = randn(rng, p)
    sm = GaussianStateModel(; A=Matrix(A), Q=Matrix(Q), b=b, x0=x0, P0=Matrix(P0))
    om = GaussianObservationModel(; C=C, R=Matrix(R), d=d)
    return LinearDynamicalSystem(; state_model=sm, obs_model=om,
        latent_dim=D, obs_dim=p, fit_bool=fill(true, 6))
end

function make_poisson_lds(D, p; seed=42)
    rng = MersenneTwister(seed)
    A = 0.9 .* StateSpaceDynamics.random_rotation_matrix(D, rng)
    Q = Matrix(0.1 * I(D))
    x0 = zeros(D)
    P0 = Matrix(0.1 * I(D))
    C = 0.3 .* randn(rng, p, D)
    log_d = log.(0.5 .+ rand(rng, p))
    b = zeros(D)
    sm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
    om = PoissonObservationModel(; C=C, log_d=log_d)
    return LinearDynamicalSystem(; state_model=sm, obs_model=om,
        latent_dim=D, obs_dim=p, fit_bool=fill(true, 6))
end

function bench_gaussian(label, D, p, T, N)
    rng = MersenneTwister(123)
    lds = make_gaussian_lds(D, p)
    _, y = StateSpaceDynamics.rand(rng, lds; tsteps=T, ntrials=N)

    StateSpaceDynamics.fit!(make_gaussian_lds(D, p), y; max_iter=1, progress=false)

    bench = @benchmark StateSpaceDynamics.fit!(
        lds_copy, $y; max_iter=$MAX_EM, tol=0.0, progress=false
    ) setup=(lds_copy = $(make_gaussian_lds)($D, $p))
    return (
        scenario=label, path="TD", D=D, p=p, T=T, N=N,
        time_s=median(bench).time / 1e9,
        mem_mb=bench.memory / 1024^2,
        allocs=bench.allocs,
    )
end

function bench_poisson(label, D, p, T, N)
    rng = MersenneTwister(123)
    lds = make_poisson_lds(D, p)
    _, y = StateSpaceDynamics.rand(rng, lds; tsteps=T, ntrials=N)

    StateSpaceDynamics.fit!(make_poisson_lds(D, p), y; max_iter=1, progress=false)

    bench = @benchmark StateSpaceDynamics.fit!(
        lds_copy, $y; max_iter=$MAX_EM, tol=0.0, progress=false
    ) setup=(lds_copy = $(make_poisson_lds)($D, $p))
    return (
        scenario=label, path="Poisson", D=D, p=p, T=T, N=N,
        time_s=median(bench).time / 1e9,
        mem_mb=bench.memory / 1024^2,
        allocs=bench.allocs,
    )
end

println("scenario,path,D,p,T,N,time_s,mem_mb,allocs")
for (label, D, p, T, N) in SCENARIOS
    @info "Benchmarking $label (D=$D, p=$p, T=$T, N=$N)"
    for f in (bench_gaussian, bench_poisson)
        try
            r = f(label, D, p, T, N)
            @printf("%s,%s,%d,%d,%d,%d,%.6f,%.3f,%d\n",
                    r.scenario, r.path, r.D, r.p, r.T, r.N,
                    r.time_s, r.mem_mb, r.allocs)
            flush(stdout)
        catch e
            @warn "Failed $(f) for $label: $e"
            @printf("%s,%s,%d,%d,%d,%d,NaN,NaN,NaN\n", label, "?", D, p, T, N)
            flush(stdout)
        end
    end
end
