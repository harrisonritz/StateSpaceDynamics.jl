# Poisson SLDS at the reported production workload shape.
# Run as:  julia --project=. -t N docs/dev/slds_workload_benchmark.jl
#
# Defaults to a reduced trial count so it finishes in a minute; override with
# SLDS_BENCH_NTRIALS / SLDS_BENCH_ITERS for the full 3000-trial case.
using StateSpaceDynamics, Random, LinearAlgebra, Printf
using StateSpaceDynamics: SLDS, LinearDynamicalSystem, GaussianStateModel,
    PoissonObservationModel
BLAS.set_num_threads(1)

stable_A(rng, D) = (A = randn(rng, D, D); real(0.9 * A / maximum(abs, eigen(A).values)))

function poisson_lds(D, N; seed=0)
    rng = MersenneTwister(seed)
    LinearDynamicalSystem(;
        state_model=GaussianStateModel(; A=stable_A(rng, D), Q=Matrix(0.1I(D)),
                                       b=zeros(D), x0=zeros(D), P0=Matrix(1.0I(D))),
        obs_model=PoissonObservationModel(; C=0.25 * randn(rng, N, D), d=0.1 * randn(rng, N)),
        latent_dim=D, obs_dim=N, fit_bool=fill(true, 6))
end

function mk_slds(K, D, N; seed=0)
    A = fill(0.05 / (K - 1), K, K)
    for k in 1:K
        A[k, k] = 0.95
    end
    SLDS(; A=A, πₖ=fill(1 / K, K), LDSs=[poisson_lds(D, N; seed=seed + k) for k in 1:K])
end

const NTRIALS = parse(Int, get(ENV, "SLDS_BENCH_NTRIALS", "300"))
const ITERS = parse(Int, get(ENV, "SLDS_BENCH_ITERS", "5"))
const D, N, TS, K = 10, 150, 100, 3

z, x, y = rand(MersenneTwister(42), mk_slds(K, D, N; seed=100), fill(TS, NTRIALS))
m = mk_slds(K, D, N; seed=7)

fit!(deepcopy(m), y; max_iter=1, progress=false)   # compile

@printf("threads=%d | K=%d D=%d N=%d T=%d ntrials=%d iters=%d\n",
        Threads.nthreads(), K, D, N, TS, NTRIALS, ITERS)
for np in (1, Threads.nthreads())
    t = @elapsed fit!(deepcopy(m), y; max_iter=ITERS, progress=false,
                      rng=MersenneTwister(1), npool=np)
    @printf("  npool=%-2d  %.2f s\n", np, t)
end
