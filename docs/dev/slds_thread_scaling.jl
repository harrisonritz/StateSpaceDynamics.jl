# Thread-scaling comparison: LDS / Gaussian-SLDS / PLDS / Poisson-SLDS.
# Run as:  julia --project=. -t N docs/dev/slds_thread_scaling.jl

using StateSpaceDynamics, Random, LinearAlgebra, Statistics
using StateSpaceDynamics: SLDS, LinearDynamicalSystem, GaussianStateModel,
    PoissonObservationModel, GaussianObservationModel

BLAS.set_num_threads(1)

function stable_A(rng, D)
    A = randn(rng, D, D); F = eigen(A)
    return real(0.9 * A / maximum(abs, F.values))
end
function poisson_lds(D, N; seed=0)
    rng = MersenneTwister(seed)
    gsm = GaussianStateModel(; A=stable_A(rng,D), Q=Matrix(0.1I(D)), b=zeros(D),
                             x0=zeros(D), P0=Matrix(1.0I(D)))
    pom = PoissonObservationModel(; C=0.3*randn(rng,N,D), d=0.1*randn(rng,N))
    LinearDynamicalSystem(; state_model=gsm, obs_model=pom, latent_dim=D, obs_dim=N,
                          fit_bool=fill(true,6))
end
function gauss_lds(D, N; seed=0)
    rng = MersenneTwister(seed)
    gsm = GaussianStateModel(; A=stable_A(rng,D), Q=Matrix(0.1I(D)), b=zeros(D),
                             x0=zeros(D), P0=Matrix(1.0I(D)))
    gom = GaussianObservationModel(; C=0.3*randn(rng,N,D), R=Matrix(0.5I(N)), d=zeros(N))
    LinearDynamicalSystem(; state_model=gsm, obs_model=gom, latent_dim=D, obs_dim=N,
                          fit_bool=fill(true,6))
end
function mk_slds(mk, K, D, N; seed=0)
    A = fill(0.05/(K-1), K, K); for k in 1:K; A[k,k]=0.95; end
    SLDS(; A=A, πₖ=fill(1/K,K), LDSs=[mk(D,N;seed=seed+k) for k in 1:K])
end

const D, N, TS, NTRIALS, K = 3, 40, 200, 16, 3
rng = MersenneTwister(42)
z, x, y = rand(rng, mk_slds(poisson_lds, K, D, N; seed=100), fill(TS, NTRIALS))
zg, xg, yg = rand(MersenneTwister(43), mk_slds(gauss_lds, K, D, N; seed=100), fill(TS, NTRIALS))

plds  = poisson_lds(D, N; seed=7)
pslds = mk_slds(poisson_lds, K, D, N; seed=7)
glds  = gauss_lds(D, N; seed=7)
gslds = mk_slds(gauss_lds, K, D, N; seed=7)

for (nm, m, dat) in (("PLDS", plds, y), ("P-SLDS", pslds, y),
                     ("LDS", glds, yg), ("G-SLDS", gslds, yg))
    fit!(deepcopy(m), dat; max_iter=2, progress=false)   # compile
end

ITERS = 10
println("threads=", Threads.nthreads(), " BLAS=", BLAS.get_num_threads(),
        " | K=$K D=$D N=$N T=$TS ntrials=$NTRIALS iters=$ITERS")
for (nm, m, dat) in (("LDS   ", glds, yg), ("G-SLDS", gslds, yg),
                     ("PLDS  ", plds, y),  ("P-SLDS", pslds, y))
    ts = minimum(@elapsed(fit!(deepcopy(m), dat; max_iter=ITERS, progress=false)) for _ in 1:3)
    println("  ", nm, "  ", round(ts, digits=3), " s")
end
