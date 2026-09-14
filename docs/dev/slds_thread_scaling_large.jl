# Larger-scale PLDS vs Poisson-SLDS thread scaling (N=150, T=500, 32 trials).
# Run as:  julia --project=. -t N docs/dev/slds_thread_scaling_large.jl

using StateSpaceDynamics, Random, LinearAlgebra
using StateSpaceDynamics: SLDS, LinearDynamicalSystem, GaussianStateModel, PoissonObservationModel
BLAS.set_num_threads(1)
function stable_A(rng,D); A=randn(rng,D,D); F=eigen(A); real(0.9*A/maximum(abs,F.values)); end
function poisson_lds(D,N;seed=0)
    rng=MersenneTwister(seed)
    LinearDynamicalSystem(; state_model=GaussianStateModel(;A=stable_A(rng,D),Q=Matrix(0.1I(D)),
        b=zeros(D),x0=zeros(D),P0=Matrix(1.0I(D))),
        obs_model=PoissonObservationModel(;C=0.3*randn(rng,N,D),d=0.1*randn(rng,N)),
        latent_dim=D,obs_dim=N,fit_bool=fill(true,6))
end
function mk_slds(K,D,N;seed=0)
    A=fill(0.05/(K-1),K,K); for k in 1:K; A[k,k]=0.95; end
    SLDS(;A=A,πₖ=fill(1/K,K),LDSs=[poisson_lds(D,N;seed=seed+k) for k in 1:K])
end
D,N,TS,NTRIALS,K = 6,150,500,32,3
z,x,y = rand(MersenneTwister(42), mk_slds(K,D,N;seed=100), fill(TS,NTRIALS))
plds = poisson_lds(D,N;seed=7); pslds = mk_slds(K,D,N;seed=7)
fit!(deepcopy(plds), y; max_iter=1, progress=false)
fit!(deepcopy(pslds), y; max_iter=1, progress=false)
ITERS=5
println("threads=", Threads.nthreads(), " | K=$K D=$D N=$N T=$TS ntrials=$NTRIALS iters=$ITERS")
tp = @elapsed fit!(deepcopy(plds), y; max_iter=ITERS, progress=false)
ts = @elapsed fit!(deepcopy(pslds), y; max_iter=ITERS, progress=false)
println("  PLDS    ", round(tp,digits=2), " s")
println("  P-SLDS  ", round(ts,digits=2), " s   (ratio ", round(ts/tp,digits=2), ")")
