# Sampling profile of a Poisson SLDS fit.
# Run as:  julia --project=. -t 4 docs/dev/slds_profile.jl

using StateSpaceDynamics, Random, LinearAlgebra, Profile
using StateSpaceDynamics: SLDS, LinearDynamicalSystem, GaussianStateModel,
    PoissonObservationModel
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
D,N,TS,NTRIALS,K = 3,40,200,16,3
z,x,y = rand(MersenneTwister(42), mk_slds(K,D,N;seed=100), fill(TS,NTRIALS))
m = mk_slds(K,D,N;seed=7)
fit!(deepcopy(m), y; max_iter=2, progress=false)
Profile.clear(); Profile.init(n=10^7, delay=0.0005)
@profile fit!(deepcopy(m), y; max_iter=20, progress=false)
open(joinpath(@__DIR__, "prof.txt"),"w") do io
    Profile.print(io; format=:flat, sortedby=:count, mincount=40, maxdepth=200)
end
println("profile written")
