# Checks a batched + precomputed-lognorm prototype of the SLDS joint log-likelihood
# against the current one, for agreement and speed.
# Run as:  julia --project=. -t 1 docs/dev/slds_loglik_kernel.jl

using StateSpaceDynamics, Random, LinearAlgebra, SpecialFunctions
using StateSpaceDynamics: SLDS, LinearDynamicalSystem, GaussianStateModel,
    PoissonObservationModel, SLDSSmoothWorkspace, joint_loglikelihood!,
    _poisson_lognorm_t, state_loglikelihood!
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

# Prototype: SLDS joint_loglikelihood! with PLDS's batched kernel + precomputed lognorm_t
function jll_fast!(ws, slds, x, y, w, lognorm_t)
    Tsteps = size(y,2); K = length(slds.LDSs); Tt = eltype(x)
    ll_vec = ws.opt.ll_vec
    @views fill!(ll_vec[1:Tsteps], zero(Tt))
    η = ws.opt.temp_dy; dx = ws.opt.temp_dx; tmp = ws.opt.temp_solve_Q
    @views for k in 1:K
        lds = slds.LDSs[k]; cc = ws.consts[k]
        C = lds.obs_model.C; d = lds.obs_model.d
        for t in 1:Tsteps
            mul!(η, C, x[:,t]); @. η = η + d
            e = dot(y[:,t], η) - sum(exp, η) - lognorm_t[t]
            e += state_loglikelihood!(cc, dx, tmp, lds, x, t, nothing)
            ll_vec[t] += w[k,t] * e
        end
    end
    return view(ll_vec, 1:Tsteps)
end

D,N,TS,K = 3,40,200,3
slds = mk_slds(K,D,N;seed=7)
z,x_,y_ = rand(MersenneTwister(42), slds, TS)
y = y_; x = x_
w = fill(1.0/K, K, TS)
ws = SLDSSmoothWorkspace(Float64, slds, TS)
ln = _poisson_lognorm_t(y)

a = copy(joint_loglikelihood!(ws, slds, x, y, w))
b = copy(jll_fast!(ws, slds, x, y, w, ln))
println("max |Δ| = ", maximum(abs, a .- b))

bench(f, n=2000) = (f(); minimum(@elapsed(f()) for _ in 1:n))
t_cur  = bench(() -> joint_loglikelihood!(ws, slds, x, y, w))
t_fast = bench(() -> jll_fast!(ws, slds, x, y, w, ln))
println("current  joint_loglikelihood!: ", round(t_cur*1e6, digits=1), " µs")
println("batched+lognorm prototype    : ", round(t_fast*1e6, digits=1), " µs")
println("kernel speedup = ", round(t_cur/t_fast, digits=2), "x")
