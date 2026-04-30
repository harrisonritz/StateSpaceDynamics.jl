using Pkg
Pkg.activate("benchmarking")

# Load the in-tree source rather than the installed package so that the
# `kalman_filter` branch's new Kalman/RTS E-step is exercised.
include(joinpath(@__DIR__, "..", "src", "StateSpaceDynamics.jl"))
using .StateSpaceDynamics
const SSD = StateSpaceDynamics

using BenchmarkTools
using StableRNGs
using Random
using LinearAlgebra
using Printf
using Statistics
using DataFrames
using CSV
using Plots
using Measures

BLAS.set_num_threads(1)
set_zero_subnormals(true);

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

struct BenchConfig
    latent_dims::Int
    obs_dims::Int
    seq_length::Int
    n_iters::Int
    n_repeats::Int
end

kf_config = BenchConfig(
    4,   # latent_dims
    8,      # obs_dims
    100,         # seq_length
    50,          # n_iters (EM iterations per fit)
    1,         # n_repeats (benchmark samples)
)

const NUM_TRIALS = 100

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60.0

# ----------------------------------------------------------------------
# Parameter + model helpers (local — not dependent on benchmarking/Params.jl,
# which `using StateSpaceDynamics` would import the installed package)
# ----------------------------------------------------------------------

struct LDSParams{T<:Real}
    A::Matrix{T}
    Q::Matrix{T}
    x0::Vector{T}
    P0::Matrix{T}
    C::Matrix{T}
    R::Matrix{T}
    b::Vector{T}
    d::Vector{T}
end

function init_params(rng::AbstractRNG, latent_dim::Int, obs_dim::Int)
    A = 0.99 * SSD.random_rotation_matrix(latent_dim, rng)

    Q = randn(rng, latent_dim, latent_dim)
    Q = Q * Q' + 1e-2I
    

    x0 = randn(rng, latent_dim)
    P0 = randn(rng, latent_dim, latent_dim)
    P0 = P0 * P0' + 1e-2I

    C = randn(rng, obs_dim, latent_dim)
    R = randn(rng, obs_dim, obs_dim)
    R = R * R' + 1e-2I

    b = randn(rng, latent_dim)
    d = randn(rng, obs_dim)

    return LDSParams(A, Q, x0, P0, C, R, b, d)
end

function build_lds(p::LDSParams, kalman::Bool)
    state_model = SSD.GaussianStateModel(; A=p.A, Q=p.Q, x0=p.x0, P0=p.P0, b=p.b)
    obs_model = SSD.GaussianObservationModel(; C=p.C, R=p.R, d=p.d)
    return SSD.LinearDynamicalSystem(
        state_model, obs_model;
        kalman_filter=kalman,
    )
end

# ----------------------------------------------------------------------
# Benchmark loop
# ----------------------------------------------------------------------

name = :kalman
kf =  true
latent_dim = kf_config.latent_dims
obs_dim = kf_config.obs_dims
seq_len = kf_config.seq_length

results = DataFrame(
    method          = Symbol[],
    latent_dim      = Int[],
    obs_dim         = Int[],
    seq_len         = Int[],
    num_trials      = Int[],
    n_iters         = Int[],
    time_sec        = Float64[],
    memory          = Int[],
    allocs          = Int[],
    test_ll_per_obs = Float64[],
)


println("\n→ latent_dim=$latent_dim, obs_dim=$obs_dim, seq_len=$(kf_config.seq_length), num_trials=$NUM_TRIALS")

rng = StableRNG(1234)
params = init_params(rng, latent_dim, obs_dim)

# Both methods fit the same training data and are evaluated on the same
# held-out test set drawn from the true generative model.
ref = build_lds(params, kf)

_, y      = rand(rng, ref; ntrials=NUM_TRIALS, tsteps=kf_config.seq_length)
_, y_test = rand(StableRNG(5678), ref; ntrials=NUM_TRIALS, tsteps=kf_config.seq_length)

# Normalization constant for per-observation log-likelihood
n_obs = obs_dim * kf_config.seq_length * NUM_TRIALS



# %% fit model
print("  $(rpad(string(name), 8)) ")
params0 = init_params(rng, latent_dim, obs_dim)
model = build_lds(params0, kf)

# Warm up / precompile
SSD.fit!(deepcopy(model), y; max_iter=1, tol=1e-6, progress=false)

VSCodeServer.@profview SSD.fit!(deepcopy(model), y; max_iter=500, tol=1e-6, progress=false)


max_iter = kf_config.n_iters
bench = @benchmark SSD.fit!(m, $y;
                            max_iter=$max_iter,
                            tol=1e-6,
                            progress=true) setup=(m = deepcopy($model)) samples=kf_config.n_repeats evals=1


# Fit once outside the benchmark to get final parameters for evaluation.
fitted = deepcopy(model)
SSD.fit!(fitted, y; max_iter=max_iter, tol=1e-8, progress=false)
test_loglik = SSD.filter_loglikelihood(fitted, y_test)



println("\n\n------------------------------------------Original model ------------------------------------------\n")
show(ref)

println("\n\n------------------------------------------Fitted model ------------------------------------------")
show(fitted)

elbos = SSD.fit!(fitted, y; max_iter=10, tol=1e-6, progress=false)
println("\n ELBO: $(round.(elbos .- elbos[1], digits=6)) \n")


println("\n\n------------------------------------------ Recovery ------------------------------------------")

println("\nGenerative model ---------------------")
gen_eigenvals = eigvals(ref.state_model.A)
println("Eigenvalues of A: ", round.(gen_eigenvals, digits=6))
println("Spectral radius of A: ", round.(maximum(abs.(gen_eigenvals)),digits=6))

println("\nFitted model ---------------------")
fitted_eigenvals = eigvals(fitted.state_model.A)
println("Eigenvalues of A: ", round.(fitted_eigenvals, digits=6))
println("Spectral radius of A: ", round.(maximum(abs.(fitted_eigenvals)),digits=6))
println("---------------------\n\n")



display(bench)
@printf("test_ll = %.6f\n", test_loglik)

