if pwd() != @__DIR__
    using Pkg
    Pkg.activate("benchmarking")
end

using StateSpaceDynamics
using LinearAlgebra
using Random
using BenchmarkTools

# Define a simple SSM for testing allocations with
function create_test_lds(latent_dim::Int, obs_dim::Int, seq_length::Int)
    A = random_rotation_matrix(latent_dim)
    C = randn(obs_dim, latent_dim)
    Q = Matrix(I(latent_dim) * 0.1)
    R = Matrix(I(obs_dim) * 0.5)
    b = zeros(latent_dim)
    d = zeros(obs_dim)
    x0_mean = zeros(latent_dim)
    x0_cov = Matrix(I(latent_dim) * 0.1)

    state_model = GaussianStateModel(A, Q, b, x0_mean, x0_cov)
    obs_model = GaussianObservationModel(C, R, d)

    lds = LinearDynamicalSystem(state_model, obs_model)

    # Generate a random sequence (single trial)
    rng = MersenneTwister(42)
    x, y = rand(lds, seq_length)

    return lds, x, y
end

println("="^60)
println("Allocation Profiling for Gaussian LDS fit!")
println("="^60)

# Create test model
lds, x, y = create_test_lds(64, 10, 100)
model = LinearDynamicalSystem(lds.state_model, lds.obs_model)

# Warm up
println("\n[Warming up...]")
fit!(deepcopy(model), y; max_iter=1, progress=false)

# Run the full benchmark
println("\n[Full benchmark: fit! with max_iter=10]")
result = @benchmark fit!(deepcopy($model), $y; max_iter=10, progress=false)
display(result)

println("\n\n" * "="^60)
println("Per-component breakdown:")
println("="^60)

# Now let's profile individual components
T = Float64
latent_dim = 64
obs_dim = 10
tsteps = size(y, 2)

# Wrap the single-trial Matrix as a 1-element Vector{Matrix} for the multi-trial API.
y_multi = [y]

# Create the workspace and FilterSmooth objects
tfs = StateSpaceDynamics.initialize_FilterSmooth(model, [tsteps])
sws_pool = [StateSpaceDynamics.SmoothWorkspace(T, latent_dim, obs_dim, tsteps) for _ in 1:Threads.nthreads()]
sws = sws_pool[1]

# Profile smooth!
println("\n[smooth! - single trial]")
fs = tfs[1]
result_smooth = @benchmark StateSpaceDynamics.smooth!($model, $fs, $y, $sws)
display(result_smooth)

# Profile sufficient_statistics!
println("\n[sufficient_statistics! - single trial]")
StateSpaceDynamics.smooth!(model, fs, y, sws)
result_ss = @benchmark StateSpaceDynamics.sufficient_statistics!($fs)
display(result_ss)

# Profile estep!
println("\n[estep! - full]")
result_estep = @benchmark StateSpaceDynamics.estep!($model, $tfs, $y_multi, $sws_pool)
display(result_estep)

# Profile mstep!
println("\n[mstep! - full]")
StateSpaceDynamics.estep!(model, tfs, y_multi, sws_pool)
result_mstep = @benchmark StateSpaceDynamics.mstep!($model, $tfs, $y_multi, $sws)
display(result_mstep)

# Profile calculate_elbo
println("\n[calculate_elbo]")
result_elbo = @benchmark StateSpaceDynamics.calculate_elbo($model, $tfs, $y_multi, $sws_pool)
display(result_elbo)

# Profile Q_state!
println("\n[Q_state]")
StateSpaceDynamics.Q_state!(sws, model, fs.E_z, fs.E_zz, fs.E_zz_prev)
result_qstate = @benchmark StateSpaceDynamics.Q_state!(sws, $model, $fs.E_z, $fs.E_zz, $fs.E_zz_prev)
display(result_qstate)

# Profile Q_obs!
println("\n[Q_obs]")
StateSpaceDynamics.Q_obs!(sws, model, fs.E_z, fs.E_zz, y_trial)
result_qobs = @benchmark StateSpaceDynamics.Q_obs!(sws, $model, $fs.E_z, $fs.E_zz, $y_trial)
display(result_qobs)

# Profile compute_smooth_constants!
println("\n[compute_smooth_constants!]")
result_csc = @benchmark StateSpaceDynamics.compute_smooth_constants!($sws, $model)
display(result_csc)

# Profile Gradient!
println("\n[Gradient!]")
x_mat = reshape(sws.X₀, latent_dim, tsteps)
StateSpaceDynamics.compute_smooth_constants!(sws, model)
result_grad = @benchmark StateSpaceDynamics.Gradient!($sws, $model, $y_trial, $x_mat)
display(result_grad)

# Profile Hessian!
println("\n[Hessian!]")
result_hess = @benchmark StateSpaceDynamics.Hessian!($sws, $model, $y_trial, $x_mat)
display(result_hess)

# Profile block_tridiagonal_inverse_logdet!
println("\n[block_tridiagonal_inverse_logdet!]")
btd = sws.btd
StateSpaceDynamics.Hessian!(sws, model, y_trial, x_mat)
StateSpaceDynamics._negate_blocks!(btd)
result_btil = @benchmark StateSpaceDynamics.block_tridiagonal_inverse_logdet!(
    $(fs.p_smooth), $(fs.p_smooth_tt1),
    $(btd.neg_sub), $(btd.neg_diag), $(btd.neg_super), $btd
)
display(result_btil)

# Profile block_tridiagonal_solve!
println("\n[block_tridiagonal_solve!]")
StateSpaceDynamics.Gradient!(sws, model, y_trial, x_mat)
copyto!(sws.grad_vec, 1, sws.grad_buf, 1, length(sws.grad_vec))
sws.grad_vec .*= -1.0
StateSpaceDynamics.Hessian!(sws, model, y_trial, x_mat)
StateSpaceDynamics._negate_blocks!(btd)
result_solve = @benchmark StateSpaceDynamics.block_tridiagonal_solve!(
    $(sws.X₀), $(btd.neg_sub), $(btd.neg_diag), $(btd.neg_super), $(sws.grad_vec), $btd
)
display(result_solve)

# Profile individual M-step functions
println("\n[update_A_b!]")
result_ab = @benchmark StateSpaceDynamics.update_A_b!($model, $tfs, $sws)
display(result_ab)

println("\n[update_Q!]")
result_q = @benchmark StateSpaceDynamics.update_Q!($model, $tfs, $sws)
display(result_q)

println("\n[update_C_d!]")
result_cd = @benchmark StateSpaceDynamics.update_C_d!($model, $tfs, $y, $sws)
display(result_cd)

println("\n[update_R!]")
result_r = @benchmark StateSpaceDynamics.update_R!($model, $tfs, $y, $sws)
display(result_r)

println("\n" * "="^60)
println("Summary of major allocation sources:")
println("="^60)

println("\nsmooth!: $(BenchmarkTools.prettymemory(result_smooth.memory))")
println("estep!: $(BenchmarkTools.prettymemory(result_estep.memory))")
println("mstep!: $(BenchmarkTools.prettymemory(result_mstep.memory))")
println("calculate_elbo: $(BenchmarkTools.prettymemory(result_elbo.memory))")
println("Q_state: $(BenchmarkTools.prettymemory(result_qstate.memory))")
println("Q_obs: $(BenchmarkTools.prettymemory(result_qobs.memory))")

# Estimate total per iteration
smooth_mem = result_smooth.memory
ss_mem = result_ss.memory
elbo_mem = result_elbo.memory
mstep_mem = result_mstep.memory

per_iter = smooth_mem + ss_mem + elbo_mem + mstep_mem
println("\nEstimated per-iteration: $(BenchmarkTools.prettymemory(per_iter))")
println("Estimated for 10 iterations: $(BenchmarkTools.prettymemory(10 * per_iter))")
