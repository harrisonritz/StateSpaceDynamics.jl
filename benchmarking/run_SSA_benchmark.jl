using Pkg
Pkg.activate("benchmarking")

include("SSA_Benchmark.jl")
using .SSA_Benchmark
using BenchmarkTools
using StableRNGs
using Printf
using CSV
using DataFrames
using StateSpaceAnalysis
using Revise
using Accessors
# using StateSpaceDynamics

# Benchmark configuration
struct BenchConfig
    latent_dims::Vector{Int}
    obs_dims::Vector{Int}
    seq_lengths::Vector{Int}
    n_iters::Int
    n_repeats::Int
end

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

# ----------------------
# LDS Benchmarks
# ----------------------


save_name = "benchmarking/results/lds_benchmark_recoverx.csv"

# lds_config = BenchConfig(
#     [2, 4, 6, 8],  # latent_dims
#     [2, 4, 6, 8],  # obs_dims
#     [100, 500, 1000],  # seq_lengths
#     100,  # n_iters
#     5     # n_repeats
# )

lds_config = BenchConfig(
    [4, 8, 16],  # latent_dims
    [8, 16],  # obs_dims
    [100],  # seq_lengths
    10,  # n_trials
    10     # n_repeats
)


# lds_implementations = [
#     SSD_LDSImplem(),
#     pykalman_LDSImplem(),
#     Dynamax_LDSImplem()
# ]

lds_implementations = [
    SSA_LDSImplem(),
    SSD_LDSImplem(),
]

lds_results = []

# latent_dim = 5
# obs_dim = 10
# seq_len = 100
# impl = SSA_LDSImplem()



# Run 1 EM iteration to compile
# instance = LDSInstance(; latent_dim=2, obs_dim=2, num_trials=10, seq_length=100)
# rng = StableRNG(1234)
# params = init_params(rng, instance)

# ssd_model = build_model(SSD_LDSImplem(), instance, params)
# x, y = build_data(rng, ssd_model, instance)

# fit_model = deepcopy(model)
# ssd_fit = StateSpaceDynamics.fit!(fit_model, y, max_iter=100, tol=1e-50)
# xhat, phat = smooth(fit_model, y)





for latent_dim in lds_config.latent_dims
    for obs_dim in lds_config.obs_dims
        # obs_dim < latent_dim && continue
        for seq_len in lds_config.seq_lengths
            println("\n→ Benchmarking LDS with latent_dim=$latent_dim, obs_dim=$obs_dim, seq_len=$seq_len")

            instance = LDSInstance(; latent_dim=latent_dim, obs_dim=obs_dim, num_trials=lds_config.n_iters, seq_length=seq_len)
            
            rng = StableRNG(1234)
            params = init_params(rng, instance)
            params_init = init_params(rng, instance)

            ssd_model = build_model(SSD_LDSImplem(), instance, params)
            x, y = build_data(rng, ssd_model, instance)

            results_row = Dict{String, Any}()
            results_row["config"] = (latent_dim=latent_dim, obs_dim=obs_dim, seq_len=seq_len)

            for impl in lds_implementations
                print("  Running $(string(impl))... ")
                try
                    # clear results Tuple
                    model = build_model(impl, instance, deepcopy(params_init));
                    result = run_benchmark(impl, deepcopy(model), y)
                    
                    recover = recover_states(impl, deepcopy(model), y, x)
                    result = @insert result.R2 = recover.R2
                    result = @insert result.R2aligned = recover.R2aligned
                    
                    results_row[string(impl)] = result

                    if result.success
                        @printf("✓ time = %.3f sec\n", result.time / 1e9)
                    else
                        println("✗ failed")
                    end
                catch e
                    results_row[string(impl)] = (time=NaN, memory=0, allocs=0, success=false, R2=NaN, R2aligned=NaN)
                    println("✗ exception: ", e)
                end
            end

            push!(lds_results, results_row)
            println("-"^50)
        end
    end
end

# Save LDS results
df_lds = DataFrame()
for row in lds_results
    config = row["config"]
    for (name, result) in row
        name == "config" && continue
        push!(df_lds, (
            implementation = name,
            latent_dim = config.latent_dim,
            obs_dim = config.obs_dim,
            seq_len = config.seq_len,
            time_sec = result[:time] / 1e9,
            memory = result[:memory],
            allocs = result[:allocs],
            success = result[:success],
            R2 = result[:R2],
            R2aligned = result[:R2aligned]
        ))
    end
end
CSV.write(save_name, df_lds)


# ----------------------
# HMM Benchmarks
# ----------------------

# hmm_config = BenchConfig(
#     [2, 4, 6, 8],          # num_states
#     [1],                   # emission_dim
#     [100, 500, 1000],      # seq_lengths
#     100,
#     5
# )

# hmm_implementations = [
#     SSD_HMMImplem(),
#     HiddenMarkovModels_Implem(),
#     Dynamax_HMMImplem()
# ]

# hmm_results = []
# max_retries = 2  # number of times to retry on failure

# for num_states in hmm_config.latent_dims
#     for seq_len in hmm_config.seq_lengths
#         println("\n→ Benchmarking HMM with num_states=$num_states, seq_len=$seq_len")

#         instance = HMMInstance(; num_states=num_states, emission_dim=1, num_trials=1, seq_length=seq_len)
#         rng = StableRNG(1234)
#         params = init_params(rng, instance)

#         ssd_model = build_model(SSD_HMMImplem(), instance, params)
#         y = build_data(rng, ssd_model, instance)

#         results_row = Dict{String, Any}()
#         results_row["config"] = (num_states=num_states, seq_len=seq_len)

#         for impl in hmm_implementations
#             print("  Running $(string(impl))... ")
#             attempt = 1
#             success = false
#             result = (time=NaN, memory=0, allocs=0, success=false)  # default placeholder

#             while attempt <= max_retries && !success
#                 try
#                     println("Attempt $attempt")
#                     model = build_model(impl, instance, params)

#                     result = run_benchmark(impl, model, y[1])
#                     success = result.success

#                     if success
#                         @printf("✓ time = %.3f sec\n", result.time / 1e9)
#                     else
#                         println("✗ failed")
#                         if attempt < max_retries
#                             println("Retrying...")
#                         end
#                     end
#                 catch e
#                     println("✗ exception: ", e)
#                     result = (time=NaN, memory=0, allocs=0, success=false)
#                     if attempt < max_retries
#                         println("Retrying after exception...")
#                     end
#                 end
#                 attempt += 1
#             end

#             # Store final result (success or failure after retries)
#             results_row[string(impl)] = result
#         end

#         push!(hmm_results, results_row)
#         println("-"^50)
#     end
# end

# # Save HMM results
# df_hmm = DataFrame()
# for row in hmm_results
#     config = row["config"]
#     for (name, result) in row
#         name == "config" && continue
#         push!(df_hmm, (
#             implementation = name,
#             num_states = config.num_states,
#             seq_len = config.seq_len,
#             time_sec = result[:time] / 1e9,
#             memory = result[:memory],
#             allocs = result[:allocs],
#             success = result[:success]
#         ))
#     end
# end
# CSV.write("benchmarking/results/hmm_benchmark_results.csv", df_hmm)
