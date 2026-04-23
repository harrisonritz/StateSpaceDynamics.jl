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

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

struct BenchConfig
    latent_dims::Vector{Int}
    obs_dims::Vector{Int}
    seq_length::Int
    n_iters::Int
    n_repeats::Int
end

kf_config = BenchConfig(
    [2, 4, 8],   # latent_dims
    [4, 8],      # obs_dims
    100,         # seq_length
    10,          # n_iters (EM iterations per fit)
    100,         # n_repeats (benchmark samples)
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
    A = SSD.random_rotation_matrix(latent_dim, rng)

    Q = randn(rng, latent_dim, latent_dim)
    Q = Q * Q' .+ 1e-3

    x0 = randn(rng, latent_dim)
    P0 = randn(rng, latent_dim, latent_dim)
    P0 = P0 * P0' .+ 1e-3

    C = randn(rng, obs_dim, latent_dim)
    R = randn(rng, obs_dim, obs_dim)
    R = R * R' .+ 1e-3

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

methods = [(:tridiag, false), (:kalman, true)]

results = DataFrame(
    method      = Symbol[],
    latent_dim  = Int[],
    obs_dim     = Int[],
    seq_len     = Int[],
    num_trials  = Int[],
    n_iters     = Int[],
    time_sec    = Float64[],
    memory      = Int[],
    allocs      = Int[],
)

for latent_dim in kf_config.latent_dims
    for obs_dim in kf_config.obs_dims
        obs_dim < latent_dim && continue

        println("\n→ latent_dim=$latent_dim, obs_dim=$obs_dim, seq_len=$(kf_config.seq_length), num_trials=$NUM_TRIALS")

        rng = StableRNG(1234)
        params = init_params(rng, latent_dim, obs_dim)

        # Generate one shared dataset from a reference model (tridiag path is
        # the default; both methods fit the same y).
        ref = build_lds(params, false)
        _, y = rand(rng, ref; ntrials=NUM_TRIALS, tsteps=kf_config.seq_length)

        for (name, kf) in methods
            print("  $(rpad(string(name), 8)) ")

            model = build_lds(params, kf)

            # Warm up / precompile
            SSD.fit!(deepcopy(model), y; max_iter=1, tol=1e-50, progress=false)

            max_iter = kf_config.n_iters
            bench = @benchmark SSD.fit!(m, $y;
                                        max_iter=$max_iter,
                                        tol=1e-50,
                                        progress=false) setup=(m = deepcopy($model)) samples=kf_config.n_repeats evals=1

            med_sec = median(bench).time / 1e9
            push!(results, (
                name, latent_dim, obs_dim,
                kf_config.seq_length, NUM_TRIALS, max_iter,
                med_sec, bench.memory, bench.allocs,
            ))
            @printf("median = %.4f sec  (mem=%d, allocs=%d)\n",
                    med_sec, bench.memory, bench.allocs)
        end
    end
end

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

results_dir = joinpath(@__DIR__, "results")
isdir(results_dir) || mkpath(results_dir)

csv_path = joinpath(results_dir, "kf_benchmark_results.csv")
CSV.write(csv_path, results)
println("\nWrote results → $csv_path")

# ----------------------------------------------------------------------
# Plot: one subplot per obs_dim, median time vs latent_dim, one line per method
# ----------------------------------------------------------------------

method_colors = Dict(:tridiag => "#E69F00", :kalman => "#0072B2")
method_markers = Dict(:tridiag => :diamond, :kalman => :circle)
method_labels = Dict(:tridiag => "Tridiagonal", :kalman => "Kalman/RTS")

gb = groupby(results, [:method, :latent_dim, :obs_dim])
agg = combine(gb, :time_sec => median => :median_time)

ovals = sort(unique(agg.obs_dim))
lvals = sort(unique(agg.latent_dim))
mvals = [:tridiag, :kalman]

y_min, y_max = extrema(agg.median_time)

subplots = Plots.Plot[]
for (j, o) in enumerate(ovals)
    panel = @view agg[agg.obs_dim .== o, :]
    p = plot(
        title = "obs_dim = $o",
        xlabel = "latent_dim",
        ylabel = (j == 1) ? "Median fit time (s)" : "",
        framestyle = :box,
        grid = :y,
        minorgrid = true,
        legend = false,
        xticks = lvals,
        ylims = (y_min * 0.9, y_max * 1.1),
    )
    for m in mvals
        sub = @view panel[panel.method .== m, :]
        isempty(sub) && continue
        xs = sort(sub, :latent_dim)
        plot!(p, xs.latent_dim, xs.median_time;
              label = method_labels[m],
              color = method_colors[m],
              marker = method_markers[m],
              linewidth = 2,
              markersize = 6)
    end
    push!(subplots, p)
end

grid_plot = plot(subplots...;
                 layout = (1, length(ovals)),
                 size = (350 * length(ovals), 320),
                 left_margin = 10mm, bottom_margin = 10mm)

legend_plot = plot(framestyle=:none, showaxis=false, size=(160, 320))
for m in mvals
    plot!(legend_plot, [NaN], [NaN];
          label = method_labels[m],
          linewidth = 2,
          marker = method_markers[m],
          seriescolor = method_colors[m])
end

final_plt = plot(grid_plot, legend_plot;
                 layout = @layout([a{0.85w} b{0.15w}]),
                 plot_title = "fit! — Tridiagonal vs Kalman (seq_len=$(kf_config.seq_length), trials=$NUM_TRIALS, iters=$(kf_config.n_iters))")
plot!(final_plt; tickfontsize=10, guidefontsize=12, legendfontsize=11)

png_path = joinpath(results_dir, "kf_benchmark.png")
savefig(final_plt, png_path)
println("Wrote plot → $png_path")
