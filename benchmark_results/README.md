# Benchmark Results

This directory contains benchmark results from CI runs. Results are automatically generated and stored as artifacts in GitHub Actions.

## Structure

- `results.json` - Full BenchmarkTools results in JSON format
- `summary.csv` - Human-readable summary of all benchmarks
- `comparison.csv` - Comparison with baseline (for PRs)

## Viewing Results

### Latest Results

The latest benchmark results are available in the [GitHub Actions artifacts](https://github.com/depasquale-lab/StateSpaceDynamics.jl/actions/workflows/benchmark.yml).

### Benchmark Suite

The benchmark suite includes:

**Gaussian LDS**

- Smoothing operations for various problem sizes
- Parameters: latent_dim ∈ {2, 4, 8}, obs_dim ∈ {5, 10, 20}, T ∈ {100, 500}

**Poisson LDS**

- Laplace approximation-based smoothing
- Parameters: latent_dim ∈ {2, 4}, obs_dim ∈ {5, 10}, T ∈ {100, 500}

**HMM**

- Forward-backward algorithm (loglikelihood computation)
- Parameters: num_states ∈ {2, 4, 8}, T ∈ {100, 500}

## Running Benchmarks Locally

To run the benchmarks on your machine:

```bash
cd benchmarking
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.develop(PackageSpec(path=".."))'
julia --project run_benchmark.jl  # Full benchmark suite (includes Python comparisons)
```

Or run just the Julia benchmarks:

```bash
julia --project=benchmarking -e 'include("../.github/workflows/benchmark_script.jl")'
```

## Interpreting Results

- **time_ns**: Median execution time in nanoseconds
- **memory_bytes**: Total memory allocated during execution
- **allocs**: Number of memory allocations
- **time_ms**: Median execution time in milliseconds (derived)
- **memory_mb**: Total memory in megabytes (derived)

## Performance Tracking

The CI workflow compares PR benchmarks against the base branch to detect:

- Performance regressions (slower execution)
- Memory usage increases
- Allocation count changes

Significant changes are reported in PR comments.
