# Benchmarking Guide

This document explains the benchmarking infrastructure for StateSpaceDynamics.jl.

## Overview

StateSpaceDynamics.jl uses automated benchmarking to track performance over time and detect regressions. Benchmarks run automatically on:

- Every push to `main`
- Every pull request
- Manual workflow dispatch

## Benchmark Workflow

The benchmark workflow ([`.github/workflows/benchmark.yml`](workflows/benchmark.yml)) performs the following:

1. **Setup**: Installs Julia and project dependencies
2. **Run Benchmarks**: Executes the benchmark suite across different problem sizes
3. **Save Results**: Stores results as workflow artifacts (retained for 90 days)
4. **Compare**: For PRs, compares against baseline from the target branch
5. **Report**: Posts comparison results as a PR comment

## What Gets Benchmarked

### Gaussian LDS Smoothing
Tests the RTS smoothing algorithm for Gaussian observations:
- Latent dimensions: 2, 4, 8
- Observation dimensions: 5, 10, 20
- Sequence lengths: 100, 500 timesteps

### Poisson LDS Smoothing
Tests the Laplace approximation-based smoothing for Poisson observations:
- Latent dimensions: 2, 4
- Observation dimensions: 5, 10
- Sequence lengths: 100, 500 timesteps

### HMM Forward-Backward
Tests the forward-backward algorithm:
- Number of states: 2, 4, 8
- Sequence lengths: 100, 500 timesteps

## Metrics Tracked

For each benchmark:
- **Execution time**: Median time in milliseconds
- **Memory usage**: Total allocated memory in MB
- **Allocations**: Number of memory allocations

## Viewing Results

### In CI

1. Navigate to the [Benchmarks workflow](https://github.com/depasquale-lab/StateSpaceDynamics.jl/actions/workflows/benchmark.yml)
2. Click on a specific run
3. Download the `benchmark-results-*` artifact
4. View `summary.csv` for a readable summary

### In Pull Requests

The benchmark workflow automatically comments on PRs with:
- Performance comparison vs. the base branch
- Speedup/slowdown percentages
- Memory usage changes

## Running Benchmarks Locally

### Quick Julia-only benchmarks

Run the same benchmarks that CI runs:

```bash
julia --project=benchmarking -e '
  using Pkg
  Pkg.instantiate()
  Pkg.develop(PackageSpec(path=pwd()))
'

# Then run the benchmark script inline or save it to a file
julia --project=benchmarking .github/workflows/benchmark.yml  # (extract the Julia script)
```

### Full benchmark suite (with Python comparisons)

The `benchmarking/` directory contains a more comprehensive suite that compares against Python libraries:

```bash
cd benchmarking
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.develop(PackageSpec(path=".."))'
julia --project run_benchmark.jl
```

This requires:
- Python with `pykalman` and `dynamax` installed
- HiddenMarkovModels.jl for HMM comparisons

## Interpreting Performance Changes

### Good Changes
- ✅ Speedup > 1.05 (5% faster)
- ✅ Memory reduction > 5%
- ✅ Fewer allocations

### Acceptable Changes
- ✓ Speedup between 0.95-1.05 (within noise)
- ✓ Memory change < 5%

### Concerning Changes
- ⚠️ Speedup < 0.95 (5% slower) - investigate
- ⚠️ Memory increase > 10% - investigate
- ⚠️ Significant allocation increase - investigate

## Adding New Benchmarks

To add a new benchmark to the CI suite:

1. Edit `.github/workflows/benchmark.yml`
2. Add your benchmark to the `SUITE` BenchmarkGroup:

```julia
SUITE["MyCategory"]["operation", "params"] =
    @benchmarkable my_function($args) samples=10 seconds=5
```

3. Test locally first
4. Submit a PR

## Best Practices

1. **Consistency**: Benchmarks use `StableRNG` for reproducibility
2. **Warm-up**: BenchmarkTools handles warm-up automatically
3. **Sample size**: Default is 10 samples with 5-second timeout per benchmark
4. **Problem sizes**: Include small, medium, and large problems
5. **Representative**: Benchmark realistic use cases, not edge cases

## Troubleshooting

### Benchmark times out
- Reduce problem size
- Increase `seconds` parameter
- Reduce `samples` parameter

### Results are noisy
- Increase `samples`
- Check for system load during benchmarking
- CI results are typically more stable than local runs

### Out of memory
- Reduce problem sizes
- Run fewer benchmarks in parallel
- GitHub Actions runners have limited memory

## Future Enhancements

Planned improvements:
- [ ] Historical performance tracking
- [ ] Benchmark result visualization
- [ ] Automated performance regression alerts
- [ ] Comparison with Python libraries in CI
- [ ] Per-commit performance dashboard
