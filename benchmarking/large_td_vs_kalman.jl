using StateSpaceDynamics, LinearAlgebra, Random, BenchmarkTools, Printf

function make_lds(D, p, kalman; seed=42)
    rng = MersenneTwister(seed)
    A = StateSpaceDynamics.random_rotation_matrix(D, rng)
    M1 = randn(rng, D, D); Q  = M1 * transpose(M1) + 1e-3 * I
    x0 = randn(rng, D)
    M2 = randn(rng, D, D); P0 = M2 * transpose(M2) + 1e-3 * I
    C  = randn(rng, p, D)
    M3 = randn(rng, p, p); R  = M3 * transpose(M3) + 1e-3 * I
    b  = randn(rng, D); d = randn(rng, p)
    sm = GaussianStateModel(; A=Matrix(A), Q=Matrix(Q), b=b, x0=x0, P0=Matrix(P0))
    om = GaussianObservationModel(; C=C, R=Matrix(R), d=d)
    return LinearDynamicalSystem(sm, om; kalman_filter=kalman)
end

# D=128, p=64, T=250, N=500
const D, p, T_t, N = 128, 64, 250, 500
@printf("Setup: D=%d, p=%d, T=%d, N=%d\n", D, p, T_t, N)
@printf("Memory: batched_x/grad ≈ 2×%.0f MB, batched_y ≈ %.0f MB, Kalman covs ≈ %.0f MB\n",
    D*T_t*N*8/1e6, p*T_t*N*8/1e6, 3*T_t*D*D*8/1e6 + 3*T_t*D*N*8/1e6)

lds_td = make_lds(D, p, false)
rng = MersenneTwister(123)
println("\nSampling data...")
@time _, y_td = StateSpaceDynamics.rand(rng, lds_td, fill(T_t, N))
println("Stacking 3D y...")
@time y3 = cat(y_td...; dims=3)

println("\n=== Warm-up (1 iter each) ===")
print("  TD     warmup: "); @time StateSpaceDynamics.fit!(make_lds(D, p, false), y_td; max_iter=1, tol=0.0, progress=false)
print("  Kalman warmup: "); @time StateSpaceDynamics.fit!(make_lds(D, p, true),  y3;   max_iter=1, tol=0.0, progress=false)

# 5 EM iters x 3 samples for a manageable wall-clock budget
n_iter = 5
n_samples = 3

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 600.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = n_samples

@printf("\n=== %d-iter fit, median of %d samples ===\n", n_iter, n_samples)

print("TD     fit!: ")
b_td = @benchmark StateSpaceDynamics.fit!(c, $y_td; max_iter=$n_iter, tol=0.0, progress=false) setup=(c = $(make_lds)($D, $p, false))
@printf("median %.3f s  mem %.1f MB  allocs %d\n",
    median(b_td).time / 1e9, b_td.memory / 1024 / 1024, b_td.allocs)

print("Kalman fit!: ")
b_kf = @benchmark StateSpaceDynamics.fit!(c, $y3;   max_iter=$n_iter, tol=0.0, progress=false) setup=(c = $(make_lds)($D, $p, true))
@printf("median %.3f s  mem %.1f MB  allocs %d\n",
    median(b_kf).time / 1e9, b_kf.memory / 1024 / 1024, b_kf.allocs)

@printf("\nTD/Kalman ratio: %.2fx (TD %s)\n",
    median(b_td).time / median(b_kf).time,
    median(b_td).time < median(b_kf).time ? "faster" : "slower")
@printf("Per-iter: TD %.1f ms,  Kalman %.1f ms\n",
    median(b_td).time / 1e6 / n_iter, median(b_kf).time / 1e6 / n_iter)
