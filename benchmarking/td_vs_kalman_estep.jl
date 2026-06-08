using StateSpaceDynamics, LinearAlgebra, Random, BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 3.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5

function make_gaussian_lds(D, p; seed=42, kalman=false)
    rng = MersenneTwister(seed)
    A = StateSpaceDynamics.random_rotation_matrix(D, rng)
    Q = (M = randn(rng, D, D); M*M' + 1e-3 * I)
    x0 = randn(rng, D)
    P0 = (M = randn(rng, D, D); M*M' + 1e-3 * I)
    C = randn(rng, p, D)
    R = (M = randn(rng, p, p); M*M' + 1e-3 * I)
    b = randn(rng, D); d = randn(rng, p)
    sm = GaussianStateModel(; A=Matrix(A), Q=Matrix(Q), b=b, x0=x0, P0=Matrix(P0))
    om = GaussianObservationModel(; C=C, R=Matrix(R), d=d)
    return LinearDynamicalSystem(sm, om; kalman_filter=kalman)
end

const D, p, T_t, N = 5, 10, 200, 8

lds_td = make_gaussian_lds(D, p; kalman=false)
rng = MersenneTwister(123)
_, y_td = StateSpaceDynamics.rand(rng, lds_td, fill(T_t, N))

tsteps_per_trial = fill(T_t, N)
tfs = StateSpaceDynamics.initialize_FilterSmooth(lds_td, tsteps_per_trial)
pool_size = Threads.maxthreadid()
sws_pool = Vector{StateSpaceDynamics.SmoothWorkspace{Float64}}(undef, pool_size)
sws_pool[1] = StateSpaceDynamics.SmoothWorkspace(
    Float64, D, p, T_t; u_dim=0, d_dim=0, ntrials=N,
)
for i in 2:pool_size
    sws_pool[i] = StateSpaceDynamics.SmoothWorkspace(
        Float64, D, p, T_t; u_dim=0, d_dim=0,
    )
end
suf = StateSpaceDynamics._initialize_td_sufficient_statistics(
    Float64, lds_td, tsteps_per_trial,
)
u_seq = [zeros(0, T_t) for _ in 1:N]
v_seq = [zeros(0, T_t) for _ in 1:N]
StateSpaceDynamics._td_init_const_blocks!(
    sws_pool[1], lds_td, tsteps_per_trial, y_td, u_seq, v_seq,
)

# Warm
for _ in 1:3
    StateSpaceDynamics.smooth!(lds_td, tfs, y_td, sws_pool, u_seq, v_seq)
    StateSpaceDynamics._aggregate_td_suff_stats!(
        suf, tfs, lds_td, u_seq, v_seq, y_td, sws_pool[1],
    )
end

println("=== TD path components (per E-step) ===")
b_smooth = @benchmark StateSpaceDynamics.smooth!(
    $lds_td, $tfs, $y_td, $sws_pool, $u_seq, $v_seq
)
println("  smooth! (full multi-trial): t=$(round(median(b_smooth).time / 1e6; digits=3)) ms allocs=$(b_smooth.allocs) mem=$(round(b_smooth.memory/1024; digits=1)) KB")

# Decompose smooth! into cov pass (precompute_shared_cov!) + mean passes (× N)
b_cov_only = @benchmark StateSpaceDynamics._precompute_shared_cov!(
    $(sws_pool[1]), $lds_td, $T_t
)
println("    _precompute_shared_cov! (1 call):    t=$(round(median(b_cov_only).time / 1e6; digits=3)) ms allocs=$(b_cov_only.allocs) mem=$(round(b_cov_only.memory/1024; digits=1)) KB")

# After precompute, time a single _smooth_mean_only! call
StateSpaceDynamics._precompute_shared_cov!(sws_pool[1], lds_td, T_t)
for trial in 1:N
    tfs[trial].p_smooth = sws_pool[1].p_smooth_shared
    tfs[trial].p_smooth_tt1 = sws_pool[1].p_smooth_tt1_shared
end
b_mean1 = @benchmark StateSpaceDynamics._smooth_mean_only!(
    $lds_td, $(tfs[1]), $(y_td[1]), $(sws_pool[1]), $(u_seq[1]), $(v_seq[1]), $(sws_pool[1])
)
println("    _smooth_mean_only! (1 trial, serial): t=$(round(median(b_mean1).time / 1e6; digits=3)) ms allocs=$(b_mean1.allocs) mem=$(round(b_mean1.memory/1024; digits=1)) KB")
println("    _smooth_mean_only! × $N (serial est):  $(round(N * median(b_mean1).time / 1e6; digits=3)) ms")

# Per-trial sub-decomp: just Gradient! vs just backsubst
let
    sws = sws_pool[1]
    fs = tfs[1]
    x_mat = reshape(fs.E_z, D, T_t)
    b_grad = @benchmark StateSpaceDynamics.Gradient!(
        $sws, $lds_td, $(y_td[1]), $x_mat, $(u_seq[1]), $(v_seq[1])
    )
    println("      Gradient!  (1 trial): t=$(round(median(b_grad).time / 1e6; digits=3)) ms allocs=$(b_grad.allocs)")

    # Backsubst alone
    n_active = D * T_t
    X0 = view(sws.X₀, 1:n_active)
    grad_vec = view(sws.grad_vec, 1:n_active)
    copyto!(X0, fs.E_z)
    StateSpaceDynamics.Gradient!(sws, lds_td, y_td[1], x_mat, u_seq[1], v_seq[1])
    for t in 1:T_t, i in 1:D
        sws.grad_vec[(t - 1) * D + i] = -sws.grad_buf[i, t]
    end
    shared_btd = sws.btd
    neg_sub_v = view(shared_btd.neg_sub, 1:(T_t - 1))
    b_bs = @benchmark StateSpaceDynamics.block_tridiagonal_backsubst!(
        $X0, $neg_sub_v, $grad_vec, $shared_btd, $T_t
    )
    println("      backsubst! (1 trial): t=$(round(median(b_bs).time / 1e6; digits=3)) ms allocs=$(b_bs.allocs)")
end

# After smooth! populates tfs, time the aggregator
StateSpaceDynamics.smooth!(lds_td, tfs, y_td, sws_pool, u_seq, v_seq)
b_agg = @benchmark StateSpaceDynamics._aggregate_td_suff_stats!(
    $suf, $tfs, $lds_td, $u_seq, $v_seq, $y_td, $(sws_pool[1])
)
println("  _aggregate_td_suff_stats!:  t=$(round(median(b_agg).time / 1e6; digits=3)) ms allocs=$(b_agg.allocs) mem=$(round(b_agg.memory/1024; digits=1)) KB")

# M-step
StateSpaceDynamics.smooth!(lds_td, tfs, y_td, sws_pool, u_seq, v_seq)
StateSpaceDynamics._aggregate_td_suff_stats!(suf, tfs, lds_td, u_seq, v_seq, y_td, sws_pool[1])
b_mstep = @benchmark StateSpaceDynamics.mstep!(
    $lds_td, $suf, $(sws_pool[1])
)
println("  mstep!:                     t=$(round(median(b_mstep).time / 1e6; digits=3)) ms allocs=$(b_mstep.allocs) mem=$(round(b_mstep.memory/1024; digits=1)) KB")

td_per_iter = median(b_smooth).time + median(b_agg).time + median(b_mstep).time
println("  TD per-iter total: $(round(td_per_iter / 1e6; digits=3)) ms")

println("\n=== Kalman path components (per E-step) ===")
lds_kf = make_gaussian_lds(D, p; kalman=true)
y3 = cat(y_td...; dims=3)
kf_data = StateSpaceDynamics.format_kf_data!(lds_kf, y3, nothing, nothing, T_t, N)
kws = StateSpaceDynamics.KalmanWorkspace(lds_kf, T_t, N)
kf_suf = StateSpaceDynamics.initialize_SufficientStatistics(lds_kf, kf_data, kws)
# Warm
for _ in 1:3
    StateSpaceDynamics.precompute_kalman_constants!(kws, lds_kf, kf_data)
    StateSpaceDynamics.smooth_cov!(lds_kf, kws)
    StateSpaceDynamics.smooth_mean!(lds_kf, kws)
    StateSpaceDynamics.sufficient_statistics!(kf_suf, kws, kf_data)
end

b_kf_precomp = @benchmark StateSpaceDynamics.precompute_kalman_constants!(
    $kws, $lds_kf, $kf_data
)
println("  precompute_kalman_constants!: t=$(round(median(b_kf_precomp).time / 1e6; digits=3)) ms allocs=$(b_kf_precomp.allocs) mem=$(round(b_kf_precomp.memory/1024; digits=1)) KB")

b_kf_cov = @benchmark StateSpaceDynamics.smooth_cov!($lds_kf, $kws)
println("  smooth_cov!:                  t=$(round(median(b_kf_cov).time / 1e6; digits=3)) ms allocs=$(b_kf_cov.allocs) mem=$(round(b_kf_cov.memory/1024; digits=1)) KB")

b_kf_mean = @benchmark StateSpaceDynamics.smooth_mean!($lds_kf, $kws)
println("  smooth_mean!:                 t=$(round(median(b_kf_mean).time / 1e6; digits=3)) ms allocs=$(b_kf_mean.allocs) mem=$(round(b_kf_mean.memory/1024; digits=1)) KB")

b_kf_ss = @benchmark StateSpaceDynamics.sufficient_statistics!($kf_suf, $kws, $kf_data)
println("  sufficient_statistics!:       t=$(round(median(b_kf_ss).time / 1e6; digits=3)) ms allocs=$(b_kf_ss.allocs) mem=$(round(b_kf_ss.memory/1024; digits=1)) KB")

b_kf_mstep = @benchmark StateSpaceDynamics.mstep!($lds_kf, $kf_suf, $kws)
println("  mstep!:                       t=$(round(median(b_kf_mstep).time / 1e6; digits=3)) ms allocs=$(b_kf_mstep.allocs) mem=$(round(b_kf_mstep.memory/1024; digits=1)) KB")

kf_per_iter = median(b_kf_precomp).time + median(b_kf_cov).time + median(b_kf_mean).time +
              median(b_kf_ss).time + median(b_kf_mstep).time
println("  Kalman per-iter total: $(round(kf_per_iter / 1e6; digits=3)) ms")

println("\n=== Per-iter side-by-side ===")
println("  TD     (smooth+aggregate+mstep): $(round(td_per_iter / 1e6; digits=3)) ms")
println("  Kalman (precomp+cov+mean+ss+mstep): $(round(kf_per_iter / 1e6; digits=3)) ms")
println("  Gap (TD - Kalman): $(round((td_per_iter - kf_per_iter) / 1e6; digits=3)) ms/iter")

# Whole-fit confirmation
println("\n=== Whole fit (1 EM iter, for sanity) ===")
StateSpaceDynamics.fit!(make_gaussian_lds(D, p; kalman=true), y3; max_iter=1, progress=false)
b_kf_1iter = @benchmark StateSpaceDynamics.fit!(
    lds_copy, $y3; max_iter=1, tol=0.0, progress=false,
) setup=(lds_copy = $(make_gaussian_lds)($D, $p; kalman=true))
println("  Kalman fit! (1 iter): t=$(round(median(b_kf_1iter).time / 1e6; digits=3)) ms allocs=$(b_kf_1iter.allocs) mem=$(round(b_kf_1iter.memory/1024; digits=1)) KB")

b_td_1iter = @benchmark StateSpaceDynamics.fit!(
    lds_copy, $y_td; max_iter=1, tol=0.0, progress=false,
) setup=(lds_copy = $(make_gaussian_lds)($D, $p; kalman=false))
println("  TD     fit! (1 iter): t=$(round(median(b_td_1iter).time / 1e6; digits=3)) ms allocs=$(b_td_1iter.allocs) mem=$(round(b_td_1iter.memory/1024; digits=1)) KB")
