#=============================================================================
Spline-Gaussian observations: manifold discovery by a monotonic normalizing flow

The model is `y = g⁻¹(z)`, `z | x ~ N(Cx + d + Dv, R)`, with `g` an element-wise
monotonic rational-quadratic spline. Three properties carry most of the weight
here, and each has a sharp test:

  1. **The spline kernel is right.** Monotone and range-preserving by
     construction, exactly invertible, and its analytic parameter gradients
     match `ForwardDiff`. If that last one is wrong the M-step silently
     optimizes the wrong thing, so it is checked over a grid that includes the
     identity tails.

  2. **The bookkeeping is right.** Because the embedded observations `z` change
     every iteration, a log-density is only meaningful on the `y` scale. So
     `elbo == loglikelihood` whenever there are no priors (the smoother is
     exact), `sum(trial_elbos) == elbo`, and a fit initialized at the identity
     warp reproduces an ordinary Gaussian LDS exactly.

  3. **The algorithm is right.** ECM gives a monotone observed-data
     log-likelihood, and on data generated from a known warp the fit recovers
     that warp and beats the linear model out of sample.
=============================================================================#

const SG_LATENT = 2

#=
ECM is monotone, but the ELBO is a sum of terms of the trace's own magnitude, so
"non-decreasing" has to be judged relative to that scale rather than in absolute
nats — near convergence the true increments shrink below the summation's own
rounding.
=#
function sg_nondecreasing(el; rtol=1e-8)
    scale = max(1.0, maximum(abs, el))
    return all(diff(collect(el)) .>= -rtol * scale)
end

function sg_state_model(::Type{T}=Float64; θ=0.15, r=0.96) where {T<:Real}
    return GaussianStateModel(
        T.(r * [cos(θ) -sin(θ); sin(θ) cos(θ)]),
        Matrix{T}(0.02I, SG_LATENT, SG_LATENT),
        zeros(T, SG_LATENT),
        zeros(T, SG_LATENT),
        Matrix{T}(0.3I, SG_LATENT, SG_LATENT),
    )
end

#=
A generating model: a known non-identity warp on a fixed interval, so a fit can
be scored against it. `bounds` is given explicitly (rather than inferred from
data) because the truth has to exist before the data does.
=#
function sg_true_model(; p::Int=5, seed::Int=1, n_bins::Int=6, scale=0.8)
    rng = StableRNG(4000 + seed)
    C = randn(rng, p, SG_LATENT)
    om = SplineGaussianObservationModel(
        C,
        Matrix(0.05I, p, p),
        zeros(p);
        bounds=(fill(-8.0, p), fill(8.0, p)),
        n_bins=n_bins,
    )
    SSD.warp_unpack!(om.warp, scale .* randn(rng, warp_nparams(om.warp)))
    return LinearDynamicalSystem(sg_state_model(), om)
end

function sg_init_model(Y; p::Int=5, seed::Int=2, n_bins::Int=6, kwargs...)
    rng = StableRNG(5000 + seed)
    om = SplineGaussianObservationModel(
        randn(rng, p, SG_LATENT),
        Matrix(1.0I, p, p),
        zeros(p);
        y=Y,
        margin=0.05,
        n_bins=n_bins,
        spline_ridge=0.0,
        kwargs...,
    )
    sm = GaussianStateModel(
        Matrix(0.9I, SG_LATENT, SG_LATENT),
        Matrix(0.1I, SG_LATENT, SG_LATENT),
        zeros(SG_LATENT),
        zeros(SG_LATENT),
        Matrix(1.0I, SG_LATENT, SG_LATENT),
    )
    return LinearDynamicalSystem(sm, om)
end

# ============================================================================
# The spline kernel
# ============================================================================

function test_warp_identity_at_zero()
    w = MonotonicWarp([-2.0, 0.0], [2.0, 5.0]; n_bins=6)
    @test is_identity_warp(w)
    for j in 1:2, y in range(w.lo[j] - 1, w.hi[j] + 1; length=25)
        z, ℓ = warp_forward(w, j, y)
        @test z ≈ y atol = 1e-12
        @test ℓ ≈ 0 atol = 1e-12
    end
    # The knot layout MonotonicSplines.RQSpline requires: endpoints on the
    # diagonal, unit derivative there.
    @test w.pX[1, :] == w.pY[1, :] == w.lo
    @test w.pX[end, :] == w.pY[end, :] == w.hi
    @test all(isone, w.dYdX[1, :])
    @test all(isone, w.dYdX[end, :])
    return nothing
end

function test_warp_monotone_and_invertible()
    rng = StableRNG(11)
    w = MonotonicWarp([-2.0, -1.0, 0.0], [2.0, 3.0, 5.0]; n_bins=7)
    SSD.warp_unpack!(w, 1.1 .* randn(rng, warp_nparams(w)))

    for j in 1:3
        ys = collect(range(w.lo[j] - 0.5, w.hi[j] + 0.5; length=801))
        zs = [warp_forward(w, j, y)[1] for y in ys]
        @test issorted(zs)
        # Range preserving: the interval maps onto itself, identity outside.
        @test warp_forward(w, j, w.lo[j])[1] ≈ w.lo[j] atol = 1e-12
        @test warp_forward(w, j, w.hi[j])[1] ≈ w.hi[j] atol = 1e-12
        @test warp_forward(w, j, w.lo[j] - 0.4)[1] ≈ w.lo[j] - 0.4
        for y in ys
            @test warp_inverse(w, j, warp_forward(w, j, y)[1]) ≈ y atol = 1e-8
        end
        # log g' against a central difference of g.
        for y in range(w.lo[j] + 0.05, w.hi[j] - 0.05; length=151)
            h = 1e-6
            fd = (warp_forward(w, j, y + h)[1] - warp_forward(w, j, y - h)[1]) / (2h)
            @test exp(warp_forward(w, j, y)[2]) ≈ fd rtol = 1e-5
        end
    end
    return nothing
end

#=
The M-step's analytic parameter gradient against ForwardDiff. This is the test
that matters most: every other spline property is enforced by construction, but
a wrong gradient would just make EM converge somewhere else, quietly.

The sample grid deliberately includes points on the identity tails, where the
gradient must be exactly zero (`lo` / `hi` are fixed, so no parameter moves a
tail point).
=#
#=
A warp whose storage is in the AD element type. `MonotonicWarp(lo, hi; ...)`
allocates `Float64` arrays, which a `ForwardDiff.Dual` cannot be written into,
so the differentiated objective rebuilds the container in the dual type and
lets `warp_unpack!` fill it.
=#
function sg_warp_of_eltype(::Type{S}, lo, hi, K::Int, min_bin) where {S}
    p = length(lo)
    return SSD.MonotonicWarp{S}(
        S.(lo),
        S.(hi),
        zeros(S, K, p),
        zeros(S, K, p),
        zeros(S, K - 1, p),
        zeros(S, K + 1, p),
        zeros(S, K + 1, p),
        zeros(S, K + 1, p),
        zeros(S, K, p),
        zeros(S, K, p),
        S(min_bin),
    )
end

function test_warp_gradient_matches_forwarddiff()
    rng = StableRNG(12)
    lo = [-2.0, -1.0, 0.5]
    hi = [2.0, 3.0, 5.0]
    K, p = 5, 3
    w = MonotonicWarp(lo, hi; n_bins=K)
    nθ = warp_nparams(w)
    @test nθ == p * (3K - 1)
    θ = 0.7 .* randn(rng, nθ)
    SSD.warp_unpack!(w, θ)

    ndata = 24
    Y = hcat([lo .+ (hi .- lo) .* rand(rng, p) for _ in 1:ndata]...)
    Y[1, 1] = lo[1] - 0.6                     # below the interval
    Y[2, 2] = hi[2] + 0.3                     # above it
    Y[3, 3] = lo[3]                           # exactly on the lower knot
    ω = randn(rng, p, ndata)

    # ω·g(y) + log g'(y), summed — the per-sample integrand the M-step accumulates.
    function obj(θv)
        wd = sg_warp_of_eltype(eltype(θv), lo, hi, K, w.min_bin)
        SSD.warp_unpack!(wd, θv)
        S = eltype(θv)
        tot = zero(S)
        for t in 1:ndata, j in 1:p
            z, ℓ = warp_forward(wd, j, S(Y[j, t]))
            tot += ω[j, t] * z + ℓ
        end
        return tot
    end

    buf = SSD.WarpGradBuffers(w)
    SSD.reset!(buf)
    for t in 1:ndata, j in 1:p
        SSD.warp_accumulate_grad!(buf, w, j, Y[j, t], ω[j, t])
    end
    G = zeros(nθ)
    SSD.warp_knot_grad_to_theta!(G, w, buf)

    @test G ≈ ForwardDiff.gradient(obj, θ) rtol = 1e-8
    return nothing
end

# The log-Jacobian seed belongs to the *sample*, not to a mixture component, so
# a K-component accumulation must not count it K times.
function test_warp_gradient_logjac_counted_once()
    rng = StableRNG(13)
    w = MonotonicWarp([-1.0], [1.0]; n_bins=4)
    SSD.warp_unpack!(w, 0.5 .* randn(rng, warp_nparams(w)))
    y, ω = 0.3, -0.7

    both = SSD.WarpGradBuffers(w)
    SSD.reset!(both)
    SSD.warp_accumulate_grad!(both, w, 1, y, ω, true)

    split = SSD.WarpGradBuffers(w)
    SSD.reset!(split)
    SSD.warp_accumulate_grad!(split, w, 1, y, ω / 2, true)
    SSD.warp_accumulate_grad!(split, w, 1, y, ω / 2, false)

    @test both.gX ≈ split.gX
    @test both.gY ≈ split.gY
    @test both.gD ≈ split.gD
    return nothing
end

#=
The warp M-step objective is written for a mixture of Gaussian components — that
is what an `SLDS` needs, with the regime responsibilities as weights. Check its
gradient by central differences, including a component whose responsibility is
exactly zero (skipped in the loop, so the log-Jacobian seeding must not depend
on which component runs first) and a sample on the identity tail.
=#
function test_spline_mixture_objective_gradient()
    rng = StableRNG(16)
    p, K, N, Ti, nb = 3, 3, 4, 18, 5
    lo, hi = fill(-2.0, p), fill(3.0, p)
    om = SplineGaussianObservationModel(
        randn(rng, p, SG_LATENT),
        Matrix(0.4I, p, p),
        zeros(p);
        bounds=(lo, hi),
        n_bins=nb,
        spline_ridge=0.07,
    )
    Y = [lo .+ (hi .- lo) .* rand(rng, p, Ti) for _ in 1:N]
    Y[1][1, 1] = lo[1] - 0.5
    mu = [[randn(rng, p, Ti) for _ in 1:N] for _ in 1:K]

    gam = [[rand(rng, Ti) for _ in 1:N] for _ in 1:K]
    for n in 1:N, t in 1:Ti
        tot = sum(gam[c][n][t] for c in 1:K)
        for c in 1:K
            gam[c][n][t] /= tot
        end
    end
    gam[2][1][1] = 0.0
    tot = gam[1][1][1] + gam[3][1][1]
    gam[1][1][1] /= tot
    gam[3][1][1] /= tot

    Rs = [Matrix(Diagonal(0.2 .+ rand(rng, p))) for _ in 1:K]

    function check(ctx, nθ, θ)
        G = zeros(nθ)
        SSD._spline_fg!(G, θ, ctx)
        Gn = similar(G)
        for i in 1:nθ
            h = 1e-6
            θp = copy(θ)
            θp[i] += h
            θm = copy(θ)
            θm[i] -= h
            Gn[i] =
                (SSD._spline_fg!(nothing, θp, ctx) - SSD._spline_fg!(nothing, θm, ctx)) /
                (2h)
        end
        return maximum(abs.(G .- Gn)) / max(1.0, maximum(abs.(Gn)))
    end

    nθ = warp_nparams(om.warp)
    θ = 0.6 .* randn(rng, nθ)

    ctx = SSD._SplineMStepCtx(om, Y, mu; gamma=gam)
    for c in 1:K
        ctx.Rchol[c] = cholesky(Symmetric(Rs[c]))
    end
    @test check(ctx, nθ, θ) < 1e-6

    # One component with no weights is the ordinary single-Gaussian objective.
    ctx1 = SSD._SplineMStepCtx(om, Y, mu[1])
    ctx1.Rchol[1] = cholesky(Symmetric(Rs[1]))
    @test check(ctx1, nθ, θ) < 1e-6
    return nothing
end

function test_warp_bounds_and_construction_errors()
    Y = [1.0 3.0; -2.0 0.0]
    lo, hi = warp_bounds(Y, 0.1)
    @test lo ≈ [1.0 - 0.2, -2.0 - 0.2]
    @test hi ≈ [3.0 + 0.2, 0.0 + 0.2]
    # A constant channel still yields a usable (unit-wide) interval.
    lo2, hi2 = warp_bounds([5.0 5.0; 0.0 1.0])
    @test hi2[1] > lo2[1]

    @test_throws ArgumentError MonotonicWarp([0.0], [1.0]; n_bins=1)
    @test_throws ArgumentError MonotonicWarp([0.0], [0.0]; n_bins=4)
    @test_throws ArgumentError MonotonicWarp([0.0], [1.0]; n_bins=4, min_bin=0.5)
    return nothing
end

function test_warp_pack_roundtrip()
    rng = StableRNG(14)
    w = MonotonicWarp([-1.0, -3.0], [1.0, 4.0]; n_bins=5)
    θ = randn(rng, warp_nparams(w))
    SSD.warp_unpack!(w, θ)
    out = zeros(warp_nparams(w))
    @test SSD.warp_pack!(out, w) ≈ θ

    w2 = MonotonicWarp([-1.0, -3.0], [1.0, 4.0]; n_bins=5)
    copy_warp!(w2, w)
    @test w2.θw == w.θw && w2.θh == w.θh && w2.θd == w.θd
    @test w2.pX ≈ w.pX && w2.pY ≈ w.pY && w2.dYdX ≈ w.dYdX
    return nothing
end

function test_warp_apply_block()
    rng = StableRNG(15)
    w = MonotonicWarp([-2.0, -2.0], [2.0, 2.0]; n_bins=5)
    SSD.warp_unpack!(w, 0.6 .* randn(rng, warp_nparams(w)))
    Y = 3 .* randn(rng, 2, 30)
    Z = similar(Y)
    total = warp_apply!(Z, w, Y)
    @test total ≈ sum(warp_forward(w, j, Y[j, t])[2] for t in axes(Y, 2), j in 1:2)
    back = similar(Y)
    warp_unapply!(back, w, Z)
    @test back ≈ Y atol = 1e-8
    @test_throws DimensionMismatch warp_apply!(zeros(3, 30), w, zeros(3, 30))
    return nothing
end

# ============================================================================
# Construction, validation and printing
# ============================================================================

function test_spline_construction_and_validation()
    rng = StableRNG(21)
    p = 4
    Y = randn(rng, p, 50)
    om = SplineGaussianObservationModel(
        randn(rng, p, SG_LATENT), Matrix(1.0I, p, p), zeros(p); y=Y, n_bins=5
    )
    lds = LinearDynamicalSystem(sg_state_model(), om)
    @test lds.obs_dim == p
    @test length(lds.fit_bool) == 7          # x0 P0 A Q | C R spline
    @test warp_channels(om.warp) == p
    @test is_identity_warp(om.warp)
    @test om.R_structure === :diagonal
    @test om.R_floor > 0                     # the constructor's numerical floor

    # Exactly one source for the knot interval.
    @test_throws ArgumentError SplineGaussianObservationModel(
        randn(rng, p, SG_LATENT), Matrix(1.0I, p, p), zeros(p)
    )
    @test_throws ArgumentError SplineGaussianObservationModel(
        randn(rng, p, SG_LATENT),
        Matrix(1.0I, p, p),
        zeros(p);
        y=Y,
        bounds=(fill(-1.0, p), fill(1.0, p)),
    )
    @test_throws ArgumentError SplineGaussianObservationModel(
        randn(rng, p, SG_LATENT), Matrix(1.0I, p, p), zeros(p); y=Y, R_structure=:banded
    )

    # A warp of the wrong width is caught by `validate_LDS`.
    bad = SplineGaussianObservationModel(
        randn(rng, p, SG_LATENT), Matrix(1.0I, p, p), zeros(p); y=Y, n_bins=5
    )
    bad.warp = MonotonicWarp(zeros(p + 1), ones(p + 1); n_bins=5)
    @test_throws DimensionMismatchError LinearDynamicalSystem(sg_state_model(), bad)
    return nothing
end

function test_spline_fit_bool_keyword_form()
    rng = StableRNG(22)
    p = 3
    Y = randn(rng, p, 40)
    om = SplineGaussianObservationModel(
        randn(rng, p, SG_LATENT), Matrix(1.0I, p, p), zeros(p); y=Y, n_bins=4
    )
    lds = LinearDynamicalSystem(
        sg_state_model(), om; fit_bool=(C=true, R=false, spline=false)
    )
    @test lds.fit_bool == Bool[1, 1, 1, 1, 1, 0, 0]
    @test_throws ArgumentError LinearDynamicalSystem(
        sg_state_model(), om; fit_bool=(warp=false,)
    )
    return nothing
end

function test_spline_show()
    rng = StableRNG(23)
    p = 3
    Y = randn(rng, p, 30)
    lds = sg_init_model(Y; p=p, n_bins=4)
    out = sprint(show, lds.obs_model)
    @test occursin("Spline-Gaussian Observation Model", out)
    @test occursin("diagonal", out)
    @test occursin("currently the identity", out)
    @test occursin("warp (spline)", sprint(show, lds))
    return nothing
end

# ============================================================================
# Reduction to the linear model
# ============================================================================

#=
The identity warp is exactly the linear model, so with the spline frozen a
spline fit must reproduce a Gaussian LDS fit parameter for parameter. This is
what makes the emission a strict generalization rather than a different model
that happens to be close.
=#
function test_spline_frozen_warp_matches_gaussian()
    rng = StableRNG(31)
    p = 4
    lds_true = sg_true_model(; p=p, seed=3)
    _, Y = rand(StableRNG(32), lds_true, fill(60, 6))

    C0 = randn(rng, p, SG_LATENT)
    R0 = Matrix(1.0I, p, p)
    d0 = zeros(p)

    om_s = SplineGaussianObservationModel(
        copy(C0), copy(R0), copy(d0); y=Y, n_bins=5, R_structure=:full, spline_ridge=0.0
    )
    lds_s = LinearDynamicalSystem(sg_state_model(; θ=0.1), om_s; fit_bool=(spline=false,))
    el_s = fit!(lds_s, Y; max_iter=25, progress=false)

    om_g = GaussianObservationModel(copy(C0), copy(R0), copy(d0))
    lds_g = LinearDynamicalSystem(sg_state_model(; θ=0.1), om_g)
    el_g = fit!(lds_g, Y; max_iter=25, progress=false)

    @test is_identity_warp(lds_s.obs_model.warp)
    @test el_s ≈ el_g rtol = 1e-10
    @test lds_s.obs_model.C ≈ lds_g.obs_model.C rtol = 1e-10
    @test lds_s.obs_model.R ≈ lds_g.obs_model.R rtol = 1e-10
    @test lds_s.state_model.A ≈ lds_g.state_model.A rtol = 1e-10
    return nothing
end

function test_spline_identity_warp_scores_as_gaussian()
    rng = StableRNG(33)
    p = 3
    Y = randn(rng, p, 45)
    C = randn(rng, p, SG_LATENT)
    R = Matrix(Diagonal(0.3 .+ rand(rng, p)))
    d = randn(rng, p)

    lds_s = LinearDynamicalSystem(
        sg_state_model(),
        SplineGaussianObservationModel(C, R, d; y=Y, n_bins=4, spline_ridge=0.0),
    )
    lds_g = LinearDynamicalSystem(sg_state_model(), GaussianObservationModel(C, R, d))

    @test elbo(lds_s, Y) ≈ elbo(lds_g, Y) rtol = 1e-10
    @test loglikelihood(lds_s, Y) ≈ loglikelihood(lds_g, Y) rtol = 1e-10
    xs, Ps = smooth(lds_s, Y)
    xg, Pg = smooth(lds_g, Y)
    @test xs ≈ xg rtol = 1e-10
    @test Ps ≈ Pg rtol = 1e-10
    return nothing
end

# ============================================================================
# Log-density bookkeeping
# ============================================================================

#=
`z` moves every iteration, so the Gaussian part alone is not a likelihood of
anything fixed. These two identities are what make the reported numbers mean
something on the observation scale.
=#
function test_spline_elbo_equals_loglikelihood()
    lds_true = sg_true_model(; p=4, seed=4)
    _, Y = rand(StableRNG(41), lds_true, fill(70, 5))
    lds = sg_init_model(Y; p=4, seed=4, n_bins=5)
    fit!(lds, Y; max_iter=12, progress=false)
    # Exact smoother + no priors + zero ridge ⇒ the bound is tight.
    @test elbo(lds, Y) ≈ loglikelihood(lds, Y) rtol = 1e-9
    return nothing
end

function test_spline_trial_elbos_sum()
    lds_true = sg_true_model(; p=4, seed=5)
    _, Y = rand(StableRNG(42), lds_true, [50, 65, 40])
    lds = sg_init_model(Y; p=4, seed=5, n_bins=5)
    fit!(lds, Y; max_iter=10, progress=false)
    te = trial_elbos(lds, Y)
    @test length(te) == 3
    @test sum(te) ≈ elbo(lds, Y) rtol = 1e-9
    return nothing
end

function test_spline_logprior_enters_elbo()
    lds_true = sg_true_model(; p=3, seed=6)
    _, Y = rand(StableRNG(43), lds_true, fill(60, 4))
    lds = sg_init_model(Y; p=3, seed=6, n_bins=4)
    fit!(lds, Y; max_iter=10, progress=false)
    ll = loglikelihood(lds, Y)
    @test elbo(lds, Y) ≈ ll rtol = 1e-9          # ridge is 0 here
    lds.obs_model.spline_ridge = 0.5
    θ = zeros(warp_nparams(lds.obs_model.warp))
    SSD.warp_pack!(θ, lds.obs_model.warp)
    @test elbo(lds, Y) ≈ ll - 0.25 * sum(abs2, θ) rtol = 1e-9
    return nothing
end

# ============================================================================
# The ECM algorithm
# ============================================================================

function test_spline_elbo_monotone()
    lds_true = sg_true_model(; p=5, seed=7)
    _, Y = rand(StableRNG(51), lds_true, fill(80, 8))
    lds = sg_init_model(Y; p=5, seed=7, n_bins=6)
    el = fit!(lds, Y; max_iter=40, tol=1e-10, progress=false)
    @test length(el) > 1
    # ECM: both conditional maximizations increase the same Q, and the warp step
    # is accepted only when it improves — so this is monotone, not merely
    # non-decreasing on average.
    @test sg_nondecreasing(el)
    @test el[end] > el[1]
    return nothing
end

function test_spline_recovers_warp_and_beats_linear()
    p = 5
    lds_true = sg_true_model(; p=p, seed=8)
    _, Yall = rand(StableRNG(52), lds_true, fill(100, 14))
    Ytr, Yte = Yall[1:10], Yall[11:end]

    lds = sg_init_model(Ytr; p=p, seed=8, n_bins=6)
    fit!(lds, Ytr; max_iter=60, tol=1e-10, progress=false)

    rng = StableRNG(53)
    lds_lin = LinearDynamicalSystem(
        GaussianStateModel(
            Matrix(0.9I, SG_LATENT, SG_LATENT),
            Matrix(0.1I, SG_LATENT, SG_LATENT),
            zeros(SG_LATENT),
            zeros(SG_LATENT),
            Matrix(1.0I, SG_LATENT, SG_LATENT),
        ),
        GaussianObservationModel(randn(rng, p, SG_LATENT), Matrix(1.0I, p, p), zeros(p)),
    )
    fit!(lds_lin, Ytr; max_iter=60, tol=1e-10, progress=false)

    # Held-out, on the observation scale, so the two are directly comparable.
    @test elbo(lds, Yte) > elbo(lds_lin, Yte)

    # The warp itself is identified (the pinned endpoints remove the affine
    # degeneracy), so compare the fitted map to the true one over the observed
    # range of each channel.
    for j in 1:p
        obs = vcat([yt[j, :] for yt in Ytr]...)
        grid = collect(range(minimum(obs), maximum(obs); length=120))
        gt = [warp_forward(lds_true.obs_model.warp, j, v)[1] for v in grid]
        gf = [warp_forward(lds.obs_model.warp, j, v)[1] for v in grid]
        @test cor(gt, gf) > 0.9
    end
    @test !is_identity_warp(lds.obs_model.warp)
    return nothing
end

#=
The central claim: smoothing a spline model is *exactly* smoothing the
corresponding Gaussian model on the pre-embedded observations. The warp enters
the emission density only through a term that does not involve `x`, so the
posterior over the latent path is untouched by it. Check that against an
explicitly embedded dataset rather than trusting the driver.
=#
function test_spline_smooth_equals_gaussian_on_embedded()
    rng = StableRNG(69)
    p = 4
    lds_true = sg_true_model(; p=p, seed=18)
    _, Y = rand(StableRNG(70), lds_true, [45, 60])

    om = SplineGaussianObservationModel(
        randn(rng, p, SG_LATENT),
        Matrix(Diagonal(0.2 .+ rand(rng, p))),
        randn(rng, p);
        y=Y,
        n_bins=5,
        spline_ridge=0.0,
    )
    SSD.warp_unpack!(om.warp, 0.7 .* randn(rng, warp_nparams(om.warp)))
    lds = LinearDynamicalSystem(sg_state_model(), om)

    # Embed by hand, and build the plain Gaussian model over the same arrays.
    Z = [similar(yt) for yt in Y]
    logjac = sum(warp_apply!(Z[n], om.warp, Y[n]) for n in eachindex(Y))
    lds_g = LinearDynamicalSystem(
        lds.state_model, GaussianObservationModel(om.C, om.R, om.d)
    )

    xs, Ps = smooth(lds, Y)
    xg, Pg = smooth(lds_g, Z)
    for n in eachindex(Y)
        @test xs[n] ≈ xg[n] rtol = 1e-12
        @test Ps[n] ≈ Pg[n] rtol = 1e-12
    end

    # And the log-densities differ by exactly the change of variables.
    @test loglikelihood(lds, Y) ≈ loglikelihood(lds_g, Z) + logjac rtol = 1e-10
    @test elbo(lds, Y) ≈ elbo(lds_g, Z) + logjac rtol = 1e-10
    @test trial_elbos(lds, Y) ≈
        trial_elbos(lds_g, Z) .+ [
        sum(warp_forward(om.warp, j, Y[n][j, t])[2] for t in axes(Y[n], 2), j in 1:p) for
        n in eachindex(Y)
    ] rtol = 1e-10
    return nothing
end

#=
A partial maximization of the warp objective is still a valid generalized-EM
step, so a one-iteration L-BFGS budget must keep the trace monotone — it only
makes it climb more slowly.
=#
function test_spline_partial_warp_maximization_still_monotone()
    p = 4
    lds_true = sg_true_model(; p=p, seed=19)
    _, Y = rand(StableRNG(71), lds_true, fill(60, 6))

    slow = sg_init_model(Y; p=p, seed=19, n_bins=5)
    el_slow = fit!(slow, Y; max_iter=30, tol=1e-12, spline_iters=1, progress=false)
    @test sg_nondecreasing(el_slow)
    @test !is_identity_warp(slow.obs_model.warp)

    fast = sg_init_model(Y; p=p, seed=19, n_bins=5)
    el_fast = fit!(fast, Y; max_iter=30, tol=1e-12, spline_iters=25, progress=false)
    @test sg_nondecreasing(el_fast)
    # More budget per step cannot end up worse at the same iteration count.
    @test el_fast[end] >= el_slow[end] - 1e-6 * abs(el_fast[end])
    return nothing
end

function test_warp_bin_lookup()
    w = MonotonicWarp([-2.0], [2.0]; n_bins=4)
    K = warp_bins(w)
    @test SSD.warp_bin(w, 1, -2.5) == 0          # below the interval
    @test SSD.warp_bin(w, 1, 2.5) == 0           # above it
    @test SSD.warp_bin(w, 1, -2.0) == 0          # the endpoints are the tails
    @test SSD.warp_bin(w, 1, 2.0) == 0
    for k in 1:K
        mid = (w.pX[k, 1] + w.pX[k + 1, 1]) / 2
        @test SSD.warp_bin(w, 1, mid) == k
        # A point exactly on an interior knot belongs to the bin it opens.
        k == 1 || @test SSD.warp_bin(w, 1, w.pX[k, 1]) == k
    end
    return nothing
end

function test_spline_R_structure()
    p = 4
    lds_true = sg_true_model(; p=p, seed=9)
    _, Y = rand(StableRNG(54), lds_true, fill(70, 6))

    lds_d = sg_init_model(Y; p=p, seed=9, n_bins=5)
    fit!(lds_d, Y; max_iter=15, progress=false)
    R = lds_d.obs_model.R
    @test R ≈ Diagonal(diag(R))
    @test all(>(0), diag(R))

    lds_f = sg_init_model(Y; p=p, seed=9, n_bins=5, R_structure=:full)
    fit!(lds_f, Y; max_iter=15, progress=false)
    @test !(lds_f.obs_model.R ≈ Diagonal(diag(lds_f.obs_model.R)))
    return nothing
end

#=
Freezing the warp mid-model must leave it untouched while everything else still
fits, and freezing `R` must leave `R` at its initial value.
=#
function test_spline_fit_bool_freezes()
    p = 3
    lds_true = sg_true_model(; p=p, seed=10)
    _, Y = rand(StableRNG(55), lds_true, fill(50, 5))

    lds = sg_init_model(Y; p=p, seed=10, n_bins=4)
    R0 = copy(lds.obs_model.R)
    lds_frozen = LinearDynamicalSystem(
        lds.state_model, lds.obs_model; fit_bool=(spline=false, R=false)
    )
    fit!(lds_frozen, Y; max_iter=10, progress=false)
    @test is_identity_warp(lds_frozen.obs_model.warp)
    @test lds_frozen.obs_model.R ≈ R0

    # `spline_iters = 0` is the other way to freeze it.
    lds2 = sg_init_model(Y; p=p, seed=11, n_bins=4)
    fit!(lds2, Y; max_iter=8, spline_iters=0, progress=false)
    @test is_identity_warp(lds2.obs_model.warp)
    return nothing
end

function test_spline_inputs_and_ragged_trials()
    rng = StableRNG(56)
    p, uy_dim = 3, 2
    tsteps = [40, 55, 30]
    Y = [randn(rng, p, Ti) for Ti in tsteps]
    V = [randn(rng, uy_dim, Ti) for Ti in tsteps]

    om = SplineGaussianObservationModel(
        randn(rng, p, SG_LATENT),
        Matrix(1.0I, p, p),
        zeros(p);
        y=Y,
        n_bins=4,
        D=randn(rng, p, uy_dim),
        spline_ridge=0.0,
    )
    lds = LinearDynamicalSystem(sg_state_model(), om)
    @test lds.uy_dim == uy_dim
    el = fit!(lds, Y; uy=V, max_iter=12, progress=false)
    @test sg_nondecreasing(el)
    @test elbo(lds, Y; uy=V) ≈ loglikelihood(lds, Y; uy=V) rtol = 1e-9

    xs, Ps = smooth(lds, Y; uy=V)
    @test length(xs) == 3
    @test size(xs[2]) == (SG_LATENT, 55)
    return nothing
end

function test_spline_single_trial_matrix_shapes()
    rng = StableRNG(57)
    p = 3
    Y = randn(rng, p, 60)
    lds = sg_init_model(Y; p=p, seed=12, n_bins=4)
    fit!(lds, Y; max_iter=8, progress=false)
    x, P = smooth(lds, Y)
    @test x isa Matrix && size(x) == (SG_LATENT, 60)
    @test P isa Array{Float64,3} && size(P) == (SG_LATENT, SG_LATENT, 60)
    return nothing
end

function test_spline_sampling_roundtrip()
    p = 4
    lds = sg_true_model(; p=p, seed=13)
    x, y = rand(StableRNG(58), lds, 200)
    @test size(x) == (SG_LATENT, 200)
    @test size(y) == (p, 200)

    xs, ys = rand(StableRNG(59), lds, [40, 60])
    @test length(ys) == 2 && size(ys[2]) == (p, 60)

    # Drawing then embedding must land back in the Gaussian the sampler used:
    # g(y) should have roughly the residual scale R around C x + d.
    z = similar(y)
    warp_apply!(z, lds.obs_model.warp, y)
    resid = z .- (lds.obs_model.C * x .+ lds.obs_model.d)
    @test isapprox(var(vec(resid)), lds.obs_model.R[1, 1]; rtol=0.5)
    return nothing
end

function test_spline_holdout_and_early_stopping()
    p = 4
    lds_true = sg_true_model(; p=p, seed=14)
    _, Yall = rand(StableRNG(60), lds_true, fill(70, 10))
    Ytr, Yte = Yall[1:7], Yall[8:end]

    lds = sg_init_model(Ytr; p=p, seed=14, n_bins=5)
    tr = fit!(lds, Ytr; y_test=Yte, max_iter=20, test_every=2, progress=false)
    @test tr isa SSD.FitTrace
    @test length(tr.test) == length(tr.test_iters)
    @test all(isfinite, tr.test)
    # A FitTrace behaves as the training-ELBO vector.
    @test sg_nondecreasing(tr.train)
    return nothing
end

#=
`rtol` reaches the ECM loops as it does the Gaussian ones: a loose relative
tolerance stops a fit the absolute one alone keeps running, on the same path.
Both the standalone driver and a warped composite member are checked, since the
composite enters through the Gaussian composite `fit!`.
=#
function test_spline_rtol_stops_early()
    p = 4
    lds_true = sg_true_model(; p=p, seed=18)
    _, Y = rand(StableRNG(68), lds_true, fill(60, 6))

    full = sg_init_model(Y; p=p, seed=18, n_bins=5)
    el_full = fit!(full, Y; max_iter=30, tol=1e-12, progress=false)
    loose = sg_init_model(Y; p=p, seed=18, n_bins=5)
    el_loose = fit!(loose, Y; max_iter=30, tol=1e-12, rtol=1e-2, progress=false)
    @test length(el_full) == 30
    @test 1 < length(el_loose) < length(el_full)
    @test el_loose ≈ el_full[1:length(el_loose)] rtol = 1e-10

    k = SG_LATENT
    function composite()
        om = SplineGaussianObservationModel(
            randn(StableRNG(69), p, k),
            Matrix(1.0I, p, p),
            zeros(p);
            y=Y,
            n_bins=5,
            spline_ridge=0.0,
        )
        aux = GaussianObservationModel(
            randn(StableRNG(70), 2, k), Matrix(1.0I, 2, 2), zeros(2)
        )
        return LinearDynamicalSystem(sg_state_model(), (kin=om, aux=aux))
    end
    Yc = (kin=Y, aux=[randn(StableRNG(71), 2, size(y, 2)) for y in Y])
    el_full = fit!(composite(), Yc; max_iter=30, tol=1e-12, progress=false)
    el_loose = fit!(composite(), Yc; max_iter=30, tol=1e-12, rtol=1e-2, progress=false)
    @test length(el_full) == 30
    @test 1 < length(el_loose) < length(el_full)
    return nothing
end

function test_spline_grouping_is_rejected()
    rng = StableRNG(61)
    p = 3
    Y = [randn(rng, p, 40) for _ in 1:4]
    lds = sg_init_model(Y; p=p, seed=15, n_bins=4)
    set_depends_on!(lds.obs_model, (C=[1, 1, 2, 2],))
    @test_throws ArgumentError fit!(lds, Y; max_iter=2, progress=false)
    @test_throws ArgumentError elbo(lds, Y)
    set_depends_on!(lds.obs_model, nothing)

    #=
    Labels passed at the call rather than declared on the model are refused
    too, at every composite entry point, on both the quadratic and the Laplace
    routes -- not silently dropped.
    =#
    k = SG_LATENT
    labels = (C=[1, 1, 2, 2],)
    function kin()
        return SplineGaussianObservationModel(
            randn(StableRNG(62), p, k), Matrix(1.0I, p, p), zeros(p); y=Y, n_bins=4
        )
    end
    quad = LinearDynamicalSystem(
        sg_state_model(),
        (
            kin=kin(),
            aux=GaussianObservationModel(randn(rng, 2, k), Matrix(1.0I, 2, 2), zeros(2)),
        ),
    )
    Yq = (kin=Y, aux=[randn(rng, 2, 40) for _ in 1:4])
    lap = LinearDynamicalSystem(
        sg_state_model(),
        (kin=kin(), spk=PoissonObservationModel(randn(rng, 2, k), zeros(2))),
    )
    Yl = (kin=Y, spk=[Float64.(rand(rng, 0:2, 2, 40)) for _ in 1:4])
    for (m, y) in ((quad, Yq), (lap, Yl))
        @test_throws ArgumentError fit!(m, y; max_iter=1, progress=false, depends_on=labels)
        @test_throws ArgumentError fit!(
            m, y; max_iter=1, progress=false, y_test=y, depends_on_test=labels
        )
        @test_throws ArgumentError smooth(m, y; depends_on=labels)
        @test_throws ArgumentError elbo(m, y; depends_on=labels)
    end
    @test_throws ArgumentError loglikelihood(quad, Yq; depends_on=labels)
    # Without labels the same calls go through.
    @test isfinite(loglikelihood(quad, Yq))
    @test isfinite(elbo(lap, Yl))
    return nothing
end

#=
The whole implementation rests on the shadow emission sharing the spline model's
arrays: every M-step write goes through a `GaussianObservationModel` built over
the same `C`, `R`, `d`, `D`, and is expected to land back here with no copy.
Assert that directly rather than only through a fit.
=#
function test_spline_gaussian_shadow_shares_arrays()
    rng = StableRNG(62)
    p = 3
    Y = randn(rng, p, 30)
    om = SplineGaussianObservationModel(
        randn(rng, p, SG_LATENT),
        Matrix(1.0I, p, p),
        zeros(p);
        y=Y,
        n_bins=4,
        D=randn(rng, p, 2),
    )
    g = SSD._gaussian_shadow(om)
    @test g isa GaussianObservationModel
    @test g.C === om.C
    @test g.R === om.R
    @test g.d === om.d
    @test g.D === om.D
    g.C[1, 1] = 99.0
    g.R[2, 2] = 7.0
    @test om.C[1, 1] == 99.0
    @test om.R[2, 2] == 7.0

    # And at the system level, including the truncated fit_bool.
    lds = LinearDynamicalSystem(
        sg_state_model(), om; fit_bool=(C=false, R=true, spline=false)
    )
    glds = SSD._gaussian_shadow(lds)
    @test glds.state_model === lds.state_model
    @test length(glds.fit_bool) == 6
    @test glds.fit_bool == lds.fit_bool[1:6]
    return nothing
end

#=
The diagonal-restricted IW MAP has the same denominator as the unrestricted one,
so when the scatter and the prior scale are both diagonal the two updates must
agree exactly. That pins the formula against the package's own `iw_map` instead
of against a transcription of it.
=#
function test_spline_diagonal_R_matches_iw_map()
    rng = StableRNG(63)
    p = 4
    Y = randn(rng, p, 30)
    C = randn(rng, p, SG_LATENT)
    d = zeros(p)
    S_diag = Diagonal(0.5 .+ rand(rng, p))
    N = 17.0

    function fitted_R(structure, prior)
        om = SplineGaussianObservationModel(
            copy(C),
            Matrix(1.0I, p, p),
            copy(d);
            y=Y,
            n_bins=4,
            R_structure=structure,
            R_prior=prior,
            R_floor=0.0,
        )
        glds = SSD._gaussian_shadow(LinearDynamicalSystem(sg_state_model(), om))
        S = Matrix(S_diag)
        if structure === :diagonal
            SSD._finalize_R_diag!(glds, S, N)
        else
            SSD._finalize_R!(glds, S, N)
        end
        return copy(om.R)
    end

    @test fitted_R(:diagonal, nothing) ≈ Matrix(S_diag) ./ N
    @test fitted_R(:diagonal, nothing) ≈ fitted_R(:full, nothing)

    Ψ = Matrix(Diagonal(0.2 .+ rand(rng, p)))
    ν = 9.0
    prior = IWPrior(; Ψ=Ψ, ν=ν)
    Rd = fitted_R(:diagonal, prior)
    @test Rd ≈ fitted_R(:full, prior)
    @test diag(Rd) ≈ (diag(Ψ) .+ diag(S_diag)) ./ (ν + N + p + 1)

    # With an off-diagonal scatter the two genuinely differ, and the diagonal
    # update ignores the off-diagonal mass rather than folding it in.
    S_full = Matrix(S_diag) .+ 0.1
    om = SplineGaussianObservationModel(
        copy(C), Matrix(1.0I, p, p), copy(d); y=Y, n_bins=4, R_floor=0.0
    )
    glds = SSD._gaussian_shadow(LinearDynamicalSystem(sg_state_model(), om))
    SSD._finalize_R_diag!(glds, copy(S_full), N)
    @test om.R ≈ Diagonal(diag(S_full) ./ N)
    return nothing
end

#=
The likelihood is unbounded above (a warp that interpolates the prediction
drives a channel's residual to zero), so `R_floor` is the numerical stop. Check
it binds, keeps `R` factorizable, and says so.
=#
function test_spline_R_floor_binds_and_warns()
    rng = StableRNG(64)
    p = 3
    Y = randn(rng, p, 40)
    om = SplineGaussianObservationModel(
        randn(rng, p, SG_LATENT),
        Matrix(1.0I, p, p),
        zeros(p);
        y=Y,
        n_bins=4,
        R_floor=5.0,          # far above any residual these data can produce
    )
    lds = LinearDynamicalSystem(sg_state_model(), om)
    @test_logs (:warn,) match_mode = :any fit!(lds, Y; max_iter=3, progress=false)
    @test all(>=(5.0), diag(om.R))
    @test isposdef(om.R)

    #=
    Both keyword spellings must work. The defaults for `spline_ridge` and
    `R_floor` are taken through `eltype(C)` precisely because the
    unparameterized form binds no `T`, and writing them as `T(...)` made that
    form an `UndefVarError`.
    =#
    om_bare = SplineGaussianObservationModel(;
        C=randn(rng, p, SG_LATENT),
        R=Matrix(1.0I, p, p),
        d=zeros(p),
        warp=MonotonicWarp(fill(-1.0, p), fill(1.0, p); n_bins=4),
    )
    @test om_bare.spline_ridge == 1e-3
    @test om_bare.R_floor == 0.0
    @test om_bare.spline_ridge isa Float64
    @test om_bare.R_structure === :diagonal

    om_f32 = SplineGaussianObservationModel(;
        C=randn(rng, Float32, p, SG_LATENT),
        R=Matrix{Float32}(1.0I, p, p),
        d=zeros(Float32, p),
        warp=MonotonicWarp(fill(-1.0f0, p), fill(1.0f0, p); n_bins=4),
    )
    @test om_f32 isa SplineGaussianObservationModel{Float32}
    @test om_f32.spline_ridge isa Float32
    @test om_f32.R_floor isa Float32

    # A floor of 0 is off, and is the raw constructor's default.
    om0 = SplineGaussianObservationModel{Float64,Matrix{Float64},Vector{Float64}}(;
        C=randn(rng, p, SG_LATENT),
        R=Matrix(1.0I, p, p),
        d=zeros(p),
        warp=MonotonicWarp(fill(-1.0, p), fill(1.0, p); n_bins=4),
    )
    @test om0.R_floor == 0.0
    @test SSD._apply_R_floor!(copy(om0.R), 0.0) ≈ om0.R
    return nothing
end

function test_spline_three_dim_observations()
    p, Ti, N = 3, 40, 5
    lds_true = sg_true_model(; p=p, seed=16)
    _, Ylist = rand(StableRNG(65), lds_true, fill(Ti, N))
    Y3 = Array{Float64,3}(undef, p, Ti, N)
    for n in 1:N
        Y3[:, :, n] .= Ylist[n]
    end

    lds3 = sg_init_model(Y3; p=p, seed=16, n_bins=5)
    el3 = fit!(lds3, Y3; max_iter=10, progress=false)
    lds_v = sg_init_model(Ylist; p=p, seed=16, n_bins=5)
    el_v = fit!(lds_v, Ylist; max_iter=10, progress=false)

    # The two layouts are the same dataset, so the fits must agree exactly.
    @test el3 ≈ el_v rtol = 1e-12
    @test lds3.obs_model.C ≈ lds_v.obs_model.C rtol = 1e-12
    @test lds3.obs_model.warp.θh ≈ lds_v.obs_model.warp.θh rtol = 1e-10

    xs, Ps = smooth(lds3, Y3)
    @test length(xs) == N && size(xs[1]) == (SG_LATENT, Ti)
    @test length(trial_elbos(lds3, Y3)) == N
    return nothing
end

function test_spline_float32()
    rng = StableRNG(66)
    p, Ti = 3, 40
    Y = [randn(rng, Float32, p, Ti) for _ in 1:3]
    om = SplineGaussianObservationModel(
        randn(rng, Float32, p, SG_LATENT),
        Matrix{Float32}(1.0I, p, p),
        zeros(Float32, p);
        y=Y,
        n_bins=4,
        spline_ridge=0.0f0,
    )
    @test om isa SplineGaussianObservationModel{Float32}
    @test om.warp isa MonotonicWarp{Float32}
    sm = GaussianStateModel(
        Matrix{Float32}(0.9I, SG_LATENT, SG_LATENT),
        Matrix{Float32}(0.1I, SG_LATENT, SG_LATENT),
        zeros(Float32, SG_LATENT),
        zeros(Float32, SG_LATENT),
        Matrix{Float32}(1.0I, SG_LATENT, SG_LATENT),
    )
    lds = LinearDynamicalSystem(sm, om)
    el = fit!(lds, Y; max_iter=6, progress=false)
    @test eltype(el) === Float32
    @test all(isfinite, el)
    @test eltype(om.warp.θh) === Float32
    @test eltype(om.R) === Float32
    return nothing
end

function test_spline_priors_shift_the_fit()
    p = 4
    lds_true = sg_true_model(; p=p, seed=17)
    _, Y = rand(StableRNG(67), lds_true, fill(60, 5))

    plain = sg_init_model(Y; p=p, seed=17, n_bins=5)
    fit!(plain, Y; max_iter=15, progress=false)

    #=
    A strong IW prior inflates `R`, with a floor that follows from the MAP
    formula alone: `R_jj = (Ψ_jj + S_jj)/(ν + N + p + 1) ≥ Ψ_jj/(ν + N + p + 1)`,
    where `N` is the total number of observed timesteps.
    =#
    Ψ = Matrix(50.0I, p, p)
    ν = 200.0
    prior = IWPrior(; Ψ=Ψ, ν=ν)
    shrunk = sg_init_model(Y; p=p, seed=17, n_bins=5, R_prior=prior)
    fit!(shrunk, Y; max_iter=15, progress=false)
    N = sum(size(yt, 2) for yt in Y)
    @test all(diag(shrunk.obs_model.R) .>= 50.0 / (ν + N + p + 1))
    @test all(diag(shrunk.obs_model.R) .> diag(plain.obs_model.R))

    #=
    And the prior's log-density must be exactly what separates the ELBO from the
    marginal log-likelihood — nothing else is set here, so the gap is the IW term
    alone.
    =#
    @test elbo(shrunk, Y) - loglikelihood(shrunk, Y) ≈
        SSD.iw_logprior_term(shrunk.obs_model.R, prior) rtol = 1e-9

    # The warp ridge shrinks the warp toward the identity.
    loose = sg_init_model(Y; p=p, seed=17, n_bins=5)
    fit!(loose, Y; max_iter=25, progress=false)
    tight = sg_init_model(Y; p=p, seed=17, n_bins=5)
    tight.obs_model.spline_ridge = 50.0
    fit!(tight, Y; max_iter=25, progress=false)
    norm_of(m) = sum(abs2, m.obs_model.warp.θh) + sum(abs2, m.obs_model.warp.θw)
    @test norm_of(tight) < norm_of(loose)
    return nothing
end

#=
An inverse-LQR state is fitted by its own *structural* M-step; the spline driver
runs the generic Gaussian state updates, which would silently discard the
symplectic constraint. Every entry point that could reach that combination must
refuse it instead.

`LQRStateModel`'s latent is the state-costate pair `[x; λ]`, so the process
noise and the emission are sized at `2n`, not `n`.
=#
function test_spline_lqr_state_model_rejected()
    rng = StableRNG(68)
    n, p = 2, 3
    d = 2n
    A = [0.96 0.07; -0.05 0.93]
    Sm = [0.06 0.01; 0.01 0.05]
    Qc = [0.25 0.04; 0.04 0.18]
    Σ = Matrix(Diagonal(fill(0.03, d)))
    sm = LQRStateModel(A, Sm, Qc, Σ)
    @test SSD._state_latent_dim(sm) == d

    Y = [randn(rng, p, 30) for _ in 1:3]
    om = SplineGaussianObservationModel(
        randn(rng, p, d), Matrix(1.0I, p, p), zeros(p); y=Y, n_bins=4
    )
    lds = LinearDynamicalSystem(sm, om)
    @test_throws ArgumentError fit!(lds, Y; max_iter=2, progress=false)
    @test_throws ArgumentError elbo(lds, Y)
    @test_throws ArgumentError smooth(lds, Y)
    @test_throws ArgumentError trial_elbos(lds, Y)

    # The same combination inside a composite, which reaches the LQR drivers.
    om_c = SplineGaussianObservationModel(
        randn(rng, p, d), Matrix(1.0I, p, p), zeros(p); y=Y, n_bins=4
    )
    om_g = GaussianObservationModel(randn(rng, p, d), Matrix(1.0I, p, p), zeros(p))
    lds_c = LinearDynamicalSystem(sm, (warped=om_c, plain=om_g))
    Yc = (warped=Y, plain=[randn(rng, p, 30) for _ in 1:3])
    @test_throws ArgumentError fit!(lds_c, Yc; max_iter=2, progress=false)
    @test_throws ArgumentError elbo(lds_c, Yc)

    # An unwarped model with the same state is of course still fine.
    lds_ok = LinearDynamicalSystem(
        LQRStateModel(A, Sm, Qc, Σ),
        GaussianObservationModel(randn(rng, p, d), Matrix(1.0I, p, p), zeros(p)),
    )
    @test elbo(lds_ok, Y) isa Real
    return nothing
end

function test_warp_copy_shape_mismatch()
    a = MonotonicWarp([-1.0], [1.0]; n_bins=4)
    b = MonotonicWarp([-1.0], [1.0]; n_bins=6)
    c = MonotonicWarp([-1.0, -1.0], [1.0, 1.0]; n_bins=4)
    @test_throws DimensionMismatch copy_warp!(a, b)
    @test_throws DimensionMismatch copy_warp!(a, c)
    return nothing
end

# ============================================================================
# Composite emissions
# ============================================================================

function test_spline_composite_fit()
    rng = StableRNG(71)
    pk, pa, k = 3, 4, SG_LATENT
    warp_seed = StableRNG(72)

    om_k = SplineGaussianObservationModel(
        randn(warp_seed, pk, k),
        Matrix(0.05I, pk, pk),
        zeros(pk);
        bounds=(fill(-8.0, pk), fill(8.0, pk)),
        n_bins=5,
    )
    SSD.warp_unpack!(om_k.warp, 0.8 .* randn(warp_seed, warp_nparams(om_k.warp)))
    om_a = GaussianObservationModel(
        randn(warp_seed, pa, k), Matrix(0.05I, pa, pa), zeros(pa)
    )
    lds_true = LinearDynamicalSystem(sg_state_model(), (kin=om_k, aux=om_a))
    _, ytr = rand(StableRNG(73), lds_true, fill(80, 8))
    Y = (kin=[t.kin for t in ytr], aux=[t.aux for t in ytr])

    om_k0 = SplineGaussianObservationModel(
        randn(rng, pk, k),
        Matrix(1.0I, pk, pk),
        zeros(pk);
        y=Y.kin,
        n_bins=5,
        spline_ridge=0.0,
    )
    om_a0 = GaussianObservationModel(randn(rng, pa, k), Matrix(1.0I, pa, pa), zeros(pa))
    lds = LinearDynamicalSystem(
        GaussianStateModel(
            Matrix(0.9I, k, k), Matrix(0.1I, k, k), zeros(k), zeros(k), Matrix(1.0I, k, k)
        ),
        (kin=om_k0, aux=om_a0),
    )
    @test length(lds.fit_bool) == 4 + 3 + 2

    el = fit!(lds, Y; max_iter=30, tol=1e-10, progress=false)
    @test sg_nondecreasing(el)
    @test elbo(lds, Y) ≈ loglikelihood(lds, Y) rtol = 1e-8
    @test !is_identity_warp(lds.obs_model.kin.warp)
    # The warped member keeps its diagonal R; the plain Gaussian member does not.
    Rk = lds.obs_model.kin.R
    @test Rk ≈ Diagonal(diag(Rk))

    xs, Ps = smooth(lds, Y)
    @test length(xs) == 8

    for j in 1:pk
        obs = vcat([yt[j, :] for yt in Y.kin]...)
        grid = collect(range(minimum(obs), maximum(obs); length=100))
        gt = [warp_forward(om_k.warp, j, v)[1] for v in grid]
        gf = [warp_forward(lds.obs_model.kin.warp, j, v)[1] for v in grid]
        @test cor(gt, gf) > 0.9
    end
    return nothing
end

function test_spline_composite_trial_elbos()
    rng = StableRNG(78)
    pk, pa, k = 3, 3, SG_LATENT
    Y = (kin=[randn(rng, pk, 35) for _ in 1:4], aux=[randn(rng, pa, 35) for _ in 1:4])
    om_k = SplineGaussianObservationModel(
        randn(rng, pk, k),
        Matrix(1.0I, pk, pk),
        zeros(pk);
        y=Y.kin,
        n_bins=4,
        spline_ridge=0.0,
    )
    om_a = GaussianObservationModel(randn(rng, pa, k), Matrix(1.0I, pa, pa), zeros(pa))
    lds = LinearDynamicalSystem(sg_state_model(), (kin=om_k, aux=om_a))
    fit!(lds, Y; max_iter=10, progress=false)

    te = trial_elbos(lds, Y)
    @test length(te) == 4
    # Had `trial_elbos` scored the raw observations instead of the embedding,
    # this would miss the change-of-variables term entirely.
    @test sum(te) ≈ elbo(lds, Y) rtol = 1e-8
    return nothing
end

function test_spline_composite_fit_bool()
    rng = StableRNG(74)
    pk, pa, k = 3, 3, SG_LATENT
    Y = (kin=[randn(rng, pk, 40) for _ in 1:4], aux=[randn(rng, pa, 40) for _ in 1:4])
    om_k = SplineGaussianObservationModel(
        randn(rng, pk, k), Matrix(1.0I, pk, pk), zeros(pk); y=Y.kin, n_bins=4
    )
    om_a = GaussianObservationModel(randn(rng, pa, k), Matrix(1.0I, pa, pa), zeros(pa))
    lds = LinearDynamicalSystem(
        sg_state_model(), (kin=om_k, aux=om_a); fit_bool=(kin=(spline=false,),)
    )
    @test lds.fit_bool == Bool[1, 1, 1, 1, 1, 1, 0, 1, 1]
    fit!(lds, Y; max_iter=6, progress=false)
    @test is_identity_warp(lds.obs_model.kin.warp)
    return nothing
end

# A warped Gaussian member alongside a Poisson one: non-quadratic, so this runs
# on the Laplace driver rather than the single-step Newton one.
function test_spline_composite_with_poisson()
    rng = StableRNG(75)
    pk, pn, k = 3, 6, SG_LATENT
    seedrng = StableRNG(76)

    om_k = SplineGaussianObservationModel(
        randn(seedrng, pk, k),
        Matrix(0.05I, pk, pk),
        zeros(pk);
        bounds=(fill(-8.0, pk), fill(8.0, pk)),
        n_bins=5,
    )
    SSD.warp_unpack!(om_k.warp, 0.8 .* randn(seedrng, warp_nparams(om_k.warp)))
    om_n = PoissonObservationModel(0.6 .* randn(seedrng, pn, k), fill(1.2, pn))
    lds_true = LinearDynamicalSystem(sg_state_model(), (kin=om_k, spk=om_n))
    _, ytr = rand(StableRNG(77), lds_true, fill(70, 6))
    Y = (kin=[t.kin for t in ytr], spk=[t.spk for t in ytr])

    lds = LinearDynamicalSystem(
        GaussianStateModel(
            Matrix(0.9I, k, k), Matrix(0.1I, k, k), zeros(k), zeros(k), Matrix(1.0I, k, k)
        ),
        (
            kin=SplineGaussianObservationModel(
                randn(rng, pk, k),
                Matrix(1.0I, pk, pk),
                zeros(pk);
                y=Y.kin,
                n_bins=5,
                spline_ridge=0.0,
            ),
            spk=PoissonObservationModel(0.1 .* randn(rng, pn, k), fill(1.0, pn)),
        ),
    )
    el = fit!(lds, Y; max_iter=25, progress=false)
    @test el[end] > el[1]
    @test !is_identity_warp(lds.obs_model.kin.warp)
    # The Laplace ELBO is a bound, so it need not equal a marginal likelihood;
    # what must hold is that recomputing it at the final parameters agrees.
    @test elbo(lds, Y) ≈ el[end] rtol = 1e-3
    xs, Ps = smooth(lds, Y)
    @test length(xs) == 6
    return nothing
end

# ============================================================================
# SLDS
# ============================================================================

function sg_slds_regime(C, warp; θ=0.2, r=0.95, p=size(C, 1))
    om = SplineGaussianObservationModel(
        copy(C), Matrix(0.05I, p, p), zeros(p); bounds=(warp.lo, warp.hi), n_bins=4
    )
    om.warp = warp
    return LinearDynamicalSystem(sg_state_model(; θ=θ, r=r), om)
end

function test_spline_slds_fit()
    rng = StableRNG(81)
    p, K = 4, 2
    lo, hi = fill(-8.0, p), fill(8.0, p)
    warp_true = MonotonicWarp(lo, hi; n_bins=4)
    SSD.warp_unpack!(warp_true, 0.8 .* randn(rng, warp_nparams(warp_true)))
    C = randn(rng, p, SG_LATENT)

    slds_true = SSD.SLDS(;
        A=[0.97 0.03; 0.04 0.96],
        πₖ=[0.5, 0.5],
        LDSs=[
            sg_slds_regime(C, warp_true; θ=0.35, r=0.97),
            sg_slds_regime(1.4 .* C, warp_true; θ=-0.05, r=0.90),
        ],
    )
    _, _, Y = rand(StableRNG(82), slds_true, fill(90, 8))

    warp0 = MonotonicWarp(lo, hi; n_bins=4)
    init_rng = StableRNG(83)
    slds = SSD.SLDS(;
        A=[0.9 0.1; 0.1 0.9],
        πₖ=[0.5, 0.5],
        LDSs=[
            sg_slds_regime(randn(init_rng, p, SG_LATENT), warp0; θ=0.2, r=0.95),
            sg_slds_regime(randn(init_rng, p, SG_LATENT), warp0; θ=-0.2, r=0.90),
        ],
    )
    for l in slds.LDSs
        l.obs_model.R .= Matrix(1.0I, p, p)
        l.obs_model.spline_ridge = 0.0
    end

    el = fit!(slds, Y; max_iter=25, smoothing_iters=2, progress=false)
    @test el[end] > el[1]
    # The warp is shared across regimes by construction and stays shared.
    @test all(l -> l.obs_model.warp === slds.LDSs[1].obs_model.warp, slds.LDSs)
    @test !is_identity_warp(slds.LDSs[1].obs_model.warp)
    # Each regime's R still obeys the emission's structure.
    for l in slds.LDSs
        @test l.obs_model.R ≈ Diagonal(diag(l.obs_model.R))
    end

    out = smooth(slds, Y)
    @test sum(out.trial_elbo) ≈ out.elbo rtol = 1e-8
    @test length(out.x) == 8
    @test size(out.γ[1]) == (2, 90)
    return nothing
end

# Regimes seeded with different warps are collapsed onto regime 1's, with a
# warning — the model fits one shared warp.
function test_spline_slds_collapses_distinct_warps()
    rng = StableRNG(84)
    p = 3
    lo, hi = fill(-5.0, p), fill(5.0, p)
    w1 = MonotonicWarp(lo, hi; n_bins=4)
    w2 = MonotonicWarp(lo, hi; n_bins=4)
    SSD.warp_unpack!(w2, 0.5 .* randn(rng, warp_nparams(w2)))
    C = randn(rng, p, SG_LATENT)
    slds = SSD.SLDS(;
        A=[0.9 0.1; 0.1 0.9],
        πₖ=[0.5, 0.5],
        LDSs=[sg_slds_regime(C, w1), sg_slds_regime(C, w2)],
    )
    Y = [randn(StableRNG(85), p, 40) for _ in 1:3]
    @test_logs (:warn,) match_mode = :any fit!(
        slds, Y; max_iter=2, smoothing_iters=1, progress=false
    )
    @test slds.LDSs[2].obs_model.warp === slds.LDSs[1].obs_model.warp

    # Mismatched shapes are an error, not a silent collapse.
    w3 = MonotonicWarp(lo, hi; n_bins=7)
    slds2 = SSD.SLDS(;
        A=[0.9 0.1; 0.1 0.9],
        πₖ=[0.5, 0.5],
        LDSs=[sg_slds_regime(C, w1), sg_slds_regime(C, w3)],
    )
    @test_throws ArgumentError fit!(slds2, Y; max_iter=1, progress=false)
    return nothing
end

#=
The switching-level checks run at every entry point, and a warped model's
`smooth` hands off to its shadow early -- so they must run on the caller's model
before that, as `fit!` runs them, rather than surfacing from the shadow or not
at all.
=#
function test_spline_slds_entry_points_validate()
    p = 3
    lo, hi = fill(-5.0, p), fill(5.0, p)
    C = randn(StableRNG(89), p, SG_LATENT)
    regimes() = [sg_slds_regime(C, MonotonicWarp(lo, hi; n_bins=4)) for _ in 1:2]
    Y = [randn(StableRNG(90), p, 30) for _ in 1:2]
    improper_π = SSD.SLDS(; A=[0.9 0.1; 0.1 0.9], πₖ=[0.7, 0.7], LDSs=regimes())
    improper_A = SSD.SLDS(; A=[0.9 0.3; 0.1 0.9], πₖ=[0.5, 0.5], LDSs=regimes())
    for bad in (improper_π, improper_A)
        @test_throws InvalidProbabilityVectorError smooth(bad, Y)
        @test_throws InvalidProbabilityVectorError elbo(bad, Y)
        @test_throws InvalidProbabilityVectorError fit!(
            deepcopy(bad), Y; max_iter=1, progress=false
        )
    end
    return nothing
end

#=
`tied_params` on a warped `SLDS`. The names are resolved against the spline
emission, so `:warp` is accepted — it is shared across regimes by construction,
so tying it asks for nothing more — and the shadow's Gaussian regimes, which do
not know that name, never see it. A tie changes only how often a shared group's
prior is counted: with an identical IW prior on an identical `R` in each regime,
tying `R` removes exactly `K - 1` copies of its log-density.
=#
function test_spline_slds_smooth_tied_params()
    p, K = 3, 2
    lo, hi = fill(-6.0, p), fill(6.0, p)
    warp = MonotonicWarp(lo, hi; n_bins=4)
    SSD.warp_unpack!(warp, 0.5 .* randn(StableRNG(86), warp_nparams(warp)))
    C = randn(StableRNG(87), p, SG_LATENT)
    slds = SSD.SLDS(;
        A=[0.9 0.1; 0.1 0.9],
        πₖ=[0.5, 0.5],
        LDSs=[sg_slds_regime(C, warp; θ=0.3), sg_slds_regime(C, warp; θ=-0.1)],
    )
    _, _, Y = rand(StableRNG(88), slds, fill(40, 3))

    prior = IWPrior(; Ψ=Matrix(0.5I, p, p), ν=float(p + 4))
    for l in slds.LDSs
        l.obs_model.R_prior = prior
    end
    kw = (; smoothing_iters=5, tol=0.0)
    free = smooth(slds, Y; kw...)
    tied = smooth(slds, Y; tied_params=(:R, :warp), kw...)

    @test tied.terminal_logz == 0
    @test tied.x ≈ free.x
    @test tied.trial_elbo ≈ free.trial_elbo
    @test free.elbo - tied.elbo ≈
        (K - 1) * SSD.iw_logprior_term(slds.LDSs[1].obs_model.R, prior) rtol = 1e-8
    @test_throws ArgumentError smooth(slds, Y; tied_params=(:not_a_parameter,), kw...)

    # `fit!` resolves the same names, and the held-out score it records runs
    # through `smooth` with them.
    tr = fit!(
        slds,
        Y[1:2];
        y_test=Y[3:3],
        tied_params=(:C, :d, :R, :warp),
        max_iter=3,
        smoothing_iters=1,
        progress=false,
    )
    @test all(isfinite, tr.test)
    @test slds.LDSs[2].obs_model.C ≈ slds.LDSs[1].obs_model.C
    @test slds.LDSs[2].obs_model.R ≈ slds.LDSs[1].obs_model.R
    return nothing
end
