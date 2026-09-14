"""
    SSDTest

Shared assertion helpers for the `docs/examples/` tutorials. Tutorials embed
calls into a `#src` section at the bottom (stripped from the rendered docs,
kept in the raw `.jl` so the test runner sees them). Pattern lifted from
HiddenMarkovModels.jl's `HMMTest` sub-package.
"""
module SSDTest

using LinearAlgebra
using Statistics
using StateSpaceDynamics
using Test

export test_em_monotone, test_em_improves, test_smooth_improves, test_lds_dimensions
export elbo_monotone

"""
    elbo_monotone(elbos; rtol=1e-7) -> Bool

Whether an ELBO trace is non-decreasing, to a tolerance scaled by the bound's
own magnitude.

A fixed absolute threshold is not safe. The sufficient statistics are
accumulated in parallel chunks, so the summation order — and with it the last
few digits of every M-step — depends on how many threads the suite runs with.
The result stays deterministic for a given thread count, but a fit that is
monotone on one thread can show dips of order `1e-4` on two, purely from
reassociation, and EM then amplifies that into a visibly different local
optimum. Measured on the Gaussian+Poisson composite fit: repeated fits agree bit
for bit at fixed threads, while the final bound moves by tens of nats across
thread counts.

`1e-7` of the bound's magnitude sits far below any real monotonicity failure,
which would be whole nats, and comfortably above the arithmetic.
"""
function elbo_monotone(elbos; rtol::Real=1e-7)
    length(elbos) < 2 && return true
    tol = rtol * max(one(eltype(elbos)), maximum(abs, elbos))
    return all(>=(-tol), diff(elbos))
end

"""
    test_em_monotone(elbos; rtol=1e-7)

Assert the ELBO trajectory returned by [`fit!`](@ref) is non-decreasing
step-by-step, to a tolerance scaled by the bound (see [`elbo_monotone`](@ref)).
Suitable for Gaussian LDS, where EM is exactly monotone. For Laplace /
variational EM use [`test_em_improves`](@ref) instead — there the inner
approximation can cause genuine local dips.
"""
function test_em_monotone(elbos; rtol::Real=1e-7)
    @testset "EM ELBO monotone" begin
        @test length(elbos) >= 1
        if length(elbos) > 1
            @test elbo_monotone(elbos; rtol=rtol)
            @test elbos[end] >= elbos[1] - rtol * max(1.0, maximum(abs, elbos))
        end
    end
    return nothing
end

"""
    test_em_improves(elbos; tol=1e-6)

Assert the ELBO trajectory ends no worse than where it started. Use this
for Laplace-EM (PoissonLDS) and variational EM (SLDS) where the inner
approximation can cause small downward steps even on a well-behaved fit.
"""
function test_em_improves(elbos; tol::Real=1e-6)
    @testset "EM ELBO improves overall" begin
        @test length(elbos) >= 1
        if length(elbos) > 1
            @test elbos[end] >= elbos[1] - tol
        end
    end
    return nothing
end

"""
    test_smooth_improves(x_true, x_pre, x_post)

Assert that the smoothed-state estimate after EM is closer to the true
latents than the pre-EM estimate, after solving for the best linear map
between the two (latent coordinates are identifiable only up to invertible
change-of-basis). `x_true`, `x_pre`, `x_post` are each `D × T` matrices.
"""
function test_smooth_improves(
    x_true::AbstractMatrix, x_pre::AbstractMatrix, x_post::AbstractMatrix
)
    @testset "Smoothing improves with EM" begin
        @test size(x_true) == size(x_pre) == size(x_post)
        err_pre = _aligned_residual(x_true, x_pre)
        err_post = _aligned_residual(x_true, x_post)
        @test err_post <= err_pre
    end
    return nothing
end

function _aligned_residual(x_true::AbstractMatrix, x_est::AbstractMatrix)
    T_map = x_true / x_est
    return norm(x_true - T_map * x_est) / sqrt(length(x_true))
end

"""
    test_lds_dimensions(lds; latent_dim, obs_dim)

Sanity-check that a [`LinearDynamicalSystem`](@ref)'s state and observation
model dimensions match the expected sizes. Catches regressions where a
constructor silently accepts mis-shaped parameters.
"""
function test_lds_dimensions(lds; latent_dim::Int, obs_dim::Int)
    @testset "LDS dimensions" begin
        @test lds.latent_dim == latent_dim
        @test lds.obs_dim == obs_dim
        @test size(lds.state_model.A) == (latent_dim, latent_dim)
        @test size(lds.state_model.Q) == (latent_dim, latent_dim)
        @test size(lds.obs_model.C) == (obs_dim, latent_dim)
    end
    return nothing
end

end # module
