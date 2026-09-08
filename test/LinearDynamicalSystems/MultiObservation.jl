#=============================================================================
Several observation models on one latent state

A `CompositeObservationModel` bundles emissions that read out one shared latent
process. The members are conditionally independent given the latent path, so
every emission quantity is a sum over them — which is what most of these tests
check, against an explicit reference or against the equivalent single-emission
model.

The strongest lever is the **stacked equivalent**: an all-Gaussian composite is
the same model as one Gaussian emission with `C = [C₁; C₂]`, `d = [d₁; d₂]` and
`R = blockdiag(R₁, R₂)`, so its smoother output, ELBO and marginal
log-likelihood must agree with that model's exactly. What the composite buys
over the stacked form is a block-diagonal `R` that stays block-diagonal, per
member `depends_on` / priors / `fit_bool`, and members that need not be Gaussian
at all.
=============================================================================#

const MO_LATENT_DIM = 2

function mo_state_model(::Type{T}=Float64; θ=0.2) where {T<:Real}
    return GaussianStateModel(
        T.(0.9 * [cos(θ) -sin(θ); sin(θ) cos(θ)]),
        Matrix{T}(0.05I, MO_LATENT_DIM, MO_LATENT_DIM),
        zeros(T, MO_LATENT_DIM),
        zeros(T, MO_LATENT_DIM),
        Matrix{T}(0.2I, MO_LATENT_DIM, MO_LATENT_DIM),
    )
end

# Deterministic emissions so a failure is reproducible.
function mo_gaussian(p::Int; seed::Int=0)
    rng = StableRNG(700 + seed)
    return GaussianObservationModel(
        randn(rng, p, MO_LATENT_DIM), Matrix(Diagonal(0.2 .+ rand(rng, p))), randn(rng, p)
    )
end

function mo_poisson(p::Int; seed::Int=0)
    rng = StableRNG(900 + seed)
    return PoissonObservationModel(0.4 .* randn(rng, p, MO_LATENT_DIM), fill(log(1.2), p))
end

function mo_composite(; θ=0.2, p1::Int=3, p2::Int=4)
    return LinearDynamicalSystem(
        mo_state_model(; θ=θ), (a=mo_gaussian(p1; seed=1), b=mo_gaussian(p2; seed=2))
    )
end

function mo_mixed(; θ=0.2, pk::Int=3, ps::Int=5)
    return LinearDynamicalSystem(
        mo_state_model(; θ=θ), (kin=mo_gaussian(pk; seed=3), spk=mo_poisson(ps; seed=4))
    )
end

"""The single-emission model an all-Gaussian composite is equivalent to."""
function mo_stacked(lds)
    a, b = lds.obs_model.a, lds.obs_model.b
    p1, p2 = size(a.C, 1), size(b.C, 1)
    R = zeros(p1 + p2, p1 + p2)
    R[1:p1, 1:p1] .= a.R
    R[(p1 + 1):end, (p1 + 1):end] .= b.R
    return LinearDynamicalSystem(
        mo_state_model(), GaussianObservationModel(vcat(a.C, b.C), R, vcat(a.d, b.d))
    )
end

function mo_data(; ntrials::Int=6, tsteps::Int=25, p1::Int=3, p2::Int=4)
    rng = StableRNG(4242)
    ya = [randn(rng, p1, tsteps) for _ in 1:ntrials]
    yb = [randn(rng, p2, tsteps) for _ in 1:ntrials]
    return (a=ya, b=yb), [vcat(ya[n], yb[n]) for n in 1:ntrials]
end

# ============================================================================
# Construction and parameter access
# ============================================================================

"""
Dimensions, `fit_bool` layout, member access and the suffixed parameter names.
"""
function test_multiobs_construction()
    @testset "construction and access" begin
        lds = mo_mixed()
        om = lds.obs_model

        # `obs_dim` is the total; each member keeps its own.
        @test lds.obs_dim == 3 + 5
        @test lds.uy_dim == 0

        # Four state slots, then `[C d D]` and `R` for the Gaussian member and
        # `[C d D]` alone for the Poisson one.
        @test length(lds.fit_bool) == 7
        @test all(lds.fit_bool)

        @test om.kin === lds.obs_model.models.kin
        @test om.C_kin === om.kin.C
        @test om.d_spk === om.spk.d
        @test :kin in propertynames(om)
        @test :C_kin in propertynames(om)

        # `QUAD` is the AND over members and lives in the type.
        @test !SSD._emission_is_quadratic(om)
        @test SSD._emission_is_quadratic(mo_composite().obs_model)

        # The keyword `fit_bool` form lowers to the positional layout.
        frozen = LinearDynamicalSystem(
            mo_state_model(),
            (kin=mo_gaussian(3; seed=3), spk=mo_poisson(5; seed=4));
            fit_bool=(P0=false, kin=(R=false,), C_spk=false),
        )
        @test frozen.fit_bool == Bool[1, 0, 1, 1, 1, 0, 0]
    end
    return nothing
end

"""Keys that would make the suffixed spelling ambiguous, and mismatched data."""
function test_multiobs_validation()
    @testset "validation" begin
        # A member may not be named after an observation parameter.
        @test_throws ArgumentError CompositeObservationModel((C=mo_gaussian(3),))
        @test_throws ArgumentError CompositeObservationModel((models=mo_gaussian(3),))
        @test_throws ArgumentError CompositeObservationModel(NamedTuple())

        lds = mo_composite()
        y, _ = mo_data()

        # Observations must arrive under the model's keys, as a NamedTuple.
        @test_throws ArgumentError SSD.Data(lds, y.a)
        @test_throws ArgumentError SSD.Data(lds, (a=y.a, c=y.b))

        # …and a single-emission model must not be given one.
        single = LinearDynamicalSystem(mo_state_model(), mo_gaussian(3))
        @test_throws ArgumentError SSD.Data(single, (y=y.a,))

        # Members must agree on the trials they saw.
        @test_throws ArgumentError SSD.Data(lds, (a=y.a, b=y.b[1:3]))
        @test_throws ArgumentError SSD.Data(lds, (a=y.a, b=vcat([randn(4, 9)], y.b[2:end])))

        # A member's row count is checked against its own emission.
        @test_throws SSD.DimensionMismatchError SSD.Data(
            lds, (a=[randn(7, 25) for _ in 1:6], b=y.b)
        )
    end
    return nothing
end

# ============================================================================
# Equivalence with the stacked single-emission model
# ============================================================================

"""
Smoothing, the ELBO and the marginal log-likelihood of an all-Gaussian composite
are the stacked model's, exactly.
"""
function test_multiobs_matches_stacked()
    @testset "matches the stacked equivalent" begin
        lds = mo_composite()
        stacked = mo_stacked(lds)
        y, ys = mo_data()

        xc, Pc = smooth(lds, y)
        xs, Ps = smooth(stacked, ys)
        @test all(isapprox.(xc, xs; rtol=1e-10))
        @test all(isapprox.(Pc, Ps; rtol=1e-10))

        @test isapprox(elbo(lds, y), elbo(stacked, ys); rtol=1e-10)
        @test isapprox(loglikelihood(lds, y), loglikelihood(stacked, ys); rtol=1e-10)

        # With no priors the Gaussian smoother is exact, so the ELBO is the
        # marginal log-likelihood.
        @test isapprox(elbo(lds, y), loglikelihood(lds, y); rtol=1e-9)
    end
    return nothing
end

"""
One EM step agrees block for block. The emission regression does not involve
`R` at all, so the stacked `[C d]` is the members' stacked; and the stacked
residual covariance's diagonal blocks are the members' `R`.

They diverge after that — the stacked model fits a *full* `R`, which is the
structure the composite exists to keep block-diagonal — so this is one step.
"""
function test_multiobs_one_em_step_matches_stacked()
    @testset "one EM step matches the stacked equivalent" begin
        lds = mo_composite()
        stacked = mo_stacked(lds)
        y, ys = mo_data()
        p1 = size(lds.obs_model.a.C, 1)

        ec = fit!(lds, y; max_iter=1, progress=false)
        es = fit!(stacked, ys; max_iter=1, progress=false)
        @test isapprox(ec[1], es[1]; rtol=1e-10)

        om, som = lds.obs_model, stacked.obs_model
        @test isapprox(vcat(om.a.C, om.b.C), som.C; rtol=1e-9)
        @test isapprox(vcat(om.a.d, om.b.d), som.d; rtol=1e-9)
        @test isapprox(om.a.R, som.R[1:p1, 1:p1]; rtol=1e-9)
        @test isapprox(om.b.R, som.R[(p1 + 1):end, (p1 + 1):end]; rtol=1e-9)
        @test isapprox(lds.state_model.A, stacked.state_model.A; rtol=1e-9)
        @test isapprox(lds.state_model.Q, stacked.state_model.Q; rtol=1e-9)
    end
    return nothing
end

"""
A one-member composite is the bare model. They agree exactly off the
equal-length fast path; on it the composite runs the per-trial mean pass rather
than the batched one, so the two summation orders differ in the last few ULPs.
"""
function test_multiobs_single_member_matches_bare()
    @testset "one member reproduces the bare model" begin
        y, _ = mo_data(; ntrials=3, tsteps=25)
        # Ragged lengths keep both models off the batched path, where they agree
        # bit for bit.
        ragged = [y.a[1][:, 1:20], y.a[2][:, 1:(27 % 25 + 12)], y.a[3]]
        solo = LinearDynamicalSystem(mo_state_model(), (y=mo_gaussian(3; seed=1),))
        bare = LinearDynamicalSystem(mo_state_model(), mo_gaussian(3; seed=1))
        @test fit!(solo, (y=ragged,); max_iter=10, progress=false) ==
            fit!(bare, ragged; max_iter=10, progress=false)
        @test solo.obs_model.y.C == bare.obs_model.C
        @test solo.state_model.Q == bare.state_model.Q

        # Equal-length: the batched mean pass differs in summation order only.
        solo2 = LinearDynamicalSystem(mo_state_model(), (y=mo_gaussian(3; seed=1),))
        bare2 = LinearDynamicalSystem(mo_state_model(), mo_gaussian(3; seed=1))
        e1 = fit!(solo2, (y=y.a,); max_iter=10, progress=false)
        e2 = fit!(bare2, y.a; max_iter=10, progress=false)
        @test isapprox(e1, e2; rtol=1e-11)
    end
    return nothing
end

# ============================================================================
# Mixed emission types
# ============================================================================

"""
The complete-data log-likelihood, its gradient and its curvature are the sums
over members — checked against an explicit per-member reference and against
`ForwardDiff`.
"""
function test_multiobs_mixed_kernels()
    @testset "mixed emission kernels" begin
        lds = mo_mixed()
        D, pk, ps = MO_LATENT_DIM, 3, 5
        rng = StableRNG(31)
        x = 0.3 .* randn(rng, D, 20)
        yt = (kin=randn(rng, pk, 20), spk=Float64.(rand(rng, 0:3, ps, 20)))

        # Reference: the state term once, plus each member's emission term.
        ccK = SSD.SmoothConstants(Float64, D, pk)
        SSD._compute_state_constants!(ccK, lds.state_model)
        SSD._compute_obs_constants!(ccK, lds.obs_model.kin)
        ccS = SSD.SmoothConstants(Float64, D, ps)
        SSD._compute_obs_constants!(ccS, lds.obs_model.spk)
        ref = sum(axes(x, 2)) do t
            SSD.state_loglikelihood!(ccK, zeros(D), zeros(D), lds, x, t, nothing) +
            SSD.observation_loglikelihood!(
                ccK, zeros(pk), zeros(pk), lds.obs_model.kin, x, yt.kin, t, nothing
            ) +
            SSD.observation_loglikelihood!(
                ccS, zeros(ps), zeros(ps), lds.obs_model.spk, x, yt.spk, t, nothing
            )
        end
        @test isapprox(sum(SSD.joint_loglikelihood(lds, x, yt)), ref; rtol=1e-12)

        f(v) = sum(SSD.joint_loglikelihood(lds, reshape(v, D, :), yt))
        ws = SSD.SmoothWorkspace(Float64, D, 0, size(x, 2))
        SSD.compute_smooth_constants!(ws, lds)
        @test isapprox(
            vec(SSD.gradient!(ws, lds, x, yt)),
            ForwardDiff.gradient(f, vec(x));
            rtol=1e-6,
            atol=1e-8,
        )

        SSD.hessian!(ws, lds, x, yt)
        H = ForwardDiff.hessian(f, vec(x))
        maxdiff = maximum(
            abs(ws.btd.H_diag[t][i, j] - H[(t - 1) * D + i, (t - 1) * D + j]) for
            t in axes(x, 2), i in 1:D, j in 1:D
        )
        @test maxdiff < 1e-6
    end
    return nothing
end

"""Sampling, fitting and scoring a Gaussian + Poisson composite."""
function test_multiobs_mixed_fit()
    @testset "mixed emission fit" begin
        truth = mo_mixed()
        _, ys = rand(StableRNG(55), truth, fill(50, 20))
        y = (kin=[t.kin for t in ys], spk=[t.spk for t in ys])

        @test all(all(v .>= 0) && all(v .== round.(v)) for v in y.spk)
        @test size(y.kin[1]) == (3, 50)
        @test size(y.spk[1]) == (5, 50)

        init = LinearDynamicalSystem(
            mo_state_model(),
            (
                kin=GaussianObservationModel(
                    truth.obs_model.kin.C .+ 0.2 .* randn(StableRNG(9), 3, MO_LATENT_DIM),
                    Matrix(1.0I, 3, 3),
                    zeros(3),
                ),
                spk=PoissonObservationModel(
                    truth.obs_model.spk.C .+ 0.2 .* randn(StableRNG(10), 5, MO_LATENT_DIM),
                    fill(log(1.0), 5),
                ),
            ),
        )
        elbos = fit!(init, y; max_iter=30, progress=false)
        @test elbo_monotone(elbos)
        @test elbos[end] > elbos[1]

        # The Gaussian member's noise scale is recovered.
        @test isapprox(diag(init.obs_model.kin.R), diag(truth.obs_model.kin.R); rtol=0.3)
        # The Poisson member's mean baseline rate is recovered.
        @test isapprox(
            mean(exp.(init.obs_model.spk.d)), mean(exp.(truth.obs_model.spk.d)); rtol=0.3
        )

        # Public entry points, and the marginal that does not exist for a
        # non-Gaussian member.
        xs, _ = smooth(init, y)
        @test length(xs) == 20 && size(xs[1]) == (MO_LATENT_DIM, 50)
        #=
        The Laplace E-step warm-starts from the previous iterate inside `fit!`
        and from the prior mean in a fresh call, so a re-evaluation lands a hair
        either side of the recorded value rather than exactly on it.
        =#
        @test isapprox(elbo(init, y), elbos[end]; rtol=1e-6)
        @test_throws ErrorException loglikelihood(init, y)
    end
    return nothing
end

# ============================================================================
# Per-member `fit_bool`, priors and `depends_on`
# ============================================================================

"""Freezing one member's emission leaves it untouched and the other free."""
function test_multiobs_fit_bool_per_member()
    @testset "fit_bool freezes one member" begin
        lds = mo_composite()
        SSD.validate_LDS(lds)
        y, _ = mo_data()
        frozen = LinearDynamicalSystem(
            mo_state_model(),
            (a=mo_gaussian(3; seed=1), b=mo_gaussian(4; seed=2));
            fit_bool=(a=(C=false, R=false),),
        )
        C0, R0 = copy(frozen.obs_model.a.C), copy(frozen.obs_model.a.R)
        Cb0 = copy(frozen.obs_model.b.C)
        fit!(frozen, y; max_iter=5, progress=false)
        @test frozen.obs_model.a.C == C0
        @test frozen.obs_model.a.R == R0
        @test frozen.obs_model.b.C != Cb0
    end
    return nothing
end

"""A prior on one member only still leaves the ELBO monotone."""
function test_multiobs_priors_per_member()
    @testset "priors on one member" begin
        y, _ = mo_data()
        a = mo_gaussian(3; seed=1)
        a.R_prior = IWPrior(Matrix(0.5I, 3, 3), 8.0)
        a.CD_prior = MNPrior(;
            M₀=zeros(3, MO_LATENT_DIM + 1),
            Λ=Matrix(2.0I, MO_LATENT_DIM + 1, MO_LATENT_DIM + 1),
        )
        lds = LinearDynamicalSystem(mo_state_model(), (a=a, b=mo_gaussian(4; seed=2)))
        elbos = fit!(lds, y; max_iter=20, progress=false)
        @test all(diff(elbos) .>= -1e-8)

        # The prior shrinks the member it is on, and only that one.
        loose = mo_composite()
        fit!(loose, y; max_iter=20, progress=false)
        @test norm(lds.obs_model.a.C) < norm(loose.obs_model.a.C)
    end
    return nothing
end

"""`depends_on` on one member, on both, and the single-group identity."""
function test_multiobs_depends_on()
    @testset "depends_on per member" begin
        ntrials = 8
        y, _ = mo_data(; ntrials=ntrials)
        session = [:s1, :s1, :s1, :s1, :s2, :s2, :s2, :s2]
        block = [:x, :x, :y, :y, :x, :x, :y, :y]

        # Group names carry the member as a suffix, in `fit_bool` order.
        lds = mo_composite()
        set_depends_on!(lds.obs_model, (C_a=session, d_a=session, R_a=session))
        @test SSD._group_names(lds.obs_model) == (:C_a, :R_a, :C_b, :R_b)
        @test Set(keys(lds.obs_model.depends_on)) == Set((:C_a, :d_a, :R_a))
        @test lds.obs_model.b.depends_on === nothing

        elbos = fit!(lds, y; max_iter=15, progress=false)
        @test all(diff(elbos) .>= -1e-8)
        @test group_labels(lds.obs_model, :C_a) == [:s1, :s2]
        @test isempty(group_labels(lds.obs_model, :C_b))
        @test !(
            group_parameter(lds.obs_model, :C_a, :s1) ≈
            group_parameter(lds.obs_model, :C_a, :s2)
        )

        # Both members grouped on different labels: the cells are the join.
        both = mo_composite()
        set_depends_on!(
            both.obs_model,
            (C_a=session, d_a=session, R_a=session, C_b=block, d_b=block, R_b=block),
        )
        grp = SSD.parameter_grouping(both, ntrials; y=SSD.Data(both, y).y)
        @test grp.ncells == 4
        @test length(grp.cell_obs) == 2
        @test all(diff(fit!(both, y; max_iter=15, progress=false)) .>= -1e-8)

        # A single group is the ungrouped fit.
        one = mo_composite()
        set_depends_on!(
            one.obs_model,
            (C_a=fill(:only, ntrials), d_a=fill(:only, ntrials), R_a=fill(:only, ntrials)),
        )
        plain = mo_composite()
        @test isapprox(
            fit!(one, y; max_iter=12, progress=false),
            fit!(plain, y; max_iter=12, progress=false);
            rtol=1e-10,
        )
        @test isapprox(
            group_parameter(one.obs_model, :C_a, :only), plain.obs_model.a.C; rtol=1e-9
        )
    end
    return nothing
end

"""One member stitched across sessions of differing width, the other shared."""
function test_multiobs_stitching()
    @testset "stitching one member" begin
        ntrials, tsteps, p1, p2, pb = 8, 25, 3, 5, 4
        rng = StableRNG(606)
        ya = vcat(
            [randn(rng, p1, tsteps) for _ in 1:4], [randn(rng, p2, tsteps) for _ in 1:4]
        )
        yb = [randn(rng, pb, tsteps) for _ in 1:ntrials]
        session = vcat(fill(:s1, 4), fill(:s2, 4))

        lds = LinearDynamicalSystem(
            mo_state_model(),
            (
                a=GaussianObservationModel(
                    randn(rng, p1, MO_LATENT_DIM), Matrix(1.0I, p1, p1), zeros(p1)
                ),
                b=mo_gaussian(pb; seed=2),
            ),
        )
        set_depends_on!(lds.obs_model, (C_a=session, d_a=session, R_a=session))

        elbos = fit!(lds, (a=ya, b=yb); max_iter=10, progress=false)
        @test all(diff(elbos) .>= -1e-8)
        @test size(group_parameter(lds.obs_model, :C_a, :s1), 1) == p1
        @test size(group_parameter(lds.obs_model, :C_a, :s2), 1) == p2
        @test size(group_parameter(lds.obs_model, :R_a, :s2)) == (p2, p2)
        # The member that does not vary keeps the one shape.
        @test size(lds.obs_model.b.C, 1) == pb
    end
    return nothing
end

# ============================================================================
# Inputs, sampling and display
# ============================================================================

"""Per-member `uy`, and the bare-array shorthand that feeds every member."""
function test_multiobs_observation_inputs()
    @testset "per-member observation inputs" begin
        ntrials, tsteps = 5, 20
        rng = StableRNG(808)
        a = GaussianObservationModel(;
            C=randn(rng, 3, MO_LATENT_DIM),
            R=Matrix(0.3I, 3, 3),
            d=zeros(3),
            D=randn(rng, 3, 2),        # 2 covariates
        )
        b = GaussianObservationModel(;
            C=randn(rng, 4, MO_LATENT_DIM),
            R=Matrix(0.3I, 4, 4),
            d=zeros(4),
            D=randn(rng, 4, 1),        # 1 covariate
        )
        lds = LinearDynamicalSystem(mo_state_model(), (a=a, b=b))
        @test lds.uy_dim == 3

        va = [randn(rng, 2, tsteps) for _ in 1:ntrials]
        vb = [randn(rng, 1, tsteps) for _ in 1:ntrials]
        y = (
            a=[randn(rng, 3, tsteps) for _ in 1:ntrials],
            b=[randn(rng, 4, tsteps) for _ in 1:ntrials],
        )

        data = SSD.Data(lds, y; uy=(a=va, b=vb))
        @test size(data.uy.a[1], 1) == 2
        @test size(data.uy.b[1], 1) == 1

        elbos = fit!(lds, y; uy=(a=va, b=vb), max_iter=8, progress=false)
        @test all(diff(elbos) .>= -1e-8)

        # A member whose `D` has columns must be given its input.
        @test_throws ArgumentError SSD.Data(lds, y; uy=(a=va, b=nothing))

        # One shared covariate set reaches every member when the widths agree.
        shared_a = GaussianObservationModel(;
            C=randn(rng, 3, MO_LATENT_DIM),
            R=Matrix(0.3I, 3, 3),
            d=zeros(3),
            D=randn(rng, 3, 1),
        )
        shared = LinearDynamicalSystem(mo_state_model(), (a=shared_a, b=b))
        d2 = SSD.Data(shared, y; uy=vb)
        @test d2.uy.a[1] === vb[1] && d2.uy.b[1] === vb[1]
    end
    return nothing
end

"""`rand` returns per-member observations, and they round-trip through `smooth`."""
function test_multiobs_sampling()
    @testset "sampling" begin
        lds = mo_mixed()

        x1, y1 = rand(StableRNG(3), lds, 30)
        @test size(x1) == (MO_LATENT_DIM, 30)
        @test y1 isa NamedTuple && keys(y1) == (:kin, :spk)
        @test size(y1.kin) == (3, 30) && size(y1.spk) == (5, 30)

        xs, ys = rand(StableRNG(3), lds, fill(30, 4))
        @test length(ys) == 4
        @test all(size(t.spk) == (5, 30) for t in ys)
        @test all(size(x) == (MO_LATENT_DIM, 30) for x in xs)

        y = (kin=[t.kin for t in ys], spk=[t.spk for t in ys])
        xh, Ph = smooth(lds, y)
        @test length(xh) == 4 && size(Ph[1]) == (MO_LATENT_DIM, MO_LATENT_DIM, 30)
    end
    return nothing
end

"""The composite prints its members, and `fit_bool` names them."""
function test_multiobs_show()
    @testset "display" begin
        lds = mo_mixed()
        out = sprint(show, lds)
        @test occursin("Composite Observation Model (2 models)", out)
        @test occursin("[kin]", out) && occursin("[spk]", out)
        @test occursin("kin: C (and d, D)", out)
        @test occursin("kin: R", out)
        @test occursin("spk: C, d", out)
        # A Poisson member has no `R` slot to name.
        @test !occursin("spk: R", out)
    end
    return nothing
end

# ============================================================================
# SLDS
# ============================================================================

function mo_slds(build; K::Int=2)
    ldss = [build(; θ=0.15 * k) for k in 1:K]
    return SLDS(; A=[0.9 0.1; 0.15 0.85], πₖ=[0.5, 0.5], LDSs=ldss)
end

"""Every regime must carry the same members, in the same order."""
function test_multiobs_slds_validation()
    @testset "SLDS validation" begin
        slds = mo_slds(mo_composite)
        validate_SLDS(slds)
        @test slds.LDSs[1].obs_dim == 7

        #=
        Regimes whose emissions differ in their members, or in their order,
        cannot be put in one `SLDS`: `LDSs` is a
        `Vector{LinearDynamicalSystem{T,S,O}}` with a single concrete `O`, and
        the member names are part of the composite's type.
        =#
        reordered = LinearDynamicalSystem(
            mo_state_model(), (b=mo_gaussian(4; seed=2), a=mo_gaussian(3; seed=1))
        )
        @test typeof(reordered.obs_model) !== typeof(mo_composite().obs_model)
        @test_throws MethodError SLDS(
            [0.9 0.1; 0.15 0.85], [0.5, 0.5], [mo_composite(), reordered]
        )
    end
    return nothing
end

"""An SLDS with a composite emission samples, fits and smooths."""
function test_multiobs_slds_fit()
    @testset "SLDS fit" begin
        for (name, build) in (("all-Gaussian", mo_composite), ("mixed", mo_mixed))
            @testset "$name" begin
                slds = mo_slds(build)
                z, _, ys = rand(StableRNG(12), slds, fill(40, 10))
                @test ys[1] isa NamedTuple
                keys_ = keys(ys[1])
                y = NamedTuple{keys_}(map(k -> [t[k] for t in ys], keys_))

                elbos = fit!(slds, y; max_iter=5, progress=false)
                @test all(isfinite, elbos)

                out = smooth(slds, y)
                @test length(out.x) == 10
                @test size(out.x[1]) == (MO_LATENT_DIM, 40)
                @test size(out.γ[1]) == (2, 40)
                @test isfinite(out.elbo)
                @test isfinite(elbo(slds, y))
            end
        end
    end
    return nothing
end

"""`tied_params` names one member; the others stay free per regime."""
function test_multiobs_slds_tied_params()
    @testset "SLDS tied_params names one member" begin
        slds = mo_slds(mo_mixed)
        _, _, ys = rand(StableRNG(13), slds, fill(40, 10))
        y = (kin=[t.kin for t in ys], spk=[t.spk for t in ys])

        fitted = mo_slds(mo_mixed)
        elbos = fit!(fitted, y; max_iter=5, progress=false, tied_params=(:C_spk, :d_spk))
        @test all(isfinite, elbos)
        @test fitted.LDSs[1].obs_model.spk.C ≈ fitted.LDSs[2].obs_model.spk.C
        @test fitted.LDSs[1].obs_model.spk.d ≈ fitted.LDSs[2].obs_model.spk.d
        @test !(fitted.LDSs[1].obs_model.kin.C ≈ fitted.LDSs[2].obs_model.kin.C)
    end
    return nothing
end

"""A composite SLDS whose Poisson member is grouped by session."""
function test_multiobs_slds_depends_on()
    @testset "SLDS with a grouped member" begin
        ntrials, tsteps = 8, 30
        rng = StableRNG(1717)
        y = (
            kin=[randn(rng, 3, tsteps) for _ in 1:ntrials],
            spk=[Float64.(rand(rng, 0:3, 5, tsteps)) for _ in 1:ntrials],
        )
        session = vcat(fill(:s1, 4), fill(:s2, 4))

        slds = mo_slds(mo_mixed)
        for lds in slds.LDSs
            set_depends_on!(lds.obs_model, (C_spk=session, d_spk=session))
        end

        elbos = fit!(slds, y; max_iter=4, progress=false)
        @test all(isfinite, elbos)
        om = slds.LDSs[1].obs_model
        @test !(group_parameter(om, :C_spk, :s1) ≈ group_parameter(om, :C_spk, :s2))
        @test size(om.kin.C, 1) == 3

        out = smooth(slds, y)
        @test length(out.x) == ntrials
    end
    return nothing
end
