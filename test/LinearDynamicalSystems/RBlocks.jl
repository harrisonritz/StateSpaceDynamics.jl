#=============================================================================
A block-diagonal observation noise: `R_blocks` on the Gaussian emissions.

The M-step keeps the within-block entries of the unconstrained `R` update, the
exact maximiser once the entries between blocks are pinned at zero. The model
is then the one a `CompositeObservationModel` of one Gaussian member per block
describes, so the two must take the same EM steps.

Included after `KernelEmissions.jl`, whose fixtures (`ke_*`, `nondecreasing`)
it reuses.
=============================================================================#

rb_project(R, blocks) = SSD._project_R_blocks!(copy(R), blocks)

function rb_state(k)
    A = k == 2 ? [0.9 -0.2; 0.2 0.9] : Matrix(0.8I, k, k)
    return GaussianStateModel(;
        A, Q=Matrix(0.1I, k, k), b=zeros(k), x0=zeros(k), P0=Matrix(1.0I, k, k)
    )
end

# Trials whose noise is correlated across every channel, so a full `R` differs.
function rb_data(rng, p, k; ntrials=6, tsteps=40)
    root = randn(rng, p, p)
    truth = LinearDynamicalSystem(
        rb_state(k),
        GaussianObservationModel(;
            C=randn(rng, p, k), R=root * root' / p + 0.3I, d=zeros(p)
        ),
    )
    return rand(rng, truth, fill(tsteps, ntrials))[2]
end

function rb_lds(p, k; R=Matrix(0.5I, p, p), kwargs...)
    C = randn(StableRNG(7), p, k)
    return LinearDynamicalSystem(
        rb_state(k), GaussianObservationModel(; C, R=copy(R), d=zeros(p), kwargs...)
    )
end

function test_r_blocks_is_the_blocks_of_the_full_update()
    p, k = 7, 2
    blocks = [[1, 4, 6], [2, 3], [5, 7]]         # not contiguous
    ys = rb_data(StableRNG(61), p, k)
    Ψ = (root = randn(StableRNG(62), p, p); root * root' + I)
    for R_prior in (nothing, IWPrior(; Ψ, ν=p + 4.0))
        full = rb_lds(p, k; R_prior)
        blk = rb_lds(p, k; R_prior, R_blocks=blocks)
        # Two scored iterates are one M-step (the first scores the start).
        fit!(full, ys; max_iter=2, tol=0.0, progress=false)
        fit!(blk, ys; max_iter=2, tol=0.0, progress=false)
        # C's update does not depend on R, so one step gives the full step's blocks.
        @test blk.obs_model.C ≈ full.obs_model.C atol = 1e-10
        @test blk.obs_model.R ≈ rb_project(full.obs_model.R, blocks) atol = 1e-10
        @test !(blk.obs_model.R ≈ full.obs_model.R)

        # It stays block diagonal, and EM stays monotone.
        els = fit!(blk, ys; max_iter=30, tol=0.0, progress=false)
        R = blk.obs_model.R
        @test R == rb_project(R, blocks)
        @test isposdef(Symmetric(R))
        @test all(diff(els) .>= -1e-8 * maximum(abs, els))
        @test elbo(blk, ys) ≈ last(els)
    end
end

function test_r_blocks_matches_a_composite_of_its_blocks()
    p, k = 7, 2
    ranges = (a=1:3, b=4:7)
    blocks = [collect(r) for r in ranges]
    ys = rb_data(StableRNG(63), p, k)
    C = randn(StableRNG(64), p, k)
    R0 = rb_project((root = randn(StableRNG(65), p, p); root * root' / p + I), blocks)

    single = LinearDynamicalSystem(
        rb_state(k),
        GaussianObservationModel(; C=copy(C), R=copy(R0), d=zeros(p), R_blocks=blocks),
    )
    members = map(ranges) do r
        GaussianObservationModel(C[r, :], R0[r, r], zeros(length(r)))
    end
    composite = LinearDynamicalSystem(rb_state(k), members)
    ys_c = map(r -> [y[r, :] for y in ys], ranges)

    @test elbo(single, ys) ≈ elbo(composite, ys_c) rtol = 1e-10
    for _ in 1:3
        # Stepwise, so a difference shows at the step that makes it.
        el_s = fit!(single, ys; max_iter=4, tol=0.0, progress=false)
        el_c = fit!(composite, ys_c; max_iter=4, tol=0.0, progress=false)
        @test el_s ≈ el_c rtol = 1e-9
        om = single.obs_model
        for (key, r) in pairs(ranges)
            m = SSD._models(composite.obs_model)[key]
            @test om.C[r, :] ≈ m.C rtol = 1e-7
            @test om.d[r] ≈ m.d atol = 1e-7
            @test om.R[r, r] ≈ m.R rtol = 1e-7
        end
        @test iszero(om.R[ranges.a, ranges.b])
        @test single.state_model.A ≈ composite.state_model.A rtol = 1e-7
        @test single.state_model.Q ≈ composite.state_model.Q rtol = 1e-7
    end
end

function test_r_blocks_checks()
    p, k = 5, 2
    build(; kwargs...) = rb_lds(p, k; kwargs...)
    @test build(; R_blocks=[[1, 2], [3, 4, 5]]) isa LinearDynamicalSystem
    @test_throws ArgumentError build(; R_blocks=[[1, 2], [3, 4]])          # 5 in none
    @test_throws ArgumentError build(; R_blocks=[[1, 2, 3], [3, 4, 5]])    # 3 twice
    @test_throws ArgumentError build(; R_blocks=[[1, 2], [3, 4, 5, 6]])    # outside
    @test_throws ArgumentError build(; R_blocks=[[1, 2], Int[], [3, 4, 5]])
    @test_throws ArgumentError build(; R_blocks=Vector{Int}[])
    @test_throws ArgumentError build(; R_blocks=[[1, 2], [3, 4, 5]], R_diagonal=true)
    full = (root = randn(StableRNG(66), p, p); root * root' + I)
    @test_throws ArgumentError build(; R_blocks=[[1, 2], [3, 4, 5]], R=full)
    kernel_prior = KernelIWPrior(
        KernelPrior([Kernel(Matrix(1.0I, p, p); name="identity")]); dof=p + 3.0
    )
    @test_throws ArgumentError build(; R_blocks=[[1, 2], [3, 4, 5]], R_prior=kernel_prior)
    labels = repeat(["a", "b"], 3)
    @test_throws ArgumentError build(; R_blocks=[[1, 2], [3, 4, 5]], depends_on=(R=labels,))
    @test occursin(
        "R_blocks = 2 blocks of 2, 3",
        sprint(show, build(; R_blocks=[[1, 2], [3, 4, 5]]).obs_model),
    )
end

function test_r_blocks_constrained_emissions()
    rng = MersenneTwister(67)
    p, k, r = 8, 2, 4
    blocks = [[1, 2, 3], [4, 5], [6, 7, 8]]
    Φ = ke_basis(rng, p, r)
    y = ke_data(rng, Φ, k)
    R0 = Matrix(1.0I, p, p)
    structured(R) = R == rb_project(R, blocks) && isposdef(Symmetric(R))

    # A basis, no prior: an ECM step, so the trace never decreases.
    function make()
        return LinearDynamicalSystem(
            ke_state(k),
            ConstrainedGaussianObservationModel(;
                basis=Φ, H=randn(MersenneTwister(1), r, k), R=R0, R_blocks=blocks
            ),
        )
    end
    lds = make()
    @test lds.obs_model.R_blocks == blocks
    trace = fit!(lds, y; max_iter=60, tol=0.0, progress=false)
    @test nondecreasing(trace)
    @test structured(lds.obs_model.R)
    @test !isdiag(lds.obs_model.R)
    @test norm(lds.obs_model.C - Φ * (Φ \ lds.obs_model.C)) <= 1e-10 * norm(lds.obs_model.C)
    @test occursin("R_blocks = 3 blocks", sprint(show, lds.obs_model))
    # One step is the blocks of the unconstrained step: H's update saw the same R.
    one_blk, one_full = make(), make()
    one_full.obs_model.R_blocks = nothing
    fit!(one_blk, y; max_iter=2, tol=0.0, progress=false)
    fit!(one_full, y; max_iter=2, tol=0.0, progress=false)
    @test !(one_blk.obs_model.R ≈ one_full.obs_model.R)
    @test one_blk.obs_model.R ≈ rb_project(one_full.obs_model.R, blocks) atol = 1e-10

    # A kernel prior on the loading, with and without a basis, in the gauges
    # whose trace is the bound's.
    for gauge in (:none, :bound)
        lds = LinearDynamicalSystem(
            ke_state(k),
            ConstrainedKernelGaussianObservationModel(;
                basis=Φ,
                prior=ke_prior(MersenneTwister(2), r),
                H=randn(MersenneTwister(1), r, k),
                R=R0,
                R_blocks=blocks,
                gauge,
            ),
        )
        trace = fit!(lds, y; max_iter=60, progress=false)
        @test nondecreasing(trace)
        @test structured(lds.obs_model.R)
        @test elbo(lds, y) ≈ last(trace)
    end
    lds = LinearDynamicalSystem(
        ke_state(k),
        KernelGaussianObservationModel(;
            prior=ke_prior(MersenneTwister(2), p),
            C=randn(MersenneTwister(1), p, k),
            R=R0,
            R_blocks=blocks,
        ),
    )
    trace = fit!(lds, y; max_iter=40, progress=false)
    @test nondecreasing(trace)
    @test structured(lds.obs_model.R)
    # The unit-moment gauge runs and keeps the structure.
    lds = LinearDynamicalSystem(
        ke_state(k),
        ConstrainedKernelGaussianObservationModel(;
            basis=Φ,
            prior=ke_prior(MersenneTwister(2), r),
            H=randn(MersenneTwister(1), r, k),
            R=R0,
            R_blocks=blocks,
            gauge=:moment,
        ),
    )
    trace = fit!(lds, y; max_iter=30, tol=0.0, progress=false)
    @test all(isfinite, trace)
    @test structured(lds.obs_model.R)

    @test_throws ArgumentError LinearDynamicalSystem(
        ke_state(k),
        ConstrainedGaussianObservationModel(;
            basis=Φ, H=randn(rng, r, k), R=R0, R_blocks=[[1, 2, 3], [4, 5]]
        ),
    )
    @test_throws ArgumentError LinearDynamicalSystem(
        ke_state(k),
        ConstrainedGaussianObservationModel(;
            basis=Φ, H=randn(rng, r, k), R=R0, R_blocks=blocks, R_diagonal=true
        ),
    )
end

# A held-out fit restores a snapshot of the model: the constraint comes with it.
function test_r_blocks_survives_holdout()
    p, k = 6, 2
    blocks = [[1, 2, 3], [4, 5, 6]]
    ys = rb_data(StableRNG(68), p, k; ntrials=8)
    lds = rb_lds(p, k; R_blocks=blocks)
    fit!(lds, ys[1:6]; y_test=ys[7:8], max_iter=6, progress=false)
    @test lds.obs_model.R_blocks == blocks
    @test lds.obs_model.R == rb_project(lds.obs_model.R, blocks)
end

@testset "Block-diagonal R" begin
    test_r_blocks_is_the_blocks_of_the_full_update()
    test_r_blocks_matches_a_composite_of_its_blocks()
    test_r_blocks_checks()
    test_r_blocks_constrained_emissions()
    test_r_blocks_survives_holdout()
end
