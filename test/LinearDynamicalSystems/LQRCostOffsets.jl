#=============================================================================
Per-trial cost-schedule offsets.

A cost schedule is one vector on an event-relative time axis; `cost_offset[i]`
says how many of its bins come before trial `i`'s first bin, so trial `i`'s
transition `t` is under `schedule[t + cost_offset[i]]`.

The one fact every test below leans on: a trial at offset `o` under schedule `S`
is *the same trial* as one at offset 0 under `S[o+1:end]`. That reference shares
nothing with the offset code — it is the ordinary single-schedule path — so the
two agreeing is the claim, not a restatement of it.

Included after `LQRLDS.jl`, whose fixtures it reuses.
=============================================================================#

const CO_L = 24                      # length of the schedule: the shared time axis
const CO_OFFSETS = [0, 3, 5, 3, 8, 0]
const CO_LENGTHS = [14, 15, 12, 15, 16, 14]   # offset + length ≤ CO_L for every trial

"""
An LQR model on the shared axis: running cost 1, running cost 2 from bin 10 on,
and a terminal cost 3. `pin` pins the terminal factor to cost 3 for every trial
(what ragged ends need); otherwise it follows the schedule at each trial's own
last bin.
"""
function _co_model(rng; poisson::Bool=true, pin::Bool=true, nregimes::Int=3)
    sm, glds = lqr_fixture(
        rng; nregimes=nregimes, terminal=true, tsteps=CO_L, onset=nregimes == 3 ? 10 : 1
    )
    if pin
        sm.terminal_regime = nregimes
        refresh!(sm)
    end
    poisson || return sm, glds
    n = SSD.plant_dim(sm)
    d = 2n
    p = 4
    C = randn(rng, p, d) .* 0.3
    C[:, (n + 1):d] .= 0
    return sm, LinearDynamicalSystem(sm, PoissonObservationModel(C, fill(1.0, p)))
end

"""The same model on the schedule a trial at `offset` actually reads."""
function _co_sliced(lds, offset::Int)
    out = deepcopy(lds)
    sm = out.state_model
    sm.schedule = sm.schedule[(offset + 1):end]
    return out
end

_co_counts(rng, p, lengths) = [Float64.(rand(rng, 0:3, p, t)) for t in lengths]

function test_cost_offset_lookup_and_validation()
    sm, lds = _co_model(StableRNG(1); poisson=false)

    for o in (0, 1, 7), t in 1:(CO_L - 7)
        @test SSD._regime(SSD._with_cost_offset(sm, o), t) == sm.schedule[t + o]
        @test SSD._regime_at(sm, t + o) == sm.schedule[t + o]
    end
    # No offset to change is no copy at all.
    @test SSD._with_cost_offset(sm, 0) === sm
    @test SSD._with_cost_offset(lds, 0) === lds
    shifted = SSD._with_cost_offset(sm, 4)
    @test shifted !== sm && shifted.cache === sm.cache && shifted.Qc === sm.Qc
    @test sm.cost_offset == 0                     # the fitted model is untouched
    # A shifted model's terminal factor follows the schedule at its own last bin,
    # unless pinned.
    unpinned, _ = _co_model(StableRNG(1); poisson=false, pin=false)
    @test SSD._terminal_regime(SSD._with_cost_offset(unpinned, 3), 14) ==
        unpinned.schedule[17]
    @test SSD._terminal_regime(SSD._with_cost_offset(sm, 3), 14) == 3

    # Validation.
    ys = [randn(StableRNG(2), lds.obs_dim, t) for t in (5, 6)]
    @test_throws ArgumentError SSD.Data(lds, ys; cost_offset=[0, -1])
    @test_throws SSD.DimensionMismatchError SSD.Data(lds, ys; cost_offset=[0, 1, 2])
    @test SSD.Data(lds, ys).cost_offset == Int[]
    @test SSD.Data(lds, ys; cost_offset=[2, 0]).cost_offset == [2, 0]
    # A model with no cost schedule has nothing to offset.
    gsm = GaussianStateModel(
        0.9 * Matrix(I, 2, 2), Matrix(0.1I, 2, 2), zeros(2), zeros(2), Matrix(1.0I, 2, 2)
    )
    glds = LinearDynamicalSystem(
        gsm,
        GaussianObservationModel(randn(StableRNG(3), 3, 2), Matrix(0.1I, 3, 3), zeros(3)),
    )
    gys = [randn(StableRNG(4), 3, 5) for _ in 1:2]
    @test_throws ArgumentError SSD.Data(glds, gys; cost_offset=[1, 0])
    @test SSD.Data(glds, gys; cost_offset=[0, 0]).cost_offset == Int[]

    # The schedule has to cover offset + length, not the length alone.
    plds = _co_model(StableRNG(5))[2]
    long = _co_counts(StableRNG(6), 4, [14])
    @test isfinite(elbo(plds, long; cost_offset=[10]))
    @test_throws SSD.DimensionMismatchError elbo(plds, long; cost_offset=[11])
    return nothing
end

function test_cost_offset_terminal_logz()
    #=
    The terminal normalizer integrates backwards from each trial's last bin, so
    its regimes are read at `offset + h - j`. Three shapes of dataset take three
    routes through `_lqr_terminal_logz_sum`: one running cost (every design
    batched together), several running costs with every trial ending at the same
    schedule bin, and several ending at different ones.
    =#
    rng = StableRNG(7)
    single = _pinned_ragged_lqr(rng; tmax=CO_L)             # one running cost
    multi, _ = _co_model(rng; poisson=false)                # three costs, pinned
    n_in = size(single.Bu, 2)

    shapes = Dict(
        # Same end bin (24) for every trial, ragged starts: one group.
        :common_end => [(14, 10), (20, 4), (9, 15), (20, 4), (24, 0)],
        # Different ends.
        :mixed_ends => [(14, 0), (15, 3), (12, 5), (15, 3), (16, 8), (14, 0)],
    )
    for (label, hw) in shapes, sm in (single, multi)
        m = size(sm.Bu, 2)
        designs = [
            (ux=randn(rng, m, h), ux0=zeros(0), off=off, count=c) for
            ((h, off), c) in zip(hw, (2, 1, 3, 1, 2, 2))
        ]
        each(d) = SSD._lqr_terminal_logz(SSD._with_cost_offset(sm, d.off), d.ux)
        reference = sum(d.count * each(d) for d in designs)
        @test SSD._lqr_terminal_logz_sum(sm, designs) ≈ reference rtol = 1e-11
        # And against the forward-moment reference, which shares no code with it.
        forward = sum(
            d.count * lqr_terminal_logz_reference(
                SSD._with_cost_offset(sm, d.off), d.ux, size(d.ux, 2)
            ) for d in designs
        )
        @test reference ≈ forward rtol = 1e-8
        # A single design, and a subset whose ends differ.
        for pick in (designs[1:1], designs[[1, 3]])
            ref = sum(d.count * each(d) for d in pick)
            @test SSD._lqr_terminal_logz_sum(sm, pick) ≈ ref rtol = 1e-11
        end
    end
    @test n_in == 3

    # Offsets matter: shifting a design on a multi-cost schedule moves log Z.
    ux = randn(rng, size(multi.Bu, 2), 12)
    a = SSD._lqr_terminal_logz_sum(multi, [(ux=ux, ux0=zeros(0), off=0, count=1)])
    b = SSD._lqr_terminal_logz_sum(multi, [(ux=ux, ux0=zeros(0), off=9, count=1)])
    @test !isapprox(a, b; rtol=1e-6)
    return nothing
end

function test_cost_offset_matches_sliced_schedule()
    for pin in (true, false)
        sm, plds = _co_model(StableRNG(10); pin=pin)
        p = plds.obs_dim
        ys = _co_counts(StableRNG(11), p, CO_LENGTHS)
        ntr = length(ys)

        # Posterior, trial by trial.
        res = smooth(plds, ys; cost_offset=CO_OFFSETS)
        xs, ps = res[1], res[2]
        for i in 1:ntr
            ref = smooth(_co_sliced(plds, CO_OFFSETS[i]), [ys[i]])
            @test xs[i] ≈ ref[1][1] rtol = 1e-8 atol = 1e-10
            @test ps[i] ≈ ref[2][1] rtol = 1e-8 atol = 1e-10
        end

        # ELBO: per trial, and in total.
        per = trial_elbos(plds, ys; cost_offset=CO_OFFSETS)
        for i in 1:ntr
            ref = trial_elbos(_co_sliced(plds, CO_OFFSETS[i]), [ys[i]])[1]
            @test per[i] ≈ ref rtol = 1e-9
        end
        @test elbo(plds, ys; cost_offset=CO_OFFSETS) ≈ sum(per) rtol = 1e-9
        # The offsets are doing something.
        @test !isapprox(elbo(plds, ys), elbo(plds, ys; cost_offset=CO_OFFSETS); rtol=1e-5)

        # The normalizer reported alongside a fit.
        ux = [zeros(0, t) for t in CO_LENGTHS]
        logz = terminal_normalizer(plds, ux; cost_offset=CO_OFFSETS)
        for i in 1:ntr
            ref = terminal_normalizer(_co_sliced(plds, CO_OFFSETS[i]), [ux[i]])[1]
            @test logz[i] ≈ ref rtol = 1e-10
        end

        # Per-regime transition counts follow each trial's own slice of the schedule.
        data = SSD.Data(plds, ys; cost_offset=CO_OFFSETS)
        SSD._prepare_lqr!(plds, data)
        tfs = SSD.initialize_FilterSmooth(plds, data.tsteps)
        pool = SSD._lqr_sws_pool(plds, data)
        hs = SSD._initialize_td_sufficient_statistics(Float64, plds, data.tsteps)
        SSD._td_init_const_blocks!(pool[1], plds, data)
        SSD.estep!(plds, hs, tfs, data, pool)
        K = length(sm.Qc)
        expected = zeros(K)
        expected_terminal = zeros(K)
        for (t_n, o) in zip(CO_LENGTHS, CO_OFFSETS)
            for t in 1:(t_n - 1)
                expected[sm.schedule[t + o]] += 1
            end
            expected_terminal[pin ? K : sm.schedule[t_n + o]] += 1
        end
        @test hs.nk ≈ expected
        @test hs.term_n ≈ expected_terminal
    end
    return nothing
end

function test_cost_offset_uniform_equals_slice_fit()
    #=
    Every trial starting at the same offset is a dataset on a schedule that
    starts there. Fitting it both ways runs the whole stack down two routes —
    the terminal-conditioning probe smooths with offset buckets in one and the
    ordinary ragged smoother in the other — and they are the same estimator.
    =#
    offset = 4
    lengths = [12, 16, 12, 20, 16, 9]
    for pin in (true, false)
        _, a = _co_model(StableRNG(20); pin=pin)
        _, b = _co_model(StableRNG(20); pin=pin)
        b = _co_sliced(b, offset)
        ys = _co_counts(StableRNG(21), a.obs_dim, lengths)
        ea = fit!(a, ys; max_iter=5, progress=false, cost_offset=fill(offset, length(ys)))
        eb = fit!(b, ys; max_iter=5, progress=false)
        @test all(isfinite, ea)
        @test ea ≈ eb rtol = 1e-6
        for k in eachindex(a.state_model.Qc)
            @test a.state_model.Qc[k] ≈ b.state_model.Qc[k] rtol = 1e-4 atol = 1e-8
        end
        @test a.state_model.A ≈ b.state_model.A rtol = 1e-4
        @test a.state_model.cost_offset == 0     # fitting never leaves an offset behind
    end
    return nothing
end

function test_cost_offset_ragged_fit()
    sm, plds = _co_model(StableRNG(30))
    ys = _co_counts(StableRNG(31), plds.obs_dim, CO_LENGTHS)
    test_offsets = [2, 0, 6]
    test_ys = _co_counts(StableRNG(32), plds.obs_dim, [13, 11, 15])

    # The trace starts at the held-out score of the pre-fit model, read at the test
    # set's own offsets.
    pre_fit = elbo(deepcopy(plds), test_ys; cost_offset=test_offsets)
    els = fit!(
        plds,
        ys;
        max_iter=6,
        progress=false,
        cost_offset=CO_OFFSETS,
        y_test=test_ys,
        cost_offset_test=test_offsets,
    )
    @test all(isfinite, els)
    @test els[end] > els[1]
    @test length(els.test) >= 1 && all(isfinite, els.test)
    @test els.test[1] ≈ pre_fit rtol = 1e-9
    @test sm.cost_offset == 0

    # A parameter grouping (`depends_on`) goes through the same offsets.
    sm2, plds2 = _co_model(StableRNG(33))
    group = [1, 2, 1, 2, 1, 2]
    set_depends_on!(sm2, (structure=group,))
    grouped = LinearDynamicalSystem(sm2, plds2.obs_model)
    gels = fit!(grouped, ys; max_iter=4, progress=false, cost_offset=CO_OFFSETS)
    @test all(isfinite, gels)
    @test isfinite(elbo(grouped, ys; cost_offset=CO_OFFSETS))

    #=
    A grouping with one group is the same estimator as no grouping. The grouped
    path pools its cells' statistics, terminal designs and their offsets
    included, so a dropped offset there shows up as a different ELBO.
    =#
    smA, ldsA = _co_model(StableRNG(60))
    _, ldsB = _co_model(StableRNG(60))
    one_group = fill(1, length(ys))
    set_depends_on!(smA, (structure=one_group, noise=one_group))
    grouped_one = LinearDynamicalSystem(smA, ldsA.obs_model)
    ea = fit!(grouped_one, ys; max_iter=4, progress=false, cost_offset=CO_OFFSETS)
    eb = fit!(ldsB, ys; max_iter=4, progress=false, cost_offset=CO_OFFSETS)
    @test maximum(abs, ea .- eb) < 1e-6

    # A composite emission has the same trial-by-trial smoother.
    sm3, lds3 = _co_model(StableRNG(34); poisson=false)
    n = SSD.plant_dim(sm3)
    C = randn(StableRNG(35), 3, 2n) .* 0.3
    C[:, (n + 1):(2n)] .= 0
    comp = LinearDynamicalSystem(
        sm3,
        (
            spk=PoissonObservationModel(C, fill(1.0, 3)),
            kin=GaussianObservationModel(
                [C[1:2, 1:n] zeros(2, n)], Matrix(0.1I, 2, 2), zeros(2)
            ),
        ),
    )
    y = (
        spk=_co_counts(StableRNG(36), 3, CO_LENGTHS),
        kin=[randn(StableRNG(37 + i), 2, t) for (i, t) in enumerate(CO_LENGTHS)],
    )
    @test isfinite(elbo(comp, y; cost_offset=CO_OFFSETS))
    per = trial_elbos(comp, y; cost_offset=CO_OFFSETS)
    @test elbo(comp, y; cost_offset=CO_OFFSETS) ≈ sum(per) rtol = 1e-9
    return nothing
end

function test_cost_offset_sampling()
    # The same draw from the same stream, whether the offset is passed or the
    # schedule is cut where the trial starts.
    _, plds = _co_model(StableRNG(40))
    for offset in (0, 3, 9)
        z1, y1 = rand(StableRNG(41), plds, 12; cost_offset=offset)
        z2, y2 = rand(StableRNG(41), _co_sliced(plds, offset), 12)
        @test z1 ≈ z2 && y1 == y2
    end
    # Several trials share one stream, so the first is the one to compare.
    z1, _ = rand(StableRNG(42), plds, [10, 12, 9]; cost_offset=[2, 5, 8])
    z2, _ = rand(StableRNG(42), _co_sliced(plds, 2), [10, 12, 9])
    @test z1[1] ≈ z2[1]
    return nothing
end

function test_cost_offset_gaussian_emission()
    #=
    A Gaussian-only emission *fits* with one covariance shared across the trials
    of a length, which assumes a shared schedule start, so `fit!` and the grouped
    ELBO refuse offsets rather than return a wrong answer. Its smoother handles
    them — the terminal-conditioning probe needs it to, and held-out scoring
    builds behaviour-only Gaussian sub-models — so the evaluation paths take them
    and are checked against the sliced-schedule reference.
    =#
    sm, glds = _co_model(StableRNG(50); poisson=false)
    ys = [
        randn(StableRNG(51 + i), glds.obs_dim, t) .* 0.4 for (i, t) in enumerate(CO_LENGTHS)
    ]
    @test_throws ArgumentError fit!(
        glds, ys; max_iter=2, progress=false, cost_offset=CO_OFFSETS
    )
    zeros_ = zeros(Int, length(ys))
    @test elbo(glds, ys; cost_offset=zeros_) ≈ elbo(glds, ys) rtol = 1e-12

    res = smooth(glds, ys; cost_offset=CO_OFFSETS)
    per = trial_elbos(glds, ys; cost_offset=CO_OFFSETS)
    for i in eachindex(ys)
        sliced = _co_sliced(glds, CO_OFFSETS[i])
        ref = smooth(sliced, [ys[i]])
        @test res[1][i] ≈ ref[1][1] rtol = 1e-9 atol = 1e-11
        @test res[2][i] ≈ ref[2][1] rtol = 1e-9 atol = 1e-11
        @test per[i] ≈ trial_elbos(sliced, [ys[i]])[1] rtol = 1e-9
    end
    @test elbo(glds, ys; cost_offset=CO_OFFSETS) ≈ sum(per) rtol = 1e-9
    @test loglikelihood(glds, ys; cost_offset=CO_OFFSETS) ≈ sum(per) rtol = 1e-9

    # A grouped Gaussian ELBO shares covariance across a cell's trials of a length.
    sm2, glds2 = _co_model(StableRNG(52); poisson=false)
    set_depends_on!(sm2, (structure=[1, 2, 1, 2, 1, 2],))
    grouped = LinearDynamicalSystem(sm2, glds2.obs_model)
    @test_throws ArgumentError elbo(grouped, ys; cost_offset=CO_OFFSETS)
    return nothing
end

"""Variable starts share a backward precision suffix only within one endpoint.

Compare every mean, covariance, lag covariance and entropy to independent
single-trial smooths, including fresh initial means and a second parameter set.
The one-workspace path is the full-factorization reference; larger pools take
the shared suffix path and must keep different endpoints in different buckets.
"""
function test_cost_offset_shared_suffix()
    lengths = [24, 20, 20, 15, 10, 8, 5, 3, 23, 20, 14, 9, 7, 4, 2]
    offsets = vcat(24 .- lengths[1:8], 23 .- lengths[9:end])
    for terminal in (false, true), pin in (false, true), width in (1, 2, 4)
        rng = StableRNG(20261008)
        sm, lds = _co_model(rng; poisson=false, pin=pin)
        sm.terminal = terminal
        sm.B0 = 0.1 .* randn(rng, lds.latent_dim, 2)
        ux0 = randn(rng, 2, length(lengths))
        ux = [randn(rng, lds.ux_dim, h) for h in lengths]
        ys = [randn(rng, lds.obs_dim, h) for h in lengths]
        data = SSD.Data(lds, ys; ux=ux, ux0=ux0, cost_offset=offsets)
        SSD._prepare_lqr!(lds, data; offsets_ok=true)
        pool = SSD._lqr_sws_pool(lds, data)
        while length(pool) < width
            push!(pool, deepcopy(first(pool)))
        end
        pool = pool[1:width]
        tfs = SSD.initialize_FilterSmooth(lds, lengths)
        for repeat in 1:2
            if repeat == 2
                sm.A .*= 0.99
                sm.Qc[1] .*= 1.1
                sm.Σf .*= 1.2
                refresh!(sm)
            end
            SSD.smooth!(lds, tfs, data, pool)
            @test tfs[2].p_smooth === tfs[3].p_smooth
            @test tfs[2].p_smooth !== tfs[10].p_smooth
            for i in eachindex(ys)
                model = SSD._trial_model(lds, data, i)
                ref = SSD.initialize_FilterSmooth(model, [lengths[i]])[1]
                ws = SSD.SmoothWorkspace(Float64, lds.latent_dim, lds.obs_dim, lengths[i])
                SSD.smooth!(model, ref, ys[i], ws, ux[i], SSD._trial(data.uy, i))
                @test tfs[i].x_smooth ≈ ref.x_smooth rtol = 1e-10 atol = 1e-10
                @test tfs[i].p_smooth ≈ ref.p_smooth rtol = 1e-10 atol = 1e-10
                @test tfs[i].p_smooth_tt1[:, :, 2:end] ≈ ref.p_smooth_tt1[:, :, 2:end] rtol =
                    1e-10 atol = 1e-10
                @test tfs[i].entropy ≈ ref.entropy rtol = 1e-12
            end
        end
    end
    return nothing
end

"""The shared-suffix probe's Fisher gradient matches the exact QR normalizer."""
function test_cost_offset_shared_suffix_gradient()
    rng = StableRNG(20261009)
    sm, lds = _co_model(rng; poisson=false)
    sm.B0 = 0.05 .* randn(rng, lds.latent_dim, 2)
    lengths = [24, 20, 15, 9, 23, 19, 14, 8]
    offsets = vcat(24 .- lengths[1:4], 23 .- lengths[5:end])
    ys = [0.2 .* randn(rng, lds.obs_dim, h) for h in lengths]
    ux = [randn(rng, lds.ux_dim, h) for h in lengths]
    data = SSD.Data(lds, ys; ux=ux, ux0=randn(rng, 2, length(lengths)), cost_offset=offsets)
    SSD._prepare_lqr!(lds, data; offsets_ok=true)
    tfs = SSD.initialize_FilterSmooth(lds, lengths)
    pool = SSD._lqr_sws_pool(lds, data)
    hs = SSD._initialize_td_sufficient_statistics(Float64, lds, lengths)
    SSD._td_init_const_blocks!(pool[1], lds, data)
    SSD.estep!(lds, hs, tfs, data, pool)
    slots = [ones(Int, 1) for _ in 1:4]
    problem = SSD._lqr_conditional_problem([lds], [hs], slots)
    theta = copy(problem.theta)
    analytic = similar(theta)
    @test isfinite(problem.evaluate!(analytic, theta))
    fd = similar(theta)
    for i in eachindex(theta)
        step = 1e-6 * max(1.0, abs(theta[i]))
        up, down = copy(theta), copy(theta)
        up[i] += step
        down[i] -= step
        fd[i] =
            (problem.evaluate!(nothing, up) - problem.evaluate!(nothing, down)) / (2step)
    end
    problem.write!(theta)
    @test maximum(abs, analytic - fd) / max(1.0, maximum(abs, fd)) < 1e-7
    return nothing
end
