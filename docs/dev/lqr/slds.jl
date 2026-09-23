#=============================================================================
Switching recovery — an `SLDS` with one `:free` discrete state and one
inverse-LQR state.

This is the configuration the type is *for*: `free_state_model` exists so an
`SLDS` can mix plain linear dynamics with LQR dynamics, and the two share one
`2n`-dimensional latent `z = [x; λ]`, so the switching is over which transition —
drift, or the solution of a control problem — generated each step.

Two generators, as in the single-system harness, and for the same reason.

  * `:rand` — the SLDS's own forward chain. Estimation is well posed, but an
    LQR matrix has reciprocal eigenvalue pairs, so the forward flow of the LQR
    state is unstable by construction and every dwell in it grows like `ρ(M)^ℓ`.
    `sscale` exists to hold that down; the tables print `ρ(M)` so it can be read
    rather than assumed.
  * `:epoch` — a delay-then-reach trial: the latent drifts under the free state
    up to a per-trial switch time, and from there the agent rolls the *optimal*
    trajectory (`simulate_lqr`, on the stable manifold) to the end of the trial,
    with the terminal condition and the reference in force. This is the
    behaviourally meaningful one, and it is the only one that stays on a
    realistic scale.

## Where the epoch generator is and is not the model

`z_t` selects the transition *into* `t`, so with `z_t` the free state for
`t ≤ t_sw` and the LQR state after, the LQR state governs the transitions
`t_sw → t_sw+1` onward. The optimal
rollout satisfies the model's own stationarity conditions exactly on that window
— including `λ_{t_sw} = P_{t_sw} x_{t_sw} + g_{t_sw}`, which is why the splice
below overwrites the costate at `t_sw` as well as after it. The cost of that is
one mismatched transition per trial, at the free state's *last* step, where the
generated costate jumps to the value the reach requires. That is the right place
to put it: the free state's `Σ` is unconstrained and can absorb it, whereas the
LQR state's cannot.

## Why the LQR state is discrete state 1

`LDSs = [lqr, free]`, not the other way round, and the delay epoch is therefore
state **2**. That ordering is forced, not stylistic. The switching M-step
allocates its responsibility-weighted statistics as `K` copies of the shape
`slds.LDSs[1]` needs (`fit_SLDS.jl`, `_initialize_td_sufficient_statistics(T,
slds.LDSs[1], …)`), so every discrete state gets containers sized for the
*first* one. A `:free` state carries one cost regime; an LQR state with a
terminal factor carries two. Put the free state first and the LQR state's
per-regime blocks are under-allocated, and the aggregator throws `BoundsError`
on `term_zz[2]`. Putting the LQR state first over-allocates the free state's
blocks instead, which is harmless — a `:free` model only ever touches regime 1.

`LQR_STATE` and `FREE_STATE` below name the indices so nothing in the scoring
depends on remembering that.

Note also what `rand` cannot do here. `_extract_state_params` hands the sampler
`cache.M[1]` — one transition per discrete state, because an `SLDS` member is
meant to carry one cost. That is exact for the models used below (with a
terminal factor and no onset, every *transition* is regime 1 and the terminal
regime is read only by the endpoint factor), but it would silently ignore a
within-state cost schedule. The `:epoch` generator has no such limit.
=============================================================================#

"""Discrete-state indices. See the note above: the order is forced by the
switching M-step's allocation, not chosen."""
const LQR_STATE = 1
const FREE_STATE = 2

"""
    SldsTruth

The generating switching system, plus what the scoring needs: the free and LQR
state models, the shared emission, the LQR state's regime map, and the range the
per-trial switch time is drawn from.
"""
struct SldsTruth
    slds::Any
    free_sm::Any
    lqr_sm::Any
    C::Matrix{Float64}
    R::Matrix{Float64}
    idx::NamedTuple
    nref::Int
    tsteps::Int
    n::Int
    onset_range::UnitRange{Int}
end

"""
    free_drift(n; decay, costate_decay, noise) -> (M, Σ)

The `:free` state's unconstrained transition: a slow drift on the observed state
block and a contraction on the costate block.

The costate half is the awkward part of a mixed free/LQR `SLDS`. Both discrete
states write the same `2n`-dimensional latent, so a free state has a costate
block whether or not the notion means anything for it — and with `C = [I 0]`
nothing observes it. A contraction there keeps it from wandering off and
supplying the fit with a second, unidentified way to explain the data; it also
makes the free state's *observable* content exactly `M[1:n, 1:n]`, which is the
only block of it the sweeps score.
"""
function free_drift(n::Int; decay::Float64=0.90, costate_decay::Float64=0.6,
                    noise::Float64=0.15)
    d = 2n
    M = zeros(d, d)
    M[1:n, 1:n] .= Matrix(decay * I, n, n)
    M[(n + 1):d, (n + 1):d] .= Matrix(costate_decay * I, n, n)
    return M, Matrix(noise * I, d, d)
end

"""
    slds_truth(; kwargs...) -> SldsTruth

The generating switching model. `stay` is the HMM's self-transition probability
(used by the `:rand` generator only — `:epoch` imposes a single switch per trial
instead), and `sscale` shrinks the control term `S`, which is the handle on the
LQR state's forward instability: `ρ(M)` falls toward `max(|μ(A)|, 1/|μ(A)|)` as
`S Q → 0`.
"""
function slds_truth(;
    n::Int=2,
    tsteps::Int=30,
    terminal::Bool=true,
    nref::Int=0,
    stay::Float64=0.93,
    sscale::Float64=1.0,
    obs_noise::Float64=0.05,
    observe_costate::Bool=false,
    state_noise::Float64=0.02,
    #=
    Looser than the single-system default (`1e-4`), and deliberately so — this
    is the one place the two harnesses disagree about the same number.

    For a single inverse-LQR system a tight costate innovation is what removes
    the `n` directions of slack the cost would otherwise drift along, and the
    ladder in `experiment_initialization` picks `1e-4` as its optimum. For a
    *switching* model the same number does the opposite: `γ` is a plug-in scored
    at the smoothed mean, the smoothed mean never sits exactly on the Riccati
    graph, and a `Σ_λλ` of `1e-4` therefore makes the LQR state's per-timestep
    likelihood hopeless at every timestep. The fit puts everything in the free
    state and `γ` sits at chance — at the *generating* parameters, not only in
    the fit. Measured: γ at the truth is 0.52 at `1e-4`, 0.56 at `2.5e-3`, 0.74
    at `1e-2` and 0.81 at `2e-2`.

    So the costate innovation is doing two different jobs. In one system it is
    slack to be removed; in a switching one it is the tolerance that makes the
    LQR state selectable at all. `experiment_switching` sweeps it rather than
    leaving the disagreement implicit.
    =#
    costate_noise::Float64=2e-2,
    onset_range::UnitRange{Int}=(tsteps ÷ 3):(2tsteps ÷ 3),
)
    d = 2n
    A, S = plant(n)
    S .*= sscale
    Qc, idx = cost_bank(n; terminal=terminal, onset=1)
    sched = schedule_for(tsteps, idx; terminal=terminal, onset=1)
    lqr_sm = LQRStateModel(
        A,
        S,
        Qc,
        mixed_noise(n; state=state_noise, costate=costate_noise);
        schedule=sched,
        terminal=terminal,
        Σf=Matrix(0.02I, n, n),
        P0=Matrix(0.2I, d, d),
        Bu=nref > 0 ? zeros(d, nref) : nothing,
        Gref=nref > 0 ? reference_map(n, nref) : nothing,
        observe_costate=observe_costate,
    )
    M, Σf = free_drift(n)
    free_sm = free_state_model(
        M,
        Σf;
        P0=Matrix(0.2I, d, d),
        Bu=nref > 0 ? zeros(d, nref) : nothing,
        observe_costate=observe_costate,
    )
    obs_dim = obs_width(n; free_C=false, observe_costate=observe_costate)
    C = emission(
        n, obs_dim; free_C=false, observe_costate=observe_costate, rng=MersenneTwister(0)
    )
    R = Matrix(obs_noise * I, obs_dim, obs_dim)
    function mk(sm)
        return LinearDynamicalSystem(
            sm, GaussianObservationModel(copy(C), copy(R), zeros(obs_dim))
        )
    end
    slds = SLDS(;
        A=[stay (1 - stay); (1 - stay) stay],
        # Trials start in the delay epoch, which is the free state.
        πₖ=[1e-6, 1.0 - 1e-6],
        LDSs=[mk(lqr_sm), mk(free_sm)],
    )
    return SldsTruth(slds, free_sm, lqr_sm, C, R, idx, nref, tsteps, n, onset_range)
end

"""
    segment_model(truth, len) -> LQRStateModel

The LQR state restricted to the last `len` timesteps of a trial, which is the
model the `:epoch` agent actually solves. The regimes are the same objects, only
the schedule is re-indexed: segment step `s` is absolute time `T - len + s`, so
a terminal regime stays in the last slot and the running cost covers the rest.

Built per distinct segment length and cached by the caller — a trial's onset
sets the horizon, and the Riccati sweep depends on it.
"""
function segment_model(truth::SldsTruth, len::Int)
    sm = truth.lqr_sm
    n = truth.n
    full = isempty(sm.schedule) ? fill(1, truth.tsteps) : sm.schedule
    return LQRStateModel(
        copy(sm.A),
        copy(sm.S),
        [copy(Q) for Q in sm.Qc],
        copy(sm.Σ);
        schedule=full[(truth.tsteps - len + 1):end],
        terminal=sm.terminal,
        Σf=copy(sm.Σf),
        P0=copy(sm.P0),
        Bu=size(sm.Bu, 2) > 0 ? copy(sm.Bu) : nothing,
        Gref=size(sm.Gref, 2) > 0 ? copy(sm.Gref) : nothing,
    )
end

"""
    simulate_slds(rng, truth, ntrials; gen, slack, process_noise, uxs)
        -> (ys, zs, xs)

Observations, the true discrete path, and the true latent path.

`:rand` rolls the SLDS's own chain. `:epoch` builds the delay-then-reach trial
described at the top of this file: free drift up to a per-trial switch, then the
optimal trajectory of the remaining horizon spliced in, costate included.
"""
function simulate_slds(
    rng::AbstractRNG,
    truth::SldsTruth,
    ntrials::Int;
    gen::Symbol=:epoch,
    slack::Float64=0.05,
    process_noise::Bool=true,
    uxs=nothing,
)
    gen in (:rand, :epoch) ||
        throw(ArgumentError("gen must be :rand or :epoch; got :$gen"))
    T, n, d = truth.tsteps, truth.n, 2 * truth.n
    obs_dim = size(truth.C, 1)
    if gen === :rand
        zs, xs, ys = rand(rng, truth.slds, fill(T, ntrials); ux=uxs)
        return ys, zs, xs
    end

    L = cholesky(Symmetric(truth.R)).L
    Lfree = cholesky(Symmetric(truth.free_sm.Σ)).L
    LP0 = cholesky(Symmetric(Matrix(truth.free_sm.P0))).L
    segs = Dict{Int,Any}()
    ys = Vector{Matrix{Float64}}(undef, ntrials)
    zs = Vector{Vector{Int}}(undef, ntrials)
    xs = Vector{Matrix{Float64}}(undef, ntrials)
    for i in 1:ntrials
        t_sw = rand(rng, truth.onset_range)
        z = Matrix{Float64}(undef, d, T)
        z[:, 1] .= truth.free_sm.x0 .+ LP0 * randn(rng, d)
        for t in 2:t_sw
            @views z[:, t] .= truth.free_sm.Mfree * z[:, t - 1] .+ truth.free_sm.h .+
                              Lfree * randn(rng, d)
        end
        len = T - t_sw + 1
        seg = get!(() -> segment_model(truth, len), segs, len)
        roll = simulate_lqr(
            rng,
            seg,
            len;
            x1=Vector(z[1:n, t_sw]),
            costate_slack=slack,
            process_noise=process_noise,
            ux=(uxs === nothing ? nothing : uxs[i][:, t_sw:T]),
        )
        #=
        Splice the whole window, costate included: the optimal path satisfies
        the LQR stationarity conditions at `t_sw` too, so overwriting `λ_{t_sw}`
        is what makes every LQR-governed transition an exact draw from the model
        (up to `slack`). The jump it creates lands in the free state's last
        residual, which is unconstrained.
        =#
        @views z[:, t_sw:T] .= roll
        zs[i] = [t <= t_sw ? FREE_STATE : LQR_STATE for t in 1:T]
        xs[i] = z
        ys[i] = truth.C * z .+ L * randn(rng, obs_dim, T)
    end
    return ys, zs, xs
end

"""
    fit_slds(truth; known_plant, free_gref, stay_init) -> SLDS

A switching model with the truth's structure and wrong values: a cold cost on
the LQR state, an over-contracting drift and inflated noise on the free state,
and a transition matrix started at a generic stickiness rather than the truth's.

`πₖ` starts uniform even where the truth is degenerate. A prior pinned at the
answer would hand the fit the first timestep of every trial for free, which on a
delay-then-reach design is most of what identifies the onset.

Both state models freeze `h` whenever the reference input is present. Its
one-hot columns already span the intercept, so leaving `h` free would fit a
rank-deficient parameterization without adding a model the data can distinguish.
"""
function fit_slds(
    truth::SldsTruth;
    known_plant::Bool,
    free_gref::Bool,
    stay_init::Float64=0.9,
    sig0_state::Float64=0.05,
    sig0_costate::Float64=5e-2,
    sigma_prior_strength::Float64=0.0,
    sigma_prior_costate::Float64=1e-4,
    qc_prior_strength::Float64=0.0,
    qc_prior_scale::Union{Float64,Vector{Float64}}=0.2,
    free_noise0::Float64=0.12,
    free_decay0::Float64=0.85,
    init::Symbol=:cold,
    fit_noise::Bool=true,
    fit_structure::Bool=true,
)
    n, d = truth.n, 2 * truth.n
    lqr_sm = deepcopy(truth.lqr_sm)
    if init !== :warm
        for Q in lqr_sm.Qc
            Q .= Matrix(0.4I, n, n)
        end
        lqr_sm.Σ .= mixed_noise(n; state=sig0_state, costate=sig0_costate)
        lqr_sm.h .= 0
        free_gref && size(lqr_sm.Gref, 2) > 0 && (lqr_sm.Gref .*= -0.3)
        if !known_plant
            lqr_sm.A .= 0.90 .* truth.lqr_sm.A .+ Matrix(0.03I, n, n)
            lqr_sm.S .= 1.5 .* truth.lqr_sm.S
        end
    end
    #=
    The principled alternative to pinning `Σ`. Pinning is a concession — the fit
    is told the innovation — and every switching row that works rests on it. A
    prior says the same thing with a strength attached, and leaves the M-step
    free to disagree with it where the data insist.
    =#
    lqr_sm.Σ_prior = sigma_prior(
        n;
        state=0.02,
        costate=sigma_prior_costate,
        strength=sigma_prior_strength,
    )
    lqr_sm.Qc_prior = qc_prior(n; scale=qc_prior_scale, strength=qc_prior_strength)
    lqr_sm.fit_flags = LQRFitFlags(;
        A=(!known_plant), S=(!known_plant), Gref=free_gref,
        h=(truth.nref == 0), Bu=false,
    )
    refresh!(lqr_sm)

    M0, Σ0 = free_drift(n; decay=free_decay0, costate_decay=0.85, noise=free_noise0)
    free_sm = free_state_model(
        M0,
        Σ0;
        P0=Matrix(0.2I, d, d),
        Bu=truth.nref > 0 ? zeros(d, truth.nref) : nothing,
        observe_costate=truth.lqr_sm.observe_costate,
        fit_flags=LQRFitFlags(; h=(truth.nref == 0), Bu=false),
    )
    obs_dim = size(truth.C, 1)
    function mk(sm)
        return LinearDynamicalSystem(
            sm, GaussianObservationModel(copy(truth.C), copy(truth.R), zeros(obs_dim))
        )
    end
    ldss = [mk(lqr_sm), mk(free_sm)]
    for lds in ldss
        lds.fit_bool[5] = false     # the emission is fixed; see `emission`
        lds.fit_bool[3] = fit_structure
        lds.fit_bool[4] = fit_noise
    end
    #=
    The free state is always free. It is the harness's catch-all, and freezing
    it would make "the LQR state explains everything" a statement about the
    initialization rather than about the data.
    =#
    ldss[FREE_STATE].fit_bool[3] = true
    ldss[FREE_STATE].fit_bool[4] = true
    return SLDS(;
        A=[stay_init (1 - stay_init); (1 - stay_init) stay_init],
        πₖ=[0.5, 0.5],
        LDSs=ldss,
    )
end

"""
    recover_slds(; kwargs...) -> NamedTuple

Simulate a switching system, refit it from a wrong start, and report both halves
of the answer: how well the LQR state's structural parameters came back, and how
well the posterior over discrete states recovered the epochs that generated the
data.

The γ scores are the point of this section. A parameter table can look fine
while the fit has assigned the wrong timesteps to the wrong state — the two are
not the same question, and a switching fit can fail either one alone.
"""
function recover_slds(;
    n::Int=2,
    tsteps::Int=30,
    ntrials::Int=120,
    gen::Symbol=:epoch,
    terminal::Bool=true,
    nref::Int=0,
    slack::Float64=0.05,
    process_noise::Bool=true,
    stay::Float64=0.93,
    sscale::Float64=1.0,
    obs_noise::Float64=0.05,
    state_noise::Float64=0.02,
    costate_noise::Float64=2e-2,
    observe_costate::Bool=false,
    known_plant::Bool=true,
    free_gref::Bool=(nref > 0),
    max_iter::Int=60,
    smoothing_iters::Int=1,
    stay_init::Float64=0.9,
    sig0_state::Float64=0.05,
    sig0_costate::Float64=5e-2,
    sigma_prior_strength::Float64=0.0,
    sigma_prior_costate::Float64=1e-4,
    qc_prior_strength::Float64=0.0,
    qc_prior_scale::Union{Float64,Vector{Float64}}=0.2,
    free_noise0::Float64=0.12,
    init::Symbol=:cold,
    fit_noise::Bool=true,
    fit_structure::Bool=true,
    #=
    Two-stage fitting, and the reason a switching LQR wants it more than a single
    system does. This section's own ladder shows an inversion: a loose `Σ_λλ` is
    what lets `γ` find the epochs, a tight one is what identifies the cost, and
    no single value does both. `anneal_costate` is the obvious response — fit
    once loose, reset the costate block tight, fit again from there — and it is
    here to be measured rather than assumed, since the single-system version of
    the same idea does not work.
    =#
    anneal_costate::Union{Nothing,Float64}=nothing,
    #=
    The posterior read separately from the fit, and read harder. An LQR state's
    forward flow is unstable, so the shared `q(x)` of a switching fit converges
    slowly — `smooth`'s default 100 inner iterations warns rather than converges
    on these models, and γ is the thing this section is about.
    =#
    post_iters::Int=400,
    seed::Int=1,
)
    rng = MersenneTwister(seed)
    truth = slds_truth(;
        n=n,
        tsteps=tsteps,
        terminal=terminal,
        nref=nref,
        stay=stay,
        sscale=sscale,
        obs_noise=obs_noise,
        observe_costate=observe_costate,
        state_noise=state_noise,
        costate_noise=costate_noise,
    )
    uxs = target_inputs(rng, nref, ntrials, tsteps)
    ys, zs, xs = simulate_slds(
        rng, truth, ntrials; gen=gen, slack=slack, process_noise=process_noise, uxs=uxs
    )
    #=
    γ at the *generating* parameters — the reference the fit is read against,
    and on a mixed free/LQR system it is nowhere near 1. With `C = [I 0]` the
    costate is inferred rather than observed, and the plug-in `γ` is scored at
    the smoothed mean, so a timestep is assigned to the LQR state only if the
    inferred `λ` already sits near the Riccati graph — which is what being in
    the LQR state would have caused. Reporting the fit without this beside it
    would charge EM for that circularity.

    It is a reference, not a ceiling: a fit is free to beat it, and does, since
    it may move the free state and the transition matrix to suit the data while
    this holds every one of them at the truth.
    =#
    truth_post = smooth(
        truth.slds, ys; ux=uxs, smoothing_iters=post_iters, progress=false
    )
    truth_elbo = truth_post.elbo
    truth_gamma = gamma_scores(truth_post.γ, zs; K=2)

    slds = fit_slds(
        truth;
        known_plant=known_plant,
        free_gref=free_gref,
        stay_init=stay_init,
        sig0_state=sig0_state,
        sig0_costate=sig0_costate,
        sigma_prior_strength=sigma_prior_strength,
        sigma_prior_costate=sigma_prior_costate,
        qc_prior_strength=qc_prior_strength,
        qc_prior_scale=qc_prior_scale,
        free_noise0=free_noise0,
        init=init,
        fit_noise=fit_noise,
        fit_structure=fit_structure,
    )
    elbos = fit!(
        slds,
        ys;
        ux=uxs,
        max_iter=max_iter,
        smoothing_iters=smoothing_iters,
        progress=false,
        rng=MersenneTwister(7seed),
    )
    elbos = elbos isa Tuple ? first(elbos) : elbos
    if anneal_costate !== nothing
        #=
        Stage two. The costate block is reset tight and the cross-blocks zeroed —
        shrinking the variance while leaving the fitted state-costate covariance
        at its old size makes `Σ` indefinite and `refresh!` fails its Cholesky.
        Everything else carries over, `γ` included, which is the point: stage one
        is there to find the epochs and stage two to sharpen the cost given them.
        =#
        sm2 = slds.LDSs[LQR_STATE].state_model
        nn, dd = truth.n, 2 * truth.n
        @views sm2.Σ[1:nn, (nn + 1):dd] .= 0
        @views sm2.Σ[(nn + 1):dd, 1:nn] .= 0
        @views sm2.Σ[(nn + 1):dd, (nn + 1):dd] .= Matrix(anneal_costate * I, nn, nn)
        refresh!(sm2)
        e2 = fit!(
            slds,
            ys;
            ux=uxs,
            max_iter=max_iter,
            smoothing_iters=smoothing_iters,
            progress=false,
            rng=MersenneTwister(11seed),
        )
        elbos = vcat(elbos, e2 isa Tuple ? first(e2) : e2)
    end
    post = smooth(slds, ys; ux=uxs, smoothing_iters=post_iters, progress=false)

    fit_lqr = slds.LDSs[LQR_STATE].state_model
    ref_lqr = deepcopy(truth.lqr_sm)
    rescale_costate!(ref_lqr; target=:trace)
    rescale_costate!(fit_lqr; target=:trace)
    γ = gamma_scores(post.γ, zs; K=2)
    gauge = gauge_compare(
        fit_lqr, ref_lqr, truth.idx,
        slds.LDSs[LQR_STATE].obs_model.C, truth.C;
        known_plant=known_plant, free_gref=free_gref, free_noise=fit_noise,
    )
    return (
        scores=gauge.raw,
        gauge=gauge,
        #=
        Only the state block of the free transition is scored. The costate block
        is unobserved under `C = [I 0]` and unconstrained by the model, so a
        number for it would be reporting the initialization plus whatever the
        smoother's prior did.
        =#
        Mfree=score(
            slds.LDSs[FREE_STATE].state_model.Mfree[1:n, 1:n],
            truth.free_sm.Mfree[1:n, 1:n],
        ),
        gamma=γ,
        truth_gamma=truth_gamma,
        onset=(
            gen === :epoch ? onset_error(post.γ, zs; lqr_state=LQR_STATE) :
            (bias=NaN, mad=NaN)
        ),
        elbo=post.elbo,
        truth_elbo=truth_elbo,
        iters=length(elbos),
        converged=length(elbos) < max_iter,
        creep=elbo_creep(elbos),
        rho=maximum(abs, eigvals(symplectic_matrix(truth.lqr_sm))),
        stay_fit=slds.A[1, 1],
        fit_sm=fit_lqr,
        ref_sm=ref_lqr,
        fit_slds=slds,
        idx=truth.idx,
        example=(y=ys[1], z=zs[1], x=xs[1], γ=post.γ[1], γ_truth=truth_post.γ[1]),
        elbos=elbos,
        ntrials=ntrials,
        tsteps=tsteps,
    )
end

# ---------------------------------------------------------------------------
# Segment-then-fit
# ---------------------------------------------------------------------------

"""
    segment_recover(; kwargs...) -> NamedTuple

Decouple the two problems the switching fit conflates: find the epochs, then fit
the cost *given* them.

The switching results say `γ` and the cost want opposite costate innovations and
that no single value or staged schedule gets both. That leaves an obvious
question the joint fit cannot answer — when the cost fails to come back, is it
because the epochs were wrong, or because a single-system inverse-LQR fit on
this much data would have failed anyway? This cuts the knot: take a
segmentation, slice out the LQR-governed timesteps, and hand them to the
single-system machinery, which the rest of the harness has already characterized.

`source` picks the segmentation:

  * `:oracle` — the true epochs. This is the *ceiling*: whatever it recovers is
    what perfect discrete-state inference would buy, and whatever it fails to
    recover is not the switching layer's fault.
  * `:fit` — the MAP path of a switching fit run at a loose costate innovation,
    which is the setting that finds epochs. This is the procedure a user could
    actually run.

The onset is fixed rather than drawn per trial (`onset_range` is a single value),
because a scheduled LQR model reads its terminal regime off `schedule[T_n]` and
ragged segments would land different trials on different regimes. Fixing it
costs the onset-detection question, which the switching table already covers,
and buys a segment fit that means what it says.
"""
function segment_recover(;
    n::Int=2,
    tsteps::Int=40,
    ntrials::Int=150,
    onset::Int=0,
    terminal::Bool=true,
    nref::Int=4,
    slack::Float64=0.05,
    obs_noise::Float64=0.05,
    observe_costate::Bool=false,
    source::Symbol=:oracle,
    known_plant::Bool=true,
    sig0_costate::Float64=1e-4,
    stage1_costate::Float64=2e-2,
    max_iter::Int=250,
    slds_iter::Int=50,
    seed::Int=1,
)
    source in (:oracle, :fit) ||
        throw(ArgumentError("source must be :oracle or :fit; got :$source"))
    t_sw = onset == 0 ? max(2, tsteps ÷ 2) : onset
    rng = MersenneTwister(seed)
    truth = slds_truth(;
        n=n,
        tsteps=tsteps,
        terminal=terminal,
        nref=nref,
        obs_noise=obs_noise,
        observe_costate=observe_costate,
        onset_range=t_sw:t_sw,
    )
    uxs = target_inputs(rng, nref, ntrials, tsteps)
    ys, zs, _ = simulate_slds(rng, truth, ntrials; gen=:epoch, slack=slack, uxs=uxs)

    zhat = if source === :oracle
        zs
    else
        slds = fit_slds(
            truth;
            known_plant=known_plant,
            free_gref=(nref > 0),
            sig0_state=0.02,
            sig0_costate=stage1_costate,
            fit_noise=false,
        )
        fit!(
            slds,
            ys;
            ux=uxs,
            max_iter=slds_iter,
            progress=false,
            rng=MersenneTwister(7seed),
        )
        g = smooth(slds, ys; ux=uxs, smoothing_iters=400, progress=false).γ
        [[argmax(view(γ, :, t)) for t in 1:tsteps] for γ in g]
    end

    #=
    One segment per trial: the LQR-governed tail. Trials whose estimated path
    never reaches the LQR state, or reaches it too late to leave two timesteps,
    are dropped — a segment shorter than that has no transition in it and the
    constructor refuses the schedule. How many were dropped is reported, since a
    procedure that quietly discards half the data is not the same procedure.
    =#
    segs = Matrix{Float64}[]
    segus = Matrix{Float64}[]
    starts = Int[]
    for i in 1:ntrials
        t0 = findfirst(==(LQR_STATE), zhat[i])
        (t0 === nothing || t0 > tsteps - 2) && continue
        # `z_t` selects the transition into `t`, so the segment's first state is t0-1.
        s0 = max(1, t0 - 1)
        push!(segs, ys[i][:, s0:tsteps])
        uxs === nothing || push!(segus, uxs[i][:, s0:tsteps])
        push!(starts, s0)
    end
    #=
    Zero segments is a result, not an error: it means the stage-one fit put no
    timestep in the LQR state on any trial, which is the collapse the joint
    table reports as a `γ` of 0.5. Say that rather than throwing, so a caller can
    print the reason instead of "FAILED".
    =#
    isempty(segs) && return (failed=:no_lqr_segments, source=source, kept=0, ntrials=ntrials)
    #=
    Equal-length segments are what make one schedule serve every trial; with a
    fixed onset and an estimated path that agrees with it, they usually are.
    Trim to the shortest so the constructor's terminal regime lands on the last
    entry of every trial.
    =#
    L = minimum(size(y, 2) for y in segs)
    segs = [y[:, (end - L + 1):end] for y in segs]
    segus = isempty(segus) ? nothing : [u[:, (end - L + 1):end] for u in segus]

    seg_truth = LqrTruth(
        segment_model(truth, L),
        truth.C,
        truth.R,
        LinearDynamicalSystem(
            segment_model(truth, L),
            GaussianObservationModel(
                copy(truth.C), copy(truth.R), zeros(size(truth.C, 1))
            ),
        ),
        truth.idx,
        nref,
        L,
    )
    sm = fit_model(
        seg_truth;
        known_plant=known_plant,
        free_gref=(nref > 0),
        sig0_costate=sig0_costate,
    )
    lds, elbos = one_fit(
        seg_truth, segs, segus, sm; free_C=false, max_iter=max_iter, tol=1e-10
    )
    ref = deepcopy(seg_truth.sm)
    rescale_costate!(ref; target=:trace)
    rescale_costate!(sm; target=:trace)
    gauge = gauge_compare(
        sm, ref, truth.idx, lds.obs_model.C, seg_truth.C;
        known_plant=known_plant, free_gref=(nref > 0),
    )
    return (
        scores=gauge.raw,
        gauge=gauge,
        source=source,
        kept=length(segs),
        ntrials=ntrials,
        seg_len=L,
        elbo=elbos[end],
        truth_elbo=elbo(seg_truth.lds, segs; ux=segus),
        iters=length(elbos),
        creep=elbo_creep(elbos),
        rho=maximum(abs, eigvals(symplectic_matrix(truth.lqr_sm))),
        tsteps=tsteps,
    )
end
