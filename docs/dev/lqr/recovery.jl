#=============================================================================
Recovery — simulate from a known model, refit from a deliberately wrong start,
and report how close every parameter block came.

The single-system driver is `recover`; the switching one is in `slds.jl`.

`known_plant` is the usual inverse-optimal-control posture: the body is known,
the objective is not. Set it `false` to estimate both — the plant is then also
started away from the truth, since a warm-started `A` would make its recovery
column report the initialization rather than the fit.

The cost is identified only up to a nonzero scalar (scaling a cost does not
change the policy it induces), so every comparison is made after
`rescale_costate!(...; target = :trace)`.
=============================================================================#

"""
    elbo_creep(elbos) -> Float64

Mean ELBO gain per iteration over the last tenth of the run, in nats.

`length(elbos) < max_iter` is the wrong question here. With `tol = 1e-10` these
fits essentially never stop on their tolerance — the bound keeps creeping
upward by tiny amounts for thousands of iterations — so a "converged" flag built
on it reads `NO` on every row and distinguishes nothing. What a reader actually
wants to know is whether the answer is still *moving*, and by how much relative
to the numbers in the table. Below ~1e-3 nats/iteration the parameters are flat
to three decimals; above ~1e-1 the row is reporting the iteration budget.
"""
function elbo_creep(elbos)
    length(elbos) < 2 && return 0.0
    k = max(1, length(elbos) ÷ 10)
    return (elbos[end] - elbos[end - k]) / k
end

"""
    fit_model(truth; known_plant, free_gref, init, jitter, rng) -> LQRStateModel

A model with the truth's *structure* and deliberately wrong *values*, ready to
be fitted back.

`init` picks how wrong:

- `:cold` — the default, and the one the sweeps use. An isotropic cost that
  knows nothing about the truth's shape, zero drift, an inflated *state* noise,
  and (when the reference is estimated) a reference map of the truth's own family
  at a third the radius and a quarter-turn of phase — wrong, but not
  *anti*-correct.

`sigma_prior_strength` and `qc_prior_strength` put inverse-Wishart priors on the
innovation and on the cost instead of relying on where they were started — see
[`sigma_prior`](@ref) and [`qc_prior`](@ref). `0` (the default) leaves the fit
unregularized, which is what every sweep outside `experiment_priors` uses.

`anneal_costate` runs the fit in two passes: the first from that (loose) costate
innovation, the second from `sig0_costate` (tight), started at whatever the first
pass reached. See the comment at the call site for why the two passes identify
different things.

`q0` is the isotropic scale of the initial cost — the one number that says how
big the fit thinks the objective is before it has seen anything. `sig0_state`
and `sig0_costate` set the two blocks of the initial `Σ`, and the
gap between them is the one piece of the cold start that is deliberately *not*
uninformative. Started isotropic, `Σ`'s costate block inflates during EM until
the model has `n` free directions in which the cost can drift at no cost to the
bound — the failure `experiment_initialization` measures. Starting it at `1e-4`
says the thing the model already implies: on the optimal path the costate is a
deterministic function of the state, so its innovation is near zero. It is a
starting point, not a constraint — the M-step is free to move it, and mostly
does not.
- `:warm` — start at the truth. Not a recovery test; it answers the different
  question of whether the generating parameters are even a *stationary point* of
  the objective, which on a misspecified row they are not.
- `:random` — a random PSD cost, for multi-start. `jitter` scales the draw.

Why a cold start matters: EM has no reason to move a parameter it starts at the
optimum of, so a warm-started block's recovery column would report the
initialization. Everything the sweeps score is started away from the truth.

When `Gref` has columns, `free_h` defaults to `false`: the reference inputs are
one-hot and their sum is an intercept, so fitting `h` would reintroduce the
common-origin gauge that the recovery exercise is meant to diagnose rather
than exploit. Pass `free_h=true` only for an explicit confounding control.
"""
function fit_model(
    truth::LqrTruth;
    known_plant::Bool,
    free_gref::Bool,
    free_h::Bool=(size(truth.sm.Gref, 2) == 0),
    init::Symbol=:cold,
    jitter::Float64=0.0,
    q0::Float64=0.4,
    sig0_state::Float64=0.05,
    sig0_costate::Float64=1e-4,
    sigma_prior_strength::Float64=0.0,
    qc_prior_strength::Float64=0.0,
    qc_prior_scale::Union{Float64,Vector{Float64}}=0.2,
    rng::AbstractRNG=MersenneTwister(0),
)
    sm = deepcopy(truth.sm)
    n = plant_dim(sm)
    d = 2n
    if init !== :warm
        for Q in sm.Qc
            if init === :random
                W = randn(rng, n, n)
                Q .= Symmetric(q0 .* Matrix(1.0I, n, n) .+ jitter .* (W * W') ./ n)
            else
                Q .= Matrix(q0 * I, n, n)
                jitter > 0 && (Q .+= jitter .* Symmetric(randn(rng, n, n)))
                Q .= Symmetric(Q)
            end
        end
        sm.Σ .= mixed_noise(n; state=sig0_state, costate=sig0_costate)
        sm.h .= 0
        if free_gref && size(sm.Gref, 2) > 0
            #=
            A structured wrong reference: one-third radius, quarter-turn phase.
            For the cosine-family truth with four targets this is exactly
            truth.Gref[:, [2,3,4,1]] / 3. A fit that barely moves therefore looks
            column-permuted. Target labels are observed; that permutation is
            an initialization error, not an allowed symmetry. The reference
            audit compares initial/final maps and probes the marginal likelihood.
            =#
            m = size(sm.Gref, 2)
            for j in 1:m, i in 1:n
                sm.Gref[i, j] = 0.5 * cos(2π * (j - 1) / m + π * (i - 1) / n + π / 2)
            end
        end
        #=
        When the plant is estimated, start it wrong too. Left at the truth it
        would simply stay there, and its recovery column would report the
        initialization rather than the fit.
        =#
        if !known_plant
            sm.A .= 0.90 .* truth.sm.A .+ Matrix(0.03I, n, n)
            sm.S .= 1.5 .* truth.sm.S
        end
    end
    #=
    Priors go on the *fit*, never on the truth. They are a statement about how
    the model should be estimated, and putting one on the generating model would
    quietly change what is being recovered.
    =#
    sm.Σ_prior = sigma_prior(
        n; state=sig0_state, costate=sig0_costate, strength=sigma_prior_strength
    )
    sm.Qc_prior = qc_prior(n; scale=qc_prior_scale, strength=qc_prior_strength)
    sm.fit_flags = LQRFitFlags(;
        A=(!known_plant),
        S=(!known_plant),
        Gref=free_gref,
        h=free_h,
        #=
        `Bu` is frozen at zero throughout. In a tracking model its costate rows
        and `−Q_k G_r` both map the input into the costate, and the type's own
        documentation says to freeze one; freezing the reduced-form one is what
        makes `Gref` the thing being estimated rather than a nuisance it hides
        behind.
        =#
        Bu=false,
    )
    refresh!(sm)
    return sm
end

"""
    one_fit(truth, ys, uxs, sm; free_C, max_iter, tol) -> (lds, elbos)

One EM run of a prepared initial model against a fixed dataset.

Freeze the emission when `C = [I 0]`: that is the point of fixing it, and a
fitted `C` would drift the latent basis out from under the comparison.
"""
function one_fit(
    truth::LqrTruth,
    ys,
    uxs,
    sm;
    free_C::Bool,
    max_iter::Int,
    tol::Float64,
    fit_noise::Bool=true,
)
    obs_dim = size(truth.C, 1)
    lds = LinearDynamicalSystem(
        sm, GaussianObservationModel(copy(truth.C), copy(truth.R), zeros(obs_dim))
    )
    free_C || (lds.fit_bool[5] = false)
    lds.fit_bool[4] = fit_noise
    elbos = fit!(lds, ys; ux=uxs, max_iter=max_iter, tol=tol, progress=false)
    return lds, (elbos isa Tuple ? first(elbos) : elbos)
end

"""
    recover(; kwargs...) -> NamedTuple

Simulate, refit, score. Returns the per-block `(rmse, corr)` scores in the
canonical scale, the ELBO the fit reached, the ELBO at the generating
parameters, whether EM stayed monotone and whether it converged at all, and
both canonicalized models so the plotting code can draw one against the other.

# Generative knobs
`gen` selects the mode: `:rand` draws from the model's own forward chain, `:lqr`
rolls out the optimal trajectory with `simulate_lqr` and adds `costate_slack` of
suboptimality. `process_noise` applies to `:lqr` only. `terminal`, `onset`,
`nref`, `drift`, `observe_costate`, `free_C`, `obs_noise`, `n`, `tsteps` and
`ntrials` shape the truth; see `lqr_truth`.

# Fitting knobs
`known_plant`, `free_gref` say which blocks EM may move. `free_h` defaults to
false when `nref > 0`, fixing the one-hot reference/intercept degeneracy.
`max_iter`, `tol`,
`mstep_iters`, `init` and `restarts` are the *procedure* — the axis the
"what helps" sweep varies while holding the generative model fixed. With
`restarts > 1` the fit is run that many times from independently jittered
starts and the one with the best final ELBO is kept, which is the standard
defence against a multimodal likelihood.
"""
function recover(;
    n::Int=4,
    tsteps::Int=20,
    ntrials::Int=120,
    gen::Symbol=:rand,
    costate_slack::Float64=0.1,
    process_noise::Bool=true,
    terminal::Bool=false,
    onset::Int=1,
    nref::Int=0,
    drift::Bool=false,
    free_C::Bool=false,
    observe_costate::Bool=false,
    known_plant::Bool=true,
    free_gref::Bool=(nref > 0),
    free_h::Bool=(nref == 0),
    obs_noise::Float64=0.05,
    max_iter::Int=250,
    tol::Float64=1e-10,
    mstep_iters::Int=100,
    init::Symbol=:cold,
    restarts::Int=1,
    q0::Float64=0.4,
    ring::Bool=false,
    state_noise::Float64=0.02,
    costate_noise::Float64=1e-4,
    sig0_state::Float64=0.05,
    sig0_costate::Float64=1e-4,
    sigma_prior_strength::Float64=0.0,
    qc_prior_strength::Float64=0.0,
    qc_prior_scale::Union{Float64,Vector{Float64}}=0.2,
    anneal_costate::Union{Nothing,Float64}=nothing,
    fit_noise::Bool=true,
    #=
    Hold out a fraction of trials, fit on the rest, and report the held-out ELBO
    per timestep alongside the parameter scores. This is the only quantity in the
    harness a user could compute without knowing the truth, so it is the only
    candidate for *selecting* a setting rather than merely scoring one — and
    whether it selects the same setting the parameter scores prefer is itself a
    result. Zero (the default) fits everything and reports `NaN`.
    =#
    test_frac::Float64=0.0,
    seed::Int=1,
)
    rng = MersenneTwister(seed)
    d = 2n
    truth = lqr_truth(;
        n=n,
        tsteps=tsteps,
        terminal=terminal,
        onset=onset,
        nref=nref,
        drift=drift,
        observe_costate=observe_costate,
        free_C=free_C,
        obs_noise=obs_noise,
        state_noise=state_noise,
        costate_noise=costate_noise,
        ring=ring,
        rng=rng,
    )
    uxs = target_inputs(rng, nref, ntrials, tsteps)
    ys = simulate(
        rng,
        truth,
        ntrials;
        gen=gen,
        slack=costate_slack,
        process_noise=process_noise,
        uxs=uxs,
    )
    ntest = test_frac > 0 ? max(1, round(Int, test_frac * ntrials)) : 0
    tr_idx = 1:(ntrials - ntest)
    te_idx = (ntrials - ntest + 1):ntrials
    ys_fit = ntest > 0 ? ys[tr_idx] : ys
    ux_fit = (uxs === nothing || ntest == 0) ? uxs : uxs[tr_idx]
    truth_elbo = elbo(truth.lds, ys_fit; ux=ux_fit)

    best = nothing
    for r in 1:restarts
        sm = fit_model(
            truth;
            known_plant=known_plant,
            free_gref=free_gref,
            free_h=free_h,
            #=
            The first start is the plain cold one on every row, so a
            `restarts = 1` row and the first restart of a `restarts = 4` row are
            the same fit. Only the extra starts are jittered, which makes the
            multi-start column a strict addition rather than a different
            experiment.
            =#
            init=(r == 1 ? init : (init === :warm ? :warm : :random)),
            jitter=(r == 1 ? 0.0 : 0.3),
            q0=q0,
            sig0_state=sig0_state,
            sig0_costate=(anneal_costate === nothing ? sig0_costate : anneal_costate),
            sigma_prior_strength=sigma_prior_strength,
            qc_prior_strength=qc_prior_strength,
            qc_prior_scale=qc_prior_scale,
            rng=MersenneTwister(1000seed + r),
        )
        sm.mstep_iters = mstep_iters
        initial_gref = copy(sm.Gref)
        lds, elbos = one_fit(
            truth,
            ys_fit,
            ux_fit,
            sm;
            free_C=free_C,
            max_iter=max_iter,
            tol=tol,
            fit_noise=fit_noise,
        )
        if best === nothing || elbos[end] > best.elbos[end]
            best = (sm=sm, lds=lds, elbos=elbos, initial_gref=initial_gref)
        end
    end
    elbos = best.elbos
    fit_sm = best.sm
    fit_lds = best.lds
    #=
    Annealing the costate innovation: fit once from a loose `Σ_λλ`, then reset
    that block to a tight value and fit again from wherever the first pass
    landed.

    The two starts are good at different things and the ladder above shows it.
    A tight costate innovation can make EM's Gref updates very slow: the E-step
    reconstructs costates under the current reference, and the complete-data
    objective strongly penalizes changing that reference while holding those
    costates fixed. This does not make the marginal likelihood flat: integrating
    out costates still leaves information through the observed state dynamics.
    A loose start can accelerate reference learning, but also changes the joint
    optimization path for costs/noise. Doing these passes in order asks whether
    the tight pass can retain what the loose pass found.
    =#
    if anneal_costate !== nothing
        sm2 = fit_sm
        #=
        The cross-blocks go too, not just the costate block. A fitted `Σ` has
        state-costate covariance in it, and shrinking the costate variance while
        leaving that covariance at its old size makes the matrix indefinite —
        `refresh!` then fails its Cholesky rather than starting a second pass.
        Rebuilding block-diagonally keeps the fitted process noise and is
        positive definite whenever that block is.
        =#
        @views sm2.Σ[1:n, (n + 1):d] .= 0
        @views sm2.Σ[(n + 1):d, 1:n] .= 0
        @views sm2.Σ[(n + 1):d, (n + 1):d] .= Matrix(sig0_costate * I, n, n)
        refresh!(sm2)
        fit_lds, elbos2 = one_fit(
            truth,
            ys_fit,
            ux_fit,
            sm2;
            free_C=free_C,
            max_iter=max_iter,
            tol=tol,
            fit_noise=fit_noise,
        )
        fit_sm = sm2
        elbos = vcat(elbos, elbos2)
    end

    #=
    The held-out score is taken *before* canonicalization, on the model as
    fitted: `rescale_costate!` changes the parameters and the emission's costate
    columns are not rescaled with them, so a likelihood evaluated after it would
    not be the fitted model's.
    =#
    heldout = if ntest > 0
        fl = LinearDynamicalSystem(
            fit_sm,
            GaussianObservationModel(
                copy(truth.C), copy(truth.R), zeros(size(truth.C, 1))
            ),
        )
        steps = sum(size(y, 2) for y in ys[te_idx])
        elbo(fl, ys[te_idx]; ux=(uxs === nothing ? nothing : uxs[te_idx])) / steps
    else
        NaN
    end

    # Compare in the canonical scale: the cost is identified up to a scalar.
    ref = deepcopy(truth.sm)
    rescale_costate!(ref; target=:trace)
    rescale_costate!(fit_sm; target=:trace)
    gauge = gauge_compare(
        fit_sm, ref, truth.idx, fit_lds.obs_model.C, truth.C;
        known_plant=known_plant, free_gref=free_gref, free_noise=fit_noise,
    )
    reference_design = reference_design_audit(uxs)
    audit_ref = deepcopy(ref)
    audit_ref.fit_flags = fit_sm.fit_flags
    reference_translation = reference_translation_audit(audit_ref, reference_design)
    return (
        scores=gauge.raw,
        gauge=gauge,
        reference_design=reference_design,
        reference_translation=reference_translation,
        reference_initial=(free_gref && size(ref.Gref, 2) > 0) ?
            score(best.initial_gref, ref.Gref) : NOSCORE,
        reference_movement=(free_gref && size(ref.Gref, 2) > 0) ?
            reference_movement(fit_sm.Gref, best.initial_gref, ref.Gref) :
            (raw=NaN, contrasts=NaN),
        elbo=elbos[end],
        truth_elbo=truth_elbo,
        heldout=heldout,
        iters=length(elbos),
        converged=length(elbos) < max_iter,
        creep=elbo_creep(elbos),
        monotone=length(elbos) < 2 || minimum(diff(elbos)) > -1e-8,
        rho=maximum(abs, eigvals(symplectic_matrix(truth.sm))),
        defect=symplectic_defect(fit_sm),
        fit_sm=fit_sm,
        fit_C=copy(fit_lds.obs_model.C),
        ref_sm=ref,
        idx=truth.idx,
        elbos=elbos,
        ntrials=ntrials,
        tsteps=tsteps,
    )
end

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

"""
    selftest() -> Bool

Check the scoring plumbing, not the models.

Everything this harness reports passes through two transformations that are easy
to get subtly wrong and impossible to notice from the tables: the block-by-block
comparison, and the canonical rescaling it is performed in. Two properties pin
both down.

1. **Identity.** A model compared against itself must score exactly `0 / 1` on
   every block, with the truth built so that no block is degenerate — a terminal
   regime, a delay regime, a reference map and a nonzero drift all present, and
   the plant estimated so its column is live.
2. **Scale invariance.** Rescaling a model's costate by an arbitrary factor and
   re-canonicalizing must return it to the same place. This is the property the
   whole comparison rests on: the cost is identified only up to a scalar, so a
   metric that moved under `rescale_costate!` would be measuring the scale rather
   than the fit.

Run it with `--selftest`. It takes a second and it is the difference between a
table of numbers and a table of numbers you can believe.
"""
function selftest(; verbose::Bool=true)
    truth = lqr_truth(; n=3, tsteps=20, terminal=true, onset=7, nref=4, drift=true)
    warm = fit_model(truth; known_plant=false, free_gref=true, init=:warm)
    ref = deepcopy(truth.sm)
    rescale_costate!(ref; target=:trace)
    rescale_costate!(warm; target=:trace)
    id = compare(warm, ref, truth.idx; known_plant=false, free_gref=true)
    worst_id = maximum(
        k -> (v = getfield(id, k); isnan(v.rmse) ? 0.0 : abs(v.rmse)), keys(id)
    )

    shifted = deepcopy(truth.sm)
    rescale_costate!(shifted, 7.3)
    rescale_costate!(shifted; target=:trace)
    scale_inv = compare(shifted, ref, truth.idx; known_plant=false, free_gref=true)
    worst_inv = maximum(
        k -> (v = getfield(scale_inv, k); isnan(v.rmse) ? 0.0 : abs(v.rmse)),
        keys(scale_inv),
    )

    # A complete state-basis rotation must look wrong raw and exact after the
    # same map is applied to every structural parameter. This catches the
    # tempting but invalid shortcut of rotating `Gref` alone.
    W = Matrix(qr(randn(MersenneTwister(991), 3, 3)).Q)
    rotated = deepcopy(ref)
    rotated.A .= W' * ref.A * W
    rotated.S .= W' * ref.S * W
    rotated.Gref .= W' * ref.Gref
    for k in eachindex(rotated.Qc)
        rotated.Qc[k] .= W' * ref.Qc[k] * W
    end
    D = zeros(6, 6)
    D[1:3, 1:3] .= W'
    D[4:6, 4:6] .= W'
    rotated.h .= D * ref.h
    rotated.Σ .= D * ref.Σ * D'
    refresh!(rotated)
    Crot = truth.C * D'
    gauge = gauge_compare(
        rotated, ref, truth.idx, Crot, truth.C;
        known_plant=false, free_gref=true,
    )
    worst_gauge = maximum(
        b -> begin
            v = getfield(gauge.procrustes, b)
            isnan(v.rmse) ? 0.0 : v.rmse
        end,
        keys(gauge.procrustes),
    )

    # The full-linear branch is not a differently spelled Procrustes call. A
    # scale/shear is an exact LQR gauge too, with the costate transforming by
    # the inverse transpose. It should fail the orthogonal check and pass the
    # general one after trace canonicalization is reapplied.
    Tlin = W * Diagonal([0.55, 1.25, 1.8])
    invT = inv(Tlin)
    sheared = deepcopy(ref)
    sheared.A .= invT * ref.A * Tlin
    sheared.S .= invT * ref.S * invT'
    sheared.Gref .= invT * ref.Gref
    for k in eachindex(sheared.Qc)
        sheared.Qc[k] .= Tlin' * ref.Qc[k] * Tlin
    end
    E = zeros(6, 6)
    E[1:3, 1:3] .= invT
    E[4:6, 4:6] .= Tlin'
    sheared.h .= E * ref.h
    sheared.Σ .= E * ref.Σ * E'
    refresh!(sheared)
    rescale_costate!(sheared; target=:trace)
    Dlin = zeros(6, 6)
    Dlin[1:3, 1:3] .= Tlin
    Dlin[4:6, 4:6] .= invT'
    linear_gauge = gauge_compare(
        sheared, ref, truth.idx, truth.C*Dlin, truth.C;
        known_plant=false, free_gref=true,
    )
    worst_linear = maximum(
        b -> begin
            v = getfield(linear_gauge.linear, b)
            isnan(v.rmse) ? 0.0 : v.rmse
        end,
        keys(linear_gauge.linear),
    )

    # A fixed emission removes the state-coordinate gauge, but one-hot target
    # codes leave a separate reference-origin gauge.  With one active running
    # cost, shifting every target by `delta` and the costate intercept by
    # `Q*delta` leaves the transition exactly unchanged.  Centred references
    # and pairwise distances must recognize that equivalence, and freezing `h`
    # must remove all n translation directions.
    origin_truth = lqr_truth(; n=3, tsteps=12, nref=4)
    origin_fit = fit_model(
        origin_truth; known_plant=true, free_gref=true, free_h=true, init=:warm
    )
    origin_ux = target_inputs(MersenneTwister(992), 4, 8, 12)
    origin_design = reference_design_audit(origin_ux)
    origin_free = reference_translation_audit(origin_fit, origin_design)
    origin_default_fit = fit_model(
        origin_truth; known_plant=true, free_gref=true, init=:warm
    )
    origin_default = reference_translation_audit(origin_default_fit, origin_design)
    delta = [0.4, -0.2, 0.3]
    translated = deepcopy(origin_truth.sm)
    translated.Gref .+= delta
    translated.h[4:6] .+= translated.Qc[1] * delta
    refresh!(translated)
    origin_geometry = reference_geometry(translated.Gref, origin_truth.sm.Gref)
    fixed_h = deepcopy(origin_fit)
    fixed_h.fit_flags = LQRFitFlags(; A=false, S=false, Qc=true, h=false,
        Bu=false, Gref=true, terminal=false)
    origin_fixed = reference_translation_audit(fixed_h, origin_design)
    terminal_truth = lqr_truth(; n=3, tsteps=12, nref=4, terminal=true)
    terminal_fit = fit_model(
        terminal_truth; known_plant=true, free_gref=true, free_h=true, init=:warm
    )
    origin_terminal = reference_translation_audit(terminal_fit, origin_design)
    multiregime_truth = lqr_truth(; n=3, tsteps=12, nref=4, onset=5)
    multiregime_fit = fit_model(
        multiregime_truth; known_plant=true, free_gref=true, free_h=true, init=:warm
    )
    origin_multiregime = reference_translation_audit(multiregime_fit, origin_design)
    origin_ok = origin_design.affine_nullity == 1 && origin_free.nullity == 3 &&
                !origin_default_fit.fit_flags.h && origin_default.nullity == 0 &&
                origin_fixed.nullity == 0 && origin_terminal.nullity == 3 &&
                origin_multiregime.nullity == 0 && origin_geometry.raw.rmse > 0.1 &&
                origin_geometry.contrasts.rmse <= 1e-12 &&
                origin_geometry.distances.rmse <= 1e-12 &&
                all(j -> translated.cache.bfwd + translated.cache.Bfwd[1][:, j] ≈
                         origin_truth.sm.cache.bfwd + origin_truth.sm.cache.Bfwd[1][:, j],
                    axes(translated.Gref, 2))

    ok = worst_id <= 1e-10 && worst_inv <= 1e-8 &&
         gauge.raw.Gref.rmse > 0.1 && worst_gauge <= 1e-8 &&
         linear_gauge.procrustes.Gref.rmse > 0.1 && worst_linear <= 1e-8 && origin_ok
    if verbose
        @printf("selftest  identity: worst block rmse %.3g  (want 0)\n", worst_id)
        @printf("selftest  rescale invariance: worst block rmse %.3g  (want ~1e-16)\n",
                worst_inv)
        @printf("selftest  gauge alignment: worst block rmse %.3g  (want ~1e-16)\n",
                worst_gauge)
        @printf("selftest  linear gauge: worst block rmse %.3g  (want ~1e-16)\n",
                worst_linear)
        @printf("selftest  reference origin: design %d, default/free/fixed/terminal/multiregime %d/%d/%d/%d/%d  (want 1, 0/3/0/3/0)\n",
                origin_design.affine_nullity, origin_default.nullity,
                origin_free.nullity, origin_fixed.nullity, origin_terminal.nullity,
                origin_multiregime.nullity)
        println("selftest  ", ok ? "PASS" : "FAIL")
    end
    return ok
end
