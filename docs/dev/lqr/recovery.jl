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
"""
function fit_model(
    truth::LqrTruth;
    known_plant::Bool,
    free_gref::Bool,
    free_h::Bool=true,
    init::Symbol=:cold,
    jitter::Float64=0.0,
    q0::Float64=0.4,
    sig0_state::Float64=0.05,
    sig0_costate::Float64=1e-4,
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
            An *uninformative* wrong reference, not an adversarial one: the
            truth's own construction at a third the radius and a quarter-turn of
            phase. Starting at `-c · Gref_true` would seed the fit on the far
            side of the sign flip the costate scale already admits, and a
            correlation of `-1` in the table would then be reporting the
            initialization rather than a failure of identification.
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
`known_plant`, `free_gref` say which blocks EM may move. `max_iter`, `tol`,
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
    free_h::Bool=true,
    obs_noise::Float64=0.05,
    max_iter::Int=250,
    tol::Float64=1e-10,
    mstep_iters::Int=100,
    init::Symbol=:cold,
    restarts::Int=1,
    q0::Float64=0.4,
    state_noise::Float64=0.02,
    costate_noise::Float64=1e-4,
    sig0_state::Float64=0.05,
    sig0_costate::Float64=1e-4,
    anneal_costate::Union{Nothing,Float64}=nothing,
    fit_noise::Bool=true,
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
    truth_elbo = elbo(truth.lds, ys; ux=uxs)

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
            rng=MersenneTwister(1000seed + r),
        )
        sm.mstep_iters = mstep_iters
        lds, elbos = one_fit(
            truth,
            ys,
            uxs,
            sm;
            free_C=free_C,
            max_iter=max_iter,
            tol=tol,
            fit_noise=fit_noise,
        )
        if best === nothing || elbos[end] > best.elbos[end]
            best = (sm=sm, lds=lds, elbos=elbos)
        end
    end
    elbos = best.elbos
    fit_sm = best.sm
    #=
    Annealing the costate innovation: fit once from a loose `Σ_λλ`, then reset
    that block to a tight value and fit again from wherever the first pass
    landed.

    The two starts are good at different things and the ladder above shows it.
    A loose costate innovation keeps `λ` a quantity the model has to *explain*,
    which is what identifies the reference — the reference enters only through
    the costate half of the affine term, so if `Σ_λλ → 0` the smoother can
    satisfy the costate recursion exactly for any `G_r` and there is nothing left
    to pin it. A tight one is what identifies the cost's shape, by removing the
    `n` free directions of slack the cost would otherwise drift along. Doing them
    in that order asks whether the second pass can keep what the first found.
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
        _, elbos2 = one_fit(
            truth,
            ys,
            uxs,
            sm2;
            free_C=free_C,
            max_iter=max_iter,
            tol=tol,
            fit_noise=fit_noise,
        )
        fit_sm = sm2
        elbos = vcat(elbos, elbos2)
    end

    # Compare in the canonical scale: the cost is identified up to a scalar.
    ref = deepcopy(truth.sm)
    rescale_costate!(ref; target=:trace)
    rescale_costate!(fit_sm; target=:trace)
    return (
        scores=compare(
            fit_sm, ref, truth.idx; known_plant=known_plant, free_gref=free_gref
        ),
        elbo=elbos[end],
        truth_elbo=truth_elbo,
        iters=length(elbos),
        converged=length(elbos) < max_iter,
        creep=elbo_creep(elbos),
        monotone=length(elbos) < 2 || minimum(diff(elbos)) > -1e-8,
        rho=maximum(abs, eigvals(symplectic_matrix(truth.sm))),
        defect=symplectic_defect(fit_sm),
        fit_sm=fit_sm,
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
    inv = compare(shifted, ref, truth.idx; known_plant=false, free_gref=true)
    worst_inv = maximum(
        k -> (v = getfield(inv, k); isnan(v.rmse) ? 0.0 : abs(v.rmse)), keys(inv)
    )

    ok = worst_id <= 1e-10 && worst_inv <= 1e-8
    if verbose
        @printf("selftest  identity: worst block rmse %.3g  (want 0)\n", worst_id)
        @printf("selftest  rescale invariance: worst block rmse %.3g  (want ~1e-16)\n",
                worst_inv)
        println("selftest  ", ok ? "PASS" : "FAIL")
    end
    return ok
end
