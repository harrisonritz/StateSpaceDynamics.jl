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
  knows nothing about the truth's shape, zero drift, an inflated `Σ`, and (when
  the reference is estimated) a reference map of the truth's own family at a
  third the radius and a quarter-turn of phase — wrong, but not *anti*-correct.
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
    rng::AbstractRNG=MersenneTwister(0),
)
    sm = deepcopy(truth.sm)
    n = plant_dim(sm)
    d = 2n
    if init !== :warm
        for Q in sm.Qc
            if init === :random
                W = randn(rng, n, n)
                Q .= Symmetric(0.25 .* Matrix(1.0I, n, n) .+ jitter .* (W * W') ./ n)
            else
                Q .= Matrix(0.4I, n, n)
                jitter > 0 && (Q .+= jitter .* Symmetric(randn(rng, n, n)))
                Q .= Symmetric(Q)
            end
        end
        sm.Σ .= Matrix(0.05I, d, d)
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
function one_fit(truth::LqrTruth, ys, uxs, sm; free_C::Bool, max_iter::Int, tol::Float64)
    obs_dim = size(truth.C, 1)
    lds = LinearDynamicalSystem(
        sm, GaussianObservationModel(copy(truth.C), copy(truth.R), zeros(obs_dim))
    )
    free_C || (lds.fit_bool[5] = false)
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
    seed::Int=1,
)
    rng = MersenneTwister(seed)
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
            rng=MersenneTwister(1000seed + r),
        )
        sm.mstep_iters = mstep_iters
        lds, elbos = one_fit(
            truth, ys, uxs, sm; free_C=free_C, max_iter=max_iter, tol=tol
        )
        if best === nothing || elbos[end] > best.elbos[end]
            best = (sm=sm, lds=lds, elbos=elbos)
        end
    end
    elbos = best.elbos
    fit_sm = best.sm

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
