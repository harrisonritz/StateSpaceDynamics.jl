#=============================================================================
Model recovery — can you tell an LQR apart from a plain LDS?

Every other experiment in this directory asks how well the *parameters* of a
known model come back. This one asks the prior question: given data, can you
tell which model generated it? That is the identifiability question a reader
actually has, because in an application nobody hands you the generating class.

## The two models

Both carry the same `2n`-dimensional latent and the same emission, so the only
thing that differs is the transition:

  * **LQR** — the symplectic, time-varying transition an optimal controller
    implies, with a cost schedule, an optional terminal factor, and the
    reference entering through `−Q_k G_r u`.
  * **LDS** — a `free_state_model`: one constant, unconstrained `M`, with the
    input entering through a free `B_u`.

The LDS is the *harder* competitor it looks: with a free `B_u` it has more free
parameters than the LQR (62 against 46 at `n = 2` with eight targets), so a
win for the LQR on held-out data is not a win on parsimony alone.

## The three generators

  * **`:rand`** — the LQR model's own forward chain. Here the LQR is *correctly
    specified* and the LDS is not, so this is the condition under which the
    comparison has to work if it is going to work anywhere.
  * **`:lqr`** — the optimal trajectory, as everywhere else in this harness.
    Note what this is: an optimal trajectory's mixed-coordinate residual is
    (up to slack and process noise) exactly zero, which is not a draw from the
    model's forward chain. The LQR model is misspecified for its own intended
    data, and the comparison inherits that.
  * **`:lds`** — a first-order attractor: `x_{t+1} = M x_t + B_u u_t` with `B_u`
    chosen so that the fixed point for target `j` is that target's ring
    location. This is a real alternative account of goal-directed reaching —
    exponential approach to the goal — and not a strawman: it produces
    trajectories that start at the origin and settle on the target, which is
    what the LQR does too. What it does not have is the LQR's time-varying
    structure or its endpoint condition, and that is the whole of what the
    comparison can detect.

## The criterion

Held-out ELBO per timestep. The two models have different parameter counts, so
an in-sample comparison would be meaningless; a held-out one handles the
difference without an information criterion's asymptotics, which these fits are
in no position to satisfy.

**The terminal factor defaults off here, and must.** With it on, `elbo` reports
`log p(y, y^term = 0)` for the LQR and `log p(y)` for the LDS — a joint density
over one more variable against a marginal over the observed data alone. They are
not the same quantity and comparing them is an arithmetic error, not a
conservative choice. `terminal = true` is available for a reader who wants to see
the size of the effect (~2 nats per trial from the factor's own normalizer), but
the number it produces is not a model comparison.
=============================================================================#

"""
    lds_competitor_truth(; n, tsteps, nref, decay, radius, noise, obs_noise) -> LqrTruth

The `:lds` generator's ground truth: a `free_state_model` whose state block
contracts toward the presented target.

`Bu[1:n, :] = (I − M_x) G_r` is what puts the fixed point on the ring: with
`x_{t+1} = M_x x_t + (I − M_x) r_j`, the map has `r_j` as its unique fixed point
and approaches it geometrically at rate `decay`. The costate block is a plain
contraction with no input — nothing observes it, and letting it wander would
hand the fit a second unidentified way to explain the data.

Returned as an [`LqrTruth`](@ref) so the same emission, inputs and simulation
paths serve both generators.
"""
function lds_competitor_truth(;
    n::Int=2,
    tsteps::Int=30,
    nref::Int=8,
    decay::Float64=0.85,
    radius::Float64=1.5,
    noise::Float64=0.02,
    obs_noise::Float64=0.05,
    observe_costate::Bool=false,
)
    d = 2n
    M = zeros(d, d)
    M[1:n, 1:n] .= Matrix(decay * I, n, n)
    M[(n + 1):d, (n + 1):d] .= Matrix(0.6I, n, n)
    Bu = zeros(d, max(nref, 0))
    if nref > 0
        G = ring_map(n, nref; radius=radius)
        Bu[1:n, :] .= (Matrix(1.0I, n, n) - M[1:n, 1:n]) * G
    end
    sm = free_state_model(
        M,
        mixed_noise(n; state=noise, costate=noise);
        Bu=nref > 0 ? Bu : nothing,
        P0=Matrix(0.2I, d, d),
        observe_costate=observe_costate,
    )
    obs_dim = obs_width(n; free_C=false, observe_costate=observe_costate)
    C = emission(
        n, obs_dim; free_C=false, observe_costate=observe_costate, rng=MersenneTwister(0)
    )
    R = Matrix(obs_noise * I, obs_dim, obs_dim)
    lds = LinearDynamicalSystem(
        sm, GaussianObservationModel(copy(C), copy(R), zeros(obs_dim))
    )
    return LqrTruth(sm, C, R, lds, (run=1, delay=nothing, term=nothing), nref, tsteps)
end

"""
    free_candidate(truth; noise0) -> LQRStateModel

The LDS hypothesis, as something to fit: a `:free` model of the same latent
dimension as the LQR one, started away from anything in particular.

`M` starts as a uniform contraction and `B_u` at zero, so the candidate is told
nothing about either the dynamics or the targets. `B_u` is free — a
*reduced-form* input coupling, which is the LDS's way of representing a
reference.

`h` is **frozen at zero**, and that is a design fix rather than a handicap. The
inputs here are a one-hot target indicator, so their columns sum to one and are
exactly collinear with a free bias: the regression's Gram is rank-deficient by
construction, the package ridges it to stay alive and says so, and the
"parameter" the bias adds is a direction the data cannot see. Freezing it leaves
the same model with an identified parameterization — `B_u`'s columns already span
the constant.
"""
function free_candidate(truth::LqrTruth; noise0::Float64=0.05, decay0::Float64=0.8)
    n = truth.nref >= 0 ? size(truth.C, 2) ÷ 2 : 0
    d = 2n
    M = Matrix(decay0 * I, d, d)
    return free_state_model(
        M,
        mixed_noise(n; state=noise0, costate=noise0);
        Bu=truth.nref > 0 ? zeros(d, truth.nref) : nothing,
        P0=Matrix(0.2I, d, d),
        observe_costate=truth.sm.observe_costate,
        fit_flags=LQRFitFlags(; h=(truth.nref == 0)),
    )
end

"""
    nparams(sm, nref) -> Int

Free parameters of a candidate's *transition*, for the record. The emission is
shared and frozen, and `x0` / `P0` are common to both, so only the state model's
own blocks differ. Symmetric blocks are counted once.
"""
function nparams(sm, nref::Int)
    d = _state_dim(sm)
    n = d ÷ 2
    tri(k) = k * (k + 1) ÷ 2
    # `h` is frozen on the free candidate whenever there is a one-hot input to
    # be collinear with; see `free_candidate`.
    sm.mode === :free && return d^2 + (nref == 0 ? d : 0) + d * nref
    return n^2 + tri(n) + length(sm.Qc) * tri(n) + d + n * nref    # A, S, Qc, h, Gref
end

_state_dim(sm) = size(sm.Σ, 1)

"""
    holdout_score(lds, ys_train, uxs_train, ys_test, uxs_test; max_iter, tol)
        -> (elbo_per_step, trace)

Fit on the training trials and score the held-out ones, in nats per timestep so
that datasets of different size and trial length are comparable.

The held-out ELBO is evaluated at the *fitted* parameters rather than at the
iteration where it peaked. Reporting the peak would be selecting on the test set,
which is the thing a held-out score exists to avoid.
"""
function holdout_score(
    lds, ys_train, uxs_train, ys_test, uxs_test; max_iter::Int=250, tol::Float64=1e-10
)
    tr = fit!(
        lds, ys_train; ux=uxs_train, max_iter=max_iter, tol=tol, progress=false
    )
    train = tr isa Tuple ? first(tr) : tr
    steps = sum(size(y, 2) for y in ys_test)
    return (elbo(lds, ys_test; ux=uxs_test) / steps, train)
end

"""
    model_recovery(; kwargs...) -> NamedTuple

Generate from one model class, fit both, and report which one a held-out score
prefers.

`gen` is `:rand` (the LQR model's own chain), `:lqr` (its optimal trajectory) or
`:lds` (the attractor competitor). Returns the per-timestep held-out ELBO of each
candidate, their difference (positive favours the LQR), the free-parameter
counts, and the winner — so a caller can build the confusion matrix that
answers "is this model class identifiable from data at all".
"""
function model_recovery(;
    n::Int=2,
    tsteps::Int=30,
    ntrials::Int=400,
    test_frac::Float64=0.3,
    gen::Symbol=:lqr,
    nref::Int=8,
    terminal::Bool=false,
    onset::Int=0,
    costate_slack::Float64=0.1,
    obs_noise::Float64=0.05,
    known_plant::Bool=true,
    sig0_costate::Float64=1e-4,
    lds_decay::Float64=0.85,
    max_iter::Int=250,
    seed::Int=1,
)
    gen in (:lqr, :lds, :rand) ||
        throw(ArgumentError("gen must be :lqr, :lds or :rand; got :$gen"))
    rng = MersenneTwister(seed)
    on = onset == 0 ? max(2, tsteps ÷ 3) : onset
    uxs = target_inputs(rng, nref, ntrials, tsteps)

    truth = if gen !== :lds
        lqr_truth(;
            n=n,
            tsteps=tsteps,
            terminal=terminal,
            onset=on,
            nref=nref,
            obs_noise=obs_noise,
            ring=true,
            rng=rng,
        )
    else
        lds_competitor_truth(;
            n=n, tsteps=tsteps, nref=nref, decay=lds_decay, obs_noise=obs_noise
        )
    end
    ys = if gen === :lqr
        simulate(rng, truth, ntrials; gen=:lqr, slack=costate_slack, uxs=uxs)
    else
        # `:rand` and `:lds` both roll their own model's forward chain.
        last(rand(rng, truth.lds, fill(tsteps, ntrials); ux=uxs))
    end

    ntest = max(1, round(Int, test_frac * ntrials))
    tr_idx, te_idx = 1:(ntrials - ntest), (ntrials - ntest + 1):ntrials
    ys_tr, ys_te = ys[tr_idx], ys[te_idx]
    ux_tr = uxs === nothing ? nothing : uxs[tr_idx]
    ux_te = uxs === nothing ? nothing : uxs[te_idx]

    #=
    The LQR candidate always carries the *structure* the LQR generator used —
    the same number of cost regimes and the same terminal factor — even when the
    data came from the LDS. Handing it a structure it cannot use is the point:
    on LDS data it is the over-specified model, and the held-out score has to
    decide whether the structure earns its keep.
    =#
    lqr_struct = lqr_truth(;
        n=n,
        tsteps=tsteps,
        terminal=terminal,
        onset=on,
        nref=nref,
        obs_noise=obs_noise,
        ring=true,
        rng=MersenneTwister(seed),
    )
    lqr_sm = fit_model(
        lqr_struct;
        known_plant=known_plant,
        free_gref=(nref > 0),
        sig0_costate=sig0_costate,
    )
    free_sm = free_candidate(lqr_struct)

    obs_dim = size(truth.C, 1)
    function wrap(sm)
        l = LinearDynamicalSystem(
            sm, GaussianObservationModel(copy(truth.C), copy(truth.R), zeros(obs_dim))
        )
        l.fit_bool[5] = false
        return l
    end
    lqr_score, lqr_trace = holdout_score(
        wrap(lqr_sm), ys_tr, ux_tr, ys_te, ux_te; max_iter=max_iter
    )
    free_score, free_trace = holdout_score(
        wrap(free_sm), ys_tr, ux_tr, ys_te, ux_te; max_iter=max_iter
    )
    #=
    The generating model's own held-out score, as the reference the two
    candidates are read against. Without it a table of two negative numbers says
    only which is larger, not whether either is any good — and on these models
    "neither" is a live possibility.
    =#
    steps = sum(size(y, 2) for y in ys_te)
    truth_score = elbo(truth.lds, ys_te; ux=ux_te) / steps
    return (
        gen=gen,
        truth=truth_score,
        lqr=lqr_score,
        lds=free_score,
        delta=lqr_score - free_score,
        winner=(lqr_score > free_score ? :lqr : :lds),
        correct=((lqr_score > free_score ? :lqr : :lds) === gen),
        p_lqr=nparams(lqr_sm, nref),
        p_lds=nparams(free_sm, nref),
        ntrain=length(tr_idx),
        ntest=length(te_idx),
        tsteps=tsteps,
    )
end
