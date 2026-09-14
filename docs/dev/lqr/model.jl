#=============================================================================
Ground truth — the generating models, and the data they produce.

Three things live here:

  * the structural pieces — `plant`, `reference_map`, `cost_bank`,
    `schedule_for`, `emission` — which the switching models in `slds.jl` build
    from as well, so the two sections study the same control problem.
  * `lqr_truth` — a single inverse-LQR system, with knobs for every structural
    feature the sweeps vary: terminal factor, reference targets, a within-trial
    cost onset, affine drift, and how much of the latent the emission sees.
  * `simulate` — the data, in a "from the model" (`:rand`) and a "from the
    agent" (`:lqr`) flavor. The switching generators are in `slds.jl`.

The design decision worth knowing: `C = [I 0]` by default. An inverse-LQR fit has
two nested identifiability problems — the emission's latent basis, and the cost
given the latents — and mixing them tells you nothing about either. Fixing the
emission to the identity isolates the second, which is the one this model is
for. `free_C` relaxes it if you want to see the cost of the other.
=============================================================================#

# ---------------------------------------------------------------------------
# Structural pieces
# ---------------------------------------------------------------------------

"""
    plant(n) -> (A, S)

A mild contraction with nearest-neighbour coupling, and a control term scaled so
the symplectic spectral radius stays near 1. An LQR matrix has reciprocal
eigenvalue pairs `(μ, 1/μ)`, so `ρ(M)^T` is how fast the model's own forward
chain diverges — and a sampler-based recovery check needs that modest.
"""
function plant(n::Int)
    A = Matrix(0.96I, n, n)
    for i in 1:(n - 1)
        A[i, i + 1] = 0.05
        A[i + 1, i] = -0.04
    end
    S = Matrix(0.05I, n, n) + fill(0.01, n, n) - Diagonal(fill(0.01, n))
    return A, S
end

"""
    reference_map(n, nref; radius) -> Matrix (n × nref)

The truth's `Gref`: column `j` is the state-space reference the agent steers
toward when target `j` is the one presented. Deterministic — a truth that moved
with the rng would make two conditions incomparable — and phase-shifted so the
columns are mutually distinct and none is a multiple of another.

With `nref = 1` there is one reference and it never varies across trials, which
is exactly the degenerate case: a constant `−Q₁ G_r u` is indistinguishable from
the affine drift `h`'s costate half unless something else breaks the tie. Two or
more targets vary within the dataset and identify the *contrasts* between
reference vectors; a terminal factor or a second cost regime is what identifies
their common level.
"""
function reference_map(n::Int, nref::Int; radius::Float64=1.5, ring::Bool=false)
    nref == 0 && return zeros(0, 0)
    (ring || n == 2) && return ring_map(n, nref; radius=radius)
    G = Matrix{Float64}(undef, n, nref)
    for j in 1:nref, i in 1:n
        G[i, j] = radius * cos(2π * (j - 1) / nref + π * (i - 1) / n)
    end
    return G
end

"""
    ring_map(n, nref; radius) -> Matrix (n × nref)

Targets equally spaced on a circle in the first two state coordinates, zero in
any others — the centre-out reaching layout, and the geometry the gold-standard
configuration uses.

A ring is the right default for `n = 2` and worth asking for explicitly above
it. It makes every target the same distance from the origin and from its
neighbours, so no target is easier than another and the recovered `Gref` can be
read as a shape rather than as a list; and it is what the experiment it stands
in for actually looks like.
"""
function ring_map(n::Int, nref::Int; radius::Float64=1.5)
    n >= 2 || throw(ArgumentError("a ring needs at least two state dimensions"))
    G = zeros(n, nref)
    for j in 1:nref
        θ = 2π * (j - 1) / nref
        G[1, j] = radius * cos(θ)
        G[2, j] = radius * sin(θ)
    end
    return G
end

"""
    mixed_noise(n; state, costate) -> Matrix (2n × 2n)

The mixed-coordinate innovation covariance, block-diagonal, with the costate
block far smaller than the state block.

That asymmetry is not a tuning knob dressed up as a default — it is what the
model is about. On the optimal path the costate is a *deterministic* function of
the state, `λ_t = P_t x_t + g_t`, so the innovation an optimal agent actually
produces has rank `n`, not `2n`: the state half carries the plant noise and the
costate half carries nothing. A full-rank `Σ` contains that only as a limit, and
the size of its costate block is how far from the limit the model sits. Set it
comparable to the state block and the fit has `n` free directions of slack in
which the cost can drift without penalty, which is exactly where these fits go
wrong.

A small positive value rather than zero: the smoother factorizes `Q^fwd = G Σ Gᵀ`,
so `Σ` has to stay positive definite.
"""
function mixed_noise(n::Int; state::Float64=0.02, costate::Float64=1e-4)
    d = 2n
    Σ = zeros(d, d)
    Σ[1:n, 1:n] .= Matrix(state * I, n, n)
    Σ[(n + 1):d, (n + 1):d] .= Matrix(costate * I, n, n)
    return Σ
end

"""
    sigma_prior(n; state, costate, strength) -> IWPrior or nothing

An inverse-Wishart prior on the mixed-coordinate innovation, centred on the
block structure [`mixed_noise`](@ref) describes.

`Ψ = (ν + d + 1) Σ₀` is what puts the prior's *mode* exactly at `Σ₀`, so
`strength` moves how hard the fit is held there without moving where "there" is.
That separation is the point: the harness has two ways to say "the costate
innovation is small" — start it there and hope, or say it as a prior and let the
M-step trade it off against the data — and the comparison is only meaningful if
the two name the same target.

`strength = 0` returns `nothing`, the unregularized fit.
"""
function sigma_prior(
    n::Int; state::Float64=0.02, costate::Float64=1e-4, strength::Float64=0.0
)
    strength > 0 || return nothing
    d = 2n
    Σ₀ = mixed_noise(n; state=state, costate=costate)
    return IWPrior(; Ψ=(strength + d + 1) .* Σ₀, ν=strength)
end

"""
    qc_prior(n; scale, strength) -> IWPrior, vector of priors, or nothing

An inverse-Wishart prior on cost matrices, centred on `scale · I`. A scalar
`scale` returns one prior to share across regimes. A vector returns one prior per
`Qc` epoch in schedule order, so running, delay, and terminal costs can have
different modes.

This is the honest version of the initial cost scale. The `q0` ladder shows the
fitted scale barely moves from where it starts, which means the analysis has a
prior in it whether or not anyone wrote one down — an infinitely strong one,
placed by the initialization and invisible in the output. Writing it as a prior
makes its strength a number a reader can see and a fit can trade against.

`Ψ = (ν + n + 1) · scale · I` puts the mode at `scale · I`; `strength = 0`
returns `nothing`.
"""
function qc_prior(
    n::Int; scale::Union{Real,AbstractVector{<:Real}}=0.2, strength::Float64=0.0
)
    strength > 0 || return nothing
    make(s) = IWPrior(;
        Ψ=(strength + n + 1) * Float64(s) .* Matrix(1.0I, n, n), ν=strength
    )
    scale isa Real && return make(scale)
    out = Vector{Union{Nothing,IWPrior{Float64}}}(undef, length(scale))
    for k in eachindex(scale)
        out[k] = make(scale[k])
    end
    return out
end

"""
    cost_bank(n; terminal, onset) -> (Qc, idx)

The cost regimes and where each one sits in `Qc`, given which structural
features are switched on.

The *running* cost is always `Qc[1]`, whatever else is present. That is not
cosmetic: `rescale_costate!(...; target = :trace)` reads the canonical scale off
`tr(Qc[1])`, and anchoring the whole comparison on a near-zero delay-epoch cost
would divide every parameter by noise. `cost_schedule` would put the delay
regime first, so this builds the schedule by hand instead.

`idx` names the regimes for the scoring code: `(run, delay, term)`, with
`nothing` where the feature is off.
"""
function cost_bank(n::Int; terminal::Bool, onset::Int)
    Qrun = Matrix(0.20I, n, n) + fill(0.03, n, n) - Diagonal(fill(0.03, n))
    Qc = [Qrun]
    delay_idx = nothing
    term_idx = nothing
    if onset > 1
        # A near-zero cost, not an exactly zero one: `Qc` must stay PSD, and a
        # singular regime makes its own Riccati sweep degenerate.
        push!(Qc, Matrix(0.02I, n, n))
        delay_idx = length(Qc)
    end
    if terminal
        push!(Qc, 3.0 * Matrix(1.0I, n, n))
        term_idx = length(Qc)
    end
    return Qc, (run=1, delay=delay_idx, term=term_idx)
end

"""
    schedule_for(tsteps, idx; terminal, onset) -> Vector{Int}

Per-timestep cost index: the delay regime up to `onset - 1`, the running cost
from there, and the terminal regime in the last slot when there is one. Entry
`t < T` indexes the transition `t → t+1`; entry `T` is read only by the terminal
factor.
"""
function schedule_for(tsteps::Int, idx; terminal::Bool, onset::Int)
    (terminal || onset > 1) || return Int[]
    sched = fill(idx.run, tsteps)
    if idx.delay !== nothing
        sched[1:(onset - 1)] .= idx.delay
    end
    terminal && (sched[end] = idx.term)
    return sched
end

"""
    LqrTruth

A generating system and everything the harness needs to fit it back: the state
model, the emission it is read through, the regime index map, and the per-trial
input sequences (the one-hot target identity, when the model tracks a reference).
"""
struct LqrTruth
    sm::Any
    C::Matrix{Float64}
    R::Matrix{Float64}
    lds::Any
    idx::NamedTuple
    nref::Int
    tsteps::Int
end

"""
    lqr_truth(; kwargs...) -> LqrTruth

The generating model.

# Structural knobs
- `n`: plant dimension; the latent is `2n`.
- `terminal`: add the terminal costate factor `λ_T = Q_T (x_T − r_T)`.
- `onset`: when `> 1`, a near-zero cost epoch runs up to `onset - 1` and the
  running cost takes over from there — a delay period, and a second source of
  within-trial variation in the cost.
- `nref`: how many distinct reference targets the input codes for. `0` is no
  input at all; `1` is a single fixed reference; `≥ 2` varies the target across
  trials.
- `drift`: put a nonzero mixed-coordinate bias `h` in the truth. Without it `h`
  is identically zero, and a zero parameter is not something a recovery metric
  can say anything about — the correlation is undefined and the relative error
  has no denominator. The `h` columns read `--` on every other row for exactly
  that reason.
- `state_noise`, `costate_noise`: the diagonal of the two blocks of `Σ`. See
  [`mixed_noise`](@ref) for why the second is three orders of magnitude below
  the first by default.
- `observe_costate`: let the emission read the costate half.
- `free_C`: draw a random readout instead of `[I 0]`, folding the latent-basis
  problem back in.

`Bu` is present but identically zero, and the sweeps freeze it. In a tracking
model `Bu`'s costate rows and `−Q_k G_r` both map the input into the costate; the
type's own documentation says to freeze one, and freezing the reduced-form one is
what makes `Gref` the thing being estimated.
"""
function lqr_truth(;
    n::Int=4,
    tsteps::Int=20,
    terminal::Bool=false,
    onset::Int=1,
    nref::Int=0,
    drift::Bool=false,
    observe_costate::Bool=false,
    free_C::Bool=false,
    obs_noise::Float64=0.05,
    state_noise::Float64=0.02,
    costate_noise::Float64=1e-4,
    ring::Bool=false,
    rng::AbstractRNG=MersenneTwister(0),
)
    d = 2n
    A, S = plant(n)
    Qc, idx = cost_bank(n; terminal=terminal, onset=onset)
    sched = schedule_for(tsteps, idx; terminal=terminal, onset=onset)
    # Deterministic, so the truth does not depend on where the rng happens to be.
    h = drift ? [fill(0.03, n); collect(range(-0.04, 0.04; length=n))] : nothing
    sm = LQRStateModel(
        A,
        S,
        Qc,
        mixed_noise(n; state=state_noise, costate=costate_noise);
        schedule=sched,
        terminal=terminal,
        Σf=Matrix(0.02I, n, n),
        P0=Matrix(0.2I, d, d),
        h=h,
        Bu=nref > 0 ? zeros(d, nref) : nothing,
        Gref=nref > 0 ? reference_map(n, nref; ring=ring) : nothing,
        observe_costate=observe_costate,
    )
    obs_dim = obs_width(n; free_C=free_C, observe_costate=observe_costate)
    C = emission(n, obs_dim; free_C=free_C, observe_costate=observe_costate, rng=rng)
    R = Matrix(obs_noise * I, obs_dim, obs_dim)
    lds = LinearDynamicalSystem(sm, GaussianObservationModel(copy(C), copy(R), zeros(obs_dim)))
    return LqrTruth(sm, C, R, lds, idx, nref, tsteps)
end

"""
    obs_width(n; free_C, observe_costate) -> Int

How many observed channels the emission has: `2n` when the costate is read
(`C = I` over the whole latent), `n` when it is not (`C = [I 0]`), and a wider
random readout when `free_C` folds the latent-basis problem back in.
"""
function obs_width(n::Int; free_C::Bool, observe_costate::Bool)
    free_C && return max(2n, 6)
    return observe_costate ? 2n : n
end

"""
    emission(n, obs_dim; free_C, observe_costate, rng)

`C = [I 0]` by default: the state is observed directly and the costate is not,
so the only thing left to identify is the cost. With `observe_costate` the
readout widens to the full `C = I`, which is the strongest condition in this
harness — the costate stops being something the smoother has to infer and
becomes something it is told. `free_C` draws a random readout instead, which
folds the latent-basis problem back in.
"""
function emission(n::Int, obs_dim::Int; free_C::Bool, observe_costate::Bool, rng)
    d = 2n
    C = zeros(obs_dim, d)
    if free_C
        C .= randn(rng, obs_dim, d)
        observe_costate || (C[:, (n + 1):d] .= 0)
    else
        want = observe_costate ? d : n
        obs_dim == want ||
            error("an identity readout needs obs_dim == $want; got $obs_dim")
        C[1:want, 1:want] .= Matrix(1.0I, want, want)
    end
    return C
end

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

"""
    target_inputs(rng, nref, ntrials, tsteps) -> Vector{Matrix} or nothing

One-hot target identity, constant within a trial: trial `i` presents target
`j(i)`, and the model's `Gref` maps that indicator to the reference the agent
steers toward. Targets are assigned round-robin rather than sampled, so every
target is presented an equal number of times and a condition's difficulty is not
partly an accident of which targets came up.

`nref = 1` degenerates to a column of ones — a single fixed reference on every
trial, which is the case where `Gref` and the affine drift `h` compete.
"""
function target_inputs(rng::AbstractRNG, nref::Int, ntrials::Int, tsteps::Int)
    nref == 0 && return nothing
    return [
        begin
            u = zeros(nref, tsteps)
            u[mod1(i, nref), :] .= 1.0
            u
        end for i in 1:ntrials
    ]
end

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

"""
    simulate(rng, truth, ntrials; gen, slack, process_noise, uxs) -> Vector{Matrix}

Observations from one of the two generative modes.

  * `:rand` — trials drawn from the forward chain the smoother assumes. Recovery
    is a well-posed estimation question here.
  * `:lqr` — trials from an agent that actually solves the control problem. This
    is the case the model is *for*, and it is harder: an exactly optimal
    trajectory has a rank-`n`, time-varying innovation that a full-rank constant
    `Σ` contains only as a limit, so fitting one is a projection rather than
    estimation. `slack` is the standard deviation of the perturbation on the
    agent's own costate — how far from exactly optimal it is — and the agent
    *acts* on the perturbed costate, so the slack moves the state too.
"""
function simulate(
    rng::AbstractRNG,
    truth::LqrTruth,
    ntrials::Int;
    gen::Symbol=:rand,
    slack::Float64=0.1,
    process_noise::Bool=true,
    uxs=nothing,
)
    gen in (:rand, :lqr) || throw(ArgumentError("gen must be :rand or :lqr; got :$gen"))
    T = truth.tsteps
    gen === :rand && return last(rand(rng, truth.lds, fill(T, ntrials); ux=uxs))
    obs_dim = size(truth.C, 1)
    L = cholesky(Symmetric(truth.R)).L
    return [
        truth.C * simulate_lqr(
            rng,
            truth.sm,
            T;
            costate_slack=slack,
            process_noise=process_noise,
            ux=(uxs === nothing ? nothing : uxs[i]),
        ) .+ L * randn(rng, obs_dim, T) for i in 1:ntrials
    ]
end
