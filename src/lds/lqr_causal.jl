#=============================================================================
Causal (closed-loop) inverse LQR — the `:causal` mode of `LQRStateModel`.

The `:lqr` mode fits the two-point boundary-value form, whose noise is a constant
`Σ` on the mixed residual. A feedback controller under plant noise does not
generate that: it re-plans every step, so its costate stays on the graph of the
Riccati map and its innovation is a time-varying, `P`-dependent image of the
plant noise. The `:causal` mode is that controller, exactly:

    x_{t+1} = W_{t+1}(A x_t + c − S(g_{t+1} + ν_{t+1})) + ε_t      (slack drives state)
    λ_{t+1} = P_{t+1} x_{t+1} + g_{t+1} + ν_{t+1},

with `W_s = (I + S P_s)⁻¹`, `P_t`, `g_t` the backward Riccati and feedforward
sweeps over the trial's own stretch of the cost schedule, plant noise
`ε ~ N(0, Σ)` and costate noise (slack) `ν ~ N(0, Ω)`, independent. With
`slack_drives_state = false` the `S ν` term is dropped and `ν` is a pure readout.

On `z = [x; λ]` this is a stable linear-Gaussian chain whose transition and
noise vary with the step, through `P_{t+1}`. In the coordinates

    ε_t     = W_{t+1}(x_{t+1} + S λ_{t+1} − A x_t − C̃ ũ)       (or x_{t+1} − Φ_t x_t − b_t ũ)
    ν_{t+1} = λ_{t+1} − P_{t+1} x_{t+1} − G_{t+1} ũ

the noise is the constant `blkdiag(Σ, Ω)` and the change of variables has unit
determinant, so the M-step profiles `Σ` and `Ω` out in closed form and carries
no Jacobian term. That is why the costate stays in the state.

Inputs (`h`, `Bu`, `Gref`) must be constant within a trial: the feedforward is
then `g_t = G_t ũ` with `ũ = [1; u]` and `G_t` a parameter-only sweep, shared by
every trial of a horizon. A horizon is a `(cost offset, length)` pair; the
offset is dropped when the schedule is empty, since the sweep cannot see it.

This file holds the constructor, the horizon caches the E-step kernels read,
and the per-horizon sufficient statistics; the M-step's causal unit lives with
the other units in `lqr_mstep.jl`.
=============================================================================#

# ============================================================================
# Construction
# ============================================================================

"""
    causal_state_model(A, S, Qc, Σ, Ω; kwargs...) -> LQRStateModel

A `LQRStateModel` in `:causal` mode: the **finite-horizon feedback controller**
of the control problem `(A, S, Qc)`, with plant noise `Σ` and costate noise `Ω`.

Sweeping backward over each trial's own stretch of the cost schedule,

```math
P_t = Q_{k(t)} + A^\\top P_{t+1} W_{t+1} A, \\qquad
W_{t} = (I + S P_{t})^{-1},
```

from `P_T = Q_{k_T}` (`terminal_cost = true`) or `P_T = 0`, with the feedforward
`g_t` of the affine terms, each transition is

```math
x_{t+1} = W_{t+1}\\big(A x_t + c - S(g_{t+1} + \\nu_{t+1})\\big) + \\varepsilon_t,
\\qquad
\\lambda_{t+1} = P_{t+1} x_{t+1} + g_{t+1} + \\nu_{t+1},
```

`ε_t ~ N(0, Σ)`, `ν_t ~ N(0, Ω)`, independent. With `slack_drives_state` and
`Ω = σ²I` this is exactly what [`simulate_lqr`](@ref) generates with
`costate_slack = σ`: the agent plans from what it knows (certainty
equivalence), *acts* on its own perturbed costate, and the plant noise lands
afterwards. `rand` samples the general model. Its innovation therefore has the structure the
theory implies rather than a free covariance — the costate innovation is
`P_{t+1}` times the state innovation plus `ν`, and the slack moves the state by
`−W_{t+1} S ν`:

```math
\\operatorname{Cov}\\begin{bmatrix} x_{t+1} \\\\ \\lambda_{t+1} \\end{bmatrix}_{\\mid z_t}
 = L_t \\begin{bmatrix} \\Sigma & 0 \\\\ 0 & \\Omega \\end{bmatrix} L_t^\\top,
\\qquad
L_t = \\begin{bmatrix} I & -W_{t+1} S \\\\ P_{t+1} & I - P_{t+1} W_{t+1} S \\end{bmatrix}.
```

With `slack_drives_state = false`, `L_t = [I 0; P_{t+1} I]`: the costate
innovation is `P ε + ν` and `ν` never reaches the state, so `Ω` is invisible to
an emission that does not read the costate.

The forward chain on `z = [x; λ]` is stable, so there is no terminal factor, no
conditioning normalizer, and [`rand`](@ref) samples the model directly.

## Compared with `:lqr` mode

`:lqr` puts a constant full-rank `Σ` on the mixed residual of the stationarity
conditions. On data from a causal controller that is misspecified — the residual
is rank `n`, `P`-dependent and time-varying — and in
`docs/dev/lqr/noise_prototype.md` that misspecification biases the recovered
cost contrast by a factor of 2–2.6. This mode is the model that data come from.

## Inputs

`h`, `Bu` and `Gref` keep their `:lqr` meanings: `c = h_x + B_{u,x} u` is the
plant drift, `f_t = h_λ + B_{u,λ} u − Q_{k(t)} G_r u` the costate forcing (the
tracking term included), and with `terminal_cost` the terminal feedforward is
`h_f − Q_{k_T} G_r u`. **Each trial's input must be constant across its bins**
(one target per trial, say): the feedforward is then a parameter-only map of
`ũ = [1; u]`, shared by every trial of a horizon. Per-regime reference columns
go through `gref_gate` (see [`set_gref_gate!`](@ref)), not through inputs that
switch on and off within a trial. Time-varying inputs are refused at every
entry point.

## Gauge

The costate scale `λ → cλ` maps `P → cP`, `S → S/c`, `Qc → cQc`, the feedforward,
`h_λ`, `B_{u,λ}` and `h_f` by `c`, and `Ω → c²Ω`; `W`, `Φ` and the state are
untouched, so the likelihood is invariant whenever the emission does not read
the costate. [`rescale_costate!`](@ref) applies it.

# Arguments
- `A`: `n × n` plant (need not be invertible).
- `S`: `n × n` symmetric control authority `B R⁻¹ Bᵀ`.
- `Qc`: one symmetric `n × n` cost, or a vector of them with a `schedule`.
- `Σ`: `n × n` positive-definite plant noise.
- `Ω`: `n × n` positive-definite costate noise.

# Keywords
- `schedule`, `terminal_regime`: as for [`LQRStateModel`](@ref). Entry
  `schedule[T]` names the terminal cost when `terminal_cost` is set.
- `terminal_cost::Bool = false`: start the sweep from `P_T = Q_{k_T}` rather
  than a free endpoint.
- `slack_drives_state::Bool = true`: see above.
- `plant_noise`, `costate_noise` (`:dense` or `:diagonal`): the structure of
  `Σ` and `Ω`. A diagonal block must be passed diagonal, and stays diagonal.
- `hf`: terminal feedforward offset (`n`), read when `terminal_cost` is set.
- `h`, `Bu`, `Gref`, `gref_gate`, `x0`, `B0`, `P0`, `observe_costate`,
  `fit_flags`, `mstep_iters`, `P0_prior`, `x0_prior`, `Qc_prior`: as for
  [`LQRStateModel`](@ref). `fit_flags.terminal` gates fitting `hf`.
- `Σ_prior`: an inverse-Wishart `IW(Ψ, ν)` on the `2n × 2n` `blkdiag(Σ, Ω)`.
  Since the two blocks are fitted separately, each takes the prior's *marginal*
  on its block, `IW(Ψ_bb, ν − n)` (and, for a diagonal block, the marginal of
  each entry, `IW(Ψ_ii, ν − 2n + 1)`).
- `fixed_costate_sigma`: pin `Ω = v·I` and fit only `Σ`.

The `:causal` mode does not support schedule boundaries (bridges, entries) or
switching models (`SLDS`) yet.
"""
function causal_state_model(
    A::AbstractMatrix{T},
    S::AbstractMatrix{T},
    Qc::Union{AbstractMatrix{T},AbstractVector{<:AbstractMatrix{T}}},
    Σ::AbstractMatrix{T},
    Ω::AbstractMatrix{T};
    schedule::AbstractVector{<:Integer}=Int[],
    terminal_cost::Bool=false,
    terminal_regime::Integer=0,
    slack_drives_state::Bool=true,
    plant_noise::Symbol=:dense,
    costate_noise::Symbol=:dense,
    hf::Union{Nothing,AbstractVector{T}}=nothing,
    h::Union{Nothing,AbstractVector{T}}=nothing,
    Bu::Union{Nothing,AbstractMatrix{T}}=nothing,
    Gref::Union{Nothing,AbstractMatrix{T}}=nothing,
    x0::Union{Nothing,AbstractVector{T}}=nothing,
    B0::Union{Nothing,AbstractMatrix{T}}=nothing,
    P0::Union{Nothing,AbstractMatrix{T}}=nothing,
    observe_costate::Bool=false,
    fit_flags::LQRFitFlags=LQRFitFlags(),
    mstep_iters::Int=100,
    P0_prior::Union{Nothing,IWPrior{T}}=nothing,
    x0_prior::Union{Nothing,MNPrior{T,Matrix{T}}}=nothing,
    Σ_prior::Union{Nothing,IWPrior{T}}=nothing,
    fixed_costate_sigma::Union{Nothing,Real}=nothing,
    Qc_prior=nothing,
    gref_gate::Union{Nothing,AbstractMatrix{Bool}}=nothing,
) where {T<:Real}
    n = size(A, 1)
    d = 2n
    Qc_vec = Qc isa AbstractMatrix ? [Qc] : collect(Qc)
    sched = collect(Int, schedule)
    term_k = Int(terminal_regime)
    opts = CausalOptions(slack_drives_state, plant_noise, costate_noise, terminal_cost)
    _check_lqr_structure(
        A, S, Qc_vec, sched, terminal_cost, term_k; require_invertible=false
    )
    if term_k != 0
        terminal_cost || throw(
            ArgumentError(
                "terminal_regime = $term_k pins the terminal cost, but `terminal_cost` " *
                "is false, so the sweep starts from a free endpoint",
            ),
        )
        1 <= term_k <= length(Qc_vec) || throw(
            ArgumentError(
                "terminal_regime must be 0 (follow the schedule) or a cost index " *
                "in 1:$(length(Qc_vec)); got $term_k",
            ),
        )
    end
    Qc_prior_value = _normalize_qc_prior(T, Qc_prior, length(Qc_vec), n)
    Σfull = _causal_noise_matrix(Σ, Ω, opts, n)
    fcs = _check_fixed_costate_sigma(T, Σfull, fixed_costate_sigma, Σ_prior, n)
    _check_causal_sigma_prior(Σ_prior, n)
    h_v, Bu_m, Gref_m, x0_v, P0_m, B0_m = _lqr_affine_defaults(
        T, n, h, Bu, Gref, x0, P0, B0, fit_flags, "causal"
    )
    hf_v = hf === nothing ? zeros(T, n) : hf
    length(hf_v) == n || throw(DimensionMismatchError("causal hf", n, length(hf_v)))
    mstep_iters >= 1 ||
        throw(ArgumentError("mstep_iters must be at least 1; got $mstep_iters"))

    MT = typeof(A)
    VT = typeof(h_v)
    sm = LQRStateModel{T,MT,VT}(
        :causal,
        A,
        similar(A, 0, 0),
        S,
        Qc_vec,
        sched,
        0,
        #= No terminal factor: the terminal cost lives in `causal.terminal_cost`
        and enters the sweep, not the likelihood, so nothing is conditioned on. =#
        false,
        term_k,
        false,
        Σfull,
        h_v,
        Bu_m,
        Gref_m,
        Matrix{T}(I, n, n),
        hf_v,
        x0_v,
        B0_m,
        P0_m,
        observe_costate,
        fit_flags,
        mstep_iters,
        P0_prior,
        x0_prior,
        Σ_prior,
        fcs,
        Qc_prior_value,
        nothing,
        nothing,
        _normalize_gref_gate(gref_gate, length(Qc_vec), size(Bu_m, 2)),
        LQRSwitch{T}[],
        opts,
        LQRCache(T, n, length(Qc_vec), size(Bu_m, 2)),
    )
    refresh!(sm)
    return sm
end

#=
`blkdiag(Σ, Ω)`, after checking each block's shape, symmetry, definiteness and
— for a `:diagonal` block — that it is diagonal. The model stores the noise in
its ordinary `2n × 2n` field so priors, rescaling and display see one matrix.
=#
function _causal_noise_matrix(
    Σ::AbstractMatrix{T}, Ω::AbstractMatrix{T}, opts::CausalOptions, n::Int
) where {T<:Real}
    for (name, X, form) in (("Σ", Σ, opts.plant_noise), ("Ω", Ω, opts.costate_noise))
        size(X) == (n, n) ||
            throw(DimensionMismatchError("causal $name rows", n, size(X, 1)))
        asym = maximum(abs, X .- transpose(X); init=zero(T))
        asym <= 1e-8 * max(one(T), maximum(abs, X; init=one(T))) ||
            throw(NotSymmetricError(name, Float64(asym)))
        isposdef(Symmetric(Matrix{T}(X))) ||
            throw(ArgumentError("causal $name must be positive definite"))
        form === :diagonal &&
            !isdiag(X) &&
            throw(
                ArgumentError(
                    "`$(name === "Σ" ? "plant_noise" : "costate_noise") = :diagonal` " *
                    "needs a diagonal $name; got off-diagonal entries",
                ),
            )
    end
    full = zeros(T, 2n, 2n)
    @views full[1:n, 1:n] .= Σ
    @views full[(n + 1):(2n), (n + 1):(2n)] .= Ω
    return full
end

#=
The `Σ_prior` of a causal model acts through its marginals on the two blocks,
`IW(Ψ_bb, ν − (2n − q))` for a `q`-dimensional block (`q = 1` per entry of a
diagonal one). Those are proper — and their pseudo-counts positive — only when
`ν > 2n − 1`; a weaker prior would *inflate* the noise rather than shrink it, and
on a small fit drive the effective count negative.
=#
function _check_causal_sigma_prior(Σ_prior, n::Int)
    Σ_prior === nothing && return nothing
    d = 2n
    size(Σ_prior.Ψ) == (d, d) ||
        throw(DimensionMismatchError("causal Σ_prior scale", (d, d), size(Σ_prior.Ψ)))
    Σ_prior.ν > d - 1 || throw(
        ArgumentError(
            "a causal model's Σ_prior acts through its marginals on the plant and " *
            "costate blocks, which are proper only for ν > 2n − 1 = $(d - 1); got " *
            "ν = $(Σ_prior.ν)",
        ),
    )
    return nothing
end

"""
    _check_causal_structure(sm)

Re-check a `:causal` model's invariants (for a model assembled field by field,
or edited after construction): the cost structure, no terminal factor and no
schedule boundaries, and a block-diagonal noise matching its declared form.
"""
function _check_causal_structure(sm::LQRStateModel{T}) where {T<:Real}
    n = _plant_dim(sm)
    _check_lqr_structure(
        sm.A,
        sm.S,
        sm.Qc,
        sm.schedule,
        sm.causal.terminal_cost,
        sm.terminal_regime;
        require_invertible=false,
    )
    sm.terminal && throw(
        ArgumentError(
            "a `:causal` model has no terminal factor; its terminal cost is " *
            "`terminal_cost` in the sweep. Build it with `causal_state_model`.",
        ),
    )
    isempty(sm.switches) ||
        throw(ArgumentError("a `:causal` model does not support schedule boundaries yet"))
    isempty(sm.Mfree) || throw(
        ArgumentError(
            "a `:causal` state model has no free transition, but `Mfree` is non-empty"
        ),
    )
    Σ = Matrix{T}(sm.Σ)
    size(Σ) == (2n, 2n) || throw(DimensionMismatchError("causal Σ rows", 2n, size(Σ, 1)))
    iszero(view(Σ, 1:n, (n + 1):(2n))) && iszero(view(Σ, (n + 1):(2n), 1:n)) || throw(
        ArgumentError(
            "a `:causal` model's noise is blkdiag(Σ, Ω): plant noise and costate noise " *
            "are independent, and their coupling comes from the Riccati map alone",
        ),
    )
    _causal_noise_matrix(Σ[1:n, 1:n], Σ[(n + 1):(2n), (n + 1):(2n)], sm.causal, n)
    v = sm.fixed_costate_sigma
    v === nothing ||
        Σ[(n + 1):(2n), (n + 1):(2n)] ≈ v * I ||
        throw(ArgumentError("fixed_costate_sigma = $v requires Ω = $v·I"))
    _check_causal_sigma_prior(sm.Σ_prior, n)
    return nothing
end

# ============================================================================
# Horizons: registry, construction, lookup
# ============================================================================

"""
    _time_to_go_dynamics(lds) -> Bool

Whether the model's transition at step `t` depends on the steps *left* in the
trial rather than only on `t` — true of a `:causal` LQR model, whose Riccati
sweep runs back from each trial's end. The ragged-length smoother shares the
leading precision blocks of different lengths, which is only valid when this is
false.
"""
_time_to_go_dynamics(::Any) = false
_time_to_go_dynamics(sm::LQRStateModel) = _is_causal(sm)
_time_to_go_dynamics(lds::LinearDynamicalSystem) = _time_to_go_dynamics(lds.state_model)

"""
    _causal_key(sm, offset, tsteps) -> NTuple{2,Int}

The horizon a trial of length `tsteps` starting `offset` bins into the schedule
belongs to. Without a schedule the sweep cannot see the offset, so every offset
shares one horizon per length.
"""
@inline _causal_key(sm::LQRStateModel, offset::Int, tsteps::Int) =
    (isempty(sm.schedule) ? 0 : offset, tsteps)

"""
    _register_causal_horizons!(sm, tsteps[, offsets])

Add the horizons of these trials to the model's registry (shared with its
variants), so the next [`refresh!`](@ref) builds them. Called by every entry point
before the cache is refreshed — the E-step only ever reads horizons, and a
missing one is a bug, reported as such.
"""
function _register_causal_horizons!(
    sm::LQRStateModel, tsteps::AbstractVector{Int}, offsets::AbstractVector{Int}=Int[]
)
    _is_causal(sm) || return sm
    reg = sm.cache.causal_keys
    seen = Set(reg)
    for (i, Ti) in enumerate(tsteps)
        key = _causal_key(sm, isempty(offsets) ? 0 : offsets[i], Ti)
        key in seen || (push!(reg, key); push!(seen, key))
    end
    return sm
end

"""
    _trial_cache_copy(c::LQRCache)

A per-trial model (one with its own initial mean, see `_trial_initial_model`)
copies the cache; a `:causal` model's horizons are read-only during the E-step
and can be large, so the copy shares them instead of duplicating every horizon
for every trial.
"""
function _trial_cache_copy(c::LQRCache)
    isempty(c.causal) && return deepcopy(c)
    seen = IdDict{Any,Any}(
        c.causal => c.causal,
        c.causal_index => c.causal_index,
        c.causal_keys => c.causal_keys,
    )
    return Base.deepcopy_internal(c, seen)
end

"""
    _causal_horizon(sm, tsteps) -> _CausalHorizon

The horizon of a trial of length `tsteps`, for `sm` as the trial sees it (its
`cost_offset`).
"""
@inline function _causal_horizon(sm::LQRStateModel{T}, tsteps::Int) where {T}
    c = sm.cache
    slot = get(c.causal_index, _causal_key(sm, sm.cost_offset, tsteps), 0)
    slot == 0 && _causal_missing_horizon(sm, tsteps)
    return c.causal[slot]
end

@noinline function _causal_missing_horizon(sm::LQRStateModel, tsteps::Int)
    throw(
        ArgumentError(
            "this `:causal` model has no horizon for a trial of length $tsteps at cost " *
            "offset $(sm.cost_offset). Horizons are registered at the public entry " *
            "points (`fit!`, `smooth`, `elbo`, `loglikelihood`, `rand`); call one of " *
            "them, or `refresh!` after `_register_causal_horizons!`.",
        ),
    )
end

"""
    _refresh_causal_head!(sm, c, n, d)

Fill the cache of a `:causal` model. The per-regime fields hold harmless
placeholders (zero transitions) since nothing reads them in this mode; `Qfwd` is
the constant innovation covariance `blkdiag(Σ, Ω)`, whose Cholesky factor the
kernels whiten with after mapping a residual through `Rz`, and every registered
horizon is rebuilt at the current parameters.
"""
function _refresh_causal_head!(
    sm::LQRStateModel{T}, c::LQRCache{T}, n::Int, d::Int
) where {T<:Real}
    c.logabsdetA = zero(T)
    copyto!(c.AinvT, Matrix{T}(I, n, n))
    copyto!(c.G, Matrix{T}(I, d, d))
    foreach(M -> fill!(M, zero(T)), c.M)
    foreach(B -> fill!(B, zero(T)), c.Bfwd)
    fill!(c.bfwd, zero(T))
    c.Qfwd = PDMat(Symmetrize!(Matrix{T}(sm.Σ)))
    m = size(sm.Bu, 2)
    for key in c.causal_keys
        slot = get(c.causal_index, key, 0)
        if slot == 0
            #= Filled before it is indexed: a sweep that throws must not leave an
            all-zero horizon behind for a caller that catches the error. =#
            H = _CausalHorizon(T, n, m, key[1], key[2])
            _fill_causal_horizon!(H, sm, c.Qfwd)
            push!(c.causal, H)
            c.causal_index[key] = length(c.causal)
        else
            _fill_causal_horizon!(c.causal[slot], sm, c.Qfwd)
        end
    end
    return nothing
end

"""
    _causal_regime(sm, offset, s) -> Int

The cost regime of transition `s` (`s → s+1`) for a trial `offset` bins into the
schedule.
"""
@inline _causal_regime(sm::LQRStateModel, offset::Int, s::Int) =
    isempty(sm.schedule) ? 1 : sm.schedule[s + offset]

"""The cost the sweep of a horizon ending at bin `tsteps` starts from."""
@inline function _causal_terminal_regime(sm::LQRStateModel, offset::Int, tsteps::Int)
    sm.terminal_regime > 0 && return sm.terminal_regime
    return _causal_regime(sm, offset, tsteps)
end

"""Scratch for one causal sweep, so an M-step evaluation sweeps without allocating."""
struct _SweepBuffers{T<:Real}
    K::Matrix{T}
    AK::Matrix{T}
    Mn::Matrix{T}
    V::Matrix{T}
    Ct::Matrix{T}
    Gk::Vector{Matrix{T}}
    ipiv::Vector{LinearAlgebra.BlasInt}
end

function _SweepBuffers(::Type{T}, n::Int, m::Int, K::Int=8) where {T}
    return _SweepBuffers{T}(
        zeros(T, n, n),
        zeros(T, n, n),
        zeros(T, n, n),
        zeros(T, n, 1 + m),
        zeros(T, n, 1 + m),
        [zeros(T, n, m) for _ in 1:K],
        zeros(LinearAlgebra.BlasInt, n),
    )
end

"""
    _causal_sweep!(P, W, G, A, S, Qc, h, Bu, Gref, hf, sm, offset, tsteps)

The backward Riccati and feedforward sweeps of one horizon, into `P[s]`, `W[s]`
and `G[s]` for `s = 1:tsteps` (`G[s]` is `n × (1 + m)`, the feedforward per unit
of `ũ = [1; u]`):

    P_T = τ Q_{k_T},  G_T = τ [h_f  −Q_{k_T} G_{k_T}],        τ = terminal_cost
    P_s = Q_{k(s)} + Aᵀ K_{s+1} A,                            K = P W
    G_s = Aᵀ (K_{s+1} C̃ + W_{s+1}ᵀ G_{s+1}) + F̃_s,
    W_s = (I + S P_s)⁻¹,

with `C̃ = [h_x  B_{u,x}]` and `F̃_s = [h_λ  B_{u,λ} − Q_{k(s)} G_{k(s)}]`
(`G_k` the gated reference map). This is [`lqr_riccati_sequence`](@ref)'s
recursion with `g_t = G_t ũ`: `K(C̃ − S G) + G = K C̃ + (I − P W S) G` and
`I − P W S = Wᵀ`. The parameters are passed explicitly so the M-step can sweep
at a trial point without writing it to the model.

Returns `false` (leaving the buffers partly filled) when the sweep stops being
finite or `I + S P` is singular — a point the optimizer should reject, which a
line search probing far from the current parameters can reach.
"""
function _causal_sweep!(
    P::AbstractVector{<:AbstractMatrix{T}},
    W::AbstractVector{<:AbstractMatrix{T}},
    G::AbstractVector{<:AbstractMatrix{T}},
    A::AbstractMatrix{T},
    S::AbstractMatrix{T},
    Qc::AbstractVector{<:AbstractMatrix{T}},
    h::AbstractVector{T},
    Bu::AbstractMatrix{T},
    Gref::AbstractMatrix{T},
    hf::AbstractVector{T},
    sm::LQRStateModel,
    offset::Int,
    tsteps::Int,
    buf::_SweepBuffers{T}=_SweepBuffers(T, size(A, 1), size(Bu, 2), length(Qc)),
) where {T<:Real}
    n = size(A, 1)
    m = size(Bu, 2)
    xr, lr = 1:n, (n + 1):(2n)
    gate = sm.gref_gate
    K, AK, V, Ct = buf.K, buf.AK, buf.V, buf.Ct
    # The reference map each regime reads, gated once per sweep rather than per step.
    if m > 0
        for k in eachindex(Qc)
            copyto!(buf.Gk[k], _gated(Gref, gate, k))
        end
    end
    kT = _causal_terminal_regime(sm, offset, tsteps)
    if sm.causal.terminal_cost
        copyto!(P[tsteps], Qc[kT])
        @views G[tsteps][:, 1] .= hf
        # Bound first: `@views` on a chained index with `end` does not lower
        # before Julia 1.11.
        GT = G[tsteps]
        m > 0 && @views mul!(GT[:, 2:end], Qc[kT], buf.Gk[kT], -one(T), zero(T))
    else
        fill!(P[tsteps], zero(T))
        fill!(G[tsteps], zero(T))
    end
    _causal_inv_step!(W[tsteps], S, P[tsteps], buf) || return false
    @views Ct[:, 1] .= h[xr]
    m > 0 && @views Ct[:, 2:end] .= Bu[xr, :]
    for s in (tsteps - 1):-1:1
        k = _causal_regime(sm, offset, s)
        mul!(K, P[s + 1], W[s + 1])
        mul!(AK, transpose(A), K)
        copyto!(P[s], Qc[k])
        mul!(P[s], AK, A, one(T), one(T))
        Symmetrize!(P[s])
        all(isfinite, P[s]) || return false
        mul!(V, K, Ct)
        mul!(V, transpose(W[s + 1]), G[s + 1], one(T), one(T))
        mul!(G[s], transpose(A), V)
        @views G[s][:, 1] .+= h[lr]
        if m > 0
            Gs = G[s]
            @views Gs[:, 2:end] .+= Bu[lr, :]
            @views mul!(Gs[:, 2:end], Qc[k], buf.Gk[k], -one(T), one(T))
        end
        all(isfinite, G[s]) || return false
        _causal_inv_step!(W[s], S, P[s], buf) || return false
    end
    return true
end

"""`W = (I + S P)⁻¹` in place; `false` when it is singular or not finite."""
function _causal_inv_step!(
    W::AbstractMatrix{T}, S, P, buf::_SweepBuffers{T}
) where {T<:Real}
    M = buf.Mn
    mul!(M, S, P)
    for i in axes(M, 1)
        M[i, i] += one(T)
    end
    all(isfinite, M) || return false
    fill!(W, zero(T))
    for i in axes(W, 1)
        W[i, i] = one(T)
    end
    if T <: LinearAlgebra.BlasFloat
        # Pivoted LU into the preallocated pivots: no allocation per step.
        _, ipiv, info = LAPACK.getrf!(M, buf.ipiv; check=false)
        info == 0 || return false
        LAPACK.getrs!('N', M, ipiv, W)
    else
        F = lu!(M; check=false)
        issuccess(F) || return false
        ldiv!(F, W)
    end
    return all(isfinite, W)
end

"""
    _fill_causal_horizon!(H, sm, B)

Sweep horizon `H` at `sm`'s parameters and build its per-transition transition,
input map, residual map and smoother templates. `B` is the innovation covariance
`blkdiag(Σ, Ω)` (a `PDMat`), so `Q_t⁻¹ = Rzᵀ B⁻¹ Rz` needs no inverse of `Q_t`.
"""
function _fill_causal_horizon!(
    H::_CausalHorizon{T}, sm::LQRStateModel{T}, B::PDMat{T}
) where {T<:Real}
    n = _plant_dim(sm)
    d = 2n
    m = size(sm.Bu, 2)
    xr, lr = 1:n, (n + 1):d
    _causal_sweep!(
        H.P,
        H.W,
        H.G,
        sm.A,
        sm.S,
        sm.Qc,
        sm.h,
        sm.Bu,
        sm.Gref,
        sm.hf,
        sm,
        H.offset,
        H.tsteps,
    ) || throw(
        NumericalStabilityError(
            "causal",
            "the backward Riccati sweep over a horizon of $(H.tsteps) steps is not " *
            "finite at the current parameters (or I + S P is singular)",
        ),
    )
    slack = sm.causal.slack_drives_state
    Ct = Matrix{T}(undef, n, 1 + m)
    @views Ct[:, 1] .= sm.h[xr]
    m > 0 && @views Ct[:, 2:end] .= sm.Bu[xr, :]
    X = Matrix{T}(undef, d, d)
    for t in 1:(H.tsteps - 1)
        s = t + 1
        Ws, Ps, Gs = H.W[s], H.P[s], H.G[s]
        M, Bt, Rz = H.M[t], H.B[t], H.Rz[t]
        Φ = Ws * sm.A
        b = Ws * (Ct .- sm.S * Gs)
        fill!(M, zero(T))
        @views begin
            M[xr, xr] .= Φ
            mul!(M[lr, xr], Ps, Φ)
            Bt[xr, :] .= b
            Bt[lr, :] .= Gs
            mul!(Bt[lr, :], Ps, b, one(T), one(T))
        end
        fill!(Rz, zero(T))
        @views begin
            if slack
                Rz[xr, xr] .= Ws
                mul!(Rz[xr, lr], Ws, sm.S)
            else
                for i in 1:n
                    Rz[i, i] = one(T)
                end
            end
            Rz[lr, xr] .= .-Ps
            for i in 1:n
                Rz[n + i, n + i] = one(T)
            end
        end
        # Q_t⁻¹ = Rzᵀ B⁻¹ Rz, and the templates the gradient and Hessian read.
        copyto!(X, Rz)
        ldiv!(B.chol, X)
        mul!(H.negQinv[t], transpose(Rz), X, -one(T), zero(T))
        Symmetrize!(H.negQinv[t])
        mul!(H.QinvM[t], H.negQinv[t], M, -one(T), zero(T))
        copyto!(H.MtQinv[t], transpose(H.QinvM[t]))
        mul!(H.negMtQinvM[t], transpose(M), H.QinvM[t], -one(T), zero(T))
    end
    return H
end

"""
    _check_causal_inputs(sm, ux, what)

Refuse an input that varies within a trial. The feedforward of a `:causal`
model is a sweep over the trial's *future* inputs; with an input that is constant
across the trial it is a parameter-only map of that input, shared by every trial
of a horizon, which is what this mode is built on.
"""
function _check_causal_inputs(sm::LQRStateModel, ux::AbstractMatrix, what::AbstractString)
    _is_causal(sm) || return nothing
    size(ux, 1) == 0 && return nothing
    u1 = view(ux, :, 1)
    tol =
        sqrt(eps(float(eltype(ux)))) * max(one(float(eltype(ux))), maximum(abs, u1; init=0))
    # The transitions read columns 1:T-1; the last one is never used.
    for t in 2:(size(ux, 2) - 1)
        maximum(abs, view(ux, :, t) .- u1; init=0) <= tol || throw(
            ArgumentError(
                "$what: a `:causal` model needs each trial's input to be constant " *
                "across its bins (its feedforward is a map of that one input), but " *
                "column $t differs from column 1. Give per-regime references through " *
                "`gref_gate` rather than inputs that switch within a trial.",
            ),
        )
    end
    return nothing
end

function _check_causal_inputs(sm::LQRStateModel, data::Data)
    _is_causal(sm) || return nothing
    for (i, u) in enumerate(data.ux)
        _check_causal_inputs(sm, u, "trial $i")
    end
    return nothing
end

# ============================================================================
# E-step kernels
#
# The same kernels as `lqr_latents.jl`, with the transition, input map and noise
# templates read per step from the trial's horizon instead of per regime. The
# input term uses the trial's own input column, which a constant input makes the
# same at every step.
# ============================================================================

"""
    _causal_residual!(out, sm, H, x, t, ux)

`z_t − M_{t−1} z_{t−1} − B_{t−1} ũ` for transition `t−1 → t` of horizon `H`.
"""
@inline function _causal_residual!(
    out::AbstractVector{T},
    H::_CausalHorizon,
    x::AbstractMatrix{T},
    t::Int,
    ux::Union{Nothing,AbstractMatrix},
) where {T<:Real}
    Bt = H.B[t - 1]
    @views mul!(out, H.M[t - 1], x[:, t - 1])
    @views out .+= Bt[:, 1]
    if ux !== nothing && size(Bt, 2) > 1
        @views mul!(out, Bt[:, 2:end], ux[:, t - 1], one(T), one(T))
    end
    @views out .= x[:, t] .- out
    return out
end

"""
    _causal_transition_loglik!(dxt, tmp, sm, x, t, ux) -> T

`cQ − ½‖B^{-1/2} Rz_{t−1} r_t‖²`: the transition density of `z_t`, through the
map to the independent innovations — whose unit determinant is why `cQ` (the
constant of `blkdiag(Σ, Ω)`) is the right normalizer at every step.
"""
function _causal_transition_loglik!(
    dxt::AbstractVector{T},
    tmp::AbstractVector{T},
    sm::LQRStateModel,
    x::AbstractMatrix{T},
    t::Int,
    ux,
) where {T<:Real}
    H = _causal_horizon(sm, size(x, 2))
    _causal_residual!(tmp, H, x, t, ux)
    mul!(dxt, H.Rz[t - 1], tmp)
    _whiten!(sm.cache.Qfwd.chol, dxt)
    return T(sm.cache.cQ) - T(0.5) * sum(abs2, dxt)
end

"""
    _state_gradient_causal!(grad, ws, sm, x, ux)

The state half of the complete-data gradient for a `:causal` model:
`−P0⁻¹(z₁ − x0) + M₁ᵀQ₁⁻¹r₂` at `t = 1`, `−Q_{t−1}⁻¹ r_t + M_tᵀ Q_t⁻¹ r_{t+1}` in
between and `−Q_{T−1}⁻¹ r_T` at the end, with every `Q` and `M` the step's own.
"""
function _state_gradient_causal!(
    grad::AbstractMatrix{T},
    ws::SmoothWorkspace{T},
    sm::LQRStateModel,
    x::AbstractMatrix{T},
    ux::Union{Nothing,AbstractMatrix},
) where {T<:Real}
    tsteps = size(x, 2)
    H = _causal_horizon(sm, tsteps)
    dxt = ws.opt.dxt
    dxt_next = ws.opt.dxt_next
    tmp2 = ws.opt.tmp2
    tmp3 = ws.opt.tmp3

    @views dxt .= x[:, 1] .- sm.x0
    mul!(tmp3, ws.consts.x_t, dxt)
    _causal_residual!(dxt_next, H, x, 2, ux)
    mul!(tmp2, H.MtQinv[1], dxt_next)
    @views grad[:, 1] .= tmp2 .+ tmp3

    @views for t in 2:(tsteps - 1)
        _causal_residual!(dxt, H, x, t, ux)
        mul!(tmp3, H.negQinv[t - 1], dxt)
        _causal_residual!(dxt_next, H, x, t + 1, ux)
        mul!(tmp2, H.MtQinv[t], dxt_next)
        grad[:, t] .= tmp3 .+ tmp2
    end

    _causal_residual!(dxt, H, x, tsteps, ux)
    mul!(tmp3, H.negQinv[tsteps - 1], dxt)
    @views grad[:, tsteps] .= tmp3
    return grad
end

"""
    _state_hessian_causal!(btd, cc, sm, tsteps)

Per-step state Hessian blocks of a `:causal` model: `Q_t⁻¹ M_t` off the diagonal
and `−M_tᵀ Q_t⁻¹ M_t − Q_{t−1}⁻¹` on it, each from its own step.
"""
function _state_hessian_causal!(
    btd, cc::SmoothConstants{T}, sm::LQRStateModel, tsteps::Int
) where {T<:Real}
    H = _causal_horizon(sm, tsteps)
    for i in 1:(tsteps - 1)
        copyto!(btd.H_sub[i], H.QinvM[i])
        copyto!(btd.H_super[i], transpose(H.QinvM[i]))
    end
    btd.H_diag[1] .= H.negMtQinvM[1] .+ cc.x_t
    for t in 2:(tsteps - 1)
        btd.H_diag[t] .= H.negMtQinvM[t] .+ H.negQinv[t - 1]
    end
    copyto!(btd.H_diag[tsteps], H.negQinv[tsteps - 1])
    return nothing
end

"""
    _gradient_batched_causal!(ws, lds, x, y, ux, uy)

[`gradient_batched!`](@ref) for a `:causal` model: the trial axis stacked into
BLAS-3 products as there, with the step's own transition and noise template.
Every trial in a batch shares one length (and the batched path refuses cost
offsets), so they share one horizon.
"""
function _gradient_batched_causal!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    ux::AbstractArray{T,3},
    uy::AbstractArray{T,3},
) where {T<:Real,S<:LQRStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(x, 2)
    sm = lds.state_model
    H = _causal_horizon(sm, tsteps)
    has_input = lds.ux_dim > 0
    C = lds.obs_model.C
    d_obs = lds.obs_model.d
    D_obs = lds.obs_model.D
    C_inv_R = ws.consts.C_inv_R
    neg_P0_inv = ws.consts.x_t

    bat = ws.batched::BatchedBuffers{T}
    grad = bat.grad_buf
    dxt = bat.dxt
    dxt_next = bat.dxt_next
    dyt = bat.dyt
    tmp1 = bat.tmp1
    tmp2 = bat.tmp2
    tmp3 = bat.tmp3

    @inline function residual!(dst, t)
        Bt = H.B[t - 1]
        @views begin
            mul!(dst, H.M[t - 1], x[:, t - 1, :])
            dst .+= Bt[:, 1]
            has_input && mul!(dst, Bt[:, 2:end], ux[:, t - 1, :], one(T), one(T))
            dst .= x[:, t, :] .- dst
        end
        return dst
    end

    @inline function emission!(t)
        @views begin
            mul!(dyt, C, x[:, t, :])
            mul!(dyt, D_obs, uy[:, t, :], one(T), one(T))
            dyt .= y[:, t, :] .- dyt .- d_obs
        end
        return mul!(tmp1, C_inv_R, dyt)
    end

    @views dxt .= x[:, 1, :] .- sm.x0
    residual!(dxt_next, 2)
    emission!(1)
    mul!(tmp2, H.MtQinv[1], dxt_next)
    mul!(tmp3, neg_P0_inv, dxt)
    @views grad[:, 1, :] .= tmp1 .+ tmp2 .+ tmp3

    @views for t in 2:(tsteps - 1)
        residual!(dxt, t)
        residual!(dxt_next, t + 1)
        emission!(t)
        mul!(tmp2, H.MtQinv[t], dxt_next)
        mul!(tmp3, H.negQinv[t - 1], dxt)
        grad[:, t, :] .= tmp1 .+ tmp3 .+ tmp2
    end

    residual!(dxt, tsteps)
    emission!(tsteps)
    mul!(tmp3, H.negQinv[tsteps - 1], dxt)
    @views grad[:, tsteps, :] .= tmp1 .+ tmp3
    return grad
end

"""
    _sample_causal_path!(rng, z, sm, ux)

One latent path from a `:causal` model: `z₁` from the initial prior, then
`z_t = M_{t−1} z_{t−1} + B_{t−1} ũ + Rz_{t−1}⁻¹ e_t`, `e_t ~ N(0, blkdiag(Σ, Ω))`.
The chain is the stable closed loop, so this is the model's own distribution
over any horizon — no terminal conditioning, no divergence.
"""
function _sample_causal_path!(
    rng::AbstractRNG, z::AbstractMatrix{T}, sm::LQRStateModel{T}, ux::AbstractMatrix{T}
) where {T<:Real}
    tsteps = size(z, 2)
    H = _causal_horizon(sm, tsteps)
    d = size(z, 1)
    L0 = cholesky(Symmetric(Matrix{T}(sm.P0))).L
    LB = sm.cache.Qfwd.chol.L
    @views z[:, 1] .= sm.x0 .+ L0 * randn(rng, T, d)
    e = Vector{T}(undef, d)
    has_input = size(ux, 1) > 0
    for t in 2:tsteps
        Bt = H.B[t - 1]
        mul!(e, LB, randn(rng, T, d))
        @views begin
            z[:, t] .= H.Rz[t - 1] \ e
            mul!(z[:, t], H.M[t - 1], z[:, t - 1], one(T), one(T))
            z[:, t] .+= Bt[:, 1]
            has_input && mul!(z[:, t], Bt[:, 2:end], ux[:, t - 1], one(T), one(T))
        end
    end
    return z
end

# ============================================================================
# Sufficient statistics, per horizon and transition
# ============================================================================

"""
    _causal_stats_slot!(hs, key, tsteps, d, reg) -> Int

The slot of horizon `key` in `hs`, allocating its per-transition blocks the
first time it is seen. Slots persist across EM iterations (the blocks are zeroed,
not reallocated), so a fit allocates them once.
"""
function _causal_stats_slot!(hs, key::NTuple{2,Int}, tsteps::Int, d::Int, reg::Int)
    T = eltype(hs.nk)
    slot = findfirst(==(key), hs.causal_keys)
    slot === nothing || return slot
    push!(hs.causal_keys, key)
    push!(hs.causal_zz, [zeros(T, reg, reg) for _ in 1:(tsteps - 1)])
    push!(hs.causal_zy, [zeros(T, reg, d) for _ in 1:(tsteps - 1)])
    push!(hs.causal_yy, [zeros(T, d, d) for _ in 1:(tsteps - 1)])
    push!(hs.causal_n, zero(T))
    return length(hs.causal_keys)
end

"""
    _aggregate_causal_stats!(hs, tfs, lds, data[, trials]) -> hs

Per-horizon, per-transition statistics of a `:causal` model from the smoother
output (see the `causal_*` fields of [`LQRSufficientStatistics`](@ref)). Every
block is zeroed first, so the result is exactly the given trials' — which is
what lets [`trial_elbos`](@ref) score one trial through the same path.

`nk[1]` is set to the total transition count, which is what the M-step's units
and noise versions count; the other regime blocks stay zero.
"""
function _aggregate_causal_stats!(
    hs,
    tfs::TrialFilterSmooth{T},
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T},
    trials::AbstractVector{Int}=Base.OneTo(length(tfs)),
) where {T<:Real,S<:LQRStateModel{T},O<:AbstractObservationModel{T}}
    sm = lds.state_model
    d = lds.latent_dim
    m = lds.ux_dim
    reg = d + 1 + m
    empty!(hs.terminal_inputs)
    empty!(hs.terminal_counts)
    empty!(hs.terminal_ux0)
    empty!(hs.terminal_offsets)
    for k in eachindex(hs.zz)
        fill!(hs.zz[k], zero(T))
        fill!(hs.zy[k], zero(T))
        fill!(hs.yy[k], zero(T))
        hs.nk[k] = zero(T)
        fill!(hs.term_zz[k], zero(T))
        hs.term_n[k] = zero(T)
    end
    for h in eachindex(hs.causal_keys)
        foreach(Z -> fill!(Z, zero(T)), hs.causal_zz[h])
        foreach(Z -> fill!(Z, zero(T)), hs.causal_zy[h])
        foreach(Z -> fill!(Z, zero(T)), hs.causal_yy[h])
        hs.causal_n[h] = zero(T)
    end
    w = Vector{T}(undef, reg)
    ntrans = zero(T)
    index = Dict{NTuple{2,Int},Int}(k => i for (i, k) in enumerate(hs.causal_keys))
    #=
    Without a schedule the sweep depends only on the steps left, so transition
    `t` of a trial of length `T_n` is transition `t + (L − T_n)` of the longest
    trial's horizon `L`: every trial is aligned at the end of that one horizon.
    The M-step then visits `L − 1` transitions however ragged the data are,
    rather than one block per (length, step). With a schedule the sweep reads
    the trial's own stretch of it, and each horizon keeps its own statistics.
    =#
    align = isempty(sm.schedule)
    L = align ? maximum(t -> size(tfs[t].x_smooth, 2), trials; init=0) : 0
    for trial in trials
        fs = tfs[trial]
        x = fs.x_smooth::Matrix{T}
        p_smooth = fs.p_smooth::Array{T,3}
        p_tt1 = fs.p_smooth_tt1::Array{T,3}
        T_n = size(x, 2)
        Th = align ? L : T_n
        shift = Th - T_n
        key = _causal_key(sm, _trial_cost_offset(data, trial), Th)
        h = get(index, key, 0)
        if h == 0
            h = _causal_stats_slot!(hs, key, Th, d, reg)
            index[key] = h
        end
        hs.causal_n[h] += T(T_n - 1)
        ntrans += T(T_n - 1)
        ux = data.ux[trial]
        w[d + 1] = one(T)
        m > 0 && @views w[(d + 2):reg] .= ux[:, 1]
        for t in 1:(T_n - 1)
            zz = hs.causal_zz[h][t + shift]
            zy = hs.causal_zy[h][t + shift]
            yy = hs.causal_yy[h][t + shift]
            @views w[1:d] .= x[:, t]
            z_next = tview(x, :, t + 1)
            BLAS.ger!(one(T), w, w, zz)
            BLAS.ger!(one(T), w, z_next, zy)
            BLAS.ger!(one(T), z_next, z_next, yy)
            @views begin
                zz[1:d, 1:d] .+= p_smooth[:, :, t]
                yy .+= p_smooth[:, :, t + 1]
                zy[1:d, :] .+= adjoint(p_tt1[:, :, t + 1])
            end
        end
    end
    isempty(hs.nk) || (hs.nk[1] = ntrans)
    return hs
end

"""
    _pool_causal_stats!(out, s) -> out

Add `s`'s per-horizon statistics into `out`, matching horizons by key — two
cells of a grouped fit need not have seen the same horizons.
"""
function _pool_causal_stats!(out, s)
    isempty(s.causal_keys) && return out
    d = size(first(s.causal_yy[1]), 1)
    reg = size(first(s.causal_zz[1]), 1)
    for (hs, key) in enumerate(s.causal_keys)
        h = _causal_stats_slot!(out, key, key[2], d, reg)
        for t in eachindex(s.causal_zz[hs])
            out.causal_zz[h][t] .+= s.causal_zz[hs][t]
            out.causal_zy[h][t] .+= s.causal_zy[hs][t]
            out.causal_yy[h][t] .+= s.causal_yy[hs][t]
        end
        out.causal_n[h] += s.causal_n[hs]
    end
    return out
end

# ============================================================================
# The causal unit of the structural M-step
#
# A causal unit's residual for transition `t` of horizon `h` (with `s = t + 1`)
# is `r = L_t z_{t+1} − Θ_t w̃_t`, `w̃ = [z_t; 1; u]`, in the coordinates of the
# independent innovations `[ε_t; ν_s]`:
#
#   slack drives the state:   L_t = [W_s  W_s S; −P_s  I],
#                             Θ_t = [W_s A  0  W_s C̃; 0  0  G_s]
#   slack is a readout:       L_t = [I  0; −P_s  I],
#                             Θ_t = [W_s A  0  W_s (C̃ − S G_s); 0  0  G_s]
#
# (`C̃ = [h_x  B_{u,x}]`). `det L_t = 1` either way, so the unit adds its scatter
# `Σ_t L Y Lᵀ − L Zyᵀ Θᵀ − Θ Zy Lᵀ + Θ Z Θᵀ` to its noise version's `R` and
# carries no Jacobian term. The noise blocks profile separately (see
# `_causal_noise_objective!`).
#
# The gradient pulls `L̄ = 𝒲 (L Y − Θ Zy)` and `Θ̄ = 𝒲 (Θ Z − L Zyᵀ)` back to
# `(W_s, P_s, G_s, A, S, C̃)` per transition, then through the sweeps in reverse
# (`s = 2 … T`, the order that completes each adjoint before it is used):
#
#   W_s = (I + S P_s)⁻¹         M̄ = −Wᵀ W̄ Wᵀ:  S̄ += M̄ P_s,  P̄_s += S M̄
#   P_s = Q_{k(s)} + Aᵀ K A      Q̄_k += P̄_s,  Ā += K A P̄_sᵀ + Kᵀ A P̄_s,  K̄ = A P̄_s Aᵀ
#   G_s = Aᵀ V + F̃_s            Ā += V Ḡ_sᵀ,  V̄ = A Ḡ_s,  F̃̄ = Ḡ_s
#   V = K C̃ + W_{s+1}ᵀ G_{s+1}  K̄ += V̄ C̃ᵀ,  C̃̄ += Kᵀ V̄,  W̄_{s+1} += G_{s+1} V̄ᵀ,
#                                Ḡ_{s+1} += W_{s+1} V̄
#   K = P_{s+1} W_{s+1}          P̄_{s+1} += K̄ W_{s+1}ᵀ,  W̄_{s+1} += P_{s+1}ᵀ K̄
#   F̃_s = [h_λ  B_{u,λ} − Q_k G_k]  and the terminal  P_T = Q_{k_T},
#   G_T = [h_f  −Q_{k_T} G_{k_T}]   in the obvious way.
#
# Every line is checked against central differences of the packed objective in
# `test/LinearDynamicalSystems/CausalLQR.jl`.
# ============================================================================

"""
    _CausalScratch{T}

A causal unit's sweeps and adjoints per horizon (`keys` matches the unit's
statistics), plus the per-transition design scratch. A non-causal unit carries an
empty one.
"""
struct _CausalScratch{T<:Real}
    keys::Vector{NTuple{2,Int}}
    P::Vector{Vector{Matrix{T}}}
    W::Vector{Vector{Matrix{T}}}
    G::Vector{Vector{Matrix{T}}}
    Pbar::Vector{Vector{Matrix{T}}}
    Wbar::Vector{Vector{Matrix{T}}}
    Gbar::Vector{Vector{Matrix{T}}}
    L::Matrix{T}
    Th::Matrix{T}
    Lbar::Matrix{T}
    Thbar::Matrix{T}
    dd::Matrix{T}
    dr::Matrix{T}
    Ct::Matrix{T}
    Ctbar::Matrix{T}
    Y::Matrix{T}
    V::Matrix{T}
    Vbar::Matrix{T}
    K::Matrix{T}
    Kbar::Matrix{T}
    nn::Matrix{T}
    nn2::Matrix{T}
    Imat::Matrix{T}
    sweep::_SweepBuffers{T}
end

function _CausalScratch(
    ::Type{T}, keys::Vector{NTuple{2,Int}}, n::Int, m::Int, K::Int=1
) where {T}
    d = 2n
    reg = d + 1 + m
    per(f) = [[f() for _ in 1:key[2]] for key in keys]
    nn() = zeros(T, n, n)
    nu() = zeros(T, n, 1 + m)
    return _CausalScratch{T}(
        copy(keys),
        per(nn),
        per(nn),
        per(nu),
        per(nn),
        per(nn),
        per(nu),
        zeros(T, d, d),
        zeros(T, d, reg),
        zeros(T, d, d),
        zeros(T, d, reg),
        zeros(T, d, d),
        zeros(T, d, reg),
        nu(),
        nu(),
        nu(),
        nu(),
        nu(),
        nn(),
        nn(),
        nn(),
        nn(),
        Matrix{T}(I, n, n),
        _SweepBuffers(T, n, m, K),
    )
end

_CausalScratch(::Type{T}) where {T} = _CausalScratch(T, NTuple{2,Int}[], 0, 0, 0)

"""
    _causal_sweep_scratch!(sc, A, S, Qs, h, Bu, Gref, hf, sm) -> Bool

Run every horizon's sweep at a trial point into `sc`; `false` if any is not
finite (an infeasible point for the optimizer). `C̃` is left in `sc.Ct`.
"""
function _causal_sweep_scratch!(
    sc::_CausalScratch{T}, A, S, Qs, hv, Bu, Gref, hf, sm::LQRStateModel
) where {T<:Real}
    n = size(A, 1)
    m = size(Bu, 2)
    @views sc.Ct[:, 1] .= hv[1:n]
    m > 0 && @views sc.Ct[:, 2:end] .= Bu[1:n, :]
    for (h, key) in enumerate(sc.keys)
        _causal_sweep!(
            sc.P[h], sc.W[h], sc.G[h], A, S, Qs, hv, Bu, Gref, hf, sm, key..., sc.sweep
        ) || return false
    end
    return true
end

"""
    _causal_design!(sc, h, t, A, S, slack)

`L_t` and `Θ_t` of transition `t` of horizon `h` into `sc.L`, `sc.Th`.
"""
function _causal_design!(
    sc::_CausalScratch{T}, h::Int, t::Int, A, S, slack::Bool
) where {T<:Real}
    s = t + 1
    return _causal_design!(sc, sc.W[h][s], sc.P[h][s], sc.G[h][s], A, S, slack)
end

function _causal_design!(
    sc::_CausalScratch{T}, Ws, Ps, Gs, A, S, slack::Bool
) where {T<:Real}
    n = size(A, 1)
    d = 2n
    xr, lr = 1:n, (n + 1):d
    ur = (d + 1):size(sc.Th, 2)
    L, Th = sc.L, sc.Th
    fill!(L, zero(T))
    fill!(Th, zero(T))
    @views begin
        if slack
            L[xr, xr] .= Ws
            mul!(L[xr, lr], Ws, S)
            mul!(Th[xr, ur], Ws, sc.Ct)
        else
            for i in 1:n
                L[i, i] = one(T)
            end
            copyto!(sc.Y, sc.Ct)
            mul!(sc.Y, S, Gs, -one(T), one(T))
            mul!(Th[xr, ur], Ws, sc.Y)
        end
        L[lr, xr] .= .-Ps
        for i in 1:n
            L[n + i, n + i] = one(T)
        end
        mul!(Th[xr, xr], Ws, A)
        Th[lr, ur] .= Gs
    end
    return nothing
end

"""
    _causal_add_scatter!(R, sc, hs, A, S, slack)

Add every transition's residual scatter of the unit into `R`.
"""
function _causal_add_scatter!(
    R::AbstractMatrix{T},
    sc::_CausalScratch{T},
    hs,
    A,
    S,
    slack::Bool,
    Wv=sc.W,
    Pv=sc.P,
    Gv=sc.G,
) where {T<:Real}
    #= `Wv[h]`, `Pv[h]`, `Gv[h]` are horizon `h`'s sweep, in the order of
    `hs.causal_keys` — the scratch's own (built in that order) in the M-step, the
    cache's in the ELBO. A horizon no trial reached adds nothing and is skipped. =#
    for (h, key) in enumerate(hs.causal_keys)
        hs.causal_n[h] > zero(T) || continue
        for t in 1:(key[2] - 1)
            s = t + 1
            _causal_design!(sc, Wv[h][s], Pv[h][s], Gv[h][s], A, S, slack)
            zz, zy, yy = hs.causal_zz[h][t], hs.causal_zy[h][t], hs.causal_yy[h][t]
            # R += L Y Lᵀ − (L Zyᵀ Θᵀ + its transpose) + Θ Z Θᵀ
            mul!(sc.dd, sc.L, yy)
            mul!(R, sc.dd, transpose(sc.L), one(T), one(T))
            mul!(sc.dr, sc.L, transpose(zy))
            mul!(sc.dd, sc.dr, transpose(sc.Th))
            R .-= sc.dd
            R .-= transpose(sc.dd)
            mul!(sc.dr, sc.Th, zz)
            mul!(R, sc.dr, transpose(sc.Th), one(T), one(T))
        end
    end
    return R
end

"""
    _causal_unit_gradient!(sc, hs, Wq, A, S, Qs, Gref, gate, sm, dA, dS, dQ, dh, dB, dG, dhf)

Accumulate the causal unit's gradient into the matrix-coordinate buffers, with
`Wq` the residual weight `𝒲` of its noise version. See the block comment above
for the chain.
"""
function _causal_unit_gradient!(
    sc::_CausalScratch{T},
    hs,
    Wq::AbstractMatrix{T},
    A::AbstractMatrix{T},
    S::AbstractMatrix{T},
    Qs::AbstractVector{<:AbstractMatrix{T}},
    Gref::AbstractMatrix{T},
    sm::LQRStateModel,
    dA::AbstractMatrix{T},
    dS::AbstractMatrix{T},
    dQ::AbstractVector{<:AbstractMatrix{T}},
    dh::AbstractVector{T},
    dB::AbstractMatrix{T},
    dG::AbstractMatrix{T},
    dhf::AbstractVector{T},
) where {T<:Real}
    n = size(A, 1)
    d = 2n
    m = size(sc.Ct, 2) - 1
    xr, lr = 1:n, (n + 1):d
    ur = (d + 1):(d + 1 + m)
    slack = sm.causal.slack_drives_state
    gate = sm.gref_gate
    fill!(sc.Ctbar, zero(T))
    for (h, key) in enumerate(sc.keys)
        off, Tn = key
        hh = h                  # the scratch was built in `hs.causal_keys` order
        hs.causal_n[h] > zero(T) || continue
        P, W, G = sc.P[h], sc.W[h], sc.G[h]
        Pbar, Wbar, Gbar = sc.Pbar[h], sc.Wbar[h], sc.Gbar[h]
        foreach(X -> fill!(X, zero(T)), Pbar)
        foreach(X -> fill!(X, zero(T)), Wbar)
        foreach(X -> fill!(X, zero(T)), Gbar)

        # Residual pullbacks, transition by transition.
        for t in 1:(Tn - 1)
            s = t + 1
            _causal_design!(sc, h, t, A, S, slack)
            zz, zy, yy = hs.causal_zz[hh][t], hs.causal_zy[hh][t], hs.causal_yy[hh][t]
            # L̄ = 𝒲 (L Y − Θ Zy),  Θ̄ = 𝒲 (Θ Z − L Zyᵀ)
            mul!(sc.dd, sc.L, yy)
            mul!(sc.dd, sc.Th, zy, -one(T), one(T))
            mul!(sc.Lbar, Wq, sc.dd)
            mul!(sc.dr, sc.Th, zz)
            mul!(sc.dr, sc.L, transpose(zy), -one(T), one(T))
            mul!(sc.Thbar, Wq, sc.dr)
            Ws, Gs = W[s], G[s]
            @views begin
                L11, L12, L21 = sc.Lbar[xr, xr], sc.Lbar[xr, lr], sc.Lbar[lr, xr]
                Txx, Txu, Tlu = sc.Thbar[xr, xr], sc.Thbar[xr, ur], sc.Thbar[lr, ur]
                if slack
                    # L₁₁ = W, L₁₂ = W S, Θ_xu = W C̃
                    Wbar[s] .+= L11
                    mul!(Wbar[s], L12, S, one(T), one(T))
                    mul!(dS, transpose(Ws), L12, one(T), one(T))
                    mul!(Wbar[s], Txu, transpose(sc.Ct), one(T), one(T))
                    mul!(sc.Ctbar, transpose(Ws), Txu, one(T), one(T))
                else
                    # Θ_xu = W (C̃ − S G)
                    copyto!(sc.Y, sc.Ct)
                    mul!(sc.Y, S, Gs, -one(T), one(T))
                    mul!(Wbar[s], Txu, transpose(sc.Y), one(T), one(T))
                    mul!(sc.V, transpose(Ws), Txu)          # Wᵀ Θ̄_xu
                    sc.Ctbar .+= sc.V
                    mul!(dS, sc.V, transpose(Gs), -one(T), one(T))
                    mul!(Gbar[s], S, sc.V, -one(T), one(T))
                end
                # L₂₁ = −P,  Θ_xx = W A,  Θ_λu = G
                Pbar[s] .-= L21
                mul!(Wbar[s], Txx, transpose(A), one(T), one(T))
                mul!(dA, transpose(Ws), Txx, one(T), one(T))
                Gbar[s] .+= Tlu
            end
        end

        # The sweeps in reverse.
        for s in 2:Tn
            Ws, Ps = W[s], P[s]
            # W_s = (I + S P_s)⁻¹: M̄ = −Wᵀ W̄ Wᵀ (held negated in nn2)
            mul!(sc.nn, transpose(Ws), Wbar[s])
            mul!(sc.nn2, sc.nn, transpose(Ws))
            mul!(dS, sc.nn2, Ps, -one(T), one(T))
            mul!(Pbar[s], S, sc.nn2, -one(T), one(T))
            if s < Tn
                k = _causal_regime(sm, off, s)
                W1, P1, G1 = W[s + 1], P[s + 1], G[s + 1]
                mul!(sc.K, P1, W1)
                # P_s = Q_k + Aᵀ K A
                dQ[k] .+= Pbar[s]
                mul!(sc.nn, sc.K, A)
                mul!(dA, sc.nn, transpose(Pbar[s]), one(T), one(T))
                mul!(sc.nn, transpose(sc.K), A)
                mul!(dA, sc.nn, Pbar[s], one(T), one(T))
                mul!(sc.nn, A, Pbar[s])
                mul!(sc.Kbar, sc.nn, transpose(A))
                # G_s = Aᵀ V + F̃_s,  V = K C̃ + W_{s+1}ᵀ G_{s+1}
                mul!(sc.V, sc.K, sc.Ct)
                mul!(sc.V, transpose(W1), G1, one(T), one(T))
                mul!(dA, sc.V, transpose(Gbar[s]), one(T), one(T))
                mul!(sc.Vbar, A, Gbar[s])
                mul!(sc.Kbar, sc.Vbar, transpose(sc.Ct), one(T), one(T))
                mul!(sc.Ctbar, transpose(sc.K), sc.Vbar, one(T), one(T))
                mul!(Wbar[s + 1], G1, transpose(sc.Vbar), one(T), one(T))
                mul!(Gbar[s + 1], W1, sc.Vbar, one(T), one(T))
                # F̃_s = [h_λ  B_{u,λ} − Q_k G_k]
                @views dh[lr] .+= Gbar[s][:, 1]
                if m > 0
                    @views begin
                        Gu = Gbar[s][:, 2:end]
                        dB[lr, :] .+= Gu
                        mul!(dQ[k], Gu, transpose(_gated(Gref, gate, k)), -one(T), one(T))
                        mul!(dG, Qs[k], _gated(Gu, gate, k), -one(T), one(T))
                    end
                end
                # K = P_{s+1} W_{s+1}
                mul!(Pbar[s + 1], sc.Kbar, transpose(W1), one(T), one(T))
                mul!(Wbar[s + 1], transpose(P1), sc.Kbar, one(T), one(T))
            elseif sm.causal.terminal_cost
                kT = _causal_terminal_regime(sm, off, Tn)
                dQ[kT] .+= Pbar[s]
                @views dhf .+= Gbar[s][:, 1]
                if m > 0
                    @views begin
                        Gu = Gbar[s][:, 2:end]
                        mul!(dQ[kT], Gu, transpose(_gated(Gref, gate, kT)), -one(T), one(T))
                        mul!(dG, Qs[kT], _gated(Gu, gate, kT), -one(T), one(T))
                    end
                end
            end
        end
    end
    # C̃ = [h_x  B_{u,x}]
    @views dh[xr] .+= sc.Ctbar[:, 1]
    m > 0 && @views dB[xr, :] .+= sc.Ctbar[:, 2:end]
    return nothing
end

# ============================================================================
# Noise: the profiled objective, its weight, and the closed-form update
# ============================================================================

"""
    _causal_noise_blocks(sm) -> NTuple{2,NamedTuple}

The two noise blocks of a causal model — plant (`1:n`) and costate (`n+1:2n`) —
with their form, the inverse-Wishart prior's marginal on each (`Ψ_b`, `ν_b`), and
the pinned costate variance, if any.
"""
function _causal_noise_blocks(sm::LQRStateModel{T}) where {T<:Real}
    n = _plant_dim(sm)
    d = 2n
    pr = sm.Σ_prior
    function block(r, form, fixed)
        if pr === nothing
            return (; r, form, fixed, Ψ=nothing, ν=zero(T))
        end
        # Marginal of IW(Ψ, ν) on a q-dimensional block: IW(Ψ_bb, ν − (2n − q)).
        q = form === :dense ? n : 1
        return (; r, form, fixed, Ψ=Matrix{T}(pr.Ψ[r, r]), ν=T(pr.ν) - T(d - q))
    end
    return (
        block(1:n, sm.causal.plant_noise, nothing),
        block((n + 1):d, sm.causal.costate_noise, sm.fixed_costate_sigma),
    )
end

"""
    _causal_noise_objective!(Wq, R, N, sm, profile, Sinv) -> T

The causal noise version's share of the structural objective, and its residual
weight `𝒲` (block-diagonal) into `Wq`: per block, `½ N_eff log det(Ψ_b + R_bb)`
for a dense profiled block (`N_eff = N`, or `ν_b + N + q + 1` with a prior), the
same entrywise for a diagonal one, `½ tr(R_λλ)/v` for a pinned costate block, and
`½ tr(Σ_b⁻¹ R_bb)` when the noise is held fixed. `Inf` when a block is not
positive definite.
"""
function _causal_noise_objective!(
    Wq::AbstractMatrix{T},
    R::AbstractMatrix{T},
    N::T,
    sm::LQRStateModel,
    profile::Bool,
    Sinv::AbstractMatrix{T},
) where {T<:Real}
    fill!(Wq, zero(T))
    fval = zero(T)
    for b in _causal_noise_blocks(sm)
        r = b.r
        Rb = view(R, r, r)
        if !profile
            @views Wq[r, r] .= Sinv[r, r]
            fval += T(0.5) * dot(view(Sinv, r, r), Rb)
        elseif b.fixed !== nothing
            v = T(b.fixed)
            for i in r
                Wq[i, i] = one(T) / v
            end
            fval += T(0.5) * tr(Rb) / v
        elseif b.form === :dense
            q = length(r)
            M = Matrix{T}(Rb)
            b.Ψ === nothing || (M .+= b.Ψ)
            Neff = b.Ψ === nothing ? N : b.ν + N + T(q) + one(T)
            chol = cholesky(Symmetric(M); check=false)
            issuccess(chol) || return T(Inf)
            fval += T(0.5) * Neff * logdet(chol)
            @views Wq[r, r] .= Neff .* inv(chol)
        else
            for (j, i) in enumerate(r)
                v = R[i, i] + (b.Ψ === nothing ? zero(T) : b.Ψ[j, j])
                v > zero(T) || return T(Inf)
                Neff = b.Ψ === nothing ? N : b.ν + N + T(2)
                fval += T(0.5) * Neff * log(v)
                Wq[i, i] = Neff / v
            end
        end
    end
    return fval
end

"""
    _causal_noise_update!(sm, R, N)

Write the noise maximizer given the structure into `sm.Σ = blkdiag(Σ, Ω)` — the
same per-block formulas the profiled objective assumed, so the two agree.
"""
function _causal_noise_update!(
    sm::LQRStateModel{T}, R::AbstractMatrix{T}, N::T
) where {T<:Real}
    N > zero(T) || return sm
    n = _plant_dim(sm)
    Σ = sm.Σ
    @views fill!(Σ[1:n, (n + 1):(2n)], zero(T))
    @views fill!(Σ[(n + 1):(2n), 1:n], zero(T))
    for b in _causal_noise_blocks(sm)
        r = b.r
        q = length(r)
        if b.fixed !== nothing
            @views Σ[r, r] .= Matrix{T}(T(b.fixed) * I, q, q)
        elseif b.form === :dense
            M = Matrix{T}(R[r, r])
            Neff = N
            if b.Ψ !== nothing
                M .+= b.Ψ
                Neff = b.ν + N + T(q) + one(T)
            end
            @views Σ[r, r] .= Symmetrize!(M ./ Neff)
        else
            @views fill!(Σ[r, r], zero(T))
            for (j, i) in enumerate(r)
                v = R[i, i] + (b.Ψ === nothing ? zero(T) : b.Ψ[j, j])
                Neff = b.Ψ === nothing ? N : b.ν + N + T(2)
                Σ[i, i] = v / Neff
            end
        end
    end
    return sm
end

"""
    _causal_noise_logprior(sm) -> T

The log-density of the noise under the block-marginal priors the M-step uses
(see [`causal_state_model`](@ref)'s `Σ_prior`), so the reported bound is the
objective EM is monotone in.
"""
function _causal_noise_logprior(sm::LQRStateModel{T}) where {T<:Real}
    sm.Σ_prior === nothing && return zero(T)
    total = zero(T)
    for b in _causal_noise_blocks(sm)
        b.fixed === nothing || continue
        r = b.r
        if b.form === :dense
            total += iw_logprior_term(Matrix{T}(sm.Σ[r, r]), IWPrior(b.Ψ, b.ν))
        else
            for (j, i) in enumerate(r)
                total += iw_logprior_term(
                    fill(T(sm.Σ[i, i]), 1, 1), IWPrior(fill(b.Ψ[j, j], 1, 1), b.ν)
                )
            end
        end
    end
    return total
end

"""
    _causal_Q_transition(sm, hs) -> T

The transition half of the state Q-term of a `:causal` model at its current
parameters, from the per-horizon statistics and the cache's own sweeps (the
ones the smoother used): `−½[N (2n log 2π + log det blkdiag(Σ, Ω)) + tr(B⁻¹ R)]`
with `R` the scatter of the innovations — no Jacobian term, since every
`L_t` has unit determinant.
"""
function _causal_Q_transition(sm::LQRStateModel{T}, hs) where {T<:Real}
    isempty(hs.causal_keys) && return zero(T)
    n = _plant_dim(sm)
    d = 2n
    m = size(sm.Bu, 2)
    c = sm.cache
    sc = _CausalScratch(T, NTuple{2,Int}[], n, m)       # design buffers only
    @views sc.Ct[:, 1] .= sm.h[1:n]
    m > 0 && @views sc.Ct[:, 2:end] .= sm.Bu[1:n, :]
    nh = length(hs.causal_keys)
    Wv = Vector{Vector{Matrix{T}}}(undef, nh)
    Pv = Vector{Vector{Matrix{T}}}(undef, nh)
    Gv = Vector{Vector{Matrix{T}}}(undef, nh)
    N = zero(T)
    for (h, key) in enumerate(hs.causal_keys)
        hs.causal_n[h] > zero(T) || continue
        slot = get(c.causal_index, key, 0)
        slot == 0 && _causal_missing_horizon(_with_cost_offset(sm, key[1]), key[2])
        H = c.causal[slot]
        Wv[h], Pv[h], Gv[h] = H.W, H.P, H.G
        N += hs.causal_n[h]
    end
    N > zero(T) || return zero(T)
    R = zeros(T, d, d)
    _causal_add_scatter!(R, sc, hs, sm.A, sm.S, sm.causal.slack_drives_state, Wv, Pv, Gv)
    Symmetrize!(R)
    B = c.Qfwd
    return T(-0.5) * (N * (T(d) * log(T(2π)) + logdet(B)) + tr(B \ R))
end
