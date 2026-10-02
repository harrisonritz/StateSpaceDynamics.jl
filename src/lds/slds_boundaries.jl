#=============================================================================
Boundaries between an SLDS's discrete states: exit bridges and entry priors.

A switching model of a task with stages — control toward a target, hold there,
control toward the next — has two things a plain SLDS does not say about the
moment the chain switches:

  * **Exit bridges.** A finite-horizon control segment plans to arrive. In the
    inverse-LQR latent that is the terminal factor `λ_T = Q_f (x_T − r_T) + h_f`,
    which a single `LQRStateModel` applies at the end of the trial. In a
    switching model the segment ends when the chain *leaves* the state, so a
    bridged state applies the same factor wherever its segment ends: at every
    exit `s_t = k → s_{t+1} ≠ k`, and at the end of the trial if the trial ends
    in it (the ordinary terminal factor, unchanged).

  * **Entry priors.** A control segment starts from a fresh plan. Without one,
    entering state `j` carries the previous state's costate through `j`'s
    adjoint equation, which says nothing about how a plan begins. With an
    `EntryPrior`, the transition into the first bin of a `j` segment entered
    from `i` is replaced by

        λ_t ~ N(μ_j + K_j (x_{t-1} − r⁽ⁱ⁾_{t-1}), P_j),
        x_t = A x_{t-1} − S λ_t + h_x + B_{u,x} u_{t-1} + ε_x,   ε_x ~ N(0, Σ_xx),

    with `r⁽ⁱ⁾ = G_r⁽ⁱ⁾ u` the reference of the state being left. The plant row
    is `j`'s own (its `A`, `S`, plant drift and plant noise), driven by the new
    plan, so the neural state stays continuous; only the costate is re-drawn.

The bridge is a factor on `(s_t, s_{t+1}, z_t)`. It enters the two halves of the
variational E-step the way any such factor does:

  * the discrete side sees `E_q(z)[log f_k(z_t)]` as a potential on every
    transition out of `k` at `t` (`SLDSDiscreteLayer`'s time-indexed
    `transition_matrix`), so forward-backward returns the exact `q(s)` of the
    chain with the factor attached, `ξ` and the chain entropy included;
  * the continuous side sees `log f_k(z_t)` weighted by
    `q(s_t = k, s_{t+1} ≠ k) = γ_k(t) − ξ_t(k, k)` — the exit weight — in the
    Laplace smoother, the ELBO and the M-step's terminal statistics, exactly as
    the end-of-trial factor is weighted by `γ_k(T)`.

The bridge reuses the state's terminal parameters (`Qc[k_f]`, `Σf`, `hf`,
`Gref`), so the M-step needs no new parameters: exits simply add terminal
statistics, weighted by their exit probability.

An entry is a different transition density on `(s_{t-1}, s_t, z_{t-1}, z_t)`,
and enters the same way: the discrete side sees
`E_q[log p_entry(i→j) − log p_j]` as a potential on the `i → j` transition, and
the continuous side swaps the ordinary transition for the entry one with weight
`ξ_{t-1}(i, j)`. In the M-step the entry parameters are an exact weighted
regression of `λ_t` on `[1; x_{t-1} − r⁽ⁱ⁾]`; entry bins leave `j`'s ordinary
transition statistics; and their plant row — which shares `j`'s structure — is
honoured by accepting the structural step only if it does not lower the
complete-data objective with that row included (a generalized M-step).

Generative semantics. A bridge is a terminal pseudo-observation like the
end-of-trial factor, "the goal is reached here", applied at every exit. Under
`condition_terminal = false` the model is the joint over the chain, the latents
and those pseudo-observations, and the reported score is `log p(y, goals = 0)`.
Under `condition_terminal = true` the score is `log p(y | goals = 0)`: the joint
less `log p(goals = 0 | θ)`, estimated by the terminal probe (an E-step on an
observation-free copy of the model), which carries the bridges and the entry
priors like the data side does. Every exit and the trial end are conditioned on,
so the probe's normalizer depends on the switching times, and the chain is
fitted against it (see `_slqr_chain_mstep!`). An entry prior is part of the prior
the normalizer integrates over, so under conditioning its update is a proposal
judged against the normalizer as well (`_slds_conditional_entry_update!`).
=============================================================================#

"""
    set_boundaries!(slds; bridge_states=Int[], entry_states=Int[],
                    entry_gain=true, entry_cov=1.0) -> slds

Configure what happens at the switches between `slds`'s discrete states.

- `bridge_states`: the states whose segment ends with a **bridge** — their
  terminal factor `λ_t = Q_f (x_t − r_t) + h_f`, applied at every exit from the
  state (and, as before, at the end of a trial that ends in it). These are the
  finite-horizon *control* stages of a staged task; a state left out (a hold, a
  free state) is left without one. A bridged state must be an inverse-LQR state
  (`mode = :lqr`) built with `terminal = true` — the bridge *is* its terminal
  factor, with its cost `Qc[k_f]` (`terminal_regime`, else the last schedule
  entry), covariance `Σf`, offset `hf` and reference `Gref`.

- `entry_states`: the states that start their segment from an
  [`EntryPrior`](@ref) — a fresh plan — when the chain enters them from another
  state: `λ_t ~ N(μ + K (x_{t-1} − r⁽ⁱ⁾_{t-1}), P)`, `r⁽ⁱ⁾` the reference of the
  state being left (zero for one without a reference), while the plant moves
  under the entered state's own dynamics driven by that plan. Each must be an
  inverse-LQR control state (`mode = :lqr`). The prior starts at `μ = 0`,
  `K = 0`, `P = entry_cov · I` and is fitted by EM; `entry_gain = false` keeps
  `K` at zero, so only the costate's mean offset and spread are learned.

Either objective works: with `condition_terminal = false` the score is the joint
`log p(y, goals = 0)`, and with `condition_terminal = true` (on every inverse-LQR
state with a terminal factor) it is `log p(y | goals = 0)`, the goals being every
exit's bridge and the trial's end.

Passing no states clears the configuration. Returns `slds`.

# Example
```julia
# control → hold → control → hold, on a banded chain
slds.A, slds.πₖ = banded_transition(4; stay=median_dwell_stay.([10, 25, 12]))
set_boundaries!(slds; bridge_states=[1, 3], entry_states=[3])
```
"""
function set_boundaries!(
    slds::SLDS{T};
    bridge_states::AbstractVector{<:Integer}=Int[],
    entry_states::AbstractVector{<:Integer}=Int[],
    entry_gain::Bool=true,
    entry_cov::Real=1.0,
) where {T<:Real}
    K = length(slds.LDSs)
    bridge = falses(K)
    entry = falses(K)
    for (name, states, mask) in
        (("bridge_states", bridge_states, bridge), ("entry_states", entry_states, entry))
        for k in states
            1 <= k <= K ||
                throw(ArgumentError("$name names state $k; the model has $K states"))
            mask[k] = true
        end
    end
    entry_cov > 0 || throw(ArgumentError("entry_cov must be positive, got $entry_cov"))
    if !any(bridge) && !any(entry)
        slds.boundaries = nothing
        return slds
    end
    entries = Union{Nothing,EntryPrior{T}}[nothing for _ in 1:K]
    for k in 1:K
        entry[k] || continue
        sm = slds.LDSs[k].state_model
        (sm isa LQRStateModel && sm.mode === :lqr) || throw(
            ArgumentError(
                "entry_states includes state $k, which is not an inverse-LQR control " *
                "state (`mode = :lqr`): an entry prior is a fresh plan's costate, " *
                "and only a control state has a plan.",
            ),
        )
        n = _plant_dim(sm)
        entries[k] = EntryPrior{T}(
            zeros(T, n), zeros(T, n, n), Matrix{T}(T(entry_cov) * I, n, n), entry_gain
        )
    end
    slds.boundaries = SLDSBoundaries{T}(collect(bridge), entries)
    _validate_boundaries(slds)
    return slds
end

"""Whether `slds` carries any entry prior."""
function _slds_has_entries(slds::SLDS)
    return slds.boundaries !== nothing && any(!isnothing, slds.boundaries.entry)
end

"""Whether `slds` carries any exit bridge."""
_slds_has_bridges(slds::SLDS) = slds.boundaries !== nothing && any(slds.boundaries.bridge)

"""
    _validate_boundaries(slds)

Refuse a boundary configuration the model cannot honour: a bridge on a state
with no terminal factor to apply, or on a non-LQR, `:free` or `:hold` state, and
an entry prior on anything but a control state.
"""
function _validate_boundaries(slds::SLDS)
    b = slds.boundaries
    b === nothing && return nothing
    K = length(slds.LDSs)
    (length(b.bridge) == K && length(b.entry) == K) ||
        throw(DimensionMismatchError("boundaries (states)", K, length(b.bridge)))
    for k in 1:K
        b.bridge[k] || continue
        sm = slds.LDSs[k].state_model
        (sm isa LQRStateModel && sm.mode === :lqr) || throw(
            ArgumentError(
                "bridge_states includes state $k, which is not an inverse-LQR " *
                "control state (`mode = :lqr`): a bridge is the state's terminal " *
                "factor, and only a finite-horizon control state has one.",
            ),
        )
        sm.terminal || throw(
            ArgumentError(
                "bridge_states includes state $k, whose model has no terminal factor. " *
                "Build it with `terminal = true`: the bridge applies that factor " *
                "(cost `Qc[k_f]`, `Σf`, `hf`, `Gref`) at every exit from the state.",
            ),
        )
    end
    for k in 1:K
        ep = b.entry[k]
        ep === nothing && continue
        sm = slds.LDSs[k].state_model
        (sm isa LQRStateModel && sm.mode === :lqr) || throw(
            ArgumentError(
                "state $k carries an entry prior but is not a `:lqr` control state"
            ),
        )
        n = _plant_dim(sm)
        (length(ep.μ) == n && size(ep.K) == (n, n) && size(ep.P) == (n, n)) || throw(
            DimensionMismatchError("entry prior of state $k (plant dim)", n, length(ep.μ)),
        )
        isposdef(Symmetric(Matrix(ep.P))) || throw(
            ArgumentError("the entry prior of state $k has a non-positive-definite P")
        )
    end
    return nothing
end

"""
    _slds_discrete_layer(slds, total_T) -> SLDSDiscreteLayer

The discrete layer for a pass over `total_T` bins, carrying the exit-bridge
buffers when `slds` has bridges.
"""
function _slds_discrete_layer(slds::SLDS{T}, total_T::Int) where {T<:Real}
    K = length(slds.LDSs)
    dl = SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(T, K, total_T))
    if _slds_has_bridges(slds)
        dl.bridge = copy(slds.boundaries.bridge)
        dl.exit_logL = zeros(T, K, total_T)
        dl.exit_w = zeros(T, K, total_T)
    end
    if _slds_has_entries(slds)
        dl.entry = [ep !== nothing for ep in slds.boundaries.entry]
        dl.entry_logL = zeros(T, K, K, total_T)
        dl.entry_w = zeros(T, K, K, total_T)
    end
    return dl
end

"""
    _bridge_regime(sm) -> Int

The cost a bridged state's exit factor is written against: `terminal_regime` when
pinned, else the last schedule entry — the terminal cost of the
`[running…, terminal]` schedules an SLDS member carries. Unlike
`_terminal_regime`, this does not depend on any trial's length: an exit can fall
anywhere in the trial.
"""
function _bridge_regime(sm::LQRStateModel)
    sm.terminal_regime > 0 && return sm.terminal_regime
    return isempty(sm.schedule) ? 1 : last(sm.schedule)
end

"""
    _bridge_residual!(out, sm, x, t, ux) -> out

The bridge residual at timestep `t`: `Λf z_t + Q_f G_r u_t − h_f`, the terminal
residual of [`_terminal_residual!`](@ref) evaluated at an exit instead of at the
end of the trial.
"""
function _bridge_residual!(
    out::AbstractVector{T},
    sm::LQRStateModel,
    x::AbstractMatrix{T},
    t::Int,
    ux::Union{Nothing,AbstractMatrix},
) where {T<:Real}
    c = sm.cache
    kf = _bridge_regime(sm)
    @views mul!(out, c.Lf[kf], x[:, t])
    if ux !== nothing && size(c.Ftrm[kf], 2) > 0
        @views mul!(out, c.Ftrm[kf], ux[:, t], one(T), one(T))
    end
    out .-= sm.hf
    return out
end

"""`log f(z_t)`, the bridge factor's log-density at `t` (residual scratch `rf`)."""
function _bridge_loglik(
    sm::LQRStateModel,
    x::AbstractMatrix{T},
    t::Int,
    ux::Union{Nothing,AbstractMatrix},
    rf::AbstractVector{T},
) where {T<:Real}
    _bridge_residual!(rf, sm, x, t, ux)
    _whiten!(sm.cache.Sf_PD.chol, rf)
    return T(sm.cache.cF) - T(0.5) * sum(abs2, rf)
end

"""
    _slds_exit_weights!(exit_w, fb_storage, seq_ends, bridge) -> exit_w

`exit_w[k, t] = q(s_t = k, s_{t+1} ≠ k) = γ_k(t) − ξ_t(k, k)` for every bridged
state `k` and every step that has a successor, zero elsewhere — in particular at a
trial's last bin, where the end-of-trial terminal factor (weighted by `γ_k(T)`)
takes over. Clamped at zero against roundoff.
"""
function _slds_exit_weights!(
    exit_w::AbstractMatrix{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    seq_ends::AbstractVector{Int},
    bridge::AbstractVector{Bool},
) where {T<:Real}
    fill!(exit_w, zero(T))
    γ = fb_storage.γ
    for trial in eachindex(seq_ends)
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        for t in t1:(t2 - 1)
            ξt = fb_storage.ξ[t]
            for k in eachindex(bridge)
                bridge[k] || continue
                exit_w[k, t] = max(zero(T), T(γ[k, t] - ξt[k, k]))
            end
        end
    end
    return exit_w
end

#=
The continuous half. Each takes the trial's `K × T` exit weights (or `nothing`,
the no-bridge fast path) and adds `Σ_t e_k(t) · {log f_k, ∇ log f_k, ∇² log f_k}`
for every bridged state. Only an LQR state with a terminal factor can carry
weight here — `_validate_boundaries` sees to that — so the other state models
fall through to no-ops.
=#

@inline _slds_exit_loglik!(_, ::SLDSSmoothWorkspace, ::SLDS, _, ::Nothing, _) = nothing

function _slds_exit_loglik!(
    ll::AbstractVector{T},
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    ew::AbstractMatrix{T},
    ux::Union{Nothing,AbstractMatrix},
) where {T<:Real}
    for (k, lds) in enumerate(slds.LDSs)
        _slds_exit_loglik_state!(ll, ws, lds.state_model, x, view(ew, k, :), ux)
    end
    return nothing
end

@inline _slds_exit_loglik_state!(_, _, ::AbstractStateModel, _, _, _) = nothing

function _slds_exit_loglik_state!(
    ll::AbstractVector{T},
    ws::SLDSSmoothWorkspace{T},
    sm::LQRStateModel,
    x::AbstractMatrix{T},
    e::AbstractVector{T},
    ux::Union{Nothing,AbstractMatrix},
) where {T<:Real}
    sm.terminal || return nothing
    rf = view(ws.opt.dxt, 1:_plant_dim(sm))
    for t in 1:(size(x, 2) - 1)
        et = e[t]
        iszero(et) && continue
        ll[t] += et * _bridge_loglik(sm, x, t, ux, rf)
    end
    return nothing
end

@inline _slds_exit_gradient!(_, ::SLDSSmoothWorkspace, ::SLDS, _, ::Nothing, _) = nothing

function _slds_exit_gradient!(
    grad::AbstractMatrix{T},
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    ew::AbstractMatrix{T},
    ux::Union{Nothing,AbstractMatrix},
) where {T<:Real}
    for (k, lds) in enumerate(slds.LDSs)
        _slds_exit_gradient_state!(grad, ws, lds.state_model, x, view(ew, k, :), ux)
    end
    return nothing
end

@inline _slds_exit_gradient_state!(_, _, ::AbstractStateModel, _, _, _) = nothing

function _slds_exit_gradient_state!(
    grad::AbstractMatrix{T},
    ws::SLDSSmoothWorkspace{T},
    sm::LQRStateModel,
    x::AbstractMatrix{T},
    e::AbstractVector{T},
    ux::Union{Nothing,AbstractMatrix},
) where {T<:Real}
    sm.terminal || return nothing
    rf = view(ws.opt.dxt, 1:_plant_dim(sm))
    LtSinv = sm.cache.LtSinv[_bridge_regime(sm)]
    for t in 1:(size(x, 2) - 1)
        et = e[t]
        iszero(et) && continue
        _bridge_residual!(rf, sm, x, t, ux)
        # ∇ log f = −Λfᵀ Σf⁻¹ r
        @views mul!(grad[:, t], LtSinv, rf, -et, one(T))
    end
    return nothing
end

@inline _slds_exit_hessian!(_, ::SLDS, ::Nothing, ::Int) = nothing

function _slds_exit_hessian!(
    H_diag::AbstractVector, slds::SLDS{T}, ew::AbstractMatrix{T}, tsteps::Int
) where {T<:Real}
    for (k, lds) in enumerate(slds.LDSs)
        _slds_exit_hessian_state!(H_diag, lds.state_model, view(ew, k, :), tsteps)
    end
    return nothing
end

@inline _slds_exit_hessian_state!(_, ::AbstractStateModel, _, ::Int) = nothing

function _slds_exit_hessian_state!(
    H_diag::AbstractVector, sm::LQRStateModel{T}, e::AbstractVector{T}, tsteps::Int
) where {T<:Real}
    sm.terminal || return nothing
    negLtSL = sm.cache.negLtSL[_bridge_regime(sm)]
    for t in 1:(tsteps - 1)
        et = e[t]
        iszero(et) && continue
        @. H_diag[t] += et * negLtSL
    end
    return nothing
end

"""
    _slds_exit_potentials!(out, ws, lds, x, ux, fs)

One state's bridge potentials over a trial: `out[t] = E_q[log f(z_t)]` for every
`t` with a successor, the plug-in at `x` plus — when `fs` is given — the
`½ tr(H_f Σ_t)` term that turns it into an expectation under the Laplace
posterior (the same correction `_add_cov_correction!` applies to the per-state
log-densities). `out[T]` is zero: the last bin has no exit.
"""
function _slds_exit_potentials!(
    out::AbstractVector{T},
    ws::SLDSSmoothWorkspace{T},
    lds::LinearDynamicalSystem,
    x::AbstractMatrix{T},
    ux::Union{Nothing,AbstractMatrix},
    fs::Union{Nothing,FilterSmooth{T}},
) where {T<:Real}
    sm = lds.state_model
    fill!(out, zero(T))
    sm isa LQRStateModel || return out
    sm.terminal || return out
    rf = view(ws.opt.dxt, 1:_plant_dim(sm))
    negLtSL = sm.cache.negLtSL[_bridge_regime(sm)]
    for t in 1:(size(x, 2) - 1)
        out[t] = _bridge_loglik(sm, x, t, ux, rf)
        if fs !== nothing
            out[t] += T(0.5) * _tr_prod(negLtSL, view(fs.p_smooth, :, :, t))
        end
    end
    return out
end

# ============================================================================
# Entry priors
# ============================================================================

"""
    _state_reference!(out, sm, u) -> out

The reference `r = G_r u` a state regulates toward, at inputs `u`; zero for a
state without one (a `:free` or Gaussian state, or a model without inputs). An
entry prior centres the new plan on the state's offset from the *previous*
state's reference.
"""
_state_reference!(out::AbstractVector, ::AbstractStateModel, _) =
    fill!(out, zero(eltype(out)))

function _state_reference!(out::AbstractVector{T}, sm::LQRStateModel, u) where {T}
    if u === nothing || _is_free(sm) || size(sm.Gref, 2) == 0
        fill!(out, zero(T))
    else
        mul!(out, sm.Gref, u)
    end
    return out
end

"""
    _EntryTerms{T}

The pieces of state `j`'s entry density, factored once per pass: the plant row's
`A`, `S`, drift and input block and its noise precision `Σ_xx⁻¹`, the prior's
`μ`, `K` and precision `P⁻¹`, the constant, and the blocks of the (negative
definite) Hessian on `(z_{t-1}, z_t)`, which do not depend on where the state is.
"""
struct _EntryTerms{T<:Real}
    n::Int
    A::Matrix{T}
    S::Matrix{T}
    hx::Vector{T}
    Bux::Matrix{T}
    μ::Vector{T}
    K::Matrix{T}
    Pinv::Matrix{T}
    Sinv::Matrix{T}
    c::T
    H_prev::Matrix{T}   # ∂²/∂z_{t-1}² (2n × 2n), x block only
    H_cur::Matrix{T}    # ∂²/∂z_t²     (2n × 2n)
    H_sub::Matrix{T}    # ∂²/∂z_t ∂z_{t-1} (rows z_t, cols z_{t-1})
end

function _EntryTerms(sm::LQRStateModel{T}, ep::EntryPrior) where {T<:Real}
    n = _plant_dim(sm)
    d = 2n
    xr, lr = 1:n, (n + 1):d
    Pc = cholesky(Symmetric(Matrix{T}(ep.P)))
    Sc = cholesky(Symmetric(Matrix{T}(sm.Σ[xr, xr])))
    Pinv = Matrix(inv(Pc))
    Sinv = Matrix(inv(Sc))
    A = Matrix{T}(sm.A)
    S = Matrix{T}(sm.S)
    K = Matrix{T}(ep.K)
    c = -T(n) * log(T(2π)) - T(0.5) * (logdet(Pc) + logdet(Sc))
    H_prev = zeros(T, d, d)
    H_prev[xr, xr] .= .-(transpose(K) * Pinv * K) .- (transpose(A) * Sinv * A)
    H_cur = zeros(T, d, d)
    H_cur[xr, xr] .= .-Sinv
    H_cur[lr, lr] .= .-Pinv .- S * Sinv * S
    H_cur[xr, lr] .= .-(Sinv * S)
    H_cur[lr, xr] .= .-(S * Sinv)
    H_sub = zeros(T, d, d)
    H_sub[xr, xr] .= Sinv * A
    H_sub[lr, xr] .= Pinv * K .+ S * Sinv * A
    return _EntryTerms{T}(
        n,
        A,
        S,
        Vector{T}(sm.h[xr]),
        Matrix{T}(sm.Bu[xr, :]),
        Vector{T}(ep.μ),
        K,
        Pinv,
        Sinv,
        c,
        H_prev,
        H_cur,
        H_sub,
    )
end

"""
    _entry_residuals!(r1, r2, et, x, t, ux, rref)

The two residuals of the entry density at local step `t ≥ 2`:
`r1 = λ_t − μ − K (x_{t-1} − r)` (the new plan against its prior) and
`r2 = x_t − A x_{t-1} + S λ_t − h_x − B_{u,x} u_{t-1}` (the plant row).
"""
function _entry_residuals!(
    r1::AbstractVector{T},
    r2::AbstractVector{T},
    et::_EntryTerms{T},
    x::AbstractMatrix{T},
    t::Int,
    ux::Union{Nothing,AbstractMatrix},
    rref::AbstractVector{T},
) where {T<:Real}
    n = et.n
    xprev = view(x, 1:n, t - 1)
    λt = view(x, (n + 1):(2n), t)
    xt = view(x, 1:n, t)
    r2 .= xprev .- rref
    mul!(r1, et.K, r2)
    r1 .= λt .- et.μ .- r1
    mul!(r2, et.A, xprev)
    r2 .= xt .- r2 .- et.hx
    mul!(r2, et.S, λt, one(T), one(T))
    if ux !== nothing && size(et.Bux, 2) > 0
        mul!(r2, et.Bux, view(ux, :, t - 1), -one(T), one(T))
    end
    return nothing
end

function _entry_loglik(
    et::_EntryTerms{T}, r1::AbstractVector{T}, r2::AbstractVector{T}
) where {T<:Real}
    return et.c - T(0.5) * (dot(r1, et.Pinv, r1) + dot(r2, et.Sinv, r2))
end

"""The ordinary forward transition's log-density into local step `t` (no terminal)."""
function _ordinary_transition_loglik(
    lds::LinearDynamicalSystem,
    x::AbstractMatrix{T},
    t::Int,
    ux::Union{Nothing,AbstractMatrix},
    tmp::AbstractVector{T},
) where {T<:Real}
    c = lds.state_model.cache
    _transition_residual!(tmp, lds, x, t, ux)
    _whiten!(c.Qfwd.chol, tmp)
    return T(c.cQ) - T(0.5) * sum(abs2, tmp)
end

"""
    _slds_entry_weights!(entry_w, fb_storage, seq_ends, entry) -> entry_w

`entry_w[i, j, t] = q(s_{t-1} = i, s_t = j)` for every entry-prior state `j`, every
`i ≠ j` and every step with a predecessor in its trial; zero elsewhere.
"""
function _slds_entry_weights!(
    entry_w::AbstractArray{T,3},
    fb_storage::HMMs.ForwardBackwardStorage,
    seq_ends::AbstractVector{Int},
    entry::AbstractVector{Bool},
) where {T<:Real}
    fill!(entry_w, zero(T))
    K = length(entry)
    for trial in eachindex(seq_ends)
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        for t in (t1 + 1):t2
            ξ = fb_storage.ξ[t - 1]
            for j in 1:K
                entry[j] || continue
                for i in 1:K
                    i == j || (entry_w[i, j, t] = max(zero(T), T(ξ[i, j])))
                end
            end
        end
    end
    return entry_w
end

#=
The continuous half: for every entry-prior state `j`, every source `i ≠ j` and
every step `t ≥ 2`, add `pw[i, j, t] · (log p_entry(i→j) − log p_j)` — the swap
of `j`'s ordinary transition into `t` for the entry one — and its derivatives.
`pw` is the trial's `K × K × T` view of `entry_w`, `nothing` without entries.
=#

@inline _slds_entry_loglik!(_, ::SLDSSmoothWorkspace, ::SLDS, _, ::Nothing, _) = nothing

function _slds_entry_loglik!(
    ll::AbstractVector{T},
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    pw::AbstractArray{T,3},
    ux::Union{Nothing,AbstractMatrix},
) where {T<:Real}
    b = slds.boundaries
    b === nothing && return nothing
    K = length(slds.LDSs)
    d, tsteps = size(x)
    for j in 1:K
        ep = b.entry[j]
        ep === nothing && continue
        lds_j = slds.LDSs[j]
        et = _EntryTerms(lds_j.state_model, ep)
        n = et.n
        r1, r2, rref, tmp = zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, d)
        for t in 2:tsteps
            ord = T(NaN)
            for i in 1:K
                w = i == j ? zero(T) : pw[i, j, t]
                iszero(w) && continue
                isnan(ord) && (ord = _ordinary_transition_loglik(lds_j, x, t, ux, tmp))
                _state_reference!(
                    rref,
                    slds.LDSs[i].state_model,
                    ux === nothing ? nothing : view(ux, :, t - 1),
                )
                _entry_residuals!(r1, r2, et, x, t, ux, rref)
                ll[t] += w * (_entry_loglik(et, r1, r2) - ord)
            end
        end
    end
    return nothing
end

@inline _slds_entry_gradient!(_, ::SLDSSmoothWorkspace, ::SLDS, _, ::Nothing, _) = nothing

function _slds_entry_gradient!(
    grad::AbstractMatrix{T},
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    pw::AbstractArray{T,3},
    ux::Union{Nothing,AbstractMatrix},
) where {T<:Real}
    b = slds.boundaries
    b === nothing && return nothing
    K = length(slds.LDSs)
    d, tsteps = size(x)
    for j in 1:K
        ep = b.entry[j]
        ep === nothing && continue
        lds_j = slds.LDSs[j]
        sm = lds_j.state_model
        c = sm.cache
        et = _EntryTerms(sm, ep)
        n = et.n
        xr, lr = 1:n, (n + 1):d
        r1, r2, rref = zeros(T, n), zeros(T, n), zeros(T, n)
        g1, g2 = zeros(T, n), zeros(T, n)
        res, tmp = zeros(T, d), zeros(T, d)
        for t in 2:tsteps
            wsum = zero(T)
            for i in 1:K
                w = i == j ? zero(T) : pw[i, j, t]
                iszero(w) && continue
                wsum += w
                _state_reference!(
                    rref,
                    slds.LDSs[i].state_model,
                    ux === nothing ? nothing : view(ux, :, t - 1),
                )
                _entry_residuals!(r1, r2, et, x, t, ux, rref)
                mul!(g1, et.Pinv, r1)
                mul!(g2, et.Sinv, r2)
                # ∂/∂x_{t-1} = Kᵀ P⁻¹ r1 + Aᵀ Σ⁻¹ r2
                @views mul!(grad[xr, t - 1], transpose(et.K), g1, w, one(T))
                @views mul!(grad[xr, t - 1], transpose(et.A), g2, w, one(T))
                # ∂/∂λ_t = −P⁻¹ r1 − S Σ⁻¹ r2 ;  ∂/∂x_t = −Σ⁻¹ r2
                @views grad[lr, t] .-= w .* g1
                @views mul!(grad[lr, t], et.S, g2, -w, one(T))
                @views grad[xr, t] .-= w .* g2
            end
            iszero(wsum) && continue
            # Remove `j`'s ordinary transition with the same total weight.
            k = _regime(sm, t - 1)
            _transition_residual!(res, lds_j, x, t, ux)
            mul!(tmp, c.negQinv, res)                  # ∂/∂z_t = −Q⁻¹ r
            @views grad[:, t] .-= wsum .* tmp
            mul!(tmp, c.MtQinv[k], res)                # ∂/∂z_{t-1} = MᵀQ⁻¹ r
            @views grad[:, t - 1] .-= wsum .* tmp
        end
    end
    return nothing
end

@inline _slds_entry_hessian!(_, ::SLDS, ::Nothing, ::Int) = nothing

function _slds_entry_hessian!(
    btd, slds::SLDS{T}, pw::AbstractArray{T,3}, tsteps::Int
) where {T<:Real}
    b = slds.boundaries
    b === nothing && return nothing
    K = length(slds.LDSs)
    for j in 1:K
        ep = b.entry[j]
        ep === nothing && continue
        sm = slds.LDSs[j].state_model
        c = sm.cache
        et = _EntryTerms(sm, ep)
        for t in 2:tsteps
            wsum = zero(T)
            for i in 1:K
                i == j || (wsum += pw[i, j, t])
            end
            iszero(wsum) && continue
            k = _regime(sm, t - 1)
            @. btd.H_diag[t - 1] += wsum * (et.H_prev - c.negMtQinvM[k])
            @. btd.H_diag[t] += wsum * (et.H_cur - c.negQinv)
            @. btd.H_sub[t - 1] += wsum * (et.H_sub - c.QinvM[k])
            btd.H_super[t - 1] .+= wsum .* transpose(et.H_sub .- c.QinvM[k])
        end
    end
    return nothing
end

"""
    _slds_entry_potentials!(Φ, ws, slds, j, x, ux, fs)

State `j`'s entry potentials over a trial, `Φ[i, t] = E_q[log p_entry(i→j)(z_t |
z_{t-1}) − log p_j(z_t | z_{t-1})]` for `i ≠ j` and `t ≥ 2` (zero elsewhere): the
plug-in at `x`, plus — when `fs` is given — the `½ tr(ΔH Σ)` term over the pair
`(z_{t-1}, z_t)` that turns it into an expectation under the Laplace posterior.
"""
function _slds_entry_potentials!(
    Φ::AbstractMatrix{T},
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    j::Int,
    x::AbstractMatrix{T},
    ux::Union{Nothing,AbstractMatrix},
    fs::Union{Nothing,FilterSmooth{T}},
) where {T<:Real}
    fill!(Φ, zero(T))
    ep = slds.boundaries.entry[j]
    ep === nothing && return Φ
    K = length(slds.LDSs)
    d, tsteps = size(x)
    lds_j = slds.LDSs[j]
    sm = lds_j.state_model
    c = sm.cache
    et = _EntryTerms(sm, ep)
    n = et.n
    r1, r2, rref, tmp = zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, d)
    ΔHp, ΔHc, ΔHs = similar(et.H_prev), similar(et.H_cur), similar(et.H_sub)
    for t in 2:tsteps
        ord = _ordinary_transition_loglik(lds_j, x, t, ux, tmp)
        corr = zero(T)
        if fs !== nothing
            k = _regime(sm, t - 1)
            ΔHp .= et.H_prev .- c.negMtQinvM[k]
            ΔHc .= et.H_cur .- c.negQinv
            ΔHs .= et.H_sub .- c.QinvM[k]
            Σ_ttm1 = view(fs.p_smooth_tt1, :, :, t)  # Cov(z_t, z_{t-1})
            corr =
                T(0.5) * (
                    _tr_prod(ΔHp, view(fs.p_smooth, :, :, t - 1)) +
                    _tr_prod(ΔHc, view(fs.p_smooth, :, :, t)) +
                    _tr_prod(transpose(ΔHs), Σ_ttm1) +
                    _tr_prod(ΔHs, transpose(Σ_ttm1))
                )
        end
        for i in 1:K
            i == j && continue
            _state_reference!(
                rref,
                slds.LDSs[i].state_model,
                ux === nothing ? nothing : view(ux, :, t - 1),
            )
            _entry_residuals!(r1, r2, et, x, t, ux, rref)
            Φ[i, t] = _entry_loglik(et, r1, r2) - ord + corr
        end
    end
    return Φ
end

# ----------------------------------------------------------------------------
# M-step
# ----------------------------------------------------------------------------

"""
    _EntryStats{T}

Weighted second moments for one entry-prior state, accumulated over every entry
into it:

- the prior's regression of `λ_t` on `ψ = [1; x_{t-1} − r⁽ⁱ⁾]` (`Sψψ`, `Sλψ`,
  `Sλλ`, total weight `N`);
- the plant row's regression of `x_t` on `w̃ = [x_{t-1}; λ_t; 1; u_{t-1}]` (`Zx`,
  `Xx`, `Yx`, weight `N`), which shares the state's structural parameters.
"""
struct _EntryStats{T<:Real}
    Sψψ::Matrix{T}
    Sλψ::Matrix{T}
    Sλλ::Matrix{T}
    Zx::Matrix{T}
    Xx::Matrix{T}
    Yx::Matrix{T}
    N::Base.RefValue{T}
end

function _EntryStats(::Type{T}, n::Int, m::Int) where {T<:Real}
    reg = 2n + 1 + m
    return _EntryStats{T}(
        zeros(T, n + 1, n + 1),
        zeros(T, n, n + 1),
        zeros(T, n, n),
        zeros(T, reg, reg),
        zeros(T, n, reg),
        zeros(T, n, n),
        Ref(zero(T)),
    )
end

"""
    _entry_stats(slds_of, j, tfs, data, entry_w, seq_ends) -> _EntryStats

Accumulate state `j`'s entry statistics over every trial, each trial's source
references read off `slds_of(trial)` (its cell's model under `depends_on`).
"""
function _entry_stats(
    slds_of,
    j::Int,
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    entry_w::AbstractArray{T,3},
    seq_ends::AbstractVector{Int};
    trials=eachindex(seq_ends),
    trial_weight::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real}
    slds1 = slds_of(first(trials))
    K = length(slds1.LDSs)
    n = _plant_dim(slds1.LDSs[j].state_model)
    d = 2n
    m = size(data.ux[1], 1)
    st = _EntryStats(T, n, m)
    xr, lr = 1:n, (n + 1):d
    reg = d + 1 + m
    # Joint moments of ω = [x_{t-1}; λ_t; 1; u_{t-1}; x_t], then sliced.
    no = reg + n
    mω = zeros(T, no)
    Cω = zeros(T, no, no)
    rref = zeros(T, n)
    for trial in trials
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        fs = tfs[trial]
        x = fs.x_smooth
        ux = data.ux[trial]
        slds_t = slds_of(trial)
        # A design standing for several trials (the terminal probe) counts for each.
        scale = trial_weight === nothing ? one(T) : T(trial_weight[trial])
        for t in 2:(t2 - t1 + 1)
            g = t1 + t - 1
            wtot = zero(T)
            for i in 1:K
                i == j || (wtot += scale * entry_w[i, j, g])
            end
            iszero(wtot) && continue
            # Means and covariance of ω under q.
            fill!(Cω, zero(T))
            @views begin
                mω[1:n] .= x[xr, t - 1]
                mω[(n + 1):d] .= x[lr, t]
                mω[d + 1] = one(T)
                m > 0 && (mω[(d + 2):reg] .= ux[:, t - 1])
                mω[(reg + 1):no] .= x[xr, t]
                Pp = fs.p_smooth[:, :, t - 1]
                Pc = fs.p_smooth[:, :, t]
                Pc1 = fs.p_smooth_tt1[:, :, t]       # Cov(z_t, z_{t-1})
                Cω[1:n, 1:n] .= Pp[xr, xr]
                Cω[(n + 1):d, (n + 1):d] .= Pc[lr, lr]
                Cω[(n + 1):d, 1:n] .= Pc1[lr, xr]
                Cω[1:n, (n + 1):d] .= transpose(Pc1[lr, xr])
                Cω[(reg + 1):no, (reg + 1):no] .= Pc[xr, xr]
                Cω[(reg + 1):no, 1:n] .= Pc1[xr, xr]
                Cω[1:n, (reg + 1):no] .= transpose(Pc1[xr, xr])
                Cω[(reg + 1):no, (n + 1):d] .= Pc[xr, lr]
                Cω[(n + 1):d, (reg + 1):no] .= Pc[lr, xr]
            end
            Eω = Cω .+ mω * transpose(mω)
            ir = 1:reg
            ix = (reg + 1):no
            @views begin
                st.Zx .+= wtot .* Eω[ir, ir]
                st.Xx .+= wtot .* Eω[ix, ir]
                st.Yx .+= wtot .* Eω[ix, ix]
            end
            # The prior's regression, per source (its reference differs).
            for i in 1:K
                i == j && continue
                w = scale * entry_w[i, j, g]
                iszero(w) && continue
                _state_reference!(
                    rref, slds_t.LDSs[i].state_model, m > 0 ? view(ux, :, t - 1) : nothing
                )
                δ = mω[1:n] .- rref                  # E[x_{t-1} − r]
                λm = mω[(n + 1):d]
                Exx = Cω[1:n, 1:n] .+ δ * transpose(δ)
                Eλx = Cω[(n + 1):d, 1:n] .+ λm * transpose(δ)
                st.Sψψ[1, 1] += w
                @views st.Sψψ[1, 2:end] .+= w .* δ
                @views st.Sψψ[2:end, 1] .+= w .* δ
                @views st.Sψψ[2:end, 2:end] .+= w .* Exx
                @views st.Sλψ[:, 1] .+= w .* λm
                @views st.Sλψ[:, 2:end] .+= w .* Eλx
                st.Sλλ .+= w .* (Cω[(n + 1):d, (n + 1):d] .+ λm * transpose(λm))
            end
            st.N[] += wtot
        end
    end
    return st
end

"""
    _update_entry_prior!(ep, st)

The exact maximizer of the entry prior's expected log-density: weighted least
squares `[μ K] = Sλψ Sψψ⁻¹` (or `μ` alone with `K` held, when `fit_gain` is off),
then `P` the weighted residual covariance. Skipped when the entries carry
negligible weight; `P` is floored so a near-degenerate scatter stays positive
definite.
"""
function _update_entry_prior!(ep::EntryPrior{T}, st::_EntryStats{T}) where {T<:Real}
    N = st.N[]
    N > sqrt(eps(T)) || return ep
    n = length(ep.μ)
    Θ = if ep.fit_gain
        F = cholesky(Symmetric(st.Sψψ); check=false)
        issuccess(F) || return ep
        st.Sλψ / F
    else
        # μ alone: E_w[λ − K δ] / N, with K held.
        μ = (st.Sλψ[:, 1] .- ep.K * st.Sψψ[2:end, 1]) ./ N
        hcat(μ, ep.K)
    end
    R =
        st.Sλλ .- Θ * transpose(st.Sλψ) .- st.Sλψ * transpose(Θ) .+
        Θ * st.Sψψ * transpose(Θ)
    P = Matrix(Symmetric((R .+ transpose(R)) ./ (2N)))
    floor = max(sqrt(eps(T)), T(1e-8) * tr(P) / n)
    E = eigen(Symmetric(P))
    P = E.vectors * Diagonal(max.(E.values, floor)) * transpose(E.vectors)
    ep.μ .= Θ[:, 1]
    ep.K .= Θ[:, 2:end]
    ep.P .= (P .+ transpose(P)) ./ 2
    return ep
end

"""
    _entry_plant_Q(sm, st) -> T

The plant row's expected log-density over state `sm`'s entries, at its current
structure and noise: `−½[N (n log 2π + log det Σ_xx) + tr(Σ_xx⁻¹ R)]` with `R` the
residual scatter of `x_t ≈ [A −S h_x B_{u,x}] w̃`.
"""
function _entry_plant_Q(sm::LQRStateModel{T}, st::_EntryStats{T}) where {T<:Real}
    N = st.N[]
    iszero(N) && return zero(T)
    n = _plant_dim(sm)
    xr = 1:n
    Θ = hcat(Matrix{T}(sm.A), -Matrix{T}(sm.S), sm.h[xr], Matrix{T}(sm.Bu[xr, :]))
    R = st.Yx .- Θ * transpose(st.Xx) .- st.Xx * transpose(Θ) .+ Θ * st.Zx * transpose(Θ)
    Sc = cholesky(Symmetric(Matrix{T}(sm.Σ[xr, xr])); check=false)
    issuccess(Sc) || return -T(Inf)
    return -T(0.5) * (N * (T(n) * log(T(2π)) + logdet(Sc)) + tr(Sc \ R))
end

"""
    _state_transition_Q(sm, hs) -> T

A state's expected complete-data log-density over the transitions and terminal
factors its statistics `hs` carry, at its current parameters, in forward
coordinates (so it applies to every state mode through the cache):
`Σ_k −½[n_k (d log 2π + log det Q) + tr(Q⁻¹ R_k)]` plus the same for the terminal
factor. This is what the structural M-step improves; the entry acceptance test
adds the entries' plant row to it.
"""
function _state_transition_Q(
    sm::LQRStateModel{T}, hs::LQRSufficientStatistics{T}
) where {T<:Real}
    c = sm.cache
    d = _state_latent_dim(sm)
    n = _plant_dim(sm)
    total = zero(T)
    logdetQ = logdet(c.Qfwd)
    for k in eachindex(hs.zz)
        nk = hs.nk[k]
        iszero(nk) && continue
        kk = min(k, length(c.M))
        m = size(hs.zz[k], 1) - d - 1
        Θ = hcat(c.M[kk], c.bfwd, m > 0 ? c.Bfwd[kk] : zeros(T, d, 0))
        zy = hs.zy[k]
        R =
            hs.yy[k] .- Θ * zy .- transpose(zy) * transpose(Θ) .+
            Θ * hs.zz[k] * transpose(Θ)
        total -= T(0.5) * (nk * (T(d) * log(T(2π)) + logdetQ) + tr(c.Qfwd \ Symmetric(R)))
    end
    if sm.terminal
        logdetF = logdet(c.Sf_PD)
        for k in eachindex(hs.term_zz)
            tn = hs.term_n[k]
            iszero(tn) && continue
            m = size(hs.term_zz[k], 1) - d - 1
            Ψ = hcat(c.Lf[k], -sm.hf, m > 0 ? c.Ftrm[k] : zeros(T, n, 0))
            Rf = Ψ * hs.term_zz[k] * transpose(Ψ)
            total -=
                T(0.5) * (tn * (T(n) * log(T(2π)) + logdetF) + tr(c.Sf_PD \ Symmetric(Rf)))
        end
    end
    return total
end

"""The structural parameters an acceptance test may have to restore."""
function _lqr_struct_snapshot(sm::LQRStateModel)
    return map(
        deepcopy, (sm.A, sm.Mfree, sm.S, sm.Qc, sm.Σ, sm.h, sm.Bu, sm.Gref, sm.Σf, sm.hf)
    )
end

function _lqr_struct_restore!(sm::LQRStateModel, snap)
    for (dst, src) in zip(
        (sm.A, sm.Mfree, sm.S, sm.h, sm.Bu, sm.Gref, sm.Σf, sm.hf, sm.Σ),
        (snap[1], snap[2], snap[3], snap[6], snap[7], snap[8], snap[9], snap[10], snap[5]),
    )
        copyto!(dst, src)
    end
    for (dst, src) in zip(sm.Qc, snap[4])
        copyto!(dst, src)
    end
    refresh!(sm)
    return sm
end

"""
    _slds_ordinary_weights(dl, fb_storage, seq_ends, k, trials)

State `k`'s ordinary-transition weights per trial: `γ_k(t) − Σ_{i≠k} q(s_{t-1} = i,
s_t = k)` when `k` carries an entry prior (the entered bins take the entry
transition instead), and `nothing` — the responsibilities themselves — otherwise.
"""
function _slds_ordinary_weights(
    dl::SLDSDiscreteLayer{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    seq_ends::AbstractVector{Int},
    k::Int,
    trials,
) where {T<:Real}
    (_has_entries(dl) && dl.entry[k]) || return nothing
    K = size(dl.A, 1)
    out = Vector{Vector{T}}(undef, length(trials))
    for (idx, trial) in enumerate(trials)
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        w = Vector{T}(fb_storage.γ[k, t1:t2])
        for t in t1:t2, i in 1:K
            i == k || (w[t - t1 + 1] -= dl.entry_w[i, k, t])
        end
        out[idx] = max.(w, zero(T))
    end
    return out
end

"""
    _pooled_entry_stats(per_group) -> _EntryStats

The entry prior's statistics pooled over trial groups (the prior is one per
state, whatever the groups' structure). The plant-row blocks are left as the
first group's: they are per group, and read per group.
"""
function _pooled_entry_stats(per_group::AbstractVector{<:_EntryStats})
    length(per_group) == 1 && return per_group[1]
    pooled = deepcopy(per_group[1])
    for st in per_group[2:end]
        pooled.Sψψ .+= st.Sψψ
        pooled.Sλψ .+= st.Sλψ
        pooled.Sλλ .+= st.Sλλ
        pooled.N[] += st.N[]
    end
    return pooled
end

"""
    _slds_entry_group_stats(b, slds_of, tfs, data, dl, seq_ends, groups)
        -> Vector{Union{Nothing,Vector{_EntryStats}}}

Every entry-prior state's statistics per trial group (one group ungrouped, one
per cell under `depends_on`), at the references the model holds now.
"""
function _slds_entry_group_stats(
    b::SLDSBoundaries{T},
    slds_of,
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    dl::SLDSDiscreteLayer{T},
    seq_ends::AbstractVector{Int},
    groups,
) where {T<:Real}
    K = length(b.entry)
    out = Vector{Union{Nothing,Vector{_EntryStats{T}}}}(nothing, K)
    for j in 1:K
        b.entry[j] === nothing && continue
        out[j] = [
            _entry_stats(slds_of, j, tfs, data, dl.entry_w, seq_ends; trials=g) for
            g in groups
        ]
    end
    return out
end

"""
    _slds_update_entry_priors!(boundaries, slds_of, tfs, data, dl, seq_ends, groups)

The joint objective's entry-prior update: each state's exact weighted
regression ([`_update_entry_prior!`](@ref)) on its statistics pooled over the
trial groups.
"""
function _slds_update_entry_priors!(
    b::SLDSBoundaries{T},
    slds_of,
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    dl::SLDSDiscreteLayer{T},
    seq_ends::AbstractVector{Int},
    groups,
) where {T<:Real}
    stats = _slds_entry_group_stats(b, slds_of, tfs, data, dl, seq_ends, groups)
    for (ep, per_group) in zip(b.entry, stats)
        per_group === nothing && continue
        _update_entry_prior!(ep, _pooled_entry_stats(per_group))
    end
    return nothing
end

"""
    _entry_lambda_Q(ep, st) -> T

The entry prior's own expected log-density over the entries `st` pools:
`−½[N (n log 2π + log det P) + tr(P⁻¹ R)]`, `R` the residual scatter of
`λ_t ≈ [μ K] [1; x_{t−1} − r]`.
"""
function _entry_lambda_Q(ep::EntryPrior{T}, st::_EntryStats{T}) where {T<:Real}
    N = st.N[]
    iszero(N) && return zero(T)
    n = length(ep.μ)
    R = _entry_residual_scatter(hcat(ep.μ, ep.K), st.Sψψ, st.Sλψ, st.Sλλ)
    Pc = cholesky(Symmetric(Matrix{T}(ep.P)); check=false)
    issuccess(Pc) || return -T(Inf)
    return -T(0.5) * (N * (T(n) * log(T(2π)) + logdet(Pc)) + tr(Pc \ R))
end

function _entry_residual_scatter(Θ, Sψψ, Sλψ, Sλλ)
    return Sλλ .- Θ * transpose(Sλψ) .- Sλψ * transpose(Θ) .+ Θ * Sψψ * transpose(Θ)
end

"""
    _slds_entry_objective(b, slds_of, sm_of, tfs, data, dl, seq_ends, groups) -> Function

A closure returning the entries' whole expected complete-data log-density at the
parameters the model holds *when it is called*: each entry prior's own row (whose
regressor `x_{t−1} − r⁽ⁱ⁾` moves with the left state's `Gref`) and the plant row
of the entered state (`sm_of(j, g)`, group `g`'s version of state `j`), both
recomputed from the current posterior. The structural step does not see these
terms, so its acceptance adds them: the joint objective's guard
([`_slds_entry_guarded`](@ref)) and the conditional one's score
(`_lqr_conditional_problem`'s `score_extra`).
"""
function _slds_entry_objective(
    b::SLDSBoundaries{T}, slds_of, sm_of, tfs, data, dl, seq_ends, groups
) where {T<:Real}
    return function ()
        stats = _slds_entry_group_stats(b, slds_of, tfs, data, dl, seq_ends, groups)
        total = zero(T)
        for (j, per_group) in enumerate(stats)
            per_group === nothing && continue
            total += _entry_lambda_Q(b.entry[j], _pooled_entry_stats(per_group))
            for (g, st) in enumerate(per_group)
                total += _entry_plant_Q(sm_of(j, g), st)
            end
        end
        return total
    end
end

"""
    _slds_entry_guarded(step!, ldss, sufs, entry_Q)

Run the structural/noise step `step!` and keep its result only if the units'
complete-data objective — their transitions and terminal factors (what `step!`
improves) plus the entries' terms `entry_Q()` (which share their structure but
which `step!` does not see; see [`_slds_entry_objective`](@ref)) — did not go
down. Otherwise every unit is restored to its incoming parameters, which is a
legitimate generalized M-step: the bound cannot decrease either way. A model
without entry priors (`entry_Q === nothing`) runs `step!` unguarded.
"""
function _slds_entry_guarded(step!, ldss, sufs, entry_Q)
    entry_Q === nothing && return step!()
    q0 = _slds_units_Q(ldss, sufs) + entry_Q()
    snaps = [_lqr_struct_snapshot(l.state_model) for l in ldss]
    result = step!()
    q1 = _slds_units_Q(ldss, sufs) + entry_Q()
    tol = sqrt(eps(typeof(q0))) * max(one(q0), abs(q0))
    if !(q1 >= q0 - tol)
        for (l, snap) in zip(ldss, snaps)
            _lqr_struct_restore!(l.state_model, snap)
        end
        @debug "structural step lowered the objective once the entries counted; kept the incoming parameters" q0 q1
    end
    return result
end

function _slds_units_Q(ldss, sufs)
    T = eltype(first(sufs).nk)
    total = zero(T)
    for (u, l) in enumerate(ldss)
        total += _state_transition_Q(l.state_model, sufs[u])
    end
    return total
end

"""
    _slds_conditional_entry_update!(b, slds_of, tfs, data, dl, seq_ends, groups,
                                    probes, sources; max_halvings=12) -> Bool

The entry priors' update when the model conditions on its goals. The score is
`ELBO(y, goals = 0) − log Ẑ`, and an entry prior is part of the prior `log Ẑ`
integrates over, so with `q` fixed its part of the score is

    G(μ, K, P) = Σⱼ E_q[log N(λ; μ_j + K_j ψ, P_j)] − log Ẑ(μ, K, P),

`log Ẑ` from probes restarted as fresh ones would be. As for the chain
([`_slqr_chain_mstep!`](@ref)), holding the probe's posterior fixed would leave a
difference of two regressions that need not be bounded, so there is no
closed-form step. Two proposals instead, each kept only if `G` rises:

1. The joint objective's weighted regression, which maximizes the data half.
2. Otherwise an ascent step along the gradient of `G` at the probe's stationary
   posterior (Danskin): the data's entry statistics less the probe's, mapped
   through the regression's own metric (`ΔΘ = ΔSλψ − Θ ΔSψψ`, `ΔP = ΔR − ΔN P`,
   both ascent directions since `P ≻ 0`), halved until `G` rises by an Armijo
   fraction of what the gradient promises and `P` stays positive definite.

If neither improves `G` the priors stay put. Returns whether they moved; either
way the probes are left smoothed at some proposal, not necessarily the kept one,
so the caller re-smooths them before reusing them.
"""
function _slds_conditional_entry_update!(
    b::SLDSBoundaries{T},
    slds_of,
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    dl::SLDSDiscreteLayer{T},
    seq_ends::AbstractVector{Int},
    groups,
    probes::AbstractVector,
    sources::AbstractVector;
    max_halvings::Int=12,
) where {T<:Real}
    states = [j for j in eachindex(b.entry) if b.entry[j] !== nothing]
    isempty(states) && return false
    stats = _slds_entry_group_stats(b, slds_of, tfs, data, dl, seq_ends, groups)
    data_st = Dict(j => _pooled_entry_stats(stats[j]) for j in states)
    saved = Dict(j => deepcopy(b.entry[j]) for j in states)
    function restore!()
        for j in states
            ep, s0 = b.entry[j], saved[j]
            ep.μ .= s0.μ
            ep.K .= s0.K
            ep.P .= s0.P
        end
    end
    function score()
        data_part = sum(_entry_lambda_Q(b.entry[j], data_st[j]) for j in states)
        isfinite(data_part) || return -T(Inf)
        logz = try
            _slqr_probes_logz!(probes, sources)
        catch err
            _lqr_rejectable(err) || rethrow()
            return -T(Inf)
        end
        return data_part - logz
    end

    base = score()
    isfinite(base) || return false
    #= The probes are now smoothed at the incoming priors: their entry statistics,
    each design weighted by its trial count, are the gradient's other half. =#
    probe_st = Dict{Int,_EntryStats{T}}()
    for j in states
        parts = [
            _entry_stats(
                _ -> probe.slds,
                j,
                probe.tfs,
                probe.data,
                probe.dl.entry_w,
                probe.seq_ends;
                trial_weight=probe.counts,
            ) for probe in probes
        ]
        probe_st[j] = _pooled_entry_stats(parts)
    end

    for j in states
        _update_entry_prior!(b.entry[j], data_st[j])
    end
    gain = score() - base
    if isfinite(gain) && gain >= 0
        @debug "terminal-conditioned entry step" proposal = :regression gain
        return true
    end
    restore!()

    directions = Dict{Int,Tuple{Matrix{T},Matrix{T}}}()
    slope = zero(T)
    for j in states
        ep, dst, pst = b.entry[j], data_st[j], probe_st[j]
        Θ = hcat(ep.μ, ep.K)
        ΔSψψ, ΔSλψ, ΔSλλ = dst.Sψψ - pst.Sψψ, dst.Sλψ - pst.Sλψ, dst.Sλλ - pst.Sλλ
        ΔN = dst.N[] - pst.N[]
        dΘ = ΔSλψ .- Θ * ΔSψψ
        ep.fit_gain || (dΘ[:, 2:end] .= zero(T))
        R = _entry_residual_scatter(Θ, ΔSψψ, ΔSλψ, ΔSλλ)
        dP = (R .+ transpose(R)) ./ 2 .- ΔN .* ep.P
        scale = one(T) / max(dst.N[], sqrt(eps(T)))
        dΘ .*= scale
        dP .*= scale
        Pinv = inv(cholesky(Symmetric(Matrix{T}(ep.P))))
        slope += dot(Pinv * dΘ, dΘ) / scale + T(0.5) * dot(Pinv * dP * Pinv, dP) / scale
        directions[j] = (dΘ, dP)
    end
    if isfinite(slope) && slope > zero(T)
        step = one(T)
        for _ in 0:max_halvings
            feasible = true
            for j in states
                ep, s0 = b.entry[j], saved[j]
                dΘ, dP = directions[j]
                ep.μ .= s0.μ .+ step .* dΘ[:, 1]
                ep.K .= s0.K .+ step .* dΘ[:, 2:end]
                P = s0.P .+ step .* dP
                ep.P .= (P .+ transpose(P)) ./ 2
                feasible &= isposdef(Symmetric(Matrix{T}(ep.P)))
            end
            if feasible
                gain = score() - base
                if isfinite(gain) && gain >= T(1e-4) * step * slope
                    @debug "terminal-conditioned entry step" proposal = :gradient step gain
                    return true
                end
            end
            step /= 2
        end
    end
    restore!()
    @debug "terminal-conditioned entry step rejected" base
    return false
end
