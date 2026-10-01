#=============================================================================
Boundaries between an SLDS's discrete states: exit bridges.

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

Generative semantics. With bridges the model is the joint over the chain, the
latents and the bridge pseudo-observations, as the terminal factor is under
`condition_terminal = false`; the reported score is `log p(y, bridges = 0)`.
Conditioning on the bridges (dividing by their probability) needs a normalizer
over the switching paths and is not implemented yet, so a model with bridges
must score the joint objective.
=============================================================================#

"""
    set_boundaries!(slds; bridge_states=Int[]) -> slds

Configure what happens at the switches between `slds`'s discrete states.

- `bridge_states`: the states whose segment ends with a **bridge** — their
  terminal factor `λ_t = Q_f (x_t − r_t) + h_f`, applied at every exit from the
  state (and, as before, at the end of a trial that ends in it). These are the
  finite-horizon *control* stages of a staged task; a state left out (a hold, a
  free state) is left without one. A bridged state must be an inverse-LQR state
  (`mode = :lqr`) built with `terminal = true` — the bridge *is* its terminal
  factor, with its cost `Qc[k_f]` (`terminal_regime`, else the last schedule
  entry), covariance `Σf`, offset `hf` and reference `Gref`.

Bridges score the joint objective `log p(y, bridges = 0)`, so every bridged
state needs `condition_terminal = false`.

Passing no states clears the configuration. Returns `slds`.

# Example
```julia
# control → hold → control → hold, on a banded chain
slds.A, slds.πₖ = banded_transition(4; stay=median_dwell_stay.([10, 25, 12]))
set_boundaries!(slds; bridge_states=[1, 3])
```
"""
function set_boundaries!(
    slds::SLDS{T}; bridge_states::AbstractVector{<:Integer}=Int[]
) where {T<:Real}
    K = length(slds.LDSs)
    bridge = falses(K)
    for k in bridge_states
        1 <= k <= K ||
            throw(ArgumentError("bridge_states names state $k; the model has $K states"))
        bridge[k] = true
    end
    if !any(bridge)
        slds.boundaries = nothing
        return slds
    end
    slds.boundaries = SLDSBoundaries{T}(
        collect(bridge), Union{Nothing,EntryPrior{T}}[nothing for _ in 1:K]
    )
    _validate_boundaries(slds)
    return slds
end

"""Whether `slds` carries any exit bridge."""
_slds_has_bridges(slds::SLDS) = slds.boundaries !== nothing && any(slds.boundaries.bridge)

"""
    _validate_boundaries(slds)

Refuse a boundary configuration the model cannot honour: a bridge on a state
with no terminal factor to apply, on a non-LQR or `:free` state, or under the
conditional objective (whose normalizer does not yet carry bridges).
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
        sm.condition_terminal && throw(
            ArgumentError(
                "bridge_states includes state $k, which conditions on its terminal " *
                "factor. Exit bridges score the joint objective `log p(y, bridges = " *
                "0)`; set `condition_terminal = false` on the bridged states.",
            ),
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
