#=============================================================================
Known epoch boundaries of an inverse-LQR cost schedule.

A cost schedule (`--cost-onset` in the smoulder driver) says which `Qc` each
transition is under, and the switch between epochs happens at a *known* bin.
Structurally that is a switching model whose discrete path is given, and the
two boundary factors a switching model can attach to its switches
(`slds_boundaries.jl`) make sense here too:

  * a **bridge** ends the epoch being left with a terminal factor of its own,
    `λ_b = Q_b (x_b − r_b) + h_f`, at the epoch's last bin `b`, against a cost
    regime `Q_b = Qc[k_b]` that no transition uses (the trial's own end has one
    the same way). `Σf` and `hf` are the trial end's. Under
    `condition_terminal` it is conditioned on, exactly like the end: the
    normalizer integrates over every factor at once.

  * an **entry** starts the new epoch from a fresh plan: the transition
    `b → b+1` is replaced by `λ_{b+1} ~ N(μ + K (x_b − r_b), P)` and the
    model's own plant row. That density is linear-Gaussian in `(z_b, z_{b+1})`,
    so it is held as one more forward transition (see [`_LQREntryCache`](@ref))
    and every kernel reads it in place of the ordinary one.

Both are carried by [`LQRSwitch`](@ref)es on the model, set with
[`set_schedule_boundaries!`](@ref). A model without switches never reaches any
of this: every kernel checks `isempty(sm.switches)` first and keeps its
original path.

Index conventions. A switch at schedule position `b` is the first position of
the new epoch: `schedule[b]` is the new epoch's regime and indexes the
transition `b → b+1`, while the epoch being left owns `b − 1 → b`. For a trial
that starts `offset` bins into the schedule, that is trial bin `t_b = b −
offset`, and the trial crosses the switch only if it holds the transition
`t_b → t_b + 1`, i.e. `1 ≤ t_b ≤ T − 1`.

The **reference gate** (`sm.gref_gate`, cost regime × input column) is the other
half: it restricts the reference each regime reads to a subset of the input
columns, `r⁽ᵏ⁾ = Gref D_k u`, so one `Gref` holds a separate map per regime
without the inputs having to be zeroed per bin — which they could not be at a
bridge, where one bin's input feeds both the new epoch's running reference and
the bridge's.
=============================================================================#

# ----------------------------------------------------------------------------
# Reference gate
# ----------------------------------------------------------------------------

"""
    _normalize_gref_gate(gate, K, m) -> Matrix{Bool}

The `K × m` gate (cost regime × input column), or the empty `0 × 0` matrix that
means "every regime reads every column".
"""
function _normalize_gref_gate(gate, K::Int, m::Int)
    gate === nothing && return Matrix{Bool}(undef, 0, 0)
    what = "gref_gate (cost regimes × input columns)"
    size(gate) == (K, m) || throw(DimensionMismatchError(what, (K, m), size(gate)))
    return Matrix{Bool}(gate)
end

"""
    _gref_for(sm, k) -> Matrix

The reference map cost regime `k` reads: `Gref` with every column the gate
closes for `k` zeroed. `Gref` itself when there is no gate.
"""
function _gref_for(sm::LQRStateModel, k::Int)
    isempty(sm.gref_gate) && return sm.Gref
    return sm.Gref .* transpose(view(sm.gref_gate, k, :))
end

"""
    set_gref_gate!(sm, gate) -> sm

Give each cost regime of an inverse-LQR model its own reference map out of one
`Gref`: regime `k` reads `r⁽ᵏ⁾ = Gref · D_k u`, where `D_k` keeps the input
columns with `gate[k, j] == true`. With the reference predictors repeated once per
regime and each copy opened to one regime, that is one independently fitted map
per regime; opening a copy to several regimes shares the map between them (for
example an epoch's running cost and the bridge that ends it).

`gate` is `length(Qc) × ux_dim`. `nothing` removes it. Columns no regime needs
(general inputs, which `Gref` leaves at zero) may be open or closed, it makes no
difference. Must be set before `depends_on` variants are built.
"""
function set_gref_gate!(sm::LQRStateModel, gate)
    sm.mode in (:lqr, :causal) || throw(
        ArgumentError("a reference gate needs a finite-horizon (`:lqr` or `:causal`) model")
    )
    sm.variants === nothing || throw(
        ArgumentError(
            "set the reference gate before `depends_on` variants are built; they share it",
        ),
    )
    sm.gref_gate = _normalize_gref_gate(gate, length(sm.Qc), size(sm.Bu, 2))
    refresh!(sm)
    return sm
end

function set_gref_gate!(lds::LinearDynamicalSystem, gate)
    return (set_gref_gate!(lds.state_model, gate); lds)
end

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

"""
    set_schedule_boundaries!(lds; bridges=Pair{Int,Int}[], entries=Int[],
                             entry_gain=true, entry_cov=1.0) -> lds

Attach boundary factors to the known epoch switches of an inverse-LQR model's
cost schedule — the deterministic counterpart of [`set_boundaries!`](@ref).

- `bridges`: `pos => k` pairs. The epoch that ends at schedule position `pos`
  (the first position of the next epoch; see [`LQRSwitch`](@ref)) ends with a
  terminal factor against cost regime `k`, `λ_pos = Qc[k] (x_pos − r_pos) + hf`,
  with the trial end's `Σf` and `hf`. `k` is normally a regime of its own that
  no transition uses, so the bridge's goal cost is fitted from the bridges alone;
  its reference is `Gref` gated to `k` (see [`set_gref_gate!`](@ref)). Needs
  `terminal = true`, and is conditioned on under `condition_terminal`.
- `entries`: positions whose transition `pos → pos + 1` starts a fresh plan,
  `λ_{pos+1} ~ N(μ + K (x_pos − r_pos), P)`, `r` the reference of the epoch
  being left (regime `schedule[pos − 1]`), while the plant moves under the
  model's own row. `μ = 0`, `K = 0`, `P = entry_cov · I` to start, fitted by EM;
  `entry_gain = false` keeps `K` at zero.

Each `pos` must satisfy `2 ≤ pos ≤ length(schedule) − 1`. A trial applies a
boundary only where it crosses it (see [`LQRSwitch`](@ref)). No bridges and no
entries clears the configuration. Must be set before `depends_on` variants are
built; the variants share the parent's switches.
"""
function set_schedule_boundaries!(
    sm::LQRStateModel{T};
    bridges=Pair{Int,Int}[],
    entries=Int[],
    entry_gain::Bool=true,
    entry_cov::Real=1.0,
) where {T<:Real}
    sm.variants === nothing || throw(
        ArgumentError(
            "set the schedule boundaries before `depends_on` variants are built; " *
            "they share them",
        ),
    )
    entry_cov > 0 || throw(ArgumentError("entry_cov must be positive, got $entry_cov"))
    n = _plant_dim(sm)
    bmap = Dict{Int,Int}()
    for p in bridges
        pos, k = Int(first(p)), Int(last(p))
        haskey(bmap, pos) &&
            throw(ArgumentError("bridges names schedule position $pos twice"))
        bmap[pos] = k
    end
    eset = Set{Int}(Int.(entries))
    length(eset) == length(entries) ||
        throw(ArgumentError("entries names a schedule position twice"))
    switches = LQRSwitch{T}[]
    for pos in sort!(collect(union(keys(bmap), eset)))
        entry = if pos in eset
            EntryPrior{T}(
                zeros(T, n),
                zeros(T, n, n),
                Matrix{T}(T(entry_cov) * I, n, n),
                entry_gain,
            )
        else
            nothing
        end
        push!(switches, LQRSwitch{T}(pos, get(bmap, pos, 0), entry))
    end
    old = sm.switches
    sm.switches = switches
    try
        _validate_switches(sm)
    catch
        sm.switches = old
        rethrow()
    end
    refresh!(sm)
    return sm
end

function set_schedule_boundaries!(lds::LinearDynamicalSystem; kwargs...)
    set_schedule_boundaries!(lds.state_model; kwargs...)
    return lds
end

"""
    _validate_switches(sm)

Refuse switches the model cannot honour: on a model that is not a
finite-horizon one or has no schedule, outside the schedule, a bridge without a
terminal factor or against a cost that does not exist, an entry prior of the
wrong size.
"""
function _validate_switches(sm::LQRStateModel)
    sw = sm.switches
    isempty(sw) && return nothing
    sm.mode === :lqr || throw(
        ArgumentError(
            "schedule boundaries need a finite-horizon (`:lqr`) model; this one is " *
            "`:$(sm.mode)`",
        ),
    )
    L = length(sm.schedule)
    L >= 3 || throw(
        ArgumentError(
            "schedule boundaries sit between the epochs of a cost schedule, and this " *
            "model has none (pass `schedule`)",
        ),
    )
    n = _plant_dim(sm)
    K = length(sm.Qc)
    seen = Set{Int}()
    for s in sw
        s.pos in seen && throw(ArgumentError("two switches at schedule position $(s.pos)"))
        push!(seen, s.pos)
        2 <= s.pos <= L - 1 || throw(
            ArgumentError(
                "a schedule boundary at position $(s.pos) is outside 2:$(L - 1): it " *
                "needs a transition into it and one out of it on the schedule",
            ),
        )
        if s.bridge != 0
            1 <= s.bridge <= K || throw(
                ArgumentError(
                    "the bridge at position $(s.pos) names cost $(s.bridge); the model " *
                    "has $K",
                ),
            )
            sm.terminal || throw(
                ArgumentError(
                    "a bridge is a terminal factor applied mid-trial, so it needs " *
                    "`terminal = true`",
                ),
            )
        end
        ep = s.entry
        if ep !== nothing
            (length(ep.μ) == n && size(ep.K) == (n, n) && size(ep.P) == (n, n)) || throw(
                DimensionMismatchError("entry prior at position $(s.pos)", n, length(ep.μ)),
            )
            isposdef(Symmetric(ep.P)) ||
                throw(ArgumentError("the entry prior at position $(s.pos) has a non-PD P"))
        end
    end
    issorted(sw; by=s -> s.pos) ||
        throw(ArgumentError("switches must be sorted by schedule position"))
    return nothing
end

"""Whether `lds` is an inverse-LQR model carrying schedule boundaries."""
_has_switches(::Any) = false
_has_switches(sm::LQRStateModel) = !isempty(sm.switches)
function _has_switches(lds::LinearDynamicalSystem{T,S}) where {T,S<:LQRStateModel}
    return _has_switches(lds.state_model)
end

"""Whether any switch carries an entry prior."""
_has_entries(sm::LQRStateModel) = any(s -> s.entry !== nothing, sm.switches)

"""The bridge regimes, for the "every regime is reached" check."""
_bridge_regimes(sm::LQRStateModel) = Int[s.bridge for s in sm.switches if s.bridge != 0]

# ----------------------------------------------------------------------------
# Lookup, in trial bins
# ----------------------------------------------------------------------------

"""
    _entry_into(sm, t) -> Int

Index into `sm.switches` of the entry prior that replaces the transition
`t − 1 → t` of a trial (with `sm`'s own `cost_offset`), or `0`.
"""
@inline function _entry_into(sm::LQRStateModel, t::Int)
    sw = sm.switches
    isempty(sw) && return 0
    a = t - 1 + sm.cost_offset
    @inbounds for i in eachindex(sw)
        s = sw[i]
        s.pos == a && return s.entry === nothing ? 0 : i
    end
    return 0
end

"""
    _bridge_at(sm, t, tsteps) -> Int

The bridge cost regime applied at bin `t` of a trial of length `tsteps`, or `0`.
A trial carries the bridge only if it also holds the transition out of `t`.
"""
@inline function _bridge_at(sm::LQRStateModel, t::Int, tsteps::Int)
    sw = sm.switches
    (isempty(sw) || t > tsteps - 1) && return 0
    a = t + sm.cost_offset
    @inbounds for s in sw
        s.pos == a && return s.bridge
    end
    return 0
end

# ----------------------------------------------------------------------------
# Derived cache
# ----------------------------------------------------------------------------

function _placeholder_entry_cache(::Type{T}, d::Int, m::Int) where {T}
    return _LQREntryCache{T}(
        zeros(T, d, d),
        zeros(T, d),
        zeros(T, d, m),
        PDMat(Matrix{T}(I, d, d)),
        zeros(T, d, d),
        zeros(T, d, d),
        zeros(T, d, d),
        zeros(T, d, d),
        zero(T),
    )
end

"""
    _refresh_switches!(sm)

Rebuild the forward transition of every entry prior (see [`_LQREntryCache`](@ref)).
Called by `refresh!` after the ordinary cache, so it reads the current `A`, `S`,
`h`, `Bu`, `Σ` and `Gref`.
"""
function _refresh_switches!(sm::LQRStateModel{T}) where {T<:Real}
    c = sm.cache
    sw = sm.switches
    if isempty(sw)
        isempty(c.switch) || empty!(c.switch)
        return nothing
    end
    n = _plant_dim(sm)
    d = 2n
    m = size(sm.Bu, 2)
    xr, lr = 1:n, (n + 1):d
    resize!(c.switch, length(sw))
    A, S = sm.A, sm.S
    Σxx = Matrix{T}(view(sm.Σ, xr, xr))
    for (i, s) in enumerate(sw)
        ep = s.entry
        if ep === nothing
            c.switch[i] = _placeholder_entry_cache(T, d, m)
            continue
        end
        K, μ, P = ep.K, ep.μ, ep.P
        SK = S * K
        Me = zeros(T, d, d)
        @views begin
            Me[xr, xr] .= A .- SK
            Me[lr, xr] .= K
        end
        b = Vector{T}(undef, d)
        @views begin
            b[xr] .= sm.h[xr] .- S * μ
            b[lr] .= μ
        end
        B = zeros(T, d, m)
        if m > 0
            G = _gref_for(sm, _regime_at(sm, s.pos - 1))
            KG = K * G
            @views begin
                B[xr, :] .= sm.Bu[xr, :] .+ S * KG
                B[lr, :] .= .-KG
            end
        end
        SP = S * P
        Q = Matrix{T}(undef, d, d)
        @views begin
            Q[xr, xr] .= Σxx .+ SP * S
            Q[xr, lr] .= .-SP
            Q[lr, xr] .= .-transpose(SP)
            Q[lr, lr] .= P
        end
        Qpd = PDMat(Symmetrize!(Q))
        negQinv = -Matrix(inv(Qpd))
        QinvM = Qpd \ Me
        negMtQinvM = -(transpose(Me) * QinvM)
        c.switch[i] = _LQREntryCache{T}(
            Me,
            b,
            B,
            Qpd,
            negQinv,
            QinvM,
            Matrix(transpose(QinvM)),
            Matrix(Symmetrize!(negMtQinvM)),
            -T(0.5) * (T(d) * log(T(2π)) + logdet(Qpd)),
        )
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------

"""
    _entry_residual!(out, ec, x, t, ux)

`z_t − M_e z_{t−1} − b_e − B_e u_{t−1}` for the entry transition into `t`.
"""
@inline function _entry_residual!(
    out::AbstractVector{T}, ec::_LQREntryCache, x::AbstractMatrix{T}, t::Int, ux
) where {T<:Real}
    @views mul!(out, ec.M, x[:, t - 1])
    if ux !== nothing && size(ec.B, 2) > 0
        @views mul!(out, ec.B, ux[:, t - 1], one(T), one(T))
    end
    @views out .= x[:, t] .- out .- ec.b
    return out
end

"""
    _factor_residual!(out, sm, x, t, k, ux)

`Λf[k] z_t + Q_k G_k u_t − h_f`: the terminal-factor residual against cost
regime `k` at bin `t` — the trial end's when `t = T`, a bridge's otherwise.
"""
@inline function _factor_residual!(
    out::AbstractVector{T}, sm::LQRStateModel, x::AbstractMatrix{T}, t::Int, k::Int, ux
) where {T<:Real}
    c = sm.cache
    @views mul!(out, c.Lf[k], x[:, t])
    if ux !== nothing && size(c.Ftrm[k], 2) > 0
        @views mul!(out, c.Ftrm[k], ux[:, t], one(T), one(T))
    end
    out .-= sm.hf
    return out
end

"""
    _state_gradient_switched!(grad, ws, lds, x, ux)

[`_state_gradient!`](@ref) for a model with schedule boundaries: each transition
reads its own blocks (an entry's where there is one) and every bridge adds its
factor's gradient at its bin.
"""
function _state_gradient_switched!(
    grad::AbstractMatrix{T},
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    ux::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,S<:LQRStateModel{T},O<:AbstractObservationModel{T}}
    tsteps = size(x, 2)
    sm = _lqr(lds)
    c = sm.cache
    n = _plant_dim(sm)
    dxt = ws.opt.dxt
    tmp2 = ws.opt.tmp2

    # Prior on z₁.
    @views dxt .= x[:, 1] .- sm.x0
    @views mul!(grad[:, 1], ws.consts.x_t, dxt)
    for t in 2:tsteps
        # The transition t−1 → t: −Q⁻¹ r into t, Mᵀ Q⁻¹ r into t − 1.
        _transition_residual!(dxt, lds, x, t, ux)
        e = _entry_into(sm, t)
        if e == 0
            @views mul!(grad[:, t], c.negQinv, dxt)
            mul!(tmp2, c.MtQinv[_regime(sm, t - 1)], dxt)
        else
            ec = c.switch[e]
            @views mul!(grad[:, t], ec.negQinv, dxt)
            mul!(tmp2, ec.MtQinv, dxt)
        end
        @views grad[:, t - 1] .+= tmp2
    end

    rf = view(dxt, 1:n)
    for t in 1:tsteps
        k = if t == tsteps
            (sm.terminal ? _terminal_regime(sm, tsteps) : 0)
        else
            _bridge_at(sm, t, tsteps)
        end
        k == 0 && continue
        _factor_residual!(rf, sm, x, t, k, ux)
        @views mul!(grad[:, t], c.LtSinv[k], rf, -one(T), one(T))
    end
    return grad
end

"""
    _state_hessian_switched!(btd, cc, sm, tsteps)

[`_state_hessian_blocks!`](@ref) for a model with schedule boundaries.
"""
function _state_hessian_switched!(
    btd, cc::SmoothConstants{T}, sm::LQRStateModel, tsteps::Int
) where {T<:Real}
    c = sm.cache
    btd.H_diag[1] .= cc.x_t
    for t in 2:tsteps
        btd.H_diag[t] .= zero(T)
    end
    for t in 2:tsteps
        e = _entry_into(sm, t)
        if e == 0
            k = _regime(sm, t - 1)
            copyto!(btd.H_sub[t - 1], c.QinvM[k])
            copyto!(btd.H_super[t - 1], transpose(c.QinvM[k]))
            btd.H_diag[t - 1] .+= c.negMtQinvM[k]
            btd.H_diag[t] .+= c.negQinv
        else
            ec = c.switch[e]
            copyto!(btd.H_sub[t - 1], ec.QinvM)
            copyto!(btd.H_super[t - 1], transpose(ec.QinvM))
            btd.H_diag[t - 1] .+= ec.negMtQinvM
            btd.H_diag[t] .+= ec.negQinv
        end
    end
    for t in 1:(tsteps - 1)
        k = _bridge_at(sm, t, tsteps)
        k == 0 || (btd.H_diag[t] .+= c.negLtSL[k])
    end
    sm.terminal && (btd.H_diag[tsteps] .+= c.negLtSL[_terminal_regime(sm, tsteps)])
    return nothing
end

"""
    _bridge_loglik!(tmp, sm, x, t, k, ux) -> T

The log-density of the bridge (or terminal) factor against regime `k` at bin `t`.
"""
function _factor_loglik!(
    tmp::AbstractVector{T}, sm::LQRStateModel, x::AbstractMatrix{T}, t::Int, k::Int, ux
) where {T<:Real}
    c = sm.cache
    rf = view(tmp, 1:_plant_dim(sm))
    _factor_residual!(rf, sm, x, t, k, ux)
    _whiten!(c.Sf_PD.chol, rf)
    return T(c.cF) - T(0.5) * sum(abs2, rf)
end

# ----------------------------------------------------------------------------
# Sufficient statistics
# ----------------------------------------------------------------------------

"""
    _switch_bin(sm, s, tsteps) -> Int

The trial bin of switch `s` for a trial of length `tsteps` (with `sm`'s own
`cost_offset`), or `0` when the trial does not cross it.
"""
@inline function _switch_bin(sm::LQRStateModel, s::LQRSwitch, tsteps::Int)
    t = s.pos - sm.cost_offset
    return 1 <= t <= tsteps - 1 ? t : 0
end

"""
    _regime_runs_switched(sm, tsteps)

[`_regime_runs`](@ref) with every entry transition left out: those are not the
regime's adjoint equation, and their statistics go to `entry_ww` instead.
"""
function _regime_runs_switched(sm::LQRStateModel, tsteps::Int)
    runs = Tuple{Int,Int,Int}[]
    tsteps >= 2 || return runs
    t0 = 0
    k0 = 0
    for t in 1:(tsteps - 1)
        skip = _entry_into(sm, t + 1) != 0
        k = skip ? 0 : _regime(sm, t)
        if k != k0
            k0 == 0 || push!(runs, (k0, t0, t - 1))
            t0, k0 = t, k
        end
    end
    k0 == 0 || push!(runs, (k0, t0, tsteps - 1))
    return runs
end

"""
    _lqr_factor_moment!(acc, k, w, x, P, ux, t, with_cov)

`w · E[[z_t; 1; u_t][z_t; 1; u_t]ᵀ]` into regime `k`'s terminal block, in the
layout `_finalize_lqr_stats!` mirrors, and `w` into its count. A bridge is a
terminal factor at an interior bin, so it lands exactly where the trial end's
factor does and every consumer of `term_zz` reads it unchanged.
"""
function _lqr_factor_moment!(
    acc,
    k::Int,
    w::T,
    x::Matrix{T},
    P::Array{T,3},
    ux::AbstractMatrix,
    t::Int,
    with_cov::Bool,
) where {T<:Real}
    d = size(x, 1)
    m = size(ux, 1)
    term_zz = acc.term_zz[k]
    xt = tview(x, :, t)
    BLAS.ger!(w, xt, xt, tview(term_zz, 1:d, 1:d))
    with_cov && @views term_zz[1:d, 1:d] .+= w .* P[:, :, t]
    for i in 1:d
        term_zz[i, d + 1] += w * x[i, t]
    end
    if m > 0
        ut = Vector{T}(view(ux, :, t))
        @views mul!(term_zz[1:d, (d + 2):(d + 1 + m)], xt, transpose(ut), w, one(T))
        BLAS.ger!(w, ut, ut, tview(term_zz, (d + 2):(d + 1 + m), (d + 2):(d + 1 + m)))
        for j in 1:m
            term_zz[d + 1, d + 1 + j] += w * ut[j]
        end
    end
    acc.term_n[k] += w
    return nothing
end

"""
    _lqr_entry_moment!(acc, e, w, x, P, P1, ux, t, with_cov)

`w · E[ω ωᵀ]`, `ω = [z_t; z_{t+1}; 1; u_t]`, into entry block `e` (both
triangles), and `w` into its count.
"""
function _lqr_entry_moment!(
    acc,
    e::Int,
    w::T,
    x::Matrix{T},
    P::Array{T,3},
    P1::Array{T,3},
    ux::AbstractMatrix,
    t::Int,
    with_cov::Bool,
) where {T<:Real}
    d = size(x, 1)
    m = size(ux, 1)
    W = acc.entry_ww[e]
    ω = Vector{T}(undef, 2d + 1 + m)
    @views begin
        ω[1:d] .= x[:, t]
        ω[(d + 1):(2d)] .= x[:, t + 1]
        ω[2d + 1] = one(T)
        m > 0 && (ω[(2d + 2):end] .= ux[:, t])
    end
    BLAS.ger!(w, ω, ω, W)
    with_cov && _lqr_entry_cov!(W, w, P, P1, t, d)
    acc.entry_n[e] += w
    return nothing
end

# The covariance half of an entry moment: `Cov([z_t; z_{t+1}])`.
function _lqr_entry_cov!(W::Matrix{T}, w::T, P, P1, t::Int, d::Int) where {T}
    a, b = 1:d, (d + 1):(2d)
    @views begin
        W[a, a] .+= w .* P[:, :, t]
        W[b, b] .+= w .* P[:, :, t + 1]
        # Cov(z_{t+1}, z_t) = p_smooth_tt1[:, :, t+1]
        W[b, a] .+= w .* P1[:, :, t + 1]
        W[a, b] .+= w .* transpose(P1[:, :, t + 1])
    end
    return W
end

"""
    _lqr_switch_stats_trial!(acc, sm, fs, ux, d, m, w, with_cov)

One trial's bridge and entry moments (the transitions themselves went through
[`_regime_runs_switched`](@ref)).
"""
function _lqr_switch_stats_trial!(
    acc, sm::LQRStateModel, fs::FilterSmooth{T}, ux, d::Int, m::Int, w::T, with_cov::Bool
) where {T<:Real}
    x = fs.x_smooth::Matrix{T}
    P = fs.p_smooth::Array{T,3}
    P1 = fs.p_smooth_tt1::Array{T,3}
    T_n = size(x, 2)
    u = ux === nothing ? zeros(T, 0, T_n) : ux
    for (e, s) in enumerate(sm.switches)
        t = _switch_bin(sm, s, T_n)
        t == 0 && continue
        s.bridge == 0 || _lqr_factor_moment!(acc, s.bridge, w, x, P, u, t, with_cov)
        s.entry === nothing || _lqr_entry_moment!(acc, e, w, x, P, P1, u, t, with_cov)
    end
    return nothing
end

# The probe's shared-covariance half of the same.
function _lqr_switch_cov_item!(acc, sm::LQRStateModel, P, P1, w::T, d::Int) where {T}
    T_n = size(P, 3)
    for (e, s) in enumerate(sm.switches)
        t = _switch_bin(sm, s, T_n)
        t == 0 && continue
        s.bridge == 0 || (@views acc.term_zz[s.bridge][1:d, 1:d] .+= w .* P[:, :, t])
        s.entry === nothing || _lqr_entry_cov!(acc.entry_ww[e], w, P, P1, t, d)
    end
    return nothing
end

# ----------------------------------------------------------------------------
# The entry transitions' share of the state Q-term
# ----------------------------------------------------------------------------

"""
    _entry_design(ec) -> Matrix

`Ψ = [−M_e  I  −b_e  −B_e]`, so that the entry residual is `Ψ ω` for
`ω = [z_b; z_{b+1}; 1; u_b]`.
"""
function _entry_design(ec::_LQREntryCache{T}) where {T}
    d = size(ec.M, 1)
    m = size(ec.B, 2)
    Ψ = zeros(T, d, 2d + 1 + m)
    @views begin
        Ψ[:, 1:d] .= .-ec.M
        for i in 1:d
            Ψ[i, d + i] = one(T)
        end
        Ψ[:, 2d + 1] .= .-ec.b
        m > 0 && (Ψ[:, (2d + 2):end] .= .-ec.B)
    end
    return Ψ
end

"""
    _lqr_entry_Q(sm, hs) -> T

`Σ_e E_q[log N(z_{b+1}; M_e z_b + b_e + B_e u_b, Q_e)]` over every entry
transition the statistics saw, at `sm`'s current parameters (its cache must be
current).
"""
function _lqr_entry_Q(sm::LQRStateModel{T}, hs::LQRSufficientStatistics{T}) where {T<:Real}
    total = zero(T)
    for (e, s) in enumerate(sm.switches)
        s.entry === nothing && continue
        N = hs.entry_n[e]
        N > zero(T) || continue
        ec = sm.cache.switch[e]
        Ψ = _entry_design(ec)
        R = Ψ * hs.entry_ww[e] * transpose(Ψ)
        total += N * ec.cQ - T(0.5) * tr(ec.Q \ Symmetrize!(R))
    end
    return total
end

"""
    _gated(G, gate, k)

`G` with the columns regime `k`'s gate closes zeroed, or `G` itself without a
gate. `G` is any matrix whose columns are the inputs (`Gref`, or a gradient
with respect to it).
"""
@inline _gated(G::AbstractMatrix, gate::Matrix{Bool}, k::Int) =
    isempty(gate) ? G : G .* transpose(view(gate, k, :))

# ----------------------------------------------------------------------------
# Exact normalizer with interior factors
# ----------------------------------------------------------------------------

#=
`log Z = log ∫ p(z_{1:T}) Π_f N(0; Λ_f z_{t_f} + F_f u − h_f, Σf) dz` over the
trial-end factor and every bridge the trial crosses, the transitions including
the entries'. It is the same backward square-root integration as
`_lqr_terminal_logz`, generalized in two ways:

  * the whitened constraint `N(a; H z_t, I)` grows by `n` rows at each bridge,
    and is folded back to at most `d = 2n` rows by an orthogonal rotation
    whenever it outgrows the state — the rows rotated out no longer involve `z`
    and contribute `exp(−½‖a₂‖²)` on their own;
  * an entry is one more linear-Gaussian transition, read off its cache.

Designs whose last bin lands at one schedule position cross the same transitions
and the same switches at every backward step — a design meets a switch exactly
when the recursion reaches it — so the input-independent recursion is shared and
the mean recursion runs over all of them at once, as in the unswitched batch.
=#

"""
    _lqr_switched_logz_sum(sm, designs) -> T

`Σ_i count_i · log Z_i` for a model with schedule boundaries.
"""
function _lqr_switched_logz_sum(sm::LQRStateModel{T}, designs) where {T}
    horizons = [size(d.ux, 2) for d in designs]
    ends = horizons .+ [_design_offset(d) for d in designs]
    by_end = Dict{Int,Vector{Int}}()
    for (i, e) in enumerate(ends)
        push!(get!(by_end, e, Int[]), i)
    end
    Lf = cholesky(Symmetric(Matrix(sm.Σf))).L
    Lq = sm.cache.G * cholesky(Symmetric(Matrix(sm.Σ))).L
    Le = [
        s.entry === nothing ? nothing : sm.cache.switch[e].Q.chol.L for
        (e, s) in enumerate(sm.switches)
    ]
    L0 = cholesky(Symmetric(Matrix(sm.P0))).L
    total = zero(T)
    for e in sort!(collect(keys(by_end)); rev=true)
        total += _lqr_switched_logz_end(sm, (; Lf, Lq, Le, L0), designs, by_end[e], e)
    end
    return total
end

# `L` with `L Lᵀ = I + G Gᵀ`, and `Σ log|Lᵢᵢ|`.
function _whiten_rows(G::AbstractMatrix{T}) where {T}
    r = size(G, 1)
    F = qr(vcat(Matrix{T}(I, r, r), transpose(G)))
    L = LowerTriangular(Matrix(transpose(F.R)))
    return L, sum(log ∘ abs, diag(L); init=zero(T))
end

function _lqr_switched_logz_end(
    sm::LQRStateModel{T}, chol, designs, idx::AbstractVector{Int}, last_bin::Int
) where {T}
    n = _plant_dim(sm)
    d = 2n
    c = sm.cache
    (; Lf, Lq, Le, L0) = chol
    order = sort(idx; by=i -> -size(designs[i].ux, 2))
    nd = length(order)
    hs = [size(designs[i].ux, 2) for i in order]
    m = size(c.Ftrm[1], 2)
    log2π = log(T(2π))
    u_at(col, a) = view(designs[order[col]].ux, :, a - (last_bin - hs[col]))

    # The trial-end factor, against the regime its last bin pins.
    kf = _terminal_regime_at(sm, last_bin)
    H = Lf \ c.Lf[kf]
    A = Matrix{T}(undef, n, nd)
    for col in 1:nd
        a = view(A, :, col)
        a .= sm.hf
        m > 0 && mul!(a, c.Ftrm[kf], u_at(col, last_bin), -one(T), one(T))
    end
    ldiv!(Lf, A)
    logLf = sum(log ∘ abs, diag(Lf))
    value = -logLf                    # shared by every design still active
    nrows = n                         # Gaussian dimensions integrated so far
    dropped = zeros(T, nd)            # ‖a₂‖² of rows rotated out, per design
    total = zero(T)

    nactive = nd
    j = 0
    while nactive > 0
        j += 1
        a = last_bin - j               # the bin this step arrives at
        # Transition a → a + 1, an entry's or the regime's.
        sw = findfirst(s -> s.pos == a, sm.switches)
        e = (sw !== nothing && sm.switches[sw].entry !== nothing) ? sw : 0
        if e == 0
            k = _regime_at(sm, a)
            M, b, B, Lw = c.M[k], c.bfwd, c.Bfwd[k], Lq
        else
            ec = c.switch[e]
            M, b, B, Lw = ec.M, ec.b, ec.B, Le[e]
        end
        L, logL = _whiten_rows(H * Lw)
        value -= logL
        Aj = view(A, :, 1:nactive)
        Aj .-= H * b
        if m > 0
            HB = H * B
            for col in 1:nactive
                mul!(view(A, :, col), HB, u_at(col, a), -one(T), one(T))
            end
        end
        ldiv!(L, Aj)
        H = L \ (H * M)

        # A bridge at bin `a`: `n` more rows for every design still active.
        if sw !== nothing && sm.switches[sw].bridge != 0
            kb = sm.switches[sw].bridge
            Hb = Lf \ c.Lf[kb]
            Ab = Matrix{T}(undef, n, nd)
            for col in 1:nactive
                ab = view(Ab, :, col)
                ab .= sm.hf
                m > 0 && mul!(ab, c.Ftrm[kb], u_at(col, a), -one(T), one(T))
            end
            ldiv!(Lf, view(Ab, :, 1:nactive))
            H = vcat(H, Hb)
            A = vcat(A, Ab)
            value -= logLf
            nrows += n
            if size(H, 1) > d
                F = qr(H)
                Qm = Matrix(F.Q * Matrix{T}(I, size(H, 1), size(H, 1)))
                Ar = transpose(Qm) * view(A, :, 1:nactive)
                for col in 1:nactive
                    dropped[col] += sum(abs2, view(Ar, (d + 1):size(H, 1), col))
                end
                H = Matrix(F.R)[1:d, :]
                Anew = Matrix{T}(undef, d, nd)
                Anew[:, 1:nactive] .= view(Ar, 1:d, :)
                A = Anew
            end
        end

        # Close every design that starts at bin `a` against the initial prior.
        while nactive > 0 && hs[nactive] == j + 1
            col = nactive
            L0w, logL0 = _whiten_rows(H * L0)
            initial = isempty(sm.B0) ? sm.x0 : sm.B0 * designs[order[col]].ux0
            r = L0w \ (view(A, :, col) - H * initial)
            total +=
                designs[order[col]].count *
                (value - logL0 - T(0.5) * (T(nrows) * log2π + sum(abs2, r) + dropped[col]))
            nactive -= 1
        end
    end
    return total
end

"""Copy `source`'s entry-prior parameters onto `target`'s (same switches)."""
function _sync_entries!(target::LQRStateModel, source::LQRStateModel)
    for (st, ss) in zip(target.switches, source.switches)
        (et, es) = (st.entry, ss.entry)
        (et === nothing || es === nothing || et === es) && continue
        copyto!(et.μ, es.μ)
        copyto!(et.K, es.K)
        copyto!(et.P, es.P)
    end
    return target
end

# ----------------------------------------------------------------------------
# M-step
# ----------------------------------------------------------------------------

"""
    _lqr_entry_prior_stats(sm, hs, e) -> _EntryStats

The entry prior's regression moments at switch `e` — `λ_{b+1}` on
`ψ = [1; x_b − r_b]`, `r_b = G u_b` the reference of the epoch being left — sliced
out of the joint moments `hs.entry_ww[e]`. Only the prior's blocks are filled;
the plant row is scored through [`_lqr_entry_Q`](@ref) instead.
"""
function _lqr_entry_prior_stats(
    sm::LQRStateModel{T}, hs::LQRSufficientStatistics{T}, e::Int
) where {T<:Real}
    n = _plant_dim(sm)
    d = 2n
    W = hs.entry_ww[e]
    m = size(W, 1) - 2d - 1
    st = _EntryStats(T, n, m)
    N = hs.entry_n[e]
    st.N[] = N
    iszero(N) && return st
    Tψ = zeros(T, n + 1, size(W, 1))
    Tψ[1, 2d + 1] = one(T)
    for i in 1:n
        Tψ[1 + i, i] = one(T)
    end
    if m > 0
        G = _gref_for(sm, _regime_at(sm, sm.switches[e].pos - 1))
        @views Tψ[2:(n + 1), (2d + 2):end] .= .-G
    end
    Tλ = zeros(T, n, size(W, 1))
    for i in 1:n
        Tλ[i, d + n + i] = one(T)
    end
    WTψ = W * transpose(Tψ)
    st.Sψψ .= Tψ * WTψ
    st.Sλψ .= Tλ * WTψ
    st.Sλλ .= Tλ * W * transpose(Tλ)
    st.Sψψ .= (st.Sψψ .+ transpose(st.Sψψ)) ./ 2
    st.Sλλ .= (st.Sλλ .+ transpose(st.Sλλ)) ./ 2
    return st
end

"""Every entry switch's prior statistics, pooled over the cells `sms`/`sufs`."""
function _lqr_pooled_entry_stats(sms, sufs, e::Int)
    return _pooled_entry_stats([
        _lqr_entry_prior_stats(sm, hs, e) for (sm, hs) in zip(sms, sufs)
    ])
end

function _entry_switches(sm::LQRStateModel)
    return [e for (e, s) in enumerate(sm.switches) if s.entry !== nothing]
end

"""
    _lqr_entry_update!(sms, sufs)

The joint objective's entry-prior update: each prior is the exact weighted
regression over every cell's entries (the priors are shared by every
`depends_on` variant). Caches are refreshed afterwards, since the entry
transitions read the priors.
"""
function _lqr_entry_update!(sms::AbstractVector{<:LQRStateModel}, sufs)
    sm1 = first(sms)
    for e in _entry_switches(sm1)
        _update_entry_prior!(sm1.switches[e].entry, _lqr_pooled_entry_stats(sms, sufs, e))
    end
    foreach(refresh!, sms)
    return nothing
end

"""
    _lqr_struct_score(ldss, sufs) -> T

What the guarded structural step must not lower: every cell's transition and
terminal Q-terms (bridges included) and its entry transitions, plus the
structural priors. The initial-state terms are left out — the structural step
does not touch them.
"""
function _lqr_struct_score(ldss, sufs)
    T = eltype(first(sufs).nk)
    total = zero(T)
    for (lds, hs) in zip(ldss, sufs)
        sm = lds.state_model
        total += _state_transition_Q(sm, hs) + _lqr_entry_Q(sm, hs)
    end
    return total +
           _grouped_state_prior_logdensity(ldss, [Int[] for _ in 1:4], T; init=false)
end

"""
    _lqr_entry_guarded(step!, ldss, sufs)

Run the structural step `step!()` as a generalized M-step that also answers for
the entry transitions: their plant row shares `A`, `S`, the plant drift and
noise, and their prior reads `Gref`, none of which the structural objective
sees. Keep the step only if the full state score did not fall; otherwise put
the structure back. (As for a switching model's entries, see
`_slds_entry_guarded`.)
"""
function _lqr_entry_guarded(step!, ldss, sufs)
    sms = [lds.state_model for lds in ldss]
    foreach(refresh!, sms)
    T = eltype(first(sufs).nk)
    q0 = _lqr_struct_score(ldss, sufs)
    snaps = [_lqr_struct_snapshot(sm) for sm in sms]
    step!()
    foreach(refresh!, sms)
    q1 = try
        _lqr_struct_score(ldss, sufs)
    catch err
        _lqr_rejectable(err) || rethrow()
        -T(Inf)
    end
    if !(q1 >= q0 - sqrt(eps(T)) * max(one(T), abs(q0)))
        @debug "structural step lowered the entry-inclusive score; reverted" q0 q1
        #= Cells of a `depends_on` model alias the arrays they share, and every
        cell's snapshot of a shared array holds the same values, so the order
        of the restores does not matter. =#
        for (sm, snap) in zip(sms, snaps)
            _lqr_struct_restore!(sm, snap)
        end
        foreach(refresh!, sms)
        return false
    end
    return true
end

"""
    _lqr_conditional_entry_update!(ldss, sufs; max_halvings=12) -> Bool

The entry priors' update under `condition_terminal`. They are part of the prior
the goals' normalizer integrates over, so with `q` fixed their share of the score
is `G = Σ_e E_q[log N(λ; μ + K ψ, P)] − log Z(μ, K, P)` with `log Z` exact. As for
a switching model (`_slds_conditional_entry_update!`): propose the joint
objective's regression, keep it if `G` rises; otherwise step along the gradient
of `G` (the data's entry statistics less the goal-conditioned prior's, which the
terminal probe supplies), halved until `G` rises by an Armijo fraction and `P`
stays positive definite. Returns whether the priors moved.
"""
function _lqr_conditional_entry_update!(ldss, sufs; max_halvings::Int=12)
    sms = [lds.state_model for lds in ldss]
    sm1 = first(sms)
    T = eltype(first(sufs).nk)
    es = _entry_switches(sm1)
    isempty(es) && return false
    data_st = Dict(e => _lqr_pooled_entry_stats(sms, sufs, e) for e in es)
    saved = Dict(e => deepcopy(sm1.switches[e].entry) for e in es)
    function restore!()
        for e in es
            ep, s0 = sm1.switches[e].entry, saved[e]
            ep.μ .= s0.μ
            ep.K .= s0.K
            ep.P .= s0.P
        end
        return foreach(refresh!, sms)
    end
    function score()
        data_part = sum(_entry_lambda_Q(sm1.switches[e].entry, data_st[e]) for e in es)
        isfinite(data_part) || return -T(Inf)
        logz = try
            foreach(refresh!, sms)
            sum(
                _lqr_terminal_logz_sum(sm, _lqr_terminal_designs(hs)) for
                (sm, hs) in zip(sms, sufs)
            )
        catch err
            _lqr_rejectable(err) || rethrow()
            return -T(Inf)
        end
        return data_part - logz
    end

    base = score()
    isfinite(base) || return false
    # The goal-conditioned prior's entry statistics at the incoming priors.
    probe_parts = Dict(e => _EntryStats{T}[] for e in es)
    for (sm, hs) in zip(sms, sufs)
        probe = _lqr_terminal_probe_cached(sm, hs)
        _lqr_sync_probe!(probe, sm)
        _lqr_probe_statistics!(probe)
        for e in es
            push!(
                probe_parts[e], _lqr_entry_prior_stats(probe.lds.state_model, probe.hs, e)
            )
        end
    end
    probe_st = Dict(e => _pooled_entry_stats(probe_parts[e]) for e in es)

    for e in es
        _update_entry_prior!(sm1.switches[e].entry, data_st[e])
    end
    gain = score() - base
    if isfinite(gain) && gain >= 0
        return true
    end
    restore!()

    directions = Dict{Int,Tuple{Matrix{T},Matrix{T}}}()
    slope = zero(T)
    for e in es
        ep, dst, pst = sm1.switches[e].entry, data_st[e], probe_st[e]
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
        directions[e] = (dΘ, dP)
    end
    if isfinite(slope) && slope > zero(T)
        step = one(T)
        for _ in 0:max_halvings
            feasible = true
            for e in es
                ep, s0 = sm1.switches[e].entry, saved[e]
                dΘ, dP = directions[e]
                ep.μ .= s0.μ .+ step .* dΘ[:, 1]
                ep.K .= s0.K .+ step .* dΘ[:, 2:end]
                P = s0.P .+ step .* dP
                ep.P .= (P .+ transpose(P)) ./ 2
                feasible &= isposdef(Symmetric(Matrix{T}(ep.P)))
            end
            if feasible
                gain = score() - base
                isfinite(gain) && gain >= T(1e-4) * step * slope && return true
            end
            step /= 2
        end
    end
    restore!()
    return false
end

"""Every cell's entry Q-term, at the parameters the cells hold when called."""
function _lqr_entry_score(sms, sufs)
    return () -> sum(_lqr_entry_Q(sm, hs) for (sm, hs) in zip(sms, sufs))
end

"""
    _refuse_slds_switches(slds)

Schedule boundaries and the reference gate belong to a single model's own known
epochs; a switching model's boundaries are [`set_boundaries!`](@ref), and its
kernels would otherwise apply a member's switches on top of them.
"""
function _refuse_slds_switches(slds)
    for (k, lds) in enumerate(slds.LDSs)
        sm = lds.state_model
        sm isa LQRStateModel || continue
        isempty(sm.switches) || throw(
            ArgumentError(
                "discrete state $k carries schedule boundaries; in a switching model " *
                "use `set_boundaries!` on the SLDS instead",
            ),
        )
        isempty(sm.gref_gate) || throw(
            ArgumentError(
                "discrete state $k carries a reference gate, which only a single " *
                "inverse-LQR model with a cost schedule supports",
            ),
        )
    end
    return nothing
end
