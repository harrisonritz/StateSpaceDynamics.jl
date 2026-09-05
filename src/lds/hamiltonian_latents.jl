#=============================================================================
Hamiltonian (inverse-LQR) latents — E-step.

The latent state is `z_t = [x_t; λ_t]` and the forward transition is
`z_{t+1} = M_{k(t)} z_t + b + B u_t + w`, `w ~ N(0, Qfwd)`, with `M_k`
symplectic and `k(t)` the cost regime in force at `t`. Structurally this is an
ordinary linear-Gaussian chain, so every kernel below mirrors its
`GaussianStateModel` counterpart in `continuous_latents.jl`; what differs is

  * the transition matrix is looked up per timestep (`cache.M[k]`) rather than
    read off one `A`, and
  * a model with `terminal` set carries one extra Gaussian factor at `t = T`,
    the soft terminal condition `0 = λ_T − Q_f x_T − h_f + ε_f`.

    Log-likelihood: state_loglikelihood!(cc, dxt, tmp, lds, x, t[, ux])
    Gradient:       _state_gradient!(grad, ws, lds, x[, ux])
    Hessian:        _state_hessian_blocks!(btd, cc, sm, tsteps)
    Constants:      _compute_state_constants!(cc, sm)
=============================================================================#

"""
    _ham(lds) -> HamiltonianStateModel

The state model of a Hamiltonian LDS, with its concrete type asserted so the
kernels below stay on the typed path.
"""
@inline _ham(lds::LinearDynamicalSystem{T,S}) where {T,S<:HamiltonianStateModel} =
    lds.state_model::S

"""
    _hamiltonian_lengths_ok(sm, tsteps)

Check that a cost schedule covers every timestep of the longest trial. Called at
each fitting / smoothing entry point, where the trial lengths are known — the
model itself is built without reference to any dataset.
"""
function _hamiltonian_lengths_ok(sm::HamiltonianStateModel, tsteps::AbstractVector{Int})
    isempty(sm.schedule) && return nothing
    T_max = maximum(tsteps)
    length(sm.schedule) >= T_max || throw(
        DimensionMismatchError(
            "cost schedule length (must cover the longest trial)",
            T_max,
            length(sm.schedule),
        ),
    )
    return nothing
end

"""
    _transition_residual!(out, lds, x, t[, ux])

`z_t − M_{k(t-1)} z_{t-1} − b − B u_{t-1}` for a Hamiltonian state model: the
same residual as the Gaussian case, with the regime-dependent transition.
Requires `t ≥ 2`.
"""
@inline function _transition_residual!(
    out::AbstractVector{T},
    lds::LinearDynamicalSystem{T0,S,O},
    x::AbstractMatrix{T},
    t::Int,
    ux::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real,S<:HamiltonianStateModel{T0},O<:AbstractObservationModel{T0}}
    sm = _ham(lds)
    c = sm.cache
    k = _regime(sm, t - 1)
    @views mul!(out, c.M[k], x[:, t - 1])
    if ux !== nothing && size(c.Bfwd[k], 2) > 0
        @views mul!(out, c.Bfwd[k], ux[:, t - 1], one(T), one(T))
    end
    @views out .= x[:, t] .- out .- c.bfwd
    return out
end

"""
    _terminal_residual!(out, sm, x[, ux])

`Λf z_T + Q_f G_r u_T − h_f` (length `n`), the residual of the soft terminal
condition `λ_T = Q_f (x_T − r_T) + h_f`. The input term is the terminal
*reference*: a reach is scored against where the target was, not against the
origin. Only meaningful when `sm.terminal` is set.
"""
@inline function _terminal_residual!(
    out::AbstractVector{T},
    sm::HamiltonianStateModel,
    x::AbstractMatrix{T},
    ux::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real}
    tsteps = size(x, 2)
    @views mul!(out, sm.cache.Lf, x[:, tsteps])
    if ux !== nothing && size(sm.cache.Ftrm, 2) > 0
        @views mul!(out, sm.cache.Ftrm, ux[:, tsteps], one(T), one(T))
    end
    out .-= sm.hf
    return out
end

"""
    state_loglikelihood!(cc, dxt, tmp, lds, x, t[, ux])

State-model contribution to the complete-data log-likelihood at timestep `t` for
a Hamiltonian LDS:

- `t == 1`: the initial-state term, as in the Gaussian case
- `t ≥ 2`:  `cQ − ½‖Qfwd^{-1/2}(z_t − M_{k(t-1)} z_{t-1} − b − B u_{t-1})‖²`
- `t == T` and `sm.terminal`: plus `cF − ½‖Σf^{-1/2}(Λf z_T − h_f)‖²`

`cc` supplies the initial-state Cholesky; everything transition-side comes from
the model's own [`HamiltonianCache`](@ref), which is shared across the trial
workspaces rather than recomputed per trial.
"""
function state_loglikelihood!(
    cc::SmoothConstants{T},
    dxt::AbstractVector{T},
    tmp::AbstractVector{T},
    lds::LinearDynamicalSystem{T0,S,O},
    x::AbstractMatrix{T},
    t::Int,
    ux::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real,S<:HamiltonianStateModel{T0},O<:AbstractObservationModel{T0}}
    sm = _ham(lds)
    c = sm.cache
    tsteps = size(x, 2)

    total = if t == 1
        @views dxt .= x[:, 1] .- sm.x0
        _whiten!(cc.P0_PD.chol, dxt)
        cc.cP0 - T(0.5) * sum(abs2, dxt)
    else
        _transition_residual!(tmp, lds, x, t, ux)
        _whiten!(c.Qfwd.chol, tmp)
        T(c.cQ) - T(0.5) * sum(abs2, tmp)
    end

    if sm.terminal && t == tsteps
        n = _plant_dim(sm)
        rf = view(tmp, 1:n)
        _terminal_residual!(rf, sm, x, ux)
        _whiten!(c.Sf_PD.chol, rf)
        total += T(c.cF) - T(0.5) * sum(abs2, rf)
    end

    return total
end

"""
    _state_gradient!(grad, ws, lds, x[, ux])

State half of the complete-data log-likelihood gradient for a Hamiltonian LDS:

- `grad[:, 1] = −P0⁻¹(z₁ − x0) + M_{k(1)}ᵀ Qfwd⁻¹ r₂`
- `grad[:, t] = −Qfwd⁻¹ r_t + M_{k(t)}ᵀ Qfwd⁻¹ r_{t+1}`
- `grad[:, T] = −Qfwd⁻¹ r_T` (plus `−Λfᵀ Σf⁻¹ (Λf z_T − h_f)` when terminal)

with `r_t = z_t − M_{k(t-1)} z_{t-1} − b − B u_{t-1}`. Note the *outgoing*
factor at step `t` uses regime `k(t)` while the *incoming* one used `k(t-1)`;
that asymmetry is the whole of the time-varying-cost bookkeeping.
"""
function _state_gradient!(
    grad::AbstractMatrix{T},
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    ux::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    tsteps = size(x, 2)
    sm = _ham(lds)
    c = sm.cache

    neg_P0_inv = ws.consts.x_t     # −P0⁻¹ (initial-state term is model-agnostic)
    neg_Q_inv = c.negQinv

    dxt = ws.opt.dxt
    dxt_next = ws.opt.dxt_next
    tmp2 = ws.opt.tmp2
    tmp3 = ws.opt.tmp3

    # t = 1: prior + the outgoing factor at t = 2.
    @views dxt .= x[:, 1] .- sm.x0
    mul!(tmp3, neg_P0_inv, dxt)
    _transition_residual!(dxt_next, lds, x, 2, ux)
    mul!(tmp2, c.MtQinv[_regime(sm, 1)], dxt_next)
    @views grad[:, 1] .= tmp2 .+ tmp3

    @views for t in 2:(tsteps - 1)
        _transition_residual!(dxt, lds, x, t, ux)
        mul!(tmp3, neg_Q_inv, dxt)
        _transition_residual!(dxt_next, lds, x, t + 1, ux)
        mul!(tmp2, c.MtQinv[_regime(sm, t)], dxt_next)
        grad[:, t] .= tmp3 .+ tmp2
    end

    _transition_residual!(dxt, lds, x, tsteps, ux)
    mul!(tmp3, neg_Q_inv, dxt)
    @views grad[:, tsteps] .= tmp3

    if sm.terminal
        n = _plant_dim(sm)
        rf = view(dxt, 1:n)
        _terminal_residual!(rf, sm, x, ux)
        # −Λfᵀ Σf⁻¹ r, accumulated onto the last column.
        @views mul!(grad[:, tsteps], c.LtSinv, rf, -one(T), one(T))
    end

    return grad
end

"""
    _state_hessian_blocks!(btd, cc, sm::HamiltonianStateModel, tsteps)

Per-timestep state-side Hessian blocks. Unlike the Gaussian case there is no one
template to copy: `H_sub[i]` and the `MᵀQ⁻¹M` half of `H_diag[t]` depend on the
regime of the transition leaving `t`, so each is looked up. The terminal factor
adds `−Λfᵀ Σf⁻¹ Λf` to the last diagonal block.
"""
function _state_hessian_blocks!(
    btd, cc::SmoothConstants{T}, sm::HamiltonianStateModel, tsteps::Int
) where {T<:Real}
    c = sm.cache
    for i in 1:(tsteps - 1)
        k = _regime(sm, i)
        copyto!(btd.H_sub[i], c.QinvM[k])
        copyto!(btd.H_super[i], transpose(c.QinvM[k]))
    end

    btd.H_diag[1] .= c.negMtQinvM[_regime(sm, 1)] .+ cc.x_t
    for t in 2:(tsteps - 1)
        btd.H_diag[t] .= c.negMtQinvM[_regime(sm, t)] .+ c.negQinv
    end
    btd.H_diag[tsteps] .= c.negQinv
    sm.terminal && (btd.H_diag[tsteps] .+= c.negLtSL)

    return nothing
end

"""
    _compute_state_constants!(cc, sm::HamiltonianStateModel)

Fill the state half of a [`SmoothConstants`](@ref) for a Hamiltonian model.

Only the initial-state terms (`P0_PD`, `cP0`, `x_t = −P0⁻¹`) are actually read by
the kernels — everything transition-side lives on the model's own cache, shared
across workspaces instead of recomputed per trial. `Q_PD` and `cQ` are filled
with the *forward* noise anyway so that generic consumers of a `SmoothConstants`
(and `logdet(cc.Q_PD)`) see a consistent picture; the regime-1 Hessian templates
are filled for the same reason.
"""
function _compute_state_constants!(
    cc::SmoothConstants{WT}, sm::HamiltonianStateModel{T}
) where {WT<:Real,T<:Real}
    d = _state_latent_dim(sm)
    c = sm.cache

    P0_w = convert(Matrix{WT}, Matrix(sm.P0))
    cc.P0_PD = PDMat(Symmetrize!(P0_w))
    cc.Q_PD = PDMat(convert(Matrix{WT}, Matrix(c.Qfwd)))

    copyto!(cc.x_t, cc.I_mat)
    ldiv!(cc.P0_PD.chol, cc.x_t)
    cc.x_t .*= -one(WT)

    copyto!(cc.tmp_QA, c.M[1])
    ldiv!(cc.Q_PD.chol, cc.tmp_QA)
    copyto!(cc.A_inv_Q, cc.tmp_QA')
    copyto!(cc.H_sub_entry, cc.tmp_QA)
    copyto!(cc.H_super_entry, cc.tmp_QA')
    copyto!(cc.xt_given_xt_1, c.negQinv)
    copyto!(cc.xt1_given_xt, c.negMtQinvM[1])

    cc.cP0 = -WT(0.5) * (WT(d) * log(WT(2π)) + logdet(cc.P0_PD))
    cc.cQ = WT(c.cQ)

    return nothing
end

"""
    _state_prior_logdensity(lds, sws) -> T

`log p(θ)` for a Hamiltonian state model. Only the initial-state priors exist
here: there is no Inverse-Wishart prior on `Σ` (the M-step profiles it out) and
no matrix-normal prior on the structural block, whose entries are shared between
`𝓔`'s (1,1) and (2,2) blocks and so do not form a free regression matrix.
"""
function _state_prior_logdensity(
    lds::LinearDynamicalSystem{T,S,O}, ::Union{Nothing,SmoothWorkspace{T}}
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    sm = lds.state_model
    total = zero(T)
    sm.P0_prior === nothing || (total += iw_logprior_term(sm.P0, sm.P0_prior))
    if sm.x0_prior !== nothing
        total += mn_logprior_term(reshape(sm.x0, :, 1), sm.P0, sm.x0_prior)
    end
    return total
end
