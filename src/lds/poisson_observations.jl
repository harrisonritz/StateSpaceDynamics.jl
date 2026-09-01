#=============================================================================
Poisson Observations

    Emission kernels: observation_loglikelihood!(cc, z, λ, om, x, y, t[, uy])
                      observation_gradient!(out, cc, buf, om, x, y, t[, uy])
                      observation_hessian!(out, cc, z, λ, om, x, y, t[, α])

    E-Step: Q_obs!(sws, lds, suf)

    M-Step: update_observation_model!(plds, tfs, y, sws_pool, w)
            — the row-wise Newton solver, in `poisson_emission_mstep.jl`;
              `_update_observation_model_lbfgs!` below is its reference.
=============================================================================#

"""
    Q_obs!(sws, lds, E_z, p_smooth, y; weights=nothing)

Allocation-free Poisson observation-model Q for a *single trial* (full
expected complete log-likelihood, including the `-log(y!)` normalizer):

```
Q = Σ_t w_t [ y_t' h_t  -  1' exp(h_t + ρ_t)  -  Σ_i log Γ(y_{t,i} + 1) ]
```

where `h_t = C · E[x_t] + d` and `ρ_{i,t} = ½ c_i' P_t c_i`, and `d` is the
canonical log-link Poisson intercept (free in ℝ). Including the factorial
term means `calculate_elbo` matches the Laplace-approximation marginal
log-likelihood at the EM fixed point, instead of being off by a fixed
data-only constant.

Both `h` and the variance correction `ρ` are formed for the whole trial at once
(see the batched kernels below), so this is a handful of `gemm`s rather than a
`tsteps`-long loop of BLAS-2 calls — `ρ` alone is `obs_dim · latent_dim² · tsteps`
work, and this runs once per trial per EM iteration on top of every M-step
evaluation.
"""
function Q_obs!(
    sws::SmoothWorkspace{T},                      # provides the Poisson Q_obs scratch
    plds::LinearDynamicalSystem{T,S,O},            # for obs_model.C, .d and .D
    E_z::AbstractMatrix{T},                       # state_dim × T
    p_smooth::AbstractArray{T,3},                 # state_dim × state_dim × T
    y::AbstractMatrix{T},                         # obs_dim × T
    uy::Union{Nothing,AbstractMatrix}=nothing;    # obs inputs (uy_dim × T) or nothing
    weights::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    C = plds.obs_model.C
    obs_dim, latent_dim = size(C)
    tsteps = size(y, 2)

    pb = poisson_batch!(sws, latent_dim, obs_dim, tsteps)
    Eta = _poisson_linear_predictor!(
        pb, C, plds.obs_model.d, plds.obs_model.D, E_z, uy, tsteps
    )
    Cpair = _poisson_pair_products!(pb, C, obs_dim)
    Ppack = _poisson_pack_cov!(pb, p_smooth, tsteps)

    # ρ[i, t] = ½ cᵢ' Pₜ cᵢ for the whole trial in one gemm.
    Rho = view(pb.Lam, 1:obs_dim, 1:tsteps)
    mul!(Rho, Cpair, Ppack, T(0.5), zero(T))

    Q_val = zero(T)
    @inbounds for t in 1:tsteps
        wt = weights === nothing ? one(T) : weights[t]
        ηcol = view(Eta, :, t)
        ρcol = view(Rho, :, t)
        acc = zero(T)
        @simd for i in 1:obs_dim
            yi = y[i, t]
            acc += yi * ηcol[i] - exp(ηcol[i] + ρcol[i]) - _log_factorial(yi)
        end
        Q_val += wt * acc
    end

    return Q_val
end

#=
`log Γ(y+1)` for observed counts. The emission normaliser is data-only — it
cancels out of every gradient — but it is what makes the reported ELBO the
Laplace marginal log-likelihood rather than that minus a constant, so it is
summed on every Q evaluation, over every neuron and bin, once per trial per EM
iteration. `loggamma` is expensive enough to show up in a profile of the E-step
at that volume.

Counts of 0 and 1 both give exactly zero, and at the bin widths this is used
with they are the overwhelming majority of the data, so short-circuiting them
skips almost every call. Anything else goes to `loggamma`, so the result is
bitwise what the plain call would have given.
=#
@inline function _log_factorial(y::T) where {T<:Real}
    (y == zero(T) || y == one(T)) && return zero(T)
    return loggamma(y + one(T))
end

"""
    _update_observation_model_lbfgs!(plds, tfs, y, sws_pool, w; uy=nothing)

Update the observation model parameters `[C d D]` of a PLDS model via LBFGS over
all `obs_dim · reg_dim` parameters at once.

Superseded by the row-wise Newton solver in `poisson_emission_mstep.jl`, which
maximises the same Q-function and is what `update_observation_model!` now calls.
Kept as the reference implementation the Newton solver is checked against, and
as a fallback for anyone who wants the old solver's exact iterates.
"""
function _update_observation_model_lbfgs!(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing;
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    plds.fit_bool[5] || return nothing

    sws = sws_pool[1]       # f(params) is sequential; one workspace suffices
    obs_dim = plds.obs_dim
    latent_dim = plds.latent_dim
    uy_dim = plds.uy_dim
    Dp1 = latent_dim + 1
    reg_dim = Dp1 + uy_dim          # columns of the stacked W = [C d D]
    n_W = obs_dim * reg_dim         # total params; identical to vec([C d D]) length

    # Param vector layout: vcat(vec(C), d, vec(D)) == vec([C d D]) column-major.
    params = vcat(vec(plds.obs_model.C), plds.obs_model.d, vec(plds.obs_model.D))

    # MN-only prior on the stacked emission matrix W = [C d D] (Poisson has no IW
    # counterpart since there is no observation-noise covariance):
    CD_prior = plds.obs_model.CD_prior

    function f(params::Vector{T})
        W_view = reshape(view(params, 1:n_W), obs_dim, reg_dim)
        @views C_view = W_view[:, 1:latent_dim]
        @views d_view = W_view[:, Dp1]
        @views D_view = W_view[:, (Dp1 + 1):reg_dim]

        copyto!(plds.obs_model.C, C_view)
        copyto!(plds.obs_model.d, d_view)
        copyto!(plds.obs_model.D, D_view)

        acc = zero(T)
        ntrials = length(tfs)

        for trial in 1:ntrials
            fs = tfs[trial]
            weights = isnothing(w) ? nothing : w[trial]
            uy_trial = isnothing(uy) ? nothing : uy[trial]
            # `x_smooth` is the same value as `E_z` for a Gaussian state model
            # — see `gradient_observation_model!` for context.
            acc += Q_obs!(
                sws, plds, fs.x_smooth, fs.p_smooth, y[trial], uy_trial; weights=weights
            )
        end

        f_prior = zero(T)
        if CD_prior !== nothing
            Wm = W_view .- CD_prior.M₀
            f_prior = T(0.5) * sum(Wm .* (Wm * CD_prior.Λ))
        end

        return -acc + f_prior
    end

    function g!(grad::Vector{T}, params::Vector{T})
        W_view = reshape(view(params, 1:n_W), obs_dim, reg_dim)
        @views C_view = W_view[:, 1:latent_dim]
        @views d_view = W_view[:, Dp1]
        @views D_view = W_view[:, (Dp1 + 1):reg_dim]
        gradient_observation_model!(grad, C_view, d_view, D_view, tfs, y, uy, sws_pool, w)
        if CD_prior !== nothing
            grad_W_view = reshape(view(grad, 1:n_W), obs_dim, reg_dim)
            grad_W_view .+= (W_view .- CD_prior.M₀) * CD_prior.Λ
        end
        return grad
    end

    opts = Optim.Options(;
        x_reltol=1e-8,
        x_abstol=1e-8,
        g_abstol=1e-8,
        f_reltol=1e-8,
        f_abstol=1e-8,
        iterations=200,
    )

    result = optimize(f, g!, params, LBFGS(; linesearch=HagerZhang()), opts)

    # surface a non-converged inner solve if applicable
    Optim.converged(result) || @warn(
        "Poisson emission M-step (LBFGS) did not converge; using last iterate",
        iterations = Optim.iterations(result),
        g_residual = Optim.g_residual(result),
    )

    # write final params back
    result_W = reshape(result.minimizer[1:n_W], obs_dim, reg_dim)
    @views plds.obs_model.C .= result_W[:, 1:latent_dim]
    @views plds.obs_model.d .= result_W[:, Dp1]
    @views plds.obs_model.D .= result_W[:, (Dp1 + 1):reg_dim]

    return nothing
end

# ============================================================================
# Batched (BLAS-3) Poisson emission kernels
#
# The per-timestep emission kernels above are the reference implementation and
# stay the interface a new observation model plugs into. Everything below is
# the same arithmetic reorganised so the O(obs_dim · latent_dim² · tsteps) work
# of a whole trial becomes one `gemm` instead of `tsteps` small BLAS-2 calls —
# which is where a Poisson fit spends most of its time, since `obs_dim` is the
# neuron count.
# ============================================================================

"""
    _poisson_pair_products!(pb, C, obs_dim)

Fill `pb.Cpair[n, p] = C[n, sym_i[p]] · C[n, sym_j[p]]` for the `nsym` distinct
entries of the symmetric `latent_dim` block, and return the active view.

This is the only place the emission matrix enters the batched curvature: with
it, `C' diag(λ_t) C` for every `t` at once is `Cpair' * Λ`.
"""
function _poisson_pair_products!(
    pb::PoissonBatchBuffers{T}, C::AbstractMatrix{T}, obs_dim::Int
) where {T<:Real}
    nsym = length(pb.sym_i)
    Cpair = view(pb.Cpair, 1:obs_dim, 1:nsym)
    @inbounds for p in 1:nsym
        Ci = view(C, :, pb.sym_i[p])
        Cj = view(C, :, pb.sym_j[p])
        col = view(Cpair, :, p)
        @simd for n in 1:obs_dim
            col[n] = Ci[n] * Cj[n]
        end
    end
    return Cpair
end

"""
    _poisson_linear_predictor!(pb, C, d, D_obs, x, uy, tsteps) -> view

Whole-trial linear predictor `η[:, t] = C x_t + d + D v_t` as one `gemm`,
written into `pb.Eta` and returned as an active view.
"""
function _poisson_linear_predictor!(
    pb::PoissonBatchBuffers{T},
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    D_obs::AbstractMatrix{T},
    x::AbstractMatrix{T},
    uy::Union{Nothing,AbstractMatrix},
    tsteps::Int,
) where {T<:Real}
    obs_dim = size(C, 1)
    Eta = view(pb.Eta, 1:obs_dim, 1:tsteps)
    mul!(Eta, C, view(x, :, 1:tsteps))
    if uy !== nothing && size(uy, 1) > 0
        mul!(Eta, D_obs, view(uy, :, 1:tsteps), one(T), one(T))
    end
    @inbounds for t in 1:tsteps
        col = view(Eta, :, t)
        @simd for i in 1:obs_dim
            col[i] += d[i]
        end
    end
    return Eta
end

"""
    _poisson_pack_cov!(pb, p_smooth, tsteps) -> view

Pack the smoothed covariances into `pb.Ppack[p, t]`, the symmetric-pair form
that pairs with `Cpair`: the diagonal entries as they are and the
off-diagonals doubled, so `dot(Cpair[n, :], Ppack[:, t]) == cₙ' P_t cₙ`.
"""
function _poisson_pack_cov!(
    pb::PoissonBatchBuffers{T}, p_smooth::AbstractArray{T,3}, tsteps::Int
) where {T<:Real}
    nsym = length(pb.sym_i)
    Ppack = view(pb.Ppack, 1:nsym, 1:tsteps)
    @inbounds for t in 1:tsteps, p in 1:nsym
        i = pb.sym_i[p]
        j = pb.sym_j[p]
        Ppack[p, t] = i == j ? p_smooth[i, j, t] : 2 * p_smooth[i, j, t]
    end
    return Ppack
end

"""
    _poisson_emission_hessian!(btd, pb, obs_model, x, uy, tsteps, weights)

Subtract the Poisson emission curvature `wₜ · C' diag(λₜ) C` from every diagonal
Hessian block, forming the whole trial in one `gemm` over the
`nsym = D(D+1)/2` distinct entries of the symmetric block rather than a
`tsteps`-long loop of `latent_dim² · obs_dim` scalar reductions.

`weights` is `nothing` for a single LDS and the regime's responsibilities
`γₖ(t)` for one regime of an SLDS — folded into the rates before the `gemm`, so
the weighted curvature costs the same as the unweighted one. Accumulates, so an
SLDS sums regimes by calling this once per regime.

Bit-for-bit this is a different summation order than the per-timestep kernel,
so results agree to rounding rather than exactly; the arithmetic, and the
`observation_hessian!` contract, are otherwise identical.
"""
function _poisson_emission_hessian!(
    btd::BlockTridiagonalWorkspace{T},
    pb::PoissonBatchBuffers{T},
    obs_model::PoissonObservationModel{T},
    x::AbstractMatrix{T},
    uy::Union{Nothing,AbstractMatrix},
    tsteps::Int,
    weights::Union{Nothing,AbstractVector{T}},
) where {T<:Real}
    C = obs_model.C
    obs_dim = size(C, 1)
    nsym = length(pb.sym_i)

    # λ[:, t] = exp(C x_t + d + D v_t) — the emission curvature's only
    # dependence on the current iterate — scaled by the timestep's weight.
    Lam = _poisson_linear_predictor!(pb, C, obs_model.d, obs_model.D, x, uy, tsteps)
    @inbounds for t in 1:tsteps
        wt = weights === nothing ? one(T) : weights[t]
        col = view(Lam, :, t)
        @simd for i in 1:obs_dim
            col[i] = wt * exp(col[i])
        end
    end

    Cpair = _poisson_pair_products!(pb, C, obs_dim)
    Hsym = view(pb.Hsym, 1:nsym, 1:tsteps)
    mul!(Hsym, transpose(Cpair), Lam)

    @inbounds for t in 1:tsteps
        Ht = btd.H_diag[t]
        for p in 1:nsym
            i = pb.sym_i[p]
            j = pb.sym_j[p]
            v = Hsym[p, t]
            Ht[i, j] -= v
            i == j || (Ht[j, i] -= v)
        end
    end
    return nothing
end

"""
    hessian!(sws, plds, x, y[, uy])

Poisson specialisation of the generic `hessian!`: the state-side blocks are
unchanged, and the emission curvature goes through the batched
[`_poisson_emission_hessian!`](@ref).
"""
function hessian!(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    tsteps = size(y, 2)
    obs_dim, latent_dim = size(lds.obs_model.C)

    _state_hessian_blocks!(sws.btd, sws.consts, tsteps)

    pb = poisson_batch!(sws, latent_dim, obs_dim, tsteps)
    _poisson_emission_hessian!(sws.btd, pb, lds.obs_model, x, uy, tsteps, nothing)

    return nothing
end

"""
    observation_loglikelihood!(cc, z, λ, obs_model, x, y, t[, uy])

Poisson emission term: with rate `λ = exp(Cx_t + d + D v_t)`,
`log p(y_t|x_t) = y⋅log(λ) - sum(λ) - sum(log(y!))`. `z` and `λ` are `obs_dim`
scratch vectors for the linear predictor and the rate. `uy` (optional) supplies
the observation input `v_t`; `nothing` or a zero-row matrix skips the `D v_t`
term. The cache argument is unused (no covariance term).
"""
function observation_loglikelihood!(
    ::SmoothConstants{T},
    z::AbstractVector{T},
    λ::AbstractVector{T},
    om::PoissonObservationModel{T0},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T0},
    t::Int,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real}
    C = om.C
    d = om.d

    # z = Cx + d (+ D v) ; λ = exp(z)
    @views mul!(z, C, x[:, t])
    if uy !== nothing
        @views mul!(z, om.D, uy[:, t], one(T), one(T))
    end
    z .+= d
    @. λ = exp(z)

    # y⋅z - λ - log(y!)  (loggamma(n+1) = log(n!) for real n≥0)
    yt = view(y, :, t)
    return dot(yt, z) - sum(λ) - sum(yi -> loggamma(yi + one(T)), yt)
end

"""
    observation_gradient!(out, cc, buf, obs_model, x, y, t[, uy])

Poisson emission gradient w.r.t. the latent `x_t`: `out = C'(y_t - λ_t)` with
`λ_t = exp(Cx_t + d + D v_t)`. The `D v_t` term is constant in `x_t`, so it
enters only through the rate `λ`. The cache argument is unused (no covariance
term); `uy` (optional) supplies `v_t`.
"""
function observation_gradient!(
    out::AbstractVector{T},
    ::SmoothConstants{T},
    buf::AbstractVector{T},
    om::PoissonObservationModel{T0},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T0},
    t::Int,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real}
    C = om.C
    d = om.d
    @views mul!(buf, C, x[:, t])
    if uy !== nothing
        @views mul!(buf, om.D, uy[:, t], one(T), one(T))
    end
    @views buf .= y[:, t] .- exp.(buf .+ d)
    return mul!(out, C', buf)
end

"""
    observation_hessian!(out, cc, z, λ, obs_model, x, y, t[, α, uy])

Poisson emission curvature: `out .+= α .* (-C' diag(λ_t) C)` with
`λ_t = exp(C x_t + d + D v_t)` — independent of `y` for the canonical log link.
The `D v_t` term enters only through the rate `λ`. `z` and `λ` are `obs_dim`
scratch for the linear predictor and the rate; `cc` is unused (no covariance in
the emission term); `uy` (optional) supplies `v_t`.
"""
function observation_hessian!(
    out::AbstractMatrix{T},
    ::SmoothConstants{T},
    z::AbstractVector{T},
    λ::AbstractVector{T},
    om::PoissonObservationModel{T0},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T0},
    t::Int,
    α::T=one(T),
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real}
    C = om.C
    d = om.d
    obs_dim, latent_dim = size(C)

    @views mul!(z, C, x[:, t])
    if uy !== nothing
        @views mul!(z, om.D, uy[:, t], one(T), one(T))
    end
    @. λ = exp(z + d)

    # out .+= α * (-C' diag(λ) C), allocation-free (O(latent² · obs) per call).
    for j in 1:latent_dim, i in 1:latent_dim
        acc = zero(T)
        for k in 1:obs_dim
            acc += C[k, i] * λ[k] * C[k, j]
        end
        out[i, j] -= α * acc
    end
    return out
end
