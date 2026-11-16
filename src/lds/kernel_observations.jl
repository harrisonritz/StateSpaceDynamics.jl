#=============================================================================
Constrained and kernel-prior Gaussian emissions: `C = Φ H`, `vec H ~ N(0, I ⊗ Σ)`.

    E-step hooks: _compute_obs_constants!, observation_loglikelihood!,
                  observation_gradient!, observation_hessian!
    q(H):         _loading_posterior, _loading_moment, _loading_excess
    M-step:       _update_loading!, update_R!, _update_kernel_weights!
    Bound:        Q_obs!, _loading_bound, elbo!

`q(x)` comes from the existing smoother: the expected emission log-likelihood
under `q(H)` is the Gaussian one at the mean loading `Ĉ = Φ Ĥ` minus
`½ x'Δx`, `Δ = E[H'Φ'R⁻¹ΦH] − Ĉ'R⁻¹Ĉ` (`obs_model.excess`), so the smoother
only needs `Δ` added to the emission precision and `Δ x` to its gradient.
=============================================================================#

const _StructuredLDS{T} = LinearDynamicalSystem{
    T,<:AbstractGaussianStateModel{T},<:StructuredGaussianObservationModel{T}
}

_has_kernel_prior(om::StructuredGaussianObservationModel) = om.prior !== nothing
function _loading_dim(om::StructuredGaussianObservationModel)
    return om.basis === nothing ? size(om.C, 1) : size(om.basis, 2)
end

# Φ X, Φ'X and H = Φ⁺C, with Φ = I when there is no basis.
_basis_mul(::Nothing, X::AbstractMatrix) = X
_basis_mul(Φ::AbstractMatrix, X::AbstractMatrix) = Φ * X
_basis_tmul(::Nothing, X::AbstractMatrix) = X
_basis_tmul(Φ::AbstractMatrix, X::AbstractMatrix) = Φ' * X
_loading_coefficients(::Nothing, C::AbstractMatrix) = Matrix(C)
_loading_coefficients(Φ::AbstractMatrix, C::AbstractMatrix) = Φ \ C

"""
    emission_posterior(obs_model) -> LoadingPosterior or nothing

The posterior `q(H)` of a kernel-prior emission's loading (`C = basis * H`; `H`
is `C` itself without a basis), as left by the last fit: its `mean`, and the
factors of its covariance ([`LoadingPosterior`](@ref)). `nothing` before a fit,
and for an emission without a prior, whose `H` is a point estimate.
"""
emission_posterior(om::StructuredGaussianObservationModel) = om.posterior

"""
    kernel_weights(obs_model) -> Vector{NamedTuple}

The fitted kernel weights of a kernel-prior emission: [`kernel_weights`](@ref)
of its prior.
"""
function kernel_weights(om::StructuredGaussianObservationModel)
    prior = om.prior
    prior === nothing && throw(ArgumentError("this emission has no kernel prior"))
    return kernel_weights(prior)
end

# ----------------------------------------------------------------------------
# Model plumbing
# ----------------------------------------------------------------------------

function _extract_obs_params(om::StructuredGaussianObservationModel)
    return (C=om.C, R=om.R, d=om.d, D=om.D)
end

# The mean pass runs per trial: the batched one has no `Δ x` term.
function _supports_batched(
    ::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel{T},O<:StructuredGaussianObservationModel{T}}
    return false
end

function _validate_obs_model(
    om::StructuredGaussianObservationModel{T}, obs_dim::Int, latent_dim::Int
) where {T<:Real}
    size(om.C) == (obs_dim, latent_dim) ||
        throw(DimensionMismatchError("C matrix", (obs_dim, latent_dim), size(om.C)))
    size(om.R) == (obs_dim, obs_dim) ||
        throw(DimensionMismatchError("R matrix", (obs_dim, obs_dim), size(om.R)))
    issymmetric(om.R) || throw(NotSymmetricError("R matrix", maximum(abs.(om.R - om.R'))))
    isposdef(om.R) || throw(NotPositiveDefiniteError("R matrix", minimum(eigvals(om.R))))
    length(om.d) == obs_dim ||
        throw(DimensionMismatchError("observation bias d", obs_dim, length(om.d)))
    size(om.D, 1) == obs_dim ||
        throw(DimensionMismatchError("D rows", obs_dim, size(om.D, 1)))
    om.gauge in (:bound, :moment, :none) ||
        throw(ArgumentError("gauge must be :bound, :moment or :none; got :$(om.gauge)"))
    om.weight_iters >= 1 || throw(ArgumentError("weight_iters must be at least 1"))
    om.tree_every >= 1 || throw(ArgumentError("tree_every must be at least 1"))
    basis = om.basis
    if basis !== nothing
        size(basis, 1) == obs_dim ||
            throw(DimensionMismatchError("basis rows", obs_dim, size(basis, 1)))
        all(isfinite, basis) || throw(ArgumentError("the basis has non-finite entries"))
        rank(basis) == size(basis, 2) ||
            throw(ArgumentError("the basis must have full column rank"))
        outside = om.C - basis * (basis \ om.C)
        norm(outside) <= sqrt(eps(T)) * max(norm(om.C), one(T)) ||
            throw(ArgumentError("the starting C must lie in the span of the basis"))
    end
    prior = om.prior
    if prior !== nothing
        r = _loading_dim(om)
        prior.dim == r || throw(
            DimensionMismatch(
                "the kernel prior is over $(prior.dim) dimensions; the loading has $r " *
                "(the columns of the basis, or the observations without one)",
            ),
        )
    end
    size(om.excess) == (latent_dim, latent_dim) ||
        throw(DimensionMismatchError("excess", (latent_dim, latent_dim), size(om.excess)))
    return nothing
end

# ----------------------------------------------------------------------------
# E-step hooks
# ----------------------------------------------------------------------------

function _compute_obs_constants!(
    cc::SmoothConstants{WT}, om::StructuredGaussianObservationModel{T}
) where {WT<:Real,T<:Real}
    C = om.C
    obs_dim = size(C, 1)
    R_w = convert(Matrix{WT}, om.R)
    cc.R_PD = PDMat(Symmetrize!(R_w))
    Rchol = cc.R_PD.chol
    copyto!(cc.tmp_RC, C)
    ldiv!(Rchol, cc.tmp_RC)
    copyto!(cc.C_inv_R, cc.tmp_RC')
    # −(C'R⁻¹C + Δ): the expected emission precision under q(H)
    mul!(cc.yt_given_xt, C', cc.tmp_RC)
    cc.yt_given_xt .= .-cc.yt_given_xt .- om.excess
    cc.cR = -WT(0.5) * (WT(obs_dim) * log(WT(2π)) + logdet(cc.R_PD))
    return nothing
end

function observation_loglikelihood!(
    cc::SmoothConstants{T},
    dyt::AbstractVector{T},
    ::AbstractVector{T},
    om::StructuredGaussianObservationModel{T0},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T0},
    t::Int,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real}
    @views mul!(dyt, om.C, x[:, t])
    if uy !== nothing
        @views mul!(dyt, om.D, uy[:, t], one(T), one(T))
    end
    @views dyt .= y[:, t] .- dyt .- om.d
    _whiten!(cc.R_PD.chol, dyt)
    xt = view(x, :, t)
    return cc.cR - T(0.5) * sum(abs2, dyt) - T(0.5) * dot(xt, om.excess, xt)
end

function observation_gradient!(
    out::AbstractVector{T},
    cc::SmoothConstants{T},
    buf::AbstractVector{T},
    om::StructuredGaussianObservationModel{T0},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T0},
    t::Int,
    uy::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real}
    @views mul!(buf, om.C, x[:, t])
    if uy !== nothing
        @views mul!(buf, om.D, uy[:, t], one(T), one(T))
    end
    @views buf .= y[:, t] .- buf .- om.d
    mul!(out, cc.C_inv_R, buf)
    @views mul!(out, om.excess, x[:, t], -one(T), one(T))
    return out
end

function observation_hessian!(
    out::AbstractMatrix{T},
    cc::SmoothConstants{T},
    ::AbstractVector{T},
    ::AbstractVector{T},
    ::StructuredGaussianObservationModel{T0},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T0},
    t::Int,
    α::T=one(T),
    ::Union{Nothing,AbstractMatrix}=nothing,
) where {T<:Real,T0<:Real}
    @. out += α * cc.yt_given_xt
    return out
end

# ----------------------------------------------------------------------------
# q(H)
# ----------------------------------------------------------------------------

# The symmetric square root of a symmetric positive definite matrix.
function _sym_sqrt(Σ::AbstractMatrix{T}) where {T<:Real}
    values, vectors = eigen(Symmetric(Matrix(Σ)))
    values[1] > 0 ||
        throw(ArgumentError("the kernels' covariance is not positive definite"))
    return Matrix(Symmetric(vectors * Diagonal(sqrt.(values)) * vectors'))
end

"""
    _loading_posterior(Σ, A1, rhs, Sxx_c, Sxx) -> LoadingPosterior

`q(H)` for `vec H ~ N(0, I ⊗ Σ)` and a Gaussian likelihood with precision
`Sxx ⊗ A1` (`A1 = Φ'R⁻¹Φ`, `Sxx = Σ_t E[x_t x_t']`), the intercept and input
coefficients profiled out: the mean solves `Σ⁻¹Ĥ + A1 Ĥ Sxx_c = rhs` with
`Sxx_c` and `rhs = Φ'R⁻¹Syx_c` the moments with `[1; u]` partialled out, and
the covariance is `(I ⊗ Σ⁻¹ + Sxx ⊗ A1)⁻¹`.

With `Σ^½ A1 Σ^½ = P λ P'` and `Sxx = Q μ Q'` the entries of `E` in
`H = Ĥ + Σ^½ P E Q'` are independent with variances `1 / (1 + λ_l μ_j)`; the
mean is the same solve in the eigenbasis of `Sxx_c`.
"""
function _loading_posterior(
    Σ::AbstractMatrix{T},
    A1::AbstractMatrix{T},
    rhs::AbstractMatrix{T},
    Sxx_c::AbstractMatrix{T},
    Sxx::AbstractMatrix{T},
) where {T<:Real}
    root = _sym_sqrt(Σ)
    λ, P = eigen(Symmetric(root * A1 * root))
    λ .= max.(λ, zero(T))
    left = root * P
    μc, Qc = eigen(Symmetric(Matrix(Sxx_c)))
    μc .= max.(μc, zero(T))
    mean = left * ((left' * rhs * Qc) ./ (one(T) .+ λ .* μc')) * Qc'
    μ, Q = eigen(Symmetric(Matrix(Sxx)))
    μ .= max.(μ, zero(T))
    return LoadingPosterior{T}(mean, left, Q, one(T) ./ (one(T) .+ λ .* μ'))
end

"""
    _loading_moment(q, S) -> Matrix

`E[H S H']` (`r × r`) for a symmetric `k × k` `S`.
"""
function _loading_moment(q::LoadingPosterior{T}, S::AbstractMatrix{T}) where {T<:Real}
    s = [dot(view(q.right, :, j), S, view(q.right, :, j)) for j in axes(q.right, 2)]
    return q.mean * S * q.mean' + q.left * Diagonal(q.var * s) * q.left'
end

# E[H H'], the second moment the kernel weights are fitted to (times k).
function _loading_moment(q::LoadingPosterior{T}) where {T<:Real}
    return _loading_moment(q, Matrix{T}(I, size(q.mean, 2), size(q.mean, 2)))
end

"""
    _loading_excess(q, A1) -> Matrix

`E[H'A1 H] − Ĥ'A1 Ĥ` (`k × k`): what the uncertainty of `H` adds to the
emission precision `Ĉ'R⁻¹Ĉ` (`A1 = Φ'R⁻¹Φ`).
"""
function _loading_excess(q::LoadingPosterior{T}, A1::AbstractMatrix{T}) where {T<:Real}
    g = [dot(view(q.left, :, l), A1, view(q.left, :, l)) for l in axes(q.left, 2)]
    Δ = q.right * Diagonal(vec(g' * q.var)) * q.right'
    return Matrix(Symmetric((Δ .+ Δ') ./ 2))
end

"""
    _loading_bound(q, Σ) -> T

`E_q[log p(H | Σ)] + H[q(H)]` for `vec H ~ N(0, I ⊗ Σ)`: minus the divergence
of `q(H)` from the prior.
"""
function _loading_bound(q::LoadingPosterior{T}, Σ::AbstractMatrix{T}) where {T<:Real}
    r, k = size(q.mean)
    F = cholesky(Symmetric(Matrix(Σ)))
    prior_term = -k * logdet(F) / 2 - tr(F \ _loading_moment(q)) / 2
    entropy =
        (k * logabsdet(q.left)[1] * 2 + r * logabsdet(q.right)[1] * 2 + sum(log, q.var)) / 2
    return prior_term + entropy + T(r * k) / 2
end

# ----------------------------------------------------------------------------
# M-step
# ----------------------------------------------------------------------------

# Φ'R⁻¹Φ and Φ'R⁻¹.
function _basis_precision(om::StructuredGaussianObservationModel{T}) where {T<:Real}
    Rchol = cholesky(Symmetric(Matrix(om.R)))
    Φ = om.basis
    weighted = Φ === nothing ? inv(Rchol) : (Rchol \ Φ)'        # Φ'R⁻¹, (r × p)
    A1 = Φ === nothing ? Matrix(weighted) : weighted * Φ
    return Matrix(Symmetric((A1 .+ A1') ./ 2)), Matrix(weighted)
end

# The latents' block of the emission regressors `[x; 1; u]`, and that block with
# the others partialled out, of `Σ reg reg'` and `Σ reg y'`.
function _loading_statistics(lds::_StructuredLDS{T}, suf::SufficientStatistics{T}) where {T}
    k = lds.latent_dim
    G = suf.obs_xx[].mat
    XY = suf.obs_xy
    rest = (k + 1):size(G, 1)
    Sxx = Matrix(Symmetric(G[1:k, 1:k]))
    coupling = G[1:k, rest] / Symmetric(G[rest, rest])
    Sxx_c = Matrix(Symmetric(Sxx - coupling * G[rest, 1:k]))
    Sxy_c = XY[1:k, :] - coupling * XY[rest, :]                   # (k × p)
    return Sxx, Sxx_c, Sxy_c
end

"""
    _update_loading!(lds, suf)

The emission's loading and `[d D]` given `R`: `q(H)` under the kernel prior (or
the maximiser without one), `C = Φ Ĥ`, and then `[d D]` from their own normal
equations given `C`.
"""
function _update_loading!(lds::_StructuredLDS{T}, suf::SufficientStatistics{T}) where {T}
    om = lds.obs_model
    k = lds.latent_dim
    Sxx, Sxx_c, Sxy_c = _loading_statistics(lds, suf)
    A1, weighted = _basis_precision(om)
    rhs = weighted * Sxy_c'                                       # Φ'R⁻¹Syx_c
    prior = om.prior
    kq, ka = _projected_q_prior(lds), _projected_a_prior(lds)
    if prior === nothing
        H = (Symmetric(A1) \ rhs) / Symmetric(Sxx_c)
        if kq !== nothing || ka !== nothing
            H = _polish_loading(lds, kq, ka, H, A1, rhs, Sxx_c)
        end
    else
        q = _loading_posterior(kernel_covariance(prior), A1, rhs, Sxx_c, Sxx)
        om.posterior = q
        H = q.mean
    end
    copyto!(om.C, _basis_mul(om.basis, H))
    G = suf.obs_xx[].mat
    rest = (k + 1):size(G, 1)
    V = (transpose(suf.obs_xy[rest, :]) .- om.C * G[1:k, rest]) / Symmetric(G[rest, rest])
    copyto!(om.d, view(V, :, 1))
    lds.uy_dim > 0 && copyto!(om.D, view(V, :, 2:(1 + lds.uy_dim)))
    kq === nothing || _set_loading!(kq, H)
    ka === nothing || (ka.loading = Matrix{T}(H))
    return nothing
end

# ----------------------------------------------------------------------------
# A kernel prior on Q seen through the learned loading
# ----------------------------------------------------------------------------

function _projected_q_prior(lds::_StructuredLDS)
    prior = lds.state_model.Q_prior
    return prior isa KernelIWPrior && _is_projected(prior) ? prior : nothing
end

function _projected_a_prior(lds::_StructuredLDS)
    prior = lds.state_model.A_prior
    ok = prior isa KernelMNPrior && !isempty(prior.loading) && prior.posterior !== nothing
    return ok ? prior : nothing
end

"""
    _projected_dynamics_term(H, ka, FU, FV) -> (g, ∂g/∂H)

The part of `E_q[log p(A | U, V, m)]` that involves the loading `H`, with the
row covariance `U` and the column covariance `V` the kernels' (`FU'FU`, `FV'FV`)
projected through `H` (by the prior's projection and the sandwich):
`g = −k/2 (log|U| + log|V|) − ½ tr(U⁻¹ E[(A − mI) V⁻¹ (A − mI)'])`, and its
gradient. `(-Inf, 0)` where either is not positive definite.
"""
function _projected_dynamics_term(
    H::AbstractMatrix{T}, ka::KernelMNPrior{T}, FU::Cholesky{T}, FV::Cholesky{T}
) where {T<:Real}
    k = size(H, 2)
    pu, pv = _Projection(ka.projection, H), _Projection(:sandwich, H)
    U, _ = _project(FU, pu)
    V, _ = _project(FV, pv)
    FUk = cholesky(Symmetric(U); check=false)
    FVk = cholesky(Symmetric(V); check=false)
    (issuccess(FUk) && issuccess(FVk)) || return T(-Inf), zeros(T, size(H))
    Ui, Vi = inv(FUk), inv(FVk)
    qc = _mn_centred(ka)
    SU = _loading_moment(qc, Vi)
    SV = qc.mean' * Ui * qc.mean + _loading_excess(qc, Ui)
    g = -k / 2 * (logdet(FUk) + logdet(FVk)) - dot(Ui, SU) / 2
    sym(X) = (X .+ X') ./ 2
    GU = sym(-k / 2 .* Ui .+ Ui * SU * Ui ./ 2)
    GV = sym(-k / 2 .* Vi .+ Vi * SV * Vi ./ 2)
    grad = _projection_gradient(pu, FU, U, GU) .+ _projection_gradient(pv, FV, V, GV)
    return g, grad
end

"""
    _projected_loading_term(H, Fz, P, ν, projection) -> (g, ∂g/∂H)

The part of `E_q[log p(Q | w, H)]` that involves the loading `H` (`r × k`),
`g = ν/2 log|K̃| − (ν − k − 1)/2 tr(K̃ P)` with `K̃` the kernels' covariance
`Z = Fz'Fz` projected through `H` and `P = E[Q⁻¹]`, and its gradient.
`(-Inf, 0)` where `K̃` is not positive definite.
"""
function _projected_loading_term(
    H::AbstractMatrix{T}, Fz::Cholesky{T}, P::AbstractMatrix{T}, ν::T, projection::Symbol
) where {T<:Real}
    c = ν - size(H, 2) - 1
    proj = _Projection(projection, H)
    K̃, _ = _project(Fz, proj)
    Fk = cholesky(Symmetric(K̃); check=false)
    issuccess(Fk) || return T(-Inf), zeros(T, size(H))
    g = ν / 2 * logdet(Fk) - c / 2 * dot(K̃, P)
    G̃ = (ν / 2) .* inv(Fk) .- (c / 2) .* P               # ∂g/∂K̃
    return g, _projection_gradient(proj, Fz, K̃, (G̃ .+ G̃') ./ 2)
end

"""
    _polish_loading(lds, kq, ka, H, A1, rhs, Sxx_c) -> H

With kernel priors on `Q` (`kq`) and/or `A` (`ka`) projected through the
loading, the loading's part of the bound is `tr(H'rhs) − ½ tr(H'A1 H Sxx_c)`
(`[d D]` profiled out, `R` held) plus those priors' terms in `H`. Starting from
the better of the current loading and the closed-form `H` (the first term's
maximiser), at most `loading_iters` (`kq`'s, else 20) L-BFGS steps on the sum;
the result is never worse than the current loading, so the bound does not
decrease.
"""
function _polish_loading(
    lds::_StructuredLDS{T},
    kq::Union{Nothing,KernelIWPrior{T}},
    ka::Union{Nothing,KernelMNPrior{T}},
    H_closed::AbstractMatrix{T},
    A1::AbstractMatrix{T},
    rhs::AbstractMatrix{T},
    Sxx_c::AbstractMatrix{T},
) where {T<:Real}
    om = lds.obs_model
    if kq !== nothing
        Fz = cholesky(Symmetric(kernel_covariance(kq.prior)))
        P = _symmetric_inverse(lds.state_model.Q)
    end
    if ka !== nothing
        FU = cholesky(Symmetric(kernel_covariance(ka.rows)))
        FV = cholesky(Symmetric(kernel_covariance(ka.cols)))
    end
    iters = kq === nothing ? 20 : kq.loading_iters
    shape = size(H_closed)
    function objective!(grad, h)
        H = reshape(h, shape)
        AHS = A1 * H * Sxx_c
        value = dot(H, rhs) - dot(H, AHS) / 2
        slope = rhs .- AHS
        if kq !== nothing
            g, ∂g = _projected_loading_term(H, Fz, P, kq.dof, kq.projection)
            value += g
            slope .+= ∂g
        end
        if ka !== nothing
            g, ∂g = _projected_dynamics_term(H, ka, FU, FV)
            value += g
            slope .+= ∂g
        end
        grad === nothing || (grad .= .-vec(slope))
        return isfinite(value) ? -value : T(Inf)
    end
    H_now = _loading_coefficients(om.basis, om.C)
    candidates = (Matrix{T}(H_closed), Matrix{T}(H_now))
    values = [objective!(nothing, vec(H)) for H in candidates]
    best = argmin(values)
    h0 = vec(copy(candidates[best]))
    iters == 0 && return candidates[best]
    f = h -> objective!(nothing, h)
    g! = (G, h) -> (objective!(G, h); G)
    fg! = (G, h) -> objective!(G, h)
    result = try
        optimize(
            Optim.OnceDifferentiable(f, g!, fg!, h0),
            h0,
            LBFGS(; linesearch=HagerZhang()),
            Optim.Options(; iterations=iters, g_tol=1e-8, f_reltol=1e-10),
        )
    catch err
        err isa LineSearchException || rethrow()
        nothing
    end
    result === nothing && return candidates[best]
    h = Optim.minimizer(result)
    (all(isfinite, h) && Optim.minimum(result) <= values[best]) || return candidates[best]
    return reshape(h, shape)
end

"""
    update_R!(lds, suf, sws)

`R` from the expected residual scatter: that at the mean loading plus
`Φ Cov_q(H; Sxx) Φ'`, what the loading's uncertainty leaves unexplained.
"""
function update_R!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:StructuredGaussianObservationModel{T}}
    lds.fit_bool[6] || return nothing
    om = lds.obs_model
    S_res = sws.elbo.obs_work
    fill!(S_res, zero(T))
    _accumulate_obs_scatter!(S_res, lds, suf, sws)
    q = om.posterior
    if q !== nothing
        k = lds.latent_dim
        Sxx = Matrix(Symmetric(suf.obs_xx[].mat[1:k, 1:k]))
        spread = _loading_moment(q, Sxx) - q.mean * Sxx * q.mean'
        S_res .+= _basis_mul(om.basis, _basis_mul(om.basis, spread)')
    end
    _finalize_R!(lds, S_res, T(suf.obs_n))
    return nothing
end

# Δ for the next E-step, at the current R.
function _refresh_excess!(om::StructuredGaussianObservationModel{T}) where {T<:Real}
    q = om.posterior
    if q === nothing
        fill!(om.excess, zero(T))
    else
        om.excess .= _loading_excess(q, _basis_precision(om)[1])
    end
    return om.excess
end

"""
    _update_kernel_weights!(om; rounds=1, update_trees)

The kernel weights given `q(H)`: fitted to `E[H H'] / k` as `k` draws.
"""
function _update_kernel_weights!(
    om::StructuredGaussianObservationModel{T};
    rounds::Int=1,
    update_trees::Bool=true,
    max_iter::Int=om.weight_iters,
) where {T<:Real}
    prior, q = om.prior, om.posterior
    (prior === nothing || q === nothing) && return nothing
    k = size(q.mean, 2)
    return fit_kernel_weights!(
        prior,
        _loading_moment(q) ./ k,
        k;
        max_iter,
        rounds,
        update_trees,
        laplace=update_trees,
    )
end

function mstep!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:StructuredGaussianObservationModel{T}}
    om = lds.obs_model
    update_initial_state_mean!(lds, suf)
    update_initial_state_covariance!(lds, suf, sws)
    update_A_b!(lds, suf, sws)
    update_Q!(lds, suf, sws)
    lds.fit_bool[5] && _update_loading!(lds, suf)
    update_R!(lds, suf, sws)
    _refresh_excess!(om)
    if _has_kernel_prior(om) && lds.fit_bool[7]
        om.iteration += 1
        _update_kernel_weights!(om; update_trees=om.iteration % om.tree_every == 0)
    end
    return nothing
end

# ----------------------------------------------------------------------------
# The bound
# ----------------------------------------------------------------------------

"""
    Q_obs!(sws, lds, suf)

The expected emission log-likelihood under `q(x) q(H)`: the Gaussian one at the
mean loading, minus `½ tr(Δ Σ_t E[x_t x_t'])`.
"""
function Q_obs!(
    sws::SmoothWorkspace{T}, lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:StructuredGaussianObservationModel{T}}
    k = lds.latent_dim
    value = _gaussian_Q_obs!(sws, lds, suf)
    Sxx = view(suf.obs_xx[].mat, 1:k, 1:k)
    return value - dot(lds.obs_model.excess, Sxx) / 2
end

function _obs_prior_logdensity(
    lds::LinearDynamicalSystem{T,S,O}, ::Union{Nothing,SmoothWorkspace{T}}
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:StructuredGaussianObservationModel{T}}
    om = lds.obs_model
    total = om.R_prior === nothing ? zero(T) : _covariance_prior_term(om.R, om.R_prior)
    prior, q = om.prior, om.posterior
    if prior !== nothing && q !== nothing
        total += _loading_bound(q, kernel_covariance(prior)) + _kernel_logprior(prior)
    end
    return total
end

"""
    elbo!(lds, suf, sws, total_entropy)

The variational bound of a constrained or kernel-prior emission from the
aggregated statistics: the Gaussian LDS terms at the mean loading, the loading's
uncertainty in the emission term, and with a prior
`E[log p(H | w)] + H[q(H)] + log p(w, trees)`.
"""
function elbo!(
    lds::LinearDynamicalSystem{T,S,O},
    suf::SufficientStatistics{T},
    sws::SmoothWorkspace{T},
    total_entropy::T,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:StructuredGaussianObservationModel{T}}
    Q_total = Q_state!(sws, lds, suf) + Q_obs!(sws, lds, suf)
    prior_term = _state_prior_logdensity(lds, sws) + _obs_prior_logdensity(lds, sws)
    return Q_total + prior_term + total_entropy
end
