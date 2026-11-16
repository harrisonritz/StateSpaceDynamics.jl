#=============================================================================
Kernel priors on a noise covariance (`Q` or `R`): an inverse-Wishart whose
mean is a kernel covariance, with a variational posterior.

    Σ ~ IW(ν, (ν − d − 1) K(w)),   E[Σ] = K(w),   w ~ the KernelPrior's hyperprior
    q(Σ) = IW(ν + N, Ψ + S)        (S, N: the expected residual scatter, its count)

The model stores `Σ̃ = E_q[Σ⁻¹]⁻¹` in `Q` / `R`, which is all the smoother and
the other M-steps need. The LDS wiring is in `continuous_latents.jl`
(`_finalize_Q!`), `gaussian_observations.jl` (`_finalize_R!`) and the
`_start_covariance_priors!` hook of the fit loops.
=============================================================================#

"""
    KernelIWPrior(prior; dof=nothing, projection=:noise_weighted, weight_iters=50,
                  tree_every=1, start_rounds=30, loading_iters=20)

A prior on a noise covariance `Σ` (`Q` of a [`GaussianStateModel`](@ref), or
`R` of a Gaussian emission) whose mean is the covariance of a
[`KernelPrior`](@ref):

    Σ ~ IW(ν, (ν − d − 1) K(w)),   E[Σ] = K(w)

with the kernel weights `w` fitted alongside the model, under the prior's
hyperprior (`levels`, `correlated`, …). `dof` is `ν`, which must exceed
`d + 1` (`d` the dimension of `Σ`); `nothing` is `d + 2`, the weakest prior with
a mean. The larger it is, the more `Σ` is pulled to `K(w)`.

Set it as the `Q_prior` or `R_prior`. `fit!` then treats `Σ` as a latent
variable with the variational posterior `q(Σ) = IW(ν + N, Ψ + S)` (`S` the
expected residual scatter, `N` its count) and stores `E[Σ⁻¹]⁻¹` in the model,
so the smoother sees the posterior's precision. Each iteration then moves the
weights to their MAP given `E[Σ⁻¹]` (at most `weight_iters` L-BFGS steps) and,
every `tree_every` iterations, updates the prior's trees from the weights'
Laplace covariance. Before the first iteration the weights are fitted to the
starting `Σ` (`start_rounds` rounds of weights and trees). The trace is the
variational bound, which never decreases while the trees are held.

# Where the kernels live

- `R`: in sensor space (`obs_dim × obs_dim`), with any Gaussian emission; for
  example `L K L'` for an unmodelled brain background beside a nuisance kernel
  for the sensor noise.
- `Q` with a **fixed basis** (`GaussianObservationModel(; C=Φ, C_fixed=true, …)`):
  in the latents' coordinates, which are the basis's (`Φ'L K L'Φ`).
- `Q` with a **learned loading** (a [`ConstrainedGaussianObservationModel`](@ref),
  `C = Φ H`): in the basis's `r` coordinates (`Φ'L K L'Φ`, `r × r`), and
  projected onto the latents through the current `H`, `K(w) = K̃(Z(w); H)`:
  - `projection=:noise_weighted` (default): `K̃ = (H'Z⁻¹H)⁻¹`, the covariance of
    the least-squares latent image of noise distributed as `Z`;
  - `projection=:euclidean`: `K̃ = H⁺ Z H⁺'`, its orthogonal projection.

  Either way the prior moves with the latents (`x → T x` takes `K̃` to
  `T K̃ T'`), so the fit stays free of any choice of coordinates. The prior then
  depends on `H`: each iteration takes the closed-form `H` and then up to
  `loading_iters` gradient steps on the bound with the prior's term in `H`.

Needs a `GaussianStateModel`. Not available in an `SLDS`, a composite emission,
with `depends_on`, with the affine `gauge_*` options of `fit!` (on `Q`), with
`R_diagonal` (on `R`), or on `Q` with a kernel prior on the loading or a free
`C` (`C_fixed=false`). See [`kernel_weights`](@ref) for the fitted weights.
"""
mutable struct KernelIWPrior{T<:Real,P<:KernelPrior{T}}
    const prior::P
    dof::T                       # NaN: the dimension of Σ plus two, set at the start
    const projection::Symbol
    const weight_iters::Int
    const tree_every::Int
    const start_rounds::Int
    const loading_iters::Int
    # the loading `H` (r × k) a projected prior is seen through; empty otherwise
    loading::Matrix{T}
    # ν_q of q(Σ) = IW(ν_q, ν_q Σ̃): ν plus the count of the data it was fitted
    # to; NaN until a fit starts
    posterior_dof::T
    # the Gaussian terms priced with Σ in the bound being evaluated (transitions
    # or bins, plus a paired matrix-normal prior's width): the training data's,
    # or a held-out set's while it is scored
    count::T
    iteration::Int
end

function KernelIWPrior(
    prior::KernelPrior{T};
    dof::Union{Nothing,Real}=nothing,
    projection::Symbol=:noise_weighted,
    weight_iters::Int=50,
    tree_every::Int=1,
    start_rounds::Int=30,
    loading_iters::Int=20,
) where {T<:Real}
    dof === nothing ||
        (isfinite(dof) && dof > 2) ||
        throw(
            ArgumentError("dof must be finite and exceed the dimension plus one; got $dof")
        )
    projection in (:noise_weighted, :euclidean) || throw(
        ArgumentError("projection must be :noise_weighted or :euclidean; got :$projection"),
    )
    weight_iters >= 1 || throw(ArgumentError("weight_iters must be at least 1"))
    tree_every >= 1 || throw(ArgumentError("tree_every must be at least 1"))
    start_rounds >= 1 || throw(ArgumentError("start_rounds must be at least 1"))
    loading_iters >= 0 || throw(ArgumentError("loading_iters must be non-negative"))
    return KernelIWPrior{T,typeof(prior)}(
        prior,
        dof === nothing ? T(NaN) : T(dof),
        projection,
        weight_iters,
        tree_every,
        start_rounds,
        loading_iters,
        zeros(T, 0, 0),
        T(NaN),
        T(NaN),
        0,
    )
end

kernel_weights(kp::KernelIWPrior) = kernel_weights(kp.prior)
kernel_covariance(kp::KernelIWPrior, args...) = kernel_covariance(kp.prior, args...)

_is_projected(kp::KernelIWPrior) = !isempty(kp.loading)
_iw_dim(kp::KernelIWPrior) = _is_projected(kp) ? size(kp.loading, 2) : kp.prior.dim
_iw_scale_factor(kp::KernelIWPrior) = kp.dof - _iw_dim(kp) - 1

# The prior's mean of Σ at the current weights (projected through the loading
# when there is one).
function _iw_mean(kp::KernelIWPrior{T}) where {T<:Real}
    Z = kernel_covariance(kp.prior)
    _is_projected(kp) || return Z
    return first(_project(cholesky(Symmetric(Z)), _Projection(kp.projection, kp.loading)))
end

# The weights' target given `P = E[Σ⁻¹]`.
function _iw_target(kp::KernelIWPrior{T}, P::Matrix{T}) where {T<:Real}
    _is_projected(kp) || return _IWMean(P, kp.dof)
    return _IWMean(P, kp.dof, _Projection(kp.projection, kp.loading))
end

# Projected priors follow the loading `H` (`r × k`).
function _set_loading!(kp::KernelIWPrior{T}, H::AbstractMatrix) where {T<:Real}
    size(H, 1) == kp.prior.dim || throw(
        DimensionMismatch(
            "the Q_prior's kernels are $(kp.prior.dim) × $(kp.prior.dim); the loading's " *
            "coefficients have $(size(H, 1)) rows",
        ),
    )
    kp.loading = Matrix{T}(H)
    return kp
end

function _symmetric_inverse(Σ::AbstractMatrix)
    return Matrix(Symmetric(inv(cholesky(Symmetric(Matrix(Σ))))))
end

# log Γ_d(a)
function _logmvgamma(d::Int, a::T) where {T<:Real}
    return d * (d - 1) / 4 * log(T(π)) + sum(loggamma(a + (1 - i) / T(2)) for i in 1:d)
end

# E[log|Σ|] under IW(ν, Ψ), from log|Ψ|.
function _iw_expected_logdet(logdetΨ::T, ν::T, d::Int) where {T<:Real}
    return logdetΨ - d * log(T(2)) - sum(digamma((ν - i + 1) / 2) for i in 1:d)
end

function _check_started(kp::KernelIWPrior)
    isfinite(kp.count) && isfinite(kp.posterior_dof) || throw(
        ArgumentError(
            "this KernelIWPrior has not been started: its posterior is set up by fit! " *
            "(or elbo) of a LinearDynamicalSystem",
        ),
    )
    return nothing
end

"""
    _kernel_iw_start!(kp, Σ, N)

Before a fit's first iteration: set the count `N`, the starting posterior
`q(Σ) = IW(ν + N, (ν + N) Σ)` and, unless the prior has been fitted before, fit
its weights and trees to the starting covariance `Σ`, as if `E[Σ⁻¹] = Σ⁻¹`.
"""
function _kernel_iw_start!(kp::KernelIWPrior{T}, Σ::AbstractMatrix, N::Real) where {T<:Real}
    size(Σ, 1) == _iw_dim(kp) || throw(
        DimensionMismatch(
            "the KernelIWPrior is $(_iw_dim(kp)) × $(_iw_dim(kp)); the covariance is " *
            "$(size(Σ, 1)) × $(size(Σ, 2))",
        ),
    )
    isfinite(kp.dof) || (kp.dof = T(_iw_dim(kp) + 2))
    kp.count = T(N)
    kp.posterior_dof = kp.dof + T(N)
    if !kp.prior.initialized
        P = _symmetric_inverse(Matrix{T}(Σ))
        _fit_kernel_weights!(
            kp.prior, _iw_target(kp, P); max_iter=5000, rounds=kp.start_rounds
        )
    end
    return kp
end

"""
    _kernel_iw_posterior!(Σ̃, kp, S, N)

The variational posterior `q(Σ) = IW(ν + N, (ν − d − 1) K(w) + S)` given the
symmetric scatter `S` of `N` terms: writes `E[Σ⁻¹]⁻¹ = (Ψ + S)/(ν + N)` into `Σ̃`,
then moves the weights to their MAP given it (and the trees, every
`tree_every` calls).
"""
function _kernel_iw_posterior!(
    Σ̃::AbstractMatrix{T}, kp::KernelIWPrior{T}, S::AbstractMatrix{T}, N::T
) where {T<:Real}
    kp.count = N
    kp.posterior_dof = kp.dof + N
    Ψ = _iw_scale_factor(kp) .* _iw_mean(kp)
    Σ̃ .= (Ψ .+ S) ./ (kp.dof + N)
    Σ̃ .= (Σ̃ .+ Σ̃') ./ 2
    kp.iteration += 1
    update_trees = kp.iteration % kp.tree_every == 0
    _fit_kernel_weights!(
        kp.prior,
        _iw_target(kp, _symmetric_inverse(Σ̃));
        max_iter=kp.weight_iters,
        rounds=1,
        update_trees,
        laplace=update_trees,
    )
    return Σ̃
end

"""
    _kernel_iw_bound(kp, Σ̃) -> T

The bound's terms for `Σ` beyond those the ELBO evaluates at the point `Σ̃`:
with `q(Σ) = IW(ν_q, ν_q Σ̃)` (`ν_q = ν` plus the training count) and `N` the
count of the data being scored,

    −N/2 (E[log|Σ|] − log|Σ̃|) + E_q[log p(Σ | w)] − E_q[log q(Σ)] + log p(w, trees)

The first term turns the `N` Gaussian terms the ELBO priced at `log|Σ̃|` into
their expectation (`E[Σ⁻¹] = Σ̃⁻¹` already holds).
"""
function _kernel_iw_bound(kp::KernelIWPrior{T}, Σ̃::AbstractMatrix{T}) where {T<:Real}
    _check_started(kp)
    d, ν, N, νq = _iw_dim(kp), kp.dof, kp.count, kp.posterior_dof
    F = cholesky(Symmetric(Matrix(Σ̃)))
    logdetΣ̃ = logdet(F)
    logdetΨq = d * log(νq) + logdetΣ̃
    Elogdet = _iw_expected_logdet(logdetΨq, νq, d)
    Ψ = _iw_scale_factor(kp) .* _iw_mean(kp)
    expected_prior =
        ν / 2 * logdet(cholesky(Symmetric(Ψ))) - ν * d / 2 * log(T(2)) -
        _logmvgamma(d, ν / 2) - (ν + d + 1) / 2 * Elogdet - tr(F \ Ψ) / 2
    expected_q =
        νq / 2 * logdetΨq - νq * d / 2 * log(T(2)) - _logmvgamma(d, νq / 2) -
        (νq + d + 1) / 2 * Elogdet - νq * d / 2
    return -N / 2 * (Elogdet - logdetΣ̃) + expected_prior - expected_q +
           _kernel_logprior(kp.prior)
end

# The ELBO's term for a covariance prior: the MAP's log density for an
# `IWPrior`, the variational terms for a `KernelIWPrior`.
_covariance_prior_term(Σ::AbstractMatrix, prior::IWPrior) = iw_logprior_term(Σ, prior)
function _covariance_prior_term(Σ::AbstractMatrix{T}, prior::KernelIWPrior{T}) where {T}
    return _kernel_iw_bound(prior, Σ)
end

# The count of a covariance's Gaussian terms in the bound of `lds` on the data
# whose statistics `suf` holds.
_q_count(lds, suf) = suf.dyn_n + _ab_prior_width(lds)
function _r_count(lds, suf)
    om = lds.obs_model
    return om isa GaussianObservationModel ? suf.obs_n + _cd_prior_width(lds) : suf.obs_n
end

"""
    _start_covariance_priors!(lds, suf; fit=true)

After the E-step of `lds` on some data, the `KernelIWPrior`s on `Q` and `R`
learn that data's count, and a `KernelMNPrior` on `A` gets its first `q(A)`. With `fit=true` (a fit's first iteration) this also
starts their posteriors, fitting the weights to the starting `Q`, `R` if they
have not been fitted; with `fit=false` (a bound scored on other data, such as
held-out trials) the posteriors are left as they are, unless none exists yet.
"""
function _start_covariance_priors!(lds, suf; fit::Bool=true)
    suf isa SufficientStatistics || return nothing
    sm, om = lds.state_model, lds.obs_model
    for (prior, Σ, count) in (
        (hasproperty(sm, :Q_prior) ? sm.Q_prior : nothing, sm, _q_count),
        (hasproperty(om, :R_prior) ? om.R_prior : nothing, om, _r_count),
    )
        prior isa KernelIWPrior || continue
        current = Σ isa AbstractObservationModel ? Σ.R : Σ.Q
        # a Q prior seen through a learned loading follows its current H
        if !(Σ isa AbstractObservationModel) && om isa StructuredGaussianObservationModel
            _set_loading!(prior, _loading_coefficients(om.basis, om.C))
        end
        N = count(lds, suf)
        if fit || !isfinite(prior.posterior_dof)
            _kernel_iw_start!(prior, current, N)
        else
            prior.count = N
        end
    end
    # a kernel prior on A: its first q(A), after Q's (it uses E[Q⁻¹])
    kp = hasproperty(sm, :A_prior) ? sm.A_prior : nothing
    if kp isa KernelMNPrior
        if om isa StructuredGaussianObservationModel
            kp.loading = Matrix(_loading_coefficients(om.basis, om.C))
        end
        if fit || kp.posterior === nothing
            stats = _transition_statistics(lds, suf)
            _mn_start!(kp, sm.A, _symmetric_inverse(sm.Q), stats...)
            copyto!(sm.A, kp.posterior.mean)
        end
        _mn_refresh_excess!(kp, sm.Q)
    end
    return nothing
end
