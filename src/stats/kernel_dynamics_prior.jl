#=============================================================================
A kernel prior on the transition matrix: a matrix normal whose row (receiver)
and column (sender) covariances are kernel covariances, with a variational
posterior for A.

    A ~ MN(m I, U(w_U), V(w_V)),   Cov(A_ij, A_kl) = U_ik V_jl
    q(A) Gaussian, exact given q(x), Q̃ = E[Q⁻¹]⁻¹ and the point b, B

The LDS wiring: `update_A_b!` / `update_Q!` (`continuous_latents.jl`), the
transition precision in `_compute_state_constants!` and `_state_gradient!`
(the uncertainty of A adds Δ_A = E[A'Q⁻¹A] − Â'Q⁻¹Â to every state with a
successor), the bound in `Q_state!` and `_state_prior_logdensity`, and the
start in `_start_covariance_priors!`.
=============================================================================#

"""
    LoadingPosterior{T}

A Gaussian posterior of a matrix (`r × k`): the loading `H` of a kernel-prior
emission ([`emission_posterior`](@ref)), or a transition matrix `A` under a
[`KernelMNPrior`](@ref):

    H = mean + left * E * right',    E_lj independent, Var E_lj = var[l, j]

so that `Cov(vec H) = (right ⊗ left) diag(vec var) (right ⊗ left)'`. `left` is
`r × r` and `right` `k × k`, both invertible.
"""
struct LoadingPosterior{T<:Real}
    mean::Matrix{T}
    left::Matrix{T}
    right::Matrix{T}
    var::Matrix{T}
end

"""
    KernelMNPrior(rows, cols; mean=:identity, projection=:noise_weighted,
                  weight_iters=50, tree_every=1, start_rounds=30)

A matrix-normal prior on the transition matrix `A` of a
[`GaussianStateModel`](@ref) whose row and column covariances are the
covariances of two [`KernelPrior`](@ref)s, with their weights fitted alongside
the model:

    A ~ MN(m I, U, V),   Cov(A_ij, A_kl) = U_ik V_jl,   Var(A_ij) = U_ii V_jj

Row `i` of `A` receives (`x_{t+1,i} = Σ_j A_ij x_{t,j} + …`) and column `j`
sends, so `rows` is a **receiver** kernel prior and `cols` a **sender** one:
with one scale weight per region, a coupling's spread factors into the
receiver's gain times the sender's. Only separable structure is expressed, and
one overall scale moves freely between `U` and `V` (only the relative profiles
are identified; the weights' hyperpriors settle the rest). `mean=:identity`
centres `A` on `m I` with `m` fitted (`:zero`: on zero).

`A` becomes a latent variable. `fit!` keeps its Gaussian posterior `q(A)`
(exact given `E[Q⁻¹]` and the point `b`, `B`), gives the smoother the
precision `E[A'Q⁻¹A]`, and adds `A`'s uncertainty to `Q`'s residual scatter.
Each iteration then updates `m` and the weights of `U` and `V` in turn (each
the MAP given the other, at most `weight_iters` L-BFGS steps; trees every
`tree_every` iterations). Before the first iteration the weights are fitted to
the starting `A` (`start_rounds` alternations). The trace is the variational
bound, which never decreases while the trees are held. See
[`dynamics_posterior`](@ref).

# Where the kernels live

The kernels describe a source-space transition `A_s ~ MN(0, K_recv, K_send)`
seen in the latents' coordinates: with `J` the map from the latents to the
sources, `U = J⁺ K_recv J⁺'` and `V = J' K_send J`. So `rows` takes
covariance-type kernels (`G K G'`, `G = Φ'L`, as for `Q`) and `cols`
reconstruction-type ones (`G⁺' K G⁺`, `G⁺` a source estimate's operator).

- With a **fixed basis** (`GaussianObservationModel(; C=Φ, C_fixed=true)`) the
  kernels are given in the latents' coordinates (`k × k`).
- With a **learned loading** (a [`ConstrainedGaussianObservationModel`](@ref))
  they are given in the basis's `r` coordinates and projected through `H`: the
  rows by `projection` (`:noise_weighted`, `(H'Z⁻¹H)⁻¹`, or `:euclidean`,
  `H⁺ZH⁺'`), the columns by the sandwich `H'ZH`. The prior then moves with the
  latents (`x → T x` takes it to that of `T A T⁻¹`), and `H` gets a gradient
  polish with the prior's term each iteration.

Needs a `GaussianStateModel` without an `AB_prior` (one prior on `A`). Not
available in an `SLDS`, a composite emission, with `depends_on`, with the
affine `gauge_*` options of `fit!`, with a kernel prior on the loading or with
a free `C`. `loglikelihood` is the plug-in one at the mean `A`.
"""
mutable struct KernelMNPrior{T<:Real,PU<:KernelPrior{T},PV<:KernelPrior{T}}
    const rows::PU
    const cols::PV
    const mean::Symbol
    m::T
    const projection::Symbol
    const weight_iters::Int
    const tree_every::Int
    const start_rounds::Int
    # the loading `H` (r × k) a projected prior is seen through; empty otherwise
    loading::Matrix{T}
    posterior::Union{Nothing,LoadingPosterior{T}}
    # Δ_A = E[A'Q⁻¹A] − Â'Q⁻¹Â, what the smoother adds to each state with a
    # successor; empty until the posterior exists
    excess::Matrix{T}
    iteration::Int
end

function KernelMNPrior(
    rows::KernelPrior{T},
    cols::KernelPrior{T};
    mean::Symbol=:identity,
    projection::Symbol=:noise_weighted,
    weight_iters::Int=50,
    tree_every::Int=1,
    start_rounds::Int=30,
) where {T<:Real}
    rows.dim == cols.dim || throw(
        DimensionMismatch(
            "the row and column kernels must have one size; got $(rows.dim) and $(cols.dim)",
        ),
    )
    mean in (:identity, :zero) ||
        throw(ArgumentError("mean must be :identity or :zero; got :$mean"))
    projection in (:noise_weighted, :euclidean) || throw(
        ArgumentError("projection must be :noise_weighted or :euclidean; got :$projection"),
    )
    weight_iters >= 1 || throw(ArgumentError("weight_iters must be at least 1"))
    tree_every >= 1 || throw(ArgumentError("tree_every must be at least 1"))
    start_rounds >= 1 || throw(ArgumentError("start_rounds must be at least 1"))
    return KernelMNPrior{T,typeof(rows),typeof(cols)}(
        rows,
        cols,
        mean,
        zero(T),
        projection,
        weight_iters,
        tree_every,
        start_rounds,
        zeros(T, 0, 0),
        nothing,
        zeros(T, 0, 0),
        0,
    )
end

"""
    dynamics_posterior(state_model) -> LoadingPosterior or nothing

`q(A)` of a state model with a [`KernelMNPrior`](@ref) once a fit has computed
it: `A = mean + left E right'`, the entries of `E` independent with variances
`var` (see [`LoadingPosterior`](@ref)). `state_model.A` holds the mean.
"""
function dynamics_posterior(sm)
    prior = hasproperty(sm, :A_prior) ? sm.A_prior : nothing
    return prior isa KernelMNPrior ? prior.posterior : nothing
end

"""
    kernel_weights(prior::KernelMNPrior) -> NamedTuple

The fitted weights of the receiver (`rows`) and sender (`cols`) kernel priors,
and the mean's scale `m`.
"""
function kernel_weights(kp::KernelMNPrior)
    return (; rows=kernel_weights(kp.rows), cols=kernel_weights(kp.cols), m=kp.m)
end

_mn_dim(kp::KernelMNPrior) = isempty(kp.loading) ? kp.rows.dim : size(kp.loading, 2)

function _mn_projections(kp::KernelMNPrior{T}) where {T<:Real}
    isempty(kp.loading) && return _identity_projection(T), _identity_projection(T)
    return _Projection(kp.projection, kp.loading), _Projection(:sandwich, kp.loading)
end

# U and V at the current weights, in the latents' coordinates.
function _mn_covariances(kp::KernelMNPrior{T}) where {T<:Real}
    pu, pv = _mn_projections(kp)
    U = first(_project(cholesky(Symmetric(kernel_covariance(kp.rows))), pu))
    V = first(_project(cholesky(Symmetric(kernel_covariance(kp.cols))), pv))
    return U, V
end

# `q(A)` shifted to the prior's mean, `A − m I`.
function _mn_centred(kp::KernelMNPrior{T}) where {T<:Real}
    q = kp.posterior
    return LoadingPosterior{T}(q.mean - kp.m * I, q.left, q.right, q.var)
end

# `E[(A − mI) V⁻¹ (A − mI)']` and `E[(A − mI)' U⁻¹ (A − mI)]`.
function _mn_row_moment(kp::KernelMNPrior{T}, V::AbstractMatrix{T}) where {T<:Real}
    return _loading_moment(_mn_centred(kp), _symmetric_inverse(V))
end
function _mn_col_moment(kp::KernelMNPrior{T}, U::AbstractMatrix{T}) where {T<:Real}
    qc = _mn_centred(kp)
    Ui = _symmetric_inverse(U)
    M = qc.mean' * Ui * qc.mean + _loading_excess(qc, Ui)
    return Matrix(Symmetric((M .+ M') ./ 2))
end

"""
    _mn_posterior!(kp, Qinv, Syx_c, Sxx_c, Sxx)

`q(A)` for `vec A ~ N(vec(m I), V ⊗ U)` and the transitions' likelihood with
precision `Sxx ⊗ Q⁻¹` (`Sxx = Σ_{t<T} E[x_t x_t']`), `b` and `B` profiled out:
`Syx_c` and `Sxx_c` are `Σ x_{t+1} x_t'` and `Sxx` with `[1; u]` partialled out.
With `Ã = (A − m I) V^{-½}` the prior is that of a kernel-prior loading
(`vec Ã ~ N(0, I ⊗ U)`), so `q(Ã)` is [`_loading_posterior`](@ref)'s, and `q(A)`
its image.
"""
function _mn_posterior!(
    kp::KernelMNPrior{T},
    Qinv::AbstractMatrix{T},
    Syx_c::AbstractMatrix{T},
    Sxx_c::AbstractMatrix{T},
    Sxx::AbstractMatrix{T},
) where {T<:Real}
    U, V = _mn_covariances(kp)
    root = _sym_sqrt(V)
    rhs = Qinv * (Syx_c .- kp.m .* Sxx_c) * root
    sym(X) = Matrix(Symmetric((X .+ X') ./ 2))
    q = _loading_posterior(U, Qinv, rhs, sym(root * Sxx_c * root), sym(root * Sxx * root))
    kp.posterior = LoadingPosterior{T}(
        kp.m * I + q.mean * root, q.left, root * q.right, q.var
    )
    return kp.posterior
end

# Δ_A = E[A'Q⁻¹A] − Â'Q⁻¹Â at the current q(A) and Q.
function _mn_refresh_excess!(kp::KernelMNPrior{T}, Q::AbstractMatrix) where {T<:Real}
    kp.posterior === nothing && return kp
    kp.excess = _loading_excess(kp.posterior, _symmetric_inverse(Matrix{T}(Q)))
    return kp
end

# `E[(A − Â) S (A − Â)']`, what A's uncertainty adds to a residual scatter.
function _mn_spread(kp::KernelMNPrior{T}, S::AbstractMatrix{T}) where {T<:Real}
    q = kp.posterior
    s = [dot(view(q.right, :, j), S, view(q.right, :, j)) for j in axes(q.right, 2)]
    M = q.left * Diagonal(q.var * s) * q.left'
    return Matrix(Symmetric((M .+ M') ./ 2))
end

"""
    _mn_update_hyper!(kp; update_trees, max_iter=kp.weight_iters)

Given `q(A)`: the mean's `m` (its maximiser,
`tr(U⁻¹ Â V⁻¹) / tr(U⁻¹ V⁻¹)`), then the row weights given `V` (the `k` columns
of `(A − mI) V^{-½}` as draws from `N(0, U)`), then the column weights given
`U`. Each step is the MAP given the rest, so the bound does not decrease while
the trees are held.
"""
function _mn_update_hyper!(
    kp::KernelMNPrior{T}; update_trees::Bool, max_iter::Int=kp.weight_iters
) where {T<:Real}
    k = _mn_dim(kp)
    pu, pv = _mn_projections(kp)
    U, V = _mn_covariances(kp)
    if kp.mean === :identity
        Ui, Vi = _symmetric_inverse(U), _symmetric_inverse(V)
        m = tr(Ui * kp.posterior.mean * Vi) / tr(Ui * Vi)
        isfinite(m) && (kp.m = m)
    end
    options = (; max_iter, rounds=1, update_trees, laplace=update_trees)
    _fit_kernel_weights!(kp.rows, _Draws(_mn_row_moment(kp, V) ./ k, T(k), pu); options...)
    U, _ = _mn_covariances(kp)
    _fit_kernel_weights!(kp.cols, _Draws(_mn_col_moment(kp, U) ./ k, T(k), pv); options...)
    return kp
end

"""
    _mn_start!(kp, A, Qinv, Syx_c, Sxx_c, Sxx)

Before a fit's first iteration: unless the weights have been fitted, centre the
prior on the starting `A` (`m = tr A / k`, the row covariance at the power of
`A − m I`, the column covariance at unit power) and alternate `start_rounds`
times between `q(A)` and the mean and weights; then the first `q(A)` itself.
"""
function _mn_start!(
    kp::KernelMNPrior{T},
    A::AbstractMatrix{T},
    Qinv::AbstractMatrix{T},
    Syx_c::AbstractMatrix{T},
    Sxx_c::AbstractMatrix{T},
    Sxx::AbstractMatrix{T},
) where {T<:Real}
    k = size(A, 1)
    if !(kp.rows.initialized && kp.cols.initialized)
        kp.mean === :identity && (kp.m = tr(A) / k)
        pu, pv = _mn_projections(kp)
        Ac = A - kp.m * I
        # a floor for a start at (or near) m I, whose deviation has no power
        floor = max(sum(abs2, A) / k, eps(T)) * T(1e-2) / k
        start = Matrix(Symmetric(Ac * Ac' ./ k)) + floor * I
        _initialize_prior!(kp.cols, _unproject(Matrix{T}(I, k, k), pv))
        _initialize_prior!(kp.rows, _unproject(start, pu))
        for _ in 1:(kp.start_rounds)
            _mn_posterior!(kp, Qinv, Syx_c, Sxx_c, Sxx)
            _mn_update_hyper!(kp; update_trees=true, max_iter=500)
        end
    end
    _mn_posterior!(kp, Qinv, Syx_c, Sxx_c, Sxx)
    return kp
end

"""
    _mn_bound(kp) -> T

`E_q[log p(A | U, V, m)] + H[q(A)] + log p(w_U, trees) + log p(w_V, trees)`.
"""
function _mn_bound(kp::KernelMNPrior{T}) where {T<:Real}
    q = kp.posterior
    q === nothing && throw(
        ArgumentError(
            "this KernelMNPrior has not been started: its posterior is set up by fit! " *
            "(or elbo) of a LinearDynamicalSystem",
        ),
    )
    k = size(q.mean, 1)
    U, V = _mn_covariances(kp)
    FU, FV = cholesky(Symmetric(U)), cholesky(Symmetric(V))
    prior_term = -k / 2 * (logdet(FU) + logdet(FV)) - tr(FU \ _mn_row_moment(kp, V)) / 2
    entropy = k * (logabsdet(q.left)[1] + logabsdet(q.right)[1]) + sum(log, q.var) / 2
    return prior_term +
           entropy +
           T(k * k) / 2 +
           _kernel_logprior(kp.rows) +
           _kernel_logprior(kp.cols)
end

# Δ_A of a state model, or nothing.
function _dynamics_excess(sm)
    prior = hasproperty(sm, :A_prior) ? sm.A_prior : nothing
    (prior isa KernelMNPrior && prior.posterior !== nothing) || return nothing
    return prior.excess
end
