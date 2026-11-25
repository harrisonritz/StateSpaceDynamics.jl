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
    MNMeanBasis(left, right; coef=1:size(left, 2), names, group, group_names,
                pooled, df=Inf, sd=1.0, sd_prior=nothing)

A structured mean for a [`KernelMNPrior`](@ref):

    E[A] = m I + Σ_c β_{coef[c]} l_c r_c',

`l_c` and `r_c` the columns of `left` and `right` (`dim × n_cols`, in the
kernels' coordinates: the basis's `r` with a learned loading, the latents'
`k` with a fixed one). Each column belongs to the coefficient `coef[c]`, so a
coefficient's term may have any rank (`names` one per coefficient), and each
coefficient to a group (`group`, indices into `group_names`). In a `pooled`
group the coefficients share a location-scale prior,

    β_p ~ t_ν(μ_g, τ_g²)   (ν = df[g]; Inf: Gaussian),

written as `β_p | λ_p ~ N(μ_g, τ_g²/λ_p)`, `λ_p ~ Gamma(ν/2, ν/2)`, with `μ_g`
and `τ_g` fitted (`τ_g` starting at `sd[g]`, with `sd_prior = (s₀, ν₀)`
pseudo-observations of `s₀²`, as the weight trees'); otherwise
`β_p ~ N(0, sd[g]²)`, fixed. A Student-t group lets a few coefficients stand
far from the rest without widening the prior of all of them.

The fit keeps `q(β)` Gaussian (exact given `q(A)`, `U`, `V`) and `q(λ)` Gamma
(variational), and updates `m`, `q(β)`, `q(λ)`, `μ` and `τ` by coordinate
ascent of the bound. With a learned loading `H` the columns are seen in the
latents as `P l_c` and `H' r_c` (`P = (H'H)⁻¹H'`), so that `A_r H ≈ H A` for
`A_r = Σ β_c l_c r_c'`; the loading's polish includes the mean's term.
"""
mutable struct MNMeanBasis{T<:Real}
    const left::Matrix{T}
    const right::Matrix{T}
    const coef::Vector{Int}
    const names::Vector{String}
    const group::Vector{Int}
    const group_names::Vector{String}
    const pooled::Vector{Bool}
    const df::Vector{T}
    const sd::Vector{T}
    const sd_prior::Union{Nothing,Tuple{T,T}}
    # q(β): mean (P) and covariance (P × P)
    mean::Vector{T}
    cov::Matrix{T}
    # pooled groups: location and scale
    mu::Vector{T}
    tau::Vector{T}
    # q(λ_p) = Gamma((ν + 1)/2, rate[p]): E[λ] and E[log λ]
    rate::Vector{T}
    lambda::Vector{T}
    loglambda::Vector{T}
    started::Bool
end

function MNMeanBasis(
    left::AbstractMatrix{T},
    right::AbstractMatrix{T};
    coef::AbstractVector{<:Integer}=1:size(left, 2),
    names::AbstractVector=["beta$(p)" for p in 1:maximum(coef; init=0)],
    group::AbstractVector{<:Integer}=ones(Int, length(names)),
    group_names::AbstractVector=["group$(g)" for g in 1:maximum(group; init=0)],
    pooled=true,
    df=T(Inf),
    sd=one(T),
    sd_prior=nothing,
) where {T<:Real}
    size(left) == size(right) || throw(
        DimensionMismatch(
            "left and right must have one size; got $(size(left)) and $(size(right))"
        ),
    )
    n_cols = size(left, 2)
    length(coef) == n_cols || throw(
        DimensionMismatch("coef needs one entry per column ($n_cols); got $(length(coef))"),
    )
    P, G = length(names), length(group_names)
    sort!(unique(collect(coef))) == collect(1:P) ||
        throw(ArgumentError("every coefficient needs a column; coef must cover 1:$P"))
    length(group) == P || throw(
        DimensionMismatch(
            "group needs one entry per coefficient ($P); got $(length(group))"
        ),
    )
    all(in(1:G), group) || throw(ArgumentError("group entries index group_names (1:$G)"))
    function per_group(x, what)
        v = x isa AbstractVector ? collect(x) : fill(x, G)
        length(v) == G || throw(
            DimensionMismatch("$what needs one entry per group ($G); got $(length(v))")
        )
        return v
    end
    pooled_ = Bool.(per_group(pooled, "pooled"))
    df_ = T.(per_group(df, "df"))
    sd_ = T.(per_group(sd, "sd"))
    all(>(0), df_) || throw(ArgumentError("df must be positive (Inf: Gaussian)"))
    all(>(0), sd_) || throw(ArgumentError("sd must be positive"))
    sdp = sd_prior === nothing ? nothing : (T(sd_prior[1]), T(sd_prior[2]))
    return MNMeanBasis{T}(
        Matrix{T}(left),
        Matrix{T}(right),
        Vector{Int}(coef),
        String.(names),
        Vector{Int}(group),
        String.(group_names),
        pooled_,
        df_,
        sd_,
        sdp,
        zeros(T, P),
        Matrix{T}(Diagonal(sd_[group] .^ 2)),
        zeros(T, G),
        copy(sd_),
        ones(T, P),
        ones(T, P),
        zeros(T, P),
        false,
    )
end

"""
    KernelMNPrior(rows, cols; mean=:identity, projection=:noise_weighted,
                  weight_iters=50, tree_every=1, start_rounds=30, mean_basis=nothing)

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
centres `A` on `m I` with `m` fitted (`:zero`: on zero). `mean_basis` (an
[`MNMeanBasis`](@ref)) adds a structured mean, `E[A] = m I + Σ_c β_c l_c r_c'`,
whose coefficients have a Gaussian posterior and a pooled (Gaussian or Student-t)
prior fitted with the rest.

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

For a learned loading with `projection=:noise_weighted`, `latent_scale > 0`
adds a ridge to the receiver precision and to the sender covariance in
latent coordinates: `U = (H'Z_rows⁻¹H + D)⁻¹`, `V = H'Z_cols H + D`.
The same `D` regularizes the solve that lifts the structured mean.
`D = latent_scale * I` in the starting coordinates and transforms with the
latents when the fitter changes gauge. This allows more latent states than
loading-basis dimensions. The default `latent_scale=0` uses the full-column-rank
projection.

Needs a `GaussianStateModel` without an `AB_prior` (one prior on `A`). Not
available in an `SLDS`, a composite emission, with `depends_on`, with the
affine `gauge_*` options of `fit!` or with a free `C`. Beside a kernel prior on
the loading the kernels are projected through the posterior mean of `H` (a
plug-in: the prior does not inform `q(H)`, and the bound may dip as `H` moves);
the emission's gauge step carries `q(A)` along. `loglikelihood` is the plug-in
one at the mean `A`.
"""
mutable struct KernelMNPrior{T<:Real,PU<:KernelPrior{T},PV<:KernelPrior{T}}
    const rows::PU
    const cols::PV
    const mean::Symbol
    m::T
    const projection::Symbol
    const latent_scale::T
    const weight_iters::Int
    const tree_every::Int
    const start_rounds::Int
    # a structured mean beyond `m I` (`MNMeanBasis`), or nothing
    const mean_basis::Union{Nothing,MNMeanBasis{T}}
    # the loading `H` (r × k) a projected prior is seen through; empty otherwise
    loading::Matrix{T}
    latent_ridge::Union{Nothing,Matrix{T}}
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
    latent_scale::Real=0,
    weight_iters::Int=50,
    tree_every::Int=1,
    start_rounds::Int=30,
    mean_basis::Union{Nothing,MNMeanBasis{T}}=nothing,
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
    scale = T(latent_scale)
    isfinite(scale) && scale >= 0 || throw(ArgumentError("latent_scale must be finite and nonnegative"))
    (scale == 0 || projection === :noise_weighted) ||
        throw(ArgumentError("latent_scale needs projection=:noise_weighted"))
    weight_iters >= 1 || throw(ArgumentError("weight_iters must be at least 1"))
    tree_every >= 1 || throw(ArgumentError("tree_every must be at least 1"))
    start_rounds >= 1 || throw(ArgumentError("start_rounds must be at least 1"))
    if mean_basis !== nothing && size(mean_basis.left, 1) != rows.dim
        throw(
            DimensionMismatch(
                "the mean basis has $(size(mean_basis.left, 1)) rows; the kernels $(rows.dim)",
            ),
        )
    end
    return KernelMNPrior{T,typeof(rows),typeof(cols)}(
        rows,
        cols,
        mean,
        zero(T),
        projection,
        scale,
        weight_iters,
        tree_every,
        start_rounds,
        mean_basis === nothing ? nothing : deepcopy(mean_basis),
        zeros(T, 0, 0),
        nothing,
        nothing,
        zeros(T, 0, 0),
        0,
    )
end

_mean_count(mb::MNMeanBasis) = length(mb.names)

# The one-hot map from columns to coefficients (n_cols × P).
function _mean_onehot(mb::MNMeanBasis{T}) where {T<:Real}
    S = zeros(T, length(mb.coef), _mean_count(mb))
    for (c, p) in enumerate(mb.coef)
        S[c, p] = one(T)
    end
    return S
end

function _mn_latent_ridge(kp::KernelMNPrior{T}, k=size(kp.loading, 2)) where {T<:Real}
    (kp.latent_scale == 0 || k == 0) && return nothing
    if kp.latent_ridge === nothing
        kp.latent_ridge = Matrix{T}(kp.latent_scale * I, k, k)
    end
    return kp.latent_ridge
end

# The columns in the latents' coordinates at the loading `H` (empty: as given):
# `P l_c` and `H' r_c`, `P = (H'H + ridge)⁻¹H'`.
function _mean_columns(mb::MNMeanBasis{T}, H::AbstractMatrix{T}, ridge=nothing) where {T<:Real}
    isempty(H) && return mb.left, mb.right
    P = _projection_gram(H, ridge) \ H'
    return P * mb.left, H' * mb.right
end
_mean_columns(kp::KernelMNPrior) = _mean_columns(kp.mean_basis, kp.loading, _mn_latent_ridge(kp))

# The prior mean `m I + a diag(β_c) b'` at the columns `cols` (default: the
# current loading's).
function _mn_mean(kp::KernelMNPrior{T}, cols=nothing) where {T<:Real}
    k = _mn_dim(kp)
    M = Matrix{T}(kp.m * I, k, k)
    mb = kp.mean_basis
    mb === nothing && return M
    a, b = cols === nothing ? _mean_columns(kp) : cols
    return M .+ a * Diagonal(mb.mean[mb.coef]) * b'
end

# `Cov(β)` on the columns, `C̃[c, c'] = Cov(β_coef[c], β_coef[c'])`.
_column_cov(mb::MNMeanBasis) = mb.cov[mb.coef, mb.coef]

# What β's uncertainty adds to the row moment `E[(A − M) V⁻¹ (A − M)']` and to the
# column moment `E[(A − M)' U⁻¹ (A − M)]`.
function _mean_row_extra(mb::MNMeanBasis{T}, cols, Vi::AbstractMatrix{T}) where {T<:Real}
    a, b = cols
    X = a * (_column_cov(mb) .* (b' * Vi * b)) * a'
    return Matrix(Symmetric((X .+ X') ./ 2))
end
function _mean_col_extra(mb::MNMeanBasis{T}, cols, Ui::AbstractMatrix{T}) where {T<:Real}
    a, b = cols
    X = b * (_column_cov(mb) .* (a' * Ui * a)) * b'
    return Matrix(Symmetric((X .+ X') ./ 2))
end

# The prior's centre and precision of each coefficient: `μ_g` and `E[λ_p]/τ_g²`
# in a pooled group, `0` and `1/sd²` otherwise.
function _mean_prior(mb::MNMeanBasis{T}) where {T<:Real}
    g = mb.group
    centre = [mb.pooled[g[p]] ? mb.mu[g[p]] : zero(T) for p in eachindex(g)]
    precision = [
        mb.pooled[g[p]] ? mb.lambda[p] / mb.tau[g[p]]^2 : inv(mb.sd[g[p]]^2) for
        p in eachindex(g)
    ]
    return centre, precision
end

# q(λ_p) for the current q(β), μ and τ: Gamma((ν + 1)/2, (ν + E[(β − μ)²]/τ²)/2).
function _mean_lambda!(mb::MNMeanBasis{T}) where {T<:Real}
    for p in eachindex(mb.group)
        g = mb.group[p]
        ν = mb.df[g]
        if !mb.pooled[g] || !isfinite(ν)
            mb.rate[p], mb.lambda[p], mb.loglambda[p] = one(T), one(T), zero(T)
            continue
        end
        e = (mb.mean[p] - mb.mu[g])^2 + mb.cov[p, p]
        shape = (ν + 1) / 2
        mb.rate[p] = (ν + e / mb.tau[g]^2) / 2
        mb.lambda[p] = shape / mb.rate[p]
        mb.loglambda[p] = digamma(shape) - log(mb.rate[p])
    end
    return mb
end

# μ_g (the λ-weighted mean) and τ_g (with the sd_prior pseudo-observations) of
# each pooled group, given q(β) and q(λ).
function _mean_groups!(mb::MNMeanBasis{T}) where {T<:Real}
    for g in eachindex(mb.group_names)
        mb.pooled[g] || continue
        rows = findall(==(g), mb.group)
        w = mb.lambda[rows]
        mb.mu[g] = sum(w .* mb.mean[rows]) / sum(w)
        e = (mb.mean[rows] .- mb.mu[g]) .^ 2 .+ [mb.cov[p, p] for p in rows]
        total, n = sum(w .* e), T(length(rows))
        if mb.sd_prior !== nothing
            s0, nu0 = mb.sd_prior
            total += nu0 * s0^2
            n += nu0
        end
        mb.tau[g] = max(sqrt(total / n), T(1e-8))
    end
    return mb
end

"""
    _mn_update_mean!(kp, U, V; sweeps=20)

Given `q(A)`, `U` and `V`: `m` and `q(β)` jointly (the generalized least squares
of `Â` on `I` and the basis's terms under the precision `V⁻¹ ⊗ U⁻¹`, `m` with a
flat prior), then `q(λ)`, `μ` and `τ`, `sweeps` times. Without a basis, `m` alone
(its maximiser).
"""
function _mn_update_mean!(
    kp::KernelMNPrior{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}; sweeps::Int=20
) where {T<:Real}
    Ui, Vi = _symmetric_inverse(U), _symmetric_inverse(V)
    Â = kp.posterior.mean
    mb = kp.mean_basis
    identity = kp.mean === :identity
    if mb === nothing
        if identity
            m = tr(Ui * Â * Vi) / tr(Ui * Vi)
            isfinite(m) && (kp.m = m)
        end
        return kp
    end
    a, b = _mean_columns(kp)
    S = _mean_onehot(mb)
    Gβ = S' * ((a' * Ui * a) .* (b' * Vi * b)) * S
    hβ = S' * vec(sum((Ui * Â * Vi * b) .* a; dims=1))
    g0β = S' * vec(sum((Ui * Vi * b) .* a; dims=1))
    g0, h0 = tr(Ui * Vi), tr(Ui * Â * Vi)
    sym(X) = Matrix(Symmetric((X .+ X') ./ 2))
    for _ in 1:sweeps
        centre, precision = _mean_prior(mb)
        Λ = sym(Gβ + Diagonal(precision))
        rhs = hβ .+ precision .* centre
        if identity
            # m's MAP under the marginal (β integrated out), then q(β | m)
            F = cholesky(Symmetric(Λ))
            m = (h0 - dot(g0β, F \ rhs)) / (g0 - dot(g0β, F \ g0β))
            isfinite(m) && (kp.m = m)
            rhs = rhs .- kp.m .* g0β
            mb.cov = sym(inv(F))
        else
            mb.cov = sym(inv(cholesky(Symmetric(Λ))))
        end
        mb.mean = mb.cov * rhs
        _mean_lambda!(mb)
        _mean_groups!(mb)
    end
    mb.started = true
    return kp
end

"""
    _mean_bound(mb) -> T

`E[log p(β | μ, τ, λ)] + H[q(β)] + E[log p(λ)] + H[q(λ)] + log p(τ)` at the
current `q(β)`, `q(λ)`, `μ` and `τ`, up to the constants the weight trees drop
(`log p(τ)`: `−ν₀ log τ − ν₀ s₀²/(2τ²)`).
"""
function _mean_bound(mb::MNMeanBasis{T}) where {T<:Real}
    P = _mean_count(mb)
    total = (logdet(Symmetric(mb.cov)) + P * (log(2π) + 1)) / 2
    for p in 1:P
        g = mb.group[p]
        if mb.pooled[g]
            τ, ν = mb.tau[g], mb.df[g]
            e = (mb.mean[p] - mb.mu[g])^2 + mb.cov[p, p]
            total += -log(2π) / 2 - log(τ) + mb.loglambda[p] / 2 - mb.lambda[p] * e / (2τ^2)
            if isfinite(ν)
                shape, rate = (ν + 1) / 2, mb.rate[p]
                Eλ, Elog = mb.lambda[p], mb.loglambda[p]
                total +=
                    ν / 2 * log(ν / 2) - loggamma(ν / 2) + (ν / 2 - 1) * Elog - ν / 2 * Eλ
                total -=
                    shape * log(rate) - loggamma(shape) + (shape - 1) * Elog - rate * Eλ
            end
        else
            s = mb.sd[g]
            total += -log(2π) / 2 - log(s) - (mb.mean[p]^2 + mb.cov[p, p]) / (2s^2)
        end
    end
    if mb.sd_prior !== nothing
        s0, nu0 = mb.sd_prior
        for g in eachindex(mb.group_names)
            mb.pooled[g] || continue
            total -= nu0 * log(mb.tau[g]) + nu0 * s0^2 / (2mb.tau[g]^2)
        end
    end
    return total
end

"""
    mean_coefficients(prior::KernelMNPrior) -> NamedTuple or nothing

The structured mean's fitted state (`nothing` without an [`MNMeanBasis`](@ref)):
the coefficients' posterior `mean` and `sd`, their `names` and `group`s, each
pooled group's `mu` and `tau`, and each coefficient's `lambda` (`E[λ]`, below one
for a coefficient its group's t prior lets stand out).
"""
function mean_coefficients(kp::KernelMNPrior)
    mb = kp.mean_basis
    mb === nothing && return nothing
    return (;
        names=mb.names,
        group=mb.group_names[mb.group],
        mean=copy(mb.mean),
        sd=sqrt.(max.(diag(mb.cov), 0)),
        mu=Dict(zip(mb.group_names, mb.mu)),
        tau=Dict(zip(mb.group_names, mb.tau)),
        lambda=copy(mb.lambda),
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
the mean's scale `m` and its basis's coefficients (`mean`,
[`mean_coefficients`](@ref); `nothing` without one).
"""
function kernel_weights(kp::KernelMNPrior)
    return (;
        rows=kernel_weights(kp.rows),
        cols=kernel_weights(kp.cols),
        m=kp.m,
        mean=mean_coefficients(kp),
    )
end

# The receivers' and senders' w-step counts (`kernel_unit_steps`), once when the
# two are one prior.
function kernel_unit_steps(kp::KernelMNPrior)
    rows = kernel_unit_steps(kp.rows)
    kp.rows === kp.cols && return rows
    return _add_unit_steps(rows, kernel_unit_steps(kp.cols))
end

_mn_dim(kp::KernelMNPrior) = isempty(kp.loading) ? kp.rows.dim : size(kp.loading, 2)

function _mn_projections(kp::KernelMNPrior{T}, H=kp.loading) where {T<:Real}
    isempty(H) && return _identity_projection(T), _identity_projection(T)
    ridge = _mn_latent_ridge(kp, size(H, 2))
    return _Projection(kp.projection, H, ridge), _Projection(:sandwich, H, ridge)
end

# U and V at the current weights, in the latents' coordinates.
function _mn_covariances(kp::KernelMNPrior{T}) where {T<:Real}
    pu, pv = _mn_projections(kp)
    U = first(_project(cholesky(Symmetric(kernel_covariance(kp.rows))), pu))
    V = first(_project(cholesky(Symmetric(kernel_covariance(kp.cols))), pv))
    return U, V
end

# `q(A)` shifted to the prior's mean `M` (`m I`, plus the mean basis's terms).
function _mn_centred(kp::KernelMNPrior{T}, M=_mn_mean(kp)) where {T<:Real}
    q = kp.posterior
    return LoadingPosterior{T}(q.mean - M, q.left, q.right, q.var)
end

# `E[(A − M) V⁻¹ (A − M)']` and `E[(A − M)' U⁻¹ (A − M)]` under q(A) and q(β).
function _mn_row_moment(kp::KernelMNPrior{T}, V::AbstractMatrix{T}) where {T<:Real}
    Vi = _symmetric_inverse(V)
    M = _loading_moment(_mn_centred(kp), Vi)
    kp.mean_basis === nothing ||
        (M += _mean_row_extra(kp.mean_basis, _mean_columns(kp), Vi))
    return M
end
function _mn_col_moment(kp::KernelMNPrior{T}, U::AbstractMatrix{T}) where {T<:Real}
    qc = _mn_centred(kp)
    Ui = _symmetric_inverse(U)
    M = qc.mean' * Ui * qc.mean + _loading_excess(qc, Ui)
    kp.mean_basis === nothing ||
        (M += _mean_col_extra(kp.mean_basis, _mean_columns(kp), Ui))
    return Matrix(Symmetric((M .+ M') ./ 2))
end

"""
    _mn_posterior!(kp, Qinv, Syx_c, Sxx_c, Sxx)

`q(A)` for `vec A ~ N(vec M, V ⊗ U)` (`M = m I` and the mean basis's terms at the
current `q(β)`) and the transitions' likelihood with
precision `Sxx ⊗ Q⁻¹` (`Sxx = Σ_{t<T} E[x_t x_t']`), `b` and `B` profiled out:
`Syx_c` and `Sxx_c` are `Σ x_{t+1} x_t'` and `Sxx` with `[1; u]` partialled out.
With `Ã = (A − M) V^{-½}` the prior is that of a kernel-prior loading
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
    M = _mn_mean(kp)
    rhs = Qinv * (Syx_c .- M * Sxx_c) * root
    sym(X) = Matrix(Symmetric((X .+ X') ./ 2))
    q = _loading_posterior(U, Qinv, rhs, sym(root * Sxx_c * root), sym(root * Sxx * root))
    kp.posterior = LoadingPosterior{T}(M + q.mean * root, q.left, root * q.right, q.var)
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
`tr(U⁻¹ Â V⁻¹) / tr(U⁻¹ V⁻¹)`; with a mean basis, `m` and `q(β)` and the
coefficients' hyperparameters, [`_mn_update_mean!`](@ref)), then the row weights
given `V` (the `k` columns of `(A − M) V^{-½}` as draws from `N(0, U)`), then the
column weights given `U`. Each step is the MAP given the rest, so the bound does not decrease while
the trees are held.
"""
function _mn_update_hyper!(
    kp::KernelMNPrior{T}; update_trees::Bool, max_iter::Int=kp.weight_iters
) where {T<:Real}
    k = _mn_dim(kp)
    pu, pv = _mn_projections(kp)
    U, V = _mn_covariances(kp)
    _mn_update_mean!(kp, U, V)
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

`E_q[log p(A | U, V, M)] + H[q(A)] + log p(w_U, trees) + log p(w_V, trees)`, and
with a mean basis its coefficients' terms ([`_mean_bound`](@ref)); the expectation
is over `q(A) q(β)`.
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
    mean_term = kp.mean_basis === nothing ? zero(T) : _mean_bound(kp.mean_basis)
    return prior_term +
           entropy +
           T(k * k) / 2 +
           mean_term +
           _kernel_logprior(kp.rows) +
           _kernel_logprior(kp.cols)
end

# Δ_A of a state model, or nothing.
function _dynamics_excess(sm)
    prior = hasproperty(sm, :A_prior) ? sm.A_prior : nothing
    (prior isa KernelMNPrior && prior.posterior !== nothing) || return nothing
    return prior.excess
end

#=============================================================================
A kernel prior on the input matrix: each group of input columns a draw from a
kernel covariance over the latents (or the rows it is linked to), with a
variational posterior for B.

    B[rows_g, c] ~ N(0, U_g(w_g))  for c in groups[g],   B[others, c] = 0
    q(B) Gaussian over the free entries, exact given q(x), Â, Q̃ = E[Q⁻¹]⁻¹

`q(A) q(B)` is mean-field: each is updated given the other's mean, with `b`
profiled out.
=============================================================================#

"""
    KernelInputPrior(priors, groups; rows=nothing, projection=:noise_weighted,
                     weight_iters=50, tree_every=1, start_rounds=30, sweeps=3)

A kernel prior on the input matrix `B` of a [`GaussianStateModel`](@ref)
(`x_{t+1} = A x_t + b + B u_t + w_t`): `priors[g]` (a [`KernelPrior`](@ref)) is
the prior of every column of `B` in `groups[g]` (input indices), so each input
of a group is a draw of its footprint on the latents from one kernel covariance
with fitted weights,

    B[:, c] ~ N(0, U_g)   (c in groups[g]),

and inputs in no group have a flat prior. `rows[g]` (latent indices, or
`nothing` for all) links a group to a subset of the latents: its columns are
zero outside them and `priors[g]` is over those latents only (latents in their
own coordinates: a fixed basis or a masked loading, such as ROI states).

The kernels live where `A`'s receivers do (see [`KernelMNPrior`](@ref)): in the
latents' coordinates with a fixed basis or a masked loading, in the basis's
`r` coordinates with a learned loading, projected through `H` by `projection`
(`B` maps inputs to the latents as `A`'s rows do, so `U = P K P'`). With a
learned loading without a prior on it the loading's polish has the prior's term.

`B` becomes a latent variable: `fit!` keeps `q(B)`, Gaussian over its free
entries (exact given `Â`, `E[Q⁻¹]` and the point `b`), mean-field with `q(A)`
under a `KernelMNPrior` (`sweeps` alternations of the two per M-step), adds
`B`'s uncertainty to `Q`'s residual scatter and the bound, and updates each
group's weights given `q(B)` (the group's columns as draws). Not with an
`AB_prior`, `depends_on`, the affine `gauge_*` options or a free `C`.
"""
mutable struct KernelInputPrior{T<:Real}
    const priors::Vector{KernelPrior{T}}
    const groups::Vector{Vector{Int}}
    const rows::Vector{Union{Nothing,Vector{Int}}}
    const projection::Symbol
    const weight_iters::Int
    const tree_every::Int
    const start_rounds::Int
    const sweeps::Int
    # the loading `H` (r × k) a projected prior is seen through; empty otherwise
    loading::Matrix{T}
    # q(B): its mean (k × m) and the covariance of vec B (zero off the free
    # entries); empty until a fit starts
    mean::Matrix{T}
    cov::Matrix{T}
    iteration::Int
end

function KernelInputPrior(
    priors::AbstractVector{<:KernelPrior{T}},
    groups::AbstractVector;
    rows=nothing,
    projection::Symbol=:noise_weighted,
    weight_iters::Int=50,
    tree_every::Int=1,
    start_rounds::Int=30,
    sweeps::Int=3,
) where {T<:Real}
    length(priors) == length(groups) || throw(
        ArgumentError(
            "one input group per prior; got $(length(priors)) priors " *
            "and $(length(groups)) groups",
        ),
    )
    isempty(priors) && throw(ArgumentError("a KernelInputPrior needs at least one group"))
    groups = [collect(Int, g) for g in groups]
    all(!isempty, groups) || throw(ArgumentError("every input group needs a column"))
    flat = reduce(vcat, groups)
    (all(>=(1), flat) && allunique(flat)) || throw(
        ArgumentError("the input groups are disjoint sets of positive column indices")
    )
    rows = rows === nothing ? fill(nothing, length(groups)) : collect(rows)
    length(rows) == length(groups) ||
        throw(ArgumentError("rows has one entry (indices or nothing) per group"))
    rows = Union{Nothing,Vector{Int}}[
        r === nothing ? nothing : collect(Int, r) for r in rows
    ]
    for (g, r) in enumerate(rows)
        r === nothing && continue
        (!isempty(r) && allunique(r) && all(>=(1), r)) ||
            throw(ArgumentError("rows[$g] are distinct positive latent indices"))
        length(r) == priors[g].dim || throw(
            DimensionMismatch(
                "group $g links $(length(r)) latents; its prior is over $(priors[g].dim)",
            ),
        )
    end
    projection in (:noise_weighted, :euclidean) || throw(
        ArgumentError("projection must be :noise_weighted or :euclidean; got :$projection"),
    )
    weight_iters >= 1 || throw(ArgumentError("weight_iters must be at least 1"))
    tree_every >= 1 || throw(ArgumentError("tree_every must be at least 1"))
    start_rounds >= 1 || throw(ArgumentError("start_rounds must be at least 1"))
    sweeps >= 1 || throw(ArgumentError("sweeps must be at least 1"))
    return KernelInputPrior{T}(
        KernelPrior{T}[deepcopy(p) for p in priors],
        groups,
        rows,
        projection,
        weight_iters,
        tree_every,
        start_rounds,
        sweeps,
        zeros(T, 0, 0),
        zeros(T, 0, 0),
        zeros(T, 0, 0),
        0,
    )
end

"""
    input_posterior(state_model) -> (mean, cov) or nothing

`q(B)` of a state model with a [`KernelInputPrior`](@ref) once a fit has
computed it: the mean (`state_model.B`) and the covariance of `vec B`.
"""
function input_posterior(sm)
    prior = hasproperty(sm, :B_prior) ? sm.B_prior : nothing
    (prior isa KernelInputPrior && !isempty(prior.mean)) || return nothing
    return (; mean=copy(prior.mean), cov=copy(prior.cov))
end

"""
    kernel_weights(prior::KernelInputPrior) -> Vector

The fitted weights of each input group's kernel prior.
"""
kernel_weights(kb::KernelInputPrior) = [kernel_weights(p) for p in kb.priors]

# The groups' w-step counts summed (`kernel_unit_steps`).
function kernel_unit_steps(kb::KernelInputPrior)
    return reduce(_add_unit_steps, (kernel_unit_steps(p) for p in kb.priors))
end

_input_projected(kb::KernelInputPrior) = !isempty(kb.loading)

function _input_projection(kb::KernelInputPrior{T}) where {T<:Real}
    _input_projected(kb) || return _identity_projection(T)
    return _Projection(kb.projection, kb.loading)
end

# Group g's covariance in the latents' coordinates (over its rows).
function _input_covariance(kb::KernelInputPrior{T}, g::Int) where {T<:Real}
    F = cholesky(Symmetric(kernel_covariance(kb.priors[g])))
    return first(_project(F, _input_projection(kb)))
end

# The latents group g's columns live on.
_input_rows(kb::KernelInputPrior, g::Int, k::Int) = something(kb.rows[g], collect(1:k))

# The free entries of vec B (k × m, column-major) and, per group, those of its
# columns.
function _input_free(kb::KernelInputPrior, k::Int, m::Int)
    owner = zeros(Int, m)
    for (g, cols) in enumerate(kb.groups), c in cols
        owner[c] = g
    end
    free = Int[]
    for c in 1:m
        r = owner[c] == 0 ? collect(1:k) : _input_rows(kb, owner[c], k)
        append!(free, (c - 1) * k .+ r)
    end
    return free
end

"""
    _input_posterior!(kb, Qinv, Suu_c, Sru_c)

`q(B)` for the transitions' likelihood (precision `Suu_c ⊗ Q⁻¹`, `Suu_c = Σ u u'`
and `Sru_c = Σ (x_{t+1} − Â x_t) u'`, both with the intercept partialled out)
and the groups' priors: dense over the free entries.
"""
function _input_posterior!(
    kb::KernelInputPrior{T},
    Qinv::AbstractMatrix{T},
    Suu_c::AbstractMatrix{T},
    Sru_c::AbstractMatrix{T},
) where {T<:Real}
    k, m = size(Sru_c)
    Λ = kron(Matrix(Symmetric(Suu_c)), Matrix(Symmetric(Qinv)))
    for (g, cols) in enumerate(kb.groups)
        Ui = _symmetric_inverse(_input_covariance(kb, g))
        r = _input_rows(kb, g, k)
        for c in cols
            idx = (c - 1) * k .+ r
            Λ[idx, idx] .+= Ui
        end
    end
    h = vec(Qinv * Sru_c)
    free = _input_free(kb, k, m)
    F = cholesky(Symmetric(Λ[free, free]))
    μ = zeros(T, k * m)
    μ[free] = F \ h[free]
    Σ = zeros(T, k * m, k * m)
    Σ[free, free] = Matrix(Symmetric(inv(F)))
    kb.mean = reshape(μ, k, m)
    kb.cov = Σ
    return kb
end

# E[B_g B_g'] over group g's rows, summed over its columns.
function _input_moment(kb::KernelInputPrior{T}, g::Int) where {T<:Real}
    k = size(kb.mean, 1)
    r = _input_rows(kb, g, k)
    M = zeros(T, length(r), length(r))
    for c in kb.groups[g]
        b = kb.mean[r, c]
        idx = (c - 1) * k .+ r
        M .+= b * b' .+ kb.cov[idx, idx]
    end
    return Matrix(Symmetric((M .+ M') ./ 2))
end

# `E[(B − B̂) S (B − B̂)']` (k × k): what B's uncertainty adds to a residual
# scatter with the inputs' moment `S` (m × m).
function _input_spread(kb::KernelInputPrior{T}, S::AbstractMatrix{T}) where {T<:Real}
    k, m = size(kb.mean)
    out = zeros(T, k, k)
    for c in 1:m, d in 1:m
        s = S[c, d]
        iszero(s) && continue
        out .+= s .* view(kb.cov, ((c - 1) * k + 1):(c * k), ((d - 1) * k + 1):(d * k))
    end
    return Matrix(Symmetric((out .+ out') ./ 2))
end

# B's spread of a state model at the inputs' moment, or nothing.
function _input_excess(sm, S::AbstractMatrix)
    prior = hasproperty(sm, :B_prior) ? sm.B_prior : nothing
    (prior isa KernelInputPrior && !isempty(prior.cov)) || return nothing
    return _input_spread(prior, Matrix(S))
end

"""
    _input_update_hyper!(kb; update_trees, max_iter=kb.weight_iters)

Each group's weights given `q(B)`: its columns as draws from `N(0, U_g)`.
"""
function _input_update_hyper!(
    kb::KernelInputPrior{T}; update_trees::Bool, max_iter::Int=kb.weight_iters
) where {T<:Real}
    proj = _input_projection(kb)
    options = (; max_iter, rounds=1, update_trees, laplace=update_trees)
    for (g, cols) in enumerate(kb.groups)
        n = T(length(cols))
        _fit_kernel_weights!(
            kb.priors[g], _Draws(_input_moment(kb, g) ./ n, n, proj); options...
        )
    end
    return kb
end

# Before the first q(B): each group's weights centred on the starting `B`.
function _input_initialize!(kb::KernelInputPrior{T}, B::AbstractMatrix{T}) where {T<:Real}
    k = size(B, 1)
    proj = _input_projection(kb)
    for (g, cols) in enumerate(kb.groups)
        kb.priors[g].initialized && continue
        r = _input_rows(kb, g, k)
        Bg = B[r, cols]
        floor = max(sum(abs2, B) / max(length(B), 1), eps(T)) * T(1e-2)
        start = Matrix(Symmetric(Bg * Bg' ./ length(cols))) + floor * I
        _initialize_prior!(kb.priors[g], _unproject(start, proj))
    end
    return kb
end

"""
    _input_bound(kb) -> T

`E_q[log p(B | U)] + H[q(B)] + Σ_g log p(w_g, trees)`, the inputs in no group
under a flat prior.
"""
function _input_bound(kb::KernelInputPrior{T}) where {T<:Real}
    isempty(kb.mean) && throw(
        ArgumentError(
            "this KernelInputPrior has not been started: its posterior is set up by fit! " *
            "(or elbo) of a LinearDynamicalSystem",
        ),
    )
    k, m = size(kb.mean)
    total = zero(T)
    for (g, cols) in enumerate(kb.groups)
        F = cholesky(Symmetric(_input_covariance(kb, g)))
        d = size(F, 1)
        n = length(cols)
        total += -n / 2 * (logdet(F) + d * log(T(2π))) - tr(F \ _input_moment(kb, g)) / 2
        total += _kernel_logprior(kb.priors[g])
    end
    free = _input_free(kb, k, m)
    total +=
        logdet(cholesky(Symmetric(kb.cov[free, free]))) / 2 +
        length(free) * (1 + log(T(2π))) / 2
    return total
end

# Before a fit's first iteration: unless fitted, each group's weights centred on
# the least-squares `B` given the starting `A` (a start at `B = 0` would centre
# them on nothing), then `start_rounds` alternations of `q(B)` and the weights;
# then the first `q(B)`.
function _input_start!(kb::KernelInputPrior{T}, lds, suf) where {T<:Real}
    sm = lds.state_model
    Qinv = _symmetric_inverse(sm.Q)
    stats = _input_statistics(lds, suf, Matrix{T}(sm.A))
    if !all(p -> p.initialized, kb.priors)
        Suu_c, Sru_c = stats
        _input_initialize!(kb, Matrix{T}(Sru_c / Symmetric(Suu_c)))
        for _ in 1:(kb.start_rounds)
            _input_posterior!(kb, Qinv, stats...)
            _input_update_hyper!(kb; update_trees=true, max_iter=500)
        end
    end
    _input_posterior!(kb, Qinv, stats...)
    return kb
end
