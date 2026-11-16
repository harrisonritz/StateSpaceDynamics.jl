#=============================================================================
Gaussian emissions whose loading is constrained to a basis, drawn from a kernel
prior, or both: `C = Φ H`, `vec H ~ N(0, I ⊗ Σ(w))`.

One struct, three classes (the aliases below). Included from `types.jl`, before
the emission unions are defined; the methods are in `kernel_observations.jl` and
`fit_kernel_LDS.jl`.
=============================================================================#

# `LoadingPosterior` is defined in `stats/kernel_dynamics_prior.jl`, which a
# transition matrix's kernel prior also uses.

"""
    StructuredGaussianObservationModel{T,M,V,B,P}

A Gaussian emission `y_t ~ N(C x_t + d + D u_t, R)` whose loading is `C = Φ H`,
with `Φ` a fixed basis (or the identity) and `H` either free or drawn from a
[`KernelPrior`](@ref). It is built through one of its three classes:

- [`ConstrainedGaussianObservationModel`](@ref): a basis, no prior;
- [`KernelGaussianObservationModel`](@ref): a prior, no basis;
- [`ConstrainedKernelGaussianObservationModel`](@ref): both.

# Fields
- `C`, `R`, `d`, `D`, `R_prior`, `R_diagonal`: as in
  [`GaussianObservationModel`](@ref). `C` is the posterior mean `Φ Ĥ`.
- `depends_on`, `group_seeds`, `variants`: always `nothing`; these emissions have
  one parameter set for every trial.
- `basis`: `Φ` (`obs_dim × r`, full column rank), or `nothing` for the identity.
- `prior`: the [`KernelPrior`](@ref) of `H`'s columns (`r × r`), or `nothing`.
- `posterior`: `q(H)` once a fit has computed it ([`LoadingPosterior`](@ref)).
- `excess`: `E[C'R⁻¹C] − Ĉ'R⁻¹Ĉ` (`k × k`), what the uncertainty of `H` adds to
  the emission precision the smoother sees. Zero without a prior.
- `gauge`: how a fit handles the latents' gauge, `:bound`, `:moment` or `:none`
  (see [`KernelGaussianObservationModel`](@ref)).
- `weight_iters`: the most L-BFGS steps the kernel weights take per EM iteration.
- `tree_every`: the prior's trees are updated every this many EM iterations.
"""
Base.@kwdef mutable struct StructuredGaussianObservationModel{
    T<:Real,
    M<:AbstractMatrix{T},
    V<:AbstractVector{T},
    B<:Union{Nothing,Matrix{T}},
    P<:Union{Nothing,KernelPrior{T}},
} <: AbstractObservationModel{T}
    C::M
    R::M
    d::V
    D::M = zeros(eltype(C), size(C, 1), 0)
    R_prior::Union{Nothing,IWPrior{T},KernelIWPrior{T}} = nothing
    R_diagonal::Bool = false
    # Read by the generic entry points; parameter grouping is not supported here.
    depends_on::Nothing = nothing
    group_seeds::Nothing = nothing
    variants::Nothing = nothing
    basis::B = nothing
    prior::P = nothing
    posterior::Union{Nothing,LoadingPosterior{T}} = nothing
    excess::Matrix{T} = zeros(eltype(C), size(C, 2), size(C, 2))
    gauge::Symbol = :bound
    weight_iters::Int = 50
    tree_every::Int = 1
    iteration::Int = 0
end

"""
    ConstrainedGaussianObservationModel(; basis, C or H, R, d, D, R_prior,
                                        R_diagonal=false)

A Gaussian emission whose loading lies in the column span of a fixed basis:
`C = basis * H` with `basis` (`obs_dim × r`, full column rank) fixed and `H`
(`r × latent_dim`) free. For example the leading left singular vectors of a
forward model, so that the latents load only on patterns the sources can make.

Give the start as `H`, or as a `C` in the span. Given the current `R`, the
M-step's `H` is the exact constrained maximiser (with `d` and `D` profiled
out), and `R`'s own update follows it, so each EM iteration is an ECM step and
the ELBO never decreases. A square, invertible basis constrains nothing.

`d` defaults to zero, `D` to no observation inputs. Not available in a
switching model, with an `LQRStateModel`, as a member of a composite emission,
or with `depends_on`.
"""
const ConstrainedGaussianObservationModel{T,M,V} = StructuredGaussianObservationModel{
    T,M,V,Matrix{T},Nothing
}

"""
    KernelGaussianObservationModel(; prior, C, R, d, D, R_prior, R_diagonal=false,
                                   gauge=:bound, weight_iters=50, tree_every=1)

A Gaussian emission whose loading has a kernel prior: each column of `C`
(`obs_dim × latent_dim`) is a draw from `N(0, Σ)`, `Σ` the covariance of
`prior`, a [`KernelPrior`](@ref) over the `obs_dim` observations. The loading
is then a latent variable with a Gaussian posterior `q(C)`, and the kernel
weights are fitted with the model.

Fitted by variational Laplace EM (`fit!`). Each iteration:

- `q(x)`: the smoother, with the emission precision `E[C'R⁻¹C]`, which the
  uncertainty of `C` makes larger than that of its mean (`obs_model.excess`);
- the dynamics: the usual M-step;
- `q(C)`: exact given `R`, with `d` and `D` profiled out; then `d`, `D`, `R`;
- the kernel weights: their MAP given `E[C C']` (at most `weight_iters` L-BFGS
  steps), and every `tree_every` iterations their Laplace covariance and the
  prior's trees ([`fit_kernel_weights!`](@ref)).

The trace `fit!` returns is the variational bound at the weights' MAP,

    E_q[log p(y, x | C)] + H[q(x)] + E_q[log p(C | w)] + H[q(C)] + log p(w, trees),

which never decreases while the trees are held (`:flat` types, or
`fit_bool` with `weights=false`); a tree update uses the weights' Laplace
covariance, which that bound leaves out, and can lower it slightly.

# The gauge

The likelihood does not change under `x → T x`, `C → C T⁻¹`, but the prior on
`C` does: it treats the columns alike. `gauge` says which of those equivalent
models a fit settles on.

- `:bound` (default): no constraint. Every dynamics parameter is free, so the
  bound itself picks the gauge, and each iteration takes the exact step along
  it, `T'T = E[C'Σ⁻¹C] / r` (a parameter-expansion step, which never lowers
  the bound). The fit converges to the gauge where the columns' second moment
  under the prior is the identity.
- `:moment`: each iteration rescales the latents to a unit centred second
  moment. That is a constraint, not a step of the bound, so the trace is not
  monotone.
- `:none`: no step. The same fixed points as `:bound`, reached slowly.

`fit_bool` has a third emission slot, `weights`. `obs_model.C` holds the
posterior mean; see [`emission_posterior`](@ref) and [`kernel_weights`](@ref).
The same restrictions as [`ConstrainedGaussianObservationModel`](@ref) apply.
"""
const KernelGaussianObservationModel{T,M,V,P<:KernelPrior{T}} = StructuredGaussianObservationModel{
    T,M,V,Nothing,P
}

"""
    ConstrainedKernelGaussianObservationModel(; basis, prior, C or H, R, d, D,
        R_prior, R_diagonal=false, gauge=:bound, weight_iters=50, tree_every=1)

A Gaussian emission `C = basis * H` whose coefficients have a kernel prior:
each column of `H` (`r × latent_dim`) is a draw from `N(0, Σ)`, `Σ` the
covariance of `prior`, a [`KernelPrior`](@ref) over the `r` columns of
`basis`. With `basis` the leading singular vectors `Φ` of a forward model `L`
and kernels `Φ'L K_g L'Φ`, it is `C = L J` with the columns of `J` drawn from
`N(0, Σ_g θ_g K_g)`: a weighted minimum-norm prior on the loading, inside the
fit. The caller projects the kernels; nothing here knows where they came from.

Fitted as a [`KernelGaussianObservationModel`](@ref), which it is when `basis`
is the identity; see there for the algorithm, the trace and `gauge`.
"""
const ConstrainedKernelGaussianObservationModel{T,M,V,P<:KernelPrior{T}} = StructuredGaussianObservationModel{
    T,M,V,Matrix{T},P
}

function _structured_model(;
    basis,
    prior,
    C=nothing,
    H=nothing,
    R::AbstractMatrix{T},
    d::Union{Nothing,AbstractVector}=nothing,
    D::Union{Nothing,AbstractMatrix}=nothing,
    R_prior::Union{Nothing,IWPrior,KernelIWPrior}=nothing,
    R_diagonal::Bool=false,
    gauge::Symbol=:bound,
    weight_iters::Int=50,
    tree_every::Int=1,
) where {T<:Real}
    (C === nothing) == (H === nothing) &&
        throw(ArgumentError("give the starting loading as exactly one of C and H"))
    if basis === nothing
        C === nothing && (C = H)
    else
        C === nothing && (C = basis * H)
    end
    Cm = Matrix{T}(C)
    p = size(Cm, 1)
    return StructuredGaussianObservationModel{
        T,Matrix{T},Vector{T},basis === nothing ? Nothing : Matrix{T},typeof(prior)
    }(;
        C=Cm,
        R=Matrix{T}(R),
        d=d === nothing ? zeros(T, p) : Vector{T}(d),
        D=D === nothing ? zeros(T, p, 0) : Matrix{T}(D),
        R_prior,
        R_diagonal,
        basis=basis === nothing ? nothing : Matrix{T}(basis),
        prior,
        gauge,
        weight_iters,
        tree_every,
    )
end

function ConstrainedGaussianObservationModel(; basis::AbstractMatrix, kwargs...)
    for name in (:prior, :gauge, :weight_iters, :tree_every)
        haskey(kwargs, name) && throw(
            ArgumentError(
                "a ConstrainedGaussianObservationModel takes no `$name`; for a kernel " *
                "prior use ConstrainedKernelGaussianObservationModel",
            ),
        )
    end
    return _structured_model(; basis, prior=nothing, kwargs...)
end

function KernelGaussianObservationModel(; prior::KernelPrior, kwargs...)
    haskey(kwargs, :basis) && throw(
        ArgumentError(
            "a KernelGaussianObservationModel takes no basis; use " *
            "ConstrainedKernelGaussianObservationModel",
        ),
    )
    return _structured_model(; basis=nothing, prior, kwargs...)
end

function ConstrainedKernelGaussianObservationModel(;
    basis::AbstractMatrix, prior::KernelPrior, kwargs...
)
    return _structured_model(; basis, prior, kwargs...)
end

# The emissions whose residual scatter and `R` update are the Gaussian ones.
const _GaussianNoiseEmission{T} = Union{
    GaussianObservationModel{T},StructuredGaussianObservationModel{T}
}
