#=============================================================================
Hamiltonian (inverse-LQR) latents — driver glue.

The EM drivers in `fit_LDS.jl` / `fit_PLDS.jl` are already generic over the
state model: they call `_initialize_td_sufficient_statistics`, `estep!`,
`elbo!` and `mstep!` and know nothing else about it. This file supplies those
four for a Hamiltonian state model, plus the entry-point preparation every
public call needs (cache refresh, schedule check, costate-readout mask).
=============================================================================#

"""
    _prepare_hamiltonian!(lds, tsteps)

Bring a Hamiltonian LDS into a consistent state before inference:

1. rebuild the derived cache, so a model whose fields were assigned by hand
   still smooths with the parameters it now holds;
2. check that the cost schedule covers the longest trial;
3. pin the emission's costate columns at zero unless `observe_costate` is set.

Called at every public entry point (`fit!`, `elbo`, `smooth`, `loglikelihood`,
`rand`) and, crucially, *before* any parallel section — the cache is shared by
every trial workspace, and refreshing it is the only write to it during an
E-step.
"""
function _prepare_hamiltonian!(
    lds::LinearDynamicalSystem{T,S,O}, tsteps::AbstractVector{Int}
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    sm = lds.state_model
    refresh!(sm)
    _hamiltonian_lengths_ok(sm, tsteps)
    sm.observe_costate || _zero_costate_readout!(lds.obs_model, _plant_dim(sm))
    return nothing
end

"""
    _costate_range(lds) -> UnitRange{Int} or nothing

The latent indices the emission is *not* allowed to read: the costate half of
`z = [x; λ]` when `observe_costate` is off, and `nothing` for every other model.

One accessor drives all three masking sites — the emission Gram (Gaussian and
composite-Gaussian members), the Poisson row-wise Newton, and the readout
zeroing above — so no emission model needs to know the state model exists.
"""
_costate_range(::LinearDynamicalSystem) = nothing

function _costate_range(
    lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    sm = lds.state_model
    sm.observe_costate && return nothing
    n = _plant_dim(sm)
    return (n + 1):(2n)
end

"""
    _zero_costate_readout!(obs_model, n)

Zero the costate columns of every emission matrix. Warns once if they were not
already zero — the model *declares* (via `observe_costate = false`) that they
are, so a nonzero column is a stale value rather than a parameter, but silently
discarding user-supplied numbers would be worse than saying so.
"""
function _zero_costate_readout!(om::AbstractObservationModel{T}, n::Int) where {T<:Real}
    C = om.C
    size(C, 2) == 2n || return nothing
    cols = view(C, :, (n + 1):(2n))
    if any(!iszero, cols)
        @warn(
            "the emission's costate columns `C[:, $(n + 1):$(2n)]` were nonzero but the " *
                "Hamiltonian state model has `observe_costate = false`; zeroing them. Set " *
                "`observe_costate = true` on the state model to let the emission read the " *
                "costate.",
            maxlog = 1
        )
    end
    fill!(cols, zero(T))
    return nothing
end

function _zero_costate_readout!(c::CompositeObservationModel, n::Int)
    for om in values(_models(c))
        _zero_costate_readout!(om, n)
    end
    return nothing
end

"""
    _mask_costate_gram!(suf, range)

Decouple the costate block of the emission normal equations so the least-squares
solve returns exactly zero there.

Zeroing the masked rows and columns of the Gram (keeping their diagonal, which
preserves the scaling) and the masked rows of the cross-product makes the system
block diagonal: the free block solves against its own sub-Gram, and the masked
coefficients solve to zero. Because `C`'s masked columns *are* zero, every
downstream consumer — the `R` update, the emission Q-term — is numerically
unchanged by the masking, so this one edit covers every emission that ends in a
linear solve.
"""
function _mask_costate_gram!(suf::NamedTuple, range::UnitRange{Int})
    for member in values(suf)
        _mask_costate_gram!(member, range)
    end
    return suf
end

function _mask_costate_gram!(
    suf::SufficientStatistics{T}, range::UnitRange{Int}
) where {T<:Real}
    G = Matrix(suf.obs_xx[])
    for j in axes(G, 2), i in range
        i == j && continue
        G[i, j] = zero(T)
        G[j, i] = zero(T)
    end
    suf.obs_xx[] = pd_gram(G; name="emission Gram [x d uy]")
    @views fill!(suf.obs_xy[range, :], zero(T))
    return suf
end

"""
    _freeze_masked_rows!(grad, H, range)

Freeze a block of the Poisson emission's row-wise Newton system: zero the masked
entries of the gradient and decouple them in each row's curvature, so the Newton
direction is exactly zero there and the coefficients stay at the zero they
started from. The constrained maximization is therefore exact, not a projection
after the fact — which is what keeps the emission M-step monotone.
"""
_freeze_masked_rows!(::AbstractMatrix, ::AbstractArray, ::Nothing) = nothing

function _freeze_masked_rows!(
    grad::AbstractMatrix{T}, H::AbstractArray{T,3}, range::UnitRange{Int}
) where {T<:Real}
    obs_dim = size(grad, 2)
    reg_dim = size(grad, 1)
    @inbounds for nrow in 1:obs_dim
        for a in range
            grad[a, nrow] = zero(T)
            for b in 1:reg_dim
                b == a && continue
                H[a, b, nrow] = zero(T)
                H[b, a, nrow] = zero(T)
            end
            #=
            Keep a positive diagonal so the row's Cholesky still succeeds; its
            value is irrelevant because the corresponding gradient is zero.
            =#
            H[a, a, nrow] = max(H[a, a, nrow], one(T))
        end
    end
    return nothing
end

# ============================================================================
# E-step / ELBO / M-step, Gaussian (and composite-quadratic) emissions
# ============================================================================

"""
    estep!(lds, hs, tfs, data, sws_pool)

Hamiltonian E-step: smooth, run the shared aggregator for the initial-state and
emission halves, then accumulate the per-regime state-side statistics the
structural M-step needs.

The emission Gram is masked afterwards on the quadratic path, where the emission
M-step is a linear solve; the Poisson path is constrained inside its own Newton
iteration instead (see [`_freeze_masked_rows!`](@ref)).
"""
function estep!(
    lds::LinearDynamicalSystem{T,S,O},
    hs::HamiltonianSufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:QuadraticEmission{T}}
    smooth!(lds, tfs, data, sws_pool)
    _aggregate_td_suff_stats!(hs.base, tfs, lds, data, sws_pool[1])
    _aggregate_hamiltonian_stats!(hs, tfs, lds, data)
    mask = _costate_range(lds)
    mask === nothing || _mask_costate_gram!(hs.base, mask)
    return hs
end

function estep!(
    lds::LinearDynamicalSystem{T,S,O},
    hs::HamiltonianSufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}};
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:HamiltonianStateModel{T},O<:NonQuadraticEmission{T}}
    smooth!(lds, tfs, data, sws_pool; max_iter=max_iter, tol=tol)
    _aggregate_td_suff_stats!(hs.base, tfs, lds, data, sws_pool[1])
    _aggregate_hamiltonian_stats!(hs, tfs, lds, data)
    #=
    A composite may mix a Gaussian member (a linear solve, masked through the
    Gram) with a Poisson one (masked in its Newton system). Masking the Gram here
    is a no-op for the Poisson member, whose emission M-step never reads it.
    =#
    mask = _costate_range(lds)
    mask === nothing || _mask_costate_gram!(hs.base, mask)
    return hs
end

# ============================================================================
# Emission hooks
#
# The state half of the ELBO and the M-step is one code path; only the emission
# half varies, and each of these hooks is the ordinary single- or
# composite-emission routine reached with this model's statistics.
# ============================================================================

"""
    _ham_obs_mstep!(lds, hs, sws)
    _ham_obs_mstep!(lds, hs, tfs, data, sws_pool)

The emission half of the M-step: the conjugate regression for a Gaussian
emission, the row-wise Newton for a Poisson one, and one call per member for a
composite.
"""
function _ham_obs_mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    hs::HamiltonianSufficientStatistics{T},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:GaussianObservationModel{T}}
    update_C_d!(lds, hs.base, sws)
    update_R!(lds, hs.base, sws)
    return nothing
end

function _ham_obs_mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    hs::HamiltonianSufficientStatistics{T},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:CompositeObservationModel{T,true}}
    subs = _obs_workspaces!(sws, lds)
    views = _obs_views(lds)
    for (i, key) in enumerate(_obs_keys(lds.obs_model))
        update_C_d!(views[key], hs.base[key], subs[i])
        update_R!(views[key], hs.base[key], subs[i])
    end
    return nothing
end

function _ham_obs_mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    hs::HamiltonianSufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:PoissonObservationModel{T}}
    update_observation_model!(lds, tfs, data.y, sws_pool; uy=data.uy)
    return nothing
end

function _ham_obs_mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    hs::HamiltonianSufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:CompositeObservationModel{T,false}}
    views = _obs_views(lds)
    datas = _member_datas(data)
    pools = _member_pools(sws_pool, lds)
    for (i, key) in enumerate(_obs_keys(lds.obs_model))
        _member_obs_mstep!(views[key], hs.base[key], tfs, datas[key], pools[i])
    end
    return nothing
end

"""
    _ham_q_obs(lds, hs, tfs, data, sws_pool) -> T

Emission Q-term for a non-quadratic emission, which has no sufficient-statistic
form and so stays a per-trial loop.
"""
function _ham_q_obs(
    lds::LinearDynamicalSystem{T,S,O},
    ::HamiltonianSufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:PoissonObservationModel{T}}
    return _poisson_q_obs_total(lds, tfs, data, sws_pool)
end

function _ham_q_obs(
    lds::LinearDynamicalSystem{T,S,O},
    hs::HamiltonianSufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:CompositeObservationModel{T,false}}
    return _composite_q_obs_total(lds, hs.base, tfs, data, sws_pool)
end

# ============================================================================
# ELBO and M-step
# ============================================================================

"""
    elbo!(lds, hs, sws, total_entropy)

ELBO of a Hamiltonian LDS with a quadratic emission, from the aggregated
statistics: the structured state Q-term, the emission Q-term, the parameter
log-priors, and the posterior entropy.
"""
function elbo!(
    lds::LinearDynamicalSystem{T,S,O},
    hs::HamiltonianSufficientStatistics{T},
    sws::SmoothWorkspace{T},
    total_entropy::T,
) where {T<:Real,S<:HamiltonianStateModel{T},O<:QuadraticEmission{T}}
    _fill_mixed_blocks!(hs, lds.state_model)
    Q_total = Q_state!(sws, lds, hs) + Q_obs!(sws, lds, hs.base)
    prior_term = _state_prior_logdensity(lds, sws) + _obs_prior_logdensity(lds, sws)
    return Q_total + prior_term + total_entropy
end

"""
    elbo!(lds, hs, tfs, data, sws_pool)

ELBO of a Hamiltonian LDS with a non-quadratic emission. Only the state half
differs from the ordinary Laplace path; the emission Q-term stays the per-trial
loop that emission already uses.
"""
function elbo!(
    lds::LinearDynamicalSystem{T,S,O},
    hs::HamiltonianSufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:NonQuadraticEmission{T}}
    total_entropy = zero(T)
    for fs in tfs.FilterSmooths
        total_entropy += fs.entropy
    end
    compute_smooth_constants!(sws_pool[1], lds)
    _fill_mixed_blocks!(hs, lds.state_model)
    Q_total = Q_state!(sws_pool[1], lds, hs) + _ham_q_obs(lds, hs, tfs, data, sws_pool)
    prior_term =
        _state_prior_logdensity(lds, sws_pool[1]) + _obs_prior_logdensity(lds, sws_pool[1])
    return Q_total + prior_term + total_entropy
end

"""
    mstep!(lds, hs, sws)
    mstep!(lds, hs, tfs, data, sws_pool)

M-step for a Hamiltonian LDS: the shared `x0` / `P0` updates, the structural
update (`fit_bool[3]`), the noise update (`fit_bool[4]`), a cache refresh, and
then the emission's own update.
"""
function mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    hs::HamiltonianSufficientStatistics{T},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:QuadraticEmission{T}}
    _ham_state_mstep!(lds, hs, sws)
    _ham_obs_mstep!(lds, hs, sws)
    return nothing
end

function mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    hs::HamiltonianSufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:NonQuadraticEmission{T}}
    _ham_state_mstep!(lds, hs, sws_pool[1])
    _ham_obs_mstep!(lds, hs, tfs, data, sws_pool)
    return nothing
end

# ============================================================================
# Public entry points
# ============================================================================

"""
    fit!(lds::LinearDynamicalSystem{T,<:HamiltonianStateModel}, y; kwargs...)

Fit a Hamiltonian (inverse-LQR) LDS by EM. Accepts the same arguments as the
[`fit!`](@ref) for any other `LinearDynamicalSystem` and returns the per-iteration
ELBO.

The E-step is an ordinary linear-Gaussian smoother on `z = [x; λ]`; the M-step
re-estimates the LQR structure (see `hamiltonian_mstep.jl`). What comes back is
a plant `A`, a control term `S = B R⁻¹ Bᵀ` and the state cost(s) `Qc` — read
them with [`lqr_parameters`](@ref).

`depends_on` grouping is not supported for this state model.
"""
function fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
    ux=nothing,
    uy=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:HamiltonianStateModel{T},O<:QuadraticEmission{T}}
    depends_on === nothing || throw(
        ArgumentError(
            "`depends_on` grouping is not supported for a HamiltonianStateModel; fit the " *
            "groups as separate models",
        ),
    )
    data = Data(lds, y; ux=ux, uy=uy)
    _prepare_hamiltonian!(lds, data.tsteps)
    return _fit_tridiag!(lds, data; max_iter=max_iter, tol=tol, progress=progress)
end

"""
    elbo(lds::LinearDynamicalSystem{T,<:HamiltonianStateModel}, y; ux, uy)

Evidence lower bound of a Hamiltonian LDS at its current parameters. With a
Gaussian emission the smoother is exact, so with no parameter priors this equals
the marginal log-likelihood.
"""
function elbo(
    lds::LinearDynamicalSystem{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    ux=nothing,
    uy=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:HamiltonianStateModel{T},O<:QuadraticEmission{T}}
    depends_on === nothing || throw(
        ArgumentError("`depends_on` grouping is not supported for a HamiltonianStateModel"),
    )
    data = Data(lds, y; ux=ux, uy=uy)
    _prepare_hamiltonian!(lds, data.tsteps)
    tfs = initialize_FilterSmooth(lds, data.tsteps)::TrialFilterSmooth{T}
    sws_pool = _ham_sws_pool(lds, data)
    hs = _initialize_td_sufficient_statistics(T, lds, data.tsteps)
    _td_init_const_blocks!(sws_pool[1], lds, data)
    estep!(lds, hs, tfs, data, sws_pool)
    total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths; init=zero(T))
    return elbo!(lds, hs, sws_pool[1], total_entropy)
end

"""
    loglikelihood(lds::LinearDynamicalSystem{T,<:HamiltonianStateModel}, y; ux, uy)

Marginal log-likelihood `log p(y)` of a Hamiltonian LDS with a Gaussian
emission.

The Kalman route used for a plain Gaussian LDS assumes one time-invariant
transition read straight off the model and has no place for the terminal factor,
neither of which holds here. Instead this is the ELBO at the exact posterior —
identical to the marginal likelihood for a linear-Gaussian model — with the
parameter log-priors removed, so it is a likelihood and not a MAP objective.
"""
function StatsAPI.loglikelihood(
    lds::LinearDynamicalSystem{T,S,O},
    y::Union{AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}}};
    ux=nothing,
    uy=nothing,
) where {T<:Real,S<:HamiltonianStateModel{T},O<:GaussianObservationModel{T}}
    return _ham_loglikelihood(lds, y; ux=ux, uy=uy)
end

#=
A composite emission has its own `loglikelihood` at the same `y::NamedTuple`
specificity, so this method has to match it there — otherwise the two are
ambiguous, each more specific in a different argument.
=#
function StatsAPI.loglikelihood(
    lds::LinearDynamicalSystem{T,S,O}, y::NamedTuple; ux=nothing, uy=nothing
) where {T<:Real,S<:HamiltonianStateModel{T},O<:CompositeObservationModel{T,true}}
    return _ham_loglikelihood(lds, y; ux=ux, uy=uy)
end

function _ham_loglikelihood(
    lds::LinearDynamicalSystem{T,S,O}, y; ux=nothing, uy=nothing
) where {T<:Real,S<:HamiltonianStateModel{T},O<:QuadraticEmission{T}}
    data = Data(lds, y; ux=ux, uy=uy)
    _prepare_hamiltonian!(lds, data.tsteps)
    tfs = initialize_FilterSmooth(lds, data.tsteps)::TrialFilterSmooth{T}
    sws_pool = _ham_sws_pool(lds, data)
    hs = _initialize_td_sufficient_statistics(T, lds, data.tsteps)
    _td_init_const_blocks!(sws_pool[1], lds, data)
    estep!(lds, hs, tfs, data, sws_pool)
    total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths; init=zero(T))
    full = elbo!(lds, hs, sws_pool[1], total_entropy)
    return full - _state_prior_logdensity(lds, sws_pool[1]) -
           _obs_prior_logdensity(lds, sws_pool[1])
end

"""
    smooth(lds::LinearDynamicalSystem{T,<:HamiltonianStateModel}, y; ux, uy)

Posterior over the state–costate path. Returns `(x_smooth, p_smooth, ll)` in the
same shapes as every other `smooth`; rows `1:n` of `x_smooth` are the state and
rows `n+1:2n` the costate.
"""
function smooth(
    lds::LinearDynamicalSystem{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    ux=nothing,
    uy=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:HamiltonianStateModel{T},O<:QuadraticEmission{T}}
    depends_on === nothing || throw(
        ArgumentError("`depends_on` grouping is not supported for a HamiltonianStateModel"),
    )
    data = Data(lds, y; ux=ux, uy=uy)
    _prepare_hamiltonian!(lds, data.tsteps)
    tfs = initialize_FilterSmooth(lds, data.tsteps)::TrialFilterSmooth{T}
    smooth!(lds, tfs, data, _ham_sws_pool(lds, data))
    return _collect_smooth_output(tfs, y)
end

"""
    _ham_sws_pool(lds, data) -> Vector{SmoothWorkspace}

Workspace pool sized for this model and dataset, capped at the trial count —
workspaces beyond `ntrials` are never touched and each carries `O(D²T)` of
block-tridiagonal storage.
"""
function _ham_sws_pool(
    lds::LinearDynamicalSystem{T,S,O}, data::Data{T}
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    npool = min(Threads.maxthreadid(), length(data.tsteps))
    return [
        SmoothWorkspace(
            T,
            lds.latent_dim,
            _ws_obs_dim(lds),
            maximum(data.tsteps);
            ux_dim=lds.ux_dim,
            uy_dim=_ws_uy_dim(lds),
        ) for _ in 1:npool
    ]
end

function smooth(
    lds::LinearDynamicalSystem{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    ux=nothing,
    uy=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:HamiltonianStateModel{T},O<:NonQuadraticEmission{T}}
    depends_on === nothing || throw(
        ArgumentError("`depends_on` grouping is not supported for a HamiltonianStateModel"),
    )
    data = Data(lds, y; ux=ux, uy=uy)
    _prepare_hamiltonian!(lds, data.tsteps)
    tfs = initialize_FilterSmooth(lds, data.tsteps)::TrialFilterSmooth{T}
    sws_pool = _ham_sws_pool(lds, data)
    smooth!(lds, tfs, data, sws_pool)
    return _collect_smooth_output(tfs, y)
end

"""
    fit!(lds::LinearDynamicalSystem{T,<:HamiltonianStateModel,<:NonQuadraticEmission}, y; ...)

Fit a Hamiltonian LDS with a Poisson (or mixed) emission by Laplace EM. Same
arguments as the Poisson [`fit!`](@ref), including the inner Newton controls.
"""
function fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    ux=nothing,
    uy=nothing,
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress=true,
    newton_max_iter::Int=20,
    newton_tol::Float64=1e-6,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:HamiltonianStateModel{T},O<:NonQuadraticEmission{T}}
    depends_on === nothing || throw(
        ArgumentError("`depends_on` grouping is not supported for a HamiltonianStateModel"),
    )
    data = Data(lds, y; ux=ux, uy=uy)
    _prepare_hamiltonian!(lds, data.tsteps)
    return _fit_laplace!(
        lds,
        data;
        max_iter=max_iter,
        tol=tol,
        progress=progress,
        newton_max_iter=newton_max_iter,
        newton_tol=newton_tol,
    )
end

"""
    elbo(lds::LinearDynamicalSystem{T,<:HamiltonianStateModel,<:NonQuadraticEmission}, y; ...)

Evidence lower bound of a Hamiltonian LDS with a Poisson emission, at the
current parameters: one Laplace E-step, then the ELBO at the resulting Gaussian
posterior approximation.
"""
function elbo(
    lds::LinearDynamicalSystem{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    ux=nothing,
    uy=nothing,
    newton_max_iter::Int=20,
    newton_tol::Float64=1e-6,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:HamiltonianStateModel{T},O<:NonQuadraticEmission{T}}
    depends_on === nothing || throw(
        ArgumentError("`depends_on` grouping is not supported for a HamiltonianStateModel"),
    )
    data = Data(lds, y; ux=ux, uy=uy)
    _prepare_hamiltonian!(lds, data.tsteps)
    tfs = initialize_FilterSmooth(lds, data.tsteps)::TrialFilterSmooth{T}
    sws_pool = _ham_sws_pool(lds, data)
    hs = _initialize_td_sufficient_statistics(T, lds, data.tsteps)
    _td_init_const_blocks!(sws_pool[1], lds, data)
    estep!(lds, hs, tfs, data, sws_pool; max_iter=newton_max_iter, tol=T(newton_tol))
    return elbo!(lds, hs, tfs, data, sws_pool)
end

# ============================================================================
# Batched mean pass (equal-length multi-trial fast path)
# ============================================================================

"""
    gradient_batched!(ws, lds, x, y, ux, uy)

Batched gradient for a Hamiltonian LDS with a Gaussian emission — the same
promotion of every `mul!` from BLAS-2 to BLAS-3 by stacking the trial axis that
the Gaussian version does, which is what keeps an aligned-epoch fit (every trial
the same length, the common case here) on the shared-covariance fast path.

Structurally identical to that version, with three substitutions: the transition
is `M_{k(t)}` looked up per timestep rather than one `A`; the bias and input
matrix are the *forward* ones `G h` and `G B_u`; and the last timestep also
carries the terminal factor's `−Λfᵀ Σf⁻¹ (Λf z_T − h_f)`.

Note which regime each term uses: the factor *entering* `t` was the transition
`t-1 → t` and so uses `k(t-1)`, while the factor *leaving* `t` uses `k(t)`.
"""
function gradient_batched!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    ux::AbstractArray{T,3},
    uy::AbstractArray{T,3},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(x, 2)
    sm = lds.state_model
    c = sm.cache
    x0 = sm.x0
    bf = c.bfwd
    Bf = c.Bfwd
    has_input = size(Bf, 2) > 0
    C = lds.obs_model.C
    d_obs = lds.obs_model.d
    D_obs = lds.obs_model.D

    C_inv_R = ws.consts.C_inv_R
    neg_P0_inv = ws.consts.x_t            # −P0⁻¹
    neg_Q_inv = c.negQinv                 # −Qfwd⁻¹

    bat = ws.batched::BatchedBuffers{T}
    grad = bat.grad_buf
    dxt = bat.dxt
    dxt_next = bat.dxt_next
    dyt = bat.dyt
    tmp1 = bat.tmp1
    tmp2 = bat.tmp2
    tmp3 = bat.tmp3

    # Transition residual into `dst` for the step t-1 → t, batched over trials.
    @inline function residual!(dst, t)
        @views begin
            mul!(dst, c.M[_regime(sm, t - 1)], x[:, t - 1, :])
            has_input && mul!(dst, Bf, ux[:, t - 1, :], one(T), one(T))
            dst .= x[:, t, :] .- dst .- bf
        end
        return dst
    end

    @inline function emission!(t)
        @views begin
            mul!(dyt, C, x[:, t, :])
            mul!(dyt, D_obs, uy[:, t, :], one(T), one(T))
            dyt .= y[:, t, :] .- dyt .- d_obs
        end
        return mul!(tmp1, C_inv_R, dyt)
    end

    @views dxt .= x[:, 1, :] .- x0
    residual!(dxt_next, 2)
    emission!(1)
    mul!(tmp2, c.MtQinv[_regime(sm, 1)], dxt_next)
    mul!(tmp3, neg_P0_inv, dxt)
    @views grad[:, 1, :] .= tmp1 .+ tmp2 .+ tmp3

    @views for t in 2:(tsteps - 1)
        residual!(dxt, t)
        residual!(dxt_next, t + 1)
        emission!(t)
        mul!(tmp2, c.MtQinv[_regime(sm, t)], dxt_next)
        mul!(tmp3, neg_Q_inv, dxt)
        grad[:, t, :] .= tmp1 .+ tmp3 .+ tmp2
    end

    residual!(dxt, tsteps)
    emission!(tsteps)
    mul!(tmp3, neg_Q_inv, dxt)
    @views grad[:, tsteps, :] .= tmp1 .+ tmp3

    if sm.terminal
        n = _plant_dim(sm)
        @views begin
            rf = tmp2[1:n, :]
            mul!(rf, c.Lf, x[:, tsteps, :])
            rf .-= sm.hf
            mul!(grad[:, tsteps, :], c.LtSinv, rf, -one(T), one(T))
        end
    end

    return grad
end

# ============================================================================
# Sampling
# ============================================================================

"""
    _sample_hamiltonian_path!(rng, z, sm, ux)

Draw one latent state–costate path from the forward form
`z_{t+1} = M_{k(t)} z_t + b + B u_t + w`. The regime lookup per step is the only
difference from the Gaussian sampler; the terminal factor is a *conditioning*
event, not part of the generative chain, so a sampled path satisfies the terminal
condition only up to `Σf` — draw with a small `Σf` if you want trajectories that
look terminally constrained.
"""
function _sample_hamiltonian_path!(
    rng::AbstractRNG,
    z::AbstractMatrix{T},
    lds::LinearDynamicalSystem{T,S,O},
    ux::AbstractMatrix{T},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    sm = lds.state_model
    #=
    Without a terminal factor the state factors form an ordinary Markov chain, so
    the forward recursion *is* the model's distribution and is O(T d²). With one,
    the terminal condition is a factor the chain does not generate — it is
    conditioned on, not sampled — so the forward roll would draw from the wrong
    distribution. The joint is still Gaussian with a block-tridiagonal precision,
    which is what the branch below samples.
    =#
    sm.terminal && return _sample_hamiltonian_path_conditional!(rng, z, lds, ux)

    c = sm.cache
    tsteps = size(z, 2)
    _warn_unstable_rollout(sm, tsteps)
    P0 = MvNormal(Vector{T}(sm.x0), Matrix(Symmetrize!(Matrix{T}(sm.P0))))
    Qd = MvNormal(zeros(T, size(z, 1)), Matrix(c.Qfwd))
    z[:, 1] = rand(rng, P0)
    has_input = size(c.Bfwd, 2) > 0
    for t in 2:tsteps
        k = _regime(sm, t - 1)
        @views begin
            mul!(z[:, t], c.M[k], z[:, t - 1])
            z[:, t] .+= c.bfwd
            has_input && mul!(z[:, t], c.Bfwd, ux[:, t - 1], one(T), one(T))
            z[:, t] .+= rand(rng, Qd)
        end
    end
    return z
end

"""
    _sample_hamiltonian_path_conditional!(rng, z, lds, ux)

Draw a path from the state-factor joint of a model that carries a terminal
factor, i.e. from

```math
p(z_{1:T}) \\propto p(z_1)\\prod_t p(z_{t+1}\\mid z_t)\\, p(y^{\\text{term}} = 0\\mid z_T).
```

That is Gaussian with a block-tridiagonal precision `H` and linear term `g`,
both of which the smoother kernels already build: `H` is the negated state-side
Hessian and `g` is the state-side gradient evaluated at `z = 0` (the gradient is
affine, `∇ = g − H z`). The path is then `μ = H⁻¹g` plus `U⁻¹ξ` for `H = UᵀU`.

Conditioning on the terminal factor is exactly what removes the transition's
unstable directions, so unlike the forward roll this stays bounded — the price
is one `(dT)²` factorization, which is why long trials are refused rather than
silently attempted.
"""
function _sample_hamiltonian_path_conditional!(
    rng::AbstractRNG,
    z::AbstractMatrix{T},
    lds::LinearDynamicalSystem{T,S,O},
    ux::AbstractMatrix{T},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    sm = lds.state_model
    d = lds.latent_dim
    tsteps = size(z, 2)
    tsteps >= 2 || throw(ArgumentError("sampling a Hamiltonian path needs tsteps ≥ 2"))
    ndof = d * tsteps
    ndof <= 4000 || throw(
        ArgumentError(
            "sampling a terminal-conditioned Hamiltonian path of $tsteps steps at latent " *
            "dimension $d needs a $(ndof)×$(ndof) factorization. Use `simulate_lqr` for " *
            "long trajectories — it follows the stable manifold directly and costs O(T).",
        ),
    )

    sws = SmoothWorkspace(
        T, d, _ws_obs_dim(lds), tsteps; ux_dim=lds.ux_dim, uy_dim=_ws_uy_dim(lds)
    )
    compute_smooth_constants!(sws, lds)
    btd = sws.btd
    _state_hessian_blocks!(btd, sws.consts, sm, tsteps)

    # Precision = −(log-density Hessian). `H_sub` is the sub-diagonal block, so
    # it is `block_tridgm`'s `lower_diag` and `H_super` its `upper_diag`.
    main = [Matrix{T}(-btd.H_diag[t]) for t in 1:tsteps]
    upper = [Matrix{T}(-btd.H_super[i]) for i in 1:(tsteps - 1)]
    lower = [Matrix{T}(-btd.H_sub[i]) for i in 1:(tsteps - 1)]
    H = Matrix(block_tridgm(main, upper, lower))
    Symmetrize!(H)

    fill!(z, zero(T))
    grad = view(sws.opt.grad_buf, :, 1:tsteps)
    _state_gradient!(grad, sws, lds, z, ux)

    F = cholesky!(Symmetric(H))
    mean_vec = F \ vec(Matrix(grad))
    ξ = randn(rng, T, ndof)
    ldiv!(F.U, ξ)
    copyto!(z, reshape(mean_vec .+ ξ, d, tsteps))
    return z
end

"""
    _warn_unstable_rollout(sm, tsteps)

Warn once when rolling the forward transition for `tsteps` steps is predicted to
diverge. `ρ(M)^T` is the growth of the fastest mode; past roughly `1/eps` the
sampled path carries no usable signal, and the caller almost certainly wants
[`simulate_lqr`](@ref).
"""
function _warn_unstable_rollout(sm::HamiltonianStateModel{T}, tsteps::Int) where {T<:Real}
    ρ = maximum(abs, eigvals(sm.cache.M[_regime(sm, 1)]))
    growth = ρ^tsteps
    if growth > 1 / sqrt(eps(T))
        @warn(
            "rolling this Hamiltonian model forward for $tsteps steps grows the fastest " *
                "mode by ~$(round(growth; sigdigits = 2)). A symplectic transition has " *
                "reciprocal eigenvalue pairs, so its forward flow is unstable by " *
                "construction and `rand` samples the model\'s own (divergent) prior. Use " *
                "`simulate_lqr` for trajectories on the stable manifold.",
            spectral_radius = ρ,
            maxlog = 1
        )
    end
    return nothing
end

"""
    _sample_hamiltonian_obs!(rng, y, z, obs_model, obs_params, uy)

Observations drawn from an already-sampled latent path, for one emission or for
each member of a composite. The Gaussian path interleaves the state and
observation recursions; here the path is drawn first (its regime lookup does not
factor through `_sample_trial!`), so the emission half is taken separately.
"""
function _sample_hamiltonian_obs!(
    rng, y, z::AbstractMatrix, om::AbstractObservationModel, obs_params, uy
)
    _sample_obs!(rng, y, obs_params, om, z, uy)
    return nothing
end

function _sample_hamiltonian_obs!(
    rng, y::NamedTuple, z::AbstractMatrix, om::CompositeObservationModel, obs_params, uy
)
    for (i, m) in enumerate(values(_models(om)))
        _sample_obs!(rng, y[i], obs_params[i], m, z, uy[i])
    end
    return nothing
end

"""
    rand([rng,] lds::LinearDynamicalSystem{T,<:HamiltonianStateModel}, tsteps; ux, uy)
    rand([rng,] lds, tsteps_per_trial::AbstractVector; ux, uy)

Sample state–costate paths and observations from a Hamiltonian LDS, in the same
shapes as [`rand`](@ref) for any other `LinearDynamicalSystem`. The returned
latent has `2n` rows: `1:n` the state, `n+1:2n` the costate.

!!! warning "This rolls an unstable recursion"
    A Hamiltonian matrix has reciprocal eigenvalue pairs `(μ, 1/μ)`, so half the
    forward transition's modes grow — the optimal trajectory lives on the stable
    manifold, and the terminal boundary condition is what selects it. This
    function samples the model's *own* generative form, `z_{t+1} = M z_t + w`,
    which therefore diverges like `ρ(M)^T` over any useful horizon.
    [`simulate_lqr`](@ref) follows the stable manifold instead and is what you
    want for ground-truth trajectories; a warning fires here when the horizon
    makes divergence likely.

    Inference is unaffected: conditioning on the data (and on the terminal
    factor, when present) concentrates the posterior back onto the stable
    manifold, which is why smoothing and fitting behave while sampling does not.
"""
function Random.rand(
    rng::AbstractRNG,
    lds::LinearDynamicalSystem{T,S,O},
    tsteps::Integer;
    ux::Union{Nothing,AbstractMatrix{T}}=nothing,
    uy::Union{Nothing,AbstractMatrix{T}}=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    depends_on === nothing || throw(
        ArgumentError("`depends_on` grouping is not supported for a HamiltonianStateModel"),
    )
    Ti = Int(tsteps)
    _prepare_hamiltonian!(lds, [Ti])
    ux_trial = _check_ux(ux, lds.ux_dim, Ti, "ux", T)
    uy_trial = _check_uy(uy, lds.uy_dim, Ti, lds.obs_model)

    z = Matrix{T}(undef, lds.latent_dim, Ti)
    _sample_hamiltonian_path!(rng, z, lds, ux_trial)
    y = _alloc_obs(lds, Ti)
    _sample_hamiltonian_obs!(
        rng, y, z, lds.obs_model, _extract_obs_params(lds.obs_model), uy_trial
    )
    return z, y
end

function Random.rand(
    rng::AbstractRNG,
    lds::LinearDynamicalSystem{T,S,O},
    tsteps_per_trial::AbstractVector{<:Integer};
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    depends_on === nothing || throw(
        ArgumentError("`depends_on` grouping is not supported for a HamiltonianStateModel"),
    )
    ntrials = length(tsteps_per_trial)
    lengths = Int[Int(t) for t in tsteps_per_trial]
    _prepare_hamiltonian!(lds, lengths)
    ux_seq = _normalize_multitrial_ux(ux, lds.ux_dim, lengths, T, "ux")
    uy_seq = _normalize_multitrial_uy(uy, lds.uy_dim, lengths, T, lds.obs_model)
    obs_params = _extract_obs_params(lds.obs_model)

    z = Vector{Matrix{T}}(undef, ntrials)
    y = Vector{typeof(_alloc_obs(lds, 1))}(undef, ntrials)
    #=
    Sampling stays serial: a Hamiltonian draw is a short matrix recursion, and a
    single RNG keeps a given seed reproducible without the child-RNG bookkeeping
    the Gaussian sampler needs for its parallel chunks.
    =#
    for i in 1:ntrials
        z[i] = Matrix{T}(undef, lds.latent_dim, lengths[i])
        y[i] = _alloc_obs(lds, lengths[i])
        _sample_hamiltonian_path!(rng, z[i], lds, ux_seq[i])
        _sample_hamiltonian_obs!(
            rng, y[i], z[i], lds.obs_model, obs_params, _trial(uy_seq, i)
        )
    end
    return z, y
end

function Random.rand(
    lds::LinearDynamicalSystem{T,S,O}, tsteps::Integer; kwargs...
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    return rand(Random.default_rng(), lds, tsteps; kwargs...)
end

function Random.rand(
    lds::LinearDynamicalSystem{T,S,O},
    tsteps_per_trial::AbstractVector{<:Integer};
    kwargs...,
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    return rand(Random.default_rng(), lds, tsteps_per_trial; kwargs...)
end
