# =============================================================================
# Group-aware fitting for models with `Varying` parameters.
#
# When any parameter of an LDS/PLDS/SLDS is a `Varying` (see `indexed.jl`), the
# public `fit!` / `smooth` / `loglikelihood` / `rand` / `elbo!` route here. Trials
# are partitioned into *regimes* — unique combinations of the six estimation
# blocks' per-trial group indices (x0, P0, [A b B], Q, [C d D], R). Within a
# regime every parameter is fixed, so a regime is an ordinary plain
# `LinearDynamicalSystem` materialized from the `Varying`s via `at` (which returns
# the stored arrays by reference, so in-place M-step `copyto!`s propagate back).
#
# The E-step runs the *existing* multi-trial smoother + sufficient-statistics
# aggregator per regime — so the fixed-trial-length block-tridiagonal fast path
# (shared covariance + batched mean pass) still fires within each regime, and
# regimes with different `obs_dim` are simply smoothed in separate batches. The
# M-step updates each block per group:
#   * mean blocks (x0, [A b B], [C d D]) — combine raw sufficient statistics over
#     the regimes in a group and run the existing regression (`update_A_b!` /
#     `update_C_d!`), which handles the input matrices `B`/`D` and MN priors;
#   * covariance blocks (P0, Q, R) — combine *per-regime residual scatters* (each
#     computed with that regime's updated mean), so a covariance block may be
#     grouped independently of its mean block.
#
# Block-consistency (`[A b B]` share one label/grouping, `[C d D]` share one) is
# enforced by `validate_LDS`, so a block's grouping is read from a representative
# parameter (`A` for dynamics, `C` for emission).
# =============================================================================

# --- detection -------------------------------------------------------------

_has_indexed(sm::GaussianStateModel) =
    is_indexed(sm.A) || is_indexed(sm.Q) || is_indexed(sm.b) ||
    is_indexed(sm.x0) || is_indexed(sm.P0) || is_indexed(sm.B)
_has_indexed(om::GaussianObservationModel) =
    is_indexed(om.C) || is_indexed(om.R) || is_indexed(om.d) || is_indexed(om.D)
_has_indexed(om::PoissonObservationModel) = is_indexed(om.C) || is_indexed(om.d)
_has_indexed(lds::LinearDynamicalSystem) =
    _has_indexed(lds.state_model) || _has_indexed(lds.obs_model)
_has_indexed(slds::SLDS) = any(_has_indexed, slds.LDSs)

# `true` iff a fit needs the group-aware path (any `Varying` present).
_has_varying(sm::GaussianStateModel) =
    is_varying(sm.A) || is_varying(sm.Q) || is_varying(sm.b) ||
    is_varying(sm.x0) || is_varying(sm.P0) || is_varying(sm.B)
_has_varying(om::GaussianObservationModel) =
    is_varying(om.C) || is_varying(om.R) || is_varying(om.d) || is_varying(om.D)
_has_varying(om::PoissonObservationModel) = is_varying(om.C) || is_varying(om.d)
_has_varying(lds::LinearDynamicalSystem) =
    _has_varying(lds.state_model) || _has_varying(lds.obs_model)
_has_varying(slds::SLDS) = any(_has_varying, slds.LDSs)

# Resolve the `labels` kwarg for the group-aware path: `nothing` is only allowed
# when there are no `Varying` parameters (an all-`Static` model needs no labels).
function _resolve_labels(model, labels)
    labels === nothing || return labels
    _has_varying(model) && throw(
        ArgumentError(
            "`labels` (a Dict{Symbol,Vector}) is required when the model has Varying parameters",
        ),
    )
    return Dict{Symbol,Vector}()
end

# --- per-trial group indices + regimes -------------------------------------

"""
    _group_indices(param, labels, ntrials) -> Vector{Int}

Per-trial group index for one parameter. Plain/`Static` params map every trial to
group 1; a `Varying` looks up each trial's label value (from `labels[param.label]`)
in its `group_ids`.
"""
function _group_indices(param, labels, ntrials::Int)
    is_varying(param) || return ones(Int, ntrials)
    key = param.label
    haskey(labels, key) || throw(
        ArgumentError("labels is missing key :$(key) required by a Varying parameter")
    )
    lab = labels[key]
    length(lab) == ntrials ||
        throw(DimensionMismatchError("labels[:$(key)] length", ntrials, length(lab)))
    gids = param.group_ids
    idx = Vector{Int}(undef, ntrials)
    for t in 1:ntrials
        j = findfirst(==(lab[t]), gids)
        j === nothing && throw(
            ArgumentError(
                "label value $(repr(lab[t])) for :$(key) (trial $t) not in group_ids $(gids)",
            ),
        )
        idx[t] = j
    end
    return idx
end

# Resolved grouping for a whole model (six block slots + regimes).
struct _IndexGroups
    gx0::Vector{Int}
    gP0::Vector{Int}
    gdyn::Vector{Int}
    gQ::Vector{Int}
    gobs::Vector{Int}
    gR::Vector{Int}
    regimes::Vector{NTuple{6,Int}}       # unique block-index tuples
    regime_trials::Vector{Vector{Int}}   # trial indices per regime
    by_x0::Vector{Vector{Int}}           # regime indices grouped by each block
    by_P0::Vector{Vector{Int}}
    by_dyn::Vector{Vector{Int}}
    by_Q::Vector{Vector{Int}}
    by_obs::Vector{Vector{Int}}
    by_R::Vector{Vector{Int}}
end

_regimes_by(regimes, comp, ng) =
    [findall(r -> regimes[r][comp] == g, eachindex(regimes)) for g in 1:ng]

function _index_groups(sm, om, labels, ntrials::Int; has_R::Bool)
    gx0 = _group_indices(sm.x0, labels, ntrials)
    gP0 = _group_indices(sm.P0, labels, ntrials)
    gdyn = _group_indices(sm.A, labels, ntrials)     # A, b, B share (validated)
    gQ = _group_indices(sm.Q, labels, ntrials)
    gobs = _group_indices(om.C, labels, ntrials)     # C, d, D share (validated)
    gR = has_R ? _group_indices(om.R, labels, ntrials) : ones(Int, ntrials)

    key2id = Dict{NTuple{6,Int},Int}()
    regimes = NTuple{6,Int}[]
    regime_id = Vector{Int}(undef, ntrials)
    for t in 1:ntrials
        key = (gx0[t], gP0[t], gdyn[t], gQ[t], gobs[t], gR[t])
        regime_id[t] = get!(key2id, key) do
            push!(regimes, key)
            length(regimes)
        end
    end
    regime_trials = [Int[] for _ in eachindex(regimes)]
    for t in 1:ntrials
        push!(regime_trials[regime_id[t]], t)
    end
    return _IndexGroups(
        gx0, gP0, gdyn, gQ, gobs, gR, regimes, regime_trials,
        _regimes_by(regimes, 1, nvals(sm.x0)),
        _regimes_by(regimes, 2, nvals(sm.P0)),
        _regimes_by(regimes, 3, nvals(sm.A)),
        _regimes_by(regimes, 4, nvals(sm.Q)),
        _regimes_by(regimes, 5, nvals(om.C)),
        _regimes_by(regimes, 6, has_R ? nvals(om.R) : 1),
    )
end

# --- materialization: plain per-regime LDS (references the Varying storage) --

function _resolve_state(sm::GaussianStateModel, gdyn, gQ, gx0, gP0)
    return GaussianStateModel(;
        A=at(sm.A, gdyn), Q=at(sm.Q, gQ), b=at(sm.b, gdyn), x0=at(sm.x0, gx0),
        P0=at(sm.P0, gP0), B=at(sm.B, gdyn), Q_prior=sm.Q_prior, P0_prior=sm.P0_prior,
        AB_prior=sm.AB_prior,
    )
end

function _resolve_obs(om::GaussianObservationModel, gobs, gR)
    C = at(om.C, gobs)
    # A no-input (zero-column) D must carry this regime's obs_dim rows so the
    # emission `D·uy` broadcasts correctly when obs_dim varies across groups.
    D = _has_input(om.D) ? at(om.D, gobs) : zeros(eltype(C), size(C, 1), 0)
    return GaussianObservationModel(;
        C=C, R=at(om.R, gR), d=at(om.d, gobs), D=D, R_prior=om.R_prior, CD_prior=om.CD_prior,
    )
end

function _resolve_obs(om::PoissonObservationModel, gobs, _gR)
    return PoissonObservationModel(; C=at(om.C, gobs), d=at(om.d, gobs), CD_prior=om.CD_prior)
end

function _resolve_lds(lds::LinearDynamicalSystem, regime::NTuple{6,Int})
    gx0, gP0, gdyn, gQ, gobs, gR = regime
    state = _resolve_state(lds.state_model, gdyn, gQ, gx0, gP0)
    obs = _resolve_obs(lds.obs_model, gobs, gR)
    return LinearDynamicalSystem(state, obs; fit_bool=lds.fit_bool)
end

# --- sufficient-statistic combination + residual scatters -------------------
# (Covariance blocks may be grouped independently of their mean, so we combine
#  per-regime scatters rather than raw suff stats.)

# Combine the *state-side* blocks (init/dyn) of `sufs[idxs]` into `dst`. Valid for
# a dynamics group whose regimes may differ in `obs_dim` (state blocks are
# latent-sized, so `dst`'s obs block is irrelevant and left untouched). Feeds the
# `[A b B]` regression (`update_A_b!` reads `dyn_xx`/`dyn_xy`).
function _combine_state_blocks!(
    dst::SufficientStatistics{T}, sufs::AbstractVector{SufficientStatistics{T}}, idxs
) where {T<:Real}
    D = size(dst.init_xy, 2)
    dr = size(dst.dyn_xy, 1)
    init_n = dyn_n = zero(T)
    fill!(dst.init_xy, zero(T))
    fill!(dst.dyn_xy, zero(T))
    init_yy = zeros(T, D, D)
    dyn_xx = zeros(T, dr, dr)
    dyn_yy = zeros(T, D, D)
    for i in idxs
        s = sufs[i]
        init_n += s.init_n
        dyn_n += s.dyn_n
        dst.init_xy .+= s.init_xy
        dst.dyn_xy .+= s.dyn_xy
        init_yy .+= s.init_yy[].mat
        dyn_xx .+= s.dyn_xx[].mat
        dyn_yy .+= s.dyn_yy[].mat
    end
    Symmetrize!(init_yy)
    Symmetrize!(dyn_xx)
    Symmetrize!(dyn_yy)
    dst.init_n = init_n
    dst.dyn_n = dyn_n
    dst.init_xx[] = PDMat(fill(init_n, 1, 1))
    dst.init_yy[] = PDMat(init_yy)
    dst.dyn_xx[] = PDMat(dyn_xx)
    dst.dyn_yy[] = PDMat(dyn_yy)
    return dst
end

# Combine the *observation-side* blocks (obs) of `sufs[idxs]` into `dst`. All
# regimes in an emission group share `obs_dim`, so this is well-defined. Feeds the
# `[C d D]` regression (`update_C_d!` reads `obs_xx`/`obs_xy`).
function _combine_obs_blocks!(
    dst::SufficientStatistics{T}, sufs::AbstractVector{SufficientStatistics{T}}, idxs
) where {T<:Real}
    orr = size(dst.obs_xy, 1)
    p = size(dst.obs_xy, 2)
    obs_n = zero(T)
    fill!(dst.obs_xy, zero(T))
    obs_xx = zeros(T, orr, orr)
    obs_yy = zeros(T, p, p)
    for i in idxs
        s = sufs[i]
        obs_n += s.obs_n
        dst.obs_xy .+= s.obs_xy
        obs_xx .+= s.obs_xx[].mat
        obs_yy .+= s.obs_yy[].mat
    end
    Symmetrize!(obs_xx)
    Symmetrize!(obs_yy)
    dst.obs_n = obs_n
    dst.obs_xx[] = PDMat(obs_xx)
    dst.obs_yy[] = PDMat(obs_yy)
    return dst
end

# Residual scatters for one regime (the part of update_P0!/update_Q!/update_R!
# before the divide / IW-MAP), summed across regimes per covariance group.
function _init_scatter(suf::SufficientStatistics{T}, x0::AbstractVector{T}) where {T}
    n = suf.init_n
    μ = vec(suf.init_xy)
    S = copy(suf.init_yy[].mat)
    BLAS.ger!(-one(T), μ, x0, S)
    BLAS.ger!(-one(T), x0, μ, S)
    BLAS.ger!(n, x0, x0, S)
    Symmetrize!(S)
    return S, n
end

# W = [A b B] (D × dyn_reg_dim); scatter of x_t - A x_{t-1} - b - B ux_{t-1}.
function _dyn_scatter(suf::SufficientStatistics{T}, W::AbstractMatrix{T}) where {T}
    S = copy(suf.dyn_yy[].mat)
    mul!(S, W, suf.dyn_xy, -one(T), one(T))
    mul!(S, transpose(suf.dyn_xy), transpose(W), -one(T), one(T))
    S .+= W * suf.dyn_xx[].mat * transpose(W)
    Symmetrize!(S)
    return S, suf.dyn_n
end

# V = [C d D] (p × obs_reg_dim); scatter of y_t - C x_t - d - D uy_t.
function _obs_scatter(suf::SufficientStatistics{T}, V::AbstractMatrix{T}) where {T}
    S = copy(suf.obs_yy[].mat)
    mul!(S, V, suf.obs_xy, -one(T), one(T))
    mul!(S, transpose(suf.obs_xy), transpose(V), -one(T), one(T))
    S .+= V * suf.obs_xx[].mat * transpose(V)
    Symmetrize!(S)
    return S, suf.obs_n
end

function _finalize_cov!(
    dest::AbstractMatrix{T}, S::AbstractMatrix{T}, n::T,
    prior::Union{Nothing,IWPrior{T}}, dim::Int,
) where {T}
    if prior === nothing
        copyto!(dest, S ./ n)
    else
        copyto!(dest, iw_map(Matrix{T}(prior.Ψ), prior.ν, S, n, dim))
    end
    return dest
end

# Assemble the stacked mean matrix [A b B] / [C d D] of a regime's plain LDS.
_stacked_dyn(lds) = _stack_mean(lds.state_model.A, lds.state_model.b, lds.state_model.B)
_stacked_obs(lds) = _stacked_obs(lds.obs_model)
_stacked_obs(om::GaussianObservationModel) = _stack_mean(om.C, om.d, om.D)
_stacked_obs(om::PoissonObservationModel) = hcat(om.C, om.d)
function _stack_mean(A::AbstractMatrix{T}, b::AbstractVector{T}, B::AbstractMatrix{T}) where {T}
    return size(B, 2) == 0 ? hcat(A, b) : hcat(A, b, B)
end

# --- shared M-step over resolved regimes -----------------------------------
# `regime_lds[r]` references the Varying storage, so `update_*!`/`copyto!` write
# back into the model. Mean blocks are updated first, then covariance blocks use
# the updated means.

# State-block M-step (x0, P0, [A b B], Q); shared by the Gaussian and Poisson
# drivers. `regime_lds[r]` references the model's storage, so `update_*!` and
# `copyto!` write results back into the `Varying`s.
function _indexed_state_mstep!(
    lds, gr::_IndexGroups, regime_lds, suf_r::AbstractVector{SufficientStatistics{T}},
    suf_comb::SufficientStatistics{T}, sws_r,
) where {T}
    sm = lds.state_model
    D = lds.latent_dim
    fit_bool = lds.fit_bool

    if fit_bool[1]                 # x0 (mean)
        for g in eachindex(gr.by_x0)
            rs = gr.by_x0[g]
            isempty(rs) && continue
            nn = sum(suf_r[r].init_n for r in rs)
            acc = zeros(T, D)
            for r in rs
                acc .+= vec(suf_r[r].init_xy)
            end
            at(sm.x0, g) .= acc ./ nn
        end
    end
    if fit_bool[3]                 # [A b B] (mean regression)
        for g in eachindex(gr.by_dyn)
            rs = gr.by_dyn[g]
            isempty(rs) && continue
            _combine_state_blocks!(suf_comb, suf_r, rs)
            update_A_b!(regime_lds[rs[1]], suf_comb, sws_r[rs[1]][1])
        end
    end
    if fit_bool[2]                 # P0 (covariance; uses updated x0)
        S0 = [_init_scatter(suf_r[r], at(sm.x0, gr.regimes[r][1])) for r in eachindex(suf_r)]
        for g in eachindex(gr.by_P0)
            rs = gr.by_P0[g]
            isempty(rs) && continue
            S = sum(S0[r][1] for r in rs)
            n = T(sum(S0[r][2] for r in rs))
            _finalize_cov!(at(sm.P0, g), S, n, sm.P0_prior, D)
        end
    end
    if fit_bool[4]                 # Q (covariance; uses updated [A b B])
        Sq = [_dyn_scatter(suf_r[r], _stacked_dyn(regime_lds[r])) for r in eachindex(suf_r)]
        for g in eachindex(gr.by_Q)
            rs = gr.by_Q[g]
            isempty(rs) && continue
            S = sum(Sq[r][1] for r in rs)
            n = T(sum(Sq[r][2] for r in rs))
            _add_mn_cov_contribution!(S, sm.AB_prior, regime_lds, rs, :dyn, gr)
            _finalize_cov!(at(sm.Q, g), S, n, sm.Q_prior, D)
        end
    end
    return nothing
end

# Gaussian emission M-step ([C d D] mean + R covariance). The emission combined
# buffer is allocated per group because emission groups may differ in `obs_dim`.
function _indexed_gaussian_obs_mstep!(
    lds, gr::_IndexGroups, regime_lds, suf_r::AbstractVector{SufficientStatistics{T}},
    sws_r,
) where {T}
    om = lds.obs_model
    fit_bool = lds.fit_bool

    if fit_bool[5]                 # [C d D] (mean regression)
        for g in eachindex(gr.by_obs)
            rs = gr.by_obs[g]
            isempty(rs) && continue
            combined = _initialize_td_sufficient_statistics(T, regime_lds[rs[1]], Int[1])
            _combine_obs_blocks!(combined, suf_r, rs)
            update_C_d!(regime_lds[rs[1]], combined, sws_r[rs[1]][1])
        end
    end
    if fit_bool[6]                 # R (covariance; uses updated [C d D])
        Sr = [_obs_scatter(suf_r[r], _stacked_obs(regime_lds[r])) for r in eachindex(suf_r)]
        for g in eachindex(gr.by_R)
            rs = gr.by_R[g]
            isempty(rs) && continue
            S = sum(Sr[r][1] for r in rs)
            n = T(sum(Sr[r][2] for r in rs))
            _add_mn_cov_contribution!(S, om.CD_prior, regime_lds, rs, :obs, gr)
            _finalize_cov!(at(om.R, g), S, n, om.R_prior, size(at(om.R, g), 1))
        end
    end
    return nothing
end

# Poisson emission M-step ([C d] via the non-conjugate LBFGS routine, per group).
# `weights` (per global trial, or `nothing`) carries the SLDS responsibilities
# γₖ; the LDS path passes `nothing`.
function _indexed_poisson_obs_mstep!(
    lds, gr::_IndexGroups, regime_lds, tfs_global::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}}, sws_r;
    weights::Union{Nothing,AbstractVector}=nothing,
) where {T}
    lds.fit_bool[5] || return nothing
    for g in eachindex(gr.by_obs)
        rs = gr.by_obs[g]
        isempty(rs) && continue
        trials = findall(==(g), gr.gobs)
        tfs_g = TrialFilterSmooth([tfs_global[i] for i in trials])
        y_g = [y[i] for i in trials]
        w_g = weights === nothing ? nothing : [weights[i] for i in trials]
        # `sws_r[rs[1]]` is a workspace pool sized for this group's obs_dim (all
        # regimes in an emission group share obs_dim), as the LBFGS routine needs.
        update_observation_model!(regime_lds[rs[1]], tfs_g, y_g, sws_r[rs[1]], w_g)
    end
    return nothing
end

# MN-prior contribution to a covariance's IW posterior scale, generalized to the
# grouped case: add `(W - M₀) Λ (W - M₀)'` once per distinct mean-group present in
# the covariance group's regimes (reduces to `update_Q!`/`update_R!` when there is
# a single group).
function _add_mn_cov_contribution!(S, prior::Nothing, regime_lds, rs, which, gr)
    return S
end
function _add_mn_cov_contribution!(S, prior::MNPrior, regime_lds, rs, which, gr)
    seen = Int[]
    for r in rs
        mg = which === :dyn ? gr.regimes[r][3] : gr.regimes[r][5]
        mg in seen && continue
        push!(seen, mg)
        W = which === :dyn ? _stacked_dyn(regime_lds[r]) : _stacked_obs(regime_lds[r])
        Wm = W .- prior.M₀
        S .+= Wm * prior.Λ * transpose(Wm)
    end
    return S
end

# --- per-regime working buffers --------------------------------------------

# Build the plain per-regime LDSs and their per-regime E-step buffers. `sws_r[r][1]`
# is sized with `ntrials = length(regime_trials[r])` so the equal-length batched
# cov-cache fast path fires within each regime; different-`obs_dim` regimes are
# smoothed in separate batches.
function _indexed_regime_setup(
    lds, gr::_IndexGroups, y::AbstractVector{<:AbstractMatrix{T}},
    ux_seq::AbstractVector{<:AbstractMatrix{T}}, uy_seq::AbstractVector{<:AbstractMatrix{T}},
) where {T}
    D = lds.latent_dim
    ux_dim = lds.state_input_dim
    uy_dim = lds.obs_input_dim
    Rn = length(gr.regimes)
    tsteps_all = [size(yt, 2) for yt in y]

    regime_lds = [_resolve_lds(lds, gr.regimes[r]) for r in 1:Rn]
    tfs_global = initialize_FilterSmooth(regime_lds[1], tsteps_all; cov_alias=false)

    pool_size = Threads.maxthreadid()
    y_r = Vector{Vector{Matrix{T}}}(undef, Rn)
    ux_r = Vector{Vector{Matrix{T}}}(undef, Rn)
    uy_r = Vector{Vector{Matrix{T}}}(undef, Rn)
    tfs_r = Vector{TrialFilterSmooth{T}}(undef, Rn)
    sws_r = Vector{Vector{SmoothWorkspace{T}}}(undef, Rn)
    suf_r = Vector{SufficientStatistics{T}}(undef, Rn)
    for r in 1:Rn
        trials = gr.regime_trials[r]
        ts = [tsteps_all[i] for i in trials]
        Tmax = maximum(ts)
        p_r = regime_lds[r].obs_dim
        y_r[r] = [Matrix{T}(y[i]) for i in trials]
        ux_r[r] = [Matrix{T}(ux_seq[i]) for i in trials]
        uy_r[r] = [Matrix{T}(uy_seq[i]) for i in trials]
        tfs_r[r] = TrialFilterSmooth([tfs_global[i] for i in trials])
        sws_r[r] = Vector{SmoothWorkspace{T}}(undef, pool_size)
        sws_r[r][1] = SmoothWorkspace(
            T, D, p_r, Tmax; u_dim=ux_dim, d_dim=uy_dim, ntrials=length(trials)
        )
        for i in 2:pool_size
            sws_r[r][i] = SmoothWorkspace(T, D, p_r, Tmax; u_dim=ux_dim, d_dim=uy_dim)
        end
        suf_r[r] = _initialize_td_sufficient_statistics(T, regime_lds[r], ts)
        _td_init_const_blocks!(sws_r[r][1], regime_lds[r], ts, y_r[r], ux_r[r], uy_r[r])
    end
    return regime_lds, tfs_global, y_r, ux_r, uy_r, tfs_r, sws_r, suf_r
end

# ELBO prior term summed over parameter groups (IW priors per covariance group;
# MN priors per unique mean-group with a representative covariance from that
# group's regimes — reduces to the plain terms when nothing is grouped).
function _indexed_prior_term(
    lds, gr::_IndexGroups, regime_lds, ::Type{T}; has_R::Bool
) where {T}
    sm = lds.state_model
    om = lds.obs_model
    prior = zero(T)
    if sm.P0_prior !== nothing
        for g in 1:nvals(sm.P0)
            prior += iw_logprior_term(at(sm.P0, g), sm.P0_prior)
        end
    end
    if sm.Q_prior !== nothing
        for g in 1:nvals(sm.Q)
            prior += iw_logprior_term(at(sm.Q, g), sm.Q_prior)
        end
    end
    if has_R && om.R_prior !== nothing
        for g in 1:nvals(om.R)
            prior += iw_logprior_term(at(om.R, g), om.R_prior)
        end
    end
    if sm.AB_prior !== nothing
        for rs in gr.by_dyn
            isempty(rs) && continue
            r = rs[1]
            prior += mn_logprior_term(
                _stacked_dyn(regime_lds[r]), regime_lds[r].state_model.Q, sm.AB_prior
            )
        end
    end
    if has_R && om.CD_prior !== nothing
        for rs in gr.by_obs
            isempty(rs) && continue
            r = rs[1]
            prior += mn_logprior_term(
                _stacked_obs(regime_lds[r]), regime_lds[r].obs_model.R, om.CD_prior
            )
        end
    end
    return prior
end

_total_entropy(tfs::TrialFilterSmooth{T}) where {T} = sum(fs.entropy for fs in tfs.FilterSmooths)

# --- Gaussian LDS driver ---------------------------------------------------

function _fit_indexed_gaussian!(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractVector{<:AbstractMatrix{T}};
    labels::AbstractDict, latent_inputs, obs_inputs, max_iter::Int, tol::Float64,
    progress::Bool,
) where {T,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    ntrials = length(y)
    tsteps_all = [size(yt, 2) for yt in y]
    ux_seq = _normalize_multitrial_control(
        latent_inputs, lds.state_input_dim, tsteps_all, T, "latent_inputs"
    )
    uy_seq = _normalize_multitrial_obs_control(
        obs_inputs, lds.obs_input_dim, tsteps_all, T, lds.obs_model
    )
    gr = _index_groups(lds.state_model, lds.obs_model, labels, ntrials; has_R=true)
    regime_lds, tfs_global, y_r, ux_r, uy_r, tfs_r, sws_r, suf_r = _indexed_regime_setup(
        lds, gr, y, ux_seq, uy_seq
    )
    Rn = length(gr.regimes)
    suf_state = _initialize_td_sufficient_statistics(T, regime_lds[1], tsteps_all)

    elbos = Vector{T}()
    sizehint!(elbos, max_iter)
    prev = -T(Inf)
    prog = progress ? Progress(max_iter; desc="Fitting indexed LDS via EM...", barlen=50) : nothing
    for _ in 1:max_iter
        for r in 1:Rn
            smooth!(regime_lds[r], tfs_r[r], y_r[r], sws_r[r], ux_r[r], uy_r[r])
            _aggregate_td_suff_stats!(
                suf_r[r], tfs_r[r], regime_lds[r], ux_r[r], uy_r[r], y_r[r], sws_r[r][1]
            )
        end
        elbo = _total_entropy(tfs_global)
        for r in 1:Rn
            compute_smooth_constants!(sws_r[r][1], regime_lds[r])
            elbo += Q_state!(sws_r[r][1], regime_lds[r], suf_r[r])
            elbo += Q_obs!(sws_r[r][1], regime_lds[r], suf_r[r])
        end
        elbo += _indexed_prior_term(lds, gr, regime_lds, T; has_R=true)
        push!(elbos, elbo)

        _indexed_state_mstep!(lds, gr, regime_lds, suf_r, suf_state, sws_r)
        _indexed_gaussian_obs_mstep!(lds, gr, regime_lds, suf_r, sws_r)

        prog !== nothing && next!(prog)
        if abs(elbo - prev) < tol
            prog !== nothing && finish!(prog)
            return elbos
        end
        prev = elbo
    end
    prog !== nothing && finish!(prog)
    return elbos
end

# --- Poisson LDS driver ----------------------------------------------------

function _fit_indexed_poisson!(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractVector{<:AbstractMatrix{T}};
    labels::AbstractDict, max_iter::Int, tol::Float64, progress::Bool,
    newton_max_iter::Int, newton_tol::Float64,
) where {T,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    ntrials = length(y)
    tsteps_all = [size(yt, 2) for yt in y]
    ux_seq = [zeros(T, 0, ti) for ti in tsteps_all]
    uy_seq = [zeros(T, 0, ti) for ti in tsteps_all]
    gr = _index_groups(lds.state_model, lds.obs_model, labels, ntrials; has_R=false)
    regime_lds, tfs_global, y_r, ux_r, uy_r, tfs_r, sws_r, suf_r = _indexed_regime_setup(
        lds, gr, y, ux_seq, uy_seq
    )
    Rn = length(gr.regimes)
    suf_state = _initialize_td_sufficient_statistics(T, regime_lds[1], tsteps_all)

    elbos = Vector{T}()
    sizehint!(elbos, max_iter)
    prev = -T(Inf)
    prog = progress ? Progress(max_iter; desc="Fitting indexed Poisson LDS...", barlen=50) : nothing
    for _ in 1:max_iter
        for r in 1:Rn
            smooth!(
                regime_lds[r], tfs_r[r], y_r[r], sws_r[r];
                max_iter=newton_max_iter, tol=T(newton_tol),
            )
            _aggregate_td_suff_stats!(
                suf_r[r], tfs_r[r], regime_lds[r], ux_r[r], uy_r[r], y_r[r], sws_r[r][1]
            )
        end
        elbo = _total_entropy(tfs_global)
        for r in 1:Rn
            compute_smooth_constants!(sws_r[r][1], regime_lds[r])
            elbo += Q_state!(sws_r[r][1], regime_lds[r], suf_r[r])
            for (j, fs) in enumerate(tfs_r[r].FilterSmooths)
                elbo += Q_obs!(sws_r[r][1], regime_lds[r], fs.x_smooth, fs.p_smooth, y_r[r][j])
            end
        end
        elbo += _indexed_prior_term(lds, gr, regime_lds, T; has_R=false)
        # MN prior on Poisson [C d], per emission group.
        if lds.obs_model.CD_prior !== nothing
            prior = lds.obs_model.CD_prior
            for rs in gr.by_obs
                isempty(rs) && continue
                W = _stacked_obs(regime_lds[rs[1]])
                Wm = W .- prior.M₀
                elbo -= T(0.5) * sum(Wm .* (Wm * prior.Λ))
            end
        end
        push!(elbos, elbo)

        _indexed_state_mstep!(lds, gr, regime_lds, suf_r, suf_state, sws_r)
        _indexed_poisson_obs_mstep!(lds, gr, regime_lds, tfs_global, y, sws_r)

        prog !== nothing && next!(prog)
        if abs(elbo - prev) < tol
            prog !== nothing && finish!(prog)
            return elbos
        end
        prev = elbo
    end
    prog !== nothing && finish!(prog)
    return elbos
end

# --- group-aware smooth / loglikelihood / rand (LDS) -----------------------

function _smooth_indexed(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractVector{<:AbstractMatrix{T}};
    labels::AbstractDict, max_iter::Int=20, tol::T=T(1e-6),
) where {T,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    ntrials = length(y)
    has_R = !_is_poisson_like(lds.obs_model)
    gr = _index_groups(lds.state_model, lds.obs_model, labels, ntrials; has_R=has_R)
    regime_lds = [_resolve_lds(lds, gr.regimes[r]) for r in eachindex(gr.regimes)]
    xs = Vector{Matrix{T}}(undef, ntrials)
    Ps = Vector{Array{T,3}}(undef, ntrials)
    for r in eachindex(gr.regimes), i in gr.regime_trials[r]
        rl = regime_lds[r]
        if _is_poisson_like(lds.obs_model)
            fs = initialize_FilterSmooth(rl, size(y[i], 2))::FilterSmooth{T}
            sws = SmoothWorkspace(T, lds.latent_dim, rl.obs_dim, size(y[i], 2))
            smooth!(rl, fs, y[i], sws; max_iter=max_iter, tol=tol)
            xs[i] = copy(fs.x_smooth)
            Ps[i] = copy(fs.p_smooth)
        else
            xi, Pi = smooth(rl, y[i])
            xs[i] = xi
            Ps[i] = Pi
        end
    end
    return xs, Ps
end

function _loglikelihood_indexed(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractVector{<:AbstractMatrix{T}};
    labels::AbstractDict,
) where {T,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    ntrials = length(y)
    gr = _index_groups(lds.state_model, lds.obs_model, labels, ntrials; has_R=true)
    regime_lds = [_resolve_lds(lds, gr.regimes[r]) for r in eachindex(gr.regimes)]
    total = zero(T)
    for r in eachindex(gr.regimes), i in gr.regime_trials[r]
        total += loglikelihood(regime_lds[r], y[i])
    end
    return total
end

function _rand_indexed(
    rng::AbstractRNG, lds::LinearDynamicalSystem{T,S,O},
    tsteps_per_trial::AbstractVector{<:Integer}; labels::AbstractDict,
) where {T,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    ntrials = length(tsteps_per_trial)
    has_R = !_is_poisson_like(lds.obs_model)
    gr = _index_groups(lds.state_model, lds.obs_model, labels, ntrials; has_R=has_R)
    D = lds.latent_dim
    regime_lds = [_resolve_lds(lds, gr.regimes[r]) for r in eachindex(gr.regimes)]
    state_params = [_extract_state_params(rl.state_model) for rl in regime_lds]
    obs_params = [_extract_obs_params(rl.obs_model) for rl in regime_lds]
    x = Vector{Matrix{T}}(undef, ntrials)
    y = Vector{Matrix{T}}(undef, ntrials)
    for r in eachindex(gr.regimes), i in gr.regime_trials[r]
        Ti = Int(tsteps_per_trial[i])
        p_r = regime_lds[r].obs_dim
        x[i] = Matrix{T}(undef, D, Ti)
        y[i] = Matrix{T}(undef, p_r, Ti)
        _sample_trial!(
            rng, x[i], y[i], state_params[r], obs_params[r], regime_lds[r].obs_model,
            zeros(T, 0, Ti), zeros(T, 0, Ti),
        )
    end
    return x, y
end

# =============================================================================
# Group-aware SLDS
#
# Each discrete-state component holds `Indexed` parameters keyed on the same
# labels (validated). Trials are partitioned into regimes; per regime a plain
# `SLDS` (K resolved components sharing the discrete layer `A`/`πₖ`) drives the
# per-trial log-likelihoods and continuous smoothing. The M-step reuses the LDS
# indexed M-step per discrete state, fed γ-weighted per-regime sufficient
# statistics.
# =============================================================================

# Plain per-regime SLDS: K resolved components sharing `A`/`πₖ` by reference.
function _resolve_slds(slds::SLDS, regime::NTuple{6,Int})
    ldss = [_resolve_lds(slds.LDSs[k], regime) for k in eachindex(slds.LDSs)]
    return SLDS(; A=slds.A, πₖ=slds.πₖ, LDSs=ldss)
end

# All components must share label/group_ids per block so the regime grouping is
# well-defined across discrete states.
function _check_slds_shared_grouping(slds::SLDS)
    K = length(slds.LDSs)
    K <= 1 && return nothing
    ref = slds.LDSs[1]
    for k in 2:K
        c = slds.LDSs[k]
        for (p1, pk, name) in (
            (ref.state_model.A, c.state_model.A, "A"),
            (ref.state_model.Q, c.state_model.Q, "Q"),
            (ref.state_model.x0, c.state_model.x0, "x0"),
            (ref.state_model.P0, c.state_model.P0, "P0"),
            (ref.obs_model.C, c.obs_model.C, "C"),
        )
            (param_label(p1) == param_label(pk) && param_group_ids(p1) == param_group_ids(pk)) ||
                throw(
                    ArgumentError(
                        "SLDS components must share the same trial label/groups for " *
                        "parameter $(name); component $k differs from component 1",
                    ),
                )
        end
    end
    return nothing
end

function _fit_indexed_slds!(
    slds::SLDS{T,S,O}, y::AbstractVector{<:AbstractMatrix{T}};
    labels::AbstractDict, max_iter::Int=50, progress::Bool=true,
) where {T,S,O}
    _check_slds_shared_grouping(slds)
    K = length(slds.LDSs)
    latent_dim = slds.LDSs[1].latent_dim
    poisson = _is_poisson_like(slds.LDSs[1].obs_model)
    has_R = !poisson
    ntrials = length(y)
    tsteps_all = [size(yt, 2) for yt in y]
    seq_ends = cumsum(tsteps_all)
    total_T = last(seq_ends)
    T_max = maximum(tsteps_all)

    gr = _index_groups(
        slds.LDSs[1].state_model, slds.LDSs[1].obs_model, labels, ntrials; has_R=has_R
    )
    Rn = length(gr.regimes)
    ux_seq = [zeros(T, 0, ti) for ti in tsteps_all]
    uy_seq = [zeros(T, 0, ti) for ti in tsteps_all]

    # Per-regime plain SLDS + workspaces + E-step buffers.
    regime_slds = [_resolve_slds(slds, gr.regimes[r]) for r in 1:Rn]
    regime_ws = [SLDSSmoothWorkspace(T, regime_slds[r], T_max) for r in 1:Rn]
    tfs = initialize_FilterSmooth(slds.LDSs[1], tsteps_all)::TrialFilterSmooth{T}

    pool_size = Threads.maxthreadid()
    y_r = Vector{Vector{Matrix{T}}}(undef, Rn)
    ux_r = Vector{Vector{Matrix{T}}}(undef, Rn)
    uy_r = Vector{Vector{Matrix{T}}}(undef, Rn)
    tfs_r = Vector{TrialFilterSmooth{T}}(undef, Rn)
    sws_r = Vector{Vector{SmoothWorkspace{T}}}(undef, Rn)
    suf_rk = Vector{SufficientStatistics{T}}(undef, Rn)
    for r in 1:Rn
        trials = gr.regime_trials[r]
        ts = [tsteps_all[i] for i in trials]
        p_r = regime_slds[r].LDSs[1].obs_dim
        y_r[r] = [y[i] for i in trials]
        ux_r[r] = [ux_seq[i] for i in trials]
        uy_r[r] = [uy_seq[i] for i in trials]
        tfs_r[r] = TrialFilterSmooth([tfs[i] for i in trials])
        sws_r[r] = [SmoothWorkspace(T, latent_dim, p_r, T_max) for _ in 1:pool_size]
        suf_rk[r] = _initialize_td_sufficient_statistics(T, regime_slds[r].LDSs[1], ts)
    end

    dl = SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(T, K, total_T))
    fb_storage = _make_slds_fb_storage(dl, seq_ends)
    obs_seq = collect(1:total_T)
    ctrl_seq = fill(nothing, total_T)
    x_samples = [Matrix{T}(undef, latent_dim, Ti) for Ti in tsteps_all]
    randn_buf = Vector{T}(undef, latent_dim)
    suf_state = _initialize_td_sufficient_statistics(T, regime_slds[1].LDSs[1], tsteps_all)

    elbos = Vector{T}(undef, max_iter)
    prog = progress ? Progress(max_iter; desc="Fitting indexed SLDS via EM...", barlen=50) : nothing

    # Warm-start: smooth each trial with uniform discrete weights.
    for trial in 1:ntrials
        r = _regime_of(gr, trial)
        w_uniform = fill(one(T) / K, K, tsteps_all[trial])
        smooth!(regime_slds[r], tfs[trial], y[trial], w_uniform; ws=regime_ws[r])
    end

    for iter in 1:max_iter
        sample_posterior!(x_samples, Random.default_rng(), tfs, randn_buf)
        elbos[iter] = _slds_indexed_estep!(
            slds, gr, regime_slds, regime_ws, tfs, fb_storage, dl, y, x_samples;
            obs_seq=obs_seq, ctrl_seq=ctrl_seq, seq_ends=seq_ends,
        )
        _slds_indexed_mstep!(
            slds, gr, regime_slds, tfs, tfs_r, fb_storage, dl, y, y_r, ux_r, uy_r,
            sws_r, suf_rk, suf_state; seq_ends=seq_ends, obs_seq=obs_seq,
            poisson=poisson, has_R=has_R,
        )
        for r in 1:Rn
            refresh_slds_constants!(regime_ws[r], regime_slds[r])
        end
        prog !== nothing && next!(prog; showvalues=[(:iteration, iter), (:ELBO, elbos[iter])])
    end
    prog !== nothing && finish!(prog)
    return elbos
end

# regime index of a global trial
function _regime_of(gr::_IndexGroups, trial::Int)
    key = (gr.gx0[trial], gr.gP0[trial], gr.gdyn[trial], gr.gQ[trial], gr.gobs[trial], gr.gR[trial])
    for r in eachindex(gr.regimes)
        gr.regimes[r] == key && return r
    end
    return 0
end

function _slds_indexed_estep!(
    slds::SLDS{T}, gr::_IndexGroups, regime_slds, regime_ws, tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage, dl::SLDSDiscreteLayer{T},
    y::AbstractVector{<:AbstractMatrix{T}}, x_samples::AbstractVector{<:AbstractMatrix{T}};
    obs_seq, ctrl_seq, seq_ends,
) where {T}
    ntrials = length(y)
    K = length(slds.LDSs)
    for trial in 1:ntrials
        r = _regime_of(gr, trial)
        ws_r = regime_ws[r]
        rslds = regime_slds[r]
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        for k in 1:K
            joint_loglikelihood!(
                view(dl.logL, k, t1:t2), ws_r, ws_r.consts[k], rslds.LDSs[k],
                x_samples[trial], y[trial],
            )
        end
    end
    HMMs.forward_backward!(
        fb_storage, dl, obs_seq, ctrl_seq; seq_ends=seq_ends, transition_marginals=true
    )
    total = zero(T)
    for trial in 1:ntrials
        r = _regime_of(gr, trial)
        ws_r = regime_ws[r]
        rslds = regime_slds[r]
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        Tsteps = t2 - t1 + 1
        w = view(fb_storage.γ, :, t1:t2)
        smooth!(rslds, tfs[trial], y[trial], w; ws=ws_r)
        xt = tfs[trial].x_smooth
        for k in 1:K
            ll = view(ws_r.ll_tmp, 1:Tsteps)
            joint_loglikelihood!(ll, ws_r, ws_r.consts[k], rslds.LDSs[k], xt, y[trial])
            for t in 1:Tsteps
                total += w[k, t] * ll[t]
            end
        end
        for k in 1:K
            total += w[k, 1] * log(slds.πₖ[k] + T(1e-12))
        end
        for t in t1:(t2 - 1)
            ξt = fb_storage.ξ[t]
            for i in 1:K, j in 1:K
                total += ξt[i, j] * log(slds.A[i, j] + T(1e-12))
            end
        end
        total -= tfs[trial].entropy
        for k in 1:K, t in 1:Tsteps
            wkt = w[k, t]
            wkt > 0 && (total += wkt * log(wkt + T(1e-12)))
        end
    end
    return total
end

function _slds_indexed_mstep!(
    slds::SLDS{T}, gr::_IndexGroups, regime_slds, tfs::TrialFilterSmooth{T},
    tfs_r, fb_storage::HMMs.ForwardBackwardStorage, dl::SLDSDiscreteLayer{T},
    y::AbstractVector{<:AbstractMatrix{T}}, y_r, ux_r, uy_r, sws_r, suf_rk, suf_state;
    seq_ends, obs_seq, poisson::Bool, has_R::Bool,
) where {T}
    K = length(slds.LDSs)
    Rn = length(gr.regimes)
    StatsAPI.fit!(dl, fb_storage, obs_seq; seq_ends=seq_ends)

    for k in 1:K
        # γₖ-weighted per-regime sufficient statistics for component k.
        for r in 1:Rn
            trials = gr.regime_trials[r]
            weights = Vector{AbstractVector{T}}(undef, length(trials))
            for (j, i) in enumerate(trials)
                t1, t2 = HMMs.seq_limits(seq_ends, i)
                weights[j] = view(fb_storage.γ, k, t1:t2)
            end
            _aggregate_td_suff_stats_weighted!(
                suf_rk[r], tfs_r[r], regime_slds[r].LDSs[k], ux_r[r], uy_r[r], y_r[r],
                weights, sws_r[r][1],
            )
        end
        comp = slds.LDSs[k]
        regime_lds_k = [regime_slds[r].LDSs[k] for r in 1:Rn]
        _indexed_state_mstep!(comp, gr, regime_lds_k, suf_rk, suf_state, sws_r)
        if poisson
            wk = [view(fb_storage.γ, k, HMMs.seq_limits(seq_ends, i)[1]:HMMs.seq_limits(seq_ends, i)[2]) for i in 1:length(y)]
            _indexed_poisson_obs_mstep!(comp, gr, regime_lds_k, tfs, y, sws_r; weights=wk)
        else
            _indexed_gaussian_obs_mstep!(comp, gr, regime_lds_k, suf_rk, sws_r)
        end
    end
    return nothing
end

# group-aware SLDS rand
function _rand_indexed_slds(
    rng::AbstractRNG, slds::SLDS{T}, tsteps_per_trial::AbstractVector{<:Integer};
    labels::AbstractDict,
) where {T}
    _check_slds_shared_grouping(slds)
    ntrials = length(tsteps_per_trial)
    has_R = !_is_poisson_like(slds.LDSs[1].obs_model)
    gr = _index_groups(
        slds.LDSs[1].state_model, slds.LDSs[1].obs_model, labels, ntrials; has_R=has_R
    )
    latent_dim = slds.LDSs[1].latent_dim
    regime_slds = [_resolve_slds(slds, gr.regimes[r]) for r in eachindex(gr.regimes)]
    state_params = [[_extract_state_params(lds.state_model) for lds in rs.LDSs] for rs in regime_slds]
    obs_params = [[_extract_obs_params(lds.obs_model) for lds in rs.LDSs] for rs in regime_slds]
    z = Vector{Vector{Int}}(undef, ntrials)
    x = Vector{Matrix{T}}(undef, ntrials)
    y = Vector{Matrix{T}}(undef, ntrials)
    for trial in 1:ntrials
        r = _regime_of(gr, trial)
        Ti = Int(tsteps_per_trial[trial])
        p_r = regime_slds[r].LDSs[1].obs_dim
        z[trial] = Vector{Int}(undef, Ti)
        x[trial] = Matrix{T}(undef, latent_dim, Ti)
        y[trial] = Matrix{T}(undef, p_r, Ti)
        _sample_slds_trial!(
            rng, z[trial], x[trial], y[trial], slds.A, slds.πₖ, state_params[r],
            obs_params[r], regime_slds[r].LDSs[1].obs_model,
        )
    end
    return z, x, y
end
