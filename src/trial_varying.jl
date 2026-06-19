# =============================================================================
# Trial-varying parameter models (generalized stitching).
#
# Each of the six M-step estimation blocks — x0, P0, [A b], Q, [C d], R — may be
# fit per trial group, and different blocks may key on different labels (supplied
# at fit/rand/smooth time as `labels::Dict{Symbol,Vector}`). The trial-invariant
# `GaussianStateModel` / `GaussianObservationModel` hot paths are untouched; this
# is a separate, deliberately simpler-but-general code path.
#
# Implementation: trials are partitioned into *regimes* — unique combinations of
# the six block-group indices. Within a regime all parameters are fixed, so a
# regime is a plain LDS (referencing the grouped value arrays, so M-step updates
# propagate). The E-step smooths per regime and aggregates per-regime sufficient
# statistics. The M-step updates each block per group:
#   * mean blocks (x0, [A b], [C d]) — sum the raw sufficient statistics of the
#     regimes in each group and run the standard regression;
#   * covariance blocks (P0, Q, R) — sum *per-regime residual scatters* (each
#     computed with that regime's updated mean), so a covariance block can be
#     grouped independently of its mean block.
# =============================================================================

"""
    _resolve_group_indices(gp::GroupedParam, labels, ntrials) -> Vector{Int}

Per-trial group index for a grouped parameter. Invariant parameters map every
trial to group 1; otherwise each trial's label value (from `labels[gp.label]`) is
matched against `gp.group_ids`.
"""
function _resolve_group_indices(gp::GroupedParam, labels, ntrials::Int)
    is_invariant(gp) && return ones(Int, ntrials)
    haskey(labels, gp.label) ||
        throw(ArgumentError("labels is missing key :$(gp.label)"))
    lab = labels[gp.label]
    length(lab) == ntrials || throw(
        DimensionMismatchError("labels[:$(gp.label)] length", ntrials, length(lab))
    )
    gidx = Vector{Int}(undef, ntrials)
    for t in 1:ntrials
        j = findfirst(==(lab[t]), gp.group_ids)
        j === nothing && throw(
            ArgumentError(
                "label value $(repr(lab[t])) for :$(gp.label) (trial $t) not in " *
                "group_ids $(gp.group_ids)",
            ),
        )
        gidx[t] = j
    end
    return gidx
end

# Resolved grouping for a whole trial-varying model.
struct _TVGroups
    gx0::Vector{Int}
    gP0::Vector{Int}
    gdyn::Vector{Int}
    gQ::Vector{Int}
    gobs::Vector{Int}
    gR::Vector{Int}
    regimes::Vector{NTuple{6,Int}}
    regime_trials::Vector{Vector{Int}}
    by_x0::Vector{Vector{Int}}
    by_P0::Vector{Vector{Int}}
    by_dyn::Vector{Vector{Int}}
    by_Q::Vector{Vector{Int}}
    by_obs::Vector{Vector{Int}}
    by_R::Vector{Vector{Int}}
end

_regimes_by(regimes, comp, ng) =
    [findall(r -> regimes[r][comp] == g, eachindex(regimes)) for g in 1:ng]

function _tv_groups(sm, om, labels, ntrials::Int; has_R::Bool)
    gx0 = _resolve_group_indices(sm.x0, labels, ntrials)
    gP0 = _resolve_group_indices(sm.P0, labels, ntrials)
    gdyn = _resolve_group_indices(sm.A, labels, ntrials)
    gQ = _resolve_group_indices(sm.Q, labels, ntrials)
    gobs = _resolve_group_indices(om.C, labels, ntrials)
    gR = has_R ? _resolve_group_indices(om.R, labels, ntrials) : ones(Int, ntrials)

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

    ngR = has_R ? ngroups(om.R) : 1
    return _TVGroups(
        gx0, gP0, gdyn, gQ, gobs, gR, regimes, regime_trials,
        _regimes_by(regimes, 1, ngroups(sm.x0)),
        _regimes_by(regimes, 2, ngroups(sm.P0)),
        _regimes_by(regimes, 3, ngroups(sm.A)),
        _regimes_by(regimes, 4, ngroups(sm.Q)),
        _regimes_by(regimes, 5, ngroups(om.C)),
        _regimes_by(regimes, 6, ngR),
    )
end

# Plain per-regime LDS that references the grouped value arrays (so in-place
# M-step updates propagate to every regime that shares a block group).
function _regime_gaussian_lds(sm, om, regime, D, p, fit_bool, ::Type{T}) where {T}
    gx0, gP0, gdyn, gQ, gobs, gR = regime
    state = GaussianStateModel(;
        A=sm.A.values[gdyn], Q=sm.Q.values[gQ], b=sm.b.values[gdyn],
        x0=sm.x0.values[gx0], P0=sm.P0.values[gP0],
    )
    obs = GaussianObservationModel(;
        C=om.C.values[gobs], R=om.R.values[gR], d=om.d.values[gobs]
    )
    return LinearDynamicalSystem(;
        state_model=state, obs_model=obs, latent_dim=D, obs_dim=p, fit_bool=fit_bool
    )
end

function _regime_poisson_lds(sm, om, regime, D, p, fit_bool, ::Type{T}) where {T}
    gx0, gP0, gdyn, gQ, gobs, _ = regime
    state = GaussianStateModel(;
        A=sm.A.values[gdyn], Q=sm.Q.values[gQ], b=sm.b.values[gdyn],
        x0=sm.x0.values[gx0], P0=sm.P0.values[gP0],
    )
    obs = PoissonObservationModel(; C=om.C.values[gobs], d=om.d.values[gobs])
    return LinearDynamicalSystem(;
        state_model=state, obs_model=obs, latent_dim=D, obs_dim=p, fit_bool=fit_bool
    )
end

"""
    _combine_all_suff!(dst, sufs, idxs)

Sum every block (init / dyn / obs) of the per-regime sufficient statistics
`sufs[idxs]` into `dst`. Sufficient statistics are additive, so this reconstructs
the totals over the (disjoint) regimes in a group.
"""
function _combine_all_suff!(
    dst::SufficientStatistics{T}, sufs::AbstractVector{SufficientStatistics{T}}, idxs
) where {T<:Real}
    D = size(dst.init_xy, 2)
    dr = size(dst.dyn_xy, 1)
    orr = size(dst.obs_xy, 1)
    p = size(dst.obs_xy, 2)

    init_n = dyn_n = obs_n = zero(T)
    fill!(dst.init_xy, zero(T))
    fill!(dst.dyn_xy, zero(T))
    fill!(dst.obs_xy, zero(T))
    init_yy = zeros(T, D, D)
    dyn_xx = zeros(T, dr, dr)
    dyn_yy = zeros(T, D, D)
    obs_xx = zeros(T, orr, orr)
    obs_yy = zeros(T, p, p)

    for i in idxs
        s = sufs[i]
        init_n += s.init_n
        dyn_n += s.dyn_n
        obs_n += s.obs_n
        dst.init_xy .+= s.init_xy
        dst.dyn_xy .+= s.dyn_xy
        dst.obs_xy .+= s.obs_xy
        init_yy .+= s.init_yy[].mat
        dyn_xx .+= s.dyn_xx[].mat
        dyn_yy .+= s.dyn_yy[].mat
        obs_xx .+= s.obs_xx[].mat
        obs_yy .+= s.obs_yy[].mat
    end

    Symmetrize!(init_yy)
    Symmetrize!(dyn_xx)
    Symmetrize!(dyn_yy)
    Symmetrize!(obs_xx)
    Symmetrize!(obs_yy)

    dst.init_n = init_n
    dst.dyn_n = dyn_n
    dst.obs_n = obs_n
    dst.init_xx[] = PDMat(fill(init_n, 1, 1))
    dst.init_yy[] = PDMat(init_yy)
    dst.dyn_xx[] = PDMat(dyn_xx)
    dst.dyn_yy[] = PDMat(dyn_yy)
    dst.obs_xx[] = PDMat(obs_xx)
    dst.obs_yy[] = PDMat(obs_yy)
    return dst
end

# Per-regime residual scatters (the part of update_P0!/update_Q!/update_R! before
# the divide / IW-MAP), so they can be summed across regimes per covariance group.
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

function _dyn_scatter(suf::SufficientStatistics{T}, W::AbstractMatrix{T}) where {T}
    # S = dyn_yy - W dyn_xy - dyn_xy' W' + W dyn_xx W'   (W = [A b])
    S = copy(suf.dyn_yy[].mat)
    mul!(S, W, suf.dyn_xy, -one(T), one(T))
    mul!(S, transpose(suf.dyn_xy), transpose(W), -one(T), one(T))
    S .+= W * suf.dyn_xx[].mat * transpose(W)
    Symmetrize!(S)
    return S, suf.dyn_n
end

function _obs_scatter(suf::SufficientStatistics{T}, V::AbstractMatrix{T}) where {T}
    # S = obs_yy - V obs_xy - obs_xy' V' + V obs_xx V'   (V = [C d])
    S = copy(suf.obs_yy[].mat)
    mul!(S, V, suf.obs_xy, -one(T), one(T))
    mul!(S, transpose(suf.obs_xy), transpose(V), -one(T), one(T))
    S .+= V * suf.obs_xx[].mat * transpose(V)
    Symmetrize!(S)
    return S, suf.obs_n
end

function _finalize_cov(
    S::AbstractMatrix{T}, n::T, prior::Union{Nothing,IWPrior{T}}, dim::Int
) where {T}
    prior === nothing && return S ./ n
    return iw_map(Matrix{T}(prior.Ψ), prior.ν, S, n, dim)
end

# Shared state-side M-step (x0, P0, [A b], Q) for both Gaussian and Poisson.
function _tv_update_state!(
    sm, gr::_TVGroups, suf_r::AbstractVector{SufficientStatistics{T}},
    suf_comb::SufficientStatistics{T}, fit_bool::AbstractVector{Bool}, D::Int,
) where {T}
    # --- mean blocks: x0 and [A b] ---
    if fit_bool[1]
        for g in eachindex(gr.by_x0)
            rs = gr.by_x0[g]
            isempty(rs) && continue
            nn = sum(suf_r[r].init_n for r in rs)
            acc = zeros(T, D)
            for r in rs
                acc .+= vec(suf_r[r].init_xy)
            end
            sm.x0.values[g] .= acc ./ nn
        end
    end
    if fit_bool[3]
        for g in eachindex(gr.by_dyn)
            rs = gr.by_dyn[g]
            isempty(rs) && continue
            _combine_all_suff!(suf_comb, suf_r, rs)
            W = transpose(suf_comb.dyn_xx[].chol \ suf_comb.dyn_xy)  # D × (D+1)
            copyto!(sm.A.values[g], view(W, :, 1:D))
            copyto!(sm.b.values[g], view(W, :, D + 1))
        end
    end

    # --- covariance blocks: P0 and Q (after means; per-regime scatters) ---
    if fit_bool[2]
        S0 = [_init_scatter(suf_r[r], sm.x0.values[gr.regimes[r][1]]) for r in eachindex(suf_r)]
        for g in eachindex(gr.by_P0)
            rs = gr.by_P0[g]
            isempty(rs) && continue
            S = sum(S0[r][1] for r in rs)
            n = sum(S0[r][2] for r in rs)
            copyto!(sm.P0.values[g], _finalize_cov(S, T(n), sm.P0_prior, D))
        end
    end
    if fit_bool[4]
        Sq = [
            _dyn_scatter(
                suf_r[r], hcat(sm.A.values[gr.regimes[r][3]], sm.b.values[gr.regimes[r][3]])
            ) for r in eachindex(suf_r)
        ]
        for g in eachindex(gr.by_Q)
            rs = gr.by_Q[g]
            isempty(rs) && continue
            S = sum(Sq[r][1] for r in rs)
            n = sum(Sq[r][2] for r in rs)
            copyto!(sm.Q.values[g], _finalize_cov(S, T(n), sm.Q_prior, D))
        end
    end
    return nothing
end

# State-side prior contribution to the ELBO (summed over each block's groups).
function _tv_state_prior_term(sm, ::Type{T}) where {T}
    prior = zero(T)
    if sm.P0_prior !== nothing
        for P0 in sm.P0.values
            prior += iw_logprior_term(P0, sm.P0_prior)
        end
    end
    if sm.Q_prior !== nothing
        for Q in sm.Q.values
            prior += iw_logprior_term(Q, sm.Q_prior)
        end
    end
    return prior
end

# =============================================================================
# Gaussian trial-varying LDS
# =============================================================================

const _TVGaussianLDS{T} = LinearDynamicalSystem{
    T,<:TrialVaryingGaussianStateModel{T},<:TrialVaryingGaussianObservationModel{T}
}
const _TVPoissonLDS{T} = LinearDynamicalSystem{
    T,<:TrialVaryingGaussianStateModel{T},<:TrialVaryingPoissonObservationModel{T}
}

function _tv_check_labels_length(y, labels)
    n = length(y)
    for (k, v) in labels
        length(v) == n ||
            throw(DimensionMismatchError("labels[:$k] length", n, length(v)))
    end
    return n
end

"""
    rand([rng,] lds, tsteps_per_trial; labels)

Sample from a trial-varying LDS. Each trial uses the parameter values selected by
its group memberships (resolved from `labels`). Returns `(x, y)` vectors of
per-trial matrices.
"""
function Random.rand(
    rng::AbstractRNG, lds::_TVGaussianLDS{T}, tsteps_per_trial::AbstractVector{<:Integer};
    labels::AbstractDict,
) where {T<:Real}
    sm, om = lds.state_model, lds.obs_model
    ntrials = length(tsteps_per_trial)
    D, p = lds.latent_dim, lds.obs_dim
    gr = _tv_groups(sm, om, labels, ntrials; has_R=true)

    disp_obs = GaussianObservationModel(;
        C=om.C.values[1], R=om.R.values[1], d=om.d.values[1]
    )
    x = Vector{Matrix{T}}(undef, ntrials)
    y = Vector{Matrix{T}}(undef, ntrials)
    for t in 1:ntrials
        Ti = Int(tsteps_per_trial[t])
        sp = (
            A=sm.A.values[gr.gdyn[t]], B=zeros(T, D, 0), Q=sm.Q.values[gr.gQ[t]],
            b=sm.b.values[gr.gdyn[t]], x0=sm.x0.values[gr.gx0[t]], P0=sm.P0.values[gr.gP0[t]],
        )
        op = (
            C=om.C.values[gr.gobs[t]], R=om.R.values[gr.gR[t]], d=om.d.values[gr.gobs[t]],
            D=zeros(T, p, 0),
        )
        x[t] = Matrix{T}(undef, D, Ti)
        y[t] = Matrix{T}(undef, p, Ti)
        _sample_trial!(rng, x[t], y[t], sp, op, disp_obs, zeros(T, 0, Ti), zeros(T, 0, Ti))
    end
    return x, y
end

function Random.rand(
    rng::AbstractRNG, lds::_TVPoissonLDS{T}, tsteps_per_trial::AbstractVector{<:Integer};
    labels::AbstractDict,
) where {T<:Real}
    sm, om = lds.state_model, lds.obs_model
    ntrials = length(tsteps_per_trial)
    D, p = lds.latent_dim, lds.obs_dim
    gr = _tv_groups(sm, om, labels, ntrials; has_R=false)

    disp_obs = PoissonObservationModel(; C=om.C.values[1], d=om.d.values[1])
    x = Vector{Matrix{T}}(undef, ntrials)
    y = Vector{Matrix{T}}(undef, ntrials)
    for t in 1:ntrials
        Ti = Int(tsteps_per_trial[t])
        sp = (
            A=sm.A.values[gr.gdyn[t]], B=zeros(T, D, 0), Q=sm.Q.values[gr.gQ[t]],
            b=sm.b.values[gr.gdyn[t]], x0=sm.x0.values[gr.gx0[t]], P0=sm.P0.values[gr.gP0[t]],
        )
        op = (C=om.C.values[gr.gobs[t]], d=om.d.values[gr.gobs[t]])
        x[t] = Matrix{T}(undef, D, Ti)
        y[t] = Matrix{T}(undef, p, Ti)
        _sample_trial!(rng, x[t], y[t], sp, op, disp_obs, zeros(T, 0, Ti), zeros(T, 0, Ti))
    end
    return x, y
end

function Random.rand(
    lds::Union{_TVGaussianLDS,_TVPoissonLDS}, tsteps_per_trial::AbstractVector{<:Integer};
    kwargs...,
)
    return rand(Random.default_rng(), lds, tsteps_per_trial; kwargs...)
end

# Build the per-regime working state used by both fit! and smooth.
function _tv_regime_setup(lds, y, gr, ::Type{T}; gaussian::Bool) where {T}
    sm, om = lds.state_model, lds.obs_model
    D, p = lds.latent_dim, lds.obs_dim
    tsteps_all = [size(yt, 2) for yt in y]
    R = length(gr.regimes)

    regime_lds = if gaussian
        [_regime_gaussian_lds(sm, om, gr.regimes[r], D, p, lds.fit_bool, T) for r in 1:R]
    else
        [_regime_poisson_lds(sm, om, gr.regimes[r], D, p, lds.fit_bool, T) for r in 1:R]
    end

    tfs_global = initialize_FilterSmooth(lds, tsteps_all; cov_alias=false)
    y_r = Vector{Vector{Matrix{T}}}(undef, R)
    u_r = Vector{Vector{Matrix{T}}}(undef, R)
    v_r = Vector{Vector{Matrix{T}}}(undef, R)
    tfs_r = Vector{TrialFilterSmooth{T}}(undef, R)
    sws_r = Vector{Vector{SmoothWorkspace{T}}}(undef, R)
    suf_r = Vector{SufficientStatistics{T}}(undef, R)
    pool_size = Threads.maxthreadid()
    for r in 1:R
        trials = gr.regime_trials[r]
        ts = [tsteps_all[i] for i in trials]
        Tmax = maximum(ts)
        y_r[r] = [Matrix{T}(y[i]) for i in trials]
        u_r[r] = [zeros(T, 0, t) for t in ts]
        v_r[r] = [zeros(T, 0, t) for t in ts]
        tfs_r[r] = TrialFilterSmooth([tfs_global[i] for i in trials])
        npool = max(1, min(pool_size, length(trials)))
        sws_r[r] = [SmoothWorkspace(T, D, p, Tmax) for _ in 1:npool]
        suf_r[r] = _initialize_td_sufficient_statistics(T, regime_lds[r], ts)
        _td_init_const_blocks!(sws_r[r][1], regime_lds[r], ts, y_r[r], u_r[r], v_r[r])
    end
    return regime_lds, tfs_global, y_r, u_r, v_r, tfs_r, sws_r, suf_r
end

"""
    fit!(lds, y; labels, max_iter=100, tol=1e-6, progress=true)

Fit a trial-varying Gaussian LDS via EM. `labels::Dict{Symbol,Vector}` supplies
the per-trial label values for every label that any block keys on.
"""
function fit!(
    lds::_TVGaussianLDS{T}, y::AbstractVector{<:AbstractMatrix{T}};
    labels::AbstractDict, max_iter::Int=100, tol::Float64=1e-6, progress::Bool=true,
) where {T<:Real}
    sm, om = lds.state_model, lds.obs_model
    D, p = lds.latent_dim, lds.obs_dim
    ntrials = _tv_check_labels_length(y, labels)
    for t in 1:ntrials
        size(y[t], 1) == p ||
            throw(DimensionMismatchError("y[$t] channels", p, size(y[t], 1)))
    end
    gr = _tv_groups(sm, om, labels, ntrials; has_R=true)
    R = length(gr.regimes)

    regime_lds, tfs_global, y_r, u_r, v_r, tfs_r, sws_r, suf_r = _tv_regime_setup(
        lds, y, gr, T; gaussian=true
    )
    suf_comb = _initialize_td_sufficient_statistics(T, regime_lds[1], [size(yt, 2) for yt in y])

    elbos = Vector{T}()
    sizehint!(elbos, max_iter)
    prev_elbo = -T(Inf)
    prog = progress ? Progress(max_iter; desc="Fitting trial-varying LDS via EM...", barlen=50) : nothing

    for _ in 1:max_iter
        # E-step
        for r in 1:R
            smooth!(regime_lds[r], tfs_r[r], y_r[r], sws_r[r], u_r[r], v_r[r])
            _aggregate_td_suff_stats!(
                suf_r[r], tfs_r[r], regime_lds[r], u_r[r], v_r[r], y_r[r], sws_r[r][1]
            )
        end

        # ELBO (current params)
        total_entropy = zero(T)
        for fs in tfs_global.FilterSmooths
            total_entropy += fs.entropy
        end
        elbo = zero(T)
        for r in 1:R
            compute_smooth_constants!(sws_r[r][1], regime_lds[r])
            elbo += Q_state!(sws_r[r][1], regime_lds[r], suf_r[r])
            elbo += Q_obs!(sws_r[r][1], regime_lds[r], suf_r[r])
        end
        elbo += total_entropy + _tv_state_prior_term(sm, T)
        if om.R_prior !== nothing
            for Rg in om.R.values
                elbo += iw_logprior_term(Rg, om.R_prior)
            end
        end
        push!(elbos, elbo)

        # M-step: shared state, then emission [C d] (mean) and R (covariance).
        _tv_update_state!(sm, gr, suf_r, suf_comb, lds.fit_bool, D)
        if lds.fit_bool[5]
            for g in eachindex(gr.by_obs)
                rs = gr.by_obs[g]
                isempty(rs) && continue
                _combine_all_suff!(suf_comb, suf_r, rs)
                V = transpose(suf_comb.obs_xx[].chol \ suf_comb.obs_xy)  # p × (D+1)
                copyto!(om.C.values[g], view(V, :, 1:D))
                copyto!(om.d.values[g], view(V, :, D + 1))
            end
        end
        if lds.fit_bool[6]
            Sr = [
                _obs_scatter(
                    suf_r[r], hcat(om.C.values[gr.regimes[r][5]], om.d.values[gr.regimes[r][5]])
                ) for r in 1:R
            ]
            for g in eachindex(gr.by_R)
                rs = gr.by_R[g]
                isempty(rs) && continue
                S = sum(Sr[r][1] for r in rs)
                n = sum(Sr[r][2] for r in rs)
                copyto!(om.R.values[g], _finalize_cov(S, T(n), om.R_prior, p))
            end
        end

        prog !== nothing && next!(prog)
        if abs(elbo - prev_elbo) < tol
            prog !== nothing && finish!(prog)
            return elbos
        end
        prev_elbo = elbo
    end
    prog !== nothing && finish!(prog)
    return elbos
end

"""
    fit!(lds::TrialVaryingPoissonLDS, y; labels, max_iter=100, tol=1e-6,
         progress=true, newton_max_iter=20, newton_tol=1e-6)

Fit a trial-varying Poisson LDS via Laplace-EM. State blocks update as in the
Gaussian case; the emission [C d] is fit per group by the non-conjugate LBFGS
M-step.
"""
function fit!(
    lds::_TVPoissonLDS{T}, y::AbstractVector{<:AbstractMatrix{T}};
    labels::AbstractDict, max_iter::Int=100, tol::Float64=1e-6, progress::Bool=true,
    newton_max_iter::Int=20, newton_tol::Float64=1e-6,
) where {T<:Real}
    sm, om = lds.state_model, lds.obs_model
    D, p = lds.latent_dim, lds.obs_dim
    ntrials = _tv_check_labels_length(y, labels)
    for t in 1:ntrials
        size(y[t], 1) == p ||
            throw(DimensionMismatchError("y[$t] channels", p, size(y[t], 1)))
    end
    gr = _tv_groups(sm, om, labels, ntrials; has_R=false)
    R = length(gr.regimes)

    regime_lds, tfs_global, y_r, u_r, v_r, tfs_r, sws_r, suf_r = _tv_regime_setup(
        lds, y, gr, T; gaussian=false
    )
    suf_comb = _initialize_td_sufficient_statistics(T, regime_lds[1], [size(yt, 2) for yt in y])

    # Per-obs-group trial views for the LBFGS emission M-step.
    obs_trials = [findall(==(g), gr.gobs) for g in 1:ngroups(om.C)]
    obs_sws_pool = [SmoothWorkspace(T, D, p, maximum(size(yt, 2) for yt in y)) for _ in 1:Threads.maxthreadid()]

    elbos = Vector{T}()
    sizehint!(elbos, max_iter)
    prev_elbo = -T(Inf)
    prog = progress ? Progress(max_iter; desc="Fitting trial-varying Poisson LDS...", barlen=50) : nothing

    for _ in 1:max_iter
        for r in 1:R
            smooth!(
                regime_lds[r], tfs_r[r], y_r[r], sws_r[r];
                max_iter=newton_max_iter, tol=T(newton_tol),
            )
            _aggregate_td_suff_stats!(
                suf_r[r], tfs_r[r], regime_lds[r], u_r[r], v_r[r], y_r[r], sws_r[r][1]
            )
        end

        total_entropy = zero(T)
        for fs in tfs_global.FilterSmooths
            total_entropy += fs.entropy
        end
        elbo = zero(T)
        for r in 1:R
            compute_smooth_constants!(sws_r[r][1], regime_lds[r])
            elbo += Q_state!(sws_r[r][1], regime_lds[r], suf_r[r])
            for (j, fs) in enumerate(tfs_r[r].FilterSmooths)
                elbo += Q_obs!(sws_r[r][1], regime_lds[r], fs.x_smooth, fs.p_smooth, y_r[r][j])
            end
        end
        elbo += total_entropy + _tv_state_prior_term(sm, T)
        push!(elbos, elbo)

        _tv_update_state!(sm, gr, suf_r, suf_comb, lds.fit_bool, D)
        if lds.fit_bool[5]
            for g in eachindex(obs_trials)
                trials = obs_trials[g]
                isempty(trials) && continue
                scratch = LinearDynamicalSystem(;
                    state_model=GaussianStateModel(;
                        A=Matrix{T}(I, D, D), Q=Matrix{T}(I, D, D), b=zeros(T, D),
                        x0=zeros(T, D), P0=Matrix{T}(I, D, D),
                    ),
                    obs_model=PoissonObservationModel(; C=om.C.values[g], d=om.d.values[g]),
                    latent_dim=D, obs_dim=p, fit_bool=fill(true, 5),
                )
                tfs_g = TrialFilterSmooth([tfs_global[i] for i in trials])
                y_g = [y[i] for i in trials]
                update_observation_model!(scratch, tfs_g, y_g, obs_sws_pool)
            end
        end

        prog !== nothing && next!(prog)
        if abs(elbo - prev_elbo) < tol
            prog !== nothing && finish!(prog)
            return elbos
        end
        prev_elbo = elbo
    end
    prog !== nothing && finish!(prog)
    return elbos
end

# Single-trial / matrix convenience.
function fit!(lds::Union{_TVGaussianLDS,_TVPoissonLDS}, y::AbstractMatrix; labels, kwargs...)
    return fit!(lds, [y]; labels=labels, kwargs...)
end

"""
    smooth(lds, y; labels)

Smooth each trial of a trial-varying LDS using its regime's parameters.
Returns `(xs, Ps)`.
"""
function smooth(lds::_TVGaussianLDS{T}, y::AbstractVector{<:AbstractMatrix{T}}; labels::AbstractDict) where {T}
    sm, om = lds.state_model, lds.obs_model
    D, p = lds.latent_dim, lds.obs_dim
    ntrials = _tv_check_labels_length(y, labels)
    gr = _tv_groups(sm, om, labels, ntrials; has_R=true)
    xs = Vector{Matrix{T}}(undef, ntrials)
    Ps = Vector{Array{T,3}}(undef, ntrials)
    for t in 1:ntrials
        rlds = _regime_gaussian_lds(sm, om, (gr.gx0[t], gr.gP0[t], gr.gdyn[t], gr.gQ[t], gr.gobs[t], gr.gR[t]), D, p, lds.fit_bool, T)
        xt, Pt = smooth(rlds, y[t])
        xs[t] = xt
        Ps[t] = Pt
    end
    return xs, Ps
end

function smooth(lds::_TVPoissonLDS{T}, y::AbstractVector{<:AbstractMatrix{T}}; labels::AbstractDict, max_iter::Int=20, tol::T=T(1e-6)) where {T}
    sm, om = lds.state_model, lds.obs_model
    D, p = lds.latent_dim, lds.obs_dim
    ntrials = _tv_check_labels_length(y, labels)
    gr = _tv_groups(sm, om, labels, ntrials; has_R=false)
    xs = Vector{Matrix{T}}(undef, ntrials)
    Ps = Vector{Array{T,3}}(undef, ntrials)
    for t in 1:ntrials
        rlds = _regime_poisson_lds(sm, om, (gr.gx0[t], gr.gP0[t], gr.gdyn[t], gr.gQ[t], gr.gobs[t], 1), D, p, lds.fit_bool, T)
        fs = initialize_FilterSmooth(rlds, size(y[t], 2))::FilterSmooth{T}
        sws = SmoothWorkspace(T, D, p, size(y[t], 2))
        smooth!(rlds, fs, y[t], sws; max_iter=max_iter, tol=tol)
        xs[t] = copy(fs.x_smooth)
        Ps[t] = copy(fs.p_smooth)
    end
    return xs, Ps
end

"""
    loglikelihood(lds, y; labels)

Total marginal log-likelihood of a trial-varying Gaussian LDS, summed over trials
using each trial's regime parameters.
"""
function loglikelihood(lds::_TVGaussianLDS{T}, y::AbstractVector{<:AbstractMatrix{T}}; labels::AbstractDict) where {T}
    sm, om = lds.state_model, lds.obs_model
    D, p = lds.latent_dim, lds.obs_dim
    ntrials = _tv_check_labels_length(y, labels)
    gr = _tv_groups(sm, om, labels, ntrials; has_R=true)
    total = zero(T)
    for t in 1:ntrials
        rlds = _regime_gaussian_lds(sm, om, (gr.gx0[t], gr.gP0[t], gr.gdyn[t], gr.gQ[t], gr.gobs[t], gr.gR[t]), D, p, lds.fit_bool, T)
        total += loglikelihood(rlds, y[t])
    end
    return total
end
