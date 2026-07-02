# =============================================================================
# Stitched observation models.
#
# A *stitched* model "stitches together" several recording sessions (groups)
# that share a common latent state model but observe different channels (e.g.
# different recorded neurons), so each group carries its own emission and the
# number of observed channels may differ across groups.
#
# The implementation decomposes a stitched fit into one plain per-group LDS per
# group, each of which **shares the latent state model by reference** (so the
# shared-dynamics M-step propagates) while owning its group's emission (so the
# per-group emission M-step propagates). This lets the stitched path reuse the
# existing single-/multi-trial smoother, sufficient-statistics aggregator, and
# `update_*!` M-step primitives without modification.
#
# The latent dynamics (A, Q, b, x0, P0) are shared across groups; per group the
# emission is `(C_g, R_g, d_g)` (Gaussian) or `(C_g, d_g)` (Poisson). Per-trial
# group assignment is supplied via the `obs_group` keyword.
# =============================================================================

"""
    _resolve_obs_group(obs_model, obs_group) -> Vector{Int}

Map per-trial group labels in `obs_group` to integer group indices `1:G`
according to `obs_model.group_ids`. Throws if a label is not found.
"""
function _resolve_obs_group(
    obs_model::AbstractStitchedObservationModel, obs_group::AbstractVector
)
    idx = Vector{Int}(undef, length(obs_group))
    for (i, label) in enumerate(obs_group)
        j = findfirst(==(label), obs_model.group_ids)
        j === nothing && throw(
            ArgumentError(
                "obs_group label $(repr(label)) (trial $i) not found in group_ids " *
                "$(obs_model.group_ids)",
            ),
        )
        idx[i] = j
    end
    return idx
end

"""
    _group_trials(gidx, G) -> Vector{Vector{Int}}

Partition trial indices by group: `out[g]` lists the (global) trial indices that
belong to group `g`.
"""
function _group_trials(gidx::AbstractVector{Int}, G::Int)
    out = [Int[] for _ in 1:G]
    for (i, g) in enumerate(gidx)
        push!(out[g], i)
    end
    return out
end

# Build a plain per-group LDS that shares the latent `state_model` (by reference,
# so the shared-dynamics M-step is reflected) and owns group `g`'s emission
# sub-model (so the per-group emission M-step is reflected back into the stitched
# model). `fit_bool` is shared with the stitched LDS.
function _group_lds(
    lds::LinearDynamicalSystem{T,S,O}, g::Int
) where {T,S,O<:AbstractStitchedObservationModel{T}}
    return LinearDynamicalSystem(
        lds.state_model, lds.obs_model.models[g]; fit_bool=lds.fit_bool
    )
end

function _check_stitched_no_controls(lds::LinearDynamicalSystem)
    lds.state_input_dim == 0 || throw(
        ArgumentError(
            "stitched models do not support dynamics-input matrices (state_model.B " *
            "must be zero-column)",
        ),
    )
    lds.obs_input_dim == 0 || throw(
        ArgumentError("stitched models do not support observation-input matrices"),
    )
    return nothing
end

# Validate that `obs_group` is the right length and that each trial's channel
# count matches its group's emission dimension. Returns the group-index vector.
function _stitched_setup(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}},
    obs_group::AbstractVector,
) where {T,S,O<:AbstractStitchedObservationModel{T}}
    _check_stitched_no_controls(lds)
    ntrials = length(y)
    length(obs_group) == ntrials ||
        throw(DimensionMismatchError("obs_group length", ntrials, length(obs_group)))
    gidx = _resolve_obs_group(lds.obs_model, obs_group)
    for i in 1:ntrials
        g = gidx[i]
        p_g = size(lds.obs_model.models[g].C, 1)
        size(y[i], 1) == p_g || throw(
            DimensionMismatchError(
                "y[$i] channels (group $(lds.obs_model.group_ids[g]))", p_g, size(y[i], 1)
            ),
        )
    end
    return gidx
end

"""
    _combine_state_suff_stats!(dst, sufs)

Sum the *state-side* blocks (init_*, dyn_*) of the per-group sufficient-statistics
in `sufs` into `dst`. The observation-side blocks of `dst` are left untouched —
the shared-dynamics state M-step (`update_initial_state_*!`, `update_A_b!`,
`update_Q!`) only reads the state blocks. Sufficient statistics are additive
sums, so summing across the (disjoint) groups reconstructs the total over all
trials.
"""
function _combine_state_suff_stats!(
    dst::SufficientStatistics{T}, sufs::AbstractVector{SufficientStatistics{T}}
) where {T<:Real}
    D = size(dst.init_xy, 2)
    dyn_reg = size(dst.dyn_xy, 1)

    init_n = zero(T)
    dyn_n = zero(T)
    fill!(dst.init_xy, zero(T))
    fill!(dst.dyn_xy, zero(T))
    init_yy = zeros(T, D, D)
    dyn_xx = zeros(T, dyn_reg, dyn_reg)
    dyn_yy = zeros(T, D, D)

    for suf in sufs
        init_n += suf.init_n
        dyn_n += suf.dyn_n
        dst.init_xy .+= suf.init_xy
        dst.dyn_xy .+= suf.dyn_xy
        init_yy .+= suf.init_yy[].mat
        dyn_xx .+= suf.dyn_xx[].mat
        dyn_yy .+= suf.dyn_yy[].mat
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

# Shared-state + per-group-obs prior contribution to the ELBO. State-side priors
# (Q/P0/AB) are added once; observation-side priors (R/CD) are added per group.
function _stitched_prior_term(
    lds::LinearDynamicalSystem{T,S,O}, group_ldss::AbstractVector
) where {T,S,O<:GaussianObservationModelStitched{T}}
    prior = zero(T)
    sm = lds.state_model
    sm.Q_prior !== nothing && (prior += iw_logprior_term(sm.Q, sm.Q_prior))
    sm.P0_prior !== nothing && (prior += iw_logprior_term(sm.P0, sm.P0_prior))
    for glds in group_ldss
        om = glds.obs_model
        om.R_prior !== nothing && (prior += iw_logprior_term(om.R, om.R_prior))
    end
    return prior
end

# =============================================================================
# Gaussian stitched LDS
# =============================================================================

"""
    rand([rng,] lds, tsteps_per_trial; obs_group)

Sample from a stitched Gaussian/Poisson LDS. Each trial is generated from the
shared state model and its group's emission (selected by `obs_group`). Returns
`(x::Vector{Matrix}, y::Vector{Matrix})`; the `y[i]` channel count matches the
emission dimension of trial `i`'s group.
"""
function Random.rand(
    rng::AbstractRNG,
    lds::LinearDynamicalSystem{T,S,O},
    tsteps_per_trial::AbstractVector{<:Integer};
    obs_group::AbstractVector,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractStitchedObservationModel{T}}
    _check_stitched_no_controls(lds)
    ntrials = length(tsteps_per_trial)
    length(obs_group) == ntrials ||
        throw(DimensionMismatchError("obs_group length", ntrials, length(obs_group)))
    gidx = _resolve_obs_group(lds.obs_model, obs_group)

    state_params = _extract_state_params(lds.state_model)
    obs_params = [_extract_obs_params(m) for m in lds.obs_model.models]

    x = Vector{Matrix{T}}(undef, ntrials)
    y = Vector{Matrix{T}}(undef, ntrials)
    for i in 1:ntrials
        g = gidx[i]
        Ti = Int(tsteps_per_trial[i])
        p_g = size(lds.obs_model.models[g].C, 1)
        x[i] = Matrix{T}(undef, lds.latent_dim, Ti)
        y[i] = Matrix{T}(undef, p_g, Ti)
        u_trial = zeros(T, 0, Ti)
        v_trial = zeros(T, 0, Ti)
        _sample_trial!(
            rng,
            x[i],
            y[i],
            state_params,
            obs_params[g],
            lds.obs_model.models[g],
            u_trial,
            v_trial,
        )
    end
    return x, y
end

function Random.rand(
    lds::LinearDynamicalSystem{T,S,O},
    tsteps_per_trial::AbstractVector{<:Integer};
    kwargs...,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractStitchedObservationModel{T}}
    return rand(Random.default_rng(), lds, tsteps_per_trial; kwargs...)
end

"""
    smooth(lds, y; obs_group)

Smooth each trial of a stitched LDS using its group's emission and the shared
state model. Returns `(xs::Vector{Matrix}, Ps::Vector{Array{T,3}})`.
"""
function smooth(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    obs_group::AbstractVector,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModelStitched{T}}
    gidx = _stitched_setup(lds, y, obs_group)
    G = length(lds.obs_model.group_ids)
    group_ldss = [_group_lds(lds, g) for g in 1:G]

    xs = Vector{Matrix{T}}(undef, length(y))
    Ps = Vector{Array{T,3}}(undef, length(y))
    for i in eachindex(y)
        xi, Pi = smooth(group_ldss[gidx[i]], y[i])
        xs[i] = xi
        Ps[i] = Pi
    end
    return xs, Ps
end

"""
    loglikelihood(lds, y; obs_group)

Total marginal (observed-data) log-likelihood of a stitched Gaussian LDS,
summed over trials using each trial's group emission (Kalman filter per group).
"""
function loglikelihood(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    obs_group::AbstractVector,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModelStitched{T}}
    gidx = _stitched_setup(lds, y, obs_group)
    G = length(lds.obs_model.group_ids)
    group_ldss = [_group_lds(lds, g) for g in 1:G]
    total = zero(T)
    for i in eachindex(y)
        total += loglikelihood(group_ldss[gidx[i]], y[i])
    end
    return total
end

"""
    fit!(lds, y; obs_group, max_iter=100, tol=1e-6, progress=true)

Fit a stitched Gaussian LDS via Expectation-Maximization. The latent dynamics
are shared across groups; each group's emission `(C_g, R_g, d_g)` is fit from its
own trials. `obs_group` is a per-trial vector of group labels matched against
`lds.obs_model.group_ids`.

Returns a `Vector{T}` of ELBO values, one per iteration.
"""
function fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    obs_group::AbstractVector,
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModelStitched{T}}
    gidx = _stitched_setup(lds, y, obs_group)
    G = length(lds.obs_model.group_ids)
    group_trials = _group_trials(gidx, G)
    D = lds.latent_dim

    group_ldss = [_group_lds(lds, g) for g in 1:G]
    tsteps_all = [size(yt, 2) for yt in y]

    # Global per-trial smoother buffers (real p_smooth so single-trial groups
    # write into them; multi-trial equal-length groups alias to shared cov).
    tfs_global = initialize_FilterSmooth(group_ldss[1], tsteps_all; cov_alias=false)

    # Per-group state: trial subset, smoother view, workspace pool, suff stats.
    y_g = Vector{Vector{Matrix{T}}}(undef, G)
    u_g = Vector{Vector{Matrix{T}}}(undef, G)
    v_g = Vector{Vector{Matrix{T}}}(undef, G)
    tfs_g = Vector{TrialFilterSmooth{T}}(undef, G)
    sws_g = Vector{Vector{SmoothWorkspace{T}}}(undef, G)
    suf_g = Vector{SufficientStatistics{T}}(undef, G)

    pool_size = Threads.maxthreadid()
    for g in 1:G
        trials = group_trials[g]
        isempty(trials) && continue
        p_g = size(lds.obs_model.models[g].C, 1)
        ts_g = [tsteps_all[i] for i in trials]
        T_max_g = maximum(ts_g)
        y_g[g] = [Matrix{T}(y[i]) for i in trials]
        u_g[g] = [zeros(T, 0, ti) for ti in ts_g]
        v_g[g] = [zeros(T, 0, ti) for ti in ts_g]
        tfs_g[g] = TrialFilterSmooth([tfs_global[i] for i in trials])
        npool = max(1, min(pool_size, length(trials)))
        sws_g[g] = [SmoothWorkspace(T, D, p_g, T_max_g) for _ in 1:npool]
        suf_g[g] = _initialize_td_sufficient_statistics(T, group_ldss[g], ts_g)
        _td_init_const_blocks!(sws_g[g][1], group_ldss[g], ts_g, y_g[g], u_g[g], v_g[g])
    end

    # Combined state suff-stats (allocated once; obs blocks unused).
    suf_state = _initialize_td_sufficient_statistics(T, group_ldss[1], tsteps_all)
    nonempty = [g for g in 1:G if !isempty(group_trials[g])]
    state_sws = sws_g[nonempty[1]][1]

    prev_elbo = -T(Inf)
    elbos = Vector{T}()
    sizehint!(elbos, max_iter)

    prog = if progress
        Progress(max_iter; desc="Fitting stitched LDS via EM...", barlen=50, showspeed=true)
    else
        nothing
    end

    for _ in 1:max_iter
        # E-step: smooth + aggregate per group.
        for g in nonempty
            smooth!(group_ldss[g], tfs_g[g], y_g[g], sws_g[g], u_g[g], v_g[g])
            _aggregate_td_suff_stats!(
                suf_g[g], tfs_g[g], group_ldss[g], u_g[g], v_g[g], y_g[g], sws_g[g][1]
            )
        end

        # ELBO with current parameters (before M-step).
        total_entropy = zero(T)
        for fs in tfs_global.FilterSmooths
            total_entropy += fs.entropy
        end
        elbo = zero(T)
        for g in nonempty
            compute_smooth_constants!(sws_g[g][1], group_ldss[g])
            elbo += Q_state!(sws_g[g][1], group_ldss[g], suf_g[g])
            elbo += Q_obs!(sws_g[g][1], group_ldss[g], suf_g[g])
        end
        elbo += total_entropy + _stitched_prior_term(lds, group_ldss[nonempty])
        push!(elbos, elbo)

        # M-step: shared state from combined suff-stats, emission per group.
        _combine_state_suff_stats!(suf_state, [suf_g[g] for g in nonempty])
        update_initial_state_mean!(group_ldss[nonempty[1]], suf_state)
        update_initial_state_covariance!(group_ldss[nonempty[1]], suf_state, state_sws)
        update_A_b!(group_ldss[nonempty[1]], suf_state, state_sws)
        update_Q!(group_ldss[nonempty[1]], suf_state, state_sws)
        for g in nonempty
            update_C_d!(group_ldss[g], suf_g[g], sws_g[g][1])
            update_R!(group_ldss[g], suf_g[g], sws_g[g][1])
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

# =============================================================================
# Poisson stitched LDS
# =============================================================================

function _stitched_prior_term(
    lds::LinearDynamicalSystem{T,S,O}, group_ldss::AbstractVector
) where {T,S,O<:PoissonObservationModelStitched{T}}
    prior = zero(T)
    sm = lds.state_model
    sm.Q_prior !== nothing && (prior += iw_logprior_term(sm.Q, sm.Q_prior))
    sm.P0_prior !== nothing && (prior += iw_logprior_term(sm.P0, sm.P0_prior))
    for glds in group_ldss
        cdp = glds.obs_model.CD_prior
        if cdp !== nothing
            Dl = glds.latent_dim
            W = Matrix{T}(undef, glds.obs_dim, Dl + 1)
            @views W[:, 1:Dl] .= glds.obs_model.C
            @views W[:, Dl + 1] .= glds.obs_model.d
            Wm = W .- cdp.M₀
            prior -= T(0.5) * sum(Wm .* (Wm * cdp.Λ))
        end
    end
    return prior
end

function smooth(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    obs_group::AbstractVector,
    max_iter::Int=20,
    tol::T=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModelStitched{T}}
    gidx = _stitched_setup(lds, y, obs_group)
    G = length(lds.obs_model.group_ids)
    group_ldss = [_group_lds(lds, g) for g in 1:G]

    xs = Vector{Matrix{T}}(undef, length(y))
    Ps = Vector{Array{T,3}}(undef, length(y))
    for i in eachindex(y)
        glds = group_ldss[gidx[i]]
        fs = initialize_FilterSmooth(glds, size(y[i], 2))::FilterSmooth{T}
        sws = SmoothWorkspace(T, glds.latent_dim, glds.obs_dim, size(y[i], 2))
        smooth!(glds, fs, y[i], sws; max_iter=max_iter, tol=tol)
        xs[i] = copy(fs.x_smooth)
        Ps[i] = copy(fs.p_smooth)
    end
    return xs, Ps
end

"""
    fit!(lds, y; obs_group, max_iter=100, tol=1e-6, progress=true,
         newton_max_iter=20, newton_tol=1e-6)

Fit a stitched Poisson LDS via Laplace-EM. Shared latent dynamics; per-group
emission `(C_g, d_g)` fit by the (non-conjugate) LBFGS emission M-step.
"""
function fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    obs_group::AbstractVector,
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
    newton_max_iter::Int=20,
    newton_tol::Float64=1e-6,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModelStitched{T}}
    gidx = _stitched_setup(lds, y, obs_group)
    G = length(lds.obs_model.group_ids)
    group_trials = _group_trials(gidx, G)
    D = lds.latent_dim

    group_ldss = [_group_lds(lds, g) for g in 1:G]
    tsteps_all = [size(yt, 2) for yt in y]

    tfs_global = initialize_FilterSmooth(group_ldss[1], tsteps_all; cov_alias=false)

    y_g = Vector{Vector{Matrix{T}}}(undef, G)
    u_g = Vector{Vector{Matrix{T}}}(undef, G)
    v_g = Vector{Vector{Matrix{T}}}(undef, G)
    tfs_g = Vector{TrialFilterSmooth{T}}(undef, G)
    sws_g = Vector{Vector{SmoothWorkspace{T}}}(undef, G)
    suf_g = Vector{SufficientStatistics{T}}(undef, G)

    pool_size = Threads.maxthreadid()
    for g in 1:G
        trials = group_trials[g]
        isempty(trials) && continue
        p_g = size(lds.obs_model.models[g].C, 1)
        ts_g = [tsteps_all[i] for i in trials]
        T_max_g = maximum(ts_g)
        y_g[g] = [Matrix{T}(y[i]) for i in trials]
        u_g[g] = [zeros(T, 0, ti) for ti in ts_g]
        v_g[g] = [zeros(T, 0, ti) for ti in ts_g]
        tfs_g[g] = TrialFilterSmooth([tfs_global[i] for i in trials])
        npool = max(1, min(pool_size, length(trials)))
        sws_g[g] = [SmoothWorkspace(T, D, p_g, T_max_g) for _ in 1:npool]
        suf_g[g] = _initialize_td_sufficient_statistics(T, group_ldss[g], ts_g)
        _td_init_const_blocks!(sws_g[g][1], group_ldss[g], ts_g, y_g[g], u_g[g], v_g[g])
    end

    suf_state = _initialize_td_sufficient_statistics(T, group_ldss[1], tsteps_all)
    nonempty = [g for g in 1:G if !isempty(group_trials[g])]
    state_sws = sws_g[nonempty[1]][1]

    elbos = Vector{T}()
    sizehint!(elbos, max_iter)
    prev_elbo = -T(Inf)

    prog = if progress
        Progress(
            max_iter; desc="Fitting stitched Poisson LDS...", barlen=50, showspeed=true
        )
    else
        nothing
    end

    for _ in 1:max_iter
        # E-step: smooth + state aggregate per group.
        for g in nonempty
            smooth!(
                group_ldss[g], tfs_g[g], y_g[g], sws_g[g]; max_iter=newton_max_iter,
                tol=T(newton_tol),
            )
            _aggregate_td_suff_stats!(
                suf_g[g], tfs_g[g], group_ldss[g], u_g[g], v_g[g], y_g[g], sws_g[g][1]
            )
        end

        # ELBO (Poisson Q_obs is per-trial and non-conjugate).
        total_entropy = zero(T)
        for fs in tfs_global.FilterSmooths
            total_entropy += fs.entropy
        end
        elbo = zero(T)
        for g in nonempty
            compute_smooth_constants!(sws_g[g][1], group_ldss[g])
            elbo += Q_state!(sws_g[g][1], group_ldss[g], suf_g[g])
            for (j, fs) in enumerate(tfs_g[g].FilterSmooths)
                elbo += Q_obs!(sws_g[g][1], group_ldss[g], fs.x_smooth, fs.p_smooth, y_g[g][j])
            end
        end
        elbo += total_entropy + _stitched_prior_term(lds, group_ldss[nonempty])
        push!(elbos, elbo)

        # M-step: shared state, then per-group emission via LBFGS.
        _combine_state_suff_stats!(suf_state, [suf_g[g] for g in nonempty])
        update_initial_state_mean!(group_ldss[nonempty[1]], suf_state)
        update_initial_state_covariance!(group_ldss[nonempty[1]], suf_state, state_sws)
        update_A_b!(group_ldss[nonempty[1]], suf_state, state_sws)
        update_Q!(group_ldss[nonempty[1]], suf_state, state_sws)
        for g in nonempty
            update_observation_model!(group_ldss[g], tfs_g[g], y_g[g], sws_g[g])
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

# Single-trial / 3D convenience overloads (stitched LDS).
function fit!(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}; obs_group, kwargs...
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractStitchedObservationModel{T}}
    return fit!(lds, [y]; obs_group=obs_group, kwargs...)
end

# =============================================================================
# Switching stitched LDS (SLDS)
#
# Each discrete state `k` carries a stitched emission across groups, so the
# component `slds.LDSs[k]` has a stitched observation model. The switching layer
# (A, πₖ) and the per-state latent dynamics are shared across groups. The fit
# reuses the plain-SLDS smoother / weighted aggregator via per-group "group
# SLDSs" that share the discrete layer and per-state state models.
# =============================================================================

# Build a per-group plain SLDS: shares `slds.A` / `slds.πₖ` (by reference) and
# each component's state model, with group `g`'s per-state emission.
function _group_slds(slds::SLDS{T,S,O}, g::Int) where {T,S,O<:AbstractStitchedObservationModel{T}}
    ldss = [_group_lds(slds.LDSs[k], g) for k in eachindex(slds.LDSs)]
    return SLDS(; A=slds.A, πₖ=slds.πₖ, LDSs=ldss)
end

function _stitched_setup_slds(
    slds::SLDS{T,S,O}, y::AbstractVector{<:AbstractMatrix{T}}, obs_group::AbstractVector
) where {T,S,O<:AbstractStitchedObservationModel{T}}
    om = slds.LDSs[1].obs_model
    ntrials = length(y)
    length(obs_group) == ntrials ||
        throw(DimensionMismatchError("obs_group length", ntrials, length(obs_group)))
    gidx = _resolve_obs_group(om, obs_group)
    for i in 1:ntrials
        g = gidx[i]
        p_g = size(om.models[g].C, 1)
        size(y[i], 1) == p_g ||
            throw(DimensionMismatchError("y[$i] channels", p_g, size(y[i], 1)))
    end
    return gidx
end

"""
    rand([rng,] slds, tsteps_per_trial; obs_group)

Sample from a stitched SLDS. Returns `(z::Vector{Vector{Int}}, x::Vector{Matrix},
y::Vector{Matrix})`; per trial the emission of the active discrete state is taken
from that trial's group.
"""
function Random.rand(
    rng::AbstractRNG,
    slds::SLDS{T,S,O},
    tsteps_per_trial::AbstractVector{<:Integer};
    obs_group::AbstractVector,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractStitchedObservationModel{T}}
    ntrials = length(tsteps_per_trial)
    length(obs_group) == ntrials ||
        throw(DimensionMismatchError("obs_group length", ntrials, length(obs_group)))
    om = slds.LDSs[1].obs_model
    gidx = _resolve_obs_group(om, obs_group)
    latent_dim = slds.LDSs[1].latent_dim

    state_params = [_extract_state_params(lds.state_model) for lds in slds.LDSs]

    z = Vector{Vector{Int}}(undef, ntrials)
    x = Vector{Matrix{T}}(undef, ntrials)
    y = Vector{Matrix{T}}(undef, ntrials)
    for trial in 1:ntrials
        g = gidx[trial]
        Ti = Int(tsteps_per_trial[trial])
        p_g = size(om.models[g].C, 1)
        obs_params = [_extract_obs_params(lds.obs_model.models[g]) for lds in slds.LDSs]
        z[trial] = Vector{Int}(undef, Ti)
        x[trial] = Matrix{T}(undef, latent_dim, Ti)
        y[trial] = Matrix{T}(undef, p_g, Ti)
        _sample_slds_trial!(
            rng,
            z[trial],
            x[trial],
            y[trial],
            slds.A,
            slds.πₖ,
            state_params,
            obs_params,
            om.models[g],
        )
    end
    return z, x, y
end

function Random.rand(
    slds::SLDS{T,S,O}, tsteps_per_trial::AbstractVector{<:Integer}; kwargs...
) where {T<:Real,S<:AbstractStateModel,O<:AbstractStitchedObservationModel{T}}
    return rand(Random.default_rng(), slds, tsteps_per_trial; kwargs...)
end

"""
    fit!(slds, y; obs_group, max_iter=50, progress=true)

Fit a stitched SLDS via variational Laplace EM. The switching layer and the
per-state latent dynamics are shared across groups; each discrete state's
emission is fit per group.
"""
function fit!(
    slds::SLDS{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    obs_group::AbstractVector,
    max_iter::Int=50,
    progress::Bool=true,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractStitchedObservationModel{T}}
    gidx = _stitched_setup_slds(slds, y, obs_group)
    K = length(slds.LDSs)
    G = length(slds.LDSs[1].obs_model.group_ids)
    latent_dim = slds.LDSs[1].latent_dim
    group_trials = _group_trials(gidx, G)

    tsteps_all = [size(yt, 2) for yt in y]
    ntrials = length(y)
    seq_ends = cumsum(tsteps_all)
    total_T = last(seq_ends)
    T_max = maximum(tsteps_all)

    # Per-group SLDS views (shared discrete layer + per-state state models).
    group_slds = [_group_slds(slds, g) for g in 1:G]
    group_ws = Union{Nothing,SLDSSmoothWorkspace{T}}[nothing for _ in 1:G]
    group_msws = Union{Nothing,SmoothWorkspace{T}}[nothing for _ in 1:G]
    for g in 1:G
        isempty(group_trials[g]) && continue
        p_g = size(slds.LDSs[1].obs_model.models[g].C, 1)
        group_ws[g] = SLDSSmoothWorkspace(T, group_slds[g], T_max)
        group_msws[g] = SmoothWorkspace(T, latent_dim, p_g, T_max)
    end

    tfs = initialize_FilterSmooth(slds.LDSs[1], tsteps_all)::TrialFilterSmooth{T}
    dl = SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(T, K, total_T))
    fb_storage = _make_slds_fb_storage(dl, seq_ends)
    obs_seq = collect(1:total_T)
    ctrl_seq = fill(nothing, total_T)
    x_samples = [Matrix{T}(undef, latent_dim, Ti) for Ti in tsteps_all]
    randn_buf = Vector{T}(undef, latent_dim)

    elbos = Vector{T}(undef, max_iter)
    prog = if progress
        Progress(max_iter; desc="Fitting stitched SLDS via EM...", barlen=50, showspeed=true)
    else
        nothing
    end

    # Warm-start: smooth each trial with uniform discrete weights.
    for trial in 1:ntrials
        g = gidx[trial]
        Ti = tsteps_all[trial]
        w_uniform = fill(one(T) / K, K, Ti)
        smooth!(group_slds[g], tfs[trial], y[trial], w_uniform; ws=group_ws[g])
    end

    for iter in 1:max_iter
        sample_posterior!(x_samples, Random.default_rng(), tfs, randn_buf)
        elbos[iter] = _stitched_slds_estep!(
            slds, group_slds, group_ws, gidx, tfs, fb_storage, dl, y, x_samples;
            obs_seq=obs_seq, ctrl_seq=ctrl_seq, seq_ends=seq_ends,
        )
        _stitched_slds_mstep!(
            slds, group_slds, group_trials, gidx, tfs, fb_storage, dl, y, group_msws;
            obs_seq=obs_seq, seq_ends=seq_ends,
        )
        for g in 1:G
            isempty(group_trials[g]) && continue
            refresh_slds_constants!(group_ws[g], group_slds[g])
        end
        prog !== nothing && next!(prog; showvalues=[(:iteration, iter), (:ELBO, elbos[iter])])
    end

    prog !== nothing && finish!(prog)
    return elbos
end

function fit!(
    slds::SLDS{T,S,O}, y::AbstractMatrix{T}; obs_group, kwargs...
) where {T<:Real,S<:AbstractStateModel,O<:AbstractStitchedObservationModel{T}}
    return fit!(slds, [y]; obs_group=obs_group, kwargs...)
end

function _stitched_slds_estep!(
    slds::SLDS{T},
    group_slds::AbstractVector,
    group_ws::AbstractVector,
    gidx::AbstractVector{Int},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    x_samples::AbstractVector{<:AbstractMatrix{T}};
    obs_seq::AbstractVector,
    ctrl_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
) where {T<:Real}
    ntrials = length(y)
    K = length(slds.LDSs)

    # Per-state log-likelihood from sampled trajectory, using each trial's group.
    for trial in 1:ntrials
        g = gidx[trial]
        ws_g = group_ws[g]::SLDSSmoothWorkspace{T}
        gslds = group_slds[g]
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        for k in 1:K
            ll_view = view(dl.logL, k, t1:t2)
            joint_loglikelihood!(ll_view, ws_g, ws_g.consts[k], gslds.LDSs[k], x_samples[trial], y[trial])
        end
    end

    HMMs.forward_backward!(
        fb_storage, dl, obs_seq, ctrl_seq; seq_ends=seq_ends, transition_marginals=true
    )

    total_elbo = zero(T)
    for trial in 1:ntrials
        g = gidx[trial]
        ws_g = group_ws[g]::SLDSSmoothWorkspace{T}
        gslds = group_slds[g]
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        Tsteps = t2 - t1 + 1
        w = view(fb_storage.γ, :, t1:t2)

        smooth!(gslds, tfs[trial], y[trial], w; ws=ws_g)

        trial_elbo = zero(T)
        x_smooth_trial = tfs[trial].x_smooth
        for k in 1:K
            ll = view(ws_g.ll_tmp, 1:Tsteps)
            joint_loglikelihood!(ll, ws_g, ws_g.consts[k], gslds.LDSs[k], x_smooth_trial, y[trial])
            for t in 1:Tsteps
                trial_elbo += w[k, t] * ll[t]
            end
        end
        for k in 1:K
            trial_elbo += w[k, 1] * log(slds.πₖ[k] + T(1e-12))
        end
        for t in t1:(t2 - 1)
            ξt = fb_storage.ξ[t]
            for i in 1:K, j in 1:K
                trial_elbo += ξt[i, j] * log(slds.A[i, j] + T(1e-12))
            end
        end
        trial_elbo -= tfs[trial].entropy
        for k in 1:K, t in 1:Tsteps
            wkt = w[k, t]
            wkt > 0 && (trial_elbo += wkt * log(wkt + T(1e-12)))
        end
        total_elbo += trial_elbo
    end
    return total_elbo
end

function _stitched_slds_mstep!(
    slds::SLDS{T},
    group_slds::AbstractVector,
    group_trials::AbstractVector,
    gidx::AbstractVector{Int},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    group_msws::AbstractVector;
    obs_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
) where {T<:Real}
    K = length(slds.LDSs)
    G = length(group_trials)

    # Discrete-layer update (shared A, πₖ).
    StatsAPI.fit!(dl, fb_storage, obs_seq; seq_ends=seq_ends)

    nonempty = [g for g in 1:G if !isempty(group_trials[g])]

    # Per-group reusable suff stats / control stubs / tfs views.
    suf_g = Vector{SufficientStatistics{T}}(undef, G)
    tfs_g = Vector{TrialFilterSmooth{T}}(undef, G)
    y_g = Vector{Vector{Matrix{T}}}(undef, G)
    u_g = Vector{Vector{Matrix{T}}}(undef, G)
    v_g = Vector{Vector{Matrix{T}}}(undef, G)
    for g in nonempty
        trials = group_trials[g]
        ts_g = [size(y[i], 2) for i in trials]
        y_g[g] = [y[i] for i in trials]
        u_g[g] = [zeros(T, 0, ti) for ti in ts_g]
        v_g[g] = [zeros(T, 0, ti) for ti in ts_g]
        tfs_g[g] = TrialFilterSmooth([tfs[i] for i in trials])
        suf_g[g] = _initialize_td_sufficient_statistics(T, group_slds[g].LDSs[1], ts_g)
    end

    suf_state = _initialize_td_sufficient_statistics(
        T, group_slds[nonempty[1]].LDSs[1], [size(yt, 2) for yt in y]
    )

    for k in 1:K
        # Aggregate weighted suff-stats per group and update emission.
        for g in nonempty
            trials = group_trials[g]
            weights = Vector{AbstractVector{T}}(undef, length(trials))
            for (j, i) in enumerate(trials)
                t1, t2 = HMMs.seq_limits(seq_ends, i)
                weights[j] = view(fb_storage.γ, k, t1:t2)
            end
            lds_kg = group_slds[g].LDSs[k]
            msws_g = group_msws[g]::SmoothWorkspace{T}
            _aggregate_td_suff_stats_weighted!(
                suf_g[g], tfs_g[g], lds_kg, u_g[g], v_g[g], y_g[g], weights, msws_g
            )
            if lds_kg.obs_model isa GaussianObservationModel{T}
                update_C_d!(lds_kg, suf_g[g], msws_g)
                update_R!(lds_kg, suf_g[g], msws_g)
            elseif lds_kg.obs_model isa PoissonObservationModel{T}
                update_observation_model!(lds_kg, tfs_g[g], y_g[g], [msws_g], weights)
            else
                throw(ArgumentError("Unsupported observation model $(typeof(lds_kg.obs_model))"))
            end
        end

        # Shared state update for discrete state k from combined suff-stats.
        _combine_state_suff_stats!(suf_state, [suf_g[g] for g in nonempty])
        lds_k1 = group_slds[nonempty[1]].LDSs[k]
        msws = group_msws[nonempty[1]]::SmoothWorkspace{T}
        update_initial_state_mean!(lds_k1, suf_state)
        update_initial_state_covariance!(lds_k1, suf_state, msws)
        update_A_b!(lds_k1, suf_state, msws)
        update_Q!(lds_k1, suf_state, msws)
    end
    return nothing
end
