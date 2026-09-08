#=============================================================================
Switching LDS (SLDS)

Optional control inputs `ux` (dynamics, `Bₖ u`) and `uy` (observation, `Dₖ v`)
are shared across regimes; the active regime `zₜ` selects which per-regime
`Bₖ` / `Dₖ` multiplies them. `nothing` / zero-row matrices skip the terms.

    Sample:         rand(rng, slds, tsteps; ux, uy)

    Log-Likelihood: joint_loglikelihood!(ws, slds, x, y, w[, ux, uy])

    Gradient:       gradient!(ws, slds, x, y, w[, ux, uy])

    Hessian:        hessian!(ws, slds, x, y, w[, uy])

    Smooth:         smooth!(slds, fs, y, w; x_sample, rng, ux, uy)  # optional joint draw

    E-Step:         estep!(slds, tfs, fb_storage, dl, y, x_samples, pool, plan; ux, uy)

    M-Step:         mstep!(slds, tfs, fb_storage, dl, y, sws; ux, uy)

    Fit:            fit!(slds, y; ux, uy, smoothing_iters)

    Infer:          smooth(slds, y; ux, uy, smoothing_iters, tol)
                                                    # -> (; x, γ, elbo, trial_elbo, p)

    ELBO:           elbo(slds, y; ux, uy) == loglikelihood(slds, y; ux, uy)
                    trial_elbos(slds, y; ux, uy)    # per trial
=============================================================================#

"""
    _make_slds_fb_storage(dl, seq_ends)

Allocate a single `HMMs.ForwardBackwardStorage` covering all trials. `seq_ends` is the
cumulative timestep index at which each trial ends (HMMs.jl convention). The fb_storage
buffers are sized at `K × sum(T_i)` and `dl.logL` is sized to match.
"""
function _make_slds_fb_storage(
    dl::SLDSDiscreteLayer{T}, seq_ends::AbstractVector{Int}
) where {T}
    total_T = last(seq_ends)
    #=
    HMMs.jl "observations" are just timestep indices into dl.logL; there is no
    control sequence. These are unrelated to the LDS ux / uy
    control-input kwargs.
    =#
    obs_seq = 1:total_T
    control_seq = fill(nothing, total_T)
    return HMMs.initialize_forward_backward(
        dl, obs_seq, control_seq; seq_ends=seq_ends, transition_marginals=true
    )
end

"""
    rand([rng,] slds, tsteps::Integer; ux=nothing, uy=nothing)
    rand([rng,] slds, tsteps_per_trial::AbstractVector{<:Integer}; ux=nothing, uy=nothing)

Sample from a Switching Linear Dynamical System.

- Scalar `tsteps`: returns one trial as `(z::Vector{Int}, x::Matrix, y::Matrix)`.
- Vector of per-trial lengths: returns `(z::Vector{Vector{Int}}, x::Vector{Matrix},
  y::Vector{Matrix})`. Trial lengths may differ.

Optional control inputs (shared across models; the active mode `zₜ` selects
which per-regime `Bₖ` / `Dₖ` multiplies them):
- `ux`: dynamics input consumed by `Bₖ` (`xₜ ~ N(Aₖ xₜ₋₁ + bₖ + Bₖ uₜ₋₁, Qₖ)`).
  Scalar form is an `(ux_dim, tsteps)` matrix; multi-trial is a vector of
  per-trial matrices.
- `uy`: observation input consumed by `Dₖ`. Same shape family as `ux`; required
  when the LDS carry a nonzero-column `D`. Supported for both Gaussian and
  Poisson emissions.
"""
function Random.rand(
    rng::AbstractRNG,
    slds::SLDS{T,S,O},
    tsteps::Integer;
    ux::Union{Nothing,AbstractMatrix{T}}=nothing,
    uy::Union{Nothing,AbstractMatrix{T},NamedTuple}=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    lds1 = slds.LDSs[1]
    latent_dim = lds1.latent_dim
    Ti = Int(tsteps)

    ux_trial = _check_ux(ux, lds1.ux_dim, Ti, "ux", T)
    uy_trial = _check_uy(uy, lds1.uy_dim, Ti, lds1.obs_model)

    z = Vector{Int}(undef, Ti)
    x = Matrix{T}(undef, latent_dim, Ti)
    y = _alloc_obs(lds1, Ti)

    if depends_on === nothing && _has_parameter_dependence(lds1)
        _single_trial_group_error("slds")
    end
    grp = _slds_parameter_grouping(slds, 1; depends_on=depends_on)
    regimes = if grp === nothing
        slds.LDSs
    else
        _slds_cell_sldss(slds, grp)[grp.trial_cell[1]].LDSs
    end

    _warn_slds_unstable_rollout(slds, Ti)
    state_params = [_extract_state_params(lds.state_model) for lds in regimes]
    obs_params = [_extract_obs_params(lds.obs_model) for lds in regimes]

    _sample_slds_trial!(
        rng,
        z,
        x,
        y,
        slds.A,
        slds.πₖ,
        state_params,
        obs_params,
        lds1.obs_model,
        ux_trial,
        uy_trial,
    )

    return z, x, y
end

function Random.rand(
    rng::AbstractRNG,
    slds::SLDS{T,S,O},
    tsteps_per_trial::AbstractVector{<:Integer};
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}},NamedTuple}=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    lds1 = slds.LDSs[1]
    latent_dim = lds1.latent_dim
    ntrials = length(tsteps_per_trial)

    ux_seq = _normalize_multitrial_ux(ux, lds1.ux_dim, tsteps_per_trial, T, "ux")
    uy_seq = _normalize_multitrial_uy(uy, lds1.uy_dim, tsteps_per_trial, T, lds1.obs_model)

    z = Vector{Vector{Int}}(undef, ntrials)
    x = Vector{Matrix{T}}(undef, ntrials)
    y = Vector{typeof(_alloc_obs(lds1, 1))}(undef, ntrials)

    #=
    Per-trial, per-regime parameter sets: one entry per trial, each a vector
    over regimes. Ungrouped, every trial shares the same vector.
    =#
    _warn_slds_unstable_rollout(slds, maximum(tsteps_per_trial))
    grp = _slds_parameter_grouping(slds, ntrials; depends_on=depends_on)
    if grp === nothing
        base_state = [_extract_state_params(lds.state_model) for lds in slds.LDSs]
        base_obs = [_extract_obs_params(lds.obs_model) for lds in slds.LDSs]
        state_of = fill(base_state, ntrials)
        obs_of = fill(base_obs, ntrials)
    else
        cell_slds = _slds_cell_sldss(slds, grp)
        cell_state = [
            [_extract_state_params(lds.state_model) for lds in sc.LDSs] for sc in cell_slds
        ]
        cell_obs = [
            [_extract_obs_params(lds.obs_model) for lds in sc.LDSs] for sc in cell_slds
        ]
        state_of = [cell_state[grp.trial_cell[n]] for n in 1:ntrials]
        obs_of = [cell_obs[grp.trial_cell[n]] for n in 1:ntrials]
    end

    for trial in 1:ntrials
        Ti = Int(tsteps_per_trial[trial])
        z[trial] = Vector{Int}(undef, Ti)
        x[trial] = Matrix{T}(undef, latent_dim, Ti)
        y[trial] = _alloc_obs(lds1, Ti)
        _sample_slds_trial!(
            rng,
            z[trial],
            x[trial],
            y[trial],
            slds.A,
            slds.πₖ,
            state_of[trial],
            obs_of[trial],
            lds1.obs_model,
            ux_seq[trial],
            _trial(uy_seq, trial),
        )
    end

    return z, x, y
end

function Random.rand(slds::SLDS, tsteps::Integer; kwargs...)
    return rand(Random.default_rng(), slds, tsteps; kwargs...)
end

function Random.rand(slds::SLDS, tsteps_per_trial::AbstractVector{<:Integer}; kwargs...)
    return rand(Random.default_rng(), slds, tsteps_per_trial; kwargs...)
end

# Core SLDS trial sampling logic. `ux_trial` / `uy_trial` are the canonicalized
function _sample_slds_trial!(
    rng,
    z_trial,
    x_trial,
    y_trial,
    A,
    πₖ,
    state_params,
    obs_params,
    obs_model_type,
    ux_trial::AbstractMatrix,
    uy_trial,
)
    tsteps = length(z_trial)
    K = size(A, 1)

    # Sample discrete state sequence using forward sampling
    z_trial[1] = rand(rng, Categorical(πₖ))
    for t in 2:tsteps
        z_trial[t] = rand(rng, Categorical(A[z_trial[t - 1], :]))
    end

    # Sample continuous states and observations given discrete sequence
    return _sample_continuous_given_discrete!(
        rng,
        x_trial,
        y_trial,
        z_trial,
        state_params,
        obs_params,
        obs_model_type,
        ux_trial,
        uy_trial,
    )
end

# Sample continuous dynamics given discrete state sequence
function _sample_continuous_given_discrete!(
    rng,
    x_trial,
    y_trial,
    z_trial,
    state_params,
    obs_params,
    obs_model_type::GaussianObservationModel,
    ux_trial::AbstractMatrix,
    uy_trial,
)
    tsteps = length(z_trial)

    # Initial state from the selected LDS
    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    y_trial[:, 1] = rand(
        rng,
        MvNormal(
            obs_params[k1].C * x_trial[:, 1] +
            obs_params[k1].d +
            obs_params[k1].D * uy_trial[:, 1],
            obs_params[k1].R,
        ),
    )

    # Subsequent states - switch dynamics based on discrete state
    for t in 2:tsteps
        k_curr = z_trial[t]

        # Continuous state follows the current discrete state's dynamics
        # (x_t | x_{t-1}, z_t=k ~ N(A_k x_{t-1} + b_k + B_k u_{t-1}, Q_k),
        # matching `hessian!`)
        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params[k_curr].A * x_trial[:, t - 1] +
                state_params[k_curr].b +
                state_params[k_curr].B * ux_trial[:, t - 1],
                state_params[k_curr].Q,
            ),
        )

        # Observation follows current discrete state's model
        y_trial[:, t] = rand(
            rng,
            MvNormal(
                obs_params[k_curr].C * x_trial[:, t] +
                obs_params[k_curr].d +
                obs_params[k_curr].D * uy_trial[:, t],
                obs_params[k_curr].R,
            ),
        )
    end
end

function _sample_continuous_given_discrete!(
    rng,
    x_trial,
    y_trial,
    z_trial,
    state_params,
    obs_params,
    obs_model_type::PoissonObservationModel,
    ux_trial::AbstractMatrix,
    uy_trial,
)
    tsteps = length(z_trial)

    # Initial state
    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    y_trial[:, 1] =
        rand.(
            rng,
            Poisson.(
                exp.(
                    obs_params[k1].C * x_trial[:, 1] +
                    obs_params[k1].d +
                    obs_params[k1].D * uy_trial[:, 1],
                ),
            ),
        )

    # Subsequent states
    for t in 2:tsteps
        k_curr = z_trial[t]

        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params[k_curr].A * x_trial[:, t - 1] +
                state_params[k_curr].b +
                state_params[k_curr].B * ux_trial[:, t - 1],
                state_params[k_curr].Q,
            ),
        )

        y_trial[:, t] =
            rand.(
                rng,
                Poisson.(
                    exp.(
                        obs_params[k_curr].C * x_trial[:, t] +
                        obs_params[k_curr].d +
                        obs_params[k_curr].D * uy_trial[:, t],
                    ),
                ),
            )
    end
end

"""
    StatsAPI.fit!(dl::SLDSDiscreteLayer, fb_storage, obs_seq; seq_ends)

Update the discrete transition matrix `dl.A` and initial-state distribution `dl.πₖ`
in place from forward-backward statistics. Mirrors HiddenMarkovModels.jl's
`fit!(::HMM, ...)` pattern using the `ξ[t2]` scratch trick: for each sequence,
`ξ[t2]` is zero by FB convention so it doubles as an accumulator for `sum(ξ[t1:t2-1])`.

Skips fitting observation distributions because the SLDS discrete layer doesn't have
parametric obs distributions; per-state log-likelihoods are filled into `dl.logL`
upstream by the SLDS E-step.
"""
function StatsAPI.fit!(
    dl::SLDSDiscreteLayer{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector;
    seq_ends::AbstractVector{Int},
) where {T<:Real}
    γ = fb_storage.γ
    ξ = fb_storage.ξ

    # Accumulate ξ[t1:t2-1] into ξ[t2] (zero by FB convention) for each trial.
    tforeach(eachindex(seq_ends)) do k
        # `local`: `t1`/`t2` are also assigned in the sequential loops below,
        # so sharing the bindings would box them (OhMyThreads rejects that).
        local t1, t2
        t1, t2 = HMMs.seq_limits(seq_ends, k)
        scratch = ξ[t2]
        fill!(scratch, zero(eltype(scratch)))
        for t in t1:(t2 - 1)
            scratch .+= ξ[t]
        end
    end

    fill!(dl.πₖ, zero(eltype(dl.πₖ)))
    fill!(dl.A, zero(eltype(dl.A)))
    for k in eachindex(seq_ends)
        t1, t2 = HMMs.seq_limits(seq_ends, k)
        dl.πₖ .+= view(γ, :, t1)
        dl.A .+= ξ[t2]
    end

    dl.πₖ ./= sum(dl.πₖ)
    for i in axes(dl.A, 1)
        s = sum(view(dl.A, i, :))
        if s > zero(T)
            dl.A[i, :] ./= s
        end
    end

    return nothing
end

"""
    _slds_lognorm_for(slds, y)      -> Vector or nothing
    _slds_lognorm_all(slds, y)      -> Vector{Vector} or nothing

The Poisson `Σᵢ log(y!)` normalizer for one trial / for every trial, or
`nothing` when the emission is Gaussian. All regimes of an SLDS share an
observation-model type, so `LDSs[1]` decides which it is; and the normalizer
depends only on the counts, so one vector per trial serves every regime.
"""
function _slds_lognorm_for(slds::SLDS, y)
    return _poisson_lognorm_one(slds.LDSs[1], y)
end

function _slds_lognorm_all(slds::SLDS, y::AbstractVector{<:AbstractMatrix})
    return _poisson_lognorm_all(slds.LDSs[1], y)
end

# A composite emission's hoisted normalizers are per member, so the whole
# dataset's are per trial then per member.
function _slds_lognorm_all(slds::SLDS, y::NamedTuple)
    return [_slds_lognorm_for(slds, _trial(y, n)) for n in 1:_ntrials(y)]
end

#=
One regime's emission log-density over a whole trial, written into `out`.

Poisson: `y_t·η_t - Σ exp(η_t) - Σᵢ log(y_{i,t}!)` with `η = C x + d + D v`
formed for the trial in a single `gemm`, and the `log(y!)` normalizer supplied
precomputed — it is constant in the latents, so recomputing it inside the
Newton line search (which is what the generic per-timestep kernel does) is pure
waste. This is the same arithmetic `fit_PLDS.jl`'s `joint_loglikelihood!` uses;
the SLDS could not reach that method because it dispatches on `SmoothWorkspace`.

Generic: the per-timestep `observation_loglikelihood!` kernel, unchanged. A
Gaussian emission has no data-only normalizer, so `lognorm_t` is `nothing`.
=#
function _slds_emission_loglik!(
    out::AbstractVector{T},
    ws::SLDSSmoothWorkspace{T},
    ::SmoothConstants{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    uy::Union{Nothing,AbstractMatrix},
    tsteps::Int,
    lognorm_t::Union{Nothing,AbstractVector{T}},
    ::Union{Nothing,Vector{ObsScratch{T}}}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:PoissonObservationModel{T}}
    om = lds.obs_model
    obs_dim, latent_dim = size(om.C)
    pb = poisson_batch!(ws, latent_dim, obs_dim, tsteps)
    Eta = _poisson_linear_predictor!(pb, om.C, om.d, om.D, x, uy, tsteps)
    @inbounds @views for t in 1:tsteps
        η = Eta[:, t]
        norm_t = lognorm_t === nothing ? _poisson_lognorm_at(y, t) : lognorm_t[t]
        out[t] = dot(y[:, t], η) - sum(exp, η) - norm_t
    end
    return out
end

function _slds_emission_loglik!(
    out::AbstractVector{T},
    ws::SLDSSmoothWorkspace{T},
    cc::SmoothConstants{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    uy::Union{Nothing,AbstractMatrix},
    tsteps::Int,
    ::Union{Nothing,AbstractVector{T}},
    ::Union{Nothing,Vector{ObsScratch{T}}}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    z = ws.opt.temp_dy
    λ = ws.opt.temp_solve_R
    @inbounds for t in 1:tsteps
        out[t] = observation_loglikelihood!(cc, z, λ, lds.obs_model, x, y, t, uy)
    end
    return out
end

"""
    _slds_trial_loglikelihood!(ll, ws, cc, lds, x, y, ux, uy, lognorm_t)

One regime's per-timestep complete-data log-density for a whole trial. Splits
into the emission half (batched where the observation model allows it) and the
state half, which is the same per-timestep recursion for every model.

Equivalent to the generic `joint_loglikelihood!` in `continuous_latents.jl`; it
exists so the Poisson emission can take the batched path and be handed its
precomputed normalizer.
"""
function _slds_trial_loglikelihood!(
    ll::AbstractVector{T},
    ws::SLDSSmoothWorkspace{T},
    cc::SmoothConstants{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::Union{AbstractMatrix{T},NamedTuple},
    ux::Union{Nothing,AbstractMatrix}=nothing,
    uy::Union{Nothing,AbstractMatrix,NamedTuple}=nothing,
    lognorm_t::Union{Nothing,AbstractVector{T},NamedTuple}=nothing,
    obs_scratch::Union{Nothing,Vector{ObsScratch{T}}}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    tsteps = _ntsteps(y)
    @assert length(ll) == tsteps

    _slds_emission_loglik!(ll, ws, cc, lds, x, y, uy, tsteps, lognorm_t, obs_scratch)

    dx = ws.opt.temp_dx
    tmp = ws.opt.temp_solve_Q
    @inbounds for t in 1:tsteps
        ll[t] += state_loglikelihood!(cc, dx, tmp, lds, x, t, ux)
    end

    return ll
end

"""
    joint_loglikelihood!(ws, slds, x, y, w[, ux, uy, lognorm_t])

Compute weighted complete-data log-likelihood for SLDS.
Returns vector of per-timestep log-likelihoods. `ux` / `uy` are the per-trial
control-input matrices (`nothing` or zero-row skips the `Bₖ u` / `Dₖ v` terms).

`lognorm_t` is the trial's Poisson `Σᵢ log(y!)` normalizer (see
[`_poisson_lognorm_t`](@ref)), constant in the latents and so worth computing
once per trial rather than once per Newton line-search evaluation; `nothing`
recomputes it, which is only what the single-trial convenience entry points do.
Ignored by a Gaussian emission.
"""
function joint_loglikelihood!(
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    y::Union{AbstractMatrix{T},NamedTuple},
    w::AbstractMatrix{T},   # K × T responsibilities/weights
    ux::Union{Nothing,AbstractMatrix}=nothing,
    uy::Union{Nothing,AbstractMatrix,NamedTuple}=nothing,
    lognorm_t::Union{Nothing,AbstractVector{T},NamedTuple}=nothing,
) where {T<:Real}
    Tsteps = _ntsteps(y)

    # Workspace ll_vec may be sized for a longer trial; only touch the active prefix.
    ll_vec = ws.opt.ll_vec
    @views fill!(ll_vec[1:Tsteps], zero(T))

    K = length(slds.LDSs)
    for k in 1:K
        _slds_trial_loglikelihood!(
            view(ws.ll_tmp, 1:Tsteps),
            ws,
            ws.consts[k],
            slds.LDSs[k],
            x,
            y,
            ux,
            uy,
            lognorm_t,
            _regime_obs(ws, k),
        )
        for t in 1:Tsteps
            ll_vec[t] += w[k, t] * ws.ll_tmp[t]
        end
    end

    return view(ll_vec, 1:Tsteps)
end

"""
    _add_cov_correction!(ll, ws, cc, lds_k, x, y, fs[, uy])

Add the second-order term that turns a plug-in per-timestep log-likelihood into
`E_q(x)[log p_k(y_t, x_t | x_{t-1})]`, in place on `ll`.

For a factor Hessian `H^{(k,t)}` and smoothed covariance `Σ`, the correction is
`½ tr(H^{(k,t)} Σ)`. Summing it against `γ` reproduces the covariance term
in [`elbo!`](@ref). The expansion is exact for Gaussian emissions and
second-order for Poisson emissions.

Uses the same factor-at-`t` convention as `joint_loglikelihood!` and `hessian!`:
`ll[t]` covers the emission at `t` plus the dynamics factor coupling
`(x_{t-1}, x_t)`, or the prior at `t == 1`. The covariances in `fs` must
correspond to `x`.

Overwrites `ws.H_obs` and the emission scratch in `ws.opt`.
"""
function _add_cov_correction!(
    ll::AbstractVector{T},
    ws::SLDSSmoothWorkspace{T},
    cc::SmoothConstants{T},
    lds_k::LinearDynamicalSystem{T},
    x::AbstractMatrix{T},
    y::Union{AbstractMatrix{T},NamedTuple},
    fs::FilterSmooth{T},
    uy::Union{Nothing,AbstractMatrix,NamedTuple}=nothing,
    obs_scratch::Union{Nothing,Vector{ObsScratch{T}}}=nothing,
) where {T<:Real}
    Tsteps = _ntsteps(y)

    # Cached state-model templates for regime k, matching `hessian!`.
    neg_Q_inv = cc.xt_given_xt_1     # -Q⁻¹
    neg_AtQinvA = cc.xt1_given_xt    # -A'Q⁻¹A
    neg_P0_inv = cc.x_t              # -P0⁻¹
    sub_entry = cc.H_sub_entry       #  Q⁻¹A
    super_entry = cc.H_super_entry   # (Q⁻¹A)'

    H_obs = ws.H_obs
    # Poisson curvature uses both scratch vectors; Gaussian ignores them.
    z = ws.opt.dyt
    λ = ws.opt.temp_dy

    @views for t in 1:Tsteps
        Σ_tt = fs.p_smooth[:, :, t]

        # Use unit weight to get this regime's emission curvature alone.
        fill!(H_obs, zero(T))
        _emission_curvature_at!(H_obs, ws, cc, lds_k, x, y, t, uy, obs_scratch)
        corr = _tr_prod(H_obs, Σ_tt)

        if t == 1
            corr += _tr_prod(neg_P0_inv, Σ_tt)
        else
            # Sum both cross-covariance traces; do not assume exact block symmetry.
            Σ_ttm1 = fs.p_smooth_tt1[:, :, t]  # Cov(x_t, x_{t-1})
            corr += _tr_prod(neg_Q_inv, Σ_tt)
            corr += _tr_prod(neg_AtQinvA, fs.p_smooth[:, :, t - 1])
            corr += _tr_prod(super_entry, Σ_ttm1)
            corr += _tr_prod(sub_entry, transpose(Σ_ttm1))
        end

        t == Tsteps && (corr += _slds_terminal_cov_correction(lds_k, fs, Tsteps))

        ll[t] += T(0.5) * corr
    end

    return ll
end

"""
    gradient!(ws, slds, x, y, w[, ux, uy])

In-place SLDS gradient: each component's complete-data gradient is scaled
per-timestep by the responsibility `w[k, t]` and accumulated. Writes into
`ws.opt.grad_buf` and returns it. `ux` (dynamics input, feeds `-Q⁻¹` /
`A'Q⁻¹` residuals via `Bₖ u`) and `uy` (observation input, feeds the emission
gradient via `Dₖ v`) are per-trial matrices; `nothing` or zero-row skips them.
"""
#=
The soft terminal condition of an inverse-LQR state model is an extra Gaussian
factor at `t = T`, on top of the transition factor. `joint_loglikelihood!` picks
it up for free because it calls `state_loglikelihood!`, which is dispatched — but
`gradient!`, `hessian!` and `_add_cov_correction!` build their state
contributions from the flat `SmoothConstants` templates, which have no slot for
it. These three helpers add it, weighted by the regime's responsibility at `T`
exactly like every other factor, and no-op for a state model that has none.
=#
@inline function _slds_terminal_gradient!(
    grad::AbstractMatrix, ws::SLDSSmoothWorkspace, lds::LinearDynamicalSystem, x, w_k, ux
)
    return _slds_terminal_gradient!(grad, ws, lds.state_model, x, w_k, ux)
end

@inline function _slds_terminal_gradient!(
    ::AbstractMatrix, ::SLDSSmoothWorkspace, ::AbstractStateModel, _, _, _
)
    return nothing
end

function _slds_terminal_gradient!(
    grad::AbstractMatrix{T},
    ws::SLDSSmoothWorkspace{T},
    sm::HamiltonianStateModel{T},
    x::AbstractMatrix{T},
    w_k::AbstractVector{T},
    ux::Union{Nothing,AbstractMatrix},
) where {T<:Real}
    sm.terminal || return nothing
    tsteps = size(x, 2)
    n = _plant_dim(sm)
    rf = view(ws.opt.dxt, 1:n)
    _terminal_residual!(rf, sm, x, ux)
    # grad[:, T] -= w · Λfᵀ Σf⁻¹ r_f
    @views mul!(grad[:, tsteps], sm.cache.LtSinv, rf, -w_k[tsteps], one(T))
    return nothing
end

@inline function _slds_terminal_hessian!(
    H_diag::AbstractVector, lds::LinearDynamicalSystem, w_k, tsteps::Int
)
    return _slds_terminal_hessian!(H_diag, lds.state_model, w_k, tsteps)
end

@inline function _slds_terminal_hessian!(::AbstractVector, ::AbstractStateModel, _, ::Int)
    return nothing
end

function _slds_terminal_hessian!(
    H_diag::AbstractVector,
    sm::HamiltonianStateModel{T},
    w_k::AbstractVector{T},
    tsteps::Int,
) where {T<:Real}
    sm.terminal || return nothing
    negLtSL = sm.cache.negLtSL
    @. H_diag[tsteps] += w_k[tsteps] * negLtSL
    return nothing
end

@inline function _slds_terminal_cov_correction(
    lds::LinearDynamicalSystem, fs::FilterSmooth, tsteps::Int
)
    return _slds_terminal_cov_correction(lds.state_model, fs, tsteps)
end

@inline function _slds_terminal_cov_correction(
    sm::AbstractStateModel{T}, ::FilterSmooth, ::Int
) where {T}
    return zero(T)
end

function _slds_terminal_cov_correction(
    sm::HamiltonianStateModel{T}, fs::FilterSmooth{T}, tsteps::Int
) where {T<:Real}
    sm.terminal || return zero(T)
    # `tr(H_f Σ_T)`; the caller's single ½ applies to it like every other term.
    return _tr_prod(sm.cache.negLtSL, view(fs.p_smooth, :, :, tsteps))
end

function gradient!(
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    y::Union{AbstractMatrix{T},NamedTuple},
    w::AbstractMatrix{T},
    ux::Union{Nothing,AbstractMatrix}=nothing,
    uy::Union{Nothing,AbstractMatrix,NamedTuple}=nothing,
) where {T<:Real}
    latent_dim, Tsteps = size(x)
    K = length(slds.LDSs)

    grad = ws.opt.grad_buf
    fill!(grad, zero(T))

    dxt = ws.opt.dxt
    dxt_next = ws.opt.dxt_next
    obs_buf = ws.opt.dyt
    tmp1 = ws.opt.tmp1
    tmp2 = ws.opt.tmp2
    tmp3 = ws.opt.tmp3

    @views for k in 1:K
        lds_k = slds.LDSs[k]
        cc = ws.consts[k]

        x0 = lds_k.state_model.x0

        A_inv_Q = cc.A_inv_Q          # A'Q^{-1}
        neg_Q_inv = cc.xt_given_xt_1  # -Q^{-1}
        neg_P0_inv = cc.x_t           # -P0^{-1}

        # Emission half, for every timestep at once where the model allows it.
        _slds_emission_gradient!(
            grad,
            ws,
            cc,
            lds_k,
            x,
            y,
            view(w, k, :),
            uy,
            Tsteps,
            tmp1,
            obs_buf,
            _regime_obs(ws, k),
        )

        # t = 1: prior, weighted by w[k,1]
        @. dxt = x[:, 1] - x0
        mul!(tmp3, neg_P0_inv, dxt)
        @. grad[:, 1] += w[k, 1] * tmp3

        Tsteps == 1 && continue

        # Outgoing dynamics term comes from the factor at time 2, weighted by w[k,2]
        _transition_residual!(dxt_next, lds_k, x, 2, ux)
        mul!(tmp2, A_inv_Q, dxt_next)
        @. grad[:, 1] += w[k, 2] * tmp2

        # 2 .. T-1: incoming factor at t (w[k,t]), outgoing factor at t+1 (w[k,t+1])
        for t in 2:(Tsteps - 1)
            _transition_residual!(dxt, lds_k, x, t, ux)
            mul!(tmp3, neg_Q_inv, dxt)
            @. grad[:, t] += w[k, t] * tmp3

            _transition_residual!(dxt_next, lds_k, x, t + 1, ux)
            mul!(tmp2, A_inv_Q, dxt_next)
            @. grad[:, t] += w[k, t + 1] * tmp2
        end

        # t = T: incoming factor at T, weighted by w[k,T]
        _transition_residual!(dxt, lds_k, x, Tsteps, ux)
        mul!(tmp3, neg_Q_inv, dxt)
        @. grad[:, Tsteps] += w[k, Tsteps] * tmp3

        # `dxt` is free again here, which is what the helper borrows for `r_f`.
        _slds_terminal_gradient!(grad, ws, lds_k, x, view(w, k, :), ux)
    end

    return grad
end

#=
One regime's emission gradient `γₖ(t)·∂ log p(yₜ|xₜ)/∂xₜ`, accumulated into
`grad` for the whole trial.

Poisson: `γₖ(t)·C'(yₜ − λₜ)`. The linear predictor is one `gemm`, the weighted
residual overwrites it in place, and `C'` applied to the whole residual block is
a second `gemm` accumulating straight into `grad` — replacing `tsteps` BLAS-2
pairs per regime. Same trick as the batched emission Hessian, and like it, a
different summation order than the per-timestep kernel: results agree to
rounding, not bit-for-bit.

Generic: the per-timestep `observation_gradient!` kernel, unchanged.
=#
function _slds_emission_gradient!(
    grad::AbstractMatrix{T},
    ws::SLDSSmoothWorkspace{T},
    ::SmoothConstants{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    weights::AbstractVector{T},
    uy::Union{Nothing,AbstractMatrix},
    tsteps::Int,
    ::AbstractVector{T},
    ::AbstractVector{T},
    ::Union{Nothing,Vector{ObsScratch{T}}}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:PoissonObservationModel{T}}
    om = lds.obs_model
    obs_dim, latent_dim = size(om.C)
    pb = poisson_batch!(ws, latent_dim, obs_dim, tsteps)
    Eta = _poisson_linear_predictor!(pb, om.C, om.d, om.D, x, uy, tsteps)

    # Overwrite η in place with the weighted residual γₖ(t)·(yₜ − exp(ηₜ)).
    @inbounds for t in 1:tsteps
        wt = weights[t]
        col = view(Eta, :, t)
        yt = view(y, :, t)
        @simd for i in 1:obs_dim
            col[i] = wt * (yt[i] - exp(col[i]))
        end
    end

    mul!(view(grad, :, 1:tsteps), transpose(om.C), Eta, one(T), one(T))
    return nothing
end

function _slds_emission_gradient!(
    grad::AbstractMatrix{T},
    ::SLDSSmoothWorkspace{T},
    cc::SmoothConstants{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    weights::AbstractVector{T},
    uy::Union{Nothing,AbstractMatrix},
    tsteps::Int,
    tmp::AbstractVector{T},
    obs_buf::AbstractVector{T},
    ::Union{Nothing,Vector{ObsScratch{T}}}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    @inbounds for t in 1:tsteps
        observation_gradient!(tmp, cc, obs_buf, lds.obs_model, x, y, t, uy)
        α = weights[t]
        @simd for i in eachindex(tmp)
            grad[i, t] += α * tmp[i]
        end
    end
    return nothing
end

#=
One regime's emission curvature, added into the diagonal Hessian blocks with
its responsibilities `γₖ(t)` as the per-timestep weight.

Gaussian: `-γₖ(t) · C'R⁻¹C` from the regime's cached template, an `O(D²)` axpy
per timestep — the per-timestep kernel is already the cheap way to do it.

Poisson: `-γₖ(t) · C' diag(λₜ) C`, which is `O(N·D²)` per timestep and is where
a Poisson SLDS fit spends most of its time (`N` is the neuron count, and this
runs per Newton step, per trial, per E-step, for every regime). Routed to the
batched kernel, which forms the whole trial as one `gemm`.
=#
function _slds_emission_hessian!(
    ws::SLDSSmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    cc::SmoothConstants{T},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    weights::AbstractVector{T},
    uy::Union{Nothing,AbstractMatrix},
    tsteps::Int,
    z::AbstractVector{T},
    λ::AbstractVector{T},
    ::Union{Nothing,Vector{ObsScratch{T}}}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:PoissonObservationModel{T}}
    obs_dim, latent_dim = size(lds.obs_model.C)
    pb = poisson_batch!(ws, latent_dim, obs_dim, tsteps)
    _poisson_emission_hessian!(ws.btd, pb, lds.obs_model, x, uy, tsteps, weights)
    return nothing
end

function _slds_emission_hessian!(
    ws::SLDSSmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    cc::SmoothConstants{T},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
    weights::AbstractVector{T},
    uy::Union{Nothing,AbstractMatrix},
    tsteps::Int,
    z::AbstractVector{T},
    λ::AbstractVector{T},
    ::Union{Nothing,Vector{ObsScratch{T}}}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    H_diag = ws.btd.H_diag
    for t in 1:tsteps
        observation_hessian!(H_diag[t], cc, z, λ, lds.obs_model, x, y, t, weights[t], uy)
    end
    return nothing
end

"""
    hessian!(ws, slds, x, y, w)

Fill `ws.btd.H_diag`, `ws.btd.H_sub`, `ws.btd.H_super` with the weighted Hessian blocks
for the Laplace/Newton step over `x₁:T` matching Zoltowski et al. Appendix B.

Convention matched:
    x_t | x_{t-1}, z_t=k ~ N(A_k x_{t-1} + b_k, Q_k)
so the dynamics factor that couples (x_{t-1}, x_t) is weighted by w[k,t] = q(z_t=k).

Weights:
- emission curvature at time t uses w[k,t]
- dynamics curvature from factor at time t uses w[k,t]
- off-diagonal block coupling (t-1,t) uses w[k,t]

`uy` (optional observation input) is forwarded to `observation_hessian!`: the
Gaussian curvature ignores it, the Poisson curvature depends on it through the
rate `λ = exp(Cx + d + Dₖ v)`. The state-side blocks never depend on inputs.
"""
function hessian!(
    ws::SLDSSmoothWorkspace{T},
    slds::SLDS{T},
    x::AbstractMatrix{T},
    y::Union{AbstractMatrix{T},NamedTuple},
    w::AbstractMatrix{T},
    uy::Union{Nothing,AbstractMatrix,NamedTuple}=nothing,
) where {T<:Real}
    Tsteps = size(x, 2)
    K = length(slds.LDSs)

    H_diag = ws.btd.H_diag
    H_sub = ws.btd.H_sub
    H_super = ws.btd.H_super

    for t in 1:Tsteps
        fill!(H_diag[t], zero(T))
    end
    for t in 1:(Tsteps - 1)
        fill!(H_sub[t], zero(T))
        fill!(H_super[t], zero(T))
    end

    # Two obs_dim scratch vectors for observation_hessian! (Poisson writes the
    # linear predictor and rate into them; Gaussian ignores both).
    z = ws.opt.dyt
    λ = ws.opt.temp_dy

    @views for k in 1:K
        lds_k = slds.LDSs[k]
        cc = ws.consts[k]

        # Cached state-model templates for regime k
        neg_Q_inv = cc.xt_given_xt_1    # -Q^{-1}
        neg_AtQinvA = cc.xt1_given_xt     # -A'Q^{-1}A
        neg_P0_inv = cc.x_t              # -P0^{-1}
        sub_entry = cc.H_sub_entry      #  Q^{-1}A
        super_entry = cc.H_super_entry    # (Q^{-1}A)'

        if Tsteps == 1
            @. H_diag[1] += w[k, 1] * neg_P0_inv
            _slds_terminal_hessian!(H_diag, lds_k, view(w, k, :), Tsteps)
            _slds_emission_hessian!(
                ws, lds_k, cc, x, y, view(w, k, :), uy, Tsteps, z, λ, _regime_obs(ws, k)
            )
            continue
        end

        # Dynamics factor at time t couples (x_{t-1}, x_t), weighted by w[k,t].
        # Off-diagonal blocks between t-1 and t therefore use w[k,t].
        for t in 2:Tsteps
            α = w[k, t]
            @. H_sub[t - 1] += α * sub_entry
            @. H_super[t - 1] += α * super_entry
        end

        # Diagonal state-model contributions:
        # - At t=1: prior term weighted by w[k,1], plus "previous-role" from factor at t=2 weighted by w[k,2]
        @. H_diag[1] += w[k, 1] * neg_P0_inv
        @. H_diag[1] += w[k, 2] * neg_AtQinvA

        # - For 2..T-1: current-role from factor at t (neg_Q_inv) weighted by w[k,t]
        #               previous-role from factor at t+1 (neg_AtQinvA) weighted by w[k,t+1]
        for t in 2:(Tsteps - 1)
            @. H_diag[t] += w[k, t] * neg_Q_inv
            @. H_diag[t] += w[k, t + 1] * neg_AtQinvA
        end

        # - At t=T: current-role from factor at T weighted by w[k,T]
        @. H_diag[Tsteps] += w[k, Tsteps] * neg_Q_inv
        _slds_terminal_hessian!(H_diag, lds_k, view(w, k, :), Tsteps)

        # Emission curvature contributions, weighted by w[k,t].
        _slds_emission_hessian!(
            ws, lds_k, cc, x, y, view(w, k, :), uy, Tsteps, z, λ, _regime_obs(ws, k)
        )
    end

    for t in 1:Tsteps
        Symmetrize!(H_diag[t])
    end

    return nothing
end

function smooth!(
    slds::SLDS{T},
    fs::FilterSmooth{T},
    y::Union{AbstractMatrix{T},NamedTuple},
    w::AbstractMatrix{T};
    ws::Union{Nothing,SLDSSmoothWorkspace{T}}=nothing,
    max_iter::Int=20,
    tol::T=T(1e-6),
    linesearch::Union{Nothing,AbstractLineSearch}=BackTrackingLS{T}(),
    x_sample::Union{Nothing,AbstractMatrix{T}}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    noise::Union{Nothing,AbstractVector{T}}=nothing,
    ux::Union{Nothing,AbstractMatrix{T}}=nothing,
    uy::Union{Nothing,AbstractMatrix{T},NamedTuple}=nothing,
    lognorm_t::Union{Nothing,AbstractVector{T},NamedTuple}=nothing,
) where {T<:Real}
    latent_dim = slds.LDSs[1].latent_dim
    tsteps = _ntsteps(y)
    n_active = latent_dim * tsteps

    ws === nothing && (ws = SLDSSmoothWorkspace(T, slds, tsteps))
    btd = ws.btd

    #=
    The Poisson `-Σ log(y!)` normalizer is constant in `x`, so the Newton line
    search must not recompute it. Callers that smooth many trials build it once
    each at fit entry; the single-trial entry points land here with `nothing`
    and pay for it once, not once per line-search evaluation.
    =#
    ln = lognorm_t === nothing ? _slds_lognorm_for(slds, y) : lognorm_t

    x = fs.x_smooth

    #=
    Warm-start the Newton iteration from the previous EM iteration's smoothed
    mean. If the smoothed mean is all zeros, use the first LDS's prior mean.
    =#
    if all(x .== 0)
        x .= slds.LDSs[1].state_model.x0
    end

    # Active-length views into (possibly) oversized workspace buffers.
    g = view(ws.opt.grad_buf, :, 1:tsteps)
    p = reshape(view(ws.opt.X0, 1:n_active), latent_dim, tsteps)
    neg_diag_v = view(btd.neg_diag, 1:tsteps)
    neg_sub_v = view(btd.neg_sub, 1:(tsteps - 1))
    neg_super_v = view(btd.neg_super, 1:(tsteps - 1))

    ϕ!() = begin
        ll = joint_loglikelihood!(ws, slds, x, y, w, ux, uy, ln)
        return sum(ll)
    end

    compute_grad! = (gcur, xcur) -> begin
        gradient!(ws, slds, xcur, y, w, ux, uy)
        copyto!(gcur, view(ws.opt.grad_buf, :, 1:tsteps))
        return nothing
    end

    build_hess! = (xcur) -> begin
        hessian!(ws, slds, xcur, y, w, uy)
        _negate_blocks!(btd, tsteps)
        return nothing
    end

    solve_dir! =
        (pcur, gcur) -> begin
            gvec = vec(gcur)
            pvec = vec(pcur)
            copyto!(pvec, gvec)
            # SPD path (negated Hessian at MAP).
            block_tridiagonal_solve_spd!(
                pvec, neg_sub_v, neg_diag_v, neg_super_v, gvec, btd
            )
            return nothing
        end

    newton_smooth!(
        Val(:max),
        x,
        g,
        p,
        compute_grad!,
        build_hess!,
        solve_dir!,
        ϕ!,
        linesearch;
        max_iter=max_iter,
        tol=tol,
    )

    # Posterior covariances at the MAP via Laplace approx.
    hessian!(ws, slds, x, y, w, uy)
    _negate_blocks!(btd, tsteps)

    logdet_precision = block_tridiagonal_inverse_logdet!(
        fs.p_smooth, fs.p_smooth_tt1, neg_sub_v, neg_diag_v, neg_super_v, btd
    )

    fs.entropy = gaussian_entropy_from_logdet(logdet_precision, n_active)

    #=
    Optional joint draw from q(x), while `btd` still holds the precision factors.
    `ws.opt.X0` is free after Newton; reuse it for the standard-normal input.
    =#
    if x_sample !== nothing
        z = view(ws.opt.X0, 1:n_active)
        #=
        `noise` lets the caller supply the standard normals instead of drawing
        them here. A multi-trial pass runs in parallel, so a shared `rng` would
        be both a data race and schedule-dependent; the caller either pre-draws
        the stream serially (preserving global-RNG semantics) or hands each
        trial its own generator.
        =#
        if noise === nothing
            randn!(rng, z)
        else
            copyto!(z, noise)
        end
        block_tridiagonal_sample!(z, btd, tsteps)
        @views x_sample .= fs.x_smooth .+ reshape(z, latent_dim, tsteps)
    end

    @views for t in 1:tsteps
        fs.p_smooth[:, :, t] .= Symmetrize!(fs.p_smooth[:, :, t])
    end

    return fs
end

"""
    smooth(slds, y, w; ux=nothing, uy=nothing)

Smooth a single trial under **given** discrete responsibilities `w` (`K × T`), without
inferring them. Returns `(x_smooth, p_smooth)`. To infer the responsibilities as well,
call [`smooth(slds, y)`](@ref) with no `w`.
"""
function smooth(
    slds::SLDS,
    y::Union{AbstractMatrix{T},NamedTuple},
    w::AbstractMatrix{T};
    ux::Union{Nothing,AbstractMatrix{T}}=nothing,
    uy::Union{Nothing,AbstractMatrix{T},NamedTuple}=nothing,
) where {T<:Real}
    lds1 = slds.LDSs[1]
    tsteps = _ntsteps(y)
    ux_m = _check_ux(ux, lds1.ux_dim, tsteps, "ux", T)
    uy_m = _check_uy(uy, lds1.uy_dim, tsteps, lds1.obs_model)
    fs = initialize_FilterSmooth(lds1, tsteps)::FilterSmooth{T}
    smooth!(slds, fs, y, w; ux=ux_m, uy=uy_m)
    return fs.x_smooth, fs.p_smooth
end

"""
    smooth(slds, y; ux=nothing, uy=nothing, smoothing_iters=100, tol=1e-6,
           return_cov=false, progress=false, depends_on=nothing)

Infer the joint posterior of a **fitted** `SLDS` with the parameters held fixed: the
continuous states `q(x)`, the discrete responsibilities `γₜ(k) = q(zₜ = k)`, and the
ELBO at those posteriors.

Alternates forward-backward over the switching chain with the Laplace/Kalman smoother
over the continuous states (Ghahramani & Hinton, 1996). Unlike the single-sample E-step
`fit!` runs during learning, the coupling here is deterministic — the discrete layer is
scored at the smoothed posterior mean, not at a draw from `q(x)` — so the result is
reproducible.

That plug-in is what makes `γ` reproducible, and it is what limits it: the mean is a
shrunk version of the latent path, so regimes that differ *only* in their dynamics
(`Aₖ`/`Qₖ`, with the emission shared) become hard to separate once the observation noise
is large relative to the process noise. Regimes that differ in their emissions, or whose
latent path the data pins down, are recovered sharply.

`fit!` runs the same alternation but keeps its forward-backward storage private, so this
is the way to read `q(z)` — a regime occupancy over time, a Viterbi-style `argmax` path,
a rate averaged over regimes — out of a model, on the data it was fitted to or on
held-out data.

`y` takes the same three shapes as [`fit!`](@ref), and `ux` / `uy` the same shape family.

# Keywords
- `smoothing_iters::Int=100`: maximum discrete↔continuous alternations.
- `tol::Real=1e-6`: stop once `max|Δγ| < tol`; `tol=0` runs exactly `smoothing_iters`
  alternations with no stopping test.
- `return_cov::Bool=false`: also return the smoothed covariances (`latent_dim² × T` per
  trial — large, hence opt-in).
- `progress::Bool=false`: show a progress bar.
- `npool::Int`: task slots for the trial-parallel passes, one per thread by default
  (capped at the trial count). Each slot costs `O(D²·T)` block-tridiagonal storage plus
  `O(N·T)` Poisson scratch; lower it to trade throughput for memory. `npool=1` runs
  every pass sequentially.
- `depends_on`: optional `NamedTuple` of per-trial label vectors overriding the
  `depends_on` declared on the regimes for this call. A held-out set has its own trial
  count, so it needs its own label vectors.

# Returns
A `NamedTuple` `(; x, γ, elbo, trial_elbo, p)`. For a single-trial matrix `y`, `x` is
`latent_dim × T`, `γ` is `K × T`, and `p` is `latent_dim × latent_dim × T`; for a 3-D
array or a vector of matrices, each is a `Vector` with one entry per trial. `elbo` is
always a scalar, and `p` is `nothing` unless `return_cov=true`.

`trial_elbo` is each trial's contribution to `elbo`, always a `Vector` with one entry
per trial (a single-trial `y` included). It sums to `elbo` up to the parameter
log-prior, which belongs to no trial — see [`trial_elbos`](@ref).

Because a converged alternation is expensive, `smooth` returns everything it computed in
one call — read its `elbo` / `trial_elbo` fields rather than calling [`elbo`](@ref) or
[`trial_elbos`](@ref) separately.
"""
function smooth(
    slds::SLDS{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    ux=nothing,
    uy=nothing,
    smoothing_iters::Int=100,
    tol::Real=1e-6,
    return_cov::Bool=false,
    progress::Bool=false,
    npool::Int=Threads.maxthreadid(),
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    #=
    Same setup as `fit!`, minus the M-step workspaces: `Data` validates and
    canonicalizes the observation / input shapes, the grouping resolves
    `depends_on` into per-cell parameter views, and the discrete layer wraps the
    (K × ΣT) log-likelihood matrix the forward-backward pass reads.
    =#
    data = Data(slds.LDSs[1], y; ux=ux, uy=uy)
    _prepare_slds!(slds, data.tsteps)
    y_seq = data.y
    ux_seq = data.ux
    uy_seq = data.uy

    K = length(slds.LDSs)
    tsteps_per_trial = data.tsteps
    ntrials = length(tsteps_per_trial)
    seq_ends = cumsum(tsteps_per_trial)
    total_T = last(seq_ends)
    T_max = maximum(tsteps_per_trial)

    grp = _slds_parameter_grouping(slds, ntrials; depends_on=depends_on, y=y_seq)
    cell_slds = grp === nothing ? nothing : _slds_cell_sldss(slds, grp)

    tfs = initialize_FilterSmooth(slds.LDSs[1], tsteps_per_trial)::TrialFilterSmooth{T}
    dl = SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(T, K, total_T))
    fb_storage = _make_slds_fb_storage(dl, seq_ends)
    obs_seq = collect(1:total_T)
    control_seq = fill(nothing, total_T)
    pool = _slds_workspace_pool(slds, cell_slds, T_max, ntrials; npool=npool)
    plan = _slds_trial_plan(grp, ntrials, length(pool.slots))
    lognorm = _slds_lognorm_all(slds, y_seq)

    #=
    No M-step runs, so the per-regime constants cached by the workspace stay
    valid for every alternation. `x_samples === nothing` throughout: the discrete
    layer is scored at the smoothed mean, so no draw from q(x) is ever needed.
    =#
    _slds_warmstart!(
        slds,
        cell_slds,
        grp,
        tfs,
        y_seq,
        nothing,
        pool,
        plan,
        tsteps_per_trial,
        K;
        ux=ux_seq,
        uy=uy_seq,
        lognorm=lognorm,
    )

    prog = if progress
        Progress(smoothing_iters; desc="Smoothing SLDS...", barlen=50, showspeed=true)
    else
        nothing
    end

    _, converged = _vem_alternate!(
        slds,
        cell_slds,
        grp,
        tfs,
        fb_storage,
        dl,
        y_seq,
        pool,
        plan;
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
        ux=ux_seq,
        uy=uy_seq,
        lognorm=lognorm,
        smoothing_iters=smoothing_iters,
        tol=T(tol),
        prog=prog,
    )
    prog !== nothing && finish!(prog)

    if tol > 0 && !converged
        @warn "SLDS smoothing did not converge" smoothing_iters tol
    end

    #=
    Taken per trial and summed here, rather than through `elbo!` /
    `_elbo_grouped!`, which return only the total. Both of those route through
    `_slds_trial_elbos` themselves, so the total is the same number it always
    was; keeping the vector is what lets `trial_elbo` come back alongside it for
    nothing extra.
    =#
    trial_elbo, prior_logdensity = if grp === nothing
        _slds_trial_elbos(
            slds,
            nothing,
            nothing,
            tfs,
            fb_storage,
            y_seq,
            pool,
            plan;
            seq_ends=seq_ends,
            ux=ux_seq,
            uy=uy_seq,
            lognorm=lognorm,
        ),
        _slds_prior_logdensity(slds)
    else
        _slds_trial_elbos(
            (cell_slds::Vector)[1],
            cell_slds::Vector,
            grp::ParameterGrouping,
            tfs,
            fb_storage,
            y_seq,
            pool,
            plan;
            seq_ends=seq_ends,
            ux=ux_seq,
            uy=uy_seq,
            lognorm=lognorm,
        ),
        _grouped_slds_prior_logdensity(cell_slds::Vector, grp::ParameterGrouping, T)
    end
    total_elbo = sum(trial_elbo) + prior_logdensity

    γ_trials = Vector{Matrix{T}}(undef, ntrials)
    x_trials = Vector{Matrix{T}}(undef, ntrials)
    p_trials = return_cov ? Vector{Array{T,3}}(undef, ntrials) : nothing
    for trial in 1:ntrials
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        γ_trials[trial] = Matrix{T}(view(fb_storage.γ, :, t1:t2))
        x_trials[trial] = copy(tfs[trial].x_smooth)
        return_cov && (p_trials[trial] = copy(tfs[trial].p_smooth))
    end

    return _collect_slds_smooth_output(
        x_trials, γ_trials, p_trials, total_elbo, trial_elbo, y
    )
end

#=
Public-shape return convention, mirroring `_collect_smooth_output` in fit_LDS.jl:
matrix in → per-trial arrays out (single trial); vector / 3-D array in → vectors out.
`y` is only inspected for its container type.

`trial_elbo` is the exception: it stays a `Vector` even for a single-trial
matrix. Its whole point is to be indexed by trial, and unwrapped to a scalar it
would be indistinguishable from `elbo` — which for one trial and no priors is
the same number.
=#
function _collect_slds_smooth_output(x, γ, p, total_elbo, trial_elbo, ::AbstractMatrix)
    return (;
        x=x[1],
        γ=γ[1],
        elbo=total_elbo,
        trial_elbo=trial_elbo,
        p=(p === nothing ? nothing : p[1]),
    )
end

function _collect_slds_smooth_output(x, γ, p, total_elbo, trial_elbo, _)
    return (; x=x, γ=γ, elbo=total_elbo, trial_elbo=trial_elbo, p=p)
end

# ============================================================================
# Trial-parallel execution: workspace pool + work partition
#
# Every per-trial pass in this file used to thread one `SLDSSmoothWorkspace`
# through a sequential loop, which is what kept an SLDS fit on a single thread
# while `fit_LDS.jl` / `fit_PLDS.jl` chunked their trials across a pool. The
# pool below is the SLDS counterpart: `ntasks == length(slots)` chunks, each
# chunk owning one slot for the whole pass, addressed by chunk position rather
# than `threadid()` (see https://julialang.org/blog/2023/07/PSA-dont-use-threadid/).
# ============================================================================

"""
    SLDSWorkspacePool{T}

One `SLDSSmoothWorkspace` per task slot, plus the per-cell workspaces a
stitching fit needs.

`uniform` records that every cell has the parent's `obs_dim` — which is every
fit that is not stitching sessions of differing width — and then a slot's own
workspace serves all its cells and nothing extra is allocated.

Otherwise `cells[slot][c]` holds the workspace slot `slot` uses for cell `c`,
built on first use. Chunks are contiguous in cell-major trial order, so a slot
touches only the one or two cells its chunk spans: the lazy build keeps the
allocation at roughly `ntasks` cell workspaces rather than `ntasks × ncells`.
"""
struct SLDSWorkspacePool{T<:Real}
    slots::Vector{SLDSSmoothWorkspace{T}}
    cells::Vector{Vector{Union{Nothing,SLDSSmoothWorkspace{T}}}}
    uniform::Bool
    tsteps::Int
end

"""
    _slds_workspace_pool(slds, cell_slds, T_max, ntrials; npool) -> SLDSWorkspacePool

Build the pool. `npool` defaults to one slot per thread, capped at the trial
count — a fit with fewer trials than threads cannot use the extra slots, and
each slot costs `O(D²·T)` block-tridiagonal storage plus `O(N·T)` Poisson
scratch, so allocating them would be pure waste.
"""
function _slds_workspace_pool(
    slds::SLDS{T},
    cell_slds::Union{Nothing,AbstractVector},
    T_max::Int,
    ntrials::Int;
    npool::Int=Threads.maxthreadid(),
) where {T<:Real}
    # A fit with fewer trials than threads cannot use the extra slots.
    n = max(1, min(npool, max(ntrials, 1)))
    slots = [SLDSSmoothWorkspace(T, slds, T_max) for _ in 1:n]

    if cell_slds === nothing
        return SLDSWorkspacePool{T}(
            slots, Vector{Vector{Union{Nothing,SLDSSmoothWorkspace{T}}}}(), true, T_max
        )
    end

    p0 = slds.LDSs[1].obs_dim
    uniform = all(sc -> sc.LDSs[1].obs_dim == p0, cell_slds)
    ncells = length(cell_slds)
    cells = [
        Vector{Union{Nothing,SLDSSmoothWorkspace{T}}}(nothing, uniform ? 0 : ncells) for
        _ in 1:n
    ]
    return SLDSWorkspacePool{T}(slots, cells, uniform, T_max)
end

"""
    _slds_solo_pool(ws) -> SLDSWorkspacePool

Wrap one caller-supplied workspace as a single-slot pool. Backs the
single-workspace entry points, which run every trial on that one workspace.
"""
function _slds_solo_pool(ws::SLDSSmoothWorkspace{T}) where {T<:Real}
    return SLDSWorkspacePool{T}(
        [ws],
        Vector{Vector{Union{Nothing,SLDSSmoothWorkspace{T}}}}(),
        true,
        length(ws.opt.ll_vec),
    )
end

"""
    _slds_pool_ws(pool, slot, cell, cell_slds) -> SLDSSmoothWorkspace

The workspace slot `slot` should use for `cell`. Allocates it on first use for
a ragged-width stitching fit; otherwise hands back the slot's own workspace.
"""
function _slds_pool_ws(
    pool::SLDSWorkspacePool{T},
    slot::Int,
    cell::Int,
    cell_slds::Union{Nothing,AbstractVector},
) where {T<:Real}
    (pool.uniform || cell_slds === nothing) && return pool.slots[slot]
    existing = pool.cells[slot][cell]
    existing !== nothing && return existing::SLDSSmoothWorkspace{T}
    fresh = _cell_slds_workspace(pool.slots[slot], cell_slds[cell], pool.tsteps)
    pool.cells[slot][cell] = fresh
    return fresh
end

"""
    refresh_slds_pool!(pool, slds)

Refresh the cached regime constants on **every** slot after an M-step. The
ungrouped passes read constants without refreshing them, so missing a slot here
would have that slot smooth against the previous iteration's parameters.
A grouped pass refreshes per cell as it goes and does not need this.
"""
function refresh_slds_pool!(pool::SLDSWorkspacePool, slds::SLDS)
    for ws in pool.slots
        refresh_slds_constants!(ws, slds)
    end
    return nothing
end

"""
    SLDSTrialPlan

The fixed partition of trials into `ntasks` chunks that every parallel pass
reuses.

`order` lists the trials in visit order — cell-major when grouped, so a chunk
walks whole cells and the regime constants are refreshed once per cell it
spans, exactly as the sequential cell loop did. `cell_of` is the matching cell
index. `bounds` holds `ntasks + 1` offsets into `order`.

Fixed for the whole fit, so the slot a trial lands on never changes: the
per-slot lazy cell workspaces are built once, and the reduction order of any
chunked accumulation is deterministic.
"""
struct SLDSTrialPlan
    order::Vector{Int}
    cell_of::Vector{Int}
    bounds::Vector{Int}
end

_plan_ntasks(plan::SLDSTrialPlan) = length(plan.bounds) - 1

function _slds_trial_plan(grp::Union{Nothing,ParameterGrouping}, ntrials::Int, npool::Int)
    if grp === nothing
        order = collect(1:ntrials)
        cell_of = ones(Int, ntrials)
    else
        order = Int[]
        cell_of = Int[]
        sizehint!(order, ntrials)
        sizehint!(cell_of, ntrials)
        for c in 1:(grp.ncells), trial in grp.cell_trials[c]
            push!(order, trial)
            push!(cell_of, c)
        end
    end

    n = length(order)
    tasks = max(1, min(npool, max(n, 1)))
    chunk = cld(max(n, 1), tasks)
    bounds = Vector{Int}(undef, tasks + 1)
    for i in 1:tasks
        bounds[i] = min((i - 1) * chunk + 1, n + 1)
    end
    bounds[tasks + 1] = n + 1
    return SLDSTrialPlan(order, cell_of, bounds)
end

"""
    _slds_fill_logL!(slds, cell_slds, grp, dl, y, x_of, pool, plan; seq_ends, ux, uy, lognorm, tfs)

Fill `dl.logL` (`K × sum(T_i)`) with every regime's log-density of the current
continuous trajectory. `x_of(trial)` supplies that trajectory: the smoothed mean
for deterministic inference, a joint draw from `q(x)` for the Monte-Carlo E-step.

The discrete update wants `E_q(x)[log p_k(y_t, x_t | x_{t-1})]`, and the two
trajectories reach it differently. A draw from `q(x)` carries the posterior
spread already, so its plug-in score is unbiased for that expectation. The
smoothed mean does not: scoring at `E_q[x]` drops the curvature term and biases
every regime's log-density by its own `½ tr(H^{(k,t)} Σ)`. Pass `tfs` on that
path — the trial's `p_smooth` / `p_smooth_tt1` must match the `x` that `x_of`
returns — and [`_add_cov_correction!`](@ref) puts the term back, per regime and
per timestep, before forward-backward normalizes across `k`. `tfs === nothing`
is the sampled path and skips the correction.

Chunks run in parallel over `pool`. Each trial writes only `dl.logL[:, t1:t2]`,
and the chunks are disjoint in trials, so the result is identical to the
sequential pass — this is a pure partition of the writes, not a reduction.

When `grp` is set, trials are visited cell by cell within a chunk so the regime
constants are refreshed once per cell the chunk spans.
"""
function _slds_fill_logL!(
    slds::SLDS{T},
    cell_slds::Union{Nothing,AbstractVector},
    grp::Union{Nothing,ParameterGrouping},
    dl::SLDSDiscreteLayer{T},
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple},
    x_of,
    pool::SLDSWorkspacePool{T},
    plan::SLDSTrialPlan;
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}},NamedTuple}=nothing,
    lognorm::Union{Nothing,AbstractVector}=nothing,
    tfs::Union{Nothing,TrialFilterSmooth{T}}=nothing,
) where {T<:Real}
    K = length(slds.LDSs)
    grouped = grp !== nothing && cell_slds !== nothing

    tforeach(1:_plan_ntasks(plan)) do slot
        #=
        `local`: these names are also bound in the sequential helpers of this
        file, and sharing a binding with the enclosing scope boxes it, which
        OhMyThreads rejects.
        =#
        local lo, hi, cur_cell, ws_t, slds_t
        lo = plan.bounds[slot]
        hi = plan.bounds[slot + 1] - 1
        lo > hi && return nothing

        cur_cell = 0
        ws_t = pool.slots[slot]
        slds_t = slds

        for idx in lo:hi
            trial = plan.order[idx]
            if grouped && plan.cell_of[idx] != cur_cell
                cur_cell = plan.cell_of[idx]
                slds_t = cell_slds[cur_cell]
                ws_t = _slds_pool_ws(pool, slot, cur_cell, cell_slds)
                refresh_slds_constants!(ws_t, slds_t)
            end

            t1, t2 = HMMs.seq_limits(seq_ends, trial)
            x_src = x_of(trial)
            y_trial = _trial(y, trial)
            ux_trial = ux === nothing ? nothing : ux[trial]
            uy_trial = uy === nothing ? nothing : _trial(uy, trial)
            ln_trial = lognorm === nothing ? nothing : lognorm[trial]
            for k in 1:K
                ll_k = view(dl.logL, k, t1:t2)::AbstractVector{T}
                _slds_trial_loglikelihood!(
                    ll_k,
                    ws_t,
                    ws_t.consts[k],
                    slds_t.LDSs[k],
                    x_src,
                    y_trial,
                    ux_trial,
                    uy_trial,
                    ln_trial,
                    _regime_obs(ws_t, k),
                )
                if tfs !== nothing
                    _add_cov_correction!(
                        ll_k,
                        ws_t,
                        ws_t.consts[k],
                        slds_t.LDSs[k],
                        x_src,
                        y_trial,
                        tfs[trial],
                        uy_trial,
                        _regime_obs(ws_t, k),
                    )
                end
            end
        end
        return nothing
    end
    return nothing
end

"""
    _slds_smooth_all!(slds, cell_slds, grp, tfs, y, x_samples, pool, plan, w_of; ...)

Run the Laplace/Newton smoother over every trial under the discrete weights
`w_of(trial)` (`K × T_i`), filling `tfs[*].x_smooth`, `tfs[*].p_smooth`, and
`tfs[*].entropy`. `x_samples === nothing` skips the joint draw (the
deterministic path); otherwise the next draw from `q(x)` lands in
`x_samples[trial]`.

Chunks run in parallel over `pool`. A trial writes only its own `tfs[trial]`
and `x_samples[trial]`, so the smoothed output does not depend on how the
chunks were scheduled.

`noise_of(trial)` returns the standard-normal vector the trial's joint draw
should consume, or `nothing` to have the trial draw its own from
`rng_of(trial)`. Between them these are the two reproducibility modes: a
pre-drawn stream keeps the global-RNG semantics under parallelism, a per-trial
generator makes the draw independent of both scheduling and thread count.
"""
function _slds_smooth_all!(
    slds::SLDS{T},
    cell_slds::Union{Nothing,AbstractVector},
    grp::Union{Nothing,ParameterGrouping},
    tfs::TrialFilterSmooth{T},
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple},
    x_samples::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}},
    pool::SLDSWorkspacePool{T},
    plan::SLDSTrialPlan,
    w_of;
    rng_of=_ -> Random.default_rng(),
    noise_of=_ -> nothing,
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}},NamedTuple}=nothing,
    lognorm::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real}
    grouped = grp !== nothing && cell_slds !== nothing

    tforeach(1:_plan_ntasks(plan)) do slot
        local lo, hi, cur_cell, ws_t, slds_t
        lo = plan.bounds[slot]
        hi = plan.bounds[slot + 1] - 1
        lo > hi && return nothing

        cur_cell = 0
        ws_t = pool.slots[slot]
        slds_t = slds

        for idx in lo:hi
            trial = plan.order[idx]
            if grouped && plan.cell_of[idx] != cur_cell
                cur_cell = plan.cell_of[idx]
                slds_t = cell_slds[cur_cell]
                ws_t = _slds_pool_ws(pool, slot, cur_cell, cell_slds)
                refresh_slds_constants!(ws_t, slds_t)
            end

            smooth!(
                slds_t,
                tfs[trial],
                _trial(y, trial),
                w_of(trial);
                ws=ws_t,
                x_sample=(x_samples === nothing ? nothing : x_samples[trial]),
                rng=rng_of(trial),
                noise=noise_of(trial),
                ux=(ux === nothing ? nothing : ux[trial]),
                uy=(uy === nothing ? nothing : _trial(uy, trial)),
                lognorm_t=(lognorm === nothing ? nothing : lognorm[trial]),
            )
        end
        return nothing
    end
    return nothing
end

# ============================================================================
# Reproducibility of the E-step's posterior draws under trial parallelism
#
# The Monte-Carlo E-step draws one joint sample per trial. Sequentially those
# draws came off a single generator in trial order; run in parallel that is
# both a data race and schedule-dependent, so the draw has to be re-sourced.
# Two modes, because the two things a user wants here are in tension:
#
#   :trial  — each trial gets its own generator, seeded from the master `rng`
#             and the trial index. The draw a trial receives is then a function
#             of the master seed alone: identical across thread counts, across
#             chunk layouts, and across serial vs. parallel. This is the default.
#
#   :global — the standard normals are drawn from the master `rng` serially, in
#             trial order, before the pass; each trial consumes its own slice.
#             That reproduces the exact stream the sequential smoother consumed,
#             for going back to an existing fit, and still runs in parallel.
# ============================================================================

"""
    _slds_noise_buffers(x_samples) -> Vector{Vector} or nothing

Per-trial standard-normal buffers for `rng_mode = :global`, sized at each
trial's `latent_dim · T_i`. `nothing` when the caller takes no draws.
"""
function _slds_noise_buffers(x_samples::AbstractVector{<:AbstractMatrix{T}}) where {T<:Real}
    return [Vector{T}(undef, length(xs)) for xs in x_samples]
end

_slds_noise_buffers(::Nothing) = nothing

"""
    _slds_draw_sources(rng, rng_mode, x_samples, noise_bufs) -> (rng_of, noise_of)

Resolve one alternation's draw source into the two callbacks
[`_slds_smooth_all!`](@ref) takes. Both are cheap closures over per-trial data;
neither touches shared mutable state inside a task.

Called once per alternation, and each call consumes exactly one draw from
`rng` — a seed in `:trial` mode, the whole noise stream in `:global`. That is
what keeps `smoothing_iters = n` identical to `n` successive
`smoothing_iters = 1` calls: either way the k-th alternation is the k-th draw
off the master generator.
"""
# Deterministic path: no draws at all.
function _slds_draw_sources(rng::AbstractRNG, ::Symbol, ::Nothing, ::Any)
    return (_ -> rng), (_ -> nothing)
end

function _slds_draw_sources(
    rng::AbstractRNG,
    rng_mode::Symbol,
    ::AbstractVector{<:AbstractMatrix},
    noise_bufs::Union{Nothing,AbstractVector{<:AbstractVector}},
)
    if rng_mode === :global
        noise_bufs === nothing && throw(
            ArgumentError(
                "rng_mode = :global needs the pre-drawn noise buffers; " *
                "this is a caller bug, not a user-facing one",
            ),
        )
        # Serial pre-draw, in trial order — the sequential smoother's stream.
        for trial in eachindex(noise_bufs)
            randn!(rng, noise_bufs[trial])
        end
        return (_ -> rng), (trial -> noise_bufs[trial])
    end

    rng_mode === :trial ||
        throw(ArgumentError("rng_mode must be :trial or :global, got $(repr(rng_mode))"))

    #=
    One seed per alternation, taken from the master generator on this thread,
    then mixed with the trial index. Per trial rather than per task, so the
    sample a trial gets does not move when the chunk layout does — and *only*
    the trial index, so that the alternation's identity comes from which draw
    off `rng` produced its seed, not from its position within a call.
    =#
    pass_seed = rand(rng, UInt64)
    return (trial -> Random.Xoshiro(hash((pass_seed, trial)))), (_ -> nothing)
end

"""
    _vem_alternate!(slds, cell_slds, grp, tfs, fb_storage, dl, y, pool, plan; smoothing_iters, tol, x_samples, ...)

Run up to `smoothing_iters` discrete↔continuous alternations of the structured-
variational E-step. One alternation is:

1. fill `dl.logL` (`K × sum(T_i)`) with per-regime log-likelihoods of the current
   continuous trajectory,
2. refresh `q(z) = γ` by forward-backward over the switching chain (HMMs.jl threads
   across trials), and
3. refresh `q(x)` by re-running the Laplace/Newton smoother on each trial under the new
   `γ`, filling `tfs[*].x_smooth`, `tfs[*].p_smooth`, and `tfs[*].entropy`.

`x_samples` selects how step 1 reads the continuous trajectory, and is the only
difference between the two callers:

- `x_samples === nothing` — plug in the smoothed mean `E_q[x]`, with the
  `½ tr(H^{(k,t)} Σ)` correction of [`_add_cov_correction!`](@ref) added back so
  step 1 still scores `E_q(x)[log p_k]` rather than the biased plug-in.
  Deterministic and reproducible; used by [`smooth`](@ref) for post-fit inference.
- `x_samples !== nothing` — plug in a joint draw from `q(x)`, and draw the next one in
  step 3. This is the vLEM Monte-Carlo E-step used by [`fit!`](@ref); `x_samples` is
  read then overwritten within each alternation.

`grp` / `cell_slds` carry an ancillary-dependency (`depends_on`) grouping; both
`nothing` keeps every step on the ungrouped code path.

`tol` selects the stopping rule. `tol == 0` runs exactly `smoothing_iters`
alternations; `tol > 0` stops early once `max|Δγ| < tol`.

Returns `(iters, converged)`.
"""
function _vem_alternate!(
    slds::SLDS{T},
    cell_slds::Union{Nothing,AbstractVector},
    grp::Union{Nothing,ParameterGrouping},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple},
    pool::SLDSWorkspacePool{T},
    plan::SLDSTrialPlan;
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}},NamedTuple}=nothing,
    lognorm::Union{Nothing,AbstractVector}=nothing,
    smoothing_iters::Int,
    tol::T=zero(T),
    x_samples::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    rng_mode::Symbol=:trial,
    noise_bufs::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
    prog=nothing,
) where {T<:Real}
    smoothing_iters >= 1 ||
        throw(ArgumentError("smoothing_iters must be ≥ 1, got $smoothing_iters"))

    K = length(slds.LDSs)

    # Deterministic path scores the smoothed mean; sampled path a draw from q(x).
    function x_of(trial)
        return x_samples === nothing ? tfs[trial].x_smooth : x_samples[trial]
    end

    function w_of(trial)
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        return view(fb_storage.γ, :, t1:t2)
    end

    # Previous-iteration γ snapshot; only allocated when there is a stopping test.
    γ_prev = tol > 0 ? fill(T(Inf), K, last(seq_ends)) : nothing
    converged = false
    iters = 0

    for iter in 1:smoothing_iters
        iters = iter

        #=
        (1) Score the current continuous trajectory under each regime. The
        deterministic path plugs in the smoothed mean, so it also needs the
        `½ tr(H Σ)` term that turns that plug-in into E_q(x)[·]; the sampled
        path gets the spread from the draw itself and passes `tfs = nothing`.
        =#
        _slds_fill_logL!(
            slds,
            cell_slds,
            grp,
            dl,
            y,
            x_of,
            pool,
            plan;
            seq_ends=seq_ends,
            ux=ux,
            uy=uy,
            lognorm=lognorm,
            tfs=(x_samples === nothing ? tfs : nothing),
        )

        # (2) Update q(z): single batched forward-backward across all trials.
        HMMs.forward_backward!(
            fb_storage,
            dl,
            obs_seq,
            control_seq;
            seq_ends=seq_ends,
            transition_marginals=true,
        )

        #=
        (3) Update q(x) under the fresh γ, drawing the next sample on the way
        out. Overwriting `x_samples` here is fine — step (1) already used the
        previous draw.
        =#
        rng_of, noise_of = _slds_draw_sources(rng, rng_mode, x_samples, noise_bufs)
        _slds_smooth_all!(
            slds,
            cell_slds,
            grp,
            tfs,
            y,
            x_samples,
            pool,
            plan,
            w_of;
            rng_of=rng_of,
            noise_of=noise_of,
            ux=ux,
            uy=uy,
            lognorm=lognorm,
        )

        prog !== nothing && next!(prog)

        if γ_prev !== nothing
            if iter > 1
                Δγ = zero(T)
                @inbounds for i in eachindex(fb_storage.γ)
                    d = abs(fb_storage.γ[i] - γ_prev[i])
                    d > Δγ && (Δγ = d)
                end
                if Δγ < tol
                    converged = true
                    break
                end
            end
            copyto!(γ_prev, fb_storage.γ)
        end
    end

    return iters, converged
end

"""
    estep!(slds, tfs, fb_storage, dl, y, x_samples, pool, plan; rng, obs_seq, control_seq, seq_ends, smoothing_iters=1)

Monte-Carlo E-step for the SLDS: `smoothing_iters` coordinate-ascent alternations of
[`_vem_alternate!`](@ref), each scoring the discrete layer against a joint draw from
`q(x)` and drawing the next one. `smoothing_iters = 1` is the standard vLEM E-step;
larger values hand the M-step a better-converged posterior at proportional cost.

`x_samples` is read (to fill `dl.logL`) then overwritten (with the fresh draw) within
each alternation. `obs_seq`/`control_seq` are the HMMs.jl placeholder sequences built in
`fit!` (timestep indices / `nothing`s) — unrelated to the LDS control-input kwargs
`ux`/`uy`. The latter, when supplied, are per-trial vectors of input matrices
(`ux[trial]` is `(ux_dim, T_trial)`, `uy[trial]` is `(uy_dim, T_trial)`); they
feed the per-regime `Bₖ u` / `Dₖ v` terms of every trial's smoother and
log-likelihood fill. `nothing` (the default) means no inputs.
"""
function estep!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple},
    x_samples::AbstractVector{<:AbstractMatrix{T}},
    pool::SLDSWorkspacePool{T},
    plan::SLDSTrialPlan;
    rng::AbstractRNG=Random.default_rng(),
    rng_mode::Symbol=:trial,
    noise_bufs::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}},NamedTuple}=nothing,
    lognorm::Union{Nothing,AbstractVector}=nothing,
    smoothing_iters::Int=1,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    _vem_alternate!(
        slds,
        nothing,
        nothing,
        tfs,
        fb_storage,
        dl,
        y,
        pool,
        plan;
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
        ux=ux,
        uy=uy,
        lognorm=lognorm,
        smoothing_iters=smoothing_iters,
        x_samples=x_samples,
        rng=rng,
        rng_mode=rng_mode,
        noise_bufs=noise_bufs,
    )
    return nothing
end

"""
    estep!(slds, tfs, fb_storage, dl, y, x_samples, slds_ws; ...)

Single-workspace E-step: the caller owns one `SLDSSmoothWorkspace` rather than
a pool, so every trial runs on it sequentially. Equivalent to the pooled form
with `npool = 1`, and the form to reach for when driving the E-step by hand.
"""
function estep!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple},
    x_samples::AbstractVector{<:AbstractMatrix{T}},
    slds_ws::SLDSSmoothWorkspace{T};
    rng::AbstractRNG=Random.default_rng(),
    rng_mode::Symbol=:trial,
    noise_bufs::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}},NamedTuple}=nothing,
    lognorm::Union{Nothing,AbstractVector}=_slds_lognorm_all(slds, y),
    smoothing_iters::Int=1,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    return estep!(
        slds,
        tfs,
        fb_storage,
        dl,
        y,
        x_samples,
        _slds_solo_pool(slds_ws),
        _slds_trial_plan(nothing, _ntrials(y), 1);
        rng=rng,
        rng_mode=rng_mode,
        noise_bufs=noise_bufs,
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
        ux=ux,
        uy=uy,
        lognorm=lognorm,
        smoothing_iters=smoothing_iters,
    )
end

# tr(A·B) without forming the product: Σ_ij A[i,j]·B[j,i].
@inline function _tr_prod(A::AbstractMatrix, B::AbstractMatrix)
    acc = zero(promote_type(eltype(A), eltype(B)))
    for j in axes(A, 2), i in axes(A, 1)
        acc += A[i, j] * B[j, i]
    end
    return acc
end

"""
    _slds_prior_logdensity(slds)

Sum of the per-regime parameter log-prior contributions, via the shared
[`_state_prior_logdensity`](@ref) and [`_obs_prior_logdensity`](@ref): IW on
`Q`/`P0`/`R`, MN on `[A b B]`/`[C d D]`, and the MN-only `[C d D]` term for
Poisson emissions that matches their M-step objective. A composite emission sums
over its members. Zero when no priors are set.

Needed so the ELBO tracks the same MAP objective the M-step optimizes; without it
the displayed ELBO can appear non-monotone under priors.
"""
function _slds_prior_logdensity(slds::SLDS{T}) where {T<:Real}
    prior_term = zero(T)
    for lds in slds.LDSs
        prior_term += _state_prior_logdensity(lds, nothing)
        prior_term += _obs_prior_logdensity(lds, nothing)
    end
    return prior_term
end

"""
    _slds_trial_elbo(slds, fs, fb_storage, y_trial, slds_ws, t1, t2, ux_trial, uy_trial, lognorm_t)

One trial's contribution to the SLDS ELBO (everything except the parameter
log-prior). Split out of `elbo!` so a caller can evaluate a trial against a
different parameter set than its neighbours, after refreshing the regime
constants.

Assumes `slds_ws.consts` already holds the constants of `slds`.
"""
function _slds_trial_elbo(
    slds::SLDS{T,S,O},
    fs::FilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    y_trial::Union{AbstractMatrix{T},NamedTuple},
    slds_ws::SLDSSmoothWorkspace{T},
    t1::Int,
    t2::Int,
    ux_trial::Union{Nothing,AbstractMatrix{T}},
    uy_trial::Union{Nothing,AbstractMatrix{T},NamedTuple},
    lognorm_t::Union{Nothing,AbstractVector{T},NamedTuple}=nothing,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:AbstractObservationModel{T}}
    K = length(slds.LDSs)
    Tsteps = t2 - t1 + 1
    w = view(fb_storage.γ, :, t1:t2)  # K × Tsteps

    trial_elbo = zero(T)
    x_smooth_trial = fs.x_smooth

    #=
    Per-regime log-density scratch, built once rather than per regime. The
    `::AbstractVector{T}` is what keeps JET quiet: analysed at the unspecialized
    signature, `T` is only known to be `<:Real`, so `view` over the workspace
    field widens to a union carrying an `Any`-eltype branch, and no
    `joint_loglikelihood!` method matches that. Asserting the element type drops
    the branch; it always held.
    =#
    ll = view(slds_ws.ll_tmp, 1:Tsteps)::AbstractVector{T}

    # E_q[log p(y, x | z)], plug-in at the posterior mean, weighted by γ.
    for k in 1:K
        _slds_trial_loglikelihood!(
            ll,
            slds_ws,
            slds_ws.consts[k],
            slds.LDSs[k],
            x_smooth_trial,
            y_trial,
            ux_trial,
            uy_trial,
            lognorm_t,
            _regime_obs(slds_ws, k),
        )
        for t in 1:Tsteps
            trial_elbo += w[k, t] * ll[t]
        end
    end

    #=
    ½ tr(H Σ) covariance correction. H = weighted Hessian (hessian! writes
    it un-negated into slds_ws.btd); Σ = p_smooth on the diagonal,
    p_smooth_tt1[:,:,t] = Cov(x_t, x_{t-1}) off it. Sum both off-diagonal
    traces rather than doubling one — don't assume exact block symmetry.
    =#
    hessian!(slds_ws, slds, x_smooth_trial, y_trial, w, uy_trial)
    H_diag = slds_ws.btd.H_diag
    H_sub = slds_ws.btd.H_sub
    H_super = slds_ws.btd.H_super
    for t in 1:Tsteps
        trial_elbo += T(0.5) * _tr_prod(H_diag[t], view(fs.p_smooth, :, :, t))
    end
    for t in 2:Tsteps
        Σ_ttm1 = view(fs.p_smooth_tt1, :, :, t)  # Cov(x_t, x_{t-1})
        trial_elbo += T(0.5) * _tr_prod(H_super[t - 1], Σ_ttm1)
        trial_elbo += T(0.5) * _tr_prod(H_sub[t - 1], transpose(Σ_ttm1))
    end

    # E_q[log p(z_1)].
    for k in 1:K
        trial_elbo += w[k, 1] * log(slds.πₖ[k] + T(1e-12))
    end

    #=
    E_q[log p(z_t | z_{t-1})] = Σ_t Σ_ij ξ_t[i,j] log A[i,j]. ξ is global-
    indexed; ξ[t2] is zero by FB convention, so iterate t1..t2-1.
    =#
    for t in t1:(t2 - 1)
        ξt = fb_storage.ξ[t]
        for i in 1:K, j in 1:K
            trial_elbo += ξt[i, j] * log(slds.A[i, j] + T(1e-12))
        end
    end

    # + H[q(x)] (filled by `smooth!` from the BT log-determinant).
    trial_elbo += fs.entropy

    #=
    + H[q(z)], the FB chain entropy
    −Σ_k γ₁ log γ₁ − Σ_t Σ_ij ξ_t[i,j] (log ξ_t[i,j] − log γ_t[i]).
    ξ_t[i,j] > 0 ⇒ γ_t[i] > 0, so both logs are safe.
    =#
    for k in 1:K
        wk1 = w[k, 1]
        wk1 > 0 && (trial_elbo -= wk1 * log(wk1))
    end
    for t in t1:(t2 - 1)
        ξt = fb_storage.ξ[t]
        tloc = t - t1 + 1
        for i in 1:K, j in 1:K
            ξij = ξt[i, j]
            ξij > 0 && (trial_elbo -= ξij * (log(ξij) - log(w[i, tloc])))
        end
    end

    return trial_elbo
end

"""
    elbo!(slds, tfs, fb_storage, y, pool, plan; seq_ends)

Evidence lower bound for the SLDS at the current variational posteriors —
q(x) the per-trial joint Gaussian from the Laplace smoother, q(z) the
forward-backward chain posterior:

    ELBO = E_q[log p(y, x | z)] + E_q[log p(z)] + H[q(x)] + H[q(z)] + log p(θ)

- `E_q[log p(y, x | z)]` is the responsibility-weighted log-density at the
  posterior mean plus the covariance correction `½ tr(H Σ)`, where `H` is the
  weighted Hessian over `x₁:T` and `Σ` the block-tridiagonal posterior
  covariance. Exact for Gaussian emissions (the weighted log-density is
  quadratic in `x`); the standard second-order/Laplace approximation for
  Poisson.
- `E_q[log p(z)]` uses the FB marginals `γ` (initial) and pairwise `ξ`
  (transitions).
- `H[q(z)]` is the Markov-chain entropy of the FB posterior,
  `−Σ γ₁ log γ₁ − Σ_t Σ_ij ξ_t(i,j) log(ξ_t(i,j)/γ_t(i))` — not the
  factorized `−Σ γ log γ`, which would overstate the entropy of a chain.
- `log p(θ)` collects per-regime IW/MN prior log-densities so the ELBO tracks
  the MAP objective the M-step optimizes (zero when no priors are set).

The continuous term is evaluated at the smoothed mean (deterministic given the
current posteriors), not at the E-step's posterior sample. For K = 1 with
Gaussian emissions and no priors this equals the exact marginal log-likelihood.

Returns a scalar. Overwrites `slds_ws.btd`'s Hessian blocks and `ll_tmp`.
"""
function elbo!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple},
    pool::SLDSWorkspacePool{T},
    plan::SLDSTrialPlan;
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}},NamedTuple}=nothing,
    lognorm::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    per_trial = _slds_trial_elbos(
        slds, nothing, nothing, tfs, fb_storage, y, pool, plan; seq_ends, ux, uy, lognorm
    )
    return sum(per_trial) + _slds_prior_logdensity(slds)
end

"""
    elbo!(slds, tfs, fb_storage, y, slds_ws; seq_ends, ux, uy)

Single-workspace ELBO: as above with the trials run sequentially on one
workspace. Returns the same number the pooled form does — the per-trial
contributions are summed in trial order either way.
"""
function elbo!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple},
    slds_ws::SLDSSmoothWorkspace{T};
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}},NamedTuple}=nothing,
    lognorm::Union{Nothing,AbstractVector}=_slds_lognorm_all(slds, y),
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    return elbo!(
        slds,
        tfs,
        fb_storage,
        y,
        _slds_solo_pool(slds_ws),
        _slds_trial_plan(nothing, _ntrials(y), 1);
        seq_ends=seq_ends,
        ux=ux,
        uy=uy,
        lognorm=lognorm,
    )
end

"""
    _slds_trial_elbos(slds, cell_slds, grp, tfs, fb_storage, y, pool, plan; ...)

Every trial's ELBO contribution, computed in parallel and returned **per trial**.
Shared by the ungrouped and grouped ELBOs, which differ only in which parameter
set each trial is scored against.

Per trial rather than per chunk deliberately: the caller then sums in trial
order, so the total is the same number whatever `npool` is. Chunk partials would
have been cheaper by one vector, but would have made the ELBO depend on the
chunk layout at rounding level, and the ELBO is a number users compare across
runs. `ntrials` extra floats is not a cost worth that.
"""
function _slds_trial_elbos(
    slds::SLDS{T},
    cell_slds::Union{Nothing,AbstractVector},
    grp::Union{Nothing,ParameterGrouping},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple},
    pool::SLDSWorkspacePool{T},
    plan::SLDSTrialPlan;
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}},NamedTuple}=nothing,
    lognorm::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real}
    per_trial = zeros(T, _ntrials(y))
    grouped = grp !== nothing && cell_slds !== nothing

    tforeach(1:_plan_ntasks(plan)) do slot
        local lo, hi, cur_cell, ws_t, slds_t
        lo = plan.bounds[slot]
        hi = plan.bounds[slot + 1] - 1
        lo > hi && return nothing

        cur_cell = 0
        ws_t = pool.slots[slot]
        slds_t = slds

        for idx in lo:hi
            trial = plan.order[idx]
            if grouped && plan.cell_of[idx] != cur_cell
                cur_cell = plan.cell_of[idx]
                slds_t = cell_slds[cur_cell]
                ws_t = _slds_pool_ws(pool, slot, cur_cell, cell_slds)
                refresh_slds_constants!(ws_t, slds_t)
            end

            t1, t2 = HMMs.seq_limits(seq_ends, trial)
            per_trial[trial] = _slds_trial_elbo(
                slds_t,
                tfs[trial],
                fb_storage,
                _trial(y, trial),
                ws_t,
                t1,
                t2,
                ux === nothing ? nothing : ux[trial],
                uy === nothing ? nothing : _trial(uy, trial),
                lognorm === nothing ? nothing : lognorm[trial],
            )
        end
        return nothing
    end

    return per_trial
end

"""
    elbo(slds, y; ux=nothing, uy=nothing, smoothing_iters=100, tol=1e-6,
         progress=false, depends_on=nothing)

Evidence lower bound of an `SLDS` at the current parameters — the `elbo` field of
[`smooth`](@ref)`(slds, y)`, which infers `q(x)` and `q(z)` by deterministic
coordinate ascent before evaluating the bound. Deterministic and reproducible.

Accepts the same observation and input forms as [`smooth`](@ref), and the same
`smoothing_iters` / `tol` controls over the alternation. Returns a scalar.

If you also want the posteriors that produced it, call [`smooth`](@ref) once and read
its `elbo` field rather than paying for the alternation twice.
"""
function elbo(
    slds::SLDS{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    ux=nothing,
    uy=nothing,
    smoothing_iters::Int=100,
    tol::Real=1e-6,
    progress::Bool=false,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    return smooth(
        slds,
        y;
        ux=ux,
        uy=uy,
        smoothing_iters=smoothing_iters,
        tol=tol,
        return_cov=false,
        progress=progress,
        depends_on=depends_on,
    ).elbo
end

"""
    loglikelihood(slds, y; kwargs...)

Variational lower bound on the marginal log-likelihood of an `SLDS`, i.e.
[`elbo`](@ref)`(slds, y)`.

The exact marginal `log p(y)` is intractable for a switching model — it requires
summing over all `K^T` discrete regime sequences — so this returns the ELBO instead.
Values are comparable across models fit to the same data, but are lower bounds, not
likelihoods. Accepts the same keywords as [`elbo`](@ref).
"""
function StatsAPI.loglikelihood(slds::SLDS, y; kwargs...)
    return elbo(slds, y; kwargs...)
end

"""
    _tie_slots(tied::Bool, n) -> Vector{Int}

Slot vector over `n` units for one parameter group: every unit on slot 1 when
the group is tied, one slot each when it is not. Feeding these to the grouped
updates (`_grouped_update_A_b!` and friends) is what makes "tied across
regimes" and "free per regime" the same code path.
"""
_tie_slots(tied::Bool, n::Int) = tied ? ones(Int, n) : collect(1:n)

"""
    _validate_tied_params(lds, tied, grouped)

Reject the `tied_params` combinations the M-step has no estimator for.

A partial tie — some but not all columns of `[A b B]` or `[C d D]` — is fitted
by residualizing the free columns out and solving the shared block by
generalized least squares (`_partial_tied_regression`). That needs the group to
*be* a least-squares regression fitted from one set of sufficient statistics per
regime, which rules out two cases:

- a **Poisson** emission, whose `[C d D]` has no sufficient-statistic form and
  is fitted by LBFGS; and
- a fit that also groups trials with `depends_on`, where a regime's regression
  is several versions across cells rather than one, and a partial tie would have
  to partition columns and cells at once.

Both are fine with the whole regression tied.
"""
function _validate_tied_params(
    lds::LinearDynamicalSystem, tied::AbstractVector{Symbol}, grouped::Bool
)
    isempty(tied) && return nothing
    D = lds.latent_dim
    #=
    An inverse-LQR state's structural parameters are not columns of a regression,
    so the partial-column reasoning below does not apply to them. A tie there is
    over named blocks, run as an alternation by `_ham_structure_phases!`, and the
    emission half is validated on its own terms.
    =#
    if lds.state_model isa HamiltonianStateModel
        obs_h = _tied_obs_cols(tied, D, lds.uy_dim)
        if !isempty(obs_h) &&
            length(obs_h) < D + 1 + lds.uy_dim &&
            lds.obs_model isa PoissonObservationModel
            throw(
                ArgumentError(
                    "tied_params: a Poisson emission's `[C d D]` is fitted by LBFGS, " *
                    "not from sufficient statistics, so there is no way to share part " *
                    "of it across regimes. Tie " *
                    "$(_join_names(_group_members(lds.obs_model, :C))) together, or " *
                    "none of them.",
                ),
            )
        end
        return nothing
    end
    dyn = _tied_dyn_cols(tied, D, lds.ux_dim)
    obs = _tied_obs_cols(tied, D, lds.uy_dim)
    partial_dyn = !isempty(dyn) && length(dyn) < D + 1 + lds.ux_dim
    partial_obs = !isempty(obs) && length(obs) < D + 1 + lds.uy_dim

    if partial_obs && lds.obs_model isa PoissonObservationModel
        throw(
            ArgumentError(
                "tied_params: a Poisson emission's `[C d D]` is fitted by LBFGS, not " *
                "from sufficient statistics, so there is no way to share part of it " *
                "across regimes. Tie $(_join_names(_group_members(lds.obs_model, :C))) " *
                "together, or none of them.",
            ),
        )
    end

    if grouped && (partial_dyn || partial_obs)
        which = partial_dyn ? "`[A b B]`" : "`[C d D]`"
        throw(
            ArgumentError(
                "tied_params: sharing part of $which across regimes is not supported " *
                "alongside `depends_on`, which already splits that regression into one " *
                "version per group of trials. Tie the whole regression, or drop " *
                "`depends_on`.",
            ),
        )
    end
    return nothing
end

"""
    _broadcast_tied_params!(slds, tied)

Copy `LDSs[1]`'s tied parameters into every other regime.

The updates fit a shared value on the first regime that uses it, and an `SLDS`'s
regimes hold separate arrays rather than aliasing one (unlike the `depends_on`
variants, which share by reference), so the fitted value has to be copied out.
Copies by *column* for the stacked regressions, so a partial tie moves only the
shared columns and leaves each regime's free ones alone. Honours `fit_bool`: a
frozen group is left exactly as the caller set it, per regime.
"""
function _broadcast_tied_params!(
    slds::SLDS{T}, tied::AbstractVector{Symbol}
) where {T<:Real}
    isempty(tied) && return nothing
    src = slds.LDSs[1]
    #=
    A Hamiltonian state's tie is applied inside its own M-step, which fits the
    shared version jointly from every state that uses it and copies it out. Its
    parameters are not columns of a regression, so the packing below has nothing
    to pack.
    =#
    if src.state_model isa HamiltonianStateModel
        _broadcast_tied_obs!(src.obs_model, slds, tied)
        return nothing
    end
    D = src.latent_dim
    dyn_cols = _tied_dyn_cols(tied, D, src.ux_dim)

    W_src = Matrix{T}(undef, D, D + 1 + src.ux_dim)
    W_dst = similar(W_src)
    isempty(dyn_cols) || _pack_dyn_W!(W_src, src)

    for k in 2:length(slds.LDSs)
        dst = slds.LDSs[k]
        if !isempty(dyn_cols) && dst.fit_bool[_G_AB]
            _pack_dyn_W!(W_dst, dst)
            @views W_dst[:, dyn_cols] .= W_src[:, dyn_cols]
            _unpack_dyn_W!(dst, W_dst)
        end
        if :Q in tied && dst.fit_bool[_G_Q]
            copyto!(dst.state_model.Q, src.state_model.Q)
        end
    end

    _broadcast_tied_obs!(src.obs_model, slds, tied)
    return nothing
end

#=
The emission half, per observation model. A composite runs it once per member,
on that member's per-regime views and its own suffixed names, so a tie may name
one emission and leave the others free.
=#
function _broadcast_tied_obs!(
    ::AbstractObservationModel, slds::SLDS{T}, tied::AbstractVector{Symbol}
) where {T<:Real}
    _broadcast_tied_obs_group!(slds.LDSs, tied, nothing)
    return nothing
end

function _broadcast_tied_obs!(
    om::CompositeObservationModel, slds::SLDS{T}, tied::AbstractVector{Symbol}
) where {T<:Real}
    for key in _obs_keys(om)
        _broadcast_tied_obs_group!([_obs_view(lds, key) for lds in slds.LDSs], tied, key)
    end
    return nothing
end

function _broadcast_tied_obs_group!(
    ldss::AbstractVector, tied::AbstractVector{Symbol}, key::Union{Nothing,Symbol}
)
    src = ldss[1]
    T = eltype(src.obs_model.C)
    D, p = src.latent_dim, src.obs_dim
    obs_cols = _tied_obs_cols(tied, D, src.uy_dim, key)
    tie_R = _tied_name(:R, key) in tied

    (isempty(obs_cols) && !tie_R) && return nothing

    V_src = Matrix{T}(undef, p, D + 1 + src.uy_dim)
    V_dst = similar(V_src)
    isempty(obs_cols) || _pack_obs_V!(V_src, src)

    for k in 2:length(ldss)
        dst = ldss[k]
        if !isempty(obs_cols) && dst.fit_bool[_G_CD]
            _pack_obs_V!(V_dst, dst)
            @views V_dst[:, obs_cols] .= V_src[:, obs_cols]
            _unpack_obs_V!(dst, V_dst)
        end
        if tie_R && length(dst.fit_bool) >= _G_R && dst.fit_bool[_G_R]
            copyto!(dst.obs_model.R, src.obs_model.R)
        end
    end
    return nothing
end

"""
    _tied_poisson_emission!(slds, tfs, data, sws, weights_of, tied; ntasks)

Poisson emission M-step for an `SLDS`. Non-conjugate, so it is one Newton solve
per distinct `[C d D]`: `K` of them at the regimes' own responsibilities, or a
single unit-weight one when `[C d D]` is tied whole — summing the per-regime weighted
objectives collapses to the unit-weight one, because the emission term does not
depend on `k` and `Σₖ γₖ(t) = 1`.

`ntasks` is how many chunks each solve splits its trials into. The solver keeps
its own per-chunk scratch, so one workspace still serves; without this the
one-element pool an SLDS has to hand pinned every solve to a single task, which
is what left the emission M-step serial while the PLDS one ran chunked.
"""
function _tied_poisson_emission!(
    ldss::AbstractVector,
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    uy::AbstractVector{<:AbstractMatrix{T}},
    sws::SmoothWorkspace{T},
    weights_of,
    tie_emission::Bool;
    ntasks::Int=1,
) where {T<:Real}
    if tie_emission
        update_observation_model!(ldss[1], tfs, y, [sws], nothing; uy=uy, ntasks=ntasks)
        return nothing
    end
    #=
    The `K` solves write into disjoint `slds.LDSs[k].obs_model` arrays and read
    only `tfs` / `data`, so they are independent. They stay sequential here and
    each parallelises over its own trials instead: `ntrials ≫ K` in every fit
    this is for, so chunking trials fills the threads and chunking regimes on
    top would only fragment them.
    =#
    for k in eachindex(ldss)
        update_observation_model!(
            ldss[k], tfs, y, [sws], weights_of(k); uy=uy, ntasks=ntasks
        )
    end
    return nothing
end

#=
Which stacked regression an update is for. The two differ only in which blocks
of the sufficient statistics, which noise covariance and which prior they read,
so the tie logic is written once and dispatched on these.
=#
struct _DynBlock end
struct _ObsBlock end

_block_stats(::_DynBlock, suf) = (suf.dyn_xx[].mat, suf.dyn_xy)
_block_stats(::_ObsBlock, suf) = (suf.obs_xx[].mat, suf.obs_xy)

_block_noise(::_DynBlock, lds) = lds.state_model.Q
_block_noise(::_ObsBlock, lds) = lds.obs_model.R

_block_prior(::_DynBlock, lds) = lds.state_model.AB_prior
_block_prior(::_ObsBlock, lds) = lds.obs_model.CD_prior

_block_group(::_DynBlock) = _G_AB
_block_group(::_ObsBlock) = _G_CD

_block_width(::_DynBlock, lds) = lds.latent_dim + 1 + lds.ux_dim
_block_width(::_ObsBlock, lds) = lds.latent_dim + 1 + lds.uy_dim

_block_write!(::_DynBlock, lds, W) = _unpack_dyn_W!(lds, W)
_block_write!(::_ObsBlock, lds, W) = _unpack_obs_V!(lds, W)

function _block_grouped_update!(::_DynBlock, ldss, sufs, slots, noise_slots, sws, bufs)
    return _grouped_update_A_b!(ldss, sufs, slots, noise_slots, sws, bufs)
end

function _block_grouped_update!(::_ObsBlock, ldss, sufs, slots, noise_slots, sws, bufs)
    return _grouped_update_C_d!(ldss, sufs, slots, noise_slots, sws, bufs)
end

"""
    _slds_update_regression!(block, slds, sufs, tied_cols, noise_slots, sws, bufs, K)
        -> Vector{Int}

Fit one stacked regression across the regimes and return its slot vector — which
regimes ended up sharing a value, for the covariance update that follows.

Free or shared whole, the fit is the grouped update the `depends_on` path uses,
which picks the pooled or the generalized-least-squares estimator from whether
the regimes also share the covariance. A partial tie goes to
[`_partial_tied_regression`](@ref), which writes every regime's stacked matrix
directly — so its slots are all distinct: the regimes agree on the tied columns
and differ everywhere else.
"""
function _slds_update_regression!(
    block,
    ldss::AbstractVector,
    sufs::AbstractVector,
    tied_cols::AbstractVector{Int},
    noise_slots::AbstractVector{Int},
    sws::SmoothWorkspace{T},
    bufs::GroupedSufBuffers{T},
    K::Int,
) where {T<:Real}
    lds1 = ldss[1]

    if isempty(tied_cols) || length(tied_cols) == _block_width(block, lds1)
        slots = _tie_slots(!isempty(tied_cols), K)
        _block_grouped_update!(block, ldss, sufs, slots, noise_slots, sws, bufs)
        return slots
    end

    slots = collect(1:K)
    lds1.fit_bool[_block_group(block)] || return slots

    stats = [_block_stats(block, sufs[k]) for k in 1:K]
    Ws = _partial_tied_regression(
        [st[1] for st in stats],
        [st[2] for st in stats],
        [_block_noise(block, ldss[k]) for k in 1:K],
        [_block_prior(block, ldss[k]) for k in 1:K],
        tied_cols,
        "tied_params",
    )
    for k in 1:K
        ldss[k].fit_bool[_block_group(block)] && _block_write!(block, ldss[k], Ws[k])
    end
    return slots
end

"""
    mstep!(slds, tfs, fb_storage, dl, y, sws; obs_seq, seq_ends, ux=nothing, uy=nothing,
           tied=Symbol[])

M-step for SLDS.

- Updates discrete parameters (`slds.A`, `slds.πₖ`) via `StatsAPI.fit!` on the discrete
  layer (uses HMMs.jl's `ξ[t2]` scratch trick).
- Updates each LDS component using γ-weighted sufficient statistics aggregated
  by `_aggregate_td_suff_stats_weighted!`. For Gaussian sub-LDSs this is the
  full suf-based M-step (regression + IW MAP). For Poisson sub-LDSs the state-
  side updates flow through the same suf path; the emission [C d] is updated
  via the existing LBFGS routine (Poisson is non-conjugate and cannot be
  folded into the regression).

`ux` / `uy` are the per-trial control-input sequences; when present, the weighted
aggregator folds `Bₖ u` / `Dₖ v` into the regression targets so `Bₖ` (Gaussian
and Poisson dynamics) and `Dₖ` (Gaussian emission, and Poisson emission via the
LBFGS routine) are re-estimated alongside `Aₖ` / `Cₖ`.

`tied` names the parameter groups (canonical `:A` / `:Q` / `:C` / `:R`, from
`tied_params`) that every regime shares. Each becomes a one-slot group over the
`K` regimes and goes through the same `_grouped_update_*!` helpers the
`depends_on` path uses, then `_broadcast_tied_params!` copies the fitted value
into the other regimes. `x0`/`P0` are tied unconditionally, below.
"""
#=============================================================================
Hamiltonian (inverse-LQR) discrete states.

The SLDS machinery is state-model agnostic almost everywhere — `state_loglikelihood!`
and `_transition_residual!` are dispatched, and `compute_smooth_constants!` fills
the same `SmoothConstants` slots from either model — so what needs its own path
is the M-step, where a symplectic parameterization is not a linear regression.

The optimizer itself needs nothing new: `_HamMStepCtx` reads only the statistics,
and its Jacobian coefficient `N_ab` comes from `sum(hs.nk)`, which the weighted
aggregator fills with the *effective* count `n̄ₖ = Σ γₖ(t)`. So the weighted
generalized M-step is the unweighted one fed weighted statistics.
=============================================================================#

"""
    _slds_aggregate_weighted!(suf, tfs, lds, data, weights, sws)

One discrete state's responsibility-weighted sufficient statistics.

A Hamiltonian state needs two passes: the base regression/emission blocks that
every model shares, and the mixed-coordinate blocks its own M-step consumes.
"""
function _slds_aggregate_weighted!(
    suf,
    tfs::TrialFilterSmooth{T},
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T},
    weights::AbstractVector{<:AbstractVector{T}},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:AbstractObservationModel{T}}
    return _aggregate_td_suff_stats_weighted!(suf, tfs, lds, data, weights, sws)
end

"""
    _slds_init_suf(suf)

The statistics carrying the initial-state blocks (`init_xy`, `init_yy`,
`init_n`). A Hamiltonian model keeps them on its base statistics, alongside the
mixed-coordinate blocks its own M-step uses.
"""
_slds_init_suf(suf) = suf

"""
    _slds_state_mstep!(ldss, sf_state, tied, slots_q, sws, bufs, K, D, ux_dim) -> slots_ab

The dynamics half of the SLDS M-step, dispatched on the state model.

A Gaussian state is the conjugate `[A b B]` regression followed by the `Q`
update. A Hamiltonian state is the constrained generalized M-step — L-BFGS on
the symplectic parameterization, accepted only when it improves — then the
closed-form noise update, which is what the ungrouped Hamiltonian fit does.

Returns the regression-version slots the caller uses for its `Q` bookkeeping.
Hamiltonian states do their own noise update here, so they return the identity
and the caller skips `_grouped_update_Q!`.
"""
function _slds_state_mstep!(
    ldss::AbstractVector{<:LinearDynamicalSystem{T,S,O}},
    sf_state::AbstractVector,
    tied::AbstractVector{Symbol},
    slots_q::AbstractVector{Int},
    sws::SmoothWorkspace{T},
    bufs,
    K::Int,
    D::Int,
    ux_dim::Int,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:AbstractObservationModel{T}}
    dyn_cols = _tied_dyn_cols(tied, D, ux_dim)
    slots_ab = _slds_update_regression!(
        _DynBlock(), ldss, sf_state, dyn_cols, slots_q, sws, bufs, K
    )
    _grouped_update_Q!(ldss, sf_state, slots_q, slots_ab, sws)
    return slots_ab
end

function mstep!(
    slds::SLDS{T,S,O},
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple},
    sws::SmoothWorkspace{T};
    obs_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}},NamedTuple}=nothing,
    tied::AbstractVector{Symbol}=Symbol[],
    sws_pool::Vector{SmoothWorkspace{T}}=[sws],
    ntasks::Int=1,
    data::Union{Nothing,Data{T}}=nothing,
    sufs::Union{Nothing,AbstractVector}=nothing,
    bufs::Union{Nothing,GroupedSufBuffers{T},NamedTuple}=nothing,
    init_scratch::Union{Nothing,LinearDynamicalSystem}=nothing,
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    K = length(slds.LDSs)
    ntrials = _ntrials(y)

    # Discrete-layer M-step (slds.A, slds.πₖ are updated in place via dl).
    StatsAPI.fit!(dl, fb_storage, obs_seq; seq_ends=seq_ends)

    #=
    `Data` canonicalizes absent ux/uy to zero-row matrices and validates the
    supplied ones. All regimes share the same input dims (enforced by
    `validate_SLDS`), so one `Data` serves every `lds_k`. `fit!` builds it once
    at entry and passes it in — the shapes cannot change between iterations, so
    re-validating every trial each M-step is pure overhead.
    =#
    dat = data === nothing ? Data(slds.LDSs[1], y; ux=ux, uy=uy) : data

    function weights_of(k)
        return [
            begin
                t1, t2 = HMMs.seq_limits(seq_ends, trial)
                view(fb_storage.γ, k, t1:t2)
            end for trial in 1:ntrials
        ]
    end

    #=
    One γ-weighted sufficient statistic per regime, all built before any update
    runs: a tied group is fitted from several regimes at once, and `Q` / `R`
    read the regression that was just written, so the updates cannot be
    interleaved with the aggregation the way a fully per-regime M-step can.
    =#
    sf = if sufs === nothing
        [_initialize_td_sufficient_statistics(T, slds.LDSs[1], dat.tsteps) for _ in 1:K]
    else
        sufs
    end
    #=
    The `K` aggregations write into disjoint `sufs[k]` and read `tfs` / `data`
    only, so they run in parallel — each on its own workspace, since the
    aggregator uses `sws` as scratch before copying the result out. `weights_of`
    is called inside the task so each regime's view vector is built there rather
    than K times up front.

    This is the axis that stays useful when `ntrials < nthreads`: the weighted
    aggregator has no per-trial parallelism of its own, so without this it is K
    sequential passes over every trial.
    =#
    let n = max(1, min(K, length(sws_pool)))
        chunk = cld(K, n)
        tforeach(1:n) do i
            local lo, hi
            lo = (i - 1) * chunk + 1
            hi = min(i * chunk, K)
            lo > hi && return nothing
            for k in lo:hi
                _slds_aggregate_weighted!(
                    sf[k], tfs, slds.LDSs[k], dat, weights_of(k), sws_pool[i]
                )
            end
            return nothing
        end
    end

    lds1 = slds.LDSs[1]
    D = lds1.latent_dim

    slots_q = _tie_slots(:Q in tied, K)
    #=
    The state-side updates read the state blocks, which for a composite emission
    are carried (identically) on every member's statistics.
    =#
    sf_state = _state_sufs(sf)
    bf = bufs === nothing ? _grouped_suf_buffers(lds1, dat.tsteps) : bufs

    #=
    `[A b B]` then `Q`, `[C d D]` then `R`: the covariance updates read the
    regression that was just written. The returned slots also tell those updates
    how many distinct regressions there are, so a partial tie counts as `K` of
    them — every regime's stacked matrix differs, in its free columns.
    =#
    _slds_state_mstep!(
        slds.LDSs, sf_state, tied, slots_q, sws, _state_bufs(bf), K, D, lds1.ux_dim
    )

    #=
    The emission half reads the base regression blocks. A Hamiltonian model wraps
    those alongside its own mixed-coordinate blocks, so unwrap before handing them
    over; `_obs_suf` is the identity for every other model.
    =#
    _slds_obs_mstep!(
        lds1.obs_model,
        slds,
        _obs_sufs(sf),
        tfs,
        dat,
        tied,
        sws,
        bf,
        K,
        weights_of;
        ntasks=ntasks,
    )

    _broadcast_tied_params!(slds, tied)

    #=
    x0/P0 are tied across modes. Since the smoother gives one q(x) per trial
    and Σₖ γₖ(t=1) = 1, the pooled unit-weight init stats are exactly the sum
    over modes of the per-mode init stats the aggregator already computed.
    =#
    D = slds.LDSs[1].latent_dim
    suf = _slds_init_suf(sf_state[1])
    init_xy = zeros(T, 1, D)
    init_yy = zeros(T, D, D)
    init_n = zero(T)
    for k in 1:K
        base_k = _slds_init_suf(sf_state[k])
        init_xy .+= base_k.init_xy
        init_yy .+= base_k.init_yy[]
        init_n += T(base_k.init_n)
    end
    copyto!(suf.init_xy, init_xy)
    suf.init_yy[] = init_yy
    suf.init_n = init_n
    _update_shared_initial_state!(slds, suf, sws; scratch=init_scratch)

    return nothing
end

"""
    _slds_obs_mstep!(obs_model, slds, sf, tfs, data, tied, sws, bufs, K, weights_of; ntasks)

The emission half of the SLDS M-step, dispatched on the observation model.

A Gaussian emission is the conjugate regression plus the IW update for `R`; a
Poisson one is the non-conjugate Newton solve. A composite runs whichever of
those each member calls for, on that member's per-regime views, statistics,
sub-workspaces and `tied_params` names — so one emission can be tied across
regimes while another is fitted per regime.
"""
function _slds_obs_mstep!(
    ::GaussianObservationModel,
    slds::SLDS{T},
    sf::AbstractVector,
    ::TrialFilterSmooth{T},
    ::Data{T},
    tied::AbstractVector{Symbol},
    sws::SmoothWorkspace{T},
    bufs,
    K::Int,
    weights_of;
    ntasks::Int=1,
) where {T<:Real}
    _slds_gaussian_obs_mstep!(slds.LDSs, sf, tied, nothing, sws, _state_bufs(bufs), K)
    return nothing
end

function _slds_obs_mstep!(
    ::PoissonObservationModel,
    slds::SLDS{T},
    ::AbstractVector,
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    tied::AbstractVector{Symbol},
    sws::SmoothWorkspace{T},
    ::Any,
    ::Int,
    weights_of;
    ntasks::Int=1,
) where {T<:Real}
    lds1 = slds.LDSs[1]
    obs_cols = _tied_obs_cols(tied, lds1.latent_dim, lds1.uy_dim)
    _tied_poisson_emission!(
        slds.LDSs,
        tfs,
        data.y,
        data.uy,
        sws,
        weights_of,
        length(obs_cols) == lds1.latent_dim + 1 + lds1.uy_dim;
        ntasks=ntasks,
    )
    return nothing
end

function _slds_obs_mstep!(
    om::CompositeObservationModel,
    slds::SLDS{T},
    sf::AbstractVector,
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    tied::AbstractVector{Symbol},
    sws::SmoothWorkspace{T},
    bufs::NamedTuple,
    K::Int,
    weights_of;
    ntasks::Int=1,
) where {T<:Real}
    subs = _obs_workspaces!(sws, slds.LDSs[1])
    datas = _member_datas(data)
    for (m, key) in enumerate(_obs_keys(om))
        views = [_obs_view(lds, key) for lds in slds.LDSs]
        _slds_member_obs_mstep!(
            _models(om)[key],
            views,
            [s[key] for s in sf],
            tfs,
            datas[key],
            tied,
            key,
            subs[m],
            bufs[key],
            K,
            weights_of;
            ntasks=ntasks,
        )
    end
    return nothing
end

function _slds_member_obs_mstep!(
    ::GaussianObservationModel,
    views::AbstractVector,
    sufs::AbstractVector,
    ::TrialFilterSmooth{T},
    ::Data{T},
    tied::AbstractVector{Symbol},
    key::Symbol,
    sws::SmoothWorkspace{T},
    bufs::GroupedSufBuffers{T},
    K::Int,
    weights_of;
    ntasks::Int=1,
) where {T<:Real}
    _slds_gaussian_obs_mstep!(views, sufs, tied, key, sws, bufs, K)
    return nothing
end

function _slds_member_obs_mstep!(
    ::PoissonObservationModel,
    views::AbstractVector,
    ::AbstractVector,
    tfs::TrialFilterSmooth{T},
    data_m::Data{T},
    tied::AbstractVector{Symbol},
    key::Symbol,
    sws::SmoothWorkspace{T},
    ::GroupedSufBuffers{T},
    ::Int,
    weights_of;
    ntasks::Int=1,
) where {T<:Real}
    lds1 = views[1]
    obs_cols = _tied_obs_cols(tied, lds1.latent_dim, lds1.uy_dim, key)
    _tied_poisson_emission!(
        views,
        tfs,
        data_m.y,
        data_m.uy,
        sws,
        weights_of,
        length(obs_cols) == lds1.latent_dim + 1 + lds1.uy_dim;
        ntasks=ntasks,
    )
    return nothing
end

# `[C d D]` then `R`, over one observation model's per-regime views.
function _slds_gaussian_obs_mstep!(
    ldss::AbstractVector,
    sufs::AbstractVector,
    tied::AbstractVector{Symbol},
    key::Union{Nothing,Symbol},
    sws::SmoothWorkspace{T},
    bufs::GroupedSufBuffers{T},
    K::Int,
) where {T<:Real}
    lds1 = ldss[1]
    obs_cols = _tied_obs_cols(tied, lds1.latent_dim, lds1.uy_dim, key)
    slots_r = _tie_slots(_tied_name(:R, key) in tied, K)
    slots_cd = _slds_update_regression!(
        _ObsBlock(), ldss, sufs, obs_cols, slots_r, sws, bufs, K
    )
    _grouped_update_R!(ldss, sufs, slots_r, slots_cd, sws)
    return nothing
end

"""
    _update_shared_initial_state!(slds, suf, sws; scratch=nothing)

Fit the single initial-state distribution `N(x0, P0)` shared by all SLDS modes
from the pooled init stats in `suf` (see `mstep!`) and copy it into every
`state_model`.

The update routines write through an `LinearDynamicalSystem`, so this needs one
to write into that is not a regime. `scratch` supplies it; `fit!` allocates it
once at entry rather than `deepcopy`ing a whole sub-model — priors, emission
matrices and all — on every EM iteration.
"""
function _update_shared_initial_state!(
    slds::SLDS{T},
    suf::SufficientStatistics{T},
    sws::SmoothWorkspace{T};
    scratch::Union{Nothing,LinearDynamicalSystem}=nothing,
) where {T<:Real}
    lds1 = scratch === nothing ? deepcopy(slds.LDSs[1]) : scratch
    #=
    Only `x0` / `P0` are read back out, but the covariance update's scatter
    reads `x0`, so the scratch must start from the current regime's values.
    =#
    if scratch !== nothing
        copyto!(lds1.state_model.x0, slds.LDSs[1].state_model.x0)
        copyto!(lds1.state_model.P0, slds.LDSs[1].state_model.P0)
    end
    update_initial_state_mean!(lds1, suf)
    update_initial_state_covariance!(lds1, suf, sws)
    fit_x0, fit_P0 = lds1.fit_bool[1], lds1.fit_bool[2]
    for k in eachindex(slds.LDSs)
        fit_x0 && copyto!(slds.LDSs[k].state_model.x0, lds1.state_model.x0)
        fit_P0 && copyto!(slds.LDSs[k].state_model.P0, lds1.state_model.P0)
    end
    return nothing
end

"""
    fit!(slds::SLDS, y; ux=nothing, uy=nothing, max_iter=50, smoothing_iters=1, progress=true)

Fit SLDS using variational Laplace EM. Runs for exactly `max_iter` iterations
(no early-stopping criterion: the E-step's posterior sampling makes the ELBO
trace noisy across iterations, so a tolerance check on successive differences
would fire spuriously). Returns the per-iteration ELBO trace.

Each E-step runs `smoothing_iters` discrete↔continuous alternations before the
M-step. The default of 1 is the standard vLEM update; larger values hand the
M-step a better-converged posterior at proportional cost per iteration.

`y` is a single trial `(obs_dim × T)` matrix, a `(obs_dim, T, ntrials)` array,
or a vector of per-trial matrices (ragged `T_i` allowed). Internally a single
batched `HMMs.ForwardBackwardStorage` of length `sum(T_i)` is allocated, with
`seq_ends = cumsum(T_i)` to demarcate trials.

Optional control inputs `ux` / `uy` accept the same shape family as `y`
(`nothing` when the regimes carry no `B` / `D`). They are shared across regimes
— the active regime `zₜ` selects which per-regime `Bₖ` / `Dₖ` multiplies the
input — and are re-estimated per regime alongside the other parameters. The
input dimensions must match across regimes (enforced by `validate_SLDS`).

Pass `depends_on` (a `NamedTuple` of per-trial label vectors) to override the
`depends_on` declared on the regimes' sub-models for this call. Every regime
must declare the same labels — the grouping of trials is a property of the data
— and `x0`/`P0` stay tied across regimes as usual.

Pass `tied_params` — a `Symbol` or a collection of them — to share parameters
across regimes instead of fitting one per regime. Names are the literal
parameter names `depends_on` and `fit_bool` use, and each means itself: `:C` is
`C`, not `[C d D]`. `tied_params = (:C, :d, :D, :R)` is the usual setup for
neural data — the recording does not change when the dynamics do, so only
`[A b B Q]` and the discrete chain switch, and the emission's parameter count is
divided by `K`. `tied_params = (:A, :b, :B, :Q)` is the mirror image: one set of
dynamics, switching emissions.

`:x0` / `:P0` are accepted and ignored — an SLDS ties its initial state across
regimes unconditionally. Combined with `depends_on` the tie is *within* a group:
each session keeps its own version, shared by every regime. Tied parameters are
broadcast before the first E-step, so no regime ever infers `q(x)` / `q(z)`
through a value the model does not have, and a frozen group (`fit_bool`) is left
exactly as the caller set it.

`[A b B]` and `[C d D]` are each fitted as one regression, so how much of one
you tie decides the cost. Tying it together with its covariance is the ordinary
pooled M-step (`O(m³)`), and so is tying a covariance on its own. Tying a
regression while its covariance still switches makes the shared fit a
generalized least-squares problem coupling the output rows (`O((p·m)³)`), and
tying only *part* of one adds a Frisch–Waugh projection in front of that. Both
are exact; tie the covariance alongside its regression when you can.

A partial tie has no reduction for a Poisson `[C d D]`, which is fitted by
LBFGS rather than from sufficient statistics, or alongside `depends_on`, which
already splits the regression per group of trials — those throw rather than
guess.

# Parallelism and reproducibility

Every per-trial pass — the smoother, the discrete layer's log-likelihood fill,
and the ELBO — runs its trials in chunks across a pool of `npool` workspaces,
one per thread by default and capped at the trial count. A chunk owns its slot
for the whole pass, so nothing is shared between concurrently running trials.

`npool` also sets the memory: each slot holds `O(D²·T)` block-tridiagonal
storage and, for a Poisson emission, `O(N·T)` batched scratch. Lower it to trade
throughput for memory; `npool = 1` runs every pass sequentially.

`rng_mode` decides how the E-step's per-trial posterior draw is sourced, which
is the only place parallelism could change results:

- `:trial` (default) gives each trial its own generator, seeded from `rng` and
  the trial index. The draw a trial receives is then a function of `rng` alone —
  identical across thread counts, across `npool`, and between the parallel and
  sequential paths.
- `:global` draws the standard normals from `rng` serially in trial order before
  each pass, reproducing the exact stream the sequential smoother consumed. Use
  it to return to an existing fit; the pass itself still runs in parallel.

What is guaranteed, in either mode: **the fit does not depend on the thread
count**. Every per-trial result — `x_smooth`, `p_smooth`, the responsibilities,
`dl.logL` — is written by exactly one trial, so those are bit-identical however
the chunks are scheduled, and the ELBO is summed in trial order so it is too.

What `npool` does change, at rounding level: the Poisson emission M-step sums
its curvature and gradient over chunks, so a different chunk count is a
different summation order. The effect is ~1e-11 relative after tens of EM
iterations. Pin `npool` alongside `rng` to reproduce a fit exactly; leave it at
the default for the best throughput on whatever machine is running.

# Held-out scoring

Pass `y_test` (with `ux_test` / `uy_test` / `depends_on_test` as needed) to
score a held-out set every `test_every` iterations, at the same parameters the
training ELBO was just evaluated at. `fit!` then returns a [`FitTrace`](@ref),
which behaves exactly as the training-ELBO vector it replaces and carries
`.test`, `.test_iters` and `.best_iter` alongside — `best_iter` is the answer
to "how many iterations before this starts overfitting?".

Set `early_stopping=true` to stop when the held-out ELBO turns over (`patience`
consecutive non-improving scores, defaulting to 1 — the first decrease; and
`min_delta` for how much counts as an improvement). `restore_best=true` (the
default) then rolls the model back to the best-scoring parameters; a fit that
runs to completion is always left at its final iterate. `test_kwargs` forwards
extra keywords to the scoring [`elbo`](@ref) call, e.g.
`(smoothing_iters=20,)` to make each SLDS check cheaper.

The held-out score comes from the public [`elbo`](@ref), whose coordinate
ascent is deterministic — unlike the fit's own Monte-Carlo E-step. The two
traces are therefore produced by different inference routines and are not
comparable in absolute terms; each is comparable to itself across iterations,
which is what the overfitting question needs.
"""
function fit!(
    slds::SLDS{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    ux=nothing,
    uy=nothing,
    max_iter::Int=50,
    smoothing_iters::Int=1,
    progress::Bool=true,
    rng::AbstractRNG=Random.default_rng(),
    rng_mode::Symbol=:trial,
    npool::Int=Threads.maxthreadid(),
    depends_on::Union{Nothing,NamedTuple}=nothing,
    tied_params=nothing,
    y_test=nothing,
    ux_test=nothing,
    uy_test=nothing,
    depends_on_test::Union{Nothing,NamedTuple}=nothing,
    test_every::Int=1,
    early_stopping::Bool=false,
    patience::Int=1,
    min_delta::Real=0.0,
    restore_best::Bool=true,
    test_kwargs::NamedTuple=NamedTuple(),
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    rng_mode in (:trial, :global) ||
        throw(ArgumentError("rng_mode must be :trial or :global, got $(repr(rng_mode))"))
    monitor = _holdout_monitor(
        T,
        y_test;
        ux_test=ux_test,
        uy_test=uy_test,
        depends_on_test=depends_on_test,
        test_every=test_every,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        restore_best=restore_best,
        test_kwargs=test_kwargs,
    )
    tied = _resolve_tied_params(
        slds.LDSs[1].state_model, slds.LDSs[1].obs_model, tied_params
    )
    #=
    `Data` centralizes shape validation and canonicalizes the three
    observation/input forms (regime dims are uniform, so validating against
    LDSs[1] covers all regimes). Absent ux/uy become zero-row matrices.
    =#
    data = Data(slds.LDSs[1], y; ux=ux, uy=uy)
    _prepare_slds!(slds, data.tsteps)
    y_seq = data.y
    ux_seq = data.ux
    uy_seq = data.uy

    K = length(slds.LDSs)
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = _ws_obs_dim(slds.LDSs[1])

    tsteps_per_trial = data.tsteps
    ntrials = length(tsteps_per_trial)
    seq_ends = cumsum(tsteps_per_trial)
    total_T = last(seq_ends)
    T_max = maximum(tsteps_per_trial)

    #=
    Ancillary parameter dependencies. `grp === nothing` (no regime declares
    `depends_on`) keeps every step on its original code path.
    =#
    grp = _slds_parameter_grouping(slds, ntrials; depends_on=depends_on, y=y_seq)
    cell_slds = grp === nothing ? nothing : _slds_cell_sldss(slds, grp)
    _validate_tied_params(slds.LDSs[1], tied, grp !== nothing)

    # Continuous-state smoother storage (per-trial sized).
    tfs = initialize_FilterSmooth(slds.LDSs[1], tsteps_per_trial)::TrialFilterSmooth{T}

    # Discrete-layer wrapper (logL sized for the batched timestep sequence).
    dl = SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(T, K, total_T))

    # Single batched fb_storage covering all trials.
    fb_storage = _make_slds_fb_storage(dl, seq_ends)

    # Cached batched HMMs.jl placeholder sequences (timestep indices / nothings).
    obs_seq = collect(1:total_T)
    control_seq = fill(nothing, total_T)

    # Workspaces — allocated once at max trial length, reused each iteration.
    # `sws` sizes its regression buffers for the (uniform) input dims so the
    # weighted aggregator can fit `[Aₖ bₖ Bₖ]` / `[Cₖ dₖ Dₖ]`.
    sws = SmoothWorkspace(
        T,
        latent_dim,
        obs_dim,
        T_max;
        ux_dim=slds.LDSs[1].ux_dim,
        uy_dim=slds.LDSs[1].uy_dim,
    )
    #=
    One SLDS workspace per task slot (plus the per-cell workspaces a ragged
    stitching fit needs), and the fixed chunking of trials over them that every
    per-trial pass reuses.
    =#
    pool = _slds_workspace_pool(slds, cell_slds, T_max, ntrials; npool=npool)
    plan = _slds_trial_plan(grp, ntrials, length(pool.slots))
    #=
    The Poisson `-Σ log(y!)` normalizer, once per trial. Constant in the
    latents, so the Newton line search must never recompute it; `nothing` for a
    Gaussian emission, which has no data-only normalizer.
    =#
    lognorm = _slds_lognorm_all(slds, y_seq)
    #=
    The M-step's regression buffers are shaped by `obs_dim` too, so a stitching
    fit needs one per cell. `_cell_workspace` shares the block-tridiagonal and
    smoothed-covariance storage, and `nothing` here keeps the single-workspace
    path for every fit whose cells have the parent's width.
    =#
    cell_mstep_sws = _slds_cell_mstep_workspaces(slds, cell_slds, sws, T_max)
    #=
    M-step scratch. `mstep_tasks` is capped at the number of aggregations there
    are to run — `K` ungrouped, `K · ncells` grouped — because that is the only
    axis these workspaces parallelise; the emission solve chunks its own trials
    and takes `ntasks` instead, which is the thread budget.
    =#
    mstep_units = grp === nothing ? K : K * grp.ncells
    mstep_tasks = max(1, min(npool, mstep_units))
    sws_pool = _slds_mstep_pool(slds, T_max, mstep_tasks)
    sws_pool[1] = sws
    cell_mstep_pools = _slds_cell_mstep_pools(slds, cell_slds, sws_pool, T_max)
    #=
    M-step objects that depend only on shapes, hoisted out of the EM loop: the
    trial shapes cannot change between iterations, so rebuilding `Data` (which
    re-validates every trial), the per-regime sufficient statistics and the
    grouped regression buffers each M-step was pure overhead. `init_scratch`
    replaces a per-iteration `deepcopy` of a whole sub-model.
    =#
    mstep_sufs = if grp === nothing
        [_initialize_td_sufficient_statistics(T, slds.LDSs[1], tsteps_per_trial) for _ in 1:K]
    else
        nothing
    end
    #=
    Built from cell 1's sub-model on the grouped path, not the parent's: under
    stitching the cells differ in channel count, and this is the model the
    grouped M-step sizes its default statistics from.
    =#
    mstep_bufs = _grouped_suf_buffers(slds.LDSs[1], data.tsteps)
    init_scratch = deepcopy(slds.LDSs[1])
    # Per-cell slices of the data and smoother storage, fixed by the partition.
    cell_views = if grp === nothing
        nothing
    else
        (
            [_subset_data(data, grp.cell_trials[c]) for c in 1:(grp.ncells)],
            [
                TrialFilterSmooth([tfs[n] for n in grp.cell_trials[c]]) for
                c in 1:(grp.ncells)
            ],
        )
    end
    x_samples = [Matrix{T}(undef, latent_dim, Ti) for Ti in tsteps_per_trial]
    # Pre-drawn standard normals, only when `:global` reproducibility is asked for.
    noise_bufs = rng_mode === :global ? _slds_noise_buffers(x_samples) : nothing

    #=
    Broadcast the tied groups before the first E-step rather than only after the
    first M-step, so no regime ever infers `q(x)` / `q(z)` through a parameter
    the model does not have. Regimes seeded from one warm start already agree,
    and this makes that a property of the fit instead of an accident of the
    caller.
    =#
    if cell_slds === nothing
        _broadcast_tied_params!(slds, tied)
    else
        for slds_c in cell_slds
            _broadcast_tied_params!(slds_c, tied)
        end
    end

    prog = if progress
        Progress(max_iter; desc="Fitting SLDS via EM...", barlen=50, showspeed=true)
    else
        nothing
    end
    elbos = Vector{T}(undef, max_iter)

    #=
    Warm-start: smooth each trial once with uniform weights, drawing the first
    sample into x_samples for the first E-step to consume.
    =#
    _slds_warmstart!(
        slds,
        cell_slds,
        grp,
        tfs,
        y_seq,
        x_samples,
        pool,
        plan,
        tsteps_per_trial,
        K;
        rng=rng,
        rng_mode=rng_mode,
        noise_bufs=noise_bufs,
        ux=ux_seq,
        uy=uy_seq,
        lognorm=lognorm,
    )

    for iter in 1:max_iter
        #=
        E-step: fill q(z) from the current samples, run forward-backward,
        re-smooth q(x), and draw the next samples for the following iteration.
        =#
        if grp === nothing
            estep!(
                slds,
                tfs,
                fb_storage,
                dl,
                y_seq,
                x_samples,
                pool,
                plan;
                rng=rng,
                rng_mode=rng_mode,
                noise_bufs=noise_bufs,
                obs_seq=obs_seq,
                control_seq=control_seq,
                seq_ends=seq_ends,
                ux=ux_seq,
                uy=uy_seq,
                lognorm=lognorm,
                smoothing_iters=smoothing_iters,
            )

            # Compute the ELBO at the current posteriors.
            elbos[iter] = elbo!(
                slds,
                tfs,
                fb_storage,
                y_seq,
                pool,
                plan;
                seq_ends=seq_ends,
                ux=ux_seq,
                uy=uy_seq,
                lognorm=lognorm,
            )

            #=
            Held-out score at the same parameters the training ELBO just used.
            Stopping here, before the M-step, leaves the model exactly at the
            scored parameters when `restore_best` is off.
            =#
            _holdout_due(monitor, iter) && _holdout_record!(monitor, slds, iter)
            if _holdout_stop(monitor)
                prog !== nothing && finish!(prog)
                resize!(elbos, iter)
                return _fit_result(monitor, elbos, slds)
            end

            # M-step: update discrete and continuous parameters.
            mstep!(
                slds,
                tfs,
                fb_storage,
                dl,
                y_seq,
                sws;
                obs_seq=obs_seq,
                seq_ends=seq_ends,
                ux=ux_seq,
                uy=uy_seq,
                tied=tied,
                sws_pool=sws_pool,
                ntasks=npool,
                data=data,
                sufs=mstep_sufs,
                bufs=mstep_bufs,
                init_scratch=init_scratch,
            )
            # Every slot, not just the first: the ungrouped passes read the
            # cached constants without refreshing them.
            refresh_slds_pool!(pool, slds)
        else
            grouping = grp::ParameterGrouping
            cells = cell_slds::Vector
            _estep_grouped!(
                cells,
                grouping,
                tfs,
                fb_storage,
                dl,
                y_seq,
                x_samples,
                pool,
                plan;
                rng=rng,
                rng_mode=rng_mode,
                noise_bufs=noise_bufs,
                obs_seq=obs_seq,
                control_seq=control_seq,
                seq_ends=seq_ends,
                ux=ux_seq,
                uy=uy_seq,
                lognorm=lognorm,
                smoothing_iters=smoothing_iters,
            )

            elbos[iter] = _elbo_grouped!(
                cells,
                grouping,
                tfs,
                fb_storage,
                y_seq,
                pool,
                plan;
                seq_ends=seq_ends,
                ux=ux_seq,
                uy=uy_seq,
                lognorm=lognorm,
            )

            #=
            Held-out score at the same parameters the training ELBO just used.
            Stopping here, before the M-step, leaves the model exactly at the
            scored parameters when `restore_best` is off.
            =#
            _holdout_due(monitor, iter) && _holdout_record!(monitor, slds, iter)
            if _holdout_stop(monitor)
                prog !== nothing && finish!(prog)
                resize!(elbos, iter)
                return _fit_result(monitor, elbos, slds)
            end

            _mstep_grouped!(
                cells,
                grouping,
                tfs,
                fb_storage,
                dl,
                data,
                sws;
                obs_seq=obs_seq,
                seq_ends=seq_ends,
                cell_sws=cell_mstep_sws,
                tied=tied,
                sws_pool=sws_pool,
                cell_sws_pools=cell_mstep_pools,
                ntasks=npool,
                bufs=mstep_bufs,
                cell_views=cell_views,
            )
        end

        prog !== nothing && next!(prog)
    end

    if prog !== nothing
        finish!(prog)
    end
    return _fit_result(monitor, elbos, slds)
end

# ============================================================================
# Ancillary parameter dependencies (`depends_on`) for the SLDS.
#
# The trial partition is a property of the *dataset*, so it must be identical
# across regimes; only the parameter values differ per regime. Given that, each
# cell is governed by one ordinary `SLDS` whose sub-LDSs hold that cell's
# parameter arrays, and the existing smoother / weighted aggregator run on it
# unchanged. Regime constants are refreshed once per cell rather than once per
# trial, so the per-cell overhead is K Cholesky sets per pass.
# ============================================================================

"""
    _slds_warmstart!(slds, cell_slds, grp, tfs, y, x_samples, pool, plan, tsteps, K; ...)

Smooth every trial once with uniform discrete weights `γ ≡ 1/K`, so the first
discrete update has a continuous trajectory to score. `x_samples` receives the
first posterior draw the Monte-Carlo E-step consumes; pass `nothing` for the
deterministic path of [`smooth`](@ref), which scores the smoothed mean instead.
"""
function _slds_warmstart!(
    slds::SLDS{T},
    cell_slds::Union{Nothing,AbstractVector},
    grp::Union{Nothing,ParameterGrouping},
    tfs::TrialFilterSmooth{T},
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple},
    x_samples::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}},
    pool::SLDSWorkspacePool{T},
    plan::SLDSTrialPlan,
    tsteps::AbstractVector{Int},
    K::Int;
    rng::AbstractRNG=Random.default_rng(),
    rng_mode::Symbol=:trial,
    noise_bufs::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}},NamedTuple}=nothing,
    lognorm::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real}
    function w_of(trial)
        return fill(one(T) / K, K, tsteps[trial])
    end

    # The warm start's draw is simply the first one off `rng`.
    rng_of, noise_of = _slds_draw_sources(rng, rng_mode, x_samples, noise_bufs)

    _slds_smooth_all!(
        slds,
        cell_slds,
        grp,
        tfs,
        y,
        x_samples,
        pool,
        plan,
        w_of;
        rng_of=rng_of,
        noise_of=noise_of,
        ux=ux,
        uy=uy,
        lognorm=lognorm,
    )
    return nothing
end

"""
    _slds_parameter_grouping(slds, ntrials; depends_on=nothing)

Trial partition for an `SLDS`, or `nothing` when no regime declares
`depends_on`. Throws when the regimes disagree about the partition.
"""
function _slds_parameter_grouping(
    slds::SLDS, ntrials::Int; depends_on::Union{Nothing,NamedTuple}=nothing, y=nothing
)
    grp = parameter_grouping(slds.LDSs[1], ntrials; depends_on=depends_on, y=y)
    grp === nothing && return nothing
    for k in 2:length(slds.LDSs)
        grp_k = parameter_grouping(slds.LDSs[k], ntrials; depends_on=depends_on, y=y)
        ok =
            grp_k !== nothing &&
            grp_k.nslots == grp.nslots &&
            grp_k.cell_state == grp.cell_state &&
            grp_k.cell_obs == grp.cell_obs &&
            grp_k.trial_cell == grp.trial_cell
        ok || throw(
            ArgumentError(
                "SLDS: every regime must declare the same `depends_on` labels; regime " *
                "$k disagrees with regime 1. The grouping of trials is a property of " *
                "the data and is shared across regimes — only the fitted parameter " *
                "values differ per regime.",
            ),
        )
    end
    return grp
end

"""
    _slds_cell_sldss(slds, grp) -> Vector{SLDS}

One `SLDS` view per cell, sharing `A` and `πₖ` by reference (so the discrete
M-step still updates the single shared chain) and holding each regime's
per-cell parameter arrays.
"""
function _slds_cell_sldss(
    slds::SLDS{T,S,O,TM,ISV}, grp::ParameterGrouping
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel,TM,ISV}
    K = length(slds.LDSs)
    return [
        SLDS{T,S,O,TM,ISV}(slds.A, slds.πₖ, [_cell_lds(slds.LDSs[k], grp, c) for k in 1:K])
        for c in 1:(grp.ncells)
    ]
end

"""
    _cell_slds_workspace(base, slds_c, tsteps) -> SLDSSmoothWorkspace

One cell's SLDS workspace for a stitching fit. Reuses `base`'s
block-tridiagonal storage, per-timestep log-density scratch, and emission-
curvature scratch — the parts sized by `latent_dim` and `tsteps` alone, none of
which depends on `obs_dim` — and allocates fresh per-regime constants and Newton
buffers at this cell's channel count.

Safe for the same reason the LDS side is: cells run one at a time, and a cell's
Hessian blocks are consumed before the next cell overwrites them.
"""
function _cell_slds_workspace(
    base::SLDSSmoothWorkspace{T}, slds_c::SLDS, tsteps::Int
) where {T<:Real}
    lds1 = slds_c.LDSs[1]
    latent_dim = lds1.latent_dim
    obs_dim = _ws_obs_dim(lds1)
    K = length(slds_c.LDSs)
    ws = SLDSSmoothWorkspace{T}(
        base.btd,                                          # shared O(D²·T)
        [SmoothConstants(T, latent_dim, obs_dim) for _ in 1:K],
        NewtonBuffers(T, latent_dim, obs_dim, tsteps),
        base.ll_tmp,                                       # shared, length T_max
        base.H_obs,                                        # shared, latent_dim square
        nothing,                                           # batched Poisson scratch
        _slds_obs_scratch(T, slds_c),                      # per-member emission scratch
    )
    refresh_slds_constants!(ws, slds_c)
    return ws
end

"""
    _slds_mstep_pool(slds, T_max, ntasks) -> Vector{SmoothWorkspace}

Scratch workspaces for the M-step's parallel axes: the `K` (or `K · ncells`)
weighted sufficient-statistic aggregations, which use a workspace as scratch and
so need one each. Capped at the number of aggregations there are to run — more
slots than units would allocate storage nothing writes to.
"""
function _slds_mstep_pool(slds::SLDS{T}, T_max::Int, ntasks::Int) where {T<:Real}
    lds1 = slds.LDSs[1]
    n = max(1, ntasks)
    return [
        SmoothWorkspace(
            T,
            lds1.latent_dim,
            _ws_obs_dim(lds1),
            T_max;
            ux_dim=lds1.ux_dim,
            uy_dim=_ws_uy_dim(lds1),
        ) for _ in 1:n
    ]
end

"""
    _slds_cell_mstep_pools(slds, cell_slds, sws_pool, tsteps) -> Vector or nothing

One M-step workspace pool per cell, parallel to `sws_pool`, so a unit
aggregation running on task `i` for cell `c` gets scratch at that cell's channel
width. `nothing` when every cell has the parent's width, which keeps the
uniform fit on `sws_pool` itself.
"""
function _slds_cell_mstep_pools(
    slds::SLDS,
    cell_slds::Union{Nothing,AbstractVector},
    sws_pool::Vector{SmoothWorkspace{T}},
    tsteps::Int,
) where {T<:Real}
    cell_slds === nothing && return nothing
    lds1 = slds.LDSs[1]
    all(sc -> sc.LDSs[1].obs_dim == lds1.obs_dim, cell_slds) && return nothing
    return [
        [
            _cell_workspace(
                base,
                lds1.latent_dim,
                sc.LDSs[1].obs_dim,
                tsteps;
                ux_dim=lds1.ux_dim,
                uy_dim=lds1.uy_dim,
            ) for base in sws_pool
        ] for sc in cell_slds
    ]
end

"""
    _slds_cell_mstep_workspaces(slds, cell_slds, base, tsteps) -> Vector or nothing

One M-step `SmoothWorkspace` per cell, or `nothing` when every cell has the
parent's `obs_dim`. The regression buffers that fit `[Cₖ dₖ Dₖ]` and the
residual scatter `R` accumulates into are both shaped by the cell's channel
count; everything expensive is shared with `base`.
"""
function _slds_cell_mstep_workspaces(
    slds::SLDS,
    cell_slds::Union{Nothing,AbstractVector},
    base::SmoothWorkspace{T},
    tsteps::Int,
) where {T<:Real}
    cell_slds === nothing && return nothing
    lds1 = slds.LDSs[1]
    p0 = lds1.obs_dim
    all(sc -> sc.LDSs[1].obs_dim == p0, cell_slds) && return nothing
    return [
        _cell_workspace(
            base,
            lds1.latent_dim,
            sc.LDSs[1].obs_dim,
            tsteps;
            ux_dim=lds1.ux_dim,
            uy_dim=lds1.uy_dim,
        ) for sc in cell_slds
    ]
end

"""
    _estep_grouped!(cell_slds, grp, tfs, fb_storage, dl, y, x_samples, slds_ws; ...)

Grouped SLDS E-step: `smoothing_iters` alternations of [`_vem_alternate!`](@ref)
with the cell views in play, so each pass over the trials refreshes the regime
constants once per cell. The forward-backward call stays global — the discrete
chain is shared by all trials.
"""
function _estep_grouped!(
    cell_slds::AbstractVector,
    grp::ParameterGrouping,
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple},
    x_samples::AbstractVector{<:AbstractMatrix{T}},
    pool::SLDSWorkspacePool{T},
    plan::SLDSTrialPlan;
    rng::AbstractRNG=Random.default_rng(),
    rng_mode::Symbol=:trial,
    noise_bufs::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}},NamedTuple}=nothing,
    lognorm::Union{Nothing,AbstractVector}=nothing,
    smoothing_iters::Int=1,
) where {T<:Real}
    #=
    Cell 1 stands in for the parent only where `_vem_alternate!` needs a regime
    count and the ungrouped fall-through; every parameter read goes through
    `cell_slds` because `grp` is non-`nothing`.
    =#
    _vem_alternate!(
        cell_slds[1],
        cell_slds,
        grp,
        tfs,
        fb_storage,
        dl,
        y,
        pool,
        plan;
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
        ux=ux,
        uy=uy,
        lognorm=lognorm,
        smoothing_iters=smoothing_iters,
        x_samples=x_samples,
        rng=rng,
        rng_mode=rng_mode,
        noise_bufs=noise_bufs,
    )
    return nothing
end

"""
    _grouped_slds_prior_logdensity(cell_slds, grp, T)

`log p(θ)` for a grouped SLDS: the per-regime terms of
[`_slds_prior_logdensity`](@ref), counted once per distinct parameter version
instead of once per regime.
"""
function _grouped_slds_prior_logdensity(
    cell_slds::AbstractVector, grp::ParameterGrouping, ::Type{T}
) where {T<:Real}
    K = length(cell_slds[1].LDSs)
    total = zero(T)
    for k in 1:K
        ldss = [cell_slds[c].LDSs[k] for c in 1:(grp.ncells)]
        total += _grouped_state_prior_logdensity(ldss, grp.cell_slot, T)
        total += _grouped_obs_prior_logdensity(ldss[1], ldss, grp.cell_slot, T)
    end
    return total
end

"""
    _elbo_grouped!(cell_slds, grp, tfs, fb_storage, y, slds_ws; seq_ends, ux, uy)

Grouped SLDS ELBO: each trial's contribution evaluated against its cell's
parameters, plus one prior term per distinct parameter version.
"""
function _elbo_grouped!(
    cell_slds::AbstractVector,
    grp::ParameterGrouping,
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    y::Union{AbstractVector{<:AbstractMatrix{T}},NamedTuple},
    pool::SLDSWorkspacePool{T},
    plan::SLDSTrialPlan;
    seq_ends::AbstractVector{Int},
    ux::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}},NamedTuple}=nothing,
    lognorm::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real}
    per_trial = _slds_trial_elbos(
        cell_slds[1],
        cell_slds,
        grp,
        tfs,
        fb_storage,
        y,
        pool,
        plan;
        seq_ends,
        ux,
        uy,
        lognorm,
    )
    return sum(per_trial) + _grouped_slds_prior_logdensity(cell_slds, grp, T)
end

"""
    _broadcast_initial_state!(cell_slds, K, do_x0, do_P0)

Copy regime 1's initial-state parameters into every other regime. `x0`/`P0` are
tied across regimes, and the grouped update writes only into regime 1's
variants, so this restores the tie — including before the `P0` update, whose
scatter reads each unit's own `x0`.
"""
function _broadcast_initial_state!(
    cell_slds::AbstractVector, K::Int, do_x0::Bool, do_P0::Bool
)
    (do_x0 || do_P0) || return nothing
    for slds_c in cell_slds
        src = slds_c.LDSs[1].state_model
        for k in 2:K
            dst = slds_c.LDSs[k].state_model
            do_x0 && copyto!(dst.x0, src.x0)
            do_P0 && copyto!(dst.P0, src.P0)
        end
    end
    return nothing
end

"""
    _mstep_grouped!(cell_slds, grp, tfs, fb_storage, dl, data, sws; obs_seq, seq_ends,
                    tied=Symbol[])

Grouped SLDS M-step.

The discrete layer and the trial partition are shared, so the work is one
γ-weighted sufficient statistic per (regime, cell). Every parameter update then
runs over that flat unit list, driven by a slot vector per group:

- a group that is neither grouped nor tied gets one slot per unit — the plain
  per-(regime, cell) fit;
- `depends_on` makes cells sharing a version share a slot *within* a regime;
- naming the group in `tied_params` drops the regime out of the slot, so the
  cells' versions are shared across regimes as well. A tied `[C d D]` is then a
  property of the cell alone: each session keeps its own emission, shared by
  every regime, which is the usual reading for neural data.

Units are laid out regime-major, so the first unit of any version belongs to
regime 1; the update writes there and the broadcasters restore the tie. `x0`/`P0`
are tied across regimes unconditionally.
"""
function _mstep_grouped!(
    cell_slds::AbstractVector,
    grp::ParameterGrouping,
    tfs::TrialFilterSmooth{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    dl::SLDSDiscreteLayer{T},
    data::Data{T},
    sws::SmoothWorkspace{T};
    obs_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    cell_sws::Union{Nothing,AbstractVector}=nothing,
    tied::AbstractVector{Symbol}=Symbol[],
    sws_pool::Vector{SmoothWorkspace{T}}=[sws],
    cell_sws_pools::Union{Nothing,AbstractVector}=nothing,
    ntasks::Int=1,
    bufs::Union{Nothing,GroupedSufBuffers{T},NamedTuple}=nothing,
    cell_views::Union{Nothing,Tuple{<:AbstractVector,<:AbstractVector}}=nothing,
) where {T<:Real}
    K = length(cell_slds[1].LDSs)
    ncells = grp.ncells
    lds1 = cell_slds[1].LDSs[1]

    # Discrete-layer M-step (slds.A, slds.πₖ are updated in place via dl).
    StatsAPI.fit!(dl, fb_storage, obs_seq; seq_ends=seq_ends)

    #=
    Per-cell slices of the data and the smoother storage. They depend only on
    the trial partition, which is fixed for the fit, so `fit!` builds them once
    and passes them in; rebuilding them each M-step re-sliced every trial for
    nothing.
    =#
    cell_data, cell_tfs = if cell_views === nothing
        (
            [_subset_data(data, grp.cell_trials[c]) for c in 1:ncells],
            [TrialFilterSmooth([tfs[n] for n in grp.cell_trials[c]]) for c in 1:ncells],
        )
    else
        cell_views
    end
    bf = bufs === nothing ? _grouped_suf_buffers(lds1, data.tsteps) : bufs

    function γ_view(k, trial)
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        return view(fb_storage.γ, k, t1:t2)
    end

    unit_lds = [cell_slds[c].LDSs[k] for k in 1:K for c in 1:ncells]
    #=
    Sized from the unit's own LDS rather than cell 1's: under stitching each
    cell contributes a different number of channels, so `obs_xy` / `obs_yy`
    differ per unit. Identical to `lds1` whenever the widths agree.
    =#
    unit_suf = [
        _initialize_td_sufficient_statistics(T, cell_slds[c].LDSs[k], cell_data[c].tsteps)
        for k in 1:K for c in 1:ncells
    ]

    #=
    The `K · ncells` unit aggregations write into disjoint `unit_suf[u]`, so
    they run in parallel across `sws_pool` — each task on its own scratch
    workspace, at the cell's own channel width when the fit stitches sessions of
    differing width.
    =#
    let nunits = K * ncells, n = max(1, min(K * ncells, length(sws_pool)))
        chunk = cld(nunits, n)
        tforeach(1:n) do i
            local lo, hi
            lo = (i - 1) * chunk + 1
            hi = min(i * chunk, nunits)
            lo > hi && return nothing
            for u in lo:hi
                k, c = fldmod1(u, ncells)
                _aggregate_td_suff_stats_weighted!(
                    unit_suf[u],
                    cell_tfs[c],
                    unit_lds[u],
                    cell_data[c],
                    [γ_view(k, n2) for n2 in grp.cell_trials[c]],
                    cell_sws_pools === nothing ? sws_pool[i] : cell_sws_pools[c][i],
                )
            end
            return nothing
        end
    end

    #=
    A cell's workspace, indexed by flat unit: `repeat` tiles the per-cell vector
    once per regime, so unit `(k-1)·ncells + c` lands on cell `c`'s entry.
    =#
    unit_sws = cell_sws === nothing ? nothing : repeat(cell_sws, K)

    #=
    A stacked regression is either shared whole across regimes or not at all
    here: `_validate_tied_params` rejects a partial tie alongside `depends_on`,
    which already splits the regression into one version per group of trials.
    =#
    D = lds1.latent_dim
    tie_dyn = length(_tied_dyn_cols(tied, D, lds1.ux_dim)) == D + 1 + lds1.ux_dim

    slots_ab = _grouped_unit_slots(grp.cell_slot[_G_AB], K, tie_dyn)
    slots_q = _grouped_unit_slots(grp.cell_slot[_G_Q], K, :Q in tied)
    # The emission's slots are per observation model; `_grouped_slds_obs_mstep!`
    # takes them from `grp.cell_slot` at each model's own ordinals.
    slots_cd = nothing

    _grouped_update_A_b!(
        unit_lds, _state_sufs(unit_suf), slots_ab, slots_q, sws, _state_bufs(bf)
    )
    _grouped_update_Q!(unit_lds, _state_sufs(unit_suf), slots_q, slots_ab, sws)

    _grouped_slds_obs_mstep!(
        lds1.obs_model,
        unit_lds,
        unit_suf,
        grp,
        K,
        ncells,
        tied,
        sws,
        bf,
        unit_sws,
        data,
        tfs,
        γ_view,
        slots_cd,
        ntasks,
    )

    #=
    Cells sharing a version share its arrays, so copying per cell is idempotent;
    doing it per cell rather than per version keeps this correct for any slot
    layout.
    =#
    for slds_c in cell_slds
        _broadcast_tied_params!(slds_c, tied)
    end

    #=
    Tied initial state, pooled over every (regime, cell) unit. Since
    Σₖ γₖ(t=1) = 1, summing the per-regime weighted init stats reproduces the
    unit-weight pooled statistic the ungrouped path uses.
    =#
    slots_x0 = repeat(grp.cell_slot[_G_X0], K)
    slots_P0 = repeat(grp.cell_slot[_G_P0], K)
    _grouped_update_x0!(unit_lds, _state_sufs(unit_suf), slots_x0, _state_bufs(bf))
    _broadcast_initial_state!(cell_slds, K, lds1.fit_bool[_G_X0], false)
    _grouped_update_P0!(unit_lds, _state_sufs(unit_suf), slots_P0, slots_x0, sws)
    _broadcast_initial_state!(cell_slds, K, false, lds1.fit_bool[_G_P0])

    return nothing
end

"""
    _grouped_unit_slots(cell_slot, K, tied) -> Vector{Int}

Slot vector over the `K · ncells` regime-major units for one parameter group,
from the group's per-cell slots.

Untied, a regime's cells get slots of their own, so no version is shared across
regimes. Tied, the cell's slot is used as-is and the same version spans every
regime — which is what makes a tied group a property of the cell rather than of
the (regime, cell) pair.
"""
function _grouped_unit_slots(cell_slot::AbstractVector{Int}, K::Int, tied::Bool)
    tied && return repeat(cell_slot, K)
    stride = maximum(cell_slot)
    return [(k - 1) * stride + s for k in 1:K for s in cell_slot]
end

"""
    _spans_all_regimes(units, ncells, K) -> Bool

Whether the flat regime-major `units` cover every regime for each cell they
touch — i.e. the version is tied across regimes, so `Σₖ γₖ(t) = 1` collapses its
responsibilities to unit weights.
"""
function _spans_all_regimes(units::AbstractVector{Int}, ncells::Int, K::Int)
    cells = unique(mod1.(units, ncells))
    return length(units) == K * length(cells)
end

"""
    _grouped_slds_obs_mstep!(obs_model, unit_lds, unit_suf, grp, K, ncells, tied, sws,
                             bufs, unit_sws, data, tfs, γ_view, slots_cd, ntasks)

The emission half of the grouped SLDS M-step, over the flat list of
`(regime, cell)` units.

Gaussian: the conjugate regression and IW update for `R`. Poisson: one
non-conjugate solve per `[C d D]` version, over the trials of every unit sharing
it — a version tied across regimes sees each of its trials once per regime and
`Σₖ γₖ(t) = 1`, so its weights collapse to the unit weights. A composite runs
whichever each member calls for, on that member's views, statistics,
sub-workspaces and slot vectors.
"""
function _grouped_slds_obs_mstep!(
    om::AbstractObservationModel,
    unit_lds::AbstractVector,
    unit_suf::AbstractVector,
    grp::ParameterGrouping,
    K::Int,
    ncells::Int,
    tied::AbstractVector{Symbol},
    sws::SmoothWorkspace{T},
    bufs,
    unit_sws,
    data::Data{T},
    tfs::TrialFilterSmooth{T},
    γ_view,
    ::Any,
    ntasks::Int,
) where {T<:Real}
    ord = _obs_slot_ordinals(om)[1]
    _grouped_slds_member_obs_mstep!(
        om,
        unit_lds,
        unit_suf,
        grp,
        K,
        ncells,
        tied,
        nothing,
        ord,
        sws,
        _state_bufs(bufs),
        unit_sws,
        data.y,
        data.uy,
        tfs,
        γ_view,
        ntasks,
    )
    return nothing
end

function _grouped_slds_obs_mstep!(
    om::CompositeObservationModel,
    unit_lds::AbstractVector,
    unit_suf::AbstractVector,
    grp::ParameterGrouping,
    K::Int,
    ncells::Int,
    tied::AbstractVector{Symbol},
    sws::SmoothWorkspace{T},
    bufs::NamedTuple,
    unit_sws,
    data::Data{T},
    tfs::TrialFilterSmooth{T},
    γ_view,
    ::Any,
    ntasks::Int,
) where {T<:Real}
    ords = _obs_slot_ordinals(om)
    for (m, key) in enumerate(_obs_keys(om))
        member_sws =
            unit_sws === nothing ? nothing : _member_unit_sws(unit_sws, unit_lds, m)
        _grouped_slds_member_obs_mstep!(
            _models(om)[key],
            _member_unit_views(unit_lds, key),
            [s[key] for s in unit_suf],
            grp,
            K,
            ncells,
            tied,
            key,
            ords[m],
            member_sws === nothing ? _obs_workspaces!(sws, unit_lds[1])[m] : member_sws[1],
            bufs[key],
            member_sws,
            data.y[key],
            data.uy[key],
            tfs,
            γ_view,
            ntasks,
        )
    end
    return nothing
end

function _grouped_slds_member_obs_mstep!(
    ::GaussianObservationModel,
    ldss::AbstractVector,
    sufs::AbstractVector,
    grp::ParameterGrouping,
    K::Int,
    ::Int,
    tied::AbstractVector{Symbol},
    key::Union{Nothing,Symbol},
    ord::UnitRange{Int},
    sws::SmoothWorkspace{T},
    bufs::GroupedSufBuffers{T},
    unit_sws,
    ::AbstractVector,
    ::AbstractVector,
    ::TrialFilterSmooth{T},
    ::Any,
    ::Int,
) where {T<:Real}
    lds1 = ldss[1]
    D = lds1.latent_dim
    tie_obs = length(_tied_obs_cols(tied, D, lds1.uy_dim, key)) == D + 1 + lds1.uy_dim
    slots_cd = _grouped_unit_slots(grp.cell_slot[ord[1]], K, tie_obs)
    slots_r = _grouped_unit_slots(grp.cell_slot[ord[2]], K, _tied_name(:R, key) in tied)
    _grouped_update_C_d!(ldss, sufs, slots_cd, slots_r, sws, bufs; unit_sws=unit_sws)
    _grouped_update_R!(ldss, sufs, slots_r, slots_cd, sws; unit_sws=unit_sws)
    return nothing
end

function _grouped_slds_member_obs_mstep!(
    ::PoissonObservationModel,
    ldss::AbstractVector,
    ::AbstractVector,
    grp::ParameterGrouping,
    K::Int,
    ncells::Int,
    tied::AbstractVector{Symbol},
    key::Union{Nothing,Symbol},
    ord::UnitRange{Int},
    sws::SmoothWorkspace{T},
    ::GroupedSufBuffers{T},
    unit_sws,
    y::AbstractVector{<:AbstractMatrix{T}},
    uy::AbstractVector{<:AbstractMatrix{T}},
    tfs::TrialFilterSmooth{T},
    γ_view,
    ntasks::Int,
) where {T<:Real}
    lds1 = ldss[1]
    D = lds1.latent_dim
    tie_obs = length(_tied_obs_cols(tied, D, lds1.uy_dim, key)) == D + 1 + lds1.uy_dim
    slots_cd = _grouped_unit_slots(grp.cell_slot[ord[1]], K, tie_obs)

    for units in _units_by_slot(slots_cd)
        trials = Int[]
        weights = Vector{SubArray{T,1}}()
        for u in units
            k, c = fldmod1(u, ncells)
            for n in grp.cell_trials[c]
                push!(trials, n)
                push!(weights, γ_view(k, n))
            end
        end
        order = sortperm(trials)
        unit_weights = _spans_all_regimes(units, ncells, K) ? nothing : weights[order]
        update_observation_model!(
            ldss[units[1]],
            TrialFilterSmooth([tfs[n] for n in trials[order]]),
            y[trials[order]],
            [_unit_ws(unit_sws, sws, units[1])],
            unit_weights;
            uy=uy[trials[order]],
            ntasks=ntasks,
        )
    end
    return nothing
end
