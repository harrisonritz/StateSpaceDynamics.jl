#=============================================================================
Per-trial ELBO contributions.

`elbo` returns one number for a whole dataset. Anything that scores trials
against each other — a held-out likelihood quoted per trial, a paired bootstrap
over test trials, an early-stopping monitor that wants to see which trials moved
— needs that number split by trial instead.

The split is exact rather than approximate. Given the parameters the trials are
independent, so the ELBO is a sum of per-trial terms plus the parameter
log-prior, which belongs to no trial:

    sum(trial_elbos(m, y)) + log p(θ) == elbo(m, y)

`log p(θ)` is zero unless IW/MN priors are set, in which case it is
[`_state_prior_logdensity`](@ref) + [`_obs_prior_logdensity`](@ref) (summed over
regimes for an SLDS). It is left out of the per-trial vector deliberately: a MAP
penalty on the parameters is not a property of any one trial, and splitting it
across trials would make the per-trial numbers depend on how many trials came
along for the ride.

Each per-trial term is the same three pieces the aggregated path sums —
`E_q[log p(x)]`, `E_q[log p(y | x)]` and `H[q(x)]` — reached through the
per-trial `Q_state!` / `Q_obs!` kernels rather than the sufficient-statistic
ones. One E-step runs for the whole dataset, so the smoothing cost is exactly
what `elbo` pays; only the Q-terms are recomputed trial by trial.
=============================================================================#

"""
    trial_elbos(lds, y; ux=nothing, uy=nothing, ...) -> Vector{T}
    trial_elbos(slds, y; ...) -> Vector{T}

Each trial's contribution to the ELBO, as a vector with one entry per trial.

Sums to [`elbo`](@ref) up to the parameter log-prior, which is a property of the
parameters rather than of any trial and so is excluded:

    sum(trial_elbos(m, y)) + log p(θ) == elbo(m, y)

Use this wherever trials must be compared or resampled — a held-out score per
trial, a paired bootstrap between two models over the same test trials — and
[`elbo`](@ref) when one number for the dataset is all that is wanted.

# Arguments
- `y`: observations, in any of the forms [`elbo`](@ref) accepts — an
  `(obs_dim, T)` matrix (one trial), an `(obs_dim, T, ntrials)` array, a
  `Vector{<:AbstractMatrix}` of per-trial matrices (ragged lengths allowed), or
  a `NamedTuple` of those for a composite emission.
- `ux` / `uy`: dynamics / observation inputs, in the same shape family as `y`.

# Keywords
- `newton_max_iter` / `newton_tol`: Newton-smoother controls, for a
  non-quadratic (Poisson, or composite with a Poisson member) emission only.
- `smoothing_iters` / `tol`: discrete↔continuous alternation controls, for an
  `SLDS` only.

Parameter grouping (`depends_on`) is not supported: a grouped model fits one
parameter set per cell of trials, and this returns contributions under the
single parameter set the model carries. Call it once per cell instead.

For an `SLDS`, [`smooth`](@ref) already returns this vector as its `trial_elbo`
field — read that rather than paying for the alternation twice.
"""
function trial_elbos end

"""
    _trial_elbo_setup(lds, y, ux, uy) -> (data, tfs, sws_pool)

The `Data`, smoother storage and workspace pool [`elbo`](@ref) builds, with the
grouped path rejected up front. Split out so the quadratic and non-quadratic
methods differ only in how they smooth.
"""
function _trial_elbo_setup(
    lds::LinearDynamicalSystem{T,S,O}, y, ux, uy
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:AbstractObservationModel{T}}
    data = Data(lds, y; ux=ux, uy=uy)
    parameter_grouping(lds, length(data.tsteps); y=data.y) === nothing || error(
        "trial_elbos does not support parameter grouping (`depends_on`): a grouped " *
        "model carries one parameter set per cell of trials, and the per-trial " *
        "contributions this returns are under a single set. Split the trials by cell " *
        "and call it once per cell, on that cell's parameters.",
    )
    tfs = initialize_FilterSmooth(lds, data.tsteps)::TrialFilterSmooth{T}
    npool = min(Threads.maxthreadid(), length(data.tsteps))
    sws_pool = [
        SmoothWorkspace(
            T,
            lds.latent_dim,
            _ws_obs_dim(lds),
            maximum(data.tsteps);
            ux_dim=lds.ux_dim,
            uy_dim=_ws_uy_dim(lds),
        ) for _ in 1:npool
    ]
    _td_init_const_blocks!(sws_pool[1], lds, data)
    return data, tfs, sws_pool
end

"""
    _accumulate_trial_elbos(lds, tfs, data, sws) -> Vector{T}

State Q-term + emission Q-term + posterior entropy, trial by trial, from a
smoother output that is already filled.

`sufficient_statistics!` converts each trial's `x_smooth` / `p_smooth` /
`p_smooth_tt1` into the `E_z` / `E_zz` / `E_zz_prev` moments the per-trial
`Q_state!` reads. The suf-based path skips that step because its aggregator
works from the raw smoother output; here the legacy per-trial kernels are
exactly what is wanted, so it is paid for.

The one place the two kernels disagree is a constant. The per-trial `Q_state!`
is written as the M-step objective, so it carries only the terms that move with
the parameters — `−½(log|P₀| + tr …)` for the initial state and
`−½((T−1)log|Q| + tr …)` for the transitions — and drops the `−½ D log 2π` each
of the `T` Gaussian factors contributes, which cannot move an argmax. The ELBO
is a reported number rather than an objective and the aggregated `Q_state!`
does carry it, so it is added back here. `test/…/TrialELBO.jl` pins the two to
each other.
"""
function _accumulate_trial_elbos(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    data::Data{T},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:AbstractObservationModel{T}}
    sufficient_statistics!(tfs)
    compute_smooth_constants!(sws, lds)
    log2π_per_step = T(-0.5) * lds.latent_dim * log(T(2π))

    per_trial = Vector{T}(undef, length(data.tsteps))
    for n in eachindex(per_trial)
        fs = tfs[n]
        per_trial[n] =
            Q_state!(sws, lds, fs.E_z, fs.E_zz, fs.E_zz_prev, data.ux[n]) +
            log2π_per_step * data.tsteps[n] +
            _trial_q_obs(sws, lds, fs, data, n) +
            fs.entropy
    end
    return per_trial
end

#=
One trial's emission Q-term, by whichever per-trial kernel the emission has:
the Gaussian's moment form, the Poisson's rate form (the same call
`_poisson_q_obs_total` makes, one trial at a time), and a composite's sum over
its members' views and sub-workspaces. These are the per-trial counterparts of
the sufficient-statistic `Q_obs!` overloads, and sum to the same totals.
=#
function _trial_q_obs(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    data::Data{T},
    n::Int,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:GaussianObservationModel{T}}
    return Q_obs!(sws, lds, fs.E_z, fs.E_zz, data.y[n], data.uy[n])
end

function _trial_q_obs(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    data::Data{T},
    n::Int,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:PoissonObservationModel{T}}
    return Q_obs!(sws, lds, fs.x_smooth, fs.p_smooth, data.y[n], data.uy[n])
end

function _trial_q_obs(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    data::Data{T},
    n::Int,
) where {T<:Real,S<:AbstractGaussianStateModel{T},O<:CompositeObservationModel{T}}
    views = _obs_views(lds)
    datas = _member_datas(data)
    subs = _obs_workspaces!(sws, lds)
    total = zero(T)
    for (i, key) in enumerate(_obs_keys(lds.obs_model))
        total += _trial_q_obs(subs[i], views[key], fs, datas[key], n)
    end
    return total
end

"""
    trial_elbos(lds, y; ux, uy)

Per-trial ELBO contributions of a `LinearDynamicalSystem` with a quadratic
emission (Gaussian, or a composite whose members all are). The smoother is exact
here, so with no parameter priors each entry is that trial's exact marginal
log-likelihood and the vector sums to [`loglikelihood`](@ref).
"""
function trial_elbos(
    lds::LinearDynamicalSystem{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    ux=nothing,
    uy=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:QuadraticEmission{T}}
    data, tfs, sws_pool = _trial_elbo_setup(lds, y, ux, uy)
    smooth!(lds, tfs, data, sws_pool)
    return _accumulate_trial_elbos(lds, tfs, data, sws_pool[1])
end

"""
    trial_elbos(plds, y; ux, uy, newton_max_iter=20, newton_tol=1e-6)

Per-trial ELBO contributions of a `LinearDynamicalSystem` with a non-quadratic
emission (Poisson, or a composite with a Poisson member). Each entry is a lower
bound on that trial's marginal log-likelihood under the Laplace posterior — not
the log-likelihood itself, which is intractable here.
"""
function trial_elbos(
    plds::LinearDynamicalSystem{T,S,O},
    y::Union{
        AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}},NamedTuple
    };
    ux=nothing,
    uy=nothing,
    newton_max_iter::Int=20,
    newton_tol::Float64=1e-6,
) where {T<:Real,S<:GaussianStateModel{T},O<:NonQuadraticEmission{T}}
    data, tfs, sws_pool = _trial_elbo_setup(plds, y, ux, uy)
    smooth!(plds, tfs, data, sws_pool; max_iter=newton_max_iter, tol=T(newton_tol))
    return _accumulate_trial_elbos(plds, tfs, data, sws_pool[1])
end

"""
    trial_elbos(slds, y; ux, uy, smoothing_iters=100, tol=1e-6, progress=false)

Per-trial ELBO contributions of an `SLDS` — the `trial_elbo` field of
[`smooth`](@ref)`(slds, y)`, which infers `q(x)` and `q(z)` by deterministic
coordinate ascent before evaluating the bound.

If you also want the posteriors that produced it, call [`smooth`](@ref) once and
read its `trial_elbo` field rather than paying for the alternation twice.
"""
function trial_elbos(
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
    ).trial_elbo
end
