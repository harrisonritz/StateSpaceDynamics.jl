#=============================================================================
Terminal conditioning for a switching inverse-LQR model.

`log p(terminal = 0 | θ)` sums over `K^T` discrete paths, so unlike the
non-switching case in `lqr_terminal.jl` there is no exact route to it. What
there is, is the same variational machinery the model already uses for the
data: run the E-step on a copy of the model whose emission loads nothing, and
its ELBO bounds `log p(terminal = 0)` for the same reason the ordinary one
bounds `log p(y)`.

That makes the reported score

    ELBO(y, terminal = 0)  −  ELBO-hat(terminal = 0)

a difference of two bounds, not a bound itself. It is still the right thing to
compare across plant dimensions and costate gauges — the confounds the joint
score carries are removed by the subtraction whether or not either term is
tight — but it is no longer a quantity anyone should call a likelihood. Both
halves are therefore reported separately: see `terminal_logz` and the
`terminal_logz` field `smooth` returns.
=============================================================================#

#=
Alternations the probe's own E-step runs. Fixed rather than inherited from the
caller so that the normalizer is one reproducible number: the fit trace, a later
`elbo` call and `terminal_logz` all have to agree at the same parameters, and
they only do if they all spend the same budget. The probe has no observations,
so its discrete posterior is driven by the dynamics alone and settles quickly.
=#
const _SLQR_PROBE_ITERS = 20

"""
    _slds_condition_terminal(slds) -> Bool

Whether this switching model reports and fits the terminal-conditioned score.

Every discrete state that *has* a terminal factor must agree. One that
conditioned while another did not would be scoring a mixture of two different
likelihoods, and the mixture weights are exactly the discrete posterior, so the
disagreement would not even be constant across trials. States with no terminal
factor at all (a `:free` regime, say) impose nothing and are not consulted.
"""
function _slds_condition_terminal(slds::SLDS)
    flags = Bool[]
    for lds in slds.LDSs
        sm = lds.state_model
        sm isa LQRStateModel || continue
        sm.terminal || continue
        push!(flags, sm.condition_terminal)
    end
    isempty(flags) && return false
    all(flags) && return true
    any(flags) && throw(
        ArgumentError(
            "inverse-LQR discrete states that carry a terminal factor must agree on " *
            "`condition_terminal`; got a mixture. Set it the same way on every such " *
            "state, or drop the terminal factor from the ones that should not have it.",
        ),
    )
    return false
end

_slds_condition_terminal(::Nothing) = false

"""
    _SLQRProbe{T}

The zero-loading copy of a switching model, with the scaffolding its E-step
needs held across M-steps so that only the alternation is repaid each time.

`designs` compresses trials that share an input trajectory *and* a horizon:
`log Z` depends on the trial only through those, so a task design repeated over
hundreds of trials is smoothed once and counted.
"""
mutable struct _SLQRProbe{T<:Real,SL,PL,PP,FB,LN,DS,SF}
    slds::SL
    data::Data{T}
    tfs::TrialFilterSmooth{T}
    dl::SLDSDiscreteLayer{T}
    fb::FB
    pool::PP
    plan::PL
    sws::Vector{SmoothWorkspace{T}}
    obs_seq::Vector{Int}
    control_seq::Vector{Nothing}
    seq_ends::Vector{Int}
    lognorm::LN
    designs::DS
    counts::Vector{T}
    design_of::Vector{Int}
    per_design::Vector{T}
    sufs::Vector{SF}
    smoothing_iters::Int
    logz::T
    started::Bool
end

"""
    _slqr_terminal_probe(slds, ux; smoothing_iters) -> _SLQRProbe

Build the probe: every member keeps its state model and loses its emission.

A one-dimensional Gaussian emission with zero loading and unit variance
contributes `-½ log 2π` per timestep whatever the latent does, so it shifts the
ELBO by a known constant and leaves the posterior exactly `q(z, s | terminal =
0)`. Parameter priors are dropped from the copy because the data side already
carries them; leaving them on would penalize the same parameters twice, once
with each sign.
"""
function _slqr_terminal_probe(
    slds::SLDS{T}, ux::AbstractVector; smoothing_iters::Int=_SLQR_PROBE_ITERS
) where {T<:Real}
    canonical = [Matrix{T}(u) for u in ux]
    designs = _lqr_terminal_designs(canonical)
    isempty(designs) && error("Terminal conditioning requires trial inputs/horizons")
    index = Dict(v.ux => i for (i, v) in enumerate(designs))
    design_of = [index[u] for u in canonical]
    d = slds.LDSs[1].latent_dim
    members = map(slds.LDSs) do lds
        sm = deepcopy(lds.state_model)
        if sm isa LQRStateModel
            sm.condition_terminal = false
            sm.depends_on = nothing
            sm.variants = nothing
            sm.P0_prior = sm.x0_prior = sm.Σ_prior = sm.Qc_prior = nothing
        end
        return LinearDynamicalSystem(
            sm, GaussianObservationModel(zeros(T, 1, d), ones(T, 1, 1), zeros(T, 1))
        )
    end
    probe_slds = SLDS(; A=copy(slds.A), πₖ=copy(slds.πₖ), LDSs=members)

    uxs = [v.ux for v in designs]
    ys = [zeros(T, 1, size(u, 2)) for u in uxs]
    data = Data(members[1], ys; ux=uxs)
    K = length(members)
    ntrials = length(data.tsteps)
    seq_ends = cumsum(data.tsteps)
    total_T = last(seq_ends)
    T_max = maximum(data.tsteps)
    tfs = initialize_FilterSmooth(members[1], data.tsteps)::TrialFilterSmooth{T}
    dl = SLDSDiscreteLayer(probe_slds.A, probe_slds.πₖ, zeros(T, K, total_T))
    fb = _make_slds_fb_storage(dl, seq_ends)
    pool = _slds_workspace_pool(probe_slds, nothing, T_max, ntrials; npool=1)
    plan = _slds_trial_plan(nothing, ntrials, length(pool.slots))
    sufs = [_initialize_td_sufficient_statistics(T, members[1], data.tsteps) for _ in 1:K]
    return _SLQRProbe(
        probe_slds,
        data,
        tfs,
        dl,
        fb,
        pool,
        plan,
        _slds_mstep_pool(probe_slds, T_max, 1),
        collect(1:total_T),
        fill(nothing, total_T),
        seq_ends,
        _slds_lognorm_all(probe_slds, data.y),
        designs,
        T[v.count for v in designs],
        design_of,
        fill(T(NaN), length(designs)),
        sufs,
        smoothing_iters,
        T(NaN),
        false,
    )
end

"""Copy the live model's parameters onto the probe, leaving the probe's own
zero-loading emission and dropped priors alone."""
function _slqr_sync_probe!(probe::_SLQRProbe, slds::SLDS)
    copyto!(probe.slds.A, slds.A)
    copyto!(probe.slds.πₖ, slds.πₖ)
    for (member, lds) in zip(probe.slds.LDSs, slds.LDSs)
        target, source = member.state_model, lds.state_model
        if target isa LQRStateModel
            for key in (:A, :S, :h, :Bu, :Gref, :hf, :Σ, :Σf, :x0, :P0)
                copyto!(getproperty(target, key), getproperty(source, key))
            end
            for k in eachindex(source.Qc)
                copyto!(target.Qc[k], source.Qc[k])
            end
        else
            for key in (:A, :b, :B, :Q, :x0, :P0)
                hasproperty(target, key) &&
                    copyto!(getproperty(target, key), getproperty(source, key))
            end
        end
        refresh!(target)
    end
    return probe
end

"""
    _slqr_probe_estep!(probe) -> probe

Run the probe's variational E-step and record `log Z-hat` and the per-regime
responsibility-weighted statistics the M-step's Fisher term reads.

The zero-loading emission's `-½ log 2π` per timestep is added back here, so
`probe.logz` is the bound on `log p(terminal = 0)` itself rather than on the
probe's own observation density.
"""
function _slqr_probe_estep!(probe::_SLQRProbe{T}) where {T<:Real}
    _prepare_slds!(probe.slds, probe.data.tsteps)
    if !probe.started
        _slds_warmstart!(
            probe.slds,
            nothing,
            nothing,
            probe.tfs,
            probe.data.y,
            nothing,
            probe.pool,
            probe.plan,
            probe.data.tsteps,
            length(probe.slds.LDSs);
            ux=probe.data.ux,
            uy=probe.data.uy,
            lognorm=probe.lognorm,
        )
        probe.started = true
    end
    _vem_alternate!(
        probe.slds,
        nothing,
        nothing,
        probe.tfs,
        probe.fb,
        probe.dl,
        probe.data.y,
        probe.pool,
        probe.plan;
        obs_seq=probe.obs_seq,
        control_seq=probe.control_seq,
        seq_ends=probe.seq_ends,
        ux=probe.data.ux,
        uy=probe.data.uy,
        lognorm=probe.lognorm,
        smoothing_iters=probe.smoothing_iters,
    )
    per_design = _slds_trial_elbos(
        probe.slds,
        nothing,
        nothing,
        probe.tfs,
        probe.fb,
        probe.data.y,
        probe.pool,
        probe.plan;
        seq_ends=probe.seq_ends,
        ux=probe.data.ux,
        uy=probe.data.uy,
        lognorm=probe.lognorm,
    )
    half_log2π = T(0.5) * log(T(2π))
    probe.logz = zero(T)
    for (i, steps) in enumerate(probe.data.tsteps)
        probe.per_design[i] = per_design[i] + T(steps) * half_log2π
        probe.logz += probe.counts[i] * probe.per_design[i]
    end

    for k in eachindex(probe.slds.LDSs)
        weights = [
            begin
                t1, t2 = HMMs.seq_limits(probe.seq_ends, trial)
                probe.counts[trial] .* Vector{T}(view(probe.fb.γ, k, t1:t2))
            end for trial in eachindex(probe.data.tsteps)
        ]
        _slds_aggregate_weighted!(
            probe.sufs[k], probe.tfs, probe.slds.LDSs[k], probe.data, weights, probe.sws[1]
        )
    end
    return probe
end

# --- normalizer-backend interface, shared with the non-switching M-step -----

#=
The probe's *statistics* are what stays fixed across the inner L-BFGS — they are
the expectation the surrogate is taken under, and recomputing them would need an
SLDS E-step per line-search point. Its *parameters* must not: the M-step context
reads `Σ` and `Σf` off these models to weight the residuals, so a probe left at
the values it was smoothed with would weight them by a stale covariance. Copy
the live parameters over, and keep the probe's nulled priors — the data side
already carries those, and counting them on both sides would cancel them.
=#
function _terminal_probe_stats!(b::_SLQRNormalizer, sms)
    probe = b.probe
    lqr = [
        k for k in eachindex(probe.slds.LDSs) if
        probe.slds.LDSs[k].state_model isa LQRStateModel &&
        !_is_free(probe.slds.LDSs[k].state_model)
    ]
    sufs = [probe.sufs[k] for k in lqr]
    psms = [probe.slds.LDSs[k].state_model for k in lqr]
    length(psms) == length(sms) || error(
        "terminal probe has $(length(psms)) inverse-LQR states, model has $(length(sms))",
    )
    for (target, source, hs) in zip(psms, sms, sufs)
        for key in (:A, :S, :h, :Bu, :Gref, :hf, :Σ, :Σf, :x0, :P0)
            copyto!(getproperty(target, key), getproperty(source, key))
        end
        for k in eachindex(source.Qc)
            copyto!(target.Qc[k], source.Qc[k])
        end
        refresh!(target)
        _fill_mixed_blocks!(hs, target)
    end
    return (sufs, psms)
end

"""
    terminal_logz(slds, y; ux, uy, smoothing_iters) -> T

The variational estimate of `log p(terminal = 0 | θ)` this model's score is
divided by, as a number on its own.

Reported separately because it is an approximation: [`elbo`](@ref) returns the
joint ELBO less this, and a difference of two bounds is not a bound. Comparing
it across fits is how to tell whether a score gap is the data fitting better or
the normalizer moving.
"""
function terminal_logz(
    slds::SLDS{T}, y; ux=nothing, uy=nothing, smoothing_iters::Int=_SLQR_PROBE_ITERS
) where {T<:Real}
    _slds_condition_terminal(slds) || return zero(T)
    data = Data(slds.LDSs[1], y; ux=ux, uy=uy)
    probe = _slqr_terminal_probe(slds, data.ux; smoothing_iters=smoothing_iters)
    _slqr_sync_probe!(probe, slds)
    _slqr_probe_estep!(probe)
    return probe.logz
end

"""Canonical per-trial inputs, reconstructed from the horizons when a caller
passed none: `log Z` depends on a trial through its inputs *and* its length, and
the length is always available."""
function _slds_probe_inputs(ux, seq_ends, ::Type{T}) where {T}
    ux === nothing || return ux
    steps = diff(vcat(0, collect(seq_ends)))
    return [zeros(T, 0, s) for s in steps]
end

"""
    _slds_terminal_trial_logz(slds, ux, seq_ends; smoothing_iters) -> Vector or nothing

Each trial's `log Z-hat`, or `nothing` when this model does not condition.

Built fresh rather than cached across iterations: the probe's cost scales with
the number of distinct designs, not trials, and a task design repeated across a
session collapses to one smoothed chain.
"""
function _slds_terminal_trial_logz(
    slds::SLDS{T}, ux, seq_ends; smoothing_iters::Int=_SLQR_PROBE_ITERS
) where {T<:Real}
    _slds_condition_terminal(slds) || return nothing
    probe = _slqr_terminal_probe(
        slds, _slds_probe_inputs(ux, seq_ends, T); smoothing_iters=smoothing_iters
    )
    _slqr_sync_probe!(probe, slds)
    _slqr_probe_estep!(probe)
    return [probe.per_design[i] for i in probe.design_of]
end
