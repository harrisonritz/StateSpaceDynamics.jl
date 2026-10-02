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

`log p(terminal = 0 | θ)` factorizes over trials given the chain, and reads only
the state side. So a `depends_on` grouping that splits the state parameters — a
cost per reward level — gives one probe per state variant, over that variant's
trials, and `log Ẑ` is their sum: the variants' state M-steps each read their own
probe, and the shared chain's reads them all. A grouping of the emission alone
(the stitched fit) leaves the one probe the parent model has always had.
=============================================================================#

#=
Alternations the probe's own E-step runs. Fixed rather than inherited from the
caller so that the normalizer is one reproducible number: the fit trace, a later
`elbo` call and `terminal_logz` all have to agree at the same parameters, and
they only do if they all spend the same budget. The probe has no observations,
so its discrete posterior is driven by the dynamics alone and settles quickly.
=#
const _SLQR_PROBE_ITERS = 20

# Seed of the probe's own stream; see `_SLQRProbe.rng` and `_slqr_restart!`.
const _SLQR_PROBE_SEED = 0x5109

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

"""Whether every cell of `grp` uses the parent's state parameters — the one
state variant — so that only the emission varies across cells."""
_slds_state_shared(grp) = all(==(1), grp.cell_state)

"""
    _slds_trial_variants(grp) -> Vector{Int} or nothing

Each trial's state-parameter variant under a `depends_on` grouping, or `nothing`
when every trial reads the parent's state parameters.

`log p(terminal = 0 | θ)` is a property of the dynamics, the chain, the inputs and
the horizon — never of the emission — and it factorizes over trials. So the
partition it needs is the state side's alone: a grouping that splits only the
emission (the stitched fit) leaves one normalizer, the parent model's, and one
that splits the state parameters (a cost per reward level, say) gives each state
variant its own, over its own trials. The chain is shared by every variant, which
is what couples them in the discrete M-step.
"""
function _slds_trial_variants(grp)
    grp === nothing && return nothing
    _slds_state_shared(grp) && return nothing
    return [grp.cell_state[grp.trial_cell[n]] for n in eachindex(grp.trial_cell)]
end

"""
    _slds_variant_view(slds, v) -> SLDS

The switching model as state variant `v` sees it: every regime's state model
replaced by its `v`-th `depends_on` variant, the chain shared **by reference**.

Parameters the declaration did not split are shared by reference across the
variants too, so a view is a window onto the live model rather than a copy of it:
writing a variant's cost through the view writes the model's, and a chain update
on the parent is already the view's. The emission is the parent's template and is
never read — the normalizer does not depend on it.
"""
function _slds_variant_view(
    slds::SLDS{T,S,O,TM,ISV}, v::Int
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel,TM,ISV}
    ldss = map(slds.LDSs) do lds
        sm = (lds.state_model.variants::Vector{S})[v]
        LinearDynamicalSystem{T,S,O}(
            sm,
            lds.obs_model,
            lds.latent_dim,
            _obs_dim(lds.obs_model),
            lds.ux_dim,
            lds.uy_dim,
            lds.fit_bool,
        )
    end
    return SLDS{T,S,O,TM,ISV}(
        slds.A, slds.πₖ, ldss, slds.A_prior, slds.πₖ_prior, slds.boundaries
    )
end

"""
    _state_trial_variants(sm, ntrials; depends_on) -> Vector{Int} or nothing

Each of `ntrials` trials' state variant for a state model that declares
`depends_on`, read from the labels stored on the model — the fitted trials — or
from `depends_on`, which relabels another set of trials (a held-out one) into
the groups the model already has. `nothing` when the state side is not grouped.

Only the state model's groups are read, so an override may carry the emission's
labels too (a stitched fit's session per trial) and they are ignored here: the
normalizer does not depend on the emission.
"""
function _state_trial_variants(
    sm::AbstractStateModel, ntrials::Int; depends_on::Union{Nothing,NamedTuple}=nothing
)
    sm isa DependentModel || return nothing
    dep = _resolve_dependence(sm)
    _any_varies(dep) || return nothing
    _build_variants!(sm, dep)
    labels = [_trial_labels_for(dep, g, sm, depends_on) for g in eachindex(dep.names)]
    for g in eachindex(dep.names)
        dep.varies[g] || continue
        length(labels[g]) == ntrials || throw(
            DimensionMismatchError(
                "depends_on labels for :$(dep.names[g])", ntrials, length(labels[g])
            ),
        )
    end
    slots = ones(Int, length(dep.names))
    return map(1:ntrials) do n
        for g in eachindex(dep.names)
            slots[g] = _slot_of(dep, g, dep.varies[g] ? labels[g][n] : nothing)
        end
        _variant_index(dep.nslots, slots)
    end
end

"""The switching counterpart of [`_state_trial_variants`](@ref): every regime
declares the same grouping, and every regime's variants are built, since the
probe for a variant reads all of them."""
function _slds_state_trial_variants(
    slds::SLDS, ntrials::Int; depends_on::Union{Nothing,NamedTuple}=nothing
)
    variants = nothing
    for lds in slds.LDSs
        v = _state_trial_variants(lds.state_model, ntrials; depends_on=depends_on)
        variants === nothing && (variants = v)
        v == variants || throw(
            ArgumentError(
                "SLDS: every regime must declare the same state `depends_on` labels; " *
                "the terminal normalizer is built per state variant, over all regimes.",
            ),
        )
    end
    return variants
end

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
    control_seq::Vector{Int}
    seq_ends::Vector{Int}
    lognorm::LN
    designs::DS
    counts::Vector{T}
    design_of::Vector{Int}
    per_design::Vector{T}
    sufs::Vector{SF}
    #=
    The probe's own stream. Both `_slds_warmstart!` and `_vem_alternate!` default
    to `Random.default_rng()`, and a normalizer that drew from the global stream
    would make the surrounding fit depend on how many numbers everything else had
    already taken — reproducible runs are the whole reason the discrete layer is
    scored deterministically in the first place.
    =#
    rng::Random.Xoshiro
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
    slds::SLDS{T}, ux::AbstractVector; ux0=nothing, smoothing_iters::Int=_SLQR_PROBE_ITERS
) where {T<:Real}
    canonical = [Matrix{T}(u) for u in ux]
    u0 = _normalize_ux0(ux0, slds.LDSs[1].state_model, length(canonical))
    designs = _lqr_terminal_designs(canonical, u0)
    isempty(designs) && error("Terminal conditioning requires trial inputs/horizons")
    index = Dict((v.ux, v.ux0) => i for (i, v) in enumerate(designs))
    design_of = [
        index[(canonical[i], Vector{T}(view(u0, :, i)))] for i in eachindex(canonical)
    ]
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
    #= The boundary factors are shared by reference: exit bridges are part of the
    terminal event being conditioned on, and entry priors part of the prior it is
    normalized under, so the probe must see the live ones — including an entry
    prior the M-step is in the middle of updating. =#
    probe_slds = SLDS(;
        A=copy(slds.A), πₖ=copy(slds.πₖ), LDSs=members, boundaries=slds.boundaries
    )

    uxs = [v.ux for v in designs]
    ys = [zeros(T, 1, size(u, 2)) for u in uxs]
    data = Data(members[1], ys; ux0=hcat((v.ux0 for v in designs)...), ux=uxs)
    K = length(members)
    ntrials = length(data.tsteps)
    seq_ends = cumsum(data.tsteps)
    total_T = last(seq_ends)
    T_max = maximum(data.tsteps)
    tfs = initialize_FilterSmooth(members[1], data.tsteps)::TrialFilterSmooth{T}
    dl = _slds_discrete_layer(probe_slds, total_T)
    fb = _make_slds_fb_storage(dl, seq_ends)
    pool = _slds_workspace_pool(probe_slds, nothing, T_max, ntrials; npool=1)
    plan = _slds_trial_plan(nothing, ntrials, length(pool.slots))
    sufs = [_initialize_td_sufficient_statistics(T, members[k], data.tsteps) for k in 1:K]
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
        collect(1:total_T),
        seq_ends,
        _slds_lognorm_all(probe_slds, data.y),
        designs,
        T[v.count for v in designs],
        design_of,
        fill(T(NaN), length(designs)),
        sufs,
        Random.Xoshiro(_SLQR_PROBE_SEED),
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
            for key in (:A, :Mfree, :S, :h, :Bu, :Gref, :hf, :Σ, :Σf, :x0, :B0, :P0)
                copyto!(getproperty(target, key), getproperty(source, key))
            end
            for k in eachindex(source.Qc)
                copyto!(target.Qc[k], source.Qc[k])
            end
        else
            for key in (:A, :b, :B, :Q, :x0, :B0, :P0)
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
    #=
    The pool caches each member's smoother constants from whenever they were
    last computed, and the smoother reads them without looking at the model. A
    probe built fresh never notices; one smoothed again after its parameters
    moved would smooth under the old ones and score under the new.
    =#
    refresh_slds_pool!(probe.pool, probe.slds)
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
            rng=probe.rng,
            ux0=probe.data.ux0,
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
        ux0=probe.data.ux0,
        ux=probe.data.ux,
        uy=probe.data.uy,
        lognorm=probe.lognorm,
        smoothing_iters=probe.smoothing_iters,
        rng=probe.rng,
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
        ux0=probe.data.ux0,
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

    #= The same statistics the data side aggregates, each design weighted by how
    many trials share it: bridged states add their terminal moments at every
    exit, and entry-prior states count only their ordinary transitions. =#
    dl, ntr = probe.dl, eachindex(probe.data.tsteps)
    for k in eachindex(probe.slds.LDSs)
        function per_trial(w)
            return [
                begin
                    t1, t2 = HMMs.seq_limits(probe.seq_ends, trial)
                    probe.counts[trial] .* Vector{T}(view(w, t1:t2))
                end for trial in ntr
            ]
        end
        weights = per_trial(view(probe.fb.γ, k, :))
        exits = if _has_bridges(dl) && dl.bridge[k]
            per_trial(view(dl.exit_w, k, :))
        else
            nothing
        end
        ordinary = _slds_ordinary_weights(dl, probe.fb, probe.seq_ends, k, ntr)
        dyn = if ordinary === nothing
            nothing
        else
            [probe.counts[trial] .* ordinary[trial] for trial in ntr]
        end
        _slds_aggregate_weighted!(
            probe.sufs[k],
            probe.tfs,
            probe.slds.LDSs[k],
            probe.data,
            weights,
            probe.sws[1];
            exit_weights=exits,
            dyn_weights=dyn,
        )
    end
    return probe
end

"""
    _slqr_restart!(probe) -> probe

Return the probe to the state it was built in, so that its next E-step is the
one a freshly built probe would run: warm start included, and the stream
reseeded.

`log Z-hat` is a function of the parameters only if every evaluation starts
from the same place. The reported score builds its probe fresh, so an M-step
that judges proposals with a probe carried over from the last point it visited
would be judging them on a different number.
"""
function _slqr_restart!(probe::_SLQRProbe)
    probe.started = false
    Random.seed!(probe.rng, _SLQR_PROBE_SEED)
    return probe
end

"""The normalizer backend for one probe standing in for every inverse-LQR state
of its model, in discrete-state order — the ungrouped (or stitched) case."""
function _SLQRNormalizer(probe::_SLQRProbe)
    n = count(
        lds -> lds.state_model isa LQRStateModel && !_is_free(lds.state_model),
        probe.slds.LDSs,
    )
    return _SLQRNormalizer([probe], [collect(1:n)])
end

"""
    _slqr_terminal_probes(slds, ux, variants; ux0, smoothing_iters) -> (probes, sources, trials)

One probe per state variant: `probes[i]` mirrors the model `sources[i]` over the
trials `trials[i]`, whose inputs are `ux[trials[i]]`.

`variants === nothing` (no grouping, or one that splits only the emission) is the
single probe the parent model has always had, over every trial. Otherwise each
variant gets a probe of its own, built from [`_slds_variant_view`](@ref), since
its trials are conditioned under its own control problems: `log Ẑ` is then the
sum of the variants' bounds. The views share the parent's chain by reference, so
a chain the M-step writes into the parent is the one every probe syncs.
"""
function _slqr_terminal_probes(
    slds::SLDS{T},
    ux::AbstractVector,
    variants::Union{Nothing,AbstractVector{Int}};
    ux0=nothing,
    smoothing_iters::Int=_SLQR_PROBE_ITERS,
) where {T<:Real}
    if variants === nothing
        probe = _slqr_terminal_probe(slds, ux; ux0=ux0, smoothing_iters=smoothing_iters)
        return [probe], [slds], [collect(eachindex(ux))]
    end
    ids = sort!(unique(variants))
    trials = [findall(==(v), variants) for v in ids]
    sources = [_slds_variant_view(slds, v) for v in ids]
    #= `ux0` is one column per trial, so each variant's probe takes its own
    trials' columns: the initial state, and with it `log Z`, moves with them. =#
    probes = [
        _slqr_terminal_probe(
            src,
            ux[tr];
            ux0=ux0 === nothing ? nothing : ux0[:, tr],
            smoothing_iters=smoothing_iters,
        ) for (src, tr) in zip(sources, trials)
    ]
    return probes, sources, trials
end

"""
    _slqr_probes_logz!(probes, sources; restart=true) -> T

`log Ẑ` at the parameters `sources` hold now: each probe synced to its source,
restarted (unless `restart = false`, for a probe built fresh), smoothed, and the
bounds summed in probe order so the total does not depend on the thread count.

Each probe owns its model copy, workspaces and stream, so the variants smooth
concurrently; a probe's own passes run on one workspace and would otherwise leave
the other threads idle.
"""
function _slqr_probes_logz!(
    probes::AbstractVector{<:_SLQRProbe{T}}, sources::AbstractVector; restart::Bool=true
) where {T<:Real}
    tforeach(eachindex(probes, sources)) do i
        _slqr_sync_probe!(probes[i], sources[i])
        restart && _slqr_restart!(probes[i])
        _slqr_probe_estep!(probes[i])
    end
    return sum(p.logz for p in probes)
end

"""
    _slqr_chain_mstep!(slds, dl, fb_storage, obs_seq, seq_ends, probes, sources) -> Bool

The discrete chain's update when the model conditions on its terminal factor.

The score is `ELBO(y, terminal = 0) − log Ẑ`, and `log Ẑ` depends on `A` and
`πₖ` through the probe's chain as surely as the data half does. With `q` held
fixed, the chain's part of it is

    g(A, π) = Σᵢⱼ Nᵢⱼ log Aᵢⱼ + Σₖ nₖ log πₖ − log Ẑ(A, π),

`N` / `n` the data's expected transition and initial counts, under the same
`log(· + 1e-12)` the ELBO takes, and `log Ẑ` from a probe restarted as a fresh
one would be. Holding the probe's posterior fixed as the state M-step does would
leave `Σᵢⱼ (Nᵢⱼ − Ξᵢⱼ) log Aᵢⱼ`, with `Ξ` the probe's expected counts: unbounded
wherever the probe expects more `i → j` transitions than the data, so there is
no closed-form update. Two proposals instead, each kept only if `g` rises:

1. The Baum–Welch update, which maximizes the data half. It is the whole answer
   when the chain barely moves `log Ẑ`.
2. Otherwise an ascent step on `g` itself, in each row's softmax logits. At the
   probe's stationary posterior `∂ log Ẑ / ∂Aᵢⱼ = Ξᵢⱼ / Aᵢⱼ` (Danskin, as for the
   state parameters), so the logit gradient is `cᵢⱼ − Aᵢⱼ Σₖ cᵢₖ` with
   `c = N − Ξ`, and likewise for `πₖ`. The step starts at a size that moves no
   logit by more than one and halves until `g` rises by an Armijo fraction of
   what the gradient promises.

With a `depends_on` grouping that splits the state parameters there is one probe
per state variant ([`_slqr_terminal_probes`](@ref)) and one chain: `log Ẑ` is the
sum of the variants' bounds, and `Ξ` the sum of their expected counts, each over
its own trials. Nothing else changes — the chain is shared, so it is fitted
against every variant's normalizer at once.

If neither improves `g` the chain stays put. Returns whether the probes are left
smoothed at the chain `slds` now holds, so the state M-step can use them as is.
"""
function _slqr_chain_mstep!(
    slds::SLDS{T},
    dl::SLDSDiscreteLayer{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    probe::_SLQRProbe{T};
    kwargs...,
) where {T<:Real}
    return _slqr_chain_mstep!(
        slds, dl, fb_storage, obs_seq, seq_ends, [probe], [slds]; kwargs...
    )
end

function _slqr_chain_mstep!(
    slds::SLDS{T},
    dl::SLDSDiscreteLayer{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    probes::AbstractVector{<:_SLQRProbe{T}},
    sources::AbstractVector;
    max_halvings::Int=12,
) where {T<:Real}
    K = length(slds.LDSs)
    # Counts first: `fit!` below uses each trial's last ξ as scratch.
    N, n = _slds_chain_counts(fb_storage, seq_ends, K, T)
    #=
    A Dirichlet chain prior adds `Σ (α − 1) log A` to `g`, which is the same as
    adding `α − 1` to the counts: `score!`, the ascent direction `c = N − Ξ` and
    (through `fit!`'s own prior keywords) the Baum–Welch proposal all see it.
    =#
    slds.A_prior === nothing || (N .+= slds.A_prior .- one(T))
    slds.πₖ_prior === nothing || (n .+= slds.πₖ_prior .- one(T))
    floor = T(1e-12)
    function score!(A, π)
        copyto!(slds.A, A)
        copyto!(slds.πₖ, π)
        logz = try
            _slqr_probes_logz!(probes, sources)
        catch err
            # A proposal the probe cannot smooth is a rejected one.
            _lqr_rejectable(err) || rethrow()
            return -T(Inf)
        end
        chain = sum(N .* log.(A .+ floor)) + sum(n .* log.(π .+ floor))
        return chain - logz
    end

    A0, π0 = copy(slds.A), copy(slds.πₖ)
    base = score!(A0, π0)
    #= A probe that cannot be smoothed at the incoming chain leaves nothing to
    compare against; keep the chain, and let the state M-step's own check say so. =#
    if !isfinite(base)
        copyto!(slds.A, A0)
        copyto!(slds.πₖ, π0)
        return false
    end
    # The probes' own counts, weighted by how many trials share each design.
    Ξ, ν = zeros(T, K, K), zeros(T, K)
    for probe in probes
        Ξp, νp = _slds_chain_counts(probe.fb, probe.seq_ends, K, T, probe.counts)
        Ξ .+= Ξp
        ν .+= νp
    end

    StatsAPI.fit!(                                               # writes slds.A / πₖ
        dl,
        fb_storage,
        obs_seq;
        seq_ends=seq_ends,
        A_prior=slds.A_prior,
        πₖ_prior=slds.πₖ_prior,
    )
    A1, π1 = copy(slds.A), copy(slds.πₖ)
    if !(A0 == A1 && π0 == π1)
        gain = score!(A1, π1) - base
        if gain >= 0
            @debug "terminal-conditioned chain step" proposal = :baum_welch gain
            return true
        end
    end

    c, cπ = N .- Ξ, n .- ν
    Gη = c .- A0 .* sum(c; dims=2)
    Gπ = cπ .- π0 .* sum(cπ)
    slope = sum(abs2, Gη) + sum(abs2, Gπ)
    if slope > zero(T)
        η0, ηπ0 = log.(A0), log.(π0)
        step = one(T) / max(maximum(abs, Gη), maximum(abs, Gπ))
        for _ in 0:max_halvings
            A = exp.(η0 .+ step .* Gη)
            A ./= sum(A; dims=2)
            π = exp.(ηπ0 .+ step .* Gπ)
            π ./= sum(π)
            gain = score!(A, π) - base
            if gain >= T(1e-4) * step * slope
                @debug "terminal-conditioned chain step" proposal = :gradient step gain
                return true
            end
            step /= 2
        end
    end
    @debug "terminal-conditioned chain step rejected" base
    copyto!(slds.A, A0)
    copyto!(slds.πₖ, π0)
    return false
end

"""
    _slds_lqr_variant_units(cell_slds, grp, K, tied) -> NamedTuple

The M-step units of a grouped switching fit under terminal conditioning: one per
`(regime, state variant)` pair, regime-major, with everything the conditional
M-step needs to know about them.

- `ldss[u]` writes through a representative cell's regime model, whose state
  arrays are the variant's;
- `cells[i]` are the cells of variant `i` — the ones whose statistics pool into
  its units, since they differ only in their emission;
- `slots` / `blockslots`: which copy of each group and structural block each unit
  fits, the pair of its version across regimes (from `tied`) and across variants
  (from the `depends_on` declaration, read off the arrays the variants share) —
  the same bookkeeping the joint grouped step uses;
- `units[i]`: variant `i`'s units in regime order, which its probe mirrors.

Variants are listed as [`_slqr_terminal_probes`](@ref) lists them, sorted, so
`units[i]` and the `i`-th probe are the same variant.
"""
function _slds_lqr_variant_units(
    cell_slds::AbstractVector, grp::ParameterGrouping, K::Int, tied::AbstractVector{Symbol}
)
    ids = sort!(unique(grp.cell_state))
    V = length(ids)
    rep = [findfirst(==(v), grp.cell_state)::Int for v in ids]
    cells = [findall(==(v), grp.cell_state) for v in ids]
    ldss = [cell_slds[rep[i]].LDSs[k] for k in 1:K for i in 1:V]
    sms = [lds.state_model for lds in ldss]

    nunits = K * V
    regime_of(u) = fldmod1(u, V)[1]
    cell_of(u) = rep[fldmod1(u, V)[2]]
    function pair(regime, group)
        return _lqr_pair_slots(
            [regime(u) for u in 1:nunits],
            [grp.cell_slot[group][cell_of(u)] for u in 1:nunits],
        )
    end
    block_slots = _lqr_block_slots(tied, K)
    variant_shares = ntuple(b -> _lqr_shares_block(sms[1:V], b), _LQR_BLOCK_N)
    blockslots = ntuple(
        b -> _lqr_pair_slots(
            [block_slots[b][regime_of(u)] for u in 1:nunits],
            [variant_shares[b] ? 1 : grp.cell_slot[_G_AB][cell_of(u)] for u in 1:nunits],
        ),
        _LQR_BLOCK_N,
    )
    #= `x0` / `P0` are tied across regimes unconditionally, so only the
    declaration can split them; the noise follows `tied` across regimes. =#
    slots = [Int[] for _ in 1:_G_Q]
    slots[_G_X0] = pair(_ -> 1, _G_X0)
    slots[_G_P0] = pair(_ -> 1, _G_P0)
    slots[_G_AB] = blockslots[_LQR_BLOCK_A]
    slots[_G_Q] = pair(u -> (:noise in tied) ? 1 : regime_of(u), _G_Q)
    units = [[(k - 1) * V + i for k in 1:K] for i in 1:V]
    return (; ldss, cells, slots, blockslots, units)
end

"""
    _slds_lqr_grouped_conditional_mstep!(cell_slds, unit_suf, grp, K, tied,
                                         probes, sources, probes_current)

The state half of a grouped switching M-step under terminal conditioning, over
the `(regime, state variant)` units of [`_slds_lqr_variant_units`](@ref): each
variant's statistics are pooled over its cells (see
[`_pool_lqr_state_stats`](@ref)), and `probes[i]` supplies the Fisher term and
`log Ẑ` for variant `i`'s units.

A grouping that leaves the state side shared has one variant, and this is then
exactly the ungrouped conditional M-step on the pooled statistics — which is
what makes the stitched fit and the single-session one the same estimator.

`probes_current` says the chain step left every probe smoothed at the current
parameters; otherwise they are re-smoothed here first.
"""
function _slds_lqr_grouped_conditional_mstep!(
    cell_slds::AbstractVector,
    unit_suf::AbstractVector,
    grp::ParameterGrouping,
    K::Int,
    tied::AbstractVector{Symbol},
    probes::AbstractVector{<:_SLQRProbe},
    sources::AbstractVector,
    probes_current::Bool;
    entry_score=nothing,
)
    (; ldss, cells, slots, blockslots, units) = _slds_lqr_variant_units(
        cell_slds, grp, K, tied
    )
    length(probes) == length(units) || error(
        "terminal probes ($(length(probes))) do not match state variants " *
        "($(length(units)))",
    )
    sms = [lds.state_model for lds in ldss]
    any(_is_free, sms) && throw(
        ArgumentError(
            "terminal conditioning is not implemented for a switching model that " *
            "mixes `:free` and inverse-LQR discrete states: the shared initial " *
            "state would be fitted from the inverse-LQR regimes alone. Use " *
            "inverse-LQR states throughout, or set `condition_terminal=false`.",
        ),
    )
    ncells = grp.ncells
    pooled = [
        _pool_lqr_state_stats([unit_suf[(k - 1) * ncells + c] for c in cells[i]]) for
        k in 1:K for i in eachindex(cells)
    ]
    probes_current || _slqr_probes_logz!(probes, sources)
    _lqr_conditional_mstep!(
        ldss,
        pooled,
        slots,
        blockslots,
        _SLQRNormalizer(probes, units);
        score_extra=entry_score,
    )
    foreach(refresh!, sms)
    return nothing
end

"""
    _slds_chain_counts(fb, seq_ends, K, T, weights=nothing) -> (N, n)

Expected transition counts `N` (`K × K`) and initial counts `n` from a
forward-backward pass, summed over trials, each trial weighted by `weights` if
given. Reads `ξ` for every step but the last, which is scratch by convention.
"""
function _slds_chain_counts(
    fb::HMMs.ForwardBackwardStorage,
    seq_ends::AbstractVector{Int},
    K::Int,
    ::Type{T},
    weights::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real}
    N = zeros(T, K, K)
    n = zeros(T, K)
    for trial in eachindex(seq_ends)
        t1, t2 = HMMs.seq_limits(seq_ends, trial)
        w = weights === nothing ? one(T) : T(weights[trial])
        n .+= w .* view(fb.γ, :, t1)
        for t in t1:(t2 - 1)
            N .+= w .* fb.ξ[t]
        end
    end
    return N, n
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
    psufs = Vector{Any}(undef, length(sms))
    psms = Vector{Any}(undef, length(sms))
    for (probe, units) in zip(b.probes, b.units)
        sufs, targets = _slqr_copy_lqr_params!(probe, sms[units])
        for (u, hs, target) in zip(units, sufs, targets)
            _fill_mixed_blocks!(hs, target)
            psufs[u] = hs
            psms[u] = target
        end
    end
    # Concretely typed again, which the M-step context's constructor dispatches on.
    return ([hs for hs in psufs], [sm for sm in psms])
end

"""
    _slqr_copy_lqr_params!(probe, sms) -> (sufs, psms)

Copy the M-step's inverse-LQR state parameters onto the matching probe members
and return those members with their statistics. `sms` holds only the
inverse-LQR states, in discrete-state order, which is how the M-step context
collects them.
"""
function _slqr_copy_lqr_params!(probe::_SLQRProbe, sms)
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
    for (target, source) in zip(psms, sms)
        for key in (:A, :S, :h, :Bu, :Gref, :hf, :Σ, :Σf, :x0, :B0, :P0)
            copyto!(getproperty(target, key), getproperty(source, key))
        end
        for k in eachindex(source.Qc)
            copyto!(target.Qc[k], source.Qc[k])
        end
        refresh!(target)
    end
    return (sufs, psms)
end

#=
The number the fit trace divides by, at the M-step's current parameters: the
probe restarted and smoothed exactly as `_slds_terminal_trial_logz` smooths a
fresh one. That is one probe E-step per call, which is why only the acceptance
check asks for it. A chain the probe cannot factor at these parameters throws,
and the caller's rejectable-error handling turns that into a rejected point.
=#
function _terminal_score_logz(b::_SLQRNormalizer, sms)
    #= The variants smooth concurrently (each probe owns its storage and stream)
    and are summed in probe order, so the total is the same on any thread count. =#
    tforeach(eachindex(b.probes, b.units)) do i
        _slqr_copy_lqr_params!(b.probes[i], sms[b.units[i]])
        _slqr_restart!(b.probes[i])
        _slqr_probe_estep!(b.probes[i])
    end
    return sum(p.logz for p in b.probes)
end

# The surrogate reads the probes' statistics and nothing else of their posteriors.
_terminal_save(b::_SLQRNormalizer) = [deepcopy(p.sufs) for p in b.probes]
function _terminal_restore!(b::_SLQRNormalizer, saved)
    for (probe, sufs) in zip(b.probes, saved)
        copyto!(probe.sufs, sufs)
    end
    return nothing
end

"""
    terminal_logz(slds, y; ux, uy, depends_on, smoothing_iters) -> T

The variational estimate of `log p(terminal = 0 | θ)` this model's score is
divided by, as a number on its own.

Reported separately because it is an approximation: [`elbo`](@ref) returns the
joint ELBO less this, and a difference of two bounds is not a bound. Comparing
it across fits is how to tell whether a score gap is the data fitting better or
the normalizer moving.

A model whose `depends_on` splits the state parameters has one normalizer per
state variant, each over its own trials, and this is their sum. `depends_on`
relabels `y`'s trials into the model's existing groups when they are not the
trials it was fitted to, exactly as it does for [`smooth`](@ref).
"""
function terminal_logz(
    slds::SLDS{T},
    y;
    ux0=nothing,
    ux=nothing,
    uy=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
    smoothing_iters::Int=_SLQR_PROBE_ITERS,
) where {T<:Real}
    _slds_condition_terminal(slds) || return zero(T)
    data = Data(slds.LDSs[1], y; ux0=ux0, ux=ux, uy=uy)
    variants = _slds_state_trial_variants(slds, length(data.tsteps); depends_on=depends_on)
    return sum(
        _slqr_trial_logz(
            slds, data.ux, variants; ux0=data.ux0, smoothing_iters=smoothing_iters
        ),
    )
end

"""
    terminal_normalizer(model, ux; depends_on, smoothing_iters) -> Vector

Each trial's `log p(terminal = 0 | θ)` under `model`, **whether or not** the model
conditions its score on it: the number that turns a joint score into the
conditional one when subtracted.

That is what makes a fit made on the joint objective (`condition_terminal =
false`) comparable across plant dimensions after the fact. Its score carries this
term, which grows with the number of closed-loop modes whatever the data say.
[`terminal_logz`](@ref) is zero for such a model, because nothing was divided by
it.

- An inverse-LQR `LinearDynamicalSystem` gets the exact value, from the backward
  square-root recursion the conditional fit uses.
- An `SLDS` gets the variational estimate its conditional score divides by: a
  zero-loading copy of the model, smoothed (see [`terminal_logz`](@ref)).

`ux` holds the per-trial input matrices the model was fitted with (`ux_dim × T_i`),
which fix the horizons as well as the inputs; a model with no inputs takes
zero-row matrices of the trials' lengths. Every entry is zero when no state carries
a terminal factor.

The normalizer is a property of the state side alone, so the emission — and any
`depends_on` grouping of it, as in a stitched fit — does not enter. A model whose
`depends_on` splits the *state* parameters (a cost per reward level, say) has one
normalizer per state variant: each trial's value is its own variant's, from a
probe over that variant's trials. The trials are read as the ones the model was
fitted to, labelled by the `depends_on` stored on it; pass `depends_on` to label a
different set (a held-out one) into the same groups.

A `:free` discrete state carries no terminal factor, but the probe still smooths
under its prior with no data to pin it, so an explosive free transition can make
that smoother's factorization fail (`PosDefException`).
"""
function terminal_normalizer(
    slds::SLDS{T},
    ux::AbstractVector{<:AbstractMatrix};
    ux0=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
    smoothing_iters::Int=_SLQR_PROBE_ITERS,
) where {T<:Real}
    lqr = [lds.state_model for lds in slds.LDSs if lds.state_model isa LQRStateModel]
    any(sm -> sm.terminal, lqr) || return zeros(T, length(ux))
    variants = _slds_state_trial_variants(slds, length(ux); depends_on=depends_on)
    return _slqr_trial_logz(
        slds, [Matrix{T}(u) for u in ux], variants; ux0=ux0, smoothing_iters=smoothing_iters
    )
end

function terminal_normalizer(
    lds::LinearDynamicalSystem{T,S},
    ux::AbstractVector{<:AbstractMatrix};
    ux0=nothing,
    depends_on::Union{Nothing,NamedTuple}=nothing,
) where {T<:Real,S<:LQRStateModel{T}}
    sm = lds.state_model
    sm.terminal || return zeros(T, length(ux))
    inputs = [Matrix{T}(u) for u in ux]
    u0 = _normalize_ux0(ux0, sm, length(inputs))
    variants = _state_trial_variants(sm, length(inputs); depends_on=depends_on)
    if variants === nothing
        _lqr_lengths_ok(sm, [size(u, 2) for u in inputs])
        refresh!(sm)
        return [_lqr_terminal_logz(sm, u, view(u0, :, i)) for (i, u) in enumerate(inputs)]
    end
    #= Exact per trial, against the trial's own variant: the non-switching
    normalizer factorizes over trials with nothing shared between them. =#
    out = zeros(T, length(inputs))
    for v in unique(variants)
        vsm = (sm.variants::Vector{S})[v]
        idx = findall(==(v), variants)
        _lqr_lengths_ok(vsm, [size(inputs[n], 2) for n in idx])
        refresh!(vsm)
        for n in idx
            out[n] = _lqr_terminal_logz(vsm, inputs[n], view(u0, :, n))
        end
    end
    return out
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
    _slds_terminal_trial_logz(slds, ux, seq_ends; variants, smoothing_iters) -> Vector or nothing

Each trial's `log Z-hat`, or `nothing` when this model does not condition.

`variants` is each trial's state variant ([`_slds_trial_variants`](@ref)), or
`nothing` for one normalizer over every trial; a trial's value then comes from
its own variant's probe.

Built fresh rather than cached across iterations: the probe's cost scales with
the number of distinct designs, not trials, and a task design repeated across a
session collapses to one smoothed chain.
"""
function _slds_terminal_trial_logz(
    slds::SLDS{T},
    ux,
    seq_ends;
    ux0=nothing,
    variants::Union{Nothing,AbstractVector{Int}}=nothing,
    smoothing_iters::Int=_SLQR_PROBE_ITERS,
) where {T<:Real}
    _slds_condition_terminal(slds) || return nothing
    return _slqr_trial_logz(
        slds,
        _slds_probe_inputs(ux, seq_ends, T),
        variants;
        ux0=ux0,
        smoothing_iters=smoothing_iters,
    )
end

"""
    _slqr_trial_logz(slds, inputs, variants; smoothing_iters) -> Vector

Each trial's `log Z-hat` from freshly built probes, one per state variant (or one
in all when `variants === nothing`), whether or not the model conditions on it.
"""
function _slqr_trial_logz(
    slds::SLDS{T},
    inputs::AbstractVector,
    variants::Union{Nothing,AbstractVector{Int}};
    ux0=nothing,
    smoothing_iters::Int=_SLQR_PROBE_ITERS,
) where {T<:Real}
    probes, sources, trials = _slqr_terminal_probes(
        slds, inputs, variants; ux0=ux0, smoothing_iters=smoothing_iters
    )
    _slqr_probes_logz!(probes, sources; restart=false)
    out = zeros(T, length(inputs))
    for (probe, tr) in zip(probes, trials)
        out[tr] .= view(probe.per_design, probe.design_of)
    end
    return out
end
