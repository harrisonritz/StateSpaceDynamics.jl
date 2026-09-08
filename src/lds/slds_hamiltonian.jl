#=============================================================================
Hamiltonian (inverse-LQR) discrete states in an `SLDS`.

The pieces of the switching M-step that need the inverse-LQR types, which are
defined after `fit_SLDS.jl`. Everything else about a Hamiltonian discrete state
already flows through the shared SLDS path: `state_loglikelihood!` and
`_transition_residual!` are dispatched, `compute_smooth_constants!` fills the
same `SmoothConstants` slots from either model, and the terminal factor is added
by the helpers in `fit_SLDS.jl` itself.
=============================================================================#

"""
    _extract_state_params(sm::HamiltonianStateModel)

The *forward* transition parameters an `SLDS` sampler rolls, read from the cache:
`M`, `Q^fwd = G Σ Gᵀ`, `G h` and the regime's forward input matrix. Presenting
them under the same names a Gaussian state uses lets `_sample_slds_trial!` draw a
switching path without knowing which kind of state it is drawing from.

An `SLDS` member carries a single cost (the discrete state *is* the epoch), so
there is one transition to hand back.

As with a single inverse-LQR model, this rolls the model's own forward flow,
which is unstable by construction — see [`_warn_unstable_rollout`](@ref) and
prefer `simulate_lqr` for trajectories on the stable manifold.
"""
function _extract_state_params(sm::HamiltonianStateModel{T}) where {T<:Real}
    c = sm.cache
    d = _state_latent_dim(sm)
    m = size(sm.Bu, 2)
    return (
        A=c.M[1],
        B=m > 0 ? c.Bfwd[1] : zeros(T, d, 0),
        Q=Matrix{T}(c.Qfwd),
        b=c.bfwd,
        x0=sm.x0,
        P0=sm.P0,
    )
end

"""
    _warn_slds_unstable_rollout(slds, tsteps)

Emit the forward-flow instability warning once per inverse-LQR discrete state a
switching sample would roll through.
"""
function _warn_slds_unstable_rollout(slds::SLDS, tsteps::Int)
    for lds in slds.LDSs
        sm = lds.state_model
        sm isa HamiltonianStateModel && !_is_free(sm) && _warn_unstable_rollout(sm, tsteps)
    end
    return nothing
end

"""
    _prepare_slds!(slds, tsteps)

Entry-point preparation for every discrete state, mirroring what
`_prepare_hamiltonian!` does for a single inverse-LQR model: refresh the derived
cache, check the horizons against the cost schedule, and zero the costate readout
when the emission is not allowed to see it.

Called before any parallel section, since the cache is shared by every trial
workspace. A no-op for a state model with no cache to refresh.
"""
function _prepare_slds!(slds::SLDS, tsteps::AbstractVector{Int})
    for lds in slds.LDSs
        _prepare_slds_regime!(lds, tsteps)
    end
    return nothing
end

_prepare_slds_regime!(::LinearDynamicalSystem, ::AbstractVector{Int}) = nothing

function _prepare_slds_regime!(
    lds::LinearDynamicalSystem{T,S,O}, tsteps::AbstractVector{Int}
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    return _prepare_hamiltonian!(lds, tsteps)
end

"""
    _slds_aggregate_weighted!(suf, tfs, lds, data, weights, sws)

One discrete state's responsibility-weighted sufficient statistics.

A Hamiltonian state needs two passes: the base regression / emission blocks that
every model shares, and the mixed-coordinate blocks its own M-step consumes.
"""
function _slds_aggregate_weighted!(
    hs::HamiltonianSufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T},
    weights::AbstractVector{<:AbstractVector{T}},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    _aggregate_td_suff_stats_weighted!(hs.base, tfs, lds, data, weights, sws)
    _aggregate_hamiltonian_stats_weighted!(hs, tfs, lds, data, weights)
    #=
    Same masking the ungrouped `estep!` applies, at the same point: with
    `observe_costate` off the emission may not read the costate half, and the
    constraint is imposed on the normal equations rather than by projecting `C`
    afterwards. Without it the costate columns leak back in through the SLDS
    emission M-step, which has no reason to know the state model has a costate.
    =#
    mask = _costate_range(lds)
    mask === nothing || _mask_costate_gram!(hs.base, mask)
    return hs
end

_slds_init_suf(hs::HamiltonianSufficientStatistics) = hs.base

#=
Partial ties: sharing some structural parameters across discrete states while
fitting the rest per state.

`tied = [:A, :S]` — one plant, a cost per state — is the configuration a
switching inverse-LQR model is usually for, and it is not expressible as a single
constrained optimization here, because `_HamMStepCtx` gives each *version* a full
copy of every block. Rather than re-layout the packed parameter vector (which the
ungrouped fit and `depends_on` also run through), it is run as two passes:

  1. the shared blocks free and tied across states, the per-state ones frozen;
  2. the per-state blocks free and untied, the shared ones frozen.

Each pass is the existing generalized M-step on a subset of coordinates, and each
accepts only an improvement, so the composition is still non-decreasing — an
alternating maximization rather than a joint one. It converges more slowly than a
joint step would, and that is the price of not disturbing the shared machinery.
=#

const _HAM_STRUCT_BLOCKS = (:A, :S, :Qc, :h, :Bu, :Gref)

"""
    _ham_tied_blocks(tied) -> NTuple{6,Bool}

Which structural blocks `tied` shares, in `_HAM_STRUCT_BLOCKS` order.
`:structure` shares all of them; naming blocks individually shares exactly those.
"""
function _ham_tied_blocks(tied::AbstractVector{Symbol})
    :structure in tied && return ntuple(_ -> true, length(_HAM_STRUCT_BLOCKS))
    return ntuple(i -> _HAM_STRUCT_BLOCKS[i] in tied, length(_HAM_STRUCT_BLOCKS))
end

"""
    _ham_flags_subset(base, keep, want) -> HamiltonianFitFlags

`base` restricted to the blocks whose shared/per-state status matches `want`.
A block the model already freezes stays frozen either way.
"""
function _ham_flags_subset(base::HamiltonianFitFlags, keep::NTuple{6,Bool}, want::Bool)
    on(i, flag) = flag && (keep[i] == want)
    return HamiltonianFitFlags(;
        A=on(1, base.A),
        S=on(2, base.S),
        Qc=on(3, base.Qc),
        h=on(4, base.h),
        Bu=on(5, base.Bu),
        Gref=on(6, base.Gref),
        # `hf` rides with the terminal factor and is treated as a per-state block.
        terminal=base.terminal && !want,
    )
end

function _ham_any_free(f::HamiltonianFitFlags)
    return f.A || f.S || f.Qc || f.h || f.Bu || f.Gref || f.terminal
end

"""
    _ham_structure_phases!(sms, sufs, tied, fit_structure, iters)

Run the structural M-step over the discrete states, honouring whichever blocks
`tied` shares. A whole-block tie (or no tie at all) is one pass; a partial tie is
the two passes described above.
"""
function _ham_structure_phases!(
    sms::AbstractVector,
    sufs::AbstractVector,
    tied::AbstractVector{Symbol},
    fit_structure::Bool,
    fit_noise::Bool,
    iters::Int,
)
    n = length(sms)
    keep = _ham_tied_blocks(tied)
    q_slots = (:noise in tied) ? ones(Int, n) : collect(1:n)
    #=
    One packed parameter layout covers every inverse-LQR state, so which blocks
    are free has to be the same for all of them — as with `fit_bool` upstream,
    a state whose freezes were quietly replaced by another's is worse than an
    error.
    =#
    base = sms[1].fit_flags
    for (k, sm) in enumerate(sms)
        sm.fit_flags == base || throw(
            ArgumentError(
                "discrete state $k has different `fit_flags` from state 1. " *
                "Inverse-LQR states share one packed parameter layout in the " *
                "M-step, so they must freeze the same blocks.",
            ),
        )
    end

    if !fit_structure || all(keep) || !any(keep)
        # One pass: every free block has the same shared/per-state status.
        ab_slots = all(keep) ? ones(Int, n) : collect(1:n)
        ctx = _HamMStepCtx(sufs, sms, ab_slots, q_slots, fit_noise)
        _ham_structure_mstep!(ctx, fit_structure, iters)
        fit_noise && _ham_noise_mstep!(ctx)
        _ham_broadcast_tied!(sms, ab_slots, q_slots)
        return nothing
    end

    shared = _ham_flags_subset(base, keep, true)
    private = _ham_flags_subset(base, keep, false)

    if _ham_any_free(shared)
        ab = ones(Int, n)
        ctx = _HamMStepCtx(sufs, sms, ab, q_slots, false; flags=shared)
        _ham_structure_mstep!(ctx, true, iters)
        _ham_broadcast_tied!(sms, ab, q_slots; blocks=keep)
        for sm in sms
            refresh!(sm)
        end
    end

    if _ham_any_free(private)
        ab = collect(1:n)
        ctx = _HamMStepCtx(sufs, sms, ab, q_slots, fit_noise; flags=private)
        _ham_structure_mstep!(ctx, true, iters)
        fit_noise && _ham_noise_mstep!(ctx)
        # `ab` is the identity here, so only a `:noise` tie broadcasts anything.
        _ham_broadcast_tied!(sms, ab, q_slots; blocks=ntuple(_ -> false, 6))
    elseif fit_noise
        ctx = _HamMStepCtx(sufs, sms, collect(1:n), q_slots, true)
        _ham_noise_mstep!(ctx)
        _ham_broadcast_tied!(sms, collect(1:n), q_slots; blocks=ntuple(_ -> false, 6))
    end
    return nothing
end

function _slds_state_mstep!(
    ldss::AbstractVector{<:LinearDynamicalSystem{T,S,O}},
    sf_state::AbstractVector,
    tied::AbstractVector{Symbol},
    ::AbstractVector{Int},
    ::SmoothWorkspace{T},
    _,
    K::Int,
    ::Int,
    ::Int,
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    sms = [lds.state_model for lds in ldss]

    for k in 1:K
        _fill_mixed_blocks!(sf_state[k], sms[k])
    end

    #=
    A `:free` state is an ordinary regression, so it is updated on its own rather
    than through the constrained context. Mixing modes across discrete states is
    the whole point of `:free`, so neither branch may assume the other is absent.
    =#
    for k in 1:K
        if _is_free(sms[k])
            _free_state_mstep!(ldss[k], sf_state[k])
            refresh!(sms[k])
        end
    end

    lqr = [k for k in 1:K if !_is_free(sms[k])]
    if !isempty(lqr)
        #=
        The constrained step optimizes every inverse-LQR state in one context, so
        the two structural/noise switches apply to all of them at once. Rather
        than quietly taking the first state's, refuse a set that disagrees — a
        frozen state that silently gets fitted is worse than an error.
        =#
        fit_structure = ldss[lqr[1]].fit_bool[3]
        fit_noise = ldss[lqr[1]].fit_bool[4]
        for k in lqr
            (ldss[k].fit_bool[3] == fit_structure && ldss[k].fit_bool[4] == fit_noise) ||
                throw(
                    ArgumentError(
                        "LDSs[$k]: inverse-LQR discrete states are optimized together, " *
                        "so they must agree on the `:A` and `:Q` entries of " *
                        "`fit_bool`. Freeze the same groups on every such state, or " *
                        "use `:free` states, which are updated one at a time and may " *
                        "differ.",
                    ),
                )
        end
        _ham_structure_phases!(
            sms[lqr],
            [sf_state[k] for k in lqr],
            tied,
            fit_structure,
            fit_noise,
            maximum(sms[k].mstep_iters for k in lqr),
        )
        for k in lqr
            refresh!(sms[k])
        end
    end
    return collect(1:K)
end
