#=============================================================================
LQR latents — sufficient statistics, M-step and ELBO.

The E-step gives the usual linear-Gaussian smoother output on `z = [x; λ]`. The
M-step must return parameters whose transition still has the LQR form,
which is done in *mixed* coordinates: with

    w_t = [x_t; λ_{t+1}],    v_t = [x_{t+1}; λ_t],

the constraint is that the regression `v_t ≈ 𝓔_k w_t + h + Bu u_t` has
`𝓔_k = [A −S; Q_k Aᵀ]` with `S`, `Q_k` positive semidefinite. The mixed
form is linear in these matrices; the optimizer represents each as `L Lᵀ`
to preserve convexity, while the forward transition is rational in `A`.

The rearrangement is free: every second moment of `(w_t, v_t)` is a block of the
joint second moment of `(z_t, z_{t+1})` the smoother already returns, so nothing
about the E-step changes. What the change of coordinates does cost is a Jacobian
term. Since `z_{t+1} − M_k z_t − b = G(v_t − 𝓔_k w_t − h)` with
`G = [I  S A⁻ᵀ; 0  −A⁻ᵀ]`,

    log det Q^fwd = log det Σ − 2 log|det A|,

so the M-step objective carries `+N log|det A|`. Dropping it would converge
EM to the wrong `A`, silently.

Profiling `Σ = R/N` out of

    F = N log|det A| − (N/2) log det Σ − ½ tr(Σ⁻¹ R(θ))

leaves the objective this file minimizes over `θ = (A, S, Q_{1:K}, h, Bu, hf)`:

    g(θ) = (N/2) log det R(θ) + (N_f/2) log det R_f(θ) − N log|det A|,

which is smooth with cheap exact gradients. It is optimized by L-BFGS warm
started at the incoming parameters and accepted only if it improved, which makes
the step a generalized M-step: the ELBO cannot decrease.

A `:hold` model (the infinite-horizon regulator, see `lqr_hold.jl`) is fitted in
the same objective. Its residual is `Lh z_{t+1} − Θh ω_t` with
`Lh = [I S; −P I]`, `Θh = [A 0 F; 0 0 G]` and `P` the stabilizing DARE
solution, over the forward statistics; `det Lh = det(I + S P)` takes the place
of `det A`, so a hold unit carries `−N log det(I + S P)`. Hold and `:lqr` units
share block copies and noise versions freely — that is how a switching model
ties one plant across a control state and a hold state — and the gradient
through `P` is one adjoint Stein solve (see `_hold_unit_gradient!`).
=============================================================================#

"""
    LQRSufficientStatistics{T}

Aggregated E-step statistics for a [`LQRStateModel`](@ref).

`base` carries the initial-state and emission halves in whatever layout the
emission's own allocator produces — one [`SufficientStatistics`](@ref), or a
composite's `NamedTuple` of them. Those halves are state-model-independent, so
the shared aggregator fills them and the shared `x0` / `P0` / emission updates
consume them unchanged; `_state_suf` picks the block the state side reads.

The state half is *per cost regime*, because a transition's regime decides which
`Q_k` it constrains. Each regime's block is laid out exactly like the Gaussian
path's `dyn_*` triple, over the regressor `[z_t; 1; u_t]`:

- `zz[k]`: `Σ_{t∈R_k} E[[z_t;1;u_t][z_t;1;u_t]ᵀ]`   (`reg × reg`)
- `zy[k]`: `Σ_{t∈R_k} E[[z_t;1;u_t] z_{t+1}ᵀ]`      (`reg × 2n`)
- `yy[k]`: `Σ_{t∈R_k} E[z_{t+1} z_{t+1}ᵀ]`          (`2n × 2n`)
- `nk[k]`: how many transitions regime `k` owns

`term_zz[k]` / `term_n[k]` hold
`Σ_{n: k(T_n)=k} E[[z_T;1;u_T][z_T;1;u_T]ᵀ]` and the corresponding
trial count. Keeping these statistics per terminal cost regime is required for
ragged trials, whose endpoints can land on different schedule entries.

`Zw` / `Xv` / `Yv` / `Omega` are the *mixed-coordinate* blocks derived from those
by [`_fill_mixed_blocks!`](@ref) at the start of each M-step; they live here
rather than in a local so an EM run allocates them once.
"""
mutable struct LQRSufficientStatistics{T<:Real,B}
    const base::B
    const zz::Vector{Matrix{T}}
    const zy::Vector{Matrix{T}}
    const yy::Vector{Matrix{T}}
    const nk::Vector{T}
    const term_zz::Vector{Matrix{T}}
    const term_n::Vector{T}
    # Mixed-coordinate blocks (derived; see `_fill_mixed_blocks!`).
    const Zw::Vector{Matrix{T}}
    const Xv::Vector{Matrix{T}}
    const Yv::Matrix{T}
    const Omega::Vector{Matrix{T}}
    #=
    The distinct terminal designs among the aggregated trials, and how many
    trials each covers. `log Z` depends on a trial only through its inputs and
    its horizon, so a task design repeated across a session is integrated once.
    Deduplicated here rather than at use: the aggregator sees each trial once,
    while the objective is evaluated many times per M-step.
    =#
    const terminal_inputs::Vector{Matrix{T}}
    const terminal_counts::Vector{T}
    const terminal_ux0::Vector{Vector{T}}
    #= Where each design starts on the cost schedule. `log Z` reads the schedule
    from the trial's end, so the offset is part of what makes two designs the
    same. Empty for statistics that never saw one (every switching unit). =#
    const terminal_offsets::Vector{Int}
    #= One block per `sm.switches` entry (unused unless it carries an entry
    prior): `Σ E[ω ωᵀ]` over the trials that cross it, `ω = [z_b; z_{b+1}; 1; u_b]`,
    and their count. The entry transition is in these and in no regime's `zz`. =#
    const entry_ww::Vector{Matrix{T}}
    const entry_n::Vector{T}
    #= A `:causal` model's statistics, per horizon (`causal_keys`, the
    `(offset, length)` pairs of `_causal_key`) and per transition `t` of it — its
    transition and noise vary with `t`, so they cannot be summed over time.
    Over `w̃ = [z_t; 1; u]`: `causal_zz[h][t] = Σ E[w̃ w̃ᵀ]`,
    `causal_zy[h][t] = Σ E[w̃ z_{t+1}ᵀ]`, `causal_yy[h][t] = Σ E[z_{t+1} z_{t+1}ᵀ]`,
    and `causal_n[h]` the transitions it holds. Without a schedule every trial
    is aligned at the end of the longest one's horizon (see
    `_aggregate_causal_stats!`). Empty in every other mode. =#
    const causal_keys::Vector{NTuple{2,Int}}
    const causal_zz::Vector{Vector{Matrix{T}}}
    const causal_zy::Vector{Vector{Matrix{T}}}
    const causal_yy::Vector{Vector{Matrix{T}}}
    const causal_n::Vector{T}
end

function _initialize_td_sufficient_statistics(
    ::Type{T}, lds::LinearDynamicalSystem{T,S,O}, tsteps_per_trial::AbstractVector{Int}
) where {T<:Real,S<:LQRStateModel{T},O<:AbstractObservationModel{T}}
    return _wrap_lqr_suff_stats(
        _base_td_sufficient_statistics(T, lds, tsteps_per_trial), lds, tsteps_per_trial
    )
end

#=
A composite emission allocates a `NamedTuple` of per-member blocks, so this
method has to be as specific in `O` as the composite's own — otherwise the two
are ambiguous. Both just wrap whatever the emission's allocator returned.
=#
function _initialize_td_sufficient_statistics(
    ::Type{T}, lds::LinearDynamicalSystem{T,S,O}, tsteps_per_trial::AbstractVector{Int}
) where {T<:Real,S<:LQRStateModel{T},O<:CompositeObservationModel{T}}
    views = _obs_views(lds)
    base = map(v -> _base_td_sufficient_statistics(T, v, tsteps_per_trial), views)
    return _wrap_lqr_suff_stats(base, lds, tsteps_per_trial)
end

"""
    _wrap_lqr_suff_stats(base, lds, tsteps_per_trial)

Allocate the per-regime state-side blocks around an already-built `base`.
"""
function _wrap_lqr_suff_stats(
    base, lds::LinearDynamicalSystem{T,S,O}, tsteps_per_trial::AbstractVector{Int}
) where {T<:Real,S<:LQRStateModel{T},O<:AbstractObservationModel{T}}
    sm = lds.state_model
    d = lds.latent_dim
    m = lds.ux_dim
    K = _nregimes(sm)
    reg = d + 1 + m
    return LQRSufficientStatistics{T,typeof(base)}(
        base,
        [zeros(T, reg, reg) for _ in 1:K],
        [zeros(T, reg, d) for _ in 1:K],
        [zeros(T, d, d) for _ in 1:K],
        zeros(T, K),
        [zeros(T, d + 1 + m, d + 1 + m) for _ in 1:K],
        zeros(T, K),
        [zeros(T, reg, reg) for _ in 1:K],
        [zeros(T, d, reg) for _ in 1:K],
        zeros(T, d, d),
        [zeros(T, d + 1 + m, d + 1 + m) for _ in 1:K],
        Matrix{T}[],
        T[],
        Vector{T}[],
        Int[],
        [zeros(T, 2d + 1 + m, 2d + 1 + m) for _ in sm.switches],
        zeros(T, length(sm.switches)),
        NTuple{2,Int}[],
        Vector{Matrix{T}}[],
        Vector{Matrix{T}}[],
        Vector{Matrix{T}}[],
        T[],
    )
end

"""
    _regime_runs(sm, tsteps) -> Vector{Tuple{Int,Int,Int}}

The transitions of a trial of length `tsteps`, grouped into maximal runs of one
cost regime as `(k, t0, t1)` — regime `k` owns transitions `t0 … t1`, i.e. the
steps `z_{t0} → z_{t0+1}` through `z_{t1} → z_{t1+1}`.

Schedules in practice are a handful of contiguous epochs, so grouping lets the
aggregator do the mean-side work with one GEMM per run instead of one rank-1
update per timestep.
"""
function _regime_runs(sm::LQRStateModel, tsteps::Int)
    isempty(sm.switches) || return _regime_runs_switched(sm, tsteps)
    runs = Tuple{Int,Int,Int}[]
    tsteps >= 2 || return runs
    t0 = 1
    k0 = _regime(sm, 1)
    for t in 2:(tsteps - 1)
        k = _regime(sm, t)
        if k != k0
            push!(runs, (k0, t0, t - 1))
            t0, k0 = t, k
        end
    end
    push!(runs, (k0, t0, tsteps - 1))
    return runs
end

"""
    _aggregate_lqr_stats!(hs, tfs, lds, data[, trials])

Accumulate the per-regime state-side statistics from the smoother output.

`E[z_t z_tᵀ] = x_t x_tᵀ + P_t` and `E[z_t z_{t+1}ᵀ] = x_t x_{t+1}ᵀ + Cov(z_t,
z_{t+1})`, where the smoother stores `p_smooth_tt1[:, :, t] = Cov(z_t, z_{t-1})`
— hence the adjoint on that term. Means go through GEMM per run; the covariance
sums are the unavoidable per-timestep part, exactly as on the Gaussian path.

`trials` restricts the sum to a subset, every other trial contributing nothing.
Trials are summed in fixed chunks, in parallel for a large aggregation (see
`serial_below`), with a result that does not depend on the thread count.
The blocks are zeroed first either way, so the result is that subset's own
statistics rather than an accumulation onto whatever was there — which is what
lets [`trial_elbos`](@ref) reach one trial's state Q-term through the same
aggregator the whole-dataset path uses.
"""
function _aggregate_lqr_stats!(
    hs::LQRSufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T},
    trials::AbstractVector{Int}=Base.OneTo(length(tfs));
    serial_below::Int=_AGGREGATE_SERIAL_WORK,
) where {T<:Real,S<:LQRStateModel{T},O<:AbstractObservationModel{T}}
    sm = lds.state_model
    _is_causal(sm) && return _aggregate_causal_stats!(hs, tfs, lds, data, trials)
    d = lds.latent_dim
    m = lds.ux_dim
    reg = d + 1 + m
    K = _nregimes(sm)

    empty!(hs.terminal_inputs)
    empty!(hs.terminal_counts)
    empty!(hs.terminal_ux0)
    empty!(hs.terminal_offsets)
    if sm.terminal && sm.condition_terminal
        index = Dict{Tuple{Matrix{T},Vector{T},Int},Int}()
        for i in trials
            u = data.ux[i]
            u0 = Vector{T}(view(data.ux0, :, i))
            off = _trial_cost_offset(data, i)
            slot = get(index, (u, u0, off), 0)
            if slot == 0
                push!(hs.terminal_inputs, Matrix{T}(u))
                push!(hs.terminal_counts, one(T))
                push!(hs.terminal_ux0, u0)
                push!(hs.terminal_offsets, off)
                index[(hs.terminal_inputs[end], u0, off)] = length(hs.terminal_inputs)
            else
                hs.terminal_counts[slot] += one(T)
            end
        end
    end

    #=
    The per-trial sums run in fixed chunks of trials, each into its own partial,
    reduced into `hs` in chunk order: the result depends on the trial count
    alone, not on the thread count. Small aggregations run the same chunks on
    the calling task, which gives the same bits without waking any threads.
    =#
    chunks = _reduction_chunks(length(trials))
    work = sum(i -> size(tfs[i].x_smooth, 2), trials; init=0) * reg^2
    nbuf = work < serial_below ? 1 : min(length(chunks), Threads.nthreads())
    partials = [_LQRStatsPartial(hs) for _ in 1:nbuf]
    for k in 1:K
        fill!(hs.zz[k], zero(T))
        fill!(hs.zy[k], zero(T))
        fill!(hs.yy[k], zero(T))
        hs.nk[k] = zero(T)
        fill!(hs.term_zz[k], zero(T))
        hs.term_n[k] = zero(T)
    end
    foreach(Z -> fill!(Z, zero(T)), hs.entry_ww)
    fill!(hs.entry_n, zero(T))
    function accumulate!(slot, chunk)
        p = _zero!(partials[slot])
        for j in chunk
            trial = trials[j]
            _lqr_stats_trial!(
                p,
                _with_cost_offset(sm, _trial_cost_offset(data, trial)),
                tfs[trial],
                data.ux[trial],
                d,
                m,
                reg,
            )
        end
        return nothing
    end
    function reduce!(slot)
        p = partials[slot]
        for k in 1:K
            hs.zz[k] .+= p.zz[k]
            hs.zy[k] .+= p.zy[k]
            hs.yy[k] .+= p.yy[k]
            hs.nk[k] += p.nk[k]
            hs.term_zz[k] .+= p.term_zz[k]
            hs.term_n[k] += p.term_n[k]
        end
        for e in eachindex(hs.entry_ww)
            hs.entry_ww[e] .+= p.entry_ww[e]
            hs.entry_n[e] += p.entry_n[e]
        end
        return nothing
    end
    if nbuf == 1
        for chunk in chunks
            accumulate!(1, chunk)
            reduce!(1)
        end
    else
        _foreach_chunk_wave(accumulate!, reduce!, chunks, nbuf)
    end

    return _finalize_lqr_stats!(hs, sm, d, m, reg, K)
end

# Below this many (timestep × regressor²) units an aggregation runs serially.
const _AGGREGATE_SERIAL_WORK = 1 << 22

#=
One chunk's share of the per-regime state statistics.
=#
struct _LQRStatsPartial{T<:Real}
    zz::Vector{Matrix{T}}
    zy::Vector{Matrix{T}}
    yy::Vector{Matrix{T}}
    nk::Vector{T}
    term_zz::Vector{Matrix{T}}
    term_n::Vector{T}
    entry_ww::Vector{Matrix{T}}
    entry_n::Vector{T}
end

function _LQRStatsPartial(hs::LQRSufficientStatistics{T}) where {T<:Real}
    return _LQRStatsPartial{T}(
        [similar(Z) for Z in hs.zz],
        [similar(Z) for Z in hs.zy],
        [similar(Z) for Z in hs.yy],
        similar(hs.nk),
        [similar(Z) for Z in hs.term_zz],
        similar(hs.term_n),
        [similar(Z) for Z in hs.entry_ww],
        similar(hs.entry_n),
    )
end

function _zero!(p::_LQRStatsPartial{T}) where {T}
    foreach(Z -> fill!(Z, zero(T)), p.zz)
    foreach(Z -> fill!(Z, zero(T)), p.zy)
    foreach(Z -> fill!(Z, zero(T)), p.yy)
    fill!(p.nk, zero(T))
    foreach(Z -> fill!(Z, zero(T)), p.term_zz)
    fill!(p.term_n, zero(T))
    foreach(Z -> fill!(Z, zero(T)), p.entry_ww)
    fill!(p.entry_n, zero(T))
    return p
end

"""
    _lqr_stats_trial!(acc, sm, fs, ux, d, m, reg[, w, with_cov])

One trial's contribution to the per-regime statistics, every moment scaled by
`w`, added into `acc` (the upper triangles of the symmetric blocks;
`_finalize_lqr_stats!` mirrors them). `with_cov = false` leaves out the
smoothed-covariance terms, for a caller that sums a covariance shared by many
trials once (the terminal-conditioning probe).
"""
function _lqr_stats_trial!(
    acc,
    sm::LQRStateModel,
    fs::FilterSmooth{T},
    ux::AbstractMatrix,
    d::Int,
    m::Int,
    reg::Int,
    w::T=one(T),
    with_cov::Bool=true,
) where {T<:Real}
    x = fs.x_smooth::Matrix{T}
    p_smooth = fs.p_smooth::Array{T,3}
    p_tt1 = fs.p_smooth_tt1::Array{T,3}
    T_n = size(x, 2)

    for (k, t0, t1) in _regime_runs(sm, T_n)
        zz = acc.zz[k]
        zy = acc.zy[k]
        yy = acc.yy[k]
        len = t1 - t0 + 1
        acc.nk[k] += w * T(len)

        x_prev = tview(x, :, t0:t1)
        x_next = tview(x, :, (t0 + 1):(t1 + 1))

        # Mean parts (BLAS-3 over the run).
        BLAS.syrk!('U', 'N', w, x_prev, one(T), tview(zz, 1:d, 1:d))
        mul!(view(zy, 1:d, :), x_prev, transpose(x_next), w, one(T))
        BLAS.syrk!('U', 'N', w, x_next, one(T), yy)
        for t in t0:t1, i in 1:d
            zz[i, d + 1] += w * x[i, t]
            zy[d + 1, i] += w * x[i, t + 1]
        end
        zz[d + 1, d + 1] += w * T(len)

        # Covariance parts.
        with_cov && @views for t in t0:t1
            zz[1:d, 1:d] .+= w .* p_smooth[:, :, t]
            yy .+= w .* p_smooth[:, :, t + 1]
            # Cov(z_t, z_{t+1}) = Cov(z_{t+1}, z_t)ᵀ = p_smooth_tt1[:,:,t+1]ᵀ
            zy[1:d, :] .+= w .* adjoint(p_tt1[:, :, t + 1])
        end

        if m > 0
            u_run = tview(ux, :, t0:t1)
            mul!(view(zz, 1:d, (d + 2):reg), x_prev, transpose(u_run), w, one(T))
            mul!(view(zy, (d + 2):reg, :), u_run, transpose(x_next), w, one(T))
            BLAS.syrk!('U', 'N', w, u_run, one(T), tview(zz, (d + 2):reg, (d + 2):reg))
            for t in t0:t1, j in 1:m
                zz[d + 1, d + 1 + j] += w * ux[j, t]
            end
        end
    end

    #=
    Terminal factor statistics: `E[[z_T; 1; u_T][z_T; 1; u_T]ᵀ]` summed over
    trials. The input block is there for the terminal *reference* — a reach
    is scored against where the target was, so the terminal residual carries
    `+Q_f G_r u_T`.
    =#
    if sm.terminal
        kT = _terminal_regime(sm, T_n)
        term_zz = acc.term_zz[kT]
        xT = tview(x, :, T_n)
        BLAS.ger!(w, xT, xT, tview(term_zz, 1:d, 1:d))
        with_cov && @views term_zz[1:d, 1:d] .+= w .* p_smooth[:, :, T_n]
        for i in 1:d
            term_zz[i, d + 1] += w * x[i, T_n]
        end
        if m > 0
            uT = tview(ux, :, T_n)
            @views mul!(term_zz[1:d, (d + 2):(d + 1 + m)], xT, transpose(uT), w, one(T))
            BLAS.ger!(w, uT, uT, tview(term_zz, (d + 2):(d + 1 + m), (d + 2):(d + 1 + m)))
            for j in 1:m
                term_zz[d + 1, d + 1 + j] += w * ux[j, T_n]
            end
        end
        acc.term_n[kT] += w
    end
    isempty(sm.switches) || _lqr_switch_stats_trial!(acc, sm, fs, ux, d, m, w, with_cov)
    return nothing
end

"""
    _finalize_lqr_stats!(hs, sm, d, m, reg, K) -> hs

Mirror the upper triangles the accumulation loops filled, for both the
transition Grams and the terminal one. Shared by the plain and the
responsibility-weighted aggregators, which differ only in how they accumulate.
"""
function _finalize_lqr_stats!(
    hs::LQRSufficientStatistics{T}, sm::LQRStateModel{T}, d::Int, m::Int, reg::Int, K::Int
) where {T<:Real}
    for k in 1:K
        zz = hs.zz[k]
        LinearAlgebra.copytri!(tview(zz, 1:d, 1:d), 'U')
        if m > 0
            LinearAlgebra.copytri!(tview(zz, (d + 2):reg, (d + 2):reg), 'U')
            @views zz[(d + 2):reg, d + 1] .= zz[d + 1, (d + 2):reg]
        end
        @views zz[d + 1, 1:d] .= zz[1:d, d + 1]
        if m > 0
            @views zz[(d + 2):reg, 1:d] .= transpose(zz[1:d, (d + 2):reg])
        end
        LinearAlgebra.copytri!(hs.yy[k], 'U')
    end
    if sm.terminal
        for k in 1:K
            term_zz = hs.term_zz[k]
            @views term_zz[d + 1, 1:d] .= term_zz[1:d, d + 1]
            term_zz[d + 1, d + 1] = hs.term_n[k]
            if m > 0
                ur = (d + 2):(d + 1 + m)
                @views term_zz[ur, 1:d] .= transpose(term_zz[1:d, ur])
                @views term_zz[ur, d + 1] .= term_zz[d + 1, ur]
                LinearAlgebra.copytri!(tview(term_zz, ur, ur), 'U')
            end
        end
    end
    return hs
end

"""
    _aggregate_lqr_stats_weighted!(hs, tfs, lds, data, weights) -> hs

The responsibility-weighted counterpart, for one discrete state of an `SLDS`.

`weights[trial][t]` is that state's responsibility `γₖ(t)`. The convention is
the one every other SLDS aggregator and kernel uses: the dynamics factor at `t`
couples `(z_{t-1}, z_t)`, so the transition *out of* `t` carries `γₖ(t+1)`, and
the terminal factor at `T` carries `γₖ(T)`.

Everything the plain aggregator counts once, this counts `γ` times — including
`nk`, which therefore holds the **effective** count `n̄ₖ = Σ γₖ(t)` rather than a
number of timesteps. The M-step's Jacobian term is scaled by that same `n̄ₖ`,
which is what keeps the generalized M-step monotone under unbalanced
responsibilities.

Accumulates per timestep rather than BLAS-3 over a schedule run: the weight
varies within a run, so there is no run to batch over.
"""
function _aggregate_lqr_stats_weighted!(
    hs::LQRSufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T},
    weights::AbstractVector{<:AbstractVector{T}};
    exit_weights::Union{Nothing,AbstractVector}=nothing,
    dyn_weights::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real,S<:LQRStateModel{T},O<:AbstractObservationModel{T}}
    sm = lds.state_model
    d = lds.latent_dim
    m = lds.ux_dim
    reg = d + 1 + m
    K = _nregimes(sm)

    for k in 1:K
        fill!(hs.zz[k], zero(T))
        fill!(hs.zy[k], zero(T))
        fill!(hs.yy[k], zero(T))
        hs.nk[k] = zero(T)
    end
    for k in 1:K
        fill!(hs.term_zz[k], zero(T))
        hs.term_n[k] = zero(T)
    end

    for trial in 1:length(tfs)
        fs = tfs[trial]
        x = fs.x_smooth::Matrix{T}
        p_smooth = fs.p_smooth::Array{T,3}
        p_tt1 = fs.p_smooth_tt1::Array{T,3}
        T_n = size(x, 2)
        #= `AbstractMatrix`, not `Matrix`: a `Data` built from an array the
        caller already owns holds *views* of it rather than copies, which is the
        ordinary case when trials are slices of one session-wide matrix. `ux` is
        only ever reached through `tview` below, which produces a `SubArray`
        either way, so nothing downstream can tell the difference. =#
        ux = data.ux[trial]::AbstractMatrix{T}
        w = weights[trial]
        #= The transitions' own weights. With an entry prior these are the
        responsibilities less the mass that entered the state at that bin, whose
        transition is the entry one (`set_boundaries!`); otherwise the
        responsibilities themselves. =#
        wdyn = dyn_weights === nothing ? w : dyn_weights[trial]

        #=
        Everything handed to `ger!` is hoisted through `tview` with a concrete
        annotation rather than `@views`: BLAS wrappers have no fallback method,
        so a view whose element type JET cannot pin becomes a "no matching
        method" report on every union-split branch.
        =#
        for t in 1:(T_n - 1)
            wt = wdyn[t + 1]::T               # the factor coupling (z_t, z_{t+1})
            iszero(wt) && continue
            k = _regime(sm, t)
            zz = hs.zz[k]::Matrix{T}
            zy = hs.zy[k]::Matrix{T}
            yy = hs.yy[k]::Matrix{T}
            hs.nk[k] += wt

            z_prev = tview(x, :, t)
            z_next = tview(x, :, t + 1)

            BLAS.ger!(wt, z_prev, z_prev, tview(zz, 1:d, 1:d))
            BLAS.ger!(wt, z_prev, z_next, tview(zy, 1:d, :))
            BLAS.ger!(wt, z_next, z_next, yy)
            @views begin
                tview(zz, 1:d, 1:d) .+= wt .* p_smooth[:, :, t]
                tview(zy, 1:d, :) .+= wt .* adjoint(p_tt1[:, :, t + 1])
                yy .+= wt .* p_smooth[:, :, t + 1]
            end

            for i in 1:d
                zz[i, d + 1] += wt * z_prev[i]
                zy[d + 1, i] += wt * z_next[i]
            end
            zz[d + 1, d + 1] += wt

            if m > 0
                u_prev = tview(ux, :, t)
                BLAS.ger!(wt, z_prev, u_prev, tview(zz, 1:d, (d + 2):reg))
                BLAS.ger!(wt, u_prev, z_next, tview(zy, (d + 2):reg, :))
                BLAS.ger!(wt, u_prev, u_prev, tview(zz, (d + 2):reg, (d + 2):reg))
                for j in 1:m
                    zz[d + 1, d + 1 + j] += wt * u_prev[j]
                end
            end
        end

        if sm.terminal
            wT = w[T_n]::T
            iszero(wT) || _add_terminal_moment!(
                hs, _terminal_regime(sm, T_n), wT, x, p_smooth, ux, T_n
            )
        end
        #=
        Exit bridges (`set_boundaries!`): the same terminal factor, applied at
        every exit from this state, weighted by the exit probability
        `q(s_t = k, s_{t+1} ≠ k)`. Its statistics land on the bridge's own cost
        regime, so the structural M-step fits that cost from every segment end,
        exits and trial ends alike.
        =#
        if exit_weights !== nothing
            e = exit_weights[trial]
            kb = _bridge_regime(sm)
            for t in 1:(T_n - 1)
                et = e[t]::T
                iszero(et) || _add_terminal_moment!(hs, kb, et, x, p_smooth, ux, t)
            end
        end
    end

    return _finalize_lqr_stats!(hs, sm, d, m, reg, K)
end

"""
    _add_terminal_moment!(hs, k, w, x, p_smooth, ux, t)

Add `w · E[[z_t; 1; u_t][z_t; 1; u_t]ᵀ]` to regime `k`'s terminal statistics and
`w` to its count: one weighted terminal factor at `t`. The end-of-trial factor
uses it at `t = T`, an exit bridge at its exit.
"""
function _add_terminal_moment!(
    hs::LQRSufficientStatistics{T},
    k::Int,
    w::T,
    x::Matrix{T},
    p_smooth::Array{T,3},
    ux::AbstractMatrix{T},
    t::Int,
) where {T<:Real}
    d = size(x, 1)
    m = size(ux, 1)
    term_zz = hs.term_zz[k]
    xt = tview(x, :, t)
    BLAS.ger!(w, xt, xt, tview(term_zz, 1:d, 1:d))
    @views term_zz[1:d, 1:d] .+= w .* p_smooth[:, :, t]
    for i in 1:d
        term_zz[i, d + 1] += w * x[i, t]
    end
    if m > 0
        ut = tview(ux, :, t)
        BLAS.ger!(w, xt, ut, tview(term_zz, 1:d, (d + 2):(d + 1 + m)))
        BLAS.ger!(w, ut, ut, tview(term_zz, (d + 2):(d + 1 + m), (d + 2):(d + 1 + m)))
        for j in 1:m
            term_zz[d + 1, d + 1 + j] += w * ut[j]
        end
    end
    hs.term_n[k] += w
    return hs
end

"""
    _fill_mixed_blocks!(hs, sm)

Rearrange the per-regime `z`-space statistics into the mixed coordinates the
structural regression lives in: `Zw[k] = Σ E[w̃ w̃ᵀ]` over `w̃ = [w_t; 1; u_t]`,
`Xv[k] = Σ E[v_t w̃ᵀ]`, and `Yv = Σ_k Σ E[v_t v_tᵀ]` (only the total is needed,
since the residual scatter sums over regimes).

Every entry is a block copy — `w_t = [x_t; λ_{t+1}]` and `v_t = [x_{t+1}; λ_t]`
each take one half from `z_t` and one from `z_{t+1}`, so the four blocks of a
mixed moment are four blocks of `(Σ E[z_tz_tᵀ], Σ E[z_tz_{t+1}ᵀ],
Σ E[z_{t+1}z_{t+1}ᵀ])`, transposed where the pairing reverses.
"""
function _fill_mixed_blocks!(
    hs::LQRSufficientStatistics{T}, sm::LQRStateModel{T}
) where {T<:Real}
    n = _plant_dim(sm)
    d = 2n
    K = _nregimes(sm)
    reg = size(hs.zz[1], 1)
    m = reg - d - 1
    xr, lr = 1:n, (n + 1):d      # state rows / costate rows of a `z`

    #=
    In `:free` mode the mixed coordinates *are* the forward ones — the regressor
    is `[z_t; 1; u_t]` and the response `z_{t+1}`, with no interleaving — so the
    rearrangement below collapses to three copies. `:hold` mode regresses
    `Lh z_{t+1}` on the same `[z_t; 1; u_t]`, and since `Lh` is a parameter it is
    applied inside the objective, so its blocks are the forward ones too.
    =#
    _is_causal(sm) && return hs              # read per horizon by the causal unit
    if _is_free(sm) || _is_hold(sm)
        copyto!(hs.Zw[1], hs.zz[1])
        copyto!(hs.Xv[1], transpose(hs.zy[1]))
        copyto!(hs.Yv, hs.yy[1])
        for k in 1:K
            copyto!(hs.Omega[k], hs.term_zz[k])
        end
        return hs
    end

    fill!(hs.Yv, zero(T))
    for k in 1:K
        P = hs.zz[k]                     # [z;1;u] Gram
        Xc = hs.zy[k]                    # [z;1;u] × z_{t+1}
        Y = hs.yy[k]
        Zw = hs.Zw[k]
        Xv = hs.Xv[k]
        fill!(Zw, zero(T))
        fill!(Xv, zero(T))

        @views begin
            # S_ww = E[w wᵀ],  w = [x_t; λ_{t+1}]
            Zw[xr, xr] .= P[xr, xr]
            Zw[xr, lr] .= Xc[xr, lr]
            Zw[lr, xr] .= transpose(Xc[xr, lr])
            Zw[lr, lr] .= Y[lr, lr]

            # S_vw = E[v wᵀ],  v = [x_{t+1}; λ_t]
            Xv[xr, xr] .= transpose(Xc[xr, xr])
            Xv[xr, lr] .= Y[xr, lr]
            Xv[lr, xr] .= P[lr, xr]
            Xv[lr, lr] .= Xc[lr, lr]

            # S_vv = E[v vᵀ], accumulated across regimes
            hs.Yv[xr, xr] .+= Y[xr, xr]
            hs.Yv[xr, lr] .+= transpose(Xc[lr, xr])
            hs.Yv[lr, xr] .+= Xc[lr, xr]
            hs.Yv[lr, lr] .+= P[lr, lr]

            # Bias column: Σ E[w] and Σ E[v]
            Zw[xr, d + 1] .= P[xr, d + 1]
            Zw[lr, d + 1] .= Xc[d + 1, lr]
            Zw[d + 1, 1:d] .= Zw[1:d, d + 1]
            Zw[d + 1, d + 1] = hs.nk[k]
            Xv[xr, d + 1] .= Xc[d + 1, xr]
            Xv[lr, d + 1] .= P[lr, d + 1]

            if m > 0
                ur = (d + 2):reg
                # Σ E[w] uᵀ and Σ E[v] uᵀ; `Xc[ur, :]` holds Σ u z_{t+1}ᵀ.
                Zw[xr, ur] .= P[xr, ur]
                Zw[lr, ur] .= transpose(Xc[ur, lr])
                Zw[ur, 1:d] .= transpose(Zw[1:d, ur])
                Zw[ur, ur] .= P[ur, ur]
                Zw[d + 1, ur] .= P[d + 1, ur]
                Zw[ur, d + 1] .= P[ur, d + 1]
                Xv[xr, ur] .= transpose(Xc[ur, xr])
                Xv[lr, ur] .= P[lr, ur]
            end
        end
    end

    for k in 1:K
        copyto!(hs.Omega[k], hs.term_zz[k])
    end
    return hs
end

# ============================================================================
# Structural M-step
# ============================================================================

"""
    _sym!(out, D) -> out

`½(D + Dᵀ)` into preallocated `out`.

The gradient of a symmetric-constrained block has to be symmetrized before it
goes into the parameter vector. This runs once per such block per gradient
evaluation, so a freshly allocated `n × n` each time is pure churn at a large
plant dimension.
"""
@inline function _sym!(out::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T<:Real}
    @inbounds for j in axes(D, 2), i in axes(D, 1)
        out[i, j] = T(0.5) * (D[i, j] + D[j, i])
    end
    return out
end

"""
    _LQRPack

Where each structural block sits in the flat L-BFGS parameter vector.

The vector is laid out **block-major**: all copies of `A`, then all copies of
`S`, then `Qc`, and so on. That is what lets a *partial* tie be one joint
optimization — sharing `A` and `S` across discrete states while fitting `Qc` per
state is just a different count of copies per block, rather than a different
problem. A version-major layout (one full copy of every block per version, which
is what this was) can only express "all blocks shared" or "no blocks shared".

`S` and `Qc` store the lower triangle of a Cholesky factor `L`, with a
log-parameterized positive diagonal and actual matrices `L Lᵀ`. This uses the
identifiable `n(n+1)/2` coordinates of a positive-definite matrix rather than a
redundant `n²` square factor. A fitted block must start positive definite; exact
singular costs remain supported when frozen, but are rejected when fitting
instead of silently becoming rank-locked.

A frozen block (see [`LQRFitFlags`](@ref)) has width zero and is neither
packed nor updated, so freezing shrinks the problem rather than projecting its
solution. `Qc`'s copy spans its regimes contiguously.

Copies of `Qc` need not all carry the same number of regimes: `Kq[v]` is copy
`v`'s count and `qoff[v]` its offset within the block. A `:hold` state carries
one cost while an `:lqr` state beside it in a switching model may carry a running
and a terminal cost, and giving each copy its own count is what lets both sit in
one packed problem without padding either. `K` is the largest count, and `w` for
the cost block is its width at that count — only ever read as "frozen or not".
"""
struct _LQRPack
    n::Int
    d::Int
    m::Int
    K::Int
    nv::NTuple{7,Int}      # copies of each block
    w::NTuple{7,Int}       # width of one copy (0 when frozen)
    base::NTuple{7,Int}    # 0-based start of each block's run of copies
    np::Int
    gcols::Vector{Int}     # input columns of `Gref` that are packed
    bcols::Vector{Int}     # input columns of `Bu` that are packed
    brows::Vector{Int}     # mixed-coordinate rows of `Bu` that are packed
    Kq::Vector{Int}        # regimes per `Qc` copy
    qoff::Vector{Int}      # 0-based offset of each `Qc` copy within its block
end

# Block ordinals, in layout order.
const _LQR_BLOCK_A = 1
const _LQR_BLOCK_S = 2
const _LQR_BLOCK_Q = 3
const _LQR_BLOCK_H = 4
const _LQR_BLOCK_B = 5
const _LQR_BLOCK_G = 6
const _LQR_BLOCK_F = 7
const _LQR_BLOCK_N = 7

"""
    _lqr_blk(p, b, v) -> UnitRange

Slots holding copy `v` of block `b`; empty when the block is frozen.
"""
@inline function _lqr_blk(p::_LQRPack, b::Int, v::Int)
    w = p.w[b]
    w == 0 && return 1:0
    off = p.base[b] + (v - 1) * w
    return (off + 1):(off + w)
end

"""
    _lqr_blk_q(p, v, k) -> UnitRange

Slots holding regime `k` of copy `v` of the cost block.
"""
@inline function _lqr_blk_q(p::_LQRPack, v::Int, k::Int)
    (p.w[_LQR_BLOCK_Q] == 0 || k > p.Kq[v]) && return 1:0
    nc = p.n * (p.n + 1) ÷ 2
    off = p.base[_LQR_BLOCK_Q] + p.qoff[v] + (k - 1) * nc
    return (off + 1):(off + nc)
end

function _LQRPack(sm::LQRStateModel)
    return _LQRPack(sm, sm.fit_flags, ntuple(_ -> 1, _LQR_BLOCK_N))
end

"""Whether any of `sms` is a `:causal` model planning against a terminal cost."""
function _any_causal_terminal_cost(sms)::Bool
    for sm in sms
        _is_causal(sm) && sm.causal.terminal_cost && return true
    end
    return false
end

function _LQRPack(sm::LQRStateModel, f::LQRFitFlags)
    return _LQRPack(sm, f, ntuple(_ -> 1, _LQR_BLOCK_N))
end

#=
Taking the flags explicitly rather than off the model is what lets a caller
freeze part of the block for one pass; taking `nv` is what lets each block carry
its own number of copies, and `Kq` each cost copy its own number of regimes
(every copy has the model's own count by default). `terminal` says whether any
model in the problem carries a terminal factor, whose offset is then packed.
=#
function _LQRPack(
    sm::LQRStateModel,
    f::LQRFitFlags,
    nv::NTuple{7,Int},
    Kq::AbstractVector{Int}=fill(_nregimes(sm), nv[_LQR_BLOCK_Q]),
    terminal::Bool=sm.terminal,
)
    n = _plant_dim(sm)
    d = 2n
    m = size(sm.Bu, 2)
    length(Kq) == nv[_LQR_BLOCK_Q] ||
        throw(DimensionMismatchError("regimes per Qc copy", nv[_LQR_BLOCK_Q], length(Kq)))
    K = maximum(Kq; init=_nregimes(sm))
    #= `Gref` is the one block that can be free in part: `Gref_cols` names the
    input columns the reference is a function of, and the rest are packed no
    more than a frozen block is. =#
    gcols = _gref_cols(f, m)
    bcols = _bu_cols(f, m)
    brows = _bu_rows(f, d)
    nc = n * (n + 1) ÷ 2
    w = (
        f.A ? n * n : 0,
        f.S ? nc : 0,
        f.Qc ? K * nc : 0,
        f.h ? d : 0,
        length(brows) * length(bcols),
        n * length(gcols),
        (terminal && f.terminal) ? n : 0,
    )
    qoff = zeros(Int, length(Kq))
    acc = 0
    for v in eachindex(Kq)
        qoff[v] = acc
        acc += f.Qc ? Kq[v] * nc : 0
    end
    bases = zeros(Int, _LQR_BLOCK_N)
    pos = 0
    for b in 1:_LQR_BLOCK_N
        bases[b] = pos
        pos += b == _LQR_BLOCK_Q ? acc : nv[b] * w[b]
    end
    return _LQRPack(
        n,
        d,
        m,
        K,
        nv,
        w,
        ntuple(b -> bases[b], _LQR_BLOCK_N),
        pos,
        gcols,
        bcols,
        brows,
        collect(Int, Kq),
        qoff,
    )
end

"""
    _LQRUnit{T,HS}

One pooled group of trials in the structural M-step: its aggregated statistics,
which copy of each structural block it uses (`v`, indexed by the `_LQR_BLOCK_*`
ordinals), which noise version (`q`), and whether its model is a `:hold`
regulator (`hold`) rather than the finite-horizon form.

Cells agreeing on every version — and on the mode — are pooled into one unit
before the objective is built. That is exact, not an approximation: with the
same `Θ` the residual scatter `R = Σ_c (Y_c − Θ X_cᵀ − X_c Θᵀ + Θ Z_c Θᵀ)` is
linear in the statistics, so summing them first gives the same `R`. An ungrouped
fit is one unit, which is why there is a single code path. Two modes never pool:
their statistics are in different coordinates (mixed for `:lqr`, forward for
`:hold`) and their residuals are different functions of the parameters.
"""
struct _LQRUnit{T<:Real,HS}
    hs::HS
    v::NTuple{7,Int}
    q::Int
    n::T
    hold::Bool
    causal::Bool
end

"""
    _LQRMStepCtx{T,HS,SM}

Everything the structural objective needs, assembled once per M-step.

Each structural block carries its own number of copies, and each unit names the
copy of each block it uses. That covers all three callers with one layout: an
ungrouped fit is one copy of everything; `depends_on` and a whole-block tie give
every block the same number of copies; a partial tie gives the shared blocks one
copy and the rest one per discrete state — solved **jointly**, in one L-BFGS run,
rather than by alternating over the two subsets.

`profile` says whether `Σ` is being profiled out — it is when the enclosing
`fit_bool` lets the noise move, and then the objective is `(N_s/2) log det R_s`
per noise version; otherwise `Σ` is held fixed and the objective is
`½ tr(Σ_s⁻¹R_s)`. Both carry the same `−N log|det A|` Jacobian term and share one
gradient expression, differing only in the weight applied to the residual
(`N_s R_s⁻¹` versus `Σ_s⁻¹`).

Note where the coupling lives: units sharing a noise version pool into one `R_s`,
so a model whose structure varies by group but whose noise does not is *not*
separable across groups — which is exactly why the objective is built jointly.

`:hold` units sit in the same context. Their residual is
`Lh z_{t+1} − Θh ω_t` in plant-row / manifold-row coordinates, with `Lh` and
`Θh` functions of the DARE solution `P` at the unit's `(A, S, Q_h)` copies; it
pools into the unit's noise version exactly like an `:lqr` residual, and the
unit carries `−N log det(I + S P)` where an `:lqr` unit's `A` carries
`−N log|det A|`. So a plant tied across a control state and a hold state is one
joint solve, with each state's cost its own copy. `hold[ui]` holds a hold unit's
steady state and gradient scratch (an empty placeholder for an `:lqr` unit), and
`lqrA[a]` whether any `:lqr` unit uses `A` copy `a` — only those carry the
`log|det A|` term, and only those need `A` invertible.

`Theta` and `Psi` are per **unit** rather than per version, since a unit's
assembled block mixes copies that no longer move together. The scratch below them
is preallocated once and reused across objective evaluations: at a large plant
dimension L-BFGS makes many evaluations and the `d × d` temporaries dominate.
"""
struct _LQRMStepCtx{T<:Real,HS,SM}
    pack::_LQRPack
    nq::Int
    terminal::Bool                   # any model carries a terminal factor
    units::Vector{_LQRUnit{T,HS}}
    hold::Vector{_HoldUnit{T}}       # [unit]; a 0-dimensional placeholder for `:lqr`
    causal::Vector{_CausalScratch{T}}  # [unit]; empty unless the unit is `:causal`
    causal_q::BitVector              # noise versions owned by `:causal` units
    lqrA::BitVector                  # `A` copies used by some `:lqr` unit
    sms::Vector{SM}
    owners::Vector{Vector{Vector{Int}}}  # [block][copy] -> models using it
    q_of::Vector{Int}                    # model -> noise version
    profile::Bool
    N_A::Vector{T}                   # transitions per `A` copy (Jacobian weight)
    N_q::Vector{T}                   # transitions per noise version
    Nf_q::Vector{T}                  # terminal factors per noise version
    active_q::BitVector              # noise versions with a meaningful effective count
    Sinv::Vector{Matrix{T}}
    Sfinv::Vector{Matrix{T}}
    # unpacked parameters, per block copy
    A::Vector{Matrix{T}}
    S::Vector{Matrix{T}}
    Qc::Vector{Vector{Matrix{T}}}
    h::Vector{Vector{T}}
    Bu::Vector{Matrix{T}}
    Gref::Vector{Matrix{T}}
    hf::Vector{Vector{T}}
    Theta::Vector{Vector{Matrix{T}}}  # [unit][regime]
    Psi::Vector{Vector{Matrix{T}}}    # [unit][terminal regime]
    R::Vector{Matrix{T}}              # [noise version], pooled
    Rf::Vector{Matrix{T}}             # [noise version], pooled
    # preallocated evaluation scratch
    dA::Vector{Matrix{T}}
    dS::Vector{Matrix{T}}
    dQ::Vector{Vector{Matrix{T}}}
    dh::Vector{Vector{T}}
    dB::Vector{Matrix{T}}
    dG::Vector{Matrix{T}}
    dhf::Vector{Vector{T}}
    W::Vector{Matrix{T}}
    Wf::Vector{Matrix{T}}
    tmp_dd::Matrix{T}                 # d × d
    tmp_dr::Matrix{T}                 # d × reg
    tmp_dr2::Matrix{T}                # d × reg
    tmp_nr::Matrix{T}                 # n × reg
    tmp_nr2::Matrix{T}                # n × reg
    tmp_nn::Matrix{T}                 # n × n
    #=
    Effective transition counts per noise version. Without a `Σ_prior` these are
    `N_q`; with one they are `ν + N + d + 1`, the posterior's count, and the
    objective, its gradient weight and the noise M-step all read them from here
    so the three cannot disagree.
    =#
    tmp_neff::Vector{T}               # [noise version]
end

"""
    _lqr_units(sufs, slots, q_slots[, hold]) -> Vector{_LQRUnit}

Pool the cells into units. `slots[b][c]` is the copy of block `b` that cell `c`
uses; cells agreeing on every block, on the noise version and on the mode
(`hold[c]`) become one unit.
"""
function _lqr_units(
    sufs::AbstractVector,
    slots::NTuple{7,Vector{Int}},
    q_slots::AbstractVector{Int},
    hold::AbstractVector{Bool}=falses(length(sufs)),
    causal::AbstractVector{Bool}=falses(length(sufs)),
)
    T = eltype(first(sufs).nk)
    keys = NTuple{10,Int}[]
    members = Vector{Int}[]
    for c in eachindex(sufs)
        key = (
            ntuple(b -> slots[b][c], _LQR_BLOCK_N)...,
            q_slots[c],
            Int(hold[c]),
            Int(causal[c]),
        )
        idx = findfirst(isequal(key), keys)
        if idx === nothing
            push!(keys, key)
            push!(members, [c])
        else
            push!(members[idx], c)
        end
    end
    units = _LQRUnit{T,eltype(sufs)}[]
    for (i, key) in enumerate(keys)
        hs = if length(members[i]) == 1
            sufs[members[i][1]]
        else
            _pool_lqr_stats(sufs, members[i])
        end
        push!(
            units,
            _LQRUnit{T,eltype(sufs)}(
                hs,
                ntuple(b -> key[b], _LQR_BLOCK_N),
                key[8],
                sum(hs.nk),
                key[9] == 1,
                key[10] == 1,
            ),
        )
    end
    return units
end

function _pool_lqr_stats(sufs::AbstractVector, idx::AbstractVector{Int})
    base = sufs[idx[1]]
    out = deepcopy(base)
    for c in idx[2:end]
        s = sufs[c]
        for k in eachindex(out.Zw)
            out.Zw[k] .+= s.Zw[k]
            out.Xv[k] .+= s.Xv[k]
            out.nk[k] += s.nk[k]
        end
        out.Yv .+= s.Yv
        for k in eachindex(out.Omega)
            out.Omega[k] .+= s.Omega[k]
            out.term_n[k] += s.term_n[k]
        end
        for e in eachindex(out.entry_ww)
            out.entry_ww[e] .+= s.entry_ww[e]
            out.entry_n[e] += s.entry_n[e]
        end
        _pool_causal_stats!(out, s)
    end
    return out
end

function _LQRMStepCtx(
    sufs::AbstractVector,
    sms::AbstractVector,
    ab_slots::AbstractVector{Int},
    q_slots::AbstractVector{Int},
    profile::Bool;
    flags::Union{Nothing,LQRFitFlags}=nothing,
)
    # Every block moves together: the whole-block case.
    return _LQRMStepCtx(
        sufs,
        sms,
        ntuple(_ -> collect(ab_slots), _LQR_BLOCK_N),
        q_slots,
        profile;
        flags=flags,
    )
end

"""
    _lqr_block_array(sm, b) -> AbstractArray

The array holding structural block `b` of `sm`, by `_LQR_BLOCK_*` ordinal. Its
*identity* is what says whether two cells share the block, which is how
[`_lqr_cell_slots`](@ref) recovers the grouping.
"""
@inline function _lqr_block_array(sm::LQRStateModel, b::Int)
    b === _LQR_BLOCK_A && return _is_free(sm) ? sm.Mfree : sm.A
    b === _LQR_BLOCK_S && return sm.S
    #= The `Qc` *vector* is rebuilt per variant even when its matrices are
    shared, so the matrix is what carries the sharing. =#
    b === _LQR_BLOCK_Q && return isempty(sm.Qc) ? sm.Mfree : first(sm.Qc)
    b === _LQR_BLOCK_H && return sm.h
    b === _LQR_BLOCK_B && return sm.Bu
    b === _LQR_BLOCK_G && return sm.Gref
    return sm.hf
end

"""
    _lqr_shares_block(sms, b) -> Bool

Whether every model in `sms` holds the *same array* for structural block `b`.
"""
function _lqr_shares_block(sms::AbstractVector, b::Int)
    arr = _lqr_block_array(first(sms), b)
    return all(sm -> _lqr_block_array(sm, b) === arr, sms)
end

"""
    _lqr_cell_slots(sms, cell_slots) -> NTuple{7,Vector{Int}}

Per-block copy indices for a `depends_on` fit: `cell_slots` — the structural
group's slot for each cell — for the pieces that actually vary across cells, and
a single shared copy for the rest.

This is what makes `(Qc = labels,)` a different model from `(structure =
labels,)` while resolving to the same cells: both split the trials the same way,
and only the number of copies of each block differs.

Which is which is read off the cells themselves rather than off the
declaration. `_build_variants!` shares a piece across cells **by reference**
exactly when the declaration left it out, so the arrays a cell holds *are* the
declaration — and the cell models, unlike the model the user declared it on,
are what this M-step is handed.
"""
function _lqr_cell_slots(sms::AbstractVector, cell_slots::AbstractVector{Int})
    shared = ones(Int, length(cell_slots))
    return ntuple(
        b -> _lqr_shares_block(sms, b) ? shared : collect(cell_slots), _LQR_BLOCK_N
    )
end

"""
    _LQRMStepCtx(sufs, sms, slots, q_slots, profile; flags)

`slots[b][c]` is the copy of block `b` that cell `c` uses. Equal across blocks is
the whole-block case; differing per block is a partial tie, and the two are the
same optimization here.
"""
function _LQRMStepCtx(
    sufs::AbstractVector,
    sms::AbstractVector,
    slots::NTuple{7,Vector{Int}},
    q_slots::AbstractVector{Int},
    profile::Bool;
    flags::Union{Nothing,LQRFitFlags}=nothing,
)
    # A pinned costate block has no unconstrained full-matrix profile. Optimize
    # structure at the current covariance, then update the free state block.
    #= A `:causal` model profiles its plant block and holds a pinned costate
    block in the same objective, so it keeps the profile either way. =#
    profile =
        profile && all(sm.fixed_costate_sigma === nothing || _is_causal(sm) for sm in sms)
    sm1 = sms[1]
    T = eltype(sm1.Σ)
    f = flags === nothing ? sm1.fit_flags : flags
    #= Checked again here, not only at construction: `Qc` is a mutable vector,
    and an alias introduced afterwards would break writeback just the same. =#
    for sm in sms
        _is_free(sm) || _check_qc_unaliased(sm.Qc)
    end
    #=
    `:hold` and `:lqr` models may share one problem (a switching model's control
    and hold states). The terminal offset is packed when any of them carries a
    terminal factor — only `:lqr` models can — and each cell's mode decides which
    residual its unit contributes.
    =#
    terminal = any(sm -> sm.terminal, sms)
    hold_c = Bool[_is_hold(sm) for sm in sms]
    causal_c = Bool[_is_causal(sm) for sm in sms]
    #= A causal model with a terminal cost reads `h_f` in its sweep's start, so
    the offset is packed for it too, although it carries no terminal factor. =#
    pack_terminal = terminal || _any_causal_terminal_cost(sms)
    if any(causal_c)
        all(causal_c) || throw(
            ArgumentError(
                "`:causal` inverse-LQR states cannot share a structural M-step with " *
                "other modes yet",
            ),
        )
        all(sm -> sm.causal == sm1.causal, sms) || throw(
            ArgumentError(
                "the `:causal` states of one fit must share their causal options"
            ),
        )
    end
    #=
    A frozen block is never shared, whatever the tie asks for: freezing means
    "keep your own value", so each cell reads back its own rather than the first
    one on some version it was grouped into. Under `depends_on` this is a no-op —
    cells on a version alias the same array — and with one cell there is nothing
    to distinguish either way.
    =#
    probe = _LQRPack(sm1, f, ntuple(_ -> 1, _LQR_BLOCK_N), [_nregimes(sm1)], pack_terminal)
    ncell = length(slots[1])
    #=
    Bound to a fresh name rather than back onto `slots`: reassigning an argument
    the closure above also reads boxes it, and the read becomes one JET cannot
    prove defined.
    =#
    eff = ntuple(b -> probe.w[b] == 0 ? collect(1:ncell) : slots[b], _LQR_BLOCK_N)
    units = _lqr_units(sufs, eff, q_slots, hold_c, causal_c)
    nv = ntuple(b -> maximum(eff[b]), _LQR_BLOCK_N)

    #=
    Which models use each copy of each block. Two jobs: a frozen block reads its
    value back from a representative, and the fitted value is written to every
    model sharing the copy — so a tie is broadcast by construction rather than by
    a separate pass that has to know which blocks it may touch.
    =#
    owners = [[Int[] for _ in 1:nv[b]] for b in 1:_LQR_BLOCK_N]
    for c in eachindex(eff[1])
        for b in 1:_LQR_BLOCK_N
            push!(owners[b][eff[b][c]], c)
        end
    end
    #= Each cost copy carries as many regimes as the most any of its owners has:
    a hold state's one cost beside an `:lqr` state's running and terminal ones. =#
    Kq = [
        maximum(c -> _nregimes(sms[c]), owners[_LQR_BLOCK_Q][v]) for v in 1:nv[_LQR_BLOCK_Q]
    ]
    pack = _LQRPack(sm1, f, nv, Kq, pack_terminal)
    n, d, m = pack.n, pack.d, pack.m
    reg = d + 1 + m
    nq = maximum(q_slots)

    #=
    The Jacobian term `−N_a log|det A_a|` is weighted by the transitions the
    models sharing that `A` actually contribute, which under a partial tie is no
    longer the same grouping as any other block. Hold units carry their own
    Jacobian, `−N log det(I + S P)`, and contribute nothing here.
    =#
    N_A = zeros(T, nv[_LQR_BLOCK_A])
    lqrA = falses(nv[_LQR_BLOCK_A])
    N_q = zeros(T, nq)
    Nf_q = zeros(T, nq)
    for u in units
        N_q[u.q] += u.n
        (terminal && !u.hold && !u.causal) && (Nf_q[u.q] += sum(u.hs.term_n))
        (u.hold || u.causal) || (lqrA[u.v[_LQR_BLOCK_A]] = true)
    end
    count_tol = sqrt(eps(T)) * max(maximum(N_q; init=zero(T)), one(T))
    active_q = BitVector(N_q .> count_tol)
    inactive = findall(!, active_q)
    if !isempty(inactive) && any(active_q)
        @warn(
            "ignoring inverse-LQR noise versions with negligible effective " *
                "transition count; their parameters will be left unchanged",
            inactive_versions = inactive,
            effective_counts = N_q[inactive],
            threshold = count_tol,
            maxlog = 3,
        )
    end
    for u in units
        (active_q[u.q] && !u.hold && !u.causal) || continue
        N_A[u.v[_LQR_BLOCK_A]] += u.n
    end

    noise_sm = Vector{typeof(sm1)}(undef, nq)
    for (c, q) in enumerate(q_slots)
        noise_sm[q] = sms[c]
    end
    Sinv = [
        if profile
            zeros(T, d, d)
        else
            Matrix(inv(PDMat(Symmetrize!(Matrix{T}(noise_sm[s].Σ)))))
        end for s in 1:nq
    ]
    Sfinv = [
        if (profile || !terminal)
            zeros(T, n, n)
        else
            Matrix(inv(PDMat(Symmetrize!(Matrix{T}(noise_sm[s].Σf)))))
        end for s in 1:nq
    ]

    U = length(units)
    Ku = [length(u.hs.nk) for u in units]
    return _LQRMStepCtx{T,eltype(sufs),typeof(sm1)}(
        pack,
        nq,
        terminal,
        units,
        [_HoldUnit(T, u.hold ? n : 0, m) for u in units],
        [
            if u.causal
                _CausalScratch(T, u.hs.causal_keys, n, m, pack.K)
            else
                _CausalScratch(T)
            end for u in units
        ],
        BitVector([any(u -> u.causal && u.q == s, units) for s in 1:nq]),
        lqrA,
        collect(sms),
        owners,
        collect(q_slots),
        profile,
        N_A,
        N_q,
        Nf_q,
        active_q,
        Sinv,
        Sfinv,
        [Matrix{T}(undef, n, n) for _ in 1:nv[_LQR_BLOCK_A]],
        [Matrix{T}(undef, n, n) for _ in 1:nv[_LQR_BLOCK_S]],
        [[Matrix{T}(undef, n, n) for _ in 1:Kq[v]] for v in 1:nv[_LQR_BLOCK_Q]],
        [Vector{T}(undef, d) for _ in 1:nv[_LQR_BLOCK_H]],
        [Matrix{T}(undef, d, m) for _ in 1:nv[_LQR_BLOCK_B]],
        [Matrix{T}(undef, n, m) for _ in 1:nv[_LQR_BLOCK_G]],
        [Vector{T}(undef, n) for _ in 1:nv[_LQR_BLOCK_F]],
        [[zeros(T, d, reg) for _ in 1:Ku[ui]] for ui in 1:U],
        [[zeros(T, n, reg) for _ in 1:Ku[ui]] for ui in 1:U],
        [Matrix{T}(undef, d, d) for _ in 1:nq],
        [Matrix{T}(undef, n, n) for _ in 1:nq],
        [zeros(T, n, n) for _ in 1:nv[_LQR_BLOCK_A]],
        [zeros(T, n, n) for _ in 1:nv[_LQR_BLOCK_S]],
        [[zeros(T, n, n) for _ in 1:Kq[v]] for v in 1:nv[_LQR_BLOCK_Q]],
        [zeros(T, d) for _ in 1:nv[_LQR_BLOCK_H]],
        [zeros(T, d, m) for _ in 1:nv[_LQR_BLOCK_B]],
        [zeros(T, n, m) for _ in 1:nv[_LQR_BLOCK_G]],
        [zeros(T, n) for _ in 1:nv[_LQR_BLOCK_F]],
        [Matrix{T}(undef, d, d) for _ in 1:nq],
        [zeros(T, n, n) for _ in 1:nq],
        Matrix{T}(undef, d, d),
        Matrix{T}(undef, d, reg),
        Matrix{T}(undef, d, reg),
        Matrix{T}(undef, n, reg),
        Matrix{T}(undef, n, reg),
        Matrix{T}(undef, n, n),
        zeros(T, nq),
    )
end

# Ungrouped convenience: one unit, one structural version, one noise version.
function _LQRMStepCtx(
    hs, sm::LQRStateModel, profile::Bool; flags::Union{Nothing,LQRFitFlags}=nothing
)
    return _LQRMStepCtx([hs], [sm], [1], [1], profile; flags=flags)
end

"""
    _lqr_structure_flags(sm, fit_structure) -> LQRFitFlags

The block layout the structural M-step should pack: the model's own `fit_flags`
when the structural group is being fitted, and every block frozen when
`fit_bool[3]` switches the whole group off.

Packing is not free of side conditions — a fitted `S` or `Qc` is packed through
a Cholesky factor and must be positive definite — so a group frozen by the
enclosing switch has to be packed as frozen too. Otherwise a legitimately
singular known block (`S = 0`, a rank-deficient cost) is refused by a fit that
was never going to move it.
"""
function _lqr_structure_flags(sm::LQRStateModel, fit_structure::Bool)
    fit_structure && return sm.fit_flags
    return LQRFitFlags(;
        A=false, S=false, Qc=false, h=false, Bu=false, Gref=false, terminal=false
    )
end

"""
    _lqr_nparams(ctx) -> Int

Length of the flat parameter vector.
"""
@inline _lqr_nparams(ctx::_LQRMStepCtx) = ctx.pack.np

function _lqr_pack_psd!(θ, r, matrix, name)
    isempty(r) && return θ
    # A frozen cost may be exactly singular, but a fitted one cannot be packed
    # into an interior Cholesky parameterization without changing the incoming
    # model (and hence the generalized-EM acceptance baseline).
    F = cholesky(Symmetric(Matrix(matrix)); check=false)
    issuccess(F) || throw(
        ArgumentError(
            "$name must be positive definite when it is fitted. A singular " *
            "`L*L'` initialization is rank-locked and cannot discover missing " *
            "cost directions; add a small diagonal initialization or freeze this block.",
        ),
    )
    L = F.L
    q = first(r)
    for j in axes(L, 2), i in j:size(L, 1)
        θ[q] = i == j ? log(L[i, j]) : L[i, j]
        q += 1
    end
    return θ
end

@inline function _lqr_unpack_psd!(matrix, factor, θ, r)
    fill!(factor, zero(eltype(factor)))
    q = first(r)
    for j in axes(factor, 2), i in j:size(factor, 1)
        factor[i, j] = i == j ? exp(θ[q]) : θ[q]
        q += 1
    end
    mul!(matrix, factor, transpose(factor))
    return matrix
end

@inline function _lqr_pack_psd_gradient!(grad, r, dL, L)
    q = first(r)
    for j in axes(L, 2), i in j:size(L, 1)
        grad[q] = i == j ? dL[i, j] * L[i, j] : dL[i, j]
        q += 1
    end
    return grad
end

"""
    _qc_owner(ctx, v, k) -> Int

A model that holds regime `k` of cost copy `v`. Every owner of the copy holds
regime 1; a higher regime is held only by owners with that many costs (an
`:lqr` state's terminal cost, beside a `:hold` state's single one).
"""
@inline function _qc_owner(ctx::_LQRMStepCtx, v::Int, k::Int)
    for c in ctx.owners[_LQR_BLOCK_Q][v]
        _nregimes(ctx.sms[c]) >= k && return c
    end
    throw(ArgumentError("no model holds regime $k of cost copy $v"))
end

"""
    _lqr_pack!(θ, ctx) -> θ

Read the current parameters into `θ`. A block's copy is read from any model that
uses it — they agree, since a tie is written to all of them.
"""
function _lqr_pack!(θ::AbstractVector{T}, ctx::_LQRMStepCtx{T}) where {T<:Real}
    p = ctx.pack
    o = ctx.owners
    for v in 1:(p.nv[_LQR_BLOCK_A])
        r = _lqr_blk(p, _LQR_BLOCK_A, v)
        isempty(r) || copyto!(view(θ, r), vec(ctx.sms[first(o[_LQR_BLOCK_A][v])].A))
    end
    for v in 1:(p.nv[_LQR_BLOCK_S])
        r = _lqr_blk(p, _LQR_BLOCK_S, v)
        _lqr_pack_psd!(θ, r, ctx.sms[first(o[_LQR_BLOCK_S][v])].S, "S")
    end
    for v in 1:(p.nv[_LQR_BLOCK_Q]), k in 1:(p.Kq[v])
        r = _lqr_blk_q(p, v, k)
        _lqr_pack_psd!(θ, r, ctx.sms[_qc_owner(ctx, v, k)].Qc[k], "Qc[$k]")
    end
    for v in 1:(p.nv[_LQR_BLOCK_H])
        r = _lqr_blk(p, _LQR_BLOCK_H, v)
        isempty(r) || copyto!(view(θ, r), ctx.sms[first(o[_LQR_BLOCK_H][v])].h)
    end
    for v in 1:(p.nv[_LQR_BLOCK_B])
        r = _lqr_blk(p, _LQR_BLOCK_B, v)
        isempty(r) || copyto!(
            view(θ, r),
            vec(view(ctx.sms[first(o[_LQR_BLOCK_B][v])].Bu, p.brows, p.bcols)),
        )
    end
    for v in 1:(p.nv[_LQR_BLOCK_G])
        r = _lqr_blk(p, _LQR_BLOCK_G, v)
        isempty(r) || copyto!(
            view(θ, r), vec(view(ctx.sms[first(o[_LQR_BLOCK_G][v])].Gref, :, p.gcols))
        )
    end
    for v in 1:(p.nv[_LQR_BLOCK_F])
        r = _lqr_blk(p, _LQR_BLOCK_F, v)
        isempty(r) || copyto!(view(θ, r), ctx.sms[first(o[_LQR_BLOCK_F][v])].hf)
    end
    return θ
end

#=============================================================================
Priors on the innovation and the cost.

Both are inverse-Wishart, and both act where the parameter is actually
determined rather than as a post-hoc shrinkage.

`Σ_prior` enters through the *profiled* objective. Profiling substitutes the
noise's own maximizer into the structural objective, so a prior on `Σ` changes
what that maximizer is: the ML value `R/N` becomes the MAP value
`(Ψ + R)/(ν + N + d + 1)`, and the term the structural parameters see becomes
`½(ν + N + d + 1) log det(Ψ + R)` in place of `½N log det R`. The residual weight
in the gradient follows the same substitution. Changing only the noise update and
leaving the profiled objective alone would make the two inconsistent and the
bound non-monotone, which is why it is done here and not in `_lqr_noise_mstep!`
alone.

`Qc_prior` is an additive penalty on the same objective, since the cost is not
profiled out: `−log p(Q_k) = ½[(ν + n + 1) log det Q_k + tr(Ψ Q_k⁻¹)]`, with
gradient `½[(ν + n + 1) Q_k⁻¹ − Q_k⁻¹ Ψ Q_k⁻¹]` accumulated into `dQ` in
matrix coordinates — the PSD chain rule downstream carries it to `θ`.

Counting: the `Σ` prior is applied once per *noise version* and each epoch's
cost prior once per *block copy*, so a tie that shares one array across discrete
states counts its prior once rather than once per state.
=============================================================================#

"""A model on noise version `s` (they agree on everything the noise reads)."""
@inline function _noise_owner(ctx::_LQRMStepCtx, s::Int)
    for (c, sm) in enumerate(ctx.sms)
        ctx.q_of[c] == s && return sm
    end
    throw(ArgumentError("no model on noise version $s"))
end

"""The `Σ` prior in force for noise version `s`, or `nothing`."""
@inline function _sigma_prior(ctx::_LQRMStepCtx, s::Int)
    for (c, sm) in enumerate(ctx.sms)
        ctx.q_of[c] == s && return sm.Σ_prior
    end
    return nothing
end

"""The cost prior in force for regime `k` of copy `v`, or `nothing`."""
@inline function _qc_prior(ctx::_LQRMStepCtx, v::Int, k::Int)
    o = ctx.owners[_LQR_BLOCK_Q][v]
    return isempty(o) ? nothing : _qc_prior(ctx.sms[first(o)], k)
end

"""
    _iw_penalty(Q, prior) -> Float64

`−log p(Q)` for `Q ~ IW(Ψ, ν)`, dropping the normalizer, which is constant in
`Q`. `Inf` if `Q` has left the cone — the optimizer reads that as a rejected step
rather than as a failure.

Split from its gradient on purpose. `_lqr_fg!` returns early when no gradient was
asked for, which is the path a line search and any finite-difference check take,
so a penalty added only in the gradient section would leave the value and the
gradient describing different objectives. That is not a subtle failure in EM —
the accept-if-improved guard keeps the *unpenalized* objective monotone and the
reported bound moves with the prior, so the fit looks healthy while L-BFGS is
following a gradient its objective does not have.
"""
function _iw_penalty(Q::AbstractMatrix{T}, prior) where {T<:Real}
    n = size(Q, 1)
    F = cholesky(Symmetric(Q); check=false)
    issuccess(F) || return T(Inf)
    w = T(prior.ν) + T(n) + one(T)
    return T(0.5) * (w * T(2sum(log, diag(F.U))) + dot(inv(F), T.(prior.Ψ)))
end

"""
    _iw_penalty_grad!(dQ, Q, prior)

Accumulate `∂(−log p(Q))/∂Q = ½[(ν + n + 1) Q⁻¹ − Q⁻¹ Ψ Q⁻¹]` into `dQ`, in
matrix coordinates — the PSD chain rule downstream turns it into a gradient with
respect to the packed factor.
"""
function _iw_penalty_grad!(
    dQ::AbstractMatrix{T}, Q::AbstractMatrix{T}, prior
) where {T<:Real}
    n = size(Q, 1)
    F = cholesky(Symmetric(Q); check=false)
    issuccess(F) || return dQ
    w = T(prior.ν) + T(n) + one(T)
    Qinv = inv(F)
    @. dQ += T(0.5) * w * Qinv
    mul!(dQ, Qinv * T.(prior.Ψ), Qinv, -T(0.5), one(T))
    return dQ
end

"""
    _lqr_unpack!(ctx, θ) -> ctx

Fill the per-copy parameter scratch from `θ`, falling back to the model's own
value for a frozen block.
"""
function _lqr_unpack!(ctx::_LQRMStepCtx{T}, θ::AbstractVector{T}) where {T<:Real}
    p = ctx.pack
    n = p.n
    o = ctx.owners
    for v in 1:(p.nv[_LQR_BLOCK_A])
        r = _lqr_blk(p, _LQR_BLOCK_A, v)
        if isempty(r)
            copyto!(ctx.A[v], ctx.sms[first(o[_LQR_BLOCK_A][v])].A)
        else
            copyto!(ctx.A[v], reshape(view(θ, r), n, n))
        end
    end
    for v in 1:(p.nv[_LQR_BLOCK_S])
        r = _lqr_blk(p, _LQR_BLOCK_S, v)
        if isempty(r)
            copyto!(ctx.S[v], ctx.sms[first(o[_LQR_BLOCK_S][v])].S)
        else
            factor = view(ctx.tmp_dd, 1:n, 1:n)
            _lqr_unpack_psd!(ctx.S[v], factor, θ, r)
        end
    end
    for v in 1:(p.nv[_LQR_BLOCK_Q]), k in 1:(p.Kq[v])
        r = _lqr_blk_q(p, v, k)
        if isempty(r)
            copyto!(ctx.Qc[v][k], ctx.sms[_qc_owner(ctx, v, k)].Qc[k])
        else
            factor = view(ctx.tmp_dd, 1:n, 1:n)
            _lqr_unpack_psd!(ctx.Qc[v][k], factor, θ, r)
        end
    end
    for v in 1:(p.nv[_LQR_BLOCK_H])
        r = _lqr_blk(p, _LQR_BLOCK_H, v)
        if isempty(r)
            copyto!(ctx.h[v], ctx.sms[first(o[_LQR_BLOCK_H][v])].h)
        else
            copyto!(ctx.h[v], view(θ, r))
        end
    end
    if p.m > 0
        for v in 1:(p.nv[_LQR_BLOCK_B])
            r = _lqr_blk(p, _LQR_BLOCK_B, v)
            copyto!(ctx.Bu[v], ctx.sms[first(o[_LQR_BLOCK_B][v])].Bu)
            isempty(r) || copyto!(
                view(ctx.Bu[v], p.brows, p.bcols),
                reshape(view(θ, r), length(p.brows), length(p.bcols)),
            )
        end
        for v in 1:(p.nv[_LQR_BLOCK_G])
            r = _lqr_blk(p, _LQR_BLOCK_G, v)
            #= The model's own matrix first, then the free columns over the top:
            a column `Gref_cols` leaves out keeps the value it was built with,
            exactly as a frozen block does. =#
            copyto!(ctx.Gref[v], ctx.sms[first(o[_LQR_BLOCK_G][v])].Gref)
            isempty(r) || copyto!(
                view(ctx.Gref[v], :, p.gcols), reshape(view(θ, r), n, length(p.gcols))
            )
        end
    end
    for v in 1:(p.nv[_LQR_BLOCK_F])
        r = _lqr_blk(p, _LQR_BLOCK_F, v)
        if isempty(r)
            copyto!(ctx.hf[v], ctx.sms[first(o[_LQR_BLOCK_F][v])].hf)
        else
            copyto!(ctx.hf[v], view(θ, r))
        end
    end
    return ctx
end

"""
    _lqr_assemble!(ctx) -> Bool

Build every unit's design from the unpacked parameters: `Θ_k` (and the terminal
`Ψ_k`) for an `:lqr` unit, and for a `:hold` unit its steady state — the DARE
solution and everything derived from it, see [`_hold_steady_state!`](@ref) —
with `Lh` and `Θh = [A 0 F; 0 0 G]`.

Returns `false` when some active hold unit has no stabilizing DARE solution at
these parameters, which the objective reports as an infeasible point.
"""
function _lqr_assemble!(ctx::_LQRMStepCtx{T}) where {T<:Real}
    p = ctx.pack
    n, d, m = p.n, p.d, p.m
    xr, lr = 1:n, (n + 1):d
    ur = (d + 1):(d + 1 + m)
    terminal = ctx.terminal
    gate = ctx.sms[1].gref_gate
    for (ui, u) in enumerate(ctx.units)
        A = ctx.A[u.v[_LQR_BLOCK_A]]
        S = ctx.S[u.v[_LQR_BLOCK_S]]
        Qs = ctx.Qc[u.v[_LQR_BLOCK_Q]]
        hv = ctx.h[u.v[_LQR_BLOCK_H]]
        if u.causal
            ctx.active_q[u.q] || continue
            _causal_sweep_scratch!(
                ctx.causal[ui],
                A,
                S,
                Qs,
                hv,
                ctx.Bu[u.v[_LQR_BLOCK_B]],
                ctx.Gref[u.v[_LQR_BLOCK_G]],
                ctx.hf[u.v[_LQR_BLOCK_F]],
                ctx.sms[1],
            ) || return false
            continue
        end
        if u.hold
            ctx.active_q[u.q] || continue
            H = ctx.hold[ui]
            _hold_steady_state!(
                H, A, S, Qs[1], hv, ctx.Bu[u.v[_LQR_BLOCK_B]], ctx.Gref[u.v[_LQR_BLOCK_G]]
            ) || return false
            Th = ctx.Theta[ui][1]
            fill!(Th, zero(T))
            @views begin
                Th[xr, xr] .= A
                Th[xr, ur] .= H.F
                Th[lr, ur] .= H.Gm
            end
            continue
        end
        for k in eachindex(ctx.Theta[ui])
            Th = ctx.Theta[ui][k]
            @views begin
                Th[xr, xr] .= A
                Th[xr, lr] .= .-S
                Th[lr, xr] .= Qs[k]
                Th[lr, lr] .= transpose(A)
                Th[:, d + 1] .= hv
                if m > 0
                    Bcol = Th[:, (d + 2):(d + 1 + m)]
                    Bcol .= ctx.Bu[u.v[_LQR_BLOCK_B]]
                    mul!(
                        Bcol[lr, :],
                        Qs[k],
                        _gated(ctx.Gref[u.v[_LQR_BLOCK_G]], gate, k),
                        -one(T),
                        one(T),
                    )
                end
            end
        end
        if terminal
            for k in eachindex(ctx.Psi[ui])
                Psi = ctx.Psi[ui][k]
                Qf = Qs[k]
                @views begin
                    Psi[:, xr] .= .-Qf
                    fill!(Psi[:, lr], zero(T))
                    for i in 1:n
                        Psi[i, n + i] = one(T)
                    end
                    Psi[:, d + 1] .= .-ctx.hf[u.v[_LQR_BLOCK_F]]
                    # Terminal reference: the residual carries +Q_f G_r u_T.
                    m > 0 && mul!(
                        Psi[:, (d + 2):(d + 1 + m)],
                        Qf,
                        _gated(ctx.Gref[u.v[_LQR_BLOCK_G]], gate, k),
                    )
                end
            end
        end
    end
    return true
end

#=
The residual scatter, and the hot loop of the whole M-step: L-BFGS calls it once
per objective evaluation, and its cost is `O(d² · reg)` per unit and regime —
independent of how many trials went into the statistics, which is why a large
plant dimension is what this has to be fast for.

Everything is written through preallocated scratch and 5-argument `mul!`. The
obvious spelling, `R .-= Th * Xvᵀ` and friends, allocates three `d × d` or
`d × reg` temporaries per unit and regime on *every* evaluation.

A hold unit's residual is `Lh z_{t+1} − Θh ω_t`, so its scatter is
`Lh Y Lhᵀ − Lh X Θhᵀ − Θh Xᵀ Lhᵀ + Θh Z Θhᵀ` over the forward statistics
(`Yv`, `Xv`, `Zw` hold `yy`, `zyᵀ`, `zz` for it; see `_fill_mixed_blocks!`).
=#
function _lqr_residuals!(ctx::_LQRMStepCtx{T}) where {T<:Real}
    terminal = ctx.terminal
    for s in 1:(ctx.nq)
        fill!(ctx.R[s], zero(T))
        fill!(ctx.Rf[s], zero(T))
    end
    TX = ctx.tmp_dd
    TZ = ctx.tmp_dr
    PO = ctx.tmp_nr
    for (ui, u) in enumerate(ctx.units)
        ctx.active_q[u.q] || continue
        hs = u.hs
        R = ctx.R[u.q]
        if u.causal
            _causal_add_scatter!(
                R,
                ctx.causal[ui],
                hs,
                ctx.A[u.v[_LQR_BLOCK_A]],
                ctx.S[u.v[_LQR_BLOCK_S]],
                ctx.sms[1].causal.slack_drives_state,
            )
            continue
        end
        if u.hold
            H = ctx.hold[ui]
            Th = ctx.Theta[ui][1]
            # R += Lh Y Lhᵀ
            mul!(TX, H.Lh, hs.Yv)
            mul!(R, TX, transpose(H.Lh), one(T), one(T))
            # R -= (Lh X) Θhᵀ + its transpose
            mul!(TZ, H.Lh, hs.Xv[1])
            mul!(TX, TZ, transpose(Th))
            R .-= TX
            R .-= transpose(TX)
            # R += Θh Z Θhᵀ
            mul!(TZ, Th, hs.Zw[1])
            mul!(R, TZ, transpose(Th), one(T), one(T))
            continue
        end
        R .+= hs.Yv
        for k in eachindex(ctx.Theta[ui])
            Th = ctx.Theta[ui][k]
            # R -= Θ Xᵀ + (Θ Xᵀ)ᵀ, then R += Θ Z Θᵀ.
            mul!(TX, Th, transpose(hs.Xv[k]))
            R .-= TX
            R .-= transpose(TX)
            mul!(TZ, Th, hs.Zw[k])
            mul!(R, TZ, transpose(Th), one(T), one(T))
        end
        if terminal
            for k in eachindex(ctx.Psi[ui])
                hs.term_n[k] > zero(T) || continue
                Psi = ctx.Psi[ui][k]
                mul!(PO, Psi, hs.Omega[k])
                mul!(ctx.Rf[u.q], PO, transpose(Psi), one(T), one(T))
            end
        end
    end
    for s in 1:(ctx.nq)
        Symmetrize!(ctx.R[s])
        terminal && Symmetrize!(ctx.Rf[s])
    end
    return ctx
end

#=============================================================================
The hold unit's gradient.

A hold unit contributes `f = (N_eff/2) log det R` (profiled; `½ tr(Σ⁻¹R)`
otherwise) with `R ∋ Σ_t (Lh z_{t+1} − Θh ω_t)(…)ᵀ`, plus `−N log det(I + S P)`.
With `𝒲 = N_eff R⁻¹` (or `Σ⁻¹`), the two designs pull back as

    ∂f/∂Lh = 𝒲 (Lh Y − Θh Xᵀ),        ∂f/∂Θh = 𝒲 (Θh Z − Lh X),

(`X = Xv`, `Y = Yv`, `Z = Zw`), and everything else is the chain through the
steady state, in reverse order of how `_hold_steady_state!` built it:

    Lh = [I S; −P I]            S̄ += L̄h[x, λ],          P̄ −= L̄h[λ, x]
    Θh = [A 0 F; 0 0 G]         Ā += Θ̄h[x, x],  F̄ = Θ̄h[x, ũ],  Ḡ = Θ̄h[λ, ũ]
    G = V K                     K̄ = Vᵀ Ḡ,       V̄ = Ḡ Kᵀ
    K = E + A_clᵀ P F           Ē = K̄,  Ā_cl = P F K̄ᵀ,  P̄ += A_cl K̄ Fᵀ,  F̄ += P A_cl K̄
    V = (I − A_clᵀ)⁻¹           Ā_cl += V V̄ᵀ V             (dV = V dA_clᵀ V)
    E = [h_λ  B_λ − Q G_r]      h̄_λ, B̄_λ += Ē;  Q̄ −= Ē_u G_rᵀ;  Ḡ_r −= Q Ē_u
    F = [h_x  B_x]              h̄_x, B̄_x += F̄
    A_cl = W A                  Ā += Wᵀ Ā_cl,    W̄ = Ā_cl Aᵀ
    W = (I + S P)⁻¹             M̄ = −Wᵀ W̄ Wᵀ:   S̄ += M̄ P,  P̄ += S M̄
    −N log det(I + S P)         S̄ −= N Wᵀ P,     P̄ −= N S Wᵀ

and last, once every route into `P` is accumulated, the DARE itself:
`dP − A_clᵀ dP A_cl = dQ + dAᵀ P A_cl + A_clᵀ P dA − A_clᵀ P dS P A_cl`, whose
adjoint is one Stein solve `Y − A_cl Y A_clᵀ = sym(P̄)`, giving

    Q̄ += Y,     Ā += 2 P A_cl Y,     S̄ −= P A_cl Y A_clᵀ P.

Every line is checked against central differences of the packed objective in
`test/LinearDynamicalSystems/HoldLDS.jl`. The matrix-coordinate gradients land
in the same `dA`/`dS`/`dQ`/`dh`/`dB`/`dG` buffers the `:lqr` units fill, so the
PSD chain rule, the cost prior, freezing and ties apply to both unchanged.
=============================================================================#
function _hold_unit_gradient!(
    ctx::_LQRMStepCtx{T}, ui::Int, u::_LQRUnit, Fq::AbstractVector, Neff::AbstractVector{T}
) where {T<:Real}
    p = ctx.pack
    n, d, m = p.n, p.d, p.m
    xr, lr = 1:n, (n + 1):d
    ur = (d + 1):(d + 1 + m)
    H = ctx.hold[ui]
    hs = u.hs
    Th = ctx.Theta[ui][1]
    vA, vS, vQ = u.v[_LQR_BLOCK_A], u.v[_LQR_BLOCK_S], u.v[_LQR_BLOCK_Q]
    vh, vB, vG = u.v[_LQR_BLOCK_H], u.v[_LQR_BLOCK_B], u.v[_LQR_BLOCK_G]
    A, S, Q = ctx.A[vA], ctx.S[vS], ctx.Qc[vQ][1]
    dA, dS, dQ = ctx.dA[vA], ctx.dS[vS], ctx.dQ[vQ][1]
    dh = ctx.dh[vh]
    P, W, Acl, V = H.P, H.W, H.Acl, H.V
    Pbar, Aclbar, Kbar, Fbar = H.Pbar, H.Aclbar, H.Kbar, H.Fbar
    N = u.n

    # Residual weights: the raw pullbacks, then 𝒲 applied on the left.
    mul!(H.dd, H.Lh, hs.Yv)
    mul!(H.dd, Th, transpose(hs.Xv[1]), -one(T), one(T))
    mul!(H.dr, Th, hs.Zw[1])
    mul!(H.dr, H.Lh, hs.Xv[1], -one(T), one(T))
    if ctx.profile
        copyto!(H.DLh, H.dd)
        ldiv!(Fq[u.q], H.DLh)
        H.DLh .*= Neff[u.q]
        copyto!(H.DTh, H.dr)
        ldiv!(Fq[u.q], H.DTh)
        H.DTh .*= Neff[u.q]
    else
        mul!(H.DLh, ctx.W[u.q], H.dd)
        mul!(H.DTh, ctx.W[u.q], H.dr)
    end

    @views begin
        # Lh = [I S; −P I]
        dS .+= H.DLh[xr, lr]
        Pbar .= .-H.DLh[lr, xr]
        # Θh = [A 0 F; 0 0 G]
        dA .+= H.DTh[xr, xr]
        Fbar .= H.DTh[xr, ur]
        Gbar = H.DTh[lr, ur]

        # G = V K
        mul!(Kbar, transpose(V), Gbar)

        # E = [h_λ  B_λ − Q G_r]
        dh[lr] .+= Kbar[:, 1]
        if m > 0
            Kbu = Kbar[:, 2:end]
            ctx.dB[vB][lr, :] .+= Kbu
            mul!(dQ, Kbu, transpose(ctx.Gref[vG]), -one(T), one(T))
            mul!(ctx.dG[vG], Q, Kbu, -one(T), one(T))
        end

        # K = E + A_clᵀ P F
        mul!(H.nr, P, H.F)
        mul!(Aclbar, H.nr, transpose(Kbar))
        mul!(H.nr, Acl, Kbar)
        mul!(Pbar, H.nr, transpose(H.F), one(T), one(T))
        mul!(Fbar, P, H.nr, one(T), one(T))

        # V = (I − A_clᵀ)⁻¹, with V̄ᵀ = K Ḡᵀ
        mul!(H.nn1, H.K, transpose(Gbar))
        mul!(H.nn2, V, H.nn1)
        mul!(Aclbar, H.nn2, V, one(T), one(T))

        # F = [h_x  B_x]
        dh[xr] .+= Fbar[:, 1]
        m > 0 && (ctx.dB[vB][xr, :] .+= Fbar[:, 2:end])
    end

    # A_cl = W A
    mul!(dA, transpose(W), Aclbar, one(T), one(T))
    mul!(H.nn1, Aclbar, transpose(A))                  # W̄
    # W = (I + S P)⁻¹:  M̄ = −Wᵀ W̄ Wᵀ (held negated in nn3)
    mul!(H.nn2, transpose(W), H.nn1)
    mul!(H.nn3, H.nn2, transpose(W))
    mul!(dS, H.nn3, P, -one(T), one(T))
    mul!(Pbar, S, H.nn3, -one(T), one(T))

    # Jacobian −N log det(I + S P)
    mul!(dS, transpose(W), P, -N, one(T))
    mul!(Pbar, S, transpose(W), -N, one(T))

    # The DARE: one adjoint Stein solve, then dQ, dA, dS.
    _sym!(H.nn2, Pbar)
    _stein_adjoint!(H.Ybar, Acl, H.nn2, H)
    dQ .+= H.Ybar
    mul!(H.nn1, P, Acl)
    mul!(dA, H.nn1, H.Ybar, T(2), one(T))
    mul!(H.nn2, H.nn1, H.Ybar)
    mul!(dS, H.nn2, transpose(H.nn1), -one(T), one(T))
    return nothing
end

function _lqr_fg!(
    grad::Union{Nothing,AbstractVector{T}}, θ::AbstractVector{T}, ctx::_LQRMStepCtx{T}
) where {T<:Real}
    p = ctx.pack
    n, d, m = p.n, p.d, p.m
    xr, lr = 1:n, (n + 1):d
    terminal = ctx.terminal
    nA = p.nv[_LQR_BLOCK_A]

    #=
    A rejected step must leave `grad` defined, not stale: L-BFGS's line search
    may ask for the gradient at a point whose objective came back infinite, and
    a zero gradient there is the honest "no information" answer.
    =#
    grad === nothing || fill!(grad, zero(T))

    _lqr_unpack!(ctx, θ)
    # `L*L'` is positive definite in exact arithmetic when the log-diagonal is
    # finite, but a line search can push a diagonal far enough toward zero that
    # the materialised Float64 matrix loses rank. Reject that boundary point
    # before it can be accepted and written back; otherwise the next EM
    # iteration cannot pack the now-singular fitted block. Frozen singular
    # blocks remain valid and are deliberately not checked here.
    for v in 1:(p.nv[_LQR_BLOCK_S])
        isempty(_lqr_blk(p, _LQR_BLOCK_S, v)) ||
            isposdef(Symmetric(ctx.S[v])) ||
            return T(Inf)
    end
    for v in 1:(p.nv[_LQR_BLOCK_Q]), k in 1:(p.Kq[v])
        isempty(_lqr_blk_q(p, v, k)) || isposdef(Symmetric(ctx.Qc[v][k])) || return T(Inf)
    end

    fval = zero(T)
    F = Vector{LU{T,Matrix{T},Vector{Int}}}(undef, nA)
    for a in 1:nA
        #= Only the finite-horizon form inverts `A`; a copy only hold units use
        carries no `log|det A|` and may be singular. =#
        ctx.lqrA[a] || continue
        Fa = lu(ctx.A[a]; check=false)
        issuccess(Fa) || return T(Inf)
        logdetA, _ = logabsdet(Fa)
        isfinite(logdetA) || return T(Inf)
        fval -= ctx.N_A[a] * T(logdetA)
        F[a] = Fa
    end

    #= A hold unit whose DARE has no stabilizing solution here has no model at
    all — an infeasible point, like a singular `A`. =#
    _lqr_assemble!(ctx) || return T(Inf)
    _lqr_residuals!(ctx)
    # Hold Jacobian: −N log det(I + S P), the analogue of −N log|det A|.
    for (ui, u) in enumerate(ctx.units)
        (u.hold && ctx.active_q[u.q]) || continue
        fval -= u.n * ctx.hold[ui].logdetM
    end

    Fq = Vector{Cholesky{T,Matrix{T}}}(undef, ctx.nq)
    Ffq = Vector{Cholesky{T,Matrix{T}}}(undef, ctx.nq)
    #=
    Effective counts per noise version. Without a prior these are the transition
    counts; with one they are `ν + N + d + 1` and `ctx.R[s]` has already had the
    prior's scale matrix folded in, so everything downstream — objective,
    gradient weight and the noise M-step — reads the posterior quantities
    through the same two names.
    =#
    Neff = ctx.tmp_neff
    for s in 1:(ctx.nq)
        Neff[s] = ctx.N_q[s]
        ctx.active_q[s] || continue
        ctx.causal_q[s] && continue          # its blocks fold their own priors
        ctx.profile || continue
        pr = _sigma_prior(ctx, s)
        pr === nothing && continue
        ctx.R[s] .+= T.(pr.Ψ)
        Neff[s] = T(pr.ν) + ctx.N_q[s] + T(size(ctx.R[s], 1)) + one(T)
    end
    for s in 1:(ctx.nq)
        ctx.active_q[s] || continue
        if ctx.causal_q[s]
            fs = _causal_noise_objective!(
                ctx.W[s],
                ctx.R[s],
                ctx.N_q[s],
                _noise_owner(ctx, s),
                ctx.profile,
                ctx.Sinv[s],
            )
            isfinite(fs) || return T(Inf)
            fval += fs
            fill!(ctx.Wf[s], zero(T))
            continue
        end
        if ctx.profile
            chol = cholesky(Symmetric(ctx.R[s]); check=false)
            issuccess(chol) || return T(Inf)
            fval += T(0.5) * Neff[s] * logdet(chol)
            Fq[s] = chol
        else
            fval += T(0.5) * dot(ctx.Sinv[s], ctx.R[s])
            copyto!(ctx.W[s], ctx.Sinv[s])
        end
        if terminal && ctx.Nf_q[s] > zero(T)
            if ctx.profile
                cholf = cholesky(Symmetric(ctx.Rf[s]); check=false)
                issuccess(cholf) || return T(Inf)
                fval += T(0.5) * ctx.Nf_q[s] * logdet(cholf)
                Ffq[s] = cholf
            else
                fval += T(0.5) * dot(ctx.Sfinv[s], ctx.Rf[s])
                copyto!(ctx.Wf[s], ctx.Sfinv[s])
            end
        else
            fill!(ctx.Wf[s], zero(T))
        end
    end

    #=
    The cost prior's *value*, here rather than in the gradient section below,
    because of the early return on the next line.
    =#
    for v in 1:(p.nv[_LQR_BLOCK_Q]), k in 1:(p.Kq[v])
        pr = _qc_prior(ctx, v, k)
        if pr !== nothing && !isempty(_lqr_blk_q(p, v, k))
            pen = _iw_penalty(ctx.Qc[v][k], pr)
            isfinite(pen) || return T(Inf)
            fval += pen
        end
    end

    grad === nothing && return fval

    for a in 1:nA
        fill!(ctx.dA[a], zero(T))
    end
    for v in 1:(p.nv[_LQR_BLOCK_S])
        fill!(ctx.dS[v], zero(T))
    end
    for v in 1:(p.nv[_LQR_BLOCK_Q]), k in 1:(p.Kq[v])
        fill!(ctx.dQ[v][k], zero(T))
    end
    for v in 1:(p.nv[_LQR_BLOCK_H])
        fill!(ctx.dh[v], zero(T))
    end
    for v in 1:(p.nv[_LQR_BLOCK_B])
        fill!(ctx.dB[v], zero(T))
    end
    for v in 1:(p.nv[_LQR_BLOCK_G])
        fill!(ctx.dG[v], zero(T))
    end
    for v in 1:(p.nv[_LQR_BLOCK_F])
        fill!(ctx.dhf[v], zero(T))
    end

    E = ctx.tmp_dr           # Θ Z − X
    Gk = ctx.tmp_dr2         # W (Θ Z − X)
    PO = ctx.tmp_nr
    GP = ctx.tmp_nr2
    gate = ctx.sms[1].gref_gate
    for (ui, u) in enumerate(ctx.units)
        ctx.active_q[u.q] || continue
        if u.causal
            vQ, vG = u.v[_LQR_BLOCK_Q], u.v[_LQR_BLOCK_G]
            _causal_unit_gradient!(
                ctx.causal[ui],
                u.hs,
                ctx.W[u.q],
                ctx.A[u.v[_LQR_BLOCK_A]],
                ctx.S[u.v[_LQR_BLOCK_S]],
                ctx.Qc[vQ],
                ctx.Gref[vG],
                ctx.sms[1],
                ctx.dA[u.v[_LQR_BLOCK_A]],
                ctx.dS[u.v[_LQR_BLOCK_S]],
                ctx.dQ[vQ],
                ctx.dh[u.v[_LQR_BLOCK_H]],
                ctx.dB[u.v[_LQR_BLOCK_B]],
                ctx.dG[vG],
                ctx.dhf[u.v[_LQR_BLOCK_F]],
            )
            continue
        end
        if u.hold
            _hold_unit_gradient!(ctx, ui, u, Fq, Neff)
            continue
        end
        vA, vS, vQ = u.v[_LQR_BLOCK_A], u.v[_LQR_BLOCK_S], u.v[_LQR_BLOCK_Q]
        vh, vB, vG = u.v[_LQR_BLOCK_H], u.v[_LQR_BLOCK_B], u.v[_LQR_BLOCK_G]
        for k in eachindex(ctx.Theta[ui])
            copyto!(E, u.hs.Xv[k])
            mul!(E, ctx.Theta[ui][k], u.hs.Zw[k], one(T), -one(T))
            if ctx.profile
                copyto!(Gk, E)
                ldiv!(Fq[u.q], Gk)
                Gk .*= Neff[u.q]
            else
                mul!(Gk, ctx.W[u.q], E)
            end
            @views begin
                ctx.dA[vA] .+= Gk[xr, xr]
                ctx.dA[vA] .+= transpose(Gk[lr, lr])
                ctx.dS[vS] .-= Gk[xr, lr]
                ctx.dQ[vQ][k] .+= Gk[lr, xr]
                ctx.dh[vh] .+= Gk[:, d + 1]
                if m > 0
                    #=
                    The input block is `B_u − [0; Q_k G_r]`, so its gradient
                    splits: `B_u` takes it whole, while `G_r` and `Q_k` share the
                    tracking half bilinearly.
                    =#
                    dBk = Gk[:, (d + 2):(d + 1 + m)]
                    ctx.dB[vB] .+= dBk
                    #= Under a reference gate regime `k` reads `G_r D_k`, so its
                    share of `∂G_r` keeps only the columns `D_k` opens. =#
                    mul!(
                        ctx.dQ[vQ][k],
                        dBk[lr, :],
                        transpose(_gated(ctx.Gref[vG], gate, k)),
                        -one(T),
                        one(T),
                    )
                    mul!(
                        ctx.dG[vG],
                        ctx.Qc[vQ][k],
                        _gated(dBk[lr, :], gate, k),
                        -one(T),
                        one(T),
                    )
                end
            end
        end
        if terminal && ctx.Nf_q[u.q] > zero(T)
            # ∂/∂Ψ_k of each terminal-regime term;
            # Ψ_k = [−Q_k  I  −h_f  Q_k G_r].
            for k in eachindex(ctx.Psi[ui])
                u.hs.term_n[k] > zero(T) || continue
                mul!(PO, ctx.Psi[ui][k], u.hs.Omega[k])
                if ctx.profile
                    copyto!(GP, PO)
                    ldiv!(Ffq[u.q], GP)
                    GP .*= ctx.Nf_q[u.q]
                else
                    mul!(GP, ctx.Wf[u.q], PO)
                end
                @views begin
                    ctx.dQ[vQ][k] .-= GP[:, xr]
                    if m > 0
                        dPu = GP[:, (d + 2):(d + 1 + m)]
                        mul!(
                            ctx.dQ[vQ][k],
                            dPu,
                            transpose(_gated(ctx.Gref[vG], gate, k)),
                            one(T),
                            one(T),
                        )
                        mul!(
                            ctx.dG[vG], ctx.Qc[vQ][k], _gated(dPu, gate, k), one(T), one(T)
                        )
                    end
                    ctx.dhf[u.v[_LQR_BLOCK_F]] .-= GP[:, d + 1]
                end
            end
        end
    end

    for a in 1:nA
        # Jacobian term: ∂(−N_a log|det A_a|)/∂A_a = −N_a A_a⁻ᵀ.
        if ctx.lqrA[a]
            fill!(ctx.tmp_nn, zero(T))
            for i in 1:n
                ctx.tmp_nn[i, i] = one(T)
            end
            ldiv!(adjoint(F[a]), ctx.tmp_nn)
            ctx.dA[a] .-= ctx.N_A[a] .* ctx.tmp_nn
        end
        r = _lqr_blk(p, _LQR_BLOCK_A, a)
        isempty(r) || copyto!(view(grad, r), vec(ctx.dA[a]))
    end
    for v in 1:(p.nv[_LQR_BLOCK_S])
        r = _lqr_blk(p, _LQR_BLOCK_S, v)
        if !isempty(r)
            # For S = L Lᵀ, dF/dL = (dF/dS + (dF/dS)ᵀ) L.
            _sym!(ctx.tmp_nn, ctx.dS[v])
            factor = view(ctx.tmp_dd, 1:n, 1:n)
            _lqr_unpack_psd!(ctx.S[v], factor, θ, r)
            mul!(ctx.dS[v], ctx.tmp_nn, factor, T(2), zero(T))
            _lqr_pack_psd_gradient!(grad, r, ctx.dS[v], factor)
        end
    end
    #=
    The cost prior's gradient, in matrix coordinates, just before the PSD chain
    rule below turns `dQ` into a gradient with respect to the packed factor. Its
    value was added above, before the early return.
    =#
    for v in 1:(p.nv[_LQR_BLOCK_Q]), k in 1:(p.Kq[v])
        pr = _qc_prior(ctx, v, k)
        if pr !== nothing && !isempty(_lqr_blk_q(p, v, k))
            _iw_penalty_grad!(ctx.dQ[v][k], ctx.Qc[v][k], pr)
        end
    end
    for v in 1:(p.nv[_LQR_BLOCK_Q]), k in 1:(p.Kq[v])
        r = _lqr_blk_q(p, v, k)
        if !isempty(r)
            _sym!(ctx.tmp_nn, ctx.dQ[v][k])
            factor = view(ctx.tmp_dd, 1:n, 1:n)
            _lqr_unpack_psd!(ctx.Qc[v][k], factor, θ, r)
            mul!(ctx.dQ[v][k], ctx.tmp_nn, factor, T(2), zero(T))
            _lqr_pack_psd_gradient!(grad, r, ctx.dQ[v][k], factor)
        end
    end
    for v in 1:(p.nv[_LQR_BLOCK_H])
        r = _lqr_blk(p, _LQR_BLOCK_H, v)
        isempty(r) || copyto!(view(grad, r), ctx.dh[v])
    end
    for v in 1:(p.nv[_LQR_BLOCK_B])
        r = _lqr_blk(p, _LQR_BLOCK_B, v)
        isempty(r) || copyto!(view(grad, r), vec(view(ctx.dB[v], p.brows, p.bcols)))
    end
    for v in 1:(p.nv[_LQR_BLOCK_G])
        r = _lqr_blk(p, _LQR_BLOCK_G, v)
        isempty(r) || copyto!(view(grad, r), vec(view(ctx.dG[v], :, p.gcols)))
    end
    for v in 1:(p.nv[_LQR_BLOCK_F])
        r = _lqr_blk(p, _LQR_BLOCK_F, v)
        isempty(r) || copyto!(view(grad, r), ctx.dhf[v])
    end

    return fval
end

"""
    _lqr_writeback!(ctx, θ) -> ctx

Write the fitted parameters back onto every model that uses each block copy.

A tie is therefore broadcast by construction: the models sharing a copy are
exactly the ones this writes it to, so there is no separate pass that has to know
which blocks a partial tie shared.
"""
function _lqr_writeback!(ctx::_LQRMStepCtx{T}, θ::AbstractVector{T}) where {T<:Real}
    _lqr_unpack!(ctx, θ)
    p = ctx.pack
    o = ctx.owners
    for v in 1:(p.nv[_LQR_BLOCK_A]), c in o[_LQR_BLOCK_A][v]
        isempty(_lqr_blk(p, _LQR_BLOCK_A, v)) || copyto!(ctx.sms[c].A, ctx.A[v])
    end
    for v in 1:(p.nv[_LQR_BLOCK_S]), c in o[_LQR_BLOCK_S][v]
        isempty(_lqr_blk(p, _LQR_BLOCK_S, v)) || copyto!(ctx.sms[c].S, ctx.S[v])
    end
    for v in 1:(p.nv[_LQR_BLOCK_Q]), k in 1:(p.Kq[v]), c in o[_LQR_BLOCK_Q][v]
        _nregimes(ctx.sms[c]) >= k || continue
        isempty(_lqr_blk_q(p, v, k)) || copyto!(ctx.sms[c].Qc[k], ctx.Qc[v][k])
    end
    for v in 1:(p.nv[_LQR_BLOCK_H]), c in o[_LQR_BLOCK_H][v]
        isempty(_lqr_blk(p, _LQR_BLOCK_H, v)) || copyto!(ctx.sms[c].h, ctx.h[v])
    end
    for v in 1:(p.nv[_LQR_BLOCK_B]), c in o[_LQR_BLOCK_B][v]
        isempty(_lqr_blk(p, _LQR_BLOCK_B, v)) || copyto!(ctx.sms[c].Bu, ctx.Bu[v])
    end
    for v in 1:(p.nv[_LQR_BLOCK_G]), c in o[_LQR_BLOCK_G][v]
        isempty(_lqr_blk(p, _LQR_BLOCK_G, v)) || copyto!(ctx.sms[c].Gref, ctx.Gref[v])
    end
    for v in 1:(p.nv[_LQR_BLOCK_F]), c in o[_LQR_BLOCK_F][v]
        isempty(_lqr_blk(p, _LQR_BLOCK_F, v)) || copyto!(ctx.sms[c].hf, ctx.hf[v])
    end
    return ctx
end

function _lqr_structure_mstep!(
    ctx::_LQRMStepCtx{T}, fit_structure::Bool, mstep_iters::Int
) where {T<:Real}
    np = _lqr_nparams(ctx)
    θ0 = zeros(T, np)
    _lqr_pack!(θ0, ctx)

    if fit_structure && np > 0
        f0 = _lqr_fg!(nothing, θ0, ctx)
        if isfinite(f0)
            f_obj(θ) = _lqr_fg!(nothing, θ, ctx)
            function g_obj!(G, θ)
                _lqr_fg!(G, θ, ctx)
                return G
            end
            opts = Optim.Options(;
                x_abstol=1e-10, g_abstol=1e-9, f_reltol=1e-12, iterations=mstep_iters
            )
            result = optimize(f_obj, g_obj!, θ0, LBFGS(; linesearch=HagerZhang()), opts)
            θ1 = Optim.minimizer(result)
            f1 = _lqr_fg!(nothing, θ1, ctx)
            #=
            Accept only a genuine improvement. L-BFGS normally returns one, but a
            line-search stall or a step into a region where `R` lost rank would
            otherwise hand EM a worse objective and break monotonicity.
            =#
            if isfinite(f1) && f1 <= f0
                _lqr_writeback!(ctx, θ1)
                copyto!(θ0, θ1)
            else
                @warn(
                    "inverse-LQR structural M-step produced no acceptable proposal; " *
                        "keeping the previous structure",
                    initial_objective = f0,
                    proposed_objective = f1,
                    optimizer_converged = Optim.converged(result),
                    maxlog = 3,
                )
            end
        else
            @warn(
                "inverse-LQR structural M-step has a non-finite initial objective; " *
                    "keeping the previous structure",
                initial_objective = f0,
                active_noise_versions = findall(ctx.active_q),
                maxlog = 3,
            )
        end
    end

    _lqr_unpack!(ctx, θ0)
    #= The accepted point always has a stabilizing DARE solution (its objective
    was finite, or it is the incoming model, which `refresh!` built). =#
    _lqr_assemble!(ctx) || throw(
        NumericalStabilityError(
            "hold",
            "a hold state's DARE has no stabilizing solution at its current parameters",
        ),
    )
    _lqr_residuals!(ctx)
    return ctx
end

"""
    _lqr_noise_mstep!(ctx)

`Σ_s = R_s/N_s` and `Σ_{f,s} = R_{f,s}/N_{f,s}` — the closed-form maximizers
given the structural parameters, which is exactly what profiling them out of the
objective assumed. Units sharing a noise version pool into that version's `R`.

With a `Σ_prior` in force the maximizer becomes the MAP value
`(Ψ + R_s)/(ν + N_s + d + 1)`, which is the same substitution the profiled
objective made — the two have to agree or EM is optimizing one thing and
reporting another. `R_s` arrives here clean: the objective folds `Ψ` into its own
copy per evaluation and `_lqr_structure_mstep!` recomputes the residuals at the
accepted parameters before returning, so the fold is applied once, here.

The terminal factor's covariance takes no prior. It is a pseudo-observation's
noise rather than a process innovation, `Σf → 0` is the hard boundary condition
it approximates, and shrinking it toward anything would work against that.
"""
function _lqr_noise_mstep!(ctx::_LQRMStepCtx{T}) where {T<:Real}
    #=
    Written to every model on the noise version, not just one: as with the
    structural blocks, models sharing a version hold separate arrays in an
    `SLDS` (they alias only under `depends_on`), so the fitted value has to
    reach all of them.
    =#
    for s in 1:(ctx.nq)
        ctx.active_q[s] || continue
        for (c, sm) in enumerate(ctx.sms)
            ctx.q_of[c] == s || continue
            if _is_causal(sm)
                _causal_noise_update!(sm, ctx.R[s], ctx.N_q[s])
                continue
            end
            pr = sm.Σ_prior
            fixed_costate = sm.fixed_costate_sigma
            # `isa`, not `!== nothing`: the field's type is a UnionAll, which only
            # an `isa` test narrows for inference (and JET).
            if fixed_costate isa Real
                n = _plant_dim(sm)
                if ctx.N_q[s] > zero(T)
                    @views sm.Σ[1:n, 1:n] .= ctx.R[s][1:n, 1:n] ./ ctx.N_q[s]
                end
                @views Symmetrize!(view(sm.Σ, 1:n, 1:n))
                @views sm.Σ[1:n, (n + 1):(2n)] .= zero(T)
                @views sm.Σ[(n + 1):(2n), 1:n] .= zero(T)
                costate_block = view(sm.Σ, (n + 1):(2n), (n + 1):(2n))
                fill!(costate_block, zero(T))
                for i in 1:n
                    costate_block[i, i] = fixed_costate
                end
            elseif pr === nothing
                if ctx.N_q[s] > zero(T)
                    copyto!(sm.Σ, ctx.R[s])
                    sm.Σ ./= ctx.N_q[s]
                    Symmetrize!(sm.Σ)
                end
            else
                d = size(sm.Σ, 1)
                copyto!(sm.Σ, ctx.R[s])
                sm.Σ .+= T.(pr.Ψ)
                sm.Σ ./= T(pr.ν) + ctx.N_q[s] + T(d) + one(T)
                Symmetrize!(sm.Σ)
            end
            if sm.terminal && ctx.Nf_q[s] > zero(T)
                copyto!(sm.Σf, ctx.Rf[s])
                sm.Σf ./= ctx.Nf_q[s]
                Symmetrize!(sm.Σf)
            end
        end
    end
    return ctx
end

"""
    _lqr_state_mstep!(lds, hs, sws)

The state half of the M-step for an ungrouped fit, shared by every emission
model.
"""
function _lqr_state_mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    hs::LQRSufficientStatistics{T},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:LQRStateModel{T},O<:AbstractObservationModel{T}}
    if lds.state_model.terminal && lds.state_model.condition_terminal
        _lqr_conditional_mstep!([lds], [hs], [ones(Int, 1) for _ in 1:4])
        return nothing
    end
    base = _state_suf(hs.base)
    update_initial_state_mean!(lds, base)
    update_initial_state_covariance!(lds, base, sws)

    sm = lds.state_model
    if _is_free(sm)
        _free_state_mstep!(lds, hs)
        refresh!(sm)
        return nothing
    end
    _fill_mixed_blocks!(hs, sm)
    if _has_entries(sm)
        #= Entry priors first (an exact update), then the structure, guarded by
        the score that also counts the entries' plant row. =#
        _lqr_entry_update!([sm], [hs])
        _lqr_entry_guarded([lds], [hs]) do
            ctx = _LQRMStepCtx(
                hs, sm, lds.fit_bool[4]; flags=_lqr_structure_flags(sm, lds.fit_bool[3])
            )
            _lqr_structure_mstep!(ctx, lds.fit_bool[3], sm.mstep_iters)
            return lds.fit_bool[4] && _lqr_noise_mstep!(ctx)
        end
        refresh!(sm)
        return nothing
    end
    ctx = _LQRMStepCtx(
        hs, sm, lds.fit_bool[4]; flags=_lqr_structure_flags(sm, lds.fit_bool[3])
    )
    _lqr_structure_mstep!(ctx, lds.fit_bool[3], sm.mstep_iters)
    lds.fit_bool[4] && _lqr_noise_mstep!(ctx)
    refresh!(sm)
    return nothing
end

#=============================================================================
`:free` mode: an unconstrained transition, so the M-step is the ordinary
conjugate regression rather than the constrained one.

Writing `Θ = [M | h | B_u]` and `ω_t = [z_t; 1; u_t]`, the transition term is

    Q = −½[ N(d log2π + logdet Σ) + tr(Σ⁻¹ R(Θ)) ],
    R(Θ) = S_vv − Θ S_vwᵀ − S_vw Θᵀ + Θ S_ww Θᵀ,

with `S_ww = Σ E[ω ωᵀ]`, `S_vw = Σ E[z_{t+1} ωᵀ]`, `S_vv = Σ E[z_{t+1} z_{t+1}ᵀ]`
— exactly the statistics the aggregator already stores as `zz`, `zy` and `yy`.
There is no Jacobian term: `G = I` here, so the mixed and forward coordinates
coincide and `logabsdetA` is zero.

Both updates are exact maximizers, so this is a true M-step (not the generalized
one the symplectic parameterization needs), and it reproduces a
`GaussianStateModel` fit to machine precision.
=============================================================================#

"""
    _free_theta(sm) -> Matrix

The current `d × (d+1+m)` regression block `[M | h | B_u]` of a `:free` model.
"""
function _free_theta(sm::LQRStateModel{T}) where {T<:Real}
    d = _state_latent_dim(sm)
    m = size(sm.Bu, 2)
    Th = Matrix{T}(undef, d, d + 1 + m)
    @views begin
        Th[:, 1:d] .= sm.Mfree
        Th[:, d + 1] .= sm.h
        m > 0 && (Th[:, (d + 2):(d + 1 + m)] .= sm.Bu)
    end
    return Th
end

"""
    _free_residual_scatter(Theta, hs) -> Matrix

`R(Θ) = S_vv − Θ S_vwᵀ − S_vw Θᵀ + Θ S_ww Θᵀ`, symmetrized.
"""
function _free_residual_scatter(
    Theta::AbstractMatrix{T}, hs::LQRSufficientStatistics{T}
) where {T<:Real}
    Sww = hs.zz[1]
    Svw = transpose(hs.zy[1])
    R = Matrix{T}(hs.yy[1])
    mul!(R, Theta, transpose(Svw), -one(T), one(T))
    mul!(R, Svw, transpose(Theta), -one(T), one(T))
    mul!(R, Theta * Sww, transpose(Theta), one(T), one(T))
    # `Symmetrize!` returns a `Symmetric` view; hand back the plain matrix so
    # callers can keep scaling it in place.
    Symmetrize!(R)
    return R
end

"""
    _free_Q_transition(sm, hs) -> T

The transition half of the state Q-term for a `:free` model, at its current
parameters.
"""
function _free_Q_transition(
    sm::LQRStateModel{T}, hs::LQRSufficientStatistics{T}
) where {T<:Real}
    N = T(hs.nk[1])
    N > zero(T) || return zero(T)
    d = _state_latent_dim(sm)
    Theta = _free_theta(sm)
    R = _free_residual_scatter(Theta, hs)
    Σ_PD = PDMat(Symmetrize!(Matrix{T}(sm.Σ)))
    return T(-0.5) * (N * (T(d) * log(T(2π)) + logdet(Σ_PD)) + tr(Σ_PD \ R))
end

"""
    _forward_Q_transition(sm, hs) -> T

The transition half of the state Q-term computed in forward coordinates from
the cache, `−½[N(d log2π + log det Q^fwd) + tr((Q^fwd)⁻¹ R^z)]` with `R^z` the
residual scatter of `z_{t+1} − M z_t − b − B u_t`. Used for a `:hold` model,
whose cache holds `M = Lh⁻¹[A 0; 0 0]`, `Q^fwd = Lh⁻¹ Σ Lh⁻ᵀ`, … — so this
equals the M-step's `−½[N(d log2π + log det Σ − 2 log det(I + SP)) +
tr(Σ⁻¹ R_h)]` term for term, since `R_h = Lh R^z Lhᵀ`.
"""
function _forward_Q_transition(
    sm::LQRStateModel{T}, hs::LQRSufficientStatistics{T}
) where {T<:Real}
    N = T(hs.nk[1])
    N > zero(T) || return zero(T)
    c = sm.cache
    d = _state_latent_dim(sm)
    m = size(sm.Bu, 2)
    Theta = Matrix{T}(undef, d, d + 1 + m)
    @views begin
        Theta[:, 1:d] .= c.M[1]
        Theta[:, d + 1] .= c.bfwd
        m > 0 && (Theta[:, (d + 2):(d + 1 + m)] .= c.Bfwd[1])
    end
    R = _free_residual_scatter(Theta, hs)
    return T(-0.5) * (N * (T(d) * log(T(2π)) + logdet(c.Qfwd)) + tr(c.Qfwd \ R))
end

"""
    _free_state_mstep!(lds, hs)

Closed-form update of `[M | h | B_u]` and `Σ` for a `:free` state model.

`fit_flags.A` / `.h` / `.Bu` select which *columns* of the regression are free;
a partial selection solves for those columns with the frozen ones held at their
current value, which is the ordinary partial least-squares update
`Θ_free = (S_vw − Θ_fix S_ww[fix, ·])[:, free] · S_ww[free, free]⁻¹`.
"""
function _free_state_mstep!(
    lds::LinearDynamicalSystem{T,S,O}, hs::LQRSufficientStatistics{T}
) where {T<:Real,S<:LQRStateModel{T},O<:AbstractObservationModel{T}}
    _free_state_mstep!([lds], [hs])
    return nothing
end

"""
    _free_state_mstep!(ldss, hss)

The same update over several **units** — the `(discrete state, cell)` pairs a
`depends_on` switching fit aggregates into — whose models may hold the *same*
arrays.

One unit per call is wrong as soon as any array is shared: each call writes
`Mfree` / `h` / `B_u` / `Σ` outright, so a shared array ends the M-step holding
the last unit's estimate rather than the estimate from every trial that
parameter was fitted from. The sufficient statistics are additive, so the fix is
to pool them: units that hold the same array get one solve over their summed
statistics, which is exactly what the LQR states' packed M-step does through its
slot vectors. Which units share what is read off array identity — the same
convention [`_lqr_shares_block`](@ref) uses, and set up by `_build_variants!`,
which shares a piece across cells by reference precisely when the declaration
left it out.

`Σ` is partitioned on its own, since `depends_on` groups the noise separately
from the structure: each version's scatter is summed at each unit's *own* `Θ`.
"""
function _free_state_mstep!(
    ldss::AbstractVector, hss::AbstractVector{<:LQRSufficientStatistics{T}}
) where {T<:Real}
    isempty(ldss) && return nothing
    sms = [lds.state_model for lds in ldss]

    #= `[M | h | B_u]` is one regression, so the three have to split the units
    the same way. Splitting only part of it is a generalized least-squares
    problem this closed form does not solve — the same partial-tie restriction
    the switching M-step states for `[A b B]` — so refuse rather than return a
    number that looks like an answer. =#
    versions = _alias_partition(sms, sm -> sm.Mfree)
    for (name, part) in ((:h, sm -> sm.h), (:Bu, sm -> sm.Bu))
        _alias_partition(sms, part) == versions || throw(
            ArgumentError(
                "a `:free` state's `[M | h | B_u]` is fitted as one regression, so " *
                "`depends_on` must group `:A`, `:h` and `:Bu` together or not at " *
                "all; `:$name` splits its trials differently from `:A`. Name " *
                "`:structure` to group the whole block.",
            ),
        )
    end

    thetas = Vector{Matrix{T}}(undef, length(sms))
    for units in versions
        Theta = _free_theta_pooled(ldss, hss, units)
        for u in units
            thetas[u] = Theta
        end
        #= Once per version, through any member: they alias the same arrays. =#
        _write_free_theta!(sms[first(units)], Theta)
    end

    for units in _alias_partition(sms, sm -> sm.Σ)
        _free_noise_mstep!(ldss, hss, thetas, units)
    end
    return nothing
end

"""
    _alias_partition(models, part) -> Vector{Vector{Int}}

The units of `models` grouped by which array `part` returns, compared by
identity: one entry per distinct array, holding the indices that share it, in
first-appearance order. Two partitions are `==` exactly when they split the
units the same way.
"""
function _alias_partition(models::AbstractVector, part)
    groups = Vector{Vector{Int}}()
    arrays = Any[]
    for (i, model) in enumerate(models)
        arr = part(model)
        slot = findfirst(a -> a === arr, arrays)
        if slot === nothing
            push!(arrays, arr)
            push!(groups, [i])
        else
            push!(groups[slot], i)
        end
    end
    return groups
end

"""
    _free_theta_pooled(ldss, hss, units) -> Matrix

The partial least-squares solve for one version of `[M | h | B_u]`, over the
summed statistics of the units sharing it.
"""
function _free_theta_pooled(
    ldss::AbstractVector,
    hss::AbstractVector{<:LQRSufficientStatistics{T}},
    units::AbstractVector{Int},
) where {T<:Real}
    lds = ldss[first(units)]
    sm = lds.state_model
    d = _state_latent_dim(sm)
    m = size(sm.Bu, 2)
    reg = d + 1 + m

    Sww = zeros(T, reg, reg)
    Svw = zeros(T, d, reg)
    N = zero(T)
    for u in units
        Sww .+= hss[u].zz[1]
        Svw .+= transpose(hss[u].zy[1])
        N += T(hss[u].nk[1])
    end

    ff = sm.fit_flags
    free_cols = Int[]
    ff.A && append!(free_cols, 1:d)
    ff.h && push!(free_cols, d + 1)
    append!(free_cols, (d + 1) .+ _bu_cols(ff, m))

    Theta = _free_theta(sm)
    if lds.fit_bool[3] && !isempty(free_cols) && N > zero(T)
        fixed = setdiff(1:reg, free_cols)
        rhs = Svw[:, free_cols]
        isempty(fixed) || mul!(rhs, Theta[:, fixed], Sww[fixed, free_cols], -one(T), one(T))
        Gm = pd_gram(Matrix{T}(Sww[free_cols, free_cols]); name="free dynamics Gram")
        # Θ_free Gm = rhs  ⇒  Gm Θ_freeᵀ = rhsᵀ, and Gm is symmetric.
        Theta[:, free_cols] .= transpose(Gm.chol \ Matrix{T}(transpose(rhs)))
        # Off unless debug logging is on for the package; the arguments are not
        # evaluated otherwise, so the `cond` and `eigvals` cost nothing.
        @debug "free regression" N = N gram_cond = cond(Sww[free_cols, free_cols]) theta_max = maximum(
            abs, Theta
        ) rho = maximum(abs, eigvals(Theta[:, 1:d]))
    end
    return Theta
end

"""
    _free_noise_mstep!(ldss, hss, thetas, units)

`Σ` for one version, as the residual scatter of every unit sharing it —
each at its own `Θ`, since the structure may be grouped differently — over
their total transition count.
"""
function _free_noise_mstep!(
    ldss::AbstractVector,
    hss::AbstractVector{<:LQRSufficientStatistics{T}},
    thetas::AbstractVector{Matrix{T}},
    units::AbstractVector{Int},
) where {T<:Real}
    lds = ldss[first(units)]
    lds.fit_bool[4] || return nothing
    sm = lds.state_model
    d = _state_latent_dim(sm)
    R = zeros(T, d, d)
    N = zero(T)
    for u in units
        hss[u].nk[1] > zero(T) || continue
        R .+= _free_residual_scatter(thetas[u], hss[u])
        N += T(hss[u].nk[1])
    end
    N > zero(T) || return nothing
    pr = sm.Σ_prior
    if pr === nothing
        R ./= N
    else
        R .+= T.(pr.Ψ)
        R ./= T(pr.ν) + N + T(d) + one(T)
    end
    Symmetrize!(R)
    copyto!(sm.Σ, R)
    @debug "free noise" N = N sigma_min = minimum(eigvals(Symmetric(R))) sigma_max = maximum(
        eigvals(Symmetric(R))
    )
    return nothing
end

"""Write one solved `Θ` back into the arrays its version's models share."""
function _write_free_theta!(sm::LQRStateModel{T}, Theta::AbstractMatrix{T}) where {T<:Real}
    d = _state_latent_dim(sm)
    m = size(sm.Bu, 2)
    @views begin
        copyto!(sm.Mfree, Theta[:, 1:d])
        copyto!(sm.h, Theta[:, d + 1])
        m > 0 && copyto!(sm.Bu, Theta[:, (d + 2):(d + 1 + m)])
    end
    return nothing
end

# ============================================================================
# ELBO
# ============================================================================

"""
    _lqr_current_residuals(sm, hs) -> (R, Rf)

Residual scatters at the model's exact stored parameters. This is the lightweight
ELBO path: unlike the structural optimizer context it does not repack PSD blocks,
allocate gradient storage, or reconstruct parameters from an optimization
vector. `Rf` sums one terminal design against each endpoint regime's statistics.
"""
function _lqr_current_residuals(
    sm::LQRStateModel{T}, hs::LQRSufficientStatistics{T}
) where {T<:Real}
    n = _plant_dim(sm)
    d = 2n
    K = _nregimes(sm)
    m = size(sm.Bu, 2)
    reg = d + 1 + m
    xr, lr = 1:n, (n + 1):d

    R = copy(hs.Yv)
    for k in 1:K
        Th = zeros(T, d, reg)
        @views begin
            Th[xr, xr] .= sm.A
            Th[xr, lr] .= .-sm.S
            Th[lr, xr] .= sm.Qc[k]
            Th[lr, lr] .= transpose(sm.A)
            Th[:, d + 1] .= sm.h
            if m > 0
                Bcol = Th[:, (d + 2):reg]
                Bcol .= sm.Bu
                mul!(Bcol[lr, :], sm.Qc[k], _gref_for(sm, k), -one(T), one(T))
            end
        end
        TX = Th * transpose(hs.Xv[k])
        R .-= TX
        R .-= transpose(TX)
        R .+= Th * hs.Zw[k] * transpose(Th)
    end
    Symmetrize!(R)

    Rf = zeros(T, n, n)
    if sm.terminal
        for k in 1:K
            hs.term_n[k] > zero(T) || continue
            Psi = zeros(T, n, reg)
            @views begin
                Psi[:, xr] .= .-sm.Qc[k]
                for i in 1:n
                    Psi[i, n + i] = one(T)
                end
                Psi[:, d + 1] .= .-sm.hf
                m > 0 && mul!(Psi[:, (d + 2):reg], sm.Qc[k], _gref_for(sm, k))
            end
            Rf .+= Psi * hs.Omega[k] * transpose(Psi)
        end
        Symmetrize!(Rf)
    end
    return R, Rf
end

"""
    Q_state!(sws, lds, hs) -> T

State-side expected complete-data log-likelihood for an LQR LDS:

```math
Q = \\underbrace{-\\tfrac12\\left(N_1 (2n\\log 2\\pi + \\log\\det P_0)
      + \\operatorname{tr}(P_0^{-1} S_1)\\right)}_{\\text{initial state}}
  - \\tfrac12\\left(N (2n\\log 2\\pi + \\log\\det\\Sigma - 2\\log|\\det A|)
      + \\operatorname{tr}(\\Sigma^{-1} R)\\right)
  - \\tfrac12\\left(N_f (n\\log 2\\pi + \\log\\det\\Sigma_f)
      + \\operatorname{tr}(\\Sigma_f^{-1} R_f)\\right).
```

The transition term is evaluated in mixed coordinates, using
`log det Q^{fwd} = log det Σ − 2 log|det A|` and
`tr((Q^{fwd})^{-1} R^z) = tr(Σ^{-1} R)`. That is exactly the identity the M-step
rests on, so computing the ELBO the same way keeps the two consistent to the
last bit — and the `2 log|det A|` is why an ELBO that ignored the Jacobian would
drift from the objective EM is actually improving.

Fills the mixed blocks itself, so it is safe to call directly on freshly
aggregated statistics.
"""
function _lqr_joint_Q_state!(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    hs::LQRSufficientStatistics{T},
) where {T<:Real,S<:LQRStateModel{T},O<:AbstractObservationModel{T}}
    sm = lds.state_model
    suf = _state_suf(hs.base)
    n = _plant_dim(sm)
    d = lds.latent_dim
    x0 = sm.x0
    log2π = log(T(2π))

    P0_PD = sws.consts.P0_PD
    P0_U = P0_PD.chol.U
    N1 = suf.init_n

    # S_init = Σ E[z₁z₁ᵀ] − μ x0ᵀ − x0 μᵀ + N₁ x0 x0ᵀ
    S_init = sws.elbo.temp
    fill!(S_init, zero(T))
    _accumulate_init_scatter!(S_init, lds, suf)
    ldiv!(P0_U', S_init)
    ldiv!(P0_U, S_init)
    Q_val = T(-0.5) * (T(N1) * (T(d) * log2π + logdet(P0_PD)) + tr(S_init))

    # `:free` mode has no constrained parameterization to profile through.
    _is_free(sm) && return Q_val + _free_Q_transition(sm, hs)
    #= `:hold` mode scores its forward transition straight off the cache — the
    same density the smoother used, so the bound is the smoother's own. =#
    _is_hold(sm) && return Q_val + _forward_Q_transition(sm, hs)
    _is_causal(sm) && return Q_val + _causal_Q_transition(sm, hs)

    #=
    Fill the mixed blocks here rather than relying on the caller: this is reached
    from the ungrouped `elbo!` and from the grouped one, and an ordering hazard
    that silently reads stale blocks is not worth the block copies it saves.
    =#
    _fill_mixed_blocks!(hs, sm)

    # Transition term, in mixed coordinates, at the exact current parameters.
    R, Rf = _lqr_current_residuals(sm, hs)
    N = sum(hs.nk)

    if N > zero(T)
        Σ_PD = PDMat(Symmetrize!(Matrix{T}(sm.Σ)))
        Q_val +=
            T(-0.5) *
            (N * (T(d) * log2π + logdet(Σ_PD) - T(2) * sm.cache.logabsdetA) + tr(Σ_PD \ R))
    end

    Nf = sum(hs.term_n)
    if sm.terminal && Nf > zero(T)
        Σf_PD = PDMat(Symmetrize!(Matrix{T}(sm.Σf)))
        Q_val += T(-0.5) * (Nf * (T(n) * log2π + logdet(Σf_PD)) + tr(Σf_PD \ Rf))
    end
    # Entry transitions, which the regime statistics left out.
    _has_entries(sm) && (Q_val += _lqr_entry_Q(sm, hs))

    return Q_val
end
