# Terminal-conditioned LQR. The normalizer is a Gaussian integral over the
# complete trajectory, not the expected terminal residual under q(z | y).

"""The distinct terminal designs an aggregate covers, with their trial counts.

Already compressed by [`_aggregate_lqr_stats!`](@ref); this only pairs the two
vectors up.
"""
function _lqr_terminal_designs(hs::LQRSufficientStatistics)
    return [(ux=u, count=c) for (u, c) in zip(hs.terminal_inputs, hs.terminal_counts)]
end

"""Compress identical input trajectories (including their horizons).

For callers holding one entry per trial rather than an aggregate — the switching
probe, which builds its designs straight from a `Data`.
"""
function _lqr_terminal_designs(inputs::Vector{Matrix{T}}) where {T}
    counts = Dict{Matrix{T},Int}()
    for u in inputs
        counts[u] = get(counts, u, 0) + 1
    end
    return [(ux=u, count=count) for (u, count) in counts]
end

"""Exact log p(terminal=0 | inputs), using backward square-root integration.
Whitening after each transition avoids exponentially large forward covariances."""
function _lqr_terminal_logz(sm::LQRStateModel{T}, ux::AbstractMatrix{T}) where {T}
    sm.terminal || return zero(T)
    n = _plant_dim(sm)
    horizon = size(ux, 2)
    kf = _terminal_regime(sm, horizon)
    c = sm.cache
    Lf = cholesky(Symmetric(Matrix(sm.Σf))).L
    H = Lf \ c.Lf[kf]
    a = Lf \ (sm.hf - c.Ftrm[kf] * view(ux, :, horizon))
    value = -sum(log, abs.(diag(Lf)))
    Lq = c.G * cholesky(Symmetric(Matrix(sm.Σ))).L
    Id = Matrix{T}(I, n, n)
    for t in (horizon - 1):-1:1
        k = _regime(sm, t)
        L = LowerTriangular(Matrix(qr(transpose(hcat(Id, H * Lq))).R)')
        a = L \ (a - H * (c.bfwd + c.Bfwd[k] * view(ux, :, t)))
        H = L \ (H * c.M[k])
        value -= sum(log, abs.(diag(L)))
    end
    L0 = cholesky(Symmetric(Matrix(sm.P0))).L
    L = LowerTriangular(Matrix(qr(transpose(hcat(Id, H * L0))).R)')
    residual = L \ (a - H * sm.x0)
    return value - sum(log, abs.(diag(L))) -
           T(0.5) * (T(n) * log(T(2π)) + sum(abs2, residual))
end

"""Sum terminal log normalizers, factoring each distinct horizon only once.

The backward covariance recursion depends on the model and trial length, not
on the input trajectory. The backward mean recursion still runs for every
distinct input, with its trial count as weight.
"""
function _lqr_terminal_logz_sum(sm::LQRStateModel{T}, designs) where {T}
    sm.terminal || return zero(T)
    by_horizon = Dict{Int,Vector{Int}}()
    for (i, design) in enumerate(designs)
        push!(get!(by_horizon, size(design.ux, 2), Int[]), i)
    end
    n = _plant_dim(sm)
    c = sm.cache
    Lf = cholesky(Symmetric(Matrix(sm.Σf))).L
    Lq = c.G * cholesky(Symmetric(Matrix(sm.Σ))).L
    L0 = cholesky(Symmetric(Matrix(sm.P0))).L
    Id = Matrix{T}(I, n, n)
    total = zero(T)
    for (horizon, indices) in by_horizon
        kf = _terminal_regime(sm, horizon)
        H = Lf \ c.Lf[kf]
        logscale = -sum(log, abs.(diag(Lf)))
        # Store the input-independent factors in backward time order.
        steps = Vector{Tuple{Int,Matrix{T},Matrix{T},Int}}(undef, horizon - 1)
        for (j, t) in enumerate((horizon - 1):-1:1)
            k = _regime(sm, t)
            L = LowerTriangular(Matrix(qr(transpose(hcat(Id, H * Lq))).R)')
            steps[j] = (t, Matrix(L), H, k)
            H = L \ (H * c.M[k])
            logscale -= sum(log, abs.(diag(L)))
        end
        L = LowerTriangular(Matrix(qr(transpose(hcat(Id, H * L0))).R)')
        logscale -= sum(log, abs.(diag(L)))
        for i in indices
            design = designs[i]
            ux = design.ux
            a = Lf \ (sm.hf - c.Ftrm[kf] * view(ux, :, horizon))
            for (t, Lt, Ht, k) in steps
                a = LowerTriangular(Lt) \ (a - Ht * (c.bfwd + c.Bfwd[k] * view(ux, :, t)))
            end
            residual = L \ (a - H * sm.x0)
            total +=
                design.count *
                (logscale - T(0.5) * (T(n) * log(T(2π)) + sum(abs2, residual)))
        end
    end
    return total
end

function Q_state!(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    hs::LQRSufficientStatistics{T},
) where {T<:Real,S<:LQRStateModel{T},O<:AbstractObservationModel{T}}
    value = _lqr_joint_Q_state!(sws, lds, hs)
    sm = lds.state_model
    if sm.terminal && sm.condition_terminal
        isempty(hs.terminal_inputs) && error(
            "terminal conditioning needs the trials' inputs and horizons, which " *
            "`_aggregate_lqr_stats!` records; this `Q_state!` was handed statistics " *
            "that no aggregation pass filled",
        )
        value -= _lqr_terminal_logz_sum(sm, _lqr_terminal_designs(hs))
    end
    return value
end

# A zero-loading Gaussian emission contributes only a constant. Its smoother
# therefore computes exactly p(z | terminal=0, inputs). Fisher's identity gives
# d log Z / d theta = E_prior_terminal[d log p(z, terminal) / d theta].
function _lqr_terminal_probe(sm::LQRStateModel{T}, hs) where {T}
    designs = _lqr_terminal_designs(hs)
    isempty(designs) && error("Terminal conditioning requires trial inputs/horizons")
    probe = deepcopy(sm)
    probe.condition_terminal = false
    probe.depends_on = nothing
    probe.variants = nothing
    probe.P0_prior = probe.x0_prior = probe.Σ_prior = probe.Qc_prior = nothing
    d = _state_latent_dim(sm)
    lds = LinearDynamicalSystem(
        probe, GaussianObservationModel(zeros(T, 1, d), ones(T, 1, 1), zeros(T, 1))
    )
    ux = [v.ux for v in designs]
    ys = [zeros(T, 1, size(u, 2)) for u in ux]
    data = Data(lds, ys; ux=ux)
    tfs = initialize_FilterSmooth(lds, data.tsteps)
    pool = _lqr_sws_pool(lds, data)
    hs = _initialize_td_sufficient_statistics(T, lds, data.tsteps)
    weights = [fill(T(v.count), size(v.ux, 2)) for v in designs]
    return (; lds, data, tfs, pool, hs, weights, designs)
end

function _lqr_sync_probe!(probe, sm)
    target = probe.lds.state_model
    for key in (:A, :S, :h, :Bu, :Gref, :hf, :Σ, :Σf, :x0, :P0)
        copyto!(getproperty(target, key), getproperty(sm, key))
    end
    for k in eachindex(sm.Qc)
        copyto!(target.Qc[k], sm.Qc[k])
    end
    refresh!(target)
    return probe
end

function _lqr_probe_statistics!(probe)
    (; lds, data, tfs, pool, hs, weights) = probe
    smooth!(lds, tfs, data, pool)
    _aggregate_td_suff_stats_weighted!(hs.base, tfs, lds, data, weights, pool[1])
    _aggregate_lqr_stats_weighted!(hs, tfs, lds, data, weights)
    _fill_mixed_blocks!(hs, lds.state_model)
    return hs
end

function _lqr_refresh_precision!(ctx)
    for s in 1:(ctx.nq)
        sm = ctx.sms[findfirst(==(s), ctx.q_of)]
        copyto!(ctx.Sinv[s], inv(cholesky(Symmetric(Matrix(sm.Σ)))))
        copyto!(ctx.Sfinv[s], inv(cholesky(Symmetric(Matrix(sm.Σf)))))
    end
    return ctx
end

# ============================================================================
# Normalizer backends
# ============================================================================

#=
Both the switching and the non-switching M-step minimize

    f(θ) = -Q_data(θ | θ′) + log Z(θ),

and both reach `∂ log Z / ∂θ` the same way: Fisher's identity turns it into the
moments of the terminal-conditioned *prior*, which a probe model with a
zero-loading emission produces through the ordinary smoother. The backends
differ only in how the **value** of `log Z` is obtained.

`_LQRExactNormalizer` computes it exactly, by Gaussian integration over the
whole chain. `f` is then a genuine majorant of `-log p(y | terminal = 0, θ)` and
EM is monotone on the conditional objective.

`_SLQRNormalizer` cannot: `log Z` for a switching model sums over `K^T` discrete
paths. It substitutes the probe's own variational bound at the current probe
posterior, which is tight at `θ′` and has the right gradient there, but is a
surrogate rather than a majorant — hence [`_slds_terminal_report`](@ref), which
reports the joint ELBO and the normalizer separately so the approximation stays
visible in the fit trace.

It is worse than "not a majorant": holding the probe posterior `r` fixed makes
its term a *lower* bound on `log Z(θ)`, so the surrogate under-states `f` away
from `θ′`, and wherever the probe's weighted scatter exceeds the data's the
difference `-Q_data + Q_probe` is unbounded below — a line search follows it
into a singular `Σ`. The inner optimizer may therefore only *propose*; what
decides is [`_lqr_accept_conditional!`](@ref), against the normalizer
re-evaluated at the proposal ([`_terminal_score_logz`](@ref)), which for the
switching backend means re-smoothing the probe there.
=#
struct _LQRExactNormalizer{P}
    probes::P
end

struct _SLQRNormalizer{P}
    probe::P
end

"""Whether the probe's own expected log-density stands in for `log Z`'s value.

`false` means the value comes from [`_terminal_extra_value`](@ref) instead, and
the data counts enter the covariance terms undiminished.
"""
_terminal_probe_weight(::_LQRExactNormalizer) = false
_terminal_probe_weight(::_SLQRNormalizer) = true

"""The part of `log Z` that is not already in the probe's expected log-density."""
function _terminal_extra_value(b::_LQRExactNormalizer, sms)
    total = zero(eltype(sms[1].A))
    for (sm, probe) in zip(sms, b.probes)
        total += _lqr_terminal_logz_sum(sm, probe.designs)
    end
    return total
end

_terminal_extra_value(::_SLQRNormalizer, sms) = zero(eltype(sms[1].A))

"""
    _terminal_score_logz(backend, sms) -> T

`log Z` at the models' *current* parameters, obtained the same way the reported
score obtains it — the number a proposal is accepted or rejected on.

For the exact backend that is the Gaussian integral itself. For the switching
one it is a fresh probe E-step, which also moves the probe's posterior to these
parameters; [`_terminal_save`](@ref) / [`_terminal_restore!`](@ref) are how a
caller that still needs the old posterior gets it back.
"""
_terminal_score_logz(b::_LQRExactNormalizer, sms) = _terminal_extra_value(b, sms)

"""Whether the score can differ from the surrogate's value at all. It cannot
when `log Z` is exact, and a caller may then skip computing it twice."""
_terminal_rescores(::_LQRExactNormalizer) = false
_terminal_rescores(::_SLQRNormalizer) = true

"""
    _terminal_save(backend) / _terminal_restore!(backend, saved)

Whatever of the backend's state the surrogate reads and scoring overwrites. The
exact backend recomputes its probe moments at every point, so has none.
"""
_terminal_save(::_LQRExactNormalizer) = nothing
_terminal_restore!(::_LQRExactNormalizer, ::Nothing) = nothing

"""Terminal-conditioned prior moments at the current parameters."""
function _terminal_probe_stats!(b::_LQRExactNormalizer, sms)
    for (sm, probe) in zip(sms, b.probes)
        _lqr_sync_probe!(probe, sm)
        _lqr_probe_statistics!(probe)
    end
    return ([p.hs for p in b.probes], [p.lds.state_model for p in b.probes])
end

"""
    _lqr_rejectable(err) -> Bool

Whether `err` is a numerical failure at one point — something a line search
should answer by rejecting that point — rather than a bug that must surface.

Unwraps the task wrappers on the way down. The probe's smoother runs its trials
through `tforeach`, so a LAPACK failure inside one arrives as a
`TaskFailedException` around the real cause, and a predicate that only matched
the leaf types would let a perfectly ordinary rejected step kill the whole fit.
"""
function _lqr_rejectable(err)
    err isa PosDefException && return true
    err isa SingularException && return true
    err isa LAPACKException && return true
    err isa NumericalStabilityError && return true
    err isa DomainError && return true
    err isa TaskFailedException && return _lqr_rejectable(err.task.result)
    err isa CompositeException &&
        return !isempty(err.exceptions) && all(_lqr_rejectable, err.exceptions)
    return false
end

"""The free part of a covariance in the conditional M-step."""
function _lqr_conditional_array(sm::LQRStateModel, key::Symbol)
    array = getproperty(sm, key)
    if key === :Σ && sm.fixed_costate_sigma !== nothing
        n = _plant_dim(sm)
        return view(array, 1:n, 1:n)
    end
    return array
end

"""Build a joint conditional generalized M-step, including initial/noise blocks.
No covariance is profiled out: its formerly conjugate optimum is invalid after
subtracting log Z. Grouped parameter versions are packed and differentiated once."""
function _lqr_conditional_problem(
    ldss::AbstractVector{<:LinearDynamicalSystem{T,<:LQRStateModel{T}}},
    sufs::AbstractVector{<:LQRSufficientStatistics{T}},
    slots::AbstractVector{<:AbstractVector{Int}},
) where {T<:Real}
    sms = [lds.state_model for lds in ldss]
    probes = [_lqr_terminal_probe(sm, hs) for (sm, hs) in zip(sms, sufs)]
    return _lqr_conditional_problem(
        ldss, sufs, slots, _lqr_cell_slots(sms, slots[_G_AB]), _LQRExactNormalizer(probes)
    )
end

function _lqr_conditional_problem(
    ldss::AbstractVector{<:LinearDynamicalSystem{T,<:LQRStateModel{T}}},
    sufs::AbstractVector{<:LQRSufficientStatistics{T}},
    slots::AbstractVector{<:AbstractVector{Int}},
    blockslots::NTuple{7,Vector{Int}},
    backend,
) where {T<:Real}
    sms = [lds.state_model for lds in ldss]
    d, n = _state_latent_dim(sms[1]), _plant_dim(sms[1])
    fit = ldss[1].fit_bool
    flags = if fit[3]
        sms[1].fit_flags
    else
        LQRFitFlags(; A=false, S=false, Qc=false, h=false, Bu=false, Gref=false, terminal=false)
    end
    probe_weight = _terminal_probe_weight(backend)
    for (hs, sm) in zip(sufs, sms)
        _fill_mixed_blocks!(hs, sm)
    end
    ctx = _LQRMStepCtx(sufs, sms, blockslots, slots[_G_Q], false; flags=flags)
    np = ctx.pack.np
    theta = zeros(T, np)
    _lqr_pack!(theta, ctx)
    extras = NamedTuple{
        (:key, :slot, :owners, :range, :factor),
        Tuple{Symbol,Int,Vector{Int},UnitRange{Int},Matrix{T}},
    }[]
    for (key, slot, enabled, width) in (
        (:x0, _G_X0, fit[1], d),
        (:P0, _G_P0, fit[2], d * (d + 1) ÷ 2),
        (
            :Σ,
            _G_Q,
            fit[4],
            sms[1].fixed_costate_sigma === nothing ? d * (d + 1) ÷ 2 : n * (n + 1) ÷ 2,
        ),
        (:Σf, _G_Q, fit[4], n * (n + 1) ÷ 2),
    )
        enabled || continue
        for v in unique(slots[slot])
            owners = findall(==(v), slots[slot])
            array = _lqr_conditional_array(sms[first(owners)], key)
            r = (length(theta) + 1):(length(theta) + width)
            append!(theta, zeros(T, width))
            factor = key === :x0 ? zeros(T, 0, 0) : zeros(T, size(array))
            if key === :x0
                theta[r] .= array
            else
                _lqr_pack_psd!(theta, r, array, string(key))
            end
            push!(extras, (; key, slot=v, owners, range=r, factor))
        end
    end
    function write!(theta)
        _lqr_writeback!(ctx, view(theta, 1:np))
        for block in extras
            array = _lqr_conditional_array(sms[first(block.owners)], block.key)
            if block.key === :x0
                copyto!(array, view(theta, block.range))
            else
                _lqr_unpack_psd!(array, block.factor, theta, block.range)
                isposdef(Symmetric(array)) || throw(PosDefException(0))
            end
            for c in block.owners[2:end]
                copyto!(_lqr_conditional_array(sms[c], block.key), array)
            end
        end
        foreach(refresh!, sms)
        _lqr_refresh_precision!(ctx)
        return nothing
    end
    #=
    `score = true` evaluates the objective the step is judged on rather than the
    one the line search descends: `-Q_data` in full, plus `log Z` from
    `_terminal_score_logz`. Under the exact backend the two coincide. Under the
    switching one the probe's fixed-posterior density drops out and the probe is
    re-smoothed at `theta` instead — value only, since that is all a judgement
    needs.
    =#
    function evaluate!(gradient, theta; score::Bool=false)
        score &&
            gradient !== nothing &&
            throw(ArgumentError("the acceptance score has no gradient"))
        gradient === nothing || fill!(gradient, zero(T))
        try
            write!(theta)
            gs = gradient === nothing ? nothing : view(gradient, 1:np)
            value = _lqr_fg!(gs, view(theta, 1:np), ctx)
            isfinite(value) || return T(Inf)

            #=
            The probe supplies `∂ log Z / ∂θ` through Fisher's identity, and for
            a switching model its expected log-density also stands in for the
            value. A value-only call under an exact normalizer does not need it,
            and skipping it there is what keeps the acceptance check cheap.
            =#
            weighted = probe_weight && !score
            want_probe = !score && (gradient !== nothing || probe_weight)
            psufs, psms =
                want_probe ? _terminal_probe_stats!(backend, sms) : (nothing, nothing)
            negative = if want_probe
                _LQRMStepCtx(psufs, psms, blockslots, slots[_G_Q], false; flags=flags)
            else
                nothing
            end
            if want_probe
                ng = gs === nothing ? nothing : zeros(T, np)
                nvalue = _lqr_fg!(ng, view(theta, 1:np), negative)
                isfinite(nvalue) || return T(Inf)
                probe_weight && (value -= nvalue)
                gs === nothing || (gs .-= ng)
            end
            #=
            Counts net of the probe's. They cancel exactly when the probe sees
            the same trials with the same horizons, which is every non-switching
            case; under switching the two posteriors put different
            responsibilities on the same timesteps and the difference is real.
            =#
            netN(s) = ctx.N_q[s] - (negative === nothing ? zero(T) : negative.N_q[s])
            netNf(s) = ctx.Nf_q[s] - (negative === nothing ? zero(T) : negative.Nf_q[s])

            #=
            Full transition/terminal covariance terms, omitted by the
            fixed-covariance structural objective.

            Through the Cholesky rather than `logdet(Symmetric(·))`: the latter
            reads the sign off an eigendecomposition, which disagrees with
            `cholesky` on the ill-conditioned covariances a line search walks
            through, and reports that disagreement as a `DomainError` from
            `log(-1.0)` instead of as the rejected step it is.
            =#
            for s in 1:(ctx.nq)
                sm = sms[findfirst(==(s), ctx.q_of)]
                chol_Σ = cholesky(Symmetric(sm.Σ); check=false)
                chol_Σf = cholesky(Symmetric(sm.Σf); check=false)
                (issuccess(chol_Σ) && issuccess(chol_Σf)) || return T(Inf)
                wN = weighted ? netN(s) : ctx.N_q[s]
                wNf = weighted ? netNf(s) : ctx.Nf_q[s]
                value += T(0.5) * (wN * logdet(chol_Σ) + wNf * logdet(chol_Σf))
                sm.Σ_prior === nothing || (value += _iw_penalty(sm.Σ, sm.Σ_prior))
            end
            value += if score
                _terminal_score_logz(backend, sms)
            else
                _terminal_extra_value(backend, sms)
            end
            # Initial state and its priors, including cross-group x0/P0 pairs.
            for (c, (sm, hs)) in enumerate(zip(sms, sufs))
                base = _state_suf(hs.base)
                mu = vec(base.init_xy)
                count = T(base.init_n)
                yy = copy(base.init_yy[])
                if weighted
                    b = _state_suf(psufs[c].base)
                    mu -= vec(b.init_xy)
                    count -= T(b.init_n)
                    yy -= b.init_yy[]
                end
                R = yy - mu * sm.x0' - sm.x0 * mu' + count * sm.x0 * sm.x0'
                chol_P0 = cholesky(Symmetric(sm.P0); check=false)
                issuccess(chol_P0) || return T(Inf)
                W = inv(chol_P0)
                value += T(0.5) * (count * logdet(chol_P0) + dot(W, R))
            end
            for c in _slot_representatives(slots[_G_P0])
                sm = sms[c]
                sm.P0_prior === nothing || (value += _iw_penalty(sm.P0, sm.P0_prior))
            end
            for c in _pair_slot_representatives(slots[_G_X0], slots[_G_P0])
                sm = sms[c]
                value -= mn_logprior_term(reshape(sm.x0, :, 1), sm.P0, sm.x0_prior)
            end
            gradient === nothing && return value

            #=
            Fisher identity: the terminal-conditioned prior moments subtracted
            from the data-posterior moments, plus the net normalization counts,
            which vanish whenever the two posteriors carry the same weight.
            =#
            gradients = Dict(
                :x0 => [zeros(T, d) for _ in 1:maximum(slots[_G_X0])],
                :P0 => [zeros(T, d, d) for _ in 1:maximum(slots[_G_P0])],
                :Σ => [zeros(T, d, d) for _ in 1:(ctx.nq)],
                :Σf => [zeros(T, n, n) for _ in 1:(ctx.nq)],
            )
            for s in 1:(ctx.nq)
                W, Wf = ctx.Sinv[s], ctx.Sfinv[s]
                gradients[:Σ][s] .=
                    T(0.5) .* (netN(s) .* W .- W * (ctx.R[s] - negative.R[s]) * W)
                gradients[:Σf][s] .=
                    T(0.5) .* (netNf(s) .* Wf .- Wf * (ctx.Rf[s] - negative.Rf[s]) * Wf)
                sm = sms[findfirst(==(s), ctx.q_of)]
                sm.Σ_prior === nothing ||
                    _iw_penalty_grad!(gradients[:Σ][s], sm.Σ, sm.Σ_prior)
            end
            for (c, (sm, hs)) in enumerate(zip(sms, sufs))
                a, b = _state_suf(hs.base), _state_suf(psufs[c].base)
                delta_mu = vec(a.init_xy - b.init_xy)
                delta_n = T(a.init_n) - T(b.init_n)
                delta_R =
                    a.init_yy[] - b.init_yy[] - delta_mu * sm.x0' - sm.x0 * delta_mu' +
                    delta_n * sm.x0 * sm.x0'
                chol_P0 = cholesky(Symmetric(sm.P0); check=false)
                issuccess(chol_P0) || return T(Inf)
                W = inv(chol_P0)
                gradients[:x0][slots[_G_X0][c]] .+= W * (delta_n .* sm.x0 .- delta_mu)
                gradients[:P0][slots[_G_P0][c]] .+=
                    T(0.5) .* (delta_n .* W .- W * delta_R * W)
            end
            for c in _slot_representatives(slots[_G_P0])
                sm = sms[c]
                sm.P0_prior === nothing ||
                    _iw_penalty_grad!(gradients[:P0][slots[_G_P0][c]], sm.P0, sm.P0_prior)
            end
            for c in _pair_slot_representatives(slots[_G_X0], slots[_G_P0])
                sm = sms[c]
                pr = sm.x0_prior
                pr === nothing && continue
                W = inv(cholesky(Symmetric(sm.P0)))
                delta = reshape(sm.x0, :, 1) - pr.M₀
                gradients[:x0][slots[_G_X0][c]] .+= vec(W * delta * pr.Λ)
                gradients[:P0][slots[_G_P0][c]] .-=
                    T(0.5) .* (W * delta * pr.Λ * delta' * W)
            end
            for block in extras
                g = gradients[block.key][block.slot]
                if block.key === :Σ && sms[1].fixed_costate_sigma !== nothing
                    g = view(g, 1:n, 1:n)
                end
                if block.key === :x0
                    gradient[block.range] .= g
                else
                    dL = (g + g') * block.factor
                    _lqr_pack_psd_gradient!(gradient, block.range, dL, block.factor)
                end
            end
            return value
        catch err
            if _lqr_rejectable(err)
                gradient === nothing || fill!(gradient, zero(T))
                return T(Inf)
            end
            rethrow()
        end
    end
    #=
    Scoring re-smooths a switching probe, and the surrogate reads the posterior
    that probe was smoothed with at θ′. Put it back afterwards, so the two can be
    called in any order: every surrogate value stays the one the line search saw.
    =#
    function score!(theta)
        saved = _terminal_save(backend)
        try
            return evaluate!(nothing, theta; score=true)
        finally
            _terminal_restore!(backend, saved)
        end
    end
    rescores = _terminal_rescores(backend)
    return (; theta, evaluate!, write!, score!, rescores)
end

"""
    _lqr_conditional_mstep!(ldss, sufs, slots)

Generalized M-step for the terminal-conditioned objective.

`-Q(θ | θ\u2032) + log Z(θ)` majorizes `-log p(y | terminal = 0, θ)`: the EM surrogate
`Q` minorizes the joint, and `log Z` is exact rather than bounded, so the sum is a
genuine majorant that touches the true objective at `θ\u2032`. Decreasing it therefore
decreases the *conditional* negative log-likelihood, and EM stays monotone on the
objective the model now reports.

The line search descends the backend's surrogate; the step is judged by
[`_lqr_accept_conditional!`](@ref), which also re-evaluates `log Z` at each
candidate. The two agree under an exact normalizer. Under the switching one they
do not, and a proposal they reject is backtracked toward `θ\u2032` rather than
dropped outright.
"""
function _lqr_conditional_mstep!(
    ldss::AbstractVector{<:LinearDynamicalSystem{T,<:LQRStateModel{T}}},
    sufs::AbstractVector{<:LQRSufficientStatistics{T}},
    slots::AbstractVector{<:AbstractVector{Int}},
) where {T<:Real}
    return _lqr_conditional_mstep!(_lqr_conditional_problem(ldss, sufs, slots), ldss)
end

function _lqr_conditional_mstep!(
    ldss::AbstractVector{<:LinearDynamicalSystem{T,<:LQRStateModel{T}}},
    sufs::AbstractVector{<:LQRSufficientStatistics{T}},
    slots::AbstractVector{<:AbstractVector{Int}},
    blockslots::NTuple{7,Vector{Int}},
    backend,
) where {T<:Real}
    return _lqr_conditional_mstep!(
        _lqr_conditional_problem(ldss, sufs, slots, blockslots, backend), ldss
    )
end

function _lqr_conditional_mstep!(problem::NamedTuple, ldss::AbstractVector)
    (; theta, evaluate!) = problem
    isempty(theta) && return nothing
    #=
    Evaluated *with* a gradient, so the probe runs here at θ\u2032 rather than first
    being exercised somewhere inside the line search. A probe that cannot be
    smoothed at the incoming parameters makes every later evaluation infinite
    too, and the M-step would then quietly accept `theta` and report nothing —
    a fit that silently stops updating its state parameters. Fail here instead,
    and say what to do about it.
    =#
    cached_grad = similar(theta)
    initial = evaluate!(cached_grad, theta)
    gradient = copy(cached_grad)
    baseline = isfinite(initial) && problem.rescores ? problem.score!(theta) : initial
    isfinite(baseline) || error(
        "the terminal-conditioned M-step objective is not finite at the current " *
        "parameters. Its normalizer is smoothed on a copy of the model carrying no " *
        "observations, which is the hardest case for the Laplace smoother: an " *
        "unstable symplectic transition, a near-singular `Sigma`, or a cost far from " *
        "its prior mode can all put that chain out of reach. Tighten the `Sigma` / " *
        "`Qc` priors, shorten the trials, or fit the joint objective instead by " *
        "setting `condition_terminal = false`.",
    )
    #=
    One `evaluate!` produces the value and the gradient together, and its
    expensive half is the probe smoothing behind the Fisher-identity gradient.
    Optim asks for the two through separate callbacks, so memoize the last point:
    a line search that evaluates both at the same θ then pays for one pass.
    =#
    seen = copy(theta)
    cached_value = Ref(initial)
    function refresh!(x)
        seen == x && return cached_value[]
        copyto!(seen, x)
        cached_value[] = evaluate!(cached_grad, x)
        return cached_value[]
    end
    f_obj(x) = refresh!(x)
    function g_obj!(G, x)
        refresh!(x)
        copyto!(G, cached_grad)
        return G
    end
    options = Optim.Options(;
        iterations=ldss[1].state_model.mstep_iters,
        g_abstol=1e-8,
        f_reltol=1e-12,
        x_abstol=1e-10,
    )
    #=
    A line search that cannot bracket a decrease — because every trial point it
    tried put the probe's chain out of reach — leaves no proposal, but not
    nothing to do: the steepest-descent fallback in the acceptance step is still
    open. Anything that is not a numerical failure at a point still surfaces.
    =#
    failure = nothing
    proposal = try
        Optim.minimizer(
            optimize(f_obj, g_obj!, theta, LBFGS(; linesearch=HagerZhang()), options)
        )
    catch err
        (err isa LineSearchException || _lqr_rejectable(err)) || rethrow()
        failure = (err, catch_backtrace())
        theta
    end
    moved = _lqr_accept_conditional!(problem, proposal, (initial, baseline), gradient)
    #=
    Declining to move is legitimate generalized EM — the E-step runs again from
    the same parameters and the objective is unchanged rather than worse — but
    after a failed line search it is worth saying why.
    =#
    if !moved && failure !== nothing
        @warn "terminal-conditioned M-step made no progress: every trial point the " *
            "line search visited was numerically out of reach. The fit continues " *
            "at the incoming parameters." exception = failure maxlog = 3
    end
    return nothing
end

#=
Halvings `_lqr_accept_conditional!` tries along each of its two directions
before keeping `θ′`. The surrogate is unbounded below under the switching
normalizer, so the L-BFGS proposal can sit arbitrarily far out: eight halvings
reach 1/256 of that step. The steepest-descent fallback starts at unit length in
the packed coordinates, a large move already, and twelve reach ~2.4e-4.
=#
const _LQR_ACCEPT_HALVINGS = 8
const _LQR_DESCENT_HALVINGS = 12

"""
    _lqr_accept_conditional!(problem, proposal, (initial, baseline), gradient) -> Bool

Move to the first candidate that improves the terminal-conditioned objective,
or stay at `θ′ = problem.theta` if none does. Returns whether the parameters moved.

Candidates are the halvings of the step toward `proposal` (the L-BFGS
minimizer of the surrogate), then the halvings of a steepest-descent step along
`-gradient`, the surrogate's gradient at `θ′`.

A candidate is accepted when the surrogate does not exceed its value at `θ′`
(`initial`) *and* the score does not exceed its (`baseline`). Both replace
`log Z` by a lower bound — the surrogate by the probe's bound at the posterior
it was smoothed with at `θ′`, the score by a fresh one — so the true objective is
at least the larger of the two, and a candidate either one rejects is one the
evidence says is no better. The surrogate is checked first because it costs no
smoothing. Under an exact normalizer the two coincide and only one is computed.

The fallback is what makes the step reliable. Its direction is the score's own
descent direction, because at a stationary probe posterior the surrogate's
gradient equals the score's (Danskin), so unless `θ′` is stationary a short
enough step along it improves both. The L-BFGS proposal carries no such
guarantee: minimizing a surrogate that is unbounded below can end in a direction
that ascends from `θ′`.
"""
function _lqr_accept_conditional!(
    problem::NamedTuple, proposal::AbstractVector, (initial, baseline), gradient
)
    (; theta, evaluate!, write!, score!, rescores) = problem
    T = eltype(theta)
    candidate = similar(theta)
    function accepted(step)
        candidate .= theta .+ step
        surrogate = evaluate!(nothing, candidate)
        (isfinite(surrogate) && surrogate <= initial) || return false
        rescores || return true
        value = score!(candidate)
        return isfinite(value) && value <= baseline
    end
    function search!(step, halvings)
        for _ in 0:halvings
            accepted(step) && return true
            step ./= 2
        end
        return false
    end

    step = proposal .- theta
    moved = any(!iszero, step) && search!(step, _LQR_ACCEPT_HALVINGS)
    if !moved
        slope = norm(gradient)
        moved =
            isfinite(slope) &&
            slope > 0 &&
            search!(gradient .* (-one(T) / slope), _LQR_DESCENT_HALVINGS)
    end
    write!(moved ? candidate : theta)
    return moved
end
