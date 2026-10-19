# An opt-in coordinate convention, applied AFTER the unrestricted M-step.
# Priors are pushed forward too, so their pullback objective is unchanged.

_gauge_members(om) = (om,)
_gauge_members(om::CompositeObservationModel) = values(_models(om))

# A tied translated intercept may not straddle different loading matrices.
function _gauge_refines(fine, coarse)
    length(fine) == length(coarse) || return false
    assignment = Dict{Any,Any}()
    for (a, b) in zip(fine, coarse)
        haskey(assignment, a) && !isequal(assignment[a], b) && return false
        assignment[a] = b
    end
    return true
end

function _gauge_same_partition(a, b)
    return _gauge_refines(a, b) && _gauge_refines(b, a)
end

function _validate_gauge_groups(model, grp, center, whiten)
    (center || whiten) && grp !== nothing || return nothing
    # With partial coefficient ties the grouped M-step assigns each prior to
    # owned columns. Translation mixes intercept and state columns; pushing
    # these cross-terms through a different ownership rule is not supported.
    if center
        sm = model.state_model
        dep = sm.depends_on
        if dep !== nothing && sm.AB_prior !== nothing
            names = isempty(sm.B) ? (:A, :b) : (:A, :b, :B)
            active = filter(n -> haskey(dep, n), names)
            isempty(active) ||
                (
                    length(active) == length(names) &&
                    all(n -> _gauge_same_partition(dep[first(names)], dep[n]), names)
                ) ||
                throw(
                    ArgumentError(
                        "centering with a dynamics prior requires A, b and nonempty B to share grouping; partial ties are unsupported",
                    ),
                )
        end
        for om in _gauge_members(model.obs_model)
            dep = om.depends_on
            if dep !== nothing && om.CD_prior !== nothing
                names = isempty(om.D) ? (:C, :d) : (:C, :d, :D)
                active = filter(n -> haskey(dep, n), names)
                isempty(active) ||
                    (
                        length(active) == length(names) &&
                        all(n -> _gauge_same_partition(dep[first(names)], dep[n]), names)
                    ) ||
                    throw(
                        ArgumentError(
                            "centering with an emission prior requires C, d and nonempty D to share grouping",
                        ),
                    )
            end
        end
    end
    return nothing
end

function _validate_gauge(model, center, whiten, max_condition)
    isfinite(max_condition) && max_condition > 1 ||
        throw(ArgumentError("gauge_max_condition must be finite and greater than one"))
    (center || whiten) || return nothing
    model.state_model isa GaussianStateModel ||
        throw(ArgumentError("post-M-step gauge normalization requires GaussianStateModel"))
    if center
        isempty(model.state_model.B0) || throw(
            ArgumentError(
                "centering with initial-state inputs B0 is not implemented; use whitening alone",
            ),
        )
        dep = model.state_model.depends_on
        if dep !== nothing &&
            haskey(dep, :A) &&
            (!haskey(dep, :b) || !_gauge_refines(dep.b, dep.A))
            throw(
                ArgumentError(
                    "centering would break shared b where A varies; preserve the original model with gauge_center=false (whitening alone is supported). Centering at x0 has the same restriction",
                ),
            )
        end
    end
    for om in _gauge_members(model.obs_model)
        om isa Union{GaussianObservationModel,PoissonObservationModel} || throw(
            ArgumentError(
                "gauge normalization supports Gaussian and Poisson emissions only"
            ),
        )
        center &&
            om.d_fixed_zero &&
            throw(
                ArgumentError(
                    "centering is incompatible with fixed-zero emission offsets; use whitening alone",
                ),
            )
        if center &&
            om.depends_on !== nothing &&
            haskey(om.depends_on, :C) &&
            (
                !haskey(om.depends_on, :d) ||
                !_gauge_refines(om.depends_on.d, om.depends_on.C)
            )
            throw(ArgumentError("centering requires d to vary wherever C varies"))
        end
    end
    return nothing
end

function _gauge_schedule!(changes, array, value)
    all(isfinite, value) || throw(ArgumentError("nonfinite affine gauge transform"))
    if haskey(changes, array)
        isapprox(changes[array], value; rtol=1e-9, atol=1e-11) || throw(
            ArgumentError("affine gauge transform is incompatible with parameter tying")
        )
    else
        changes[array] = value
    end
    return nothing
end

# h_new = L*h_old, with h=[x;1;inputs]. The same change must be used
# on prior means and column precisions, including intercept cross terms.
function _gauge_design(T, c, width)
    k = length(c)
    L = Matrix{eltype(T)}(I, width, width)
    L[1:k, 1:k] .= T
    L[1:k, k + 1] .= -T * c
    return L
end

_gauge_iw(::Nothing, T) = nothing
_gauge_iw(p::IWPrior, T) = IWPrior(; Ψ=Matrix(Symmetric(T * p.Ψ * T')), ν=p.ν)
_gauge_mn(::Nothing, L, T, shift) = nothing
function _gauge_mn(p::MNPrior, L, T, shift)
    M = Matrix(T * p.M₀ / L)
    shift === nothing || (M[:, length(shift) + 1] .-= shift)
    return MNPrior(; M₀=M, Λ=Matrix(Symmetric(L * p.Λ * L')))
end

function _gauge_prior_plan(models, T, c)
    plan = Any[]
    seen = IdDict()
    for model in models
        sm = model.state_model
        if !haskey(seen, sm)
            seen[sm] = true
            L = _gauge_design(T, c, size(sm.A, 1) + 1 + size(sm.B, 2))
            xp = sm.x0_prior
            xpnew = if xp === nothing
                nothing
            else
                MNPrior(; M₀=isempty(sm.B0) ? T * (xp.M₀ .- c) : T * xp.M₀, Λ=copy(xp.Λ))
            end
            push!(plan, (sm, :AB_prior, _gauge_mn(sm.AB_prior, L, T, T * c)))
            push!(plan, (sm, :x0_prior, xpnew))
            push!(plan, (sm, :Q_prior, _gauge_iw(sm.Q_prior, T)))
            push!(plan, (sm, :P0_prior, _gauge_iw(sm.P0_prior, T)))
        end
        for om in _gauge_members(model.obs_model)
            haskey(seen, om) && continue
            seen[om] = true
            L = _gauge_design(T, c, size(om.C, 2) + 1 + size(om.D, 2))
            # Emission row coordinates (and R) do not change.
            p = om.CD_prior
            pnew = if p === nothing
                nothing
            else
                MNPrior(; M₀=Matrix(p.M₀ / L), Λ=Matrix(Symmetric(L * p.Λ * L')))
            end
            push!(plan, (om, :CD_prior, pnew))
        end
    end
    return plan
end

# Rebuild transformed priors when a saved, already-canonical model is loaded.
# The arrays themselves are already in the saved frame and must not move twice.
function _restore_affine_gauge_priors!(
    model, map, offset, logprior_offset; whitening_skips=0, last_condition=1.0
)
    R = eltype(model.state_model.A)
    T = Matrix{R}(map)
    c = -(T \ Vector{R}(offset))
    sm = model.state_model
    sms = sm.variants === nothing ? [sm] : [sm; sm.variants]
    oms = Any[]
    for om in _gauge_members(model.obs_model)
        push!(oms, om)
        om.variants === nothing || append!(oms, om.variants)
    end
    # The plan operates on state/emission members independently; these shells
    # are only for visiting all unique objects, never for inference.
    shells = [
        LinearDynamicalSystem{R,typeof(s),typeof(o)}(
            s, o, model.latent_dim, size(o.C, 1), size(s.B, 2), size(o.D, 2), model.fit_bool
        ) for s in sms for o in oms
    ]
    for (member, field, value) in _gauge_prior_plan(shells, T, c)
        setproperty!(member, field, value)
    end
    gauge = AffineGauge{R}(;
        map=T,
        offset=Vector{R}(offset),
        logprior_offset=R(logprior_offset),
        whitening_skips=whitening_skips,
        last_condition=R(last_condition),
    )
    for s in sms
        s.gauge = gauge
    end
    return model
end

# Only the row-covariance log determinants change. Compute their constant
# analytically: subtracting two enormous (but invariant) prior quadratics
# would lose the reference objective precisely when the scales are extreme.
function _gauge_prior_shift(models, grp, logdetT)
    R = eltype(first(models).state_model.A)
    weight = zero(R)
    k = first(models).latent_dim
    if grp === nothing
        sm = first(models).state_model
        sm.Q_prior === nothing || (weight += sm.Q_prior.ν + k + 1)
        sm.P0_prior === nothing || (weight += sm.P0_prior.ν + k + 1)
        sm.AB_prior === nothing || (weight += size(sm.AB_prior.M₀, 2))
        sm.x0_prior === nothing || (weight += size(sm.x0_prior.M₀, 2))
    else
        slots = grp.cell_slot
        for u in _slot_representatives(slots[_G_Q])
            p = models[u].state_model.Q_prior
            p === nothing || (weight += p.ν + k + 1)
        end
        for u in _slot_representatives(slots[_G_P0])
            p = models[u].state_model.P0_prior
            p === nothing || (weight += p.ν + k + 1)
        end
        for u in _pair_slot_representatives(slots[_G_X0], slots[_G_P0])
            p = models[u].state_model.x0_prior
            p === nothing || (weight += size(p.M₀, 2))
        end
        owned = _grouped_dyn_prior_cols(models, slots[_G_AB])
        for u in _pair_slot_representatives(slots[_G_AB], slots[_G_Q])
            p = models[u].state_model.AB_prior
            p === nothing && continue
            cols = _unit_prior_cols(owned, u)
            weight += cols === nothing ? size(p.M₀, 2) : length(cols)
        end
    end
    return weight * logdetT
end

"""
    _affine_gauge!(parent, models, posteriors, grp; center, whiten, max_condition)

Apply z=T*(x-c) without altering the observation distribution or the reference
MAP objective. Whitening uses a Cholesky solve, never eigenvalue clipping or
jitter. Unsafe whitening is skipped, while centering still proceeds. Q is
always transformed by congruence; it is never overwritten with an identity.
"""
function _affine_gauge!(
    parent, models, posteriors, grp; center=false, whiten=false, max_condition=1e8
)
    (center || whiten) || return nothing
    k = parent.latent_dim
    R = eltype(parent.state_model.A)
    c = zeros(R, k)
    bins = 0
    for tfs in posteriors, fs in tfs.FilterSmooths
        center && (c .+= vec(sum(fs.x_smooth; dims=2)))
        bins += size(fs.x_smooth, 2)
    end
    center && (c ./= bins)
    T = Matrix{R}(I, k, k)
    gauge = parent.state_model.gauge
    if gauge === nothing
        gauge = AffineGauge{R}(; map=copy(T), offset=zeros(R, k))
    end
    if whiten
        # A single common map is essential for tied/session readouts. Use the
        # trial-count-weighted process covariance if Q differs across cells.
        Q = zeros(R, k, k)
        n = sum(length(t.FilterSmooths) for t in posteriors)
        for (model, tfs) in zip(models, posteriors)
            Q .+= (length(tfs.FilterSmooths) / n) .* model.state_model.Q
        end
        vals = eigvals(Symmetric(Q))
        condition = minimum(vals) > 0 ? maximum(vals) / minimum(vals) : R(Inf)
        gauge.last_condition = condition
        if !all(isfinite, vals) || !isfinite(condition) || condition > max_condition
            gauge.whitening_skips += 1
            @warn "Skipping unsafe process-covariance whitening; centering remains available" condition max_condition
        else
            # A reliable eigenspectrum alone does not replace the SPD check.
            F = cholesky(Symmetric(Q); check=false)
            if issuccess(F)
                T = F.L \ T
            else
                gauge.whitening_skips += 1
                @warn "Skipping process-covariance whitening: Cholesky failed"
            end
        end
        # Whitening Q can make another covariance (especially P0) harder to
        # represent. Check the actual congruences and their round-trip before
        # touching the model, rather than trusting Q's condition number alone.
        safe = true
        for model in models, covariance in (model.state_model.Q, model.state_model.P0)
            transformed = Matrix(Symmetric(T * covariance * T'))
            if !all(isfinite, transformed) ||
                !issuccess(cholesky(Symmetric(transformed); check=false))
                safe = false
                break
            end
            restored = (T \ transformed) / T'
            if norm(restored - covariance) > 1e-8 * max(norm(covariance), floatmin(R))
                safe = false
                break
            end
        end
        if !safe
            gauge.whitening_skips += 1
            T = Matrix{R}(I, k, k)
            @warn "Skipping whitening: a covariance congruence failed its SPD/round-trip check"
        end
    end
    allmodels = [parent; models]
    changes = IdDict()
    for model in allmodels
        sm = model.state_model
        _gauge_schedule!(changes, sm.A, (T * sm.A) / T)
        _gauge_schedule!(changes, sm.b, T * (sm.b - (I - sm.A) * c))
        _gauge_schedule!(changes, sm.B, T * sm.B)
        _gauge_schedule!(changes, sm.Q, Matrix(Symmetric(T * sm.Q * T')))
        _gauge_schedule!(changes, sm.x0, T * (sm.x0 - c))
        _gauge_schedule!(changes, sm.B0, T * sm.B0)
        _gauge_schedule!(changes, sm.P0, Matrix(Symmetric(T * sm.P0 * T')))
        for om in _gauge_members(model.obs_model)
            _gauge_schedule!(changes, om.d, om.d + om.C * c)
            _gauge_schedule!(changes, om.C, om.C / T)
        end
    end
    # Prepare all changes before any model arrays are written.
    plan = _gauge_prior_plan(allmodels, T, c)
    prior_shift = _gauge_prior_shift(models, grp, logabsdet(T)[1])
    for (array, value) in changes
        copyto!(array, value)
    end
    for (model, field, value) in plan
        setproperty!(model, field, value)
    end
    for model in allmodels
        model.state_model.gauge = gauge
    end
    gauge.logprior_offset += prior_shift
    gauge.map = T * gauge.map
    gauge.offset = T * (gauge.offset - c)
    # E-step fully replaces S, lag covariances, entropy and sufficient stats on
    # the next iteration. Only its warm-start means survive this boundary.
    # Invalidate the other buffers, rather than silently mixing coordinate frames.
    identity_map = T == Matrix{R}(I, k, k)
    for tfs in posteriors, fs in tfs.FilterSmooths
        if identity_map
            fs.x_smooth .-= c
        else
            fs.x_smooth .= T * (fs.x_smooth .- c)
        end
        fill!(fs.p_packed, R(NaN))
        fill!(fs.p_smooth, R(NaN))
        fill!(fs.p_smooth_tt1, R(NaN))
        fill!(fs.xcov_sum, R(NaN))
        fs.entropy = R(NaN)
    end
    return gauge
end
