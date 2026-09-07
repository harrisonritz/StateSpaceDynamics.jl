# ============================================================================
# Held-out (test-set) scoring during EM.
#
# Every driver already records the training ELBO once per iteration. This adds
# an optional second trace: the ELBO of a held-out dataset, evaluated at the
# *same* parameters, every `test_every` iterations — and an optional early stop
# when it turns over.
#
# The scoring call is the ordinary public `elbo(model, y; ...)`, which for every
# model family here runs a deterministic E-step (an exact smoother for the
# Gaussian LDS and the Hamiltonian model, a Laplace/Newton solve for a Poisson
# emission, and deterministic coordinate ascent for the SLDS — note that this is
# *not* the SLDS fit's Monte-Carlo E-step). Scoring the same parameters on the
# same data therefore returns the same number every time, which is what makes
# stopping on it well-posed.
# ============================================================================

"""
    FitTrace{T} <: AbstractVector{T}

What [`fit!`](@ref) returns when it is given held-out data. It *is* the
per-iteration training-ELBO vector — it indexes, iterates and plots exactly as
the plain `Vector{T}` returned without held-out data, so existing code is
unaffected — and it carries the held-out trace alongside.

# Fields
- `train::Vector{T}`: training ELBO, one per EM iteration (the vector `fit!`
  returns when no test data is passed).
- `test::Vector{T}`: held-out ELBO, one per scored iteration.
- `test_iters::Vector{Int}`: the iterations at which `test` was evaluated, so
  `plot(trace.test_iters, trace.test)` lines up with `plot(trace.train)`.
- `best_iter::Int`: iteration with the highest held-out ELBO (`0` if never
  scored). This is the answer to "how many iterations before I overfit?".
- `stopped_early::Bool`: whether the fit stopped on the held-out criterion
  rather than running to `max_iter` or meeting the training tolerance.

```julia
trace = fit!(lds, y_train; y_test=y_test)
trace[end]                        # final training ELBO, as before
plot(trace)                       # training curve, as before
plot(trace.test_iters, trace.test)  # held-out curve
trace.best_iter                   # where held-out peaked
```
"""
struct FitTrace{T<:Real} <: AbstractVector{T}
    train::Vector{T}
    test::Vector{T}
    test_iters::Vector{Int}
    best_iter::Int
    stopped_early::Bool
end

Base.size(tr::FitTrace) = size(tr.train)
Base.getindex(tr::FitTrace, i::Int) = tr.train[i]
Base.IndexStyle(::Type{<:FitTrace}) = IndexLinear()

function Base.show(io::IO, ::MIME"text/plain", tr::FitTrace{T}) where {T}
    print(io, "FitTrace{$T}: $(length(tr.train)) iterations")
    if !isempty(tr.test)
        print(io, ", $(length(tr.test)) held-out evaluations, best at iteration ")
        print(io, tr.best_iter)
        tr.stopped_early && print(io, " (stopped early)")
    end
    return nothing
end

"""
    HoldoutMonitor{T}

Internal state for held-out scoring inside an EM driver: the test dataset and
the scoring/stopping policy, plus the trace accumulated so far and the
best-so-far parameter snapshot.

`nothing` in a driver's `monitor` slot means "no held-out data", which is the
default and leaves every code path exactly as it was.
"""
mutable struct HoldoutMonitor{T<:Real,Y,UX,UY,NT<:NamedTuple}
    y::Y
    ux::UX
    uy::UY
    depends_on::Union{Nothing,NamedTuple}
    kwargs::NT
    every::Int
    early_stopping::Bool
    patience::Int
    min_delta::T
    restore_best::Bool
    values::Vector{T}
    iters::Vector{Int}
    best_value::T
    best_iter::Int
    n_bad::Int
    stopped_early::Bool
    snapshot::Any
end

"""
    _holdout_monitor(T, y_test; ux_test, uy_test, depends_on_test, test_every,
                     early_stopping, patience, min_delta, restore_best, test_kwargs)

Build the monitor, or return `nothing` when no held-out data was given. Every
`fit!` calls this and passes the result down to its driver as `monitor`.
"""
function _holdout_monitor(
    ::Type{T},
    y_test;
    ux_test=nothing,
    uy_test=nothing,
    depends_on_test::Union{Nothing,NamedTuple}=nothing,
    test_every::Int=1,
    early_stopping::Bool=false,
    patience::Int=1,
    min_delta::Real=0.0,
    restore_best::Bool=true,
    test_kwargs::NamedTuple=NamedTuple(),
) where {T<:Real}
    y_test === nothing && return nothing
    test_every >= 1 || throw(ArgumentError("test_every must be >= 1, got $test_every"))
    patience >= 1 || throw(ArgumentError("patience must be >= 1, got $patience"))
    min_delta >= 0 || throw(ArgumentError("min_delta must be >= 0, got $min_delta"))
    return HoldoutMonitor{
        T,typeof(y_test),typeof(ux_test),typeof(uy_test),typeof(test_kwargs)
    }(
        y_test,
        ux_test,
        uy_test,
        depends_on_test,
        test_kwargs,
        test_every,
        early_stopping,
        patience,
        T(min_delta),
        restore_best,
        T[],
        Int[],
        T(-Inf),
        0,
        0,
        false,
        nothing,
    )
end

"""
    _holdout_due(monitor, iter) -> Bool

Whether iteration `iter` is a scoring iteration. Iteration 1 always is, so the
trace starts at the model's pre-fit held-out score and a malformed test set
fails on the first pass rather than `test_every` iterations in.
"""
@inline function _holdout_due(mon::HoldoutMonitor, iter::Int)
    return (iter - 1) % mon.every == 0
end

@inline _holdout_due(::Nothing, ::Int) = false

"""
    _holdout_record!(monitor, model, iter)

Score the held-out set at the model's current parameters, append it to the
trace, and update the best-so-far state. Drivers call this *between* the
training-ELBO evaluation and the M-step, so both traces describe the same
parameters, and any snapshot is of the parameters that earned the score.

Sets `monitor.stopped_early` when the held-out ELBO has failed to improve on
the best by more than `min_delta` for `patience` consecutive scored iterations
and `early_stopping` is on.
"""
function _holdout_record!(mon::HoldoutMonitor{T}, model, iter::Int) where {T<:Real}
    v = T(
        elbo(model, mon.y; ux=mon.ux, uy=mon.uy, depends_on=mon.depends_on, mon.kwargs...)
    )
    push!(mon.values, v)
    push!(mon.iters, iter)

    if v > mon.best_value + mon.min_delta
        mon.best_value = v
        mon.best_iter = iter
        mon.n_bad = 0
        # Snapshot only when it could be used: restoring is gated on an actual
        # early stop, so a run to `max_iter` never pays for the deepcopy.
        if mon.restore_best && mon.early_stopping
            mon.snapshot = _param_snapshot(model)
        end
    else
        mon.n_bad += 1
        if mon.early_stopping && mon.n_bad >= mon.patience
            mon.stopped_early = true
        end
    end
    return v
end

_holdout_record!(::Nothing, ::Any, ::Int) = nothing

"""
    _holdout_stop(monitor) -> Bool

Whether the driver should leave its EM loop now.
"""
@inline _holdout_stop(mon::HoldoutMonitor) = mon.stopped_early
@inline _holdout_stop(::Nothing) = false

"""
    _fit_result(monitor, train_elbos, model)

What a driver returns. With no held-out data this is the training ELBO vector
itself — byte-for-byte the previous return value, so nothing downstream
changes. With held-out data it is a [`FitTrace`](@ref), which behaves as that
same vector and carries the held-out trace.

Restores the best-scoring parameters when (and only when) early stopping
actually fired and `restore_best` is set: a fit that ran to completion is left
at its final iterate, as it always was.
"""
_fit_result(::Nothing, elbos::Vector, ::Any) = elbos

function _fit_result(mon::HoldoutMonitor{T}, elbos::Vector{T}, model) where {T<:Real}
    if mon.restore_best && mon.stopped_early && mon.snapshot !== nothing
        _restore_params!(model, mon.snapshot)
    end
    return FitTrace{T}(elbos, mon.values, mon.iters, mon.best_iter, mon.stopped_early)
end

# ============================================================================
# Parameter snapshot / in-place restore.
#
# `LinearDynamicalSystem` is immutable, so the model object the caller holds
# keeps its identity through a fit and only the (mutable) sub-models change.
# Restoring therefore writes *into* those sub-models rather than rebinding
# them. A grouped fit's per-cell models are views sharing the parent's
# `variants` arrays by reference, so snapshotting the parent captures the
# grouped parameters too.
# ============================================================================

"""
    _param_snapshot(model)

A private deep copy of everything an M-step can move. Discarded once restored,
so [`_restore_params!`](@ref) may move its fields rather than copy them again.
"""
_param_snapshot(lds::LinearDynamicalSystem) =
    (deepcopy(lds.state_model), deepcopy(lds.obs_model))

function _param_snapshot(slds::SLDS)
    return (copy(slds.A), copy(slds.πₖ), [_param_snapshot(l) for l in slds.LDSs])
end

"""
    _restore_params!(model, snapshot) -> model

Write `snapshot` back into `model` in place.
"""
function _restore_params!(lds::LinearDynamicalSystem, snap::Tuple{Any,Any})
    _copy_model!(lds.state_model, snap[1])
    _copy_model!(lds.obs_model, snap[2])
    return lds
end

function _restore_params!(slds::SLDS, snap::Tuple)
    copyto!(slds.A, snap[1])
    copyto!(slds.πₖ, snap[2])
    for (l, s) in zip(slds.LDSs, snap[3])
        _restore_params!(l, s)
    end
    return slds
end

"""
    _copy_model!(dest, src) -> dest

Field-wise copy between two state / observation models of the same type,
preserving `dest`'s object identity.
"""
function _copy_model!(dest::M, src::M) where {M}
    ismutabletype(M) || throw(
        ArgumentError(
            "cannot restore parameters into an immutable $(nameof(M)); a " *
            "`_copy_model!` method is needed for it",
        ),
    )
    for i in 1:fieldcount(M)
        setfield!(dest, i, getfield(src, i))
    end
    return dest
end

#=
A composite emission is itself immutable; its members hold the parameters, so
recurse into them. Constrained as `M<:CompositeObservationModel` rather than by
spelling out `{T,QUAD,NT}`: that form stays strictly more specific than the
generic method above even where the member `NamedTuple` type is not known, so a
composite can never fall through to the `setfield!` path.
=#
function _copy_model!(dest::M, src::M) where {M<:CompositeObservationModel}
    d = _models(dest)
    s = _models(src)
    for k in keys(d)
        _copy_model!(d[k], s[k])
    end
    return dest
end
