#=============================================================================
Scoring — how close is a fit to the truth, per parameter block.

Two numbers per block, both computed on the *entries*: a relative RMSE (divided
by the RMS of the true entries, so blocks of different magnitude are comparable)
and a Pearson correlation (which ignores scale entirely, and so answers the
different question of whether the estimate has the right shape). A block whose
truth is ~0, or whose entries are constant, has no answer to either question and
reports `NaN` rather than a number that would look like one.

Statistics.jl is a dependency of the package but not of `docs/`, and these are a
dozen lines, so they live here rather than constraining how the script is run.
=============================================================================#

_mean(v) = sum(v) / length(v)
_var(v) = (m=_mean(v); sum(abs2, v .- m) / length(v))

function _cor(a, b)
    va, vb = _var(a), _var(b)
    (va <= 1e-24 || vb <= 1e-24) && return NaN
    return _mean((a .- _mean(a)) .* (b .- _mean(b))) / sqrt(va * vb)
end

"""
    _entries(M; sym) -> Vector

The independent entries of a block: the upper triangle (diagonal included) of a
symmetric matrix, every entry otherwise. Counting a symmetric matrix's
off-diagonals twice would weight them double in the RMSE and inflate the
correlation's sample size for free.
"""
function _entries(M::AbstractMatrix; sym::Bool=false)
    return sym ? [M[i, j] for j in axes(M, 2) for i in 1:j] : vec(collect(M))
end
_entries(v::AbstractVector; sym::Bool=false) = collect(v)

"""
    score(fit, ref; sym) -> (rmse, corr)

Relative RMSE and Pearson correlation of one parameter block against its truth.
"""
function score(fit, ref; sym::Bool=false)
    f = _entries(fit; sym=sym)
    r = _entries(ref; sym=sym)
    scale = sqrt(_mean(abs2.(r)))
    rmse = scale > 1e-12 ? sqrt(_mean(abs2.(f .- r))) / scale : NaN
    return (rmse=rmse, corr=_cor(f, r))
end

const NOSCORE = (rmse=NaN, corr=NaN)

"""The `n × n` plant block of a `2n × 2n` mixed-coordinate matrix."""
_state_block(M::AbstractMatrix) = (n = size(M, 1) ÷ 2; view(M, 1:n, 1:n))

"""
    score_worst(fits, refs; sym) -> (rmse, corr)

The worst regime, when a block is a vector of matrices: the largest relative
RMSE and the smallest correlation over the cost regimes. Pooling the regimes'
entries instead would flatter the fit — the truth's regimes differ in magnitude
by an order of magnitude, and a correlation across that spread is mostly
measuring which regime an entry came from.
"""
function score_worst(fits, refs; sym::Bool=false)
    ss = [score(fits[k], refs[k]; sym=sym) for k in eachindex(refs)]
    return (rmse=maximum(s.rmse for s in ss), corr=minimum(s.corr for s in ss))
end

"""
    closed_loop_or_nothing(sm; k) -> Matrix or nothing

`(I + S P)⁻¹ A`, the steady-state closed-loop plant of regime `k`. This is the
scale-invariant summary of the whole fit — `S → S/c`, `Q → cQ` leaves `S P`
alone — so it is the one comparison that needs no canonicalization to be
meaningful. Returns `nothing` when the Riccati iteration finds no stabilizing
solution, which an intermediate fit is entitled to do.
"""
function closed_loop_or_nothing(sm; k::Int=1)
    try
        return closed_loop_dynamics(sm; k=k)
    catch err
        err isa NumericalStabilityError || rethrow()
        return nothing
    end
end

"""
    compare(fit_sm, ref, idx; known_plant, free_gref) -> NamedTuple of (rmse, corr)

Every scored block, in the canonical scale. A frozen block reads `--`: frozen at
the truth it is exactly right, and a column of `0.000/1.00` is noise in the
table rather than a result.

A frozen `Σ` reads `--` for the same reason `A` does on a known-plant row: it is
sitting exactly where it was put, and `0.000/1.00` in that column is the
initialization, not a result.

`idx` is the truth's regime map (`(run, delay, term)`), so the cost is scored one
regime at a time rather than pooled. Pooling would flatter the fit — the regimes
differ in magnitude by an order of magnitude, and a correlation across that
spread is mostly measuring which regime an entry came from. They are also
identified by *different* things, which is the point of separating them: the
running cost by the within-trial transitions, the delay cost by the transitions
before onset, and the terminal cost by the endpoint condition alone.

`S` is reported even when frozen, because the canonical rescaling divides it by
the fitted cost scale — so on a known-plant row its relative RMSE is exactly the
relative error in `tr(Qc[1])`, and its correlation is exactly ±1. That makes it
the cheapest readout of the cost's overall size, which is otherwise
canonicalized away.
"""
function compare(
    fit_sm, ref, idx; known_plant::Bool, free_gref::Bool=false, free_noise::Bool=true
)
    cl_f = closed_loop_or_nothing(fit_sm)
    cl_r = closed_loop_or_nothing(ref)
    reg(i) = i === nothing ? NOSCORE : score(fit_sm.Qc[i], ref.Qc[i]; sym=true)
    has_ref = free_gref && size(ref.Gref, 2) > 0
    return (
        Qc=reg(idx.run),
        Qdel=reg(idx.delay),
        Qterm=reg(idx.term),
        A=known_plant ? NOSCORE : score(fit_sm.A, ref.A),
        S=score(fit_sm.S, ref.S; sym=true),
        Gref=has_ref ? score(fit_sm.Gref, ref.Gref) : NOSCORE,
        #=
        The *state* block of the innovation only. `rescale_costate!` multiplies
        `Σ`'s costate rows and columns by the cost scale `c`, so a fit whose cost
        is a hundred times too small reports a `Σ` a hundred times too large —
        which is the cost-scale error again, wearing a different name, and `S`
        already reports that exactly. `Σ[1:n, 1:n]` is untouched by the
        transformation, so it is the part of the innovation that means the same
        thing in both models: the process noise on the plant.
        =#
        Sig=(
            free_noise ? score(_state_block(fit_sm.Σ), _state_block(ref.Σ); sym=true) :
            NOSCORE
        ),
        h=score(fit_sm.h, ref.h),
        cl=(cl_f === nothing || cl_r === nothing) ? NOSCORE : score(cl_f, cl_r),
    )
end

# ---------------------------------------------------------------------------
# Latent-coordinate gauge checks
# ---------------------------------------------------------------------------

"""
    gauge_maps(Cfit, Cref, n)

Estimate the map `x_ref = T*x_fit` from `Cfit ≈ Cref*T`, once with an
orthogonal Procrustes constraint and once as a full least-squares map. The
second is diagnostic: LQR coordinates admit a general invertible gauge, with
the costate transforming contragrediently, even though PCA/FA initialization
often reduces the practical ambiguity to a rotation.
"""
function gauge_maps(Cfit::AbstractMatrix, Cref::AbstractMatrix, n::Int)
    X, Y = Matrix(view(Cref, :, 1:n)), Matrix(view(Cfit, :, 1:n))
    F = svd(X'Y)
    proc = F.U * F.Vt
    linear = X \ Y
    return (procrustes=proc, linear=linear,
        nonorthogonality=norm(linear'linear - I) / sqrt(n))
end

function _gauge_compare(fit, ref, idx, T; known_plant::Bool, free_gref::Bool,
                        free_noise::Bool=true)
    n = plant_dim(fit)
    invT = inv(T)
    mapped_Q(i) = invT' * fit.Qc[i] * invT
    # Reapply the same trace convention after a non-orthogonal basis change.
    c = n / tr(mapped_Q(idx.run))
    reg(i) = i === nothing ? NOSCORE : score(c .* mapped_Q(i), ref.Qc[i]; sym=true)
    clf = closed_loop_or_nothing(fit)
    clr = closed_loop_or_nothing(ref)
    hfit = vcat(T * fit.h[1:n], c .* (invT' * fit.h[(n + 1):(2n)]))
    return (
        Qc=reg(idx.run),
        Qdel=reg(idx.delay),
        Qterm=reg(idx.term),
        A=known_plant ? NOSCORE : score(T*fit.A*invT, ref.A),
        S=score((T*fit.S*T') ./ c, ref.S; sym=true),
        Gref=(free_gref && size(ref.Gref, 2) > 0) ? score(T*fit.Gref, ref.Gref) : NOSCORE,
        Sig=free_noise ? score(T*_state_block(fit.Σ)*T', _state_block(ref.Σ); sym=true) : NOSCORE,
        h=score(hfit, ref.h),
        cl=(clf === nothing || clr === nothing) ? NOSCORE : score(T*clf*invT, clr),
    )
end

_gauge_noscores() = (
    Qc=NOSCORE, Qdel=NOSCORE, Qterm=NOSCORE, A=NOSCORE, S=NOSCORE,
    Gref=NOSCORE, Sig=NOSCORE, h=NOSCORE, cl=NOSCORE,
)


"""
    gauge_compare(fit, ref, idx, Cfit, Cref; ...) -> NamedTuple

Raw, Procrustes-aligned, and full-linear parameter recovery. All parameter
blocks are transformed consistently; in particular `Gref` is never rotated by
itself. `Ggram` scores `Gref'Gref`, which is invariant only to the orthogonal
gauge. If the full-linear score improves over Procrustes, the mismatch includes
scale/shear and is not "only a rotation."
"""
function gauge_compare(fit, ref, idx, Cfit, Cref; known_plant::Bool,
        free_gref::Bool=false, free_noise::Bool=true)
    maps = gauge_maps(Cfit, Cref, plant_dim(fit))
    hasref = free_gref && size(ref.Gref, 2) > 0
    sv = svdvals(maps.linear)
    linear_valid = !isempty(sv) && minimum(sv) > sqrt(eps(eltype(sv))) * maximum(sv)
    linear_scores = linear_valid ? _gauge_compare(
        fit, ref, idx, maps.linear;
        known_plant=known_plant, free_gref=free_gref, free_noise=free_noise,
    ) : _gauge_noscores()
    return (
        raw=compare(fit, ref, idx; known_plant=known_plant,
            free_gref=free_gref, free_noise=free_noise),
        procrustes=_gauge_compare(fit, ref, idx, maps.procrustes;
            known_plant=known_plant, free_gref=free_gref, free_noise=free_noise),
        linear=linear_scores,
        Ggram=hasref ? score(fit.Gref'fit.Gref, ref.Gref'ref.Gref; sym=true) : NOSCORE,
        maps=maps,
        C=(
            procrustes=score(Cfit[:, 1:plant_dim(fit)]*maps.procrustes',
                Cref[:, 1:plant_dim(fit)]),
            linear=linear_valid ? score(
                Cfit[:, 1:plant_dim(fit)]*inv(maps.linear),
                Cref[:, 1:plant_dim(fit)],
            ) : NOSCORE,
        ),
        linear_valid=linear_valid,
    )
end

# ---------------------------------------------------------------------------
# Discrete-state (γ) scoring
#
# The SLDS adds a question the single-system harness does not have: did the fit
# find the *epochs*? Three numbers, because they fail in different ways.
# ---------------------------------------------------------------------------

"""
    gamma_scores(γs, zs; K) -> NamedTuple

How well a fitted posterior over discrete states recovers the path that
generated the data.

- `acc` — balanced accuracy of the MAP state, averaged over the `K` true states
  rather than over timesteps. Plain accuracy is unreadable when the epochs are
  unbalanced: a model that calls every timestep "LQR" scores whatever fraction
  of the data is LQR, which on a delay-then-reach design is already most of it.
- `post` — the mean posterior mass the fit puts on the *true* state. This is the
  soft version, and it separates "confidently right" from "barely right" in a
  way a MAP accuracy cannot.
- `xent` — mean per-timestep cross-entropy `−log γ[z_t, t]`, in nats. Unbounded
  above, so it is the one that notices confident mistakes; `log K` is the score
  of a model that has learned nothing.

Label switching is not handled, and deliberately: the two discrete states of
this harness are structurally different (one free, one LQR) and are built in a
fixed order, so a permuted fit is a genuine failure and should read as one.
"""
function gamma_scores(γs, zs; K::Int=2)
    correct = zeros(Int, K)
    count_k = zeros(Int, K)
    postsum = 0.0
    xent = 0.0
    total = 0
    for (γ, z) in zip(γs, zs)
        for t in eachindex(z)
            k = z[t]
            count_k[k] += 1
            correct[k] += (argmax(view(γ, :, t)) == k)
            p = max(γ[k, t], 1e-12)
            postsum += p
            xent -= log(p)
            total += 1
        end
    end
    seen = count_k .> 0
    return (
        acc=_mean(correct[seen] ./ count_k[seen]),
        post=postsum / total,
        xent=xent / total,
    )
end

"""
    onset_error(γs, zs; lqr_state) -> (bias, mad)

Switch-time error for a single-switch (delay-then-reach) design, in timesteps:
the fitted onset is the first `t` at which the MAP state is `lqr_state`, and
this reports the mean signed error and the mean absolute error against the true
onset. `NaN` when a trial has no switch to find.

A signed bias is worth separating from the magnitude: a fit that is *late* on
every trial has mistaken part of the movement for the delay, which is a
different failure from one that is merely noisy about where the boundary is.
"""
function onset_error(γs, zs; lqr_state::Int=1)
    errs = Float64[]
    for (γ, z) in zip(γs, zs)
        t_true = findfirst(==(lqr_state), z)
        t_true === nothing && continue
        t_fit = findfirst(t -> argmax(view(γ, :, t)) == lqr_state, eachindex(z))
        push!(errs, t_fit === nothing ? length(z) - t_true : t_fit - t_true)
    end
    isempty(errs) && return (bias=NaN, mad=NaN)
    return (bias=_mean(errs), mad=_mean(abs.(errs)))
end
