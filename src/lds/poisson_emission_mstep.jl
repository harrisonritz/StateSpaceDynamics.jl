#=============================================================================
Poisson emission M-step (Newton)

    Solve: update_observation_model!(plds, tfs, y, sws_pool, w; uy)
    Reference: _update_observation_model_lbfgs! (same objective, LBFGS)

The Q-function of the Poisson emission,

    F(W) = Σₙ Σₜ wₜ [ exp(ηₙₜ + ρₙₜ) − yₙₜ ηₙₜ ] + ½ tr[(W−M₀) Λ (W−M₀)']

with `ηₙₜ = wₙ' zₜ`, `zₜ = [xₜ; 1; vₜ]` and `ρₙₜ = ½ cₙ' Pₜ cₙ`, **separates over
the rows of `W = [C d D]`**: neuron `n` shares nothing with neuron `m` — not the
data, not the prior (the MN penalty is `½ Σₙ (wₙ−mₙ)' Λ (wₙ−mₙ)`), not the
latents, which the E-step has already fixed. Each row is its own
`reg_dim = latent_dim + 1 + uy_dim` problem, and each is strictly convex: `η` is
linear in `wₙ`, `ρ` is a PSD quadratic form in `cₙ`, and `exp` of a convex
function is convex.

That is what this file exploits. The previous solver ran one LBFGS over all
`obs_dim · reg_dim` parameters at once, which needed 25–40 iterations — 30-odd
sweeps over every trial — per M-step. Newton on each row separately converges in
a handful of steps from the previous M-step's warm start, because it sees the
exact curvature of a small strictly convex problem instead of a limited-memory
approximation of a large one.

The per-row Hessian is

    Hₙ = Σₜ wₜ λₙₜ (gₙₜ gₙₜ' + P̃ₜ) + Λ,     gₙₜ = zₜ + [Pₜ cₙ; 0; 0],

the second term being the curvature of `ρ` itself (`∂²ρₙₜ/∂cₙ∂cₙ' = Pₜ`), and
`P̃ₜ` is `Pₜ` padded with zero rows/columns for the `d` and `D` blocks that `ρ`
does not depend on.

Everything below is organised so a trial's work is a handful of `gemm`s: the
linear predictor, the variance correction `ρ` (through the symmetric-pair
packing described in `PoissonBatchBuffers`), and `Pₜ cₙ` for every `(n, t)` at
once. Trials are chunked across the workspace pool exactly as the gradient
already was, and the per-row accumulators are reduced in a fixed chunk order so
the result does not depend on how the chunks are scheduled.
=============================================================================#

"""
    PoissonMStepBuffers{T<:Real}

Per-chunk scratch for the Newton emission M-step. One is built per task at the
top of `update_observation_model!` and reused across that solve's Newton
iterations; nothing here survives the call.

Sized for the widest emission and longest trial the chunk will see. `H` and
`grad` are the per-row accumulators (`reg_dim × reg_dim × obs_dim` and
`reg_dim × obs_dim`); `fval` accumulates each row's objective, which is what
lets the line search give every neuron its own step length.
"""
struct PoissonMStepBuffers{T<:Real}
    latent_dim::Int
    reg_dim::Int
    obs_dim::Int
    nsym::Int
    sym_i::Vector{Int}
    sym_j::Vector{Int}
    Zaug::Matrix{T}        # (reg_dim × tsteps)            [xₜ; 1; vₜ]
    Eta::Matrix{T}         # (obs_dim × tsteps)            ηₙₜ
    Lam::Matrix{T}         # (obs_dim × tsteps)            wₜ·exp(η + ρ)
    Wy::Matrix{T}          # (obs_dim × tsteps)            wₜ·yₙₜ
    Cpair::Matrix{T}       # (obs_dim × nsym)              C[:, i] .* C[:, j]
    Ppack::Matrix{T}       # (nsym × tsteps)               packed Pₜ (off-diag doubled)
    CPflat::Matrix{T}      # (obs_dim × latent_dim·tsteps) Pₜ cₙ for every (n, t)
    Spack::Matrix{T}       # (nsym × obs_dim)              Σₜ λₙₜ Pₜ, packed
    G::Matrix{T}           # (tsteps × reg_dim)            gₙₜ for the current row
    Gw::Matrix{T}          # (tsteps × reg_dim)            λₙₜ · gₙₜ
    H::Array{T,3}          # (reg_dim × reg_dim × obs_dim)
    grad::Matrix{T}        # (reg_dim × obs_dim)
    fval::Vector{T}        # (obs_dim,)
end

function PoissonMStepBuffers(
    ::Type{T}, latent_dim::Int, obs_dim::Int, uy_dim::Int, tsteps::Int; curvature::Bool=true
) where {T<:Real}
    reg_dim = latent_dim + 1 + uy_dim
    sym_i = Int[]
    sym_j = Int[]
    for j in 1:latent_dim, i in j:latent_dim
        push!(sym_i, i)
        push!(sym_j, j)
    end
    nsym = length(sym_i)
    # The line-search pass never touches the curvature buffers, and they are the
    # big ones — `CPflat` alone is obs_dim · latent_dim · tsteps.
    z3(a, b, c) = curvature ? zeros(T, a, b, c) : zeros(T, 0, 0, 0)
    z2(a, b) = curvature ? zeros(T, a, b) : zeros(T, 0, 0)
    return PoissonMStepBuffers{T}(
        latent_dim,
        reg_dim,
        obs_dim,
        nsym,
        sym_i,
        sym_j,
        zeros(T, reg_dim, tsteps),
        zeros(T, obs_dim, tsteps),
        zeros(T, obs_dim, tsteps),
        zeros(T, obs_dim, tsteps),
        zeros(T, obs_dim, nsym),
        zeros(T, nsym, tsteps),
        z2(obs_dim, latent_dim * tsteps),
        z2(nsym, obs_dim),
        z2(tsteps, reg_dim),
        z2(tsteps, reg_dim),
        z3(reg_dim, reg_dim, obs_dim),
        z2(reg_dim, obs_dim),
        zeros(T, obs_dim),
    )
end

"""Zero the accumulators a fresh pass over the trials writes into."""
function _reset_mstep_accumulators!(buf::PoissonMStepBuffers{T}, curvature::Bool) where {T}
    fill!(buf.fval, zero(T))
    if curvature
        fill!(buf.H, zero(T))
        fill!(buf.grad, zero(T))
    end
    return nothing
end

"""
    _poisson_mstep_trial!(buf, W, x, p_smooth, y, uy, weights, curvature)

Add one trial's contribution to `buf`: each row's objective always, and its
gradient and Hessian when `curvature` is true.

`W` is the stacked emission `[C d D]` (`obs_dim × reg_dim`), `x` the smoothed
means and `p_smooth` the smoothed covariances the E-step produced. Every
quantity that depends on more than one `(n, t)` pair is formed as a `gemm`; the
per-row loop that follows only does work that genuinely differs by row.
"""
function _poisson_mstep_trial!(
    buf::PoissonMStepBuffers{T},
    W::AbstractMatrix{T},
    x::AbstractMatrix{T},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
    uy::Union{Nothing,AbstractMatrix},
    weights::Union{Nothing,AbstractVector{T}},
    curvature::Bool,
    active::Union{Nothing,AbstractVector{Bool}}=nothing,
) where {T<:Real}
    latent_dim = buf.latent_dim
    reg_dim = buf.reg_dim
    nsym = buf.nsym
    obs_dim = size(y, 1)
    tsteps = size(y, 2)
    # A caller may leave `uy` off entirely; the `D` columns of `z` are then zero,
    # which the preallocated buffers already are.
    uy_dim = uy === nothing ? 0 : min(reg_dim - latent_dim - 1, size(uy, 1))
    C = view(W, 1:obs_dim, 1:latent_dim)

    # z_t = [x_t; 1; v_t]
    Z = view(buf.Zaug, 1:reg_dim, 1:tsteps)
    @inbounds for t in 1:tsteps
        for a in 1:latent_dim
            Z[a, t] = x[a, t]
        end
        Z[latent_dim + 1, t] = one(T)
        for a in 1:uy_dim
            Z[latent_dim + 1 + a, t] = uy[a, t]
        end
    end

    Eta = view(buf.Eta, 1:obs_dim, 1:tsteps)
    mul!(Eta, view(W, 1:obs_dim, 1:reg_dim), Z)

    # ρ[n, t] = ½ cₙ' Pₜ cₙ as one gemm over the nsym distinct entries of Pₜ.
    Cpair = view(buf.Cpair, 1:obs_dim, 1:nsym)
    @inbounds for p in 1:nsym
        Ci = view(C, :, buf.sym_i[p])
        Cj = view(C, :, buf.sym_j[p])
        col = view(Cpair, :, p)
        @simd for n in 1:obs_dim
            col[n] = Ci[n] * Cj[n]
        end
    end
    Ppack = view(buf.Ppack, 1:nsym, 1:tsteps)
    @inbounds for t in 1:tsteps, p in 1:nsym
        i = buf.sym_i[p]
        j = buf.sym_j[p]
        Ppack[p, t] = i == j ? p_smooth[i, j, t] : 2 * p_smooth[i, j, t]
    end

    Lam = view(buf.Lam, 1:obs_dim, 1:tsteps)
    mul!(Lam, Cpair, Ppack, T(0.5), zero(T))          # Lam := ρ
    @inbounds for t in 1:tsteps
        wt = weights === nothing ? one(T) : weights[t]
        λcol = view(Lam, :, t)
        ηcol = view(Eta, :, t)
        @simd for n in 1:obs_dim
            λcol[n] = wt * exp(λcol[n] + ηcol[n])
        end
    end

    Wy = view(buf.Wy, 1:obs_dim, 1:tsteps)
    @inbounds for t in 1:tsteps
        wt = weights === nothing ? one(T) : weights[t]
        ycol = view(Wy, :, t)
        @simd for n in 1:obs_dim
            ycol[n] = wt * y[n, t]
        end
    end

    fval = buf.fval
    @inbounds for t in 1:tsteps
        λcol = view(Lam, :, t)
        ηcol = view(Eta, :, t)
        ycol = view(Wy, :, t)
        @simd for n in 1:obs_dim
            fval[n] += λcol[n] - ycol[n] * ηcol[n]
        end
    end

    curvature || return nothing

    #=
    Pₜ cₙ for every (n, t) in one gemm: `p_smooth` is contiguous, so its
    (latent_dim, latent_dim·tsteps) reshape is a plain matrix and
    `CPflat[n, (t-1)·latent_dim + a] = (Pₜ cₙ)[a]` (Pₜ is symmetric).
    =#
    ncols = latent_dim * tsteps
    Pflat = view(reshape(p_smooth, latent_dim, :), :, 1:ncols)
    CPflat = view(buf.CPflat, 1:obs_dim, 1:ncols)
    mul!(CPflat, C, Pflat)

    # Spack[:, n] = Σₜ λₙₜ Pₜ, packed like Ppack (off-diagonals doubled).
    Spack = view(buf.Spack, 1:nsym, 1:obs_dim)
    mul!(Spack, Ppack, transpose(Lam))

    G = view(buf.G, 1:tsteps, 1:reg_dim)
    Gw = view(buf.Gw, 1:tsteps, 1:reg_dim)
    @inbounds for t in 1:tsteps
        G[t, latent_dim + 1] = one(T)
        for a in 1:uy_dim
            G[t, latent_dim + 1 + a] = uy[a, t]
        end
    end

    @inbounds for n in 1:obs_dim
        # A row whose Newton decrement has already fallen below tolerance is
        # frozen for the rest of the solve; the per-row work is the expensive
        # part of this loop, so skipping it is what keeps a handful of stubborn
        # rows (a unit some session never fired, whose optimum runs to d = −∞)
        # from charging every other row for their iterations.
        active === nothing || active[n] || continue
        # gₙₜ = zₜ + [Pₜ cₙ; 0; 0] — only the latent block differs from zₜ.
        for a in 1:latent_dim
            col = view(G, :, a)
            @simd for t in 1:tsteps
                col[t] = x[a, t] + CPflat[n, (t - 1) * latent_dim + a]
            end
        end
        λrow = view(Lam, n, 1:tsteps)
        for a in 1:reg_dim
            gcol = view(G, :, a)
            wcol = view(Gw, :, a)
            @simd for t in 1:tsteps
                wcol[t] = gcol[t] * λrow[t]
            end
        end

        Hn = view(buf.H, :, :, n)
        mul!(Hn, transpose(Gw), G, one(T), one(T))
        # Σₜ λₙₜ P̃ₜ — the curvature of ρ, in the latent block only.
        for p in 1:nsym
            i = buf.sym_i[p]
            j = buf.sym_j[p]
            v = Spack[p, n]
            if i == j
                Hn[i, i] += v
            else
                half = v / 2
                Hn[i, j] += half
                Hn[j, i] += half
            end
        end

        gn = view(buf.grad, :, n)
        mul!(gn, transpose(G), λrow, one(T), one(T))
        mul!(gn, Z, view(Wy, n, 1:tsteps), -one(T), one(T))
    end

    return nothing
end

"""
    _poisson_mstep_pass!(bufs, chunks, W, tfs, y, uy, w, curvature)

One pass over every trial, chunked across `bufs`. Returns nothing; the
per-chunk accumulators are reduced by the caller in chunk order.
"""
function _poisson_mstep_pass!(
    bufs::Vector{PoissonMStepBuffers{T}},
    chunks::Vector{<:AbstractVector{Int}},
    W::AbstractMatrix{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}},
    curvature::Bool,
    active::Union{Nothing,AbstractVector{Bool}}=nothing,
) where {T<:Real}
    tforeach(eachindex(chunks)) do task_idx
        buf = bufs[task_idx]
        _reset_mstep_accumulators!(buf, curvature)
        for k in chunks[task_idx]
            fs = tfs[k]
            uy_k = uy === nothing ? nothing : uy[k]
            w_k = w === nothing ? nothing : w[k]
            _poisson_mstep_trial!(
                buf, W, fs.x_smooth, fs.p_smooth, y[k], uy_k, w_k, curvature, active
            )
        end
    end
    return nothing
end

"""Sum the chunks' row objectives into `out`, in chunk order."""
function _reduce_fval!(
    out::AbstractVector{T}, bufs::Vector{PoissonMStepBuffers{T}}
) where {T<:Real}
    fill!(out, zero(T))
    for buf in bufs
        @inbounds @simd for n in eachindex(out)
            out[n] += buf.fval[n]
        end
    end
    return out
end

"""
    _poisson_mstep_prior!(fval, grad, H, W, active, prior)

Fold the matrix-normal penalty on `[C d D]` into the row objectives, gradients
and Hessians: `½ (wₙ−mₙ)' Λ (wₙ−mₙ)`, `Λ (wₙ−mₙ)` and `Λ`. `grad`/`H` are
skipped when `nothing`, which is what the line-search pass wants; `active`
restricts the gradient and Hessian terms to the rows still being solved, whose
accumulators are the only ones a curvature pass filled.

The objective term is added for every row regardless, since the line search
compares whole-row objectives.
"""
function _poisson_mstep_prior!(
    fval::AbstractVector{T},
    grad::Union{Nothing,AbstractMatrix{T}},
    H::Union{Nothing,AbstractArray{T,3}},
    W::AbstractMatrix{T},
    active::Union{Nothing,AbstractVector{Bool}},
    prior,
) where {T<:Real}
    prior === nothing && return nothing
    Λ = prior.Λ
    M₀ = prior.M₀
    obs_dim, reg_dim = size(W)
    dev = Vector{T}(undef, reg_dim)
    Λdev = Vector{T}(undef, reg_dim)
    @inbounds for n in 1:obs_dim
        for a in 1:reg_dim
            dev[a] = W[n, a] - M₀[n, a]
        end
        mul!(Λdev, Λ, dev)
        fval[n] += T(0.5) * dot(dev, Λdev)
        active === nothing || active[n] || continue
        if grad !== nothing
            gn = view(grad, :, n)
            for a in 1:reg_dim
                gn[a] += Λdev[a]
            end
        end
        if H !== nothing
            Hn = view(H, :, :, n)
            for b in 1:reg_dim, a in 1:reg_dim
                Hn[a, b] += Λ[a, b]
            end
        end
    end
    return nothing
end

"""
    _poisson_newton_direction!(Δ, decrement, H, grad, active) -> decrement

Per-row Newton direction `Δₙ = −Hₙ⁻¹ gₙ`, in place.

`Hₙ` is positive definite wherever the row is identified. A neuron that some
session never fired — the padded rows an `union` unit alignment creates — has
`λₙₜ → 0`, and its Hessian degenerates as its optimum runs off to `dₙ = −∞`.
Rather than fail the whole M-step for one such row, its solve is retried with a
Levenberg ridge; the row then still descends (Newton on `exp` moves by −1 per
step), which is the same non-convergence the LBFGS solver expressed by running
out of iterations, only bounded. A row that cannot be factorised even then gets
a zero direction and stops moving.

Writes each row's Newton decrement `gₙ' Hₙ⁻¹ gₙ` into `decrement` — the natural
convergence measure, being twice the objective decrease a full Newton step
predicts. Rows already marked inactive are skipped and get a zero direction.
"""
function _poisson_newton_direction!(
    Δ::AbstractMatrix{T},
    decrement::AbstractVector{T},
    H::AbstractArray{T,3},
    grad::AbstractMatrix{T},
    active::AbstractVector{Bool},
) where {T<:Real}
    reg_dim, obs_dim = size(grad)
    work = Matrix{T}(undef, reg_dim, reg_dim)
    @inbounds for n in 1:obs_dim
        decrement[n] = zero(T)
        if !active[n]
            fill!(view(Δ, :, n), zero(T))
            continue
        end
        Hn = view(H, :, :, n)
        gn = view(grad, :, n)
        δn = view(Δ, :, n)
        copyto!(work, Hn)
        # `Gw' G` is symmetric up to rounding; make that exact for the factorisation.
        for b in 1:reg_dim, a in (b + 1):reg_dim
            m = T(0.5) * (work[a, b] + work[b, a])
            work[a, b] = m
            work[b, a] = m
        end
        F = cholesky!(Symmetric(work); check=false)
        if !issuccess(F)
            copyto!(work, Hn)
            for b in 1:reg_dim, a in (b + 1):reg_dim
                m = T(0.5) * (work[a, b] + work[b, a])
                work[a, b] = m
                work[b, a] = m
            end
            scale = zero(T)
            for a in 1:reg_dim
                scale = max(scale, abs(work[a, a]))
            end
            ridge = max(T(1e-10), T(1e-8) * scale)
            for a in 1:reg_dim
                work[a, a] += ridge
            end
            F = cholesky!(Symmetric(work); check=false)
            if !issuccess(F)
                fill!(δn, zero(T))
                continue
            end
        end
        copyto!(δn, gn)
        ldiv!(F, δn)
        decrement[n] = dot(gn, δn)
        @simd for a in 1:reg_dim
            δn[a] = -δn[a]
        end
    end
    return decrement
end

"""
    update_observation_model!(plds, tfs, y, sws_pool, w; uy=nothing)

Update the Poisson emission `[C d D]` by row-wise Newton on the exact Q-function
(see the header of this file for why the rows separate and what the curvature
is). `uy` is the per-trial vector of observation-input matrices, `w` the
per-trial timestep weights the SLDS path supplies; both may be `nothing`.

Each Newton step takes one curvature pass over the trials plus one line-search
pass per trial step length tried, both chunked across `sws_pool`. Every row gets
its own backtracked step, so a neuron whose problem is badly scaled cannot hold
back the rest, and a row stops being solved for once a full Newton step promises
it less than `tol · |Q|` — the same units the caller's ELBO tolerance is in.
"""
function update_observation_model!(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing;
    uy::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    max_iter::Int=50,
    tol::Real=1e-12,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    plds.fit_bool[5] || return nothing

    obs_dim = plds.obs_dim
    latent_dim = plds.latent_dim
    uy_dim = plds.uy_dim
    reg_dim = latent_dim + 1 + uy_dim
    ntrials = length(tfs)
    ntrials == 0 && return nothing
    all(size(yk, 1) == obs_dim for yk in y) || throw(
        DimensionMismatch(
            "Poisson emission M-step: every trial must have obs_dim = $(obs_dim) rows " *
            "(a stitched fit passes each cell its own trials and its own model)",
        ),
    )
    tsteps_max = maximum(size(yk, 2) for yk in y)
    prior = plds.obs_model.CD_prior

    # W = [C d D], the layout every kernel here and the LBFGS reference share.
    W = Matrix{T}(undef, obs_dim, reg_dim)
    @views W[:, 1:latent_dim] .= plds.obs_model.C
    @views W[:, latent_dim + 1] .= plds.obs_model.d
    uy_dim > 0 && (@views W[:, (latent_dim + 2):reg_dim] .= plds.obs_model.D)

    ntasks = max(1, min(ntrials, length(sws_pool)))
    chunk_size = max(1, cld(ntrials, ntasks))
    chunks = collect(partition(1:ntrials, chunk_size))
    ntasks = length(chunks)

    curv_bufs = [
        PoissonMStepBuffers(T, latent_dim, obs_dim, uy_dim, tsteps_max) for _ in 1:ntasks
    ]
    ls_bufs = [
        PoissonMStepBuffers(T, latent_dim, obs_dim, uy_dim, tsteps_max; curvature=false) for
        _ in 1:ntasks
    ]

    Δ = zeros(T, reg_dim, obs_dim)
    W_trial = similar(W)
    fval = zeros(T, obs_dim)
    fval_trial = zeros(T, obs_dim)
    decrement = zeros(T, obs_dim)
    step = ones(T, obs_dim)
    solving = trues(obs_dim)
    stepping = falses(obs_dim)
    slope = zeros(T, obs_dim)

    H = curv_bufs[1].H
    grad = curv_bufs[1].grad

    for _ in 1:max_iter
        _poisson_mstep_pass!(curv_bufs, chunks, W, tfs, y, uy, w, true, solving)
        # Reduce onto chunk 1's accumulators, in chunk order, so the result does
        # not depend on how the chunks were scheduled.
        for c in 2:ntasks
            Hc = curv_bufs[c].H
            gc = curv_bufs[c].grad
            @inbounds @simd for i in eachindex(H)
                H[i] += Hc[i]
            end
            @inbounds @simd for i in eachindex(grad)
                grad[i] += gc[i]
            end
        end
        _reduce_fval!(fval, curv_bufs)
        _poisson_mstep_prior!(fval, grad, H, W, solving, prior)

        _poisson_newton_direction!(Δ, decrement, H, grad, solving)

        #=
        Stop a row once a full Newton step promises less than `threshold` — the
        decrement is twice the predicted improvement, so this is a bound on what
        the row still has to gain, in the same units as the ELBO the caller
        checks for convergence.
        =#
        threshold = max(T(tol), T(tol) * abs(sum(fval)))
        any_active = false
        @inbounds for n in 1:obs_dim
            solving[n] || continue
            if decrement[n] <= threshold
                solving[n] = false
            else
                any_active = true
            end
        end
        any_active || break

        #=
        Armijo, per row: each neuron keeps halving its own step until its own
        objective accepts it. Rows settle at α = 1 within a step or two, so this
        is normally one extra (cheap, curvature-free) pass over the trials.
        =#
        @inbounds for n in 1:obs_dim
            step[n] = zero(T)
            stepping[n] = false
            solving[n] || continue
            slope[n] = dot(view(grad, :, n), view(Δ, :, n))
            if slope[n] < zero(T)
                step[n] = one(T)
                stepping[n] = true
            else
                # A non-descent direction means the row is at (or numerically
                # past) its optimum; nothing left to do for it.
                solving[n] = false
            end
        end

        for _ in 1:25
            any(stepping) || break
            @inbounds for n in 1:obs_dim, a in 1:reg_dim
                W_trial[n, a] = W[n, a] + step[n] * Δ[a, n]
            end
            _poisson_mstep_pass!(ls_bufs, chunks, W_trial, tfs, y, uy, w, false)
            _reduce_fval!(fval_trial, ls_bufs)
            _poisson_mstep_prior!(fval_trial, nothing, nothing, W_trial, nothing, prior)
            still_stepping = false
            @inbounds for n in 1:obs_dim
                stepping[n] || continue
                if isfinite(fval_trial[n]) &&
                    fval_trial[n] <= fval[n] + T(1e-4) * step[n] * slope[n]
                    stepping[n] = false
                else
                    step[n] *= T(0.5)
                    still_stepping = true
                end
            end
            still_stepping || break
        end

        # A row that exhausted its backtracking takes no step rather than one
        # its own objective rejected, and stops being solved for.
        @inbounds for n in 1:obs_dim
            if stepping[n]
                step[n] = zero(T)
                solving[n] = false
            end
        end
        @inbounds for n in 1:obs_dim, a in 1:reg_dim
            W[n, a] += step[n] * Δ[a, n]
        end
    end

    @views plds.obs_model.C .= W[:, 1:latent_dim]
    @views plds.obs_model.d .= W[:, latent_dim + 1]
    uy_dim > 0 && (@views plds.obs_model.D .= W[:, (latent_dim + 2):reg_dim])
    return nothing
end
