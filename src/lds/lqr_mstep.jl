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

`term_zz` / `term_n` hold `Σ_n E[[z_T;1;u_T][z_T;1;u_T]ᵀ]` and the trial count,
the statistics of the terminal factor — the input block carries its reference.

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
    const term_zz::Matrix{T}
    term_n::T
    # Mixed-coordinate blocks (derived; see `_fill_mixed_blocks!`).
    const Zw::Vector{Matrix{T}}
    const Xv::Vector{Matrix{T}}
    const Yv::Matrix{T}
    const Omega::Matrix{T}
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
        zeros(T, d + 1 + m, d + 1 + m),
        zero(T),
        [zeros(T, reg, reg) for _ in 1:K],
        [zeros(T, d, reg) for _ in 1:K],
        zeros(T, d, d),
        zeros(T, d + 1 + m, d + 1 + m),
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
    trials::AbstractVector{Int}=Base.OneTo(length(tfs)),
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
    fill!(hs.term_zz, zero(T))
    hs.term_n = zero(T)

    for trial in trials
        fs = tfs[trial]
        x = fs.x_smooth::Matrix{T}
        p_smooth = fs.p_smooth::Array{T,3}
        p_tt1 = fs.p_smooth_tt1::Array{T,3}
        T_n = size(x, 2)
        ux = data.ux[trial]

        for (k, t0, t1) in _regime_runs(sm, T_n)
            zz = hs.zz[k]
            zy = hs.zy[k]
            yy = hs.yy[k]
            len = t1 - t0 + 1
            hs.nk[k] += T(len)

            x_prev = tview(x, :, t0:t1)
            x_next = tview(x, :, (t0 + 1):(t1 + 1))

            # Mean parts (BLAS-3 over the run).
            BLAS.syrk!('U', 'N', one(T), x_prev, one(T), tview(zz, 1:d, 1:d))
            mul!(view(zy, 1:d, :), x_prev, transpose(x_next), one(T), one(T))
            BLAS.syrk!('U', 'N', one(T), x_next, one(T), yy)
            for t in t0:t1, i in 1:d
                zz[i, d + 1] += x[i, t]
                zy[d + 1, i] += x[i, t + 1]
            end
            zz[d + 1, d + 1] += T(len)

            # Covariance parts.
            @views for t in t0:t1
                zz[1:d, 1:d] .+= p_smooth[:, :, t]
                yy .+= p_smooth[:, :, t + 1]
                # Cov(z_t, z_{t+1}) = Cov(z_{t+1}, z_t)ᵀ = p_smooth_tt1[:,:,t+1]ᵀ
                zy[1:d, :] .+= adjoint(p_tt1[:, :, t + 1])
            end

            if m > 0
                u_run = tview(ux, :, t0:t1)
                mul!(view(zz, 1:d, (d + 2):reg), x_prev, transpose(u_run), one(T), one(T))
                mul!(view(zy, (d + 2):reg, :), u_run, transpose(x_next), one(T), one(T))
                BLAS.syrk!(
                    'U', 'N', one(T), u_run, one(T), tview(zz, (d + 2):reg, (d + 2):reg)
                )
                for t in t0:t1, j in 1:m
                    zz[d + 1, d + 1 + j] += ux[j, t]
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
            xT = tview(x, :, T_n)
            BLAS.ger!(one(T), xT, xT, tview(hs.term_zz, 1:d, 1:d))
            @views hs.term_zz[1:d, 1:d] .+= p_smooth[:, :, T_n]
            for i in 1:d
                hs.term_zz[i, d + 1] += x[i, T_n]
            end
            if m > 0
                uT = tview(ux, :, T_n)
                @views mul!(
                    hs.term_zz[1:d, (d + 2):(d + 1 + m)], xT, transpose(uT), one(T), one(T)
                )
                BLAS.ger!(
                    one(T),
                    uT,
                    uT,
                    tview(hs.term_zz, (d + 2):(d + 1 + m), (d + 2):(d + 1 + m)),
                )
                for j in 1:m
                    hs.term_zz[d + 1, d + 1 + j] += ux[j, T_n]
                end
            end
            hs.term_n += one(T)
        end
    end

    return _finalize_lqr_stats!(hs, sm, d, m, reg, K)
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
        @views hs.term_zz[d + 1, 1:d] .= hs.term_zz[1:d, d + 1]
        hs.term_zz[d + 1, d + 1] = hs.term_n
        if m > 0
            ur = (d + 2):(d + 1 + m)
            @views hs.term_zz[ur, 1:d] .= transpose(hs.term_zz[1:d, ur])
            @views hs.term_zz[ur, d + 1] .= hs.term_zz[d + 1, ur]
            LinearAlgebra.copytri!(tview(hs.term_zz, ur, ur), 'U')
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
    weights::AbstractVector{<:AbstractVector{T}},
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
    fill!(hs.term_zz, zero(T))
    hs.term_n = zero(T)

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

        #=
        Everything handed to `ger!` is hoisted through `tview` with a concrete
        annotation rather than `@views`: BLAS wrappers have no fallback method,
        so a view whose element type JET cannot pin becomes a "no matching
        method" report on every union-split branch.
        =#
        for t in 1:(T_n - 1)
            wt = w[t + 1]::T                  # the factor coupling (z_t, z_{t+1})
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
            if !iszero(wT)
                xT = tview(x, :, T_n)
                BLAS.ger!(wT, xT, xT, tview(hs.term_zz, 1:d, 1:d))
                @views hs.term_zz[1:d, 1:d] .+= wT .* p_smooth[:, :, T_n]
                for i in 1:d
                    hs.term_zz[i, d + 1] += wT * x[i, T_n]
                end
                if m > 0
                    uT = tview(ux, :, T_n)
                    BLAS.ger!(wT, xT, uT, tview(hs.term_zz, 1:d, (d + 2):(d + 1 + m)))
                    BLAS.ger!(
                        wT,
                        uT,
                        uT,
                        tview(hs.term_zz, (d + 2):(d + 1 + m), (d + 2):(d + 1 + m)),
                    )
                    for j in 1:m
                        hs.term_zz[d + 1, d + 1 + j] += wT * uT[j]
                    end
                end
                hs.term_n += wT
            end
        end
    end

    return _finalize_lqr_stats!(hs, sm, d, m, reg, K)
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
    rearrangement below collapses to three copies.
    =#
    if _is_free(sm)
        copyto!(hs.Zw[1], hs.zz[1])
        copyto!(hs.Xv[1], transpose(hs.zy[1]))
        copyto!(hs.Yv, hs.yy[1])
        copyto!(hs.Omega, hs.term_zz)
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

    copyto!(hs.Omega, hs.term_zz)
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

`S` and `Qc` store square factors `L`, with the actual matrices `L Lᵀ`.
This keeps every optimizer iterate positive semidefinite, including for grouped
and switching fits. Initialize free blocks positive definite to avoid the
zero-gradient stationary point of a zero factor.

A frozen block (see [`LQRFitFlags`](@ref)) has width zero and is neither
packed nor updated, so freezing shrinks the problem rather than projecting its
solution. `Qc`'s copy spans all `K` regimes contiguously.
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
    p.w[_LQR_BLOCK_Q] == 0 && return 1:0
    nn = p.n * p.n
    off = p.base[_LQR_BLOCK_Q] + (v - 1) * p.w[_LQR_BLOCK_Q] + (k - 1) * nn
    return (off + 1):(off + nn)
end

function _LQRPack(sm::LQRStateModel)
    return _LQRPack(sm, sm.fit_flags, ntuple(_ -> 1, _LQR_BLOCK_N))
end

function _LQRPack(sm::LQRStateModel, f::LQRFitFlags)
    return _LQRPack(sm, f, ntuple(_ -> 1, _LQR_BLOCK_N))
end

#=
Taking the flags explicitly rather than off the model is what lets a caller
freeze part of the block for one pass; taking `nv` is what lets each block carry
its own number of copies.
=#
function _LQRPack(sm::LQRStateModel, f::LQRFitFlags, nv::NTuple{7,Int})
    n = _plant_dim(sm)
    d = 2n
    m = size(sm.Bu, 2)
    K = _nregimes(sm)
    #= `Gref` is the one block that can be free in part: `Gref_cols` names the
    input columns the reference is a function of, and the rest are packed no
    more than a frozen block is. =#
    gcols = _gref_cols(f, m)
    bcols = _bu_cols(f, m)
    brows = _bu_rows(f, d)
    w = (
        f.A ? n * n : 0,
        f.S ? n * n : 0,
        f.Qc ? K * n * n : 0,
        f.h ? d : 0,
        length(brows) * length(bcols),
        n * length(gcols),
        (sm.terminal && f.terminal) ? n : 0,
    )
    bases = zeros(Int, _LQR_BLOCK_N)
    pos = 0
    for b in 1:_LQR_BLOCK_N
        bases[b] = pos
        pos += nv[b] * w[b]
    end
    return _LQRPack(n, d, m, K, nv, w, ntuple(b -> bases[b], _LQR_BLOCK_N), pos, gcols, bcols, brows)
end

"""
    _LQRUnit{T,HS}

One pooled group of trials in the structural M-step: its aggregated statistics,
which copy of each structural block it uses (`v`, indexed by the `_LQR_BLOCK_*`
ordinals), and which noise version (`q`).

Cells agreeing on every version are pooled into one unit before the objective is
built. That is exact, not an approximation: with the same `Θ` the residual
scatter `R = Σ_c (Y_c − Θ X_cᵀ − X_c Θᵀ + Θ Z_c Θᵀ)` is linear in the
statistics, so summing them first gives the same `R`. An ungrouped fit is one
unit, which is why there is a single code path.
"""
struct _LQRUnit{T<:Real,HS}
    hs::HS
    v::NTuple{7,Int}
    q::Int
    n::T
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

`Theta` and `Psi` are per **unit** rather than per version, since a unit's
assembled block mixes copies that no longer move together. The scratch below them
is preallocated once and reused across objective evaluations: at a large plant
dimension L-BFGS makes many evaluations and the `d × d` temporaries dominate.
"""
struct _LQRMStepCtx{T<:Real,HS,SM}
    pack::_LQRPack
    nq::Int
    units::Vector{_LQRUnit{T,HS}}
    sms::Vector{SM}
    owners::Vector{Vector{Vector{Int}}}  # [block][copy] -> models using it
    q_of::Vector{Int}                    # model -> noise version
    profile::Bool
    N_A::Vector{T}                   # transitions per `A` copy (Jacobian weight)
    N_q::Vector{T}                   # transitions per noise version
    Nf_q::Vector{T}                  # terminal factors per noise version
    kf::Int
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
    Psi::Vector{Matrix{T}}            # [unit]
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
end

"""
    _lqr_units(sufs, slots, q_slots) -> Vector{_LQRUnit}

Pool the cells into units. `slots[b][c]` is the copy of block `b` that cell `c`
uses; cells agreeing on every block and on the noise version become one unit.
"""
function _lqr_units(
    sufs::AbstractVector, slots::NTuple{7,Vector{Int}}, q_slots::AbstractVector{Int}
)
    T = eltype(first(sufs).nk)
    keys = NTuple{8,Int}[]
    members = Vector{Int}[]
    for c in eachindex(sufs)
        key = (ntuple(b -> slots[b][c], _LQR_BLOCK_N)..., q_slots[c])
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
                hs, ntuple(b -> key[b], _LQR_BLOCK_N), key[8], sum(hs.nk)
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
        out.Omega .+= s.Omega
        out.term_n += s.term_n
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
    sm1 = sms[1]
    T = eltype(sm1.Σ)
    f = flags === nothing ? sm1.fit_flags : flags
    #=
    A frozen block is never shared, whatever the tie asks for: freezing means
    "keep your own value", so each cell reads back its own rather than the first
    one on some version it was grouped into. Under `depends_on` this is a no-op —
    cells on a version alias the same array — and with one cell there is nothing
    to distinguish either way.
    =#
    probe = _LQRPack(sm1, f, ntuple(_ -> 1, _LQR_BLOCK_N))
    ncell = length(slots[1])
    #=
    Bound to a fresh name rather than back onto `slots`: reassigning an argument
    the closure above also reads boxes it, and the read becomes one JET cannot
    prove defined.
    =#
    eff = ntuple(b -> probe.w[b] == 0 ? collect(1:ncell) : slots[b], _LQR_BLOCK_N)
    units = _lqr_units(sufs, eff, q_slots)
    nv = ntuple(b -> maximum(eff[b]), _LQR_BLOCK_N)
    pack = _LQRPack(sm1, f, nv)
    n, d, m, K = pack.n, pack.d, pack.m, pack.K
    reg = d + 1 + m
    nq = maximum(q_slots)

    #=
    The Jacobian term `−N_a log|det A_a|` is weighted by the transitions the
    models sharing that `A` actually contribute, which under a partial tie is no
    longer the same grouping as any other block.
    =#
    N_A = zeros(T, nv[_LQR_BLOCK_A])
    N_q = zeros(T, nq)
    Nf_q = zeros(T, nq)
    for u in units
        N_A[u.v[_LQR_BLOCK_A]] += u.n
        N_q[u.q] += u.n
        sm1.terminal && (Nf_q[u.q] += u.hs.term_n)
    end
    kf = (sm1.terminal && !isempty(sm1.schedule)) ? sm1.schedule[end] : 1

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
        if (profile || !sm1.terminal)
            zeros(T, n, n)
        else
            Matrix(inv(PDMat(Symmetrize!(Matrix{T}(noise_sm[s].Σf)))))
        end for s in 1:nq
    ]

    U = length(units)
    return _LQRMStepCtx{T,eltype(sufs),typeof(sm1)}(
        pack,
        nq,
        units,
        collect(sms),
        owners,
        collect(q_slots),
        profile,
        N_A,
        N_q,
        Nf_q,
        kf,
        Sinv,
        Sfinv,
        [Matrix{T}(undef, n, n) for _ in 1:nv[_LQR_BLOCK_A]],
        [Matrix{T}(undef, n, n) for _ in 1:nv[_LQR_BLOCK_S]],
        [[Matrix{T}(undef, n, n) for _ in 1:K] for _ in 1:nv[_LQR_BLOCK_Q]],
        [Vector{T}(undef, d) for _ in 1:nv[_LQR_BLOCK_H]],
        [Matrix{T}(undef, d, m) for _ in 1:nv[_LQR_BLOCK_B]],
        [Matrix{T}(undef, n, m) for _ in 1:nv[_LQR_BLOCK_G]],
        [Vector{T}(undef, n) for _ in 1:nv[_LQR_BLOCK_F]],
        [[zeros(T, d, reg) for _ in 1:K] for _ in 1:U],
        [zeros(T, n, reg) for _ in 1:U],
        [Matrix{T}(undef, d, d) for _ in 1:nq],
        [Matrix{T}(undef, n, n) for _ in 1:nq],
        [zeros(T, n, n) for _ in 1:nv[_LQR_BLOCK_A]],
        [zeros(T, n, n) for _ in 1:nv[_LQR_BLOCK_S]],
        [[zeros(T, n, n) for _ in 1:K] for _ in 1:nv[_LQR_BLOCK_Q]],
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
    )
end

# Ungrouped convenience: one unit, one structural version, one noise version.
function _LQRMStepCtx(hs, sm::LQRStateModel, profile::Bool)
    return _LQRMStepCtx([hs], [sm], [1], [1], profile)
end

"""
    _lqr_nparams(ctx) -> Int

Length of the flat parameter vector.
"""
@inline _lqr_nparams(ctx::_LQRMStepCtx) = ctx.pack.np

function _lqr_pack_psd!(θ, r, matrix, name)
    # Symmetry alone describes a Hamiltonian system, not a convex LQR problem.
    # Keep old indefinite models readable, but refuse to fit them as LQR.
    E = eigen(Symmetric(Matrix(matrix)))
    tolerance = 100 * eps(eltype(matrix)) * max(one(eltype(matrix)), maximum(abs, E.values))
    minimum(E.values) >= -tolerance || throw(
        ArgumentError(
            "$name must be positive semidefinite to fit an LQR model; " *
            "restart from a valid initialization (minimum eigenvalue $(minimum(E.values)))",
        ),
    )
    if !isempty(r)
        factor = E.vectors * Diagonal(sqrt.(max.(E.values, zero(eltype(matrix)))))
        copyto!(view(θ, r), vec(factor))
    end
    return θ
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
    for v in 1:(p.nv[_LQR_BLOCK_Q]), k in 1:(p.K)
        r = _lqr_blk_q(p, v, k)
        _lqr_pack_psd!(θ, r, ctx.sms[first(o[_LQR_BLOCK_Q][v])].Qc[k], "Qc[$k]")
    end
    for v in 1:(p.nv[_LQR_BLOCK_H])
        r = _lqr_blk(p, _LQR_BLOCK_H, v)
        isempty(r) || copyto!(view(θ, r), ctx.sms[first(o[_LQR_BLOCK_H][v])].h)
    end
    for v in 1:(p.nv[_LQR_BLOCK_B])
        r = _lqr_blk(p, _LQR_BLOCK_B, v)
        isempty(r) || copyto!(view(θ, r), vec(view(ctx.sms[first(o[_LQR_BLOCK_B][v])].Bu, p.brows, p.bcols)))
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
            factor = reshape(view(θ, r), n, n)
            mul!(ctx.S[v], factor, transpose(factor))
        end
    end
    for v in 1:(p.nv[_LQR_BLOCK_Q]), k in 1:(p.K)
        r = _lqr_blk_q(p, v, k)
        if isempty(r)
            copyto!(ctx.Qc[v][k], ctx.sms[first(o[_LQR_BLOCK_Q][v])].Qc[k])
        else
            factor = reshape(view(θ, r), n, n)
            mul!(ctx.Qc[v][k], factor, transpose(factor))
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
                view(ctx.Bu[v], p.brows, p.bcols), reshape(view(θ, r), length(p.brows), length(p.bcols))
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

function _lqr_assemble!(ctx::_LQRMStepCtx{T}) where {T<:Real}
    p = ctx.pack
    n, d, m = p.n, p.d, p.m
    xr, lr = 1:n, (n + 1):d
    terminal = ctx.sms[1].terminal
    for (ui, u) in enumerate(ctx.units)
        A = ctx.A[u.v[_LQR_BLOCK_A]]
        S = ctx.S[u.v[_LQR_BLOCK_S]]
        Qs = ctx.Qc[u.v[_LQR_BLOCK_Q]]
        hv = ctx.h[u.v[_LQR_BLOCK_H]]
        for k in 1:(p.K)
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
                    mul!(Bcol[lr, :], Qs[k], ctx.Gref[u.v[_LQR_BLOCK_G]], -one(T), one(T))
                end
            end
        end
        if terminal
            Psi = ctx.Psi[ui]
            Qf = Qs[ctx.kf]
            @views begin
                Psi[:, xr] .= .-Qf
                fill!(Psi[:, lr], zero(T))
                for i in 1:n
                    Psi[i, n + i] = one(T)
                end
                Psi[:, d + 1] .= .-ctx.hf[u.v[_LQR_BLOCK_F]]
                # Terminal reference: the residual carries +Q_f G_r u_T.
                m > 0 && mul!(Psi[:, (d + 2):(d + 1 + m)], Qf, ctx.Gref[u.v[_LQR_BLOCK_G]])
            end
        end
    end
    return ctx
end

#=
The residual scatter, and the hot loop of the whole M-step: L-BFGS calls it once
per objective evaluation, and its cost is `O(d² · reg)` per unit and regime —
independent of how many trials went into the statistics, which is why a large
plant dimension is what this has to be fast for.

Everything is written through preallocated scratch and 5-argument `mul!`. The
obvious spelling, `R .-= Th * Xvᵀ` and friends, allocates three `d × d` or
`d × reg` temporaries per unit and regime on *every* evaluation.
=#
function _lqr_residuals!(ctx::_LQRMStepCtx{T}) where {T<:Real}
    p = ctx.pack
    terminal = ctx.sms[1].terminal
    for s in 1:(ctx.nq)
        fill!(ctx.R[s], zero(T))
        fill!(ctx.Rf[s], zero(T))
    end
    TX = ctx.tmp_dd
    TZ = ctx.tmp_dr
    PO = ctx.tmp_nr
    for (ui, u) in enumerate(ctx.units)
        hs = u.hs
        R = ctx.R[u.q]
        R .+= hs.Yv
        for k in 1:(p.K)
            Th = ctx.Theta[ui][k]
            # R -= Θ Xᵀ + (Θ Xᵀ)ᵀ, then R += Θ Z Θᵀ.
            mul!(TX, Th, transpose(hs.Xv[k]))
            R .-= TX
            R .-= transpose(TX)
            mul!(TZ, Th, hs.Zw[k])
            mul!(R, TZ, transpose(Th), one(T), one(T))
        end
        if terminal
            Psi = ctx.Psi[ui]
            mul!(PO, Psi, hs.Omega)
            mul!(ctx.Rf[u.q], PO, transpose(Psi), one(T), one(T))
        end
    end
    for s in 1:(ctx.nq)
        Symmetrize!(ctx.R[s])
        terminal && Symmetrize!(ctx.Rf[s])
    end
    return ctx
end

function _lqr_fg!(
    grad::Union{Nothing,AbstractVector{T}}, θ::AbstractVector{T}, ctx::_LQRMStepCtx{T}
) where {T<:Real}
    p = ctx.pack
    n, d, m, K = p.n, p.d, p.m, p.K
    xr, lr = 1:n, (n + 1):d
    terminal = ctx.sms[1].terminal
    nA = p.nv[_LQR_BLOCK_A]

    #=
    A rejected step must leave `grad` defined, not stale: L-BFGS's line search
    may ask for the gradient at a point whose objective came back infinite, and
    a zero gradient there is the honest "no information" answer.
    =#
    grad === nothing || fill!(grad, zero(T))

    _lqr_unpack!(ctx, θ)

    fval = zero(T)
    F = Vector{LU{T,Matrix{T},Vector{Int}}}(undef, nA)
    for a in 1:nA
        Fa = lu(ctx.A[a]; check=false)
        issuccess(Fa) || return T(Inf)
        logdetA, _ = logabsdet(Fa)
        isfinite(logdetA) || return T(Inf)
        fval -= ctx.N_A[a] * T(logdetA)
        F[a] = Fa
    end

    _lqr_assemble!(ctx)
    _lqr_residuals!(ctx)

    for s in 1:(ctx.nq)
        if ctx.profile
            chol = cholesky(Symmetric(ctx.R[s]); check=false)
            issuccess(chol) || return T(Inf)
            fval += T(0.5) * ctx.N_q[s] * logdet(chol)
            copyto!(ctx.W[s], inv(chol))
            ctx.W[s] .*= ctx.N_q[s]
        else
            fval += T(0.5) * dot(ctx.Sinv[s], ctx.R[s])
            copyto!(ctx.W[s], ctx.Sinv[s])
        end
        if terminal && ctx.Nf_q[s] > zero(T)
            if ctx.profile
                cholf = cholesky(Symmetric(ctx.Rf[s]); check=false)
                issuccess(cholf) || return T(Inf)
                fval += T(0.5) * ctx.Nf_q[s] * logdet(cholf)
                copyto!(ctx.Wf[s], inv(cholf))
                ctx.Wf[s] .*= ctx.Nf_q[s]
            else
                fval += T(0.5) * dot(ctx.Sfinv[s], ctx.Rf[s])
                copyto!(ctx.Wf[s], ctx.Sfinv[s])
            end
        else
            fill!(ctx.Wf[s], zero(T))
        end
    end

    grad === nothing && return fval

    for a in 1:nA
        fill!(ctx.dA[a], zero(T))
    end
    for v in 1:(p.nv[_LQR_BLOCK_S])
        fill!(ctx.dS[v], zero(T))
    end
    for v in 1:(p.nv[_LQR_BLOCK_Q]), k in 1:K
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
    for (ui, u) in enumerate(ctx.units)
        Ws = ctx.W[u.q]
        vA, vS, vQ = u.v[_LQR_BLOCK_A], u.v[_LQR_BLOCK_S], u.v[_LQR_BLOCK_Q]
        vh, vB, vG = u.v[_LQR_BLOCK_H], u.v[_LQR_BLOCK_B], u.v[_LQR_BLOCK_G]
        for k in 1:K
            copyto!(E, u.hs.Xv[k])
            mul!(E, ctx.Theta[ui][k], u.hs.Zw[k], one(T), -one(T))
            mul!(Gk, Ws, E)
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
                    mul!(
                        ctx.dQ[vQ][k], dBk[lr, :], transpose(ctx.Gref[vG]), -one(T), one(T)
                    )
                    mul!(ctx.dG[vG], ctx.Qc[vQ][k], dBk[lr, :], -one(T), one(T))
                end
            end
        end
        if terminal && ctx.Nf_q[u.q] > zero(T)
            # ∂/∂Ψ of the terminal term; Ψ = [−Q_f  I  −h_f  Q_f G_r].
            mul!(PO, ctx.Psi[ui], u.hs.Omega)
            mul!(GP, ctx.Wf[u.q], PO)
            @views begin
                ctx.dQ[vQ][ctx.kf] .-= GP[:, xr]
                if m > 0
                    dPu = GP[:, (d + 2):(d + 1 + m)]
                    mul!(ctx.dQ[vQ][ctx.kf], dPu, transpose(ctx.Gref[vG]), one(T), one(T))
                    mul!(ctx.dG[vG], ctx.Qc[vQ][ctx.kf], dPu, one(T), one(T))
                end
                ctx.dhf[u.v[_LQR_BLOCK_F]] .-= GP[:, d + 1]
            end
        end
    end

    for a in 1:nA
        # Jacobian term: ∂(−N_a log|det A_a|)/∂A_a = −N_a A_aâ»áµ€.
        copyto!(ctx.tmp_nn, transpose(inv(F[a])))
        ctx.dA[a] .-= ctx.N_A[a] .* ctx.tmp_nn
        r = _lqr_blk(p, _LQR_BLOCK_A, a)
        isempty(r) || copyto!(view(grad, r), vec(ctx.dA[a]))
    end
    for v in 1:(p.nv[_LQR_BLOCK_S])
        r = _lqr_blk(p, _LQR_BLOCK_S, v)
        if !isempty(r)
            # For S = L Lᵀ, dF/dL = (dF/dS + (dF/dS)ᵀ) L.
            _sym!(ctx.tmp_nn, ctx.dS[v])
            mul!(
                reshape(view(grad, r), n, n),
                ctx.tmp_nn,
                reshape(view(θ, r), n, n),
                T(2),
                zero(T),
            )
        end
    end
    for v in 1:(p.nv[_LQR_BLOCK_Q]), k in 1:K
        r = _lqr_blk_q(p, v, k)
        if !isempty(r)
            _sym!(ctx.tmp_nn, ctx.dQ[v][k])
            mul!(
                reshape(view(grad, r), n, n),
                ctx.tmp_nn,
                reshape(view(θ, r), n, n),
                T(2),
                zero(T),
            )
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
    for v in 1:(p.nv[_LQR_BLOCK_Q]), k in 1:(p.K), c in o[_LQR_BLOCK_Q][v]
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
            end
        end
    end

    _lqr_unpack!(ctx, θ0)
    _lqr_assemble!(ctx)
    _lqr_residuals!(ctx)
    return ctx
end

"""
    _lqr_noise_mstep!(ctx)

`Σ_s = R_s/N_s` and `Σ_{f,s} = R_{f,s}/N_{f,s}` — the closed-form maximizers
given the structural parameters, which is exactly what profiling them out of the
objective assumed. Units sharing a noise version pool into that version's `R`.
"""
function _lqr_noise_mstep!(ctx::_LQRMStepCtx{T}) where {T<:Real}
    #=
    Written to every model on the noise version, not just one: as with the
    structural blocks, models sharing a version hold separate arrays in an
    `SLDS` (they alias only under `depends_on`), so the fitted value has to
    reach all of them.
    =#
    for s in 1:(ctx.nq)
        for (c, sm) in enumerate(ctx.sms)
            ctx.q_of[c] == s || continue
            if ctx.N_q[s] > zero(T)
                copyto!(sm.Σ, ctx.R[s])
                sm.Σ ./= ctx.N_q[s]
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
    ctx = _LQRMStepCtx(hs, sm, lds.fit_bool[4])
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
    sm = lds.state_model
    d = _state_latent_dim(sm)
    m = size(sm.Bu, 2)
    reg = d + 1 + m
    Sww = hs.zz[1]
    Svw = Matrix{T}(transpose(hs.zy[1]))
    N = T(hs.nk[1])

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
    end

    if lds.fit_bool[4] && N > zero(T)
        R = _free_residual_scatter(Theta, hs)
        R ./= N
        copyto!(sm.Σ, R)
    end

    @views begin
        copyto!(sm.Mfree, Theta[:, 1:d])
        copyto!(sm.h, Theta[:, d + 1])
        m > 0 && copyto!(sm.Bu, Theta[:, (d + 2):reg])
    end
    return nothing
end

# ============================================================================
# ELBO
# ============================================================================

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
function Q_state!(
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
    copyto!(S_init, suf.init_yy[])
    μ_sum = vec(suf.init_xy)
    BLAS.ger!(-one(T), μ_sum, x0, S_init)
    BLAS.ger!(-one(T), x0, μ_sum, S_init)
    BLAS.ger!(T(N1), x0, x0, S_init)
    ldiv!(P0_U', S_init)
    ldiv!(P0_U, S_init)
    Q_val = T(-0.5) * (T(N1) * (T(d) * log2π + logdet(P0_PD)) + tr(S_init))

    # `:free` mode has no constrained parameterization to profile through.
    _is_free(sm) && return Q_val + _free_Q_transition(sm, hs)

    #=
    Fill the mixed blocks here rather than relying on the caller: this is reached
    from the ungrouped `elbo!` and from the grouped one, and an ordering hazard
    that silently reads stale blocks is not worth the block copies it saves.
    =#
    _fill_mixed_blocks!(hs, sm)

    # Transition term, in mixed coordinates, at the model's current parameters.
    ctx = _LQRMStepCtx(hs, sm, false)
    θ = zeros(T, _lqr_nparams(ctx))
    _lqr_pack!(θ, ctx)
    _lqr_unpack!(ctx, θ)
    _lqr_assemble!(ctx)
    _lqr_residuals!(ctx)

    # One unit and one noise version here: `Q_state!` is always called per cell.
    if ctx.N_q[1] > zero(T)
        Σ_PD = PDMat(Symmetrize!(Matrix{T}(sm.Σ)))
        Q_val +=
            T(-0.5) * (
                ctx.N_q[1] * (T(d) * log2π + logdet(Σ_PD) - T(2) * sm.cache.logabsdetA) +
                dot(ctx.Sinv[1], ctx.R[1])
            )
    end

    if sm.terminal && ctx.Nf_q[1] > zero(T)
        Σf_PD = PDMat(Symmetrize!(Matrix{T}(sm.Σf)))
        Q_val +=
            T(-0.5) *
            (ctx.Nf_q[1] * (T(n) * log2π + logdet(Σf_PD)) + dot(ctx.Sfinv[1], ctx.Rf[1]))
    end

    return Q_val
end
