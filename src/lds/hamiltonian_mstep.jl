#=============================================================================
Hamiltonian (inverse-LQR) latents — sufficient statistics, M-step and ELBO.

The E-step gives the usual linear-Gaussian smoother output on `z = [x; λ]`. The
M-step must return parameters whose transition still has the Hamiltonian form,
which is done in *mixed* coordinates: with

    w_t = [x_t; λ_{t+1}],    v_t = [x_{t+1}; λ_t],

the constraint is that the regression `v_t ≈ 𝓔_k w_t + h + Bu u_t` has
`𝓔_k = [A −S; Q_k Aᵀ]` with `S`, `Q_k` symmetric — *linear* in the free
parameters, where the forward symplectic transition `M_k` is rational in them.

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
    HamiltonianSufficientStatistics{T}

Aggregated E-step statistics for a [`HamiltonianStateModel`](@ref).

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
mutable struct HamiltonianSufficientStatistics{T<:Real,B}
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
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    return _wrap_hamiltonian_suff_stats(
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
) where {T<:Real,S<:HamiltonianStateModel{T},O<:CompositeObservationModel{T}}
    views = _obs_views(lds)
    base = map(v -> _base_td_sufficient_statistics(T, v, tsteps_per_trial), views)
    return _wrap_hamiltonian_suff_stats(base, lds, tsteps_per_trial)
end

"""
    _wrap_hamiltonian_suff_stats(base, lds, tsteps_per_trial)

Allocate the per-regime state-side blocks around an already-built `base`.
"""
function _wrap_hamiltonian_suff_stats(
    base, lds::LinearDynamicalSystem{T,S,O}, tsteps_per_trial::AbstractVector{Int}
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    sm = lds.state_model
    d = lds.latent_dim
    m = lds.ux_dim
    K = _nregimes(sm)
    reg = d + 1 + m
    return HamiltonianSufficientStatistics{T,typeof(base)}(
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
function _regime_runs(sm::HamiltonianStateModel, tsteps::Int)
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
    _aggregate_hamiltonian_stats!(hs, tfs, lds, data)

Accumulate the per-regime state-side statistics from the smoother output.

`E[z_t z_tᵀ] = x_t x_tᵀ + P_t` and `E[z_t z_{t+1}ᵀ] = x_t x_{t+1}ᵀ + Cov(z_t,
z_{t+1})`, where the smoother stores `p_smooth_tt1[:, :, t] = Cov(z_t, z_{t-1})`
— hence the adjoint on that term. Means go through GEMM per run; the covariance
sums are the unavoidable per-timestep part, exactly as on the Gaussian path.
"""
function _aggregate_hamiltonian_stats!(
    hs::HamiltonianSufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    sm = lds.state_model
    d = lds.latent_dim
    m = lds.ux_dim
    reg = d + 1 + m
    K = _nregimes(sm)
    ntrials = length(tfs)

    for k in 1:K
        fill!(hs.zz[k], zero(T))
        fill!(hs.zy[k], zero(T))
        fill!(hs.yy[k], zero(T))
        hs.nk[k] = zero(T)
    end
    fill!(hs.term_zz, zero(T))
    hs.term_n = zero(T)

    for trial in 1:ntrials
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

    return _finalize_hamiltonian_stats!(hs, sm, d, m, reg, K)
end

"""
    _finalize_hamiltonian_stats!(hs, sm, d, m, reg, K) -> hs

Mirror the upper triangles the accumulation loops filled, for both the
transition Grams and the terminal one. Shared by the plain and the
responsibility-weighted aggregators, which differ only in how they accumulate.
"""
function _finalize_hamiltonian_stats!(
    hs::HamiltonianSufficientStatistics{T},
    sm::HamiltonianStateModel{T},
    d::Int,
    m::Int,
    reg::Int,
    K::Int,
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
    _aggregate_hamiltonian_stats_weighted!(hs, tfs, lds, data, weights) -> hs

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
function _aggregate_hamiltonian_stats_weighted!(
    hs::HamiltonianSufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T},
    weights::AbstractVector{<:AbstractVector{T}},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
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
        ux = data.ux[trial]::Matrix{T}
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

    return _finalize_hamiltonian_stats!(hs, sm, d, m, reg, K)
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
    hs::HamiltonianSufficientStatistics{T}, sm::HamiltonianStateModel{T}
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
    _HamPack

Where each structural block sits in the flat L-BFGS parameter vector. A frozen
block (see [`HamiltonianFitFlags`](@ref)) gets an empty range and is neither
packed nor updated, so freezing shrinks the problem rather than projecting its
solution.
"""
struct _HamPack
    n::Int
    d::Int
    m::Int
    K::Int
    iA::UnitRange{Int}
    iS::UnitRange{Int}
    iQ::Vector{UnitRange{Int}}
    ih::UnitRange{Int}
    iB::UnitRange{Int}
    iG::UnitRange{Int}
    ihf::UnitRange{Int}
    np::Int
end

function _HamPack(sm::HamiltonianStateModel)
    return _HamPack(sm, sm.fit_flags)
end

#=
Taking the flags explicitly rather than off the model is what lets a *partial*
tie work: sharing `A` and `S` across discrete states while fitting `Qc` per state
is run as two passes over the same machinery, one with only the shared blocks
free and one with only the per-state blocks free. See `_ham_partial_tie_mstep!`.
=#
function _HamPack(sm::HamiltonianStateModel, f::HamiltonianFitFlags)
    n = _plant_dim(sm)
    d = 2n
    m = size(sm.Bu, 2)
    K = _nregimes(sm)
    widths = Int[
        f.A ? n * n : 0,
        f.S ? n * n : 0,
        f.h ? d : 0,
        (f.Bu && m > 0) ? d * m : 0,
        (f.Gref && m > 0) ? n * m : 0,
        (sm.terminal && f.terminal) ? n : 0,
    ]
    #=
    Lay the blocks out in a fixed order — A, S, Q₁…Q_K, h, Bu, Gref, hf — so a
    packed vector means the same thing across M-steps and can be warm started
    from the previous one.
    =#
    pos = 0
    iA = (pos + 1):(pos + widths[1])
    pos += widths[1]
    iS = (pos + 1):(pos + widths[2])
    pos += widths[2]
    iQ = Vector{UnitRange{Int}}(undef, K)
    for k in 1:K
        w = f.Qc ? n * n : 0
        iQ[k] = (pos + 1):(pos + w)
        pos += w
    end
    ih = (pos + 1):(pos + widths[3])
    pos += widths[3]
    iB = (pos + 1):(pos + widths[4])
    pos += widths[4]
    iG = (pos + 1):(pos + widths[5])
    pos += widths[5]
    ihf = (pos + 1):(pos + widths[6])
    pos += widths[6]
    return _HamPack(n, d, m, K, iA, iS, iQ, ih, iB, iG, ihf, pos)
end
"""
    _HamUnit{T,HS}

One pooled group of trials in the structural M-step: its aggregated statistics,
which structural parameter version it uses (`ab`), and which noise version
(`q`).

Cells that share both versions are pooled into one unit before the objective is
built. That is exact, not an approximation: with the same `Θ` the residual
scatter `R = Σ_c (Y_c − Θ X_cᵀ − X_c Θᵀ + Θ Z_c Θᵀ)` is linear in the
statistics, so summing them first gives the same `R`. An ungrouped fit is one
unit, which is why there is a single code path.
"""
struct _HamUnit{T<:Real,HS}
    hs::HS
    ab::Int
    q::Int
    n::T
end

"""
    _HamMStepCtx{T,HS,SM}

Everything the structural objective needs, assembled once per M-step.

The layout generalizes over `depends_on`: `nab` structural versions and `nq`
noise versions, with each unit naming the pair it uses. The flat parameter
vector is `nab` copies of the single-version layout in `pack`, laid end to end,
so version `a`'s block `X` lives at `(a-1)·pack.np .+ pack.iX`.

`profile` says whether `Σ` is being profiled out — it is when the enclosing
`fit_bool` lets the noise move, and then the objective is `(N_s/2) log det R_s`
per noise version; otherwise `Σ` is held fixed and the objective is
`½ tr(Σ_s⁻¹R_s)`. Both carry the same `−N log|det A|` Jacobian term and share one
gradient expression, differing only in the weight applied to the residual
(`N_s R_s⁻¹` versus `Σ_s⁻¹`).

Note where the coupling lives: units sharing a noise version pool into one `R_s`,
so a model whose structure varies by group but whose noise does not is *not*
separable across groups — which is exactly why the objective is built jointly
rather than fitted group by group.
"""
struct _HamMStepCtx{T<:Real,HS,SM}
    pack::_HamPack
    nab::Int
    nq::Int
    units::Vector{_HamUnit{T,HS}}
    sms::Vector{SM}                  # one structural variant per `ab` version
    profile::Bool
    N_ab::Vector{T}                  # transitions per structural version
    N_q::Vector{T}                   # transitions per noise version
    Nf_q::Vector{T}                  # terminal factors per noise version
    kf::Int
    Sinv::Vector{Matrix{T}}
    Sfinv::Vector{Matrix{T}}
    # scratch, reused across objective evaluations
    A::Vector{Matrix{T}}
    S::Vector{Matrix{T}}
    Qc::Vector{Vector{Matrix{T}}}
    h::Vector{Vector{T}}
    Bu::Vector{Matrix{T}}
    Gref::Vector{Matrix{T}}
    hf::Vector{Vector{T}}
    Theta::Vector{Vector{Matrix{T}}}  # [ab version][regime]
    Psi::Vector{Matrix{T}}            # [ab version]
    R::Vector{Matrix{T}}              # [noise version], pooled
    Rf::Vector{Matrix{T}}             # [noise version], pooled
end

"""
    _ham_units(sufs, ab_slots, q_slots) -> Vector{_HamUnit}

Pool cells that share both a structural and a noise version into one unit each.
"""
function _ham_units(
    sufs::AbstractVector, ab_slots::AbstractVector{Int}, q_slots::AbstractVector{Int}
)
    T = eltype(first(sufs).nk)
    pairs = Tuple{Int,Int}[]
    members = Vector{Int}[]
    for c in eachindex(sufs)
        key = (ab_slots[c], q_slots[c])
        i = findfirst(isequal(key), pairs)
        if i === nothing
            push!(pairs, key)
            push!(members, [c])
        else
            push!(members[i], c)
        end
    end
    units = _HamUnit{T,eltype(sufs)}[]
    for (i, (a, q)) in enumerate(pairs)
        hs = if length(members[i]) == 1
            sufs[members[i][1]]
        else
            _pool_ham_stats(sufs, members[i])
        end
        push!(units, _HamUnit{T,eltype(sufs)}(hs, a, q, sum(hs.nk)))
    end
    return units
end

"""
    _pool_ham_stats(sufs, idx) -> HamiltonianSufficientStatistics

Sum the mixed-coordinate blocks of several cells into the first one's shape. The
per-regime `Zw` / `Xv`, the total `Yv`, the terminal `Ω` and the counts all add,
because every cell lives in the same latent space with the same cost schedule.

Writes into a fresh copy: the cells' own statistics are still needed for their
`x0` / `P0` updates and for the ELBO.
"""
function _pool_ham_stats(sufs::AbstractVector, idx::AbstractVector{Int})
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

function _HamMStepCtx(
    sufs::AbstractVector,
    sms::AbstractVector,
    ab_slots::AbstractVector{Int},
    q_slots::AbstractVector{Int},
    profile::Bool;
    flags::Union{Nothing,HamiltonianFitFlags}=nothing,
)
    units = _ham_units(sufs, ab_slots, q_slots)
    sm1 = sms[1]
    T = eltype(sm1.Σ)
    pack = _HamPack(sm1, flags === nothing ? sm1.fit_flags : flags)
    n, d, m, K = pack.n, pack.d, pack.m, pack.K
    reg = d + 1 + m
    nab = maximum(ab_slots)
    nq = maximum(q_slots)

    N_ab = zeros(T, nab)
    N_q = zeros(T, nq)
    Nf_q = zeros(T, nq)
    for u in units
        N_ab[u.ab] += u.n
        N_q[u.q] += u.n
        sm1.terminal && (Nf_q[u.q] += u.hs.term_n)
    end
    kf = (sm1.terminal && !isempty(sm1.schedule)) ? sm1.schedule[end] : 1

    #=
    A noise version's inverse comes from any unit that uses it — every such unit
    points at the same `Σ` array, since a shared parameter is shared by
    reference across variants.
    =#
    noise_sm = Vector{typeof(sm1)}(undef, nq)
    for u in units
        noise_sm[u.q] = sms[u.ab]
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

    return _HamMStepCtx{T,eltype(sufs),typeof(sm1)}(
        pack,
        nab,
        nq,
        units,
        collect(sms),
        profile,
        N_ab,
        N_q,
        Nf_q,
        kf,
        Sinv,
        Sfinv,
        [Matrix{T}(undef, n, n) for _ in 1:nab],
        [Matrix{T}(undef, n, n) for _ in 1:nab],
        [[Matrix{T}(undef, n, n) for _ in 1:K] for _ in 1:nab],
        [Vector{T}(undef, d) for _ in 1:nab],
        [Matrix{T}(undef, d, m) for _ in 1:nab],
        [Matrix{T}(undef, n, m) for _ in 1:nab],
        [Vector{T}(undef, n) for _ in 1:nab],
        [[zeros(T, d, reg) for _ in 1:K] for _ in 1:nab],
        [zeros(T, n, d + 1 + m) for _ in 1:nab],
        [Matrix{T}(undef, d, d) for _ in 1:nq],
        [Matrix{T}(undef, n, n) for _ in 1:nq],
    )
end

# Ungrouped convenience: one unit, one structural version, one noise version.
function _HamMStepCtx(hs, sm::HamiltonianStateModel, profile::Bool)
    return _HamMStepCtx([hs], [sm], [1], [1], profile)
end

"""
    _ham_offset(ctx, a) -> Int

Where structural version `a`'s block starts in the flat parameter vector.
"""
@inline _ham_offset(ctx::_HamMStepCtx, a::Int) = (a - 1) * ctx.pack.np

@inline function _ham_range(ctx::_HamMStepCtx, a::Int, r::UnitRange{Int})
    isempty(r) && return r
    off = _ham_offset(ctx, a)
    return (first(r) + off):(last(r) + off)
end

"""
    _ham_nparams(ctx) -> Int

Total free parameters: one structural layout per version.
"""
@inline _ham_nparams(ctx::_HamMStepCtx) = ctx.nab * ctx.pack.np

"""
    _ham_pack!(θ, ctx)

Write every structural version's current parameters into the flat vector `θ`,
which is where L-BFGS starts. Symmetric blocks are packed as full matrices:
their gradients come back exactly symmetric, so a symmetric start stays
symmetric through every L-BFGS iterate without any `vech` bookkeeping.
"""
function _ham_pack!(θ::AbstractVector{T}, ctx::_HamMStepCtx{T}) where {T<:Real}
    p = ctx.pack
    for a in 1:(ctx.nab)
        sm = ctx.sms[a]
        rA = _ham_range(ctx, a, p.iA)
        isempty(rA) || copyto!(view(θ, rA), vec(sm.A))
        rS = _ham_range(ctx, a, p.iS)
        isempty(rS) || copyto!(view(θ, rS), vec(sm.S))
        for k in 1:(p.K)
            rQ = _ham_range(ctx, a, p.iQ[k])
            isempty(rQ) || copyto!(view(θ, rQ), vec(sm.Qc[k]))
        end
        rh = _ham_range(ctx, a, p.ih)
        isempty(rh) || copyto!(view(θ, rh), sm.h)
        rB = _ham_range(ctx, a, p.iB)
        isempty(rB) || copyto!(view(θ, rB), vec(sm.Bu))
        rG = _ham_range(ctx, a, p.iG)
        isempty(rG) || copyto!(view(θ, rG), vec(sm.Gref))
        rf = _ham_range(ctx, a, p.ihf)
        isempty(rf) || copyto!(view(θ, rf), sm.hf)
    end
    return θ
end

"""
    _ham_unpack!(ctx, θ)

Fill the context's scratch parameters from `θ`, taking any frozen block from
that version's own model instead. Symmetric blocks are symmetrized on the way in
so an accumulated rounding asymmetry can never make `𝓔` leave the Hamiltonian
form.
"""
function _ham_unpack!(ctx::_HamMStepCtx{T}, θ::AbstractVector{T}) where {T<:Real}
    p = ctx.pack
    n = p.n
    for a in 1:(ctx.nab)
        sm = ctx.sms[a]
        rA = _ham_range(ctx, a, p.iA)
        if isempty(rA)
            copyto!(ctx.A[a], sm.A)
        else
            copyto!(ctx.A[a], reshape(view(θ, rA), n, n))
        end
        rS = _ham_range(ctx, a, p.iS)
        if isempty(rS)
            copyto!(ctx.S[a], sm.S)
        else
            copyto!(ctx.S[a], reshape(view(θ, rS), n, n))
            Symmetrize!(ctx.S[a])
        end
        for k in 1:(p.K)
            rQ = _ham_range(ctx, a, p.iQ[k])
            if isempty(rQ)
                copyto!(ctx.Qc[a][k], sm.Qc[k])
            else
                copyto!(ctx.Qc[a][k], reshape(view(θ, rQ), n, n))
                Symmetrize!(ctx.Qc[a][k])
            end
        end
        rh = _ham_range(ctx, a, p.ih)
        isempty(rh) ? copyto!(ctx.h[a], sm.h) : copyto!(ctx.h[a], view(θ, rh))
        if p.m > 0
            rB = _ham_range(ctx, a, p.iB)
            if isempty(rB)
                copyto!(ctx.Bu[a], sm.Bu)
            else
                copyto!(ctx.Bu[a], reshape(view(θ, rB), p.d, p.m))
            end
            rG = _ham_range(ctx, a, p.iG)
            if isempty(rG)
                copyto!(ctx.Gref[a], sm.Gref)
            else
                copyto!(ctx.Gref[a], reshape(view(θ, rG), n, p.m))
            end
        end
        rf = _ham_range(ctx, a, p.ihf)
        isempty(rf) ? copyto!(ctx.hf[a], sm.hf) : copyto!(ctx.hf[a], view(θ, rf))
    end
    return ctx
end

"""
    _ham_assemble!(ctx)

Build `Θ_{a,k} = [𝓔_{a,k}  h_a  B_{a,k}]` for every structural version and
regime, and, when there is a terminal factor, `Ψ_a = [−Q_f  I  −h_f  Q_f G_r]`.

The input block `B_{a,k} = B_u − [0; Q_{a,k} G_r]` is regime-dependent: its
costate half is the tracking term `−Q_k r_t`, tied to that regime's own cost
rather than free.
"""
function _ham_assemble!(ctx::_HamMStepCtx{T}) where {T<:Real}
    p = ctx.pack
    n, d, m = p.n, p.d, p.m
    xr, lr = 1:n, (n + 1):d
    for a in 1:(ctx.nab)
        for k in 1:(p.K)
            Th = ctx.Theta[a][k]
            @views begin
                Th[xr, xr] .= ctx.A[a]
                Th[xr, lr] .= .-ctx.S[a]
                Th[lr, xr] .= ctx.Qc[a][k]
                Th[lr, lr] .= transpose(ctx.A[a])
                Th[:, d + 1] .= ctx.h[a]
                if m > 0
                    Bcol = Th[:, (d + 2):(d + 1 + m)]
                    Bcol .= ctx.Bu[a]
                    mul!(Bcol[lr, :], ctx.Qc[a][k], ctx.Gref[a], -one(T), one(T))
                end
            end
        end
        if ctx.sms[1].terminal
            Psi = ctx.Psi[a]
            @views begin
                Psi[:, xr] .= .-ctx.Qc[a][ctx.kf]
                fill!(Psi[:, lr], zero(T))
                for i in 1:n
                    Psi[i, n + i] = one(T)
                end
                Psi[:, d + 1] .= .-ctx.hf[a]
                # Terminal reference: the residual carries +Q_f G_r u_T.
                m > 0 && mul!(Psi[:, (d + 2):(d + 1 + m)], ctx.Qc[a][ctx.kf], ctx.Gref[a])
            end
        end
    end
    return ctx
end

"""
    _ham_residuals!(ctx)

Residual scatters at the current scratch parameters, pooled per noise version:
`R_s = Σ_{u: q(u)=s} Σ_k (Y_u − Θ X_uᵀ − X_u Θᵀ + Θ Z_u Θᵀ)` and, for the
terminal factor, `R_{f,s} = Σ_u Ψ Ω_u Ψᵀ`. Both are symmetrized: they are
positive semi-definite by construction, and the factorizations downstream need
that exactly.
"""
function _ham_residuals!(ctx::_HamMStepCtx{T}) where {T<:Real}
    for s in 1:(ctx.nq)
        fill!(ctx.R[s], zero(T))
        fill!(ctx.Rf[s], zero(T))
    end
    for u in ctx.units
        hs = u.hs
        R = ctx.R[u.q]
        R .+= hs.Yv
        for k in 1:(ctx.pack.K)
            Th = ctx.Theta[u.ab][k]
            TX = Th * transpose(hs.Xv[k])
            R .-= TX
            R .-= transpose(TX)
            R .+= Th * hs.Zw[k] * transpose(Th)
        end
        if ctx.sms[1].terminal
            Psi = ctx.Psi[u.ab]
            ctx.Rf[u.q] .+= Psi * hs.Omega * transpose(Psi)
        end
    end
    for s in 1:(ctx.nq)
        Symmetrize!(ctx.R[s])
        ctx.sms[1].terminal && Symmetrize!(ctx.Rf[s])
    end
    return ctx
end

"""
    _ham_fg!(grad, θ, ctx) -> objective

The structural M-step objective and its gradient. Minimizing

```math
g(\\theta) = \\sum_s \\tfrac{N_s}{2}\\log\\det R_s(\\theta)
           + \\sum_s \\tfrac{N_{f,s}}{2}\\log\\det R_{f,s}(\\theta)
           - \\sum_a N_a \\log|\\det A_a|
```

(the `Σ`-profiled case; with `Σ` frozen the `log det` terms become
`½ tr(Σ_s⁻¹R_s)`) over every structural version's `(A, S, Q_{1:K}, h, B_u, G_r,
h_f)`.

The elementwise gradient with respect to a unit's stacked regression is
`G_k = W_{q(u)} (Θ_k Z_k − X_k)` with `W_s = N_s R_s⁻¹` profiled and `W_s = Σ_s⁻¹`
frozen, and chains onto the structural blocks by

- `∂/∂A   = Σ_k (G_k[x,x] + G_k[λ,λ]ᵀ) − N_a A_a⁻ᵀ`  (`A` sits in `𝓔` twice,
  once transposed — and the `A⁻ᵀ` is the Jacobian term)
- `∂/∂S   = −Σ_k G_k[x,λ]`, symmetrized
- `∂/∂Q_k = G_k[λ,x] − G_k[λ,u] G_rᵀ`, symmetrized, plus the terminal factor's
  share for `k = k_f`
- `∂/∂G_r = −Σ_k Q_k G_k[λ,u]`, plus the terminal factor's share

Returns `Inf` (with `grad` zeroed) at a `θ` where a residual scatter is not
positive definite or some `A_a` is singular, which the line search reads as a
rejected step.
"""
function _ham_fg!(
    grad::Union{Nothing,AbstractVector{T}}, θ::AbstractVector{T}, ctx::_HamMStepCtx{T}
) where {T<:Real}
    p = ctx.pack
    n, d, m, K = p.n, p.d, p.m, p.K
    xr, lr = 1:n, (n + 1):d
    terminal = ctx.sms[1].terminal

    #=
    A rejected step must leave `grad` defined, not stale: L-BFGS's line search
    may ask for the gradient at a point whose objective came back infinite, and
    a zero gradient there is the honest "no information" answer.
    =#
    grad === nothing || fill!(grad, zero(T))

    _ham_unpack!(ctx, θ)

    fval = zero(T)
    F = Vector{Any}(undef, ctx.nab)
    for a in 1:(ctx.nab)
        Fa = lu(ctx.A[a]; check=false)
        issuccess(Fa) || return T(Inf)
        logdetA, _ = logabsdet(Fa)
        isfinite(logdetA) || return T(Inf)
        fval -= ctx.N_ab[a] * T(logdetA)
        F[a] = Fa
    end

    _ham_assemble!(ctx)
    _ham_residuals!(ctx)

    W = Vector{Matrix{T}}(undef, ctx.nq)
    Wf = Vector{Matrix{T}}(undef, ctx.nq)
    for s in 1:(ctx.nq)
        if ctx.profile
            chol = cholesky(Symmetric(ctx.R[s]); check=false)
            issuccess(chol) || return T(Inf)
            fval += T(0.5) * ctx.N_q[s] * logdet(chol)
            W[s] = ctx.N_q[s] .* Matrix(inv(chol))
        else
            fval += T(0.5) * dot(ctx.Sinv[s], ctx.R[s])
            W[s] = ctx.Sinv[s]
        end
        if terminal && ctx.Nf_q[s] > zero(T)
            if ctx.profile
                cholf = cholesky(Symmetric(ctx.Rf[s]); check=false)
                issuccess(cholf) || return T(Inf)
                fval += T(0.5) * ctx.Nf_q[s] * logdet(cholf)
                Wf[s] = ctx.Nf_q[s] .* Matrix(inv(cholf))
            else
                fval += T(0.5) * dot(ctx.Sfinv[s], ctx.Rf[s])
                Wf[s] = ctx.Sfinv[s]
            end
        else
            Wf[s] = zeros(T, n, n)
        end
    end

    grad === nothing && return fval

    dA = [zeros(T, n, n) for _ in 1:(ctx.nab)]
    dS = [zeros(T, n, n) for _ in 1:(ctx.nab)]
    dQ = [[zeros(T, n, n) for _ in 1:K] for _ in 1:(ctx.nab)]
    dh = [zeros(T, d) for _ in 1:(ctx.nab)]
    dB = [zeros(T, d, m) for _ in 1:(ctx.nab)]
    dG = [zeros(T, n, m) for _ in 1:(ctx.nab)]
    dhf = [zeros(T, n) for _ in 1:(ctx.nab)]

    for u in ctx.units
        a = u.ab
        Ws = W[u.q]
        for k in 1:K
            Gk = Ws * (ctx.Theta[a][k] * u.hs.Zw[k] .- u.hs.Xv[k])
            @views begin
                dA[a] .+= Gk[xr, xr]
                dA[a] .+= transpose(Gk[lr, lr])
                dS[a] .-= Gk[xr, lr]
                dQ[a][k] .+= Gk[lr, xr]
                dh[a] .+= Gk[:, d + 1]
                if m > 0
                    #=
                    The input block is `B_u − [0; Q_k G_r]`, so its gradient
                    splits: `B_u` takes it whole, while `G_r` and `Q_k` share the
                    tracking half bilinearly.
                    =#
                    dBk = Gk[:, (d + 2):(d + 1 + m)]
                    dB[a] .+= dBk
                    mul!(dQ[a][k], dBk[lr, :], transpose(ctx.Gref[a]), -one(T), one(T))
                    mul!(dG[a], ctx.Qc[a][k], dBk[lr, :], -one(T), one(T))
                end
            end
        end
        if terminal && ctx.Nf_q[u.q] > zero(T)
            # ∂/∂Ψ of the terminal term; Ψ = [−Q_f  I  −h_f  Q_f G_r].
            GP = Wf[u.q] * (ctx.Psi[a] * u.hs.Omega)
            @views begin
                dQ[a][ctx.kf] .-= GP[:, xr]
                if m > 0
                    dPu = GP[:, (d + 2):(d + 1 + m)]
                    mul!(dQ[a][ctx.kf], dPu, transpose(ctx.Gref[a]), one(T), one(T))
                    mul!(dG[a], ctx.Qc[a][ctx.kf], dPu, one(T), one(T))
                end
                dhf[a] .-= GP[:, d + 1]
            end
        end
    end

    for a in 1:(ctx.nab)
        # Jacobian term: ∂(−N_a log|det A_a|)/∂A_a = −N_a A_a⁻ᵀ.
        dA[a] .-= ctx.N_ab[a] .* transpose(inv(F[a]::LU{T,Matrix{T},Vector{Int}}))
        rA = _ham_range(ctx, a, p.iA)
        isempty(rA) || copyto!(view(grad, rA), vec(dA[a]))
        rS = _ham_range(ctx, a, p.iS)
        isempty(rS) || copyto!(view(grad, rS), vec(_sym(dS[a])))
        for k in 1:K
            rQ = _ham_range(ctx, a, p.iQ[k])
            isempty(rQ) || copyto!(view(grad, rQ), vec(_sym(dQ[a][k])))
        end
        rh = _ham_range(ctx, a, p.ih)
        isempty(rh) || copyto!(view(grad, rh), dh[a])
        rB = _ham_range(ctx, a, p.iB)
        isempty(rB) || copyto!(view(grad, rB), vec(dB[a]))
        rG = _ham_range(ctx, a, p.iG)
        isempty(rG) || copyto!(view(grad, rG), vec(dG[a]))
        rf = _ham_range(ctx, a, p.ihf)
        isempty(rf) || copyto!(view(grad, rf), dhf[a])
    end

    return fval
end

"""
    _sym(D) -> Matrix

`(D + Dᵀ)/2`, the gradient with respect to a symmetric parameter matrix.
"""
@inline _sym(D::AbstractMatrix{T}) where {T<:Real} = T(0.5) .* (D .+ transpose(D))

"""
    _ham_writeback!(ctx, θ)

Copy the optimized structural parameters out of `θ` into each version's model,
leaving frozen blocks untouched.
"""
function _ham_writeback!(ctx::_HamMStepCtx{T}, θ::AbstractVector{T}) where {T<:Real}
    _ham_unpack!(ctx, θ)
    p = ctx.pack
    for a in 1:(ctx.nab)
        sm = ctx.sms[a]
        isempty(p.iA) || copyto!(sm.A, ctx.A[a])
        isempty(p.iS) || copyto!(sm.S, ctx.S[a])
        for k in 1:(p.K)
            isempty(p.iQ[k]) || copyto!(sm.Qc[k], ctx.Qc[a][k])
        end
        isempty(p.ih) || copyto!(sm.h, ctx.h[a])
        isempty(p.iB) || copyto!(sm.Bu, ctx.Bu[a])
        isempty(p.iG) || copyto!(sm.Gref, ctx.Gref[a])
        isempty(p.ihf) || copyto!(sm.hf, ctx.hf[a])
    end
    return ctx
end

"""
    _ham_structure_mstep!(ctx, fit_structure, mstep_iters) -> ctx

Optimize the structural parameters, warm started at their current values.

L-BFGS on the objective above is a *generalized* M-step: EM only needs the
objective not to get worse, so the result is accepted only when it beats the
starting point. That also makes a failed inner solve harmless — the parameters
simply do not move that iteration.

Leaves the scratch (and hence `ctx.R` / `ctx.Rf`) at the accepted parameters, so
the noise update can read them without recomputing.
"""
function _ham_structure_mstep!(
    ctx::_HamMStepCtx{T}, fit_structure::Bool, mstep_iters::Int
) where {T<:Real}
    np = _ham_nparams(ctx)
    θ0 = zeros(T, np)
    _ham_pack!(θ0, ctx)

    if fit_structure && np > 0
        f0 = _ham_fg!(nothing, θ0, ctx)
        if isfinite(f0)
            f_obj(θ) = _ham_fg!(nothing, θ, ctx)
            function g_obj!(G, θ)
                _ham_fg!(G, θ, ctx)
                return G
            end
            opts = Optim.Options(;
                x_abstol=1e-10, g_abstol=1e-9, f_reltol=1e-12, iterations=mstep_iters
            )
            result = optimize(f_obj, g_obj!, θ0, LBFGS(; linesearch=HagerZhang()), opts)
            θ1 = Optim.minimizer(result)
            f1 = _ham_fg!(nothing, θ1, ctx)
            #=
            Accept only a genuine improvement. L-BFGS normally returns one, but a
            line-search stall or a step into a region where `R` lost rank would
            otherwise hand EM a worse objective and break monotonicity.
            =#
            if isfinite(f1) && f1 <= f0
                _ham_writeback!(ctx, θ1)
                copyto!(θ0, θ1)
            end
        end
    end

    _ham_unpack!(ctx, θ0)
    _ham_assemble!(ctx)
    _ham_residuals!(ctx)
    return ctx
end

"""
    _ham_noise_mstep!(ctx)

`Σ_s = R_s/N_s` and `Σ_{f,s} = R_{f,s}/N_{f,s}` — the closed-form maximizers
given the structural parameters, which is exactly what profiling them out of the
objective assumed. Units sharing a noise version pool into that version's `R`.
"""
function _ham_noise_mstep!(ctx::_HamMStepCtx{T}) where {T<:Real}
    written = falses(ctx.nq)
    for u in ctx.units
        s = u.q
        written[s] && continue
        written[s] = true
        sm = ctx.sms[u.ab]
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
    return ctx
end

"""
    _ham_state_mstep!(lds, hs, sws)

The state half of the M-step for an ungrouped fit, shared by every emission
model.
"""
function _ham_state_mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    hs::HamiltonianSufficientStatistics{T},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
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
    ctx = _HamMStepCtx(hs, sm, lds.fit_bool[4])
    _ham_structure_mstep!(ctx, lds.fit_bool[3], sm.mstep_iters)
    lds.fit_bool[4] && _ham_noise_mstep!(ctx)
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
function _free_theta(sm::HamiltonianStateModel{T}) where {T<:Real}
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
    Theta::AbstractMatrix{T}, hs::HamiltonianSufficientStatistics{T}
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
    sm::HamiltonianStateModel{T}, hs::HamiltonianSufficientStatistics{T}
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
    lds::LinearDynamicalSystem{T,S,O}, hs::HamiltonianSufficientStatistics{T}
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
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
    (ff.Bu && m > 0) && append!(free_cols, (d + 2):reg)

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

State-side expected complete-data log-likelihood for a Hamiltonian LDS:

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
    hs::HamiltonianSufficientStatistics{T},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
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
    ctx = _HamMStepCtx(hs, sm, false)
    θ = zeros(T, _ham_nparams(ctx))
    _ham_pack!(θ, ctx)
    _ham_unpack!(ctx, θ)
    _ham_assemble!(ctx)
    _ham_residuals!(ctx)

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
