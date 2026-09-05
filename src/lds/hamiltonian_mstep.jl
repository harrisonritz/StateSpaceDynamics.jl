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

`term_zz` / `term_n` hold `Σ_n E[[z_T;1][z_T;1]ᵀ]` and the trial count, the
statistics of the terminal factor.

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
        zeros(T, d + 1, d + 1),
        zero(T),
        [zeros(T, reg, reg) for _ in 1:K],
        [zeros(T, d, reg) for _ in 1:K],
        zeros(T, d, d),
        zeros(T, d + 1, d + 1),
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

        # Terminal factor statistics: E[[z_T; 1][z_T; 1]ᵀ], summed over trials.
        if sm.terminal
            xT = tview(x, :, T_n)
            BLAS.ger!(one(T), xT, xT, tview(hs.term_zz, 1:d, 1:d))
            @views hs.term_zz[1:d, 1:d] .+= p_smooth[:, :, T_n]
            for i in 1:d
                hs.term_zz[i, d + 1] += x[i, T_n]
            end
            hs.term_n += one(T)
        end
    end

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
    end

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
    hs::HamiltonianSufficientStatistics{T}, sm::HamiltonianStateModel{T}
) where {T<:Real}
    n = _plant_dim(sm)
    d = 2n
    K = _nregimes(sm)
    reg = size(hs.zz[1], 1)
    m = reg - d - 1
    xr, lr = 1:n, (n + 1):d      # state rows / costate rows of a `z`

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
    ihf::UnitRange{Int}
    np::Int
end

function _HamPack(sm::HamiltonianStateModel)
    n = _plant_dim(sm)
    d = 2n
    m = size(sm.Bu, 2)
    K = _nregimes(sm)
    f = sm.fit_flags
    widths = Int[
        f.A ? n * n : 0,
        f.S ? n * n : 0,
        f.h ? d : 0,
        (f.Bu && m > 0) ? d * m : 0,
        (sm.terminal && f.terminal) ? n : 0,
    ]
    #=
    Lay the blocks out in a fixed order — A, S, Q₁…Q_K, h, Bu, hf — so a packed
    vector means the same thing across M-steps and can be warm started from the
    previous one.
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
    ihf = (pos + 1):(pos + widths[5])
    pos += widths[5]
    return _HamPack(n, d, m, K, iA, iS, iQ, ih, iB, ihf, pos)
end

"""
    _HamMStepCtx{T}

Everything the structural objective needs, assembled once per M-step: the mixed
blocks, the current values of any frozen parameter, the packing plan, and the
scratch the objective writes its trial parameters into.

`profile` says whether `Σ` is being profiled out — it is when the enclosing
`fit_bool` lets the noise move, and then the objective is `(N/2) log det R`;
otherwise `Σ` is held fixed and the objective is `½ tr(Σ⁻¹R)`. Both carry the
same `−N log|det A|` Jacobian term and share one gradient expression, differing
only in the weight `W` applied to the residual (`N R⁻¹` versus `Σ⁻¹`).
"""
struct _HamMStepCtx{T<:Real,HS,SM}
    pack::_HamPack
    hs::HS
    sm::SM
    profile::Bool
    N::T
    Nf::T
    kf::Int
    Sinv::Matrix{T}
    Sfinv::Matrix{T}
    # scratch, reused across objective evaluations
    A::Matrix{T}
    S::Matrix{T}
    Qc::Vector{Matrix{T}}
    h::Vector{T}
    Bu::Matrix{T}
    hf::Vector{T}
    Theta::Vector{Matrix{T}}
    Psi::Matrix{T}
    R::Matrix{T}
    Rf::Matrix{T}
end

function _HamMStepCtx(
    hs::HamiltonianSufficientStatistics{T}, sm::HamiltonianStateModel{T}, profile::Bool
) where {T<:Real}
    pack = _HamPack(sm)
    n, d, m, K = pack.n, pack.d, pack.m, pack.K
    reg = d + 1 + m
    N = sum(hs.nk)
    Nf = sm.terminal ? hs.term_n : zero(T)
    kf = (sm.terminal && !isempty(sm.schedule)) ? sm.schedule[end] : 1
    Sinv = profile ? zeros(T, d, d) : Matrix(inv(PDMat(Symmetrize!(Matrix{T}(sm.Σ)))))
    Sfinv = if profile || !sm.terminal
        zeros(T, n, n)
    else
        Matrix(inv(PDMat(Symmetrize!(Matrix{T}(sm.Σf)))))
    end
    return _HamMStepCtx{T,typeof(hs),typeof(sm)}(
        pack,
        hs,
        sm,
        profile,
        N,
        Nf,
        kf,
        Sinv,
        Sfinv,
        Matrix{T}(undef, n, n),
        Matrix{T}(undef, n, n),
        [Matrix{T}(undef, n, n) for _ in 1:K],
        Vector{T}(undef, d),
        Matrix{T}(undef, d, m),
        Vector{T}(undef, n),
        [zeros(T, d, reg) for _ in 1:K],
        zeros(T, n, d + 1),
        Matrix{T}(undef, d, d),
        Matrix{T}(undef, n, n),
    )
end

"""
    _ham_pack!(θ, ctx)

Write the model's current structural parameters into the flat vector `θ`, which
is where L-BFGS starts. Symmetric blocks are packed as full matrices: their
gradients come back exactly symmetric, so a symmetric start stays symmetric
through every L-BFGS iterate without any `vech` bookkeeping.
"""
function _ham_pack!(θ::AbstractVector{T}, ctx::_HamMStepCtx{T}) where {T<:Real}
    p, sm = ctx.pack, ctx.sm
    isempty(p.iA) || copyto!(view(θ, p.iA), vec(sm.A))
    isempty(p.iS) || copyto!(view(θ, p.iS), vec(sm.S))
    for k in 1:(p.K)
        isempty(p.iQ[k]) || copyto!(view(θ, p.iQ[k]), vec(sm.Qc[k]))
    end
    isempty(p.ih) || copyto!(view(θ, p.ih), sm.h)
    isempty(p.iB) || copyto!(view(θ, p.iB), vec(sm.Bu))
    isempty(p.ihf) || copyto!(view(θ, p.ihf), sm.hf)
    return θ
end

"""
    _ham_unpack!(ctx, θ)

Fill the context's scratch parameters from `θ`, taking any frozen block from the
model instead. Symmetric blocks are symmetrized on the way in so an accumulated
rounding asymmetry can never make `𝓔` leave the Hamiltonian form.
"""
function _ham_unpack!(ctx::_HamMStepCtx{T}, θ::AbstractVector{T}) where {T<:Real}
    p, sm = ctx.pack, ctx.sm
    n = p.n
    if isempty(p.iA)
        copyto!(ctx.A, sm.A)
    else
        copyto!(ctx.A, reshape(view(θ, p.iA), n, n))
    end
    if isempty(p.iS)
        copyto!(ctx.S, sm.S)
    else
        copyto!(ctx.S, reshape(view(θ, p.iS), n, n))
        Symmetrize!(ctx.S)
    end
    for k in 1:(p.K)
        if isempty(p.iQ[k])
            copyto!(ctx.Qc[k], sm.Qc[k])
        else
            copyto!(ctx.Qc[k], reshape(view(θ, p.iQ[k]), n, n))
            Symmetrize!(ctx.Qc[k])
        end
    end
    isempty(p.ih) ? copyto!(ctx.h, sm.h) : copyto!(ctx.h, view(θ, p.ih))
    if p.m > 0
        if isempty(p.iB)
            copyto!(ctx.Bu, sm.Bu)
        else
            copyto!(ctx.Bu, reshape(view(θ, p.iB), p.d, p.m))
        end
    end
    isempty(p.ihf) ? copyto!(ctx.hf, sm.hf) : copyto!(ctx.hf, view(θ, p.ihf))
    return ctx
end

"""
    _ham_assemble!(ctx)

Build `Θ_k = [𝓔_k  h  Bu]` for every regime and, when there is a terminal
factor, `Ψ = [−Q_f  I  −h_f]`, from the unpacked scratch parameters.
"""
function _ham_assemble!(ctx::_HamMStepCtx{T}) where {T<:Real}
    p = ctx.pack
    n, d, m = p.n, p.d, p.m
    xr, lr = 1:n, (n + 1):d
    for k in 1:(p.K)
        Th = ctx.Theta[k]
        @views begin
            Th[xr, xr] .= ctx.A
            Th[xr, lr] .= .-ctx.S
            Th[lr, xr] .= ctx.Qc[k]
            Th[lr, lr] .= transpose(ctx.A)
            Th[:, d + 1] .= ctx.h
            m > 0 && (Th[:, (d + 2):(d + 1 + m)] .= ctx.Bu)
        end
    end
    if ctx.sm.terminal
        @views begin
            ctx.Psi[:, xr] .= .-ctx.Qc[ctx.kf]
            fill!(ctx.Psi[:, lr], zero(T))
            for i in 1:n
                ctx.Psi[i, n + i] = one(T)
            end
            ctx.Psi[:, d + 1] .= .-ctx.hf
        end
    end
    return ctx
end

"""
    _ham_residuals!(ctx)

Residual scatters at the current scratch parameters:
`R = Σ_k (Yv − Θ_k Xv_kᵀ − Xv_k Θ_kᵀ + Θ_k Zw_k Θ_kᵀ)` and, for the terminal
factor, `R_f = Ψ Ω Ψᵀ`. Both are symmetrized: they are positive semi-definite by
construction, and the factorizations downstream need that exactly.
"""
function _ham_residuals!(ctx::_HamMStepCtx{T}) where {T<:Real}
    hs = ctx.hs
    copyto!(ctx.R, hs.Yv)
    for k in 1:(ctx.pack.K)
        Th = ctx.Theta[k]
        TX = Th * transpose(hs.Xv[k])
        ctx.R .-= TX
        ctx.R .-= transpose(TX)
        ctx.R .+= Th * hs.Zw[k] * transpose(Th)
    end
    Symmetrize!(ctx.R)
    if ctx.sm.terminal
        mul!(ctx.Rf, ctx.Psi * hs.Omega, transpose(ctx.Psi))
        Symmetrize!(ctx.Rf)
    end
    return ctx
end

"""
    _ham_fg!(grad, θ, ctx) -> objective

The structural M-step objective and its gradient. Minimizing

```math
g(\\theta) = \\tfrac{N}{2}\\log\\det R(\\theta)
           + \\tfrac{N_f}{2}\\log\\det R_f(\\theta)
           - N\\log|\\det A|
```

(the `Σ`-profiled case; with `Σ` frozen the two `log det` terms become
`½ tr(Σ⁻¹R)` and `½ tr(Σ_f⁻¹R_f)`) over `θ = (A, S, Q_{1:K}, h, B_u, h_f)`.

The elementwise gradient with respect to a regime's stacked regression is
`G_k = W (Θ_k Z_k − X_k)` with `W = N R⁻¹` profiled and `W = Σ⁻¹` frozen, and
chains onto the structural blocks by

- `∂/∂A   = Σ_k (G_k[x,x] + G_k[λ,λ]ᵀ) − N A⁻ᵀ`  (`A` sits in `𝓔` twice, once
  transposed — and the `A⁻ᵀ` is the Jacobian term)
- `∂/∂S   = −Σ_k G_k[x,λ]`, symmetrized
- `∂/∂Q_k = G_k[λ,x]`, symmetrized, plus the terminal factor's share for `k = k_f`

Returns `Inf` (and leaves `grad` alone) at a `θ` where a residual scatter is not
positive definite or `A` is singular, which the line search reads as a rejected
step.
"""
function _ham_fg!(
    grad::Union{Nothing,AbstractVector{T}}, θ::AbstractVector{T}, ctx::_HamMStepCtx{T}
) where {T<:Real}
    p = ctx.pack
    n, d, m, K = p.n, p.d, p.m, p.K
    xr, lr = 1:n, (n + 1):d

    #=
    A rejected step must leave `grad` defined, not stale: L-BFGS's line search
    may ask for the gradient at a point whose objective came back infinite, and
    a zero gradient there is the honest "no information" answer.
    =#
    grad === nothing || fill!(grad, zero(T))

    _ham_unpack!(ctx, θ)
    F = lu(ctx.A; check=false)
    issuccess(F) || return T(Inf)
    logdetA, _ = logabsdet(F)
    isfinite(logdetA) || return T(Inf)

    _ham_assemble!(ctx)
    _ham_residuals!(ctx)

    fval = -ctx.N * T(logdetA)
    W = ctx.Sinv
    if ctx.profile
        chol = cholesky(Symmetric(ctx.R); check=false)
        issuccess(chol) || return T(Inf)
        fval += T(0.5) * ctx.N * logdet(chol)
        W = ctx.N .* Matrix(inv(chol))
    else
        fval += T(0.5) * dot(ctx.Sinv, ctx.R)
    end

    Wf = ctx.Sfinv
    if ctx.sm.terminal
        if ctx.profile
            cholf = cholesky(Symmetric(ctx.Rf); check=false)
            issuccess(cholf) || return T(Inf)
            fval += T(0.5) * ctx.Nf * logdet(cholf)
            Wf = ctx.Nf .* Matrix(inv(cholf))
        else
            fval += T(0.5) * dot(ctx.Sfinv, ctx.Rf)
        end
    end

    grad === nothing && return fval

    dA = zeros(T, n, n)
    dS = zeros(T, n, n)
    dh = zeros(T, d)
    dB = zeros(T, d, m)
    for k in 1:K
        Gk = W * (ctx.Theta[k] * ctx.hs.Zw[k] .- ctx.hs.Xv[k])
        @views begin
            dA .+= Gk[xr, xr]
            dA .+= transpose(Gk[lr, lr])
            dS .-= Gk[xr, lr]
            if !isempty(p.iQ[k])
                dQ = Gk[lr, xr]
                Gq = view(grad, p.iQ[k])
                copyto!(Gq, vec(T(0.5) .* (dQ .+ transpose(dQ))))
            end
            dh .+= Gk[:, d + 1]
            m > 0 && (dB .+= Gk[:, (d + 2):(d + 1 + m)])
        end
    end

    if ctx.sm.terminal
        # ∂/∂Ψ of the terminal term; Ψ = [−Q_f  I  −h_f].
        GP = Wf * (ctx.Psi * ctx.hs.Omega)
        @views begin
            if !isempty(p.iQ[ctx.kf])
                dQf = .-GP[:, xr]
                Gq = reshape(view(grad, p.iQ[ctx.kf]), n, n)
                Gq .+= T(0.5) .* (dQf .+ transpose(dQf))
            end
            isempty(p.ihf) || (view(grad, p.ihf) .= .-GP[:, d + 1])
        end
    end

    # Jacobian term: ∂(−N log|det A|)/∂A = −N A⁻ᵀ.
    dA .-= ctx.N .* transpose(inv(F))

    isempty(p.iA) || copyto!(view(grad, p.iA), vec(dA))
    isempty(p.iS) || copyto!(view(grad, p.iS), vec(T(0.5) .* (dS .+ transpose(dS))))
    isempty(p.ih) || copyto!(view(grad, p.ih), dh)
    isempty(p.iB) || copyto!(view(grad, p.iB), vec(dB))

    return fval
end

"""
    _ham_writeback!(sm, ctx, θ)

Copy the optimized structural parameters out of `θ` into the model, leaving
frozen blocks untouched.
"""
function _ham_writeback!(
    sm::HamiltonianStateModel{T}, ctx::_HamMStepCtx{T}, θ::AbstractVector{T}
) where {T<:Real}
    _ham_unpack!(ctx, θ)
    p = ctx.pack
    isempty(p.iA) || copyto!(sm.A, ctx.A)
    isempty(p.iS) || copyto!(sm.S, ctx.S)
    for k in 1:(p.K)
        isempty(p.iQ[k]) || copyto!(sm.Qc[k], ctx.Qc[k])
    end
    isempty(p.ih) || copyto!(sm.h, ctx.h)
    isempty(p.iB) || copyto!(sm.Bu, ctx.Bu)
    isempty(p.ihf) || copyto!(sm.hf, ctx.hf)
    return sm
end

"""
    _ham_structure_mstep!(lds, hs) -> ctx

Optimize the structural parameters, warm started at their current values.

L-BFGS on the objective above is a *generalized* M-step: EM only needs the
objective not to get worse, so the result is accepted only when it beats the
starting point. That also makes a failed inner solve harmless — the parameters
simply do not move that iteration.

Returns the context, whose scratch holds the residual scatters at the accepted
parameters, so the noise update can read them without recomputing.
"""
function _ham_structure_mstep!(
    lds::LinearDynamicalSystem{T,S,O}, hs::HamiltonianSufficientStatistics{T}
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    sm = lds.state_model
    profile = lds.fit_bool[4]
    ctx = _HamMStepCtx(hs, sm, profile)

    θ0 = zeros(T, ctx.pack.np)
    _ham_pack!(θ0, ctx)

    if lds.fit_bool[3] && ctx.pack.np > 0
        f0 = _ham_fg!(nothing, θ0, ctx)
        if isfinite(f0)
            f_obj(θ) = _ham_fg!(nothing, θ, ctx)
            function g_obj!(G, θ)
                _ham_fg!(G, θ, ctx)
                return G
            end
            opts = Optim.Options(;
                x_abstol=1e-10, g_abstol=1e-9, f_reltol=1e-12, iterations=sm.mstep_iters
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
                _ham_writeback!(sm, ctx, θ1)
                copyto!(θ0, θ1)
            end
        end
    end

    # Leave the scratch (and hence `ctx.R` / `ctx.Rf`) at the accepted parameters.
    _ham_unpack!(ctx, θ0)
    _ham_assemble!(ctx)
    _ham_residuals!(ctx)
    return ctx
end

"""
    _ham_noise_mstep!(sm, ctx)

`Σ = R/N` and `Σf = R_f/N_f` — the closed-form maximizers given the structural
parameters, which is exactly what profiling them out of the objective assumed.
"""
function _ham_noise_mstep!(
    sm::HamiltonianStateModel{T}, ctx::_HamMStepCtx{T}
) where {T<:Real}
    if ctx.N > zero(T)
        copyto!(sm.Σ, ctx.R)
        sm.Σ ./= ctx.N
        Symmetrize!(sm.Σ)
    end
    if sm.terminal && ctx.Nf > zero(T)
        copyto!(sm.Σf, ctx.Rf)
        sm.Σf ./= ctx.Nf
        Symmetrize!(sm.Σf)
    end
    return sm
end

"""
    _ham_state_mstep!(lds, hs, sws)

The state half of the M-step, shared by every emission model.
"""
function _ham_state_mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    hs::HamiltonianSufficientStatistics{T},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:HamiltonianStateModel{T},O<:AbstractObservationModel{T}}
    base = _state_suf(hs.base)
    update_initial_state_mean!(lds, base)
    update_initial_state_covariance!(lds, base, sws)

    _fill_mixed_blocks!(hs, lds.state_model)
    ctx = _ham_structure_mstep!(lds, hs)
    lds.fit_bool[4] && _ham_noise_mstep!(lds.state_model, ctx)
    refresh!(lds.state_model)
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

Requires [`_fill_mixed_blocks!`](@ref) to have run for the current statistics,
which `elbo!` arranges.
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

    # Transition term, in mixed coordinates, at the model's current parameters.
    ctx = _HamMStepCtx(hs, sm, false)
    θ = zeros(T, ctx.pack.np)
    _ham_pack!(θ, ctx)
    _ham_unpack!(ctx, θ)
    _ham_assemble!(ctx)
    _ham_residuals!(ctx)

    if ctx.N > zero(T)
        Σ_PD = PDMat(Symmetrize!(Matrix{T}(sm.Σ)))
        Q_val +=
            T(-0.5) * (
                ctx.N * (T(d) * log2π + logdet(Σ_PD) - T(2) * sm.cache.logabsdetA) +
                dot(ctx.Sinv, ctx.R)
            )
    end

    if sm.terminal && ctx.Nf > zero(T)
        Σf_PD = PDMat(Symmetrize!(Matrix{T}(sm.Σf)))
        Q_val +=
            T(-0.5) * (ctx.Nf * (T(n) * log2π + logdet(Σf_PD)) + dot(ctx.Sfinv, ctx.Rf))
    end

    return Q_val
end
