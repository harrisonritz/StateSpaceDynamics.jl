# =============================================================================
# Information-form Kalman filter + RTS smoother for Gaussian LDS.
#
# Activated when `lds.kalman_filter == true`. Ported from StateSpaceAnalysis.
#
# Key design choices:
#
# * The covariance forward-backward pass does **not** depend on observations.
#   It is run **once per E-step** and its output (predicted/filtered/smoothed
#   covariances, gains, and per-step entropy contribution) is shared across all
#   trials. This is the main speedup over the block-tridiagonal path when
#   `ntrials` is large.
#
# * Optional input support: `x_{t+1} = A·x_t + b + B·u_t + ε` and
#   `x_1 ~ N(x0 + B0·u0, P0)`. Activated when `state_model.B` / `state_model.B0`
#   are supplied (not `nothing`). Inputs flow in through kwargs `u`, `u0`.
#
# * Covariances are stored as `PDMat` so Cholesky factors stay cached; all
#   stabilization goes through `tol_PD` (eigen-floor) rather than
#   `make_posdef!` / `stabilize_covariance_matrix` (absolute floor), which
#   preserves conditioning when the covariance has large dynamic range.
#
# * Reference-sharing: each trial's `FilterSmooth.p_smooth` /
#   `FilterSmooth.p_smooth_tt1` is set to alias the shared 3-D arrays in
#   `KalmanWorkspace`. `sufficient_statistics!` and `Q_state!`/`Q_obs!` only
#   **read** these fields, so aliasing is safe — but mutating through any one
#   trial's view would corrupt all trials.
# =============================================================================

"""
    precompute_kalman_constants!(kws::KalmanWorkspace, lds; tol=1e-6)

Refresh the cached `PDMat` wrappers of `Q`, `R`, `P0` (applying `tol_PD`) and
the derived constants `CiR = C' R^{-1}` (D × p) and `CiRC = C' R^{-1} C` (D × D).
Called once at the start of each E-step.
"""
function precompute_kalman_constants!(
    kws::KalmanWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O};
    tol::Real=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    C = lds.obs_model.C

    kws.Q_PD[]  = tol_PD(lds.state_model.Q;  tol=tol)
    kws.R_PD[]  = tol_PD(lds.obs_model.R;    tol=tol)
    kws.P0_PD[] = tol_PD(lds.state_model.P0; tol=tol)

    # CiR = C' * R^{-1}  (D × p).  R_PD \ C solves R * X = C via cached Cholesky.
    RinvC = kws.R_PD[] \ C               # p × D
    copyto!(kws.CiR, RinvC')             # D × p

    # CiRC = CiR * C  (D × D, symmetric)
    mul!(kws.CiRC, kws.CiR, C)
    Symmetrize!(kws.CiRC)

    return nothing
end

"""
    covariance_forward_backward!(kws::KalmanWorkspace, lds; tol=1e-6)

Run the covariance-only information-form forward pass and the RTS backward
pass. Populates `kws.pred_cov`, `kws.filt_cov`, `kws.pred_icov`,
`kws.smooth_cov`, `kws.G`, `kws.p_smooth_shared`, `kws.p_smooth_tt1_shared`,
and `kws.shared_entropy` (the joint Gaussian entropy of `p(x_{1:T}|y)` — a
scalar shared across all trials).

The joint entropy is computed via the Markov backward factorization

```
H(x_{1:T}|y) = H(x_T|y) + Σ_{t=1}^{T-1} H(x_t | x_{t+1}, y)
```

using the backward-conditional covariance
`Σ_{t|x_{t+1},y} = Σ_{t|t} − G_t · Σ_{t+1|t} · G_t'`, which is cheap to form
alongside the standard RTS recursion.

Must be called after `precompute_kalman_constants!`.
"""
function covariance_forward_backward!(
    kws::KalmanWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O};
    tol::Real=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    Tsteps = kws.tsteps
    D      = kws.latent_dim
    A      = lds.state_model.A
    Q_mat  = kws.Q_PD[].mat
    CiRC   = kws.CiRC
    tmp1   = kws.cov_tmp1   # D×D scratch — written freely, may be corrupted by tol_PD
    tmp2   = kws.cov_tmp2   # D×D scratch — ditto

    # -------- Forward pass --------
    # t = 1: seed from prior P0.
    kws.pred_cov[1] = kws.P0_PD[]

    # pred_icov_mat[:,:,1] = P0^{-1} via Cholesky (no allocation)
    pi1 = @view kws.pred_icov_mat[:, :, 1]
    fill!(pi1, zero(T))
    @inbounds for i in 1:D; pi1[i, i] = one(T); end
    ldiv!(kws.P0_PD[].chol, pi1)

    # filt_prec[1] = pred_icov[1] + CiRC  →  filt_cov[1] = inv(filt_prec[1])
    tmp1 .= pi1 .+ CiRC                              # no allocation
    Symmetrize!(tmp1)
    kws.filt_cov[1] = inv(tol_PD(Symmetric(tmp1); tol=tol))  # tmp1 corrupted by eigen!

    @inbounds for t in 2:Tsteps
        # pred_cov[t] = A · filt_cov[t-1] · A' + Q
        # X_A_Xt(filt_cov[t-1], A) = A * filt_cov * A' via trmm + syrk
        copyto!(tmp1, X_A_Xt(kws.filt_cov[t - 1], A))
        tmp1 .+= Q_mat
        Symmetrize!(tmp1)
        kws.pred_cov[t] = tol_PD(Symmetric(tmp1); tol=tol)   # tmp1 corrupted by eigen!

        # pred_icov_mat[:,:,t] = pred_cov[t]^{-1}  (Cholesky back-substitution)
        pit = @view kws.pred_icov_mat[:, :, t]
        fill!(pit, zero(T))
        @inbounds for i in 1:D; pit[i, i] = one(T); end
        ldiv!(kws.pred_cov[t].chol, pit)

        # filt_cov[t] = inv(pred_icov[t] + CiRC)
        tmp1 .= pit .+ CiRC
        Symmetrize!(tmp1)
        kws.filt_cov[t] = inv(tol_PD(Symmetric(tmp1); tol=tol))  # tmp1 corrupted
    end

    # -------- Backward (RTS) pass --------
    kws.smooth_cov[Tsteps] = kws.filt_cov[Tsteps]

    fill!(kws.p_smooth_shared, zero(T))
    fill!(kws.p_smooth_tt1_shared, zero(T))
    @views kws.p_smooth_shared[:, :, Tsteps] .= kws.smooth_cov[Tsteps].mat

    ent_logdet = logdet(kws.smooth_cov[Tsteps])

    @inbounds for t in (Tsteps - 1):-1:1
        filt_t_mat = kws.filt_cov[t].mat
        pi_t1      = @view kws.pred_icov_mat[:, :, t + 1]   # P_pred[t+1]^{-1}

        # G_t = filt_cov[t] · A' · pred_icov[t+1]
        # Write into tmp2 (concrete Matrix{T}) for type-stable X_A_Xt calls below.
        mul!(tmp1, filt_t_mat, A')    # tmp1 = P_filt · A'
        mul!(tmp2, tmp1, pi_t1)       # tmp2 = G_t
        @views kws.G[:, :, t] .= tmp2 # store for smooth_mean backward pass

        # Both triple products use X_A_Xt (trmm + syrk, ~2x faster than gemm×2 at large D).
        # tmp2 holds G_t as a concrete Matrix{T} — avoids SubArray JET inference issues.
        # GpGt is shared by smooth_cov and back_condP — compute once, use twice.
        GpGt = X_A_Xt(kws.pred_cov[t + 1], tmp2)    # G_t · P_pred[t+1] · G_t'
        GsGt = X_A_Xt(kws.smooth_cov[t + 1], tmp2)  # G_t · P_smooth[t+1] · G_t'

        # smooth_cov[t] = filt_cov[t] + G_t · (smooth_t+1 − pred_t+1) · G_t'
        tmp1 .= filt_t_mat .+ GsGt .- GpGt
        Symmetrize!(tmp1)
        kws.smooth_cov[t] = tol_PD(Symmetric(tmp1); tol=tol)  # tmp1 corrupted
        @views kws.p_smooth_shared[:, :, t] .= kws.smooth_cov[t].mat

        # Backward-conditional cov for entropy chain-rule:
        # Sigma_{t|x_{t+1},y} = filt_cov[t] - G_t · pred_cov[t+1] · G_t'
        tmp2 .= filt_t_mat .- GpGt   # reuse GpGt; overwrites tmp2 (G_t no longer needed)
        Symmetrize!(tmp2)
        back_condP = tol_PD(Symmetric(tmp2); tol=tol)  # tmp2 corrupted
        ent_logdet += logdet(back_condP)
    end

    # Lag-1 cross-covariance: Cov(x_t, x_{t-1} | y) = smooth_cov[t] · G_{t-1}'
    @inbounds @views for t in 2:Tsteps
        mul!(kws.p_smooth_tt1_shared[:, :, t], kws.smooth_cov[t].mat, kws.G[:, :, t - 1]')
    end

    kws.shared_entropy[] =
        T(0.5) * (T(Tsteps) * T(D) * (one(T) + log(T(2π))) + ent_logdet)

    return nothing
end

"""
    _precompute_trial_inputs!(kws, lds, y; u=nothing, u0=nothing)

Fill `kws.y_minus_d`, `kws.CiRY`, and `kws.Bu` for every trial. `y_minus_d[:,t,n]
= y[:,t,n] − d`, `CiRY[:,t,n] = CiR · y_minus_d[:,t,n]`, and `Bu[:,t,n] =
B · u[:,t,n]` (or zero if `B === nothing`).

Input validation:
- `B !== nothing` requires `u` to be supplied with shape `(u_dim, Tsteps, ntrials)`.
- `B === nothing` with non-`nothing` `u` → error (silent-dropping is too easy to
  miss).
- `B0` / `u0` handled identically (see `smooth!`).
"""
function _precompute_trial_inputs!(
    kws::KalmanWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3};
    u::Union{Nothing,AbstractArray{T,3}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    d = lds.obs_model.d
    B = lds.state_model.B
    CiR = kws.CiR

    # Validate u / B consistency
    if B !== nothing
        u === nothing && throw(
            DimensionMismatchError("u (required because B !== nothing)", 1, 0),
        )
        if size(u, 1) != size(B, 2)
            throw(DimensionMismatchError("u rows vs B cols", size(B, 2), size(u, 1)))
        end
        if size(u, 2) != kws.tsteps || size(u, 3) != kws.ntrials
            throw(
                DimensionMismatchError(
                    "u shape (u_dim, T, ntrials)",
                    (size(B, 2), kws.tsteps, kws.ntrials),
                    size(u),
                ),
            )
        end
    elseif u !== nothing
        throw(
            ArgumentError(
                "u was supplied but state_model.B is nothing; set B before passing u",
            ),
        )
    end

    ntrials = kws.ntrials
    Tsteps  = kws.tsteps

    @inbounds @views for n in 1:ntrials
        for t in 1:Tsteps
            ytn = y[:, t, n]
            ymd = kws.y_minus_d[:, t, n]
            @. ymd = ytn - d
            mul!(kws.CiRY[:, t, n], CiR, ymd)
        end
    end

    if B === nothing
        fill!(kws.Bu, zero(T))
    else
        @inbounds @views for n in 1:ntrials
            mul!(kws.Bu[:, :, n], B, u[:, :, n])
        end
    end

    return nothing
end

"""
    _filter_mean_trial!(kws, lds, n; x0_eff)

Run the information-form mean-only forward pass for trial `n`, using the
shared covariance pass already populated in `kws`. `x0_eff` is the effective
initial mean for this trial (`x0 + B0·u0[:,n]` if `B0 !== nothing`, otherwise
`x0`).
"""
function _filter_mean_trial!(
    kws::KalmanWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    n::Int,
    x0_eff::AbstractVector{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    Tsteps = kws.tsteps
    A      = lds.state_model.A
    b      = lds.state_model.b

    @views begin
        pred     = kws.pred_mean[:, :, n]
        filt     = kws.filt_mean[:, :, n]
        info_buf = kws.mean_tmp[:, n]   # pre-allocated D-vector for this trial

        # t = 1
        pred[:, 1] .= x0_eff
        mul!(info_buf, kws.pred_icov_mat[:, :, 1], pred[:, 1])
        info_buf .+= kws.CiRY[:, 1, n]
        mul!(filt[:, 1], kws.filt_cov[1].mat, info_buf)

        @inbounds for t in 2:Tsteps
            # pred[:, t] = A · filt[:, t-1] + b + Bu[:, t-1, n]
            mul!(pred[:, t], A, filt[:, t - 1])
            pred[:, t] .+= b
            pred[:, t] .+= kws.Bu[:, t - 1, n]

            mul!(info_buf, kws.pred_icov_mat[:, :, t], pred[:, t])
            info_buf .+= kws.CiRY[:, t, n]
            mul!(filt[:, t], kws.filt_cov[t].mat, info_buf)
        end
    end

    return nothing
end

"""
    _smooth_mean_trial!(kws, n)

Backward RTS mean pass for trial `n`, using the shared gains `kws.G`.
"""
function _smooth_mean_trial!(kws::KalmanWorkspace{T}, n::Int) where {T<:Real}
    Tsteps = kws.tsteps

    @views begin
        smooth_m = kws.smooth_mean[:, :, n]
        filt_m   = kws.filt_mean[:, :, n]
        pred_m   = kws.pred_mean[:, :, n]

        smooth_m[:, Tsteps] .= filt_m[:, Tsteps]

        @inbounds for t in (Tsteps - 1):-1:1
            # smooth[:, t] = filt[:, t] + G[:, :, t] * (smooth[:, t+1] - pred[:, t+1])
            delta = smooth_m[:, t + 1] .- pred_m[:, t + 1]
            smooth_m[:, t] .= filt_m[:, t]
            mul!(smooth_m[:, t], kws.G[:, :, t], delta, one(T), one(T))
        end
    end

    return nothing
end

"""
    smooth!(lds, tfs, y, kws::KalmanWorkspace; u=nothing, u0=nothing)

Kalman/RTS E-step for a Gaussian LDS with `lds.kalman_filter == true`. Runs a
single covariance forward-backward pass (shared across all trials) and
per-trial information-form mean passes.

Each trial's `FilterSmooth` gets `p_smooth` and `p_smooth_tt1` set to alias the
shared 3-D arrays in `kws`, and its `entropy` set to the shared joint-entropy
scalar. Mean-field fields (`x_smooth`) are filled per-trial.

# Arguments
- `lds`: Gaussian LDS with `kalman_filter == true`.
- `tfs`: `TrialFilterSmooth` with one `FilterSmooth` per trial.
- `y`: Observations, size `(obs_dim, Tsteps, ntrials)`.
- `kws`: Preallocated `KalmanWorkspace`.

# Keyword arguments
- `u`: Input sequence, size `(u_dim, Tsteps, ntrials)`. Required iff
    `lds.state_model.B !== nothing`.
- `u0`: Initial-state inputs, size `(u0_dim, ntrials)`. Required iff
    `lds.state_model.B0 !== nothing`.
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    kws::KalmanWorkspace{T};
    u::Union{Nothing,AbstractArray{T,3}}=nothing,
    u0::Union{Nothing,AbstractMatrix{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    ntrials = size(y, 3)
    ntrials == kws.ntrials ||
        throw(DimensionMismatchError("y ntrials vs kws.ntrials", kws.ntrials, ntrials))

    # 1. Refresh cached PDMats and CiR/CiRC.
    precompute_kalman_constants!(kws, lds)

    # 2. Shared covariance forward-backward (once, reused across trials).
    covariance_forward_backward!(kws, lds)

    # 3. Precompute per-trial Bu and CiRY.
    _precompute_trial_inputs!(kws, lds, y; u=u)

    # 4. Validate / stage B0 · u0.
    B0 = lds.state_model.B0
    if B0 !== nothing
        u0 === nothing && throw(
            DimensionMismatchError("u0 (required because B0 !== nothing)", 1, 0),
        )
        if size(u0, 1) != size(B0, 2)
            throw(DimensionMismatchError("u0 rows vs B0 cols", size(B0, 2), size(u0, 1)))
        end
        if size(u0, 2) != ntrials
            throw(
                DimensionMismatchError(
                    "u0 trials", ntrials, size(u0, 2)
                ),
            )
        end
    elseif u0 !== nothing
        throw(
            ArgumentError(
                "u0 was supplied but state_model.B0 is nothing; set B0 before passing u0",
            ),
        )
    end

    x0 = lds.state_model.x0

    # 5. Per-trial mean passes. Threads write to disjoint (D, T) slices, so no locking.
    if ntrials == 1
        x0_eff = B0 === nothing ? copy(x0) : x0 .+ B0 * view(u0, :, 1)
        _filter_mean_trial!(kws, lds, 1, x0_eff)
        _smooth_mean_trial!(kws, 1)
    else
        @threads for n in 1:ntrials
            x0_eff = B0 === nothing ? copy(x0) : x0 .+ B0 * view(u0, :, n)
            _filter_mean_trial!(kws, lds, n, x0_eff)
            _smooth_mean_trial!(kws, n)
        end
    end

    # 6. Wire results into each FilterSmooth. `p_smooth` / `p_smooth_tt1` alias the
    #    shared workspace arrays — safe because downstream code only reads them.
    shared_ent = kws.shared_entropy[]
    @views for n in 1:ntrials
        fs = tfs[n]
        copyto!(fs.x_smooth, kws.smooth_mean[:, :, n])
        fs.p_smooth      = kws.p_smooth_shared
        fs.p_smooth_tt1  = kws.p_smooth_tt1_shared
        fs.entropy       = shared_ent
    end

    return tfs
end

"""
    estep!(lds, tfs, y, kws::KalmanWorkspace, sws_pool; u=nothing, u0=nothing)

Kalman-path E-step. Runs `smooth!` via the Kalman/RTS backend, computes
sufficient statistics, and calls `calculate_elbo` using the companion
`SmoothWorkspace` pool (which still owns the Cholesky buffers used by
`Q_state!`/`Q_obs!`).
"""
function estep!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    kws::KalmanWorkspace{T},
    sws_pool::Vector{SmoothWorkspace{T}};
    u::Union{Nothing,AbstractArray{T,3}}=nothing,
    u0::Union{Nothing,AbstractMatrix{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    smooth!(lds, tfs, y, kws; u=u, u0=u0)
    sufficient_statistics!(tfs)
    return calculate_elbo(lds, tfs, y, sws_pool)
end

# =============================================================================
# Input-aware M-step updates for B, B0.
#
# When `lds.state_model.B !== nothing`, `update_A_b!` / `update_Q!` need to
# include the `B·u_{t-1}` term in the regressor/innovation. The joint
# closed-form solution is obtained by augmenting the regressor
# `[x_{t-1}; 1]` with `u_{t-1}`; the math is unchanged, only matrix shapes
# grow. Analogously, `update_initial_state!` jointly fits `x0` and `B0` when
# `B0 !== nothing`, by augmenting `[1]` with `u0[:, n]`.
#
# These are invoked from the Kalman-path `mstep!` only (gated on `B !== nothing`
# and `B0 !== nothing`). The non-Kalman path is untouched.
# =============================================================================

"""
    _update_A_b_B!(lds, tfs, w, u)

Jointly update `A`, `b`, `B` by solving
`[A | b | B]' = Szz \\ Sxz'` where the regressor is `[x_{t-1}; 1; u_{t-1}]`.
Requires `lds.state_model.B !== nothing` and `u !== nothing`.
"""
function _update_A_b_B!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}},
    u::AbstractArray{T,3},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    D = lds.latent_dim
    udim = size(u, 1)
    ntrials = length(tfs)
    K = D + 1 + udim     # regressor length

    Sxz = zeros(T, D, K)
    Szz = zeros(T, K, K)

    @views for trial in 1:ntrials
        fs      = tfs[trial]
        tsteps  = size(fs.E_z, 2)
        weights = w === nothing ? nothing : w[trial]

        for t in 2:tsteps
            wt  = weights === nothing ? one(T) : weights[t]
            ut1 = u[:, t - 1, trial]          # u_{t-1}

            # Sxz = E[x_t [x_{t-1}; 1; u_{t-1}]']
            Sxz[:, 1:D]       .+= wt .* fs.E_zz_prev[:, :, t]
            Sxz[:, D + 1]     .+= wt .* fs.E_z[:, t]
            mul!(Sxz[:, (D + 2):K], fs.E_z[:, t:t], ut1', wt, one(T))   # +wt * E[x_t] * u'

            # Szz = E[[x_{t-1}; 1; u_{t-1}] [x_{t-1}; 1; u_{t-1}]']
            Szz[1:D, 1:D]         .+= wt .* fs.E_zz[:, :, t - 1]
            Szz[1:D, D + 1]       .+= wt .* fs.E_z[:, t - 1]
            Szz[D + 1, 1:D]       .+= wt .* fs.E_z[:, t - 1]
            Szz[D + 1, D + 1]      += wt

            mul!(Szz[1:D, (D + 2):K], fs.E_z[:, (t - 1):(t - 1)], ut1', wt, one(T))
            mul!(Szz[(D + 2):K, 1:D], ut1, fs.E_z[:, (t - 1):(t - 1)]', wt, one(T))
            Szz[D + 1, (D + 2):K] .+= wt .* ut1
            Szz[(D + 2):K, D + 1] .+= wt .* ut1
            mul!(Szz[(D + 2):K, (D + 2):K], ut1, ut1', wt, one(T))
        end
    end

    F = factorize(Symmetric(Szz))
    ABB = (F \ Sxz')'    # D × K = [A | b | B]

    copyto!(lds.state_model.A, view(ABB, :, 1:D))
    copyto!(lds.state_model.b, view(ABB, :, D + 1))
    if lds.fit_bool[7]
        copyto!(lds.state_model.B, view(ABB, :, (D + 2):K))
    end

    return nothing
end

"""
    _update_Q_B!(lds, tfs, sws, w, u)

Update the process noise covariance `Q` when the dynamics include a
`B·u_{t-1}` input term. Accumulates the innovation covariance
`E[(x_t − A x_{t-1} − b − B u_{t-1})(·)']` and divides by total weight (or
falls back to the IW-MAP update if `Q_prior` is set).
"""
function _update_Q_B!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    sws::SmoothWorkspace{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}},
    u::AbstractArray{T,3},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    D = lds.latent_dim
    A = lds.state_model.A
    b = lds.state_model.b
    B = lds.state_model.B::AbstractMatrix{T}
    ntrials = length(tfs)

    Q_sum = sws.Q_sum
    fill!(Q_sum, zero(T))
    total_weight = zero(T)

    Bu_t = Vector{T}(undef, D)
    innov = Matrix{T}(undef, D, D)
    tmpDD = Matrix{T}(undef, D, D)

    @views for trial in 1:ntrials
        fs      = tfs[trial]
        tsteps  = size(fs.E_z, 2)
        weights = w === nothing ? nothing : w[trial]

        for t in 2:tsteps
            wt    = weights === nothing ? one(T) : weights[t]
            Σt    = fs.E_zz[:, :, t]
            Σtm1  = fs.E_zz[:, :, t - 1]
            Σcr   = fs.E_zz_prev[:, :, t]
            μt    = fs.E_z[:, t]
            μtm1  = fs.E_z[:, t - 1]
            ut1   = u[:, t - 1, trial]

            mul!(Bu_t, B, ut1)             # m_t = B · u_{t-1}
            # innov = E[(x_t − A x_{t-1} − b − m_t)(·)']
            copyto!(innov, Σt)
            mul!(innov, Σcr, A', -one(T), one(T))         # -= Σcr · A'
            mul!(innov, A, Σcr', -one(T), one(T))         # -= A · Σcr'
            mul!(tmpDD, A, Σtm1)
            mul!(innov, tmpDD, A', one(T), one(T))        # += A Σtm1 A'

            mul!(innov, reshape(μt, :, 1), reshape(b, 1, :), -one(T), one(T))
            mul!(innov, reshape(b, :, 1), reshape(μt, 1, :), -one(T), one(T))
            mul!(innov, reshape(μt, :, 1), reshape(Bu_t, 1, :), -one(T), one(T))
            mul!(innov, reshape(Bu_t, :, 1), reshape(μt, 1, :), -one(T), one(T))

            mul!(tmpDD, A, reshape(μtm1, :, 1) * reshape(b, 1, :))   # A · μtm1 · b'
            innov .+= tmpDD
            innov .+= tmpDD'                                # + b · μtm1' · A'

            mul!(tmpDD, A, reshape(μtm1, :, 1) * reshape(Bu_t, 1, :))  # A μtm1 m'
            innov .+= tmpDD
            innov .+= tmpDD'                                # + m μtm1' A'

            mul!(innov, reshape(b, :, 1), reshape(Bu_t, 1, :), one(T), one(T))  # + b m'
            mul!(innov, reshape(Bu_t, :, 1), reshape(b, 1, :), one(T), one(T))  # + m b'
            mul!(innov, reshape(b, :, 1), reshape(b, 1, :), one(T), one(T))     # + b b'
            mul!(innov, reshape(Bu_t, :, 1), reshape(Bu_t, 1, :), one(T), one(T))  # + m m'

            Q_sum .+= wt .* innov
            total_weight += wt
        end
    end

    if lds.state_model.Q_prior === nothing
        Q_sum ./= total_weight
    else
        Ψ, ν = lds.state_model.Q_prior.Ψ, lds.state_model.Q_prior.ν
        copyto!(Q_sum, iw_map(Ψ, ν, Q_sum, total_weight, D))
    end

    copyto!(lds.state_model.Q, Symmetrize!(Q_sum))
    return nothing
end

"""
    _update_initial_state_B0!(lds, tfs, w, u0)

Joint closed-form update for `x0` and `B0` when `B0 !== nothing`. Solves
`[x0 | B0]' = Szz \\ Sxz'` with regressor `[1; u0[:, n]]`, response `E[x_1^n]`,
and then updates `P0` from the centered residuals across trials.
"""
function _update_initial_state_B0!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}},
    u0::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    D = lds.latent_dim
    u0dim = size(u0, 1)
    ntrials = length(tfs)
    K = 1 + u0dim

    Sxz = zeros(T, D, K)
    Szz = zeros(T, K, K)

    @views for n in 1:ntrials
        fs = tfs[n]
        wt = w === nothing ? one(T) : w[n][1]
        u0n = u0[:, n]

        E_x1 = fs.E_z[:, 1]
        Sxz[:, 1]     .+= wt .* E_x1
        mul!(Sxz[:, 2:K], reshape(E_x1, :, 1), u0n', wt, one(T))

        Szz[1, 1]      += wt
        Szz[1, 2:K]   .+= wt .* u0n
        Szz[2:K, 1]   .+= wt .* u0n
        mul!(Szz[2:K, 2:K], u0n, u0n', wt, one(T))
    end

    F = factorize(Symmetric(Szz))
    X = (F \ Sxz')'           # D × K = [x0 | B0]

    if lds.fit_bool[1]
        copyto!(lds.state_model.x0, view(X, :, 1))
    end
    if lds.fit_bool[8]
        copyto!(lds.state_model.B0, view(X, :, 2:K))
    end

    # Update P0 from centered residuals: P0 = (1/N) Σ_n (E_zz_1^n − m_n m_n')
    # where m_n = x0 + B0 u0[:, n].
    if lds.fit_bool[2]
        P0_sum = zeros(T, D, D)
        total_weight = zero(T)
        m_n = Vector{T}(undef, D)
        @views for n in 1:ntrials
            fs = tfs[n]
            wt = w === nothing ? one(T) : w[n][1]
            u0n = u0[:, n]

            copyto!(m_n, lds.state_model.x0)
            mul!(m_n, lds.state_model.B0, u0n, one(T), one(T))

            P0_sum .+= wt .* (fs.E_zz[:, :, 1] - m_n * m_n')
            total_weight += wt
        end

        if lds.state_model.P0_prior === nothing
            P0_sum ./= total_weight
        else
            Ψ, ν = lds.state_model.P0_prior.Ψ, lds.state_model.P0_prior.ν
            copyto!(P0_sum, iw_map(Ψ, ν, P0_sum, total_weight, D))
        end

        copyto!(lds.state_model.P0, Symmetrize!(P0_sum))
    end

    return nothing
end

"""
    mstep!(lds, tfs, y, kws::KalmanWorkspace, sws; u=nothing, u0=nothing, w=nothing)

Kalman-path M-step: dispatches the initial-state, dynamics, and process-noise
updates to their input-aware variants when `B` / `B0` are supplied, and reuses
the existing `update_C_d!` / `update_R!` for the observation model.

`sws` is still a `SmoothWorkspace` — only a single one is needed because the
M-step is sequential, not per-trial.
"""
function mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    kws::KalmanWorkspace{T},
    sws::SmoothWorkspace{T};
    u::Union{Nothing,AbstractArray{T,3}}=nothing,
    u0::Union{Nothing,AbstractMatrix{T}}=nothing,
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    B  = lds.state_model.B
    B0 = lds.state_model.B0

    # Initial state (x0, P0, optionally B0)
    if B0 === nothing
        update_initial_state_mean!(lds, tfs, w)
        update_initial_state_covariance!(lds, tfs, sws, w)
    else
        u0 === nothing && throw(
            ArgumentError("mstep!: state_model.B0 is set but u0 was not supplied"),
        )
        _update_initial_state_B0!(lds, tfs, w, u0)
    end

    # Dynamics (A, b, optionally B) and process noise Q
    if B === nothing
        lds.fit_bool[3] && update_A_b!(lds, tfs, sws, w)
        lds.fit_bool[4] && update_Q!(lds, tfs, sws, w)
    else
        u === nothing && throw(
            ArgumentError("mstep!: state_model.B is set but u was not supplied"),
        )
        u_concrete = u::AbstractArray{T,3}
        lds.fit_bool[3] && _update_A_b_B!(lds, tfs, w, u_concrete)
        lds.fit_bool[4] && _update_Q_B!(lds, tfs, sws, w, u_concrete)
    end

    # Observation (C, d, R) — unchanged
    update_C_d!(lds, tfs, y, sws, w)
    update_R!(lds, tfs, y, sws, w)

    return nothing
end

"""
    _fit_kalman!(lds, y; u, u0, max_iter, tol, progress)

Kalman-path EM driver. Called from the main `fit!` in `gaussian.jl` when
`lds.kalman_filter == true`.
"""
function _fit_kalman!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3};
    u::Union{Nothing,AbstractArray{T,3}}=nothing,
    u0::Union{Nothing,AbstractMatrix{T}}=nothing,
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    eltype(y) === T || error("Observed data must be of type $(T); got $(eltype(y))")

    tsteps = size(y, 2)
    ntrials = size(y, 3)

    kws = KalmanWorkspace(lds, tsteps, ntrials)
    sws_pool = [
        SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, tsteps) for
        _ in 1:Threads.maxthreadid()
    ]

    tfs = initialize_FilterSmooth(lds, tsteps, ntrials)
    prev_elbo = -T(Inf)
    elbos = Vector{T}()
    sizehint!(elbos, max_iter)

    prog = progress ? Progress(max_iter; desc="Fitting LDS (Kalman) via EM...", barlen=50, showspeed=true) : nothing

    for _ in 1:max_iter
        elbo = estep!(lds, tfs, y, kws, sws_pool; u=u, u0=u0)
        mstep!(lds, tfs, y, kws, sws_pool[1]; u=u, u0=u0)

        push!(elbos, elbo)
        progress && prog !== nothing && next!(prog)

        if abs(elbo - prev_elbo) < tol
            progress && prog !== nothing && finish!(prog)
            return elbos
        end
        prev_elbo = elbo
    end

    progress && prog !== nothing && finish!(prog)
    return elbos
end
