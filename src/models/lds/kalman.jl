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

# using BenchmarkTools

"""
    _fit_kalman!(lds, y; u, u0, max_iter, tol, progress)

Kalman-path EM driver. Called from the main `fit!` in `gaussian.jl` when
`lds.kalman_filter == true`.
"""
function _fit_kalman!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3};
    u0::Union{Nothing,AbstractMatrix{T}}=nothing,
    u::Union{Nothing,AbstractArray{T,3}}=nothing,
    d::Union{Nothing,AbstractArray{T,3}}=nothing,
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
    monotonicity_check::Bool=true,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    eltype(y) === T || error("Observed data must be of type $(T); got $(eltype(y))")

    tsteps = size(y, 2)
    ntrials = size(y, 3)

    # format inputs and preallocate workspace + sufficient statistics
    data = format_kf_data!(lds, y, u0, u, d, tsteps, ntrials)
    validate_kalman_inputs(lds, data, ntrials, tsteps)
    kws = KalmanWorkspace(lds, tsteps, ntrials)
    suf = initialize_SufficientStatistics(lds, data, kws) # reset sufficient statistics

    # preallocate elbo
    prev_elbo = -T(Inf)
    elbos = Vector{T}()
    sizehint!(elbos, max_iter)

    prog = if progress
        Progress(max_iter; desc="Fitting LDS (Kalman) via EM...", barlen=50, showspeed=true)
    else
        nothing
    end

    for iter in 1:max_iter
        estep!(lds, suf, kws, data)
        # elbo = marginal_loglikelihood(lds, kws)
        # elbo = compute_elbo(lds, suf, kws)        
        mstep!(lds, suf, kws)
        elbo = compute_elbo(lds, suf, kws)        

        # report progress
        push!(elbos, elbo)
        progress && prog !== nothing && next!(prog)

        if monotonicity_check && (elbo - prev_elbo) < 0
            @warn "ELBO decreased from $(prev_elbo) to $(elbo) at iteration $(iter); this should not happen with a correct implementation. Consider reducing `tol` or checking for numerical issues."
        elseif (elbo - prev_elbo) < tol && (elbo - prev_elbo) > 0
            progress && prog !== nothing && finish!(prog)
            update_vestigials!(lds, data)
            return elbos
        end
        prev_elbo = elbo
    end
    progress && prog !== nothing && finish!(prog)
    update_vestigials!(lds, data)
    return elbos
end

function update_vestigials!(
    lds::LinearDynamicalSystem{T,S,O}, data::Data{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}

    # update vestigial parameters for backward compatibility (to be removed in future)
    if all(data.u0[:, 1] .== 1)
        lds.state_model.x0 = vec(lds.state_model.B0)
    end
    if all(data.d[:, 1, :] .== 1)
        lds.obs_model.d = vec(lds.obs_model.D)
    end
    if all(data.u[:, 1, :] .== 1)
        lds.state_model.b = vec(lds.state_model.B)
    end
    return lds
end

function format_kf_data!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3},
    u0::Union{Nothing,AbstractMatrix{T}},
    u::Union{Nothing,AbstractArray{T,3}},
    d::Union{Nothing,AbstractArray{T,3}},
    tsteps::Int,
    ntrials::Int,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}

    # Move x0, b, and d into u0, u, and d respectively
    if u0 === nothing
        u0_formatted = ones(T, 1, ntrials)
        lds.state_model.B0 = reshape(lds.state_model.x0, :, 1)
    else
        u0_formatted = u0
    end

    if u === nothing
        u_formatted = ones(T, 1, tsteps, ntrials)
        lds.state_model.B = reshape(lds.state_model.b, :, 1)
    else
        u_formatted = u
    end

    if d === nothing
        d_formatted = ones(T, 1, tsteps, ntrials)
        lds.obs_model.D = reshape(lds.obs_model.d, :, 1)
    else
        d_formatted = d
    end

    data = Data(; y=y, u0=u0_formatted, u=u_formatted, d=d_formatted)

    return data
end

@views function initialize_SufficientStatistics(
    model::LinearDynamicalSystem{T,S,O}, data::Data{T}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    latent_dim = model.latent_dim
    obs_dim = model.obs_dim
    tsteps = size(data.y, 2);
    ntrials = size(data.y, 3);
    u0_dim = model.init_input_dim
    u_dim = model.state_input_dim
    d_dim = model.obs_input_dim

    y_wide = reshape(data.y, size(data.y, 1), size(data.y, 2)*size(data.y, 3));
    u_wide = reshape(
        data.u[:, 1:(end - 1), :], size(data.u, 1), (size(data.u, 2)-1)*size(data.u, 3)
    );
    d_wide = reshape(data.d, size(data.d, 1), size(data.d, 2)*size(data.d, 3));

    PD_init(T, dim) = PDMat(diagm(ones(T, dim)))

    # precompute initial conditions
    init_xx = zeros(T, u0_dim, u0_dim);
    mul!(init_xx, data.u0, data.u0', one(T), zero(T));

    # precompute dynamics — populate kws.dyn_xx_buf in place. Only the bottom-right
    # u·uᵀ block is constant across E-steps; the (1,1) and (1,2) blocks are
    # overwritten each iteration in `sufficient_statistics!`.
    fill!(kws.dyn_xx_buf, zero(T))
    dyn_uu = u_wide*u_wide'
    kws.dyn_xx_buf[(latent_dim + 1):end, (latent_dim + 1):end] .= tol_PD(dyn_uu).mat

    # precompute observations — same pattern for obs_xx_buf.
    fill!(kws.obs_xx_buf, zero(T))
    obs_dd = d_wide*d_wide'
    kws.obs_xx_buf[(latent_dim + 1):end, (latent_dim + 1):end] .= tol_PD(obs_dd).mat

    obs_xy = zeros(T, latent_dim+d_dim, obs_dim)
    mul!(obs_xy[(latent_dim + 1):end, :], d_wide, y_wide', one(T), zero(T));

    obs_yy = zeros(T, obs_dim, obs_dim)
    mul!(obs_yy, y_wide, y_wide', one(T), zero(T));

    # Seed PDMat refs with valid initial values; sufficient_statistics! rewraps
    # them each E-step from the workspace buffers.
    init_dyn_xx = copy(kws.dyn_xx_buf)
    init_dyn_xx[1:latent_dim, 1:latent_dim] .= Matrix{T}(I, latent_dim, latent_dim)
    init_obs_xx = copy(kws.obs_xx_buf)
    init_obs_xx[1:latent_dim, 1:latent_dim] .= Matrix{T}(I, latent_dim, latent_dim)

    return SufficientStatistics{T}(
        # initial conditions
        ntrials,                                        # init_n
        Ref(tol_PD(init_xx)),                           # init_xx
        zeros(T, u0_dim, latent_dim),                   # init_xy
        Ref(PD_init(T, latent_dim)),                    # init_yy

        # dynamics model
        (tsteps - 1.0) * ntrials,                       # dyn_n
        Ref(PDMat(init_dyn_xx)),                        # dyn_xx
        zeros(T, latent_dim+u_dim, latent_dim),         # dyn_xy
        Ref(PD_init(T, latent_dim)),                    # dyn_yy

        # observation model
        tsteps * ntrials,                               # obs_n
        Ref(PDMat(init_obs_xx)),                        # obs_xx
        obs_xy,                                         # obs_xy
        Ref(tol_PD(obs_yy)),                            # obs_yy
    )
end

# ==== E-STEP =============================================================================

# using BenchmarkTools

"""
    estep!(lds, suf, y, kws::KalmanWorkspace; u=nothing, u0=nothing)

Kalman-path E-step. Runs `smooth!` via the Kalman/RTS backend, computes
sufficient statistics, and calls `calculate_elbo` using the companion
`SmoothWorkspace` pool (which still owns the Cholesky buffers used by
`Q_state!`/`Q_obs!`).
"""
function estep!(
    lds::LinearDynamicalSystem{T,S,O},
    suf::SufficientStatistics{T},
    kws::KalmanWorkspace{T},
    data::Data{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    precompute_kalman_constants!(kws, lds, data)

    smooth_cov!(lds, kws)

    smooth_mean!(lds, kws)

    return sufficient_statistics!(suf, kws, data)
end

"""
    validate_kalman_inputs(lds, data, ntrials, tsteps)

Validate input/parameter dimensional consistency for the Kalman path. Called
**once** at fit entry (not per E-step). Throws `DimensionMismatchError` or
`ArgumentError` on any mismatch between `B`, `B0`, `D` and the supplied `u`,
`u0`, `d` arrays in `data`.
"""
function validate_kalman_inputs(
    lds::LinearDynamicalSystem{T,S,O}, data::Data{T}, ntrials::Int, tsteps::Int
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    B0 = lds.state_model.B0
    if B0 !== nothing
        data.u0 === nothing &&
            throw(DimensionMismatchError("u0 (required because B0 !== nothing)", 1, 0))
        if size(data.u0, 1) != size(B0, 2)
            throw(
                DimensionMismatchError("u0 rows vs B0 cols", size(B0, 2), size(data.u0, 1))
            )
        end
        if size(data.u0, 2) != ntrials
            throw(
                DimensionMismatchError(
                    "u0 shape (u0_dim, ntrials)", (size(B0, 2), ntrials), size(data.u0)
                ),
            )
        end
    elseif data.u0 !== nothing
        throw(
            ArgumentError(
                "u0 was supplied but state_model.B0 is nothing; set B0 before passing u0"
            ),
        )
    end

    B = lds.state_model.B
    if B !== nothing
        data.u === nothing &&
            throw(DimensionMismatchError("u (required because B !== nothing)", 1, 0))
        if size(data.u, 1) != size(B, 2)
            throw(DimensionMismatchError("u rows vs B cols", size(B, 2), size(data.u, 1)))
        end
        if size(data.u, 2) != tsteps || size(data.u, 3) != ntrials
            throw(
                DimensionMismatchError(
                    "u shape (u_dim, T, ntrials)",
                    (size(B, 2), tsteps, ntrials),
                    size(data.u),
                ),
            )
        end
    elseif data.u !== nothing
        throw(
            ArgumentError(
                "u was supplied but state_model.B is nothing; set B before passing u"
            ),
        )
    end

    D = lds.obs_model.D
    if D !== nothing
        data.d === nothing &&
            throw(DimensionMismatchError("d (required because D !== nothing)", 1, 0))
        if size(data.d, 1) != size(D, 2)
            throw(DimensionMismatchError("d rows vs D cols", size(D, 2), size(data.d, 1)))
        end
        if size(data.d, 2) != tsteps || size(data.d, 3) != ntrials
            throw(
                DimensionMismatchError(
                    "d shape (d_dim, T, ntrials)",
                    (size(D, 2), tsteps, ntrials),
                    size(data.d),
                ),
            )
        end
    elseif data.d !== nothing
        throw(
            ArgumentError(
                "d was supplied but obs_model.D is nothing; set D before passing d"
            ),
        )
    end

    return nothing
end

"""
    precompute_kalman_constants!(kws::KalmanWorkspace, lds; tol=1e-6)

Refresh the cached `PDMat` wrappers of `Q`, `R`, `P0` (applying `tol_PD`) and
the derived constants `CiR = C' R^{-1}` (D × p) and `CiRC = C' R^{-1} C` (D × D).
Called once at the start of each E-step. Assumes inputs have already been
validated by `validate_kalman_inputs` at fit entry.
"""
function precompute_kalman_constants!(
    kws::KalmanWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T};
    tol::Real=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    kws.P0_PD[] = tol_PD(lds.state_model.P0; tol=tol)
    kws.Q_PD[] = tol_PD(lds.state_model.Q; tol=tol)
    kws.R_PD[] = tol_PD(lds.obs_model.R; tol=tol)

    C = lds.obs_model.C
    copyto!(kws.CiR, C'/kws.R_PD[])
    kws.CiRC[] = tol_PD(Xt_invA_X(kws.R_PD[], C))

    if data.u0 !== nothing
        @views mul!(kws.pred_mean[:, 1, :], lds.state_model.B0, data.u0)
    end

    if data.u !== nothing
        B = lds.state_model.B
        @views for n in axes(data.u, 3)
            mul!(kws.Bu[:, :, n], B, data.u[:, :, n])
        end
    end

    kws.y_minus_d .= data.y
    if data.d !== nothing
        D = lds.obs_model.D
        @views for n in axes(data.d, 3)
            mul!(kws.y_minus_d[:, :, n], D, data.d[:, :, n], -one(T), one(T))
        end
    end

    @views for n in axes(data.y, 3)
        mul!(kws.CiRY[:, :, n], kws.CiR, kws.y_minus_d[:, :, n])
    end
    return kws
end

# ==== SMOOTH COVARIANCE =============================================================================

function smooth_cov!(
    lds::LinearDynamicalSystem{T,S,O}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    @inline forwards_cov!(lds, kws)

    @inline backwards_cov!(lds, kws)
end

@views function forwards_cov!(
    lds::LinearDynamicalSystem{T,S,O}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    # tt = 1: pred_cov[1] is just the (toleranced) P0; pred_icov[1] is its
    # inverse; filt_cov[1] = inv(inv(P0) + CiRC) via the in-place info_update.
    kws.pred_cov[1] = kws.P0_PD[]
    kws.pred_icov[1] = inv(kws.pred_cov[1])
    info_update!(kws.filt_cov[1], kws.cov_tmp2, kws.pred_cov[1], kws.CiRC[])

    for tt in eachindex(kws.filt_cov)[2:end]
        # pred_cov[tt] = A·filt_cov[tt-1]·A' + Q
        kws.cov_tmp1 .= X_A_Xt(kws.filt_cov[tt - 1], lds.state_model.A)
        kws.cov_tmp1 .+= kws.Q_PD[].mat
        Symmetrize!(kws.cov_tmp1)
        kws.pred_cov[tt] = PDMat(kws.cov_tmp1)
        kws.pred_icov[tt] = inv(kws.pred_cov[tt])

        # filt_cov[tt] = inv(inv(pred_cov[tt]) + CiRC), in place via cached Cholesky.
        info_update!(kws.filt_cov[tt], kws.cov_tmp2, kws.pred_cov[tt], kws.CiRC[])
    end

    return kws
end

function backwards_cov!(
    lds::LinearDynamicalSystem{T,S,O}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}

    # init smoothed cov & accumulators
    kws.smooth_cov[end] = kws.filt_cov[end];
    kws.sum_smooth_cov_all .= kws.smooth_cov[end].mat
    kws.sum_smooth_cov_prev .= zeros(T, kws.latent_dim, kws.latent_dim)
    kws.sum_smooth_cov_next .= kws.smooth_cov[end].mat
    kws.sum_smooth_xcov .= zeros(T, kws.latent_dim, kws.latent_dim)

    # smooth covariance + joint Gaussian entropy ================================
    # H(x_{1:T}|y) = H(x_T|y) + Σ_{t=1}^{T-1} H(x_t | x_{t+1}, y)
    # backward-conditional cov: Σ_{t|x_{t+1},y} = filt_cov[t] - G[t]*pred_cov[t+1]*G[t]'
    #                                             = filt_cov[t] - filt_cov[t]*(G[t]*A)'
    ent_logdet = logdet(kws.smooth_cov[end].mat)

    @views for tt in eachindex(kws.filt_cov)[(end - 1):-1:1]

        # reverse kalman gain G[t] = filt_cov[t] · A' · pred_cov[t+1]^{-1}
        mul!(kws.G[:, :, tt], kws.filt_cov[tt], lds.state_model.A', one(T), zero(T));
        kws.G[:, :, tt] /= kws.pred_cov[tt + 1];

        # smoothed covariance — Joseph-style form, algebraically equivalent to
        # the standard RTS update `P_s[t] = P_f[t] + G[t]·(P_s[t+1] - P_p[t+1])·G[t]'`
        # but expanded so every term is manifestly PSD:
        #
        #   smooth_cov[t] = G·(smooth_cov[t+1] + Q)·G' + (I - G·A)·filt_cov[t]·(I - G·A)'
        #
        # The two summands are each `X·A·X'` of a PD matrix, so their sum is
        # symmetric and PSD up to floating-point roundoff. We Symmetrize!
        # defensively before wrapping as a PDMat — an earlier attempt that
        # built `smooth_cov[t+1] + Q` into a separate `Ref{PDMat}` first
        # tripped the strict cholesky in `PDMat(::Matrix)` from accumulated
        # asymmetry; folding the sum into the X_A_Xt argument avoids that.
        mul!(kws.cov_tmp1, kws.G[:, :, tt], lds.state_model.A, one(T), zero(T))
        kws.cov_tmp2 .=
            X_A_Xt(kws.smooth_cov[tt + 1] + kws.Q_PD[], kws.G[:, :, tt]) .+
            X_A_Xt(kws.filt_cov[tt], I - kws.cov_tmp1)
        Symmetrize!(kws.cov_tmp2)
        kws.smooth_cov[tt] = PDMat(kws.cov_tmp2)

        # accumulate smoothed covs
        kws.sum_smooth_cov_all .+= kws.smooth_cov[tt].mat;
        kws.sum_smooth_cov_prev .+= kws.smooth_cov[tt].mat;
        if tt > 1
            kws.sum_smooth_cov_next .+= kws.smooth_cov[tt].mat;
        end

        mul!(kws.sum_smooth_xcov, kws.G[:, :, tt], kws.smooth_cov[tt + 1].mat, one(T), one(T));

        # entropy contribution of backward-conditional cov
        # ent_logdet += logdet(kws.filt_cov[tt]) .+ logdet_Q .- logdet(kws.pred_cov[tt + 1])
        ent_logdet += logdet(kws.filt_cov[tt]) .- logdet(kws.pred_cov[tt + 1])
    end

    kws.shared_entropy[] =
        T(0.5) * (kws.tsteps * kws.latent_dim * (one(T) + log(T(2π))) + ent_logdet + (kws.tsteps-1) * logdet(kws.Q_PD[]))

    return kws
end

# ==== SMOOTH MEAN =============================================================================

function smooth_mean!(
    lds::LinearDynamicalSystem{T,S,O}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    forwards_mean!(lds, kws)
    # bench = @benchmark forwards_mean!($lds, $kws) samples=100
    # display(bench)

    return backwards_mean!(kws)
end

function forwards_mean!(
    lds::LinearDynamicalSystem{T,S,O}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    @views begin
        # filter initial mean
        mul!(kws.mean_tmp, kws.pred_icov[1], kws.pred_mean[:, 1, :], one(T), zero(T))
        kws.mean_tmp .+= kws.CiRY[:, 1, :]
        mul!(kws.filt_mean[:, 1, :], kws.filt_cov[1], kws.mean_tmp, one(T), zero(T))

        # pre-fill predicted means
        kws.pred_mean[:, 2:end, :] .= kws.Bu[:, 1:(end - 1), :]
    end

    @views for tt in axes(kws.pred_mean, 2)[2:end]
        mul!(
            kws.pred_mean[:, tt, :],
            lds.state_model.A,
            kws.filt_mean[:, tt - 1, :],
            one(T),
            one(T),
        );
        # kws.pred_mean[:,tt,:] .+= kws.Bu[:,tt-1,:];

        mul!(kws.mean_tmp, kws.pred_icov[tt].mat, kws.pred_mean[:, tt, :], one(T), zero(T));
        kws.mean_tmp .+= kws.CiRY[:, tt, :];

        mul!(kws.filt_mean[:, tt, :], kws.filt_cov[tt].mat, kws.mean_tmp, one(T), zero(T));
    end

    return kws
end

function backwards_mean!(kws::KalmanWorkspace{T}) where {T<:Real}
    kws.smooth_mean[:, end, :] .= kws.filt_mean[:, end, :];

    @views for tt in eachindex(kws.pred_icov)[(end-1):-1:1]
        kws.mean_tmp .= kws.smooth_mean[:, tt + 1, :] .- kws.pred_mean[:, tt + 1, :];
        mul!(kws.smooth_mean[:, tt, :], kws.G[:, :, tt], kws.mean_tmp, one(T), zero(T));
        kws.smooth_mean[:, tt, :] .+= kws.filt_mean[:, tt, :];
    end

    return kws
end

# ==== SUFFICIENT STATISTICS =============================================================================
# using BenchmarkTools

function sym_syrk!(out, x::Matrix{T}) where {T<:Real}
    BLAS.syrk!('U', 'N', one(T), x, one(T), out)
    LinearAlgebra.copytri!(out, 'U')
    return out
end

@inline function aggregate_xx(
    smooth_mean::Matrix{T},
    # smooth_cov::PDMat{T,Matrix{T}}, 
    smooth_cov::Matrix{T},
    ntrials::Int,
)::PDMat{T,Matrix{T}} where {T<:Real}
    xx = smooth_cov*ntrials;
    sym_syrk!(xx, smooth_mean)
    # mul!(xx, smooth_mean, smooth_mean', 1.0, 1.0)

    return PDMat(xx)
end

# @inline function aggregate_xx(
#     smooth_mean::Matrix{T}, 
#     smooth_cov::AbstractVector{PDMat{T,Matrix{T}}}, 
#     ntrials::Int
#     )::PDMat{T,Matrix{T}} where {T<:Real}

#     xx = sum(smooth_cov).mat*ntrials;
#     sym_syrk!(xx, smooth_mean)

#     return tol_PD(xx)

# end

@views @inline function sufficient_statistics!(
    suf::SufficientStatistics{T}, kws::KalmanWorkspace{T}, data::Data{T}
) where {T<:Real}

    # initial conditions -------
    kws.x_init .= kws.smooth_mean[:, 1, :]
    suf.init_n = kws.ntrials
    # init_xx (preset)
    # init_xy
    mul!(suf.init_xy, data.u0, kws.x_init', one(T), zero(T));
    # init_yy
    suf.init_yy[] = aggregate_xx(kws.x_init, kws.smooth_cov[1].mat, suf.init_n);

    # transitions -------
    suf.dyn_n = (kws.tsteps - 1) * kws.ntrials
    kws.x_prev .= reshape(kws.smooth_mean[:, 1:(end - 1), :], kws.latent_dim, suf.dyn_n)
    kws.x_next .= reshape(kws.smooth_mean[:, 2:end, :], kws.latent_dim, suf.dyn_n)
    u_prev = reshape(data.u[:, 1:(end - 1), :], size(data.u, 1), suf.dyn_n)

    # Reuse the preallocated workspace buffer; only the bottom-right u·uᵀ block
    # is constant — populated once in `initialize_SufficientStatistics` and never
    # mutated below. PDMat's cholesky() makes its own copy of the factors, and
    # downstream readers (mstep!, compute_elbo) operate via `XX + prior` /
    # `X_A_Xt(XX, W)` which produce fresh PDMats — they never write into .mat.
    dyn_xx = kws.dyn_xx_buf
    dyn_xx[1:kws.latent_dim, 1:kws.latent_dim] .= kws.sum_smooth_cov_prev .* kws.ntrials
    BLAS.syrk!('U', 'N', one(T), kws.x_prev, one(T), dyn_xx[1:kws.latent_dim, 1:kws.latent_dim])
    mul!(dyn_xx[1:kws.latent_dim, (kws.latent_dim + 1):end], kws.x_prev, u_prev', one(T), zero(T))
    LinearAlgebra.copytri!(dyn_xx, 'U')
    suf.dyn_xx[] = PDMat(dyn_xx)

    # dyn_xy
    suf.dyn_xy[1:kws.latent_dim, :] = kws.sum_smooth_xcov .* kws.ntrials;
    mul!(suf.dyn_xy[1:kws.latent_dim, :], kws.x_prev, kws.x_next', one(T), one(T))
    mul!(suf.dyn_xy[(kws.latent_dim + 1):end, :], u_prev, kws.x_next', one(T), zero(T))
    # dyn_yy
    # suf.dyn_yy[] = aggregate_xx(kws.x_next, kws.smooth_cov[2:end], kws.ntrials);
    suf.dyn_yy[] = aggregate_xx(kws.x_next, kws.sum_smooth_cov_next, kws.ntrials);

    # observations -------
    suf.obs_n = kws.tsteps * kws.ntrials
    kws.x_cur .= reshape(kws.smooth_mean, kws.latent_dim, suf.obs_n)
    y_cur = reshape(data.y, kws.obs_dim, suf.obs_n)
    d_cur = reshape(data.d, kws.obs_input_dim, suf.obs_n)

    # obs_xx
    # obs_xx = deepcopy(suf.obs_xx[].mat);
    # obs_xx[1:kws.latent_dim, 1:kws.latent_dim] = sum(kws.smooth_cov).mat * kws.ntrials;
    # sym_syrk!(obs_xx[1:kws.latent_dim, 1:kws.latent_dim], x_cur)
    # mul!(obs_xx[1:kws.latent_dim, (kws.latent_dim+1):end], x_cur, d_cur', 1.0, 0.0)
    # mul!(obs_xx[(kws.latent_dim+1):end, 1:kws.latent_dim], d_cur, x_cur', 1.0, 0.0)
    # suf.obs_xx[] = tol_PD(obs_xx);

    obs_xx = kws.obs_xx_buf
    obs_xx[1:kws.latent_dim, 1:kws.latent_dim] .= kws.sum_smooth_cov_all .* kws.ntrials
    BLAS.syrk!('U', 'N', one(T), kws.x_cur, one(T), obs_xx[1:kws.latent_dim, 1:kws.latent_dim])
    mul!(obs_xx[1:kws.latent_dim, (kws.latent_dim + 1):end], kws.x_cur, d_cur', one(T), zero(T))
    LinearAlgebra.copytri!(obs_xx, 'U')
    suf.obs_xx[] = PDMat(obs_xx)

    # obs_xy
    return mul!(suf.obs_xy[1:kws.latent_dim, :], kws.x_cur, y_cur', one(T), zero(T))
    # obs_yy (preset)

end

# ==== M-STEP =============================================================================
# Legacy overloads — Λ supplied as a raw PDMat (or `nothing`) with implicit
# zero prior mean. Retained while the workspace still stores `*_lambda`
# fields directly. New callers should prefer `mn_map(XX, XY, ::MNPrior)`,
# which is the same math but lets the caller specify a non-zero `M₀`.
function regress(
    XX::PDMat{T,Matrix{T}}, XY::AbstractMatrix{T}, prior_lambda::PDMat{T,Matrix{T}}
) where {T<:Real}
    return transpose((XX + prior_lambda) \ XY)
end
function regress(
    XX::PDMat{T,Matrix{T}}, XY::AbstractMatrix{T}, prior_lambda::Nothing
) where {T<:Real}
    return transpose(XX \ XY)
end

# MNPrior overload — full matrix-normal prior with arbitrary `M₀`. Delegates
# to the type-level `mn_map` helper so the same math is shared with the
# tridiag/Poisson M-steps once they migrate.
function regress(
    XX::PDMat{T,Matrix{T}}, XY::AbstractMatrix{T}, prior::MNPrior{T}
) where {T<:Real}
    return mn_map(XX, XY, prior)
end

function est_cov(
    W::AbstractMatrix{T},
    XX::PDMat{T,Matrix{T}},
    XY::AbstractMatrix{T},
    YY::PDMat{T,Matrix{T}},
    N::Int,
    prior_lambda::PDMat{T,Matrix{T}},
    prior_df::Int,
    prior_mu::AbstractMatrix{T},
)::Matrix{T} where {T<:Real}
    Wxy = W*XY;

    Cov =
        (
            YY .- Wxy .- Wxy' .+ X_A_Xt(XX, W) .+ X_A_Xt(prior_lambda, W) .+
            (prior_df * prior_mu)
        ) / (N .+ prior_df);

    return Cov # make PD, but don't save as PDMat
end

function est_cov(
    W::AbstractMatrix{T},
    XX::PDMat{T,Matrix{T}},
    XY::AbstractMatrix{T},
    YY::PDMat{T,Matrix{T}},
    N::Int,
    prior_lambda::Nothing,
    prior_df::Int,
    prior_mu::AbstractMatrix{T},
)::Matrix{T} where {T<:Real}
    Wxy = W*XY;

    Cov = (YY .- Wxy .- Wxy' .+ X_A_Xt(XX, W) .+ (prior_df * prior_mu)) / (N .+ prior_df);

    return Cov # make PD, but don't save as PDMat
end

function mstep!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}

    # TODO: include filt_bool

    # initials ===============================================
    B0 = regress(suf.init_xx[], suf.init_xy, kws.B0_lambda)

    P0 = est_cov(
        B0,
        suf.init_xx[],
        suf.init_xy,
        suf.init_yy[],
        suf.init_n,
        kws.B0_lambda,
        kws.P0_df,
        kws.P0_mu,
    )

    lds.state_model.B0 .= B0
    lds.state_model.P0 .= P0

    # dynamics ===============================================
    AB = regress(suf.dyn_xx[], suf.dyn_xy, kws.AB_lambda)

    A = AB[:, 1:kws.latent_dim];
    B = AB[:, (kws.latent_dim + 1):end];

    Q = est_cov(
        AB,
        suf.dyn_xx[],
        suf.dyn_xy,
        suf.dyn_yy[],
        suf.dyn_n,
        kws.AB_lambda,
        kws.Q_df,
        kws.Q_mu,
    )

    lds.state_model.A .= A
    lds.state_model.B .= B
    lds.state_model.Q .= Q

    # observations ===============================================
    CD = regress(suf.obs_xx[], suf.obs_xy, kws.CD_lambda)

    C = CD[:, 1:kws.latent_dim];
    D = CD[:, (kws.latent_dim + 1):end];

    R = est_cov(
        CD,
        suf.obs_xx[],
        suf.obs_xy,
        suf.obs_yy[],
        suf.obs_n,
        kws.CD_lambda,
        kws.R_df,
        kws.R_mu,
    )

    lds.obs_model.C .= C
    lds.obs_model.D .= D
    return lds.obs_model.R .= R
end

# ==== COMPUTE ELBO =============================================================================

# full priors
function log_post(
    n::Int,
    v::Int,
    v0::Int,
    vN::Int,
    lam0::PDMat{T,Matrix{T}},
    lamN::PDMat{T,Matrix{T}},
    Sig0::Matrix{T},
    SigN::PDMat{T,Matrix{T}},
) where {T<:Real}
    return -0.5*n*v*log(2pi) .+ 0.5*v*logdet(lam0) .+ -0.5*v*logdet(lamN) .+
           0.5*v0*logdet(0.5 .* Sig0) .+ -0.5*vN*logdet(0.5 .* SigN) .+
           -SpecialFunctions.loggamma(0.5 .* v0) .+ SpecialFunctions.loggamma(0.5 .* vN)
end;

# no beta prior
function log_post(
    n::Int,
    v::Int,
    v0::Int,
    vN::Int,
    lam0::Nothing,
    lamN::PDMat{T,Matrix{T}},
    Sig0::Matrix{T},
    SigN::PDMat{T,Matrix{T}},
) where {T<:Real}
    return -0.5*n*v*log(2pi) .+ -0.5*v*logdet(lamN) .+ 0.5*v0*logdet(0.5 .* Sig0) .+
           -0.5*vN*logdet(0.5 .* SigN) .+ -SpecialFunctions.loggamma(0.5 .* v0) .+
           SpecialFunctions.loggamma(0.5 .* vN)
end;

# no cov prior
function log_post(
    n::Int,
    v::Int,
    vN::Int,
    lam0::PDMat{T,Matrix{T}},
    lamN::PDMat{T,Matrix{T}},
    SigN::PDMat{T,Matrix{T}},
) where {T<:Real}
    return -0.5*n*v*log(2pi) .+ 0.5*v*logdet(lam0) .+ -0.5*v*logdet(lamN) .+
           -0.5*vN*logdet(0.5 .* SigN) .+ SpecialFunctions.loggamma(0.5 .* vN)
end;

# no prior
function log_post(
    n::Int,
    v::Int,
    vN::Int,
    lam0::Nothing,
    lamN::PDMat{T,Matrix{T}},
    SigN::PDMat{T,Matrix{T}},
) where {T<:Real}
    return -0.5*n*v*log(2pi) .+ -0.5*v*logdet(lamN) .+ -0.5*vN*logdet(0.5 .* SigN) .+
           SpecialFunctions.loggamma(0.5 .* vN)
end;

function compute_elbo(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    elbo = 0.0;

    P0_PD = tol_PD(lds.state_model.P0)
    Q_PD = tol_PD(lds.state_model.Q)
    R_PD = tol_PD(lds.obs_model.R)

    # Initial Conditions --------------------------------------
    n = suf.init_n
    v = kws.latent_dim;
    v0 = kws.P0_df;
    vN = v0 + (n - kws.init_input_dim);
    lam0 = kws.B0_lambda;
    lamN = lam0 === nothing ? suf.init_xx[] : lam0 + suf.init_xx[];
    Sig0 = kws.P0_mu * kws.P0_df
    SigN = P0_PD * vN;

    if v0 > 0
        elbo += log_post(n, v, v0, vN, lam0, lamN, Sig0, SigN)
    else
        elbo += log_post(n, v, vN, lam0, lamN, SigN)
    end

    isfinite(elbo) || throw(
        NumericalStabilityError("ELBO (initial conditions)", "non-finite log-posterior")
    )

    # Dynamics --------------------------------------
    n = suf.dyn_n
    v = kws.latent_dim;
    v0 = kws.Q_df;
    vN = v0 + (n - (kws.latent_dim + kws.state_input_dim));
    lam0 = kws.AB_lambda;
    lamN = lam0 === nothing ? suf.dyn_xx[] : lam0 + suf.dyn_xx[];
    Sig0 = kws.Q_mu * kws.Q_df
    SigN = Q_PD * vN;

    if v0 > 0
        elbo += log_post(n, v, v0, vN, lam0, lamN, Sig0, SigN)
    else
        elbo += log_post(n, v, vN, lam0, lamN, SigN)
    end

    isfinite(elbo) ||
        throw(NumericalStabilityError("ELBO (dynamics)", "non-finite log-posterior"))

    # Observations --------------------------------------
    n = suf.obs_n
    v = kws.obs_dim;
    v0 = kws.R_df;
    vN = v0 + (n - (kws.latent_dim + kws.obs_input_dim));
    lam0 = kws.CD_lambda;
    lamN = lam0 === nothing ? suf.obs_xx[] : lam0 + suf.obs_xx[];
    Sig0 = kws.R_mu * kws.R_df
    SigN = R_PD * vN;

    if v0 > 0
        elbo += log_post(n, v, v0, vN, lam0, lamN, Sig0, SigN)
    else
        elbo += log_post(n, v, vN, lam0, lamN, SigN)
    end
    isfinite(elbo) ||
        throw(NumericalStabilityError("ELBO (observations)", "non-finite log-posterior"))

    # Gaussian posterior entropy H[q(x_{1:T}|y)], shared across trials
    # Required for ELBO monotonicity: ELBO = NIW_marginal + H[q(x)]
    elbo += kws.ntrials * kws.shared_entropy[]

    return elbo
end

# function marginal_loglikelihood(
#     lds::LinearDynamicalSystem{T,S,O},
#     kws::KalmanWorkspace{T},
# ) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}

#     total_ll = zero(T)
#     CVCR = PDMat(Matrix(I(lds.obs_dim))) # placeholder, will be updated in loop
#     Cmu = zeros(T, lds.obs_dim)
#     MV = MvNormal(Cmu, CVCR)

#     @inline @views for t in eachindex(kws.pred_cov)

#         CVCR = tol_PD(X_A_Xt(kws.pred_cov[t], lds.obs_model.C) .+ lds.obs_model.R)

#         @inline @views for n in 1:kws.ntrials

#             mul!(Cmu, lds.obs_model.C, kws.pred_mean[:,t,n])
#             MV = MvNormal(Cmu, CVCR)
#             total_ll += logpdf(MV, kws.y_minus_d[:,t,n]);

#         end
#     end

#     return total_ll

# end

# function logpdf_sum!(
#     MV::MvNormal{T,PDMat{T,Matrix{T}}},
#     x::AbstractVector{T},
#     out::T,
# ) where {T<:Real} 
# out += Distributions.logpdf(MV, x)
# out
# end

function marginal_loglikelihood(
    lds::LinearDynamicalSystem{T,S,O}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    
    total_ll = zero(T)
    # kws.obs_pd_tmp[] = PDMat(Matrix{T}(I, lds.obs_dim, lds.obs_dim))
    MV = MvNormal(Matrix{T}(I, lds.obs_dim, lds.obs_dim))

    Cmu = zeros(T, lds.obs_dim, kws.tsteps * kws.ntrials)
    mul!(Cmu, lds.obs_model.C, reshape(kws.pred_mean, kws.latent_dim, kws.tsteps*kws.ntrials))
    kws.innovation .= kws.y_minus_d .- reshape(Cmu, kws.obs_dim, kws.tsteps, kws.ntrials)

    @views for t in eachindex(kws.pred_cov)
        kws.obs_pd_tmp[] = tol_PD(
            X_A_Xt(kws.pred_cov[t], lds.obs_model.C) .+ lds.obs_model.R
        )
        MV = MvNormal(kws.obs_pd_tmp[])

        for n in axes(kws.innovation, 3)
            total_ll += Distributions.logpdf(MV, kws.innovation[:, t, n]);
        end
    end

    return total_ll
end
