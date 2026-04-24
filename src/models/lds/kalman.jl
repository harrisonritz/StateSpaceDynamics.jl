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

using BenchmarkTools



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

    # format inputs and preallocate workspace + sufficient statistics
    u0, u, d = format_kf_inputs(lds, u0, u, tsteps, ntrials)
    kws = KalmanWorkspace(lds, tsteps, ntrials)
    suf = initialize_SufficientStatistics(lds, y, u0, u, d) # reset sufficient statistics

    # preallocate elbo
    prev_elbo = -T(Inf)
    elbos = Vector{T}()
    sizehint!(elbos, max_iter)

    prog = progress ? Progress(max_iter; desc="Fitting LDS (Kalman) via EM...", barlen=50, showspeed=true) : nothing

    for _ in 1:max_iter

        
        estep!(lds, suf, kws, y; u=u, u0=u0)
        mstep!(lds, suf, kws, y; u=u, u0=u0)
        elbo = elbo(lds, suf, y, kws; u=u, u0=u0)

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


function format_kf_inputs(
    lds::LinearDynamicalSystem{T,S,O},
    u0::Union{Nothing,AbstractMatrix{T}},
    u::Union{Nothing,AbstractArray{T,3}},
    tsteps::Int,
    ntrials::Int,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    # fold x0 into u0, b into u, and d into d

    u0_formatted = u0 === nothing ? ones(T, 1, ntrials) : u0
    u_formatted = u === nothing ? repeat(lds.obs_model.b, 1, tsteps, ntrials) : u
    d_formatted = repeat(lds.obs_model.d, 1, tsteps, ntrials)

    return u0_formatted, u_formatted, d_formatted

end


function initialize_SufficientStatistics(
    model::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3},
    u0::AbstractMatrix{T},
    u::AbstractArray{T,3},
    d::AbstractArray{T,3},
    ) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}

    latent_dim = model.latent_dim
    obs_dim = model.obs_dim
    u_dim = size(model.state_model.B, 2)
    u0_dim = size(model.state_model.B0, 2)

    y_wide = reshape(y, size(y, 1), size(y, 2)*size(y, 3));
    u_wide = reshape(u[:,1:end-1,:], size(u, 1), (size(u, 2)-1)*size(u, 3));
    d_wide = reshape(d, size(d, 1), (size(d, 2)-1)*size(d, 3));
    
    return SufficientStatistics{T}(

        # initial conditions
        tol_PD(Matrix(u0 * u0')),                       # init_xx
        zeros(T, u0_dim, latent_dim),                   # init_xy
        PDMat(diagm(ones(T,latent_dim))),               # init_yy

        # transitions model: ADD PDMat(diagm(ones(T,latent_dim))),
        # zeros(T, latent_dim+u_dim, latent_dim+u_dim),   # dyn_xx
        zeros(T, latent_dim+u_dim, latent_dim),         # dyn_xy
        zeros(T, latent_dim, latent_dim),               # dyn_yy
        tol_PD(Matrix(u_wide * u_wide')),               # dyn_uu
        
        # observation model
        # zeros(T, obs_dim+1, obs_dim+1),                 # obs_xx
        zeros(T, obs_dim+1, obs_dim),                   # obs_xy
        tol_PD(Matrix(y_wide * y_wide')),               # obs_yy
        tol_PD(Matrix(d_wide * d_wide')),               # obs_dd 
        d_wide * y_wide',                               # obs_dy


    )
end


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
    y::AbstractArray{T,3};
    u::Union{Nothing,AbstractArray{T,3}}=nothing,
    u0::Union{Nothing,AbstractMatrix{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    
    precompute_kalman_constants!(kws, lds, y; u=u, u0=u0)

    smooth_cov!(lds, kws)

    smooth_mean!(lds, kws)

    sufficient_statistics!(suf, kws)

end


"""
    precompute_kalman_constants!(kws::KalmanWorkspace, lds; tol=1e-6)

Refresh the cached `PDMat` wrappers of `Q`, `R`, `P0` (applying `tol_PD`) and
the derived constants `CiR = C' R^{-1}` (D × p) and `CiRC = C' R^{-1} C` (D × D).
Called once at the start of each E-step.
"""
function precompute_kalman_constants!(
    kws::KalmanWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3};
    u::Union{Nothing,AbstractArray{T,3}}=nothing,
    u0::Union{Nothing,AbstractMatrix{T}}=nothing,
    tol::Real=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}

    C = lds.obs_model.C

    kws.Q_PD[]  = tol_PD(lds.state_model.Q;  tol=tol)
    kws.R_PD[]  = tol_PD(lds.obs_model.R;    tol=tol)
    kws.P0_PD[] = tol_PD(lds.state_model.P0; tol=tol)

    copyto!(kws.CiR, C'/kws.R_PD[])
    kws.CiRC[] = tol_PD(Xt_invA_X(kws.R_PD[], C))
    
    @inbounds @views for n in axes(y, 3)
        mul!(kws.CiRY[:,:,n], kws.CiR, y[:, :, n] .- lds.obs_model.d);
    end


    # Ongoing inputs
    B = lds.state_model.B

    if B !== nothing     # Validate u / B consistency
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

    
    kws.Bu .= lds.state_model.b
    if u !== nothing
        @inbounds @views for n in axes(u, 3)
            mul!(kws.Bu[:, :, n], B, u[:, :, n])
        end
    end


    # Initial inputs
    B0 = lds.state_model.B0

    if B0 !== nothing     # Validate u / B consistency
        u0 === nothing && throw(
            DimensionMismatchError("u0 (required because B0 !== nothing)", 1, 0),
        )
        if size(u0, 1) != size(B0, 2)
            throw(DimensionMismatchError("u0 rows vs B0 cols", size(B0, 2), size(u0, 1)))
        end
        if size(u0, 2) != kws.ntrials
            throw(
                DimensionMismatchError(
                    "u0 shape (u0_dim, ntrials)",
                        (size(B0, 2), kws.ntrials),
                        size(u0),
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

    kws.pred_mean[:,1,:] .= lds.state_model.x0;
    if u0 !== nothing
        mul!(kws.pred_mean[:,1,:], B0, u0, 1.0, 1.0);
    end
    
end




# ==== SMOOTH COVARIANCE =============================================================================

function smooth_cov!(
    lds::LinearDynamicalSystem{T,S,O}, 
    kws::KalmanWorkspace{T}
    ) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
        
    forwards_cov!(lds, kws)
    
    backwards_cov!(lds, kws)

end


function forwards_cov!(
    lds::LinearDynamicalSystem{T,S,O},
    kws::KalmanWorkspace{T}
    ) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}

    kws.pred_cov[1] = kws.P0_PD[];
    kws.filt_cov[1] = PDMat(inv(inv(kws.P0_PD[]) + kws.CiRC[]));

    @inbounds @views for tt in eachindex(kws.filt_cov)[2:end]

        kws.pred_cov[tt] = PDMat(X_A_Xt(kws.filt_cov[tt-1], lds.state_model.A) + kws.Q_PD[].mat);
        kws.pred_icov[tt] = inv(kws.pred_cov[tt]);
        kws.filt_cov[tt] = inv(kws.CiRC[] + kws.pred_icov[tt]);

    end

    return kws

end


function backwards_cov!(
    lds::LinearDynamicalSystem{T,S,O},
    kws::KalmanWorkspace{T}
    ) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}

    kws.smooth_cov[end] = kws.filt_cov[end];
    # At = lds.state_model.A'

    # smooth covariance ================================
    @inbounds @views for tt in eachindex(kws.filt_cov)[end-1:-1:1]

        # reverse kalman gain
        mul!(kws.G[:,:,tt], kws.filt_cov[tt], lds.state_model.A', 1.0, 0.0);
        kws.G[:,:,tt] /= kws.pred_cov[tt+1];

        # smoothed covariancess
        mul!(kws.cov_tmp1, kws.G[:,:,tt], lds.state_model.A, 1.0, 0.0);
        kws.smooth_cov[tt] = PDMat(X_A_Xt(kws.smooth_cov[tt+1] + kws.Q_PD[], kws.G[:,:,tt]) .+ 
                                     X_A_Xt(kws.filt_cov[tt], I - kws.cov_tmp1));

        # smoothed cross-cov
        # mul!(kws.cov_tmp2, kws.G[:,:,tt], kws.smooth_cov[tt+1], 1.0, 1.0);

    end

    return kws

end






# ==== SMOOTH MEAN =============================================================================

function smooth_mean!(
    lds::LinearDynamicalSystem{T,S,O}, 
    kws::KalmanWorkspace{T},
    ) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
        
    forwards_mean!(lds, kws)
    # bench = @benchmark forwards_mean!($lds, $kws) samples=100
    # display(bench)
    
    backwards_mean!(kws)

end


function forwards_mean!(
    lds::LinearDynamicalSystem{T,S,O},
    kws::KalmanWorkspace{T}
    ) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    
    @inbounds @views for tt in axes(kws.pred_mean,2)[2:end]

        mul!(kws.pred_mean[:,tt,:], lds.state_model.A, kws.filt_mean[:,tt-1,:], 1.0, 0.0);
        kws.pred_mean[:,tt,:] .+= kws.Bu[:,tt-1,:];

        mul!(kws.mean_tmp, kws.pred_icov[tt], kws.pred_mean[:,tt,:], 1.0, 0.0);
        kws.mean_tmp .+= kws.CiRY[:,tt,:];

        mul!(kws.filt_mean[:,tt,:], kws.filt_cov[tt], kws.mean_tmp, 1.0, 0.0);

    end

    return kws

end


function backwards_mean!(
    kws::KalmanWorkspace{T}
    ) where {T<:Real}
    
    kws.smooth_mean[:,end,:] .= kws.filt_mean[:, end,:];

    @inbounds @views for tt in eachindex(kws.pred_icov)[end-1:-1:1]

        kws.mean_tmp .= kws.smooth_mean[:,tt+1,:] .- kws.pred_mean[:,tt+1,:];
        mul!(kws.smooth_mean[:,tt,:], kws.G[:,:,tt], kws.mean_tmp, 1.0, 0.0);
        kws.smooth_mean[:,tt,:] .+= kws.filt_mean[:,tt,:];

    end

    return kws

end




# ==== SUFFICIENT STATISTICS =============================================================================


function sufficient_statistics!(
    suf::SufficientStatistics{T},
    kws::KalmanWorkspace{T},
) where {T<:Real}

    # initial conditions
    mul!(suf.init_xx, kws.pred_mean[:,1,:], kws.pred_mean[:,1,:]', 1.0, 0.0);
    suf.init_xx .= kws.pred_mean[:,1,:] * kws.pred_mean[:,1,:]' .+ kws.pred_cov[1].mat


    # xx_obs accumulator over all trials and time
    Mflat_obs = reshape(kws.smooth_mean, n, T * N_trials)
    mul!(S.est.xx_obs, Mflat_obs, Mflat_obs', 1.0, 1.0)    # single n×n GEMM

    # xy_obs
    Yflat = reshape(S.dat.y_train, m, T * N_trials)
    mul!(S.est.xy_obs, Mflat_obs, Yflat', 1.0, 1.0)

    # xx_dyn (latent block): M_cur = M_smooth[:, 1:T-1, :], M_next = M_smooth[:, 2:T, :]
    Mcur_flat  = reshape(@view(M_smooth[:, 1:T-1, :]), n, (T-1)*N_trials)
    Mnext_flat = reshape(@view(M_smooth[:, 2:T,   :]), n, (T-1)*N_trials)
    mul!(xx_dyn_xx, Mcur_flat, Mcur_flat', 1.0, 1.0)
    mul!(xy_dyn_xx, Mcur_flat, Mnext_flat', 1.0, 1.0)
    mul!(yy_dyn,    Mnext_flat, Mnext_flat', 1.0, 1.0)





    
    # # init ===============================================
    # S.est.xy_init .= zeros(S.dat.u0_dim, S.dat.x_dim);
    # S.est.yy_init .= S.est.smooth_cov[1] .* S.dat.n_train;
    # S.est.n_init .= copy(S.dat.n_train);


    # # dyn ===============================================
    # S.est.xx_dyn .= zeros(S.dat.x_dim + S.dat.u_dim, S.dat.x_dim + S.dat.u_dim);
    # S.est.xx_dyn[1:S.dat.x_dim,1:S.dat.x_dim] .= sum(S.est.smooth_cov[1:end-1]) .* S.dat.n_train;
    # S.est.xx_dyn[(S.dat.x_dim+1):end, (S.dat.x_dim+1):end] .= copy(S.est.uu_dyn);
    
    # S.est.xy_dyn .= zeros(S.dat.x_dim + S.dat.u_dim, S.dat.x_dim);
    # S.est.xy_dyn[1:S.dat.x_dim,:] .= S.est.smooth_xcov*S.dat.n_train;

    # S.est.yy_dyn .= sum(S.est.smooth_cov[2:end]) * S.dat.n_train;

    # S.est.n_dyn .= (S.dat.n_steps-1) * S.dat.n_train;


    # # obs ===============================================
    # S.est.xx_obs .= sum(S.est.smooth_cov) * S.dat.n_train;
    # S.est.xy_obs .= zeros(S.dat.x_dim, S.dat.y_dim);
    # S.est.n_obs .= S.dat.n_steps * S.dat.n_train;

end


