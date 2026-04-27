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
    u0::Union{Nothing,AbstractMatrix{T}}=nothing,
    u::Union{Nothing,AbstractArray{T,3}}=nothing,
    d::Union{Nothing,AbstractArray{T,3}}=nothing,
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    eltype(y) === T || error("Observed data must be of type $(T); got $(eltype(y))")

    tsteps = size(y, 2)
    ntrials = size(y, 3)

    # format inputs and preallocate workspace + sufficient statistics
    data = format_kf_data!(lds, y, u0, u, d, tsteps, ntrials)
    kws = KalmanWorkspace(lds, tsteps, ntrials)
    suf = initialize_SufficientStatistics(lds, data) # reset sufficient statistics

    # preallocate elbo
    prev_elbo = -T(Inf)
    elbos = Vector{T}()
    sizehint!(elbos, max_iter)

    prog = progress ? Progress(max_iter; desc="Fitting LDS (Kalman) via EM...", barlen=50, showspeed=true) : nothing

    for _ in 1:max_iter

        estep!(lds, suf, kws, data)
        mstep!(lds, suf, kws, data)
        elbo = elbo(lds, suf, kws; data)

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

        println("\nu0 shape = ($(size(u0_formatted)))")
        println("state_model.B0 shape = ($(size(lds.state_model.B0)))")
    end

    if u === nothing
        u_formatted = ones(T, 1, tsteps, ntrials)
        lds.state_model.B = reshape(lds.state_model.b, :, 1)

        println("\nu shape = ($(size(u_formatted)))")
        println("state_model.B shape = ($(size(lds.state_model.B)))")
    end

    if d === nothing
        d_formatted = ones(T, 1, tsteps, ntrials)
        lds.obs_model.D = reshape(lds.obs_model.d, :, 1)

        println("\nd shape = ($(size(d_formatted)))")
        println("obs_model.D shape = ($(size(lds.obs_model.D)))")
    end

    data = Data(
        y = y,
        u0 = u0_formatted,
        u = u_formatted,
        d = d_formatted,
    )

    return data

end


function initialize_SufficientStatistics(
    model::LinearDynamicalSystem{T,S,O},
    data::Data{T},
    ) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}

    latent_dim = model.latent_dim
    obs_dim = model.obs_dim
    u0_dim = model.init_input_dim
    u_dim = model.state_input_dim
    d_dim = model.obs_input_dim

    y_wide = reshape(data.y, size(data.y, 1), size(data.y, 2)*size(data.y, 3));
    u_wide = reshape(data.u[:,1:end-1,:], size(data.u, 1), (size(data.u, 2)-1)*size(data.u, 3));
    d_wide = reshape(data.d, size(data.d, 1), size(data.d, 2)*size(data.d, 3));

    PD_init(T, dim) = PDMat(diagm(ones(T,dim)))

    # precompute initial conditions
    init_xx = zeros(T, u0_dim, u0_dim);
    mul!(init_xx, data.u0, data.u0', 1.0, 0.0);
    
    # precompute dynamics
    dyn_xx = zeros(T, latent_dim+u_dim, latent_dim+u_dim)
    dyn_xx[1:latent_dim, 1:latent_dim] = I(latent_dim)
    mul!(dyn_xx[(latent_dim+1):end, (latent_dim+1):end], u_wide, u_wide', 1.0, 0.0);

    # precompute observations
    obs_xx = zeros(T, latent_dim+d_dim, latent_dim+d_dim)
    obs_xx[1:latent_dim, 1:latent_dim] = I(latent_dim)
    mul!(obs_xx[(latent_dim+1):end, (latent_dim+1):end], d_wide, d_wide', 1.0, 0.0);

    obs_xy = zeros(T, latent_dim+d_dim, obs_dim)
    mul!(obs_xy[(latent_dim+1):end, :], d_wide, y_wide', 1.0, 0.0);

    obs_yy = zeros(T, obs_dim, obs_dim)
    mul!(obs_yy, y_wide, y_wide', 1.0, 0.0);

    
    return SufficientStatistics{T}(

        # initial conditions
        Ref(tol_PD(init_xx)),                           # init_xx
        zeros(T, u0_dim, latent_dim),                   # init_xy
        Ref(PD_init(T, latent_dim)),                    # init_yy

        # transitions model
        Ref(tol_PD(dyn_xx)),                            # dyn_xx
        zeros(T, latent_dim+u_dim, latent_dim),         # dyn_xy
        Ref(PD_init(T, latent_dim)),                    # dyn_yy
        
        # observation model
        Ref(tol_PD(obs_xx)),                            # obs_xx
        obs_xy,                                         # obs_xy
        Ref(tol_PD(obs_yy)),                            # obs_yy
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
    data::Data{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    
    precompute_kalman_constants!(kws, lds, data)

    smooth_cov!(lds, kws)

    smooth_mean!(lds, kws)

    sufficient_statistics!(suf, kws, data)

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
    data::Data{T};
    tol::Real=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}

    C = lds.obs_model.C

    kws.Q_PD[]  = tol_PD(lds.state_model.Q;  tol=tol)
    kws.R_PD[]  = tol_PD(lds.obs_model.R;    tol=tol)
    kws.P0_PD[] = tol_PD(lds.state_model.P0; tol=tol)


    copyto!(kws.CiR, C'/kws.R_PD[])
    kws.CiRC[] = tol_PD(Xt_invA_X(kws.R_PD[], C))

    
    @inbounds @views for n in axes(data.y, 3)
        mul!(kws.CiRY[:,:,n], kws.CiR, data.y[:, :, n] .- kws.Dd[:,:,n]);
    end


    # Initial inputs
    B0 = lds.state_model.B0

    if B0 !== nothing     # Validate u / B consistency
        data.u0 === nothing && throw(
            DimensionMismatchError("u0 (required because B0 !== nothing)", 1, 0),
        )
        if size(data.u0, 1) != size(B0, 2)
            throw(DimensionMismatchError("u0 rows vs B0 cols", size(B0, 2), size(data.u0, 1)))
        end
        if size(data.u0, 2) != kws.ntrials
            throw(
                DimensionMismatchError(
                    "u0 shape (u0_dim, ntrials)",
                        (size(B0, 2), kws.ntrials),
                        size(data.u0),
                ),
            )
        end
    elseif data.u0 !== nothing
        throw(
            ArgumentError(
                "u0 was supplied but state_model.B0 is nothing; set B0 before passing u0",
            ),
        )
    end

    if data.u0 !== nothing
        mul!(kws.pred_mean[:,1,:], B0, data.u0, 1.0, 1.0);
    end


    # state inputs
    B = lds.state_model.B

    if B !== nothing     # Validate u / B consistency
        data.u === nothing && throw(
            DimensionMismatchError("u (required because B !== nothing)", 1, 0),
        )
        if size(data.u, 1) != size(B, 2)
            throw(DimensionMismatchError("u rows vs B cols", size(B, 2), size(data.u, 1)))
        end
        if size(data.u, 2) != kws.tsteps || size(data.u, 3) != kws.ntrials
            throw(
                DimensionMismatchError(
                    "u shape (u_dim, T, ntrials)",
                    (size(B, 2), kws.tsteps, kws.ntrials),
                    size(data.u),
                ),
            )
        end
    elseif data.u !== nothing
        throw(
            ArgumentError(
                "u was supplied but state_model.B is nothing; set B before passing u",
            ),
        )
    end

    
    if data.u !== nothing
        @inbounds @views for n in axes(data.u, 3)
            mul!(kws.Bu[:, :, n], B, data.u[:, :, n])
        end
    end



    # observation inputs
    D = lds.obs_model.D

    if D !== nothing     # Validate d / D consistency
        data.d === nothing && throw(
            DimensionMismatchError("d (required because D !== nothing)", 1, 0),
        )
        if size(data.d, 1) != size(D, 2)
            throw(DimensionMismatchError("d rows vs D cols", size(D, 2), size(data.d, 1)))
        end
        if size(data.d, 2) != kws.tsteps || size(data.d, 3) != kws.ntrials
            throw(
                DimensionMismatchError(
                    "d shape (d_dim, T, ntrials)",
                    (size(D, 2), kws.tsteps, kws.ntrials),
                    size(data.d),
                ),
            )
        end
    elseif data.d !== nothing
        throw(
            ArgumentError(
                "d was supplied but obs_model.D is nothing; set D before passing d",
            ),
        )
    end

    if data.d !== nothing
        @inbounds @views for n in axes(data.d, 3)
            mul!(kws.Dd[:, :, n], D, data.d[:, :, n])
        end
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


function aggregate_xx(
    smooth_mean::AbstractArray{T,2}, 
    smooth_cov::PDMat{T,Matrix{T}}, 
    ntrials::Int
    )::PDMat{T,Matrix{T}} where {T<:Real}

    xx = smooth_cov.mat*ntrials;
    mul!(xx, smooth_mean, smooth_mean', 1.0, 1.0)
    pd_xx = tol_PD(xx);
    return pd_xx

end

function aggregate_xx(
    smooth_mean::AbstractArray{T,2}, 
    smooth_cov::AbstractVector{PDMat{T,Matrix{T}}}, 
    ntrials::Int
    )::PDMat{T,Matrix{T}} where {T<:Real}

    xx = sum(smooth_cov).mat*ntrials;
    mul!(xx, smooth_mean, smooth_mean', 1.0, 1.0)
    pd_xx = tol_PD(xx);
    return pd_xx

end


@views function sufficient_statistics!(
    suf::SufficientStatistics{T},
    kws::KalmanWorkspace{T},
    data::Data{T},
    ) where {T<:Real}

    # initial conditions -------
    # init_xx (preset)
    # init_xy
    mul!(suf.init_xy, data.u0, kws.pred_mean[:,1,:]', 1.0, 0.0);
    # init_yy
    suf.init_yy[] = aggregate_xx(kws.smooth_mean[:,1,:], kws.smooth_cov[1], kws.ntrials);


    # transitions -------
    x_prev = reshape(kws.smooth_mean[:,1:end-1,:], kws.latent_dim, (kws.tsteps-1) * kws.ntrials)
    x_next = reshape(kws.smooth_mean[:,2:end,:], kws.latent_dim, (kws.tsteps-1) * kws.ntrials)
    u_prev = reshape(data.u[:,1:end-1,:], size(data.u, 1), (size(data.u, 2)-1)*size(data.u, 3))
    #     dyn_xx
    dyn_xx = deepcopy(suf.dyn_xx[].mat);
    dyn_xx[1:kws.latent_dim, 1:kws.latent_dim] = sum(kws.smooth_cov[1:end-1]).mat .* kws.ntrials;
    mul!(dyn_xx[1:kws.latent_dim, 1:kws.latent_dim], x_prev, x_prev', 1.0, 1.0)
    mul!(dyn_xx[1:kws.latent_dim, (kws.latent_dim+1):end], x_prev, u_prev', 1.0, 0.0)
    mul!(dyn_xx[(kws.latent_dim+1):end, 1:kws.latent_dim], u_prev, x_prev', 1.0, 0.0)
    suf.dyn_xx[] = tol_PD(dyn_xx);
    #     dyn_xy
    mul!(suf.dyn_xy[1:kws.latent_dim,:], x_prev, x_next', 1.0, 0.0)
    #     dyn_yy
    suf.dyn_yy[] = aggregate_xx(x_next, kws.smooth_cov[2:end], kws.ntrials);


    # observations -------
    x_cur = reshape(kws.smooth_mean, kws.latent_dim, kws.tsteps * kws.ntrials)
    y_cur = reshape(data.y, kws.obs_dim, kws.tsteps * kws.ntrials)
    d_cur = reshape(data.d, kws.obs_input_dim, kws.tsteps * kws.ntrials)
    # obs_xx
    obs_xx = deepcopy(suf.obs_xx[].mat);
    obs_xx[1:kws.latent_dim, 1:kws.latent_dim] = sum(kws.smooth_cov).mat * kws.ntrials;
    mul!(obs_xx[1:kws.latent_dim, 1:kws.latent_dim], x_cur, x_cur', 1.0, 1.0)
    mul!(obs_xx[1:kws.latent_dim, (kws.latent_dim+1):end], x_cur, d_cur', 1.0, 0.0)
    mul!(obs_xx[(kws.latent_dim+1):end, 1:kws.latent_dim], d_cur, x_cur', 1.0, 0.0)
    suf.obs_xx[] = tol_PD(obs_xx);
    # obs_xy
    mul!(suf.obs_xy[1:kws.latent_dim,:], x_cur, y_cur', 1.0, 0.0)
    #     obs_yy (preset)

end



# ==== M-STEP =============================================================================
regress(XX::PDMat{T,Matrix{T}}, XY::AbstractMatrix{T}, lam::T) where {T<:Real} = PDMat(XX + lam) \ XY
regress(XX::PDMat{T,Matrix{T}}, XY::AbstractMatrix{T}) where {T<:Real} = XX \ XY

est_cov(
    XX::PDMat{T,Matrix{T}},
    XY::AbstractMatrix{T},
    YY::PDMat{T,Matrix{T}},
    lam::T,
    df::T,
    mu::T,
) where {T<:Real} 



end


function mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    suf::SufficientStatistics{T},
    kws::KalmanWorkspace{T},
    data::Data{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}

    # update initial conditions
    # initials ===============================================
    # Mean
    W = (PDMat(suf.init_xx + lds.prior_init) \ suf.init_xy)';
    B0 = W

    # Covariance
    Wxy = W*S.est.xy_init;
    P0e = (S.est.yy_init .- Wxy .- Wxy' .+ X_A_Xt(S.est.xx_init, W) .+ W*S.prm.lam_B0*W' + (S.prm.df_P0 * S.prm.mu_P0)) / 
            ((S.est.n_init[1] + S.prm.df_P0) - size(S.est.xx_init,1));


    P0 = format_noise(P0e, S.prm.P0_type);

    


    # latents ===============================================
    # Mean
    W = ((S.est.xx_dyn_PD[1] + S.prm.lam_AB) \ S.est.xy_dyn)';
    A = W[:, 1:S.dat.x_dim];
    B = W[:, (S.dat.x_dim+1):end];

    # Covariance
    Wxy = W*S.est.xy_dyn;
    Qe = (S.est.yy_dyn .- Wxy .- Wxy' .+ X_A_Xt(S.est.xx_dyn_PD[1], W) .+ W*S.prm.lam_AB*W' + (S.prm.df_Q * S.prm.mu_Q)) / 
        ((S.est.n_dyn[1] + S.prm.df_Q) - size(S.est.xx_dyn,1));

    Q = format_noise(Qe, S.prm.Q_type);




    # emissions ===============================================
    # Mean
    W = ((S.est.xx_obs_PD[1] + S.prm.lam_C) \ S.est.xy_obs)';
    C = deepcopy(W);

    # Covariance
    Wxy = W*S.est.xy_obs;
    Re = (S.est.yy_obs .- Wxy .- Wxy' .+ X_A_Xt(S.est.xx_obs_PD[1], W) .+ W*S.prm.lam_C*W' + (S.prm.df_R * S.prm.mu_R)) / 
            ((S.est.n_obs[1] + S.prm.df_R) - size(S.est.xx_obs,1));

    R = format_noise(Re, S.prm.R_type);





end