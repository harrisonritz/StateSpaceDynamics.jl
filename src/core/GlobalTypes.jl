"""
Abstract type for Dynamical Systems. I.e. LDS, etc.
"""
abstract type AbstractStateModel{T<:Real} end
abstract type AbstractObservationModel{T<:Real} end

"""
    FilterSmooth{T<:Real}

Per-trial container for smoothed estimates and associated covariance matrices.
A multi-trial fit holds one of these per trial (see `TrialFilterSmooth`); trial lengths
may differ.

# Fields
- `x_smooth::Matrix{T}`: smoothed state estimates `(latent_dim × T_trial)`
- `p_smooth::Array{T,3}`: smoothed covariances `(latent_dim × latent_dim × T_trial)`
- `p_smooth_tt1::Array{T,3}`: lag-1 cross covariances `(latent_dim × latent_dim × T_trial)`
- `E_z::Matrix{T}`: posterior mean `(latent_dim × T_trial)`
- `E_zz::Array{T,3}`: second moment `E[zₜzₜ']` `(latent_dim × latent_dim × T_trial)`
- `E_zz_prev::Array{T,3}`: second moment `E[zₜzₜ₋₁']` `(latent_dim × latent_dim × T_trial)`
- `entropy::T`: posterior entropy `H[q(x)]` for this trial
"""
mutable struct FilterSmooth{T<:Real}
    x_smooth::Matrix{T}
    p_smooth::Array{T,3}
    p_smooth_tt1::Array{T,3}
    E_z::Matrix{T}
    E_zz::Array{T,3}
    E_zz_prev::Array{T,3}
    entropy::T
end

function Base.show(io::IO, fs::FilterSmooth; gap="")
    println(io, gap, "Filter Smooth Object:")
    println(io, gap, "---------------------")
    println(io, gap, " size(x_smooth)  = ($(size(fs.x_smooth,1)), $(size(fs.x_smooth,2)))")
    println(
        io,
        gap,
        " size(p_smooth)  = ($(size(fs.p_smooth,1)), $(size(fs.p_smooth,2)), $(size(fs.p_smooth,3)))",
    )
    println(
        io,
        gap,
        " size(E_z)       = ($(size(fs.E_z,1)), $(size(fs.E_z,2)), $(size(fs.E_z,3)))",
    )
    println(
        io,
        gap,
        " size(E_zz)      = ($(size(fs.E_zz,1)), $(size(fs.E_zz,2)), $(size(fs.E_zz,3)), $(size(fs.E_zz,4)))",
    )
    println(
        io,
        gap,
        " size(E_zz_prev) = ($(size(fs.E_zz_prev,1)), $(size(fs.E_zz_prev,2)), $(size(fs.E_zz_prev,3)), $(size(fs.E_zz_prev,4)))",
    )

    return nothing
end

struct TrialFilterSmooth{T<:Real}
    FilterSmooths::Vector{FilterSmooth{T}}
end

Base.getindex(f::TrialFilterSmooth, i::Int) = f.FilterSmooths[i]
function Base.setindex!(
    f::TrialFilterSmooth, value::FilterSmooth{T}, i::Int
) where {T<:Real}
    return (f.FilterSmooths[i] = value)
end
Base.length(f::TrialFilterSmooth) = length(f.FilterSmooths)

mutable struct SufficientStatistics{T<:Real}

    # initial conditions. `init_n` is the effective sample count (e.g.
    # `ntrials` for unweighted fits; `Σₙ w[n,1]` for SLDS-style soft
    # responsibility weights). Stored as `T` rather than `Int` so the
    # weighted aggregator can flow non-integer counts through the M-step
    # without truncation.
    init_n::T
    init_xx::Base.RefValue{PDMat{T,Matrix{T}}}
    init_xy::Matrix{T}
    init_yy::Base.RefValue{PDMat{T,Matrix{T}}}

    # transitions model
    dyn_n::T
    dyn_xx::Base.RefValue{PDMat{T,Matrix{T}}}
    dyn_xy::Matrix{T}
    dyn_yy::Base.RefValue{PDMat{T,Matrix{T}}}

    # observation model
    obs_n::T
    obs_xx::Base.RefValue{PDMat{T,Matrix{T}}}
    obs_xy::Matrix{T}
    obs_yy::Base.RefValue{PDMat{T,Matrix{T}}}
end

Base.@kwdef struct Data{T<:Real}
    y::Array{T,3}
    u::Array{T,3}
    d::Array{T,3}
    trial_pred::Matrix{T} = Matrix{T}(undef, 0, 0)
end

