#=============================================================================
Sampling a switching model with boundary factors.

Entry priors are a proper transition density, so a forward draw honours them
exactly: on a switch `i → j` into an entry-prior state the new costate is drawn
from the prior, `λ_t ~ N(μ + K (x_{t-1} − r⁽ⁱ⁾), P)`, and the state from `j`'s
plant row given it.

Exit bridges are not a density but a condition — "the goal is reached where the
state is left" — like the end-of-trial terminal factor. Drawing from the model a
fit with bridges assumes means drawing the chain, the latents and the
observations *given* every goal is met: `p(s, z, y | goals = 0)`. The switch
times then move to where the goals are reachable, and the latent path is pulled
onto (near) each stage's stable manifold rather than rolled forward along the
unstable Hamiltonian flow. `rand` does that by default for a model with bridges.

Given the path `s`, every factor (transitions, entry densities, terminal and
bridge factors) is quadratic in `z`, so `z | s, goals` is Gaussian: one Newton
step of the observation-free smoother lands on its mean, its negated Hessian is
the block-tridiagonal precision `Λ_s`, and a draw goes through its block
Cholesky. The same factorization gives the path's evidence exactly,

    log Z(s) = log ∫ p(z, goals | s) dz = ℓ_s(ẑ) + (dT/2) log 2π − ½ log det Λ_s,

so the paths can be sampled with `z` integrated out:

- A **left-to-right** chain (banded, starting in stage 1; see
  `banded_transition`) is its changepoints, and each changepoint's conditional
  given the others is enumerable: collapsed Gibbs over them, every candidate
  scored by `log p(s) + log Z(s)`, then `z | s`. This mixes in a few sweeps.
- Any other chain is sampled by blocked Gibbs, alternating `z | s` with
  `s | z, goals` — an HMM whose emissions are each state's transition density at
  the path and whose time-indexed transitions carry the bridge and entry
  potentials (the discrete layer the E-step uses, scored at a point), drawn by
  forward filtering and backward sampling. Exact conditionals, but `s | z` is
  sharp — a sampled path pins its exits — so it mixes slowly and needs more
  sweeps.

Both target `p(s, z | goals = 0)` exactly; `gibbs_sweeps` sets how long they run
from a draw of the prior chain.
=============================================================================#

# Default sweeps of the two samplers: the collapsed one mixes in a few, the
# blocked one, whose path step is pinned by the sampled latents, far more slowly.
const _COLLAPSED_SWEEPS = 20
const _BLOCKED_SWEEPS = 200

"""Whether `rand` draws `slds` given its goals when the caller leaves it to the
model: by default exactly when it has exit bridges."""
_slds_rand_conditional(slds::SLDS, conditional::Nothing) = _slds_has_bridges(slds)
_slds_rand_conditional(::SLDS, conditional::Bool) = conditional

"""
    _draw_entry!(rng, zt, z_prev, sm_prev, sm, ep, u_prev)

A switch into entry-prior state `sm` from `sm_prev`: `λ_t` from the prior about
the left state's reference, then `x_t` from the plant row given it — the density
`_entry_residuals!` scores.
"""
function _draw_entry!(
    rng::AbstractRNG,
    zt::AbstractVector{T},
    z_prev::AbstractVector{T},
    sm_prev::AbstractStateModel,
    sm::LQRStateModel,
    ep::EntryPrior,
    u_prev,
) where {T<:Real}
    n = _plant_dim(sm)
    xr, lr = 1:n, (n + 1):(2n)
    rref = zeros(T, n)
    _state_reference!(rref, sm_prev, u_prev)
    xprev = z_prev[xr]
    λ = rand(rng, MvNormal(ep.μ .+ ep.K * (xprev .- rref), Symmetric(Matrix{T}(ep.P))))
    mean_x = sm.A * xprev .- sm.S * λ .+ sm.h[xr]
    if u_prev !== nothing && size(sm.Bu, 2) > 0
        mean_x .+= sm.Bu[xr, :] * u_prev
    end
    zt[xr] .= rand(rng, MvNormal(mean_x, Symmetric(Matrix{T}(sm.Σ[xr, xr]))))
    zt[lr] .= λ
    return zt
end

"""
    _sample_slds_boundary_trial!(rng, s, z, y, slds_t, state_params, obs_params,
                                 obs_model, ux, uy; conditional, gibbs_sweeps)

One trial of a model with boundary factors. `slds_t` is the trial's model (its
`depends_on` cell, initial state already set from `ux0`), `state_params` its
forward parameters per regime. Draws the prior chain, then either the forward
path with entry draws (`conditional = false`) or the Gibbs chain given the goals,
then the observations given the path.
"""
function _sample_slds_boundary_trial!(
    rng::AbstractRNG,
    s::AbstractVector{Int},
    z::AbstractMatrix{T},
    y,
    slds_t::SLDS,
    state_params,
    obs_params,
    obs_model,
    ux::AbstractMatrix,
    uy;
    conditional::Bool,
    gibbs_sweeps::Union{Nothing,Int},
) where {T<:Real}
    Ti = length(s)
    s[1] = rand(rng, Categorical(slds_t.πₖ))
    for t in 2:Ti
        s[t] = rand(rng, Categorical(slds_t.A[s[t - 1], :]))
    end
    if conditional
        _gibbs_given_goals!(rng, s, z, slds_t, ux; sweeps=gibbs_sweeps)
    else
        _forward_with_entries!(rng, s, z, slds_t, state_params, ux)
    end
    _sample_obs_given_path!(rng, y, z, s, obs_params, obs_model, uy)
    return nothing
end

function _forward_with_entries!(
    rng, s, z::AbstractMatrix{T}, slds_t, state_params, ux
) where {T}
    b = slds_t.boundaries
    p1 = state_params[s[1]]
    z[:, 1] = rand(rng, MvNormal(p1.x0, p1.P0))
    m = size(ux, 1)
    for t in 2:length(s)
        k, i = s[t], s[t - 1]
        u_prev = m > 0 ? view(ux, :, t - 1) : nothing
        if k != i && b !== nothing && b.entry[k] !== nothing
            _draw_entry!(
                rng,
                view(z, :, t),
                view(z, :, t - 1),
                slds_t.LDSs[i].state_model,
                slds_t.LDSs[k].state_model,
                b.entry[k],
                u_prev,
            )
        else
            p = state_params[k]
            z[:, t] = rand(rng, MvNormal(p.A * z[:, t - 1] + p.b + p.B * ux[:, t - 1], p.Q))
        end
    end
    return z
end

function _sample_obs_given_path!(rng, y, z, s, obs_params, obs_model, uy)
    @views for t in eachindex(s)
        y[:, t] = _draw_obs(rng, obs_model, obs_params[s[t]], z[:, t], uy[:, t])
    end
    return y
end

function _sample_obs_given_path!(
    rng, y, z, s, obs_params, obs_model::CompositeObservationModel, uy
)
    for (i, m) in enumerate(values(_models(obs_model)))
        y_m, uy_m = y[i], uy[i]
        @views for t in eachindex(s)
            y_m[:, t] = _draw_obs(rng, m, obs_params[s[t]][i], z[:, t], uy_m[:, t])
        end
    end
    return y
end

"""
    _goal_model(slds_t) -> SLDS

`slds_t` with every emission replaced by a zero-loading one, so its smoother and
discrete scores are the path's prior and goal factors alone (the emission adds a
constant). The boundaries are shared, and the state models are copies so that a
caller's `observe_costate` masking cannot reach them.
"""
function _goal_model(slds_t::SLDS{T}) where {T<:Real}
    d = slds_t.LDSs[1].latent_dim
    members = map(slds_t.LDSs) do lds
        LinearDynamicalSystem(
            deepcopy(lds.state_model),
            GaussianObservationModel(zeros(T, 1, d), ones(T, 1, 1), zeros(T, 1)),
        )
    end
    return SLDS(;
        A=copy(slds_t.A), πₖ=copy(slds_t.πₖ), LDSs=members, boundaries=slds_t.boundaries
    )
end

"""
    _gibbs_given_goals!(rng, s, z, slds_t, ux; sweeps) -> (s, z)

Blocked Gibbs sampling of `p(s, z | goals = 0)` for one trial, from the path `s`
the caller drew from the prior chain: `z | s` first, then `sweeps` rounds of
`s | z` and `z | s`, so the returned pair is one state of the chain.
"""
function _gibbs_given_goals!(
    rng::AbstractRNG,
    s::AbstractVector{Int},
    z::AbstractMatrix{T},
    slds_t::SLDS,
    ux::AbstractMatrix;
    sweeps::Union{Nothing,Int},
) where {T<:Real}
    sweeps === nothing ||
        sweeps >= 0 ||
        throw(ArgumentError("gibbs_sweeps must be non-negative; got $sweeps"))
    goal = _goal_model(slds_t)
    Ti = length(s)
    _prepare_slds!(goal, [Ti])
    ws = SLDSSmoothWorkspace(T, goal, Ti)
    ux_t = size(ux, 1) > 0 ? ux : nothing
    if _is_left_to_right(goal)
        for _ in 1:something(sweeps, _COLLAPSED_SWEEPS)
            _changepoint_sweep!(rng, s, ws, goal, ux_t)
        end
        _sample_path_given_states!(rng, z, ws, goal, s, ux_t)
        return s, z
    end
    dl = _slds_discrete_layer(goal, Ti)
    _sample_path_given_states!(rng, z, ws, goal, s, ux_t)
    for _ in 1:something(sweeps, _BLOCKED_SWEEPS)
        _sample_states_given_path!(rng, s, dl, ws, goal, z, ux_t)
        _sample_path_given_states!(rng, z, ws, goal, s, ux_t)
    end
    return s, z
end

"""Whether the chain only stays or steps forward and starts in stage 1, so a
path is its changepoints."""
function _is_left_to_right(slds::SLDS)
    K = length(slds.LDSs)
    slds.πₖ[1] == 1 || return false
    return all(slds.A[i, j] == 0 for i in 1:K, j in 1:K if !(j == i || j == i + 1))
end

"""`log p(s)` under the chain."""
function _log_path_prior(slds::SLDS{T}, s::AbstractVector{Int}) where {T<:Real}
    lp = log(slds.πₖ[s[1]])
    for t in 2:length(s)
        lp += log(slds.A[s[t - 1], s[t]])
    end
    return T(lp)
end

"""
    _path_precision!(ws, goal, s, ux) -> (L, C, g)

The path's posterior precision `Λ_s` in block-Cholesky form, and the gradient
of its log-density at zero, so that `ẑ = Λ_s⁻¹ g`.
"""
function _path_precision!(
    ws::SLDSSmoothWorkspace{T}, goal::SLDS{T}, s::AbstractVector{Int}, ux
) where {T<:Real}
    Ti = length(s)
    d = goal.LDSs[1].latent_dim
    w, ew, pw = _path_weights(goal, s)
    y = zeros(T, 1, Ti)
    origin = zeros(T, d, Ti)
    g = copy(gradient!(ws, goal, origin, y, w, ux, nothing; ew=ew, pw=pw))
    hessian!(ws, goal, origin, y, w, nothing; ew=ew, pw=pw)
    D = [Matrix{T}(-ws.btd.H_diag[t]) for t in 1:Ti]
    Ssub = [Matrix{T}(-ws.btd.H_sub[t]) for t in 1:(Ti - 1)]
    L, C = _btd_cholesky(D, Ssub)
    return L, C, g
end

"""
    _path_log_evidence(ws, goal, s, ux) -> T

`log ∫ p(z, goals | s) dz`, exact: the density is Gaussian in `z`, so it is the
log-density at its mode plus the Laplace volume. The zero-loading emission's
constant is included, which is the same for every path. A path the goals make
impossible to factor scores `-Inf`.
"""
function _path_log_evidence(
    ws::SLDSSmoothWorkspace{T}, goal::SLDS{T}, s::AbstractVector{Int}, ux
) where {T<:Real}
    Ti = length(s)
    d = goal.LDSs[1].latent_dim
    L, C, g = try
        _path_precision!(ws, goal, s, ux)
    catch err
        err isa PosDefException || rethrow()
        return -T(Inf)
    end
    zhat = _btd_solve(L, C, g)
    w, ew, pw = _path_weights(goal, s)
    ℓ = sum(joint_loglikelihood!(ws, goal, zhat, zeros(T, 1, Ti), w, ux; ew=ew, pw=pw))
    logdetΛ = 2 * sum(sum(log, diag(Lt)) for Lt in L)
    return ℓ + T(d * Ti) / 2 * log(T(2π)) - logdetΛ / 2
end

"""
    _changepoint_sweep!(rng, s, ws, goal, ux) -> s

One collapsed Gibbs sweep over a left-to-right path's changepoints: `τ_k`, the
first bin of stage `k + 1` (`T + 1` when the trial never reaches it), drawn in
turn from its exact conditional given the others, each candidate scored by
`log p(s) + log Z(s)` with the latents integrated out.
"""
function _changepoint_sweep!(
    rng::AbstractRNG, s::AbstractVector{Int}, ws::SLDSSmoothWorkspace{T}, goal::SLDS{T}, ux
) where {T<:Real}
    K, Ti = length(goal.LDSs), length(s)
    τ = [something(findfirst(>=(k + 1), s), Ti + 1) for k in 1:(K - 1)]
    path(τ) = [1 + count(<=(t), τ) for t in 1:Ti]
    for k in 1:(K - 1)
        prev = k == 1 ? 1 : τ[k - 1]
        prev > Ti && continue             # stage k is never reached
        next = k == K - 1 ? Ti + 1 : τ[k + 1]
        candidates = collect((prev + 1):min(next - 1, Ti))
        next > Ti && push!(candidates, Ti + 1)
        isempty(candidates) && continue
        scores = map(candidates) do c
            τ[k] = c
            sc = path(τ)
            _log_path_prior(goal, sc) + _path_log_evidence(ws, goal, sc, ux)
        end
        τ[k] = candidates[_draw_log_categorical(rng, scores)]
    end
    s .= path(τ)
    return s
end

"""
The one-hot weights a fixed path puts on the smoother's factors: responsibilities
`w`, exits `ew` (bridged states only) and entries `pw` (entry-prior states only).
"""
function _path_weights(goal::SLDS{T}, s::AbstractVector{Int}) where {T<:Real}
    K, Ti = length(goal.LDSs), length(s)
    w = zeros(T, K, Ti)
    for t in 1:Ti
        w[s[t], t] = one(T)
    end
    b = goal.boundaries
    ew = nothing
    pw = nothing
    if _slds_has_bridges(goal)
        ew = zeros(T, K, Ti)
        for t in 1:(Ti - 1)
            k = s[t]
            (b.bridge[k] && s[t + 1] != k) && (ew[k, t] = one(T))
        end
    end
    if _slds_has_entries(goal)
        pw = zeros(T, K, K, Ti)
        for t in 2:Ti
            i, j = s[t - 1], s[t]
            (i != j && b.entry[j] !== nothing) && (pw[i, j, t] = one(T))
        end
    end
    return w, ew, pw
end

"""
    _sample_path_given_states!(rng, z, ws, goal, s, ux) -> z

`z ~ p(z | s, goals = 0)`. Every factor is quadratic in `z` given the path, so
one Newton step from zero is the mean, and the negated Hessian is the precision.
"""
function _sample_path_given_states!(
    rng::AbstractRNG,
    z::AbstractMatrix{T},
    ws::SLDSSmoothWorkspace{T},
    goal::SLDS{T},
    s::AbstractVector{Int},
    ux,
) where {T<:Real}
    d, Ti = size(z)
    L, C, g = _path_precision!(ws, goal, s, ux)
    mean = _btd_solve(L, C, g)
    noise = _btd_draw(L, C, randn(rng, T, d, Ti))
    z .= mean .+ noise
    return z
end

"""
    _btd_cholesky(D, S) -> (L, C)

The block Cholesky factor of the symmetric positive definite block-tridiagonal
matrix with diagonal blocks `D[t]` and sub-diagonal blocks `S[t] = Λ[t+1, t]`:
lower block-bidiagonal, diagonal blocks `L[t]` (lower triangular) and
sub-diagonal blocks `C[t]`, so that `Λ = 𝐋 𝐋ᵀ`.
"""
function _btd_cholesky(D::AbstractVector{<:AbstractMatrix{T}}, S) where {T<:Real}
    Ti = length(D)
    L = Vector{LowerTriangular{T,Matrix{T}}}(undef, Ti)
    C = Vector{Matrix{T}}(undef, Ti - 1)
    L[1] = cholesky(Symmetric(D[1])).L
    for t in 1:(Ti - 1)
        C[t] = S[t] / transpose(L[t])
        L[t + 1] = cholesky(Symmetric(D[t + 1] - C[t] * transpose(C[t]))).L
    end
    return L, C
end

"""`Λ \\ g` through the block Cholesky factor (columns of `g` are timesteps)."""
function _btd_solve(L, C, g::AbstractMatrix{T}) where {T<:Real}
    Ti = size(g, 2)
    v = similar(g)
    v[:, 1] = L[1] \ g[:, 1]
    for t in 2:Ti
        v[:, t] = L[t] \ (g[:, t] - C[t - 1] * v[:, t - 1])
    end
    return _btd_back!(L, C, v)
end

"""`𝐋⁻ᵀ ε`: standard-normal `ε` mapped to a draw with covariance `Λ⁻¹`."""
_btd_draw(L, C, ε::AbstractMatrix) = _btd_back!(L, C, copy(ε))

function _btd_back!(L, C, v::AbstractMatrix)
    Ti = size(v, 2)
    v[:, Ti] = transpose(L[Ti]) \ v[:, Ti]
    for t in (Ti - 1):-1:1
        v[:, t] = transpose(L[t]) \ (v[:, t] - transpose(C[t]) * v[:, t + 1])
    end
    return v
end

"""
    _sample_states_given_path!(rng, s, dl, ws, goal, z, ux) -> s

`s ~ p(s | z, goals = 0)`: each state's transition (and terminal) density at the
path, the bridge and entry potentials on the time-indexed transitions, then
forward filtering and backward sampling.
"""
function _sample_states_given_path!(
    rng::AbstractRNG,
    s::AbstractVector{Int},
    dl::SLDSDiscreteLayer{T},
    ws::SLDSSmoothWorkspace{T},
    goal::SLDS{T},
    z::AbstractMatrix{T},
    ux,
) where {T<:Real}
    Ti = length(s)
    K = length(goal.LDSs)
    _slds_fill_logL!(
        goal,
        nothing,
        nothing,
        dl,
        [zeros(T, 1, Ti)],
        _ -> z,
        _slds_solo_pool(ws),
        _slds_trial_plan(nothing, 1, 1);
        seq_ends=[Ti],
        ux=ux === nothing ? nothing : [ux],
        exit_logL=_has_bridges(dl) ? dl.exit_logL : nothing,
        bridge=dl.bridge,
        entry_logL=_has_entries(dl) ? dl.entry_logL : nothing,
        entry=dl.entry,
    )
    # Forward filter, in logs.
    logα = zeros(T, K, Ti)
    logα[:, 1] .= log.(dl.πₖ) .+ dl.logL[:, 1]
    for t in 2:Ti
        At = HMMs.transition_matrix(dl, t)
        prev = logα[:, t - 1]
        top = maximum(prev)
        α = exp.(prev .- top)
        for j in 1:K
            acc = zero(T)
            for i in 1:K
                acc += α[i] * At[i, j]
            end
            logα[j, t] = top + log(acc) + dl.logL[j, t]
        end
    end
    # Backward sample.
    s[Ti] = _draw_log_categorical(rng, view(logα, :, Ti))
    for t in (Ti - 1):-1:1
        At = HMMs.transition_matrix(dl, t + 1)
        lp = [logα[i, t] + log(At[i, s[t + 1]]) for i in 1:K]
        s[t] = _draw_log_categorical(rng, lp)
    end
    return s
end

function _draw_log_categorical(rng::AbstractRNG, lp::AbstractVector{T}) where {T<:Real}
    top = maximum(lp)
    isfinite(top) || throw(
        NumericalStabilityError(
            "rand", "no discrete state is reachable given the sampled path and goals"
        ),
    )
    p = exp.(lp .- top)
    return rand(rng, Categorical(p ./ sum(p)))
end
