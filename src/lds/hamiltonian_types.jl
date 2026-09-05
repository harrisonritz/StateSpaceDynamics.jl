#=============================================================================
Hamiltonian (inverse-LQR) latents — model type, derived cache, and structure
utilities.

    Model:      HamiltonianStateModel, HamiltonianCostSchedule
    Cache:      HamiltonianCache, refresh!
    Structure:  hamiltonian_matrix, symplectic_matrix, symplectic_defect,
                riccati_solution, closed_loop_dynamics, lqr_parameters,
                rescale_costate!

The E-step kernels live in `hamiltonian_latents.jl` and the M-step in
`hamiltonian_mstep.jl`.
=============================================================================#

"""
    HamiltonianFitFlags(; A=true, S=true, Qc=true, h=true, Bu=true, terminal=true)

Which structural parameters of a [`HamiltonianStateModel`](@ref) the M-step is
free to move. Every flag defaults to `true`.

Freezing is what makes the *inverse* problem inverse: with a known plant you set
`A = false, S = false` and let EM estimate the cost matrices alone. A frozen
parameter keeps whatever value it was constructed with — its gradient block is
never packed, so freezing also shrinks the M-step problem rather than merely
projecting its solution.

`terminal` gates the terminal factor's own offset `hf` (its covariance `Σf`
follows the enclosing `fit_bool`'s noise slot). The terminal *cost* is one of the
`Qc` matrices, so it moves with `Qc` — and the terminal factor always contributes
to the objective when the model has one, since leaving it out would make the
gradient with respect to that cost matrix wrong.
"""
struct HamiltonianFitFlags
    A::Bool
    S::Bool
    Qc::Bool
    h::Bool
    Bu::Bool
    terminal::Bool
end

function HamiltonianFitFlags(;
    A::Bool=true,
    S::Bool=true,
    Qc::Bool=true,
    h::Bool=true,
    Bu::Bool=true,
    terminal::Bool=true,
)
    return HamiltonianFitFlags(A, S, Qc, h, Bu, terminal)
end

"""
    HamiltonianCache{T}

Everything the smoother and the ELBO need, derived from a
[`HamiltonianStateModel`](@ref)'s natural parameters by [`refresh!`](@ref).

The latent state is `z_t = [x_t; λ_t]` (`2n`), and the forward transition
`z_{t+1} = M_k z_t + b + B u_t + noise` is what the block-tridiagonal smoother
consumes. Only `M` varies across cost regimes: the noise map `G` depends on `A`
and `S` alone, so `Qfwd`, `bfwd` and `Bfwd` are shared by every regime.

# Fields
- `M::Vector{Matrix{T}}`: one `2n × 2n` symplectic transition per cost regime.
- `G::Matrix{T}`: `[I  S A⁻ᵀ; 0  −A⁻ᵀ]`, the map from mixed-coordinate
    innovations to forward ones.
- `AinvT::Matrix{T}`, `logabsdetA::T`: `A⁻ᵀ` and `log|det A|` (the Jacobian term
    the M-step objective carries).
- `Qfwd::DensePDMat{T}`: `G Σ Gᵀ`, the forward process-noise covariance.
- `bfwd`, `Bfwd`: `G h` and `G Bu`.
- `negQinv`, `QinvM`, `MtQinv`, `negMtQinvM`, `cQ`: Cholesky-derived templates
    for the gradient and Hessian blocks, the per-regime ones indexed by regime.
- `Sf_PD`, `Lf`, `negLtSL`, `LtSinv`, `cF`: the terminal factor's covariance,
    its design matrix `Λf = [−Q_f  I]`, and the derived curvature / gradient
    templates. Present (as identity placeholders) even when the model carries no
    terminal factor.
"""
mutable struct HamiltonianCache{T<:Real}
    const n::Int
    const M::Vector{Matrix{T}}
    const G::Matrix{T}
    const AinvT::Matrix{T}
    logabsdetA::T
    Qfwd::DensePDMat{T}
    const bfwd::Vector{T}
    const Bfwd::Matrix{T}
    const negQinv::Matrix{T}
    const QinvM::Vector{Matrix{T}}
    const MtQinv::Vector{Matrix{T}}
    const negMtQinvM::Vector{Matrix{T}}
    cQ::T
    Sf_PD::DensePDMat{T}
    const Lf::Matrix{T}
    const negLtSL::Matrix{T}
    const LtSinv::Matrix{T}
    cF::T
end

function HamiltonianCache(::Type{T}, n::Int, nregimes::Int, ux_dim::Int) where {T<:Real}
    d = 2n
    return HamiltonianCache{T}(
        n,
        [zeros(T, d, d) for _ in 1:nregimes],
        zeros(T, d, d),
        zeros(T, n, n),
        zero(T),
        PDMat(Matrix{T}(I, d, d)),
        zeros(T, d),
        zeros(T, d, ux_dim),
        zeros(T, d, d),
        [zeros(T, d, d) for _ in 1:nregimes],
        [zeros(T, d, d) for _ in 1:nregimes],
        [zeros(T, d, d) for _ in 1:nregimes],
        zero(T),
        PDMat(Matrix{T}(I, n, n)),
        zeros(T, n, d),
        zeros(T, d, d),
        zeros(T, d, n),
        zero(T),
    )
end

"""
    HamiltonianStateModel{T,M,V} <: AbstractGaussianStateModel{T}

Latent-state model for **inverse LQR**: the latent state is the LQR
state–costate pair `z_t = [x_t; λ_t]` and the transition is constrained to the
Hamiltonian (symplectic) form implied by a linear-quadratic optimal control
problem, so that fitting the SSM *is* recovering the plant and the cost.

## The structure being fitted

For the discrete-time problem

```math
\\min_u \\sum_{t=1}^{T-1} \\tfrac12 (x_t' Q_t x_t + u_t' R u_t)
        + \\tfrac12 x_T' Q_T x_T
\\quad\\text{s.t.}\\quad x_{t+1} = A x_t + B u_t
```

stationarity of the Lagrangian gives `u_t = −R⁻¹Bᵀλ_{t+1}` and the two-point
boundary value problem

```math
\\begin{bmatrix} x_{t+1} \\\\ \\lambda_t \\end{bmatrix}
  = \\mathcal{E}_t \\begin{bmatrix} x_t \\\\ \\lambda_{t+1} \\end{bmatrix},
\\qquad
\\mathcal{E}_t = \\begin{bmatrix} A & -S \\\\ Q_t & A^\\top \\end{bmatrix},
\\qquad S := B R^{-1} B^\\top,
```

with the terminal condition `λ_T = Q_T x_T`. `S` and `Q_t` are symmetric. This
**mixed** form is linear in the free parameters, which is what the M-step
estimates in; the smoother instead needs a forward chain on `z`, obtained by
eliminating `λ_{t+1}`:

```math
M_t = \\begin{bmatrix} A + S A^{-\\top} Q_t & -S A^{-\\top} \\\\
                       -A^{-\\top} Q_t      &  A^{-\\top} \\end{bmatrix},
\\qquad M_t^\\top J M_t = J,
\\qquad J = \\begin{bmatrix} 0 & I \\\\ -I & 0 \\end{bmatrix}.
```

`A` must therefore be invertible — true of any discretized plant, and checked at
construction.

## Noise

The model is stochastic in the **mixed** coordinates,

```math
\\begin{bmatrix} x_{t+1} \\\\ \\lambda_t \\end{bmatrix}
  = \\mathcal{E}_t \\begin{bmatrix} x_t \\\\ \\lambda_{t+1} \\end{bmatrix}
  + h + B_u u_t + \\varepsilon_t, \\qquad \\varepsilon_t \\sim N(0, \\Sigma),
```

so `Σ`'s leading block is genuine plant process noise and its trailing block is
*costate slack* — how far from exactly optimal the behavior is. The equivalent
forward noise is `G Σ Gᵀ` with `G = [I  S A⁻ᵀ; 0  −A⁻ᵀ]`, and since `G` is
invertible **the costate must carry process noise**: a singular `Σ` makes the
forward process-noise covariance singular and the smoother's precision
undefined. A free `Σ` spans exactly the same model class as a free forward
covariance, so nothing is lost by parameterizing it here.

## Time-varying cost

`Qc` holds `K` cost matrices and `schedule` says which one each timestep uses:
`schedule[t]` indexes the transition `t → t+1` for `t < T`, and `schedule[T]`
the terminal factor. A running-plus-terminal cost is
`HamiltonianCostSchedule(T; terminal=true)`; see [`cost_schedule`](@ref).
An empty `schedule` means "regime 1 everywhere", which requires `K == 1`.

## Terminal condition

When `terminal` is set, `λ_T = Q_{k_T} x_T` enters as a **soft pseudo-observation**
— `0 = λ_T − Q_{k_T} x_T − h_f + ε_f`, `ε_f ~ N(0, Σ_f)`, with `Σ_f → 0` the hard
boundary condition. It is a factor of the emission, not of the chain: the model
is the proper joint `p(z) p(y | z) p(y^{term} | z_T)`, `elbo` and `loglikelihood`
report `log p(y, y^{term} = 0)`, and no extra normalizer is involved.

Two consequences are worth stating plainly.

*It is what makes the model well-behaved.* A symplectic matrix has reciprocal
eigenvalue pairs, so the forward transition is unstable by construction and the
unconditioned chain diverges like `ρ(M)^T`. Conditioning on the terminal factor
removes exactly the unstable directions — that is what a boundary condition does
to a two-point boundary value problem — so the posterior, and `rand`'s
terminal-conditioned draw, stay bounded.

*It is a conditioning event, so recovery from `rand` output is biased.* `rand`
draws `p(z, y | y^{term} = 0)`, which is what trials actually look like, while the
fitted objective is `p(y, y^{term} = 0)`; the two differ by `p(y^{term} = 0 | θ)`,
which depends on the parameters. Maximizing the latter on data drawn from the
former is therefore a selection effect, not an unbiased estimator, and a
self-consistency recovery check will not land on the generating parameters. When
that matters, encode the terminal cost as the last *regime of the transition
schedule* instead (`terminal = false`, and give the final transitions their own
`Qc`): that model is a proper directed chain and recovers its own parameters.

## Identifiability

`(λ, S, Q) → (cλ, c⁻¹S, cQ)` leaves the state dynamics unchanged: this is the
classical inverse-optimal-control scale invariance (scaling a cost does not
change the policy). The product `S·Q` is identified, `S` and `Q` separately are
not. Use [`rescale_costate!`](@ref) to put a fit in a canonical scale before
comparing runs.

# Fields
- `A::M`: `n × n` plant dynamics. Invertible.
- `S::M`: `n × n` symmetric `B R⁻¹ Bᵀ` — the control authority weighted by the
    control cost. `B` and `R` are not separately identified; only `S` is.
- `Qc::Vector{M}`: `K` symmetric `n × n` state-cost matrices.
- `schedule::Vector{Int}`: per-timestep cost index, or empty for a single cost.
- `terminal::Bool`: whether the terminal costate factor is active.
- `Σ::M`: `2n × 2n` positive-definite mixed-coordinate innovation covariance.
- `h::V`: `2n` mixed-coordinate bias. Its costate half is `−Q x*` for a
    tracking target `x*`.
- `Bu::M`: `2n × ux_dim` mixed-coordinate input matrix (exogenous drift or a
    time-varying target — *not* the LQR control, which has been eliminated).
- `Σf::M`, `hf::V`: terminal factor covariance (`n × n`) and offset (`n`).
- `x0::V`, `P0::M`: prior on `z₁ = [x₁; λ₁]` (`2n`).
- `observe_costate::Bool`: whether the emission may read the costate. `false`
    (the default) pins the costate columns of `C` at zero.
- `fit_flags::HamiltonianFitFlags`: which structural parameters move.
- `mstep_iters::Int`: L-BFGS iterations per M-step.
- `P0_prior`, `x0_prior`: optional priors on the initial state, as on
    [`GaussianStateModel`](@ref).
- `cache::HamiltonianCache{T}`: derived forward parameters. Rebuilt by
    [`refresh!`](@ref), which the constructors and the M-step call for you —
    call it yourself after mutating a field by hand.

The enclosing `LinearDynamicalSystem`'s `fit_bool` still has its usual four
state slots: `[x0, P0, structure, noise]`, where `structure` gates the
`(A, S, Qc, h, Bu)` update (refined by `fit_flags`) and `noise` gates `Σ`/`Σf`.

See also [`lqr_parameters`](@ref), [`riccati_solution`](@ref),
[`symplectic_matrix`](@ref).
"""
mutable struct HamiltonianStateModel{T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
               AbstractGaussianStateModel{T}
    A::M
    S::M
    Qc::Vector{M}
    schedule::Vector{Int}
    terminal::Bool
    Σ::M
    h::V
    Bu::M
    Σf::M
    hf::V
    x0::V
    P0::M
    observe_costate::Bool
    fit_flags::HamiltonianFitFlags
    mstep_iters::Int
    P0_prior::Union{Nothing,IWPrior{T}}
    x0_prior::Union{Nothing,MNPrior{T,Matrix{T}}}
    depends_on::Union{Nothing,NamedTuple}
    variants::Union{Nothing,Vector{HamiltonianStateModel{T,M,V}}}
    cache::HamiltonianCache{T}
end

"""
    _plant_dim(sm) -> Int

The plant dimension `n`. The latent state is twice this.
"""
@inline _plant_dim(sm::HamiltonianStateModel) = size(sm.A, 1)

_state_latent_dim(sm::HamiltonianStateModel) = 2 * size(sm.A, 1)
_state_ux_dim(sm::HamiltonianStateModel) = size(sm.Bu, 2)

"""
    _nregimes(sm) -> Int

How many distinct cost matrices the model carries.
"""
@inline _nregimes(sm::HamiltonianStateModel) = length(sm.Qc)

"""
    _regime(sm, t) -> Int

Cost index in force at timestep `t`: `schedule[t]` for a scheduled model, and 1
when the schedule is empty (a single cost everywhere).
"""
@inline function _regime(sm::HamiltonianStateModel, t::Int)
    sched = sm.schedule
    return isempty(sched) ? 1 : sched[t]
end

"""
    cost_schedule(tsteps; terminal=false, onset=1, nregimes=terminal ? 2 : 1)

Build the per-timestep cost index vector for the common shapes.

- `cost_schedule(T)` — one running cost on every transition.
- `cost_schedule(T; terminal=true)` — running cost 1 on transitions
  `1 … T-1`, terminal cost 2 at `T`.
- `cost_schedule(T; terminal=true, onset=k)` — cost 1 (typically zero, a
  no-cost epoch such as a delay period) up to `k-1`, cost 2 from `k` on, and
  terminal cost 3 at `T`.

The result is a plain `Vector{Int}`; write your own when you want a shape this
does not cover. Entry `t < T` indexes the transition `t → t+1`; entry `T` is
read only by the terminal factor.
"""
function cost_schedule(
    tsteps::Integer;
    terminal::Bool=false,
    onset::Integer=1,
    nregimes::Integer=(onset > 1 ? 2 : 1) + (terminal ? 1 : 0),
)
    tsteps >= 2 || throw(ArgumentError("cost_schedule needs tsteps ≥ 2; got $tsteps"))
    (1 <= onset <= tsteps) ||
        throw(ArgumentError("cost_schedule onset must lie in 1:$tsteps; got $onset"))
    sched = fill(1, Int(tsteps))
    running = 1
    if onset > 1
        running = 2
        for t in Int(onset):Int(tsteps)
            sched[t] = running
        end
    end
    if terminal
        sched[end] = running + 1
    end
    maxk = maximum(sched)
    maxk <= nregimes || throw(
        ArgumentError(
            "cost_schedule produced $maxk regimes but `nregimes = $nregimes` was asked " *
            "for; drop `nregimes` to take the derived value",
        ),
    )
    return sched
end

#=
Structural checks shared by the constructor and `validate_LDS`. Kept separate
from `_validate_state_model` so the constructor can fail before building a cache
against nonsense.
=#
function _check_hamiltonian_structure(
    A::AbstractMatrix{T},
    S::AbstractMatrix{T},
    Qc::AbstractVector{<:AbstractMatrix{T}},
    schedule::AbstractVector{Int},
    terminal::Bool,
) where {T<:Real}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatchError("Hamiltonian A columns", n, size(A, 2)))
    size(S) == (n, n) || throw(DimensionMismatchError("Hamiltonian S rows", n, size(S, 1)))
    isempty(Qc) &&
        throw(ArgumentError("a HamiltonianStateModel needs at least one cost matrix `Qc`"))
    for (k, Q) in enumerate(Qc)
        size(Q) == (n, n) ||
            throw(DimensionMismatchError("Hamiltonian Qc[$k] rows", n, size(Q, 1)))
        asym = maximum(abs, Q .- transpose(Q); init=zero(T))
        asym <= 1e-8 * max(one(T), maximum(abs, Q; init=one(T))) ||
            throw(NotSymmetricError("Qc[$k]", Float64(asym)))
    end
    asym_S = maximum(abs, S .- transpose(S); init=zero(T))
    asym_S <= 1e-8 * max(one(T), maximum(abs, S; init=one(T))) ||
        throw(NotSymmetricError("S", Float64(asym_S)))

    #=
    Invertibility is not a numerical nicety: the forward symplectic transition
    is built from `A⁻ᵀ`, so a singular plant has no forward Markov
    representation at all and the smoother cannot run.
    =#
    F = lu(Matrix(A); check=false)
    issuccess(F) || throw(
        NumericalStabilityError(
            "A",
            "the Hamiltonian plant matrix is singular. The forward symplectic " *
            "transition is built from A⁻ᵀ, so `A` must be invertible; a discretized " *
            "plant (A = exp(Aᶜ·dt)) always is",
        ),
    )
    logdetA, _ = logabsdet(F)
    isfinite(logdetA) ||
        throw(NumericalStabilityError("A", "the plant matrix has a non-finite log|det|"))

    K = length(Qc)
    if isempty(schedule)
        K == 1 || throw(
            ArgumentError(
                "a HamiltonianStateModel with $K cost matrices needs a `schedule` saying " *
                "which timesteps use which; an empty schedule means one cost everywhere",
            ),
        )
    else
        length(schedule) >= 2 ||
            throw(ArgumentError("a cost schedule must cover at least 2 timesteps"))
        for (t, k) in enumerate(schedule)
            (1 <= k <= K) || throw(
                ArgumentError(
                    "cost schedule entry $t is $k, outside 1:$K (the number of `Qc` " *
                    "matrices)",
                ),
            )
        end
        #=
        A cost index the schedule never reaches gets no gradient and would sit
        at its starting value forever, silently. Transitions read `1:end-1`;
        the last entry is read only when there is a terminal factor.
        =#
        used = Set(view(schedule, 1:(length(schedule) - 1)))
        terminal && push!(used, schedule[end])
        for k in 1:K
            k in used || @warn(
                "Qc[$k] is never used by the cost schedule, so the M-step cannot move " *
                    "it. Drop it, or point some timestep at it.",
                maxlog = 3
            )
        end
    end
    return nothing
end

"""
    HamiltonianStateModel(A, S, Qc, Σ; kwargs...)

Build a Hamiltonian (inverse-LQR) state model from the plant `A` (`n × n`,
invertible), the control term `S = B R⁻¹ Bᵀ` (`n × n`, symmetric), the state
cost(s) `Qc` (one symmetric `n × n` matrix, or a vector of them), and the
mixed-coordinate innovation covariance `Σ` (`2n × 2n`, positive definite).

# Keywords
- `schedule`: per-timestep cost index (see [`cost_schedule`](@ref)). Required
  when `Qc` holds more than one matrix.
- `terminal::Bool = false`: enable the terminal costate factor. Its cost is
  `Qc[schedule[T]]`.
- `Σf`, `hf`: terminal factor covariance and offset. Default `I` and `0`.
- `h`, `Bu`: mixed-coordinate bias (`2n`) and input matrix (`2n × ux_dim`).
- `x0`, `P0`: prior on `z₁ = [x₁; λ₁]`. Default `0` and `I`.
- `observe_costate::Bool = false`: let the emission read the costate.
- `fit_flags`, `mstep_iters`, `P0_prior`, `x0_prior`: see the type docstring.

Everything derived (the symplectic transitions, the forward noise) is built
here; you never pass it in.
"""
function HamiltonianStateModel(
    A::AbstractMatrix{T},
    S::AbstractMatrix{T},
    Qc::Union{AbstractMatrix{T},AbstractVector{<:AbstractMatrix{T}}},
    Σ::AbstractMatrix{T};
    schedule::AbstractVector{<:Integer}=Int[],
    terminal::Bool=false,
    Σf::Union{Nothing,AbstractMatrix{T}}=nothing,
    hf::Union{Nothing,AbstractVector{T}}=nothing,
    h::Union{Nothing,AbstractVector{T}}=nothing,
    Bu::Union{Nothing,AbstractMatrix{T}}=nothing,
    x0::Union{Nothing,AbstractVector{T}}=nothing,
    P0::Union{Nothing,AbstractMatrix{T}}=nothing,
    observe_costate::Bool=false,
    fit_flags::HamiltonianFitFlags=HamiltonianFitFlags(),
    mstep_iters::Int=100,
    P0_prior::Union{Nothing,IWPrior{T}}=nothing,
    x0_prior::Union{Nothing,MNPrior{T,Matrix{T}}}=nothing,
) where {T<:Real}
    n = size(A, 1)
    d = 2n
    Qc_vec = Qc isa AbstractMatrix ? [Qc] : collect(Qc)
    sched = collect(Int, schedule)

    _check_hamiltonian_structure(A, S, Qc_vec, sched, terminal)

    size(Σ) == (d, d) || throw(DimensionMismatchError("Hamiltonian Σ rows", d, size(Σ, 1)))

    h_v = h === nothing ? zeros(T, d) : h
    Bu_m = Bu === nothing ? zeros(T, d, 0) : Bu
    x0_v = x0 === nothing ? zeros(T, d) : x0
    P0_m = P0 === nothing ? Matrix{T}(I, d, d) : P0
    Σf_m = Σf === nothing ? Matrix{T}(I, n, n) : Σf
    hf_v = hf === nothing ? zeros(T, n) : hf

    length(h_v) == d || throw(DimensionMismatchError("Hamiltonian h", d, length(h_v)))
    size(Bu_m, 1) == d ||
        throw(DimensionMismatchError("Hamiltonian Bu rows", d, size(Bu_m, 1)))
    length(x0_v) == d || throw(DimensionMismatchError("Hamiltonian x0", d, length(x0_v)))
    size(P0_m) == (d, d) ||
        throw(DimensionMismatchError("Hamiltonian P0 rows", d, size(P0_m, 1)))
    size(Σf_m) == (n, n) ||
        throw(DimensionMismatchError("Hamiltonian Σf rows", n, size(Σf_m, 1)))
    length(hf_v) == n || throw(DimensionMismatchError("Hamiltonian hf", n, length(hf_v)))

    mstep_iters >= 1 ||
        throw(ArgumentError("mstep_iters must be at least 1; got $mstep_iters"))

    MT = typeof(A)
    VT = typeof(h_v)
    sm = HamiltonianStateModel{T,MT,VT}(
        A,
        S,
        Qc_vec,
        sched,
        terminal,
        Σ,
        h_v,
        Bu_m,
        Σf_m,
        hf_v,
        x0_v,
        P0_m,
        observe_costate,
        fit_flags,
        mstep_iters,
        P0_prior,
        x0_prior,
        nothing,
        nothing,
        HamiltonianCache(T, n, length(Qc_vec), size(Bu_m, 2)),
    )
    refresh!(sm)
    return sm
end

"""
    refresh!(sm::HamiltonianStateModel) -> sm

Rebuild the derived cache from the natural parameters: the per-regime symplectic
transitions `M_k`, the noise map `G`, the forward noise / bias / input, and the
Cholesky-derived gradient and Hessian templates.

Called for you by the constructors and at the end of every M-step. Call it
yourself after assigning to `A`, `S`, `Qc`, `Σ`, `h`, `Bu`, `Σf` or `hf` by
hand — the smoother reads the cache, not the fields.

# Throws
- `NumericalStabilityError` when `A` has become singular
- `PosDefException` when `Σ` or `Σf` is not positive definite
"""
function refresh!(sm::HamiltonianStateModel{T}) where {T<:Real}
    n = _plant_dim(sm)
    d = 2n
    c = sm.cache
    A = sm.A
    S = sm.S

    F = lu(Matrix(A); check=false)
    issuccess(F) || throw(
        NumericalStabilityError(
            "A",
            "the plant matrix has become singular; the symplectic transition needs A⁻ᵀ",
        ),
    )
    logdetA, _ = logabsdet(F)
    c.logabsdetA = T(logdetA)
    # A⁻ᵀ = (A⁻¹)ᵀ.
    copyto!(c.AinvT, transpose(inv(F)))
    AinvT = c.AinvT

    #=
    M_k = [A + S A⁻ᵀ Q_k   −S A⁻ᵀ ]     with the right block column, and hence
          [   −A⁻ᵀ Q_k       A⁻ᵀ  ]     the noise map G = [I  −M₁₂; 0  −M₂₂],
    independent of the regime. Building M₁₁ as `A − M₁₂ Q_k` and M₂₁ as
    `−M₂₂ Q_k` reuses those two blocks instead of recomputing `S A⁻ᵀ`.
    =#
    M12 = -S * AinvT                       # n × n
    M22 = AinvT
    for (k, Qk) in enumerate(sm.Qc)
        Mk = c.M[k]
        @views begin
            copyto!(Mk[1:n, 1:n], A)
            mul!(Mk[1:n, 1:n], M12, Qk, -one(T), one(T))     # A − M₁₂ Q_k
            copyto!(Mk[1:n, (n + 1):d], M12)
            mul!(Mk[(n + 1):d, 1:n], M22, Qk, -one(T), zero(T))
            copyto!(Mk[(n + 1):d, (n + 1):d], M22)
        end
    end

    fill!(c.G, zero(T))
    @views begin
        for i in 1:n
            c.G[i, i] = one(T)
        end
        c.G[1:n, (n + 1):d] .= .-M12
        c.G[(n + 1):d, (n + 1):d] .= .-M22
    end
    G = c.G

    # Forward noise / bias / input: Qfwd = G Σ Gᵀ, bfwd = G h, Bfwd = G Bu.
    GS = G * Matrix(sm.Σ)
    Qfwd = GS * transpose(G)
    c.Qfwd = PDMat(Symmetrize!(Qfwd))
    mul!(c.bfwd, G, sm.h)
    size(c.Bfwd, 2) > 0 && mul!(c.Bfwd, G, sm.Bu)

    Qchol = c.Qfwd.chol
    Imat = Matrix{T}(I, d, d)
    copyto!(c.negQinv, Imat)
    ldiv!(Qchol, c.negQinv)
    c.negQinv .*= -one(T)

    for k in eachindex(sm.Qc)
        copyto!(c.QinvM[k], c.M[k])
        ldiv!(Qchol, c.QinvM[k])                 # Qfwd⁻¹ M_k
        copyto!(c.MtQinv[k], transpose(c.QinvM[k]))  # M_kᵀ Qfwd⁻¹ (Qfwd⁻¹ symmetric)
        mul!(c.negMtQinvM[k], transpose(c.M[k]), c.QinvM[k])
        c.negMtQinvM[k] .*= -one(T)
    end
    c.cQ = -T(0.5) * (T(d) * log(T(2π)) + logdet(c.Qfwd))

    # Terminal factor: 0 = Λf z_T − hf + ε,  Λf = [−Q_f  I].
    Σf_w = Matrix{T}(sm.Σf)
    c.Sf_PD = PDMat(Symmetrize!(Σf_w))
    fill!(c.Lf, zero(T))
    if sm.terminal
        kf = isempty(sm.schedule) ? 1 : sm.schedule[end]
        @views begin
            c.Lf[:, 1:n] .= .-sm.Qc[kf]
            for i in 1:n
                c.Lf[i, n + i] = one(T)
            end
        end
    end
    copyto!(c.LtSinv, transpose(c.Lf))
    rdiv!(c.LtSinv, c.Sf_PD.chol)                # Λfᵀ Σf⁻¹  (d × n)
    mul!(c.negLtSL, c.LtSinv, c.Lf)
    c.negLtSL .*= -one(T)
    c.cF = -T(0.5) * (T(n) * log(T(2π)) + logdet(c.Sf_PD))

    return sm
end

# ============================================================================
# Structure accessors and diagnostics
# ============================================================================

"""
    hamiltonian_matrix(sm[, k]) -> Matrix

The mixed-form Hamiltonian matrix `𝓔_k = [A  −S; Q_k  Aᵀ]` mapping
`[x_t; λ_{t+1}]` to `[x_{t+1}; λ_t]` under cost regime `k` (default 1). Linear in
the model's free parameters, which is why it is the form the M-step estimates.
"""
function hamiltonian_matrix(sm::HamiltonianStateModel{T}, k::Int=1) where {T<:Real}
    n = _plant_dim(sm)
    E = Matrix{T}(undef, 2n, 2n)
    @views begin
        copyto!(E[1:n, 1:n], sm.A)
        E[1:n, (n + 1):(2n)] .= .-sm.S
        copyto!(E[(n + 1):(2n), 1:n], sm.Qc[k])
        copyto!(E[(n + 1):(2n), (n + 1):(2n)], transpose(sm.A))
    end
    return E
end

"""
    symplectic_matrix(sm[, k]) -> Matrix

The forward transition `M_k` on `z = [x; λ]`, which satisfies `Mᵀ J M = J`. This
is the matrix the smoother actually propagates; see
[`symplectic_defect`](@ref) to check it numerically.
"""
symplectic_matrix(sm::HamiltonianStateModel, k::Int=1) = copy(sm.cache.M[k])

"""
    symplectic_form(n) -> Matrix

The `2n × 2n` canonical symplectic form `J = [0 I; −I 0]`.
"""
function symplectic_form(::Type{T}, n::Int) where {T<:Real}
    J = zeros(T, 2n, 2n)
    @views begin
        for i in 1:n
            J[i, n + i] = one(T)
            J[n + i, i] = -one(T)
        end
    end
    return J
end
symplectic_form(n::Int) = symplectic_form(Float64, n)

"""
    symplectic_defect(M) -> Real
    symplectic_defect(sm[, k]) -> Real

`‖Mᵀ J M − J‖∞`, zero for an exactly symplectic transition. A diagnostic: the
parameterization makes `M` symplectic by construction, so a defect much above
the conditioning of `A` means `A` is close to singular.
"""
function symplectic_defect(M::AbstractMatrix{T}) where {T<:Real}
    d = size(M, 1)
    isodd(d) && throw(ArgumentError("a symplectic matrix has even dimension; got $d"))
    J = symplectic_form(T, d ÷ 2)
    return maximum(abs, transpose(M) * J * M .- J)
end

function symplectic_defect(sm::HamiltonianStateModel, k::Int=1)
    return symplectic_defect(sm.cache.M[k])
end

"""
    lqr_parameters(sm) -> NamedTuple

The recovered control problem: `(A = plant, S = B R⁻¹ Bᵀ, Qc = [state costs],
schedule, terminal)`. `B` and `R` are not separately identified — only their
combination `S` is — and the overall cost scale is free (see
[`rescale_costate!`](@ref)).
"""
function lqr_parameters(sm::HamiltonianStateModel)
    return (
        A=copy(sm.A),
        S=copy(sm.S),
        Qc=[copy(Q) for Q in sm.Qc],
        schedule=copy(sm.schedule),
        terminal=sm.terminal,
    )
end

"""
    riccati_solution(sm; k=1, max_iter=1000, tol=1e-12) -> Matrix

The stabilizing solution `P` of the discrete algebraic Riccati equation implied
by regime `k`,

```math
P = Q_k + A^\\top P (I + S P)^{-1} A,
```

by fixed-point iteration from `P = Q_k`. `P` is the steady-state costate map
`λ_t = P x_t` of the infinite-horizon problem, so a fit whose smoothed costate
tracks `P x` is one the LQR interpretation fits well.

Returns the converged `P`; throws `NumericalStabilityError` if the iteration
does not converge, which is the honest answer when regime `k`'s cost admits no
stabilizing solution.
"""
function riccati_solution(
    sm::HamiltonianStateModel{T}; k::Int=1, max_iter::Int=1000, tol::Real=1e-12
) where {T<:Real}
    A = Matrix{T}(sm.A)
    S = Matrix{T}(sm.S)
    P = Matrix{T}(sm.Qc[k])
    n = size(A, 1)
    Imat = Matrix{T}(I, n, n)
    for _ in 1:max_iter
        # P⁺ = Q + Aᵀ P (I + S P)⁻¹ A
        Mmat = Imat + S * P
        Pnext = Matrix{T}(sm.Qc[k]) + transpose(A) * P * (Mmat \ A)
        Symmetrize!(Pnext)
        delta = maximum(abs, Pnext .- P)
        P = Pnext
        if delta <= T(tol) * max(one(T), maximum(abs, P))
            return P
        end
    end
    throw(
        NumericalStabilityError(
            "Qc[$k]",
            "the discrete Riccati iteration did not converge in $max_iter steps; this " *
            "cost regime may admit no stabilizing solution (an indefinite cost, or a " *
            "plant that is not stabilizable through `S`)",
        ),
    )
end

"""
    closed_loop_dynamics(sm; k=1, P=riccati_solution(sm; k=k)) -> Matrix

The steady-state closed-loop plant `(I + S P)⁻¹ A` — how the *state* evolves once
the optimal control `u = −R⁻¹Bᵀλ` is substituted back in. Computable from `S`
and `P` alone, so it does not need `B` and `R` separately.
"""
function closed_loop_dynamics(
    sm::HamiltonianStateModel{T}; k::Int=1, P::AbstractMatrix{T}=riccati_solution(sm; k=k)
) where {T<:Real}
    n = _plant_dim(sm)
    return (Matrix{T}(I, n, n) + Matrix{T}(sm.S) * P) \ Matrix{T}(sm.A)
end

"""
    rescale_costate!(sm, c) -> sm
    rescale_costate!(sm; target=:trace) -> sm

Apply the inverse-optimal-control scale transformation
`(λ, S, Q, h, Σ, …) → (cλ, c⁻¹S, cQ, …)`, which leaves the state dynamics — and
hence the fit — unchanged, and put the model in a canonical scale.

With `target = :trace` the scale is chosen so that `tr(Qc[1]) == n`; with
`target = :opnorm`, so that the largest absolute eigenvalue of `Qc[1]` is 1.
Pass a number instead to set `c` yourself.

Two fits of the same data are only comparable in their cost matrices after this
(or some other) normalization. **The emission's costate columns are not
rescaled** — rescale `C[:, n+1:2n] ./= c` yourself if `observe_costate` is set.
"""
function rescale_costate!(sm::HamiltonianStateModel{T}, c::Real) where {T<:Real}
    #=
    Any nonzero `c` is a symmetry, negative included: `−S'λ' = −(S/c)(cλ) = −Sλ`
    holds for either sign, so the costate's *sign* is unidentified along with its
    scale. That is why `:trace` normalization is a genuine canonicalization — it
    pins the sign as well, by making `tr(Qc[1])` positive.
    =#
    iszero(c) && throw(ArgumentError("the costate scale must be nonzero; got $c"))
    cT = T(c)
    n = _plant_dim(sm)
    d = 2n
    sm.S ./= cT
    for Q in sm.Qc
        Q .*= cT
    end
    sm.Σf .*= cT^2
    sm.hf .*= cT
    @views begin
        # λ-rows of the 2n-vectors / matrices scale by c; x-rows are untouched.
        sm.h[(n + 1):d] .*= cT
        sm.x0[(n + 1):d] .*= cT
        sm.P0[(n + 1):d, :] .*= cT
        sm.P0[:, (n + 1):d] .*= cT
        sm.Σ[(n + 1):d, :] .*= cT
        sm.Σ[:, (n + 1):d] .*= cT
        size(sm.Bu, 2) > 0 && (sm.Bu[(n + 1):d, :] .*= cT)
    end
    return refresh!(sm)
end

function rescale_costate!(
    sm::HamiltonianStateModel{T}; target::Symbol=:trace
) where {T<:Real}
    n = _plant_dim(sm)
    Q1 = sm.Qc[1]
    scale = if target === :trace
        tr(Q1) / T(n)
    elseif target === :opnorm
        maximum(abs, eigvals(Symmetric(Matrix{T}(Q1))))
    else
        throw(ArgumentError("rescale_costate! target must be :trace or :opnorm; got :$target"))
    end
    abs(scale) > eps(T) || throw(
        ArgumentError(
            "cannot normalize by `Qc[1]`: its " *
            (target === :trace ? "trace" : "spectral radius") *
            " is ~0, so the cost carries no scale to normalize against",
        ),
    )
    return rescale_costate!(sm, one(T) / scale)
end

# ============================================================================
# Forward simulation
# ============================================================================

"""
    _ham_input_matrix(sm, ux, tsteps) -> Matrix

The exogenous input sequence as a concrete matrix: `ux` itself, or a zero-row
matrix when the model takes no input. Normalizing here rather than branching at
every use keeps the rollout concretely typed — a `Union{Nothing,AbstractMatrix}`
guarded by a runtime flag is not something inference can narrow.
"""
function _ham_input_matrix(
    sm::HamiltonianStateModel{T}, ux::Union{Nothing,AbstractMatrix{T}}, tsteps::Int
) where {T<:Real}
    ux === nothing && return zeros(T, 0, tsteps)
    size(ux, 2) >= tsteps || throw(
        DimensionMismatchError("ux columns (must cover the horizon)", tsteps, size(ux, 2)),
    )
    size(ux, 1) == size(sm.Bu, 2) ||
        throw(DimensionMismatchError("ux rows", size(sm.Bu, 2), size(ux, 1)))
    return Matrix{T}(ux)
end

"""
    lqr_riccati_sequence(sm, tsteps; ux=nothing) -> (P, g, W)

The finite-horizon Riccati sweep for the control problem this model encodes, run
backward over `tsteps` steps.

Returns `P[t]` and `g[t]` such that the optimal costate is `λ_t = P_t x_t + g_t`,
together with `W[t] = (I + S P_t)⁻¹`, the factor the closed-loop forward map
uses. The recursion is

```math
P_T = Q_{k_T},\\qquad
P_t = Q_{k_t} + A^\\top P_{t+1} W_{t+1} A,
```
```math
g_T = h_f,\\qquad
g_t = A^\\top P_{t+1} W_{t+1} (c_t - S g_{t+1}) + A^\\top g_{t+1} + f_t,
```

where `c_t` and `f_t` are the state and costate halves of the affine term
`h + B_u u_t` — so a nonzero `h` is a *tracking* problem and `g` is its
feedforward.

The terminal condition is `P_T = Q_{k_T}`, `g_T = h_f` when the model carries a
terminal factor and `P_T = 0`, `g_T = 0` (a free endpoint) when it does not.
"""
function lqr_riccati_sequence(
    sm::HamiltonianStateModel{T}, tsteps::Int; ux::Union{Nothing,AbstractMatrix{T}}=nothing
) where {T<:Real}
    tsteps >= 2 || throw(ArgumentError("lqr_riccati_sequence needs tsteps ≥ 2"))
    _hamiltonian_lengths_ok(sm, [tsteps])
    n = _plant_dim(sm)
    d = 2n
    A = Matrix{T}(sm.A)
    S = Matrix{T}(sm.S)
    Imat = Matrix{T}(I, n, n)

    P = [zeros(T, n, n) for _ in 1:tsteps]
    g = [zeros(T, n) for _ in 1:tsteps]
    W = [Matrix{T}(I, n, n) for _ in 1:tsteps]

    if sm.terminal
        copyto!(P[tsteps], sm.Qc[_regime(sm, tsteps)])
        copyto!(g[tsteps], sm.hf)
    end
    W[tsteps] = (Imat + S * P[tsteps]) \ Imat

    #=
    Affine term of transition t = `h + Bu u_t`, split into its state and costate
    halves. Written into a preallocated vector rather than rebuilt, so the loop
    below allocates nothing per step.
    =#
    vbuf = Vector{T}(undef, d)
    ux_mat = _ham_input_matrix(sm, ux, tsteps)
    has_input = size(ux_mat, 1) > 0
    function affine!(t)
        copyto!(vbuf, sm.h)
        has_input && mul!(vbuf, sm.Bu, view(ux_mat, :, t), one(T), one(T))
        return (view(vbuf, 1:n), view(vbuf, (n + 1):d))
    end

    for t in (tsteps - 1):-1:1
        c_t, f_t = affine!(t)
        AtPW = transpose(A) * P[t + 1] * W[t + 1]
        P[t] .= Matrix{T}(sm.Qc[_regime(sm, t)]) .+ AtPW * A
        Symmetrize!(P[t])
        g[t] .= AtPW * (c_t .- S * g[t + 1]) .+ transpose(A) * g[t + 1] .+ f_t
        W[t] = (Imat + S * P[t]) \ Imat
    end
    return P, g, W
end

"""
    simulate_lqr([rng,] sm, tsteps; x1, process_noise, costate_slack, ux)

Roll out the optimal trajectory of the control problem `sm` encodes, as a
`2n × tsteps` array whose rows `1:n` are the state and `n+1:2n` the costate.

**This, not `rand`, is how to generate ground truth for inverse LQR.** A
Hamiltonian matrix has reciprocal eigenvalue pairs `(μ, 1/μ)`, so its forward
flow is unstable by construction: half the modes grow. The optimal trajectory
lives on the stable manifold — the boundary condition is exactly what selects it
— and this function follows that manifold directly, via the backward Riccati
sweep and the closed-loop forward map

```math
x_{t+1} = W_{t+1}(A x_t + c_t - S g_{t+1} + \\varepsilon_t),
\\qquad \\lambda_t = P_t x_t + g_t .
```

Rolling the forward transition `z_{t+1} = M z_t + w` instead — which is what
`rand` does, that being the model's own generative form — diverges over any
useful horizon.

# Keywords
- `x1`: initial state (`n`). Default: drawn from the state block of the model's
  initial prior.
- `process_noise = true`: inject `ε_t ~ N(0, Σ[1:n, 1:n])`, the *state* half of
  the mixed-coordinate innovation. That is genuine plant noise: the optimal
  policy is unchanged by it (certainty equivalence), so the costate still tracks
  `P x + g` exactly.
- `costate_slack = 0`: standard deviation of extra noise added to `λ`, i.e. how
  far from exactly optimal the behavior is. Zero means a perfectly optimal
  agent, whose trajectories sit on a measure-zero set of the model — fitting
  those drives `Σ`'s costate block to singularity, so give a small positive
  value when generating data to fit.
- `ux`: exogenous input sequence (`ux_dim × tsteps`), when the model has one.

Note what this implies about the model: the exactly-optimal trajectory has a
*rank-`n`* innovation, supported on the graph of the Riccati map. A full-rank
`Σ` — which the smoother requires — contains that only as a limit. That is the
precise sense in which the costate must carry process noise.
"""
function simulate_lqr(
    rng::AbstractRNG,
    sm::HamiltonianStateModel{T},
    tsteps::Integer;
    x1::Union{Nothing,AbstractVector{T}}=nothing,
    process_noise::Bool=true,
    costate_slack::Real=0,
    ux::Union{Nothing,AbstractMatrix{T}}=nothing,
) where {T<:Real}
    Ti = Int(tsteps)
    n = _plant_dim(sm)
    d = 2n
    refresh!(sm)
    P, g, W = lqr_riccati_sequence(sm, Ti; ux=ux)
    A = Matrix{T}(sm.A)
    S = Matrix{T}(sm.S)

    z = Matrix{T}(undef, d, Ti)
    x = if x1 === nothing
        @views rand(
            rng,
            MvNormal(
                Vector{T}(sm.x0[1:n]), Matrix(Symmetrize!(Matrix{T}(sm.P0[1:n, 1:n])))
            ),
        )
    else
        Vector{T}(x1)
    end

    Σx = Matrix(Symmetrize!(Matrix{T}(view(sm.Σ, 1:n, 1:n))))
    noise = MvNormal(zeros(T, n), Σx)
    slack = T(costate_slack)
    vbuf = Vector{T}(undef, d)
    ux_mat = _ham_input_matrix(sm, ux, Ti)
    has_input = size(ux_mat, 1) > 0

    for t in 1:Ti
        @views z[1:n, t] .= x
        @views z[(n + 1):d, t] .= P[t] * x .+ g[t]
        slack > 0 && (@views z[(n + 1):d, t] .+= slack .* randn(rng, T, n))
        t == Ti && break
        copyto!(vbuf, sm.h)
        has_input && mul!(vbuf, sm.Bu, view(ux_mat, :, t), one(T), one(T))
        @views rhs = A * x .+ vbuf[1:n] .- S * g[t + 1]
        process_noise && (rhs .+= rand(rng, noise))
        x = W[t + 1] * rhs
    end
    return z
end

function simulate_lqr(sm::HamiltonianStateModel, tsteps::Integer; kwargs...)
    return simulate_lqr(Random.default_rng(), sm, tsteps; kwargs...)
end
