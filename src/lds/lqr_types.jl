#=============================================================================
LQR latents — model type, derived cache, and structure
utilities.

    Model:      LQRStateModel, cost_schedule
    Cache:      LQRCache, refresh!
    Structure:  lqr_matrix, symplectic_matrix, symplectic_defect,
                riccati_solution, closed_loop_dynamics, lqr_parameters,
                rescale_costate!

The E-step kernels live in `lqr_latents.jl` and the M-step in
`lqr_mstep.jl`.
=============================================================================#

"""
    LQRFitFlags(; A=true, S=true, Qc=true, h=true, Bu=true, Gref=true,
                        terminal=true, Gref_cols=nothing, Bu_cols=nothing)

Which structural parameters of a [`LQRStateModel`](@ref) the M-step is
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

## Estimating the reference from part of the input

`Gref_cols` narrows `Gref` to a subset of the input columns: those columns are
estimated and every other column of `Gref` keeps its constructed value, normally
zero. `nothing` (the default) leaves the whole matrix free, and `Gref = false`
freezes all of it — `Gref_cols` is the middle setting between them.

It is what "the reference depends on *this* variable" means when the variable is
one block of a wider design. With inputs `[reward, direction[T.2] … direction[T.8],
reward:direction…]`, naming the seven direction columns fits a free reference
vector per direction while pinning the reference at zero for everything else —
whereas a fully free `Gref` would also read a reference off reward, which is a
different claim about the task. Like a frozen block, the narrowed columns are
never packed, so this shrinks the M-step problem rather than projecting it.

`Bu_cols` applies the same restriction to the general input matrix. In a
packed input `[ux; ur; ref]`, select only `ux` for `Bu_cols` and only `ur` for
`Gref_cols`; initialise the other `Bu` columns to zero and the known-reference
columns of `Gref` to a fixed selection matrix. An empty `Bu_cols` freezes all
columns. Excluded columns retain their constructed values throughout fitting.

Columns are 1-based indices into the model's input width, validated when the
model is built (which is the first point that width is known).
"""
struct LQRFitFlags
    A::Bool
    S::Bool
    Qc::Bool
    h::Bool
    Bu::Bool
    Gref::Bool
    terminal::Bool
    Gref_cols::Union{Nothing,Vector{Int}}
    Bu_cols::Union{Nothing,Vector{Int}}
end

function LQRFitFlags(;
    A::Bool=true,
    S::Bool=true,
    Qc::Bool=true,
    h::Bool=true,
    Bu::Bool=true,
    Gref::Bool=true,
    terminal::Bool=true,
    Gref_cols::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    Bu_cols::Union{Nothing,AbstractVector{<:Integer}}=nothing,
)
    cols = Gref_cols === nothing ? nothing : sort!(unique(collect(Int, Gref_cols)))
    if cols !== nothing
        isempty(cols) && throw(
            ArgumentError(
                "Gref_cols is empty, which would leave nothing of the reference map " *
                "to estimate; pass `Gref = false` to freeze it outright, or `nothing` " *
                "to estimate every column",
            ),
        )
        minimum(cols) >= 1 || throw(
            ArgumentError("Gref_cols must be 1-based column indices; got $(minimum(cols))"),
        )
    end
    bcols = Bu_cols === nothing ? nothing : sort!(unique(collect(Int, Bu_cols)))
    if bcols !== nothing && !isempty(bcols)
        minimum(bcols) >= 1 || throw(ArgumentError("Bu_cols must be 1-based column indices"))
    end
    return LQRFitFlags(A, S, Qc, h, Bu, Gref, terminal, cols, bcols)
end

"""
    _LQR_STRUCT_NAMES

The pieces of the structural block, in the order the M-step packs them — the
`_LQR_BLOCK_*` ordinals of `lqr_mstep.jl`, so `_LQR_STRUCT_NAMES[b]` names
block `b`. `:terminal` is the terminal factor's offset `hf`, named after the flag
that gates it.

They are one joint estimate, and `depends_on` still resolves them to the single
group `:structure`. What the names buy is the *degree* of grouping: naming
`:Qc` alone splits the cost across groups while every other piece stays one
shared array, which the M-step delivers in a single solve because it already
carries a per-block count of copies (that is how an SLDS ties `A` and `S` while
`Qc` switches).
"""
const _LQR_STRUCT_NAMES = (:A, :S, :Qc, :h, :Bu, :Gref, :terminal)

#=
`Gref_cols` makes the flags no longer a plain-data struct, and the default `==`
on those falls back to `===` — which would call two separately built but
identical sets of flags different, and an SLDS refuses to run when its states
disagree on them. Compare by value instead.
=#
function Base.:(==)(a::LQRFitFlags, b::LQRFitFlags)
    return a.A == b.A &&
           a.S == b.S &&
           a.Qc == b.Qc &&
           a.h == b.h &&
           a.Bu == b.Bu &&
           a.Gref == b.Gref &&
           a.terminal == b.terminal &&
           a.Gref_cols == b.Gref_cols && a.Bu_cols == b.Bu_cols
end

function Base.hash(f::LQRFitFlags, h::UInt)
    return hash(
        (f.A, f.S, f.Qc, f.h, f.Bu, f.Gref, f.terminal, f.Gref_cols, f.Bu_cols), hash(:LQRFitFlags, h)
    )
end

"""
    _gref_cols(f, m) -> Vector{Int}

The input columns of `Gref` the M-step packs, given an input width of `m`: every
column unless [`LQRFitFlags`](@ref)'s `Gref_cols` narrows them, and none
when `Gref` is frozen or the model has no input at all.
"""
@inline function _gref_cols(f::LQRFitFlags, m::Int)
    (f.Gref && m > 0) || return Int[]
    return f.Gref_cols === nothing ? collect(1:m) : f.Gref_cols
end

"""
    _check_gref_cols(f, m)

Reject `Gref_cols` that names a column the model does not have. Checked at model
construction because that is where the input width is first known — the flags
themselves are built before anyone knows how wide `Gref` will be.
"""
@inline function _bu_cols(f::LQRFitFlags, m::Int)
    (f.Bu && m > 0) || return Int[]
    return f.Bu_cols === nothing ? collect(1:m) : f.Bu_cols
end

function _check_gref_cols(f::LQRFitFlags, m::Int)
    if f.Bu_cols !== nothing && !isempty(f.Bu_cols)
        maximum(f.Bu_cols) <= m || throw(ArgumentError(
            "fit_flags.Bu_cols names column $(maximum(f.Bu_cols)) of a $(m)-column input"
        ))
    end
    cols = f.Gref_cols
    cols === nothing && return nothing
    m > 0 || throw(
        ArgumentError(
            "fit_flags.Gref_cols names reference columns $(cols), but this model has " *
            "no input for the reference to be a function of; give it a `Bu`/`Gref` " *
            "width, or drop Gref_cols",
        ),
    )
    maximum(cols) <= m || throw(
        ArgumentError(
            "fit_flags.Gref_cols names column $(maximum(cols)) of a $(m)-column input"
        ),
    )
    return nothing
end

"""
    LQRCache{T}

Everything the smoother and the ELBO need, derived from a
[`LQRStateModel`](@ref)'s natural parameters by [`refresh!`](@ref).

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
- `bfwd`, `Bfwd`: the forward bias `G h` and the forward input matrices
    `G (B_u - [0; Q_k G_r])`, one per regime. Unlike the noise map, the input
    matrix *is* regime-dependent whenever a reference is in play, because the
    tracking term `-Q_k r_t` carries that regime's own cost.
- `Ftrm`: `Q_{k_T} G_r`, the terminal factor's input block.
- `negQinv`, `QinvM`, `MtQinv`, `negMtQinvM`, `cQ`: Cholesky-derived templates
    for the gradient and Hessian blocks, the per-regime ones indexed by regime.
- `Sf_PD`, `Lf`, `negLtSL`, `LtSinv`, `cF`: the terminal factor's covariance,
    its design matrix `Λf = [−Q_f  I]`, and the derived curvature / gradient
    templates. Present (as identity placeholders) even when the model carries no
    terminal factor.
"""
mutable struct LQRCache{T<:Real}
    const n::Int
    const M::Vector{Matrix{T}}
    const G::Matrix{T}
    const AinvT::Matrix{T}
    logabsdetA::T
    Qfwd::DensePDMat{T}
    const bfwd::Vector{T}
    const Bfwd::Vector{Matrix{T}}
    const Ftrm::Matrix{T}
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

function LQRCache(::Type{T}, n::Int, nregimes::Int, ux_dim::Int) where {T<:Real}
    d = 2n
    return LQRCache{T}(
        n,
        [zeros(T, d, d) for _ in 1:nregimes],
        zeros(T, d, d),
        zeros(T, n, n),
        zero(T),
        PDMat(Matrix{T}(I, d, d)),
        zeros(T, d),
        [zeros(T, d, ux_dim) for _ in 1:nregimes],
        zeros(T, n, ux_dim),
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
    LQRStateModel{T,M,V} <: AbstractGaussianStateModel{T}

Latent-state model for **inverse LQR**: the latent state is the LQR
state–costate pair `z_t = [x_t; λ_t]` and the transition is constrained to the
LQR (symplectic) form implied by a linear-quadratic optimal control
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
`cost_schedule(T; terminal=true)`; see [`cost_schedule`](@ref).
An empty `schedule` means "regime 1 everywhere", which requires `K == 1`.

## Tracking a reference

For the tracking problem — cost `½(x_t - r_t)^\\top Q_t (x_t - r_t)` against a
known reference `r_t`, with an optional known disturbance `d_t` — the same
stationarity conditions give

```math
\\begin{bmatrix} x_{t+1} \\\\ \\lambda_t \\end{bmatrix}
  = \\mathcal{E}_t \\begin{bmatrix} x_t \\\\ \\lambda_{t+1} \\end{bmatrix}
  + \\begin{bmatrix} d_t \\\\ -Q_t r_t \\end{bmatrix},
\\qquad \\lambda_T = Q_{k_T}(x_T - r_T).
```

So the affine term's costate half is **not free**: it is `-Q_t r_t`, tied to the
same cost matrix that sits in the lower-left block of `\\mathcal{E}_t`, and it
varies with the regime because `Q_t` does. A free, regime-shared `B_u` cannot
represent that — it would fit a reduced-form input coupling unconstrained by,
and so uninformative about, the cost.

`Gref` supplies it. Writing `r_t = G_r u_t`, the mixed-coordinate input matrix
for regime `k` is

```math
B_u - \\begin{bmatrix} 0 \\\\ Q_k G_r \\end{bmatrix},
```

and the terminal factor picks up `+ Q_{k_T} G_r u_T` in its residual — a reach is
scored against where the target was, not against the origin.

Pass the reference itself as the input (`ux_dim = n`) and `G_r` is a plain
selection: freeze it at `I` with `LQRFitFlags(; Gref = false)`. Pass task
regressors instead (a target identity, say) and `G_r` is estimated, mapping them
to the reference the agent was actually steering toward.

With `K = 1` and no terminal factor, `B_u`'s costate rows and `-Q_1 G_r` both map
the input into the costate and are not separately identified; freeze one. Several
cost regimes, or a terminal factor, separate them.


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

## Grouping (`depends_on`)

Parameters may be estimated separately per group of trials, as on any other
model. The state side offers four groups, matching the `fit_bool` slots:
`:x0`, `:P0`, `:structure` (the whole joint block — `A`, `S`, every `Qc`, `h`,
`Bu`, `Gref`, `hf`) and `:noise` (`Σ`, `Σf`).

```julia
set_depends_on!(state_model, (structure = condition,))      # cost per condition
set_depends_on!(obs_model, (C = session, d = session,       # stitching: one shared
                            D = session, R = session))      # plant, per-session readout
```

The structural block gets *one* name because its pieces are one joint estimate;
freeze pieces within it with [`LQRFitFlags`](@ref), which composes with
grouping — a frozen parameter keeps its starting value in every group and so is
effectively shared while the rest vary.

Note what does *not* separate: groups sharing a noise version pool into that
version's residual scatter, so a model whose cost varies by condition but whose
noise does not is coupled across conditions and is fitted jointly, not condition
by condition.

## Identifiability

`(λ, S, Q) → (cλ, c⁻¹S, cQ)` leaves the state dynamics unchanged for any nonzero
`c`: this is the classical inverse-optimal-control scale invariance (scaling a
cost does not change the policy), and it fixes neither the scale nor the sign.
The product `S·Q` is identified, `S` and `Q` separately are not. Use
[`rescale_costate!`](@ref) to put a fit in a canonical scale before comparing
runs.

Beyond that invariance there is a sharper caveat worth knowing before reading a
fitted cost as *the* cost. An **exactly optimal** agent has `λ_t = P_t x_t` — the
costate is a deterministic function of the state — so its innovation in the mixed
coordinates is

```math
\varepsilon_t = \begin{bmatrix} I + S P_{t+1} \\ -A^\top P_{t+1}\end{bmatrix} w_t,
```

*rank `n` and time-varying* through the Riccati sweep, while this model's `Σ` is
full rank and constant. Fitting such trajectories is therefore a projection onto
the model rather than estimation within it, and the maximum-likelihood cost need
not be the generating one — measurably so, and not because the optimizer failed:
EM started *at* the generating parameters converges to the same other optimum.
On data the model itself generates, EM recovers the cost to a few percent.

What sharpens identification, in rough order of effect: letting the emission read
the costate (`observe_costate = true`), a terminal condition, behaviour that is
noisily rather than exactly optimal, a cost that changes within the trial, and
more trials. This is the well-known ill-posedness of inverse optimal control,
not an artifact of the parameterization.

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
- `Bu::M`: `2n × ux_dim` mixed-coordinate input matrix — an exogenous drift or
    disturbance, *not* the LQR control, which has been eliminated. Free and
    shared across regimes.
- `Gref::M`: `n × ux_dim` **reference map**, `r_t = G_r u_t`. See "Tracking"
    below. All-zero (the default) means no reference and the model is the
    regulation problem.
- `Σf::M`, `hf::V`: terminal factor covariance (`n × n`) and offset (`n`).
- `x0::V`, `P0::M`: prior on `z₁ = [x₁; λ₁]` (`2n`).
- `observe_costate::Bool`: whether the emission may read the costate. `false`
    (the default) pins the costate columns of `C` at zero.
- `fit_flags::LQRFitFlags`: which structural parameters move.
- `mstep_iters::Int`: L-BFGS iterations per M-step.
- `P0_prior`, `x0_prior`: optional priors on the initial state, as on
    [`GaussianStateModel`](@ref).
- `cache::LQRCache{T}`: derived forward parameters. Rebuilt by
    [`refresh!`](@ref), which the constructors and the M-step call for you —
    call it yourself after mutating a field by hand.

The enclosing `LinearDynamicalSystem`'s `fit_bool` still has its usual four
state slots: `[x0, P0, structure, noise]`, where `structure` gates the
`(A, S, Qc, h, Bu)` update (refined by `fit_flags`) and `noise` gates `Σ`/`Σf`.

See also [`lqr_parameters`](@ref), [`riccati_solution`](@ref),
[`symplectic_matrix`](@ref).
"""
mutable struct LQRStateModel{T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
               AbstractGaussianStateModel{T}
    mode::Symbol
    A::M
    Mfree::M
    S::M
    Qc::Vector{M}
    schedule::Vector{Int}
    terminal::Bool
    Σ::M
    h::V
    Bu::M
    Gref::M
    Σf::M
    hf::V
    x0::V
    P0::M
    observe_costate::Bool
    fit_flags::LQRFitFlags
    mstep_iters::Int
    P0_prior::Union{Nothing,IWPrior{T}}
    x0_prior::Union{Nothing,MNPrior{T,Matrix{T}}}
    depends_on::Union{Nothing,NamedTuple}
    variants::Union{Nothing,Vector{LQRStateModel{T,M,V}}}
    cache::LQRCache{T}
end

"""
    _plant_dim(sm) -> Int

The plant dimension `n`. The latent state is twice this.
"""
@inline function _plant_dim(sm::LQRStateModel)
    return sm.mode === :free ? size(sm.Mfree, 1) >> 1 : size(sm.A, 1)
end

_state_latent_dim(sm::LQRStateModel) = 2 * _plant_dim(sm)
_state_ux_dim(sm::LQRStateModel) = size(sm.Bu, 2)

"""
    plant_dim(sm::LQRStateModel) -> Int

The plant dimension `n`. The model's `latent_dim` is `2n` — the state and its
costate — so this is the dimension of the control problem itself, and the one
`A`, `S`, `Qc` and `Gref` are sized by.

```jldoctest
julia> sm = LQRStateModel(6);

julia> (plant_dim(sm), StateSpaceDynamics._state_latent_dim(sm))
(3, 6)
```
"""
plant_dim(sm::LQRStateModel) = _plant_dim(sm)

"""
    _is_free(sm) -> Bool

Whether the model's transition is an unconstrained `2n × 2n` matrix rather than
the symplectic form an LQR implies. See the `mode` field.
"""
@inline _is_free(sm::LQRStateModel) = sm.mode === :free

"""
    _require_lqr(sm, what)

Throw an informative `ArgumentError` when an LQR-only quantity is asked of a
`:free` model, which has no plant, no cost and no costate interpretation.
"""
function _require_lqr(sm::LQRStateModel, what::AbstractString)
    _is_free(sm) && throw(
        ArgumentError(
            "$what is an LQR quantity, and this model is in `:free` mode — its " *
            "transition is an unconstrained matrix with no plant, cost or costate " *
            "to read off. Build it with `LQRStateModel(A, S, Qc, Σ)` if you " *
            "want the constrained form.",
        ),
    )
    return nothing
end

"""
    _lqr_struct_varies(sm) -> NTuple{7,Bool}

Which pieces of the structural block get one copy per `depends_on` group, in
`_LQR_BLOCK_*` order. All of them for `(structure = labels,)`, exactly the named ones
for `(Qc = labels,)`, and none at all when nothing groups the structural block.

Every piece not named here is a single array shared by every variant, so it is
estimated jointly from all the trials — shared *and* fitted, which freezing it
would not give.
"""
function _lqr_struct_varies(sm::LQRStateModel)
    dep = sm.depends_on
    dep === nothing && return ntuple(_ -> false, length(_LQR_STRUCT_NAMES))
    names = keys(dep)
    :structure in names && return ntuple(_ -> true, length(_LQR_STRUCT_NAMES))
    return ntuple(b -> _LQR_STRUCT_NAMES[b] in names, length(_LQR_STRUCT_NAMES))
end

"""
    _nregimes(sm) -> Int

How many distinct cost matrices the model carries.
"""
@inline _nregimes(sm::LQRStateModel) = _is_free(sm) ? 1 : length(sm.Qc)

"""
    _regime(sm, t) -> Int

Cost index in force at timestep `t`: `schedule[t]` for a scheduled model, and 1
when the schedule is empty (a single cost everywhere).
"""
@inline function _regime(sm::LQRStateModel, t::Int)
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
function _check_lqr_structure(
    A::AbstractMatrix{T},
    S::AbstractMatrix{T},
    Qc::AbstractVector{<:AbstractMatrix{T}},
    schedule::AbstractVector{Int},
    terminal::Bool,
) where {T<:Real}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatchError("LQR A columns", n, size(A, 2)))
    size(S) == (n, n) || throw(DimensionMismatchError("LQR S rows", n, size(S, 1)))
    isempty(Qc) &&
        throw(ArgumentError("an LQRStateModel needs at least one cost matrix `Qc`"))
    for (k, Q) in enumerate(Qc)
        size(Q) == (n, n) || throw(DimensionMismatchError("LQR Qc[$k] rows", n, size(Q, 1)))
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
            "the LQR plant matrix is singular. The forward symplectic " *
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
                "an LQRStateModel with $K cost matrices needs a `schedule` saying " *
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
    LQRStateModel(A, S, Qc, Σ; kwargs...)

Build an LQR state model from the plant `A` (`n × n`,
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
- `Gref`: reference map (`n × ux_dim`), `r_t = Gref * u_t`. Default all-zero (no
  reference). See "Tracking a reference" on the type.
- `x0`, `P0`: prior on `z₁ = [x₁; λ₁]`. Default `0` and `I`.
- `observe_costate::Bool = false`: let the emission read the costate.
- `fit_flags`, `mstep_iters`, `P0_prior`, `x0_prior`: see the type docstring.

Everything derived (the symplectic transitions, the forward noise) is built
here; you never pass it in.
"""
function LQRStateModel(
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
    Gref::Union{Nothing,AbstractMatrix{T}}=nothing,
    x0::Union{Nothing,AbstractVector{T}}=nothing,
    P0::Union{Nothing,AbstractMatrix{T}}=nothing,
    observe_costate::Bool=false,
    fit_flags::LQRFitFlags=LQRFitFlags(),
    mstep_iters::Int=100,
    P0_prior::Union{Nothing,IWPrior{T}}=nothing,
    x0_prior::Union{Nothing,MNPrior{T,Matrix{T}}}=nothing,
) where {T<:Real}
    n = size(A, 1)
    d = 2n
    Qc_vec = Qc isa AbstractMatrix ? [Qc] : collect(Qc)
    sched = collect(Int, schedule)

    _check_lqr_structure(A, S, Qc_vec, sched, terminal)

    size(Σ) == (d, d) || throw(DimensionMismatchError("LQR Σ rows", d, size(Σ, 1)))

    h_v = h === nothing ? zeros(T, d) : h
    Bu_m = Bu === nothing ? zeros(T, d, 0) : Bu
    #=
    `Gref` sizes itself off whatever input width the model has, so a model with
    no reference is exactly the regulation problem and one with a reference need
    only say what the reference is.
    =#
    Gref_m = Gref === nothing ? zeros(T, n, size(Bu_m, 2)) : Gref
    x0_v = x0 === nothing ? zeros(T, d) : x0
    P0_m = P0 === nothing ? Matrix{T}(I, d, d) : P0
    Σf_m = Σf === nothing ? Matrix{T}(I, n, n) : Σf
    hf_v = hf === nothing ? zeros(T, n) : hf

    length(h_v) == d || throw(DimensionMismatchError("LQR h", d, length(h_v)))
    size(Bu_m, 1) == d || throw(DimensionMismatchError("LQR Bu rows", d, size(Bu_m, 1)))
    size(Gref_m, 1) == n ||
        throw(DimensionMismatchError("LQR Gref rows", n, size(Gref_m, 1)))
    size(Gref_m, 2) == size(Bu_m, 2) || throw(
        DimensionMismatchError(
            "LQR Gref columns (must match the input width)",
            size(Bu_m, 2),
            size(Gref_m, 2),
        ),
    )
    _check_gref_cols(fit_flags, size(Gref_m, 2))
    length(x0_v) == d || throw(DimensionMismatchError("LQR x0", d, length(x0_v)))
    size(P0_m) == (d, d) || throw(DimensionMismatchError("LQR P0 rows", d, size(P0_m, 1)))
    size(Σf_m) == (n, n) || throw(DimensionMismatchError("LQR Σf rows", n, size(Σf_m, 1)))
    length(hf_v) == n || throw(DimensionMismatchError("LQR hf", n, length(hf_v)))

    mstep_iters >= 1 ||
        throw(ArgumentError("mstep_iters must be at least 1; got $mstep_iters"))

    MT = typeof(A)
    VT = typeof(h_v)
    sm = LQRStateModel{T,MT,VT}(
        :lqr,
        A,
        similar(A, 0, 0),
        S,
        Qc_vec,
        sched,
        terminal,
        Σ,
        h_v,
        Bu_m,
        Gref_m,
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
        LQRCache(T, n, length(Qc_vec), size(Bu_m, 2)),
    )
    refresh!(sm)
    return sm
end

"""
    free_state_model(M, Σ; kwargs...) -> LQRStateModel

A `LQRStateModel` in `:free` mode: the latent transition is the
unconstrained `2n × 2n` matrix `M` rather than the symplectic form an LQR
implies, and `Σ` is the forward process noise directly.

This exists so that an [`SLDS`](@ref) can mix plain linear dynamics with LQR
dynamics. `SLDS` stores one concrete state-model type for every discrete state,
so a "plain dynamics" state has to *be* an `LQRStateModel` — this is that
state. The latent dimension is `2n` in both modes, which is what lets the two
share one continuous latent path.

A free model has no plant, no cost and no costate: `A`, `S`, `Qc` and `Gref` are
empty, `terminal` is off, and the LQR readouts (`lqr_parameters`,
`riccati_solution`, `rescale_costate!`, …) throw rather than invent an answer.
Its M-step is the ordinary closed-form regression, not the constrained one.

# Arguments
- `M`: the `2n × 2n` transition. Its size sets the latent dimension, so it must
  be even-sized.
- `Σ`: the `2n × 2n` process noise.

# Keywords
`h`, `Bu`, `x0`, `P0`, `fit_flags`, `mstep_iters`, `P0_prior`, `x0_prior` as for
the LQR constructor. `Qc`, `Gref`, `schedule`, `terminal`, `Σf` and `hf` are not
accepted — they have no meaning here.

# Examples
```julia
free = free_state_model(0.9 * Matrix(I, 4, 4), Matrix(0.1I, 4, 4))
lqr  = LQRStateModel(A, S, Qc, Σ)          # same latent_dim = 4
slds = SLDS(; A=P, πₖ=π, LDSs=[LinearDynamicalSystem(free, obs),
                               LinearDynamicalSystem(lqr, obs)])
```

# Throws
- `ArgumentError` when `M` is not square with an even size
- `DimensionMismatchError` when `Σ` or any keyword disagrees with `2n`
"""
function free_state_model(
    M::AbstractMatrix{T},
    Σ::AbstractMatrix{T};
    h::Union{Nothing,AbstractVector{T}}=nothing,
    Bu::Union{Nothing,AbstractMatrix{T}}=nothing,
    x0::Union{Nothing,AbstractVector{T}}=nothing,
    P0::Union{Nothing,AbstractMatrix{T}}=nothing,
    observe_costate::Bool=true,
    fit_flags::LQRFitFlags=LQRFitFlags(),
    mstep_iters::Int=100,
    P0_prior::Union{Nothing,IWPrior{T}}=nothing,
    x0_prior::Union{Nothing,MNPrior{T,Matrix{T}}}=nothing,
) where {T<:Real}
    d = size(M, 1)
    size(M, 2) == d ||
        throw(DimensionMismatchError("free transition columns", d, size(M, 2)))
    isodd(d) && throw(
        ArgumentError(
            "a free transition must be even-sized: the latent is the state-costate " *
            "pair `[x; λ]`, so `latent_dim = 2n`. Got $(d)×$(d).",
        ),
    )
    n = d >> 1

    size(Σ) == (d, d) || throw(DimensionMismatchError("free Σ rows", d, size(Σ, 1)))

    h_v = h === nothing ? zeros(T, d) : h
    Bu_m = Bu === nothing ? zeros(T, d, 0) : Bu
    x0_v = x0 === nothing ? zeros(T, d) : x0
    P0_m = P0 === nothing ? Matrix{T}(I, d, d) : P0

    length(h_v) == d || throw(DimensionMismatchError("free h", d, length(h_v)))
    size(Bu_m, 1) == d || throw(DimensionMismatchError("free Bu rows", d, size(Bu_m, 1)))
    length(x0_v) == d || throw(DimensionMismatchError("free x0", d, length(x0_v)))
    size(P0_m) == (d, d) || throw(DimensionMismatchError("free P0 rows", d, size(P0_m, 1)))

    mstep_iters >= 1 ||
        throw(ArgumentError("mstep_iters must be at least 1; got $mstep_iters"))

    MT = typeof(M)
    VT = typeof(h_v)
    empty_m = similar(M, 0, 0)
    sm = LQRStateModel{T,MT,VT}(
        :free,
        empty_m,
        M,
        empty_m,
        MT[],
        Int[],
        false,
        Σ,
        h_v,
        Bu_m,
        similar(M, n, 0),
        Matrix{T}(I, n, n),
        zeros(T, n),
        x0_v,
        P0_m,
        observe_costate,
        fit_flags,
        mstep_iters,
        P0_prior,
        x0_prior,
        nothing,
        nothing,
        LQRCache(T, n, 1, size(Bu_m, 2)),
    )
    refresh!(sm)
    return sm
end

"""
    LQRStateModel(latent_dim; mode=:lqr, kwargs...) -> LQRStateModel

Build a default model of a stated **total** latent dimension, rather than from
its parameter matrices.

This is the spelling an [`SLDS`](@ref) wants: every discrete state shares one
continuous latent path, so the natural thing to say is "all `K` states are
`latent_dim`-dimensional" and let each work out its own `n`. `latent_dim` is the
state *and* costate together, so it must be even — `latent_dim = 4` is two
states and two costates.

`mode = :lqr` gives a mildly contractive plant with a unit cost; `mode = :free`
gives a contractive unconstrained transition (see [`free_state_model`](@ref)).
Every keyword of the corresponding matrix constructor is accepted and overrides
the default it names.

# Throws
- `ArgumentError` when `latent_dim` is odd, non-positive, or `mode` is neither
  `:lqr` nor `:free`
"""
function LQRStateModel(
    latent_dim::Integer; T::Type{<:Real}=Float64, mode::Symbol=:lqr, kwargs...
)
    latent_dim > 0 || throw(ArgumentError("latent_dim must be positive; got $latent_dim"))
    isodd(latent_dim) && throw(
        ArgumentError(
            "latent_dim must be even: the latent is the state-costate pair `[x; λ]`, " *
            "so a plant of dimension `n` gives `latent_dim = 2n`. Got $latent_dim — " *
            "did you mean $(latent_dim + 1)?",
        ),
    )
    n = Int(latent_dim) >> 1
    d = 2n
    if mode === :free
        return free_state_model(
            Matrix{T}(T(0.9) * I, d, d), Matrix{T}(T(0.1) * I, d, d); kwargs...
        )
    elseif mode === :lqr
        return LQRStateModel(
            Matrix{T}(T(0.95) * I, n, n),
            Matrix{T}(T(0.05) * I, n, n),
            Matrix{T}(I, n, n),
            Matrix{T}(T(0.1) * I, d, d);
            kwargs...,
        )
    else
        throw(ArgumentError("mode must be :lqr or :free; got :$mode"))
    end
end

"""
    refresh!(sm::LQRStateModel) -> sm

Rebuild the derived cache from the natural parameters: the per-regime symplectic
transitions `M_k`, the noise map `G`, the forward noise / bias / input, and the
Cholesky-derived gradient and Hessian templates.

Called for you by the constructors and at the end of every M-step. Call it
yourself after assigning to `A`, `S`, `Qc`, `Σ`, `h`, `Bu`, `Σf` or `hf` by
hand — the smoother reads the cache, not the fields.

Any parameter-group variants this model has built are refreshed too. They alias
the parent's arrays for every group that does not vary, so a write to the parent
*is* a write to theirs — but each variant carries its own derived cache, and a
grouped model smooths through those. Without this a hand-written parameter would
reach the fields and never the transitions actually used, which fails silently
rather than loudly. Variants hold no variants of their own, so the recursion is
one level deep.

# Throws
- `NumericalStabilityError` when `A` has become singular
- `PosDefException` when `Σ` or `Σf` is not positive definite
"""
function refresh!(sm::LQRStateModel{T}) where {T<:Real}
    n = _plant_dim(sm)
    d = 2n
    c = sm.cache
    if _is_free(sm)
        _refresh_free_head!(sm, c, n, d)
    else
        _refresh_lqr_head!(sm, c, n, d)
    end
    _refresh_tail!(sm, c, n, d)
    return sm
end

#=
`:free` mode is the same cache filled a different way: the transition *is* the
stored matrix, so the change of coordinates is the identity and the forward
noise, bias and input are the mixed ones unchanged. Nothing downstream of
`refresh!` branches on the mode — the smoother kernels read `M`, `Qfwd`,
`negMtQinvM` and neither knows nor cares which branch wrote them.
=#
function _refresh_free_head!(
    sm::LQRStateModel{T}, c::LQRCache{T}, n::Int, d::Int
) where {T<:Real}
    c.logabsdetA = zero(T)                # G = I carries no Jacobian
    copyto!(c.AinvT, Matrix{T}(I, n, n))  # unused; kept finite for `show`

    copyto!(c.M[1], sm.Mfree)
    copyto!(c.G, Matrix{T}(I, d, d))

    Q = Matrix{T}(sm.Σ)
    c.Qfwd = PDMat(Symmetrize!(Q))
    copyto!(c.bfwd, sm.h)
    size(sm.Bu, 2) > 0 && copyto!(c.Bfwd[1], sm.Bu)
    return nothing
end

function _refresh_lqr_head!(
    sm::LQRStateModel{T}, c::LQRCache{T}, n::Int, d::Int
) where {T<:Real}
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
    #=
    The forward input matrix is per-regime: the mixed-coordinate input block is
    `B_u - [0; Q_k G_r]`, whose costate half carries the tracking term `-Q_k r_t`
    and so varies with the regime's own cost. `G` itself does not — it depends on
    `A` and `S` alone — which is why the noise and bias stay shared.
    =#
    m = size(sm.Bu, 2)
    if m > 0
        Bmix = Matrix{T}(undef, d, m)
        for k in eachindex(sm.Qc)
            copyto!(Bmix, sm.Bu)
            @views mul!(Bmix[(n + 1):d, :], sm.Qc[k], sm.Gref, -one(T), one(T))
            mul!(c.Bfwd[k], G, Bmix)
        end
    end

    return nothing
end

#=
Shared by both modes: everything downstream of `Qfwd`, `M` and the terminal
factor's own parameters.
=#
function _refresh_tail!(
    sm::LQRStateModel{T}, c::LQRCache{T}, n::Int, d::Int
) where {T<:Real}
    Qchol = c.Qfwd.chol
    Imat = Matrix{T}(I, d, d)
    copyto!(c.negQinv, Imat)
    ldiv!(Qchol, c.negQinv)
    c.negQinv .*= -one(T)

    for k in 1:_nregimes(sm)
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
    fill!(c.Ftrm, zero(T))
    if sm.terminal
        kf = isempty(sm.schedule) ? 1 : sm.schedule[end]
        @views begin
            c.Lf[:, 1:n] .= .-sm.Qc[kf]
            for i in 1:n
                c.Lf[i, n + i] = one(T)
            end
        end
        # Terminal tracking: λ_T = Q_f (x_T - r_T), so the residual carries +Q_f G_r u_T.
        size(c.Ftrm, 2) > 0 && mul!(c.Ftrm, sm.Qc[kf], sm.Gref)
    end
    copyto!(c.LtSinv, transpose(c.Lf))
    rdiv!(c.LtSinv, c.Sf_PD.chol)                # Λfᵀ Σf⁻¹  (d × n)
    mul!(c.negLtSL, c.LtSinv, c.Lf)
    c.negLtSL .*= -one(T)
    c.cF = -T(0.5) * (T(n) * log(T(2π)) + logdet(c.Sf_PD))

    #= Last, so a variant is rebuilt only from a parent whose own cache is
    already current — they alias its arrays, so the order is what makes the two
    consistent rather than merely both refreshed. =#
    _refresh_variants!(sm)
    return nothing
end

"""
    _refresh_variants!(sm)

Rebuild the derived cache of every variant this model has built.

Variants alias the parent's arrays for every group that does not vary, so a
write to the parent *is* a write to theirs — but each carries its **own** derived
cache, and a grouped model smooths through those, not through the parent's. So a
hand-written parameter would reach the fields and never the transitions actually
used: silent, and large. On a two-session stitched inverse-LQR fit, rescaling the
costate without this moves the ELBO by tens of thousands of nats across an
operation that is supposed to be an exact symmetry.

A no-op for a model with no variants — which is every variant, since they hold
none of their own, and every ungrouped model — so the recursion terminates after
one level and the ordinary path pays one `=== nothing` test.
"""
function _refresh_variants!(sm::LQRStateModel)
    variants = sm.variants
    variants === nothing && return sm
    for v in variants
        v === sm || refresh!(v)
    end
    return sm
end

# ============================================================================
# Structure accessors and diagnostics
# ============================================================================

"""
    lqr_matrix(sm[, k]) -> Matrix

The mixed-form LQR matrix `𝓔_k = [A  −S; Q_k  Aᵀ]` mapping
`[x_t; λ_{t+1}]` to `[x_{t+1}; λ_t]` under cost regime `k` (default 1). Linear in
the model's free parameters, which is why it is the form the M-step estimates.
"""
function lqr_matrix(sm::LQRStateModel{T}, k::Int=1) where {T<:Real}
    _require_lqr(sm, "the LQR matrix")
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
symplectic_matrix(sm::LQRStateModel, k::Int=1) = copy(sm.cache.M[k])

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

function symplectic_defect(sm::LQRStateModel, k::Int=1)
    _require_lqr(sm, "the symplectic defect")
    return symplectic_defect(sm.cache.M[k])
end

"""
    lqr_parameters(sm) -> NamedTuple

The recovered control problem: `(A = plant, S = B R⁻¹ Bᵀ, Qc = [state costs],
schedule, terminal)`. `B` and `R` are not separately identified — only their
combination `S` is — and the overall cost scale is free (see
[`rescale_costate!`](@ref)).
"""
function lqr_parameters(sm::LQRStateModel)
    _require_lqr(sm, "`lqr_parameters`")
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
    sm::LQRStateModel{T}; k::Int=1, max_iter::Int=1000, tol::Real=1e-12
) where {T<:Real}
    _require_lqr(sm, "the Riccati solution")
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
    sm::LQRStateModel{T}; k::Int=1, P::AbstractMatrix{T}=riccati_solution(sm; k=k)
) where {T<:Real}
    _require_lqr(sm, "the closed-loop dynamics")
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

## A grouped model

One `c` serves the whole model, groups included. It has to: the transformation
rescales the *costate*, and every group's parameters — and the one emission that
reads them — are written against the same latent, so a factor per group would put
the groups on scales that no longer compare with each other, which is the one
thing a canonical scale is for. `target` therefore reads the scale off `Qc[1]` of
the group whose arrays the model itself holds, and the other groups keep their
size relative to it.

Each array is transformed exactly once. That matters because a grouped model
shares the pieces the declaration did not name **by reference**: with
`depends_on = (Qc = labels,)` every group holds the same `S`, and visiting the
groups in turn would divide it by `c` once per group.
"""
function rescale_costate!(sm::LQRStateModel{T}, c::Real) where {T<:Real}
    _require_lqr(sm, "rescaling the costate")
    #=
    Any nonzero `c` is a symmetry, negative included: `−S'λ' = −(S/c)(cλ) = −Sλ`
    holds for either sign, so the costate's *sign* is unidentified along with its
    scale. That is why `:trace` normalization is a genuine canonicalization — it
    pins the sign as well, by making `tr(Qc[1])` positive.
    =#
    iszero(c) && throw(ArgumentError("the costate scale must be nonzero; got $c"))
    cT = T(c)
    #= Every array the parent holds is some variant's slot 1, so walking the
    variants reaches all of them; the parent still needs its own `refresh!`,
    since each variant caches its own derived transition. =#
    seen = Base.IdSet{Any}()
    variants = sm.variants
    for v in (variants === nothing ? (sm,) : variants)
        _rescale_costate_arrays!(v, cT, seen)
        refresh!(v)
    end
    return refresh!(sm)
end

"""
    _rescale_costate_arrays!(sm, c, seen) -> sm

The array-level half of [`rescale_costate!`](@ref): scale each of this model's
parameter arrays by the power of `c` the transformation gives it, skipping any
array already in `seen` and adding the rest to it.

`A` and `Gref` are absent because the transformation leaves them alone — the
plant and the reference are in state coordinates, which do not move.
"""
function _rescale_costate_arrays!(
    sm::LQRStateModel{T}, cT::T, seen::Base.IdSet
) where {T<:Real}
    n = _plant_dim(sm)
    d = 2n
    _rescale_once!(seen, sm.S) && (sm.S ./= cT)
    for Q in sm.Qc
        _rescale_once!(seen, Q) && (Q .*= cT)
    end
    _rescale_once!(seen, sm.Σf) && (sm.Σf .*= cT^2)
    _rescale_once!(seen, sm.hf) && (sm.hf .*= cT)
    @views begin
        # λ-rows of the 2n-vectors / matrices scale by c; x-rows are untouched.
        _rescale_once!(seen, sm.h) && (sm.h[(n + 1):d] .*= cT)
        _rescale_once!(seen, sm.x0) && (sm.x0[(n + 1):d] .*= cT)
        if _rescale_once!(seen, sm.P0)
            sm.P0[(n + 1):d, :] .*= cT
            sm.P0[:, (n + 1):d] .*= cT
        end
        if _rescale_once!(seen, sm.Σ)
            sm.Σ[(n + 1):d, :] .*= cT
            sm.Σ[:, (n + 1):d] .*= cT
        end
        if size(sm.Bu, 2) > 0 && _rescale_once!(seen, sm.Bu)
            sm.Bu[(n + 1):d, :] .*= cT
        end
    end
    return sm
end

"""Whether `array` is new to `seen`, recording it either way."""
function _rescale_once!(seen::Base.IdSet, array)
    array in seen && return false
    push!(seen, array)
    return true
end

function rescale_costate!(sm::LQRStateModel{T}; target::Symbol=:trace) where {T<:Real}
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
    _lqr_input_matrix(sm, ux, tsteps) -> Matrix

The exogenous input sequence as a concrete matrix: `ux` itself, or a zero-row
matrix when the model takes no input. Normalizing here rather than branching at
every use keeps the rollout concretely typed — a `Union{Nothing,AbstractMatrix}`
guarded by a runtime flag is not something inference can narrow.
"""
function _lqr_input_matrix(
    sm::LQRStateModel{T}, ux::Union{Nothing,AbstractMatrix{T}}, tsteps::Int
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
    sm::LQRStateModel{T}, tsteps::Int; ux::Union{Nothing,AbstractMatrix{T}}=nothing
) where {T<:Real}
    _require_lqr(sm, "the Riccati sequence")
    tsteps >= 2 || throw(ArgumentError("lqr_riccati_sequence needs tsteps ≥ 2"))
    _lqr_lengths_ok(sm, [tsteps])
    n = _plant_dim(sm)
    d = 2n
    A = Matrix{T}(sm.A)
    S = Matrix{T}(sm.S)
    Imat = Matrix{T}(I, n, n)

    P = [zeros(T, n, n) for _ in 1:tsteps]
    g = [zeros(T, n) for _ in 1:tsteps]
    W = [Matrix{T}(I, n, n) for _ in 1:tsteps]

    #=
    Normalize the input first: `_lqr_input_matrix` is the single shape check, and
    the terminal block below reads it too.
    =#
    vbuf = Vector{T}(undef, d)
    ux_mat = _lqr_input_matrix(sm, ux, tsteps)
    has_input = size(ux_mat, 1) > 0

    if sm.terminal
        kT = _regime(sm, tsteps)
        copyto!(P[tsteps], sm.Qc[kT])
        copyto!(g[tsteps], sm.hf)
        # Terminal reference: λ_T = Q_f(x_T − r_T) + h_f, so g_T = h_f − Q_f r_T.
        if has_input
            g[tsteps] .-= sm.Qc[kT] * (sm.Gref * view(ux_mat, :, tsteps))
        end
    end
    W[tsteps] = (Imat + S * P[tsteps]) \ Imat
    #=
    Affine term of transition t: `h + B_u u_t − [0; Q_{k(t)} G_r u_t]`. The
    tracking half carries this regime's own cost, which is why it needs `t`.
    =#
    function affine!(t)
        copyto!(vbuf, sm.h)
        if has_input
            u_t = view(ux_mat, :, t)
            mul!(vbuf, sm.Bu, u_t, one(T), one(T))
            mul!(
                view(vbuf, (n + 1):d), sm.Qc[_regime(sm, t)], sm.Gref * u_t, -one(T), one(T)
            )
        end
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
LQR matrix has reciprocal eigenvalue pairs `(μ, 1/μ)`, so its forward
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
- `costate_slack = 0`: standard deviation of the perturbation `ν_t` on the
  agent's costate — how far from exactly optimal the behavior is. The agent
  *acts* on the perturbed costate (`u_t = −R⁻¹Bᵀλ_{t+1}`), so the slack moves
  the state too, which is what makes it visible to an emission that reads only
  the state. Zero means a perfectly optimal agent, whose trajectories sit on a
  measure-zero set of the model: give a positive value when generating data to
  fit back.
- `ux`: exogenous input sequence (`ux_dim × tsteps`), when the model has one.

Note what this implies about the model: the exactly-optimal trajectory has a
*rank-`n`* innovation, supported on the graph of the Riccati map. A full-rank
`Σ` — which the smoother requires — contains that only as a limit. That is the
precise sense in which the costate must carry process noise.
"""
function simulate_lqr(
    rng::AbstractRNG,
    sm::LQRStateModel{T},
    tsteps::Integer;
    x1::Union{Nothing,AbstractVector{T}}=nothing,
    process_noise::Bool=true,
    costate_slack::Real=0,
    ux::Union{Nothing,AbstractMatrix{T}}=nothing,
) where {T<:Real}
    _require_lqr(sm, "`simulate_lqr`")
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
    #=
    Costate slack has to be drawn ahead of the forward pass, because the agent
    *acts* on its own perturbed costate: the control at step t is
    `u_t = −R⁻¹Bᵀλ_{t+1}`, so `ν_{t+1}` moves `x_{t+1}`. Adding the perturbation
    to λ after the fact instead would leave the state trajectory — and hence any
    observation that does not read the costate — completely unaffected by it.
    =#
    ν = [slack > 0 ? slack .* randn(rng, T, n) : zeros(T, n) for _ in 1:Ti]
    vbuf = Vector{T}(undef, d)
    ux_mat = _lqr_input_matrix(sm, ux, Ti)
    has_input = size(ux_mat, 1) > 0

    for t in 1:Ti
        @views z[1:n, t] .= x
        @views z[(n + 1):d, t] .= P[t] * x .+ g[t] .+ ν[t]
        t == Ti && break
        copyto!(vbuf, sm.h)
        has_input && mul!(vbuf, sm.Bu, view(ux_mat, :, t), one(T), one(T))
        # Only the *state* half of the affine term enters the forward map; the
        # costate half is already folded into `g` by the backward sweep.
        # (I + S P_{t+1}) x_{t+1} = A x_t + c_t − S(g_{t+1} + ν_{t+1}) + ε_t
        @views rhs = A * x .+ vbuf[1:n] .- S * (g[t + 1] .+ ν[t + 1])
        process_noise && (rhs .+= rand(rng, noise))
        x = W[t + 1] * rhs
    end
    return z
end

function simulate_lqr(sm::LQRStateModel, tsteps::Integer; kwargs...)
    return simulate_lqr(Random.default_rng(), sm, tsteps; kwargs...)
end
