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
                        terminal=true, Gref_cols=nothing, Bu_cols=nothing, Bu_rows=nothing)

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
columns. Excluded entries retain their constructed values throughout fitting.
`Bu_rows=1:n` additionally restricts fitting to the plant rows of the mixed
input matrix, so general inputs act as plant disturbances without creating a
free linear term in the state cost. Initialize excluded costate rows to zero.
The default `nothing` retains the general mixed-input model; `Bu_rows` applies
only to LQR mode, not to the unconstrained `free_state_model`.

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
    Bu_rows::Union{Nothing,Vector{Int}}
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
    Bu_rows::Union{Nothing,AbstractVector{<:Integer}}=nothing,
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
        minimum(bcols) >= 1 ||
            throw(ArgumentError("Bu_cols must be 1-based column indices"))
    end
    brows = Bu_rows === nothing ? nothing : sort!(unique(collect(Int, Bu_rows)))
    if brows !== nothing && !isempty(brows)
        minimum(brows) >= 1 || throw(ArgumentError("Bu_rows must be 1-based row indices"))
    end
    return LQRFitFlags(A, S, Qc, h, Bu, Gref, terminal, cols, bcols, brows)
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
           a.Gref_cols == b.Gref_cols &&
           a.Bu_cols == b.Bu_cols &&
           a.Bu_rows == b.Bu_rows
end

function Base.hash(f::LQRFitFlags, h::UInt)
    return hash(
        (f.A, f.S, f.Qc, f.h, f.Bu, f.Gref, f.terminal, f.Gref_cols, f.Bu_cols, f.Bu_rows),
        hash(:LQRFitFlags, h),
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

@inline function _bu_rows(f::LQRFitFlags, d::Int)
    f.Bu || return Int[]
    return f.Bu_rows === nothing ? collect(1:d) : f.Bu_rows
end

function _check_gref_cols(f::LQRFitFlags, m::Int, d::Int)
    if f.Bu_rows !== nothing && !isempty(f.Bu_rows)
        maximum(f.Bu_rows) <= d ||
            throw(ArgumentError("Bu_rows exceeds the $d mixed-coordinate rows"))
    end
    if f.Bu_cols !== nothing && !isempty(f.Bu_cols)
        maximum(f.Bu_cols) <= m || throw(
            ArgumentError(
                "fit_flags.Bu_cols names column $(maximum(f.Bu_cols)) of a $(m)-column input",
            ),
        )
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
    CausalOptions(; slack_drives_state=true, plant_noise=:dense,
                    costate_noise=:dense, terminal_cost=false)

The noise and horizon options of a `:causal` [`LQRStateModel`](@ref) (see
[`causal_state_model`](@ref)); ignored in every other mode.

- `slack_drives_state`: whether the agent *acts* on its perturbed costate. With
  `true` the costate slack `ν` moves the state, `x_{t+1} = Φ_t x_t + ε_t − W_{t+1} S ν_{t+1}`;
  with `false` it is a pure readout of the costate and never reaches the state.
- `plant_noise`, `costate_noise`: `:dense` or `:diagonal` — the structure of the
  plant noise `Σ` and of the costate noise `Ω`.
- `terminal_cost`: whether the Riccati sweep starts from the terminal cost,
  `P_T = Q_{k_T}` (with feedforward `g_T = h_f − Q_{k_T} G_r u`), or from a free
  endpoint, `P_T = 0`, `g_T = 0`. The `:causal` mode has no terminal *factor*; this
  is the cost the controller plans against.
"""
struct CausalOptions
    slack_drives_state::Bool
    plant_noise::Symbol
    costate_noise::Symbol
    terminal_cost::Bool
    function CausalOptions(
        slack_drives_state::Bool, plant_noise::Symbol, costate_noise::Symbol,
        terminal_cost::Bool,
    )
        for (name, s) in (("plant_noise", plant_noise), ("costate_noise", costate_noise))
            s in (:dense, :diagonal) ||
                throw(ArgumentError("$name must be :dense or :diagonal; got :$s"))
        end
        return new(slack_drives_state, plant_noise, costate_noise, terminal_cost)
    end
end

function CausalOptions(;
    slack_drives_state::Bool=true,
    plant_noise::Symbol=:dense,
    costate_noise::Symbol=:dense,
    terminal_cost::Bool=false,
)
    return CausalOptions(slack_drives_state, plant_noise, costate_noise, terminal_cost)
end

"""
    _CausalHorizon{T}

Everything the smoother needs for one *horizon* of a `:causal` model — the trials
that share a cost-schedule offset and a length, and therefore the same backward
Riccati sweep.

With `s = t + 1`, `W_s = (I + S P_s)⁻¹`, `Φ_t = W_s A`, the trial's constant
input `ũ = [1; u]` and the feedforward `g_s = G_s ũ`, transition `t` is

    z_{t+1} = M_t z_t + B_t ũ + L_t [ε_t; ν_{t+1}],   [ε; ν] ~ N(0, blkdiag(Σ, Ω)),
    M_t = [Φ_t  0; P_s Φ_t  0],   B_t = [b_t; P_s b_t + G_s],   b_t = W_s (C̃ − S G_s),

with `C̃ = [h_x  B_{u,x}]`. `Rz[t] = L_t⁻¹` maps a forward residual back to the
independent innovations: `[W_s  W_s S; −P_s  I]` when the slack drives the state,
`[I 0; −P_s I]` when it does not. Both have unit determinant, so the forward
noise's normalizing constant is the same at every step (the cache's `cQ`).

- `P`, `W`, `G`: the sweep, `1:tsteps` (`G` is `n × (1 + ux_dim)`)
- `M`, `B`, `Rz`: per transition, `1:tsteps-1`
- `negQinv`, `QinvM`, `MtQinv`, `negMtQinvM`: the smoother's gradient and
  Hessian templates per transition, with `Q_t⁻¹ = Rzᵀ blkdiag(Σ, Ω)⁻¹ Rz`
"""
struct _CausalHorizon{T<:Real}
    offset::Int
    tsteps::Int
    P::Vector{Matrix{T}}
    W::Vector{Matrix{T}}
    G::Vector{Matrix{T}}
    M::Vector{Matrix{T}}
    B::Vector{Matrix{T}}
    Rz::Vector{Matrix{T}}
    negQinv::Vector{Matrix{T}}
    QinvM::Vector{Matrix{T}}
    MtQinv::Vector{Matrix{T}}
    negMtQinvM::Vector{Matrix{T}}
end

function _CausalHorizon(::Type{T}, n::Int, m::Int, offset::Int, tsteps::Int) where {T}
    d = 2n
    sq(k, r, c) = [zeros(T, r, c) for _ in 1:k]
    return _CausalHorizon{T}(
        offset,
        tsteps,
        sq(tsteps, n, n),
        sq(tsteps, n, n),
        sq(tsteps, n, 1 + m),
        sq(tsteps - 1, d, d),
        sq(tsteps - 1, d, 1 + m),
        sq(tsteps - 1, d, d),
        sq(tsteps - 1, d, d),
        sq(tsteps - 1, d, d),
        sq(tsteps - 1, d, d),
        sq(tsteps - 1, d, d),
    )
end

"""
    _LQREntryCache{T}

The forward transition an entry prior puts in place of the ordinary one: for the
fresh plan `λ' = μ + K (x − r) + e₁`, `e₁ ~ N(0, P)`, and the plant row
`x' = A x − S λ' + h_x + B_{u,x} u + e₂`, `e₂ ~ N(0, Σ_xx)`, the pair
`z' = [x'; λ']` is

    z' = M z + b + B u + w,    w ~ N(0, Q),

with `M = [A − S K  0; K  0]`, `b = [h_x − S μ; μ]`,
`B = [B_{u,x} + S K G; −K G]` (`G` the reference map of the epoch being left,
so `r = G u`), and `Q = [Σ_xx + S P S  −S P; −P S  P]`. The change of variables
from `(e₁, e₂)` is unit-triangular, so this *is* the entry density, not an
approximation of it, and every kernel that reads an ordinary transition reads
this one the same way. The derived blocks mirror [`LQRCache`](@ref)'s.
"""
mutable struct _LQREntryCache{T<:Real}
    M::Matrix{T}
    b::Vector{T}
    B::Matrix{T}
    Q::DensePDMat{T}
    negQinv::Matrix{T}
    QinvM::Matrix{T}
    MtQinv::Matrix{T}
    negMtQinvM::Matrix{T}
    cQ::T
end

"""
    LQRSwitch{T}

A known boundary between two epochs of an [`LQRStateModel`](@ref)'s cost
schedule, at schedule position `pos`: the first position of the new epoch, so
the epoch being left ends at bin `pos` (its last transition is `pos − 1 → pos`)
and the new one's first transition is `pos → pos + 1`. This is the
deterministic counterpart of a switching model's discrete transition, and it
carries the same two boundary factors (see [`set_boundaries!`](@ref) for the
switching version and [`set_schedule_boundaries!`](@ref) for this one):

- `bridge`: a cost regime (`0` for none). The epoch being left then ends with a
  terminal factor `λ_pos = Q_b (x_pos − r_pos) + h_f`, `Q_b = Qc[bridge]`, with
  the trial-end factor's `Σf` and `hf` and the reference `G_b u_pos`. Under
  `condition_terminal` it is conditioned on, like the trial's end.
- `entry`: an [`EntryPrior`](@ref) (or `nothing`). The transition `pos → pos+1`
  then starts a fresh plan, `λ_{pos+1} ~ N(μ + K (x_pos − r_pos), P)` with `r`
  the reference of the epoch being left, in place of the costate carried through
  the adjoint equation; the plant row keeps the model's own `A`, `S`, plant drift
  and plant noise.

A trial applies a boundary only where it holds the transition `pos → pos + 1`:
one that starts after `pos` or ends at or before it never crosses it.
"""
mutable struct LQRSwitch{T<:Real}
    pos::Int
    bridge::Int
    entry::Union{Nothing,EntryPrior{T}}
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
    innovations to forward ones. In `:hold` mode it is `Lh⁻¹`, the map from the
    plant-row / manifold-row innovations; in `:free` mode the identity.
- `AinvT::Matrix{T}`, `logabsdetA::T`: `A⁻ᵀ` and `log|det G⁻¹|`, the Jacobian
    term the M-step objective carries — `log|det A|` in `:lqr` mode,
    `log det(I + S P)` in `:hold` mode (where `AinvT` is unused and holds `I`),
    and zero in `:free` mode.
- `Qfwd::DensePDMat{T}`: `G Σ Gᵀ`, the forward process-noise covariance.
- `bfwd`, `Bfwd`: the forward bias `G h` and the forward input matrices
    `G (B_u - [0; Q_k G_r])`, one per regime. Unlike the noise map, the input
    matrix *is* regime-dependent whenever a reference is in play, because the
    tracking term `-Q_k r_t` carries that regime's own cost.
- `Ftrm`: one `Q_k G_r` terminal-factor input block per cost regime.
- `negQinv`, `QinvM`, `MtQinv`, `negMtQinvM`, `cQ`: Cholesky-derived templates
    for the gradient and Hessian blocks, the per-regime ones indexed by regime.
- `causal_keys`, `causal`, `causal_index`: a `:causal` model's horizons — the
    `(offset, length)` keys it has been asked to smooth (a registry shared by
    reference with its parameter-group variants, so a horizon registered on the
    parent is built for every variant), one [`_CausalHorizon`](@ref) per key, and
    the key → horizon lookup. Empty in every other mode.
- `Sf_PD`, `Lf`, `negLtSL`, `LtSinv`, `cF`: the terminal factor's covariance,
    per-regime design matrices `Λf[k] = [−Q_k  I]`, and their derived
    curvature / gradient templates. Present (as identity placeholders) even when
    the model carries no terminal factor. Indexing these by regime is what lets a
    ragged trial use `schedule[T_trial]` rather than the longest schedule's end.
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
    const Ftrm::Vector{Matrix{T}}
    const negQinv::Matrix{T}
    const QinvM::Vector{Matrix{T}}
    const MtQinv::Vector{Matrix{T}}
    const negMtQinvM::Vector{Matrix{T}}
    cQ::T
    Sf_PD::DensePDMat{T}
    const Lf::Vector{Matrix{T}}
    const negLtSL::Vector{Matrix{T}}
    const LtSinv::Vector{Matrix{T}}
    cF::T
    # One per `sm.switches` entry; only those carrying an entry prior are filled.
    switch::Vector{_LQREntryCache{T}}
    const causal_keys::Vector{NTuple{2,Int}}
    const causal::Vector{_CausalHorizon{T}}
    const causal_index::Dict{NTuple{2,Int},Int}
end

function LQRCache(
    ::Type{T},
    n::Int,
    nregimes::Int,
    ux_dim::Int;
    causal_keys::Vector{NTuple{2,Int}}=NTuple{2,Int}[],
) where {T<:Real}
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
        [zeros(T, n, ux_dim) for _ in 1:nregimes],
        zeros(T, d, d),
        [zeros(T, d, d) for _ in 1:nregimes],
        [zeros(T, d, d) for _ in 1:nregimes],
        [zeros(T, d, d) for _ in 1:nregimes],
        zero(T),
        PDMat(Matrix{T}(I, n, n)),
        [zeros(T, n, d) for _ in 1:nregimes],
        [zeros(T, d, d) for _ in 1:nregimes],
        [zeros(T, d, n) for _ in 1:nregimes],
        zero(T),
        _LQREntryCache{T}[],
        causal_keys,
        _CausalHorizon{T}[],
        Dict{NTuple{2,Int},Int}(),
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
construction. Symplectic structure alone does not guarantee a minimizing
controller: convex LQR also requires `S` and every `Qc` to be positive
semidefinite. The M-step enforces this with square factors and refuses
indefinite starting matrices. Construction still accepts symmetric indefinite
matrices so historical fits can be loaded and diagnosed.

## Noise

The model is stochastic in the **mixed** coordinates,

```math
\\begin{bmatrix} x_{t+1} \\\\ \\lambda_t \\end{bmatrix}
  = \\mathcal{E}_t \\begin{bmatrix} x_t \\\\ \\lambda_{t+1} \\end{bmatrix}
  + h + B_u u_t + \\varepsilon_t, \\qquad \\varepsilon_t \\sim N(0, \\Sigma),
```

so `Σ`'s leading block is plant process noise and its trailing block is the
costate's own innovation. The equivalent forward noise is `G Σ Gᵀ` with
`G = [I  S A⁻ᵀ; 0  −A⁻ᵀ]`, and since `G` is invertible **the costate must carry
process noise**: a singular `Σ` makes the forward process-noise covariance
singular and the smoother's precision undefined. A free `Σ` spans exactly the
same model class as a free forward covariance, so nothing is lost by
parameterizing it here.

**The costate block does not measure suboptimality.** An exactly optimal agent
under plant noise re-plans after every disturbance, so its costate innovation is
not zero but a fixed image of the plant noise,
`ε^λ_t = −Aᵀ P_{t+1} (I + S P_{t+1})⁻¹ ε^x_t` (the lower block of the formula
under *Identifiability*). `Σ_λλ` therefore grows with the plant noise whether or
not the agent is optimal: in the measurements behind
`docs/dev/lqr/biological.md`, adding substantial suboptimality to an optimal
agent changed the adjoint residual by 0.3% while the residual from the Riccati
graph `λ_t = P_t x_t + g_t` went from exactly 0 to about 1. Suboptimality is a
departure from that graph — what `simulate_lqr`'s `costate_slack` generates —
and this model has no parameter that isolates it.

## Time-varying cost

`Qc` holds `K` cost matrices and `schedule` says which one each timestep uses:
`schedule[t]` indexes the transition `t → t+1` for `t < T`, and `schedule[T]`
the terminal factor. A running-plus-terminal cost is
`cost_schedule(T; terminal=true)`; see [`cost_schedule`](@ref).
An empty `schedule` means "regime 1 everywhere", which requires `K == 1`.

## Per-trial schedules

By default every trial reads the schedule from its first entry, so bin `t` of
every trial has the same cost. That is only the right model when trials start at
the same time relative to whatever the schedule is timed against. When they do
not — a window that opens at target onset and closes at the go cue starts
at a different time in each trial — pass `cost_offset` to `fit!`, `elbo`,
`smooth`, `loglikelihood` and `trial_elbos`: one non-negative integer per trial,
the number of schedule bins before that trial's first bin. Trial `i`'s
transition `t` then uses `schedule[t + cost_offset[i]]`, so the schedule is
written once on the event-relative time axis and each trial reads its own slice.

The schedule must cover `maximum(cost_offset .+ tsteps)`. The fitted cost
matrices are shared by every trial; only *which* one a bin uses is per trial.
The terminal factor still follows `schedule[T + offset]` unless `terminal_regime`
pins it, which is what a ragged-end dataset wants.

Offsets cost nothing at run time: the per-regime matrices are cached once per
parameter set and a lookup is one extra integer add. `fit!` takes them with
Poisson and composite emissions (the Laplace smoother treats every trial
separately). A Gaussian-only emission *fits* with one covariance shared across the
trials of a length, which assumes a shared start, so its `fit!` and its grouped
`elbo` throw on non-zero offsets; its `smooth`, `trial_elbos` and ungrouped
`elbo` accept them, bucketing trials by `(length, offset)`.

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

If those regressors have a constant sum — one-hot target indicators are the
standard case — the reference origin needs its own constraint. With one active
running cost and fitted affine drift, for example,
`G_r → G_r + δ1ᵀ` and `h_λ → h_λ + Qδ` leave the model unchanged. Multiple
targets identify their contrasts, not this common translation. When an
absolute reference origin is required, freeze `h` (`LQRFitFlags(; h = false)`),
supply genuinely different active running costs that share one `h`, or impose
the origin yourself — for instance by reporting only the centred columns of
`G_r`, since no fit flag constrains them to sum to zero. A terminal factor does
not fix the origin when its offset `h_f` is also fitted.

With `K = 1` and no terminal factor, `B_u`'s costate rows and `-Q_1 G_r` both map
the input into the costate and are not separately identified; freeze one. Several
cost regimes, or a terminal factor, separate them.


## Terminal condition

When `terminal` is set, `λ_T = Q_{k_T} x_T` enters as a **soft pseudo-observation**
— `0 = λ_T − Q_{k_T} x_T − h_f + ε_f`, `ε_f ~ N(0, Σ_f)`, with `Σ_f → 0` the hard
boundary condition.

*It is what makes the model well-behaved.* A symplectic matrix has reciprocal
eigenvalue pairs, so the forward transition is unstable by construction and the
unconditioned chain diverges like `ρ(M)^T`. Conditioning on the terminal factor
removes exactly the unstable directions — that is what a boundary condition does
to a two-point boundary value problem — so the posterior, and `rand`'s
terminal-conditioned draw, stay bounded.

*It is a conditioning event, and `condition_terminal` decides whether the score
treats it as one.* The pseudo-observation defines the joint
`p(z) p(y | z) p(y^{term} | z_T)`, but trials under this model are drawn
**given** `y^{term} = 0` — which is what `rand` produces. This conditions on the
costate boundary relation at the observed final timestep; it does not constrain
`x_T` to equal the target. The two scores differ by `p(y^{term} = 0 | θ)`, a
function of the parameters.

With `condition_terminal = true` (the default) `elbo` and `loglikelihood` report

```math
\\log p(y \\mid y^{term} = 0, θ)
  = \\log p(y, y^{term} = 0 \\mid θ) - \\log p(y^{term} = 0 \\mid θ),
```

and the M-step optimizes the same quantity. The normalizer is computed exactly,
by backward square-root Gaussian integration over the whole chain, so EM stays
monotone on the conditional objective: `−Q(θ | θ′) + log Z(θ)` majorizes it and
touches it at `θ′`.

The subtraction is not cosmetic. Three things it fixes:

- **Dimension.** A plant/costate pair the emission never reads still contributes
  its own terminal log-density to the joint score, so a sweep over plant
  dimension scored on the joint is partly a sweep of that density. Under
  conditioning the unused pair cancels exactly, in numerator and denominator
  alike.
- **Gauge.** The inverse-optimal-control rescaling
  `(λ, S, Q, h, Σ_f, h_f) → (cλ, c⁻¹S, cQ, …, c²Σ_f, c h_f)` leaves the plant
  posterior alone but moves the joint score by `−n log|c|` per trial, which is
  unbounded above as `c → 0` when no prior pins the scale. It leaves the
  whitened terminal residual untouched, so the conditional score is exactly
  invariant — see [`rescale_costate!`](@ref).
- **Recovery.** `rand` draws `p(z, y | y^{term} = 0)`. Maximizing the joint on
  those draws is a selection effect rather than an estimator, and a
  self-consistency check will not land on the generating parameters. The
  conditional objective is the likelihood of the process that produced them.

Set `condition_terminal = false` to score and fit the joint
`log p(y, y^{term} = 0)` instead. That is what fits made before this option
existed report, and what the exactly-solvable regression tests compare against;
it is also the cheaper objective, since no normalizer is evaluated.

A switching model's normalizer sums over `K^T` discrete paths and has no exact
form. It is estimated variationally instead, and reported separately by
[`terminal_logz`](@ref) so the approximation stays auditable.

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

A second exact symmetry acts on the cost alone, whenever control enters through
fewer channels than the plant has states (`rank S = r < n`). Take any symmetric
`M` with `S M = 0` — a cost on directions the controller cannot push — and move

```math
Q_k \\to Q_k + M - A^\\top M A \\quad\\text{(every cost the transitions follow)},
\\qquad
Q_T \\to Q_T + M \\quad\\text{(a separate terminal cost)},
```

together with the latent change `λ → λ + M x` it induces: `Σ → L Σ Lᵀ` and
`h → L h` with `L = [I 0; −AᵀM I]`, and `x0`, `P0` by `[I 0; M I]`. The gains and
the closed loop do not move, only the costate does, and when the emission does
not read the costate the likelihood — joint or terminal-conditioned — is
unchanged exactly. This **shaping class** has dimension `(n − r)(n − r + 1)/2`,
so a fitted `Qc` is identified only modulo it; with a separate terminal cost the
identified combinations are `S Q_T` and `Q_k − Q_T + Aᵀ Q_T A`. A terminal
factor written against a cost the transitions also follow pins `M = 0` (the two
moves must agree), and a reference input (`Gref`) shrinks the class. A
`Qc_prior` or `Σ_prior` picks one member, which is the prior's choice rather
than the data's. See `docs/dev/lqr/parameterization.md` §2.4.

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

## Infinite-horizon hold mode (`mode = :hold`)

[`hold_state_model`](@ref) builds a *stationary* regulator — "hold the state at
a reference" — instead of a finite-horizon reach. It shares the plant `A`, the
control authority `S`, the mixed bias `h`, the exogenous input `Bu` and the
reference map `Gref` with the finite-horizon form, and carries **one** cost
`Q_h = Qc[1]`. With `P` the stabilizing solution of the DARE
`P = Q_h + Aᵀ P (I + S P)⁻¹ A`, `W = (I + S P)⁻¹` and `A_cl = W A`, the
transition is

```math
\\begin{bmatrix} I & S \\\\ -P & I \\end{bmatrix} z_{t+1}
  = \\begin{bmatrix} A & 0 & F \\\\ 0 & 0 & G \\end{bmatrix}
    \\begin{bmatrix} x_t \\\\ \\lambda_t \\\\ \\tilde u_t \\end{bmatrix}
  + \\varepsilon_t, \\qquad \\varepsilon_t \\sim N(0, \\Sigma),
```

with `ũ_t = [1; u_t]`, `F = [h_x  B_{u,x}]` and the stationary feedforward
`G = (I − A_clᵀ)⁻¹ (E + A_clᵀ P F)`, `E = [h_λ  B_{u,λ} − Q_h G_r]`. The first
row is exactly the plant row of the finite-horizon mixed equation, so `Σ`'s
leading block keeps its meaning as plant noise; the second places the costate on
the stable manifold, `λ_{t+1} = P x_{t+1} + g_t + ε^λ_t`, softly. The forward
transition is `M = [A_cl  0; P A_cl  0]`, stable by construction, so a hold model
needs no terminal factor — and may not carry one.

The forward and stochastic coordinates differ by `Lh = [I S; −P I]`, whose
determinant `det(I + S P)` plays the role `det A` plays for the finite-horizon
form: the M-step objective carries `−N log det(I + S P)`. That objective is
optimized jointly with any `:lqr` states in the same context, which is what lets
an [`SLDS`](@ref) tie the plant (`tied = [:A, :S]`) between a control state and a
hold state while each keeps its own cost.

## Regularizing the innovation and the cost

`Σ_prior` is an [`IWPrior`](@ref) on the mixed-coordinate innovation. A single
`Qc_prior::IWPrior` applies the same prior to every cost regime. To regularize
epochs differently, pass a vector aligned with `Qc`:

```julia
Qc_prior = [running_prior, nothing, terminal_prior]
```

Here the middle cost is unregularized, and the last entry applies to the terminal
cost when `schedule[end] == 3`. The vector length must equal `length(Qc)`; its
meaning follows cost-regime indices, not the number or order of schedule runs.

# Fields
- `mode::Symbol`: `:lqr` (the finite-horizon symplectic form documented above),
    `:hold` (the infinite-horizon regulator, see [`hold_state_model`](@ref)) or
    `:free` (an unconstrained transition, see [`free_state_model`](@ref)).
- `A::M`: `n × n` plant dynamics. Invertible in `:lqr` mode.
- `S::M`: `n × n` symmetric `B R⁻¹ Bᵀ` — the control authority weighted by the
    control cost. `B` and `R` are not separately identified; only `S` is.
- `Qc::Vector{M}`: `K` symmetric `n × n` state-cost matrices.
- `schedule::Vector{Int}`: per-timestep cost index, or empty for a single cost.
- `cost_offset::Int`: where this model's trial starts on the schedule. The cost in
    force on transition `t` is `schedule[t + cost_offset]`. Always `0` on a fitted
    model; a per-trial copy carries the trial's own offset, which is how trials
    that start at different times relative to one event share one event-timed
    schedule. See "Per-trial schedules" below.
- `terminal::Bool`: whether the terminal costate factor is active.
- `terminal_regime::Int`: which cost the terminal factor is written against.
    `0` (the default) means "whatever the schedule says at this trial's own last
    timestep", `schedule[T_trial]`. A positive `k` pins it to `Qc[k]` for every
    trial whatever its length. The default is the only behaviour that existed
    before this field, and is what every equal-length dataset wants. It is
    *ragged* data that needs the override: `schedule[end]` marks the last index
    of the **longest** trial, so with a dedicated terminal regime a shorter trial
    ends under a running cost instead, and the terminal cost is fitted from the
    maximal-length trials alone. See [`cost_schedule`](@ref).
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
- `B0::M`: optional `2n × ux0_dim` initial-input map. With columns, trial `i`
    starts under `N(B0 * ux0[:, i], P0)`; otherwise it uses `x0`. The first
    state `fit_bool` slot fits `B0` when present.
- `observe_costate::Bool`: whether the emission may read the costate. `false`
    (the default) pins the costate columns of `C` at zero.
- `fit_flags::LQRFitFlags`: which structural parameters move.
- `mstep_iters::Int`: L-BFGS iterations per M-step.
- `Σ_prior::Union{Nothing,IWPrior{T}} = nothing`: optional inverse-Wishart prior
  on the mixed-coordinate innovation. See "Regularizing the innovation and the
  cost" below.
- `fixed_costate_sigma = nothing`: when set to a positive variance, hold the
  costate block of `Σ` at that variance times identity and its state cross block
  at zero; the state block remains fitted. Cannot be combined with `Σ_prior`.
- `Qc_prior = nothing`: optional inverse-Wishart prior on the cost matrices.
  Pass one `IWPrior` to share it across all regimes, or one entry per `Qc`
  regime (each an `IWPrior` or `nothing`) to specify epoch-specific priors,
  including the terminal regime. See below.
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
    cost_offset::Int
    terminal::Bool
    terminal_regime::Int
    condition_terminal::Bool
    Σ::M
    h::V
    Bu::M
    Gref::M
    Σf::M
    hf::V
    x0::V
    B0::M
    P0::M
    observe_costate::Bool
    fit_flags::LQRFitFlags
    mstep_iters::Int
    P0_prior::Union{Nothing,IWPrior{T}}
    x0_prior::Union{Nothing,MNPrior{T,Matrix{T}}}
    Σ_prior::Union{Nothing,IWPrior{T}}
    fixed_costate_sigma::Union{Nothing,T}
    Qc_prior::Union{Nothing,IWPrior{T},AbstractVector}
    depends_on::Union{Nothing,NamedTuple}
    variants::Union{Nothing,Vector{LQRStateModel{T,M,V}}}
    gref_gate::Matrix{Bool}
    switches::Vector{LQRSwitch{T}}
    causal::CausalOptions
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
    _is_hold(sm) -> Bool

Whether the model is the infinite-horizon regulator of [`hold_state_model`](@ref):
a stationary transition whose costate lies on the DARE's stable manifold,
rather than the finite-horizon symplectic form. See the `mode` field.
"""
@inline _is_hold(sm::LQRStateModel) = sm.mode === :hold

"""
    _is_causal(sm) -> Bool

Whether the model is the finite-horizon *causal* controller of
[`causal_state_model`](@ref): the closed loop of the backward Riccati sweep, with
the costate on its graph, rather than the two-point boundary-value form.
"""
@inline _is_causal(sm::LQRStateModel) = sm.mode === :causal

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
    _require_finite_horizon(sm, what)

Throw an informative `ArgumentError` when a quantity that exists only for the
finite-horizon symplectic form is asked of a `:hold` (or `:free`) model.
"""
function _require_finite_horizon(sm::LQRStateModel, what::AbstractString)
    _require_lqr(sm, what)
    _is_causal(sm) && throw(
        ArgumentError(
            "$what belongs to the symplectic two-point boundary-value form, and this " *
            "model is in `:causal` mode — a closed-loop controller whose forward " *
            "transition `[Φ_t 0; P_{t+1} Φ_t 0]` is not symplectic. See " *
            "`lqr_riccati_sequence` for its per-step controller.",
        ),
    )
    _is_hold(sm) && throw(
        ArgumentError(
            "$what belongs to the finite-horizon symplectic form, and this model is " *
            "in `:hold` mode — a stationary regulator whose forward transition " *
            "`[A_cl 0; P A_cl 0]` is not symplectic. See `closed_loop_dynamics` and " *
            "`riccati_solution` for its steady state.",
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
    return isempty(sched) ? 1 : sched[t + sm.cost_offset]
end

"""
    _regime_at(sm, a) -> Int

Cost index at position `a` *on the schedule* — `_regime(sm, a - sm.cost_offset)`.
For a caller that already works in schedule coordinates, such as one comparing
trials with different offsets.
"""
@inline function _regime_at(sm::LQRStateModel, a::Int)
    sched = sm.schedule
    return isempty(sched) ? 1 : sched[a]
end

"""
    _with_cost_offset(sm, offset) -> LQRStateModel

`sm` as seen by a trial that starts `offset` bins into the schedule, so that
`_regime(sm, t) == schedule[t + offset]` for it. This is the per-trial schedule:
the schedule is one vector on an event-relative time axis, and each trial reads
the slice of it its own bins fall on.

Returns `sm` itself when it already carries `offset`, so a dataset with no
offsets costs nothing. Otherwise it is a shallow copy — every parameter array,
the schedule and the derived cache are shared by reference, only the offset
differs — which is why it is safe on a parallel E-step (the cache is read-only
there) and cheap enough to make once per trial. It is **not** a fit target:
nothing that updates parameters should be handed one.
"""
function _with_cost_offset(sm::LQRStateModel{T,M,V}, offset::Int) where {T,M,V}
    sm.cost_offset == offset && return sm
    fields = ntuple(Val(fieldcount(LQRStateModel{T,M,V}))) do i
        fieldname(LQRStateModel{T,M,V}, i) === :cost_offset ? offset : getfield(sm, i)
    end
    return LQRStateModel{T,M,V}(fields...)
end

"""
    _with_cost_offset(lds, offset) -> LinearDynamicalSystem

The same, for the whole model: `lds` itself when its state model already carries
`offset`, and otherwise a copy that shares everything but the state model's
offset.
"""
function _with_cost_offset(
    lds::LinearDynamicalSystem{T,S,O}, offset::Int
) where {T<:Real,S<:LQRStateModel{T},O<:AbstractObservationModel{T}}
    sm = _with_cost_offset(lds.state_model, offset)
    sm === lds.state_model && return lds
    return LinearDynamicalSystem{T,S,O}(
        sm, lds.obs_model, lds.latent_dim, lds.obs_dim, lds.ux_dim, lds.uy_dim, lds.fit_bool
    )
end

"""
    _terminal_regime(sm, tsteps) -> Int

Cost index the terminal factor of a trial of length `tsteps` is written against.

`schedule[tsteps]` unless `sm.terminal_regime` is set, which pins it. The pinned
form is what a ragged dataset needs: the schedule is one vector indexed by
timestep, so `schedule[tsteps] == nregimes` can hold for exactly one trial
length, and on an event-edged window that can be a handful of trials out of
thousands — leaving the terminal cost fitted from those alone while every other
trial ends under a running cost.

One `Int` compare on a branch taken once per trial, not once per timestep, so
the default path costs nothing measurable.
"""
function _normalize_cost_offset(offset, ::LQRStateModel, ntrials::Int)
    offset === nothing && return Int[]
    values = collect(Int, offset isa Integer ? [offset] : offset)
    length(values) == ntrials ||
        throw(DimensionMismatchError("cost_offset length", ntrials, length(values)))
    all(>=(0), values) || throw(
        ArgumentError(
            "cost_offset counts the schedule bins before each trial's first bin, so " *
            "it cannot be negative; got a minimum of $(minimum(values))",
        ),
    )
    return values
end

@inline function _terminal_regime(sm::LQRStateModel, tsteps::Int)
    k = sm.terminal_regime
    return k > 0 ? k : _regime(sm, tsteps)
end

"""
    _terminal_regime_at(sm, a) -> Int

[`_terminal_regime`](@ref) for a trial whose last bin is at position `a` on the
schedule.
"""
@inline function _terminal_regime_at(sm::LQRStateModel, a::Int)
    k = sm.terminal_regime
    return k > 0 ? k : _regime_at(sm, a)
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

"""
    _check_qc_unaliased(Qc)

Refuse a cost vector two of whose entries are the same array.

The M-step packs one parameter block per cost *index* and writes each back to
`Qc[k]`, so `Qc = [Q, Q]` would have its two copies optimized independently and
then overwritten by whichever is written last — the stored model would not be the
one the optimizer accepted, and EM could decrease. A shared cost is expressed by
pointing several `schedule` entries at one index instead.
"""
function _check_qc_unaliased(Qc::AbstractVector)
    for j in 2:length(Qc), i in 1:(j - 1)
        Qc[i] === Qc[j] && throw(
            ArgumentError(
                "Qc[$i] and Qc[$j] are the same array. Each cost index is fitted as " *
                "its own parameter, so aliased entries would be optimized separately " *
                "and overwritten on writeback. To reuse one cost across epochs, point " *
                "the schedule entries at a single index (e.g. `schedule = [1, 1, 2]` " *
                "rather than `Qc = [Q, Q, Qf]`), or pass `copy(Q)` if the costs " *
                "should be fitted separately.",
            ),
        )
    end
    return nothing
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
    terminal_regime::Int=0;
    bridge_regimes::AbstractVector{Int}=Int[],
    require_invertible::Bool=true,
) where {T<:Real}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatchError("LQR A columns", n, size(A, 2)))
    size(S) == (n, n) || throw(DimensionMismatchError("LQR S rows", n, size(S, 1)))
    isempty(Qc) &&
        throw(ArgumentError("an LQRStateModel needs at least one cost matrix `Qc`"))
    _check_qc_unaliased(Qc)
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
    representation at all and the smoother cannot run. (A `:causal` model only
    ever applies `A` forward, so it skips this.)
    =#
    if require_invertible
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
        isfinite(logdetA) || throw(
            NumericalStabilityError("A", "the plant matrix has a non-finite log|det|")
        )
    end

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
        # A pinned terminal cost is reached by every trial's last step rather
        # than by any schedule entry, so it is used even when nothing points at
        # it -- and warning that it is not would be exactly backwards.
        terminal && push!(used, terminal_regime > 0 ? terminal_regime : schedule[end])
        # So is a bridge's cost, read only at the switch it ends.
        union!(used, bridge_regimes)
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
    _normalize_qc_prior(T, prior, K, n)

Validate and normalize the cost-prior specification stored by
[`LQRStateModel`](@ref). A scalar prior is shared by all `K` cost regimes; a
vector addresses regimes in `Qc` order and may contain `nothing` for unregularized
epochs.
"""
function _normalize_qc_prior(::Type{T}, prior, K::Int, n::Int) where {T<:Real}
    prior === nothing && return nothing
    if prior isa IWPrior{T}
        size(prior.Ψ) == (n, n) ||
            throw(DimensionMismatchError("LQR Qc_prior scale", (n, n), size(prior.Ψ)))
        return prior
    end
    prior isa AbstractVector || throw(
        ArgumentError("Qc_prior must be an IWPrior or a vector of IWPrior/nothing entries"),
    )
    length(prior) == K ||
        throw(DimensionMismatchError("LQR Qc_prior entries", K, length(prior)))
    out = Vector{Union{Nothing,IWPrior{T}}}(undef, K)
    for k in 1:K
        pk = prior[k]
        if pk === nothing
            out[k] = nothing
        elseif pk isa IWPrior{T}
            size(pk.Ψ) == (n, n) ||
                throw(DimensionMismatchError("LQR Qc_prior[$k] scale", (n, n), size(pk.Ψ)))
            out[k] = pk
        else
            throw(ArgumentError("Qc_prior[$k] must be an IWPrior or nothing"))
        end
    end
    return out
end

"""Return the inverse-Wishart prior for cost regime `k`, or `nothing`."""
@inline function _qc_prior(sm::LQRStateModel, k::Int)
    prior = sm.Qc_prior
    return prior isa AbstractVector ? prior[k] : prior
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
- `Σ_prior`: inverse-Wishart prior on the innovation.
- `fixed_costate_sigma`: fixed isotropic costate innovation variance, with zero
  state cross covariance.
- `Qc_prior`: one inverse-Wishart prior shared across every cost matrix, or a
  vector aligned with `Qc` whose entries are inverse-Wishart priors or `nothing`.
  The last scheduled cost can therefore have its own terminal prior. See
  "Regularizing the innovation and the cost" on the type.
- `gref_gate`: a `length(Qc) × ux_dim` `Bool` matrix giving each cost regime its
  own reference columns; see [`set_gref_gate!`](@ref).
- `bridges`, `entries`, `entry_gain`, `entry_cov`: boundary factors at the
  schedule's epoch switches; see [`set_schedule_boundaries!`](@ref). A bridge's
  cost regime counts as used.

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
    terminal_regime::Integer=0,
    condition_terminal::Bool=true,
    Σf::Union{Nothing,AbstractMatrix{T}}=nothing,
    hf::Union{Nothing,AbstractVector{T}}=nothing,
    h::Union{Nothing,AbstractVector{T}}=nothing,
    Bu::Union{Nothing,AbstractMatrix{T}}=nothing,
    Gref::Union{Nothing,AbstractMatrix{T}}=nothing,
    x0::Union{Nothing,AbstractVector{T}}=nothing,
    B0::Union{Nothing,AbstractMatrix{T}}=nothing,
    P0::Union{Nothing,AbstractMatrix{T}}=nothing,
    observe_costate::Bool=false,
    fit_flags::LQRFitFlags=LQRFitFlags(),
    mstep_iters::Int=100,
    P0_prior::Union{Nothing,IWPrior{T}}=nothing,
    x0_prior::Union{Nothing,MNPrior{T,Matrix{T}}}=nothing,
    Σ_prior::Union{Nothing,IWPrior{T}}=nothing,
    fixed_costate_sigma::Union{Nothing,Real}=nothing,
    Qc_prior=nothing,
    gref_gate::Union{Nothing,AbstractMatrix{Bool}}=nothing,
    bridges=Pair{Int,Int}[],
    entries=Int[],
    entry_gain::Bool=true,
    entry_cov::Real=1.0,
) where {T<:Real}
    n = size(A, 1)
    d = 2n
    Qc_vec = Qc isa AbstractMatrix ? [Qc] : collect(Qc)
    sched = collect(Int, schedule)

    term_k = Int(terminal_regime)
    _check_lqr_structure(
        A, S, Qc_vec, sched, terminal, term_k; bridge_regimes=Int[last(p) for p in bridges]
    )
    #=
    Pinning the terminal cost only means anything when there is a terminal
    factor to pin, and it has to name a cost that exists. Both are mistakes
    worth catching at construction rather than as an out-of-bounds index deep
    in a gradient.
    =#
    if term_k != 0
        terminal || throw(
            ArgumentError(
                "terminal_regime = $term_k pins the cost of a terminal factor, " *
                "but `terminal` is false, so there is no terminal factor",
            ),
        )
        1 <= term_k <= length(Qc_vec) || throw(
            ArgumentError(
                "terminal_regime must be 0 (follow the schedule) or a cost index " *
                "in 1:$(length(Qc_vec)); got $term_k",
            ),
        )
    end
    Qc_prior_value = _normalize_qc_prior(T, Qc_prior, length(Qc_vec), n)

    size(Σ) == (d, d) || throw(DimensionMismatchError("LQR Σ rows", d, size(Σ, 1)))
    fcs = _check_fixed_costate_sigma(T, Σ, fixed_costate_sigma, Σ_prior, n)
    h_v, Bu_m, Gref_m, x0_v, P0_m, B0_m = _lqr_affine_defaults(
        T, n, h, Bu, Gref, x0, P0, B0, fit_flags, "LQR"
    )
    Σf_m = Σf === nothing ? Matrix{T}(I, n, n) : Σf
    hf_v = hf === nothing ? zeros(T, n) : hf
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
        0,
        terminal,
        term_k,
        condition_terminal,
        Σ,
        h_v,
        Bu_m,
        Gref_m,
        Σf_m,
        hf_v,
        x0_v,
        B0_m,
        P0_m,
        observe_costate,
        fit_flags,
        mstep_iters,
        P0_prior,
        x0_prior,
        Σ_prior,
        fcs,
        Qc_prior_value,
        nothing,
        nothing,
        _normalize_gref_gate(gref_gate, length(Qc_vec), size(Bu_m, 2)),
        LQRSwitch{T}[],
        CausalOptions(),
        LQRCache(T, n, length(Qc_vec), size(Bu_m, 2)),
    )
    if isempty(bridges) && isempty(entries)
        refresh!(sm)
    else
        set_schedule_boundaries!(
            sm; bridges=bridges, entries=entries, entry_gain=entry_gain, entry_cov=entry_cov
        )
    end
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

In a switching model alongside inverse-LQR states, its `observe_costate` is set
to theirs at every entry point, tied emission or not: the emission reads the
same latent coordinates in every mode. With the default `observe_costate =
false` on the LQR side, the free state's emission therefore does not load on
coordinates `n+1:2n` — it feels them only through its own dynamics, which mix
them into the coordinates it does read. On its own, a free model reads all of
them (`observe_costate = true` by default).

# Arguments
- `M`: the `2n × 2n` transition. Its size sets the latent dimension, so it must
  be even-sized.
- `Σ`: the `2n × 2n` process noise.

# Keywords
`h`, `Bu`, `x0`, `P0`, `fit_flags`, `mstep_iters`, `P0_prior`, `x0_prior` and
`Σ_prior` as for the LQR constructor. `Qc`, `Gref`, `schedule`, `terminal`, `Σf`,
`hf` and `Qc_prior` are not accepted — they have no meaning here.

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
    B0::Union{Nothing,AbstractMatrix{T}}=nothing,
    P0::Union{Nothing,AbstractMatrix{T}}=nothing,
    observe_costate::Bool=true,
    fit_flags::LQRFitFlags=LQRFitFlags(),
    mstep_iters::Int=100,
    P0_prior::Union{Nothing,IWPrior{T}}=nothing,
    x0_prior::Union{Nothing,MNPrior{T,Matrix{T}}}=nothing,
    Σ_prior::Union{Nothing,IWPrior{T}}=nothing,
    Qc_prior=nothing,
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
    fit_flags.Bu_rows === nothing ||
        throw(ArgumentError("Bu_rows is supported only in LQR mode"))
    Qc_prior === nothing ||
        throw(ArgumentError("Qc_prior has no meaning in :free mode — there is no cost"))
    n = d >> 1

    size(Σ) == (d, d) || throw(DimensionMismatchError("free Σ rows", d, size(Σ, 1)))

    h_v = h === nothing ? zeros(T, d) : h
    Bu_m = Bu === nothing ? zeros(T, d, 0) : Bu
    x0_v = x0 === nothing ? zeros(T, d) : x0
    P0_m = P0 === nothing ? Matrix{T}(I, d, d) : P0

    length(h_v) == d || throw(DimensionMismatchError("free h", d, length(h_v)))
    size(Bu_m, 1) == d || throw(DimensionMismatchError("free Bu rows", d, size(Bu_m, 1)))
    length(x0_v) == d || throw(DimensionMismatchError("free x0", d, length(x0_v)))
    B0 === nothing ||
        size(B0, 1) == d ||
        throw(DimensionMismatchError("free B0 rows", d, size(B0, 1)))
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
        0,
        false,
        0,
        false,
        Σ,
        h_v,
        Bu_m,
        similar(M, n, 0),
        Matrix{T}(I, n, n),
        zeros(T, n),
        x0_v,
        B0 === nothing ? zeros(T, d, 0) : Matrix{T}(B0),
        P0_m,
        observe_costate,
        fit_flags,
        mstep_iters,
        P0_prior,
        x0_prior,
        Σ_prior,
        nothing,
        #=
        A `:free` model has no cost, so a cost prior would have nothing to act
        on. It is rejected above rather than silently carried.
        =#
        nothing,
        nothing,
        nothing,
        Matrix{Bool}(undef, 0, 0),
        LQRSwitch{T}[],
        CausalOptions(),
        LQRCache(T, n, 1, size(Bu_m, 2)),
    )
    refresh!(sm)
    return sm
end

#=
Structural checks of a `:hold` model, shared by its constructor and
`validate_LDS`. Unlike the finite-horizon form, `A` need not be invertible —
the hold transition never inverts it — but the DARE must have a stabilizing
solution, which `refresh!` checks.
=#
function _check_hold_structure(
    A::AbstractMatrix{T},
    S::AbstractMatrix{T},
    Qc::AbstractVector{<:AbstractMatrix{T}},
    schedule::AbstractVector{Int},
    terminal::Bool,
) where {T<:Real}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatchError("hold A columns", n, size(A, 2)))
    size(S) == (n, n) || throw(DimensionMismatchError("hold S rows", n, size(S, 1)))
    length(Qc) == 1 || throw(
        ArgumentError(
            "a `:hold` model carries exactly one cost `Q_h` (got $(length(Qc))): it is " *
            "a stationary regulator, so there is no schedule of costs to switch among. " *
            "Use one hold state per cost, or `:lqr` mode for a time-varying cost.",
        ),
    )
    Q = Qc[1]
    size(Q) == (n, n) || throw(DimensionMismatchError("hold Q_h rows", n, size(Q, 1)))
    asym = maximum(abs, Q .- transpose(Q); init=zero(T))
    asym <= 1e-8 * max(one(T), maximum(abs, Q; init=one(T))) ||
        throw(NotSymmetricError("Q_h", Float64(asym)))
    asym_S = maximum(abs, S .- transpose(S); init=zero(T))
    asym_S <= 1e-8 * max(one(T), maximum(abs, S; init=one(T))) ||
        throw(NotSymmetricError("S", Float64(asym_S)))
    isempty(schedule) || throw(
        ArgumentError(
            "a `:hold` model has a single stationary cost, so it takes no `schedule`"
        ),
    )
    terminal && throw(
        ArgumentError(
            "a `:hold` model cannot carry a terminal factor: its costate already lies " *
            "on the stable manifold of the infinite-horizon problem, which is what a " *
            "terminal boundary condition would otherwise select. Use `:lqr` mode for " *
            "a finite-horizon reach.",
        ),
    )
    return nothing
end

"""
    hold_state_model(A, S, Q_h, Σ; kwargs...) -> LQRStateModel

A `LQRStateModel` in `:hold` mode: the **infinite-horizon** regulator that
holds the state at a reference, with its costate (softly) on the stable
manifold of the discrete algebraic Riccati equation.

With `P` the stabilizing solution of `P = Q_h + Aᵀ P (I + S P)⁻¹ A`,
`W = (I + S P)⁻¹` and the closed loop `A_cl = W A` (`ρ(A_cl) < 1`), each
transition is

```math
x_{t+1} = A x_t - S \\lambda_{t+1} + h_x + B_{u,x} u_t + \\varepsilon^x_t,
\\qquad
\\lambda_{t+1} = P x_{t+1} + G \\tilde u_t + \\varepsilon^\\lambda_t,
```

`ε_t ~ N(0, Σ)`, `ũ_t = [1; u_t]`. The first row is the plant row of the
finite-horizon model unchanged, so `Σ[1:n, 1:n]` is plant noise as there; the
second is the stable-manifold condition. `G = (I − A_clᵀ)⁻¹ (E + A_clᵀ P F)` is
the stationary feedforward of a quasi-constant forcing, with
`F = [h_x  B_{u,x}]` the plant drift and disturbance and
`E = [h_λ  B_{u,λ} − Q_h G_r]` the costate forcing, tracking term
`−Q_h r_t = −Q_h G_r u_t` included — so a reference input moves the held state
exactly as it moves the finite-horizon controller's target.

The forward transition `M = [A_cl  0; P A_cl  0]` is stable, so a hold model is
well behaved over any horizon without a terminal factor, and `rand` samples it
directly. The plant, cost and reference have the same meaning as in `:lqr`
mode, which is what lets an [`SLDS`](@ref) tie `A` and `S` across a control
state and a hold state (`tied = [:A, :S]`) while each keeps its own cost.

## Cost

A hold model carries exactly one cost, `Qc = [Q_h]`; its schedule is empty and
`_nregimes` is 1. In a switching model whose `:lqr` states carry more cost
matrices (a running and a terminal cost, say), the structural M-step gives each
`Qc` copy its own number of regimes, so nothing needs padding. Tying `:Qc` ties
regime `k` of every state that has one: a hold state's `Q_h` then shares the
`:lqr` states' `Qc[1]`.

## Gauge

The inverse-optimal-control scaling `λ → cλ` maps `P → cP`, `Q_h → cQ_h`,
`S → S/c`, the costate rows of `h`/`Bu` and of `x0`/`P0` by `c`, and `Σ`'s
costate block by `c` on each side; `S P` and hence `A_cl` and `det(I + SP)` are
unchanged. [`rescale_costate!`](@ref) applies exactly that map, and the
likelihood is invariant whenever the emission does not read the costate.

# Arguments
- `A`: `n × n` plant (need not be invertible).
- `S`: `n × n` symmetric control authority `B R⁻¹ Bᵀ`.
- `Q_h`: the `n × n` symmetric state cost (or a one-element vector of it).
- `Σ`: `2n × 2n` positive-definite innovation covariance, in plant-row /
  manifold-row coordinates.

# Keywords
`h`, `Bu`, `Gref`, `x0`, `B0`, `P0`, `observe_costate`, `fit_flags`,
`mstep_iters`, `P0_prior`, `x0_prior`, `Σ_prior`, `fixed_costate_sigma` and
`Qc_prior` as for [`LQRStateModel`](@ref). `fit_flags.terminal` has nothing to
gate here. `schedule` must be empty and `terminal` false; they are accepted only
so that a mistaken request fails with a clear message.

# Throws
- `ArgumentError` for a terminal factor, a schedule or more than one cost
- `DimensionMismatchError` when a size disagrees with `n` or `2n`
- `NumericalStabilityError` when the DARE has no stabilizing solution (`(A, S)`
  not stabilizable, or `(Q_h, A)` not detectable)
"""
function hold_state_model(
    A::AbstractMatrix{T},
    S::AbstractMatrix{T},
    Q_h::Union{AbstractMatrix{T},AbstractVector{<:AbstractMatrix{T}}},
    Σ::AbstractMatrix{T};
    schedule::AbstractVector{<:Integer}=Int[],
    terminal::Bool=false,
    h::Union{Nothing,AbstractVector{T}}=nothing,
    Bu::Union{Nothing,AbstractMatrix{T}}=nothing,
    Gref::Union{Nothing,AbstractMatrix{T}}=nothing,
    x0::Union{Nothing,AbstractVector{T}}=nothing,
    B0::Union{Nothing,AbstractMatrix{T}}=nothing,
    P0::Union{Nothing,AbstractMatrix{T}}=nothing,
    observe_costate::Bool=false,
    fit_flags::LQRFitFlags=LQRFitFlags(),
    mstep_iters::Int=100,
    P0_prior::Union{Nothing,IWPrior{T}}=nothing,
    x0_prior::Union{Nothing,MNPrior{T,Matrix{T}}}=nothing,
    Σ_prior::Union{Nothing,IWPrior{T}}=nothing,
    fixed_costate_sigma::Union{Nothing,Real}=nothing,
    Qc_prior=nothing,
) where {T<:Real}
    n = size(A, 1)
    d = 2n
    Qc_vec = Q_h isa AbstractMatrix ? [Q_h] : collect(Q_h)
    _check_hold_structure(A, S, Qc_vec, collect(Int, schedule), terminal)
    Qc_prior_value = _normalize_qc_prior(T, Qc_prior, 1, n)

    size(Σ) == (d, d) || throw(DimensionMismatchError("hold Σ rows", d, size(Σ, 1)))
    fcs = _check_fixed_costate_sigma(T, Σ, fixed_costate_sigma, Σ_prior, n)
    h_v, Bu_m, Gref_m, x0_v, P0_m, B0_m = _lqr_affine_defaults(
        T, n, h, Bu, Gref, x0, P0, B0, fit_flags, "hold"
    )
    mstep_iters >= 1 ||
        throw(ArgumentError("mstep_iters must be at least 1; got $mstep_iters"))

    MT = typeof(A)
    VT = typeof(h_v)
    sm = LQRStateModel{T,MT,VT}(
        :hold,
        A,
        similar(A, 0, 0),
        S,
        Qc_vec,
        Int[],
        0,
        false,
        0,
        #= No terminal factor, so nothing to condition on. =#
        false,
        Σ,
        h_v,
        Bu_m,
        Gref_m,
        Matrix{T}(I, n, n),
        zeros(T, n),
        x0_v,
        B0_m,
        P0_m,
        observe_costate,
        fit_flags,
        mstep_iters,
        P0_prior,
        x0_prior,
        Σ_prior,
        fcs,
        Qc_prior_value,
        nothing,
        nothing,
        Matrix{Bool}(undef, 0, 0),
        LQRSwitch{T}[],
        CausalOptions(),
        LQRCache(T, n, 1, size(Bu_m, 2)),
    )
    refresh!(sm)
    return sm
end

#=
`fixed_costate_sigma` pins the costate block of `Σ` at `v·I` with no
cross-covariance. Shared by the `:lqr` and `:hold` constructors.
=#
function _check_fixed_costate_sigma(
    ::Type{T}, Σ::AbstractMatrix{T}, fixed, Σ_prior, n::Int
) where {T<:Real}
    fixed === nothing && return nothing
    d = 2n
    Σ_prior === nothing ||
        throw(ArgumentError("fixed_costate_sigma cannot be combined with Σ_prior"))
    isfinite(fixed) && fixed > 0 ||
        throw(ArgumentError("fixed_costate_sigma must be finite and positive"))
    v = T(fixed)
    isfinite(v) && v > 0 || throw(
        ArgumentError("fixed_costate_sigma is outside the covariance's numeric range")
    )
    isapprox(Σ[1:n, (n + 1):d], zeros(T, n, n); atol=zero(T)) &&
        isapprox(Σ[(n + 1):d, 1:n], zeros(T, n, n); atol=zero(T)) &&
        isapprox(Σ[(n + 1):d, (n + 1):d], Matrix{T}(v * I, n, n)) ||
        throw(ArgumentError("fixed_costate_sigma requires Σ = blockdiag(Σ_state, v*I)"))
    return v
end

#=
Defaults and shape checks for the affine pieces and the initial state, shared by
the `:lqr` and `:hold` constructors (`label` names the mode in messages).
=#
function _lqr_affine_defaults(
    ::Type{T},
    n::Int,
    h,
    Bu,
    Gref,
    x0,
    P0,
    B0,
    fit_flags::LQRFitFlags,
    label::AbstractString,
) where {T<:Real}
    d = 2n
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

    length(h_v) == d || throw(DimensionMismatchError("$label h", d, length(h_v)))
    size(Bu_m, 1) == d || throw(DimensionMismatchError("$label Bu rows", d, size(Bu_m, 1)))
    size(Gref_m, 1) == n ||
        throw(DimensionMismatchError("$label Gref rows", n, size(Gref_m, 1)))
    size(Gref_m, 2) == size(Bu_m, 2) || throw(
        DimensionMismatchError(
            "$label Gref columns (must match the input width)",
            size(Bu_m, 2),
            size(Gref_m, 2),
        ),
    )
    _check_gref_cols(fit_flags, size(Gref_m, 2), d)
    length(x0_v) == d || throw(DimensionMismatchError("$label x0", d, length(x0_v)))
    B0 === nothing ||
        size(B0, 1) == d ||
        throw(DimensionMismatchError("$label B0 rows", d, size(B0, 1)))
    size(P0_m) == (d, d) ||
        throw(DimensionMismatchError("$label P0 rows", d, size(P0_m, 1)))
    B0_m = B0 === nothing ? zeros(T, d, 0) : Matrix{T}(B0)
    return h_v, Bu_m, Gref_m, x0_v, P0_m, B0_m
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

`mode = :lqr` gives a mildly contractive plant with a unit cost; `mode = :hold`
the same plant, control authority and cost as an infinite-horizon regulator
(see [`hold_state_model`](@ref)); `mode = :causal` the same as a finite-horizon
feedback controller with plant and costate noise `0.1 I` (see
[`causal_state_model`](@ref)); `mode = :free` a contractive unconstrained
transition (see [`free_state_model`](@ref)). Every keyword of the corresponding
matrix constructor is accepted and overrides the default it names.

# Throws
- `ArgumentError` when `latent_dim` is odd, non-positive, or `mode` is not one
  of `:lqr`, `:hold`, `:causal` and `:free`
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
    elseif mode === :hold
        return hold_state_model(
            Matrix{T}(T(0.95) * I, n, n),
            Matrix{T}(T(0.05) * I, n, n),
            Matrix{T}(I, n, n),
            Matrix{T}(T(0.1) * I, d, d);
            kwargs...,
        )
    elseif mode === :causal
        return causal_state_model(
            Matrix{T}(T(0.95) * I, n, n),
            Matrix{T}(T(0.05) * I, n, n),
            Matrix{T}(I, n, n),
            Matrix{T}(T(0.1) * I, n, n),
            Matrix{T}(T(0.1) * I, n, n);
            kwargs...,
        )
    else
        throw(ArgumentError("mode must be :lqr, :hold, :causal or :free; got :$mode"))
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
- `NumericalStabilityError` when `A` has become singular (`:lqr` mode), or the
  DARE has no stabilizing solution (`:hold` mode)
- `PosDefException` when `Σ` or `Σf` is not positive definite
"""
function refresh!(sm::LQRStateModel{T}) where {T<:Real}
    n = _plant_dim(sm)
    d = 2n
    c = sm.cache
    if _is_free(sm)
        _refresh_free_head!(sm, c, n, d)
    elseif _is_hold(sm)
        _refresh_hold_head!(sm, c, n, d)
    elseif _is_causal(sm)
        _refresh_causal_head!(sm, c, n, d)
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

#=
`:hold` mode fills the same cache from the stationary regulator. The model is
stochastic in plant-row / manifold-row coordinates, `Lh z_{t+1} = Θh ω_t + ε`
with `Lh = [I S; −P I]` and `Θh = [A 0 F; 0 0 G]`, so the forward transition is
`Lh⁻¹ Θh`, and `Lh⁻¹` has the closed form

    Lh⁻¹ = [ W     −S Wᵀ ]       W = (I + S P)⁻¹,
           [ P W     Wᵀ  ]

(check: `(I + SP)W = I`, and `Wᵀ = (I + PS)⁻¹` since `S` and `P` are
symmetric). It is the hold model's `G`: `M = G [A 0; 0 0] = [A_cl 0; P A_cl 0]`,
`bfwd = G [h_x; g₀]`, `Bfwd = G [B_{u,x}; G_u]` and `Qfwd = G Σ Gᵀ`. The
Jacobian `log|det G⁻¹| = log det(I + S P)` goes where `log|det A|` goes for the
finite-horizon form.
=#
function _refresh_hold_head!(
    sm::LQRStateModel{T}, c::LQRCache{T}, n::Int, d::Int
) where {T<:Real}
    m = size(sm.Bu, 2)
    H = _HoldUnit(T, n, m)
    _hold_steady_state!(H, sm.A, sm.S, sm.Qc[1], sm.h, sm.Bu, sm.Gref) || throw(
        NumericalStabilityError(
            "hold",
            "the discrete algebraic Riccati equation P = Q_h + Aᵀ P (I + S P)⁻¹ A has " *
            "no stabilizing solution at these parameters (`(A, S)` must be " *
            "stabilizable and `(Q_h, A)` detectable), so the infinite-horizon " *
            "regulator is undefined",
        ),
    )
    c.logabsdetA = H.logdetM
    copyto!(c.AinvT, I)                    # unused in hold mode; kept finite for `show`
    xr, lr = 1:n, (n + 1):d

    G = c.G
    @views begin
        copyto!(G[xr, xr], H.W)
        mul!(G[xr, lr], sm.S, transpose(H.W), -one(T), zero(T))
        mul!(G[lr, xr], H.P, H.W)
        copyto!(G[lr, lr], transpose(H.W))
    end

    M = c.M[1]
    fill!(M, zero(T))
    @views begin
        copyto!(M[xr, xr], H.Acl)
        mul!(M[lr, xr], H.P, H.Acl)
    end

    # Plant-row / manifold-row bias and input, mapped forward by G.
    vbuf = Vector{T}(undef, d)
    @views begin
        vbuf[xr] .= sm.h[xr]
        vbuf[lr] .= H.Gm[:, 1]
    end
    mul!(c.bfwd, G, vbuf)
    if m > 0
        Bmix = Matrix{T}(undef, d, m)
        @views begin
            Bmix[xr, :] .= sm.Bu[xr, :]
            Bmix[lr, :] .= H.Gm[:, 2:end]
        end
        mul!(c.Bfwd[1], G, Bmix)
    end

    Qfwd = G * Matrix{T}(sm.Σ) * transpose(G)
    c.Qfwd = PDMat(Symmetrize!(Qfwd))
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
            @views mul!(Bmix[(n + 1):d, :], sm.Qc[k], _gref_for(sm, k), -one(T), one(T))
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
    for k in 1:_nregimes(sm)
        fill!(c.Lf[k], zero(T))
        fill!(c.Ftrm[k], zero(T))
        fill!(c.negLtSL[k], zero(T))
        fill!(c.LtSinv[k], zero(T))
    end
    if sm.terminal
        for k in 1:_nregimes(sm)
            Lf = c.Lf[k]
            Ftrm = c.Ftrm[k]
            LtSinv = c.LtSinv[k]
            negLtSL = c.negLtSL[k]
            @views begin
                Lf[:, 1:n] .= .-sm.Qc[k]
                for i in 1:n
                    Lf[i, n + i] = one(T)
                end
            end
            # Terminal tracking: λ_T = Q_k (x_T - r_T), so the residual
            # carries +Q_k G_r u_T.
            size(Ftrm, 2) > 0 && mul!(Ftrm, sm.Qc[k], _gref_for(sm, k))
            copyto!(LtSinv, transpose(Lf))
            rdiv!(LtSinv, c.Sf_PD.chol)             # Λfᵀ Σf⁻¹  (d × n)
            mul!(negLtSL, LtSinv, Lf)
            negLtSL .*= -one(T)
        end
    end
    c.cF = -T(0.5) * (T(n) * log(T(2π)) + logdet(c.Sf_PD))
    _refresh_switches!(sm)

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

In `:hold` mode this is the stationary forward transition `[A_cl 0; P A_cl 0]`,
which is stable rather than symplectic, and in `:free` mode the stored matrix.
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
    _require_finite_horizon(sm, "the symplectic defect")
    return symplectic_defect(sm.cache.M[k])
end

"""
    lqr_parameters(sm) -> NamedTuple

The recovered control problem: `(A = plant, S = B R⁻¹ Bᵀ, Qc = [state costs],
schedule, terminal)`. `B` and `R` are not separately identified — only their
combination `S` is — and the overall cost scale is free (see
[`rescale_costate!`](@ref)).

A `:hold` model adds its infinite-horizon steady state: `P` (the stabilizing
DARE solution), `A_cl = (I + S P)⁻¹ A` (the closed loop) and `feedforward`, the
`n × (1 + ux_dim)` map `G` with `λ = P x + G [1; u]` on the stable manifold.
"""
function lqr_parameters(sm::LQRStateModel{T}) where {T<:Real}
    _require_lqr(sm, "`lqr_parameters`")
    base = (
        A=copy(sm.A),
        S=copy(sm.S),
        Qc=[copy(Q) for Q in sm.Qc],
        schedule=copy(sm.schedule),
        terminal=_is_causal(sm) ? sm.causal.terminal_cost : sm.terminal,
    )
    _is_hold(sm) || return base
    H = _hold_unit_at(sm)
    return (; base..., P=copy(H.P), A_cl=copy(H.Acl), feedforward=copy(H.Gm))
end

"""
    _hold_unit_at(sm) -> _HoldUnit

The steady state of a `:hold` model at its current parameters; throws
`NumericalStabilityError` when the DARE has no stabilizing solution.
"""
function _hold_unit_at(sm::LQRStateModel{T}) where {T<:Real}
    H = _HoldUnit(T, _plant_dim(sm), size(sm.Bu, 2))
    _hold_steady_state!(H, sm.A, sm.S, sm.Qc[1], sm.h, sm.Bu, sm.Gref) || throw(
        NumericalStabilityError(
            "hold", "the DARE has no stabilizing solution at the current parameters"
        ),
    )
    return H
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
does not converge. Nonconvergence alone does not prove that a stabilizing
solution does not exist; convergence may also be slow or numerically difficult.

For a `:hold` model this is the `P` its transition is built on, computed by
structure-preserving doubling (quadratically convergent, and checked against the
equation); `k` must be 1 and `max_iter`/`tol` are not used.
"""
function riccati_solution(
    sm::LQRStateModel{T}; k::Int=1, max_iter::Int=1000, tol::Real=1e-12
) where {T<:Real}
    _require_lqr(sm, "the Riccati solution")
    if _is_hold(sm)
        k == 1 || throw(ArgumentError("a `:hold` model has one cost; got k = $k"))
        return _hold_unit_at(sm).P
    end
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
and `P` alone, so it does not need `B` and `R` separately. For a `:hold` model
it is the state block of the forward transition, `ρ < 1` by construction.
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
`(λ, S, Q, h, Σ, …) → (cλ, c⁻¹S, cQ, …)`, which preserves the plant dynamics,
and put the model in a canonical scale.

What it preserves depends on the score. With the emission transformed as
described below, and no parameter priors:

- no terminal factor: the marginal likelihood is unchanged;
- a terminal factor, `condition_terminal = true`: the conditional score is
  unchanged, exactly. `Σf → c²Σf` and `hf → c·hf` are part of the
  transformation, so the *whitened* terminal residual — and therefore the
  conditioning event — does not move;
- a terminal factor, `condition_terminal = false`: the joint log-density shifts
  by `-n * log(abs(c))` per trial, because only the `|Σf|^{-1/2}` factor
  survives the whitening. That direction is unbounded above as `c → 0`.

Parameter-prior penalties can change in any of these cases.

A `:hold` model carries no terminal factor, so its marginal likelihood is
unchanged. The same map is exact for it: `S P` is invariant (`P → cP`), so the
closed loop, `det(I + S P)` and the plant row are untouched, the feedforward and
the manifold row scale by `c`, and so does `Σ`'s costate block on each side.

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
    sm.fixed_costate_sigma === nothing ||
        c == 1 ||
        throw(
            ArgumentError(
                "costate rescaling would change fixed_costate_sigma; disable rescaling"
            ),
        )
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
terminal factor and `P_T = 0`, `g_T = 0` (a free endpoint) when it does not. A
`:causal` model starts from its terminal cost when `terminal_cost` is set; its
transitions are exactly this sweep's closed loop.

A `:hold` model is stationary, so its sequence is too: `P_t ≡ P` and
`W_t ≡ (I + S P)⁻¹` from the DARE, and `g_{t+1} = G [1; u_t]` — the feedforward
of the input in force on the transition into `t + 1`, exactly the manifold row
of the model — with `g_1 = G [1; u_1]`.
"""
function lqr_riccati_sequence(
    sm::LQRStateModel{T}, tsteps::Int; ux::Union{Nothing,AbstractMatrix{T}}=nothing
) where {T<:Real}
    _require_lqr(sm, "the Riccati sequence")
    tsteps >= 2 || throw(ArgumentError("lqr_riccati_sequence needs tsteps ≥ 2"))
    _lqr_lengths_ok(sm, [tsteps])
    _is_hold(sm) && return _hold_riccati_sequence(sm, tsteps, ux)
    _has_switches(sm) && throw(
        ArgumentError(
            "lqr_riccati_sequence (and simulate_lqr) do not model schedule boundaries; " *
            "sample a model with bridges or entries with `rand`",
        ),
    )
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

    #= A `:causal` model has no terminal factor, but plans against a terminal
    cost when `causal.terminal_cost` is set — the same start for the sweep. =#
    if (_is_causal(sm) ? sm.causal.terminal_cost : sm.terminal)
        kT = _terminal_regime(sm, tsteps)
        copyto!(P[tsteps], sm.Qc[kT])
        copyto!(g[tsteps], sm.hf)
        # Terminal reference: λ_T = Q_f(x_T − r_T) + h_f, so g_T = h_f − Q_f r_T.
        if has_input
            g[tsteps] .-= sm.Qc[kT] * (_gref_for(sm, kT) * view(ux_mat, :, tsteps))
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
            k = _regime(sm, t)
            mul!(view(vbuf, (n + 1):d), sm.Qc[k], _gref_for(sm, k) * u_t, -one(T), one(T))
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

#=
The stationary counterpart of the backward sweep, for a `:hold` model: one DARE
solution for every step, and the feedforward of each step's own input.
=#
function _hold_riccati_sequence(
    sm::LQRStateModel{T}, tsteps::Int, ux::Union{Nothing,AbstractMatrix{T}}
) where {T<:Real}
    H = _hold_unit_at(sm)
    ux_mat = _lqr_input_matrix(sm, ux, tsteps)
    m = size(ux_mat, 1)
    P = [copy(H.P) for _ in 1:tsteps]
    W = [copy(H.W) for _ in 1:tsteps]
    g = Vector{Vector{T}}(undef, tsteps)
    ũ = Vector{T}(undef, 1 + m)
    ũ[1] = one(T)
    for t in 1:tsteps
        s = max(t - 1, 1)                  # the transition into t uses u_{t-1}
        m > 0 && @views ũ[2:end] .= ux_mat[:, s]
        g[t] = H.Gm * ũ
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
x_{t+1} = W_{t+1}(A x_t + c_t - S g_{t+1}) + \\varepsilon_t,
\\qquad \\lambda_t = P_t x_t + g_t ,
```

with `W_{t+1} = (I + S P_{t+1})^{-1}`. This is ordinary causal stochastic LQR: the
agent chooses `u_t` from what it knows at `t` (certainty equivalence), and the
plant noise `ε_t` arrives afterwards, unfiltered by the controller.

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
- `noise_timing = :causal`: when the plant noise enters. `:causal` adds it after
  the closed-loop step, as above, so `Cov(x_{t+1} | x_t) = Σ_xx` — the process a
  feedback controller under plant noise generates. `:implicit` reproduces the
  rollout used before this option existed, `x_{t+1} = W_{t+1}(A x_t + c_t −
  S g_{t+1} + ε_t)`: the noise is passed through `W_{t+1}`, as if the control
  already reacted to the realized `ε_t` (variance `W Σ_xx Wᵀ`). That is a
  perturbed solution of the stationarity conditions rather than a causal agent;
  keep it only to reproduce old simulations.
- `costate_slack = 0`: standard deviation of the perturbation `ν_t` on the
  agent's costate — how far from exactly optimal the behavior is. The agent
  *acts* on the perturbed costate (`u_t = −R⁻¹Bᵀλ_{t+1}`), so the slack moves
  the state too, which is what makes it visible to an emission that reads only
  the state. Zero means a perfectly optimal agent, whose trajectories sit on a
  measure-zero set of the model: give a positive value when generating data to
  fit back.
- `ux`: exogenous input sequence (`ux_dim × tsteps`), when the model has one.

For a `:hold` model the sweep is the stationary one (see
[`lqr_riccati_sequence`](@ref)), so this follows the infinite-horizon regulator:
`process_noise` is the plant row's noise and `costate_slack` the manifold row's.
A hold model's forward flow is itself stable, so `rand` is equally usable for it.

Note what this implies about the model: the exactly-optimal trajectory has a
*rank-`n`* innovation, supported on the graph of the Riccati map. A full-rank
`Σ` — which the smoother requires — contains that only as a limit. That is the
precise sense in which the costate must carry process noise.

It also means data from this function are **not** draws from the model the
inverse-LQR fit assumes. Under causal plant noise the mixed-coordinate residual
the fit models as `N(0, Σ)` is `[I + S P_{t+1}; −Aᵀ P_{t+1}] ε_t`-shaped: rank `n`,
correlated between state and costate rows, and time-varying through `P_{t+1}`.
Fitting a constant full-rank `Σ` to it is a misspecified (if often useful)
approximation, so recovery from `simulate_lqr` tests robustness to that
misspecification, while recovery from `rand` tests self-consistency. Report
which.
"""
function simulate_lqr(
    rng::AbstractRNG,
    sm::LQRStateModel{T},
    tsteps::Integer;
    x1::Union{Nothing,AbstractVector{T}}=nothing,
    process_noise::Bool=true,
    costate_slack::Real=0,
    ux::Union{Nothing,AbstractMatrix{T}}=nothing,
    noise_timing::Symbol=:causal,
) where {T<:Real}
    _require_lqr(sm, "`simulate_lqr`")
    noise_timing in (:causal, :implicit) || throw(
        ArgumentError("noise_timing must be :causal or :implicit, got :$noise_timing")
    )
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
        # (I + S P_{t+1}) x̄_{t+1} = A x_t + c_t − S(g_{t+1} + ν_{t+1}), the step
        # the agent plans from what it knows at t; plant noise lands on top.
        @views rhs = A * x .+ vbuf[1:n] .- S * (g[t + 1] .+ ν[t + 1])
        if noise_timing === :implicit
            process_noise && (rhs .+= rand(rng, noise))
            x = W[t + 1] * rhs
        else
            x = W[t + 1] * rhs
            process_noise && (x .+= rand(rng, noise))
        end
    end
    return z
end

function simulate_lqr(sm::LQRStateModel, tsteps::Integer; kwargs...)
    return simulate_lqr(Random.default_rng(), sm, tsteps; kwargs...)
end
