# # Inverse LQR with Hamiltonian latents
#
# Standard state-space models ask *how does the latent state evolve?* This one
# asks *what was the state evolving toward?* — it fits a latent linear-quadratic
# regulator and returns the plant and the cost function that the observed
# behaviour is optimal for.

using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using LaTeXStrings
using StableRNGs

rng = StableRNG(1234);

ssd_palette = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#e87ba4"] # hide
default(; # hide
    palette=ssd_palette, framestyle=:box, grid=true, gridalpha=0.12, # hide
    linewidth=2, size=(760, 420), titlefontsize=12, guidefontsize=10, # hide
    legendfontsize=9, foreground_color_legend=nothing, # hide
) # hide

# ## From LQR to a state-space model
#
# For the discrete-time control problem
#
# ```math
# \min_u \sum_{t=1}^{T-1} \tfrac12\left(x_t^\top Q_t x_t + u_t^\top R u_t\right)
#        + \tfrac12 x_T^\top Q_T x_T
# \quad\text{subject to}\quad x_{t+1} = A x_t + B u_t,
# ```
#
# stationarity of the Lagrangian in ``(u_t, x_t, \lambda_t)`` eliminates the
# control, ``u_t = -R^{-1}B^\top\lambda_{t+1}``, and leaves a two-point boundary
# value problem in the state and its costate ``\lambda``:
#
# ```math
# \begin{bmatrix} x_{t+1} \\ \lambda_t \end{bmatrix}
#   = \underbrace{\begin{bmatrix} A & -S \\ Q_t & A^\top \end{bmatrix}}_{\mathcal{E}_t}
#     \begin{bmatrix} x_t \\ \lambda_{t+1} \end{bmatrix},
# \qquad S := B R^{-1} B^\top,
# \qquad \lambda_T = Q_T x_T .
# ```
#
# Note what is and is not identifiable: only the combination ``S = BR^{-1}B^\top``
# appears, never ``B`` and ``R`` separately.
#
# The smoother needs a forward chain on ``z_t = [x_t; \lambda_t]``, which comes
# from eliminating ``\lambda_{t+1}``:
#
# ```math
# M_t = \begin{bmatrix} A + SA^{-\top}Q_t & -SA^{-\top} \\
#                       -A^{-\top}Q_t     &  A^{-\top} \end{bmatrix},
# \qquad M_t^\top J M_t = J,
# \qquad J = \begin{bmatrix} 0 & I \\ -I & 0\end{bmatrix}.
# ```
#
# So the latent dimension is **twice** the plant dimension, and the transition is
# constrained to be symplectic. `HamiltonianStateModel` carries the natural
# parameters ``(A, S, Q_{1:K})`` and derives ``M_t`` from them, so every M-step
# returns a model that is still exactly a control problem.

plant_dim = 2
obs_dim = 8
tsteps = 30

A = [0.97 0.05; -0.04 0.95]          # the plant
S = [0.05 0.01; 0.01 0.04]           # B R⁻¹ Bᵀ — control authority
Qc = [0.20 0.03; 0.03 0.15]          # running state cost
Σ = Matrix(Diagonal(fill(0.02, 2 * plant_dim)));

# The noise lives in the **mixed** coordinates ``[x_{t+1}; \lambda_t]``: its
# leading block is genuine plant process noise and its trailing block is *costate
# slack*, how far from exactly optimal the behaviour is. The costate must carry
# some — the forward process-noise covariance is ``G\Sigma G^\top`` with ``G``
# invertible, so a singular ``\Sigma`` leaves the smoother's precision undefined.

state_model = HamiltonianStateModel(A, S, Qc, Σ; P0=Matrix(0.2I, 4, 4))

# The emission reads the state but not the costate — the default, since the
# costate is an inferred intention rather than something recorded.

C = randn(rng, obs_dim, 2 * plant_dim)
C[:, (plant_dim + 1):end] .= 0
obs_model = GaussianObservationModel(C, Matrix(0.05I, obs_dim, obs_dim), zeros(obs_dim))
lds = LinearDynamicalSystem(state_model, obs_model)

# ## Simulating an optimal trajectory
#
# A symplectic matrix has reciprocal eigenvalue pairs ``(\mu, 1/\mu)``, so half
# its modes grow: the *forward* Hamiltonian flow is unstable by construction. The
# optimal trajectory lives on the stable manifold, and the boundary condition is
# what selects it. `simulate_lqr` follows that manifold directly, through the
# backward Riccati sweep and the closed-loop forward map.

println("spectral radius of M: ", maximum(abs, eigvals(symplectic_matrix(state_model))))

ntrials = 80
zs = [simulate_lqr(rng, state_model, tsteps; costate_slack=0.03) for _ in 1:ntrials]
ys = [C * z .+ sqrt(0.05) .* randn(rng, obs_dim, tsteps) for z in zs];

p1 = plot(
    zs[1][1, :];
    label=L"x_1",
    xlabel="time",
    ylabel="value",
    title="One optimal trajectory: state and costate",
)
plot!(p1, zs[1][2, :]; label=L"x_2")
plot!(p1, zs[1][3, :]; label=L"\lambda_1", linestyle=:dash)
plot!(p1, zs[1][4, :]; label=L"\lambda_2", linestyle=:dash)
p1

# The costate is the gradient of the cost-to-go: it leads the state, and decays
# as the trajectory settles.

# ## Recovering the cost
#
# The usual inverse-optimal-control setting is a **known plant and an unknown
# cost** — freeze `A` and `S` and let EM estimate `Qc`. Freezing shrinks the
# M-step problem rather than projecting its solution.

init = HamiltonianStateModel(
    copy(A),
    copy(S),
    Matrix(0.4I, plant_dim, plant_dim),      # deliberately wrong starting cost
    Matrix(0.05I, 4, 4);
    P0=Matrix(0.2I, 4, 4),
    fit_flags=HamiltonianFitFlags(; A=false, S=false),
)
fit_lds = LinearDynamicalSystem(
    init,
    GaussianObservationModel(copy(C), Matrix(0.05I, obs_dim, obs_dim), zeros(obs_dim)),
)
elbos = fit!(fit_lds, ys; max_iter=150, tol=1e-10, progress=false)

plot(
    elbos;
    xlabel="EM iteration",
    ylabel="ELBO",
    label=false,
    title="EM is monotone: the M-step is a generalized one",
)

# The cost is identified only up to a nonzero scalar — scaling a cost does not
# change the policy it induces, the classical inverse-optimal-control invariance
# — so compare in a canonical scale.

truth = HamiltonianStateModel(copy(A), copy(S), copy(Qc), copy(Σ))
rescale_costate!(truth; target=:trace)
rescale_costate!(init; target=:trace)

println("true  Qc = ", round.(truth.Qc[1]; digits=3))
println("fitted Qc = ", round.(init.Qc[1]; digits=3))

# The structure survives fitting, which is the whole point:

println("symplectic defect: ", symplectic_defect(init))
println("Qc symmetric:      ", init.Qc[1] ≈ init.Qc[1]')

# ## What comes out
#
# `lqr_parameters` returns the recovered control problem, and the Riccati
# solution turns it into the steady-state feedback law.

params = lqr_parameters(init)
P = riccati_solution(init)
println("Riccati P = ", round.(P; digits=3))
println("closed loop eigenvalues = ",
        round.(abs.(eigvals(closed_loop_dynamics(init; P=P))); digits=3))

# The closed loop is stable while the open-loop plant is barely so — that gap is
# the control doing its work.

println("open loop eigenvalues   = ", round.(abs.(eigvals(A)); digits=3))

# ## Time-varying cost
#
# The cost may change within a trial: a delay epoch with little cost, a movement
# epoch with more, and a terminal cost at the end. `cost_schedule` builds the
# per-timestep index, `schedule[t]` governing the transition ``t \to t+1`` and
# `schedule[T]` the terminal factor.

schedule = cost_schedule(tsteps; terminal=true, onset=16)
println("regimes in force: ", unique(schedule))

varying = HamiltonianStateModel(
    copy(A),
    copy(S),
    [0.05 * Matrix(I, 2, 2), copy(Qc), 2.0 * Matrix(I, 2, 2)],
    copy(Σ);
    schedule=schedule,
    terminal=true,
    Σf=Matrix(0.02I, plant_dim, plant_dim),
    P0=Matrix(0.2I, 4, 4),
)
z_var = simulate_lqr(rng, varying, tsteps; process_noise=false, x1=[1.0, -0.6])

p2 = plot(
    z_var[1, :]; label=L"x_1", xlabel="time", title="Cost that switches at t = 16"
)
plot!(p2, z_var[2, :]; label=L"x_2")
vline!(p2, [16]; label="cost onset", linestyle=:dot, color=:black)
p2

# With almost no cost before ``t = 16`` the state drifts under the open-loop
# plant; once the running cost switches on it is driven to the origin, and the
# terminal cost pins the endpoint.

# ## Poisson observations
#
# Nothing above is specific to Gaussian emissions — spike counts work the same
# way, and so do composite emissions mixing kinematics with spikes.

C_spk = 0.4 .* randn(rng, 12, 4)
C_spk[:, (plant_dim + 1):end] .= 0
plds = LinearDynamicalSystem(
    HamiltonianStateModel(copy(A), copy(S), copy(Qc), copy(Σ); P0=Matrix(0.2I, 4, 4)),
    PoissonObservationModel(C_spk, fill(0.5, 12)),
)
counts = [Float64.(rand(rng, 0:3, 12, tsteps)) for _ in 1:10]
poisson_elbos = fit!(plds, counts; max_iter=15, progress=false)
println("Poisson ELBO: ", round(poisson_elbos[1]; digits=1), " -> ",
        round(poisson_elbos[end]; digits=1))

using SSDTest  #src
@test length(elbos) >= 2  #src
@test minimum(diff(elbos)) > -1e-8  #src
@test symplectic_defect(init) < 1e-9  #src
@test init.Qc[1] ≈ init.Qc[1]' atol = 1e-12  #src
@test size(z_var) == (4, tsteps)  #src
@test all(isfinite, z_var)  #src
@test minimum(diff(poisson_elbos)) > -1e-6  #src
@test all(iszero, plds.obs_model.C[:, (plant_dim + 1):end])  #src
