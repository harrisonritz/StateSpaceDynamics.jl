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
#
# Fit here to data the *model* generates, so that recovery is a well-posed
# question; the section after this one is about what happens when the data comes
# from an exactly-optimal agent instead.

model_lds = LinearDynamicalSystem(
    HamiltonianStateModel(copy(A), copy(S), copy(Qc), copy(Σ); P0=Matrix(0.2I, 4, 4)),
    GaussianObservationModel(copy(C), Matrix(0.05I, obs_dim, obs_dim), zeros(obs_dim)),
)
_, ys_model = rand(StableRNG(99), model_lds, fill(20, 120))

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
elbos = fit!(fit_lds, ys_model; max_iter=250, tol=1e-10, progress=false)

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

# ## When the agent is *exactly* optimal
#
# The trajectories from `simulate_lqr` above are a harder case, and worth being
# explicit about. An exactly-optimal agent has `λ_t = P_t x_t` — the costate is a
# deterministic function of the state — so its innovation in the mixed
# coordinates is
#
# ```math
# \varepsilon_t = \begin{bmatrix} I + S P_{t+1} \\ -A^\top P_{t+1}\end{bmatrix} w_t,
# ```
#
# which is **rank ``n`` and time-varying** through the Riccati sweep. The model's
# ``\Sigma`` is full rank and constant — it has to be, since the smoother's
# precision is otherwise undefined — so it contains that process only as a limit.
# Fitting near-optimal trajectories is therefore a projection onto the model
# rather than estimation within it, and the maximum-likelihood cost need not be
# the generating one:

fit_on_optimal = HamiltonianStateModel(
    copy(A), copy(S), Matrix(0.4I, plant_dim, plant_dim), Matrix(0.05I, 4, 4);
    P0=Matrix(0.2I, 4, 4), fit_flags=HamiltonianFitFlags(; A=false, S=false),
)
lds_opt = LinearDynamicalSystem(
    fit_on_optimal,
    GaussianObservationModel(copy(C), Matrix(0.05I, obs_dim, obs_dim), zeros(obs_dim)),
)
elbos_opt = fit!(lds_opt, ys; max_iter=150, tol=1e-10, progress=false)
println("ELBO at the generating parameters: ", round(elbo(lds, ys); digits=1))
println("ELBO the fit reaches:              ", round(elbos_opt[end]; digits=1))

# The fit beats the truth, and it still does when EM is started *at* the truth —
# so this is the model's projection, not a failure of the optimizer. It is the
# classical ill-posedness of inverse optimal control showing up concretely. What
# helps, in rough order: letting the emission read the costate
# (`observe_costate=true`), a terminal condition, behaviour that is noisily rather
# than exactly optimal, and more trials.

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

# ## Tracking a reference
#
# For a tracking cost ``\tfrac12(x_t - r_t)^\top Q_t (x_t - r_t)`` the affine term
# is ``[d_t;\, -Q_t r_t]``: the costate half is **not free**, it is tied to the
# same cost matrix in ``\mathcal{E}_t`` and varies with the regime because
# ``Q_t`` does. `Gref` supplies exactly that — pass the reference as the input
# and freeze ``G_r`` at the identity.

track_sm = HamiltonianStateModel(
    copy(A),
    [0.20 0.02; 0.02 0.18],                     # more control authority
    [copy(Qc), 200.0 * Matrix(I, 2, 2)],        # heavy terminal cost
    copy(Σ);
    schedule=cost_schedule(tsteps; terminal=true),
    terminal=true,
    Bu=zeros(4, plant_dim),
    Gref=Matrix(1.0I, plant_dim, plant_dim),    # the reference *is* the input
    Σf=Matrix(1e-6I, plant_dim, plant_dim),
    P0=Matrix(0.2I, 4, 4),
)
target = [1.5, -0.8]
z_track = simulate_lqr(
    track_sm, tsteps; process_noise=false, x1=zeros(plant_dim),
    ux=repeat(target, 1, tsteps),
)

p3 = plot(z_track[1, :]; label=L"x_1", xlabel="time", title="Tracking a fixed target")
plot!(p3, z_track[2, :]; label=L"x_2")
hline!(p3, target; label="target", linestyle=:dot, color=:black)
p3

println("distance to target at the end: ",
        round(norm(z_track[1:plant_dim, end] .- target); digits=4))

# The costate is the gradient of the cost-to-go, so it vanishes as the state
# reaches the target — `λ_T = Q_f(x_T − r_T)` is the terminal condition, and with
# a heavy terminal cost that pins the endpoint.

# ## Parameters that vary by group
#
# `depends_on` estimates parameters separately per group of trials. The state
# side has four groups matching the `fit_bool` slots: `:x0`, `:P0`, `:structure`
# (the whole joint block) and `:noise`. Stitching sessions — one shared plant and
# cost, a per-session readout — is the observation side:
#
# ```julia
# set_depends_on!(state_model, (structure = condition,))    # cost per condition
# set_depends_on!(obs_model, (C = session, d = session,     # per-session readout,
#                             D = session, R = session))    # shared LQR structure
# ```
#
# Groups sharing a noise version pool into that version's residual scatter, so a
# model whose cost varies by condition but whose noise does not is fitted jointly
# rather than condition by condition.

# ## Switching between control problems
#
# `cost_schedule` says *when* the cost changes. An `SLDS` instead **infers** it:
# the discrete state becomes the cost epoch, and the responsibilities say which
# control problem was active at each moment.
#
# Every discrete state shares one continuous latent path — the SLDS forms each
# timestep as the responsibility-weighted mixture ``\ell_t = \sum_k w_{kt}
# \ell_t^{(k)}(x)`` — so they all carry the same ``2n``-dimensional ``z``. The
# switching is over parameters, never over dimension. Because the discrete state
# *is* the epoch, each member carries a single `Qc`.

# The data so far came from a *single* control problem, so start by giving the
# switching model something to find: half the trials under a cheap cost, half
# under an expensive one, sharing the plant.
#
# Draw them from each model's **own prior** with `rand`, not from `simulate_lqr`.
# That matters more than it looks, and the reason is worth stating before the
# result — see the note below.

sw_T = 16
Q_lo = [0.20 0.03; 0.03 0.15]
Q_hi = [1.20 0.00; 0.00 0.90]
gen(Q) = HamiltonianStateModel(copy(A), copy(S), copy(Q), copy(Σ); P0=Matrix(0.2I, 4, 4))
sw_state(Q) = LinearDynamicalSystem(
    gen(Q),
    GaussianObservationModel(copy(C), Matrix(0.05I, obs_dim, obs_dim), zeros(obs_dim)),
)
ys_sw = vcat(
    [rand(StableRNG(50 + i), sw_state(Q_lo), sw_T)[2] for i in 1:30],
    [rand(StableRNG(90 + i), sw_state(Q_hi), sw_T)[2] for i in 1:30],
)

slds = SLDS(;
    A=[0.92 0.08; 0.08 0.92],
    πₖ=[0.5, 0.5],
    LDSs=[sw_state([0.15 0.0; 0.0 0.15]), sw_state([1.5 0.0; 0.0 1.5])],
)
sw_elbos = fit!(slds, ys_sw; max_iter=30, progress=false, rng=StableRNG(7))
println("switching ELBO: ", round(sw_elbos[1]; digits=1), " -> ",
        round(sw_elbos[end]; digits=1))
println("tr(Qc) recovered: ",
        round.(sort([tr(l.state_model.Qc[1]) for l in slds.LDSs]); digits=3),
        "   truth: ", round.([tr(Q_lo), tr(Q_hi)]; digits=3))

# Each state stays a genuine control problem: the M-step optimizes the symplectic
# parameterization, not a free transition.

println("symplectic defects: ",
        [round(symplectic_defect(l.state_model); sigdigits=2) for l in slds.LDSs])

# The responsibilities say which problem was active. Switching here is between
# trials, so they should be near-constant within a trial and differ across halves.

γ = smooth(slds, ys_sw).γ
mean_resp(trial) = sum(γ[trial][1, :]) / size(γ[trial], 2)
lo_resp = sum(mean_resp(n) for n in 1:30) / 30
hi_resp = sum(mean_resp(n) for n in 31:60) / 30
println("mean γ₁: cheap-cost trials ", round(lo_resp; digits=3),
        "   expensive ", round(hi_resp; digits=3))

# !!! note "Read the converged bound, not the `fit!` trace"
#     The trace `fit!` returns is the bound at whatever `q` the inner variational
#     alternation reached that iteration — `smoothing_iters` of it, one by
#     default. For an inverse-LQR state that inner problem is unusually hard,
#     because a symplectic transition's forward flow is unstable and the shared
#     `q(x)` converges slowly, so the trace sits well below `elbo(slds, y)` at the
#     same parameters (~170 nats here, against ~0.5 for a Gaussian `SLDS`) and can
#     dip while the parameters are still improving. Use `elbo(slds, y)`, or raise
#     `smoothing_iters`, when you want to check that a fit is making progress.

# !!! warning "Switching cannot be recovered from exactly-optimal trajectories"
#     `simulate_lqr` follows the *stable manifold* — the Riccati solution — while
#     the model's own density is the forward symplectic chain with process noise,
#     which is divergent. They are different distributions, and this model is a
#     relaxation of exact optimality rather than a description of it.
#
#     For a single fit that mismatch costs some accuracy. For *switching* it is
#     fatal, because the responsibilities compare two such densities and the
#     mismatch is larger than the cost signal. Measured on the models above:
#     classifying trials by `loglik(cheap) − loglik(expensive)` is **100%**
#     correct on data drawn from the models' own priors, and **50%** — chance —
#     on `simulate_lqr` data, where the margin comes out positive for the cheap
#     model even on expensive-cost trials.
#
#     So: infer switching from real behaviour or from `rand`, and treat
#     `simulate_lqr` as the tool for *displaying* an optimal trajectory, not for
#     generating data to recover switching from.

# `tied_params` shares a parameter version across states. For an inverse-LQR
# model the structural parameters are coordinates of one constrained
# parameterization rather than separable regression columns, so `:structure` ties
# the whole block ``(A, S, Q_c, h, B_u, G_r)`` and `:noise` ties ``\Sigma``.
#
# Individual blocks may also be named, and that is usually the model you want:
# `tied_params = [:A, :S]` shares the plant and fits a cost per discrete state —
# one body, one task, a goal that changes. A partial tie cannot be a single
# constrained optimization here, since each parameter version owns a full copy of
# every block, so it runs as two alternating passes (shared free, then per-state
# free). Each accepts only an improvement, so the bound still cannot decrease; it
# converges more slowly than a joint step would.

shared_plant = SLDS(;
    A=[0.92 0.08; 0.08 0.92],
    πₖ=[0.5, 0.5],
    LDSs=[sw_state([0.15 0.0; 0.0 0.15]), sw_state([1.5 0.0; 0.0 1.5])],
)
fit!(shared_plant, ys_sw; max_iter=15, progress=false, rng=StableRNG(7),
     tied_params=[:A, :S])
sp1, sp2 = shared_plant.LDSs[1].state_model, shared_plant.LDSs[2].state_model
println("plant shared: ", sp1.A == sp2.A, "   cost differs: ", sp1.Qc[1] != sp2.Qc[1])

#
# A discrete state can also drop the LQR constraint entirely. `free_state_model`
# gives an unconstrained ``2n \times 2n`` transition, so one model can mix "the
# subject was optimizing" with "the subject was doing something else":
#
# ```julia
# mixed = SLDS(; A = P, πₖ = π,
#              LDSs = [sw_state(Qc),                                   # optimizing
#                      LinearDynamicalSystem(free_state_model(M, Σ),   # not
#                                            obs_model)])
# ```
#
# !!! note "The inferred switch time is biased toward smoothness"
#     The costate is the gradient of the cost-to-go, so it *jumps* when the cost
#     changes. The structured variational approximation weights the two costate
#     equations by the responsibilities rather than allowing the jump, so
#     transitions are smoothed over and the recovered switch time carries a bias
#     whose sign depends on the cost contrast. That is inherent to the
#     approximation, not to the fit.

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
@test minimum(diff(elbos_opt)) > -1e-8  #src
@test maximum(abs, init.Qc[1] .- truth.Qc[1]) / maximum(abs, truth.Qc[1]) < 0.3  #src
@test symplectic_defect(init) < 1e-9  #src
@test init.Qc[1] ≈ init.Qc[1]' atol = 1e-12  #src
@test size(z_var) == (4, tsteps)  #src
@test all(isfinite, z_var)  #src
@test minimum(diff(poisson_elbos)) > -1e-6  #src
@test all(iszero, plds.obs_model.C[:, (plant_dim + 1):end])  #src
@test norm(z_track[1:plant_dim, end] .- target) < 0.05  #src
@test size(z_track) == (4, tsteps)  #src
@test sw_elbos[end] > sw_elbos[1]  #src
@test all(symplectic_defect(l.state_model) < 1e-9 for l in slds.LDSs)  #src
@test length(slds.LDSs) == 2  #src
@test sp1.A == sp2.A  #src
@test sp1.Qc[1] != sp2.Qc[1]  #src
@test abs(lo_resp - hi_resp) > 0.25  #src
