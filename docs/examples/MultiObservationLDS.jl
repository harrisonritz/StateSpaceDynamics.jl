# # Several observation models on one latent state
#
# Sometimes one latent process is measured in more than one way at once. A
# reaching experiment records spike counts *and* hand kinematics; the two are
# different kinds of measurement of the same underlying dynamics, and neither on
# its own constrains the latent state as well as both together.
#
# `StateSpaceDynamics` lets you hand `LinearDynamicalSystem` a `NamedTuple` of
# observation models instead of one. They may be of different types — here a
# Poisson emission for spikes and a Gaussian one for kinematics — and each keeps
# its own parameters, priors and channel count.

using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using StableRNGs
using Statistics

rng = StableRNG(2718);

ssd_palette = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#e87ba4"] # hide
default(; # hide
    palette=ssd_palette, framestyle=:box, grid=true, gridalpha=0.12, # hide
    linewidth=2, size=(760, 420), titlefontsize=12, guidefontsize=10, # hide
    legendfontsize=9, foreground_color_legend=nothing, # hide
) # hide

# ## Model
#
# The latent state evolves as it always does; what changes is that *both*
# emissions read from it:
#
# ```math
# \begin{aligned}
#     x_{t+1}       &= A x_t + b + \varepsilon_t,   & \varepsilon_t &\sim \mathcal{N}(0, Q), \\
#     \lambda_t     &= \exp(C^{\mathrm{spk}} x_t + d^{\mathrm{spk}}),
#                                                   & y^{\mathrm{spk}}_{t,i} &\sim \mathrm{Poisson}(\lambda_{t,i}), \\
#     y^{\mathrm{kin}}_t &\sim \mathcal{N}(C^{\mathrm{kin}} x_t + d^{\mathrm{kin}}, R^{\mathrm{kin}}).
# \end{aligned}
# ```
#
# The two emissions are conditionally independent given ``x_t``, so the joint
# log-likelihood is just their sum — which is what makes the smoother, the ELBO
# and the M-step decompose member by member.

latent_dim = 2
spk_dim = 12   # neurons
kin_dim = 4    # e.g. hand position and velocity

A = 0.95 * [cos(0.15) -sin(0.15); sin(0.15) cos(0.15)]
Q = Matrix(0.02 * I(latent_dim))
b = zeros(latent_dim)
x0 = zeros(latent_dim)
P0 = Matrix(0.1 * I(latent_dim))
state_model = GaussianStateModel(A, Q, b, x0, P0)

C_spk = 0.6 * randn(rng, spk_dim, latent_dim)
d_spk = fill(log(2.0), spk_dim)           # baseline ~2 spikes per bin
spk = PoissonObservationModel(C_spk, d_spk)

C_kin = randn(rng, kin_dim, latent_dim)
d_kin = zeros(kin_dim)
R_kin = Matrix(0.05 * I(kin_dim))
kin = GaussianObservationModel(C_kin, R_kin, d_kin)

# Hand both to the constructor as a `NamedTuple`. The keys are yours to choose;
# they are how you address the data, the parameters and the fitting options from
# here on.

true_lds = LinearDynamicalSystem(state_model, (spk=spk, kin=kin))

# A member is reached by its key, and a member's parameter by the key-suffixed
# name — the same spelling `depends_on`, `fit_bool` and `tied_params` use.

true_lds.obs_model.spk

#-

size(true_lds.obs_model.C_kin)

# `obs_dim` is the total across members, and `fit_bool` gains one block per
# emission: `[C d D]` and `R` for the Gaussian member, `[C d D]` alone for the
# Poisson one (which has no noise covariance).

(; obs_dim=true_lds.obs_dim, fit_bool=true_lds.fit_bool)

# ## Sampling
#
# `rand` returns the observations as a `NamedTuple` under the same keys.

tsteps = 100
ntrials = 25
latents, observations = rand(rng, true_lds, fill(tsteps, ntrials));

y = (
    spk=[trial.spk for trial in observations],
    kin=[trial.kin for trial in observations],
);

(; spk=size(y.spk[1]), kin=size(y.kin[1]))

# Spikes are counts and kinematics are continuous, on quite different scales —
# which is exactly why keeping them as separate emissions matters. Stacking them
# into one Gaussian block would model the counts as Gaussian and force a single
# noise covariance across both.

p_data = plot(
    heatmap(y.spk[1]; ylabel="neuron", xlabel="time", title="spike counts (trial 1)",
        colorbar=false),
    plot(y.kin[1]'; xlabel="time", ylabel="kinematics", title="kinematics (trial 1)",
        labels=["dim 1" "dim 2" "dim 3" "dim 4"]),
    layout=(1, 2), size=(900, 340),
)

# ## Fitting
#
# `fit!` takes the observations under the same keys. Everything else — the EM
# loop, the ELBO, `depends_on`, priors — works as it does for a single emission.
# Because one member is Poisson, the model is fitted with the iterative Laplace
# E-step; an all-Gaussian composite would use the single-step smoother instead,
# chosen automatically from the members' types.

naive_lds = LinearDynamicalSystem(
    GaussianStateModel(
        0.9 * Matrix(I(latent_dim)),
        Matrix(0.1 * I(latent_dim)),
        zeros(latent_dim),
        zeros(latent_dim),
        Matrix(0.1 * I(latent_dim)),
    ),
    (
        spk=PoissonObservationModel(
            0.6 * randn(rng, spk_dim, latent_dim), fill(log(1.0), spk_dim)
        ),
        kin=GaussianObservationModel(
            randn(rng, kin_dim, latent_dim),
            Matrix(1.0 * I(kin_dim)),
            zeros(kin_dim),
        ),
    ),
)

x_pre, _ = smooth(naive_lds, y)
elbos = fit!(naive_lds, y; max_iter=50, progress=false)
x_post, _ = smooth(naive_lds, y);

p_elbo = plot(elbos; xlabel="iteration", ylabel="ELBO", legend=false,
    color="#2a78d6", title="Laplace-EM convergence")

# The recovered emissions sit on the right scales: the Gaussian member's noise
# covariance and the Poisson member's baseline rates are both estimated from
# their own data, with no shared noise term forcing a compromise.

(;
    R_kin_fitted=round.(diag(naive_lds.obs_model.kin.R); digits=3),
    R_kin_true=round.(diag(R_kin); digits=3),
    rate_spk_fitted=round(mean(exp.(naive_lds.obs_model.spk.d)); digits=2),
    rate_spk_true=round(mean(exp.(d_spk)); digits=2),
)

# ## Freezing one emission
#
# `fit_bool` names members, so you can hold one readout fixed — a decoder
# calibrated elsewhere, say — while the rest of the model adapts around it.
#
# ```julia
# lds = LinearDynamicalSystem(state_model, (spk = spk, kin = kin);
#                             fit_bool = (kin = (C = false, R = false),))
# ```
#
# The same suffixed names carry into `depends_on`, so one emission can be
# estimated per recording session while another stays shared:
#
# ```julia
# set_depends_on!(lds.obs_model, (C_spk = session, d_spk = session))
# group_parameter(lds.obs_model, :C_spk, :session_a)
# ```
#
# and into an SLDS's `tied_params`, where `(:C_spk, :d_spk)` shares the spike
# readout across regimes while the kinematics readout switches with them.

# ## Tests  #src

using SSDTest  #src
using Test  #src

test_em_improves(elbos)  #src
@test all(>=(0), y.spk[1])  #src
@test length(naive_lds.fit_bool) == 7  #src
@test naive_lds.obs_dim == spk_dim + kin_dim  #src
@test size(naive_lds.obs_model.C_kin) == (kin_dim, latent_dim)  #src
@test elbo_monotone(elbos)  #src
@test isapprox(mean(exp.(naive_lds.obs_model.spk.d)), mean(exp.(d_spk)); rtol=0.4)  #src
