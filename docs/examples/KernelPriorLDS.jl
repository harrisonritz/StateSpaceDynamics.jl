# # Kernel priors on a parcellated cortex
#
# A Gaussian LDS fitted to MEG sensors has an emission `C` (sensors × latents)
# that, physically, is a forward model applied to a source pattern: `C = L J`,
# `L` the lead field and `J` (sources × latents) how each latent is spread over
# the cortex. [`ConstrainedKernelGaussianObservationModel`](@ref) fits the LDS
# with that structure built in. The loading is restricted to a basis `Φ` of the
# forward's strong directions, `C = Φ H`, and each column of `H` gets a
# Gaussian prior whose covariance is built from kernels on the cortex, with the
# kernels' weights fitted alongside the dynamics.
#
# This tutorial builds every prior of a model-comparison study on a surface
# parcellation (Schaefer parcels grouped into Yeo networks) on a small
# synthetic cortex. The same code runs on real data once the synthetic arrays
# are replaced by the real ones; [what you need](@ref kp-ingredients) lists
# them.

using StateSpaceDynamics
using LinearAlgebra
using SparseArrays
using Random
using Plots
using StableRNGs
using Statistics

rng = StableRNG(1234);

ssd_palette = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#e87ba4"] # hide
default(; # hide
    palette=ssd_palette, framestyle=:box, grid=true, gridalpha=0.12, # hide
    linewidth=2, size=(760, 420), titlefontsize=12, guidefontsize=10, # hide
    legendfontsize=9, foreground_color_legend=nothing, # hide
) # hide

# ## The model
#
# ```math
# \begin{aligned}
#     y_t &\sim \mathcal{N}(\Phi H x_t + d, R), \\
#     h_j &\sim \mathcal{N}(0, \Sigma_H), \qquad
#     \Sigma_H = \Phi^\top \Big(L\,\Gamma\,L^\top + \nu N\Big)\Phi, \\
#     w &\sim \text{hyperprior},
# \end{aligned}
# ```
#
# where ``h_j`` is the ``j``-th column of ``H``, ``\Gamma`` is a covariance
# over the source columns (the dipoles, times their orientations) built from
# weighted kernels, ``N`` is a sensor-noise covariance with its own weight
# ``\nu`` (the nuisance kernel, which lets the emission carry structure that is
# not cortical), and ``w`` collects every weight. The package never sees ``L``.
# It takes ``\Phi`` and everything already projected to ``H``'s ``r``
# coordinates:
#
# - a **sum-of-kernels** prior (`kind="kernels"` in the study) takes each
#   kernel as an ``r \times m`` factor ``\Phi^\top L F_g``, where
#   ``K_g = F_g F_g^\top`` is the source covariance;
# - a **scale–correlation** prior (`kind="sc"`) takes ``G = \Phi^\top L``
#   (``r \times n``), the depth variance of each column, a partition of the
#   columns into scale blocks, and correlation factors on the columns.
#
# ## [What you need from the pipeline](@id kp-ingredients)
#
# | array | shape | in the dynameg pipeline |
# |---|---|---|
# | `L` | `p × n` | `space.fit_gain(free)`: the forward in the fit's (whitened) sensor coordinates |
# | `w` | `n` | `depth_prior .* orientation_prior(...)`: the depth (and orientation) prior of each column |
# | `Φ` | `p × r` | `leadfield_basis(gain, weights, EMISSION_RANK)`: top left singular vectors of `L * Diagonal(sqrt.(w))` |
# | `N` | `p × p` | `space.adhoc(free)`: the nuisance (sensor-noise) covariance |
# | `column_parcel` | `n` | each column's parcel index, `0` for none: `dipole_parcel` repeated over orientations |
# | `parcel_names`, `parcel_network` | `P` | `parcel_names` and `parcel_network(name)` |
# | network / parcel / geodesic factors | `n × m` each | `separate_networks`, `separate_parcels`, `separate_geodesy` |
# | `F_geo` | `n × m` | `geodesy(space, src, GEO_WIDTH)`: the whole-cortex geodesic factor, before the depth `scale` |
#
# ## A synthetic cortex
#
# Three networks of three parcels, ten dipoles per parcel with one (normal)
# orientation each, and a few dipoles outside every parcel. The dipoles lie on
# a line so that a geodesic distance is just the distance along it.

networks = ["Vis", "SomMot", "Default"]
parcel_names = [string(net, "_", j) for net in networks for j in 1:3]
parcel_network = [net for net in networks for _ in 1:3]
P = length(parcel_names)
column_parcel = vcat([fill(c, 10) for c in 1:P]..., zeros(Int, 6))   # 0: in no parcel
n = length(column_parcel)
surface_mm = Float64.(1:n)                                            # position along the surface
column_network = [c == 0 ? "none" : parcel_network[c] for c in column_parcel]

p, r, k = 40, 24, 3                       # sensors, basis rank, latents
L = randn(rng, p, n) .* (0.3 .+ rand(rng, 1, n))     # deep dipoles have weak gain
w = 1 ./ vec(sum(abs2, L; dims=1))                    # depth prior
Φ = svd(L .* sqrt.(w)').U[:, 1:r]
G = Φ' * L                                            # r × n
Gw = G .* sqrt.(w)'                                   # G with the depth prior's scale
N = Matrix(1.0I, p, p)                                # whitened sensors: noise is I
Z_noise = Φ' * N * Φ;

# A Gaussian of the distance, and a factor of a PSD matrix:

gauss(d; width=3.0) = exp(-d^2 / (2 * width^2))
psd_factor(K) = (E = eigen(Symmetric(K)); E.vectors .* sqrt.(max.(E.values, 0))');

# ## Sum-of-kernels priors: `vb_depth`, `vb_depth-geo`, `vb_depth-parcel`
#
# ``\Sigma_H = \sum_g \theta_g\,\Phi^\top L K_g L^\top \Phi + \nu\,\Phi^\top N \Phi``,
# one weight per kernel, each with a fixed Gaussian prior on its log
# (`levels=:flat`, the default; `prior_sd` is the notebook's `KERNEL_PRIOR_SD`).
# The source covariances, as factors:
#
# - `depth`: ``K = \mathrm{diag}(w)``, weighted minimum norm;
# - `geo`: ``\mathrm{diag}(\sqrt w)\, K_{\text{geo}}\, \mathrm{diag}(\sqrt w)``, a
#   Gaussian of the distance along the surface;
# - `parcel`: the dipoles of a parcel share one current, a column per parcel,
#   all parcels in **one** kernel with one weight.
#
# `factor=true` passes ``\Phi^\top L F`` (``r \times m``) and the package keeps a
# factor of its rank, so ``m`` can be the number of dipoles.

F_geo = psd_factor([gauss(surface_mm[i] - surface_mm[j]) for i in 1:n, j in 1:n])
F_parcel = Float64[column_parcel[i] == c for i in 1:n, c in 1:P]
cortical = Dict("depth" => Gw, "geo" => Gw * F_geo, "parcel" => Gw * F_parcel)

function kernel_prior(names; prior_sd=1.0)
    kernels = [Kernel(cortical[g]; name=g, type=:cortex, factor=true) for g in names]
    push!(kernels, Kernel(Z_noise; name="I", type=:nuisance))
    return KernelPrior(kernels; prior_sd)
end

kernel_prior(["depth", "geo"])

# ## Scale–correlation priors: the `sc_*` models
#
# Here ``\Gamma = S\,C\,S``, with
#
# - **scales** ``S = \mathrm{diag}(s)``, ``s_i^2 = \theta_{b(i)}\,w_i``: the
#   depth prior times one weight per *scale block* ``b``. The block is the
#   column's parcel (`parcel_scale`), its network (`network_scale`), or one
#   block for the whole cortex (`global_scale`);
# - **coherence** ``C = I + \sum_c \rho_c (R_c - I_c)``, a correlation matrix
#   from *correlation families*: `network` (the dipoles of a network share a
#   current), `parcel` (they share one within a parcel) and `geo` (within a
#   parcel, nearby dipoles more than distant ones). The families go coarse to
#   fine, and each kernel lies inside one kernel of every earlier family.
#
# A [`ScaleCorrelation`](@ref) holds both. Its scale weights have the `type` of
# the scale selector, and each [`CorrelationFamily`](@ref) has its own
# `type`. The nuisance kernel stays an ordinary [`Kernel`](@ref) beside it.
#
# The families, built once. A network or parcel kernel is a single indicator
# column (a correlation of one across its dipoles), and a geodesic kernel is a
# factor of the within-parcel Gaussian. A kernel's `name` is also its `unit`,
# the key that matches it with the scale of the same parcel in a joint tree,
# so the parcel and geo kernels are named by their parcel.

indicator(mask) = sparse(reshape(Float64.(mask), :, 1))
function geo_factor(c)
    cols = findall(==(c), column_parcel)
    F = zeros(n, length(cols))
    F[cols, :] = psd_factor([gauss(surface_mm[i] - surface_mm[j]) for i in cols, j in cols])
    return sparse(F)
end

FAMILIES = Dict(
    :network => CorrelationFamily(
        [indicator(column_network .== net) for net in networks];
        type=:network, names=networks,
    ),
    :parcel => CorrelationFamily(
        [indicator(column_parcel .== c) for c in 1:P];
        type=:parcel, names=parcel_names, groups=parcel_network,
    ),
    :geo => CorrelationFamily(
        [geo_factor(c) for c in 1:P]; type=:geo, names=parcel_names, groups=parcel_network
    ),
);

# ### The scale blocks
#
# The dipoles in no parcel get a block of their own, `"none"`, as in dynameg.
# Its `group` (the node it hangs from in a nested tree) is `"none"` too.

function scale_blocks(scale)
    if scale === :parcel
        names = [parcel_names; "none"]
        block = [c == 0 ? P + 1 : c for c in column_parcel]
        return (; names, block, groups=[parcel_network; "none"], type=:parcel_scale)
    elseif scale === :network
        names = [networks; "none"]
        block = [findfirst(==(net), names) for net in column_network]
        return (; names, block, groups=nothing, type=:network_scale)
    elseif scale === :global
        return (; names=["global"], block=ones(Int, n), groups=nothing, type=:global_scale)
    end
    throw(ArgumentError("scale must be :parcel, :network or :global"))
end;

# ### The trees
#
# Each type of weight gets a hyperprior on its logs (or logits), set by
# `levels`. dynameg's modes map onto them as follows:
#
# | dynameg | SSD.jl `levels` | meaning |
# |---|---|---|
# | `mode="nested"` | `:nested`, `groups` = the network | parcel ~ its network's mean, networks ~ a root |
# | `mode="pooled"` | `:hierarchical` | every parcel ~ one fitted mean and spread |
# | `mode="fixed"` | `:flat` | a fixed Gaussian on each weight |
# | network coherences (always pooled) | `:hierarchical` | |
# | `network_scale` (always pooled) | `:hierarchical` | |
# | `global_scale` (one weight) | `:flat` (SD `prior_sd`) | one value has no spread to fit |
# | `joint=(a, b, …)` | `correlated=[(a, b, …)]` | one multivariate tree: a unit's weights of those types covary (LKJ prior) |
# | `joint_mode` | the shared `levels` of the joined types | the joined types must use the same `levels` |
# | `sd_prior`, `root_sd`, `lkj_eta` | the same keywords of [`KernelPrior`](@ref) | defaults `(0.5, 4.0)`, `3.0`, `2.0` |
#
# A dynameg pooled tree has one node between the leaves and the root. That
# is `:hierarchical`, whose one node is the root.
#
# One function builds every `sc_*` prior:

function sc_prior(;
    scale=:parcel, families=(), mode=:pooled, joint=(), joint_mode=mode, kwargs...
)
    tree(m) = m === :nested ? :nested : m === :pooled ? :hierarchical : :flat
    blocks = scale_blocks(scale)
    structure = ScaleCorrelation(
        G, w, blocks.block;
        names=blocks.names, type=blocks.type, groups=blocks.groups,
        families=[FAMILIES[f] for f in families],
    )
    levels = Dict{Symbol,Symbol}()
    levels[blocks.type] = scale === :parcel ? tree(mode) : scale === :network ? :hierarchical : :flat
    for f in families
        levels[f] = f === :network ? :hierarchical : tree(mode)
    end
    for member in joint                         # a joint tree's members share levels
        levels[member] = tree(joint_mode)
    end
    return KernelPrior(
        [Kernel(Z_noise; name="I", type=:nuisance)];
        structure, levels, correlated=isempty(joint) ? () : [Tuple(joint)], kwargs...,
    )
end

sc_prior(; families=(:network, :parcel, :geo), joint=(:parcel_scale, :parcel, :geo))

# ## Every model of the study
#
# Each dynameg `MODELS` entry, as arguments to the two builders. The `kind`
# picks the builder. A scale selector that dynameg lists in `families`
# (`"global_scale"`, `"network_scale"`) becomes `scale=`. `"parcel_scale"` is
# the default, and `"I"` is always the nuisance kernel. Variants that differ
# only in their `evoked` / `bands` lifts have the same fit and are listed once
# (see [below](@ref kp-lifts)).

MODELS = [
    "control" => (; kind=:kernels, cortical=["depth"], flat=true),
    "vb_depth" => (; kind=:kernels, cortical=["depth"]),
    "vb_depth-geo" => (; kind=:kernels, cortical=["depth", "geo"]),
    "vb_depth-parcel" => (; kind=:kernels, cortical=["depth", "parcel"]),
    "sc_parcel_scale-nested" => (; kind=:sc, mode=:nested),
    "sc_parcel_scale-pooled" => (; kind=:sc, mode=:pooled),
    "sc_geo-ind-nested" => (; kind=:sc, families=(:geo,), mode=:nested),
    "sc_geo-ind-pooled" => (; kind=:sc, families=(:geo,), mode=:pooled),
    "sc_geo-joint-nested" =>
        (; kind=:sc, families=(:geo,), mode=:nested, joint=(:parcel_scale, :geo)),
    "sc_geo-joint-pooled" =>
        (; kind=:sc, families=(:geo,), mode=:pooled, joint=(:parcel_scale, :geo)),
    "sc_parcel-ind-nested" => (; kind=:sc, families=(:parcel,), mode=:nested),
    "sc_parcel-ind-pooled" => (; kind=:sc, families=(:parcel,), mode=:pooled),
    "sc_parcel-joint-nested" =>
        (; kind=:sc, families=(:parcel,), mode=:nested, joint=(:parcel_scale, :parcel)),
    "sc_parcel-joint-pooled" =>
        (; kind=:sc, families=(:parcel,), mode=:pooled, joint=(:parcel_scale, :parcel)),
    "sc_network-ind-pooled" => (; kind=:sc, families=(:network,)),
    "sc_globalScale-geo-ind-pooled" => (; kind=:sc, scale=:global, families=(:geo,)),
    "sc_globalScale-parcel-ind-pooled" => (; kind=:sc, scale=:global, families=(:parcel,)),
    "sc_globalScale-network-ind-pooled" =>
        (; kind=:sc, scale=:global, families=(:network,)),
    "sc_globalScale-npg-ind-pooled" =>
        (; kind=:sc, scale=:global, families=(:network, :parcel, :geo)),
    "sc_netScale-geo-ind-pooled" => (; kind=:sc, scale=:network, families=(:geo,)),
    "sc_netScale-parcel-ind-pooled" => (; kind=:sc, scale=:network, families=(:parcel,)),
    "sc_netScale-network-ind-pooled" => (; kind=:sc, scale=:network, families=(:network,)),
    "sc_netScale-network-joint-pooled" => (;
        kind=:sc, scale=:network, families=(:network,), joint=(:network_scale, :network)
    ),
    "sc_npg-ind-pooled" => (; kind=:sc, families=(:network, :parcel, :geo)),
    "sc_npg-joint-pooled" => (;
        kind=:sc, families=(:network, :parcel, :geo), joint=(:parcel_scale, :parcel, :geo)
    ),
]

function prior_of(spec)
    spec.kind === :kernels && return kernel_prior(spec.cortical)
    return sc_prior(; Base.structdiff(spec, (; kind=nothing))...)
end;

# ## Data and the warm start
#
# Three latents, each a pattern in one parcel of a different network. The truth
# lies in the span of `Φ`, as the constraint assumes.

J = zeros(n, k)
for (j, c) in enumerate((2, 5, 8))
    cols = findall(==(c), column_parcel)
    J[cols, j] .= sqrt.(w[cols]) .* (2 .+ 0.3 .* randn(rng, length(cols)))
end
state() = GaussianStateModel(;
    A=[0.9 -0.25 0; 0.25 0.9 0; 0 0 0.85], Q=Matrix(0.1I, k, k),
    b=zeros(k), x0=zeros(k), P0=Matrix(1.0I, k, k),
)
truth = LinearDynamicalSystem(
    state(), GaussianObservationModel(; C=Φ * Φ' * L * J, R=Matrix(4.0I, p, p), d=zeros(p))
)
y_train = rand(rng, truth, fill(50, 4))[2]
y_test = rand(rng, truth, fill(50, 8))[2];

# As in the notebook, every model starts from the same unconstrained fit. Its
# loading is projected into the span of `Φ`.

warm = LinearDynamicalSystem(
    state(),
    GaussianObservationModel(; C=randn(rng, p, k), R=Matrix(1.0I, p, p), d=zeros(p)),
)
fit!(warm, y_train; max_iter=200, progress=false)
C0 = Φ * (Φ' * warm.obs_model.C)
R0 = copy(warm.obs_model.R);

# ## The fit
#
# One emission per model and the same EM for all. dynameg keeps the latents at
# a unit second moment every iteration, which is `gauge=:moment` here. The
# default, `:bound`, lets the variational bound choose the gauge instead (see
# [`KernelGaussianObservationModel`](@ref)). Both give the same likelihood
# terms. They differ in how the scale is split between the latents and `H`,
# and so in what the prior on `H` sees.
#
# The `control` is the same EM with nothing to shrink `H`. dynameg gets it by
# fitting the weights to the warm start, multiplying `Σ_H` by `10⁸` and
# holding them (`flat=True, learn=False`). The same steps here:
#
# 1. `fit!(…; max_iter=1)` runs only the start, which fits the weights to the
#    warm start's loading (in the unit-moment gauge) and returns before any
#    M-step;
# 2. the log weights are shifted by `log(1e8)`;
# 3. the weights' `fit_bool` slot (the 7th, named `weights`) is switched off.

function fit_model(spec; max_iter=60, gauge=:moment)
    om = ConstrainedKernelGaussianObservationModel(;
        basis=Φ, prior=prior_of(spec), C=C0, R=R0, gauge
    )
    lds = LinearDynamicalSystem(deepcopy(warm.state_model), om)
    if get(spec, :flat, false)
        fit!(lds, y_train; max_iter=1, progress=false)
        om.prior.weights .+= log(1e8)
        lds.fit_bool[7] = false
    end
    trace = fit!(lds, y_train; max_iter, progress=false)
    return lds, trace
end

fits = Dict(name => fit_model(spec) for (name, spec) in MODELS);

# Without the exact replica, a [`ConstrainedGaussianObservationModel`](@ref)
# is the plain control: `C = Φ H` with `H` free, a point estimate rather than
# a posterior.

ml = LinearDynamicalSystem(
    deepcopy(warm.state_model), ConstrainedGaussianObservationModel(; basis=Φ, C=C0, R=R0)
)
fit!(ml, y_train; max_iter=60, progress=false);

# ## Comparing the models
#
# `loglikelihood` of a kernel-prior emission is the Gaussian one at the mean
# loading, a plug-in score comparable across all of them and with the
# unconstrained warm start.

names_in_order = first.(MODELS)
heldout = [loglikelihood(fits[name][1], y_test) for name in names_in_order]
baseline = loglikelihood(ml, y_test)
bar(
    names_in_order, heldout .- baseline;
    xrotation=60, legend=false, size=(860, 480), bottom_margin=18Plots.mm,
    ylabel="held-out log-lik. − constrained ML", title="Held-out fit by prior",
)

# The truth here is one parcel per latent, uniform within it. So the priors
# that can say "these dipoles move together" (parcel and network coherence,
# the `parcel` kernel) do best, and the `control`, with nothing to shrink
# `H`, does worst. That ranking belongs to this synthetic truth, not to real
# data.
#
# ## Reading the weights
#
# [`kernel_weights`](@ref) gives one row per weight: its `type`, `name`,
# unconstrained `weight`, natural `value` (a variance weight ``\theta``, or a
# correlation ``\rho``) and Laplace `sd`. In a pooled parcel-scale model the
# three active parcels stand out:

rows = kernel_weights(fits["sc_parcel_scale-pooled"][1].obs_model)
scales = filter(row -> row.type === :parcel_scale, rows)
bar(
    [row.name for row in scales], [row.weight for row in scales];
    yerror=[row.sd for row in scales], legend=false, xrotation=45,
    ylabel="log θ (± Laplace SD)", title="Parcel scales, pooled tree",
)

# The coherences of a joint model, and the noise weight:

rows = kernel_weights(fits["sc_npg-joint-pooled"][1].obs_model)
[(row.type, row.name, round(row.value; digits=3)) for row in rows if row.type ∉ (:parcel_scale,)]

# The fitted hyperprior is the trees' state, on the prior. Each tree covers one
# type, or one correlated set of types:

prior = fits["sc_npg-joint-pooled"][1].obs_model.prior
[(tree.types, tree.levels, round.(tree.sds; digits=3)) for tree in prior.trees]

# A joint tree also has the across-unit correlation of its types. Here a
# parcel's scale and its parcel coherence go together: the active parcels are
# also the coherent ones.

joint_tree = only(filter(tree -> length(tree.types) > 1, prior.trees))
joint_tree.types, round.(joint_tree.corr; digits=2)

# ## [Evoked and band priors (after the fit)](@id kp-lifts)
#
# dynameg's `evoked` and `bands` options give a subspace of the latents its own
# prior *after* the fit: the input-driven directions, or the eigenmodes of `A`
# in one frequency band. They do not change the fit, and SSD.jl does not run
# them (the source-space maps they feed stay in dynameg). The step they share
# can be done with the package, though: fit a fresh prior to the second moment
# of ``H`` restricted to the subspace. With ``q(H)`` from
# [`emission_posterior`](@ref), ``H = \bar H + A E B^\top`` and
# ``\operatorname{Var} E_{lj} = V_{lj}``, so for a projector ``M`` onto the
# subspace
#
# ```math
# \mathbb{E}[H M H^\top] = \bar H M \bar H^\top
#     + A\,\mathrm{diag}\!\big(V\,\mathrm{diag}(B^\top M B)\big)\,A^\top .
# ```

function subspace_moment(q, M)
    D = Diagonal(q.var * diag(q.right' * M * q.right))
    return q.mean * M * q.mean' + q.left * D * q.left'
end

# The bands: a real basis of the eigenvectors of `A` whose frequency lies in
# the band. Here, the rotating pair versus the slow real mode.

function band_projector(A, select)
    E = eigen(A)
    keep = findall(select, angle.(E.values))
    V = reduce(hcat, [[real(E.vectors[:, i]) imag(E.vectors[:, i])] for i in keep])
    Q = Matrix(qr(V).Q)[:, 1:rank(V)]
    return Q * Q'
end

lds = fits["sc_geo-joint-pooled"][1]
q = emission_posterior(lds.obs_model)
A = lds.state_model.A
for (band, select) in (("slow", θ -> abs(θ) < 0.05), ("rotation", θ -> abs(θ) >= 0.05))
    M = band_projector(A, select)
    dims = round(Int, tr(M))
    band_prior = sc_prior(; families=(:geo,), joint=(:parcel_scale, :geo))
    fit_kernel_weights!(band_prior, subspace_moment(q, M) ./ dims, dims; rounds=30)
    top = sort(
        filter(row -> row.type === :parcel_scale, kernel_weights(band_prior));
        by=row -> -row.weight,
    )
    println(rpad(band, 9), dims, " directions; largest parcel scale: ", top[1].name)
end

# An evoked prior is the same with `M` the projector onto the span of the
# latents' expected response to the inputs. A model fitted with
# `fit!(lds, y; ux=…)` has that response from its `A` and `B`.
#
# ## What carries over, and what does not
#
# - `flat_iter` (constrained iterations with the prior off before it comes on)
#   has no keyword. Fit a `ConstrainedGaussianObservationModel` for those
#   iterations, then start the kernel-prior emission from its `C` and `R`.
# - dynameg centres a logit tree at the mean of its starting logits, and
#   SSD.jl at `logit(0.1)`. The trees are fitted, so this changes where the
#   first iterations start, not the model.
# - These emissions need a `GaussianStateModel` and are not available in an
#   `SLDS`, a composite emission, or with `depends_on`.
