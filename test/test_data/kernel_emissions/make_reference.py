"""Reference values for the kernel-emission tests, from the dynameg Python code.

Run (numpy and scipy are all it needs; the heavy imports of dynameg's
``validate`` modules, mne and pandas, are stubbed below):

    uv venv venv && uv pip install --python venv/bin/python numpy scipy
    venv/bin/python make_reference.py /path/to/dynameg

It writes ``reference.mat`` (MATLAB v5, read by MAT.jl) next to this file.
Every value is a float64 array or scalar; one-dimensional arrays are stored
as columns ``(n, 1)``, three-dimensional ones as they are in numpy (``Z[g, i,
j]``), and every index (``*_groups``, ``*_membership``) is 1-based.

Cases, by key prefix (the functions are named as in dynameg):

``flat1_``, ``flat2_``
    ``dynameg.inverse.fit_kernels(Z, S, method="laplace", n_draws=n,
    prior_sd=..., prior_mean=...)``: the MAP of ``n/2 (log det V + tr(V⁻¹ S))
    + ½ Σ (log θ_g − ν_g)² / σ_g²``, ``V = Σ θ_g Z_g``. ``flat1`` leaves the
    centres to the "alone" rule, ``ν_g = log(tr S / tr Z_g)`` (saved as
    ``prior_mean``), with ``σ = 1``; ``flat2`` gives both per kernel.
    ``objective`` is ``log det V + tr(V⁻¹ S) + (2/n) × the prior term``;
    ``information`` is ``½ tr(V⁻¹ Z_g V⁻¹ Z_h)``; ``posterior_cov`` is
    ``(n θθ' ∘ information + diag(1/σ²))⁻¹``; ``free_energy`` is
    ``−n/2 (log det V + tr(V⁻¹ S) + p log 2π) − prior term + ½ log
    det(posterior_cov diag(1/σ²))``. The script checks all of these.
    ``theta`` and the rest are what ``fit_kernels(..., max_iter=5000,
    tol=1e-13)`` returns: its stopping rule (the objective's relative change)
    leaves ``theta`` about 5e-7 (relative) from the MAP, and a port that
    stops elsewhere differs by as much. The ``*_map_*`` keys are the MAP
    itself, the root of the gradient ``n/2 θ_g (tr(Z_g V⁻¹) − tr(Z_g V⁻¹ S
    V⁻¹)) + (log θ_g − ν_g) / σ_g²`` found here by ``scipy.optimize.root``
    (to about 1e-13), and ``fit_kernels(..., theta=MAP, fit=False)`` there:
    compare a tightly converged fit with those.
``tree1_``
    ``shared_parcel_prior.Hierarchy.update``: leaves → groups → root, with
    ``sd_prior = (s₀, ν₀)`` and ``root_prior = (start, root_sd)``. ``sds`` is
    ``(leaf SD about its group, group SD about the root)``.
``joint1_``
    ``shared_scale_correlation.JointHierarchy.update``. Its Ω-step is L-BFGS-B
    with *numerical* gradients, so these values are a fixed point only as far
    as that optimizer goes: **test this case with a loose tolerance** (1e-6,
    say). On this problem it goes far: the script repeats the sweeps with the
    Ω-step solved from its analytic gradient (d = 2) and prints the
    difference, about 4e-11, also saved as ``exact_difference``. To check an
    optimum rather than the numbers: the last Ω-step minimized

        nll(σ, R) = n_omega/2 log det Ω − (η − 1) log det R + ½ tr(Ω⁻¹ S_omega),
        Ω = diag(σ) R diag(σ),

    with ``S`` = ``resid' resid + Σ_u V_u + Σ_n counts_n var_n`` from the
    saved ``means`` and ``variances``, ``S_omega = S + ν₀ s₀² I`` and
    ``n_omega = U + ν₀``; ``omega_nll`` is its value at the saved ``(sds,
    corr)``, which another optimizer should match or beat.
``sc1_``
    ``shared_scale_correlation.ScaleCorrelationPrior`` at a random
    ``x = (log θ (4), logits a (2 net, then 4 par), log ν (1))``. The factors
    ``sc1_net_F*``, ``sc1_par_F*`` are as given to ``Family``, before the
    class scales their rows to unit norm. ``Gamma`` is ``S C S`` built here
    from the definition, and ``G Γ G' + ν Φ'NΦ`` is checked against
    ``covariance``. ``objective`` and ``gradient`` are without the trees'
    prior (``with_prior=False``).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.optimize import root


def load_reference(checkout: Path):
    """``(fit_kernels, shared_parcel_prior, shared_scale_correlation)`` from
    the checkout, with stand-ins for the modules that need mne or pandas."""
    sys.dont_write_bytecode = True  # nothing is written into the checkout
    sys.path[:0] = [str(checkout / "src"), str(checkout / "validate")]

    def unavailable(*args, **kwargs):
        raise RuntimeError("stubbed by make_reference.py")

    for name, attributes in (("dynameg.export", ("load_dataset",)),
                             ("dynameg.fitting", ("load_fit", "run_fit"))):  # fmt: skip
        stub = types.ModuleType(name)
        for attribute in attributes:
            setattr(stub, attribute, unavailable)
        sys.modules[name] = stub
    from dynameg.inverse import fit_kernels
    import shared_parcel_prior
    import shared_scale_correlation

    return fit_kernels, shared_parcel_prior, shared_scale_correlation


def one_based(index):
    return np.asarray(index, float) + 1


def spd(rng, d, scale=1.0, mix=0.3):
    """``scale (I + mix A)`` for a random symmetric ``A``, made positive
    definite by raising its eigenvalues to at least ``scale / 10``."""
    A = rng.standard_normal((d, d))
    values, vectors = np.linalg.eigh(scale * (np.eye(d) + mix * (A + A.T) / 2))
    return (vectors * np.maximum(values, scale / 10)) @ vectors.T


# --------------------------------------------------------------------------
# A. fit_kernels, a flat prior, Laplace


def flat_cases(fit_kernels):
    rng = np.random.default_rng(20261101)
    p, ranks, n = 6, (6, 3, 2, 6), 40
    Z = np.stack([F @ F.T for F in (rng.standard_normal((p, m)) for m in ranks)])
    true = np.array([2.0, 0.5, 1.0, 0.1])
    Y = np.linalg.cholesky(np.einsum("g,gij->ij", true, Z)) @ rng.standard_normal(
        (p, n)
    )
    S = Y @ Y.T / n
    options = dict(method="laplace", n_draws=n, max_iter=5000, tol=1e-13)
    alone = np.log(np.trace(S) / np.einsum("gii->g", Z))

    out = {}
    cases = (("flat1", np.full(4, 1.0), None),
             ("flat2", np.array([0.5, 1.0, 2.0, 1.0]), np.array([0.3, -0.2, 0.0, -1.0])))  # fmt: skip
    for name, sd, mean in cases:
        if mean is None:  # the scalar SD and the empirical centres, as specified
            fit = fit_kernels(Z, S, prior_sd=1.0, **options)
            given = fit_kernels(Z, S, prior_sd=1.0, prior_mean=alone, **options)
            assert given.converged
            np.testing.assert_allclose(given.theta, fit.theta, rtol=1e-12)
            mean = alone
        else:
            fit = fit_kernels(Z, S, prior_sd=sd, prior_mean=mean, **options)
        assert fit.converged, name
        theta, lam = fit.theta, np.log(fit.theta)
        # What the outputs are, from the docstring of fit_kernels.
        V = np.einsum("g,gij->ij", theta, Z)
        Vi = np.linalg.inv(V)
        fit_term = np.linalg.slogdet(V)[1] + np.sum(Vi * S)
        penalty = np.sum((lam - mean) ** 2 / sd**2) / 2
        z = np.einsum("gij,ji->g", Z, Vi)
        q = np.einsum("gij,ji->g", Z, Vi @ S @ Vi)
        gradient = n / 2 * theta * (z - q) + (lam - mean) / sd**2
        information = np.einsum("gij,hji->gh", Vi @ Z, Vi @ Z) / 2
        curvature = n * np.outer(theta, theta) * information + np.diag(1 / sd**2)
        cov = np.linalg.inv(curvature)
        energy = (-n / 2 * (fit_term + p * np.log(2 * np.pi)) - penalty
                  + np.linalg.slogdet(cov @ np.diag(1 / sd**2))[1] / 2)  # fmt: skip

        def slope(lam):
            Vi = np.linalg.inv(np.einsum("g,gij->ij", np.exp(lam), Z))
            z = np.einsum("gij,ji->g", Z, Vi)
            q = np.einsum("gij,ji->g", Z, Vi @ S @ Vi)
            return n / 2 * np.exp(lam) * (z - q) + (lam - mean) / sd**2

        np.testing.assert_allclose(slope(lam), gradient, rtol=0, atol=1e-12)
        best = root(slope, lam, tol=1e-15).x
        assert np.abs(slope(best)).max() < 1e-12
        at_map = fit_kernels(Z, S, theta=np.exp(best), fit=False, prior_sd=sd,
                             prior_mean=mean, **options)  # fmt: skip
        print(f"{name}: {len(fit.objective) - 1} steps, |gradient in log θ| "
              f"{np.abs(gradient).max():.1e}, θ = {theta}, "
              f"{np.abs(theta / at_map.theta - 1).max():.1e} from the MAP")  # fmt: skip
        assert np.abs(theta / at_map.theta - 1).max() < 1e-5
        assert abs(fit.free_energy - at_map.free_energy) < 1e-6
        assert np.abs(gradient).max() < 1e-6 * n
        np.testing.assert_allclose(fit.objective[-1], fit_term + 2 * penalty / n,
                                   rtol=1e-12)  # fmt: skip
        np.testing.assert_allclose(fit.information, information, rtol=1e-9)
        np.testing.assert_allclose(fit.posterior_cov, cov, rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(fit.free_energy, energy, rtol=1e-12)
        np.testing.assert_allclose(fit.model, V, rtol=1e-12)
        out.update({
            f"{name}_Z": Z, f"{name}_S": S, f"{name}_n": float(n),
            f"{name}_prior_sd": sd, f"{name}_prior_mean": mean,
            f"{name}_theta": theta, f"{name}_free_energy": fit.free_energy,
            f"{name}_posterior_cov": fit.posterior_cov,
            f"{name}_objective": fit.objective[-1],
            f"{name}_information": fit.information,
            f"{name}_gradient_norm": np.abs(gradient).max(),
            f"{name}_map_theta": at_map.theta,
            f"{name}_map_free_energy": at_map.free_energy,
            f"{name}_map_posterior_cov": at_map.posterior_cov,
            f"{name}_map_objective": at_map.objective[-1],
            f"{name}_map_information": at_map.information,
        })  # fmt: skip
    out["flat_theta_true"] = true
    return out


# --------------------------------------------------------------------------
# B. Hierarchy.update, leaves → groups → root


def tree_case(pp):
    rng = np.random.default_rng(20261102)
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    a = rng.standard_normal(12) + np.array([-1.0, 0.5, 1.5])[groups]
    v = rng.uniform(0.05, 0.3, 12)
    start = float(a.mean())
    sd_prior, root_sd = (0.5, 4.0), 3.0
    tree = pp.Hierarchy([groups], [np.full(3, start)], [np.zeros(3)],
                        np.full(2, 1.0), start, sd_prior=sd_prior,
                        root_prior=(start, root_sd))  # fmt: skip
    fitted = tree.update(a, v, sweeps=20000, tol=1e-13)
    again = fitted.update(a, v, sweeps=1, tol=0.0)
    moved = max(np.abs(again.means[0] - fitted.means[0]).max(),
                np.abs(again.variances[0] - fitted.variances[0]).max(),
                np.abs(again.sds - fitted.sds).max(), abs(again.root - fitted.root))  # fmt: skip
    print(f"tree1: one more sweep moves it by {moved:.1e}; sds {fitted.sds}, "
          f"root {fitted.root:.6f}")  # fmt: skip
    assert moved < 1e-11
    assert np.all(fitted.sds > pp.SD_FLOOR + 1e-6)  # the floor is not active
    return {
        "tree1_a": a, "tree1_v": v, "tree1_groups": one_based(groups),
        "tree1_start": start, "tree1_start_sd": 1.0,
        "tree1_sd_prior": np.array(sd_prior), "tree1_root_sd": root_sd,
        "tree1_means": fitted.means[0], "tree1_variances": fitted.variances[0],
        "tree1_sds": fitted.sds, "tree1_root": float(fitted.root),
    }  # fmt: skip


# --------------------------------------------------------------------------
# C. JointHierarchy.update, d = 2


def joint_case(ssc):
    rng = np.random.default_rng(20261103)
    d, n_units = 2, 15
    groups = np.repeat(np.arange(3), 5)
    mix = np.linalg.cholesky(np.array([[1.0, 0.6], [0.6, 1.0]]))
    offsets = np.array([[-1.0, 0.5], [0.0, -0.5], [1.5, 1.0]])
    a = rng.standard_normal((n_units, d)) @ mix.T + offsets[groups]
    V = np.stack([spd(rng, d, 0.05) for _ in range(n_units)])
    start = a.mean(axis=0)
    eta, sd_prior, root_sd = 2.0, (0.5, 4.0), 3.0
    tree = ssc.JointHierarchy.start(groups, start, sd=1.0, lkj_eta=eta,
                                    sd_prior=sd_prior, root_sd=root_sd)  # fmt: skip
    fitted = tree.update(a, V, sweeps=20000, tol=1e-12)
    again = fitted.update(a, V, sweeps=1, tol=0.0)
    moved = max(np.abs(getattr(again, name) - getattr(fitted, name)).max()
                for name in ("means", "variances", "sds", "corr", "psi", "root"))  # fmt: skip
    print(f"joint1: one more sweep moves it by {moved:.1e}; sds {fitted.sds}, "
          f"corr {fitted.corr[1, 0]:.6f}, psi {fitted.psi}")  # fmt: skip
    assert moved < 1e-4
    assert np.all(fitted.sds > ssc.SD_FLOOR + 1e-6)
    assert np.all(fitted.psi > ssc.SD_FLOOR + 1e-6)
    # The second moment and objective of the last Ω-step: update() computes
    # the nodes and S, then Ω, so the returned nodes are those S was made of.
    counts = np.bincount(groups).astype(float)
    resid = a - fitted.means[groups]
    S = resid.T @ resid + V.sum(axis=0) + np.einsum("n,nij->ij", counts,
                                                    fitted.variances)  # fmt: skip
    S = (S + S.T) / 2
    s0, nu0 = sd_prior
    S_omega, n_omega = S + nu0 * s0**2 * np.eye(d), n_units + nu0

    def nll(sigma, R):
        omega = sigma[:, None] * R * sigma[None, :]
        return (n_omega / 2 * np.linalg.slogdet(omega)[1]
                - (eta - 1) * np.linalg.slogdet(R)[1]
                + np.trace(np.linalg.solve(omega, S_omega)) / 2)  # fmt: skip

    exact = exact_joint(ssc, tree, a, V)
    off = max(np.abs(getattr(fitted, name) - exact[name]).max() for name in exact)
    print(f"joint1: {off:.1e} from the fixed point with an exact Ω-step")
    assert off < 1e-6
    return {
        "joint1_exact_difference": off,
        "joint1_a": a, "joint1_V": V, "joint1_groups": one_based(groups),
        "joint1_start": start, "joint1_start_sd": 1.0, "joint1_lkj_eta": eta,
        "joint1_sd_prior": np.array(sd_prior), "joint1_root_sd": root_sd,
        "joint1_means": fitted.means, "joint1_variances": fitted.variances,
        "joint1_sds": fitted.sds, "joint1_corr": fitted.corr,
        "joint1_psi": fitted.psi, "joint1_root": fitted.root,
        "joint1_S": S, "joint1_S_omega": S_omega, "joint1_n_omega": float(n_omega),
        "joint1_omega_nll": nll(fitted.sds, fitted.corr),
        "joint1_last_sweep_change": moved,
    }  # fmt: skip


def exact_joint(ssc, tree, a, V, sweeps=20000, tol=1e-14):
    """The sweeps of ``JointHierarchy.update`` for ``d = 2`` with hyperpriors,
    the Ω-step solved from its analytic gradient: how far the reference's
    numerical L-BFGS-B leaves it from the fixed point."""
    groups, eta = tree.groups, tree.lkj_eta
    (s0, nu0), n_units = tree.sd_prior, len(a)
    n = n_units + nu0
    counts = np.bincount(groups).astype(float)
    sums = np.zeros((len(counts), 2))
    np.add.at(sums, groups, a)
    sigma, R, psi, top = tree.sds.copy(), tree.corr.copy(), tree.psi.copy(), tree.root

    def omega_step(S, sigma, R):
        def slope(x):  # of n (l₁ + l₂) + (n/2 − (η − 1)) log(1 − ρ²) + ½ tr(Ω⁻¹S)
            s1, s2, rho = np.exp(x[0]), np.exp(x[1]), np.tanh(x[2])
            c = 1 - rho**2
            A, B, C = S[0, 0] / s1**2, S[1, 1] / s2**2, S[0, 1] / (s1 * s2)
            return np.array([
                n - (A - rho * C) / c, n - (B - rho * C) / c,
                -(n - 2 * (eta - 1)) * rho - C + (A + B - 2 * rho * C) * rho / c,
            ])  # fmt: skip

        x = root(slope, [*np.log(sigma), np.arctanh(R[1, 0])], tol=1e-15).x
        assert np.abs(slope(x)).max() < 1e-10
        return np.exp(x[:2]), np.array([[1.0, np.tanh(x[2])], [np.tanh(x[2]), 1.0]])

    for _ in range(sweeps):
        old = np.concatenate([top, sigma, R[1], psi])
        Oi = np.linalg.inv(sigma[:, None] * R * sigma[None, :])
        var = np.linalg.inv(counts[:, None, None] * Oi + np.diag(1 / psi**2))
        means = np.einsum("nij,nj->ni", var, sums @ Oi.T + top / psi**2)
        resid = a - means[groups]
        S = resid.T @ resid + V.sum(axis=0) + np.einsum("n,nij->ij", counts, var)
        sigma, R = omega_step(S + nu0 * s0**2 * np.eye(2), sigma, R)
        m0, r = tree.root_prior
        prec = len(means) / psi**2 + 1 / r**2
        top = (means.sum(axis=0) / psi**2 + m0 / r**2) / prec
        total = ((means - top) ** 2 + np.einsum("nii->ni", var) + 1 / prec).sum(axis=0)
        psi = np.sqrt((total + nu0 * s0**2) / (len(means) + nu0))
        if np.abs(np.concatenate([top, sigma, R[1], psi]) - old).max() < tol:
            break
    return dict(means=means, variances=var, sds=sigma, corr=R, psi=psi, root=top)


# --------------------------------------------------------------------------
# D. ScaleCorrelationPrior


def scale_correlation_case(ssc):
    rng = np.random.default_rng(20261104)
    p, n_cols, r = 8, 12, 6
    gain = rng.standard_normal((p, n_cols))
    basis = np.linalg.svd(gain, full_matrices=False)[0][:, :r]
    variance = rng.uniform(0.5, 2.0, n_cols)
    membership = np.repeat(np.arange(4), 3)
    scale_names = [f"b{i}" for i in range(4)]
    scale_groups = np.array([0, 0, 1, 1])
    net_cover = [np.arange(0, 6), np.arange(6, 12)]
    net = []
    for cols in net_cover:
        F = np.zeros((n_cols, 1))
        F[cols] = 1.0
        net.append(F)
    par = []
    for block in range(4):
        F = np.zeros((n_cols, 2))
        F[membership == block] = rng.standard_normal((3, 2))
        par.append(F)
    par_groups = np.array([0, 0, 1, 1])
    families = [
        ssc.Family("net", net, ["n0", "n1"], None, "pooled"),
        ssc.Family("par", par, list(scale_names), par_groups, "nested"),
    ]
    nuisance = np.eye(p)
    prior = ssc.ScaleCorrelationPrior.build(
        gain, basis, variance, membership, scale_names, families, [nuisance],
        ["I"], scale_groups=scale_groups, n_orient=1,
    )  # fmt: skip
    n_x = prior.n_scales + prior.n_rho + prior.n_nuisance
    assert (prior.n_scales, prior.n_rho, prior.n_nuisance) == (4, 6, 1)
    assert prior.ancestors == [[[], []], [[0], [0], [1], [1]]]
    x = 0.5 * rng.standard_normal(n_x)
    S, n_draws = spd(rng, r), 5.0

    theta, rho, nu = prior.weights(x)
    u, logits, w = prior.split(x)
    rest = prior.rho(logits)[1]
    covariance = prior.covariance(x)
    value, gradient = prior.objective(x, S, n_draws, with_prior=False)
    information = prior.information(x)

    # Γ = S C S from the definition: C = I + Σ_c ρ_c (R_c − I_c), R_c = B_c B_c'
    # with B_c's rows at unit norm, s_i² = θ_g(i) d_i.
    C = np.eye(n_cols)
    for rho_c, F in zip(rho, net + par):
        norm = np.linalg.norm(F, axis=1)
        covered = norm > 0
        B = np.zeros_like(F)
        B[covered] = F[covered] / norm[covered, None]
        C += rho_c * (B @ B.T - np.diag(covered.astype(float)))
    scale = np.sqrt(theta[membership] * variance)
    Gamma = scale[:, None] * C * scale[None, :]
    G = basis.T @ gain
    explicit = G @ Gamma @ G.T + nu[0] * basis.T @ nuisance @ basis
    np.testing.assert_allclose(covariance, explicit, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(prior.G, G, rtol=0, atol=0)
    assert np.linalg.eigvalsh(C).min() > 0
    # The stick: ρ_net = σ(a), ρ_par = (1 − ρ of its net) σ(a).
    sigmoid = 1 / (1 + np.exp(-logits))
    np.testing.assert_allclose(rho[:2], sigmoid[:2], rtol=1e-12)
    np.testing.assert_allclose(rho[2:], (1 - rho[[0, 0, 1, 1]]) * sigmoid[2:],
                               rtol=1e-12)  # fmt: skip
    # The objective, and its gradient and information by central differences.
    np.testing.assert_allclose(
        value,
        n_draws / 2 * (np.linalg.slogdet(explicit)[1]
                       + np.trace(np.linalg.solve(explicit, S))),
        rtol=1e-12,
    )  # fmt: skip
    step = 1e-6
    numeric = np.zeros(n_x)
    derivative = np.zeros((n_x, r, r))
    for i in range(n_x):
        e = np.zeros(n_x)
        e[i] = step
        numeric[i] = (prior.objective(x + e, S, n_draws, with_prior=False)[0]
                      - prior.objective(x - e, S, n_draws, with_prior=False)[0]) / (2 * step)  # fmt: skip
        derivative[i] = (prior.covariance(x + e) - prior.covariance(x - e)) / (2 * step)
    np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=1e-7)
    Vi = np.linalg.inv(covariance)
    np.testing.assert_allclose(
        information, np.einsum("iab,jba->ij", Vi @ derivative, Vi @ derivative) / 2,
        rtol=1e-6, atol=1e-8,
    )  # fmt: skip
    print(f"sc1: θ {theta}, ρ {rho}, ν {nu}, objective {value:.6f}")

    out = {
        "sc1_gain": gain, "sc1_basis": basis, "sc1_G": G, "sc1_variance": variance,
        "sc1_membership": one_based(membership),
        "sc1_scale_groups": one_based(scale_groups),
        "sc1_par_groups": one_based(par_groups), "sc1_nuisance": nuisance,
        "sc1_x": x, "sc1_theta": theta, "sc1_rho": rho, "sc1_nu": nu,
        "sc1_rho_rest": rest, "sc1_rho_jacobian": prior.rho_jacobian(logits),
        "sc1_Gamma": Gamma, "sc1_C": C, "sc1_covariance": covariance,
        "sc1_S": S, "sc1_n": n_draws, "sc1_objective": value,
        "sc1_gradient": gradient, "sc1_information": information,
    }  # fmt: skip
    for family, factors in (("net", net), ("par", par)):
        for i, F in enumerate(factors):
            out[f"sc1_{family}_F{i}"] = F
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    fit_kernels, pp, ssc = load_reference(Path(sys.argv[1]).resolve())
    out = {}
    out.update(flat_cases(fit_kernels))
    out.update(tree_case(pp))
    out.update(joint_case(ssc))
    out.update(scale_correlation_case(ssc))
    out = {key: np.asarray(value, np.float64) for key, value in out.items()}
    assert all(np.all(np.isfinite(value)) for value in out.values())
    target = Path(__file__).resolve().parent / "reference.mat"
    savemat(target, out, format="5", oned_as="column")
    # savemat dates the 116-byte text of the header: fix it, so that the same
    # inputs give the same file.
    content = bytearray(target.read_bytes())
    content[:116] = b"MATLAB 5.0 MAT-file, written by make_reference.py".ljust(116)
    target.write_bytes(content)
    print(f"wrote {target}: {len(out)} keys, {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
