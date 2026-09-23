"""
    reference_permutation_audit(Gfit, Gtruth; max_columns=8)

Exhaustively match small target banks without changing their reported recovery
score. `order` indexes fitted columns in truth order. The positive scalar is a
separate diagnostic: a permutation alone cannot correct a radius error.
Target labels are observed inputs, so neither operation is an allowed gauge.
"""
function reference_permutation_audit(Gfit, Gtruth; max_columns::Int=8)
    size(Gfit) == size(Gtruth) || throw(DimensionMismatch("reference sizes differ"))
    m = size(Gfit, 2)
    1 <= m <= max_columns || throw(ArgumentError("audit needs 1:$max_columns columns"))
    best = Ref((error=Inf, order=Int[], scale=NaN, scaled=Inf))
    function visit!(order, remaining)
        if isempty(remaining)
            F = Gfit[:, order]
            e = score(F, Gtruth).rmse
            if e < best[].error
                a = max(0.0, sum(F .* Gtruth) / max(sum(abs2, F), eps()))
                best[] = (error=e, order=copy(order), scale=a,
                    scaled=score(a .* F, Gtruth).rmse)
            end
            return
        end
        for j in remaining
            visit!(vcat(order, j), filter(!=(j), remaining))
        end
    end
    visit!(Int[], collect(1:m))
    return best[]
end

"""
    reference_terminal_evidence(lds, uxs)

Log evidence of the terminal event, `Σ log p(terminal = 0 | θ, u)`, before
conditioning on observations. The package computes the same normalizer by
backward square-root integration; this forward pass is kept as an independent
check on it, and supplies the normalizer's curvature to [`reference_em_audit`](@ref).
"""
function reference_terminal_evidence(lds, uxs)
    sm = lds.state_model
    sm.terminal || return 0.0
    total = 0.0
    for u in uxs
        mu, V = copy(sm.x0), Matrix(sm.P0)
        T = size(u, 2)
        for t in 1:(T - 1)
            k = SSD._regime(sm, t)
            M = sm.cache.M[k]
            mu = M * mu + sm.cache.bfwd + sm.cache.Bfwd[k] * u[:, t]
            V = M * V * M' + Matrix(sm.cache.Qfwd)
        end
        k = SSD._terminal_regime(sm, T)
        L = sm.cache.Lf[k]
        r = L * mu - sm.hf + sm.cache.Ftrm[k] * u[:, T]
        W = Symmetric(L * V * L' + Matrix(sm.Σf))
        total += logpdf(MvNormal(zeros(length(r)), W), -r)
    end
    return total
end

#=
Value, gradient and negative Hessian of `value` at `g` by central differences.
Every objective audited here is an exact quadratic in `vec(Gref)` (the reference
moves means, never covariances), so the differences are exact up to rounding.
=#
function _quadratic_information(value, g, step)
    p = length(g)
    base = value(g)
    grad, H = zeros(p), zeros(p, p)
    E = Matrix(step * I, p, p)
    for i in 1:p
        plus, minus = value(g + E[:, i]), value(g - E[:, i])
        grad[i] = (plus - minus) / (2step)
        H[i, i] = -(plus - 2base + minus) / step^2
        for j in 1:(i - 1)
            a, b = E[:, i], E[:, j]
            H[i, j] = H[j, i] = -(value(g+a+b) - value(g+a-b) -
                value(g-a+b) + value(g-a-b)) / (4step^2)
        end
    end
    return base, grad, H
end

# Evaluate `f(work)` at a copy of `lds` whose `Gref` is set from `v`.
function _gref_evaluator(f, lds)
    work = deepcopy(lds)
    sm = work.state_model
    return v -> begin
        sm.Gref .= reshape(v, size(sm.Gref))
        refresh!(sm)
        f(work)
    end
end

"""
    reference_likelihood_profile(lds, ys, uxs; conditional=true, step=0.25)

With all other parameters fixed, the Gaussian marginal log likelihood is an
exact quadratic in `vec(Gref)`. Recover its observed information and unique
optimum by central differences. This integrates out costates and bypasses EM;
it distinguishes slow EM from an actually flat reference likelihood. The
input model is copied. This small diagnostic is not a general fitting method.

For a terminal model, `conditional` sets the copy's `condition_terminal`:
`true` scores `log p(y | terminal = 0)` — the package default, what EM maximizes,
and the likelihood of the terminal-conditioned sampler's draws — and `false`
scores the joint `log p(y, terminal = 0)`. Without a terminal factor they agree.
"""
function reference_likelihood_profile(lds, ys, uxs; conditional::Bool=true, step=0.25)
    lds.obs_model isa GaussianObservationModel ||
        throw(ArgumentError("quadratic reference audit requires Gaussian observations"))
    step > 0 || throw(ArgumentError("finite-difference step must be positive"))
    g = vec(copy(lds.state_model.Gref))
    p = length(g)
    0 < p <= 32 || throw(ArgumentError("quadratic audit requires 1:32 Gref entries"))
    value = _gref_evaluator(lds) do work
        work.state_model.condition_terminal = conditional
        elbo(work, ys; ux=uxs)
    end
    base, grad, H = _quadratic_information(value, g, step)
    ev = eigvals(Symmetric(H))
    cutoff = max(maximum(abs, ev) * 1e-7, 1e-8)
    r = count(>(cutoff), ev)
    optimum = r == p ? g + Symmetric(H) \ grad : fill(NaN, p)
    gain = r == p ? value(optimum) - base : NaN
    # An off-grid point checks that numerical differences captured a quadratic.
    delta = step .* sin.(1:p)
    residual = abs(value(g + delta) - (base + dot(grad, delta) - dot(delta, H * delta)/2))
    return (; rank=r, dimension=p, eigenvalues=ev, information=H, gradient=grad,
        optimum=reshape(optimum, size(lds.state_model.Gref)), gain,
        quadratic_residual=residual, conditional=(conditional && lds.state_model.terminal))
end

"""
Check one Gref-only EM step against its exact quadratic solution. The complete
information uses the mixed-coordinate residual, independently of the M-step's
packed objective. When the terminal event is conditioned on, the M-step
maximizes `Q(θ | θ′) − Σ log Z(θ)`, so the normalizer's own curvature — from the
independent [`reference_terminal_evidence`](@ref) — is subtracted. For fixed
nuisance parameters the predicted update is `I_mstep \\ gradient`, the gradient
being that of the objective EM ascends (Fisher's identity). Generalized
eigenvalues of observed versus M-step information quantify the fraction of error
removed per ideal EM step along each eigenmode; a tiny fraction predicts slow EM
even when the marginal likelihood has a unique, well-determined maximum.
"""
function reference_em_audit(lds, ys, uxs, profile; step=0.25)
    sm = lds.state_model
    conditioned = sm.terminal && sm.condition_terminal
    profile.conditional == conditioned || throw(ArgumentError(
        "profile objective (conditional=$(profile.conditional)) is not the one EM " *
        "ascends (conditional=$conditioned)"))
    n, m = size(sm.Gref)
    W = inv(Matrix(sm.Σ))[(n+1):2n, (n+1):2n]
    complete = zeros(n*m, n*m)
    for u in uxs
        T = size(u, 2)
        for t in 1:(T-1)
            Q = sm.Qc[SSD._regime(sm, t)]
            complete .+= kron(u[:, t] * u[:, t]', Q' * W * Q)
        end
        if sm.terminal
            Q = sm.Qc[SSD._terminal_regime(sm, T)]
            complete .+= kron(u[:, T] * u[:, T]', Q' * (Matrix(sm.Σf) \ Q))
        end
    end
    normalizer = if conditioned
        _quadratic_information(
            _gref_evaluator(w -> reference_terminal_evidence(w, uxs), lds),
            vec(copy(sm.Gref)), step,
        )[3]
    else
        zeros(n*m, n*m)
    end
    mstep = Symmetric(complete - normalizer)
    delta = mstep \ profile.gradient
    work = deepcopy(lds)
    work.fit_bool .= false
    work.fit_bool[3] = true
    work.state_model.fit_flags = LQRFitFlags(; A=false, S=false, Qc=false,
        h=false, Bu=false, Gref=true, terminal=false)
    # LQR fit! includes a final E-step: two reported iterations mean one M-step.
    fit!(work, ys; ux=uxs, max_iter=2, tol=0.0, progress=false)
    actual = vec(work.state_model.Gref - sm.Gref)
    fractions = eigvals(Symmetric(profile.information), mstep)
    return (; step_relative_error=norm(actual-delta)/max(norm(delta), 1e-12),
        fractions, predicted_step=delta, actual_step=actual)
end

function reference_audit_selftest(; verbose::Bool=true)
    truth = lqr_truth(; n=3, nref=4, tsteps=6)
    initial = fit_model(truth; known_plant=true, free_gref=true)
    @assert initial.Gref ≈ truth.sm.Gref[:, [2, 3, 4, 1]] / 3
    matching = reference_permutation_audit(initial.Gref, truth.sm.Gref)
    @assert matching.order == [4, 1, 2, 3]
    @assert matching.error ≈ 2/3
    @assert matching.scale ≈ 3
    @assert matching.scaled < 1e-12
    rng = MersenneTwister(19)
    us = target_inputs(rng, 4, 8, 6)
    ys = simulate(rng, truth, 8; uxs=us)
    probe = deepcopy(truth.lds)
    probe.state_model.Gref .= initial.Gref
    refresh!(probe.state_model)
    relabelled = deepcopy(probe)
    relabelled.state_model.Gref .= probe.state_model.Gref[:, matching.order]
    refresh!(relabelled.state_model)
    original_ll = elbo(probe, ys; ux=us)
    @assert isapprox(elbo(relabelled, ys; ux=[u[matching.order, :] for u in us]),
        original_ll; atol=1e-7, rtol=0)
    @assert abs(elbo(relabelled, ys; ux=us) - original_ll) > 1e-3
    G0 = copy(probe.state_model.Gref)
    p = reference_likelihood_profile(probe, ys, us)
    @assert p.rank == 12
    @assert p.quadratic_residual < 1e-5
    @assert p.gain > 0
    em = reference_em_audit(probe, ys, us, p)
    @assert em.step_relative_error < 1e-3
    @assert all(0 .< em.fractions .< 1)
    @assert probe.state_model.Gref == G0

    #=
    Terminal conditioning. The joint and conditional objectives differ by exactly
    the terminal evidence, computed three ways: the package's two `elbo`s, the
    package's normalizer, and the independent forward pass. The profiles must
    then differ by the normalizer's curvature, and the conditional EM step —
    whose M-step carries `−log Z` — must still match its quadratic prediction.
    =#
    tt = lqr_truth(; n=3, nref=4, tsteps=6, terminal=true, onset=3)
    trng = MersenneTwister(23)
    tus = target_inputs(trng, 4, 8, 6)
    tys = simulate(trng, tt, 8; uxs=tus)
    tprobe = deepcopy(tt.lds)
    tprobe.state_model.Gref .= fit_model(tt; known_plant=true, free_gref=true).Gref
    refresh!(tprobe.state_model)
    @assert tprobe.state_model.condition_terminal
    joint_model = deepcopy(tprobe)
    joint_model.state_model.condition_terminal = false
    evidence = reference_terminal_evidence(tprobe, tus)
    @assert evidence < 0
    @assert isapprox(elbo(joint_model, tys; ux=tus) - elbo(tprobe, tys; ux=tus),
        evidence; atol=1e-8, rtol=0)
    @assert isapprox(sum(u -> SSD._lqr_terminal_logz(tprobe.state_model, u), tus),
        evidence; atol=1e-8, rtol=0)
    pc = reference_likelihood_profile(tprobe, tys, tus; conditional=true)
    pj = reference_likelihood_profile(tprobe, tys, tus; conditional=false)
    @assert pc.conditional && !pj.conditional
    @assert pc.rank == pj.rank == 12
    @assert max(pc.quadratic_residual, pj.quadratic_residual) < 1e-5
    Hz = _quadratic_information(
        _gref_evaluator(w -> reference_terminal_evidence(w, tus), tprobe),
        vec(copy(tprobe.state_model.Gref)), 0.25,
    )[3]
    @assert norm(pj.information - pc.information - Hz) <= 1e-6 * norm(pj.information)
    tem = reference_em_audit(tprobe, tys, tus, pc)
    @assert tem.step_relative_error < 1e-3
    @assert all(0 .< tem.fractions .< 1)
    verbose && println("reference audit selftest: PASS (labels, quadratic likelihood, " *
        "exact EM step, terminal conditioning)")
    return true
end

"""
Audit the apparent target permutation in `params_model_full.png`. All arms use
the same state observations, from one draw with both state and costate observed.
The Gaussian likelihood probes hold nuisance parameters fixed and are not a
claim of global joint identifiability. Every likelihood reported is the one EM
maximizes, `log p(y | terminal = 0)` (the package's default `condition_terminal`);
the oracle controls also print the joint `log p(y, terminal = 0)` optimum, to
show what scoring the terminal event as an observation would cost.
"""
function experiment_reference_audit(cfg; figures::Bool=true)
    n, T, N = cfg.n, cfg.tsteps, cfg.ntrials
    3 <= n <= 8 || throw(ArgumentError("use quick/default/full tier (3:8 plant dimensions)"))
    rng = MersenneTwister(1)
    full = lqr_truth(; n=n, tsteps=T, nref=4, terminal=true,
        onset=_onset(0, T), observe_costate=true)
    us = target_inputs(rng, 4, N, T)
    ys_full = simulate(rng, full, N; uxs=us)
    state = lqr_truth(; n=n, tsteps=T, nref=4, terminal=true, onset=_onset(0, T))
    ys_state = [y[1:n, :] for y in ys_full]
    section("Reference permutation audit — fixed C, fixed h, four labelled targets")
    println("Paired seed=1; N=$N, T=$T, n=$n; $(cfg.max_iter) reported EM iterations.")
    println("Columns are observed input labels; matching below is diagnostic only.")
    results = Dict{String,Any}()
    for (label, truth, ys, noise) in (
        ("state, tight", state, ys_state, 1e-4),
        ("state, loose", state, ys_state, 5e-2),
        ("state+costate, tight", full, ys_full, 1e-4),
        ("state+costate, loose", full, ys_full, 5e-2),
    )
        initial = fit_model(truth; known_plant=true, free_gref=true, sig0_costate=noise)
        G0 = copy(initial.Gref)
        fit, trace = one_fit(truth, ys, us, initial; free_C=false,
            max_iter=cfg.max_iter, tol=1e-10)
        G, Gtrue = fit.state_model.Gref, truth.sm.Gref
        perm = reference_permutation_audit(G, Gtrue)
        @printf("\n%-23s raw %.4f; movement %.4f; matched %.4f; matched+scale %.4f (×%.3f)\n",
            label, score(G, Gtrue).rmse, reference_movement(G, G0, Gtrue).raw,
            perm.error, perm.scaled, perm.scale)
        println("  fit[:, ", perm.order, "] matches truth order")
        reordered = deepcopy(fit)
        reordered.state_model.Gref .= G[:, perm.order]
        refresh!(reordered.state_model)
        ll = elbo(fit, ys; ux=us)
        ll_permuted = elbo(reordered, ys; ux=us)
        # Relabelling input rows with the same permutation must restore the model.
        relabelled = elbo(reordered, ys; ux=[u[perm.order, :] for u in us])
        @printf("  permutation alone Δll %.3f; permutation + input relabel Δll %.3g\n",
            ll_permuted - ll, relabelled - ll)
        profile = reference_likelihood_profile(fit, ys, us)
        @printf("  fixed-nuisance information rank %d/%d; λmin %.4g; direct G gain %.3f; quadratic residual %.3g\n",
            profile.rank, profile.dimension, minimum(profile.eigenvalues),
            profile.gain, profile.quadratic_residual)
        @printf("  direct G optimum error %.4f (other fitted parameters held fixed)\n",
            score(profile.optimum, Gtrue).rmse)
        em = reference_em_audit(fit, ys, us, profile)
        @printf("  G-only EM step relative error %.3g; fraction corrected per step %.3g … %.3g\n",
            em.step_relative_error, minimum(em.fractions), maximum(em.fractions))
        results[label] = (; fit, G0, permutation=perm, profile, em, trace)
    end
    # Oracle nuisance controls distinguish structural information from a bad
    # joint fit. The truth is used only here, never as a recovery initialization.
    for (label, truth, ys) in (("state", state, ys_state), ("state+costate", full, ys_full))
        oracle = deepcopy(truth.lds)
        oracle.state_model.Gref .= fit_model(truth; known_plant=true, free_gref=true).Gref
        refresh!(oracle.state_model)
        p = reference_likelihood_profile(oracle, ys, us; conditional=true)
        pj = reference_likelihood_profile(oracle, ys, us; conditional=false)
        @printf("\nOracle nuisance, %-15s conditional rank %d/%d; λmin %.4g; G error %.4f; gain %.3f; residual %.3g\n",
            label, p.rank, p.dimension, minimum(p.eigenvalues),
            score(p.optimum, truth.sm.Gref).rmse, p.gain, p.quadratic_residual)
        @printf("  (joint objective instead: G error %.4f)\n",
            score(pj.optimum, truth.sm.Gref).rmse)
        results["oracle " * label] = p
        results["oracle joint " * label] = pj
    end
    if figures
        Gtrue = state.sm.Gref
        tight = results["state, tight"]
        panels = [heat(G; clims=(-1.5, 1.5), title=label) for (label, G) in (
            ("Truth (labelled columns)", Gtrue),
            ("Initialization = truth[:, [2,3,4,1]] / 3", tight.G0),
            ("State observed, tight costate noise", tight.fit.state_model.Gref),
            ("State + costate observed, loose start", results["state+costate, loose"].fit.state_model.Gref),
        )]
        path = save_fig(plot(panels...; layout=(1, 4), size=(1600, 350)), "reference_permutation_audit")
        println("\nFigure: ", path)
    end
    return results
end
