# Gaussian variational smoothing for linear Gaussian states and Gaussian/Poisson
# emissions. No dense (k*T)^2 covariance is formed: precision is block tridiagonal.

function _inference_controls(inference, variational_max_iter, variational_tol)
    return if inference === :laplace
        (;)
    else
        (; inference, variational_max_iter, variational_tol)
    end
end

function _check_inference(inference, max_iter, tol)
    inference in (:laplace, :mean_corrected, :variational) ||
        throw(ArgumentError("inference must be :laplace, :mean_corrected or :variational"))
    max_iter > 0 || throw(ArgumentError("variational_max_iter must be positive"))
    isfinite(tol) && tol > 0 ||
        throw(ArgumentError("variational_tol must be finite and positive"))
    return nothing
end

function _validate_variational_model(lds, inference, max_iter, tol)
    _check_inference(inference, max_iter, tol)
    inference === :laplace && return nothing
    lds.state_model isa GaussianStateModel ||
        throw(ArgumentError("non-Laplace inference currently requires GaussianStateModel"))
    all(
        t -> t isa Union{GaussianObservationModel,PoissonObservationModel},
        if lds.obs_model isa CompositeObservationModel
            values(_models(lds.obs_model))
        else
            (lds.obs_model,)
        end,
    ) || throw(
        ArgumentError("non-Laplace inference supports Gaussian and Poisson emissions only"),
    )
    return nothing
end

_vi_members(om, y, uy) = ((; om, y, uy),)
function _vi_members(om::CompositeObservationModel, y, uy)
    return map(keys(_models(om))) do key
        (;
            om=getproperty(_models(om), key),
            y=getproperty(y, key),
            uy=uy === nothing ? nothing : getproperty(uy, key),
        )
    end
end

# Use the same packed-covariance GEMM as the emission M-step. In particular,
# computing c' P c must not introduce a per-neuron cubic operation.
function _vi_rho!(rho, om, P, sws)
    p, k = size(om.C)
    nt = size(P, 3)
    pb = poisson_batch!(sws, k, p, nt)
    pairs = _poisson_pair_products!(pb, om.C, p)
    packed = _poisson_pack_cov!(pb, P, nt)
    mul!(rho, pairs, packed, eltype(P)(0.5), zero(eltype(P)))
    return rho
end

function _vi_terms(lds, y, uy, nt)
    terms = _vi_members(lds.obs_model, y, uy)
    all(t -> t.om isa Union{PoissonObservationModel,GaussianObservationModel}, terms) ||
        throw(
            ArgumentError(
                "Gaussian variational inference supports Gaussian and Poisson emissions only",
            ),
        )
    return map(filter(t -> t.om isa PoissonObservationModel, terms)) do t
        (;
            t...,
            rho=zeros(eltype(t.om.C), size(t.y)),
            eta=zeros(eltype(t.om.C), size(t.y)),
            rate=zeros(eltype(t.om.C), size(t.y)),
        )
    end
end

function _vi_rates!(term, x)
    mul!(term.eta, term.om.C, x)
    term.eta .+= term.om.d
    isempty(term.om.D) || mul!(term.eta, term.om.D, term.uy, true, true)
    @. term.rate = exp(term.eta + term.rho)
    return term.rate
end

function _vi_rate_adjustment(terms, x)
    value = zero(eltype(x))
    for term in terms
        _vi_rates!(term, x)
        @inbounds for j in eachindex(term.rate)
            value += term.rate[j] - exp(term.eta[j])
        end
    end
    return value
end

function _vi_gradient!(g, sws, lds, x, y, ux, uy, terms)
    gradient!(g, sws, lds, x, y, ux, uy)
    for term in terms
        _vi_rates!(term, x)
        # eta is scratch until the next rate evaluation.
        @. term.eta = term.rate - exp(term.eta)
        mul!(g, term.om.C', term.eta, -one(eltype(g)), one(eltype(g)))
    end
    return g
end

function _vi_precision!(sws, lds, x, y, uy, terms, nt, base)
    for term in terms
        _vi_rates!(term, x)
    end
    return _vi_assemble_precision!(sws, terms, map(term -> term.rate, terms), nt, base)
end

# Write J + Σ C' diag(rate) C into sws.btd.neg_*, for the given per-member rates:
# the current expected rates, or stored sites when warm-starting.
function _vi_assemble_precision!(sws, terms, rates, nt, base)
    for t in 1:nt
        copyto!(sws.btd.neg_diag[t], base[1][t])
        t == nt && continue
        copyto!(sws.btd.neg_sub[t], base[2][t])
        copyto!(sws.btd.neg_super[t], base[3][t])
    end
    for (term, rate) in zip(terms, rates)
        p, k = size(term.om.C)
        pb = poisson_batch!(sws, k, p, nt)
        pairs = _poisson_pair_products!(pb, term.om.C, p)
        nsym = length(pb.sym_i)
        Hsym = view(pb.Hsym, 1:nsym, 1:nt)
        mul!(Hsym, transpose(pairs), rate)
        for t in 1:nt
            Ht = sws.btd.neg_diag[t]
            @inbounds for j in 1:nsym
                a, b = pb.sym_i[j], pb.sym_j[j]
                value = Hsym[j, t]
                Ht[a, b] += value
                a == b || (Ht[b, a] += value)
            end
        end
    end
    return nothing
end

function _vi_mean!(sws, lds, fs, y, ux, uy, terms; max_iter, tol)
    x = fs.x_smooth
    k, nt = size(x)
    g = view(sws.opt.grad_buf, :, 1:nt)
    direction = reshape(view(sws.opt.X0, 1:(k * nt)), k, nt)
    base = _vi_base_precision(sws, lds, x, y, uy)
    iterations = Ref(0)
    lognorm = _emission_lognorm(lds, y)
    function objective()
        return _joint_loglikelihood_total(sws, lds, x, y, lognorm, ux, uy) -
               _vi_rate_adjustment(terms, x)
    end
    grad! = (out, cur) -> begin
        iterations[] += 1
        _vi_gradient!(out, sws, lds, cur, y, ux, uy, terms)
    end
    hess! = cur -> _vi_precision!(sws, lds, cur, y, uy, terms, nt, base)
    function solve!(p, g)
        block_tridiagonal_solve_spd!(
            vec(p),
            view(sws.btd.neg_sub, 1:(nt - 1)),
            view(sws.btd.neg_diag, 1:nt),
            view(sws.btd.neg_super, 1:(nt - 1)),
            vec(g),
            sws.btd,
        )
        return nothing
    end
    initial = objective()
    newton_smooth!(
        Val(:max),
        x,
        g,
        direction,
        grad!,
        hess!,
        solve!,
        objective,
        BackTrackingLS{eltype(x)}();
        max_iter,
        tol=tol^2,
    )
    # Report a scale-aware stationarity criterion, not a line-search stop flag.
    grad!(g, x)
    hess!(x)
    solve!(direction, g)
    residual = sqrt(max(zero(eltype(x)), dot(g, direction)) / (k * nt))
    return (;
        converged=residual <= tol,
        iterations=iterations[] - 1,
        mean_residual=residual,
        initial_mean_objective=initial,
        final_mean_objective=objective(),
    )
end

# Base Gaussian precision J: state prior plus Gaussian observation factors.
# Obtain it directly, avoiding subtraction of large Poisson curvatures.
function _vi_base_precision(sws, lds, x, y, uy)
    nt = size(x, 2)
    _state_hessian_blocks!(sws.btd, sws.consts, lds.state_model, nt)
    for t in _vi_members(lds.obs_model, y, uy)
        t.om isa GaussianObservationModel || continue
        H = t.om.C' * (t.om.R \ t.om.C)
        for j in 1:nt
            sws.btd.H_diag[j] .-= H
        end
    end
    # GaussianStateModel and Gaussian emissions are time invariant. Reuse the
    # interior/off-diagonal matrices instead of allocating 3*T dense blocks.
    diag = fill(-sws.btd.H_diag[min(2, nt)], nt)
    diag[1] = -sws.btd.H_diag[1]
    diag[nt] = -sws.btd.H_diag[nt]
    return diag, fill(-sws.btd.H_sub[1], nt - 1), fill(-sws.btd.H_super[1], nt - 1)
end

# `lognorm` is the data-only emission normalizer, hoisted out of the line search.
function _vi_bound(sws, lds, x, y, ux, uy, terms, P, Plag, entropy, base, lognorm)
    diag, sub, super = base
    trace_JS = zero(eltype(x))
    for t in axes(x, 2)
        trace_JS += dot(diag[t], view(P, :, :, t))
        t == 1 && continue
        trace_JS += 2dot(sub[t - 1], view(Plag, :, :, t))
    end
    return _joint_loglikelihood_total(sws, lds, x, y, lognorm, ux, uy) -
           _vi_rate_adjustment(terms, x) - trace_JS / 2 + entropy
end

#=
Σₜ tr((Sₜₜ Δₜ)²) with Δₜ = targetₜ − precisionₜ: the precision mismatch seen
through the current covariance. Unlike a raw block norm it is unchanged by a
linear latent map (S → TST', Δ → T⁻ᵀΔT⁻¹), so whitening cannot change when a
trial stops. A quarter of it approximates the bound gain still available from
the covariance (the Gaussian KL expanded to second order, per time block).
=#
function _vi_precision_mismatch(P, precision, target, vb, nt)
    Δ = vb.delta
    W = vb.work
    k = size(W, 1)
    total = zero(eltype(W))
    for t in 1:nt
        @. Δ = target[t] - precision[t]
        mul!(W, view(P, :, :, t), Δ)
        @inbounds for j in 1:k, i in 1:k
            total += W[i, j] * W[j, i]
        end
    end
    return max(total, zero(total))
end

#=
Whether a variational solve can start from the previous E-step's posterior
instead of a fresh Laplace solve: the trial must carry finite expected rates
of the right shape for every Poisson member, and a finite mean.
=#
function _vi_can_warm_start(lds, fs, y)
    sites = fs.vi_sites
    isempty(sites) && return false
    all(isfinite, fs.x_smooth) || return false
    i = 0
    for member in _vi_members(lds.obs_model, y, nothing)
        member.om isa PoissonObservationModel || continue
        i += 1
        i <= length(sites) || return false
        size(sites[i]) == size(member.y) || return false
        all(r -> isfinite(r) && r >= 0, sites[i]) || return false
    end
    return i == length(sites)
end

# Keep the expected rates at the returned posterior to seed the next E-step.
function _vi_store_sites!(fs, terms, x)
    sites = fs.vi_sites
    if length(sites) != length(terms) ||
        any(size(s) != size(t.rate) for (s, t) in zip(sites, terms))
        empty!(sites)
        foreach(t -> push!(sites, similar(t.rate)), terms)
    end
    for (site, term) in zip(sites, terms)
        copyto!(site, _vi_rates!(term, x))
    end
    return sites
end

# Each rejected step costs a factorization and selected inverse; halving ten
# times reaches a step of about 1e-3, below which stalls are numerical.
const _VI_MAX_BACKTRACKS = 10

function _vi_store_covariances!(fs, P, Plag, entropy)
    nt = size(P, 3)
    fs.entropy = entropy
    if is_compact(fs)
        _pack_covariances!(fs.p_packed, P, nt)
        fill!(fs.xcov_sum, zero(eltype(P)))
        for t in 2:nt
            fs.xcov_sum .+= view(Plag, :, :, t)
        end
    else
        copyto!(fs.p_smooth, P)
        copyto!(fs.p_smooth_tt1, Plag)
    end
    return fs
end

"""
Refine a Gaussian posterior. Mean correction holds the Laplace S fixed. Full VI
uses damped natural updates of both mean and block-tridiagonal precision; every
accepted update increases the actual Gaussian ELBO (SPD alone is not an
acceptance rule).

`warm=false` refines the Laplace Gaussian just computed by `smooth!`.
`warm=true` (full VI only) instead starts from the previous E-step's
posterior: its mean, already carried into the current coordinate frame, and
its expected rates `fs.vi_sites`, giving precision `J + Σ C' diag(sites) C`
under the current parameters. Near EM convergence that is within a step or two
of the optimum, and it skips the Laplace solve altogether.
"""
function _gaussian_variational!(
    lds, fs, y, sws, ux, uy; inference, max_iter, tol, warm::Bool=false
)
    _check_inference(inference, max_iter, tol)
    empty!(fs.inference_stats)
    inference === :laplace && return fs
    lds.state_model isa GaussianStateModel || throw(
        ArgumentError(
            "Gaussian variational smoothing currently requires GaussianStateModel"
        ),
    )
    warm &&
        inference !== :variational &&
        throw(ArgumentError("only full Gaussian VI can warm-start without a Laplace solve"))
    x = fs.x_smooth
    k, nt = size(x)
    terms = _vi_terms(lds, y, uy, nt)
    isempty(terms) && return fs # the Gaussian solution is already exact
    P = is_compact(fs) ? view(sws.agg.p_smooth_shared, :, :, 1:nt) : fs.p_smooth
    Plag = is_compact(fs) ? view(sws.agg.p_smooth_tt1_shared, :, :, 1:nt) : fs.p_smooth_tt1
    fs.inference_stats["method"] = String(inference)
    if inference === :mean_corrected
        for term in terms
            _vi_rho!(term.rho, term.om, P, sws)
        end
        result = _vi_mean!(sws, lds, fs, y, ux, uy, terms; max_iter, tol)
        for (key, value) in pairs(result)
            fs.inference_stats[String(key)] = value
        end
        result.converged ||
            @warn "Gaussian mean correction reached its stationarity/line-search limit" max_iter residual =
                result.mean_residual maxlog = 3
        return fs
    end
    return _vi_optimize!(lds, fs, y, sws, ux, uy, terms, P, Plag; max_iter, tol, warm)
end

#=
Full Gaussian VI for one trial. The cost is dominated by block-tridiagonal
factorizations, so each candidate is factorized exactly once:
`block_tridiagonal_inverse_logdet!` yields its covariance, lag blocks and log
determinant, and caches the Cholesky factors that `block_tridiagonal_backsubst!`
then reuses for the mean step and, once the candidate is accepted, for the next
iteration's stationarity residual. `factored` records whether that cache holds
the accepted precision; only the Laplace start lacks it.
=#
function _vi_optimize!(lds, fs, y, sws, ux, uy, terms, P, Plag; max_iter, tol, warm)
    x = fs.x_smooth
    k, nt = size(x)
    R = eltype(x)
    n = k * nt
    btd = sws.btd
    vb = vi_buffers!(sws, k, nt)
    precision = view(vb.precision, 1:nt)
    target = view(vb.target, 1:nt)
    candidateP = view(vb.P, :, :, 1:nt)
    candidateLag = view(vb.Plag, :, :, 1:nt)
    oldx = view(vb.oldx, :, 1:nt)
    g = view(vb.g, :, 1:nt)
    direction = view(vb.direction, :, 1:nt)
    diagonal = view(btd.neg_diag, 1:nt)
    lognorm = _emission_lognorm(lds, y)
    base = _vi_base_precision(sws, lds, x, y, uy)
    sub, super = base[2], base[3]
    if warm
        _vi_assemble_precision!(sws, terms, fs.vi_sites, nt, base)
        ld = block_tridiagonal_inverse_logdet!(P, Plag, sub, diagonal, super, btd)
        entropy = gaussian_entropy_from_logdet(ld, n)
        factored = true
    else
        # The Laplace precision is available cheaply from the mode Hessian, and
        # smooth! has just written its covariance and entropy.
        hessian!(sws, lds, x, y, uy)
        _negate_blocks!(btd, nt)
        entropy = fs.entropy
        factored = false
    end
    for t in 1:nt
        copyto!(precision[t], btd.neg_diag[t])
    end
    for term in terms
        _vi_rho!(term.rho, term.om, P, sws)
    end
    value = _vi_bound(sws, lds, x, y, ux, uy, terms, P, Plag, entropy, base, lognorm)
    stats = fs.inference_stats
    stats["warm_start"] = warm
    stats["initial_elbo"] = value
    accepted_elbos = [value]
    stats["accepted_elbos"] = accepted_elbos
    converged = false
    stalled = false
    for iter in 1:max_iter
        _vi_gradient!(g, sws, lds, x, y, ux, uy, terms)
        _vi_precision!(sws, lds, x, y, uy, terms, nt, base)
        for t in 1:nt
            copyto!(target[t], btd.neg_diag[t])
        end
        # Stationarity, as two coordinate-free per-dimension residuals: the
        # covariance-scaled mean score and the precision mismatch.
        if factored
            block_tridiagonal_backsubst!(vec(direction), sub, vec(g), btd, nt)
        else
            block_tridiagonal_solve_spd!(vec(direction), sub, precision, super, vec(g), btd)
        end
        score = max(zero(R), dot(g, direction))
        mismatch = _vi_precision_mismatch(P, precision, target, vb, nt)
        mean_error = sqrt(score / n)
        precision_error = sqrt(mismatch / n)
        predicted_gain = score / 2 + mismatch / 4
        stats["iterations"] = iter
        stats["mean_residual"] = mean_error
        stats["precision_residual"] = precision_error
        stats["predicted_gain"] = predicted_gain
        if max(mean_error, precision_error) <= tol
            converged = true
            break
        end
        copyto!(oldx, x)
        accepted = false
        step = one(R)
        for _ in 1:_VI_MAX_BACKTRACKS
            for t in 1:nt
                @. btd.neg_diag[t] = (1 - step) * precision[t] + step * target[t]
            end
            ld = block_tridiagonal_inverse_logdet!(
                candidateP, candidateLag, sub, diagonal, super, btd
            )
            # Natural step for the mean, against the candidate precision.
            block_tridiagonal_backsubst!(vec(direction), sub, vec(g), btd, nt)
            @. x = oldx + step * direction
            candidate_entropy = gaussian_entropy_from_logdet(ld, n)
            for term in terms
                _vi_rho!(term.rho, term.om, candidateP, sws)
            end
            candidate_value = _vi_bound(
                sws,
                lds,
                x,
                y,
                ux,
                uy,
                terms,
                candidateP,
                candidateLag,
                candidate_entropy,
                base,
                lognorm,
            )
            if isfinite(candidate_value) && candidate_value >= value
                for t in 1:nt
                    copyto!(precision[t], btd.neg_diag[t])
                end
                copyto!(P, candidateP)
                copyto!(Plag, candidateLag)
                entropy = candidate_entropy
                value = candidate_value
                push!(accepted_elbos, value)
                factored = true # the cached factors are this accepted precision's
                accepted = true
                break
            end
            step /= 2
        end
        if !accepted
            copyto!(x, oldx)
            # Restore rho for the accepted covariance, including on return.
            for term in terms
                _vi_rho!(term.rho, term.om, P, sws)
            end
            stalled = true
            #=
            Bound differences below its rounding scale cannot be resolved by a
            line search. Call the trial stationary only when the gain a full
            step predicts is itself below that scale, never on a stall alone.
            The prediction's covariance term uses only diagonal blocks of S, so
            with strongly autocorrelated latents it can understate the gain;
            this flag is a diagnostic and does not shorten the search above.
            =#
            converged = predicted_gain <= sqrt(eps(R)) * max(one(R), abs(value))
            break
        end
    end
    converged ||
        @warn "Gaussian VI stopped before its stationarity tolerance" max_iter tol stalled maxlog =
            3
    stats["converged"] = converged
    stats["stalled"] = stalled
    stats["final_elbo"] = value
    _vi_store_sites!(fs, terms, x)
    return _vi_store_covariances!(fs, P, Plag, entropy)
end
