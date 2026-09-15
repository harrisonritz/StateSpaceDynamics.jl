#=============================================================================
Smoulder-scale Poisson recovery.

This deliberately contains no session labels, observation variants, or
stitching.  The only known per-trial label is reward, and only `Qc` depends on
it.  Consequently every reward shares one plant, control matrix, reference
map, innovation covariance, and Poisson emission.

The plant dimension is `n`; the smoother latent is the 2n-dimensional
state-costate pair.  The `:smoulder` tier uses n=12, 1000 trials, 100 bins, 150
Poisson channels, eight ring targets, and three rewards.
=============================================================================#

"""Dimensions for the Poisson suite, with small fallbacks for the old tiers."""
function smoulder_dimensions(cfg)
    p = hasproperty(cfg, :obs_dim) ? cfg.obs_dim : max(12, 4cfg.n)
    nr = hasproperty(cfg, :rewards) ? cfg.rewards : 3
    return (n=cfg.n, tsteps=cfg.tsteps, ntrials=cfg.ntrials, obs_dim=p,
            rewards=nr, seeds=cfg.seeds, max_iter=cfg.max_iter)
end

"""Balanced known reward labels and target identities (both trial-level)."""
function smoulder_design(ntrials::Int, tsteps::Int; nrewards::Int=3, ntargets::Int=8)
    rewards = [mod1(i, nrewards) for i in 1:ntrials]
    targets = [mod1(i, ntargets) for i in 1:ntrials]
    uxs = [begin
        u = zeros(ntargets, tsteps)
        u[targets[i], :] .= 1
        u
    end for i in 1:ntrials]
    return rewards, targets, uxs
end

"""A stable random orthogonal matrix, used only to audit gauge recovery."""
function random_orthogonal(rng::AbstractRNG, n::Int)
    F = qr(randn(rng, n, n))
    W = Matrix(F.Q)
    det(W) < 0 && (W[:, 1] .*= -1)
    return W
end

"""
    orthogonal_alignment(Cfit, Cref, n) -> T

Find the state-coordinate map `x_ref = T*x_fit` from the emissions.  Since
`Cfit ≈ Cref*T`, this is the orthogonal Procrustes solution to
`min_T ||Cfit - Cref*T||`.  Costate coordinates transform covariantly under an
orthogonal change of state basis, so the same `T` applies to both halves.
"""
function orthogonal_alignment(Cfit::AbstractMatrix, Cref::AbstractMatrix, n::Int)
    F = svd(view(Cref, :, 1:n)' * view(Cfit, :, 1:n))
    return F.U * F.Vt
end

"""Unconstrained state-coordinate map, for testing whether rotation is enough."""
function linear_alignment(Cfit::AbstractMatrix, Cref::AbstractMatrix, n::Int)
    return Matrix(view(Cref, :, 1:n)) \ Matrix(view(Cfit, :, 1:n))
end

function transformed_scores(fs, rs, Cf, Cr, rewards, T)
    invT = inv(T)
    fv = [group_variant(fs, :Qc, r) for r in rewards]
    rv = [group_variant(rs, :Qc, r) for r in rewards]
    c = plant_dim(fs) / tr(invT' * fv[1].Qc[1] * invT)
    return (
        Qc=score_worst(
            [c .* (invT' * v.Qc[1] * invT) for v in fv], [v.Qc[1] for v in rv]; sym=true
        ),
        Qterm=score_worst(
            [c .* (invT' * v.Qc[end] * invT) for v in fv], [v.Qc[end] for v in rv]; sym=true
        ),
        A=score(T*fs.A*invT, rs.A),
        S=score((T*fs.S*T') ./ c, rs.S; sym=true),
        Gref=score(T*fs.Gref, rs.Gref),
        cl=score_worst(
            [T*closed_loop_dynamics(v)*invT for v in fv],
            [closed_loop_dynamics(v) for v in rv],
        ),
        C=score(view(Cf, :, 1:size(T, 1))*invT, view(Cr, :, 1:size(T, 1))),
    )
end

"""Score all parameters after one consistent state-coordinate transform."""
function smoulder_scores(fit_lds, ref_lds, rewards)
    fit = deepcopy(fit_lds)
    ref = deepcopy(ref_lds)
    rescale_costate!(fit.state_model; target=:trace)
    rescale_costate!(ref.state_model; target=:trace)
    fs, rs = fit.state_model, ref.state_model
    n = plant_dim(fs)
    T = orthogonal_alignment(fit.obs_model.C, ref.obs_model.C, n)
    L = linear_alignment(fit.obs_model.C, ref.obs_model.C, n)

    fv = [group_variant(fs, :Qc, r) for r in rewards]
    rv = [group_variant(rs, :Qc, r) for r in rewards]
    raw_run = score_worst([v.Qc[1] for v in fv], [v.Qc[1] for v in rv]; sym=true)
    raw_term = score_worst([v.Qc[end] for v in fv], [v.Qc[end] for v in rv]; sym=true)
    raw_cl = score_worst(
        [closed_loop_dynamics(v) for v in fv],
        [closed_loop_dynamics(v) for v in rv],
    )
    Gf, Gr = fs.Gref, rs.Gref
    return (
        raw=(
            Qc=raw_run, Qterm=raw_term, A=score(fs.A, rs.A),
            S=score(fs.S, rs.S; sym=true), Gref=score(Gf, Gr), cl=raw_cl,
            C=score(view(fit.obs_model.C, :, 1:n), view(ref.obs_model.C, :, 1:n)),
        ),
        aligned=transformed_scores(fs, rs, fit.obs_model.C, ref.obs_model.C, rewards, T),
        linear=transformed_scores(fs, rs, fit.obs_model.C, ref.obs_model.C, rewards, L),
        # Pairwise target geometry is invariant to any orthogonal gauge.
        Ggram=score(Gf'Gf, Gr'Gr; sym=true),
        rotation=T,
        linear_map=L,
        nonorthogonality=norm(L'L - I) / sqrt(n),
    )
end

"""Poisson loadings with realistic low rates and no costate readout."""
function smoulder_emission(rng::AbstractRNG, obs_dim::Int, n::Int)
    C = zeros(obs_dim, 2n)
    C[:, 1:n] .= 0.35 .* randn(rng, obs_dim, n) ./ sqrt(n)
    baserate = exp.(range(log(0.15), log(1.2); length=obs_dim))
    d = log.(baserate)
    return C, d
end

"""Reward-specific running and terminal costs; both vary with reward."""
function reward_costs(n::Int, nrewards::Int)
    base = Matrix(0.20I, n, n)
    for i in 1:(n - 1)
        base[i, i + 1] = base[i + 1, i] = 0.025
    end
    scales = collect(range(0.6, 1.4; length=nrewards))
    return [[scales[r] .* base, scales[r] .* (3.0I(n))] for r in 1:nrewards]
end

"""Construct the grouped Poisson truth and materialize its reward variants."""
function smoulder_truth(; n::Int, tsteps::Int, ntrials::Int, obs_dim::Int,
                         nrewards::Int=3, ntargets::Int=8, seed::Int=1)
    rng = MersenneTwister(seed)
    rewards, targets, uxs = smoulder_design(
        ntrials, tsteps; nrewards=nrewards, ntargets=ntargets
    )
    A, S = plant(n)
    costs = reward_costs(n, nrewards)
    schedule = fill(1, tsteps)
    schedule[end] = 2
    sm = LQRStateModel(
        A, S, deepcopy(costs[1]), mixed_noise(n; state=0.02, costate=1e-4);
        schedule=schedule, terminal=true, Σf=Matrix(0.02I, n, n),
        P0=Matrix(0.2I, 2n, 2n), Bu=zeros(2n, ntargets),
        Gref=ring_map(n, ntargets), observe_costate=false,
    )
    set_depends_on!(sm, (Qc=rewards,))
    C, d = smoulder_emission(rng, obs_dim, n)
    lds = LinearDynamicalSystem(sm, PoissonObservationModel(C, d))
    for r in 1:nrewards
        v = group_variant(sm, :Qc, r)
        for k in eachindex(v.Qc)
            v.Qc[k] .= costs[r][k]
        end
        refresh!(v)
    end
    refresh!(sm)
    return (lds=lds, rewards=rewards, targets=targets, uxs=uxs, costs=costs)
end

function poisson_observations(rng::AbstractRNG, C, d, z)
    η = C * z .+ d
    rates = exp.(clamp.(η, -12.0, 8.0))
    y = Matrix{Float64}(undef, size(rates))
    for j in eachindex(rates)
        y[j] = rand(rng, Poisson(rates[j]))
    end
    return y
end

"""Generate optimal reaches from the reward-appropriate control problem."""
function simulate_smoulder_lqr(rng::AbstractRNG, truth; slack::Float64=0.05)
    sm = truth.lds.state_model
    C, d = truth.lds.obs_model.C, truth.lds.obs_model.d
    N, T = length(truth.rewards), size(truth.uxs[1], 2)
    xs = Vector{Matrix{Float64}}(undef, N)
    ys = Vector{Matrix{Float64}}(undef, N)
    for i in 1:N
        v = group_variant(sm, :Qc, truth.rewards[i])
        xs[i] = simulate_lqr(
            rng, v, T; costate_slack=slack, process_noise=true, ux=truth.uxs[i]
        )
        ys[i] = poisson_observations(rng, C, d, xs[i])
    end
    return xs, ys
end

# Recovery cells with the same seed must see the same synthetic dataset.  At
# the full tier this also avoids redrawing and retaining 15 million counts for
# every initialization/prior condition.
const SMOULDER_DATA_CACHE = Dict{Tuple,Any}()

function smoulder_lqr_dataset(; n, tsteps, ntrials, obs_dim, nrewards, seed)
    key = (:lqr, n, tsteps, ntrials, obs_dim, nrewards, seed)
    return get!(SMOULDER_DATA_CACHE, key) do
        truth = smoulder_truth(; n=n, tsteps=tsteps, ntrials=ntrials,
            obs_dim=obs_dim, nrewards=nrewards, seed=seed)
        _, ys = simulate_smoulder_lqr(MersenneTwister(100seed), truth)
        (truth=truth, ys=ys)
    end
end

"""A memory-light PCA/FA-style loading initialization from Poisson counts."""
function count_pca_initialization(ys, n::Int)
    p = size(first(ys), 1)
    total = sum(size(y, 2) for y in ys)
    μ = zeros(p)
    for y in ys
        μ .+= vec(sum(y; dims=2))
    end
    μ ./= total
    d = log.(max.(μ, 0.05))
    S = zeros(p, p)
    for y in ys
        z = sqrt.(y .+ 3 / 8)
        z .-= vec(sum(z; dims=2)) ./ size(z, 2)
        mul!(S, z, z', 1.0, 1.0)
    end
    E = eigen(Symmetric(S))
    ord = sortperm(E.values; rev=true)[1:n]
    Cstate = E.vectors[:, ord]
    Cstate .*= 0.25
    return Cstate, d
end

function smoulder_fit_model(truth, ys; emission_init::Symbol=:pca,
        known_plant::Bool=false, q0::Float64=0.2, sig0_costate::Float64=1e-4,
        sigma_prior_strength::Float64=0.0, qc_prior_strength::Float64=0.0,
        fit_emission::Bool=true, seed::Int=1)
    rng = MersenneTwister(10_000 + seed)
    ref = truth.lds.state_model
    n, nr = plant_dim(ref), length(unique(truth.rewards))
    ntargets = size(ref.Gref, 2)
    A0 = known_plant ? copy(ref.A) : Matrix(0.92I, n, n)
    S0 = known_plant ? copy(ref.S) : Matrix(0.05I, n, n)
    Q0 = [Matrix(q0*I, n, n), Matrix(3q0*I, n, n)]
    sm = LQRStateModel(
        A0, S0, Q0, mixed_noise(n; state=0.05, costate=sig0_costate);
        schedule=copy(ref.schedule), terminal=true, Σf=copy(ref.Σf),
        P0=Matrix(0.2I, 2n, 2n), Bu=zeros(2n, ntargets),
        Gref=0.35 .* ring_map(n, ntargets), observe_costate=false,
        fit_flags=LQRFitFlags(A=!known_plant, S=!known_plant, Bu=false, Gref=true),
    )
    set_depends_on!(sm, (Qc=truth.rewards,))
    sm.Σ_prior = sigma_prior(
        n; state=0.02, costate=1e-4, strength=sigma_prior_strength
    )
    sm.Qc_prior = qc_prior(n; scale=[q0, 3q0], strength=qc_prior_strength)
    for r in 1:nr
        v = group_variant(sm, :Qc, r)
        for Q in v.Qc
            Q .= Matrix(q0*I, n, n)
        end
        v.Qc[end] .= Matrix(3q0*I, n, n)
        refresh!(v)
    end

    Cstate, d0 = if emission_init === :truth
        (copy(view(truth.lds.obs_model.C, :, 1:n)), copy(truth.lds.obs_model.d))
    elseif emission_init === :rotated
        W = random_orthogonal(rng, n)
        (Matrix(view(truth.lds.obs_model.C, :, 1:n)) * W, copy(truth.lds.obs_model.d))
    elseif emission_init === :random
        (0.08 .* randn(rng, size(truth.lds.obs_model.C, 1), n),
         copy(truth.lds.obs_model.d))
    elseif emission_init === :pca
        count_pca_initialization(ys, n)
    else
        throw(ArgumentError("unknown emission initialization :$emission_init"))
    end
    C0 = zeros(size(Cstate, 1), 2n)
    C0[:, 1:n] .= Cstate
    lds = LinearDynamicalSystem(sm, PoissonObservationModel(C0, d0))
    lds.fit_bool[5] = fit_emission
    return lds
end

function recover_smoulder_lqr(; n::Int=12, tsteps::Int=100, ntrials::Int=1000,
        obs_dim::Int=150, nrewards::Int=3, max_iter::Int=100,
        emission_init::Symbol=:pca, known_plant::Bool=false, q0::Float64=0.2,
        sig0_costate::Float64=1e-4, sigma_prior_strength::Float64=0.0,
        qc_prior_strength::Float64=0.0, fit_emission::Bool=true, seed::Int=1)
    data = smoulder_lqr_dataset(; n=n, tsteps=tsteps, ntrials=ntrials,
        obs_dim=obs_dim, nrewards=nrewards, seed=seed)
    truth, ys = data.truth, data.ys
    fit = smoulder_fit_model(truth, ys; emission_init=emission_init,
        known_plant=known_plant, q0=q0, sig0_costate=sig0_costate,
        sigma_prior_strength=sigma_prior_strength,
        qc_prior_strength=qc_prior_strength, fit_emission=fit_emission, seed=seed)
    trace = fit!(fit, ys; ux=truth.uxs, max_iter=max_iter, tol=1e-5,
        newton_max_iter=10, newton_tol=1e-5, progress=false)
    return (scores=smoulder_scores(fit, truth.lds, 1:nrewards),
            elbos=collect(trace), fit=fit, truth=truth.lds,
            rewards=truth.rewards)
end

function print_smoulder_header()
    @printf("%-30s %8s %8s %8s %8s %8s %8s %8s\n",
        "condition", "Q raw", "Q proc", "Q linear", "Gr raw", "Gr proc",
        "Gr linear", "G'G")
    println("-"^98)
end

function print_smoulder_row(label, r)
    s = r.scores
    @printf("%-30s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n", label,
        s.raw.Qc.rmse, s.aligned.Qc.rmse, s.linear.Qc.rmse, s.raw.Gref.rmse,
        s.aligned.Gref.rmse, s.linear.Gref.rmse, s.Ggram.rmse)
end

function aggregate_smoulder(rs)
    good = filter(!isnothing, rs)
    isempty(good) && return nothing
    med(f) = center([Float64(f(r)) for r in good])[1]
    pair(which, block) = (
        rmse=med(r -> getproperty(getproperty(r.scores, which), block).rmse),
        corr=med(r -> getproperty(getproperty(r.scores, which), block).corr),
    )
    blocks = keys(good[1].scores.raw)
    raw = NamedTuple{blocks}(map(b -> pair(:raw, b), blocks))
    aligned = NamedTuple{blocks}(map(b -> pair(:aligned, b), blocks))
    linear = NamedTuple{blocks}(map(b -> pair(:linear, b), blocks))
    return (scores=(raw=raw, aligned=aligned, linear=linear,
        Ggram=(rmse=med(r -> r.scores.Ggram.rmse),
               corr=med(r -> r.scores.Ggram.corr))),)
end

"""Initialization and prior sweep on known reward-dependent costs."""
function experiment_smoulder_lqr(cfg; figures::Bool=true)
    c = smoulder_dimensions(cfg)
    transitions = c.ntrials * (c.tsteps - 1)
    conditions = [
        ("PCA baseline", (;)),
        ("PCA, q0=0.05", (; q0=0.05)),
        ("PCA, q0=0.8", (; q0=0.8)),
        ("PCA, Sigma_ll=1e-5", (; sig0_costate=1e-5)),
        ("PCA, Sigma_ll=1e-3", (; sig0_costate=1e-3)),
        ("PCA, Sigma_ll=1e-2", (; sig0_costate=1e-2)),
        ("Sigma prior 1% transitions", (; sigma_prior_strength=0.01transitions)),
        ("Sigma prior 10% transitions", (; sigma_prior_strength=0.10transitions)),
        ("Qc prior 1% transitions", (; qc_prior_strength=0.01transitions)),
        ("Sigma+Qc priors (10%,1%)", (;
            sigma_prior_strength=0.10transitions,
            qc_prior_strength=0.01transitions,
        )),
        ("oracle C init (fitted)", (; emission_init=:truth)),
        ("oracle C fixed", (; emission_init=:truth, fit_emission=false)),
        ("oracle plant", (; known_plant=true)),
    ]
    base = (; n=c.n, tsteps=c.tsteps, ntrials=c.ntrials, obs_dim=c.obs_dim,
        nrewards=c.rewards, max_iter=c.max_iter)
    res = cells(recover_smoulder_lqr,
        [lab => (; base..., kw...) for (lab, kw) in conditions]; seeds=c.seeds)
    section("6 — Smoulder-scale grouped Poisson LQR: initialization and priors" *
        "\n     (plant $(c.n), latent $(2c.n), $(c.obs_dim) neurons, " *
        "$(c.ntrials) trials x $(c.tsteps), $(c.rewards) known rewards)")
    print_smoulder_header()
    for (lab, _) in conditions
        a = aggregate_smoulder(res[lab])
        a === nothing || print_smoulder_row(lab, a)
    end
    return res
end

"""Exact gauge audit plus fitted controls for the Gref rotation question."""
function experiment_smoulder_gref(cfg; figures::Bool=true)
    c = smoulder_dimensions(cfg)
    truth = smoulder_truth(; n=c.n, tsteps=c.tsteps, ntrials=max(c.rewards, 8),
        obs_dim=c.obs_dim, nrewards=c.rewards, seed=91)
    sm, C = truth.lds.state_model, truth.lds.obs_model.C
    W = random_orthogonal(MersenneTwister(92), c.n)
    Crot = copy(C); Crot[:, 1:c.n] .= C[:, 1:c.n] * W
    # A rotated representation of the exact same control problem.
    fake = deepcopy(truth.lds)
    fake.obs_model.C .= Crot
    fsm = fake.state_model
    fsm.A .= W' * sm.A * W
    fsm.S .= W' * sm.S * W
    fsm.Gref .= W' * sm.Gref
    for r in 1:c.rewards
        fv, rv = group_variant(fsm, :Qc, r), group_variant(sm, :Qc, r)
        for k in eachindex(fv.Qc)
            fv.Qc[k] .= W' * rv.Qc[k] * W
        end
        refresh!(fv)
    end
    refresh!(fsm)
    exact = smoulder_scores(fake, truth.lds, 1:c.rewards)

    section("6b — Gref gauge audit")
    println("An exact rotation leaves the model unchanged but makes raw parameters disagree:")
    print_smoulder_header()
    print_smoulder_row("exact rotated representation", (scores=exact,))
    @printf("   emission Procrustes residual: %.3e\n", exact.aligned.C.rmse)
    @printf("   full-linear residual: %.3e; non-orthogonality: %.3e\n",
        exact.linear.C.rmse, exact.nonorthogonality)
    println("   A, S, every reward Qc, closed loop, and Gref are transformed together.")

    # On the exact tier these controls are intentionally separate jobs: the
    # LQR sweep already contains the corresponding fitted rows. On smaller
    # tiers run them here so `--quick --only=smoulder-gref` is a useful test.
    if c.ntrials <= 400
        base = (; n=c.n, tsteps=c.tsteps, ntrials=c.ntrials, obs_dim=c.obs_dim,
            nrewards=c.rewards, max_iter=c.max_iter)
        for (lab, kw) in [
            ("PCA/free basis", (;)),
            ("truth C, fitted", (; emission_init=:truth)),
            ("truth C, fixed", (; emission_init=:truth, fit_emission=false)),
            ("known plant", (; known_plant=true)),
        ]
            r = recover_smoulder_lqr(; base..., kw..., seed=1)
            print_smoulder_row(lab, r)
        end
    else
        println("   Fitted controls are reported by `smoulder-lqr`; not duplicated at full scale.")
    end
    return exact
end

# ---------------------------------------------------------------------------
# Two-state free -> controlled generator and recovery
# ---------------------------------------------------------------------------

function smoulder_segment(sm, len::Int)
    full = isempty(sm.schedule) ? fill(1, len) : sm.schedule
    sched = fill(1, len); sched[end] = length(sm.Qc)
    return LQRStateModel(copy(sm.A), copy(sm.S), [copy(Q) for Q in sm.Qc], copy(sm.Σ);
        schedule=sched, terminal=sm.terminal, Σf=copy(sm.Σf), P0=copy(sm.P0),
        h=copy(sm.h), Bu=copy(sm.Bu), Gref=copy(sm.Gref), observe_costate=false)
end

function simulate_smoulder_slqr(rng::AbstractRNG, truth; slack::Float64=0.05)
    lqr = truth.lds.state_model
    n, N, T = plant_dim(lqr), length(truth.rewards), size(truth.uxs[1], 2)
    Mfree, Qfree = free_drift(n; decay=0.96, costate_decay=0.7, noise=0.08)
    Lfree = cholesky(Symmetric(Qfree)).L
    C, d = truth.lds.obs_model.C, truth.lds.obs_model.d
    ys, xs, zs = Vector{Matrix{Float64}}(undef, N),
                 Vector{Matrix{Float64}}(undef, N), Vector{Vector{Int}}(undef, N)
    for i in 1:N
        onset = max(3, round(Int, T*(0.35 + 0.20*mod(i, 7)/6)))
        z = zeros(2n, T)
        z[:, 1] .= 0.2 .* randn(rng, 2n)
        for t in 2:onset
            z[:, t] .= Mfree*z[:, t-1] .+ Lfree*randn(rng, 2n)
        end
        v = group_variant(lqr, :Qc, truth.rewards[i])
        seg = smoulder_segment(v, T-onset+1)
        roll = simulate_lqr(rng, seg, T-onset+1; x1=Vector(z[1:n, onset]),
            costate_slack=slack, process_noise=true, ux=truth.uxs[i][:, onset:T])
        z[:, onset:T] .= roll
        xs[i] = z
        zs[i] = [t <= onset ? FREE_STATE : LQR_STATE for t in 1:T]
        ys[i] = poisson_observations(rng, C, d, z)
    end
    return ys, xs, zs
end

function smoulder_slqr_dataset(; n, tsteps, ntrials, obs_dim, nrewards, seed)
    key = (:slqr, n, tsteps, ntrials, obs_dim, nrewards, seed)
    return get!(SMOULDER_DATA_CACHE, key) do
        truth = smoulder_truth(; n=n, tsteps=tsteps, ntrials=ntrials,
            obs_dim=obs_dim, nrewards=nrewards, seed=seed)
        ys, _, zs = simulate_smoulder_slqr(MersenneTwister(200seed), truth)
        (truth=truth, ys=ys, zs=zs)
    end
end

function smoulder_slqr_fit(truth, ys; sig0_costate::Float64=2e-2,
        sigma_prior_strength::Float64=0.0, qc_prior_strength::Float64=0.0,
        fit_noise::Bool=true, init::Symbol=:pca)
    base = smoulder_fit_model(truth, ys; emission_init=init, known_plant=false,
        sig0_costate=sig0_costate, sigma_prior_strength=sigma_prior_strength,
        qc_prior_strength=qc_prior_strength)
    n = plant_dim(base.state_model)
    M, Q = free_drift(n; decay=0.88, costate_decay=0.8, noise=0.12)
    free = free_state_model(M, Q; P0=Matrix(0.2I, 2n, 2n),
        Bu=zeros(2n, size(base.state_model.Gref, 2)), observe_costate=false,
        fit_flags=LQRFitFlags(Bu=false))
    set_depends_on!(free, (Qc=truth.rewards,))
    om1 = deepcopy(base.obs_model); om2 = deepcopy(base.obs_model)
    l1 = LinearDynamicalSystem(base.state_model, om1)
    l2 = LinearDynamicalSystem(free, om2)
    l1.fit_bool[4] = fit_noise
    slds = SLDS(A=[0.95 0.05; 0.05 0.95], πₖ=[0.1, 0.9], LDSs=[l1, l2])
    return slds
end

function recover_smoulder_slqr(; n::Int=12, tsteps::Int=100, ntrials::Int=1000,
        obs_dim::Int=150, nrewards::Int=3, max_iter::Int=75,
        sig0_costate::Float64=2e-2, sigma_prior_strength::Float64=0.0,
        qc_prior_strength::Float64=0.0, fit_noise::Bool=true, seed::Int=1)
    data = smoulder_slqr_dataset(; n=n, tsteps=tsteps, ntrials=ntrials,
        obs_dim=obs_dim, nrewards=nrewards, seed=seed)
    truth, ys, zs = data.truth, data.ys, data.zs
    slds = smoulder_slqr_fit(truth, ys; sig0_costate=sig0_costate,
        sigma_prior_strength=sigma_prior_strength,
        qc_prior_strength=qc_prior_strength, fit_noise=fit_noise)
    trace = fit!(slds, ys; ux=truth.uxs, max_iter=max_iter, smoothing_iters=2,
        progress=false, rng=MersenneTwister(300seed), tied_params=(:C, :d))
    post = smooth(slds, ys; ux=truth.uxs, smoothing_iters=100, progress=false)
    fitlds = slds.LDSs[LQR_STATE]
    return (scores=smoulder_scores(fitlds, truth.lds, 1:nrewards),
        gamma=gamma_scores(post.γ, zs; K=2), onset=onset_error(post.γ, zs;
            lqr_state=LQR_STATE), elbos=collect(trace), fit=slds, truth=truth.lds)
end

function experiment_smoulder_slqr(cfg; figures::Bool=true)
    c0 = smoulder_dimensions(cfg)
    s = cfg.slds
    p = c0.obs_dim
    transitions = s.ntrials*(s.tsteps-1)
    conditions = [
        ("loose Sigma_ll baseline", (;)),
        ("tight Sigma_ll=1e-4", (; sig0_costate=1e-4)),
        ("Sigma pinned", (; fit_noise=false)),
        ("Sigma prior 1% transitions", (; sigma_prior_strength=0.01transitions)),
        ("Sigma prior 10% transitions", (; sigma_prior_strength=0.10transitions)),
        ("Sigma+Qc priors", (; sigma_prior_strength=0.10transitions,
            qc_prior_strength=0.01transitions)),
    ]
    base = (; n=s.n, tsteps=s.tsteps, ntrials=s.ntrials, obs_dim=p,
        nrewards=c0.rewards, max_iter=s.max_iter)
    res = cells(recover_smoulder_slqr,
        [lab => (; base..., kw...) for (lab, kw) in conditions]; seeds=s.seeds)
    section("6c — Smoulder-scale two-state Poisson SLQR (free -> controlled)" *
        "\n     (reward remains a known Qc grouping; the epoch is inferred)")
    @printf("%-30s %9s %9s %9s %9s %9s\n",
        "condition", "Q align", "Gr align", "gamma", "onset", "G'G")
    println("-"^82)
    for (lab, _) in conditions
        rs = filter(!isnothing, res[lab])
        isempty(rs) && continue
        r = aggregate_smoulder(rs)
        med(f) = center([Float64(f(x)) for x in rs])[1]
        @printf("%-30s %9.3f %9.3f %9.3f %9.2f %9.3f\n", lab,
            r.scores.aligned.Qc.rmse, r.scores.aligned.Gref.rmse,
            med(x -> x.gamma.acc), med(x -> x.onset.mad), r.scores.Ggram.rmse)
    end
    return res
end

"""Structural checks for known-reward grouping and coordinate scoring."""
function smoulder_selftest()
    truth = smoulder_truth(
        n=3, tsteps=10, ntrials=9, obs_dim=12, nrewards=3, seed=909
    )
    sm = truth.lds.state_model
    vs = [group_variant(sm, :Qc, r) for r in 1:3]
    @assert vs[1].A === vs[2].A === vs[3].A
    @assert vs[1].S === vs[2].S === vs[3].S
    @assert vs[1].Gref === vs[2].Gref === vs[3].Gref
    @assert !(vs[1].Qc[1] ≈ vs[2].Qc[1])
    @assert !(vs[1].Qc[end] ≈ vs[2].Qc[end])

    W = random_orthogonal(MersenneTwister(910), 3)
    rotated = deepcopy(truth.lds)
    rotated.obs_model.C[:, 1:3] .= truth.lds.obs_model.C[:, 1:3] * W
    rsm = rotated.state_model
    rsm.A .= W' * sm.A * W
    rsm.S .= W' * sm.S * W
    rsm.Gref .= W' * sm.Gref
    for r in 1:3
        vr, vt = group_variant(rsm, :Qc, r), group_variant(sm, :Qc, r)
        for k in eachindex(vr.Qc)
            vr.Qc[k] .= W' * vt.Qc[k] * W
        end
        refresh!(vr)
    end
    refresh!(rsm)
    sc = smoulder_scores(rotated, truth.lds, 1:3)
    @assert sc.raw.Gref.rmse > 0.1
    @assert sc.aligned.Gref.rmse < 1e-10
    @assert sc.linear.Gref.rmse < 1e-10
    @assert sc.aligned.Qc.rmse < 1e-10
    @assert sc.aligned.A.rmse < 1e-10
    @assert sc.aligned.S.rmse < 1e-10
    @assert sc.aligned.cl.rmse < 1e-10
    @assert sc.Ggram.rmse < 1e-10
    println("smoulder selftest passed")
    return nothing
end
