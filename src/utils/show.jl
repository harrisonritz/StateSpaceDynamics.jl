# Pretty-printing (`Base.show`) for the LDS / SLDS model types. Extracted from
# types.jl so the type definitions and constructors stay free of display logic.
# Included after types.jl because each method signature references a type defined
# there.

# Pretty print function that doesn't truncate arrays of model objects

"""
    print_full([io::Union{IO, Base.TTY}, ] obj)

Prints full description of object `obj`, overriding both `io`-based limits as
well as the limits set in the default pretty printing of `StateSpaceDynamics`
objects.
"""
function print_full(io::Union{IO,Base.TTY}, obj)
    println(IOContext(io, :limit => false), obj)

    return nothing
end

print_full(obj) = print_full(stdout, obj)

#=
One line per parameter group that depends on an ancillary variable, e.g.

  Depends on:
   C, d, D  ->  2 groups (:session_a, :session_b)

Prints nothing when `depends_on` is unset, so the display of an ordinary model
is unchanged.
=#
function _show_depends_on(io::IO, model::DependentModel; gap="")
    model.depends_on === nothing && return nothing
    dep = _resolve_dependence(model)
    any(dep.varies) || return nothing

    println(io, gap, " Depends on:")
    for g in eachindex(dep.names)
        dep.varies[g] || continue
        members = if dep.names[g] === :A
            "A, b, B"
        elseif dep.names[g] === :C
            "C, d, D"
        else
            String(dep.names[g])
        end
        labels = join(map(repr, dep.labels[g]), ", ")
        println(io, gap, "  $members  ->  $(dep.nslots[g]) groups ($labels)")
    end
    return nothing
end

function Base.show(io::IO, gsm::GaussianStateModel; gap="")
    println(io, gap, "Gaussian State Model:")
    println(io, gap, "---------------------")

    if size(gsm.A, 1) > 4 || size(gsm.A, 2) > 4
        println(io, gap, " State Parameters:")
        println(io, gap, "  size(A)  = ($(size(gsm.A,1)), $(size(gsm.A,2)))")
        println(io, gap, "  size(Q)  = ($(size(gsm.Q,1)), $(size(gsm.Q,2)))")
        println(io, gap, " Initial State:")
        println(io, gap, "  size(b)  = ($(length(gsm.b)), )")
        println(io, gap, "  size(x0) = ($(length(gsm.x0)), )")
        println(io, gap, "  size(P0) = ($(size(gsm.P0,1)), $(size(gsm.P0,2)))")
    else
        println(io, gap, " State Parameters:")
        println(io, gap, "  A  = $(round.(gsm.A, sigdigits=3))")
        println(io, gap, "  Q  = $(round.(gsm.Q, sigdigits=3))")
        println(io, gap, " Initial State:")
        println(io, gap, "  b  = $(round.(gsm.b, digits=2))")
        println(io, gap, "  x0 = $(round.(gsm.x0, digits=2))")
        println(io, gap, "  P0 = $(round.(gsm.P0, sigdigits=3))")
    end

    println(io, gap, " Dynamics input:")
    println(io, gap, "  size(B)  = ($(size(gsm.B,1)), $(size(gsm.B,2)))")

    _show_depends_on(io, gsm; gap=gap)

    return nothing
end

#=
`:free` mode has no plant, cost or costate, so printing the LQR block would be
printing empty matrices. Show what it actually carries instead.
=#
function _show_free_state_model(io::IO, lqr_sm::LQRStateModel, n::Int; gap="")
    d = 2n
    println(io, gap, "LQR State Model (:free — unconstrained dynamics):")
    println(io, gap, "--------------------------------------------------------")
    println(io, gap, " Latent dim = $d   [no costate interpretation in :free mode]")
    if d <= 8
        println(io, gap, "  M = $(round.(lqr_sm.Mfree, sigdigits=3))")
    else
        println(io, gap, "  size(M) = ($d, $d)")
    end
    println(io, gap, " Noise:")
    println(io, gap, "  size(Σ)  = ($(size(lqr_sm.Σ, 1)), $(size(lqr_sm.Σ, 2)))")
    println(io, gap, " Initial state:")
    println(io, gap, "  size(x0) = ($(length(lqr_sm.x0)),)")
    println(io, gap, "  size(P0) = ($(size(lqr_sm.P0, 1)), $(size(lqr_sm.P0, 2)))")
    println(io, gap, " Dynamics input:")
    println(io, gap, "  size(Bu) = ($(size(lqr_sm.Bu, 1)), $(size(lqr_sm.Bu, 2)))")
    f = lqr_sm.fit_flags
    free = String[
        s for
        (s, on) in (("M", f.A), ("h", f.h), ("Bu", f.Bu && size(lqr_sm.Bu, 2) > 0)) if on
    ]
    println(io, gap, " Fitting:")
    println(io, gap, "  free: " * (isempty(free) ? "(none)" : join(free, ", ")))
    return nothing
end

#=
`:hold` mode shares the plant, cost and inputs with `:lqr` mode, but has no
schedule, no terminal factor and no symplectic structure; what characterizes it
is the steady state, so that is what the summary reports.
=#
function _show_hold_state_model(io::IO, lqr_sm::LQRStateModel{T}, n::Int; gap="") where {T}
    println(io, gap, "LQR State Model (:hold — infinite-horizon regulator):")
    println(io, gap, "--------------------------------------------------------")
    println(io, gap, " Plant dim n = $n, latent dim 2n = $(2n)   [z = (x; λ), λ ≈ P x + g]")
    if n <= 4
        println(io, gap, "  A     = $(round.(lqr_sm.A, sigdigits=3))")
        println(io, gap, "  S     = $(round.(lqr_sm.S, sigdigits=3))   [= B R⁻¹ Bᵀ]")
        println(io, gap, "  Q_h   = $(round.(lqr_sm.Qc[1], sigdigits=3))")
    else
        println(io, gap, "  size(A)  = ($n, $n)")
        println(io, gap, "  size(S)  = ($n, $n)   [= B R⁻¹ Bᵀ]")
        println(io, gap, "  size(Q_h) = ($n, $n)")
    end
    H = _HoldUnit(T, n, size(lqr_sm.Bu, 2))
    ok = _hold_steady_state!(
        H, lqr_sm.A, lqr_sm.S, lqr_sm.Qc[1], lqr_sm.h, lqr_sm.Bu, lqr_sm.Gref
    )
    println(io, gap, " Steady state:")
    if ok
        ρ = maximum(abs, eigvals(H.Acl))
        println(io, gap, "  closed-loop spectral radius ρ(A_cl) = $(round(ρ, sigdigits=3))")
    else
        println(io, gap, "  (no stabilizing DARE solution at these parameters)")
    end
    println(io, gap, " Noise (plant row / manifold row):")
    println(io, gap, "  size(Σ)  = ($(size(lqr_sm.Σ,1)), $(size(lqr_sm.Σ,2)))")
    println(io, gap, " Initial state:")
    println(io, gap, "  size(x0) = ($(length(lqr_sm.x0)),)")
    println(io, gap, "  size(P0) = ($(size(lqr_sm.P0,1)), $(size(lqr_sm.P0,2)))")
    println(io, gap, " Dynamics input:")
    println(io, gap, "  size(Bu)   = ($(size(lqr_sm.Bu,1)), $(size(lqr_sm.Bu,2)))")
    println(
        io,
        gap,
        "  size(Gref) = ($(size(lqr_sm.Gref,1)), $(size(lqr_sm.Gref,2)))" *
        (all(iszero, lqr_sm.Gref) ? "   [no reference]" : "   [tracking]"),
    )
    f = lqr_sm.fit_flags
    free = String[
        s for (s, on) in (
            ("A", f.A),
            ("S", f.S),
            ("Q_h", f.Qc),
            ("h", f.h),
            ("Bu", f.Bu),
            ("Gref", f.Gref && size(lqr_sm.Gref, 2) > 0),
        ) if on
    ]
    println(io, gap, " Fitting:")
    println(io, gap, "  free: " * (isempty(free) ? "(none)" : join(free, ", ")))
    println(io, gap, "  observe_costate = $(lqr_sm.observe_costate)")
    return nothing
end

function Base.show(io::IO, lqr_sm::LQRStateModel; gap="")
    n = _plant_dim(lqr_sm)
    if _is_free(lqr_sm)
        return _show_free_state_model(io, lqr_sm, n; gap=gap)
    end
    _is_hold(lqr_sm) && return _show_hold_state_model(io, lqr_sm, n; gap=gap)
    causal = _is_causal(lqr_sm)
    println(io, gap, causal ? "Causal LQR State Model:" : "LQR State Model:")
    println(io, gap, "--------------------------------------")
    println(io, gap, " Plant dim n = $n, latent dim 2n = $(2n)   [z = (x; λ)]")

    small = n <= 4
    println(io, gap, " LQR structure:")
    if small
        println(io, gap, "  A     = $(round.(lqr_sm.A, sigdigits=3))")
        println(io, gap, "  S     = $(round.(lqr_sm.S, sigdigits=3))   [= B R⁻¹ Bᵀ]")
        for (k, Q) in enumerate(lqr_sm.Qc)
            println(io, gap, "  Qc[$k] = $(round.(Q, sigdigits=3))")
        end
    else
        println(io, gap, "  size(A)  = ($n, $n)")
        println(io, gap, "  size(S)  = ($n, $n)   [= B R⁻¹ Bᵀ]")
        println(io, gap, "  Qc       = $(length(lqr_sm.Qc)) cost matrices of ($n, $n)")
    end

    println(io, gap, " Cost schedule:")
    if isempty(lqr_sm.schedule)
        println(io, gap, "  (none) — one cost on every transition")
    else
        counts = [count(==(k), lqr_sm.schedule) for k in 1:length(lqr_sm.Qc)]
        println(
            io, gap, "  $(length(lqr_sm.schedule)) timesteps; per-regime counts = $counts"
        )
    end
    if causal
        println(io, gap, "  terminal cost:   $(lqr_sm.causal.terminal_cost)")
    else
        println(io, gap, "  terminal factor: $(lqr_sm.terminal)")
    end
    lqr_sm.terminal_regime > 0 && println(
        io,
        gap,
        "  terminal cost:   Qc[$(lqr_sm.terminal_regime)] pinned " *
        "(every trial's own last step, whatever its length)",
    )
    for s in lqr_sm.switches
        parts = String[]
        s.bridge == 0 || push!(parts, "bridge against Qc[$(s.bridge)]")
        s.entry === nothing ||
            push!(parts, "entry prior" * (s.entry.fit_gain ? "" : " (gain held)"))
        println(io, gap, "  switch at position $(s.pos): " * join(parts, ", "))
    end
    isempty(lqr_sm.gref_gate) || println(
        io,
        gap,
        "  reference gate:  each regime reads " *
        join(
            (
                "Qc[$k]: $(count(view(lqr_sm.gref_gate, k, :)))" for
                k in axes(lqr_sm.gref_gate, 1)
            ),
            ", ",
        ) *
        " input column(s)",
    )

    if causal
        o = lqr_sm.causal
        println(io, gap, " Noise (plant ε, costate slack ν, independent):")
        println(
            io, gap, "  Σ (plant):   $(o.plant_noise),  Ω (costate): $(o.costate_noise)"
        )
        println(
            io,
            gap,
            "  slack drives state: $(o.slack_drives_state)" *
            (o.slack_drives_state ? "   [agent acts on its costate]" : "   [readout]"),
        )
    else
        println(io, gap, " Noise (mixed coordinates on [x_{t+1}; λ_t]):")
        println(io, gap, "  size(Σ)  = ($(size(lqr_sm.Σ,1)), $(size(lqr_sm.Σ,2)))")
    end
    lqr_sm.terminal &&
        println(io, gap, "  size(Σf) = ($(size(lqr_sm.Σf,1)), $(size(lqr_sm.Σf,2)))")

    println(io, gap, " Initial state:")
    println(io, gap, "  size(x0) = ($(length(lqr_sm.x0)),)")
    println(io, gap, "  size(P0) = ($(size(lqr_sm.P0,1)), $(size(lqr_sm.P0,2)))")
    println(io, gap, " Dynamics input:")
    println(io, gap, "  size(Bu)   = ($(size(lqr_sm.Bu,1)), $(size(lqr_sm.Bu,2)))")
    println(
        io,
        gap,
        "  size(Gref) = ($(size(lqr_sm.Gref,1)), $(size(lqr_sm.Gref,2)))" *
        (all(iszero, lqr_sm.Gref) ? "   [no reference]" : "   [tracking]"),
    )

    f = lqr_sm.fit_flags
    println(io, gap, " Fitting:")
    println(
        io,
        gap,
        "  free: " * join(
            String[
                s for (s, on) in (
                    ("A", f.A),
                    ("S", f.S),
                    ("Qc", f.Qc),
                    ("h", f.h),
                    ("Bu", f.Bu),
                    ("Gref", f.Gref && size(lqr_sm.Gref, 2) > 0),
                    (
                        "terminal",
                        f.terminal &&
                            (causal ? lqr_sm.causal.terminal_cost : lqr_sm.terminal),
                    ),
                ) if on
            ],
            ", ",
        ),
    )
    println(io, gap, "  observe_costate = $(lqr_sm.observe_costate)")
    causal || println(
        io,
        gap,
        "  symplectic defect = $(round(symplectic_defect(lqr_sm), sigdigits=3))",
    )

    return nothing
end

function Base.show(io::IO, gom::GaussianObservationModel; gap="")
    println(io, gap, "Gaussian Observation Model:")
    println(io, gap, "---------------------------")

    if size(gom.C, 1) > 3 || size(gom.C, 2) > 3
        println(io, gap, " size(C) = ($(size(gom.C,1)), $(size(gom.C,2)))")
        println(io, gap, " size(R) = ($(size(gom.R,1)), $(size(gom.R,2)))")
        println(io, gap, " size(d) = ($(length(gom.d)),)")
        println(io, gap, " size(D) = ($(size(gom.D,1)), $(size(gom.D,2)))")
    else
        println(io, gap, " C = $(round.(gom.C, digits=2))")
        println(io, gap, " R = $(round.(gom.R, digits=2))")
        println(io, gap, " d = $(round.(gom.d, digits=2))")
        println(io, gap, " D = $(round.(gom.D, digits=2))")
    end

    _show_depends_on(io, gom; gap=gap)

    return nothing
end

function Base.show(io::IO, pom::PoissonObservationModel; gap="")
    obs_dim, nstate = size(pom.C)

    println(io, gap, "Poisson Observation Model:")
    println(io, gap, "--------------------------")

    if obs_dim > 4 || nstate > 4
        println(io, gap, " size(C) = ($obs_dim, $nstate)")
        println(io, gap, " size(d) = ($(length(pom.d)),)")
    else
        println(io, gap, " C    = $(round.(pom.C, digits=2))")
        println(io, gap, " d    = $(round.(pom.d, sigdigits = 3))")
        println(
            io,
            gap,
            " rate = $(round.(exp.(pom.d), digits = 2))   # exp(d) for inspection only",
        )
    end

    _show_depends_on(io, pom; gap=gap)

    return nothing
end

#=
Human-readable names for the `fit_bool` slots, in order. The compound entries
"A (and b, B)" / "C (and d, D)" reflect that each row is fit jointly as one
regression — the bias and user-input columns are not gated independently. A
composite emission prefixes each member's slots with the member name.
=#
_obs_fit_labels(::GaussianObservationModel) = ["C (and d, D)", "R"]
_obs_fit_labels(::PoissonObservationModel) = ["C, d"]

function _obs_fit_labels(c::CompositeObservationModel)
    labels = String[]
    for key in _obs_keys(c)
        for label in _obs_fit_labels(_models(c)[key])
            push!(labels, "$key: $label")
        end
    end
    return labels
end

function _fit_bool_labels(lds::LinearDynamicalSystem)
    state = if lds.obs_model isa PoissonObservationModel
        ["x0", "P0", "A (and b)", "Q"]
    else
        ["x0", "P0", "A (and b, B)", "Q"]
    end
    return vcat(state, _obs_fit_labels(lds.obs_model))
end

function Base.show(io::IO, com::CompositeObservationModel; gap="")
    models = _models(com)
    println(io, gap, "Composite Observation Model ($(length(models)) models):")
    println(io, gap, "-------------------------------------------")
    for key in keys(models)
        println(io, gap, " [$key]")
        Base.show(io, models[key]; gap=gap * "  ")
    end
    return nothing
end

function Base.show(io::IO, lds::LinearDynamicalSystem; gap="")
    println(io, gap, "Linear Dynamical System:")
    println(io, gap, "------------------------")
    Base.show(io, lds.state_model; gap=gap * " ")
    Base.show(io, lds.obs_model; gap=gap * " ")
    println(io, gap, " Parameters to update:")
    println(io, gap, " ---------------------")

    prms = _fit_bool_labels(lds)[lds.fit_bool]

    println(io, gap, "  $(join(prms, ", "))")
    return nothing
end

function Base.show(io::IO, slds::SLDS; gap="")
    K = length(slds.LDSs)

    println(io, gap, "Switching Linear Dynamical System (SLDS):")
    println(io, gap, "-----------------------------------------")
    println(io, gap, " Number of discrete states: $K")

    if K > 3
        println(io, gap, " size(A)  = ($(size(slds.A,1)), $(size(slds.A,2)))")
        println(io, gap, " size(πₖ) = ($(length(slds.πₖ)),)")
    else
        println(io, gap, " A  = $(round.(slds.A, sigdigits=3))")
        println(io, gap, " πₖ = $(round.(slds.πₖ, sigdigits=3))")
    end
    #= The chain priors, when set, in the same compact-or-shape form as the
    chain itself; an unpenalised chain prints nothing, as before. =#
    if slds.A_prior !== nothing
        if K > 3
            println(io, gap, " A_prior  = Dirichlet rows, size $(size(slds.A_prior))")
        else
            println(io, gap, " A_prior  = Dirichlet $(round.(slds.A_prior, sigdigits=3))")
        end
    end
    if slds.πₖ_prior !== nothing
        if K > 3
            println(io, gap, " πₖ_prior = Dirichlet, length $(length(slds.πₖ_prior))")
        else
            println(io, gap, " πₖ_prior = Dirichlet $(round.(slds.πₖ_prior, sigdigits=3))")
        end
    end

    println(io, gap, " Linear Dynamical Systems:")
    println(io, gap, " -------------------------")

    # Show details of first LDS
    if K > 0
        println(io, gap, "  State 1:")
        Base.show(io, slds.LDSs[1]; gap=gap * "   ")

        if K > 1
            println(io, gap, "  ... and $(K-1) more state(s)")
        end
    end

    return nothing
end
