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

# Describe one (possibly Indexed) parameter: a representative value's dimensions
# plus any grouping. Values are not printed in full so the display is stable for
# both plain and Varying parameters.
function _show_param(io, gap, name, p)
    v = at(p, 1)
    dims = ndims(v) == 1 ? "($(length(v)),)" : "$(size(v))"
    if is_varying(p)
        println(
            io, gap,
            "  $name: Varying — $(nvals(p)) groups by :$(param_label(p)) = $(param_group_ids(p)); size $dims",
        )
    elseif is_indexed(p)
        println(io, gap, "  $name: Static; size $dims")
    else
        println(io, gap, "  $name: size $dims")
    end
    return nothing
end

function Base.show(io::IO, gsm::GaussianStateModel; gap="")
    println(io, gap, "Gaussian State Model:")
    println(io, gap, "---------------------")
    _show_param(io, gap, "A ", gsm.A)
    _show_param(io, gap, "Q ", gsm.Q)
    _show_param(io, gap, "b ", gsm.b)
    _show_param(io, gap, "x0", gsm.x0)
    _show_param(io, gap, "P0", gsm.P0)
    _show_param(io, gap, "B ", gsm.B)
    return nothing
end

function Base.show(io::IO, gom::GaussianObservationModel; gap="")
    println(io, gap, "Gaussian Observation Model:")
    println(io, gap, "---------------------------")
    _show_param(io, gap, "C", gom.C)
    _show_param(io, gap, "R", gom.R)
    _show_param(io, gap, "d", gom.d)
    _show_param(io, gap, "D", gom.D)
    return nothing
end

function Base.show(io::IO, pom::PoissonObservationModel; gap="")
    println(io, gap, "Poisson Observation Model:")
    println(io, gap, "--------------------------")
    _show_param(io, gap, "C", pom.C)
    _show_param(io, gap, "d", pom.d)
    return nothing
end

function Base.show(io::IO, lds::LinearDynamicalSystem; gap="")
    println(io, gap, "Linear Dynamical System:")
    println(io, gap, "------------------------")
    Base.show(io, lds.state_model; gap=gap * " ")
    Base.show(io, lds.obs_model; gap=gap * " ")
    println(io, gap, " Parameters to update:")
    println(io, gap, " ---------------------")

    if _is_poisson_like(lds.obs_model)
        # C and d are either both updated or neither
        prms = ["x0", "P0", "A (and b)", "Q", "C, d"][lds.fit_bool[1:5]]
    else
        # Same labels for BTD and Kalman backends (length 6). The compound
        # entries "A (and b, B)" / "C (and d, D)" reflect that each row is
        # fit jointly as one regression — the bias and user-input columns
        # are not gated independently.
        prms = ["x0", "P0", "A (and b, B)", "Q", "C (and d, D)", "R"][lds.fit_bool[1:6]]
    end

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
