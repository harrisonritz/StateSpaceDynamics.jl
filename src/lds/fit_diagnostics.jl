"""
    parameter_diagnostics(model; posterior=nothing)

Compact, read-only numerical summaries for an LDS/SLDS, including every built
parameter-group variant and composite emission. `A_spectral_radius` is max |λ|;
`A_opnorm` also exposes nonnormal amplification that eigenvalues miss. Matrix
condition numbers and norms depend on latent coordinates. Nonfinite metrics
are `nothing`, with explicit finite/singular indicators where appropriate.

`posterior` may be the training TrialFilterSmooth, or a vector of per-cell
ones, already computed by the E-step. This adds latent-mean/RMS and emission
offset-cancellation diagnostics without an additional smoothing pass.
"""
function parameter_diagnostics(model; posterior=nothing)
    result = Dict{String,Any}("states" => Any[], "observations" => Any[])
    latent = posterior === nothing ? nothing : _diagnostic_latents(posterior)
    result["latent"] = latent
    ldss = model isa SLDS ? model.LDSs : (model,)
    for (regime, lds) in enumerate(ldss)
        if lds.state_model isa GaussianStateModel && lds.state_model.gauge !== nothing
            g = lds.state_model.gauge
            result["affine_gauge"] = Dict(
                "whitening_skips" => g.whitening_skips,
                "last_condition" => _diagnostic_finite(g.last_condition),
                "reference_map_norm" => _diagnostic_finite(norm(g.map)),
                "reference_offset_norm" => _diagnostic_finite(norm(g.offset)),
                "logprior_offset" => _diagnostic_finite(g.logprior_offset),
            )
        end
        _diagnostic_variants!(result["states"], lds.state_model, regime, "state", latent)
        if lds.obs_model isa CompositeObservationModel
            for (name, member) in pairs(_models(lds.obs_model))
                _diagnostic_variants!(
                    result["observations"], member, regime, String(name), latent
                )
            end
        else
            _diagnostic_variants!(
                result["observations"], lds.obs_model, regime, "observation", latent
            )
        end
    end
    return result
end

_diagnostic_finite(x::Real) = isfinite(x) ? Float64(x) : nothing

function _diagnostic_matrix!(row, prefix, matrix; covariance=false)
    row[prefix * "_finite"] = all(isfinite, matrix)
    isempty(matrix) && (row[prefix * "_norm"] = 0.0; return nothing)
    all(isfinite, matrix) || return nothing
    row[prefix * "_norm"] = _diagnostic_finite(norm(matrix))
    if covariance
        values = eigvals(Symmetric(matrix))
        row[prefix * "_eig_min"] = _diagnostic_finite(minimum(values))
        row[prefix * "_eig_max"] = _diagnostic_finite(maximum(values))
        row[prefix * "_positive_definite"] = minimum(values) > 0
        row[prefix * "_condition"] = if minimum(values) > 0
            _diagnostic_finite(maximum(values) / minimum(values))
        else
            nothing
        end
    end
end

function _diagnostic_variants!(rows, root, regime, member, latent)
    variants =
        if hasproperty(root, :variants) &&
            root.variants !== nothing &&
            !isempty(root.variants)
            root.variants
        else
            (root,)
        end
    dep = _resolve_dependence(root)
    for (index, variant) in enumerate(variants)
        slots = _variant_slots(dep.nslots, index)
        labels = Dict(
            String(dep.names[g]) => string(dep.labels[g][slots[g]]) for
            g in eachindex(slots) if dep.varies[g]
        )
        row = Dict{String,Any}(
            "regime" => regime,
            "member" => member,
            "variant" => index,
            "groups" => labels,
            "type" => string(nameof(typeof(variant))),
        )
        try
            for name in (:b, :B, :B0, :x0, :C, :d, :D)
                hasproperty(variant, name) || continue
                value = getproperty(variant, name)
                row[String(name) * "_norm"] = _diagnostic_finite(norm(value))
                row[String(name) * "_finite"] = all(isfinite, value)
                name == :d && (
                    row["d_maxabs"] =
                        isempty(value) ? 0.0 : _diagnostic_finite(maximum(abs, value))
                )
            end
            for name in (:Q, :P0, :R, :Σ)
                hasproperty(variant, name) || continue
                _diagnostic_matrix!(
                    row, String(name), getproperty(variant, name); covariance=true
                )
            end
            if hasproperty(variant, :A) && all(isfinite, variant.A)
                A = variant.A
                row["A_kind"] = variant isa GaussianStateModel ? "transition" : "plant"
                row["A_norm"] = _diagnostic_finite(norm(A))
                row["A_opnorm"] = _diagnostic_finite(opnorm(A))
                row["A_spectral_radius"] = _diagnostic_finite(maximum(abs, eigvals(A)))
                s = svdvals(I - A)
                row["I_minus_A_smin"] = _diagnostic_finite(minimum(s))
                row["I_minus_A_condition"] =
                    minimum(s) > 0 ? _diagnostic_finite(maximum(s) / minimum(s)) : nothing
            end
            if latent !== nothing &&
                latent["finite"] &&
                hasproperty(variant, :C) &&
                hasproperty(variant, :d) &&
                size(variant.C, 2) == length(latent["mean"])
                projected = variant.C * latent["mean"]
                effective = variant.d + projected
                row["C_latent_mean_norm"] = _diagnostic_finite(norm(projected))
                row["d_plus_C_mean_norm"] = _diagnostic_finite(norm(effective))
                row["d_plus_C_mean_maxabs"] = _diagnostic_finite(maximum(abs, effective))
            end
        catch err
            err isa InterruptException && rethrow()
            row["diagnostic_error"] = sprint(showerror, err)
        end
        push!(rows, row)
    end
end

function _diagnostic_latents(posterior)
    cells = posterior isa TrialFilterSmooth ? (posterior,) : posterior
    total = nothing
    n = 0
    squared = 0.0
    maxabs = 0.0
    inference = Dict{String,Any}[]
    for cell in cells, fs in cell.FilterSmooths
        x = fs.x_smooth
        total === nothing && (total = zeros(size(x, 1)))
        total .+= vec(sum(x; dims=2))
        n += size(x, 2)
        squared += sum(abs2, x)
        maxabs = max(maxabs, maximum(abs, x))
        isempty(fs.inference_stats) || push!(inference, fs.inference_stats)
    end
    n == 0 && return nothing
    center = total ./ n
    return Dict{String,Any}(
        "n_bins" => n,
        "mean" => all(isfinite, center) ? center : nothing,
        "finite" => all(isfinite, center) && isfinite(squared),
        "mean_norm" => _diagnostic_finite(norm(center)),
        "rms" => _diagnostic_finite(sqrt(squared / (n * length(center)))),
        "maxabs" => _diagnostic_finite(maxabs),
        "inference" => if isempty(inference)
            nothing
        else
            Dict(
                "method" => first(inference)["method"],
                "trials" => length(inference),
                "unconverged" => count(d -> !get(d, "converged", true), inference),
                "max_iterations" => maximum(get(d, "iterations", 0) for d in inference),
                "max_mean_residual" => _diagnostic_finite(
                    maximum(get(d, "mean_residual", 0.0) for d in inference)
                ),
                "max_precision_residual" => _diagnostic_finite(
                    maximum(get(d, "precision_residual", 0.0) for d in inference)
                ),
            )
        end,
    )
end
