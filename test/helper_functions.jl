# Test the analytical gradient against the numerical gradient
function test_gradient(objective, objective_grad!, params; atol::Float64=1e-6)
    # numerical gradient
    G_numeric = ForwardDiff.gradient(objective, params)
    # analytic gradient
    G_analytic = zeros(3, 2)
    objective_grad!(G_analytic, params)
    # compare
    @test isapprox(G_numeric, G_analytic, atol=atol)
end

function print_models(true_model, est_model, data...)
    println("True Model:")
    println("true_model: ", true_model)
    println("loglikelihood: ", SSM.loglikelihood(true_model, data...))
    println()
    println("Estimated Model:")
    println("est_model: ", est_model)
    return println("loglikelihood: ", SSM.loglikelihood(est_model, data...))
end
