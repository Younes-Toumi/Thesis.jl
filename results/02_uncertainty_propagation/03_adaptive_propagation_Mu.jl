using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Printf

Random.seed!(42)

# parametric inputs
x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input
 
x_names, w_names, u_names, v_names = spec_names(specs)
physical_model = model_gfunction

data_aug_train, data_phys_train =    build_augmented_design(physical_model, specs, 40)
data_aug_test,  data_phys_test  =    build_augmented_design(physical_model, specs, 1001)


# # initialize GP on θ-space
metamodel = GaussianProcess(data_aug_train, :y, kernel_type=GPMatern52())
fit!(metamodel)

μ_pred, σ_pred = predict(metamodel, Matrix(data_aug_test[:, w_names]))

println("Q²:  $(round(q2(data_aug_test.y, μ_pred), digits=5))")


# ================================================================
# CABO LOOP
# ================================================================


Ng          = 500
Nx          = 2000
tol_BO      = 1e-3
tol_BC      = 1e-2
max_iter    = 25
qoi_type    = :mean

println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
println("    Running CABO with Params")
println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
println("Ng          = $Ng")
println("Nx          = $Nx")
println("tol_BO      = $tol_BO")
println("tol_BC      = $tol_BC")
println("max_iter    = $max_iter")
println("qoi_type    = $qoi_type")
println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

cabo_min = @time "cabo min loop" cabo_loop(
    physical_model,
    metamodel,
    data_aug_train,
    specs;
    Ng = Ng, Nx = Nx, qoi_type = qoi_type,
    max_iter = max_iter, direction = :min,
    tol_BO = tol_BO, tol_BC = tol_BC
)

cabo_max = @time "cabo max loop" cabo_loop(
    physical_model,
    metamodel,
    cabo_min.data,
    specs;
    Ng = Ng, Nx = Nx, qoi_type = qoi_type,
    max_iter = max_iter, direction = :max,
    tol_BO = tol_BO, tol_BC = tol_BC
)

# ================================================================
# Plotting Results
# ================================================================

true_min = ([-0.567, 0.535], -1.351)
true_max = ([0.556, 0.811], 1.327)
true_bound = (true_min[2], true_max[2])

analytical_qoi = θ -> g_function_E(θ[1], θ[2]) # found in physicalmodels.jl

plot_cabo_landscape = plot_epistemic_landscape(
    specs;
    cabo_min = cabo_min,
    cabo_max = cabo_max,
    analytical_qoi = analytical_qoi,
    true_min = true_min,
    true_max = true_max,
    qoi_label = "Expected response E[g|θ]",
)

plot_cabo_history = plot_convergence_history(cabo_min=cabo_min, cabo_max=cabo_max)

plot_cabo_convergence = plot_bound_convergence(
    cabo_min=cabo_min, 
    cabo_max=cabo_max,
    true_bound=true_bound, 
    qoi_label="μ bound", 
    combined_budget=true)

display(plot_cabo_landscape)
display(plot_cabo_history)
display(plot_cabo_convergence)


# ================================================================
# Summary
# ================================================================

println("\n" * "="^60)
println("CABO results")
println("="^60)
println("MIN  E ≈ $(round(cabo_min.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_min.θ_bound, digits=4))")
println("MAX  E ≈ $(round(cabo_max.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_max.θ_bound, digits=4))")
println("-"^60)
println("Expected:  MIN ≈ $(true_min[2]) at θ = $(true_min[1])")
println("           MAX ≈ $(true_max[2]) at θ = $(true_max[1])")
println("="^60)