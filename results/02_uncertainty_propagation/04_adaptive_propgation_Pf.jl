using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Printf

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

# ==============================================================================
# 10. Usage — mirrors your UQ.jl example end to end
# ==============================================================================

# parametric inputs
x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input
 
physical_model = model_gfunction


x_names, w_names, u_names, v_names = spec_names(specs)
y_symbol = physical_model.name


n_train, n_test = 50, 1001


data_aug_train, data_phys_train =    build_augmented_design(physical_model, specs, n_train)
data_aug_test,  data_phys_test  =    build_augmented_design(physical_model, specs, n_test)


# # initialize GP on θ-space
kernel() = GPMatern52()
metamodel = GaussianProcess(data_aug_train, :y, kernel_type=kernel())
@time "fit!" fit!(metamodel)

μ_test, σ_test = @time "predict:" predict(metamodel, Matrix(data_aug_test[:, w_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")

# ================================================================
# CABO LOOP
# ================================================================

Ng          = 2000
Nx          = 5000
tol_BO      = 1e-3
tol_BC      = 1e-2
max_iter    = 25
qoi_type    = :pf
y_star      = -1.427

println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
println("    Running CABO with Params")
println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
println("Ng          = $Ng")
println("Nx          = $Nx")
println("tol_BO      = $tol_BO")
println("tol_BC      = $tol_BC")
println("max_iter    = $max_iter")
println("qoi_type    = $qoi_type")
println("y_star      = $y_star")

println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

cabo_min = @time "cabo min loop" cabo_loop(
    physical_model,
    metamodel,
    data_aug_train,
    specs;
    Ng = Ng, Nx = Nx, qoi_type = qoi_type, y_star = y_star,
    max_iter = max_iter, direction = :min,
    tol_BO = tol_BO, tol_BC = tol_BC
)

cabo_max = @time "cabo max loop" cabo_loop(
    physical_model,
    metamodel,
    cabo_min.data,
    specs;
    Ng = Ng, Nx = Nx, qoi_type = qoi_type, y_star = y_star,
    max_iter = max_iter, direction = :max,
    tol_BO = tol_BO, tol_BC = tol_BC
)

# ================================================================
# Plotting Results
# ================================================================

true_min = ([+1.50, 0.0], 0.0)
true_max = ([-0.57, +0.52], 0.0734)

plot_cabo_landscape = plot_epistemic_landscape(
    specs;
    cabo_min = cabo_min,
    cabo_max = cabo_max,
    analytical_qoi = nothing,
    true_min = true_min,
    true_max = true_max,
    qoi_label = "Probability Failure Pf",
)

plot_cabo_history = plot_convergence_history(cabo_min=cabo_min, cabo_max=cabo_max)

plot_cabo_convergence = plot_bound_convergence(
    cabo_min=cabo_min, 
    cabo_max=cabo_max,
    true_bound=(true_min[2], true_max[2]), 
    qoi_label="Pf bound", 
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
println("MIN  Pf ≈ $(round(cabo_min.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_min.θ_bound, digits=4))")
println("MAX  Pf ≈ $(round(cabo_max.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_max.θ_bound, digits=4))")
println("-"^60)
println("Expected:  MIN ≈ $(true_min[2]) at θ = $(true_min[1])")
println("           MAX ≈ $(true_max[2]) at θ = $(true_max[1])")
println("="^60)