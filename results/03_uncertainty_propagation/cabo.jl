using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Distributions
using Metaheuristics
using QuasiMonteCarlo

specs = [
    
    InputSpec(-1.5, 1.5, θ -> Normal(θ, 0.1)),
    InputSpec(-1.5, 1.5, θ -> Normal(θ, 0.1)),
]

const v_LOWER, v_UPPER =  -2.2, 2.2
const u_LOWER, u_UPPER =  -4.0, 4.0 # effective support of a standard normal

lb_v, ub_v = [v_LOWER, v_LOWER], [v_UPPER, v_UPPER]
lb_u, ub_u = [u_LOWER, u_LOWER], [u_UPPER, u_UPPER]

const bounds_v = boxconstraints(lb = lb_v, ub = ub_v)
const bounds_u = boxconstraints(lb = lb_u, ub = ub_u)

physical_model = g_function

n_ale = 2
n_epi = 2
w_names = [:u1, :u2, :v1, :v2]
u_names = [:u1, :u2]
v_names = [:v1, :v2]

n_train, n_test = 20, 1001

data_aug_train, data_phys_train =    build_augmented_design(physical_model, specs, n_train; seed=42)
data_aug_test,  data_phys_test  =    build_augmented_design(physical_model, specs, n_test; seed=123)


# initialize GP on θ-space
kernel() = GPMatern52() #  + GPSquaredExponential()
metamodel = GaussianProcess(data_aug_train, :y, kernel_type=kernel())
@time "fit!" fit!(metamodel)

μ_test, σ_test = @time "predict:" predict(metamodel, Matrix(data_aug_test[:, w_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")
 
cabo_min = @time "CABO MIN" cabo_loop(
    metamodel,
    data_aug_train,
    w_names,
    specs;
    max_iter = 15,
    direction = :min,
    tol_BO       = 5e-3,
    tol_BC       = 5e-2

)
 
cabo_max = @time "CABO MAX" cabo_loop(
    metamodel,
    cabo_min.data,
    w_names,
    specs;
    max_iter = 15,
    direction = :max,
    tol_BO       = 5e-3,
    tol_BC       = 5e-2
)

# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("CABO results")
println("="^60)
println("MIN  E[g] ≈ $(round(cabo_min.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_min.θ_bound, digits=3))")
println("MAX  E[g] ≈ $(round(cabo_max.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_max.θ_bound, digits=3))")
println("-"^60)
println("Expected:  MIN ≈ −1.35  at (−0.57, 0.52)")
println("           MAX ≈  1.33  at ( 0.55, 0.81)")
println("="^60)



# plot related
Θs_min = reduce(hcat, cabo_min.θ_history)'
Θs_max = reduce(hcat, cabo_max.θ_history)'
using Plots, Statistics

# ============================================================
# Grid over epistemic space (x1, θσ)
# ============================================================
n_μ1 = 500
n_μ2 = 500

μ1_grid = range(-1.5, 1.5, length=n_μ1)
μ2_grid  = range(-1.5, 1.5, length=n_μ2)

MeanSurface = zeros(n_μ1, n_μ2)

# ============================================================
# Compute response
# ============================================================
for (i, μ1_v) in enumerate(μ1_grid)
    for (j, μ2_v) in enumerate(μ2_grid)
        MeanSurface[j, i]  = g_function_E(μ1_v, μ2_v)
    end
end

# ============================================================
# Heatmap
# ============================================================
plt = heatmap(
    μ1_grid,
    μ2_grid,
    MeanSurface,
    xlabel="μ1",
    ylabel="μ2",
    c=:thermal,
    title="expected response function: E[g(x1, x2)]",
    colorbar=true,
    xlims = (-1.6, 1.6),
    ylims = (-1.6, 1.6),
    legend = :outerbottom,
    legendcolumns=4,
)

# ── overlay the initial training points (μ1, μ2 columns from data_aug) ──────
data_aug_train_epi = Matrix{Float64}(undef, n_train, n_epi)
for i in 1:n_train
    data_aug_train_epi[i, :] = augmented_to_epistemic(data_aug_train[i, v_names], specs)
end

scatter!(plt,
    data_aug_train_epi[:, 1], data_aug_train_epi[:, 2];
    marker = :diamond, color = :cyan, ms = 5,
    label  = "init samples", markerstrokewidth=0,
)

scatter!(plt,
    Θs_min[:, 1], Θs_min[:, 2];
    marker = :cross, color = :green, ms = 5,
    label  = "added min", markerstrokewidth=2,
)

scatter!(plt,
    Θs_max[:, 1], Θs_max[:, 2];
    marker = :cross, color = :red, ms = 5,
    label  = "added max", markerstrokewidth=2,
)

scatter!(plt,
    [cabo_min.θ_bound[1]], [cabo_min.θ_bound[2]];
    marker = :star, color = :green, ms = 7,
    label  = "cabo min", markerstrokewidth=1,
)

scatter!(plt,
    [cabo_max.θ_bound[1]], [cabo_max.θ_bound[2]];
    marker = :star, color = :red, ms = 7,
    label  = "cabo max", markerstrokewidth=1,
)

# ── mark true min and max in epistemic space ──────────────────────────────────
min_idx = argmin(MeanSurface)
max_idx = argmax(MeanSurface)

x_min, y_min, z_min = μ1_grid[min_idx[2]], μ1_grid[min_idx[1]], minimum(MeanSurface)
x_max, y_max, z_max = μ1_grid[max_idx[2]], μ1_grid[max_idx[1]], maximum(MeanSurface)

dy = 0.2

scatter!(plt,
    [μ1_grid[min_idx[2]]], [μ2_grid[min_idx[1]]];
    marker = :circle, color = :green, ms = 5, label = "True min",
)
annotate!(
    x_min, y_min + dy,
    text("($(round(x_min, digits=2)), $(round(y_min, digits=2)), $(round(z_min, digits=2)))", :black, 8)
)


scatter!(plt,
    [μ1_grid[max_idx[2]]], [μ2_grid[max_idx[1]]];
    marker = :rect, color = :red, ms = 5, label = "True max",
)

annotate!(
    x_max, y_max + dy,
    text("($(round(x_max, digits=2)), $(round(y_max, digits=2)), $(round(z_max, digits=2)))", :black, 8)
)

p1 = scatter(
    cabo_min.L_BO_history,
    xlabel = "Iteration",
    ylabel = "L_BO",
    title = "MIN: L_BO History",
    legend = false
)

p2 = scatter(
    cabo_min.L_BC_history,
    xlabel = "Iteration",
    ylabel = "L_BC",
    title = "MIN: L_BC History",
    ylims = (0, 1),
    legend = false
)

p3 = scatter(
    cabo_max.L_BO_history,
    xlabel = "Iteration",
    ylabel = "L_BO",
    title = "MAX: L_BO History",
    legend = false
)

p4 = scatter(
    cabo_max.L_BC_history,
    xlabel = "Iteration",
    ylabel = "L_BC",
    title = "MAX: L_BC History",
    ylims = (0, 1),
    legend = false
)

history = plot(
    p1, p2, p3, p4,
    layout = (2, 2),
    size = (900, 700)
)

display(plt)
display(history)