""" file2. jl written the 14-15.04.2026.
This file explores a very basic gaussian process implementation. Also:
    1. testing gp on 1d and 2d functions
    2. estimate Q²
    3. nice plot for 1d functions

Next things the would be interresting to try out next:
    - adaptive sampling
    - probability box
"""

using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra
Random.seed!(42)


# ============================================================
# Inputs
# ============================================================
x1 = RandomVariable.(Uniform(-5, 5), :x1)
x2 = RandomVariable.(Uniform(-5, 5), :x2)
X = [x1, x2]

# X = RandomVariable.(Uniform(0, 1), :x)
# ============================================================
# Model
# ============================================================

model = Model(
    rv -> (rv.x1 .^ 2 .+ rv.x2 .- 11) .^ 2 .+ (rv.x1 .+ rv.x2 .^ 2 .- 7) .^ 2,
    :y
) # himmelblau

# ============================================================
# Sampling
# ============================================================
design_train = LatinHypercubeSampling(80)

data_train = sample(X, design_train)
evaluate!(model, data_train)

x_names = propertynames(data_train[:, Not(:y)])

# ============================================================
# Initial GP hyperparameters
# ============================================================

# using Statistics
function median_pairwise_distance(X::Matrix)
    n = size(X, 1)
    dists = [norm(X[i,:] - X[j,:]) for i in 1:n for j in i+1:n]
    return median(dists)
end

θ0 = (
    lengthscale = positive(median_pairwise_distance(Matrix(data_train[:, x_names]))),
    variance    = positive(var(data_train[:, :y])),
    noise       = positive(0.01 * var(data_train[:, :y])),
)


# θ0 = (
#     lengthscale = positive(abs(mean(data_train[:, :y]))),
#     variance    = positive(var(data_train[:, :y])),
#     noise       = positive(1e-1),
# )

metamodel = GaussianProcess(
    data_train,
    :y,
    θ0
)

# ============================================================
# Train GP
# ============================================================

fit!(metamodel)

# ============================================================
# Testing
# ============================================================
n_test = 1000
design_test = LatinHypercubeSampling(n_test)

data_test = sample(X, design_test)
evaluate!(model, data_test)

X_test = data_test[:, metamodel.x_names]
y_test = data_test[:, :y]

μ, σ = predict(
    metamodel,
    X_test
)

y_true = y_test
y_pred = μ

mse_val     = mse(y_true, y_pred)
rmse_val    = rmse(y_true, y_pred)
nrmse_val   = nrmse(y_true, y_pred)
nrmse_val_2 = nrmse(y_true, y_pred, method=:minmax)
q2_val      = q2(y_true, y_pred)

println("MSE:               $(round(mse_val, digits=5))")
println("RMSE:              $(round(rmse_val, digits=5))")
println("nRMSE (std):       $(round(nrmse_val, digits=5))")
println("nRMSE (minmax):    $(round(nrmse_val_2, digits=5))")
println("Q²:                $(round(q2_val, digits=5))")

##################################

if length(x_names) == 1
    # Extract x values
    x_vals = X_test[:, :x]

    # Sort indices
    perm = sortperm(x_vals)

    # Apply sorting
    x_sorted  = x_vals[perm]
    y_sorted  = y_test[perm]
    μ_sorted  = μ[perm]
    σ_sorted  = σ[perm]

    # ── Plot ─────────────────────────────────────

    plot(x_sorted, y_sorted,
        label     = "True function",
        color     = :black,
        lw        = 2,
        linestyle = :dash)

    plot!(x_sorted, μ_sorted,
        label = "GP mean",
        color = :blue,
        lw    = 2)

    plot!(x_sorted, μ_sorted .+ 2σ_sorted,
        fillrange  = μ_sorted .- 2σ_sorted,
        fillalpha  = 0.2,
        fillcolor  = :blue,
        linealpha  = 0,
        label      = "±2σ band")

    scatter!(data_train[:, :x], data_train[:, :y],
        label  = "Training points",
        color  = :red,
        ms     = 5,
        marker = :circle)

    xlabel!("x")
    ylabel!("f(x)")
    title!("GP surrogate | n = $n_train | Q² = $(round(Q2, digits=4))")   
end