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
# Inputs + Model
# ============================================================
x1 = RandomVariable.(Uniform(-5, 5), :x1)
x2 = RandomVariable.(Uniform(-5, 5), :x2)
X = [x1, x2]

model = Model(
    rv -> (rv.x1 .^ 2 .+ rv.x2 .- 11) .^ 2 .+ (rv.x1 .+ rv.x2 .^ 2 .- 7) .^ 2,
    :y
) # himmelblau

# ============================================================
# Sampling for: 1. train and 2. test #
# ============================================================

n_train, n_test = 80, 1000

design_train = LatinHypercubeSampling(n_train)
design_test = LatinHypercubeSampling(n_test)

data_train = sample(X, design_train)
data_test = sample(X, design_test)

evaluate!(model, data_train)
evaluate!(model, data_test)

X_names = [x.name for x in X]


# ============================================================
# Scaling datasets
# ============================================================

pipeline = fit_pipeline(data_train, :y, MinMaxScaler, ZScoreScaler)

# apply to any dataset
data_train_scaled = SurrogateModelling.transform(pipeline, data_train, :y)
data_test_scaled  = SurrogateModelling.transform(pipeline, data_test, :y)


# ============================================================
# Initial GP hyperparameters
# ============================================================

metamodel = GaussianProcess(data_train_scaled, :y;  kernel=GPSquaredExponential())
fit!(metamodel)


# # ============================================================
# # Testing
# # ============================================================

X_test_scaled = data_test_scaled[:, X_names]

μ_scaled, σ_scaled = predict(
    metamodel,
    X_test_scaled
)

# inverse transform
μ = SurrogateModelling.inverse_mean(pipeline, μ_scaled)
σ = sqrt.(SurrogateModelling.inverse_variance(pipeline, σ_scaled.^2))

y_true = data_test[:, :y]
y_pred = μ

mse_val     = mse(y_true, y_pred)
rmse_val    = rmse(y_true, y_pred)
nrmse_val   = nrmse(y_true, y_pred)
q2_val      = q2(y_true, y_pred)

println("MSE:               $(round(mse_val, digits=5))")
println("RMSE:              $(round(rmse_val, digits=5))")
println("nRMSE (std):       $(round(nrmse_val, digits=5))")
println("Q²:                $(round(q2_val, digits=5))")