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
x1 = RandomVariable.(Normal(0, 1), :x1)
x2 = RandomVariable.(Normal(0, 1), :x2)
X = [x1, x2]

model = Model(
    rv -> (rv.x1 .^ 2 .+ rv.x2 .- 11) .^ 2 .+ (rv.x1 .+ rv.x2 .^ 2 .- 7) .^ 2,
    :y
) # himmelblau

# ============================================================
# Sampling for: 1. train and 2. test #
# ============================================================

n_train, n_test = 20, 1000

design_train = LatinHypercubeSampling(n_train)
design_test = LatinHypercubeSampling(n_test)

data_train = sample(X, design_train)
data_test = sample(X, design_test)

evaluate!(model, data_train)
evaluate!(model, data_test)

X_names = [x.name for x in X]

X_train = data_train[:, X_names]
X_test = data_test[:, X_names]
y_test = data_test[:, :y]

# ============================================================
# Scaling datasets
# ============================================================

pipeline = fit_pipeline(data_train, :y, MinMaxScaler, ZScoreScaler)

# apply to any dataset
data_train_scaled = SurrogateModelling.transform(pipeline, data_train, :y)
data_test_scaled  = SurrogateModelling.transform(pipeline, data_test, :y)

X_test_scaled = data_test_scaled[:, X_names]


# ============================================================
# Initial GP hyperparameters
# ============================================================
metamodel = GaussianProcess(data_train_scaled, :y)

@time "fit!" fit!(metamodel)
μ_scaled, σ_scaled = @time "predict" predict(metamodel, X_test_scaled)

μ = SurrogateModelling.inverse_mean(pipeline, μ_scaled)
σ = sqrt.(SurrogateModelling.inverse_variance(pipeline, σ_scaled.^2))

y_pred = μ

println("MSE:               $(round(mse(y_test, y_pred), digits=5))")
println("Q²:                $(round(q2(y_test, y_pred), digits=5))")