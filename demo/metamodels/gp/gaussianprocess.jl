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

n_train, n_test = 100, 1000

design_train = LatinHypercubeSampling(n_train)
design_test = LatinHypercubeSampling(n_test)

data_train = sample(X, design_train)
data_test = sample(X, design_test)

evaluate!(model, data_train)
evaluate!(model, data_test)

X_names = [x.name for x in X]
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
# TODO: revise automatic parameter selection
metamodels = [
    GaussianProcess(data_train_scaled, :y),
    GaussianProcess(data_train_scaled, :y;  mean=GPConstMean()),
    GaussianProcess(data_train_scaled, :y;  kernel=GPMatern52()), 
    GaussianProcess(data_train_scaled, :y;  kernel=0.25*GPMatern52() + 0.75*GPSquaredExponential()), # GPMatern52() * GPSquaredExponential() works too
]

messages = [
    "Normal GP: default mean (zero) and kernel (squared exponential)",
    "Normal GP: with constant mean instead of zeromean",
    "Normal GP: with GPMatern52 kernel",
    "Normal GP: with composite kernel"
]

for (metamodel, message) in zip(metamodels, messages)
    println("\n============================================================")
    println("$message")
    println("============================================================\n")

    @time "fit!" fit!(metamodel)
    μ_scaled, σ_scaled = @time "predict" predict(metamodel, Matrix(X_test_scaled))

    μ = SurrogateModelling.inverse_mean(pipeline, μ_scaled)
    σ = sqrt.(SurrogateModelling.inverse_variance(pipeline, σ_scaled.^2))


    global y_pred = μ

    println("MSE:               $(round(mse(y_test, y_pred), digits=5))")
    println("Q²:                $(round(q2(y_test, y_pred), digits=5))")

end