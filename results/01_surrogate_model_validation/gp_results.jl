using SurrogateModelling
using UncertaintyQuantification: sample, evaluate
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra

# ============================================================
# Inputs + Model
# ============================================================
x1 = RandomVariable.(Uniform(-pi, pi), :x1)
x2 = RandomVariable.(Uniform(-pi, pi), :x2)
x3 = RandomVariable.(Uniform(-pi, pi), :x3)
X = [x1, x2, x3]

model = Model(
    rv -> ishigami.(rv.x1, rv.x2, rv.x3),
    :y
) # ishigami

# ============================================================
# Sampling for: 1. train and 2. test #
# ============================================================

n_train, n_test = 50, 10000

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
# Initial GP hyperparameters
# ============================================================

metamodels = [
    GaussianProcess(data_train, :y,  kernel_type=GPSquaredExponential()),
    GaussianProcess(data_train, :y,  kernel_type=GPMatern12()),
    GaussianProcess(data_train, :y;  kernel_type=GPMatern32()), 
    GaussianProcess(data_train, :y;  kernel_type=GPMatern52())
]

messages = [
    "GP (ZeroMean): with kernel: GPSquaredExponential",
    "GP (ZeroMean): with kernel: GPMatern12",
    "GP (ZeroMean): with kernel: GPMatern32",
    "GP (ZeroMean): with kernel: GPMatern52"
]

for (metamodel, message) in zip(metamodels, messages)
    println("\n============================================================")
    println("$message")
    println("============================================================\n")

    @time "fit!" fit!(metamodel)
    μ, σ = @time "predict" predict(metamodel, Matrix(X_test))

    global y_pred = μ

    println("RMSE:               $(round(rmse(y_test, y_pred), digits=5))")
    println("Q²:                $(round(q2(y_test, y_pred), digits=5))")

end