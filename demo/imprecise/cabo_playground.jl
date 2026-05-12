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
x2 = RandomVariable.(Uniform(-5, 5), :x2)
x3 = RandomVariable.(Uniform(-5, 5), :x3)

X = [x1, x2, x3]

model = Model(
    rv -> rv.x1 .* (rv.x2.^2 .+ rv.x2 + cos.(π .* rv.x3) .- 7),
    :y
)

# ============================================================
# Sampling for: 1. train and 2. test #
# ============================================================

n_train = 15
design_train = MonteCarlo(n_train)

data_train = sample(X, design_train)

evaluate!(model, data_train)

X_names = [x.name for x in X]
