using UncertaintyQuantification
using SurrogateModelling
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


# ============================================================
# Initial PCE Stuff
# ============================================================
bases = [SurrogateModelling.LegendreBasis(), SurrogateModelling.LegendreBasis()] # can be automated based on input
degree = TotalDegree(50) # can be automated based on availabel samples
solver = SurrogateModelling.LASSOSolver(λ=1e1)
# solver = SurrogateModelling.OLSSolver()

metamodel = SurrogateModelling.PolynomialChaosExpansion(data_train, :y, bases, degree; solver=solver)
@time "fit!" fit!(metamodel)

# ============================================================
# Testing
# ============================================================

X_test = data_test[:, X_names]

y_pred = @time "predict" predict(
    metamodel,
    X_test
)

y_true = data_test[:, :y]


mse_val     = mse(y_true, y_pred)
rmse_val    = rmse(y_true, y_pred)
nrmse_val   = nrmse(y_true, y_pred)
q2_val      = q2(y_true, y_pred)

println("\nmetrics:")
println("MSE:               $(round(mse_val, digits=5))")
println("RMSE:              $(round(rmse_val, digits=5))")
println("nRMSE (std):       $(round(nrmse_val, digits=5))")
println("Q²:                $(round(q2_val, digits=5))")


n_nonzero = count(!iszero, metamodel.coeffs)
n_zero = count(iszero, metamodel.coeffs)

println("coeffs for λ=0: $(metamodel.coeffs)\n")
println("Nonzero elements: ", n_nonzero)
println("Zero elements: ", n_zero)