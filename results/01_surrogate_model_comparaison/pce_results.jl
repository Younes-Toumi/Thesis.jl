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
# PCE Modelling Stuff
# ============================================================
# TODO: Automatic selection of bases depending on input distribution (or convert to sns)
# TODO: Compute the total possible degree depending on the available training samples

bases = [SurrogateModelling.LegendreBasis(), SurrogateModelling.LegendreBasis(), SurrogateModelling.LegendreBasis()] # Uniform inputs
degree = TotalDegree(5) # can be automated based on availabel samples

# λ sparcity penality: a compromise between approximation accuracy and model sparcity.

metamodels = [
    SurrogateModelling.PolynomialChaosExpansion(data_train, :y, bases, degree; solver=SurrogateModelling.OLSSolver()), # chooses Least Squares by default
    SurrogateModelling.PolynomialChaosExpansion(data_train, :y, bases, degree; solver=SurrogateModelling.LASSOSolver()), # or not
]

messages = [
    "Normal PCE, OLS solver:",
    "Sparce PCE, LASSO solver with automatic λ:",
]

for (metamodel, message) in zip(metamodels, messages)
    println("\n============================================================")
    println("$message")
    println("============================================================\n")

    @time "fit!" fit!(metamodel)
    global y_pred = @time "predict" predict(metamodel, X_test)


    if solver_name(metamodel.solver) == "OLS"
        println("\nOLS solver: n coeffs for λ = 0: $(size(metamodel.coeffs))\n")
    end
    if solver_name(metamodel.solver) == "LASSO"
        println("\nLASSO solver: n coeffs for λ = $(round(metamodel.solver.λ, digits=4)): $(size(metamodel.coeffs))\n")
    end

    println("Nonzero elements: ", count(!iszero, metamodel.coeffs))
    println("Zero elements: ", count(iszero, metamodel.coeffs))

    println("\nmetrics:")
    println("RMSE:               $(round(rmse(y_test, y_pred), digits=4))")
    println("Q²:                $(round(q2(y_test, y_pred), digits=4))")

end
