using UncertaintyQuantification
using SurrogateModelling
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra
Random.seed!(42)

run(`clear`)

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
# PCE Modelling Stuff
# ============================================================
# TODO: Automatic selection of bases depending on input distribution (or convert to sns)
# TODO: Compute the total possible degree depending on the available training samples

bases = [SurrogateModelling.LegendreBasis(), SurrogateModelling.LegendreBasis()] # can be automated based on input
degree = TotalDegree(15) # can be automated based on availabel samples

# λ sparcity penality: a compromise between approximation accuracy and model sparcity.

metamodels = [
    SurrogateModelling.PolynomialChaosExpansion(data_train, :y, bases, degree), # chooses Least Squares by default
    SurrogateModelling.PolynomialChaosExpansion(data_train, :y, bases, degree; solver=SurrogateModelling.OLSSolver()), # or not
    SurrogateModelling.PolynomialChaosExpansion(data_train, :y, bases, degree; solver=SurrogateModelling.LASSOSolver(λ=0.1)), # also the LASSO one
    SurrogateModelling.PolynomialChaosExpansion(data_train, :y, bases, degree; solver=SurrogateModelling.LASSOSolver()) # automatic lambda selection
]

messages = [
    "Normal PCE, default solver:",
    "Normal PCE, OLS solver:",
    "Sparce PCE, LASSO solver with specific λ:",
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
    println("MSE:               $(round(mse(y_test, y_pred), digits=5))")
    println("Q²:                $(round(q2(y_test, y_pred), digits=5))")

end
