using SurrogateModelling
using UncertaintyQuantification
using UncertaintyQuantification: sample, wrap, isimprecise, middle, minimize, RobustOrthoMADS, bounds, map_to_precise_inputs
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra


# =======================================================================
# Step 1. Augmented Space Setup: Four Gaussian Mixture Function g(x1, x2)
# =======================================================================

x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input
physical_model = model_gfunction
print("Gaussian Mixture...\n")

function pce_test(physical_model, specs; n_train=50)

    _, w_names, _, _ = spec_names(specs)
    y_symbol = physical_model.name

    p_max = 4
    solver = SurrogateModelling.LASSOSolver
    bases = [SurrogateModelling.HermiteBasis() for _ in w_names]
    degree = QBall(p_max, 0.50)


    data_aug_test, _ = build_augmented_design(physical_model, specs, 1001)
    W_test = Matrix(data_aug_test[:, w_names])
    y_test = data_aug_test[:, y_symbol]

    data_aug_train, _ = build_augmented_design(physical_model, specs, 50)


    pce         = SurrogateModelling.PolynomialChaosExpansion(data_aug_train, y_symbol, bases, degree; solver=solver())
    fit!(pce)

    y_pred = predict(pce, W_test)

    q2_val = q2(y_test, y_pred)

    print("for p_max = $p_max, n_train = $n_train -> Q2: $(round(q2_val, digits=3))")

end

pce_test(physical_model, specs; n_train=50)