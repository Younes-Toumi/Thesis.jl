using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames

Random.seed!(42)

physical_model = Model(
    df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2,
    :y
)

# 1. defining parametric inputs:
x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

# 2. decoupling inputs:
specs = InputSpec.([x1, x2])
x_names, w_names, u_names, v_names = spec_names(specs)
y_symbol = physical_model.name

# 3. building dataframes
n_train = 50
data_aug_train, data_phys_train = build_augmented_design(physical_model, specs, n_train)
data_aug_test, data_phys_test = build_augmented_design(physical_model, specs, 10000)

W_aug_test = Matrix(data_aug_test[:, w_names])
y_test = Vector(data_aug_test[:, y_symbol])


p_max = 4
bases = fill(SurrogateModelling.HermiteBasis(), length(w_names))
degree = TotalDegree(p_max)


# 4. training the metamodel
pce = SurrogateModelling.PolynomialChaosExpansion(data_aug_train, y_symbol, bases, degree; solver=LASSOSolver())
fit!(pce)

# 5. prediction capacity
y_pred = predict(pce, W_aug_test) 

rmse_val = rmse(y_test, y_pred)
q2_val = q2(y_test, y_pred)

println("PCE RMSE: $(round(rmse_val, digits=3))")
println("PCE Q2: $(round(q2_val, digits=3))")