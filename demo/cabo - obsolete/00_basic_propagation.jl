using UncertaintyQuantification
using Random
using DataFrames
Random.seed!(42)


# ============================================================
# Inputs + Model
# ============================================================

x1 = IntervalVariable(-1.0, 1.0, :x1)
x2 = IntervalVariable(-1.0, 1.0, :x2)

# x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
# x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

X = [x1, x2]

model = Model(
    rv -> rv.x1 .+ rv.x2,
    :y
)

# ============================================================
# Sampling for: 1. train and 2. test #
# ============================================================

n_train = 15
design_train = MonteCarlo(n_train)

data_train = sample(X, design_train)

@time propagate_intervals!(model, data_train)


using Plots

x1_width = [(data_train.x1[i].ub - data_train.x1[i].lb) for i in 1:n_train]
x2_width = [(data_train.x2[i].ub - data_train.x2[i].lb) for i in 1:n_train]
y_width  = [(data_train.y[i].ub  - data_train.y[i].lb)  for i in 1:n_train]

scatter(
    x1_width .+ x2_width,
    y_width,
    xlabel = "Input uncertainty",
    ylabel = "Output uncertainty",
    label = ""
)