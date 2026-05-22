using UncertaintyQuantification
using Random
using DataFrames
Random.seed!(42)


# ============================================================
# Inputs + Model
# ============================================================

x1 = IntervalVariable(-1.0, 1.0, :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => 0, :σ => Interval(0.5, 1.5))), :x2)

X = [x1, x2]

model = Model(
    rv -> rv.x1 .* rv.x2 .+ rv.x1 .+ rv.x2 .+ 1,
    :y
)

# ============================================================
# Sampling for: 1. train and 2. test #
# ============================================================

n_train = 15
design_train = MonteCarlo(n_train)

data_train = sample(X, design_train)

@time propagate_intervals!(model, data_train)
