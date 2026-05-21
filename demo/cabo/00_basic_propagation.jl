using UncertaintyQuantification
using Random
using DataFrames
Random.seed!(42)


# ============================================================
# Inputs + Model
# ============================================================

x1 = RandomVariable(Normal(0, 1), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.3, 1.8), :σ => 2.0)), :x2)
x3 = IntervalVariable(-0.5, 1.3, :x3)

X = [x1, x2, x3]

model = Model(
    rv -> rv.x1 .* ((rv.x2 .* rv.x2) .+ rv.x2 + cos.(π .* rv.x3) .- 7),
    :y
)

# ============================================================
# Sampling for: 1. train and 2. test #
# ============================================================

n_train = 15
design_train = MonteCarlo(n_train)

data_train = sample(X, design_train)

@time propagate_intervals!(model, data_train)
