using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
Random.seed!(42)

# ── Variables & Model ─────────────────────────────────────────────
x1 = RandomVariable.(Uniform(-5, 5), :x1)
x2 = RandomVariable.(Uniform(-5, 5), :x2)
X  = [x1, x2]

model = Model(
    rv -> rv.x1.^2 .+ rv.x2.^2,
    :y
)

# ── Sampling ──────────────────────────────────────────────────────
design_train = LatinHypercubeSampling(100)
data_train   = sample(X, design_train)
evaluate!(model, data_train)

x_names = propertynames(data_train[:, Not(:y)])

# ── Scaling ───────────────────────────────────────────────────────
X_train = Matrix(data_train[:, x_names])
y_train = Vector(data_train[:, :y])

pipeline = fit_pipeline(
    X_train,
    y_train,
    MinMaxScaler,
    MinMaxScaler
)

X_train_scaled = transform_input(pipeline, X_train)
y_train_scaled = transform_output(pipeline, y_train)

println("input transformed min: $(round.(pipeline.input_scaler.min, digits=3))")
println("input transformed max: $(round.(pipeline.input_scaler.max, digits=3))")