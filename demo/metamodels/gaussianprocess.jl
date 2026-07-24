# ==============================================================================
# Demonstrates the GaussianProcess surrogate: construction, fitting, prediction,
# in-place evaluation (mean / variance / samples), and adaptive refitting.
# ==============================================================================

using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Statistics
using ParameterHandling
Random.seed!(42)

# ============================================================================================
# 1. Set up a simple 2D test problem
# ============================================================================================
# We use Ishigami-style inputs on a small domain purely for illustration -
# any UncertaintyQuantification.jl RandomVariable/IntervalVariable works here.
x1 = RandomVariable(Uniform(-1, 1), :x1)
x2 = RandomVariable(Uniform(-1, 1), :x2)

# a smooth, mildly nonlinear 2D test function
model = Model(df -> sin.(2 .* df.x1) .+ 0.5 .* df.x2 .^ 2, :y)

n_train, n_test = 40, 1000

data_train = sample([x1, x2], MonteCarlo(n_train))
data_test  = sample([x1, x2], MonteCarlo(n_test))

UncertaintyQuantification.evaluate!(model, data_train)   # writes the :y column into data_train
UncertaintyQuantification.evaluate!(model, data_test)

# ============================================================================================
# 2. Construct and fit a GaussianProcess
# ============================================================================================
# kernel_type selects the covariance family; ARD (one lengthscale per input dimension) 
# is used by default for every stationary kernel. learn_noise=false is for deterministic models

gp = GaussianProcess(data_train, :y; kernel_type = GPMatern52())
# gp = GaussianProcess(data_train, :y; kernel_type = GPSquaredExponential())
# gp = GaussianProcess(data_train, :y; kernel_type = GPSquaredExponential() + GPMatern52())


println("\nUnfitted ARD hyperparameters:")
println("    lengthscale = ", gp.θ0.lengthscale)
println("    variance    = ", gp.θ0.variance)

@time "fit!" fit!(gp)

println("\nFitted hyperparameters:")
println("    lengthscale = ", gp.θ.lengthscale)
println("    variance    = ", gp.θ.variance)

# ============================================================================================
# 3. Predict and evaluate fit quality
# ============================================================================================
X_test = Matrix(data_test[:, [:x1, :x2]])

# predict() is non-mutating and returns the requested quantities directly.
# Default mode=:mean_and_var returns (mean, std)
μ, σ = predict(gp, X_test)

println("\nMSE: ", round(mse(data_test.y, μ), digits=5))
println("Q²:  ", round(q2(data_test.y, μ), digits=5))

# predict() also supports cheaper single-quantity modes and raw variance:
μ_only  = predict(gp, X_test; mode=:mean)           # cheapest - never forms full covariance
σ2_only = predict(gp, X_test; mode=:var)            # marginal VARIANCE (not std)

# evaluate!() is the in-place, DataFrame-writing counterpart of predict() -
# useful when you want predictions to stay attached to the original dataset.
SurrogateModelling.evaluate!(gp, data_test; mode=:mean_and_var)
println("\ndata_test now has columns: ", names(data_test))

# ============================================================================================
# 4. Draw posterior sample paths
# ============================================================================================
# mode=:sample draws n_samples JOINTLY CORRELATED realizations from the
# posterior - unlike the other modes, this needs the full joint covariance,
# since samples must be correlated across query points to look like coherent
# functions rather than independent per-point noise.
n_samples = 5
samples = predict(gp, X_test[1:20, :]; mode=:sample, n_samples=n_samples)
println("\nSample matrix size (n_points × n_samples): ", size(samples))

# the same thing, written directly into a DataFrame as y_sample_1 ... y_sample_5:
data_small = data_test[1:20, :]
SurrogateModelling.evaluate!(gp, data_small; mode=:sample, n_samples=n_samples)
println("Sample columns added: ", filter(c -> occursin("sample", string(c)), names(data_small)))

# ============================================================================================
# 5. Adaptive refitting (warm-started, for use inside an adaptive-sampling loop)
# ============================================================================================
# refit!() appends new points and re-optimises hyperparameters starting from
# the CURRENT optimum (few restarts, small perturbations) - much cheaper than
# a full fit!() and appropriate once you already have a reasonable fit.
x_new = reshape([0.3, -0.2], 1, 2)
y_new = [sin(2*0.3) + 0.5*(-0.2)^2]

refit!(gp, x_new, y_new)
println("\nAfter refit!: n_train = ", size(gp.X, 1))
