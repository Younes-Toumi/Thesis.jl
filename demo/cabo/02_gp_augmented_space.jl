using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Distributions
using LinearAlgebra
using Statistics          # FIX 1: mean, var
using Printf              # FIX 2: @printf

Random.seed!(42)

# ============================================================
# Epistemic domain bounds  (FIX 3: define before PSO uses them)
# ============================================================
const μ_FIXED = 0.0

const x1_UPPER = 1.0
const x1_LOWER = -1.0

const θ_σ_UPPER = 1.5
const θ_σ_LOWER = 0.5
# ============================================================
# Feature column names expected by the GP
# ============================================================
const X_names = [:x1, :u2, :θ_σ]

# ============================================================
# True model — operates on physical inputs (x1, x2)
# ============================================================
analytical_model(x1, x2) = x1 .+ x2 .+x1 .* x2 .+ 1
analytical_variance(x1, σ) = σ^2*(x1^2 + 2*x1 + 1) + x1^2 + 2*x1 - (x1 + 1)^2 + 1
# ============================================================

# ============================================================
# Augmented-space helpers
# ============================================================

inverse_cdf_x2(u2, θ_σ; μ=μ_FIXED) = quantile.(Normal.(μ, θ_σ), u2)

function mc_augmented(n::Int)
    pts   = rand(n, 3)
    u2_raw = pts[:, 1]
    θ_σ_raw = θ_σ_LOWER .+ (θ_σ_UPPER - θ_σ_LOWER) .* pts[:, 2]
    x1_raw = x1_LOWER .+ (x1_UPPER - x1_LOWER) .* pts[:, 3]
    return x1_raw, u2_raw, θ_σ_raw
end

# ============================================================
# Initial training design D₀
# ============================================================
n_train = 20

x1_train, u2_train, θ_σ_train = mc_augmented(n_train)
x2_train = inverse_cdf_x2(u2_train, θ_σ_train)
y_train  = analytical_model(x1_train, x2_train)

data_aug_train = DataFrame(
    :x1  => x1_train,
    :u2  => u2_train,
    :θ_σ => θ_σ_train,
    :y   => y_train,
)

metamodel = GaussianProcess(data_aug_train, :y, kernel=GPSquaredExponential())
@time "fit!" fit!(metamodel)

# ============================================================
# Quick accuracy check on a held-out test set
# ============================================================
n_test = 100
x1_test, u2_test, θ_σ_test = mc_augmented(n_test)
x2_test  = inverse_cdf_x2(u2_test, θ_σ_test)
y_test_v = analytical_model(x1_test, x2_test)

data_aug_test = DataFrame(
    :x1  => x1_test,
    :u2  => u2_test,
    :θ_σ => θ_σ_test,
    :y   => y_test_v,
)

μ_test, σ_test = @time "predict:" predict(metamodel, Matrix(data_aug_test[:, X_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")