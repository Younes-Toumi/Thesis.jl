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
const X3_LB, X3_UB = -0.5,  1.3   # x3 interval
const TH_LB, TH_UB = -1.3,  1.8   # θ_μ p-box parameter

# ============================================================
# Feature column names expected by the GP
# ============================================================
const X_names = [:x1, :u2, :x3, :θ_μ]

# ============================================================
# True model and analytical variance (for validation only)
# ============================================================
analytical_model(x1, x2, x3) =
    x1 .* (x2.^2 .+ x2 .+ cos.(π .* x3) .- 7)

analytical_variance(x3, μ) =
    μ.^4 .+ 2μ.^3 .+ 2μ.^2 .* cos.(π.*x3) .+ 11μ.^2 .+
    2μ .* cos.(π.*x3) .+ 10μ .+ cos.(π.*x3).^2 .- 6cos.(π.*x3) .+ 45

# ============================================================
# Augmented-space helpers
# ============================================================
const σ_FIXED = 2.0

inverse_cdf_x2(u2, θ_μ; σ=σ_FIXED) = quantile.(Normal.(θ_μ, σ), u2)

function mc_augmented(n::Int)
    pts   = rand(n, 4)
    x1_raw = quantile.(Normal(0, 1), pts[:, 1])
    u2_raw = pts[:, 2]
    x3_raw = X3_LB .+ (X3_UB - X3_LB) .* pts[:, 3]
    θμ_raw = TH_LB .+ (TH_UB - TH_LB) .* pts[:, 4]
    return x1_raw, u2_raw, x3_raw, θμ_raw
end

# ============================================================
# Initial training design D₀
# ============================================================
n_train = 100

x1_train, u2_train, x3_train, θ_μ_train = mc_augmented(n_train)
x2_train = inverse_cdf_x2(u2_train, θ_μ_train)
y_train  = analytical_model(x1_train, x2_train, x3_train)

data_aug_train = DataFrame(
    :x1  => x1_train,
    :u2  => u2_train,
    :x3  => x3_train,
    :θ_μ => θ_μ_train,
    :y   => y_train,
)

metamodel = GaussianProcess(data_aug_train, :y, kernel=GPMatern52())
@time "fit!" fit!(metamodel)

# ============================================================
# Quick accuracy check on a held-out test set
# ============================================================
n_test = 100
x1_test, u2_test, x3_test, θ_μ_test = mc_augmented(n_test)
x2_test  = inverse_cdf_x2(u2_test, θ_μ_test)
y_test_v = analytical_model(x1_test, x2_test, x3_test)

data_aug_test = DataFrame(
    :x1  => x1_test,
    :u2  => u2_test,
    :x3  => x3_test,
    :θ_μ => θ_μ_test,
    :y   => y_test_v,
)

μ_test, σ_test = predict(metamodel, data_aug_test[:, X_names])

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2( data_aug_test.y, μ_test), digits=5))")