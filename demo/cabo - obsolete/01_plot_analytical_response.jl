using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Distributions
using ParameterHandling
using LinearAlgebra

Random.seed!(42)



# G-function parameters
const α_g = [2.0 3.0 1.0 4.0;
             3.0 2.0 4.0 1.0]'

const β_g = [-0.5 0.5 -0.5 0.5;
             -0.5 -0.5 0.5 0.5]'

const c_g = [1.0, -1.5, -1.5, 2.0]'

"""
G-function: g(x1, x2) = Σᵢ cᵢ exp(-αᵢ₁(x1-βᵢ₁)² - αᵢ₂(x2-βᵢ₂)²)
"""
function g_function(x1::Float64, x2::Float64)
    result = 0.0
    for i in 1:4
        result += c_g[i] * exp(-α_g[i,1] * (x1 - β_g[i,1])^2 - α_g[i,2] * (x2 - β_g[i,2])^2)
    end
    return result
end

function g_expected(μ1::Float64, μ2::Float64, σ::Float64)

    result = 0.0
    σ² = σ^2

    for i in 1:4

        α1 = α_g[i,1]
        α2 = α_g[i,2]

        β1 = β_g[i,1]
        β2 = β_g[i,2]

        # E[exp(-α(X-β)^2)]
        term1 =
            exp(
                -α1 * (μ1 - β1)^2 /
                (1 + 2 * α1 * σ²)
            ) /
            sqrt(1 + 2 * α1 * σ²)

        term2 =
            exp(
                -α2 * (μ2 - β2)^2 /
                (1 + 2 * α2 * σ²)
            ) /
            sqrt(1 + 2 * α2 * σ²)

        result += c_g[i] * term1 * term2
    end

    return result
end


# ============== Case 1: Imprecise Means Only ==============
# x1 ~ N(θ1, 0.1²), x2 ~ N(θ2, 0.1²)
# θ1, θ2 ∈ [-1.5, 1.5]

function analytical_model(u::Vector{Float64}, θ::Vector{Float64}, σ)
    # u = [u1, u2] are standard normal samples
    # θ = [θ1, θ2] are the imprecise means
    u1, u2      = u[1], u[2]
    θ_μ1, θ_μ2  = θ[1], θ[2]

    x1 = inverse_cdf_x1(u1, θ_μ1; σ=σ)
    x2 = inverse_cdf_x2(u2, θ_μ2; σ=σ)

    return g_function(x1, x2)
end

const σ_FIXED = 0.1

const θμ1_UPPER = 1.5
const θμ1_LOWER = -1.5

const θμ2_UPPER = 1.5
const θμ2_LOWER = -1.5

inverse_cdf_x1(u1, θ_μ; σ=σ_FIXED) = quantile.(Normal.(θ_μ, σ), u1)
inverse_cdf_x2(u2, θ_μ; σ=σ_FIXED) = quantile.(Normal.(θ_μ, σ), u2)


# ============================================================
function mc_augmented(n::Int)
    pts = rand(n, 4)

    u1_raw = pts[:, 1]
    u2_raw = pts[:, 2]
    
    μ1_raw = θμ1_LOWER .+ (θμ1_UPPER - θμ1_LOWER) .* pts[:, 3]
    μ2_raw = θμ2_LOWER .+ (θμ2_UPPER - θμ2_LOWER) .* pts[:, 4]

    return u1_raw, u2_raw, μ1_raw, μ2_raw
end

# ============================================================
# Build initial design D₀ and evaluate true model
# ============================================================

x_names = [:u1, :u2, :θ_μ1, :θ_μ2]

n_train = 20   # paper uses ~2(d+1)–5(d+1) for d=4 → 10–25 is reasonable

u1_train, u2_train, θ_μ1_train, θ_μ2_train = mc_augmented(n_train)

# Recover physical x2 via inverse CDF — this is what the true model sees
x1_train = inverse_cdf_x1(u1_train, θ_μ1_train)
x2_train = inverse_cdf_x2(u2_train, θ_μ2_train)

# Evaluate true model at physical inputs
y_train = g_function.(x1_train, x2_train)


# Augmented training DataFrame: GP trains on (x1, u2, θ_σ) — NOT on x2 directly
data_aug_train = DataFrame(
    :u1  => u1_train,
    :u2  => u2_train,
    :θ_μ1 => θ_μ1_train,
    :θ_μ2 => θ_μ2_train,
    :y   => y_train
)

using Plots, Distributions, Statistics

# ============================================================
# Grid over epistemic space (x1, θσ)
# ============================================================
n_μ1 = 2000
n_μ2 = 2000

μ1_grid = range(θμ1_LOWER, θμ1_UPPER, length=n_μ1)
μ2_grid  = range(θμ2_LOWER, θμ2_UPPER, length=n_μ2)

MeanSurface = zeros(n_μ1, n_μ2)

# ============================================================
# Compute response
# ============================================================
for (i, μ1_v) in enumerate(μ1_grid)
    for (j, μ2_v) in enumerate(μ2_grid)
        MeanSurface[j, i]  = g_expected(μ1_v, μ2_v, σ_FIXED)
    end
end

# ============================================================
# Heatmap
# ============================================================
plt = heatmap(
    μ1_grid,
    μ2_grid,
    MeanSurface,
    xlabel="μ1",
    ylabel="μ2",
    c=:thermal,
    title="expected response function: E[g(x1, x2)]",
    colorbar=true
)

# ── overlay the initial training points (μ1, μ2 columns from data_aug) ──────
scatter!(plt,
    data_aug_train.θ_μ1, data_aug_train.θ_μ2;
    marker = :diamond, color = :cyan, ms = 5,
    label  = "Initial samples", markerstrokewidth=0
)

# ── mark true min and max in epistemic space ──────────────────────────────────
min_idx = argmin(MeanSurface)
max_idx = argmax(MeanSurface)

x_min, y_min, z_min = μ1_grid[min_idx[2]], μ1_grid[min_idx[1]], minimum(MeanSurface)
x_max, y_max, z_max = μ1_grid[max_idx[2]], μ1_grid[max_idx[1]], maximum(MeanSurface)

dy = 0.2

scatter!(plt,
    [μ1_grid[min_idx[2]]], [μ2_grid[min_idx[1]]];
    marker = :circle, color = :red, ms = 5, label = "True min"
)
annotate!(
    x_min, y_min + dy,
    text("($(round(x_min, digits=3)), $(round(y_min, digits=3)), $(round(z_min, digits=3)))", :black, 8)
)


scatter!(plt,
    [μ1_grid[max_idx[2]]], [μ2_grid[max_idx[1]]];
    marker = :rect, color = :red, ms = 5, label = "True max"
)

annotate!(
    x_max, y_max + dy,
    text("($(round(x_max, digits=3)), $(round(y_max, digits=3)), $(round(z_max, digits=3)))", :black, 8)
)

println("y range: [$(round(minimum(MeanSurface), digits=3)),  $(round(maximum(MeanSurface), digits=3))]")
print(round.([μ1_grid[min_idx[2]], μ2_grid[min_idx[1]]], digits=3))
print("\n")
print(round.([μ1_grid[max_idx[2]], μ2_grid[max_idx[1]]], digits=3))

display(plt)
print("\n")
savefig(plt, "./assets/mean_response_function.png")