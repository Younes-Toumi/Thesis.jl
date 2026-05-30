using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Distributions
using ParameterHandling
using LinearAlgebra

Random.seed!(42)

# ============================================================
# True model — operates on physical inputs (x1, x2)
# ============================================================
analytical_model(x1, x2) = x1 .+ x2 .+x1 .* x2 .+ 1
analytical_variance(x1, σ) = σ^2*(x1^2 + 2*x1 + 1) + x1^2 + 2*x1 - (x1 + 1)^2 + 1

const μ_FIXED = 0.0

const x1_UPPER = 1.5
const x1_LOWER = -0.5

const θ_σ_UPPER = 1.5
const θ_σ_LOWER = 0.5

inverse_cdf_x2(u2, θ_σ; μ=μ_FIXED) = quantile.(Normal.(μ, θ_σ), u2)


# ============================================================
function mc_augmented(n::Int)
    pts = rand(n, 3)

    u2_raw = pts[:, 1]
    x1_raw = x1_LOWER .+ (x1_UPPER - x1_LOWER) .* pts[:, 2]
    θσ_raw = θ_σ_LOWER .+ (θ_σ_UPPER - θ_σ_LOWER) .* pts[:, 3]

    return x1_raw, u2_raw, θσ_raw
end

# ============================================================
# Build initial design D₀ and evaluate true model
# ============================================================

x_names = [:x1, :u2, :θ_σ]

n_train = 20   # paper uses ~2(d+1)–5(d+1) for d=4 → 10–25 is reasonable

x1_train, u2_train, θ_σ_train = mc_augmented(n_train)

# Recover physical x2 via inverse CDF — this is what the true model sees
x2_train = inverse_cdf_x2(u2_train, θ_σ_train)

# Evaluate true model at physical inputs
y_train = analytical_model(x1_train, x2_train)


# Augmented training DataFrame: GP trains on (x1, u2, θ_σ) — NOT on x2 directly
data_aug_train = DataFrame(
    :x1  => x1_train,
    :u2  => u2_train,
    :θ_σ => θ_σ_train,
    :y   => y_train
)

using Plots, Distributions, Statistics


# ============================================================
# Grid over epistemic space (x1, θσ)
# ============================================================
n_x1 = 500
n_σ = 500

x1_grid = range(x1_LOWER, x1_UPPER, length=n_x1)
σ_grid  = range(θ_σ_LOWER, θ_σ_UPPER, length=n_σ)

MeanSurface = zeros(n_x1, n_σ)
VarSurface  = zeros(n_x1, n_σ)

# ============================================================
# Compute conditional moments
# ============================================================
for (i, x1_v) in enumerate(x1_grid)
    for (j, σ_v) in enumerate(σ_grid)

        Vy = analytical_variance(x1_v, σ_v)

        VarSurface[i, j]  = Vy
    end
end
# ============================================================
# Heatmap
# ============================================================
plt = heatmap(
    x1_grid,
    σ_grid,
    VarSurface,
    xlabel="x1",
    ylabel="θ_σ",
    c=:thermal,
    title="Conditional response variance V_y(x1, σ)",
    colorbar=true
)

# ── overlay the initial training points (x1, θ_σ columns from data_aug) ──────
scatter!(plt,
    data_aug_train.x1, data_aug_train.θ_σ;
    marker = :diamond, color = :cyan, ms = 5,
    label  = "Initial samples", markerstrokewidth=0
)

# ── mark true min and max in epistemic space ──────────────────────────────────
min_idx = argmin(VarSurface)
max_idx = argmax(VarSurface)

scatter!(plt,
    [x1_grid[min_idx[2]]], [σ_grid[min_idx[1]]];
    marker = :circle, color = :red, ms = 5, label = "True min"
)
scatter!(plt,
    [x1_grid[max_idx[2]]], [σ_grid[max_idx[1]]];
    marker = :rect, color = :red, ms = 5, label = "True max"
)

println("V range: [$(round(minimum(VarSurface), digits=2)),  $(round(maximum(VarSurface), digits=2))]")

display(plt)