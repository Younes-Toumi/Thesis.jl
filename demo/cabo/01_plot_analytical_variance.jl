using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Distributions
using ParameterHandling
using LinearAlgebra

Random.seed!(42)

# ============================================================
# True model — operates on physical inputs (x1, x2, x3)
# ============================================================
analytical_model(x1, x2, x3) = x1 .* (x2.^2 .+ x2 .+ cos.(π .* x3) .- 7)
analytical_variance(x3, μ) = μ.^4 .+ 2 .* μ.^3 .+ 2 .* μ.^2 .* cos.(pi .* x3) .+ 11 .* μ.^2 .+ 2 .* μ .* cos.(pi .* x3) .+ 10 .* μ .+ cos.(pi .* x3).^2 .- 6 .* cos.(pi .* x3) .+ 45

# ============================================================
# Augmented space definition
#
#   x1  ~ N(0,1)                precise PDF      → sample normally
#   u2  ~ U(0,1)                auxiliary        → inverse CDF surrogate
#   x3  ∈ [-0.5, 1.3]           interval         → sample uniformly over bounds
#   θ_μ ∈ [-1.3,  1.8]          p-box parameter  → sample uniformly over bounds
#
#   Physical x2 is recovered as: x2 = F⁻¹(u2; θ_μ, σ=2.0)
#   This is the Rosenblatt transform — makes x2 a deterministic function of (u2, θ_μ)
# ============================================================
const σ_FIXED = 2.0  # known σ of the p-box; only μ is uncertain

"""
    inverse_cdf_x2(u2, θ_μ; σ=σ_FIXED)

Recover the physical x2 from the auxiliary uniform u2 and the p-box parameter θ_μ
via the inverse Normal CDF.
"""
inverse_cdf_x2(u2, θ_μ; σ=σ_FIXED) = quantile.(Normal.(θ_μ, σ), u2)

# ============================================================
# Latin Hypercube sampling in augmented space
# (manual LHS across 4 dimensions: x1, u2, x3, θ_μ)
# ============================================================
function mc_augmented(n::Int)
    pts = rand(n, 4)

    x1_raw = quantile(Normal(0, 1), pts[:, 1])
    u2_raw = pts[:, 2]
    x3_raw = -0.5 .+ (1.3 - -0.5) .* pts[:, 3]
    θμ_raw = -1.3 .+ (1.8 - -1.3) .* pts[:, 4]

    return x1_raw, u2_raw, x3_raw, θμ_raw
end

# ============================================================
# Build initial design D₀ and evaluate true model
# ============================================================

x_names = [:x1, :u2, :x3, :θ_μ]

n_train = 30   # paper uses ~2(d+1)–5(d+1) for d=4 → 10–25 is reasonable

x1_train, u2_train, x3_train, θ_μ_train = mc_augmented(n_train)

# Recover physical x2 via inverse CDF — this is what the true model sees
x2_train = inverse_cdf_x2(u2_train, θ_μ_train)

# Evaluate true model at physical inputs
y_train = analytical_model(x1_train, x2_train, x3_train)


# Augmented training DataFrame: GP trains on (x1, u2, x3, θ_μ) — NOT on x2 directly
data_aug_train = DataFrame(
    :x1  => x1_train,
    :u2  => u2_train,
    :x3  => x3_train,
    :θ_μ => θ_μ_train,
    :y   => y_train
)

using Plots, Distributions, Statistics


# ============================================================
# Grid over epistemic space (x3, θμ)
# ============================================================
n_x3 = 100
n_μ = 100

x3_grid = range(-0.5, 1.3, length=n_x3)
μ_grid  = range(-1.3, 1.8, length=n_μ)

MeanSurface = zeros(n_x3, n_μ)
VarSurface  = zeros(n_x3, n_μ)

# ============================================================
# Compute conditional moments
# ============================================================
for (i, x3_v) in enumerate(x3_grid)
    for (j, μ_v) in enumerate(μ_grid)

        Vy = analytical_variance(x3_v, μ_v)

        VarSurface[i, j]  = Vy
    end
end
# ============================================================
# Heatmap
# ============================================================
plt = heatmap(
    x3_grid,
    μ_grid,
    VarSurface,
    xlabel="x3",
    ylabel="θ_μ",
    c=:thermal,
    title="Conditional response variance V_y(x3, μ)",
    colorbar=true
)

# ── overlay the initial training points (x3, θ_μ columns from data_aug) ──────
scatter!(plt,
    data_aug_train.x3, data_aug_train.θ_μ;
    marker = :diamond, color = :cyan, ms = 5,
    label  = "Initial samples", markerstrokewidth=0
)

# ── mark true min and max in epistemic space ──────────────────────────────────
min_idx = argmin(VarSurface)
max_idx = argmax(VarSurface)

scatter!(plt,
    [x3_grid[min_idx[2]]], [μ_grid[min_idx[1]]];
    marker = :circle, color = :red, ms = 5, label = "True min"
)
scatter!(plt,
    [x3_grid[max_idx[2]]], [μ_grid[max_idx[1]]];
    marker = :rect, color = :red, ms = 5, label = "True max"
)

println("V range: [$(round(minimum(VarSurface), digits=2)),  $(round(maximum(VarSurface), digits=2))]")

display(plt)
