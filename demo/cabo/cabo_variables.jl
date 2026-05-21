using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Distributions
using ParameterHandling
using LinearAlgebra
using Plots

Random.seed!(42)

# ============================================================
# True model — operates on physical inputs (x1, x2, x3)
# ============================================================
true_model(x1, x2, x3) = x1 .* (x2.^2 .+ x2 .+ cos.(π .* x3) .- 7)

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
function lhs_augmented(n::Int)
    # Build permutation matrix — standard LHS construction
    perms = hcat([shuffle(0:n-1) for _ in 1:4]...)   # n × 4, each col a permutation
    U     = rand(n, 4)                                 # uniform jitter within strata
    pts   = (perms .+ U) ./ n                          # ∈ (0,1)^4

    # Map each column to its physical domain
    x1_raw = quantile(Normal(0, 1), pts[:, 1])         # N(0,1) via inverse CDF
    u2_raw = pts[:, 2]                                  # already U(0,1)
    x3_raw = -0.5 .+ 1.8 .* pts[:, 3]                 # U(-0.5, 1.3)
    θμ_raw = -1.3 .+ (1.8 - -1.3) .* pts[:, 4]                 # U(-1.3, 1.8)

    return x1_raw, u2_raw, x3_raw, θμ_raw
end

# ============================================================
# Build initial design D₀ and evaluate true model
# ============================================================
n_train = 1_000   # paper uses ~2(d+1)–5(d+1) for d=4 → 10–25 is reasonable

x1, u2, x3, θ_μ = lhs_augmented(n_train)

# Recover physical x2 via inverse CDF — this is what the true model sees
x2 = inverse_cdf_x2(u2, θ_μ)

# Evaluate true model at physical inputs
y = true_model(x1, x2, x3)

# ------------------------------------------------------------
# x1 ~ Normal(0,1)
# ------------------------------------------------------------
xgrid1 = range(-4, 4, length=500)

p1 = histogram(
    x1,
    bins=50,
    normalize=:pdf,
    alpha=0.5,
    label="Samples",
    title="x1 ~ Normal(0,1)",
    xlabel="x1",
    ylabel="Density"
)

plot!(p1, xgrid1, pdf.(Normal(0,1), xgrid1),
      lw=2, label="True PDF")

# ------------------------------------------------------------
# x2 from p-box
# ------------------------------------------------------------
xgrid2 = range(minimum(x2)-1, maximum(x2)+1, length=500)

p2 = histogram(
    x2,
    bins=50,
    normalize=:pdf,
    alpha=0.5,
    label="Generated x2",
    title="x2 from p-box",
    xlabel="x2",
    ylabel="Density"
)

plot!(p2, xgrid2,
      pdf.(Normal(-1.3, 2.0), xgrid2),
      lw=2,
      label="μ=-1.3")

plot!(p2, xgrid2,
      pdf.(Normal(1.8, 2.0), xgrid2),
      lw=2,
      label="μ=1.8")

# ------------------------------------------------------------
# x3 ~ Uniform(-0.5,1.3)
# ------------------------------------------------------------
xgrid3 = range(-0.5, 1.3, length=500)

p3 = histogram(
    x3,
    bins=50,
    normalize=:pdf,
    alpha=0.5,
    label="Samples",
    title="x3 ~ Uniform(-0.5,1.3)",
    xlabel="x3",
    ylabel="Density"
)

plot!(p3, xgrid3,
      pdf.(Uniform(-0.5,1.3), xgrid3),
      lw=2,
      label="True PDF")

# ------------------------------------------------------------
# θμ ~ Uniform(-1.3,1.8)
# ------------------------------------------------------------
xgrid4 = range(-1.3, 1.8, length=500)

p4 = histogram(
    θ_μ,
    bins=50,
    normalize=:pdf,
    alpha=0.5,
    label="Samples",
    title="θμ ~ Uniform(-1.3,1.8)",
    xlabel="θμ",
    ylabel="Density"
)

plot!(p4, xgrid4,
      pdf.(Uniform(-1.3,1.8), xgrid4),
      lw=2,
      label="True PDF")

plot(p1, p2, p3, p4, layout=(2,2), size=(1000,700))