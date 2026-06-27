# ============================================================
# PCK usage example — Ishigami, matching your existing GP-comparison pattern
#
# NOTE: degree_type / basis_type / solver_type keyword names below are
# inferred from your module's exports (TotalDegree, LegendreBasis,
# LASSOSolver). Adjust to match your actual PolynomialChaosExpansion
# constructor signature if these don't line up exactly.
# ============================================================

using SurrogateModelling
using UncertaintyQuantification: sample, evaluate!
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra

# ============================================================
# Inputs + Model
# ============================================================
x1 = RandomVariable.(Uniform(-pi, pi), :x1)
x2 = RandomVariable.(Uniform(-pi, pi), :x2)
x3 = RandomVariable.(Uniform(-pi, pi), :x3)
X = [x1, x2, x3]

model = model_ishigami

# ============================================================
# Sampling for: 1. train and 2. test
# ============================================================
n_train, n_test = 50, 10000
design_train = LatinHypercubeSampling(n_train)
design_test  = LatinHypercubeSampling(n_test)

data_train = sample(X, design_train)
data_test  = sample(X, design_test)

evaluate!(model, data_train)
evaluate!(model, data_test)

X_names = [x.name for x in X]
X_test  = data_test[:, X_names]
y_test  = data_test[:, :y]

# ============================================================
# Compare GP, PCE, and PCK on the same data
# ============================================================

# ── 1. Plain GP (for reference) ──────────────────────────────
gp = GaussianProcess(data_train, :y; kernel_type = GPMatern52())

println("\n============================================================")
println("GP (ZeroMean): kernel = GPMatern52")
println("============================================================\n")
@time "fit!" fit!(gp)
μ_gp, σ_gp = @time "predict" predict(gp, Matrix(X_test))
println("RMSE: $(round(rmse(y_test, μ_gp), digits=5))")
println("Q²:   $(round(q2(y_test, μ_gp), digits=5))")

# ── 2. Plain PCE (for reference) ─────────────────────────────
pce = PolynomialChaosExpansion(data_train, :y;
    degree_type = TotalDegree(5),
    basis_type  = LegendreBasis(),    # Legendre ↔ Uniform inputs (Wiener-Askey)
    solver_type = LASSOSolver()
)

println("\n============================================================")
println("PCE: TotalDegree(5), LegendreBasis, LASSOSolver")
println("============================================================\n")
@time "fit!" fit!(pce)
μ_pce = @time "predict" predict(pce, Matrix(X_test))
println("RMSE: $(round(rmse(y_test, μ_pce), digits=5))")
println("Q²:   $(round(q2(y_test, μ_pce), digits=5))")

# ── 3. Sequential PCK: same PCE trend + GP residual ──────────
# Build a FRESH unfitted PCE (PCK fits it internally — don't reuse the
# already-fitted `pce` object above)
pce_trend = PolynomialChaosExpansion(data_train, :y;
    degree_type = TotalDegree(5),
    basis_type  = LegendreBasis(),
    solver_type = LASSOSolver()
)

pck = PolynomialChaosKriging(data_train, :y, pce_trend;
    kernel_type = GPMatern52(),
    learn_noise = false
)

println("\n============================================================")
println("PCK (sequential): $(model_name(pck))")
println("============================================================\n")
@time "fit!" fit!(pck)
μ_pck, σ_pck = @time "predict" predict(pck, Matrix(X_test))
println("RMSE: $(round(rmse(y_test, μ_pck), digits=5))")
println("Q²:   $(round(q2(y_test, μ_pck), digits=5))")

# ============================================================
# Summary
# ============================================================
println("\n" * "="^60)
println("Summary")
println("="^60)
println("GP   Q² = $(round(q2(y_test, μ_gp),  digits=4))")
println("PCE  Q² = $(round(q2(y_test, μ_pce), digits=4))")
println("PCK  Q² = $(round(q2(y_test, μ_pck), digits=4))")