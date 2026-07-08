# ============================================================
# PCK usage example — Ishigami, matching your existing GP-comparison pattern
#
# NOTE: degree_type / basis_type / solver_type keyword names below are
# inferred from your module's exports (TotalDegree, HermiteBasis,
# LASSOSolver). Adjust to match your actual PolynomialChaosExpansion
# constructor signature if these don't line up exactly.
# ============================================================

using SurrogateModelling
using UncertaintyQuantification: sample, evaluate!
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra


Random.seed!(42)

# ============================================================
# Inputs + Model
# ============================================================
x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)


x = RandomVariable(Normal(μ, σ), :x)
x = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x)
x = IntervalVariable(lb, ub, :x)

specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input
physical_model = model_gfunction
print("Gaussian Mixture...\n")


# Ishigami, all-interval inputs
# x1 = IntervalVariable(-pi, pi, :x1)
# x2 = IntervalVariable(-pi, pi, :x2)
# x3 = IntervalVariable(-pi, pi, :x3)

# specs = InputSpec.([x1, x2, x3])
# physical_model = model_ishigami        # ← the actual Ishigami model, not g_function

print("Ishigami...\n")




x_names, w_names, u_names, v_names = spec_names(specs)
y_symbol = physical_model.name

n_train, n_test = 50, 1001
data_aug_train, _ = build_augmented_design(physical_model, specs, n_train)
data_aug_test,  _ = build_augmented_design(physical_model, specs, n_test)

W_test = data_aug_test[:, w_names]
y_test = data_aug_test[:, y_symbol]

# ============================================================
# Compare GP, PCE, and PCK on the same data
# ============================================================
p_max = 3

bases = fill(SurrogateModelling.HermiteBasis(), length(w_names))
degree = TotalDegree(p_max) # can be automated based on availabel samples


# ── 1. Plain GP (for reference) ──────────────────────────────
gp = GaussianProcess(data_aug_train, :y; kernel_type = GPMatern52())

println("\n============================================================")
println("GP (ZeroMean): kernel = GPMatern52")
println("============================================================\n")
@time "fit!" fit!(gp)
μ_gp, σ_gp = @time "predict" predict(gp, Matrix(W_test))
println("RMSE: $(round(rmse(y_test, μ_gp), digits=5))")
println("Q²:   $(round(q2(y_test, μ_gp), digits=5))")

# ── 2. Plain PCE (for reference) ─────────────────────────────
pce = SurrogateModelling.PolynomialChaosExpansion(
    data_aug_train, 
    :y,
    bases,
    degree;
    solver = SurrogateModelling.LASSOSolver()
)

println("\n============================================================")
println("PCE: TotalDegree($p_max), HermiteBasis, LASSOSolver")
println("============================================================\n")
@time "fit!" fit!(pce)
μ_pce = @time "predict" predict(pce, Matrix(W_test))
println("RMSE: $(round(rmse(y_test, μ_pce), digits=5))")
println("Q²:   $(round(q2(y_test, μ_pce), digits=5))")

# ── 3. Sequential PCK: same PCE trend + GP residual ──────────
# Build a FRESH unfitted PCE (PCK fits it internally — don't reuse the
# already-fitted `pce` object above)
pce_trend = SurrogateModelling.PolynomialChaosExpansion(
    data_aug_train, 
    :y,
    bases,
    degree;
    solver = SurrogateModelling.LASSOSolver()
)

pck = SurrogateModelling.PolynomialChaosKriging(data_aug_train, :y, pce_trend;
    kernel_type = GPMatern52(),
    learn_noise = false
)

println("\n============================================================")
println("PCK (sequential): $(model_name(pck))")
println("============================================================\n")
@time "fit!" fit!(pck)
μ_pck, σ_pck = @time "predict" predict(pck, Matrix(W_test))
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