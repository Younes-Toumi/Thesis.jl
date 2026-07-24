# ==============================================================================
# demo/02_polynomial_chaos_expansion.jl
#
# Demonstrates PolynomialChaosExpansion: truncation schemes, OLS vs LASSO
# solvers, automatic degree selection, and WHY the choice of truncation
# scheme matters for functions with genuine variable interactions.
# ==============================================================================

using UncertaintyQuantification

using SurrogateModelling
using SurrogateModelling: LegendreBasis, TotalDegree, QBall
using Random
using DataFrames
using Statistics

Random.seed!(42)

# =======================================================================================
# 1. Test problem: a function with REAL interaction structure
# =======================================================================================
# Each RandomVariable must be standard-normal-compatible for HermiteBasis, or
# Uniform(-1,1)-compatible for LegendreBasis (matching the basis's orthogonality
# weight). We use Uniform(-1,1) + LegendreBasis here.
x1 = RandomVariable(Uniform(-1, 1), :x1)
x2 = RandomVariable(Uniform(-1, 1), :x2)

# a function whose value depends on the PRODUCT of x1 and x2 - i.e. genuine
# interaction, not just two independent main effects
model = Model(df -> sin.(3 .* df.x1 .* df.x2) .+ 0.3 .* df.x1, :y)

n_train, n_test = 50, 1000
data_train = sample([x1, x2], LatinHypercubeSampling(n_train))
data_test  = sample([x1, x2], MonteCarlo(n_test))
UncertaintyQuantification.evaluate!(model, data_train)
UncertaintyQuantification.evaluate!(model, data_test)

y_symbol = :y
bases = [LegendreBasis(), LegendreBasis()]   # one basis per input dimension
X_test = Matrix(data_test[:, [:x1, :x2]])


# =======================================================================================
# 2. Truncation schemes compared at the SAME degree
# =======================================================================================
# TotalDegree admits any (i,j) with i+j <= p - including genuine interaction
# terms where BOTH i and j are large.
# QBall(p, q) with q<1 is a hyperbolic-cross scheme that PRUNES interaction
# terms aggressively - appropriate for functions dominated by main effects
# with only weak interactions, but a poor fit for a function like this one
# whose entire content IS the interaction term.
p_max = 6

degree_td   = TotalDegree(p_max)
degree_qb05 = QBall(p_max, 0.5)

pce_td   = SurrogateModelling.PolynomialChaosExpansion(data_train, y_symbol, bases, degree_td;   solver=LASSOSolver())
pce_qb05 = SurrogateModelling.PolynomialChaosExpansion(data_train, y_symbol, bases, degree_qb05; solver=LASSOSolver())

fit!(pce_td)
fit!(pce_qb05)

println("Truncation scheme comparison at p_max=$p_max:")
println("  TotalDegree:  Q² = ", round(q2_of(pce_td),   digits=4), "  (n_terms=", n_terms(degree_td,   2), ")")
println("  QBall(q=0.5): Q² = ", round(q2_of(pce_qb05), digits=4), "  (n_terms=", n_terms(degree_qb05, 2), ")")
println("""
  -> QBall's sparser basis has fewer candidate terms with both x1 and x2 at
     nonzero degree simultaneously, so it will typically fit worse here.
     QBall is the better choice for high-dimensional problems with weak
     interactions""")

# =======================================================================================
# 3. OLS vs LASSO
# =======================================================================================
# OLS fits ALL candidate coefficients with no regularisation - fine when
# n_train comfortably exceeds the number of terms, prone to severe overfitting
# otherwise (see the affordability discussion in part 4).
pce_ols   = SurrogateModelling.PolynomialChaosExpansion(data_train, y_symbol, bases, degree_td; solver=OLSSolver())
pce_lasso = SurrogateModelling.PolynomialChaosExpansion(data_train, y_symbol, bases, degree_td; solver=LASSOSolver())

fit!(pce_ols)
fit!(pce_lasso)

println("Solver comparison (TotalDegree, p_max=$p_max):")
println("  OLS:   Q² = ", round(q2_of(pce_ols),   digits=4),
        "  (nonzero coeffs: ", count(!iszero, pce_ols.coeffs),   "/", length(pce_ols.coeffs),   ")")
println("  LASSO: Q² = ", round(q2_of(pce_lasso), digits=4),
        "  (nonzero coeffs: ", count(!iszero, pce_lasso.coeffs), "/", length(pce_lasso.coeffs), ")")

# =======================================================================================
# 4. How much data does a given degree actually need?
# =======================================================================================
# n_terms grows combinatorially with degree and dimension. The rule of thumb
# is n_train >~ 2-3x n_terms for a stable fit - below that, OLS in particular
# can produce wildly overfit, useless coefficients.
println("\nAffordability check (d=2 dimensions):")
for p in [2, 4, 6, 8, 10]
    n_p = n_terms(TotalDegree(p), 2)
    println("  p=$p: n_terms=$n_p  ", n_p <= n_train/2 ? "(comfortable at n_train=$n_train)" : "(risky at n_train=$n_train)")
end

# max_affordable_degree answers this directly: the largest degree whose term
# count stays within oversample_factor x n_train.
p_afford = max_affordable_degree(p -> TotalDegree(p), 2, n_train)
println("\nmax_affordable_degree(TotalDegree, d=2, n=$n_train) = $p_afford")

# =======================================================================================
# 5. Automatic degree selection
# =======================================================================================
# Instead of picking p_max by hand, pass a FUNCTION p -> degree and let the
# constructor sweep degrees, capped by affordability, picking by LOO error.
pce_auto = SurrogateModelling.PolynomialChaosExpansion(
    data_train, y_symbol, bases, TotalDegree(p_afford); solver = LASSOSolver(),
)

fit!(pce_auto)

q2_val =  q2(data_test.y, predict(pce_auto, X_test))

println("\nAuto-selected PCE: Q² = ", round(q2_val, digits=4))
