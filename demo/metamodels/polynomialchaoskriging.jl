# ==============================================================================
# demo/03_polynomial_chaos_kriging.jl
#
# Demonstrates PolynomialChaosKriging (PCK): a low-degree PCE "trend" combined
# with a GP residual. Shows the case PCK exists for - a response surface a
# low-degree PCE alone cannot resolve, where PCK recovers GP-level accuracy.
# ==============================================================================

using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Statistics

Random.seed!(42)

# ================================================================================
# 1. Test problem: a function with LOCALISED, sharp structure
# ================================================================================
# A Gaussian bump is a good illustration: it needs high polynomial degree to
# resolve accurately (a low-degree PCE will smooth right over it), but a GP
# handles a localised feature like this natively via its kernel's lengthscale.
x1 = RandomVariable(Uniform(-3.2, 3.2), :x1)
x2 = RandomVariable(Uniform(-3.2, 3.2), :x2)

model = Model(
    df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2,
    :y
)

n_train, n_test = 30, 10000
data_train = sample([x1, x2], LatinHypercubeSampling(n_train))
data_test  = sample([x1, x2], LatinHypercubeSampling(n_test))

UncertaintyQuantification.evaluate!(model, data_train)
UncertaintyQuantification.evaluate!(model, data_test)

y_symbol = :y
bases = [LegendreBasis(), LegendreBasis()]
X_test = Matrix(data_test[:, [:x1, :x2]])
q2_of(m) = q2(data_test.y, predict(m, X_test))

# ================================================================================
# 2. A LOW-DEGREE PCE alone - deliberately too low to resolve the bump
# ================================================================================
pck_trend_degree = 4 # kept low on purpose

pce = SurrogateModelling.PolynomialChaosExpansion(
    data_train, y_symbol, bases, TotalDegree(pck_trend_degree);
    solver = LASSOSolver(),
)
fit!(pce)
println(
    "Pure PCE (degree $pck_trend_degree):           Q² = ", round(q2(data_test.y, predict(pce, X_test; mode=:mean)), digits=4),
    "  MSE = ", round(mse(data_test.y, predict(pce, X_test; mode=:mean)), digits=4)
    )

# ================================================================================
# 3. A pure GP, for comparison
# ================================================================================
gp = GaussianProcess(data_train, y_symbol; kernel_type=GPMatern52())
fit!(gp)
println(
    "Pure GP:                       Q² = ", round(q2(data_test.y, predict(gp, X_test; mode=:mean)), digits=4),
    "  MSE = ", round(mse(data_test.y, predict(gp, X_test; mode=:mean)), digits=4)
    )
# ================================================================================
# 4. Polynomial Chaos Kriging: same low-degree trend, PLUS a GP residual
# ================================================================================
# PCK's constructor takes an (unfitted or fitted) PolynomialChaosExpansion as
# its trend, and fits a GP on the RESIDUAL between that trend and the data.
# The trend degree is kept deliberately low here - PCK's whole point is that
# the polynomial only needs to capture broad/global structure; the GP residual
# picks up whatever the low-degree trend misses, including sharp local
# features that would otherwise require an infeasibly high PCE degree.
pce_trend = SurrogateModelling.PolynomialChaosExpansion(
    data_train, y_symbol, bases, TotalDegree(pck_trend_degree);
    solver = LASSOSolver(),
)

pck = PolynomialChaosKriging(data_train, y_symbol, pce_trend; kernel_type=GPMatern52())
fit!(pck)

println(
    "PCK (degree $pck_trend_degree + GP residual):  Q² = ", round(q2(data_test.y, predict(pck, X_test; mode=:mean)), digits=4),
    "  MSE = ", round(mse(data_test.y, predict(pck, X_test; mode=:mean)), digits=4)
    )

# ================================================================================
# 5. PCK prediction modes - mirrors the GP's mean/var/mean_and_var interface,
#    since PCK also carries predictive uncertainty (unlike a bare PCE).
# ================================================================================
print("\n")

μ_gp, σ_gp = predict(gp, X_test; mode=:mean_and_var)
println("GP predictive std range:  [", round(minimum(σ_gp), digits=4), ", ", round(maximum(σ_gp), digits=4), "]")

μ_pck, σ_pck = predict(pck, X_test; mode=:mean_and_var)
println("PCK predictive std range: [", round(minimum(σ_pck), digits=4), ", ", round(maximum(σ_pck), digits=4), "]")