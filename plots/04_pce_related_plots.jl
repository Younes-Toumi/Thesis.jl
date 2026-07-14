using Random
using Plots
using Distributions
using UncertaintyQuantification
using UncertaintyQuantification: sample
using SurrogateModelling
using DataFrames
using Statistics

gr()
Random.seed!(42)

# ---------------------------------------------------------
# Model definition (Forrester-style function on [-1,1])
# ---------------------------------------------------------
X = RandomVariable(Uniform(-1, 1), :x)

model = Model(
    rv -> (6 .* rv.x .- 2).^2 .* sin.(12 .* rv.x .- 4),
    :y
)

# plain function for the smooth analytical reference curve
forrester(x) = (6x - 2)^2 * sin(12x - 4)

y_symbol = :y
bases    = [SurrogateModelling.LegendreBasis()]   # d=1 -> vector of length 1

# ---------------------------------------------------------
# Training data (small design -> makes truncation/regularization effects visible)
# ---------------------------------------------------------
n_train = 20
design_train = LatinHypercubeSampling(n_train)
data_train = sample(X, design_train)
evaluate!(model, data_train)

# held-out test set for Q² reporting in legends
n_test = 1001
design_test = LatinHypercubeSampling(n_test)
data_test = sample(X, design_test)
evaluate!(model, data_test)
X_test = Matrix(data_test[:, [:x]])

# fine grid for smooth plotting curves
x_plot = collect(range(-1, 1, length=400))
X_plot = reshape(x_plot, :, 1)
y_plot_true = forrester.(x_plot)

# ---------------------------------------------------------
# Q² helper
# ---------------------------------------------------------
q2(y_true, y_pred) = 1 - sum((y_true .- y_pred).^2) / sum((y_true .- mean(y_true)).^2)

# ============================================================
# EXPERIMENT 1 — OLS under increasing truncation degree
# (NOTE: QBall reduces to TotalDegree in 1D since there are no
#  interaction terms to prune; degree is varied instead, which
#  is the meaningful 1D analogue of a "truncation" comparison.)
# ============================================================
p_maxs_exp1 = [1, 5, 10, 15]
colors_exp1 = [:steelblue, :seagreen, :darkorange, :firebrick]

p1 = plot(
    x_plot, y_plot_true;
    lc = :black, lw = 2, label = "Analytical function",
    xlabel = "x", ylabel = "y", ls = :dash,
    title = "PCE fits under increasing truncation degree (OLS)",
    legend = :topright, size = (900, 550),
    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
)
scatter!(p1, data_train.x, data_train.y;
    mc = :black, ms = 5, msw = 0, label = "Training samples")

for (p_max, c) in zip(p_maxs_exp1, colors_exp1)
    degree = TotalDegree(p_max)
    solver = SurrogateModelling.LASSOSolver(λ=0.0)   # λ=0 -> OLS-equivalent
    pce = SurrogateModelling.PolynomialChaosExpansion(data_train, y_symbol, bases, degree; solver=solver)
    fit!(pce)

    y_pred_plot = predict(pce, X_plot)
    y_pred_test = predict(pce, X_test)
    q2_val = round(q2(data_test.y, y_pred_test), digits=3)

    plot!(p1, x_plot, y_pred_plot;
        lc = c, lw = 2,
        label = "p_max=$p_max  (Q²=$q2_val)")
end

display(p1)

# ============================================================
# EXPERIMENT 2 — LASSO under fixed degree, varying λ
# ============================================================
p_max_exp2 = 15
degree_exp2 = TotalDegree(p_max_exp2)
λs = [0.0, 5,  20, 100]
colors_exp2 = [:steelblue, :seagreen, :darkorange, :firebrick]

p2 = plot(
    x_plot, y_plot_true;
    lc = :black, lw = 2, label = "Analytical function",
    xlabel = "x", ylabel = "y", ls=:dash,
    title = "PCE fits under LASSO regularization (TotalDegree, p_max=$p_max_exp2)",
    legend = :bottomright, size = (900, 550),
    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
)
scatter!(p2, data_train.x, data_train.y;
    mc = :black, ms = 5, msw = 0, label = "Training samples")

for (λ, c) in zip(λs, colors_exp2)
    solver = SurrogateModelling.LASSOSolver(λ=λ)
    pce = SurrogateModelling.PolynomialChaosExpansion(data_train, y_symbol, bases, degree_exp2; solver=solver)
    fit!(pce)

    y_pred_plot = predict(pce, X_plot)
    y_pred_test = predict(pce, X_test)
    q2_val = round(q2(data_test.y, y_pred_test), digits=3)
    n_active = count(!iszero, pce.coeffs)

    label = λ == 0.0 ? "λ=0 (OLS)  (Q²=$q2_val, terms=$n_active)" : "λ=$λ  (Q²=$q2_val, terms=$n_active)"
    plot!(p2, x_plot, y_pred_plot;
        lc = c, lw = 2, label = label)
end

display(p2)
