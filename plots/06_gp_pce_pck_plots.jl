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
# Model definition (same Forrester-style function as before)
# ---------------------------------------------------------
X = RandomVariable(Uniform(0, 1), :x)

model = Model(
    rv -> (6 .* rv.x .- 2).^2 .* sin.(12 .* rv.x .- 4),
    :y
)

forrester(x) = (6x - 2)^2 * sin(12x - 4)

y_symbol = :y
bases    = [SurrogateModelling.LegendreBasis()]
p_max    = 15   # modest trend degree -> PCE underfits a bit, motivating PCK's residual

# ---------------------------------------------------------
# Training / test data
# ---------------------------------------------------------
n_train = 5
design_train = LatinHypercubeSampling(n_train)
data_train = sample(X, design_train)
evaluate!(model, data_train)

n_test = 1001
design_test = LatinHypercubeSampling(n_test)
data_test = sample(X, design_test)
evaluate!(model, data_test)
X_test = Matrix(data_test[:, [:x]])

q2(y_true, y_pred) = 1 - sum((y_true .- y_pred).^2) / sum((y_true .- mean(y_true)).^2)

# fine grid for smooth plotting
x_plot = collect(range(0, 1, length=400))
X_plot = reshape(x_plot, :, 1)
y_plot_true = forrester.(x_plot)

# ---------------------------------------------------------
# Colors, matching the convention used throughout: GP blue, PCK green, PCE orange
# ---------------------------------------------------------
COLOR_GP  = :steelblue
COLOR_PCE = :darkorange
COLOR_PCK = :seagreen

# ---------------------------------------------------------
# Fit the three surrogates
# ---------------------------------------------------------
kernel_type = GPMatern52

gp = SurrogateModelling.GaussianProcess(data_train, y_symbol; kernel_type=kernel_type())
fit!(gp)

solver = SurrogateModelling.LASSOSolver()   # OLS-equivalent
pce = SurrogateModelling.PolynomialChaosExpansion(data_train, y_symbol, bases, TotalDegree(15); solver=solver)
fit!(pce)

pce_trend = SurrogateModelling.PolynomialChaosExpansion(data_train, y_symbol, bases, TotalDegree(4); solver=SurrogateModelling.LASSOSolver())
pck = SurrogateModelling.PolynomialChaosKriging(data_train, y_symbol, pce_trend; kernel_type=kernel_type())
fit!(pck)

# ---------------------------------------------------------
# Predictions ( mean + std where available )
# ---------------------------------------------------------
μ_gp, σ2_gp = predict(gp, X_plot; mode=:mean_and_var)
σ_gp = sqrt.(max.(σ2_gp, 0.0))
q2_gp = q2(data_test.y, predict(gp, X_test; mode=:mean))

μ_pce = predict(pce, X_plot)
q2_pce = q2(data_test.y, predict(pce, X_test))

μ_pck, σ2_pck = predict(pck, X_plot; mode=:mean_and_var)
σ_pck = sqrt.(max.(σ2_pck, 0.0))
q2_pck = q2(data_test.y, predict(pck, X_test; mode=:mean))

# ---------------------------------------------------------
# Common y-limits across all three panels for fair visual comparison
# ---------------------------------------------------------
all_y = vcat(y_plot_true, data_train.y,
             μ_gp .+ 1.96 .* σ_gp, μ_gp .- 1.96 .* σ_gp,
             μ_pck .+ 1.96 .* σ_pck, μ_pck .- 1.96 .* σ_pck,
             μ_pce)
pad = 0.05 * (maximum(all_y) - minimum(all_y))
ylims_common = (minimum(all_y) - pad, maximum(all_y) + pad)

# ---------------------------------------------------------
# Panel 1: GP
# ---------------------------------------------------------
p_gp = plot(
    x_plot, μ_gp;
    ribbon = 1.96 .* σ_gp, fillalpha = 0.2,
    lc = COLOR_GP, fillcolor = COLOR_GP, lw = 2,
    label = "GP mean ± 1.96σ",
    title = "GP  (Q²=$(round(q2_gp, digits=3)))",
    xlabel = "x", ylabel = "y", ylims = ylims_common,
    legend = :top, legendfontsize = 7,
)
plot!(p_gp, x_plot, y_plot_true; lc = :black, lw = 2, ls = :dash, label = "Analytical")
scatter!(p_gp, data_train.x, data_train.y; mc = :black, ms = 5, msw = 0, label = "Training data")

# ---------------------------------------------------------
# Panel 2: PCE (no uncertainty -- purely deterministic)
# ---------------------------------------------------------
p_pce = plot(
    x_plot, μ_pce;
    lc = COLOR_PCE, lw = 2, label = "PCE prediction",
    title = "PCE  (Q²=$(round(q2_pce, digits=3)))",
    xlabel = "x", ylims = ylims_common,
    legend = :top, legendfontsize = 7,
)
plot!(p_pce, x_plot, y_plot_true; lc = :black, lw = 2, ls = :dash, label = "Analytical")
scatter!(p_pce, data_train.x, data_train.y; mc = :black, ms = 5, msw = 0, label = "Training data")

# ---------------------------------------------------------
# Panel 3: PCK
# ---------------------------------------------------------
p_pck = plot(
    x_plot, μ_pck;
    ribbon = 1.96 .* σ_pck, fillalpha = 0.2,
    lc = COLOR_PCK, fillcolor = COLOR_PCK, lw = 2,
    label = "PCK mean ± 1.96σ",
    title = "PCK  (Q²=$(round(q2_pck, digits=3)))",
    xlabel = "x", ylims = ylims_common,
    legend = :top, legendfontsize = 7,
)
plot!(p_pck, x_plot, y_plot_true; lc = :black, lw = 2, ls = :dash, label = "Analytical")
scatter!(p_pck, data_train.x, data_train.y; mc = :black, ms = 5, msw = 0, label = "Training data")

# ---------------------------------------------------------
# Combine into 1x3 layout
# ---------------------------------------------------------
p_combined = plot(p_gp, p_pce, p_pck, layout = (1, 3), size = (1500, 480),
    left_margin = 5Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm)

display(p_combined)

println("Saved plot.")
println("GP  Q² = $(round(q2_gp, digits=4))")
println("PCE Q² = $(round(q2_pce, digits=4))")
println("PCK Q² = $(round(q2_pck, digits=4))")