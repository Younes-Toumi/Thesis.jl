using Random
using Plots
using Distributions
using UncertaintyQuantification
using UncertaintyQuantification: sample
using SurrogateModelling
using DataFrames
using Statistics

gr()
# ---------------------------------------------------------
# Model definition (same Forrester-style function as before)
# ---------------------------------------------------------
X = RandomVariable(Uniform(0.0, 1.0), :x)

model = Model(
    rv -> (6 .* rv.x .- 2).^2 .* sin.(12 .* rv.x .- 4),
    :y
)

y_symbol = :y
bases    = [SurrogateModelling.LegendreBasis()]
p_max    = 15
degree   = TotalDegree(p_max)

# ---------------------------------------------------------
# Training / test data
# ---------------------------------------------------------
n_train = 10
design_train = LatinHypercubeSampling(n_train)
data_train = sample(X, design_train)
evaluate!(model, data_train)

n_test = 1001
design_test = LatinHypercubeSampling(n_test)
data_test = sample(X, design_test)
evaluate!(model, data_test)
X_test = Matrix(data_test[:, [:x]])

q2(y_true, y_pred) = 1 - sum((y_true .- y_pred).^2) / sum((y_true .- mean(y_true)).^2)

# ---------------------------------------------------------
# λ grid, log-spaced. λ=0 can't sit on a log axis, so the grid
# starts at a small positive value; λ=0 (OLS) is annotated
# separately rather than plotted on the log x-axis.
# ---------------------------------------------------------
λ_grid = 10.0 .^ range(-6, 2, length=60)   # 1e-4 .. 100

n_active   = Vector{Int}(undef, length(λ_grid))
q2_vals    = Vector{Float64}(undef, length(λ_grid))

for (i, λ) in enumerate(λ_grid)
    solver = SurrogateModelling.LASSOSolver(λ=λ)
    pce = SurrogateModelling.PolynomialChaosExpansion(data_train, y_symbol, bases, degree; solver=solver)
    fit!(pce)
    n_active[i] = count(!iszero, pce.coeffs)
    q2_vals[i]  = q2(data_test.y, predict(pce, X_test))
end

# ---------------------------------------------------------
# Automatic λ selection (the algorithm's own LOO-CV path)
# ---------------------------------------------------------
solver_auto = SurrogateModelling.LASSOSolver()   # λ=nothing -> internal auto-selection
pce_auto = SurrogateModelling.PolynomialChaosExpansion(data_train, y_symbol, bases, degree; solver=solver_auto)
fit!(pce_auto)

λ_selected = solver_auto.λ                        # mutated in place by solve()
n_active_selected = count(!iszero, pce_auto.coeffs)
q2_selected = q2(data_test.y, predict(pce_auto, X_test))

println("Auto-selected λ = $(round(λ_selected, sigdigits=4))")
println("  retained coefficients: $n_active_selected")
println("  Q² (test): $(round(q2_selected, digits=4))")

# ---------------------------------------------------------
# Plot 1 (top): retained coefficients vs λ
# ---------------------------------------------------------
p_top = plot(
    λ_grid, n_active;
    xscale = :log10, lc = :steelblue, lw = 2, marker = :circle, ms = 4, msw = 0,
    ylabel = "Retained coefficients",
    title  = "LASSO regularisation path (TotalDegree, p_max=$p_max)",
    legend = :outertopright,
    label  = "Retained (nonzero) coefficients",
)
vline!(p_top, [λ_selected]; ls = :dash, lc = :firebrick, lw = 2,
    label = "Auto-selected λ ≈ $(round(λ_selected, sigdigits=3))")
scatter!(p_top, [λ_selected], [n_active_selected];
    mc = :firebrick, ms = 7, msw = 0, label = false)

# ---------------------------------------------------------
# Plot 2 (bottom): Q² vs λ
# ---------------------------------------------------------
p_bottom = plot(
    λ_grid, q2_vals;
    xscale = :log10, lc = :seagreen, lw = 2, marker = :circle, ms = 4, msw = 0,
    xlabel = "λ (log scale)", ylabel = "Q² (test)",
    legend = :outertopright,
    label  = "Q² across λ",
)
vline!(p_bottom, [λ_selected]; ls = :dash, lc = :firebrick, lw = 2,
    label = "Auto-selected λ ≈ $(round(λ_selected, sigdigits=3))")
scatter!(p_bottom, [λ_selected], [q2_selected];
    mc = :firebrick, ms = 7, msw = 0, label = "Selected fit (Q²=$(round(q2_selected, digits=3)))")

p_combined = plot(p_top, p_bottom, layout = (2, 1), size = (900, 700),
    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

display(p_combined)
