using SurrogateModelling
using UncertaintyQuantification
using UncertaintyQuantification: sample, evaluate
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra


# ==============================================================================
# bootstrap_ensemble_usage.jl
# Examples: bootstrap GP and PCE on Ishigami, plus CABO integration.
# ==============================================================================

# ── shared setup (reuse the data from your existing code) ─────────────────────
x = RandomVariable.(Uniform(0, 1), :x)
X = [x]
physical_model = Model(rv -> forrester.(rv.x), :y)

n_train, n_test = 10, 1000

design_train = LatinHypercubeSampling(n_train)
design_test = LatinHypercubeSampling(n_test)

data_train = sample(X, design_train)
data_test = sample(X, design_test)

evaluate!(physical_model, data_train)
evaluate!(physical_model, data_test)

x_names = [x.name for x in X]
X_test = data_test[:, x_names]
y_test = data_test[:, :y]
X_test_mat = Matrix(X_test)   # Matrix version needed for predict(ensemble, X)



println("\n══ Normal GP ══════════════════════════════════════")
gp = GaussianProcess(
    data_train, :y;
    kernel_type  = GPSquaredExponential(),
)
@time "fit!" fit!(gp)

μ_gp, σ_gp = @time "predict" predict(gp, X_test_mat)
println("RMSE: $(round(rmse(y_test, μ_gp), digits=4))")
println("Q²:   $(round(q2( y_test, μ_gp), digits=4))")


# ==============================================================================
# 1.  Bootstrap GP
# ==============================================================================

n_bootstraps = 50

println("\n══ Bootstrap GP (B=$n_bootstraps) ══════════════════════════════════════")
boots_gp = gp_bootstrap(
    data_train, :y;
    kernel_type  = GPSquaredExponential(),
    n_bootstraps = n_bootstraps,
    rng_seed     = 0
)
@time "fit!" fit!(boots_gp)

μ_boots_gp, σ_boots_gp = @time "predict" predict(boots_gp, X_test_mat)
println("RMSE: $(round(rmse(y_test, μ_boots_gp), digits=4))")
println("Q²:   $(round(q2( y_test, μ_boots_gp), digits=4))")
calibration_report(σ_boots_gp, μ_boots_gp, y_test)

# # ==============================================================================
# # 2.  Bootstrap PCE  (OLS)
# # ==============================================================================

bases = [SurrogateModelling.LegendreBasis()] # Uniform inputs
degree = TotalDegree(5) # can be automated based on availabel samples


println("\n══ Bootstrap PCE – OLS (B=$n_bootstraps) ══════════════════════════════")
bpce_ols = pce_bootstrap(
    data_train, :y, bases, degree;
    solver       = SurrogateModelling.OLSSolver(),
    n_bootstraps = n_bootstraps,
    rng_seed     = 0
)
@time "fit!" fit!(bpce_ols)

μ_bpce, σ_bpce = @time "predict" predict(bpce_ols, X_test_mat)
println("RMSE: $(round(rmse(y_test, μ_bpce), digits=4))")
println("Q²:   $(round(q2( y_test, μ_bpce), digits=4))")
calibration_report(σ_bpce, μ_bpce, y_test)

# # ==============================================================================
# # 3.  Bootstrap PCE  (LASSO – sparse)
# # ==============================================================================

println("\n══ Bootstrap PCE – LASSO (B=$n_bootstraps) ════════════════════════════")
bpce_lasso = pce_bootstrap(
    data_train, :y, bases, degree;
    solver       = LASSOSolver(),
    n_bootstraps = n_bootstraps,
    rng_seed     = 0
)
@time "fit!" fit!(bpce_lasso)

μ_bl, σ_bl = @time "predict" predict(bpce_lasso, X_test_mat)
println("RMSE: $(round(rmse(y_test, μ_bl), digits=4))")
println("Q²:   $(round(q2( y_test, μ_bl), digits=4))")
calibration_report(σ_bl, μ_bl, y_test)




using Plots

# ============================================================
# Sort x for clean plotting
# ============================================================
using Plots

# sort for smooth plotting
idx = sortperm(vec(X_test_mat))

x_plot = vec(X_test_mat[idx])

μ_gp_plot = μ_gp[idx]
σ_gp_plot = σ_gp[idx]

μ_bgp_plot = μ_boots_gp[idx]
σ_bgp_plot = σ_boots_gp[idx]

μ_ols_plot = μ_bpce[idx]
σ_ols_plot = σ_bpce[idx]

μ_lasso_plot = μ_bl[idx]
σ_lasso_plot = σ_bl[idx]

x_train_plot = vec(data_train[:, :x])
y_train_plot = vec(data_train[:, :y])


p_gp = plot(
    x_plot,
    μ_gp_plot;
    ribbon = 2 .* σ_gp_plot,
    label = "μ(x)",
    xlabel = "x",
    ylabel = "y",
    title = "GP",
)

scatter!(
    p_gp,
    x_train_plot,
    y_train_plot;
    label = "Training data",
)


p_bgp = plot(
    x_plot,
    μ_bgp_plot;
    ribbon = 2 .* σ_bgp_plot,
    label = "μ(x)",
    xlabel = "x",
    ylabel = "y",
    title = "Bootstrap GP",
)

scatter!(
    p_bgp,
    x_train_plot,
    y_train_plot;
    label = "Training data",
)


p_ols = plot(
    x_plot,
    μ_ols_plot;
    ribbon = 2 .* σ_ols_plot,
    label = "μ(x)",
    xlabel = "x",
    ylabel = "y",
    title = "Bootstrap PCE (OLS)",
)

scatter!(
    p_ols,
    x_train_plot,
    y_train_plot;
    label = "Training data",
)

p_lasso = plot(
    x_plot,
    μ_lasso_plot;
    ribbon = 2 .* σ_lasso_plot,
    label = "μ(x)",
    xlabel = "x",
    ylabel = "y",
    title = "Bootstrap PCE (LASSO)",
)

scatter!(
    p_lasso,
    x_train_plot,
    y_train_plot;
    label = "Training data",
)

plot(
    p_gp,
    p_bgp,
    p_ols,
    p_lasso;
    layout = (4,1),
    size = (900,900)
)