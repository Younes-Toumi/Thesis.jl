using Random
using Plots
using Distributions
using UncertaintyQuantification
using UncertaintyQuantification: sample
using SurrogateModelling
using DataFrames
using AbstractGPs
using KernelFunctions
using LinearAlgebra

Random.seed!(42)

gr()

# ---------------------------------------------------------
# Training data
# ---------------------------------------------------------

X = RandomVariable.(Uniform(0.0, 1.0), :x)

model = Model(
    rv -> (6 .* rv.x .- 2).^2 .* sin.(12 .* rv.x .- 4),
    :y
)

n_train, n_test = 10, 1001

design_train = MonteCarlo(n_train)
design_test  = MonteCarlo(n_test)

data_train = sample(X, design_train)
data_test  = sample(X, design_test)

evaluate!(model, data_train)
evaluate!(model, data_test)

# ---------------------------------------------------------
# Prediction grid
# ---------------------------------------------------------

x_grid = DataFrame(
    x = collect(range(-1,1,length=1001))
)

x_plot = x_grid.x

# ---------------------------------------------------------
# 1) Initial GP prior (naive, hand-chosen hyperparameters)
#
# A PRIOR is, by definition, unconditioned on data: zero mean, constant-
# width band. Built directly from build_kernel + a raw AbstractGPs GP(),
# rather than through the fitted `GaussianProcess` wrapper (which only
# has meaning post-conditioning), avoiding any dependency on an
# unconfirmed constructor keyword or field.
# ---------------------------------------------------------

θ_init = (
    lengthscale = [1.0],   # ARD-shaped (vector), matching the codebase convention
    variance    = 1.0,
)

kernel_prior = SurrogateModelling.build_kernel(GPSquaredExponential(), θ_init)
gp_prior_raw = GP(kernel_prior)                       # AbstractGPs defaults to zero mean
fx_prior     = gp_prior_raw(x_plot', 1e-8)             # small jitter, purely numerical

mean_prior, var_prior = mean_and_var(fx_prior)
std_prior = sqrt.(var_prior)

# ---------------------------------------------------------
# Candidate points
# ---------------------------------------------------------

p1 = plot(
    x_plot,
    mean_prior,
    ribbon = 2 .* std_prior,
    fillalpha = 0.25,
    color=:steelblue,
    linewidth=2,
    label="Prior mean ±2σ",
    xlabel="x",
    ylabel="f(x)",
    title="Initial GP prior\nθ = 1 (naive, unconditioned)",
    legend=:bottomleft,
)

scatter!(
    p1,
    data_train.x,
    data_train.y,
    color=:black,
    marker=:circle,
    label="Training samples"
)


# ---------------------------------------------------------
# 2) Prior with ARD-optimized hyperparameters (still unconditioned)
#
# Optimize hyperparameters via fit!, then build a fresh, ZERO-MEAN,
# unconditioned prior using the OPTIMIZED kernel -- this isolates the
# effect of the hyperparameter fit from the effect of conditioning on
# data, which is what panel 3 shows.
# ---------------------------------------------------------

gp_ard = GaussianProcess(
    data_train,
    :y;
    kernel_type = GPSquaredExponential()
)

fit!(gp_ard)   # optimizes θ AND builds gp_ard.posterior; we only need the optimized kernel here

kernel_ard = gp_ard.kernel_posterior

prior_ard = GP(kernel_ard)                 # zero mean, optimized kernel, still UNCONDITIONED
fx_prior_ard = prior_ard(x_plot', 1e-8)

μ_ard, σ2_ard = mean_and_var(fx_prior_ard)
σ_ard = sqrt.(σ2_ard)

p2 = plot(
    x_plot,
    μ_ard,
    ribbon=2 .* σ_ard,
    fillalpha=0.25,
    color=:steelblue,
    linewidth=2,
    label="ARD prior ±2σ",
    xlabel="x",
    ylabel="f(x)",
    title="ARD-informed prior\nOptimized hyperparameters, still unconditioned",
    legend=:bottomleft,
)

scatter!(
    p2,
    data_train.x,
    data_train.y,
    color=:black,
    marker=:circle,
    label="Training samples"
)


# ---------------------------------------------------------
# 3) Posterior GP (conditioned on the training data)
#
# predict() with mode=:mean_and_var returns (mean, VARIANCE) -- explicit
# sqrt needed for a ±2σ band, matching panel 2's convention exactly
# (the original code's `σ_post` was used unsquare-rooted, an
# inconsistency with panel 2's own correct sqrt.(σ2_ard) two lines above it).
# ---------------------------------------------------------

μ_post, σ2_post = predict(
    gp_ard,
    Matrix(x_grid);
    mode = :mean_and_var,
)
σ_post = sqrt.(σ2_post)

p3 = plot(
    x_plot,
    μ_post,
    ribbon=2 .* σ_post,
    fillalpha=0.25,
    color=:steelblue,
    linewidth=2,
    label="Posterior mean ±2σ",
    xlabel="x",
    ylabel="f(x)",
    title="Posterior GP\nConditioned on training data",
    legend=:bottomleft,
)

scatter!(
    p3,
    data_train.x,
    data_train.y,
    color=:black,
    marker=:circle,
    label="Training samples"
)

# ---------------------------------------------------------
# Final figure
# ---------------------------------------------------------

p_combined = plot(
    p1,
    p2,
    p3,
    layout=(1,3),
    size=(1900,450),
    top_margin=7Plots.mm,
    xlims = [0, 1]
)

display(p_combined)