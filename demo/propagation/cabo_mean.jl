# ==============================================================================
# demo/06_cabo.jl
#
# Demonstrates a full CABO (Confidence-based Adaptive Bayesian Optimization)
# run: bounding the expected response E[g|theta] of a simple function over an
# epistemic parameter theta, starting from a small initial GP, adaptively
# adding points, and reporting the final bound with a decoupled, high-fidelity
# estimate (see estimate_final_bound).
# ==============================================================================

using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Statistics

# ------------------------------------------------------------------------------
# 1. Problem setup
# ------------------------------------------------------------------------------
# A single epistemic parameter theta controlling the mean of a Normal input.
# x ~ Normal(theta, 0.1), theta in [-1, 1]. We bound E[g(x)] over theta, where
# g is a simple nonlinear function -- small and fast enough to run as a demo,
# but structurally the same problem CABO is built for.
x1 = RandomVariable(
    ProbabilityBox{Normal}(Dict(:μ => Interval(-1.0, 1.0), :σ => 0.1)), :x1
)
specs = InputSpec.([x1])

physical_model = Model(df -> sin.(3 .* df.x1) .+ 0.2 .* df.x1 .^ 2, :y)
function epistemic_model(μ; σ = 0.1) 
    return exp(-9 .* σ^2 ./ 2) .* sin.(3 .* μ) .+ 0.2 .* μ.^2 .+ 0.002 
end

x_names, w_names, u_names, v_names = spec_names(specs)

# ------------------------------------------------------------------------------
# 2. Initial training design and GP
# ------------------------------------------------------------------------------
n_train = 15
data_aug_train, _ = build_augmented_design(physical_model, specs, n_train)

gp = GaussianProcess(data_aug_train, :y; kernel_type=GPMatern52())
fit!(gp)

# ------------------------------------------------------------------------------
# 3. Run CABO to bound E[g|theta] -- both directions
# ------------------------------------------------------------------------------
# Ng/Nx are kept SMALL here for a fast demo run. For a real study, size them
# using the statistical arguments in the thesis (expected successes for :pf,
# MC standard error for :mean/:var) -- see demo/05_eole_sampling.jl for the
# underlying sampler these feed into.
cabo_min = cabo_loop(
    physical_model, gp, data_aug_train, specs;
    Ng=100, Nx=200, qoi_type=:mean,
    max_iter=10, direction=:min,
    tol_BO=1e-3, tol_BC=2e-2,
)

cabo_max = cabo_loop(
    physical_model, cabo_min.gp, cabo_min.data, specs;
    Ng=100, Nx=200, qoi_type=:mean,
    max_iter=10, direction=:max,
    tol_BO=1e-3, tol_BC=2e-2,
)

true_min = ([-0.5517], -0.8958)
true_max = ([ 0.5278],  1.0457)

println("\n" * "="^60)
println("CABO results")
println("="^60)
println("MIN E[g] ≈ ", round(cabo_min.μ_bound, digits=4), "  at θ = ", round.(cabo_min.θ_bound, digits=4))
println("MAX E[g] ≈ ", round(cabo_max.μ_bound, digits=4), "  at θ = ", round.(cabo_max.θ_bound, digits=4))
println("-"^60)
println("Analytical results")
println("="^60)
println("MIN E[g] ≈ ", round(true_min[2], digits=4), "  at θ = ", round.(true_min[1], digits=4))
println("MAX E[g] ≈ ", round(true_max[2], digits=4), "  at θ = ", round.(true_max[1], digits=4))


# ------------------------------------------------------------------------------
# 4. Post-processing plots (see src/propagation/cabo/cabo_plots.jl)
# ------------------------------------------------------------------------------
# These three functions are dimension-, QoI-, and analytical-function-agnostic:
# pass a `analytical_qoi` function if you have a ground truth to compare
# against (auto-derives true_min/true_max from the same grid); omit it to just
# see the search history.
plt_landscape = plot_epistemic_landscape(
    specs;
    cabo_min = cabo_min, cabo_max = cabo_max,
    qoi_label = "E[g|θ]",
    analytical_qoi = θ -> epistemic_model(θ[1])  # supply a ground-truth function here if available
)

plt_history = plot_convergence_history(cabo_min=cabo_min, cabo_max=cabo_max)

plt_bound = plot_bound_convergence(
    cabo_min=cabo_min, cabo_max=cabo_max,
    qoi_label = "E[g] bound",
)

display(plt_landscape)
display(plt_history)
display(plt_bound)