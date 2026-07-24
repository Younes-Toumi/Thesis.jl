# ==============================================================================
# Demonstrates EOLE (Expansion Optimal Linear Estimation) / truncated
# Karhunen-Loeve sampling of a GP posterior.
#
# Three experiments, each answering one question:
#   A. What does EOLE compute, and does it agree with exact sampling?
#   B. How much faster is EOLE than exact sampling, at realistic scale?
#   C. What do the two look like SIDE BY SIDE, as sample paths on a plot?
# ==============================================================================

using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Statistics
using LinearAlgebra
using KernelFunctions
using Plots

gr()

# ------------------------------------------------------------------------------
# 0. Fit a GP to sample from (shared by all three experiments below)
# ------------------------------------------------------------------------------
x = RandomVariable(Uniform(0, 1), :x)
physical_model = model_forrester
specs = InputSpec.([x])

x_names, w_names, u_names, v_names = spec_names(specs)

data_aug_train, data_phys_train = build_augmented_design(physical_model, specs, 20)
data_aug_test, data_phys_test   = build_augmented_design(physical_model, specs, 1000)

gp = GaussianProcess(data_aug_train, :y; kernel_type=GPMatern52())
fit!(gp)

println("lengthscale = ", gp.θ.lengthscale, "   variance = ", gp.θ.variance)

y_test = data_aug_test.y
y_pred = predict(gp, Matrix(data_aug_test[:, w_names]), mode = :mean)

println("MSE: $(mse(y_test, y_pred))")
println("Q2: $(q2(y_test, y_pred))")


# The EOLE support grid W defines the eigenbasis. It's independent of any
# particular query set -- build it once and reuse it in every experiment below.
N0 = 100   # support-grid resolution -- larger N0 = better approximation, more upfront cost
W_support, _ = build_augmented_design(nothing, [InputSpec(x)], N0)
W_support = Matrix(W_support)
X_train   = gp.X
y_train   = gp.y

# ==============================================================================
# EXPERIMENT A -- correctness: does EOLE agree with exact posterior sampling?
# ==============================================================================
# EOLE works by eigendecomposing the POSTERIOR covariance evaluated on the
# fixed support grid W (NOT the query points), retaining the top r eigenmodes
# that capture `energy_threshold` of the variance. This one-time
# eigendecomposition (O(N0^3)) is the ONLY expensive step -- every subsequent
# query, at ANY new set of points, is then cheap matrix-vector algebra.
#
# Here we check EOLE's accuracy against EXACT sampling at the SAME query
# points: both should agree on the mean-of-means, since EOLE is mean-unbiased.
println("="^70)
println("EXPERIMENT A: EOLE vs. exact sampling, at a small, fixed query set")
println("="^70)

n_check = 500
n_realisations = 50

X_check_a = randn(n_check, length(w_names))

# exact: draw directly from the joint posterior at these query points
samples_exact = predict(gp, X_check_a, mode=:sample, n_samples=n_realisations) 

# EOLE: same query points, via the truncated eigenbasis
gp_samples_a! = build_kl_sampler(gp, W_support, X_train; N_samples=n_realisations, Nx=n_check)
qoi_buf_eole = Vector{Float64}(undef, n_realisations)
gp_samples_a!(qoi_buf_eole, X_check_a; qoi_type=:mean)   # mean over the n_check points, per realization

exact_mean_of_means = mean(vec(mean(samples_exact, dims=1)))
println("mean-of-means (mean QoI, averaged across realizations):")
println("  exact sampling: ", round(exact_mean_of_means, digits=4))
println("  EOLE:           ", round(mean(qoi_buf_eole), digits=4))
println("  -> should agree closely; EOLE is mean-unbiased regardless of N0")
println()


# ==============================================================================
# EXPERIMENT B -- performance: how much faster is EOLE at realistic scale?
# ==============================================================================
# Exact sampling needs a FRESH O(Nx^3) Cholesky factorisation of the query-set
# covariance EVERY TIME the query set changes. EOLE factorises ONCE (O(N0^3))
# and reuses it for any number of subsequent, DIFFERENT query sets.
#
# This matters enormously for CABO's inner loop, where the query set changes
# on every one of ~10,000 PSO evaluations per outer iteration (see
# demo/06_cabo.jl). For a ONE-OFF query (like this demo), the distinction is
# much less important -- direct sampling is simpler and exact, and EOLE only
# pays for itself when the SAME eigenbasis is reused across many queries.
# Timed here anyway, to make the cost difference concrete.
println("="^70)
println("EXPERIMENT B: timing -- classical sampling vs. building + querying EOLE")
println("="^70)

Nx_b, n_samples_b = 5000, 10
X_check_b = randn(Nx_b, length(w_names))

samples_classical_b = @time "classical sampling          " predict(gp, X_check_b; mode=:sample, n_samples=n_samples_b)

gp_samples_b! = @time "building the EOLE sampler (once)" build_kl_sampler(gp, W_support, X_train; N_samples=n_samples_b, Nx=Nx_b)
eole_buf_b = Matrix{Float64}(undef, Nx_b, n_samples_b)
@time "EOLE sampling (reusing the above)" gp_samples_b!(eole_buf_b, X_check_b; qoi_type=:samples)

println()
println("classical samples matrix: ", size(samples_classical_b))
println("EOLE samples matrix:      ", size(eole_buf_b))
println()