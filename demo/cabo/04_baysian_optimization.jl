using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Distributions
using LinearAlgebra
using Statistics          # FIX 1: mean, var
using Printf              # FIX 2: @printf

Random.seed!(42)

# ============================================================
# Epistemic domain bounds
# ============================================================
const μ_FIXED = 0.0

const x1_LOWER, x1_UPPER = -0.5, 1.5
const θ_σ_LOWER, θ_σ_UPPER = 0.5, 1.5

# ============================================================
# Feature column names expected by the GP
# ============================================================
const X_names = [:x1, :u2, :θ_σ]

# ============================================================
# True model and analytical variance (for validation only)
# ============================================================
analytical_model(x1, x2) = x1 .+ x2 .+x1 .* x2 .+ 1
analytical_variance(x1, σ) = σ^2*(x1^2 + 2*x1 + 1) + x1^2 + 2*x1 - (x1 + 1)^2 + 1

# ============================================================
# Augmented-space helpers
# ============================================================

inverse_cdf_x2(u2, θ_σ; μ=μ_FIXED) = quantile.(Normal.(μ, θ_σ), u2)

function mc_augmented(n::Int)
    pts = rand(n, 3)

    u2_raw = pts[:, 1]
    x1_raw = x1_LOWER .+ (x1_UPPER - x1_LOWER) .* pts[:, 2]
    θ_σ_raw = θ_σ_LOWER .+ (θ_σ_UPPER - θ_σ_LOWER) .* pts[:, 3]

    return x1_raw, u2_raw, θ_σ_raw
end

# ============================================================
# Initial training design D₀
# ============================================================
n_train = 200

x1_train, u2_train, θ_σ_train = mc_augmented(n_train)
x2_train = inverse_cdf_x2(u2_train, θ_σ_train)
y_train  = analytical_model(x1_train, x2_train)

data_aug_train = DataFrame(
    :x1  => x1_train,
    :u2  => u2_train,
    :θ_σ => θ_σ_train,
    :y   => y_train,
)

metamodel = GaussianProcess(data_aug_train, :y, kernel=GPMatern52())
@time "fit!" fit!(metamodel)

# ============================================================
# Quick accuracy check on a held-out test set
# ============================================================
n_test = 1000
x1_test, u2_test, θ_σ_test = mc_augmented(n_test)
x2_test  = inverse_cdf_x2(u2_test, θ_σ_test)
y_test_v = analytical_model(x1_test, x2_test)

data_aug_test = DataFrame(
    :x1  => x1_test,
    :u2  => u2_test,
    :θ_σ => θ_σ_test,
    :y   => y_test_v,
)

μ_test, σ_test = predict(metamodel, data_aug_test[:, X_names])

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2( data_aug_test.y, μ_test), digits=5))")

# ============================================================
# estimate_variance
#
#   V̂(x3, θ) = Var_{x1,u2} [ μ_GP(x1, u2, x3, θ) ]
#
# Draws Nx aleatory samples, queries GP mean, returns sample variance.
# ============================================================
function estimate_variance(gp, x3::Float64, θμ::Float64, Nx::Int)
    x1_s = randn(Nx)
    u2_s = rand(Nx)

    df = DataFrame(
        :x1  => x1_s,
        :u2  => u2_s,
        :x3  => fill(x3, Nx),
        :θ_σ => fill(θμ, Nx),
    )

    # FIX 4: predict returns (μ, σ) — unpack, keep only mean
    μ_preds, σ_preds = predict(gp, df[:, X_names])

    return μ_preds, σ_preds
end

# ============================================================
# variance_moments_mcs
#
# Samples Ng epistemic points and collects V̂ at each.
# Returns the mean, std, and the full sample vector.
# ============================================================
function variance_moments_mcs(gp, Ng::Int; Nx::Int=500)
    x3_epi = X3_LB .+ (X3_UB - X3_LB) .* rand(Ng)
    θ_epi  = TH_LB .+ (TH_UB - TH_LB) .* rand(Ng)

    V_samples = [estimate_variance(gp, x3_epi[i], θ_epi[i], Nx)
                 for i in 1:Ng]

    μ_V = mean(V_samples)
    σ_V = std(V_samples)      # std, not sqrt(var): same thing, one call

    return μ_V, σ_V, V_samples
end

# ============================================================
# EI_v — Expected Improvement acquisition  (FIX 6: was missing)
#
#   EI_v(x3*, θ*) = (1/Ng) Σᵢ max( V̂(x3*, θ*) − V̂ᵢ, 0 )
#
# V_samples is the fixed reference vector from variance_moments_mcs.
# PSO calls this many times; the maximum it finds is L_v^BO.
# ============================================================
function EI_v(
    gp, 
    Nx::Int=500,
    Ng::Int=500
)
    μ_preds, σ_preds = variance_moments_mcs(gp, Ng::Int; Nx::Int=500)
    x_star = argmin(μ_preds) + σ_preds * σ
    μ_V_star, σ_V_star = estimate_variance(gp, x_star, Nx)
    return mean(max.(μ_V_star .- μ_preds, 0.0))
end

# # ============================================================
# # cabo_loop — full CABO Bayesian Optimisation
# #
# # Each iteration:
# #   1. variance_moments_mcs  → μ_V, σ_V, V_samples
# #   2. PSO maximises EI_v    → x3*, θ*,  L_v^BO
# #   3. δ_BO = L_v^BO / σ_V  → stop if < δ_tol
# #   4. Evaluate true model at n_new points near (x3*, θ*)
# #   5. Refit GP
# # ============================================================
# function cabo_loop(gp_init, data_train::DataFrame;
#                    Ng::Int          = 200,
#                    Nx::Int          = 500,
#                    δ_tol::Float64   = 0.01,
#                    max_iter::Int    = 30,
#                    n_new::Int       = 5,
#                    n_particles::Int = 30,
#                    pso_iter::Int    = 100)

# # # ============================================================
# # # estimate_variance
# # #
# # #   V̂(x3, θ) = Var_{x1,u2} [ μ_GP(x1, u2, x3, θ) ]
# # #
# # # Draws Nx aleatory samples, queries GP mean, returns sample variance.
# # # ============================================================
# # function estimate_variance(gp, x3::Float64, θμ::Float64, Nx::Int)
# #     x1_s = randn(Nx)
# #     u2_s = rand(Nx)

# #     df = DataFrame(
# #         :x1  => x1_s,
# #         :u2  => u2_s,
# #         :x3  => fill(x3, Nx),
# #         :θ_σ => fill(θμ, Nx),
# #     )

# #     # FIX 4: predict returns (μ, σ) — unpack, keep only mean
# #     μ_preds, _ = predict(gp, df[:, X_names])

# #     return var(μ_preds)
# # end

# # # ============================================================
# # # variance_moments_mcs
# # #
# # # Samples Ng epistemic points and collects V̂ at each.
# # # Returns the mean, std, and the full sample vector.
# # # ============================================================
# # function variance_moments_mcs(gp, Ng::Int; Nx::Int=500)
# #     x3_epi = X3_LB .+ (X3_UB - X3_LB) .* rand(Ng)
# #     θ_epi  = TH_LB .+ (TH_UB - TH_LB) .* rand(Ng)

# #     V_samples = [estimate_variance(gp, x3_epi[i], θ_epi[i], Nx)
# #                  for i in 1:Ng]

# #     μ_V = mean(V_samples)
# #     σ_V = std(V_samples)      # std, not sqrt(var): same thing, one call

# #     return μ_V, σ_V, V_samples
# # end

# # ============================================================
# # EI_v — Expected Improvement acquisition  (FIX 6: was missing)
# #
# #   EI_v(x3*, θ*) = (1/Ng) Σᵢ max( V̂(x3*, θ*) − V̂ᵢ, 0 )
# #
# # V_samples is the fixed reference vector from variance_moments_mcs.
# # PSO calls this many times; the maximum it finds is L_v^BO.
# # ============================================================
# function EI_v(
#     gp, 
#     x1_star::Float64, 
#     θ_star::Float64,
#     V_samples::Vector{Float64};
#     Nx::Int=500
# )
#     x_star = argmin(μ) + α * σ
#     μ_V_star, σ_V_star = estimate_variance(gp, x_star, Nx)
#     return mean(max.(μ_V_star .- V_samples, 0.0))
# end

# # ============================================================
# # cabo_loop — full CABO Bayesian Optimisation
# #
# # Each iteration:
# #   1. variance_moments_mcs  → μ_V, σ_V, V_samples
# #   2. PSO maximises EI_v    → x3*, θ*,  L_v^BO
# #   3. δ_BO = L_v^BO / σ_V  → stop if < δ_tol
# #   4. Evaluate true model at n_new points near (x3*, θ*)
# #   5. Refit GP
# # ============================================================
# function cabo_loop(gp_init, data_train::DataFrame;
#                    Ng::Int          = 200,
#                    Nx::Int          = 500,
#                    δ_tol::Float64   = 0.01,
#                    max_iter::Int    = 30,
#                    n_new::Int       = 5,
#                    n_particles::Int = 30,
#                    pso_iter::Int    = 100)

#     gp   = gp_init
#     data = copy(data_train)

#     V_history = Float64[]
#     δ_history = Float64[]

#     for iter in 1:max_iter
#         println("\n── BO Iteration $iter / $max_iter " * "─"^28)

#         # Step 1: reference variance distribution
#     #     μ_V, σ_V, V_samples = variance_moments_mcs(gp, Ng; Nx=Nx)
#     #     @printf("  μ_V = %.5f,  σ_V = %.5f\n", μ_V, σ_V)

#     #     # Step 2: PSO → L_v^BO and next epistemic candidate
#     #     acq = (x3, θ) -> EI_v(gp, x3, θ, V_samples; Nx=Nx)

#     #     x3_star, θ_star, L_v_BO_val = pso_maximize(acq;
#     #                                         n_particles = n_particles,
#     #                                         max_iter    = pso_iter)

#     #     @printf("  x3* = %+.4f,  θ* = %+.4f,  L_v^BO = %.6f\n",
#     #             x3_star, θ_star, L_v_BO_val)

#     #     # Step 3: δ_BO convergence criterion
#     #     δ_BO = L_v_BO_val / σ_V
#     #     push!(δ_history, δ_BO)
#     #     @printf("  δ_BO = %.6f  (threshold %.4f)\n", δ_BO, δ_tol)

#     #     if δ_BO < δ_tol
#     #         println("  ✓ Converged at iteration $iter")
#     #         break
#     #     end

#     #     # Step 4: true model calls at (x3*, θ*)
#     #     x1_new = randn(n_new)
#     #     u2_new = rand(n_new)
#     #     x2_new = inverse_cdf_x2(u2_new, θ_star)
#     #     y_new  = analytical_model(x1_new, x2_new, fill(x3_star, n_new))

#     #     append!(data, DataFrame(
#     #         :x1  => x1_new,
#     #         :u2  => u2_new,
#     #         :x3  => fill(x3_star, n_new),
#     #         :θ_σ => fill(θ_star,  n_new),
#     #         :y   => y_new,
#     #     ))

#     #     # Step 5: refit GP on expanded data
#     #     gp = GaussianProcess(data, :y, kernel=GPMatern52())
#     #     fit!(gp)

#     #     V_star = estimate_variance(gp, x3_star, θ_star, Nx)
#     #     push!(V_history, V_star)
#     #     @printf("  V̂(x3*, θ*) = %.6f   [n_train = %d]\n", V_star, nrow(data))
#     # end

#     # V_upper  = maximum(V_history)
#     # idx_best = argmax(V_history)

#     # println("\n" * "═"^50)
#     # @printf("  Variance upper bound : %.6f\n", V_upper)
#     # @printf("  Found at iteration   : %d\n",   idx_best)
#     # println("═"^50)

#     # return gp, data, V_history, δ_history
# end

# # # ============================================================
# # # Run
# # # ============================================================
# # gp_bo, data_bo, V_history, δ_history = cabo_loop(
# #     metamodel,
# #     data_aug_train;
# #     Ng          = 1000,    # epistemic MC samples per BO step
# #     Nx          = 10000,    # aleatory MC samples inside estimate_variance
# #     δ_tol       = 0.01,   # convergence threshold
# #     max_iter    = 30,     # hard cap
# #     n_new       = 5,      # true-model calls added per iteration
# #     n_particles = 30,     # PSO swarm size
# #     pso_iter    = 100,    # PSO iterations per BO step
# # )

# # # ── Results ─────────────────────────────────────────────────
# # V_upper = maximum(V_history)
# # println("\nVariance upper bound: $V_upper")

# # # Validate against analytical solution at the found worst-case epistemic point
# # # (scan a grid for the analytical maximum — only possible because we have a closed form)
# # x3_grid = range(X3_LB, X3_UB; length=200)
# # θ_grid  = range(TH_LB, TH_UB; length=200)
# # V_analytical_max = maximum(analytical_variance(x3, θ) for x3 in x3_grid, θ in θ_grid)
# # println("Analytical maximum:   $V_analytical_max")
# # println("Relative error:       $(round(abs(V_upper - V_analytical_max)/V_analytical_max*100, digits=2))%")
#     gp   = gp_init
#     data = copy(data_train)

#     V_history = Float64[]
#     δ_history = Float64[]

#     for iter in 1:max_iter
#         println("\n── BO Iteration $iter / $max_iter " * "─"^28)

#         # Step 1: reference variance distribution
#     #     μ_V, σ_V, V_samples = variance_moments_mcs(gp, Ng; Nx=Nx)
#     #     @printf("  μ_V = %.5f,  σ_V = %.5f\n", μ_V, σ_V)

#     #     # Step 2: PSO → L_v^BO and next epistemic candidate
#     #     acq = (x3, θ) -> EI_v(gp, x3, θ, V_samples; Nx=Nx)

#     #     x3_star, θ_star, L_v_BO_val = pso_maximize(acq;
#     #                                         n_particles = n_particles,
#     #                                         max_iter    = pso_iter)

#     #     @printf("  x3* = %+.4f,  θ* = %+.4f,  L_v^BO = %.6f\n",
#     #             x3_star, θ_star, L_v_BO_val)

#     #     # Step 3: δ_BO convergence criterion
#     #     δ_BO = L_v_BO_val / σ_V
#     #     push!(δ_history, δ_BO)
#     #     @printf("  δ_BO = %.6f  (threshold %.4f)\n", δ_BO, δ_tol)

#     #     if δ_BO < δ_tol
#     #         println("  ✓ Converged at iteration $iter")
#     #         break
#     #     end

#     #     # Step 4: true model calls at (x3*, θ*)
#     #     x1_new = randn(n_new)
#     #     u2_new = rand(n_new)
#     #     x2_new = inverse_cdf_x2(u2_new, θ_star)
#     #     y_new  = analytical_model(x1_new, x2_new, fill(x3_star, n_new))

#     #     append!(data, DataFrame(
#     #         :x1  => x1_new,
#     #         :u2  => u2_new,
#     #         :x3  => fill(x3_star, n_new),
#     #         :θ_σ => fill(θ_star,  n_new),
#     #         :y   => y_new,
#     #     ))

#     #     # Step 5: refit GP on expanded data
#     #     gp = GaussianProcess(data, :y, kernel=GPMatern52())
#     #     fit!(gp)

#     #     V_star = estimate_variance(gp, x3_star, θ_star, Nx)
#     #     push!(V_history, V_star)
#     #     @printf("  V̂(x3*, θ*) = %.6f   [n_train = %d]\n", V_star, nrow(data))
#     # end

#     # V_upper  = maximum(V_history)
#     # idx_best = argmax(V_history)

#     # println("\n" * "═"^50)
#     # @printf("  Variance upper bound : %.6f\n", V_upper)
#     # @printf("  Found at iteration   : %d\n",   idx_best)
#     # println("═"^50)

#     # return gp, data, V_history, δ_history
# end

# # # ============================================================
# # # Run
# # # ============================================================
# # gp_bo, data_bo, V_history, δ_history = cabo_loop(
# #     metamodel,
# #     data_aug_train;
# #     Ng          = 1000,    # epistemic MC samples per BO step
# #     Nx          = 10000,    # aleatory MC samples inside estimate_variance
# #     δ_tol       = 0.01,   # convergence threshold
# #     max_iter    = 30,     # hard cap
# #     n_new       = 5,      # true-model calls added per iteration
# #     n_particles = 30,     # PSO swarm size
# #     pso_iter    = 100,    # PSO iterations per BO step
# # )

# # # ── Results ─────────────────────────────────────────────────
# # V_upper = maximum(V_history)
# # println("\nVariance upper bound: $V_upper")

# # # Validate against analytical solution at the found worst-case epistemic point
# # # (scan a grid for the analytical maximum — only possible because we have a closed form)
# # x3_grid = range(X3_LB, X3_UB; length=200)
# # θ_grid  = range(TH_LB, TH_UB; length=200)
# # V_analytical_max = maximum(analytical_variance(x3, θ) for x3 in x3_grid, θ in θ_grid)
# # println("Analytical maximum:   $V_analytical_max")
# # println("Relative error:       $(round(abs(V_upper - V_analytical_max)/V_analytical_max*100, digits=2))%")