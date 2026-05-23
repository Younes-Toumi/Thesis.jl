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
const θσ_LOWER, θσ_UPPER = 0.5, 1.5

# ============================================================
# Feature column names expected by the GP
# ============================================================
const X_names = [:x1, :u2, :θσ]

# ============================================================
# True model and analytical variance (for validation only)
# ============================================================
analytical_model(x1, x2) = x1 .+ x2 .+x1 .* x2 .+ 1
analytical_variance(x1, σ) = σ^2*(x1^2 + 2*x1 + 1) + x1^2 + 2*x1 - (x1 + 1)^2 + 1

# ============================================================
# Augmented-space helpers
# ============================================================

inverse_cdf_x2(u2, θσ; μ=μ_FIXED) = quantile.(Normal.(μ, θσ), u2)

function mc_augmented(n::Int)
    pts = rand(n, 3)

    u2_raw = pts[:, 1]
    x1_raw = x1_LOWER .+ (x1_UPPER - x1_LOWER) .* pts[:, 2]
    θσ_raw = θσ_LOWER .+ (θσ_UPPER - θσ_LOWER) .* pts[:, 3]

    return x1_raw, u2_raw, θσ_raw
end

# ============================================================
# Initial training design D₀
# ============================================================
n_train = 200

x1_train, u2_train, θσ_train = mc_augmented(n_train)
x2_train = inverse_cdf_x2(u2_train, θσ_train)
y_train  = analytical_model(x1_train, x2_train)

data_aug_train = DataFrame(
    :x1  => x1_train,
    :u2  => u2_train,
    :θσ => θσ_train,
    :y   => y_train,
)

metamodel = GaussianProcess(data_aug_train, :y, kernel=GPMatern52())
@time "fit!" fit!(metamodel)

# ============================================================
# Quick accuracy check on a held-out test set
# ============================================================
n_test = 1000
x1_test, u2_test, θσ_test = mc_augmented(n_test)
x2_test  = inverse_cdf_x2(u2_test, θσ_test)
y_test_v = analytical_model(x1_test, x2_test)

data_aug_test = DataFrame(
    :x1  => x1_test,
    :u2  => u2_test,
    :θσ => θσ_test,
    :y   => y_test_v,
)

μ_test, σ_test = predict(metamodel, data_aug_test[:, X_names])

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2( data_aug_test.y, μ_test), digits=5))")

function estimate_Vy(gp, x1::Float64, θσ::Float64, Nx::Int)

    x1_s = fill(x1, Nx)
    u2_s = rand(Nx)
    θσ_s = fill(θσ, Nx)
    
    df = DataFrame(
        :x1  => x1_s,
        :u2  => u2_s,
        :θσ => θσ_s,
    )

    μ_preds, σ_preds = predict(gp, df[:, X_names])

    return μ_preds
end

function variance_moments_mcs(gp, x1, θσ, Ng::Int, Nx::Int)

    V_samples = [estimate_variance(gp, x1, θσ, Nx) for j in j:Ng]

    μ_V = mean(V_samples)
    σ_V = std(V_samples)

    return μ_V, σ_V, V_samples
end

function EI_v(gp, x1, θσ, Nx::Int, Ng::Int)

    α = 1.0

    μ_V, σ_V, V_samples = variance_moments_mcs(gp, x1, θσ, Ng::Int, Nx::Int)
    (x1_star, θσ_star) = argmin(μ_V + α * σ_V) # TODO FIX this

    μ_V_star, σ_V_star = estimate_variance(gp, x1_star, θσ_star, Nx)
    return mean(max.(μ_V_star .- V_samples, 0.0))
end




# ============================================================
# cabo_loop — full CABO Bayesian Optimisation
#
# Each iteration:
#   1. variance_moments_mcs  → μ_V, σ_V, V_samples
#   2. PSO maximises EI_v    → x3*, θ*,  L_v^BO
#   3. δ_BO = L_v^BO / σ_V  → stop if < δ_tol
#   4. Evaluate true model at n_new points near (x3*, θ*)
#   5. Refit GP
# ============================================================
function cabo_loop(gp_init, data_train::DataFrame;
                   Ng::Int          = 200,
                   Nx::Int          = 500,
                   max_iter::Int    = 30,
                   n_new::Int       = 5)

    gp   = gp_init
    data = copy(data_train)

    V_history = Float64[]
    δ_history = Float64[]

    for iter in 1:max_iter
        println("\n── BO Iteration $iter / $max_iter " * "─"^28)

        # Step 1: reference variance distribution
        μ_V, σ_V, V_samples = variance_moments_mcs(gp, x1, θσ, Ng, Nx)
        @printf("  μ_V = %.5f,  σ_V = %.5f\n", μ_V, σ_V)

        # Step 2: PSO → L_v^BO and next epistemic candidate
        acq = (x1, θσ) -> EI_v(gp, x1, θσ, Nx, Ng)

        data_pso = data[:, X_names]
        bounds_pso = [
            x1_LOWER x1_UPPER
            θσ_LOWER θσ_UPPER
        ]

        best_x, best_val, history = pso_optimize(acq, data_pso, bounds_pso; max_iter=100, mode=:max)

        (x1_plus, θσ_plus) = best_x

        @printf("  x1* = %+.4f,  θσ* = %+.4f,  L_v^BO = %.6f\n",
                x1_plus, θσ_plus, best_val)

        # Step 3: δ_BO convergence criterion
        δ_BO = best_val / (max(V_samples) - min(V_samples))
        push!(δ_history, δ_BO)
        @printf("  δ_BO = %.6f  (threshold %.4f)\n", δ_BO, δ_tol)

        if δ_BO < 0.01
            println("  ✓ Converged at iteration $iter")
            break
        end

        # Step 4: true model calls at (x3*, θ*)
        x1_new = fill(x3_plus, n_new)
        u2_new = rand(n_new)
        x2_new = inverse_cdf_x2(u2_new, θσ_plus)
        y_new  = analytical_model(x1_new, x2_new)

        θσ_new = fill(θσ_star,  n_new)


        append!(data, DataFrame(
            :x1  => x1_new,
            :u2  => u2_new,
            :θσ => θσ_new,
            :y   => y_new,
        ))

        # Step 5: refit GP on expanded data
        gp = GaussianProcess(data, :y, kernel=GPMatern52())
        fit!(gp)

        V_star = estimate_variance(gp, x1_plus, θσ_star, Nx)
        push!(V_history, V_star)
        @printf("  V̂(x1*, θ*) = %.6f   [n_train = %d]\n", V_star, nrow(data))
    end

    V_upper  = maximum(V_history)
    idx_best = argmax(V_history)

    println("\n" * "═"^50)
    @printf("  Variance upper bound : %.6f\n", V_upper)
    @printf("  Found at iteration   : %d\n",   idx_best)
    println("═"^50)

    return gp, data, V_history, δ_history
end

# ============================================================
# Run
# ============================================================
gp_bo, data_bo, V_history, δ_history = cabo_loop(
    metamodel,
    data_aug_train;
    Ng          = 1000,    # epistemic MC samples per BO step
    Nx          = 10000,    # aleatory MC samples inside estimate_variance
    max_iter    = 30,     # hard cap
    n_new       = 5,      # true-model calls added per iteration
)

# ── Results ─────────────────────────────────────────────────
V_upper = maximum(V_history)
println("\nVariance upper bound: $V_upper")

# Validate against analytical solution at the found worst-case epistemic point
# (scan a grid for the analytical maximum — only possible because we have a closed form)
x1_grid = range(x1_LOWER, x1_UPPER; length=200)
θσ_grid  = range(θσ_LOWER, θσ_UPPER; length=200)

V_analytical_max = maximum(analytical_variance(x1, θσ) for x1 in x1_grid, θσ in θσ_grid)

println("Analytical maximum:   $V_analytical_max")
println("Relative error:       $(round(abs(V_upper - V_analytical_max)/V_analytical_max*100, digits=2))%")