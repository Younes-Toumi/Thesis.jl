using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Distributions
using LinearAlgebra
using Statistics
using Printf
using Metaheuristics
Random.seed!(42)

# ============================================================
# Epistemic domain bounds  (FIX 3: define before PSO uses them)
# ============================================================
const μ_FIXED = 0.0

const x1_UPPER = 1.5
const x1_LOWER = -0.5

const θ_σ_UPPER = 1.5
const θ_σ_LOWER = 0.5

const BOUNDS = (
    x1 = (x1_LOWER, x1_UPPER),
    θσ = (θ_σ_LOWER, θ_σ_UPPER)
)

# bounds must be boxconstraints
lb = [x1_LOWER, θ_σ_LOWER]
ub = [x1_UPPER, θ_σ_UPPER]
bounds = Metaheuristics.boxconstraints(lb=lb, ub=ub)


# ============================================================
# Feature column names expected by the GP
# ============================================================
const X_names = [:x1, :u2, :θ_σ]

# ============================================================
# True model — operates on physical inputs (x1, x2)
# ============================================================
analytical_model(x1, x2) = x1 .+ x2 .+x1 .* x2 .+ 1
analytical_variance(x1, σ) = σ^2*(x1^2 + 2*x1 + 1) + x1^2 + 2*x1 - (x1 + 1)^2 + 1
# ============================================================

# ============================================================
# Augmented-space helpers
# ============================================================

inverse_cdf_x2(u2, θ_σ; μ=μ_FIXED) = quantile.(Normal.(μ, θ_σ), u2)

function mc_augmented(n::Int)

    # aleatory variable
    u2 = rand(n)

    # epistemic variable
    θσ = rand(n) .* (θ_σ_UPPER - θ_σ_LOWER) .+ θ_σ_LOWER

    # physical design variable
    x1 = rand(n) .* (x1_UPPER - x1_LOWER) .+ x1_LOWER

    return x1, u2, θσ
end

# ============================================================
# Initial training design D₀
# ============================================================
n_train = 10

x1_train, u2_train, θ_σ_train = mc_augmented(n_train)
x2_train = inverse_cdf_x2(u2_train, θ_σ_train)
y_train  = analytical_model(x1_train, x2_train)

data_aug_train = DataFrame(
    :x1  => x1_train,
    :u2  => u2_train,
    :θ_σ => θ_σ_train,
    :y   => y_train,
)

# initialize GP on θ-space
metamodel = GaussianProcess(data_aug_train, :y, kernel_type=GPSquaredExponential())
@time "fit!" fit!(metamodel)

# ============================================================
# Quick accuracy check on a held-out test set
# ============================================================
n_test = 1001
x1_test, u2_test, θ_σ_test = mc_augmented(n_test)
x2_test  = inverse_cdf_x2(u2_test, θ_σ_test)
y_test_v = analytical_model(x1_test, x2_test)

data_aug_test = DataFrame(
    :x1  => x1_test,
    :u2  => u2_test,
    :θ_σ => θ_σ_test,
    :y   => y_test_v,
)

μ_test, σ_test = predict(metamodel, Matrix(data_aug_test[:, X_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")

function cabo_loop(
    gp_init,
    data_aug_train;
    max_iter,
    Nx,
    Ng,
    n_new
)
    data = copy(data_aug_train)
    gp = gp_init
    for iter in 1:max_iter

        print("\n\nIteration n° $iter\n")
        # part 1 incumbent:
        # u = rand(Ng, Nx) # TODO needs to be only Nx fix later
        u = rand(Nx)
        
        result_star = @time "incument part " Metaheuristics.optimize(
            x -> bo_incumbent_objective(gp, x, u, Ng, Nx),
            bounds,
            PSO(N=100)
        )

        θ_star = minimizer(result_star)
        x1_star, θσ_star = θ_star
         
        # print("\n AT STAR: \n")
        V_star_samples, μ_V_star, σ_V_star = variance_moments_mcs(gp, x1_star, u, θσ_star, Ng, Nx)


        # print("θ_star: $θ_star\n")

        # part 2 acquisition:
        result_plus = @time "ei part" Metaheuristics.optimize(
            x -> - bo_ei_objective(gp, x, u, μ_V_star, Ng, Nx),
            bounds,
            PSO(N=100)
        )

        θ_plus = minimizer(result_plus)
        x1_plus, θσ_plus = θ_plus

        # convergence check
        V_samples, _, _ = variance_moments_mcs(gp, x1_plus, u, θσ_plus, Ng, Nx)

        Δ_BO = 1e-3

        # print("\nL_bo_plus: $L_bo_plus\n")
        print("θ_star: $θ_star\n")
        print("θ_plus: $θ_plus\n")

        L_BO_best = - Metaheuristics.minimum(result_plus)
        print("L_BO_best: $L_BO_best\n")


        # part 3 expensive model
        u2_new = rand(n_new)
        x2_new = inverse_cdf_x2(u2_new, θσ_plus)

        y_new = analytical_model(x1_plus, x2_new)

        new_data = DataFrame(
            :x1  => fill(x1_plus, n_new),
            :u2  => u2_new,
            :θ_σ => fill(θσ_plus, n_new),
            :y   => y_new
        )

        append!(data, new_data)

        # update GP
        gp = GaussianProcess(data, :y, kernel_type=GPSquaredExponential())
        fit!(gp)

        μ_pred, σ_pred = predict(gp, Matrix(data_aug_test[:, X_names]))

        println("MSE: $(round(mse(data_aug_test.y, μ_pred), digits=5))")
        println("Q²:  $(round(q2(data_aug_test.y, μ_pred), digits=5)) \n")


        # if L_BO_best < 1e-3
        #     break
        # end

    end
end

@time "\ncabo_loop: \n" cabo_loop(
    metamodel,
    data_aug_train;
    Ng          = 10,    # epistemic MC samples per BO step
    Nx          = 10,    # aleatory MC samples inside estimate_variance
    max_iter    = 3,     # hard cap
    n_new       = 2,      # true-model calls added per iteration
)

print("\n####################################################\n")