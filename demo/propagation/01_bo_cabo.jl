using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Distributions
using ParameterHandling
using LinearAlgebra
using Metaheuristics
using QuasiMonteCarlo

# response function
function g_function(x1::Float64, x2::Float64)
    α_g = [2.0 3.0 1.0 4.0; 3.0 2.0 4.0 1.0]
    β_g = [-0.5 0.5 -0.5 0.5; -0.5 -0.5 0.5 0.5]
    c_g = [1.0, -1.5, -1.5, 2.0]

    result = 0.0
    for i in 1:4
        result += c_g[i] * exp(-α_g[1,i] * (x1 - β_g[1,i])^2 - α_g[2,i] * (x2 - β_g[2,i])^2)
    end
    return result
end

const σ_FIXED = 0.1

const θ_μ1_UPPER = 1.5
const θ_μ1_LOWER = -1.5

const θ_μ2_UPPER = 1.5
const θ_μ2_LOWER = -1.5

# bounds must be boxconstraints
lb = [θ_μ1_LOWER, θ_μ2_LOWER]
ub = [θ_μ1_UPPER, θ_μ2_UPPER]

bounds_θ = Metaheuristics.boxconstraints(lb=lb, ub=ub)
bounds_u = Metaheuristics.boxconstraints(lb = [0.0, 0.0], ub = [1.0, 1.0])


function build_design(physical_model, n_samples::Int, x_names::Vector{Symbol}; seed::Int=42)
    Random.seed!(seed)

    lhs = QuasiMonteCarlo.sample(
        n_samples,
        [0.0, 0.0],
        [1.0, 1.0],
        LatinHypercubeSample()
    )'

    # 1. sampling θ
    θ_μ1 = θ_μ1_LOWER .+ (θ_μ1_UPPER - θ_μ1_LOWER) .* lhs[:, 1]
    θ_μ2 = θ_μ2_LOWER .+ (θ_μ2_UPPER - θ_μ2_LOWER) .* lhs[:, 2]

    # 2. physical sampling
    x1 = rand.(Normal.(θ_μ1, σ_FIXED))
    x2 = rand.(Normal.(θ_μ2, σ_FIXED))

    # 2. encoding
    u1 = [cdf(Normal(θ_μ1[i], σ_FIXED), x1[i]) for i in eachindex(x1)]
    u2 = [cdf(Normal(θ_μ2[i], σ_FIXED), x2[i]) for i in eachindex(x2)]

    # Evaluate true model at physical inputs
    y = physical_model.(x1, x2)

    data_aug_train = DataFrame(
        x_names[1]  => u1,
        x_names[2]  => u2,
        x_names[3]  => θ_μ1,
        x_names[4]  => θ_μ2,
        :y          => y,
    )
    return data_aug_train
end


function qmc_samples(Nx)
    X = QuasiMonteCarlo.sample(
        Nx,
        zeros(2),
        ones(2),
        SobolSample()
    )'

    u1 = X[:, 1]
    u2 = X[:, 2]

    return u1, u2
end

# ============================================================
# Build initial design D₀ and evaluate true model
# ============================================================

x_names = [:u1, :u2, :θ_μ1, :θ_μ2]

n_train, n_test = 50, 1001
data_aug_train = build_design(g_function, n_train, x_names; seed=1)
data_aug_test  = build_design(g_function, n_test,  x_names; seed=2)

# initialize GP on θ-space
metamodel = GaussianProcess(data_aug_train, :y, kernel_type=GPMatern52())
@time "fit!" fit!(metamodel)

μ_test, σ_test = @time "predict:" predict(metamodel, Matrix(data_aug_test[:, x_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")

function cabo_loop(
    gp_init,
    data_aug_train;
    max_iter,
    Nx,
    n_new
)
    data = copy(data_aug_train)
    gp = gp_init
    
    # part 1 incumbent:    
    u1, u2 = qmc_samples(Nx)

    for iter in 1:max_iter

        print("\n\nIteration n° $iter\n")
        
        # Part 1: BO
        θ_result_star = Metaheuristics.optimize(
            θ -> bo_incumbent_objective_response(gp, θ, [u1, u2], Nx),
            bounds_θ,
            PSO(N = 30)
        )

        θ_star = minimizer(θ_result_star)
        Θμ1_star, Θμ2_star = θ_star
        print("θ_star: $θ_star\n")

        μ_M_star, σ_M2_star = estimate_propagation(gp, u1, u2, Θμ1_star, Θμ2_star, Nx)
        
        θ_result_plus = Metaheuristics.optimize(
            θ -> - AEI_objective(gp, θ, [u1, u2], Nx, μ_M_star),
            bounds_θ,
            PSO(N = 30)
        )

        θ_plus = minimizer(θ_result_plus)
        L_BO = -AEI_objective(gp, θ_plus, [u1, u2], Nx, μ_M_star)
        θμ1_plus, θμ2_plus = θ_plus
        print("θ_plus: $θ_plus\n")

        # Part 2: BC
        u_result_plus = Metaheuristics.optimize(
            u -> - PVC(gp, u, θ_plus),
            bounds_u,
            PSO(N = 30)
        )

        u_plus = minimizer(u_result_plus)
        L_BC = PVC(gp, u_plus, θ_plus)
        u1_plus, u2_plus = u_plus
        print("u_plus: $u_plus\n")

        # adding new sample point:

        # Evaluate true model at physical inputs
        x1_plus = quantile.(Normal.(θμ1_plus, σ_FIXED), u1_plus)
        x2_plus = quantile.(Normal.(θμ2_plus, σ_FIXED), u2_plus)
        y_plus = g_function.(x1_plus, x2_plus)

        data_aug_plus = DataFrame(
            x_names[1]  => u1_plus,
            x_names[2]  => u2_plus,
            x_names[3]  => θμ1_plus,
            x_names[4]  => θμ2_plus,
            :y          => y_plus,
        )
        
        append!(data, data_aug_plus)

        # update GP
        gp = GaussianProcess(data, :y, kernel_type=GPMatern52())
        fit!(gp)

        μ_pred, σ_pred = predict(gp, Matrix(data_aug_test[:, x_names]))

        println("MSE: $(round(mse(data_aug_test.y, μ_pred), digits=5))")
        println("Q²:  $(round(q2(data_aug_test.y, μ_pred), digits=5)) \n")

        println("L_BO: $L_BO")
        println("L_BC: $L_BC")

    end

    return gp
end

@time "\ncabo_loop: \n" cabo_loop(
    metamodel,
    data_aug_train;
    Nx          = 500,    # aleatory MC samples inside estimate_variance
    max_iter    = 5,     # hard cap
    n_new       = 1,      # true-model calls added per iteration
)

print("\n# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #\n")