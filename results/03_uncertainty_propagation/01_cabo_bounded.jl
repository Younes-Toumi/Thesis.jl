using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Distributions
using ParameterHandling
using LinearAlgebra
using Metaheuristics
using QuasiMonteCarlo

Random.seed!(42)

const x1_UPPER =  Float64(π)
const x1_LOWER = -Float64(π)

const x2_UPPER =  Float64(π)
const x2_LOWER = -Float64(π)

const x3_UPPER =  Float64(π)
const x3_LOWER = -Float64(π)

# bounds must be boxconstraints
lb = [x1_LOWER, x2_LOWER, x3_LOWER]
ub = [x1_UPPER, x2_UPPER, x3_UPPER]

const bounds_x = boxconstraints(lb = [x1_LOWER, x2_LOWER, x3_LOWER], ub = [x1_UPPER, x2_UPPER, x3_UPPER])

function build_design(physical_model, n_samples::Int, x_names::Vector{Symbol})
   
    lhs = QuasiMonteCarlo.sample(
        n_samples,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        LatinHypercubeSample()
    )'

    # 1. sampling x
    x1 = x1_LOWER .+ (x1_UPPER - x1_LOWER) .* lhs[:, 1]
    x2 = x2_LOWER .+ (x2_UPPER - x2_LOWER) .* lhs[:, 2]
    x3 = x3_LOWER .+ (x3_UPPER - x3_LOWER) .* lhs[:, 3]

    # Evaluate true model at physical inputs
    y = physical_model.(x1, x2, x3)

    data_aug_train = DataFrame(
        x_names[1]  => x1,
        x_names[2]  => x2,
        x_names[3]  => x3,

        :y          => y,
    )
    return data_aug_train
end

# ============================================================
# Build initial design D₀ and evaluate true model
# ============================================================
physical_model = ishigami

x_names = [:x1, :x2, :x3]

n_train, n_test = 50, 1001
data_aug_train = build_design(physical_model, n_train, x_names)
data_aug_test  = build_design(physical_model, n_test,  x_names)

# initialize GP on θ-space
metamodel = GaussianProcess(data_aug_train, :y, kernel_type= GPMatern52())
@time "fit!" fit!(metamodel)

μ_test, σ_test = @time "predict:" predict(metamodel, Matrix(data_aug_test[:, x_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")

function make_pso(; N::Int=80, iters::Int=200, ω=0.8, C1=2.0, C2=2.0)
    p = PSO(N=N, C1=C1, C2=C2, ω=ω)
    p.options.iterations = iters
    return p
end

# # # # # # # # # # # # # # # # # # # # #
function cabo_loop(
    gp_init,
    data_aug_train,
    x_names;
    max_iter::Int = 20,
    direction::Symbol = :min,
    tol::Float64 = 1e-8
)

    data = copy(data_aug_train)
    gp   = gp_init
    sign_dir = (direction == :min) ? +1 : -1

    x_history    = Vector{Vector{Float64}}()
    L_BO_history = Float64[]

    for iter in 1:max_iter

        println("\n━━━ CABO Iteration $iter / $max_iter ━━━")

        # ─────────────────────────────────────────────
        # BO step: optimize surrogate directly in x-space
        # ─────────────────────────────────────────────
        res = Metaheuristics.optimize(
            x -> sign_dir * mean(predict(gp, reshape(x,1,:))[1]),
            bounds_x,
            make_pso()
        )

        x_star = minimizer(res)
        x1_star, x2_star, x3_star = x_star
        y_star = physical_model(x1_star, x2_star, x3_star)

        println("  incumbent x* = $(round.(x_star, digits=4))  y ≈ $(round(y_star, digits=5))")


        μ_best, σ_best = predict(gp, reshape(x_star,1,:))
        # ─────────────────────────────────────────────
        # acquisition step (still valid if GP uncertainty used)
        # ─────────────────────────────────────────────

        res_plus = Metaheuristics.optimize(
            x -> AEI_objective_direct(gp, x, μ_best[1], sign_dir),
            bounds_x,
            make_pso()
        )

        x_plus = minimizer(res_plus)
        x1_plus, x2_plus, x3_plus = x_plus
        y_plus = physical_model(x1_plus, x2_plus, x3_plus)

        L_BO   = -minimum(res_plus)

        println("  acquisition x⁺ = $(round.(x_plus, digits=4)) AEI = $(round(L_BO, digits=4))")

        # ─────────────────────────────────────────────
        # true evaluation
        # ─────────────────────────────────────────────

        append!(data, DataFrame(
            x_names[1] => [x1_plus],
            x_names[2] => [x2_plus],
            x_names[3] => [x3_plus],
            :y         => [y_plus],
        ))

        gp = GaussianProcess(data, :y, kernel_type = GPMatern52())
        fit!(gp)

        push!(x_history, copy(x_plus))
        push!(L_BO_history, L_BO)

        if L_BO < tol
            println("\n✓ converged")
            break
        end
    end

    result_bound = Metaheuristics.optimize(
        x -> sign_dir * (predict(gp, reshape(x,1,:))[1])[1],
        bounds_x,
        make_pso(N=50, iters=200)
    )

    x_bound = minimizer(result_bound)
    y_bound = (predict(gp, reshape(x_bound,1,:))[1])[1]
    dir_str = uppercase(string(direction))
    println("\n  ► $(dir_str) bound ≈ $(round(y_bound, sigdigits=5))" *
            "  at  x = [$(round(x_bound[1],digits=4)), $(round(x_bound[2],digits=4)), $(round(x_bound[3],digits=4))]")
 
    return (
        gp = gp,
        data = data,
        x_bound = x_bound,
        y_bound = y_bound,
        x_history = x_history,
        L_BO_history = L_BO_history
    )
end


x_names = [:x1, :x2, :x3]
 
cabo_min = @time "CABO MIN" cabo_loop(
    metamodel,
    data_aug_train,
    x_names;
    max_iter = 20,
    direction = :min,
    tol       = 1e-5,
)
 
cabo_max = @time "CABO MAX" cabo_loop(
    metamodel,
    data_aug_train,
    x_names;
    max_iter = 20,
    direction = :max,
    tol       = 1e-5,
)
 
# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("CABO results")
println("="^60)
println("MIN  g ≈ $(round(cabo_min.y_bound, digits=4))" *
        "  at x = $(round.(cabo_min.x_bound, digits=3))")
println("MAX  g ≈ $(round(cabo_max.y_bound, digits=4))" *
        "  at x = $(round.(cabo_max.x_bound, digits=3))")
println("-"^60)
println("="^60)






# plot related
xs_min = reduce(hcat, cabo_min.x_history)'
xs_max = reduce(hcat, cabo_max.x_history)'
using Plots, Statistics

# ============================================================
# Grid over epistemic space (x1, θσ)
# ============================================================
n_points = 100

x1_grid = range(-pi, pi, length=n_points)
x2_grid = range(-pi, pi, length=n_points)
x3_grid = range(-pi, pi, length=n_points)


ResponseSurface = zeros(n_points, n_points, n_points)

# ============================================================
# Compute response
# ============================================================
for (i, x1_v) in enumerate(x1_grid)
    for (j, x2_v) in enumerate(x2_grid)
        for (k, x3_v) in enumerate(x3_grid)
            ResponseSurface[k, j, i]  = physical_model(x1_v, x2_v, x3_v)
        end
    end
end


# ── mark true min and max in epistemic space ──────────────────────────────────
min_idx = argmin(ResponseSurface)
max_idx = argmax(ResponseSurface)

x1_min, x2_min, x3_min, y_min = x1_grid[min_idx[3]], x2_grid[min_idx[2]], x3_grid[min_idx[1]], minimum(ResponseSurface)
x1_max, x2_max, x3_max, y_max = x1_grid[max_idx[3]], x2_grid[max_idx[2]], x3_grid[max_idx[1]], maximum(ResponseSurface)


println("\n" * "="^60)
println("Analytical results")
println("="^60)
println("MIN  g ≈ $(round(y_min, digits=4))" *
        "  at x = $(round.([x1_min, x2_min, x3_min], digits=3))")
println("MAX  g ≈ $(round(y_max, digits=4))" *
        "  at x = $(round.([x1_max, x2_max, x3_max], digits=3))")
println("-"^60)
println("="^60)

p1 = scatter(
    cabo_min.L_BO_history,
    xlabel = "Iteration",
    ylabel = "L_AEI",
    title = "MIN: L_AEI History",
    legend = false
)

p2 = scatter(
    cabo_min.L_BO_history,
    xlabel = "Iteration",
    ylabel = "L_AEI",
    title = "MAX: L_AEI History",
    legend = false
)

history = plot(
    p1, p2,
    layout = (1, 2),
    size = (900, 500)
)

display(history)
