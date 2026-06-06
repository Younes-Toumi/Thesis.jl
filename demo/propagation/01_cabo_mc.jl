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

function g_expected(μ1::Float64, μ2::Float64; σ::Float64 = 0.1)
    α_g = [2.0 3.0 1.0 4.0; 3.0 2.0 4.0 1.0]'
    β_g = [-0.5 0.5 -0.5 0.5; -0.5 -0.5 0.5 0.5]'
    c_g = [1.0, -1.5, -1.5, 2.0]'

    result = 0.0
    σ² = σ^2

    for i in 1:4

        α1 = α_g[i,1]
        α2 = α_g[i,2]

        β1 = β_g[i,1]
        β2 = β_g[i,2]

        # E[exp(-α(X-β)^2)]
        term1 =
            exp(
                -α1 * (μ1 - β1)^2 /
                (1 + 2 * α1 * σ²)
            ) /
            sqrt(1 + 2 * α1 * σ²)

        term2 =
            exp(
                -α2 * (μ2 - β2)^2 /
                (1 + 2 * α2 * σ²)
            ) /
            sqrt(1 + 2 * α2 * σ²)

        result += c_g[i] * term1 * term2
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

const bounds_θ = boxconstraints(lb = [θ_μ1_LOWER, θ_μ2_LOWER], ub = [θ_μ1_UPPER, θ_μ2_UPPER])
const bounds_z = boxconstraints(lb = [-1.0, -1.0], ub = [1.0, 1.0])

function build_design(physical_model, n_samples::Int, x_names::Vector{Symbol})
   
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

# ============================================================
# Build initial design D₀ and evaluate true model
# ============================================================

x_names = [:u1, :u2, :θ_μ1, :θ_μ2]

n_train, n_test = 40, 1001
data_aug_train = build_design(g_function, n_train, x_names)
data_aug_test  = build_design(g_function, n_test,  x_names)

# initialize GP on θ-space
kernel() = GPMatern52()
metamodel = GaussianProcess(data_aug_train, :y, kernel_type= kernel())
@time "fit!" fit!(metamodel)

μ_test, σ_test = @time "predict:" predict(metamodel, Matrix(data_aug_test[:, x_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")


function make_pso(; N::Int=80, iters::Int=200, ω=0.8, C1=2.0, C2=2.0)
    p = PSO(N=N, C1=C1, C2=C2, ω=ω)
    p.options.iterations = iters
    return p
end


function cabo_loop(
    gp_init,
    data_aug_train,
    x_names;                                # e.g. [:u1, :u2, :θ_μ1, :θ_μ2]
    max_iter  :: Int     = 20,
    Nx        :: Int     = 500,
    direction :: Symbol  = :min,            # :min  or  :max
    tol       :: Float64 = 1e-8
)
    data     = copy(data_aug_train)
    gp       = gp_init
    sign_dir = (direction == :min) ? +1 : -1   # FIX: was `sign` (shadows Base.sign)
 
    θ_history    = Vector{Vector{Float64}}()
    u_history    = Vector{Vector{Float64}}()
    z_history    = Vector{Vector{Float64}}()
    L_BO_history = Float64[]
    L_BC_history = Float64[]
 
    # ── Fixed LHS samples for MC integration (stable across all iterations) ───
    # Using quasi-random samples reduces variance in the AEI landscape compared
    # to pure MC, making the BO optimisation more reliable.
    lhs         = QuasiMonteCarlo.sample(Nx, [0.0, 0.0], [1.0, 1.0],
                                         LatinHypercubeSample())'
    u1_mc = lhs[:, 1]    # length-Nx vector, fixed for entire loop
    u2_mc = lhs[:, 2]
 
    # ── Shared kernel factory (same kernel for every GP fit) ──────────────────
 
    for iter in 1:max_iter
        println("\n━━━ CABO Iteration $iter / $max_iter  [$(direction)] ━━━")
 
        u1 = u1_mc
        u2 = u2_mc
 
        # ════ Part 1: BO engine ═══════════════════════════════════════════════
 
        # 1a. Incumbent: θ* = argmin/argmax E_z[μ_GP(z, θ)]
        #     FIX: fresh PSO each call; incumbent uses pure mean (see .jl)
        res_star = Metaheuristics.optimize(
            θ -> sign_dir * bo_incumbent_objective_response(gp, θ, [u1, u2], Nx),
            bounds_θ,
            make_pso()          # ← fresh PSO
        )
        θ_star           = minimizer(res_star)
        μ_M_star, _      = estimate_propagation(gp, u1, u2, θ_star[1], θ_star[2], Nx)
        println("  Incumbent  θ* = ($(round(θ_star[1],digits=4)), $(round(θ_star[2],digits=4)))" *
                "   E[g|θ*] ≈ $(round(μ_M_star, sigdigits=5))")
 
        # 1b. θ⁺ = argmax AEI(θ ; μ_M_star)
        #     FIX: fresh PSO; correct AEI formula for both directions (see .jl)
        res_plus = Metaheuristics.optimize(
            θ -> AEI_objective(gp, θ, [u1, u2], Nx, μ_M_star, sign_dir),
            bounds_θ,
            make_pso()          # ← fresh PSO
        )
        θ_plus        = minimizer(res_plus)
        L_BO          = -minimum(res_plus)      # AEI value (positive)
        θμ1_plus, θμ2_plus = θ_plus
        println("  Acquisition θ⁺ = ($(round(θ_plus[1],digits=4)), $(round(θ_plus[2],digits=4)))" *
                "   AEI = $(round(L_BO, sigdigits=4))")
 
        # ════ Part 2: BC engine ═══════════════════════════════════════════════
 
        # z⁺ = argmax σ²_GP(z, θ⁺)   (most uncertain aleatory point at θ⁺)
        res_z  = Metaheuristics.optimize(
            z -> BC_objective_z(gp, z, θ_plus),
            bounds_z,
            make_pso()          # ← fresh PSO
        )
        z_plus        = minimizer(res_z)
        u_plus        = cdf.(Normal(), z_plus)
        L_BC          = -minimum(res_z)         # PVC value (positive)
        u1_plus, u2_plus = u_plus
        println("  BC sample   z⁺ = ($(round(z_plus[1],digits=4)), $(round(z_plus[2],digits=4)))" *
                "   PVC = $(round(L_BC, sigdigits=4))")
 
        # ════ True-model evaluation ═══════════════════════════════════════════
 
        x1_plus = θμ1_plus + σ_FIXED * z_plus[1]
        x2_plus = θμ2_plus + σ_FIXED * z_plus[2]
        y_plus  = g_function(x1_plus, x2_plus)
 
        # ════ Augment training data ═══════════════════════════════════════════
        # FIX: wrap scalars in single-element arrays for DataFrame constructor
        append!(data, DataFrame(
            x_names[1] => [u1_plus],
            x_names[2] => [u2_plus],
            x_names[3] => [θμ1_plus],
            x_names[4] => [θμ2_plus],
            :y         => [y_plus],
        ))
 
        # ════ Refit GP with consistent kernel ════════════════════════════════
        gp = GaussianProcess(data, :y, kernel_type = kernel())
        fit!(gp)
 
        push!(θ_history,    copy(θ_plus))
        push!(u_history,    copy(u_plus))
        push!(z_history,    copy(z_plus))
        push!(L_BO_history, L_BO)
        push!(L_BC_history, L_BC)
 
        # Convergence: AEI has dropped below tolerance
        if L_BO < tol
            println("\n  ✓ Converged (AEI = $L_BO < tol = $tol) at iteration $iter")
            break
        end
    end
 
    # ════ Final bound estimate using the enriched GP ═════════════════════════
    # Use a larger MC set for a low-noise final answer.
    Nx_final  = 5_000
    lhs_final = QuasiMonteCarlo.sample(Nx_final, [0.0, 0.0], [1.0, 1.0],
                                       LatinHypercubeSample())'
    u1_f = lhs_final[:, 1];  u2_f = lhs_final[:, 2]
 
    res_bound = Metaheuristics.optimize(
        θ -> sign_dir * bo_incumbent_objective_response(gp, θ, [u1_f, u2_f], Nx_final),
        bounds_θ,
        make_pso(N=50, iters=200)
    )
    θ_bound  = minimizer(res_bound)
    μ_bound, _ = estimate_propagation(gp, u1_f, u2_f, θ_bound[1], θ_bound[2], Nx_final)
    dir_str = uppercase(string(direction))
    println("\n  ► $(dir_str) bound ≈ $(round(μ_bound, sigdigits=5))" *
            "  at  θ = ($(round(θ_bound[1],digits=4)), $(round(θ_bound[2],digits=4)))")
 
    return (
        gp           = gp,
        data         = data,
        θ_bound      = θ_bound,
        μ_bound      = μ_bound,
        θ_history    = θ_history,
        u_history    = u_history,
        z_history    = z_history,
        L_BO_history = L_BO_history,
        L_BC_history = L_BC_history,
    )
end




# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
 
x_names = [:u1, :u2, :θ_μ1, :θ_μ2]
 
cabo_min = @time "CABO MIN" cabo_loop(
    metamodel,
    data_aug_train,
    x_names;
    Nx       = 500,
    max_iter = 20,
    direction = :min,
    tol       = 1e-6,
)
 
cabo_max = @time "CABO MAX" cabo_loop(
    metamodel,
    data_aug_train,
    x_names;
    Nx       = 500,
    max_iter = 20,
    direction = :max,
    tol       = 1e-6,
)
 
# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("CABO results")
println("="^60)
println("MIN  E[g] ≈ $(round(cabo_min.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_min.θ_bound, digits=3))")
println("MAX  E[g] ≈ $(round(cabo_max.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_max.θ_bound, digits=3))")
println("-"^60)
println("Expected:  MIN ≈ −1.35  at (−0.59, 0.51)")
println("           MAX ≈  1.33  at ( 0.55, 0.83)")
println("="^60)



# plot related
Θs_min = reduce(hcat, cabo_min.θ_history)'
Θs_max = reduce(hcat, cabo_max.θ_history)'
using Plots, Statistics

# ============================================================
# Grid over epistemic space (x1, θσ)
# ============================================================
n_μ1 = 100
n_μ2 = 100

μ1_grid = range(-1.5, 1.5, length=n_μ1)
μ2_grid  = range(-1.5, 1.5, length=n_μ2)

MeanSurface = zeros(n_μ1, n_μ2)

# ============================================================
# Compute response
# ============================================================
for (i, μ1_v) in enumerate(μ1_grid)
    for (j, μ2_v) in enumerate(μ2_grid)
        MeanSurface[j, i]  = g_expected(μ1_v, μ2_v)
    end
end

# ============================================================
# Heatmap
# ============================================================
plt = heatmap(
    μ1_grid,
    μ2_grid,
    MeanSurface,
    xlabel="μ1",
    ylabel="μ2",
    c=:thermal,
    title="expected response function: E[g(x1, x2)]",
    colorbar=true
)

# ── overlay the initial training points (μ1, μ2 columns from data_aug) ──────
scatter!(plt,
    data_aug_train.θ_μ1, data_aug_train.θ_μ2;
    marker = :diamond, color = :cyan, ms = 5,
    label  = "Initial samples", markerstrokewidth=0
)

scatter!(plt,
    Θs_min[:, 1], Θs_min[:, 2];
    marker = :cross, color = :green, ms = 7,
    label  = "added min samples", markerstrokewidth=3
)

scatter!(plt,
    Θs_max[:, 1], Θs_max[:, 2];
    marker = :cross, color = :red, ms = 7,
    label  = "added max samples", markerstrokewidth=3
)

# ── mark true min and max in epistemic space ──────────────────────────────────
min_idx = argmin(MeanSurface)
max_idx = argmax(MeanSurface)

x_min, y_min, z_min = μ1_grid[min_idx[2]], μ1_grid[min_idx[1]], minimum(MeanSurface)
x_max, y_max, z_max = μ1_grid[max_idx[2]], μ1_grid[max_idx[1]], maximum(MeanSurface)

dy = 0.2

scatter!(plt,
    [μ1_grid[min_idx[2]]], [μ2_grid[min_idx[1]]];
    marker = :circle, color = :green, ms = 5, label = "True min"
)
annotate!(
    x_min, y_min + dy,
    text("($(round(x_min, digits=2)), $(round(y_min, digits=2)), $(round(z_min, digits=2)))", :black, 8)
)


scatter!(plt,
    [μ1_grid[max_idx[2]]], [μ2_grid[max_idx[1]]];
    marker = :rect, color = :red, ms = 5, label = "True max"
)

annotate!(
    x_max, y_max + dy,
    text("($(round(x_max, digits=2)), $(round(y_max, digits=2)), $(round(z_max, digits=2)))", :black, 8)
)

println("y range: [$(round(minimum(MeanSurface), digits=2)),  $(round(maximum(MeanSurface), digits=2))]")
print([μ1_grid[min_idx[2]]], [μ2_grid[min_idx[1]]])
print("\n")
print([μ1_grid[max_idx[2]]], [μ2_grid[max_idx[1]]])

p1 = scatter(
    cabo_min.L_BO_history,
    xlabel = "Iteration",
    ylabel = "L_BO",
    title = "MIN: L_BO History",
    legend = false
)

p2 = scatter(
    cabo_min.L_BC_history,
    xlabel = "Iteration",
    ylabel = "L_BC",
    title = "MIN: L_BC History",
    legend = false
)

p3 = scatter(
    cabo_max.L_BO_history,
    xlabel = "Iteration",
    ylabel = "L_BO",
    title = "MAX: L_BO History",
    legend = false
)

p4 = scatter(
    cabo_max.L_BC_history,
    xlabel = "Iteration",
    ylabel = "L_BC",
    title = "MAX: L_BC History",
    legend = false
)

history = plot(
    p1, p2, p3, p4,
    layout = (2, 2),
    size = (900, 700)
)

display(plt)
display(history)