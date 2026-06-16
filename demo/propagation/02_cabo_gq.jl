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

# ──────────────────────────────────────────────────────────────────────────────
# bayesianoptimization.jl  –  BO engine for CABO
# ──────────────────────────────────────────────────────────────────────────────

φ(z) = pdf(Normal(), z)
Φ(z) = cdf(Normal(), z)


using FastGaussQuadrature   # add to Project.toml: FastGaussQuadrature

"""
    make_gh_nodes(m::Int) → (u_nodes, weights)

Build tensorised 2-D Gauss-Hermite quadrature in u-space.
m nodes per dimension → m² total points.
The nodes are converted from z-space (GH is defined for N(0,1))
to u-space (Φ(z)) to match the GP input encoding.
"""
function make_gh_nodes(m::Int = 15)
    t1d, w1d = gausshermite(m)          # physicist convention: Σwᵢf(tᵢ) ≈ ∫f(t)exp(−t²)dt

    z1d = t1d .* sqrt(2)                # N(0,1) nodes  (scale by √2)
    w1d = w1d ./ sqrt(π)               # N(0,1) weights (divide by √π,  NO exp term)
    # sanity: sum(w1d) should equal 1.0 exactly

    u_nodes = Matrix{Float64}(undef, m^2, 2)
    weights  = Vector{Float64}(undef, m^2)
    idx = 1
    for i in 1:m, j in 1:m
        u_nodes[idx, 1] = cdf(Normal(), z1d[i])
        u_nodes[idx, 2] = cdf(Normal(), z1d[j])
        weights[idx]    = w1d[i] * w1d[j]
        idx += 1
    end
    return u_nodes, weights
end

function make_gl_nodes(m::Int = 7)

    x1d, w1d = gausslegendre(m)   # nodes in [-1, 1]

    # map to [0, 1]
    u1d = (x1d .+ 1) ./ 2
    w1d = w1d ./ 2

    u_nodes = Matrix{Float64}(undef, m^2, 2)
    weights  = Vector{Float64}(undef, m^2)

    idx = 1
    for i in 1:m, j in 1:m
        u_nodes[idx, 1] = u1d[i]
        u_nodes[idx, 2] = u1d[j]
        weights[idx]    = w1d[i] * w1d[j]
        idx += 1
    end

    return u_nodes, weights
end

# TODO: add propagation for bounds
const GL_NODES, GL_WEIGHTS = make_gl_nodes(15)
const GH_NODES, GH_WEIGHTS = make_gh_nodes(15)   # deterministic points

"""
    estimate_propagation_gh(gp, Θμ1, Θμ2)

Gauss-Hermite version of estimate_propagation.
No Nx argument needed — nodes are fixed and deterministic.
"""
function estimate_propagation_gh(gp, Θμ1, Θμ2)
    Np  = size(GH_NODES, 1)
    X   = hcat(GH_NODES[:, 1],
               GH_NODES[:, 2],
               fill(Θμ1, Np),
               fill(Θμ2, Np))
    μ, σ = predict(gp, X)
    w    = GH_WEIGHTS

    μ_M  = dot(w, μ)                               # weighted mean
    # Weighted law of total variance
    σ_ep2 = dot(w, σ .^ 2)                         # E_z[σ²_GP]  epistemic
    σ_al2 = dot(w, (μ .- μ_M) .^ 2)               # Var_z[μ_GP] aleatoric
    σ_M2  = max(0.0, σ_ep2 + σ_al2)

    return μ_M, σ_M2
end

"""
    estimate_propagation(gp, u1, u2, Θμ1, Θμ2, Nx)

MC estimate of the mean and total variance of E_z[g(z, θ)] at a given θ:

  μ_M  =  E_z[ μ_GP(z, θ) ]                                (posterior mean)
  σ_M² =  E_z[ σ²_GP(z, θ) ] + Var_z[ μ_GP(z, θ) ]       (total uncertainty)

The second term decomposes as:
  - E_z[σ²_GP]   : epistemic uncertainty (limited training data)
  - Var_z[μ_GP]  : aleatory variability of g across z

Both are estimated from Nx MC / LHS samples u1, u2 ∈ [0,1].
"""
function estimate_propagation(gp, u1, u2, Θμ1, Θμ2, Nx)
    X   = hcat(u1, u2, fill(Θμ1, Nx), fill(Θμ2, Nx))
    μ, σ = predict(gp, X)
    μ_M  = mean(μ)
    σ_M2 = max(0.0, mean(σ .^ 2) + var(μ))   # guard against FP rounding < 0
    return μ_M, σ_M2
end


"""
    bo_incumbent_objective_response(gp, θ, u, Nx) → μ_M

Return the **pure posterior-mean** estimate of E_z[g(z, θ)].

BUG FIX (was: `μ_M + α·√σ_M2`):
  The incumbent θ* must reflect the current *best-known* estimate of
  the objective, not an optimistic UCB/LCB.  Adding α·σ biases the
  search toward uncertain regions, making μ_M_star inaccurate and
  corrupting every AEI call that follows.

Usage: multiply externally by +1 (minimisation) or −1 (maximisation)
before passing as the PSO objective.
"""
function bo_incumbent_objective_response(gp, θ)
    μ_M, σ_M = estimate_propagation_gh(gp, θ[1], θ[2])
    return μ_M
end


"""
    AEI_objective(gp, θ, u, Nx, μ_M_star, sign_dir) → −AEI

Augmented Expected Improvement for CABO.
  sign_dir = +1  →  minimisation   EI = E[ max(η*  − Y(θ), 0) ]
  sign_dir = −1  →  maximisation   EI = E[ max(Y(θ) − η*,  0) ]

Returns **−AEI** (≤ 0) so PSO (a minimiser) effectively maximises it.

BUG FIX – maximisation branch (was: sign·(μ_M_star−μ_M)·Φ(z) with z=(μ_M_star−μ_M)/σ_M):
  For maximisation the z-score must be (μ_M − μ_M_star)/σ_M (positive
  when improvement is likely). Using the minimisation z-score with a sign
  flip gives Φ(negative z) < 0.5 for all promising points, which drives
  the AEI toward zero exactly where we want it to be large – causing the
  observed stagnation where every iteration returns the same θ.

Correct closed-form EI for each direction:
  MIN: (η* − μ_M)·Φ((η* − μ_M)/σ_M) + σ_M·φ((η* − μ_M)/σ_M)
  MAX: (μ_M − η*)·Φ((μ_M − η*)/σ_M) + σ_M·φ((μ_M − η*)/σ_M)
"""
function AEI_objective(gp, θ, μ_M_star, sign_dir)

    μ_M, σ_M2 = estimate_propagation_gh(gp, θ[1], θ[2])
    σ_M       = sqrt(max(σ_M2, 1e-12))

    σ_M < 1e-12 && return 0.0      # numerically flat region → no gain

    if sign_dir == 1                # ── minimisation ──────────────────────
        z   = (μ_M_star - μ_M) / σ_M
        aei = (μ_M_star - μ_M) * Φ(z) + σ_M * φ(z)
    else                            # ── maximisation ──────────────────────
        z   = (μ_M - μ_M_star) / σ_M   # ← sign-flipped z  (the critical fix)
        aei = (μ_M - μ_M_star) * Φ(z) + σ_M * φ(z)
    end

    return -aei     # PSO minimises → return −AEI (always ≤ 0)
end

function AEI_objective_direct(gp, x, f_best, sign_dir)

    μ, σ = predict(gp, reshape(x,1,:))

    μ = μ[1]
    σ = max(σ[1], 1e-12)

    if sign_dir == 1                # ── minimisation ──────────────────────
        z   = (f_best - μ) / σ
        aei = (f_best - μ) * Φ(z) + σ * φ(z)
    else                            # ── maximisation ──────────────────────
        z   = (μ - f_best) / σ   # ← sign-flipped z  (the critical fix)
        aei = (μ - f_best) * Φ(z) + σ * φ(z)
    end

    return -aei     # PSO minimises → return −AEI (always ≤ 0)
end

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
const bounds_z = boxconstraints(lb = [-3.0, -3.0], ub = [3.0, 3.0])

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

n_train, n_test = 20, 1001
data_aug_train = build_design(g_function, n_train, x_names)
data_aug_test  = build_design(g_function, n_test,  x_names)

# initialize GP on θ-space
kernel() = GPMatern52()
metamodel = GaussianProcess(data_aug_train, :y, kernel_type=kernel())
@time "fit!" fit!(metamodel)

μ_test, σ_test = @time "predict:" predict(metamodel, Matrix(data_aug_test[:, x_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")


function make_pso(; N::Int=50, iters::Int=100, ω=0.8, C1=2.0, C2=2.0)
    p = PSO(N=N, C1=C1, C2=C2, ω=ω)
    p.options.iterations = iters
    return p
end


function cabo_loop(
    gp_init,
    data_aug_train,
    x_names;                                # e.g. [:u1, :u2, :θ_μ1, :θ_μ2]
    max_iter  :: Int     = 20,
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
 
    # ── Shared kernel factory (same kernel for every GP fit) ──────────────────
    # FIX: was GPSquaredExponential() in loop vs mixed kernel for initial fit.
 
    for iter in 1:max_iter
        println("\n━━━ CABO Iteration $iter / $max_iter  [$(direction)] ━━━")
 
        # ════ Part 1: BO engine ═══════════════════════════════════════════════
 
        # 1a. Incumbent: θ* = argmin/argmax E_z[μ_GP(z, θ)]
        res_star = Metaheuristics.optimize(
            θ -> sign_dir * bo_incumbent_objective_response(gp, θ),
            bounds_θ,
            make_pso()          # ← fresh PSO
        )
        θ_star           = minimizer(res_star)
        μ_M_star, _      = estimate_propagation_gh(gp, θ_star[1], θ_star[2])
        println("  Incumbent  θ* = ($(round(θ_star[1],digits=4)), $(round(θ_star[2],digits=4)))" *
                "   E[g|θ*] ≈ $(round(μ_M_star, sigdigits=5))")
 
        # 1b. θ⁺ = argmax AEI(θ ; μ_M_star)
        res_plus = Metaheuristics.optimize(
            θ -> AEI_objective(gp, θ, μ_M_star, sign_dir),
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
    
    res_bound = Metaheuristics.optimize(
        θ -> sign_dir * bo_incumbent_objective_response(gp, θ),
        bounds_θ,
        make_pso(N=50, iters=300)
    )
    θ_bound  = minimizer(res_bound)
    μ_bound, _ = estimate_propagation_gh(gp, θ_bound[1], θ_bound[2])
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
    max_iter = 20,
    direction = :min,
    tol       = 1e-4,
)
 
cabo_max = @time "CABO MAX" cabo_loop(
    metamodel,
    data_aug_train,
    x_names;
    max_iter = 20,
    direction = :max,
    tol       = 1e-4,
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
println("Expected:  MIN ≈ −1.35  at (−0.56, 0.53)")
println("           MAX ≈  1.33  at ( 0.56, 0.80)")
println("="^60)



# plot related
Θs_min = reduce(hcat, cabo_min.θ_history)'
Θs_max = reduce(hcat, cabo_max.θ_history)'
using Plots, Statistics

# ============================================================
# Grid over epistemic space (x1, θσ)
# ============================================================
n_μ1 = 200
n_μ2 = 200

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
    colorbar=true,
    xlims = (-2, 2),
    ylims = (-2, 2)
)

# ── overlay the initial training points (μ1, μ2 columns from data_aug) ──────
scatter!(plt,
    data_aug_train.θ_μ1, data_aug_train.θ_μ2;
    marker = :diamond, color = :cyan, ms = 5,
    label  = "Initial samples", markerstrokewidth=0
)

scatter!(plt,
    Θs_min[:, 1], Θs_min[:, 2];
    marker = :cross, color = :green, ms = 5,
    label  = "added min samples", markerstrokewidth=2
)

scatter!(plt,
    Θs_max[:, 1], Θs_max[:, 2];
    marker = :cross, color = :red, ms = 5,
    label  = "added max samples", markerstrokewidth=2
)

scatter!(plt,
    [cabo_min.θ_bound[1]], [cabo_min.θ_bound[2]];
    marker = :star, color = :green, ms = 7,
    label  = "cabo min", markerstrokewidth=1
)

scatter!(plt,
    [cabo_max.θ_bound[1]], [cabo_max.θ_bound[2]];
    marker = :star, color = :red, ms = 7,
    label  = "cabo max", markerstrokewidth=1
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