using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Printf


# ==============================================================================
# 10. Usage — mirrors your UQ.jl example end to end
# ==============================================================================

# parametric inputs
x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input
 
x_names, w_names, u_names, v_names = spec_names(specs)
y_symbol = model_gfunction.name
physical_model = g_function


n_train, n_test = 50, 1001


data_aug_train, data_phys_train =    build_augmented_design(model_gfunction, specs, n_train)
data_aug_test,  data_phys_test  =    build_augmented_design(model_gfunction, specs, n_test)


# # initialize GP on θ-space
kernel() = GPMatern52()
metamodel = GaussianProcess(data_aug_train, :y, kernel_type=kernel())
@time "fit!" fit!(metamodel)

μ_test, σ_test = @time "predict:" predict(metamodel, Matrix(data_aug_test[:, w_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")

Ng = 1000
Nx = 2000

print("Params: Ng = $Ng, Nx = $Nx\n")

cabo_min = @time "cabo min loop" cabo_loop(
    model_gfunction,        # physical_model — explicit, no longer a global lookup
    metamodel,
    data_aug_train,
    y_symbol,
    specs;
    Ng = Ng, Nx = Nx, qoi_type = :pf, y_star = -1.427,
    max_iter = 25, direction = :min,
    tol_BO = 1e-3, tol_BC = 1e-2
)

cabo_max = @time "cabo max loop" cabo_loop(
    model_gfunction,        # physical_model — explicit, no longer a global lookup
    metamodel,
    cabo_min.data,
    y_symbol,
    specs;
    Ng = Ng, Nx = Nx, qoi_type = :pf, y_star = -1.427,
    max_iter = 25, direction = :max,
    tol_BO = 1e-3, tol_BC = 1e-2
)

# plot related
Θs_min = reduce(hcat, cabo_min.θ_history)'
Θs_max = reduce(hcat, cabo_max.θ_history)'
using Plots, Statistics

# ============================================================
# Grid over epistemic space (x1, θσ)
# ============================================================
n_μ1 = 300
n_μ2 = 300

μ1_grid = range(-1.5, 1.5, length=n_μ1)
μ2_grid  = range(-1.5, 1.5, length=n_μ2)

MeanSurface = zeros(n_μ1, n_μ2)

# ============================================================
# Compute response
# ============================================================
for (i, μ1_v) in enumerate(μ1_grid)
    for (j, μ2_v) in enumerate(μ2_grid)
        MeanSurface[j, i]  = g_function_E(μ1_v, μ2_v)
    end
end

# ============================================================
# Heatmap
# ============================================================
plt = heatmap(
    μ1_grid,
    μ2_grid,
    MeanSurface,
    xlabel="θ₁",
    ylabel="θ₂",
    c=:thermal,
    title="Expected conditioned response E[g | θ₁, θ₂] = Μ(θ₁, θ₂)",
    colorbar=true,
    xlims = (-1.6, 1.6),
    ylims = (-1.6, 1.6),
    legend = :outerbottom,
    legendcolumns=4,
)

# ── overlay the initial training points (μ1, μ2 columns from data_aug) ──────
data_aug_train_epi = Matrix{Float64}(undef, n_train, length(v_names))
for i in 1:n_train
    data_aug_train_epi[i, :] = augmented_to_epistemic(data_aug_train[i, v_names], specs)
end

scatter!(plt,
    data_aug_train_epi[:, 1], data_aug_train_epi[:, 2];
    marker = :diamond, color = :cyan, ms = 5,
    label  = "init samples", markerstrokewidth=0,
)

scatter!(plt,
    Θs_min[:, 1], Θs_min[:, 2];
    marker = :cross, color = :green, ms = 5,
    label  = "added min", markerstrokewidth=2,
)

scatter!(plt,
    Θs_max[:, 1], Θs_max[:, 2];
    marker = :cross, color = :red, ms = 5,
    label  = "added max", markerstrokewidth=2,
)

scatter!(plt,
    [cabo_min.θ_bound[1]], [cabo_min.θ_bound[2]];
    marker = :star, color = :green, ms = 7,
    label  = "cabo min", markerstrokewidth=1,
)

scatter!(plt,
    [cabo_max.θ_bound[1]], [cabo_max.θ_bound[2]];
    marker = :star, color = :red, ms = 7,
    label  = "cabo max", markerstrokewidth=1,
)

# ── mark true min and max in epistemic space ──────────────────────────────────
min_idx = argmin(MeanSurface)
max_idx = argmax(MeanSurface)

x_min, y_min, z_min = μ1_grid[min_idx[2]], μ1_grid[min_idx[1]], minimum(MeanSurface)
x_max, y_max, z_max = μ1_grid[max_idx[2]], μ1_grid[max_idx[1]], maximum(MeanSurface)

dy = 0.3

scatter!(plt,
    [μ1_grid[min_idx[2]]], [μ2_grid[min_idx[1]]];
    marker = :circle, color = :green, ms = 5, label = "True min",
)
annotate!(
    x_min, y_min + dy,
    text("($(round(x_min, digits=2)), $(round(y_min, digits=2)), $(round(z_min, digits=2)))", :black, 8)
)


scatter!(plt,
    [μ1_grid[max_idx[2]]], [μ2_grid[max_idx[1]]];
    marker = :rect, color = :red, ms = 5, label = "True max",
)

annotate!(
    x_max, y_max + dy,
    text("($(round(x_max, digits=2)), $(round(y_max, digits=2)), $(round(z_max, digits=2)))", :black, 8)
)

p1 = plot(
    cabo_min.L_BO_history,
    xlabel = "Iteration",
    ylabel = "L_BO",
    lw = 2, marker = :circle, ls = :dash,
    title = "MIN: L_BO History",
    legend = false
)

p2 = plot(
    cabo_min.L_BC_history,
    xlabel = "Iteration",
    ylabel = "L_BC",
    lw = 2, marker = :circle, ls = :dash,
    title = "MIN: L_BC History",
    # ylims = (0, 1),
    legend = false
)

p3 = plot(
    cabo_max.L_BO_history,
    xlabel = "Iteration",
    ylabel = "L_BO",
    lw = 2, marker = :circle, ls = :dash,
    title = "MAX: L_BO History",
    legend = false
)

p4 = plot(
    cabo_max.L_BC_history,
    xlabel = "Iteration",
    ylabel = "L_BC",
    lw = 2, marker = :circle, ls = :dash,
    title = "MAX: L_BC History",
    # ylims = (0, 1),
    legend = false
)

history = plot(
    p1, p2, p3, p4,
    layout = (2, 2),
    size = (900, 700)
)

# ---------------------------------------------------------
# Reusable plot helper (legend pushed OUTSIDE the plot area)
# ---------------------------------------------------------
function plot_convergence(cabo_min, cabo_max, n_train, true_bound, ylabel_str, title_str)
    p1 = plot(
        eachindex(cabo_min.bound_history) .+ n_train, cabo_min.bound_history;
        lw = 2, marker = :circle,
        label = "lower cabo bound",
        ylabel = ylabel_str * " min", title = title_str,
        legend = :topleft,
        size = (760, 440),
        left_margin = 5Plots.mm, right_margin = 5Plots.mm,
    )

    hline!(p1, [true_bound[1]]; ls = :dash, c = :gray, label = "final lower")
    
    p2 = plot(
        eachindex(cabo_max.bound_history) .+ length(cabo_min.bound_history) .+ n_train, cabo_max.bound_history;
        lw = 2, marker = :circle,
        label = "upper cabo bound",
        xlabel = "Total budget", ylabel = ylabel_str * " max",
        legend = :topleft,
        size = (760, 440),
        left_margin = 5Plots.mm, right_margin = 5Plots.mm,
    )

    hline!(p2, [true_bound[2]]; ls = :dash, c = :gray, label = "final upper")
        

    return p1, p2
end

p_mean_lower, p_mean_upper = plot_convergence(cabo_min, cabo_max, n_train, [0.0, 8.32 * 10^(-2)], "Pf bound", "Probability failure convergence")

plt_bound = plot(
    p_mean_lower, p_mean_upper,
    layout = (2, 1),
)




# ── Summary ───────────────────────────────────────────────────────────────────
# println("\n" * "="^60)
# println("CABO results")
# println("="^60)
# println("MIN  E[g] ≈ $(round(cabo_min.μ_bound, digits=3))" *
#         "  at θ = $(round.(cabo_min.θ_bound, digits=3))")
# println("MAX  E[g] ≈ $(round(cabo_max.μ_bound, digits=3))" *
#         "  at θ = $(round.(cabo_max.θ_bound, digits=3))")
# println("-"^60)
# println("Expected:  MIN ≈ $(round(z_min, digits=3)) at θ = [$(round(x_min, digits=3)), $(round(y_min, digits=3))]")
# println("           MAX ≈ $(round(z_max, digits=3)) at θ = [$(round(x_max, digits=3)), $(round(y_max, digits=3))]")
# println("-"^60)
# println("Difference:  MIN ≈ $(round(cabo_min.μ_bound - z_min, digits=3))")
# println("                     MAX ≈ $(round(cabo_max.μ_bound - z_max, digits=3))")
# println("="^60)


# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("CABO results")
println("="^60)
println("MIN  Pf ≈ $(round(cabo_min.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_min.θ_bound, digits=4))")
println("MAX  Pf ≈ $(round(cabo_max.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_max.θ_bound, digits=4))")
println("-"^60)
println("Expected:  MIN ≈ $(0.0) at θ = [$(-0.163), $(-0.963)]")
println("           MAX ≈ $(8.32* 10^(-2)) at θ = [$(-0.56), $(0.501)]")
println("-"^60)
println("Difference:  MIN ≈ $(round(cabo_min.μ_bound - 0, digits=4))")
println("                     MAX ≈ $(round(cabo_max.μ_bound - 8.32 * 10^(-2), digits=4))")
println("="^60)



display(plt)
display(history)
display(plt_bound)





# n_samples = 50, bound prop
# fit!: 6.958732 seconds (19.45 M allocations: 3.128 GiB, 9.80% gc time)
# predict:: 0.000740 seconds (158 allocations: 1.836 MiB)
# MSE: 0.04018
# Q²:  0.8657
# Params: Ng = 1000, Nx = 1000

# ━━━ CABO Iteration 1 / 30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.771, 0.71]    μ_qoi(θ*) ≈ -1.18e+00    σ_qoi(θ*) ≈ 8.44e-02
# bo objective : 303.821842 seconds (2.02 M allocations: 362.178 GiB, 38.26% gc time)
#     Acquisition θ⁺ = [-0.5853, 0.5442]    EI = 0.198    COV = 0.07172
# pvc objective : 0.671763 seconds (1.30 M allocations: 1.565 GiB, 42.02% gc time)

# ━━━ CABO Iteration 2 / 30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.585, 0.544]    μ_qoi(θ*) ≈ -1.39e+00    σ_qoi(θ*) ≈ 2.55e-02
# bo objective : 301.888050 seconds (2.02 M allocations: 362.634 GiB, 37.87% gc time)
#     Acquisition θ⁺ = [-0.5329, 0.5089]    EI = 0.0132    COV = 0.01837
# pvc objective : 0.411933 seconds (1.30 M allocations: 1.578 GiB, 11.07% gc time)

# ━━━ CABO Iteration 3 / 30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.585, 0.544]    μ_qoi(θ*) ≈ -1.38e+00    σ_qoi(θ*) ≈ 1.56e-02
# bo objective : 306.088287 seconds (2.02 M allocations: 363.081 GiB, 37.32% gc time)
#     Acquisition θ⁺ = [1.4891, -1.4535]    EI = 0.0005    COV = 0.0113
# pvc objective : 1.831973 seconds (1.30 M allocations: 1.585 GiB, 78.06% gc time)

# ━━━ CABO Iteration 4 / 30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.585, 0.544]    μ_qoi(θ*) ≈ -1.38e+00    σ_qoi(θ*) ≈ 1.37e-02
# bo objective : 303.141075 seconds (2.02 M allocations: 363.528 GiB, 38.26% gc time)
#     Acquisition θ⁺ = [-0.6315, 0.5713]    EI = 0.0092    COV = 0.009905
# pvc objective : 0.677578 seconds (1.30 M allocations: 1.593 GiB, 34.63% gc time)

# ━━━ CABO Iteration 5 / 30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.585, 0.544]    μ_qoi(θ*) ≈ -1.36e+00    σ_qoi(θ*) ≈ 9.91e-03
# bo objective : 301.103335 seconds (2.02 M allocations: 363.976 GiB, 38.34% gc time)
#     Acquisition θ⁺ = [-0.6364, 0.5345]    EI = 0.0051    COV = 0.007274
# pvc objective : 0.444500 seconds (1.30 M allocations: 1.604 GiB, 22.25% gc time)

# ━━━ CABO Iteration 6 / 30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.585, 0.544]    μ_qoi(θ*) ≈ -1.36e+00    σ_qoi(θ*) ≈ 9.61e-03
# bo objective : 299.514899 seconds (2.02 M allocations: 364.424 GiB, 38.34% gc time)
#     Acquisition θ⁺ = [-1.4256, 1.1141]    EI = 0.0004    COV = 0.007065
# pvc objective : 1.463921 seconds (1.30 M allocations: 1.612 GiB, 75.70% gc time)

# ✓ converged

#   ► MIN bound ≈ -1.3658  at  θ = [-0.5853, 0.5442]
# cabo min loop: 1850.593234 seconds (29.04 M allocations: 2.174 TiB, 38.16% gc time)

# ━━━ CABO Iteration 1 / 30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.463, 1.203]    μ_qoi(θ*) ≈ 9.23e-01    σ_qoi(θ*) ≈ 5.44e-02
# bo objective : 319.367625 seconds (2.02 M allocations: 367.118 GiB, 38.12% gc time)
#     Acquisition θ⁺ = [0.6387, 0.5291]    EI = 0.2486    COV = 0.05897
# pvc objective : 0.683747 seconds (1.30 M allocations: 1.673 GiB, 42.71% gc time)

# ━━━ CABO Iteration 2 / 30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.639, 0.529]    μ_qoi(θ*) ≈ 1.23e+00    σ_qoi(θ*) ≈ 1.70e-02
# bo objective : 310.261040 seconds (2.02 M allocations: 367.565 GiB, 38.64% gc time)
#     Acquisition θ⁺ = [0.5874, 0.8365]    EI = 0.0505    COV = 0.01383
# pvc objective : 0.501534 seconds (1.30 M allocations: 1.682 GiB, 14.08% gc time)

# ━━━ CABO Iteration 3 / 30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.587, 0.836]    μ_qoi(θ*) ≈ 1.36e+00    σ_qoi(θ*) ≈ 1.50e-02
# bo objective : 314.536870 seconds (2.07 M allocations: 368.015 GiB, 38.11% gc time)
#     Acquisition θ⁺ = [0.5618, 0.9322]    EI = 0.0119    COV = 0.011
# pvc objective : 0.795411 seconds (1.33 M allocations: 1.693 GiB, 46.58% gc time)

# ━━━ CABO Iteration 4 / 30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.587, 0.836]    μ_qoi(θ*) ≈ 1.37e+00    σ_qoi(θ*) ≈ 1.20e-02
# bo objective : 309.458134 seconds (2.07 M allocations: 368.462 GiB, 38.35% gc time)
#     Acquisition θ⁺ = [-1.4873, -1.4907]    EI = 0.0005    COV = 0.008771
# pvc objective : 0.770564 seconds (1.33 M allocations: 1.702 GiB, 43.66% gc time)

# ✓ converged

#   ► MAX bound ≈ 1.3835  at  θ = [0.5874, 0.8365]
# cabo max loop: 1281.629705 seconds (22.46 M allocations: 1.477 TiB, 38.28% gc time)

# ============================================================
# CABO results
# ============================================================
# MIN  E[g] ≈ -1.366  at θ = [-0.585, 0.544]
# MAX  E[g] ≈ 1.383  at θ = [0.587, 0.836]
# ------------------------------------------------------------
# Expected:  MIN ≈ -1.351 at θ = [-0.567, 0.527]
#            MAX ≈ 1.327 at θ = [0.557, 0.808]
# ------------------------------------------------------------
# Difference:  MIN ≈ -0.015
#                      MAX ≈ 0.056
# ============================================================


# ystar = -1.427 
# fit!: 4.624304 seconds (18.31 M allocations: 1.576 GiB, 5.10% gc time, 41.77% compilation time: <1% of which was recompilation)
# predict:: 0.000580 seconds (158 allocations: 1.836 MiB)
# MSE: 0.03204
# Q²:  0.89301
# Params: Ng = 1000, Nx = 2000

# ━━━ CABO Iteration 1 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 2.724635 seconds (9.35 k allocations: 3.629 GiB, 44.27% gc time)
# u objective : 0.000974 seconds (25.07 k allocations: 2.269 MiB)

#     Incumbent θ* = [0.031, -0.031]    μ_qoi(θ*) ≈ 0.00e+00    σ_qoi(θ*) ≈ 0.00e+00
#     Acquisition θ⁺ = [-1.3211, 1.489]    EI = 0.0    COV = 0.0
#     Current estimated bound: 0.0 @ θ = [0.031, -0.031]


# ✓ converged

#   ► MIN bound ≈ 0.0  at  θ = [0.0309, -0.0309]
# cabo min loop: 13.043036 seconds (1.50 M allocations: 15.150 GiB, 47.60% gc time, 0.39% compilation time: <1% of which was recompilation)

# ━━━ CABO Iteration 1 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 534.370529 seconds (2.08 M allocations: 726.764 GiB, 37.20% gc time)
# u objective : 0.002047 seconds (30.15 k allocations: 2.823 MiB)

#     Incumbent θ* = [-1.018, 0.586]    μ_qoi(θ*) ≈ 1.42e-02    σ_qoi(θ*) ≈ 5.77e-02
#     Acquisition θ⁺ = [-0.7898, 0.5978]    EI = 0.1329    COV = 4.071
#     Current estimated bound: 0.0 @ θ = [-0.79, 0.598]


# ━━━ CABO Iteration 2 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 539.926393 seconds (2.08 M allocations: 727.658 GiB, 37.33% gc time)
# u objective : 0.001000 seconds (24.90 k allocations: 2.345 MiB)

#     Incumbent θ* = [-0.79, 0.598]    μ_qoi(θ*) ≈ 1.46e-01    σ_qoi(θ*) ≈ 2.53e-01
#     Acquisition θ⁺ = [-0.7953, 0.607]    EI = 0.1003    COV = 1.734
#     Current estimated bound: 0.0 @ θ = [-0.79, 0.598]


# ━━━ CABO Iteration 3 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 538.882021 seconds (2.08 M allocations: 728.552 GiB, 37.48% gc time)
# u objective : 0.001692 seconds (45.90 k allocations: 4.254 MiB)

#     Incumbent θ* = [-0.79, 0.598]    μ_qoi(θ*) ≈ 1.67e-01    σ_qoi(θ*) ≈ 2.73e-01
#     Acquisition θ⁺ = [-0.7824, 0.5894]    EI = 0.1094    COV = 1.634
#     Current estimated bound: 0.0 @ θ = [-0.782, 0.589]


# ━━━ CABO Iteration 4 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 513.024582 seconds (2.08 M allocations: 729.448 GiB, 37.73% gc time)
# u objective : 0.001038 seconds (30.15 k allocations: 2.905 MiB)

#     Incumbent θ* = [-0.79, 0.598]    μ_qoi(θ*) ≈ 1.16e-01    σ_qoi(θ*) ≈ 2.31e-01
#     Acquisition θ⁺ = [-0.7973, 0.5979]    EI = 0.085    COV = 1.997
#     Current estimated bound: 0.0 @ θ = [-0.797, 0.598]


# ━━━ CABO Iteration 5 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 523.301621 seconds (2.08 M allocations: 730.342 GiB, 37.47% gc time)
# u objective : 0.047819 seconds (1.05 M allocations: 98.147 MiB, 21.77% gc time)

#     Incumbent θ* = [-0.797, 0.598]    μ_qoi(θ*) ≈ 1.30e-01    σ_qoi(θ*) ≈ 2.53e-01
#     Acquisition θ⁺ = [-0.7999, 0.5828]    EI = 0.0975    COV = 1.941
#     Current estimated bound: 0.0 @ θ = [-0.782, 0.589]


# ━━━ CABO Iteration 6 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 530.760203 seconds (2.08 M allocations: 731.247 GiB, 37.41% gc time)
# u objective : 0.077073 seconds (1.05 M allocations: 102.419 MiB)

#     Incumbent θ* = [-0.782, 0.589]    μ_qoi(θ*) ≈ 6.89e-02    σ_qoi(θ*) ≈ 1.91e-01
#     Acquisition θ⁺ = [-0.3752, 0.6753]    EI = 0.0627    COV = 2.771
#     Current estimated bound: 0.0 @ θ = [-0.375, 0.675]


# ━━━ CABO Iteration 7 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 519.894474 seconds (2.08 M allocations: 732.141 GiB, 37.57% gc time)
# u objective : 0.001090 seconds (30.15 k allocations: 3.033 MiB)

#     Incumbent θ* = [-0.782, 0.589]    μ_qoi(θ*) ≈ 1.71e-01    σ_qoi(θ*) ≈ 2.90e-01
#     Acquisition θ⁺ = [-0.5601, 0.5849]    EI = 0.1889    COV = 1.693
#     Current estimated bound: 0.0 @ θ = [-0.56, 0.585]


# ━━━ CABO Iteration 8 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 540.448834 seconds (2.08 M allocations: 733.035 GiB, 37.19% gc time)
# u objective : 0.001302 seconds (35.40 k allocations: 3.546 MiB)

#     Incumbent θ* = [-0.56, 0.585]    μ_qoi(θ*) ≈ 3.19e-01    σ_qoi(θ*) ≈ 3.69e-01
#     Acquisition θ⁺ = [-0.549, 0.5868]    EI = 0.1731    COV = 1.155
#     Current estimated bound: 0.001 @ θ = [-0.549, 0.587]


# ━━━ CABO Iteration 9 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 547.264732 seconds (2.08 M allocations: 733.929 GiB, 37.20% gc time)
# u objective : 0.078857 seconds (1.05 M allocations: 102.419 MiB)

#     Incumbent θ* = [-0.56, 0.585]    μ_qoi(θ*) ≈ 3.75e-01    σ_qoi(θ*) ≈ 3.78e-01
#     Acquisition θ⁺ = [-0.5483, 0.587]    EI = 0.1782    COV = 1.01
#     Current estimated bound: 0.0 @ θ = [-0.549, 0.587]


# ━━━ CABO Iteration 10 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 538.802878 seconds (2.08 M allocations: 734.825 GiB, 37.44% gc time)
# u objective : 0.038387 seconds (1.05 M allocations: 105.166 MiB)

#     Incumbent θ* = [-0.548, 0.587]    μ_qoi(θ*) ≈ 2.50e-01    σ_qoi(θ*) ≈ 3.06e-01
#     Acquisition θ⁺ = [-0.5218, 0.5912]    EI = 0.1414    COV = 1.225
#     Current estimated bound: 0.0 @ θ = [-0.375, 0.675]


# ━━━ CABO Iteration 11 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 529.460789 seconds (2.08 M allocations: 735.719 GiB, 37.79% gc time)
# u objective : 0.002255 seconds (30.15 k allocations: 3.116 MiB)

#     Incumbent θ* = [-0.375, 0.675]    μ_qoi(θ*) ≈ 9.80e-02    σ_qoi(θ*) ≈ 1.42e-01
#     Acquisition θ⁺ = [-0.3251, 0.6986]    EI = 0.0694    COV = 1.449
#     Current estimated bound: 0.0 @ θ = [-0.325, 0.699]


# ━━━ CABO Iteration 12 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 532.652841 seconds (2.08 M allocations: 736.613 GiB, 37.82% gc time)
# u objective : 0.001880 seconds (24.90 k allocations: 2.590 MiB)

#     Incumbent θ* = [-0.325, 0.699]    μ_qoi(θ*) ≈ 9.86e-02    σ_qoi(θ*) ≈ 1.29e-01
#     Acquisition θ⁺ = [-0.3333, 0.7025]    EI = 0.0552    COV = 1.308
#     Current estimated bound: 0.089 @ θ = [-0.325, 0.699]


# ━━━ CABO Iteration 13 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 514.595166 seconds (2.13 M allocations: 737.510 GiB, 37.73% gc time)
# u objective : 0.001457 seconds (36.10 k allocations: 3.679 MiB)

#     Incumbent θ* = [-0.333, 0.703]    μ_qoi(θ*) ≈ 1.21e-01    σ_qoi(θ*) ≈ 1.24e-01
#     Acquisition θ⁺ = [1.0906, -1.4801]    EI = 0.0025    COV = 1.026
#     Current estimated bound: 0.089 @ θ = [-0.325, 0.699]


# ━━━ CABO Iteration 14 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 550.251953 seconds (2.13 M allocations: 738.404 GiB, 37.81% gc time)
# u objective : 0.001703 seconds (30.75 k allocations: 3.148 MiB)

#     Incumbent θ* = [-0.333, 0.703]    μ_qoi(θ*) ≈ 1.20e-01    σ_qoi(θ*) ≈ 1.29e-01
#     Acquisition θ⁺ = [-0.3317, 0.7041]    EI = 0.0558    COV = 1.078
#     Current estimated bound: 0.0 @ θ = [-0.332, 0.704]


# ━━━ CABO Iteration 15 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 543.536698 seconds (2.13 M allocations: 739.301 GiB, 37.88% gc time)
# u objective : 0.077245 seconds (1.07 M allocations: 107.454 MiB)

#     Incumbent θ* = [-0.325, 0.699]    μ_qoi(θ*) ≈ 8.10e-02    σ_qoi(θ*) ≈ 1.19e-01
#     Acquisition θ⁺ = [-0.3097, 0.7097]    EI = 0.0493    COV = 1.465
#     Current estimated bound: 0.158 @ θ = [-0.31, 0.71]


# ━━━ CABO Iteration 16 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 544.238251 seconds (2.13 M allocations: 740.196 GiB, 37.51% gc time)
# u objective : 0.044353 seconds (1.07 M allocations: 109.285 MiB, 14.85% gc time)

#     Incumbent θ* = [-0.31, 0.71]    μ_qoi(θ*) ≈ 1.95e-01    σ_qoi(θ*) ≈ 1.98e-01
#     Acquisition θ⁺ = [-0.2891, 0.7082]    EI = 0.0881    COV = 1.016
#     Current estimated bound: 0.014 @ θ = [-0.289, 0.708]


# ━━━ CABO Iteration 17 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 2.948339 seconds (9.30 k allocations: 3.705 GiB, 41.96% gc time)
# u objective : 0.036486 seconds (1.07 M allocations: 110.506 MiB)

#     Incumbent θ* = [-0.289, 0.708]    μ_qoi(θ*) ≈ 6.11e-02    σ_qoi(θ*) ≈ 7.88e-02
#     Acquisition θ⁺ = [0.9318, 0.1812]    EI = 0.0    COV = 1.291
#     Current estimated bound: 0.016 @ θ = [-0.289, 0.708]


# ━━━ CABO Iteration 18 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 538.266758 seconds (2.13 M allocations: 741.987 GiB, 37.99% gc time)
# u objective : 0.036771 seconds (1.07 M allocations: 110.506 MiB)

#     Incumbent θ* = [-0.289, 0.708]    μ_qoi(θ*) ≈ 5.67e-02    σ_qoi(θ*) ≈ 7.79e-02
#     Acquisition θ⁺ = [-0.2811, 0.7187]    EI = 0.0314    COV = 1.374
#     Current estimated bound: 0.0 @ θ = [-0.281, 0.719]


# ━━━ CABO Iteration 19 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 534.073579 seconds (2.13 M allocations: 742.884 GiB, 37.97% gc time)
# u objective : 0.001103 seconds (25.40 k allocations: 2.754 MiB)

#     Incumbent θ* = [-1.018, 0.586]    μ_qoi(θ*) ≈ 5.90e-03    σ_qoi(θ*) ≈ 4.41e-02
#     Acquisition θ⁺ = [-0.5773, 0.4008]    EI = 0.0626    COV = 7.468
#     Current estimated bound: 0.066 @ θ = [-0.577, 0.401]


# ━━━ CABO Iteration 20 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 535.138253 seconds (2.13 M allocations: 743.780 GiB, 38.11% gc time)
# u objective : 0.001318 seconds (30.75 k allocations: 3.395 MiB)

#     Incumbent θ* = [-0.577, 0.401]    μ_qoi(θ*) ≈ 7.33e-02    σ_qoi(θ*) ≈ 3.27e-02
#     Acquisition θ⁺ = [-0.8959, 0.4668]    EI = 0.029    COV = 0.4466
#     Current estimated bound: 0.071 @ θ = [-0.577, 0.401]


# ━━━ CABO Iteration 21 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 536.692922 seconds (2.13 M allocations: 744.677 GiB, 37.93% gc time)
# u objective : 0.001765 seconds (20.05 k allocations: 2.274 MiB)

#     Incumbent θ* = [-0.896, 0.467]    μ_qoi(θ*) ≈ 4.30e-02    σ_qoi(θ*) ≈ 9.71e-02
#     Acquisition θ⁺ = [-0.7335, 0.3973]    EI = 0.0448    COV = 2.257
#     Current estimated bound: 0.066 @ θ = [-0.734, 0.397]


# ━━━ CABO Iteration 22 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 530.709660 seconds (2.13 M allocations: 745.571 GiB, 37.74% gc time)
# u objective : 0.038617 seconds (1.07 M allocations: 115.694 MiB)

#     Incumbent θ* = [-0.896, 0.467]    μ_qoi(θ*) ≈ 6.24e-02    σ_qoi(θ*) ≈ 1.23e-01
#     Acquisition θ⁺ = [-0.873, 0.4552]    EI = 0.0436    COV = 1.971
#     Current estimated bound: 0.071 @ θ = [-0.734, 0.397]


# ━━━ CABO Iteration 23 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 540.266177 seconds (2.13 M allocations: 746.468 GiB, 37.93% gc time)
# u objective : 0.039926 seconds (1.07 M allocations: 116.915 MiB)

#     Incumbent θ* = [-0.896, 0.467]    μ_qoi(θ*) ≈ 6.84e-02    σ_qoi(θ*) ≈ 1.15e-01
#     Acquisition θ⁺ = [-0.8845, 0.4537]    EI = 0.0441    COV = 1.682
#     Current estimated bound: 0.106 @ θ = [-0.734, 0.397]


# ━━━ CABO Iteration 24 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 539.876698 seconds (2.13 M allocations: 747.362 GiB, 38.09% gc time)
# u objective : 0.040904 seconds (1.07 M allocations: 116.915 MiB)

#     Incumbent θ* = [-0.734, 0.397]    μ_qoi(θ*) ≈ 1.16e-01    σ_qoi(θ*) ≈ 3.90e-02
#     Acquisition θ⁺ = [-0.7714, 0.3992]    EI = 0.0161    COV = 0.3361
#     Current estimated bound: 0.097 @ θ = [-0.771, 0.399]


# ━━━ CABO Iteration 25 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 550.308824 seconds (2.13 M allocations: 748.259 GiB, 37.79% gc time)
# u objective : 0.059866 seconds (1.07 M allocations: 118.136 MiB, 33.44% gc time)

#     Incumbent θ* = [-0.771, 0.399]    μ_qoi(θ*) ≈ 1.11e-01    σ_qoi(θ*) ≈ 5.54e-02
#     Acquisition θ⁺ = [-0.7514, 0.4002]    EI = 0.0225    COV = 0.4998
#     Current estimated bound: 0.095 @ θ = [-0.751, 0.4]


#   ► MAX bound ≈ 0.0975  at  θ = [-0.7514, 0.4002]
# cabo max loop: 13035.858515 seconds (82.03 M allocations: 17.535 TiB, 37.67% gc time)

# ============================================================
# CABO results
# ============================================================
# MIN  Pf ≈ 0.0  at θ = [0.0309, -0.0309]
# MAX  Pf ≈ 0.0975  at θ = [-0.7514, 0.4002]
# ------------------------------------------------------------
# Expected:  MIN ≈ 0.0 at θ = [-0.163, -0.963]
#            MAX ≈ 0.08320000000000002 at θ = [-0.56, 0.501]
# ------------------------------------------------------------
# Difference:  MIN ≈ 0.0
#                      MAX ≈ 0.0143
# ============================================================