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

Ng = 500
Nx = 500

print("Params: Ng = $Ng, Nx = $Nx\n")

cabo_min = @time "cabo min loop" cabo_loop(
    model_gfunction,        # physical_model — explicit, no longer a global lookup
    metamodel,
    data_aug_train,
    y_symbol,
    specs;
    Ng = Ng, Nx = Nx, qoi_type = :mean,
    max_iter = 25, direction = :min,
    tol_BO = 1e-3, tol_BC = 1e-2
)

cabo_max = @time "cabo max loop" cabo_loop(
    model_gfunction,        # physical_model — explicit, no longer a global lookup
    metamodel,
    cabo_min.data,
    y_symbol,
    specs;
    Ng = Ng, Nx = Nx, qoi_type = :mean,
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
    ylims = (0, 1),
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
    ylims = (0, 1),
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

p_mean_lower, p_mean_upper = plot_convergence(cabo_min, cabo_max, n_train, [-1.351, 1.327], "μ bound", "Expected bound convergence")

plt_bound = plot(
    p_mean_lower, p_mean_upper,
    layout = (2, 1),
)




# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("CABO results")
println("="^60)
println("MIN  E[g] ≈ $(round(cabo_min.μ_bound, digits=3))" *
        "  at θ = $(round.(cabo_min.θ_bound, digits=3))")
println("MAX  E[g] ≈ $(round(cabo_max.μ_bound, digits=3))" *
        "  at θ = $(round.(cabo_max.θ_bound, digits=3))")
println("-"^60)
println("Expected:  MIN ≈ $(round(z_min, digits=3)) at θ = [$(round(x_min, digits=3)), $(round(y_min, digits=3))]")
println("           MAX ≈ $(round(z_max, digits=3)) at θ = [$(round(x_max, digits=3)), $(round(y_max, digits=3))]")
println("-"^60)
println("Difference:  MIN ≈ $(round(cabo_min.μ_bound - z_min, digits=3))")
println("                     MAX ≈ $(round(cabo_max.μ_bound - z_max, digits=3))")
println("="^60)




display(plt)
display(history)
display(plt_bound)





# fit!: 5.634253 seconds (18.55 M allocations: 1.902 GiB, 7.10% gc time)
# predict:: 0.000610 seconds (158 allocations: 1.836 MiB)
# MSE: 0.05552
# Q²:  0.81338
# Params: Ng = 500, Nx = 500

# ━━━ CABO Iteration 1 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 338.589306 seconds (4.03 M allocations: 475.661 GiB, 34.53% gc time)
# pvc objective : 0.375252 seconds (2.57 M allocations: 1.866 GiB, 13.74% gc time)

#     Incumbent θ* = [-0.833, 0.71]    μ_qoi(θ*) ≈ -9.21e-01    σ_qoi(θ*) ≈ 6.25e-02
#     Acquisition θ⁺ = [-0.9549, 0.5382]    EI = 0.0889    COV = 0.06789
#     Current estimated bound: -1.113 @ θ = [-0.955, 0.538]


# ━━━ CABO Iteration 2 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 329.031439 seconds (4.03 M allocations: 476.128 GiB, 35.00% gc time)
# pvc objective : 0.560710 seconds (2.57 M allocations: 1.891 GiB, 35.30% gc time)

#     Incumbent θ* = [-0.955, 0.538]    μ_qoi(θ*) ≈ -1.12e+00    σ_qoi(θ*) ≈ 2.43e-02
#     Acquisition θ⁺ = [1.4937, -1.5]    EI = 0.0018    COV = 0.02175
#     Current estimated bound: -1.113 @ θ = [-0.955, 0.538]


# ━━━ CABO Iteration 3 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 342.803179 seconds (4.03 M allocations: 476.571 GiB, 33.86% gc time)
# pvc objective : 0.393187 seconds (2.57 M allocations: 1.906 GiB, 16.23% gc time)

#     Incumbent θ* = [-0.955, 0.538]    μ_qoi(θ*) ≈ -1.12e+00    σ_qoi(θ*) ≈ 2.48e-02
#     Acquisition θ⁺ = [-0.8959, 0.49]    EI = 0.0144    COV = 0.02221
#     Current estimated bound: -1.13 @ θ = [-0.896, 0.49]


# ━━━ CABO Iteration 4 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 331.253604 seconds (4.03 M allocations: 477.022 GiB, 34.71% gc time)
# pvc objective : 0.640135 seconds (2.57 M allocations: 1.922 GiB, 40.28% gc time)

#     Incumbent θ* = [-0.955, 0.538]    μ_qoi(θ*) ≈ -1.13e+00    σ_qoi(θ*) ≈ 1.87e-02
#     Acquisition θ⁺ = [-0.8922, 0.5035]    EI = 0.0162    COV = 0.01657
#     Current estimated bound: -1.179 @ θ = [-0.892, 0.503]


# ━━━ CABO Iteration 5 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 340.419792 seconds (4.03 M allocations: 477.469 GiB, 34.37% gc time)
# pvc objective : 0.619835 seconds (2.57 M allocations: 1.943 GiB, 39.73% gc time)

#     Incumbent θ* = [-0.892, 0.503]    μ_qoi(θ*) ≈ -1.19e+00    σ_qoi(θ*) ≈ 1.75e-02
#     Acquisition θ⁺ = [-0.7834, 0.4875]    EI = 0.0271    COV = 0.01469
#     Current estimated bound: -1.211 @ θ = [-0.783, 0.488]


# ━━━ CABO Iteration 6 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 340.880586 seconds (4.03 M allocations: 477.920 GiB, 34.72% gc time)
# pvc objective : 0.474840 seconds (2.57 M allocations: 1.960 GiB, 25.17% gc time)

#     Incumbent θ* = [-0.783, 0.488]    μ_qoi(θ*) ≈ -1.22e+00    σ_qoi(θ*) ≈ 2.52e-02
#     Acquisition θ⁺ = [-0.774, 0.4993]    EI = 0.0104    COV = 0.02068
#     Current estimated bound: -1.261 @ θ = [-0.774, 0.499]


# ━━━ CABO Iteration 7 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 336.175949 seconds (4.03 M allocations: 478.384 GiB, 34.68% gc time)
# pvc objective : 0.619997 seconds (2.57 M allocations: 1.988 GiB, 39.77% gc time)

#     Incumbent θ* = [-0.774, 0.499]    μ_qoi(θ*) ≈ -1.27e+00    σ_qoi(θ*) ≈ 1.92e-02
#     Acquisition θ⁺ = [-0.6861, 0.4973]    EI = 0.0165    COV = 0.01513
#     Current estimated bound: -1.291 @ θ = [-0.686, 0.497]


# ━━━ CABO Iteration 8 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 333.757778 seconds (4.03 M allocations: 478.834 GiB, 34.96% gc time)
# pvc objective : 0.600107 seconds (2.57 M allocations: 2.006 GiB, 37.46% gc time)

#     Incumbent θ* = [-0.686, 0.497]    μ_qoi(θ*) ≈ -1.30e+00    σ_qoi(θ*) ≈ 2.50e-02
#     Acquisition θ⁺ = [-0.6326, 0.5151]    EI = 0.0127    COV = 0.01921
#     Current estimated bound: -1.341 @ θ = [-0.633, 0.515]


# ━━━ CABO Iteration 9 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 341.134149 seconds (4.03 M allocations: 479.278 GiB, 34.49% gc time)
# pvc objective : 0.525783 seconds (2.57 M allocations: 2.022 GiB, 21.63% gc time)

#     Incumbent θ* = [-0.633, 0.515]    μ_qoi(θ*) ≈ -1.34e+00    σ_qoi(θ*) ≈ 2.43e-02
#     Acquisition θ⁺ = [-0.5888, 0.5389]    EI = 0.0129    COV = 0.01807
#     Current estimated bound: -1.348 @ θ = [-0.633, 0.515]


# ━━━ CABO Iteration 10 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 335.575197 seconds (4.03 M allocations: 479.729 GiB, 35.22% gc time)
# pvc objective : 0.667487 seconds (2.57 M allocations: 2.040 GiB, 41.54% gc time)

#     Incumbent θ* = [-0.633, 0.515]    μ_qoi(θ*) ≈ -1.35e+00    σ_qoi(θ*) ≈ 2.04e-02
#     Acquisition θ⁺ = [-0.5941, 0.5213]    EI = 0.01    COV = 0.01513
#     Current estimated bound: -1.352 @ θ = [-0.633, 0.515]


# ━━━ CABO Iteration 11 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 338.979402 seconds (4.03 M allocations: 480.176 GiB, 35.10% gc time)
# pvc objective : 0.433114 seconds (2.57 M allocations: 2.062 GiB, 18.98% gc time)

#     Incumbent θ* = [-0.633, 0.515]    μ_qoi(θ*) ≈ -1.35e+00    σ_qoi(θ*) ≈ 1.15e-02
#     Acquisition θ⁺ = [-0.5975, 0.4611]    EI = 0.0108    COV = 0.008493
#     Current estimated bound: -1.353 @ θ = [-0.633, 0.515]


# ━━━ CABO Iteration 12 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 345.502615 seconds (4.03 M allocations: 480.626 GiB, 34.36% gc time)
# pvc objective : 0.671496 seconds (2.57 M allocations: 2.081 GiB, 41.49% gc time)

#     Incumbent θ* = [-0.633, 0.515]    μ_qoi(θ*) ≈ -1.36e+00    σ_qoi(θ*) ≈ 1.07e-02
#     Acquisition θ⁺ = [-0.6175, 0.4846]    EI = 0.008    COV = 0.007862
#     Current estimated bound: -1.357 @ θ = [-0.617, 0.485]


# ━━━ CABO Iteration 13 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 337.704500 seconds (4.03 M allocations: 481.070 GiB, 34.82% gc time)
# pvc objective : 0.673719 seconds (2.57 M allocations: 2.099 GiB, 41.06% gc time)

#     Incumbent θ* = [-0.617, 0.485]    μ_qoi(θ*) ≈ -1.36e+00    σ_qoi(θ*) ≈ 8.43e-03
#     Acquisition θ⁺ = [-1.5, 1.2259]    EI = 0.0007    COV = 0.006193
#     Current estimated bound: -1.358 @ θ = [-0.617, 0.485]


# ✓ converged

#   ► MIN bound ≈ -1.3575  at  θ = [-0.6175, 0.4846]
# cabo min loop: 4432.356464 seconds (96.56 M allocations: 6.142 TiB, 34.67% gc time)

# ━━━ CABO Iteration 1 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 343.200769 seconds (4.13 M allocations: 481.525 GiB, 34.71% gc time)
# pvc objective : 0.548303 seconds (2.63 M allocations: 2.121 GiB, 28.95% gc time)

#     Incumbent θ* = [0.71, 0.957]    μ_qoi(θ*) ≈ 1.09e+00    σ_qoi(θ*) ≈ 2.44e-02
#     Acquisition θ⁺ = [0.6421, 1.1708]    EI = 0.0327    COV = 0.02248
#     Current estimated bound: 1.091 @ θ = [0.71, 0.957]


# ━━━ CABO Iteration 2 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 340.811154 seconds (4.13 M allocations: 481.969 GiB, 34.76% gc time)
# pvc objective : 0.627053 seconds (2.63 M allocations: 2.139 GiB, 36.02% gc time)

#     Incumbent θ* = [0.71, 0.957]    μ_qoi(θ*) ≈ 1.09e+00    σ_qoi(θ*) ≈ 2.28e-02
#     Acquisition θ⁺ = [0.7041, 0.7283]    EI = 0.0237    COV = 0.02089
#     Current estimated bound: 1.21 @ θ = [0.704, 0.728]


# ━━━ CABO Iteration 3 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 349.187078 seconds (4.13 M allocations: 482.425 GiB, 34.76% gc time)
# pvc objective : 0.602421 seconds (2.63 M allocations: 2.163 GiB, 33.26% gc time)

#     Incumbent θ* = [0.704, 0.728]    μ_qoi(θ*) ≈ 1.21e+00    σ_qoi(θ*) ≈ 1.82e-02
#     Acquisition θ⁺ = [0.6761, 0.6188]    EI = 0.0145    COV = 0.0151
#     Current estimated bound: 1.211 @ θ = [0.704, 0.728]


# ━━━ CABO Iteration 4 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 339.168630 seconds (4.13 M allocations: 482.871 GiB, 35.30% gc time)
# pvc objective : 0.618141 seconds (2.63 M allocations: 2.186 GiB, 33.79% gc time)

#     Incumbent θ* = [0.704, 0.728]    μ_qoi(θ*) ≈ 1.21e+00    σ_qoi(θ*) ≈ 1.78e-02
#     Acquisition θ⁺ = [0.4128, 1.4923]    EI = 0.0028    COV = 0.01471
#     Current estimated bound: 1.214 @ θ = [0.704, 0.728]


# ━━━ CABO Iteration 5 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 339.263774 seconds (4.13 M allocations: 483.328 GiB, 34.87% gc time)
# pvc objective : 0.540637 seconds (2.63 M allocations: 2.210 GiB, 26.42% gc time)

#     Incumbent θ* = [0.704, 0.728]    μ_qoi(θ*) ≈ 1.21e+00    σ_qoi(θ*) ≈ 1.86e-02
#     Acquisition θ⁺ = [0.6512, 0.7068]    EI = 0.0125    COV = 0.01535
#     Current estimated bound: 1.272 @ θ = [0.651, 0.707]


# ━━━ CABO Iteration 6 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 340.661714 seconds (4.13 M allocations: 483.771 GiB, 35.05% gc time)
# pvc objective : 0.707624 seconds (2.63 M allocations: 2.229 GiB, 40.51% gc time)

#     Incumbent θ* = [0.651, 0.707]    μ_qoi(θ*) ≈ 1.27e+00    σ_qoi(θ*) ≈ 2.09e-02
#     Acquisition θ⁺ = [0.6003, 0.7384]    EI = 0.0212    COV = 0.0164
#     Current estimated bound: 1.322 @ θ = [0.6, 0.738]


# ━━━ CABO Iteration 7 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 351.007546 seconds (4.13 M allocations: 484.228 GiB, 34.55% gc time)
# pvc objective : 0.528929 seconds (2.63 M allocations: 2.254 GiB, 15.05% gc time)

#     Incumbent θ* = [0.6, 0.738]    μ_qoi(θ*) ≈ 1.32e+00    σ_qoi(θ*) ≈ 1.67e-02
#     Acquisition θ⁺ = [-0.0783, 1.4258]    EI = 0.0021    COV = 0.01263
#     Current estimated bound: 1.324 @ θ = [0.6, 0.738]


# ━━━ CABO Iteration 8 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 344.047529 seconds (4.13 M allocations: 484.675 GiB, 34.97% gc time)
# pvc objective : 0.589029 seconds (2.63 M allocations: 2.279 GiB, 31.02% gc time)

#     Incumbent θ* = [0.6, 0.738]    μ_qoi(θ*) ≈ 1.32e+00    σ_qoi(θ*) ≈ 1.95e-02
#     Acquisition θ⁺ = [-1.442, -1.2605]    EI = 0.001    COV = 0.01471
#     Current estimated bound: 1.321 @ θ = [0.6, 0.738]


# ━━━ CABO Iteration 9 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 347.690637 seconds (4.13 M allocations: 485.132 GiB, 34.82% gc time)
# pvc objective : 0.701702 seconds (2.63 M allocations: 2.304 GiB, 40.79% gc time)

#     Incumbent θ* = [0.6, 0.738]    μ_qoi(θ*) ≈ 1.32e+00    σ_qoi(θ*) ≈ 1.83e-02
#     Acquisition θ⁺ = [0.5869, 0.7638]    EI = 0.0084    COV = 0.01386
#     Current estimated bound: 1.343 @ θ = [0.587, 0.764]


# ━━━ CABO Iteration 10 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 347.521490 seconds (4.13 M allocations: 485.575 GiB, 34.68% gc time)
# pvc objective : 0.662906 seconds (2.63 M allocations: 2.325 GiB, 36.25% gc time)

#     Incumbent θ* = [0.587, 0.764]    μ_qoi(θ*) ≈ 1.34e+00    σ_qoi(θ*) ≈ 1.67e-02
#     Acquisition θ⁺ = [-0.2195, -1.4974]    EI = 0.001    COV = 0.01248
#     Current estimated bound: 1.342 @ θ = [0.587, 0.764]


# ━━━ CABO Iteration 11 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 350.680636 seconds (4.13 M allocations: 486.032 GiB, 34.69% gc time)
# pvc objective : 0.625308 seconds (2.63 M allocations: 2.351 GiB, 33.13% gc time)

#     Incumbent θ* = [0.587, 0.764]    μ_qoi(θ*) ≈ 1.34e+00    σ_qoi(θ*) ≈ 1.77e-02
#     Acquisition θ⁺ = [0.5763, 0.7731]    EI = 0.0074    COV = 0.01319
#     Current estimated bound: 1.343 @ θ = [0.587, 0.764]


# ━━━ CABO Iteration 12 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 347.567434 seconds (4.13 M allocations: 486.475 GiB, 34.44% gc time)
# pvc objective : 0.604653 seconds (2.63 M allocations: 2.372 GiB, 30.13% gc time)

#     Incumbent θ* = [0.6, 0.738]    μ_qoi(θ*) ≈ 1.34e+00    σ_qoi(θ*) ≈ 9.20e-03
#     Acquisition θ⁺ = [0.587, 0.7339]    EI = 0.0044    COV = 0.006865
#     Current estimated bound: 1.331 @ θ = [0.576, 0.773]


# ━━━ CABO Iteration 13 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 350.010240 seconds (4.13 M allocations: 486.932 GiB, 34.64% gc time)
# pvc objective : 0.629993 seconds (2.63 M allocations: 2.399 GiB, 32.82% gc time)

#     Incumbent θ* = [0.576, 0.773]    μ_qoi(θ*) ≈ 1.33e+00    σ_qoi(θ*) ≈ 5.26e-03
#     Acquisition θ⁺ = [0.5789, 0.818]    EI = 0.0049    COV = 0.003966
#     Current estimated bound: 1.334 @ θ = [0.576, 0.773]


# ━━━ CABO Iteration 14 / 25 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# bo objective : 359.078069 seconds (4.13 M allocations: 487.379 GiB, 34.06% gc time)
# pvc objective : 0.730913 seconds (2.63 M allocations: 2.425 GiB, 39.92% gc time)

#     Incumbent θ* = [0.576, 0.773]    μ_qoi(θ*) ≈ 1.33e+00    σ_qoi(θ*) ≈ 4.46e-03
#     Acquisition θ⁺ = [1.5, 1.2726]    EI = 0.0006    COV = 0.003357
#     Current estimated bound: 1.332 @ θ = [0.576, 0.773]


# ✓ converged

#   ► MAX bound ≈ 1.3328  at  θ = [0.5763, 0.7731]
# cabo max loop: 4906.913086 seconds (106.18 M allocations: 6.714 TiB, 34.74% gc time)

# ============================================================
# CABO results
# ============================================================
# MIN  E[g] ≈ -1.358  at θ = [-0.617, 0.485]
# MAX  E[g] ≈ 1.333  at θ = [0.576, 0.773]
# ------------------------------------------------------------
# Expected:  MIN ≈ -1.351 at θ = [-0.567, 0.527]
#            MAX ≈ 1.327 at θ = [0.557, 0.808]
# ------------------------------------------------------------
# Difference:  MIN ≈ -0.007
#                      MAX ≈ 0.005
# ============================================================