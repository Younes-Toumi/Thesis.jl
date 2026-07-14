using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Printf

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

# ==============================================================================
# 10. Usage — mirrors your UQ.jl example end to end
# ==============================================================================

# parametric inputs
x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input
 
physical_model = model_gfunction


x_names, w_names, u_names, v_names = spec_names(specs)
y_symbol = physical_model.name


n_train, n_test = 50, 1001


data_aug_train, data_phys_train =    build_augmented_design(physical_model, specs, n_train)
data_aug_test,  data_phys_test  =    build_augmented_design(physical_model, specs, n_test)


# # initialize GP on θ-space
kernel() = GPMatern52()
metamodel = GaussianProcess(data_aug_train, :y, kernel_type=kernel())
@time "fit!" fit!(metamodel)

μ_test, σ_test = @time "predict:" predict(metamodel, Matrix(data_aug_test[:, w_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")

Ng = 1000
Nx = 1500

print("Params: Ng = $Ng, Nx = $Nx\n")

cabo_min = @time "cabo min loop" cabo_loop(
    physical_model,        # physical_model — explicit, no longer a global lookup
    metamodel,
    data_aug_train,
    y_symbol,
    specs;
    Ng = Ng, Nx = Nx, qoi_type = :pf, y_star = -1.427,
    max_iter = 25, direction = :min,
    tol_BO = 1e-3, tol_BC = 1e-2
)

cabo_max = @time "cabo max loop" cabo_loop(
    physical_model,        # physical_model — explicit, no longer a global lookup
    metamodel,
    cabo_min.data,
    y_symbol,
    specs;
    Ng = Ng, Nx = Nx, qoi_type = :pf, y_star = -1.427,
    max_iter = 25, direction = :max,
    tol_BO = 1e-3, tol_BC = 1e-2
)

cabo_max_2 = @time "cabo max 2 loop" cabo_loop(
    physical_model,        # physical_model — explicit, no longer a global lookup
    cabo_max.gp,
    cabo_max.data,
    y_symbol,
    specs;
    Ng = Ng, Nx = Nx, qoi_type = :pf, y_star = -1.427,
    max_iter = 10, direction = :max,
    tol_BO = 1e-3, tol_BC = 1e-2
)


cabo_max_3 = @time "cabo max 3 loop" cabo_loop(
    physical_model,        # physical_model — explicit, no longer a global lookup
    cabo_max_2.gp,
    cabo_max_2.data,
    y_symbol,
    specs;
    Ng = 400, Nx = 3000, qoi_type = :pf, y_star = -1.427,
    max_iter = 5, direction = :max,
    tol_BO = 1e-3, tol_BC = 1e-2
)


# plot related
Θs_min = reduce(hcat, cabo_min.θ_history)'
Θs_max = reduce(hcat, cabo_max.θ_history)'

using Plots, Statistics

# ── overlay the initial training points (μ1, μ2 columns from data_aug) ──────
data_aug_train_epi = Matrix{Float64}(undef, n_train, length(v_names))
for i in 1:n_train
    data_aug_train_epi[i, :] = augmented_to_epistemic(data_aug_train[i, v_names], specs)
end

plt = scatter(
    data_aug_train_epi[:, 1], data_aug_train_epi[:, 2];
    marker = :diamond, color = :cyan, ms = 5,
    label  = "init samples", markerstrokewidth=0,
    xlabel="θ₁",
    ylabel="θ₂",
    title="Probability failure P(θ₁, θ₂)",
    xlims = (-1.6, 1.6),
    ylims = (-1.6, 1.6),
    legend = :outerbottom,
    legendcolumns=4,
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

x_min, y_min, z_min = 1.5, 0.0, 7.59e-13
x_max, y_max, z_max = -0.57, 0.520, 0.073434

dy = 0.3

scatter!(plt,
    [x_min], [y_min];
    marker = :circle, color = :green, ms = 5, label = "True min",
)
annotate!(
    x_min - 0.25, y_min + dy,
    text("($(round(x_min, digits=2)), $(round(y_min, digits=2)), $(round(z_min, digits=2)))", :black, 8)
)


scatter!(plt,
    [x_max], [y_max];
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
    vcat(cabo_max.L_BO_history, cabo_max_2.L_BO_history),
    xlabel = "Iteration",
    ylabel = "L_BO",
    lw = 2, marker = :circle, ls = :dash,
    title = "MAX: L_BO History",
    legend = false
)

p4 = plot(
    vcat(cabo_max.L_BC_history, cabo_max_2.L_BC_history),
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

p_mean_lower, p_mean_upper = plot_convergence(cabo_min, cabo_max, n_train, [0.0, 7.34 * 10^(-2)], "Pf bound", "Probability failure convergence")

plt_bound = plot(
    p_mean_lower, p_mean_upper,
    layout = (2, 1),
)

display(plt)
display(history)
display(plt_bound)





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
println("           MAX ≈ $(0.083) at θ = [$(-0.56), $(0.501)]")
println("-"^60)
println("Difference:  MIN ≈ $(round(cabo_min.μ_bound - 0, digits=4))")
println("                     MAX ≈ $(round(cabo_max.μ_bound - 8.32 * 10^(-2), digits=4))")
println("="^60)


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








# julia> cabo_min
# (gp = GaussianProcess([-0.9121227157632513 -0.7333686021723974 0.4958503473474535 0.3853204664075677; -0.26342200283283385 -0.360990263400287 1.22652812003661 1.0364333894937896; … ; 0.7426304167041882 1.2942911696289197 -0.5433565641746597 0.30210433514478463; 0.9774577118904798 -0.42948128300244953 -0.5469591535686902 0.48392606481037326], [1.149545372502138, 0.20691372859109047, 0.5085243465528537, -0.012732170944935214, 0.7586530271658661, -0.6231143649642246, 0.036434690814056925, 0.2109141005202068, 0.3427140001176971, 0.35291665520325677  …  -1.4246133304652382, -1.4293368674585818, -1.3955399098144428, -1.4184888924175938, -1.4317832594674564, -1.4187721896802383, -1.4299149581786055, -1.4214487559952032, -1.428461781981392, -1.4316281533201822], GPZeroMean(), GPMatern52(), (lengthscale = [6.618879581345839, 6.618879581345839, 0.5632461147244492, 0.4822172776732614], variance = 0.2424803492693159, noise = 2.0762281583832773e-8), [36.98485072389604, 216.0086038260927, -3.056798788313837, -3.4048174181935726, -24.44268553616173, -18.954925058802285], AbstractGPs.PosteriorGP{AbstractGPs.GP{AbstractGPs.ZeroMean{Float64}, ScaledKernel{TransformedKernel{Matern52Kernel{Distances.Euclidean}, ARDTransform{Vector{Float64}}}, Float64}}, @NamedTuple{α::Vector{Float64}, C::Cholesky{Float64, Matrix{Float64}}, x::ColVecs{Float64, Adjoint{Float64, Matrix{Float64}}, SubArray{Float64, 1, Matrix{Float64}, Tuple{Int64, Base.Slice{Base.OneTo{Int64}}}, true}}, δ::Vector{Float64}}}(AbstractGPs.GP{AbstractGPs.ZeroMean{Float64}, ScaledKernel{TransformedKernel{Matern52Kernel{Distances.Euclidean}, ARDTransform{Vector{Float64}}}, Float64}}(AbstractGPs.ZeroMean{Float64}(), Matern 5/2 Kernel (metric = Distances.Euclidean(0.0))
#         - ARD Transform (dims: 4)
#         - σ² = 0.2424803492693159), (α = [6.01511992155547, -0.3935584143401068, 0.5635794987549093, -0.08128294871098171, 3.6815174852267676, 13.035415989859699, -3.8677214227843506, -0.03499949288943991, 2.547637991489814, 2.283754999971649  …  21.066253962099044, -22.6234426565501, -6.377715013413929, -13.26507459894534, -51.98053618484269, -12.764276253712179, -76.15530408813915, 31.93981376432372, -9.294075983385865, -38.069061103361065], C = Cholesky{Float64, Matrix{Float64}}([0.4924229475454164 0.08211065802599071 … 0.07985344318689354 0.08101484150200705; 5.321685e-318 0.48552878298598806 … -0.0077173463752040275 -0.006197921519102971; … ; 0.0 1.446633846817892 … 0.010715516872127415 0.0009339166811573085; 7.046328273e-315 0.0 … 0.0 0.011022427552673218], 'U', 0), x = SubArray{Float64, 1, Matrix{Float64}, Tuple{Int64, Base.Slice{Base.OneTo{Int64}}}, true}[[-0.9121227157632513, -0.7333686021723974, 0.4958503473474535, 0.3853204664075677], [-0.26342200283283385, -0.360990263400287, 1.22652812003661, 1.0364333894937896], [0.8800701795469846, 0.5881294871054898, -0.8064212470182401, -0.8778962950512286], [1.3413377254485106, 0.18531094633781778, 1.8807936081512553, 0.17637416478086146], [1.253457197355948, 0.07608345130069918, -0.4399131656732337, -0.6128129910166273], [1.0293262315400853, -0.362360481399916, 0.6128129910166273, -0.7388468491852137], [0.24645263097186076, -0.8568084172517362, 0.8778962950512289, 0.02506890825871106], [-0.30388242871719906, 1.3315162112880021, -0.02506890825871106, -0.27931903444745415], [-0.2565825124377286, -1.592646929166358, -0.9541652531461947, -0.07526986209982976], [1.343934045969313, -0.23955879711084346, 0.3853204664075677, -0.02506890825871106]  …  [2.9234519635435583, -0.3666474195694719, -0.723415305134832, 0.44102068034812214], [1.5366815337870088, 0.5810685231647523, -0.6175663349640561, 0.3656204130120367], [3.8665425748567905, -0.5972875987358988, -0.8098970849614038, 0.5565443695552963], [0.4710017489836547, -1.9674460484762868, -0.5594179338701134, 0.6570767716523862], [0.24949706997707521, 1.8113121616135033, -0.4725092980165443, 0.28103898952361583], [4.0, -0.03238196315824714, -0.9161950018527953, 0.40191314855555627], [2.353920263188154, -0.7472257259439582, -0.7070170394754886, 0.5215109421068773], [1.9455777871758102, -0.6636114696272173, -0.6505275509454771, 0.458817751461431], [0.7426304167041882, 1.2942911696289197, -0.5433565641746597, 0.30210433514478463], [0.9774577118904798, -0.42948128300244953, -0.5469591535686902, 0.48392606481037326]], δ = [1.149545372502138, 0.20691372859109047, 0.5085243465528537, -0.012732170944935214, 0.7586530271658661, -0.6231143649642246, 0.036434690814056925, 0.2109141005202068, 0.3427140001176971, 0.35291665520325677  …  -1.4246133304652382, -1.4293368674585818, -1.3955399098144428, -1.4184888924175938, -1.4317832594674564, -1.4187721896802383, -1.4299149581786055, -1.4214487559952032, -1.428461781981392, -1.4316281533201822])), :y, [:u1, :u2, :v1, :v2], ParameterHandling.value ∘ ParameterHandling.var"#unflatten_to_NamedTuple#flatten##13"{Float64, @NamedTuple{lengthscale::ParameterHandling.BoundedArray{Float64, Vector{Float64}, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}, Float64}, variance::ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}, noise::ParameterHandling.Positive{Float64, typeof(exp), Float64}}, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_empty_Tuple#flatten##12"{Float64, Tuple{}}, ParameterHandling.var"#unflatten_Positive#flatten##19"{Float64, ParameterHandling.Positive{Float64, typeof(exp), Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}, ParameterHandling.var"#unflatten_Bounded#flatten##20"{Float64, ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}, ParameterHandling.var"#unflatten_Bounded#flatten##24"{Float64, ParameterHandling.BoundedArray{Float64, Vector{Float64}, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Vector#flatten##3"{Float64, Float64}}}}((lengthscale = ParameterHandling.BoundedArray{Float64, Vector{Float64}, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}, Float64}([-2.2765834537686973, -2.656783527566208, -2.0853066542477974, -2.0853066542477974], 0.27840102690804197, 6.618879581346839, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}(6.340478554436797, 0.27840102690904195), 1.0e-12), variance = ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}(-24.12459936745371, 1.0e-10, 1.0e10, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}(1.0e10, 1.01e-10), 1.0e-12), noise = ParameterHandling.Positive{Float64, typeof(exp), Float64}(-16.279452446511264, exp, 1.4901161193847656e-8)), ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_empty_Tuple#flatten##12"{Float64, Tuple{}}, ParameterHandling.var"#unflatten_Positive#flatten##19"{Float64, ParameterHandling.Positive{Float64, typeof(exp), Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}, ParameterHandling.var"#unflatten_Bounded#flatten##20"{Float64, ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}, ParameterHandling.var"#unflatten_Bounded#flatten##24"{Float64, ParameterHandling.BoundedArray{Float64, Vector{Float64}, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Vector#flatten##3"{Float64, Float64}}}(2, 4, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_empty_Tuple#flatten##12"{Float64, Tuple{}}, ParameterHandling.var"#unflatten_Positive#flatten##19"{Float64, ParameterHandling.Positive{Float64, typeof(exp), Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}, ParameterHandling.var"#unflatten_Bounded#flatten##20"{Float64, ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}(1, 1, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_empty_Tuple#flatten##12"{Float64, Tuple{}}, ParameterHandling.var"#unflatten_Positive#flatten##19"{Float64, ParameterHandling.Positive{Float64, typeof(exp), Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}(0, 1, ParameterHandling.var"#unflatten_to_empty_Tuple#flatten##12"{Float64, Tuple{}}(()), ParameterHandling.var"#unflatten_Positive#flatten##19"{Float64, ParameterHandling.Positive{Float64, typeof(exp), Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}(ParameterHandling.Positive{Float64, typeof(exp), Float64}(-16.279452446511264, exp, 1.4901161193847656e-8), ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}())), ParameterHandling.var"#unflatten_Bounded#flatten##20"{Float64, ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}(ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}(-24.12459936745371, 1.0e-10, 1.0e10, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}(1.0e10, 1.01e-10), 1.0e-12), ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}())), ParameterHandling.var"#unflatten_Bounded#flatten##24"{Float64, ParameterHandling.BoundedArray{Float64, Vector{Float64}, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Vector#flatten##3"{Float64, Float64}}(ParameterHandling.BoundedArray{Float64, Vector{Float64}, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}, Float64}([-2.2765834537686973, -2.656783527566208, -2.0853066542477974, -2.0853066542477974], 0.27840102690804197, 6.618879581346839, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}(6.340478554436797, 0.27840102690904195), 1.0e-12), ParameterHandling.var"#unflatten_to_Vector#flatten##3"{Float64, Float64}()))), AbstractGPs.ZeroMean{Float64}(), Matern 5/2 Kernel (metric = Distances.Euclidean(0.0))
#         - ARD Transform (dims: 4)
#         - σ² = 0.3332879535641224, SurrogateModelling.var"#refit!##8#refit!##9"{GaussianProcess, AbstractGPs.ZeroMean{Float64}}(GaussianProcess(#= circular reference @-2 =#), AbstractGPs.ZeroMean{Float64}()), Matern 5/2 Kernel (metric = Distances.Euclidean(0.0))
#         - ARD Transform (dims: 4)
#         - σ² = 0.2424803492693159, false), data = 51×5 DataFrame
#  Row │ u1          u2          v1          v2          y          
#      │ Float64     Float64     Float64     Float64     Float64    
# ─────┼────────────────────────────────────────────────────────────
#    1 │ -0.912123   -0.733369    0.49585     0.38532     1.14955
#    2 │ -0.263422   -0.36099     1.22653     1.03643     0.206914
#    3 │  0.88007     0.588129   -0.806421   -0.877896    0.508524
#    4 │  1.34134     0.185311    1.88079     0.176374   -0.0127322
#    5 │  1.25346     0.0760835  -0.439913   -0.612813    0.758653
#    6 │  1.02933    -0.36236     0.612813   -0.738847   -0.623114
#    7 │  0.246453   -0.856808    0.877896    0.0250689   0.0364347
#    8 │ -0.303882    1.33152    -0.0250689  -0.279319    0.210914
#    9 │ -0.256583   -1.59265    -0.954165   -0.0752699   0.342714
#   10 │  1.34393    -0.239559    0.38532    -0.0250689   0.352917
#   11 │ -0.423909   -0.663199   -1.22653     0.954165   -0.379233
#   12 │ -0.462554   -0.773354    1.47579     0.227545    0.0258017
#   13 │ -0.119039    1.46788     1.34076     1.22653     0.0961719
#   14 │ -0.749586   -0.522075    1.64485    -1.12639    -0.0702106
#   15 │ -0.215367   -0.675235    0.954165   -0.806421   -0.372741
#   16 │ -0.953711   -0.45652    -1.64485     0.125661   -0.261231
#   17 │ -0.329651    0.154521   -2.32635    -0.553385    0.103818
#   18 │  0.638332   -0.550334    0.806421   -1.64485    -0.113729
#   19 │  0.39621     0.810289   -0.0752699  -0.227545    0.196198
#   20 │ -0.0435438   0.805073   -0.227545   -1.88079     0.0464949
#   21 │  1.94868     0.23371     0.439913    0.331853    1.07779
#   22 │ -4.44053    -0.160517   -0.612813    1.64485    -0.0465131
#   23 │  1.04474    -0.0199413   0.125661    1.47579     0.745384
#   24 │ -0.473818    0.766785    0.227545    0.806421    0.820239
#   25 │ -0.391413   -0.343663    0.279319    1.88079     0.708425
#   26 │  0.475902   -0.273607   -0.67449    -0.49585     0.845347
#   27 │  1.10833    -0.0999418   0.67449     1.12639     0.69819
#   28 │ -0.734145    0.574991    2.32635     0.612813    0.0243057
#   29 │  0.824299    1.42301    -0.38532     0.279319   -1.35413
#   30 │  0.0506846  -0.356633    0.0250689   0.67449    -0.133016
#   31 │  1.19326    -0.155097    0.553385   -0.67449    -0.708864
#   32 │  0.35722     0.391809   -0.125661   -0.439913    0.408539
#   33 │  0.621094   -0.373652   -0.877896    0.738847   -0.899328
#   34 │  0.35945    -0.608144    0.0752699   0.877896    0.425784
#   35 │ -0.304174    0.271668   -1.47579    -1.47579     0.0337382
#   36 │  0.323578    0.620402   -1.03643    -0.176374    0.181573
#   37 │ -0.701961   -0.953858   -0.49585     2.32635    -0.0466917
#   38 │ -1.06169     0.539063    0.176374   -2.32635    -0.0977071
#   39 │  0.515659   -0.228616   -1.34076     0.553385   -0.838581
#   40 │  0.763917    0.177897    0.738847   -1.22653    -0.28991
#   41 │ -2.47389    -1.81343    -0.279319   -0.125661    0.784938
#   42 │  0.0521757   0.0343081  -0.738847   -1.34076     0.139142
#   43 │ -0.89241     0.940964   -0.176374   -0.331853    0.601206
#   44 │ -1.01585     0.445701    0.331853    1.34076     0.824706
#   45 │ -0.569461   -0.858371   -1.88079    -0.954165    0.0439492
#   46 │  0.135177   -0.293366   -1.12639     0.439913   -0.987519
#   47 │ -2.26466     1.24446     1.03643    -0.38532    -0.368609
#   48 │ -0.310583   -0.941848   -0.331853    0.49585    -1.39464
#   49 │  0.993262   -0.35915     1.12639    -1.03643    -0.118143
#   50 │ -0.583488   -0.71855    -0.553385    0.0752699  -0.172756
#   51 │  4.0         4.0        -1.13246    -1.59386     0.445732, θ_bound = [0.5863032735824936, 0.46287100545986304], μ_bound = 0.0, θ_history = [[-1.145695321386079, -1.371692502442772]], L_BO_history = [0.0], L_BC_history = [0.0], bound_history = [0.0, 0.0])

# julia> 



# julia> cabo_max
# (gp = GaussianProcess([-0.9121227157632513 -0.7333686021723974 0.4958503473474535 0.3853204664075677; -0.26342200283283385 -0.360990263400287 1.22652812003661 1.0364333894937896; … ; 0.7426304167041882 1.2942911696289197 -0.5433565641746597 0.30210433514478463; 0.9774577118904798 -0.42948128300244953 -0.5469591535686902 0.48392606481037326], [1.149545372502138, 0.20691372859109047, 0.5085243465528537, -0.012732170944935214, 0.7586530271658661, -0.6231143649642246, 0.036434690814056925, 0.2109141005202068, 0.3427140001176971, 0.35291665520325677  …  -1.4246133304652382, -1.4293368674585818, -1.3955399098144428, -1.4184888924175938, -1.4317832594674564, -1.4187721896802383, -1.4299149581786055, -1.4214487559952032, -1.428461781981392, -1.4316281533201822], GPZeroMean(), GPMatern52(), (lengthscale = [6.618879581345839, 6.618879581345839, 0.5632461147244492, 0.4822172776732614], variance = 0.2424803492693159, noise = 2.0762281583832773e-8), [36.98485072389604, 216.0086038260927, -3.056798788313837, -3.4048174181935726, -24.44268553616173, -18.954925058802285], AbstractGPs.PosteriorGP{AbstractGPs.GP{AbstractGPs.ZeroMean{Float64}, ScaledKernel{TransformedKernel{Matern52Kernel{Distances.Euclidean}, ARDTransform{Vector{Float64}}}, Float64}}, @NamedTuple{α::Vector{Float64}, C::Cholesky{Float64, Matrix{Float64}}, x::ColVecs{Float64, Adjoint{Float64, Matrix{Float64}}, SubArray{Float64, 1, Matrix{Float64}, Tuple{Int64, Base.Slice{Base.OneTo{Int64}}}, true}}, δ::Vector{Float64}}}(AbstractGPs.GP{AbstractGPs.ZeroMean{Float64}, ScaledKernel{TransformedKernel{Matern52Kernel{Distances.Euclidean}, ARDTransform{Vector{Float64}}}, Float64}}(AbstractGPs.ZeroMean{Float64}(), Matern 5/2 Kernel (metric = Distances.Euclidean(0.0))
#         - ARD Transform (dims: 4)
#         - σ² = 0.2424803492693159), (α = [6.01511992155547, -0.3935584143401068, 0.5635794987549093, -0.08128294871098171, 3.6815174852267676, 13.035415989859699, -3.8677214227843506, -0.03499949288943991, 2.547637991489814, 2.283754999971649  …  21.066253962099044, -22.6234426565501, -6.377715013413929, -13.26507459894534, -51.98053618484269, -12.764276253712179, -76.15530408813915, 31.93981376432372, -9.294075983385865, -38.069061103361065], C = Cholesky{Float64, Matrix{Float64}}([0.4924229475454164 0.08211065802599071 … 0.07985344318689354 0.08101484150200705; 5.321685e-318 0.48552878298598806 … -0.0077173463752040275 -0.006197921519102971; … ; 0.0 1.446633846817892 … 0.010715516872127415 0.0009339166811573085; 7.046328273e-315 0.0 … 0.0 0.011022427552673218], 'U', 0), x = SubArray{Float64, 1, Matrix{Float64}, Tuple{Int64, Base.Slice{Base.OneTo{Int64}}}, true}[[-0.9121227157632513, -0.7333686021723974, 0.4958503473474535, 0.3853204664075677], [-0.26342200283283385, -0.360990263400287, 1.22652812003661, 1.0364333894937896], [0.8800701795469846, 0.5881294871054898, -0.8064212470182401, -0.8778962950512286], [1.3413377254485106, 0.18531094633781778, 1.8807936081512553, 0.17637416478086146], [1.253457197355948, 0.07608345130069918, -0.4399131656732337, -0.6128129910166273], [1.0293262315400853, -0.362360481399916, 0.6128129910166273, -0.7388468491852137], [0.24645263097186076, -0.8568084172517362, 0.8778962950512289, 0.02506890825871106], [-0.30388242871719906, 1.3315162112880021, -0.02506890825871106, -0.27931903444745415], [-0.2565825124377286, -1.592646929166358, -0.9541652531461947, -0.07526986209982976], [1.343934045969313, -0.23955879711084346, 0.3853204664075677, -0.02506890825871106]  …  [2.9234519635435583, -0.3666474195694719, -0.723415305134832, 0.44102068034812214], [1.5366815337870088, 0.5810685231647523, -0.6175663349640561, 0.3656204130120367], [3.8665425748567905, -0.5972875987358988, -0.8098970849614038, 0.5565443695552963], [0.4710017489836547, -1.9674460484762868, -0.5594179338701134, 0.6570767716523862], [0.24949706997707521, 1.8113121616135033, -0.4725092980165443, 0.28103898952361583], [4.0, -0.03238196315824714, -0.9161950018527953, 0.40191314855555627], [2.353920263188154, -0.7472257259439582, -0.7070170394754886, 0.5215109421068773], [1.9455777871758102, -0.6636114696272173, -0.6505275509454771, 0.458817751461431], [0.7426304167041882, 1.2942911696289197, -0.5433565641746597, 0.30210433514478463], [0.9774577118904798, -0.42948128300244953, -0.5469591535686902, 0.48392606481037326]], δ = [1.149545372502138, 0.20691372859109047, 0.5085243465528537, -0.012732170944935214, 0.7586530271658661, -0.6231143649642246, 0.036434690814056925, 0.2109141005202068, 0.3427140001176971, 0.35291665520325677  …  -1.4246133304652382, -1.4293368674585818, -1.3955399098144428, -1.4184888924175938, -1.4317832594674564, -1.4187721896802383, -1.4299149581786055, -1.4214487559952032, -1.428461781981392, -1.4316281533201822])), :y, [:u1, :u2, :v1, :v2], ParameterHandling.value ∘ ParameterHandling.var"#unflatten_to_NamedTuple#flatten##13"{Float64, @NamedTuple{lengthscale::ParameterHandling.BoundedArray{Float64, Vector{Float64}, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}, Float64}, variance::ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}, noise::ParameterHandling.Positive{Float64, typeof(exp), Float64}}, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_empty_Tuple#flatten##12"{Float64, Tuple{}}, ParameterHandling.var"#unflatten_Positive#flatten##19"{Float64, ParameterHandling.Positive{Float64, typeof(exp), Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}, ParameterHandling.var"#unflatten_Bounded#flatten##20"{Float64, ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}, ParameterHandling.var"#unflatten_Bounded#flatten##24"{Float64, ParameterHandling.BoundedArray{Float64, Vector{Float64}, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Vector#flatten##3"{Float64, Float64}}}}((lengthscale = ParameterHandling.BoundedArray{Float64, Vector{Float64}, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}, Float64}([-2.2765834537686973, -2.656783527566208, -2.0853066542477974, -2.0853066542477974], 0.27840102690804197, 6.618879581346839, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}(6.340478554436797, 0.27840102690904195), 1.0e-12), variance = ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}(-24.12459936745371, 1.0e-10, 1.0e10, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}(1.0e10, 1.01e-10), 1.0e-12), noise = ParameterHandling.Positive{Float64, typeof(exp), Float64}(-16.279452446511264, exp, 1.4901161193847656e-8)), ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_empty_Tuple#flatten##12"{Float64, Tuple{}}, ParameterHandling.var"#unflatten_Positive#flatten##19"{Float64, ParameterHandling.Positive{Float64, typeof(exp), Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}, ParameterHandling.var"#unflatten_Bounded#flatten##20"{Float64, ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}, ParameterHandling.var"#unflatten_Bounded#flatten##24"{Float64, ParameterHandling.BoundedArray{Float64, Vector{Float64}, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Vector#flatten##3"{Float64, Float64}}}(2, 4, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_empty_Tuple#flatten##12"{Float64, Tuple{}}, ParameterHandling.var"#unflatten_Positive#flatten##19"{Float64, ParameterHandling.Positive{Float64, typeof(exp), Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}, ParameterHandling.var"#unflatten_Bounded#flatten##20"{Float64, ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}(1, 1, ParameterHandling.var"#unflatten_to_Tuple#flatten##11"{Float64, Int64, Int64, ParameterHandling.var"#unflatten_to_empty_Tuple#flatten##12"{Float64, Tuple{}}, ParameterHandling.var"#unflatten_Positive#flatten##19"{Float64, ParameterHandling.Positive{Float64, typeof(exp), Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}}(0, 1, ParameterHandling.var"#unflatten_to_empty_Tuple#flatten##12"{Float64, Tuple{}}(()), ParameterHandling.var"#unflatten_Positive#flatten##19"{Float64, ParameterHandling.Positive{Float64, typeof(exp), Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}(ParameterHandling.Positive{Float64, typeof(exp), Float64}(-16.279452446511264, exp, 1.4901161193847656e-8), ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}())), ParameterHandling.var"#unflatten_Bounded#flatten##20"{Float64, ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}}(ParameterHandling.Bounded{Float64, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}, Float64}(-24.12459936745371, 1.0e-10, 1.0e10, ParameterHandling.var"#transform#bounded##0"{Float64, Float64}(1.0e10, 1.01e-10), 1.0e-12), ParameterHandling.var"#unflatten_to_Real#flatten##2"{Float64, Float64}())), ParameterHandling.var"#unflatten_Bounded#flatten##24"{Float64, ParameterHandling.BoundedArray{Float64, Vector{Float64}, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}, Float64}, ParameterHandling.var"#unflatten_to_Vector#flatten##3"{Float64, Float64}}(ParameterHandling.BoundedArray{Float64, Vector{Float64}, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}, Float64}([-2.2765834537686973, -2.656783527566208, -2.0853066542477974, -2.0853066542477974], 0.27840102690804197, 6.618879581346839, ParameterHandling.var"#transform#bounded##1"{Float64, Float64}(6.340478554436797, 0.27840102690904195), 1.0e-12), ParameterHandling.var"#unflatten_to_Vector#flatten##3"{Float64, Float64}()))), AbstractGPs.ZeroMean{Float64}(), Matern 5/2 Kernel (metric = Distances.Euclidean(0.0))
#         - ARD Transform (dims: 4)
#         - σ² = 0.3332879535641224, SurrogateModelling.var"#refit!##8#refit!##9"{GaussianProcess, AbstractGPs.ZeroMean{Float64}}(GaussianProcess(#= circular reference @-2 =#), AbstractGPs.ZeroMean{Float64}()), Matern 5/2 Kernel (metric = Distances.Euclidean(0.0))
#         - ARD Transform (dims: 4)
#         - σ² = 0.2424803492693159, false), data = 76×5 DataFrame
#  Row │ u1          u2           v1          v2          y          
#      │ Float64     Float64      Float64     Float64     Float64    
# ─────┼─────────────────────────────────────────────────────────────
#    1 │ -0.912123   -0.733369     0.49585     0.38532     1.14955
#    2 │ -0.263422   -0.36099      1.22653     1.03643     0.206914
#    3 │  0.88007     0.588129    -0.806421   -0.877896    0.508524
#    4 │  1.34134     0.185311     1.88079     0.176374   -0.0127322
#    5 │  1.25346     0.0760835   -0.439913   -0.612813    0.758653
#    6 │  1.02933    -0.36236      0.612813   -0.738847   -0.623114
#    7 │  0.246453   -0.856808     0.877896    0.0250689   0.0364347
#    8 │ -0.303882    1.33152     -0.0250689  -0.279319    0.210914
#    9 │ -0.256583   -1.59265     -0.954165   -0.0752699   0.342714
#   10 │  1.34393    -0.239559     0.38532    -0.0250689   0.352917
#   11 │ -0.423909   -0.663199    -1.22653     0.954165   -0.379233
#   12 │ -0.462554   -0.773354     1.47579     0.227545    0.0258017
#   13 │ -0.119039    1.46788      1.34076     1.22653     0.0961719
#   14 │ -0.749586   -0.522075     1.64485    -1.12639    -0.0702106
#   15 │ -0.215367   -0.675235     0.954165   -0.806421   -0.372741
#   16 │ -0.953711   -0.45652     -1.64485     0.125661   -0.261231
#   17 │ -0.329651    0.154521    -2.32635    -0.553385    0.103818
#   18 │  0.638332   -0.550334     0.806421   -1.64485    -0.113729
#   19 │  0.39621     0.810289    -0.0752699  -0.227545    0.196198
#   20 │ -0.0435438   0.805073    -0.227545   -1.88079     0.0464949
#   21 │  1.94868     0.23371      0.439913    0.331853    1.07779
#   22 │ -4.44053    -0.160517    -0.612813    1.64485    -0.0465131
#   23 │  1.04474    -0.0199413    0.125661    1.47579     0.745384
#   24 │ -0.473818    0.766785     0.227545    0.806421    0.820239
#   25 │ -0.391413   -0.343663     0.279319    1.88079     0.708425
#   26 │  0.475902   -0.273607    -0.67449    -0.49585     0.845347
#   27 │  1.10833    -0.0999418    0.67449     1.12639     0.69819
#   28 │ -0.734145    0.574991     2.32635     0.612813    0.0243057
#   29 │  0.824299    1.42301     -0.38532     0.279319   -1.35413
#   30 │  0.0506846  -0.356633     0.0250689   0.67449    -0.133016
#   31 │  1.19326    -0.155097     0.553385   -0.67449    -0.708864
#   32 │  0.35722     0.391809    -0.125661   -0.439913    0.408539
#   33 │  0.621094   -0.373652    -0.877896    0.738847   -0.899328
#   34 │  0.35945    -0.608144     0.0752699   0.877896    0.425784
#   35 │ -0.304174    0.271668    -1.47579    -1.47579     0.0337382
#   36 │  0.323578    0.620402    -1.03643    -0.176374    0.181573
#   37 │ -0.701961   -0.953858    -0.49585     2.32635    -0.0466917
#   38 │ -1.06169     0.539063     0.176374   -2.32635    -0.0977071
#   39 │  0.515659   -0.228616    -1.34076     0.553385   -0.838581
#   40 │  0.763917    0.177897     0.738847   -1.22653    -0.28991
#   41 │ -2.47389    -1.81343     -0.279319   -0.125661    0.784938
#   42 │  0.0521757   0.0343081   -0.738847   -1.34076     0.139142
#   43 │ -0.89241     0.940964    -0.176374   -0.331853    0.601206
#   44 │ -1.01585     0.445701     0.331853    1.34076     0.824706
#   45 │ -0.569461   -0.858371    -1.88079    -0.954165    0.0439492
#   46 │  0.135177   -0.293366    -1.12639     0.439913   -0.987519
#   47 │ -2.26466     1.24446      1.03643    -0.38532    -0.368609
#   48 │ -0.310583   -0.941848    -0.331853    0.49585    -1.39464
#   49 │  0.993262   -0.35915      1.12639    -1.03643    -0.118143
#   50 │ -0.583488   -0.71855     -0.553385    0.0752699  -0.172756
#   51 │  4.0         4.0         -1.13246    -1.59386     0.445732
#   52 │  3.09699     1.36351     -0.42011     0.507772   -0.823739
#   53 │ -0.710346    1.85254     -0.60703     0.371968   -1.27571
#   54 │ -0.354045    2.0666      -0.307564    0.280852   -1.37771
#   55 │ -3.05508     0.494276    -0.828563    0.583902   -0.719939
#   56 │  0.146821   -2.68125     -0.659448    0.598355   -1.31557
#   57 │ -0.215376    2.18819     -0.442445    0.289319   -1.41643
#   58 │  1.85006    -3.78143     -0.70409     0.557194   -1.0427
#   59 │  0.312582    0.595846    -0.374593    0.338263   -1.3752
#   60 │ -0.993133    1.9386      -0.434592    0.44697    -1.18348
#   61 │  0.514367    4.0         -0.182473    0.352892   -0.593324
#   62 │ -0.421252   -0.49688     -0.479554    0.410382   -1.38545
#   63 │  0.407524    1.76284     -0.510732    0.308854   -1.42655
#   64 │  0.441181   -1.17916     -0.64308     0.552553   -1.39578
#   65 │  0.911252   -0.320265    -0.683282    0.482866   -1.39735
#   66 │ -0.0338461  -0.77169     -0.436258    0.449514   -1.40523
#   67 │ -0.933715   -0.227049    -0.363818    0.411718   -1.41498
#   68 │  0.185831   -1.19906     -0.642044    0.503704   -1.37011
#   69 │ -0.813912   -0.901659    -0.376353    0.494928   -1.42758
#   70 │  0.976096    0.704266    -0.609909    0.41376    -1.41389
#   71 │ -0.08029     0.257528    -0.471174    0.421862   -1.43166
#   72 │  0.456659    0.368954    -0.561008    0.433707   -1.41922
#   73 │ -0.269071   -0.503125    -0.429123    0.454371   -1.42624
#   74 │  0.416082    0.00588481  -0.538076    0.446816   -1.42916
#   75 │  0.767619    0.764676    -0.567167    0.38006    -1.42964
#   76 │  0.322342    0.521529    -0.464978    0.401902   -1.42877, θ_bound = [-0.5684911070124284, 0.49137504052676384], μ_bound = 0.0582, θ_history = [[-0.5023613129566977, 0.5992432126895966], [-0.703825097670875, 0.44757055789353895], [-0.3727439563641952, 0.34125339266730714], [-0.9143987216073478, 0.6799784459992573], [-0.7566272041755194, 0.6949189627101462], [-0.5274146884590953, 0.35126138305782595], [-0.8001857667835538, 0.6520391721360528], [-0.4505841283254952, 0.40861510254661493], [-0.5186332264742042, 0.5324608294881874], [-0.22339501194211775, 0.42557986660374336]  …  [-0.4381943832070465, 0.4928866441327493], [-0.739290452744919, 0.5948369717743112], [-0.45260400320403504, 0.5852990071402551], [-0.7067699434236379, 0.49519523459127956], [-0.5592766199089589, 0.5043354285278934], [-0.6560543580325179, 0.5176422280459039], [-0.5125008991275051, 0.5406925502229973], [-0.6317814605581665, 0.5322897410202954], [-0.6625214304046309, 0.4568522276622158], [-0.5524405632329177, 0.48176234165794707]], L_BO_history = [0.24057359066666661, 0.19451799066666664, 0.208742902, 0.23484381333333323, 0.17806059999999999, 0.18464838933333325, 0.22769769333333326, 0.18177511666666676, 0.23194276666666677, 0.1544934933333336  …  0.17308884266666663, 0.20029508533333354, 0.14428527333333344, 0.21681261200000032, 0.15612957200000016, 0.1582176293333338, 0.11629044733333342, 0.21013458866666693, 0.212793266, 0.15187631000000015], L_BC_history = [0.568530954959864, 0.5176166506700298, 0.5386163516120436, 0.5789554922860096, 0.4916780417304696, 0.5039198679292975, 0.5741684250489621, 0.5029200658141088, 0.570815278445316, 0.45698463432282865  …  0.48798946926715286, 0.545883478357585, 0.4402920174811822, 0.5431445092977626, 0.44864973987991436, 0.4438846515517096, 0.37611383285085465, 0.5451524633003304, 0.5550558670018062, 0.44440676678299296], bound_history = [0.3892, 0.3095, 0.3483, 0.2817, 0.3449, 0.3377, 0.4096, 0.3255, 0.0607, 0.0587  …  0.024, 0.0285, 0.0247, 0.0504, 0.049, 0.0471, 0.0566, 0.0546, 0.0574, 0.0582])

# julia> 