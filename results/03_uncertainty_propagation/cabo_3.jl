# ==============================================================================
# augmented_space_v2.jl
# Generalized augmented-space construction for CABO, supporting THREE input
# categories, unified under UncertaintyQuantification.jl's type system:
#
#   1. PRECISE   — RandomVariable(dist, name)                       fully aleatory
#   2. INTERVAL  — IntervalVariable(lb, ub, name)                    fully epistemic
#   3. HYBRID    — RandomVariable(ProbabilityBox{D}(params), name)   mixed
#
# Each spec contributes 0 or 1 u-column (aleatory, SNS) and 0..K v-columns
# (epistemic, SNS), where K = number of Interval-valued parameters for a
# HYBRID spec (K=1 reproduces your old single-θ `dist_factory` convention;
# K=0 degenerates to PRECISE automatically).
#
# ⚠ FIELD-NAME ASSUMPTIONS — VERIFY BEFORE RELYING ON THE UQ.jl DISPATCH BELOW
#   RandomVariable    : fields  .dist, .name
#   IntervalVariable  : fields  .lb, .ub, .name
#   ProbabilityBox{D} : field   .parameters :: Dict{Symbol,Any}
#   Interval          : fields  .lb, .ub
#
#   Run this first and adjust the methods marked # CHECK if any name differs:
#     println(fieldnames(typeof(x1)))        # RandomVariable
#     println(fieldnames(typeof(x2)))        # RandomVariable (wraps ProbabilityBox)
#     println(fieldnames(typeof(x2.dist)))   # ProbabilityBox
#     println(fieldnames(typeof(x3)))        # IntervalVariable
# ==============================================================================

using SurrogateModelling
using SurrogateModelling: g_function
using UncertaintyQuantification
using Random
using DataFrames
using Printf


# ==============================================================================
# 10. Usage — mirrors your UQ.jl example end to end
# ==============================================================================

# parametric inputs
# lb, ub = -1.5429, 1.5429
x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.0, 1.0), :σ => 1.0)), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.0, 1.0), :σ => 1.0)), :x2)

# interval inputs
# x1 = IntervalVariable(-1.5, 1.5, :x1)
# x2 = IntervalVariable(-1.5, 1.5, :x2)

# x1 = RandomVariable(Normal(0.0, 1.0), :x1)
# x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.3, 1.8), :σ => 2.0)), :x2)
# x3 = IntervalVariable(-0.5, 1.3, :x3)


specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input
 
w_names, u_names, v_names = spec_names(specs)


Φ(z)     = cdf(Normal(), z)


physical_model = (x1, x2) -> x1 - x2 - -2.0
analytical_pf = (μ1, μ2) -> Φ(-(μ1 - μ2 - -2.0) / sqrt(1^2 + 1^2))


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
Nx = 1000

print("Params: Ng = $Ng, Nx = $Nx\n")

cabo_min = @time "cabo min loop" cabo_loop(
    physical_model,        # physical_model — explicit, no longer a global lookup
    metamodel,
    data_aug_train,
    specs;
    Ng = Ng, Nx = Nx, qoi_type = :pf,
    max_iter = 20, direction = :min,
    tol_BO = 1e-3, tol_BC = 1e-2
)


cabo_max = @time "cabo max loop" cabo_loop(
    physical_model,        # physical_model — explicit, no longer a global lookup
    metamodel,
    cabo_min.data,
    specs;
    Ng = Ng, Nx = Nx, qoi_type = :pf,
    max_iter = 20, direction = :max,
    tol_BO = 1e-3, tol_BC = 1e-2
)

# plot related
Θs_min = reduce(hcat, cabo_min.θ_history)'
Θs_max = reduce(hcat, cabo_max.θ_history)'
using Plots, Statistics

# ============================================================
# Grid over epistemic space (x1, θσ)
# ============================================================
n_μ1 = 500
n_μ2 = 500

μ1_grid = range(-1.0, 1.0, length=n_μ1)
μ2_grid  = range(-1.0, 1.0, length=n_μ2)

PfSurface = zeros(n_μ1, n_μ2)

# ============================================================
# Compute response
# ============================================================
for (i, μ1_v) in enumerate(μ1_grid)
    for (j, μ2_v) in enumerate(μ2_grid)
        PfSurface[j, i]  = analytical_pf(μ1_v, μ2_v)
    end
end

# ============================================================
# Heatmap
# ============================================================
plt = heatmap(
    μ1_grid,
    μ2_grid,
    PfSurface,
    xlabel="μ1",
    ylabel="μ2",
    c=:thermal,
    title="expected Pf function: Pf(μ1, μ2)]",
    colorbar=true,
    xlims = (-1.1, 1.1),
    ylims = (-1.1, 1.1),
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
min_idx = argmin(PfSurface)
max_idx = argmax(PfSurface)

x_min, y_min, z_min = μ1_grid[min_idx[2]], μ1_grid[min_idx[1]], minimum(PfSurface)
x_max, y_max, z_max = μ1_grid[max_idx[2]], μ1_grid[max_idx[1]], maximum(PfSurface)

dy = 0.2

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
    ylims = (0, 1),
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
    ylims = (0, 1),
    legend = false
)

history = plot(
    p1, p2, p3, p4,
    layout = (2, 2),
    size = (900, 700)
)


# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("CABO results")
println("="^60)
println("MIN  Pf ≈ $(round(cabo_min.μ_bound, digits=5))" *
        "  at θ = $(round.(cabo_min.θ_bound, digits=3))")
println("MAX  Pf ≈ $(round(cabo_max.μ_bound, digits=5))" *
        "  at θ = $(round.(cabo_max.θ_bound, digits=3))")
println("-"^60)
println("Expected:  MIN ≈ $(round(z_min, digits=5)) at θ = [$(round(x_min, digits=3)), $(round(y_min, digits=3))]")
println("           MAX ≈ $(round(z_max, digits=5)) at θ = [$(round(x_max, digits=3)), $(round(y_max, digits=3))]")
println("-"^60)
println("Difference:  MIN ≈ $(round(cabo_min.μ_bound - z_min, digits=5))")
println("                     MAX ≈ $(round(cabo_max.μ_bound - z_max, digits=5))")
println("="^60)




display(plt)
display(history)






# Initial samples: 40
# fit!: 5.293165 seconds (18.22 M allocations: 1.362 GiB, 6.34% gc time, 42.20% compilation time: <1% of which was recompilation)
# predict:: 0.000760 seconds (158 allocations: 1.530 MiB)
# MSE: 0.04247
# Q²:  0.85848
# Params: Ng = 1000, Nx = 1000

# ━━━ CABO Iteration 1 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.501, 0.656]    μ_qoi(θ*) ≈ -1.25e+00    σ_qoi(θ*) ≈ 6.29e-02
# bo objective : 284.673920 seconds (2.02 M allocations: 357.692 GiB, 38.78% gc time)
#     Acquisition θ⁺ = [-0.7767, 0.672]    EI = 0.0723    COV = 0.05041
# pvc objective : 0.415616 seconds (1.30 M allocations: 1.486 GiB, 32.51% gc time)

# ━━━ CABO Iteration 2 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.501, 0.656]    μ_qoi(θ*) ≈ -1.23e+00    σ_qoi(θ*) ≈ 5.26e-02
# bo objective : 278.500776 seconds (2.02 M allocations: 358.139 GiB, 38.42% gc time)
#     Acquisition θ⁺ = [-0.5282, 0.5734]    EI = 0.0307    COV = 0.04283
# pvc objective : 0.487910 seconds (1.30 M allocations: 1.492 GiB, 39.23% gc time)

# ━━━ CABO Iteration 3 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.528, 0.573]    μ_qoi(θ*) ≈ -1.31e+00    σ_qoi(θ*) ≈ 1.63e-02
# bo objective : 287.836487 seconds (2.02 M allocations: 358.592 GiB, 38.15% gc time)
#     Acquisition θ⁺ = [-0.5414, 0.543]    EI = 0.0081    COV = 0.01244
# pvc objective : 0.347880 seconds (1.30 M allocations: 1.502 GiB, 22.86% gc time)

# ━━━ CABO Iteration 4 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.528, 0.573]    μ_qoi(θ*) ≈ -1.33e+00    σ_qoi(θ*) ≈ 1.65e-02
# bo objective : 284.082749 seconds (2.02 M allocations: 359.039 GiB, 37.98% gc time)
#     Acquisition θ⁺ = [-1.1371, 0.6644]    EI = 0.0088    COV = 0.01246
# pvc objective : 0.530842 seconds (1.30 M allocations: 1.508 GiB, 44.41% gc time)

# ━━━ CABO Iteration 5 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.528, 0.573]    μ_qoi(θ*) ≈ -1.33e+00    σ_qoi(θ*) ≈ 1.35e-02
# bo objective : 280.096303 seconds (2.02 M allocations: 359.488 GiB, 38.40% gc time)
#     Acquisition θ⁺ = [-1.4794, 0.3031]    EI = 0.0007    COV = 0.01021
# pvc objective : 0.418452 seconds (1.30 M allocations: 1.516 GiB, 32.12% gc time)

# ━━━ CABO Iteration 6 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.528, 0.573]    μ_qoi(θ*) ≈ -1.33e+00    σ_qoi(θ*) ≈ 1.56e-02
# bo objective : 281.051609 seconds (2.02 M allocations: 359.935 GiB, 39.04% gc time)
#     Acquisition θ⁺ = [-0.1045, 1.5]    EI = 0.0009    COV = 0.01176
# pvc objective : 0.321657 seconds (1.30 M allocations: 1.523 GiB, 15.35% gc time)

# ━━━ CABO Iteration 7 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.528, 0.573]    μ_qoi(θ*) ≈ -1.32e+00    σ_qoi(θ*) ≈ 1.75e-02
# bo objective : 284.818577 seconds (2.02 M allocations: 360.388 GiB, 38.62% gc time)
#     Acquisition θ⁺ = [1.5, -0.8391]    EI = 0.0006    COV = 0.01318
# pvc objective : 0.396298 seconds (1.30 M allocations: 1.534 GiB, 28.04% gc time)

# ━━━ CABO Iteration 8 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.528, 0.573]    μ_qoi(θ*) ≈ -1.32e+00    σ_qoi(θ*) ≈ 1.33e-02
# bo objective : 290.676045 seconds (2.02 M allocations: 360.835 GiB, 38.01% gc time)
#     Acquisition θ⁺ = [-0.4865, 0.5251]    EI = 0.0072    COV = 0.01003
# pvc objective : 0.390241 seconds (1.30 M allocations: 1.541 GiB, 28.82% gc time)

# ━━━ CABO Iteration 9 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.486, 0.525]    μ_qoi(θ*) ≈ -1.35e+00    σ_qoi(θ*) ≈ 1.31e-02
# bo objective : 283.719082 seconds (2.02 M allocations: 361.284 GiB, 38.39% gc time)
#     Acquisition θ⁺ = [-0.4682, 0.5138]    EI = 0.0057    COV = 0.009716
# pvc objective : 0.342353 seconds (1.30 M allocations: 1.550 GiB, 7.72% gc time)

# ━━━ CABO Iteration 10 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [-0.541, 0.543]    μ_qoi(θ*) ≈ -1.36e+00    σ_qoi(θ*) ≈ 1.34e-02
# bo objective : 282.705257 seconds (2.02 M allocations: 361.731 GiB, 38.75% gc time)
#     Acquisition θ⁺ = [-1.4953, -1.4577]    EI = 0.0004    COV = 0.009883

# ✓ converged

#   ► MIN bound ≈ -1.3537  at  θ = [-0.5414, 0.543]
# cabo min loop: 2864.161915 seconds (38.10 M allocations: 3.550 TiB, 38.54% gc time, 0.00% compilation time: <1% of which was recompilation)

# ━━━ CABO Iteration 1 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.501, 0.964]    μ_qoi(θ*) ≈ 1.29e+00    σ_qoi(θ*) ≈ 3.12e-02
# bo objective : 285.744455 seconds (2.02 M allocations: 363.081 GiB, 38.22% gc time)
#     Acquisition θ⁺ = [0.5741, 0.8333]    EI = 0.0481    COV = 0.02423
# pvc objective : 1.409284 seconds (1.30 M allocations: 1.585 GiB, 76.78% gc time)

# ━━━ CABO Iteration 2 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.574, 0.833]    μ_qoi(θ*) ≈ 1.38e+00    σ_qoi(θ*) ≈ 2.41e-02
# bo objective : 284.483759 seconds (2.02 M allocations: 363.528 GiB, 38.59% gc time)
#     Acquisition θ⁺ = [0.5634, 0.7214]    EI = 0.0145    COV = 0.01752
# pvc objective : 0.364985 seconds (1.30 M allocations: 1.593 GiB, 24.39% gc time)

# ━━━ CABO Iteration 3 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.574, 0.833]    μ_qoi(θ*) ≈ 1.39e+00    σ_qoi(θ*) ≈ 1.60e-02
# bo objective : 288.246780 seconds (2.02 M allocations: 363.976 GiB, 38.33% gc time)
#     Acquisition θ⁺ = [0.5137, 0.8182]    EI = 0.0095    COV = 0.01153
# pvc objective : 0.413986 seconds (1.30 M allocations: 1.604 GiB, 19.94% gc time)

# ━━━ CABO Iteration 4 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.514, 0.818]    μ_qoi(θ*) ≈ 1.39e+00    σ_qoi(θ*) ≈ 1.48e-02
# bo objective : 286.752186 seconds (2.02 M allocations: 364.424 GiB, 38.22% gc time)
#     Acquisition θ⁺ = [0.5418, 0.8274]    EI = 0.0085    COV = 0.0106
# pvc objective : 0.534413 seconds (1.30 M allocations: 1.612 GiB, 42.43% gc time)

# ━━━ CABO Iteration 5 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.574, 0.833]    μ_qoi(θ*) ≈ 1.42e+00    σ_qoi(θ*) ≈ 9.57e-04
# bo objective : 286.904134 seconds (2.02 M allocations: 364.881 GiB, 38.39% gc time)
#     Acquisition θ⁺ = [0.6925, 0.8673]    EI = 0.051    COV = 0.0006762
# pvc objective : 0.362967 seconds (1.30 M allocations: 1.626 GiB, 20.68% gc time)

# ━━━ CABO Iteration 6 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.574, 0.833]    μ_qoi(θ*) ≈ 1.42e+00    σ_qoi(θ*) ≈ 8.30e-04
# bo objective : 284.562669 seconds (2.02 M allocations: 365.328 GiB, 38.37% gc time)
#     Acquisition θ⁺ = [-0.778, -1.5]    EI = 0.0012    COV = 0.0005855
# pvc objective : 0.525957 seconds (1.30 M allocations: 1.635 GiB, 41.61% gc time)

# ━━━ CABO Iteration 7 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.574, 0.833]    μ_qoi(θ*) ≈ 1.42e+00    σ_qoi(θ*) ≈ 7.88e-04
# bo objective : 284.542707 seconds (2.02 M allocations: 365.775 GiB, 38.62% gc time)
#     Acquisition θ⁺ = [0.6005, 0.8035]    EI = 0.0116    COV = 0.0005564
# pvc objective : 0.410552 seconds (1.30 M allocations: 1.643 GiB, 29.27% gc time)

# ━━━ CABO Iteration 8 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.563, 0.721]    μ_qoi(θ*) ≈ 1.41e+00    σ_qoi(θ*) ≈ 6.26e-04
# bo objective : 287.663823 seconds (2.02 M allocations: 366.222 GiB, 38.33% gc time)
#     Acquisition θ⁺ = [0.3786, 0.5945]    EI = 0.4398    COV = 0.0004454
# pvc objective : 0.540588 seconds (1.30 M allocations: 1.652 GiB, 43.07% gc time)

# ━━━ CABO Iteration 9 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.563, 0.721]    μ_qoi(θ*) ≈ 1.40e+00    σ_qoi(θ*) ≈ 8.00e-04
# bo objective : 281.972745 seconds (2.02 M allocations: 366.671 GiB, 38.74% gc time)
#     Acquisition θ⁺ = [0.6147, 0.935]    EI = 0.091    COV = 0.0005697
# pvc objective : 0.569655 seconds (1.30 M allocations: 1.663 GiB, 42.87% gc time)

# ━━━ CABO Iteration 10 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.563, 0.721]    μ_qoi(θ*) ≈ 1.41e+00    σ_qoi(θ*) ≈ 5.11e-04
# bo objective : 284.176324 seconds (2.02 M allocations: 367.118 GiB, 38.61% gc time)
#     Acquisition θ⁺ = [0.3999, 0.8217]    EI = 0.0886    COV = 0.0003626
# pvc objective : 0.435612 seconds (1.30 M allocations: 1.673 GiB, 31.77% gc time)

# ━━━ CABO Iteration 11 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.563, 0.721]    μ_qoi(θ*) ≈ 1.41e+00    σ_qoi(θ*) ≈ 3.34e-04
# bo objective : 294.312344 seconds (2.02 M allocations: 367.565 GiB, 38.19% gc time)
#     Acquisition θ⁺ = [0.8801, 1.4999]    EI = 0.0014    COV = 0.0002371
# pvc objective : 0.349244 seconds (1.30 M allocations: 1.682 GiB, 7.79% gc time)

# ━━━ CABO Iteration 12 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.563, 0.721]    μ_qoi(θ*) ≈ 1.40e+00    σ_qoi(θ*) ≈ 4.04e-04
# bo objective : 290.025217 seconds (2.07 M allocations: 368.015 GiB, 38.19% gc time)
#     Acquisition θ⁺ = [0.5071, 0.6862]    EI = 0.0711    COV = 0.0002874
# pvc objective : 0.375446 seconds (1.33 M allocations: 1.693 GiB, 21.60% gc time)

# ━━━ CABO Iteration 13 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.563, 0.721]    μ_qoi(θ*) ≈ 1.41e+00    σ_qoi(θ*) ≈ 2.26e-04
# bo objective : 289.292541 seconds (2.07 M allocations: 368.462 GiB, 38.07% gc time)
#     Acquisition θ⁺ = [0.6007, 0.8673]    EI = 0.0255    COV = 0.00016
# pvc objective : 0.570779 seconds (1.33 M allocations: 1.702 GiB, 43.90% gc time)

# ━━━ CABO Iteration 14 / 20 ━━━━━━━━━━━━━━━━━━━━━━━━━━━
#     Incumbent θ* = [0.563, 0.721]    μ_qoi(θ*) ≈ 1.41e+00    σ_qoi(θ*) ≈ 8.46e-05
# bo objective : 0.898594 seconds (9.01 k allocations: 1.845 GiB, 7.21% gc time)
#     Acquisition θ⁺ = [-1.3544, -0.9748]    EI = 0.0    COV = 5.985e-5

# ✓ converged

#   ► MAX bound ≈ 1.387  at  θ = [0.5634, 0.7214]
# cabo max loop: 3772.863808 seconds (50.82 M allocations: 4.709 TiB, 38.44% gc time)

# ============================================================
# CABO results
# ============================================================
# MIN  E[g] ≈ -1.354  at θ = [-0.541, 0.543]
# MAX  E[g] ≈ 1.387  at θ = [0.563, 0.721]
# ------------------------------------------------------------
# Expected:  MIN ≈ -1.351 at θ = [-0.568, 0.526]
#            MAX ≈ 1.327 at θ = [0.556, 0.803]
# ------------------------------------------------------------
# Difference:  MIN ≈ -0.003
#                      MAX ≈ 0.06
# ============================================================