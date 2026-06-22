using SurrogateModelling
using SurrogateModelling: g_function
using UncertaintyQuantification
using Random
using DataFrames
using Printf
using Plots

# ==============================================================================
# 10. Usage — mirrors your UQ.jl example end to end
# ==============================================================================

# parametric inputs
# lb, ub = -1.5429, 1.5429
# x = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x)
x = RandomVariable(Normal(0, 1), :x)
# x = IntervalVariable(-2.0, 2.0, :x)

specs = InputSpec.([x])     # broadcasts dispatch over each UQ.jl input
 
w_names, u_names, v_names = spec_names(specs)

physical_model = x -> sin(x)
expected_response = μ -> exp(-(1 * 0.1)^2 / 2 ) * sin(μ)


n_train, n_test = 15, 1001


data_aug_train, data_phys_train =    build_augmented_design(physical_model, specs, n_train)
data_aug_test,  data_phys_test  =    build_augmented_design(physical_model, specs, n_test)

X_train  = data_phys_train[:, :x]
y_train  = data_phys_train[:, :y]

X_test  = data_phys_test[:, :x]
y_test  = data_phys_test[:, :y]

W_test  = Matrix(data_aug_test[:, w_names])


# # initialize GP on θ-space
kernel() = GPSquaredExponential()
metamodel = GaussianProcess(data_aug_train, :y, kernel_type=kernel())
@time "fit!" fit!(metamodel)

μ_test, σ_test = @time "predict:" predict(metamodel, Matrix(data_aug_test[:, w_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")


# 1. dense support grid (separate from training data)

N0 = 100
kl_samples = 100

W_eole, _ = build_augmented_design(nothing, specs, N0)
W_eole = Matrix(W_eole)
W_support    = Matrix(data_aug_train[:, w_names])

print("\n\n")

# 2. build the sampler (draws ξ_s here, fixes them)
f = @time "build_kl_sampler: " build_kl_sampler(metamodel, W_eole, W_support; N_samples=kl_samples)

# 3. evaluate at many points for plotting
S = @time "batch calling" f(W_test)

# Sort test points
perm = sortperm(X_test)

X_plot  = data_phys_test[perm, :x]
y_plot  = data_phys_test[perm, :y]
μ_plot  = μ_test[perm]
σ_plot  = σ_test[perm]

S_plot = S[:, perm]

p1 = plot(title  = "kl samples", xlabel = "x", ylabel = "y", legend = :topleft, size = (800, 420))

# GP uncertainty band ±2σ
plot!(p1, X_plot, μ_plot .+ 2*σ_plot,      fillrange = μ_plot .- 2*σ_plot, fillalpha = 0.5, linealpha = 0, color = :steelblue, label = "GP ±2σ")
 
for i in 1:kl_samples
    plot!(p1, X_plot, S_plot[i, :], linewidth = 2, label = "")
end

# Training data — should lie exactly on ALL sample curves
scatter!(p1, X_train, y_train,   color = :red, markersize = 5, markerstrokewidth = 0, label = "training data")

# True function and GP mean
plot!(p1, X_plot, y_plot,        color = :black, linewidth = 2, label = "true function")
plot!(p1, X_plot, μ_plot,             color = :blue, linewidth = 2, linestyle = :dash, label = "GP posterior mean")

display(p1)