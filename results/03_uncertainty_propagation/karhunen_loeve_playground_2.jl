using SurrogateModelling
using SurrogateModelling: g_function
using UncertaintyQuantification
using Random
using DataFrames
using Printf
using Plots

Random.seed!(42)

# ==============================================================================
# 10. Usage — mirrors your UQ.jl example end to end
# ==============================================================================

# parametric inputs
# lb, ub = -1.5429, 1.5429
x = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x)

specs = InputSpec.([x])     # broadcasts dispatch over each UQ.jl input
 
w_names, u_names, v_names = spec_names(specs)

physical_model = x -> sin(x)
expected_response = μ -> exp(-(1 * 0.1)^2 / 2 ) * sin(μ)


n_train, n_test = 50, 1001


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
fit!(metamodel)

μ_test, σ_test = predict(metamodel, Matrix(data_aug_test[:, w_names]))

# println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
# println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")


# 1. dense support grid (separate from training data)

N0 = 500
Ng = 500

W_eole, _ = build_augmented_design(nothing, specs, N0)
W_eole = Matrix(W_eole)
W_support    = Matrix(data_aug_train[:, w_names])

print("\n\n")

# 2. build the sampler (draws ξ_s here, fixes them)
gp_samples = build_kl_sampler(metamodel, W_eole, W_support; N_samples=Ng)


Nx = 500

W_aug, _ = build_augmented_design(nothing, specs, Nx)
u_samples = Matrix(W_aug[:, u_names])
# let me select best candidate
v_data = Matrix(data_aug_train[:, v_names])
v_star_index, μ_qoi, σ_qoi = best_candidate(:mean, gp_samples, u_samples, v_data, :min; α=1.0)

v_star     = Vector(data_aug_train[v_star_index, v_names])
μ_qoi_star = μ_qoi[v_star_index]
σ_qoi_star = σ_qoi[v_star_index]
θ_star     = augmented_to_epistemic(v_star, specs)

@printf("Nx, Ng = (%d, %d)    Incumbent θ* = %s    μ_qoi(θ*) ≈ %.2e    σ_qoi(θ*) ≈ %.2e\n",
        Nx, Ng, string(round.(θ_star, digits=3)), μ_qoi_star, σ_qoi_star)

μ_grid = range(-1.5, 1.5, length=500)

MeanVals = zeros(500)

# ============================================================
# Compute response
# ============================================================
for (i, μ_v) in enumerate(μ_grid)
    MeanVals[i]  = expected_response(μ_v)
end



p1 = plot(title  = "kl samples", xlabel = "μ", ylabel = "y", legend = :topleft, size = (800, 420))

# GP uncertainty band ±2σ
plot!(p1, μ_grid, MeanVals, color = :steelblue, linewidth = 2, label = "E[g]")


v_grid    = range(-2.2, 2.2, length=200)   # SNS epistemic coordinate
θ_vals    = Float64[]
eole_vals = Float64[]
gp_vals   = Float64[]

for v_val in v_grid
    v   = [v_val]
    θ   = augmented_to_epistemic(v, specs)[1]
    μ_v, _ = estimate_propagation_qoi(:mean, gp_samples, u_samples, v)

    gp_val = mean(predict(metamodel, hcat(u_samples, repeat(v', Nx,1)); mode=:mean))


    push!(θ_vals, θ)
    push!(eole_vals, μ_v)
    push!(gp_vals, gp_val)

end
plot!(p1, θ_vals, eole_vals, color=:crimson, lw=2, label="EOLE estimate")
plot!(p1, θ_vals, gp_vals, color=:green, lw=2, label="GP estimate")

display(p1)