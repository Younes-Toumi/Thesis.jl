using SurrogateModelling
using UncertaintyQuantification
using UncertaintyQuantification: LatinHypercubeSampling, sample, evaluate!
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra
using Plots
using KernelFunctions

Random.seed!(42)


# ============================================================
# Inputs + Model
# ============================================================
X = [RandomVariable.(Uniform(0, 1), :x)]

model = Model(
    rv -> (6 .* rv.x .- 2).^2 .* sin.(12 .* rv.x .- 4),
    :y
) # forrester

# ============================================================
# Sampling for: 1. train and 2. test #
# ============================================================

n_train, n_test = 10, 1000

design_train = MonteCarlo(n_train)
design_test = MonteCarlo(n_test)

data_train = sample(X, design_train)
data_test = sample(X, design_test)

evaluate!(model, data_train)
evaluate!(model, data_test)

X_train = data_train[:, :x]
y_train = data_train[:, :y]

X_test = data_test[:, :x]
y_test = data_test[:, :y]

# ============================================================
# Initial GP hyperparameters
# ============================================================

metamodel = GaussianProcess(data_train, :y, kernel_type=GPSquaredExponential())

@time "fit!" fit!(metamodel)

μ, σ = predict(metamodel, reshape(X_test, :, 1))
y_pred = μ


println("MSE:               $(round(mse(y_test, y_pred), digits=5))")
println("Q²:                $(round(q2(y_test, y_pred), digits=5))")

# ── Posterior EOLE sampler with batch dispatch ────────────────────────────────
function build_kl_sampler(gp, W::Matrix{Float64}, X_train::Matrix{Float64};
                           N_samples, energy_threshold=0.99, δ=1e-8)

    kern    = gp.kernel_posterior
    N0      = size(W, 1)

    # Prior kernel matrices
    # ── in build_kl_sampler (replace all three comprehensions) ────────────────────
    K_W   = kernelmatrix(kern, RowVecs(W),       RowVecs(W))
    K_tr  = kernelmatrix(kern, RowVecs(X_train), RowVecs(X_train))
    K_ctW = kernelmatrix(kern, RowVecs(X_train), RowVecs(W))

    L = cholesky(Symmetric(K_tr + δ*I)).L
    A = L' \ (L \ K_ctW)                # K_train⁻¹ K_ctW  (n_train × N0) — reused in closure

    # ── FIX 1: eigendecompose the POSTERIOR covariance, not the prior ─────────
    # K_W_post = K_W − K_ctW^T K_train⁻¹ K_ctW
    K_W_post = Symmetric(K_W .- K_ctW' * A)

    eig    = eigen(K_W_post)
    λ_post = max.(eig.values, 0.0)
    V_post = eig.vectors
    idx    = sortperm(λ_post, rev=true)
    λ_post, V_post = λ_post[idx], V_post[:, idx]

    # Drop modes below numerical floor to avoid 1/√λ blowup
    floor  = max(1e-10 * sum(λ_post), 1e-14)
    keep   = λ_post .> floor
    V_r, λ_r = V_post[:, keep], λ_post[keep]
    r = sum(keep)

    print("KL: N0 = $N0,  r = $r modes\n")

    # ── FIX 2: Form 2 coefficient — gives Var[h(w)] ≈ k_post(w,w) ────────────
    Ξ         = randn(r, N_samples)
    coeff_mat = V_r * (Ξ ./ reshape(sqrt.(λ_r), :, 1))  # N0 × N_samples

    # ── Closure: dispatch on AbstractVector (single) vs AbstractMatrix (batch) ─
    return function gp_samples(input)
        if input isa AbstractVector
            # ── single point ──────────────────────────────────────────────────
            kW   = vec(kernelmatrix(kern, RowVecs(reshape(input,1,:)), RowVecs(W)))
            ktr  = vec(kernelmatrix(kern, RowVecs(reshape(input,1,:)), RowVecs(X_train)))
            μ_w  = predict(gp, reshape(input, 1, :); mode=:mean)
            kW_post = kW .- A' * ktr           # N0-vector  (= 0 at training pts)
            return μ_w .+ coeff_mat' * kW_post  # N_samples-vector

        else
            # ── batch: input is Nx × d ────────────────────────────────────────
            K_qW   = @time "kernelmatrix" kernelmatrix(kern, RowVecs(input), RowVecs(W))        # Nx_q × N0
            K_qtr  = @time "kernelmatrix" kernelmatrix(kern, RowVecs(input), RowVecs(X_train))  # Nx_q × n_train
            μ_q    = @time "predict" predict(gp, input; mode=:mean)        # Nx_q-vector (one predict call)
            K_qW_post = @time "K_qW_post" K_qW .- K_qtr * A         # Nx_q × N0

            return @time "return" μ_q' .+ coeff_mat' * K_qW_post'  # N_samples × Nx_q
        end
    end
end



# 1. dense support grid (separate from training data)
W       = rand(50, 1)
kl_samples = 1000

print("\n\n")

# 2. build the sampler (draws ξ_s here, fixes them)
f = @time "build_kl_sampler: " build_kl_sampler(metamodel, W, reshape(X_train, :, 1); N_samples=kl_samples)

# 3. evaluate at many points for plotting
S = @time "nomal calling" reduce(hcat, f([x]) for x in X_test)
print("\n")
S = @time "batch calling" f(reshape(X_test, :, 1))

# Sort test points
perm = sortperm(X_test)

X_plot  = X_test[perm]
y_plot  = y_test[perm]
μ_plot  = μ[perm]
σ_plot  = σ[perm]

S_plot = S[:, perm]

p1 = plot(title  = "kl posterior samples — Forrester  (N0=$n_train)", xlabel = "x", ylabel = "y", legend = :topleft, size = (800, 420))

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

