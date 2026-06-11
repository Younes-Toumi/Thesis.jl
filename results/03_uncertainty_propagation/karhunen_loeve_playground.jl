using SurrogateModelling
using UncertaintyQuantification
using UncertaintyQuantification: LatinHypercubeSampling, sample, evaluate!
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra
using Plots

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

fit!(metamodel)

μ, σ = predict(metamodel, reshape(X_test, :, 1))
y_pred = μ


println("MSE:               $(round(mse(y_test, y_pred), digits=5))")
println("Q²:                $(round(q2(y_test, y_pred), digits=5))")


function build_kl_sampler(metamodel,
                             W        :: Matrix{Float64},
                             X_train  :: Matrix{Float64};
                             N_samples        :: Int     = 20,
                             energy_threshold :: Float64 = 0.99
    )
 
    kern    = metamodel.kernel_prior
    N0      = size(W, 1)
    n_train = size(X_train, 1)
 
    # ── 1. Eigendecomposition of the kernel matrix at support points ──────────
    K_W = [kern(W[i,:], W[j,:]) for i in 1:N0, j in 1:N0]
 
    eig = eigen(Symmetric(K_W))
    λ   = max.(eig.values, 0.0)           # clamp numerical negatives near zero
    V   = eig.vectors
    idx = sortperm(λ, rev=true)
    λ   = λ[idx];  V = V[:, idx]
 
    cumE = cumsum(λ) ./ sum(λ)
    r    = something(findfirst(≥(energy_threshold), cumE), N0)
    V_r  = V[:, 1:r]
 
    println("kl: N0 = $N0,  r = $r modes,  energy = $(round(100*cumE[r],digits=2))%")
 
    # ── 2. Conditioning quantities at training points ─────────────────────────
    K_train    = [kern(X_train[i,:], X_train[j,:]) for i in 1:n_train, j in 1:n_train]
    L          = cholesky(Symmetric(K_train + 1e-8*I)).L
    K_cross_tW = [kern(X_train[i,:], W[j,:]) for i in 1:n_train, j in 1:N0]
 
    # ── 3. Draw N_samples realisations once; fix ξ for the sampler's lifetime ─
    #
    #  Coefficient derivation (the two λ factors cancel):
    #    h_s(w) = Σ_k  sqrt(λ_k) · ξ_k  ·  (1/sqrt(λ_k)) · k_cross(w,W)ᵀ V[:,k]
    #           = k_cross(w,W)ᵀ · (V_r · ξ_s)
    #
    ξ         = randn(r, N_samples)                    # r × N_samples
    coeff_mat = V_r * ξ                                # N0 × N_samples
 
    #  Condition every realisation on training data in one batched solve:
    #  h_s(X_train) = K_cross_tW · coeff[:,s]          → n_train × N_samples
    #  α_h[:,s]     = (K_train+δI)⁻¹ h_s(X_train)     → n_train × N_samples
    α_h_mat = L' \ (L \ (K_cross_tW * coeff_mat))     # n_train × N_samples
 
    # ── 4. Return the closure ─────────────────────────────────────────────────
    return function(w::AbstractVector)
        kW  = [kern(w, W[j,:])       for j in 1:N0]       # N0-vector
        ktr = [kern(w, X_train[j,:]) for j in 1:n_train]  # n_train-vector
        μ_w = predict(metamodel, reshape(w, 1, :))[1][1]  # GP posterior mean
 
        # Matheron's update for all N_samples simultaneously:
        #   f_s*(w) = μ_post(w)  +  (coeff[:,s]ᵀ kW)  −  (α_h[:,s]ᵀ ktr)
        return μ_w .+ (coeff_mat' * kW) .- (α_h_mat' * ktr)  # N_samples-vector
    end
end


# 1. dense support grid (separate from training data)
W       = rand(5000, 1)
kl_samples = 200

# 2. build the sampler (draws ξ_s here, fixes them)
f = @time "build_kl_sampler: " build_kl_sampler(metamodel, W, reshape(X_train, :, 1); N_samples=kl_samples)

# 3. evaluate at many points for plotting
S = reduce(hcat, f([x]) for x in X_test)


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

