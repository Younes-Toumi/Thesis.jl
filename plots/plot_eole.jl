using Random
using Plots
using Distributions
using UncertaintyQuantification
using UncertaintyQuantification: sample
using SurrogateModelling
using DataFrames
using KernelFunctions
using LinearAlgebra

gr()
Random.seed!(42)

# ---------------------------------------------------------
# Model + training data
# ---------------------------------------------------------
X = RandomVariable(Uniform(0.0, 1.0), :x)

model = Model(
    rv -> (6 .* rv.x .- 2).^2 .* sin.(12 .* rv.x .- 4),
    :y
)
forrester(x) = (6x - 2)^2 * sin(12x - 4)

n_train = 10
design_train = LatinHypercubeSampling(n_train)
data_train = sample(X, design_train)
evaluate!(model, data_train)

gp = GaussianProcess(data_train, :y; kernel_type=GPMatern52())
fit!(gp)

# ---------------------------------------------------------
# EOLE realization generator -- SAME eigendecomposition as
# build_kl_sampler_new, but returns the RAW (unaggregated)
# (n_query x Ng) matrix of realization VALUES at each query point,
# instead of reducing each realization down to a scalar QoI.
# This is exactly what's needed for "show me G^(j)(x)".
# ---------------------------------------------------------
function eole_realizations(gp, W::Matrix{Float64}, X_train::Matrix{Float64},
                            X_query::Matrix{Float64}; N_samples::Int,
                            energy_threshold=0.99, δ=1e-8)
    kern    = gp.kernel_posterior
    N0      = size(W, 1)

    K_W   = kernelmatrix(kern, RowVecs(W),       RowVecs(W))
    K_tr  = kernelmatrix(kern, RowVecs(X_train), RowVecs(X_train))
    K_ctW = kernelmatrix(kern, RowVecs(X_train), RowVecs(W))
    L = cholesky(Symmetric(K_tr + δ*I)).L
    A = L' \ (L \ K_ctW)

    K_W_post = Symmetric(K_W .- K_ctW' * A)
    eig    = eigen(K_W_post)
    λ_post = max.(eig.values, 0.0)
    V_post = eig.vectors
    idx    = sortperm(λ_post, rev=true)
    λ_post, V_post = λ_post[idx], V_post[:, idx]

    total_energy      = sum(λ_post)
    cumulative_energy = cumsum(λ_post) ./ total_energy
    r_energy = searchsortedfirst(cumulative_energy, energy_threshold)
    floor_   = max(1e-10 * total_energy, 1e-14)
    r_floor  = something(findlast(λ_post .> floor_), r_energy)
    r        = min(r_energy, r_floor)
    V_r, λ_r = V_post[:, 1:r], λ_post[1:r]

    Ng = N_samples
    Ξ         = randn(r, Ng)
    coeff_mat = V_r * (Ξ ./ reshape(sqrt.(λ_r), :, 1))    # N0 × Ng

    K_qW      = kernelmatrix(kern, RowVecs(X_query), RowVecs(W))
    K_qtr     = kernelmatrix(kern, RowVecs(X_query), RowVecs(X_train))
    μ_q       = predict(gp, X_query; mode=:mean)
    K_qW_post = K_qW .- K_qtr * A                          # n_query × N0

    G = μ_q .+ K_qW_post * coeff_mat                        # n_query × Ng
    return G, r, cumulative_energy[r]
end

# ---------------------------------------------------------
# EOLE support grid (defines the eigenbasis) and plotting/query grid
# ---------------------------------------------------------
N0 = 200
W  = reshape(collect(range(0, 1, length=N0)), :, 1)

n_plot  = 300
x_plot  = collect(range(0, 1, length=n_plot))
X_query = reshape(x_plot, :, 1)

X_train_mat = Matrix(data_train[:, [:x]])

n_realizations = 10
G, r, energy_captured = eole_realizations(
    gp, W, X_train_mat, X_query; N_samples=n_realizations
)
println("EOLE: retained r=$r modes, capturing $(round(100*energy_captured, digits=2))% of posterior energy over W")

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
p = plot(
    x_plot, forrester.(x_plot);
    color = :black, lw = 2, ls = :dash,
    label = "Analytical (Forrester function)",
    xlabel = "x", ylabel = "y",
    title = "EOLE realizations",
    legend = :topleft, size = (950, 550),
)

for j in 1:n_realizations
    plot!(
        p, x_plot, G[:, j];
        alpha = 1.0, lw = 2,
        label = j == 1 ? "EOLE realizations  G_j(x)" : "",
    )
end

scatter!(
    p, data_train.x, data_train.y;
    color = :black, ms = 6, msw = 0,
    label = "Training data",
)

display(p)
