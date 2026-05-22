using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Distributions
using ParameterHandling
using LinearAlgebra

function pso_optimize(f::Function,
                      initial_particles::Matrix{Float64},
                      bounds::Matrix{Float64};
                      max_iter::Int = 100,
                      w::Float64 = 0.8,
                      c1::Float64 = 1.5,
                      c2::Float64 = 1.5,
                      mode::Symbol = :max,
                      delta::Float64 = 1e-4,
                      track::Bool = true)

    n_particles, dim = size(initial_particles)

    pos = copy(initial_particles)
    vel = zeros(n_particles, dim)

    # evaluate swarm
    fvals = [f(pos[i, :]) for i in 1:n_particles]

    pbest_pos = copy(pos)
    pbest_val = copy(fvals)

    best_idx = argmax(pbest_val)
    gbest_pos = copy(pbest_pos[best_idx, :])
    gbest_val = pbest_val[best_idx]

    history = track ? Vector{Matrix{Float64}}() : nothing

    better(a, b) = mode == :max ? (a > b) : (a < b)

    for iter in 1:max_iter
        for i in 1:n_particles

            r1 = rand(dim)
            r2 = rand(dim)

            for j in 1:dim
                vel[i, j] =
                    w * vel[i, j] +
                    c1 * r1[j] * (pbest_pos[i, j] - pos[i, j]) +
                    c2 * r2[j] * (gbest_pos[j] - pos[i, j])

                pos[i, j] += vel[i, j]
                pos[i, j] = clamp(pos[i, j], bounds[1, j], bounds[2, j])
            end

            fval = f(pos[i, :])

            if better(fval, pbest_val[i])
                pbest_val[i] = fval
                pbest_pos[i, :] = pos[i, :]
            end

            if better(fval, gbest_val)
                gbest_val = fval
                gbest_pos = copy(pos[i, :])
            end
        end

        if track
            push!(history, copy(pos))
        end

        # ---------------------------
        # stopping criterion
        # ---------------------------
        f_max = maximum(pbest_val)
        f_min = minimum(pbest_val)

        opt_range = max(f_max - f_min, 1e-12)

        relative_gap =
            mode == :max ?
            (f_max - gbest_val) / opt_range :
            (gbest_val - f_min) / opt_range

        if relative_gap < delta && iter > 0.2 * max_iter
            println("Converged at iteration ", iter)
            break
        end
    end

    return gbest_pos, gbest_val, history
end


# ============================================================
# True model — operates on physical inputs (x1, x2)
# ============================================================
analytical_model(x1, x2) = x1 .+ x2 .+x1 .* x2 .+ 1
analytical_variance(x1, σ) = σ^2*(x1^2 + 2*x1 + 1) + x1^2 + 2*x1 - (x1 + 1)^2 + 1

f(v) = analytical_variance(v[1], v[2])



# ============================================================
# Augmented space definition
#
#   u2  ~ U(0,1)                auxiliary        → inverse CDF surrogate
#   x1  ∈ [-1.0, 1.0]           interval         → sample uniformly over bounds
#   θ_σ ∈ [-1.0,  1.0]          p-box parameter  → sample uniformly over bounds
#
#   Physical x2 is recovered as: x2 = F⁻¹(u2; μ, θ_σ)
#   This is the Rosenblatt transform — makes x2 a deterministic function of (u2, θ_σ)
# ============================================================
const μ_FIXED = 0.0

const x1_UPPER = 1.5
const x1_LOWER = -0.5

const θ_σ_UPPER = 1.5
const θ_σ_LOWER = 0.5

bounds = [
    x1_LOWER   θ_σ_LOWER
    x1_UPPER   θ_σ_UPPER
]

"""
    inverse_cdf_x2(u2, θ_σ)

Recover the physical x2 from the auxiliary uniform u2 and the p-box parameter θ_σ
via the inverse Normal CDF.
"""
inverse_cdf_x2(u2, θ_σ; μ=μ_FIXED) = quantile.(Normal.(μ, θ_σ), u2)


# ============================================================
function mc_augmented(n::Int)
    pts = rand(n, 3)

    u2_raw = pts[:, 1]
    x1_raw = x1_LOWER .+ (x1_UPPER - x1_LOWER) .* pts[:, 2]
    θσ_raw = θ_σ_LOWER .+ (θ_σ_UPPER - θ_σ_LOWER) .* pts[:, 3]

    return x1_raw, u2_raw, θσ_raw
end

# ============================================================
# Build initial design D₀ and evaluate true model
# ============================================================

x_names = [:x1, :u2, :θ_σ]

n_train = 20   # paper uses ~2(d+1)–5(d+1) for d=4 → 10–25 is reasonable

x1_train, u2_train, θ_σ_train = mc_augmented(n_train)

# Recover physical x2 via inverse CDF — this is what the true model sees
x2_train = inverse_cdf_x2(u2_train, θ_σ_train)

# Evaluate true model at physical inputs
y_train = analytical_model(x1_train, x2_train)


# Augmented training DataFrame: GP trains on (x1, u2, θ_σ) — NOT on x2 directly
data_aug_train = DataFrame(
    :x1  => x1_train,
    :u2  => u2_train,
    :θ_σ => θ_σ_train,
    :y   => y_train
)


initial_swarm = hcat(data_aug_train.x1, data_aug_train.θ_σ)

best_x, best_val, history =
    pso_optimize(f, initial_swarm, bounds;
                 max_iter=100,
                 mode=:max)

all_particles = reduce(vcat, history)

using Plots, Distributions, Statistics


# ============================================================
# Grid over epistemic space (x1, θσ)
# ============================================================
n_x1 = 500
n_σ = 500

x1_grid = range(x1_LOWER, x1_UPPER, length=n_x1)
σ_grid  = range(θ_σ_LOWER, θ_σ_UPPER, length=n_σ)

VarSurface  = zeros(n_x1, n_σ)

# ============================================================
# Compute conditional moments
# ============================================================
for (i, x1_v) in enumerate(x1_grid)
    for (j, σ_v) in enumerate(σ_grid)

        Vy = analytical_variance(x1_v, σ_v)

        VarSurface[i, j]  = Vy
    end
end
# ============================================================
# Heatmap
# ============================================================
plt = heatmap(
    x1_grid,
    σ_grid,
    VarSurface,
    xlabel="x1",
    ylabel="θ_σ",
    c=:thermal,
    title="Conditional response variance V_y(x1, σ)",
    colorbar=true
)

# ── overlay the initial training points (x1, θ_σ columns from data_aug) ──────
scatter!(plt,
    data_aug_train.x1, data_aug_train.θ_σ;
    marker = :diamond, color = :cyan, ms = 5,
    label  = "Initial samples", markerstrokewidth=0
)

# ── mark true min and max in epistemic space ──────────────────────────────────
min_idx = argmin(VarSurface)
max_idx = argmax(VarSurface)

scatter!(plt,
    [x1_grid[min_idx[2]]], [σ_grid[min_idx[1]]];
    marker = :circle, color = :red, ms = 5, label = "True min"
)
scatter!(plt,
    [x1_grid[max_idx[2]]], [σ_grid[max_idx[1]]];
    marker = :rect, color = :red, ms = 5, label = "True max"
)

scatter!(plt,
    all_particles[:,1],
    all_particles[:,2];
    label="PSO samples",
    alpha=1.0,
    marker=:cross,
    ms=7,
    color=:green
)

scatter!(plt,
    [best_x[1]],
    [best_x[2]];
    marker=:star5,
    ms=10,
    color=:yellow,
    label="PSO optimum"
)


println("V range: [$(round(minimum(VarSurface), digits=5)),  $(round(maximum(VarSurface), digits=5))]")
println("best val: $(round(best_val, digits=5))")


display(plt)
