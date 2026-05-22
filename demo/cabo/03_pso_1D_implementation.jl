using Random
using Plots
using UncertaintyQuantification

# ============================================================
# Objective function (true function you want to optimize)
# ============================================================
f(x) = (6x[1] - 2)^2 * sin(12x[1] - 4)
# f(x) = (6x - 2)^2 * sin(12x - 4)

# ============================================================
# PSO (clean 1D version)
# ============================================================
function pso_optimize(f::Function,
                      initial_particles::Matrix{Float64},
                      bounds::Matrix{Float64};
                      max_iter::Int = 200,
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

# training data (random observations)
X = RandomVariable.(Uniform(-1, 1), :x)
model = Model(rv -> (6 .* rv.x .- 2) .^2 .* sin.(12 .* rv.x .- 4), :y)

n_train = 20
design_train = LatinHypercubeSampling(n_train)
data_train = sample(X, design_train)
evaluate!(model, data_train)

x_train = data_train[:, :x]
y_train = data_train[:, :y]

# initial swarm (IMPORTANT: external)
initial_swarm = reshape(x_train, :, 1)
bounds = reshape([-1.0, 1.0], 2, 1)

best_x, best_y, history =
    pso_optimize(f, initial_swarm, bounds;
                 max_iter=200,
                 mode=:min)

xs = reshape(range(-1, 1, length=500), :, 1)
ys = f.(xs)

p = plot(xs, ys,
    label="true function",
    lw=2,
    size=(900, 600))

# training data (only one label)
scatter!(p, x_train, y_train,
    label="training data",
    color=:red)

# PSO trajectory (only one label total)
all_particles = reduce(vcat, history)
scatter!(p, all_particles,
         f.(all_particles),
         label="PSO samples",
         alpha=0.1)

# final optimum (only one label)
scatter!(p, [best_x], [best_y],
    label="optimum",
    ms=8,
    color=:green)

title!("PSO optimization result")
xlabel!("x")
ylabel!("f(x)")
display(p)