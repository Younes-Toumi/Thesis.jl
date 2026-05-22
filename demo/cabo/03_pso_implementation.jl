using Random
using Plots

# ============================================================
# Objective function (true function you want to optimize)
# ============================================================
f(x) = (6x - 2)^2 * sin(12x - 4)

# ============================================================
# PSO (clean 1D version)
# ============================================================
function pso_optimize(f::Function,
                      initial_particles::Vector{Float64};
                      lb::Float64 = 0.0,
                      ub::Float64 = 1.0,
                      max_iter::Int = 100,
                      w::Float64 = 0.8,
                      c1::Float64 = 1.5,
                      c2::Float64 = 1.5,
                      mode::Symbol = :max,
                      delta::Float64 = 1e-3,
                      track::Bool = true)

    n_particles = length(initial_particles)

    pos = copy(initial_particles)
    vel = zeros(n_particles)

    fvals = [f(pos[i]) for i in 1:n_particles]

    pbest_pos = copy(pos)
    pbest_val = copy(fvals)

    best_idx = argmax(pbest_val)
    gbest_pos = pbest_pos[best_idx]
    gbest_val = pbest_val[best_idx]

    # storage for visualization
    history = track ? Vector{Vector{Float64}}() : nothing

    # objective direction
    better(a, b) = mode == :max ? (a > b) : (a < b)

    for iter in 1:max_iter
        for i in 1:n_particles

            r1 = rand()
            r2 = rand()

            vel[i] =
                w * vel[i] +
                c1 * r1 * (pbest_pos[i] - pos[i]) +
                c2 * r2 * (gbest_pos - pos[i])

            pos[i] += vel[i]
            pos[i] = clamp(pos[i], lb, ub)

            fval = f(pos[i])

            if better(fval, pbest_val[i])
                pbest_val[i] = fval
                pbest_pos[i] = pos[i]
            end

            if better(fval, gbest_val)
                gbest_val = fval
                gbest_pos = pos[i]
            end
        end

        if track
            push!(history, copy(pos))
        end

        # stopping criterion
        f_max = maximum(pbest_val)
        f_min = minimum(pbest_val)

        opt_range = f_max - f_min + 1e-12

        relative_gap =
            mode == :max ? (f_max - gbest_val) / opt_range : (gbest_val - f_min) / opt_range

        if relative_gap < delta && iter > Int(0.10*max_iter) # we want to have at least 10% of iterations in case of plateaux
            println("Converged at iteration ", iter)
            break
        end



    end

    return gbest_pos, gbest_val, history
end

# training data (random observations)
X = RandomVariable.(Uniform(0, 1.5), :x)
model = Model(rv -> (6 .* rv.x .- 2) .^2 .* sin.(12 .* rv.x .- 4), :y)

n_train = 15
design_train = MonteCarlo(n_train)
data_train = sample(X, design_train)
evaluate!(model, data_train)

x_train = data_train[:, :x]
y_train = data_train[:, :y]

# initial swarm (IMPORTANT: external)
initial_swarm = x_train

best_x, best_y, history =
    pso_optimize(f, initial_swarm;
                 lb=0.0, ub=1.5,
                 max_iter=100,
                 mode=:min)

xs = range(0, 1.5, length=500)
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