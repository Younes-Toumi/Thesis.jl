using Random
using Plots
using UncertaintyQuantification
using UncertaintyQuantification: sample
using SurrogateModelling

# ============================================================
# Objective function (true function you want to optimize)
# ============================================================
f(x) = (6x[1] - 2)^2 * sin(12x[1] - 4) # f(x) = (6x - 2)^2 * sin(12x - 4)

# training data (random observations)
X = RandomVariable.(Uniform(-1, 1), :x)
model = Model(rv -> (6 .* rv.x .- 2) .^2 .* sin.(12 .* rv.x .- 4), :y)

n_train = 20
design_train = LatinHypercubeSampling(n_train)
data_train = sample(X, design_train)
evaluate!(model, data_train)

x_train = data_train[:, :x]
y_train = data_train[:, :y]

bounds = [-1.0, 1.0]

# initial swarm (IMPORTANT: external)
initial_swarm = reshape(x_train, :, 1)
pso_bounds = reshape(bounds, 2, 1)

best_x, best_y, history = pso_optimize(f, initial_swarm, pso_bounds; max_iter=200, mode=:min)

x_true = reshape(range(-1, 1, length=500), :, 1)
y_true = f.(x_true)

p = plot(x_true, y_true, label="true function",
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