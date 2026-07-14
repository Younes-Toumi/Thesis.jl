using Random
using Plots
using Distributions
using UncertaintyQuantification
using UncertaintyQuantification: sample
using SurrogateModelling
using DataFrames

gr()

Random.seed!(42)

# ---------------------------------------------------------
# Model definition
# ---------------------------------------------------------

X = RandomVariable.(Uniform(-1, 1), :x)

model = Model(
    rv -> (6 .* rv.x .- 2).^2 .* sin.(12 .* rv.x .- 4),
    :y
)

n_test = 1001

design_test = LatinHypercubeSampling(n_test)

data_test = sample(X, design_test)

evaluate!(model, data_test)


# ---------------------------------------------------------
# Static Monte Carlo design
# ---------------------------------------------------------

n_static = 15

design_static = MonteCarlo(n_static)

data_static = sample(X, design_static)

evaluate!(model, data_static)


gp_static = GaussianProcess(
    data_static,
    :y;
    kernel_type = GPMatern52()
)

fit!(gp_static)


# Prediction grid
x_plot = reshape(
    collect(range(-1, 1, length=1001)),
    :, 1
)

μ_static, σ_static = predict(gp_static, x_plot)


# ---------------------------------------------------------
# Adaptive design
# ---------------------------------------------------------

n_initial = 5
n_final   = 15

design_adaptive = MonteCarlo(n_initial)

data_adaptive = sample(X, design_adaptive)

evaluate!(model, data_adaptive)


# store initial samples
data_initial = deepcopy(data_adaptive)


while nrow(data_adaptive) < n_final

    gp = GaussianProcess(
        data_adaptive,
        :y;
        kernel_type = GPMatern52()
    )

    fit!(gp)


    μ, σ = predict(gp, x_plot)


    # maximum uncertainty criterion
    idx = argmax(σ)

    x_new = x_plot[idx, 1]


    new_point = DataFrame(
        x = [x_new]
    )

    evaluate!(model, new_point)

    data_adaptive = vcat(
        data_adaptive,
        new_point
    )

end


gp_adaptive = GaussianProcess(
    data_adaptive,
    :y;
    kernel_type = GPMatern52()
)

fit!(gp_adaptive)

μ_adaptive, σ_adaptive = predict(gp_adaptive, x_plot)



# ---------------------------------------------------------
# True analytical solution
# ---------------------------------------------------------

y_true = [
    (6*x - 2)^2 * sin(12*x - 4)
    for x in x_plot[:,1]
]


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

p1 = plot(
    x_plot[:,1],
    y_true,
    lw=2,
    label="Analytical",
    xlabel="x",
    ylabel="y",
    title="Static Monte Carlo (n=15)",
    legend=:topright
)

plot!(
    p1,
    x_plot[:,1],
    μ_static,
    ribbon=2σ_static,
    label="GP mean ± 2σ"
)

scatter!(
    p1,
    data_static.x,
    data_static.y,
    label="Training samples",
    ms=5
)



p2 = plot(
    x_plot[:,1],
    y_true,
    lw=2,
    label="Analytical",
    xlabel="x",
    title="Adaptive design (5 → 15)",
    legend=:topright
)

plot!(
    p2,
    x_plot[:,1],
    μ_adaptive,
    ribbon=2σ_adaptive,
    label="GP mean ± 2σ"
)


# initial points
scatter!(
    p2,
    data_initial.x,
    data_initial.y,
    label="Initial samples",
    marker=:circle,
    ms=5
)


# added adaptive points
added = data_adaptive[n_initial+1:end,:]

scatter!(
    p2,
    added.x,
    added.y,
    label="Adaptive samples",
    marker=:diamond,
    ms=6
)



plot(
    p1,
    p2,
    layout=(1,2),
    size=(1200,450)
)