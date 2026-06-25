using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra

Random.seed!(42)

# ============================================================
# 0. Forrester Function f(x1)
# ============================================================
# x = IntervalVariable(0, 1, :x1)
# specs = InputSpec.([x])     # broadcasts dispatch over each UQ.jl input
# physical_model = model_forrester
# print("Forrester...\n")


# ============================================================
# 1. Four Gaussian Mixture Function g(x1, x2)
# ============================================================
# x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
# x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

# specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input
# physical_model = model_gfunction
# print("Gaussian Mixture...\n")

# ============================================================
# 2. Ishigami Function f(x1, x2, x3)
# ============================================================
x1 = IntervalVariable(-pi, pi, :x1)
x2 = IntervalVariable(-pi, pi, :x2)
x3 = IntervalVariable(-pi, pi, :x3)

specs = InputSpec.([x1, x2, x3])     # broadcasts dispatch over each UQ.jl input
physical_model = model_ishigami
print("Ishigami...\n")


x_names, w_names, u_names, v_names = spec_names(specs)
y_symbol = physical_model.name

n_reff = 100
data_aug_train_reff, _ = build_augmented_design(physical_model, specs, n_reff)

gp_reff = GaussianProcess(data_aug_train_reff, y_symbol; kernel_type=GPMatern52())
fit!(gp_reff)

# q2_val_reff = q2_loo(df -> GaussianProcess(df, y_symbol; kernel_type=gp_reff.kernel_type), data_aug_train_reff, y_symbol)
q2_val_reff = q2_loo_gp_fast(gp_reff)

print("n₀: $n_reff without adaptive sampling:\n")
print("Q²: $(round(q2_val_reff, digits=3))\n")


# ~ ~ ~ ~ ~ ~ adaptive part ~ ~ ~ ~ ~ ~ #
n_init = 20

data_aug_train_adpt, _ = build_augmented_design(physical_model, specs, n_init)
y_symbol = physical_model.name

gp_adpt = GaussianProcess(data_aug_train_adpt, y_symbol; kernel_type=GPMatern52())
fit!(gp_adpt)

adaptive_result =  adaptive_sampling(
    physical_model,
    specs,
    gp_adpt,
    data_aug_train_adpt;
    n_budget = n_reff - n_init
);

print("done ...")


using Plots

# User-defined reference values
q2_threshold = 0.99
q2_n50 = q2_val_reff

# Extract history
n_samples = adaptive_result.n_history
q2_values = clamp.(adaptive_result.q2_history, -1.0, 1.0)

p1 = plot(
    n_samples,
    q2_values;
    marker = :circle,
    color = :black,
    linewidth = 2,
    linestyle = :dash,
    xlabel = "Number of samples",
    ylabel = "Q²",
    ylim = (-1, 1),
    label = "Adaptive sampling",
    legend = :bottomright,
)

hline!(
    [q2_threshold];
    color = :red,
    linestyle = :dash,
    linewidth = 2,
    label = "Q² threshold"
)

hline!(
    [q2_n50];
    color = :blue,
    linestyle = :dashdot,
    linewidth = 2,
    label = "Q² with n₀ = $n_reff"
)

display(p1)