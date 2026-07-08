using SurrogateModelling
using UncertaintyQuantification
using UncertaintyQuantification: sample, wrap, isimprecise, middle, minimize, RobustOrthoMADS, bounds, map_to_precise_inputs
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra


# =======================================================================
# Step 1. Augmented Space Setup: Four Gaussian Mixture Function g(x1, x2)
# =======================================================================

# x1 = IntervalVariable(-pi, pi, :x1)
# x2 = IntervalVariable(-pi, pi, :x2)
# x3 = IntervalVariable(-pi, pi, :x3)

# specs = InputSpec.([x1, x2, x3])
# physical_model = model_ishigami

# print("Ishigami...\n")


x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input
physical_model = model_simple
print("Gaussian Mixture...\n")

x_names, w_names, u_names, v_names = spec_names(specs)
y_symbol = physical_model.name


function compare_surrogates_evolution(
    physical_model, 
    specs,
    n_trains
)
    pck_p_max = 2

    kernel_type = GPMatern52
    bases = fill(SurrogateModelling.HermiteBasis(), length(w_names))
    pce_solver = SurrogateModelling.LASSOSolver

    pck_degree = TotalDegree(pck_p_max)

    gp_q2s     = zeros(length(n_trains))
    pck_q2s    = zeros(length(n_trains))

    data_aug_test, _ = build_augmented_design(physical_model, specs, 10000)
    W_test = Matrix(data_aug_test[:, w_names])
    y_test = data_aug_test[:, y_symbol]

    for (idx, n_train) in enumerate(n_trains)
        print("currently at n_train = $n_train ...\n")
        data_aug_train, _ = build_augmented_design(physical_model, specs, n_train)

        gp          = SurrogateModelling.GaussianProcess(data_aug_train, y_symbol;          kernel_type=kernel_type())
        pce_trend   = SurrogateModelling.PolynomialChaosExpansion(data_aug_train, y_symbol, bases, pck_degree; solver=pce_solver())
        pck         = SurrogateModelling.PolynomialChaosKriging(data_aug_train, y_symbol,   pce_trend, kernel_type=kernel_type())

        fit!(gp)
        fit!(pck)

        gp_pred =   predict(gp, W_test)[1]
        pck_pred =  predict(pck, W_test)[1]


        gp_q2 =     q2(y_test, gp_pred)
        pck_q2 =    q2(y_test, pck_pred)

        # gp_q2   = q2_loo(df -> SurrogateModelling.GaussianProcess(df, y_symbol; kernel_type=kernel_type()),                            data_aug_train, y_symbol)
        # pce_q2  = q2_loo(df -> SurrogateModelling.PolynomialChaosExpansion(df, y_symbol, bases, pce_degree; solver=pce_solver()),            data_aug_train, y_symbol)
        # pck_q2 = q2_loo(df -> begin
        #     trend = SurrogateModelling.PolynomialChaosExpansion(df, y_symbol, bases, pck_degree; solver=pce_solver())
        #     SurrogateModelling.PolynomialChaosKriging(df, y_symbol, trend, kernel_type=kernel_type())
        # end, data_aug_train, y_symbol)

        gp_q2s[idx] = gp_q2
        pck_q2s[idx] = pck_q2

    end

    return gp_q2s, pck_q2s
end

n_trains = Int[20 + i for i in 1:2:40]

gp_q2s, pck_q2s = compare_surrogates_evolution(
    physical_model, 
    specs,
    n_trains,
)


gp_q2s .= max.(gp_q2s, -1)
pck_q2s .= max.(pck_q2s, -1)

print("\ndone...\n")

using Plots

p1 = plot(
    n_trains, gp_q2s;
    lw = 2, marker = :circle, ls = :dash,
    label = "GP",
    xlabel = "training samples n₀", ylabel = "Q² LOO", title = "Ishigami - Evolution of Q² with training samples n₀",
    legend = :topleft,
    size = (760, 440),
    ylims = [-1.0, 1.0],
    left_margin = 5Plots.mm, right_margin = 5Plots.mm,
)

plot!(p1,
    n_trains, pck_q2s;
    lw = 2, marker = :circle, ls = :dash,
    label = "PCK",
)

display(p1)

print("gp_q2s: $gp_q2s\n")
print("pck_q2s: $pck_q2s\n")

