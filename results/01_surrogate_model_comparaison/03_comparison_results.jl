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

# x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
# x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

# specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input
# physical_model = model_gfunction
# print("Gaussian Mixture...\n")

x1 = IntervalVariable(-pi, pi, :x1)
x2 = IntervalVariable(-pi, pi, :x2)
x3 = IntervalVariable(-pi, pi, :x3)


specs = InputSpec.([x1, x2, x3])     # broadcasts dispatch over each UQ.jl input
physical_model = model_ishigami
print("Ishigami...\n")


x_names, w_names, u_names, v_names = spec_names(specs)
y_symbol = physical_model.name

# # =======================================================================
# # Step 2. Training the GP, PCE, PCK
# # =======================================================================

# n_samples = 10
# data_aug_train, _ = build_augmented_design(physical_model, specs, n_samples)

# n_pool = 1_000_000
# data_aug_pool, _ = build_augmented_design(physical_model, specs, n_pool)



# pce_p_max = 7
# pck_p_max = 3

# kernel_type = GPMatern52
# bases = fill(SurrogateModelling.HermiteBasis(), length(w_names))
# pce_solver = SurrogateModelling.LASSOSolver

# pce_degree = QBall(pce_p_max, 0.5)
# pck_degree = QBall(pck_p_max, 0.5)

# gp          = SurrogateModelling.GaussianProcess(data_aug_train, y_symbol;          kernel_type=kernel_type())
# pce         = SurrogateModelling.PolynomialChaosExpansion(data_aug_train, y_symbol, bases, pce_degree; solver=pce_solver())
# pce_trend   = SurrogateModelling.PolynomialChaosExpansion(data_aug_train, y_symbol, bases, pck_degree; solver=pce_solver())
# pck         = SurrogateModelling.PolynomialChaosKriging(data_aug_train, y_symbol,   pce_trend, kernel_type=kernel_type())

# gp_fit_value,  gp_fit_time,  _... = @timed fit!(gp)
# pce_fit_value, pce_fit_time, _... = @timed fit!(pce)
# pck_fit_value, pck_fit_time, _... = @timed fit!(pck)

# print("\n")

# q2_gp   = q2_loo(df -> SurrogateModelling.GaussianProcess(df, y_symbol; kernel_type=kernel_type()),                            data_aug_train, y_symbol)
# print("q2 gp done...\n")


# q2_pce  = q2_loo(df -> SurrogateModelling.PolynomialChaosExpansion(df, y_symbol, bases, pce_degree; solver=pce_solver()),            data_aug_train, y_symbol)
# print("q2 pce done...\n")


# q2_pck  = q2_loo(df -> SurrogateModelling.PolynomialChaosKriging(df, y_symbol, pce_trend, kernel_type=kernel_type()),     data_aug_train, y_symbol)

# print("\n")

# print("n₀: $n_samples:\n")
# print("Q² GP:  $(round(q2_gp, digits=3))\n")
# print("Q² PCE: $(round(q2_pce, digits=3))\n")
# print("Q² PCK: $(round(q2_pck, digits=3))\n")


# gp_μ_pool,  gp_time_pool,  _...  = @timed predict(gp, Matrix(data_aug_pool[:, w_names]))
# pce_μ_pool, pce_time_pool, _...  = @timed predict(pce, Matrix(data_aug_pool[:, w_names]))
# pck_μ_pool, pck_time_pool, _...  = @timed predict(pck, Matrix(data_aug_pool[:, w_names]))



function compare_surrogates_averaged(
    physical_model, 
    specs;
    n_train=50,
    n_runs=10
)

    n_pool = 1_000_000
    data_aug_pool, _ = build_augmented_design(physical_model, specs, n_pool)

    pck_p_max = 3

    kernel_type = GPMatern52
    bases = fill(SurrogateModelling.HermiteBasis(), length(w_names))
    pce_solver = SurrogateModelling.LASSOSolver

    pck_degree = QBall(pck_p_max, 0.5)


    gp_q2s     = zeros(n_runs)
    pck_q2s    = zeros(n_runs)


    gp_pool_times     = zeros(n_runs)
    pck_pool_times    = zeros(n_runs)


    gp_fit_times     = zeros(n_runs)
    pck_fit_times    = zeros(n_runs)

    for r in 1:n_runs
        print("currently at r = $r ...\n")
        data_aug_train, _ = build_augmented_design(physical_model, specs, n_train)

        gp          = SurrogateModelling.GaussianProcess(data_aug_train, y_symbol;          kernel_type=kernel_type())
        pce_trend   = SurrogateModelling.PolynomialChaosExpansion(data_aug_train, y_symbol, bases, pck_degree; solver=pce_solver())
        pck         = SurrogateModelling.PolynomialChaosKriging(data_aug_train, y_symbol,   pce_trend, kernel_type=kernel_type())

        gp_fit_value,  gp_fit_time,  _... = @timed fit!(gp)
        pck_fit_value, pck_fit_time, _... = @timed fit!(pck)

        gp_q2   = q2_loo(df -> SurrogateModelling.GaussianProcess(df, y_symbol; kernel_type=kernel_type()),                            data_aug_train, y_symbol)
        pck_q2 = q2_loo(df ->
            begin
                trend = SurrogateModelling.PolynomialChaosExpansion(df, y_symbol, bases, pck_degree; solver=pce_solver())
                SurrogateModelling.PolynomialChaosKriging(df, y_symbol, trend; kernel_type=kernel_type())
            end, data_aug_train, y_symbol)

        gp_μ_pool,  gp_pool_time,  _...  = @timed predict(gp, Matrix(data_aug_pool[:, w_names]))
        pck_μ_pool, pck_pool_time, _...  = @timed predict(pck, Matrix(data_aug_pool[:, w_names]))

        gp_q2s[r] = gp_q2
        pck_q2s[r] = pck_q2

        gp_fit_times[r] = gp_fit_time
        pck_fit_times[r] = pck_fit_time

        gp_pool_times[r] = gp_pool_time
        pck_pool_times[r] = pck_pool_time

    end


    print("\nLatex Array\n")

    println(
        rpad("Surrogate", 12), rpad(" & ", 3), 
        rpad("Q²_mean", 3), rpad(" & ", 3),
        rpad("Q²_std", 3), rpad(" & ", 3),

        rpad("t_trn_mean", 3),  rpad(" & ", 3),
        rpad("t_trn_std", 3),  rpad(" & ", 3),

        rpad("t_ifr_mean", 3),  rpad(" & ", 3),
        rpad("t_ifr_std", 3),  rpad(" \\\\", 3), "\n"
    )

    println(rpad("\\gls{gp}", 12),                                               rpad(" & ", 3),
            rpad("\$" * "$(round(mean(gp_q2s),          digits=2))" * "\$",  3), rpad(" & ", 3),
            rpad("\$" * "$(round(std(gp_q2s),           digits=2))" * "\$",  3), rpad(" & ", 3),
            rpad("\$" * "$(round(mean(gp_fit_times),   digits=2))" * "\$",  3), rpad(" & ", 3),
            rpad("\$" * "$(round(std(gp_fit_times),    digits=2))" * "\$",  3), rpad(" & ", 3),
            rpad("\$" * "$(round(mean(gp_pool_times),   digits=2))" * "\$",  3), rpad(" & ", 3),
            rpad("\$" * "$(round(std(gp_pool_times),    digits=2))" * "\$",  3), rpad(" \\\\", 3),
    )


    println(rpad("\\gls{pck}", 12),                                               rpad(" & ", 3),
            rpad("\$" * "$(round(mean(pck_q2s),          digits=2))" * "\$",  3), rpad(" & ", 3),
            rpad("\$" * "$(round(std(pck_q2s),           digits=2))" * "\$",  3), rpad(" & ", 3),
            rpad("\$" * "$(round(mean(pck_fit_times),   digits=2))" * "\$",  3), rpad(" & ", 3),
            rpad("\$" * "$(round(std(pck_fit_times),    digits=2))" * "\$",  3), rpad(" & ", 3),
            rpad("\$" * "$(round(mean(pck_pool_times),   digits=2))" * "\$",  3), rpad(" & ", 3),
            rpad("\$" * "$(round(std(pck_pool_times),    digits=2))" * "\$",  3), rpad(" \\\\", 3),
    )

    return gp_q2s, pck_q2s, gp_fit_times, pck_fit_times, gp_pool_times, pck_pool_times

end

n_train = 50
n_runs = 10

gp_q2s, pck_q2s, gp_fit_times, pck_fit_times, gp_pool_times, pck_pool_times = compare_surrogates_averaged(
    physical_model, 
    specs;
    n_train=n_train,
    n_runs=n_runs
)



print("\ndone...\n")

using StatsPlots

function plot_results(qoi, n_train, n_runs, type, func_name)

    surrogate_names = ["GP", "PCK"]
    groups = repeat(surrogate_names, inner=n_runs)
    values = vec(qoi)                       # column-major: all SqExp, then all Matern12, ...

    if type == :q2
        title = "$func_name,  n₀: $n_train"
        p = boxplot(groups, values;
            ylabel = "Q² (LOO)",
            title  = title, legend = false)
    end
    if type == :time_train
        title = "$func_name, n₀: $n_train"
        p = boxplot(groups, values;
            ylabel = "t_train [s]",
            title  = title, legend = false)
    end

    if type == :time_pool
        title = "$func_name, n_ifr.= 10⁶"
        p = boxplot(groups, values;
            ylabel = "t_infer. [s]",
            title  = title, legend = false)
    end

    return p
end

p1 = plot_results([gp_q2s; pck_q2s], n_train, n_runs, :q2, "Ishigami")
p2 = plot_results([gp_fit_times; pck_fit_times], n_train, n_runs, :time_train, "Ishigami")
p3 = plot_results([gp_pool_times; pck_pool_times], n_train, n_runs, :time_pool, "Ishigami")

plot!(p1, ylims=[-1.0, 1.0])

p = plot(
    p1, p2, p3,
    layout = (1, 3),
    size = (1300, 500),
    margin=7Plots.mm,
    guidefontsize=14,
    tickfontsize=14,
    titlefontsize=14
);

display(p)