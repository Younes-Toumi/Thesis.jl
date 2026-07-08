using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra
using KernelFunctions


function compare_pce_variants_averaged(
    physical_model, 
    specs;
    n_train=50,
    n_runs=10
)

    _, w_names, _, _ = spec_names(specs)
    p_max = 15

    degrees = [
        TotalDegree(p_max), 
        TotalDegree(p_max), 
        QBall(p_max, 0.75),
        QBall(p_max, 0.50),
        QBall(p_max, 0.25),
    ]
    
    solvers = [
        SurrogateModelling.OLSSolver,
        SurrogateModelling.LASSOSolver, 
        SurrogateModelling.LASSOSolver, 
        SurrogateModelling.LASSOSolver, 
        SurrogateModelling.LASSOSolver, 
    ]
        
    variant_names = ["OLS (TD)", "LASSO (TD)", "LASSO (QB-0.75)", "LASSO (QB-0.50)", "LASSO (QB-0.25)"]

    Q2          = zeros(n_runs, length(variant_names))
    times       = zeros(n_runs, length(variant_names))
    n_coeffs    = zeros(n_runs, length(variant_names))

    y_symbol = physical_model.name

    data_aug_test, _ = build_augmented_design(physical_model, specs, 1001)
    W_test = Matrix(data_aug_test[:, w_names])
    y_test = data_aug_test[:, y_symbol]

    for r in 1:n_runs
        print("currently at r = $r ...\n")
        data_aug_train, _ = build_augmented_design(physical_model, specs, n_train)

        for i in eachindex(variant_names)
            bases = [SurrogateModelling.HermiteBasis() for _ in w_names]

            pce = SurrogateModelling.PolynomialChaosExpansion(data_aug_train, y_symbol, bases, degrees[i]; solver=solvers[i]())
            
            fit_result = @timed fit!(pce)
            y_pred = predict(pce, W_test)
            Q2[r, i] = q2(y_test, y_pred)

            # Q2[r, i] = q2_loo(df -> SurrogateModelling.PolynomialChaosExpansion(df, y_symbol, bases, degrees[i]; solver=solvers[i]()), data_aug_train, y_symbol)
            times[r, i] = fit_result.time
            n_coeffs[r, i] = Int(count(!iszero, pce.coeffs))

        end
    end

    println(rpad("Variant", 12), rpad("Q² mean", 12), rpad("Q² std", 12), rpad("Q² min", 12), rpad("Q² max", 12), rpad("Q² neg", 12))
    for (i, name) in enumerate(variant_names)
        col = Q2[:, i]
        println(rpad(name, 12),
                rpad(round(mean(col),       digits=4), 12),
                rpad(round(std(col),        digits=4), 12),
                rpad(round(minimum(col),    digits=4), 12),
                rpad(round(maximum(col),    digits=4), 12),
                rpad(sum(col .<= 0)                  , 12),
                )
    end

    print("\n")
    println(rpad("Variants", 12), rpad("t mean", 12), rpad("t std", 12), rpad("t min", 12), rpad("t max", 12))
    for (i, name) in enumerate(variant_names)
        col = times[:, i]
        println(rpad(name, 12),
                rpad(round(mean(col),       digits=4), 12),
                rpad(round(std(col),        digits=4), 12),
                rpad(round(minimum(col),    digits=4), 12),
                rpad(round(maximum(col),    digits=4), 12)

        )
    end

    print("\n")
    println(rpad("Variants", 12), rpad("n_coeffs", 12))
    for (i, name) in enumerate(variant_names)
        col = n_coeffs[:, i]
        println(rpad(name, 12),
                rpad(round(mean(col),       digits=2), 12)
        )
    end

    print("\nLatex Array\n")

    variant_names_latex = ["OLS (TD)", "LASSO (TD)", "LASSO (QB-0.75)", "LASSO (QB-0.50)", "LASSO (QB-0.25)"]


    println(
        rpad("Variant", 12), rpad(" & ", 3), 
        rpad("Q² mean", 3), rpad(" & ", 3),
        rpad("Q² min", 3),  rpad(" & ", 3),
        rpad("Q² max", 3),  rpad(" & ", 3),
        rpad("Q² std", 3),  rpad(" & ", 3),
        rpad("t mean", 3),  rpad(" & ", 3),
        rpad("t min", 3),   rpad(" & ", 3),
        rpad("t max", 3),   rpad(" & ", 3),
        rpad("t std", 3),   rpad(" &", 3),
        rpad("coeff", 3),   rpad(" \\\\", 3),
        )


    for i in eachindex(variant_names_latex)
        col_q2 = Q2[:, i]
        col_time = times[:, i]
        col_coeff = n_coeffs[:, i]

        col_name = variant_names_latex[i]


        println(rpad(col_name, 12),                             rpad(" & ", 3),
                rpad("\$" * "$(round(mean(col_q2),        digits=2))" * "\$",  3),  rpad(" & ", 3),
                rpad("\$" * "$(round(minimum(col_q2),     digits=2))" * "\$",  3),  rpad(" & ", 3),
                rpad("\$" * "$(round(maximum(col_q2),     digits=2))" * "\$",  3),  rpad(" & ", 3),
                rpad("\$" * "$(round(std(col_q2),         digits=2))" * "\$",  3),  rpad(" & ", 3),
                rpad("\$" * "$(round(mean(col_time),      digits=2))" * "\$",  3),  rpad(" & ", 3),
                rpad("\$" * "$(round(minimum(col_time),   digits=2))" * "\$",  3),  rpad(" & ", 3),
                rpad("\$" * "$(round(maximum(col_time),   digits=2))" * "\$",  3),  rpad(" & ", 3),
                rpad("\$" * "$(round(std(col_time),       digits=2))" * "\$",  3),  rpad(" & ", 3),
                rpad("\$" * "$(round(mean(col_coeff),     digits=2))" * "\$", 3),  rpad(" \\\\", 3),

        )
    end

    return Q2, times
end

n_train = 500
n_runs = 3

# ============================================================
# 1. Four Gaussian Mixture Function g(x1, x2)
# ============================================================
x1_g = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2_g = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

specs_g = InputSpec.([x1_g, x2_g])     # broadcasts dispatch over each UQ.jl input
 
print("# ============================================================\n")
print("# 1. Four Gaussian Mixture Function g(x1, x2) n₀ = $n_train\n")
print("# ============================================================\n\n")

Q2_g, times_g = compare_pce_variants_averaged(
    model_gfunction, 
    specs_g;
    n_train=n_train,
    n_runs=n_runs
)


# ============================================================
# 2. Ishigami Function f(x1, x2, x3)
# ============================================================
x1_f = IntervalVariable(-pi, pi, :x1)
x2_f = IntervalVariable(-pi, pi, :x2)
x3_f = IntervalVariable(-pi, pi, :x3)

specs_f = InputSpec.([x1_f, x2_f, x3_f])     # broadcasts dispatch over each UQ.jl input


print("\n# ============================================================\n")
print("# 2. Ishigami Function f(x1, x2, x3) with n₀ = $n_train\n")
print("# ============================================================\n\n")

Q2_f, times_f = compare_pce_variants_averaged(
    model_ishigami, 
    specs_f;
    n_train=n_train,
    n_runs=n_runs
)

print("\ndone...\n")

using StatsPlots

function plot_results(qoi, n_train, type, func_name)

    variant_names = ["OLS (TD)", "LASSO (TD)", "LASSO (Q.75)", "LASSO (Q.50)", "LASSO (Q.25)"]
    n_runs, n_k = size(qoi)
    # repeat each kernel name down its column, stack all columns into one long vector

    groups = repeat(variant_names, inner=n_runs)

    values = vec(qoi)                       # column-major: all SqExp, then all Matern12, ...

    if type == :q2
        title = "Q² - $func_name,  n₀=$n_train"
        p = boxplot(groups, values;
            ylabel = "Q²", ylims=[-1, 1],
            xrotation = 45,
            title  = title, legend = false)
    end
    if type == :time
        title = "t [s] - $func_name, n₀=$n_train"
        p = boxplot(groups, values;
            ylabel = "t [s]",
            xrotation = 45,
            title  = title, legend = false)
    end

    return p
end

p1 = plot_results(Q2_g, n_train, :q2, "Gaussian Mixture")
p2 = plot_results(times_g, n_train, :time, "Gaussian Mixture")
p3 = plot_results(Q2_f, n_train, :q2, "Ishigami")
p4 = plot_results(times_f, n_train, :time, "Ishigami")


p = plot(
    p1, p3,
    layout = (1, 2),
    size = (1400, 600),
    bottommargin=15Plots.mm,
    leftmargin=7Plots.mm,
    guidefontsize=14,
    tickfontsize=14,
    titlefontsize=14
);

pp = plot(
    p2, p4,
    layout = (1, 2),
    size = (1400, 600),
    bottommargin=15Plots.mm,
    leftmargin=7Plots.mm,
    guidefontsize=14,
    tickfontsize=14,
    titlefontsize=14
);

display(p)

display(pp)