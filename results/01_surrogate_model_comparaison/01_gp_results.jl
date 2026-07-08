using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra
using KernelFunctions

function compare_kernels_averaged(
    physical_model, 
    specs;
    n_train=50,
    n_runs=15
)

    kernel_array = [GPSquaredExponential, GPMatern12, GPMatern32, GPMatern52]
    kernel_names = ["SE", "Matérn-1/2", "Matérn-3/2", "Matérn-5/2"]

    Q2    = zeros(n_runs, length(kernel_array))
    times = zeros(n_runs, length(kernel_array))

    y_symbol = physical_model.name


    for r in 1:n_runs
        print("currently at r = $r ...\n")
        # Random.seed!(1000 + r)   # different design each run, but reproducible overall
        data_aug_train, _ = build_augmented_design(physical_model, specs, n_train)

        for (k, kernel) in enumerate(kernel_array)   # SAME data, all kernels — paired
            gp = GaussianProcess(data_aug_train, y_symbol; kernel_type=kernel())
            fit_result = @timed fit!(gp)

            # Q2[r, k] = q2_loo(df -> GaussianProcess(df, y_symbol; kernel_type=gp.kernel_type), data_aug_train, y_symbol)
            Q2[r, k] = q2_loo_gp_fast(gp)

            times[r, k] = fit_result.time

        end
    end

    println(rpad("Kernel", 12), rpad("Q² mean", 12), rpad("Q² std", 12), rpad("Q² min", 12), rpad("Q² max", 12), rpad("Q² neg", 12))
    for (k, name) in enumerate(kernel_names)
        col = Q2[:, k]
        println(rpad(name, 12),
                rpad(round(mean(col),       digits=4), 12),
                rpad(round(std(col),        digits=4), 12),
                rpad(round(minimum(col),    digits=4), 12),
                rpad(round(maximum(col),    digits=4), 12),
                rpad(sum(col .<= 0)                  , 12),
                )
    end

    print("\n")
    println(rpad("Kernel", 12), rpad("t mean", 12), rpad("t std", 12), rpad("t min", 12), rpad("t max", 12))
    for (k, name) in enumerate(kernel_names)
        col = times[:, k]
        println(rpad(name, 12),
                rpad(round(mean(col),       digits=4), 12),
                rpad(round(std(col),        digits=4), 12),
                rpad(round(minimum(col),    digits=4), 12),
                rpad(round(maximum(col),    digits=4), 12)

        )
    end

    print("\nLatex Array\n")

    kernel_names_latex = ["SE", "Matérn-\$1/2\$", "Matérn-\$3/2\$", "Matérn-\$5/2\$"]


    println(
        rpad("Kernel", 12), rpad(" & ", 3), 
        rpad("Q² mean", 3), rpad(" & ", 3),
        rpad("Q² min", 3),  rpad(" & ", 3),
        rpad("Q² max", 3),  rpad(" & ", 3),
        rpad("Q² std", 3),  rpad(" & ", 3),
        rpad("t mean", 3),  rpad(" & ", 3),
        rpad("t min", 3),   rpad(" & ", 3),
        rpad("t max", 3),   rpad(" & ", 3),
        rpad("t std", 3),   rpad(" \\\\", 3), "\n"
        
        )


    for k in eachindex(kernel_names_latex)
        col_q2 = Q2[:, k]
        col_time = times[:, k]
        col_name = kernel_names_latex[k]

        println(rpad(col_name, 12),                             rpad(" & ", 3),
                rpad("\$" * round(mean(col_q2),        digits=2)* "\$", 3),  rpad(" & ", 3),
                rpad("\$" * round(minimum(col_q2),     digits=2)* "\$", 3),  rpad(" & ", 3),
                rpad("\$" * round(maximum(col_q2),     digits=2)* "\$", 3),  rpad(" & ", 3),
                rpad("\$" * round(std(col_q2),         digits=2)* "\$", 3),  rpad(" & ", 3),
                rpad("\$" * round(mean(col_time),      digits=2)* "\$", 3),  rpad(" & ", 3),
                rpad("\$" * round(minimum(col_time),   digits=2)* "\$", 3),  rpad(" & ", 3),
                rpad("\$" * round(maximum(col_time),   digits=2)* "\$", 3),  rpad(" & ", 3),
                rpad("\$" * round(std(col_time),       digits=2)* "\$", 3),  rpad(" \\\\", 3),
        )
    end

    return Q2, times
end

n_train = 50
n_runs = 10
# ============================================================
# 1. Four Gaussian Mixture Function g(x1, x2)
# ============================================================
x1_g = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2_g = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

specs_g = InputSpec.([x1_g, x2_g])     # broadcasts dispatch over each UQ.jl input
 
print("# ============================================================\n")
print("# 1. Four Gaussian Mixture Function g(x1, x2) n₀ = $n_train\n")
print("# ============================================================\n\n")

Q2_g, times_g = compare_kernels_averaged(
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

Q2_f, times_f = compare_kernels_averaged(
    model_gfunction, 
    specs_f;
    n_train=n_train,
    n_runs=n_runs
)

print("\ndone...\n")

using StatsPlots

function plot_results(qoi, n_train, type, func_name)
    kernel_names = ["SqExp", "Matern12", "Matern32", "Matern52"]
    n_runs, n_k = size(qoi)
    # repeat each kernel name down its column, stack all columns into one long vector
    groups = repeat(kernel_names, inner=n_runs)

    values = vec(qoi)                       # column-major: all SqExp, then all Matern12, ...
    if type == :q2
        title = "Q² LOO - $func_name,  n₀=$n_train"
        p = boxplot(groups, values;
            xlabel = "",    
            ylabel = "Q² (LOO)",
            ylims = [0, 1],
            title  = title, legend = false)
    end
    if type == :time
        title = "t [s] - $func_name, n₀=$n_train"
        p = boxplot(groups, values;
            xlabel = "",        
            ylabel = "t [s]",
            # ylims = (0, 4),
            title  = title, legend = false)
    end

    return p
end

p1 = plot_results(Q2_g, n_train, :q2, "Gaussian Mixture")
p2 = plot_results(times_g, n_train, :time, "Gaussian Mixture")
p3 = plot_results(Q2_f, n_train, :q2, "Ishigami")
p4 = plot_results(times_f, n_train, :time, "Ishigami")


p = plot(
    p1, p2,
    p3, p4,
    layout = (2, 2),
    size = (1300, 900),
    margin=5Plots.mm,
    guidefontsize=14,
    tickfontsize=14,
    titlefontsize=14
);

display(p)