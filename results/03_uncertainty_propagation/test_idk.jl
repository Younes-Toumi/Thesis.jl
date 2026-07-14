using SurrogateModelling
using UncertaintyQuantification
using UncertaintyQuantification: sample, wrap, isimprecise, middle, minimize, RobustOrthoMADS, bounds, map_to_precise_inputs

x1_test = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2_test = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

X_test = [x1_test, x2_test]

physical_model_test = model_gfunction

# performance function: failure when g ≤ 0, so g_performance = g(x) (negative = failure)
y_star = -1.427
g_performance(df) = df.y .- y_star    # failure when g ≤ y*  ⟺  (g - y*) ≤ 0

# heavier, rare-event-capable sim for the lower bound; light MC for the upper
# heavier, rare-event-capable sim for the lower bound; light MC for the upper

candidates = Int.([1e3, 2.5e3, 5e3, 7.5e3, 1e4, 2.5e4, 5e4, 7.5e4, 1e5, 2.5e5, 5e5])

for candidate in candidates
    print("at $candidate:\n")
    dl = DoubleLoop(
        MonteCarlo(candidate)
    )

    pf_bounds_jasp, res_lb_jasp, res_ub_jasp = probability_of_failure(
        physical_model,        # your UQModel that writes :y via evaluate!
        g_performance,          # df -> performance values (failure when ≤ 0)
        [x1_test, x2_test],
        dl
    )

    println("Pf bounds: ", pf_bounds_jasp)                      # Interval(pf_lb, pf_ub)
    println("θ at Pf_min: ", res_lb_jasp, "   θ at Pf_max: ", res_ub_jasp)

    print("\n\n")

end


using DataFrames

results = DataFrame(
    budget = [
        1_000,
        2_500,
        5_000,
        7_500,
        10_000,
        25_000,
        50_000,
        100_000,
        250_000,
        500_000
    ],

    q_min = [
        5.430229496309763e-13,
        7.059298345202693e-13,
        6.516275395571715e-13,
        7.602321294833669e-13,
        5.701740971125252e-13,
        7.439414409944375e-13,
        7.846681622167609e-13,
        7.385112114981279e-13,
        7.732646802745103e-13,
        7.588745721092895e-13
    ],

    q_max = [
        0.08799999999999998,
        0.0928,
        0.0788,
        0.07586666666666667,
        0.08129999999999998,
        0.07708,
        0.07582,
        0.07491,
        0.073756,
        0.073434
    ],
)

function plot_convergence(results, ylabel_str, title_str)
    p = plot(
        results.budget, results.q_min;
        xscale = :log10, lw = 2, marker = :circle,
        label = "lower bound",
        xlabel = "Total budget", ylabel = ylabel_str, title = title_str,
        legend = :topleft,
        size = (760, 440),
        left_margin = 5Plots.mm, right_margin = 5Plots.mm,
    )
    plot!(p, results.budget, results.q_max;
          lw = 2, marker = :circle, label = "upper bound")
    hline!(p, [results.q_min[end]]; ls = :dash, c = :gray, label = "final lower")
    hline!(p, [results.q_max[end]]; ls = :dash, c = :black, label = "final upper")
    return p
end

p = plot_convergence(
    results,
    "Failure probability",
    "Convergence of failure probability bounds"
)

display(p)