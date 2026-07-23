using SurrogateModelling
using UncertaintyQuantification
using UncertaintyQuantification: sample, wrap, isimprecise, middle, minimize, RobustOrthoMADS, bounds, map_to_precise_inputs
using Printf

x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

X = [x1, x2]

physical_model = model_gfunction

# performance function: failure when g ≤ 0, so g_performance = g(x) (negative = failure)
y_star = -1.427
g_performance(df) = df.y .- y_star    # failure when g ≤ y*  ⟺  (g - y*) ≤ 0

# candidates = Int.([1e3, 2.5e3, 5e3, 7.5e3, 1e4, 2.5e4, 5e4, 7.5e4, 1e5, 2.5e5, 5e5])
candidates = Int.([1e3, 2.5e3, 5e3])

for candidate in candidates
    print("at $candidate:\n")
    dl = DoubleLoop(
        MonteCarlo(candidate)
    )

    pf_bounds, res_lb, res_ub = probability_of_failure(
        physical_model,        # your UQModel that writes :y via evaluate!
        g_performance,          # df -> performance values (failure when ≤ 0)
        [x1, x2],
        dl
    )
    @printf("Pf min: %.2e", pf_bounds.lb); println(" @ θ_min: $(round.(res_lb, digits=3))")
    @printf("Pf max: %.2e", pf_bounds.ub); println(" @ θ_min: $(round.(res_ub, digits=3))")

    print("\n\n")

end