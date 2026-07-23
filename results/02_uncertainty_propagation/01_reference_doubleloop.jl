using SurrogateModelling
using UncertaintyQuantification
using UncertaintyQuantification: sample, wrap, isimprecise, middle, minimize, RobustOrthoMADS, bounds, map_to_precise_inputs
using Statistics
using DataFrames
using Plots

x1_g = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2_g = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

X_g = [x1_g, x2_g]

physical_model = model_gfunction

relative_change(new, old) = abs(new - old) / max(abs(old), eps())

# ---------------------------------------------------------
# Double-loop reference (single QoI per call — unchanged logic)
# ---------------------------------------------------------
function true_doubleloop(
    model, imprecise_inputs;
    n_total::Int, k, qoi::Symbol, y_star::Float64,
)
    inputs = wrap(imprecise_inputs)
    imp  = filter(isimprecise, inputs)
    prec = filter(!isimprecise, inputs)

    lb, ub = float.(bounds(inputs))
    d = length(lb)

    n_θ = 1000
    n_u = Int(ceil(n_total / n_θ))

    qoi_vals = Vector{Float64}(undef, n_θ)
    θ_array  = Matrix{Float64}(undef, n_θ, d)


    for i in 1:n_θ
        θ = lb .+ rand(d) .* (ub .- lb)
        θ_inputs = map_to_precise_inputs(θ, imp)
        df = sample([prec..., θ_inputs...], n_u)
        evaluate!(model, df)
        y = df[:, model.name]
        qoi_vals[i] =
            qoi == :mean ? mean(y) :
            qoi == :pf   ? mean(y .<= y_star) :
            error("Unknown QoI.")
        θ_array[i, :] .= θ
    end

    imin = argmin(qoi_vals)
    imax = argmax(qoi_vals)
    return (
        bounds   = [minimum(qoi_vals), maximum(qoi_vals)],
        θ_bounds = [θ_array[imin, :], θ_array[imax, :]],
        n_calls  = n_θ * n_u,
    )
end

# ---------------------------------------------------------
# Generic single-QoI convergence study
# (qoi is now a parameter → one study per QoI, fully separated)
# ---------------------------------------------------------
function convergence_study(
    physical_model, inputs;
    budgets, k, qoi::Symbol, y_star, tol = 0.01,
)
    results = DataFrame(
        budget = Int[], calls = Int[],
        q_min = Float64[], q_max = Float64[],
        rel_min = Float64[], rel_max = Float64[],
        θ_min = Vector[], θ_max = Vector[]
    )
    previous = nothing

    for n_total in budgets
        println("\n[$qoi] Budget = $n_total")
        res = true_doubleloop(physical_model, inputs;
                              n_total = n_total, k = k, qoi = qoi, y_star = y_star)

        if previous === nothing
            rmin = NaN; rmax = NaN
        else
            rmin = relative_change(res.bounds[1], previous.q_min)
            rmax = relative_change(res.bounds[2], previous.q_max)
        end

        push!(results, (n_total, res.n_calls,
                        res.bounds[1], res.bounds[2], rmin, rmax, res.θ_bounds[1], res.θ_bounds[2]))

        previous = (q_min = res.bounds[1], q_max = res.bounds[2])

    end
    return results
end

# ---------------------------------------------------------
# Reusable plot helper (legend pushed OUTSIDE the plot area)
# ---------------------------------------------------------
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

# =========================================================
# Run TWO SEPARATE studies
# =========================================================
k_mean = 5.0
k_pf = 0.01

budgets = Int[
    i * 10^e
    for e in 3:7
    for i in 1:3:7
]

# ---- Study 1: expected response (mean) ----
results_mean = convergence_study(
    physical_model, [x1_g, x2_g];
    budgets = budgets, k = k_mean, qoi = :mean, y_star = -1.427, tol = 0.01,
)
p_mean = plot_convergence(results_mean, "μ bounds", "Expected response convergence")
plot!(p_mean, ylims=[-1.5, 1.5])
display(p_mean)

# ---- Study 2: failure probability (Pf) ----
results_pf = convergence_study(
    physical_model, [x1_g, x2_g];
    budgets = budgets, k = k_pf, qoi = :pf, y_star = -1.427, tol = 0.001,
)
p_pf = plot_convergence(results_pf, "Pf bounds", "Failure probability convergence")
display(p_pf)

println("\nMean converged at budget:  ", results_mean.budget[end], "  (", results_mean.calls[end], " calls)")
println("Pf   converged at budget:  ", results_pf.budget[end],   "  (", results_pf.calls[end],   " calls)")