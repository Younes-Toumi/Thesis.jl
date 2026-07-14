using SurrogateModelling
using UncertaintyQuantification
using UncertaintyQuantification: sample, wrap, isimprecise, middle, minimize, RobustOrthoMADS, bounds, map_to_precise_inputs

x1_g = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2_g = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

X_g = [x1_g, x2_g]

physical_model = model_gfunction


# design = MonteCarlo(15000)

# df = sample(X_g, design)

# propagate_intervals!(physical_model, df)

# # performance function: failure when g ≤ 0, so g_performance = g(x) (negative = failure)
# y_star = -1.0
# g_performance(df) = df.y .- y_star    # failure when g ≤ y*  ⟺  (g - y*) ≤ 0

# # heavier, rare-event-capable sim for the lower bound; light MC for the upper
# # heavier, rare-event-capable sim for the lower bound; light MC for the upper
# dl = DoubleLoop(
#     SubSetSimulation(1000, 0.1, 10, Normal()),   # lb: subset simulation resolves small Pf
#     MonteCarlo(5000)                            # ub: plain MC is fine, Pf is moderate
# )

# pf_bounds, res_lb, res_ub = probability_of_failure(
#     physical_model,        # your UQModel that writes :y via evaluate!
#     g_performance,          # df -> performance values (failure when ≤ 0)
#     [x1_g, x2_g],
#     dl
# )

# println("Pf bounds: ", pf_bounds)                      # Interval(pf_lb, pf_ub)
# println("θ at Pf_min: ", res_lb, "   θ at Pf_max: ", res_ub)


# using UncertaintyQuantification: sample, wrap, isimprecise, bounds, map_to_precise_inputs
# using Statistics, Random

# """
#     true_doubleloop(model, imprecise_inputs; N_θ, n_u, qoi=:mean, y_star=0.0, seed=42)

# Brute-force double-loop MCS exactly as in eq:dl_outer–eq:dl_bounds:
# outer = N_θ uniform draws over the epistemic box, inner = n_u aleatory
# draws at each θ on the TRUE model. Bounds = min/max over outer samples.
# No optimizer — this is the unambiguous reference.
# """
# function true_doubleloop(
#     model, 
#     imprecise_inputs;
#     n_total::Int, 
#     k::Int,
#     qoi::Symbol,
#     y_star::Float64
# )

#     inputs = wrap(imprecise_inputs)
#     imp  = filter(isimprecise, inputs)
#     prec = filter(!isimprecise, inputs)

#     lb, ub = float.(bounds(inputs))
#     d = length(lb)

#     n_θ = Int(ceil(sqrt(n_total / k)))
#     n_u = Int(ceil(sqrt(n_total * k)))

#     qoi_vals = Vector{Float64}(undef, n_θ)
#     θ_array = Matrix{Float64}(undef, n_θ, d)


#     for k in 1:n_θ
#         # ── OUTER: one uniform θ draw across the box ──
#         θ = lb .+ rand(d) .* (ub .- lb)
#         θ_inputs = map_to_precise_inputs(θ, imp)

#         # ── INNER: n_u aleatory draws at this θ on the TRUE model ──
#         df = sample([prec..., θ_inputs...], n_u)
#         evaluate!(model, df)
#         y = df[:, model.name]

#         qoi_vals[k] = qoi === :mean ? mean(y) :
#                       qoi === :pf   ? mean(y .<= y_star) :
#                       error("qoi must be :mean or :pf")

#         θ_array[k, :] = θ
#     end

#     min_idx, max_idx = argmin(qoi_vals), argmax(qoi_vals)

#     return (
#         bounds   = [minimum(qoi_vals), maximum(qoi_vals)],
#         θ_bounds = [θ_array[min_idx, :], θ_array[max_idx, :]],
#         n_calls  = n_θ * n_u,
#     )
# end

# k = 50
# n_budgets = Int.([1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7])
# y_star = -1.43


# for i in eachindex(n_budgets)
#     print("\n\nn@ i, budget: [$i, $(n_budgets[i])] = = = = = = = = = = = = = = = = = = = = \n")
    
#     n_total = n_budgets[i]
#     res_mean = true_doubleloop(
#         physical_model, 
#         [x1_g, x2_g]; 
#         n_total=n_total,
#         k=k, 
#         qoi=:mean, 
#         y_star=y_star
#     )

#     print(
#         "~ μ_bound estimation | actual calls: $(res_mean.n_calls)\n" *
#         "    μ_MIN bound: $(round(res_mean.bounds[1], digits=4))  $(rpad("", 4)) @ θ_MIN = $(round.(res_mean.θ_bounds[1], digits=3))\n" *
#         "    μ_MAX bound: $(round(res_mean.bounds[2], digits=4))  $(rpad("", 4)) @ θ_MAX = $(round.(res_mean.θ_bounds[2], digits=3))\n\n"
#         )

#     res_pf = true_doubleloop(
#         physical_model, 
#         [x1_g, x2_g]; 
#         n_total=n_total,
#         k=k, 
#         qoi=:pf, 
#         y_star=y_star
#     )

#     print(
#         "~ Pf_bound estimation | actual calls: $(res_pf.n_calls)\n" *
#         "    Pf_MIN bound: $(round(res_pf.bounds[1], digits=4)) $(rpad("", 4)) @ θ_MIN = $(round.(res_pf.θ_bounds[1], digits=3))\n" *
#         "    Pf_MAX bound: $(round(res_pf.bounds[2], digits=4)) $(rpad("", 4)) @ θ_MAX = $(round.(res_pf.θ_bounds[2], digits=3))\n"
#         )
# end


using SurrogateModelling
using UncertaintyQuantification
using Statistics
using DataFrames
using Plots

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
# results_pf = convergence_study(
#     physical_model, [x1_g, x2_g];
#     budgets = budgets, k = k_pf, qoi = :pf, y_star = -1.427, tol = 0.001,
# )
# p_pf = plot_convergence(results_pf, "Pf bounds", "Failure probability convergence")
# display(p_pf)

println("\nMean converged at budget:  ", results_mean.budget[end], "  (", results_mean.calls[end], " calls)")
# println("Pf   converged at budget:  ", results_pf.budget[end],   "  (", results_pf.calls[end],   " calls)")



# 11×8 DataFrame
#  Row │ budget     calls      q_min     q_max    rel_min      rel_max      θ_min                  θ_max                
#      │ Int64      Int64      Float64   Float64  Float64      Float64      Vector                 Vector               
# ─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │   9000000    9003478  -1.34113  1.3218   0.00335165   0.0023401    [-0.611241, 0.492063]  [0.533226, 0.861077]
#    2 │  10000000   10006880  -1.33769  1.32765  0.00256345   0.00442608   [-0.505443, 0.493615]  [0.557131, 0.797073]
#    3 │  30000000   30007600  -1.35153  1.32676  0.0103453    0.000671829  [-0.567772, 0.535292]  [0.556216, 0.811401]
#    4 │  50000000   50013356  -1.34869  1.32648  0.00210265   0.000205868  [-0.603127, 0.518459]  [0.54738, 0.80003]
#    5 │  70000000   70009078  -1.34887  1.32683  0.000137116  0.000260497  [-0.58562, 0.537132]   [0.544348, 0.803543]
#    6 │  90000000   90011002  -1.35129  1.32785  0.00179744   0.000767554  [-0.554845, 0.521338]  [0.565159, 0.800665]
#    7 │ 100000000  100020753  -1.34613  1.32618  0.00382342   0.00125868   [-0.611598, 0.513249]  [0.567222, 0.786311]
#    8 │ 300000000  300002580  -1.34908  1.3272   0.00219062   0.000773406  [-0.588571, 0.526362]  [0.552553, 0.809119]
#    9 │ 500000000  500000000  -1.35053  1.32586  0.00107584   0.00101308   [-0.565242, 0.530077]  [0.542804, 0.808629]
#   10 │ 700000000  700052113  -1.35045  1.3269   5.435e-5     0.000785795  [-0.561927, 0.522674]  [0.564084, 0.786902]
#   11 │ 900000000  900052611  -1.35004  1.32703  0.000305846  9.96579e-5   [-0.552161, 0.531763]  [0.564653, 0.787348]


#  Row │ budget     calls      q_min    q_max      rel_min  rel_max    θ_min                   θ_max                 
#      │ Int64      Int64      Float64  Float64    Float64  Float64    Vector                  Vector                
# ─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │   9000000    9000107      0.0  0.0996016      0.0  0.0529311  [-1.39603, 0.0768529]   [-0.582688, 0.491087]
#    2 │  10000000   10016205      0.0  0.120755       0.0  0.212377   [-1.3977, 1.30183]      [-0.574042, 0.529415]
#    3 │  30000000   30048894      0.0  0.0958606      0.0  0.206155   [0.495728, -0.0831476]  [-0.624898, 0.53756]
#    4 │  50000000   50033472      0.0  0.0929054      0.0  0.0308277  [-1.4314, 1.34412]      [-0.512488, 0.484513]
#    5 │  70000000   70000000      0.0  0.0942857      0.0  0.0148571  [0.398022, 0.59228]     [-0.54791, 0.489486]
#    6 │  90000000   90031660      0.0  0.0869018      0.0  0.0783146  [-1.12958, 1.33739]     [-0.553339, 0.524021]
#    7 │ 100000000  100040751      0.0  0.0943847      0.0  0.0861081  [0.461153, 1.31512]     [-0.573126, 0.534364]
#    8 │ 300000000  300179000      0.0  0.0855172      0.0  0.0939502  [0.712189, 1.2049]      [-0.544562, 0.52057]
#    9 │ 500000000  500047202      0.0  0.0865847      0.0  0.0124825  [0.903321, 0.503609]    [-0.54288, 0.527899]
#   10 │ 700000000  700128792      0.0  0.0840108      0.0  0.0297267  [0.93604, 0.900654]     [-0.538593, 0.519477]
#   11 │ 900000000  900008190      0.0  0.0832669      0.0  0.0088549  [-0.163046, -0.963475]  [-0.560362, 0.501431]