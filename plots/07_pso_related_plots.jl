using Random
using Plots

gr()

# ---------------------------------------------------------
# Objective: Forrester function, MAXIMIZED directly
# ---------------------------------------------------------
forrester(x) = (6x - 2)^2 * sin(12x - 4)

lb, ub = -0.0, 1.2

# ---------------------------------------------------------
# Minimal, transparent PSO (maximization), same hyperparameters
# style as make_pso() elsewhere in the project: inertia + cognitive
# + social terms, N particles, fixed iteration budget.
# ---------------------------------------------------------
function simple_pso_maximize(f, lb, ub; N=20, ω=0.8, c1=1.5, c2=1.5, iterations=30)
    x = lb .+ rand(N) .* (ub - lb)          # initial positions
    v = zeros(N)

    x_initial = copy(x)                      # save BEFORE any update

    pbest_x = copy(x)
    pbest_f = f.(x)

    gbest_idx = argmax(pbest_f)
    gbest_x   = pbest_x[gbest_idx]
    gbest_f   = pbest_f[gbest_idx]

    best_history = Float64[gbest_f]           # best-so-far, index 0 = before any iteration

    for iter in 1:iterations
        for i in 1:N
            r1, r2 = rand(), rand()
            v[i] = ω*v[i] + c1*r1*(pbest_x[i] - x[i]) + c2*r2*(gbest_x - x[i])
            x[i] = clamp(x[i] + v[i], lb, ub)

            fi = f(x[i])
            if fi > pbest_f[i]
                pbest_f[i] = fi
                pbest_x[i] = x[i]
            end
        end

        gbest_idx = argmax(pbest_f)
        if pbest_f[gbest_idx] > gbest_f
            gbest_f = pbest_f[gbest_idx]
            gbest_x = pbest_x[gbest_idx]
        end
        push!(best_history, gbest_f)
    end

    return (x_initial=x_initial, x_final=x, best_history=best_history,
            gbest_x=gbest_x, gbest_f=gbest_f)
end

result = simple_pso_maximize(forrester, lb, ub; N=10, iterations=15)

println("Found max ≈ $(round(result.gbest_f, digits=4)) at x ≈ $(round(result.gbest_x, digits=4))")

# ---------------------------------------------------------
# Panel 1 (left): best-found value vs. iteration
# ---------------------------------------------------------
p_left = plot(
    0:length(result.best_history)-1, result.best_history;
    lc = :steelblue, lw = 2, marker = :circle, ms = 4, msw = 0,
    xlabel = "PSO iteration", ylabel = "Best f(x) found so far",
    title = "PSO convergence (maximising Forrester function)",
    legend = :bottomright,
    label = "Best value found so far",
)

# ---------------------------------------------------------
# Panel 2 (right): analytical function + initial vs final particles
# ---------------------------------------------------------
x_plot = collect(range(lb, ub, length=400))
y_plot = forrester.(x_plot)

p_right = plot(
    x_plot, y_plot;
    lc = :black, lw = 2, label = "Forrester function",
    xlabel = "x", ylabel = "f(x)",
    title = "Particle positions: initial vs. final",
    legend = :topleft,
)
scatter!(p_right, result.x_initial, forrester.(result.x_initial);
    mc = :red, ms = 6, msw = 0, alpha = 0.8, label = "Initial particles")
scatter!(p_right, result.x_final, forrester.(result.x_final);
    mc = :green, ms = 6, msw = 0, alpha = 0.8, label = "Final particles")
scatter!(p_right, [result.gbest_x], [result.gbest_f];
    marker = :star5, mc = :gold, ms = 12, msw = 1, msc = :black,
    label = "Best found")

# ---------------------------------------------------------
# Combine
# ---------------------------------------------------------
p_combined = plot(p_left, p_right, layout = (1, 2), size = (1300, 500),
    left_margin = 5Plots.mm, bottom_margin = 6Plots.mm)

display(p_combined)

