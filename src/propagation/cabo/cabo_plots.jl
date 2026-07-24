# ==============================================================================
# Shared helper: recover the original training-set size from a cabo_loop result.
# data grows by exactly one row per iteration, matching θ_history's length.
# ==============================================================================
_n_train(result) = nrow(result.data) - length(result.θ_history)

# ==============================================================================
# 1. Epistemic landscape (1D line or 2D heatmap)
# ==============================================================================

function plot_epistemic_landscape(
    specs;
    cabo_min = nothing,
    cabo_max = nothing,
    analytical_qoi = nothing,
    true_min = nothing,
    true_max = nothing,
    qoi_label::String = "QoI",
    grid_bounds = nothing,
    n_grid::Int = 300,
)
    cabo_min === nothing && cabo_max === nothing &&
        error("Provide at least one of cabo_min / cabo_max.")

    _, _, _, v_names = spec_names(specs)
    d = length(v_names)
    d > 2 && error("plot_epistemic_landscape only supports 1D or 2D epistemic spaces (got d=$d).")

    # ── gather sample points in epistemic space ──────────────────────────────
    ref = cabo_min !== nothing ? cabo_min : cabo_max
    n_init = _n_train(ref)

    init_θ = n_init > 0 ?
        reduce(vcat, [augmented_to_epistemic(Vector(ref.data[i, v_names]), specs)' for i in 1:n_init]) :
        nothing

    min_added_θ = (cabo_min !== nothing && !isempty(cabo_min.θ_history)) ?
        reduce(hcat, cabo_min.θ_history)' : nothing
    max_added_θ = (cabo_max !== nothing && !isempty(cabo_max.θ_history)) ?
        reduce(hcat, cabo_max.θ_history)' : nothing

    # ── grid bounds ───────────────────────────────────────────────────────────
    if grid_bounds === nothing
        parts = filter(!isnothing, [init_θ, min_added_θ, max_added_θ])
        all_pts = isempty(parts) ? zeros(0, d) : vcat(parts...)
        grid_bounds = [ (minimum(all_pts[:, k]), maximum(all_pts[:, k])) for k in 1:d ]
    end

    common = (init_θ=init_θ, min_added_θ=min_added_θ, max_added_θ=max_added_θ,
              cabo_min=cabo_min, cabo_max=cabo_max)

    if d == 1
        return _landscape_1d(grid_bounds[1], n_grid, analytical_qoi, qoi_label,
                              true_min, true_max; common...)
    else
        return _landscape_2d(grid_bounds, n_grid, analytical_qoi, qoi_label,
                              true_min, true_max; common...)
    end
end

function _landscape_2d(grid_bounds, n_grid, analytical_qoi, qoi_label, true_min, true_max;
                        init_θ, min_added_θ, max_added_θ, cabo_min, cabo_max)

    (lo1, hi1), (lo2, hi2) = grid_bounds
    g1 = range(lo1, hi1, length=n_grid)
    g2 = range(lo2, hi2, length=n_grid)

    if analytical_qoi !== nothing
        Z = [analytical_qoi([θ1, θ2]) for θ2 in g2, θ1 in g1]   # row=θ2(y), col=θ1(x) -- matches heatmap(x,y,Z)

        if true_min === nothing
            idx = argmin(Z)
            true_min = ([g1[idx[2]], g2[idx[1]]], Z[idx])
        end
        if true_max === nothing
            idx = argmax(Z)
            true_max = ([g1[idx[2]], g2[idx[1]]], Z[idx])
        end

        plt = heatmap(g1, g2, Z; xlabel="θ₁", ylabel="θ₂", c=:thermal,
                      title="$(qoi_label)(θ₁,θ₂)", colorbar=true,
                      legend=:outerbottom, legendcolumns=4)
    else
        plt = plot(xlabel="θ₁", ylabel="θ₂", title="Epistemic search - $(qoi_label)",
                   legend=:outerbottom, legendcolumns=4, xlims=(lo1, hi1), ylims=(lo2, hi2))
    end

    init_θ !== nothing && scatter!(plt, init_θ[:, 1], init_θ[:, 2];
        marker=:diamond, color=:cyan, ms=5, label="init samples", markerstrokewidth=0)
    
    min_added_θ !== nothing && scatter!(plt, min_added_θ[:, 1], min_added_θ[:, 2];
        marker=:cross, color=:green, ms=5, label="added min", markerstrokewidth=2)
    
    max_added_θ !== nothing && scatter!(plt, max_added_θ[:, 1], max_added_θ[:, 2];
        marker=:cross, color=:red, ms=5, label="added max", markerstrokewidth=2)
    
    cabo_min !== nothing && scatter!(plt, [cabo_min.θ_bound[1]], [cabo_min.θ_bound[2]];
        marker=:star, color=:green, ms=7, label="cabo min", markerstrokewidth=1)
    
    cabo_max !== nothing && scatter!(plt, [cabo_max.θ_bound[1]], [cabo_max.θ_bound[2]];
        marker=:star, color=:red, ms=7, label="cabo max", markerstrokewidth=1)

    if true_min !== nothing
        θm, vm = true_min
        scatter!(plt, [θm[1]], [θm[2]]; marker=:circle, color=:green, ms=5, label="true min")
        annotate!(plt, θm[1], θm[2] + 0.05*(hi2-lo2),
            text("($(round(θm[1],digits=2)), $(round(θm[2],digits=2)), $(round(vm,digits=2)))", :black, 8))
    end
    if true_max !== nothing
        θM, vM = true_max
        scatter!(plt, [θM[1]], [θM[2]]; marker=:rect, color=:red, ms=5, label="true max")
        annotate!(plt, θM[1], θM[2] + 0.05*(hi2-lo2),
            text("($(round(θM[1],digits=2)), $(round(θM[2],digits=2)), $(round(vM,digits=2)))", :black, 8))
    end

    return plt
end

function _landscape_1d(bounds1, n_grid, analytical_qoi, qoi_label, true_min, true_max;
                        init_θ, min_added_θ, max_added_θ, cabo_min, cabo_max)

    lo, hi = bounds1
    g = collect(range(lo, hi, length=n_grid))

    local yval  # declare once, in the enclosing scope

    if analytical_qoi !== nothing
    y = [analytical_qoi([θ]) for θ in g]

    if true_min === nothing
        idx = argmin(y)
        true_min = ([g[idx]], y[idx])
    end
    if true_max === nothing
        idx = argmax(y)
        true_max = ([g[idx]], y[idx])
    end

    plt = plot(g, y; lc=:black, lw=2, label="Analytical",
               xlabel="θ", ylabel=qoi_label, title="$(qoi_label)(θ)",
               legend=:outerbottom, legendcolumns=4)
    yval = θ -> analytical_qoi([θ])
    else
        plt = plot(xlabel="θ", title="Epistemic search — $(qoi_label)",
                legend=:outerbottom, legendcolumns=4, xlims=(lo, hi), yticks=false)
        yval = θ -> 0.0
    end

    if init_θ !== nothing
        scatter!(plt, init_θ[:, 1], yval.(init_θ[:, 1]);
            marker=:diamond, color=:cyan, ms=5, label="init samples", markerstrokewidth=0)
    end
    if min_added_θ !== nothing
        scatter!(plt, min_added_θ[:, 1], yval.(min_added_θ[:, 1]);
            marker=:cross, color=:green, ms=5, label="added min", markerstrokewidth=2)
    end
    if max_added_θ !== nothing
        scatter!(plt, max_added_θ[:, 1], yval.(max_added_θ[:, 1]);
            marker=:cross, color=:red, ms=5, label="added max", markerstrokewidth=2)
    end
    if cabo_min !== nothing
        scatter!(plt, [cabo_min.θ_bound[1]], [yval(cabo_min.θ_bound[1])];
            marker=:star, color=:green, ms=7, label="cabo min", markerstrokewidth=1)
    end
    if cabo_max !== nothing
        scatter!(plt, [cabo_max.θ_bound[1]], [yval(cabo_max.θ_bound[1])];
            marker=:star, color=:red, ms=7, label="cabo max", markerstrokewidth=1)
    end
    if true_min !== nothing
        θm, vm = true_min
        scatter!(plt, [θm[1]], [vm]; marker=:circle, color=:green, ms=6, label="true min")
    end
    if true_max !== nothing
        θM, vM = true_max
        scatter!(plt, [θM[1]], [vM]; marker=:rect, color=:red, ms=6, label="true max")
    end

    return plt
end

# ==============================================================================
# 2. BO / BC convergence history
# ==============================================================================

function plot_convergence_history(; cabo_min=nothing, cabo_max=nothing)
    cabo_min === nothing && cabo_max === nothing &&
        error("Provide at least one of cabo_min / cabo_max.")

    panels = Plots.Plot[]
    for (result, label) in ((cabo_min, "MIN"), (cabo_max, "MAX"))
        result === nothing && continue

        push!(panels, plot(result.L_BO_history;
            xlabel="Iteration", ylabel="L_BO", lw=2, marker=:circle, ls=:dash,
            title="$label: L_BO History", legend=false))

        push!(panels, plot(result.L_BC_history;
            xlabel="Iteration", ylabel="L_BC", lw=2, marker=:circle, ls=:dash,
            title="$label: L_BC History", ylims=(0, 1), legend=false))
    end

    n_dirs = length(panels) ÷ 2
    return plot(panels...; layout=(n_dirs, 2), size=(900, 350 * n_dirs))
end

# ==============================================================================
# 3. Bound convergence vs. budget
# ==============================================================================

function plot_bound_convergence(;
    cabo_min = nothing,
    cabo_max = nothing,
    true_bound = nothing,
    qoi_label::String = "QoI bound",
    title_str::String = "Bound convergence",
    combined_budget::Bool = false,
)
    cabo_min === nothing && cabo_max === nothing &&
        error("Provide at least one of cabo_min / cabo_max.")

    panels = Plots.Plot[]

    if cabo_min !== nothing
        offset = _n_train(cabo_min)
        p = plot(
            eachindex(cabo_min.bound_history) .+ offset, cabo_min.bound_history;
            lw=2, marker=:circle, label="lower cabo bound",
            xlabel="Total budget", ylabel=qoi_label * " (min)",
            title=title_str, legend=:topleft,
        )
        if true_bound !== nothing && true_bound[1] !== nothing
            hline!(p, [true_bound[1]]; ls=:dash, c=:gray, label="true lower")
        end
        push!(panels, p)
    end

    if cabo_max !== nothing
        offset = _n_train(cabo_max)
        if combined_budget && cabo_min !== nothing
            offset += length(cabo_min.bound_history)
        end
        p = plot(
            eachindex(cabo_max.bound_history) .+ offset, cabo_max.bound_history;
            lw=2, marker=:circle, label="upper cabo bound",
            xlabel="Total budget", ylabel=qoi_label * " (max)",
            legend=:topleft,
        )
        if true_bound !== nothing && true_bound[2] !== nothing
            hline!(p, [true_bound[2]]; ls=:dash, c=:gray, label="true upper")
        end
        push!(panels, p)
    end

    return plot(panels...; layout=(length(panels), 1), size=(900, 300 * length(panels)))
end

export
    plot_epistemic_landscape, plot_convergence_history, plot_bound_convergence
