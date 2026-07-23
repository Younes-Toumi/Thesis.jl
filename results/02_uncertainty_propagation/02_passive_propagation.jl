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

x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input
physical_model = model_gfunction
print("Gaussian Mixture...\n")

x_names, w_names, u_names, v_names = spec_names(specs)
y_symbol = physical_model.name

function compare_surrogates_evolution(
    physical_model, 
    specs,
    n_trains
)
    pck_p_max = 3

    kernel_type = GPMatern52
    bases = fill(SurrogateModelling.HermiteBasis(), length(w_names))
    pce_solver = SurrogateModelling.LASSOSolver
    pck_degree = TotalDegree(pck_p_max)

    gps     = Array{SurrogateModelling.GaussianProcess}(undef, length(n_trains))
    pcks    = Array{SurrogateModelling.PolynomialChaosKriging}(undef, length(n_trains))

    for (idx, n_train) in enumerate(n_trains)
        print("currently at n_train = $n_train ...\n")
        data_aug_train, _ = build_augmented_design(physical_model, specs, n_train)

        gp          = SurrogateModelling.GaussianProcess(data_aug_train, y_symbol;          kernel_type=kernel_type())
        pce_trend   = SurrogateModelling.PolynomialChaosExpansion(data_aug_train, y_symbol, bases, pck_degree; solver=pce_solver())
        pck         = SurrogateModelling.PolynomialChaosKriging(data_aug_train, y_symbol,   pce_trend, kernel_type=kernel_type())

        fit!(gp)
        fit!(pck)

        gps[idx] = gp
        pcks[idx] = pck

    end

    return gps, pcks
end

n_trains = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]


gps, pcks = compare_surrogates_evolution(
    physical_model, 
    specs,
    n_trains,
)

function true_doubleloop(
    model, imprecise_inputs;
    n_total::Int, k, qoi::Symbol, y_star::Float64,
    surrogate::Bool = false, specs = nothing,
    batch_size::Int = 200,   # NEW: how many θ's aleatory grids to stack per predict() call
)
    inputs = wrap(imprecise_inputs)
    imp  = filter(isimprecise, inputs)
    prec = filter(!isimprecise, inputs)
    lb, ub = float.(bounds(inputs))
    d = length(lb)

    n_θ = Int(ceil(sqrt(n_total / k)))
    n_u = Int(ceil(sqrt(n_total * k)))

    qoi_vals = Vector{Float64}(undef, n_θ)
    θ_array  = Matrix{Float64}(undef, n_θ, d)

    if surrogate
        specs === nothing && error("Pass `specs` when surrogate=true.")
        x_names, w_names, u_names, v_names = spec_names(specs)
        d_u, d_v = length(u_names), length(v_names)

        relaxed_bounds = Tuple{Float64,Float64}[]
        for s in specs
            if s isa IntervalSpec
                push!(relaxed_bounds, compute_relaxed_bounds(s.θ_L, s.θ_U; v_L=s.v_L, v_U=s.v_U))
            elseif s isa HybridSpec
                for j in 1:length(s.param_names)
                    push!(relaxed_bounds, compute_relaxed_bounds(s.θ_L[j], s.θ_U[j]; v_L=s.v_L[j], v_U=s.v_U[j]))
                end
            end
        end

        # ── batched buffer: B θ's worth of aleatory rows stacked, ONE predict() per batch ──
        X_batch = Matrix{Float64}(undef, batch_size * n_u, d_u + d_v)

        θ_batch = Matrix{Float64}(undef, batch_size, d)

        i = 1
        while i <= n_θ
            b_end = min(i + batch_size - 1, n_θ)
            B = b_end - i + 1

            for (b, θi) in enumerate(i:b_end)
                θ = lb .+ rand(d) .* (ub .- lb)
                θ_array[θi, :] .= θ
                θ_batch[b, :]   .= θ
                rows = ((b-1)*n_u + 1):(b*n_u)
                @views X_batch[rows, 1:d_u] .= randn(n_u, d_u)
                for j in 1:d_v
                    lb_j, ub_j = relaxed_bounds[j]
                    @views X_batch[rows, d_u+j] .= θ_to_v(θ[j], lb_j, ub_j)
                end
            end

            # ONE predict() call for the WHOLE batch, instead of B separate calls
            X_view = Matrix(@view X_batch[1:(B*n_u), :])
            y_batch = predict(model, X_view; mode=:mean)

            for (b, θi) in enumerate(i:b_end)
                rows = ((b-1)*n_u + 1):(b*n_u)
                y_b = @view y_batch[rows]
                qoi_vals[θi] = qoi == :mean ? mean(y_b) :
                               qoi == :pf   ? mean(y_b .<= y_star) :
                               error("Unknown QoI.")
            end

            i = b_end + 1
        end
    else
        # unchanged physical-model branch
        for i in 1:n_θ
            θ = lb .+ rand(d) .* (ub .- lb)
            θ_inputs = map_to_precise_inputs(θ, imp)
            df = sample([prec..., θ_inputs...], n_u)
            evaluate!(model, df)
            y = df[:, model.name]
            qoi_vals[i] = qoi == :mean ? mean(y) : qoi == :pf ? mean(y .<= y_star) : error("Unknown QoI.")
            θ_array[i, :] .= θ
        end
    end

    imin, imax = argmin(qoi_vals), argmax(qoi_vals)
    return (bounds = [minimum(qoi_vals), maximum(qoi_vals)],
            θ_bounds = [θ_array[imin, :], θ_array[imax, :]],
            n_calls = n_θ * n_u)
end


# surrogate propagation — same call, two extra kwargs

for (idx, n_train) in enumerate(n_trains)
    result_mean_gp = @time "gp mean: " true_doubleloop(
        gps[idx], [x1, x2];
        n_total=1*10^6, k=5.0, qoi=:mean, y_star=-1.427,
        surrogate=true, specs=specs
    )

    println("gp  (n₀ = $n_train): bounds = $(round.(result_mean_gp.bounds, digits=3))")

    result_mean_pck = @time "pck mean: " true_doubleloop(
        pcks[idx], [x1, x2];
        n_total=1*10^6, k=5.0, qoi=:mean, y_star=-1.427,
        surrogate=true, specs=specs
    )

    println("pck (n₀ = $n_train): bounds = $(round.(result_mean_pck.bounds, digits=3))\n")



    result_pf_gp = @time "gp pf: " true_doubleloop(
        gps[idx], [x1, x2];
        n_total=10^6, k=0.01, qoi=:pf, y_star=-1.427,
        surrogate=true, specs=specs
    )

    println("gp  (n₀ = $n_train): pf = $(round.(result_pf_gp.bounds, digits=3))")


    result_pf_pck = @time "pck pf: " true_doubleloop(
        pcks[idx], [x1, x2];
        n_total=10^7, k=0.01, qoi=:pf, y_star=-1.427,
        surrogate=true, specs=specs
    )

    println("pck (n₀ = $n_train): pf = $(round.(result_pf_pck.bounds, digits=3))\n")


end