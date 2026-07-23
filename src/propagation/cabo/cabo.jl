function estimate_qoi!(qoi_buffer, qoi_type, gp_samples!, W_buffer, n_u, v; y_star = nothing)
    @inbounds for j in eachindex(v)
        for i in axes(W_buffer, 1)
            W_buffer[i, n_u+j] = v[j]
        end
    end
    gp_samples!(qoi_buffer, W_buffer; qoi_type=qoi_type, y_star=y_star)
    return nothing
end


function best_candidate(qoi_type, gp_samples!, qoi_buffer, W_buffer, n_u, v_data, direction; α=1.0, y_star=nothing)
    n = size(v_data, 1)
    μ_qoi = Vector{Float64}(undef, n)
    σ_qoi = Vector{Float64}(undef, n)
    for i in 1:n
        μ_qoi[i], σ_qoi[i] = estimate_propagation_qoi(qoi_type, gp_samples!, qoi_buffer, W_buffer, n_u, v_data[i, :]; y_star)
    end


    candidates = μ_qoi .+ α .* σ_qoi

    idx = (direction == :min) ? argmin(candidates) : argmax(candidates)

    return idx, μ_qoi, σ_qoi
end

function estimate_propagation_qoi(qoi_type, gp_samples!, qoi_buffer, W_buffer, n_u, v; y_star = nothing)
    estimate_qoi!(qoi_buffer, qoi_type, gp_samples!, W_buffer, n_u, v; y_star)
    return mean(qoi_buffer), std(qoi_buffer)
end


function estimate_final_bound(
    gp, data, specs, qoi_type, direction, y_star,
    u_names, v_names, w_names;
    Nx_final::Int = 10_000,
)
    n_u = length(u_names)
    n_v = length(v_names)

    u_final        = randn(Nx_final, n_u)


    v_data = Matrix(data[:, v_names])
    n_candidates = size(v_data, 1)

    qoi_of_mean = Vector{Float64}(undef, n_candidates)
    X_probe = Matrix{Float64}(undef, Nx_final, n_u + n_v)
    @views X_probe[:, 1:n_u] .= u_final

    for i in 1:n_candidates
        v = @view v_data[i, :]
        @views X_probe[:, n_u+1:end] .= v'
        μ_pred = predict(gp, X_probe; mode=:mean)
        qoi_of_mean[i] = qoi_type == :mean ? mean(μ_pred) :
                         qoi_type == :var  ? var(μ_pred; corrected=true) :
                         qoi_type == :pf   ? mean(μ_pred .< y_star) :
                         error("Unknown qoi_type: $qoi_type")
    end

    idx = direction == :min ? argmin(qoi_of_mean) : argmax(qoi_of_mean)
    v_bound = Vector(v_data[idx, :])
    θ_bound = augmented_to_epistemic(v_bound, specs)

    return (μ_bound = qoi_of_mean[idx], θ_bound = θ_bound, v_bound = v_bound)
end

function cabo_loop(
    physical_model,
    gp_init,
    data_aug_train,
    specs;
    Ng = 1000,
    Nx = 5000,
    qoi_type = :mean,
    y_star = -1.427,
    max_iter::Int = 25,
    direction::Symbol = :min,
    tol_BO::Float64 = 1e-3,
    tol_BC::Float64 = 1e-2
)
    # ── Everything dimension-dependent derives from `specs` ──────────────────
    x_names, w_names, u_names, v_names  = spec_names(specs)
    bounds_u, bounds_v = build_bounds(specs)

    y_symbol = physical_model.name

    n_w = length(w_names)
    n_u = length(u_names)
    n_v = length(v_names)
 
    data = copy(data_aug_train)
    gp   = gp_init
    sign_dir = (direction == :min) ? +1 : -1
 
    θ_history     = Vector{Vector{Float64}}()
    L_BO_history  = Float64[]
    L_BC_history  = Float64[]
    bound_history = Float64[]

    kernel_type = gp_init.kernel_type
    N0 = 500 # 1000   # separate, smaller - sized for EOLE eigenbasis resolution

    # W_aug, _  = build_augmented_design(nothing, specs, Nx)
    # u_samples = Matrix(W_aug[:, u_names])

    u_samples = randn(Nx, n_u)

    W_eole, _ = build_augmented_design(nothing, specs, N0)
    W_eole = Matrix(W_eole)


    for iter in 1:max_iter

        W_buffer = Matrix{Float64}(undef, Nx, n_w)
        qoi_buffer = Vector{Float64}(undef, Ng)

        @views W_buffer[:, 1:n_u] .= u_samples

        W_support    = Matrix(data[:, w_names])
        gp_samples! = build_kl_sampler(gp, W_eole, W_support; N_samples=Ng, Nx=Nx)
        
        println("\n━━━ CABO Iteration $iter / $max_iter ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # ════ Part 1: BO engine ═══════════════════════════════════════════════
        v_data = Matrix(data[:, v_names])
        v_star_index, μ_qoi, σ_qoi = best_candidate(qoi_type, gp_samples!, qoi_buffer, W_buffer, n_u, v_data, direction; α=1.0, y_star = y_star)
 
        v_star     = Vector(data[v_star_index, v_names])
        μ_qoi_star = μ_qoi[v_star_index]
        σ_qoi_star = σ_qoi[v_star_index]
        θ_star     = augmented_to_epistemic(v_star, specs)
 
        res_v = @time "bo objective " Metaheuristics.optimize(
            v -> ei_objective(qoi_buffer, W_buffer, gp_samples!, n_u, qoi_type, v, μ_qoi_star, sign_dir; y_star = y_star),
            bounds_v,
            PSO(N=60)
        )

        L_BO   = -minimum(res_v)

        v_plus = minimizer(res_v)
        θ_plus = augmented_to_epistemic(v_plus, specs)
 
        COV_star = σ_qoi_star / abs(μ_qoi_star + 1e-8)

        # ════ Part 2: BC engine ═══════════════════════════════════════════════
        if qoi_type == :pf
            res_u = @time "u objective " Metaheuristics.optimize(
                u -> u_objective(gp, u, v_plus, y_star),
                bounds_u, PSO(N=60)
            )            
        else
            W     = Matrix(data[:, w_names])
            K_bc  = kernelmatrix(gp.kernel_posterior, RowVecs(W))
            cholK = cholesky(Symmetric(K_bc + 1e-8I))
    
            W_prime, v2_sum = precompute_h_pvc_terms(gp, cholK, W, v_plus, u_samples)
    
            res_u = @time "pvc objective " Metaheuristics.optimize(
                u -> pvc_objective(gp, cholK, W, u, v_plus, W_prime, v2_sum),
                bounds_u, PSO(N=60)
            )
        end

        u_plus = minimizer(res_u)
        w_plus = vcat(u_plus, v_plus)
        x_plus = augmented_to_physical(w_plus, specs)
        row_plus = DataFrame(Dict(x_names .=> x_plus))

        UncertaintyQuantification.evaluate!(physical_model, row_plus)
        y_plus = row_plus[1, y_symbol]

        # Generalized row constructions
        new_row = merge(
            NamedTuple(zip(u_names, u_plus)),
            NamedTuple(zip(v_names, v_plus)),
            (y = y_plus,)
        )

        append!(data, DataFrame([new_row]))
 
        refit!(gp, reshape(w_plus, 1, :), [y_plus])

        # monitoring the current bound
        result = estimate_final_bound(
            gp, data, specs, qoi_type, direction, y_star,
            u_names, v_names, w_names;
            Nx_final=50_000,
        )

        μ_qoi_current_best_bound = result.μ_bound
        θ_current_best_bound           = result.θ_bound

        @printf("\n    Incumbent θ* = %s    μ_qoi(θ*) ≈ %.2e    σ_qoi(θ*) ≈ %.2e\n",
               string(round.(θ_star, digits=3)), μ_qoi_star, σ_qoi_star)
 
        println("    Acquisition θ⁺ = $(round.(θ_plus, digits=4))    EI = $(round(L_BO, digits=4))" *
                        "    COV = $(round(COV_star, sigdigits=4))")
        
        println("    Current estimated bound: $(round(μ_qoi_current_best_bound, digits=3)) @ θ = $(round.(θ_current_best_bound, digits=3))\n")
        

        push!(bound_history, μ_qoi_current_best_bound)
        push!(θ_history, copy(collect(θ_plus)))
        push!(L_BO_history, L_BO)
        push!(L_BC_history, COV_star)

        if L_BO < tol_BO && COV_star < tol_BC
            println("\n ✓ converged")
            break
        end


    end
    
    result = estimate_final_bound(
            gp, data, specs, qoi_type, direction, y_star,
            u_names, v_names, w_names;
            Nx_final=50_000,
        )
        
    μ_qoi_bound_final = result.μ_bound
    θ_bound           = result.θ_bound

    push!(bound_history, μ_qoi_bound_final)


    println("\n  ► $(uppercase(string(direction))) bound ≈ $(round(μ_qoi_bound_final, sigdigits=5))" *
            "  at  θ = $(round.(θ_bound, digits=4))")

    return (
        gp = gp, data = data,
        θ_bound = θ_bound, μ_bound = μ_qoi_bound_final,
        θ_history = θ_history, L_BO_history = L_BO_history, L_BC_history = L_BC_history, bound_history = bound_history
    )
end


export
    estimate_qoi!, best_candidate, estimate_propagation_qoi, estimate_final_bound,
    cabo_loop
