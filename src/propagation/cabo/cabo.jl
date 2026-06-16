
function make_pso(; N::Int=50, ω=0.8, C1=2.0, C2=2.0)
    p = PSO(N=N, ω=ω, C1=C1, C2=C2)
    p.options.iterations = 100
    return p
end

function estimate_qoi(qoi_type, gp_samples, u_samples, v)
    Nx  = size(u_samples, 1)
    X   = hcat(u_samples, repeat(v', Nx, 1))

    μ_gps = gp_samples(X)

    if qoi_type == :mean
        return vec(mean(μ_gps, dims=2))

    elseif qoi_type == :var
        return vec(var(μ_gps, dims=2; corrected=true))

    elseif qoi_type == :pf
        return vec(mean(μ_gps .< 0, dims=2))

    end
end

function estimate_propagation_qoi(qoi_type, gp_samples, u_samples, v)
    qoi = vec(estimate_qoi(qoi_type, gp_samples, u_samples, v))
    return mean(qoi), std(qoi)
end

function cabo_loop(
    gp_init,
    data_aug_train,
    w_names,
    specs;
    Ng = 100,
    Nx = 100,
    qoi_type = :mean,
    max_iter::Int = 20,
    direction::Symbol = :min,
    tol_BO::Float64 = 5e-3,
    tol_BC::Float64 = 2.5e-2

)

    data = copy(data_aug_train)
    gp   = gp_init
    sign_dir = (direction == :min) ? +1 : -1

    θ_history    = Vector{Vector{Float64}}()
    L_BO_history = Float64[]
    L_BC_history = Float64[]

    W_aug, W_phys =    build_augmented_design(nothing, specs, Nx; seed=42)
    u_samples = Matrix(W_aug[:, u_names])

    for iter in 1:max_iter
        X_train = data[:, w_names]
        gp_samples = build_kl_sampler(gp, Matrix(W_aug), Matrix(X_train); N_samples=Ng)
        
        println("\n━━━ CABO Iteration $iter / $max_iter ━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # ════ Part 1: BO engine ═══════════════════════════════════════════════

        # 1a. Incumbent: θ* = argmin [μ_qoi + α σ_qoi] or argmax [μ_qoi + α σ_qoi]:
        # extracting the incubent from the data, as suggested by the papers, and not use a PSO optimization
        v_data = Matrix(data[:, v_names])
        n_samples, _ = size(data)

        μ_qoi = Vector{Float64}(undef, n_samples)
        σ_qoi = Vector{Float64}(undef, n_samples)

        for i in 1:n_samples
            μ_val, σ_val = estimate_propagation_qoi(qoi_type, gp_samples, u_samples, v_data[i, :])
            
            μ_qoi[i] = μ_val
            σ_qoi[i] = σ_val
        end
        
        candidates = μ_qoi .+ 1.0 .* σ_qoi # α = 1.0
        v_star_index = (direction == :min) ? argmin(candidates) : argmax(candidates)

        v_star = Vector(data[v_star_index, v_names])        
        μ_qoi_star = μ_qoi[v_star_index]
        σ_qoi_star = σ_qoi[v_star_index]

        θ_star = augmented_to_epistemic(v_star, specs)
 
        @printf("    Incumbent θ* = %s    μ_qoi(θ*) ≈ %.2e    σ_qoi(θ*) ≈ %.2e\n",
                string(round.(θ_star, digits=3)),
                μ_qoi_star,
                σ_qoi_star
        )


         
        # 1b. v⁺ = argmax EI(v)
        res_v =  Metaheuristics.optimize(
            v -> ei_objective(qoi_type, gp_samples, u_samples, v, μ_qoi_star, sign_dir),
            bounds_v,
            make_pso()
        )

        v_plus = minimizer(res_v)
        θ_plus = augmented_to_epistemic(v_plus, specs)        

        L_BO   = -minimum(res_v)
        μ_qoi_plus, σ_qoi_plus = estimate_propagation_qoi(qoi_type, gp_samples, u_samples, v_plus)
        
        COV_plus = σ_qoi_plus / abs(μ_qoi_plus)

        println("    Acquisition θ⁺ = $(round.(θ_plus, digits=4))    EI = $(round(L_BO/span, digits=4))" *
                "    COV = $(round(COV_plus, sigdigits=4))")


        if L_BO/span < tol_BO && COV_plus < tol_BC
            println("\n✓ converged")
            break
        end

        # ════ Part 2: BC engine ═══════════════════════════════════════════════
        W = Matrix(data[:, w_names])
        K = kernelmatrix(gp.kernel_posterior, RowVecs(W))
        cholK = cholesky(Symmetric(K + 1e-8I))

        W_prime, v2_sum = precompute_h_pvc_terms(gp, cholK, W, v_plus, u_samples)

        res_u  = Metaheuristics.optimize(
            u -> pvc_objective(gp, cholK, W, u, v_plus, W_prime, v2_sum),
            bounds_u,
            make_pso()
        )

        u_plus        = minimizer(res_u)
 
        u1_plus, u2_plus = u_plus
        v1_plus, v2_plus = v_plus

        w_plus = vcat(u_plus, v_plus)
        x_plus = augmented_to_physical(w_plus, specs)
        y_plus = physical_model(x_plus...)

        append!(data, DataFrame(
            w_names[1] => [u1_plus],
            w_names[2] => [u2_plus],
            w_names[3] => [v1_plus],
            w_names[4] => [v2_plus],            
            :y         => [y_plus],
        ))

        # gp = GaussianProcess(data, :y, kernel_type = kernel())
        # @time "    fit!" fit!(gp)
        refit!(gp, reshape(w_plus, 1, :), [y_plus])

        push!(θ_history, copy(collect(θ_plus)))
        push!(L_BO_history, L_BO/span)
        push!(L_BC_history, COV_plus)

    end


    v_data = Matrix(data[:, v_names])
    n_samples, _ = size(data)

    μ_qoi_bound = Vector{Float64}(undef, n_samples)
    σ_qoi_bound = Vector{Float64}(undef, n_samples)

    X_train = data[:, w_names]        
    gp_samples = build_kl_sampler(gp, Matrix(W_aug), Matrix(X_train); N_samples=Ng)


    for i in 1:n_samples
        μ_val, σ_val = estimate_propagation_qoi(qoi_type, gp_samples, u_samples, v_data[i, :])
        
        μ_qoi_bound[i] = μ_val
        σ_qoi_bound[i] = σ_val
    end
    
    candidates = μ_qoi_bound # .+ 1.0 .* σ_qoi # α = 1.0
    v_bound_index = (direction == :min) ? argmin(candidates) : argmax(candidates)

    v_bound = Vector(data[v_bound_index, v_names])        
    μ_qoi_bound_final = μ_qoi_bound[v_bound_index]     
    dir_str = uppercase(string(direction))

    θ_bound = augmented_to_epistemic(v_bound, specs)        


    println("\n  ► $(dir_str) bound ≈ $(round(μ_qoi_bound_final, sigdigits=5))" *
            "  at  θ = $(round.(θ_bound, digits=4))")
 
    return (
        gp = gp,
        data = data,
        θ_bound = θ_bound,
        μ_bound = μ_qoi_bound_final,
        θ_history = θ_history,
        L_BO_history = L_BO_history,
        L_BC_history = L_BC_history

    )
end