function make_pso(; N::Int=50, ω=0.8, C1=2.0, C2=2.0)
    p = Metaheuristics.PSO(N=N, ω=ω, C1=C1, C2=C2)
    p.options.iterations = 200
    return p
end


function estimate_qoi(qoi_type, gp_samples, u_samples, v)
    if u_samples !== nothing
        Nx  = size(u_samples, 1)
        X   = hcat(u_samples, repeat(v', Nx, 1))

    else
        X   = v
    end

    μ_gps = gp_samples(X)

    # print("μ_gps [min/max = $(round(minimum(μ_gps), digits=2)) / $(round(maximum(μ_gps), digits=2))] = $(round.(μ_gps[1:5], digits=3))\n")

    if qoi_type == :mean
        return vec(mean(μ_gps, dims=2))

    elseif qoi_type == :var
        return vec(var(μ_gps, dims=2; corrected=true))

    elseif qoi_type == :pf
        return vec(mean(μ_gps .< 0, dims=2))

    end
end


function best_candidate(qoi_type, gp_samples, u_samples, v_data, direction; α=1.0)
    n = size(v_data, 1)
    μ_qoi = Vector{Float64}(undef, n)
    σ_qoi = Vector{Float64}(undef, n)
    for i in 1:n
        μ_qoi[i], σ_qoi[i] = estimate_propagation_qoi(qoi_type, gp_samples, u_samples, v_data[i, :])
    end
    candidates = μ_qoi .+ α .* σ_qoi
    idx = (direction == :min) ? argmin(candidates) : argmax(candidates)

    return idx, μ_qoi, σ_qoi
end

# ==============================================================================
# Updated call site — Nx must now be passed explicitly so the buffers can be
# sized correctly up front
# ==============================================================================
#=
gp_samples = build_kl_sampler(gp, Matrix(W_aug), X_train; N_samples=Ng, Nx=Nx)
=#

# ==============================================================================
# estimate_qoi — unchanged from the previous round of fixes; gp_samples now
# does ALL the qoi-aware reduction internally, including the in-place version
# ==============================================================================
# function estimate_qoi(qoi_type, gp_samples, u_samples, v)
#     Nx = size(u_samples, 1)
#     X  = hcat(u_samples, repeat(v', Nx, 1))
#     return gp_samples(X; qoi_type=qoi_type)
# end


function estimate_propagation_qoi(qoi_type, gp_samples, u_samples, v)
    qoi = vec(estimate_qoi(qoi_type, gp_samples, u_samples, v))
    return mean(qoi), std(qoi)
end


function cabo_loop(
    physical_model,
    gp_init,
    data_aug_train,
    specs;
    Ng = 100,
    Nx = 100,
    qoi_type = :mean,
    max_iter::Int = 20,
    direction::Symbol = :min,
    tol_BO::Float64 = 5e-3,
    tol_BC::Float64 = 2.5e-2
)
    # ── Everything dimension-dependent derives from `specs` ──────────────────
    w_names, u_names, v_names  = spec_names(specs)
    bounds_u, bounds_v = build_bounds(specs)
 
    data = copy(data_aug_train)
    gp   = gp_init
    sign_dir = (direction == :min) ? +1 : -1
 
    θ_history    = Vector{Vector{Float64}}()
    L_BO_history = Float64[]
    L_BC_history = Float64[]

    if u_names != Symbol[]
        W_aug, _  = build_augmented_design(nothing, specs, Nx)
        u_samples = Matrix(W_aug[:, u_names])

    else
        u_samples = nothing

    end

    N0 = 500   # separate, smaller — sized for EOLE eigenbasis resolution


    for iter in 1:max_iter
        W_eole, _ = build_augmented_design(nothing, specs, N0)
        W_eole = Matrix(W_eole)

        W_support    = Matrix(data[:, w_names])
        gp_samples = build_kl_sampler(gp, W_eole, W_support; N_samples=Ng)
 
        println("\n━━━ CABO Iteration $iter / $max_iter ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
 
        # ════ Part 1: BO engine ═══════════════════════════════════════════════
        v_data = Matrix(data[:, v_names])
        v_star_index, μ_qoi, σ_qoi = best_candidate(qoi_type, gp_samples, u_samples, v_data, direction; α=1.0)
 
        v_star     = Vector(data[v_star_index, v_names])
        μ_qoi_star = μ_qoi[v_star_index]
        σ_qoi_star = σ_qoi[v_star_index]
        θ_star     = augmented_to_epistemic(v_star, specs)
 
        @printf("    Incumbent θ* = %s    μ_qoi(θ*) ≈ %.2e    σ_qoi(θ*) ≈ %.2e\n",
                string(round.(θ_star, digits=3)), μ_qoi_star, σ_qoi_star)
 
        res_v = @time "bo objective " Metaheuristics.optimize(
            v -> ei_objective(qoi_type, gp_samples, u_samples, v, μ_qoi_star, sign_dir),
            bounds_v,
            make_pso()
        )
 
        v_plus = minimizer(res_v)
        θ_plus = augmented_to_epistemic(v_plus, specs)
        L_BO   = -minimum(res_v)
 
        # μ_qoi_plus, σ_qoi_plus = estimate_propagation_qoi(qoi_type, gp_samples, u_samples, v_plus)
        COV_star = σ_qoi_star / abs(μ_qoi_star)
 
        println("    Acquisition θ⁺ = $(round.(θ_plus, digits=4))    EI = $(round(L_BO, digits=4))" *
                "    COV = $(round(COV_star, sigdigits=4))")
 
 
        # ════ Part 2: BC engine ═══════════════════════════════════════════════
        if u_samples !== nothing
            if qoi_type == :pf
                res_u = @time "u objective " Metaheuristics.optimize(
                    u -> u_objective(gp, u, v_plus),
                    bounds_u, make_pso()
                )            
            else
                W     = Matrix(data[:, w_names])
                K_bc  = kernelmatrix(gp.kernel_posterior, RowVecs(W))
                cholK = cholesky(Symmetric(K_bc + 1e-8I))
        
                W_prime, v2_sum = precompute_h_pvc_terms(gp, cholK, W, v_plus, u_samples)
        
                res_u = @time "pvc objective " Metaheuristics.optimize(
                    u -> pvc_objective(gp, cholK, W, u, v_plus, W_prime, v2_sum),
                    bounds_u, make_pso()
                )
            end

            u_plus = minimizer(res_u)
            w_plus = vcat(u_plus, v_plus)
            x_plus = augmented_to_physical(w_plus, specs)
            y_plus = physical_model(x_plus...)
    
            # Generalized row constructions
            new_row = merge(
                NamedTuple(zip(u_names, u_plus)),
                NamedTuple(zip(v_names, v_plus)),
                (y = y_plus,)
            )

        else
            w_plus = v_plus
            x_plus = augmented_to_physical(w_plus, specs)
            y_plus = physical_model(x_plus...)

            # Generalized row constructions
            new_row = merge(
                NamedTuple(zip(v_names, v_plus)),
                (y = y_plus,)
            )

        end

        append!(data, DataFrame([new_row]))
 
        refit!(gp, reshape(w_plus, 1, :), [y_plus])
 
        push!(θ_history, copy(collect(θ_plus)))
        push!(L_BO_history, L_BO)
        push!(L_BC_history, COV_star)

        if L_BO < tol_BO && COV_star < tol_BC
            println("\n✓ converged")
            break
        end


    end
 
    # ── Final bound ──────────────────────────────────────────────────────────
    # STAGE 1: locate the bound. α=0 so the location is chosen on the pure mean,
    # not inflated by σ. θ_bound is fixed here and never changes below.
    W_eole_final, _   = build_augmented_design(nothing, specs, N0)
    W_eole_final      = Matrix(W_eole_final)
    W_support_final   = Matrix(data[:, w_names])
    gp_samples_final  = build_kl_sampler(gp, W_eole_final, W_support_final; N_samples=Ng)
    v_data_final      = Matrix(data[:, v_names])

    v_bound_index, μ_qoi_bound, _ =
        best_candidate(qoi_type, gp_samples_final, u_samples, v_data_final, direction; α=0.0)
    v_bound = Vector(data[v_bound_index, v_names])
    θ_bound = augmented_to_epistemic(v_bound, specs)

    if u_samples !== nothing
        # STAGE 2: refine the GP AT θ_bound. The main loop refined at v_plus, but
        # the selected θ_bound may never have been refined there — that residual
        # GP error at this slice is the only bias source left once EOLE and
        # quadrature are gone. A few BC points here shrink it.
        n_final_refine = 5
        for _ in 1:n_final_refine
            W = Matrix(data[:, w_names])
            if qoi_type == :pf
                res_u = Metaheuristics.optimize(
                    u -> u_objective(gp, u, v_bound),
                    bounds_u, make_pso()
                )
            else
                K_bc  = kernelmatrix(gp.kernel_posterior, RowVecs(W))
                cholK = cholesky(Symmetric(K_bc + 1e-8I))
                W_prime, v2_sum = precompute_h_pvc_terms(gp, cholK, W, v_bound, u_samples)
                res_u = Metaheuristics.optimize(
                    u -> pvc_objective(gp, cholK, W, u, v_bound, W_prime, v2_sum),
                    bounds_u, make_pso()
                )
            end

            u_bound = minimizer(res_u)
            w_bound = vcat(u_bound, v_bound)
            x_bound = augmented_to_physical(w_bound, specs)
            y_bound = physical_model(x_bound...)

            new_row = merge(
                NamedTuple(zip(u_names, u_bound)),
                NamedTuple(zip(v_names, v_bound)),
                (y = y_bound,)
            )
            append!(data, DataFrame([new_row]))
            refit!(gp, reshape(w_bound, 1, :), [y_bound])
        end

        # STAGE 3: report the DIRECT integral on the refined GP (no EOLE), large Nx.
        Nx_final        = 10_000
        W_aug_final, _  = build_augmented_design(nothing, specs, Nx_final)
        u_samples_final = Matrix(W_aug_final[:, u_names])
        X_final         = hcat(u_samples_final, repeat(v_bound', Nx_final, 1))
        μ_pred          = predict(gp, X_final; mode=:mean)

        if qoi_type == :mean
            μ_qoi_bound_final = mean(μ_pred)
        elseif qoi_type == :var
            μ_qoi_bound_final = var(μ_pred; corrected=true)
        elseif qoi_type == :pf
            μ_qoi_bound_final = mean(μ_pred .< 0)
        end
    else
        μ_qoi_bound_final = μ_qoi_bound[v_bound_index]
    end

    println("\n  ► $(uppercase(string(direction))) bound ≈ $(round(μ_qoi_bound_final, sigdigits=5))" *
            "  at  θ = $(round.(θ_bound, digits=4))")

    return (
        gp = gp, data = data,
        θ_bound = θ_bound, μ_bound = μ_qoi_bound_final,
        θ_history = θ_history, L_BO_history = L_BO_history, L_BC_history = L_BC_history
    )
end

