function ei_objective(qoi_type, gp_samples, u_samples, v, μ_qoi_star, sign_dir; y_star = nothing)

    qoi = estimate_qoi(qoi_type, gp_samples, u_samples, v; y_star = y_star)
    L_bo = mean(max.(sign_dir .* (μ_qoi_star .- qoi), 0.0))


    # if qoi_type == :mean # closed form of the EI
    #     μ_qoi, σ_qoi = estimate_propagation_qoi(qoi_type, gp_samples, u_samples, v; y_star = nothing)
    #     z = (μ_qoi_star - μ_qoi) / σ_qoi

    #     L_bo = sign_dir .* (μ_qoi - μ_qoi_star) * Φ(sign_dir .* z) + σ_qoi * φ(sign_dir .* z)

    # else 
    #     qoi = estimate_qoi(qoi_type, gp_samples, u_samples, v; y_star = y_star)
    #     L_bo = mean(max.(sign_dir .* (μ_qoi_star .- qoi), 0.0))
    # end

    return - L_bo
end