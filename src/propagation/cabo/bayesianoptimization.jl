function ei_objective(qoi_type, gp_samples, u_samples, v, μ_qoi_star, sign_dir)

    qoi = estimate_qoi(qoi_type, gp_samples, u_samples, v)
    L_bo = mean(max.(sign_dir .* (μ_qoi_star .- qoi), 0.0))
    
    return - L_bo
end