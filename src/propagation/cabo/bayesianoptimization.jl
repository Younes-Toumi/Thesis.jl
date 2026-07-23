function ei_objective(qoi_buffer, W_buffer, gp_samples!, n_u, qoi_type, v, μ_qoi_star, sign_dir; y_star=nothing)
    @inbounds for j in eachindex(v)
        for i in axes(W_buffer, 1)
            W_buffer[i, n_u+j] = v[j]
        end
    end
    gp_samples!(qoi_buffer, W_buffer; qoi_type=qoi_type, y_star=y_star)
    L_bo = mean(x -> max(sign_dir * (μ_qoi_star - x), 0.0), qoi_buffer)
    return -L_bo
end

export
    ei_objective
    