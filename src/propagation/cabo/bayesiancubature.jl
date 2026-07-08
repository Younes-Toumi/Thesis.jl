# mean and var related

function precompute_h_pvc_terms(gp, cholK, W, v_plus::AbstractVector, u_samples)
    
    kern = gp.kernel_posterior
    Nx      = size(u_samples, 1)
    W_prime = hcat(u_samples, repeat(v_plus', Nx, 1))    # Nx × d
    K_prime = kernelmatrix(kern, RowVecs(W_prime), RowVecs(W))
    v2_sum  = cholK.L \ vec(sum(K_prime, dims=1))         # N₀-vector
    return W_prime, v2_sum
end

function h_pvc(gp, cholK, W, u::AbstractVector, v_plus::AbstractVector, W_prime, v2_sum)
    
    kern = gp.kernel_posterior
    Nx      = size(W_prime, 1)
    w_mat   = reshape([u; v_plus], 1, :)

    k_vec       = vec(kernelmatrix(kern, RowVecs(w_mat), RowVecs(W)))
    k_cross_sum = sum(kernelmatrix(kern, RowVecs(w_mat), RowVecs(W_prime))) # Nx evals

    v1          = cholK.L \ k_vec

    return (k_cross_sum - dot(v1, v2_sum)) / Nx
end


function pvc_objective(gp, cholK, W, u::AbstractVector, v_plus::AbstractVector, W_prime, v2_sum)
    h   = h_pvc(gp, cholK, W, u, v_plus, W_prime, v2_sum)
    phi = φ_vec(u)
    return -max(0.0, h * phi)
end



# Pf related 
function u_objective(gp, u::AbstractVector, v_plus::AbstractVector)
    w        = vcat(u, v_plus)
    μ_w, σ_w = predict(gp, reshape(w, 1, :), mode=:mean_and_var)
    return abs(μ_w[1]) / max(σ_w[1], 1e-10)
end