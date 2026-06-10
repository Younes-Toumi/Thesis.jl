# ──────────────────────────────────────────────────────────────────────────────
# bayesiancubature.jl  –  BC engine for CABO
# ──────────────────────────────────────────────────────────────────────────────
# No logic changes needed here – the PVC objective is correct.
# Minor: added docstrings and a guard in PVC for robustness.
# ──────────────────────────────────────────────────────────────────────────────

"""
    PVC(gp, u, θp) → σ²_GP

Posterior Variance Contribution at the single augmented point (u, θ).
Returns the GP posterior variance σ²_GP(u, θ) – the BC engine maximises
this to find the aleatory sample u⁺ that is most uncertain at θ⁺.

u  : 2-element vector  [u1, u2]  ∈ [0,1]²
θp : 2-element vector  [θμ1, θμ2]
"""
function PVC(gp, u, θp)
    θμ1, θμ2 = θp
    u1,  u2  = u
    x        = [u1, u2, θμ1, θμ2]
    _, σ     = predict(gp, reshape(x, 1, :))
    return max(0.0, σ[1]^2)    # guard: variance is always ≥ 0
end


"""
    PVC_z(gp, z, θp) → σ²_GP

PVC evaluated in standard-normal z-space: maps z → u via the normal CDF.
Optimising in z-space over [−3,3]² covers ≈99.7 % of the N(0,1) mass,
which matches the distribution we integrate over in estimate_propagation.
"""
function PVC_z(gp, z, θp)
    return PVC(gp, cdf.(Normal(), z), θp)
end


"""
    BC_objective_z(gp, z, θp) → −σ²_GP

Returns −PVC_z so that PSO (a minimiser) maximises the posterior variance.
"""
function BC_objective_z(gp, z, θp)
    return -PVC_z(gp, z, θp)
end



# function k_vec(kernel, W, w)
#     m = size(W,1)
#     k = zeros(m)
#     for j in 1:m
#         k[j] = kernel(w, W[j,:])
#     end
#     return k
# end

# function eole_stuff(gp, W)
#     # 2. EOLE covariance matrix
#     m = size(W,1)
#     K = zeros(m,m)

#     μy = w -> predict(gp, reshape(w, 1, :))[1][1]


#     for i in 1:m
#         for j in 1:m
#             K[i,j] = gp.kernel_prior(W[i,:], W[j,:])
#         end
#     end

#     cholK = cholesky(Symmetric(K + 1e-8I))
# end