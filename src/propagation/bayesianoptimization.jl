function estimate_V(gp, x1, u2, θσ, Nx)

    x1 = fill(x1, Nx)
    θσ  = fill(θσ, Nx)

    X = hcat(x1, u2, θσ)
    μ, σ = predict(gp, X)

    Vy = var(μ)

    return Vy
end

function estimate_V_KL(gp, x1, u2, θσ; tol=1e-6)
    Nx = length(u2)
    Xq = hcat(fill(x1, Nx), u2, fill(θσ, Nx))

    μ, Σ = gp_posterior_mean_cov(gp, Xq)   # use the trained GP object
    gsample, Kcut = sample_gp_kl(μ, Σ; tol=tol)

    return var(gsample)
end

function variance_moments_mcs(gp, x1, u2, θσ, Ng, Nx)

    V_samples = [estimate_V(gp, x1, u2[j, :], θσ, Nx) for j in 1:Ng]
    # V_samples = [estimate_V_KL(gp, x1, u2, θσ) for j in 1:Ng]

    return V_samples, mean(V_samples), std(V_samples)
end

function bo_incumbent_objective(gp, θ, u, Ng, Nx)
    x1, θσ = θ[1], θ[2]
    u2 = u

    V_samples, μ_V, σ_V = variance_moments_mcs(gp, x1, u2, θσ, Ng, Nx)

    α = 1.0
    return μ_V + α * σ_V # or -(μ_V + α σ_V) to maximize
end


function bo_ei_objective(gp, θ, u, μ_V_star, Ng, Nx)
    x1, θσ = θ[1], θ[2]
    u2 = u
    
    V_samples = [estimate_V(gp, x1, u2[j, :], θσ, Nx) for j in 1:Ng]
    # V_samples = [estimate_V_KL(gp, x1, u2, θσ) for j in 1:Ng]

    return mean(max.(μ_V_star .- V_samples, 0))
end