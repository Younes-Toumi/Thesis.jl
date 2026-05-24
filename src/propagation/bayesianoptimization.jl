function estimate_V(gp, x1, u2, θσ, Nx)

    x1 = fill(x1, Nx)
    θσ  = fill(θσ, Nx)

    X = hcat(x1, u2, θσ)
    μ, σ = predict(gp, X)

    Vy = var(μ)

    return Vy
end


function variance_moments_mcs(gp, x1, u2, θσ, Ng, Nx)

    V_samples = [estimate_V(gp, x1, u2[j, :], θσ, Nx) for j in 1:Ng]
    return mean(V_samples), std(V_samples)
end

function bo_incumbent_objective(gp, θ, u, Ng, Nx)
    x1, θσ = θ[1], θ[2]
    u2 = u

    μ_V, σ_V = variance_moments_mcs(gp, x1, u2, θσ, Ng, Nx)

    α = 1.0
    return μ_V + α * σ_V # or -(μ_V + α σ_V) to maximize
end


function bo_ei_objective(gp, θ, u, μ_V_star, Ng, Nx)
    x1, θσ = θ[1], θ[2]
    u2 = u
    
    V_samples = [estimate_V(gp, x1, u2[j, :], θσ, Nx) for j in 1:Ng]

    return mean(max.(μ_V_star .- V_samples, 0))
end