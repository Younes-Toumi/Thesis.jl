φ(z) = pdf(Normal(), z)
Φ(z) = cdf(Normal(), z)

function estimate_propagation(gp, u1, u2, Θμ1, Θμ2, Nx)

    X = hcat(
        u1,
        u2,
        fill(Θμ1, Nx), 
        fill(Θμ2, Nx)
    )

    μ, σ = predict(gp, X)

    μ_M = mean(μ)
    σ_M2 = mean(σ .^ 2) + var(μ)

    return μ_M, σ_M2
end

function bo_incumbent_objective_response(gp, θ, u, Nx)
    θμ1, θμ2 = θ[1], θ[2]
    u1, u2 = u[1], u[2]

    α = 1.0
    μ_samples, σ_samples = estimate_propagation(gp, u1, u2, θμ1, θμ2, Nx)

    return μ_samples + α * sqrt(σ_samples)
end


function AEI_objective(gp, θ, u, Nx, μ_M_star)

    θμ1, θμ2 = θ
    u1, u2 = u
    μ_M, σ_M2 = estimate_propagation(gp, u1, u2, θμ1, θμ2, Nx)

    σ_M = sqrt(max(σ_M2, 1e-12))

    if σ_M < 1e-12
        return 0.0
    end

    z = (μ_M_star - μ_M) / σ_M

    aei = (μ_M_star - μ_M) * Φ(z) + σ_M * φ(z)

    return aei   # PSO minimizes
end



using Distributions

function bo_ei_objective_response(gp, θ, θ_star, Ng, Nx)
    vals = Float64[]
    for i in 1:Ng
        u = rand(2)
        f = estimate_propagation(gp, u[1], u[2], θ[1], θ[2], Nx)[1]
        f_star = estimate_propagation(gp, u[1], u[2], θ_star[1], θ_star[2], Nx)[1]
        push!(vals, max(f_star - f, 0.0))
    end
    return mean(vals)
end







######################################################

function estimate_V(gp, x1, u2, θσ, Nx)

    x1 = fill(x1, Nx)
    θσ  = fill(θσ, Nx)

    X = hcat(x1, u2, θσ)
    μ, σ = predict(gp, X)

    Vy = var(μ)

    return Vy
end


function estimate_V_EOLE(gp, x1, u2, θσ, Nx, Ng)

    x1_vec = fill(x1, Nx)
    θσ_vec = fill(θσ, Nx)

    W = hcat(x1_vec, u2, θσ_vec)

    # build factory once
    factory = eole_stuff(gp, W)

    # variance for each GP realization
    V_samples = zeros(Ng)

    for j in 1:Ng

        # one GP realization
        f = factory()

        u2_eval = rand(Nx)

        X_eval = hcat(
            fill(x1, Nx),
            u2_eval,
            fill(θσ, Nx)
        )

        yvals = [f(X_eval[i,:]) for i in 1:Nx]

        # aleatory variance
        V_samples[j] = var(yvals)
    end

    return V_samples
end




# kernel vectors
function k_vec(kernel, W, w)
    m = size(W,1)
    k = zeros(m)
    for j in 1:m
        k[j] = kernel(w, W[j,:])
    end
    return k
end


function posterior_sample_factory(kernel, W, V, λ, μy, K, r)

    cholK = cholesky(Symmetric(K + 1e-8I))

    return function ()
        ξ = randn(r)   # <-- randomness here

        λr = λ[1:r]
        Vr = V[:, 1:r]

        h = w -> begin
            kvec = k_vec(kernel, W, w)
            dot(kvec, Vr * (ξ ./ sqrt.(λr)))
        end

        hW = [h(W[i, :]) for i in eachindex(eachrow(W))]

        return w -> begin
            kvec = k_vec(kernel, W, w)
            α = cholK \ hW
            μ_hat_w = dot(kvec, α)
            μy(w) - μ_hat_w + h(w)
        end
    end
end

function eole_stuff(gp, W)
    # 2. EOLE covariance matrix
    m = size(W,1)
    K = zeros(m,m)

    μy = w -> predict(gp, reshape(w, 1, :))[1][1]


    for i in 1:m
        for j in 1:m
            K[i,j] = gp.kernel_prior(W[i,:], W[j,:])
        end
    end

    eig = eigen(Symmetric(K))
    λ = eig.values
    λ = max.(λ, 0.0) # last mode is negative but close to 0, numerical stuff
    V = eig.vectors

    # reordering
    idx = sortperm(λ, rev=true)
    λ = λ[idx]
    V = V[:, idx]

    # r effective
    energy = cumsum(λ) ./ sum(λ)
    r = findfirst(x -> x ≥ 0.99, energy)


    μy = w -> predict(gp, reshape(w, 1, :))[1][1]

    factory = posterior_sample_factory(
        gp.kernel_prior,
        W,
        V,
        λ,
        μy,
        K,
        r
    )
    return factory
end




function variance_moments_mcs(gp, x1, u2, θσ, Ng, Nx)

    # V_samples = [estimate_V(gp, x1, u2[j, :], θσ, Nx) for j in 1:Ng]
    V_samples = estimate_V_EOLE(gp, x1, u2, θσ, Nx, Ng)
    
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
    
    # V_samples = [estimate_V(gp, x1, u2[j, :], θσ, Nx) for j in 1:Ng]
    # V_samples = estimate_V_EOLE(gp, x1, u2, θσ, Nx, Ng)

    return mean(max.(μ_V_star .- V_samples, 0))
end