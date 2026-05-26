function kernelmatrix(k, X, Y)
    n = size(X, 1)
    m = size(Y, 1)
    K = Matrix{Float64}(undef, n, m)

    for i in 1:n
        for j in 1:m
            K[i, j] = k(X[i, :], Y[j, :])
        end
    end

    return K
end

function gp_posterior_mean_cov(gp, Xq; noise=1e-8)

    k = build_kernel(gp.kernel, gp.θ)

    K   = kernelmatrix(k, gp.X, gp.X) .+ noise * I(size(gp.X,1))
    Kq  = kernelmatrix(k, Xq, gp.X)
    Kqq = kernelmatrix(k, Xq, Xq)

    α = K \ gp.y
    μ = Kq * α

    Σ = Kqq - Kq * (K \ Kq')
    Σ = Symmetric((Σ + Σ') / 2)

    return μ, Σ
end

function sample_gp_kl(gp, Xq; tol=1e-6)
    μ, Σ = gp_posterior_mean_cov(gp, Xq)

    F = eigen(Symmetric((Σ + Σ') / 2))

    p = sortperm(F.values, rev=true)
    λ = F.values[p]
    Q = F.vectors[:, p]

    cum = cumsum(λ) / sum(λ)
    Kcut = findfirst(>(1 - tol), cum)
    Kcut === nothing && (Kcut = length(λ))

    λc = λ[1:Kcut]
    Qc = Q[:, 1:Kcut]

    z = randn(Kcut)
    sample = μ .+ Qc * (sqrt.(λc) .* z)

    return sample, Kcut
end