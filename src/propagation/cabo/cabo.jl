struct CABOModel
    d_u::Int
    d_θ::Int
    u_dist::Distribution
    θ_bounds::Vector{Tuple{Float64,Float64}}
    transform::Function   # (u, θ) → x (GP input)
    simulator::Function   # (u, θ) → y
end

function sample_UΘ(prob::CABOModel, n::Int)
    U = [rand(prob.u_dist) for _ in 1:n]

    Θ = [
        [lb + rand()*(ub - lb) for (lb, ub) in prob.θ_bounds]
        for _ in 1:n
    ]

    return U, Θ
end

function build_X(prob::CABOModel, U, Θ)
    n = length(U)
    X = Matrix{Float64}(undef, n, prob.d_u + prob.d_θ)

    for i in 1:n
        X[i, :] = vcat(U[i], Θ[i])
    end

    return X
end

function build_data(prob::CABOModel, U, Θ)
    n = length(U)

    X = build_X(prob, U, Θ)

    y = [prob.simulator(U[i], Θ[i]) for i in 1:n]

    df = DataFrame(X, :auto)
    df[!, :y] = y

    return df
end