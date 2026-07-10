using LinearAlgebra
using BenchmarkTools

function expensive_function()
    return cond(rand(3, 3))
end

function monte_carlo_bad(N)
    results = []

    for i in 1:N
        push!(results, expensive_function())
    end

    return results
end


function monte_carlo_good(N::Int):: Vector{Float64}

    results = Vector{Float64}(undef, N)
    results = [expensive_function() for _ in 1:N]

    return results
end

res1 = monte_carlo_bad(5000);
res2 = monte_carlo_good(5000);


function tempo_bad(w)
    x = w[:, 1]
    y = w[:, 2]
    return x.^2 + y.^2
end

function tempo_good!(z:: Vector{Float64}, w::Matrix{Float64})

    x = @view w[:, 1]
    y = @view w[:, 2]

    @. z = x^2 + y^2
end

w = rand(20_000_000, 2)
z = similar(@view w[:,1])

res1 = @time tempo_bad(w);
res2 = @time tempo_good!(z, w);