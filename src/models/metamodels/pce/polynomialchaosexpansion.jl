mutable struct PolynomialChaosExpansion <: UQModel
    X::Matrix{Float64}
    y::Vector{Float64}

    y_symbol::Symbol
    x_names::Vector{Symbol}
end

function PolynomialChaosExpansion(
    data::DataFrame,
    y_symbol::Symbol,
)

    x_names = propertynames(data[:, Not(y_symbol)])

    X = Matrix(data[:, x_names])
    y = Vector(data[:, y_symbol])

    println("Called from Thesis.jl")

    return PolynomialChaosExpansion(
        X,
        y,
        y_symbol,
        x_names,
    )
end


function fit!(pce::PolynomialChaosExpansion)
    return pce
end

function predict(pce::PolynomialChaosExpansion, Xnew::DataFrame)
 
    Xmat = Matrix(Xnew[:, gp.x_names])
    Xvec = [Xmat[i, :] for i in 1:size(Xmat, 1)]
    return nothing
end


function evaluate!(
    pce       ::PolynomialChaosExpansion,
    data     ::DataFrame;
    mode     ::Symbol = :mean,
    n_samples::Int    = 1
)
    return nothing

end