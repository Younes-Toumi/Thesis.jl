mutable struct PolynomialChaosExpansion <: UQModel
    X::Matrix{Float64}
    y::Vector{Float64}

    y_symbol::Symbol
    x_names::Vector{Symbol}

    bases::Vector{<:AbstractPCEBasis}
    indices::Vector{Vector{Int}}

    solver::AbstractPCESolver
    coeffs::Union{Nothing, Vector{Float64}}
end

function PolynomialChaosExpansion(
    data::DataFrame,
    y_symbol::Symbol,
    bases::Vector{<:AbstractPCEBasis},
    degree::AbstractPCEDegree;
    solver::AbstractPCESolver = OLSSolver()
)

    x_names = propertynames(data[:, Not(y_symbol)])

    X = Matrix(data[:, x_names])
    y = Vector(data[:, y_symbol])

    d = length(bases)
    indices = multivariate_indices(degree, d)

    return PolynomialChaosExpansion(
        X,
        y,
        y_symbol,
        x_names,
        bases,
        indices,
        solver,
        nothing
    )
end


function fit!(pce::PolynomialChaosExpansion)
    A = build_design_matrix(pce.bases, pce.indices, pce.X)
    println(size(A))
    pce.coeffs = solve(pce.solver, A, pce.y)

    return pce
end

function predict(pce::PolynomialChaosExpansion, Xnew::DataFrame)

    Xmat = Matrix(Xnew[:, pce.x_names])

    A_pred = build_design_matrix(pce.bases, pce.indices, Xmat)

    return A_pred * pce.coeffs
end


function evaluate!(
    pce::PolynomialChaosExpansion,
    data::DataFrame
)
    pce.coeffs === nothing && error("Call fit!(pce) first.")

    y_pred = predict(pce, data)

    data[!, pce.y_symbol] = y_pred

    return data
end