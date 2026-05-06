struct LeastSquaresSolver <: AbstractPCESolver
end

function solve(::LeastSquaresSolver, A::Matrix, y::Vector)
    return A \ y
end