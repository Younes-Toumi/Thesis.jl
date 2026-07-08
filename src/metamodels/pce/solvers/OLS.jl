# Ordinaty Least Square solver
struct OLSSolver <: AbstractPCESolver end

solver_name(::OLSSolver) = "OLS"


function solve(::OLSSolver, A::Matrix, y::Vector)
    return (A' * A) \ (A' * y)
end