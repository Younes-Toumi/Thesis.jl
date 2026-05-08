# Ordinaty Least Square solver
struct OLSSolver <: AbstractPCESolver
end

function solve(::OLSSolver, A::Matrix, y::Vector)
    return (A' * A) \ (A' * y)

end