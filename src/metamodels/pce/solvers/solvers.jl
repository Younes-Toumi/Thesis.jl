abstract type AbstractPCESolver end

function solve(::AbstractPCESolver, A::Matrix, y::Vector)
    error("solve not implemented for this pce solver")
end