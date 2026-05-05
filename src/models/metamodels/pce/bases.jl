struct PolynomialChaosBasis
    bases::Vector{<:AbstractOrthogonalBasis}
    degree::AbstractPCEDegree
    d::Int
    α::Vector{Vector{Int}}

    function PolynomialChaosBasis(bases::Vector{<:AbstractOrthogonalBasis},
                                 degree::AbstractPCEDegree)

        d = length(bases)
        α = multivariate_indices(degree, d)
        return new(bases, degree, d, α)
    end
end