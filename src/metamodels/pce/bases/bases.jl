# ============================================================
# bases/bases.jl
# ============================================================

abstract type AbstractPCEBasis end

# ── Interface guards ──────────────────────────────────────────

"""
    evaluate(basis::AbstractPCEBasis, x::Real, n::Int) -> Real

Evaluates the nth orthogonal polynomial of `basis` at point `x`.
Must be implemented by every concrete basis type.
"""
function evaluate(b::AbstractPCEBasis, x::Real, n::Int)
    error("evaluate not implemented for $(typeof(b))")
end

"""
    map_to_domain(basis::AbstractPCEBasis, x::Vector) -> Vector

Maps input samples from standard normal space to the natural
domain of the basis polynomials.

All inputs are assumed to arrive in standard normal space
(after to_standard_normal_space! transform). Each basis
then maps to its own domain:
  Hermite  → identity (domain is already ℝ)
  Legendre → normal CDF → Uniform[-1,1]
"""
function map_to_domain(b::AbstractPCEBasis, x::Vector)
    error("map_to_domain not implemented for $(typeof(b))")
end

"""
    quadrature_nodes(basis::AbstractPCEBasis, n::Int) -> Vector

Returns n Gauss quadrature nodes for numerical integration
in the natural domain of the basis.
"""
function quadrature_nodes(b::AbstractPCEBasis, n::Int)
    error("quadrature_nodes not implemented for $(typeof(b))")
end

"""
    quadrature_weights(basis::AbstractPCEBasis, n::Int) -> Vector

Returns n Gauss quadrature weights corresponding to quadrature_nodes.
Weights are normalised so that Σwᵢ = 1.
"""
function quadrature_weights(b::AbstractPCEBasis, n::Int)
    error("quadrature_weights not implemented for $(typeof(b))")
end

basis_name(::AbstractPCEBasis) = "Unknown Basis"

# ── Shared multivariate evaluation ───────────────────────────

"""
    evaluate_basis(bases::Vector{<:AbstractPCEBasis},
                   indices::Vector{Vector{Int}},
                   x::Vector{<:Real}) -> Vector{Float64}

Evaluates all multivariate basis functions at point x.
Returns a vector of length P = length(indices).

For each multi-index α:
    Ψ_α(x) = ∏ⱼ ψ_{αⱼ}(xⱼ)

This is the core operation used to build the design matrix A.
"""
function evaluate_basis(
    bases   ::Vector{<:AbstractPCEBasis},
    indices ::Vector{Vector{Int}},
    x       ::Vector{<:Real}
)
    length(bases) == length(x) || error(
        "Number of bases ($(length(bases))) must match input dimension ($(length(x)))"
    )

    result = Vector{Float64}(undef, length(indices))

    for (i, α) in enumerate(indices)
        val = 1.0
        for (j, αⱼ) in enumerate(α)
            val *= evaluate(bases[j], x[j], αⱼ)
        end
        result[i] = val
    end

    return result
end

"""
    build_design_matrix(bases, indices, X) -> Matrix{Float64}

Builds the design matrix A ∈ ℝⁿˣᴾ where:
    Aᵢⱼ = Ψ_{αⱼ}(xᵢ) = ∏ₖ ψ_{αⱼₖ}(xᵢₖ)

n = number of training points
P = number of basis terms
"""
function build_design_matrix(
    bases   ::Vector{<:AbstractPCEBasis},
    indices ::Vector{Vector{Int}},
    X       ::Matrix{<:Real}
)
    n = size(X, 1)
    P = length(indices)
    A = Matrix{Float64}(undef, n, P)

    for i in 1:n
        A[i, :] = evaluate_basis(bases, indices, X[i, :])
    end

    return A
end