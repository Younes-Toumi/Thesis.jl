"""
    PolynomialChaosExpansion <: UQModel

A (sparse or ordinary) Polynomial Chaos Expansion surrogate.

# Fields
- `X::Matrix{Float64}`, `y::Vector{Float64}`: Training data
- `y_symbol::Symbol`, `x_names::Vector{Symbol}`: Column bookkeeping
- `bases::Vector{<:AbstractPCEBasis}`: One basis per input dimension (e.g. Hermite/Legendre)
- `indices::Vector{Vector{Int}}`: Admissible multi-indices defining the basis (from a
  `AbstractPCEDegree` truncation scheme)
- `solver::AbstractPCESolver`: `OLSSolver()` or `LASSOSolver()`
- `coeffs::Union{Nothing, Vector{Float64}}`: Fitted coefficients, or `nothing` before `fit!`
"""
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

# ============================================================
# Constructor — fixed degree
# ============================================================
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

"""
    max_affordable_degree(degree_ctor::Function, d::Int, n::Int; oversample_factor=3.0) -> Int

Returns the largest degree `p` such that `n_terms(degree_ctor(p), d) ≤ oversample_factor × n`.

This is a direct answer to "given `n` training points in `d` dimensions, how
high can the polynomial degree go before there are more coefficients than
data can support" -- with no fitting involved, just the combinatorial term count.

# Example
```julia
max_affordable_degree(p -> TotalDegree(p), 4, 50)          # TotalDegree, 4D, n=50
max_affordable_degree(p -> QBall(p, 0.5), 4, 50)            # QBall is sparser -> allows higher p
```
"""
function max_affordable_degree(degree_ctor::Function, d::Int, n::Int; oversample_factor::Float64=1.0)
    p = 0
    while n_terms(degree_ctor(p + 1), d) <= oversample_factor * n
        p += 1
    end
    return p
end

# ============================================================
# Training
# ============================================================
function fit!(pce::PolynomialChaosExpansion)
    A = build_design_matrix(pce.bases, pce.indices, pce.X)
    pce.coeffs = solve(pce.solver, A, pce.y)

    return pce
end

# ============================================================
# Prediction
# ============================================================
function predict(pce::PolynomialChaosExpansion, Xnew::AbstractMatrix{Float64}; mode=:mean)
    A_pred = build_design_matrix(pce.bases, pce.indices, Xnew)
    return A_pred * pce.coeffs
end

"""
    evaluate!(pce::PolynomialChaosExpansion, data::DataFrame; mode=:mean)

Compute PCE predictions and write them into `data[!, pce.y_symbol]` in-place.

PCE is deterministic, so `:mean` is the only supported mode (unlike the GP's
`evaluate!`, which also supports `:var`/`:mean_and_var`/`:sample` since a GP
carries predictive uncertainty that a bare PCE does not).

# See also
[`predict`](@ref) for a non-mutating version.
"""
function evaluate!(
    pce::PolynomialChaosExpansion,
    data::DataFrame;
    mode::Symbol = :mean,
)
    pce.coeffs === nothing && error("Call fit!(pce) first.")

    if mode === :mean
        Xmat = Matrix(data[:, pce.x_names])     # sliced by name, matches training column order
        data[!, pce.y_symbol] = predict(pce, Xmat)
    else
        throw(ArgumentError("Unknown mode: $mode. PolynomialChaosExpansion only supports :mean."))
    end

    return nothing
end

export
    PolynomialChaosExpansion, fit!, predict, evaluate!, max_affordable_degree
