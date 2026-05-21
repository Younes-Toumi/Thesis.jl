# ============================================================
# bases/HermiteBasis.jl
# ============================================================
# Probabilist's Hermite polynomials He_n(x)
# Natural distribution: Normal(0,1)
# Domain: (-∞, ∞)
# Orthogonality weight: w(x) = exp(-x²/2) / √(2π)
#
# Use for: Normal inputs
# Recurrence: He_n(x) = x·He_{n-1}(x) - (n-1)·He_{n-2}(x)
#   He_0(x) = 1
#   He_1(x) = x
#   He_2(x) = x² - 1
#   He_3(x) = x³ - 3x
# ============================================================

struct HermiteBasis <: AbstractPCEBasis
    normalize::Bool
end

HermiteBasis() = HermiteBasis(true)

basis_name(::HermiteBasis) = "Hermite (Normal)"

# ── 1D evaluation via recurrence ─────────────────────────────

"""
    evaluate(::HermiteBasis, x, n) -> Real

Evaluates the nth probabilist's Hermite polynomial at x.
Uses the three-term recurrence for numerical stability.
Normalized so that E[ψₙ²] = 1 under Normal(0,1).
"""
function evaluate(b::HermiteBasis, x::Real, n::Int)
    val = _He(x, n)
    return b.normalize ? val / sqrt(factorial(n > 20 ? big(n) : n)) : val
end

function _He(x::Real, n::Int)
    n == 0 && return one(x)
    n == 1 && return x

    He_prev, He_curr = one(x), x
    for i in 2:n
        He_prev, He_curr = He_curr, x * He_curr - (i-1) * He_prev
    end
    return He_curr
end

# ── Domain mapping ────────────────────────────────────────────

"""
    map_to_domain(::HermiteBasis, x) -> Vector

Identity — Hermite polynomials are defined on ℝ,
and inputs arrive already in standard normal space.
"""
map_to_domain(::HermiteBasis, x::Vector) = x

# ── Gauss-Hermite quadrature ──────────────────────────────────

"""
    quadrature_nodes(::HermiteBasis, n) -> Vector

Returns n Gauss-Hermite quadrature nodes scaled for
the probabilist's convention (Normal(0,1) weight).
"""
function quadrature_nodes(::HermiteBasis, n::Int)
    x, _ = gausshermite(n)
    return sqrt(2) .* x   # physicist → probabilist scaling
end

"""
    quadrature_weights(::HermiteBasis, n) -> Vector

Returns n normalised Gauss-Hermite quadrature weights.
"""
function quadrature_weights(::HermiteBasis, n::Int)
    _, w = gausshermite(n)
    return w ./ sqrt(π)   # normalisation for Normal(0,1)
end