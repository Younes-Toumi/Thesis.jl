# ============================================================
# bases/LegendreBasis.jl
# ============================================================
# Legendre polynomials P_n(x)
# Natural distribution: Uniform[-1,1]
# Domain: [-1, 1]
# Orthogonality weight: w(x) = 1/2
#
# Use for: Uniform inputs
# Recurrence: P_n(x) = ((2n-1)·x·P_{n-1}(x) - (n-1)·P_{n-2}(x)) / n
#   P_0(x) = 1
#   P_1(x) = x
#   P_2(x) = (3x² - 1) / 2
#   P_3(x) = (5x³ - 3x) / 2
# ============================================================

struct LegendreBasis <: AbstractPCEBasis
    normalize::Bool
end

LegendreBasis() = LegendreBasis(true)

basis_name(::LegendreBasis) = "Legendre (Uniform)"

# ── 1D evaluation via recurrence ─────────────────────────────

"""
    evaluate(::LegendreBasis, x, n) -> Real

Evaluates the nth Legendre polynomial at x ∈ [-1,1].
Uses the three-term recurrence for numerical stability.
Normalized so that E[ψₙ²] = 1 under Uniform[-1,1].
"""
function evaluate(b::LegendreBasis, x::Real, n::Int)
    val = _P(x, n)
    return b.normalize ? val * sqrt(2n + 1) : val
end

function _P(x::Real, n::Int)
    n == 0 && return one(x)
    n == 1 && return x

    P_prev, P_curr = one(x), x
    for i in 2:n
        P_prev, P_curr = P_curr, ((2i-1) * x * P_curr - (i-1) * P_prev) / i
    end
    return P_curr
end

# ── Domain mapping ────────────────────────────────────────────

"""
    map_to_domain(::LegendreBasis, x) -> Vector

Maps from standard normal space to Uniform[-1,1].
Uses the probability integral transform:
    x ~ Normal(0,1) → Φ(x) ~ Uniform[0,1] → 2Φ(x)-1 ~ Uniform[-1,1]
"""
function map_to_domain(::LegendreBasis, x::Vector)
    return quantile.(Uniform(-1, 1), cdf.(Normal(), x))
end

# ── Gauss-Legendre quadrature ─────────────────────────────────

"""
    quadrature_nodes(::LegendreBasis, n) -> Vector

Returns n Gauss-Legendre quadrature nodes on [-1,1].
"""
function quadrature_nodes(::LegendreBasis, n::Int)
    x, _ = gausslegendre(n)
    return x
end

"""
    quadrature_weights(::LegendreBasis, n) -> Vector

Returns n normalised Gauss-Legendre quadrature weights.
Normalised so Σwᵢ = 1 (consistent with Uniform[-1,1] density).
"""
function quadrature_weights(::LegendreBasis, n::Int)
    _, w = gausslegendre(n)
    return w ./ 2   # gausslegendre returns weights summing to 2
end