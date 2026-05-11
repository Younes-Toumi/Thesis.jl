# ============================================================
# degrees/HyperbolicCross.jl
# ============================================================
# ∏(αᵢ + 1) ≤ p + 1
# Sparse — excludes high-order interaction terms.
# Good for high dimensions where interactions are weak.
# Fewer terms than TD for same p.
# ============================================================

struct HyperbolicCross <: AbstractPCEDegree
    p::Int
    HyperbolicCross(p::Int) = p >= 0 ? new(p) : error("degree p must be ≥ 0")
end

degree_name(d::HyperbolicCross) = "Hyperbolic Cross p=$(d.p) (HC)"

isadmissible(idx::Vector{Int}, deg::HyperbolicCross) = prod(idx .+ 1) <= deg.p + 1