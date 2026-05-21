# ============================================================
# degrees/TensorProduct.jl
# ============================================================
# max(α₁, ..., αd) ≤ p
# Includes ALL combinations up to degree p per dimension.
# Most expensive — grows as (p+1)^d. Only use for small d.
# ============================================================

struct TensorProduct <: AbstractPCEDegree
    p::Int
    TensorProduct(p::Int) = p >= 0 ? new(p) : error("degree p must be ≥ 0")
end

degree_name(d::TensorProduct) = "Tensor Product p=$(d.p) (TP)"

isadmissible(idx::Vector{Int}, deg::TensorProduct) = maximum(idx) <= deg.p