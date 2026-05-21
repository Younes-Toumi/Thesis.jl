# ============================================================
# degrees/TotalDegree.jl
# ============================================================
# |α| = α₁ + α₂ + ... + αd ≤ p
# Standard choice — good balance of accuracy and cost.
# Number of terms: C(d+p, p) = (d+p)! / (d! p!)
# ============================================================

struct TotalDegree <: AbstractPCEDegree
    p::Int
    TotalDegree(p::Int) = p >= 0 ? new(p) : error("degree p must be ≥ 0")
end

degree_name(d::TotalDegree) = "Total Degree p=$(d.p) (TD)"

isadmissible(idx::Vector{Int}, deg::TotalDegree) = sum(idx) <= deg.p