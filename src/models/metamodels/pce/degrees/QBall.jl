# ============================================================
# degrees/QBall.jl
# ============================================================
# ||α||_q ≤ p   where q ∈ (0, 1]
# Generalises TD (q=1) toward sparsity (q→0).
# Smaller q → fewer terms → sparser PCE.
# Best for high-dimensional problems with sparse interactions.
# ============================================================

struct QBall <: AbstractPCEDegree
    p::Int
    q::Float64
    function QBall(p::Int, q::Float64=0.5)
        p >= 0    || error("degree p must be ≥ 0")
        0 < q <= 1 || error("q must be in (0, 1]")
        return new(p, q)
    end
end

degree_name(d::QBall) = "Q-Ball p=$(d.p) q=$(d.q) (QB)"

isadmissible(idx::Vector{Int}, deg::QBall) = norm(idx, deg.q) <= deg.p