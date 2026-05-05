# ============================================================
# degrees/degrees.jl
# ============================================================

abstract type AbstractPCEDegree end

# ── Interface guards ──────────────────────────────────────────
"""
    isadmissible(idx::Vector{Int}, deg::AbstractPCEDegree) -> Bool

Returns true if the multi-index `idx` belongs to the index set defined by `deg`.
Must be implemented by every concrete degree type.
"""
function isadmissible(idx::Vector{Int}, deg::AbstractPCEDegree)
    error("isadmissible not implemented for $(typeof(deg))")
end

degree_name(::AbstractPCEDegree) = "Unknown Degree"

"""
    n_terms(deg::AbstractPCEDegree, d::Int) -> Int

Returns the number of basis terms for `d` input dimensions.
Useful for determining required sample size (rule of thumb: n ≥ 2-3 × n_terms).
"""
n_terms(deg::AbstractPCEDegree, d::Int) = length(multivariate_indices(deg, d))

# ── Core index generation — dispatch on isadmissible ─────────
"""
    multivariate_indices(deg::AbstractPCEDegree, d::Int) -> Vector{Vector{Int}}

Generates all multi-indices α = (α₁,...,αd) admissible under `deg`.
The zero index (0,...,0) is always included — it corresponds to the constant term.

Works for any concrete degree type that implements `isadmissible`.
"""
function multivariate_indices(deg::AbstractPCEDegree, d::Int)
    max_size = BigInt(deg.p + 1)^d   # upper bound on iterations

    idx       = zeros(Int, d)
    index_set = [copy(idx)]

    deg.p == 0 && return index_set   # constant only

    idx[1] += 1

    for _ in 1:max_size
        if isadmissible(idx, deg)
            push!(index_set, copy(idx))
        end

        # ── increment idx like a mixed-radix counter ──────────
        carry = true
        for i in 1:d
            if carry
                idx[i] += 1
                carry    = false
            end
            if !isadmissible(idx, deg)
                idx[i] = 0
                carry   = true
            end
        end

        iszero(idx) && break
    end

    return index_set
end