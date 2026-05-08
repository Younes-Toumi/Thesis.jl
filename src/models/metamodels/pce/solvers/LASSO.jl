# Least Absolute Shrinkage and Selection Operator

struct LASSOSolver <: AbstractPCESolver
    λ        :: Union{Float64, Nothing}  # nothing = auto-select via LOO-CV
    tol      :: Float64
    max_steps:: Int
end

# Convenience constructor with sensible defaults
function LASSOSolver(; λ=nothing, tol=1e-10, max_steps=500)
    return LASSOSolver(λ, tol, max_steps)
end


# Step 2 — this is all we write next
function soft_threshold(z::Float64, λ::Float64)::Float64
    return sign(z) * max(abs(z) - λ, 0)
end

function solve(solver::LASSOSolver, A::Matrix, y::Vector)
    
    _, P = size(A) # P number of coefficients
    
    # step 1: normalize columns for fair penalty
    col_norms = vec(sqrt.(sum(A .^ 2, dims=1)))
    col_norms[col_norms .< solver.tol] .= 1.0   # avoid division by zero
    A_n = A ./ col_norms'                       # normalized design matrix
   
    # step 2: initialize
    c = zeros(P)
    r = copy(y - A_n * c) # running residual: r = y - A_n * c, initially y since c=0

    # step 3: coordinate descent loop
    for _ in 1:solver.max_steps
        c_old = copy(c)
        for j in 1:P
            # add back column j's contribution to get partial residual
            r .+= A_n[:, j] .* c[j]

            # compute OLS update for c[j] alone (A_n columns have unit norm, so denominator = 1)
            z = dot(A_n[:, j], r)

            # apply soft threshold
            c[j] = soft_threshold(z, solver.λ)

            # subtract new contribution back out
            r .-= A_n[:, j] .* c[j]
        end

        # check convergence
        if maximum(abs.(c - c_old)) < solver.tol
            break
        end
    end

    # step 4: denormalize before returning
    return c ./ col_norms
end
