# Least Absolute Shrinkage and Selection Operator

mutable struct LASSOSolver <: AbstractPCESolver
    λ        :: Union{Float64, Nothing}  # nothing = auto-select via LOO-CV
    tol      :: Float64
    max_steps:: Int
end

solver_name(::LASSOSolver) = "LASSO"


# Convenience constructor with sensible defaults
function LASSOSolver(; λ=nothing, tol=1e-8, max_steps=500)
    return LASSOSolver(λ, tol, max_steps)
end


# Step 2 — this is all we write next
function soft_threshold(z::Float64, λ::Float64)::Float64
    return sign(z) * max(abs(z) - λ, 0)
end

function lambda_grid(A_n::Matrix, y::Vector; n_lambdas::Int=500, eps::Float64=1e-8)
    
    λ_max = maximum(abs.(A_n' * y))
    λ_min = λ_max * eps
    
    return exp.(range(log(λ_max), log(λ_min), length=n_lambdas))
end

function coordinate_descent(
    A_n::Matrix, 
    y::Vector, 
    λ::Float64;
    c_init::Vector{Float64} = zeros(size(A_n, 2)),
    tol::Float64=1e-8, max_steps::Int=500)

    N, P = size(A_n)
    c = copy(c_init)          # start from provided guess instead of zeros
    r = y - A_n * c           # residual consistent with c_init

    for _ in 1:max_steps
        c_old = copy(c)
        for j in 1:P
            aj = @view A_n[:, j]
            r .+= aj .* c[j]
            ρ = dot(aj, r)
            c[j] = j == 1 ? ρ / dot(aj, aj) : soft_threshold(ρ, λ)
            r .-= aj .* c[j]
        end
        maximum(abs.(c - c_old)) < tol && break
    end

    return c
end

function solve_path(A_n::Matrix, y::Vector, lambdas::Vector{Float64};
                    tol::Float64=1e-8, max_steps::Int=500)

    N, P = size(A_n)
    # store one coefficient vector per λ
    coef_path = zeros(P, length(lambdas))

    for (i, λ) in enumerate(lambdas)
        # warm start: initialize from previous solution
        c_init = i == 1 ? zeros(P) : copy(coef_path[:, i-1]) # <- warm start
        coef_path[:, i] = coordinate_descent(A_n, y, λ; c_init=c_init, tol=tol, max_steps=max_steps)

    end

    return coef_path  # P × n_lambdas matrix
end


function loo_error(A_n::Matrix, y::Vector, c::Vector; tol::Float64=1e-8)
    N = length(y)
    active = findall(abs.(c) .> tol)
    isempty(active) && return 1.0

    A_active = A_n[:, active]

    # Re-fit OLS on the active set so the analytical LOO formula is EXACT.
    # (LASSO selects the support; OLS estimates the coefficients on it.)
    c_ols = A_active \ y                      # least-squares on selected columns

    Q = Matrix(qr(A_active).Q)
    h = vec(sum(Q .^ 2, dims=2))
    r = y .- A_active * c_ols                 # OLS residual — matches the leverage formula

    denom = 1.0 .- h
    denom[denom .< tol] .= tol
    loo_res = r ./ denom
    return mean(loo_res .^ 2) / var(y)
end

function solve(solver::LASSOSolver, A::Matrix, y::Vector)
    
    _, P = size(A) # P number of coefficients
    
    # step 1: normalize columns for fair penalty
    col_norms = vec(sqrt.(sum(A .^ 2, dims=1)))
    col_norms[col_norms .< solver.tol] .= 1.0   # avoid division by zero
    A_n = A ./ col_norms'                       # normalized design matrix
   

    # --- Fixed λ: just run coordinate descent once ---
    if !isnothing(solver.λ)
        c = coordinate_descent(A_n, y, solver.λ, tol=solver.tol, max_steps=solver.max_steps)
        c[abs.(c) .< solver.tol] .= 0.0
        return c ./ col_norms
    end



    # --- Auto λ: run full path + LOO-CV ---
    lambdas   = lambda_grid(A_n, y)           # 1. build λ grid
    coef_path = solve_path(A_n, y, lambdas, tol=solver.tol, max_steps=solver.max_steps) # 2. solve across grid (warm starts)
    
    # 3. compute LOO error for each λ, pick best
    errors = [loo_error(A_n, y, coef_path[:, i]) for i in 1:length(lambdas)]
    best_i = argmin(errors)

    # 4. extract best coefficients, clean up near-zeros, denormalize
    c_best = coef_path[:, best_i]

    c_best[abs.(c_best) .< solver.tol] .= 0.0
    
    solver.λ = lambdas[best_i] 
    return c_best ./ col_norms
end