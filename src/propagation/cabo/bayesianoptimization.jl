# ──────────────────────────────────────────────────────────────────────────────
# bayesianoptimization.jl  –  BO engine for CABO
# ──────────────────────────────────────────────────────────────────────────────

φ(z) = pdf(Normal(), z)
Φ(z) = cdf(Normal(), z)


using FastGaussQuadrature   # add to Project.toml: FastGaussQuadrature

"""
    make_gh_nodes(m::Int) → (u_nodes, weights)

Build tensorised 2-D Gauss-Hermite quadrature in u-space.
m nodes per dimension → m² total points.
The nodes are converted from z-space (GH is defined for N(0,1))
to u-space (Φ(z)) to match the GP input encoding.
"""
function make_gh_nodes(m::Int = 7)
    t1d, w1d = gausshermite(m)          # physicist convention: Σwᵢf(tᵢ) ≈ ∫f(t)exp(−t²)dt

    z1d = t1d .* sqrt(2)                # N(0,1) nodes  (scale by √2)
    w1d = w1d ./ sqrt(π)               # N(0,1) weights (divide by √π,  NO exp term)
    # sanity: sum(w1d) should equal 1.0 exactly

    u_nodes = Matrix{Float64}(undef, m^2, 2)
    weights  = Vector{Float64}(undef, m^2)
    idx = 1
    for i in 1:m, j in 1:m
        u_nodes[idx, 1] = cdf(Normal(), z1d[i])
        u_nodes[idx, 2] = cdf(Normal(), z1d[j])
        weights[idx]    = w1d[i] * w1d[j]
        idx += 1
    end
    return u_nodes, weights
end

const GH_NODES, GH_WEIGHTS = make_gh_nodes(7)   # 49 deterministic points

"""
    estimate_propagation_gh(gp, Θμ1, Θμ2)

Gauss-Hermite version of estimate_propagation.
No Nx argument needed — nodes are fixed and deterministic.
"""
function estimate_propagation_gh(gp, Θμ1, Θμ2)
    Np  = size(GH_NODES, 1)
    X   = hcat(GH_NODES[:, 1],
               GH_NODES[:, 2],
               fill(Θμ1, Np),
               fill(Θμ2, Np))
    μ, σ = predict(gp, X)
    w    = GH_WEIGHTS

    μ_M  = dot(w, μ)                               # weighted mean
    # Weighted law of total variance
    σ_ep2 = dot(w, σ .^ 2)                         # E_z[σ²_GP]  epistemic
    σ_al2 = dot(w, (μ .- μ_M) .^ 2)               # Var_z[μ_GP] aleatoric
    σ_M2  = max(0.0, σ_ep2 + σ_al2)

    return μ_M, σ_M2
end

"""
    estimate_propagation(gp, u1, u2, Θμ1, Θμ2, Nx)

MC estimate of the mean and total variance of E_z[g(z, θ)] at a given θ:

  μ_M  =  E_z[ μ_GP(z, θ) ]                                (posterior mean)
  σ_M² =  E_z[ σ²_GP(z, θ) ] + Var_z[ μ_GP(z, θ) ]       (total uncertainty)

The second term decomposes as:
  - E_z[σ²_GP]   : epistemic uncertainty (limited training data)
  - Var_z[μ_GP]  : aleatory variability of g across z

Both are estimated from Nx MC / LHS samples u1, u2 ∈ [0,1].
"""
function estimate_propagation(gp, u1, u2, Θμ1, Θμ2, Nx)
    X   = hcat(u1, u2, fill(Θμ1, Nx), fill(Θμ2, Nx))
    μ, σ = predict(gp, X)
    μ_M  = mean(μ)
    σ_M2 = max(0.0, mean(σ .^ 2) + var(μ))   # guard against FP rounding < 0
    return μ_M, σ_M2
end


"""
    bo_incumbent_objective_response(gp, θ, u, Nx) → μ_M

Return the **pure posterior-mean** estimate of E_z[g(z, θ)].

BUG FIX (was: `μ_M + α·√σ_M2`):
  The incumbent θ* must reflect the current *best-known* estimate of
  the objective, not an optimistic UCB/LCB.  Adding α·σ biases the
  search toward uncertain regions, making μ_M_star inaccurate and
  corrupting every AEI call that follows.

Usage: multiply externally by +1 (minimisation) or −1 (maximisation)
before passing as the PSO objective.
"""
function bo_incumbent_objective_response(gp, θ, u, Nx)
    u1, u2   = u[1], u[2]
    # μ_M, σ_M   = estimate_propagation(gp, u1, u2, θ[1], θ[2], Nx)
    μ_M, σ_M = estimate_propagation_gh(gp, θ[1], θ[2])
    return μ_M
end


"""
    AEI_objective(gp, θ, u, Nx, μ_M_star, sign_dir) → −AEI

Augmented Expected Improvement for CABO.
  sign_dir = +1  →  minimisation   EI = E[ max(η*  − Y(θ), 0) ]
  sign_dir = −1  →  maximisation   EI = E[ max(Y(θ) − η*,  0) ]

Returns **−AEI** (≤ 0) so PSO (a minimiser) effectively maximises it.

BUG FIX – maximisation branch (was: sign·(μ_M_star−μ_M)·Φ(z) with z=(μ_M_star−μ_M)/σ_M):
  For maximisation the z-score must be (μ_M − μ_M_star)/σ_M (positive
  when improvement is likely). Using the minimisation z-score with a sign
  flip gives Φ(negative z) < 0.5 for all promising points, which drives
  the AEI toward zero exactly where we want it to be large – causing the
  observed stagnation where every iteration returns the same θ.

Correct closed-form EI for each direction:
  MIN: (η* − μ_M)·Φ((η* − μ_M)/σ_M) + σ_M·φ((η* − μ_M)/σ_M)
  MAX: (μ_M − η*)·Φ((μ_M − η*)/σ_M) + σ_M·φ((μ_M − η*)/σ_M)
"""
function AEI_objective(gp, θ, u, Nx, μ_M_star, sign_dir)
    u1, u2    = u[1], u[2]
    # μ_M, σ_M2 = estimate_propagation(gp, u1, u2, θ[1], θ[2], Nx)
    μ_M, σ_M2 = estimate_propagation_gh(gp, θ[1], θ[2])
    σ_M       = sqrt(max(σ_M2, 1e-12))

    σ_M < 1e-12 && return 0.0      # numerically flat region → no gain

    if sign_dir == 1                # ── minimisation ──────────────────────
        z   = (μ_M_star - μ_M) / σ_M
        aei = (μ_M_star - μ_M) * Φ(z) + σ_M * φ(z)
    else                            # ── maximisation ──────────────────────
        z   = (μ_M - μ_M_star) / σ_M   # ← sign-flipped z  (the critical fix)
        aei = (μ_M - μ_M_star) * Φ(z) + σ_M * φ(z)
    end

    return -aei     # PSO minimises → return −AEI (always ≤ 0)
end