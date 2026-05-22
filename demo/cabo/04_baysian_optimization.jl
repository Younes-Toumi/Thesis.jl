using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Distributions
using LinearAlgebra
using Statistics          # FIX 1: mean, var
using Printf              # FIX 2: @printf

Random.seed!(42)

# ============================================================
# Epistemic domain bounds  (FIX 3: define before PSO uses them)
# ============================================================
const X3_LB, X3_UB = -0.5,  1.3   # x3 interval
const TH_LB, TH_UB = -1.3,  1.8   # θ_μ p-box parameter

# ============================================================
# Feature column names expected by the GP
# ============================================================
const X_names = [:x1, :u2, :x3, :θ_μ]

# ============================================================
# True model and analytical variance (for validation only)
# ============================================================
analytical_model(x1, x2, x3) =
    x1 .* (x2.^2 .+ x2 .+ cos.(π .* x3) .- 7)

analytical_variance(x3, μ) =
    μ.^4 .+ 2μ.^3 .+ 2μ.^2 .* cos.(π.*x3) .+ 11μ.^2 .+
    2μ .* cos.(π.*x3) .+ 10μ .+ cos.(π.*x3).^2 .- 6cos.(π.*x3) .+ 45

# ============================================================
# Augmented-space helpers
# ============================================================
const σ_FIXED = 2.0

inverse_cdf_x2(u2, θ_μ; σ=σ_FIXED) = quantile.(Normal.(θ_μ, σ), u2)

function mc_augmented(n::Int)
    pts   = rand(n, 4)
    x1_raw = quantile.(Normal(0, 1), pts[:, 1])
    u2_raw = pts[:, 2]
    x3_raw = X3_LB .+ (X3_UB - X3_LB) .* pts[:, 3]
    θμ_raw = TH_LB .+ (TH_UB - TH_LB) .* pts[:, 4]
    return x1_raw, u2_raw, x3_raw, θμ_raw
end

# ============================================================
# Initial training design D₀
# ============================================================
n_train = 200

x1_train, u2_train, x3_train, θ_μ_train = mc_augmented(n_train)
x2_train = inverse_cdf_x2(u2_train, θ_μ_train)
y_train  = analytical_model(x1_train, x2_train, x3_train)

data_aug_train = DataFrame(
    :x1  => x1_train,
    :u2  => u2_train,
    :x3  => x3_train,
    :θ_μ => θ_μ_train,
    :y   => y_train,
)

metamodel = GaussianProcess(data_aug_train, :y, kernel=GPMatern52())
@time "fit!" fit!(metamodel)

# ============================================================
# Quick accuracy check on a held-out test set
# ============================================================
n_test = 1000
x1_test, u2_test, x3_test, θ_μ_test = mc_augmented(n_test)
x2_test  = inverse_cdf_x2(u2_test, θ_μ_test)
y_test_v = analytical_model(x1_test, x2_test, x3_test)

data_aug_test = DataFrame(
    :x1  => x1_test,
    :u2  => u2_test,
    :x3  => x3_test,
    :θ_μ => θ_μ_test,
    :y   => y_test_v,
)

μ_test, σ_test = predict(metamodel, data_aug_test[:, X_names])

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2( data_aug_test.y, μ_test), digits=5))")

# ============================================================
# estimate_variance
#
#   V̂(x3, θ) = Var_{x1,u2} [ μ_GP(x1, u2, x3, θ) ]
#
# Draws Nx aleatory samples, queries GP mean, returns sample variance.
# ============================================================
function estimate_variance(gp, x3::Float64, θμ::Float64, Nx::Int)
    x1_s = randn(Nx)
    u2_s = rand(Nx)

    df = DataFrame(
        :x1  => x1_s,
        :u2  => u2_s,
        :x3  => fill(x3, Nx),
        :θ_μ => fill(θμ, Nx),
    )

    # FIX 4: predict returns (μ, σ) — unpack, keep only mean
    μ_preds, _ = predict(gp, df[:, X_names])

    return var(μ_preds)
end

# ============================================================
# variance_moments_mcs
#
# Samples Ng epistemic points and collects V̂ at each.
# Returns the mean, std, and the full sample vector.
# ============================================================
function variance_moments_mcs(gp, Ng::Int; Nx::Int=500)
    x3_epi = X3_LB .+ (X3_UB - X3_LB) .* rand(Ng)
    θ_epi  = TH_LB .+ (TH_UB - TH_LB) .* rand(Ng)

    V_samples = [estimate_variance(gp, x3_epi[i], θ_epi[i], Nx)
                 for i in 1:Ng]

    μ_V = mean(V_samples)
    σ_V = std(V_samples)      # std, not sqrt(var): same thing, one call

    return μ_V, σ_V, V_samples
end

# ============================================================
# EI_v — Expected Improvement acquisition  (FIX 6: was missing)
#
#   EI_v(x3*, θ*) = (1/Ng) Σᵢ max( V̂(x3*, θ*) − V̂ᵢ, 0 )
#
# V_samples is the fixed reference vector from variance_moments_mcs.
# PSO calls this many times; the maximum it finds is L_v^BO.
# ============================================================
function EI_v(gp, x3_star::Float64, θ_star::Float64,
              V_samples::Vector{Float64}; Nx::Int=500)

    V_star = estimate_variance(gp, x3_star, θ_star, Nx)
    return mean(max.(V_star .- V_samples, 0.0))
end

# ============================================================
# pso_maximize
#
# Maximises a scalar f(x3, θ) over [X3_LB,X3_UB]×[TH_LB,TH_UB].
# Returns (x3_opt, θ_opt, f_max).
# f_max is what the paper calls L_v^BO.
# ============================================================
function pso_maximize(f::Function;
                      n_particles::Int = 30,
                      max_iter::Int    = 100,
                      w::Float64       = 0.72,
                      c1::Float64      = 1.49,
                      c2::Float64      = 1.49)

    # FIX 5: θ lower bound was 1.3 — must be TH_LB = -1.3
    pos = hcat(
        X3_LB .+ (X3_UB - X3_LB) .* rand(n_particles),
        TH_LB .+ (TH_UB - TH_LB) .* rand(n_particles),
    )   # n_particles × 2

    vel       = zeros(n_particles, 2)
    f_vals    = [f(pos[i,1], pos[i,2]) for i in 1:n_particles]

    pbest_pos = copy(pos)
    pbest_val = copy(f_vals)

    gbest_idx = argmax(pbest_val)
    gbest_pos = copy(pbest_pos[gbest_idx, :])
    gbest_val = pbest_val[gbest_idx]

    for _ in 1:max_iter
        for i in 1:n_particles
            r1, r2 = rand(2), rand(2)

            vel[i, :] = w  .* vel[i, :] .+
                         c1 .* r1 .* (pbest_pos[i, :] .- pos[i, :]) .+
                         c2 .* r2 .* (gbest_pos        .- pos[i, :])

            pos[i, :] .+= vel[i, :]
            pos[i, 1]   = clamp(pos[i, 1], X3_LB, X3_UB)
            pos[i, 2]   = clamp(pos[i, 2], TH_LB, TH_UB)

            fval = f(pos[i, 1], pos[i, 2])

            if fval > pbest_val[i]
                pbest_val[i]    = fval
                pbest_pos[i, :] = pos[i, :]
            end
            if fval > gbest_val
                gbest_val = fval
                gbest_pos = copy(pos[i, :])
            end
        end
    end

    return gbest_pos[1], gbest_pos[2], gbest_val   # x3*, θ*, L_v^BO
end

# ============================================================
# cabo_loop — full CABO Bayesian Optimisation
#
# Each iteration:
#   1. variance_moments_mcs  → μ_V, σ_V, V_samples
#   2. PSO maximises EI_v    → x3*, θ*,  L_v^BO
#   3. δ_BO = L_v^BO / σ_V  → stop if < δ_tol
#   4. Evaluate true model at n_new points near (x3*, θ*)
#   5. Refit GP
# ============================================================
function cabo_loop(gp_init, data_train::DataFrame;
                   Ng::Int          = 200,
                   Nx::Int          = 500,
                   δ_tol::Float64   = 0.01,
                   max_iter::Int    = 30,
                   n_new::Int       = 5,
                   n_particles::Int = 30,
                   pso_iter::Int    = 100)

    gp   = gp_init
    data = copy(data_train)

    V_history = Float64[]
    δ_history = Float64[]

    for iter in 1:max_iter
        println("\n── BO Iteration $iter / $max_iter " * "─"^28)

        # Step 1: reference variance distribution
        μ_V, σ_V, V_samples = variance_moments_mcs(gp, Ng; Nx=Nx)
        @printf("  μ_V = %.5f,  σ_V = %.5f\n", μ_V, σ_V)

        # Step 2: PSO → L_v^BO and next epistemic candidate
        acq = (x3, θ) -> EI_v(gp, x3, θ, V_samples; Nx=Nx)

        x3_star, θ_star, L_v_BO_val = pso_maximize(acq;
                                            n_particles = n_particles,
                                            max_iter    = pso_iter)

        @printf("  x3* = %+.4f,  θ* = %+.4f,  L_v^BO = %.6f\n",
                x3_star, θ_star, L_v_BO_val)

        # Step 3: δ_BO convergence criterion
        δ_BO = L_v_BO_val / σ_V
        push!(δ_history, δ_BO)
        @printf("  δ_BO = %.6f  (threshold %.4f)\n", δ_BO, δ_tol)

        if δ_BO < δ_tol
            println("  ✓ Converged at iteration $iter")
            break
        end

        # Step 4: true model calls at (x3*, θ*)
        x1_new = randn(n_new)
        u2_new = rand(n_new)
        x2_new = inverse_cdf_x2(u2_new, θ_star)
        y_new  = analytical_model(x1_new, x2_new, fill(x3_star, n_new))

        append!(data, DataFrame(
            :x1  => x1_new,
            :u2  => u2_new,
            :x3  => fill(x3_star, n_new),
            :θ_μ => fill(θ_star,  n_new),
            :y   => y_new,
        ))

        # Step 5: refit GP on expanded data
        gp = GaussianProcess(data, :y, kernel=GPMatern52())
        fit!(gp)

        V_star = estimate_variance(gp, x3_star, θ_star, Nx)
        push!(V_history, V_star)
        @printf("  V̂(x3*, θ*) = %.6f   [n_train = %d]\n", V_star, nrow(data))
    end

    V_upper  = maximum(V_history)
    idx_best = argmax(V_history)

    println("\n" * "═"^50)
    @printf("  Variance upper bound : %.6f\n", V_upper)
    @printf("  Found at iteration   : %d\n",   idx_best)
    println("═"^50)

    return gp, data, V_history, δ_history
end

# ============================================================
# Run
# ============================================================
gp_bo, data_bo, V_history, δ_history = cabo_loop(
    metamodel,
    data_aug_train;
    Ng          = 1000,    # epistemic MC samples per BO step
    Nx          = 10000,    # aleatory MC samples inside estimate_variance
    δ_tol       = 0.01,   # convergence threshold
    max_iter    = 30,     # hard cap
    n_new       = 5,      # true-model calls added per iteration
    n_particles = 30,     # PSO swarm size
    pso_iter    = 100,    # PSO iterations per BO step
)

# ── Results ─────────────────────────────────────────────────
V_upper = maximum(V_history)
println("\nVariance upper bound: $V_upper")

# Validate against analytical solution at the found worst-case epistemic point
# (scan a grid for the analytical maximum — only possible because we have a closed form)
x3_grid = range(X3_LB, X3_UB; length=200)
θ_grid  = range(TH_LB, TH_UB; length=200)
V_analytical_max = maximum(analytical_variance(x3, θ) for x3 in x3_grid, θ in θ_grid)
println("Analytical maximum:   $V_analytical_max")
println("Relative error:       $(round(abs(V_upper - V_analytical_max)/V_analytical_max*100, digits=2))%")