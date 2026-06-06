# ==============================================================================
# augmented_space.jl
# SNS augmented space construction for CABO
# Following Hong et al. (2021) and Wei et al. (2021)
#
# The Rosenblatt chain:
#   θ  (interval)  →  p(θ) auxiliary  →  v = T(θ)   (SNS, v ∈ (v_L, v_U))
#   x  (given θ)   →  F_{x|θ}(x)     →  u = S(x|θ) (SNS)
#   Augmented point: w = (u, v),  y = g(x(u,v))
# ==============================================================================

using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Distributions
using ParameterHandling
using LinearAlgebra
using Metaheuristics
using QuasiMonteCarlo

Φ(z)   = cdf(Normal(), z)
Φ⁻¹(p) = quantile(Normal(), p)

# ==============================================================================
# 1.  Relaxed bounds for the auxiliary density p(θ) ~ Uniform(lb, ub)
# ==============================================================================
#
# We want:  Φ⁻¹(F_p(θ_L)) = v_L   and   Φ⁻¹(F_p(θ_U)) = v_U
#
# Solving:  (θ_L - lb)/(ub - lb) = αL = Φ(v_L)
#           (θ_U - lb)/(ub - lb) = αU = Φ(v_U)
#
# Gives:    span = (θ_U - θ_L) / (αU - αL)       (= ub - lb)
#           lb   = θ_L - αL * span                 (extend BELOW θ_L)
#           ub   = θ_U + (1 - αU) * span           (extend ABOVE θ_U)
#
# For symmetric v_L = -v_U:  αL = 1 - αU  →  lb extension = ub extension = ε
#   ε = αL * (θ_U - θ_L) / (αU - αL)
#   e.g. v_L=-2.2, v_U=2.2, θ∈[-π,π]:  ε ≈ 0.0143 * 2π ≈ 0.0898

"""
    compute_relaxed_bounds(θ_L, θ_U; v_L=-2.2, v_U=2.2)

Return (lb, ub) for p(θ) ~ Uniform(lb, ub) such that
  Φ⁻¹(F_p(θ_L)) = v_L  and  Φ⁻¹(F_p(θ_U)) = v_U.

Paper recommends |v_L|, |v_U| ∈ (1.5, 2.2) so that the original interval
is well-covered while tails beyond ±2.2 contribute negligible probability.
"""
function compute_relaxed_bounds(θ_L::Real, θ_U::Real; v_L::Real=-2.2, v_U::Real=2.2)
    L    = Float64(θ_U - θ_L)
    span = L / (Φ(v_U) - Φ(v_L))
    
    lb   = Float64(θ_L) -        Φ(v_L)  * span
    ub   = Float64(θ_U) + (1.0 - Φ(v_U)) * span

    return lb, ub
end

# ==============================================================================
# 2.  θ ↔ v transforms (using CDF of auxiliary density, NOT pdf)
# ==============================================================================

"""
    θ_to_v(θ, lb, ub)  →  v ∈ ℝ

Map epistemic variable θ to SNS via the auxiliary CDF.
"""
θ_to_v(θ, lb, ub) = Φ⁻¹((θ - lb) / (ub - lb))

"""
    v_to_θ(v, lb, ub)  →  θ ∈ [θ_L, θ_U] approximately

Inverse: θ = lb + (ub - lb) * Φ(v)
"""
v_to_θ(v, lb, ub) = lb + (ub - lb) * Φ(v)

# ==============================================================================
# 3.  x ↔ u transforms (using CDF of x|θ, NOT pdf; x ≠ θ as argument)
# ==============================================================================

"""
    x_to_u(x, dist_x_given_θ)  →  u ∈ ℝ

Map physical variable x to SNS via the conditional CDF F_{x|θ}(x).
  u = Φ⁻¹(F_{x|θ}(x))

"""
x_to_u(x, dist) = Φ⁻¹(cdf(dist, x))

"""
    u_to_x(u, dist_x_given_θ)  →  x

Inverse: x = F⁻¹_{x|θ}(Φ(u))
"""
u_to_x(u, dist) = quantile(dist, Φ(u))

# ==============================================================================
# 4.  Build augmented design
# ==============================================================================

"""
    EpistemicSpec

Specification for one epistemic variable θ_i.
  bounds = (θ_L, θ_U)       — the ORIGINAL interval (not relaxed)
  v_L, v_U                  — SNS boundaries (recommend ±2.2)
  dist_factory               — θ_i -> Distribution for x_i | θ_i
                               Set to `nothing` for pure-interval inputs
                               (x_i = θ_i, no aleatory component).
"""
struct EpistemicSpec
    θ_L          :: Float64
    θ_U          :: Float64
    v_L          :: Float64
    v_U          :: Float64
    dist_factory :: Union{Function, Nothing}   # θ -> Distributions.Distribution
end

# Convenience constructor with default SNS bounds
EpistemicSpec(θ_L, θ_U, dist_factory=nothing; v_L=-2.2, v_U=2.2) =
    EpistemicSpec(Float64(θ_L), Float64(θ_U), Float64(v_L), Float64(v_U), dist_factory)


"""
    build_augmented_design(physical_model, specs, n_samples;
                           seed=42, y_symbol=:y) → DataFrame

Generate augmented training data for CABO.

For each spec with dist_factory = nothing   (pure interval):
  • θ_i is sampled from the relaxed auxiliary uniform
  • v_i = Φ⁻¹(F_p(θ_i))
  • No u_i column (x_i = θ_i is passed directly to the model)

For each spec with dist_factory = f         (hybrid aleatory+epistemic):
  • θ_i is sampled from the relaxed auxiliary uniform → v_i
  • x_i is sampled from f(θ_i) → u_i = Φ⁻¹(F_{x|θ_i}(x_i))

physical_model :: (x1, x2, ...) -> Float64

Returned DataFrame columns:  [u1, u2, ...], [v1, v2, ...], y
(u columns only appear for specs with a dist_factory)
"""
function build_augmented_design(
    physical_model,
    specs     :: Vector{EpistemicSpec},
    n_samples :: Int;
    seed      :: Int    = 42,
    y_symbol  :: Symbol = :y
)
    n_epi = length(specs)

    # Relaxed bounds for each spec: ok
    relaxed = [compute_relaxed_bounds(s.θ_L, s.θ_U; v_L=s.v_L, v_U=s.v_U) for s in specs]
    relaxed_lbs     = [r[1] for r in relaxed]
    relaxed_ubs     = [r[2] for r in relaxed]


    # LHS sampling of θ in the relaxed interval: ok
    Random.seed!(seed)
    θ_samples = QuasiMonteCarlo.sample(n_samples, relaxed_lbs, relaxed_ubs, LatinHypercubeSample())' # n_samples × n_epi

    # Compute v for all specs: ok
    v_samples = similar(θ_samples)
    for j in 1:n_epi
        v_samples[:, j] = θ_to_v.(θ_samples[:, j], relaxed_lbs[j], relaxed_ubs[j])
    end

    # For hybrid specs: sample x|θ and compute u: ok
    hybrid_idx = findall(s -> s.dist_factory !== nothing, specs) # finds the index of inputs having Fx|θ(x)
    u_samples      = Matrix{Float64}(undef, n_samples, length(hybrid_idx))
    x_samples     = similar(θ_samples)             # physical inputs (x or θ for pure interval)

    for j in 1:n_epi
        s = specs[j]
        
        # if dist_factory is nothing -> x is an interval so we have Fx(x) and no θ dependence: ok
        if s.dist_factory === nothing
            x_samples[:, j] = θ_samples[:, j] # Pure interval: x_i = θ_i
        
        # now we have Fx|θ(x) and θ dependence: ok
        else
            # Hybrid: sample x_i from f(x_i | θ_i) and transform to u_i
            u_col = findfirst(==(j), hybrid_idx)

            for i in 1:n_samples
                # we select sampled θ, pass it to Fx|θ(X) and sample x
                θ_ij  = θ_samples[i, j]
                dist  = s.dist_factory(θ_ij)
                x_ij  = rand(dist)
                x_samples[i, j]    = x_ij
                u_samples[i, u_col] = x_to_u(x_ij, dist)            
            end
        end
    end

    # Evaluate physical model at x: ok
    y = [physical_model(x_samples[i, :]...) for i in 1:n_samples]


    # Build Augmented DataFrame: w = (u, v) = (u1, u2, ..., un, v1, v2, ..., vn): ok
    aug_df, phys_df = DataFrame(), DataFrame()
    u_col = 1
    for j in 1:n_epi
        # starting with aleatory u1, u2, ...
        if specs[j].dist_factory !== nothing
            aug_df[!, Symbol("u$j")] = u_samples[:, u_col]
            u_col += 1
        end
    end

    # and then v1, v2, ...
    for j in 1:n_epi
        aug_df[!, Symbol("v$j")] = v_samples[:, j]
        phys_df[!, Symbol("x$j")] = x_samples[:, j]
    end

    aug_df[!, y_symbol] = y
    phys_df[!, y_symbol] = y

    return aug_df, phys_df
end

# ==============================================================================
# 5.  Prediction helper: reconstruct x from (u, v) for a single point
# ==============================================================================

"""
    augmented_to_physical(w, specs, relaxed_bounds) → x_vector

Given a point w = (u..., v...) in the augmented space, recover the physical x.
"""
function augmented_to_physical(w, specs)

    relaxed_bounds = [compute_relaxed_bounds(s.θ_L, s.θ_U; v_L=s.v_L, v_U=s.v_U) for s in specs]


    n_epi        = length(specs)
    u_offset = count(s -> s.dist_factory !== nothing, specs)
    u_idx    = 0
    x        = Vector{Float64}(undef, n_epi)
    for j in 1:n_epi
        lb, ub = relaxed_bounds[j]
        v_j    = w[u_offset + j]
        θ_j    = v_to_θ(v_j, lb, ub)
        s      = specs[j]

        if s.dist_factory === nothing
            x[j] = θ_j

        else
            u_idx += 1
            u_j    = w[u_idx]
            dist   = s.dist_factory(θ_j)
            x[j]   = u_to_x(u_j, dist)
        end
    end
    return x
end









# ── Case A: pure interval (no aleatory) ───────────────────────────────────────
# x_i = θ_i ∈ [−π, π], dist_factory = nothing
# Augmented space: w = (v1, v2, v3)
# G(v) = ishigami(v_to_θ(v1), v_to_θ(v2), v_to_θ(v3))
# This is global min/max optimization in v-space.

specs = [
    EpistemicSpec(-π, π, nothing),    # x1 = θ1 (pure interval)
    EpistemicSpec(-π, π, nothing),    # x2 = θ2
    EpistemicSpec(-π, π, nothing),    # x3 = θ3
]

# other examples
# σ = 0.1
# specs = [
#     EpistemicSpec(-π, π, θ -> Normal(θ, σ)),
#     EpistemicSpec(-π, π, θ -> Normal(θ, σ)),
#     EpistemicSpec(-π, π, θ -> Normal(θ, σ)),
# ]


const v1_LOWER, v1_UPPER =  -2.2, 2.2
const v2_LOWER, v2_UPPER =  -2.2, 2.2
const v3_LOWER, v3_UPPER =  -2.2, 2.2


# bounds must be bovconstraints
lb = [v1_LOWER, v2_LOWER, v3_LOWER]
ub = [v1_UPPER, v2_UPPER, v3_UPPER]

const bounds_v = boxconstraints(lb = lb, ub = ub)


physical_model = ishigami

w_names = [:v1, :v2, :v3]

n_train, n_test = 30, 1001

data_aug_train, data_phys_train =    build_augmented_design(physical_model, specs, n_train; seed=42)
data_aug_test,  data_phys_test  =    build_augmented_design(physical_model, specs, n_test; seed=123)

# initialize GP on θ-space
kernel() = GPMatern52() + GPSquaredExponential()
metamodel = GaussianProcess(data_aug_train, :y, kernel_type=kernel())
@time "fit!" fit!(metamodel)

μ_test, σ_test = @time "predict:" predict(metamodel, Matrix(data_aug_test[:, w_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")

function make_pso(; N::Int=80, iters::Int=200, ω=0.8, C1=2.0, C2=2.0)
    p = PSO(N=N, C1=C1, C2=C2, ω=ω)
    p.options.iterations = iters
    return p
end



# # # # # # # # # # # # # # # # # # # # #
function cabo_loop(
    gp_init,
    data_aug_train,
    w_names;
    max_iter::Int = 20,
    direction::Symbol = :min,
    tol::Float64 = 1e-8
)

    data = copy(data_aug_train)
    gp   = gp_init
    sign_dir = (direction == :min) ? +1 : -1

    θ_history    = Vector{Vector{Float64}}()
    L_BO_history = Float64[]

    for iter in 1:max_iter

        println("\n━━━ CABO Iteration $iter / $max_iter ━━━")

        # ─────────────────────────────────────────────
        # BO step: optimize surrogate directly in x-space
        # ─────────────────────────────────────────────

        # so I need to compute from data the current best guess
        μ_M, σ_M = predict(gp, Matrix(data[:, w_names]))
        
        candidates = μ_M .+ 1.0 .* σ_M # α = 1.0
        v_star_index = (direction == :min) ? argmin(candidates) : argmax(candidates)

        v_star = Vector(data[v_star_index, w_names])
        
        θ_star = augmented_to_physical(v_star, specs)
        
        y_star = physical_model(θ_star...)

        println("  incumbent θ* = $(round.(θ_star, digits=4))  y ≈ $(round(y_star, digits=5))")


        μ_best, σ_best = predict(gp, reshape(v_star, 1,:))
        # ─────────────────────────────────────────────
        # acquisition step (still valid if GP uncertainty used)
        # ─────────────────────────────────────────────

        res_plus = Metaheuristics.optimize(
            v -> AEI_objective_direct(gp, v, μ_best[1], sign_dir),
            bounds_v,
            make_pso()
        )

        v_plus = minimizer(res_plus)
        v1_plus, v2_plus, v3_plus = v_plus
        θ_plus = augmented_to_physical(v_plus, specs)        
        y_plus = physical_model(θ_plus...)


        L_BO   = -minimum(res_plus)

        println("  acquisition θ⁺ = $(round.(θ_plus, digits=4)) AEI = $(round(L_BO, digits=4))")

        # ─────────────────────────────────────────────
        # true evaluation
        # ─────────────────────────────────────────────

        append!(data, DataFrame(
            w_names[1] => [v1_plus],
            w_names[2] => [v2_plus],
            w_names[3] => [v3_plus],
            :y         => [y_plus],
        ))

        gp = GaussianProcess(data, :y, kernel_type = kernel())
        fit!(gp)

        push!(θ_history, copy(collect(θ_plus)))
        push!(L_BO_history, L_BO)

        if L_BO < tol
            println("\n✓ converged")
            break
        end
    end

    result_bound = Metaheuristics.optimize(
        v -> sign_dir * (predict(gp, reshape(v,1,:))[1])[1],
        bounds_v,
        make_pso(N=50, iters=200)
    )

    v_bound = minimizer(result_bound)
    θ_bound = augmented_to_physical(v_bound, specs)        

    y_bound = (predict(gp, reshape(v_bound, 1, :))[1])[1]

    dir_str = uppercase(string(direction))
    println("\n  ► $(dir_str) bound ≈ $(round(y_bound, sigdigits=5))" *
            "  at  θ = [$(round(θ_bound[1],digits=4)), $(round(θ_bound[2],digits=4)), $(round(θ_bound[3],digits=4))]")
 
    return (
        gp = gp,
        data = data,
        θ_bound = θ_bound,
        y_bound = y_bound,
        θ_history = θ_history,
        L_BO_history = L_BO_history
    )
end
 
cabo_min = @time "CABO MIN" cabo_loop(
    metamodel,
    data_aug_train,
    w_names;
    max_iter = 20,
    direction = :min,
    tol       = 1e-5
)
 
cabo_max = @time "CABO MAX" cabo_loop(
    metamodel,
    data_aug_train,
    w_names;
    max_iter = 20,
    direction = :max,
    tol       = 1e-5,
)
 
# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("CABO results")
println("="^60)
println("MIN  g ≈ $(round(cabo_min.y_bound, digits=4))" *
        "  at x = $(round.(cabo_min.θ_bound, digits=3))")
println("MAX  g ≈ $(round(cabo_max.y_bound, digits=4))" *
        "  at x = $(round.(cabo_max.θ_bound, digits=3))")
println("-"^60)
println("="^60)






# plot related
xs_min = reduce(hcat, cabo_min.θ_history)'
xs_max = reduce(hcat, cabo_max.θ_history)'
using Plots, Statistics

# ============================================================
# Grid over epistemic space (x1, θσ)
# ============================================================
n_points = 100

x1_grid = range(-pi, pi, length=n_points)
x2_grid = range(-pi, pi, length=n_points)
x3_grid = range(-pi, pi, length=n_points)


ResponseSurface = zeros(n_points, n_points, n_points)

# ============================================================
# Compute response
# ============================================================
for (i, x1_v) in enumerate(x1_grid)
    for (j, x2_v) in enumerate(x2_grid)
        for (k, x3_v) in enumerate(x3_grid)
            ResponseSurface[k, j, i]  = physical_model(x1_v, x2_v, x3_v)
        end
    end
end


# ── mark true min and max in epistemic space ──────────────────────────────────
min_idx = argmin(ResponseSurface)
max_idx = argmax(ResponseSurface)

x1_min, x2_min, x3_min, y_min = x1_grid[min_idx[3]], x2_grid[min_idx[2]], x3_grid[min_idx[1]], minimum(ResponseSurface)
x1_max, x2_max, x3_max, y_max = x1_grid[max_idx[3]], x2_grid[max_idx[2]], x3_grid[max_idx[1]], maximum(ResponseSurface)


println("\n" * "="^60)
println("Analytical results")
println("="^60)
println("MIN  g ≈ $(round(y_min, digits=4))" *
        "  at x = $(round.([x1_min, x2_min, x3_min], digits=3))")
println("MAX  g ≈ $(round(y_max, digits=4))" *
        "  at x = $(round.([x1_max, x2_max, x3_max], digits=3))")
println("-"^60)
println("="^60)

p1 = scatter(
    cabo_min.L_BO_history,
    xlabel = "Iteration",
    ylabel = "L_AEI",
    title = "MIN: L_AEI History",
    legend = false
)

p2 = scatter(
    cabo_max.L_BO_history,
    xlabel = "Iteration",
    ylabel = "L_AEI",
    title = "MAX: L_AEI History",
    legend = false
)

history = plot(
    p1, p2,
    layout = (1, 2),
    size = (900, 500)
)

display(history)
