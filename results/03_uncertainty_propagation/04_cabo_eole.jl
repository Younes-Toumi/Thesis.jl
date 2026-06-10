# ==============================================================================
# eole_sampling.jl
# GPR conditioning sampling scheme via EOLE (KL expansion)
# Following Hong et al. (2021) "Collaborative and Adaptive BO"
#
# Core formula — Matheron's update:
#
#   f*(w) = μ_GP(w)  +  h(w)  −  k(w, X)(K + δI)⁻¹ h(X)
#
# where:
#   μ_GP(w)          = GP posterior mean at w  (already conditions on training data)
#   h(w)             = k(w, X) V_r (ξ / √λ_r)   EOLE approximation of a prior sample
#   K + δI           = prior kernel matrix at X PLUS regularisation
#   X                = GP training inputs  ← MUST equal the fitted GP training set
#
# The EOLE approximation replaces the true KL eigenfunction integrals with the
# discrete eigenvectors of K evaluated on the training set, retaining the r
# leading modes that explain ≥ energy_threshold of total variance.
# ==============================================================================

using LinearAlgebra, Statistics, Random

# ==============================================================================
# 1.  Cache struct
#     Built once per GP update; reused for all v queries in one CABO iteration.
# ==============================================================================


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
using FastGaussQuadrature

φ(z) = pdf(Normal(), z)
Φ(z)   = cdf(Normal(), z)
Φ⁻¹(p) = quantile(Normal(), p)
φ_vec(u) = prod(pdf.(Normal(), u))   # φ(u) = ∏ N(u_i; 0,1)

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

function augmented_to_epistemic(v, specs)

    relaxed_bounds = [compute_relaxed_bounds(s.θ_L, s.θ_U; v_L=s.v_L, v_U=s.v_U) for s in specs]

    n_epi        = length(specs)
    θ        = Vector{Float64}(undef, n_epi)

    for j in 1:n_epi
        lb, ub = relaxed_bounds[j]
        θ_j    = v_to_θ(v[j], lb, ub)    
        θ[j] = θ_j

    end
    return θ
end


σ = 0.1

specs = [
    EpistemicSpec(-1.5, 1.5, θ -> Normal(θ, σ), v_L=-2.2, v_U=2.2),
    EpistemicSpec(-1.5, 1.5, θ -> Normal(θ, σ), v_L=-2.2, v_U=2.2),
]

const v1_LOWER, v1_UPPER =  -2.2, 2.2
const v2_LOWER, v2_UPPER =  -2.2, 2.2

lb_v = [v1_LOWER, v2_LOWER]
ub_v = [v1_UPPER, v2_UPPER]

const bounds_v = boxconstraints(lb = lb_v, ub = ub_v)


# bounds for the BC PSO are simply the effective support of a standard normal
const u1_LOWER, u1_UPPER =  -4.0, 4.0
const u2_LOWER, u2_UPPER =  -4.0, 4.0

lb_u = [u1_LOWER, u2_LOWER]
ub_u = [u1_UPPER, u2_UPPER]

const bounds_u = boxconstraints(lb = lb_u, ub = ub_u)


physical_model = g_function

n_ale = 2
n_epi = 2
w_names = [:u1, :u2, :v1, :v2]
u_names = [:u1, :u2]
v_names = [:v1, :v2]

n_train, n_test = 20, 1001

data_aug_train, data_phys_train =    build_augmented_design(physical_model, specs, n_train; seed=42)
data_aug_test,  data_phys_test  =    build_augmented_design(physical_model, specs, n_test; seed=123)

# initialize GP on θ-space
kernel() = GPMatern52() + GPSquaredExponential()
metamodel = GaussianProcess(data_aug_train, :y, kernel_type=kernel())
@time "fit!" fit!(metamodel)

μ_test, σ_test = @time "predict:" predict(metamodel, Matrix(data_aug_test[:, w_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")

function make_pso(; N::Int=50, ω=0.8, C1=2.0, C2=2.0)
    p = PSO(N=N, ω=ω, C1=C1, C2=C2)
    return p
end


function estimate_qoi(qoi_type, gp_sample, u, v)
    Nx, _  = size(u)

    X = hcat(u, repeat(v', Nx, 1))
    μ_gp, _ = gp_sample(X)

    if qoi_type == :mean
        qoi = 1/Nx * sum(μ_gp)

    elseif qoi_type == :var
        qoi = 1/(Nx - 1) * sum((μ_gp .- 1/Nx .* sum(μ_gp)).^2)

    elseif qoi_type == :pf
        qoi = 1/(Nx) * sum(μ_gp .< 0)
    end

    return qoi
end


function estimate_propagation_qoi(qoi_type, gp_samples, u, v)

    μ_qoi, σ_qoi = nothing, nothing
    qoi = Vector{Float64}(undef, Ng)

    for j in 1:Ng
        gp_sample = gp_samples[j]
        qoi[j] = estimate_qoi(qoi_type, gp_sample, u, v)
    end

    μ_qoi = 1/Ng        * sum(qoi)
    σ_qoi = 1/(Ng - 1)  * sum((qoi .- μ_qoi).^2)

    return μ_qoi, σ_qoi
end


function ei_objective(qoi_type, gp_samples, u, v, μ_qoi_star)
    
    L_bo_sum = 0

    for j in 1:Ng
        gp_sample = gp_samples[j]
        qoi = estimate_qoi(qoi_type, gp_sample, u, v)
        L_bo_sum += max(μ_qoi_star - qoi, 0)
    end

    L_bo = 1/Ng * L_bo_sum
    
    return L_bo
end



"""
    h_pvc(gp, u, v_plus, V_GH) → Float64
 
h(u, v+) = ∫ k_post((u,v+),(u′,v+)) φ(u′) du′
         ≈ Σ_j w_j · k_post((u,v+),(u_j^GH,v+))
 
V_GH from precompute_bc avoids recomputing Cholesky back-solves inside PSO.
"""
function h_pvc(gp, cholK, u::AbstractVector, v_plus::AbstractVector)
    
    h_init = 0
    for i in 1:Nx
        # estimating cov...
        cov_i = nothing
        w_j    = [GH_NODES[j, :]; v_plus]
        k_prior = kern(w_q, w_j)
        cov_j  = k_prior - dot(v_q, V_GH[:, j])
        h     += cov_i
    end
    return h
end


function pvc_objective(gp, cholK, W, u::AbstractVector, v_plus::AbstractVector)
    h   = h_pvc(gp, cholK, W, u, v_plus, V_GH)
    phi = φ_vec(u)
    return -(h * phi)
end




# # # # # # # # # # # # # # # # # # # # #
function cabo_loop(
    gp_init,
    data_aug_train,
    w_names;
    max_iter::Int = 20,
    direction::Symbol = :min,
    tol_BO::Float64 = 5e-3,
    tol_BC::Float64 = 2.5e-2

)

    data = copy(data_aug_train)
    gp   = gp_init
    sign_dir = (direction == :min) ? +1 : -1

    θ_history    = Vector{Vector{Float64}}()
    L_BO_history = Float64[]
    L_BC_history = Float64[]



    # TODO: generate Nx samples
    # TODO: compute the initial span
    # qoi = estimate_qoi(qoi_type, gp_sample, u, v)
    # span = maximum(qoi) - minimum(qoi)


    for iter in 1:max_iter
        # TODO: generate the Ng GPR samples
        gp_samples = nothing # spmething that uses the eole/kl 
        println("\n━━━ CABO Iteration $iter / $max_iter ━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # ════ Part 1: BO engine ═══════════════════════════════════════════════

        # 1a. Incumbent: θ* = argmin [μ_qoi + α σ_qoi] or argmax [μ_qoi + α σ_qoi]:
        # extracting the incubent from the data, as suggested by the papers, and not use a PSO optimization
        v_data = Matrix(data[:, v_names])
        n_samples, _ = size(data)

        μ_qoi = Vector{Float64}(undef, n_samples)
        σ_qoi = Vector{Float64}(undef, n_samples)

        for i in 1:n_samples
            μ_val, σ_val = estimate_propagation_qoi(gp_samples, v_data[i, :])
            μ_qoi[i] = μ_val
            σ_qoi[i] = σ_val
        end
            
        candidates = μ_qoi .+ 1.0 .* σ_qoi # α = 1.0
        v_star_index = (direction == :min) ? argmin(candidates) : argmax(candidates)

        v_star = Vector(data[v_star_index, v_names])        
        μ_qoi_star = μ_qoi[v_star_index]
        σ_qoi_star = σ_qoi[v_star_index]

        θ_star = augmented_to_epistemic(v_star, specs)

        println("    Incumbent  θ* = $(round.(θ_star, digits=3))" *
                "    μ_qoi(θ*) ≈ $(round(μ_qoi_star, digits=3))" * 
                "    σ_qoi(θ*) ≈ $(round(σ_qoi_star, digits=3))"
        )
 
        # 1b. v⁺ = argmax EI(v)
        res_v = Metaheuristics.optimize(
            v -> EI_objective_eole(v, gp_samples, μ_qoi_star),
            bounds_v,
            make_pso()
        )

        v_plus = minimizer(res_v)
        θ_plus = augmented_to_epistemic(v_plus, specs)        

        L_BO   = -minimum(res_v)
        μ_qoi_plus, σ_qoi_plus = estimate_propagation_qoi(gp, v_plus)
        COV_plus = σ_qoi_plus / abs(μ_qoi_plus)

        println("     Acquisition θ⁺ = $(round.(θ_plus, digits=4))    AEI = $(round(L_BO/span, digits=4))" *
                "    COV = $(round(COV_plus, sigdigits=4))")


        if L_BO/span < tol_BO && COV_plus < tol_BC
            println("\n✓ converged")
            break
        end

        # ════ Part 2: BC engine ═══════════════════════════════════════════════
        W = Matrix(data[:, w_names])
        m = size(W, 1)
        K = zeros(m, m)
        # TODO: USE kernel_posterior here!
        for i in 1:m
            for j in 1:m
                K[i, j] = gp.kernel_prior(W[i,:], W[j,:])
            end
        end

        cholK = cholesky(Symmetric(K + 1e-8I))

        V_GH = precompute_bc(gp, cholK, W, v_plus)   # one Cholesky solve per GH node
        res_u  = Metaheuristics.optimize(
            u -> my_BC_objective(gp, cholK, W, u, v_plus, V_GH),
            bounds_u,
            make_pso()
        )

        u_plus        = minimizer(res_u)
 
        u1_plus, u2_plus = u_plus
        v1_plus, v2_plus = v_plus

        w_plus = vcat(u_plus, v_plus)
        x_plus = augmented_to_physical(w_plus, specs)
        y_plus = physical_model(x_plus...)

        append!(data, DataFrame(
            w_names[1] => [u1_plus],
            w_names[2] => [u2_plus],
            w_names[3] => [v1_plus],
            w_names[4] => [v2_plus],            
            :y         => [y_plus],
        ))

        gp = GaussianProcess(data, :y, kernel_type = kernel())
        fit!(gp)

        push!(θ_history, copy(collect(θ_plus)))
        push!(L_BO_history, L_BO/span)
        push!(L_BC_history, COV_plus)

    end

    eole = build_eole(gp, Matrix(data[:, w_names]), n_ale)


    res_bound = Metaheuristics.optimize(
        v -> sign_dir * my_BO_objective(gp, eole, v),
        bounds_v,
        make_pso()
    )
    v_bound  = minimizer(res_bound)
    μ_bound, _ = estimate_propagation_eole(gp, eole, v_bound)
    dir_str = uppercase(string(direction))

    θ_bound = augmented_to_epistemic(v_bound, specs)        


    println("\n  ► $(dir_str) bound ≈ $(round(μ_bound, sigdigits=5))" *
            "  at  θ = $(round.(θ_bound, digits=4))")
 

    
    return (
        gp = gp,
        data = data,
        θ_bound = θ_bound,
        μ_bound = μ_bound,
        θ_history = θ_history,
        L_BO_history = L_BO_history,
        L_BC_history = L_BC_history

    )
end
 
cabo_min = @time "CABO MIN" cabo_loop(
    metamodel,
    data_aug_train,
    w_names;
    max_iter = 20,
    direction = :min,
    tol_BO       = 5e-3,
    tol_BC       = 5e-2

)
 
cabo_max = @time "CABO MAX" cabo_loop(
    metamodel,
    data_aug_train,
    w_names;
    max_iter = 20,
    direction = :max,
    tol_BO       = 5e-3,
    tol_BC       = 5e-2
)


# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("CABO results")
println("="^60)
println("MIN  E[g] ≈ $(round(cabo_min.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_min.θ_bound, digits=3))")
println("MAX  E[g] ≈ $(round(cabo_max.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_max.θ_bound, digits=3))")
println("-"^60)
println("Expected:  MIN ≈ −1.35  at (−0.57, 0.52)")
println("           MAX ≈  1.33  at ( 0.55, 0.81)")
println("="^60)



# plot related
Θs_min = reduce(hcat, cabo_min.θ_history)'
Θs_max = reduce(hcat, cabo_max.θ_history)'
using Plots, Statistics

# ============================================================
# Grid over epistemic space (x1, θσ)
# ============================================================
n_μ1 = 500
n_μ2 = 500

μ1_grid = range(-1.5, 1.5, length=n_μ1)
μ2_grid  = range(-1.5, 1.5, length=n_μ2)

MeanSurface = zeros(n_μ1, n_μ2)

# ============================================================
# Compute response
# ============================================================
for (i, μ1_v) in enumerate(μ1_grid)
    for (j, μ2_v) in enumerate(μ2_grid)
        MeanSurface[j, i]  = g_function_E(μ1_v, μ2_v)
    end
end

# ============================================================
# Heatmap
# ============================================================
plt = heatmap(
    μ1_grid,
    μ2_grid,
    MeanSurface,
    xlabel="μ1",
    ylabel="μ2",
    c=:thermal,
    title="expected response function: E[g(x1, x2)]",
    colorbar=true,
    xlims = (-1.6, 1.6),
    ylims = (-1.6, 1.6),
    legend = :outerbottom,
    legendcolumns=4,
)

# ── overlay the initial training points (μ1, μ2 columns from data_aug) ──────
data_aug_train_epi = Matrix{Float64}(undef, n_train, n_epi)
for i in 1:n_train
    data_aug_train_epi[i, :] = augmented_to_epistemic(data_aug_train[i, v_names], specs)
end

scatter!(plt,
    data_aug_train_epi[:, 1], data_aug_train_epi[:, 2];
    marker = :diamond, color = :cyan, ms = 5,
    label  = "init samples", markerstrokewidth=0,
)

scatter!(plt,
    Θs_min[:, 1], Θs_min[:, 2];
    marker = :cross, color = :green, ms = 5,
    label  = "added min", markerstrokewidth=2,
)

scatter!(plt,
    Θs_max[:, 1], Θs_max[:, 2];
    marker = :cross, color = :red, ms = 5,
    label  = "added max", markerstrokewidth=2,
)

scatter!(plt,
    [cabo_min.θ_bound[1]], [cabo_min.θ_bound[2]];
    marker = :star, color = :green, ms = 7,
    label  = "cabo min", markerstrokewidth=1,
)

scatter!(plt,
    [cabo_max.θ_bound[1]], [cabo_max.θ_bound[2]];
    marker = :star, color = :red, ms = 7,
    label  = "cabo max", markerstrokewidth=1,
)

# ── mark true min and max in epistemic space ──────────────────────────────────
min_idx = argmin(MeanSurface)
max_idx = argmax(MeanSurface)

x_min, y_min, z_min = μ1_grid[min_idx[2]], μ1_grid[min_idx[1]], minimum(MeanSurface)
x_max, y_max, z_max = μ1_grid[max_idx[2]], μ1_grid[max_idx[1]], maximum(MeanSurface)

dy = 0.2

scatter!(plt,
    [μ1_grid[min_idx[2]]], [μ2_grid[min_idx[1]]];
    marker = :circle, color = :green, ms = 5, label = "True min",
)
annotate!(
    x_min, y_min + dy,
    text("($(round(x_min, digits=2)), $(round(y_min, digits=2)), $(round(z_min, digits=2)))", :black, 8)
)


scatter!(plt,
    [μ1_grid[max_idx[2]]], [μ2_grid[max_idx[1]]];
    marker = :rect, color = :red, ms = 5, label = "True max",
)

annotate!(
    x_max, y_max + dy,
    text("($(round(x_max, digits=2)), $(round(y_max, digits=2)), $(round(z_max, digits=2)))", :black, 8)
)

p1 = scatter(
    cabo_min.L_BO_history,
    xlabel = "Iteration",
    ylabel = "L_BO",
    title = "MIN: L_BO History",
    legend = false
)

p2 = scatter(
    cabo_min.L_BC_history,
    xlabel = "Iteration",
    ylabel = "L_BC",
    title = "MIN: L_BC History",
    ylims = (0, 1),
    legend = false
)

p3 = scatter(
    cabo_max.L_BO_history,
    xlabel = "Iteration",
    ylabel = "L_BO",
    title = "MAX: L_BO History",
    legend = false
)

p4 = scatter(
    cabo_max.L_BC_history,
    xlabel = "Iteration",
    ylabel = "L_BC",
    title = "MAX: L_BC History",
    ylims = (0, 1),
    legend = false
)

history = plot(
    p1, p2, p3, p4,
    layout = (2, 2),
    size = (900, 700)
)

display(plt)
display(history)
