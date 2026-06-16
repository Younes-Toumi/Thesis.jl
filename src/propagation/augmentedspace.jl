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
    InputSpec

Specification for one epistemic variable θ_i.
  bounds = (θ_L, θ_U)       — the ORIGINAL interval (not relaxed)
  v_L, v_U                  — SNS boundaries (recommend ±2.2)
  dist_factory               — θ_i -> Distribution for x_i | θ_i
                               Set to `nothing` for pure-interval inputs
                               (x_i = θ_i, no aleatory component).
"""
struct InputSpec
    θ_L          :: Float64
    θ_U          :: Float64
    v_L          :: Float64
    v_U          :: Float64
    dist_factory :: Union{Function, Nothing}   # θ -> Distributions.Distribution
end

# Convenience constructor with default SNS bounds
InputSpec(θ_L, θ_U, dist_factory=nothing; v_L=-2.2, v_U=2.2) =
    InputSpec(Float64(θ_L), Float64(θ_U), Float64(v_L), Float64(v_U), dist_factory)


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
    specs     :: Vector{InputSpec},
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


    if physical_model !== nothing
        # Evaluate physical model at x: ok
        y = [physical_model(x_samples[i, :]...) for i in 1:n_samples]

        aug_df[!, y_symbol] = y
        phys_df[!, y_symbol] = y
    end

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