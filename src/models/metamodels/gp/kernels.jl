# ============================================================
# Abstract Kernel Interface
# ============================================================
# Every concrete kernel must implement:
#   - default_θ(k, X, y)   → data-driven NamedTuple of constrained params
#   - build_kernel(k, θ)   → KernelFunctions.jl kernel object
#   - kernel_name(k)       → human-readable string
#
# Adding a new kernel = one new file, zero changes elsewhere.
# ============================================================

abstract type AbstractGPKernel end

# ── Interface guards — clear error if a method is missing ─────
function default_θ(k::AbstractGPKernel, ::Matrix, ::Vector)
    error("default_θ not implemented for kernel $(typeof(k)). ")
end

function build_kernel(k::AbstractGPKernel, ::NamedTuple)
    error("build_kernel not implemented for kernel $(typeof(k)). ")
end

kernel_name(k::AbstractGPKernel) = string(typeof(k))  # fallback


# ── Shared utilities used by all kernels ─────────────────────

"""
    median_pairwise_distance(X::Matrix) -> Float64

Computes the median Euclidean distance between all pairs of rows in X.
Used as a data-driven initialisation for the lengthscale hyperparameter.
"""
function median_pairwise_distance(X::Matrix)
    n = size(X, 1)
    n < 2 && return 1.0  # fallback for tiny datasets
    dists = [norm(X[i, :] - X[j, :]) for i in 1:n for j in i+1:n]
    m = median(dists)
    return m > 0 ? m : 1.0  # guard against degenerate cases
end

"""
    default_ard_θ(X::Matrix, y::Vector) -> NamedTuple

Returns the standard Automatic Relevance Determination (ARD) hyperparameter initialisation shared by
stationary kernels (Matern32, Matern52, SqExponential):
  - lengthscale: one per input dimension, initialised to median pairwise distance
  - variance:    initialised to var(y)
  - noise:       initialised to 1% of var(y)
"""
function default_ard_θ(X::Matrix, y::Vector)
    d  = size(X, 2)
    l  = median_pairwise_distance(X)
    σ² = var(y)
    return (
        lengthscale = positive(fill(l, d)),   # Vector -> ARD
        variance    = positive(σ²), # bounded, not just positive

        # Bound noise to a small fraction of output variance.
        # Prevents the optimiser escaping into the noise = ∞, lengthscale=0 degeneracy.
        noise       = positive(0.01 * σ²),

    )
end


# ── Parameter Constraints ────────────────────────────────────────────
# Thin wrappers around ParameterHandling:
#   1. param_positive:  Constrains a parameter to (0, ∞). Use for lengthscales, variances, noise.
#   2. param_bounded:   Constrains a parameter to (lo, hi). Use for parameters with known mathematical bounds.
#   3. param_free:      Unconstrained parameter, x ∈ (-∞, ∞)

param_positive(x)        = ParameterHandling.positive(x)
param_bounded(x, lo, hi) = ParameterHandling.bounded(x, lo, hi)
param_free(x)            = x