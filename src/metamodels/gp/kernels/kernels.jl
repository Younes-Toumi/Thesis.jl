"""
Abstract Kernel Interface. Every concrete kernel must implement:
  - default_θ(k, X, y)   → data-driven NamedTuple of constrained params
  - build_kernel(k, θ)   → KernelFunctions.jl kernel object
  - kernel_name(k)       → human-readable string
"""

abstract type AbstractGPKernel end

# Interface guards
function default_θ(k::AbstractGPKernel, ::Matrix, ::Vector)
    error("default_θ not implemented for kernel $(typeof(k)). ")
end

function build_kernel(k::AbstractGPKernel, ::NamedTuple)
    error("build_kernel not implemented for kernel $(typeof(k)). ")
end

kernel_name(k::AbstractGPKernel) = string(typeof(k))  # fallback


# Shared utilities used by all kernels

"""
    median_pairwise_distance(X::Matrix) -> Float64
    min_pairwise_distance(X::Matrix) -> Float64
    max_pairwise_distance(X::Matrix) -> Float64


Computes the median, min and max Euclidean distance between all pairs of rows in X.
Used as a data-driven initialisation for the lengthscale hyperparameter.
"""
function median_pairwise_distance(X::Matrix)
    n = size(X, 1)
    n < 2 && return 1.0  # fallback for tiny datasets
    dists = [norm(X[i, :] - X[j, :]) for i in 1:n for j in i+1:n]
    m = median(dists)
    return m > 0 ? m : 1.0  # guard against degenerate cases
end

function min_pairwise_distance(X::Matrix)
    n = size(X,1); n < 2 && return 1.0
    m = minimum(norm(X[i,:]-X[j,:]) for i in 1:n for j in i+1:n)
    m > 0 ? m : 1.0
end
function max_pairwise_distance(X::Matrix)
    n = size(X,1); n < 2 && return 1.0
    maximum(norm(X[i,:]-X[j,:]) for i in 1:n for j in i+1:n)
end

"""
    default_ard_θ(X::Matrix, y::Vector) -> NamedTuple

Returns the standard Automatic Relevance Determination (ARD) hyperparameter initialisation shared by
stationary kernels (Materns & SqExponential):
  - lengthscale: one per input dimension, initialised to median pairwise distance
  - variance:    initialised to var(y)
  - noise:       initialised to 1e-7 (deterministic simulator)
"""
function default_ard_θ(X::Matrix, y::Vector)
    d  = size(X, 2)
    σ² = var(y)

    l = [median_pairwise_distance(X[:, j:j]) for j in 1:d]   # one l per input    

    l_min = 0.1 * min_pairwise_distance(X)  # blocks l → 0
    l_max = max_pairwise_distance(X)        # blocks l → ∞

    l = clamp.(l, l_min, l_max)
    σ² = clamp.(σ², 1e-10, 1e10) # safeguad against σ² = 0

    return (
        lengthscale = ParameterHandling.bounded(l, l_min, l_max),
        variance    = ParameterHandling.bounded(σ², 1e-12, 1e12),
        noise       = ParameterHandling.positive(1e-7),
    )
end

export 
    build_kernel,
    GPSquaredExponential, GPMatern52, GPMatern32, GPMatern12, GPCompositeKernel
    kernel_name, default_θ, default_ard_θ, build_kernel,
    median_pairwise_distance, min_pairwise_distance, max_pairwise_distance