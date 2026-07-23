"""
Abstract Mean Interface. Every concrete mean must implement:
  - build_mean(m)     → AbstractGps.jl mean object
  - build_mean(m, θ)   → AbstractGps.jl kernel object
  - mean_name(m)      → human-readable string
"""
abstract type AbstractGPMean end

function build_mean(m::AbstractGPMean)
    error("build_mean not implemented for mean $(typeof(m)). ")
end

mean_name(m::AbstractGPMean) = string(typeof(m))  # fallback

export
    build_mean, mean_name,
    GPZeroMean, GPConstMean