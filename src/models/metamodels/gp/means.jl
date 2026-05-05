abstract type AbstractGPMean end

function build_mean(m::AbstractGPMean)
    error("build_mean not implemented for mean $(typeof(m)). ")
end

mean_name(m::AbstractGPMean) = string(typeof(m))  # fallback
