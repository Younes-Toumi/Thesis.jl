abstract type AbstractMean end

function build_mean(m::AbstractMean)
    error("build_mean not implemented for mean $(typeof(m)). ")
end

mean_name(k::AbstractMean) = string(typeof(m))  # fallback
