struct GPCustomMean <: AbstractGPMean
    learn::Bool   # true → c is optimised, false → fixed at mean(y)
end

GPCustomMean() = GPCustomMean(true)   # default: learn the custom

mean_name(::GPCustomMean) = "Custom Mean"
has_parameters(::GPCustomMean) = true

function build_mean(::GPCustomMean, X::Matrix, y::Vector)
    return AbstractGPs.ZeroMean()
end 

default_mean_θ(::GPCustomMean, ::Matrix, y::Vector) = (
    c = mean(y),   # unconstrained - mean can be anything
)