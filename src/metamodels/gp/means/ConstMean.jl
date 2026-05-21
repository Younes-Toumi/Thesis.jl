struct GPConstMean <: AbstractGPMean
    learn::Bool   # true → c is optimised, false → fixed at mean(y)
end

GPConstMean() = GPConstMean(true)   # default: learn the constant

mean_name(::GPConstMean) = "Constant Mean"
has_parameters(::GPConstMean) = true

build_mean(::GPConstMean, X::Matrix, y::Vector) = AbstractGPs.ConstMean(mean(y))

default_mean_θ(::GPConstMean, ::Matrix, y::Vector) = (
    c = mean(y),   # unconstrained — mean can be anything
)