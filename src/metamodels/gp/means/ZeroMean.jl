struct GPZeroMean <: AbstractGPMean end

mean_name(::GPZeroMean) = "Zero Mean"
build_mean(::GPZeroMean, X::Matrix, y::Vector) = AbstractGPs.ZeroMean()
default_mean_θ(::GPZeroMean, ::Matrix, ::Vector) = (;)   # empty NamedTuple