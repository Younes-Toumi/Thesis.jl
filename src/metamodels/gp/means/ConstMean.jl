struct GPConstMean <: AbstractGPMean end

mean_name(::GPConstMean) = "Constant Mean"
build_mean(::GPConstMean, X::Matrix, y::Vector) = AbstractGPs.ConstMean(mean(y))
default_mean_θ(::GPConstMean, ::Matrix, y::Vector) = (c = mean(y))