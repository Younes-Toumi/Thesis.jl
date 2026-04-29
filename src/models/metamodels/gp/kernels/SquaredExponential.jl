struct GPSquaredExponential <: AbstractGPKernel end

kernel_name(::GPSquaredExponential) = "Squared Exponential"

default_θ(::GPSquaredExponential, X::Matrix, y::Vector) = default_ard_θ(X, y)

function build_kernel(::GPSquaredExponential, θ::NamedTuple)
    return θ.variance * with_lengthscale(SqExponentialKernel(), θ.lengthscale)
end
