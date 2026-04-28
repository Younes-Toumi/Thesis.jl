struct SquaredExponential <: AbstractKernel end

kernel_name(::SquaredExponential) = "Squared Exponential"

default_θ(::SquaredExponential, X::Matrix, y::Vector) = default_ard_θ(X, y)

function build_kernel(::SquaredExponential, θ::NamedTuple)
    return θ.variance * with_lengthscale(SqExponentialKernel(), θ.lengthscale)
end
