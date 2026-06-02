struct GPMatern12 <: AbstractGPKernel end

kernel_name(::GPMatern12) = "Matérn 1/2"
 
default_θ(::GPMatern12, X::Matrix, y::Vector) = default_ard_θ(X, y)

function build_kernel(::GPMatern12, θ::NamedTuple)
    return θ.variance * with_lengthscale(Matern12Kernel(), θ.lengthscale)
end
