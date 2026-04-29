struct GPMatern32 <: AbstractGPKernel end

kernel_name(::GPMatern32) = "Matérn 3/2"
 
default_θ(::GPMatern32, X::Matrix, y::Vector) = default_ard_θ(X, y)

function build_kernel(::GPMatern32, θ::NamedTuple)
    return θ.variance * with_lengthscale(Matern32Kernel(), θ.lengthscale)
end
