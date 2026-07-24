struct GPMatern52 <: AbstractGPKernel end

kernel_name(::GPMatern52) = "Matérn 5/2"
 
default_θ(::GPMatern52, X::Matrix, y::Vector) = default_ard_θ(X, y)
default_θ0(::GPMatern52, X::Matrix, y::Vector) = default_ard_θ0(X, y)

function build_kernel(::GPMatern52, θ::NamedTuple)
    return θ.variance * with_lengthscale(Matern52Kernel(), θ.lengthscale)
end
 
