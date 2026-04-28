struct Matern32 <: AbstractKernel end

kernel_name(::Matern32) = "Matérn 3/2"
 
default_θ(::Matern32, X::Matrix, y::Vector) = default_ard_θ(X, y)

function build_kernel(::Matern32, θ::NamedTuple)
    return θ.variance * with_lengthscale(Matern32Kernel(), θ.lengthscale)
end
