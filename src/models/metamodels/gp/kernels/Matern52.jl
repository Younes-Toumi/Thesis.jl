struct Matern52 <: AbstractKernel end

kernel_name(::Matern52) = "Matérn 5/2"
 
default_θ(::Matern52, X::Matrix, y::Vector) = default_ard_θ(X, y)

function build_kernel(::Matern52, θ::NamedTuple)
    return θ.variance * with_lengthscale(Matern52Kernel(), θ.lengthscale)
end
 
