"""
Implementation of the composite kernel logic. This allows to define:
```julia
kernel_type = GPMatern52() + GPSquaredExponential()
kernel_type = GPMatern52() * GPSquaredExponential()
kernel_type = 0.2* GPMatern52() + 0.8* GPSquaredExponential()
```
"""

struct GPCompositeKernel <: AbstractGPKernel
    kernels ::Vector{AbstractGPKernel}
    op      ::Symbol   # :sum or :product
end

# addition and multiplication of kernels
Base.:+(a::AbstractGPKernel, b::AbstractGPKernel) = GPCompositeKernel([a, b], :sum)
Base.:*(a::AbstractGPKernel, b::AbstractGPKernel) = GPCompositeKernel([a, b], :product)

kernel_name(k::GPCompositeKernel) = join(kernel_name.(k.kernels),
    k.op === :sum ? " + " : " × ")


    function default_θ(k::GPCompositeKernel, X::Matrix, y::Vector)
    vals = Tuple(default_θ(ki, X, y) for ki in k.kernels)
    keys = Tuple(Symbol("k$i") for i in 1:length(k.kernels))
    return NamedTuple{keys}(vals)
end


function build_kernel(k::GPCompositeKernel, θ::NamedTuple)
    built = map(enumerate(k.kernels)) do (i, ki)
        θi = θ[Symbol("k$i")]   # direct index — no startswith, no string()
        build_kernel(ki, θi)
    end

    return k.op === :sum ? reduce(+, built) : reduce(*, built)
end



struct ScaledGPKernel <: AbstractGPKernel
    kernel ::AbstractGPKernel
    scale  ::Float64
end

# scalar scaling of kernels
Base.:*(a::Real, k::AbstractGPKernel) = ScaledGPKernel(k, a)
Base.:*(k::AbstractGPKernel, a::Real) = ScaledGPKernel(k, a)

kernel_name(k::ScaledGPKernel) = "$(k.scale) × $(kernel_name(k.kernel))"

function default_θ(k::ScaledGPKernel, X::Matrix, y::Vector)
    # scale the variance initialisation by the scalar
    θ = default_θ(k.kernel, X, y)
    return merge(θ, (variance = positive(k.scale * ParameterHandling.value(θ.variance)),))
end

function build_kernel(k::ScaledGPKernel, θ::NamedTuple)
    return k.scale * build_kernel(k.kernel, θ)
end