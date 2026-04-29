struct GPCompositeKernel <: AbstractGPKernel
    kernels ::Vector{AbstractGPKernel}
    op      ::Symbol   # :sum or :product
end

# ── Convenience constructors ──────────────────────────────────
Base.:+(a::AbstractGPKernel, b::AbstractGPKernel) = GPCompositeKernel([a, b], :sum)
Base.:*(a::AbstractGPKernel, b::AbstractGPKernel) = GPCompositeKernel([a, b], :product)

kernel_name(k::GPCompositeKernel) = join(kernel_name.(k.kernels),
    k.op === :sum ? " + " : " × ")

# ── default_θ: nested NamedTuple instead of flat prefixed keys ─
function default_θ(k::GPCompositeKernel, X::Matrix, y::Vector)
    # nested: (k1 = (...), k2 = (...))
    # Mooncake handles nested NamedTuples fine
    vals = Tuple(default_θ(ki, X, y) for ki in k.kernels)
    keys = Tuple(Symbol("k$i") for i in 1:length(k.kernels))
    return NamedTuple{keys}(vals)
end

# ── build_kernel: index directly, no string ops ───────────────

function build_kernel(k::GPCompositeKernel, θ::NamedTuple)
    built = map(enumerate(k.kernels)) do (i, ki)
        θi = θ[Symbol("k$i")]   # direct index — no startswith, no string()
        build_kernel(ki, θi)
    end

    return k.op === :sum ? reduce(+, built) : reduce(*, built)
end