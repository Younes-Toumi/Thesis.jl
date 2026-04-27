# ============================================================
# Gaussian Process surrogate model
# ============================================================

mutable struct GaussianProcess <: UQModel
    X::Matrix{Float64}
    y::Vector{Float64}

    θ::NamedTuple

    kernel
    posterior

    y_symbol::Symbol
    x_names::Vector{Symbol}

    unflatten
end

# ============================================================
# Constructor
# ============================================================

function GaussianProcess(
    data::DataFrame,
    output::Symbol,
    θ::NamedTuple
)

    x_names = propertynames(data[:, Not(output)])

    X = Matrix(data[:, x_names])
    y = Vector(data[:, output])

    flat_θ0, unflatten = value_flatten(θ)

    return GaussianProcess(
        X,
        y,
        θ,
        nothing,
        nothing,
        output,
        x_names,
        unflatten
    )
end

# ============================================================
# Kernel builder
# ============================================================

function build_kernel(θ)

    return θ.variance *
        with_lengthscale(
            Matern52Kernel(),
            θ.lengthscale
        )
end

# ============================================================
# Negative Log Marginal Likelihood
# ============================================================

function build_nlml(model)
    X = model.X'
    y = model.y
    y_scale = var(y)                    # empirical signal scale

    function nlml(flat_θ)
        θ = model.unflatten(flat_θ)
        kernel = build_kernel(θ)
        gp = GP(kernel)
        jitter = 1e-2 * y_scale         # ~0.1% of signal
        fx = gp(X, θ.noise + jitter)
        return -logpdf(fx, y)
    end
    return nlml
end

# ============================================================
# Training
# ============================================================

function fit!(model::GaussianProcess)

    flat_θ0, _ = value_flatten(model.θ)

    nlml = build_nlml(model)

    result = optimize(
        nlml,
        θ -> only(Zygote.gradient(nlml, θ)), # mooncake
        flat_θ0,
        LBFGS(),
        Optim.Options(
            show_trace = false,
            iterations = 100
        );
        inplace = false # autodiff= AutoZygote() DifferentiationInterface.jl
    )

    θ_opt = model.unflatten(result.minimizer)

    # println("\nOptimised hyperparameters:")
    # println("lengthscale = ", θ_opt.lengthscale)
    # println("variance    = ", θ_opt.variance)
    # println("noise       = ", θ_opt.noise)

    model.θ = θ_opt

    kernel = build_kernel(θ_opt)

    gp = GP(kernel)

    jitter = 1e-6
    fx = gp(model.X', θ_opt.noise + jitter)

    model.posterior = posterior(fx, model.y)

    model.kernel = kernel

    return model

end

# ============================================================
# Prediction
# ============================================================

function predict(
    model::GaussianProcess,
    Xnew::DataFrame
)

    # Convert DataFrame → Matrix
    Xmat = Matrix(Xnew[:, model.x_names])

    # Convert Matrix → Vector of vectors
    Xvec = [Xmat[i, :] for i in 1:size(Xmat, 1)]

    μ, var = mean_and_var(
        model.posterior,
        Xvec
    )

    σ = sqrt.(var)

    return μ, σ

end