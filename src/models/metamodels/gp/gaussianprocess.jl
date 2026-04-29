# ============================================================
# Gaussian Process surrogate model
# ============================================================

mutable struct GaussianProcess <: UQModel
    X::Matrix{Float64}
    y::Vector{Float64}

    mean:: AbstractGPMean
    kernel:: AbstractGPKernel
    θ::NamedTuple

    posterior::Union{AbstractGPs.PosteriorGP, Nothing}

    y_symbol::Symbol
    x_names::Vector{Symbol}

    unflatten
    learn_noise::Bool
end

# ============================================================
# Constructor
# ============================================================

function GaussianProcess(
    data::DataFrame,
    output::Symbol;
    mean::AbstractGPMean          = GPZeroMean(),
    kernel::AbstractGPKernel      = GPMatern52(),
    θ::Union{NamedTuple, Nothing} = nothing,
    learn_noise::Bool             = false
)

    x_names = propertynames(data[:, Not(output)])

    X = Matrix(data[:, x_names])
    y = Vector(data[:, output])

    θ0 = isnothing(θ) ? default_θ(kernel, X, y) : θ
    flat_θ0, unflatten = value_flatten(θ0)

    return GaussianProcess(
        X,
        y,
        mean,
        kernel,
        θ0,
        nothing,
        output,
        x_names,
        unflatten,
        learn_noise
    )
end


# ============================================================
# Training
# ============================================================

function fit!(gp::GaussianProcess)
    
    # TODO mean is fixed to 0 for now, maybe improuve?
    y_scale = var(gp.y)
    flat_θ0, _ = value_flatten(gp.θ)

    mean = build_mean(gp.mean, gp.X, gp.y)

    function nlml(flat_θ)
        θ = gp.unflatten(flat_θ)
        kernel = build_kernel(gp.kernel, θ)
        
        f = GP(mean, kernel)

        # TODO noise: fixed adaptive jitter for deterministic simulators learned parameter for noisy observations?

        noise = gp.learn_noise ? θ.noise : 1e-1 * y_scale

        fx = f(gp.X', noise)
        return -logpdf(fx, gp.y)
    end

    result = optimize(
        nlml,
        (g, θ) -> begin
            _, grad = DifferentiationInterface.value_and_gradient(
                nlml, AutoMooncake(; config=nothing), θ
            )
            g .= grad
        end,
        flat_θ0,
        LBFGS(),
        Optim.Options(show_trace=false, iterations=100);
        inplace=true   # ← note: true when gradient is written in-place
    )

    θ_opt = gp.unflatten(result.minimizer)

    # building posterior
    gp.θ = θ_opt
    kernel_opt = build_kernel(gp.kernel, θ_opt)
    f_opt      = GP(kernel_opt)
    noise      = gp.learn_noise ? θ_opt.noise : 1e-5   # tight jitter for posterior
    fx         = f_opt(gp.X', noise)
 
    gp.θ         = θ_opt
    gp.posterior = posterior(fx, gp.y)
 
    return gp
end


function predict(gp::GaussianProcess, Xnew::DataFrame)
    gp.posterior === nothing && error("GP has not been trained yet. Call fit!(gp) first.")
 
    Xmat = Matrix(Xnew[:, gp.x_names])
    Xvec = [Xmat[i, :] for i in 1:size(Xmat, 1)]
 
    μ, v  = mean_and_var(gp.posterior, Xvec)
    σ     = sqrt.(v)

    return μ, σ
end


function evaluate!(
    gp       ::GaussianProcess,
    data     ::DataFrame;
    mode     ::Symbol = :mean,
    n_samples::Int    = 1
)
    gp.posterior === nothing && error("Call fit!(gp) before evaluate!.")

    # ── prepare inputs ────────────────────────────────────────
    Xmat = Matrix(data[:, gp.x_names])
    Xvec = [Xmat[i, :] for i in 1:size(Xmat, 1)]

    # ── project posterior onto test points ────────────────────
    fp = gp.posterior(Xvec)

    # ── column name helpers ───────────────────────────────────
    col_mean   = Symbol(string(gp.y_symbol, "_mean"))
    col_var    = Symbol(string(gp.y_symbol, "_var"))

    if mode === :mean
        data[!, col_mean] = mean(fp)

    elseif mode === :var
        data[!, col_var] = var(fp)

    elseif mode === :mean_and_var
        data[!, col_mean] = mean(fp)
        data[!, col_var]  = var(fp)

    elseif mode === :sample
        samples = rand(fp, n_samples)   # Matrix: n_points × n_samples
        for i in 1:n_samples
            col = Symbol(string(gp.y_symbol, "_sample_", i))
            data[!, col] = samples[:, i]
        end

    else
        throw(ArgumentError("Unknown mode: $mode. Choose :mean, :var, :mean_and_var, or :sample."))
    end

    return nothing
end