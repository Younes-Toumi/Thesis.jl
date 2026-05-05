# ============================================================
# Gaussian Process surrogate model
# ============================================================
"""
    GaussianProcess <: UQModel

A Gaussian Process (GP) surrogate model for uncertainty quantification.

Stores training data, kernel/mean hyperparameters, and the trained posterior
distribution. Must be fitted with [`fit!`](@ref) before predictions can be made.

# Fields
- `X::Matrix{Float64}`: Training inputs of shape `(n_samples, n_dims)`
- `y::Vector{Float64}`: Training y_symbols of length `n_samples`
- `mean::AbstractGPMean`: Prior mean function (e.g. `GPZeroMean()`)
- `kernel::AbstractGPKernel`: Covariance kernel (e.g. `GPMatern52()`)
- `θ::NamedTuple`: Current kernel hyperparameters
- `posterior`: Trained GP posterior, or `nothing` if not yet fitted
- `y_symbol::Symbol`: Name of the y_symbol column in the original `DataFrame`
- `x_names::Vector{Symbol}`: Names of the input columns, used for column-safe prediction
- `unflatten`: Inverse of `value_flatten` - reconstructs a `NamedTuple` from a flat vector
- `learn_noise::Bool`: If `true`, observation noise σ² is included as a learnable hyperparameter

# See also
[`fit!`](@ref), [`predict`](@ref), [`evaluate!`](@ref)
"""
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
"""
    GaussianProcess(data, y_symbol; mean, kernel, θ, learn_noise)

Construct a `GaussianProcess` surrogate from a `DataFrame`.

Extracts inputs `X` and output `y` from `data`, initialises hyperparameters,
and prepares the model for training via [`fit!`](@ref).

# Arguments
- `data::DataFrame`: Training dataset containing both inputs and the y_symbol column
- `y_symbol::Symbol`: Name of the y_symbol column in `data`

# Keyword Arguments
- `mean::AbstractGPMean`: Prior mean function. Defaults to `GPZeroMean()`
- `kernel::AbstractGPKernel`: Covariance kernel. Defaults to `GPMatern52()`
- `θ::Union{NamedTuple, Nothing}`: Initial hyperparameters. If `nothing`,
  sensible defaults are inferred from `X` and `y` via `default_θ`. Defaults to `nothing`
- `learn_noise::Bool`: Whether to learn observation noise as a hyperparameter.
  Set to `true` for noisy simulators, `false` for deterministic ones. Defaults to `false`

# Returns
- `GaussianProcess`: An unfitted model ready to be trained with `fit!(gp)`

# Examples
```julia
gp = GaussianProcess(train_df, :y)
gp = GaussianProcess(train_df, :y; mean=GPConstMean())
gp = GaussianProcess(train_df, :y; kernel=GPSquaredExponential(), learn_noise=true)
```

# Notes
- All columns in `data` except `y_symbol` are treated as inputs
- Input scaling should be applied to `data` **before** constructing the GP
  (see your `Scalings` module)
"""
function GaussianProcess(
    data::DataFrame,
    y_symbol::Symbol;
    mean::AbstractGPMean          = GPZeroMean(),
    kernel::AbstractGPKernel      = GPMatern52(),
    θ::Union{NamedTuple, Nothing} = nothing,
    learn_noise::Bool             = false
)

    x_names = propertynames(data[:, Not(y_symbol)])

    X = Matrix(data[:, x_names])
    y = Vector(data[:, y_symbol])

    # Infer sensible starting values from data if not provided (avoids flat/degenerate regions)
    θ0 = isnothing(θ) ? default_θ(kernel, X, y) : θ

    # Optim.jl needs a flat Vector{Float64}; unflatten reconstructs the NamedTuple after optimisation
    flat_θ0, unflatten = value_flatten(θ0)

    return GaussianProcess(
        X,
        y,
        mean,
        kernel,
        θ0,
        nothing,
        y_symbol,
        x_names,
        unflatten,
        learn_noise
    )
end


# ============================================================
# Training
# ============================================================

"""
    fit!(gp::GaussianProcess)

Train the GP by minimizing the negative log marginal likelihood (LML) over the kernel
hyperparameters, then store the optimised posterior in `gp.posterior`.

Optimisation is performed with L-BFGS using exact gradients via `Mooncake.jl`
automatic differentiation.

# Arguments
- `gp::GaussianProcess`: An unfitted (or previously fitted) GP model. Modified in-place.

# Returns
- `gp::GaussianProcess`: The same object, now with `gp.θ` and `gp.posterior` updated

# Notes
- **Noise handling:**
  - If `gp.learn_noise = true`: noise variance σ² is included in `θ` and optimised
  - If `gp.learn_noise = false`: noise is set to `0.1 * var(y)` during training
    and tightened to `1e-5` for the posterior (suitable for deterministic simulators)
- Convergence is not guaranteed — check `gp.θ` for reasonable hyperparameter values
  if predictions look off

# References
- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, Ch. 5
  (marginal likelihood optimisation)

# See also
[`predict`](@ref), [`evaluate!`](@ref)
"""
function fit!(gp::GaussianProcess)
    
    y_scale = var(gp.y) # scale noise relative to output, this avoids hardcoding across problems
    flat_θ0, _ = value_flatten(gp.θ) # re-flatten in case fit! is called again after manual θ updates

    mean = build_mean(gp.mean, gp.X, gp.y)
    Xt = collect(gp.X')

    function nlml(flat_θ)
        θ = gp.unflatten(flat_θ)                # recover NamedTuple so kernel can unpack named fields
        kernel = build_kernel(gp.kernel, θ)     # kernels are immutable -> rebuild on every call
        
        f = GP(mean, kernel)

        # TODO noise: fixed adaptive jitter for deterministic simulators learned parameter for noisy observations?
        # learn_noise = true  -> σ² is a free parameter (noisy observations)
        # learn_noise = false -> adaptive jitter = 10% of output variance (deterministic simulator)
        noise = gp.learn_noise ? θ.noise : 1e-1 * y_scale

        fx = f(Xt, noise)
    
        val = -logpdf(fx, gp.y) # negative because Optim.jl minimises
        
        println(
            "nlml val: $(round.(val, digits=4)) ", 
            "lengthscale: $(round.(θ.lengthscale, digits=4)) ",
            "variance: $(round.(θ.variance, digits=4)) ",
            "noise: $(round.(θ.noise, digits=4))")

        return val

    end


    # TODO: refactor this into a seperate optimize.jl file
    # Run from default θ0 + n_restarts-1 random perturbations
    function run_optimization(flat_θ0)
        return optimize(
            nlml,
            (g, θ) -> begin
                _, grad = DifferentiationInterface.value_and_gradient(
                    nlml, AutoMooncake(; config=nothing), θ
                )
                g .= grad
            end,
            flat_θ0,
            LBFGS(),
            Optim.Options(show_trace=false, iterations=50);
            inplace=true
        )
    end

    # Run from default θ0 + n_restarts-1 random perturbations
    flat_θ0, _ = value_flatten(gp.θ)

    n_restarts = 5
    results = map(1:n_restarts) do i
        println("RUN NUMBER $i AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\n\n\n")
        θ_start = i == 1 ? flat_θ0 : flat_θ0 .+ 0.5 .* randn(length(flat_θ0))
        try
            run_optimization(θ_start)
        catch
            nothing  # skip failed starts (e.g. Cholesky errors)
        end
    end

    # Keep the result with the lowest NLML
    valid   = filter(!isnothing, results)
    isempty(valid) && error("All optimisation restarts failed.")
    best    = argmin(r -> r.minimum, valid)
    θ_opt = gp.unflatten(best.minimizer)

    ######

    # result = optimize(
    #     nlml,
    #     (g, θ) -> begin
    #         # computes nlml(θ) and ∇nlml(θ) in one backward pass
    #         _, grad = DifferentiationInterface.value_and_gradient(nlml, AutoMooncake(; config=nothing), θ)
    #         g .= grad # in-place, no new allocation
    #     end,
    #     flat_θ0,
    #     LBFGS(),
    #     Optim.Options(show_trace=false, iterations=50);
    #     inplace=true   # true when gradient is written in-place
    # )

    # θ_opt = gp.unflatten(result.minimizer)

    # Build posterior with optimised hyperparameters
    gp.θ = θ_opt
    kernel_opt = build_kernel(gp.kernel, θ_opt)
    f_opt      = GP(kernel_opt)

    # Tighten jitter from 0.1*var(y) → 1e-5: interpolate training data closely for posterior
    noise      = gp.learn_noise ? θ_opt.noise : 1e-5
    fx         = f_opt(gp.X', noise)
 
    gp.θ         = θ_opt
    gp.posterior = posterior(fx, gp.y)
 
    return gp
end

# ============================================================
# Prediction
# ============================================================

"""
    predict(gp::GaussianProcess, Xnew::DataFrame) -> (μ, σ)

Return the posterior predictive mean and standard deviation at new input locations.

# Arguments
- `gp::GaussianProcess`: A fitted GP model (i.e. `fit!` has been called)
- `Xnew::DataFrame`: New input points. Must contain the same input columns as the training data

# Returns
- `μ::Vector{Float64}`: Posterior predictive mean at each row of `Xnew`
- `σ::Vector{Float64}`: Posterior predictive standard deviation (√variance) at each row

# Examples
```julia
fit!(gp)
μ, σ = predict(gp, X_new)
```

# Notes
- `σ` reflects **epistemic uncertainty** only when `learn_noise = false`.
  With learned noise it also includes aleatoric uncertainty.
- Throws an error if called before `fit!(gp)`

# See also
[`evaluate!`](@ref) for in-place prediction directly into a `DataFrame`
"""
function predict(gp::GaussianProcess, Xnew::DataFrame)
    gp.posterior === nothing && error("GP has not been trained yet. Call fit!(gp) first.")
 
    Xmat = Matrix(Xnew[:, gp.x_names])
    Xvec = [Xmat[i, :] for i in 1:size(Xmat, 1)]
 
    μ, v  = mean_and_var(gp.posterior, Xvec)
    σ     = sqrt.(v)

    return μ, σ
end

# ============================================================
# Evalution
# ============================================================

"""
    evaluate!(gp, data; mode=:mean, n_samples=1)

Compute GP predictions and write the results as new columns into `data` in-place.

The output column names are derived from `gp.y_symbol`:
- `:mean`         -> adds `y_mean`
- `:var`          -> adds `y_var`
- `:mean_and_var` -> adds both `y_mean` and `y_var`
- `:sample`       -> adds `y_sample_1`, `y_sample_2`, ... for `n_samples` draws

# Arguments
- `gp::GaussianProcess`: A fitted GP model
- `data::DataFrame`: Dataset to predict on. Modified in-place. Must contain the input columns

# Keyword Arguments
- `mode::Symbol`: What to compute. One of `:mean`, `:var`, `:mean_and_var`, `:sample`.
  Defaults to `:mean`
- `n_samples::Int`: Number of posterior samples to draw. Only used when `mode = :sample`.
  Defaults to `1`

# Returns
- `nothing` — results are written directly into `data`

# Examples
```julia
fit!(gp)

evaluate!(gp, df)                                  # adds :y_mean
evaluate!(gp, df; mode=:mean_and_var)              # adds :y_mean and :y_var
evaluate!(gp, df; mode=:sample, n_samples=50)      # adds :y_sample_1 … :y_sample_50
```

# Notes
- Prefer `evaluate!` over `predict` when you want to keep predictions attached
  to the original dataset for downstream analysis or export
- Throws an error if `fit!(gp)` has not been called first

# See also
[`predict`](@ref) for a non-mutating version that returns `(μ, σ)` directly
"""
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