"""
    GaussianProcess <: UQModel

A Gaussian Process (GP) surrogate model for uncertainty quantification.

Stores training data, kernel/mean hyperparameters, and the trained posterior
distribution. Must be fitted before predictions can be made.

# Fields
- `X::Matrix{Float64}`: Training inputs of shape `(n_samples, n_dims)`
- `y::Vector{Float64}`: Training outputs of length `n_samples`
- `mean_type::AbstractGPMean`: Prior mean specification (e.g. `GPZeroMean()`)
- `kernel_type::AbstractGPKernel`: Kernel specification (e.g. `GPMatern52()`)
- `θ::NamedTuple`: Current kernel hyperparameters
- `flat_θ::Vector{Float64}`: Flattened (unconstrained) form of `θ`, as used by Optim.jl
- `posterior`: Trained GP posterior, or `nothing` if not yet fitted
- `y_symbol::Symbol`: Name of the output column in the original `DataFrame`
- `x_names::Vector{Symbol}`: Names of the input columns, used for column-safe prediction
- `unflatten`: Inverse of `value_flatten` - reconstructs a `NamedTuple` from a flat vector
- `mean_prior`, `kernel_prior`: Prior mean/kernel objects built at construction time
- `mean_posterior`, `kernel_posterior`: Mean/kernel objects built at the optimised `θ`
- `learn_noise::Bool`: If `true`, observation noise σ² is included as a learnable
  hyperparameter; if `false`, a fixed adaptive jitter is used instead

# Notes
here the subscript "_prior/_posterior" refers to building the mean and or kernel
either with the initial θ / the optimal θ.
"""
mutable struct GaussianProcess <: UQModel
    X::Matrix{Float64}
    y::Vector{Float64}

    mean_type:: AbstractGPMean
    kernel_type:: AbstractGPKernel
    θ::NamedTuple
    flat_θ::Vector{Float64}

    posterior::Union{AbstractGPs.PosteriorGP, Nothing}

    y_symbol::Symbol
    x_names::Vector{Symbol}

    unflatten
    mean_prior
    kernel_prior
    mean_posterior
    kernel_posterior
    learn_noise::Bool
end

"""
    GaussianProcess(data, y_symbol; mean_type, kernel_type, θ, learn_noise)

Construct a `GaussianProcess` surrogate from a `DataFrame`.

Extracts inputs `X` and output `y` from `data`, initialises hyperparameters,
and prepares the model for training via `fit!`.

# Arguments
- `data::DataFrame`: Training dataset containing both inputs and the output column
- `y_symbol::Symbol`: Name of the output column in `data`

# Keyword Arguments
- `mean_type::AbstractGPMean`: Prior mean specification. Defaults to `GPZeroMean()`
- `kernel_type::AbstractGPKernel`: Kernel specification. Defaults to `GPMatern52()`
- `θ::Union{NamedTuple, Nothing}`: Initial hyperparameters. If `nothing`
sensible defaults are inferred from `X` and `y`. Defaults to `nothing`

- `learn_noise::Bool`: Whether to learn observation noise as a hyperparameter.
Set to `true` for noisy simulators, `false` for deterministic ones. Defaults to `false`

# Returns
- `GaussianProcess`: An unfitted model ready to be trained with `fit!(gp)`

# Examples
```julia
gp = GaussianProcess(train_df, :y)
gp = GaussianProcess(train_df, :y; mean_type=GPConstMean())
gp = GaussianProcess(train_df, :y; kernel_type=GPSquaredExponential(), learn_noise=true)
```

# Notes
- All columns in `data` except `y_symbol` are treated as inputs
- Input scaling should be applied to `data` **before** constructing the GP

"""
function GaussianProcess(
    data::DataFrame,
    y_symbol::Symbol;
    mean_type::AbstractGPMean          = GPZeroMean(),
    kernel_type::AbstractGPKernel      = GPMatern52(),
    θ::Union{NamedTuple, Nothing}      = nothing,
    learn_noise::Bool                  = false
)

    x_names = propertynames(data[:, Not(y_symbol)])

    X = Matrix(data[:, x_names])
    y = Vector(data[:, y_symbol])

    # Infer starting values from data if not provided (avoids flat/degenerate regions)
    θ0 = isnothing(θ) ? default_θ(kernel_type, X, y) : θ

    # Optim.jl needs a flat Vector{Float64}; unflatten reconstructs the NamedTuple after optimisation
    flat_θ0, unflatten = value_flatten(θ0)

    mean_prior = build_mean(mean_type, X, y)
    kernel_prior = build_kernel(kernel_type, unflatten(flat_θ0))

    return GaussianProcess(
        X,
        y,
        mean_type,
        kernel_type,
        θ0,
        flat_θ0,
        nothing,
        y_symbol,
        x_names,
        unflatten,
        mean_prior,
        kernel_prior,
        nothing,
        nothing,
        learn_noise
    )
end


# ============================================================
# Training
# ============================================================

"""
    fit!(gp::GaussianProcess)

Train the GP by minimizing the negative log marginal likelihood (NLML) over the kernel
hyperparameters, then store the optimised posterior in `gp.posterior`.

Optimisation is performed with L-BFGS using exact gradients via `Mooncake.jl`
automatic differentiation, from 10 random restarts (see Notes).

# Arguments
- `gp::GaussianProcess`: An unfitted (or previously fitted) GP model. Modified in-place.

# Returns
- `gp::GaussianProcess`: The same object, now with `gp.θ` and `gp.posterior` updated

# Notes
- **Noise handling:**
  - If `gp.learn_noise = true`: noise variance σ² is included in `θ` and optimised
  - If `gp.learn_noise = false`: noise is set to `0.1 * var(y)` during hyperparameter
    optimisation (keeps the marginal-likelihood surface well-conditioned), then
    tightened to `1e-8` for the final posterior (near-exact interpolation,
    appropriate for a deterministic simulator)
- Runs 10 optimisation restarts (1 from the current `θ`, 9 from increasingly large
  random perturbations of it) and keeps the lowest-NLML result. Use [`refit!`](@ref)
  instead for subsequent adaptive-sampling updates -- it warm-starts from the
  current optimum with far fewer restarts, since the hyperparameters shouldn't
  need to move far after adding one point.
- Convergence is not guaranteed - check `gp.θ` for reasonable hyperparameter values
  if predictions look off

    """
function fit!(gp::GaussianProcess)

    mean = build_mean(gp.mean_type, gp.X, gp.y)
    Xt = collect(gp.X')


    function nlml(flat_θ)

        θ = gp.unflatten(flat_θ)                # recover NamedTuple so kernel can unpack named fields
        kernel = build_kernel(gp.kernel_type, θ)     # kernels are immutable -> rebuild on every call

        f = GP(mean, kernel)

        # when learn_noise = true  -> σ² is a free parameter (noisy observations) else 1e-8
        noise = gp.learn_noise ? θ.noise : 1e-8

        fx = f(Xt, noise)

        val = -logpdf(fx, gp.y) # negative because Optim.jl minimises

        return val

    end

    # Run from default θ0 + n_restarts-1 random perturbations
    function run_optimization(flat_θ0)
        return Optim.optimize(
            nlml,
            (g, θ) -> begin
                _, grad = DifferentiationInterface.value_and_gradient(
                    nlml, AutoMooncake(; config=nothing), θ
                )
                g .= grad
            end,
            flat_θ0,
            LBFGS(),
            Optim.Options(show_trace=false);
            inplace=true
        )
    end

    flat_θ0, _ = value_flatten(gp.θ)

    n_restarts = 10

    results = map(1:n_restarts) do i
        spread = 0.5 * (i - 1)
        θ_start = i == 1 ? flat_θ0 : flat_θ0 .+ spread .* randn(length(flat_θ0))

        try
            run_optimization(θ_start)
        catch e
            @warn "Optimization failed at restart $i..."
            nothing
        end
    end

    # Keep the result with the lowest NLML
    valid   = filter(!isnothing, results)
    isempty(valid) && error("All optimisation restarts failed.")
    best    = argmin(r -> r.minimum, valid)
    θ_opt = gp.unflatten(best.minimizer)

    # Build posterior with optimised hyperparameters
    kernel_opt = build_kernel(gp.kernel_type, θ_opt)
    f_opt      = GP(kernel_opt)

    noise      = gp.learn_noise ? θ_opt.noise : 1e-8
    fx         = f_opt(gp.X', noise)

    gp.θ         = θ_opt
    gp.posterior = posterior(fx, gp.y)
    gp.flat_θ = best.minimizer
    gp.mean_posterior = x -> mean(gp.posterior(x))
    gp.kernel_posterior = build_kernel(gp.kernel_type, θ_opt)

    return gp
end


"""
    refit!(gp::GaussianProcess, X_new, y_new)

Append new training points and re-optimise hyperparameters, WARM-STARTING
from `gp.θ` (the previous optimum) instead of restarting from scratch.

It runs only 3 restarts from small perturbations of the current optimum, since the
hyperparameters shouldn't need to move far after adding one point.

# Arguments
- `gp::GaussianProcess`: A previously fitted model. Modified in-place.
- `X_new::AbstractMatrix{Float64}`: New training input rows to append
- `y_new::AbstractVector{Float64}`: Corresponding new output values

# Returns
- `gp::GaussianProcess`: The same object, with `gp.X`, `gp.y`, `gp.θ`, and
  `gp.posterior` all updated

# Notes
- Noise handling is identical to [`fit!`](@ref): `0.1 * var(y)` during
  optimisation, tightened to `1e-8` for the posterior when `learn_noise = false`

"""
function refit!(gp::GaussianProcess, X_new::AbstractMatrix{Float64}, y_new::AbstractVector{Float64})

    gp.X = vcat(gp.X, X_new)
    gp.y = vcat(gp.y, y_new)

    mean = build_mean(gp.mean_type, gp.X, gp.y)
    Xt = collect(gp.X')

    function nlml(flat_θ)
        θ = gp.unflatten(flat_θ)                     # recover NamedTuple so kernel can unpack named fields
        kernel = build_kernel(gp.kernel_type, θ)     # kernels are immutable -> rebuild on every call

        f = GP(mean, kernel)

        # when learn_noise = true  -> σ² is a free parameter (noisy observations) else: jitter = 1e-8
        noise = gp.learn_noise ? θ.noise : 1e-8

        fx = f(Xt, noise)
        val = -logpdf(fx, gp.y) # negative because Optim.jl minimises

        return val

    end

    function run_optimization(flat_θ0)
        return Optim.optimize(
            nlml,
            (g, θ) -> begin
                _, grad = DifferentiationInterface.value_and_gradient(
                    nlml, AutoMooncake(; config=nothing), θ
                )
                g .= grad
            end,
            flat_θ0,
            LBFGS(),
            Optim.Options(show_trace=false);
            inplace=true
        )
    end

    n_restarts = 3
    flat_θ0, _ = value_flatten(gp.θ)

    results = map(1:n_restarts) do i
        spread = 0.5 * (i - 1)
        θ_start = i == 1 ? flat_θ0 : flat_θ0 .+ spread .* randn(length(flat_θ0))

        try
            run_optimization(θ_start)
        catch e
            @warn "Optimization failed at restart $i..."
            nothing
        end
    end

    # Keep the result with the lowest NLML
    valid   = filter(!isnothing, results)
    isempty(valid) && error("All optimisation restarts failed.")
    best    = argmin(r -> r.minimum, valid)
    θ_opt = gp.unflatten(best.minimizer)

    # Build posterior with optimised hyperparameters
    kernel_opt = build_kernel(gp.kernel_type, θ_opt)
    f_opt      = GP(kernel_opt)

    noise      = gp.learn_noise ? θ_opt.noise : 1e-8
    fx         = f_opt(gp.X', noise)

    gp.θ         = θ_opt
    gp.posterior = posterior(fx, gp.y)
    gp.flat_θ = best.minimizer    # ← store the UNCONSTRAINED optimum
    gp.mean_posterior = x -> mean(gp.posterior(x))
    gp.kernel_posterior = build_kernel(gp.kernel_type, θ_opt)

    return gp
end


"""
    predict(gp::GaussianProcess, Xnew::AbstractMatrix{Float64}; mode=:mean_and_var, n_samples=1)

Return posterior predictions at new input locations. This is the single source
of truth for GP prediction -- [`evaluate!`](@ref) is a thin DataFrame-writing
wrapper around this function; both share identical logic for every mode.

# Arguments
- `gp::GaussianProcess`: A fitted GP model (i.e. `fit!` has been called)
- `Xnew::AbstractMatrix{Float64}`: New input points, one row per point, same
  column order as `gp.x_names`. Accepts plain `Matrix` or a `@view`/`SubArray`.

# Keyword Arguments
- `mode::Symbol`: One of:
    - `:mean` - returns `μ::Vector{Float64}` only. Never forms the full posterior
    covariance (cheapest mode).

    - `:var` - returns `σ²::Vector{Float64}`, the marginal **variance** at each
    point (matches `evaluate!`'s `_var` column and `AbstractGPs`'
    `mean_and_var` convention).

    - `:mean_and_var` (default) - returns `(μ, σ)`, mean and **standard
    deviation** (√variance). This is the long-standing return convention used
    throughout the rest of the codebase (`μ, σ = predict(gp, X)`) and is kept
    exactly as-is for backward compatibility.

    - `:sample` - draws `n_samples` jointly-correlated realizations from the
    posterior at `Xnew` and returns them as an `(n_points × n_samples)` matrix.
    Unlike the other modes, this genuinely needs the full joint covariance
    (samples must be correlated across points), so it cannot use a
    marginal-only fast path.

    - `n_samples::Int`: Number of posterior samples to draw. Only used when
    `mode = :sample`. Defaults to `1`.

# Returns
Depends on `mode` -- see above.

# Examples
```julia
fit!(gp)
μ, σ         = predict(gp, X_new)                               # :mean_and_var (default)
μ            = predict(gp, X_new; mode=:mean)
σ2           = predict(gp, X_new; mode=:var)
G            = predict(gp, X_new; mode=:sample, n_samples=20)
```

"""
function predict(
    gp::GaussianProcess,
    Xnew::AbstractMatrix{Float64};
    mode::Symbol = :mean_and_var,
    n_samples::Int = 1,
)
    gp.posterior === nothing && error("GP has not been trained yet. Call fit!(gp) first.")

    Xrows = RowVecs(Xnew)

    if mode === :mean
        # mean-only path: never forms the n×n posterior covariance
        return mean(gp.posterior(Xrows))

    elseif mode === :var
        _, v = mean_and_var(gp.posterior, Xrows)   # marginal VARIANCE
        return v

    elseif mode === :mean_and_var
        μ, v = mean_and_var(gp.posterior, Xrows)
        return μ, sqrt.(v)                          # (mean, STD) preserves existing convention

    elseif mode === :sample
        fp = gp.posterior(Xrows)                    # full joint covariance
        return rand(fp, n_samples)                  # n_points × n_samples

    else
        throw(ArgumentError("Unknown mode: $mode. Choose :mean, :var, :mean_and_var, or :sample."))
    end
end

"""
    evaluate!(gp, data; mode=:mean, n_samples=1)

Compute GP predictions and write the results as new columns into `data` in-place.

A thin wrapper around `predict`, all computation happens there; this
function only extracts the input matrix, calls `predict`, and writes the
result into the appropriate column(s) of `data`. Keeping the two in sync this
way means there is exactly one implementation of the actual GP query logic.

The output column names are derived from `gp.y_symbol`:
- `:mean`         -> adds `y_mean`  (writes to `gp.y_symbol` directly, per original convention)
- `:var`          -> adds `y_var`
- `:mean_and_var` -> adds both `y_mean` and `y_var`
- `:sample`       -> adds `y_sample_1`, `y_sample_2`, ... for `n_samples` draws

# Arguments
- `gp::GaussianProcess`: A fitted GP model
- `data::DataFrame`: Dataset to predict on. Modified in-place. Must contain the input columns

# Keyword Arguments
- `mode::Symbol`: What to compute. One of `:mean`, `:var`, `:mean_and_var`, `:sample`.
- `n_samples::Int`: Number of posterior samples to draw. Only used when `mode = :sample`.

# Returns
- `nothing` - results are written directly into `data`

# Examples
```julia
fit!(gp)

evaluate!(gp, df)                                  # adds :y (mean)
evaluate!(gp, df; mode=:mean_and_var)              # adds :y_mean and :y_var
evaluate!(gp, df; mode=:sample, n_samples=50)      # adds :y_sample_1 ... :y_sample_50
```

# Notes
- Prefer `evaluate!` over `predict` when you want to keep predictions attached
  to the original dataset for downstream analysis or export

- `:mean_and_var` internally requests `predict(...; mode=:mean_and_var)` (which
returns std, per its own convention) and squares it back to variance for the
`_var` column

"""
function evaluate!(
    gp        ::GaussianProcess,
    data      ::Union{DataFrame, DataFrameRow};
    mode      ::Symbol = :mean,
    n_samples ::Int    = 1,
)
    gp.posterior === nothing && error("Call fit!(gp) before evaluate!.")

    Xmat = Matrix(data[:, gp.x_names])

    col_mean = gp.y_symbol
    col_var  = Symbol(string(gp.y_symbol, "_var"))

    if mode === :mean
        data[!, gp.y_symbol] = predict(gp, Xmat; mode=:mean)

    elseif mode === :var
        data[!, col_var] = predict(gp, Xmat; mode=:var)

    elseif mode === :mean_and_var
        μ, σ = predict(gp, Xmat; mode=:mean_and_var)
        data[!, col_mean] = μ
        data[!, col_var]  = σ .^ 2      # square back to variance for the "_var" column

    elseif mode === :sample
        samples = predict(gp, Xmat; mode=:sample, n_samples=n_samples)
        for i in 1:n_samples
            col = Symbol(string(gp.y_symbol, "_sample_", i))
            data[!, col] = samples[:, i]
        end

    else
        throw(ArgumentError("Unknown mode: $mode. Choose :mean, :var, :mean_and_var, or :sample."))
    end

    return nothing
end



export GaussianProcess, fit!, refit!, predict, evaluate!