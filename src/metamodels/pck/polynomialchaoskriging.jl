"""
    PolynomialChaosKriging <: UQModel

Sequential Polynomial Chaos Kriging (PCK) surrogate: a deterministic PCE
trend plus a zero-mean Gaussian Process modelling the residual.

    y(x) ≈ Σ_α β_α Ψ_α(x)  +  Z(x)
           └──── PCE ────┘   └ GP ┘
             (trend, fit first)   (residual, fit second — "sequential")

The PCE captures the global, smooth structure of the response; the residual
GP captures whatever local/non-polynomial structure is left over, and
supplies the model's predictive variance (the PCE trend itself is treated
as deterministic once fit, so it contributes zero variance).

The PCE basis is selected once, up front, using ordinary PCE regression; it is
NOT re-optimised against the Kriging covariance structure (that would be
"optimal" PCK, a more expensive iterative variant).

# Fields
- `X::Matrix{Float64}`: Training inputs
- `y::Vector{Float64}`: Training outputs
- `x_names::Vector{Symbol}`: Input column names
- `y_symbol::Symbol`: Output column name
- `pce::PolynomialChaosExpansion`: The trend model (construct and configure
this yourself — degree/basis/solver — exactly as you would for standalone PCE)

- `gp_residual::Union{GaussianProcess,Nothing}`: The residual model, built
internally during `fit!`; `nothing` until then

- `kernel_type::AbstractGPKernel`: Kernel for the residual GP
- `mean_type::AbstractGPMean`: Mean function for the residual GP (defaults
to `GPZeroMean()` — see Notes)

- `learn_noise::Bool`: Forwarded to the residual GP
"""
mutable struct PolynomialChaosKriging <: UQModel
    X::Matrix{Float64}
    y::Vector{Float64}
    x_names::Vector{Symbol}
    y_symbol::Symbol

    pce::PolynomialChaosExpansion
    gp_residual::Union{GaussianProcess, Nothing}

    kernel_type::AbstractGPKernel
    mean_type::AbstractGPMean
    learn_noise::Bool
end

"""
    PolynomialChaosKriging(data, y_symbol, pce; kernel_type, mean_type, learn_noise)

Construct a sequential PCK surrogate.

# Arguments
- `data::DataFrame`: Training dataset (inputs + `y_symbol` column)
- `y_symbol::Symbol`: Name of the output column
- `pce::PolynomialChaosExpansion`: An UNFITTED PCE object, already
constructed on the SAME `(data, y_symbol)` with whatever degree/basis/
solver settings you want — PCK does not choose these for you, it just
orchestrates fitting

# Keyword Arguments
- `kernel_type::AbstractGPKernel`: Kernel for the residual GP. Defaults to `GPMatern52()`
- `mean_type::AbstractGPMean`: Mean function for the residual GP. Defaults
to `GPZeroMean()` — see Notes for why this should generally stay zero
- `learn_noise::Bool`: Forwarded to the residual GP. Defaults to `false`

# Examples
```julia
pce = PolynomialChaosExpansion(data_train, :y;
    degree_type = TotalDegree(5), 
    basis_type = [LegendreBasis(), LegendreBasis()], 
    solver_type = LASSOSolver()
)

pck = PolynomialChaosKriging(data_train, :y, pce; kernel_type = GPMatern52())
fit!(pck)
μ, σ = predict(pck, Matrix(X_test))
```

# Notes
- `mean_type` defaults to `GPZeroMean()` deliberately: any constant offset
the data needs should already be absorbed by the PCE's own degree-0 term.
Giving the residual GP a non-zero mean on top of that is redundant and
can make the two trend components fight each other during fitting.

- `pce` must be built on the SAME training data you pass here. PCK does not
re-slice or re-sample it.
"""
function PolynomialChaosKriging(
    data::DataFrame,
    y_symbol::Symbol,
    pce::PolynomialChaosExpansion;
    kernel_type::AbstractGPKernel = GPMatern52(),
    mean_type::AbstractGPMean     = GPZeroMean(),
    learn_noise::Bool             = false
)
    x_names = propertynames(data[:, Not(y_symbol)])
    X = Matrix(data[:, x_names])
    y = Vector(data[:, y_symbol])

    return PolynomialChaosKriging(
        X, y, x_names, y_symbol,
        pce, nothing,
        kernel_type, mean_type, learn_noise
    )
end


model_name(pck::PolynomialChaosKriging) =
    "PCK (PCE trend + $(kernel_name(pck.kernel_type)) residual)"


"""
    fit!(pck::PolynomialChaosKriging)

Fit the PCK in two sequential steps:
1. Fit the PCE trend on the original data
2. Fit a residual GP on `y - PCE(X)` (the part the trend didn't explain)

# Arguments
- `pck::PolynomialChaosKriging`: An unfitted (or previously fitted) PCK. Modified in-place

# Returns
- `pck`: The same object, now with `pck.pce` and `pck.gp_residual` both fitted

# Notes
- Both sub-fits reuse the existing `fit!` implementations
for `PolynomialChaosExpansion` and `GaussianProcess` — PCK adds no new
optimisation logic of its own, only the trend/residual orchestration

"""
function fit!(pck::PolynomialChaosKriging)
    # 1. Fit the trend
    fit!(pck.pce)
    μ_pce_train = predict(pck.pce, pck.X)

    # 2. Residuals = what the trend didn't capture
    residuals = pck.y .- μ_pce_train

    # 3. Fit a zero-mean residual GP on those residuals
    resid_df = DataFrame(pck.X, pck.x_names)
    resid_df[!, pck.y_symbol] = residuals

    gp_residual = GaussianProcess(resid_df, pck.y_symbol;
        mean_type   = pck.mean_type,
        kernel_type = pck.kernel_type,
        learn_noise = pck.learn_noise
    )
    fit!(gp_residual)

    pck.gp_residual = gp_residual
    return pck
end

"""
    predict(pck::PolynomialChaosKriging, Xnew::Matrix{Float64}; mode=:mean_and_var) -> (μ, σ)

Predict at new input locations: trend + residual mean, residual variance only
(the PCE trend is deterministic once fit, so it contributes zero variance).

# Returns
- `μ::Vector{Float64}`: `PCE(Xnew) .+ residual_GP_mean(Xnew)`
- `σ::Vector{Float64}`: residual GP's posterior std (only present if `mode=:mean_and_var`)

# Examples
```julia
fit!(pck)
μ, σ = predict(pck, X_new)
μ_only = predict(pck, X_new; mode=:mean)
```
"""
function predict(pck::PolynomialChaosKriging, Xnew::Matrix{Float64}; mode::Symbol = :mean_and_var, n_samples:: Int = 1)
    pck.gp_residual === nothing && error("PCK has not been trained yet. Call fit!(pck) first.")

    μ_pce = predict(pck.pce, Xnew)

    if mode === :mean
        μ_resid = predict(pck.gp_residual, Xnew; mode = :mean)
        return μ_pce .+ μ_resid

    elseif mode === :var
        var_resid = predict(pck.gp_residual, Xnew; mode = :var)
        return var_resid
    elseif mode === :mean_and_var
        μ_resid, σ_resid = predict(pck.gp_residual, Xnew; mode = :mean_and_var)
        return μ_pce .+ μ_resid, σ_resid

    elseif mode === :sample
        μ_sample_resid = predict(pck.gp_residual, Xnew; mode = :sample, n_samples=n_samples)
        return μ_pce .+ μ_sample_resid
    
    else
        throw(ArgumentError("Unknown mode: $mode. Choose :mean, :var, :mean_and_var, or :sample."))
    end
end

"""
    refit!(pck::PolynomialChaosKriging, X_new, y_new)

Add new points and refit ONLY the residual GP (warm-started via GP's own
`refit!`), keeping the PCE trend FIXED at whatever it was fit to originally.

This is the right choice inside an adaptive loop (e.g. CABO) where you're
adding one point at a time and re-fitting the full PCE every iteration
would be wasteful: as long as the initial design was reasonably space-
filling, the PCE trend's global polynomial structure doesn't need to move
much, and only the residual GP needs to absorb new local information.

If you DO want the trend itself to update (e.g. after adding many points,
or if the trend looks stale), just call `fit!(pck)` again instead — that
refits both stages from scratch.

# Arguments
- `pck::PolynomialChaosKriging`: A previously fitted PCK. Modified in-place
- `X_new::AbstractMatrix{Float64}`: New training inputs (one or more rows)
- `y_new::AbstractVector{Float64}`: Corresponding new outputs
"""
function refit!(pck::PolynomialChaosKriging, X_new::AbstractMatrix{Float64}, y_new::AbstractVector{Float64})
    pck.gp_residual === nothing && error("Call fit!(pck) before refit!.")

    pck.X = vcat(pck.X, X_new)
    pck.y = vcat(pck.y, y_new)

    μ_pce_new = predict(pck.pce, X_new)     # PCE trend stays fixed - just evaluate it at the new points
    residual_new = y_new .- μ_pce_new

    refit!(pck.gp_residual, X_new, residual_new)   # warm-started GP refit, as already implemented

    return pck
end



"""
    evaluate!(pck, data; mode=:mean, n_samples=1)

Compute PCK predictions and write the results as new columns into `data` in-place.

A thin wrapper around `predict`, all computation happens there; this
function only extracts the input matrix, calls `predict`, and writes the
result into the appropriate column(s) of `data`. Keeping the two in sync this
way means there is exactly one implementation of the actual PCK query logic.

The output column names are derived from `pck.y_symbol`:
- `:mean`         -> adds `y_mean`  (writes to `pck.y_symbol` directly, per original convention)
- `:var`          -> adds `y_var`
- `:mean_and_var` -> adds both `y_mean` and `y_var`
- `:sample`       -> adds `y_sample_1`, `y_sample_2`, ... for `n_samples` draws

# Arguments
- `pck::PolynomialChaosKriging`: A fitted PCK model
- `data::DataFrame`: Dataset to predict on. Modified in-place. Must contain the input columns

# Keyword Arguments
- `mode::Symbol`: What to compute. One of `:mean`, `:var`, `:mean_and_var`, `:sample`.
- `n_samples::Int`: Number of posterior samples to draw. Only used when `mode = :sample`.

# Returns
- `nothing` - results are written directly into `data`

# Examples
```julia
fit!(pck)

evaluate!(pck, df)                                  # adds :y (mean)
evaluate!(pck, df; mode=:mean_and_var)              # adds :y_mean and :y_var
evaluate!(pck, df; mode=:sample, n_samples=50)      # adds :y_sample_1 … :y_sample_50
```

# Notes
- Prefer `evaluate!` over `predict` when you want to keep predictions attached
  to the original dataset for downstream analysis or export

- `:mean_and_var` internally requests `predict(...; mode=:mean_and_var)` (which
returns std, per its own convention) and squares it back to variance for the
`_var` column

"""
function evaluate!(
    pck       ::PolynomialChaosKriging,
    data      ::Union{DataFrame, DataFrameRow};
    mode      ::Symbol = :mean,
    n_samples ::Int    = 1,
)
    pck.gp_residual === nothing && error("Call fit!(pck) before evaluate!.")

    Xmat = Matrix(data[:, pck.x_names])

    col_mean = pck.y_symbol
    col_var  = Symbol(string(pck.y_symbol, "_var"))

    if mode === :mean
        data[!, pck.y_symbol] = predict(pck, Xmat; mode=:mean)

    elseif mode === :var
        data[!, col_var] = predict(pck, Xmat; mode=:var)

    elseif mode === :mean_and_var
        μ, σ = predict(pck, Xmat; mode=:mean_and_var)
        data[!, col_mean] = μ
        data[!, col_var]  = σ .^ 2      # square back to variance for the "_var" column

    elseif mode === :sample
        samples = predict(pck, Xmat; mode=:sample, n_samples=n_samples)
        for i in 1:n_samples
            col = Symbol(string(pck.y_symbol, "_sample_", i))
            data[!, col] = samples[:, i]
        end

    else
        throw(ArgumentError("Unknown mode: $mode. Choose :mean, :var, :mean_and_var, or :sample."))
    end

    return nothing
end


export 
    PolynomialChaosKriging,
    fit!, refit!, predict, evaluate!