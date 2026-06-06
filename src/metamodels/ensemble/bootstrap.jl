# ==============================================================================
# bootstrap_ensemble.jl
# Surrogate-agnostic bootstrap ensemble providing pointwise (μ, σ) predictions.
#
# Design principle
# ────────────────
# The struct holds two closures supplied by the caller:
#   factory    :: (df::DataFrame)              -> fitted_model
#   predict_fn :: (model, X::Matrix{Float64})  -> Vector{Float64}   # mean only
#
# predict(ensemble, X) returns (μ, σ) – identical signature to GaussianProcess –
# so BootstrapEnsemble is a drop-in replacement in CABO and any other framework
# that expects a GP-style interface.
#
# Uncertainty source
# ──────────────────
# σ is the EMPIRICAL std across the B member predictions.  It estimates the
# epistemic uncertainty caused by limited training data: if every bootstrap
# model agrees at x, σ(x)≈0; where they disagree, σ(x) is large.  This is
# qualitatively the same signal the GP posterior std provides, but it is
# surrogate-agnostic and requires no analytical derivation.
# ==============================================================================

using Statistics:      mean, std
using Random:          MersenneTwister
using DataFrames:      DataFrame, nrow, propertynames, Not
using Distributions:   Normal, quantile

# ── Core struct ────────────────────────────────────────────────────────────────

"""
    BootstrapEnsemble

Surrogate-agnostic bootstrap ensemble.  Wraps B independently fitted copies of
any surrogate model, each trained on a bootstrap resample of the training data.

Construct via the generic constructor or the convenience helpers
`gp_bootstrap` / `pce_bootstrap`.
"""
mutable struct BootstrapEnsemble
    data         :: DataFrame
    y_symbol     :: Symbol
    x_names      :: Vector{Symbol}
    factory      :: Function            # (df::DataFrame) → fitted_model
    predict_fn   :: Function            # (model, X::Matrix) → Vector{Float64}
    n_bootstraps :: Int
    rng_seed     :: Int
    members      :: Union{Nothing, Vector{Any}}
end

# ── Generic constructor ────────────────────────────────────────────────────────

"""
    BootstrapEnsemble(data, y_symbol, factory, predict_fn; n_bootstraps, rng_seed)

Generic constructor.

Arguments
─────────
`factory`    – closure that creates AND fits a model on a given DataFrame.
               Signature: `(df::DataFrame) -> fitted_model`

`predict_fn` – closure that returns the MEAN prediction vector for a fitted
               model at new inputs X::Matrix.
               Signature: `(model, X::Matrix{Float64}) -> Vector{Float64}`

Examples
────────
# GP-backed ensemble
factory    = df -> (gp = GaussianProcess(df, :y); fit!(gp); gp)
predict_fn = (gp, X) -> first(predict(gp, X))   # discard GP σ, keep μ

# PCE-backed ensemble
factory    = df -> (pce = PolynomialChaosExpansion(df, :y, bases, degree); fit!(pce); pce)
predict_fn = (pce, X) -> predict(pce, DataFrame(X, x_names))
"""
function BootstrapEnsemble(
    data         :: DataFrame,
    y_symbol     :: Symbol,
    factory      :: Function,
    predict_fn   :: Function;
    n_bootstraps :: Int = 100,
    rng_seed     :: Int = 42
)
    x_names = [n for n in propertynames(data) if n !== y_symbol]
    return BootstrapEnsemble(
        data, y_symbol, x_names,
        factory, predict_fn,
        n_bootstraps, rng_seed,
        nothing
    )
end

# ── fit! ───────────────────────────────────────────────────────────────────────

"""
    fit!(ensemble; verbose=true) → ensemble

Train all B bootstrap members.  Each member is fitted on an independent
n-of-n resample with replacement of the original n training points.

Variance across members ≡ bootstrap epistemic uncertainty.
"""
function fit!(ensemble::BootstrapEnsemble; verbose::Bool = true)
    rng              = MersenneTwister(ensemble.rng_seed)
    n                = nrow(ensemble.data)
    B                = ensemble.n_bootstraps
    ensemble.members = Vector{Any}(undef, B)

    for b in 1:B
        verbose && print("\r  Bootstrap member $b / $B …")
        idx = rand(rng, 1:n, n)                              # resample w/ replacement
        ensemble.members[b] = ensemble.factory(ensemble.data[idx, :])
    end
    verbose && println("\r  ✓ $B bootstrap members fitted.           ")
    return ensemble
end

# ── predict ────────────────────────────────────────────────────────────────────

"""
    predict(ensemble, X::Matrix{Float64}) → (μ, σ)

Ensemble mean and empirical std of member predictions.

Returns a `(μ, σ)` tuple of `Vector{Float64}` – identical to the
GaussianProcess interface, so the ensemble is a drop-in for any code
that calls `μ, σ = predict(model, X)`.
"""
function predict(ensemble::BootstrapEnsemble, X::Matrix{Float64})
    ensemble.members === nothing && error("Call fit!(ensemble) first.")
    n    = size(X, 1)
    B    = ensemble.n_bootstraps
    preds = Matrix{Float64}(undef, n, B)

    for b in 1:B
        preds[:, b] = ensemble.predict_fn(ensemble.members[b], X)
    end

    μ = vec(mean(preds; dims = 2))
    σ = vec(std(preds;  dims = 2, corrected = true))   # sample std (B-1 denominator)
    return μ, σ
end

"""
    predict(ensemble, X::DataFrame) → (μ, σ)

Convenience overload: selects `x_names` columns in the correct order and
delegates to the Matrix method.
"""
function predict(ensemble::BootstrapEnsemble, X::DataFrame)
    return predict(ensemble, Matrix(X[:, ensemble.x_names]))
end

# ── evaluate! (UQModel compatibility) ─────────────────────────────────────────

function evaluate!(ensemble::BootstrapEnsemble, data::DataFrame)
    ensemble.members === nothing && error("Call fit!(ensemble) first.")
    X_mat = Matrix(data[:, ensemble.x_names])
    μ, _  = predict(ensemble, X_mat)
    data[!, ensemble.y_symbol] = μ
    return data
end

# ==============================================================================
# Convenience constructors
# ==============================================================================

# ── gp_bootstrap ───────────────────────────────────────────────────────────────

"""
    gp_bootstrap(data, y_symbol; kernel_type, n_bootstraps, rng_seed)

Bootstrap ensemble of GaussianProcess models.

The analytical GP posterior variance is intentionally discarded; the bootstrap
variance replaces it.  Use this to compare bootstrap vs. analytical GP
uncertainty, or when the analytical σ is unavailable / unreliable.

Note: B × N_train GP fits are needed. Keep n_bootstraps ≤ 50 for CABO loops
where predict() is called thousands of times per iteration.
"""
function gp_bootstrap(
    data         :: DataFrame,
    y_symbol     :: Symbol;
    kernel_type              = GPSquaredExponential(),
    n_bootstraps :: Int     = 50,
    rng_seed     :: Int     = 42
)
    factory = df -> begin
        gp = GaussianProcess(df, y_symbol; kernel_type = kernel_type)
        fit!(gp)
        return gp
    end
    predict_fn = (gp, X::Matrix{Float64}) -> first(predict(gp, X))   # μ only
    return BootstrapEnsemble(data, y_symbol, factory, predict_fn;
                             n_bootstraps = n_bootstraps, rng_seed = rng_seed)
end

# ── pce_bootstrap ──────────────────────────────────────────────────────────────

"""
    pce_bootstrap(data, y_symbol, bases, degree; solver, n_bootstraps, rng_seed)

Bootstrap ensemble of PolynomialChaosExpansion models.

This is the recommended choice for CABO: PCE prediction is O(p) (number of
basis terms) vs. O(N²) for GP, so the ensemble overhead stays manageable.

The PCE predict_fn captures `x_names` in a closure so it can reconstruct the
required DataFrame from a raw Matrix.
"""
function pce_bootstrap(
    data         :: DataFrame,
    y_symbol     :: Symbol,
    bases        :: Vector{<:AbstractPCEBasis},
    degree       :: AbstractPCEDegree;
    solver       :: AbstractPCESolver = OLSSolver(),
    n_bootstraps :: Int               = 100,
    rng_seed     :: Int               = 42
)
    # Capture x_names here so the closure needs no access to the ensemble struct.
    x_names = [n for n in propertynames(data) if n !== y_symbol]

    factory = df -> begin
        pce = PolynomialChaosExpansion(df, y_symbol, bases, degree; solver = solver)
        fit!(pce)
        return pce
    end
    predict_fn = (pce, X::Matrix{Float64}) -> predict(pce, DataFrame(X, x_names))

    return BootstrapEnsemble(data, y_symbol, factory, predict_fn;
                             n_bootstraps = n_bootstraps, rng_seed = rng_seed)
end

# ==============================================================================
# Diagnostics
# ==============================================================================

"""
    calibration_coverage(σ_pred, y_pred, y_true; α=0.95) → Float64

Fraction of test points where the (1-α) prediction interval covers y_true.
Ideal value = α.

  > α  → overconfident  (σ too small, intervals too narrow)
  < α  → underconfident (σ too large,  intervals too wide)
"""
function calibration_coverage(σ_pred, y_pred, y_true; α = 0.95)
    z    = quantile(Normal(), (1 + α) / 2)
    hits = abs.(y_pred .- y_true) .≤ z .* σ_pred
    return mean(hits)
end

"""
    calibration_report(σ_pred, y_pred, y_true; levels=[0.5,0.9,0.95,0.99])

Print coverage at multiple confidence levels.  A well-calibrated ensemble
should match each level within a few percent.
"""
function calibration_report(σ_pred, y_pred, y_true;
                             levels = [0.50, 0.90, 0.95, 0.99])
    println("\n  Calibration report")
    println("  ─────────────────────────────")
    println("  Nominal α    Empirical coverage")
    for α in levels
        cov = calibration_coverage(σ_pred, y_pred, y_true; α = α)
        flag = abs(cov - α) < 0.05 ? "✓" : "✗"
        print("α * 100 = $(α * 100) | cov * 100 = $(cov * 100) | flag: $flag\n")
    end
end