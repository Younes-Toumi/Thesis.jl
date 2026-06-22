module SurrogateModelling

# ── Dependencies ──────────────────────────────────────────────
using Statistics
using LinearAlgebra
using DataFrames
using Plots
using Random
using Printf
using UncertaintyQuantification
using DifferentiationInterface
using Clustering
using FastGaussQuadrature
using Distributions
using QuasiMonteCarlo
using Mooncake

using KernelFunctions
using AbstractGPs
using ParameterHandling
using Optim
using Zygote

using Metaheuristics



# 0. helper functions ────────────────────────────────────────────────────────────────────────── #
φ(z)     = pdf(Normal(), z)
Φ(z)     = cdf(Normal(), z)
Φ⁻¹(p)   = quantile(Normal(), p)
φ_vec(u) = prod(pdf.(Normal(), u))


# 1. metamodels ────────────────────────────────────────────────────────────────────────── #
# 1.1. Gaussian Process

# Kernel Related
include("metamodels/gp//kernels/kernels.jl")
include("metamodels/gp/kernels/Matern52.jl")
include("metamodels/gp/kernels/Matern32.jl")
include("metamodels/gp/kernels/Matern12.jl")
include("metamodels/gp/kernels/SquaredExponential.jl")
include("metamodels/gp/kernels/Composite.jl")

export
    AbstractGPKernel, GPMatern52, GPMatern32, GPMatern12, GPSquaredExponential, GPCompositeKernel,
    kernel_name

# Mean Related
include("metamodels/gp/means/means.jl")
include("metamodels/gp/means/ZeroMean.jl")
include("metamodels/gp/means/ConstMean.jl")

export 
    AbstractGPMean, GPZeroMean, GPConstMean,
    mean_name


# GP Constructor Related


include("metamodels/gp/gaussianprocess.jl")

export 
    GaussianProcess,
    fit!, refit!, predict


# 1.2. Polynomial Chaos Expansion

# Degree related:
include("metamodels/pce/degrees/degrees.jl")
include("metamodels/pce/degrees/TotalDegree.jl")
include("metamodels/pce/degrees/TensorProduct.jl")
include("metamodels/pce/degrees/HyperbolicCross.jl")
include("metamodels/pce/degrees/QBall.jl")

export
    AbstractPCEDegree,
    TotalDegree, TensorProduct, HyperbolicCross, QBall,
    n_terms, degree_name, multivariate_indices

# Basis related
include("metamodels/pce/bases/bases.jl")
include("metamodels/pce/bases/Hermite.jl")
include("metamodels/pce/bases/Legendre.jl")

export
    AbstractPCEBasis, HermiteBasis, LegendreBasis,
    build_design_matrix

# Solver related:
include("metamodels/pce/solvers/solvers.jl")
include("metamodels/pce/solvers/OLS.jl")
include("metamodels/pce/solvers/LASSO.jl")

export
    OSLSolver, LASSOSolver,
    solver_name


# Constructor Related
include("metamodels/pce/polynomialchaosexpansion.jl")

export 
    PolynomialChaosExpansion,
    fit!, predict


# 2. Scalings ────────────────────────────────────────────────────────────────────────── #

using Statistics

include("scalings/scalings.jl")
include("scalings/minmax.jl")
include("scalings/zscore.jl")
include("scalings/scaling_pipeline.jl")

export 
    MinMaxScaler, ZScoreScaler, 
    fit_pipeline, transform_input, transform_output, transform
    inverse_mean, inverse_variance


# 3. Metrics ────────────────────────────────────────────────────────────────────────── #

include("metrics/metrics.jl")

export 
    mse, rmse, nrmse, q2

# 4. CABO ────────────────────────────────────────────────────────────────────────── #

include("propagation/augmentedspace.jl")
include("propagation/cabo/bayesianoptimization.jl") 
include("propagation/cabo/bayesiancubature.jl")
include("propagation/cabo/kl_sampling.jl")
include("propagation/cabo/cabo.jl")

export 
    ei_objective,
    precompute_h_pvc_terms, h_pvc, pvc_objective, u_objective,
    build_kl_sampler,
    compute_relaxed_bounds, θ_to_v, v_to_θ, x_to_u, u_to_x, 
    AbstractInputSpec, PreciseSpec, IntervalSpec, HybridSpec, InputSpec, spec_names, build_augmented_design, augmented_to_physical, augmented_to_epistemic, build_bounds,
    estimate_qoi, estimate_propagation_qoi, best_candidate, make_pso, cabo_loop


# models 
include("physicalmodels/test_models.jl")

export
    ishigami, forrester, g_function, g_function_E,
    model_ishigami, model_forrester, model_gfunction

# bootstrap
include("metamodels/ensemble/bootstrap.jl")

export
    BootstrapEnsemble, fit!, predict, evalaute!,
    gp_bootstrap, pce_bootstrap,
    calibration_coverage, calibration_report

end

