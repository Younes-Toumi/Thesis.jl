module SurrogateModelling

# ── Dependencies ──────────────────────────────────────────────
using Statistics
using LinearAlgebra
using DataFrames
using Plots
using Random
using UncertaintyQuantification
using DifferentiationInterface
using Clustering
using FastGaussQuadrature
using Distributions: Normal, Uniform, cdf, quantile

import Mooncake

# 1. metamodels ────────────────────────────────────────────────────────────────────────── #
# 1.1. Gaussian Process

# Kernel Related
include("metamodels/gp//kernels/kernels.jl")
include("metamodels/gp/kernels/Matern52.jl")
include("metamodels/gp/kernels/Matern32.jl")
include("metamodels/gp/kernels/SquaredExponential.jl")
include("metamodels/gp/kernels/Composite.jl")

export
    AbstractGPKernel, GPMatern52, GPMatern32, GPSquaredExponential, GPCompositeKernel,
    kernel_name

# Mean Related
include("metamodels/gp/means/means.jl")
include("metamodels/gp/means/ZeroMean.jl")
include("metamodels/gp/means/ConstMean.jl")

export 
    AbstractGPMean, GPZeroMean, GPConstMean,
    mean_name


# GP Constructor Related
using KernelFunctions
using AbstractGPs
using ParameterHandling
using Optim
using Zygote

include("metamodels/gp/gaussianprocess.jl")

export 
    GaussianProcess,
    fit!, predict


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


# 4. Adaptive Sampling
include("adaptive/particleswarmoptimization.jl")

export
    pso_optimize
end