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
include("models/metamodels/gp/kernels.jl")
include("models/metamodels/gp/kernels/Matern52.jl")
include("models/metamodels/gp/kernels/Matern32.jl")
include("models/metamodels/gp/kernels/SquaredExponential.jl")
include("models/metamodels/gp/kernels/Composite.jl")

export
    AbstractGPKernel, GPMatern52, GPMatern32, GPSquaredExponential, GPCompositeKernel,
    kernel_name

# Mean Related
include("models/metamodels/gp/means.jl")
include("models/metamodels/gp/means/ZeroMean.jl")
include("models/metamodels/gp/means/ConstMean.jl")

export 
    AbstractGPMean, GPZeroMean, GPConstMean,
    mean_name


# GP Constructor Related
using KernelFunctions
using AbstractGPs
using ParameterHandling
using Optim
using Zygote

include("models/metamodels/gp/gaussianprocess.jl")

export 
    GaussianProcess,
    fit!, predict


# 1.2. Polynomial Chaos Expansion

# Degree related:
include("models/metamodels/pce/degrees.jl")
include("models/metamodels/pce/degrees/TotalDegree.jl")
include("models/metamodels/pce/degrees/TensorProduct.jl")
include("models/metamodels/pce/degrees/HyperbolicCross.jl")
include("models/metamodels/pce/degrees/QBall.jl")

export
    AbstractPCEDegree,
    TotalDegree, TensorProduct, HyperbolicCross, QBall,
    n_terms, degree_name, multivariate_indices

# Basis related
include("models/metamodels/pce/bases.jl")
include("models/metamodels/pce/bases/Hermite.jl")
include("models/metamodels/pce/bases/Legendre.jl")

export
    AbstractPCEBasis, HermiteBasis, LegendreBasis,
    build_design_matrix

# Solver related:
include("models/metamodels/pce/solvers.jl")
include("models/metamodels/pce/solvers/OLS.jl")
include("models/metamodels/pce/solvers/LASSO.jl")

export
    OSLSolver, LASSOSolver


# Constructor Related
include("models/metamodels/pce/polynomialchaosexpansion.jl")

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

end