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

import Mooncake

# 1. models ────────────────────────────────────────────────────────────────────────── #
# 1.1. metamodels
# 1.1.1. GPs and kernels

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


# Gaussian Process

using KernelFunctions
using AbstractGPs
using ParameterHandling
using Optim
using Zygote

include("models/metamodels/gp/gaussianprocess.jl")

export 
    GaussianProcess,
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