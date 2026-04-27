module SurrogateModelling

# ── Dependencies ──────────────────────────────────────────────
using Statistics
using LinearAlgebra
using DataFrames
using Plots
using Random
using UncertaintyQuantification

# 1. models ────────────────────────────────────────────────────────────────────────── #
# 1.1. metamodels
# 1.1.1. gp

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
    fit_pipeline, transform_input, transform_output

end