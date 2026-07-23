module SurrogateModelling

# ================================================================================= #
# -1. Dependencies
# ================================================================================= # 

using LinearAlgebra
using DataFrames
using Plots
using Random
using Printf
using DifferentiationInterface
using Clustering
using FastGaussQuadrature
using Distributions
using QuasiMonteCarlo
using Mooncake
using Statistics

using KernelFunctions
using AbstractGPs
using ParameterHandling
using Optim
using Zygote

using Metaheuristics

using UncertaintyQuantification

# ================================================================================= #
# 0. helper one-liner functions
# ================================================================================= # 

φ(z)     = pdf(Normal(), z)
Φ(z)     = cdf(Normal(), z)
Φ⁻¹(p)   = quantile(Normal(), p)
φ_vec(u) = prod(pdf.(Normal(), u))

# ================================================================================= #
# 1. metamodels
# ================================================================================= # 

# 1.1. Gaussian Process ----------------------------------------------------------- #
# ================================================================================= # 

# 1.1.1. Kernel Related
include("metamodels/gp//kernels/kernels.jl")
include("metamodels/gp/kernels/Matern52.jl")
include("metamodels/gp/kernels/Matern32.jl")
include("metamodels/gp/kernels/Matern12.jl")
include("metamodels/gp/kernels/SquaredExponential.jl")
include("metamodels/gp/kernels/Composite.jl")

# 1.1.2. Mean Related
include("metamodels/gp/means/means.jl")
include("metamodels/gp/means/ZeroMean.jl")
include("metamodels/gp/means/ConstMean.jl")

# 1.1.3. GP Constructor Related
include("metamodels/gp/gaussianprocess.jl")


# 1.2. Polynomial Chaos Expansion ------------------------------------------------- #
# ================================================================================= # 

# 1.2.1. PCE Degree related:
include("metamodels/pce/degrees/degrees.jl")
include("metamodels/pce/degrees/TotalDegree.jl")
include("metamodels/pce/degrees/TensorProduct.jl")
include("metamodels/pce/degrees/HyperbolicCross.jl")
include("metamodels/pce/degrees/QBall.jl")

# 1.2.2. PCE Basis related
include("metamodels/pce/bases/bases.jl")
include("metamodels/pce/bases/Hermite.jl")
include("metamodels/pce/bases/Legendre.jl")

# 1.2.3. PCE Solver related:
include("metamodels/pce/solvers/solvers.jl")
include("metamodels/pce/solvers/OLS.jl")
include("metamodels/pce/solvers/LASSO.jl")

# 1.2.4. PCE Constructor Related
include("metamodels/pce/polynomialchaosexpansion.jl")

# 1.3 Polynomial Chaos Kriging ---------------------------------------------------- #
# ================================================================================= # 

include("metamodels/pck/polynomialchaoskriging.jl")



# ================================================================================= #
# 2. scalings
# ================================================================================= #

include("scalings/scalings.jl")
include("scalings/minmax.jl")
include("scalings/zscore.jl")
include("scalings/scaling_pipeline.jl")

# ================================================================================= #
# 3. Metrics
# ================================================================================= #

include("metrics/metrics.jl")

# ================================================================================= #
# 3. CABO
# ================================================================================= #

include("propagation/augmentedspace.jl")
include("propagation/cabo/bayesianoptimization.jl") 
include("propagation/cabo/bayesiancubature.jl")
include("propagation/cabo/kl_sampling.jl")
include("propagation/cabo/cabo.jl")
include("propagation/cabo/cabo_plots.jl")

# ================================================================================= #
# 4. Physical Models
# ================================================================================= #

include("physicalmodels/test_models.jl")


end

