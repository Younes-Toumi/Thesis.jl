# Master Thesis SurrogateModelling.jl
*Surrogate Modelling for Complex Numerical Models under Imprecise Probabilities*

This repository contains the code developed for my Master's thesis in Computational Engineering at Leibniz University Hannover (LUH), carried out at the Institute for Risk and Reliability.

## Context
Uncertainty-quantification problems rarely come with precisely known probability distribution for every input. Some quantities are only known to lie within an interval or distribution family whose parameters are themselves uncertain. This mixture of aleatory and epistemic uncertainty is commonly represented via imprecise probabilities. Propagating this kind of uncertainty through an expensive numerical model (e.g. an FE simulation) typically requires a double-loop procedure: an outer loop searching the epistemic space for the bounding case, and an inner loop propagating the aleatory uncertainty at each candidate. Done directly on the true model, this is prohibitively expensive. To addresses that cost surrogate models are built in an augmented standard-normal space, and by using the Confidence-based Adaptive Bayesian Optimization (CABO) scheme to actively steer both loops with as few true model evaluations as possible all of which implemented in Julia.

## What this Repository Implements
- **Gaussian Process (GP)**: regression with ARD kernels (Matérn 5/2, 3/2, 1/2, Squared Exponential).
  
- **Polynomial Chaos Expansion (PCE)**: with multiple truncation schemes (Total Degree, Tensor Product, Hyperbolic Cross, Q-Ball / hyperbolic cross), Hermite/Legendre bases, and both OLS and LASSO solvers.
  
- **Polynomial Chaos Kriging (PCK)**: a low-degree PCE trend combined with a GP residual, used where a pure PCE cannot resolve strongly localised or highly-interacting response surfaces without an infeasible degree.
  
- **Augmented-Space Construction**: a small type hierarchy (PreciseSpec, IntervalSpec, HybridSpec) mapping arbitrary precise/interval/hybrid UQ.jl inputs into a joint standard-normal-space representation (u, v).
  
- **CABO**: the adaptive bounding loop itself:
  - _Bayesian-optimization "BO engine"_ that searches the epistemic space for the bounding QoI value.
  - _Bayesian-cubature "BC engine"_ that decides where to spend the next expensive model evaluation. Supports mean, variance, and failure-probability (Pf) quantities of interest.
  - _EOLE (Expansion Optimal Linear Estimation) sampling_: a truncated Karhunen–Loève representation of the GP posterior.

## Repository Structure

```
Thesis.jl
├── src/                        # the SurrogateModelling.jl package
│   ├── metamodels/
│   │   ├── gp/                 # Gaussian Process: kernels, means, fit!/predict/evaluate!
│   │   ├── pce/                # Polynomial Chaos Expansion: degrees, bases, solvers
│   │   └── pck/                # Polynomial Chaos Kriging
│   ├── propagation/
│   │   ├── augmentedspace.jl   # input-spec system, augmented-space mappings
│   │   └── cabo/               # CABO loop, EOLE/KL sampling, BO/BC acquisition functions
│   ├── scalings/               # input/output scaling pipelines
│   ├── metrics/                # MSE, Q², LOO-CV variants
│   └── physicalmodels/         # benchmark test problems (Ishigami, Forrester, Gaussian-Mixture)
├── Project.toml
└── Manifest.toml           # committed intentionally -- pins exact package versions
```

## Getting Started

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Minimal Working Example
```julia
using SurrogateModelling
using UncertaintyQuantification

# a simple 2D test function
x1 = RandomVariable(Uniform(-1, 1), :x1)
x2 = RandomVariable(Uniform(-1, 1), :x2)
model = Model(rv -> rv.x1.^2 .+ rv.x2.^2, :y)

data_train = sample([x1, x2], MonteCarlo(50))
data_test = sample([x1, x2], MonteCarlo(1000))

UncertaintyQuantification.evaluate!(model, data_train)
UncertaintyQuantification.evaluate!(model, data_test)

gp = GaussianProcess(data_train, :y; kernel_type=GPMatern52())
fit!(gp)

μ, σ = predict(gp, Matrix(data_test[:, [:x1, :x2]]))
println("Q² = ", q2(data_test.y, μ))
```


## Key References

### Framework & Theory
- **Imprecise probability \& p-boxes**: Beer, Ferson \& Kreinovich (2013); Faes et al. (2021)
- **Aleatory / epistemic distinction**: Li et al. (2026)
- **Gaussian process regression**: Rasmussen \& Williams (2006)
- **Polynomial chaos expansion**: Wiener (1938); Sudret (2012)
- **Polynomial chaos kriging**: Schöbi \& Sudret (ETH Zürich)
- **CABO lineage, NIPI, NISS, twin-engine**: Wei et al. (2019); Wei et al. (2021); Hong et al. (2023)

### Methods Used
- **EOLE / random field discretisation**: Li \& Der Kiureghian (1993)
- **U-function / AK-MCS**: Echard et al. (2013)
- **Bayesian cubature**: O'Hagan (1991)
- **Particle swarm optimisation**: Poli, Kennedy \& Blackwell (2007)
- **Space-filling designs**: McKay et al. (1979); Santner et al. (2018)
- **Julia UQ ecosystem**: Behrensdorf et al., `UncertaintyQuantification.jl`

## Acknowledgements

Built on top of `UncertaintyQuantification.jl`, `AbstractGPs.jl`, `KernelFunctions.jl`, and `Metaheuristics.jl`.