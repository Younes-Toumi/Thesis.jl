# ==============================================================================
# Demonstrates the augmented-space input-spec system: PreciseSpec, IntervalSpec,
# and HybridSpec, how they map into the joint standard-normal (u,v) space, and
# how to go back and forth between physical space, epistemic space, and the
# augmented space a surrogate is actually trained on.
# ==============================================================================

using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Statistics

# ==============================================================================
# 1. The three spec types
# ==============================================================================
# PreciseSpec  - fully aleatory: x ~ dist, no epistemic component.
#                 1 u-column, 0 v-columns.
#
# IntervalSpec - fully epistemic: x = theta in [theta_L, theta_U], no randomness.
#                 0 u-columns, 1 v-column.
#
# HybridSpec   - x | theta ~ realize(theta), where theta itself is only known
#                 to lie in an interval with theta in [-1.5, 1.5]). 
#                 1 u-column, K v-columns (K = number
#                 of epistemic parameters in realize()).
#
# All three are constructed via InputSpec(...), dispatching on the actual
# UncertaintyQuantification.jl input type

x1 = RandomVariable(Normal(0.0, 1.0), :x1)      # -> PreciseSpec
x2 = IntervalVariable(-1.5, 1.5, :x2)           # -> IntervalSpec
x3 = RandomVariable(ProbabilityBox{Normal}(     # -> HybridSpec (K=1: the mean)
    Dict(
        :μ => Interval(-1.0, 1.0), 
        :σ => 0.2)
    ), :x3
)                                                        

specs = InputSpec.([x1, x2, x3])

for s in specs
    println(typeof(s), "  n_u_dims=", n_u_dims(s), "  n_v_dims=", n_v_dims(s))
end

# ==============================================================================
# 2. Column-name bookkeeping
# ==============================================================================
# spec_names derives the augmented-space column names from the specs. x_names
# matches the physical-space DataFrame exactly; 
# u_names/v_names are positional (u1, u2, ... / v1, v2, ...), except
# when there's only ONE u or ONE v dimension overall, in which case the
# numeric subscript is dropped (just :u / :v) since no disambiguation is needed.
x_names, w_names, u_names, v_names = spec_names(specs)

println("\nx_names: ", x_names)   # [:x1, :x2, :x3] - matches the declared names exactly
println("u_names: ", u_names)     # one u-dim from x1 (PreciseSpec) + one from x3 (HybridSpec)
println("v_names: ", v_names)     # one v-dim from x2 (IntervalSpec) + one from x3 (HybridSpec)
println("w_names: ", w_names)     # vcat(u_names, v_names) - the full augmented-space column order

# ==============================================================================
# 3. Building an augmented training design
# ==============================================================================
# build_augmented_design draws a joint design over the epistemic space (a
# single Latin Hypercube spanning every epistemic dimension across all specs),
# and independent aleatory draws for the aleatory dimensions, then evaluates
# the physical model if one is provided.
physical_model = Model(df -> df.x1 .+ df.x2 .+ df.x3, :y)

n_pool = 1000
data_aug, data_phys = build_augmented_design(physical_model, specs, n_pool)

println("\nAugmented-space columns: ", names(data_aug))   # u..., v..., :y
println("Physical-space columns:  ", names(data_phys))    # :x1, :x2, :x3, :y

# sanity check: x2 (IntervalSpec) must stay within its declared bounds
# sampling the outer epistemic design from relaxed bounds
# to make the true interval land on a finite v-support
x2_range = extrema(data_phys.x2)
println("\nx2 declared bounds: [-1.5, 1.5]")
println("x2 sampled range:   ", x2_range,)

# ==============================================================================
# 4. Round-tripping: augmented space <-> physical space <-> epistemic space
# ==============================================================================
# augmented_to_physical maps one augmented-space row (u,v) back to physical x.
# augmented_to_epistemic maps the v-part alone to the epistemic theta values.
w_example = Vector(data_aug[1, w_names])
x_reconstructed = augmented_to_physical(w_example, specs)
θ_reconstructed  = augmented_to_epistemic(w_example[length(u_names)+1:end], specs)

println("\nRound-trip check on row 1:")
println("    physical x from data_phys:             ", Vector(data_phys[1, [:x1, :x2, :x3]]))
println("    physical x from augmented_to_physical: ", x_reconstructed)

# ==============================================================================
# 5. Search-space bounds for the epistemic (v) and aleatory (u) dimensions
# ==============================================================================
# build_bounds returns Metaheuristics.jl-compatible box constraints, used by
# CABO's inner PSO searches (see demo/05_cabo.jl).
bounds_u, bounds_v = build_bounds(specs)
println("\nv-space bounds (epistemic search box): ", bounds_v)
println("u-space bounds (aleatory search box):  ", bounds_u)

# ==============================================================================
# 6. Mean and Std of the augmented coordinates
# ==============================================================================
print("\n")
println("w_names:", w_names)
println("mean: ", round.(mean(Matrix(data_aug[:, w_names]), dims = 1), digits=3))
println("std:  ", round.(std(Matrix(data_aug[:, w_names]),  dims = 1), digits=3))

# NOTE:
# the std of the epistemic coordinate will sit at 0.916 because the support is ± 2.2 to get 1.0 std, change it to ± 4
# [Φ(-2.2), Φ(2.2)] ≈ [0.01390, 0.98610]
# [Φ(-4.0), Φ(4.0)] ≈ [0.00003, 0.99997]