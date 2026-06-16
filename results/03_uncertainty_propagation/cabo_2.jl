# ==============================================================================
# augmented_space_v2.jl
# Generalized augmented-space construction for CABO, supporting THREE input
# categories, unified under UncertaintyQuantification.jl's type system:
#
#   1. PRECISE   — RandomVariable(dist, name)                       fully aleatory
#   2. INTERVAL  — IntervalVariable(lb, ub, name)                    fully epistemic
#   3. HYBRID    — RandomVariable(ProbabilityBox{D}(params), name)   mixed
#
# Each spec contributes 0 or 1 u-column (aleatory, SNS) and 0..K v-columns
# (epistemic, SNS), where K = number of Interval-valued parameters for a
# HYBRID spec (K=1 reproduces your old single-θ `dist_factory` convention;
# K=0 degenerates to PRECISE automatically).
#
# ⚠ FIELD-NAME ASSUMPTIONS — VERIFY BEFORE RELYING ON THE UQ.jl DISPATCH BELOW
#   RandomVariable    : fields  .dist, .name
#   IntervalVariable  : fields  .lb, .ub, .name
#   ProbabilityBox{D} : field   .parameters :: Dict{Symbol,Any}
#   Interval          : fields  .lb, .ub
#
#   Run this first and adjust the methods marked # CHECK if any name differs:
#     println(fieldnames(typeof(x1)))        # RandomVariable
#     println(fieldnames(typeof(x2)))        # RandomVariable (wraps ProbabilityBox)
#     println(fieldnames(typeof(x2.dist)))   # ProbabilityBox
#     println(fieldnames(typeof(x3)))        # IntervalVariable
# ==============================================================================

using SurrogateModelling
using SurrogateModelling: g_function
using Distributions
using UncertaintyQuantification
using QuasiMonteCarlo
using Random
using DataFrames
using Printf
using ParameterHandling
using Metaheuristics
using KernelFunctions
using LinearAlgebra

φ(z)     = pdf(Normal(), z)
Φ(z)     = cdf(Normal(), z)
Φ⁻¹(p)   = quantile(Normal(), p)
φ_vec(u) = prod(pdf.(Normal(), u))

# ==============================================================================
# 1.  Relaxed bounds  (unchanged from your original)
# ==============================================================================
function compute_relaxed_bounds(θ_L::Real, θ_U::Real; v_L::Real=-2.2, v_U::Real=2.2)
    L    = Float64(θ_U - θ_L)
    span = L / (Φ(v_U) - Φ(v_L))
    lb   = Float64(θ_L) -        Φ(v_L)  * span
    ub   = Float64(θ_U) + (1.0 - Φ(v_U)) * span
    return lb, ub
end

# ==============================================================================
# 2.  θ ↔ v   and   3.  x ↔ u    (unchanged)
# ==============================================================================
θ_to_v(θ, lb, ub) = Φ⁻¹((θ - lb) / (ub - lb))
v_to_θ(v, lb, ub) = lb + (ub - lb) * Φ(v)
x_to_u(x, dist)   = Φ⁻¹(cdf(dist, x))
u_to_x(u, dist)   = quantile(dist, Φ(u))

# ==============================================================================
# 4.  Spec hierarchy
# ==============================================================================
abstract type AbstractInputSpec end

"Fully aleatory: x ~ dist (fixed, no epistemic component). 1 u-col, 0 v-cols."
struct PreciseSpec <: AbstractInputSpec
    dist :: Distributions.Distribution
    name :: Symbol
end

"Fully epistemic: x = θ ∈ [θ_L, θ_U]. 0 u-cols, 1 v-col."
struct IntervalSpec <: AbstractInputSpec
    θ_L  :: Float64
    θ_U  :: Float64
    v_L  :: Float64
    v_U  :: Float64
    name :: Symbol
end

"""
Hybrid: x | θ ~ realize(θ_vec). K = length(param_names) epistemic parameters.
K=1 reproduces the old single-θ `dist_factory` convention exactly.
1 u-column, K v-columns.
"""
struct HybridSpec <: AbstractInputSpec
    realize      :: Function          # θ_vec::Vector{Float64} -> Distributions.Distribution
    param_names  :: Vector{Symbol}    # bookkeeping/printing only
    θ_L          :: Vector{Float64}
    θ_U          :: Vector{Float64}
    v_L          :: Vector{Float64}
    v_U          :: Vector{Float64}
    name         :: Symbol
end

make_dist(s::HybridSpec, θ_vec::AbstractVector{Float64}) = s.realize(collect(θ_vec))

n_u_dims(::PreciseSpec)  = 1
n_u_dims(::IntervalSpec) = 0
n_u_dims(::HybridSpec)   = 1
n_v_dims(::PreciseSpec)  = 0
n_v_dims(::IntervalSpec) = 1
n_v_dims(s::HybridSpec)  = length(s.param_names)

# ==============================================================================
# 5.  Backward-compatible InputSpec(...) constructor — OLD calling convention
#     InputSpec(-1.5, 1.5, nothing)              → IntervalSpec
#     InputSpec(-1.5, 1.5, θ -> Normal(θ, σ))    → HybridSpec (K=1)
#     Both forms from earlier in this conversation keep working unchanged.
# ==============================================================================
function InputSpec(θ_L, θ_U, dist_factory=nothing; v_L=-2.2, v_U=2.2, name=gensym(:x))
    if dist_factory === nothing
        return IntervalSpec(Float64(θ_L), Float64(θ_U), Float64(v_L), Float64(v_U), name)
    else
        realize = θ_vec -> dist_factory(θ_vec[1])
        return HybridSpec(realize, [:θ], [Float64(θ_L)], [Float64(θ_U)],
                           [Float64(v_L)], [Float64(v_U)], name)
    end
end

# ==============================================================================
# 6.  NEW: dispatch constructors from UncertaintyQuantification.jl input types
# ==============================================================================

"PRECISE or HYBRID, depending on whether x.dist wraps a ProbabilityBox."
function InputSpec(x::RandomVariable; v_L=-2.2, v_U=2.2)
    if x.dist isa ProbabilityBox                                # CHECK: field `.dist`
        return InputSpec(x.dist, x.name; v_L=v_L, v_U=v_U)       # CHECK: field `.name`
    else
        return PreciseSpec(x.dist, x.name)                       # CHECK: `.dist`, `.name`
    end
end

"Pure interval input."
function InputSpec(x::IntervalVariable; v_L=-2.2, v_U=2.2)
    return IntervalSpec(Float64(x.lb), Float64(x.ub), Float64(v_L), Float64(v_U), x.name)
    # CHECK: fields `.lb`, `.ub`, `.name`
end

"""
ProbabilityBox{D} with a Dict mixing fixed values and `Interval`s.
Builds the K epistemic parameters into a HybridSpec; if NO parameter is an
Interval, degenerates to a PreciseSpec instead (handles the edge case where
a ProbabilityBox happens to have every parameter fixed).
"""
function InputSpec(pb::ProbabilityBox{D}, name::Symbol; v_L=-2.2, v_U=2.2) where D
    epi_names    = Symbol[]
    fixed_params = Dict{Symbol,Float64}()
    θ_L, θ_U     = Float64[], Float64[]

    for (k, val) in pb.parameters                       # CHECK: field `.parameters`
        if val isa Interval                               # CHECK: type `Interval`, fields below
            push!(epi_names, k)
            push!(θ_L, Float64(val.lb)); push!(θ_U, Float64(val.ub))   # CHECK: `.lb`, `.ub`
        else
            fixed_params[k] = Float64(val)
        end
    end

    # Assumes constructor positional order == struct field order — true for
    # Normal, Uniform, Gamma, etc.; verify for exotic distribution types.
    field_order = fieldnames(D)

    if isempty(epi_names)
        args = [fixed_params[f] for f in field_order]
        return PreciseSpec(D(args...), name)
    end

    realize = θ_vec -> begin
        all_p = merge(fixed_params, Dict(zip(epi_names, θ_vec)))
        D([all_p[f] for f in field_order]...)
    end

    n = length(epi_names)
    return HybridSpec(realize, epi_names, θ_L, θ_U,
                       fill(Float64(v_L), n), fill(Float64(v_U), n), name)
end

# ==============================================================================
# 7.  Column-name bookkeeping  (flat u1,u2,...,v1,v2,... — matches your convention)
# ==============================================================================
function spec_names(specs::Vector{<:AbstractInputSpec})
    u_names = Symbol[]
    v_names = Symbol[]

    for (i, s) in enumerate(specs)
        if n_u_dims(s) == 1
            push!(u_names, Symbol("u$i"))
        end
    end

    for (i, s) in enumerate(specs)
        k = n_v_dims(s)
        if k == 1
            push!(v_names, Symbol("v$i"))
        elseif k > 1
            for j in 1:k
                push!(v_names, Symbol("v$(i)_$(j)"))   # e.g. v2_1, v2_2 if x2 has 2 epistemic params
            end
        end
    end
    w_names = vcat(u_names, v_names)
    return w_names, u_names, v_names
end

# ==============================================================================
# 8.  build_augmented_design — generalized for all three spec types
# ==============================================================================
function build_augmented_design(
    physical_model,
    specs     :: Vector{<:AbstractInputSpec},
    n_samples :: Int;
    seed      :: Int    = 42,
    y_symbol  :: Symbol = :y
)
    # ── Collect epistemic dims across ALL specs for a single JOINT LHS ───────
    relaxed_lbs, relaxed_ubs = Float64[], Float64[]
    spec_epi_range = Vector{UnitRange{Int}}(undef, length(specs))
    col = 0
    for (i, s) in enumerate(specs)
        k = n_v_dims(s)
        if k == 0
            spec_epi_range[i] = 1:0
            continue
        end
        θ_Ls = s isa IntervalSpec ? [s.θ_L] : s.θ_L
        θ_Us = s isa IntervalSpec ? [s.θ_U] : s.θ_U
        v_Ls = s isa IntervalSpec ? [s.v_L] : s.v_L
        v_Us = s isa IntervalSpec ? [s.v_U] : s.v_U
        for j in 1:k
            lb, ub = compute_relaxed_bounds(θ_Ls[j], θ_Us[j]; v_L=v_Ls[j], v_U=v_Us[j])
            push!(relaxed_lbs, lb); push!(relaxed_ubs, ub)
        end
        spec_epi_range[i] = (col+1):(col+k)
        col += k
    end
    n_epi_total = col

    Random.seed!(seed)
    θ_samples = n_epi_total > 0 ?
        QuasiMonteCarlo.sample(n_samples, relaxed_lbs, relaxed_ubs, LatinHypercubeSample())' :
        Matrix{Float64}(undef, n_samples, 0)

    v_samples = similar(θ_samples)
    for j in 1:n_epi_total
        v_samples[:, j] = θ_to_v.(θ_samples[:, j], relaxed_lbs[j], relaxed_ubs[j])
    end

    w_names, u_names, v_names = spec_names(specs)
    u_samples = Matrix{Float64}(undef, n_samples, length(u_names))
    x_samples = Matrix{Float64}(undef, n_samples, length(specs))

    u_col = 0
    for (i, s) in enumerate(specs)
        if s isa PreciseSpec
            u_col += 1
            for row in 1:n_samples
                x_ij = rand(s.dist)
                x_samples[row, i]     = x_ij
                u_samples[row, u_col] = x_to_u(x_ij, s.dist)
            end

        elseif s isa IntervalSpec
            θ_col = spec_epi_range[i].start
            x_samples[:, i] = θ_samples[:, θ_col]

        elseif s isa HybridSpec
            u_col += 1
            θ_cols = spec_epi_range[i]
            for row in 1:n_samples
                θ_vec = θ_samples[row, θ_cols]
                dist  = make_dist(s, θ_vec)
                x_ij  = rand(dist)
                x_samples[row, i]     = x_ij
                u_samples[row, u_col] = x_to_u(x_ij, dist)
            end
        end
    end

    aug_df, phys_df = DataFrame(), DataFrame()
    for (k, nm) in enumerate(u_names); aug_df[!, nm] = u_samples[:, k]; end
    for (k, nm) in enumerate(v_names); aug_df[!, nm] = v_samples[:, k]; end
    for (i, s) in enumerate(specs);    phys_df[!, s.name] = x_samples[:, i]; end

    if physical_model !== nothing
        y = [physical_model(x_samples[row, :]...) for row in 1:n_samples]
        aug_df[!, y_symbol]  = y
        phys_df[!, y_symbol] = y
    end

    return aug_df, phys_df
end

# ==============================================================================
# 9.  augmented_to_physical / augmented_to_epistemic — generalized
# ==============================================================================
function augmented_to_physical(w, specs::Vector{<:AbstractInputSpec})
    w_names, u_names, v_names = spec_names(specs)
    n_u, n_v = length(u_names), length(v_names)
    u_vec, v_vec = w[1:n_u], w[n_u+1 : n_u+n_v]

    x = Vector{Float64}(undef, length(specs))
    u_idx, v_idx = 0, 0
    for (i, s) in enumerate(specs)
        if s isa PreciseSpec
            u_idx += 1
            x[i] = u_to_x(u_vec[u_idx], s.dist)

        elseif s isa IntervalSpec
            v_idx += 1
            lb, ub = compute_relaxed_bounds(s.θ_L, s.θ_U; v_L=s.v_L, v_U=s.v_U)
            x[i]  = v_to_θ(v_vec[v_idx], lb, ub)

        elseif s isa HybridSpec
            u_idx += 1
            k = length(s.param_names)
            θ_vec = Vector{Float64}(undef, k)
            for j in 1:k
                v_idx += 1
                lb, ub = compute_relaxed_bounds(s.θ_L[j], s.θ_U[j]; v_L=s.v_L[j], v_U=s.v_U[j])
                θ_vec[j] = v_to_θ(v_vec[v_idx], lb, ub)
            end
            dist = make_dist(s, θ_vec)
            x[i] = u_to_x(u_vec[u_idx], dist)
        end
    end
    return x
end

function augmented_to_epistemic(v, specs::Vector{<:AbstractInputSpec})
    θ_all = Float64[]
    v_idx = 0
    for s in specs
        if s isa IntervalSpec
            v_idx += 1
            lb, ub = compute_relaxed_bounds(s.θ_L, s.θ_U; v_L=s.v_L, v_U=s.v_U)
            push!(θ_all, v_to_θ(v[v_idx], lb, ub))
        elseif s isa HybridSpec
            for j in 1:length(s.param_names)
                v_idx += 1
                lb, ub = compute_relaxed_bounds(s.θ_L[j], s.θ_U[j]; v_L=s.v_L[j], v_U=s.v_U[j])
                push!(θ_all, v_to_θ(v[v_idx], lb, ub))
            end
        end
        # PreciseSpec contributes nothing — it has no epistemic component
    end
    return θ_all
end


function build_bounds(specs::Vector{<:AbstractInputSpec})
    lb_u, ub_u = Float64[], Float64[]
    lb_v, ub_v = Float64[], Float64[]
 
    for s in specs
        if n_u_dims(s) == 1
            push!(lb_u, -4.0); push!(ub_u, 4.0)   # effective N(0,1) support — invariant across spec types
        end
        if s isa IntervalSpec
            push!(lb_v, s.v_L); push!(ub_v, s.v_U)
        elseif s isa HybridSpec
            append!(lb_v, s.v_L); append!(ub_v, s.v_U)
        end
    end
 
    return boxconstraints(lb=lb_u, ub=ub_u), boxconstraints(lb=lb_v, ub=ub_v)
end

function best_candidate(qoi_type, gp_samples, u_samples, v_data, direction; α=1.0)
    n = size(v_data, 1)
    μ_qoi = Vector{Float64}(undef, n)
    σ_qoi = Vector{Float64}(undef, n)
    for i in 1:n
        μ_qoi[i], σ_qoi[i] = estimate_propagation_qoi(qoi_type, gp_samples, u_samples, v_data[i, :])
    end
    candidates = μ_qoi .+ α .* σ_qoi
    idx = (direction == :min) ? argmin(candidates) : argmax(candidates)
    return idx, μ_qoi, σ_qoi
end

function make_pso(; N::Int=50, ω=0.8, C1=2.0, C2=2.0)
    p = PSO(N=N, ω=ω, C1=C1, C2=C2)
    p.options.iterations = 100
    return p
end


function cabo_loop(
    physical_model,
    gp_init,
    data_aug_train,
    specs;
    Ng = 100,
    Nx = 100,
    qoi_type = :mean,
    max_iter::Int = 20,
    direction::Symbol = :min,
    tol_BO::Float64 = 5e-3,
    tol_BC::Float64 = 2.5e-2
)
    # ── Everything dimension-dependent derives from `specs` ──────────────────
    w_names, u_names, v_names  = spec_names(specs)
    bounds_u, bounds_v = build_bounds(specs)
 
    data = copy(data_aug_train)
    gp   = gp_init
    sign_dir = (direction == :min) ? +1 : -1
 
    θ_history    = Vector{Vector{Float64}}()
    L_BO_history = Float64[]
    L_BC_history = Float64[]
 
    W_aug, _  = build_augmented_design(nothing, specs, Nx; seed=42)
    u_samples = Matrix(W_aug[:, u_names])
 
    for iter in 1:max_iter
        W_support    = Matrix(data[:, w_names])
        gp_samples = build_kl_sampler(gp, Matrix(W_aug), W_support; N_samples=Ng)
 
        println("\n━━━ CABO Iteration $iter / $max_iter ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
 
        # ════ Part 1: BO engine ═══════════════════════════════════════════════
        v_data = Matrix(data[:, v_names])
        v_star_index, μ_qoi, σ_qoi = best_candidate(qoi_type, gp_samples, u_samples, v_data, direction; α=1.0)
 
        v_star     = Vector(data[v_star_index, v_names])
        μ_qoi_star = μ_qoi[v_star_index]
        σ_qoi_star = σ_qoi[v_star_index]
        θ_star     = augmented_to_epistemic(v_star, specs)
 
        @printf("    Incumbent θ* = %s    μ_qoi(θ*) ≈ %.2e    σ_qoi(θ*) ≈ %.2e\n",
                string(round.(θ_star, digits=3)), μ_qoi_star, σ_qoi_star)
 
        res_v = Metaheuristics.optimize(
            v -> ei_objective(qoi_type, gp_samples, u_samples, v, μ_qoi_star, sign_dir),
            bounds_v,
            make_pso()
        )
 
        v_plus = minimizer(res_v)
        θ_plus = augmented_to_epistemic(v_plus, specs)
        L_BO   = -minimum(res_v)
 
        # μ_qoi_plus, σ_qoi_plus = estimate_propagation_qoi(qoi_type, gp_samples, u_samples, v_plus)
        COV_star = σ_qoi_star / abs(μ_qoi_star)
 
        println("    Acquisition θ⁺ = $(round.(θ_plus, digits=4))    EI = $(round(L_BO, digits=4))" *
                "    COV = $(round(COV_star, sigdigits=4))")
 
        if L_BO < tol_BO && COV_star < tol_BC
            println("\n✓ converged")
            break
        end
 
        # ════ Part 2: BC engine ═══════════════════════════════════════════════
        if qoi_type == :pf
            res_u = Metaheuristics.optimize(
                u -> u_objective(gp, u, v_plus),
                bounds_u, make_pso()
            )            
        else
            W     = Matrix(data[:, w_names])
            K_bc  = kernelmatrix(gp.kernel_posterior, RowVecs(W))
            cholK = cholesky(Symmetric(K_bc + 1e-8I))
    
            W_prime, v2_sum = precompute_h_pvc_terms(gp, cholK, W, v_plus, u_samples)
    
            res_u = Metaheuristics.optimize(
                u -> pvc_objective(gp, cholK, W, u, v_plus, W_prime, v2_sum),
                bounds_u, make_pso()
            )
        end

        u_plus = minimizer(res_u)
        w_plus = vcat(u_plus, v_plus)
        x_plus = augmented_to_physical(w_plus, specs)
        y_plus = physical_model(x_plus...)
 
        # Generalized row constructions
        new_row = merge(
            NamedTuple(zip(u_names, u_plus)),
            NamedTuple(zip(v_names, v_plus)),
            (y = y_plus,)
        )

        append!(data, DataFrame([new_row]))
 
        refit!(gp, reshape(w_plus, 1, :), [y_plus])
 
        push!(θ_history, copy(collect(θ_plus)))
        push!(L_BO_history, L_BO)
        push!(L_BC_history, COV_star)
    end
 
    # ── Final bound: pure mean (α=0), rebuild sampler on the FINAL gp first ──
    W_support_final     = Matrix(data[:, w_names])
    gp_samples_final  = build_kl_sampler(gp, Matrix(W_aug), W_support_final; N_samples=Ng)
    v_data_final       = Matrix(data[:, v_names])
 
    v_bound_index, μ_qoi_bound, _ = best_candidate(qoi_type, gp_samples_final, u_samples, v_data_final, direction; α=0.0)
    μ_qoi_bound_final = μ_qoi_bound[v_bound_index]
    θ_bound            = augmented_to_epistemic(Vector(data[v_bound_index, v_names]), specs)
 
    println("\n  ► $(uppercase(string(direction))) bound ≈ $(round(μ_qoi_bound_final, sigdigits=5))" *
            "  at  θ = $(round.(θ_bound, digits=4))")
 


    return (
        gp = gp, data = data,
        θ_bound = θ_bound, μ_bound = μ_qoi_bound_final,
        θ_history = θ_history, L_BO_history = L_BO_history, L_BC_history = L_BC_history
    )
end


# ==============================================================================
# 10. Usage — mirrors your UQ.jl example end to end
# ==============================================================================


# x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
# x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)
# specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input

x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.0, 1.0), :σ => 1.0)), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.0, 1.0), :σ => 1.0)), :x2)
specs = InputSpec.([x1, x2])

# x1 = RandomVariable(Normal(0.0, 1.0), :x1)
# x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.3, 1.8), :σ => 2.0)), :x2)
# x3 = IntervalVariable(-0.5, 1.3, :x3)
# specs = InputSpec.([x1, x2, x3])     # broadcasts dispatch over each UQ.jl input
 
w_names, u_names, v_names = spec_names(specs)
println("u_names = ", u_names, "   v_names = ", v_names)


function physical_model(x1, x2) 
    return x1 - x2 - (-0.0)
    # return x1 * (x2^2 + x2 + cos(pi * x3) - 7)

end


n_train, n_test = 10, 1001


data_aug_train, data_phys_train =    build_augmented_design(physical_model, specs, n_train; seed=42)
data_aug_test,  data_phys_test  =    build_augmented_design(physical_model, specs, n_test; seed=123)


# # initialize GP on θ-space
kernel() = GPMatern52() #  + GPSquaredExponential()
metamodel = GaussianProcess(data_aug_train, :y, kernel_type=kernel())
@time "fit!" fit!(metamodel)

μ_test, σ_test = @time "predict:" predict(metamodel, Matrix(data_aug_test[:, w_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")

cabo_min = @time "cabo min loop" cabo_loop(
    physical_model,        # physical_model — explicit, no longer a global lookup
    metamodel,
    data_aug_train,
    specs;
    Ng = 1000, Nx = 1500, qoi_type = :pf,
    max_iter = 20, direction = :min,
    tol_BO = 5e-3, tol_BC = 2.5e-2
)


cabo_max = @time "cabo max loop" cabo_loop(
    physical_model,        # physical_model — explicit, no longer a global lookup
    metamodel,
    cabo_min.data,
    specs;
    Ng = 1000, Nx = 1500, qoi_type = :pf,
    max_iter = 20, direction = :max,
    tol_BO = 5e-3, tol_BC = 2.5e-2
)


# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("CABO results")
println("="^60)
println("MIN  Pf ≈ $(round(cabo_min.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_min.θ_bound, digits=3))")
println("MAX  Pf ≈ $(round(cabo_max.μ_bound, digits=4))" *
        "  at θ = $(round.(cabo_max.θ_bound, digits=3))")

println("="^60)



σ1, σ2 = 1.0, 1.0
θ1_L, θ1_U = -1.0, 1.0
θ2_L, θ2_U = -1.0, 1.0

Pf_analytical(θ1, θ2; σ1=σ1, σ2=σ2) = Φ((θ2 - θ1 + -0.0) / sqrt(σ1^2 + σ2^2))

function Pf_bounds_analytical(θ1_L, θ1_U, θ2_L, θ2_U; σ1=σ1, σ2=σ2)
    Pf_min = Pf_analytical(θ1_U, θ2_L; σ1=σ1, σ2=σ2)   # θ2-θ1 most negative
    Pf_max = Pf_analytical(θ1_L, θ2_U; σ1=σ1, σ2=σ2)   # θ2-θ1 most positive
    return Pf_min, Pf_max
end

Pf_min, Pf_max = Pf_bounds_analytical(θ1_L, θ1_U, θ2_L, θ2_U)

println("Analytical results")
println("="^60)
println("Pf_min ≈ $(round(Pf_min, sigdigits=4))  at (θ1, θ2) = ($θ1_U, $θ2_L)")
println("Pf_max ≈ $(round(Pf_max, sigdigits=4))  at (θ1, θ2) = ($θ1_L, $θ2_U)")