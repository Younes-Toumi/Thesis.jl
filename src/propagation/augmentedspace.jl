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
    x_names = Symbol[]

    for (i, s) in enumerate(specs)
        push!(x_names, Symbol("x$i"))
    end


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
    return x_names, w_names, u_names, v_names
end

# ==============================================================================
# 8.  build_augmented_design — generalized for all three spec types
# ==============================================================================
function build_augmented_design(
    physical_model,
    specs     :: Vector{<:AbstractInputSpec},
    n_samples :: Int;
    seed      :: Int = 42,
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

    # Random.seed!(seed)
    θ_samples = n_epi_total > 0 ?
        QuasiMonteCarlo.sample(n_samples, relaxed_lbs, relaxed_ubs, LatinHypercubeSample())' :
        # QuasiMonteCarlo.sample(n_samples, relaxed_lbs, relaxed_ubs, SobolSample())' :
        Matrix{Float64}(undef, n_samples, 0)

    v_samples = similar(θ_samples)
    for j in 1:n_epi_total
        v_samples[:, j] = θ_to_v.(θ_samples[:, j], relaxed_lbs[j], relaxed_ubs[j])
    end

    x_names, w_names, u_names, v_names = spec_names(specs)
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
        UncertaintyQuantification.evaluate!(physical_model, phys_df)
        aug_df[!, y_symbol] = phys_df[!, y_symbol]

        # y = [physical_model(x_samples[row, :]...) for row in 1:n_samples]
        # aug_df[!, y_symbol]  = y
        # phys_df[!, y_symbol] = y
    end

    return aug_df, phys_df
end

# ==============================================================================
# 9.  augmented_to_physical / augmented_to_epistemic — generalized
# ==============================================================================
function augmented_to_physical(w, specs::Vector{<:AbstractInputSpec})
    x_names, w_names, u_names, v_names = spec_names(specs)
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
 
    return Metaheuristics.boxconstraints(lb=lb_u, ub=ub_u), Metaheuristics.boxconstraints(lb=lb_v, ub=ub_v)
end