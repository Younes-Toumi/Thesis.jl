using SurrogateModelling
using UncertaintyQuantification
using UncertaintyQuantification: sample, wrap, isimprecise, middle, minimize, RobustOrthoMADS, bounds, map_to_precise_inputs
using Random
using DataFrames
using ParameterHandling
using LinearAlgebra


# =======================================================================
# Step 1. Augmented Space Setup: Four Gaussian Mixture Function g(x1, x2)
# =======================================================================

x1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1.5, 1.5), :σ => 0.1)), :x2)

specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input
physical_model = model_gfunction
print("Gaussian Mixture...\n")

x_names, w_names, u_names, v_names = spec_names(specs)
y_symbol = physical_model.name


# # =======================================================================
# # Step 2. Training the GP, PCE, PCK
# # =======================================================================

# n_samples = 10
# data_aug_train, _ = build_augmented_design(physical_model, specs, n_samples)

# n_pool = 1_000_000
# data_aug_pool, _ = build_augmented_design(physical_model, specs, n_pool)



# pce_p_max = 7
# pck_p_max = 3

# kernel_type = GPMatern52
# bases = fill(SurrogateModelling.HermiteBasis(), length(w_names))
# pce_solver = SurrogateModelling.LASSOSolver

# pce_degree = QBall(pce_p_max, 0.5)
# pck_degree = QBall(pck_p_max, 0.5)

# gp          = SurrogateModelling.GaussianProcess(data_aug_train, y_symbol;          kernel_type=kernel_type())
# pce         = SurrogateModelling.PolynomialChaosExpansion(data_aug_train, y_symbol, bases, pce_degree; solver=pce_solver())
# pce_trend   = SurrogateModelling.PolynomialChaosExpansion(data_aug_train, y_symbol, bases, pck_degree; solver=pce_solver())
# pck         = SurrogateModelling.PolynomialChaosKriging(data_aug_train, y_symbol,   pce_trend, kernel_type=kernel_type())

# gp_fit_value,  gp_fit_time,  _... = @timed fit!(gp)
# pce_fit_value, pce_fit_time, _... = @timed fit!(pce)
# pck_fit_value, pck_fit_time, _... = @timed fit!(pck)

# print("\n")

# q2_gp   = q2_loo(df -> SurrogateModelling.GaussianProcess(df, y_symbol; kernel_type=kernel_type()),                            data_aug_train, y_symbol)
# print("q2 gp done...\n")


# q2_pce  = q2_loo(df -> SurrogateModelling.PolynomialChaosExpansion(df, y_symbol, bases, pce_degree; solver=pce_solver()),            data_aug_train, y_symbol)
# print("q2 pce done...\n")


# q2_pck  = q2_loo(df -> SurrogateModelling.PolynomialChaosKriging(df, y_symbol, pce_trend, kernel_type=kernel_type()),     data_aug_train, y_symbol)

# print("\n")

# print("n₀: $n_samples:\n")
# print("Q² GP:  $(round(q2_gp, digits=3))\n")
# print("Q² PCE: $(round(q2_pce, digits=3))\n")
# print("Q² PCK: $(round(q2_pck, digits=3))\n")


# gp_μ_pool,  gp_time_pool,  _...  = @timed predict(gp, Matrix(data_aug_pool[:, w_names]))
# pce_μ_pool, pce_time_pool, _...  = @timed predict(pce, Matrix(data_aug_pool[:, w_names]))
# pck_μ_pool, pck_time_pool, _...  = @timed predict(pck, Matrix(data_aug_pool[:, w_names]))


function compare_surrogates_evolution(
    physical_model, 
    specs,
    n_trains
)
    pck_p_max = 3

    kernel_type = GPMatern52
    bases = fill(SurrogateModelling.HermiteBasis(), length(w_names))
    pce_solver = SurrogateModelling.LASSOSolver
    pck_degree = QBall(pck_p_max, 0.5)

    gps     = Array{SurrogateModelling.GaussianProcess}(undef, length(n_trains))
    pcks    = Array{SurrogateModelling.PolynomialChaosKriging}(undef, length(n_trains))

    for (idx, n_train) in enumerate(n_trains)
        print("currently at n_train = $n_train ...\n")
        data_aug_train, _ = build_augmented_design(physical_model, specs, n_train)

        gp          = SurrogateModelling.GaussianProcess(data_aug_train, y_symbol;          kernel_type=kernel_type())
        pce_trend   = SurrogateModelling.PolynomialChaosExpansion(data_aug_train, y_symbol, bases, pck_degree; solver=pce_solver())
        pck         = SurrogateModelling.PolynomialChaosKriging(data_aug_train, y_symbol,   pce_trend, kernel_type=kernel_type())

        fit!(gp)
        fit!(pck)

        gps[idx] = gp
        pcks[idx] = pck

    end

    return gps, pcks
end

# n_trains = [50, 100, 150, 200, 250, 300, 400, 500]
n_trains = [300, 350]

gps, pcks = compare_surrogates_evolution(
    physical_model, 
    specs,
    n_trains,
)

# =======================================================================
# Step 3. Treat the GP as a the "LF" Physical Model 
# =======================================================================

function true_doubleloop(
    model, imprecise_inputs;
    n_total::Int, k, qoi::Symbol, y_star::Float64,
    surrogate::Bool = false,        # ← flag: false = physical model, true = augmented-space surrogate
    specs = nothing,                # ← required when surrogate=true
)
    inputs = wrap(imprecise_inputs)
    imp  = filter(isimprecise, inputs)
    prec = filter(!isimprecise, inputs)

    lb, ub = float.(bounds(inputs))
    d = length(lb)

    n_θ = Int(ceil(sqrt(n_total / k)))
    n_u = Int(ceil(sqrt(n_total * k)))

    qoi_vals = Vector{Float64}(undef, n_θ)
    θ_array  = Matrix{Float64}(undef, n_θ, d)

    if surrogate
        specs === nothing && error("Pass `specs` when surrogate=true.")
        x_names, w_names, u_names, v_names = spec_names(specs)
    end

    for i in 1:n_θ
        θ = lb .+ rand(d) .* (ub .- lb)

        if !surrogate
            # ── PHYSICAL: sample x ~ N(θ, σ²) in physical space, evaluate true model ──
            θ_inputs = map_to_precise_inputs(θ, imp)
            df = sample([prec..., θ_inputs...], n_u)
            evaluate!(model, df)
            y = df[:, model.name]
        else
            # ── SURROGATE: build augmented (u,v) df, evaluate GP in SNS ──
            df = build_inner_augmented(specs, θ, n_u, u_names, v_names)
            SurrogateModelling.evaluate!(model, df)
            y = df[:, model.y_symbol]
        end

        qoi_vals[i] =
            qoi == :mean ? mean(y) :
            qoi == :pf   ? mean(y .<= y_star) :
            error("Unknown QoI.")
        θ_array[i, :] .= θ
    end

    imin = argmin(qoi_vals)
    imax = argmax(qoi_vals)
    return (
        bounds   = [minimum(qoi_vals), maximum(qoi_vals)],
        θ_bounds = [θ_array[imin, :], θ_array[imax, :]],
        n_calls  = n_θ * n_u,
    )
end



"""
    build_inner_augmented(specs, θ, n_u, u_names, v_names) -> DataFrame

Inner-loop augmented design for a FIXED epistemic θ (in raw interval space).
v-columns: θ mapped to SNS via the SAME relaxed-bounds + θ_to_v chain that
build_augmented_design used at training time. u-columns: n_u standard-normal
aleatory draws. Column layout matches w_names = [u_names...; v_names...].
"""
function build_inner_augmented(specs, θ, n_u, u_names, v_names)
    df = DataFrame()

    # ── aleatory u-columns: n_u standard-normal draws (SNS convention) ──
    for un in u_names
        df[!, un] = randn(n_u)
    end

    # ── epistemic v-columns: map each θ component through the SAME chain
    #    used in build_augmented_design (compute_relaxed_bounds → θ_to_v) ──
    v_vals = Float64[]
    θ_idx = 0
    for s in specs
        if s isa IntervalSpec
            θ_idx += 1
            lb, ub = compute_relaxed_bounds(s.θ_L, s.θ_U; v_L=s.v_L, v_U=s.v_U)
            push!(v_vals, θ_to_v(θ[θ_idx], lb, ub))

        elseif s isa HybridSpec
            for j in 1:length(s.param_names)
                θ_idx += 1
                lb, ub = compute_relaxed_bounds(s.θ_L[j], s.θ_U[j]; v_L=s.v_L[j], v_U=s.v_U[j])
                push!(v_vals, θ_to_v(θ[θ_idx], lb, ub))
            end
        end
        # PreciseSpec contributes no v-column
    end

    for (j, vn) in enumerate(v_names)
        df[!, vn] = fill(v_vals[j], n_u)
    end

    return df
end



# surrogate propagation — same call, two extra kwargs

for (idx, n_train) in enumerate(n_trains)
    result_mean_gp = @time "gp mean: " true_doubleloop(
        gps[idx], [x1, x2];
        n_total=2*10^8, k=5.0, qoi=:mean, y_star=-1.427,
        surrogate=true, specs=specs
    )

    println("gp  (n₀ = $n_train): bounds = $(round.(result_mean_gp.bounds, digits=3))")


    result_mean_pck = @time "pck mean: " true_doubleloop(
        pcks[idx], [x1, x2];
        n_total=2*10^8, k=5.0, qoi=:mean, y_star=-1.427,
        surrogate=true, specs=specs
    )

    println("pck (n₀ = $n_train): bounds = $(round.(result_mean_pck.bounds, digits=3))\n")



    result_pf_gp = @time "gp pf: " true_doubleloop(
        gps[idx], [x1, x2];
        n_total=5*10^8, k=0.01, qoi=:pf, y_star=-1.427,
        surrogate=true, specs=specs
    )

    println("gp  (n₀ = $n_train): pf = $(round.(result_pf_gp.bounds, digits=3))")


    result_pf_pck = @time "pck pf: " true_doubleloop(
        pcks[idx], [x1, x2];
        n_total=5*10^8, k=0.01, qoi=:pf, y_star=-1.427,
        surrogate=true, specs=specs
    )

    println("pck (n₀ = $n_train): pf = $(round.(result_pf_pck.bounds, digits=3))\n")


end


# Gaussian Mixture...
# currently at n_train = 50 ...
# currently at n_train = 100 ...
# currently at n_train = 150 ...
# currently at n_train = 200 ...
# currently at n_train = 250 ...
# currently at n_train = 300 ...
# currently at n_train = 400 ...
# currently at n_train = 500 ...
# gp mean: : 139.176813 seconds (5.81 M allocations: 267.042 GiB, 48.14% gc time)
# gp  (n₀ = 50): bounds = [-1.267, 1.519]
# pce mean: : 267.067228 seconds (800.51 M allocations: 171.394 GiB, 24.07% gc time)
# pce (n₀ = 50): bounds = [-0.322, 0.327]
# pck mean: : 220.957890 seconds (800.91 M allocations: 330.887 GiB, 41.57% gc time)
# pck (n₀ = 50): bounds = [-1.267, 1.515]

# gp pf: : 222.876968 seconds (31.34 M allocations: 669.267 GiB, 30.51% gc time)
# gp  (n₀ = 50): pf = [0.0, 0.0]
# pce pf: : 509.750078 seconds (2.02 G allocations: 429.197 GiB, 15.41% gc time)
# pce (n₀ = 50): pf = [0.0, 0.003]
# pck pf: : 413.106095 seconds (2.03 G allocations: 829.219 GiB, 29.92% gc time)
# pck (n₀ = 50): pf = [0.0, 0.0]

# gp mean: : 314.375854 seconds (860.42 k allocations: 490.367 GiB, 48.73% gc time)
# gp  (n₀ = 100): bounds = [-1.362, 1.275]
# pce mean: : 296.758088 seconds (800.45 M allocations: 171.391 GiB, 22.72% gc time)
# pce (n₀ = 100): bounds = [-0.352, 0.321]
# pck mean: : 400.410917 seconds (800.76 M allocations: 554.439 GiB, 45.89% gc time)
# pck (n₀ = 100): bounds = [-1.31, 1.249]

# gp pf: : 467.439821 seconds (31.53 M allocations: 1.200 TiB, 38.08% gc time)
# gp  (n₀ = 100): pf = [0.0, 0.382]
# pce pf: : 508.464959 seconds (2.02 G allocations: 429.197 GiB, 15.17% gc time)
# pce (n₀ = 100): pf = [0.0, 0.001]
# pck pf: : 680.939682 seconds (2.03 G allocations: 1.357 TiB, 37.07% gc time)
# pck (n₀ = 100): pf = [0.0, 0.053]

# gp mean: : 471.249439 seconds (860.42 k allocations: 713.927 GiB, 54.46% gc time)
# gp  (n₀ = 150): bounds = [-1.329, 1.258]
# pce mean: : 291.281369 seconds (800.45 M allocations: 171.391 GiB, 24.43% gc time)
# pce (n₀ = 150): bounds = [-1.61, 1.061]
# pck mean: : 655.752563 seconds (800.76 M allocations: 777.999 GiB, 49.95% gc time)
# pck (n₀ = 150): bounds = [-1.286, 1.219]

# gp pf: : 941.554759 seconds (31.53 M allocations: 1.747 TiB, 35.96% gc time)
# gp  (n₀ = 150): pf = [0.0, 0.059]
# pce pf: : 612.985574 seconds (2.02 G allocations: 429.197 GiB, 14.19% gc time)
# pce (n₀ = 150): pf = [0.0, 1.0]
# pck pf: : 1271.989829 seconds (2.03 G allocations: 1.903 TiB, 37.33% gc time)
# pck (n₀ = 150): pf = [0.0, 0.021]

# gp mean: : 766.590306 seconds (860.42 k allocations: 937.486 GiB, 54.77% gc time)
# gp  (n₀ = 200): bounds = [-1.372, 1.304]
# pce mean: : 307.428382 seconds (800.45 M allocations: 171.391 GiB, 24.81% gc time)
# pce (n₀ = 200): bounds = [-0.409, 0.312]
# pck mean: : 854.881246 seconds (800.76 M allocations: 1001.557 GiB, 50.83% gc time)
# pck (n₀ = 200): bounds = [-1.371, 1.304]

# gp pf: : 1411.338667 seconds (31.53 M allocations: 2.294 TiB, 39.27% gc time)
# gp  (n₀ = 200): pf = [0.0, 0.395]
# pce pf: : 617.234721 seconds (2.02 G allocations: 429.197 GiB, 14.50% gc time)
# pce (n₀ = 200): pf = [0.0, 0.0]
# pck pf: : 1655.695031 seconds (2.03 G allocations: 2.450 TiB, 38.02% gc time)
# pck (n₀ = 200): pf = [0.0, 0.39]

# gp mean: : 896.116493 seconds (860.42 k allocations: 1.134 TiB, 51.52% gc time)
# gp  (n₀ = 250): bounds = [-1.33, 1.358]
# pce mean: : 313.420221 seconds (800.45 M allocations: 171.391 GiB, 24.62% gc time)
# pce (n₀ = 250): bounds = [-0.708, 0.686]
# pck mean: : 1104.062678 seconds (800.76 M allocations: 1.196 TiB, 53.04% gc time)
# pck (n₀ = 250): bounds = [-1.321, 1.359]

# gp pf: : 1733.444673 seconds (31.53 M allocations: 2.841 TiB, 38.05% gc time)
# gp  (n₀ = 250): pf = [0.0, 0.03]
# pce pf: : 604.110104 seconds (2.02 G allocations: 429.197 GiB, 13.69% gc time)
# pce (n₀ = 250): pf = [0.0, 0.003]
# pck pf: : 1928.540848 seconds (2.03 G allocations: 2.997 TiB, 35.02% gc time)
# pck (n₀ = 250): pf = [0.0, 0.048]

