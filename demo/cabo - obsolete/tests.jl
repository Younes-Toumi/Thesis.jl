using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
using Printf
using KernelFunctions
using LinearAlgebra
using ParameterHandling
using Plots


function build_kl_sampler_new(gp, W::Matrix{Float64}, X_train::Matrix{Float64};
                           N_samples, Nx, energy_threshold=0.99, δ=1e-8)
    kern    = gp.kernel_posterior
    N0      = size(W, 1)
    n_train = size(X_train, 1)
    Ng      = N_samples

    K_W   = kernelmatrix(kern, RowVecs(W),       RowVecs(W))
    K_tr  = kernelmatrix(kern, RowVecs(X_train), RowVecs(X_train))
    K_ctW = kernelmatrix(kern, RowVecs(X_train), RowVecs(W))
    L = cholesky(Symmetric(K_tr + δ*I)).L
    A = L' \ (L \ K_ctW)

    K_W_post = Symmetric(K_W .- K_ctW' * A)
    eig    = eigen(K_W_post)
    λ_post = max.(eig.values, 0.0)
    V_post = eig.vectors
    idx    = sortperm(λ_post, rev=true)
    λ_post, V_post = λ_post[idx], V_post[:, idx]

    total_energy      = sum(λ_post)
    cumulative_energy = cumsum(λ_post) ./ total_energy
    r_energy = searchsortedfirst(cumulative_energy, energy_threshold)
    floor_   = max(1e-10 * total_energy, 1e-14)
    r_floor  = something(findlast(λ_post .> floor_), r_energy)
    r        = min(r_energy, r_floor)
    V_r, λ_r = V_post[:, 1:r], λ_post[1:r]

    Ξ         = randn(r, Ng)
    coeff_mat = V_r * (Ξ ./ reshape(sqrt.(λ_r), :, 1))   # N0 × Ng, built once

    # ── preallocated ONCE, reused every call ──────────────────────────────
    K_qW_buf         = Matrix{Float64}(undef, Nx, N0)
    K_qtr_buf        = Matrix{Float64}(undef, Nx, n_train)
    K_qW_post_buf    = Matrix{Float64}(undef, Nx, N0)
    kW_post_mean_buf = Vector{Float64}(undef, N0)

    realization_bufs = [Vector{Float64}(undef, Nx) for _ in 1:Threads.maxthreadid()]

    return function gp_samples_new!(qoi_buffer, input; qoi_type::Symbol=:mean, y_star=nothing)
        @assert size(input, 1) == Nx "gp_samples!: input size $(size(input,1)) != Nx=$Nx"

        kernelmatrix!(K_qW_buf,  kern, RowVecs(input), RowVecs(W))
        kernelmatrix!(K_qtr_buf, kern, RowVecs(input), RowVecs(X_train))
        μ_q = predict(gp, input; mode=:mean)          # small Nx-vector — see note below

        K_qW_post_buf .= K_qW_buf
        mul!(K_qW_post_buf, K_qtr_buf, A, -1.0, 1.0)   # fused: K_qW_post = K_qW - K_qtr*A, in place

        if qoi_type === :mean
            sum!(reshape(kW_post_mean_buf, 1, :), K_qW_post_buf)
            kW_post_mean_buf ./= Nx
            mul!(qoi_buffer, coeff_mat', kW_post_mean_buf)   # writes directly into qoi_buffer
            qoi_buffer .+= mean(μ_q)

        else  # :var or :pf — one realization at a time, one small reused buffer
            for s in 1:Ng
                buf = realization_bufs[Threads.threadid()]
                mul!(buf, K_qW_post_buf, view(coeff_mat, :, s))
                buf .+= μ_q
                qoi_buffer[s] = qoi_type === :var ? var(buf; corrected=true) :
                                                    count(<(y_star), buf) / Nx
            end
        end
        return qoi_buffer
    end
end


# ==============================================================================
# 10. Usage — mirrors your UQ.jl example end to end
# ==============================================================================

# parametric inputs
x1 = RandomVariable(Nornal(0.0, 1.0), :x1)
x2 = IntervalVariable(-1.0, 1.0, :x2)

specs = InputSpec.([x1, x2])     # broadcasts dispatch over each UQ.jl input
 
x_names, w_names, u_names, v_names = spec_names(specs)
y_symbol = model_gfunction.name
physical_model = g_function


n_train, n_test = 20, 1001


data_aug_train, data_phys_train =    build_augmented_design(model_gfunction, specs, n_train)
data_aug_test,  data_phys_test  =    build_augmented_design(model_gfunction, specs, n_test)


# # initialize GP on θ-space
kernel() = GPMatern52()
gp = GaussianProcess(data_aug_train, :y, kernel_type=kernel())
@time "fit!" fit!(gp)


println("lengthscales: ", ParameterHandling.value(gp.θ.lengthscale))
println("variance: ",     ParameterHandling.value(gp.θ.variance))
println("noise",     ParameterHandling.value(gp.θ.noise))



μ_test, σ_test = @time "predict:" predict(gp, Matrix(data_aug_test[:, w_names]))

println("MSE: $(round(mse(data_aug_test.y, μ_test), digits=5))")
println("Q²:  $(round(q2(data_aug_test.y, μ_test), digits=5))")


N0 = 500
Ng = 400
Nx = 400

W_train = Matrix(data_aug_train[:, w_names])
W, _         = build_augmented_design(nothing, specs, N0)
W = Matrix(W)

# Build BOTH closures on the exact same gp/W/W_train
Random.seed!(1)
gp_samples_old = build_kl_sampler_old(gp, W, W_train; N_samples=Ng)          # old version

Random.seed!(1)   # SAME seed → identical Ξ draw inside both closures
gp_samples_new! = build_kl_sampler_new(gp, W, W_train; N_samples=Ng, Nx=Nx)  # new version

# identical input batch
u_final        = randn(Nx, 2)
v_final_min =  θ_to_v.([-0.567, 0.527], [-1.5429, -1.5429], [1.5429, 1.5429])
v_final_max =  θ_to_v.([0.557, 0.808], [-1.5429, -1.5429], [1.5429, 1.5429])


input_min = hcat(u_final, repeat(v_final_min', Nx, 1))

# OLD: full matrix, reduce by hand
old_full = gp_samples_old(input_min)                    # Ng × Nx
old_mean = vec(mean(old_full, dims=2))               # Ng-vector

# NEW: buffer-based
qoi_buffer_min = Vector{Float64}(undef, Ng)
gp_samples_new!(qoi_buffer_min, input_min; qoi_type=:mean)

diff = old_mean .- qoi_buffer_min
println("Nx = $Nx, Ng = $Ng, N0 = $N0")
println("max abs diff: ", maximum(abs.(diff)))
println("old mean of means: ", mean(old_mean))
println("new mean of means: ", mean(qoi_buffer_min))



p1_min = scatter(input_max[:, 1], qoi_buffer_min, title="min - u1 variablility")
p2_min = scatter(input_max[:, 2], qoi_buffer_min, title="min - u2 variablility")
p3_min = scatter(input_max[:, 3], qoi_buffer_min, title="min - v1 variablility")
p4_min = scatter(input_max[:, 4], qoi_buffer_min, title="min - v2 variablility")

p_min = plot(
    p1_min, p2_min,
    p3_min, p4_min,
    layout = (2, 2),
    size = (1300, 900),
    margin=5Plots.mm,
    guidefontsize=14,
    tickfontsize=14,
    titlefontsize=14
);

display(p_min)



input_max = hcat(u_final, repeat(v_final_max', Nx, 1))

# OLD: full matrix, reduce by hand
old_full = gp_samples_old(input_max)                    # Ng × Nx
old_mean = vec(mean(old_full, dims=2))               # Ng-vector

# NEW: buffer-based
qoi_buffer_max = Vector{Float64}(undef, Ng)
gp_samples_new!(qoi_buffer_max, input_max; qoi_type=:mean)

diff = old_mean .- qoi_buffer_max
println("Nx = $Nx, Ng = $Ng, N0 = $N0")
println("max abs diff: ", maximum(abs.(diff)))
println("old mean of means: ", mean(old_mean))
println("new mean of means: ", mean(qoi_buffer_max))

p1_max = scatter(input_max[:, 1], qoi_buffer_max, title="max - u1 variablility")
p2_max = scatter(input_max[:, 2], qoi_buffer_max, title="max - u2 variablility")
p3_max = scatter(input_max[:, 3], qoi_buffer_max, title="max - v1 variablility")
p4_max = scatter(input_max[:, 4], qoi_buffer_max, title="max - v2 variablility")

p_max = plot(
    p1_max, p2_max,
    p3_max, p4_max,
    layout = (2, 2),
    size = (1300, 900),
    margin=5Plots.mm,
    guidefontsize=14,
    tickfontsize=14,
    titlefontsize=14
);

display(p_max)