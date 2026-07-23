function build_kl_sampler(gp, W::Matrix{Float64}, X_train::Matrix{Float64};
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
    nthreads = Threads.nthreads()
    buffers = [
        (
            K_qW = Matrix{Float64}(undef, Nx, N0),
            K_qtr = Matrix{Float64}(undef, Nx, n_train),
            K_qW_post = Matrix{Float64}(undef, Nx, N0),
            kW_mean = Vector{Float64}(undef, N0)
        )
        for _ in 1:nthreads
    ]
    realization_bufs = [Vector{Float64}(undef, Nx) for _ in 1:Threads.maxthreadid()]

    return function gp_samples!(qoi_buffer, input; qoi_type::Symbol=:mean, y_star=nothing)
        fill!(qoi_buffer, 0.0)

        @assert size(input, 1) == Nx "gp_samples!: input size $(size(input,1)) != Nx=$Nx"

        tid = Threads.threadid()
        buf = buffers[tid]

        K_qW_buf = buf.K_qW
        K_qtr_buf = buf.K_qtr
        K_qW_post_buf = buf.K_qW_post
        kW_post_mean_buf = buf.kW_mean

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
            Threads.@threads :static for s in 1:Ng
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

export 
    build_kl_sampler