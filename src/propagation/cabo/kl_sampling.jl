function build_kl_sampler(gp, W::Matrix{Float64}, X_train::Matrix{Float64};
                           N_samples, energy_threshold=0.99, δ=1e-8)

    kern    = gp.kernel_posterior
    N0      = size(W, 1)

    # Prior kernel matrices
    # ── in build_kl_sampler  ────────────────────
    K_W   = kernelmatrix(kern, RowVecs(W),       RowVecs(W))
    K_tr  = kernelmatrix(kern, RowVecs(X_train), RowVecs(X_train))
    K_ctW = kernelmatrix(kern, RowVecs(X_train), RowVecs(W))

    L = cholesky(Symmetric(K_tr + δ*I)).L
    A = L' \ (L \ K_ctW)                # K_train⁻¹ K_ctW  (n_train × N0) — reused in closure

    # ── FIX 1: eigendecompose the POSTERIOR covariance, not the prior ─────────
    # K_W_post = K_W − K_ctW^T K_train⁻¹ K_ctW
    K_W_post = Symmetric(K_W .- K_ctW' * A)

    eig    = eigen(K_W_post)
    λ_post = max.(eig.values, 0.0)
    V_post = eig.vectors
    idx    = sortperm(λ_post, rev=true)
    λ_post, V_post = λ_post[idx], V_post[:, idx]

   # ── Energy-based truncation — this is what energy_threshold was always for ──
    total_energy      = sum(λ_post)
    cumulative_energy = cumsum(λ_post) ./ total_energy
    r_energy = searchsortedfirst(cumulative_energy, energy_threshold)

    # Numerical floor as a secondary guard
    floor   = max(1e-10 * total_energy, 1e-14)
    r_floor = something(findlast(λ_post .> floor), r_energy)

    r    = min(r_energy, r_floor)
    V_r  = V_post[:, 1:r]
    λ_r  = λ_post[1:r]

    # println("  EOLE (posterior): N0=$N0, r=$r active modes " *
    #         "($(round(100*cumulative_energy[r], digits=1))% energy captured)")

    # ── Form 2 coefficient — gives Var[h(w)] ≈ k_post(w,w) ────────────
    Ξ         = randn(r, N_samples)
    coeff_mat = V_r * (Ξ ./ reshape(sqrt.(λ_r), :, 1))  # N0 × N_samples

    # ── Closure: dispatch on AbstractVector (single) vs AbstractMatrix (batch) ─
    return function gp_samples(input)
        if input isa AbstractVector
            # ── single point ──────────────────────────────────────────────────
            kW   = vec(kernelmatrix(kern, RowVecs(reshape(input,1,:)), RowVecs(W)))
            ktr  = vec(kernelmatrix(kern, RowVecs(reshape(input,1,:)), RowVecs(X_train)))
            μ_w  = predict(gp, reshape(input, 1, :); mode=:mean)
            kW_post = kW .- A' * ktr           # N0-vector  (= 0 at training pts)
            return μ_w .+ coeff_mat' * kW_post  # N_samples-vector

        else
            # ── batch: input is Nx × d ────────────────────────────────────────
            K_qW   = kernelmatrix(kern, RowVecs(input), RowVecs(W))        # Nx_q × N0
            K_qtr  = kernelmatrix(kern, RowVecs(input), RowVecs(X_train))  # Nx_q × n_train
            μ_q    = predict(gp, input; mode=:mean)        # Nx_q-vector (one predict call)
            K_qW_post = K_qW .- K_qtr * A         # Nx_q × N0

            return μ_q' .+ coeff_mat' * K_qW_post'  # N_samples × Nx_q
        end
    end
end