"""
    select_alc(gp, pool_aug, ref_grid, w_names) -> idx

ALC: pick the pool point whose addition most reduces TOTAL predictive
variance over a fixed reference grid. Uses the GP rank-1 variance-reduction
formula — no model evaluation needed to score candidates.

For each candidate x*, adding it reduces variance at every grid point x by
    Δσ²(x) = cov(x, x*)² / (σ²(x*) + noise)
so the score is Σ_x Δσ²(x), and we pick the candidate maximizing it.
"""
function select_alc(gp, pool_aug, ref_grid, w_names; jitter=1e-6)
    kern  = gp.kernel_posterior
    Xpool = Matrix(pool_aug[:, w_names])
    Xref  = Matrix(ref_grid[:, w_names])
    Xtr   = gp.X

    # posterior variance at candidates (this is correct as-is)
    _, σ_pool = predict(gp, Xpool; mode=:mean_and_var)

    # POSTERIOR cross-covariance between candidates and ref grid:
    #   k_post(x*, x) = k(x*,x) − k(x*,Xtr) Ktr⁻¹ k(Xtr,x)
    Ktr   = kernelmatrix(kern, RowVecs(Xtr)) + jitter*I
    L     = cholesky(Symmetric(Ktr)).L
    K_pool_tr = kernelmatrix(kern, RowVecs(Xpool), RowVecs(Xtr))   # n_pool × n_tr
    K_tr_ref  = kernelmatrix(kern, RowVecs(Xtr),  RowVecs(Xref))   # n_tr × n_ref
    K_pool_ref_prior = kernelmatrix(kern, RowVecs(Xpool), RowVecs(Xref))

    A = L' \ (L \ K_tr_ref)                          # Ktr⁻¹ k(Xtr, Xref)
    K_pool_ref_post = K_pool_ref_prior .- K_pool_tr * A   # n_pool × n_ref, ≈0 at duplicates

    denom  = σ_pool .^ 2 .+ jitter
    scores = vec(sum(K_pool_ref_post .^ 2, dims=2)) ./ denom
    return argmax(scores)
end


"""
    adaptive_sampling_maxvar(physical_model, specs, kernel_type; kwargs...)

Greedy max-variance (ALM) active learning. Starts from a small design and
adds one point at a time at the location of highest GP posterior variance,
until either `n_budget` points have been added OR test Q² reaches `q2_threshold`.

Returns (gp, data, n_history, q2_history) so you can plot Q² vs. n_samples.

# Keyword arguments
- `n_init`        : initial design size (default 2·n_dims)
- `n_budget`      : max number of points to ADD (default 50)
- `q2_threshold`  : stop early once test Q² reaches this (default 0.99)
- `n_pool`        : candidate pool size evaluated each iteration (default 2000)
- `data_test`     : held-out test set for tracking Q² (required)
- `seed`          : base RNG seed for reproducibility (default 42)
"""
function adaptive_sampling(
    physical_model,
    specs,
    gp_init,
    data_aug_train;
    n_budget     :: Int     = 50,
    q2_threshold :: Float64 = 0.99,
    n_pool       :: Int     = 2000,
    seed         :: Int     = 42,
)
    y_symbol = physical_model.name

    # ── initial design + first fit (full fit!, not warm-started) ─────────────
    data = copy(data_aug_train)
    gp   = gp_init

    x_names, w_names, u_names, v_names  = spec_names(specs)
    spec_cols = [s.name for s in specs]        # physical-input columns, spec order

    # ── history tracking ─────────────────────────────────────────────────────
    n_history  = Int[size(gp.X, 1)]
    # q2_val = q2_loo(df -> GaussianProcess(df, y_symbol; kernel_type=gp.kernel_type), data_aug_train, y_symbol)
    q2_val = q2_loo_gp_fast(gp)
    
    q2_history = Float64[q2_val]
    @printf("init   n: %3d   Q²: %.4f\n", n_history[end], q2_history[end])


    ref_grid_aug, _ = build_augmented_design(nothing, specs, 1000)

    # ── adaptive loop ────────────────────────────────────────────────────────
    for iter in 1:n_budget
        Random.seed!(42+iter)
        q2_history[end] >= q2_threshold && (println("✓ Q² threshold reached"); break)

        # fresh candidate pool each iteration (seeded → reproducible)
        pool_aug, pool_phys = build_augmented_design(nothing, specs, n_pool)

        # max-variance selection: argmax of posterior σ over the pool
        _, σ_pool = predict(gp, Matrix(pool_aug[:, w_names]), mode=:mean_and_var)
        idx = argmax(σ_pool)
        # idx = select_alc(gp, pool_aug, ref_grid_aug, w_names)

        # evaluate the TRUE model at the single selected point
        x_new = [pool_phys[idx, c] for c in spec_cols]

        w_new = Matrix(pool_aug[idx:idx, w_names])      # 1 × d

        row_new_phys = DataFrame(Dict(x_names .=> x_new))
        UncertaintyQuantification.evaluate!(physical_model, row_new_phys)
        y_new = row_new_phys[1, y_symbol]

        # Generalized row constructions
        row_new_aug = merge(
            NamedTuple(zip(w_names, w_new)),
            (y = y_new,)
        )

        append!(data, DataFrame([row_new_aug]))

        # warm-started refit (refit! appends to gp.X / gp.y internally)
        gp = GaussianProcess(data, y_symbol; kernel_type=gp_init.kernel_type)
        fit!(gp)

        # track
        # q2_val = q2_loo(df -> GaussianProcess(df, y_symbol; kernel_type=gp_init.kernel_type), data, y_symbol)
        q2_val = q2_loo_gp_fast(gp)

        push!(n_history,  size(gp.X, 1))
        push!(q2_history, q2_val)
        @printf("iter %3d   n: %3d   added σ: %.2e   Q²: %.3f\n",
                iter, n_history[end], σ_pool[idx], q2_history[end])

    end

    # rebuild the final dataset from the GP's accumulated points
    data_final = data
    return (gp = gp, data = data_final, n_history = n_history, q2_history = q2_history)
end