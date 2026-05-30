using LinearAlgebra
using Random

# ------------------------------------------------------------
# Kernel vector k(w, W)
# ------------------------------------------------------------
function eval_k_vec(kernel, w, W)
    m = size(W, 1)
    k = zeros(Float64, m)
    @inbounds for j in 1:m
        k[j] = kernel(w, view(W, j, :))
    end
    return k
end

# ------------------------------------------------------------
# Kernel matrix K = k(W, W)
# ------------------------------------------------------------
function eval_K_mat(gp)
    W = gp.X
    m = size(W, 1)
    K = zeros(Float64, m, m)

    @inbounds for i in 1:m
        wi = view(W, i, :)
        for j in i:m
            kij = gp.kernel_prior(wi, view(W, j, :))
            K[i, j] = kij
            K[j, i] = kij
        end
    end

    return K
end

# ------------------------------------------------------------
# Prior mean vector b(W)
# ------------------------------------------------------------
function eval_b_vec(meanfun, W)
    [meanfun(view(W, i, :)) for i in 1:size(W, 1)]
end

# ------------------------------------------------------------
# You must define this for your problem.
# In the NIPI setting, u is usually standard normal.
# ------------------------------------------------------------
function sample_u_given_v(v; rng=Random.default_rng())
    error("Define sample_u_given_v(v; rng) for your model.")
end

# ------------------------------------------------------------
# Integrated kernel vector:
# k̄(v, W) = E_u[k((u,v), W)]
# ------------------------------------------------------------
function integrated_k_vec(gp, v; nmc=128, rng=Random.default_rng())
    W = gp.X
    acc = zeros(Float64, size(W, 1))

    for s in 1:nmc
        u = sample_u_given_v(v; rng=rng)
        w = vcat(u, v)
        acc .+= eval_k_vec(gp.kernel_prior, w, W)
    end

    return acc ./ nmc
end

# ------------------------------------------------------------
# Integrated scalar:
# k̄00(v) = E_{u,u'}[k((u,v),(u',v))]
# ------------------------------------------------------------
function integrated_k00(gp, v; nmc=128, rng=Random.default_rng())
    acc = 0.0

    for s in 1:nmc
        u1 = sample_u_given_v(v; rng=rng)
        u2 = sample_u_given_v(v; rng=rng)

        w1 = vcat(u1, v)
        w2 = vcat(u2, v)

        acc += gp.kernel_prior(w1, w2)
    end

    return acc / nmc
end

# ------------------------------------------------------------
# NIPI marginal prediction at epistemic point v
# Returns (μM, σM2)
# ------------------------------------------------------------
function nipi(gp, v; nmc=128, jitter=1e-8, rng=Random.default_rng())

    W = gp.X
    y = gp.y

    # training kernel matrix
    K = eval_K_mat(gp)
    F = cholesky(Symmetric(K + jitter * I))

    # prior mean on training points
    bW = eval_b_vec(gp.mean_prior, W)

    # solve K α = (y - bW)
    α = F \ (y .- bW)

    # integrated prior mean at v
    bM = 0.0
    for s in 1:nmc
        u = sample_u_given_v(v; rng=rng)
        w = vcat(u, v)
        bM += gp.mean_prior(w)
    end
    bM /= nmc

    # integrated kernel vector
    kbar = integrated_k_vec(gp, v; nmc=nmc, rng=rng)

    # marginal mean
    μM = bM + dot(kbar, α)

    # integrated self-covariance term
    k00 = integrated_k00(gp, v; nmc=nmc, rng=rng)

    # marginal variance
    q = F \ kbar
    σM2 = max(k00 - dot(kbar, q), 0.0)

    return μM, σM2
end