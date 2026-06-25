mse(y_true, y_pred)  = mean((y_true .- y_pred).^2)
rmse(y_true, y_pred) = sqrt(mse(y_true, y_pred))

function nrmse(y_true, y_pred; method=:std)
    denom = if method == :minmax
        maximum(y_true) - minimum(y_true)

    elseif method == :mean
        mean(y_true)

    elseif method == :std
        std(y_true)

    elseif method == :var
        var(y_true)
        
    else
        throw(ArgumentError("Unknown nrmse method: $method. Choose :minmax, :mean, :std, or :var."))
    end

    iszero(denom) && throw(ArgumentError("Normalisation denominator is zero, cannot compute nRMSE."))

    return rmse(y_true, y_pred) / denom
end

function q2(y_true, y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1 - ss_res / ss_tot
end


"""
    q2_loo(build_model, data, y_symbol, x_names) -> Float64

Surrogate-agnostic leave-one-out Q². Refits on each (n-1)-point fold and
predicts the held-out point. Works for any surrogate exposing fit!/predict
(GP, PCE, PCK), so it is fair for cross-model comparison.

Same SS_res / SS_tot structure as `q2`, but residuals come from held-out
folds rather than an independent test set.
"""
function q2_loo(build_model, data::DataFrame, y_symbol::Symbol)
    n = nrow(data)
    y = data[:, y_symbol]
    y_loo = similar(y, Float64)

    for i in 1:n
        train_fold = data[setdiff(1:n, i), :]
        test_row   = data[i:i, :]

        model = build_model(train_fold)
        fit!(model)
        X_test = Matrix(test_row[:, model.x_names])
        μ      = predict(model, X_test, mode=:mean)
            
        y_loo[i] = μ[1]
    end

    return q2(y, y_loo)
end



"""
    q2_loo_gp_fast(gp) -> Float64

Analytical LOO Q² for a FITTED GP (Rasmussen & Williams eq. 5.12).
Hyperparameters held fixed at the full-data fit — fast, exact for that
assumption, but GP-only and slightly optimistic vs. true refit-LOO.
"""
function q2_loo_gp_fast(gp::GaussianProcess)
    gp.posterior === nothing && error("Fit the GP first.")
    kern  = gp.kernel_posterior
    K     = kernelmatrix(kern, RowVecs(gp.X)) + 1e-8I   # match your posterior jitter
    Kinv  = inv(cholesky(Symmetric(K)))
    α     = Kinv * gp.y

    μ_loo_resid = α ./ diag(Kinv)                       # = μ_LOO,i − y_i
    ss_res = sum(abs2, μ_loo_resid)
    ss_tot = sum(abs2, gp.y .- mean(gp.y))
    return 1 - ss_res / ss_tot
end