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