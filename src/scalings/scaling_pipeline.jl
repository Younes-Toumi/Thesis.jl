struct ScalingPipeline
    input_scaler
    output_scaler
end

# ============================================================
# 1. matrix aware
# ============================================================

function fit_pipeline(X::Matrix, y::Vector, input_type, output_type)
    input_scaler  = fit_scaler(input_type, X)
    output_scaler = fit_scaler(output_type, reshape(y, :, 1))
    return ScalingPipeline(input_scaler, output_scaler)
end

function transform_input(p::ScalingPipeline, X::Matrix)
    return transform(p.input_scaler, X)
end

function transform_output(p::ScalingPipeline, y::Vector)
    return vec(transform(p.output_scaler, reshape(y, :, 1)))
end

# ============================================================
# Dataframe aware
# ============================================================

function fit_pipeline(data::DataFrame, output::Symbol, input_type, output_type)
    x_names = propertynames(data[:, Not(output)])
    X = Matrix(data[:, x_names])
    y = Vector(data[:, output])
    return fit_pipeline(X, y, input_type, output_type)
end

function transform(p::ScalingPipeline, data::DataFrame, output::Symbol)
    x_names  = propertynames(data[:, Not(output)])
    X_scaled = transform_input(p, Matrix(data[:, x_names]))
    y_scaled = transform_output(p, Vector(data[:, output]))

    df = DataFrame(X_scaled, collect(x_names))
    df[!, output] = y_scaled
    return df
end

function inverse_transform(p::ScalingPipeline, data::DataFrame, output::Symbol)
    x_names  = propertynames(data[:, Not(output)])
    X        = inverse_transform(p.input_scaler, Matrix(data[:, x_names]))
    y        = inverse_mean(p, Vector(data[:, output]))

    df = DataFrame(X, collect(x_names))
    df[!, output] = y
    return df
end


# ============================================================
# transform mena and variance
# ============================================================
 
function inverse_mean(p::ScalingPipeline, μ::Vector)
    return vec(inverse_transform(p.output_scaler, reshape(μ, :, 1)))
end

function inverse_variance(p::ScalingPipeline, σ2::Vector)
    return variance_transform(p.output_scaler, σ2)
end

export 
    MinMaxScaler, ZScoreScaler, 
    fit_pipeline, transform_input, transform_output, transform,
    inverse_mean, inverse_variance