struct ScalingPipeline
    input_scaler
    output_scaler
end

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

function inverse_mean(p::ScalingPipeline, μ::Vector)
    return vec(inverse_transform(p.output_scaler, reshape(μ, :, 1)))
end

function inverse_variance(p::ScalingPipeline, σ2::Vector)
    return variance_transform(p.output_scaler, σ2)
end