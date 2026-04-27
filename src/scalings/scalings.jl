abstract type AbstractScaler end

function fit_scaler(::Type{T}, X) where T
    error("fit_scaler not implemented for type $T")
end

function transform(s::AbstractScaler, X)
    error("transform not implemented for $(typeof(s))")
end

function inverse_transform(s::AbstractScaler, X)
    error("inverse_transform not implemented for $(typeof(s))")
end

function variance_transform(s::AbstractScaler, V)
    error("variance_transform not implemented for $(typeof(s))")
end