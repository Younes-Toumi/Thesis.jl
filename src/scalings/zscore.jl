struct ZScoreScaler <: AbstractScaler
    μ::Vector{Float64}
    σ::Vector{Float64}
end

function fit_scaler(::Type{ZScoreScaler}, X::Matrix)
    μ = vec(mean(X, dims=1))
    σ = vec(std(X, dims=1))
    σ[σ .== 0.0] .= 1.0
    return ZScoreScaler(μ, σ)
end

function transform(s::ZScoreScaler, X::Matrix)
    return (X .- s.μ') ./ s.σ'
end

function inverse_transform(s::ZScoreScaler, X::Matrix)
    return X .* s.σ' .+ s.μ'
end

function variance_transform(s::ZScoreScaler, V::AbstractArray)
    return V .* (s.σ .^ 2)
end