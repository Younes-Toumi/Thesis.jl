"""
File containing some analytical UQ models using during the thesis. 

Forrester:  f(x)            = (6x - 2)² sin(12x - 4)
Himmelblau: f(x₁, x₂)       = (x₁² + x₂ - 11)² + (x₁ + x₂² - 7)²
Ishigami:   f(x₁, x₂, x₃)   = sin(x₁) + 7 sin(x₂)² + 0.1 x₃⁴ sin(x₁)
Gaussian Mixture:
            g(x₁, x₂)       = Σ γᵢ exp(-αᵢ₁ (x₁ - βᵢ₁)² - αᵢ₂ (x₂ - βᵢ₂)²)
"""

model_ishigami = Model(
    df -> sin.(df.x1) .+ 7.0 .* sin.(df.x2).^2 .+ 0.1 .* (df.x3).^4 .* sin.(df.x1),
    :y
)

model_forrester = Model(
    df -> (6 .* df.x .- 2).^2 .* sin.(12 .* df.x .- 4),
    :y
)

model_gfunction = Model(
    df -> begin
        α_g = [2.0 3.0 1.0 4.0; 3.0 2.0 4.0 1.0]
        β_g = [-0.5 0.5 -0.5 0.5; -0.5 -0.5 0.5 0.5]
        c_g = [1.0, -1.5, -1.5, 2.0]    

        sum(
            c_g[i] .* exp.(
                .- α_g[1, i] .* (df.x1 .- β_g[1, i]).^2
                .- α_g[2, i] .* (df.x2 .- β_g[2, i]).^2
                )
            for i in 1:4
        )
    end,
    :y
)

model_simple = Model(
    df -> df.x1 .- df.x2 - -2,
    :y
)

model_himmelblau = Model(
    df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2,
    :y
)

function g_function_E(μ1::Float64, μ2::Float64; σ::Float64 = 0.1)
    α_g = [2.0 3.0 1.0 4.0; 3.0 2.0 4.0 1.0]'
    β_g = [-0.5 0.5 -0.5 0.5; -0.5 -0.5 0.5 0.5]'
    c_g = [1.0, -1.5, -1.5, 2.0]'

    result = 0.0
    σ² = σ^2

    for i in 1:4

        α1 = α_g[i,1]
        α2 = α_g[i,2]

        β1 = β_g[i,1]
        β2 = β_g[i,2]

        # E[exp(-α(X-β)^2)]
        term1 =
            exp(
                -α1 * (μ1 - β1)^2 /
                (1 + 2 * α1 * σ²)
            ) /
            sqrt(1 + 2 * α1 * σ²)

        term2 =
            exp(
                -α2 * (μ2 - β2)^2 /
                (1 + 2 * α2 * σ²)
            ) /
            sqrt(1 + 2 * α2 * σ²)

        result += c_g[i] * term1 * term2
    end

    return result
end



export
    model_ishigami, model_forrester, model_gfunction, model_simple, model_himmelblau,
    g_function_E
