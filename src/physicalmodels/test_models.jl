model_ishigami = Model(
    df -> sin.(df.x1) .+ 7.0 .* sin.(df.x2).^2 .+ 0.1 .* (df.x3).^4 .* sin.(df.x1),
    :y
)

model_forrester = Model(
    df -> (6 .* df.x1 .- 2).^2 .* sin.(12 .* df.x1 .- 4),
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
    df -> df.x1.^2 .- df.x2.^2,
    :y
)


function ishigami(
    x1::Float64, x2::Float64, x3::Float64;
    a::Float64 = 7.0,
    b::Float64 = 0.1)

    return sin(x1) + a * sin(x2)^2 + b * x3^4 * sin(x1)
end

function forrester(x::Float64)
    return (6 * x - 2)^2 * sin(12 * x - 4)
end


function g_function(x1::Float64, x2::Float64)
    α_g = [2.0 3.0 1.0 4.0; 3.0 2.0 4.0 1.0]
    β_g = [-0.5 0.5 -0.5 0.5; -0.5 -0.5 0.5 0.5]
    c_g = [1.0, -1.5, -1.5, 2.0]

    result = 0.0
    for i in 1:4
        result += c_g[i] * exp(-α_g[1,i] * (x1 - β_g[1,i])^2 - α_g[2,i] * (x2 - β_g[2,i])^2)
    end
    return result
end



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