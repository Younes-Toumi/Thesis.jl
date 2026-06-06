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