function ishigami(
    x1::Float64, x2::Float64, x3::Float64;
    a::Float64 = 7.0,
    b::Float64 = 0.1)

    return sin(x1) + a * sin(x2)^2 + b * x3^4 * sin(x1)
end