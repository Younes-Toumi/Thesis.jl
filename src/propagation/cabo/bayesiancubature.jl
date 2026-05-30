function PVC(gp, u, θp)

    θμ1, θμ2 = θp
    u1, u2 = u

    x = hcat([u1], [u2], [θμ1], [θμ2])

    μ, σ = predict(gp, x)

    σ2 = σ[1]^2

    w = pdf.(Uniform(0,1), u1) * pdf.(Uniform(0,1), u2)

    return w * σ2
end