struct AugmentedModel
    g::Function
    u_specs::Vector
    θ_specs::Vector
end

function sample(am::AugmentedModel, n::Int)
    d_u = length(am.u_specs)
    d_θ = length(am.θ_specs)

    U = [rand.(am.u_specs) for _ in 1:n]
    Θ = [rand.(am.θ_specs) for _ in 1:n]

    return U, Θ
end

function to_physical(am::AugmentedModel, u, θ)
    x = similar(θ)

    for i in eachindex(θ)
        if am.u_specs[i] isa Normal
            x[i] = θ[i] + std(am.u_specs[i]) * u[i]
        else
            x[i] = quantile(am.u_specs[i], u[i])
        end
    end

    return x
end

