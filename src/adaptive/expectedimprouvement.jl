function propose_next(x, gp, f_best; N=2000)

    best_x = nothing
    best_ei = -Inf

    for i in 1:N
        ei = EI(x, gp, f_best)

        if ei > best_ei
            best_ei = ei
            best_x = x
        end
    end

    return best_x, best_ei
end

function EI(x, model, f_best; N=50)
    improvements = 0.0

    for i in 1:N
        y = predict(model, x)   # stochastic or surrogate-based
        println(y, f_best)
        improvements += max(f_best .- y, 0.0)
    end

    return improvements ./ N
end