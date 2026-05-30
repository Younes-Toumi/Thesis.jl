# initialize GP
# initialize D₀

for iter in 1:max_iter

    # ==========================================
    # PART 1 — epistemic BO
    # ==========================================

    # compute x* (incumbent)
    # compute EI_v(θ)
    # optimize EI_v using PSO
    # code to get θ_plus ...


    # ==========================================
    # PART 2 — aleatory BC
    # ==========================================

    # conditioned on θ_plus:
    # choose best aleatory point u_plus
    # code to get u_plus ...


    # ==========================================
    # PART 3 — expensive model evaluation
    # ==========================================
    # new evaluation y_plus = model(u_plus, θ_plus)
    # adding new sample append!(D, (u_plus, θ_plus, y_plus))


    # ==========================================
    # PART 4 — retrain GP
    # ==========================================

    # retraining gp fit_gp()


    # ==========================================
    # PART 5 — convergence
    # ==========================================
    # if BO_converged && BC_converged
    #    break
    # end
    
end