using Plots
# ============================================================
# DATA — extracted from the run log
# ============================================================
n_train = [50, 100, 150, 200, 250]
 
# Expected response (mean) bounds: each row = (lower, upper) per n_train
gp_mean  = [-1.267 1.519; -1.362 1.275; -1.329 1.258; -1.372 1.304; -1.33  1.358]
pce_mean = [-0.322 0.327; -0.352 0.321; -1.61  1.061; -0.409 0.312; -0.708 0.686]
pck_mean = [-1.267 1.515; -1.31  1.249; -1.286 1.219; -1.371 1.304; -1.321 1.359]
 
# Pf bounds: each row = (lower, upper) per n_train
gp_pf  = [0.0 0.0; 0.0 0.382; 0.0 0.059; 0.0 0.395; 0.0 0.03]
pce_pf = [0.0 0.003; 0.0 0.001; 0.0 1.0; 0.0 0.0; 0.0 0.003]
pck_pf = [0.0 0.0; 0.0 0.053; 0.0 0.021; 0.0 0.39; 0.0 0.048]
 
# ============================================================
# REFERENCE VALUES — EDIT THESE to your own analytical/MCS values
# ============================================================
REF_MEAN_LOWER = -1.351
REF_MEAN_UPPER =  1.327
REF_PF_LOWER   =  0.0
REF_PF_UPPER   =  0.083
 
# ============================================================
# Colors per your spec
# ============================================================
COLOR_GP  = :blue
COLOR_PCK = :green
COLOR_PCE = :orange
 
# ============================================================
# Plot 1 — Expected response (mean): lower bound (top) / upper bound (bottom)
# ============================================================
p1_lower = plot(
    ylabel = "Lower bound",
    title  = "Expected-response LOWER bound vs. initial design size",
    legend = :left,
    grid   = true,
    margin = 5Plots.mm,
)
plot!(p1_lower, n_train, gp_mean[:, 1],  ls=:dash, lw=1.5, color=COLOR_GP,  marker=:circle, ms=5, msw=0, label="GP")
plot!(p1_lower, n_train, pck_mean[:, 1], ls=:dash, lw=1.5, color=COLOR_PCK, marker=:circle, ms=5, msw=0, label="PCK")
#plot!(p1_lower, n_train, pce_mean[:, 1], ls=:dash, lw=1.5, color=COLOR_PCE, marker=:circle, ms=5, msw=0, label="PCE")
hline!(p1_lower, [REF_MEAN_LOWER], lc=:black, ls=:dash, lw=1.0, label="Reference")
 
p1_upper = plot(
    xlabel = "Initial design size n₀",
    ylabel = "Upper bound",
    title  = "Expected-response UPPER bound vs. initial design size",
    legend = :left,
    grid   = true,
    margin = 5Plots.mm,
)
plot!(p1_upper, n_train, gp_mean[:, 2],  ls=:dash, lw=1.5, color=COLOR_GP,  marker=:circle, ms=5, msw=0, label="GP")
plot!(p1_upper, n_train, pck_mean[:, 2], ls=:dash, lw=1.5, color=COLOR_PCK, marker=:circle, ms=5, msw=0, label="PCK")
#plot!(p1_upper, n_train, pce_mean[:, 2], ls=:dash, lw=1.5, color=COLOR_PCE, marker=:circle, ms=5, msw=0, label="PCE")
hline!(p1_upper, [REF_MEAN_UPPER], lc=:black, ls=:dash, lw=1.0, label="Reference")
 
p1 = plot(p1_lower, p1_upper, layout=(2, 1), size=(1000, 500))
display(p1)
 
# ============================================================
# Plot 2 — Pf: lower bound (top) / upper bound (bottom)
# ============================================================
p2_lower = plot(
    ylabel = "Lower bound",
    title  = "Pf LOWER bound vs. initial design size",
    legend = :left,
    grid   = true,
    margin = 5Plots.mm,
)
plot!(p2_lower, n_train, gp_pf[:, 1],  ls=:dash, lw=1.5, color=COLOR_GP,  marker=:circle, ms=5, msw=0, label="GP")
plot!(p2_lower, n_train, pck_pf[:, 1], ls=:dash, lw=1.5, color=COLOR_PCK, marker=:circle, ms=5, msw=0, label="PCK")
#plot!(p2_lower, n_train, pce_pf[:, 1], ls=:dash, lw=1.5, color=COLOR_PCE, marker=:circle, ms=5, msw=0, label="PCE")
hline!(p2_lower, [REF_PF_LOWER], lc=:black, ls=:dash, lw=1.0, label="Reference")
 
p2_upper = plot(
    xlabel = "Initial design size n₀",
    ylabel = "Upper bound",
    title  = "Pf UPPER bound vs. initial design size",
    legend = :left,
    grid   = true,
    margin = 5Plots.mm,
)
plot!(p2_upper, n_train, gp_pf[:, 2],  ls=:dash, lw=1.5, color=COLOR_GP,  marker=:circle, ms=5, msw=0, label="GP")
plot!(p2_upper, n_train, pck_pf[:, 2], ls=:dash, lw=1.5, color=COLOR_PCK, marker=:circle, ms=5, msw=0, label="PCK")
#plot!(p2_upper, n_train, pce_pf[:, 2], ls=:dash, lw=1.5, color=COLOR_PCE, marker=:circle, ms=5, msw=0, label="PCE")
hline!(p2_upper, [REF_PF_UPPER], lc=:black, ls=:dash, lw=1.0, label="Reference")
 
p2 = plot(p2_lower, p2_upper, layout=(2, 1), size=(1000, 500))
display(p2)