using Plots
# ============================================================
# DATA — extracted from the run log
# ============================================================
n_train = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

# Expected response (mean) bounds: each row = (lower, upper) per n_train
gp_mean  = [-1.193 1.291;
            -1.278 1.280;
            -1.352 1.316;
            -1.293 1.320;
            -1.374 1.378;
            -1.350 1.310;
            -1.355 1.342;
            -1.350 1.369;
            -1.350 1.316;
            -1.362 1.367]

pck_mean = [-1.158 1.328;
            -1.232 1.282;
            -1.354 1.306;
            -1.296 1.320;
            -1.402 1.367;
            -1.343 1.300;
            -1.366 1.337;
            -1.343 1.363;
            -1.352 1.313;
            -1.364 1.359]

# Pf bounds: each row = (lower, upper) per n_train
gp_pf  = [0.0 0.0; 0.0 0.382; 0.0 0.059; 0.0 0.395; 0.0 0.03]
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
 
# ============================================================
# Plot 1 — Expected response (mean): lower bound (top) / upper bound (bottom)
# ============================================================
p1_lower = plot(
    ylabel = "Lower bound",
    title  = "Expected-response LOWER bound vs. initial design size",
    legend = :left,
    grid   = true,
    ylims  = [-1.4, -1.15],
    margin = 5Plots.mm,
)
plot!(p1_lower, n_train, gp_mean[:, 1],  ls=:dash, lw=1.5, color=COLOR_GP,  marker=:circle, ms=5, msw=0, label="GP")
plot!(p1_lower, n_train, pck_mean[:, 1], ls=:dash, lw=1.5, color=COLOR_PCK, marker=:circle, ms=5, msw=0, label="PCK")
hline!(p1_lower, [REF_MEAN_LOWER], lc=:black, ls=:dash, lw=1.0, label="Reference")
 
p1_upper = plot(
    xlabel = "Initial design size n₀",
    ylabel = "Upper bound",
    title  = "Expected-response UPPER bound vs. initial design size",
    legend = :left,
    grid   = true,
    ylims  = [1.2, 1.4],
    margin = 5Plots.mm,
)
plot!(p1_upper, n_train, gp_mean[:, 2],  ls=:dash, lw=1.5, color=COLOR_GP,  marker=:circle, ms=5, msw=0, label="GP")
plot!(p1_upper, n_train, pck_mean[:, 2], ls=:dash, lw=1.5, color=COLOR_PCK, marker=:circle, ms=5, msw=0, label="PCK")
hline!(p1_upper, [REF_MEAN_UPPER], lc=:black, ls=:dash, lw=1.0, label="Reference")
 
p1 = plot(p1_lower, p1_upper, layout=(2, 1), size=(1000, 500))
display(p1)
 
# # ============================================================
# # Plot 2 — Pf: lower bound (top) / upper bound (bottom)
# # ============================================================
# p2_lower = plot(
#     ylabel = "Lower bound",
#     title  = "Pf LOWER bound vs. initial design size",
#     legend = :left,
#     grid   = true,
#     margin = 5Plots.mm,
# )
# plot!(p2_lower, n_train, gp_pf[:, 1],  ls=:dash, lw=1.5, color=COLOR_GP,  marker=:circle, ms=5, msw=0, label="GP")
# plot!(p2_lower, n_train, pck_pf[:, 1], ls=:dash, lw=1.5, color=COLOR_PCK, marker=:circle, ms=5, msw=0, label="PCK")
# hline!(p2_lower, [REF_PF_LOWER], lc=:black, ls=:dash, lw=1.0, label="Reference")
 
# p2_upper = plot(
#     xlabel = "Initial design size n₀",
#     ylabel = "Upper bound",
#     title  = "Pf UPPER bound vs. initial design size",
#     legend = :left,
#     grid   = true,
#     margin = 5Plots.mm,
# )
# plot!(p2_upper, n_train, gp_pf[:, 2],  ls=:dash, lw=1.5, color=COLOR_GP,  marker=:circle, ms=5, msw=0, label="GP")
# plot!(p2_upper, n_train, pck_pf[:, 2], ls=:dash, lw=1.5, color=COLOR_PCK, marker=:circle, ms=5, msw=0, label="PCK")
# hline!(p2_upper, [REF_PF_UPPER], lc=:black, ls=:dash, lw=1.0, label="Reference")
 
# p2 = plot(p2_lower, p2_upper, layout=(2, 1), size=(1000, 500))
# display(p2)


using UncertaintyQuantification
using SurrogateModelling

x1 = RandomVariable(Normal(0, 1), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(
    Dict( :μ => Interval(-1.5, 1.5), 
          :σ => Interval(0.1, 0.5)
    )), :x2
)
x3 = IntervalVariable(-pi, pi, :x3)

specs = InputSpec.([x1, x2, x3])

physical_model = Model(df -> df.x1 .+ df.x2 .+ df.x3, :y)
data_augmented, data_physical = build_augmented_design(
    physical_model,     # simple UQ.jl model
    specs,              # Input Specification
    1000                  # samples
)


x = RandomVariable(Uniform(-pi, pi, :x))
specs = InputSpec.([x])

physical_model = Model(df -> df.x, :y)
data_augmented, data_physical = build_augmented_design(
    physical_model,     # simple UQ.jl model
    specs,              # Input Specification
    1000                  # samples
)

spec_names(specs)