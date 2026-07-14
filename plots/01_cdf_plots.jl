using UncertaintyQuantification
using Distributions
using Plots

gr()

θ_lb, θ_ub   = -0.5, 0.5
x3_lb, x3_ub = -1.0, 1.0

x1 = RandomVariable(Normal(0.0, 1.0), :x1)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(θ_lb, θ_ub), :σ => 1.0)), :x2)
x3 = IntervalVariable(x3_lb, x3_ub, :x3)

# ---------------------------------------------------------
# 1) Parametric p-box and free p-box examples
# ---------------------------------------------------------

xgrid = range(-3, 3, length=500)

# -------------------------
# Parametric p-box (Normal with uncertain mean)
#
# F(x; μ) = Φ(x - μ) is DECREASING in μ (shifting the distribution right
# means, at a fixed x, LESS mass has accumulated). So:
#   upper envelope (max F)  <->  μ = θ_lb  (the smallest μ, shifted furthest left)
#   lower envelope (min F)  <->  μ = θ_ub  (the largest μ, shifted furthest right)
# -------------------------

cdf_x2_ub = cdf.(Normal(θ_lb, 1.0), xgrid)   # μ=θ_lb -> UPPER curve
cdf_x2_lb = cdf.(Normal(θ_ub, 1.0), xgrid)   # μ=θ_ub -> LOWER curve

means = range(θ_lb, θ_ub, length=6)

p1 = plot(
    xgrid,
    cdf_x2_lb,
    fillrange=cdf_x2_ub,
    fillalpha=0.25,
    color=:steelblue,
    linewidth=2,
    label="Admissible CDFs",
    xlabel="x",
    ylabel="CDF F(x)",
    title="Parametric p-box\nNormal(μ ∈ [-0.5,0.5], σ=1)",
    legend=:outerbottom,
)

plot!(
    p1,
    xgrid,
    cdf_x2_lb,
    color=:blue,
    linewidth=2,
    label="Lower bound (μ = θ_ub = 0.5)"
)

plot!(
    p1,
    xgrid,
    cdf_x2_ub,
    color=:red,
    linewidth=2,
    label="Upper bound (μ = θ_lb = -0.5)"
)

for μ in means
    plot!(
        p1,
        xgrid,
        cdf.(Normal(μ,1),xgrid),
        color=:gray,
        alpha=0.4,
        linewidth=1,
        label=""
    )
end

# -------------------------
# Free p-box (interval variable, NO distributional assumption)
#
# The true envelope over ALL distributions supported on [a,b] is achieved
# by the two extreme point masses:
#   F_upper(x) = 1{x >= a}   (all mass at the lower bound a -> earliest possible rise)
#   F_lower(x) = 1{x >= b}   (all mass at the upper bound b -> latest possible rise)
# These are STEP functions -- the defining "box" shape of a free p-box.
# -------------------------

xgrid3 = range(-3, 3, length=500)

cdf_x3_ub = Float64.(xgrid3 .>= x3_lb)   # point mass at a=x3_lb -> UPPER step
cdf_x3_lb = Float64.(xgrid3 .>= x3_ub)   # point mass at b=x3_ub -> LOWER step

p2 = plot(
    xgrid3,
    cdf_x3_lb,
    fillrange=cdf_x3_ub,
    fillalpha=0.25,
    color=:steelblue,
    linewidth=2,
    label="Admissible CDFs",
    xlabel="x",
    ylabel="CDF F(x)",
    title="Free p-box\nInterval x ∈ [-1,1]",
    legend=:outerbottom,
)

plot!(
    p2,
    xgrid3,
    cdf_x3_lb,
    color=:blue,
    linewidth=2,
    seriestype=:steppost,
    label="Lower bound (point mass at x₃ = x₃_ub = 1.0)"
)

plot!(
    p2,
    xgrid3,
    cdf_x3_ub,
    color=:red,
    linewidth=2,
    seriestype=:steppost,
    label="Upper bound (point mass at x₃ = x₃_lb = -1.0)"
)

# -------------------------
# Illustrative admissible members of the free p-box's credal set.
#
# IMPORTANT: a member must have support ENTIRELY inside [x3_lb, x3_ub] to be
# valid. Plain Normal/Uniform distributions leak probability mass outside
# any finite window, so they are NOT strictly admissible. Instead we use:
#   - Uniform(x3_lb, x3_ub)            -- exact full-width support
#   - truncated(Normal(...), a, b)     -- properly renormalised, F(a)=0, F(b)=1
#   - TriangularDist(a, b, mode)       -- exact bounded support
#   - a few point masses (step CDFs)   -- the extreme corner cases
# Every one of these must lie between the two bounding steps above.
# -------------------------

smooth_dists = [
    Uniform(x3_lb, x3_ub),
    truncated(Normal(0.0, 0.3), x3_lb, x3_ub),
    truncated(Normal(-0.4, 0.2), x3_lb, x3_ub),
    truncated(Normal(0.4, 0.2), x3_lb, x3_ub),
    truncated(Normal(0.0, 0.6), x3_lb, x3_ub),
    TriangularDist(x3_lb, x3_ub, 0.0),
]

for d in smooth_dists
    plot!(
        p2,
        xgrid3,
        cdf.(d, xgrid3),
        color=:gray,
        alpha=0.4,
        linewidth=1,
        label=""
    )
end

# a few near-degenerate point masses, to show the extreme corner members too
for c in [x3_lb + 0.15, -0.3, 0.3, x3_ub - 0.15]
    plot!(
        p2,
        xgrid3,
        Float64.(xgrid3 .>= c),
        color=:gray,
        alpha=0.4,
        linewidth=1,
        seriestype=:steppost,
        label=""
    )
end

pp = plot(
    p1,
    p2,
    layout=(1,2),
    size=(1200,600),
    top_margin=7Plots.mm
)

display(pp)

# ---------------------------------------------------------
# 2) Type 1 / Type 2 / Type 3 comparison
# ---------------------------------------------------------

# Type 1
p_type1 = plot(
    xgrid,
    cdf.(Normal(0,1),xgrid),
    color=:blue,
    linewidth=2,
    xlabel="x",
    ylabel="CDF F(x)",
    title="Type 1 input\nPrecise probability",
    label="Unique CDF",
    legend=:outerbottom,
)

# Type 2
p_type2 = plot(
    xgrid,
    cdf_x2_lb,
    fillrange=cdf_x2_ub,
    fillalpha=0.25,
    color=:steelblue,
    linewidth=2,
    xlabel="x",
    ylabel="CDF F(x)",
    title="Type 2 input\nParametric p-box",
    label="Admissible CDFs",
    legend=:outerbottom,
)

plot!(
    p_type2,
    xgrid,
    cdf_x2_lb,
    color=:blue,
    linewidth=2,
    label="Lower bound"
)

plot!(
    p_type2,
    xgrid,
    cdf_x2_ub,
    color=:red,
    linewidth=2,
    label="Upper bound"
)

# Type 3
p_type3 = plot(
    xgrid3,
    cdf_x3_lb,
    fillrange=cdf_x3_ub,
    fillalpha=0.25,
    color=:steelblue,
    linewidth=2,
    xlabel="x",
    ylabel="CDF F(x)",
    title="Type 3 input\nFree p-box",
    label="Admissible CDFs",
    legend=:outerbottom,
)

plot!(
    p_type3,
    xgrid3,
    cdf_x3_lb,
    color=:blue,
    linewidth=2,
    seriestype=:steppost,
    label="Lower bound"
)

plot!(
    p_type3,
    xgrid3,
    cdf_x3_ub,
    color=:red,
    linewidth=2,
    seriestype=:steppost,
    label="Upper bound"
)

ppp = plot(
    p_type1,
    p_type2,
    p_type3,
    layout=(1,3),
    size=(1500,500),
    top_margin=7Plots.mm
)

display(ppp)