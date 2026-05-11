""" file1. jl written the 13.04.2026.
This file explores a very basic surrogate modelling workflow. Also understanding:
    1. how to define input rv, a model and :y rv
    2. defining sampling strategy
    3. desining and testing metamodel

Next things the would be interresting to try out next:
    - adaptive sampling
    - probability box
    - implementing other surrogate models (x
"""

run(`clear`)

using SurrogateModelling
using UncertaintyQuantification
using PlotlyJS
using DataFrames

# 1. defining inputs X: X = [x₁, x₂]
x1 = RandomVariable.(Uniform(-5, 5), :x1);
x2 = RandomVariable.(Uniform(-5, 5), :x2);
X = [x1, x2]

# 2. defining :y
model = Model(
    rv -> (rv.x1 .^ 2 .+ rv.x2 .- 11) .^ 2 .+ (rv.x1 .+ rv.x2 .^ 2 .- 7) .^ 2,
    :y
) # himmelblau

# 3. defining two sampling strategy
design = LatinHypercubeSampling(500)

# 4. generating training data
data_train = sample(X, design) # uses the desing to sample input
evaluate!(model, data_train) # modifies input `data_train` to add ouput `y`


# ── infer input column names ──────────────────────────────
x_names = propertynames(data_train[:, Not(:y)])
length(x_names) == 2 || error("plot3d_gp requires exactly 2 input columns, got $(length(x_names))")

x1_name, x2_name = x_names
x1_vals = data_train[:, x1_name]
x2_vals = data_train[:, x2_name]
y_vals  = data_train[:, :y]

# ── build grid for surface ────────────────────────────────
x1_lim = [0.95*minimum(x1_vals), 1.05*maximum(x1_vals)]
x2_lim = [0.95*minimum(x2_vals), 1.05*maximum(x2_vals)]
y_lim =  [0.95*minimum(y_vals), 1.05*maximum(y_vals)]

# ── interpolate onto grid using nearest neighbour ─────────
# simple approach: build surface from scattered data_train via triangulation
# for a clean surface, we just scatter plot the data_train
layout = Layout(
        margin = attr(l=10, r=10, t=30, b=10),   # left, right, top, bottom
    scene = attr(
        xaxis = attr(title=string(x1_name), range=x1_lim),
        yaxis = attr(title=string(x2_name), range=x2_lim),
        zaxis = attr(title="y",             range=y_lim),
        aspectratio = attr(x=1, y=1, z=0.7),  # ← squish z axis

        ),
    title = "Himmelblau"
)

PlotlyJS.plot(
    PlotlyJS.mesh3d(
        x          = x1_vals,
        y          = x2_vals,
        z          = y_vals,
        intensity  = y_vals,          # ← values that drive the color
        colorscale = "Jet",           # ← "Jet", "Viridis", "Plasma", "RdBu", etc.
        showscale  = true,            # ← shows the colorbar
        colorbar   = attr(
            title      = "y",
            titleside  = "right",
            thickness  = 15,
        ),
    ),
    layout
)