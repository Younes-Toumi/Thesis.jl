""" file1. jl written the 13.04.2026.
This file explores a very basic surrogate modelling workflow. Also understanding:
    1. how to define input rv, a model and output rv
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

# 1. defining inputs X: X = [x₁, x₂]
x1 = RandomVariable.(Normal(0, 1), :x1);
x2 = RandomVariable.(Normal(0, 1), :x2);
X = [x1, x2]

# 2. defining output y: y = x₁² + x₂²
model = Model(rv -> rv.x1.^2 .+ rv.x2.^2, :y)

# 3. defining two sampling strategy
design_deterministic = FullFactorial([5, 5])
design_random = MonteCarlo(100)

design = design_random

# 4. generating training data
data_train = sample(X, design) # uses the desing to sample input
evaluate!(model, data_train) # modifies input `data_train` to add ouput `y`