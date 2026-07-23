using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
Random.seed!(42)

# ── Variables & Model ─────────────────────────────────────────────

y_true = rand(Normal(0.0, 1.0), 1001)
y_pred = y_true .+ 0.1*rand()

mse_val     = mse(y_true, y_pred)
rmse_val    = rmse(y_true, y_pred)
nrmse_val   = nrmse(y_true, y_pred)
nrmse_val_2 = nrmse(y_true, y_pred, method=:minmax)
q2_val      = q2(y_true, y_pred)

println("MSE:               $(round(mse_val, digits=5))")
println("RMSE:              $(round(rmse_val, digits=5))")
println("nRMSE (std):       $(round(nrmse_val, digits=5))")
println("nRMSE (minmax):    $(round(nrmse_val_2, digits=5))")
println("Q²:                $(round(q2_val, digits=5))")