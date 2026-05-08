This repo is for the master thesis I will be undergoing in the context of my studies in Computational Engineering at the Leibniz University Hanover (LUH), specifically in the Institute for Risk and Reliability.

Contains:
- src folder: functions, structs, etc
- demo folder: small demonstation of how to use the implemented functions

Currently contains:
- Surrogage Modelling Methods:
-   Gaussian Process
-   (work in progress)
- Metrics for Accuracy (mse, rmse, nrmse, q2)
- Scalings:
-   MinMaxScaler
-   ZScoreScaler


Todo:
- PCE: Sparce PCE
- GP: add piece wise constant mean?


# TODO - GP:
- include a seperate optimize.jl
- how should each restart be done for hyperparams? just random?


# TODO - PCE:
- implement the PCE logic to have comparable surrogate models
- add sparse functionality
- add a default selection for bases
- add max degree selection based on data


# TODO - Validation:
- ???


# TODO - Other stuff:
- Bootstrap for variance estimation for PCE, RS, etc...
- Adaptive sampling
- PCK

# Possible innovative ideas?
- Symbolic regression based construction of kernel?
- k-means for mean in GP?