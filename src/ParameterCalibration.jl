module ParameterCalibration

using LinearAlgebra
using Statistics
using Random
using ScoreEstimation
using ProgressMeter
using FastSDE

# Public API modules/files
include("types.jl")
include("numerics.jl")
include("gfdt.jl")
include("calibration.jl")
include("estimators.jl")
include("finite_diff.jl")
include("fastsde.jl")
include("config.jl")
include("estimators_builders.jl")
include("observables.jl")

export
    # Types
    GFDTModel,
    SimSpec,
    # Estimator containers and config
    AbstractJacobianEstimator,
    NNTrainConfig,
    AnalyticJacobianEstimator,
    GaussianJacobianEstimator,
    NeuralJacobianEstimator,
    FiniteDiffJacobianEstimator,
    SensitivityMethod,
    GFDTAnalyticScore,
    GFDTGaussianScore,
    GFDTNeuralScore,
    FiniteDifference,
    # Core GFDT
    B_gfdt,
    # Calibration utilities
    calibration_loop,
    # Score estimators
    gaussian_score_from_data,
    # Numerical baselines
    finite_difference_jacobian,
    # Integration
    simulate,
    # Config
    load_config,
    load_extra_config,
    # Estimator builders/utilities
    build_responses,
    build_analytic_estimator,
    build_gaussian_estimator,
    build_neural_estimator,
    build_finite_diff_estimator,
    stats_A, build_A_of_x, build_make_A_of_x

end # module
