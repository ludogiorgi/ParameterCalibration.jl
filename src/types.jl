"""
Core types used throughout the ParameterCalibration package.

Includes:
- GFDTModel: container for score and dynamics with parameter derivatives.
- SimSpec: simulation specification passed to the FastSDE wrapper.
- SensitivityMethod hierarchy: method tags and lightweight holders.
"""

# Model with score and dynamics (normalized coordinates)
Base.@kwdef struct GFDTModel{Sf,Sd,Sj,Ff,Sg,DF,DG,DDivF,DdivM,DDivDivM,Tθ,Tx}
    # Score & derivatives
    s::Sf                             # s(x) :: d-vector
    divs::Sd                          # div s(x) :: scalar (used in :isotropic)
    Js::Sj                            # ∇s(x) :: d×d (used in :general)

    # Drift/Diffusion & θ-Jacobians
    F::Ff                             # F(x, θ) :: d-vector
    Σ::Sg                             # Σ(x, θ) :: d×d
    dF_dθ::DF                         # dF/dθ(x, θ) :: d×p, columns are Ψ_j
    dΣ_dθ::DG                         # dΣ/dθ(x, θ) :: d×d×p, slices[:,:,j] = Λ_j

    # Analytic spatial divergences
    div_dF_dθ::DDivF                  # ∇·Ψ_j(x) :: p-vector
    divM::DdivM                       # (div M_j)(x) :: d×p
    divdivM::DDivDivM                 # ∇·(div M_j)(x) :: p-vector

    # Parameters & mode
    θ::Tθ
    mode::Symbol = :general           # :general or :isotropic
    xeltype::Type{Tx}=Float64
end

# Simulation specification (FastSDE wrapper)
Base.@kwdef struct SimSpec
    dt::Float64
    Nsteps::Int
    n_ens::Int = 1
    u0::AbstractVector = zeros(1)
    burn_in::Int = 0
    resolution::Int = 1
    rng::Any = Random.default_rng()
    seed::Int = 12345
    timestepper::Symbol = :euler
    sigma_inplace::Bool = true
    noise::Symbol = :shared   # :shared or :independent (reserved)
    Δt_multiplier::Float64 = 1.0
    Tmax::Float64 = 10.0
end

# Method tags for building sensitivities
abstract type SensitivityMethod end
struct GFDTAnalyticScore  <: SensitivityMethod end
struct GFDTGaussianScore  <: SensitivityMethod end

# Lightweight method tag for NN-based score
struct GFDTNeuralScore <: SensitivityMethod end

# Finite-difference method descriptor
Base.@kwdef struct FiniteDifference <: SensitivityMethod
    h_rel::Float64 = 1e-4
    h_abs::Float64 = 1e-6
end

# Convenience constructor to wrap existing NN closures
# Deprecated constructor removed during cleanup: use GFDTNeuralScore() as a tag

# Immutable container for NN + KGMM training settings read from config
Base.@kwdef struct NNTrainConfig
    preprocessing::Bool = true
    σ::Float64
    neurons::Vector{Int}
    n_epochs::Int
    epochs_re::Int = 10
    epochs_ref::Int
    batch_size::Int
    lr::Float64
    use_gpu::Bool
    verbose::Bool
    probes::Int
    rademacher::Bool
    kgmm_kwargs::NamedTuple
end

# SimRunConfig removed; use SimSpec everywhere (now includes Δt_multiplier and Tmax)

# Estimator containers
abstract type AbstractJacobianEstimator end

Base.@kwdef struct AnalyticJacobianEstimator <: AbstractJacobianEstimator
    model_view::GFDTModel
    θ::Vector{Float64}
    X::Matrix{Float64}
    Δt::Float64
    Tmax::Float64
    mean_center::Bool = true
    responses::Union{Nothing,Array{Float64,3}} = nothing
    S::Union{Nothing,Matrix{Float64}} = nothing
end

Base.@kwdef struct GaussianJacobianEstimator <: AbstractJacobianEstimator
    model_view::GFDTModel
    θ::Vector{Float64}
    X::Matrix{Float64}
    Δt::Float64
    Tmax::Float64
    mean_center::Bool = true
    responses::Union{Nothing,Array{Float64,3}} = nothing
    S::Union{Nothing,Matrix{Float64}} = nothing
end

Base.@kwdef struct NeuralJacobianEstimator <: AbstractJacobianEstimator
    model_view::GFDTModel
    θ::Vector{Float64}
    X::Matrix{Float64}
    Δt::Float64
    Tmax::Float64
    mean_center::Bool = true
    nn_method::Union{Nothing,GFDTNeuralScore} = nothing
    responses::Union{Nothing,Array{Float64,3}} = nothing
    S::Union{Nothing,Matrix{Float64}} = nothing
end

Base.@kwdef struct FiniteDiffJacobianEstimator <: AbstractJacobianEstimator
    θ::Vector{Float64}
    simulator::Function
    free_idx::Vector{Int} = collect(1:length(θ))
    h_rel::Float64 = 1e-4
    h_abs::Float64 = 1e-6
    scheme::Symbol = :central
    n_rep::Int = 1
    S::Union{Nothing,Matrix{Float64}} = nothing
end
