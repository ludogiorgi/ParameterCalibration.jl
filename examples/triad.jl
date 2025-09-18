using Pkg
Pkg.activate(".")
Pkg.instantiate()

using ScoreEstimation
using Plots
using Statistics
using KernelDensity
using FastSDE

# ---------------- 3D Triad SDE ----------------
const params = (dᵤ=0.2, wᵤ=0.4, dₜ=2.0, σ₁=0.3, σ₂=0.3)

dim = 3
dt = 0.01
Nsteps = 10_000
u0 = [0.0, 0.0, 0.0]
resolution = 10

function drift!(du, u, t)
    du[1] = -params.dᵤ * u[1] - params.wᵤ * u[2] + u[3]
    du[2] = -params.dᵤ * u[2] + params.wᵤ * u[1]
    du[3] = -params.dₜ * u[3]
end

function diffusion!(du, u, t)
    du[1] = params.σ₁
    du[2] = params.σ₂
    du[3] = 1.5 * (tanh(u[1]) + 1)
end

obs_nn = evolve(u0, dt, Nsteps, drift!, diffusion!;
                timestepper=:euler, resolution=resolution, n_ens=100)

@info "Trajectory shape: $(size(obs_nn))"

# Normalize observations
M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S
obs_uncorr = obs

