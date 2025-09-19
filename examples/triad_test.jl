#!/usr/bin/env julia
#=
triad_test.jl

Diagnostic script for the three-dimensional triad example.
Compares stationary statistics between the true parameters and an
initial perturbed guess, produces visualizations of their bivariate
marginals, and reports GFDT response functions and Jacobians for both
parameter sets.

Run from the project root with:
    julia --project=examples examples/triad_test.jl
=#

using Pkg
Pkg.activate(@__DIR__)
try
    Pkg.develop(path=joinpath(@__DIR__, ".."))
catch err
    @warn "Development path add failed (likely already added)" exception=err
end

using ParameterCalibration
const PC = ParameterCalibration

using Statistics, Random, LinearAlgebra, Printf
using GLMakie
using KernelDensity
using TOML

# --------------------------------------------------------------------------------------
# Shared helpers (mirrors definitions in triad_compute.jl)
# --------------------------------------------------------------------------------------

function load_triad_model(cfg_path::AbstractString)
    raw = TOML.parsefile(cfg_path)
    mdl = get(raw, "triad_model", Dict{String,Any}())
    return (
        d_u = Float64(get(mdl, "d_u", 0.2)),
        w_u = Float64(get(mdl, "w_u", 0.4)),
        d_t = Float64(get(mdl, "d_t", 2.0)),
        sigma1 = Float64(get(mdl, "sigma1", 0.3)),
        sigma2 = Float64(get(mdl, "sigma2", 0.3)),
        sigma3 = Float64(get(mdl, "sigma3", 1.5)),
    )
end

function build_triad_A_of_x(μ::AbstractVector, Σ::AbstractVector)
    μv = collect(Float64.(μ))
    Σv = collect(Float64.(Σ))
    function A_of_x(x::AbstractVector)
        u1 = x[1] * Σv[1] + μv[1]
        u2 = x[2] * Σv[2] + μv[2]
        u3 = x[3] * Σv[3] + μv[3]
        return Float64[u1^2, u2^2, u1 * u2, u3^2, u1 * u3, u2 * u3]
    end
    labels = ["u1^2", "u2^2", "u1u2", "u3^2", "u1u3", "u2u3"]
    return (A_of_x, labels)
end

function make_builders(μ::AbstractVector, Σ::AbstractVector)
    μv = collect(Float64.(μ))
    Σv = collect(Float64.(Σ))
    invΣ = 1.0 ./ Σv
    p = 6

    function F_norm(x::AbstractVector, θ::AbstractVector)
        u1 = x[1] * Σv[1] + μv[1]
        u2 = x[2] * Σv[2] + μv[2]
        u3 = x[3] * Σv[3] + μv[3]
        f1 = -θ[1] * u1 - θ[2] * u2 + u3
        f2 = -θ[1] * u2 + θ[2] * u1
        f3 = -θ[3] * u3
        return Float64[f1 * invΣ[1], f2 * invΣ[2], f3 * invΣ[3]]
    end

    function Σ_norm(x::AbstractVector, θ::AbstractVector)
        u1 = x[1] * Σv[1] + μv[1]
        σ1 = θ[4] * invΣ[1]
        σ2 = θ[5] * invΣ[2]
        σ3 = θ[6] * (tanh(u1) + 1.0) * invΣ[3]
        S = zeros(Float64, 3, 3)
        S[1, 1] = σ1
        S[2, 2] = σ2
        S[3, 3] = σ3
        return S
    end

    function dF_dθ_norm(x::AbstractVector, θ::AbstractVector)
        u1 = x[1] * Σv[1] + μv[1]
        u2 = x[2] * Σv[2] + μv[2]
        u3 = x[3] * Σv[3] + μv[3]
        J = zeros(Float64, 3, length(θ))
        J[1, 1] = -u1 * invΣ[1]
        J[2, 1] = -u2 * invΣ[2]
        J[1, 2] = -u2 * invΣ[1]
        J[2, 2] = u1 * invΣ[2]
        J[3, 3] = -u3 * invΣ[3]
        return J
    end

    function dΣ_dθ_norm(x::AbstractVector, θ::AbstractVector)
        u1 = x[1] * Σv[1] + μv[1]
        A = zeros(Float64, 3, 3, length(θ))
        A[1, 1, 4] = invΣ[1]
        A[2, 2, 5] = invΣ[2]
        A[3, 3, 6] = (tanh(u1) + 1.0) * invΣ[3]
        return A
    end

    function div_dF_dθ_norm(_x::AbstractVector, _θ::AbstractVector)
        return Float64[-2.0, 0.0, -1.0, 0.0, 0.0, 0.0]
    end

    divM_norm = (_x, _θ) -> zeros(Float64, 3, p)
    divdivM_norm = (_x, _θ) -> zeros(Float64, p)

    return (
        F_norm = F_norm,
        Σ_norm = Σ_norm,
        dF_dθ_norm = dF_dθ_norm,
        dΣ_dθ_norm = dΣ_dθ_norm,
        div_dF_dθ_norm = div_dF_dθ_norm,
        divM_norm = divM_norm,
        divdivM_norm = divdivM_norm,
    )
end

function make_simulators(spec::PC.SimSpec)
    function simulator_obs(θ::AbstractVector)
        drift_θ! = function (du, u, t)
            du[1] = -θ[1] * u[1] - θ[2] * u[2] + u[3]
            du[2] = -θ[1] * u[2] + θ[2] * u[1]
            du[3] = -θ[3] * u[3]
        end
        sigma_θ! = function (du, u, t)
            du[1] = θ[4]
            du[2] = θ[5]
            du[3] = θ[6] * (tanh(u[1]) + 1.0)
        end
        PC.simulate(spec.u0, drift_θ!, sigma_θ!; spec = spec)
    end

    function make_simulator_norm(μ::AbstractVector, Σ::AbstractVector)
        μv = collect(Float64.(μ))
        Σv = collect(Float64.(Σ))
        function simulator_norm(θ::AbstractVector)
            Xp = simulator_obs(θ)
            return (Xp .- μv) ./ Σv
        end
        return simulator_norm
    end

    return simulator_obs, make_simulator_norm
end

# --------------------------------------------------------------------------------------
# Load configuration and simulate trajectories
# --------------------------------------------------------------------------------------

cfg_path = joinpath(@__DIR__, "..", "config", "config_triad.toml")
nn_cfg, spec_obs = PC.load_config(cfg_path)
extra = PC.load_extra_config(cfg_path)
model_cfg = load_triad_model(cfg_path)

θ_true = Float64[model_cfg.d_u, model_cfg.w_u, model_cfg.d_t, model_cfg.sigma1, model_cfg.sigma2, model_cfg.sigma3]
θ_init = θ_true .* extra.calibration.θ_init_multipliers

Δt_eff = spec_obs.dt * spec_obs.resolution * spec_obs.Δt_multiplier

simulator_obs, make_simulator_norm = make_simulators(spec_obs)

println("Simulating trajectories for θ_true and θ_init ...")
X_true = simulator_obs(θ_true)
X_init = simulator_obs(θ_init)

μ_true = vec(mean(X_true, dims = 2))
Σ_true = vec(std(X_true, dims = 2))
X_true_norm = (X_true .- μ_true) ./ Σ_true
X_init_norm = (X_init .- μ_true) ./ Σ_true

A_of_x, obs_labels = build_triad_A_of_x(μ_true, Σ_true)

mean_true = PC.stats_A(X_true_norm, A_of_x)
mean_init = PC.stats_A(X_init_norm, A_of_x)

println("Observables mean values (normalized trajectory processed through A_of_x):")
for (i, lab) in enumerate(obs_labels)
    @printf("  %-8s  true=% .6e   init=% .6e   diff=% .3e
", lab, mean_true[i], mean_init[i], mean_true[i] - mean_init[i])
end

# --------------------------------------------------------------------------------------
# 2D kernel density comparison for (u1, u2)
# --------------------------------------------------------------------------------------

u1_true = vec(X_true[1, :])
u2_true = vec(X_true[2, :])
u1_init = vec(X_init[1, :])
u2_init = vec(X_init[2, :])

nbins = 160
margin1 = 0.1 * (maximum(u1_true) - minimum(u1_true))
margin2 = 0.1 * (maximum(u2_true) - minimum(u2_true))
xmin = min(minimum(u1_true), minimum(u1_init)) - margin1
ymin = min(minimum(u2_true), minimum(u2_init)) - margin2
xmax = max(maximum(u1_true), maximum(u1_init)) + margin1
ymax = max(maximum(u2_true), maximum(u2_init)) + margin2
xgrid = range(xmin, xmax; length = nbins)
ygrid = range(ymin, ymax; length = nbins)

kd_true = kde((u1_true, u2_true), (xgrid, ygrid))
kd_init = kde((u1_init, u2_init), (xgrid, ygrid))

fig = Figure(resolution = (960, 420))
ax1 = Axis(fig[1, 1], xlabel = "u1", ylabel = "u2", title = "θ_true")
ax2 = Axis(fig[1, 2], xlabel = "u1", ylabel = "u2", title = "θ_init")
hm_true = heatmap!(ax1, xgrid, ygrid, kd_true.density'; colormap = :viridis)
heatmap!(ax2, xgrid, ygrid, kd_init.density'; colormap = :viridis)
Colorbar(fig[:, 3], hm_true, label = "PDF")
figdir = joinpath(@__DIR__, "..", "figures"); mkpath(figdir)
density_path = joinpath(figdir, "triad_density_comparison.png")
save(density_path, fig)
println("Saved density comparison figure to: " * density_path)

# --------------------------------------------------------------------------------------
# Build GFDT estimators for θ_true and θ_init
# --------------------------------------------------------------------------------------

function build_estimators_for(θ_ref::Vector{Float64},
                              simulator_obs::Function,
                              make_simulator_norm::Function,
                              spec::PC.SimSpec,
                              nn_cfg::PC.NNTrainConfig,
                              Δt_eff::Float64)
    X_ref = simulator_obs(θ_ref)
    μr = vec(mean(X_ref, dims = 2))
    Σr = vec(std(X_ref, dims = 2))
    Xn = (X_ref .- μr) ./ Σr
    A_of_x, labels = build_triad_A_of_x(μr, Σr)
    builders = make_builders(μr, Σr)
    model = PC.GFDTModel(
        s = x -> zeros(Float64, 3),
        divs = x -> 0.0,
        Js = x -> zeros(Float64, 3, 3),
        F = builders.F_norm,
        Σ = builders.Σ_norm,
        dF_dθ = builders.dF_dθ_norm,
        dΣ_dθ = builders.dΣ_dθ_norm,
        div_dF_dθ = builders.div_dF_dθ_norm,
        divM = builders.divM_norm,
        divdivM = builders.divdivM_norm,
        θ = copy(θ_ref),
        mode = :general,
        xeltype = Float64,
    )
    est_gauss = PC.build_gaussian_estimator(Xn, model, θ_ref; Δt = Δt_eff, Tmax = spec.Tmax, mean_center = true, A_of_x = A_of_x)
    Random.seed!(spec.seed)
    est_nn, _ = PC.build_neural_estimator(Xn, model, θ_ref, nn_cfg; Δt = Δt_eff, Tmax = spec.Tmax, mean_center = true, A_of_x = A_of_x)
    simulator_norm = make_simulator_norm(μr, Σr)
    est_fd = PC.build_finite_diff_estimator(simulator_norm, θ_ref, A_of_x)
    return (
        X = X_ref,
        μ = μr,
        Σ = Σr,
        A = A_of_x,
        labels = labels,
        ests = Dict(:gaussian => est_gauss, :neural => est_nn, :finite_diff => est_fd),
    )
end

println("Building estimators for θ_true ...")
res_true = build_estimators_for(θ_true, simulator_obs, make_simulator_norm, spec_obs, nn_cfg, Δt_eff)
println("Building estimators for θ_init ...")
res_init = build_estimators_for(θ_init, simulator_obs, make_simulator_norm, spec_obs, nn_cfg, Δt_eff)

function plot_responses(res; Δt_eff::Float64, title::String)
    methods = [sym for sym in (:gaussian, :neural) if haskey(res.ests, sym)]
    C = Dict(sym => res.ests[sym].responses for sym in methods)
    isempty(C) && return nothing
    m = size(first(values(C)), 1)
    P = size(first(values(C)), 2)
    ts = (0:size(first(values(C)), 3)-1) .* Δt_eff
    fig = Figure(resolution = (320 * P, 240 * m), fontsize = 12)
    colors = Dict(:gaussian => :seagreen, :neural => :indianred)
    fig[0, :] = Label(fig, title; fontsize = 16, font = :bold)
    grid = fig[1, 1] = GridLayout()
    for i in 1:m, j in 1:P
        ax = Axis(grid[i, j], xlabel = j == P ? "t" : "", ylabel = i == 1 ? "" : "", title = "A$(i) vs θ$(j)")
        for sym in methods
            lines!(ax, ts, vec(@view C[sym][i, j, :]); color = colors[sym], linewidth = 2.0, label = String(sym))
        end
        if i == 1 && j == P
            axislegend(ax, position = :rt, framevisible = false)
        end
    end
    return fig
end

fig_true = plot_responses(res_true; Δt_eff = Δt_eff, title = "Responses for θ_true")
fig_init = plot_responses(res_init; Δt_eff = Δt_eff, title = "Responses for θ_init")

if fig_true !== nothing
    path_true = joinpath(figdir, "responses_theta_true_triad.png")
    save(path_true, fig_true)
    println("Saved response figure for θ_true to: " * path_true)
end
if fig_init !== nothing
    path_init = joinpath(figdir, "responses_theta_init_triad.png")
    save(path_init, fig_init)
    println("Saved response figure for θ_init to: " * path_init)
end

function print_jacobians(res, tag::String)
    println("
Parameter Jacobians S for " * tag * " (rows=observables, cols=parameters)")
    for (sym, est) in res.ests
        S = est.S
        S === nothing && continue
        println("-- Method: $(sym), size=$(size(S))")
        for i in 1:size(S, 1)
            @printf("S[%2d]:", i)
            for j in 1:size(S, 2)
                @printf(" % .4e", S[i, j])
            end
            print('
')
        end
    end
end

print_jacobians(res_true, "θ_true")
print_jacobians(res_init, "θ_init")

println("Done.")
