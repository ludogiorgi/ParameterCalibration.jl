#!/usr/bin/env julia
"""
reduced_1d_compute.jl

Heavy-lift computation script for the reduced 1D GFDT calibration example.

Responsibility:
  * Activate the local examples environment (separate from core package deps)
  * Run simulations, build estimators (analytic, gaussian, neural, finite diff)
  * Perform calibration loops for each method
  * Save all intermediate & final numerical objects required for plotting to an HDF5 file

The complementary script `reduced_1d_plot.jl` consumes the HDF5 file and generates
high-quality publication-ready figures using GLMakie.

Usage (from repo root):
  julia --project=examples examples/reduced_1d_compute.jl

After completion, run the plotting script:
  julia --project=examples examples/reduced_1d_plot.jl

Outputs are stored under `figures/` (created if missing) and numerical data in
`examples/data/reduced_1d_results.h5`.
"""

using Pkg
Pkg.activate(@__DIR__)
# Ensure we use local development version of the package (only needed first time)
try
    Pkg.develop(path=joinpath(@__DIR__, ".."))
catch err
    @warn "Development path add failed (likely already added)" exception=err
end

using LinearAlgebra, Statistics, Printf, Random, ProgressMeter, Dates
using HDF5
using ParameterCalibration
const PC = ParameterCalibration

# --------------------------------------------------------------------------------------
# Configuration & Parameters
# --------------------------------------------------------------------------------------
const PHYS_PARAMS = nothing  # populated from config below
const RESULT_DIR = joinpath(@__DIR__, "data")
const RESULT_FILE = joinpath(RESULT_DIR, "reduced_1d_results.h5")
const S_TXT_FILE = joinpath(RESULT_DIR, "S_matrices.txt")

mkpath(RESULT_DIR)

# -----------------------------
# Helpers
# -----------------------------
function _write_matrix_plain(io::IO, M::AbstractMatrix{<:Real})
    nr, nc = size(M)
    for i in 1:nr
        for j in 1:nc
            # Note: @printf requires a literal format string
            Printf.@printf(io, "% .6e", float(M[i,j]))
            j < nc && print(io, ' ')
        end
        print(io, '\n')
    end
end

# Builders for normalized model components ------------------------------------------------
function make_builders(μ::AbstractVector, Σ::AbstractVector)
    d = 1
    S1 = Float64(Σ[1]); M1 = Float64(μ[1])
    F_norm = (x, θ) -> begin
        xphys = x[1]*S1 + M1
        @inbounds return [(θ[1] + θ[2]*xphys + θ[3]*xphys^2 - θ[4]*xphys^3) / S1]
    end
    Σ_norm = (_x, θ) -> reshape([θ[5]/S1], d, d)
    dF_dθ_norm = (x, θ) -> begin
        xphys = x[1]*S1 + M1
        reshape([1/S1, xphys/S1, (xphys^2)/S1, (-xphys^3)/S1, 0.0], d, :)
    end
    dΣ_dθ_norm = (_x, θ) -> begin
        A = zeros(Float64, d, d, length(θ)); A[1,1,5] = 1/S1; A
    end
    div_dF_dθ_norm = (x, _θ) -> begin
        xphys = x[1]*S1 + M1
        [0.0, 1.0, 2.0*xphys, -3.0*xphys^2, 0.0]
    end
    divM_norm    = (_x, θ) -> zeros(Float64, d, length(θ))
    divdivM_norm = (_x, θ) -> zeros(Float64, length(θ))
    return (F_norm=F_norm, Σ_norm=Σ_norm, dF_dθ_norm=dF_dθ_norm, dΣ_dθ_norm=dΣ_dθ_norm,
            div_dF_dθ_norm=div_dF_dθ_norm, divM_norm=divM_norm, divdivM_norm=divdivM_norm)
end

# Analytic score builder ------------------------------------------------------------------
function make_analytic_score(μ::AbstractVector, Σ::AbstractVector)
    μ1 = μ[1]; Σ1 = Σ[1]
    return function (θ::AbstractVector)
        F̃, a, b, c, s = θ
        s_fn = function (x)
            u = (x[1]*Σ1 + μ1)
            f = F̃ + a*u + b*u^2 - c*u^3
            return [Σ1 * 2 * f / (s^2)]
        end
        Js_fn = function (x)
            u = (x[1]*Σ1 + μ1)
            fp = a + 2*b*u - 3*c*u^2
            return reshape([Σ1^2 * 2 * fp / (s^2)], 1, 1)
        end
        return (s=s_fn, Js=Js_fn)
    end
end

# Simulator (returns trajectories) ----------------------------------
function make_simulators(spec::PC.SimSpec)
    function simulator_obs(θ::AbstractVector)
        drift_θ! = (du,u,t) -> (du[1] = θ[1] + θ[2]*u[1] + θ[3]*u[1]^2 - θ[4]*u[1]^3)
        sigma_θ! = (du,u,t) -> (du[1] = θ[5])
        Xp = PC.simulate(spec.u0, drift_θ!, sigma_θ!; spec=spec)
        return Xp
    end
    function make_simulator_norm(μ::AbstractVector, Σ::AbstractVector)
        function norm_simulator(θ::AbstractVector)
            S1 = Float64(Σ[1]); M1 = Float64(μ[1])
            drift_θ! = (du,u,t) -> begin
                uphys = u[1]
                f = θ[1] + θ[2]*uphys + θ[3]*uphys^2 - θ[4]*uphys^3
                du[1] = f
            end
            sigma_θ! = (du,u,t) -> (du[1] = θ[5])
            Xp = PC.simulate(spec.u0, drift_θ!, sigma_θ!; spec=spec)
            return (Xp .- M1) ./ S1
        end
        return norm_simulator
    end
    return simulator_obs, make_simulator_norm
end

# Main pipeline ----------------------------------------------------------------------------
function run_pipeline(; verbose=true, save_path=RESULT_FILE)
    verbose && println("Loading configuration ...")
    cfg_path = joinpath(@__DIR__, "..", "config", "config_reduced_1d.toml")
    nn_cfg, sim_cfg = PC.load_config(cfg_path)
    extra = PC.load_extra_config(cfg_path)

    θ_true = Float64[extra.model.F_tilde, extra.model.a, extra.model.b, extra.model.c, extra.model.s]
    verbose && println("Simulating observed data with true parameters ...")
    simulator_obs, make_simulator_norm = make_simulators(sim_cfg)
    Xobs = simulator_obs(θ_true)
    μ = vec(mean(Xobs, dims=2)); Σ = vec(std(Xobs, dims=2))
    X = (Xobs .- μ) ./ Σ

    use_moments    = Tuple(extra.observables.use_moments)
    use_indicators = Tuple(extra.observables.use_indicators)
    thresholds     = extra.observables.thresholds
    make_A_of_x = PC.build_make_A_of_x(; use_moments=use_moments, use_indicators=use_indicators, thresholds=thresholds)
    A_of_x, obs_labels = make_A_of_x(μ[1], Σ[1])
    # IMPORTANT: A_of_x is defined on normalized coordinates; use X (normalized)
    A_target = PC.stats_A(X, A_of_x)
    builders = make_builders(μ, Σ)
    θ0 = copy(θ_true)
    base_model = PC.GFDTModel(
        s=x->zeros(Float64,1), divs=x->0.0, Js=x->zeros(Float64,1,1),
        F=builders.F_norm, Σ=builders.Σ_norm, dF_dθ=builders.dF_dθ_norm, dΣ_dθ=builders.dΣ_dθ_norm,
        div_dF_dθ=builders.div_dF_dθ_norm, divM=builders.divM_norm, divdivM=builders.divdivM_norm,
        θ=θ0, mode=:general, xeltype=Float64)

    Δt_eff = sim_cfg.dt * sim_cfg.resolution * sim_cfg.Δt_multiplier
    Tmax   = sim_cfg.Tmax
    analytic_builder = make_analytic_score(μ, Σ)

    verbose && println("Building estimators ...")
    est_true  = PC.build_analytic_estimator(X, base_model, θ0; Δt=Δt_eff, Tmax=Tmax, mean_center=true, analytic_builder=analytic_builder, A_of_x=A_of_x)
    est_gauss = PC.build_gaussian_estimator(X, base_model, θ0; Δt=Δt_eff, Tmax=Tmax, mean_center=true, A_of_x=A_of_x)
    Random.seed!(sim_cfg.seed)
    est_nn, nn_pre = PC.build_neural_estimator(X, base_model, θ0, nn_cfg; Δt=Δt_eff, Tmax=Tmax, mean_center=true, A_of_x=A_of_x)
    simulator_norm = make_simulator_norm(μ, Σ)
    est_fd    = PC.build_finite_diff_estimator(simulator_norm, θ0, A_of_x)

    # Score/Jacobian sampling grid
    xs = collect(range(extra.plots.xs_min, extra.plots.xs_max; length=extra.plots.xs_len))
    P = length(θ0)

    # Conjugate variables B(x)
    verbose && println("Computing conjugate variables B(x) and responses ...")
    B_true  = zeros(Float64, P, length(xs))
    B_gauss = similar(B_true); B_nn = similar(B_true)
    for (k, ξ) in enumerate(xs)
        x = (ξ,)
        @views B_true[:,k]  .= PC.B_gfdt(est_true.model_view,  collect(x))
        @views B_gauss[:,k] .= PC.B_gfdt(est_gauss.model_view, collect(x))
        @views B_nn[:,k]    .= PC.B_gfdt(est_nn.model_view,    collect(x))
    end

    # Response functions C_{A,B}(t)
    C_true  = est_true.responses; C_gauss = est_gauss.responses; C_nn = est_nn.responses
    Kmax = size(C_true, 3) - 1; ts = (0:Kmax) .* Δt_eff

    # Parameter Jacobians
    S_true_all  = est_true.S;  S_gauss_all = est_gauss.S; S_nn_all = est_nn.S

    # Calibration loop setup
    verbose && println("Running calibration loops ...")
    all_methods = Dict(
        :analytic => PC.GFDTAnalyticScore(),
        :gaussian => PC.GFDTGaussianScore(),
        :neural   => PC.GFDTNeuralScore(),
        :finite_diff => PC.FiniteDifference(),
    )
    methods = [(sym => all_methods[sym]) for sym in extra.calibration.methods if haskey(all_methods, sym)]
    θ_init = θ0 .* extra.calibration.θ_init_multipliers
    results = Dict{Symbol,Any}()
    @showprogress for (sym, method) in methods
        res = PC.calibration_loop(method, A_target;
            simulator_obs, make_builders, make_A_of_x,
            θ0=θ_init,
            Δt=Δt_eff, Tmax=Tmax,
            nn_cfg = sym == :neural ? nn_cfg : nothing,
            make_analytic_score = sym == :analytic ? make_analytic_score : nothing,
            make_simulator_norm = sym == :finite_diff ? make_simulator_norm : nothing,
            maxiters=extra.calibration.maxiters,
            tol_θ=extra.calibration.tol_θ,
            damping=extra.calibration.damping,
            mean_center=extra.calibration.mean_center,
            free_idx=extra.calibration.free_idx)
        results[sym] = res
    end

    # Write S matrices per iteration to a human-readable text file
    verbose && println("Writing S matrices (per iteration) to: " * S_TXT_FILE)
    open(S_TXT_FILE, "w") do io
        println(io, "GFDT calibration sensitivity matrices S per iteration")
        println(io, "File generated on: $(Dates.format(Dates.now(), DateFormat("yyyy-mm-dd HH:MM:SS")))")
        println(io)
        for (sym, res) in results
            println(io, repeat("=", 8), " Method: ", String(sym), " ", repeat("=", 8))
            for (it, S) in enumerate(res.S_list)
                println(io, "Iteration ", it, ": S size = ", size(S))
                _write_matrix_plain(io, S)
                println(io)
            end
            println(io)
        end
    end

    # Serialize everything to HDF5
    verbose && println("Saving results to: $save_path")
    h5open(save_path, "w") do h
        h["meta/θ_true"] = θ_true
        h["meta/θ_init"] = θ_init
        h["meta/Δt_eff"] = Δt_eff
        h["meta/Tmax"] = Tmax
        h["meta/xs"] = xs
        h["meta/obs_labels"] = collect(obs_labels)
        h["meta/pnames"] = ["F_tilde","a","b","c","s"]
        # Thresholds: store as separate name/value arrays for HDF5 compatibility
        try
            tnames = collect(keys(thresholds))
            tvals = [getproperty(thresholds, k) for k in tnames]
            h["meta/threshold_names"] = string.(tnames)
            h["meta/threshold_values"] = collect(Float64.(tvals))
        catch _
            # Fallback: skip thresholds if not serializable
        end
        h["data/μ"] = μ; h["data/Σ"] = Σ
        h["data/Xobs"] = Xobs
        h["data/A_target"] = A_target
        # Score and Jacobian sampled values
        s_true_vals  = [est_true.model_view.s([ξ])[1]  for ξ in xs]
        s_gauss_vals = [est_gauss.model_view.s([ξ])[1] for ξ in xs]
        s_nn_vals    = [est_nn.model_view.s([ξ])[1]    for ξ in xs]
        J_true_vals  = [est_true.model_view.Js([ξ])[1,1]  for ξ in xs]
        J_gauss_vals = [est_gauss.model_view.Js([ξ])[1,1] for ξ in xs]
        J_nn_vals    = [est_nn.model_view.Js([ξ])[1,1]    for ξ in xs]
        h["score/s_true"] = s_true_vals
        h["score/s_gauss"] = s_gauss_vals
        h["score/s_nn"] = s_nn_vals
        h["score/J_true"] = J_true_vals
        h["score/J_gauss"] = J_gauss_vals
        h["score/J_nn"] = J_nn_vals
        # Conjugate variables
        h["conjugate/B_true"] = B_true
        h["conjugate/B_gauss"] = B_gauss
        h["conjugate/B_nn"] = B_nn
        # Response functions
        h["responses/C_true"] = C_true
        h["responses/C_gauss"] = C_gauss
        h["responses/C_nn"] = C_nn
    h["responses/ts"] = collect(ts)
        # Jacobians S
        h["jacobian/S_true"] = S_true_all
        h["jacobian/S_gauss"] = S_gauss_all
        h["jacobian/S_nn"] = S_nn_all
        # Calibration convergence per method
        for (sym, res) in results
            grp = create_group(h, "calibration/$(Symbol(sym))")
            θ_iters = hcat(res.θ_path...)  # parameters over iterations
            grp["θ_iters"] = θ_iters
            # Observables trajectory per iteration
            G_mat = reduce(hcat, res.G_list)
            grp["G_iters"] = G_mat
        end
    end

    verbose && println("Done.")
    return save_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline()
end
