#!/usr/bin/env julia
"""
triad_compute.jl

Computation pipeline for the three-dimensional triad calibration example.
Simulates the reference trajectory, builds GFDT-based estimators, runs the
calibration loop for the requested methods, and stores all intermediate and
final products required for subsequent plotting.

Run from the project root with:
    julia --project=examples examples/triad_compute.jl

After completion, execute `triad_plot.jl` to generate figures.
"""

using Pkg
Pkg.activate(@__DIR__)
try
    Pkg.develop(path=joinpath(@__DIR__, ".."))
catch err
    @warn "Development path add failed (likely already added)" exception=err
end

using LinearAlgebra, Statistics, Printf, Random, ProgressMeter, Dates
using HDF5
using TOML

using ParameterCalibration
const PC = ParameterCalibration

const RESULT_DIR = joinpath(@__DIR__, "data")
const RESULT_FILE = joinpath(RESULT_DIR, "triad_results.h5")
const S_TXT_FILE = joinpath(RESULT_DIR, "S_matrices_triad.txt")

mkpath(RESULT_DIR)

function _write_matrix_plain(io::IO, M::AbstractMatrix{<:Real})
    nr, nc = size(M)
    for i in 1:nr
        for j in 1:nc
            Printf.@printf(io, "% .6e", float(M[i, j]))
            j < nc && print(io, ' ')
        end
        print(io, '
')
    end
end

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
        # Observable set probing mixed moments of the triad state
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

function run_pipeline(; verbose = true, save_path = RESULT_FILE)
    verbose && println("Loading configuration ...")
    cfg_path = joinpath(@__DIR__, "..", "config", "config_triad.toml")
    nn_cfg, sim_cfg = PC.load_config(cfg_path)
    extra = PC.load_extra_config(cfg_path)
    model_cfg = load_triad_model(cfg_path)

    θ_true = Float64[model_cfg.d_u, model_cfg.w_u, model_cfg.d_t, model_cfg.sigma1, model_cfg.sigma2, model_cfg.sigma3]
    θ_init = θ_true .* extra.calibration.θ_init_multipliers

    simulator_obs, make_simulator_norm = make_simulators(sim_cfg)

    verbose && println("Simulating observed trajectory ...")
    Xobs = simulator_obs(θ_true)
    μ = vec(mean(Xobs, dims = 2))
    Σ = vec(std(Xobs, dims = 2))
    X = (Xobs .- μ) ./ Σ

    make_A_of_x = (μ_vec, Σ_vec) -> build_triad_A_of_x(μ_vec, Σ_vec)
    A_of_x, obs_labels = make_A_of_x(μ, Σ)
    A_target = PC.stats_A(X, A_of_x)

    builders = make_builders(μ, Σ)
    θ0 = copy(θ_true)
    base_model = PC.GFDTModel(
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
        θ = θ0,
        mode = :general,
        xeltype = Float64,
    )

    Δt_eff = sim_cfg.dt * sim_cfg.resolution * sim_cfg.Δt_multiplier
    Tmax = sim_cfg.Tmax

    verbose && println("Building GFDT estimators (gaussian, neural, finite differences) ...")
    est_gauss = PC.build_gaussian_estimator(X, base_model, θ0; Δt = Δt_eff, Tmax = Tmax, mean_center = true, A_of_x = A_of_x)
    Random.seed!(sim_cfg.seed)
    est_nn, nn_state = PC.build_neural_estimator(X, base_model, θ0, nn_cfg; Δt = Δt_eff, Tmax = Tmax, mean_center = true, A_of_x = A_of_x)
    simulator_norm = make_simulator_norm(μ, Σ)
    est_fd = PC.build_finite_diff_estimator(simulator_norm, θ0, A_of_x)

    estimators = Dict(:gaussian => est_gauss, :neural => est_nn)
    fd_estimator = est_fd

    xs = collect(range(extra.plots.xs_min, extra.plots.xs_max; length = extra.plots.xs_len))
    L = length(xs)
    x_slice_norm = zeros(Float64, 3, L)
    x_slice_phys = zeros(Float64, 3, L)
    for (k, ξ) in enumerate(xs)
        x_slice_norm[:, k] .= (ξ, 0.0, 0.0)
        x_slice_phys[1, k] = μ[1] + Σ[1] * ξ
        x_slice_phys[2, k] = μ[2]
        x_slice_phys[3, k] = μ[3]
    end

    score_data = Dict{Symbol,Dict{Symbol,Any}}()
    conj_data = Dict{Symbol,Array{Float64,2}}()
    jac_data = Dict{Symbol,Array{Float64,2}}()
    responses_data = Dict{Symbol,Array{Float64,3}}()

    for (sym, est) in estimators
        s_vals = zeros(Float64, 3, L)
        J_vals = zeros(Float64, 3, 3, L)
        B_vals = zeros(Float64, length(θ0), L)
        for k in 1:L
            x = @view x_slice_norm[:, k]
            s_vals[:, k] .= est.model_view.s(Vector{Float64}(x))
            J_vals[:, :, k] .= est.model_view.Js(Vector{Float64}(x))
            B_vals[:, k] .= PC.B_gfdt(est.model_view, Vector{Float64}(x))
        end
        score_data[sym] = Dict(:s => s_vals, :J => J_vals)
        conj_data[sym] = B_vals
        responses_data[sym] = est.responses
        jac_data[sym] = est.S
    end
    jac_data[:finite_diff] = fd_estimator.S

    all_methods = Dict(
        :gaussian => PC.GFDTGaussianScore(),
        :neural => PC.GFDTNeuralScore(),
        :finite_diff => PC.FiniteDifference(),
    )
    methods = [(sym => all_methods[sym]) for sym in extra.calibration.methods if haskey(all_methods, sym)]

    results = Dict{Symbol,Any}()
    verbose && println("Running calibration loops ...")
    @showprogress for (sym, method) in methods
        res = PC.calibration_loop(method, A_target;
            simulator_obs = simulator_obs,
            make_builders = make_builders,
            make_A_of_x = make_A_of_x,
            θ0 = θ_init,
            Δt = Δt_eff,
            Tmax = Tmax,
            nn_cfg = sym == :neural ? nn_cfg : nothing,
            make_analytic_score = nothing,
            make_simulator_norm = sym == :finite_diff ? make_simulator_norm : nothing,
            maxiters = extra.calibration.maxiters,
            tol_θ = extra.calibration.tol_θ,
            damping = extra.calibration.damping,
            free_idx = extra.calibration.free_idx,
            line_search = get(extra.calibration, :line_search, false),
            line_search_max = get(extra.calibration, :line_search_max, 4),
            callback = get(extra.calibration, :callback, false),
            callback_dt = get(extra.calibration, :callback_dt, 0.0),
            callback_Nsteps = get(extra.calibration, :callback_Nsteps, 0),
            lb = sim_cfg.lb, gb = sim_cfg.gb)
        results[sym] = res
    end

    verbose && println("Writing S matrices to text file: " * S_TXT_FILE)
    open(S_TXT_FILE, "w") do io
        println(io, "GFDT calibration sensitivity matrices S per iteration")
        println(io, "File generated on: $(Dates.format(Dates.now(), DateFormat("yyyy-mm-dd HH:MM:SS")))")
        println(io)
        for (sym, res) in results
            println(io, repeat("=", 8), " Method: ", String(sym), " ", repeat("=", 8))
            for (it, S) in enumerate(res.S_list)
                _write_matrix_plain(io, S)
                println(io)
            end
            println(io)
        end
    end

    verbose && println("Saving results to: " * save_path)
    h5open(save_path, "w") do h
        h["meta/θ_true"] = θ_true
        h["meta/θ_init"] = θ_init
        h["meta/Δt_eff"] = Δt_eff
        h["meta/Tmax"] = Tmax
        h["meta/obs_labels"] = collect(obs_labels)
        h["meta/pnames"] = ["d_u", "w_u", "d_t", "sigma1", "sigma2", "sigma3"]
        h["meta/state_labels"] = ["u1", "u2", "u3"]
        h["meta/methods"] = string.(collect(keys(estimators)))
        h["meta/calibration_methods"] = string.(collect(keys(results)))
        h["slice/x_norm"] = x_slice_norm
        h["slice/x_phys"] = x_slice_phys
        h["slice/x_axis"] = xs
        h["data/μ"] = μ
        h["data/Σ"] = Σ
        h["data/Xobs"] = Xobs
        h["data/A_target"] = A_target

        score_group = create_group(h, "score")
        for (sym, data) in score_data
            grp = create_group(score_group, String(sym))
            grp["s"] = data[:s]
            grp["J"] = data[:J]
        end

        conj_group = create_group(h, "conjugate")
        for (sym, Bvals) in conj_data
            conj_group[String(sym)] = Bvals
        end

        resp_group = create_group(h, "responses")
        ts = nothing
        for (sym, C) in responses_data
            grp = create_group(resp_group, String(sym))
            grp["C"] = C
            ts === nothing && (ts = (0:size(C, 3)-1) .* Δt_eff)
        end
        ts === nothing && (ts = Float64[])
        resp_group["ts"] = collect(ts)

        jac_group = create_group(h, "jacobian")
        for (sym, S) in jac_data
            jac_group[String(sym)] = S
        end

        calib_group = create_group(h, "calibration")
        for (sym, res) in results
            grp = create_group(calib_group, String(sym))
            grp["θ_iters"] = hcat(res.θ_path...)
            grp["G_iters"] = reduce(hcat, res.G_list)
        end
    end

    verbose && println("Done.")
    return save_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline()
end
