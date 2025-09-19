#!/usr/bin/env julia
#=
reduced_1d_plot.jl

Plotting & visualization script for reduced 1D GFDT calibration example.
Consumes the HDF5 data produced by `reduced_1d_compute.jl` and generates
publication-quality figures using GLMakie.

Usage (from repo root):
    julia --project=examples examples/reduced_1d_plot.jl

Figures are saved in `figures/`.
=#
using Pkg
Pkg.activate(@__DIR__)

using GLMakie
using LinearAlgebra, Statistics, Printf
using HDF5

const DATA_FILE = joinpath(@__DIR__, "data", "reduced_1d_results.h5")
const FIG_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIG_DIR)

# Color & style palette ------------------------------------------------------------------
const COL_ANALYTIC = :crimson
const COL_GAUSS    = :black
const COL_NEURAL   = :steelblue
const COL_FD       = :darkorange

const FONT_SIZE = 22
set_theme!(Theme(fontsize=FONT_SIZE, Axis=(xlabelsize=FONT_SIZE, ylabelsize=FONT_SIZE, titlealign=:left, titlesize=FONT_SIZE+2, xgridvisible=false, ygridvisible=false)))

# Utility to put nice framed axis styling ------------------------------------------------
function styled_axis!(ax::Axis; xlabel="", ylabel="", title="")
    ax.xlabel = xlabel
    ax.ylabel = ylabel
    ax.title = title
    ax.bottomspinevisible = true
    ax.leftspinevisible = true
    ax.rightspinevisible = false
    ax.topspinevisible = false
    return ax
end

function load_data(path::AbstractString)
    @assert isfile(path) "Data file not found: $(path). Run reduced_1d_compute.jl first."
    h5open(path, "r") do h
        meta = Dict(
            :θ_true => read(h, "meta/θ_true"),
            :θ_init => read(h, "meta/θ_init"),
            :Δt_eff => read(h, "meta/Δt_eff"),
            :Tmax   => read(h, "meta/Tmax"),
            :xs     => read(h, "meta/xs"),
            :obs_labels => read(h, "meta/obs_labels"),
            :pnames => read(h, "meta/pnames"),
            :thresholds => begin
                if haskey(h, "meta/threshold_values") && haskey(h, "meta/threshold_names")
                    names = read(h, "meta/threshold_names")
                    vals = read(h, "meta/threshold_values")
                    Dict(Symbol(n)=>vals[i] for (i,n) in enumerate(names))
                elseif haskey(h, "meta/thresholds")
                    # Backward compatibility if stored as raw vector
                    read(h, "meta/thresholds")
                else
                    nothing
                end
            end,
        )
        data = Dict(
            :μ => read(h, "data/μ"),
            :Σ => read(h, "data/Σ"),
            :Xobs => read(h, "data/Xobs"),
            :A_target => read(h, "data/A_target"),
        )
        score = Dict(
            :s_true => read(h, "score/s_true"),
            :s_gauss => read(h, "score/s_gauss"),
            :s_nn => read(h, "score/s_nn"),
            :J_true => read(h, "score/J_true"),
            :J_gauss => read(h, "score/J_gauss"),
            :J_nn => read(h, "score/J_nn"),
        )
        conjugate = Dict(
            :B_true => read(h, "conjugate/B_true"),
            :B_gauss => read(h, "conjugate/B_gauss"),
            :B_nn => read(h, "conjugate/B_nn"),
        )
        responses = Dict(
            :C_true => read(h, "responses/C_true"),
            :C_gauss => read(h, "responses/C_gauss"),
            :C_nn => read(h, "responses/C_nn"),
            :ts => read(h, "responses/ts"),
        )
        jac = Dict(
            :S_true => read(h, "jacobian/S_true"),
            :S_gauss => read(h, "jacobian/S_gauss"),
            :S_nn => read(h, "jacobian/S_nn"),
        )
        # Calibration groups dynamic
        calib = Dict{Symbol,Dict}()
        for grp_name in keys(h["calibration"]) |> collect
            gpath = joinpath("calibration", grp_name)
            calib[Symbol(grp_name)] = Dict(
                :θ_iters => read(h, joinpath(gpath, "θ_iters")),
                :G_iters => read(h, joinpath(gpath, "G_iters")),
            )
        end
        return (meta=meta, data=data, score=score, conjugate=conjugate, responses=responses, jac=jac, calib=calib)
    end
end

# Figure 1: Score & Jacobian --------------------------------------------------------------
function fig_score_and_jac(meta, score)
    xs = meta[:xs]
    fig = Figure(resolution=(900, 580))
    line_objs = Makie.AbstractPlot[]
    labels = String[]

    ax1 = Axis(fig[1,1]); styled_axis!(ax1; xlabel="x", ylabel="s(x)", title="Score")
    push!(line_objs, lines!(ax1, xs, score[:s_true]; color=COL_ANALYTIC, linewidth=4)); push!(labels, "Analytical")
    push!(line_objs, lines!(ax1, xs, score[:s_gauss]; color=COL_GAUSS, linewidth=3, linestyle=:dash)); push!(labels, "Gaussian")
    push!(line_objs, lines!(ax1, xs, score[:s_nn]; color=COL_NEURAL, linewidth=3)); push!(labels, "KGMM")

    ax2 = Axis(fig[2,1]); styled_axis!(ax2; xlabel="x", ylabel="∂s/∂x", title="Jacobian")
    lines!(ax2, xs, score[:J_true]; color=COL_ANALYTIC, linewidth=4)
    lines!(ax2, xs, score[:J_gauss]; color=COL_GAUSS, linewidth=3, linestyle=:dash)
    lines!(ax2, xs, score[:J_nn]; color=COL_NEURAL, linewidth=3)

    leg = Legend(fig, line_objs, labels; framevisible=true, orientation=:horizontal, tellwidth=false)
    leg.halign = :center
    fig[3,1] = leg
    fig
end

# Figure 2: Conjugate variables B(x) ------------------------------------------------------
function fig_conjugate(meta, conjugate)
    xs = meta[:xs]; pnames = meta[:pnames]; P = size(conjugate[:B_true], 1)
    fig = Figure(resolution=(900, 260*P + 60))
    line_objs = Makie.AbstractPlot[]; labels = String[]
    for j in 1:P
        ax = Axis(fig[j,1]); styled_axis!(ax; xlabel="x (normalized)", ylabel="B(x)", title=pnames[j])
        if j == 1
            push!(line_objs, lines!(ax, xs, conjugate[:B_true][j,:]; color=COL_ANALYTIC, linewidth=4)); push!(labels, "Analytical")
            push!(line_objs, lines!(ax, xs, conjugate[:B_gauss][j,:]; color=COL_GAUSS, linewidth=3, linestyle=:dash)); push!(labels, "Gaussian")
            push!(line_objs, lines!(ax, xs, conjugate[:B_nn][j,:]; color=COL_NEURAL, linewidth=3)); push!(labels, "KGMM")
        else
            lines!(ax, xs, conjugate[:B_true][j,:]; color=COL_ANALYTIC, linewidth=4)
            lines!(ax, xs, conjugate[:B_gauss][j,:]; color=COL_GAUSS, linewidth=3, linestyle=:dash)
            lines!(ax, xs, conjugate[:B_nn][j,:]; color=COL_NEURAL, linewidth=3)
        end
    end
    leg = Legend(fig, line_objs, labels; framevisible=true, orientation=:horizontal, tellwidth=false)
    leg.halign = :center
    fig[P+1,1] = leg
    fig
end

# Figure 3: Response functions ------------------------------------------------------------
function fig_responses(meta, responses)
    ts = responses[:ts]; C_true = responses[:C_true]; C_gauss = responses[:C_gauss]; C_nn = responses[:C_nn]
    obs_labels = meta[:obs_labels]; pnames = meta[:pnames]
    m = length(obs_labels); P = length(pnames)
    fig = Figure(resolution=(1000, max(260*m, 260) + 60))
    grid = fig[1,1] = GridLayout()
    line_objs = Makie.AbstractPlot[]; labels = String[]
    for i in 1:m, j in 1:P
        ax = Axis(grid[i,j])
        styled_axis!(ax; xlabel="t", ylabel="⟨A_t B_0⟩", title="$(pnames[j]) (A=$(obs_labels[i]))")
        if i==1 && j==1
            push!(line_objs, lines!(ax, ts, vec(@view C_true[i,j,:]); color=COL_ANALYTIC, linewidth=3.5)); push!(labels, "Analytical")
            push!(line_objs, lines!(ax, ts, vec(@view C_gauss[i,j,:]); color=COL_GAUSS, linewidth=3, linestyle=:dash)); push!(labels, "Gaussian")
            push!(line_objs, lines!(ax, ts, vec(@view C_nn[i,j,:]); color=COL_NEURAL, linewidth=3)); push!(labels, "KGMM")
        else
            lines!(ax, ts, vec(@view C_true[i,j,:]); color=COL_ANALYTIC, linewidth=3.5)
            lines!(ax, ts, vec(@view C_gauss[i,j,:]); color=COL_GAUSS, linewidth=3, linestyle=:dash)
            lines!(ax, ts, vec(@view C_nn[i,j,:]); color=COL_NEURAL, linewidth=3)
        end
    end
    leg = Legend(fig, line_objs, labels; framevisible=true, orientation=:horizontal, tellwidth=false)
    leg.halign = :center
    fig[2,1] = leg
    fig
end

# Figure 4: Calibration convergence -------------------------------------------------------
function fig_calibration(meta, calib, data)
    obs_labels = meta[:obs_labels]; A_target = data[:A_target]
    m = length(obs_labels)
    fig = Figure(resolution=(520*m, 460))
    grid = fig[1,1] = GridLayout()
    mapping = Dict(:analytic=>COL_ANALYTIC, :gaussian=>COL_GAUSS, :neural=>COL_NEURAL, :finite_diff=>COL_FD)
    label_map = Dict(:analytic=>"Analytical", :gaussian=>"Gaussian", :neural=>"KGMM", :finite_diff=>"Finite diff")
    line_objs = Makie.AbstractPlot[]; labels = String[]
    for j in 1:m
        ax = Axis(grid[1,j]); styled_axis!(ax; xlabel="iteration", ylabel=obs_labels[j], title="$(obs_labels[j]) convergence")
        hlines!(ax, [A_target[j]]; color=:gray, linestyle=:dot, linewidth=2)
        for (sym, col) in mapping
            haskey(calib, sym) || continue
            its = 1:size(calib[sym][:G_iters], 2)
            Gj = calib[sym][:G_iters][j, :]
            if j==1
                push!(line_objs, lines!(ax, its, Gj; color=col, linewidth=3)); push!(labels, label_map[sym])
            else
                lines!(ax, its, Gj; color=col, linewidth=3)
            end
        end
    end
    leg = Legend(fig, line_objs, labels; framevisible=true, orientation=:horizontal, tellwidth=false)
    leg.halign = :center
    fig[2,1] = leg
    fig
end

# Figure 5: Parameter evolution -----------------------------------------------------------
function fig_parameter_trajectories(meta, calib)
    pnames = meta[:pnames]; θ_true = meta[:θ_true]
    Np = length(pnames)
    fig = Figure(resolution=(380*Np, 440))
    grid = fig[1,1] = GridLayout()
    mapping = Dict(:analytic=>COL_ANALYTIC, :gaussian=>COL_GAUSS, :neural=>COL_NEURAL, :finite_diff=>COL_FD)
    label_map = Dict(:analytic=>"Analytical", :gaussian=>"Gaussian", :neural=>"KGMM", :finite_diff=>"Finite diff")
    line_objs = Makie.AbstractPlot[]; labels = String[]
    for (j,pn) in enumerate(pnames)
        ax = Axis(grid[1,j]); styled_axis!(ax; xlabel="iteration", ylabel=pn, title=pn)
        # True parameter horizontal reference line (user asked for vertical; interpreted as horizontal since y is parameter value)
        hlines!(ax, [θ_true[j]]; color=:gray, linestyle=:dot, linewidth=2)
        for (sym, col) in mapping
            haskey(calib, sym) || continue
            θ_iters = calib[sym][:θ_iters]
            if j==1
                push!(line_objs, lines!(ax, 1:size(θ_iters,2), θ_iters[j,:]; color=col, linewidth=3)); push!(labels, label_map[sym])
            else
                lines!(ax, 1:size(θ_iters,2), θ_iters[j,:]; color=col, linewidth=3)
            end
        end
    end
    leg = Legend(fig, line_objs, labels; framevisible=true, orientation=:horizontal, tellwidth=false)
    leg.halign = :center
    fig[2,1] = leg
    fig
end

# Figure 6 (new): Merged parameter trajectories (perturbed only) + observable convergence -------
function fig_calibration_and_parameters(meta, calib, data; tol=1e-12)
    pnames = meta[:pnames]; θ_true = meta[:θ_true]; θ_init = meta[:θ_init]
    obs_labels = meta[:obs_labels]; A_target = data[:A_target]
    # Identify perturbed parameters (initial value differs from true) within tolerance
    perturbed = findall(j -> !(isapprox(θ_true[j], θ_init[j]; atol=tol, rtol=0.0)), eachindex(θ_true))
    if isempty(perturbed)
        perturbed = collect(eachindex(θ_true))  # fallback: show all
    end
    npert = length(perturbed); m = length(obs_labels)
    # Layout: two stacked GridLayouts so column counts may differ
    fig = Figure(resolution=(max(420*npert, 420*m), 480 + 480))
    top = fig[1,1] = GridLayout()
    bottom = fig[2,1] = GridLayout()
    mapping = Dict(:analytic=>COL_ANALYTIC, :gaussian=>COL_GAUSS, :neural=>COL_NEURAL, :finite_diff=>COL_FD)
    label_map = Dict(:analytic=>"Analytical", :gaussian=>"Gaussian", :neural=>"KGMM", :finite_diff=>"Finite diff")
    line_objs = Makie.AbstractPlot[]; labels = String[]
    # Row 1: parameter trajectories (perturbed only)
    for (colidx, j) in enumerate(perturbed)
        pn = pnames[j]
    ax = Axis(top[1, colidx]); styled_axis!(ax; xlabel="iteration", ylabel=pn, title=pn * " convergence")
        hlines!(ax, [θ_true[j]]; color=:gray, linestyle=:dot, linewidth=2)
        # Use a fixed ordered list of methods for stable legend ordering
        for sym in (:analytic, :gaussian, :neural, :finite_diff)
            col = get(mapping, sym, nothing); isnothing(col) && continue
            haskey(calib, sym) || continue
            θ_iters = calib[sym][:θ_iters]
            if colidx == 1
                push!(line_objs, lines!(ax, 1:size(θ_iters,2), θ_iters[j,:]; color=col, linewidth=3)); push!(labels, label_map[sym])
            else
                lines!(ax, 1:size(θ_iters,2), θ_iters[j,:]; color=col, linewidth=3)
            end
        end
    end
    # Row 2: observable convergence
    for j in 1:m
        ax = Axis(bottom[1,j]); styled_axis!(ax; xlabel="iteration", ylabel=obs_labels[j], title="$(obs_labels[j]) convergence")
        hlines!(ax, [A_target[j]]; color=:gray, linestyle=:dot, linewidth=2)
        for sym in (:analytic, :gaussian, :neural, :finite_diff)
            col = get(mapping, sym, nothing); isnothing(col) && continue
            haskey(calib, sym) || continue
            its = 1:size(calib[sym][:G_iters], 2)
            Gj = calib[sym][:G_iters][j, :]
            # Add lines; legend already captured from parameter row
            lines!(ax, its, Gj; color=col, linewidth=3)
        end
    end
    # Unified legend at bottom (row 3)
    leg = Legend(fig, line_objs, labels; framevisible=true, orientation=:horizontal, tellwidth=false)
    leg.halign = :center
    fig[3,1] = leg
    fig
end

function save_figure(fig::Figure, name::AbstractString)
    path = joinpath(FIG_DIR, name)
    save(path, fig; px_per_unit=2)
    @info "Saved figure" path
end

function main()
    bundle = load_data(DATA_FILE)
    meta, data, score, conjugate, responses, jac, calib = bundle
    save_figure(fig_score_and_jac(meta, score), "score_jacobian_reduced1d.png")
    save_figure(fig_conjugate(meta, conjugate), "conjugate_B_reduced1d.png")
    save_figure(fig_responses(meta, responses), "response_functions_reduced1d.png")
    # Replaced separate calibration & parameter trajectory figures with merged figure
    save_figure(fig_calibration_and_parameters(meta, calib, data), "calibration_parameters_reduced1d.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
