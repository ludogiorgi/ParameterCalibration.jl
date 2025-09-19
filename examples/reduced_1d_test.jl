#!/usr/bin/env julia
#=
reduced_1d_test.jl

Compare stationary / long-time empirical PDFs of the reduced 1D system
for (1) true parameters and (2) initial perturbed parameters from the
example configuration. Produces a GLMakie plot overlaying both PDFs.

Run (from repo root):
    julia --project=examples examples/reduced_1d_test.jl

Output:
    figures/pdf_comparison_reduced1d.png
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

function reduced1d_observable_factory(; use_moments=(:m1, :m2), use_indicators=(), thresholds::NamedTuple = (;))
    moms = use_moments isa Tuple ? use_moments : (use_moments,)
    inds = use_indicators isa Tuple ? use_indicators : (use_indicators,)
    α = hasproperty(thresholds, :α) ? getfield(thresholds, :α) : nothing
    β = hasproperty(thresholds, :β) ? getfield(thresholds, :β) : nothing
    return function (μ_phys::Real, Σ_phys::Real)
        fns = Vector{Function}()
        labels = String[]
        for m in moms
            if m === :m1
                push!(fns, u -> u);           push!(labels, "u")
            elseif m === :m2
                push!(fns, u -> u^2);         push!(labels, "u^2")
            elseif m === :m3
                push!(fns, u -> u^3);         push!(labels, "u^3")
            elseif m === :m4
                push!(fns, u -> u^4);         push!(labels, "u^4")
            else
                error("Unknown moment symbol \$(m)")
            end
        end
        if :ge in inds
            @assert α !== nothing "Thresholds must include :α when using :ge indicator"
            push!(fns, u -> (u ≥ α ? 1.0 : 0.0)); push!(labels, "1{u≥α}")
        end
        if :le in inds
            @assert β !== nothing "Thresholds must include :β when using :le indicator"
            push!(fns, u -> (u ≤ β ? 1.0 : 0.0)); push!(labels, "1{u≤β}")
        end
        function A_of_x(x::AbstractVector)
            u = x[1] * Σ_phys + μ_phys
            return Float64[f(u) for f in fns]
        end
        return (A_of_x, labels)
    end
end

# --------------------------------------------------------------------------------------
# Load configuration
# --------------------------------------------------------------------------------------
cfg_path = joinpath(@__DIR__, "..", "config", "config_reduced_1d.toml")
nn_cfg, spec_obs = PC.load_config(cfg_path)
extra = PC.load_extra_config(cfg_path)

θ_true = Float64[extra.model.F_tilde, extra.model.a, extra.model.b, extra.model.c, extra.model.s]
θ_init = θ_true .* extra.calibration.θ_init_multipliers

# Effective GFDT correlation parameters (not strictly needed here, but we keep consistency)
Δt_eff = spec_obs.dt * spec_obs.resolution * spec_obs.Δt_multiplier

# --------------------------------------------------------------------------------------
# Simulation utilities (physical coordinates)
# --------------------------------------------------------------------------------------
function simulate_phys(θ::AbstractVector, spec::PC.SimSpec)
    drift_θ! = (du,u,t) -> (du[1] = θ[1] + θ[2]*u[1] + θ[3]*u[1]^2 - θ[4]*u[1]^3)
    sigma_θ! = (du,u,t) -> (du[1] = θ[5])
    PC.simulate(spec.u0, drift_θ!, sigma_θ!; spec=spec)
end

# We'll produce a relatively long trajectory; user can alter Nsteps in config.
Random.seed!(spec_obs.seed)
X_true = simulate_phys(θ_true, spec_obs)
Random.seed!(spec_obs.seed)  # same seed for comparability
X_init = simulate_phys(θ_init, spec_obs)

u_true = vec(X_true[1,:])
u_init = vec(X_init[1,:])

_σ_true = std(u_true); _σ_init = std(u_init)

"""Observables (shared across comparisons): build factory from config thresholds."""
μ = vec(mean(X_true, dims=2)); Σ = vec(std(X_true, dims=2))
μ_phys = μ[1]; Σ_phys = Σ[1]
use_moments    = Tuple(extra.observables.use_moments)
use_indicators = Tuple(extra.observables.use_indicators)
# p_indicator removed in updated config; thresholds now supplied directly
thresholds_cfg = extra.observables.thresholds
make_A_of_x = reduced1d_observable_factory(; use_moments=use_moments, use_indicators=use_indicators, thresholds=thresholds_cfg)
A_of_x_tmp, obs_labels_tmp = make_A_of_x(μ_phys, Σ_phys)
const A_of_x = A_of_x_tmp
const obs_labels = obs_labels_tmp
const thresholds = thresholds_cfg

function mean_observables(u_series::AbstractVector, A_of_x::Function)
    # A_of_x expects normalized x; we currently have physical u. Convert to normalized x = (u - μ_phys)/Σ_phys
    T = length(u_series)
    acc = nothing
    @inbounds for t in 1:T
        xnorm = [(u_series[t] - μ_phys)/Σ_phys]
        Aval = A_of_x(xnorm)
        if acc === nothing
            acc = zero.(Aval)
        end
        acc .+= Aval
    end
    acc ./= max(T,1)
    return acc
end

mean_true  = mean_observables(u_true, A_of_x)
mean_init  = mean_observables(u_init, A_of_x)
println("Observables mean values:")
for (i, lab) in enumerate(obs_labels)
    @printf("  %-8s  true=% .6e   init=% .6e   diff=% .3e\n", lab, mean_true[i], mean_init[i], mean_true[i]-mean_init[i])
end

# --------------------------------------------------------------------------------------
# PDF estimation using KernelDensity.jl (shared grid & shared bandwidth)
# --------------------------------------------------------------------------------------
kd_true = kde(u_true)
kd_init = kde(u_init)

# Harmonize onto a common x-grid spanning both supports
nbins = 512
lo = min(first(kd_true.x), first(kd_init.x))
hi = max(last(kd_true.x), last(kd_init.x))
xs_plot = range(lo, hi; length=nbins)

function _lin_interp(xsrc, ysrc, xq)
    out = similar(xq, Float64)
    n = length(xsrc)
    @inbounds for (k, x) in enumerate(xq)
        if x <= xsrc[1]
            out[k] = ysrc[1]
        elseif x >= xsrc[end]
            out[k] = ysrc[end]
        else
            i = searchsortedfirst(xsrc, x)
            i = clamp(i, 2, n)
            x1 = xsrc[i-1]; x2 = xsrc[i]
            y1 = ysrc[i-1]; y2 = ysrc[i]
            t = (x - x1)/(x2 - x1)
            out[k] = (1-t)*y1 + t*y2
        end
    end
    out
end

pdf_true_plot = _lin_interp(kd_true.x, kd_true.density, xs_plot)
pdf_init_plot = _lin_interp(kd_init.x, kd_init.density, xs_plot)

# --------------------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------------------
fig = Figure(resolution=(800,500))
ax = Axis(fig[1,1], xlabel="u (physical)", ylabel="PDF", title="Reduced 1D: Stationary PDF (True vs Initial)")
lines!(ax, xs_plot, pdf_true_plot, color=:dodgerblue, linewidth=2.5, label="True θ (KDE)")
lines!(ax, xs_plot, pdf_init_plot, color=:crimson, linewidth=2.5, label="Initial θ_init (KDE)")
axislegend(ax, position=:rt)
fig
# Save
figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)
out_path = joinpath(figdir, "pdf_comparison_reduced1d.png")
save(out_path, fig)
println("Saved PDF comparison figure to: " * out_path)

# Optional: print basic divergence metrics
function trapz(x, y)
    s = 0.0
    @inbounds for i in 2:length(x)
        s += 0.5 * (x[i]-x[i-1]) * (y[i] + y[i-1])
    end
    s
end
function discrete_kl(p::AbstractVector, q::AbstractVector)
    ϵ = 1e-12
    s = 0.0
    @inbounds for i in eachindex(p)
        pi = max(p[i], ϵ); qi = max(q[i], ϵ)
        s += pi * log(pi/qi)
    end
    s
end
Zp = trapz(xs_plot, pdf_true_plot); Zq = trapz(xs_plot, pdf_init_plot)
# Normalize numerically before KL
p_norm = pdf_true_plot ./ max(Zp, eps())
q_norm = pdf_init_plot ./ max(Zq, eps())
kl_pq = discrete_kl(p_norm, q_norm)
kl_qp = discrete_kl(q_norm, p_norm)
println(@sprintf("KL(true||init)=%.4e   KL(init||true)=%.4e", kl_pq, kl_qp))

# Simple L2 distance
l2 = sqrt(trapz(xs_plot, (pdf_true_plot .- pdf_init_plot).^2))
println(@sprintf("L2 distance = %.4e", l2))

# --------------------------------------------------------------------------------------
# Response functions & parameter Jacobians for both θ_true and θ_init
# Methods:
#   Responses: analytic, gaussian, neural
#   Jacobians: analytic, gaussian, neural, finite_diff
# Generates two figures (one per parameter set) with panels (observable x parameter)
# and prints S matrices for all four methods for each parameter set.
# --------------------------------------------------------------------------------------

using Dates

# Helper builders (duplicated from compute script for self-containment)
function _make_builders(μ::AbstractVector, Σ::AbstractVector)
    d = 1; S1 = Float64(Σ[1]); M1 = Float64(μ[1])
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

function _make_analytic_score(μ::AbstractVector, Σ::AbstractVector)
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

# Normalized simulator builder (local copy)
function _make_simulator(spec::PC.SimSpec, μ::AbstractVector, Σ::AbstractVector)
    μ_obs = collect(Float64.(μ)); Σ_obs = collect(Float64.(Σ))
    function simulator(θ::AbstractVector)
        drift_θ! = (du,u,t) -> (du[1] = θ[1] + θ[2]*u[1] + θ[3]*u[1]^2 - θ[4]*u[1]^3)
        sigma_θ! = (du,u,t) -> (du[1] = θ[5])
        Xp = PC.simulate(spec.u0, drift_θ!, sigma_θ!; spec=spec)
        return (Xp .- μ_obs) ./ Σ_obs
    end
    return simulator
end

function _build_estimators(θ_ref::Vector{Float64}, spec::PC.SimSpec; label::String)
    X_ref = simulate_phys(θ_ref, spec)
    μr = vec(mean(X_ref, dims=2)); Σr = vec(std(X_ref, dims=2))
    Xn = (X_ref .- μr) ./ Σr
    A_vec, obs_lbls = make_A_of_x(μr[1], Σr[1])
    builders = _make_builders(μr, Σr)
    model = PC.GFDTModel(
        s = x->zeros(Float64,1), divs=x->0.0, Js=x->zeros(Float64,1,1),
        F=builders.F_norm, Σ=builders.Σ_norm, dF_dθ=builders.dF_dθ_norm, dΣ_dθ=builders.dΣ_dθ_norm,
        div_dF_dθ=builders.div_dF_dθ_norm, divM=builders.divM_norm, divdivM=builders.divdivM_norm,
        θ=copy(θ_ref), mode=:general, xeltype=Float64)
    analytic_builder = _make_analytic_score(μr, Σr)
    est_analytic = PC.build_analytic_estimator(Xn, model, θ_ref; Δt=Δt_eff, Tmax=spec.Tmax, mean_center=true, analytic_builder=analytic_builder, A_of_x=A_vec)
    est_gauss    = PC.build_gaussian_estimator(Xn, model, θ_ref; Δt=Δt_eff, Tmax=spec.Tmax, mean_center=true, A_of_x=A_vec)
    Random.seed!(spec.seed)
    est_nn, _ = PC.build_neural_estimator(Xn, model, θ_ref, nn_cfg; Δt=Δt_eff, Tmax=spec.Tmax, mean_center=true, A_of_x=A_vec)
    simulator_norm = _make_simulator(spec, μr, Σr)
    est_fd = PC.build_finite_diff_estimator(simulator_norm, θ_ref, A_vec)
    return (
        X=X_ref, μ=μr, Σ=Σr, A=A_vec, labels=obs_lbls, thresholds=thresholds,
        ests=Dict(
            :analytic=>est_analytic,
            :gaussian=>est_gauss,
            :neural=>est_nn,
            :finite_diff=>est_fd
        ),
        tag=label
    )
end

println("\n[Responses/Jacobians] Building estimators for θ_true and θ_init ...")
θ_init = θ_true .* extra.calibration.θ_init_multipliers
res_true = _build_estimators(θ_true, spec_obs; label="θ_true")
res_init = _build_estimators(θ_init, spec_obs; label="θ_init")

function _plot_responses(res; Δt_eff=Δt_eff)
    ests = res.ests
    # Only methods with responses
    r_methods = [:analytic, :gaussian, :neural]
    C = Dict(sym => ests[sym].responses for sym in r_methods)
    C_any = first(values(C))
    nA, P, Kp1 = size(C_any)
    ts = (0:Kp1-1) .* Δt_eff
    fig = Figure(resolution=(300*P, 220*nA), fontsize=12)
    colors = Dict(:analytic=>:dodgerblue, :gaussian=>:seagreen, :neural=>:crimson)
    for a in 1:nA
        for p in 1:P
            ax = Axis(fig[a, p], xlabel=p==P ? "t" : "", ylabel=p==1 ? "A$(a)" : "", title=p==1 ? "A$(a)" : "")
            for sym in r_methods
                lines!(ax, ts, (@views C[sym][a,p,:]), color=colors[sym], linewidth=1.8, label=String(sym))
            end
            a==1 && p==P && axislegend(ax, position=:rt, framevisible=false)
        end
    end
    fig
end

fig_true = _plot_responses(res_true)
fig_init = _plot_responses(res_init)

figdir = joinpath(@__DIR__, "..", "figures"); mkpath(figdir)
save(joinpath(figdir, "responses_theta_true.png"), fig_true)
save(joinpath(figdir, "responses_theta_init.png"), fig_init)
println("Saved response figures to figures/responses_theta_{true,init}.png")

# Print Jacobian S matrices
function _print_S(res)
    println("\nParameter Jacobians S for " * res.tag * " (size: (nA x P))")
    for (sym, est) in res.ests
        S = est.S
        println("-- Method: $(sym), size=$(size(S))")
        for i in 1:size(S,1)
            @printf("%s[%2d]: ", "S", i)
            for j in 1:size(S,2)
                @printf(" % .4e", S[i,j])
            end
            print('\n')
        end
    end
end
_print_S(res_true)
_print_S(res_init)

nothing
