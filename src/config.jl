import TOML

"""
    load_config(path::AbstractString = "config/config.toml") -> (nn_cfg::NNTrainConfig, spec::SimSpec)

Load configuration from TOML, apply defaults, and return typed structures
`NNTrainConfig` and `SimSpec` used by the package. If `path` does not exist,
falls back to "config.toml" in the project root.
"""
function load_config(path::AbstractString = "config/config.toml")
    # Parse TOML (with fallback) into a Dict
    cfg = Dict{String,Any}()
    local parsed = nothing
    if isfile(path)
        try
            parsed = TOML.parsefile(path)
        catch err
            @warn "Failed parsing config; using defaults" path err
        end
    end
    if parsed === nothing
        alt = "config.toml"
        if isfile(alt)
            try
                parsed = TOML.parsefile(alt)
            catch err
                @warn "Failed parsing fallback config; using defaults" alt err
            end
        end
    end
    if parsed !== nothing
        cfg = parsed
    end

    # Extract tables with defaulting
    sim = get(cfg, "simulation", Dict{String,Any}())
    obs = get(cfg, "observables", Dict{String,Any}())
    nn  = get(cfg, "score_nn",  Dict{String,Any}())

    # Simulation/observables
    dt          = Float64(get(sim, "dt", 0.01))
    Nsteps      = Int(get(sim, "Nsteps", 10_000))
    resolution  = Int(get(sim, "resolution", 1))
    n_ens       = Int(get(sim, "n_ens", 1))
    u0          = Vector{Float64}(get(sim, "u0", [0.0]))
    burn_in     = Int(get(sim, "burn_in", 0))
    timestepper = Symbol(get(sim, "timestepper", "euler"))
    sigma_inpl  = Bool(get(sim, "sigma_inplace", true))
    noise_sym   = Symbol(get(sim, "noise", "shared"))
    rng_seed    = Int(get(sim, "rng_seed", 0xC0FFEE))
    Δt_mult     = Float64(get(obs, "dt_multiplier", 1.0))
    Tmax        = Float64(get(obs, "Tmax", 10.0))
    # Optional bounds (both must be present to be used). Accept scalar or vector.
    lb_raw      = get(sim, "lb", nothing)
    gb_raw      = get(sim, "gb", nothing)
    _scalar_bound(x, name) = x === nothing ? nothing : (
        x isa Real ? Float64(x) : error("Configuration error: $(name) must be a scalar; got $(typeof(x))")
    )
    lb_val = _scalar_bound(lb_raw, "lb")
    gb_val = _scalar_bound(gb_raw, "gb")
    if (lb_val === nothing) != (gb_val === nothing)
        error("Configuration error: provide both lb and gb or neither.")
    elseif lb_val !== nothing && gb_val !== nothing
        if !(lb_val < gb_val)
            error("Configuration error: require lb < gb; got lb=$(lb_val), gb=$(gb_val)")
        end
    end
    spec = SimSpec(
        dt=dt, Nsteps=Nsteps, n_ens=n_ens, u0=u0,
        burn_in=burn_in, resolution=resolution, seed=rng_seed,
        timestepper=timestepper, sigma_inplace=sigma_inpl, noise=noise_sym,
        Δt_multiplier=Δt_mult, Tmax=Tmax,
    lb=lb_val, gb=gb_val,
    )

    # NN score training
    preprocessing = Bool(get(nn, "preprocessing", true))
    σ_val     = Float64(get(nn, "sigma", 0.1))
    neurons_v = get(nn, "neurons", Int[])
    neurons   = isempty(neurons_v) ? Int[128, 128] : Int.(neurons_v)
    n_epochs  = Int(get(nn, "epochs", 100))
    epochs_re = Int(get(nn, "epochs_re", n_epochs))
    batch_sz  = Int(get(nn, "batch_size", 8192))
    lr        = Float64(get(nn, "lr", 1e-3))
    use_gpu   = Bool(get(nn, "use_gpu", false))
    verbose   = Bool(get(nn, "verbose", false))
    probes    = Int(get(nn, "probes", 1))
    radem     = Bool(get(nn, "rademacher", true))
    epochs_rf = 0
    kgmm_cfg  = get(nn, "kgmm", Dict{String,Any}())
    kgmm_kwargs = (
        prob = Float64(get(kgmm_cfg, "prob", 0.01)),
        conv_param = Float64(get(kgmm_cfg, "conv_param", 1e-4)),
        show_progress = Bool(get(kgmm_cfg, "show_progress", false)),
    )
    nn_cfg = NNTrainConfig(
        preprocessing=preprocessing,
        σ=σ_val, neurons=collect(Int.(neurons)), n_epochs=n_epochs,
        epochs_re=epochs_re,
        epochs_ref=epochs_rf, batch_size=batch_sz, lr=lr, use_gpu=use_gpu,
        verbose=verbose, probes=probes, rademacher=radem, kgmm_kwargs=kgmm_kwargs,
    )
    return nn_cfg, spec
end

function load_extra_config(path::AbstractString = "config/config.toml")
    # Helper to parse TOML or fall back
    function _parse_or_default(p)
        if isfile(p)
            try
                return TOML.parsefile(p)
            catch err
                @warn "Failed parsing config in load_extra_config; using defaults" path err
            end
        end
        return Dict{String,Any}()
    end
    cfg = _parse_or_default(path)

    # Utilities
    _gets(d::Dict, k::String, default) = get(d, k, default)
    _getsym(x) = Symbol(lowercase(String(x)))
    function _getsyms(v)
        if v === nothing
            return Symbol[]
        elseif v isa AbstractVector
            return [ _getsym(s) for s in v ]
        else
            return [_getsym(v)]
        end
    end

    mdl = get(cfg, "model", Dict{String,Any}())
    obs = get(cfg, "observables", Dict{String,Any}())
    cal = get(cfg, "calibration", Dict{String,Any}())
    plt = get(cfg, "plots",       Dict{String,Any}())

    # Model (true parameters for simulation of observed data)
    model = (
        F_tilde = Float64(get(mdl, "F_tilde", 0.6)),
        a = Float64(get(mdl, "a", -0.0222)),
        b = Float64(get(mdl, "b", -0.2)),
        c = Float64(get(mdl, "c", 0.0494)),
        s = Float64(get(mdl, "s", 0.7071)),
    )

    # Observables settings
    use_moments    = _getsyms(get(obs, "use_moments", ["m1","m2","m3","m4"]))
    use_indicators = _getsyms(get(obs, "use_indicators", ["ge","le"]))
    thresholds_raw = get(obs, "thresholds", (-1.5, 2.5))
    # Normalize thresholds into a NamedTuple expected by observables.build_A_of_x
    # Accepted user forms:
    #   single number x          -> (α = x,)   (interpreted as lower/"ge" threshold)
    #   vector/tuple length 1    -> (α = v[1],)
    #   vector/tuple length 2    -> (α = v[1], β = v[2])
    #   Dict with keys "α"/"β"    -> map to NamedTuple
    #   Already a NamedTuple     -> passed through
    function _norm_thresholds(traw)
        if traw isa NamedTuple
            return traw
        elseif traw isa AbstractDict
            a = haskey(traw, "α") ? traw["α"] : (haskey(traw, "a") ? traw["a"] : nothing)
            b = haskey(traw, "β") ? traw["β"] : (haskey(traw, "b") ? traw["b"] : nothing)
            nt = (;)
            if a !== nothing && b !== nothing
                return (α = Float64(a), β = Float64(b))
            elseif a !== nothing
                return (α = Float64(a),)
            elseif b !== nothing
                return (β = Float64(b),)
            else
                return (α = -1.5, β = 2.5) # fallback
            end
        elseif traw isa AbstractVector || traw isa Tuple
            L = length(traw)
            if L == 1
                return (α = Float64(traw[1]),)
            elseif L >= 2
                return (α = Float64(traw[1]), β = Float64(traw[2]))
            else
                return (α = -1.5, β = 2.5)
            end
        elseif traw isa Real
            return (α = Float64(traw),)
        else
            # Unknown format; use default canonical pair
            return (α = -1.5, β = 2.5)
        end
    end
    thresholds = _norm_thresholds(thresholds_raw)

    # Calibration settings
    methods_syms   = _getsyms(get(cal, "methods", ["analytic","gaussian","neural","finite_diff"]))
    line_search    = Bool(get(cal, "line_search", false))
    line_search_max = Int(get(cal, "line_search_max", 4))
    callback_flag   = Bool(get(cal, "callback", false))
    callback_dt     = Float64(get(cal, "callback_dt", 0.0))
    callback_Nsteps = Int(get(cal, "callback_Nsteps", 0))
    maxiters       = Int(get(cal, "maxiters", 4))
    tol_theta      = Float64(get(cal, "tol_theta", 1e-6))
    damping        = Float64(get(cal, "damping", 1.0))
    mean_center    = Bool(get(cal, "mean_center", true))
    rng_offset     = Int(get(cal, "rng_offset", 1))
    θ_init_mult    = collect(Float64.(get(cal, "theta_init_multipliers", [1.10, 0.80, 1.20, 0.90, 1.05])))
    free_idx_val   = get(cal, "free_idx", nothing)
    free_idx_vec   = free_idx_val === nothing ? nothing : collect(Int.(free_idx_val))

    # Plot settings
    xs_min         = Float64(get(plt, "xs_min", -2.5))
    xs_max         = Float64(get(plt, "xs_max",  2.5))
    xs_len         = Int(get(plt, "xs_len", 300))

    return (
        model = model,
        observables = (
            use_moments=use_moments,
            use_indicators=use_indicators,
            thresholds=thresholds,
        ),
        calibration = (
            methods=methods_syms,
            maxiters=maxiters,
            tol_θ=tol_theta,
            damping=damping,
            mean_center=mean_center,
            rng_offset=rng_offset,
            θ_init_multipliers=θ_init_mult,
            free_idx=free_idx_vec,
            line_search=line_search,
            line_search_max=line_search_max,
            callback=callback_flag,
            callback_dt=callback_dt,
            callback_Nsteps=callback_Nsteps,
        ),
        plots = (
            xs_min=xs_min,
            xs_max=xs_max,
            xs_len=xs_len,
        ),
    )
end
