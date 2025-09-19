using Plots
using Printf



function weight_inverse_cov(A_of_x::Function, X_obs::AbstractMatrix; base_jitter::Real=1e-8, max_tries::Int=5)
    m = length(A_of_x(@view X_obs[:,1])); T = size(X_obs,2)
    Aseries = Matrix{Float64}(undef, m, T)
    for t in 1:T
        @views Aseries[:,t] .= A_of_x(X_obs[:,t])
    end
    μ = mean(Aseries, dims=2)
    Acenter = Aseries .- μ
    Σ = Symmetric((Acenter * Acenter') / max(T-1, 1))
    # Compute inverse covariance via Cholesky factorization with jitter escalation
    jitter = base_jitter
    local chol::Union{Cholesky{Float64,Matrix{Float64}},Nothing} = nothing
    for k in 1:max_tries
        try
            chol = cholesky(Symmetric(Matrix(Σ) + jitter*I), check=true)
            break
        catch err
            if k == max_tries
                rethrow(err)
            end
            jitter *= 10
        end
    end
    @assert chol !== nothing
    # Build inverse via a single solve; chol \ I returns (Σ + jitter*I)^{-1}
    W = chol \ I
    return Symmetric(Matrix(W))
end

@inline function _call_make_A_of_x(make_A_of_x::Function, μ::AbstractVector, Σ::AbstractVector)
    if applicable(make_A_of_x, μ, Σ)
        return make_A_of_x(μ, Σ)
    elseif applicable(make_A_of_x, μ[1], Σ[1])
        return make_A_of_x(μ[1], Σ[1])
    else
        error("make_A_of_x does not accept (μ::Vector, Σ::Vector) nor scalars. Provide a compatible callable.")
    end
end

function stats_A(X::AbstractMatrix, A_of_x::Function)
    T = size(X, 2)
    m = length(A_of_x(@view X[:, 1]))
    A = zeros(Float64, m)
    @inbounds for t in 1:T
        A .+= A_of_x(@view X[:, t])
    end
    A ./= max(T, 1)
    return A
end

function score_callback!(X_norm::AbstractMatrix, s_fn::Function; dt::Real, Nsteps::Integer, method::Symbol, iteration::Integer, lb::Union{Nothing,Float64}=nothing, gb::Union{Nothing,Float64}=nothing)
    dt > 0 || return
    Nsteps > 0 || return
    d, _ = size(X_norm)
    x0 = copy(vec(X_norm[:, 1]))
    drift! = function (du, u, t)
        sval = s_fn(Vector{Float64}(u))
        du .= Float64.(sval)
    end
    sqrt2 = sqrt(2.0)
    sigma! = (du, u, t) -> (du .= sqrt2)
    if lb !== nothing || gb !== nothing
        @assert (lb !== nothing && gb !== nothing) "Provide both lb and gb or neither."
        traj = FastSDE.evolve(copy(x0), dt, Nsteps, drift!, sigma!; timestepper=:euler, resolution=1, sigma_inplace=true, n_ens=1, boundary=(lb, gb))
    else
        traj = FastSDE.evolve(copy(x0), dt, Nsteps, drift!, sigma!; timestepper=:euler, resolution=1, sigma_inplace=true, n_ens=1)
    end
    X_sim = Float64.(traj)
    figdir = joinpath(pwd(), "figures", "callback")
    mkpath(figdir)
    plt = plot(layout=(d, 1), size=(800, max(240, 240 * d)))
    for i in 1:d
        obs = vec(X_norm[i, :])
        sim = vec(X_sim[i, :])
        legendpos = i == 1 ? :topright : :none
        histogram!(plt, obs; subplot=i, normalize=:pdf, bins=:auto, alpha=0.4, label=i == 1 ? "observed" : "", color=:steelblue, legend=legendpos)
        histogram!(plt, sim; subplot=i, normalize=:pdf, bins=:auto, alpha=0.4, label=i == 1 ? "score" : "", color=:darkorange, legend=legendpos)
        xlabel!(plt, "x_$(i)"; subplot=i)
        ylabel!(plt, "pdf"; subplot=i)
        title!(plt, "Dimension $(i)"; subplot=i)
    end
    filename = joinpath(figdir, @sprintf("score_callback_%s_iter%02d.png", String(method), iteration))
    savefig(plt, filename)
end

function make_regularizer(θ::AbstractVector; λ::Real=0.0, λvec::AbstractVector=nothing, Cprior::AbstractMatrix=nothing, θprior::AbstractVector=nothing)
    p = length(θ)
    λd = λvec === nothing ? fill(float(λ), p) : collect(float.(λvec))
    Γ = Diagonal(λd)
    if Cprior !== nothing
        Γ = Symmetric(Cprior' * Γ * Cprior)
    end
    θp = θprior === nothing ? copy(θ) : collect(float.(θprior))
    return Symmetric(Matrix(Γ)), θp
end

function newton_step(S::AbstractMatrix, W::Union{AbstractMatrix,Symmetric}, Γ::Symmetric,
                     G::AbstractVector, A::AbstractVector; jitter::Real=1e-10)
    @assert size(S,1) == length(A) == length(G)
    @assert size(S,2) == size(Γ,1) == size(Γ,2)
    p = size(S,2)
    M   = Matrix{Float64}(undef, p, p)
    tmp = similar(G)
    mul!(tmp, W, (G .- A))          # tmp = W (G-A)
    rhs = S' * tmp
    mul!(M, S', W * S)              # M = S' W S
    M .+= Matrix(Γ)
    ϑ = _spd_solve!(M, rhs; jitter=jitter)
    local cval::Float64
    try
        cval = cond(M)
    catch
        cval = NaN
    end
    return ϑ, (cond=cval, nrm_rhs=norm(rhs))
end

"""
    calibration_loop(method::SensitivityMethod, A_target; simulator_obs, make_builders, make_A_of_x,
                     θ0, Δt, Tmax; nn_cfg=nothing, make_analytic_score=nothing,
                     make_simulator_norm=nothing, W=nothing, Γ=nothing,
                     free_idx=nothing, maxiters=8, tol_θ=1e-6, damping=1.0,
                     mean_center=true, adapt_W=false, line_search=false, line_search_max=4)

Iterative plain-Newton calibration for a single specified `method` using precomputed target observables `A_target`.

Key arguments:
- `method`: one of `GFDTAnalyticScore`, `GFDTGaussianScore`, `GFDTNeuralScore`, `FiniteDifference`.
- `A_target`: target observable vector (must match rows of sensitivities; validated on first iteration).
- `simulator_obs(θ)`: returns trajectory in observed (physical) coordinates.
- `make_builders(μ, Σ)`: returns normalized dynamics/derivative closures.
- `make_A_of_x(μ₁, Σ₁)`: builds observable closure for normalized coordinates.
- `θ0`: initial parameter vector.
- `Δt, Tmax`: correlation / GFDT settings.
- `nn_cfg`: neural score training config (only for `GFDTNeuralScore`).
- `make_analytic_score`: builder for analytic score (only for `GFDTAnalyticScore`).
- `make_simulator_norm`: builds normalized simulator (finite-difference method).
- `W`: optional fixed weight matrix; if omitted and `adapt_W=false`, one is built from initial θ0 only.
- `adapt_W`: when true (and `W` not supplied), recomputes weight matrix each iteration from current trajectory.
- `Γ`: optional regularization (symmetric) matrix.
- `free_idx`: subset of parameters to update. For `FiniteDifference`, the Jacobian is built only
    for these indices (others receive zero update); for analytic / gaussian / neural methods the
    full Jacobian is built and then restricted.
- `mean_center`: whether to mean-center responses where applicable.
- `line_search`: enable backtracking step halving based on observable distance.
- `line_search_max`: maximum number of halvings attempted per iteration when `line_search=true`.
- `callback`: when true and method is Gaussian/Neural, generates score-based diagnostic figures each iteration.
- `callback_dt`, `callback_Nsteps`: integration settings for the callback simulator (ignored when `callback=false`).

Returns: NamedTuple with fields `θ, θ_path, G_list, S_list, A_target, W, Γ, nn, adapt_W`.

Note: Adaptive weight updates can improve robustness if observable covariance shifts along the trajectory; disable for strict comparability runs.
"""
function calibration_loop(method::SensitivityMethod, A_target::AbstractVector;
                          simulator_obs::Function,
                          make_builders::Function,
                          make_A_of_x::Function,
                          θ0::AbstractVector,
                          Δt::Real, Tmax::Real,
                          nn_cfg::Union{Nothing,NNTrainConfig}=nothing,
                          make_analytic_score::Union{Nothing,Function}=nothing,
                          make_simulator_norm::Union{Nothing,Function}=nothing,
                          W::Union{Nothing,AbstractMatrix,Symmetric}=nothing,
                          Γ::Union{Nothing,Symmetric}=nothing,
                          free_idx::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                          maxiters::Integer=8, tol_θ::Real=1e-6, damping::Real=1.0,
                          mean_center::Bool=true, adapt_W::Bool=false,
                          line_search::Bool=false, line_search_max::Integer=4,
                          callback::Bool=false, callback_dt::Real=0.0, callback_Nsteps::Integer=0,
                          lb::Union{Nothing,Float64}=nothing, gb::Union{Nothing,Float64}=nothing)

    # Weight matrix initialisation: if provided externally keep as fixed; otherwise build from θ0.
    # If adapt_W=true and W not supplied, W will be recomputed each iteration from current trajectory.
    if W === nothing
        X_init_obs = simulator_obs(θ0)
        μ_init = vec(mean(X_init_obs, dims=2)); Σ_init = vec(std(X_init_obs, dims=2))
            A_init, _ = _call_make_A_of_x(make_A_of_x, μ_init, Σ_init)
        Wmat_initial = weight_inverse_cov(A_init, (X_init_obs .- μ_init) ./ Σ_init)
    else
        Wmat_initial = Symmetric(Matrix(W))
    end
    Γmat = Γ === nothing ? Symmetric(zeros(length(θ0), length(θ0))) : Γ

    θ = collect(Float64.(θ0))
    θ_path = Vector{Vector{Float64}}()
    G_list = Vector{Vector{Float64}}()
    S_list = Vector{Matrix{Float64}}()

    p = ProgressMeter.Progress(maxiters, "Calibration iterations"; showspeed=false)
    nn_current = nothing
    Wmat_current = Wmat_initial
    for _it in 1:maxiters
        ProgressMeter.next!(p)
        push!(θ_path, copy(θ))
        # First iteration can optionally reuse a shared initial state to align starting observables across methods
        local Xobs::AbstractMatrix
        local μ::Vector{Float64}
        local Σ::Vector{Float64}
        local Xn::Array{Float64,2}
        local A_of_x_iter::Function
        local G::Vector{Float64}
        # Simulate model trajectory and exit early if it is invalid 
        Xobs = simulator_obs(θ)
        μ = vec(mean(Xobs, dims=2)); Σ = vec(std(Xobs, dims=2))
        Xn = (Xobs .- μ) ./ Σ
        if any(!isfinite, Xn)
            @warn "Calibration: simulated trajectory contains non-finite values (NaN/Inf); exiting loop early" iteration=_it
            break
        end
        # Build model observables
        A_of_x_iter, _ = _call_make_A_of_x(make_A_of_x, μ, Σ)
        # Use normalized trajectory for observables (A_of_x expects normalized x)
        G = stats_A(Xn, A_of_x_iter)
        push!(G_list, copy(G))
        builders = make_builders(μ, Σ)
        base_model = GFDTModel(
                            s=x->zeros(Float64,1), divs=x->0.0, Js=x->zeros(Float64,1,1),
                            F=builders.F_norm, Σ=builders.Σ_norm, dF_dθ=builders.dF_dθ_norm, dΣ_dθ=builders.dΣ_dθ_norm,
                            div_dF_dθ=builders.div_dF_dθ_norm, divM=builders.divM_norm, divdivM=builders.divdivM_norm,
                            θ=θ, mode=:general, xeltype=Float64)

        # Build sensitivities S per method
        S = if method isa GFDTAnalyticScore
            @assert make_analytic_score !== nothing "Provide `make_analytic_score` for GFDTAnalyticScore."
            analytic_builder = make_analytic_score(μ, Σ)
            build_analytic_estimator(Xn, base_model, θ; Δt=Δt, Tmax=Tmax,
                                     mean_center=mean_center, analytic_builder=analytic_builder,
                                     A_of_x=A_of_x_iter, store_response=false).S
        elseif method isa GFDTGaussianScore
            est_gauss = build_gaussian_estimator(Xn, base_model, θ; Δt=Δt, Tmax=Tmax,
                                     mean_center=mean_center, A_of_x=A_of_x_iter, store_response=false)
            if callback
                score_callback!(Xn, est_gauss.model_view.s; dt=callback_dt, Nsteps=callback_Nsteps, method=:gaussian, iteration=_it, lb=lb, gb=gb)
            end
            est_gauss.S
        elseif method isa GFDTNeuralScore
            @assert nn_cfg !== nothing "Provide `nn_cfg` for GFDTNeuralScore training."
            # Policy: if preprocessing==true, train from scratch each iteration; else continue training
            local nn_in = nn_cfg.preprocessing ? nothing : nn_current
            est_nn, nn_new = build_neural_estimator(Xn, base_model, θ, nn_cfg; Δt=Δt, Tmax=Tmax,
                                                    mean_center=mean_center, A_of_x=A_of_x_iter,
                                                    store_response=false, nn=nn_in)
            if callback
                score_callback!(Xn, est_nn.model_view.s; dt=callback_dt, Nsteps=callback_Nsteps, method=:neural, iteration=_it, lb=lb, gb=gb)
            end
            nn_current = nn_new
            est_nn.S
        elseif method isa FiniteDifference
            # Build FD Jacobian on requested subset (or all if none specified)
            @assert make_simulator_norm !== nothing "Provide `make_simulator_norm` for FiniteDifference."
            simulator_norm = make_simulator_norm(μ, Σ)
            finite_difference_jacobian(
                simulator_norm, θ, A_of_x_iter;
                free_idx = free_idx === nothing ? collect(1:length(θ)) : collect(free_idx),
                h_rel = method.h_rel, h_abs = method.h_abs
            )
        else
            error("Unsupported method type $(typeof(method))")
        end
        push!(S_list, Matrix(S))
        if _it == 1
            @assert length(A_target) == size(S,1) "A_target length (" * string(length(A_target)) * ") does not match number of observable rows in sensitivities (" * string(size(S,1)) * ")."
        end

        # Newton step and update (respect optional free_idx subset)
        local S_use::AbstractMatrix
        local Γ_use::Symmetric
        local free::Vector{Int}
        if free_idx === nothing
            S_use = S
            Γ_use = Γmat
            free = collect(1:length(θ))
        else
            free = collect(free_idx)
            S_use = S[:, free]
            Γ_use = Symmetric(Matrix(Γmat[free, free]))
        end

        Δθ_sub, _diag = newton_step(S_use, Wmat_current, Γ_use, G, A_target; jitter=1e-10)
        # Build full update vector for convergence check
        Δθ_all = if length(Δθ_sub) == length(θ)
            Δθ_sub
        else
            Δθ_full = zeros(Float64, length(θ))
            @inbounds for (k, idx) in enumerate(free)
                Δθ_full[idx] = Δθ_sub[k]
            end
            Δθ_full
        end
        prev_distance = norm(G .- A_target)
        actual_step = similar(θ)
        if line_search
            θ_base = copy(θ)
            step_scale = damping
            attempt = 0
            success = false
            while true
                θ_candidate = θ_base .- step_scale .* Δθ_all
                X_ls = simulator_obs(θ_candidate)
                μ_ls = vec(mean(X_ls, dims=2))
                Σ_ls = vec(std(X_ls, dims=2))
                Xn_ls = (X_ls .- μ_ls) ./ Σ_ls
                dist_candidate = Inf
                A_ls = nothing
                G_ls = nothing
                if all(isfinite, Xn_ls)
                    A_ls, _ = _call_make_A_of_x(make_A_of_x, μ_ls, Σ_ls)
                    G_ls = stats_A(Xn_ls, A_ls)
                    dist_candidate = norm(G_ls .- A_target)
                end
                if dist_candidate <= prev_distance
                    @assert A_ls !== nothing && G_ls !== nothing
                    θ .= θ_candidate
                    actual_step .= θ_base .- θ
                    μ = μ_ls
                    Σ = Σ_ls
                    Xn = Xn_ls
                    A_of_x_iter = A_ls
                    G = G_ls
                    G_list[end] = copy(G_ls)
                    success = true
                    break
                else
                    attempt += 1
                    if attempt > line_search_max
                        break
                    end
                    step_scale *= 0.5
                end
            end
            if !success
                @warn "Calibration: line search failed to reduce observable distance; exiting loop" iteration=_it
                break
            end
        else
            actual_step .= damping .* Δθ_all
            θ .-= actual_step
        end

        # Adaptive weight matrix update (only when not user-supplied and flag is on)
        if adapt_W && W === nothing
            # Recompute W from current normalized trajectory and updated observables
            A_adapt, _ = _call_make_A_of_x(make_A_of_x, μ, Σ)
            Wmat_current = weight_inverse_cov(A_adapt, Xn)
        end
        if norm(actual_step) / max(norm(θ), eps()) ≤ tol_θ
            break
        end
    end

    return (; θ=θ, θ_path=θ_path, G_list=G_list, S_list=S_list,
        A_target=A_target, W=Wmat_current, Γ=Γmat, nn=nn_current,
        adapt_W=adapt_W)
end
