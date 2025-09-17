function stats_AG(A_of_x::Function, X_obs::AbstractMatrix, X_mod::AbstractMatrix)
    Tobs = size(X_obs,2); Tmod = size(X_mod,2)
    m = length(A_of_x(@view X_obs[:,1]))
    A = zeros(Float64, m); G = zeros(Float64, m)
    @inbounds for t in 1:Tobs
        A .+= A_of_x(@view X_obs[:,t])
    end
    A ./= max(Tobs, 1)
    @inbounds for t in 1:Tmod
        G .+= A_of_x(@view X_mod[:,t])
    end
    G ./= max(Tmod, 1)
    return A, G
end

function weight_inverse_cov(A_of_x::Function, X_obs::AbstractMatrix)
    m = length(A_of_x(@view X_obs[:,1])); T = size(X_obs,2)
    Aseries = Matrix{Float64}(undef, m, T)
    for t in 1:T
        @views Aseries[:,t] .= A_of_x(X_obs[:,t])
    end
    μ = mean(Aseries, dims=2)
    Acenter = Aseries .- μ
    Σ = Symmetric((Acenter * Acenter') / max(T-1, 1))
    Winv = Σ + 1e-8I
    W = inv(Winv)
    return Symmetric((W + W')/2)
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
    calibration_loop(method::SensitivityMethod, base_model::GFDTModel, A_of_x::Function;
                     sim_norm::Function,
                     X_obs_norm::AbstractMatrix,
                     θ0::AbstractVector,
                     Δt::Real, Tmax::Real,
                     nn_cfg::Union{Nothing,NNTrainConfig}=nothing,
                     analytic_builder::Union{Nothing,Function}=nothing,
                     W::Union{Nothing,AbstractMatrix,Symmetric}=nothing,
                     Γ::Union{Nothing,Symmetric}=nothing,
                     free_idx::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                     maxiters::Integer=8, tol_θ::Real=1e-6, damping::Real=1.0,
                     mean_center::Bool=true)

Iterative plain-Newton calibration for a single specified `method`.

Inputs:
- `method`: one of `GFDTAnalyticScore`, `GFDTGaussianScore`, `GFDTNeuralScore`, `FiniteDifference`.
- `base_model`: supplies dynamics/θ-derivatives; score is updated per iteration depending on `method`.
- `A_of_x`: observables (act on normalized state; return physical units if desired).
- `sim_norm`: closure `(θ) -> X::Matrix` in normalized coordinates.
- `X_obs_norm`: observed trajectory in normalized coordinates.
- `θ0`: initial parameters.
- `Δt, Tmax`: GFDT correlation settings.
- `nn_cfg`: training settings for the neural method. Policy: if `nn_cfg.preprocessing==true`,
  a fresh NN is trained from scratch at every iteration; otherwise the NN is
  continued from the previous iteration for `nn_cfg.epochs_re` epochs.
- `analytic_builder`: function `θ -> (s=..., Js=...)` for analytic method.
- `W, Γ`: optional weight and regularizer.
- `free_idx`: optional subset of parameter indices for FiniteDifference; defaults to all.

Returns a NamedTuple with fields: θ, θ_path, G_list, S_list, A_target, W, Γ.
"""
function calibration_loop(method::SensitivityMethod, base_model::GFDTModel, A_of_x::Function;
                          sim_norm::Function,
                          X_obs_norm::AbstractMatrix,
                          θ0::AbstractVector,
                          Δt::Real, Tmax::Real,
                          nn_cfg::Union{Nothing,NNTrainConfig}=nothing,
                          analytic_builder::Union{Nothing,Function}=nothing,
                          W::Union{Nothing,AbstractMatrix,Symmetric}=nothing,
                          Γ::Union{Nothing,Symmetric}=nothing,
                          free_idx::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                          maxiters::Integer=8, tol_θ::Real=1e-6, damping::Real=1.0,
                          mean_center::Bool=true)
    # Targets/weights from observed normalized data
    A_target, _ = stats_AG(A_of_x, X_obs_norm, X_obs_norm)
    Wmat = W === nothing ? weight_inverse_cov(A_of_x, X_obs_norm) : Symmetric(Matrix(W))
    Γmat = Γ === nothing ? Symmetric(zeros(length(θ0), length(θ0))) : Γ

    θ = collect(Float64.(θ0))
    θ_path = Vector{Vector{Float64}}()
    G_list = Vector{Vector{Float64}}()
    S_list = Vector{Matrix{Float64}}()

    p = ProgressMeter.Progress(maxiters, "Calibration iterations"; showspeed=false)
    nn_current = nothing
    for _it in 1:maxiters
        ProgressMeter.next!(p)
        push!(θ_path, copy(θ))
        # Simulate model trajectory and exit early if it is invalid
        Xn = sim_norm(θ)
        if any(!isfinite, Xn)
            @warn "Calibration: simulated trajectory contains non-finite values (NaN/Inf); exiting loop early" iteration=_it
            break
        end
        m = length(A_of_x(@view Xn[:,1]))
        G = zeros(Float64, m)
        Tn = size(Xn,2)
        @inbounds for t in 1:Tn
            G .+= A_of_x(@view Xn[:,t])
        end
        G ./= max(Tn, 1)
        push!(G_list, copy(G))

        # Build sensitivities S per method
        S = if method isa GFDTAnalyticScore
            @assert analytic_builder !== nothing "Provide `analytic_builder` for GFDTAnalyticScore."
            build_analytic_estimator(Xn, base_model, θ; Δt=Δt, Tmax=Tmax,
                                     mean_center=mean_center, analytic_builder=analytic_builder,
                                     A_of_x=A_of_x, store_response=false).S
        elseif method isa GFDTGaussianScore
            build_gaussian_estimator(Xn, base_model, θ; Δt=Δt, Tmax=Tmax,
                                     mean_center=mean_center, A_of_x=A_of_x, store_response=false).S
        elseif method isa GFDTNeuralScore
            @assert nn_cfg !== nothing "Provide `nn_cfg` for GFDTNeuralScore training."
            # Policy: if preprocessing==true, train from scratch each iteration; else continue training
            local nn_in = nn_cfg.preprocessing ? nothing : nn_current
            est_nn, nn_new = build_neural_estimator(Xn, base_model, θ, nn_cfg; Δt=Δt, Tmax=Tmax,
                                                    mean_center=mean_center, A_of_x=A_of_x,
                                                    store_response=false, nn=nn_in)
            nn_current = nn_new
            est_nn.S
        elseif method isa FiniteDifference
            free = free_idx === nothing ? collect(1:length(θ)) : collect(free_idx)
            finite_difference_jacobian(sim_norm, θ, A_of_x; free_idx=free, h_rel=method.h_rel, h_abs=method.h_abs)
        else
            error("Unsupported method type $(typeof(method))")
        end
        push!(S_list, Matrix(S))

        # Newton step and update (respect optional free_idx subset)
        local S_use::AbstractMatrix
        local Γ_use::Symmetric
        local free::Vector{Int}
        if free_idx === nothing || method isa FiniteDifference
            # If FD, S already corresponds to selected parameters; treat as full update
            S_use = S
            Γ_use = Γmat
            free = collect(1:length(θ))
        else
            free = collect(free_idx)
            S_use = S[:, free]
            Γ_use = Symmetric(Matrix(Γmat[free, free]))
        end

        Δθ_sub, _diag = newton_step(S_use, Wmat, Γ_use, G, A_target; jitter=1e-10)
        # Build full update vector for convergence check and apply update
        Δθ_all = if length(Δθ_sub) == length(θ)
            Δθ_sub
        else
            Δθ_full = zeros(Float64, length(θ))
            @inbounds for (k, idx) in enumerate(free)
                Δθ_full[idx] = Δθ_sub[k]
            end
            Δθ_full
        end
        θ .-= damping .* Δθ_all
        if norm(Δθ_all) / max(norm(θ), eps()) ≤ tol_θ
            break
        end
    end

    return (; θ=θ, θ_path=θ_path, G_list=G_list, S_list=S_list,
            A_target=A_target, W=Wmat, Γ=Γmat, nn=nn_current)
end
