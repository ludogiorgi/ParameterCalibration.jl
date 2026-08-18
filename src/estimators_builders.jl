using LinearAlgebra

"""
    _make_model_view(base::GFDTModel, θ::AbstractVector, s::Function, Js::Function)

Build a lightweight GFDTModel view that reuses dynamics/derivatives from `base`
but injects `s`, `Js` and corresponding `divs` at the provided `θ`.
"""
function _make_model_view(base::GFDTModel, θ::AbstractVector, s::Function, Js::Function)
    divs = x -> tr(Js(x))
    return GFDTModel(
        s = s,
        divs = divs,
        Js = Js,
        F = base.F,
        Σ = base.Σ,
        dF_dθ = base.dF_dθ,
        dΣ_dθ = base.dΣ_dθ,
        div_dF_dθ = base.div_dF_dθ,
        divM = base.divM,
        divdivM = base.divdivM,
        θ = θ,
        mode = base.mode,
        xeltype = base.xeltype,
    )
end

@inline _Kmax(Δt::Real, Tmax::Real, T::Integer) = Int(clamp(floor(Tmax/Δt), 0, T-1))

"""
    build_responses(model::GFDTModel, X::AbstractMatrix, A_of_x::Function;
                    Δt::Real, Tmax::Real, mean_center::Bool=true)
        -> (responses::Array{Float64,3}, S::Matrix{Float64})

Compute time-lagged responses C[i,j,k] = E[A_i(t+k) B_j(t)] for k=0..Kmax and
integrate them to obtain the Jacobian. Composite Simpson quadrature is the
default; `:trapezoid` and the legacy `:rectangle` rule are also available.
Mean-center A and B if requested.
"""
function build_responses(model_v::GFDTModel,
                         X::AbstractMatrix,
                         A_of_x::Function;
                         Δt::Real,
                         Tmax::Real,
                         mean_center::Bool=true,
                         quadrature::Symbol=:simpson)
    d, T = size(X)
    A1 = A_of_x(@view X[:,1])
    m = length(A1)
    p = length(model_v.θ)

    Aseries = Matrix{Float64}(undef, m, T)
    Bseries = Matrix{Float64}(undef, p, T)
    @inbounds for t in 1:T
        x = @view X[:,t]
        @views Aseries[:,t] = A_of_x(x)
        @views Bseries[:,t] = B_gfdt(model_v, x)
    end

    return build_responses_from_series(Aseries, Bseries;
                                       Δt=Δt, Tmax=Tmax,
                                       mean_center=mean_center,
                                       quadrature=quadrature)
end

"""
    integrate_responses(responses, Δt; quadrature=:simpson)

Integrate the continuous-time GFDT correlation along its lag dimension.
Composite Simpson quadrature is the default because the former left-rectangle
rule produced a visible `O(Δt)` bias at coarse response cadence.  If the
number of intervals is odd, Simpson is used up to the penultimate interval and
a trapezoid closes the final interval.
"""
function integrate_responses(responses::AbstractArray{<:Real,3},
                             Δt::Real;
                             quadrature::Symbol=:simpson)
    nintervals = size(responses, 3) - 1
    nintervals >= 0 || throw(ArgumentError("responses must contain at least one lag"))
    dt = float(Δt)
    if quadrature === :rectangle || nintervals == 0
        return -dt .* dropdims(sum(responses, dims=3), dims=3)
    elseif quadrature === :trapezoid
        interior = nintervals > 1 ? dropdims(sum(@view(responses[:, :, 2:end-1]), dims=3), dims=3) :
                                   zeros(Float64, size(responses, 1), size(responses, 2))
        return -dt .* (0.5 .* responses[:, :, 1] .+ interior .+
                       0.5 .* responses[:, :, end])
    elseif quadrature === :simpson
        q = isodd(nintervals) ? nintervals - 1 : nintervals
        if q == 0
            return -0.5dt .* (responses[:, :, 1] .+ responses[:, :, 2])
        end
        integral = Matrix{Float64}(responses[:, :, 1] .+ responses[:, :, q + 1])
        @inbounds for k in 1:q-1
            integral .+= (isodd(k) ? 4.0 : 2.0) .* responses[:, :, k + 1]
        end
        integral .*= dt / 3
        if q < nintervals
            integral .+= 0.5dt .* (responses[:, :, q + 1] .+ responses[:, :, q + 2])
        end
        return -integral
    end
    throw(ArgumentError("quadrature must be :simpson, :trapezoid, or :rectangle"))
end

"""
    build_responses_from_series(Aseries, Bseries; Δt, Tmax, mean_center=true,
                                quadrature=:simpson)

Integrate a full GFDT-style response matrix from already evaluated observable
and conjugate-variable series.  This separates expensive model/score
evaluation from the shared FFT correlation engine and enables vectorized or
batched `B` construction without changing the estimator definition.
"""
function build_responses_from_series(Aseries::AbstractMatrix,
                                     Bseries::AbstractMatrix;
                                     Δt::Real,
                                     Tmax::Real,
                                     mean_center::Bool=true,
                                     quadrature::Symbol=:simpson)
    size(Aseries, 2) == size(Bseries, 2) ||
        throw(DimensionMismatch("Aseries and Bseries must share the time dimension"))
    T = size(Aseries, 2)
    Awork = Matrix{Float64}(Aseries)
    Bwork = Matrix{Float64}(Bseries)
    if mean_center
        Awork .-= mean(Awork, dims=2)
        Bwork .-= mean(Bwork, dims=2)
    end
    Kmax = _Kmax(Δt, Tmax, T)
    responses = xcorr_matrix_one_sided(Awork, Bwork, Kmax)
    S = integrate_responses(responses, Δt; quadrature=quadrature)
    return responses, S
end

"""
    build_analytic_estimator(X, model::GFDTModel, θ; Δt, Tmax, mean_center=true,
                             analytic_builder=nothing, A_of_x=nothing, store_response=true)
"""
function build_analytic_estimator(X::AbstractMatrix,
                                  model::GFDTModel,
                                  θ::AbstractVector;
                                  Δt::Real,
                                  Tmax::Real,
                                  mean_center::Bool=true,
                                  analytic_builder=nothing,
                                  A_of_x::Union{Nothing,Function}=nothing,
                                  store_response::Bool=true)
    local s_use::Function, Js_use::Function
    if analytic_builder !== nothing
        sJs = analytic_builder(θ)
        s_use, Js_use = sJs.s, sJs.Js
    else
        s_use = x -> model.s(x, θ)
        Js_use = x -> model.Js(x, θ)
    end
    model_v = _make_model_view(model, θ, s_use, Js_use)
    local responses, S
    if A_of_x === nothing
        responses, S = (nothing, nothing)
    else
        responses, S = build_responses(model_v, X, A_of_x; Δt=Δt, Tmax=Tmax, mean_center=mean_center)
    end
    return AnalyticJacobianEstimator(
        model_view=model_v, θ=collect(Float64.(θ)), X=Float64.(X),
        Δt=float(Δt), Tmax=float(Tmax), mean_center=mean_center,
        responses=responses, S=S)
end

"""
    build_gaussian_estimator(X, model::GFDTModel, θ; Δt, Tmax, mean_center=true,
                             A_of_x=nothing, store_response=true, jitter=1e-10)
"""
function build_gaussian_estimator(X::AbstractMatrix,
                                  model::GFDTModel,
                                  θ::AbstractVector;
                                  Δt::Real,
                                  Tmax::Real,
                                  mean_center::Bool=true,
                                  A_of_x::Union{Nothing,Function}=nothing,
                                  store_response::Bool=true,
                                  jitter::Real=1e-10)
    gs = gaussian_score_from_data(X; jitter=jitter)
    s_use, Js_use = gs.s, gs.Js
    model_v = _make_model_view(model, θ, s_use, Js_use)
    local responses, S
    if A_of_x === nothing
        responses, S = (nothing, nothing)
    else
        responses, S = build_responses(model_v, X, A_of_x; Δt=Δt, Tmax=Tmax, mean_center=mean_center)
    end
    return GaussianJacobianEstimator(
        model_view=model_v, θ=collect(Float64.(θ)), X=Float64.(X),
        Δt=float(Δt), Tmax=float(Tmax), mean_center=mean_center,
        responses=responses, S=S)
end

"""
    build_neural_estimator(X, model::GFDTModel, θ, cfg::NNTrainConfig;
                           Δt, Tmax, mean_center=true,
                           A_of_x=nothing, store_response=true)
"""
function build_neural_estimator(X::AbstractMatrix,
                                model::GFDTModel,
                                θ::AbstractVector,
                                cfg::NNTrainConfig;
                                Δt::Real,
                                Tmax::Real,
                                mean_center::Bool=true,
                                A_of_x::Union{Nothing,Function}=nothing,
                                store_response::Bool=true,
                                nn::Union{Nothing,Any}=nothing)
    D = size(X, 1)
    # Train or continue training using ScoreEstimation high-level wrapper.
    # Prefer the wrapper signature that accepts `nn` (if available),
    # otherwise fall back to older API and manual Jacobian computation.
    local nn_tr
    local jac_fn::Function
    local s_use::Function
    local Js_use::Function

    function _mk_s_js(nnA)
        sF = function (x::AbstractVector)
            Xb = reshape(Float32.(x), D, 1)
            y = -Array(nnA(Xb)) ./ Float32(cfg.σ)
            return Float64.(vec(y))
        end
        jF = function (x::AbstractVector)
            Xb = reshape(Float32.(x), D, 1)
            Jraw = jac_fn(Xb)
            if Jraw isa AbstractMatrix
                return Float64.(Jraw)
            elseif Jraw isa AbstractArray && ndims(Jraw) == 3
                return Float64.(Jraw[:, :, 1])
            elseif Jraw isa AbstractVector && eltype(Jraw) <: AbstractMatrix
                return Float64.(Jraw[1])
            else
                Jr = Array(Jraw)
                return ndims(Jr) == 3 ? Float64.(Jr[:, :, 1]) : Float64.(Jr)
            end
        end
        return sF, jF
    end
    # Determine epochs to use: scratch uses cfg.n_epochs; retrain uses cfg.epochs_re
    n_epochs_use = (nn === nothing || cfg.preprocessing) ? max(cfg.n_epochs, 0) : max(cfg.epochs_re, 0)
    # KGMM builds its random tree from a noisy draw. Rarely a leaf receives no
    # samples on the next EMA draw, and older ScoreEstimation releases expose
    # that as a one-column DimensionMismatch. Retrying rebuilds the stochastic
    # partition without changing the estimator or silently switching training
    # objectives. The outer experiment seed still makes the retry reproducible.
    train_result = nothing
    for attempt in 1:3
        try
            train_result = ScoreEstimation.train(
                X;
                preprocessing=cfg.preprocessing,
                σ=cfg.σ,
                neurons=cfg.neurons,
                n_epochs=n_epochs_use,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                use_gpu=cfg.use_gpu,
                verbose=cfg.verbose,
                kgmm_kwargs=cfg.kgmm_kwargs,
                divergence=false,
                probes=cfg.probes,
                rademacher=cfg.rademacher,
                jacobian=true,
                nn=nn,
            )
            break
        catch err
            retryable = cfg.preprocessing && err isa DimensionMismatch && attempt < 3
            retryable || rethrow()
            @warn "KGMM partition mismatch; rebuilding stochastic partition" attempt
        end
    end
    nn_tr, _train_losses, _val, _div_fn, jac_fn, _ = train_result
    s_use, Js_use = _mk_s_js(nn_tr)
    model_v = _make_model_view(model, θ, s_use, Js_use)
    local responses, S
    if A_of_x === nothing
        responses, S = (nothing, nothing)
    else
        responses, S = build_responses(model_v, X, A_of_x; Δt=Δt, Tmax=Tmax, mean_center=mean_center)
    end
    return NeuralJacobianEstimator(
        model_view=model_v, θ=collect(Float64.(θ)), X=Float64.(X),
        Δt=float(Δt), Tmax=float(Tmax), mean_center=mean_center,
        nn_method=nothing,
        responses=responses, S=S), nn_tr
end

"""
    build_finite_diff_estimator(simulator, θ, A_of_x; free_idx=1:length(θ),
                                h_rel=1e-4, h_abs=1e-6, scheme=:central, n_rep=1)
"""
function build_finite_diff_estimator(simulator::Function,
                                     θ::AbstractVector,
                                     A_of_x::Function;
                                     free_idx::AbstractVector{<:Integer}=collect(1:length(θ)),
                                     h_rel::Real=1e-4,
                                     h_abs::Real=1e-6,
                                     scheme::Symbol=:central,
                                     n_rep::Int=1)
    θf = collect(Float64.(θ))
    S = finite_difference_jacobian(simulator, θf, A_of_x; free_idx=collect(free_idx),
                                   h_rel=float(h_rel), h_abs=float(h_abs))
    return FiniteDiffJacobianEstimator(
        θ=θf, simulator=simulator,
        free_idx=collect(Int.(free_idx)), h_rel=float(h_rel), h_abs=float(h_abs),
        scheme=scheme, n_rep=Int(n_rep), S=S)
end
