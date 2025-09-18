"""
    finite_difference_jacobian(simulator, θ, A_of_x; free_idx=1:length(θ), h_rel=1e-4, h_abs=1e-6)

Compute the parameter Jacobian S = ∂E[A]/∂θ for arbitrary models by central
finite differences. `simulator` is a closure `(θ)->X::d×T`. `A_of_x` maps a
state vector to an m-vector of observables. Returns an m×p matrix.

Notes:
- Uses independent simulations per ±h call. For CRN, bake noise sharing into
  `simulator` if desired.
"""
function finite_difference_jacobian(simulator::Function,
                                    θ::AbstractVector,
                                    A_of_x::Function;
                                    free_idx::AbstractVector{<:Integer}=collect(1:length(θ)),
                                    h_rel::Real=1e-2,
                                    h_abs::Real=1e-4)
    θ0 = collect(Float64.(θ))
    p  = length(θ0)
    free = collect(free_idx)
    # Baseline mean (not strictly required but can be useful to return)
    X0 = simulator(θ0)
    m = length(A_of_x(@view X0[:,1]))
    S = zeros(Float64, m, p)
    for j in free
        θj = θ0[j]
        h  = max(float(h_abs), float(h_rel) * max(abs(θj), 1.0))
        θp = copy(θ0); θp[j] = θj + h
        θm = copy(θ0); θm[j] = θj - h
        Xp = simulator(θp)
        Xm = simulator(θm)
        # Means of A over trajectories
        Ap = zeros(Float64, m); Am = zeros(Float64, m)
        Tp = size(Xp,2); Tm = size(Xm,2)
        @inbounds for t in 1:Tp
            Ap .+= A_of_x(@view Xp[:,t])
        end
        @inbounds for t in 1:Tm
            Am .+= A_of_x(@view Xm[:,t])
        end
        Ap ./= max(Tp, 1); Am ./= max(Tm, 1)
        @views S[:,j] .= (Ap .- Am) ./ (2h)
    end
    return S
end