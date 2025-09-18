
"""
    moment_observables(u::Real)

Return the first four raw moments evaluated at scalar physical state `u`:
[1, u, u^2, u^3, u^4] excluding the constant 1; here we return [u, u^2, u^3, u^4].
"""
@inline function moment_observables(u::Real)
    u2 = u*u
    u3 = u2*u
    u4 = u2*u2
    return (u, u2, u3, u4)
end

"""
    indicator_observables_from_p(u_samples::AbstractVector, p::Real)

Given samples of the (physical) state `u_samples`, compute thresholds (α, β)
so that P(u ≥ α)=p and P(u ≤ β)=p and return two closures:
- I_ge(u) = 1 if u ≥ α else 0
- I_le(u) = 1 if u ≤ β else 0
along with the thresholds as a NamedTuple (α=..., β=...).
"""
function indicator_observables(th::NamedTuple)
    α, β = th.α, th.β
    I_ge = (u -> (u ≥ α ? 1.0 : 0.0))
    I_le = (u -> (u ≤ β ? 1.0 : 0.0))
    return I_ge, I_le
end

"""
    build_A_of_x(μ_phys, Σ_phys; use_moments, use_indicators, thresholds)

Construct an observable mapping `A_of_x(x_norm) -> Vector` where `x_norm` is a
normalized (dimension-1) state and the physical variable is `u = x_norm[1]*Σ_phys + μ_phys`.

Arguments:
  use_moments      :: symbols in (:m1,:m2,:m3,:m4)
  use_indicators   :: symbols subset of (:ge,:le)
  thresholds       :: NamedTuple with fields α and/or β (only required for the selected indicators)

Returns: (A_of_x, labels)
"""
function build_A_of_x(μ_phys::Real, Σ_phys::Real; use_moments=(:m1,:m2,:m3,:m4), use_indicators=(), thresholds::NamedTuple)
    moms = use_moments isa Symbol ? (use_moments,) : Tuple(use_moments)
    inds = use_indicators isa Symbol ? (use_indicators,) : Tuple(use_indicators)

    if length(inds) > 0
        α_sup = haskey(thresholds, :α) ? getfield(thresholds, :α) : nothing
        β_sup = haskey(thresholds, :β) ? getfield(thresholds, :β) : nothing
        if :ge in inds
            @assert α_sup !== nothing "Thresholds provided must include :α for :ge indicator"
            I_ge = (u -> (u ≥ α_sup ? 1.0 : 0.0))
        end
        if :le in inds
            @assert β_sup !== nothing "Thresholds provided must include :β for :le indicator"
            I_le = (u -> (u ≤ β_sup ? 1.0 : 0.0))
        end
    end

    fns = Vector{Function}(); labels = String[]
    for m in moms
        if m === :m1
            push!(fns, u->u);           push!(labels, "u")
        elseif m === :m2
            push!(fns, u->u^2);         push!(labels, "u^2")
        elseif m === :m3
            push!(fns, u->u^3);         push!(labels, "u^3")
        elseif m === :m4
            push!(fns, u->u^4);         push!(labels, "u^4")
        else
            error("Unknown moment symbol $(m)")
        end
    end
    for ind in inds
        if ind === :ge
            @assert I_ge !== nothing
            push!(fns, I_ge);           push!(labels, "1{u≥α}")
        elseif ind === :le
            @assert I_le !== nothing
            push!(fns, I_le);           push!(labels, "1{u≤β}")
        else
            error("Unknown indicator $(ind); use :ge or :le")
        end
    end

    A_of_x = x -> begin
        u = (x[1] * Σ_phys + μ_phys)
        vcat((f(u) for f in fns)...)
    end
    return A_of_x, labels
end

function stats_A(X::AbstractMatrix, A_of_x::Function)
    T = size(X,2)
    m = length(A_of_x(@view X[:,1]))
    A = zeros(Float64, m)
    @inbounds for t in 1:T
        A .+= A_of_x(@view X[:,t])
    end
    A ./= max(T, 1)
    return A
end

"""
    build_make_A_of_x(; use_moments, use_indicators, thresholds)

Return a closure `make_A_of_x(μ_phys, Σ_phys)` that produces `(A_of_x, labels)` by
internally calling `build_A_of_x(μ_phys, Σ_phys; ...)`. This is convenient in calibration
loops where μ, Σ are re-estimated each iteration.
"""
function build_make_A_of_x(; use_moments=(:m1,:m2,:m3,:m4), use_indicators=(), thresholds::NamedTuple)
    return (μ_phys, Σ_phys) -> build_A_of_x(μ_phys, Σ_phys; use_moments=use_moments, use_indicators=use_indicators, thresholds=thresholds)
end