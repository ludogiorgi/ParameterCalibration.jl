# Observables and threshold estimators

module Observables

using Statistics

export moment_observables, indicator_observables_from_p, estimate_quantile_threshold, build_A_of_x, thresholds_from_p

"""
    estimate_quantile_threshold(u::AbstractVector, p::Real)

Return α such that P(U ≥ α) ≈ p on samples `u` (empirical (1-p)-quantile).
Equivalent β such that P(U ≤ β) ≈ p is the empirical p-quantile.
"""
function estimate_quantile_threshold(u::AbstractVector, p::Real)
    @assert 0.0 ≤ p ≤ 1.0 "p must be in [0,1]"
    # For indicator 1{u ≥ α} with P(u ≥ α)=p → α = quantile(u, 1-p)
    α = quantile(collect(Float64.(u)), 1 - float(p))
    β = quantile(collect(Float64.(u)), float(p))
    return (α=α, β=β)
end

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
Also return (α, β).
"""
function indicator_observables_from_p(u_samples::AbstractVector, p::Real)
    th = estimate_quantile_threshold(u_samples, p)
    α, β = th.α, th.β
    I_ge = (u -> (u ≥ α ? 1.0 : 0.0))
    I_le = (u -> (u ≤ β ? 1.0 : 0.0))
    return I_ge, I_le, (α=α, β=β)
end

"""
    thresholds_from_p(u_samples, p) -> (α=..., β=...)

Convenience wrapper returning only the (α, β) thresholds without constructing
indicator closures. Useful when you want to precompute thresholds (e.g. from
an initial parameter guess) and later pass them to `build_A_of_x` via the
`thresholds` keyword so that they remain fixed throughout calibration.
"""
function thresholds_from_p(u_samples::AbstractVector, p::Real)
    t = estimate_quantile_threshold(u_samples, p)
    return (α=t.α, β=t.β)
end

"""
    build_A_of_x(μ_phys, Σ_phys; use_moments, use_indicators, u_samples, thresholds, p)

Construct an observable mapping `A_of_x(x_norm) -> Vector` where `x_norm` is a
normalized state and the physical variable is `u = x_norm[1]*Σ_phys + μ_phys`.

Usage modes for indicator thresholds:
1. Preferred (fixed thresholds): supply `thresholds = (α=..., β=...)` (one or both
   fields as needed for the chosen indicators) obtained e.g. via
   `thresholds_from_p(u_samples, p0)` using an initial parameter guess.
2. Backward-compatible: omit `thresholds` and pass `p` plus `u_samples` to compute
   thresholds on the fly (DEPRECATED path – emits a warning). This reproduces the
   previous behaviour but is not stable if the underlying distribution changes.

Arguments:
  use_moments      :: symbols in (:m1,:m2,:m3,:m4)
  use_indicators   :: symbols subset of (:ge,:le)
  u_samples        :: required only if computing thresholds from `p`
  thresholds       :: NamedTuple with fields α and/or β. If provided, overrides `p`.
  p                :: probability level (deprecated path if `thresholds` not supplied)

Returns: (A_of_x, labels, thresholds_used)
"""
function build_A_of_x(μ_phys::Real, Σ_phys::Real; use_moments=(:m1, :m2, :m3, :m4), use_indicators=(),
                      u_samples::AbstractVector=nothing, thresholds::Union{Nothing,NamedTuple}=nothing,
                      p::Real=0.5)
    # Normalize selector arguments to tuples
    moms = use_moments isa Symbol ? (use_moments,) : Tuple(use_moments)
    inds = use_indicators isa Symbol ? (use_indicators,) : Tuple(use_indicators)
    
    # Prepare indicator functions if requested
    local I_ge::Union{Nothing,Function} = nothing
    local I_le::Union{Nothing,Function} = nothing
    local αβ::Union{Nothing,NamedTuple} = nothing

    if length(inds) > 0
        if thresholds !== nothing
            # Use supplied thresholds (may contain only α or only β depending on indicators)
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
            αβ = (α = α_sup, β = β_sup)
        else
            # Deprecated path: compute thresholds from p each call
            @assert u_samples !== nothing "u_samples must be provided to compute indicator thresholds"
            @warn "build_A_of_x: using keyword p to compute thresholds dynamically is deprecated; precompute via thresholds_from_p and pass thresholds=... instead." p
            I_ge, I_le, αβ = indicator_observables_from_p(u_samples, p)
        end
    end

    # Build ordered list of functions and labels
    fns = Vector{Function}()
    labels = String[]
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
    return A_of_x, labels, αβ
end

end # module
