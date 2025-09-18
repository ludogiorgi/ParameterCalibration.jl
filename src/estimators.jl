"""
    gaussian_score_from_data(X; jitter=1e-10)

Given samples X (d×T), return a NamedTuple with linear Gaussian-closure score:
  s(x)  = -C⁻¹ (x - μ)
  Js(x) = -C⁻¹ (constant)

Also returns μ and Cinv for downstream use.
"""
function gaussian_score_from_data(X::AbstractMatrix; jitter::Real=1e-10, max_tries::Int=5)
    μ = vec(mean(X, dims=2))
    Xc = X .- μ
    T = size(X,2)
    C = Symmetric((Xc * Xc') / max(T-1, 1))
    # Cholesky with jitter escalation
    jitter_local = jitter
    local chol::Union{Cholesky{Float64,Matrix{Float64}},Nothing} = nothing
    for k in 1:max_tries
        try
            chol = cholesky(Symmetric(Matrix(C) + jitter_local*I), check=true)
            break
        catch err
            if k == max_tries
                rethrow(err)
            end
            jitter_local *= 10
        end
    end
    @assert chol !== nothing
    # Build inverse explicitly via single solve (kept for API compatibility)
    Cinv = Symmetric(Matrix(chol \ I))
    s = function (x::AbstractVector)
        return -Cinv * (Float64.(x) .- μ)
    end
    Js = function (_x::AbstractVector)
        return Matrix(-Cinv)
    end
    return (s=s, Js=Js, μ=μ, Cinv=Cinv)
end

