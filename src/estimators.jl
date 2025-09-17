"""
    gaussian_score_from_data(X; jitter=1e-10)

Given samples X (d×T), return a NamedTuple with linear Gaussian-closure score:
  s(x)  = -C⁻¹ (x - μ)
  Js(x) = -C⁻¹ (constant)

Also returns μ and Cinv for downstream use.
"""
function gaussian_score_from_data(X::AbstractMatrix; jitter::Real=1e-10)
    μ = vec(mean(X, dims=2))
    Xc = X .- μ
    T = size(X,2)
    C = Symmetric((Xc * Xc') / max(T-1, 1))
    Cj = Symmetric(Matrix(C) + jitter*I)
    Cinv = inv(Cj)
    s = function (x::AbstractVector)
        return -Cinv * (Float64.(x) .- μ)
    end
    Js = function (_x::AbstractVector)
        return Matrix(-Cinv)
    end
    return (s=s, Js=Js, μ=μ, Cinv=Cinv)
end

