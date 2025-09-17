@inline function _build_Mj!(Mcol::AbstractMatrix, Σ::AbstractMatrix, Λj::AbstractMatrix)
    mul!(Mcol, Σ, transpose(Λj))    # Mcol = Σ Λᵀ
    Mcol .+= Λj * transpose(Σ)      # Mcol += Λ Σᵀ
    return Mcol
end

function B_gfdt(model::GFDTModel, x)::Vector
    θ     = model.θ
    s     = model.s(x)
    Ψ     = model.dF_dθ(x, θ)               # d×p
    Σ     = model.Σ(x, θ)
    Λall  = model.dΣ_dθ(x, θ)               # d×d×p
    divΨ  = model.div_dF_dθ(x, θ)           # p
    divM  = model.divM(x, θ)                # d×p
    div2M = model.divdivM(x, θ)             # p

    d, p  = size(Ψ)
    out   = Vector{eltype(s)}(undef, p)

    Mcol  = Matrix{eltype(s)}(undef, d, d)
    tmpv  = similar(s)

    use_general = (model.mode == :general)
    if use_general
        @assert model.Js !== nothing "GFDTModel.Js must be provided in :general mode."
        Js = model.Js(x)                    # d×d
        JsT = transpose(Js)
        @inbounds for j in 1:p
            Λj    = @view Λall[:,:,j]
            Ψj    = @view Ψ[:,j]
            divMj = @view divM[:,j]

            _build_Mj!(Mcol, Σ, Λj)
            term1 = divΨ[j]
            term2 = -0.5 * div2M[j]
            term3 = -0.5 * sum(Mcol .* JsT)
            term4 =  dot(Ψj, s)
            term5 = -dot(divMj, s)
            mul!(tmpv, Mcol, s)
            term6 = -0.5 * dot(s, tmpv)
            out[j] = term1 + term2 + term3 + term4 + term5 + term6
        end
    else
        # :isotropic -> assume M_j = c_j I with c_j = tr(M_j)/d, use divergence of score
        divs = model.divs(x)
        nrm2s = dot(s, s)
        @inbounds for j in 1:p
            Λj    = @view Λall[:,:,j]
            Ψj    = @view Ψ[:,j]
            divMj = @view divM[:,j]

            _build_Mj!(Mcol, Σ, Λj)
            cj = tr(Mcol) / d

            term1 = divΨ[j]
            term2 = -0.5 * div2M[j]
            term3 = -0.5 * (cj * divs)
            term4 =  dot(Ψj, s)
            term5 = -dot(divMj, s)
            term6 = -0.5 * (cj * nrm2s)
            out[j] = term1 + term2 + term3 + term4 + term5 + term6
        end
    end
    return out
end
