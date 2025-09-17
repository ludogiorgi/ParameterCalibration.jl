function _spd_solve!(A::AbstractMatrix, b::AbstractVector; jitter::Real=1e-10)
    @assert size(A,1)==size(A,2)==length(b)
    local jit = float(jitter)
    # Try increasing jitter up to 1e4× if needed
    for _try in 1:6
        @inbounds for i in axes(A,1)
            A[i,i] += jit
        end
        try
            F = cholesky!(Symmetric(A, :U))
            return F \ b
        catch err
            if err isa LinearAlgebra.PosDefException
                jit *= 10
                continue
            else
                rethrow()
            end
        end
    end
    # Last resort: Tikhonov regularization + generic solve
    @inbounds for i in axes(A,1)
        A[i,i] += max(jit, 1e-6)
    end
    return A \ b
end
using DSP: xcorr

"""
    xcorr_one_sided(x, y, Kmax)

One-sided raw cross-correlation at nonnegative lags using `DSP.xcorr`.
Returns c[k+1] = (1/(n-k)) * sum_{t=1}^{n-k} x[t+k] * y[t] for k=0..K,
matching the repository’s original unbiased normalization by available pairs.
"""
function xcorr_one_sided(x::StridedVector{T}, y::StridedVector{T}, Kmax::Int) where {T<:Real}
    n = length(x); @assert length(y) == n
    K = min(Kmax, n-1)
    # DSP.xcorr returns lags -(n-1):(n-1); center at index n
    c_full = xcorr(x, y)
    @assert length(c_full) == 2n - 1
    c_pos = @view c_full[n:n+K]  # length K+1, lags 0..K
    # Unbiased normalization by number of overlapping pairs per lag
    out = similar(x, K+1)
    @inbounds for k in 0:K
        out[k+1] = c_pos[k+1] / (n - k)
    end
    return out
end
