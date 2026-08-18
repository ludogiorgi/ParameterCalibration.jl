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

"""
    xcorr_matrix_one_sided(X, Y, Kmax)

Compute every one-sided cross-correlation between rows of `X` and rows of `Y`
using shared batched FFTs.  The result satisfies
`C[i,j,k+1] = sum_t X[i,t+k]Y[j,t]/(N-k)` for lags `k=0:Kmax`.

Unlike repeatedly calling `xcorr_one_sided`, this transforms each input row
once instead of once per row pair.  This matters for GFDT/LR response matrices,
where the same observable and conjugate-variable series are reused many times.
"""
function xcorr_matrix_one_sided(X::AbstractMatrix{<:Real},
                                Y::AbstractMatrix{<:Real},
                                Kmax::Integer)
    m, N = size(X)
    p, Ny = size(Y)
    N == Ny || throw(DimensionMismatch("X and Y must have the same number of samples"))
    N > 0 || throw(ArgumentError("time series must be nonempty"))
    K = clamp(Int(Kmax), 0, N - 1)
    nfft = nextprod((2, 3, 5, 7), 2N - 1)

    # Time is the first/contiguous dimension so FFTW can transform columns
    # efficiently.  Zero padding prevents circular wraparound for all lags.
    Xpad = zeros(Float64, nfft, m)
    Ypad = zeros(Float64, nfft, p)
    @views Xpad[1:N, :] .= transpose(X)
    @views Ypad[1:N, :] .= transpose(Y)
    FX = rfft(Xpad, 1)
    FY = rfft(Ypad, 1)

    correlations = Array{Float64}(undef, m, p, K + 1)
    spectrum = Vector{ComplexF64}(undef, size(FX, 1))
    @inbounds for i in 1:m, j in 1:p
        @views @. spectrum = FX[:, i] * conj(FY[:, j])
        c = irfft(spectrum, nfft)
        for k in 0:K
            correlations[i, j, k + 1] = c[k + 1] / (N - k)
        end
    end
    return correlations
end
