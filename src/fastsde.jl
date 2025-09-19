"""
    simulate(u0, drift!, sigma!; spec::SimSpec)

Wrapper around FastSDE.evolve returning a trajectory matrix `X::Matrix{Float64}`
of size d×T after burn-in and downsampling by `resolution`. Supports ensemble
size `n_ens`. Common random numbers across calls can be achieved by fixing
`spec.seed` in the provided `spec` (the same seed yields identical noise).

Notes:
- Requires user-provided `drift!` and `sigma!` matching FastSDE API.
- For full-matrix noise, pass an appropriate `sigma!` and `sigma_inplace=false`.

 Decision: Keep this thin wrapper instead of calling `FastSDE.evolve` directly.
 Reasons: (1) centralize type conversions to `Float64` for consistent downstream
 numerics; (2) pin a stable, repository-specific API via `SimSpec` so the rest
 of the code doesn’t depend on FastSDE call signatures; (3) single place to
 extend behavior (e.g., noise control, CRN) without touching callers.
"""
function simulate(u0::AbstractVector, drift!::Function, sigma!::Function; spec::SimSpec)
    kwargs = (
        timestepper = spec.timestepper,
        resolution   = spec.resolution,
        sigma_inplace= spec.sigma_inplace,
        n_ens        = spec.n_ens,
        seed         = spec.seed,
    )
    if spec.lb !== nothing && spec.gb !== nothing
        lbv = spec.lb; gbv = spec.gb
        X = FastSDE.evolve(
            collect(Float64.(u0)),
            spec.dt,
            spec.Nsteps,
            drift!,
            sigma!;
            kwargs...,
            boundary = (lbv, gbv),
        )
        return Float64.(X)
    else
        X = FastSDE.evolve(
            collect(Float64.(u0)),
            spec.dt,
            spec.Nsteps,
            drift!,
            sigma!;
            kwargs...,
        )
        return Float64.(X)
    end
end
