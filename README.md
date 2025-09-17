# ParameterCalibration

Julia tools to assemble GFDT conjugate observables B(x), build sensitivity matrices S from trajectories, and perform Newton-style parameter updates. Designed for arbitrary dimension d and general drift/diffusion parameterizations.

## Install

- `julia --project=.` then: `using Pkg; Pkg.instantiate()`

## Quick Start

```
using ParameterCalibration

# Define a model (schematic example)
model = GFDTModel(
    s = x -> s(x),            # score
    divs = x -> 0.0,          # only used for :isotropic mode
    Js = x -> Js(x),          # ∇s, used for :general mode
    F = (x,θ) -> F(x,θ),      # drift
    Σ = (x,θ) -> Σ(x,θ),      # diffusion
    dF_dθ = (x,θ) -> dF_dθ(x,θ),
    dΣ_dθ = (x,θ) -> dΣ_dθ(x,θ),
    div_dF_dθ = (x,θ) -> div_dF_dθ(x,θ),
    divM = (x,θ) -> divM(x,θ),
    divdivM = (x,θ) -> divdivM(x,θ),
    θ = θ0,
    mode = :general,
)

# Conjugate observable
B = B_gfdt(model, x)

# Sensitivity matrix for statistics A(x)
responses, S = build_responses(model, X, A_of_x; Δt=Δt, Tmax=Tmax)

# One-step Newton update
A, G = stats_AG(A_of_x, X_obs, X_mod)
W = weight_inverse_cov(A_of_x, X_obs)
Γ, _ = make_regularizer(model.θ; λ=0.0)
Δθ, diag = newton_step(S, W, Γ, G, A)
```

## Score Estimators

```
gs = gaussian_score_from_data(X)  # X is d×T samples
s, Js = gs.s, gs.Js               # closures usable in GFDTModel
```

## Examples

### Reduced 1D Calibration Workflow (Refactored)

The previous monolithic example script has been split for better reproducibility and
faster iteration when only figures need updating.

1. Heavy computation & data export: `examples/reduced_1d_compute.jl`
2. Plotting & figure generation: `examples/reduced_1d_plot.jl`

Each example now uses a dedicated environment under `examples/` so optional
dependencies (e.g. GLMakie, HDF5) do not pollute the core package dependency set.

### Running the example

From the repository root:

```
julia --project=examples -e 'using Pkg; Pkg.instantiate(); Pkg.develop(path=".")'
julia --project=examples examples/reduced_1d_compute.jl   # generates HDF5 data file
julia --project=examples examples/reduced_1d_plot.jl      # creates figures in ./figures
```

Artifacts:
- Numerical results: `examples/data/reduced_1d_results.h5`
- Figures: `figures/score_jacobian_reduced1d.png`, `figures/conjugate_B_reduced1d.png`,
  `figures/response_functions_reduced1d.png`, `figures/calibration_results_reduced1d.png`,
  `figures/parameter_trajectories_reduced1d.png`

The legacy entry point `examples/reduced_1d.jl` now only prints a message directing
users to the two-stage workflow above.

## Requirements

- Julia 1.11.x (compat set in `Project.toml`)

Notes:
- `Manifest.toml` pins exact versions.
