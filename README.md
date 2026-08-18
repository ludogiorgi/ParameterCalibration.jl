# ParameterCalibration.jl

Julia implementation of statistical parameter calibration through the
generalized fluctuation-dissipation theorem (GFDT). The package constructs
parameter-sensitivity estimators from unperturbed trajectories and supports
analytic, Gaussian-score, neural-score, and finite-difference comparisons.

This public repository contains only the release code and lightweight inputs.
The manuscript, submission records, internal project context, generated runs,
and machine-specific paths are maintained separately.

## Requirements

- Julia 1.11

The committed `Manifest.toml` pins the package environment used for the release.

## Install and load

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using ParameterCalibration
```

## Reproducibility examples

Two curated systems are included:

- `examples/reduced_1d_compute.jl` and `examples/reduced_1d_plot.jl`
- `examples/triad_compute.jl` and `examples/triad_plot.jl`

Prepare their dedicated environment from the repository root:

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate(); Pkg.develop(path=".")'
julia --project=examples examples/reduced_1d_compute.jl
julia --project=examples examples/reduced_1d_plot.jl
```

The committed text files under `examples/data/` are small reference inputs.
Generated HDF5 data and figures are intentionally ignored.

## Repository contents

- `src/`: package implementation
- `config/`: final example configurations
- `examples/`: final reduced one-dimensional and triad workflows
- `test/`: source, metadata, and public-boundary checks

## Citation

See `CITATION.cff`. The software is released under the BSD 3-Clause License.
