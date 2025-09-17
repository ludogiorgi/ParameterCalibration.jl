"""
This example script has been refactored into two stages:

  1. `reduced_1d_compute.jl` – performs all simulations, estimator construction, and
     calibration runs, storing numerical outputs to an HDF5 file.
  2. `reduced_1d_plot.jl` – loads the stored results and generates publication-quality
     figures using GLMakie.

New workflow (from repository root):

  julia --project=examples examples/reduced_1d_compute.jl
  julia --project=examples examples/reduced_1d_plot.jl

This legacy single-script version was replaced to improve reproducibility,
separation of concerns, and to minimize heavy recomputation when only plots
need updating.

If you intended to run the full pipeline directly, execute the two commands above.
"""

println("[ParameterCalibration] The monolithic 'reduced_1d.jl' example has been split.")
println("Run the computation stage:")
println("  julia --project=examples examples/reduced_1d_compute.jl")
println("Then generate figures:")
println("  julia --project=examples examples/reduced_1d_plot.jl")

