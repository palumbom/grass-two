# imports
using JLD2
using CUDA
using GRASS
using Printf
using FileIO
using Revise
using Statistics
using EchelleCCFs
using Distributions
using BenchmarkTools

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
const datafile = string(abspath(joinpath(data, "gpu_accuracy.jld2")))

# set up paramaters for spectrum
lines = [5500.0, 5500.85, 5501.4, 5502.20, 5502.5, 5503.05]
depths = [0.75, 0.4, 0.65, 0.55, 0.25, 0.7]
templates = ["FeI_5434", "FeI_5576", "FeI_6173", "FeI_5432", "FeI_5576", "FeI_5250.6"]
blueshifts = zeros(length(lines))
resolution = 7e5
buffer = 0.6

# create composite types
disk = DiskParams(Nt=1)
spec = SpecParams(lines=lines, depths=depths, templates=templates,
                  blueshifts=blueshifts, resolution=resolution, buffer=buffer)

# get the spectra
println(">>> Performing CPU synthesis...")
wavs_cpu64, flux_cpu64 = synthesize_spectra(spec, disk, seed_rng=true, verbose=true, use_gpu=false)

println(">>> Performing GPU synthesis (double precision)...")
wavs_gpu64, flux_gpu64 = synthesize_spectra(spec, disk, seed_rng=true, verbose=true, use_gpu=true)

println(">>> Performing GPU synthesis (single precision)...")
wavs_gpu32, flux_gpu32 = synthesize_spectra(spec, disk, seed_rng=true, verbose=true, use_gpu=true, precision=Float32)

# slice output
flux_cpu64 = dropdims(mean(flux_cpu64, dims=2), dims=2)
flux_gpu64 = dropdims(mean(flux_gpu64, dims=2), dims=2)
flux_gpu32 = dropdims(mean(flux_gpu32, dims=2), dims=2)

# get flux residuals
resids64 = flux_cpu64 .- flux_gpu64
resids32 = flux_cpu64 .- flux_gpu32

# compute velocities
v_grid, ccf1 = calc_ccf(wavs_cpu64, flux_cpu64, spec, normalize=true, mask_type=EchelleCCFs.GaussianCCFMask)
rvs_cpu64, sigs_cpu64 = calc_rvs_from_ccf(v_grid, ccf1, frac_of_width_to_fit=0.5)

v_grid, ccf1 = calc_ccf(wavs_gpu64, flux_gpu64, spec, normalize=true, mask_type=EchelleCCFs.GaussianCCFMask)
rvs_gpu64, sigs_gpu64 = calc_rvs_from_ccf(v_grid, ccf1, frac_of_width_to_fit=0.5)

v_grid, ccf1 = calc_ccf(wavs_gpu32, flux_gpu32, spec, normalize=true, mask_type=EchelleCCFs.GaussianCCFMask)
rvs_gpu32, sigs_gpu32 = calc_rvs_from_ccf(v_grid, ccf1, frac_of_width_to_fit=0.5)

# get velocity residuals
v_resid64 = rvs_cpu64 - rvs_gpu64
v_resid32 = rvs_cpu64 - rvs_gpu32

# report some diagnostics
println()
@show maximum(abs.(resids64))
@show maximum(abs.(resids32))

println()
@show mean(v_resid64)
@show mean(v_resid32)

# write to disk
jldsave(datafile, resids64=resids64, resids32=resids32,
        v_resids64=v_resid64, v_resid32=v_resid32,
        wavs_cpu64=wavs_cpu64, flux_cpu64=flux_cpu64,
        wavs_gpu64=wavs_cpu64, flux_gpu64=flux_cpu64,
        wavs_gpu32=wavs_cpu64, flux_gpu32=flux_cpu64)
