using CSV
using CUDA
using GRASS
using Printf
using Revise
using PyCall
using DataFrames
using Statistics
using EchelleCCFs
using Polynomials
using BenchmarkTools
using HypothesisTests

include(joinpath(abspath(@__DIR__), "paths.jl"))
datadir = abspath(string(data))
outdir = abspath(joinpath(string(data)), "mean_spec")
if !isdir(outdir)
    mkdir(outdir)
end

# get line properties
lp = GRASS.LineProperties()
line_species = GRASS.get_species(lp)
rest_wavelengths = GRASS.get_rest_wavelength(lp)
line_depths = GRASS.get_depth(lp)
line_names = GRASS.get_name(lp)

# get optimized depths
df = CSV.read(joinpath(datadir, "optimized_depth.csv"), DataFrame)

# make some spectra
for (idx, file) in enumerate(lp.file)
    # set up paramaters for spectrum
    Nt = 250
    lines = [rest_wavelengths[idx]]
    depths = [df[idx, "optimized_depth"]]
    templates = [file]
    variability = repeat([true], length(lines))
    blueshifts = zeros(length(lines))
    resolution = 7e5
    seed_rng = false

    disk = DiskParams(Nt=Nt)
    spec1 = SpecParams(lines=lines, depths=depths, variability=variability,
                       blueshifts=blueshifts, templates=templates,
                       resolution=resolution, oversampling=2.0)
    wavs_out, flux_out = synthesize_spectra(spec1, disk, seed_rng=seed_rng,
                                            verbose=true, use_gpu=true)

    flux_out = dropdims(mean(flux_out, dims=2), dims=2)

    df_out = DataFrame("wavs"=>wavs_out, "flux"=>flux_out)

    CSV.write(joinpath(outdir, line_names[idx] * ".csv"), df_out)
end
