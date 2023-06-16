# imports
using CSV
using CUDA
using JLD2
using GRASS
using Printf
using Revise
using FileIO
using DataFrames
using Statistics
using EchelleCCFs

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
const datafile = string(abspath(joinpath(data, "spectra_for_bin.jld2")))

# get input data properties
lp = GRASS.LineProperties(exclude=["CI_5380", "NaI_5896"])
line_species = GRASS.get_species(lp)
rest_wavelengths = GRASS.get_rest_wavelength(lp)
line_depths = GRASS.get_depth(lp)
line_names = GRASS.get_name(lp)
line_titles = replace.(line_names, "_" => " ")
line_files = GRASS.get_file(lp)

# draw random line depths and centers
nlines = 10
# lines = Float64[]
lines = range(5200, 5800, length=nlines*length(rest_wavelengths))
depths = Float64[]
templates = String[]
for i in eachindex(rest_wavelengths)
    # ltemp = rand(Uniform(minimum(rest_wavelengths), maximum(rest_wavelengths)), nlines)
    # ltemp = rand(Uniform(5200, 5400), nlines)
    dtemp = rand(Normal(line_depths[i]-0.05, 0.05), nlines)
    # push!(lines, ltemp...)
    push!(depths, dtemp...)
    push!(templates, repeat([line_files[i]], nlines)...)
end

# re-shuffle template order by generating random indices
idx = randperm(length(lines))
depths = depths[idx]
templates = templates[idx]

# synthesize a spectrum
N = 132
Nt = 1000
variability = trues(length(lines))
resolution = 7e5
seed_rng = true

# do a quick synthesis for precompilation purposes
disk0 = DiskParams(N=N, Nt=10)
spec0 = SpecParams(lines=lines[1:2], depths=depths[1:2],
                   variability=variability[1:2], templates=templates[1:2],
                   resolution=resolution)
wavs0, flux0 = synthesize_spectra(spec0, disk0, seed_rng=true, verbose=true, use_gpu=true)


# now do the actual synthesis
disk = DiskParams(N=N, Nt=Nt)
spec1 = SpecParams(lines=lines, depths=depths, variability=variability, templates=templates, resolution=resolution)
wavs, flux = synthesize_spectra(spec1, disk, seed_rng=true, verbose=true, use_gpu=true)

# write it to a JLD2
jldsave(datafile, wavs=wavs, flux=flux, templates=spec1.templates, lines=spec1.lines, depths=spec1.depths)
