# imports
using CSV
using CUDA
using JLD2
using GRASS
using Printf
using Revise
using FileIO
using Random
using DataFrames
using Statistics
using EchelleCCFs
using Distributions

# make sure there is a GPU
@assert CUDA.functional()

# get the name of template from the command line args
template_idx = tryparse(Int, ARGS[1])
lp = GRASS.LineProperties(exclude=["CI_5380", "NaI_5896"])
line_names = GRASS.get_name(lp)
template = line_names[template_idx]
println(">>> Template = " * template)

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
datafile = string(abspath(joinpath(data, template * "_picket_fence.jld2")))

# get input data properties
lp = GRASS.LineProperties(exclude=["CI_5380", "NaI_5896"])
line_species = GRASS.get_species(lp)
rest_wavelengths = GRASS.get_rest_wavelength(lp)
line_depths = GRASS.get_depth(lp)
line_names = GRASS.get_name(lp)
line_titles = replace.(line_names, "_" => " ")
line_files = GRASS.get_file(lp)

# get index of line template I want
idx = findfirst(x -> contains(x, template), line_names)

# parameters for spectrum
nlines = 250
lines = collect(range(4800, 6200, length=nlines)) .+ 0.1 .* rand(nlines)
depths = repeat([line_depths[idx]], nlines)
blueshifts = zeros(nlines)
templates = repeat([line_files[idx]], nlines)

# synthesize the spectra
Nt = 1000
variability = trues(length(lines))
resolution = 7e5
seed_rng = true

# do a quick synthesis for to make sure it's precompiled
disk0 = DiskParams(Nt=10)
spec0 = SpecParams(lines=lines[1:2], depths=depths[1:2],
                   blueshifts=blueshifts[1:2], variability=variability[1:2],
                   templates=templates[1:2], resolution=resolution)
wavs0, flux0 = synthesize_spectra(spec0, disk0, seed_rng=seed_rng,
                                  verbose=true, use_gpu=true)

# now do the actual synthesis
disk = DiskParams(Nt=Nt)
spec1 = SpecParams(lines=lines, depths=depths, blueshifts=blueshifts,
                   variability=variability, templates=templates,
                   resolution=resolution)
wavs, flux = synthesize_spectra(spec1, disk, seed_rng=seed_rng,
                                verbose=true, use_gpu=true)

# write it to a JLD2
jldsave(datafile, wavs=wavs, flux=flux,
        templates=spec1.templates, lines=spec1.lines,
        depths=spec1.depths)

