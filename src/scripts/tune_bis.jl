using CSV
using CUDA
using FFTW
using JLD2
using GRASS
using FileIO
using Printf
using Revise
using LsqFit
using DataFrames
using Statistics
using EchelleCCFs
using Polynomials
using Distributions
using BenchmarkTools
using HypothesisTests

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
plotdir = abspath(string(figures))
datadir = abspath(string(data))

# get the template name
template_idx = tryparse(Int, ARGS[1])
lp = GRASS.LineProperties(exclude=["CI_5380", "NaI_5896"])
line_names = GRASS.get_name(lp)
template = line_names[template_idx]
println(">>> Template = " * template)

# get input data properties
line_species = GRASS.get_species(lp)
rest_wavelengths = GRASS.get_rest_wavelength(lp)
line_depths = GRASS.get_depth(lp)
line_names = GRASS.get_name(lp)
line_titles = replace.(line_names, "_" => " ")
line_files = GRASS.get_file(lp)

# set number of loops
Nloops = 8

# set number of levels tried
number_levels = 100

# set up array to hold BIS levels
b1_array = zeros(number_levels)
b2_array = zeros(number_levels)
b3_array = zeros(number_levels)
b4_array = zeros(number_levels)

# set up array to hold correlation coeffs
r_array = zeros(number_levels, Nloops)

# set up parameters for synthesis
Nt = 160
lines = [rest_wavelengths[template_idx]]
templates = [template]
blueshifts = zeros(length(lines))
depths = [line_depths[template_idx]]
resolution = 7e5

# synthesize the line
disk = DiskParams(Nt=Nt)
spec = SpecParams(lines=lines, depths=depths, templates=templates,
                  blueshifts=blueshifts, oversampling=3.0)

# do an initial synthesis to get the width (and precompile methods)
wavs0, flux0 = synthesize_spectra(spec, disk, verbose=false, use_gpu=true)
v_grid0, ccf0 = calc_ccf(wavs0, flux0, spec)
rvs0, sigs0= calc_rvs_from_ccf(v_grid0, ccf0)

# get width of line at ~92.5% flux
idxl, idxr = GRASS.find_wing_index(0.925, mean(flux0, dims=2)[:,1])

# get width in angstroms
width_ang = wavs0[idxr] - wavs0[idxl]

# convert to velocity
width_vel = GRASS.c_ms * width_ang / wavs0[argmin(flux0[:,1])]

# allocate memory that will be reused in ccf computation
Δv_step = 100.0
Δv_max = round((width_vel + 1e3)/100) * 100; @show Δv_max
v_grid = range(-Δv_max, Δv_max, step=Δv_step)
projection = zeros(length(spec.lambdas), 1)
proj_flux = zeros(length(spec.lambdas))
ccf1 = zeros(length(v_grid), Nt)

# iterate over BIS regions
for i in 1:number_levels
    # draw BIS levels
    b1 = rand(Uniform(0.10, 0.80), 1)[1]
    b2 = rand(Uniform(b1, 0.85), 1)[1]
    b3 = rand(Uniform(b2, 0.85), 1)[1]
    b4 = rand(Uniform(b3, 0.90), 1)[1]

    # set the values in the array
    b1_array[i] = b1
    b2_array[i] = b2
    b3_array[i] = b3
    b4_array[i] = b4

    # repeat for Nloop times
    for j in 1:Nloops
        wavs, flux = synthesize_spectra(spec, disk, verbose=false, use_gpu=true)

        # measure velocities
        GRASS.calc_ccf!(v_grid, projection, proj_flux, ccf1, wavs, flux, lines,
                        depths, resolution, Δv_step=Δv_step, Δv_max=Δv_max)
        rvs, sigs = calc_rvs_from_ccf(v_grid, ccf1)

        # measure bisector
        bis, int = GRASS.calc_bisector(wavs, flux, nflux=100, top=0.99)

        # convert bis to velocity scale
        for t in 1:Nt
            bis[:, t] = (bis[:,t] .- lines[1]) * GRASS.c_ms / lines[1]
        end

        # calculate summary statistics
        bis_inv_slope = GRASS.calc_bisector_inverse_slope(bis, int, b1=b1, b2=b2, b3=b3, b4=b4)

        # data to fit
        xdata = bis_inv_slope .- mean(bis_inv_slope)
        ydata = rvs .- mean(rvs)

        r_array[i,j] = Statistics.cor(xdata, ydata)
    end
end

# save the results to a file
datafile = string(abspath(joinpath(data, template * "_tune_bis.jld2")))
jldsave(datafile, b1=b1_array, b2=b2_array, b3=b3_array, b4=b4_array, r_array=r_array)
