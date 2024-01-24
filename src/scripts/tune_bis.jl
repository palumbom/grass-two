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

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
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
Nloops = 100

# set number of levels tried
number_levels = 400

# set up array to hold BIS levels
b1_array = zeros(number_levels)
b2_array = zeros(number_levels)
b3_array = zeros(number_levels)
b4_array = zeros(number_levels)

# set up array to hold curvature levels
c1_array = zeros(number_levels)
c2_array = zeros(number_levels)
c3_array = zeros(number_levels)
c4_array = zeros(number_levels)
c5_array = zeros(number_levels)
c6_array = zeros(number_levels)

# set up array to hold correlation coeffs
r_bis_array = zeros(number_levels, Nloops)
r_curve_array = zeros(number_levels, Nloops)

# get optimized depths
df = CSV.read(joinpath(datadir, "optimized_depth.csv"), DataFrame)

# set up parameters for synthesis
Nt = 400
lines = [rest_wavelengths[template_idx]]
templates = [template]
blueshifts = zeros(length(lines))
depths = [df[template_idx, "optimized_depth"]]
resolution = 7e5

# synthesize the line
disk = DiskParams(Nt=Nt)
spec = SpecParams(lines=lines, depths=depths, templates=templates,
                  blueshifts=blueshifts, oversampling=2.0)

# do an initial synthesis to get the width (and precompile methods)
wavs0, flux0 = synthesize_spectra(spec, disk, verbose=false, use_gpu=true)
v_grid0, ccf0 = calc_ccf(wavs0, flux0, spec)
rvs0, sigs0= calc_rvs_from_ccf(v_grid0, ccf0)

# get width of line at ~95% flux
idxl, idxr = GRASS.find_wing_index(0.95, mean(flux0, dims=2)[:,1])

# get width in angstroms
width_ang = wavs0[idxr] - wavs0[idxl]

# convert to velocity
width_vel = GRASS.c_ms * width_ang / wavs0[argmin(flux0[:,1])]

# get width in velocity for ccf
Δv_step = 100.0
Δv_max = round((width_vel + 1e3)/100) * 100
if Δv_max < 15e3
    Δv_max = 15e3
end
@show Δv_max

# allocate memory that will be reused in ccf computation
v_grid = range(-Δv_max, Δv_max, step=Δv_step)
projection = zeros(length(spec.lambdas), 1)
proj_flux = zeros(length(spec.lambdas))
ccf1 = zeros(length(v_grid), Nt)

# iterate over BIS regions
for i in 1:number_levels
    # draw BIS levels
    b1 = rand(Uniform(0.15, 0.75), 1)[1]
    b2 = rand(Uniform(b1 + 0.05, 0.85), 1)[1]
    b3 = rand(Uniform(b2, 0.85), 1)[1]
    b4 = rand(Uniform(b3 + 0.05, 0.95), 1)[1]

    # set the values in the array
    b1_array[i] = b1
    b2_array[i] = b2
    b3_array[i] = b3
    b4_array[i] = b4

    # draw curvature levels
    c1 = rand(Uniform(0.15, 0.65), 1)[1]
    c2 = rand(Uniform(c1 + 0.05, 0.75), 1)[1]

    c3 = rand(Uniform(c2, 0.75), 1)[1]
    c4 = rand(Uniform(c3 + 0.05, 0.85), 1)[1]

    c5 = rand(Uniform(c4, 0.85), 1)[1]
    c6 = rand(Uniform(c5 + 0.05, 0.95), 1)[1]

    # set the values in the array
    c1_array[i] = c1
    c2_array[i] = c2
    c3_array[i] = c3
    c4_array[i] = c4
    c5_array[i] = c5
    c6_array[i] = c6

    # repeat for Nloop times
    for j in 1:Nloops
        wavs, flux = synthesize_spectra(spec, disk, verbose=false, use_gpu=true)

        # measure velocities
        GRASS.calc_ccf!(v_grid, projection, proj_flux, ccf1, wavs, flux, lines,
                        depths, resolution, Δv_step=Δv_step, Δv_max=Δv_max,
                        mask_type=EchelleCCFs.GaussianCCFMask)
        rvs, sigs = calc_rvs_from_ccf(v_grid, ccf1, frac_of_width_to_fit=0.50)

        # ydata
        ydata = rvs .- mean(rvs)

        # # measure bisector
        # bis, int = GRASS.calc_bisector(wavs, flux, nflux=100, top=0.99)
        # # convert bis to velocity scale
        # for t in 1:Nt
        #     bis[:, t] = (bis[:,t] .- lines[1]) * GRASS.c_ms / lines[1]
        # end

        # measure bisector
        bis, int = GRASS.calc_bisector(v_grid, ccf1, nflux=100, top=0.99)

        # calculate summary statistics
        bis_inv_slope = GRASS.calc_bisector_inverse_slope(bis, int, b1=b1, b2=b2, b3=b3, b4=b4)
        bis_curvature = GRASS.calc_bisector_curvature(bis, int, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6)

        # data to fit
        xdata1 = bis_inv_slope .- mean(bis_inv_slope)
        r_bis_array[i,j] = Statistics.cor(xdata1, ydata)

        xdata2 = bis_curvature .- mean(bis_curvature)
        r_curve_array[i,j] = Statistics.cor(xdata2, ydata)
    end
end

# save the results to a file
jldsave(string(abspath(joinpath(data, template * "_tune_bis.jld2"))),
        b1=b1_array, b2=b2_array, b3=b3_array,  b4=b4_array,
        c1=c1_array, c2=c2_array, c3=c3_array, c4=c4_array,
        c5=c5_array, c6=c6_array, r_bis_array=r_bis_array,
        r_curve_array=r_curve_array)
