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
number_levels = 750

# generate bis array
bis_array = rand(Uniform(0.15, 0.95), 4, number_levels)
curve_array = rand(Uniform(0.15, 0.95), 6, number_levels)
for i in 1:number_levels
    # sort trial
    bis_array[:, i] .= sort(view(bis_array, :, i))
    curve_array[:, i] .= sort(view(curve_array, :, i))

    # determine if we need to iterate bis
    d1 = bis_array[2,i] - bis_array[1,i]
    d2 = bis_array[4,i] - bis_array[3,i]

    iter_bis = ((d1 >= 0.05) & (d2 >= 0.05))
    while !iter_bis
        bis_array[:, i] .= sort(rand(Uniform(0.15, 0.95), 4))

        d1 = bis_array[2,i] - bis_array[1,i]
        d2 = bis_array[4,i] - bis_array[3,i]

        iter_bis = ((d1 >= 0.05) & (d2 >= 0.05))
    end

    # determine if we need to iterate curve
    d1 = curve_array[2,i] - curve_array[1,i]
    d2 = curve_array[4,i] - curve_array[3,i]
    d3 = curve_array[6,i] - curve_array[5,i]

    iter_curve = ((d1 >= 0.05) & (d2 >= 0.05) & (d3 >= 0.05))
    while !iter_curve
        curve_array[:, i] .= sort(rand(Uniform(0.15, 0.95), 6))

        d1 = curve_array[2,i] - curve_array[1,i]
        d2 = curve_array[4,i] - curve_array[3,i]
        d3 = curve_array[6,i] - curve_array[5,i]

        iter_curve = ((d1 >= 0.05) & (d2 >= 0.05) & (d3 >= 0.05))
    end
end

# set up array to hold correlation coeffs
r_bis_array = zeros(number_levels, Nloops)
r_curve_array = zeros(number_levels, Nloops)

# get optimized depths
df = CSV.read(joinpath(datadir, "optimized_depth.csv"), DataFrame)

# set up parameters for synthesis
Nt = 160
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
    b1 = bis_array[1, i]
    b2 = bis_array[2, i]
    b3 = bis_array[3, i]
    b4 = bis_array[4, i]

    # set the values in the array
    c1 = curve_array[1, i]
    c2 = curve_array[2, i]
    c3 = curve_array[3, i]
    c4 = curve_array[4, i]
    c5 = curve_array[5, i]
    c6 = curve_array[6, i]

    # repeat for Nloop times
    for j in 1:Nloops
        @show j
        wavs, flux = synthesize_spectra(spec, disk, verbose=false, use_gpu=true)

        # measure velocities
        projection .= 0.0
        proj_flux .= 0.0
        ccf1 .= 0.0
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
        bis_array=bis_array, curve_array=curve_array, r_bis_array=r_bis_array,
        r_curve_array=r_curve_array)
