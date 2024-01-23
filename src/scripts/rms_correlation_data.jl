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
using BenchmarkTools
using HypothesisTests

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
# include("paths.jl")
plotdir = string(figures)
datadir = string(data)
datafile = string(abspath(joinpath(data, "rms_table.csv")))

# get lines to construct templates
lp = GRASS.LineProperties()
name = GRASS.get_name(lp)
λrest = GRASS.get_rest_wavelength(lp)
depth = GRASS.get_depth(lp)
lfile = GRASS.get_file(lp)
llevel = GRASS.get_lower_level(lp)
ulevel = GRASS.get_upper_level(lp)

# set up data frame to hold things
df = DataFrame(line=name, raw_rms=zeros(length(name)), raw_rms_sig=zeros(length(name)),
               bis_inv_slope_corr=zeros(length(name)), bis_inv_slope_rms=zeros(length(name)), bis_inv_slope_sig=zeros(length(name)),
               bis_span_corr=zeros(length(name)), bis_span_rms=zeros(length(name)), bis_span_sig=zeros(length(name)),
               bis_curve_corr=zeros(length(name)), bis_curve_rms=zeros(length(name)), bis_curve_sig=zeros(length(name)),
               bis_tuned_corr=zeros(length(name)), bis_tuned_rms=zeros(length(name)), bis_tuned_sig=zeros(length(name)))

holder_names = ["bis_inv_slope", "bis_span", "bis_curve", "bis_tuned"]

# set number of loops
Nloops = 200

raw_rms = zeros(Nloops)
rms_dict = Dict("bis_inv_slope"=>zeros(Nloops),
                "bis_span"=>zeros(Nloops),
                "bis_curve"=>zeros(Nloops),
                "bis_tuned"=>zeros(Nloops))

corr_dict = Dict("bis_inv_slope"=>zeros(Nloops),
                 "bis_span"=>zeros(Nloops),
                 "bis_curve"=>zeros(Nloops),
                 "bis_tuned"=>zeros(Nloops))

# read in tuned BIS data
bis_df = CSV.read(joinpath(data, "tuned_params.csv"), DataFrame)

# get optimized depths
df_dep = CSV.read(joinpath(datadir, "optimized_depth.csv"), DataFrame)

# loop over lines
for i in eachindex(lp.λrest)
    # print status
    println("\t>>> Template: " * name[i])

    # get the tuned BIS levels
    bis_params_idx = findfirst(x -> contains(name[i], x), bis_df.line)
    b1 = bis_df[bis_params_idx, "b1"]
    b2 = bis_df[bis_params_idx, "b2"]
    b3 = bis_df[bis_params_idx, "b3"]
    b4 = bis_df[bis_params_idx, "b4"]

    # set up spec
    Nt = round(Int, 60 * 40 / 15)
    lines = [λrest[i]]
    templates = [lfile[i]]
    depths = [df_dep[i, "optimized_depth"]]
    blueshifts = zeros(length(lines))
    resolution = 7e5

    # initialize objects
    disk = DiskParams(Nt=Nt)
    spec = SpecParams(lines=lines, depths=depths, templates=templates,
                      blueshifts=blueshifts, oversampling=1.0)

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

    # allocate memory that will be reused in ccf computation
    Δv_step = 100.0
    Δv_max = round((width_vel + 1e3)/100) * 100
    if Δv_max < 15e3
        Δv_max = 15e3
    end
    @show Δv_max

    # allocate memory for ccf
    v_grid = range(-Δv_max, Δv_max, step=Δv_step)
    projection = zeros(length(spec.lambdas), 1)
    proj_flux = zeros(length(spec.lambdas))
    ccf = zeros(length(v_grid), Nt)

    # number of loops
    for k in 1:Nloops
        # print loop status
        # println("\t\t>>> Loop " * string(k) * " of " * string(Nloops))

        # synthesize the line
        wavs, flux = synthesize_spectra(spec, disk, verbose=false, use_gpu=true)

        # measure bisector
        bis, int = GRASS.calc_bisector(wavs, flux, nflux=100, top=0.99)

        # convert bis to velocity scale
        for i in 1:Nt
            bis[:, i] = (bis[:,i] .- lines[1]) * GRASS.c_ms / lines[1]
        end

        # measure velocities
        GRASS.calc_ccf!(v_grid, projection, proj_flux, ccf, wavs, flux, lines,
                        depths, resolution, Δv_step=Δv_step, Δv_max=Δv_max,
                        mask_type=EchelleCCFs.GaussianCCFMask)
        rvs, sigs = calc_rvs_from_ccf(v_grid, ccf, frac_of_width_to_fit=0.5)

        # set the rms in the table
        raw_rms[k] = calc_rms(rvs)

        # calculate summary statistics
        bis_inv_slope = GRASS.calc_bisector_inverse_slope(bis, int)
        bis_span = GRASS.calc_bisector_span(bis, int)
        bis_curve = GRASS.calc_bisector_curvature(bis, int)
        bis_inv_slope_tuned = GRASS.calc_bisector_inverse_slope(bis, int, b1=b1, b2=b2, b3=b3, b4=b4)

        # set up holder for summary statistis and loop over it
        holder = [bis_inv_slope, bis_span, bis_curve, bis_inv_slope_tuned]

        for j in eachindex(holder_names)
            # data to fit
            xdata = holder[j] .- mean(holder[j])
            ydata = rvs .- mean(rvs)

            # perform the fit
            pfit = Polynomials.fit(xdata, ydata, 1)

            # decorrelate the velocities
            rvs_to_subtract = pfit.(xdata)
            rvs_rms_decorr = calc_rms(rvs .- rvs_to_subtract)

            # assign rms
            rms_dict[holder_names[j]][k] = rvs_rms_decorr
            corr_dict[holder_names[j]][k] = Statistics.cor(xdata, ydata)
        end
    end

    # take the average
    df[i, "raw_rms"] = mean(raw_rms)
    df[i, "raw_rms_sig"] = std(raw_rms)
    for j in eachindex(holder_names)
        # assign rms
        df[i, holder_names[j]*"_rms"] = mean(rms_dict[holder_names[j]])
        df[i, holder_names[j]*"_sig"] = std(rms_dict[holder_names[j]])
        df[i, holder_names[j]*"_corr"] = mean(corr_dict[holder_names[j]])
    end
end

CSV.write(datafile, df)
