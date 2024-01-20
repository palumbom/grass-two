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
plotdir = string(figures)
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
               bis_slope_corr=zeros(length(name)), bis_slope_rms=zeros(length(name)), bis_slope_sig=zeros(length(name)),
               bis_curve_corr=zeros(length(name)), bis_curve_rms=zeros(length(name)), bis_curve_sig=zeros(length(name)),
               bis_bot_corr=zeros(length(name)), bis_bot_rms=zeros(length(name)), bis_bot_sig=zeros(length(name)))

holder_names = ["bis_inv_slope", "bis_span", "bis_slope", "bis_curve", "bis_bot"]

# set number of loops
Nloops = 1000

raw_rms = zeros(Nloops)
rms_dict = Dict("bis_inv_slope"=>zeros(Nloops),
                "bis_span"=>zeros(Nloops),
                "bis_slope"=>zeros(Nloops),
                "bis_curve"=>zeros(Nloops),
                "bis_bot"=>zeros(Nloops))

corr_dict = Dict("bis_inv_slope"=>zeros(Nloops),
                 "bis_span"=>zeros(Nloops),
                 "bis_slope"=>zeros(Nloops),
                 "bis_curve"=>zeros(Nloops),
                 "bis_bot"=>zeros(Nloops))

# loop over lines
for i in eachindex(lp.λrest)
    println("\t>>> Template: " * string(splitdir(lfile[i])[2]))

    # number of loops
    for k in 1:Nloops
        println("\t\t>>> Loop " * string(k) * " of " * string(Nloops))

        # set up parameters for synthesis
        Nt = round(Int, 60 * 40 / 15)
        lines = [λrest[i]]
        templates = [lfile[i]]
        depths = [depth[i]]
        resolution = 7e5

        # synthesize the line
        disk = DiskParams(Nt=Nt)
        spec = SpecParams(lines=lines, depths=depths, templates=templates)
        wavs, flux = synthesize_spectra(spec, disk, verbose=false, use_gpu=true)

        # measure bisector
        bis, int = GRASS.calc_bisector(wavs, flux, nflux=100, top=0.99)

        # convert bis to velocity scale
        for i in 1:Nt
            bis[:, i] = (bis[:,i] .- lines[1]) * GRASS.c_ms / lines[1]
        end

        # measure velocities
        v_grid, ccf = calc_ccf(wavs, flux, spec, Δv_step=50.0, Δv_max=30e3)
        rvs, sigs = calc_rvs_from_ccf(v_grid, ccf)

        # set the rms in the table
        raw_rms[k] = calc_rms(rvs)

        # smooth the bisector and cut off bottom and top measurements
        bis = GRASS.moving_average(bis, 4)[3:end-1, :]
        int = GRASS.moving_average(int, 4)[3:end-1, :]

        # calculate summary statistics
        bis_inv_slope = GRASS.calc_bisector_inverse_slope(bis, int)
        bis_span = GRASS.calc_bisector_span(bis, int)
        bis_slope = GRASS.calc_bisector_slope(bis, int)
        bis_curve = GRASS.calc_bisector_curvature(bis, int)
        bis_bot = GRASS.calc_bisector_bottom(bis, int, rvs)

        # set up holder for summary statistis and loop over it
        holder = [bis_inv_slope, bis_span, bis_slope, bis_curve, bis_bot]

        for j in eachindex(holder_names)
            # data to fit
            xdata = holder[j]
            ydata = rvs

            # perform the fit
            pfit = Polynomials.fit(xdata, ydata, 1)
            xmodel = range(minimum(xdata), maximum(xdata), length=5)
            ymodel = pfit.(xmodel)

            # decorrelate the velocities
            rvs_to_subtract = pfit.(xdata)
            rvs_rms_decorr = calc_rms(rvs .- rvs_to_subtract)

            # assign rms
            rms_dict[holder_names[j]][k] = rvs_rms_decorr

            # get goodness of fit
            TSS = sum((ydata .- mean(ydata)).^2.0)
            RSS = sum((ydata .- pfit.(xdata)).^2.0)

            corr_dict[holder_names[j]][k] = (1.0 - RSS / TSS)
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
