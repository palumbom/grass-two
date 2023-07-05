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
using Polynomials

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
const datafile = string(abspath(joinpath(data, "spectra_for_bin.jld2")))

function make_ccf_plots(wavs, flux, lines, depths, title, filename; plot_correlation=true)
    # calculate a ccf for one line
    v_grid, ccf1 = calc_ccf(wavs, flux, lines, depths,
                            7e5, mask_type=EchelleCCFs.TopHatCCFMask,
                            Δv_step=125.0)
    rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1)
    bis, int = GRASS.calc_bisector(v_grid, ccf1, nflux=20)
    bis_inv_slope = GRASS.calc_bisector_inverse_slope(bis, int)

    # get data to fit / plot
    xdata = rvs1 .- mean(rvs1)
    ydata = bis_inv_slope .- mean(bis_inv_slope)

    # do the fit
    pfit = Polynomials.fit(xdata, ydata, 1)
    xmodel = range(-2, 2, length=5)
    ymodel = pfit.(xmodel)

    # get slope string to annotate
    slope = round(coeffs(pfit)[2], digits=3)
    fit_label = "\$ " .* string(slope) .* "\$"

    # calc mean to plot
    mean_bis = dropdims(mean(bis, dims=2), dims=2)[2:end-2]
    mean_int = dropdims(mean(int, dims=2), dims=2)[2:end-2]
    mean_bis .-= mean(mean_bis)

    # make plotting objects
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8))
    fig.subplots_adjust(wspace=0.05)
    # ax1.set_box_aspect(0.75)
    # ax2.set_box_aspect(0.75)

    ax1.plot(mean_bis, mean_int, c="k", lw=3.0)
    ax2.scatter(xdata, ydata, c="k", s=1.5, alpha=0.9)
    if plot_correlation
        label = L"{\rm Slope } \approx\ " * fit_label
        ax2.plot(xmodel, ymodel, c="k", ls="--", label=label, lw=2.5)
        ax2.legend()
    end

    ax1.set_xlabel(L"\Delta v\ {\rm (m\ s}^{-1}{\rm )}")
    ax1.set_ylabel(L"{\rm Normalized\ CCF}")
    ax2.set_xlabel(L"{\rm RV\ - \overline{\rm RV}\ } {\rm (m\ s}^{-1}{\rm )}")
    ax2.set_ylabel(L"{\rm BIS}\ - \overline{\rm BIS}\ {\rm (m\ s}^{-1}{\rm )}")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    if !isempty(title)
        fig.suptitle("{\\rm " * replace(title, " "=> "\\ ") * "}", y=0.98)
    end
    fig.savefig(filename)
    plt.clf(); plt.close()
end

# get line properties
lp = GRASS.LineProperties()

# read in the data
d = load(datafile)
wavs = d["wavs"]
flux = d["flux"]
templates = d["templates"]
lines = d["lines"]
depths = d["depths"]

# collect lines
lines = collect(lines)

# plot the spectrum
# plt.plot(wavs, flux[:,1])
# plt.show()

# do all lines
outfile = string(abspath(joinpath(figures, "ccf_bin.pdf")))
make_ccf_plots(wavs, flux, lines, depths, "", outfile, plot_correlation=false)

# do a single line (a nice one)
outfile = string(abspath(joinpath(figures, "FeI_5576_ccf.pdf")))
idx = findfirst(x -> occursin.("FeI_5576", x), templates)
make_ccf_plots(wavs, flux, [lines[idx]], [depths[idx]], "Single Line", outfile)

# now do all lines of a given template
line_names = GRASS.get_name(lp)
for i in line_names
    outfile = string(abspath(joinpath(figures, i * "_rv_vs_bis.pdf")))
    idx = findall(x -> occursin.(i, x), templates)
    make_ccf_plots(wavs, flux, lines[idx], depths[idx], i, outfile)
end

# get list of line idxs in each depth bin
depth_bins = range(0.0, 1.0, step=0.1)
idxs = []
for i in eachindex(depth_bins)
    # get indices
    i == 1 && continue
    idx = map(x -> (x .<= depth_bins[i]) & (x .> depth_bins[i - 1]), depths)
    push!(idxs, idx)
end

for i in eachindex(idxs)
    # get line title
    title = "Depths gt " * string(depth_bins[i]) * " and lt " * string(depth_bins[i+1])
    println(title)

    # outfile
    outfile = string(abspath(joinpath(figures, string(depth_bins[i]) * "_rv_vs_bis.pdf")))

    # get the lines and make plots
    lines_i = view(lines, idxs[i])
    depths_i = view(depths, idxs[i])
    if isempty(lines_i)
        continue
    end
    make_ccf_plots(wavs, flux, lines_i, depths_i, title, outfile)
end

# get formation temperatues
line_info = CSV.read(GRASS.datdir * "line_info.csv", DataFrame)
avg_temp_80 = line_info.avg_temp_80
avg_temp_50 = line_info.avg_temp_50

# read in formation temps
# TODO measure temperature weighted by information content (slope)
# plt.scatter(line_info.air_wavelength, line_info.avg_temp_50, c="tab:blue")
# plt.scatter(line_info.air_wavelength, line_info.avg_temp_80, c="k")
# plt.xlabel("Wavelength")
# plt.ylabel("Avg. Line Formation Temperature")
# plt.show()

# get indices for bins
temp_bins = range(4000.0, 5000.0, step=125.0)
idxs = []
for i in eachindex(depth_bins)
    # get indices
    i == 1 && continue
    idx = map(x -> (x .<= temp_bins[i]) & (x .> temp_bins[i - 1]), avg_temp_50)
    push!(idxs, idx)
end

for i in eachindex(idxs)
    # get line title
    title = "Temp gt " * string(temp_bins[i]) * " and lt " * string(temp_bins[i+1])
    println(title)

    # outfile
    outfile = string(abspath(joinpath(figures, string(temp_bins[i]) * "_rv_vs_bis.pdf")))

    # get the lines and make plots
    lines_i = view(lines, idxs[i])
    depths_i = view(depths, idxs[i])
    if isempty(lines_i)
        continue
    end
    make_ccf_plots(wavs, flux, lines_i, depths_i, title, outfile)
end
