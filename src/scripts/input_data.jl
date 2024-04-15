# imports
using CSV
using JLD2
using GRASS
using PyCall
using FileIO
using DataFrames
using Statistics
import Polynomials: fit as pfit, coeffs

# plotting imports
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
using LaTeXStrings
mpl.style.use(joinpath(GRASS.moddir, "fig.mplstyle"))
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]


# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))

# read in the spectrum
datafile = string(abspath(joinpath(static, "FeI_6302_spec.jld2")))
d = load(datafile)
wavs = d["wavs"]
flux = d["flux"]
nois = d["noise"]

# air wavelength for line of interest
airwav = 6302.4932

# normalize by continuum
nois ./= maximum(flux)
flux ./= maximum(flux)

# find the minimum of the line of interest
minbuff = 25
idx = findfirst(x -> x .>= airwav, wavs)
min = argmin(flux[idx-minbuff:idx+minbuff]) + idx - (minbuff+1)
bot = flux[min]
depth = 1.0 - bot

# find the linewing indices
wavbuff = 0.2
idx1 = findfirst(x -> x .> wavs[min] - wavbuff, wavs)
idx1 = argmax(flux[idx1:min]) + idx1
idx2 = findfirst(x -> x .> wavs[min] + wavbuff, wavs)
idx2 = argmax(flux[min:idx2]) + min

# get view of line
wavs_iso = view(wavs, idx1:idx2)
flux_iso = view(flux, idx1:idx2)
nois_iso = view(flux, idx1:idx2)

# get the bisector
bis, int1 = GRASS.calc_bisector(wavs_iso, flux_iso, nflux=50, top=maximum(flux_iso) - 0.01)

# smooth it
bis = GRASS.moving_average(bis, 4)
int1 = GRASS.moving_average(int1, 4)

# pad flux_iso with ones
cont_idxl = findall(x -> (1.0 .- x) .< 0.001, flux[1:idx1])
cont_idxr = findall(x -> (1.0 .- x) .< 0.001, flux[idx2+2:end]) .+ idx2

flux_padl = ones(length(cont_idxl))
flux_padr = ones(length(cont_idxr))

# construct arrays to fit on
wavs_fit = vcat(wavs[cont_idxl], copy(view(wavs, idx1:idx2)), wavs[cont_idxr])
flux_fit = vcat(flux_padl, copy(view(flux, idx1:idx2)), flux_padr)
nois_fit = vcat(nois[cont_idxl], copy(view(nois, idx1:idx2)), nois[cont_idxr])

# fit the line wings
lfit, rfit = GRASS.fit_line_wings(wavs_fit, flux_fit, nois_iso=nois_fit, debug=false)

# get index to replace wings above
top = 0.8 * depth + bot
lwing, rwing = GRASS.find_wing_index(top, flux_iso, min=argmin(flux_iso))
lwing += 1

# replace line wings and measure line width function
wavs_smooth = copy(wavs)
flux_smooth = copy(flux)
GRASS.replace_line_wings(lfit, rfit, wavs_smooth, flux_smooth, min, top, debug=false)
int2, wid = GRASS.calc_width_function(wavs_smooth, flux_smooth, nflux=100, top=0.999)

# evaluate model fits to plot
l_wavs_model = range(wavs[1], wavs_iso[lwing], step=minimum(diff(wavs)))
r_wavs_model = range(wavs_iso[rwing], wavs[end], step=minimum(diff(wavs)))
l_flux_model = GRASS.fit_voigt(l_wavs_model, lfit.param)
r_flux_model = GRASS.fit_voigt(r_wavs_model, rfit.param)

# make plot objects
fig = plt.figure(figsize=(12.8, 4.8))
gs0 = mpl.gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[2, 1.25, 1.25], figure=fig, wspace=0.05)
ax1 = fig.add_subplot(gs0[1])
ax2 = fig.add_subplot(gs0[2])
ax3 = fig.add_subplot(gs0[3])

# plot the spectra + fits
ax1.plot(wavs, flux, c="k", label=L"{\rm Binned\ Spectrum}")
ax1.plot(l_wavs_model, l_flux_model, c=colors[1], lw=2, ls="-.", label=L"{\rm Blue\ Wing\ Model}")
ax1.plot(r_wavs_model, r_flux_model, c=colors[2], lw=2, ls="--", label=L"{\rm Red\ Wing\ Model}")
ax1.legend(fontsize=12)

# plot the bisector
bfit2 = pfit(int1[3:10], bis[3:10], 1)
bis[1:3] .= bfit2.(int1[1:3])
ax2.plot(GRASS.c_ms .* (bis .- mean(bis)) ./ airwav, int1, c="k")

# plot the width function
ax3.plot(wid, int2, c="k")

# plot horizontal lines
ax1.axhline(top, ls="--", c="gray")
# ax2.axhline(top, ls="--", c="gray")
ax3.axhline(top, ls="--", c="gray")

# set plot stuff
ax1.set_xlim(wavs[min] - 0.76, wavs[min] + 0.76)
ax1.set_xlabel(L"{\rm Wavelength\ (\AA)}")
ax1.set_ylabel(L"{\rm Normalized\ Flux}")
ax2.set_xlabel(L"{\rm Relative\ Velocity\ (m\ s^{-1})}")
ax3.set_xlabel(L"{\rm Width\ (\AA)}")

# set ticks
ax1.set_xticks([6302, 6302.5, 6303.0])
ax3.set_xticks([0, 1, 2])
ax3.set_xlim(-0.1, 2.1)

# hide superfluous y axes
ax2.get_yaxis().set_ticklabels([])
ax3.get_yaxis().set_ticklabels([])

# make the ylimits match
ax2.set_ylim(ax1.get_ylim()...)
ax3.set_ylim(ax1.get_ylim()...)

# save the figure
plotfile = string(abspath(joinpath(figures, "input_fit_example.pdf")))
fig.tight_layout()
fig.savefig(plotfile)
plt.clf(); plt.close()

