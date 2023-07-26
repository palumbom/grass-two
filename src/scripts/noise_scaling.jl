# imports
using JLD2
using CUDA
using GRASS
using Printf
using FileIO
using Profile
using Statistics
using EchelleCCFs
using BenchmarkTools

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
const datafile = string(abspath(joinpath(data, "spectra_for_bin.jld2")))

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

# isolate a chunk of spectrum around a good line
idx = findall(x -> occursin.("FeI_5250.6", x), templates)
line_centers = sort!(lines[idx])
line_center = line_centers[5]

buffer = 1.0
idx1 = findlast(x -> x .<= line_center - buffer, wavs)
idx2 = findlast(x -> x .<= line_center + buffer, wavs)

# take isolated view
wavs_iso = copy(view(wavs, idx1:idx2))
flux_iso = copy(view(flux, idx1:idx2, :))

# make noises to loop over
SNRs = range(100.0, 1000.0, step=10.0)

#=# allocate memory for rvs stuff
rv_std = zeros(length(SNRs))

# loop over them
flux_noise = similar(flux_iso)
for i in eachindex(SNRs)
    # copy over SNR infinity flux
    flux_noise .= flux_iso

    # add noise
    GRASS.add_noise!(flux_noise, SNRs[i])

    # calculate ccf
    v_grid, ccf1 = calc_ccf(wavs_iso, flux_noise, lines[idx], depths[idx], 7e5)
    rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1)
    rv_std[i] = std(rvs1)
end

plt.plot(SNRs, rv_std)
plt.show()
=#

# get SNR ~ 500 spectrum
wavs_500 = copy(wavs)
flux_500 = copy(flux)
GRASS.add_noise!(flux_500, 500.0)

rvs_std = zeros(length(idx))

# include more and more lines in ccf
for i in 1:length(idx)
    @show i
    # get lines to include in ccf
    ls = lines[idx[1:i]]
    ds = depths[idx[1:i]]

    # calculate ccf
    v_grid, ccf1 = calc_ccf(wavs_500, flux_500, ls, ds, 7e5)
    rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1)
    rvs_std[i] = std(rvs1)
end

plt.plot(1:length(idx), rvs_std)
plt.savefig("derp.pdf")
plt.close()
