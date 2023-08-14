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
using Polynomials

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
idx = findall(x -> occursin.("FeI_5576", x), templates)
sort_idx = sortperm(lines[idx])
line_centers = lines[idx][sort_idx]
the_depths = depths[idx][sort_idx]

line_center = line_centers[5]
the_depth = the_depths[5]

buffer = 1.0
idx1 = findlast(x -> x .<= line_center - buffer, wavs)
idx2 = findlast(x -> x .<= line_center + buffer, wavs)

# take isolated view
wavs_iso = copy(view(wavs, idx1:idx2))
flux_iso = copy(view(flux, idx1:idx2, :))

# get bisectors
bis, int = GRASS.calc_bisector(wavs_iso, flux_iso, nflux=50)
bis = GRASS.c_ms .* (bis .- line_center) ./ (line_center)

# compute velocities
v_grid, ccf1 = calc_ccf(wavs_iso, flux_iso, [line_center], [1.0 - minimum(flux_iso)], 7e5)
rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1)

# subtract off the mean
rvs1 .-= mean(rvs1)

# get the mean bisector
mean_bis = mean(bis, dims=2)
mean_int = mean(int, dims=2)

# get the redshifted bisectors
blu_idx = rvs1 .< 0.5
red_idx = rvs1 .> 0.5

# get means for red and blue
mean_bis_red = mean(bis[:, red_idx], dims=2)
mean_bis_blu = mean(bis[:, blu_idx], dims=2)
mean_int_red = mean(int[:, red_idx], dims=2)
mean_int_blu = mean(int[:, blu_idx], dims=2)

# plot them all
plt.plot(mean_bis, mean_int, c="k")
plt.plot(mean_bis_red, mean_int_red, c="red")
plt.plot(mean_bis_blu, mean_int_blu, c="tab:blue")
plt.show()

plt.plot(mean_bis .- mean_bis_red, mean_int, c="red", ls="--")
plt.plot(mean_bis .- mean_bis_blu, mean_int, c="tab:blue", ls="--")
plt.show()
