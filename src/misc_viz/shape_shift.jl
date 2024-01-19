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

# compute velocities
v_grid, ccf1 = calc_ccf(wavs_iso, flux_iso, [line_center], [1.0 - minimum(flux_iso)], 7e5)
rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1)

# dep, intw = calc_width_function(v_grid, ccf1)

# calculate ccf bisector
vel, int = GRASS.calc_bisector(v_grid, ccf1, nflux=50)
vel = vel[2:end, :]
int = int[2:end, :]

# AA = AbstractArray
# function calc_bisector_slope(bis::AA{T,1}, int::AA{T,1}) where T<:AbstractFloat
#     @assert maximum(int) <= 1.0
#     @assert minimum(int) >= 0.0

#     # get total depth
#     dep = one(T) - minimum(int)
#     bot = minimum(int)

#     # find indices
#     idx25 = findfirst(x -> x .> 0.25 * dep + bot, int)
#     idx80 = findfirst(x -> x .> 0.80 * dep + bot, int)

#     # get views
#     bis_view = view(bis, idx25:idx80)
#     int_view = view(int, idx25:idx80)

#     # linear fit
#     pfit = Polynomials.fit(bis_view, int_view, 1)

#     plt.plot(bis_view, int_view)
#     plt.axhline(int[idx25])
#     plt.axhline(int[idx80])
#     plt.plot(bis_view, pfit.(bis_view))
#     plt.show()

#     # return inverse slope
#     return -coeffs(pfit)[2]
# end

# function calc_bisector_slope(bis::AA{T,2}, int::AA{T,2}) where T<:AbstractFloat
#     out = zeros(size(bis,2))
#     for i in 1:size(bis,2)
#         out[i] = calc_bisector_slope(bis[:,i], int[:,i])
#     end
#     return out
# end


# get BIS
bis_span = GRASS.calc_bisector_span(vel, int)
bis_slope = GRASS.calc_bisector_slope(vel, int)
bis_inv_slope = GRASS.calc_bisector_inverse_slope(vel, int)
bis_curvature = GRASS.calc_bisector_curvature(vel, int)
bis_bottom = GRASS.calc_bisector_bottom(vel, int, rvs1)

# get mean
meanvel = dropdims(mean(vel, dims=2), dims=2)
meanint = dropdims(mean(int, dims=2), dims=2)
residvel = meanvel .- vel

# calculate slope for resid bisectors
slopes = zeros(size(residvel,2))
for i in eachindex(slopes)
    pfit = Polynomials.fit(residvel[:,i], meanint, 1)
    slopes[i] = coeffs(pfit)[2]
end

# plt.plot(vel, int)
# plt.show()

# plot residual
# plt.plot(residvel, meanint)
# plt.show()


# plt.scatter(slopes, rvs1, c="k", s=2)
# plt.scatter(bis_span .- mean(bis_span), rvs1, c="tab:blue", s=2)
# plt.scatter(bis_slope .- mean(bis_slope), rvs1, c="tab:orange", s=2)
# plt.scatter(bis_inv_slope .- mean(bis_inv_slope), rvs1, c="tab:red", s=2)
# plt.scatter(bis_curvature .- mean(bis_curvature), rvs1, c="tab:pink", s=2)
plt.scatter(bis_bottom .- mean(bis_bottom), rvs1, c="tab:brown", s=2)
# plt.xlim(-1.5,1.5)
plt.show()


# # subtract off the mean
# rvs1 .-= mean(rvs1)

# # get the mean bisector
# mean_bis = mean(bis, dims=2)
# mean_int = mean(int, dims=2)

# # get the redshifted bisectors
# blu_idx = rvs1 .< 0.5
# red_idx = rvs1 .> 0.5

# # get means for red and blue
# mean_bis_red = mean(bis[:, red_idx], dims=2)
# mean_bis_blu = mean(bis[:, blu_idx], dims=2)
# mean_int_red = mean(int[:, red_idx], dims=2)
# mean_int_blu = mean(int[:, blu_idx], dims=2)

# # plot them all
# plt.plot(mean_bis, mean_int, c="k")
# plt.plot(mean_bis_red, mean_int_red, c="red")
# plt.plot(mean_bis_blu, mean_int_blu, c="tab:blue")
# plt.show()

# plt.plot(mean_bis .- mean_bis_red, mean_int, c="red", ls="--")
# plt.plot(mean_bis .- mean_bis_blu, mean_int, c="tab:blue", ls="--")
# plt.show()
