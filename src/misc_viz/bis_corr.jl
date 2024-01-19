# imports
using CSV
using CUDA
using JLD2
using FFTW
using GRASS
using Printf
using Revise
using FileIO
using Random
using DataFrames
using Statistics
using EchelleCCFs
using Distributions
using LinearAlgebra

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; #plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# get command line args and output directories
# include(joinpath(abspath(@__DIR__), "paths.jl"))
include("paths.jl")

# lines to plot
# linez = ["FeI_5250.2", "FeI_5250.6"]
# linez = ["FeI_5379", "TiII_5381"]
linez = ["FeI_5379", "FeI_6302"]

fnames = GRASS.soldir .* linez

airwavs = [0.0, 0.0]
airwavs = [5379.5734, 5381.0216]

linestyles = ["-", "--", ":"]

bisinfo1 = GRASS.SolarData(fname=fnames[1], extrapolate=false, contiguous_only=true,
                           relative=true, strip_cols=false)
bisinfo2 = GRASS.SolarData(fname=fnames[2], extrapolate=false, contiguous_only=true,
                           relative=true, strip_cols=false)

# the_key = (:s, :mu02)
the_key = (:s, :mu03)
# the_key = (:c, :mu10)
bis1 = bisinfo1.bis[the_key]
int1 = bisinfo1.int[the_key]
bis2 = bisinfo2.bis[the_key]
int2 = bisinfo2.int[the_key]

@show minimum(int1)
@show minimum(int2)

# interpolate onto same intensity grid
int0 = range(0.45, 0.8, length=100)

# get number of epochs for loop
n_epochs = minimum([size(bis1, 2), size(bis2, 2)])

# interp onto same intensity grid and smooth with moving average
for i in 1:n_epochs
    itp1 = GRASS.linear_interp(int1[:,i], bis1[:,i])
    itp2 = GRASS.linear_interp(int2[:,i], bis2[:,i])

    bis1[:,i] .= GRASS.moving_average(itp1.(int0), 4)
    int1[:,i] .= GRASS.moving_average(int0, 4)
    bis2[:,i] .= GRASS.moving_average(itp2.(int0), 4)
    int2[:,i] .= GRASS.moving_average(int0, 4)
end

# subtract off the mean
# bis1 .-= mean(bis1, dims=1)
# bis2 .-= mean(bis2, dims=1)

# plt.plot(mean(bis1, dims=2), mean(int1, dims=2), c="tab:blue")
# plt.plot(mean(bis2, dims=2), mean(int2, dims=2), c="tab:orange")
# plt.xlabel("delta v (m/s)")
# plt.ylabel("relative flux")
# plt.show()

# calculate dot products
dp = zeros(n_epochs)
for i in 1:n_epochs
    dp[i] = dot(bis1[:,i], bis2[:,i]) / (norm(bis1[:,i]) * norm(bis2[:,i]))
end

plt.plot(eachindex(dp) .* 15 .- 15, dp)
plt.xlabel("Delta t (s)")
plt.ylabel("Similarity")
plt.show()


# # get power spectrum of velocities
# sampling_rate = 1.0/15.0
# F = fftshift(fft(dp))
# freqs = fftshift(fftfreq(length(dp), sampling_rate))

# # throw away samplings of frequencies less than 0.0s
# idx = findall(freqs .> 0.0)
# F = F[idx]
# freqs = freqs[idx]

# # bin the frequencies
# minfreq = log10(minimum(freqs))
# maxfreq = log10(maximum(freqs))
# freqstep = 1e-1
# freqs_binned = collect(10.0 .^ (range(minfreq, maxfreq, step=freqstep)))
# freqs_bincen = (freqs_binned[2:end] .+ freqs_binned[1:end-1])./2
# F_binned = zeros(length(freqs_binned)-1)
# for i in eachindex(F_binned)
#     j = findall(x -> (x .>= freqs_binned[i]) .& (x .<= freqs_binned[i+1]), freqs)
#     F_binned[i] = mean(abs2.(F[j]))
# end

# fig, ax1 = plt.subplots()
# ax1.scatter(freqs, abs2.(F), s=1, c="k")
# ax1.scatter(freqs_bincen, F_binned, marker="x", c="red")
# ax1.axvline(1.0 / (60.0 * 20.0), ls=":", c="k", alpha=0.75, label=L"{\rm 20\ min.}")
# ax1.axvline(1.0 / (15.0), ls="--", c="k", alpha=0.75, label=L"{\rm 15\ sec.}")
# ax1.set_xscale("log")
# ax1.set_yscale("log")
# ax1.legend()
# ax1.set_xlabel("Frequency [Hz]")
# ax1.set_ylabel(L"{\rm Power\ [(m\ s}^{-1}{\rm )}^2 {\rm \ Hz}^{-1}{\rm ]}")
# # ax1.set_xlim(1e-8, maximum(freqs))
# # ax1.set_ylim(1, 1e5)
# # fig.savefig(joinpath(outdir, names[i] * "power_spec.pdf"))
# plt.show()
# plt.clf(); plt.close()
