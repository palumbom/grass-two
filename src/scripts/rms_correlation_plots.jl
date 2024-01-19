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
include("paths.jl")#joinpath(abspath(@__DIR__), "paths.jl"))
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

# read in the data file
df = CSV.read(datafile, DataFrame)

# read in the line info with formation temperature
tempfile = joinpath(GRASS.moddir, "data", "line_info.csv")
df2 = CSV.read(tempfile, DataFrame)
rename!(df2, :name=>:line)

idx = []
for i in 1:length(df2.line)
    if df2.line[i] in df.line
        push!(idx, i)
    end
end

df[!, "avg_temp_50"] = df2.avg_temp_50[idx]
df[!, "avg_temp_80"] = df2.avg_temp_80[idx]

# make a colorbar
cmap = plt.cm.inferno
dat = df.avg_temp_50
cnorm = mpl.colors.Normalize(vmin=minimum(dat), vmax=maximum(dat))
smap = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)

# now plot the data
fig, ax1 = plt.subplots()
ax1.errorbar(df.raw_rms, df.bis_inv_slope_rms, xerr=df.raw_rms_sig, yerr=df.bis_inv_slope_sig, linestyle="none", c="k", capsize=0.5, zorder=0)
ax1.scatter(df.raw_rms, df.bis_inv_slope_rms, marker="o", c=dat, zorder=1, norm=cnorm, cmap=cmap)

# plot the one to one line
xmin, xmax = ax1.get_xlim()
ax1.plot(range(xmin, xmax, length=3), range(xmin, xmax, length=3), c="k", ls="--")

cbar = fig.colorbar(smap, ax=ax1)
# cbar.ax.tick_params(labelsize=11)
cbar.ax.set_ylabel(L"{\rm T}_{1/2}\ {\rm (K)}")

ax1.set_xlabel(L"{\rm RV\ RMS\ (m\ s}^{-1}{\rm )}")
ax1.set_ylabel(L"{\rm BIS\ RMS\ (m\ s}^{-1}{\rm )}")

fig.savefig(joinpath(plotdir, "rms_vs_rms.pdf"))
# plt.show()
plt.clf(); plt.close()


# now plot rms stair plot thing
ydata = df.raw_rms
yerrs = df.raw_rms_sig

# sort the data in descending order
idx = reverse(sortperm(ydata))

xdata = range(1, length(ydata), step=1)

# get ticks and tick labels
xticks = xdata
xticklabels = []
for i in eachindex(name)
    title = replace(name[i], "_" => "\\ ")
    tidx = findfirst("I", title)
    title = "\${\\rm " * title * "\\ \\AA }\$"
    push!(xticklabels, title)
end

fig, ax1 = plt.subplots(figsize=(9.2,4.8))

ax1.errorbar(xdata, ydata[idx], yerr=yerrs[idx], linestyle="none", c="k", capsize=0.5, zorder=0)
ax1.scatter(xdata, ydata[idx], marker="o", c=dat, zorder=1, norm=cnorm, cmap=cmap)

cbar = fig.colorbar(smap, ax=ax1)
# cbar.ax.tick_params(labelsize=11)
cbar.ax.set_ylabel(L"{\rm T}_{1/2}\ {\rm (K)}")

ax1.set_ylabel(L"{\rm RV\ RMS\ (m\ s}^{-1}{\rm )}")

ax1.set_xticks(xticks)
ax1.set_xticklabels(xticklabels[idx], rotation=90)
ax1.grid(false)

fig.tight_layout()
fig.savefig(joinpath(plotdir, "rms_ladder.pdf"), bbox_inches="tight")
plt.clf(); plt.close()

# plot rms vs temp
# fig, axs = plt.subplots(figsize=(12.4,4.8), ncols=3, nrows=1, sharey=true)
fig, ax1 = plt.subplots()
# ax1, ax2, ax3 = axs

ax1.errorbar(df.avg_temp_50, df.raw_rms, yerr=df.raw_rms_sig, linestyle="none", c="k", capsize=0.5, zorder=0)
ax1.scatter(df.avg_temp_50, df.raw_rms, marker="o", c=dat, zorder=1, norm=cnorm, cmap=cmap)

# ax1.set_xlim(minimum(df.avg_temp_50), maximum(df.avg_temp_50))
# ax1.set_ylim(minimum(df.raw_rms), maximum(df.raw_rms))

# ax2.errorbar(depth, df.raw_rms, yerr=df.raw_rms_sig, linestyle="none", c="k", capsize=0.5, zorder=0)
# ax2.scatter(depth, df.raw_rms, marker="o", c=dat, zorder=1, norm=cnorm, cmap=cmap)

# ax3.errorbar(λrest, df.raw_rms, yerr=df.raw_rms_sig, linestyle="none", c="k", capsize=0.5, zorder=0)
# derp = ax3.scatter(λrest, df.raw_rms, marker="o", c=dat, zorder=1, norm=cnorm, cmap=cmap)

# cbar = fig.colorbar(smap, ax=ax3)
cbar = fig.colorbar(smap, ax=ax1)
# cbar.ax.tick_params(labelsize=11)
cbar.ax.set_ylabel(L"{\rm T}_{1/2}\ {\rm (K)}")

ax1.set_xlabel(L"{\rm T}_{1/2}\ {\rm (K)}")
# ax2.set_xlabel(L"{\rm Depth}")
# ax3.set_xlabel(L"{\rm Wavelength\ (\AA)}")
ax1.set_ylabel(L"{\rm RV\ RMS\ (m\ s}^{-1}{\rm )}")

fig.tight_layout()
fig.subplots_adjust(hspace=0.1, wspace=0.075)
fig.savefig(joinpath(plotdir, "rms_vs_temp.pdf"), bbox_inches="tight")
plt.clf(); plt.close()


