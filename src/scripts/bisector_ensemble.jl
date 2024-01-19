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

# get lines to construct templates
lp = GRASS.LineProperties()
name = GRASS.get_name(lp)
λrest = GRASS.get_rest_wavelength(lp)
depth = GRASS.get_depth(lp)
lfile = GRASS.get_file(lp)
llevel = GRASS.get_lower_level(lp)
ulevel = GRASS.get_upper_level(lp)

# allocate memory for bisectors
vels = zeros(100, length(name))
ints = zeros(100, length(name))

# loop over lines
for i in eachindex(lp.λrest)
    println("\t>>> Template: " * string(splitdir(lfile[i])[2]))

    # set up parameters for synthesis
    Nt = round(Int, 60 * 40 / 15)
    lines = [λrest[i]]
    templates = [lfile[i]]
    depths = [depth[i]]
    resolution = 7e5

    # synthesize the line
    disk = DiskParams(Nt=Nt)
    spec = SpecParams(lines=lines, depths=depths, templates=templates, oversampling=4.0)
    wavs, flux = synthesize_spectra(spec, disk, verbose=false, use_gpu=true)

    # measure velocities
    v_grid, ccf = calc_ccf(wavs, flux, spec, Δv_step=50.0, Δv_max=30e3)
    rvs, sigs = calc_rvs_from_ccf(v_grid, ccf)

    # measure ccf bisector
    vel, int = GRASS.calc_bisector(v_grid, ccf, nflux=100, top=0.99)

    vels[:, i] .= mean(vel, dims=2)
    ints[:, i] .= mean(int, dims=2)
end

# get indices to sort on depth
idx = sortperm(depth)

# get cmap
cmap = plt.get_cmap("tab20")
colors = ["#fd7e6d", "#fa2004", "#74afdd", "#2e7fbe", "#b3e65b", "#87c41d",
          "#ca71cb", "#a03ca2", "#ffb65c", "#f38600", "#ffec5c", "#facc00",
          "#aea6de", "#6859c1", "#fb9dcc", "#f74aa4", "#84dccd", "#31af9a"]

# make figure objects
fig, (ax1,ax2) = plt.subplots(figsize=(7.6,5.2), ncols=2, nrows=1, sharey=true, gridspec_kw=Dict("width_ratios" => [3, 1]))

# loop over bisectors
global k = 0
for (i,j) in enumerate(idx)
    # get view, chopping off bottom measurement
    vel = @view vels[3:end-1, j]
    int = @view ints[3:end-1, j]

    # subtract off mean
    vel .-= mean(vel)

    # move it over in velocity arbitrarily
    vel .-= ((length(idx) - k) * 180)

     if vel[end] - vel[1] < -50
        continue
    end

    global k += 1

    # ax1.plot(vel, int, c=cmap(k/length(idx)))
    ax1.plot(vel, int, c=colors[k])
end

# get intensity run of smallest depth
min_dep = minimum(depth)

# loop over bisectors
global k = 0
for (i,j) in enumerate(idx)
    # get view, chopping off bottom measurement
    vel = @view vels[3:end-1, j]
    int = @view ints[3:end-1, j]

    # find indices
    idx_dep = findfirst(x -> x .>= 1.0 .- min_dep, int)

    # subtract off mean
    vel .-= mean(vel[idx_dep:end])

    if vel[end] - vel[1] < -50
        continue
    end

    global k += 1

    title = replace(name[j], "_" => "\\ ")
    tidx = findfirst("I", title)
    # title = title[1:tidx-1] * "\\ " * title[tidx:end]
    title = ("\${\\rm " * title * "\\ \\AA }\$")

    # ax2.plot(vel, int, label=title, c=cmap(k/length(idx)))
    ax2.plot(vel, int, label=title, c=colors[k])
end

# ax1.set_xlabel(L"{\rm Arbitrary\ Velocity}")
# ax2.set_xlabel(L"{\rm Arbitrary\ Velocity}")
fig.supxlabel(L"{\rm Arbitrary\ Velocity}", y=0.075)
ax1.set_ylabel(L"{\rm Normalized\ Intensity}")

ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

fig.tight_layout()

fig.subplots_adjust(wspace=0.02)

fig.savefig(joinpath(plotdir, "bis_dossier1.pdf"), bbox_inches="tight")
plt.clf(); plt.close()

# make figure objects
fig, (ax1,ax2) = plt.subplots(figsize=(7.6,5.2), ncols=2, nrows=1, sharey=true, gridspec_kw=Dict("width_ratios" => [3, 1]))

# loop over bisectors
remaining_lines = length(idx) - k

global k = 0
for (i,j) in enumerate(idx)
    # get view, chopping off bottom measurement
    vel = @view vels[3:end-1, j]
    int = @view ints[3:end-1, j]

    # subtract off mean
    vel .-= mean(vel)

    # move it over in velocity arbitrarily
    vel .-= ((length(idx) - k) * 180)

    if vel[end] - vel[1] >= -50
        continue
    end

    global k += 1

    title = replace(name[j], "_" => "\\ ")

    if occursin("5383", title)
        vel .+= 500
    end

    # ax1.plot(vel, int, c=cmap(k/remaining_lines))
    ax1.plot(vel, int, c=colors[k])
end

# get intensity run of smallest depth
min_dep = minimum(depth)

# loop over bisectors
global k = 0
for (i,j) in enumerate(idx)
    # get view, chopping off bottom measurement
    vel = @view vels[3:end-1, j]
    int = @view ints[3:end-1, j]

    # find indices
    idx_dep = findfirst(x -> x .>= 1.0 .- min_dep, int)

    # subtract off mean
    vel .-= mean(vel[idx_dep:end])

    if vel[end] - vel[1] >= -50
        continue
    end

    global k += 1

    title = replace(name[j], "_" => "\\ ")
    tidx = findfirst("I", title)
    # title = title[1:tidx-1] * "\\ " * title[tidx:end]
    title = ("\${\\rm " * title * "\\ \\AA }\$")

    # ax2.plot(vel, int, label=title, c=cmap(k/remaining_lines))
    ax2.plot(vel, int, label=title, c=colors[k])
end

# ax1.set_xlabel(L"{\rm Arbitrary\ Velocity}")
# ax2.set_xlabel(L"{\rm Arbitrary\ Velocity}")
fig.supxlabel(L"{\rm Arbitrary\ Velocity}", y=0.075)
ax1.set_ylabel(L"{\rm Normalized\ Intensity}")

ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

fig.tight_layout()

fig.subplots_adjust(wspace=0.02)

fig.savefig(joinpath(plotdir, "bis_dossier2.pdf"), bbox_inches="tight")
plt.clf(); plt.close()

