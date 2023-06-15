# imports
using CSV
using GRASS
using PyCall
using DataFrames
using Statistics

# plotting imports
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
using LaTeXStrings
mpl.style.use(joinpath(GRASS.moddir, "fig.mplstyle"))
adjust_text = pyimport("adjustText")

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))

# set LARS spectra absolute dir and read line info file
const data_dir = "/storage/group/ebf11/default/mlp95/lars_spectra/"

# get summary of present line atomic params etc.
lp = LineProperties(exclude=[""])
files = lp.file
airwavs = GRASS.get_rest_wavelength(lp)
names = GRASS.get_name(lp)

# get the spectra
larsdir = "/storage/home/mlp95/ford_dir/mlp95/lars_spectra/"
dirs = filter(isdir, readdir(larsdir, join=true))
nregions = length(dirs)

# get the wavelength regions
wavmins = zeros(length(dirs))
fluxmins = zeros(length(dirs))
wavmaxs = zeros(length(dirs))
fluxmaxs = zeros(length(dirs))
for i in eachindex(dirs)
    # get subset'd dataframes
    df = GRASS.sort_spectrum_data(dir=dirs[i])
    df = subset(df, :axis => x -> x .== "c")
    file = joinpath(dirs[i], df.fname[1])

    # read in the file
    wavs, flux = GRASS.read_spectrum(file)
    flux = mean(flux,dims=2)./maximum(mean(flux,dims=2))

    wavmins[i] = minimum(wavs)
    fluxmins[i] = minimum(flux)
    wavmaxs[i] = maximum(wavs)
    fluxmaxs[i] = maximum(flux)
end

# sort on wavmin indices
idx0 = sortperm(wavmins)
wavmins .= wavmins[idx0]
fluxmins .= fluxmins[idx0]
wavmaxs .= wavmaxs[idx0]
fluxmaxs .= fluxmaxs[idx0]
dirs .= dirs[idx0]

# get width of wavelength regions
wav_wids = wavmaxs .- wavmins
wav_rats = wav_wids ./ wav_wids[1]

# create plot objects
fig = plt.figure(figsize=(21,8))
gss = mpl.gridspec.GridSpec(1, nregions, width_ratios=wav_rats)
axs = [plt.subplot(ax) for ax in gss]
fig.subplots_adjust(wspace=0.05)

# loop over files
for i in eachindex(dirs)
    # get subset'd dataframes
    df = GRASS.sort_spectrum_data(dir=dirs[i])
    df = subset(df, :axis => x -> x .== "c")
    file = joinpath(dirs[i], df.fname[1])

    # read in the file
    wavs, spec = GRASS.read_spectrum(file)

    # take a simple mean and roughly normalize
    wavs = dropdims(mean(wavs, dims=2), dims=2)
    spec = dropdims(mean(spec, dims=2), dims=2)
    spec ./= maximum(spec)

    # smooth the spectrum
    wavs = GRASS.moving_average(wavs, 5)
    spec = GRASS.moving_average(spec, 5)

    # interpolate up to high res for nice plotting of minima
    itp = GRASS.linear_interp(wavs, spec)
    wavs2 = range(minimum(wavs), maximum(wavs), step=minimum(diff(wavs)/5))
    spec2 = itp.(wavs2)

    # plot the data
    axs[i].plot(wavs2, spec2, c="k")

    # set the xlimits
    axs[i].set_xlim(minimum(wavs2) - 0.5, maximum(wavs2) + 0.5)
    axs[i].set_ylim(-0.1, 1.075)
    # axs[i].set_box_aspect(0.5)

    # axs[i].grid(false)

    # set the ticks
    wavmin = round(Int,minimum(wavs2))
    wavmid = round(Int, median(wavs2))
    wavmax = round(Int,maximum(wavs2))

    # find airwavs in wavelength region
    # idk = findall(x -> (x .<= maximum(wavs2) .& (x .>= minimum(wavs2))), airwavs)
    airwav_idx = findall(x -> (x .<= maximum(wavs2)) .& (x .>= minimum(wavs2)), airwavs)
    airwav_ann = airwavs[airwav_idx]
    names_ann = names[airwav_idx]

    texts = []
    for j in eachindex(airwav_idx)
        # find location on axis
        idx2 = findfirst(x-> x .>= airwav_ann[j], wavs2)
        min = argmin(spec2[idx2-50:idx2+50]) + idx2 - 50

        # set rotation
        if contains(names_ann[j], "5896")
            rotation = 0.0
            x1 = 0.0
            x2 = 0.0
            y1 = 0.025
            y2 = 0.1
        elseif contains(names_ann[j], "5250.6")
            rotation = 270.0
            x1 = -0.15
            x2 = -0.4
            y1 = 0.015
            y2 = 0.22
        elseif contains(names_ann[j], "NiI_5435")
            rotation = 270.0
            x1 = 0.0
            x2 = 0.0
            y1 = 0.015
            y2 = 0.28
        elseif contains(names_ann[j], "5436.3")
            rotation = 270.0
            x1 = 0.0
            x2 = 0.0
            y1 = 0.015
            y2 = 0.235
        elseif contains(names_ann[j], "5436.6")
            rotation = 270.0
            x1 = -0.15
            x2 = -0.4
            y1 = 0.015
            y2 = 0.22
        elseif contains(names_ann[j], "6169")
            rotation = 270.0
            x1 = 0.0
            x2 = 0.0
            y1 = 0.015
            y2 = 0.25
        else
            rotation = 270.0
            x1 = 0.0
            x2 = 0.0
            y1 = 0.025
            y2 = 0.22
        end

        # annotate with line name
        arrowprops = Dict("facecolor"=>"black", "lw"=>1.5, "arrowstyle"=>"-")
        if (contains(names_ann[j], "5896") | contains(names_ann[j], "5380"))
            title = replace(names_ann[j], "_" => "\\ ")
            idx = findfirst('I', title)
            title = title[1:idx-1] * "\\ " * title[idx:end]
            title = ("\$^\\ddagger{\\rm " * title * "}\$")
        else
            title = replace(names_ann[j], "_" => "\\ ")
            idx = findfirst('I', title)
            title = title[1:idx-1] * "\\ " * title[idx:end]
            title = ("\${\\rm " * title * "}\$")
        end
        txt = axs[i].annotate(title, rotation=rotation,
                              (wavs2[min] - x1, spec2[min] - y1),
                              (wavs2[min] - x2, spec2[min] - y2),
                              horizontalalignment="center", arrowprops=arrowprops,
                              fontsize=14)
        push!(texts, txt)
    end

    # set the xticks
    axs[i].set_xticks([wavmid-2, wavmid, wavmid+2])
    axs[i].xaxis.set_tick_params(rotation=45, labelsize=18)
    axs[i].yaxis.set_tick_params(labelsize=18)

    # deal with axis break decoration stuff
    d = 0.01
    if i > 1
        axs[i].yaxis.set_tick_params(left=false, labelleft=false)
        axs[i].spines["left"].set_visible(false)
        axs[i].plot([-d, +d], [-d, +d], transform=axs[i].transAxes, c="k", clip_on=false, lw=1)
        axs[i].plot([-d, +d], [1-d, 1+d], transform=axs[i].transAxes, c="k", clip_on=false, lw=1)
    end

    if i < length(dirs)
        axs[i].spines["right"].set_visible(false)
        axs[i].plot([1-d, 1+d], [-d, +d], transform=axs[i].transAxes, c="k", clip_on=false, lw=1)
        axs[i].plot([1-d, 1+d], [1-d, 1+d], transform=axs[i].transAxes, c="k", clip_on=false, lw=1)
    end
end

# make axis labels
fig.supxlabel(L"{\rm Wavelength\ (\AA)}", y=-0.01, fontsize=21)
fig.supylabel(L"{\rm Normalized\ Flux}", x=0.09, fontsize=21)

# save the fig
fig.savefig(joinpath(static, "spectra_collage.pdf"))
plt.clf(); plt.close()
