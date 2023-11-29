# alias stuff
AA = AbstractArray
AF = AbstractFloat

# pkgs
using Base.Threads
using JLD2
using GRASS
using Peaks
using Printf
using FileIO
using Profile
using Statistics
using EchelleCCFs
using Polynomials
using BenchmarkTools
# using RvSpectML
# using AbstractGPs, TemporalGPs, KernelFunctions

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; #plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# get the name of template from the command line args
# template_idx = 1
template_idx = tryparse(Int, ARGS[1])
lp = GRASS.LineProperties(exclude=["CI_5380", "NaI_5896"])
line_names = GRASS.get_name(lp)
template = line_names[template_idx]

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
# include("paths.jl")
plotsubdir = string(joinpath(figures, "snr_plots"))

if !isdir(plotsubdir)
    mkdir(plotsubdir)
end

# read the data
d = load(string(joinpath(data, template * "_rvs_std_out.jld2")))
nlines_to_do = d["nlines_to_do"]
snrs_for_lines = d["snrs_for_lines"]
resolutions = d["resolutions"]
rvs_std_out = d["rvs_std_out"]
rvs_std_decorr_out = d["rvs_std_decorr_out"]

norm = mpl.colors.LogNorm(vmin=0.1, vmax=1.0)

# plot heatmap at each resolution
for i in eachindex(resolutions)
    # get views
    rvs_std_view = view(rvs_std_out, i, :, :)
    rvs_std_decorr_view = view(rvs_std_decorr_out, i, :, :)
    impr = 100.0 .* (rvs_std_view .- rvs_std_decorr_view) ./ rvs_std_view

    # # println(minimum(rvs_std_view))
    # # println(minimum(rvs_std_decorr_view))

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 8.6))
    # img1 = ax1.imshow(rvs_std_view, cmap="viridis_r", origin="lower", norm=norm)
    # img2 = ax2.imshow(rvs_std_decorr_view, cmap="viridis_r", origin="lower", norm=norm)

    # formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=false)

    # # fig.tight_layout()
    # fig.subplots_adjust(right=0.85)
    # fig.subplots_adjust(wspace=0.025)
    # cax = fig.add_axes([0.8675, 0.15, 0.04, 0.7])
    # cb = fig.colorbar(img2, cax=cax, ticks=range(0.1, 1.0, step=0.1))#, format=formatter)
    # cb.set_label(L"{\rm RV\ RMS\ (m\ s}^{-1} {\rm )}")
    # cb.set_ticklabels(latexstring.(range(0.1, 1.0, step=0.1)))

    # ax1.set_xticks(0:length(snrs_for_lines)-1, labels=string.(round.(Int, snrs_for_lines)))
    # ax2.set_xticks(0:length(snrs_for_lines)-1, labels=string.(round.(Int, snrs_for_lines)))
    # ax1.set_yticks(0:length(nlines_to_do)-1, labels=string.(nlines_to_do))
    # ax2.set_yticks(0:length(nlines_to_do)-1, labels=[])

    # ax1.set_xlabel(L"{\rm Per\ pixel\ SNR}")
    # ax2.set_xlabel(L"{\rm Per\ pixel\ SNR}")
    # ax1.set_ylabel(L"{\rm Number\ of\ lines\ in\ CCF}")

    # ax1.grid(false)
    # ax2.grid(false)
    # for j in 1:length(snrs_for_lines)-1
    #     ax1.axvline(j-0.5, ls="--", c="k", alpha=0.75)
    #     ax2.axvline(j-0.5, ls="--", c="k", alpha=0.75)
    # end

    # for j in 1:length(nlines_to_do)-1
    #     ax1.axhline(j-0.5, ls="--", c="k", alpha=0.75)
    #     ax2.axhline(j-0.5, ls="--", c="k", alpha=0.75)
    # end

    # rd1 = round.(rvs_std_view, digits=2)
    # rd2 = round.(rvs_std_decorr_view, digits=2)

    # for i in 0:length(nlines_to_do)-1
    #     for j in 0:length(snrs_for_lines)-1
    #         text1 = ax1.text(j, i, rd1[i+1, j+1], ha="center", va="center", color="w")
    #         text2 = ax2.text(j, i, rd2[i+1, j+1], ha="center", va="center", color="w")
    #     end
    # end

    # ax1.set_title(L"{\rm Uncorrected\ RVs}")
    # ax2.set_title(L"{\rm Corrected\ RVs}")

    # fig.savefig(joinpath(plotsubdir, template * "_" * string(i) * "_heatmap.pdf"))
    # plt.close()


    # now plot the improvement
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12.8, 4.3))
    img1 = ax1.imshow(impr, cmap="viridis", origin="lower", vmin=0.0, vmax=15.0)

    cb = fig.colorbar(img1, ax=ax1)
    cb.set_label(L"{\rm \%\ Improvement}")

    ax1.set_xlabel(L"{\rm Per\ pixel\ SNR}")
    ax1.set_ylabel(L"{\rm Number\ of\ lines\ in\ CCF}")

    ax1.grid(false)
    for j in 1:length(snrs_for_lines)-1
        ax1.axvline(j-0.5, ls="--", c="k", alpha=0.75)
    end

    for j in 1:length(nlines_to_do)-1
        ax1.axhline(j-0.5, ls="--", c="k", alpha=0.75)
    end

    rd1 = round.(impr, sigdigits=2)

    for k in eachindex(rd1)
        if rd1[k] < 0.1
            rd1[k] = round(rd1[k], digits=2)
        end
    end

    for i in 0:length(nlines_to_do)-1
        for j in 0:length(snrs_for_lines)-1
            text1 = ax1.text(j, i, rd1[i+1, j+1], ha="center", va="center", color="w", fontsize=10)
        end
    end

    fig.savefig(joinpath(plotsubdir, template * "_" * string(i) * "_improvement.pdf"), bbox_inches="tight")
    plt.close()
end

# set up colors for plots
pcolors = plt.cm.viridis(range(0, 1, length=length(snrs_for_lines)))

# # set up plots
# fig1, ax1 = plt.subplots(figsize=(7.2,4.8))

# # plot snr vs number of lines
# for i in eachindex(snrs_for_lines)
#     ax1.plot(nlines_to_do, rvs_std_out[end,:,i], label="SNR = " * string(snrs_for_lines[i]), c=pcolors[i,:])
# end

# # set axis stuff
# ax1.set_xlabel(L"{\rm Number\ of\ lines\ in\ CCF}")
# ax1.set_ylabel(L"{\rm RV\ RMS\ (m\ s}^{-1} {\rm )}")
# ax1.set_xscale("log", base=2)
# ax1.set_yscale("log", base=2)
# ax1.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
# plt.show()


# fig1.savefig(string(joinpath(plotsubdir, template * "_std_vs_number_of_lines.pdf")))
# plt.clf(); plt.close("all")

# set up plots
# fig1, ax1 = plt.subplots(figsize=(7.2,4.8))

# # plot snr vs number of lines
# for i in eachindex(snrs_for_lines)

#     dat = (rvs_std_out[3, :,i] .- rvs_std_decorr_out[3, :,i]) ./ rvs_std_out[3, :,i]
#     ax1.plot(nlines_to_do, dat .* 100, label="SNR = " * string(snrs_for_lines[i]), c=pcolors[i,:])
# end

# # set axis stuff
# ax1.set_xlabel(L"{\rm Number\ of\ lines\ in\ CCF}")
# ax1.set_ylabel(L"{\rm \% \ Improvement\ in\ RV\ RMS}")
# ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
# fig1.tight_layout()
# plt.show()
# fig1.savefig(string(joinpath(plotsubdir, template * "_improvement_vs_number_of_lines_same.pdf")))
# plt.clf(); plt.close("all")

