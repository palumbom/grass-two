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
using LsqFit
using PyCall
using Profile
using Statistics
using EchelleCCFs
using Polynomials
using BenchmarkTools

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; #plt.ioff()
pe = pyimport("matplotlib.patheffects")
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# get the name of template from the command line args
template_idx = 8 #tryparse(Int, ARGS[1])
lp = GRASS.LineProperties(exclude=["CI_5380", "NaI_5896"])
line_names = GRASS.get_name(lp)
template = line_names[template_idx]
@show template

# get command line args and output directories
# include(joinpath(abspath(@__DIR__), "paths.jl"))
include("paths.jl")
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

# set the color bar normalization
# cnorm = mpl.colors.LogNorm(vmin=0.3, vmax=0.6)
cnorm = mpl.colors.Normalize(vmin=0.25, vmax=0.65)

# plot heatmap at each resolution
if false
    for i in eachindex(resolutions)
        @show resolutions[i]

        # slice the data for the given resolution
        rvs_std_view = dropdims(mean(view(rvs_std_out, i, :, :, :), dims=2), dims=2)
        rvs_std_decorr_view = dropdims(mean(view(rvs_std_decorr_out, i, :, :, :), dims=2), dims=2)

        std_rvs_std_view = dropdims(std(view(rvs_std_out, i, :, :, :), dims=2), dims=2)
        std_rvs_std_decorr_view = dropdims(std(view(rvs_std_decorr_out, i, :, :, :), dims=2), dims=2)

        # get improvement
        impr = 100.0 .* (rvs_std_view .- rvs_std_decorr_view) ./ rvs_std_view

        # create figure objects
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 8.6))
        img1 = ax1.imshow(rvs_std_view, cmap="viridis_r", origin="lower", norm=cnorm)
        img2 = ax2.imshow(rvs_std_decorr_view, cmap="viridis_r", origin="lower", norm=cnorm)

        # fig.tight_layout()
        fig.subplots_adjust(right=0.85)
        fig.subplots_adjust(wspace=0.025)
        cax = fig.add_axes([0.8675, 0.15, 0.04, 0.7])
        cb = fig.colorbar(img2, cax=cax, ticks=range(0.1, 1.0, step=0.1))
        cb.set_label(L"{\rm RMS\ RV\ (m\ s}^{-1} {\rm )}", fontsize=20)
        cb.set_ticklabels(latexstring.(range(0.1, 1.0, step=0.1)), fontsize=18)

        ax1.set_xticks(0:length(snrs_for_lines)-1, labels=latexstring.(round.(Int, snrs_for_lines)), fontsize=18)
        ax2.set_xticks(0:length(snrs_for_lines)-1, labels=latexstring.(round.(Int, snrs_for_lines)), fontsize=18)
        ax1.set_yticks(0:length(nlines_to_do)-1, labels=latexstring.(nlines_to_do), fontsize=18)
        ax2.set_yticks(0:length(nlines_to_do)-1, labels=[], fontsize=18)

        ax1.set_xlabel(L"{\rm Per\ pixel\ SNR}", fontsize=21)
        ax2.set_xlabel(L"{\rm Per\ pixel\ SNR}", fontsize=21)
        ax1.set_ylabel(L"{\rm Number\ of\ lines\ in\ CCF}", fontsize=21)

        # plot grid lines manually
        ax1.grid(false)
        ax2.grid(false)
        for j in 1:length(snrs_for_lines)-1
            ax1.axvline(j-0.5, ls="--", c="k", alpha=0.75)
            ax2.axvline(j-0.5, ls="--", c="k", alpha=0.75)
        end

        for j in 1:length(nlines_to_do)-1
            ax1.axhline(j-0.5, ls="--", c="k", alpha=0.75)
            ax2.axhline(j-0.5, ls="--", c="k", alpha=0.75)
        end

        # round the digits
        rd1 = latexstring.(round.(rvs_std_view, digits=2))
        rd2 = latexstring.(round.(rvs_std_decorr_view, digits=2))
        # rd1 = round.(rvs_std_view, digits=2)
        # rd2 = round.(rvs_std_decorr_view, digits=2)

        # make path effects
        pe_text = [pe.Stroke(linewidth=0.025, foreground="k")]

        for i in 0:length(nlines_to_do)-1
            for j in 0:length(snrs_for_lines)-1
                text1 = ax1.text(j, i, rd1[i+1, j+1], weight="bold", ha="center", va="center", color="w", fontsize=16)#, path_effects=pe_text)
                text2 = ax2.text(j, i, rd2[i+1, j+1], weight="bold", ha="center", va="center", color="w", fontsize=16)#, path_effects=pe_text)
            end
        end

        ax1.set_title(L"{\rm Uncorrected\ RVs}", fontsize=21)
        ax2.set_title(L"{\rm Corrected\ RVs}", fontsize=21)

        fig.savefig(joinpath(plotsubdir, template * "_" * string(i) * "_heatmap.pdf"))
        plt.close()

    #=    # now plot the improvement
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6.8, 8.6))
        img1 = ax1.imshow(impr, cmap="viridis", origin="lower", vmin=0.0, vmax=30.0)

        ax1.set_xticks(0:length(snrs_for_lines)-1, labels=string.(round.(Int, snrs_for_lines)), fontsize=14)
        ax1.set_yticks(0:length(nlines_to_do)-1, labels=string.(nlines_to_do), fontsize=14)

        cb = fig.colorbar(img1, ax=ax1, shrink=0.8)
        cb.set_label(L"{\rm \%\ Improvement}")

        ax1.set_xlabel(L"{\rm Per\ pixel\ SNR}", fontsize=16)
        ax1.set_ylabel(L"{\rm Number\ of\ lines\ in\ CCF}", fontsize=16)

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
                text1 = ax1.text(j, i, rd1[i+1, j+1], ha="center", va="center", color="w", fontsize=14)
            end
        end

        fig.savefig(joinpath(plotsubdir, template * "_" * string(i) * "_improvement.pdf"), bbox_inches="tight")
        plt.close()=#
    end
end

# parse out the highest res sim
idx = length(resolutions) - 2

rvs_std1 = dropdims(mean(rvs_std_out[idx, :, :, :], dims=2), dims=2)
rvs_std2 = dropdims(mean(rvs_std_decorr_out[idx, :, :, :], dims=2), dims=2)

sig_rvs_std1 = dropdims(std(rvs_std_out[idx, :, :, :], dims=2), dims=2)
sig_rvs_std2 = dropdims(std(rvs_std_decorr_out[idx, :, :, :], dims=2), dims=2)

# wts1 = 1.0 ./ sig_rvs_std1 .^2.0
# wts2 = 1.0 ./ sig_rvs_std2 .^2.0

# get improvement
impr = 100.0 .* (rvs_std1 .- rvs_std2) ./ rvs_std1

# power law model
@. power_law(x, p) = p[1] * x ^ (-p[2])

# get xmodels
xmodel1 = range(minimum(snrs_for_lines), maximum(snrs_for_lines), length=1000)
xmodel2 =range(minimum(nlines_to_do), maximum(nlines_to_do)+100, length=1000)

#=# now plot improvement as a function of per pixel snr
pcolors = plt.cm.viridis(range(0, 1, length=length(nlines_to_do)))
fig1, ax1 = plt.subplots()
for i in 1:size(impr,1)
    # plot the data
    ax1.errorbar(snrs_for_lines, rvs_std2[i,:], yerr=sig_rvs_std2[i,:],  c=pcolors[i,:], ls="none", fmt="o", capsize=2.0)

    # get the model
    p0 = [1.0, 0.5]
    pl = curve_fit(power_law, snrs_for_lines, rvs_std2[i,:], wts2[i,:], p0)

    # plot the model
    ax1.plot(xmodel1, power_law(xmodel1, pl.param), ls="--", c=c=pcolors[i,:], label=L"\alpha =\ " *latexstring(round(pl.param[2], digits=3)))
end
# ax1.set_xscale("log", base=2)
# ax1.set_yscale("log", base=2)
ax1.legend()
ax1.set_xlabel(L"{\rm Per\ pixel\ SNR}")
ax1.set_ylabel(L"{\rm Corrected\ RMS\ RV\ (m\ s}^{-1} {\rm )}")
plt.show()
=#


# get colors
cmap = plt.get_cmap("plasma")
bounds = 1:length(snrs_for_lines)+2
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

# plot vs number of lines in ccf
fig, ax1 = plt.subplots(figsize=1.2 .* (6.4,4.8))
for i in 1:size(impr,2)
    # plot the data
    ax1.errorbar(nlines_to_do, rvs_std2[:,i], yerr=sig_rvs_std2[:,i], c=sm.to_rgba(i), ls="none", fmt="o", capsize=2.0)

    # get the model
    p0 = [1.0, 0.5]
    pl = curve_fit(power_law, nlines_to_do, rvs_std2[:,i], wts2[:,i], p0)

    # plot the model
    ax1.plot(xmodel2, power_law(xmodel2, pl.param), ls="--", c=sm.to_rgba(i), label=L"\alpha \sim\ " * latexstring(round(pl.param[2], digits=2)))
end

# set the xscale
ax1.set_xscale("log", base=2)
ax1.set_yscale("log", base=2)

# deal with the ticks
# ax1.set_xticks([50, 100, 150, 200, 250, 300, 450, 550, 800])
ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.set_yticks(range(0.3, 1.2, step=0.1))
ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

# make colorbar
# sm.set_array([])
cb = fig.colorbar(sm)
cb.set_ticks(collect(1:length(snrs_for_lines)) .+ 0.5)
cb.ax.set_ylim(1.0, length(snrs_for_lines) + 1.0)
cb.set_ticklabels([latexstring(Int(i)) for i in snrs_for_lines])
cb.set_label(L"{\rm Per\ pixel\ SNR}")

# make the legend, axis labels
ax1.legend()
ax1.set_xlabel(L"{\rm Number\ of\ lines\ in\ CCF}")
ax1.set_ylabel(L"{\rm Corrected\ RMS\ RV\ (m\ s}^{-1} {\rm )}")
fig.tight_layout()
fig.savefig(joinpath(plotsubdir, template * "_power_law_scaling.pdf"))
plt.show()


# # set up colors for plots
# pcolors = plt.cm.viridis(range(0, 1, length=length(snrs_for_lines)))

# # set up plots
# fig1, ax1 = plt.subplots(figsize=(7.2,4.8))

# # plot snr vs number of lines
# for i in eachindex(snrs_for_lines)
#     ax1.plot(nlines_to_do, rvs_std_out[end,:,i], label="SNR = " * string(snrs_for_lines[i]), c=pcolors[i,:])
# end

# # set axis stuff
# ax1.set_xlabel(L"{\rm Number\ of\ lines\ in\ CCF}")
# ax1.set_ylabel(L"{\rm RMS\ RV\ (m\ s}^{-1} {\rm )}")
# # ax1.set_xscale("log", base=2)
# # ax1.set_yscale("log", base=2)
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
# ax1.set_ylabel(L"{\rm \% \ Improvement\ in\ RMS\ RV}")
# ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
# fig1.tight_layout()
# plt.show()
# fig1.savefig(string(joinpath(plotsubdir, template * "_improvement_vs_number_of_lines_same.pdf")))
# plt.clf(); plt.close("all")

