# imports
using JLD2
using CUDA
using GRASS
using Printf
using FileIO
using Revise
using Statistics
using EchelleCCFs
using Distributions
using BenchmarkTools

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

function format_sci_not(number::Float64)
    formatted_number = @sprintf("%.2e", number)
    idx = findfirst('e', formatted_number)
    mantissa = formatted_number[1:idx-1]
    exponent = formatted_number[idx+1:end]
    if exponent[2] == '0'
        exponent = exponent[1] * exponent[3:end]
    end
    exponent = "10^{" * exponent * "}"
    latex_string = "\\(" * mantissa * :"\\times" * exponent * "\\)"
    return latex_string
end

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
const datafile = string(abspath(joinpath(data, "gpu_accuracy.jld2")))
const plotfile = string(abspath(joinpath(figures, "gpu_accuracy.pdf")))

# read in the data
d = load(datafile)
wavs_cpu64 = d["wavs_cpu64"]
flux_cpu64 = d["flux_cpu64"]
wavs_gpu64 = d["wavs_gpu64"]
flux_gpu64 = d["flux_gpu64"]
wavs_gpu32 = d["wavs_gpu32"]
flux_gpu32 = d["flux_gpu32"]
resids32 = d["resids32"]
resids64 = d["resids64"]

# set up plot
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(7,9.5), sharex=true)

ms = 2.0
ax1.plot(wavs_cpu64, flux_cpu64, ls="-", c="k", ms=ms, lw=3.0, label=L"{\rm CPU\ (Float64)}", zorder=0)
# ax1.plot(wavs_gpu64, flux_gpu_mean64, ls="--", c=colors[1], ms=ms, label=L"{\rm GPU\ (Float64)}")
ax1.scatter(wavs_gpu64, flux_gpu64, marker="s", alpha=0.75, c=colors[1], s=3.0, label=L"{\rm GPU\ (Float64)}", zorder=1)
# ax1.plot(wavs_gpu32, flux_gpu_mean32, ls=":", c=colors[2], ms=ms, label=L"{\rm GPU\ (Float32)}")
ax1.scatter(wavs_gpu32, flux_gpu32, marker="^", alpha=0.75, c=colors[2], s=3.0, label=L"{\rm GPU\ (Float32)}", zorder=2)
ax2.scatter(wavs_cpu64, resids64, s=5, marker="s", c=colors[1], alpha=0.9,
            label=L"{\rm |\ Max.\ residual\ | } \sim\ " * format_sci_not(maximum(abs.(resids64))))
ax3.scatter(wavs_cpu64, resids32, s=5, marker="^", c=colors[2], alpha=0.9,
            label=L"{\rm |\ Max.\ residual\ | } \sim\ " * format_sci_not(maximum(abs.(resids32))))

# inset axis for marginal distribution of residuals
ax2_histy = ax2.inset_axes([1.0015, 0, 0.075, 1], sharey=ax2)
ax2_histy.hist(resids64, bins="auto", density=true, histtype="step", orientation="horizontal", color=colors[1])
ax2_histy.tick_params(axis="both", labelleft=false, labelbottom=false)
ax2_histy.set_xticks([])
ax2_histy.get_yaxis().set_visible(false)
ax2_histy.spines["left"].set_visible(false)
ax2_histy.spines["bottom"].set_visible(false)
ax2_histy.spines["top"].set_visible(false)
ax2_histy.spines["right"].set_visible(false)
ax2_histy.grid(false)

ax3_histy = ax3.inset_axes([1.0015, 0, 0.075, 1], sharey=ax3)
ax3_histy.hist(resids32, bins="auto", density=true, histtype="step", orientation="horizontal", color=colors[2])
ax3_histy.tick_params(axis="both", labelleft=false, labelbottom=false)
ax3_histy.set_xticks([])
ax3_histy.get_yaxis().set_visible(false)
ax3_histy.spines["left"].set_visible(false)
ax3_histy.spines["bottom"].set_visible(false)
ax3_histy.spines["top"].set_visible(false)
ax3_histy.spines["right"].set_visible(false)
ax3_histy.grid(false)

ax1.set_xlim(minimum(wavs_cpu64)-0.1, maximum(wavs_cpu64)+0.1)
ax1.set_ylim(0.15, 1.05)

# do manual formatting of tick labels since the default is ROUGH
if maximum(abs.(resids64)) < 1e-13
    ax2.set_ylim(-1.1e-14, 1.1e-14)
    ax2.ticklabel_format(axis="y", useOffset=true, style="sci")
    ax2.yaxis.offsetText.set_visible(false)
    ax2.set_ylabel(L"({\rm Flux}_{\rm CPU} - {\rm Flux}_{\rm GPU}) \times 10^{14}", fontsize=15)
end

if maximum(abs.(resids32)) < 1e-2
    ax3.set_ylim(-1.5e-3, 1.5e-3)
    ax3.ticklabel_format(axis="y", useOffset=true, style="sci")
    ax3.yaxis.offsetText.set_visible(false)
    ax3.set_yticks([-1.0e-3, -0.5e-3, 0.0, 0.5e-3, 1.0e-3])
    ax3.set_yticklabels([L"-1.0", L"-0.5", L"0.0", L"0.5", L"1.0"])
    ax3.set_ylabel(L"({\rm Flux}_{\rm CPU} - {\rm Flux}_{\rm GPU}) \times 10^{3}", fontsize=15)
end

legend = ax1.legend(loc="lower left", mode="expand", ncol=3, fontsize=12.5,
                    bbox_to_anchor=(0, 1.02, 1, 0.2), handletextpad=0.33)
for legend_handle in legend.legendHandles
    legend_handle._sizes = [20.0]
end

legend = ax2.legend(loc="upper left", handletextpad=-1.75)
for item in legend.legendHandles
    item.set_visible(false)
end

legend = ax3.legend(loc="upper left", handletextpad=-1.75)
for item in legend.legendHandles
    item.set_visible(false)
end


ax1.set_ylabel(L"{\rm Normalized\ Flux}", labelpad=18)
ax3.set_xlabel(L"{\rm Wavelength\ (\AA)}")

fig.tight_layout()
fig.subplots_adjust(hspace=0.05)
fig.savefig(plotfile)
plt.clf(); plt.close()
