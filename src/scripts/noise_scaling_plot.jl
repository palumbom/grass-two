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

# set instrument strings
instruments = [L"{\rm (KPF)}", L"{\rm (NEID)}", L"{\rm (EXPRESS)}", L"{\rm (ESPRESSO\ UHR)}", L"{\rm (PEPSI)}", L"{\rm (Fictional)}"]

# set the color bar normalization
cnorm = mpl.colors.Normalize(vmin=0.3, vmax=0.6)
cnorm_impr = mpl.colors.Normalize(vmin=0.0, vmax=30.0)

# get stats along Ntrials
# slice the data for the given resolution
rvs_std = dropdims(mean(rvs_std_out, dims=3), dims=3)
rvs_std_decorr = dropdims(mean(rvs_std_decorr_out, dims=3), dims=3)

std_rvs_std = dropdims(std(rvs_std_out, dims=3), dims=3) #./ sqrt(size(rvs_std_out, 3))
std_rvs_std_decorr = dropdims(std(rvs_std_decorr_out, dims=3), dims=3) #./ sqrt(size(rvs_std_out, 3))

# get improvement
impr = 100.0 .* (1.0 .- (rvs_std_decorr ./ rvs_std))

# get uncertainty on improvement
impr_std = (rvs_std_decorr ./ rvs_std.^2.0).^2.0 .* std_rvs_std .^ 2.0
impr_std += (1.0 ./ rvs_std.^2.0).^2.0 .* std_rvs_std_decorr .^2.0
impr_std = 100.0 .* sqrt.(impr_std)

# plot heatmap at each resolution
if true
    println(">>> Plotting heatmap of RMS RVs")
    for i in eachindex(resolutions)
        @show resolutions[i]

        # take the right view
        rvs_std_view = view(rvs_std, i, :, :)
        rvs_std_decorr_view = view(rvs_std_decorr, i, :, :)
        std_rvs_std_view = view(std_rvs_std, i, :, :)
        std_rvs_std_decorr_view = view(std_rvs_std_decorr, i, :, :)
        impr_view = view(impr, i, :, :)
        impr_std_view = view(impr_std, i, :, :)

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

        # decide whether to include errors
        include_errors = false
        if include_errors
            for i in eachindex(rd1)
                if rvs_std_view[i] >= 10.0
                    rd1[i] *= L"\\ ~~ \pm\ " * latexstring.(round.(std_rvs_std_view[i], digits=2))
                else
                    rd1[i] *= L"\\ ~ \pm\ " * latexstring.(round.(std_rvs_std_view[i], digits=2))
                end

                if rvs_std_decorr_view[i] >= 10.0
                    rd2[i] *= L"\\ ~~ \pm\ " * latexstring.(round.(std_rvs_std_decorr_view[i], digits=2))
                else
                    rd2[i] *= L"\\ ~ \pm\ " * latexstring.(round.(std_rvs_std_decorr_view[i], digits=2))
                end
            end
        end

        for i in 0:length(nlines_to_do)-1
            for j in 0:length(snrs_for_lines)-1
                text1 = ax1.text(j, i, rd1[i+1, j+1], weight="bold", ha="center", va="center", color="w", fontsize=16)
                text2 = ax2.text(j, i, rd2[i+1, j+1], weight="bold", ha="center", va="center", color="w", fontsize=16)
            end
        end

        ax1.set_title(L"{\rm Uncorrected\ RVs}", fontsize=21)
        ax2.set_title(L"{\rm Corrected\ RVs}", fontsize=21)
        fig.savefig(joinpath(plotsubdir, template * "_" * string(i) * "_heatmap.pdf"))
        plt.close()
    end
end

# plot improvement heatmap at each resolution
if true
    println(">>> Plotting heatmap of improvements")
    for i in eachindex(resolutions)
        @show resolutions[i]

        # take the right view
        rvs_std_view = view(rvs_std, i, :, :)
        rvs_std_decorr_view = view(rvs_std_decorr, i, :, :)
        std_rvs_std_view = view(std_rvs_std, i, :, :)
        std_rvs_std_decorr_view = view(std_rvs_std_decorr, i, :, :)
        impr_view = view(impr, i, :, :)
        impr_std_view = view(impr_std, i, :, :)

        # now plot the improvement
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6.8, 8.6))
        img1 = ax1.imshow(impr_view, cmap="viridis", origin="lower", norm=cnorm_impr)

        ax1.set_xticks(0:length(snrs_for_lines)-1, labels=latexstring.(round.(Int, snrs_for_lines)), fontsize=18)
        ax1.set_yticks(0:length(nlines_to_do)-1, labels=latexstring.(nlines_to_do), fontsize=18)

        cb = fig.colorbar(img1, ax=ax1, shrink=0.85)
        cb.set_label(L"{\rm \%\ Improvement}", fontsize=20)
        cb.set_ticklabels(latexstring.(range(0, 30, step=5)), fontsize=18)

        ax1.set_xlabel(L"{\rm Per\ pixel\ SNR}", fontsize=21)
        ax1.set_ylabel(L"{\rm Number\ of\ lines\ in\ CCF}", fontsize=21)

        ax1.grid(false)
        for j in 1:length(snrs_for_lines)-1
            ax1.axvline(j-0.5, ls="--", c="k", alpha=0.75)
        end

        for j in 1:length(nlines_to_do)-1
            ax1.axhline(j-0.5, ls="--", c="k", alpha=0.75)
        end

        # assemble the string to proint
        rd1 = round.(impr_view, digits=1)
        rd2 = round.(impr_std_view, digits=1)
        rdp = latexstring.(rd1) .* L"\\ \ \ \pm\ " .* latexstring.(rd2)
        for i in eachindex(impr_view)
            if impr_view[i] >= 10.0
                rdp[i] = latexstring(rd1[i]) * L"\\ ~~ \pm\ " * latexstring(rd2[i])
            else
                rdp[i] = latexstring(rd1[i]) * L"\\ ~ \pm\ " * latexstring(rd2[i])
            end
        end

        for i in 0:length(nlines_to_do)-1
            for j in 0:length(snrs_for_lines)-1
                if impr_std_view[i+1, j+1] > impr_view[i+1, j+1]
                    text1 = ax1.text(j, i, latexstring(L"{\rm -}"), ha="center", va="center", color="w", fontsize=14)
                else
                    text1 = ax1.text(j, i, rdp[i+1, j+1], ha="center", va="center", color="w", fontsize=14)
                end
            end
        end

        fig.savefig(joinpath(plotsubdir, template * "_" * string(i) * "_improvement.pdf"), bbox_inches="tight")
        plt.close()
    end
end

# power law corrected rms
if true
    println(">>> Plotting power law of RMS RVs")
    for i in 1:length(resolutions)
        # parse the data
        rvs_std1 = view(rvs_std, i, :, :)
        rvs_std2 = view(rvs_std_decorr, i, :, :)
        sig_rvs_std1 = view(std_rvs_std, i, :, :)
        sig_rvs_std2 = view(std_rvs_std_decorr, i, :, :)
        impr_view = view(impr, i, :, :)
        impr_std_view = view(impr_std, i, :, :)

        # power law model
        @. power_law(x, p) = p[1] * x ^ (-p[2])

        # get xmodels
        xmodel2 =range(minimum(nlines_to_do)-2, maximum(nlines_to_do)+64, length=2000)

        # get colors
        cmap = plt.get_cmap("plasma")
        bounds = 1:length(snrs_for_lines)+2
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        # plot vs number of lines in ccf
        fig, ax1 = plt.subplots(figsize=1.2 .* (6.4,4.8))
        for i in 1:size(rvs_std1, 2)
            # slice the data
            ydata = rvs_std2[:,i]
            yerrs = sig_rvs_std2[:,i]
            wts = 1.0 ./ yerrs .^ 2.0

            # plot the data
            ax1.errorbar(nlines_to_do, ydata, yerr=yerrs, c=sm.to_rgba(i), ls="none", fmt="d", capsize=2.0)

            # get the model
            p0 = [1.0, 0.5]
            pl = curve_fit(power_law, nlines_to_do, ydata, wts, p0)

            # plot the model
            ax1.plot(xmodel2, power_law(xmodel2, pl.param), ls="--", c=sm.to_rgba(i), label=L"\alpha \sim\ " * latexstring(round(pl.param[2], digits=3)))
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
        cb = fig.colorbar(sm)
        cb.set_ticks(collect(1:length(snrs_for_lines)) .+ 0.5)
        cb.ax.set_ylim(1.0, length(snrs_for_lines) + 1.0)
        cb.set_ticklabels([latexstring(Int(i)) for i in snrs_for_lines])
        cb.set_label(L"{\rm Per\ pixel\ SNR}")

        # make the legend, axis labels
        ax1.legend()
        ax1.set_xlabel(L"{\rm Number\ of\ lines\ in\ CCF}")
        ax1.set_ylabel(L"{\rm Corrected\ RMS\ RV\ (m\ s}^{-1} {\rm )}")

        # figure out the title
        # from: https://stackoverflow.com/questions/52213829/in-julia-insert-commas-into-integers-for-printing-like-python-3-6
        function commas(num::Integer)
            str = string(num)
            return replace(str, r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => "{,}")
        end

        title_res = latexstring(commas(Int(resolutions[i])))
        ax1.set_title(L"R =\ " * title_res * L"\ " * instruments[i])

        # save the figure
        fig.tight_layout()
        fig.savefig(joinpath(plotsubdir, template * "_" * string(i) * "_power_law_scaling.pdf"))
        plt.clf()
        plt.close()
    end
end

# power law improvement
if true
    println(">>> Plotting power law of improvements")
    for i in 1:length(resolutions)
        @show resolutions[i]

        # parse the data
        rvs_std1 = view(rvs_std, i, :, :)
        rvs_std2 = view(rvs_std_decorr, i, :, :)
        sig_rvs_std1 = view(std_rvs_std, i, :, :)
        sig_rvs_std2 = view(std_rvs_std_decorr, i, :, :)
        impr_view = view(impr, i, :, :)
        impr_std_view = view(impr_std, i, :, :)

        # power law model
        @. power_law(x, p) = p[1] * x ^ (-p[2])

        # get xmodels
        xmodel2 =range(minimum(nlines_to_do)-2, maximum(nlines_to_do)+64, length=2000)

        # get colors
        cmap = plt.get_cmap("plasma")
        bounds = 1:length(snrs_for_lines)+2
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        # plot vs number of lines in ccf
        fig, ax1 = plt.subplots(figsize=1.2 .* (6.4,4.8))
        for i in 1:size(rvs_std1, 2)
            # slice the data
            ydata = impr_view[:,i]
            yerrs = impr_std_view[:,i]
            wts = 1.0 ./ yerrs .^ 2.0

            # decide whether to plot the data
            if all(yerrs .>= ydata)
                continue
            end

            # mask bad data points
            idx = ydata .> yerrs
            if sum(idx) < 3
                continue
            end

            # plot the data
            ax1.errorbar(nlines_to_do[idx], ydata[idx], yerr=yerrs[idx], c=sm.to_rgba(i), ls="none", fmt="d", capsize=2.0)

            # get the model
            p0 = [1.0, 0.5]
            pl = curve_fit(power_law, nlines_to_do[idx], ydata[idx], wts[idx], p0)

            # plot the model
            ax1.plot(xmodel2, power_law(xmodel2, pl.param), ls="--", c=sm.to_rgba(i), label=L"\alpha \sim\ " * latexstring(round(pl.param[2], digits=3)))
        end

        if length(ax1.lines) == 0
            println("\t>>> Nothing to plot!")
            plt.clf()
            plt.close()
            continue
        end

        # set the xscale
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log", base=2)

        # deal with the ticks
        # ax1.set_xticks([50, 100, 150, 200, 250, 300, 450, 550, 800])
        ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        # ax1.set_yticks(range(0.3, 1.2, step=0.1))
        ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

        # make colorbar
        cb = fig.colorbar(sm)
        cb.set_ticks(collect(1:length(snrs_for_lines)) .+ 0.5)
        cb.ax.set_ylim(1.0, length(snrs_for_lines) + 1.0)
        cb.set_ticklabels([latexstring(Int(i)) for i in snrs_for_lines])
        cb.set_label(L"{\rm Per\ pixel\ SNR}")

        # make the legend, axis labels
        ax1.legend()
        ax1.set_xlabel(L"{\rm Number\ of\ lines\ in\ CCF}")
        ax1.set_ylabel(L"{\rm \%\ Improvement}")

        # figure out the title
        # from: https://stackoverflow.com/questions/52213829/in-julia-insert-commas-into-integers-for-printing-like-python-3-6
        function commas(num::Integer)
            str = string(num)
            return replace(str, r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => "{,}")
        end

        title_res = latexstring(commas(Int(resolutions[i])))
        ax1.set_title(L"R =\ " * title_res * L"\ " * instruments[i])

        # save the figure
        fig.tight_layout()
        fig.savefig(joinpath(plotsubdir, template * "_" * string(i) * "_power_law_improvement_scaling.pdf"))
        plt.clf()
        plt.close()
    end
end
