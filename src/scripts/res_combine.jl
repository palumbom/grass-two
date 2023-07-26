# imports
using CSV
using CUDA
using JLD2
using GRASS
using Printf
using Revise
using FileIO
using DataFrames
using Statistics
using EchelleCCFs
using Polynomials

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
const datafile = string(abspath(joinpath(data, "spectra_for_bin.jld2")))

function make_ccf_plots(wavs, flux, lines, depths, title, filename; plot_correlation=true)
    # calculate a ccf for one line
    v_grid, ccf1 = calc_ccf(wavs, flux, lines, depths,
                            7e5, mask_type=EchelleCCFs.TopHatCCFMask,
                            Δv_step=125.0)
    rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1)
    bis, int = GRASS.calc_bisector(v_grid, ccf1, nflux=20)
    bis_inv_slope = GRASS.calc_bisector_inverse_slope(bis, int)

    # get data to fit / plot
    xdata = rvs1 .- mean(rvs1)
    ydata = bis_inv_slope .- mean(bis_inv_slope)

    # do the fit
    pfit = Polynomials.fit(xdata, ydata, 1)
    xmodel = range(-2, 2, length=5)
    ymodel = pfit.(xmodel)

    # get slope string to annotate
    slope = round(coeffs(pfit)[2], digits=3)
    fit_label = "\$ " .* string(slope) .* "\$"

    # calc mean to plot
    mean_bis = dropdims(mean(bis, dims=2), dims=2)[2:end-2]
    mean_int = dropdims(mean(int, dims=2), dims=2)[2:end-2]
    mean_bis .-= mean(mean_bis)

    # make plotting objects
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8))
    fig.subplots_adjust(wspace=0.05)
    # ax1.set_box_aspect(0.75)
    # ax2.set_box_aspect(0.75)

    ax1.plot(mean_bis, mean_int, c="k", lw=3.0)
    ax2.scatter(xdata, ydata, c="k", s=1.5, alpha=0.9)
    if plot_correlation
        label = L"{\rm Slope } \approx\ " * fit_label
        ax2.plot(xmodel, ymodel, c="k", ls="--", label=label, lw=2.5)
        ax2.legend()
    end

    ax1.set_xlabel(L"\Delta v\ {\rm (m\ s}^{-1}{\rm )}")
    ax1.set_ylabel(L"{\rm Normalized\ CCF}")
    ax2.set_xlabel(L"{\rm RV\ - \overline{\rm RV}\ } {\rm (m\ s}^{-1}{\rm )}")
    ax2.set_ylabel(L"{\rm BIS}\ - \overline{\rm BIS}\ {\rm (m\ s}^{-1}{\rm )}")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    if !isempty(title)
        fig.suptitle("{\\rm " * replace(title, " "=> "\\ ") * "}", y=0.98)
    end
    fig.savefig(filename)
    plt.clf(); plt.close()
end

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

# get lines
line_names = GRASS.get_name(lp)
idx = findall(x -> occursin.("FeI_6302", x), templates)

# make resolution grid
resolutions = reverse([0.98e5, 1.2e5, 1.37e5, 2.7e5, 7e5])
oversampling = 4.0

for i in eachindex(resolutions)
    res = resolutions[i]

    if res != 7e5
        # do an initial conv to get output size
        wavs_to_deg = view(wavs, :, 1)
        flux_to_deg = view(flux, :, 1)

        wavs_degd, flux_degd = GRASS.convolve_gauss(wavs_to_deg, flux_to_deg, new_res=res,
                                                    oversampling=oversampling)

        # allocate memory
        wavs_out = zeros(size(wavs_degd, 1), size(flux, 2))
        flux_out = zeros(size(wavs_degd, 1), size(flux, 2))

        # loop over epochs and convolve
        for j in 1:size(flux,2)
            flux_to_deg = view(flux, :, j)
            wavs_degd, flux_degd = GRASS.convolve_gauss(wavs_to_deg, flux_to_deg, new_res=res,
                                                        oversampling=oversampling)

            # copy to array
            wavs_out[:, j] .= wavs_degd
            flux_out[:, j] .= flux_degd
        end
        wavs_out = wavs_out[:,1]
    else
        wavs_out = copy(wavs)
        flux_out = copy(flux)
    end

    # make plots
    outfile = string(abspath(joinpath(figures, string(res) * "_rv_vs_bis.pdf")))
    make_ccf_plots(wavs_out, flux_out, lines[idx], depths[idx], string(res), outfile)
end
