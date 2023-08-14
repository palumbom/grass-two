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
const datafile = string(abspath(joinpath(data, "picket_fence.jld2")))
plotsubdir = string(joinpath(figures, "snr_plots"))
if !isdir(plotsubdir)
    mkdir(plotsubdir)
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

# get separation between lines
line_sep = mean(diff(sort(lines)))

function get_lines(wavs, flux, last_line_number)
    idx = findfirst(x -> x .>= lines[last_line_number] + 0.5 * line_sep, wavs)
    return view(wavs, 1:idx), view(flux, 1:idx, :)
end

function get_one_line(wavs, flux, line_number)
    idx1 = findfirst(x -> x .>= lines[line_number] - 0.5 * line_sep, wavs)
    idx2 = findfirst(x -> x .>= lines[line_number] + 0.5 * line_sep, wavs)
    return view(wavs, idx1:idx2), view(flux, idx1:idx2, 1)
end

# TODO do they not match or is this numerical noise????
for i in 1:10
    # get just line
    wavs_temp, flux_temp = get_one_line(wavs, flux, i)

    # calculate the depth
    println(maximum(flux_temp[:,1]) .- minimum(flux_temp[:,1]))

    # get bisector
    bis, int = GRASS.calc_bisector(wavs_temp, flux_temp, nflux=50)

    vel = GRASS.c_ms * (bis .- lines[i]) ./ (lines[i])

    plt.plot(vel, int)
end
plt.show()


#=nlines_to_do = 100
function std_vs_number_of_lines(snr::T) where T<:Float64
    #


    # allocate memory
    rvs_std = zeros(nlines_to_do)

    # include more and more lines in ccf
    for i in 1:nlines_to_do
        # get lines to include in ccf
        ls = lines[1:i]
        ds = depths[1:i]

        # get view of spectrum
        wavs_temp, flux_temp = get_lines(wavs, flux, i)

        # get spectrum at specified snr
        wavs_snr = copy(wavs_temp)
        flux_snr = copy(flux_temp)
        GRASS.add_noise!(flux_snr, snr)

        # calculate ccf
        v_grid, ccf1 = calc_ccf(wavs_snr, flux_snr, ls, ds, 7e5)
        rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1)
        rvs_std[i] = std(rvs1)

        # get ccf bisector
        bis, int = GRASS.calc_bisector(v_grid, ccf1, nflux=50, top=0.99)
        bis_inv_slope = GRASS.calc_bisector_inverse_slope(bis, int)

        # scatter plot with correlations
        xdata = rvs1 .- mean(rvs1)
        ydata = bis_inv_slope .- mean(bis_inv_slope)

        pfit = Polynomials.fit(xdata, ydata, 1)
        xmodel = range(minimum(xdata), maximum(xdata), length=5)
        ymodel = pfit.(xmodel)

        slope = round(coeffs(pfit)[2], digits=3)
        fit_label = "\$ " .* string(slope) .* "\$"

        # plot
        fig2, ax2 = plt.subplots()
        ax2.scatter(xdata, ydata, c="k", s=2)
        ax2.plot(xmodel, ymodel, ls="--", c="k", label=L"{\rm Slope } \approx\ " * fit_label, lw=2.5)

        ax2.set_xlabel(L"{\rm RV\ - \overline{\rm RV}\ } {\rm (m\ s}^{-1}{\rm )}")
        ax2.set_ylabel(L"{\rm BIS}\ - \overline{\rm BIS}\ {\rm (m\ s}^{-1}{\rm )}")
        ax2.set_title("SNR = " * string(Int(snr)) * ", Nlines = " * string(i))
        ax2.legend()
        ofile = string(joinpath(plotsubdir, "rv_vs_bis_snr_" * string(Int(snr)) * "_lines_" * string(i) * ".pdf"))
        fig2.savefig(ofile)
        plt.clf()
    end
    return rvs_std
end

# snrs to loop over
snrs_for_lines = range(100.0, 1000.0, step=100.0)

# set up colors
pcolors = plt.cm.rainbow(range(0, 1, length=snrs_for_lines))
rvs_std_out = zeros(nlines_to_do, length(snrs_for_lines))

for i in eachindex(snrs_for_lines)
    # get the stuff
    out = std_vs_number_of_lines(snrs_for_lines[i])
    rvs_std_out[:, i] = out
end

# write the data
jldsave(string(joinpath(data, "rvs_std_out.jld2")), snrs_for_lines=snrs_for_lines, rvs_std_out=rvs_std_out)

# set up plots
fig1, ax1 = plt.subplots()

# plot snr vs number of lines
for i in eachindex(snrs_for_lines)
    ax1.plot(1:nlines_to_do, rvs_std_out[:,i], label="SNR = " * string(snrs_for_lines[i]), c=pcolors[i,:])
end

ax1.set_xlabel("Number of lines")
ax1.set_ylabel("RV RMS (m/s)")
ax1.set_xscale("log", base=2)
ax1.set_yscale("log", base=2)
ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig1.savefig(string(joinpath(figures, "std_vs_number_of_lines_same.pdf")))
plt.clf(); plt.close("all")

=#









#=
# collect lines
lines = collect(lines)

# isolate a chunk of spectrum around a good line
idx = findall(x -> occursin.("FeI_5576", x), templates)
line_centers = sort!(lines[idx])
line_center = line_centers[5]

buffer = 1.0
idx1 = findlast(x -> x .<= line_center - buffer, wavs)
idx2 = findlast(x -> x .<= line_center + buffer, wavs)

# take isolated view
wavs_iso = copy(view(wavs, idx1:idx2))
flux_iso = copy(view(flux, idx1:idx2, :))

# make noises to loop over
SNRs = range(100.0, 1000.0, step=10.0)

# # allocate memory for rvs stuff
# rv_std = zeros(length(SNRs))

# # loop over them
# flux_noise = similar(flux_iso)
# for i in eachindex(SNRs)
#     # copy over SNR infinity flux
#     flux_noise .= flux_iso

#     # add noise
#     GRASS.add_noise!(flux_noise, SNRs[i])

#     # calculate ccf
#     v_grid, ccf1 = calc_ccf(wavs_iso, flux_noise, lines[idx], depths[idx], 7e5)
#     rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1)
#     rv_std[i] = std(rvs1)
# end

# plt.plot(SNRs, rv_std)
# plt.show()
=#
