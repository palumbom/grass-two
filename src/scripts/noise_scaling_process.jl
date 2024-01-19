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
template_idx = tryparse(Int, ARGS[1])
lp = GRASS.LineProperties(exclude=["CI_5380", "NaI_5896"])
line_names = GRASS.get_name(lp)
template = line_names[template_idx]

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
plotsubdir = string(joinpath(figures, "snr_plots"))
datafile = string(abspath(joinpath(data, template * "_picket_fence.jld2")))

if !isdir(plotsubdir)
    mkdir(plotsubdir)
end

function std_vs_number_of_lines(wavs::AA{T,1}, flux::AA{T,2},
                                resolutions::AA{T,1},
                                nlines_to_do::AA{Int,1},
                                rvs_std::AA{T,2},
                                rvs_std_decorr::AA{T,2}, snr::T;
                                plot::Bool=false, oversampling::T=4.0) where T<:Float64
    # include more and more lines in ccf
    for i in eachindex(resolutions)
        # degrade the resolution
        wavs_degd, flux_degd = GRASS.convolve_gauss(wavs, flux,
                                                    new_res=resolutions[i],
                                                    oversampling=oversampling)

        # get the location of the minima
        pks, vals = findminima(flux_degd[:,1])
        pks, proms = peakproms!(pks, flux_degd[:,1]; minprom=0.1)

        # get view of first line
        wavs_window = view(wavs_degd, 1:pks[1]+round(Int, (pks[2] - pks[1])/2))
        flux_window = view(flux_degd, 1:pks[1]+round(Int, (pks[2] - pks[1])/2), 1)

        # get width of line at ~92.5% flux
        idxl, idxr = GRASS.find_wing_index(0.925, flux_window)

        # get width in angstroms
        width_ang = wavs_window[idxr] - wavs_window[idxl]

        # convert to velocity
        width_vel = GRASS.c_ms * width_ang / wavs_degd[pks[1]]

        # get the velocity step and velocity window for CCF
        Δv_step = 50.0
        Δv_max = round((width_vel + 1e3)/100) * 100

        v_grid = range(-Δv_max, Δv_max, step=Δv_step)

        # allocate memory that will be reused in line loop
        len_v = 1 + round(Int, (Δv_max * 2) / Δv_step)
        ccf1 = zeros(len_v, size(flux_degd,2))
        projection_full = zeros(length(wavs_degd), 1)
        proj_flux_full = zeros(length(wavs_degd))
        ccf1 = zeros(len_v, size(flux_degd,2))
        bis_inv_slope = zeros(size(flux_degd,2))
        xdata = zeros(size(flux_degd,2))
        ydata = zeros(size(flux_degd,2))

        # get spectrum at specified snr
        GRASS.add_noise!(flux_degd, snr)

        for j in eachindex(nlines_to_do)
            # get number of lines
            n_lines_j = nlines_to_do[j]

            # get lines to include in ccf
            ls = wavs_degd[pks[1:n_lines_j]]
            ds = 1.0 .- flux_degd[pks[1:n_lines_j],1]

            # ls = view(lines, 1:nlines_to_do[i])
            # ds = view(depths, 1:nlines_to_do[i])

            # get view of spectrum
            lidx = 1
            if nlines_to_do[j] >= length(pks)
                ridx = length(wavs_degd)
            else
                ridx = ceil(Int, pks[nlines_to_do[j]] + (pks[nlines_to_do[j] + 1] - pks[nlines_to_do[j]])/2)
            end

            wavs_view = view(wavs_degd, lidx:ridx)
            flux_view = view(flux_degd, lidx:ridx, :)

            # get views of memory
            projection = view(projection_full, 1:length(wavs_view), :)
            proj_flux = view(proj_flux_full, 1:length(wavs_view))

            # calculate ccf
            GRASS.calc_ccf!(v_grid, projection, proj_flux, ccf1,
                            wavs_view, flux_view, ls, ds,
                            resolutions[i], Δv_step=Δv_step,
                            Δv_max=Δv_max)
            rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1)
            rvs_std[i,j] = std(rvs1)

            # get ccf bisector
            vel, int = GRASS.calc_bisector(v_grid, ccf1, nflux=100, top=0.99)

            # smooth the bisector
            vel = GRASS.moving_average(vel, 4)
            int = GRASS.moving_average(int, 4)

            # calc bisector summary statistics
            bis_inv_slope .= GRASS.calc_bisector_inverse_slope(vel, int)
            # bis_span = GRASS.calc_bisector_span(vel, int)
            # bis_slope = GRASS.calc_bisector_slope(vel, int)
            # bis_curve = GRASS.calc_bisector_curvature(vel, int)
            # bis_bot = GRASS.calc_bisector_bottom(vel, int, rvs1)

            # data to fit
            xdata .= bis_inv_slope
            ydata .= rvs1

            # perform the fit
            pfit = Polynomials.fit(xdata, ydata, 1)
            xmodel = range(minimum(xdata), maximum(xdata), length=5)
            ymodel = pfit.(xmodel)

            # decorrelate the velocities
            rvs_to_subtract = pfit.(xdata)
            rvs_std_decorr[i,j] = std(rvs1 .- rvs_to_subtract)

            # plot
            if plot
                # get the slope of the fit to plot
                slope = round(coeffs(pfit)[2], digits=3)
                fit_label = "\$ " .* string(slope) .* "\$"

                fig2, ax2 = plt.subplots()
                ax2.scatter(xdata, ydata, c="k", s=2)
                ax2.plot(xmodel, ymodel, ls="--", c="k", label=L"{\rm Slope } \approx\ " * fit_label, lw=2.5)

                # ax2.set_xlabel(L"{\rm RV\ - \overline{\rm RV}\ } {\rm (m\ s}^{-1}{\rm )}")
                ax2.set_xlabel(L"{\rm RV\ } {\rm (m\ s}^{-1}{\rm )}")
                # ax2.set_ylabel(L"{\rm BIS}\ - \overline{\rm BIS}\ {\rm (m\ s}^{-1}{\rm )}")
                ax2.set_ylabel(L"{\rm BIS\ } {\rm (m\ s}^{-1}{\rm )}")
                ax2.set_title("SNR = " * string(Int(snr)) * ", Nlines = " * string(nlines_to_do[j]))
                ax2.legend()
                ofile = string(joinpath(plotsubdir, "rv_vs_bis_snr_" * string(Int(snr)) * "_lines_" * string(nlines_to_do[j]) * ".pdf"))
                fig2.savefig(ofile)
                plt.clf(); plt.close()
            end
        end
    end
    return nothing
end

# read in the data
d = load(datafile)
wavs = d["wavs"]
flux = d["flux"][:, 1:500]
lines = d["lines"]
depths = d["depths"]
templates = d["templates"]

# snrs to loop over
nlines_to_do = [50, 100, 150, 200, 250, 300, 400, 500]
snrs_for_lines = [200.0, 300.0, 400.0, 500.0, 750.0, 1000.0]
resolutions = [0.98e5, 1.2e5, 1.37e5, 2.7e5, 5.0e5]

# allocate memory for output
rvs_std_out = zeros(length(resolutions), length(nlines_to_do), length(snrs_for_lines))
rvs_std_decorr_out = zeros(length(resolutions), length(nlines_to_do), length(snrs_for_lines))

# loop over snrs
@threads for i in eachindex(snrs_for_lines)
    @show i

    # get views of output arrays
    v1 = view(rvs_std_out, :, :, i)
    v2 = view(rvs_std_decorr_out, :, :, i)

    # get the stuff
    std_vs_number_of_lines(wavs, flux, resolutions, nlines_to_do, v1, v2, snrs_for_lines[i])
end

# write the data
jldsave(string(joinpath(data, template * "_rvs_std_out.jld2")),
        nlines_to_do=nlines_to_do,
        snrs_for_lines=snrs_for_lines,
        resolutions=resolutions,
        rvs_std_out=rvs_std_out,
        rvs_std_decorr_out=rvs_std_decorr_out)
