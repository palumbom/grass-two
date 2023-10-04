# get stuff precompiled on the main process
using Distributed
Pkg.instantiate()
Pkg.precompile()
sleep(5)

# add processes
if isone(nprocs())
    addprocs(7)
end

# get libraries load everywhere
@everywhere begin
    using Pkg; Pkg.activate(".")
    using JLD2
    using GRASS
    using Printf
    using FileIO
    using Profile
    using Statistics
    using EchelleCCFs
    using Polynomials
    using SharedArrays
    using BenchmarkTools
    # using AbstractGPs, TemporalGPs, KernelFunctions
    # using RvSpectML

    # plotting
    using LaTeXStrings
    import PyPlot; plt = PyPlot; mpl = plt.matplotlib; #plt.ioff()
    mpl.style.use(GRASS.moddir * "fig.mplstyle")
    colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]
end

# get the name of template from the command line args
@everywhere begin
    template_idx = tryparse(Int, ARGS[1])
    lp = GRASS.LineProperties(exclude=["CI_5380", "NaI_5896"])
    line_names = GRASS.get_name(lp)
    template = line_names[template_idx]
end

# get command line args and output directories
@everywhere begin
    include(joinpath(abspath(@__DIR__), "paths.jl"))
    plotsubdir = string(joinpath(figures, "snr_plots"))
    datafile = string(abspath(joinpath(data, template * "_picket_fence.jld2")))
end

if !isdir(plotsubdir)
    mkdir(plotsubdir)
end

# read in the data
@everywhere begin
    d = load(datafile)
    wavs = d["wavs"]
    flux = d["flux"]
    lines = d["lines"]
    depths = d["depths"]
    templates = d["templates"]
end

@everywhere begin
    # get separation between lines
    line_sep = mean(diff(sort(lines)))

    function get_lines(wavs, flux, last_line_number)
        if last_line_number == length(lines)
            return view(wavs, :), view(flux, :, :)
        else
            idx = findfirst(x -> x .>= lines[last_line_number] + 0.5 * line_sep, wavs)
            return view(wavs, 1:idx), view(flux, 1:idx, :)
        end
    end

    function get_one_line(wavs, flux, line_number; t=1)
        idx1 = findfirst(x -> x .>= lines[line_number] - 0.5 * line_sep, wavs)
        idx2 = findfirst(x -> x .>= lines[line_number] + 0.5 * line_sep, wavs)
        return view(wavs, idx1:idx2), view(flux, idx1:idx2, t)
    end
end

@everywhere begin
    function std_vs_number_of_lines(nlines_to_do::AbstractArray{Int,1},
                                    resolutions::AbstractArray{T,1},
                                    rvs_std::AbstractArray{T,2},
                                    rvs_std_decorr::AbstractArray{T,2}, snr::T;
                                    plot::Bool=false, oversampling::T=4.0) where T<:Float64
        # include more and more lines in ccf
        for i in eachindex(nlines_to_do)
            # get lines to include in ccf
            ls = view(lines, 1:nlines_to_do[i])
            ds = view(depths, 1:nlines_to_do[i])

            # get view of spectrum
            wavs_temp, flux_temp = get_lines(wavs, flux, nlines_to_do[i])

            for j in eachindex(resolutions)
                # degrade the resolution
                if resolutions[j] == 7e5
                    wavs_degd = copy(wavs_temp)
                    flux_degd = copy(flux_temp)
                else
                    # do an initial conv to get output size
                    wavs_to_deg = wavs_temp[:, 1]
                    flux_to_deg = flux_temp[:, 1]
                    wavs_size, flux_size = GRASS.convolve_gauss(wavs_to_deg,
                                                                flux_to_deg,
                                                                new_res=resolutions[j],
                                                                oversampling=oversampling)

                    # allocate memory
                    wavs_degd = wavs_size
                    flux_degd = zeros(size(wavs_size, 1), size(flux_temp, 2))

                    # loop over epochs and convolve
                    for k in 1:size(flux_temp, 2)
                        flux_to_deg = view(flux_temp, :, k)
                        wavs_out, flux_out = GRASS.convolve_gauss(wavs_to_deg,
                                                                  flux_to_deg,
                                                                  new_res=resolutions[j],
                                                                  oversampling=oversampling)

                        # copy to array
                        flux_degd[:, k] .= flux_out
                    end
                end

                # get spectrum at specified snr
                wavs_snr = wavs_degd
                flux_snr = copy(flux_degd)
                GRASS.add_noise!(flux_snr, snr)

                # calculate ccf
                v_grid, ccf1 = calc_ccf(wavs_snr, flux_snr, ls, ds, resolutions[j])
                rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1)
                rvs_std[i,j] = std(rvs1)

                # get ccf bisector
                vel, int = GRASS.calc_bisector(v_grid, ccf1, nflux=100, top=0.99)

                # smooth the bisector
                vel = GRASS.moving_average(vel, 4)
                int = GRASS.moving_average(int, 4)

                # calc bisector summary statistics
                bis_inv_slope = GRASS.calc_bisector_inverse_slope(vel, int)
                bis_span = GRASS.calc_bisector_span(vel, int)
                bis_slope = GRASS.calc_bisector_slope(vel, int)
                bis_curve = GRASS.calc_bisector_curvature(vel, int)
                bis_bot = GRASS.calc_bisector_bottom(vel, int, rvs1)

                # data to fit
                xdata = bis_inv_slope
                ydata = rvs1

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
                    ax2.set_title("SNR = " * string(Int(snr)) * ", Nlines = " * string(nlines_to_do[i]))
                    ax2.legend()
                    ofile = string(joinpath(plotsubdir, "rv_vs_bis_snr_" * string(Int(snr)) * "_lines_" * string(nlines_to_do[i]) * ".pdf"))
                    fig2.savefig(ofile)
                    plt.clf(); plt.close()
                end
            end
        end
        return nothing
    end
end

# snrs to loop over
@everywhere begin
    nlines_to_do = [5, 10, 25, 50, 100, 125, 150, 200, 250]
    snrs_for_lines = [100.0, 150.0, 200.0, 250.0, 500.0, 750.0, 1000.0]
    resolutions = [0.98e5, 1.2e5, 1.37e5, 2.7e5, 7e5]
end

# allocate memory for output
rvs_std_out = SharedArray(zeros(length(nlines_to_do), length(resolutions), length(snrs_for_lines)))
rvs_std_decorr_out = SharedArray(zeros(length(nlines_to_do), length(resolutions), length(snrs_for_lines)))

@sync @distributed for i in eachindex(snrs_for_lines)
    @show i

    # get views of output arrays
    v1 = view(rvs_std_out, :, :, i)
    v2 = view(rvs_std_decorr_out, :, :, i)

    # get the stuff
    std_vs_number_of_lines(nlines_to_do, resolutions, v1, v2, snrs_for_lines[i])
end

# write the data
jldsave(string(joinpath(data, template * "_rvs_std_out.jld2")),
        snrs_for_lines=snrs_for_lines,
        rvs_std_out=Array(rvs_std_out),
        rvs_std_decorr_out=Array(rvs_std_decorr_out))

# d = load(string(joinpath(data, "rvs_std_out.jld2")))
# snrs_for_lines = d["snrs_for_lines"]
# rvs_std_out = d["rvs_std_out"]
# rvs_std_decorr_out = d["rvs_std_decorr_out"]

# # set up colors for plots
# pcolors = plt.cm.rainbow(range(0, 1, length=length(snrs_for_lines)))

# # set up plots
# fig1, ax1 = plt.subplots(figsize=(7.2,4.8))

# # plot snr vs number of lines
# for i in eachindex(snrs_for_lines)
#     ax1.plot(nlines_to_do, rvs_std_out[:,i], label="SNR = " * string(snrs_for_lines[i]), c=pcolors[i,:])
# end

# # set axis stuff
# ax1.set_xlabel(L"{\rm Number\ of\ lines\ in\ CCF}")
# ax1.set_ylabel(L"{\rm RV\ RMS\ (m\ s}^{-1} {\rm )}")
# ax1.set_xscale("log", base=2)
# ax1.set_yscale("log", base=2)
# ax1.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
# fig1.savefig(string(joinpath(plotsubdir, template * "_std_vs_number_of_lines.pdf")))
# plt.clf(); plt.close("all")

# # set up plots
# fig1, ax1 = plt.subplots(figsize=(7.2,4.8))

# # plot snr vs number of lines
# for i in eachindex(snrs_for_lines)
#     dat = (rvs_std_out[:,i] .- rvs_std_decorr_out[:,i]) ./ rvs_std_out[:,i]
#     ax1.plot(nlines_to_do, dat .* 100, label="SNR = " * string(snrs_for_lines[i]), c=pcolors[i,:])
# end

# # set axis stuff
# ax1.set_xlabel(L"{\rm Number\ of\ lines\ in\ CCF}")
# ax1.set_ylabel(L"{\rm \% \ Improvement\ in\ RV\ RMS}")
# ax1.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
# fig1.savefig(string(joinpath(plotsubdir, template * "_improvement_vs_number_of_lines_same.pdf")))
# plt.clf(); plt.close("all")


# # TODO do they not match or is this numerical noise????
# for i in 1:3
#     # get just one line
#     if i == 2
#         wavs_temp, flux_temp = get_one_line(wavs, flux, i, t=12)
#     else
#         wavs_temp, flux_temp = get_one_line(wavs, flux, i, t=1)
#     end

#     # ccf
#     ls = [lines[i]]
#     ds = [maximum(flux_temp[:,1]) .- minimum(flux_temp[:,1])]
#     v_grid, ccf1 = calc_ccf(wavs_temp, flux_temp, ls, ds, 7e5, Δv_step=100.0)

#     # get bisector
#     vel, int = GRASS.calc_bisector(v_grid, ccf1, nflux=100, top=0.95)

#     vel = GRASS.moving_average(vel, 4)[3:end-1]
#     int = GRASS.moving_average(int, 4)[3:end-1]

#     plt.plot(vel[3:end-1], int[3:end-1], label=string(i))
# end
# plt.legend()
# plt.show()
