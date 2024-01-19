using CUDA
using FFTW
using JLD2
using GRASS
using FileIO
using Printf
using Revise
using LsqFit
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

# get the name of template from the command line args
template = "FeI_5434"

# get command line args and output directories
include("paths.jl")
datafile = string(abspath(joinpath(data, template * "_picket_fence.jld2")))

# set output directory
outdir = "/storage/work/mlp95/grass_output/plots/rms_stuff/"

# get lines to construct templates
lp = GRASS.LineProperties()
names = GRASS.get_name(lp)

@. harvey(x, p) = 4 * p[1] * p[2] / (1 + (2π * x * p[2])^2)

# for i in eachindex(lp.λrest)
# let i = 1
i = 9
    println("\t>>> Doing " * names[i])

    # set up stuff for synthesis
    Nt = 32000
    lines = [5576.0881]
    templates = [lp.file[i]]
    depths = [lp.depth[i]]
    variability = trues(length(templates))
    resolution = 7e5
    disk = DiskParams(Nt=Nt)
    spec = SpecParams(lines=lines, depths=depths, variability=variability,
                      templates=templates, resolution=resolution, contiguous_only=true)

    # synthesize spectra
    lambdas1, outspec1 = synthesize_spectra(spec, disk, seed_rng=false, verbose=true, use_gpu=true)

    # compute velocities
    println("\t>>> Calculating CCFs...")
    v_grid, ccf = calc_ccf(lambdas1, outspec1, lines, depths, 7e5)
    println("\t>>> Calculating RVs...")
    rvs, sigs = calc_rvs_from_ccf(v_grid, ccf)

    # get power spectrum of velocities
    sampling_rate = 1.0/15.0
    F = fftshift(fft(rvs))
    freqs = fftshift(fftfreq(length(rvs), sampling_rate))

    # throw away samplings of frequencies less than 0.0s
    idx = findall(freqs .> 0.0)
    F = F[idx]
    freqs = freqs[idx]

    # bin the frequencies
    minfreq = log10(minimum(freqs))
    maxfreq = log10(maximum(freqs))
    freqstep = 1e-1
    freqs_binned = collect(10.0 .^ (range(minfreq, maxfreq, step=freqstep)))
    freqs_bincen = (freqs_binned[2:end] .+ freqs_binned[1:end-1])./2
    F_binned = zeros(length(freqs_binned)-1)
    for i in eachindex(F_binned)
        j = findall(x -> (x .>= freqs_binned[i]) .& (x .<= freqs_binned[i+1]), freqs)
        F_binned[i] = mean(abs2.(F[j]))
    end

    # throw away nans
    idx = (.!isnan.(F_binned)) .& (freqs_bincen .>= 1e-4)

    freqs_bincen_fit = copy(freqs_bincen[idx])
    F_binned_fit = copy(F_binned[idx])

    # F_binned_fit[.!idx] .= 1e3^2.0

    # fit the harvey function
    p0 = [1e3, 400.0]
    cfit = curve_fit(harvey, freqs_bincen_fit, sqrt.(F_binned_fit), p0)

    # get title
    title = replace(names[i], "_" => "\\ ")
    idx = findfirst('I', title)
    title = title[1:idx-1] * "\\ " * title[idx:end] * "\\ \\AA"

    xdata = collect(10.0 .^ (range(minfreq, 0.001, step=freqstep)))

    # plot it
    fig, ax1 = plt.subplots()
    ax1.scatter(freqs, sqrt.(abs2.(F)), s=1, c="k")
    ax1.scatter(freqs_bincen, sqrt.(F_binned), marker="x", c="red")
    ax1.plot(xdata, harvey(xdata, cfit.param))
    ax1.axvline(1.0 / (60.0 * 20.0), ls=":", c="k", alpha=0.75, label=L"{\rm 20\ min.}")
    ax1.axvline(1.0 / (15.0), ls="--", c="k", alpha=0.75, label=L"{\rm 15\ sec.}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()
    # ax1.set_title(("\${\\rm " * title * "}\$"))
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel(L"{\rm Power\ [(m\ s}^{-1}{\rm )}^2 {\rm \ Hz}^{-1}{\rm ]}")
    # ax1.set_xlim(1e-8, maximum(freqs))
    ax1.set_ylim(1, 1e5)
    # fig.savefig(joinpath(outdir, names[i] * "power_spec.pdf"))
    plt.show()
    plt.clf(); plt.close()


    # running mean on different timescales
    # in multiples of 15sec: 20*15sec = 5 min,
    scales = range(4, 3000, step=2)
    rms_smooth = zeros(length(scales))
    rms_resids = zeros(length(scales))
    for i in eachindex(scales)
        # smooth the time series
        rvs_smooth_1 = GRASS.moving_average(rvs, scales[i])

        # get smoothed rms and resid rms
        rms_smooth[i] = calc_rms(rvs_smooth_1)
        rms_resids[i] = calc_rms(rvs .- rvs_smooth_1)
    end

    # convert scale to minutes
    scales_min = scales .* 15.0 ./ 60.0

    # make plot stuff
    fig, ax1 = plt.subplots()
    ax1.axhline(0.1, ls="--", c="k", alpha=0.75)
    ax1.axvline(8.0 * 60.0, ls="--", c="k", alpha=0.75)
    ax1.plot(scales_min, rms_smooth, c="red", label=L"{\rm SNR} = \infty")
    ax1.plot(scales_min, rms_resids, c="green")
    ax1.set_xscale("log")
    ax1.set_xlabel(L"{\rm Smoothing\ timescale\ [min.]}")
    ax1.set_ylabel(L"{\rm RMS\ RV\ [m\ s}^{-1}{\rm ]}")
    ax1.set_xlim(1.0, 7e3)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title(("\${\\rm " * title * "}\$"))
    ax1.legend()
    fig.savefig(joinpath(outdir, names[i] * "rms_smoothing.pdf"))
    plt.show()
    plt.clf(); plt.close()
# end


#=# loop over times
ts = collect(1:Nt)
rms_i = zeros(length(ts))
avg_i = zeros(length(ts))
resid = zeros(length(ts))
for i in ts
    i == 1 && continue
    # smooth the time series

    avg_i[i] = mean(view(rvs, 1:i))
    rms_i[i] = calc_rms(view(rvs, 1:i) .- avg_i[i])
    # resid[i] = view(rvs, 1:i) .- avg_i[i]
end

# # make plot stuff
# fig1, ax1 = plt.subplots()
# ax1.plot(ts, rms_i)=#
