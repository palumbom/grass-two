using CUDA
using FFTW
using JLD2
using GRASS
using FileIO
using Printf
using Revise
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

# set output directory
outdir = "/storage/work/m/mlp95/grass_output/plots/rms_stuff/"

# get lines to construct templates
lp = GRASS.LineProperties()
names = GRASS.get_name(lp)

# # set up stuff for lines
# N = 132
# Nt = 32000
# lines = collect(range(5400.0, 5500.0, length=length(lp.λrest)))
# templates = lp.file
# depths = lp.depth

for i in eachindex(lp.λrest)
    println("\t>>> Doing " * names[i])

    # set up stuff for synthesis
    N = 132
    Nt = 32000
    lines = [5576.0881]
    templates = [lp.file[i]]
    depths = [lp.depth[i]]
    variability = trues(length(templates))
    resolution = 7e5
    disk = DiskParams(N=N, Nt=Nt)
    spec = SpecParams(lines=lines, depths=depths, variability=variability,
                      templates=templates, resolution=resolution)

    # synthesize spectra
    lambdas1, outspec1 = synthesize_spectra(spec, disk, seed_rng=true, verbose=true, use_gpu=true)
    outspec2 = GRASS.add_noise(outspec1, 500.0)

    # compute velocities
    println("\t>>> Calculating CCFs...")
    v_grid, ccf = calc_ccf(lambdas1, outspec1, spec)
    println("\t>>> Calculating RVs...")
    rvs, sigs = calc_rvs_from_ccf(v_grid, ccf)

    println("\t>>> Calculating CCFs...again...")
    v_grid2, ccf2 = calc_ccf(lambdas1, outspec2, spec)
    println("\t>>> Calculating RVs...again...")
    rvs2, sigs2 = calc_rvs_from_ccf(v_grid2, ccf2)

    # save the spectra and rvs
    #=datafile = "rms_time.jld2"
    jldsave(datafile, lambdas1=lambdas1, outspec1=outspec1, outspec2=outspec2,
            rvs=rvs, sigs=sigs, rvs2=rvs2, sigs2=sigs2)=#

    # load the data file
    # datafile = "rms_time.jld2"
    # d = load(datafile)
    # lambdas1 = d["lambdas1"]
    # outspec1 = d["outspec1"]
    # outspec2 = d["outspec2"]
    # rvs = d["rvs"]
    # sigs = d["sigs"]
    # rvs2 = d["rvs2"]
    # sigs2 = d["sigs2"]

    # rvs = rvs[1:16000]
    # rvs2 = rvs2[1:16000]

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

    # get title
    title = replace(names[i], "_" => "\\ ")
    idx = findfirst('I', title)
    title = title[1:idx-1] * "\\ " * title[idx:end] * "\\ \\AA"

    # plot it
    fig, ax1 = plt.subplots()
    ax1.scatter(freqs, abs2.(F), s=1, c="k")
    ax1.scatter(freqs_bincen, F_binned, marker="x", c="red")
    ax1.axvline(1.0 / (60.0 * 20.0), ls=":", c="k", alpha=0.75, label=L"{\rm 20\ min.}")
    ax1.axvline(1.0 / (15.0), ls="--", c="k", alpha=0.75, label=L"{\rm 15\ sec.}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_title(("\${\\rm " * title * "}\$"))
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel(L"{\rm Power\ [(m\ s}^{-1}{\rm )}^2 {\rm \ Hz}^{-1}{\rm ]}")
    # ax1.set_xlim(1e-8, maximum(freqs))
    # ax1.set_ylim(1, 1e5)
    fig.savefig(joinpath(outdir, names[i] * "power_spec.pdf"))
    # plt.show()
    plt.clf(); plt.close()


    # running mean on different timescales
    # in multiples of 15sec: 20*15sec = 5 min,
    scales = range(4, 4000, step=2)
    rms_smooth = zeros(length(scales))
    rms_resids = zeros(length(scales))
    rms_smooth2 = zeros(length(scales))
    rms_resids2 = zeros(length(scales))
    for i in eachindex(scales)
        # smooth the time series
        rvs_smooth_1 = GRASS.moving_average(rvs, scales[i])
        rvs_smooth_2 = GRASS.moving_average(rvs2, scales[i])

        # get smoothed rms and resid rms
        rms_smooth[i] = calc_rms(rvs_smooth_1)
        rms_resids[i] = calc_rms(rvs .- rvs_smooth_1)
        rms_smooth2[i] = calc_rms(rvs_smooth_2)
        rms_resids2[i] = calc_rms(rvs2 .- rvs_smooth_2)
    end

    # convert scale to minutes
    scales_min = scales .* 15.0 ./ 60.0

    # make plot stuff
    fig, ax1 = plt.subplots()
    ax1.axhline(0.1, ls="--", c="k", alpha=0.75)
    ax1.axvline(8.0 * 60.0, ls="--", c="k", alpha=0.75)
    ax1.plot(scales_min, rms_smooth, c="red", label=L"{\rm SNR} = \infty")
    ax1.plot(scales_min, rms_resids, c="green")
    ax1.plot(scales_min, rms_smooth2, ls=":", c="red", label=L"{\rm SNR} = 500")
    ax1.plot(scales_min, rms_resids2, ls=":", c="green")
    ax1.set_xscale("log")
    ax1.set_xlabel(L"{\rm Smoothing\ timescale\ [min.]}")
    ax1.set_ylabel(L"{\rm RMS\ RV\ [m\ s}^{-1}{\rm ]}")
    ax1.set_xlim(1.0, 7e3)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title(("\${\\rm " * title * "}\$"))
    ax1.legend()
    fig.savefig(joinpath(outdir, names[i] * "rms_smoothing.pdf"))
    # plt.show()
    plt.clf(); plt.close()
end


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
