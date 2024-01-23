# environment + packages
using CSV
using CUDA
using GRASS
using Peaks
using Optim
using LsqFit
using Statistics
using DataFrames
using EchelleCCFs
using EchelleCCFs: λ_air_to_vac, calc_doppler_factor, MeasureRvFromCCFQuadratic as QuadraticFit

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
plotdir = string(abspath(joinpath(figures, "iag_comparison")))
datadir = string(abspath(data))

if !isdir(plotdir)
    mkdir(plotdir)
end

# decide whether to use gpu
use_gpu = true
@assert CUDA.functional()

# get data
lp = GRASS.LineProperties(exclude=["CI_5380", "NaI_5896"])
files = GRASS.get_file(lp)
line_names = GRASS.get_name(lp)

# read in optimized depths
df = CSV.read(joinpath(datadir, "optimized_depth.csv"), DataFrame)

# wavelength of line to synthesize/compare to iag
for (i, file) in enumerate(files)
    if !contains(file, "5383")
        continue
    end
    println(">>> Running " * line_names[i] * "...")

    # i = 9
    # file = line_names[i]

    # get properties from line
    line_name = line_names[i]
    airwav = lp.λrest[i]
    depth = lp.depth[i]

    # get IAG spectrum and normalize it
    wavs_iag0, flux_iag0 = GRASS.read_iag_atlas(isolate=true, airwav=airwav, buffer=1.5)
    flux_iag0 ./= maximum(flux_iag0)

    # convolve IAG spectrum to LARS resolution
    wavs_iag, flux_iag = GRASS.convolve_gauss(wavs_iag0, flux_iag0, new_res=1e6, oversampling=4.0)

   # get depth from IAG spectrum
    buff = 0.12575
    if contains("FeI_5383", line_name)
        buff = 0.3
    elseif contains("FeI_5434", line_name)
        buff = 0.3
    elseif contains("FeI_5382", line_name)
        buff = 0.2
    elseif contains("FeI_5576", line_name)
        buff = 0.25
    elseif contains("CaI_6169.0", line_name)
        buff = 0.25
    elseif contains("CaI_6169.5", line_name)
        buff = 0.1475
    elseif contains("FeI_6170", line_name)
        buff = 0.175
    elseif contains("FeI_6301", line_name)
        buff = 0.25
    elseif contains("FeI_6302", line_name)
        buff = 0.15
    end

    idxl = findfirst(x -> x .>= airwav - buff, wavs_iag)
    idxr = findfirst(x -> x .>= airwav + buff, wavs_iag)
    iag_bot = minimum(view(flux_iag, idxl:idxr))
    iag_depth = 1.0 - iag_bot

    # get the depth for the simulation
    sim_depth = 0.8439837292977859
    # sim_depth = df[i, "optimized_depth"]

    # simulate the spectrum
    lines = [airwav]
    depths = [sim_depth]
    templates = [file]
    resolution = 7e5
    spec = SpecParams(lines=lines, depths=depths, templates=templates,
                      resolution=resolution, buffer=1.5, oversampling=2.0)
    disk = DiskParams(Nt=100)

    # simulate the spectrum
    wavs_sim, flux_sim = synthesize_spectra(spec, disk, use_gpu=use_gpu,
                                            verbose=false, seed_rng=true)
    flux_sim = dropdims(mean(flux_sim, dims=2), dims=2)

    # interpolate iag on synth wavelength grid
    itp = GRASS.linear_interp(wavs_iag, flux_iag, bc=NaN)
    flux_iag = itp.(wavs_sim)
    wavs_iag = copy(wavs_sim)

    # get width in velocity for CCF
    idxl_sim, idxr_sim = GRASS.find_wing_index(0.95, flux_sim)

    # get width in angstroms
    width_ang = wavs_sim[idxr_sim] - wavs_sim[idxl_sim]

    # convert to velocity
    width_vel = GRASS.c_ms * width_ang / wavs_sim[argmin(flux_sim)]
    Δv_max = round((width_vel + 1e3)/100) * 100

    # calculate ccfs for spectra
    v_grid_iag, ccf_iag = GRASS.calc_ccf(wavs_iag, flux_iag, lines, depths,
                                         7e5, Δv_step=100.0, Δv_max=Δv_max,
                                         mask_type=EchelleCCFs.GaussianCCFMask)

    v_grid_sim, ccf_sim = GRASS.calc_ccf(wavs_sim, flux_sim, lines, depths,
                                         7e5, Δv_step=100.0, Δv_max=Δv_max,
                                         mask_type=EchelleCCFs.GaussianCCFMask)

    # plt.plot(wavs_iag, flux_iag)
    # plt.plot(wavs_sim, flux_sim)
    # plt.plot(v_grid_iag, ccf_iag)
    # plt.plot(v_grid_sim, ccf_sim)
    # plt.show()
    # break

    # get bisectors
    vel_iag, int_iag = GRASS.calc_bisector(v_grid_iag, ccf_iag, nflux=50, top=0.9)
    vel_sim, int_sim = GRASS.calc_bisector(v_grid_sim, ccf_sim, nflux=50, top=0.9)

    # plt.plot(vel_iag, int_iag)
    # plt.plot(vel_sim, int_sim)
    # plt.show()

    # compute velocity as mean bisector between N and M % depth
    N = 0.20
    M = 0.70
    idx1 = findfirst(x -> x .>= N * iag_depth + iag_bot, int_iag)
    idx2 = findfirst(x -> x .>= M * iag_depth + iag_bot, int_iag)
    if isnothing(idx2)
        idx2 = findfirst(x -> x .>= 0.9, int_iag)
    end
    rv_iag = mean(view(vel_iag, idx1:idx2))

    idx1 = findfirst(x -> x .>= N * sim_depth + minimum(flux_sim), int_sim)
    idx2 = findfirst(x -> x .>= M * sim_depth + minimum(flux_sim), int_sim)
    if isnothing(idx2)
        idx2 = findfirst(x -> x .>= 0.9, int_sim)
    end
    rv_sim = mean(view(vel_sim, idx1:idx2))

    # transform to lab frame
    vel_iag .-= rv_iag
    vel_sim .-= rv_sim
    wavs_iag ./= calc_doppler_factor(rv_iag)
    wavs_sim ./= calc_doppler_factor(rv_sim)

    # interpolate IAG onto synthetic wavelength grid
    itp = GRASS.linear_interp(wavs_iag, flux_iag)
    flux_iag = itp.(wavs_sim)
    wavs_iag = copy(wavs_sim)

    # re-compute line isolation indices because of interpolation
    idxl = findfirst(x -> x .>= airwav - buff, wavs_iag)
    idxr = findfirst(x -> x .>= airwav + buff, wavs_iag)

    # recompute bisectors b/c of interpolation
    v_grid_iag, ccf_iag = GRASS.calc_ccf(wavs_iag, flux_iag, lines, depths,
                                         7e5, Δv_step=100.0, Δv_max=Δv_max,
                                         mask_type=EchelleCCFs.GaussianCCFMask)

    v_grid_sim, ccf_sim = GRASS.calc_ccf(wavs_sim, flux_sim, lines, depths,
                                         7e5, Δv_step=100.0, Δv_max=Δv_max,
                                         mask_type=EchelleCCFs.GaussianCCFMask)

    # set errant ccf values
    idx0 = iszero.(ccf_iag)
    idxnz = findfirst(x -> x .> 0.0, ccf_iag)
    ccf_iag[idx0] .= ccf_iag[idxnz]

    # get bisectors
    vel_iag, int_iag = GRASS.calc_bisector(v_grid_iag, ccf_iag, nflux=50, top=0.9)
    vel_sim, int_sim = GRASS.calc_bisector(v_grid_sim, ccf_sim, nflux=50, top=0.9)

    # find mean velocities in order to align bisectors
    N = 0.10
    M = 0.70
    idx1 = findfirst(x -> x .>= N * sim_depth + minimum(flux_sim), int_sim)
    idx2 = findfirst(x -> x .>= M * sim_depth + minimum(flux_sim), int_sim)
    if isnothing(idx2)
        idx2 = findfirst(x -> x .>= 0.9, int_sim)
    end
    rv_sim = mean(view(vel_sim, idx1:idx2))

    idx1 = findfirst(x -> x .>= N * iag_depth + iag_bot, int_iag)
    idx2 = findfirst(x -> x .>= M * iag_depth + iag_bot, int_iag)
    if isnothing(idx2)
        idx2 = findfirst(x -> x .>= 0.9, int_iag)
    end
    rv_iag = mean(view(vel_iag, idx1:idx2))

    # align the bisectors
    vel_sim .-= rv_sim
    vel_iag .-= rv_iag

    # big function for plotting
    function comparison_plots()
        # make plot objects
        fig = plt.figure(figsize=(6.4,4.8))
        gs = mpl.gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], figure=fig, hspace=0.05)
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])

        # plot spectra
        ax1.plot(wavs_sim, flux_sim, marker="o", c="black", ms=3.0, lw=1.5, markevery=2, label=L"{\rm Synthetic}")
        ax1.plot(wavs_iag, flux_iag, marker="s", c=colors[1], ms=2.0, lw=1.0, markevery=2, label=L"{\rm IAG}")
        # ax1.plot(wavs_iag, flux_mod, alpha=0.9, marker="o", c=colors[2], ms=2.0, lw=1.0, markevery=2, label=L"{\rm Cleaned\ IAG}")

        # plot resids
        ax2.plot(wavs_sim, flux_iag .- flux_sim, c=colors[1], marker="s", ms=2.0, lw=0, markevery=2)
        # ax2.plot(wavs_sim, flux_mod .- flux_sim, c=colors[2], marker="^", ms=2.0, lw=0, markevery=2)

        # find limits
        idx_min = argmin(flux_sim)
        idx1 = idx_min - findfirst(x -> x .> 0.95, flux_sim[idx_min:-1:1])
        idx2 = idx_min + findfirst(x -> x .> 0.95, flux_sim[idx_min:end])

        # set limits
        min_idx = argmin(flux_iag)
        ax1.set_xlim(wavs_sim[idx1-50], wavs_sim[idx2+50])
        ax1.set_ylim(minimum(flux_sim) - 0.1, 1.1)
        ax2.set_xlim(wavs_sim[idx1-50], wavs_sim[idx2+50])
        ax2.set_ylim(-0.0575, 0.0575)

        # set tick labels, axis labels, etc.
        ax1.set_xticklabels([])
        ax1.set_ylabel(L"{\rm Normalized\ Flux}", labelpad=15)
        ax2.set_xlabel(L"{\rm Wavelength\ (\AA)}")
        ax2.set_ylabel(L"{\rm IAG\ -\ GRASS}")
        ax1.legend()
        fig.tight_layout()

        # set the title
        title = replace(line_name, "_" => "\\ ")
        idx = findfirst('I', title)
        title = title[1:idx-1] * "\\ " * title[idx:end] * "\\ \\AA"
        ax1.set_title(("\${\\rm " * title * "}\$"))

        # save the plot
        fig.subplots_adjust(wspace=0.05)
        fig.savefig(joinpath(plotdir, line_name * "_line.pdf"))
        plt.clf(); plt.close()

        # plot the bisectors
        fig = plt.figure(figsize=(6.4,4.8))
        gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1.1], figure=fig, wspace=0.05)
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])

        # plot bisectors
        ax1.plot(vel_sim[4:end], int_sim[4:end], marker="o", color="black", ms=3.0, lw=2.0, markevery=1, label=L"{\rm Synthetic}")
        ax1.plot(vel_iag[4:end], int_iag[4:end], marker="s", c=colors[1], ms=2.0, lw=1.0, markevery=1, label=L"{\rm IAG}")
        # ax1.plot(vel_sim2, int_sim2, marker="o", color="black", ms=3.0, lw=2.0, markevery=1, label=L"{\rm Derp}")
        # ax1.plot(vel_mod[2:end], int_mod[2:end], marker="^", c=colors[2], ms=2.0, lw=1.0, markevery=1, label=L"{\rm Cleaned\ IAG}")

        # plot residuals
        ax2.plot(vel_iag[4:end] .- vel_sim[4:end], int_iag[4:end], c=colors[1], marker="s", ms=2.0, lw=0.0, markevery=1)
        # ax2.plot(vel_mod[2:end] .- vel_sim[2:end], int_mod[2:end], c=colors[2], marker="o", ms=2.0, lw=0.0, markevery=2)

        # set tick labels, axis labels, etc.
        ax2.set_yticklabels([])
        ax2.yaxis.tick_right()
        ax1.set_ylim(minimum(flux_sim) - 0.05, 1.05)
        ax2.set_xlim(-35, 35)
        ax2.set_ylim(minimum(flux_sim) - 0.05, 1.05)
        ax1.set_xlabel(L"{\rm Relative\ Velocity\ (m\ s^{-1})}", fontsize=13)
        ax1.set_ylabel(L"{\rm Normalized\ Intensity}")
        ax2.set_xlabel(L"{\rm IAG\ -\ GRASS\ (m\ s^{-1})}", fontsize=13)
        ax1.legend(labelspacing=0.25)
        # ax2.legend(loc="upper right", prop=Dict("size"=>12.5), labelspacing=0.25)

        # set the title
        fig.suptitle(("\${\\rm " * title * "}\$"), y=0.95)

        # save the plot
        fig.subplots_adjust(hspace=0.05)
        fig.savefig(joinpath(plotdir, line_name * "_bisector.pdf"))
        plt.clf(); plt.close()
        return nothing
    end
    comparison_plots()
end
