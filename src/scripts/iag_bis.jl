# environment + packages
using Pkg; Pkg.activate(".")
using CSV
using CUDA
using GRASS
using Peaks
using LsqFit
using Statistics
using DataFrames
using EchelleCCFs: λ_air_to_vac, calc_doppler_factor, MeasureRvFromCCFQuadratic as QuadraticFit

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# get command line args and output directories
run, plot = parse_args(ARGS)
grassdir, plotdir, datadir = check_plot_dirs()

outdir = plotdir * "iag_comparison/"
if !isdir(outdir)
    mkdir(outdir)
end

# decide whether to use gpu
use_gpu = CUDA.functional()

# model the iag blends
function model_iag_blends(wavs_sim::AbstractArray{T,1}, flux_sim::AbstractArray{T,1},
                          wavs_iag::AbstractArray{T,1}, flux_iag::AbstractArray{T,1};
                          plot=false) where T<:Float64
    # calculate the residuals
    resids = flux_iag .- flux_sim

    # models for fit
    @. gaussian(x, a, b, c) = a * exp(-(x - b)^2/(2 * c^2)) + 1
    function tel_model(x, p)
        n = length(p) ÷ 3
        out = ones(length(x))
        for i in 1:n
            out .*= gaussian(x, p[3i-2:3i]...)
        end
        return out .* flux_sim
    end

    # identify peaks in the residuals
    max_resid = 0.01
    minprom = 0.0005

    # guess the number of fits needed from minima
    m_inds = Peaks.argmaxima(.-resids, 10, strict=false)
    m_inds, m_proms = Peaks.peakproms(m_inds, .-resids, minprom=minprom, strict=false)
    m_inds, m_widths, m_left, m_right = Peaks.peakwidths(m_inds, .-resids, m_proms, strict=false)

    idx = findfirst(x -> isapprox.(x, minimum(wavs_sim), atol=1e-1), wavs_iag[m_inds])
    m_inds = deleteat!(m_inds, [idx])
    m_widths = deleteat!(m_widths, [idx])
    m_left = deleteat!(m_left, [idx])
    m_right = deleteat!(m_right, [idx])

    # convert width from pixels to wavelength
    m_widths .*= (wavs_iag[2] - wavs_iag[1])

    # set initial guess parameters
    nfits = length(m_inds)
    pgrid = zeros(nfits, 3)
    p0 = Array{Float64,1}[]
    for i in 1:nfits
        thresh = abs(wavs_sim[argmin(flux_sim)] - wavs_iag[m_inds[i]])
        if thresh <= 0.1
            continue
        end
        pgrid[i,1] = -m_proms[i]
        pgrid[i,2] = wavs_iag[m_inds[i]]
        pgrid[i,3] = m_widths[i]
    end
    p0 = Array{Float64,1}(vcat(p0, pgrid'...))

    # do the fit
    fit = curve_fit(tel_model, wavs_iag, flux_iag, p0)
    resids .= flux_iag./(tel_model(wavs_iag, fit.param)./flux_sim) .- flux_sim

    # plot diagnostics
    if plot
        fig = plt.figure(figsize=(8,6))
        gs = mpl.gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], figure=fig, hspace=0.05)
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])
        ax1.plot(wavs_iag, flux_iag, label="Observed IAG", color="tab:blue")
        ax1.plot(wavs_iag, tel_model(wavs_iag, fit.param), label="Modeled IAG ", ls=":", color="tab:green")
        for i in m_inds
            thresh = abs(wavs_sim[argmin(flux_sim)] - wavs_iag[i])
            if thresh <= 0.1
                continue
            end
            ax1.axvline(wavs_sim[i])
        end
        ax2.scatter(wavs_sim, flux_iag./tel_model(wavs_iag, fit.param), c="k", s=0.5)
        ax1.legend()
        ax1.set_xticklabels([])
        ax1.set_ylabel(L"{\rm Normalized\ Intensity}")
        ax2.set_xlabel(L"{\rm Wavelength\ (\AA)}")
        ax2.set_ylabel(L"{\rm IAG/Model}")
        plt.show()
        plt.clf(); plt.close()
    end

    return flux_iag./(tel_model(wavs_iag, fit.param)./flux_sim)
end

# figure 3 -- compare synthetic and IAG spectra + bisectors
function main()
    # get data
    lp = GRASS.LineProperties(exclude=[""])
    files = GRASS.get_file(lp)
    line_names = GRASS.get_name(lp)

    # wavelength of line to synthesize/compare to iag
    for (i, file) in enumerate(files)
        if !contains(file, "FeI_5434")
            continue
        end

        println(">>> Running " * line_names[i] * "...")

        # get properties from line
        line_name = line_names[i]
        airwav = lp.λrest[i]
        depth = lp.depth[i]

        # get IAG spectrum and normalize it
        wavs_iag, flux_iag = GRASS.read_iag_atlas(isolate=true, airwav=airwav, buffer=1.5)
        flux_iag ./= maximum(flux_iag)

        # convolve IAG spectrum to LARS resolution
        wavs_iag1, flux_iag1 = GRASS.convolve_gauss(wavs_iag, flux_iag, new_res=7e5, oversampling=1.0)

        # get depth from IAG spectrum
        idx1 = findfirst(x -> x .<= airwav - 0.125, wavs_iag)
        idx2 = findfirst(x -> x .>= airwav + 0.125, wavs_iag)
        botind = argmin(view(flux_iag, idx1:idx2)) + idx1
        iag_depth = 1.0 - minimum(view(flux_iag, idx1:idx2))

        # set up for GRASS spectrum simulation
        function calculate_ideal_depth()
            dep_iter = true
            iters = 0
            while dep_iter
                # simulate the spectrum
                lines = [airwav]
                depths = [iag_depth - 0.05 + iters * 0.0005]
                templates = [file]
                resolution = 1e6
                spec = SpecParams(lines=lines, depths=depths, templates=templates, resolution=resolution)
                disk = DiskParams(N=132, Nt=5)

                # simulate the spectrum
                wavs_sim, flux_sim = synthesize_spectra(spec, disk, use_gpu=use_gpu, verbose=false)
                flux_sim = dropdims(mean(flux_sim, dims=2), dims=2)

                # get the depth difference
                dep_diff = (1.0 - minimum(flux_sim)) - iag_depth
                if (abs(dep_diff) < 0.001) | (dep_diff > 0.0)
                    dep_iter = false
                    return depths[1]
                end
                iters += 1
            end
        end

        # get the depth
        # sim_depth = 0.760858156406211
        sim_depth = 0.8582716997283104
        println("\t>>> Calculating ideal depth for line synthesis...")
        # sim_depth = calculate_ideal_depth()
        println("\t>>> Ideal depth = " * string(sim_depth))

        # simulate the spectrum
        lines = [airwav]
        depths = [sim_depth]
        templates = [file]
        resolution = 1e6
        spec = SpecParams(lines=lines, depths=depths, templates=templates,
                          resolution=resolution, buffer=1.5)
        disk = DiskParams(N=132, Nt=50)

        # simulate the spectrum
        wavs_sim, flux_sim = synthesize_spectra(spec, disk, use_gpu=use_gpu, verbose=false)
        flux_sim = dropdims(mean(flux_sim, dims=2), dims=2)

        # get bisector for IAG and synthetic spectra
        bis_iag, int_iag = GRASS.calc_bisector(wavs_iag, flux_iag, top=0.9)
        bis_sim, int_sim = GRASS.calc_bisector(wavs_sim, flux_sim, top=0.9)

        # convert wavelengths to vel grids
        vel_iag = GRASS.c_ms .* (bis_iag .- airwav) ./ (airwav)
        vel_sim = GRASS.c_ms .* (bis_sim .- airwav) ./ (airwav)

        # compute velocity as mean bisector between N and M % depth
        N = 0.25
        M = 0.50
        idx1 = findfirst(x -> x .>= N * iag_depth + minimum(flux_iag), int_iag)
        idx2 = findfirst(x -> x .>= M * iag_depth + minimum(flux_iag), int_iag)
        rv_iag = mean(view(vel_iag, idx1:idx2))

        idx1 = findfirst(x -> x .>= N * sim_depth + minimum(flux_sim), int_sim)
        idx2 = findfirst(x -> x .>= M * sim_depth + minimum(flux_sim), int_sim)
        rv_sim = mean(view(vel_sim, idx1:idx2))

        # transform the bisectors
        vel_iag .-= rv_iag
        vel_sim .-= rv_sim
        wavs_iag ./= calc_doppler_factor(rv_iag)
        wavs_sim ./= calc_doppler_factor(rv_sim)

        # interpolate IAG onto synthetic wavelength grid
        itp = GRASS.linear_interp(wavs_iag, flux_iag, bc=NaN)
        flux_iag = itp.(wavs_sim)
        wavs_iag = copy(wavs_sim)

        # clean the IAG spectrum
        flux_mod = model_iag_blends(wavs_sim, flux_sim, wavs_iag, flux_iag, plot=false)
        mod_depth = 1.0 - minimum(flux_mod)

        # recompute bisectors b/c of interpolation
        bis_sim, int_sim = GRASS.calc_bisector(wavs_sim, flux_sim, top=0.9)
        bis_iag, int_iag = GRASS.calc_bisector(wavs_iag, flux_iag, top=0.9)
        bis_mod, int_mod = GRASS.calc_bisector(wavs_iag, flux_mod, top=0.9)

        # transform bisectors to velocities
        vel_sim = GRASS.c_ms .* (bis_sim .- airwav) ./ (airwav)
        vel_iag = GRASS.c_ms .* (bis_iag .- airwav) ./ (airwav)
        vel_mod = GRASS.c_ms .* (bis_mod .- airwav) ./ (airwav)

        # find mean velocities in order to align bisectors
        idx1 = findfirst(x -> x .>= N * sim_depth + minimum(flux_sim), int_sim)
        idx2 = findfirst(x -> x .>= M * sim_depth + minimum(flux_sim), int_sim)
        rv_sim = mean(view(vel_sim, idx1:idx2))

        idx1 = findfirst(x -> x .>= N * iag_depth + minimum(flux_iag), int_iag)
        idx2 = findfirst(x -> x .>= M * iag_depth + minimum(flux_iag), int_iag)
        rv_iag = mean(view(vel_iag, idx1:idx2))

        idx1 = findfirst(x -> x .>= N * mod_depth + minimum(flux_mod), int_mod)
        idx2 = findfirst(x -> x .>= M * mod_depth + minimum(flux_mod), int_mod)
        rv_mod = mean(view(vel_mod, idx1:idx2))

        # zero the velocities to talign bisectors
        vel_sim .-= rv_sim
        vel_iag .-= rv_iag
        vel_mod .-= rv_mod

        # big function for plotting
        function comparison_plots()
            # make plot objects
            fig = plt.figure()
            gs = mpl.gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], figure=fig, hspace=0.05)
            ax1 = fig.add_subplot(gs[1])
            ax2 = fig.add_subplot(gs[2])

            # plot spectra
            ax1.plot(wavs_sim, flux_sim, c="black", lw= 1.5, label=L"{\rm Synthetic}")
            ax1.plot(wavs_iag, flux_iag, marker="s", c=colors[1], ms=2.0, lw=1.0, markevery=2, label=L"{\rm IAG}")
            ax1.plot(wavs_iag, flux_mod, alpha=0.9, marker="o", c=colors[2], ms=2.0, lw=1.0, markevery=2, label=L"{\rm Cleaned\ IAG}")

            # plot resids
            ax2.plot(wavs_sim, flux_iag .- flux_sim, c=colors[1], marker="s", ms=2.0, lw=0, markevery=2)
            ax2.plot(wavs_sim, flux_mod .- flux_sim, c=colors[2], marker="o", ms=2.0, lw=0, markevery=2)

            # set limits
            min_idx = argmin(flux_iag)
            ax1.set_xlim(airwav - 0.5, airwav + 0.5)
            ax1.set_ylim(minimum(flux_sim) - 0.1, 1.1)
            ax2.set_xlim(airwav - 0.5, airwav + 0.5)
            ax2.set_ylim(-0.125, 0.125)

            # set tick labels, axis labels, etc.
            ax1.set_xticklabels([])
            ax1.set_ylabel(L"{\rm Normalized\ Flux}", labelpad=10)
            ax2.set_xlabel(L"{\rm Wavelength\ (\AA)}")
            ax2.set_ylabel(L"{\rm IAG\ -\ Synthetic}")
            ax1.legend()
            fig.tight_layout()

            # set the title
            ax1.set_title(("\${\\rm " * replace(line_name, "_" => "\\ ") * "}\$"))

            # save the plot
            fig.savefig(joinpath(outdir, line_name * "_line.pdf"))
            plt.clf(); plt.close()

            # plot the bisectors
            fig = plt.figure()
            gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1.1], figure=fig, wspace=0.05)
            ax1 = fig.add_subplot(gs[1])
            ax2 = fig.add_subplot(gs[2])

            # plot bisectors
            ax1.plot(vel_sim[2:end], int_sim[2:end], color="black", lw=2.0, markevery=4, label=L"{\rm Synthetic}")
            ax1.plot(vel_iag[2:end], int_iag[2:end], marker="s", c=colors[1], ms=2.0, lw=1.0, markevery=4, label=L"{\rm IAG}")
            ax1.plot(vel_mod[2:end], int_mod[2:end], marker="o", c=colors[2], ms=2.0, lw=1.0, markevery=4, label=L"{\rm Cleaned\ IAG}")

            # plot residuals
            ax2.plot(vel_iag[2:end] .- vel_sim[2:end], int_iag[2:end], c=colors[1], marker="s", ms=2.0, lw=0.0, markevery=4)
            ax2.plot(vel_mod[2:end] .- vel_sim[2:end], int_mod[2:end], c=colors[2], marker="o", ms=2.0, lw=0.0, markevery=4)

            # set tick labels, axis labels, etc.
            ax2.set_yticklabels([])
            ax2.yaxis.tick_right()
            # ax1.set_xlim(5434.4, 5434.6)
            ax1.set_ylim(0.1, 1.1)
            ax2.set_xlim(-20, 20)
            ax2.set_ylim(0.1, 1.1)
            ax1.set_xlabel(L"{\rm Relative\ Velocity\ (ms^{-1})}")
            ax1.set_ylabel(L"{\rm Normalized\ Intensity}")
            ax2.set_xlabel(L"{\rm IAG\ -\ Synthetic\ (ms^{-1})}")
            ax1.legend(loc="upper right", prop=Dict("size"=>10), labelspacing=0.25)

            # set the title
            ax1.set_title(("\${\\rm " * replace(line_name, "_" => "\\ ") * "}\$"))

            # save the plot
            fig.savefig(joinpath(outdir, line_name * "_bisector.pdf"))
            plt.clf(); plt.close()
            return nothing
        end
        comparison_plots()
    end
    return nothing
end

if (run | plot)
    main()
end
