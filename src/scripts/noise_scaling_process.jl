# alias stuff
AA = AbstractArray
AF = AbstractFloat

# pkgs
using CSV
using DataFrames
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

# get the name of template from the command line args
template_idx = tryparse(Int, ARGS[1])
lp = GRASS.LineProperties(exclude=["CI_5380", "NaI_5896"])
line_names = GRASS.get_name(lp)
template = line_names[template_idx]
@show template

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
datadir = string(abspath(data))
datafile = string(abspath(joinpath(data, template * "_picket_fence.jld2")))

# get tuned BIS data
df_tuned = CSV.read(joinpath(datadir, "tuned_params.csv"), DataFrame)
const b1 = df_tuned[template_idx, "b1"]
const b2 = df_tuned[template_idx, "b2"]
const b3 = df_tuned[template_idx, "b3"]
const b4 = df_tuned[template_idx, "b4"]

# set number of trials
const Ntrials = 50

# function to loop over
function std_vs_number_of_lines(wavs::AA{T,1}, flux::AA{T,2}, lines::AA{T,1},
                                depths::AA{T,1}, resolutions::AA{T,1},
                                nlines_to_do::AA{Int,1}, rvs_std::AA{T,3},
                                rvs_std_decorr::AA{T,3}, snr::T;
                                plot::Bool=false, oversampling::T=4.0) where T<:Float64
    # include more and more lines in ccf
    for i in eachindex(resolutions)
        # degrade the resolution
        wavs_degd, flux_degd = GRASS.convolve_gauss(wavs, flux, new_res=resolutions[i],
                                                    oversampling=oversampling)

        # get the location of the minima
        mean_flux_degd = dropdims(mean(flux_degd, dims=2), dims=2)
        pks, vals = findminima(mean_flux_degd)
        pks, proms = peakproms!(pks, mean_flux_degd; minprom=0.1)

        # get view of first line
        wavs_window = view(wavs_degd, 1:pks[1]+round(Int, (pks[2] - pks[1])/2))
        flux_window = view(flux_degd, 1:pks[1]+round(Int, (pks[2] - pks[1])/2), 1)

        # get width of line at ~95% flux
        idxl, idxr = GRASS.find_wing_index(0.95, flux_window)

        # get width in angstroms
        width_ang = wavs_window[idxr] - wavs_window[idxl]

        # convert to velocity
        width_vel = GRASS.c_ms * width_ang / wavs_degd[pks[1]]

        # set the velocity step and get velocity window for CCF
        Δv_step = 100.0
        Δv_max = round((width_vel + 1.5e3)/100) * 100
        if Δv_max < 15e3
            Δv_max = 15e3
        end
        @show Δv_max

        # create the velocity grid
        v_grid = range(-Δv_max, Δv_max, step=Δv_step)

        # allocate memory that will be reused in line loop
        len_v = 1 + round(Int, (Δv_max * 2) / Δv_step)
        ccf1 = zeros(len_v, size(flux_degd,2))
        projection_full = zeros(length(wavs_degd), 1)
        proj_flux_full = zeros(length(wavs_degd))
        bis_inv_slope = zeros(size(flux_degd,2))
        xdata = zeros(size(flux_degd,2))
        ydata = zeros(size(flux_degd,2))

        # allocate flux snr
        flux_snr = deepcopy(flux_degd)

        # loop over number of lines
        for j in eachindex(nlines_to_do)
            # loop over Ntrials
            for n in 1:Ntrials
                # copy it over
                copyto!(flux_snr, flux_degd)

                # get spectrum at specified snr
                GRASS.add_noise!(flux_snr, snr)

                # get number of lines
                n_lines_j = nlines_to_do[j]

                # get lines to include in ccf
                npks_idx = view(pks, 1:n_lines_j)
                ls = wavs_degd[npks_idx]
                ds = 1.0 .- flux_snr[npks_idx]

                # get indices for view of spectrum
                lidx = 1
                if nlines_to_do[j] >= length(pks)
                    ridx = length(wavs_degd)
                else
                    ridx = ceil(Int, pks[nlines_to_do[j]] + (pks[nlines_to_do[j] + 1] - pks[nlines_to_do[j]])/2)
                end

                # take the view
                wavs_view = view(wavs_degd, lidx:ridx)
                flux_view = view(flux_snr, lidx:ridx, :)

                # get views of memory
                projection = view(projection_full, 1:length(wavs_view), :)
                proj_flux = view(proj_flux_full, 1:length(wavs_view))

                # re-zero the allocated memory
                ccf1 .= 0.0
                projection_full .= 0.0
                proj_flux_full .= 0.0
                bis_inv_slope .= 0.0
                xdata .= 0.0
                ydata .= 0.0

                # calculate ccf
                GRASS.calc_ccf!(v_grid, projection, proj_flux, ccf1,
                                wavs_view, flux_view, ls, ds, resolutions[i],
                                Δv_step=Δv_step, Δv_max=Δv_max,
                                mask_type=EchelleCCFs.GaussianCCFMask)

                # calculate the RVs and get the RMS
                rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1, frac_of_width_to_fit=0.5)
                rvs_std[i,j,n] = calc_rms(rvs1)

                # get ccf bisector
                vel, int = GRASS.calc_bisector(v_grid, ccf1, nflux=100, top=0.99)

                # calc bisector summary statistics
                bis_inv_slope .= GRASS.calc_bisector_inverse_slope(vel, int, b1=b1, b2=b2, b3=b3, b4=b4)

                # data to fit
                xdata .= (bis_inv_slope .- mean(bis_inv_slope))
                ydata .= (rvs1 .- mean(rvs1))

                # perform the fit
                pfit = Polynomials.fit(xdata, ydata, 1)

                # decorrelate the velocities
                rvs_to_subtract = pfit.(xdata)
                rvs_std_decorr[i,j,n] = calc_rms(rvs1 .- rvs_to_subtract)
            end
        end
    end
    return nothing
end

# read in the data
d = load(datafile)
wavs = d["wavs"]
flux = d["flux"][:, 1:160]
lines = d["lines"]
depths = d["depths"]
templates = d["templates"]

# snrs to loop over
nlines_to_do = [50, 100, 150, 200, 250, 300, 400, 500]
snrs_for_lines = [200.0, 300.0, 400.0, 500.0, 750.0, 1000.0]
resolutions = [0.98e5, 1.2e5, 1.375e5, 1.9e5, 2.705e5, 3.5e5]

# allocate memory for output
rvs_std_out = zeros(length(resolutions), length(nlines_to_do), Ntrials, length(snrs_for_lines))
rvs_std_decorr_out = zeros(length(resolutions), length(nlines_to_do), Ntrials, length(snrs_for_lines))

# loop over snrs
@threads for k in eachindex(snrs_for_lines)
    # get views of output arrays
    v1 = view(rvs_std_out, :, :, :, k)
    v2 = view(rvs_std_decorr_out, :, :, :, k)

    # get the stuff
    std_vs_number_of_lines(wavs, flux, lines, depths, resolutions, nlines_to_do, v1, v2, snrs_for_lines[k])
end

# write the data out
outfile = string(joinpath(data, template * "_rvs_std_out.jld2"))
jldsave(outfile,
        nlines_to_do=nlines_to_do,
        snrs_for_lines=snrs_for_lines,
        resolutions=resolutions,
        rvs_std_out=rvs_std_out,
        rvs_std_decorr_out=rvs_std_decorr_out)
