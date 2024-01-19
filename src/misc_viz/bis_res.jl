using CUDA
using GRASS
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
colors = ["k", "#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

include(joinpath(abspath(@__DIR__), "paths.jl"))
outdir = abspath(joinpath(string(figures), "rvs_bis/"))
if !isdir(outdir)
    mkdir(outdir)
end

function round_and_format(num::Float64)
    rounded_num = Int(round(num))
    formatted_num = collect(string(rounded_num))
    num_length = length(formatted_num)

    if num_length <= 3
        return prod(formatted_num)
    end

    comma_idx = mod(num_length, 3)
    if comma_idx == 0
        comma_idx = 3
    end

    while comma_idx < num_length
        insert!(formatted_num, comma_idx+1, ',')
        comma_idx += 4
        num_length += 1
    end

    return replace(prod(formatted_num), "," => "{,}")
end

# get line properties
lp = GRASS.LineProperties()
line_species = GRASS.get_species(lp)
rest_wavelengths = GRASS.get_rest_wavelength(lp)
line_depths = GRASS.get_depth(lp)
line_names = GRASS.get_name(lp)

# set oversampling and resolutions, set color sequence
oversampling = 4.0
resolutions = reverse([0.98e5, 1.2e5, 1.37e5, 2.7e5, 7e5])
instruments = ["LARS", "PEPSI", "EXPRES", "NEID", "KPF"]

# get labels for resolutions
labels = "\$ R \\sim " .* round_and_format.(resolutions) .* "{\\rm \\ (" .* instruments .* ")}" .* "\$"

depths_plot = zeros(length(lp.file))
slopes_plot = zeros(length(lp.file), length(resolutions))

for (idx, file) in enumerate(lp.file)
    if !contains(file, "FeI_5434")
        continue
    end

    # set up paramaters for spectrum
    N = 132
    Nt = 2
    lines = [rest_wavelengths[idx]]
    depths = [line_depths[idx]]
    geffs = [0.0]
    templates = [file]
    variability = repeat([true], length(lines))
    resolution = 7e5
    seed_rng = true

    disk = DiskParams(Nt=Nt)
    spec = SpecParams(lines=lines, depths=depths, variability=variability,
                       geffs=geffs, templates=templates, resolution=resolution, oversampling=8.0)
    wavs, flux = synthesize_spectra(spec, disk, seed_rng=true, verbose=true, use_gpu=true)

    # set up plot
    fig, ax1 = plt.subplots(figsize=(6.4, 6.4))

    for j in eachindex(resolutions)
        if resolutions[j] == 7e5
            wavs_degd = wavs
            flux_degd = flux
        else
            # degrade the resolution
            wavs_degd, flux_degd = GRASS.convolve_gauss(wavs, flux,
                                                        new_res=resolutions[j],
                                                        oversampling=15.0)
        end

        mask_width = (GRASS.c_ms/resolutions[j]) * (2.0/10.0)
        v_grid, ccf1 = calc_ccf(wavs_degd, flux_degd, lines, depths,
                                resolutions[j], mask_width=mask_width,
                                mask_type=EchelleCCFs.TopHatCCFMask,
                                Δv_step=10.0, Δv_max=30e3)

        bis, int = GRASS.calc_bisector(v_grid, ccf1, nflux=100, top=0.9)
        bis = mean(bis, dims=2)
        int = mean(int, dims=2)

        bis = GRASS.moving_average(bis, 4)
        int = GRASS.moving_average(int, 4)


        ax1.plot(bis[4:end-4], int[4:end-4], label=labels[j], c=colors[j])

    end


    ax1.set_xlabel(L"\Delta v\ {\rm (m s } ^{-1} {\rm )}")
    ax1.set_ylabel(L"{\rm Normalized\ Flux}")
    # ax1.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
    fig.tight_layout()
    fig.savefig(joinpath(homedir(), "bis_res.pdf"))
    # plt.show()
    plt.clf(); plt.close()
end
