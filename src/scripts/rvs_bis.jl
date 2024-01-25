using CSV
using CUDA
using GRASS
using Printf
using Revise
using PyCall
using DataFrames
using Statistics
using EchelleCCFs
using Polynomials
using BenchmarkTools
using HypothesisTests

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
pe = pyimport("matplotlib.patheffects")
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

include(joinpath(abspath(@__DIR__), "paths.jl"))
datadir = abspath(string(data))
outdir = abspath(string(figures))
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

# get optimized depths
df = CSV.read(joinpath(datadir, "optimized_depth.csv"), DataFrame)

for (idx, file) in enumerate(lp.file)
    if !contains(file, "TiII_5381")
        continue
    end

    # get line title
    title = replace(line_names[idx], "_" => "\\ ")
    tidx = findfirst('I', title)
    title = title[1:tidx-1] * "\\ " * title[tidx:end]
    title = ("\${\\rm " * title * "\\ \\AA }\$")

    # set up paramaters for spectrum
    Nt = 500
    lines = [rest_wavelengths[idx]]
    depths = [df[idx, "optimized_depth"]]
    templates = [file]
    variability = repeat([true], length(lines))
    blueshifts = zeros(length(lines))
    resolution = 7e5
    seed_rng = false

    disk = DiskParams(Nt=Nt)
    spec1 = SpecParams(lines=lines, depths=depths, variability=variability,
                       blueshifts=blueshifts, templates=templates,
                       resolution=resolution, oversampling=2.0)
    wavs_out, flux_out = synthesize_spectra(spec1, disk, seed_rng=seed_rng,
                                            verbose=true, use_gpu=true)

    # convolve down and oversample
    # wavs_out, flux_out = GRASS.convolve_gauss(wavs_out, flux_out, new_res=6e5, oversampling=4.0)

    # set mask width
    mask_width = (GRASS.c_ms/resolution)

    # calculate a ccf
    println("\t>>> Calculating CCF...")
    v_grid, ccf1 = calc_ccf(wavs_out, flux_out, lines, depths,
                            resolution, mask_width=mask_width,
                            mask_type=EchelleCCFs.GaussianCCFMask,
                            Δv_step=100.0, Δv_max=32e3)
    rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1, frac_of_width_to_fit=0.5)

    # calculate bisector
    bis, int = GRASS.calc_bisector(v_grid, ccf1, nflux=100, top=0.99)

    # convert bis to velocity scale
    # bis, int = GRASS.calc_bisector(wavs_out, flux_out, nflux=80, top=0.99)
    # for i in 1:Nt
    #     bis[:, i] = (bis[:,i] .- lines[1]) * GRASS.c_ms / lines[1]
    # end

    # get continuum and depth
    top = 1.0
    dep = top - minimum(int)

    # set the BIS regions
    b1 = 0.10
    b2 = 0.40
    b3 = 0.55
    b4 = 0.90

    # get the BIS regions
    idx10 = findfirst(x -> x .> top - b1 * dep, int[:,1])
    idx40 = findfirst(x -> x .> top - b2 * dep, int[:,1])
    idx55 = findfirst(x -> x .> top - b3 * dep, int[:,1])
    idx90 = findfirst(x -> x .> top - b4 * dep, int[:,1])

    # create fig + axes objects
    println("\t>>> Plotting...")
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18.2, 7.2))
    fig.subplots_adjust(wspace=0.05)

    # plot the bis regions
    ax1.axhline(int[idx10,1], ls="--", c="k")
    ax1.axhline(int[idx40,1], ls="--", c="k")
    ax1.axhline(int[idx55,1], ls=":", c="k")
    ax1.axhline(int[idx90,1], ls=":", c="k")

    # get BIS
    bis_inv_slope = GRASS.calc_bisector_inverse_slope(bis, int)#, b1=b1, b2=b2, b3=b3, b4=b4)
    bis_amplitude = GRASS.calc_bisector_span(bis, int)
    # bis_amplitude = maximum(dropdims(minimum(int, dims=1), dims=1)) .- dropdims(minimum(int, dims=1), dims=1)
    bis_curvature = GRASS.calc_bisector_curvature(bis, int)

    # smooth bisector for plotting
    bis = GRASS.moving_average(bis, 4)
    int = GRASS.moving_average(int, 4)

    # mean subtract the bisector plotting (remove bottommost and topmost measurements)
    idx_start = 1#4
    idx_d_end = 0#1
    mean_bis = dropdims(mean(bis, dims=2), dims=2)[idx_start:end-idx_d_end]
    mean_int = dropdims(mean(int, dims=2), dims=2)[idx_start:end-idx_d_end]
    bis .-= mean(mean_bis)

    # plot the variability in bisector on exaggerated scale
    for i in 1:2:200
        bis_i = view(bis, :, i)[idx_start:end-idx_d_end]
        int_i = view(int, :, i)[idx_start:end-idx_d_end]

        delt_bis = (mean_bis .- mean(mean_bis) .- bis_i) .* 50.0
        if i == 1
            ax1.plot(mean_bis .- mean(mean_bis) .+ delt_bis, int_i, alpha=0.15, c=colors[1], lw=2.0, label=L"{\rm Bisector\ Variations}")
        else
            ax1.plot(mean_bis .- mean(mean_bis) .+ delt_bis, int_i, alpha=0.15, c=colors[1], lw=2.0)
        end
    end

    # plot the mean bisector
    mean_bis .-= mean(mean_bis)
    ax1.plot(mean_bis, mean_int, c=colors[1], lw=3.0, label=L"{\rm Mean\ Bisector}")

    # subtract off mean
    xdata = bis_inv_slope .- mean(bis_inv_slope)
    ydata = rvs1 .- mean(rvs1)

    # fit to the BIS
    pfit = Polynomials.fit(xdata, ydata, 1)
    xmodel = range(minimum(xdata)-0.1*std(xdata), maximum(xdata)+0.1*std(xdata), length=5)
    ymodel = pfit.(xmodel)

    @show calc_rms(rvs1)
    @show calc_rms(rvs1 .- pfit.(xdata))

    # get the slope of the fit
    corr = round(Statistics.cor(xdata, ydata), digits=3)
    fit_label = "\$ " .* string(corr) .* "\$"

    # set patheffecfts
    path_effects =[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()]

    # plot BIS and apparent RV
    ax2.scatter(xdata, ydata, c=colors[1], s=8, alpha=0.9)
    ax2.plot(xmodel, ymodel, c=colors[1], path_effects=path_effects, ls="--", lw=2.5, label=L"{\rm R } \approx\ " * fit_label)

    # fit to the bis span
    xdata = bis_amplitude .- mean(bis_amplitude)
    ydata = rvs1 .- mean(rvs1)

    pfit = Polynomials.fit(xdata, ydata, 1)
    xmodel = range(minimum(xdata)-0.1*std(xdata), maximum(xdata)+0.1*std(xdata), length=5)
    ymodel = pfit.(xmodel)

    @show calc_rms(rvs1 .- pfit.(xdata))

    # get the slope of the fit
    corr = round(Statistics.cor(xdata, ydata), digits=3)
    fit_label = "\$ " .* string(corr) .* "\$"

    # set patheffecfts
    path_effects =[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()]

    # plot BIS and apparent RV
    ax3.scatter(xdata, ydata, c=colors[1], s=8, alpha=0.9)
    ax3.plot(xmodel, ymodel, c=colors[1], path_effects=path_effects, ls="--", lw=2.5, label=L"{\rm R } \approx\ " * fit_label)

    # set font sizes
    title_font = 26
    tick_font = 20
    legend_font = 20

    # set plot stuff for first plot
    ax1.legend(loc="upper center", ncol=2, columnspacing=0.7)
    ax1.set_ylim(0.15, 1.10)
    ax1.set_xlabel(L"\Delta v\ {\rm (m\ s}^{-1}{\rm )}", fontsize=title_font)
    ax1.set_ylabel(L"{\rm Normalized\ Intensity}", fontsize=title_font)
    ax1.tick_params(axis="both", which="major", labelsize=tick_font+4)

    # set plot stuff for second plot
    ax2.tick_params(axis="both", which="major", labelsize=tick_font+4)
    ax2.legend(loc="upper right", fontsize=legend_font, ncol=1,
              columnspacing=0.8, handletextpad=0.5, labelspacing=0.08)
    ax2.yaxis.tick_right()
    ax2.set_yticklabels([])
    # ax2.yaxis.set_label_position("right")
    ax2.set_xlabel(L"{\rm BIS}\ - \overline{\rm BIS}\ {\rm (m\ s}^{-1}{\rm )}", fontsize=title_font)#, x=0.03, y=0.52)
    # ax2.set_ylabel(L"{\rm RV\ - \overline{\rm RV}\ } {\rm (m\ s}^{-1}{\rm )}", fontsize=title_font)#, x=0.55, y=0.05)


    ax3.tick_params(axis="both", which="major", labelsize=tick_font+4)
    ax3.legend(loc="upper left", fontsize=legend_font, ncol=1,
              columnspacing=0.8, handletextpad=0.5, labelspacing=0.08)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.set_xlabel(L"{\rm a_b}\ - \overline{\rm a_b}\ {\rm (m\ s}^{-1}{\rm )}", fontsize=title_font)#, x=0.03, y=0.52)
    ax3.set_ylabel(L"{\rm RV\ - \overline{\rm RV}\ } {\rm (m\ s}^{-1}{\rm )}", fontsize=title_font)#, x=0.55, y=0.05)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    # fig.savefig(joinpath(outdir, "rvs_bis.pdf"), bbox_inches="tight")
    plt.show()
    plt.close("all")
end
