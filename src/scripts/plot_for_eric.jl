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
color = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# animation stuff
pe = pyimport("matplotlib.patheffects")
cmx = pyimport("matplotlib.cm")
ffmpeg = pyimport("ffmpeg")
pycopy = pyimport("copy")
colors = pyimport("matplotlib.colors")
animation = pyimport("matplotlib.animation")
LineCollection = pyimport("matplotlib.collections").LineCollection

# get paths
include("paths.jl")
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

# get optimized depths
df = CSV.read(joinpath(datadir, "optimized_depth.csv"), DataFrame)

# get line properties
lp = GRASS.LineProperties()
line_species = GRASS.get_species(lp)
rest_wavelengths = GRASS.get_rest_wavelength(lp)
line_depths = GRASS.get_depth(lp)
line_names = GRASS.get_name(lp)

# get line title
idx = 9
file = line_names[idx]
title = replace(line_names[idx], "_" => "\\ ")
tidx = findfirst('I', title)
title = title[1:tidx-1] * "\\ " * title[tidx:end]
title = ("\${\\rm " * title * "\\ \\AA }\$")

# set the color
instr_idx = 4
c = color[instr_idx]
instrument = ["LARS", "PEPSI", "NEID", "KPF"]
res_float = [7e5, 2.7e5, 1.2e5, 0.98e5]
res_string = ["700{,}000", "270{,}000", "120{,}000", "98{,}000",]
corrs = [0.85, 0.75, 0.70, 0.65]

# set the dpi
dpi = 100

# set up paramaters for spectrum
Nt = 500
lines = [rest_wavelengths[idx]]
depths = [df[idx, "optimized_depth"]]
templates = [file]
variability = repeat([true], length(lines))
blueshifts = zeros(length(lines))
resolution = 7e5
seed_rng = false

# set up objects for synthesis
disk = DiskParams(Nt=Nt)
spec1 = SpecParams(lines=lines, depths=depths, variability=variability,
                   blueshifts=blueshifts, templates=templates,
                   resolution=resolution, oversampling=2.0)

global keep_going = true
while keep_going
    # generate the spectra
    wavs_out, flux_out = synthesize_spectra(spec1, disk, seed_rng=seed_rng,
                                            verbose=true, use_gpu=true)

    # convolve down and oversample
    if instr_idx != 1
        wavs_out, flux_out = GRASS.convolve_gauss(wavs_out, flux_out, new_res=res_float[instr_idx], oversampling=4.0)
    end

    # set mask width
    mask_width = (GRASS.c_ms/resolution)

    # calculate a ccf
    println("\t>>> Calculating CCF...")
    v_grid, ccf1 = calc_ccf(wavs_out, flux_out, lines, depths,
                            resolution, mask_width=mask_width,
                            mask_type=EchelleCCFs.GaussianCCFMask,
                            Δv_step=50.0, Δv_max=32e3)
    rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1, frac_of_width_to_fit=0.5)

    # calculate bisector
    bis, int = GRASS.calc_bisector(v_grid, ccf1, nflux=100, top=0.99)

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

    # get BIS
    bis_inv_slope = GRASS.calc_bisector_inverse_slope(bis, int, b1=b1, b2=b2, b3=b3, b4=b4)

    # mean subtract the bisector plotting (remove bottommost and topmost measurements)
    idx_start = 3#4
    idx_d_end = 0#1
    mean_bis = dropdims(mean(bis, dims=2), dims=2)[idx_start:end-idx_d_end]
    mean_int = dropdims(mean(int, dims=2), dims=2)[idx_start:end-idx_d_end]
    bis .-= mean(mean_bis)

    # subtract off mean
    xdata = bis_inv_slope .- mean(bis_inv_slope)
    ydata = rvs1 .- mean(rvs1)

    corr = round(Statistics.cor(xdata, ydata), digits=3)
    fit_label = "\$ " .* string(round(corr, digits=3)) .* "\$"

    @show corr

    # fit to the BIS
    pfit = Polynomials.fit(xdata, ydata, 1)
    xmodel = range(minimum(xdata)-0.75*std(xdata), maximum(xdata)+0.75*std(xdata), length=5)
    ymodel = pfit.(xmodel)

    # decide whether to move on
    global keep_going = abs(corr) <= corrs[instr_idx]

    if !keep_going
        global bis = bis
        global int = int
        global idx10 = idx10
        global idx40 = idx40
        global idx55 = idx55
        global idx90 = idx90
        global xdata = xdata
        global ydata = ydata
        global xmodel = xmodel
        global ymodel = ymodel
        global mean_int = mean_int
        global mean_bis = mean_bis
        global fit_label = fit_label
        global idx_start = idx_start
        global idx_d_end = idx_d_end
        global bis_inv_slope = bis_inv_slope
        global wavs_out = wavs_out
        global flux_out = flux_out
    end
end

# now plot the final frame
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12.15, 7.2))
fig.subplots_adjust(wspace=0.05)

# plot the bis regions
ax1.axhline(int[idx10,1], ls="--", c="k")
ax1.axhline(int[idx40,1], ls="--", c="k")
ax1.axhline(int[idx55,1], ls=":", c="k")
ax1.axhline(int[idx90,1], ls=":", c="k")
ax2.plot(xmodel, ymodel, c="k", ls="--", label=L"{\rm R } \approx\ " * fit_label * L"{\rm , SNR } \sim \infty", zorder=1)

# create lines
path_effects =[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()]

# annotate vt and vb
arrowprops = Dict("facecolor"=>"black", "shrink"=>0.05, "width"=>2.0,"headwidth"=>8.0)
ax1.annotate(L"v_b", xy=(550,int[idx55,1]), xytext=(550,0.4), arrowprops=arrowprops, fontsize=24, ha="center")
ax1.annotate("", xy=(550, int[idx90,1]), xytext=(550,0.38), arrowprops=arrowprops, fontsize=24, ha="center")

ax1.annotate(L"v_t", xy=(550,int[idx10,1]), xytext=(550,0.8), arrowprops=arrowprops, fontsize=24, ha="center")
ax1.annotate("", xy=(550, int[idx40,1]), xytext=(550,0.78), arrowprops=arrowprops, fontsize=24, ha="center")

# annotate the xaxis
ax1.text(-200, 1.05, L"R \sim\ " * latexstring(res_string[instr_idx]) * L"{\rm \ @ \ SNR } \sim \infty", fontsize=18)
# ax1.text(-200, 1.05, L"R \sim\ " * latexstring(res_string[instr_idx]), fontsize=18)
ax1.text(-200, 1.0, L"{\rm (Variability\ exaggerated\ 50x)}", fontsize=18)

# set font sizes
title_font = 26
tick_font = 20
legend_font = 20

# set windows
# ax1.legend(loc="upper center", ncol=2, columnspacing=0.7)
ax1.set_xlim(-210, 710)
ax1.set_ylim(0.15, 1.10)
ax1.set_xlabel(L"\Delta v\ {\rm (m\ s}^{-1}{\rm )}", fontsize=title_font)
ax1.set_ylabel(L"{\rm Normalized\ Flux}", fontsize=title_font)
ax1.tick_params(axis="both", which="major", labelsize=tick_font+4)

# set plot stuff for second plot
ax2.set_xlim(-1.25, 1.25)
ax2.set_ylim(-2.25, 2.25)
ax2.tick_params(axis="both", which="major", labelsize=tick_font+4)
# ax2.legend(loc="upper right", fontsize=legend_font, ncol=1,
          # columnspacing=0.8, handletextpad=0.5, labelspacing=0.08)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_xlabel(L"{\rm BIS}\ - \overline{\rm BIS}\ {\rm (m\ s}^{-1}{\rm )}", fontsize=title_font)#, x=0.03, y=0.52)
ax2.set_ylabel(L"{\rm RV\ - \overline{\rm RV}\ } {\rm (m\ s}^{-1}{\rm )}", fontsize=title_font)#, x=0.55, y=0.05)
# ax2.legend(loc="upper right")

# fig.suptitle()
fig.tight_layout()
fig.subplots_adjust(wspace=0.05)

ax2.plot(xdata, ydata, alpha=0.8, marker="o", color=c, ls="none", lw=2.0, zorder=0)

for i in 1:Nt
    bis_i = view(bis, :, i)[idx_start:end-idx_d_end]
    int_i = view(int, :, i)[idx_start:end-idx_d_end]

    delt_bis = (mean_bis .- mean(mean_bis) .- bis_i) .* 50.0

    # also plot the current one as new line
    ax1.plot(mean_bis .- mean(mean_bis) .+ delt_bis, int_i, alpha=0.1, c=c, zorder=0)

    # plot the last frame
    if i == Nt
        ax1.plot(dropdims(mean(bis, dims=2), dims=2)[idx_start:end-idx_d_end],
                 dropdims(mean(int, dims=2), dims=2)[idx_start:end-idx_d_end],
                 c=c, path_effects=path_effects, ls="-", alpha=1.0, lw=2.0, zorder=1)[1]
    end
end

fig.savefig("/storage/home/mlp95/rvs_bis_no_noise_" * instrument[instr_idx] * ".pdf")
plt.close("all")








# now inject noise
wavs_nois = copy(wavs_out)
flux_nois = copy(flux_out)

# set snr
snr = 500.0 * sqrt(144)

# get spectrum at specified snr
GRASS.add_noise!(flux_nois, snr)

mask_width = (GRASS.c_ms/resolution)

# calculate a ccf
println("\t>>> Calculating CCF...")
v_grid, ccf1 = calc_ccf(wavs_nois, flux_nois, lines, depths,
                        resolution, mask_width=mask_width,
                        mask_type=EchelleCCFs.GaussianCCFMask,
                        Δv_step=50.0, Δv_max=32e3)
rvs1, sigs1 = calc_rvs_from_ccf(v_grid, ccf1, frac_of_width_to_fit=0.5)

# calculate bisector
bis, int = GRASS.calc_bisector(v_grid, ccf1, nflux=100, top=0.99)

# get continuum and depth
top = 1.0
dep = top - minimum(int)

# set the BIS regions
b1 = 0.20
b2 = 0.40
b3 = 0.55
b4 = 0.90

# get the BIS regions
idx10 = findfirst(x -> x .> top - b1 * dep, int[:,1])
idx40 = findfirst(x -> x .> top - b2 * dep, int[:,1])
idx55 = findfirst(x -> x .> top - b3 * dep, int[:,1])
idx90 = findfirst(x -> x .> top - b4 * dep, int[:,1])

# get BIS
bis_inv_slope = GRASS.calc_bisector_inverse_slope(bis, int, b1=b1, b2=b2, b3=b3, b4=b4)

# mean subtract the bisector plotting (remove bottommost and topmost measurements)
idx_start = 3#4
idx_d_end = 0#1
mean_bis = dropdims(mean(bis, dims=2), dims=2)[idx_start:end-idx_d_end]
mean_int = dropdims(mean(int, dims=2), dims=2)[idx_start:end-idx_d_end]
bis .-= mean(mean_bis)

# subtract off mean
xdata = bis_inv_slope .- mean(bis_inv_slope)
ydata = rvs1 .- mean(rvs1)

# fit to the BIS
pfit = Polynomials.fit(xdata, ydata, 1)
xmodel = range(minimum(xdata)-0.75*std(xdata), maximum(xdata)+0.75*std(xdata), length=5)
ymodel = pfit.(xmodel)


# now plot the final frame
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12.15, 7.2))
fig.subplots_adjust(wspace=0.05)

# plot the bis regions
ax1.axhline(int[idx10,1], ls="--", c="k")
ax1.axhline(int[idx40,1], ls="--", c="k")
ax1.axhline(int[idx55,1], ls=":", c="k")
ax1.axhline(int[idx90,1], ls=":", c="k")
ax2.plot(xmodel, ymodel, c="k", ls="--", label=L"{\rm R } \approx\ " * fit_label * L"{\rm , SNR } \sim \infty", zorder=1)

# create lines
path_effects =[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()]

# annotate vt and vb
arrowprops = Dict("facecolor"=>"black", "shrink"=>0.05, "width"=>2.0,"headwidth"=>8.0)
ax1.annotate(L"v_b", xy=(550,int[idx55,1]), xytext=(550,0.4), arrowprops=arrowprops, fontsize=24, ha="center")
ax1.annotate("", xy=(550, int[idx90,1]), xytext=(550,0.38), arrowprops=arrowprops, fontsize=24, ha="center")

ax1.annotate(L"v_t", xy=(550,int[idx10,1]), xytext=(550,0.8), arrowprops=arrowprops, fontsize=24, ha="center")
ax1.annotate("", xy=(550, int[idx40,1]), xytext=(550,0.78), arrowprops=arrowprops, fontsize=24, ha="center")

# annotate the xaxis
ax1.text(-200, 1.05, L"R \sim\ " * latexstring(res_string[instr_idx]) * L"{\rm \ @ \ SNR } \sim 500", fontsize=18)
# ax1.text(-200, 1.05, L"R \sim\ " * latexstring(res_string[instr_idx]), fontsize=18)
ax1.text(-200, 1.0, L"{\rm (Variability\ exaggerated\ 50x)}", fontsize=18)

# set font sizes
title_font = 26
tick_font = 20
legend_font = 20

# set windows
# ax1.legend(loc="upper center", ncol=2, columnspacing=0.7)
ax1.set_xlim(-210, 710)
ax1.set_ylim(0.15, 1.10)
ax1.set_xlabel(L"\Delta v\ {\rm (m\ s}^{-1}{\rm )}", fontsize=title_font)
ax1.set_ylabel(L"{\rm Normalized\ Flux}", fontsize=title_font)
ax1.tick_params(axis="both", which="major", labelsize=tick_font+4)

# set plot stuff for second plot
ax2.set_xlim(-1.25, 1.25)
ax2.set_ylim(-2.25, 2.25)
ax2.tick_params(axis="both", which="major", labelsize=tick_font+4)
# ax2.legend(loc="upper right", fontsize=legend_font, ncol=1,
          # columnspacing=0.8, handletextpad=0.5, labelspacing=0.08)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_xlabel(L"{\rm BIS}\ - \overline{\rm BIS}\ {\rm (m\ s}^{-1}{\rm )}", fontsize=title_font)#, x=0.03, y=0.52)
ax2.set_ylabel(L"{\rm RV\ - \overline{\rm RV}\ } {\rm (m\ s}^{-1}{\rm )}", fontsize=title_font)#, x=0.55, y=0.05)
# ax2.legend(loc="upper right")

# fig.suptitle()
fig.tight_layout()
fig.subplots_adjust(wspace=0.05)

ax2.plot(xdata, ydata, alpha=0.8, marker="o", color=c, ls="none", lw=2.0, zorder=0)

for i in 1:Nt
    bis_i = view(bis, :, i)[idx_start:end-idx_d_end]
    int_i = view(int, :, i)[idx_start:end-idx_d_end]

    delt_bis = (mean_bis .- mean(mean_bis) .- bis_i) .* 50.0

    # also plot the current one as new line
    ax1.plot(mean_bis .- mean(mean_bis) .+ delt_bis, int_i, alpha=0.1, c=c, zorder=0)

    # plot the last frame
    if i == Nt
        ax1.plot(dropdims(mean(bis, dims=2), dims=2)[idx_start:end-idx_d_end],
                 dropdims(mean(int, dims=2), dims=2)[idx_start:end-idx_d_end],
                 c=c, path_effects=path_effects, ls="-", alpha=1.0, lw=2.0, zorder=1)[1]
    end
end

fig.savefig("/storage/home/mlp95/rvs_bis_SNR_500_" * instrument[instr_idx] * ".pdf")
plt.close("all")





