# imports
using JLD2
using CUDA
using GRASS
using Printf
using FileIO
using Profile
using Statistics
using BenchmarkTools

# plotting imports
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
const outfile = joinpath(data, "benchmark.jld2")

# read in the data
d = load(outfile)
max_cpu = d["max_cpu"]
nlines = d["nlines"]
n_res = d["n_res"]
n_lam = d["n_lam"]
b_cpu = d["b_cpu"]
b_gpu = d["b_gpu"]
b_gpu32 = d["b_gpu32"]

# get mean gpu benchmark
b_gpu_avg = dropdims(mean(b_gpu, dims=2), dims=2)
b_gpu_std = dropdims(std(b_gpu, dims=2), dims=2)

b_gpu_avg32 = dropdims(mean(b_gpu32, dims=2), dims=2)
b_gpu_std32 = dropdims(std(b_gpu32, dims=2), dims=2)

# compute speedup
speedup = b_cpu[1:max_cpu]./b_gpu_avg[1:max_cpu]

# report largest benchmore for double precision gpu
println(">>> Max GPU benchmark = " * string(maximum(b_gpu_avg)))

# plotting function (use globals who cares i don't)
function plot_scaling(filename; logscale=true)
    # create plotting objects
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()

    # log scale it
    if logscale
        ax1.set_yscale("symlog")
        ax2.set_yscale("symlog")
        scale = "logscale"
    else
        scale = "linscale"
    end

    # plot on ax1
    ms = 7.5
    ax1.plot(n_res[1:max_cpu], b_cpu[1:max_cpu], marker="o", ms=ms, c="k", label=L"{\rm CPU\ (Float64)}")
    ax1.plot(n_res, b_gpu_avg, marker="s", ms=ms, c=colors[1], label=L"{\rm GPU\ (Float64)}")
    ax1.plot(n_res, b_gpu_avg32, marker="^", ms=ms, c=colors[2], label=L"{\rm GPU\ (Float32)}")

    # plot on twin axis
    ax2.plot(n_lam[1:max_cpu], b_cpu[1:max_cpu], marker="o", ms=ms, c="k")
    ax2.plot(n_lam, b_gpu_avg, marker="s", ms=ms, c=colors[1])
    ax2.plot(n_lam, b_gpu_avg32, marker="^", ms=ms, c=colors[2])
    ax2.grid(false)

    # minor tick locator
    if logscale
        locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8))
        ax1.yaxis.set_minor_locator(locmin)
    end

    # axis label stuff
    ax1.set_xlabel(L"{\rm \#\ of\ res.\ elements}")
    ax1.set_ylabel(L"{\rm Synthesis\ time\ (s)}")
    ax2.set_xlabel(L"{\rm Width\ of\ spectrum\ (\AA)}")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.clf(); plt.close()
    return nothing
end

# plot it
plot_scaling(joinpath(figures, "scaling_bench_logscale.pdf"), logscale=true)
plot_scaling(joinpath(figures, "scaling_bench_linscale.pdf"), logscale=false)
