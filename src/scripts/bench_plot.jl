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
const datafile = string(abspath(joinpath(data, "benchmark.jld2")))

# read in the data
d = load(datafile)
max_cpu = d["max_cpu"]
nlines = d["nlines"]
n_res = d["n_res"]
n_lam = d["n_lam"]
t_cpu = d["t_cpu"]
t_gpu = d["t_gpu"]
t_gpu32 = d["t_gpu32"]
m_cpu = d["m_cpu"] * 9.5367431640625e-7
m_gpu = d["m_gpu"] * 9.5367431640625e-7
m_gpu32 = d["m_gpu32"]*  9.5367431640625e-7


# get mean gpu benchmark
t_gpu_avg = dropdims(mean(t_gpu, dims=2), dims=2)
t_gpu_std = dropdims(std(t_gpu, dims=2), dims=2)

t_gpu_avg32 = dropdims(mean(t_gpu32, dims=2), dims=2)
t_gpu_std32 = dropdims(std(t_gpu32, dims=2), dims=2)

m_gpu_avg = dropdims(mean(m_gpu, dims=2), dims=2)
m_gpu_std = dropdims(std(m_gpu, dims=2), dims=2)

m_gpu_avg32 = dropdims(mean(m_gpu32, dims=2), dims=2)
m_gpu_std32 = dropdims(std(m_gpu32, dims=2), dims=2)

# compute speedup
speedup = t_cpu[1:max_cpu]./t_gpu_avg[1:max_cpu]

# plot the speedup
plotfile0 = string(abspath(joinpath(figures, "speedup.pdf")))
plt.plot(n_res[1:max_cpu], speedup)
plt.savefig(plotfile0)

# report largest benchmore for double precision gpu
println(">>> Max GPU benchmark = " * string(maximum(t_gpu_avg)))

# plotting function (use globals who cares i don't)
function plot_scaling(filename; logscale=true)
    # create plotting objects
    # fig, (ax1,ax2) = plt.subplots(figsize=(6.4,9.0), nrows=2, ncols=1, sharex=true)
    fig, ax1 = plt.subplots()
    ax1_t = ax1.twiny()

    # log scale it
    if logscale
        ax1.set_yscale("symlog")
        ax1_t.set_yscale("symlog")
        scale = "logscale"
    else
        scale = "linscale"
    end

    # plot on ax1
    ms = 7.5
    ax1.plot(n_res[1:max_cpu], t_cpu[1:max_cpu], marker="o", ms=ms, c="k", label=L"{\rm CPU\ (Float64)}")
    ax1.plot(n_res, t_gpu_avg32, alpha=1.0, marker="^", mec="k", ms=ms, c=colors[2], label=L"{\rm GPU\ (Float32)}")
    ax1.plot(n_res, t_gpu_avg, alpha=0.66, marker="s", mec="k", ms=ms, c=colors[1], label=L"{\rm GPU\ (Float64)}")

    # plot on twin axis
    ax1_t.plot(n_lam[1:max_cpu], t_cpu[1:max_cpu], marker="o", ms=ms, c="k")
    ax1_t.plot(n_lam, t_gpu_avg32, alpha=1.0, marker="^", mec="k", ms=ms, c=colors[2])
    ax1_t.plot(n_lam, t_gpu_avg, alpha=0.66, marker="s", mec="k", ms=ms, c=colors[1])
    ax1_t.grid(false)

    # minor tick locator
    if logscale
        locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8))
        ax1.yaxis.set_minor_locator(locmin)
    end

    # # plot memory on axis
    # ax2.plot(n_res[1:max_cpu], m_cpu[1:max_cpu], marker="o", ms=ms, c="k")
    # ax2.plot(n_res, m_gpu_avg, marker="s", ms=ms, c=colors[1])
    # ax2.plot(n_res, m_gpu_avg32, marker="^", ms=ms, c=colors[2])

    # axis label stuff
    ax1.set_xlabel(L"{\rm \#\ of\ pixels}")
    # ax2.set_xlabel(L"{\rm \#\ of\ pixels}")
    ax1.set_ylabel(L"{\rm Synthesis\ Time\ (s)}")
    # ax2.set_ylabel(L"{\rm Memory\ Allocated\ (MiB)}")
    ax1_t.set_xlabel(L"{\rm Width\ of\ spectrum\ (\AA)}")
    ax1.legend()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.025)
    fig.savefig(filename)
    plt.clf(); plt.close()
    return nothing
end

# plot it on logscale
plotfile1 = string(abspath(joinpath(figures, "scaling_bench_logscale.pdf")))
plot_scaling(plotfile1, logscale=true)

# plot it on linscale
# plotfile2 = string(abspath(joinpath(figures, "scaling_bench_linscale.pdf")))
# plot_scaling(plotfile2, logscale=false)
