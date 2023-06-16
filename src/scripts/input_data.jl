# imports
using CSV
using GRASS
using PyCall
using DataFrames
using Statistics

# plotting imports
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
using LaTeXStrings
mpl.style.use(joinpath(GRASS.moddir, "fig.mplstyle"))

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))

# read in the spectrum
datafile = string(abspath(joinpath(static, "FeI_6302_spec.jld2")))
d = load(datafile)
wavs = d["wavs"]
flux = d["flux"]
nois = d["noise"]


