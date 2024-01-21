using CSV
using Glob
using JLD2
using GRASS
using FileIO
using Printf
using Revise
using DataFrames
using Statistics
using Distributions
using BenchmarkTools
using HypothesisTests

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# get command line args and output directories
include("paths.jl")#joinpath(abspath(@__DIR__), "paths.jl"))
plotdir = abspath(string(figures))
datadir = abspath(string(data))

# get the line names
lp = GRASS.LineProperties(exclude=["CI_5380", "NaI_5896"])
line_names = GRASS.get_name(lp)

# find the files
files = Glob.glob("*tune*", datadir)

# initialize a data frame to save results to
df = DataFrame("line" => line_names,
               "b1" => zeros(length(line_names)),
               "b2" => zeros(length(line_names)),
               "b3" => zeros(length(line_names)),
               "b4" => zeros(length(line_names)),
               "med_pearson" => zeros(length(line_names)))

# loop over line names
for (i, name) in enumerate(line_names)
    # get index
    idx = findfirst(x -> contains(x, name), files)
    f = files[idx]

    # load in the data
    d = load(f)
    b1 = d["b1"]
    b2 = d["b2"]
    b3 = d["b3"]
    b4 = d["b4"]
    r_array = d["r_array"]

    # get average and std across trials
    r_avg = dropdims(mean(r_array, dims=2), dims=2)
    r_med = dropdims(median(r_array, dims=2), dims=2)
    r_std = dropdims(std(r_array, dims=2), dims=2)

    # get the best out of each set of trials
    r_best = argmax(abs.(r_array), dims=2)
    r_best = dropdims(r_array[r_best], dims=2)

    # find the best ones
    # idx_sort = sortperm(abs.(r_avg))
    idx_sort_med = sortperm(abs.(r_med))
    idx_sort_best = sortperm(abs.(r_best))

    # set the values in the data frame
    # round to nearest percent
    df[i, "b1"] = round(b1[idx_sort_med[end]], digits=2)
    df[i, "b2"] = round(b2[idx_sort_med[end]], digits=2)
    df[i, "b3"] = round(b3[idx_sort_med[end]], digits=2)
    df[i, "b4"] = round(b4[idx_sort_med[end]], digits=2)
    df[i, "med_pearson"] = r_avg[idx_sort_med[end]]
end

# write it to disk
outfile = joinpath(data, "tuned_params.csv")
CSV.write(outfile, df)
