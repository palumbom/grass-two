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
               "bis_med_pearson" => zeros(length(line_names)),
               "c1" => zeros(length(line_names)),
               "c2" => zeros(length(line_names)),
               "c3" => zeros(length(line_names)),
               "c4" => zeros(length(line_names)),
               "c5" => zeros(length(line_names)),
               "c6" => zeros(length(line_names)),
               "curve_med_pearson" => zeros(length(line_names)),)

# loop over line names
for (i, name) in enumerate(line_names)
    # get index
    idx = findfirst(x -> contains(x, name), files)
    f = files[idx]

    # load in the data
    d = load(f)
    @show d
    b1 = d["b1"]
    b2 = d["b2"]
    b3 = d["b3"]
    b4 = d["b4"]
    r_bis_array = d["r_bis_array"]

    # get average and std across trials
    r_bis_avg = dropdims(mean(r_bis_array, dims=2), dims=2)
    r_bis_med = dropdims(median(r_bis_array, dims=2), dims=2)
    r_bis_std = dropdims(std(r_bis_array, dims=2), dims=2)

    # get the best out of each set of trials
    r_bis_best = argmax(abs.(r_bis_array), dims=2)
    r_bis_best = dropdims(r_bis_array[r_best], dims=2)

    # find the best ones
    idx_sort_bis_med = sortperm(abs.(r_bis_med))
    idx_sort_bis_best = sortperm(abs.(r_bis_best))

    # set the values in the data frame
    # round to nearest percent
    df[i, "b1"] = round(b1[idx_sort_bis_med[end]], digits=2)
    df[i, "b2"] = round(b2[idx_sort_bis_med[end]], digits=2)
    df[i, "b3"] = round(b3[idx_sort_bis_med[end]], digits=2)
    df[i, "b4"] = round(b4[idx_sort_bis_med[end]], digits=2)
    df[i, "bis_med_pearson"] = r_bis_med[idx_sort_bis_med[end]]

    # load in the data
    c1 = d["c1"]
    c2 = d["c2"]
    c3 = d["c3"]
    c4 = d["c4"]
    c5 = d["c5"]
    c6 = d["c6"]
    r_curve_array = d["r_curve_array"]

    # get average and std across trials
    r_curve_avg = dropdims(mean(r_curve_array, dims=2), dims=2)
    r_curve_med = dropdims(median(r_curve_array, dims=2), dims=2)
    r_curve_std = dropdims(std(r_curve_array, dims=2), dims=2)

    # get the best out of each set of trials
    r_curve_best = argmax(abs.(r_curve_array), dims=2)
    r_curve_best = dropdims(r_curve_array[r_best], dims=2)

    # find the best ones
    idx_sort_curve_med = sortperm(abs.(r_curve_med))
    idx_sort_curve_best = sortperm(abs.(r_curve_best))

    # set the values in the data frame
    # round to nearest percent
    df[i, "c1"] = round(c1[idx_sort_curve_med[end]], digits=2)
    df[i, "c2"] = round(c2[idx_sort_curve_med[end]], digits=2)
    df[i, "c3"] = round(c3[idx_sort_curve_med[end]], digits=2)
    df[i, "c4"] = round(c4[idx_sort_curve_med[end]], digits=2)
    df[i, "c5"] = round(c5[idx_sort_curve_med[end]], digits=2)
    df[i, "c6"] = round(c6[idx_sort_curve_med[end]], digits=2)
    df[i, "curve_med_pearson"] = r_curve_med[idx_sort_curve_med[end]]
end

# write it to disk
outfile = joinpath(data, "tuned_params.csv")
CSV.write(outfile, df)
