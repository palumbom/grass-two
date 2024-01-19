# imports
using Pkg; Pkg.activate(".")
using CUDA
using GRASS
using Statistics

# plotting imports
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
using PyCall; animation = pyimport("matplotlib.animation");
pe = pyimport("matplotlib.patheffects");
mpl.style.use(GRASS.moddir * "fig.mplstyle")

# lines to plot
linez = ["FeI_5434", "FeI_5383"]
labels1 = [L"{\rm Fe\ I\ 5434\ \AA}", L"{\rm Fe\ I\ 5383\ \AA}"]
labels2 = [L"\mu = 1.0", L"\mu = 0.4"]
linestyles = ["-", "--", ":"]
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]
restwavs = [5434.5232, 6170.0, 5383.0]

# initialize plot objects
fig, ax1 = plt.subplots()

# set the key
keyz = [(:c, :mu10), (:n, :mu04)]

for i in eachindex(linez)
    # get input data
    bisinfo = GRASS.SolarData(fname = GRASS.soldir * linez[i], extrapolate=true)

    for j in eachindex(keyz)
        # find average and std
        avg_bis = mean(bisinfo.bis[keyz[j]], dims=2)
        avg_int = mean(bisinfo.int[keyz[j]], dims=2)
        std_bis = 0.5 * std(bisinfo.bis[keyz[j]], dims=2)
        std_int = 0.5 * std(bisinfo.int[keyz[j]], dims=2)

        # convert to doppler velocity
        avg_bis = avg_bis ./ restwavs[i] .* GRASS.c_ms
        std_bis = std_bis ./ restwavs[i] .* GRASS.c_ms

        # cut off top portion, where uncertainty is large
        idx = findfirst(avg_int .> 0.8)[1]
        avg_bis = avg_bis[2:idx]
        avg_int = avg_int[2:idx]
        std_bis = std_bis[2:idx]
        std_int = std_int[2:idx]

        # fix dimensions
        y = reshape(avg_int, length(avg_int))
        x1 = reshape(avg_bis .+ std_bis, length(avg_int))
        x2 = reshape(avg_bis .- std_bis, length(avg_int))

        # plot the curve
        if j == 1
            ax1.fill_betweenx(y, x1, x2, color=colors[i], alpha=0.5, label=labels1[i])
        else
            ax1.fill_betweenx(y, x1, x2, color=colors[i], alpha=0.5)
        end

        if i == 1
            ax1.plot(avg_bis, avg_int, ls=linestyles[j], c="k", label=labels2[j])
            ax1.plot(avg_bis, avg_int, ls=linestyles[j], color=colors[i])
        else
            ax1.plot(avg_bis, avg_int, ls=linestyles[j], color=colors[i])
        end
    end
end

# set axes labels and save the figure
# ax1.legend(loc="upper right")
ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=2)
ax1.set_ylim()
ax1.set_xlabel(L"{\rm Relative\ Velocity\ (m\ s}^{-1} {\rm )}")
ax1.set_ylabel(L"{\rm Normalized\ Intensity}")
fig.savefig("/Users/michael/Desktop/line_bisectorz.pdf")
plt.clf(); plt.close()
