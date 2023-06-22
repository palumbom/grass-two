using BenchmarkTools
using Printf
using Revise
using GRASS

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
using PyCall; animation = pyimport("matplotlib.animation")
mpl.style.use(GRASS.moddir * "fig.mplstyle")

# set Desktop directory
run, plot = parse_args(ARGS)
grassdir, plotdir, datadir = check_plot_dirs()

# set up stuff for lines
N = 132
Nt = 40
lines = [5434.5]
templates = ["FeI_5434"]
depths = [0.8]
variability = [true]
resolution = 700000.0
disk = DiskParams(N=N, Nt=Nt)
spec = SpecParams(lines=lines, depths=depths, variability=variability,
                  templates=templates, resolution=resolution)

# synthesize spectra
lambdas1, outspec1 = synthesize_spectra(spec, disk, seed_rng=true, verbose=true, use_gpu=true)

# loop over SNRs
nloop = 2
A = 0.8
snr = range(50, 5050, step=500.0)
rms = zeros(length(snr))
idealized = zeros(length(snr))
for i in eachindex(snr)
    for j in 1:nloop
        # degrade the resolution
        outspec2 = GRASS.add_noise(outspec1, snr[i])

        # calculate the rms
        v_grid, ccf = calc_ccf(lambdas1, outspec2, spec, normalize=true)
        rvs, sigs = calc_rvs_from_ccf(v_grid, ccf)
        rms[i] += calc_rms(rvs)

        # get idealized result
        mid = (1.0 - (1.0 - depths[1])) / 2.0
        bot = argmin(outspec2[:,1])
        ind1 = GRASS.searchsortednearest(outspec2[:,1][1:bot], mid)
        ind2 = GRASS.searchsortednearest(outspec2[:,1][bot:end], mid)
        fwhm = lambdas1[ind2] - lambdas1[ind1]
        fwhm = (fwhm / lines[1]) * 3e8
        idealized[i] += (A * fwhm / (snr[i] * sqrt(ind2-ind1)))
    end
end

# take the average
rms ./= nloop
idealized ./= nloop

# now plot the result
fig, ax = plt.subplots()
ax.plot(snr, rms, color="tab:blue", label="Granulation + Photon")
ax.plot(snr, idealized, color="tab:orange", label="Photon noise only")#, A = " * string(A))
ax.axhline(0.3, linestyle="--", alpha=0.5, c="k")
ax.set_xlabel("SNR per pixel")
ax.set_ylabel("RMS RV (m/s)")
# ax.set_xscale("log", base=10)
# ax.set_yscale("log", base=10)
ax.legend()
fig.savefig(plotdir * "photon_noise.pdf")
plt.show()
plt.clf(); plt.close()

# idx1 = findfirst(x -> x .< 0.5, rms)
# idx2 = findfirst(x -> x .< 0.3, rms)
# idx3 = findfirst(x -> x .< 0.2, rms)

# println("SNR needed for 50 cm/s --> " * string(snr[idx1]))
# println("SNR needed for 30 cm/s --> " * string(snr[idx2]))
# println("SNR needed for 20 cm/s --> " * string(snr[idx3]))
