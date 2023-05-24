using Pkg; Pkg.activate(".")
using CSV
using CUDA
using GRASS
using Printf
using Revise
using DataFrames
using Statistics
using EchelleCCFs
using BenchmarkTools

using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")

# set up paramaters for spectrum
N = 132
Nt = 10
lines = [5250.6]#, 5434.5, 6173.4]
depths = [0.8]#, 0.9, 0.6]
geffs = [0.0]#, 0.0, 0.0]
templates = ["FeI_5250.6"]#, "FeI_5434", "FeI_6173"]
variability = trues(length(lines))
resolution = 7e5
seed_rng = true
use_gpu = CUDA.functional()

disk = DiskParams(N=N, Nt=Nt)
spec1 = SpecParams(lines=lines, depths=depths, variability=variability,
                   geffs=geffs, templates=templates, resolution=resolution)
wavs, flux = synthesize_spectra(spec1, disk, seed_rng=seed_rng, verbose=true, use_gpu=use_gpu)

plt.plot(wavs, flux)
plt.show()
