using CUDA
using GRASS
using Printf
using Random
using Revise
using Statistics
using LinearAlgebra
using EchelleCCFs
using BenchmarkTools

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
ConnectionPatch = mpl.patches.ConnectionPatch

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
const plotfile = string(abspath(joinpath(figures, "model_grid.pdf")))

# make sure we have a GPU
@assert CUDA.functional()

# set up paramaters for spectrum
# N = 197
# Nt = 1

# Nsubgrid = range(4, 80, step=2)

# v_out = []
# w_out = []

# for i in Nsubgrid
#     disk = DiskParams(N=N, Nt=Nt, inclination=89.99, Nsubgrid=i)

#     # get size of sub-tiled grid
#     Nϕ = disk.N
#     Nθ_max = maximum(disk.Nθ)
#     Nϕ_sub = Nϕ * Nsubgrid
#     Nθ_sub = maximum(disk.Nθ) * Nsubgrid

#     # allocate memory on GPU
#     CUDA.@sync begin
#         μs = CUDA.zeros(Float64, Nϕ, Nθ_max)
#         wts = CUDA.zeros(Float64, Nϕ, Nθ_max)
#         z_rot = CUDA.zeros(Float64, Nϕ, Nθ_max)
#         ax_code = CUDA.zeros(Int32, Nϕ, Nθ_max)
#     end

#     # convert scalars from disk params to desired precision
#     ρs = convert(Float64, disk.ρs)
#     A = convert(Float64, disk.A)
#     B = convert(Float64, disk.B)
#     C = convert(Float64, disk.C)
#     v0 = convert(Float64, disk.v0)
#     u1 = convert(Float64, disk.u1)
#     u2 = convert(Float64, disk.u2)

#     # copy data to GPU
#     CUDA.@sync begin
#         # get observer vectoir and rotation matrix
#         O⃗ = CuArray{Float64}(disk.O⃗)
#         Nθ = CuArray{Int32}(disk.Nθ)
#         R_x = CuArray{Float64}(disk.R_x)
#     end

#     # compute geometric parameters, average over subtiles
#     # println("\t>>> Computing geometry...")
#     threads1 = 256
#     blocks1 = cld(Nϕ * Nθ_max, prod(threads1))
#     CUDA.@sync @captured @cuda threads=threads1 blocks=blocks1 GRASS.precompute_quantities_gpu!(μs, wts, z_rot, ax_code, Nϕ,
#                                                                                                 Nθ_max, i, Nθ, R_x, O⃗,
#                                                                                                 ρs, A, B, C, v0, u1, u2)

#     # copy arrays from gpu
#     μs_gpu = Array(μs)
#     wts_gpu = Array(wts)
#     z_rot_gpu = Array(z_rot) .* GRASS.c_ms
#     ax_codes_gpu = Array(ax_code)

#     push!(v_out, sum(z_rot_gpu))
#     push!(w_out, sum(wts_gpu))
# end

# plt.scatter(Nsubgrid, v_out)
# plt.xlabel("Nsubgrid")
# plt.ylabel("sum(v_rot)")
# plt.show()

# now compare
# set up paramaters for spectrum
lines = [5500.0]
depths = [0.75]
templates = ["FeI_5434"]
blueshifts = zeros(length(lines))
resolution = 7e5
buffer = 1.5

# create composite types
disk1 = DiskParams(Nt=1, Nsubgrid=1600, inclination=90.0)
disk2 = DiskParams(Nt=1, Nsubgrid=40, inclination=90.0)
spec = SpecParams(lines=lines, depths=depths, templates=templates,
                  blueshifts=blueshifts, resolution=resolution, buffer=buffer)

println(">>> Performing GPU synthesis (Nsubgrid = 240)...")
wavs_gpu1, flux_gpu1 = synthesize_spectra(spec, disk1, seed_rng=true, verbose=true, use_gpu=true)

println(">>> Performing GPU synthesis (Nsubgrid = 40)...")
wavs_gpu2, flux_gpu2 = synthesize_spectra(spec, disk2, seed_rng=true, verbose=true, use_gpu=true)

# slice output
flux_gpu1 = dropdims(mean(flux_gpu1, dims=2), dims=2)
flux_gpu2 = dropdims(mean(flux_gpu2, dims=2), dims=2)

# get flux residuals
resids = flux_gpu1 .- flux_gpu2

# compute velocities
v_grid1, ccf1 = calc_ccf(wavs_gpu1, flux_gpu1, spec, normalize=true, mask_type=EchelleCCFs.GaussianCCFMask)
rvs_gpu1, sigs_gpu1 = calc_rvs_from_ccf(v_grid1, ccf1, frac_of_width_to_fit=0.5)

v_grid2, ccf2 = calc_ccf(wavs_gpu2, flux_gpu2, spec, normalize=true, mask_type=EchelleCCFs.GaussianCCFMask)
rvs_gpu2, sigs_gpu2 = calc_rvs_from_ccf(v_grid2, ccf2, frac_of_width_to_fit=0.5)

# get velocity residuals
v_resids = rvs_gpu1 - rvs_gpu2

# report some diagnostics
println()
@show maximum(abs.(resids))

println()
@show maximum(abs.(v_resids))
