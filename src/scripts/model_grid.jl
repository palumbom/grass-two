# using Pkg; Pkg.activate(".")
using CUDA
using GRASS
using Printf
using Random
using Revise
using Statistics
using EchelleCCFs
using BenchmarkTools

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
const plotfile = string(abspath(joinpath(figures, "model_grid.pdf")))

# set up paramaters for spectrum
N = 50
Nt = 1
lines = [5500.0]
depths = [0.75]
templates = ["FeI_5434"]
variability = falses(length(lines))
blueshifts = zeros(length(lines))
resolution = 7e5
seed_rng = true

disk = DiskParams(N=N, Nt=Nt, inclination=60.0, Nsubgrid=40)
spec = SpecParams(lines=lines, depths=depths, variability=variability,
                  blueshifts=blueshifts, templates=templates,
                  resolution=resolution)

# precompute CPU quantities
soldata_cpu = GRASS.SolarData()

# make sure we have a GPU
@assert CUDA.functional()

# get size of sub-tiled grid
Nϕ = disk.N
Nθ_max = maximum(disk.Nθ)
Nsubgrid = disk.Nsubgrid
Nϕ_sub = Nϕ * Nsubgrid
Nθ_sub = maximum(disk.Nθ) * Nsubgrid

# allocate memory on GPU
CUDA.@sync begin
    μs = CUDA.zeros(Float64, Nϕ, Nθ_max)
    wts = CUDA.zeros(Float64, Nϕ, Nθ_max)
    z_rot = CUDA.zeros(Float64, Nϕ, Nθ_max)
    ax_code = CUDA.zeros(Int32, Nϕ, Nθ_max)
end

# convert scalars from disk params to desired precision
ρs = convert(Float64, disk.ρs)
A = convert(Float64, disk.A)
B = convert(Float64, disk.B)
C = convert(Float64, disk.C)
v0 = convert(Float64, disk.v0)
u1 = convert(Float64, disk.u1)
u2 = convert(Float64, disk.u2)

# copy data to GPU
CUDA.@sync begin
    # get observer vectoir and rotation matrix
    O⃗ = CuArray{Float64}(disk.O⃗)
    Nθ = CuArray{Int32}(disk.Nθ)
    R_x = CuArray{Float64}(disk.R_x)
end

# compute geometric parameters, average over subtiles
threads1 = 256
blocks1 = cld(Nϕ * Nθ_max, prod(threads1))
CUDA.@sync @captured @cuda threads=threads1 blocks=blocks1 GRASS.precompute_quantities_gpu!(μs, wts, z_rot, ax_code, Nϕ,
                                                                                            Nθ_max, Nsubgrid, Nθ, R_x, O⃗,
                                                                                            ρs, A, B, C, v0, u1, u2)

# allocations on GPU
if CUDA.functional()
    soldata_gpu = GRASS.GPUSolarData(soldata_cpu)
    gpu_allocs = GRASS.GPUAllocs(spec, disk)

    # precompute GPU quantities
    GRASS.get_keys_and_cbs_gpu!(gpu_allocs, soldata_gpu)
    μs_gpu = Array(μs)
    wts_gpu = Array(wts)
    z_rot_gpu = Array(z_rot)
    ax_codes_gpu = Array(ax_code)
end

# disk coordinates
ϕe = disk.ϕe
ϕc = disk.ϕc
θe = disk.θe
θc = disk.θc
R_x = disk.R_x
R_y = disk.R_y
R_z = disk.R_z

# get color scalar mappable
dat = wts_gpu ./ maximum(wts_gpu)
cmap = plt.cm.inferno

# dat = z_rot_gpu .* 3e8
# cmap = plt.cm.seismic

norm = mpl.colors.Normalize(vmin=minimum(dat), vmax=maximum(dat))
smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

# initialize figure
fig, ax = plt.subplots(1,1, figsize=(8,8))

# loop over grid positions
println("\t>>> Plotting!")
for i in 1:length(ϕe)-1
    lat = range(ϕe[i], ϕe[i+1], length=4)
    for j in 1:disk.Nθ[i]
        lon = range(θe[i,j], θe[i,j+1], length=4)

        border = (([lat[1], lat[2], lat[3], lat[4], lat[4], lat[4], lat[4], lat[3], lat[2], lat[1], lat[1], lat[1]]),
                  ([lon[1], lon[1], lon[1], lon[1], lon[2], lon[3], lon[4], lon[4], lon[4], lon[4], lon[3], lon[2]]))


        out = GRASS.sphere_to_cart.(1.0, border...)
        x = getindex.(out, 1)
        y = getindex.(out, 2)
        z = getindex.(out, 3)

        # rotate it
        for k in eachindex(x)
            x0 = x[k]
            y0 = y[k]
            z0 = z[k]

            x[k] = x0 * R_x[1,1] + y0 * R_x[1,2] + z0 * R_x[1,3]
            y[k] = x0 * R_x[2,1] + y0 * R_x[2,2] + z0 * R_x[2,3]
            z[k] = x0 * R_x[3,1] + y0 * R_x[3,2] + z0 * R_x[3,3]
        end

        idx = z .>= 0
        if any(idx)
            ax.plot(x[idx], y[idx], color="k", lw=1)
            ax.fill(x[idx], y[idx], c=smap.to_rgba(dat[i,j]))
        end
    end
end

# get equator coords
latitude = deg2rad(0.0)
longitude = deg2rad.(range(0.0, 360.0, length=200))
x_eq = []
y_eq = []
z_eq = []
for i in eachindex(longitude)
    out = GRASS.sphere_to_cart.(1.0, latitude, longitude[i])
    x = getindex(out, 1)
    y = getindex(out, 2)
    z = getindex(out, 3)

    x0 = x
    y0 = y
    z0 = z

    x = x0 * R_x[1,1] + y0 * R_x[1,2] + z0 * R_x[1,3]
    y = x0 * R_x[2,1] + y0 * R_x[2,2] + z0 * R_x[2,3]
    z = x0 * R_x[3,1] + y0 * R_x[3,2] + z0 * R_x[3,3]

    push!(x_eq, x)
    push!(y_eq, y)
    push!(z_eq, z)
end

# sort the values on increasing x
idx_eq = sortperm(x_eq)
x_eq = x_eq[idx_eq]
y_eq = y_eq[idx_eq]
z_eq = z_eq[idx_eq]

idx_eq = z_eq .> 0.0

# plot the equator
ax.plot(x_eq[idx_eq], y_eq[idx_eq], color="white", ls="--", zorder=3, alpha=0.75)

# get meridians
latitude = deg2rad.(range(-89.0, 89.0, length=200))
longitude = deg2rad.(range(0.0, 360.0, step=90.0))

for j in eachindex(longitude)
    out = GRASS.sphere_to_cart.(1.0, latitude, longitude[j])

    out = hcat(out...)

    x = out[1,:]
    y = out[2,:]
    z = out[3,:]

    x0 = x
    y0 = y
    z0 = z

    x = x0 .* R_x[1,1] .+ y0 .* R_x[1,2] .+ z0 .* R_x[1,3]
    y = x0 .* R_x[2,1] .+ y0 .* R_x[2,2] .+ z0 .* R_x[2,3]
    z = x0 .* R_x[3,1] .+ y0 .* R_x[3,2] .+ z0 .* R_x[3,3]

    # plot the meridian
    idx = z .> 0.0
    ax.plot(x[idx], y[idx], color="white", ls="--", zorder=3, alpha=0.75)
end

ax.set_xlabel(L"\Delta {\rm x\ [Stellar\ Radii]}")
ax.set_ylabel(L"\Delta {\rm y\ [Stellar\ Radii]}")
ax.set_aspect("equal")
ax.grid(false)
ax.invert_xaxis()
cb = fig.colorbar(smap, ax=ax, fraction=0.1, shrink=0.8)
cb.set_label(L"{\rm Weighted\ Relative\ Intensity}")
fig.savefig(plotfile)
plt.clf(); plt.close();

