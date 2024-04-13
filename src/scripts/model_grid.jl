# using Pkg; Pkg.activate(".")
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
N = 28
Nt = 1
disk = DiskParams(N=N, Nt=Nt, inclination=60.0, Nsubgrid=40)

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
println("\t>>> Computing geometry...")
threads1 = 256
blocks1 = cld(Nϕ * Nθ_max, prod(threads1))
CUDA.@sync @captured @cuda threads=threads1 blocks=blocks1 GRASS.precompute_quantities_gpu!(μs, wts, z_rot, ax_code, Nϕ,
                                                                                            Nθ_max, Nsubgrid, Nθ, R_x, O⃗,
                                                                                            ρs, A, B, C, v0, u1, u2)

# copy arrays from gpu
μs_gpu = Array(μs)
wts_gpu = Array(wts)
z_rot_gpu = Array(z_rot)
ax_codes_gpu = Array(ax_code)

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
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

# plot circle background to smooth jagged edges
circle1 = mpl.patches.Circle((0, 0), 1.01, color="k", zorder=0)
ax1.add_patch(circle1)

# coords for zoom in cell
ϕidx = 12
θidx = 51

# loop over grid positions
println("\t>>> Plotting!")
for i in 1:length(ϕe)-1
    lat = range(ϕe[i], ϕe[i+1], length=4)
    for j in 1:disk.Nθ[i]
        lon = range(θe[i,j], θe[i,j+1], length=4)

        border = (([lat[1], lat[2], lat[3], lat[4], lat[4], lat[4], lat[4], lat[3], lat[2], lat[1], lat[1], lat[1], lat[1]]),
                  ([lon[1], lon[1], lon[1], lon[1], lon[2], lon[3], lon[4], lon[4], lon[4], lon[4], lon[3], lon[2], lon[1]]))


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

        if (i == ϕidx) & (j == θidx)
            color="tab:blue"
            lw = 3
            zorder=2

            global dat_to_save = dat[i,j]
        else
            color="k"
            lw = 1
            zorder=1
        end

        if any(idx)
            ax1.plot(x[idx], y[idx], color=color, lw=lw, zorder=zorder)
            ax1.fill(x[idx], y[idx], c=smap.to_rgba(dat[i,j]), zorder=0)
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
ax1.plot(x_eq[idx_eq], y_eq[idx_eq], color="white", ls="--", zorder=3, alpha=0.75)

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
    ax1.plot(x[idx], y[idx], color="white", ls="--", zorder=3, alpha=0.75)
end

# plot zoom in on cell
lat_cen = ϕc[ϕidx]
lon_cen = θc[ϕidx,θidx]

x, y, z = GRASS.sphere_to_cart.(1.0, lat_cen, lon_cen)
x0 = x
y0 = y
z0 = z
x = x0 * R_x[1,1] + y0 * R_x[1,2] + z0 * R_x[1,3]
y = x0 * R_x[2,1] + y0 * R_x[2,2] + z0 * R_x[2,3]
z = x0 * R_x[3,1] + y0 * R_x[3,2] + z0 * R_x[3,3]


lat = range(ϕe[ϕidx], ϕe[ϕidx+1], length=5)
lon = range(θe[ϕidx,θidx], θe[ϕidx,θidx+1], length=5)

border = (([lat[1], lat[2], lat[3], lat[4], lat[5], lat[5], lat[5], lat[5], lat[5], lat[4], lat[3], lat[2], lat[1], lat[1], lat[1], lat[1], lat[1]]),
         ([lon[1], lon[1], lon[1], lon[1], lon[1], lon[2], lon[3], lon[4], lon[5], lon[5], lon[5], lon[5], lon[5], lon[4], lon[3], lon[2], lon[1]]))

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
ax2.plot(x, y, color="tab:blue", lw=6)


# set limits
buffer = 0.018
dx = (maximum(x) + buffer) - (minimum(x) - buffer)
dy = (maximum(y) + buffer) - (minimum(y) - buffer)

if dx == dy
    ax2.set_xlim(minimum(x) - buffer, maximum(x) + buffer)
    ax2.set_ylim(minimum(y) - buffer, maximum(y) + buffer)
elseif dy > dx
    delt = dy - dx
    ax2.set_xlim(minimum(x) - buffer - delt/2, maximum(x) + buffer + delt/2)
    ax2.set_ylim(minimum(y) - buffer, maximum(y) + buffer)
else
    delt = dx - dy
    ax2.set_xlim(minimum(x) - buffer, maximum(x) + buffer)
    ax2.set_ylim(minimum(y) - buffer - delt/2, maximum(y) + buffer + delt/2)
end

# plot cells around focused cell
for i in ϕidx-1:ϕidx+1
    # find the starting index
    idx_idk = findfirst(x -> x .>= θc[ϕidx, θidx], θc[i, :])
    for j in idx_idk-2:idx_idk+2
        lat2 = range(ϕe[i], ϕe[i+1], length=5)
        lon2 = range(θe[i,j], θe[i,j+1], length=5)

        border = (([lat2[1], lat2[2], lat2[3], lat2[4], lat2[5], lat2[5], lat2[5], lat2[5], lat2[5], lat2[4], lat2[3], lat2[2], lat2[1], lat2[1], lat2[1], lat2[1], lat2[1]]),
                 ([lon2[1], lon2[1], lon2[1], lon2[1], lon2[1], lon2[2], lon2[3], lon2[4], lon2[5], lon2[5], lon2[5], lon2[5], lon2[5], lon2[4], lon2[3], lon2[2], lon2[1]]))

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
        ax2.plot(x, y, color="k", lw=1, alpha=0.33, zorder=0)

    end
end

# now get the edgges of the sub-tiles
for i in eachindex(lat)
    for j in eachindex(lon)
        if i == length(lat)
            continue
        elseif j == length(lon)
            continue
        end
        lat2 = range(lat[i], lat[i+1], length=5)
        lon2 = range(lon[j], lon[j+1], length=5)

        border = (([lat2[1], lat2[2], lat2[3], lat2[4], lat2[5], lat2[5], lat2[5], lat2[5], lat2[5], lat2[4], lat2[3], lat2[2], lat2[1], lat2[1], lat2[1], lat2[1], lat2[1]]),
                 ([lon2[1], lon2[1], lon2[1], lon2[1], lon2[1], lon2[2], lon2[3], lon2[4], lon2[5], lon2[5], lon2[5], lon2[5], lon2[5], lon2[4], lon2[3], lon2[2], lon2[1]]))

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
        ax2.plot(x[1:5], y[1:5], color="k", lw=1.5, ls="--")
        ax2.plot(x[5:9], y[5:9], color="k", lw=1.5, ls="--")

        if j == length(lon) - 1
            ax2.plot(x[9:13], y[9:13], color="k", lw=1.5, ls="--")
        end

        if i == 1
            ax2.plot(x[13:end], y[13:end], color="k", lw=1.5, ls="--")
        end


        # now do a scatter for grid centers
        lat3 = (first(lat2) + last(lat2)) / 2.0
        lon3 = (first(lon2) + last(lon2)) / 2.0
        x0, y0, z0 = GRASS.sphere_to_cart.(1.0, lat3, lon3)
        xcen = x0 * R_x[1,1] + y0 * R_x[1,2] + z0 * R_x[1,3]
        ycen = x0 * R_x[2,1] + y0 * R_x[2,2] + z0 * R_x[2,3]
        zcen = x0 * R_x[3,1] + y0 * R_x[3,2] + z0 * R_x[3,3]

        # get the mu
        μ = GRASS.calc_mu([xcen,ycen,zcen], [0.0,0.0,1e6])
        ld = GRASS.quad_limb_darkening(μ, 0.4, 0.26)

        # get the area of the subtile
        dA = sin(π/2 - lat3) * (maximum(lat2) - minimum(lat2)) * (maximum(lon2) - minimum(lon2))
        dA_proj = dA * μ

        tdat = ld * dA_proj * 16 / maximum(wts_gpu)
        ax2.fill(x, y, color=smap.to_rgba(tdat))
        ax2.scatter([xcen], [ycen], c="k", s=60)
        # ax2.scatter([xcen], [ycen], color=smap.to_rgba(tdat), s=100)
    end
end

# plot a square corresponding to zoom-in window
square_x = ax2.get_xlim()
square_y = ax2.get_ylim()

square_x_coords = [square_x[1], square_x[1], square_x[2], square_x[2], square_x[1]]
square_y_coords = [square_y[1], square_y[2], square_y[2], square_y[1], square_y[1]]

ax1.plot(square_x_coords, square_y_coords, c="white", lw=1.5)
ax1.plot(square_x_coords, square_y_coords, c="k", lw=1)

# draw lines connecting zoom-in
xy1_lr = (square_x[1], square_y[1])
xy1_ur = (square_x[1], square_y[2])

xy2_ll = (square_x[2], square_y[1])
xy2_ul = (square_x[2], square_y[2])

l1 = ConnectionPatch(xyA=xy1_lr, xyB=xy2_ll, coordsA=ax1.transData, coordsB=ax2.transData, color="white", lw=1.5)
l2 = ConnectionPatch(xyA=xy1_ur, xyB=xy2_ul, coordsA=ax1.transData, coordsB=ax2.transData, color="white", lw=1.5)

fig.add_artist(l1)
fig.add_artist(l2)

l3 = ConnectionPatch(xyA=xy1_lr, xyB=xy2_ll, coordsA=ax1.transData, coordsB=ax2.transData, color="k", lw=1)
l4 = ConnectionPatch(xyA=xy1_ur, xyB=xy2_ul, coordsA=ax1.transData, coordsB=ax2.transData, color="k", lw=1)

fig.add_artist(l3)
fig.add_artist(l4)


ax1.set_xlabel(L"\Delta x\ {\rm [Stellar\ Radii]}", fontsize=14)
ax1.set_ylabel(L"\Delta y\ {\rm [Stellar\ Radii]}", fontsize=14)
ax2.set_xlabel(L"\Delta x\ {\rm [Stellar\ Radii]}", fontsize=14)
ax2.set_ylabel(L"\Delta y\ {\rm [Stellar\ Radii]}", fontsize=14)
ax1.tick_params(axis="both", labelsize=14)
ax2.tick_params(axis="both", labelsize=14)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax1.set_aspect("equal")
ax2.set_aspect("equal")
ax1.grid(false)
ax2.grid(false)
ax1.invert_xaxis()
ax2.invert_xaxis()

fig.tight_layout()

cax = fig.add_axes([ax2.get_position().x1+0.085,ax1.get_position().y0,0.02,ax1.get_position().height])
cb = fig.colorbar(smap, cax=cax)
cb.set_label(L"{\rm Relative\ Weight}", fontsize=14)
fig.savefig(plotfile)
# plt.show()
plt.clf(); plt.close();

