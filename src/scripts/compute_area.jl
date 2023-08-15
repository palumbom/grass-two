using GRASS
using Statistics

sq_arc_to_sq_rad = 2.35e-11
area_dc = (π * 30.0^2) * sq_arc_to_sq_rad
area_095_07 = (π * 40.0 * 10.0) * sq_arc_to_sq_rad
area_06 = (π * 40.0 * 20.0) * sq_arc_to_sq_rad
area_05_03 = (π * 40.0 * 30.0) * sq_arc_to_sq_rad

areas = [area_05_03, area_05_03, area_05_03,
         area_06,
         area_095_07, area_095_07, area_095_07, area_095_07, area_095_07,
         area_dc]
mus = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]


function calc_average_area(N)
    u1 = 0.4
    u2 = 0.26
    area = 0.0
    sum_wts = 0.0
    for i in range(-1.0, 1.0, N)
        for j in range(-1.0, 1.0, N)
            r2 = i^2.0 + j^2.0
            r2 >= 1.0 && continue

            mu = 1.0 - r2

            idx = GRASS.searchsortednearest(mu, mus)
            wts = GRASS.quad_limb_darkening(mu,u1,u2)
            sum_wts += wts
            area += areas[idx] * wts
        end
    end
    return area, area/sum_wts
end

area, avg_area = calc_average_area(1024)

avg_area_arcsec2 = avg_area / sq_arc_to_sq_rad
println()
@show sqrt(avg_area_arcsec2 / π)
println()


# set up paramaters for spectrum
N = 197
Nt = 1
lines = [5500.0]
depths = [0.75]
templates = ["FeI_5250.6"]
variability = trues(length(lines))
blueshifts = zeros(length(lines))
resolution = 7e5
seed_rng = true

disk = DiskParams(N=N, Nt=Nt, inclination=90.0, Nsubgrid=16)
spec = SpecParams(lines=lines, depths=depths, variability=variability,
                  blueshifts=blueshifts, templates=templates,
                  resolution=resolution)

# allocations on CPU
soldata_cpu = GRASS.SolarData()
wsp = GRASS.SynthWorkspace(disk)

# precompute CPU quantities
GRASS.precompute_quantities!(wsp, disk)
μs_cpu = wsp.μs
cbs_cpu = wsp.cbs
wts_cpu = wsp.wts
z_rot_cpu = wsp.z_rot
ax_codes_cpu = wsp.ax_codes


idx = wsp.μs .> 0.9
mean_dA = sum(wsp.dA[idx] .* wsp.ld[idx])/sum(wsp.ld[idx])
mean_dA *= (deg2rad(0.5)^2)
mean_dA /= sq_arc_to_sq_rad

@show N
@show sqrt(mean_dA);
println()
