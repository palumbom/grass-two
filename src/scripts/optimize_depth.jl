# environment + packages
using Pkg; Pkg.activate(".")
using CSV
using CUDA
using GRASS
using Peaks
using Optim
using LsqFit
using Statistics
using DataFrames
using EchelleCCFs
using EchelleCCFs: λ_air_to_vac, calc_doppler_factor, MeasureRvFromCCFQuadratic as QuadraticFit

# plotting
using LaTeXStrings
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()
mpl.style.use(GRASS.moddir * "fig.mplstyle")
colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
plotdir = string(abspath(joinpath(figures, "iag_comparison")))
datadir = string(abspath(data))

if !isdir(plotdir)
    mkdir(plotdir)
end

# decide whether to use gpu
use_gpu = true
@assert CUDA.functional()

# get data
lp = GRASS.LineProperties(exclude=["CI_5380", "NaI_5896"])
files = GRASS.get_file(lp)
line_names = GRASS.get_name(lp)

# dataframe for optimized depth
df_dep = DataFrame("line" => line_names, "optimized_depth" => zeros(length(line_names)))

# wavelength of line to synthesize/compare to iag
for (i, file) in enumerate(files)
    println(">>> Running " * line_names[i] * "...")

    # get properties from line
    line_name = line_names[i]
    airwav = lp.λrest[i]
    depth = lp.depth[i]

    # get IAG spectrum and normalize it
    wavs_iag0, flux_iag0 = GRASS.read_iag_atlas(isolate=true, airwav=airwav, buffer=1.5)
    flux_iag0 ./= maximum(flux_iag0)

    # convolve IAG spectrum to LARS resolution
    wavs_iag, flux_iag = GRASS.convolve_gauss(wavs_iag0, flux_iag0, new_res=7e5, oversampling=4.0)

   # get depth from IAG spectrum
    buff = 0.12575
    if contains("FeI_5383", line_name)
        buff = 0.3
    elseif contains("FeI_5434", line_name)
        buff = 0.3
    elseif contains("FeI_5382", line_name)
        buff = 0.2
    elseif contains("FeI_5576", line_name)
        buff = 0.25
    elseif contains("CaI_6169.0", line_name)
        buff = 0.25
    elseif contains("CaI_6169.5", line_name)
        buff = 0.1475
    elseif contains("FeI_6170", line_name)
        buff = 0.175
    elseif contains("FeI_6301", line_name)
        buff = 0.25
    elseif contains("FeI_6302", line_name)
        buff = 0.15
    end
    @show buff
    idxl = findfirst(x -> x .>= airwav - buff, wavs_iag)
    idxr = findfirst(x -> x .>= airwav + buff, wavs_iag)
    iag_bot = minimum(view(flux_iag, idxl:idxr))
    iag_depth = 1.0 - iag_bot

    # set up for GRASS spectrum simulation
    function depth_diff(x)
        # set up params from spectrum
        lines = [airwav]
        depths = x
        templates = [file]
        resolution = 7e5
        disk = DiskParams(Nt=5)
        spec = SpecParams(lines=lines, depths=depths, templates=templates, resolution=resolution)

        # simulate the spectrum
        wavs_sim, flux_sim = synthesize_spectra(spec, disk, use_gpu=use_gpu,
                                                verbose=false, seed_rng=true)
        flux_sim = dropdims(mean(flux_sim, dims=2), dims=2)
        return abs((1.0 - minimum(flux_sim)) - iag_depth)
    end

    # get the depth for the simulation
    println("\t>>> Calculating optimized depth for line synthesis...")
    lb = [0.0]
    ub = [1.0]
    optimizer = Fminbox(GradientDescent())
    options = Optim.Options(f_abstol=1e-3, outer_f_abstol=1e-3, iterations=50, outer_iterations=50)
    results = optimize(depth_diff, lb, ub, [iag_depth], optimizer, options)
    sim_depth = results.minimizer[1]
    df_dep[i, "optimized_depth"] = sim_depth
    println("\t>>> Optimized depth = " * string(sim_depth))
end

# write out optimized depth file
CSV.write(joinpath(data, "optimized_depth.csv"), df_dep)
