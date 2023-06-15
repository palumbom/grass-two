# imports
using JLD2
using CUDA
using GRASS
using Printf
using FileIO
using Profile
using Statistics
using BenchmarkTools

# get command line args and output directories
include(joinpath(abspath(@__DIR__), "paths.jl"))
const outfile = joinpath(data, "benchmark.jld2")

function benchmark_cpu(spec::SpecParams, disk::DiskParams)
    lambdas1, outspec1 = synthesize_spectra(spec, disk, verbose=false, seed_rng=true)
    return nothing
end

function benchmark_gpu(spec::SpecParams, disk::DiskParams, precision::DataType)
    lambdas1, outspec1 = synthesize_spectra(spec, disk, verbose=false, seed_rng=true, use_gpu=true, precision=precision)
    return nothing
end

# benchmarking wrapper function
function bmark_everything(b_cpu, b_gpu, b_gpu32, lines, depths; max_cpu=8)
    for i in eachindex(lines)
        @printf(">>> Benchmarking %s of %s\n", i, length(lines))

        # get number of gpu samplings to do
        n_gpu_loops = size(b_gpu, 2)

        # get lines and depths
        lines_i = lines[1:i]
        depths_i = depths[1:i]
        templates_i = repeat(["FeI_5434"], length(lines_i))
        resolution_i = 7e5

        # create spec, disk instances
        spec = SpecParams(lines=lines_i, depths=depths_i, templates=templates_i, resolution=resolution_i)
        disk = DiskParams(N=132, Nt=50)

        # CPU bench loop
        if i <= max_cpu
            @printf("\t>>> Performing CPU bench (N=132, Nt=50): ")
            Profile.clear_malloc_data()
            b_cpu[i] = @belapsed benchmark_cpu($spec, $disk)
            println()
        end

        # GPU bench loop
        @printf("\t>>> Performing GPU double precision bench (N=132, Nt=50): ")
        for j in 1:n_gpu_loops
            Profile.clear_malloc_data()
            CUDA.synchronize()
            b_gpu[i,j] = @belapsed benchmark_gpu($spec, $disk, $Float64)
            CUDA.synchronize()
        end
        println()

        @printf("\t>>> Performing GPU single precision bench (N=132, Nt=50): ")
        for j in 1:n_gpu_loops
            Profile.clear_malloc_data()
            CUDA.synchronize()
            b_gpu32[i,j] = @belapsed benchmark_gpu($spec, $disk, $Float32)
            CUDA.synchronize()
        end
        println()
    end
    return nothing
end

function main()
    # line parameters
    nlines = 24
    lines = range(5434.5, step=5.0, length=nlines)
    depths = repeat([0.75], length(lines))

    # calculate number of resolution elements per spectrum
    n_res = similar(lines)
    n_lam = similar(lines)
    for i in eachindex(lines)
        # calculate wavelength coverage
        lines1 = lines[1:i]
        coverage = (minimum(lines1) - 0.75, maximum(lines1) + 0.75)

        # generate Delta ln lambda
        Δlnλ = (1.0 / 7e5)
        lnλs = range(log(coverage[1]), log(coverage[2]), step=Δlnλ)
        lambdas = exp.(lnλs)
        n_res[i] = length(lambdas)
        n_lam[i] = last(lambdas) - first(lambdas)
    end

    # allocate memory for benchmark results and run it
    n_gpu_loops = 16
    max_cpu = minimum([18, length(lines)])
    b_cpu = similar(lines)
    b_gpu = zeros(length(lines), n_gpu_loops)
    b_gpu32 = zeros(length(lines), n_gpu_loops)
    bmark_everything(b_cpu, b_gpu, b_gpu32, lines, depths, max_cpu=max_cpu)

    # write results to disk
    save(outfile,
         "max_cpu", max_cpu,
         "nlines", nlines,
         "n_res", n_res,
         "n_lam", n_lam,
         "b_cpu", b_cpu,
         "b_gpu", b_gpu,
         "b_gpu32", b_gpu32)
    return nothing
end

@assert CUDA.functional()
main()
