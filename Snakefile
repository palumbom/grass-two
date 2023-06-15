import os

os.environ["JULIA_PROJECT"] = "."
os.environ["JULIA_NUM_THREADS"] = "auto"

envvars:
    "JULIA_PROJECT",
    "JULIA_NUM_THREADS",

rule julia_manifest:
    input: "Project.toml"
    output: "Manifest.toml"
    shell: "julia -e 'using Pkg; Pkg.instantiate()'"

rule gpu_accuracy_data:
    input: "Manifest.toml"
    output: "src/data/gpu_accuracy.jld2"
    script: "src/scripts/gpu_accuracy_data.jl"

rule gpu_accuracy_data:
    input:
        "Manifest.toml",
        data="src/data/gpu_accuracy.jld2",
        # ^ Can name these for easier reference in the script.
        # https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#julia
    output: "src/tex/figures/gpu_accuracy.pdf"
    script: "src/scripts/gpu_accuracy_plot.jl"
