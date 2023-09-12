#!/bin/bash
#SBATCH -A dfc13_mri
#SBATCH -p mgc-mri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=1:00:00
#SBATCH --job-name=gpu_accuracy
#SBATCH --chdir=/storage/home/mlp95/work/grass-two
#SBATCH --output=/storage/home/mlp95/work/logs/accuracy.%j.out

echo "About to start: $SLURM_JOB_NAME"
date
echo "Job id: $SLURM_JOBID"
echo "About to change into $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR
echo "About to start Julia"
julia -e 'using Pkg; Pkg.activate("."); using CUDA; CUDA.set_runtime_version!("local")'
julia src/scripts/gpu_accuracy_data.jl
julia src/scripts/gpu_accuracy_plot.jl
echo "Julia exited"
date
