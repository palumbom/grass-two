#!/bin/bash
#SBATCH -A dfc13_mri
#SBATCH -p mgc-mri
#SBATCH --array=1-22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=64GB
#SBATCH --time=72:00:00
#SBATCH --job-name=noiseplot
#SBATCH --chdir=/storage/home/mlp95/work/grass-two
#SBATCH --output=/storage/home/mlp95/work/logs/noiseproc_%A-%a.out

echo "About to start: $SLURM_JOB_NAME"
date
echo "Job id: $SLURM_JOBID"
echo "About to change into $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR
echo "About to start Julia"
julia --threads auto src/scripts/noise_scaling_process.jl $SLURM_ARRAY_TASK_ID
julia src/scripts/noise_scaling_plot.jl $SLURM_ARRAY_TASK_ID
echo "Julia exited"
date
