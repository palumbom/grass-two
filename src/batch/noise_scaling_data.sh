#!/bin/bash
#SBATCH -A dfc13_mri
#SBATCH -p mgc-mri
#SBATCH --array=1-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=48:00:00
#SBATCH --job-name=noisedata
#SBATCH --chdir=/storage/home/mlp95/work/grass-two
#SBATCH --output=/storage/home/mlp95/work/logs/noisedata_%A-%a.out

echo "About to start: $SLURM_JOB_NAME"
date
echo "Job id: $SLURM_JOBID"
echo "About to change into $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR
echo "About to start Julia"
srun julia src/scripts/noise_scaling_data.jl $SLURM_ARRAY_TASK_ID
srun julia src/scripts/noise_scaling_plot.jl $SLURM_ARRAY_TASK_ID
echo "Julia exited"
date
