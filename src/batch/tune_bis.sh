#!/bin/bash
#SBATCH -A dfc13_mri
#SBATCH -p mgc-mri
#SBATCH --array=1-22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --time=48:00:00
#SBATCH --job-name=tunebis
#SBATCH --chdir=/storage/home/mlp95/work/grass-two
#SBATCH --output=/storage/home/mlp95/work/logs/tunebis_%A-%a.out

echo "About to start: $SLURM_JOB_NAME"
date
echo "Job id: $SLURM_JOBID"
echo "About to change into $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR
echo "About to start Julia"
julia src/scripts/tune_bis.jl $SLURM_ARRAY_TASK_ID
echo "Julia exited"
date
