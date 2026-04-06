#!/bin/bash

#SBATCH --account=def-ko1 #account to charge the calculation
#SBATCH --time=01:00:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G #memory requested
#SBATCH --output=slurm_out/E_mat_stats_%A.out
#SBATCH --error=slurm_err/E_mat_stats_%A.err

mkdir -p $PWD/slurm_out
mkdir -p $PWD/slurm_err

EXPERIMENT="alexnet_cifar10"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env_rorqual/bin/activate

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/
cp experiments/$EXPERIMENT/matrices_task_0.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
cp experiments/$EXPERIMENT/matrices_task_1.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
cp experiments/$EXPERIMENT/matrices_task_2.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
cp experiments/$EXPERIMENT/matrices_task_3.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/

echo "Zip files ready"

mkdir -p experiments/$EXPERIMENT/matrices/
unzip "$SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_0.zip" -d "$SLURM_TMPDIR/experiments/$EXPERIMENT/"
unzip "$SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_1.zip" -d "$SLURM_TMPDIR/experiments/$EXPERIMENT/"
unzip "$SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_2.zip" -d "$SLURM_TMPDIR/experiments/$EXPERIMENT/"
unzip "$SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_3.zip" -d "$SLURM_TMPDIR/experiments/$EXPERIMENT/"

echo "Matrices unzipped to..."
ls $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/

#cp -r experiments/$EXPERIMENT/matrices/* $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/
echo "Matrices ready. Computing statistics."
python compute_matrix_statistics.py --experiment_name $EXPERIMENT --temp_dir $SLURM_TMPDIR