#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=02:00:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=1 #number of CPU requested
#SBATCH --mem-per-cpu=8G #memory requested
#SBATCH --array 0-16

module load StdEnv/2020 scipy-stack/2023a #load the required module
source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/matrices/
echo "Copying matrices..."
cp -r /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/matrices/* $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/matrices/
echo "Matrices copied to temp directory..."
python compute_matrix_statistics.py --default_index $SLURM_ARRAY_TASK_ID --matrices_path $SLURM_TMPDIR
