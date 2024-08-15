#!/bin/bash

#SBATCH --account=def-ko1 #account to charge the calculation
#SBATCH --time=01:00:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=10 #number of CPU requested
#SBATCH --mem-per-cpu=2G #memory requested
#SBATCH --array=8

module load StdEnv/2020 scipy-stack/2023a #load the required module
source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/weights/
echo "Copying weights..."
cp /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/weights/* $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/weights/
echo "Weights copied to temp directory..."
python generate_matrices.py --nb_workers=$SLURM_CPUS_PER_TASK --default_index=$SLURM_ARRAY_TASK_ID --weights_path=$SLURM_TMPDIR
