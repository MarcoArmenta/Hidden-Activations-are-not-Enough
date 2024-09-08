#!/bin/bash

#SBATCH --account=<ACCOUNT_NAME> #account to charge the calculation
#SBATCH --time=01:00:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=1 #number of CPU requested
#SBATCH --mem-per-cpu=1G #memory requested
#SBATCH --array 8
module load StdEnv/2020 scipy-stack/2023a #load the required module
source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/matrices/
echo "Copying matrices..."
cp -r experiments/$SLURM_ARRAY_TASK_ID/matrices/* $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/matrices/
echo "Matrices ready"
python compute_matrix_statistics.py --default_index $SLURM_ARRAY_TASK_ID --temp_dir $SLURM_TMPDIR
