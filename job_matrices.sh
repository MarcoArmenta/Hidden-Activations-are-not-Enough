#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=01:00:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=40 #number of CPU requested
#SBATCH --mem=2G #memory requested
#SBATCH --array=0-1

module load StdEnv/2020 scipy-stack/2022a #load the required module
source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python matrices_on_epoch.py --chunk=$SLURM_ARRAY_TASK_ID --nb_workers=$SLURM_CPUS_PER_TASK