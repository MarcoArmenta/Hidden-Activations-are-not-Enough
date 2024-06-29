#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=02:00:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=30 #number of CPU requested
#SBATCH --mem-per-cpu=30G #memory requested

module load StdEnv/2020 scipy-stack/2023a #load the required module
source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python generate_matrices.py --nb_workers=$SLURM_CPUS_PER_TASK --default_index 0