#!/bin/bash

#SBATCH --account=<ACCOUNT_NAME> #account to charge the calculation
#SBATCH --time=06:00:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=1 #number of CPU requested
#SBATCH --mem-per-cpu=2G #memory requested
#SBATCH --array=7,11

module load StdEnv/2020 scipy-stack/2023a #load the required module
source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python read_results.py --default_index $SLURM_ARRAY_TASK_ID
echo "Grid search finished !!!"
