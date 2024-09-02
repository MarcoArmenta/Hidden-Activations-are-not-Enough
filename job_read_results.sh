#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=06:00:00 #hour:minutes:seconds
#SBATCH --array=7,11

module load CCEnv arch/avx512 StdEnv/2020
module load python/3.9
source env/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)

echo "All data ready!"

python read_results.py --default_index $SLURM_ARRAY_TASK_ID
echo "Grid search finished !!!"