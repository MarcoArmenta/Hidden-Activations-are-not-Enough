#!/bin/bash

#SBATCH --account=def-ko1 #account to charge the calculation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=05:00:00 #hour:minutes:seconds
#SBATCH --mem-per-cpu=10G #memory requested
#SBATCH --array=4

module load StdEnv/2020 scipy-stack/2022a #load the required module
source env/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python matrices_on_epoch.py --default_index=$SLURM_ARRAY_TASK_ID --nb_workers=$SLURM_CPUS_PER_TASK
python generate_adversarial_examples.py --default_index=$SLURM_ARRAY_TASK_ID --nb_workers=$SLURM_CPUS_PER_TASK
python grid_search.py --default_index=$SLURM_ARRAY_TASK_ID --nb_workers=$SLURM_CPUS_PER_TASK --output_file grid_search_inde
python read_results.py --default_index=$SLURM_ARRAY_TASK_ID