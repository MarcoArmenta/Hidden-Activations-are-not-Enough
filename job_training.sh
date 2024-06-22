#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=01:00:00 #hour:minutes:seconds
#SBATCH --mem=30G #memory requested

module load StdEnv/2020 scipy-stack/2023a #load the required module
source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
python training.py --default_index 0