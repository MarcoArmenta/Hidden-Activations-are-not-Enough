#!/bin/bash

#SBATCH --account=def-ko1 #account to charge the calculation
#SBATCH --node-list=cp3705 # GPU on IQ cluster
#SBATCH --time=02:00:00 #hour:minutes:seconds
#SBATCH --mem=30G #memory requested

module load StdEnv/2020 scipy-stack/2022a #load the required module
source env/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
python training.py --default_index 4