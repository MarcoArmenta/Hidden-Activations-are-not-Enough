#!/bin/bash

#SBATCH --account=def-bruestle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --mem=6 G


source env/bin/activate
python scratch-train.py