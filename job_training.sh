#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=01:30:00 #hour:minutes:seconds
#SBATCH --mem=40G #memory requested

module load StdEnv/2020 scipy-stack/2023a cuda cudnn #load the required module

mkdir -p $SLURM_TMPDIR/data/MNIST/
mkdir -p $SLURM_TMPDIR/data/FashionMNIST/
echo "Copying datasets..."
cp -r /home/armenta/scratch/MatrixStatistics/data/MNIST/* $SLURM_TMPDIR/data/MNIST/
echo "MNIST ready"
cp -r /home/armenta/scratch/MatrixStatistics/data/FashionMNIST/* $SLURM_TMPDIR/data/FashionMNIST/
echo "Fashion ready"

source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
echo "Environment ready"
python training.py --default_index 13 --data_path $SLURM_TMPDIR/data/
python training.py --default_index 14 --data_path $SLURM_TMPDIR/data/
