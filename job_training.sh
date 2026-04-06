#!/bin/bash

#SBATCH --account=def-assem
#SBATCH --gpus=a100_2g.10gb:1
#SBATCH --cpus-per-task=3
#SBATCH --time=00:30:00
#SBATCH --mem=31G
#SBATCH --output=slurm_out/A_train_%A.out
#SBATCH --error=slurm_err/A_train_%A.err

mkdir $SLURM_SUBMIT_DIR/slurm_err
mkdir $SLURM_SUBMIT_DIR/slurm_out

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
mkdir -p $SLURM_TMPDIR/data/cifar-100-python/
#mkdir -p $SLURM_TMPDIR/data/cifar-10-batches-py/
echo "Copying datasets..."
cp -r data/cifar-100-python/* $SLURM_TMPDIR/data/cifar-100-python/
#cp -r data/cifar-10-batches-py/* $SLURM_TMPDIR/data/cifar-10-batches-py/

echo "data ready..."
source env_narval/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)

#export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_USE_CUDA_DSA=1

# Hyperparamter tuning
python scratch-train.py --temp_dir $SLURM_TMPDIR --model vgg --dataset cifar100

# One training saving weights at checkpoints/ hyperparameters should be in constants/constants.py
#python training.py --experiment_name vgg_cifar100 --temp_dir $SLURM_TMPDIR