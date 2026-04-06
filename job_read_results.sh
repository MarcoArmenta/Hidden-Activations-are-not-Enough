#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=01:00:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=4 #number of CPU requested
#SBATCH --mem-per-cpu=50G #memory requested
#SBATCH --mem=200G
#SBATCH --output=slurm_out/I_read_results_%A.out
#SBATCH --error=slurm_err/I_read_results_%A.err

EXPERIMENT='alexnet_cifar10'

mkdir -p $PWD/slurm_out
mkdir -p $PWD/slurm_err

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a #load the required module
source env_rorqual/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Weights
mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Copying weights..."
cp experiments/$EXPERIMENT/weights/* $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/

# Dataset
#mkdir -p $SLURM_TMPDIR/data/cifar-100-python/
mkdir -p $SLURM_TMPDIR/data/cifar-10-batches-py/
echo "Copying datasets..."
#cp -r data/cifar-100-python/* $SLURM_TMPDIR/data/cifar-100-python/
cp -r data/cifar-10-batches-py/* $SLURM_TMPDIR/data/cifar-10-batches-py/

# Matrix Statistics
echo "Copying matrix statistics..."
mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/
cp experiments/$EXPERIMENT/matrices/matrix_statistics.json $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/

# Adversarial Examples
echo "Copying adversarial examples..."
mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/
cp -r experiments/$EXPERIMENT/adversarial_examples/* $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/

# Adversarial Matrices
echo "Copying adversarial matrices..."
mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_matrices/
cp experiments/$EXPERIMENT/adv_matrices_task_0.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
cp experiments/$EXPERIMENT/adv_matrices_task_1.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
cp experiments/$EXPERIMENT/adv_matrices_task_2.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
cp experiments/$EXPERIMENT/adv_matrices_task_3.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
echo "Decompress..."
unzip experiments/$EXPERIMENT/adv_matrices_task_0.zip -d $SLURM_TMPDIR/experiments/$EXPERIMENT/
unzip experiments/$EXPERIMENT/adv_matrices_task_1.zip -d $SLURM_TMPDIR/experiments/$EXPERIMENT/
unzip experiments/$EXPERIMENT/adv_matrices_task_2.zip -d $SLURM_TMPDIR/experiments/$EXPERIMENT/
unzip experiments/$EXPERIMENT/adv_matrices_task_3.zip -d $SLURM_TMPDIR/experiments/$EXPERIMENT/

# Preprocess Cache
echo "Geting cache... It is OK if there is no cache files saved."
mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/preprocessed/
cp -r experiments/$EXPERIMENT/preprocessed/* $SLURM_TMPDIR/experiments/$EXPERIMENT/preprocessed/

# Counds-per-attack
echo "Geting cache... It is OK if there is no cache files saved."
mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/counts_per_attack/
cp -r experiments/$EXPERIMENT/counts_per_attack/* $SLURM_TMPDIR/experiments/$EXPERIMENT/counts_per_attack/


python read_results.py --experiment_name $EXPERIMENT --temp_dir $SLURM_TMPDIR
echo "Grid search finished !!!"
