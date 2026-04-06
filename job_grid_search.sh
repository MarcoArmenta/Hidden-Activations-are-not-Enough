#!/bin/bash

#SBATCH --account=def-bouchary  #account to charge the calculation
#SBATCH --time=48:00:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=64 #number of CPU requested
#SBATCH --mem-per-cpu=42G #memory requested
#SBATCH --output=slurm_out/H_grid_search_%A.out
#SBATCH --error=slurm_err/H_grid_search_%A.err

EXPERIMENT='alexnet_cifar10'

mkdir -p $PWD/slurm_out
mkdir -p $PWD/slurm_err

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a #load the required module
source env_rorqual/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Copying weights..."
cp experiments/$EXPERIMENT/weights/* $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/

echo "Copying datasets..."
#mkdir -p $SLURM_TMPDIR/data/cifar-10-batches-py/
#cp -r data/cifar-10-batches-py/* $SLURM_TMPDIR/data/cifar-10-batches-py/
mkdir -p $SLURM_TMPDIR/data/cifar-100-python/
cp -r data/cifar-100-python/* $SLURM_TMPDIR/data/cifar-100-python/

echo "Copying matrix statistics..."
mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/
cp experiments/$EXPERIMENT/matrices/matrix_statistics.json $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices/

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

echo "Copying adversarial examples..."
mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/
cp -r experiments/$EXPERIMENT/adversarial_examples/* $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/

echo "Copying rejection level data..."
mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices/
cp -r experiments/$EXPERIMENT/rejection_levels/* $SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/

echo "Decompress..."
unzip $SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices_task_0.zip -d $SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/
unzip $SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices_task_1.zip -d $SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/
unzip $SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices_task_2.zip -d $SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/
unzip $SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices_task_3.zip -d $SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/

echo "All data ready!"

python grid_search.py --nb_workers $SLURM_CPUS_PER_TASK --experiment_name $EXPERIMENT --temp_dir $SLURM_TMPDIR --rej_lev 0
echo "Grid search finished !!!"