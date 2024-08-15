#!/bin/bash

#SBATCH --account=def-ko1 #account to charge the calculation
#SBATCH --time=15:00:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=10 #number of CPU requested
#SBATCH --mem-per-cpu=3G #memory requested
#SBATCH --array=8

module load StdEnv/2020 scipy-stack/2023a #load the required module
source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/weights/
echo "Copying weights..."
cp /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/weights/* $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/weights/
echo "Weights copied to temp directory..."

mkdir -p $SLURM_TMPDIR/data/MNIST/
mkdir -p $SLURM_TMPDIR/data/FashionMNIST/
echo "Copying datasets..."
cp -r /home/armenta/scratch/MatrixStatistics/data/MNIST/* $SLURM_TMPDIR/data/MNIST/
echo "MNIST ready"
cp -r /home/armenta/scratch/MatrixStatistics/data/FashionMNIST/* $SLURM_TMPDIR/data/FashionMNIST/
echo "Fashion ready"

python generate_adversarial_examples.py --default_index $SLURM_ARRAY_TASK_ID --nb_workers $SLURM_CPUS_PER_TASK --weights_path=$SLURM_TMPDIR --data_path $SLURM_TMPDIR/data/
