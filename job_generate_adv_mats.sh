#!/bin/bash

#SBATCH --account=<ACCOUNT_NAME> #account to charge the calculation
#SBATCH --time=4:00:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=15 #number of CPU requested
#SBATCH --mem-per-cpu=2G #memory requested
#SBATCH --array=8

module load StdEnv/2020 scipy-stack/2023a #load the required module
source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/weights/
echo "Copying weights..."
cp experiments/$SLURM_ARRAY_TASK_ID/weights/* $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/weights/
echo "Weights copied to temp directory..."

mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/adversarial_examples/
echo "Copying adversarial examples..."
cp -r experiments/$SLURM_ARRAY_TASK_ID/adversarial_examples/* $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/adversarial_examples/
echo "Adv examples ready."

mkdir -p experiments/$SLURM_ARRAY_TASK_ID/adversarial_matrices/

python generate_adversarial_matrices.py --nb_workers $SLURM_CPUS_PER_TASK --default_index $SLURM_ARRAY_TASK_ID --temp_dir $SLURM_TMPDIR

echo "Done!"
