#!/bin/bash

#SBATCH --account=def-bruestle #account to charge the calculation
#SBATCH --time=1:00:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=1 #number of CPU requested
#SBATCH --mem-per-cpu=15G #memory requested
#SBATCH --array=13

module load StdEnv/2020 scipy-stack/2023a #load the required module
source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/weights/
echo "Copying weights..."
cp /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/weights/* $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/weights/

echo "Copying matrix statistics..."
mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/matrices/
cp /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/matrices/matrix_statistics.json $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/matrices/

mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/matrices/

echo "Copying adversarial matrices..."
mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/adversarial_matrices/
cp /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/adversarial_matrices/adversarial_matrices.zip $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/adversarial_matrices/
echo "Decompress..."
unzip /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/adversarial_matrices/adversarial_matrices.zip -d $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/adversarial_matrices/

echo "Copying adversarial examples..."
mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/adversarial_examples/
cp -r /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/adversarial_examples/* $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/adversarial_examples/

echo "Copying exp dataset train..."
mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/matrices/
cp /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/exp_dataset_train.pth $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/

echo "Copying rejection level matrices..."
cp /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/matrices/matrices.zip $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/matrices/

echo "Decompress..."
unzip $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/matrices/matrices.zip -d $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/matrices/

echo "All data ready!"

python grid_search.py --nb_workers=$SLURM_CPUS_PER_TASK --default_index $SLURM_ARRAY_TASK_ID --temp_dir $SLURM_TMPDIR --rej_lev 0
echo "Grid search finished !!!"
