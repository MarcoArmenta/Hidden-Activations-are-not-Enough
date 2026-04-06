#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=04:00:00 #hour:minutes:seconds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=124G #memory requested
#SBATCH --output=slurm_out/C_adv_ex_%A.out
#SBATCH --error=slurm_err/C_adv_ex_%A.err

# Create output and error directories if they don't exist
mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

# Set variables
EXPERIMENT="alexnet_cifar10"

# Prepare environment
echo "Using temporary directory: $SLURM_TMPDIR"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env_rorqual/bin/activate

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Copying weights..."
cp experiments/$EXPERIMENT/weights/* $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Weights copied to temp directory..."

mkdir -p "$SLURM_TMPDIR/data/cifar-10-batches-py/"
echo "Copying datasets..."
cp -r data/cifar-10-batches-py/* "$SLURM_TMPDIR/data/cifar-10-batches-py/" || { echo "Failed to copy dataset"; exit 1; }
echo "CIFAR10 ready"

mkdir gpu-monitor
GPU_LOGFILE="gpu-monitor.$EXPERIMENT.adv_ex.log"
INTERVAL=30  # seconds between GPU checks

monitor_gpu() {
  echo "Timestamp, GPU Utilization (%), GPU Memory Used (MiB), GPU Memory Total (MiB)" > "$GPU_LOGFILE"
  while true; do
    timestamp=$(date +%Y-%m-%dT%H:%M:%S)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
               --format=csv,noheader,nounits \
    | awk -v ts="$timestamp" '{print ts", "$1", "$2", "$3}' >> "$GPU_LOGFILE"
    sleep $INTERVAL
  done
}

# Start monitoring in background
monitor_gpu &
MONITOR_PID=$!
echo "GPU monitor started in background (PID $MONITOR_PID)"

python generate_adversarial_examples.py --experiment_name $EXPERIMENT --temp_dir=$SLURM_TMPDIR

