#!/bin/bash
#SBATCH --account=def-jcbus
#SBATCH --array=0-7
#SBATCH --time=00:20:00  # Increased to accommodate potential longer runs
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=280G  # Increased to prevent segmentation faults
#SBATCH --output=slurm_out/B_mats_%A_%a.out
#SBATCH --error=slurm_err/B_mats_%A_%a.err
#SBATCH --exclude=fc10512

mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

# Set variables
EXPERIMENT="vgg_cifar100"
TASK_ID=$SLURM_ARRAY_TASK_ID
ZIP_FILE=$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/matrices_task_$TASK_ID.zip

# Prepare environment
TEMP_DIR=$SLURM_TMPDIR
echo "Using temporary directory: $TEMP_DIR"
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a #cuda/12.2 cudnn
source env_fir/bin/activate

# Copy data and weights to temporary directory
echo "Copying datasets..."
#mkdir -p $TEMP_DIR/data/cifar-10-batches-py/
#cp -r data/cifar-10-batches-py/* $TEMP_DIR/data/cifar-10-batches-py/ || { echo "Failed to copy dataset"; exit 1; }

mkdir -p $SLURM_TMPDIR/data/cifar-100-python/
cp -r data/cifar-100-python/* $SLURM_TMPDIR/data/cifar-100-python/

echo "Copying weights for task $TASK_ID..."
mkdir -p $TEMP_DIR/experiments/$EXPERIMENT/weights/
cp experiments/$EXPERIMENT/weights/* $TEMP_DIR/experiments/$EXPERIMENT/weights/

#echo "copy imagenet.."
#mkdir -p "$TEMP_DIR/data/ILSVRC2012/"
#cp -r /datashare/imagenet/ILSVRC2012/* $TEMP_DIR/data/ILSVRC2012/
#echo "imagenet ready!!!"

# Check for existing zip file and unzip if present
if [ -f "$ZIP_FILE" ]; then
    echo "Found existing zip file: $ZIP_FILE"
    cp "$ZIP_FILE" "$TEMP_DIR/experiments/$EXPERIMENT/" || { echo "Failed to copy zip file"; exit 1; }
    echo "Unzipping existing matrices for task $TASK_ID..."
    unzip -o "$TEMP_DIR/experiments/$EXPERIMENT/matrices_task_$TASK_ID.zip" -d "$TEMP_DIR/experiments/$EXPERIMENT/" || { echo "Unzipping failed"; exit 1; }
    echo "Existing matrices unzipped to $TEMP_DIR/experiments/$EXPERIMENT/matrices"
fi

mkdir gpu-monitor/
GPU_LOGFILE="gpu-monitor/$EXPERIMENT.mats.$TASK_ID.log"
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

# Run Python script in the foreground
echo "Generating matrices for task $TASK_ID..."
timeout 20m python generate_matrices.py --temp_dir $TEMP_DIR --experiment $EXPERIMENT --chunk_id $TASK_ID --total_chunks 8 --batch_size 1800 --num_samples_per_class 100

# Zip the matrices directory
echo "Zipping matrices for task $TASK_ID..."
cd $TEMP_DIR/experiments/$EXPERIMENT
zip -r matrices_task_$TASK_ID.zip matrices || { echo "Zipping failed"; exit 1; }

# Verify the zip
cd $SLURM_SUBMIT_DIR
python -m utils.data_integrity --verify-zip $TEMP_DIR/experiments/$EXPERIMENT/matrices_task_$TASK_ID.zip || { echo "Zip verification failed"; exit 1; }
cd $TEMP_DIR/experiments/$EXPERIMENT

# Copy the zip file to $SLURM_SUBMIT_DIR
echo "Copying zip file to $SLURM_SUBMIT_DIR..."
cp matrices_task_$TASK_ID.zip $SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/ || { echo "Failed to copy zip file"; exit 1; }
echo "Task $TASK_ID completed successfully"