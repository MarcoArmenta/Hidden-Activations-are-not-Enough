#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=09:00:00 #hour:minutes:seconds
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G #memory requested
#SBATCH --output=slurm_out/D_rej_lev_%A_%a.out
#SBATCH --error=slurm_err/D_rej_lev_%A_%a.err

EXPERIMENT="vgg_cifar100"
ZIP_OUTPUT_FILE="matrices_task_$SLURM_ARRAY_TASK_ID.zip"
# Create output and error directories if they don't exist
mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env_fir/bin/activate

# Prepare temp directories and weights
mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Copying weights..."
cp $SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/weights/* $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Weights copied to temp directory..."

EXPERIMENT_DATA_TRAIN="$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/rejection_levels/exp_dataset_train.pth"
EXPERIMENT_DATA_LABELS="$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/rejection_levels/exp_dataset_labels.pth"
mkdir -p "$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/"

#mkdir -p "$SLURM_TMPDIR/data/cifar-10-batches-py/"
echo "Copying datasets..."
#cp -r data/cifar-10-batches-py/* "$SLURM_TMPDIR/data/cifar-10-batches-py/" || { echo "Failed to copy dataset"; exit 1; }
#echo "CIFAR10 ready"
mkdir -p $SLURM_TMPDIR/data/cifar-100-python/
cp -r data/cifar-100-python/* $SLURM_TMPDIR/data/cifar-100-python/

if [ -f "$EXPERIMENT_DATA_TRAIN" ]; then
    echo "Found existing experiment data train file: $EXPERIMENT_DATA_TRAIN"
    cp "$EXPERIMENT_DATA_TRAIN" "$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/exp_dataset_train.pth" || { echo "Failed to copy file"; exit 1; }
fi

if [ -f "$EXPERIMENT_DATA_LABELS" ]; then
    echo "Found existing experiment data labels file: $EXPERIMENT_DATA_LABELS"
    cp "$EXPERIMENT_DATA_LABELS" "$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/exp_dataset_labels.pth" || { echo "Failed to copy file"; exit 1; }
fi

# If matrices.zip exists on permanent storage, copy and unzip into tmp
ZIP_FILE="$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/rejection_levels/matrices_task_$SLURM_ARRAY_TASK_ID.zip"
if [ -f "$ZIP_FILE" ]; then
    echo "Found existing zip file: $ZIP_FILE"
    cp "$ZIP_FILE" "$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/"
    cd "$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/"
    unzip -o $ZIP_OUTPUT_FILE
    echo "Unzipped existing matrices"
    cd -
fi

mkdir -p gpu-monitor
GPU_LOGFILE="gpu-monitor/$EXPERIMENT.rej_lev.task-$SLURM_ARRAY_TASK_ID.log"
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

# start monitor in background
monitor_gpu &
MONITOR_PID=$!
echo "GPU monitor started in background (PID $MONITOR_PID)"

# Launch 4 workers (chunk_id 0..3), binding each to one GPU
echo "Starting worker for chunk $SLURM_ARRAY_TASK_ID on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
timeout 8h python compute_matrices_for_rejection_level.py \
    --experiment_name $EXPERIMENT \
    --temp_dir $SLURM_TMPDIR \
    --batch_size 1800 \
    --chunk_id $SLURM_ARRAY_TASK_ID \
    --total_chunks 8 \

# Zip matrices directory
MATRICES_DIR="$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices"
ZIP_OUTPUT_DIR="$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels"

if [ -d "$MATRICES_DIR" ]; then
    echo "Zipping matrices from $MATRICES_DIR ..."
    cd "$ZIP_OUTPUT_DIR" || { echo "Failed to cd to $ZIP_OUTPUT_DIR"; }
    zip -r $ZIP_OUTPUT_FILE matrices || echo "Zip failed"

    # Verify the zip
    cd $SLURM_SUBMIT_DIR
    python -m utils.data_integrity --verify-zip $ZIP_OUTPUT_DIR/$ZIP_OUTPUT_FILE || echo "Zip verification failed"
    cd - >/dev/null || true
else
    echo "No matrices directory found at $MATRICES_DIR, skipping zipping"
fi

# Copy the zip file back to HOME_DIR (permanent storage)
TEMP_ZIP="$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/$ZIP_OUTPUT_FILE"
DEST_DIR="$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/rejection_levels/"
mkdir -p $DEST_DIR
if [ -f "$TEMP_ZIP" ]; then
    echo "Copying zip file $TEMP_ZIP to $DEST_DIR"
    mkdir -p "$DEST_DIR"
    cp "$TEMP_ZIP" "$DEST_DIR" || { echo "Failed to copy zip file"; exit 1; }
else
    echo "No zip file to copy from temp dir ($TEMP_ZIP)"
fi

echo "Job completed"
