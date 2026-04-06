#!/bin/bash

#SBATCH --account=def-bruestle #account to charge the calculation
#SBATCH --time=12:00:00 #hour:minutes:seconds
#SBATCH --array=0-15
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=280G  # Increased to prevent segmentation faults
#SBATCH --output=slurm_out/F_adv_mats_%A_%a.out
#SBATCH --error=slurm_err/F_adv_mats_%A_%a.err
#SBATCH --exclude=fc10512


EXPERIMENT="vgg_cifar100"
ZIP_FILE="$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/adv_matrices_task_$SLURM_ARRAY_TASK_ID.zip"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env_fir/bin/activate

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Copying weights..."
cp $SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/weights/* $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Weights copied to temp directory..."

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/
echo "Copying adversarial examples..."
#cp -r $SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/adversarial_examples/* $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/
tar cf - -C $SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/adversarial_examples . | tar xvf - -C $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples
echo "Adv examples ready."

mkdir -p $SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/adversarial_matrices/

if [ -f "$ZIP_FILE" ]; then
    echo "Found existing experiment data labels file: $ZIP_FILE"
    cp "$ZIP_FILE" "$SLURM_TMPDIR/experiments/$EXPERIMENT/adv_matrices_task_$SLURM_ARRAY_TASK_ID.zip" || { echo "Failed to copy file"; exit 1; }
    echo "Unzipping $ZIP_FILE to temporary directory..."
    unzip "$SLURM_TMPDIR/experiments/$EXPERIMENT/adv_matrices_task_$SLURM_ARRAY_TASK_ID.zip" -d "$SLURM_TMPDIR/experiments/$EXPERIMENT/" || { echo "Failed to unzip file"; exit 1; }
    ls $SLURM_TMPDIR/experiments/$EXPERIMENT/
    echo "Unzip completed."
fi

mkdir -p gpu-monitor/
GPU_LOGFILE="gpu-monitor/$EXPERIMENT.adv_mats.$SLURM_ARRAY_TASK_ID.log"
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

# Run Python script in background and capture its PID
timeout 11h python generate_adversarial_matrices.py --experiment_name $EXPERIMENT --temp_dir $SLURM_TMPDIR --chunk_id $SLURM_ARRAY_TASK_ID --total_chunks 16 --batch_size 1800 --samples_per_attack 500

# Zip the matrices
echo "Zipping matrices..."
cd $SLURM_TMPDIR/experiments/$EXPERIMENT/
zip -r adv_matrices_task_$SLURM_ARRAY_TASK_ID.zip adversarial_matrices/

# Verify the zip
cd $SLURM_SUBMIT_DIR
python -m utils.data_integrity --verify-zip $SLURM_TMPDIR/experiments/$EXPERIMENT/adv_matrices_task_$SLURM_ARRAY_TASK_ID.zip || { echo "Zip verification failed"; exit 1; }
cd -
# Copy the zip file to the permanent directory
# echo "Zip file: $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_matrices.zip"
echo "Copying zip file $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_$SLURM_ARRAY_TASK_ID.zip to $SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/adversarial_matrices/"
mkdir -p $SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/adversarial_matrices/
cp $SLURM_TMPDIR/experiments/$EXPERIMENT/adv_matrices_task_$SLURM_ARRAY_TASK_ID.zip $SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/ || { echo "Failed to copy zip file"; exit 1; }
ls $SLURM_TMPDIR/experiments/$EXPERIMENT/
echo "Done!"