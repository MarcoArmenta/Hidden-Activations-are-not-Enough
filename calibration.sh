#!/bin/bash
# ==============================================================
# calibration.sh — Standalone GPU Calibration
#
# Submits calibration Slurm jobs for each experiment.
# Produces experiments/$EXP/calibration.json with optimal
# batch_size and SLURM resource estimates.
#
# Usage:
#   bash calibration.sh
#
# After calibration completes, run:
#   bash run_experiment.sh
# ==============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_config.sh"

DRY_RUN=false

# ==============================================================
# Pre-flight checks
# ==============================================================
echo "=============================================================="
echo "  Calibration — Pre-flight Checks"
echo "=============================================================="

# Check we're in the project root
if [ ! -f "constants/constants.py" ]; then
    echo "ERROR: Must run from the project root directory."
    echo "       Expected to find constants/constants.py"
    exit 1
fi

# Validate experiment names
echo ""
echo "Validating experiments..."
for EXP in "${EXPERIMENTS[@]}"; do
    python3 -c "
from constants.constants import DEFAULT_EXPERIMENTS
if '$EXP' not in DEFAULT_EXPERIMENTS:
    import sys
    print('ERROR: \"$EXP\" not found in DEFAULT_EXPERIMENTS (constants/constants.py)')
    print('Available experiments:', ', '.join(sorted(DEFAULT_EXPERIMENTS.keys())))
    sys.exit(1)
print('  OK: $EXP')
" || exit 1
done

# Check datasets (calibration needs the dataset but not pretrained weights)
echo ""
echo "Checking datasets..."
export EXPERIMENT_LIST="${EXPERIMENTS[*]}"
python3 << 'DATASET_CHECK_EOF'
import sys
import os
sys.path.insert(0, '.')
from constants.constants import DEFAULT_EXPERIMENTS

experiments = os.environ.get("EXPERIMENT_LIST", "").split()
datasets_needed = set()
for exp in experiments:
    if exp in DEFAULT_EXPERIMENTS:
        ds = DEFAULT_EXPERIMENTS[exp].get('dataset', 'mnist')
        datasets_needed.add(ds)

dataset_dirs = {
    'cifar10': 'data/cifar-10-batches-py',
    'cifar100': 'data/cifar-100-python',
    'mnist': 'data/MNIST',
    'fashion': 'data/FashionMNIST',
    'imagenet': 'data/ILSVRC2012',
}

missing = []
for ds in datasets_needed:
    dir_path = dataset_dirs.get(ds)
    if dir_path and os.path.isdir(dir_path):
        print(f'  OK: {ds} ({dir_path})')
    elif ds == 'imagenet':
        # ImageNet is at /datashare/imagenet/ILSVRC2012/ on the cluster
        if os.path.isdir('/datashare/imagenet/ILSVRC2012'):
            print(f'  OK: {ds} (/datashare/imagenet/ILSVRC2012)')
        else:
            print(f'  DOWNLOAD NEEDED: {ds}')
            missing.append(ds)
    else:
        print(f'  DOWNLOAD NEEDED: {ds}')
        missing.append(ds)

if missing:
    print('\nDownloading missing datasets...')
    for ds in missing:
        if ds == 'imagenet':
            print('  ERROR: ImageNet must be available at /datashare/imagenet/ILSVRC2012/')
            sys.exit(1)
        elif ds == 'cifar10':
            from torchvision.datasets import CIFAR10
            CIFAR10(root='./data', train=True, download=True)
            CIFAR10(root='./data', train=False, download=True)
            print(f'  Downloaded: {ds}')
        elif ds == 'cifar100':
            from torchvision.datasets import CIFAR100
            CIFAR100(root='./data', train=True, download=True)
            CIFAR100(root='./data', train=False, download=True)
            print(f'  Downloaded: {ds}')
        elif ds == 'mnist':
            import torchvision
            torchvision.datasets.MNIST(root='./data', train=True, download=True)
            print(f'  Downloaded: {ds}')
        elif ds == 'fashion':
            import torchvision
            torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
            print(f'  Downloaded: {ds}')

print('Datasets ready.')
DATASET_CHECK_EOF

echo ""
echo "Pre-flight checks complete."
echo ""

# ==============================================================
# Submit calibration jobs
# ==============================================================
echo "=============================================================="
echo "  Calibration — Submission"
echo "=============================================================="

mkdir -p "$SLURM_OUT_DIR" "$SLURM_ERR_DIR"

for EXP in "${EXPERIMENTS[@]}"; do
    JOB_DIR="experiments/$EXP/orchestrator_jobs"
    mkdir -p "$JOB_DIR"

    echo ""
    echo "--- $EXP ---"

    # Skip if calibration.json already exists
    if [ -f "experiments/$EXP/calibration.json" ]; then
        echo "  [0] Calibration:         SKIPPED (calibration.json exists)"
        continue
    fi

    DATASET_FOR_CALIB=$(get_experiment_dataset "$EXP")
    CALIB_COPY_DATA=$(get_dataset_copy_commands "$DATASET_FOR_CALIB")

    cat > "$JOB_DIR/calibrate.sh" << CALIB_EOF
#!/bin/bash
#SBATCH --account=$GPU_ACCOUNT
#SBATCH $CALIB_GPU
#SBATCH --cpus-per-task=$CALIB_CPUS
#SBATCH --time=$CALIB_TIME
#SBATCH --mem=$CALIB_MEM
#SBATCH --output=$SLURM_OUT_DIR/PIPE_CALIB_${EXP}_%A.out
#SBATCH --error=$SLURM_ERR_DIR/PIPE_CALIB_${EXP}_%A.err

mkdir -p \$SLURM_SUBMIT_DIR/$SLURM_OUT_DIR \$SLURM_SUBMIT_DIR/$SLURM_ERR_DIR
module load $MODULES
source $ENV_NAME/bin/activate

$CALIB_COPY_DATA

# GPU monitoring during calibration
mkdir -p \$SLURM_SUBMIT_DIR/gpu-monitor/
GPU_LOGFILE="\$SLURM_SUBMIT_DIR/gpu-monitor/$EXP.calibration.log"
monitor_gpu() {
  echo "Timestamp, GPU Util (%), Mem Used (MiB), Mem Total (MiB)" > "\$GPU_LOGFILE"
  while true; do
    ts=\$(date +%Y-%m-%dT%H:%M:%S)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -v t="\$ts" '{print t", "\$1", "\$2", "\$3}' >> "\$GPU_LOGFILE"
    sleep 30
  done
}
monitor_gpu &
MONITOR_PID=\$!

# Always use full params for calibration so results are reusable
python calibrate.py \\
    --experiment_name $EXP \\
    --temp_dir \$SLURM_TMPDIR \\
    --target_utilization 0.93 \\
    --timing_samples 50 \\
    --total_chunks 8 \\
    --num_samples_per_class 100 \\
    --samples_per_attack 500

kill \$MONITOR_PID 2>/dev/null || true
echo "Calibration complete for $EXP."
CALIB_EOF

    if [ "$DRY_RUN" = "false" ]; then
        CALIB_JOB=$(submit_job "$JOB_DIR/calibrate.sh" "")
        echo "  [0] Calibration:         $CALIB_JOB"
    else
        echo "  [DRY RUN] Calibration script generated: $JOB_DIR/calibrate.sh"
    fi
done

echo ""
echo "=============================================================="
echo "  Calibration — Summary"
echo "=============================================================="
echo ""
echo "  Experiments: ${EXPERIMENTS[*]}"
echo "  GPU account: $GPU_ACCOUNT"
echo ""
echo "  After calibration completes, run:"
echo "    bash run_experiment.sh"
echo "=============================================================="
