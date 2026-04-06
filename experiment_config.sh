#!/bin/bash
# ==============================================================
# experiment_config.sh — Defaults, Resource Profiles & Helpers
#
# Provides default values for experiment settings, resource
# profiles, accounts, and helper functions.
# Sourced by calibration.sh, run_experiment.sh, and job_recovery.sh.
#
# All variables use ${VAR:-default} so that callers can override
# them by setting values before sourcing this file.
# ==============================================================

# --- Default configuration (overridable) ---
ACCOUNT="${ACCOUNT:-def-assem}"
GPU_ACCOUNT="${GPU_ACCOUNT:-}"              # Account for GPU jobs (defaults to ACCOUNT if empty)
CPU_ACCOUNT="${CPU_ACCOUNT:-}"              # Account for CPU jobs (defaults to ACCOUNT if empty)
TOTAL_CHUNKS="${TOTAL_CHUNKS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1800}"
NUM_SAMPLES_PER_CLASS="${NUM_SAMPLES_PER_CLASS:-500}"
SAMPLES_PER_ATTACK="${SAMPLES_PER_ATTACK:-500}"
TEST_SIZE="${TEST_SIZE:--1}"
ENV_NAME="${ENV_NAME:-env}"
MODULES="${MODULES:-StdEnv/2023 python/3.11.5 scipy-stack/2025a}"
SLURM_OUT_DIR="${SLURM_OUT_DIR:-slurm_out}"
SLURM_ERR_DIR="${SLURM_ERR_DIR:-slurm_err}"
# --- Incremental save settings ---
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"           # Incremental save every N new matrices
SAVE_CHECK_SECONDS="${SAVE_CHECK_SECONDS:-60}"  # How often background process checks
SAVE_GRACE_SECONDS="${SAVE_GRACE_SECONDS:-180}" # Seconds before wall time to trigger emergency save
# --- Relaunch sentinel ---
MAX_SENTINEL_CYCLES="${MAX_SENTINEL_CYCLES:-5}"  # Max whole-pipeline re-launch cycles

# --- Resource profiles (normal mode, overridable) ---
# Step A (CPU-only — small networks, transfer learning)
A_CPUS="${A_CPUS:-4}"
A_TIME="${A_TIME:-06:00:00}"
A_MEM="${A_MEM:-15G}"
# Step B
B_GPU="${B_GPU:---gpus=h100:1}"
B_CPUS="${B_CPUS:-12}"
B_TIME="${B_TIME:-00:20:00}"
B_MEM="${B_MEM:-280G}"
# Step C (per-attack defaults — each attack runs as a separate Slurm job)
C_GPU="${C_GPU:---gpus=h100:1}"
C_CPUS="${C_CPUS:-4}"
C_TIME="${C_TIME:-03:00:00}"
C_MEM="${C_MEM:-32G}"
# Step D (Adv Matrices)
D_GPU="${D_GPU:---gpus=h100:1}"
D_CPUS="${D_CPUS:-12}"
D_TIME="${D_TIME:-12:00:00}"
D_MEM="${D_MEM:-280G}"
# Step E (Representation Comparison - GPU)
E_GPU="${E_GPU:---gpus=h100:1}"
E_CPUS="${E_CPUS:-8}"
E_TIME="${E_TIME:-08:00:00}"
E_MEM="${E_MEM:-128G}"
# Step G (Theorem 4.5 Validation - GPU)
G_GPU="${G_GPU:---gpus=h100:1}"
G_CPUS="${G_CPUS:-4}"
G_TIME="${G_TIME:-06:00:00}"
G_MEM="${G_MEM:-64G}"
# Step F (LaTeX Tables - CPU-only, lightweight)
F_CPUS="${F_CPUS:-2}"
F_TIME="${F_TIME:-00:15:00}"
F_MEM="${F_MEM:-4G}"
# Audit
AUDIT_CPUS="${AUDIT_CPUS:-4}"
AUDIT_TIME="${AUDIT_TIME:-00:30:00}"
AUDIT_MEM="${AUDIT_MEM:-32G}"
# Calibration
CALIB_GPU="${CALIB_GPU:---gpus=h100:1}"
CALIB_CPUS="${CALIB_CPUS:-4}"
CALIB_TIME="${CALIB_TIME:-01:00:00}"
CALIB_MEM="${CALIB_MEM:-32G}"

# ==============================================================
# Experiments to process (overridable)
# ==============================================================
if [ -z "${EXPERIMENTS+x}" ]; then
    EXPERIMENTS=("alexnet_cifar10")
fi

# --- Resolve per-type accounts (default to ACCOUNT) ---
GPU_ACCOUNT="${GPU_ACCOUNT:-$ACCOUNT}"
CPU_ACCOUNT="${CPU_ACCOUNT:-$ACCOUNT}"

# --- Load environment (needed for pre-flight Python calls) ---
module load $MODULES 2>/dev/null || true
if [ -d "$ENV_NAME" ]; then
    source $ENV_NAME/bin/activate
fi

# ==============================================================
# Helper functions
# ==============================================================
submit_job() {
    local script="$1"
    local deps="$2"
    local sbatch_cmd="sbatch --parsable"
    if [ -n "$deps" ]; then
        sbatch_cmd="sbatch --parsable --dependency=afterok:${deps}"
    fi
    local job_id
    job_id=$($sbatch_cmd "$script")
    echo "$job_id"
}

submit_job_afterany() {
    local script="$1"
    local deps="$2"
    local sbatch_cmd="sbatch --parsable"
    if [ -n "$deps" ]; then
        sbatch_cmd="sbatch --parsable --dependency=afterany:${deps}"
    fi
    local job_id
    job_id=$($sbatch_cmd "$script")
    echo "$job_id"
}

enforce_min_time() {
    # Ensures SLURM time is at least a minimum floor (default 30 min)
    # Usage: enforce_min_time "HH:MM:SS" ["HH:MM:SS_floor"]
    local time_str="$1"
    local min_time="${2:-00:30:00}"
    local h m s
    IFS=: read -r h m s <<< "$time_str"
    local total=$(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))
    IFS=: read -r h m s <<< "$min_time"
    local min_seconds=$(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))
    if [ "$total" -lt "$min_seconds" ]; then
        echo "$min_time"
    else
        echo "$time_str"
    fi
}

enforce_min_mem() {
    # Ensures SLURM memory is at least a minimum floor (default 16G)
    # Usage: enforce_min_mem "XG" ["XG_floor"]
    # All values assumed to be in "XG" format (integer followed by G)
    local mem_str="$1"
    local min_mem="${2:-16G}"
    # Strip G suffix; if not present, return as-is (can't compare)
    local mem_val="${mem_str%G}"
    local min_val="${min_mem%G}"
    # Only compare if both stripped to integers
    if [[ "$mem_val" =~ ^[0-9]+$ ]] && [[ "$min_val" =~ ^[0-9]+$ ]]; then
        if [ "$mem_val" -lt "$min_val" ]; then
            echo "$min_mem"
            return
        fi
    fi
    echo "$mem_str"
}

double_mem() {
    # Doubles a memory value in "XG" format, capped at 480G
    # Usage: double_mem "16G" → "32G"
    local mem_str="$1"
    local max_mem=480
    local mem_val="${mem_str%G}"
    local doubled=$((mem_val * 2))
    if [ "$doubled" -gt "$max_mem" ]; then
        doubled=$max_mem
    fi
    echo "${doubled}G"
}

double_time() {
    # Doubles a time value in "HH:MM:SS" format, capped at 48:00:00
    # Usage: double_time "03:00:00" → "06:00:00"
    local time_str="$1"
    local max_seconds=172800  # 48 hours
    local h m s
    IFS=: read -r h m s <<< "$time_str"
    local total=$(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))
    local doubled=$(( total * 2 ))
    if [ "$doubled" -gt "$max_seconds" ]; then
        doubled=$max_seconds
    fi
    local new_h=$(( doubled / 3600 ))
    local new_m=$(( (doubled % 3600) / 60 ))
    local new_s=$(( doubled % 60 ))
    printf "%02d:%02d:%02d" "$new_h" "$new_m" "$new_s"
}

detect_last_job_state() {
    # Query sacct for the most recent job matching a log pattern.
    # Returns: OOM_KILLED, TIMEOUT, FAILED, COMPLETED, CANCELLED, or UNKNOWN
    # Usage: detect_last_job_state "PIPE_A_alexnet_cifar10" "slurm_out"
    local log_prefix="$1"
    local slurm_out_dir="${2:-slurm_out}"

    # Find the most recent job ID from log files matching this prefix
    local latest_job=""
    for f in "$slurm_out_dir"/${log_prefix}_*.out; do
        [ -f "$f" ] || continue
        local fname=$(basename "$f")
        local job_id=$(echo "$fname" | grep -oP '_(\d+)\.out$' | grep -oP '\d+')
        if [ -n "$job_id" ]; then
            if [ -z "$latest_job" ] || [ "$job_id" -gt "$latest_job" ]; then
                latest_job="$job_id"
            fi
        fi
    done

    if [ -z "$latest_job" ]; then
        echo "UNKNOWN"
        return
    fi

    # Query sacct for the job state
    local state
    state=$(sacct --jobs="$latest_job" --parsable2 --noheader --format=State 2>/dev/null | head -1 | cut -d'|' -f1)

    case "$state" in
        OUT_OF_MEMORY)  echo "OOM_KILLED" ;;
        TIMEOUT)        echo "TIMEOUT" ;;
        FAILED)         echo "FAILED" ;;
        COMPLETED)      echo "COMPLETED" ;;
        CANCELLED*)     echo "CANCELLED" ;;
        *)              echo "UNKNOWN" ;;
    esac
}

read_checkpoint_field() {
    # $1 = checkpoint file path, $2 = field name
    # Returns: field value, or empty string if missing
    if [ -f "$1" ]; then
        python3 -c "import json; print(json.load(open('$1')).get('$2',''))" 2>/dev/null || echo ""
    else
        echo ""
    fi
}

# Determine dataset dirs to copy based on experiment
get_dataset_copy_commands() {
    local dataset="$1"
    case "$dataset" in
        cifar10)
            echo 'mkdir -p $SLURM_TMPDIR/data/cifar-10-batches-py/'
            echo 'cp -r $SLURM_SUBMIT_DIR/data/cifar-10-batches-py/* $SLURM_TMPDIR/data/cifar-10-batches-py/ 2>/dev/null || true'
            ;;
        cifar100)
            echo 'mkdir -p $SLURM_TMPDIR/data/cifar-100-python/'
            echo 'cp -r $SLURM_SUBMIT_DIR/data/cifar-100-python/* $SLURM_TMPDIR/data/cifar-100-python/ 2>/dev/null || true'
            ;;
        mnist)
            echo 'mkdir -p $SLURM_TMPDIR/data/MNIST/'
            echo 'cp -r $SLURM_SUBMIT_DIR/data/MNIST/* $SLURM_TMPDIR/data/MNIST/ 2>/dev/null || true'
            ;;
        fashion)
            echo 'mkdir -p $SLURM_TMPDIR/data/FashionMNIST/'
            echo 'cp -r $SLURM_SUBMIT_DIR/data/FashionMNIST/* $SLURM_TMPDIR/data/FashionMNIST/ 2>/dev/null || true'
            ;;
        imagenet)
            echo '# ImageNet: reading directly from /datashare/imagenet/ILSVRC2012/ (NFS)'
            ;;
    esac
}

# ==============================================================
# Generate and submit pipeline for each experiment
# ==============================================================

# Get dataset for each experiment
get_experiment_dataset() {
    python3 -c "
from constants.constants import DEFAULT_EXPERIMENTS
print(DEFAULT_EXPERIMENTS.get('$1', {}).get('dataset', 'cifar10'))
"
}

get_experiment_epochs() {
    python3 -c "
from constants.constants import DEFAULT_EXPERIMENTS
print(DEFAULT_EXPERIMENTS.get('$1', {}).get('epochs', 0))
"
}

get_experiment_num_classes() {
    python3 -c "
from constants.constants import DEFAULT_EXPERIMENTS
d = DEFAULT_EXPERIMENTS.get('$1', {}).get('dataset', 'cifar10')
print({'cifar10':10,'cifar100':100,'mnist':10,'fashion':10,'imagenet':1000}.get(d, 10))
"
}

read_checkpoint_status() {
    # $1 = checkpoint file path
    # Returns: complete, partial, failed, or missing
    if [ -f "$1" ]; then
        python3 -c "import json; print(json.load(open('$1')).get('status','partial'))"
    else
        echo "missing"
    fi
}
