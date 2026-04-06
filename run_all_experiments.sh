#!/bin/bash
# =============================================================================
# Run all experiments for the paper revision.
#
# This script submits the full pipeline for each experiment, plus the
# representation comparison (Step H) that produces the head-to-head results.
#
# Usage:
#   bash run_all_experiments.sh              # Full runs
#   bash run_all_experiments.sh --test       # Quick test runs
#   bash run_all_experiments.sh --compare    # Only run comparison on existing data
# =============================================================================

set -euo pipefail

TEST_FLAG=""
COMPARE_ONLY=false

for arg in "$@"; do
    case $arg in
        --test)     TEST_FLAG="--test" ;;
        --compare)  COMPARE_ONLY=true ;;
    esac
done

# ---- Experiments to run ----
# Priority 1: Must complete before resubmission
PRIORITY_1=(
    "lenet_cifar10"
    "alexnet_cifar10"
    "resnet_cifar10"
)

# Priority 2: Strongly recommended
PRIORITY_2=(
    "vgg_cifar10"
)

# Priority 3: Nice to have
PRIORITY_3=(
    "resnet_cifar100"
)

ALL_EXPERIMENTS=("${PRIORITY_1[@]}" "${PRIORITY_2[@]}")

# =============================================================================
# STEP 1: Run full pipelines (A→F) for each experiment
# =============================================================================
if [ "$COMPARE_ONLY" = false ]; then
    echo "=============================================="
    echo "  SUBMITTING PIPELINES FOR ALL EXPERIMENTS"
    echo "=============================================="
    echo ""

    for exp in "${ALL_EXPERIMENTS[@]}"; do
        echo "--- Submitting pipeline for: $exp ---"
        bash run_experiment.sh $TEST_FLAG --skip-audit "$exp"
        echo ""
    done

    echo ""
    echo "All pipelines submitted. Monitor with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "After all jobs complete, run:"
    echo "  bash run_all_experiments.sh --compare"
fi

# =============================================================================
# STEP 2: Run representation comparison (Step E)
# =============================================================================
if [ "$COMPARE_ONLY" = true ]; then
    echo "=============================================="
    echo "  RUNNING REPRESENTATION COMPARISON"
    echo "=============================================="
    echo ""

    # Check which experiments have completed pipeline
    READY_EXPERIMENTS=()
    for exp in "${ALL_EXPERIMENTS[@]}"; do
        if [ -d "experiments/$exp/adversarial_matrices" ] && \
           [ "$(ls experiments/$exp/adversarial_matrices/ 2>/dev/null | wc -l)" -gt 0 ]; then
            READY_EXPERIMENTS+=("$exp")
            echo "  [READY] $exp"
        else
            echo "  [SKIP]  $exp (no adversarial matrices yet)"
        fi
    done

    if [ ${#READY_EXPERIMENTS[@]} -eq 0 ]; then
        echo ""
        echo "No experiments ready for comparison. Run pipelines first."
        exit 1
    fi

    echo ""
    echo "Running comparison for: ${READY_EXPERIMENTS[*]}"
    echo ""

    python compare_representations.py \
        --experiment "${READY_EXPERIMENTS[@]}" \
        2>&1 | tee "reports/comparison_$(date +%Y%m%d_%H%M%S).txt"

    echo ""
    echo "Comparison complete. Results saved to:"
    for exp in "${READY_EXPERIMENTS[@]}"; do
        echo "  experiments/$exp/comparison/representation_comparison.json"
    done
fi

# =============================================================================
# STEP 3: Generate LaTeX tables
# =============================================================================
if [ "$COMPARE_ONLY" = true ]; then
    echo ""
    echo "=============================================="
    echo "  GENERATING LATEX TABLES"
    echo "=============================================="
    mkdir -p tables
    python generate_latex_tables.py --experiments "${READY_EXPERIMENTS[@]}" --output tables/
    echo "Tables saved to tables/"
fi
