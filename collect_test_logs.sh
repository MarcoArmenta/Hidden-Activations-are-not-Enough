#!/bin/bash
# ==============================================================
# collect_test_logs.sh — Collect test diagnostics into one file
#
# After running --test mode, this script collects all errors
# and key outputs into a single test_diagnostics.txt file
# that can be shared for debugging.
#
# Usage:
#   bash collect_test_logs.sh
#   bash collect_test_logs.sh normal   # for normal mode logs
# ==============================================================

MODE="${1:-test}"

if [ "$MODE" = "test" ]; then
    OUT_DIR="slurm_out_test"
    ERR_DIR="slurm_err_test"
else
    OUT_DIR="slurm_out"
    ERR_DIR="slurm_err"
fi

DIAG_FILE="test_diagnostics.txt"

if [ ! -d "$ERR_DIR" ] && [ ! -d "$OUT_DIR" ]; then
    echo "ERROR: No log directories found ($OUT_DIR, $ERR_DIR)."
    echo "Run the pipeline first, then collect diagnostics."
    exit 1
fi

echo "Collecting diagnostics from $OUT_DIR/ and $ERR_DIR/..."
{
    echo "=============================================================="
    echo "  Pipeline Test Diagnostics"
    echo "  Generated: $(date)"
    echo "  Mode: $MODE"
    echo "=============================================================="
    echo ""

    # --- Job summary from seff (if sacct is available) ---
    echo "=== Recent Job Summary ==="
    if command -v sacct &>/dev/null; then
        sacct --user="$USER" --starttime="$(date -d '24 hours ago' '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || date -v-24H '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || echo '2025-01-01')" \
              --format="JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS" 2>/dev/null | head -50
    else
        echo "(sacct not available)"
    fi
    echo ""

    # --- Calibration output ---
    echo "=== Calibration Output ==="
    CALIB_LOG=$(ls -t "$OUT_DIR"/PIPE_CALIB_* 2>/dev/null | head -1)
    if [ -n "$CALIB_LOG" ]; then
        echo "--- $CALIB_LOG ---"
        cat "$CALIB_LOG"
    else
        echo "(no calibration log found)"
    fi
    echo ""

    # --- Calibration JSON ---
    echo "=== Calibration JSON ==="
    for f in experiments/*/calibration.json; do
        if [ -f "$f" ]; then
            echo "--- $f ---"
            cat "$f"
            echo ""
        fi
    done
    if ! ls experiments/*/calibration.json &>/dev/null; then
        echo "(no calibration.json found)"
    fi
    echo ""

    # --- Error logs (non-empty only) ---
    echo "=== Error Logs (non-empty) ==="
    if [ -d "$ERR_DIR" ]; then
        for err_file in "$ERR_DIR"/*.err; do
            [ -f "$err_file" ] || continue
            if [ -s "$err_file" ]; then
                echo ""
                echo "--- $(basename "$err_file") ---"
                tail -50 "$err_file"
                echo ""
            fi
        done
    fi
    echo ""

    # --- Failed step outputs (last 30 lines of each) ---
    echo "=== Step Outputs (last 30 lines each) ==="
    if [ -d "$OUT_DIR" ]; then
        for out_file in "$OUT_DIR"/*.out; do
            [ -f "$out_file" ] || continue
            echo ""
            echo "--- $(basename "$out_file") ---"
            tail -30 "$out_file"
        done
    fi

    echo ""
    echo "=== End of Diagnostics ==="

} > "$DIAG_FILE"

echo "Diagnostics saved to: $DIAG_FILE"
echo "Share this file for debugging."
