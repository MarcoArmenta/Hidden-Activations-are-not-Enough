"""
Relaunch Sentinel: check if the pipeline needs whole-pipeline re-invocation
after the error scan and auto_resubmit cycle.

Submitted as a lightweight CPU Slurm job with afterany dependency on the
error scan job. If retryable errors (OOM/TIMEOUT) remain and the cycle
count is below the cap, re-invokes run_experiment.sh to restart the
pipeline (which skips completed steps and doubles resources for failures).

Usage:
    python relaunch_sentinel.py --experiment alexnet_cifar10
    python relaunch_sentinel.py --experiment alexnet_cifar10 --test --skip-audit
    python relaunch_sentinel.py --experiment alexnet_cifar10 --max-cycles 3
    python relaunch_sentinel.py --experiment alexnet_cifar10 --dry-run
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

from utils.atomic_io import atomic_json_dump

RETRYABLE_ERROR_TYPES = {"OOM", "TIMEOUT", "CUDA_OOM"}
DEFAULT_MAX_CYCLES = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Relaunch sentinel: cyclical pipeline re-invocation on retryable errors"
    )
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--test", action="store_true", default=False,
                        help="Pass --test to re-invoked run_experiment.sh")
    parser.add_argument("--skip-audit", action="store_true", default=False,
                        help="Pass --skip-audit to re-invoked run_experiment.sh")
    parser.add_argument("--max-cycles", type=int,
                        default=int(os.environ.get("MAX_SENTINEL_CYCLES", DEFAULT_MAX_CYCLES)),
                        help=f"Maximum re-launch cycles (default: {DEFAULT_MAX_CYCLES})")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print re-launch command without executing")
    return parser.parse_args()


def read_cycle_count(experiment):
    path = os.path.join("experiments", experiment, "checkpoints", "cycle_count.txt")
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError, OSError):
        return 0


def write_cycle_count(experiment, count):
    path = os.path.join("experiments", experiment, "checkpoints", "cycle_count.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(count))
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


def read_auto_resubmit_status(experiment):
    path = os.path.join("experiments", experiment, "auto_resubmit_status.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_unresolved_errors(experiment):
    """Load overall_errors.json and split unresolved errors into retryable vs non-retryable."""
    path = os.path.join("experiments", experiment, "overall_errors.json")
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: Cannot read overall_errors.json: {exc}")
        return [], [], None

    errors = data.get("errors", [])
    unresolved = [e for e in errors if not e.get("resolved", False)]
    retryable = [e for e in unresolved if e.get("error_type") in RETRYABLE_ERROR_TYPES]
    non_retryable = [e for e in unresolved if e.get("error_type") not in RETRYABLE_ERROR_TYPES]
    return retryable, non_retryable, data


def update_sentinel_metadata(experiment, data, metadata):
    if data is None:
        return
    data["_sentinel"] = metadata
    path = os.path.join("experiments", experiment, "overall_errors.json")
    atomic_json_dump(data, path)


def main():
    args = parse_args()
    exp = args.experiment

    print(f"=== Relaunch Sentinel for {exp} ===")
    print(f"Max cycles: {args.max_cycles}")

    # Step 1: Check if auto_resubmit already took action
    resubmit_status = read_auto_resubmit_status(exp)
    if resubmit_status and resubmit_status.get("submitted", 0) > 0:
        n = resubmit_status["submitted"]
        print(f"auto_resubmit.py submitted {n} job(s) in this cycle.")
        print("Deferring to auto_resubmit's error scan chain. Exiting.")
        return

    # Step 2: Load errors
    retryable, non_retryable, data = load_unresolved_errors(exp)

    if not retryable:
        if non_retryable:
            types = sorted(set(e.get("error_type", "UNKNOWN") for e in non_retryable))
            print(f"No retryable errors. {len(non_retryable)} non-retryable error(s) (types: {types}).")
            print("Manual intervention required for non-retryable errors.")
            update_sentinel_metadata(exp, data, {
                "action": "stopped_non_retryable",
                "non_retryable_count": len(non_retryable),
                "non_retryable_types": types,
                "timestamp": datetime.now().isoformat(),
            })
        else:
            print("No unresolved errors. Pipeline complete!")
            update_sentinel_metadata(exp, data, {
                "action": "pipeline_complete",
                "timestamp": datetime.now().isoformat(),
            })
        return

    # Step 3: Check cycle count
    cycle = read_cycle_count(exp)
    max_cycles = args.max_cycles

    print(f"Found {len(retryable)} retryable error(s), cycle {cycle}/{max_cycles}:")
    for e in retryable:
        phase = e.get("phase", "?")
        chunk = e.get("grid_index")
        chunk_str = f" chunk={chunk}" if chunk is not None else ""
        etype = e.get("error_type", "?")
        print(f"  - Step {phase}{chunk_str}: {etype}")

    if non_retryable:
        types = sorted(set(e.get("error_type", "UNKNOWN") for e in non_retryable))
        print(f"Also {len(non_retryable)} non-retryable error(s) (types: {types}) — logged but not triggering re-launch.")

    if cycle >= max_cycles:
        print(f"Maximum cycle count reached ({cycle}/{max_cycles}). Stopping.")
        update_sentinel_metadata(exp, data, {
            "action": "stopped_at_max_cycles",
            "cycle": cycle,
            "max_cycles": max_cycles,
            "retryable_remaining": len(retryable),
            "timestamp": datetime.now().isoformat(),
        })
        return

    # Step 4: Re-invoke pipeline
    new_cycle = cycle + 1
    write_cycle_count(exp, new_cycle)

    cmd = ["bash", "run_experiment.sh"]
    if args.skip_audit:
        cmd.append("--skip-audit")
    if args.test:
        cmd.append("--test")
    cmd.append(exp)

    print(f"Re-launching pipeline (cycle {new_cycle}/{max_cycles}): {' '.join(cmd)}")

    update_sentinel_metadata(exp, data, {
        "action": "relaunching",
        "cycle": new_cycle,
        "max_cycles": max_cycles,
        "retryable_remaining": len(retryable),
        "command": " ".join(cmd),
        "timestamp": datetime.now().isoformat(),
    })

    # Clear stale auto_resubmit status before re-invoking pipeline
    status_path = os.path.join("experiments", exp, "auto_resubmit_status.json")
    if os.path.isfile(status_path):
        os.remove(status_path)

    if args.dry_run:
        print(f"[DRY-RUN] Would execute: {' '.join(cmd)}")
        return

    # Execute from SLURM_SUBMIT_DIR (or current directory)
    cwd = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())
    result = subprocess.run(cmd, cwd=cwd)

    if result.returncode != 0:
        print(f"WARNING: run_experiment.sh exited with code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"Pipeline re-launched successfully (cycle {new_cycle}/{max_cycles}).")


if __name__ == "__main__":
    main()
