"""
Automatic resource-failure retry: detect OOM and timeout failures and resubmit
affected pipeline chain with doubled memory and time.

Reads overall_errors.json (produced by collect_errors.py), identifies retryable
failures (OOM, timeout), doubles --mem and --time in the affected Slurm scripts,
and resubmits the downstream dependency chain.

Usage:
    python auto_resubmit.py --experiment alexnet_cifar10
    python auto_resubmit.py --experiment alexnet_cifar10 --test
    python auto_resubmit.py --experiment alexnet_cifar10 --dry-run
"""

import os
import re
import sys
import json
import argparse
import subprocess
from collections import defaultdict
from datetime import datetime

from utils.atomic_io import atomic_json_dump


# Error types that trigger automatic retry with resource doubling
RETRYABLE_ERROR_TYPES = {"OOM", "TIMEOUT", "CUDA_OOM"}

# Pipeline dependency graph: step -> set of upstream steps it depends on
DEPENDS_ON = {
    "A": set(),
    "B": {"A"},
    "C": {"A"},
    "G": {"A"},
    "D": {"A", "C"},
    "E": {"A", "B", "C", "D"},
    "F": {"E", "G"},
    "AUDIT": {"E", "G", "F"},
}

# Topological submission order
TOPO_ORDER = ["A", "B", "C", "G", "D", "E", "F", "AUDIT"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatic resource-failure retry with doubled memory and time"
    )
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--test", action="store_true", default=False,
                        help="Use test-mode directories")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Maximum auto retries (default: 2)")
    parser.add_argument("--mem-cap", type=int, default=480,
                        help="Memory cap in GB (default: 480)")
    parser.add_argument("--time-cap", type=str, default="48:00:00",
                        help="Time cap in HH:MM:SS (default: 48:00:00)")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print sbatch commands without executing")
    return parser.parse_args()


def parse_time_to_seconds(time_str):
    """Parse 'HH:MM:SS' to total seconds."""
    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS format, got: {time_str}")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + s


def seconds_to_time(total_seconds):
    """Format total seconds as 'HH:MM:SS'."""
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_errors(experiment):
    """Read overall_errors.json (new schema) and return (retryable_entries, all_entries).

    New schema top-level keys: experiment_name, last_updated, errors[]
    Each error entry has: job_id, error_type (OOM|CUDA_OOM|TIMEOUT|RUNTIME|UNKNOWN),
    phase (step letter), grid_index (chunk), timestamp, original_resources,
    retry_resources, resolved, message.

    Maps each entry to the internal format expected by the rest of auto_resubmit.py:
      - step        <- phase
      - chunk       <- grid_index
      - error_detected  <- not resolved
      - slurm_state     <- "COMPLETED" if resolved else "FAILED"
    """
    path = os.path.join("experiments", experiment, "overall_errors.json")
    if not os.path.isfile(path):
        print(f"No error report found at {path}")
        return [], []

    with open(path) as f:
        report = json.load(f)

    raw_entries = report.get("errors", [])
    all_entries = []
    for e in raw_entries:
        resolved = e.get("resolved", False)
        entry = dict(e)  # shallow copy so we don't mutate the original
        entry["step"] = e.get("phase")
        entry["chunk"] = e.get("grid_index")
        entry["error_detected"] = not resolved
        entry["slurm_state"] = "COMPLETED" if resolved else "FAILED"
        all_entries.append(entry)

    retryable_entries = [
        e for e in all_entries
        if e.get("error_detected") and e.get("error_type") in RETRYABLE_ERROR_TYPES
    ]
    return retryable_entries, all_entries


def update_retry_resources(experiment, job_id, new_mem_gb, new_time_hours):
    """Write retry_resources back into overall_errors.json for a specific job_id.

    Updates the entry in-place and atomically writes the file back.
    new_mem_gb: int, new memory in GB
    new_time_hours: float, new time in hours (converted to HH:MM:SS string)
    """
    path = os.path.join("experiments", experiment, "overall_errors.json")
    if not os.path.isfile(path):
        print(f"  WARNING: Cannot update retry_resources — {path} not found")
        return

    with open(path) as f:
        report = json.load(f)

    total_seconds = int(new_time_hours * 3600)
    time_str = seconds_to_time(total_seconds)

    updated = False
    for entry in report.get("errors", []):
        if str(entry.get("job_id")) == str(job_id):
            entry["retry_resources"] = {
                "mem_gb": new_mem_gb,
                "time": time_str,
            }
            updated = True
            break

    if not updated:
        print(f"  WARNING: job_id {job_id} not found in {path} — retry_resources not written")
        return

    atomic_json_dump(report, path)


def _write_resubmit_status(experiment, submitted=0, max_retries_reached=False):
    """Write auto_resubmit_status.json for the relaunch sentinel."""
    status = {
        "submitted": submitted,
        "max_retries_reached": max_retries_reached,
        "timestamp": datetime.now().isoformat(),
    }
    path = os.path.join("experiments", experiment, "auto_resubmit_status.json")
    atomic_json_dump(status, path)


def check_retry_count(experiment, max_retries):
    """Read/increment retry counter. Returns current count (before increment).

    Aborts (sys.exit) if max retries already reached.
    Reads from auto_retry_count, with backward-compat fallback to oom_retry_count.
    """
    counter_path = os.path.join("experiments", experiment, "auto_retry_count")
    legacy_path = os.path.join("experiments", experiment, "oom_retry_count")

    current = 0
    # Try new counter first, then legacy
    if os.path.isfile(counter_path):
        try:
            with open(counter_path) as f:
                current = int(f.read().strip())
        except (ValueError, OSError):
            current = 0
    elif os.path.isfile(legacy_path):
        try:
            with open(legacy_path) as f:
                current = int(f.read().strip())
        except (ValueError, OSError):
            current = 0

    if current >= max_retries:
        print(f"Auto retry limit reached ({current}/{max_retries}). No further retries.")
        # Write status so sentinel knows auto_resubmit exhausted retries
        _write_resubmit_status(experiment, submitted=0, max_retries_reached=True)
        sys.exit(0)

    # Increment (always write to new counter)
    with open(counter_path, "w") as f:
        f.write(str(current + 1))

    print(f"Auto retry {current + 1}/{max_retries}")
    return current


def step_to_script(experiment, step, chunk, test_mode):
    """Map (step, chunk) to the Slurm script path in orchestrator_jobs/."""
    job_dir = os.path.join("experiments", experiment, "orchestrator_jobs")
    if test_mode:
        job_dir = os.path.join("experiments", experiment, "orchestrator_jobs_test")
        if not os.path.isdir(job_dir):
            job_dir = os.path.join("experiments", experiment, "orchestrator_jobs")

    if step == "A":
        return os.path.join(job_dir, "step_A.sh")
    elif step == "B":
        return os.path.join(job_dir, f"step_B_chunk_{chunk}.sh")
    elif step == "C":
        return os.path.join(job_dir, f"step_C_attack_{chunk}.sh")
    elif step == "D":
        return os.path.join(job_dir, f"step_D_chunk_{chunk}.sh")
    elif step == "E":
        return os.path.join(job_dir, "step_E.sh")
    elif step == "G":
        return os.path.join(job_dir, "step_G.sh")
    elif step == "F":
        return os.path.join(job_dir, "step_F.sh")
    elif step == "AUDIT":
        return os.path.join(job_dir, "final_audit.sh")
    else:
        return None


def double_memory(script_path, mem_cap_gb):
    """Double the --mem value in a Slurm script, capped at mem_cap_gb."""
    with open(script_path) as f:
        content = f.read()

    def replacer(m):
        old_val = int(m.group(1))
        new_val = min(old_val * 2, mem_cap_gb)
        return f"#SBATCH --mem={new_val}G"

    new_content, count = re.subn(r"#SBATCH --mem=(\d+)G", replacer, content)
    if count == 0:
        print(f"  WARNING: No --mem=...G found in {script_path}")
        return None

    with open(script_path, "w") as f:
        f.write(new_content)

    # Extract old and new values for reporting
    old_match = re.search(r"#SBATCH --mem=(\d+)G", content)
    new_match = re.search(r"#SBATCH --mem=(\d+)G", new_content)
    old_val = old_match.group(1) if old_match else "?"
    new_val = new_match.group(1) if new_match else "?"
    return (old_val, new_val)


def double_time(script_path, time_cap_seconds):
    """Double the --time value in a Slurm script, capped at time_cap_seconds."""
    with open(script_path) as f:
        content = f.read()

    match = re.search(r"#SBATCH --time=(\d{2}):(\d{2}):(\d{2})", content)
    if not match:
        print(f"  WARNING: No --time=HH:MM:SS found in {script_path}")
        return None

    old_h, old_m, old_s = int(match.group(1)), int(match.group(2)), int(match.group(3))
    old_seconds = old_h * 3600 + old_m * 60 + old_s
    new_seconds = min(old_seconds * 2, time_cap_seconds)
    old_str = f"{old_h:02d}:{old_m:02d}:{old_s:02d}"
    new_str = seconds_to_time(new_seconds)

    new_content = content.replace(
        f"#SBATCH --time={match.group(1)}:{match.group(2)}:{match.group(3)}",
        f"#SBATCH --time={new_str}",
    )

    with open(script_path, "w") as f:
        f.write(new_content)

    return (old_str, new_str)


def get_retry_set(failed_entries, all_entries):
    """Compute the set of (step, chunk) pairs that need resubmission.

    Includes the failed jobs plus all transitive downstream steps.
    Returns: (failed_pairs, downstream_pairs, affected_steps)
      - failed_pairs: set of (step, chunk) that failed (need resource doubling)
      - downstream_pairs: set of (step, chunk) downstream (resubmit as-is)
      - affected_steps: set of step letters in the retry
    """
    # Step-level set of failed steps
    failed_steps = {e["step"] for e in failed_entries}
    failed_pairs = {(e["step"], e.get("chunk")) for e in failed_entries}

    # Propagate downstream: any step whose upstream intersects affected set
    affected_steps = set(failed_steps)
    for step in TOPO_ORDER:
        if step in affected_steps:
            continue
        upstream = DEPENDS_ON.get(step, set())
        if upstream & affected_steps:
            affected_steps.add(step)

    # Build downstream pairs from all_entries
    # For downstream steps: include jobs that didn't complete successfully
    downstream_pairs = set()
    completed_steps = set()
    for e in all_entries:
        if e.get("slurm_state") == "COMPLETED" and not e.get("error_detected"):
            completed_steps.add((e["step"], e.get("chunk")))

    for step in affected_steps:
        if step in failed_steps:
            continue  # Failed steps are handled via failed_pairs
        # For downstream steps, find their entries
        step_entries = [e for e in all_entries if e["step"] == step]
        if step_entries:
            for e in step_entries:
                pair = (e["step"], e.get("chunk"))
                if pair not in completed_steps:
                    downstream_pairs.add(pair)
        else:
            # Step never ran (e.g., was cancelled) -- submit with no chunk
            downstream_pairs.add((step, None))

    return failed_pairs, downstream_pairs, affected_steps


def submit_job(script_path, dep_ids, dry_run):
    """Submit a Slurm job, optionally with dependencies. Returns job ID."""
    cmd = ["sbatch", "--parsable"]
    if dep_ids:
        dep_str = ":".join(dep_ids)
        cmd.append(f"--dependency=afterok:{dep_str}")
    cmd.append(script_path)

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return f"DRY_{os.path.basename(script_path)}"

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR submitting {script_path}: {result.stderr.strip()}")
        return None
    job_id = result.stdout.strip().split(";")[0]  # handle array job output
    return job_id


def submit_retry_chain(experiment, failed_pairs, downstream_pairs, affected_steps,
                       mem_cap_gb, time_cap_seconds, test_mode, dry_run,
                       failed_entries_ref=None):
    """Submit the retry chain in topological order with correct dependencies.

    failed_entries_ref: list of retryable error entries (from load_errors) used
    to look up job_id for each (step, chunk) pair so retry_resources can be
    written back to overall_errors.json via update_retry_resources().
    """
    # Build a lookup from (step, chunk) -> job_id for retryable entries
    failed_job_lookup = {}
    if failed_entries_ref:
        for e in failed_entries_ref:
            key = (e.get("step"), e.get("chunk"))
            failed_job_lookup[key] = e.get("job_id")

    # Merge all pairs for lookup
    all_retry = {}  # step -> list of (step, chunk, is_failed)
    for step, chunk in failed_pairs:
        all_retry.setdefault(step, []).append((step, chunk, True))
    for step, chunk in downstream_pairs:
        all_retry.setdefault(step, []).append((step, chunk, False))

    new_job_ids = defaultdict(list)  # step -> [job_ids]
    all_submitted = []

    for step in TOPO_ORDER:
        entries = all_retry.get(step)
        if not entries:
            continue

        # Build dependency list from upstream retries only
        dep_ids = []
        for upstream in DEPENDS_ON.get(step, set()):
            if upstream in new_job_ids:
                dep_ids.extend(new_job_ids[upstream])

        for _, chunk, is_failed in entries:
            script = step_to_script(experiment, step, chunk, test_mode)
            if script is None or not os.path.isfile(script):
                print(f"  WARNING: Script not found for step {step} chunk {chunk}: {script}")
                continue

            new_mem_gb = None
            new_time_hours = None
            if is_failed:
                mem_info = double_memory(script, mem_cap_gb)
                if mem_info:
                    print(f"  [{step}] Doubled memory: {mem_info[0]}G -> {mem_info[1]}G ({os.path.basename(script)})")
                    new_mem_gb = int(mem_info[1])
                time_info = double_time(script, time_cap_seconds)
                if time_info:
                    print(f"  [{step}] Doubled time: {time_info[0]} -> {time_info[1]} ({os.path.basename(script)})")
                    new_time_hours = parse_time_to_seconds(time_info[1]) / 3600.0

                # Write retry_resources back to overall_errors.json
                orig_job_id = failed_job_lookup.get((step, chunk))
                if orig_job_id is not None and new_mem_gb is not None and new_time_hours is not None:
                    update_retry_resources(experiment, orig_job_id, new_mem_gb, new_time_hours)

            job_id = submit_job(script, dep_ids, dry_run)
            if job_id:
                new_job_ids[step].append(job_id)
                all_submitted.append(job_id)
                action = "resource-retry" if is_failed else "downstream"
                chunk_str = f" chunk={chunk}" if chunk is not None else ""
                print(f"  [{step}] Submitted {action}{chunk_str}: {job_id}")

    # Submit error_scan with afterany on ALL retry jobs
    if all_submitted:
        errscan_script = step_to_script(experiment, "ERRSCAN", None, test_mode)
        # error_scan.sh lives directly in orchestrator_jobs
        job_dir = os.path.join("experiments", experiment, "orchestrator_jobs")
        if test_mode:
            test_dir = os.path.join("experiments", experiment, "orchestrator_jobs_test")
            if os.path.isdir(test_dir):
                job_dir = test_dir
        errscan_script = os.path.join(job_dir, "error_scan.sh")

        if os.path.isfile(errscan_script):
            cmd = ["sbatch", "--parsable",
                   f"--dependency=afterany:{':'.join(all_submitted)}",
                   errscan_script]
            if dry_run:
                print(f"  [DRY-RUN] {' '.join(cmd)}")
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    scan_id = result.stdout.strip().split(";")[0]
                    print(f"  [ERRSCAN] Resubmitted error scan: {scan_id} (afterany)")
                    # Chain the relaunch sentinel after error scan
                    sentinel_script = os.path.join(job_dir, "sentinel.sh")
                    if os.path.isfile(sentinel_script):
                        sentinel_cmd = ["sbatch", "--parsable",
                                        f"--dependency=afterany:{scan_id}",
                                        sentinel_script]
                        sresult = subprocess.run(sentinel_cmd, capture_output=True, text=True)
                        if sresult.returncode == 0:
                            sentinel_id = sresult.stdout.strip().split(";")[0]
                            print(f"  [SENTINEL] Resubmitted sentinel: {sentinel_id} (afterany on error scan)")
                        else:
                            print(f"  WARNING: Failed to resubmit sentinel: {sresult.stderr.strip()}")
                else:
                    print(f"  WARNING: Failed to resubmit error_scan: {result.stderr.strip()}")
        else:
            print(f"  WARNING: error_scan.sh not found at {errscan_script}")

    return all_submitted


def main():
    args = parse_args()
    experiment = args.experiment
    time_cap_seconds = parse_time_to_seconds(args.time_cap)

    print(f"Auto retry check for experiment: {experiment}")

    # Clear stale status from prior cycle so sentinel doesn't read old data
    _write_resubmit_status(experiment, submitted=0)

    # Load errors
    retryable_entries, all_entries = load_errors(experiment)
    if not retryable_entries:
        print("No retryable failures (OOM/timeout) detected. Nothing to retry.")
        _write_resubmit_status(experiment, submitted=0)
        return

    print(f"Found {len(retryable_entries)} retryable failure(s):")
    for e in retryable_entries:
        chunk_str = f" chunk={e.get('chunk')}" if e.get("chunk") is not None else ""
        print(f"  Step {e['step']}{chunk_str} (job {e['job_id']}, type={e['error_type']})")

    # Check retry count
    check_retry_count(experiment, args.max_retries)

    # Compute retry set
    failed_pairs, downstream_pairs, affected_steps = get_retry_set(retryable_entries, all_entries)
    print(f"Affected steps: {sorted(affected_steps)}")
    if downstream_pairs:
        print(f"Downstream resubmissions: {sorted(downstream_pairs)}")

    # Submit retry chain
    submitted = submit_retry_chain(
        experiment, failed_pairs, downstream_pairs, affected_steps,
        args.mem_cap, time_cap_seconds, args.test, args.dry_run,
        failed_entries_ref=retryable_entries,
    )

    if submitted:
        print(f"Resubmitted {len(submitted)} job(s).")
    else:
        print("No jobs were submitted.")

    # Write status for relaunch sentinel
    _write_resubmit_status(experiment, submitted=len(submitted) if submitted else 0)


if __name__ == "__main__":
    main()
