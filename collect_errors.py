"""
Collect pipeline errors into experiments/{experiment}/overall_errors.json.

Replaces the inline Python previously embedded in run_experiment.sh.
Can run as a Slurm job or directly on the login node.

Usage:
    python collect_errors.py --experiment alexnet_cifar10
    python collect_errors.py --experiment alexnet_cifar10 --test
    python collect_errors.py --experiment alexnet_cifar10 --include-audit-report
"""

import os
import re
import sys
import json
import glob
import argparse
import subprocess
from datetime import datetime

from utils.error_classification import (
    LOG_PATTERN, STEP_LABELS, STEP_ORDER,
    ERROR_CATEGORIES,
    classify_error, get_error_category, extract_traceback, read_tail,
    parse_log_filename, normalize_error_type,
)
from utils.atomic_io import atomic_json_dump


SCHEMA_VERSION = "3.0"

# Patterns that indicate runtime errors in .out files of COMPLETED jobs
RUNTIME_ERROR_RE = re.compile(
    r'ERROR:|FAILED:|Traceback|RuntimeError|AttributeError|urllib\.error|CUDA error|OOM:'
)

# Regex for extracting --mem=<N>G from Slurm scripts
_MEM_RE = re.compile(r'--mem=(\d+(?:\.\d+)?)G', re.I)
# Regex for extracting --time=HH:MM:SS from Slurm scripts
_TIME_RE = re.compile(r'--time=(\d+):(\d+):(\d+)')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect pipeline errors into overall_errors.json"
    )
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--test", action="store_true", default=False,
                        help="Use test-mode log directories")
    parser.add_argument("--include-audit-report", action="store_true", default=False,
                        help="Merge integrity summary from audit_report.json")
    return parser.parse_args()


# ── Slurm script resource parsing ────────────────────────────────────────

def _find_slurm_script(experiment, step, chunk=None):
    """Locate the Slurm script for a given step/chunk.

    Searches experiments/{experiment}/orchestrator_jobs/ and
    experiments/{experiment}/recovery_jobs/ for scripts matching
    the step and optional chunk/attack suffix.
    """
    # Search orchestrator_jobs first, then recovery_jobs (fallback)
    search_dirs = [
        os.path.join("experiments", experiment, "orchestrator_jobs"),
        os.path.join("experiments", experiment, "recovery_jobs"),
    ]

    # Build candidate filenames
    candidates = []
    if chunk is not None:
        # Chunked steps (B, D) use step_B_chunk_0.sh
        candidates.append(f"step_{step}_chunk_{chunk}.sh")
        # Attack steps (C) use step_C_attack_FGSM.sh
        candidates.append(f"step_{step}_attack_{chunk}.sh")
        # Array jobs: step_B.sh (single script for all chunks)
        candidates.append(f"step_{step}.sh")
    else:
        candidates.append(f"step_{step}.sh")
        # Recovery audit script uses a different name
        if step == "AUDIT":
            candidates.append("final_audit.sh")

    for script_dir in search_dirs:
        if not os.path.isdir(script_dir):
            continue
        for candidate in candidates:
            path = os.path.join(script_dir, candidate)
            if os.path.isfile(path):
                return path
    return None


def _parse_mem_from_step(experiment, step, chunk=None):
    """Read the Slurm script for a step and extract --mem=XG as float GB.

    Returns None if the script is not found or memory is not specified.
    """
    script = _find_slurm_script(experiment, step, chunk)
    if not script:
        return None
    try:
        with open(script) as f:
            content = f.read()
        m = _MEM_RE.search(content)
        if m:
            return float(m.group(1))
    except OSError:
        pass
    return None


def _parse_time_from_step(experiment, step, chunk=None):
    """Read the Slurm script for a step and extract --time=HH:MM:SS as float hours.

    Returns None if the script is not found or time is not specified.
    """
    script = _find_slurm_script(experiment, step, chunk)
    if not script:
        return None
    try:
        with open(script) as f:
            content = f.read()
        m = _TIME_RE.search(content)
        if m:
            hours = int(m.group(1))
            minutes = int(m.group(2))
            seconds = int(m.group(3))
            return round(hours + minutes / 60.0 + seconds / 3600.0, 4)
    except OSError:
        pass
    return None


# ── Job discovery and analysis ───────────────────────────────────────────

def discover_jobs(experiment, slurm_out_dir, slurm_err_dir):
    """Discover all pipeline jobs for this experiment from log filenames.

    Scans both .out and .err directories to find all job IDs.
    Returns a list of job dicts with associated file paths.
    """
    jobs = {}  # keyed by job_id to deduplicate

    for directory, ext in [(slurm_err_dir, "err"), (slurm_out_dir, "out")]:
        if not os.path.isdir(directory):
            continue
        for fname in os.listdir(directory):
            parsed = parse_log_filename(fname)
            if not parsed:
                continue
            if parsed["exp"] != experiment:
                # Step 2b logs include the attack name in the filename:
                #   PIPE_2b_alexnet_cifar10_FGSM_12345.out
                # The regex parses exp="alexnet_cifar10_FGSM" instead of "alexnet_cifar10".
                # Detect this case and store the attack suffix as the chunk.
                if parsed["step"] == "C" and parsed["exp"].startswith(experiment + "_"):
                    attack_suffix = parsed["exp"][len(experiment) + 1:]
                    parsed["exp"] = experiment
                    parsed["chunk"] = attack_suffix
                else:
                    continue
            if parsed["step"] == "ERRSCAN":
                continue  # skip our own logs

            job_id = parsed["job_id"]
            if job_id not in jobs:
                jobs[job_id] = {
                    "job_id": job_id,
                    "prefix": parsed["prefix"],
                    "step": parsed["step"],
                    "chunk": int(parsed["chunk"]) if parsed["chunk"] and parsed["chunk"].isdigit() else parsed["chunk"],
                    "err_file": None,
                    "out_file": None,
                }

            fpath = os.path.join(directory, fname)
            if parsed["ext"] == "err":
                jobs[job_id]["err_file"] = fpath
            elif parsed["ext"] == "out":
                jobs[job_id]["out_file"] = fpath

    return list(jobs.values())


def deduplicate_jobs(jobs):
    """Keep only the most recent job per (step, chunk) group."""
    groups = {}
    for job in jobs:
        key = (job["step"], job["chunk"])
        existing = groups.get(key)
        if existing is None or int(job["job_id"]) > int(existing["job_id"]):
            groups[key] = job
    return list(groups.values())


def query_sacct(job_ids):
    """Query sacct for job status. Returns dict keyed by job_id.

    Handles sacct being unavailable (e.g., on login nodes without Slurm).
    """
    if not job_ids:
        return {}
    try:
        ids_str = ",".join(job_ids)
        result = subprocess.run(
            ["sacct", "--jobs=" + ids_str, "--parsable2", "--noheader",
             "--format=JobID,State,ExitCode,Elapsed"],
            capture_output=True, text=True, timeout=30,
        )
        data = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 4:
                jid = parts[0].split(".")[0]  # strip .batch suffix
                if jid in job_ids and jid not in data:
                    data[jid] = {
                        "state": parts[1],
                        "exit_code": parts[2],
                        "elapsed": parts[3],
                    }
        return data
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}


def detect_error(job, sacct_info):
    """Determine if a job has an error and classify it.

    Returns (has_error, error_type, error_category, traceback, err_tail, source_file).
    """
    slurm_state = sacct_info.get("state", "UNKNOWN")
    exit_code = sacct_info.get("exit_code", "?")

    has_error = False
    source_file = None

    # Check Slurm state
    if slurm_state in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY"):
        has_error = True

    # Check non-zero exit code
    if not has_error and exit_code not in ("0:0", "?") and ":" in exit_code:
        try:
            main_code = int(exit_code.split(":")[0])
            if main_code != 0:
                has_error = True
        except ValueError:
            pass

    # For failed jobs: scan .err file
    if has_error:
        tail_text = ""
        if job.get("err_file"):
            tail_text = read_tail(job["err_file"])
            source_file = job["err_file"]
        error_type = classify_error(tail_text)
        # Slurm OUT_OF_MEMORY kills may leave .err empty; force oom classification
        if slurm_state == "OUT_OF_MEMORY":
            error_type = "oom"
        # Slurm TIMEOUT may leave .err empty; force timeout classification
        if slurm_state == "TIMEOUT":
            error_type = "timeout"
        traceback = extract_traceback(tail_text)
        if error_type is None:
            error_type = "code" if traceback else "unknown"
        return True, error_type, get_error_category(error_type), traceback, tail_text, source_file

    # For COMPLETED/UNKNOWN jobs: scan both .err and .out for runtime errors
    for file_key in ["err_file", "out_file"]:
        fpath = job.get(file_key)
        if not fpath or not os.path.isfile(fpath):
            continue
        content = read_tail(fpath, 50)
        if RUNTIME_ERROR_RE.search(content):
            error_type = classify_error(content)
            traceback = extract_traceback(content)
            if error_type is None:
                error_type = "code" if traceback else "unknown"
            return True, error_type, get_error_category(error_type), traceback, content, fpath

    return False, None, None, None, None, None


def load_audit_report(experiment):
    """Load audit_report.json if it exists."""
    path = os.path.join("experiments", experiment, "audit_report.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def build_integrity_summary(audit_report):
    """Extract integrity summary from audit_report.json."""
    if not audit_report:
        return None
    summary = audit_report.get("summary", {})
    return {
        "total_checks": summary.get("total", 0),
        "ok": summary.get("ok", 0),
        "missing": summary.get("missing", 0),
        "corrupt": summary.get("corrupt", 0),
    }


# ── Append-mode merge ────────────────────────────────────────────────────

def _load_existing_errors(out_path):
    """Load existing overall_errors.json for append-mode merge.

    Returns the existing data dict, or None if file does not exist or is invalid.
    """
    if not os.path.isfile(out_path):
        return None
    try:
        with open(out_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("errors"), list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _merge_errors(existing_errors, new_errors):
    """Merge new errors into existing list by job_id (newer replaces older)."""
    merged = {e["job_id"]: e for e in existing_errors}
    for entry in new_errors:
        merged[entry["job_id"]] = entry
    return list(merged.values())


# ── Error message extraction ─────────────────────────────────────────────

def _build_error_message(err_tail, traceback_text):
    """Build a concise error message string from tail/traceback.

    Returns the traceback if available, otherwise the last 5 lines of err_tail,
    or a generic message.
    """
    if traceback_text:
        return traceback_text.strip()
    if err_tail:
        lines = err_tail.strip().split("\n")
        return "\n".join(lines[-5:])
    return "No error details available"


def main():
    args = parse_args()
    experiment = args.experiment
    test_mode = args.test

    slurm_out_dir = "slurm_out_test" if test_mode else "slurm_out"
    slurm_err_dir = "slurm_err_test" if test_mode else "slurm_err"
    output_dir = os.path.join("experiments", experiment)

    # Discover jobs
    jobs = discover_jobs(experiment, slurm_out_dir, slurm_err_dir)
    jobs = deduplicate_jobs(jobs)

    # Query sacct
    job_ids = {j["job_id"] for j in jobs}
    sacct_data = query_sacct(job_ids)

    # Analyze each job
    steps_output = []
    error_types = {}
    error_category_summary = {cat: 0 for cat in ERROR_CATEGORIES}
    jobs_with_errors = 0
    jobs_succeeded = 0

    # Collect enforced-schema error entries
    new_errors = []

    now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    for job in jobs:
        jid = job["job_id"]
        info = sacct_data.get(jid, {})
        slurm_state = info.get("state", "UNKNOWN")
        exit_code = info.get("exit_code", "?")
        elapsed = info.get("elapsed", "?")
        step = job["step"]
        step_label = STEP_LABELS.get(step, step)

        has_error, error_type, error_category, traceback, err_tail, source_file = \
            detect_error(job, info)

        entry = {
            "step": step,
            "step_label": step_label,
            "chunk": job["chunk"],
            "job_id": jid,
            "slurm_state": slurm_state,
            "exit_code": exit_code,
            "elapsed": elapsed,
            "error_detected": has_error,
        }

        if has_error:
            jobs_with_errors += 1
            entry["error_type"] = error_type
            entry["error_category"] = error_category
            if traceback:
                entry["traceback"] = traceback
            if err_tail:
                # Truncate to last 40 lines to keep JSON manageable
                tail_lines = err_tail.strip().split("\n")
                entry["err_tail"] = "\n".join(tail_lines[-40:])
            if source_file:
                entry["err_file"] = os.path.relpath(
                    source_file, os.environ.get("SLURM_SUBMIT_DIR", ".")
                )
            error_types[error_type] = error_types.get(error_type, 0) + 1
            error_category_summary[error_category] = \
                error_category_summary.get(error_category, 0) + 1

            # Build enforced-schema error entry
            mem_gb = _parse_mem_from_step(experiment, step, job["chunk"])
            time_hours = _parse_time_from_step(experiment, step, job["chunk"])
            new_errors.append({
                "job_id": jid,
                "error_type": normalize_error_type(error_type),
                "phase": step,
                "grid_index": job["chunk"],
                "timestamp": now_iso,
                "original_resources": {
                    "memory_gb": mem_gb,
                    "time_hours": time_hours,
                },
                "retry_resources": {
                    "memory_gb": None,
                    "time_hours": None,
                },
                "resolved": False,
                "message": _build_error_message(err_tail, traceback),
            })
        else:
            if slurm_state == "COMPLETED":
                jobs_succeeded += 1

        steps_output.append(entry)

    # Sort: errors first, then by step order
    def sort_key(e):
        idx = STEP_ORDER.index(e["step"]) if e["step"] in STEP_ORDER else 99
        return (0 if e["error_detected"] else 1, idx, e.get("chunk") or 0)

    steps_output.sort(key=sort_key)

    pipeline_success = jobs_with_errors == 0

    # Build backward-compat summary
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": "test" if test_mode else "normal",
        "pipeline_success": pipeline_success,
        "total_jobs": len(jobs),
        "jobs_succeeded": jobs_succeeded,
        "jobs_with_errors": jobs_with_errors,
        "error_category_summary": error_category_summary,
        "error_types_summary": error_types,
    }

    # Optionally include integrity summary from audit_report.json
    if args.include_audit_report:
        audit_report = load_audit_report(experiment)
        integrity = build_integrity_summary(audit_report)
        if integrity:
            summary["integrity"] = integrity

    # Append-mode: merge with existing errors
    out_path = os.path.join(output_dir, "overall_errors.json")
    existing_data = _load_existing_errors(out_path)
    if existing_data and isinstance(existing_data.get("errors"), list):
        merged_errors = _merge_errors(existing_data["errors"], new_errors)
    else:
        merged_errors = new_errors

    # Mark errors as resolved if their step+chunk completed in current scan
    completed_pairs = set()
    for entry in steps_output:
        if not entry.get("error_detected"):
            completed_pairs.add((entry["step"], entry.get("chunk")))

    for error in merged_errors:
        pair = (error.get("phase"), error.get("grid_index"))
        if pair in completed_pairs:
            error["resolved"] = True

    # Build enforced-schema report
    report = {
        "experiment_name": experiment,
        "last_updated": now_iso,
        "errors": merged_errors,
        "_steps_detail": steps_output,
        "_summary": summary,
    }

    # Write output atomically
    os.makedirs(output_dir, exist_ok=True)
    atomic_json_dump(report, out_path)

    # Print summary
    status_str = "SUCCESS" if pipeline_success else "FAILURE"
    print(f"Error scan complete: {status_str}")
    print(f"  Total jobs scanned: {len(jobs)}")
    print(f"  Jobs succeeded:     {jobs_succeeded}")
    print(f"  Jobs with errors:   {jobs_with_errors}")
    if error_types:
        print(f"  Error types:        {error_types}")
    if any(v > 0 for v in error_category_summary.values()):
        nonzero = {k: v for k, v in error_category_summary.items() if v > 0}
        print(f"  Error categories:   {nonzero}")
    print(f"  Report written to:  {out_path}")


if __name__ == "__main__":
    main()
