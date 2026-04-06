"""
Pipeline Report — Comprehensive post-run analysis for Slurm ML pipelines.

Generates a timestamped, non-overwriting report covering:
  0. Quick pass/fail summary (from overall_errors.json if available)
  1. Job status summary (from sacct + log filenames)
  2. Calibration results
  3. Pipeline step completion (artifact verification)
  4. Error analysis (categorized failures)
  5. GPU monitoring summary
  6. Recovery recommendations

Usage:
    python pipeline_report.py --experiment alexnet_cifar10
    python pipeline_report.py --experiment alexnet_cifar10 --test --total-chunks 2
"""

import os
import sys
import re
import csv
import json
import glob
import subprocess
import argparse
from datetime import datetime

from utils.error_classification import (
    LOG_PATTERN, STEP_ORDER, STEP_LABELS_PREFIXED as STEP_LABELS,
    ERROR_PATTERNS, classify_error, get_error_category, read_tail,
    parse_log_filename,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a pipeline run report.")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--total-chunks", type=int, default=8)
    return parser.parse_args()


# ── overall_errors.json consumer ─────────────────────────────────────────

def load_overall_errors(experiment):
    """Load overall_errors.json if available.

    Returns a 2-tuple (report_dict, steps_list) where:
      - report_dict is a flat summary dict suitable for Section 0 display
        (contains keys like pipeline_success, total_jobs, …)
      - steps_list is a list of per-job step dicts for Section 4 error display

    Handles both the new schema (top-level errors[], _steps_detail, _summary)
    and any residual old-schema files (top-level steps[], pipeline_success, …).

    Returns (None, []) when the file is absent or unreadable.
    """
    path = os.path.join("experiments", experiment, "overall_errors.json")
    if not os.path.isfile(path):
        return None, []
    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, []

    # ── New schema: has _summary and/or _steps_detail keys ────────────────
    if "_summary" in raw or "_steps_detail" in raw:
        summary = raw.get("_summary", {})
        steps_list = raw.get("_steps_detail", [])

        # If _steps_detail is absent, reconstruct a minimal list from errors[]
        if not steps_list:
            steps_list = []
            for err in raw.get("errors", []):
                steps_list.append({
                    "job_id": err.get("job_id", "?"),
                    "step_label": err.get("step_label", err.get("step", "?")),
                    "chunk": err.get("chunk"),
                    "slurm_state": err.get("slurm_state", err.get("state", "?")),
                    "error_detected": True,
                    "error_type": err.get("error_type", "?"),
                    "error_category": err.get("error_category", "unknown"),
                    "traceback": err.get("traceback"),
                    "err_tail": err.get("err_tail", ""),
                })

        report_dict = dict(summary)  # shallow copy so callers can't mutate _summary
        # Propagate top-level fields that Section 0 may still want
        for key in ("schema_version", "error_scan_failed"):
            if key in raw and key not in report_dict:
                report_dict[key] = raw[key]
        return report_dict, steps_list

    # ── Old schema: pipeline_success / steps / … at top level ─────────────
    steps_list = raw.get("steps", [])
    # Build a summary dict that mirrors what the new schema puts in _summary
    report_dict = {k: v for k, v in raw.items() if k != "steps"}
    return report_dict, steps_list


# ── Section 1: Job Status ────────────────────────────────────────────────

def discover_jobs(experiment, slurm_out_dir):
    """Parse log filenames to find all jobs for this experiment."""
    jobs = []
    if not os.path.isdir(slurm_out_dir):
        return jobs
    for fname in os.listdir(slurm_out_dir):
        parsed = parse_log_filename(fname)
        if not parsed:
            continue
        if parsed["exp"] != experiment:
            continue
        jobs.append({
            "job_id": parsed["job_id"], "prefix": parsed["prefix"],
            "step": parsed["step"],
            "chunk": int(parsed["chunk"]) if parsed["chunk"] else None,
            "filename": fname,
        })
    return jobs


def query_sacct(job_ids):
    """Query sacct for job status. Returns dict keyed by job_id."""
    if not job_ids:
        return {}
    try:
        ids_str = ",".join(job_ids)
        result = subprocess.run(
            ["sacct", "--jobs=" + ids_str, "--parsable2", "--noheader",
             "--format=JobID,JobName%40,State,ExitCode,Elapsed,MaxRSS"],
            capture_output=True, text=True, timeout=30,
        )
        data = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 6:
                jid = parts[0].split(".")[0]  # strip .batch suffix
                if jid in job_ids:
                    data[jid] = {
                        "name": parts[1], "state": parts[2],
                        "exit_code": parts[3], "elapsed": parts[4],
                        "max_rss": parts[5],
                    }
        return data
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}


# ── Section 2: Calibration ───────────────────────────────────────────────

def read_calibration(experiment):
    path = os.path.join("experiments", experiment, "calibration.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Section 3: Step Completion ───────────────────────────────────────────

def check_step_completion(experiment, total_chunks):
    sys.path.insert(0, ".")
    from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS
    from utils.data_integrity import verify_experiment

    exp_config = DEFAULT_EXPERIMENTS.get(experiment, {})
    dataset = exp_config.get("dataset", "cifar10")
    num_classes = {"cifar100": 100, "imagenet": 1000}.get(dataset, 10)
    epochs = exp_config.get("epochs", exp_config.get("epoch", 1))
    exp_dir = os.path.join("experiments", experiment)

    # Step 1: trained weights
    weights_file = os.path.join(exp_dir, "weights", f"epoch_{epochs}.pth")
    step_a_ok = os.path.exists(weights_file)

    # Steps 2a-5 via verify_experiment
    report = None
    if os.path.isdir(exp_dir):
        report = verify_experiment(
            experiment_dir=exp_dir, experiment_name=experiment,
            num_classes=num_classes, num_samples_per_class=1000,
            total_chunks=total_chunks,
            num_samples_rejection_level=10000,
            attacks_list=ATTACKS, sample_ratio=0.0,
        )
    return step_a_ok, report, epochs


# ── Section 4: Error Analysis ────────────────────────────────────────────

def read_last_lines(filepath, n=50):
    """Read the last n lines of a file."""
    return read_tail(filepath, n)


def analyze_errors(jobs, slurm_out_dir, slurm_err_dir, sacct_data):
    """Fall-back error analysis when overall_errors.json is not available."""
    errors = []
    seen_jobs = set()

    for job in jobs:
        jid = job["job_id"]
        if jid in seen_jobs:
            continue
        seen_jobs.add(jid)

        info = sacct_data.get(jid, {})
        state = info.get("state", "UNKNOWN")
        step_label = STEP_LABELS.get(job["step"], job["step"])
        chunk_str = f" chunk {job['chunk']}" if job["chunk"] is not None else ""

        # For FAILED/TIMEOUT/CANCELLED jobs: scan .err files
        if state not in ("COMPLETED", "UNKNOWN"):
            err_pattern = os.path.join(slurm_err_dir, f"*_{jid}.err")
            err_files = glob.glob(err_pattern)
            last_lines = ""
            category = "Unknown"

            if err_files:
                last_lines = read_last_lines(err_files[0], 30)
                if not last_lines:
                    last_lines = "(could not read error file)"

                error_type = classify_error(last_lines)
                if error_type:
                    category = f"{error_type} ({get_error_category(error_type)})"

            errors.append({
                "job_id": jid, "step": f"{step_label}{chunk_str}",
                "state": state, "category": category,
                "last_lines": last_lines,
            })
            continue

        # For COMPLETED jobs: scan .err and .out files for runtime errors
        for ext, directory in [("err", slurm_err_dir), ("out", slurm_out_dir)]:
            pattern = os.path.join(directory, f"*_{jid}.{ext}")
            files = glob.glob(pattern)
            for f in files:
                content = read_last_lines(f, 50)
                if re.search(r'ERROR:|FAILED:|Traceback|RuntimeError|AttributeError|urllib\.error', content):
                    category = "Unknown"
                    error_type = classify_error(content)
                    if error_type:
                        category = f"{error_type} ({get_error_category(error_type)})"
                    errors.append({
                        "job_id": jid, "step": f"{step_label}{chunk_str}",
                        "state": f"{state} (Runtime Error)",
                        "category": category,
                        "last_lines": content,
                    })
                    break  # one error entry per job is enough
            else:
                continue
            break  # break outer loop if inner found an error

    return errors


# ── Section 5: GPU Monitoring ────────────────────────────────────────────

def parse_gpu_logs(experiment, total_chunks):
    gpu_dir = "gpu-monitor"
    results = {}
    patterns = [("Calibration", f"{experiment}.calibration.log")]
    for step in ["B", "D"]:
        for c in range(total_chunks):
            patterns.append((f"{step}.{c}", f"{experiment}.{step}.{c}.log"))

    for label, filename in patterns:
        filepath = os.path.join(gpu_dir, filename)
        if not os.path.exists(filepath):
            continue
        utils_list, mems_list = [], []
        try:
            with open(filepath) as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 4:
                        try:
                            utils_list.append(float(row[1].strip()))
                            mems_list.append(float(row[2].strip()))
                        except ValueError:
                            continue
        except Exception:
            continue
        if utils_list and mems_list:
            results[label] = {
                "peak_util": max(utils_list),
                "avg_util": sum(utils_list) / len(utils_list),
                "peak_mem_mib": max(mems_list),
                "samples": len(utils_list),
            }
    return results


# ── Section 6: Recovery Recommendations ──────────────────────────────────

def generate_recommendations(step_a_ok, integrity_report):
    recs = []
    if not step_a_ok:
        recs.append("Re-run Step 1 (Training) -- weights file missing")

    if integrity_report is None:
        recs.append("Experiment directory missing -- run full pipeline")
        return recs

    steps = integrity_report.get("steps", {})

    # Step 2a
    bad_b = []
    for i, e in enumerate(steps.get("matrices_zips", [])):
        if e["status"] != "OK":
            bad_b.append(str(i))
    if bad_b:
        recs.append(f"Re-run Step 2a chunks [{', '.join(bad_b)}] -- matrix zips {'/'.join(e['status'] for e in steps['matrices_zips'] if e['status']!='OK')}")

    # Step 2b
    bad_c = [os.path.basename(os.path.dirname(e["path"])) for e in steps.get("adversarial_examples", []) if e["status"] != "OK"]
    if bad_c:
        recs.append(f"Re-run Step 2b -- missing attacks: {', '.join(bad_c)}")

    # Step 3 (Adv Matrices)
    bad_d = []
    for i, e in enumerate(steps.get("adv_matrices_zips", [])):
        if e["status"] != "OK":
            bad_d.append(str(i))
    if bad_d:
        recs.append(f"Re-run Step 3 chunks [{', '.join(bad_d)}]")

    # Dependency propagation
    if bad_c and not bad_d:
        recs.append("  (propagated) Step 3 needs re-run due to Step 2b failure")

    return recs


# ── Report Writer ────────────────────────────────────────────────────────

def write_report(experiment, test_mode, total_chunks, jobs, sacct_data,
                 calibration, step_a_ok, integrity_report, epochs,
                 errors, gpu_data, recommendations, overall_errors,
                 overall_errors_steps=None):

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    mode_str = "test" if test_mode else "normal"

    lines = []
    w = lines.append

    w("=" * 70)
    w(f"  Pipeline Report: {experiment}")
    w(f"  Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  Mode: {mode_str}")
    w("=" * 70)
    w("")

    # --- Section 0: Quick Summary from overall_errors.json ---
    if overall_errors:
        pipeline_ok = overall_errors.get("pipeline_success", False)
        status_icon = "PASS" if pipeline_ok else "FAIL"
        w(f"=== Pipeline Status: {status_icon} ===")
        w(f"  (from overall_errors.json, schema v{overall_errors.get('schema_version', '?')})")
        w(f"  Total jobs:       {overall_errors.get('total_jobs', '?')}")
        w(f"  Jobs succeeded:   {overall_errors.get('jobs_succeeded', '?')}")
        w(f"  Jobs with errors: {overall_errors.get('jobs_with_errors', '?')}")
        cat_summary = overall_errors.get("error_category_summary", {})
        if any(v > 0 for v in cat_summary.values()):
            nonzero = {k: v for k, v in cat_summary.items() if v > 0}
            w(f"  Error categories: {nonzero}")
        integrity = overall_errors.get("integrity")
        if integrity:
            w(f"  Integrity:        {integrity.get('ok', 0)} OK, "
              f"{integrity.get('missing', 0)} missing, "
              f"{integrity.get('corrupt', 0)} corrupt")
        if overall_errors.get("error_scan_failed"):
            w("  WARNING: Error scan itself failed — data may be incomplete")
        w("")

    # --- Section 1: Job Status ---
    w("=== Section 1: Job Status Summary ===")
    w(f"  {'Step':<25} {'JobID':<12} {'State':<14} {'Exit':<8} {'Elapsed':<12} {'MaxRSS'}")
    w(f"  {'-'*25} {'-'*12} {'-'*14} {'-'*8} {'-'*12} {'-'*10}")

    # Group and sort jobs by step order
    step_groups = {}
    for job in jobs:
        step_groups.setdefault(job["step"], []).append(job)

    for step_key in STEP_ORDER:
        for job in step_groups.get(step_key, []):
            jid = job["job_id"]
            info = sacct_data.get(jid, {})
            state = info.get("state", "UNKNOWN")
            label = STEP_LABELS.get(step_key, step_key)
            chunk_str = f" c{job['chunk']}" if job["chunk"] is not None else ""
            marker = "  *** FAILED ***" if state in ("FAILED", "TIMEOUT", "CANCELLED") else ""
            w(f"  {label+chunk_str:<25} {jid:<12} {state:<14} {info.get('exit_code','?'):<8} {info.get('elapsed','?'):<12} {info.get('max_rss','?')}{marker}")

    if not jobs:
        w("  (no Slurm log files found)")
    w("")

    # --- Section 2: Calibration ---
    w("=== Section 2: Calibration Results ===")
    if calibration:
        gpu_util = calibration.get("peak_matrix_memory_bytes", 0) / max(calibration.get("gpu_memory_bytes", 1), 1) * 100
        w(f"  Batch size:           {calibration['batch_size']}")
        w(f"  GPU:                  {calibration.get('gpu_name', '?')}")
        w(f"  GPU utilization:      {gpu_util:.1f}%")
        w(f"  Avg time/matrix:      {calibration.get('avg_seconds_per_matrix', '?')}s")
        w(f"  Timing samples:       {calibration.get('timing_samples', '?')}")
        sr = calibration.get("slurm_resources", {})
        if sr:
            w("  Estimated step durations:")
            for step, res in sorted(sr.items()):
                w(f"    Step {step}: time={res.get('time','?')}  mem={res.get('mem','?')}")
    else:
        w("  Calibration not performed or file missing.")
    w("")

    # --- Section 3: Step Completion ---
    w("=== Section 3: Pipeline Step Completion ===")
    w(f"  [1] Training weights:       {'OK' if step_a_ok else 'MISSING'}  (epoch_{epochs}.pth)")

    if integrity_report:
        steps = integrity_report.get("steps", {})

        def count_status(entries):
            ok = sum(1 for e in entries if e["status"] == "OK")
            return f"{ok}/{len(entries)} OK" + (f"  (MISSING: {', '.join(str(i) for i,e in enumerate(entries) if e['status']!='OK')})" if ok < len(entries) else "")

        w(f"  [2a] Matrix zips:           {count_status(steps.get('matrices_zips', []))}")
        adv = steps.get("adversarial_examples", [])
        adv_ok = sum(1 for e in adv if e["status"] == "OK")
        w(f"  [2b] Adversarial examples:  {adv_ok}/{len(adv)} OK")
        if adv_ok < len(adv):
            for e in adv:
                if e["status"] != "OK":
                    w(f"        MISSING: {os.path.basename(os.path.dirname(e['path']))}")
        w(f"  [3] Adv matrix zips:        {count_status(steps.get('adv_matrices_zips', []))}")

        summary = integrity_report.get("summary", {})
        w(f"\n  Summary: {summary.get('ok',0)} OK, {summary.get('missing',0)} MISSING, {summary.get('corrupt',0)} CORRUPT")
    else:
        w("  (experiment directory not found)")
    w("")

    # --- Section 4: Error Analysis ---
    w("=== Section 4: Error Analysis ===")
    _steps_for_errors = overall_errors_steps if overall_errors_steps is not None else []
    if overall_errors and not overall_errors.get("error_scan_failed"):
        # Use structured data from overall_errors.json
        err_steps = [s for s in _steps_for_errors if s.get("error_detected")]
        if err_steps:
            for s in err_steps:
                chunk_str = f" chunk {s['chunk']}" if s.get("chunk") is not None else ""
                cat_str = s.get("error_category", "unknown")
                w(f"  Job {s['job_id']} ({s['step_label']}{chunk_str}) -- "
                  f"{s['slurm_state']} -- Type: {s.get('error_type', '?')} ({cat_str})")
                tb = s.get("traceback")
                tail = s.get("err_tail", "")
                display = tb or tail
                if display and display.strip():
                    for line in display.strip().split("\n")[-15:]:
                        w(f"    | {line.rstrip()}")
                w("")
        else:
            w("  No errors detected.")
    elif errors:
        # Fall back to live .err scanning
        w("  (from live log scanning — overall_errors.json not available)")
        for err in errors:
            w(f"  Job {err['job_id']} ({err['step']}) -- {err['state']} -- Category: {err['category']}")
            if err["last_lines"].strip():
                for line in err["last_lines"].strip().split("\n")[-15:]:
                    w(f"    | {line.rstrip()}")
            w("")
    else:
        w("  No errors detected.")
    w("")

    # --- Section 5: GPU Monitoring ---
    w("=== Section 5: GPU Monitoring Summary ===")
    if gpu_data:
        w(f"  {'Step':<18} {'Peak Util%':<12} {'Avg Util%':<12} {'Peak Mem (MiB)':<16} {'Samples'}")
        w(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*16} {'-'*8}")
        for label in sorted(gpu_data.keys()):
            d = gpu_data[label]
            w(f"  {label:<18} {d['peak_util']:<12.1f} {d['avg_util']:<12.1f} {d['peak_mem_mib']:<16.0f} {d['samples']}")
    else:
        w("  No GPU monitoring data found.")
    w("")

    # --- Section 6: Recovery ---
    w("=== Section 6: Recovery Recommendations ===")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            w(f"  {i}. {rec}")
        w(f"\n  Suggested: bash run_experiment.sh --skip-audit {experiment}")
    else:
        w("  All steps completed successfully. No recovery needed.")
    w("")
    w("=" * 70)

    # Print to stdout
    report_text = "\n".join(lines)
    print(report_text)

    # Save to file
    os.makedirs("reports", exist_ok=True)
    report_file = os.path.join("reports", f"{experiment}_report_{timestamp}.txt")
    with open(report_file, "w") as f:
        f.write(report_text + "\n")
    print(f"\nReport saved to: {report_file}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    experiment = args.experiment
    total_chunks = args.total_chunks
    test_mode = args.test

    slurm_out_dir = "slurm_out_test" if test_mode else "slurm_out"
    slurm_err_dir = "slurm_err_test" if test_mode else "slurm_err"

    # Validate experiment
    sys.path.insert(0, ".")
    from constants.constants import DEFAULT_EXPERIMENTS
    if experiment not in DEFAULT_EXPERIMENTS:
        print(f"ERROR: '{experiment}' not in DEFAULT_EXPERIMENTS")
        print(f"Available: {', '.join(sorted(DEFAULT_EXPERIMENTS.keys()))}")
        sys.exit(1)

    # Load overall_errors.json if available
    overall_errors, overall_errors_steps = load_overall_errors(experiment)

    # Gather data
    jobs = discover_jobs(experiment, slurm_out_dir)
    job_ids = list({j["job_id"] for j in jobs})
    sacct_data = query_sacct(set(job_ids))
    calibration = read_calibration(experiment)
    step_a_ok, integrity_report, epochs = check_step_completion(experiment, total_chunks)
    errors = analyze_errors(jobs, slurm_out_dir, slurm_err_dir, sacct_data)
    gpu_data = parse_gpu_logs(experiment, total_chunks)
    recommendations = generate_recommendations(step_a_ok, integrity_report)

    write_report(
        experiment=experiment, test_mode=test_mode,
        total_chunks=total_chunks, jobs=jobs, sacct_data=sacct_data,
        calibration=calibration, step_a_ok=step_a_ok,
        integrity_report=integrity_report, epochs=epochs,
        errors=errors, gpu_data=gpu_data,
        recommendations=recommendations,
        overall_errors=overall_errors,
        overall_errors_steps=overall_errors_steps,
    )


if __name__ == "__main__":
    main()
