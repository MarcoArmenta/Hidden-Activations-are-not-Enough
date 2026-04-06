"""
Data integrity verification and zip-with-verification utilities.

Usage as CLI:
    python -m utils.data_integrity --experiment_dir experiments/alexnet_cifar10 \
        --experiment_name alexnet_cifar10 --total_chunks 4

Usage as library:
    from utils.data_integrity import verify_zip, verify_pth_in_zip, zip_and_verify
"""

import os
import sys
import json
import zipfile
import random
import shutil
import argparse
from io import BytesIO
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# 1. verify_zip
# ---------------------------------------------------------------------------

def verify_zip(zip_path: str) -> dict:
    """Check that a zip file is not corrupt.

    Returns:
        {valid: bool, error: str|None, file_count: int, file_list: [...]}
    """
    zip_path = str(zip_path)
    result = {"valid": False, "error": None, "file_count": 0, "file_list": []}

    if not os.path.exists(zip_path):
        result["error"] = f"File does not exist: {zip_path}"
        return result

    if not zipfile.is_zipfile(zip_path):
        result["error"] = f"Not a valid zip file: {zip_path}"
        return result

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                result["error"] = f"Corrupt entry in zip: {bad}"
                return result
            result["file_list"] = zf.namelist()
            result["file_count"] = len(result["file_list"])
    except zipfile.BadZipFile as exc:
        result["error"] = f"BadZipFile: {exc}"
        return result
    except Exception as exc:
        result["error"] = f"Unexpected error: {exc}"
        return result

    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# 2. verify_pth_in_zip
# ---------------------------------------------------------------------------

def verify_pth_in_zip(zip_path: str, sample_ratio: float = 0.1) -> dict:
    """Open a zip, sample .pth/.pt files, and try to torch.load each one.

    Returns:
        {valid: bool, total_pth: int, sampled: int, corrupt: [...], errors: [...]}
    """
    zip_path = str(zip_path)
    result = {"valid": False, "total_pth": 0, "sampled": 0, "corrupt": [], "errors": []}

    if not os.path.exists(zip_path):
        result["errors"].append(f"File does not exist: {zip_path}")
        return result

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            pth_files = [n for n in zf.namelist() if n.endswith((".pth", ".pt"))]
            result["total_pth"] = len(pth_files)

            if not pth_files:
                result["valid"] = True
                return result

            sample_size = max(1, int(len(pth_files) * sample_ratio))
            sampled = random.sample(pth_files, min(sample_size, len(pth_files)))
            result["sampled"] = len(sampled)

            for name in sampled:
                try:
                    data = zf.read(name)
                    torch.load(BytesIO(data), map_location="cpu", weights_only=False)
                except Exception as exc:
                    result["corrupt"].append(name)
                    result["errors"].append(f"{name}: {exc}")
    except Exception as exc:
        result["errors"].append(f"Failed to open zip: {exc}")
        return result

    result["valid"] = len(result["corrupt"]) == 0
    return result


# ---------------------------------------------------------------------------
# 3. verify_experiment
# ---------------------------------------------------------------------------

def verify_experiment(
    experiment_dir: str,
    experiment_name: str,
    num_classes: int,
    num_samples_per_class: int,
    total_chunks: int,
    num_samples_rejection_level: int,
    attacks_list: list,
    sample_ratio: float = 0.1,
) -> dict:
    """Audit an experiment directory for expected artifacts.

    Checks:
        - matrices_task_{0..total_chunks-1}.zip
        - adv_matrices_task_{0..total_chunks-1}.zip
        - rejection_levels/matrices_task_{0..total_chunks-1}.zip
        - weights/
        - adversarial_examples/{attack}/adversarial_examples.pth
        - matrices/matrix_statistics.json
        - grid_search/
        - rejection_levels/reject_at_*.json
    """
    experiment_dir = str(experiment_dir)
    report = {
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "steps": {},
        "summary": {"total_checks": 0, "ok": 0, "missing": 0, "corrupt": 0},
    }

    def _check_zip(zip_path, step_name):
        entry = {"path": zip_path, "status": "OK", "details": {}}
        report["summary"]["total_checks"] += 1

        if not os.path.exists(zip_path):
            entry["status"] = "MISSING"
            report["summary"]["missing"] += 1
            return entry

        zv = verify_zip(zip_path)
        entry["details"]["zip_check"] = zv
        if not zv["valid"]:
            entry["status"] = "CORRUPT"
            report["summary"]["corrupt"] += 1
            return entry

        pv = verify_pth_in_zip(zip_path, sample_ratio=sample_ratio)
        entry["details"]["pth_check"] = pv
        if not pv["valid"]:
            entry["status"] = "CORRUPT"
            report["summary"]["corrupt"] += 1
            return entry

        report["summary"]["ok"] += 1
        return entry

    def _check_file(file_path):
        report["summary"]["total_checks"] += 1
        if os.path.exists(file_path):
            report["summary"]["ok"] += 1
            return {"path": file_path, "status": "OK"}
        else:
            report["summary"]["missing"] += 1
            return {"path": file_path, "status": "MISSING"}

    def _check_dir(dir_path):
        report["summary"]["total_checks"] += 1
        if os.path.isdir(dir_path):
            report["summary"]["ok"] += 1
            return {"path": dir_path, "status": "OK"}
        else:
            report["summary"]["missing"] += 1
            return {"path": dir_path, "status": "MISSING"}

    # --- matrices zips ---
    matrices_zips = []
    for i in range(total_chunks):
        zp = os.path.join(experiment_dir, f"matrices_task_{i}.zip")
        matrices_zips.append(_check_zip(zp, f"matrices_task_{i}"))
    report["steps"]["matrices_zips"] = matrices_zips

    # --- adv_matrices zips ---
    adv_matrices_zips = []
    for i in range(total_chunks):
        zp = os.path.join(experiment_dir, f"adv_matrices_task_{i}.zip")
        adv_matrices_zips.append(_check_zip(zp, f"adv_matrices_task_{i}"))
    report["steps"]["adv_matrices_zips"] = adv_matrices_zips

    # --- rejection_levels zips ---
    rej_zips = []
    for i in range(total_chunks):
        zp = os.path.join(experiment_dir, "rejection_levels", f"matrices_task_{i}.zip")
        rej_zips.append(_check_zip(zp, f"rejection_levels/matrices_task_{i}"))
    report["steps"]["rejection_level_zips"] = rej_zips

    # --- weights ---
    report["steps"]["weights"] = _check_dir(os.path.join(experiment_dir, "weights"))

    # --- adversarial examples ---
    adv_examples = []
    for attack in attacks_list:
        fp = os.path.join(experiment_dir, "adversarial_examples", attack, "adversarial_examples.pth")
        adv_examples.append(_check_file(fp))
    report["steps"]["adversarial_examples"] = adv_examples

    # --- matrix_statistics.json ---
    report["steps"]["matrix_statistics"] = _check_file(
        os.path.join(experiment_dir, "matrices", "matrix_statistics.json")
    )

    # --- grid_search ---
    report["steps"]["grid_search"] = _check_dir(os.path.join(experiment_dir, "grid_search"))

    # --- rejection level json files ---
    rej_jsons = []
    rej_dir = os.path.join(experiment_dir, "rejection_levels")
    if os.path.isdir(rej_dir):
        for f in sorted(os.listdir(rej_dir)):
            if f.startswith("reject_at_") and f.endswith(".json"):
                rej_jsons.append({"path": os.path.join(rej_dir, f), "status": "OK"})
    if rej_jsons:
        report["steps"]["rejection_level_jsons"] = rej_jsons
    else:
        report["summary"]["total_checks"] += 1
        report["summary"]["missing"] += 1
        report["steps"]["rejection_level_jsons"] = [
            {"path": os.path.join(rej_dir, "reject_at_*.json"), "status": "MISSING"}
        ]

    return report


# ---------------------------------------------------------------------------
# 4. zip_and_verify
# ---------------------------------------------------------------------------

def zip_and_verify(
    src_dir: str,
    zip_path: str,
    cleanup: bool = False,
    sample_ratio: float = 0.1,
) -> dict:
    """Zip *src_dir* into *zip_path*, then verify the resulting archive.

    Only deletes *src_dir* if verification passes **and** ``cleanup=True``.

    Returns:
        {success: bool, zip_path: str, file_count: int, errors: [...]}
    """
    src_dir = str(src_dir)
    zip_path = str(zip_path)
    # Ensure zip_path ends with .zip for shutil
    if zip_path.endswith(".zip"):
        archive_base = zip_path[:-4]
    else:
        archive_base = zip_path
        zip_path = zip_path + ".zip"

    result = {"success": False, "zip_path": zip_path, "file_count": 0, "errors": []}

    if not os.path.isdir(src_dir):
        result["errors"].append(f"Source directory does not exist: {src_dir}")
        return result

    try:
        shutil.make_archive(archive_base, "zip", src_dir)
    except Exception as exc:
        result["errors"].append(f"Failed to create archive: {exc}")
        return result

    zv = verify_zip(zip_path)
    if not zv["valid"]:
        result["errors"].append(f"Zip verification failed: {zv['error']}")
        return result
    result["file_count"] = zv["file_count"]

    pv = verify_pth_in_zip(zip_path, sample_ratio=sample_ratio)
    if not pv["valid"]:
        result["errors"].extend(pv["errors"])
        return result

    result["success"] = True

    if cleanup:
        shutil.rmtree(src_dir, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# 5. CLI helpers
# ---------------------------------------------------------------------------

_COLOR_RESET = "\033[0m"
_COLOR_GREEN = "\033[92m"
_COLOR_RED = "\033[91m"
_COLOR_YELLOW = "\033[93m"
_COLOR_BOLD = "\033[1m"


def _status_color(status: str) -> str:
    if status == "OK":
        return f"{_COLOR_GREEN}{status}{_COLOR_RESET}"
    elif status == "MISSING":
        return f"{_COLOR_YELLOW}{status}{_COLOR_RESET}"
    else:
        return f"{_COLOR_RED}{status}{_COLOR_RESET}"


def _print_report(report: dict) -> None:
    """Pretty-print a verification report to stdout."""
    print(f"\n{_COLOR_BOLD}=== Data Integrity Report ==={_COLOR_RESET}")
    print(f"Experiment: {report['experiment_name']}")
    print(f"Directory:  {report['experiment_dir']}\n")

    for step_name, step_data in report["steps"].items():
        print(f"{_COLOR_BOLD}  [{step_name}]{_COLOR_RESET}")
        if isinstance(step_data, list):
            for item in step_data:
                status = _status_color(item["status"])
                path = os.path.basename(item["path"])
                print(f"    {status}  {path}")
                if "details" in item:
                    details = item["details"]
                    if "zip_check" in details and not details["zip_check"]["valid"]:
                        print(f"         zip error: {details['zip_check']['error']}")
                    if "pth_check" in details and not details["pth_check"]["valid"]:
                        for err in details["pth_check"]["errors"]:
                            print(f"         pth error: {err}")
        elif isinstance(step_data, dict):
            status = _status_color(step_data["status"])
            path = os.path.basename(step_data["path"])
            print(f"    {status}  {path}")
        print()

    s = report["summary"]
    print(f"{_COLOR_BOLD}--- Summary ---{_COLOR_RESET}")
    print(f"  Total checks: {s['total_checks']}")
    print(f"  {_COLOR_GREEN}OK{_COLOR_RESET}:      {s['ok']}")
    print(f"  {_COLOR_YELLOW}MISSING{_COLOR_RESET}: {s['missing']}")
    print(f"  {_COLOR_RED}CORRUPT{_COLOR_RESET}: {s['corrupt']}")
    print()


def _cli_verify_single_zip(args):
    """Handler for --verify-zip mode."""
    zv = verify_zip(args.verify_zip)
    if zv["valid"]:
        print(f"{_COLOR_GREEN}OK{_COLOR_RESET}  {args.verify_zip}  ({zv['file_count']} files)")
    else:
        print(f"{_COLOR_RED}FAIL{_COLOR_RESET}  {args.verify_zip}  error: {zv['error']}")
        sys.exit(1)

    if args.full_check:
        pv = verify_pth_in_zip(args.verify_zip, sample_ratio=1.0)
    else:
        pv = verify_pth_in_zip(args.verify_zip)

    if pv["total_pth"] > 0:
        if pv["valid"]:
            print(f"  pth check: {_COLOR_GREEN}OK{_COLOR_RESET}  (sampled {pv['sampled']}/{pv['total_pth']})")
        else:
            print(f"  pth check: {_COLOR_RED}FAIL{_COLOR_RESET}  corrupt: {pv['corrupt']}")
            sys.exit(1)


def _cli_verify_experiment(args):
    """Handler for full experiment verification."""
    # Try to pull experiment metadata from constants
    try:
        from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS
    except ImportError:
        print("Warning: Could not import constants. Using provided CLI args only.", file=sys.stderr)
        ATTACKS = []
        DEFAULT_EXPERIMENTS = {}

    attacks_list = ATTACKS
    num_classes = 10
    num_samples_per_class = 1000
    num_samples_rejection_level = 10000

    if args.experiment_name in DEFAULT_EXPERIMENTS:
        exp = DEFAULT_EXPERIMENTS[args.experiment_name]
        dataset = exp.get("dataset", "cifar10")
        if dataset == "cifar100":
            num_classes = 100
        elif dataset == "imagenet":
            num_classes = 1000

    report = verify_experiment(
        experiment_dir=args.experiment_dir,
        experiment_name=args.experiment_name,
        num_classes=num_classes,
        num_samples_per_class=num_samples_per_class,
        total_chunks=args.total_chunks,
        num_samples_rejection_level=num_samples_rejection_level,
        attacks_list=attacks_list,
        sample_ratio=1.0 if args.full_check else 0.1,
    )

    _print_report(report)

    if report["summary"]["corrupt"] > 0:
        sys.exit(2)
    elif report["summary"]["missing"] > 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Data integrity verification for experiment artifacts.",
    )

    parser.add_argument(
        "--verify-zip",
        type=str,
        default=None,
        help="Verify a single zip file and exit.",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Path to the experiment directory (e.g., experiments/alexnet_cifar10).",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name (key in DEFAULT_EXPERIMENTS).",
    )
    parser.add_argument(
        "--total_chunks",
        type=int,
        default=4,
        help="Number of chunks / array tasks (default: 4).",
    )
    parser.add_argument(
        "--full-check",
        action="store_true",
        default=False,
        help="Check 100%% of .pth files instead of sampling 10%%.",
    )

    args = parser.parse_args()

    if args.verify_zip:
        _cli_verify_single_zip(args)
    elif args.experiment_dir and args.experiment_name:
        _cli_verify_experiment(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
