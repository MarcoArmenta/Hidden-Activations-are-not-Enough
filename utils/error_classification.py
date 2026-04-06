"""
Shared error classification logic for pipeline error scanning and reporting.

Used by both collect_errors.py (Slurm error scan) and pipeline_report.py.
"""

import re


# ── Step metadata ─────────────────────────────────────────────────────────

STEP_ORDER = [
    "CALIB", "PREAUDIT", "A", "B", "C", "D", "E", "G", "F",
    "AUDIT", "DISPATCH", "ERRSCAN",
]

STEP_LABELS = {
    "CALIB": "Calibration",
    "PREAUDIT": "Pre-Audit",
    "A": "Training",
    "B": "Matrices",
    "C": "Adv Examples",
    "D": "Adv Matrices",
    "E": "Rep. Comparison",
    "G": "Theorem 4.5",
    "F": "LaTeX Tables",
    "AUDIT": "Final Audit",
    "DISPATCH": "Dispatcher",
    "ERRSCAN": "Error Scan",
}

# Labels with step ID prefix (used by pipeline_report.py)
STEP_LABELS_PREFIXED = {
    k: (f"{k} ({v})" if len(k) == 1 else v)
    for k, v in STEP_LABELS.items()
}


# ── Error patterns ────────────────────────────────────────────────────────

ERROR_PATTERNS = [
    ("cuda_oom",      re.compile(r"CUDA out of memory", re.I)),
    ("oom",           re.compile(r"out of memory|oom-kill|Killed\s+by\s+signal\s+9|oom_kill|cannot allocate memory", re.I)),
    ("timeout",       re.compile(r"DUE TO TIME LIMIT|CANCELLED.*TIME", re.I)),
    ("cuda_error",    re.compile(r"CUDA error|NCCL", re.I)),
    ("network_error", re.compile(r"network.unreachable|ConnectionError|urllib.*Error", re.I)),
    ("missing_file",  re.compile(r"FileNotFoundError|No such file|not found", re.I)),
    ("module_error",  re.compile(r"ModuleNotFoundError|ImportError", re.I)),
    ("zip_error",     re.compile(r"Zip.*failed|BadZipFile|Zip verification", re.I)),
    ("assertion",     re.compile(r"AssertionError", re.I)),
    ("permission",    re.compile(r"PermissionError|Permission denied", re.I)),
]

# Map fine-grained error types to 4 high-level categories
ERROR_CATEGORY_MAP = {
    # slurm: resource limit issues managed by the scheduler
    "oom":           "slurm",
    "timeout":       "slurm",
    # code: bugs or logic errors in the Python pipeline
    "code":          "code",
    "assertion":     "code",
    # data: missing or corrupt files / datasets
    "missing_file":  "data",
    "zip_error":     "data",
    # environment: CUDA, network, modules, permissions
    "cuda_oom":      "environment",
    "cuda_error":    "environment",
    "network_error": "environment",
    "module_error":  "environment",
    "permission":    "environment",
    # unknown
    "unknown":       "unknown",
}

ERROR_CATEGORIES = ["slurm", "code", "data", "environment", "unknown"]

# Map fine-grained types to the 4-value enforced enum
NORMALIZED_ERROR_TYPE = {
    "oom": "OOM",
    "cuda_oom": "CUDA_OOM",
    "timeout": "TIMEOUT",
    "code": "RUNTIME",
    "assertion": "RUNTIME",
    "missing_file": "RUNTIME",
    "zip_error": "RUNTIME",
    "cuda_error": "RUNTIME",
    "network_error": "RUNTIME",
    "module_error": "RUNTIME",
    "permission": "RUNTIME",
    "unknown": "UNKNOWN",
}


def normalize_error_type(fine_type):
    """Map fine-grained error type to enforced enum: OOM|CUDA_OOM|TIMEOUT|RUNTIME|UNKNOWN."""
    if fine_type is None:
        return "UNKNOWN"
    return NORMALIZED_ERROR_TYPE.get(fine_type, "UNKNOWN")


# ── Log filename pattern ──────────────────────────────────────────────────

# Matches both regular and array job log filenames.
# Regular:   PIPE_E_alexnet_cifar10_12345.out
# Chunked:   PIPE_B_alexnet_cifar10_c3_12345.out
# Array job:  PIPE_D_alexnet_cifar10_12345_3.out
# Groups: (1)prefix (2)step (3)experiment (4)chunk_or_None (5)job_id (6)array_task_or_None (7)ext
LOG_PATTERN = re.compile(
    r'^(PIPE|REC)_([A-Za-z0-9]+)_(.+?)(?:_c(\d+))?_(\d+)(?:_(\d+))?\.(out|err)$'
)


def parse_log_filename(fname):
    """Parse a log filename into a dict with prefix, step, exp, chunk, job_id, ext.

    Handles both regular and array job log filenames. For array jobs,
    the array task ID is stored in the chunk field.
    """
    m = LOG_PATTERN.match(fname)
    if not m:
        return None
    prefix, step, exp, chunk, job_id, array_task, ext = m.groups()
    # For array jobs, use the array task ID as the chunk identifier
    if array_task is not None:
        chunk = array_task
    return {
        "prefix": prefix, "step": step, "exp": exp,
        "chunk": chunk, "job_id": job_id, "ext": ext,
    }


# ── Classification functions ──────────────────────────────────────────────

def classify_error(text):
    """Classify error text into a fine-grained error type.

    Returns the first matching error type from ERROR_PATTERNS,
    or None if no pattern matches.
    """
    for error_type, pattern in ERROR_PATTERNS:
        if pattern.search(text):
            return error_type
    return None


def get_error_category(error_type):
    """Map a fine-grained error type to a high-level category.

    Returns one of: 'slurm', 'code', 'data', 'environment', 'unknown'.
    """
    if error_type is None:
        return "unknown"
    return ERROR_CATEGORY_MAP.get(error_type, "unknown")


def extract_traceback(text):
    """Extract the last Python traceback from text.

    Returns the traceback string, or None if not found.
    """
    lines = text.split("\n")
    tb_start = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Traceback (most recent call last):"):
            tb_start = i
            break
    if tb_start is not None:
        tb_end = len(lines)
        for i in range(tb_start + 1, len(lines)):
            line = lines[i]
            if line and not line.startswith(" ") and not line.startswith("Traceback"):
                tb_end = i + 1  # include the error line
                break
        return "\n".join(lines[tb_start:tb_end]).strip()
    return None


def read_tail(filepath, n=80):
    """Read the last n lines of a file."""
    try:
        with open(filepath) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:
        return ""
