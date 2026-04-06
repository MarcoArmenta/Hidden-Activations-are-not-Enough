"""Tests for utils.error_classification module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.error_classification import (
    classify_error,
    get_error_category,
    extract_traceback,
    read_tail,
    ERROR_CATEGORIES,
    ERROR_CATEGORY_MAP,
    LOG_PATTERN,
    STEP_LABELS,
    STEP_ORDER,
    parse_log_filename,
)


# ── classify_error ────────────────────────────────────────────────────────

class TestClassifyError:
    def test_oom(self):
        assert classify_error("slurmstepd: error: Detected 1 oom-kill event") == "oom"

    def test_oom_killed(self):
        assert classify_error("some process Killed by signal 9") == "oom"

    def test_oom_cannot_allocate(self):
        assert classify_error("cannot allocate memory for buffer") == "oom"

    def test_cuda_out_of_memory(self):
        # HI-13: "CUDA out of memory" now classified as "cuda_oom" (environment)
        # rather than "oom" (slurm), because it requires code-level batch_size
        # fixes, not more Slurm memory.
        assert classify_error("CUDA out of memory. Tried to allocate 2.00 GiB") == "cuda_oom"

    def test_cuda_error(self):
        assert classify_error("RuntimeError: CUDA error: device-side assert triggered") == "cuda_error"

    def test_timeout(self):
        assert classify_error("JOB 12345 CANCELLED AT 2024-01-01 DUE TO TIME LIMIT") == "timeout"

    def test_timeout_cancelled(self):
        assert classify_error("CANCELLED BY TIME LIMIT") == "timeout"

    def test_network_error(self):
        assert classify_error("urllib.error.URLError: <urlopen error network unreachable>") == "network_error"

    def test_connection_error(self):
        assert classify_error("ConnectionError: connection refused") == "network_error"

    def test_missing_file(self):
        assert classify_error("FileNotFoundError: [Errno 2] No such file") == "missing_file"

    def test_not_found(self):
        assert classify_error("Error: weights file not found at path/to/file") == "missing_file"

    def test_module_error(self):
        assert classify_error("ModuleNotFoundError: No module named 'torchattacks'") == "module_error"

    def test_import_error(self):
        assert classify_error("ImportError: cannot import name 'foo' from 'bar'") == "module_error"

    def test_zip_error(self):
        assert classify_error("BadZipFile: File is not a zip file") == "zip_error"

    def test_zip_verification_failed(self):
        assert classify_error("Zip verification failed for matrices_task_0.zip") == "zip_error"

    def test_assertion_error(self):
        assert classify_error("AssertionError: expected 10 but got 5") == "assertion"

    def test_permission_error(self):
        assert classify_error("PermissionError: [Errno 13] Permission denied") == "permission"

    def test_no_match(self):
        assert classify_error("Everything is fine, no errors here") is None

    def test_empty_string(self):
        assert classify_error("") is None

    def test_priority_oom_over_missing(self):
        # OOM comes first in the list, so it should win
        text = "out of memory while loading file not found"
        assert classify_error(text) == "oom"


# ── get_error_category ────────────────────────────────────────────────────

class TestGetErrorCategory:
    def test_slurm_oom(self):
        assert get_error_category("oom") == "slurm"

    def test_slurm_timeout(self):
        assert get_error_category("timeout") == "slurm"

    def test_code_error(self):
        assert get_error_category("code") == "code"

    def test_code_assertion(self):
        assert get_error_category("assertion") == "code"

    def test_data_missing_file(self):
        assert get_error_category("missing_file") == "data"

    def test_data_zip(self):
        assert get_error_category("zip_error") == "data"

    def test_env_cuda(self):
        assert get_error_category("cuda_error") == "environment"

    def test_env_network(self):
        assert get_error_category("network_error") == "environment"

    def test_env_module(self):
        assert get_error_category("module_error") == "environment"

    def test_env_permission(self):
        assert get_error_category("permission") == "environment"

    def test_unknown_type(self):
        assert get_error_category("unknown") == "unknown"

    def test_none_type(self):
        assert get_error_category(None) == "unknown"

    def test_unrecognized_type(self):
        assert get_error_category("totally_new_error") == "unknown"

    def test_all_mapped_types_have_valid_categories(self):
        for error_type, category in ERROR_CATEGORY_MAP.items():
            assert category in ERROR_CATEGORIES, \
                f"{error_type} maps to '{category}' which is not in ERROR_CATEGORIES"


# ── extract_traceback ─────────────────────────────────────────────────────

class TestExtractTraceback:
    def test_simple_traceback(self):
        text = """Some output
Traceback (most recent call last):
  File "foo.py", line 42, in main
    do_something()
  File "bar.py", line 10, in do_something
    raise ValueError("bad")
ValueError: bad
more output"""
        tb = extract_traceback(text)
        assert tb is not None
        assert "Traceback (most recent call last):" in tb
        assert "ValueError: bad" in tb

    def test_no_traceback(self):
        text = "Everything is fine\nNo errors here\nAll good"
        assert extract_traceback(text) is None

    def test_empty_string(self):
        assert extract_traceback("") is None

    def test_multiple_tracebacks_returns_last(self):
        text = """Traceback (most recent call last):
  File "a.py", line 1
FirstError: first

Some middle text

Traceback (most recent call last):
  File "b.py", line 2
SecondError: second"""
        tb = extract_traceback(text)
        assert "SecondError: second" in tb
        assert "FirstError" not in tb

    def test_traceback_at_end_of_text(self):
        text = """output
Traceback (most recent call last):
  File "foo.py", line 1, in <module>
RuntimeError: CUDA error"""
        tb = extract_traceback(text)
        assert tb is not None
        assert "RuntimeError: CUDA error" in tb


# ── read_tail ─────────────────────────────────────────────────────────────

class TestReadTail:
    def test_nonexistent_file(self):
        assert read_tail("/nonexistent/path/file.txt") == ""

    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        result = read_tail(str(f), n=3)
        assert "line3" in result
        assert "line4" in result
        assert "line5" in result

    def test_fewer_lines_than_n(self, tmp_path):
        f = tmp_path / "short.txt"
        f.write_text("only one line\n")
        result = read_tail(str(f), n=100)
        assert "only one line" in result


# ── LOG_PATTERN ───────────────────────────────────────────────────────────

class TestLogPattern:
    def test_pipe_with_chunk(self):
        p = parse_log_filename("PIPE_2a_alexnet_cifar10_c3_12345678.err")
        assert p is not None
        assert p["prefix"] == "PIPE"
        assert p["step"] == "2a"
        assert p["exp"] == "alexnet_cifar10"
        assert p["chunk"] == "3"
        assert p["job_id"] == "12345678"
        assert p["ext"] == "err"

    def test_rec_no_chunk(self):
        p = parse_log_filename("REC_1_resnet_cifar100_99999999.out")
        assert p is not None
        assert p["prefix"] == "REC"
        assert p["step"] == "1"
        assert p["exp"] == "resnet_cifar100"
        assert p["chunk"] is None
        assert p["job_id"] == "99999999"

    def test_errscan(self):
        p = parse_log_filename("PIPE_ERRSCAN_alexnet_cifar10_11111111.out")
        assert p is not None
        assert p["step"] == "ERRSCAN"

    def test_no_match(self):
        assert parse_log_filename("random_file.txt") is None

    def test_array_job(self):
        p = parse_log_filename("PIPE_3_alexnet_cifar10_12345_3.out")
        assert p is not None
        assert p["prefix"] == "PIPE"
        assert p["step"] == "3"
        assert p["exp"] == "alexnet_cifar10"
        assert p["chunk"] == "3"  # array task ID becomes chunk
        assert p["job_id"] == "12345"
        assert p["ext"] == "out"


# ── Metadata consistency ─────────────────────────────────────────────────

class TestMetadata:
    def test_all_step_order_has_labels(self):
        for step in STEP_ORDER:
            assert step in STEP_LABELS, f"Step '{step}' in STEP_ORDER but not in STEP_LABELS"
