import json
import os
import tempfile
import pytest

REQUIRED_TOP_LEVEL = {"experiment_name", "last_updated", "errors"}
REQUIRED_ERROR_FIELDS = {
    "job_id", "error_type", "phase", "grid_index", "timestamp",
    "original_resources", "retry_resources", "resolved", "message",
}
VALID_ERROR_TYPES = {"OOM", "CUDA_OOM", "TIMEOUT", "RUNTIME", "UNKNOWN"}


def validate_schema(data):
    """Validate overall_errors.json against enforced schema."""
    errors = []
    for field in REQUIRED_TOP_LEVEL:
        if field not in data:
            errors.append(f"Missing top-level field: {field}")
    if not isinstance(data.get("errors"), list):
        errors.append("'errors' must be a list")
        return errors
    for i, entry in enumerate(data["errors"]):
        for field in REQUIRED_ERROR_FIELDS:
            if field not in entry:
                errors.append(f"errors[{i}]: missing field '{field}'")
        if entry.get("error_type") not in VALID_ERROR_TYPES:
            errors.append(f"errors[{i}]: invalid error_type '{entry.get('error_type')}'")
        for res_key in ("original_resources", "retry_resources"):
            res = entry.get(res_key)
            if isinstance(res, dict):
                if "memory_gb" not in res or "time_hours" not in res:
                    errors.append(f"errors[{i}].{res_key}: missing memory_gb or time_hours")
    return errors


def test_schema_validates_correct_entry():
    data = {
        "experiment_name": "alexnet_cifar10",
        "last_updated": "2026-03-27T12:00:00",
        "errors": [{
            "job_id": "12345",
            "error_type": "OOM",
            "phase": "B",
            "grid_index": 3,
            "timestamp": "2026-03-27T12:00:00",
            "original_resources": {"memory_gb": 280.0, "time_hours": 0.33},
            "retry_resources": {"memory_gb": 480.0, "time_hours": 0.33},
            "resolved": False,
            "message": "CUDA out of memory",
        }],
    }
    assert validate_schema(data) == []


def test_schema_rejects_missing_fields():
    data = {"experiment_name": "test", "last_updated": "now", "errors": [{}]}
    errs = validate_schema(data)
    assert len(errs) > 0


def test_schema_rejects_invalid_error_type():
    data = {
        "experiment_name": "test",
        "last_updated": "now",
        "errors": [{
            "job_id": "1", "error_type": "oom", "phase": "A",
            "grid_index": None, "timestamp": "now",
            "original_resources": {"memory_gb": 15.0, "time_hours": 6.0},
            "retry_resources": {"memory_gb": None, "time_hours": None},
            "resolved": False, "message": "test",
        }],
    }
    errs = validate_schema(data)
    assert any("invalid error_type" in e for e in errs)


from utils.error_classification import normalize_error_type


def test_normalize_error_type_oom():
    assert normalize_error_type("oom") == "OOM"
    assert normalize_error_type("cuda_oom") == "CUDA_OOM"


def test_normalize_error_type_timeout():
    assert normalize_error_type("timeout") == "TIMEOUT"


def test_normalize_error_type_code():
    assert normalize_error_type("code") == "RUNTIME"
    assert normalize_error_type("assertion") == "RUNTIME"
    assert normalize_error_type("missing_file") == "RUNTIME"


def test_normalize_error_type_unknown():
    assert normalize_error_type("unknown") == "UNKNOWN"
    assert normalize_error_type(None) == "UNKNOWN"
    assert normalize_error_type("something_else") == "UNKNOWN"
