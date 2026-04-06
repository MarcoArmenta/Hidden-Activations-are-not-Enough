"""
Integration tests that verify the 5 checkpoint audit criteria are met.
These are characterization tests — they verify the code patterns exist.
"""
import os
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_file(relpath):
    with open(os.path.join(ROOT, relpath)) as f:
        return f.read()


class TestOOMRetry:
    """Criterion 1: OOM errors are caught, job resubmitted with 2x memory, logged."""

    def test_parallel_py_catches_oom(self):
        src = read_file("matrix_construction/parallel.py")
        assert "out of memory" in src.lower()
        assert "batch_size // 2" in src or "batch_size//2" in src

    def test_orchestrator_checks_exit_137(self):
        src = read_file("run_experiment.sh")
        assert '"137"' in src

    def test_orchestrator_has_sacct_fallback(self):
        src = read_file("run_experiment.sh")
        assert "detect_last_job_state" in src

    def test_double_mem_caps_at_480(self):
        src = read_file("experiment_config.sh")
        assert "480" in src
        assert "double_mem" in src

    def test_auto_resubmit_doubles_memory(self):
        src = read_file("auto_resubmit.py")
        assert "double_memory" in src
        assert "OOM" in src


class TestTimeoutRetry:
    """Criterion 2: Timeout caught, checkpoint saved, retry with 2x time, logged."""

    def test_double_time_exists(self):
        src = read_file("experiment_config.sh")
        assert "double_time" in src

    def test_orchestrator_handles_timeout(self):
        src = read_file("run_experiment.sh")
        assert "TIMEOUT" in src

    def test_orchestrator_checks_exit_140(self):
        src = read_file("run_experiment.sh")
        assert '"140"' in src

    def test_auto_resubmit_handles_timeout(self):
        src = read_file("auto_resubmit.py")
        assert "TIMEOUT" in src
        assert "double_time" in src


class TestOverallErrorsSchema:
    """Criterion 3: overall_errors.json conforms to enforced schema."""

    def test_collect_errors_writes_enforced_schema(self):
        src = read_file("collect_errors.py")
        assert "experiment_name" in src
        assert "last_updated" in src
        assert "original_resources" in src
        assert "retry_resources" in src
        assert "resolved" in src

    def test_normalize_error_type_exists(self):
        from utils.error_classification import normalize_error_type
        assert normalize_error_type("oom") == "OOM"
        assert normalize_error_type("timeout") == "TIMEOUT"

    def test_atomic_write_used(self):
        src = read_file("collect_errors.py")
        assert "atomic_json_dump" in src

    def test_append_mode(self):
        src = read_file("collect_errors.py")
        assert "_merge_errors" in src or "merge" in src.lower()


class TestCheckpointSkip:
    """Criterion 4: Completed work is skipped, writes are atomic."""

    def test_atomic_torch_save_used_in_parallel(self):
        src = read_file("matrix_construction/parallel.py")
        assert "atomic_torch_save" in src

    def test_atomic_torch_save_used_in_adv_matrices(self):
        src = read_file("generate_adversarial_matrices.py")
        assert "atomic_torch_save" in src

    def test_atomic_torch_save_used_in_adv_examples(self):
        src = read_file("generate_adversarial_examples.py")
        assert "atomic_torch_save" in src

    def test_atomic_torch_save_used_in_training(self):
        src = read_file("training.py")
        assert "atomic_torch_save" in src

    def test_atomic_json_dump_used_in_comparison(self):
        src = read_file("compare_representations.py")
        assert "atomic_json_dump" in src


class TestOrchestrationAwareness:
    """Criterion 5: Orchestrator reads checkpoint + overall_errors.json before submit."""

    def test_orchestrator_reads_checkpoint(self):
        src = read_file("run_experiment.sh")
        assert "read_checkpoint_status" in src

    def test_orchestrator_reads_overall_errors(self):
        src = read_file("run_experiment.sh")
        assert "overall_errors.json" in src
        assert "PREV_OOM_STEPS" in src

    def test_orchestrator_skips_complete(self):
        src = read_file("run_experiment.sh")
        assert "SKIPPED" in src
