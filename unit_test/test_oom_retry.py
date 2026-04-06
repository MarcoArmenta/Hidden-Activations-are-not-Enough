"""Tests for collect_errors.py bug fixes and auto_resubmit.py."""

import os
import sys
import json
import pytest
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collect_errors import discover_jobs, detect_error
from auto_resubmit import (
    double_memory, double_time, parse_time_to_seconds, seconds_to_time,
    get_retry_set, step_to_script, load_errors,
    check_retry_count, DEPENDS_ON, TOPO_ORDER, RETRYABLE_ERROR_TYPES,
)


# -- Bug 1: Step 2b attack log discovery -----------------------------------

class TestStepCDiscovery:
    """discover_jobs() should find Step 2b attack logs where the filename
    includes the attack name: PIPE_2b_alexnet_cifar10_FGSM_12345.out
    """

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.out_dir = os.path.join(self.tmpdir, "slurm_out")
        self.err_dir = os.path.join(self.tmpdir, "slurm_err")
        os.makedirs(self.out_dir)
        os.makedirs(self.err_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def _touch(self, directory, filename):
        path = os.path.join(directory, filename)
        with open(path, "w") as f:
            f.write("")

    def test_step_c_attack_discovered(self):
        self._touch(self.out_dir, "PIPE_C_alexnet_cifar10_FGSM_12345.out")
        self._touch(self.err_dir, "PIPE_C_alexnet_cifar10_FGSM_12345.err")

        jobs = discover_jobs("alexnet_cifar10", self.out_dir, self.err_dir)
        assert len(jobs) == 1
        job = jobs[0]
        assert job["step"] == "C"
        assert job["chunk"] == "FGSM"
        assert job["job_id"] == "12345"

    def test_step_c_multiple_attacks(self):
        for attack in ["FGSM", "PGD", "CW"]:
            jid = str(10000 + hash(attack) % 1000)
            self._touch(self.out_dir, f"PIPE_C_alexnet_cifar10_{attack}_{jid}.out")
            self._touch(self.err_dir, f"PIPE_C_alexnet_cifar10_{attack}_{jid}.err")

        jobs = discover_jobs("alexnet_cifar10", self.out_dir, self.err_dir)
        assert len(jobs) == 3
        attacks = {j["chunk"] for j in jobs}
        assert attacks == {"FGSM", "PGD", "CW"}

    def test_step_c_does_not_match_wrong_experiment(self):
        # resnet_cifar10_FGSM should not match experiment=alexnet_cifar10
        self._touch(self.out_dir, "PIPE_C_resnet_cifar10_FGSM_12345.out")

        jobs = discover_jobs("alexnet_cifar10", self.out_dir, self.err_dir)
        assert len(jobs) == 0

    def test_normal_steps_still_work(self):
        self._touch(self.out_dir, "PIPE_A_alexnet_cifar10_11111.out")
        self._touch(self.out_dir, "PIPE_B_alexnet_cifar10_c3_22222.out")
        self._touch(self.err_dir, "PIPE_A_alexnet_cifar10_11111.err")
        self._touch(self.err_dir, "PIPE_B_alexnet_cifar10_c3_22222.err")

        jobs = discover_jobs("alexnet_cifar10", self.out_dir, self.err_dir)
        assert len(jobs) == 2
        steps = {j["step"] for j in jobs}
        assert steps == {"A", "B"}


# -- Bug 2: OUT_OF_MEMORY Slurm state forces oom classification -----------

class TestOOMClassification:
    """When slurm_state is OUT_OF_MEMORY, error_type should be 'oom'
    even if the .err file is empty.
    """

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_out_of_memory_empty_err(self):
        err_file = os.path.join(self.tmpdir, "test.err")
        with open(err_file, "w") as f:
            f.write("")

        job = {"err_file": err_file, "out_file": None}
        sacct_info = {"state": "OUT_OF_MEMORY", "exit_code": "0:125"}

        has_error, error_type, category, *_ = detect_error(job, sacct_info)
        assert has_error is True
        assert error_type == "oom"
        assert category == "slurm"

    def test_out_of_memory_with_err_content(self):
        err_file = os.path.join(self.tmpdir, "test.err")
        with open(err_file, "w") as f:
            f.write("slurmstepd: error: Detected 1 oom-kill event\n")

        job = {"err_file": err_file, "out_file": None}
        sacct_info = {"state": "OUT_OF_MEMORY", "exit_code": "0:125"}

        has_error, error_type, category, *_ = detect_error(job, sacct_info)
        assert has_error is True
        assert error_type == "oom"
        assert category == "slurm"

    def test_failed_state_not_forced_to_oom(self):
        """FAILED state with code error should remain as code error."""
        err_file = os.path.join(self.tmpdir, "test.err")
        with open(err_file, "w") as f:
            f.write("Traceback (most recent call last):\n  File 'x.py'\nValueError: bad\n")

        job = {"err_file": err_file, "out_file": None}
        sacct_info = {"state": "FAILED", "exit_code": "1:0"}

        has_error, error_type, category, *_ = detect_error(job, sacct_info)
        assert has_error is True
        assert error_type == "code"  # not forced to oom


# -- Timeout classification ------------------------------------------------

class TestTimeoutClassification:
    """When slurm_state is TIMEOUT, error_type should be 'timeout'
    even if the .err file is empty.
    """

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_timeout_empty_err(self):
        err_file = os.path.join(self.tmpdir, "test.err")
        with open(err_file, "w") as f:
            f.write("")

        job = {"err_file": err_file, "out_file": None}
        sacct_info = {"state": "TIMEOUT", "exit_code": "0:0"}

        has_error, error_type, category, *_ = detect_error(job, sacct_info)
        assert has_error is True
        assert error_type == "timeout"

    def test_timeout_with_err_content(self):
        err_file = os.path.join(self.tmpdir, "test.err")
        with open(err_file, "w") as f:
            f.write("slurmstepd: error: *** JOB 12345 ON node CANCELLED AT ... DUE TO TIME LIMIT ***\n")

        job = {"err_file": err_file, "out_file": None}
        sacct_info = {"state": "TIMEOUT", "exit_code": "0:0"}

        has_error, error_type, category, *_ = detect_error(job, sacct_info)
        assert has_error is True
        assert error_type == "timeout"


# -- auto_resubmit.py functions --------------------------------------------

class TestDoubleMemory:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_doubles_memory(self):
        script = os.path.join(self.tmpdir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\n#SBATCH --mem=128G\n#SBATCH --time=01:00:00\n")

        result = double_memory(script, 480)
        assert result == ("128", "256")

        with open(script) as f:
            content = f.read()
        assert "#SBATCH --mem=256G" in content

    def test_respects_cap(self):
        script = os.path.join(self.tmpdir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\n#SBATCH --mem=300G\n")

        result = double_memory(script, 480)
        assert result == ("300", "480")

        with open(script) as f:
            content = f.read()
        assert "#SBATCH --mem=480G" in content

    def test_no_mem_directive(self):
        script = os.path.join(self.tmpdir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\n#SBATCH --time=01:00:00\n")

        result = double_memory(script, 480)
        assert result is None


class TestDoubleTime:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_doubles_time(self):
        script = os.path.join(self.tmpdir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\n#SBATCH --mem=128G\n#SBATCH --time=01:00:00\n")

        result = double_time(script, 172800)  # 48h cap
        assert result == ("01:00:00", "02:00:00")

        with open(script) as f:
            content = f.read()
        assert "#SBATCH --time=02:00:00" in content

    def test_respects_cap(self):
        script = os.path.join(self.tmpdir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\n#SBATCH --time=30:00:00\n")

        result = double_time(script, 172800)  # 48h = 172800s
        assert result == ("30:00:00", "48:00:00")

        with open(script) as f:
            content = f.read()
        assert "#SBATCH --time=48:00:00" in content

    def test_no_time_directive(self):
        script = os.path.join(self.tmpdir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\n#SBATCH --mem=128G\n")

        result = double_time(script, 172800)
        assert result is None

    def test_non_round_time(self):
        script = os.path.join(self.tmpdir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\n#SBATCH --time=01:30:45\n")

        result = double_time(script, 172800)
        # 1h30m45s = 5445s, doubled = 10890s = 3h1m30s
        assert result == ("01:30:45", "03:01:30")

        with open(script) as f:
            content = f.read()
        assert "#SBATCH --time=03:01:30" in content


class TestTimeHelpers:
    def test_parse_time_to_seconds(self):
        assert parse_time_to_seconds("01:00:00") == 3600
        assert parse_time_to_seconds("48:00:00") == 172800
        assert parse_time_to_seconds("00:30:00") == 1800
        assert parse_time_to_seconds("01:30:45") == 5445

    def test_seconds_to_time(self):
        assert seconds_to_time(3600) == "01:00:00"
        assert seconds_to_time(172800) == "48:00:00"
        assert seconds_to_time(1800) == "00:30:00"
        assert seconds_to_time(5445) == "01:30:45"

    def test_parse_time_invalid(self):
        with pytest.raises(ValueError):
            parse_time_to_seconds("1:00")


class TestRetryableErrorTypes:
    def test_both_types_present(self):
        assert "OOM" in RETRYABLE_ERROR_TYPES
        assert "TIMEOUT" in RETRYABLE_ERROR_TYPES

    def test_load_errors_filters_retryable(self):
        """load_errors should return both OOM and TIMEOUT entries from new schema."""
        tmpdir = tempfile.mkdtemp()
        try:
            exp_dir = os.path.join(tmpdir, "experiments", "test_exp")
            os.makedirs(exp_dir)
            errors = {
                "experiment_name": "test_exp",
                "last_updated": "2026-01-01T00:00:00",
                "errors": [
                    {"job_id": "111", "error_type": "OOM",
                     "phase": "B", "grid_index": 3,
                     "timestamp": "2026-01-01T00:00:00",
                     "original_resources": {}, "retry_resources": None,
                     "resolved": False, "message": ""},
                    {"job_id": "222", "error_type": "TIMEOUT",
                     "phase": "C", "grid_index": "FGSM",
                     "timestamp": "2026-01-01T00:00:00",
                     "original_resources": {}, "retry_resources": None,
                     "resolved": False, "message": ""},
                    {"job_id": "100", "error_type": "RUNTIME",
                     "phase": "A", "grid_index": None,
                     "timestamp": "2026-01-01T00:00:00",
                     "original_resources": {}, "retry_resources": None,
                     "resolved": False, "message": ""},
                    {"job_id": "333", "error_type": "OOM",
                     "phase": "D", "grid_index": 0,
                     "timestamp": "2026-01-01T00:00:00",
                     "original_resources": {}, "retry_resources": None,
                     "resolved": True, "message": ""},
                ]
            }
            with open(os.path.join(exp_dir, "overall_errors.json"), "w") as f:
                json.dump(errors, f)

            orig_dir = os.getcwd()
            os.chdir(tmpdir)
            try:
                retryable, all_entries = load_errors("test_exp")
                # OOM (unresolved) and TIMEOUT (unresolved) are retryable
                assert len(retryable) == 2
                types = {e["error_type"] for e in retryable}
                assert types == {"OOM", "TIMEOUT"}
                assert len(all_entries) == 4
                # Verify internal field mapping
                oom_entry = next(e for e in all_entries if e["job_id"] == "111")
                assert oom_entry["step"] == "B"
                assert oom_entry["chunk"] == 3
                assert oom_entry["error_detected"] is True
                assert oom_entry["slurm_state"] == "FAILED"
                # Resolved entry should have error_detected=False and slurm_state=COMPLETED
                resolved_entry = next(e for e in all_entries if e["job_id"] == "333")
                assert resolved_entry["error_detected"] is False
                assert resolved_entry["slurm_state"] == "COMPLETED"
            finally:
                os.chdir(orig_dir)
        finally:
            shutil.rmtree(tmpdir)


class TestGetRetrySet:
    def test_single_oom_propagates_downstream(self):
        oom_entries = [{"step": "B", "chunk": 3, "job_id": "111",
                        "error_detected": True, "error_type": "oom"}]
        all_entries = [
            {"step": "A", "chunk": None, "job_id": "100",
             "slurm_state": "COMPLETED", "error_detected": False},
            {"step": "B", "chunk": 3, "job_id": "111",
             "slurm_state": "OUT_OF_MEMORY", "error_detected": True, "error_type": "oom"},
            {"step": "E", "chunk": None, "job_id": "200",
             "slurm_state": "CANCELLED", "error_detected": True, "error_type": "unknown"},
            {"step": "F", "chunk": None, "job_id": "300",
             "slurm_state": "CANCELLED", "error_detected": True, "error_type": "unknown"},
        ]

        failed_pairs, downstream_pairs, affected_steps = get_retry_set(oom_entries, all_entries)
        assert ("B", 3) in failed_pairs
        assert "E" in affected_steps
        assert "F" in affected_steps
        # A should not be affected (it completed)
        assert "A" not in affected_steps

    def test_no_failures(self):
        failed_pairs, downstream_pairs, affected_steps = get_retry_set([], [])
        assert len(failed_pairs) == 0
        assert len(downstream_pairs) == 0
        assert len(affected_steps) == 0

    def test_timeout_propagates_downstream(self):
        timeout_entries = [{"step": "C", "chunk": "FGSM", "job_id": "222",
                            "error_detected": True, "error_type": "timeout"}]
        all_entries = [
            {"step": "A", "chunk": None, "job_id": "100",
             "slurm_state": "COMPLETED", "error_detected": False},
            {"step": "C", "chunk": "FGSM", "job_id": "222",
             "slurm_state": "TIMEOUT", "error_detected": True, "error_type": "timeout"},
            {"step": "D", "chunk": 0, "job_id": "300",
             "slurm_state": "CANCELLED", "error_detected": True, "error_type": "unknown"},
        ]

        failed_pairs, downstream_pairs, affected_steps = get_retry_set(timeout_entries, all_entries)
        assert ("C", "FGSM") in failed_pairs
        assert "D" in affected_steps
        assert "E" in affected_steps


class TestStepToScript:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.exp = "test_exp"
        self.job_dir = os.path.join(self.tmpdir, "experiments", self.exp, "orchestrator_jobs")
        os.makedirs(self.job_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_step_a(self):
        path = step_to_script(self.exp, "A", None, False)
        assert path.endswith("step_A.sh")

    def test_step_b_chunk(self):
        path = step_to_script(self.exp, "B", "3", False)
        assert path.endswith("step_B_chunk_3.sh")

    def test_step_c_attack(self):
        path = step_to_script(self.exp, "C", "FGSM", False)
        assert path.endswith("step_C_attack_FGSM.sh")

    def test_step_d_chunk(self):
        path = step_to_script(self.exp, "D", "5", False)
        assert path.endswith("step_D_chunk_5.sh")

    def test_step_e(self):
        path = step_to_script(self.exp, "E", None, False)
        assert path.endswith("step_E.sh")

    def test_step_f(self):
        path = step_to_script(self.exp, "F", None, False)
        assert path.endswith("step_F.sh")

    def test_step_g(self):
        path = step_to_script(self.exp, "G", None, False)
        assert path.endswith("step_G.sh")

    def test_audit(self):
        path = step_to_script(self.exp, "AUDIT", None, False)
        assert path.endswith("final_audit.sh")


class TestRetryCount:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.exp_dir = os.path.join(self.tmpdir, "experiments", "test_exp")
        os.makedirs(self.exp_dir)
        # Temporarily change to tmpdir so file paths work
        self.orig_dir = os.getcwd()
        os.chdir(self.tmpdir)

    def teardown_method(self):
        os.chdir(self.orig_dir)
        shutil.rmtree(self.tmpdir)

    def test_first_retry(self):
        count = check_retry_count("test_exp", max_retries=2)
        assert count == 0
        # File should now contain 1 (using new counter name)
        with open(os.path.join(self.exp_dir, "auto_retry_count")) as f:
            assert f.read().strip() == "1"

    def test_second_retry(self):
        with open(os.path.join(self.exp_dir, "auto_retry_count"), "w") as f:
            f.write("1")
        count = check_retry_count("test_exp", max_retries=2)
        assert count == 1

    def test_max_retries_reached(self):
        with open(os.path.join(self.exp_dir, "auto_retry_count"), "w") as f:
            f.write("2")
        with pytest.raises(SystemExit):
            check_retry_count("test_exp", max_retries=2)

    def test_legacy_oom_retry_count_compat(self):
        """Should read from oom_retry_count if auto_retry_count doesn't exist."""
        with open(os.path.join(self.exp_dir, "oom_retry_count"), "w") as f:
            f.write("1")
        count = check_retry_count("test_exp", max_retries=2)
        assert count == 1
        # Should write to new counter name
        assert os.path.isfile(os.path.join(self.exp_dir, "auto_retry_count"))


class TestDependencyGraph:
    def test_topo_order_covers_all_steps(self):
        for step in DEPENDS_ON:
            assert step in TOPO_ORDER, f"Step {step} in DEPENDS_ON but not in TOPO_ORDER"

    def test_upstream_before_downstream(self):
        for step, upstreams in DEPENDS_ON.items():
            step_idx = TOPO_ORDER.index(step)
            for upstream in upstreams:
                up_idx = TOPO_ORDER.index(upstream)
                assert up_idx < step_idx, \
                    f"Upstream {upstream} (idx={up_idx}) should come before {step} (idx={step_idx})"
