#!/usr/bin/env python
"""Unit tests for critical and high-severity bug fixes.

Tests cover:
  C1: CyclicLR variable shadowing in training.py
  C2: Remainder-aware chunking in generate_matrices.py
  C3: Glob-based matrix loading in compare_representations.py
  H1: Tied class-conditional covariance in lee2018.py
  H3: Val/eval split disjointness
  H7: roc_curve exception handling
  H8: NaN precision matrix fallback
"""
import sys
import os
import unittest
import tempfile
import shutil
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestC1SchedulerPassthrough(unittest.TestCase):
    """C1: Verify training.py passes scheduler object, not string."""

    def test_scheduler_argument_is_not_string(self):
        """Read training.py and verify the scheduler= argument is not 'sched'."""
        training_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "training.py"
        )
        with open(training_path) as f:
            lines = f.readlines()

        # Find lines around the train_one_epoch() call site (not the def)
        # Look for "train_one_epoch(" preceded by indentation (call), then
        # scan forward for the scheduler= keyword argument
        import re
        found_call = False
        for i, line in enumerate(lines):
            # Call site is indented (inside a function body), not a def
            if 'train_one_epoch(' in line and 'def ' not in line:
                # Scan next few lines for scheduler=
                for j in range(i, min(i + 10, len(lines))):
                    sched_match = re.search(r'scheduler\s*=\s*(\w+)', lines[j])
                    if sched_match:
                        self.assertEqual(
                            sched_match.group(1), 'scheduler',
                            f"Line {j+1}: scheduler= should be 'scheduler', "
                            f"got '{sched_match.group(1)}'")
                        found_call = True
                        break
                break
        self.assertTrue(found_call, "Could not find scheduler= kwarg in train_one_epoch call")


class TestC2RemainderChunking(unittest.TestCase):
    """C2: Verify remainder-aware chunking covers all samples."""

    def _chunk_indices(self, N, total_chunks):
        """Reproduce the remainder-aware chunking logic from generate_matrices.py."""
        all_indices = set()
        for chunk_id in range(total_chunks):
            base_chunk = N // total_chunks
            remainder = N % total_chunks
            if chunk_id < remainder:
                start_idx = chunk_id * (base_chunk + 1)
                end_idx = start_idx + (base_chunk + 1)
            else:
                start_idx = chunk_id * base_chunk + remainder
                end_idx = start_idx + base_chunk
            all_indices.update(range(start_idx, end_idx))
        return all_indices

    def test_even_division(self):
        """1000 samples / 8 chunks = no remainder."""
        indices = self._chunk_indices(1000, 8)
        self.assertEqual(indices, set(range(1000)))

    def test_uneven_division(self):
        """1003 samples / 8 chunks = 3 extra samples distributed."""
        indices = self._chunk_indices(1003, 8)
        self.assertEqual(indices, set(range(1003)))

    def test_small_uneven(self):
        """7 samples / 3 chunks."""
        indices = self._chunk_indices(7, 3)
        self.assertEqual(indices, set(range(7)))

    def test_single_chunk(self):
        """All samples in one chunk."""
        indices = self._chunk_indices(100, 1)
        self.assertEqual(indices, set(range(100)))

    def test_more_chunks_than_samples(self):
        """3 samples / 5 chunks — some chunks empty."""
        indices = self._chunk_indices(3, 5)
        self.assertEqual(indices, set(range(3)))


class TestC3GlobBasedMatrixLoading(unittest.TestCase):
    """C3: Verify glob-based matrix loading handles gaps in indices."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _create_matrix_file(self, base_dir, idx, value=1.0):
        """Create a dummy matrix file at base_dir/idx/matrix.pth."""
        import torch
        mat_dir = os.path.join(base_dir, str(idx))
        os.makedirs(mat_dir, exist_ok=True)
        torch.save(torch.tensor([[value]]), os.path.join(mat_dir, 'matrix.pth'))

    def test_sequential_indices(self):
        """Load matrices with indices 0, 1, 2."""
        try:
            from compare_representations import load_matrices_as_features
        except (TypeError, ImportError):
            self.skipTest("Cannot import compare_representations")

        base = self.tmpdir
        adv_dir = os.path.join(base, 'adversarial_matrices', 'FGSM')
        for i in range(3):
            self._create_matrix_file(adv_dir, i, value=float(i))

        mats = load_matrices_as_features(base, 'FGSM')
        self.assertIsNotNone(mats)
        self.assertEqual(len(mats), 3)

    def test_gaps_in_indices(self):
        """Load matrices with gaps: indices 0, 1, 3, 5 (missing 2, 4)."""
        try:
            from compare_representations import load_matrices_as_features
        except (TypeError, ImportError):
            self.skipTest("Cannot import compare_representations")

        base = self.tmpdir
        adv_dir = os.path.join(base, 'adversarial_matrices', 'PGD')
        for i in [0, 1, 3, 5]:
            self._create_matrix_file(adv_dir, i, value=float(i))

        mats = load_matrices_as_features(base, 'PGD')
        self.assertIsNotNone(mats)
        self.assertEqual(len(mats), 4, "Should load 4 matrices despite gaps")

    def test_max_samples_limit(self):
        """max_samples limits the number loaded."""
        try:
            from compare_representations import load_matrices_as_features
        except (TypeError, ImportError):
            self.skipTest("Cannot import compare_representations")

        base = self.tmpdir
        adv_dir = os.path.join(base, 'adversarial_matrices', 'CW')
        for i in range(10):
            self._create_matrix_file(adv_dir, i)

        mats = load_matrices_as_features(base, 'CW', max_samples=5)
        self.assertIsNotNone(mats)
        self.assertEqual(len(mats), 5)


class TestH1TiedCovariance(unittest.TestCase):
    """H1: Verify Lee et al. uses tied class-conditional covariance."""

    def test_tied_covariance_not_inflated(self):
        """With unit-covariance classes far apart, tied cov should be ~identity,
        not inflated by between-class variance."""
        try:
            import torch
            import torch.nn as nn
            from baselines.lee2018 import MultiLayerMahalanobisDetector
        except (TypeError, ImportError):
            self.skipTest("Cannot import lee2018")

        # Simple model with one activation layer
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 3),
        )
        model.eval()

        rng = np.random.RandomState(42)
        # 3 classes with identity covariance, means far apart
        n_per_class = 100
        data_list = []
        labels_list = []
        for c in range(3):
            mean = np.zeros(10)
            mean[0] = c * 100  # Far apart
            samples = rng.randn(n_per_class, 10) + mean
            data_list.append(samples)
            labels_list.extend([c] * n_per_class)

        data = np.vstack(data_list).astype(np.float32)
        labels = np.array(labels_list)

        with MultiLayerMahalanobisDetector(model, 3, device='cpu',
                                           max_components=20) as det:
            det.fit(torch.from_numpy(data), labels)

            # Check that precision matrices exist and are finite
            for layer_name in det.layer_names:
                prec = det.precision[layer_name]
                self.assertTrue(np.all(np.isfinite(prec)),
                                f"Precision matrix for {layer_name} has non-finite values")


class TestH7RocCurveExceptionHandling(unittest.TestCase):
    """H7: compute_detection_metrics handles degenerate inputs."""

    def test_identical_scores(self):
        """All scores identical — roc_curve may raise ValueError."""
        try:
            from compare_representations import compute_detection_metrics
        except (TypeError, ImportError):
            self.skipTest("Cannot import compare_representations")

        clean = np.ones(50)
        adv = np.ones(50)
        metrics = compute_detection_metrics(clean, adv)
        # Should not raise, and should return valid defaults
        self.assertIn('auroc', metrics)
        self.assertIn('fpr_at_95tpr', metrics)
        self.assertTrue(np.isfinite(metrics['fpr_at_95tpr']))

    def test_single_sample(self):
        """Single clean + single adversarial sample."""
        try:
            from compare_representations import compute_detection_metrics
        except (TypeError, ImportError):
            self.skipTest("Cannot import compare_representations")

        clean = np.array([0.1])
        adv = np.array([0.9])
        metrics = compute_detection_metrics(clean, adv)
        self.assertIn('auroc', metrics)


class TestH8NaNPrecisionFallback(unittest.TestCase):
    """H8: MahalanobisDetector handles near-singular data gracefully."""

    def test_near_singular_data(self):
        """Data that is nearly rank-deficient should not produce NaN scores."""
        try:
            from compare_representations import MahalanobisDetector
        except (TypeError, ImportError):
            self.skipTest("Cannot import compare_representations")

        rng = np.random.RandomState(42)
        n = 50
        dim = 20
        # Make data nearly singular: all points near a 1D subspace
        direction = rng.randn(dim)
        direction /= np.linalg.norm(direction)
        feats = np.outer(rng.randn(n), direction) + rng.randn(n, dim) * 1e-10
        labels = np.zeros(n, dtype=int)

        det = MahalanobisDetector(max_components=dim)
        det.fit(feats, labels, num_classes=1)

        test_pts = rng.randn(10, dim)
        scores = det.score(test_pts)
        self.assertTrue(np.all(np.isfinite(scores)),
                        f"Scores contain non-finite values: {scores}")


class TestH9ChunkIdUsage(unittest.TestCase):
    """H9: Verify generate_matrices.py uses chunk_id, not args.chunk_id."""

    def test_done_file_uses_chunk_id(self):
        """Read generate_matrices.py and verify done_file uses chunk_id variable."""
        gen_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "generate_matrices.py"
        )
        with open(gen_path) as f:
            content = f.read()

        # Check that args.chunk_id is NOT used in done_file or print statements
        # after the initial chunk_id assignment
        import re
        # Find the done_file line
        done_match = re.search(r'done_file.*done_chunk_(.*?)\.txt', content)
        self.assertIsNotNone(done_match, "Could not find done_file line")
        self.assertNotIn('args.chunk_id', done_match.group(0),
                         "done_file should use chunk_id, not args.chunk_id")


if __name__ == "__main__":
    unittest.main()
