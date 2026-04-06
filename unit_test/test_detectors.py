#!/usr/bin/env python
"""Unit tests for the 6 anomaly detectors in compare_representations.py.

Each test:
  1. Fits the detector on synthetic training data (2 Gaussian clusters)
  2. Verifies score output shape matches input
  3. Verifies AUROC > 0.5 on clearly separable anomalous data
"""
import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from compare_representations import (
        MahalanobisDetector, KNNDetector, KDEDetector,
        GMMDetector, OCSVMDetector, IsolationForestDetector,
        DETECTORS, compute_detection_metrics,
    )
except TypeError:
    # knowledgematrix uses tuple[int] syntax requiring Python 3.9+;
    # fall back to importing detectors directly from the module source
    # without triggering the utils.utils import chain.
    import importlib.util
    _mod_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "compare_representations.py")
    # Patch: load only the detector classes by exec'ing a trimmed version
    import types
    from sklearn.covariance import LedoitWolf
    from sklearn.neighbors import NearestNeighbors, KernelDensity
    from sklearn.mixture import GaussianMixture
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
    from sklearn.decomposition import TruncatedSVD

    _src = open(_mod_path).read()
    # Extract just the class definitions and DETECTORS/compute_detection_metrics
    _ns = {'np': np, 'LedoitWolf': LedoitWolf, 'TruncatedSVD': TruncatedSVD,
            'NearestNeighbors': NearestNeighbors, 'KernelDensity': KernelDensity,
            'GaussianMixture': GaussianMixture, 'OneClassSVM': OneClassSVM,
            'IsolationForest': IsolationForest,
            'roc_auc_score': roc_auc_score, 'roc_curve': roc_curve,
            'average_precision_score': average_precision_score}
    # Extract from "class MahalanobisDetector" to end of "def compute_detection_metrics"
    import re
    _match = re.search(r'(class MahalanobisDetector.*?)(# ---+\n# Main comparison logic)',
                       _src, re.DOTALL)
    if _match:
        exec(_match.group(1), _ns)
    MahalanobisDetector = _ns['MahalanobisDetector']
    KNNDetector = _ns['KNNDetector']
    KDEDetector = _ns['KDEDetector']
    GMMDetector = _ns['GMMDetector']
    OCSVMDetector = _ns['OCSVMDetector']
    IsolationForestDetector = _ns['IsolationForestDetector']
    DETECTORS = _ns['DETECTORS']
    compute_detection_metrics = _ns['compute_detection_metrics']


def make_synthetic_data(n_train=500, n_test_clean=200, n_test_adv=200,
                        dim=50, num_classes=5, seed=42):
    """Generate synthetic data with separable clean vs anomalous samples.

    Clean data: clusters around class means within a ball of radius 1.
    Adversarial data: points shifted far from all class means.
    """
    rng = np.random.RandomState(seed)
    class_means = rng.randn(num_classes, dim) * 3

    # Training data
    train_labels = rng.randint(0, num_classes, size=n_train)
    train_features = np.array([
        class_means[c] + rng.randn(dim) * 0.5
        for c in train_labels
    ])

    # Clean test data (from same distribution)
    test_labels = rng.randint(0, num_classes, size=n_test_clean)
    test_clean = np.array([
        class_means[c] + rng.randn(dim) * 0.5
        for c in test_labels
    ])

    # Adversarial test data (far from all class means)
    offset = rng.randn(dim) * 20
    test_adv = offset + rng.randn(n_test_adv, dim) * 0.5

    return train_features, train_labels, test_clean, test_adv, num_classes


class TestDetectorAPI(unittest.TestCase):
    """Test that all detectors follow the fit()/score() API contract."""

    @classmethod
    def setUpClass(cls):
        cls.train_feats, cls.train_labels, cls.test_clean, cls.test_adv, cls.num_classes = \
            make_synthetic_data()

    def _test_detector(self, detector):
        """Generic test for any detector instance."""
        detector.fit(self.train_feats, self.train_labels, self.num_classes)

        # Score shapes
        clean_scores = detector.score(self.test_clean)
        self.assertEqual(clean_scores.shape, (len(self.test_clean),),
                         f"Clean score shape mismatch for {type(detector).__name__}")

        adv_scores = detector.score(self.test_adv)
        self.assertEqual(adv_scores.shape, (len(self.test_adv),),
                         f"Adv score shape mismatch for {type(detector).__name__}")

        # AUROC should be well above 0.5 for clearly separable data
        metrics = compute_detection_metrics(clean_scores, adv_scores)
        self.assertGreater(metrics['auroc'], 0.5,
                           f"AUROC too low for {type(detector).__name__}: {metrics['auroc']:.3f}")

        return metrics

    def test_mahalanobis(self):
        metrics = self._test_detector(MahalanobisDetector(max_components=30))
        self.assertGreater(metrics['auroc'], 0.8)

    def test_knn(self):
        metrics = self._test_detector(KNNDetector(k=5, max_components=30))
        self.assertGreater(metrics['auroc'], 0.8)

    def test_kde(self):
        metrics = self._test_detector(KDEDetector(bandwidth=1.0, max_components=30))
        self.assertGreater(metrics['auroc'], 0.8)

    def test_gmm(self):
        metrics = self._test_detector(GMMDetector(n_components=5, max_components=30))
        self.assertGreater(metrics['auroc'], 0.8)

    def test_ocsvm(self):
        metrics = self._test_detector(OCSVMDetector(nu=0.05, max_components=30))
        self.assertGreater(metrics['auroc'], 0.8)

    def test_isolation_forest(self):
        metrics = self._test_detector(IsolationForestDetector(n_estimators=100, max_components=30))
        self.assertGreater(metrics['auroc'], 0.8)


class TestDETECTORSDict(unittest.TestCase):
    """Test the DETECTORS convenience dict."""

    def test_all_detectors_present(self):
        expected = {'Mahalanobis', 'KNN', 'KDE', 'GMM', 'OCSVM', 'IsolationForest'}
        self.assertEqual(set(DETECTORS.keys()), expected)

    def test_factories_return_correct_types(self):
        expected_types = {
            'Mahalanobis': MahalanobisDetector,
            'KNN': KNNDetector,
            'KDE': KDEDetector,
            'GMM': GMMDetector,
            'OCSVM': OCSVMDetector,
            'IsolationForest': IsolationForestDetector,
        }
        for name, factory in DETECTORS.items():
            det = factory()
            self.assertIsInstance(det, expected_types[name])


class TestComputeDetectionMetrics(unittest.TestCase):
    """Test the compute_detection_metrics helper."""

    def test_perfect_separation(self):
        clean = np.zeros(100)
        adv = np.ones(100)
        metrics = compute_detection_metrics(clean, adv)
        self.assertAlmostEqual(metrics['auroc'], 1.0, places=2)
        self.assertAlmostEqual(metrics['aupr'], 1.0, places=2)
        self.assertAlmostEqual(metrics['fpr_at_95tpr'], 0.0, places=2)

    def test_random_scores(self):
        rng = np.random.RandomState(0)
        clean = rng.randn(500)
        adv = rng.randn(500)
        metrics = compute_detection_metrics(clean, adv)
        # Random scores should give AUROC ~0.5
        self.assertAlmostEqual(metrics['auroc'], 0.5, delta=0.1)

    def test_output_keys(self):
        clean = np.zeros(10)
        adv = np.ones(10)
        metrics = compute_detection_metrics(clean, adv)
        expected_keys = {'auroc', 'aupr', 'fpr_at_95tpr', 'tpr_at_fpr5',
                         'tpr_at_fpr10', 'n_adv', 'n_clean'}
        self.assertEqual(set(metrics.keys()), expected_keys)
        self.assertEqual(metrics['n_clean'], 10)
        self.assertEqual(metrics['n_adv'], 10)


class TestHighDimensional(unittest.TestCase):
    """Test that SVD reduction works for high-dimensional inputs."""

    def test_svd_triggered(self):
        """When dim > max_components, SVD should reduce without error."""
        rng = np.random.RandomState(42)
        dim = 1000
        n = 200
        num_classes = 3
        feats = rng.randn(n, dim)
        labels = rng.randint(0, num_classes, size=n)

        for name, factory in DETECTORS.items():
            det = factory()
            det.fit(feats, labels, num_classes)
            scores = det.score(feats)
            self.assertEqual(scores.shape, (n,), f"Shape mismatch for {name} with SVD")


class TestSVDPath(unittest.TestCase):
    """Verify that the SVD attribute is set correctly based on dimensionality."""

    def test_svd_is_set_when_dim_exceeds_max_components(self):
        """When dim > max_components, detector.svd should NOT be None."""
        rng = np.random.RandomState(42)
        dim = 500
        max_comp = 50
        n = 200
        num_classes = 3
        feats = rng.randn(n, dim)
        labels = rng.randint(0, num_classes, size=n)

        for name, factory in DETECTORS.items():
            det = factory()
            det.max_components = max_comp
            det.fit(feats, labels, num_classes)
            self.assertIsNotNone(
                det.svd,
                f"{name}: svd should NOT be None when dim ({dim}) > max_components ({max_comp})")

    def test_svd_is_none_when_dim_within_max_components(self):
        """When dim <= max_components, detector.svd should be None."""
        rng = np.random.RandomState(42)
        dim = 20
        max_comp = 256  # default, well above dim
        n = 200
        num_classes = 3
        feats = rng.randn(n, dim)
        labels = rng.randint(0, num_classes, size=n)

        for name, factory in DETECTORS.items():
            det = factory()
            det.max_components = max_comp
            det.fit(feats, labels, num_classes)
            self.assertIsNone(
                det.svd,
                f"{name}: svd should be None when dim ({dim}) <= max_components ({max_comp})")


class TestSingleClassData(unittest.TestCase):
    """All training labels belong to a single class.

    MahalanobisDetector should use its global fallback for classes with
    insufficient data.  All detectors should fit and score without error.
    """

    def test_single_class_fit_and_score(self):
        rng = np.random.RandomState(7)
        n = 200
        dim = 30
        num_classes = 5  # declare 5 classes, but only class 2 appears
        feats = rng.randn(n, dim)
        labels = np.full(n, 2, dtype=int)

        test_pts = rng.randn(50, dim)

        for name, factory in DETECTORS.items():
            det = factory()
            # Should not raise
            det.fit(feats, labels, num_classes)
            scores = det.score(test_pts)
            self.assertEqual(scores.shape, (50,),
                             f"{name}: score shape mismatch on single-class data")
            self.assertTrue(np.all(np.isfinite(scores)),
                            f"{name}: non-finite scores on single-class data")


class TestZeroVarianceFeatures(unittest.TestCase):
    """Some feature columns are constant (zero variance).

    LedoitWolf covariance estimation and SVD must handle this gracefully.
    """

    def test_constant_columns_no_crash(self):
        rng = np.random.RandomState(99)
        n = 300
        dim = 40
        num_classes = 3
        feats = rng.randn(n, dim)
        # Set 10 columns to a constant value
        feats[:, 0:10] = 5.0

        labels = rng.randint(0, num_classes, size=n)
        test_pts = rng.randn(60, dim)
        test_pts[:, 0:10] = 5.0

        for name, factory in DETECTORS.items():
            det = factory()
            det.fit(feats, labels, num_classes)
            scores = det.score(test_pts)
            self.assertEqual(scores.shape, (60,),
                             f"{name}: score shape mismatch with zero-variance columns")
            self.assertTrue(np.all(np.isfinite(scores)),
                            f"{name}: non-finite scores with zero-variance columns")

    def test_constant_columns_high_dim_with_svd(self):
        """Zero-variance columns in a high-dim setting that triggers SVD."""
        rng = np.random.RandomState(99)
        n = 200
        dim = 500
        num_classes = 3
        feats = rng.randn(n, dim)
        # Half the columns constant
        feats[:, :250] = 0.0

        labels = rng.randint(0, num_classes, size=n)
        test_pts = rng.randn(40, dim)
        test_pts[:, :250] = 0.0

        for name, factory in DETECTORS.items():
            det = factory()
            det.fit(feats, labels, num_classes)
            scores = det.score(test_pts)
            self.assertEqual(scores.shape, (40,),
                             f"{name}: score shape mismatch with zero-var + SVD")
            self.assertTrue(np.all(np.isfinite(scores)),
                            f"{name}: non-finite scores with zero-var + SVD")


class TestScorePolarity(unittest.TestCase):
    """Core contract: higher score = more anomalous.

    For clearly separable data, mean(adv_scores) > mean(clean_scores)
    must hold for ALL 6 detectors.
    """

    @classmethod
    def setUpClass(cls):
        cls.train_feats, cls.train_labels, cls.test_clean, cls.test_adv, cls.num_classes = \
            make_synthetic_data(n_train=800, n_test_clean=300, n_test_adv=300,
                                dim=50, num_classes=5, seed=123)

    def test_adv_scores_higher_than_clean_for_all_detectors(self):
        for name, factory in DETECTORS.items():
            det = factory()
            det.fit(self.train_feats, self.train_labels, self.num_classes)

            clean_scores = det.score(self.test_clean)
            adv_scores = det.score(self.test_adv)

            self.assertGreater(
                np.mean(adv_scores), np.mean(clean_scores),
                f"{name}: mean(adv_scores)={np.mean(adv_scores):.4f} should be > "
                f"mean(clean_scores)={np.mean(clean_scores):.4f}")


class TestSmallDataset(unittest.TestCase):
    """Very small training set (n=10, dim=5, 2 classes).

    All detectors should still fit and score without errors.
    """

    def test_tiny_data_fit_and_score(self):
        rng = np.random.RandomState(0)
        n_train = 10
        dim = 5
        num_classes = 2
        feats = rng.randn(n_train, dim)
        labels = rng.randint(0, num_classes, size=n_train)

        test_pts = rng.randn(20, dim)

        for name, factory in DETECTORS.items():
            det = factory()
            det.fit(feats, labels, num_classes)
            scores = det.score(test_pts)
            self.assertEqual(scores.shape, (20,),
                             f"{name}: score shape mismatch on tiny dataset")
            self.assertTrue(np.all(np.isfinite(scores)),
                            f"{name}: non-finite scores on tiny dataset")


class TestEmptyInput(unittest.TestCase):
    """score() on an empty array (0 samples).

    Should either return an empty array or raise a clear error.
    Detectors are first fit on valid data, then scored on 0-row input.
    """

    @classmethod
    def setUpClass(cls):
        cls.train_feats, cls.train_labels, cls.test_clean, cls.test_adv, cls.num_classes = \
            make_synthetic_data(n_train=100, dim=20, seed=55)

    def test_empty_score_returns_empty_or_raises(self):
        empty_input = np.empty((0, 20))

        for name, factory in DETECTORS.items():
            det = factory()
            det.fit(self.train_feats, self.train_labels, self.num_classes)

            try:
                scores = det.score(empty_input)
                # If it returns, it must be an empty array
                self.assertEqual(
                    scores.shape[0], 0,
                    f"{name}: score on empty input should return 0-length array, "
                    f"got shape {scores.shape}")
            except (ValueError, IndexError) as e:
                # A clear error is also acceptable
                self.assertIsInstance(e, (ValueError, IndexError),
                                     f"{name}: unexpected error type on empty input: {type(e)}")


if __name__ == "__main__":
    unittest.main()
