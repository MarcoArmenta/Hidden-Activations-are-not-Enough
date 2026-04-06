"""
Fair comparison of adversarial detection across representations.

Applies the SAME set of 6 detectors to three representations:
  1. Knowledge Matrices    — M(W,f)(x), the full forward-pass matrix
  2. Penultimate Features  — activations from the last hidden layer
  3. All-Layer Features    — concatenation of all hidden-layer activations

By using the same detectors, any performance difference is due entirely to
the REPRESENTATION, not the detection algorithm.  This is the core experiment
for the claim "Hidden Activations Are Not Enough."

Usage:
    # On cluster (after pipeline steps A-F complete):
    python compare_representations.py --experiment lenet_cifar10
    python compare_representations.py --experiment alexnet_cifar10 --temp_dir $SLURM_TMPDIR

    # SVD rank ablation:
    python compare_representations.py --experiment alexnet_cifar10 --svd_ablation

    # Run for multiple experiments:
    python compare_representations.py --experiment lenet_cifar10 alexnet_cifar10 resnet_cifar10 vgg_cifar10
"""

import gc
import os
import json
import time
import random
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from typing import Union
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.decomposition import TruncatedSVD

from utils.utils import (
    get_model, get_dataset, get_input_shape, get_num_classes,
    get_device, subset,
)
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS, ATTACK_CATEGORIES
from baselines.lee2018 import MultiLayerMahalanobisDetector
from utils.atomic_io import atomic_json_dump
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

class AllLayerExtractor:
    """Extracts activations from every layer using forward hooks.
    Architecture-agnostic: works with any nn.Module."""

    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.ReLU, torch.nn.ELU, torch.nn.Tanh,
                                   torch.nn.LeakyReLU, torch.nn.PReLU,
                                   torch.nn.Sigmoid, torch.nn.GELU)):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            self.features[name] = output.detach().cpu()
        return hook_fn

    def extract(self, x):
        """Run forward pass and return concatenated features from all activation layers."""
        self.features = {}
        with torch.no_grad():
            _ = self.model(x)
        if not self.features:
            return None
        parts = []
        for name in sorted(self.features.keys()):
            feat = self.features[name]
            parts.append(feat.reshape(feat.shape[0], -1))
        return torch.cat(parts, dim=1).numpy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def extract_penultimate_features(model, data, batch_size=128):
    """Extract penultimate-layer features in batches."""
    model.eval()
    device = next(model.parameters()).device
    feats = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device).float()
            f = model.forward(batch, return_penultimate=True)
            feats.append(f.detach().cpu().numpy().reshape(f.shape[0], -1))
    return np.vstack(feats) if feats else np.zeros((0, 1))


def extract_all_layer_features(model, data, batch_size=128):
    """Extract concatenated all-layer activation features."""
    device = next(model.parameters()).device
    with AllLayerExtractor(model) as extractor:
        feats = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device).float()
            f = extractor.extract(batch)
            if f is not None:
                feats.append(f)
    return np.vstack(feats) if feats else np.zeros((0, 1))


def load_matrices_as_features(base_path, attack_name, max_samples=None):
    """Load pre-computed knowledge matrices and flatten them into feature vectors.
    Uses glob-based discovery to handle gaps in index numbering."""
    mat_dir = Path(base_path) / 'adversarial_matrices' / attack_name
    if not mat_dir.exists():
        return None
    # M6: Filter out non-numeric directory names, sort safely
    mat_paths = [p for p in mat_dir.glob('*/matrix.pth') if p.parent.name.isdigit()]
    mat_paths = sorted(mat_paths, key=lambda p: int(p.parent.name))
    if max_samples:
        mat_paths = mat_paths[:max_samples]
    mats = []
    for mp in mat_paths:
        m = torch.load(mp, map_location='cpu', weights_only=True)
        mats.append(m.numpy().ravel())
    return np.array(mats) if mats else None


def load_train_matrices(base_path, num_classes, per_class=1000):
    """Load training matrices and flatten.
    Uses glob-based discovery to handle gaps in index numbering."""
    mat_dir = Path(base_path) / 'matrices'
    mats = []
    labels = []
    for c in range(num_classes):
        class_dir = mat_dir / str(c)
        if not class_dir.exists():
            continue
        # Discover all matrix files, sorted by index (filter non-numeric dirs)
        mat_paths = [p for p in class_dir.glob('*/matrix.pt') if p.parent.name.isdigit()]
        mat_paths = sorted(mat_paths, key=lambda p: int(p.parent.name))
        for mp in mat_paths[:per_class]:
            m = torch.load(mp, map_location='cpu', weights_only=True)
            mats.append(m.numpy().ravel())
            labels.append(c)
    return np.array(mats), np.array(labels)


def compute_km_features_on_the_fly(model, images, device, batch_size=1800):
    """Compute knowledge matrices on-the-fly and return as flattened numpy array.
    Fallback for when pre-computed test KMs are missing."""
    matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)
    mats = []
    for i in range(len(images)):
        try:
            mat = matrix_computer.forward(images[i].to(device))
            mats.append(mat.cpu().numpy().ravel())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                gc.collect()
                torch.cuda.empty_cache()
                old_bs = matrix_computer.batch_size
                new_bs = max(1, old_bs // 2)
                print(f"    OOM computing KM {i}, halving batch_size {old_bs}->{new_bs}", flush=True)
                matrix_computer = KnowledgeMatrixComputer(model, batch_size=new_bs, device=device)
                mat = matrix_computer.forward(images[i].to(device))
                mats.append(mat.cpu().numpy().ravel())
            else:
                raise
        if (i + 1) % 100 == 0:
            print(f"    Computed {i+1}/{len(images)} test KMs on-the-fly", flush=True)
    return np.array(mats) if mats else None


# ---------------------------------------------------------------------------
# Mahalanobis detector (same for all representations)
# ---------------------------------------------------------------------------

class MahalanobisDetector:
    """Per-class Mahalanobis distance detector with LedoitWolf covariance.
    Uses TruncatedSVD to handle high-dimensional features."""

    def __init__(self, max_components=256):
        self.max_components = max_components
        self.svd = None
        self.class_means = {}
        self.class_precisions = {}

    def fit(self, features, labels, num_classes):
        """Fit the detector on training data."""
        N, D = features.shape
        n_comp = min(self.max_components, D, max(1, N - 1))

        # Reduce dimensionality if needed
        if D > n_comp:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
            proj = self.svd.fit_transform(features)
        else:
            self.svd = None
            proj = features

        # Global fallback
        try:
            lw_global = LedoitWolf().fit(proj)
            global_mean = lw_global.location_
            global_prec = lw_global.precision_
        except Exception:
            global_mean = np.mean(proj, axis=0)
            global_prec = np.eye(proj.shape[1])

        # Per-class statistics
        for c in range(num_classes):
            mask = (labels == c)
            class_data = proj[mask]
            if class_data.shape[0] < 2:
                self.class_means[c] = global_mean
                self.class_precisions[c] = global_prec
            else:
                try:
                    lw = LedoitWolf().fit(class_data)
                    prec = lw.precision_
                    if np.any(np.isnan(prec)):
                        import warnings
                        warnings.warn(f"NaN in precision matrix for class {c}, using identity fallback")
                        prec = np.eye(proj.shape[1])
                    self.class_means[c] = lw.location_
                    self.class_precisions[c] = prec
                except Exception:
                    self.class_means[c] = global_mean
                    self.class_precisions[c] = global_prec

    def score(self, features):
        """Compute min Mahalanobis distance to any class (higher = more anomalous)."""
        if self.svd is not None:
            proj = self.svd.transform(features.reshape(features.shape[0], -1))
        else:
            proj = features.reshape(features.shape[0], -1)

        N = proj.shape[0]
        min_dists = np.full(N, np.inf)
        for c in self.class_means:
            diff = proj - self.class_means[c]
            prec = self.class_precisions[c]
            dists_sq = np.einsum('ij,jk,ik->i', diff, prec, diff)
            min_dists = np.minimum(min_dists, np.sqrt(np.maximum(dists_sq, 0.0)))
        return min_dists


class KNNDetector:
    """K-nearest neighbor anomaly detector with TruncatedSVD.
    Score = mean distance to k nearest neighbors (higher = more anomalous)."""

    def __init__(self, k=10, max_components=256):
        self.k = k
        self.max_components = max_components
        self.svd = None
        self.knn = None

    def fit(self, features, labels, num_classes):
        N, D = features.shape
        n_comp = min(self.max_components, D, max(1, N - 1))
        if D > n_comp:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
            proj = self.svd.fit_transform(features)
        else:
            self.svd = None
            proj = features
        self.knn = NearestNeighbors(n_neighbors=max(1, min(self.k, len(proj) - 1)),
                                    metric='euclidean')
        self.knn.fit(proj)

    def score(self, features):
        proj = self.svd.transform(features.reshape(features.shape[0], -1)) \
            if self.svd is not None else features.reshape(features.shape[0], -1)
        distances, _ = self.knn.kneighbors(proj)
        return distances.mean(axis=1)


class KDEDetector:
    """Kernel Density Estimation anomaly detector with TruncatedSVD.
    Score = negative log-likelihood (higher = more anomalous)."""

    def __init__(self, bandwidth='scott', max_components=256):
        self.bandwidth = bandwidth
        self.max_components = max_components
        self.svd = None
        self.kde = None

    def fit(self, features, labels, num_classes):
        N, D = features.shape
        n_comp = min(self.max_components, D, max(1, N - 1))
        if D > n_comp:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
            proj = self.svd.fit_transform(features)
        else:
            self.svd = None
            proj = features
        self.kde = KernelDensity(bandwidth=self.bandwidth)
        self.kde.fit(proj)

    def score(self, features):
        proj = self.svd.transform(features.reshape(features.shape[0], -1)) \
            if self.svd is not None else features.reshape(features.shape[0], -1)
        return -self.kde.score_samples(proj)  # negate: higher = more anomalous


class GMMDetector:
    """Gaussian Mixture Model anomaly detector with TruncatedSVD.
    Score = negative log-likelihood (higher = more anomalous)."""

    def __init__(self, n_components=10, max_components=256):
        self.n_gmm_components = n_components
        self.max_components = max_components
        self.svd = None
        self.gmm = None

    def fit(self, features, labels, num_classes):
        N, D = features.shape
        n_comp = min(self.max_components, D, max(1, N - 1))
        if D > n_comp:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
            proj = self.svd.fit_transform(features)
        else:
            self.svd = None
            proj = features
        n_gmm = min(self.n_gmm_components, len(proj), proj.shape[1])
        n_gmm = max(1, n_gmm)
        self.gmm = GaussianMixture(n_components=n_gmm,
                                   covariance_type='diag',
                                   random_state=0)
        self.gmm.fit(proj)

    def score(self, features):
        proj = self.svd.transform(features.reshape(features.shape[0], -1)) \
            if self.svd is not None else features.reshape(features.shape[0], -1)
        return -self.gmm.score_samples(proj)  # negate: higher = more anomalous


class OCSVMDetector:
    """One-Class SVM anomaly detector with TruncatedSVD.
    Score = negative decision function (higher = more anomalous)."""

    def __init__(self, nu=0.05, max_components=256):
        self.nu = nu
        self.max_components = max_components
        self.svd = None
        self.ocsvm = None

    def fit(self, features, labels, num_classes):
        N, D = features.shape
        n_comp = min(self.max_components, D, max(1, N - 1))
        if D > n_comp:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
            proj = self.svd.fit_transform(features)
        else:
            self.svd = None
            proj = features
        self.ocsvm = OneClassSVM(kernel='rbf', nu=self.nu)
        self.ocsvm.fit(proj)

    def score(self, features):
        proj = self.svd.transform(features.reshape(features.shape[0], -1)) \
            if self.svd is not None else features.reshape(features.shape[0], -1)
        return -self.ocsvm.decision_function(proj)  # negate: higher = more anomalous


class IsolationForestDetector:
    """Isolation Forest anomaly detector with TruncatedSVD.
    Score = negative anomaly score (higher = more anomalous)."""

    def __init__(self, n_estimators=100, max_components=256):
        self.n_estimators = n_estimators
        self.max_components = max_components
        self.svd = None
        self.iforest = None

    def fit(self, features, labels, num_classes):
        N, D = features.shape
        n_comp = min(self.max_components, D, max(1, N - 1))
        if D > n_comp:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
            proj = self.svd.fit_transform(features)
        else:
            self.svd = None
            proj = features
        self.iforest = IsolationForest(n_estimators=self.n_estimators, random_state=0)
        self.iforest.fit(proj)

    def score(self, features):
        proj = self.svd.transform(features.reshape(features.shape[0], -1)) \
            if self.svd is not None else features.reshape(features.shape[0], -1)
        return -self.iforest.decision_function(proj)  # negate: higher = more anomalous


# All detectors with default hyperparameters
DETECTORS = {
    'Mahalanobis': lambda: MahalanobisDetector(max_components=256),
    'KNN': lambda: KNNDetector(k=10, max_components=256),
    'KDE': lambda: KDEDetector(bandwidth='scott', max_components=256),
    'GMM': lambda: GMMDetector(n_components=10, max_components=256),
    'OCSVM': lambda: OCSVMDetector(nu=0.05, max_components=256),
    'IsolationForest': lambda: IsolationForestDetector(n_estimators=100, max_components=256),
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_detection_metrics(clean_scores, adv_scores):
    """Compute AUROC, AUPR, FPR@95TPR from clean and adversarial scores.
    Convention: higher score = more anomalous."""
    n_clean = len(clean_scores)
    n_adv = len(adv_scores)
    labels = np.concatenate([np.zeros(n_clean), np.ones(n_adv)])
    scores = np.concatenate([clean_scores, adv_scores])

    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.5

    try:
        aupr = average_precision_score(labels, scores)
    except ValueError:
        aupr = 0.0

    # FPR at 95% TPR
    try:
        fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
        # HI-6: Use interpolation for accurate FPR@95TPR on coarse ROC curves
        fpr_at_95tpr = float(np.interp(0.95, tpr_arr, fpr_arr))

        # TPR at fixed FPR thresholds
        # HI-5: Clamp index to avoid -1 wraparound
        if len(fpr_arr) > 1:
            idx_5 = max(0, np.searchsorted(fpr_arr, 0.05, side='right') - 1)
            tpr_at_5 = tpr_arr[idx_5]
            idx_10 = max(0, np.searchsorted(fpr_arr, 0.10, side='right') - 1)
            tpr_at_10 = tpr_arr[idx_10]
        else:
            tpr_at_5 = 0
            tpr_at_10 = 0
    except ValueError:
        fpr_at_95tpr = 1.0
        tpr_at_5 = 0.0
        tpr_at_10 = 0.0

    return {
        'auroc': float(auroc),
        'aupr': float(aupr),
        'fpr_at_95tpr': float(fpr_at_95tpr),
        'tpr_at_fpr5': float(tpr_at_5),
        'tpr_at_fpr10': float(tpr_at_10),
        'n_adv': n_adv,
        'n_clean': n_clean,
    }


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def run_comparison(experiment_name, temp_dir=None, svd_ablation=False):
    """Run fair comparison for one experiment.

    Fits all 6 detectors on each of 3 representations and evaluates on
    every available adversarial attack, producing AUROC/AUPR/FPR@95TPR.

    If svd_ablation=True, additionally sweeps SVD rank for Mahalanobis.
    """
    # NEW-H13: Global random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    exp_config = DEFAULT_EXPERIMENTS[experiment_name]
    dataset = exp_config['dataset']
    arch_idx = exp_config['architecture_index']
    epoch = exp_config['epochs']

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)
    device = get_device()

    base = f'{temp_dir}/experiments/{experiment_name}' if temp_dir else f'experiments/{experiment_name}'

    # Training saves epoch_{epochs-1}.pth as final checkpoint; try both conventions
    weights_dir = Path(base) / 'weights'
    weights_path = None
    for candidate_epoch in [epoch, epoch - 1]:
        candidate = weights_dir / f'epoch_{candidate_epoch}.pth'
        if candidate.exists():
            weights_path = candidate
            break
    if weights_path is None:
        epoch_files = sorted(weights_dir.glob('epoch_*.pth'),
                             key=lambda p: int(p.stem.split('_')[1]))
        if epoch_files:
            weights_path = epoch_files[-1]
        else:
            print(f"  Skipping {experiment_name}: no weights in {weights_dir}")
            return None
    print(f"  Using weights: {weights_path.name}")

    # Check if adversarial matrices exist
    adv_mat_dir = Path(base) / 'adversarial_matrices'
    if not adv_mat_dir.exists():
        print(f"  Skipping {experiment_name}: no adversarial matrices")
        return None

    print(f"\n{'='*70}", flush=True)
    print(f"  EXPERIMENT: {experiment_name}", flush=True)
    _ARCH_NAMES = {-4: 'LeNet', -3: 'AlexNet', -2: 'ResNet18', -1: 'VGG11'}
    print(f"  Architecture: {_ARCH_NAMES.get(arch_idx, 'custom')}", flush=True)
    print(f"  Dataset: {dataset}, Classes: {num_classes}", flush=True)
    print(f"{'='*70}", flush=True)

    model = get_model(weights_path, arch_idx, input_shape, num_classes, device)
    model.eval()

    # -----------------------------------------------------------------------
    # 1. Load training data for fitting detectors
    # -----------------------------------------------------------------------
    print("  Loading training data...", flush=True)
    train_set, _ = get_dataset(dataset, data_loader=False, data_path=temp_dir)
    # HI-7: Reconstruct the exact training indices used by Step 2a's KM
    # computation (first 500 per class, sequentially) so penultimate/all-layer
    # features are extracted from the same samples as KM training data.
    per_class = 500
    train_indices = []
    class_indices = {}
    for idx in range(len(train_set)):
        _, label = train_set[idx]
        label = int(label)
        class_indices.setdefault(label, []).append(idx)
    for c in sorted(class_indices.keys()):
        train_indices.extend(class_indices[c][:per_class])
    # Build aligned training tensors
    train_data_list = []
    train_labels_list = []
    for idx in train_indices:
        img, label = train_set[idx]
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        train_data_list.append(img)
        train_labels_list.append(int(label))
    train_data = torch.stack(train_data_list)
    train_labels_np = np.array(train_labels_list, dtype=int)
    print(f"    Training samples: {len(train_data)} ({per_class} per class x {num_classes} classes)", flush=True)

    # -----------------------------------------------------------------------
    # 2. Extract training representations (with cost measurement)
    # -----------------------------------------------------------------------
    computational_cost = {}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    print("  Extracting training PENULTIMATE features...", flush=True)
    train_penult = extract_penultimate_features(model, train_data)
    cost_penult_time = time.perf_counter() - t0
    cost_penult_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"    Shape: {train_penult.shape}  ({cost_penult_time:.1f}s)", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    print("  Extracting training ALL-LAYER features...", flush=True)
    train_alllayer = extract_all_layer_features(model, train_data)
    cost_alllayer_time = time.perf_counter() - t0
    cost_alllayer_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"    Shape: {train_alllayer.shape}  ({cost_alllayer_time:.1f}s)", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    print("  Loading training KNOWLEDGE MATRICES...", flush=True)
    train_matrices, train_mat_labels = load_train_matrices(base, num_classes, per_class=500)
    cost_matrix_time = time.perf_counter() - t0
    cost_matrix_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    if train_matrices is None or len(train_matrices) == 0:
        raise RuntimeError(
            f"FATAL: No training matrices found at {Path(base) / 'matrices'}. "
            f"Step 2a output is missing or zip extraction failed. "
            f"Check that matrices_task_*.zip files exist and were extracted."
        )
    else:
        print(f"    Shape: {train_matrices.shape}  ({cost_matrix_time:.1f}s)", flush=True)
        train_mat_labels = train_mat_labels.astype(int)

    n_train = len(train_data)
    computational_cost = {
        'penultimate': {
            'seconds_per_1000': float(cost_penult_time * 1000 / n_train),
            'peak_gpu_memory_gb': float(cost_penult_mem),
            'feature_dim': int(train_penult.shape[1]),
        },
        'all_layer': {
            'seconds_per_1000': float(cost_alllayer_time * 1000 / n_train),
            'peak_gpu_memory_gb': float(cost_alllayer_mem),
            'feature_dim': int(train_alllayer.shape[1]),
        },
    }
    if train_matrices is not None:
        computational_cost['knowledge_matrix'] = {
            'loading_seconds_per_1000': float(cost_matrix_time * 1000 / len(train_matrices)),
            'peak_gpu_memory_gb': float(cost_matrix_mem),
            'feature_dim': int(train_matrices.shape[1]),
            'note': 'loading_seconds measures disk I/O only; matrix computation cost is in Step 2a',
        }

    # -----------------------------------------------------------------------
    # 3. Build training representation dict
    # -----------------------------------------------------------------------
    train_reps = {
        'penultimate': (train_penult, train_labels_np),
        'all_layer': (train_alllayer, train_labels_np),
    }
    if train_matrices is not None:
        train_reps['knowledge_matrix'] = (train_matrices, train_mat_labels)

    rep_names = list(train_reps.keys())

    # -----------------------------------------------------------------------
    # 4. Fit all 6 detectors on each representation
    # -----------------------------------------------------------------------
    # fitted_detectors[det_name][rep_name] = fitted detector instance
    fitted_detectors = {}
    for det_name, det_factory in DETECTORS.items():
        fitted_detectors[det_name] = {}
        for rn in rep_names:
            feats, labs = train_reps[rn]
            print(f"  Fitting {det_name} on {rn}...", flush=True)
            det = det_factory()
            det.fit(feats, labs, num_classes)
            fitted_detectors[det_name][rn] = det

    # -----------------------------------------------------------------------
    # 5. Score clean test data
    # -----------------------------------------------------------------------
    print("  Scoring clean test data...", flush=True)
    # CR-3: Use canonical test set from Step 2b if available, ensuring all
    # representations evaluate on the exact same images.
    canonical_test_path = Path(base) / 'adversarial_examples' / 'test' / 'adversarial_examples.pth'
    if canonical_test_path.exists():
        test_data = torch.load(canonical_test_path, map_location='cpu', weights_only=True)
        if len(test_data) > 2000:
            test_data = test_data[:2000]
        print(f"    Using canonical test set from Step 2b ({len(test_data)} samples)")
    else:
        _, test_set = get_dataset(dataset, data_loader=False, data_path=temp_dir)
        test_data, _ = subset(test_set, 2000, input_shape)
        print(f"    Using random test subset ({len(test_data)} samples)")

    # If KM test matrices exist, align sample count; otherwise compute on-the-fly
    if 'knowledge_matrix' in rep_names:
        test_mats = load_matrices_as_features(base, 'test', max_samples=2000)
        if test_mats is None or len(test_mats) == 0:
            # Fallback: compute test KMs on-the-fly
            print(f"    WARNING: No pre-computed test KMs at "
                  f"{Path(base) / 'adversarial_matrices' / 'test'}")
            print(f"    Computing test knowledge matrices on-the-fly "
                  f"({len(test_data)} samples)...", flush=True)
            test_mats = compute_km_features_on_the_fly(model, test_data, device)
            if test_mats is None or len(test_mats) == 0:
                raise RuntimeError(
                    f"FATAL: Could not compute test knowledge matrices. "
                    f"Check GPU memory and model state."
                )
            print(f"    Computed {len(test_mats)} test KMs on-the-fly")
        # Align sample counts
        n_km_test = len(test_mats)
        n_clean = min(n_km_test, len(test_data))
        if n_clean < len(test_data):
            print(f"    Aligning clean test samples: {len(test_data)} -> {n_clean} "
                  f"(KM test count)")
            test_data = test_data[:n_clean]
        test_mats = test_mats[:n_clean]

    # Extract test representations once
    test_feats = {
        'penultimate': extract_penultimate_features(model, test_data),
        'all_layer': extract_all_layer_features(model, test_data),
    }
    if 'knowledge_matrix' in rep_names:
        test_feats['knowledge_matrix'] = test_mats

    # clean_scores[det_name][rep_name] = 1D array of scores
    clean_scores = {}
    for det_name in DETECTORS:
        clean_scores[det_name] = {}
        for rn in rep_names:
            if rn in fitted_detectors[det_name] and rn in test_feats:
                clean_scores[det_name][rn] = fitted_detectors[det_name][rn].score(test_feats[rn])

    # -----------------------------------------------------------------------
    # 6. Score adversarial data per attack
    # -----------------------------------------------------------------------
    results = {}
    available_attacks = []
    for attack in ATTACKS:
        adv_path = Path(base) / 'adversarial_examples' / attack / 'adversarial_examples.pth'
        if adv_path.exists():
            available_attacks.append(attack)

    if not available_attacks:
        print("  No adversarial examples found!")
        return None

    print(f"  Found {len(available_attacks)} attacks: {available_attacks}", flush=True)

    for attack in available_attacks:
        print(f"  Scoring attack: {attack}...", flush=True)
        adv_data = torch.load(
            Path(base) / 'adversarial_examples' / attack / 'adversarial_examples.pth',
            map_location='cpu', weights_only=True)
        if len(adv_data) > 2000:
            adv_data = adv_data[:2000]

        # NEW-C3: Load KM adversarial features first to determine aligned count.
        # Truncate adv_data BEFORE extracting penultimate/all-layer features
        # so all representations use the exact same adversarial samples.
        km_feats = None
        if 'knowledge_matrix' in rep_names:
            km_feats = load_matrices_as_features(base, attack, max_samples=2000)
            if km_feats is not None and len(km_feats) > 0:
                n_km = len(km_feats)
                if n_km < len(adv_data):
                    print(f"    Aligning adv samples for {attack}: {len(adv_data)} -> {n_km} "
                          f"(KM count)")
                    adv_data = adv_data[:n_km]
                    km_feats = km_feats[:len(adv_data)]

        # Extract adversarial representations once per attack
        adv_feats = {
            'penultimate': extract_penultimate_features(model, adv_data),
            'all_layer': extract_all_layer_features(model, adv_data),
        }
        if km_feats is not None and len(km_feats) > 0:
            adv_feats['knowledge_matrix'] = km_feats

        # A4: Verify sample alignment across representations
        rep_counts = {rn: len(af) for rn, af in adv_feats.items()}
        if len(set(rep_counts.values())) > 1:
            min_count = min(rep_counts.values())
            print(f"    WARNING: Sample count mismatch for {attack}: {rep_counts}. "
                  f"Truncating all to {min_count}.")
            adv_feats = {rn: af[:min_count] for rn, af in adv_feats.items()}
        print(f"    Aligned counts for {attack}: {rep_counts}")

        # results[attack][det_name][rep_name] = metrics dict
        attack_results = {}
        for det_name in DETECTORS:
            attack_results[det_name] = {}
            for rn in rep_names:
                if rn not in adv_feats or rn not in clean_scores[det_name]:
                    continue
                adv_scores = fitted_detectors[det_name][rn].score(adv_feats[rn])
                metrics = compute_detection_metrics(
                    clean_scores[det_name][rn], adv_scores)
                attack_results[det_name][rn] = metrics

        results[attack] = attack_results
        # Free per-attack memory immediately to avoid accumulation
        del adv_data, adv_feats
        if km_feats is not None:
            del km_feats
        gc.collect()

    # Free per-attack memory before SVD ablation
    gc.collect()

    # -----------------------------------------------------------------------
    # 7. SVD rank ablation (optional)
    # -----------------------------------------------------------------------
    svd_ablation_results = None
    if svd_ablation:
        print("\n  Running SVD rank ablation (Mahalanobis)...", flush=True)
        svd_ranks = [16, 32, 64, 128, 256, 512]
        svd_ablation_results = {}

        # Phase A: Extract and cache adversarial features ONCE for all ranks.
        # Features are invariant across SVD ranks — only the detector changes.
        # Memory cost: ~16 attacks × 3 reps × 800 × 256 × 8 bytes ≈ 100 MB.
        cached_adv_feats = {}  # (rep_name, attack) -> numpy array
        print("    Caching adversarial features for SVD ablation...", flush=True)
        for attack in available_attacks:
            adv_path = Path(base) / 'adversarial_examples' / attack / 'adversarial_examples.pth'
            if not adv_path.exists():
                continue
            adv_data_svd = torch.load(adv_path, map_location='cpu', weights_only=True)
            if len(adv_data_svd) > 2000:
                adv_data_svd = adv_data_svd[:2000]
            # Load KM features and use for alignment (once per attack)
            km_feats_svd = None
            if 'knowledge_matrix' in rep_names:
                km_feats_svd = load_matrices_as_features(base, attack, max_samples=2000)
                if km_feats_svd is not None and len(km_feats_svd) > 0:
                    adv_data_svd = adv_data_svd[:len(km_feats_svd)]
                    cached_adv_feats[('knowledge_matrix', attack)] = km_feats_svd
            # Extract penultimate and all-layer features once
            if 'penultimate' in rep_names:
                cached_adv_feats[('penultimate', attack)] = extract_penultimate_features(model, adv_data_svd)
            if 'all_layer' in rep_names:
                cached_adv_feats[('all_layer', attack)] = extract_all_layer_features(model, adv_data_svd)
            del adv_data_svd, km_feats_svd
            gc.collect()
        print(f"    Cached features for {len(available_attacks)} attacks", flush=True)

        # Phase B: Sweep SVD ranks using cached features (CPU-only, fast).
        for rank in svd_ranks:
            svd_ablation_results[rank] = {}
            for rn in rep_names:
                feats, labs = train_reps[rn]
                if feats.shape[1] < rank:
                    continue
                print(f"    rank={rank}, rep={rn}...", flush=True)
                det = MahalanobisDetector(max_components=rank)
                det.fit(feats, labs, num_classes)
                clean_sc = det.score(test_feats[rn]) if rn in test_feats else None
                if clean_sc is None:
                    continue
                aurocs = []
                for attack in available_attacks:
                    af = cached_adv_feats.get((rn, attack))
                    if af is None or len(af) == 0:
                        continue
                    adv_sc = det.score(af)
                    m = compute_detection_metrics(clean_sc, adv_sc)
                    aurocs.append(m['auroc'])
                svd_ablation_results[rank][rn] = {
                    'mean_auroc': float(np.mean(aurocs)) if aurocs else None,
                    'n_attacks': len(aurocs),
                }
        del cached_adv_feats
        gc.collect()

    # -----------------------------------------------------------------------
    # 7b. Lee et al. (2018) multi-layer Mahalanobis baseline
    # -----------------------------------------------------------------------
    lee2018_results = {}
    print("\n  Running Lee et al. (2018) multi-layer Mahalanobis baseline...", flush=True)
    try:
        # H3: Split test_data into val (LR training) and eval (final metrics)
        n_test = len(test_data)
        n_val = n_test // 2
        lee_val_data = test_data[:n_val]
        lee_eval_data = test_data[n_val:]

        with MultiLayerMahalanobisDetector(
            model, num_classes, device=device, max_components=256,
            batch_size=64, epsilon=0.005  # H4: input preprocessing
        ) as lee_det:
            lee_det.fit(train_data, train_labels_np)

            # Per-layer scores for val clean data (used for LR training)
            val_layer_scores = lee_det.score_per_layer(lee_val_data)

            # H2: Per-attack logistic regression
            for attack in available_attacks:
                adv_path = Path(base) / 'adversarial_examples' / attack / 'adversarial_examples.pth'
                if not adv_path.exists():
                    continue
                adv_data = torch.load(adv_path, map_location='cpu', weights_only=True)
                if len(adv_data) > 2000:
                    adv_data = adv_data[:2000]

                # M-8: Skip attacks with too few adversarial examples for LR split
                if len(adv_data) < 4:
                    print(f"    Skipping {attack} for Lee: only {len(adv_data)} adv examples (need >= 4)")
                    continue

                # Split adversarial data: first half for LR training, second for eval
                n_adv = len(adv_data)
                n_adv_val = n_adv // 2
                adv_val = adv_data[:n_adv_val]
                adv_eval = adv_data[n_adv_val:]

                # Fit logistic regression per-attack on val split
                adv_val_layer_scores = lee_det.score_per_layer(adv_val)
                lee_det.fit_logistic(val_layer_scores, adv_val_layer_scores)
                print(f"    LR fitted on {attack} (val={n_adv_val})")

                # NEW-H1: Score clean eval data AFTER fitting LR for this attack,
                # so it uses the correct per-attack logistic regression model.
                eval_clean_scores = lee_det.score(lee_eval_data)

                # Evaluate on eval split
                lee_adv_scores = lee_det.score(adv_eval)
                lee2018_results[attack] = compute_detection_metrics(
                    eval_clean_scores, lee_adv_scores
                )
                del adv_data, adv_val, adv_eval
                gc.collect()

        if lee2018_results:
            lee_aurocs = [m['auroc'] for m in lee2018_results.values()]
            print(f"    Lee et al. mean AUROC: {np.mean(lee_aurocs):.4f} "
                  f"({len(lee_aurocs)} attacks)")
    except Exception as e:
        print(f"    WARNING: Lee et al. baseline failed: {e}")

    # -----------------------------------------------------------------------
    # 8. Aggregate and report
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  RESULTS: {experiment_name}")
    print(f"  6 detectors x {len(rep_names)} representations x {len(available_attacks)} attacks")
    print(f"{'='*70}")

    # Summary: average AUROC per (detector, representation)
    aggregate = {}  # aggregate[det_name][rep_name] = list of aurocs
    for det_name in DETECTORS:
        aggregate[det_name] = {rn: [] for rn in rep_names}

    for attack in available_attacks:
        if attack not in results:
            continue
        for det_name in DETECTORS:
            for rn in rep_names:
                if rn in results[attack].get(det_name, {}):
                    aggregate[det_name][rn].append(
                        results[attack][det_name][rn]['auroc'])

    # Print grid: rows = detectors, columns = representations
    short_names = {'penultimate': 'Penult.', 'all_layer': 'AllLayer',
                   'knowledge_matrix': 'KnowMat'}
    header = f"  {'Detector':<18s}"
    for rn in rep_names:
        header += f" | {short_names[rn]:>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for det_name in DETECTORS:
        line = f"  {det_name:<18s}"
        for rn in rep_names:
            vals = aggregate[det_name][rn]
            if vals:
                line += f" | {np.mean(vals):>8.4f}"
            else:
                line += f" | {'---':>8s}"
        print(line)

    # Category breakdown (best detector per representation)
    print(f"\n  BY CATEGORY (best detector per representation):")
    for cat_name, cat_attacks in ATTACK_CATEGORIES.items():
        cat_line = f"    {cat_name:<16s}"
        for rn in rep_names:
            best_auroc = -1
            for det_name in DETECTORS:
                aurocs = [results[a][det_name][rn]['auroc']
                         for a in cat_attacks
                         if a in results and rn in results[a].get(det_name, {})]
                if aurocs and np.mean(aurocs) > best_auroc:
                    best_auroc = np.mean(aurocs)
            if best_auroc >= 0:
                cat_line += f" | {best_auroc:>8.4f}"
            else:
                cat_line += f" | {'---':>8s}"
        print(cat_line)

    # Computational cost summary
    print(f"\n  COMPUTATIONAL COST:")
    for rn in rep_names:
        if rn in computational_cost:
            cc = computational_cost[rn]
            time_key = 'loading_seconds_per_1000' if 'loading_seconds_per_1000' in cc else 'seconds_per_1000'
            label = 'load' if 'loading_seconds_per_1000' in cc else 'extract'
            print(f"    {short_names[rn]:<12s}: {cc[time_key]:.1f} s/1000 ({label}), "
                  f"{cc['peak_gpu_memory_gb']:.2f} GB peak, dim={cc['feature_dim']}")

    # SVD ablation summary
    if svd_ablation_results:
        print(f"\n  SVD RANK ABLATION (Mahalanobis mean AUROC):")
        for rank in sorted(svd_ablation_results.keys()):
            line = f"    rank={rank:<4d}"
            for rn in rep_names:
                if rn in svd_ablation_results[rank]:
                    val = svd_ablation_results[rank][rn]['mean_auroc']
                    line += f" | {short_names[rn]}={val:.4f}" if val else f" | {short_names[rn]}=---"
                else:
                    line += f" | {short_names[rn]}=---"
            print(line)

    # Save results
    out_dir = Path(f'experiments/{experiment_name}/comparison/')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'representation_comparison.json'

    # Compute average AUROC summary
    avg_auroc = {}
    for det_name in DETECTORS:
        avg_auroc[det_name] = {}
        for rn in rep_names:
            vals = aggregate[det_name][rn]
            avg_auroc[det_name][rn] = float(np.mean(vals)) if vals else None

    save_data = {
        'experiment': experiment_name,
        'dataset': dataset,
        'architecture_index': arch_idx,
        'representations': rep_names,
        'detectors': list(DETECTORS.keys()),
        'per_attack': results,
        'average_auroc': avg_auroc,
        'computational_cost': computational_cost,
    }
    if lee2018_results:
        save_data['lee2018_baseline'] = lee2018_results
        lee_aurocs = [m['auroc'] for m in lee2018_results.values()]
        save_data['lee2018_mean_auroc'] = float(np.mean(lee_aurocs))
    if svd_ablation_results:
        # Convert int keys to strings for JSON
        save_data['svd_ablation'] = {
            str(k): v for k, v in svd_ablation_results.items()
        }

    atomic_json_dump(save_data, out_file)
    print(f"\n  Results saved to {out_file}")

    return save_data


def print_cross_experiment_summary(all_results):
    """Print comparison across all experiments."""
    if not all_results:
        return

    print(f"\n\n{'#'*70}")
    print(f"  CROSS-EXPERIMENT SUMMARY")
    print(f"{'#'*70}")

    all_reps = set()
    all_dets = set()
    for r in all_results:
        all_reps.update(r['representations'])
        all_dets.update(r.get('detectors', ['Mahalanobis']))
    all_reps = sorted(all_reps)
    all_dets = sorted(all_dets)

    short_names = {'penultimate': 'Penult.', 'all_layer': 'AllLayer',
                   'knowledge_matrix': 'KnowMat'}

    # Per-detector summary
    for det_name in all_dets:
        print(f"\n  --- {det_name} ---")
        header = f"  {'Experiment':<20s}"
        for rn in all_reps:
            header += f" | {short_names[rn]:>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for r in all_results:
            line = f"  {r['experiment']:<20s}"
            avg = r['average_auroc']
            for rn in all_reps:
                val = avg.get(det_name, {}).get(rn) if isinstance(avg.get(det_name), dict) else None
                if val is not None:
                    line += f" | {val:>8.4f}"
                else:
                    line += f" | {'---':>8s}"
            print(line)

    # Win counts (best representation across all detectors and attacks)
    print(f"\n  WINS (highest AUROC per attack, best detector):")
    wins = {rn: 0 for rn in all_reps}
    total = 0
    for r in all_results:
        for attack, attack_res in r['per_attack'].items():
            best_auroc = -1
            best_rep = None
            for det_name in all_dets:
                det_res = attack_res.get(det_name, attack_res)
                for rn in all_reps:
                    auroc_val = None
                    if isinstance(det_res.get(rn), dict):
                        auroc_val = det_res[rn].get('auroc')
                    if auroc_val is not None and auroc_val > best_auroc:
                        best_auroc = auroc_val
                        best_rep = rn
            if best_rep:
                wins[best_rep] += 1
                total += 1

    if total > 0:
        for rn in all_reps:
            long_name = {'penultimate': 'Penultimate Features',
                         'all_layer': 'All-Layer Features',
                         'knowledge_matrix': 'Knowledge Matrices'}[rn]
            print(f"    {long_name:<25s}: {wins[rn]}/{total} ({100*wins[rn]/total:.1f}%)")


def parse_args():
    parser = ArgumentParser(description="Fair comparison of adversarial detection representations")
    parser.add_argument("--experiment", nargs="+",
                        default=["lenet_cifar10"],
                        help="Experiment name(s)")
    parser.add_argument("--temp_dir", type=str, default=None,
                        help="Temporary directory for cluster")
    parser.add_argument("--svd_ablation", action="store_true",
                        help="Run SVD rank ablation for Mahalanobis detector")
    return parser.parse_args()


def main():
    args = parse_args()
    all_results = []
    for exp in args.experiment:
        result = run_comparison(exp, args.temp_dir,
                                svd_ablation=args.svd_ablation)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        print_cross_experiment_summary(all_results)


if __name__ == "__main__":
    main()
