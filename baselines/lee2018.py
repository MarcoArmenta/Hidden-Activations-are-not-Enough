"""
Lee et al. (2018) "A Simple Unified Framework for Detecting
Out-of-Distribution Examples and Adversarial Examples" (NeurIPS 2018).

Multi-layer Mahalanobis distance detector -- the standard baseline
for adversarial/OOD detection papers.

Key differences from our single-representation Mahalanobis:
  - Uses features from ALL layers (with learned per-layer weights)
  - Adds input preprocessing (small adversarial perturbation to enhance features)
  - Trains a logistic regression on per-layer Mahalanobis scores
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import TruncatedSVD


class MultiLayerMahalanobisDetector:
    """Multi-layer Mahalanobis distance detector (Lee et al., 2018).

    Extracts intermediate features from every activation layer in a model,
    computes per-class Mahalanobis distances at each layer, then combines
    the per-layer scores via logistic regression.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classifier (must be in eval mode).
    num_classes : int
        Number of output classes.
    device : str or torch.device
        Device for inference (default ``'cpu'``).
    max_components : int
        Maximum feature dimension per layer; layers with higher
        dimensionality are reduced via TruncatedSVD (default 256).
    batch_size : int
        Batch size used during feature extraction to avoid OOM
        (default 64).
    """

    def __init__(self, model, num_classes, device='cpu',
                 max_components=256, batch_size=64, epsilon=0.0):
        self.model = model
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.max_components = max_components
        self.batch_size = batch_size
        self.epsilon = epsilon

        # Per-layer statistics (populated by ``fit``)
        self.class_means = {}   # layer_name -> {class_idx -> mean_vector}
        self.precision = {}     # layer_name -> precision matrix (tied)
        self.svd = {}           # layer_name -> TruncatedSVD or None
        self.layer_names = []   # ordered list of hooked layer names

        # M-2: Cached torch tensors for preprocessing (populated by ``fit``)
        self._prec_tensors = {}       # layer_name -> torch.Tensor
        self._mean_tensors = {}       # layer_name -> {class_idx -> torch.Tensor}
        self._components_tensors = {} # layer_name -> torch.Tensor or None

        # Logistic regression combiner (populated by ``fit_logistic``)
        self.logreg = None

        # Forward-hook bookkeeping
        self._features = {}
        self._hooks = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self):
        """Attach forward hooks to all activation layers in the model."""
        activation_types = (
            nn.ReLU, nn.ELU, nn.Tanh,
            nn.LeakyReLU, nn.Sigmoid, nn.GELU,
        )
        for name, module in self.model.named_modules():
            if isinstance(module, activation_types):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, inp, output):
            self._features[name] = output.detach().cpu()
        return hook_fn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False

    def cleanup(self):
        """Remove all forward hooks from the model."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, x, preprocess=False):
        """Run a forward pass and return per-layer flattened features.

        Parameters
        ----------
        x : torch.Tensor
            Input batch ``(N, C, H, W)``.
        preprocess : bool
            If True and epsilon > 0, apply input preprocessing (Lee et al. 2018):
            perturb input in the direction that reduces Mahalanobis distance.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from layer name to array of shape ``(N, D_layer)``.
        """
        if preprocess and self.epsilon > 0 and self.precision:
            return self._extract_features_with_preprocessing(x)

        self._features = {}
        with torch.no_grad():
            _ = self.model(x.to(self.device).float())
        result = {}
        for name in sorted(self._features.keys()):
            feat = self._features[name]
            result[name] = feat.reshape(feat.shape[0], -1).numpy()
        return result

    def _extract_features_with_preprocessing(self, x):
        """Extract features after input preprocessing (Lee et al. 2018).

        1. Forward pass to get layer features (with gradients)
        2. Compute Mahalanobis distance loss (sum over layers of min-over-classes)
           entirely in PyTorch so gradients flow back to the input
        3. Backprop to get input gradient
        4. Perturb: x_preprocessed = x - epsilon * sign(grad)
        5. Re-extract features from preprocessed input
        """
        # NC1-FIX: .detach() ensures x_input is a leaf tensor so .grad
        # is populated after backward(). Without detach, .to()/.float() may
        # return a non-leaf if x is already on-device and float, causing
        # the preprocessing perturbation to be a no-op.
        x_input = x.to(self.device).float().detach().requires_grad_(True)

        # Forward pass with gradient-enabled hooks
        self._features = {}
        # Temporarily replace hooks with non-detaching versions
        for h in self._hooks:
            h.remove()
        grad_hooks = []
        try:
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.ReLU, nn.ELU, nn.Tanh,
                                       nn.LeakyReLU, nn.Sigmoid, nn.GELU)):
                    def make_grad_hook(n):
                        def hook_fn(module, inp, output):
                            self._features[n] = output
                        return hook_fn
                    grad_hooks.append(module.register_forward_hook(make_grad_hook(name)))

            _ = self.model(x_input)

            # Compute Mahalanobis loss entirely in PyTorch to preserve gradients.
            # Pre-convert numpy statistics to torch tensors.
            # NC1-FIX: Use plain zero tensor. The reassignment
            # loss = loss + ... creates a new non-leaf connected to the
            # computation graph. The old requires_grad=True leaf was unused.
            loss = torch.zeros(1, device=self.device)
            for layer_name in self.layer_names:
                if layer_name not in self._features or layer_name not in self.precision:
                    continue
                feat = self._features[layer_name].reshape(x_input.shape[0], -1)

                # SVD projection: apply in numpy (linear transform) then wrap back
                components_t = self._components_tensors.get(layer_name)
                if components_t is not None:
                    proj = feat.float() @ components_t
                else:
                    proj = feat.float()

                prec_t = self._prec_tensors[layer_name]

                # Min Mahalanobis distance over classes (in torch)
                min_dists_sq = None
                for c in range(self.num_classes):
                    if c not in self._mean_tensors.get(layer_name, {}):
                        continue
                    mean_t = self._mean_tensors[layer_name][c]
                    diff = proj - mean_t.unsqueeze(0)
                    # dists_sq_c[i] = diff[i] @ prec @ diff[i]
                    dists_sq_c = torch.einsum('ij,jk,ik->i', diff, prec_t, diff)
                    dists_sq_c = torch.clamp(dists_sq_c, min=0.0)
                    if min_dists_sq is None:
                        min_dists_sq = dists_sq_c
                    else:
                        min_dists_sq = torch.minimum(min_dists_sq, dists_sq_c)

                if min_dists_sq is not None:
                    loss = loss + min_dists_sq.sum()

            # Backprop to input
            if x_input.grad is not None:
                x_input.grad.zero_()
            loss.backward()

            # NC1 diagnostic: verify gradient actually flowed to input
            if x_input.grad is None:
                import warnings
                warnings.warn(
                    f"Lee preprocessing: no gradient on x_input "
                    f"(loss={loss.item():.6f}). Preprocessing is a no-op.")
            elif x_input.grad.abs().max().item() == 0:
                import warnings
                warnings.warn(
                    f"Lee preprocessing: gradient is all zeros "
                    f"(loss={loss.item():.6f}). Preprocessing has no effect.")

        finally:
            # Remove gradient hooks, restore detaching hooks (H5: leak-safe)
            for h in grad_hooks:
                h.remove()
            self._hooks = []
            self._register_hooks()

        # Perturb input
        if x_input.grad is not None:
            x_preprocessed = x_input.detach() - self.epsilon * x_input.grad.sign()
        else:
            x_preprocessed = x_input.detach()

        # Re-extract features with standard (detached) hooks
        self._features = {}
        with torch.no_grad():
            _ = self.model(x_preprocessed)
        result = {}
        for name in sorted(self._features.keys()):
            feat = self._features[name]
            result[name] = feat.reshape(feat.shape[0], -1).numpy()
        return result

    def _extract_features_batched(self, data, preprocess=False):
        """Extract features in batches and concatenate across samples.

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            Full dataset ``(N, C, H, W)``.
        preprocess : bool
            If True, apply input preprocessing before feature extraction.

        Returns
        -------
        dict[str, np.ndarray]
            Per-layer arrays of shape ``(N, D_layer)``.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        all_features = {}  # layer_name -> list of arrays
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_features = self._extract_features(batch, preprocess=preprocess)
            for name, feat in batch_features.items():
                all_features.setdefault(name, []).append(feat)

        return {
            name: np.vstack(parts) for name, parts in all_features.items()
        }

    # ------------------------------------------------------------------
    # Fitting (training phase)
    # ------------------------------------------------------------------

    def fit(self, train_loader_or_data, labels):
        """Compute per-layer, per-class Mahalanobis statistics.

        Parameters
        ----------
        train_loader_or_data : torch.Tensor, np.ndarray, or DataLoader
            Training inputs.  If a ``DataLoader`` is passed the *labels*
            argument is ignored and labels are read from the loader.
        labels : np.ndarray or None
            Integer class labels aligned with the data tensor.
        """
        self.model.eval()

        # ----- Extract features from training data -----
        if isinstance(train_loader_or_data, torch.utils.data.DataLoader):
            all_features = {}
            all_labels = []
            for batch_x, batch_y in train_loader_or_data:
                feats = self._extract_features(batch_x)
                for name, feat in feats.items():
                    all_features.setdefault(name, []).append(feat)
                all_labels.append(batch_y.numpy())
            features_dict = {
                n: np.vstack(parts) for n, parts in all_features.items()
            }
            labels = np.concatenate(all_labels).astype(int)
        else:
            data = train_loader_or_data
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            features_dict = self._extract_features_batched(data)
            labels = np.asarray(labels, dtype=int)

        self.layer_names = sorted(features_dict.keys())

        # ----- Per-layer statistics -----
        for layer_name in self.layer_names:
            raw = features_dict[layer_name]         # (N, D)
            N, D = raw.shape

            # Dimensionality reduction if needed
            n_comp = min(self.max_components, D, max(1, N - 1))
            if D > n_comp:
                svd = TruncatedSVD(n_components=n_comp, random_state=0)
                proj = svd.fit_transform(raw)
                self.svd[layer_name] = svd
            else:
                proj = raw
                self.svd[layer_name] = None

            # Per-class means and covariances
            self.class_means[layer_name] = {}
            global_mean = np.mean(proj, axis=0)
            per_class_covs = []
            for c in range(self.num_classes):
                mask = (labels == c)
                class_data = proj[mask]
                if class_data.shape[0] >= 2:
                    self.class_means[layer_name][c] = np.mean(class_data, axis=0)
                    try:
                        lw = LedoitWolf().fit(class_data)
                        per_class_covs.append(lw.covariance_)
                    except Exception:
                        per_class_covs.append(np.eye(proj.shape[1]))
                elif class_data.shape[0] == 1:
                    self.class_means[layer_name][c] = class_data[0]
                else:
                    self.class_means[layer_name][c] = global_mean

            # Tied covariance: sample-weighted pooled covariance (Lee et al. 2018)
            if per_class_covs:
                class_counts = []
                for c in range(self.num_classes):
                    mask = (labels == c)
                    class_counts.append(max(mask.sum() - 1, 0))
                total_weight = sum(class_counts)
                if total_weight > 0:
                    tied_cov = sum(w * cov for w, cov in zip(class_counts, per_class_covs)) / total_weight
                else:
                    tied_cov = np.mean(per_class_covs, axis=0)
            else:
                tied_cov = np.eye(proj.shape[1])

            try:
                precision = np.linalg.inv(tied_cov)
                if np.any(np.isnan(precision)):
                    precision = np.linalg.inv(tied_cov + 1e-6 * np.eye(tied_cov.shape[0]))
            except np.linalg.LinAlgError:
                precision = np.linalg.inv(tied_cov + 1e-6 * np.eye(tied_cov.shape[0]))

            self.precision[layer_name] = precision

            # M-2: Cache torch tensors for preprocessing
            self._prec_tensors[layer_name] = torch.tensor(
                precision, device=self.device, dtype=torch.float32)
            self._mean_tensors[layer_name] = {}
            for c in self.class_means[layer_name]:
                self._mean_tensors[layer_name][c] = torch.tensor(
                    self.class_means[layer_name][c],
                    device=self.device, dtype=torch.float32)
            svd_obj = self.svd[layer_name]
            if svd_obj is not None:
                self._components_tensors[layer_name] = torch.tensor(
                    svd_obj.components_.T, device=self.device, dtype=torch.float32)
            else:
                self._components_tensors[layer_name] = None

    # ------------------------------------------------------------------
    # Per-layer Mahalanobis scores
    # ------------------------------------------------------------------

    def _mahalanobis_scores_per_layer(self, features_dict):
        """Compute min-over-classes Mahalanobis distance at each layer.

        Parameters
        ----------
        features_dict : dict[str, np.ndarray]
            Per-layer features for N samples.

        Returns
        -------
        np.ndarray
            Shape ``(N, num_layers)`` -- one score per layer per sample.
        """
        N = None
        per_layer_scores = []

        for layer_name in self.layer_names:
            raw = features_dict[layer_name]
            if N is None:
                N = raw.shape[0]

            # Project if SVD was fitted
            svd = self.svd[layer_name]
            if svd is not None:
                proj = svd.transform(raw.reshape(raw.shape[0], -1))
            else:
                proj = raw.reshape(raw.shape[0], -1)

            prec = self.precision[layer_name]

            # Min Mahalanobis distance over classes
            min_dists = np.full(proj.shape[0], np.inf)
            for c in range(self.num_classes):
                if c not in self.class_means[layer_name]:
                    continue
                diff = proj - self.class_means[layer_name][c]
                dists_sq = np.einsum('ij,jk,ik->i', diff, prec, diff)
                dists = np.sqrt(np.maximum(dists_sq, 0.0))
                min_dists = np.minimum(min_dists, dists)

            per_layer_scores.append(min_dists)

        if not per_layer_scores:
            return np.zeros((N or 0, 0))
        return np.column_stack(per_layer_scores)  # (N, num_layers)

    # ------------------------------------------------------------------
    # Logistic regression combiner
    # ------------------------------------------------------------------

    def fit_logistic(self, clean_scores, adv_scores):
        """Train logistic regression to combine per-layer scores.

        Parameters
        ----------
        clean_scores : np.ndarray
            Per-layer scores for clean examples, shape ``(N_clean, L)``.
        adv_scores : np.ndarray
            Per-layer scores for adversarial examples, shape ``(N_adv, L)``.
        """
        X = np.vstack([clean_scores, adv_scores])
        y = np.concatenate([
            np.zeros(len(clean_scores)),
            np.ones(len(adv_scores)),
        ])
        self.logreg = LogisticRegressionCV(
            cv=3, max_iter=1000, random_state=0,
        )
        self.logreg.fit(X, y)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, data):
        """Compute anomaly scores for a batch of inputs.

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            Input data ``(N, C, H, W)``.

        Returns
        -------
        np.ndarray
            1-D array of length N (higher = more anomalous).
        """
        self.model.eval()

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        features_dict = self._extract_features_batched(data, preprocess=(self.epsilon > 0))
        layer_scores = self._mahalanobis_scores_per_layer(features_dict)

        if self.logreg is not None:
            # Logistic regression probability of being adversarial
            proba = self.logreg.predict_proba(layer_scores)
            if proba.shape[1] >= 2:
                return proba[:, 1]
            # Degenerate case: only one class seen during LR training
            return proba[:, 0]
        else:
            # Fallback: simple average across layers
            return layer_scores.mean(axis=1)

    def score_per_layer(self, data):
        """Return raw per-layer scores (useful for ``fit_logistic``).

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            Input data ``(N, C, H, W)``.

        Returns
        -------
        np.ndarray
            Shape ``(N, num_layers)``.
        """
        self.model.eval()

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        features_dict = self._extract_features_batched(data, preprocess=(self.epsilon > 0))
        return self._mahalanobis_scores_per_layer(features_dict)
