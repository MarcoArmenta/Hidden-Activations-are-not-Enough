import os
import json
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import TruncatedSVD
from sklearn.covariance import LedoitWolf
from argparse import Namespace
from pathlib import Path
from typing import Union
import multiprocessing as mp
from joblib import dump, load, Parallel, delayed

from utils.utils import get_ellipsoid_data, zero_std, get_model, get_dataset, subset, get_num_classes, get_input_shape, get_device, get_parameters_baseline
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS

# Optional: force-cache rebuild by setting env var REBUILD_CACHE=1
FORCE_REBUILD_CACHE = os.environ.get('REBUILD_CACHE', '0') == '1'


def _ensure_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _vectorized_min_mahalanobis_with_prec(feats: np.ndarray, class_means: dict, class_precisions: dict) -> np.ndarray:
    """
    feats: (N, D) numpy array in the SAME projection space used to fit class_precisions
    class_means: dict{k: 1D numpy array (D,)}
    class_precisions: dict{k: 2D numpy array (D, D)} (precision matrices)
    returns: (N,) min Mahalanobis distance to any class (in projected space)
    """
    feats = np.asarray(feats).reshape(feats.shape[0], -1)
    N, D = feats.shape
    min_dists = np.full((N,), np.inf, dtype=float)

    # Vectorized loop across classes using einsum
    for k in class_means.keys():
        mean_k = np.asarray(class_means[k]).ravel()
        diff = feats - mean_k  # (N, D)
        prec = np.asarray(class_precisions[k])
        # squared distances = (diff @ prec * diff).sum(axis=1) -> einsum
        dists_sq = np.einsum('ij,jk,ik->i', diff, prec, diff)
        min_dists = np.minimum(min_dists, np.sqrt(np.maximum(dists_sq, 0.0)))

    return min_dists


def _detect_n_cpus(num_classes: int) -> int:
    """Detect available CPU count for parallel ops; cap at num_classes (no need for more jobs)."""
    n = None
    for var in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_CPUS_PER_NODE', 'CPUS', 'NUM_CPUS'):
        v = os.environ.get(var)
        if v and v.isdigit():
            n = int(v)
            break
    if n is None:
        n = os.cpu_count() or 1
    n = max(1, int(n))
    return min(n, num_classes)


def reject_predicted_attacks(
        experiment_name: str,
        weights_path: str,
        architecture_index: int,
        input_shape,
        num_classes: int,
        ellipsoids: dict,
        t_epsilon: float = 2,
        epsilon: float = 0.1,
        epsilon_p: float = 0.1,
        verbose: bool = True,
        temp_dir: Union[str, None] = None
    ) -> str:
    """
    Goes over the dataset and predicts if it is an adversarial example or not.
    This function expects adversarial_matrices present (one per example).
    It preloads matrices per-attack to avoid repeated IO and excessive .to(device) calls.
    """
    if temp_dir is not None:
        reject_path = f'{temp_dir}/experiments/{experiment_name}/rejection_levels/reject_at_{t_epsilon}_{epsilon}.json'
        Path(f'{temp_dir}/experiments/{experiment_name}/rejection_levels/').mkdir(parents=True, exist_ok=True)
    else:
        reject_path = f'experiments/{experiment_name}/rejection_levels/reject_at_{t_epsilon}_{epsilon}.json'
        Path(f'experiments/{experiment_name}/rejection_levels/').mkdir(parents=True, exist_ok=True)

    device = get_device()

    model = get_model(
        path=Path(weights_path),
        architecture_index=architecture_index,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device
    )

    if os.path.exists(reject_path):
        print("Loading rejection level...", flush=True)
        with open(reject_path, 'r') as file:
            reject_at = json.load(file)[0]
    else:
        print(f"File does not exists: {reject_path}", flush=True)
        return

    if reject_at <= 0:
        print(f"Rejection level too small: {reject_at}", flush=True)
        return

    print(f"Will reject when 'zero dims' < {reject_at}.", flush=True)
    adv_succes = {attack: [] for attack in ["test"] + ATTACKS}  # Save adversarial examples that were not detected
    results = []  # (Rejected, Was attacked)
    counts = {
        key: {
            'not_rejected_and_attacked': 0,
            'not_rejected_and_not_attacked': 0,
            'rejected_and_attacked': 0,
            'rejected_and_not_attacked': 0
        } for key in ["test"] + ATTACKS
    }

    if temp_dir is not None:
        path_adv_matrices = Path(f'{temp_dir}/experiments/{experiment_name}/adversarial_matrices/')
        test_labels = torch.load(f'{temp_dir}/experiments/{experiment_name}/adversarial_examples/test/labels.pth').to(device)
    else:
        path_adv_matrices = Path(f'experiments/{experiment_name}/adversarial_matrices/')
        test_labels = torch.load(f'experiments/{experiment_name}/adversarial_examples/test/labels.pth').to(device)

    test_acc = 0
    counts_file = Path(f'experiments/{experiment_name}/counts_per_attack/counts_per_attack_{t_epsilon}_{epsilon}_{epsilon_p}.json')
    Path(f'experiments/{experiment_name}/counts_per_attack/').mkdir(parents=True, exist_ok=True)

    for a in ["test"] + ATTACKS:
        print(f"Trying for {a}", flush=True)

        if counts_file.exists():
            with open(counts_file, 'r') as file:
                counts_current = json.load(file)
            if all(value == 0 for value in counts_current[a].values()):
                counts = counts_current
            else:
                print(f"Attack {a} found and loaded.", flush=True)
                continue

        # Load attacked dataset (images) — keep on device for model inference
        try:
            if temp_dir is not None:
                attacked_dataset = torch.load(f'{temp_dir}/experiments/{experiment_name}/adversarial_examples/{a}/adversarial_examples.pth').to(device)
            else:
                attacked_dataset = torch.load(f'experiments/{experiment_name}/adversarial_examples/{a}/adversarial_examples.pth').to(device)
        except Exception:
            print(f"Attack {a} not found.", flush=True)
            continue

        # Preload matrices for attack 'a' (load on CPU once, then move to device in bulk)
        matrices_cache = []
        i = 0
        while True:
            current_matrix_path = path_adv_matrices / Path(f"{a}/{i}/matrix.pth")
            if not os.path.exists(current_matrix_path):
                if i == 0:
                    print(f"Attack {a} not found (no matrices).", flush=True)
                    matrices_cache = []
                break
            try:
                mat_cpu = torch.load(current_matrix_path, map_location='cpu')
            except Exception as e:
                print(f"Error loading {current_matrix_path}: {e}", flush=True)
                break
            matrices_cache.append(mat_cpu)
            i += 1

        if len(matrices_cache) == 0:
            # no matrices -> skip this attack
            continue

        # Move matrices to device once if needed by zero_std
        matrices_cache = [m.to(device) for m in matrices_cache]

        not_rejected_and_attacked = 0
        not_rejected_and_not_attacked = 0
        rejected_and_attacked = 0
        rejected_and_not_attacked = 0

        # iterate over available attacked examples (use min length to avoid index error)
        n_examples = min(len(attacked_dataset), len(matrices_cache))
        for i in range(n_examples):
            im = attacked_dataset[i]  # already on device
            # batch model inference avoided: single sample here is fine for small counts; consider batching if large
            logits = model.forward(im.unsqueeze(0).float())
            pred = torch.argmax(logits, dim=1).item()

            mat = matrices_cache[i]
            b = get_ellipsoid_data(ellipsoids, torch.tensor(pred), "std")
            c = zero_std(mat, b, epsilon_p).item()

            res = ((reject_at > c), (a != "test"))

            if not res[0] and a != "test":
                not_rejected_and_attacked += 1
                counts[a]['not_rejected_and_attacked'] += 1
                if len(adv_succes[a]) < 10:
                    adv_succes[a].append(im)

            if res[0] and a != 'test':
                rejected_and_attacked += 1
                counts[a]['rejected_and_attacked'] += 1

            if res[0] and a == "test":
                rejected_and_not_attacked += 1
                counts[a]['rejected_and_not_attacked'] += 1

            if not res[0] and a == "test":
                not_rejected_and_not_attacked += 1
                counts[a]['not_rejected_and_not_attacked'] += 1
                if pred == test_labels[i]:
                    test_acc += 1

            results.append(res)

        if verbose:
            print("Attack method: ", a, flush=True)
            if a == 'test':
                print(f'Wrongly rejected test data : {rejected_and_not_attacked} out of {n_examples}', flush=True)
                print(f'Trusted test data : {not_rejected_and_not_attacked} out of {n_examples}', flush=True)
                if counts['test']['not_rejected_and_not_attacked'] == 0:
                    test_acc = 0
                else:
                    test_acc = test_acc / counts['test']['not_rejected_and_not_attacked']
                print("Accuracy on test data that was not rejected: ", test_acc, flush=True)
            else:
                print(f'Detected adversarial examples : {rejected_and_attacked} out of {n_examples}', flush=True)
                print(f'Successful adversarial examples : {not_rejected_and_attacked} out of {n_examples}', flush=True)

        with open(counts_file, 'w') as json_file:
            json.dump(counts, json_file, indent=4)

        test_accuracy = f'experiments/{experiment_name}/counts_per_attack/test_accuracy_{t_epsilon}_{epsilon}_{epsilon_p}.json'
        with open(test_accuracy, 'w') as json_file:
            json.dump([test_acc], json_file, indent=4)

    good_defence = 0
    wrongly_rejected = 0
    num_att = 0
    for rej, att in results:
        if att:
            good_defence += int(rej)
            num_att += 1
        else:
            wrongly_rejected += int(rej)

    percentage_good = good_defence / num_att if num_att != 0 else 0
    print(f"Percentage of good defences: {percentage_good}", flush=True)

    percentage_bad = wrongly_rejected / (len(results) - num_att) if len(results) != num_att else 0
    print(f"Percentage of wrong rejections: {percentage_bad}", flush=True)

    return f"Percentage of good defences: {percentage_good}\nPercentage of wrong rejections: {percentage_bad}"


def get_features(data: torch.Tensor, model, batch_size: int = 256) -> np.ndarray:
    """
    Extracts features from the data using the model's penultimate layer (batched).
    Returns a numpy array (N, D).
    """
    model.eval()
    device = get_device()
    features_list = []
    with torch.no_grad():
        if isinstance(data, torch.Tensor):
            total = len(data)
            for i in range(0, total, batch_size):
                batch = data[i: i + batch_size].to(device).float()
                feats = model.forward(batch, return_penultimate=True)
                feats_np = feats.detach().cpu().numpy().reshape(feats.shape[0], -1)
                features_list.append(feats_np)
        else:
            # fallback (e.g., list of tensors)
            for i in range(0, len(data), batch_size):
                batch_items = data[i:i + batch_size]
                batch = torch.stack(batch_items).to(device).float()
                feats = model.forward(batch, return_penultimate=True)
                feats_np = feats.detach().cpu().numpy().reshape(feats.shape[0], -1)
                features_list.append(feats_np)
    return np.vstack(features_list) if len(features_list) > 0 else np.zeros((0,))


def reject_predicted_attacks_baseline(
        experiment_name: str,
        weights_path: str,
        architecture_index: int,
        input_shape,
        num_classes: int,
        verbose: bool = True,
        temp_dir: Union[str, None] = None
    ) -> None:
    """
    Baseline detectors (operates on model features).
    Caches train features and per-attack features, caches fitted sklearn detectors per param.
    """
    model = get_model(
        path=weights_path,
        architecture_index=architecture_index,
        input_shape=input_shape,
        num_classes=num_classes,
        device=get_device()
    )

    dataset = DEFAULT_EXPERIMENTS[f'{experiment_name}']['dataset']

    # Load train and test data (train_data is dataset images)
    if temp_dir is not None:
        train_data, _ = get_dataset(dataset, data_loader=False, data_path=temp_dir)
        test_labels = torch.load(f'{temp_dir}/experiments/{experiment_name}/adversarial_examples/test/labels.pth').to(get_device())
    else:
        train_data, _ = get_dataset(dataset, data_loader=False)
        test_labels = torch.load(f'experiments/{experiment_name}/adversarial_examples/test/labels.pth').to(get_device())

    # Get features from training data (subset)
    print("Extracting features from training data...", flush=True)
    train_data, train_labels = subset(train_data, 10000, input_shape)  # returns tensors
    # Convert train_labels to numpy for indexing
    train_labels_np = _ensure_np(train_labels).astype(int)

    # Cache train features to avoid recomputing
    cache_base = f'{temp_dir}/experiments/{experiment_name}' if temp_dir is not None else f'experiments/{experiment_name}'
    cache_dir = os.path.join(cache_base, 'preprocessed')
    os.makedirs(cache_dir, exist_ok=True)
    train_features_file = os.path.join(cache_dir, f'train_features_{dataset}_n10000.npy')

    if (not FORCE_REBUILD_CACHE) and os.path.exists(train_features_file):
        print("Loading cached train features...", flush=True)
        train_features = np.load(train_features_file)
    else:
        print("Computing train features (cached)...", flush=True)
        train_features = get_features(train_data, model)
        np.save(train_features_file, train_features)

    print("Features ready!", flush=True)

    parameters = get_parameters_baseline(dataset)
    methods = list(parameters.keys())

    counts = {
        method: {
            str(parameters[method][parameter]): {
                attack: {
                    'not_rejected_and_attacked': 0,
                    'not_rejected_and_not_attacked': 0,
                    'rejected_and_attacked': 0,
                    'rejected_and_not_attacked': 0
                } for attack in ["test"] + ATTACKS
            } for parameter in range(len(parameters[method]))
        } for method in methods
    }

    test_acc = {
        method: {
            str(parameters[method][parameter]): 0
            for parameter in range(len(parameters[method]))
        } for method in methods
    }

    # compute class means/covariances using numpy arrays
    class_means = {}
    class_covariances = {}
    for class_idx in range(num_classes):
        mask = (train_labels_np == class_idx)
        class_data = train_features[mask]
        if class_data.size == 0:
            class_means[class_idx] = np.zeros(train_features.shape[1])
            class_covariances[class_idx] = np.eye(train_features.shape[1])
        else:
            class_means[class_idx] = np.mean(class_data, axis=0)
            class_covariances[class_idx] = np.cov(class_data, rowvar=False)

    # vectorized mahalanobis helper (uses above _vectorized_min_mahalanobis_with_prec)
    # for features we keep previous approach (covariances computed directly) — it'll be ok if feature dim is reasonable
    def get_min_mahalanobis_distances(features_np: np.ndarray) -> np.ndarray:
        feats = np.asarray(features_np).reshape(features_np.shape[0], -1)
        N = feats.shape[0]
        min_dists = np.full((N,), np.inf, dtype=float)
        # precompute inverse covs
        inv_covs = {}
        for k in range(num_classes):
            cov = np.asarray(class_covariances[k])
            try:
                inv_covs[k] = np.linalg.pinv(cov)
            except Exception:
                inv_covs[k] = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))
        for k in range(num_classes):
            mean_k = np.asarray(class_means[k]).ravel()
            diff = feats - mean_k
            inv_cov = inv_covs[k]
            dists_sq = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
            min_dists = np.minimum(min_dists, np.sqrt(np.maximum(dists_sq, 0.0)))
        return min_dists

    counts_file = Path(f'experiments/{experiment_name}/counts_per_attack/baseline_counts.json')
    test_accuracy_file = Path(f'experiments/{experiment_name}/counts_per_attack/baseline_accuracy.json')

    lock = mp.Lock()
    for param in range(len(parameters['knn'])):
        print("Training detection models (with cache)...", flush=True)
        cache_base = temp_dir if temp_dir is not None else f'experiments/{experiment_name}'
        cache_dir = os.path.join(cache_base, 'preprocessed')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'detectors_param{param}_features.joblib')

        if (not FORCE_REBUILD_CACHE) and os.path.exists(cache_file):
            print("Loading trained detectors from cache...", flush=True)
            knn, kde, gmm, ocsvm, iforest, mahalanobis_threshold = load(cache_file)
        else:
            train_features_np = np.asarray(train_features)

            knn = NearestNeighbors(n_neighbors=parameters['knn'][param], metric="euclidean")
            kde = KernelDensity(bandwidth=parameters['kde'][param])
            gmm = GaussianMixture(n_components=parameters['gmm'][param])
            ocsvm = OneClassSVM(kernel='rbf', nu=parameters['ocsvm'][param])
            iforest = IsolationForest(n_estimators=parameters['iforest'][param])

            print('Fitting knn', flush=True)
            knn.fit(train_features_np)
            print('Fitting kde', flush=True)
            kde.fit(train_features_np)
            print('Fitting gmm', flush=True)
            gmm.fit(train_features_np)
            print('Fitting ocsvm', flush=True)
            ocsvm.fit(train_features_np)
            print('Fitting iforest', flush=True)
            iforest.fit(train_features_np)

            print('Fitting mahalanobis', flush=True)
            mahalanobis_threshold = np.percentile(
                get_min_mahalanobis_distances(train_features_np),
                parameters['mahalanobis'][param]
            )

            dump((knn, kde, gmm, ocsvm, iforest, mahalanobis_threshold), cache_file)

        Path(f'experiments/{experiment_name}/counts_per_attack/').mkdir(parents=True, exist_ok=True)

        output_file = Path(f'experiments/{experiment_name}/grid_search/baseline.txt')

        if not output_file.exists():
            with open(output_file, 'w') as f:
                f.write("method,parameter,attack,experiment_name,good_defence,wrong_rejection\n")

        for method in methods:
            results = []
            for a in ["test"] + ATTACKS:
                with lock:
                    if output_file.exists():
                        with open(output_file, 'r') as f:
                            existing_lines = f.readlines()
                        prefix = f"{method},{parameters[method][param]},{a}"
                        if any(line.startswith(prefix) for line in existing_lines):
                            print(f"Result already exists for method: {method}, parameter: {str(parameters[method][param])}, attack: {a}, skipping...", flush=True)
                            continue

                print(f"\nEvaluating {a} examples baseline", flush=True)

                if counts_file.exists():
                    with open(counts_file, 'r') as file:
                        counts_current = json.load(file)
                    if all(value == 0 for value in counts_current[method][str(parameters[method][param])][a].values()):
                        counts = counts_current
                    else:
                        print(f'Found and loaded method: {method}, parameter: {str(parameters[method][param])}, attack: {a}', flush=True)
                        continue

                if test_accuracy_file.exists():
                    with open(test_accuracy_file, 'r') as file:
                        test_acc = json.load(file)

                try:
                    if temp_dir is not None:
                        attacked_dataset = torch.load(f'{temp_dir}/experiments/{experiment_name}/adversarial_examples/{a}/adversarial_examples.pth')
                    else:
                        attacked_dataset = torch.load(f'experiments/{experiment_name}/adversarial_examples/{a}/adversarial_examples.pth')
                except Exception:
                    print(f"Attack {a} not found.", flush=True)
                    continue

                # Cache per-attack features to disk to avoid recomputing
                cache_base = temp_dir if temp_dir is not None else f'experiments/{experiment_name}'
                cache_dir = os.path.join(cache_base, 'preprocessed', 'attack_features')
                os.makedirs(cache_dir, exist_ok=True)
                attack_feat_file = os.path.join(cache_dir, f'{a}_features.npy')

                if (not FORCE_REBUILD_CACHE) and os.path.exists(attack_feat_file):
                    current_features = np.load(attack_feat_file)
                else:
                    # attacked_dataset may be on cpu; get_features handles batching & device moves
                    current_features = get_features(attacked_dataset, model)
                    np.save(attack_feat_file, current_features)

                # Ensure numpy 2D
                current_features = np.asarray(current_features).reshape(current_features.shape[0], -1)
                print('Predictions on baseline...', flush=True)
                if method == 'knn':
                    distances, _ = knn.kneighbors(current_features)
                    average_distance = distances.mean(axis=1)
                    predictions = average_distance < np.percentile(average_distance, 10)
                elif method == 'kde':
                    scores = kde.score_samples(current_features)
                    predictions = scores < np.percentile(scores, 10)
                elif method == 'gmm':
                    scores = gmm.score_samples(current_features)
                    predictions = scores < np.percentile(scores, 10)
                elif method == 'ocsvm':
                    predictions = ocsvm.predict(current_features) == -1
                elif method == 'iforest':
                    predictions = iforest.predict(current_features) == -1
                elif method == 'softmax':
                    # batch forward for softmax predictions
                    device = get_device()
                    logits_list = []
                    attacked_t = attacked_dataset if isinstance(attacked_dataset, torch.Tensor) else torch.tensor(attacked_dataset)
                    attacked_t = attacked_t.to(device).float()
                    B = 256
                    with torch.no_grad():
                        for i in range(0, len(attacked_t), B):
                            batch = attacked_t[i:i+B]
                            logits = model.forward(batch)
                            probs = torch.nn.functional.softmax(logits, dim=1)
                            confidences = probs.max(dim=1).values.detach().cpu().numpy()
                            logits_list.extend(confidences)
                    predictions = np.array(logits_list) < np.array(parameters['softmax'][param])
                elif method == 'mahalanobis':
                    predictions = get_min_mahalanobis_distances(current_features) < mahalanobis_threshold
                else:
                    raise ValueError(f"Method {method} not found.")

                for i, is_adversarial in enumerate(predictions):
                    if method == 'softmax':
                        is_adversarial = bool(is_adversarial)
                    results.append((is_adversarial, a != "test"))
                    if a == "test":
                        if not is_adversarial:
                            counts[method][str(parameters[method][param])][a]['not_rejected_and_not_attacked'] += 1
                            pred = torch.argmax(model.forward(torch.tensor(attacked_dataset[i]).unsqueeze(0).to(get_device()).float()))
                            if pred == test_labels[i]:
                                test_acc[method][str(parameters[method][param])] += 1
                        else:
                            counts[method][str(parameters[method][param])][a]['rejected_and_not_attacked'] += 1
                    else:
                        if is_adversarial:
                            counts[method][str(parameters[method][param])][a]['rejected_and_attacked'] += 1
                        else:
                            counts[method][str(parameters[method][param])][a]['not_rejected_and_attacked'] += 1

                if verbose:
                    print(f"\nResults for {method.upper()}:")
                    if a == 'test':
                        print(f'Wrongly rejected test data: {counts[method][str(parameters[method][param])][a]["rejected_and_not_attacked"]}', flush=True)
                        print(f'Trusted test data: {counts[method][str(parameters[method][param])][a]["not_rejected_and_not_attacked"]}', flush=True)
                        if counts[method][str(parameters[method][param])][a]['not_rejected_and_not_attacked'] > 0:
                            test_acc[method][str(parameters[method][param])] /= counts[method][str(parameters[method][param])][a]['not_rejected_and_not_attacked']
                        else:
                            test_acc[method][str(parameters[method][param])] = 0
                        print(f"Accuracy on trusted test data: {test_acc[method][str(parameters[method][param])]}", flush=True)
                    else:
                        print(f'Detected adversarial examples: {counts[method][str(parameters[method][param])][a]["rejected_and_attacked"]}', flush=True)
                        print(f'Missed adversarial examples: {counts[method][str(parameters[method][param])][a]["not_rejected_and_attacked"]}', flush=True)

                good_defence = 0
                wrongly_rejected = 0
                num_att = 0
                for rej, att in results:
                    if att:
                        good_defence += int(rej)
                        num_att += 1
                    else:
                        wrongly_rejected += int(rej)

                perc_good = good_defence / num_att if num_att != 0 else 0
                perc_bad = wrongly_rejected / (len(results) - num_att) if len(results) != num_att else 0
                result_line = f"{method},{parameters[method][param]},{a},{experiment_name},{perc_good},{perc_bad}\n"

                with lock:
                    with open(output_file, 'a') as f:
                        f.write(result_line)
                    with open(counts_file, 'w') as f:
                        json.dump(counts, f, indent=4)
                    with open(test_accuracy_file, 'w') as f:
                        json.dump(test_acc, f, indent=4)


def reject_predicted_attacks_baseline_matrices(
        experiment_name: str,
        num_classes: int,
        dataset: str,
        verbose: bool = True,
        temp_dir: Union[str, None] = None
    ) -> None:
    """
    Baseline detectors running directly on matrices (concatenated matrices per class).
    This function caches the concatenated matrix tensor and caches detectors per param.
    Uses global TruncatedSVD + LedoitWolf shrinkage per-class to compute stable Mahalanobis distances.
    """
    counts_file = Path(f'experiments/{experiment_name}/counts_per_attack/baseline_matrices_counts.json')
    test_accuracy_file = Path(f'experiments/{experiment_name}/counts_per_attack/baseline_matrices_accuracy.json')
    Path(f'experiments/{experiment_name}/counts_per_attack/').mkdir(parents=True, exist_ok=True)

    num_train_examples = 10000
    num_train_examples_per_class = num_train_examples // num_classes

    print('Preparing concatenated matrices (using cache if available)...', flush=True)
    # TODO: create directory preprocessed in the shell script and copy the data to compute node
    base_path = Path(temp_dir) / Path(f'experiments/{experiment_name}') if temp_dir is not None else Path(f'experiments/{experiment_name}')
    preproc_dir = base_path / Path('preprocessed')
    #preproc_dir = os.path.join(base_path, 'preprocessed')
    preproc_dir.mkdir(parents=True, exist_ok=True)
    #os.makedirs(preproc_dir, exist_ok=True)

    train_data_file = os.path.join(preproc_dir, f'train_matrices_n{num_train_examples}_c{num_classes}.pt')
    train_labels_file = os.path.join(preproc_dir, f'train_labels_n{num_train_examples}_c{num_classes}.pt')

    try:
        if (not FORCE_REBUILD_CACHE) and os.path.exists(train_data_file) and os.path.exists(train_labels_file):
            print('Loading concatenated tensors from cache...', flush=True)
            train_data = torch.load(train_data_file, map_location='cpu')
            train_labels = torch.load(train_labels_file, map_location='cpu')
            # keep on CPU for sklearn; move to device only when needed
        else:
            matrix_files = [
                f'{base_path}/matrices/{i}/{j}/matrix.pt'
                for i in range(num_classes)
                for j in range(num_train_examples_per_class)
            ]
            mats = []
            for file in matrix_files:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"Expected matrix file missing: {file}")
                t = torch.load(file, map_location='cpu')
                # ensure shape (1, ...)
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                else:
                    t = t.unsqueeze(0)
                mats.append(t)
            train_data = torch.cat(mats, dim=0)  # CPU tensor
            train_labels = torch.cat([
                torch.full((num_train_examples_per_class,), i, dtype=torch.long)
                for i in range(num_classes)
            ])
            torch.save(train_data, train_data_file)
            torch.save(train_labels, train_labels_file)
    except Exception as e:
        print(f"Error preparing matrices: {e}", flush=True)
        return

    print(f"Loaded {len(train_data)} training examples with shape {train_data.shape}", flush=True)

    # Keep train_data as CPU tensor; convert to numpy for sklearn and flatten per-sample
    train_data_np = train_data.detach().cpu().numpy().reshape(train_data.shape[0], -1)

    parameters = get_parameters_baseline(dataset)
    methods = list(parameters.keys())

    counts = {
        method: {
            str(parameters[method][parameter]): {
                attack: {
                    'not_rejected_and_attacked': 0,
                    'not_rejected_and_not_attacked': 0,
                    'rejected_and_attacked': 0,
                    'rejected_and_not_attacked': 0
                } for attack in ["test"] + ATTACKS
            } for parameter in range(len(parameters[method]))
        } for method in methods
    }

    test_acc = {
        method: {
            str(parameters[method][parameter]): 0
            for parameter in range(len(parameters[method]))
        } for method in methods
    }

    print("Mahalanobis preparation (fast & stable)...", flush=True)
    base_path_str = f'{temp_dir}/experiments/{experiment_name}' if temp_dir is not None else f'experiments/{experiment_name}'
    class_means_file = Path(base_path_str) / 'counts_per_attack' / 'mah_means_proj.npz'
    class_prec_file = Path(base_path_str) / 'counts_per_attack' / 'mah_prec_proj.npz'
    pca_file = Path(base_path_str) / 'counts_per_attack' / 'mah_svd.npz'

    # Decide projection dimension (cap to keep memory reasonable)
    N, D = train_data_np.shape
    # choose n_components <= min(N-1, D), cap at 256 (tunable)
    n_components = min(256, D, max(1, N - 1))
    # but avoid trivial sizes for very small N
    n_components = max(1, min(n_components, N - 1))

    n_jobs = _detect_n_cpus(num_classes)

    if (not FORCE_REBUILD_CACHE) and class_means_file.exists() and class_prec_file.exists() and pca_file.exists():
        print("Loading PCA + class means + precisions from cache...", flush=True)
        with np.load(str(pca_file)) as pz:
            svd_components = pz['components']
            svd_explained = pz['explained_variance'] if 'explained_variance' in pz.files else None
        with np.load(str(class_means_file)) as mzip:
            class_means = {int(k.split('_')[1]): mzip[k] for k in mzip.files}
        with np.load(str(class_prec_file)) as pzip:
            class_precisions = {int(k.split('_')[1]): pzip[k] for k in pzip.files}
        # Create SVD object wrapper (we only need transform via components)
        class _SVDWrapper:
            def __init__(self, components):
                self.components_ = components
            def transform(self, X):
                # X (n, D) -> X @ components_.T (n, k)
                return np.dot(X, self.components_.T)
        svd = _SVDWrapper(svd_components)
        print("Loaded Mahalanobis projection & class precisions.", flush=True)
    else:
        # Compute global TruncatedSVD on flattened matrices to reduce dimension
        print(f"Computing global TruncatedSVD with n_components={n_components} (N={N}, D={D})...", flush=True)
        svd = TruncatedSVD(n_components=n_components, random_state=0)
        train_proj = svd.fit_transform(train_data_np)  # (N, k)
        print("SVD done; projected shape:", train_proj.shape, flush=True)

        # Precompute a global fallback LedoitWolf on projected data (used for small classes)
        print("Fitting global LedoitWolf as fallback...", flush=True)
        lw_global = LedoitWolf().fit(train_proj)
        global_mean = lw_global.location_.astype(np.float32)
        global_precision = lw_global.precision_.astype(np.float32)

        # Build per-class projected data
        train_labels_np = train_labels.detach().cpu().numpy().astype(int)
        class_proj_list = []
        for class_idx in range(num_classes):
            mask = (train_labels_np == class_idx)
            class_data_proj = train_proj[mask]  # shape (n_k, k)
            class_proj_list.append(class_data_proj)

        # parallel fit per-class LedoitWolf for projected data
        def _fit_class(idx, X_proj):
            if X_proj.shape[0] < 2:
                # not enough samples — use global fallback
                return idx, global_mean, global_precision
            try:
                lw = LedoitWolf().fit(X_proj)
                return idx, lw.location_.astype(np.float32), lw.precision_.astype(np.float32)
            except Exception:
                # fallback to global
                return idx, global_mean, global_precision

        print(f"Fitting per-class LedoitWolf in parallel with n_jobs={n_jobs}...", flush=True)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_class)(idx, class_proj_list[idx]) for idx in range(num_classes)
        )

        class_means = {}
        class_precisions = {}
        for idx, mu, prec in results:
            class_means[idx] = mu
            class_precisions[idx] = prec

        # Save PCA components + class_means + class_precisions
        print("Saving PCA components and class Mahalanobis data to cache...", flush=True)
        os.makedirs(os.path.dirname(str(class_means_file)), exist_ok=True)
        # svd.components_ shape (k, D) ; save as components
        np.savez_compressed(str(pca_file), components=svd.components_)
        np.savez_compressed(str(class_means_file), **{f'mean_{k}': class_means[k] for k in class_means})
        np.savez_compressed(str(class_prec_file), **{f'prec_{k}': class_precisions[k] for k in class_precisions})

    # helper: project features then compute vectorized min Mahalanobis
    def get_min_mahalanobis_distances_matrices(features_np: np.ndarray) -> np.ndarray:
        # features_np is (n, D_full)
        proj = svd.transform(np.asarray(features_np).reshape(features_np.shape[0], -1))
        return _vectorized_min_mahalanobis_with_prec(proj, class_means, class_precisions)

    print("Mahalanobis ready (projected).", flush=True)
    if temp_dir is not None:
        test_labels = torch.load(f'{temp_dir}/experiments/{experiment_name}/adversarial_examples/test/labels.pth')
    else:
        test_labels = torch.load(f'experiments/{experiment_name}/adversarial_examples/test/labels.pth')

    print("Start baseline for matrices...", flush=True)
    lock = mp.Lock()
    for param in range(len(parameters['knn'])):
        print("Training detection models (with cache)...", flush=True)
        cache_base = temp_dir if temp_dir is not None else f'experiments/{experiment_name}'
        cache_dir = os.path.join(cache_base, 'preprocessed')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'detectors_param{param}_matrices.joblib')

        if (not FORCE_REBUILD_CACHE) and os.path.exists(cache_file):
            print("Loading trained detectors from cache (matrices)...", flush=True)
            knn, kde, gmm, ocsvm, iforest, mahalanobis_threshold = load(cache_file)
        else:
            print('Start training methods...', flush=True)
            knn = NearestNeighbors(n_neighbors=parameters['knn'][param], metric="euclidean")

            kde = KernelDensity(bandwidth=parameters['kde'][param])

            gmm = GaussianMixture(n_components=parameters['gmm'][param])

            ocsvm = OneClassSVM(kernel='rbf', nu=parameters['ocsvm'][param])

            iforest = IsolationForest(n_estimators=parameters['iforest'][param])

            print('Fitting.', flush=True)
            knn.fit(train_data_np)
            print('KNN ready...', flush=True)
            kde.fit(train_data_np)
            print('KDE ready...', flush=True)
            gmm.fit(train_data_np)
            print('GAUSSIAN ready...', flush=True)
            ocsvm.fit(train_data_np)
            print('OCSVM ready...', flush=True)
            iforest.fit(train_data_np)
            print('IFOREST ready...', flush=True)

            mahalanobis_threshold = np.percentile(get_min_mahalanobis_distances_matrices(train_data_np),
                                                 parameters['mahalanobis'][param])
            print('MAHALANOBIS ready...', flush=True)
            dump((knn, kde, gmm, ocsvm, iforest, mahalanobis_threshold), cache_file)

        output_file = Path(f'experiments/{experiment_name}/grid_search/baseline_matrices.txt')
        Path(f'experiments/{experiment_name}/grid_search/').mkdir(parents=True, exist_ok=True)
        if not output_file.exists():
            with open(output_file, 'w') as f:
                f.write("method,parameter,experiment_name,good_defence,wrong_rejection\n")

        for method in methods:
            results = []
            for a in ["test"] + ATTACKS:
                with lock:
                    if output_file.exists():
                        with open(output_file, 'r') as f:
                            existing_lines = f.readlines()
                        prefix = f"{method},{parameters[method][param]},{a}"
                        if any(line.startswith(prefix) for line in existing_lines):
                            print(f"Result already exists for method: {method}, parameter: {str(parameters[method][param])}, attack: {a}, skipping...", flush=True)
                            continue

                if counts_file.exists():
                    with open(counts_file, 'r') as file:
                        counts_current = json.load(file)
                    if all(value == 0 for value in counts_current[method][str(parameters[method][param])][a].values()):
                        counts = counts_current
                    else:
                        print(f'Found and loaded method: {method}, parameter: {str(parameters[method][param])}, attack: {a}', flush=True)
                        continue

                if test_accuracy_file.exists():
                    with open(test_accuracy_file, 'r') as file:
                        test_acc = json.load(file)

                # Efficiently load adversarial matrices for attack `a` into a numpy array (CPU)
                attack_mats = []
                i = 0
                attack_base = Path(f'{base_path}/adversarial_matrices/{a}' if temp_dir is not None
                                   else f'experiments/{experiment_name}/adversarial_matrices/{a}')
                while True:
                    mat_path = attack_base / Path(f'{i}/matrix.pth')
                    if not os.path.exists(mat_path):
                        break
                    try:
                        m = torch.load(mat_path, map_location='cpu')
                    except Exception:
                        break
                    attack_mats.append(m)
                    i += 1

                if len(attack_mats) == 0:
                    print(f"Attack {a} not found.", flush=True)
                    continue

                # Convert to numpy matrix dataset (N, D)
                attacked_np = np.stack([_ensure_np(m).reshape(-1) for m in attack_mats], axis=0)
                print('Predicting baseline for matrices...', flush=True)
                if method == 'knn':
                    distances, _ = knn.kneighbors(attacked_np)
                    average_distance = distances.mean(axis=1)
                    predictions = average_distance < np.percentile(average_distance, 10)
                elif method == 'kde':
                    scores = kde.score_samples(attacked_np)
                    predictions = scores < np.percentile(scores, 10)
                elif method == 'gmm':
                    scores = gmm.score_samples(attacked_np)
                    predictions = scores < np.percentile(scores, 10)
                elif method == 'ocsvm':
                    predictions = ocsvm.predict(attacked_np) == -1
                elif method == 'iforest':
                    predictions = iforest.predict(attacked_np) == -1
                elif method == 'mahalanobis':
                    predictions = get_min_mahalanobis_distances_matrices(attacked_np) < mahalanobis_threshold
                else:
                    raise ValueError(f"Method {method} not found.")

                for i, is_adversarial in enumerate(predictions):
                    results.append((is_adversarial, a != "test"))
                    if a == "test":
                        if not is_adversarial:
                            counts[method][str(parameters[method][param])][a]['not_rejected_and_not_attacked'] += 1
                            # simple heuristic for prediction on matrix-form data
                            pred = int(np.argmax(attacked_np[i].sum(axis=0))) if attacked_np.ndim == 2 else 0
                            if pred == test_labels[i]:
                                test_acc[method][str(parameters[method][param])] += 1
                        else:
                            counts[method][str(parameters[method][param])][a]['rejected_and_not_attacked'] += 1
                    else:
                        if is_adversarial:
                            counts[method][str(parameters[method][param])][a]['rejected_and_attacked'] += 1
                        else:
                            counts[method][str(parameters[method][param])][a]['not_rejected_and_attacked'] += 1

                if verbose:
                    print(f"\nResults for {method.upper()}:")
                    if a == 'test':
                        print(f'Wrongly rejected test data: {counts[method][str(parameters[method][param])][a]["rejected_and_not_attacked"]}', flush=True)
                        print(f'Trusted test data: {counts[method][str(parameters[method][param])][a]["not_rejected_and_not_attacked"]}', flush=True)
                        if counts[method][str(parameters[method][param])][a]['not_rejected_and_not_attacked'] > 0:
                            test_acc[method][str(parameters[method][param])] = test_acc[method][str(parameters[method][param])] / counts[method][str(parameters[method][param])][a]['not_rejected_and_not_attacked']
                        else:
                            test_acc[method][str(parameters[method][param])] = 0
                        print(f"Accuracy on trusted test data: {test_acc[method][str(parameters[method][param])]}", flush=True)
                    else:
                        print(f'Detected adversarial examples: {counts[method][str(parameters[method][param])][a]["rejected_and_attacked"]}', flush=True)
                        print(f'Missed adversarial examples: {counts[method][str(parameters[method][param])][a]["not_rejected_and_attacked"]}', flush=True)

                good_defence = 0
                wrongly_rejected = 0
                num_att = 0
                for rej, att in results:
                    if att:
                        good_defence += int(rej)
                        num_att += 1
                    else:
                        wrongly_rejected += int(rej)

                result_line = f"{method},{str(parameters[method][param])},{a},{experiment_name},{good_defence/num_att},{wrongly_rejected/(len(results)-num_att)}\n"

                with lock:
                    with open(output_file, 'a') as f:
                        f.write(result_line)
                    with open(counts_file, 'w') as f:
                        json.dump(counts, f, indent=4)
                    with open(test_accuracy_file, 'w') as f:
                        json.dump(test_acc, f, indent=4)


def main(
        experiment_name: Union[str, None] = None,
        t_epsilon: Union[float, None] = None,
        epsilon: Union[float, None] = None,
        epsilon_p: Union[float, None] = None,
        temp_dir: Union[str, None] = None,
        baseline: bool = False
    ) -> str:

    args = Namespace(
        experiment_name=experiment_name,
        t_epsilon=t_epsilon,
        epsilon=epsilon,
        epsilon_p=epsilon_p,
        temp_dir=temp_dir
    )

    if args.experiment_name is not None:
        experiment = DEFAULT_EXPERIMENTS[f'{args.experiment_name}']
        architecture_index = experiment['architecture_index']
        dataset = experiment['dataset']
        epoch = experiment['epochs']
    else:
        raise ValueError("Experiment not specified in constants/constants.py")

    print("Detecting adversarial examples for Experiment: ", args.experiment_name, flush=True)

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{args.experiment_name}/weights') / Path(f'epoch_{epoch}.pth')
        matrices_path = Path(f'{args.temp_dir}/experiments/{args.experiment_name}/matrices/matrix_statistics.json')
        ellipsoids_file = open(f"{args.temp_dir}/experiments/{args.experiment_name}/matrices/matrix_statistics.json")
    else:
        weights_path = Path(f'experiments/{args.experiment_name}/weights') / Path(f'epoch_{epoch}.pth')
        matrices_path = Path(f'experiments/{args.experiment_name}/matrices/matrix_statistics.json')
        ellipsoids_file = open(f"experiments/{args.experiment_name}/matrices/matrix_statistics.json")

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    if not matrices_path.exists():
        raise ValueError(f"Matrix statistics have to be computed")

    ellipsoids = json.load(ellipsoids_file)

    if baseline:
        reject_predicted_attacks_baseline(
            experiment_name=args.experiment_name,
            weights_path=weights_path,
            architecture_index=architecture_index,
            input_shape=input_shape,
            num_classes=num_classes,
            verbose=True,
            temp_dir=args.temp_dir
        )
        print("BASELINE FINISHED... \n", flush=True)
        reject_predicted_attacks_baseline_matrices(
            experiment_name=args.experiment_name,
            num_classes=num_classes,
            dataset=dataset,
            verbose=True,
            temp_dir=args.temp_dir
        )
        print("MATRIX BASELINE FINISHED...\n", flush=True)

    result = reject_predicted_attacks(
        experiment_name=args.experiment_name,
        weights_path=weights_path,
        architecture_index=architecture_index,
        input_shape=input_shape,
        num_classes=num_classes,
        ellipsoids=ellipsoids,
        t_epsilon=args.t_epsilon,
        epsilon=args.epsilon,
        epsilon_p=args.epsilon_p,
        verbose=True,
        temp_dir=args.temp_dir
    )

    return result


if __name__ == '__main__':
    main()
