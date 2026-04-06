"""
Empirical validation of Theorem 4.5 (Distance Lower Bound).

Theorem 4.5:  ||M(W,f)(x) - M(W,f)(x')|| >= gamma * ||f_W(x) - f_W(x')||

Knowledge matrix distances lower-bound logit distances.  This script
estimates gamma from data, verifies the bound holds, and shows that KMs
amplify logit-space separations MORE than penultimate activations do.

The existing pipeline saves only misclassified adversarial examples,
discarding the clean-adversarial pairing.  This script generates
adversarial examples on-the-fly for a small subset (~200 samples) and
keeps ALL pairs regardless of misclassification.

Usage:
    python validate_theorem45.py --experiment alexnet_cifar10
    python validate_theorem45.py --experiment alexnet_cifar10 --num_samples 200 --temp_dir $SLURM_TMPDIR
    python validate_theorem45.py --experiment alexnet_cifar10 --attacks FGSM PGD CW DeepFool
"""

import json
import time
import torch
import torchattacks
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

from utils.utils import (
    get_model, get_dataset, get_input_shape, get_num_classes,
    get_device, subset,
)
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS
from compare_representations import extract_penultimate_features


# ---------------------------------------------------------------------------
# Attack generation (keeps all pairs, not just misclassified)
# ---------------------------------------------------------------------------

# Same mapping as generate_adversarial_examples.py
ATTACK_MAP = {
    "GN": "GN", "FGSM": "FGSM", "PGD": "PGD",
    "EOTPGD": "EOTPGD", "MIFGSM": "MIFGSM", "VMIFGSM": "VMIFGSM",
    "CW": "CW", "DeepFool": "DeepFool", "Pixle": "Pixle",
    "APGD": "APGD", "APGDT": "APGDT", "FAB": "FAB", "Square": "Square",
    "SPSA": "SPSA", "EADL1": "EADL1", "EADEN": "EADEN",
}


def generate_adversarial_pairs(model, data, labels, attack_name, device,
                               batch_size=8):
    """Generate adversarial examples, keeping ALL paired (clean, adv) samples.

    Returns:
        (clean_tensor, adv_tensor): tensors on CPU of shape (N, C, H, W).
        Returns (None, None) if the attack class is unavailable.
    """
    cls_name = ATTACK_MAP.get(attack_name)
    if cls_name is None:
        print(f"  Unknown attack: {attack_name}", flush=True)
        return None, None
    attack_cls = getattr(torchattacks, cls_name, None)
    if attack_cls is None:
        print(f"  WARNING: {attack_name} ({cls_name}) not available in "
              f"torchattacks {torchattacks.__version__}. Skipping.", flush=True)
        return None, None

    model.eval()
    attack_instance = attack_cls(model)

    adv_list = []
    clean_list = []
    for i in range(0, len(data), batch_size):
        xb = data[i:i + batch_size].to(device).float()
        yb = labels[i:i + batch_size].to(device)
        try:
            adv_batch = attack_instance(xb, yb)
        except Exception as e:
            print(f"  Error in {attack_name} batch {i}: {e}", flush=True)
            del xb, yb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        clean_list.append(xb.cpu())
        adv_list.append(adv_batch.detach().cpu())
        del xb, yb, adv_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not clean_list:
        return None, None
    return torch.cat(clean_list), torch.cat(adv_list)


# ---------------------------------------------------------------------------
# Distance computations
# ---------------------------------------------------------------------------

def compute_logit_distances(model, clean, adversarial, device, batch_size=64):
    """||f_W(x) - f_W(x')|| for each pair (Frobenius / L2 norm on logit vectors)."""
    model.eval()
    distances = []
    with torch.no_grad():
        for i in range(0, len(clean), batch_size):
            c = clean[i:i + batch_size].to(device).float()
            a = adversarial[i:i + batch_size].to(device).float()
            out_c = model(c)
            out_a = model(a)
            d = torch.linalg.norm(out_c.double() - out_a.double(), dim=1)  # per-sample L2
            distances.append(d.cpu().numpy())
            del c, a, out_c, out_a, d
    return np.concatenate(distances)


def compute_penultimate_distances(model, clean, adversarial, batch_size=128):
    """||h(x) - h(x')|| for each pair."""
    feats_c = extract_penultimate_features(model, clean, batch_size=batch_size)
    feats_a = extract_penultimate_features(model, adversarial, batch_size=batch_size)
    return np.linalg.norm(feats_c - feats_a, axis=1)


def compute_matrix_distances(model, clean, adversarial, device,
                             batch_size_mc=1800):
    """||M(W,f)(x) - M(W,f)(x')|| for each pair (Frobenius norm).

    KnowledgeMatrixComputer expects 3D input (C, H, W), so we process
    one sample at a time.
    """
    model.eval()
    mc = KnowledgeMatrixComputer(model, batch_size=batch_size_mc, device=device)

    n = len(clean)
    distances = np.zeros(n)
    for i in range(n):
        # 3D input: (C, H, W) — no unsqueeze!
        sample_c = clean[i].to(device).float()
        sample_a = adversarial[i].to(device).float()
        mat_c = mc.forward(sample_c)
        mat_a = mc.forward(sample_a)
        distances[i] = torch.linalg.norm((mat_c.double() - mat_a.double())).item()
        del mat_c, mat_a, sample_c, sample_a
        if torch.cuda.is_available() and (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"    Matrix distances: {i + 1}/{n}", flush=True)
    return distances


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _stats(arr):
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
    }


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path, data):
    """Atomic JSON write: write to .tmp then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


def _load_checkpoint(path):
    """Load checkpoint if valid, else return {}."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'per_attack' not in data:
            print(f"  WARNING: Checkpoint has unexpected structure, starting fresh.",
                  flush=True)
            return {}
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: Corrupt checkpoint ({e}), starting fresh.", flush=True)
        return {}


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_theorem45(experiment_name, num_samples=200, attacks=None,
                       temp_dir=None, matrix_batch_size=1800):
    """Run Theorem 4.5 validation for one experiment.

    Returns:
        dict: full results structure.
    """
    exp_config = DEFAULT_EXPERIMENTS[experiment_name]
    dataset = exp_config['dataset']
    arch_idx = exp_config['architecture_index']
    epoch = exp_config['epochs']

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)
    device = get_device()

    base = (f'{temp_dir}/experiments/{experiment_name}'
            if temp_dir else f'experiments/{experiment_name}')

    # Find weights (same logic as isomorphism_experiment.py)
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
            raise FileNotFoundError(f"No weights found in {weights_dir}")
    print(f"Using weights: {weights_path}", flush=True)

    model = get_model(weights_path, arch_idx, input_shape, num_classes, device)
    model.eval()

    # Load test data
    print("Loading test data...", flush=True)
    _, test_set = get_dataset(dataset, data_loader=False, data_path=temp_dir)
    test_data, test_labels = subset(test_set, num_samples, input_shape)
    print(f"Test subset: {test_data.shape}", flush=True)

    attack_list = attacks if attacks is not None else ATTACKS

    # --- Checkpoint: resume from previous partial run ---
    ckpt_path = Path(f'experiments/{experiment_name}/theorem45/theorem45_checkpoint.json')
    ckpt = _load_checkpoint(ckpt_path)

    per_attack = {}
    all_gammas = []
    all_amp_M = []
    all_amp_h = []

    if (ckpt
            and ckpt.get('experiment') == experiment_name
            and ckpt.get('num_samples') == num_samples):
        completed = set(ckpt['per_attack'].keys())
        remaining = [a for a in attack_list if a not in completed]
        # Reconstruct aggregated lists from checkpoint
        for atk_name, atk_result in ckpt['per_attack'].items():
            per_attack[atk_name] = atk_result
            all_gammas.append(atk_result['gamma_empirical'])
            all_amp_M.append(atk_result['amplification_M_median'])
            all_amp_h.append(atk_result['amplification_h_median'])
        if completed:
            print(f"  Resuming from checkpoint: {len(completed)} attacks done "
                  f"({', '.join(sorted(completed))}), "
                  f"{len(remaining)} remaining.", flush=True)
    else:
        remaining = list(attack_list)
        if ckpt:
            print(f"  Checkpoint discarded (experiment/num_samples mismatch).",
                  flush=True)

    attack_times = []
    for idx, attack_name in enumerate(remaining):
        eta_str = ""
        if attack_times:
            avg_t = np.mean(attack_times)
            eta_sec = avg_t * (len(remaining) - idx)
            eta_str = f"  |  ETA: {eta_sec / 60:.0f}min"
        print(f"\n{'='*60}", flush=True)
        print(f"  Attack: {attack_name}  [{idx + 1}/{len(remaining)}]{eta_str}",
              flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.perf_counter()

        # Generate paired adversarial examples
        clean, adv = generate_adversarial_pairs(
            model, test_data, test_labels, attack_name, device)
        if clean is None:
            print(f"  Skipping {attack_name} (generation failed).", flush=True)
            continue

        n_pairs = len(clean)
        print(f"  Pairs: {n_pairs}", flush=True)

        # Compute distances
        print("  Computing logit distances...", flush=True)
        d_f = compute_logit_distances(model, clean, adv, device)

        print("  Computing penultimate distances...", flush=True)
        d_h = compute_penultimate_distances(model, clean, adv)

        print("  Computing matrix distances...", flush=True)
        d_M = compute_matrix_distances(model, clean, adv, device,
                                       batch_size_mc=matrix_batch_size)

        # Filter pairs where d_f > 0 to avoid division by zero
        # Use relative threshold to avoid discarding valid pairs in low-magnitude regimes
        d_f_max = float(np.max(d_f)) if len(d_f) > 0 else 1.0
        eps_threshold = max(1e-12, 1e-6 * d_f_max)
        valid = d_f > eps_threshold
        n_valid = int(valid.sum())
        if n_valid == 0:
            print(f"  WARNING: All logit distances are ~0 for {attack_name}. "
                  f"Skipping.", flush=True)
            continue

        d_f_v = d_f[valid]
        d_h_v = d_h[valid]
        d_M_v = d_M[valid]

        # Compute ratios
        ratio_M = d_M_v / d_f_v   # d_M / d_f per pair
        ratio_h = d_h_v / d_f_v   # d_h / d_f per pair

        gamma_empirical = float(np.min(ratio_M))

        # Bootstrap 95% CI for gamma
        n_bootstrap = 1000
        boot_gammas = []
        rng = np.random.RandomState(42 + idx)
        for _ in range(n_bootstrap):
            idx = rng.choice(len(ratio_M), size=len(ratio_M), replace=True)
            boot_gammas.append(float(np.min(ratio_M[idx])))
        gamma_ci_lower = float(np.percentile(boot_gammas, 2.5))
        gamma_ci_upper = float(np.percentile(boot_gammas, 97.5))

        elapsed = time.perf_counter() - t0

        result = {
            'num_pairs': n_pairs,
            'num_valid_pairs': n_valid,
            'gamma_empirical': gamma_empirical,
            'gamma_ci_95': [gamma_ci_lower, gamma_ci_upper],
            'bound_satisfaction_rate_note': "Tautological (gamma = min ratio, so always 1.0). See gamma_ci_95 instead.",
            'amplification_M_median': float(np.median(ratio_M)),
            'amplification_h_median': float(np.median(ratio_h)),
            'amplification_M_mean': float(np.mean(ratio_M)),
            'amplification_h_mean': float(np.mean(ratio_h)),
            'd_M_stats': _stats(d_M_v),
            'd_h_stats': _stats(d_h_v),
            'd_f_stats': _stats(d_f_v),
            'elapsed_seconds': float(elapsed),
        }
        per_attack[attack_name] = result

        all_gammas.append(gamma_empirical)
        all_amp_M.append(float(np.median(ratio_M)))
        all_amp_h.append(float(np.median(ratio_h)))

        print(f"  gamma = {gamma_empirical:.4f}  |  "
              f"gamma 95% CI = [{gamma_ci_lower:.4f}, {gamma_ci_upper:.4f}]  |  "
              f"d_M/d_f = {np.median(ratio_M):.2f} (med)  |  "
              f"d_h/d_f = {np.median(ratio_h):.2f} (med)  |  "
              f"{elapsed:.1f}s", flush=True)

        attack_times.append(elapsed)

        # Save checkpoint after each attack
        _save_checkpoint(ckpt_path, {
            'experiment': experiment_name,
            'num_samples': num_samples,
            'per_attack': per_attack,
        })

        # Cleanup
        del clean, adv, d_f, d_h, d_M
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    aggregate = {}
    if all_gammas:
        aggregate = {
            'gamma_global': float(np.min(all_gammas)),
            'mean_amplification_M': float(np.mean(all_amp_M)),
            'mean_amplification_h': float(np.mean(all_amp_h)),
            'ratio_M_over_h': float(np.mean(all_amp_M) / np.mean(all_amp_h))
                if np.mean(all_amp_h) > 1e-12 else None,
        }

    # Summary
    print(f"\n\n{'#'*60}", flush=True)
    print(f"  THEOREM 4.5 VALIDATION: {experiment_name}", flush=True)
    print(f"{'#'*60}", flush=True)
    if aggregate:
        print(f"  Global gamma:           {aggregate['gamma_global']:.4f}",
              flush=True)
        print(f"  Mean amplif. (KM):      {aggregate['mean_amplification_M']:.2f}",
              flush=True)
        print(f"  Mean amplif. (penult):  {aggregate['mean_amplification_h']:.2f}",
              flush=True)
        r = aggregate.get('ratio_M_over_h')
        if r is not None:
            print(f"  KM / penult ratio:      {r:.2f}x", flush=True)
    else:
        print("  No valid results.", flush=True)

    # Save
    out_dir = Path(f'experiments/{experiment_name}/theorem45/')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'theorem45_results.json'

    save_data = {
        'experiment': experiment_name,
        'num_samples': num_samples,
        'per_attack': per_attack,
        'aggregate': aggregate,
    }
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {out_file}", flush=True)

    # Clean up checkpoint (no longer needed)
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"  Checkpoint cleaned up.", flush=True)

    return save_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser(
        description="Empirical validation of Theorem 4.5: knowledge matrix "
                    "distances lower-bound logit distances."
    )
    parser.add_argument(
        "--experiment", type=str, default="alexnet_cifar10",
        help="Experiment name (key in DEFAULT_EXPERIMENTS)."
    )
    parser.add_argument(
        "--num_samples", type=int, default=200,
        help="Number of test samples to use (per attack)."
    )
    parser.add_argument(
        "--attacks", nargs="+", default=None,
        help="Subset of attacks to validate (e.g. --attacks FGSM PGD CW). "
             "If not specified, all ATTACKS are used."
    )
    parser.add_argument(
        "--matrix_batch_size", type=int, default=1800,
        help="Batch size for KnowledgeMatrixComputer."
    )
    parser.add_argument(
        "--temp_dir", type=str, default=None,
        help="Temporary directory (cluster SLURM_TMPDIR)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Theorem 4.5 Validation", flush=True)
    print(f"  Experiment:    {args.experiment}", flush=True)
    print(f"  Samples:       {args.num_samples}", flush=True)
    print(f"  Attacks:       {args.attacks or 'all'}", flush=True)
    print(f"  Temp dir:      {args.temp_dir}", flush=True)

    t_start = time.perf_counter()

    validate_theorem45(
        experiment_name=args.experiment,
        num_samples=args.num_samples,
        attacks=args.attacks,
        temp_dir=args.temp_dir,
        matrix_batch_size=args.matrix_batch_size,
    )

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 3600:.2f}h)", flush=True)


if __name__ == "__main__":
    main()
