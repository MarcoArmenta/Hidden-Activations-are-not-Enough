"""
Out-of-Distribution (OOD) detection using knowledge matrices.

Loads a trained model and its matrix statistics (class ellipsoids),
computes knowledge matrices for an OOD dataset, and checks whether
the Mahalanobis-style distances reveal OOD inputs.

Example usage:
    python ood_detection.py --in_experiment mlp_mnist --ood_dataset fashion
    python ood_detection.py --in_experiment mlp_mnist --ood_dataset fashion --num_samples 500
"""

import json
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from utils.utils import (
    get_model, get_dataset, get_input_shape, get_num_classes,
    get_device, get_ellipsoid_data, zero_std, subset,
)
from constants.constants import DEFAULT_EXPERIMENTS
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer


def parse_args():
    parser = ArgumentParser(description="OOD detection via knowledge matrices")
    parser.add_argument("--in_experiment", type=str, required=True,
                        help="Experiment name for the in-distribution model (e.g., mlp_mnist)")
    parser.add_argument("--ood_dataset", type=str, required=True,
                        help="OOD dataset name (e.g., fashion, cifar10)")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of OOD samples to evaluate")
    parser.add_argument("--epsilon_p", type=float, default=0.1,
                        help="Threshold for zero_std (same as detection pipeline)")
    parser.add_argument("--temp_dir", type=str, default=None,
                        help="Temporary directory for cluster runs")
    return parser.parse_args()


def main():
    args = parse_args()

    exp_config = DEFAULT_EXPERIMENTS[args.in_experiment]
    in_dataset = exp_config['dataset']
    arch_idx = exp_config['architecture_index']
    epoch = exp_config['epochs']

    input_shape = get_input_shape(in_dataset)
    num_classes = get_num_classes(in_dataset)
    device = get_device()

    # Verify OOD dataset has compatible input shape
    ood_input_shape = get_input_shape(args.ood_dataset)
    if ood_input_shape != input_shape:
        print(f"WARNING: Input shape mismatch: model expects {input_shape}, "
              f"OOD dataset has {ood_input_shape}. Results may be unreliable.")

    # Load model
    base = f'{args.temp_dir}/experiments/{args.in_experiment}' if args.temp_dir else f'experiments/{args.in_experiment}'
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
        if not epoch_files:
            raise FileNotFoundError(f"No weight files in {weights_dir}")
        weights_path = epoch_files[-1]
    print(f"Using weights: {weights_path}")
    model = get_model(weights_path, arch_idx, input_shape, num_classes, device)
    model.eval()

    # Load matrix statistics (class ellipsoids)
    stats_path = Path(base) / 'matrices/matrix_statistics.json'
    with open(stats_path) as f:
        ellipsoids = json.load(f)

    # Load OOD dataset
    _, ood_set = get_dataset(args.ood_dataset, data_loader=False, data_path=args.temp_dir)
    ood_data, ood_labels = subset(ood_set, args.num_samples, input_shape)

    # Also load a sample of in-distribution test data for comparison
    _, in_set = get_dataset(in_dataset, data_loader=False, data_path=args.temp_dir)
    in_data, in_labels = subset(in_set, args.num_samples, input_shape)

    # Compute knowledge matrices and zero-dimension counts
    kmc = KnowledgeMatrixComputer(model)

    def compute_zero_dims(data, label_name):
        zero_counts = []
        for i in range(len(data)):
            x = data[i].to(device)  # 3D tensor (C, H, W)
            with torch.no_grad():
                mat = kmc.forward(x)
                pred = torch.argmax(model.forward(x.unsqueeze(0).float()), dim=1)
            std = get_ellipsoid_data(ellipsoids, pred, "std")
            c = zero_std(mat, std, args.epsilon_p).item()
            zero_counts.append(c)
            if (i + 1) % 100 == 0:
                print(f"  [{label_name}] Processed {i+1}/{len(data)}", flush=True)
        return np.array(zero_counts)

    print(f"\nComputing knowledge matrices for IN-distribution ({in_dataset}) samples...")
    in_zeros = compute_zero_dims(in_data, "IN")

    print(f"\nComputing knowledge matrices for OOD ({args.ood_dataset}) samples...")
    ood_zeros = compute_zero_dims(ood_data, "OOD")

    # Report statistics
    print("\n" + "=" * 60)
    print("OOD DETECTION RESULTS (Knowledge Matrix Zero Dimensions)")
    print("=" * 60)
    print(f"In-distribution:  {in_dataset} ({len(in_data)} samples)")
    print(f"Out-of-distribution: {args.ood_dataset} ({len(ood_data)} samples)")
    print(f"Epsilon_p: {args.epsilon_p}")
    print("-" * 60)
    print(f"IN  zero dims — mean: {in_zeros.mean():.2f}, std: {in_zeros.std():.2f}, "
          f"min: {in_zeros.min()}, max: {in_zeros.max()}")
    print(f"OOD zero dims — mean: {ood_zeros.mean():.2f}, std: {ood_zeros.std():.2f}, "
          f"min: {ood_zeros.min()}, max: {ood_zeros.max()}")

    # Simple threshold-based detection (using in-distribution statistics)
    for percentile in [5, 10, 25]:
        threshold = np.percentile(in_zeros, percentile)
        ood_detected = (ood_zeros < threshold).sum()
        in_rejected = (in_zeros < threshold).sum()
        print(f"\nThreshold at {percentile}th percentile of IN ({threshold:.1f}):")
        print(f"  OOD detected (TPR): {ood_detected}/{len(ood_zeros)} "
              f"({100*ood_detected/len(ood_zeros):.1f}%)")
        print(f"  IN rejected  (FPR): {in_rejected}/{len(in_zeros)} "
              f"({100*in_rejected/len(in_zeros):.1f}%)")

    # AUROC computation
    from sklearn.metrics import roc_auc_score
    labels = np.concatenate([np.zeros(len(in_zeros)), np.ones(len(ood_zeros))])
    scores = np.concatenate([-in_zeros, -ood_zeros])  # lower zero dims = more anomalous
    auroc = roc_auc_score(labels, scores)
    print(f"\nAUROC: {auroc:.4f}")

    # Save results
    results = {
        'in_experiment': args.in_experiment,
        'in_dataset': in_dataset,
        'ood_dataset': args.ood_dataset,
        'num_samples': args.num_samples,
        'epsilon_p': args.epsilon_p,
        'in_zero_mean': float(in_zeros.mean()),
        'in_zero_std': float(in_zeros.std()),
        'ood_zero_mean': float(ood_zeros.mean()),
        'ood_zero_std': float(ood_zeros.std()),
        'auroc': float(auroc),
    }
    out_path = Path(f'experiments/{args.in_experiment}/ood_detection/')
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / f'ood_{args.ood_dataset}.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_path / f'ood_{args.ood_dataset}.json'}")


if __name__ == "__main__":
    main()
