"""
GPU Calibration Script — Automated batch_size tuning and time/memory estimation.

Runs before the main pipeline to:
1. Train the model for 2 epochs with random weights → measure training time/memory
2. Binary-search for the optimal batch_size achieving ~93% GPU utilization
3. Time matrix computation on ~50 samples → estimate total pipeline duration
4. Time adversarial attacks on 10 samples → estimate Step C duration
5. Save everything to experiments/{experiment}/calibration.json

If calibration.json already exists, exits immediately.

Usage:
    python calibrate.py --experiment_name alexnet_cifar10 --temp_dir $SLURM_TMPDIR
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS
from utils.utils import (
    get_architecture, get_dataset, get_input_shape,
    get_num_classes, get_device, subset
)


def parse_args():
    parser = ArgumentParser(description="GPU calibration for matrix computation pipeline")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--temp_dir", type=str, default=None)
    parser.add_argument("--target_utilization", type=float, default=0.70)
    parser.add_argument("--timing_samples", type=int, default=50)
    parser.add_argument("--total_chunks", type=int, default=8)
    parser.add_argument("--num_samples_per_class", type=int, default=100)
    parser.add_argument("--samples_per_attack", type=int, default=500)
    parser.add_argument("--force", action="store_true", help="Force re-calibration even if calibration.json exists")
    return parser.parse_args()


def seconds_to_slurm_time(seconds):
    """Convert seconds to HH:MM:SS format for SLURM --time."""
    seconds = int(math.ceil(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def bytes_to_slurm_mem(nbytes):
    """Convert bytes to 'XG' format for SLURM --mem, rounded up."""
    gb = math.ceil(nbytes / (1024 ** 3))
    return f"{max(gb, 1)}G"


def train_2_epochs(model, train_loader, device):
    """Train model for 2 epochs, return (time_seconds, peak_memory_bytes)."""
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    start = time.time()
    for epoch in range(2):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    elapsed = time.time() - start

    peak_mem = torch.cuda.max_memory_allocated(device)
    return elapsed, peak_mem


def probe_batch_size(model, sample_input, batch_size, device):
    """
    Try computing one matrix with the given batch_size.
    Returns (success, peak_memory_bytes).
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        mc = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)
        mc.forward(sample_input.to(device))
        peak = torch.cuda.max_memory_allocated(device)
        return True, peak
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False, 0
        raise


def find_optimal_batch_size(model, sample_input, device, target_utilization=0.85):
    """
    Binary search for the largest batch_size that keeps GPU memory <= target.
    Returns (batch_size, peak_memory_bytes).
    """
    total_memory = torch.cuda.get_device_properties(device).total_memory
    target_mem = int(total_memory * target_utilization)

    C, H, W = model.input_shape
    total_positions = C * H * W

    print(f"GPU total memory: {total_memory / 1e9:.1f} GB", flush=True)
    print(f"Target memory ({target_utilization*100:.0f}%): {target_mem / 1e9:.1f} GB", flush=True)
    print(f"Total input positions: {total_positions}", flush=True)

    # Phase 1: exponential growth to find upper bound
    best_bs = 64
    best_peak = 0
    test_bs = 256

    print("Phase 1: Finding upper bound...", flush=True)
    while test_bs <= total_positions:
        success, peak = probe_batch_size(model, sample_input, test_bs, device)
        if success and peak <= target_mem:
            best_bs = test_bs
            best_peak = peak
            print(f"  batch_size={test_bs}: OK (peak={peak/1e9:.2f} GB, {peak/total_memory*100:.1f}%)", flush=True)
            test_bs *= 2
        else:
            if success:
                print(f"  batch_size={test_bs}: Over target (peak={peak/1e9:.2f} GB, {peak/total_memory*100:.1f}%)", flush=True)
            else:
                print(f"  batch_size={test_bs}: OOM", flush=True)
            break

    # Phase 2: binary search between best_bs and test_bs
    low = best_bs
    high = min(test_bs, total_positions)

    print(f"Phase 2: Binary search [{low}, {high}]...", flush=True)
    while low <= high:
        mid = (low + high) // 2
        if mid == best_bs:
            break

        success, peak = probe_batch_size(model, sample_input, mid, device)
        if success and peak <= target_mem:
            best_bs = mid
            best_peak = peak
            print(f"  batch_size={mid}: OK (peak={peak/1e9:.2f} GB, {peak/total_memory*100:.1f}%)", flush=True)
            low = mid + 1
        else:
            if success:
                print(f"  batch_size={mid}: Over target (peak={peak/1e9:.2f} GB)", flush=True)
            else:
                print(f"  batch_size={mid}: OOM", flush=True)
            high = mid - 1

    print(f"Optimal batch_size: {best_bs} (peak={best_peak/1e9:.2f} GB, {best_peak/total_memory*100:.1f}%)", flush=True)
    return best_bs, best_peak


def time_matrix_computation(model, dataset_tensor, batch_size, device, n_samples=50):
    """
    Compute n_samples matrices, return average seconds per matrix.
    """
    mc = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)
    model.eval()

    n = min(n_samples, len(dataset_tensor))
    times = []

    for i in range(n):
        im = dataset_tensor[i].to(device)
        torch.cuda.empty_cache()

        start = time.time()
        mc.forward(im)
        elapsed = time.time() - start
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            avg_so_far = sum(times) / len(times)
            print(f"  Timed {i+1}/{n} matrices (avg: {avg_so_far:.2f}s)", flush=True)

    avg = sum(times) / len(times)
    print(f"Average time per matrix: {avg:.3f}s ({n} samples)", flush=True)
    return avg


def calibrate_adversarial_attacks(model, test_data, test_labels, weights_path,
                                  architecture_index, input_shape, num_classes,
                                  temp_dir, n_timing_samples=10):
    """
    Time each adversarial attack on a small sample and extrapolate total runtime.
    Returns (total_time_seconds, per_attack_times_dict).
    """
    from generate_adversarial_examples import apply_attack
    from pathlib import Path
    import tempfile

    total_test = len(test_data)
    attack_list = ["test"] + list(ATTACKS)
    per_attack_seconds = {}

    # Use a temporary path for attack outputs (we only care about timing)
    timing_dir = Path(temp_dir or tempfile.gettempdir()) / "calibration_attacks"

    # Take a small subset for timing
    n = min(n_timing_samples, total_test)
    timing_data = test_data[:n]
    timing_labels = test_labels[:n]

    print(f"  Timing {len(attack_list)} attacks on {n} samples...", flush=True)

    for attack_name in attack_list:
        # Use a fresh directory for each attack so apply_attack doesn't skip
        atk_dir = timing_dir / f"timing_{attack_name}"
        atk_dir.mkdir(parents=True, exist_ok=True)
        # Remove any prior timing artifacts
        save_path = atk_dir / f"{attack_name}/adversarial_examples.pth"
        if save_path.exists():
            save_path.unlink()

        start = time.time()
        try:
            apply_attack(
                attack_name=attack_name,
                data=timing_data,
                labels=timing_labels,
                weights_path=weights_path,
                architecture_index=architecture_index,
                path_adv_examples=atk_dir,
                input_shape=input_shape,
                num_classes=num_classes,
                batch_size=8,
            )
        except Exception as e:
            print(f"    {attack_name}: FAILED ({e})", flush=True)
            per_attack_seconds[attack_name] = 0.0
            continue
        elapsed = time.time() - start

        # Extrapolate to full test set
        estimated = (elapsed / n) * total_test
        per_attack_seconds[attack_name] = round(estimated, 2)
        print(f"    {attack_name}: {elapsed:.1f}s / {n} samples -> {estimated:.0f}s estimated", flush=True)

    total_time = sum(per_attack_seconds.values())

    # Check if trained weights exist — attacks on random weights converge
    # trivially fast, so timing must be penalized to avoid underestimation
    trained_weights_exist = any(
        f.startswith("epoch_") and f.endswith(".pth")
        for f in os.listdir(weights_path)
    ) if os.path.isdir(weights_path) else False

    random_penalty = 1.0
    if not trained_weights_exist:
        random_penalty = 10.0
        print(f"  WARNING: Using random weights — applying {random_penalty}x penalty to C estimate", flush=True)
        total_time *= random_penalty

    print(f"  Total estimated adversarial time: {total_time:.0f}s "
          f"({total_time/3600:.1f}h)", flush=True)

    return total_time, per_attack_seconds, trained_weights_exist, random_penalty


def main():
    args = parse_args()
    experiment = args.experiment_name

    if experiment not in DEFAULT_EXPERIMENTS:
        print(f"ERROR: '{experiment}' not found in DEFAULT_EXPERIMENTS")
        sys.exit(1)

    exp_config = DEFAULT_EXPERIMENTS[experiment]

    # Check for existing calibration
    calib_dir = f"experiments/{experiment}"
    calib_path = os.path.join(calib_dir, "calibration.json")

    if os.path.exists(calib_path) and not args.force:
        print(f"Calibration already exists: {calib_path}")
        print("Use --force to re-calibrate.")
        with open(calib_path) as f:
            calib = json.load(f)
        print(f"  batch_size: {calib['batch_size']}")
        print(f"  avg_seconds_per_matrix: {calib['avg_seconds_per_matrix']:.3f}")
        sys.exit(0)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Calibration requires a GPU.")
        sys.exit(1)

    device = get_device()
    dataset_name = exp_config['dataset']
    architecture_index = exp_config['architecture_index']
    total_epochs = exp_config['epochs']
    training_batch_size = exp_config['batch_size']
    input_shape = get_input_shape(dataset_name)
    num_classes = get_num_classes(dataset_name)

    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory

    print("=" * 60, flush=True)
    print(f"  GPU Calibration: {experiment}", flush=True)
    print(f"  GPU: {gpu_name} ({gpu_mem / 1e9:.1f} GB)", flush=True)
    print(f"  Architecture index: {architecture_index}", flush=True)
    print(f"  Dataset: {dataset_name}", flush=True)
    print(f"  Input shape: {input_shape}", flush=True)
    print("=" * 60, flush=True)

    # --- Load dataset ---
    print("\nLoading dataset...", flush=True)
    train_loader, _ = get_dataset(
        data_set=dataset_name,
        batch_size=training_batch_size,
        data_loader=True,
        data_path=args.temp_dir
    )
    train_set, _ = get_dataset(
        data_set=dataset_name,
        data_loader=False,
        data_path=args.temp_dir
    )
    sample_data, _ = subset(train_set, min(args.timing_samples + 10, len(train_set)), input_shape)

    # --- Create model with random weights ---
    # pretrained=False avoids internet downloads on compute nodes.
    # No state_dict loading — calibration measures memory/timing based on
    # architecture and tensor shapes, not weight values.
    print("\nCreating model...", flush=True)
    model = get_architecture(
        input_shape, num_classes, architecture_index,
        pretrained=False, freeze_features=False
    ).to(device)

    # Defensive: ensure input_shape is correct for KnowledgeMatrixComputer
    model.input_shape = input_shape
    print(f"Model input_shape: {model.input_shape}", flush=True)

    # --- Step 1: Train 2 epochs, measure time/memory ---
    print("\n--- Step 1: Training (2 epochs) ---", flush=True)
    train_time, train_peak_mem = train_2_epochs(model, train_loader, device)
    print(f"  2-epoch training time: {train_time:.1f}s", flush=True)
    print(f"  Peak training memory: {train_peak_mem / 1e9:.2f} GB", flush=True)

    # --- Step 2: Binary search for optimal batch_size ---
    print("\n--- Step 2: Batch size calibration ---", flush=True)
    model.eval()
    sample_input = sample_data[0].to(device)
    batch_size, peak_matrix_mem = find_optimal_batch_size(
        model, sample_input, device, args.target_utilization
    )

    # --- Step 3: Time matrix computation ---
    print("\n--- Step 3: Matrix computation timing ---", flush=True)
    avg_time = time_matrix_computation(
        model, sample_data, batch_size, device, args.timing_samples
    )

    # --- Step 4: Calibrate adversarial attack timing ---
    print("\n--- Step 4: Adversarial attack timing ---", flush=True)

    # Load test dataset for attack timing
    _, test_set = get_dataset(
        data_set=dataset_name,
        data_loader=False,
        data_path=args.temp_dir
    )
    test_data_subset, test_labels_subset = subset(
        test_set, min(len(test_set), 10000), input_shape
    )
    # Find weights path (random weights are fine — timing depends on architecture)
    weights_path = os.path.join(calib_dir, "weights")

    adv_total_time, per_attack_seconds, trained_weights_exist, random_penalty = calibrate_adversarial_attacks(
        model=model,
        test_data=test_data_subset,
        test_labels=test_labels_subset,
        weights_path=weights_path,
        architecture_index=architecture_index,
        input_shape=input_shape,
        num_classes=num_classes,
        temp_dir=args.temp_dir,
        n_timing_samples=10,
    )

    # --- Step 5: Estimate pipeline durations ---
    print("\n--- Step 5: Estimating pipeline durations ---", flush=True)

    num_attacks = len(ATTACKS) + 1  # +1 for "test"
    time_padding = 1.15      # +15% for A, B
    time_padding_adv = 3.0   # 3× for D (adversarial examples are much slower)
    mem_padding = 1.20       # +20%
    adv_grace_seconds = 1800  # 30 min grace for Step C

    # Training: extrapolate from 2 epochs
    est_train_time = (train_time / 2) * total_epochs * time_padding
    est_train_mem = int(train_peak_mem * mem_padding)

    # Step B: matrices per chunk
    est_B_per_chunk = avg_time * num_classes * args.num_samples_per_class / args.total_chunks * time_padding
    est_B_mem = int(peak_matrix_mem * mem_padding)

    # Step C: adversarial examples (calibrated from attack timing)
    est_C_time = adv_total_time * time_padding + adv_grace_seconds
    MIN_C_MEM = 64 * (1024 ** 3)  # 64 GB minimum — adversarial generation loads all attacks + stores examples
    est_C_mem = max(int(train_peak_mem * mem_padding), MIN_C_MEM)

    # Per-attack Slurm estimates (for parallelized Step C)
    per_attack_slurm = {}
    MIN_PER_ATTACK_SECONDS = 3600  # 1h floor per attack
    MIN_PER_ATTACK_MEM = 16 * (1024 ** 3)  # 16 GB floor per attack
    per_attack_mem = max(int(train_peak_mem * mem_padding), MIN_PER_ATTACK_MEM)
    for atk_name, atk_secs in per_attack_seconds.items():
        padded = atk_secs * random_penalty * time_padding + adv_grace_seconds
        padded = max(padded, MIN_PER_ATTACK_SECONDS)
        per_attack_slurm[atk_name] = {
            "time": seconds_to_slurm_time(padded),
            "time_seconds": round(padded, 2),
            "mem": bytes_to_slurm_mem(per_attack_mem),
            "mem_bytes": per_attack_mem,
        }

    # Step 3: adversarial matrices per chunk
    est_D_per_chunk = avg_time * num_attacks * args.samples_per_attack / args.total_chunks * time_padding_adv
    est_D_mem = int(peak_matrix_mem * mem_padding)

    slurm_resources = {
        "1": {
            "time": seconds_to_slurm_time(est_train_time),
            "mem": bytes_to_slurm_mem(est_train_mem),
            "time_seconds": est_train_time,
            "mem_bytes": est_train_mem,
        },
        "2a": {
            "time": seconds_to_slurm_time(est_B_per_chunk),
            "mem": bytes_to_slurm_mem(est_B_mem),
            "time_seconds": est_B_per_chunk,
            "mem_bytes": est_B_mem,
        },
        "2b": {
            "time": seconds_to_slurm_time(est_C_time),
            "mem": bytes_to_slurm_mem(est_C_mem),
            "time_seconds": est_C_time,
            "mem_bytes": est_C_mem,
            "per_attack_seconds": per_attack_seconds,
            "per_attack_slurm": per_attack_slurm,
            "trained_weights_exist": trained_weights_exist,
        },
        "3": {
            "time": seconds_to_slurm_time(est_D_per_chunk),
            "mem": bytes_to_slurm_mem(est_D_mem),
            "time_seconds": est_D_per_chunk,
            "mem_bytes": est_D_mem,
        },
    }

    print(f"\n  Step 1 (Training):       time={slurm_resources['1']['time']}, mem={slurm_resources['1']['mem']}", flush=True)
    print(f"  Step 2a (Matrices/chunk): time={slurm_resources['2a']['time']}, mem={slurm_resources['2a']['mem']}", flush=True)
    print(f"  Step 2b (AdvExamples):    time={slurm_resources['2b']['time']}, mem={slurm_resources['2b']['mem']}", flush=True)
    print(f"  Step 3 (AdvMats/chunk):  time={slurm_resources['3']['time']}, mem={slurm_resources['3']['mem']}", flush=True)

    # --- Save calibration.json ---
    calibration = {
        "experiment_name": experiment,
        "gpu_name": gpu_name,
        "gpu_memory_bytes": gpu_mem,
        "target_utilization": args.target_utilization,
        "batch_size": batch_size,
        "peak_matrix_memory_bytes": peak_matrix_mem,
        "peak_training_memory_bytes": train_peak_mem,
        "avg_seconds_per_matrix": round(avg_time, 4),
        "train_2epoch_seconds": round(train_time, 2),
        "timing_samples": args.timing_samples,
        "total_epochs": total_epochs,
        "num_classes": num_classes,
        "slurm_resources": slurm_resources,
        "calibration_params": {
            "total_chunks": args.total_chunks,
            "num_samples_per_class": args.num_samples_per_class,
            "samples_per_attack": args.samples_per_attack,
        },
        "input_shape": list(input_shape),
        "total_positions": input_shape[0] * input_shape[1] * input_shape[2],
        "architecture_index": architecture_index,
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(calib_dir, exist_ok=True)
    with open(calib_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nCalibration saved to: {calib_path}", flush=True)
    print("=" * 60, flush=True)
    print(f"  batch_size = {batch_size}", flush=True)
    print(f"  GPU utilization = {peak_matrix_mem / gpu_mem * 100:.1f}%", flush=True)
    print(f"  avg time/matrix = {avg_time:.3f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
