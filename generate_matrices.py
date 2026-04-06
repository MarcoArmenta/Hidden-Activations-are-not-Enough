"""
    This script computes matrices for a subset of a dataset for a neural network trained with specific hyper parameters.
"""
import os
import time
import torch
from argparse import ArgumentParser, Namespace

from matrix_construction.parallel import ParallelMatrixConstruction
from constants.constants import DEFAULT_EXPERIMENTS
from utils.utils import get_device


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", "--experiment", type=str, default=None, dest="experiment_name", help="The index for default experiment")
    parser.add_argument("--num_samples_per_class", type=int, default=1000, help="Number of data samples per class")
    parser.add_argument("--temp_dir", default=None, type=str, help="Temporary directory for data")
    parser.add_argument("--chunk_id", type=int, default=0, help="Chunk ID to process (set by Slurm task ID)")
    parser.add_argument("--total_chunks", type=int, default=4, help="Total number of chunks")
    parser.add_argument("--batch_size", type=int, default=3072, help="Number of columns of knowledge matrix to process at a time per GPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment_name is None:
        raise ValueError("Default index not specified in constants/constants.py")

    experiment = args.experiment_name

    dataset = DEFAULT_EXPERIMENTS[experiment]['dataset']
    epochs = DEFAULT_EXPERIMENTS[experiment]['epochs'] if dataset != 'imagenet' else None
    architecture_index = DEFAULT_EXPERIMENTS[experiment]['architecture_index']
    num_samples = args.num_samples_per_class

    chunk_id = int(os.getenv('SLURM_ARRAY_TASK_ID', args.chunk_id))
    if chunk_id is None:
        raise ValueError("chunk_id must be provided or set via SLURM_ARRAY_TASK_ID")

    # Remainder-aware chunking (same pattern as generate_adversarial_matrices.py)
    N = num_samples
    base_chunk = N // args.total_chunks
    remainder = N % args.total_chunks
    if chunk_id < remainder:
        start_idx = chunk_id * (base_chunk + 1)
        end_idx = start_idx + (base_chunk + 1)
    else:
        start_idx = chunk_id * base_chunk + remainder
        end_idx = start_idx + base_chunk
    chunk_size = end_idx - start_idx

    if args.temp_dir is not None:
        weights_path = f'{args.temp_dir}/experiments/{experiment}/weights/'
        save_path = f'{args.temp_dir}/experiments/{experiment}/matrices'
    else:
        weights_path = f'experiments/{experiment}/weights/'
        save_path = f'experiments/{experiment}/matrices'

    if not os.path.exists(weights_path):
        raise ValueError(f"Model needs to be trained first")

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    gpu_id = chunk_id % max(1, gpu_count)  # Cycle through available GPUs
    torch.cuda.set_device(gpu_id)

    dict_exp = {
        "epochs": epochs,
        "weights_path": weights_path,
        "save_path": save_path,
        "data_name": dataset,
        'num_samples': num_samples,
        'chunk_size': chunk_size,
        'architecture_index': architecture_index,
        'batch_size': args.batch_size,
        'start_idx': start_idx,
        'device': get_device(chunk_id, torch.cuda.device_count())
    }

    print(f"Processing chunk {chunk_id} on GPU {gpu_id}", flush=True)

    t_start = time.perf_counter()
    mat_constructer = ParallelMatrixConstruction(dict_exp)
    success = mat_constructer.values_on_epoch(chunk_id=chunk_id)
    t_elapsed = time.perf_counter() - t_start
    print(f"Chunk {chunk_id} wall-clock time: {t_elapsed:.1f}s ({t_elapsed/3600:.2f}h)", flush=True)

    if success:
        done_file = os.path.join(save_path, f"done_chunk_{chunk_id}.txt")
        with open(done_file, 'w') as f:
            f.write("done")

        # Count generated .pth files as a sanity check
        pth_count = sum(1 for _, _, files in os.walk(save_path) for f in files if f.endswith(('.pth', '.pt')))
        print(f"Chunk {chunk_id} completed and saved to {save_path} ({pth_count} .pth files)", flush=True)
        print(f"Chunk {chunk_id} completed!", flush=True)
    else:
        print(f'An error has occurred at chunk_id = {chunk_id}')


if __name__ == '__main__':
    main()