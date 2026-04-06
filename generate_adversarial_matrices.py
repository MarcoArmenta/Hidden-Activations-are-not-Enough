import gc
import os
import sys
import time
import torch
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

from utils.utils import get_model, get_num_classes, get_input_shape, get_device
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS
from utils.atomic_io import atomic_torch_save


def parse_args(
        parser:Union[ArgumentParser, None] = None
    ) -> Namespace:
    """
        Args:
            parser: the parser to use.
        Returns:
            The parsed arguments.
    """
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        "--experiment_name", "--experiment",
        type = str,
        default = None,
        dest = "experiment_name",
        help = "Name of experiment."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=18816,
        help="Number of colums in matrix to process at the same time."
    )
    parser.add_argument(
        "--temp_dir",
        type = str,
        default = None,
        help = "Temporary directory to save and read data. Useful when using clusters."
    )
    parser.add_argument(
        "--chunk_id",
        type = int,
        default = 0,
        help = "Current chunk id or slurm task id"
    )
    parser.add_argument(
        "--total_chunks",
        type = int,
        default = 4,
        help = "Temporary directory to save and read data. Useful when using clusters."
    )
    parser.add_argument(
        "--samples_per_attack",
        type=int,
        default=200,
        help="Number of adversarial examples per attack."
    )

    return parser.parse_args()

def save_one_matrix_with_retry(im, model, matrix_computer, device, max_retries=3):
    """Compute a single matrix with OOM retry logic.

    On OOM, destroys and recreates the KnowledgeMatrixComputer with halved
    batch_size (matching the proven pattern from matrix_construction/parallel.py).

    Args:
        im: the image tensor (3D: C, H, W).
        model: the neural network model (needed to recreate matrix_computer).
        matrix_computer: KnowledgeMatrixComputer instance.
        device: torch device.
        max_retries: number of retries on OOM.

    Returns:
        Tuple of (computed matrix tensor, matrix_computer) — the computer may
        have been recreated with a smaller batch_size.
    """
    for attempt in range(max_retries + 1):
        try:
            mat = matrix_computer.forward(im.to(device))
            return mat, matrix_computer
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and attempt < max_retries:
                old_bs = matrix_computer.batch_size
                new_bs = max(1, old_bs // 2)
                print(f"    OOM on attempt {attempt+1}, halving batch_size {old_bs}->{new_bs} and retrying...", flush=True)
                del matrix_computer
                gc.collect()
                torch.cuda.empty_cache()
                matrix_computer = KnowledgeMatrixComputer(model, batch_size=new_bs, device=device)
                continue
            raise


def save_one_matrix(
        im: torch.Tensor,
        attack: str,
        i: int,
        experiment_name: str,
        model,
        matrix_computer,
        temp_dir: Union[str, None],
        device
    ):
    """
        Args:
            im: the image to save the matrix of.
            attack: the attack to save the matrix of.
            i: the index of the image.
            experiment_name: the name of the experiment.
            model: the neural network model (needed for OOM retry recreation).
            matrix_computer: KnowledgeMatrixComputer instance.
            temp_dir: temporary directory path.
            device: torch device.

        Returns:
            The (possibly recreated) matrix_computer.
    """

    if temp_dir is not None:
        matrix_save_path = Path(f'{temp_dir}/experiments/{experiment_name}/adversarial_matrices') / f'{attack}' / f'{i}/matrix.pth'
    else:
        matrix_save_path = Path(f'experiments/{experiment_name}/adversarial_matrices') / f'{attack}' / f'{i}/matrix.pth'

    if not matrix_save_path.exists():
        mat, matrix_computer = save_one_matrix_with_retry(im, model, matrix_computer, device)
        matrix_save_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(mat.cpu(), matrix_save_path)
        del mat
        torch.cuda.empty_cache()

    return matrix_computer

def generate_matrices_for_attacks(
        experiment_name: str,
        temp_dir: Union[str, None],
        weights_path: Path,
        architecture_index: int,
        input_shape,
        num_classes: int,
        device,
        batch_size: int = 18816,
        chunk_id: int = 0,
        total_chunks: int = 4,
        samples_per_attack: int = 200,
    ) -> int:
    """
        Calls the save_one_matrix function for each adversarial example.

        Args:
            default_index: the index of the default experiment (See constants/constants.py).
            temp_dir: the temporary directory to save the matrices.
            weights_path: the path to the weights.
            architecture_index: the index of the architecture (See constants/constants.py).
            residual: whether the model has residual connections.
            input_shape: the shape of the input.
            dropout: whether the model has dropout layers.
            nb_workers: the number of workers.
    """
    model = get_model(
        path=weights_path,
        architecture_index=architecture_index,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device
    )
    matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)
    attacks_processed = []
    total_saved = 0
    total_failed = 0
    for attack in ['test'] + ATTACKS:
        if temp_dir is not None:
            path_adv_examples = Path(temp_dir) / f'experiments/{experiment_name}/adversarial_examples' / f"{attack}/adversarial_examples.pth"
        else:
            path_adv_examples = Path(f'experiments/{experiment_name}/adversarial_examples') / f"{attack}/adversarial_examples.pth"
        if not path_adv_examples.exists():
            print(f'WARNING: Attack "{attack}" adversarial examples not found at '
                  f'{path_adv_examples}. Skipping.', flush=True)
            continue
        attacked_dataset = torch.load(path_adv_examples, weights_only=True)[:samples_per_attack]

        print(f"Generating matrices for attack {attack}.", flush=True)

        N = attacked_dataset.shape[0]
        base_chunk = N // total_chunks
        remainder = N % total_chunks
        # distribute the remainder among the first `remainder` chunks
        if chunk_id < remainder:
            start = chunk_id * (base_chunk + 1)
            end = start + (base_chunk + 1)
        else:
            start = chunk_id * base_chunk + remainder
            end = start + base_chunk

        # Bound check
        start = max(0, start)
        end = min(N, end)

        print(f"Worker chunk_id={chunk_id} handling indices [{start}, {end}) out of {N}", flush=True)


        # iterate only over the slice for this chunk
        model.eval()
        failed_indices = []
        for i in range(start, end):
            try:
                print(f'Chunk {chunk_id} - Matrix {i}/{N}', flush=True)
                matrix_computer = save_one_matrix(attacked_dataset[i].to(device),
                                attack,
                                i,
                                experiment_name,
                                model,
                                matrix_computer,
                                temp_dir,
                                device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    failed_indices.append(i)
                    print(f'ERROR: CUDA OOM - Chunk {chunk_id} - Attack {attack} - Matrix {i}/{N}: {e}', flush=True)
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                    continue
                raise
            except Exception as e:
                failed_indices.append(i)
                print(f'ERROR: Chunk {chunk_id} - Attack {attack} - Matrix {i}/{N} FAILED: {type(e).__name__}: {e}', flush=True)
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                continue

        if failed_indices:
            print(f'WARNING: Chunk {chunk_id} - Attack {attack} had {len(failed_indices)} failed matrices: {failed_indices}', flush=True)

        attacks_processed.append(attack)
        total_failed += len(failed_indices)
        total_saved += (end - start) - len(failed_indices)

    # Write done_file checkpoint
    if temp_dir is not None:
        save_path = os.path.join(temp_dir, 'experiments', experiment_name, 'adversarial_matrices')
    else:
        save_path = os.path.join('experiments', experiment_name, 'adversarial_matrices')
    os.makedirs(save_path, exist_ok=True)
    done_file = os.path.join(save_path, f"done_advmat_chunk_{chunk_id}.txt")
    with open(done_file, 'w') as f:
        f.write(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"attacks_processed: {len(attacks_processed)}\n")
        f.write(f"total_matrices: {total_saved}\n")
        f.write(f"total_failed: {total_failed}\n")

    return total_failed

def main() -> None:
    """
        Main function to generate adversarial matrices.
    """
    args = parse_args()
    if args.experiment_name is not None:
        experiment = DEFAULT_EXPERIMENTS[f'{args.experiment_name}']
        architecture_index = experiment['architecture_index']
        dataset = experiment['dataset']
        epoch = experiment['epochs']

    else:
        raise ValueError("Experiment not specified. Use --experiment_name")

    print("Experiment: ", args.experiment_name, flush=True)

    t_start = time.perf_counter()

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{args.experiment_name}/weights/epoch_{epoch}.pth')
    else:
        weights_path = Path(f'experiments/{args.experiment_name}/weights/epoch_{epoch}.pth')

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)
    device = get_device()

    total_failed = generate_matrices_for_attacks(
        experiment_name = args.experiment_name,
        temp_dir = args.temp_dir,
        weights_path = weights_path,
        architecture_index = architecture_index,
        input_shape = input_shape,
        num_classes = num_classes,
        batch_size = args.batch_size,
        device=device,
        chunk_id=args.chunk_id,
        total_chunks=args.total_chunks,
        samples_per_attack=args.samples_per_attack,
    )

    # Count generated .pth files as a sanity check
    if args.temp_dir is not None:
        adv_mat_dir = os.path.join(args.temp_dir, 'experiments', args.experiment_name, 'adversarial_matrices')
    else:
        adv_mat_dir = os.path.join('experiments', args.experiment_name, 'adversarial_matrices')
    pth_count = sum(1 for _, _, files in os.walk(adv_mat_dir) for f in files if f.endswith(('.pth', '.pt'))) if os.path.isdir(adv_mat_dir) else 0
    t_elapsed = time.perf_counter() - t_start
    print(f"----CHUNK {args.chunk_id} ADVERSARIAL MATRICES COMPUTED ({pth_count} .pth files, {t_elapsed:.1f}s / {t_elapsed/3600:.2f}h)----", flush=True)

    if total_failed > 0:
        print(f"ERROR: {total_failed} total matrices failed due to CUDA OOM or other errors", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
