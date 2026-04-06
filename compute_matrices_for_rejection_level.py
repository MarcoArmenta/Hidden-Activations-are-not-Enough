import os
import torch
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from constants.constants import DEFAULT_EXPERIMENTS
from utils.utils import get_model, subset, get_dataset, get_num_classes, get_input_shape, get_device


def parse_args(
        parser:Union[ArgumentParser, None] = None
    ) -> Namespace:
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type = str,
        default = None,
        help = "Name of experiment. Check constants/constants.py.",
    )
    parser.add_argument(
        "--num_samples_rejection_level",
        type = int,
        default = 10000,
        help = "Number of train samples to compute rejection level.",
    )
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 18816, # for h100 gpu alexnet 3x224x224
        help = "Batch size used by matrix computer."
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
        help = "Chunk id for this worker (0-based)."
    )
    parser.add_argument(
        "--total_chunks",
        type = int,
        default = 4,
        help = "Total number of chunks / workers."
    )

    return parser.parse_args()


def compute_one_matrix(args: tuple) -> None:
    (
        im,
        label,
        experiment_name,
        i,
        temp_dir,
        batch_size,
        matrix_computer,
        pred
    ) = args

    if temp_dir is not None:
        path_experiment_matrix = Path(f'{temp_dir}/experiments/{experiment_name}/rejection_levels/matrices/{i}/matrix.pth')
        path_prediction = Path(f'{temp_dir}/experiments/{experiment_name}/rejection_levels/matrices/{i}/prediction.pth')
        Path(f'{temp_dir}/experiments/{experiment_name}/rejection_levels/matrices/{i}/').mkdir(parents=True, exist_ok=True)
    else:
        path_experiment_matrix = Path(f'experiments/{experiment_name}/rejection_levels/matrices/{i}/matrix.pth')
        path_prediction = Path(f'experiments/{experiment_name}/rejection_levels/matrices/{i}/prediction.pth')
        Path(f'experiments/{experiment_name}/rejection_levels/matrices/{i}/').mkdir(parents=True, exist_ok=True)

    # if it is not correctly classified, do not use it for rejection level
    if pred != label:
        return

    if os.path.exists(path_experiment_matrix):
        return

    mat = matrix_computer.forward(im)

    torch.save(pred.cpu().detach(), path_prediction)
    torch.save(mat.cpu().detach(), path_experiment_matrix)
    del mat
    torch.cuda.empty_cache()


def compute_matrices_for_rejection_level(
        exp_dataset_train: torch.Tensor,
        exp_dataset_labels: torch.Tensor,
        experiment_name: str,
        weights_path: Path,
        architecture_index: int,
        input_shape,
        num_classes: int,
        temp_dir:Union[str, None] = None,
        device: torch.device = 'cpu',
        batch_size: int = 18816,
        chunk_id: int = 0,
        total_chunks: int = 4,
    ) -> None:

    Path(f'experiments/{experiment_name}/rejection_levels/').mkdir(parents=True, exist_ok=True)
    print(f"Computing matrices for rejection level... chunk {chunk_id}/{total_chunks}", flush=True)

    # Ensure model is loaded onto correct device
    model = get_model(path=weights_path,
                      architecture_index=architecture_index,
                      input_shape=input_shape,
                      num_classes=num_classes,
                      device=device)

    matrix_computer = KnowledgeMatrixComputer(model, batch_size=batch_size, device=device)

    #data = data[chunk_id * chunk_size:(chunk_id + 1) * chunk_size].to(device)
    # Compute chunk boundaries
    N = len(exp_dataset_train)
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
            im = exp_dataset_train[i].unsqueeze(0).to(device)
            with torch.no_grad():
                pred = torch.argmax(model.forward(im)).cpu()

            args = (exp_dataset_train[i].to(device),
                    exp_dataset_labels[i].to(device),
                    experiment_name,
                    i,
                    temp_dir,
                    batch_size,
                    matrix_computer,
                    pred
                    )
            print(f'Chunk {chunk_id} - Matrix {i}/{N}', flush=True)
            compute_one_matrix(args)
        except Exception as e:
            failed_indices.append(i)
            print(f'ERROR: Chunk {chunk_id} - Matrix {i}/{N} FAILED: {type(e).__name__}: {e}', flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    if failed_indices:
        print(f'WARNING: Chunk {chunk_id} had {len(failed_indices)} failed matrices: {failed_indices}', flush=True)


def main() -> None:
    args = parse_args()
    if args.experiment_name is not None:
        experiment = DEFAULT_EXPERIMENTS[f'{args.experiment_name}']
        architecture_index = experiment['architecture_index']
        dataset = experiment['dataset']
        epoch = experiment['epochs']
    else:
        raise ValueError("Default index not specified in constants/constants.py")

    print("Computing matrices for rejection level for Experiment: ", args.experiment_name, flush=True)

    # Use get_device() which may respect CUDA_VISIBLE_DEVICES
    device = get_device()

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{args.experiment_name}/weights') / f'epoch_{epoch}.pth'
    else:
        weights_path = Path(f'experiments/{args.experiment_name}/weights') / f'epoch_{epoch}.pth'

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained; missing weights at {weights_path}")

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)

    Path(f'experiments/{args.experiment_name}/rejection_levels/').mkdir(parents=True, exist_ok=True)
    if args.temp_dir is not None:
        rej_dir = Path(f'{args.temp_dir}/experiments/{args.experiment_name}/rejection_levels/')
    else:
        rej_dir = Path(f'experiments/{args.experiment_name}/rejection_levels/')
    rej_dir.mkdir(parents=True, exist_ok=True)
    exp_dataset_train_file = rej_dir / 'exp_dataset_train.pth'
    exp_dataset_labels_file = rej_dir / 'exp_dataset_labels.pth'

    if exp_dataset_train_file.exists() and exp_dataset_labels_file.exists():
        exp_dataset_train = torch.load(exp_dataset_train_file)
        exp_dataset_labels = torch.load(exp_dataset_labels_file)
    else:
        train_set, _ = get_dataset(
            data_set=dataset,
            data_loader=False
        )
        exp_dataset_train, exp_dataset_labels = subset(
            train_set = train_set,
            length = args.num_samples_rejection_level,
            input_shape = input_shape
        )

        torch.save(obj=exp_dataset_train, f=str(exp_dataset_train_file))
        torch.save(obj=exp_dataset_labels, f=str(exp_dataset_labels_file))

    compute_matrices_for_rejection_level(
        exp_dataset_train = exp_dataset_train,
        exp_dataset_labels = exp_dataset_labels,
        experiment_name = args.experiment_name,
        weights_path = weights_path,
        architecture_index = architecture_index,
        input_shape = input_shape,
        num_classes = num_classes,
        temp_dir = args.temp_dir,
        device=device,
        batch_size = args.batch_size,
        chunk_id = args.chunk_id,
        total_chunks = args.total_chunks,
    )
    # Count generated .pth files as a sanity check
    if args.temp_dir is not None:
        rej_mat_dir = os.path.join(args.temp_dir, 'experiments', args.experiment_name, 'rejection_levels', 'matrices')
    else:
        rej_mat_dir = os.path.join('experiments', args.experiment_name, 'rejection_levels', 'matrices')
    pth_count = sum(1 for _, _, files in os.walk(rej_mat_dir) for f in files if f.endswith(('.pth', '.pt'))) if os.path.isdir(rej_mat_dir) else 0
    print(f"---ALL MATRICES COMPUTED FOR THIS WORKER ({pth_count} .pth files)----", flush=True)


if __name__ == '__main__':
    main()
