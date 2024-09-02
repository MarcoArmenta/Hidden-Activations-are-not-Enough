from utils.utils import get_model, subset, get_dataset, zip_and_cleanup
from constants.constants import DEFAULT_EXPERIMENTS
from matrix_construction.representation import MlpRepresentation
from pathlib import Path
import argparse
import torch
import os
from multiprocessing import Pool


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default_index",
        type=int,
        default=0,
        help="Index of default trained networks.",
    )
    parser.add_argument(
        "--num_samples_rejection_level",
        type=int,
        default=10000,
        help="Number of train samples to compute rejection level.",
    )
    parser.add_argument(
        "--nb_workers",
        type=int,
        default=8,
        help="How many processes in parallel for adversarial examples computations.",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Temporary directory to save and read data. Useful when using clusters."
    )
    return parser.parse_args()


def compute_one_matrix(args):
    im, label, weights_path, architecture_index, residual, input_shape, default_index, dropout, i, temp_dir = args

    model = get_model(weights_path, architecture_index, residual, input_shape, dropout)
    representation = MlpRepresentation(model)
    pred = torch.argmax(model.forward(im))
    if temp_dir is not None:
        path_experiment_matrix = Path(f'{temp_dir}/experiments/{default_index}/rejection_levels/matrices/{i}/matrix.pth')
    else:
        path_experiment_matrix = Path(f'experiments/{default_index}/rejection_levels/matrices/{i}/matrix.pth')
    # if it is not correctly classified, do not use it for rejection level
    if pred != label:
        return

    if os.path.exists(path_experiment_matrix):
        return
    mat = representation.forward(im)
    if temp_dir is not None:
        path_prediction = Path(f'{temp_dir}/experiments/{default_index}/rejection_levels/matrices/{i}/prediction.pth')
        Path(f'{temp_dir}/experiments/{default_index}/rejection_levels/matrices/{i}/').mkdir(parents=True, exist_ok=True)
    else:
        path_prediction = Path(f'experiments/{default_index}/rejection_levels/matrices/{i}/prediction.pth')
        Path(f'experiments/{default_index}/rejection_levels/matrices/{i}/').mkdir(parents=True, exist_ok=True)

    torch.save(pred, path_prediction)
    torch.save(mat, path_experiment_matrix)


def compute_matrices_for_rejection_level(exp_dataset_train: torch.Tensor,
                                         exp_dataset_labels: torch.Tensor,
                                         default_index,
                                         weights_path,
                                         architecture_index,
                                         residual,
                                         input_shape,
                                         dropout,
                                         nb_workers: int = 8,
                                         temp_dir=None) -> None:

    Path(f'experiments/{default_index}/rejection_levels/').mkdir(parents=True, exist_ok=True)

    print("Computing matrices for rejection level...", flush=True)

    with Pool(processes=nb_workers) as pool:
        args = [(exp_dataset_train[i],
                 exp_dataset_labels[i],
                 weights_path,
                 architecture_index,
                 residual,
                 input_shape,
                 default_index,
                 dropout,
                 i,
                 temp_dir) for i in range(len(exp_dataset_train))]

        pool.map(compute_one_matrix, args)


def main():
    args = parse_args()
    if args.default_index is not None:
        try:
            experiment = DEFAULT_EXPERIMENTS[f'experiment_{args.default_index}']

            architecture_index = experiment['architecture_index']
            residual = experiment['residual']
            dropout = experiment['dropout']
            dataset = experiment['dataset']
            epoch = experiment['epoch'] - 1

        except KeyError:
            print(f"Error: Default index {args.default_index} does not exist.")
            return -1

    else:
        raise ValueError("Default index not specified in constants/constants.py")

    print("Computing matrices for rejection level for Experiment: ", args.default_index,flush=True)

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{args.default_index}/weights') / f'epoch_{epoch}.pth'
    else:
        weights_path = Path(f'experiments/{args.default_index}/weights') / f'epoch_{epoch}.pth'

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    input_shape = (3, 32, 32) if dataset == 'cifar10' else (1, 28, 28)
    train_set, test_set = get_dataset(dataset, data_loader=False)
    exp_dataset_train, exp_dataset_labels = subset(train_set, args.num_samples_rejection_level, input_shape=input_shape)

    Path(f'experiments/{args.default_index}/rejection_levels/').mkdir(parents=True, exist_ok=True)
    torch.save(exp_dataset_train, f'experiments/{args.default_index}/rejection_levels/exp_dataset_train.pth')

    compute_matrices_for_rejection_level(exp_dataset_train,
                                         exp_dataset_labels,
                                         args.default_index,
                                         weights_path,
                                         architecture_index,
                                         residual,
                                         input_shape,
                                         dropout,
                                         args.nb_workers,
                                         args.temp_dir)

    if args.temp_dir is not None:
        zip_and_cleanup(f'{args.temp_dir}/experiments/{args.default_index}/rejection_levels/matrices/',
                        f'experiments/{args.default_index}/rejection_levels/matrices/matrices',
                        clean=False)


if __name__ == '__main__':
    main()
