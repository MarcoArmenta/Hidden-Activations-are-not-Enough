from utils.utils import get_ellipsoid_data, zero_std, get_model, subset, get_dataset
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS
from matrix_construction.representation import MlpRepresentation
from pathlib import Path
import argparse
import torch
import json
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
        "--std",
        type=float,
        default=1,
        help="This times the standard deviation gives a margin for rejection level.",
    )
    parser.add_argument(
        "--d1",
        type=float,
        default=0.1,
        help="Determines how small should the standard deviation be per coordinate on matrix statistics.",
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

    return parser.parse_args()


def process_sample(args):
    im, label, weights_path, architecture_index, residual, input_shape, ellipsoids, d1, default_index, dropout, i = args
    path_experiment_matrix = Path(f'experiments/{default_index}/matrices/zero_counting/{i}/matrix.pth')
    Path(f'experiments/{default_index}/matrices/zero_counting/{i}/').mkdir(parents=True, exist_ok=True)

    model = get_model(weights_path, architecture_index, residual, input_shape, dropout)
    representation = MlpRepresentation(model)
    pred = torch.argmax(model.forward(im))
    if pred != label:
        return None

    if os.path.exists(path_experiment_matrix):
        mat = torch.load(path_experiment_matrix)
    else:
        mat = representation.forward(im)
        torch.save(mat, path_experiment_matrix)

    a = get_ellipsoid_data(ellipsoids, pred, "std")
    b = zero_std(mat, a, d1)
    c = b.expand([1])
    return c


def compute_rejection_level(exp_dataset_train: torch.Tensor,
                             exp_dataset_labels: torch.Tensor,
                             default_index,
                             weights_path,
                             architecture_index,
                             residual,
                             input_shape,
                             dropout,
                             ellipsoids: dict,
                             std: float = 2,
                             d1: float = 0.1,
                             nb_workers: int = 8) -> None:

    # Compute mean and std of number of (almost) zero dims
    reject_path = f'experiments/{default_index}/rejection_levels/reject_at_{std}_{d1}.json'
    Path(f'experiments/{default_index}/rejection_levels/').mkdir(parents=True, exist_ok=True)

    print("Computing rejection level...", flush=True)

    with Pool(processes=nb_workers) as pool:
        args = [(exp_dataset_train[i],
                 exp_dataset_labels[i],
                 weights_path,
                 architecture_index,
                 residual,
                 input_shape,
                 ellipsoids,
                 d1,
                 default_index,
                 dropout,
                 i) for i in range(len(exp_dataset_train))]
        results = pool.map(process_sample, args)

    zeros = torch.cat([result for result in results if result is not None]).float()

    reject_at = zeros.mean().item() - std*zeros.std().item()

    with open(reject_path, 'w') as json_file:
        json.dump([reject_at], json_file, indent=4)


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

    print("Experiment: ", args.default_index)

    weights_path = Path(f'experiments/{args.default_index}/weights') / f'epoch_{epoch}.pth'
    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    matrices_path = Path(f'experiments/{args.default_index}/matrices/matrix_statistics.json')
    if not matrices_path.exists():
        raise ValueError(f"Matrix statistics have to be computed")

    input_shape = (3, 32, 32) if dataset == 'cifar10' else (1, 28, 28)
    train_set, test_set = get_dataset(dataset, data_loader=False)
    exp_dataset_train, exp_dataset_labels = subset(train_set, args.num_samples_rejection_level, input_shape=input_shape)

    ellipsoids_file = open(f"experiments/{args.default_index}/matrices/matrix_statistics.json")
    ellipsoids = json.load(ellipsoids_file)

    compute_rejection_level(exp_dataset_train,
                            exp_dataset_labels,
                            args.default_index,
                            weights_path,
                            architecture_index,
                            residual,
                            input_shape,
                            dropout,
                            ellipsoids,
                            args.std,
                            args.d1,
                            args.nb_workers)


if __name__ == '__main__':
    main()
