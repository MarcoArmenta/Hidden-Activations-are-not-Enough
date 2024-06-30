from utils.utils import get_ellipsoid_data, zero_std, get_model
from constants.constants import DEFAULT_EXPERIMENTS
from pathlib import Path
import argparse
import torch
import json
import os


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

    return parser.parse_args()


def process_sample(ellipsoids, d1, default_index, i):
    path_experiment_matrix = Path(f'experiments/{default_index}/rejection_levels/matrices/{i}/matrix.pth')
    path_prediction = Path(f'experiments/{default_index}/rejection_levels/matrices/{i}/prediction.pth')

    if os.path.exists(path_experiment_matrix):
        mat = torch.load(path_experiment_matrix)
        pred = torch.load(path_prediction)
        a = get_ellipsoid_data(ellipsoids, pred, "std")
        b = zero_std(mat, a, d1)
        c = b.expand([1])
        return c
    else:
        return None


def compute_rejection_level(exp_dataset_train: torch.Tensor,
                             default_index,
                             ellipsoids: dict,
                             std: float = 2,
                             d1: float = 0.1) -> None:

    # Compute mean and std of number of (almost) zero dims
    reject_path = f'experiments/{default_index}/rejection_levels/reject_at_{std}_{d1}.json'
    Path(f'experiments/{default_index}/rejection_levels/').mkdir(parents=True, exist_ok=True)

    print("Computing rejection level...", flush=True)

    results = []
    for i in range(len(exp_dataset_train)):
        results.append(process_sample(ellipsoids, d1, default_index, i))

    zeros = torch.cat([result for result in results if result is not None]).float()

    reject_at = zeros.mean().item() - std*zeros.std().item()

    print(f"Rejection level: {reject_at}", flush=True)

    with open(reject_path, 'w') as json_file:
        json.dump([reject_at], json_file, indent=4)


def main():
    args = parse_args()
    if args.default_index is not None:
        try:
            experiment = DEFAULT_EXPERIMENTS[f'experiment_{args.default_index}']
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

    exp_dataset_train = torch.load(f'experiments/{args.default_index}/rejection_levels/exp_dataset_train.pth')
    ellipsoids_file = open(f"experiments/{args.default_index}/matrices/matrix_statistics.json")
    ellipsoids = json.load(ellipsoids_file)

    compute_rejection_level(exp_dataset_train,
                            args.default_index,
                            ellipsoids,
                            args.std,
                            args.d1)


if __name__ == '__main__':
    main()
