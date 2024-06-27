"""
    This script computes matrices for a subset of a dataset for a neural network trained with specific hyper parameters.
"""
import os
import argparse
from multiprocessing import Pool

from matrix_construction.matrix_construction import MatrixConstruction
from utils.utils import compute_train_statistics
from constants.constants import DEFAULT_EXPERIMENTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_index", type=int, default=0, help="The index for default experiment")
    parser.add_argument("--num_samples_per_class", type=int, default=4000,
                        help="Number of data samples per class to compute matrices.")
    parser.add_argument("--nb_workers", type=int, default=8, help="Number of threads for parallel computation")
    return parser.parse_args()


def compute_matrices(exp, chunk_id):
    exp.values_on_epoch(chunk_id=chunk_id)


def main():
    args = parse_args()
    if args.default_index is not None:
        try:
            experiment = DEFAULT_EXPERIMENTS[f'experiment_{args.default_index}']

            epoch = experiment['epoch'] - 1
            dataset = experiment['dataset']
            architecture_index = experiment['architecture_index']
            residual = experiment['residual']
            dropout = experiment['dropout']
            num_samples = args.num_samples_per_class

        except KeyError:
            print(f"Error: Default index {args.default_index} does not exist.")
            print(f"When computing matrices of new model add the experiment to constants.constants.py"
                  f" inside DEFAULT_EXPERIMENTS and provide the corresponding --default_index when running this script.")
            return -1
    else:
        raise ValueError("Default index not specified in constants/constants.py")

    chunk_size = num_samples // args.nb_workers

    weights_path = f'experiments/{args.default_index}/weights/'
    if not os.path.exists(weights_path):
        ValueError(f"Model needs to be trained first")

    save_path = f'experiments/{args.default_index}/matrices'

    dict_exp = {"epochs": epoch,
                "weights_path": weights_path,
                "save_path": save_path,
                "data_name": dataset,
                'num_samples': num_samples,
                'chunk_size': chunk_size,
                'architecture_index': architecture_index,
                'residual': residual,
                'dropout': dropout,
                }

    exp = MatrixConstruction(dict_exp)
    chunks = list(range(num_samples // chunk_size))
    arguments = list(zip([exp for _ in range(len(chunks))], chunks))

    print(f"Computing matrices...", flush=True)
    with Pool(processes=args.nb_workers) as pool:
        pool.starmap(compute_matrices, arguments)

    print('Computing matrix statistics', flush=True)
    compute_train_statistics(args.default_index)


if __name__ == '__main__':
    main()
