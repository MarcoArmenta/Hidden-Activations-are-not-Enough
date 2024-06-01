"""
    This script computes matrices for a subset of a dataset for a neural network trained with specific hyperparameters.
"""
import os
import argparse
from multiprocessing import Pool

from matrix_construction.matrix_construction import MatrixConstruction
from utils.utils import compute_train_statistics
from __init__ import DEFAULT_TRAININGS


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="The datasets to train the model on.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="Optimizer to train the model with.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="The learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default=8,
        help="The batch size.",
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default=0,
        help="The epoch at which to compute matrices.",
    )
    parser.add_argument(
        "--default_hyper_parameters",
        action='store_true',
        help="If not called, computes matrices on a default network from:"
             f"{DEFAULT_TRAININGS}",
    )
    parser.add_argument(
        "--default_index",
        type=int,
        default=0,
        help="Index of default trained networks.",
    )
    parser.add_argument(
        "--num_samples_per_class",
        type=int,
        default=500,
        help="Number of data samples per class to compute matrices.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Number of matrices to compute at a time for parallelization",
    )
    parser.add_argument(
        "--nb_workers",
        type=int,
        default=8,
        help="Number of threads for parallel computation",
    )

    return parser.parse_args()


def compute_matrices(exp, chunk_id):
    exp.values_on_epoch(chunk_id=chunk_id,
                        train=True)


if __name__ == '__main__':
    device = "cpu"

    args = parse_args()
    num_samples = args.num_samples_per_class
    chunk_size = args.chunk_size

    if args.default_hyper_parameters:
        index = args.default_index
        print(f"Loading default experiment {index}.")
        experiment = DEFAULT_TRAININGS[f'experiment_{index}']

        optimizer_name = experiment['optimizer']
        dataset = experiment['dataset']
        lr = experiment['lr']
        batch_size = experiment['batch_size']
        epoch = experiment['epoch']-1

    else:
        print("Loading")
        optimizer_name = args.optimizer
        dataset = args.dataset
        lr = args.lr
        batch_size = args.batch_size
        epoch = args.epoch-1

    weights_path = f'experiments/weights/{dataset}/{optimizer_name}/{lr}/{batch_size}/'

    if not os.path.exists(weights_path + f'epoch_{epoch}.pth'):
        ValueError(f"Experiment needs to be trained with hyperparameters: {weights_path}")

    save_path = f'experiments/matrices/{dataset}/{optimizer_name}/{lr}/{batch_size}/'

    dict_exp = {"epochs": epoch,
                "weights_path": weights_path,
                "save_path": save_path,
                "device": device,
                "data_name": dataset,
                'num_samples': num_samples,
                'chunk_size': chunk_size,
                }

    exp = MatrixConstruction(dict_exp)
    chunks = list(range(num_samples//chunk_size))
    arguments = list(zip([exp for _ in range(len(chunks))], chunks))

    print(f"Computing matrices...", flush=True)
    with Pool(processes=args.nb_workers) as pool:
        results = pool.starmap(compute_matrices, arguments)

    print('Computing matrix statistics', flush=True)
    compute_train_statistics(dataset, optimizer_name, lr, batch_size, epoch)
