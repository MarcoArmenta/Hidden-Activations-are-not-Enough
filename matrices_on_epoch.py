"""
    This script computes matrices for a subset of a dataset for a neural network trained with specific hyper parameters.
"""
import os
import argparse
from multiprocessing import Pool

from matrix_construction.matrix_construction import MatrixConstruction
from utils.utils import compute_train_statistics
from constants.constants import DEFAULT_EXPERIMENTS
from utils.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_index", type=int, default=0, help="The index for default experiment")
    parser.add_argument("--num_samples_per_class", type=int, default=1000, help="Number of data samples per class to compute matrices.")
    parser.add_argument("--nb_workers", type=int, default=2, help="Number of threads for parallel computation")
    return parser.parse_args()


def compute_matrices(exp, chunk_id):
    exp.values_on_epoch(chunk_id=chunk_id,
                        train=True)


def main():
    args = parse_args()
    if args.default_index is not None:
        try:
            experiment = DEFAULT_EXPERIMENTS[f'experiment_{args.default_index}']

            architecture_index = experiment['architecture_index']
            dataset = experiment['dataset']
            optimizer_name = experiment['optimizer']
            lr = experiment['lr']
            batch_size = experiment['batch_size']
            epoch = experiment['epoch'] - 1
            residual = experiment['residual']
            num_samples = args.num_samples_per_class

        except KeyError:
            print(f"Error: Default index {args.default_index} does not exist.")
            print(f"When computing matrices of new model, add the experiment to constants.constants.py inside DEFAULT_EXPERIMENTS"
                  f"and provide the corresponding --default_index when running this script.")
            return

    #device = get_device()
    device = 'cpu'
    chunk_size = num_samples // args.nb_workers
    experiment_path = f'{dataset}/{architecture_index}/{optimizer_name}/{lr}/{batch_size}'

    weights_path = f'experiments/weights/{experiment_path}'

    if not os.path.exists(weights_path + f'/epoch_{epoch}.pth'):
        ValueError(f"Experiment needs to be trained with hyper-parameters: {experiment_path}")

    save_path = f'experiments/matrices/{experiment_path}'

    dict_exp = {"epochs": epoch,
                "weights_path": weights_path,
                "save_path": save_path,
                "device": device,
                "data_name": dataset,
                'num_samples': num_samples,
                'chunk_size': chunk_size,
                'architecture_index': architecture_index,
                'residual': residual
                }

    exp = MatrixConstruction(dict_exp)
    chunks = list(range(num_samples // chunk_size))
    arguments = list(zip([exp for _ in range(len(chunks))], chunks))

    print(f"Computing matrices...", flush=True)
    with Pool(processes=args.nb_workers) as pool:
        results = pool.starmap(compute_matrices, arguments)

    print('Computing matrix statistics', flush=True)
    compute_train_statistics(dataset, optimizer_name, lr, batch_size, epoch, architecture_index=architecture_index)


if __name__ == '__main__':
    main()