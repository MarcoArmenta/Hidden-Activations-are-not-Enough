"""
    Creates matrices of existing adversarial examples
"""
import torch
import argparse
from multiprocessing import Pool
from pathlib import Path

from matrix_construction.representation import MlpRepresentation
from utils.utils import get_model, get_dataset
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default_index",
        type=int,
        default=0,
        help="Index of default trained network.",
    )
    parser.add_argument(
        "--nb_workers",
        type=int,
        default=8,
        help="How many processes in parallel for adversarial examples computations and their matrices.",
    )

    return parser.parse_args()


def save_one_matrix(im, attack, i, path_adv_matrices, weights_path, architecture_index, residual, input_shape, dropout):
    model = get_model(weights_path, architecture_index, residual, input_shape, dropout)
    representation = MlpRepresentation(model)
    matrix_save_path = path_adv_matrices / f'{attack}' / f'{i}/matrix.pth'
    matrix_save_path.parent.mkdir(parents=True, exist_ok=True)
    if not matrix_save_path.exists():
        mat = representation.forward(im)
        torch.save(mat, matrix_save_path)


def generate_matrices_for_attacks(path_adv_matrices,
                                  experiment_dir,
                                  weights_path,
                                  architecture_index,
                                  residual,
                                  input_shape,
                                  dropout,
                                  nb_workers):
    for attack in ['test'] + ATTACKS:
        path_adv_examples = experiment_dir / f"{attack}/adversarial_examples.pth"
        attacked_dataset = torch.load(path_adv_examples)

        print(f"Generating matrices for attack {attack}.", flush=True)

        arguments = [(attacked_dataset[i],
                      attack,
                      i,
                      path_adv_matrices,
                      weights_path,
                      architecture_index,
                      residual,
                      input_shape,
                      dropout)
                     for i in range(len(attacked_dataset))]

        with Pool(processes=nb_workers) as pool:
            pool.starmap(save_one_matrix, arguments)


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
            print(f"When computing adversarial examples of new model, add the experiment to constants.constants.py inside DEFAULT_EXPERIMENTS"
                  f"and provide the corresponding --default_index N when running this script.")
            return -1

    else:
        raise ValueError("Default index not specified in constants/constants.py")

    print("Experiment: ", args.default_index)

    weights_path = Path(f'experiments/{args.default_index}/weights') / f'epoch_{epoch}.pth'
    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    input_shape = (3, 32, 32) if dataset == 'cifar10' else (1, 28, 28)
    _, test_set = get_dataset(dataset, data_loader=False)

    experiment_dir = Path(f'experiments/{args.default_index}/adversarial_examples')
    experiment_dir.mkdir(parents=True, exist_ok=True)

    path_adv_matrices = Path(f'experiments/{args.default_index}/adversarial_matrices/')
    path_adv_matrices.mkdir(parents=True, exist_ok=True)

    generate_matrices_for_attacks(path_adv_matrices,
                                  experiment_dir,
                                  weights_path,
                                  architecture_index,
                                  residual,
                                  input_shape,
                                  dropout,
                                  args.nb_workers)


if __name__ == "__main__":
    main()
