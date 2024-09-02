"""
    Creates matrices of existing adversarial examples
"""
import torch
import argparse
from multiprocessing import Pool
from pathlib import Path

from matrix_construction.representation import MlpRepresentation
from utils.utils import get_model
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS
from utils.utils import zip_and_cleanup


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
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Temporary directory to save and read data. Useful when using clusters."
    )

    return parser.parse_args()


def save_one_matrix(im, attack, i, default_index, weights_path, architecture_index, residual, input_shape, dropout, temp_dir):
    model = get_model(weights_path, architecture_index, residual, input_shape, dropout)
    representation = MlpRepresentation(model)
    if temp_dir is not None:
        matrix_save_path = Path(f'{temp_dir}/experiments/{default_index}/adversarial_matrices') / f'{attack}' / f'{i}/matrix.pth'
    else:
        matrix_save_path = Path(f'experiments/{default_index}/adversarial_matrices') / f'{attack}' / f'{i}/matrix.pth'

    matrix_save_path.parent.mkdir(parents=True, exist_ok=True)
    if not matrix_save_path.exists():
        mat = representation.forward(im)
        torch.save(mat, matrix_save_path)


def generate_matrices_for_attacks(default_index,
                                  temp_dir,
                                  weights_path,
                                  architecture_index,
                                  residual,
                                  input_shape,
                                  dropout,
                                  nb_workers):
    for attack in ['test'] + ATTACKS:
        if temp_dir is not None:
            path_adv_examples = Path(temp_dir) / f'experiments/{default_index}/adversarial_examples' / f"{attack}/adversarial_examples.pth"
        else:
            path_adv_examples = Path(f'experiments/{default_index}/adversarial_examples') / f"{attack}/adversarial_examples.pth"
        if not path_adv_examples.exists():
            print(f'Attak {attack} does NOT exists.', flush=True)
            continue
        attacked_dataset = torch.load(path_adv_examples)

        print(f"Generating matrices for attack {attack}.", flush=True)

        arguments = [(attacked_dataset[i],
                      attack,
                      i,
                      default_index,
                      weights_path,
                      architecture_index,
                      residual,
                      input_shape,
                      dropout, temp_dir)
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

    print("Experiment: ", args.default_index, flush=True)

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{args.default_index}/weights/epoch_{epoch}.pth')
    else:
        weights_path = Path(f'experiments/{args.default_index}/weights/epoch_{epoch}.pth')

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    input_shape = (3, 32, 32) if dataset == 'cifar10' else (1, 28, 28)

    generate_matrices_for_attacks(args.default_index,
                                  args.temp_dir,
                                  weights_path,
                                  architecture_index,
                                  residual,
                                  input_shape,
                                  dropout,
                                  args.nb_workers)

    if args.temp_dir is not None:
        zip_and_cleanup(f'{args.temp_dir}/experiments/{args.default_index}/adversarial_matrices/',
                        f'experiments/{args.default_index}/adversarial_matrices/adversarial_matrices', clean=False)


if __name__ == "__main__":
    main()
