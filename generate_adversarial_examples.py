"""
    Creates adversarial examples and tries to detect them using matrix statistics
"""
import os
import json
import torch
import torchattacks
import argparse
from multiprocessing import Pool
from pathlib import Path

from matrix_construction.representation import MlpRepresentation
from utils.utils import get_model, get_dataset, subset
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
        default=1,
        help="How many processes in parallel for adversarial examples computations and their matrices.",
    )

    return parser.parse_args()


def apply_attack(attack_name, data, labels, weights_path, architecture_index):
    model = get_model(weights_path, architecture_index)

    attacks_classes = dict(zip(["test"] + ATTACKS,
                               [torchattacks.VANILA(model),
                                torchattacks.GN(model),
                                torchattacks.FGSM(model),
                                torchattacks.RFGSM(model),
                                torchattacks.PGD(model),
                                torchattacks.EOTPGD(model),
                                torchattacks.FFGSM(model),
                                torchattacks.TPGD(model),
                                torchattacks.MIFGSM(model),
                                torchattacks.UPGD(model),
                                torchattacks.DIFGSM(model),
                                torchattacks.Jitter(model),
                                torchattacks.NIFGSM(model),
                                torchattacks.PGDRS(model),
                                torchattacks.SINIFGSM(model),
                                torchattacks.VMIFGSM(model),
                                torchattacks.VNIFGSM(model),
                                torchattacks.CW(model),
                                torchattacks.PGDL2(model),
                                torchattacks.PGDRSL2(model),
                                torchattacks.DeepFool(model),
                                torchattacks.SparseFool(model),
                                torchattacks.OnePixel(model),
                                torchattacks.Pixle(model),
                                torchattacks.FAB(model),
                                ]))

    attacked_data = attacks_classes[attack_name](data, labels)
    attacked_predictions = torch.argmax(model(attacked_data), dim=1)
    misclassified = (labels != attacked_predictions).sum().item()
    total = data.size(0)

    print(f"Attack: {attack_name}")
    print(f"Misclassified after attack: {misclassified} out of {total}")

    # Filter only the attacked images where labels != attacked_predictions
    misclassified_indexes = labels != attacked_predictions
    misclassified_images = attacked_data[misclassified_indexes]

    return attack_name, misclassified_images

#(a, path_adv_matrices, path_adv_examples, weights_path, architecture_index)
def generate_matrices_for_attacks(attack, path_adv_matrices, dataset_path, weights_path, architecture_index):

    current_path = Path(path_adv_matrices) / f"{attack}/"
    attacked_dataset = torch.load(dataset_path)

    model = get_model(weights_path, architecture_index)
    representation = MlpRepresentation(model)

    for i in range(len(attacked_dataset[attack])):
        im = attacked_dataset[attack][i]
        if not os.path.exists(current_path / f'{i}/matrix.pth'):
            mat = representation.forward(im)
            torch.save(mat, current_path / f'{i}/matrix.pth')
    print(f"Attack {attack} finished.")


def generate_adversarial_examples_and_their_matrices(exp_dataset_test: torch.Tensor,
                                                     exp_labels_test: torch.Tensor,
                                                     weights_path,
                                                     architecture_index,
                                                     representation,
                                                     experiment_path,
                                                     nb_workers,
                                                     ):

    experiment_dir = Path('experiments/adversarial_examples') / experiment_path
    experiment_dir.mkdir(parents=True, exist_ok=True)
    path_adv_examples = experiment_dir / 'adversarial_examples.pth'

    if path_adv_examples.exists():
        print("Loading adversarial examples...")
        attacked_dataset = torch.load(path_adv_examples)
    else:
        print("Computing adversarial examples...")

        exp_dataset_test = exp_dataset_test.detach().clone()
        exp_labels_test = exp_labels_test.detach().clone()

        arguments = [(attack_name,
                      exp_dataset_test,
                      exp_labels_test,
                      weights_path,
                      architecture_index)
                     for attack_name in ["test"] + ATTACKS]

        with Pool(processes=nb_workers) as pool:
            results = pool.starmap(apply_attack, arguments)

        attacked_dataset = {attack_name: result for attack_name, result in results}

        torch.save(attacked_dataset, path_adv_examples)

        number_of_attacks_path = experiment_dir / 'number_examples_per_attack.json'
        nb_attacks = {a: len(attacked_dataset[a]) for a in attacked_dataset.keys()}
        with number_of_attacks_path.open('w') as json_file:
            json.dump(nb_attacks, json_file, indent=4)

    path_adv_matrices = experiment_dir / 'matrices'
    path_adv_matrices.mkdir(parents=True, exist_ok=True)

    arguments = [(a, path_adv_matrices, path_adv_examples, weights_path, architecture_index) for a in ["test"] + ATTACKS]
    with Pool(processes=nb_workers) as pool:
        pool.starmap(generate_matrices_for_attacks, arguments)


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

        except KeyError:
            print(f"Error: Default index {args.default_index} does not exist.")
            print(f"When computing adversarial examples of new model, add the experiment to constants.constants.py inside DEFAULT_EXPERIMENTS"
                  f"and provide the corresponding --default_index N when running this script.")
            return

    experiment_path = f'{dataset}/{architecture_index}/{optimizer_name}/{lr}/{batch_size}'

    print("Experiment: ", experiment_path)

    weights_path = Path('experiments/weights') / experiment_path / f'epoch_{epoch}.pth'
    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained with hyper-parameters: {weights_path}")

    matrices_path = Path('experiments/matrices') / experiment_path / f'{epoch}/matrix_statistics.json'
    if not matrices_path.exists():
        raise ValueError(f"Matrix statistics have to be computed: {matrices_path}")

    train_set, test_set = get_dataset(dataset, data_loader=False)

    model = get_model(weights_path, architecture_index)
    representation = MlpRepresentation(model)

    exp_dataset_test, exp_labels_test = subset(test_set, len(test_set))

    generate_adversarial_examples_and_their_matrices(exp_dataset_test,
                                                     exp_labels_test,
                                                     weights_path,
                                                     architecture_index,
                                                     representation,
                                                     experiment_path=experiment_path,
                                                     nb_workers=args.nb_workers)


if __name__ == "__main__":
    main()
