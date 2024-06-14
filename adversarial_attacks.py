"""
    Creates adversarial examples and tries to detect them using matrix statistics
"""
import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
import torchattacks
import argparse
from multiprocessing import Pool

from matrix_construction.representation import MlpRepresentation
from utils.utils import get_ellipsoid_data, zero_std, get_model, subset, get_dataset
from constants.constants import DEFAULT_EXPERIMENTS


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset_size",
        type=int,
        default=10000,
        help="Size of data subset to .",
    )
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
        "--d2",
        type=float,
        default=0.1,
        help="Determines how small should the standard deviation be per coordinate when detecting.",
    )
    parser.add_argument(
        "--nb_workers",
        type=int,
        default=8,
        help="How many processes in parallel for adversarial examples computations.",
    )

    return parser.parse_args()


def apply_attack(args):
    attack_name, attack, data, labels, model = args

    if attack_name == 'None':
        return attack_name, data

    attacked_data = attack(data, labels)
    attacked_predictions = torch.argmax(model(attacked_data), dim=1)
    misclassified = (labels != attacked_predictions).sum().item()
    total = data.size(0)

    print(f"Attack: {attack_name}")
    print(f"Misclassified after attack: {misclassified} out of {total}")

    # Filter only the attacked images where labels != attacked_predictions
    misclassified_indexes = labels != attacked_predictions
    misclassified_images = attacked_data[misclassified_indexes]

    return attack_name, misclassified_images


def reject_predicted_attacks(exp_dataset_train: torch.Tensor,
                             exp_dataset_test: torch.Tensor,
                             exp_labels_test: torch.Tensor,
                             representation,
                             ellipsoids: dict,
                             experiment_path,
                             num_samples_rejection_level: int = 5000,
                             std: float = 2,
                             d1: float = 0.1,
                             d2: float = 0.1,
                             nb_workers: int = 1,
                             verbose: bool = True) -> None:

    model = representation.model

    attacks = ["GN", "FGSM", "RFGSM", "PGD", "EOTPGD", "FFGSM", "TPGD", "MIFGSM", "UPGD", "DIFGSM", "NIFGSM",
               "PGDRS", "SINIFGSM", "VMIFGSM", "VNIFGSM", "CW", "PGDL2", "PGDRSL2", "DeepFool", "SparseFool",
               "OnePixel", "Pixle", "FAB"]

    attacks_cls = dict(zip(["None"]+attacks,
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

    #path_adv_examples = 'experiments/adversarial_examples/' + experiment_path + f'/adversarial_examples_{num_samples_rejection_level}.pth'
    path_adv_examples = 'experiments/adversarial_examples/' + f'{experiment_path}' + f'/adversarial_examples_{num_samples_rejection_level}.pth'

    if os.path.exists(path_adv_examples):
        print("Loading adversarial examples...")
        attacked_dataset = torch.load(path_adv_examples)
    else:
        print("Computing adversarial examples...")
        arguments = [(a,
                      attacks_cls[a],
                      exp_dataset_test[num_samples_rejection_level:].detach(),
                      exp_labels_test[num_samples_rejection_level:].detach(),
                      model
                      )
                     for a in ["None"] + attacks
                     ]

        with Pool(processes=nb_workers) as pool:
            results = pool.map(apply_attack, arguments)

        attacked_dataset = dict(results)

        if not os.path.exists('experiments/adversarial_examples/' + experiment_path + '/'):
            os.makedirs('experiments/adversarial_examples/' + experiment_path + '/')

        torch.save(attacked_dataset, path_adv_examples)
    # TODO: script that only computes adversarial examples in parallel
    number_of_attacks_path = 'experiments/adversarial_examples/' + experiment_path + f'/number_examples_per_attack_{num_samples_rejection_level}.json'
    nb_attacks = {a: len(attacked_dataset[a]) for a in attacks}
    with open(number_of_attacks_path, 'w') as json_file:
        json.dump(nb_attacks, json_file, indent=4)

    if verbose:
        tt = 0
        print("Number of adversarial examples per attack method.")
        for a in ['None'] + attacks:
            print(a, len(attacked_dataset[a]))
            tt += len(attacked_dataset[a])

        print("Total adv examples: ", tt)

    # Compute mean and std of number of (almost) zero dims
    reject_path = 'experiments/adversarial_examples/' + experiment_path + f'/reject_at_{num_samples_rejection_level}_{std}_{d1}.json'
    if os.path.exists(reject_path):
        print("Loading rejection level...")
        file = open(reject_path)
        reject_at = json.load(file)[0]
    else:
        print("Compute rejection level...")
        zeros = torch.Tensor()
        for i in range(len(exp_dataset_train[:num_samples_rejection_level])):
            im = exp_dataset_train[i]
            pred = torch.argmax(model.forward(im))

            mat = representation.forward(im)

            a = get_ellipsoid_data(ellipsoids, pred, "std")
            b = zero_std(mat, a, d1)
            c = b.expand([1])

            zeros = torch.cat((zeros, c))

        reject_at = zeros.mean().item() - std*zeros.std().item()

        with open(reject_path, 'w') as json_file:
            json.dump([reject_at], json_file, indent=4)

    if reject_at <= 0:
        print(f"Rejection level is {reject_at}")
        return

    print(f"Will reject when 'zero dims' < {reject_at}.")
    adv_succes = []  # Save adversarial examples that were not detected
    results = []  # (Rejected, Was attacked)

    counts = {key: 0 for key in ["None"] + attacks}

    path_adv_matrices = 'experiments/adversarial_examples/' + experiment_path + '/matrices'

    for a in ["None"]+attacks:
        not_rejected_and_attacked = 0
        rejected_and_attacked = 0
        rejected_and_not_attacked = 0
        current_path = path_adv_matrices + f"/{a}/"
        for i in range(len(attacked_dataset[a])):
            im = attacked_dataset[a][i]
            pred = torch.argmax(model.forward(im))

            if os.path.exists(current_path + f'{i}/matrix.pth'):
                mat = torch.load(current_path + f'{i}/matrix.pth')
            else:
                mat = representation.forward(im)
                os.makedirs(current_path, exist_ok=True)
                torch.save(mat, current_path + 'matrix.pth')

            b = get_ellipsoid_data(ellipsoids, pred, "std")
            c = zero_std(mat, b, d2).item()

            res = ((reject_at > c), (a != "None"))

            # if not rejected and it was an attack
            if not res[0] and a != "None":
                not_rejected_and_attacked += 1
                counts[a] += 1
                adv_succes.append(im)

            # if rejected and it was an attack
            if res[0] and a != 'None':
                rejected_and_attacked += 1

            # if rejected and it was test data
            if res[0] and a == "None":
                rejected_and_not_attacked += 1
                counts[a] += 1

            results.append(res)

        if verbose:
            print("Method: ", a)
            if a == 'None':
                print(f'Wrong rejection! : {rejected_and_not_attacked} out of {len(attacked_dataset[a])}')

            print(f'Defence! : {rejected_and_attacked} out of {len(attacked_dataset[a])}')
            print(f'Attacked! : {not_rejected_and_attacked} out of {len(attacked_dataset[a])}')

    good_defence = 0
    wrongly_rejected = 0
    num_att = 0
    for rej, att in results:
        if att:
            good_defence += int(rej)
            num_att += 1
        else:
            wrongly_rejected += int(rej)
    print(f"Percentage of good defences: {good_defence/num_att}")
    print(f"Percentage of wrong rejections: {wrongly_rejected/(len(results)-num_att)}")

    counts_tensor = torch.tensor([counts[key] for key in ["None"] + attacks], dtype=torch.float)
    num_attacked_samples = torch.tensor([len(attacked_dataset[key]) for key in ["None"] + attacks], dtype=torch.float)
    normalized_counts = counts_tensor / num_attacked_samples
    probabilities = {key: normalized_counts[i].item() for i, key in enumerate(["None"] + attacks)}
    probabilities['None'] = wrongly_rejected/(len(results)-num_att)
    probs = 'experiments/adversarial_examples/' + experiment_path + f'/prob-adv-success-per-attack_' \
                                                                    f'{num_samples_rejection_level}_{std}_{d1}_{d2}.json'
    with open(probs, 'w') as json_file:
        json.dump(probabilities, json_file, indent=4)  # indent=4 is optional, for pretty printing

    torch.save(adv_succes,
               'experiments/adversarial_examples/'
               + experiment_path +
               f'/adv_success_{num_samples_rejection_level}_{std}_{d1}_{d2}.pth')

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
            print(f"When computing matrices of new model, add the experiment to constants.constants.py inside DEFAULT_EXPERIMENTS"
                  f"and provide the corresponding --default_index when running this script.")
            return

    experiment_path = f'{dataset}/{architecture_index}/{optimizer_name}/{lr}/{batch_size}'

    print("Experiment: ", experiment_path)

    weights_path = 'experiments/weights/' + experiment_path + f'/{epoch}/epoch_{epoch}.pth'
    if not os.path.exists(weights_path):
        ValueError(f"Experiment needs to be trained with hyper-parameters: {weights_path}")

    matrices_path = 'experiments/matrices/' + f'{experiment_path}/{epoch}/matrix_statistics.json'
    if not os.path.exists(matrices_path):
        ValueError(f"Matrix statistics have to be computed: {matrices_path}")

    train_set, test_set = get_dataset(dataset, data_loader=False)

    model = get_model(weights_path, architecture_index)
    representation = MlpRepresentation(model)
    ellipsoids_file = open(matrices_path)
    ellipsoids: dict = json.load(ellipsoids_file)

    # Make set of images for experiment
    exp_dataset_train, _ = subset(train_set, args.subset_size)
    exp_dataset_test, exp_labels_test = subset(test_set, args.subset_size)

    reject_predicted_attacks(exp_dataset_train,
                             exp_dataset_test,
                             exp_labels_test,
                             representation,
                             ellipsoids,
                             experiment_path=experiment_path,
                             num_samples_rejection_level=args.subset_size // 2,
                             std=args.std,
                             d1=args.d1,
                             d2=args.d2,
                             nb_workers=args.nb_workers,
                             verbose=True)


if __name__ == "__main__":
    args = parse_args()
    if args.default_hyper_parameters:
        print("Loading default experiment.")
        index = args.default_index
        experiment = DEFAULT_EXPERIMENTS[f'experiment_{index}']

        optimizer_name = experiment['optimizer']
        dataset = experiment['dataset']
        lr = experiment['lr']
        batch_size = experiment['batch_size']
        epoch = experiment['epoch']-1
        hidden_layers_idx = experiment['hidden_layers_idx']

    else:
        print("Loading custom experiment.")
        optimizer_name = args.optimizer
        dataset = args.dataset
        lr = args.lr
        batch_size = args.batch_size
        epoch = args.epoch-1

