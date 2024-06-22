from utils.utils import get_ellipsoid_data, zero_std, get_model, subset, get_dataset
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS
from matrix_construction.representation import MlpRepresentation
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
    parser.add_argument(
        "--d2",
        type=float,
        default=0.1,
        help="Determines how small should the standard deviation be per coordinate when detecting.",
    )
    parser.add_argument(
        "--num_samples_rejection_level",
        type=int,
        default=10000,
        help="Number of train samples to compute rejection level.",
    )

    return parser.parse_args()


def reject_predicted_attacks(exp_dataset_train: torch.Tensor,
                             exp_dataset_labels: torch.Tensor,
                             default_index,
                             weights_path,
                             architecture_index,
                             residual,
                             input_shape,
                             ellipsoids: dict,
                             num_samples_rejection_level: int = 5000,
                             std: float = 2,
                             d1: float = 0.1,
                             d2: float = 0.1,
                             verbose: bool = True) -> None:

    # Compute mean and std of number of (almost) zero dims
    reject_path = f'experiments/{default_index}/rejection_levels/reject_at_{num_samples_rejection_level}_{std}_{d1}.json'
    model = get_model(weights_path, architecture_index, residual, input_shape)

    if os.path.exists(reject_path):
        print("Loading rejection level...")
        file = open(reject_path)
        reject_at = json.load(file)[0]
    else:
        print("Computing rejection level...")
        representation = MlpRepresentation(model)
        zeros = torch.Tensor()
        for i in range(len(exp_dataset_train)):
            im = exp_dataset_train[i]
            pred = torch.argmax(model.forward(im))
            if pred != exp_dataset_labels[i]:
                continue

            mat = representation.forward(im)

            a = get_ellipsoid_data(ellipsoids, pred, "std")
            b = zero_std(mat, a, d1)
            c = b.expand([1])
            zeros = torch.cat((zeros, c))

        reject_at = zeros.mean().item() - std*zeros.std().item()

        with open(reject_path, 'w') as json_file:
            json.dump([reject_at], json_file, indent=4)

    if reject_at <= 0:
        raise ValueError(f"Rejection level is {reject_at}")

    print(f"Will reject when 'zero dims' < {reject_at}.")
    adv_succes = []  # Save adversarial examples that were not detected
    results = []  # (Rejected, Was attacked)
    # For test counts how many were trusted, and for attacks how many where detected
    counts = {key: 0 for key in ["test"] + ATTACKS}

    path_adv_matrices = f'experiments/{default_index}/adversarial_matrices/'
    attacked_dataset = torch.load(f'experiments/{default_index}/adversarial_examples/adversarial_examples.pth')

    for a in ["test"]+ATTACKS:
        not_rejected_and_attacked = 0
        not_rejected_and_not_attacked = 0
        rejected_and_attacked = 0
        rejected_and_not_attacked = 0

        for i in range(len(attacked_dataset[a])):
            current_matrix_path = path_adv_matrices + f"/{a}/{i}/matrix.pth"
            im = attacked_dataset[a][i]
            pred = torch.argmax(model.forward(im))
            mat = torch.load(current_matrix_path)

            b = get_ellipsoid_data(ellipsoids, pred, "std")
            c = zero_std(mat, b, d2).item()

            res = ((reject_at > c), (a != "test"))

            # if not rejected and it was an attack
            # so detected adversarial example
            if not res[0] and a != "test":
                not_rejected_and_attacked += 1
                counts[a] += 1
                adv_succes.append(im)

            # if rejected and it was an attack
            if res[0] and a != 'test':
                rejected_and_attacked += 1

            # if rejected and it was test data
            # so wrong rejection of natural data
            if res[0] and a == "test":
                rejected_and_not_attacked += 1

            # if not rejected and it was test data
            if not res[0] and a == "test":
                not_rejected_and_not_attacked += 1
                counts[a] += 1

            results.append(res)

        if verbose:
            print("Attack method: ", a)
            if a == 'test':
                print(f'Wrong rejections : {rejected_and_not_attacked} out of {len(attacked_dataset[a])}')
                print(f'Trusted test data : {not_rejected_and_not_attacked} out of {len(attacked_dataset[a])}')

            print(f'Detected adversarial examples : {rejected_and_attacked} out of {len(attacked_dataset[a])}')
            print(f'Successful adversarial examples : {not_rejected_and_attacked} out of {len(attacked_dataset[a])}')

    counts_file = f'experiments/{default_index}/adversarial_examples/number_of_detections_per_attack.json'
    with open(counts_file, 'w') as json_file:
        json.dump(counts, json_file, indent=4)

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

    counts_tensor = torch.tensor([counts[key] for key in ["test"] + ATTACKS], dtype=torch.float)
    num_attacked_samples = torch.tensor([len(attacked_dataset[key]) for key in ["test"] + ATTACKS], dtype=torch.float)
    normalized_counts = counts_tensor / num_attacked_samples
    probabilities = {key: normalized_counts[i].item() for i, key in enumerate(["test"] + ATTACKS)}
    probs = f'experiments/{default_index}/adversarial_examples/prob-adv-success-per-attack_{num_samples_rejection_level}_{std}_{d1}_{d2}.json'
    with open(probs, 'w') as json_file:
        json.dump(probabilities, json_file, indent=4)  # indent=4 is optional, for pretty printing

    torch.save(adv_succes,
               f'experiments/{default_index}/adversarial_examples/adversarial_success/adv_success_{num_samples_rejection_level}_{std}_{d1}_{d2}.pth')


def main():
    args = parse_args()
    if args.default_index is not None:
        try:
            experiment = DEFAULT_EXPERIMENTS[f'experiment_{args.default_index}']

            architecture_index = experiment['architecture_index']
            residual = experiment['residual']
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
    exp_dataset_train, exp_dataset_labels = subset(train_set, args.num_samples_rejection_level)

    ellipsoids_file = open(f"experiments/{args.default_index}/matrices/matrix_statistics.json")
    ellipsoids = json.load(ellipsoids_file)

    reject_predicted_attacks(exp_dataset_train,
                             exp_dataset_labels,
                             args.default_index,
                             weights_path,
                             architecture_index,
                             residual,
                             input_shape,
                             ellipsoids,
                             args.num_samples_rejection_level,
                             args.std,
                             args.d1,
                             args.d2,
                             verbose=True)


if __name__ == '__main__':
    main()
