from utils.utils import get_ellipsoid_data, zero_std, get_model, subset, get_dataset


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