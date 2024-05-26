"""
    Creates adversarial examples and checks if it's within the ellipsoid
"""

import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
import torchattacks
import matplotlib.pyplot as plt
import numpy.random as rand
import argparse

from representation import MlpRepresentation
from data_mod import get_ellipsoid_data, zero_std, get_model, subset

from manual_training import DEFAULT_TRAININGS

ADV = ["GN", "FGSM", "BIM", "RFGSM", "PGD",
        "EOTPGD", "FFGSM", "TPGD", "MIFGSM", "UPGD", "APGD", "APGDT", "DIFGSM"]
    # "NIFGSM","PGDRS","SINIFGSM","VMIFGSM","VNIFGSM","CW","PGDL2","PGDRSL2","DeepFool","SparseFool","OnePixel","Pixle",
    # "FAB","AutoAttack","Square","SPSA","JSMA","EADL1","EADEN","PIFGSM","PIFGSMPP"]


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
        help="Optimizer used to train the model.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate used to train the model with.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size used to train the model with.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=20,
        help="Epoch of training to do the analysis.",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=10000,
        help="Size of data subset to .",
    )
    parser.add_argument(
        "--default_training",
        type=bool,
        default=True,
        help="Wether to use a default trained network.",
    )
    parser.add_argument(
        "--default_index",
        type=int,
        default=0,
        help="Index of default trained networks.",
    )


    return parser.parse_args()


def reject_predicted_attacks(exp_dataset_train: torch.Tensor,
                             exp_dataset_test: torch.Tensor,
                             exp_labels_test: torch.Tensor,
                             representation,
                             ellipsoids: dict,
                             model,
                             num_samples_rejection_level: int = 5000,
                             std: float = 2,
                             d1: float = 0.1,
                             d2: float = 0.1) -> None:

    #"EOTPGD", "FFGSM", "TPGD", "MIFGSM", "UPGD", "APGD", "APGDT", "DIFGSM", "TIFGSM", "Jitter"]

    attacks = [a for a in ADV if a != "BIM"]  # Can't predict BIM with this method
    #["GN", "FGSM", "BIM", "RFGSM", "PGD", "EOTPGD", "FFGSM", "TPGD", "MIFGSM", "UPGD", "APGD", "APGDT", "DIFGSM",
    # "TIFGSM", "Jitter",
    # "NIFGSM", "PGDRS", "SINIFGSM", "VMIFGSM", "VNIFGSM", "CW", "PGDL2", "PGDRSL2", "DeepFool", "SparseFool",
    # "OnePixel", "Pixle",
    # "FAB", "AutoAttack", "Square", "SPSA", "JSMA", "EADL1", "EADEN", "PIFGSM", "PIFGSMPP"]
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
                            torchattacks.APGD(model),
                            torchattacks.APGDT(model),
                            torchattacks.DIFGSM(model),
                            #torchattacks.TIFGSM(model),
                            #torchattacks.Jitter(model),
                            ]))

    path_adv_examples = 'experiments/adversarial_examples/' + EXPERIMENT_PATH + '/adversarial_examples.pth'

    if os.path.exists(path_adv_examples):
        print("Loading adversarial examples...")
        attacked_dataset = torch.load(path_adv_examples)
    else:
        print("Computing adversarial examples...")
        attacked_dataset = {a: attacks_cls[a](exp_dataset_test[num_samples_rejection_level:], exp_labels_test[num_samples_rejection_level:]) for a in ["None"]+attacks}
        torch.save(attacked_dataset, path_adv_examples)

    # Compute mean and std of number of (almost) zero dims
    reject_path = 'experiments/adversarial_examples/' + EXPERIMENT_PATH + '/reject_at.json'
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
    print(f"Will reject when 'zero dims' < {reject_at}.")
    adv_succes = [] # Save 10 adversarial examples that were not detected
    results = []  # (Rejected, Was attacked)
    for i in range(len(exp_dataset_test[num_samples_rejection_level:])):
        a = rand.choice(["None"]+attacks, p=[0.5]+[1/(2*len(attacks)) for _ in attacks])
        im = attacked_dataset[a][i]
        pred = torch.argmax(model.forward(im))
        mat = representation.forward(im)

        b = get_ellipsoid_data(ellipsoids, pred, "std")
        c = zero_std(mat, b, d2).item()

        res = ((reject_at > c), (a != "None"))

        if not res[0] and len(adv_succes) < 10:
            adv_succes.append(im)

        results.append(res)

    torch.save(torch.tensor(adv_succes), 'experiments/adversarial_examples/' + EXPERIMENT_PATH + '/adv_success.pth')

    good_defence = 0
    wrongly_rejected = 0
    num_att = 0
    for rej, att in results:
        if att:
            good_defence += int(rej)
            num_att += 1
        else:
            wrongly_rejected += int(rej)
    print(f"Percentage of good defences: {good_defence/num_att}.")
    print(f"Percentage of wrong rejections: {wrongly_rejected/(len(results)-num_att)}.")


def show_adv_img(img: torch.Tensor, attacked_img: torch.Tensor, label: str, output: str, attack: str) -> None:
    _, axes = plt.subplots(1, 2, figsize=(15, 15))

    axes[0].imshow(img[0], cmap='gray')
    axes[0].set_title(f"Label: {label}")
    axes[1].imshow(attacked_img[0], cmap='gray')
    axes[1].set_title(f"Attack: {attack}. Output: {output}")
    for i in range(2): axes[i].axis("off")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    if args.default_training:
        index = args.default_index
        experiment = DEFAULT_TRAININGS[f'experiment_{index}']

        optimizer_name = experiment['optimizer']
        dataset = experiment['dataset']
        lr = experiment['lr']
        batch_size = experiment['batch_size']
        epoch = experiment['epoch']-1

    else:
        optimizer_name = args.optimizer
        dataset = args.dataset
        lr = args.lr
        batch_size = args.batch_size
        epoch = args.epoch-1

    EXPERIMENT_PATH = f'{dataset}/{optimizer_name}/{lr}/{batch_size}'
    print("Experiment: ", EXPERIMENT_PATH)
    MATRICES_PATH = 'experiments/matrices/' + EXPERIMENT_PATH + f'/{epoch}/'
    WEIGHT_PATH = 'experiments/weights/' + EXPERIMENT_PATH + f'/epoch_{epoch}.pth'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if args.dataset == "mnist":
        train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    elif args.dataset == "fashion":
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    representation = MlpRepresentation(get_model(WEIGHT_PATH))
    ellipsoids_file = open(f"{MATRICES_PATH}/matrix_statistics.json")
    ellipsoids: dict = json.load(ellipsoids_file)
    model = get_model(WEIGHT_PATH)

    # Make set of images for experiment
    exp_dataset_train, exp_labels_train = subset(train_set, args.subset_size)
    exp_dataset_test, exp_labels_test = subset(test_set, args.subset_size)

    reject_predicted_attacks(exp_dataset_train,
                             exp_dataset_test,
                             exp_labels_test,
                             representation,
                             ellipsoids,
                             model=model,
                             num_samples_rejection_level=args.subset_size//2,
                             std=1,
                             d1=0.1,
                             d2=0.1)
