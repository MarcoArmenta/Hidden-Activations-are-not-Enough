import json
import torch
import torchvision
import torchvision.transforms as transforms
import numpy.random as rand
import argparse

from representation import MlpRepresentation
from utils.utils import get_ellipsoid_data, is_in_ellipsoid, zero_std, get_model, subset

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
    parser.add_argument(
        "--subset_size",
        type=int,
        default=10000,
        help="Size of data subset to .",
    )

    return parser.parse_args()


def reject_predicted_out_of_dist(exp_dataset_train: torch.Tensor,
                                 exp_dataset_test: torch.Tensor,
                                 representation,
                                 ellipsoids: dict,
                                 std: float,
                                 dataset: str,
                                 d1: float=0.1,
                                 d2: float=0.1,
                                 n: int=5000) -> None:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if dataset == "mnist":
        out_of_dist = "fashion"
        out_dist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    elif dataset == "fashion":
        out_of_dist = "mnist"
        out_dist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    else:
        raise NotImplementedError(f"Data set must be 'mnist' or 'fashion', not '{dataset}'.")
    model = get_model(WEIGHT_PATH)
    out_dist_data, _ = subset(out_dist_test, len(exp_dataset_test), input_shape=(1, 28, 28))
    print(f"Model: {dataset}")
    print(f"Out of distribution data: {out_of_dist}")
    # Compute mean and std of number of (almost) zero dims in in_dist_data
    print("Compute rejection level.")
    zeros = torch.Tensor()
    in_ell = torch.Tensor()
    for im in exp_dataset_train[:n]:
        pred = torch.argmax(model.forward(im))
        mat = representation.forward(im)
        mean_mat = get_ellipsoid_data(ellipsoids, pred, "mean")
        std_mat = get_ellipsoid_data(ellipsoids, pred, "std")
        zeros = torch.cat((zeros, zero_std(mat, std_mat, d1).expand([1])))
        in_ell = torch.cat((in_ell, is_in_ellipsoid(mat, mean_mat, std_mat, std=std).expand([1])))
    zeros_lb = zeros.mean().item() - std*zeros.std().item()
    zeros_ub = zeros.mean().item() + std*zeros.std().item()
    in_ell_lb = in_ell.mean().item() - std*in_ell.std().item()
    in_ell_ub = in_ell.mean().item() + std*in_ell.std().item()
    print(f"Will reject when 'zero dims' is not in [{zeros_lb}, {zeros_ub}] or when 'in ell' is not in [{in_ell_lb},{in_ell_ub}].")
    results = []  # (Rejected, Was out of dist)
    data = {"in dist": exp_dataset_test[n:], "out dist": out_dist_data[n:]}
    for i in range(len(exp_dataset_test[n:])):
        d = rand.choice(["in dist", "out dist"])
        im = data[d][i]
        pred = torch.argmax(model.forward(im))
        mat = representation.forward(im)
        mean_mat = get_ellipsoid_data(ellipsoids, pred, "mean")
        std_mat = get_ellipsoid_data(ellipsoids, pred, "std")
        zero_dims = zero_std(mat, std_mat, d2).item()
        in_ell = is_in_ellipsoid(mat, mean_mat, std_mat, std=std).item()
        reject_zeros = (zeros_lb > zero_dims) or (zeros_ub < zero_dims)
        reject_dims = (in_ell_lb > in_ell) or (in_ell_ub < in_ell)
        results.append((reject_zeros or reject_dims, (d == "out dist")))
    good_defence = 0
    wrongly_rejected = 0
    num_out_dist = 0
    for rej,out in results:
        if out:
            good_defence += int(rej)
            num_out_dist += 1
        else:
            wrongly_rejected += int(rej)
    print(f"Percentage of good defences: {good_defence/num_out_dist}.")
    print(f"Percentage of wrong rejections: {wrongly_rejected/(len(results)-num_out_dist)}.")


if __name__ == "__main__":
    args = parse_args()
    if args.default_training:
        print('Loading experiment...')
        index = args.default_index
        experiment = DEFAULT_TRAININGS[f'experiment_{index}']

        optimizer_name = experiment['optimizer']
        dataset = experiment['dataset']
        lr = experiment['lr']
        batch_size = experiment['batch_size']
        epoch = experiment['epoch'] - 1

    else:
        optimizer_name = args.optimizer
        dataset = args.dataset
        lr = args.lr
        batch_size = args.batch_size
        epoch = args.epoch - 1

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

    # Make set of images for experiment
    exp_dataset_train, exp_labels_train = subset(train_set, args.subset_size)
    exp_dataset_test, exp_labels_test = subset(test_set, args.subset_size)



    reject_predicted_out_of_dist(exp_dataset_train,
                                 exp_dataset_test,
                                 representation,
                                 ellipsoids,
                                 std=2,
                                 dataset=args.dataset,
                                 n=args.subset_size//4)