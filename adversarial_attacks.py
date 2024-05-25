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
from mlp import MLP
from data_mod import get_ellipsoid_data, is_in_ellipsoid, zero_std, get_model, subset

from manual_training import DEFAULT_TRAININGS


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


def adv_attack_exp(exp_dataset: torch.Tensor, exp_labels: torch.Tensor, representation, ellipsoids, n: int, overwrite: bool=False) -> None:
    # Attack n inputs. Compute the prediction and how many dimensions of the matrix is within 2 standard deviations.
    save = True
    if os.path.exists(f"{MATRICES_PATH}adv_predictions.json") or os.path.exists(f"{MATRICES_PATH}adv_ellipsoids.json"):
        print("Files already exist")
        print(f"Overwrite = {overwrite}")
        save = overwrite

    model: MLP = get_model(WEIGHT_PATH)

    no_attack = torchattacks.VANILA(model)
    GN_attack = torchattacks.GN(model)
    FGSM_attack = torchattacks.FGSM(model)
    BIM_attack = torchattacks.BIM(model)
    RFGSM_attack = torchattacks.RFGSM(model)
    PGD_attack = torchattacks.PGD(model)
    EOTPGD_attack = torchattacks.EOTPGD(model)
    # MIFGSM_attack = torchattacks.MIFGSM(model)

    attacks = dict(zip(["None"]+ADV, 
                       [no_attack(exp_dataset, exp_labels),
                        GN_attack(exp_dataset, exp_labels),
                        FGSM_attack(exp_dataset, exp_labels),
                        BIM_attack(exp_dataset, exp_labels),
                        RFGSM_attack(exp_dataset, exp_labels),
                        PGD_attack(exp_dataset, exp_labels),
                        EOTPGD_attack(exp_dataset, exp_labels)]))

    results = torch.zeros([len(ATT),n],dtype=torch.long)
    in_ellipsoid = torch.zeros([len(ATT),n])

    for a in ["None"]+ADV:
        print(f"Compute {a} attack.")
        for i in range(n):
            results[ATT[a]][i] = torch.argmax(model.forward(attacks[a][i]))
            in_ellipsoid[ATT[a]][i] = is_in_ellipsoid(representation.forward(attacks[a][i]),
                                                      get_ellipsoid_data(ellipsoids,results[ATT[a]][i],"mean"),
                                                      get_ellipsoid_data(ellipsoids,results[ATT[a]][i],"std"))
    if save:
        data_pred = {a: results[ATT[a]].tolist() for a in ["None"]+ADV}
        data_pred["Labels"] = exp_labels.tolist()
        data_ell = {a: in_ellipsoid[ATT[a]].tolist() for a in ["None"]+ADV}

        with open(f"{MATRICES_PATH}adv_predictions.json", 'w') as json_pred:
            json.dump(data_pred, json_pred, indent=4)
        with open(f"{MATRICES_PATH}adv_ellipsoids.json", 'w') as json_ell:
            json.dump(data_ell, json_ell, indent=4)


def get_attacks():
    # Returns the list of indices of the dataset where the attack worked
    attacks_file = open(f"{MATRICES_PATH}adv_predictions.json")
    attacks = json.load(attacks_file)
    working_attacks = {a: [] for a in ADV}
    for i in range(len(attacks["None"])):
        if attacks["None"][i] == attacks["Labels"][i]:
            for a in ADV:
                if attacks[a][i] != attacks["Labels"][i]:
                    working_attacks[a].append(i)
    return working_attacks


def in_ellipsoid_attacked(exp_dataset: torch.Tensor, exp_labels: torch.Tensor, representation, ellipsoids: dict, overwrite: bool=False) -> None:
    save = True
    if os.path.exists(f"{MATRICES_PATH}in_ellipsoid_attacked.json"):
        print("File already exists")
        print(f"Overwrite = {overwrite}")
        save = overwrite
    model = get_model(WEIGHT_PATH)
    working_attacks = get_attacks()
    cases = ["working", "not working", "bad prediction"]
    data_to_save = {a: {c: {"mean": 0, "std": 0} for c in cases} for a in ADV}
    for a in ADV:
        print(f"Computing for {a} on {DATA_SET}")
        data_for_attacks = {c: torch.Tensor() for c in cases}
        for i,im in enumerate(exp_dataset):
            pred = torch.argmax(model.forward(im))
            mat = representation.forward(im)
            in_ell = is_in_ellipsoid(mat,
                                    get_ellipsoid_data(ellipsoids,pred,"mean"),
                                    get_ellipsoid_data(ellipsoids,pred,"std"))
            if pred == exp_labels[i]:
                if i in working_attacks[a]:
                    data_for_attacks["working"] = torch.cat((data_for_attacks["working"], in_ell.expand([1])))
                else:
                    data_for_attacks["not working"] = torch.cat((data_for_attacks["not working"], in_ell.expand([1])))
            else:
                data_for_attacks["bad prediction"] = torch.cat((data_for_attacks["bad prediction"], in_ell.expand([1])))
        if save:
            for c in cases:
                data_to_save[a][c]["mean"] = data_for_attacks[c].mean().item()
                data_to_save[a][c]["std"] = data_for_attacks[c].std().item()
            with open(f"{MATRICES_PATH}in_ellipsoid_attacked.json", 'w') as json_data:
                json.dump(data_to_save, json_data, indent=4)


def zero_dim_attacked(exp_dataset: torch.Tensor, exp_labels: torch.Tensor, representation, ellipsoids: dict, eps: float, overwrite: bool=False) -> None:
    save = True
    if os.path.exists(f"{MATRICES_PATH}zero_dim_attacked.json"):
        print("File already exists")
        print(f"Overwrite = {overwrite}")
        save = overwrite
    model = get_model(WEIGHT_PATH)
    working_attacks = get_attacks()
    cases = ["working", "not working", "bad prediction"]
    data_to_save = {a: {c: {"true image" : {"mean": 0, "std": 0}, "attacked image" : {"mean": 0, "std": 0}} for c in cases} for a in ADV}
    attacks_cls = dict(zip(ADV,
                           [torchattacks.GN(model),
                            torchattacks.FGSM(model),
                            torchattacks.BIM(model),
                            torchattacks.RFGSM(model),
                            torchattacks.PGD(model),
                            torchattacks.EOTPGD(model)]))
    # Make set of working attacks
    adv_dataset = {att: torch.zeros([len(working_attacks[att]),1,28,28]) for att in working_attacks}
    adv_labels = {att: torch.zeros([len(working_attacks[att])],dtype=torch.long) for att in working_attacks}
    for att in working_attacks:
        for i,j in enumerate(working_attacks[att]):
            adv_dataset[att][i] = exp_dataset[j][0]
            adv_labels[att][i] = exp_labels[j]
    attacked_dataset = {a: attacks_cls[a](adv_dataset[a], adv_labels[a]) for a in ADV}
    for a in ADV:
        print(f"Computing for {a} on {DATA_SET}")
        data_for_attacks = {c: {"true image": torch.Tensor(), "attacked image": torch.Tensor()} for c in cases}
        for i,im in enumerate(adv_dataset[a]):
            pred = torch.argmax(model.forward(im))
            mat = representation.forward(im)
            zeros = zero_std(mat, get_ellipsoid_data(ellipsoids,pred,"std"), eps)
            att_pred = torch.argmax(model.forward(attacked_dataset[a][i]))
            att_mat = representation.forward(attacked_dataset[a][i])
            att_zeros = zero_std(att_mat, get_ellipsoid_data(ellipsoids,att_pred,"std"), eps)
            # print(f"Zeros: {zeros.item()}")
            # print(f"Attacked zeros: {att_zeros.item()}")
            if pred == exp_labels[i]:
                if i in working_attacks[a]:
                    data_for_attacks["working"]["true image"] = torch.cat((data_for_attacks["working"]["true image"], zeros.expand([1])))
                    data_for_attacks["working"]["attacked image"] = torch.cat((data_for_attacks["working"]["attacked image"], att_zeros.expand([1])))
                else:
                    data_for_attacks["not working"]["true image"] = torch.cat((data_for_attacks["not working"]["true image"], zeros.expand([1])))
                    data_for_attacks["not working"]["attacked image"] = torch.cat((data_for_attacks["not working"]["attacked image"], att_zeros.expand([1])))
            else:
                data_for_attacks["bad prediction"]["true image"] = torch.cat((data_for_attacks["bad prediction"]["true image"], zeros.expand([1])))
                data_for_attacks["bad prediction"]["attacked image"] = torch.cat((data_for_attacks["bad prediction"]["attacked image"], att_zeros.expand([1])))
        if save:
            for c in cases:
                data_to_save[a][c]["true image"]["mean"] = data_for_attacks[c]["true image"].mean().item()
                data_to_save[a][c]["attacked image"]["mean"] = data_for_attacks[c]["attacked image"].mean().item()
                data_to_save[a][c]["true image"]["std"] = data_for_attacks[c]["true image"].std().item()
                data_to_save[a][c]["attacked image"]["std"] = data_for_attacks[c]["attacked image"].std().item()
            with open(f"{MATRICES_PATH}zero_dim_attacked.json", 'w') as json_data:
                json.dump(data_to_save, json_data, indent=4)


def zero_dim_out_of_dist(exp_dataset: torch.Tensor, representation, ellipsoids: dict, eps: float) -> None:
    # Same principle as "zero_dim_attacked", but with 'out of distribution' data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if DATA_SET == "mnist":
        out_of_dist = "fashion"
        out_dist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    elif DATA_SET == "fashion":
        out_of_dist = "mnist"
        out_dist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    else:
        raise NotImplementedError(f"Data set must be 'mnist' or 'fashion', not '{DATA_SET}'.")
    model: MLP = get_model(WEIGHT_PATH, input_shape=IN_SHAPE)
    out_dist_data, _ = subset(out_dist_train, len(exp_dataset), input_shape=IN_SHAPE)
    print(f"Model: {DATA_SET}")
    print(f"Out of distribution data: {out_of_dist} (n = {len(exp_dataset)})")
    zeros = torch.Tensor()
    in_ell = torch.Tensor()
    for im in out_dist_data:
        pred = torch.argmax(model.forward(im))
        mat = representation.forward(im)
        mean_mat = get_ellipsoid_data(ellipsoids,pred,"mean")
        std_mat = get_ellipsoid_data(ellipsoids,pred,"std")
        zeros = torch.cat((zeros, zero_std(mat, std_mat, eps).expand([1])))
        in_ell = torch.cat((in_ell, is_in_ellipsoid(mat, mean_mat, std_mat).expand([1])))
    print(f"Average zeros = {zeros.mean().item()}")
    print(f"Standard deviation = {zeros.std().item()}")
    print(f"Average in_dim = {in_ell.mean().item()}")
    print(f"Standard deviation = {in_ell.std().item()}")


def reject_predicted_attacks(exp_dataset_train: torch.Tensor,
                             exp_dataset_test: torch.Tensor,
                             exp_labels_test: torch.Tensor,
                             representation,
                             ellipsoids: dict,
                             n: int = 5000,
                             std: float = 2,
                             d1: float = 0.1,
                             d2: float = 0.1) -> None:

    model = get_model(WEIGHT_PATH)
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
                            ]))

    path_adv_examples = 'experiments/adversarial_examples/' + EXPERIMENT_PATH + '/adversarial_examples.pth'

    if os.path.exists(path_adv_examples):
        print("Loading adversarial examples...")
        attacked_dataset = torch.load(path_adv_examples)
    else:
        print("Computing adversarial examples...")
        attacked_dataset = {a: attacks_cls[a](exp_dataset_test[n:], exp_labels_test[n:]) for a in ["None"]+attacks}
        #attacked_dataset = {a: attacks_cls[a](exp_dataset_test, exp_labels_test) for a in ["None"]+attacks}
        #os.makedirs(f'experiments/adversarial_examples/{EXPERIMENT_PATH}/')
        torch.save(attacked_dataset, path_adv_examples)

    # Compute mean and std of number of (almost) zero dims
    print("Compute rejection level.")
    zeros = torch.Tensor()
    for i in range(len(exp_dataset_train[:n])):
        im = exp_dataset_train[i]
        pred = torch.argmax(model.forward(im))
        mat = representation.forward(im)

        a = get_ellipsoid_data(ellipsoids, pred, "std")
        b = zero_std(mat, a, d1)
        c = b.expand([1])

        zeros = torch.cat((zeros, c))

    reject_at = zeros.mean().item() - std*zeros.std().item()
    print(f"Will reject when 'zero dims' < {reject_at}.")
    results = []  # (Rejected, Was attacked)ygyh
    for i in range(len(exp_dataset_test[n:])):
        a = rand.choice(["None"]+attacks, p=[0.5]+[1/(2*len(attacks)) for _ in attacks])
        im = attacked_dataset[a][i]
        pred = torch.argmax(model.forward(im))
        mat = representation.forward(im)

        b = get_ellipsoid_data(ellipsoids, pred, "std")
        c = zero_std(mat, b, d2).item()

        res = ((reject_at > c), (a != "None"))
        #TODO: save images that break the deffence
        #if res[0]:

        results.append(res)
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


def reject_predicted_out_of_dist(exp_dataset_train: torch.Tensor,
                                 exp_dataset_test: torch.Tensor,
                                 representation,
                                 ellipsoids: dict,
                                 eps: float,
                                 std: float,
                                 dataset: str) -> None:
    # Same principle as with 'reject_predicted_attacks', but with 'out of distribution' data
    n = 5000  # Sample size to compute mean and std of zero dims
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if dataset == "mnist":
        out_of_dist = "fashion"
        out_dist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    elif dataset == "fashion":
        out_of_dist = "mnist"
        out_dist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    else:
        raise NotImplementedError(f"Data set must be 'mnist' or 'fashion', not '{dataset}'.")
    model: MLP = get_model(WEIGHT_PATH)
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
        mean_mat = get_ellipsoid_data(ellipsoids,pred,"mean")
        std_mat = get_ellipsoid_data(ellipsoids,pred,"std")
        zeros = torch.cat((zeros, zero_std(mat, std_mat, eps).expand([1])))
        in_ell = torch.cat((in_ell, is_in_ellipsoid(mat, mean_mat, std_mat).expand([1])))
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
        mean_mat = get_ellipsoid_data(ellipsoids,pred,"mean")
        std_mat = get_ellipsoid_data(ellipsoids,pred,"std")
        zero_dims = zero_std(mat, std_mat, eps).item()
        in_ell = is_in_ellipsoid(mat, mean_mat, std_mat).item()
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
        print('loading..')
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

    ADV = ["GN", "FGSM", "BIM", "RFGSM", "PGD",
           "EOTPGD"]  # , "FFGSM","TPGD","MIFGSM","UPGD","APGD","APGDT","DIFGSM","TIFGSM","Jitter",
    # "NIFGSM","PGDRS","SINIFGSM","VMIFGSM","VNIFGSM","CW","PGDL2","PGDRSL2","DeepFool","SparseFool","OnePixel","Pixle",
    # "FAB","AutoAttack","Square","SPSA","JSMA","EADL1","EADEN","PIFGSM","PIFGSMPP"]
    ATT = {a: i for i, a in enumerate(["None"] + ADV)}

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

    reject_predicted_attacks(exp_dataset_train,
                             exp_dataset_test,
                             exp_labels_test,
                             representation,
                             ellipsoids,
                             n=args.subset_size//4,
                             std=1,
                             d1=0.1,
                             d2=0.1)

    #reject_predicted_out_of_dist(exp_dataset_train, exp_dataset_test, representation, ellipsoids, eps=0.1, std=1.5)
    reject_predicted_out_of_dist(exp_dataset_train, exp_dataset_test, representation, ellipsoids, eps=0.1, std=2)
