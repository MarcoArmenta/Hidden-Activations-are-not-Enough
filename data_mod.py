"""
    Modify data (in the input space) and see where it maps in the matrix space.
"""

import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from representation import MlpRepresentation
from mlp import MLP
from manual_training import get_architecture


torch.manual_seed(123456)

DATA_SET = "mnist"
IN_SHAPE = (1,28,28) if DATA_SET == "mnist" or DATA_SET == "fashion" else (3,32,32)


def interpolate(data1: torch.Tensor, data2: torch.Tensor, alpha: float=0.5) -> torch.Tensor:
    return (1-alpha)*data1 + alpha*data2

def rotate(data: torch.Tensor, angle: float) -> torch.Tensor:
    return transforms.functional.rotate(data,angle)

def get_ellipsoid_data(ellipsoids: dict, result: torch.Tensor, param: str) -> torch.Tensor:
    """

    :param: ellipsoids: matrix statistics dictionary with keys the classes and mean and std
    :param: result
    :param: ellipsoids

    """
    return torch.Tensor(ellipsoids[str(result.item())][param])

def is_in_ellipsoid(matrix: torch.Tensor,
                    ellipsoid_mean: torch.Tensor,
                    ellipsoid_std: torch.Tensor,
                    std: float=2) -> torch.LongTensor:
    # Gives the opposite of what would be intuitive (ie more non zero if attacked)
    low_bound = torch.le(ellipsoid_mean-std*ellipsoid_std, matrix)
    up_bound = torch.le(matrix, ellipsoid_mean+std*ellipsoid_std)
    return torch.count_nonzero(torch.logical_and(low_bound, up_bound))


def zero_std(matrix: torch.Tensor,
             ellipsoid_std: torch.Tensor,
             eps: float=0) -> torch.LongTensor:
    return torch.count_nonzero(torch.logical_and((ellipsoid_std <= eps), (matrix > eps)))  # ¬(P => Q) <==> P ∧ ¬Q

def get_model(path: str):
    weight_path = torch.load(path, map_location=torch.device('cpu'))
    model = get_architecture()
    model.load_state_dict(weight_path)
    return model

def subset(train_set, length: int, input_shape=(1, 28, 28)):
    idx = torch.randint(low=0,high=len(train_set),size=[length],generator=torch.Generator("cpu"))
    exp_dataset = torch.zeros([length,input_shape[0],input_shape[1],input_shape[2]])
    exp_labels = torch.zeros([length],dtype=torch.long)
    for i,j in enumerate(idx):
        exp_dataset[i] = train_set[j][0]
        exp_labels[i] = train_set.targets[j]
    return (exp_dataset, exp_labels)

def show_img(digit5: torch.Tensor, digit0: torch.Tensor) -> None:
    _, axes = plt.subplots(2, 3, figsize=(15, 15))

    axes[0][0].imshow(digit5[0], cmap='gray')
    axes[0][0].set_title(f'Digit 5')
    axes[0][1].imshow(digit0[0], cmap='gray')
    axes[0][1].set_title(f'Digit 0')
    axes[0][2].imshow(interpolate(digit5[0], digit0[0]), cmap='gray')
    axes[0][2].set_title(f'Average of 5 and 0')

    axes[1][0].imshow(rotate(digit5, -30)[0], cmap='gray')
    axes[1][0].set_title(f'Digit 5, rotated 30°')
    axes[1][1].imshow(rotate(digit5, -60)[0], cmap='gray')
    axes[1][1].set_title(f'Digit 5, rotated 60°')
    axes[1][2].imshow(rotate(digit5, -90)[0], cmap='gray')
    axes[1][2].set_title(f'Digit 5, rotated 90°')
    
    for i in range(2):
        for j in range(3): axes[i][j].axis("off")
    plt.show()

def ellipsoid_while_rotating(train_set: torch.Tensor, overwrite: bool=False) -> None:
    path = f"data/out_of_distribution/test_sam/{DATA_SET}_vs_{DATA_SET}/{DATA_SET}_sgd_small-mlp_0.01_8_vs_{DATA_SET}/"
    save = True
    if os.path.exists(f"{path}rotate.json"):
        print("File already exists")
        print(f"Overwrite = {overwrite}")
        save = overwrite
    exp_dataset, exp_labels = subset(train_set, 100, IN_SHAPE)
    model = get_model(f"data/MLP_weights/{DATA_SET}/sgd/true labels/0.01/8/epoch_100.pth")
    representation = MlpRepresentation(model)
    ellipsoids_file = open(f"{path}true labels/100/matrix_statistics.json")
    ellipsoids: dict = json.load(ellipsoids_file)

    data = {f"{i} (input {exp_labels[i]})": {"predictions": [], "in_start_ellipsoid": [], "in_current_ellipsoid": []} for i in range(len(exp_dataset))}
    for i,im in enumerate(exp_dataset):
        pred = torch.argmax(model.forward(im))
        for ang in range(10):
            rot_im = rotate(im, 36*ang)  # 36*ang = 360*(ang/10)
            mat = representation.forward(rot_im)
            rot_pred = torch.argmax(model.forward(rot_im))
            in_start_ellipsoid = is_in_ellipsoid(mat,
                                     get_ellipsoid_data(ellipsoids, pred, "mean"),
                                     get_ellipsoid_data(ellipsoids, pred, "std"))
            in_current_ellipsoid = is_in_ellipsoid(mat,
                                     get_ellipsoid_data(ellipsoids, rot_pred, "mean"),
                                     get_ellipsoid_data(ellipsoids, rot_pred, "std"))
            data[f"{i} (input {exp_labels[i]})"]["predictions"].append(rot_pred.item())
            data[f"{i} (input {exp_labels[i]})"]["in_start_ellipsoid"].append(in_start_ellipsoid.item())
            data[f"{i} (input {exp_labels[i]})"]["in_current_ellipsoid"].append(in_current_ellipsoid.item())
    if save:
        with open(f"{path}rotate.json", "w") as json_data:
            json.dump(data, json_data, indent=4)


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    ellipsoid_while_rotating(train_set)
    # distance_to_matrix(train_set)
    # show_img(digit3, digit0)
    # show_mnist_digit(True)
