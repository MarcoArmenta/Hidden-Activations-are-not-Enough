import os
import json

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST, FashionMNIST


def fashion_dataset(train_batch_size: int, test_batch_size: int, experiment="true labels"): # -> dict[str, int|str|None]:
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])
    train_dataset = FashionMNIST("./data", train=True, download=True, transform=transforms)
    test_dataset = FashionMNIST("./data", train=False, download=True, transform=transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=(experiment != "true labels"))
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=(experiment != "true labels"))

    dict_data = {"x_train": next(iter(train_loader))[0],
                 "y_train": next(iter(train_loader))[1],
                 "x_test": next(iter(test_loader))[0],
                 "y_test": next(iter(test_loader))[1],
                 "num_classes": 10,
                 "num_features": 784,
                 "train_loader": train_loader,
                 "test_loader": test_loader,
                 "train_data": train_dataset,
                 "test_data": test_dataset,
                 "input shape": (1,28,28),
                 }
    return dict_data


def mnist_dataset(): # -> dict[str, int|str|None]:
    path = f'experiments/datasets/mnist/'

    print(path + 'train_dataset.pt')
    train = torch.jit.load(path + 'train_dataset.pt')
    val = torch.jit.load(path + 'val_dataset.pt')
    test = torch.jit.load(path + 'test_dataset.pt')

    print(type(train))
    print(len(train))
    input('loding data...')
    train_loader = train
    test_loader = test
    val_loader = val

    train_dataset = MNIST("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_dataset = MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

    dict_data = {"x_train": next(iter(train_loader))[0],
                 "y_train": next(iter(train_loader))[1],
                 "x_test": next(iter(test_loader))[0],
                 "y_test": next(iter(test_loader))[1],
                 "num_classes": 10,
                 "num_features": 784,
                 "train_loader": train_loader,
                 "test_loader": test_loader,
                 "val_loader": val_loader,
                 "train_data": train_dataset,
                 "test_data": test_dataset,
                 "input shape": (1,28,28),
                 }
    return dict_data


def compute_chunk_of_matrices(data, representation, epoch: int, clas, train=True, chunk_size=10, root=None, chunk_id=0) -> None:
    if root is not None:
        directory = root + '/' + str(epoch) + '/' + str(clas) + '/'

    else:
        directory = str(epoch) + '/' + str(clas) + '/'

    directory += 'train/' if train else 'test/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    data = data[chunk_id*chunk_size:(chunk_id+1)*chunk_size]

    for i, d in enumerate(data):
        idx = chunk_id*chunk_size+i
        # if matrix was already computed, pass to next sample of data
        if os.path.exists(directory+str(idx)+'/'+'matrix.pt'):
            continue
        # if the path has not been created, then no one is working on this sample
        if not os.path.exists(directory+str(idx)+'/'):
            os.makedirs(directory+str(idx)+'/')
        # if the path has been created, someone else is already computing the matrix
        else:
            continue

        rep = representation.forward(d)
        torch.save(rep, directory+str(idx)+'/'+'matrix.pt')
