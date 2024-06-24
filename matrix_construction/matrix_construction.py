"""
    This script contains functions for computing several matrices from neural networks in parallel.
"""
import os
import torch
from torch.utils.data import DataLoader, Subset

from model_zoo.mlp import MLP
from matrix_construction.representation import MlpRepresentation
from utils.utils import get_architecture, get_dataset


def compute_chunk_of_matrices(data: torch.Tensor,
                              representation: MLP,
                              clas: int,
                              chunk_size: int = 10,
                              save_path=None,
                              chunk_id: int = 0) -> None:
    """
    Given a subset of data and an MlpRepresentation, it computes and saves accordingly
    the induced matrices in the corresponding chunk of samples in data
    """
    if save_path is not None:
        directory = save_path + '/' + str(clas) + '/'

    else:
        directory = '/' + str(clas) + '/'

    os.makedirs(directory, exist_ok=True)

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


class MatrixConstruction:
    def __init__(self, dict_exp) -> None:
        self.epoch: int = dict_exp["epochs"]
        self.num_samples: int = dict_exp["num_samples"]
        self.dataname: str = dict_exp["data_name"].lower()
        self.weights_path = dict_exp["weights_path"]
        self.chunk_size = dict_exp['chunk_size']
        self.save_path = dict_exp['save_path']
        self.architecture_index = dict_exp['architecture_index']
        self.residual = dict_exp['residual']
        self.dropout = dict_exp['dropout']

        self.num_classes = 10
        self.data = get_dataset(self.dataname, data_loader=False)[0]

    def compute_matrices_on_dataset(self, model: MLP, chunk_id: int) -> None:
        if isinstance(model, MLP):
            representation = MlpRepresentation(model=model)
        else:
            raise ValueError(f"Architecture not supported: {model}."
                             f"Expects MLP")

        for i in range(self.num_classes):
            train_indices = [idx for idx, target in enumerate(self.data.targets) if target in [i]]
            sub_train_dataloader = DataLoader(Subset(self.data, train_indices),
                                              batch_size=int(self.num_samples),
                                              drop_last=True)

            x_train = next(iter(sub_train_dataloader))[0] # 0 for input and 1 for label

            compute_chunk_of_matrices(x_train,
                                      representation,
                                      i,
                                      save_path=self.save_path,
                                      chunk_id=chunk_id,
                                      chunk_size=self.chunk_size)

    def values_on_epoch(self, chunk_id: int) -> None:
        path = os.getcwd()
        directory = f"{self.weights_path}"
        new_path = os.path.join(path, directory)
        model_file = f'epoch_{self.epoch}.pth'
        model_path = os.path.join(new_path, model_file)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        input_shape = (3, 32, 32) if self.dataname == 'cifar10' else (1, 28, 28)
        model = get_architecture(architecture_index=self.architecture_index,
                                 residual=self.residual,
                                 input_shape=input_shape,
                                 dropout=self.dropout,
                                 )
        model.load_state_dict(state_dict)

        self.compute_matrices_on_dataset(model, chunk_id=chunk_id)
