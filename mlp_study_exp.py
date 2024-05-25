import os
import torch
from torch.utils.data import DataLoader, Subset

from mlp import MLP
from representation import MlpRepresentation
from manual_training import get_architecture
from useful_functions import compute_chunk_of_matrices
from manual_training import get_dataset


class Experiment:
    def __init__(self, dict_exp) -> None:
        self.epochs: int = dict_exp["epochs"]
        self.num_samples: int = dict_exp["num_samples"]
        self.dataname: str = dict_exp["data name"].lower()
        self.weights_path = dict_exp["weights_path"]

        self.num_classes = 10

        self.data = get_dataset(self.dataname)

        self.device: str = dict_exp["device"]

    def compute_matrices_epoch_on_dataset(self, model: MLP, epoch: int, root: str, chunk_id: int, chunk_size: int = 10, train=True) -> None:
        if train:
            dataset = self.data[0]
        else:
            dataset = self.data[1]

        if isinstance(model, MLP):
            representation = MlpRepresentation(model=model, device=self.device)
        else:
            ValueError(f"Architecture not supported: {model}."
                       f"Expects MLP")

        for i in range(self.num_classes):
            train_indices = [idx for idx, target in enumerate(dataset.targets) if target in [i]]
            sub_train_dataloader = DataLoader(Subset(dataset, train_indices),
                                              batch_size=int(self.num_samples),
                                              drop_last=True)

            x_train = next(iter(sub_train_dataloader))[0] # 0 for input and 1 for label

            compute_chunk_of_matrices(x_train,
                                      representation,
                                      epoch,
                                      i,
                                      train=train,
                                      root=root,
                                      chunk_id=chunk_id,
                                      chunk_size=chunk_size)

    def values_on_epoch(self, root: str, chunk_id: int, epoch: int = 10, chunk_size: int = 10, train=True) -> None:
        path = os.getcwd()
        directory = f"{self.weights_path}"
        new_path = os.path.join(path, directory)
        model_file = f'epoch_{epoch}.pth'
        model_path = os.path.join(new_path, model_file)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        model = get_architecture()
        model.load_state_dict(state_dict)

        self.compute_matrices_epoch_on_dataset(model, epoch, root, chunk_id=chunk_id, chunk_size=chunk_size, train=train)
