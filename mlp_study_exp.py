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


    def average_std_distance_matrices_train_test(self, model: MLP, epoch: int, root: str, chunk_id: int) -> None:
        train_dataset = self.data["train_data"]
        test_dataset = self.data["test_data"]

        print("Constructing representation.", flush=True)

        if isinstance(model, MLP):
            representation = MlpRepresentation(model=model, device=self.device)
        else:
            ValueError("Architecture not supported!")

        print("Representation constructed.", flush=True)
        print("Computing matrices...", flush=True)

        for i in range(self.data["num_classes"]):
            train_indices = [idx for idx, target in enumerate(train_dataset.targets) if target in [i]]
            sub_train_dataloader = DataLoader(Subset(train_dataset, train_indices),
                                              batch_size=int(self.num_samples),
                                              drop_last=True)

            test_indices = [idx for idx, target in enumerate(test_dataset.targets) if target in [i]]
            sub_test_dataloader = DataLoader(Subset(test_dataset, test_indices),
                                             batch_size=int(self.num_samples),
                                             drop_last=True)

            x_train = next(iter(sub_train_dataloader))[0]  # 0 for input and 1 for label
            x_test = next(iter(sub_test_dataloader))[0]

            compute_chunk_of_matrices(x_train,
                                      representation,
                                      epoch,
                                      i,
                                      train=True,
                                      root=root,
                                      chunk_id=chunk_id)

            compute_chunk_of_matrices(x_test,
                                      representation,
                                      epoch,
                                      i,
                                      train=False,
                                      root=root,
                                      chunk_id=chunk_id)

    def compute_matrices_epoch_on_dataset(self, model: MLP, epoch: int, root: str, chunk_id: int, chunk_size: int = 10, train=True) -> None:
        if train:
            dataset = self.data[0]
        else:
            dataset = self.data[1]

        print("Constructing representation", flush=True)

        if isinstance(model, MLP):
            representation = MlpRepresentation(model=model, device=self.device)
        else:
            ValueError(f"Architecture not supported: {model}."
                       f"Expects MLP")

        print("Representation constructed", flush=True)
        print("Computing matrices", flush=True)

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

        print("Chunk completed...", flush=True)

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
