"""
    This script contains functions for computing several matrices from neural networks in parallel.
"""

import gc
import os
import torch
from torch.utils.data import DataLoader, Subset
from typing import Union

from matrix_construction.matrix_computation import MlpRepresentation, ConvRepresentation_2D
from utils.utils import get_architecture, get_dataset, get_num_classes, get_device, _remap_state_dict_keys
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from utils.atomic_io import atomic_torch_save
from knowledgematrix.models.alexnet import AlexNet
from knowledgematrix.models.resnet18 import ResNet18
from knowledgematrix.models.vgg11 import VGG11


class ParallelMatrixConstruction:
    """
    A class for computing neural network matrices in parallel across multiple samples.
    Takes a dictionary of experiment parameters and handles parallel matrix computation
    for different model architectures (MLP, CNN, etc) and datasets.

    Args:
        dict_exp (dict): Dictionary containing experiment parameters including:
            - epochs: Number of training epochs
            - num_samples: Number of samples to process
            - data_name: Name of dataset
            - weights_path: Path to model weights
            - chunk_size: Size of parallel processing chunks
            - save_path: Where to save computed matrices
            - architecture_index: Index of model architecture
    """
    def __init__(self, dict_exp: dict,) -> None:
        self.epoch: int = dict_exp["epochs"]
        self.num_samples: int = dict_exp["num_samples"]
        self.dataname: str = dict_exp["data_name"].lower()
        self.weights_path: str = dict_exp["weights_path"]
        self.chunk_size: int = dict_exp['chunk_size']
        self.save_path: str = dict_exp['save_path']
        self.architecture_index: int = dict_exp['architecture_index']
        self.device = dict_exp['device']
        self.batch_size = dict_exp['batch_size']
        self.verbose = True
        self.num_classes: int = get_num_classes(self.dataname)
        self.imagenet = True if self.dataname=='imagenet' else False

        self.data = get_dataset(self.dataname, data_loader=False)[0]
        if self.dataname in ['cifar10', 'cifar100', 'imagenet']:
            self.input_shape = (3, 224, 224)
        elif self.dataname == 'mnist1d':
            self.input_shape = (1, 1, 40)
        elif self.dataname in ['mnist', 'fashion']:
            self.input_shape = (1, 28, 28)
        else:
            raise ValueError(f'Input size is not supported for {self.dataname}')

    def _validate_input_dictionary(self, dict_exp: dict):
        correct_keys = {'epochs', 'num_samples', 'data_name', 'weights_path', 'chunk_size', 'save_path', 'architecture_index',
                        'device', 'batch_size', 'verbose'}
        correct_types = [int, int, str, str, int, str, int, str, int, bool]
        keys = dict_exp.keys()
        if correct_keys != keys:
            raise ValueError(f'Dictionary of inputs should have keys {correct_keys} and got {keys}')

        i = 0
        for key, val in dict_exp.items():
            if type(val) != correct_types[i]:
                raise ValueError(f'Values of input dictionary at Key: {key}, should be {correct_types[i]}, got {type(val)}')
            i += 1


    def compute_matrices_on_dataset(
            self,
            model: Union[AlexNet, ResNet18, VGG11],
            chunk_id: int
    ) -> None:
        """
        Computes matrices for all classes in the dataset in parallel.
        """
        self._model = model  # stored for OOM recovery in compute_chunk_of_matrices
        matrix_computer = KnowledgeMatrixComputer(model, batch_size=self.batch_size, device=self.device)

        # Resolve targets, handling Subset objects (e.g., from ImageNet random_split)
        if hasattr(self.data, 'targets'):
            all_targets = self.data.targets
        elif hasattr(self.data, 'dataset'):
            underlying = self.data.dataset
            if hasattr(underlying, 'targets'):
                all_targets = [underlying.targets[i] for i in self.data.indices]
            elif hasattr(underlying, 'labels'):
                all_targets = [underlying.labels[i] for i in self.data.indices]
            else:
                all_targets = [self.data[i][1] for i in range(len(self.data))]
        else:
            all_targets = [self.data[i][1] for i in range(len(self.data))]

        for i in range(self.num_classes):
            if self.dataname == 'mnist1d':
                x_train = [self.data[idx][0] for idx, (_, target) in enumerate(self.data) if target in [i]]
            else:
                train_indices = [idx for idx, target in enumerate(all_targets) if target == i]
                if len(train_indices) == 0:
                    if self.verbose:
                        print(f"Class {i}: no samples in this split, skipping.", flush=True)
                    continue
                sub_train_dataloader = DataLoader(
                    Subset(self.data, train_indices),
                    batch_size=int(self.num_samples),
                    drop_last=False
                )

                x_train = next(iter(sub_train_dataloader))[0]  # 0 for input and 1 for label

            self.compute_chunk_of_matrices(
                data=x_train,
                matrix_computer=matrix_computer,
                out_class=i,
                chunk_id=chunk_id
            )

        return True

    def values_on_epoch(self, chunk_id: int) -> None:
        """
        Loads the model state dictionary and computes matrices for all classes in the dataset.
        """
        path = os.getcwd()
        directory = f"{self.weights_path}"
        new_path = os.path.join(path, directory)
        model_file = "pretrained-weights.pth" if self.imagenet else f"epoch_{self.epoch}.pth"
        model_path = os.path.join(new_path, model_file)

        checkpoint = torch.load(model_path, map_location=self.device)
        # Support both new format (full checkpoint dict) and legacy format (bare state_dict)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Use pretrained=False to avoid internet downloads on compute nodes.
        # The state_dict loaded below overwrites all weights anyway.
        model = get_architecture(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            architecture_index=self.architecture_index,
            pretrained=False,
            freeze_features=False,
        )
        model.to(self.device)
        state_dict = _remap_state_dict_keys(model, state_dict)
        model.load_state_dict(state_dict)

        return self.compute_matrices_on_dataset(model, chunk_id=chunk_id)

    def compute_chunk_of_matrices(
            self,
            data: torch.Tensor,
            matrix_computer: KnowledgeMatrixComputer,
            out_class: int,
            chunk_id: int,
    ) -> None:
        """
        Computes and saves matrices induced by a neural network for a chunk of data samples from a given class.

        Args:
            data (torch.Tensor): Input data tensor containing samples from a single class
            matrix_computer (MlpRepresentation|ConvRepresentation_2D): Object that computes the matrix representation
                of the neural network at each input point
            out_class (int): The class label for this chunk of data
            chunk_id (int, optional): Index of the chunk being processed. Defaults to 0.

        The matrices are saved to disk in the following structure:
        save_path/class_label/sample_index/matrix.pt
        """
        directory = f"{self.save_path if self.save_path is not None else ''}/{out_class}/"
        os.makedirs(directory, exist_ok=True)

        data = data[chunk_id*self.chunk_size:(chunk_id+1)*self.chunk_size].to(self.device)

        for i, d in enumerate(data):
            idx = chunk_id*self.chunk_size+i
            if self.verbose:
                print(f'Matrix: {i}/{data.shape[0]}. Chunk: {chunk_id}. Class: {out_class}', flush=True)

            root = os.path.join(directory, f"{idx}")
            matrix_path = os.path.join(root, "matrix.pt")
            if os.path.exists(matrix_path):
                # if matrix was already computed, pass to next sample of data
                continue
            for attempt in range(4):
                try:
                    matrix = matrix_computer.forward(d)
                    break
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower() or attempt == 3:
                        raise
                    new_bs = max(1, matrix_computer.batch_size // 2)
                    print(f"[OOM] Halving batch_size: {matrix_computer.batch_size} -> {new_bs}", flush=True)
                    del matrix_computer
                    gc.collect()
                    torch.cuda.empty_cache()
                    matrix_computer = KnowledgeMatrixComputer(self._model, batch_size=new_bs, device=self.device)
            os.makedirs(root, exist_ok=True)
            atomic_torch_save(matrix, matrix_path)
            del matrix
            gc.collect()
            torch.cuda.empty_cache()
