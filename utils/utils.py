import os
import inspect
import torch
import torch.nn as nn
import json
import random
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import shutil
from pathlib import Path
from typing import Union

from model_zoo.mlp import MLP
from model_zoo.cnn import CNN_2D
from knowledgematrix.models.alexnet import AlexNet
from knowledgematrix.models.resnet18 import ResNet18
from knowledgematrix.models.vgg11 import VGG11
from constants.constants import ARCHITECTURES


from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from typing import Tuple, Optional, Callable


def _remap_state_dict_keys(model, state_dict):
    """Remap state_dict keys when layer indices differ.

    The knowledgematrix VGG11 builds different layer sequences for
    pretrained=True vs pretrained=False (extra AdaptiveAvgPool2d),
    causing FC layer indices to shift. This remaps saved keys to
    match the model by pairing parameters in order by shape.
    """
    model_sd = model.state_dict()
    if set(state_dict.keys()) == set(model_sd.keys()):
        return state_dict  # Keys already match

    model_keys = list(model_sd.keys())
    saved_keys = list(state_dict.keys())

    if len(model_keys) != len(saved_keys):
        raise RuntimeError(
            f"Cannot remap state_dict: model has {len(model_keys)} params, "
            f"saved has {len(saved_keys)}"
        )

    remapped = {}
    for mk, sk in zip(model_keys, saved_keys):
        if model_sd[mk].shape != state_dict[sk].shape:
            raise RuntimeError(
                f"Shape mismatch during remap: model {mk} {model_sd[mk].shape} "
                f"vs saved {sk} {state_dict[sk].shape}"
            )
        remapped[mk] = state_dict[sk]

    return remapped


class ImageNetVal(Dataset):
    """
    Custom Dataset for ImageNet validation set, loading images and ground truth labels.

    Args:
        root (str): Path to validation images (e.g., /datashare/imagenet/ILSVRC2012/val/).
        gt_path (str): Path to ILSVRC2012_validation_ground_truth.txt.
        transform (Callable, optional): Transforms for images (e.g., Resize, ToTensor, Normalize).
        target_transform (Callable, optional): Transforms for labels.

    Loads 50,000 validation images with labels (0-999, matching ILSVRC2012_ID - 1).
    Images are sorted alphabetically to match ground truth order.
    """
    def __init__(
        self,
        root: str,
        gt_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.gt_path = gt_path

        # Load ground truth labels (50,000 lines, ILSVRC2012_ID 1-1000)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found at {gt_path}")
        with open(gt_path, 'r') as f:
            self.labels = [int(line.strip()) - 1 for line in f.readlines()]  # Convert to 0-indexed (0-999)

        if len(self.labels) != 50000:
            raise ValueError(f"Expected 50,000 labels, got {len(self.labels)} in {gt_path}")

        # Get sorted image paths (alphabetical order to match ground truth)
        self.image_paths = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.jpeg', '.jpg'))]
        )

        if len(self.image_paths) != 50000:
            raise ValueError(f"Expected 50,000 validation images, got {len(self.image_paths)} in {root}")

        # Verify image filenames (e.g., ILSVRC2012_val_00000001.JPEG)
        for i, path in enumerate(self.image_paths[:5], 1):
            expected = f"ILSVRC2012_val_{str(i).zfill(8)}.JPEG"
            if os.path.basename(path) != expected:
                print(f"Warning: Image {path} does not match expected {expected}. Labels may misalign.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        try:
            image = Image.open(self.image_paths[index]).convert('RGB')
        except Exception as e:
            print(f"Error loading image {self.image_paths[index]}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))  # Fallback black image

        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


def get_imagenet_val_dataset(
    data_path: str = '/datashare/imagenet/ILSVRC2012',
    transform: Optional[transforms.Compose] = None,
    batch_size: int = 32
) -> Tuple[DataLoader, ImageNetVal]:
    """
    Loads ImageNet validation dataset as a proxy for test set with real labels.

    Args:
        data_path (str): Base path (val in data_path/val/, gt in data_path/ILSVRC2012_devkit_t12/data/).
        transform (transforms.Compose, optional): Image transforms.
        batch_size (int): Batch size for DataLoader.

    Returns:
        Tuple[DataLoader, ImageNetVal]: Validation DataLoader and dataset.
    """
    if transform is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),  # Standard ImageNet preprocessing
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    val_root = os.path.join(data_path, 'validation')
    if not os.path.isdir(val_root):
        val_root = os.path.join(data_path, 'val')
    gt_path = os.path.join(data_path, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')

    # Auto-detect ImageFolder (class subdirs) vs flat format
    subdirs = [d for d in os.listdir(val_root) if os.path.isdir(os.path.join(val_root, d))]
    if len(subdirs) > 10:
        # ImageFolder format (class subdirectories)
        val_set = torchvision.datasets.ImageFolder(val_root, transform=transform)
        val_set.labels = [s[1] for s in val_set.samples]
    else:
        # Flat format with ground truth file
        val_set = ImageNetVal(
            root=val_root,
            gt_path=gt_path,
            transform=transform
        )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Adjust for Nibi cluster I/O
        pin_memory=True  # Faster GPU transfers
    )

    return val_loader, val_set

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_imagenet1k_loaders(
    root_dir="/datashare/imagenet/ILSVRC2012",
    batch_size=256,
    num_workers=4,
    image_size=224,
):
    """
    Loads the ImageNet-1K (ILSVRC2012) dataset from Nibi cluster.
    Expects structure:
        root_dir/train/<class_name>/*.JPEG
        root_dir/val/<class_name>/*.JPEG
    """

    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "validation")
    if not os.path.isdir(val_dir):
        val_dir = os.path.join(root_dir, "val")

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation images.", flush=True)
    print(f"Number of classes: {len(train_dataset.classes)}", flush=True)

    return train_loader, val_loader
'''
if __name__ == "__main__":
    train_loader, val_loader = get_imagenet1k_loaders(
        root_dir="/datashare/imagenet/ILSVRC2012",
        batch_size=128,
        num_workers=4,
    )

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}", flush=True)
    
'''


def get_imagenet_loaders(
    root_dir="/datashare/imagenet/winter21_whole",
    batch_size=128,
    num_workers=8,
    image_size=224,
):
    """
    Loads ImageNet-style dataset located at root_dir (e.g., /datashare/imagenet/winter21_whole/).

    Each subdirectory of root_dir should correspond to one class.
    """

    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Define training and validation transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Load datasets (assuming a single folder with all classes)
    train_dataset = datasets.ImageFolder(root=root_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=root_dir, transform=val_transform)

    # Split into train/val sets (optional, if dataset isn’t already split)
    # 90% train, 10% validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation images.")
    print(f"Number of classes: {len(train_dataset.dataset.classes)}")

    return train_loader, val_loader


#if __name__ == "__main__":
#    train_loader, val_loader = get_imagenet_loaders()

    # Example: inspect one batch
#    images, labels = next(iter(train_loader))
#    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")



def get_device(trial_number: int = 1, gpu_count: int = 1) -> torch.device:
    """
        Returns:
            The device to use.
    """
    if gpu_count == 0:
        print("DEVICE: cpu", flush=True)
        return torch.device("cpu")
        # Assign GPU based on trial number (e.g., trial 0 -> cuda:0, trial 1 -> cuda:1)

    if torch.cuda.is_available():
        print("DEVICE: cuda")
        gpu_id = trial_number % gpu_count
        print(f"DEVICE: cuda:{gpu_id}", flush=True)
        return torch.device(f"cuda:{gpu_id}")
    elif torch.backends.mps.is_available():
        print("DEVICE: mps", flush=True)
        return torch.device("mps")
    else:
        print("DEVICE: cpu", flush=True)
        return torch.device("cpu")


def get_architecture(
        input_shape = (1, 28, 28),
        num_classes:int = 10,
        architecture_index:int = 0,
        pretrained = True,
        freeze_features = True
    ) -> Union[MLP, CNN_2D, ResNet18, AlexNet, VGG11]:
    """
        Args:
            input_shape: The shape of the input data.
            num_classes: The number of classes in the dataset.
            architecture_index: The index of the architecture to use (See constants/constants.py).
            residual: Whether to use residual connections.
            dropout: Whether to use dropout.
        Returns:
            The architecture to use.

    if architecture_index <= 7 and architecture_index >= 0:
        model = MLP(
            input_shape = input_shape,
            num_classes = num_classes,
            hidden_sizes = ARCHITECTURES[architecture_index],
            residual = residual,
            bias = True,
            dropout = dropout,
        )
    """
    if architecture_index == -4:
        print("Lenet LOADED", flush=True)
        model = CNN_2D(input_shape=input_shape,
                       num_classes=num_classes,
                       channels=(6, 16),
                       padding=((2, 2), (0, 0)),
                       fc=(784, 84),
                       kernel_size=((5, 5), (5, 5)),
                       bias=False,
                       activation="relu",
                       pooling="avg")
    elif architecture_index in (-3, -2, -1):
        arch_map = {-3: AlexNet, -2: ResNet18, -1: VGG11}
        cls = arch_map[architecture_index]

        if pretrained:
            # Load pretrained torchvision model from torch hub cache, then pass
            # it to the knowledgematrix constructor via pretrained_model to avoid
            # internet downloads on compute nodes.
            #
            # The knowledgematrix constructors have a bug: they check
            # isinstance(pretrained_model, vgg11) where vgg11 is a function,
            # not a type, which raises TypeError. We monkey-patch the module
            # reference to the actual torchvision class so isinstance works.
            import importlib
            from torchvision.models import alexnet as tv_alexnet, resnet18 as tv_resnet18, vgg11 as tv_vgg11
            from torchvision.models.alexnet import AlexNet as _TVAlexNet
            from torchvision.models.resnet import ResNet as _TVResNet
            from torchvision.models.vgg import VGG as _TVVGG

            tv_fn_map = {-3: tv_alexnet, -2: tv_resnet18, -1: tv_vgg11}
            tv_cls_map = {-3: _TVAlexNet, -2: _TVResNet, -1: _TVVGG}
            km_mod_map = {
                -3: ('knowledgematrix.models.alexnet', 'alexnet'),
                -2: ('knowledgematrix.models.resnet18', 'resnet18'),
                -1: ('knowledgematrix.models.vgg11', 'vgg11'),
            }

            tv_fn = tv_fn_map[architecture_index]
            try:
                tv_model = tv_fn(weights='DEFAULT')
            except Exception as e:
                raise RuntimeError(
                    "Torchvision pretrained weights not cached and download failed.\n"
                    "Compute nodes have no internet. Pre-cache on a login node:\n"
                    "  python -c \"import torchvision; torchvision.models.vgg11(weights='DEFAULT')\"\n"
                    f"Original error: {e}"
                ) from e

            mod_name, attr_name = km_mod_map[architecture_index]
            km_mod = importlib.import_module(mod_name)
            orig_ref = getattr(km_mod, attr_name)
            setattr(km_mod, attr_name, tv_cls_map[architecture_index])
            try:
                model = cls(input_shape, num_classes, pretrained=True, pretrained_model=tv_model)
            finally:
                setattr(km_mod, attr_name, orig_ref)
        else:
            model = cls(input_shape, num_classes, pretrained=False)

        if freeze_features and pretrained:
            for layer in model.layers:
                if isinstance(layer, nn.Conv2d):
                    for param in layer.parameters():
                        param.requires_grad = False
        model.input_shape = input_shape
    else:
        model = CNN_2D(
            input_shape = input_shape,
            num_classes = num_classes,
            channels = ARCHITECTURES[architecture_index][0],
            fc = ARCHITECTURES[architecture_index][1]
        )
    return model


def get_model(
        path: Path,
        architecture_index: int,
        input_shape,
        num_classes: int,
        device: torch.device = torch.device('cpu'),
    ) -> Union[MLP, CNN_2D, ResNet18, AlexNet, VGG11]:
    """ 
        Args:
            path: The path to the model weights.
            architecture_index: The index of the architecture to use (See constants/constants.py).
            residual: Whether to use residual connections.
            input_shape: The shape of the input data.
            num_classes: The number of classes in the dataset.
            dropout: Whether to use dropout.
        Returns:
            The model to use.
    """
    checkpoint = torch.load(str(path), map_location=torch.device(device), weights_only=False)
    # Support both new format (full checkpoint dict) and legacy format (bare state_dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    # Use pretrained=False to avoid internet downloads on compute nodes.
    # The state_dict loaded below overwrites all weights anyway.
    model = get_architecture(
                architecture_index = architecture_index,
                input_shape = input_shape,
                num_classes = num_classes,
                pretrained = False,
                freeze_features = False,
            ).to(device)
    state_dict = _remap_state_dict_keys(model, state_dict)
    model.load_state_dict(state_dict)
    return model


def get_input_shape(
        data_set: str
    ):
    """
        Args:
            data_set: The dataset to use.
        Returns:
            The input shape. Default is imagenet size (3, 224, 224)
    """
    if data_set == 'mnist' or data_set == 'fashion':
        return (1, 28, 28)
    else:
        return (3, 224, 224)


def get_num_classes(
        data_set: str
    ) -> int:
    """
        Args:
            data_set: The dataset to use.
        Returns:
            The number of classes.
    """
    if data_set == 'cifar100':
        return 100
    elif data_set == 'imagenet':
        return 1000
    else:
        return 10


def get_dataset(
        data_set: str, 
        batch_size:int = 32,
        data_loader:bool = True,
        data_path:Union[str, None] = None
    ) -> tuple:
    """
        Args:
            data_set: The dataset to use.
            batch_size: The batch size.
            data_loader: Whether to use a data loader.
            data_path: The path to the data.
        Returns:
            The dataset to use.
    """
    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
    ])
    if data_path is None:
        data_path = './data'
    else:
        data_path = data_path + '/data'

    if data_set == 'mnist':
        try:
            train_set = torchvision.datasets.MNIST(
                root = data_path,
                train = True,
                transform = transform,
                download = False
            )
            test_set = torchvision.datasets.MNIST(
                root = data_path,
                train = False,
                transform = transform,
                download = False
            )
        except RuntimeError:
            raise RuntimeError(
                f"MNIST not found at {data_path}. Pre-download on the login node first."
            )
    elif data_set == 'fashion':
        try:
            train_set = torchvision.datasets.FashionMNIST(
                root = data_path,
                train = True,
                transform = transform,
                download = False
            )
            test_set = torchvision.datasets.FashionMNIST(
                root = data_path,
                train = False,
                transform = transform,
                download = False
            )
        except RuntimeError:
            raise RuntimeError(
                f"FashionMNIST not found at {data_path}. Pre-download on the login node first."
            )
    elif data_set == 'cifar10':
        # Use ImageNet normalization for pretrained models
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.Resize(224),  # Resize to match AlexNet input
            transforms.RandomHorizontalFlip(),  # Optional augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),  # Resize to match AlexNet input
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        try:
            train_set = CIFAR10(root=data_path or './data', train=True, download=False, transform=train_transform)
            test_set = CIFAR10(root=data_path or './data', train=False, download=False, transform=test_transform)
        except RuntimeError:
            raise RuntimeError(
                f"CIFAR-10 not found at {data_path}. Pre-download on the login node first."
            )

    elif data_set == 'cifar100':
        # Use ImageNet normalization for pretrained models
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.Resize(224),  # Resize to match AlexNet input
            transforms.RandomHorizontalFlip(),  # Optional augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),  # Resize to match AlexNet input
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        try:
            train_set = CIFAR100(root=data_path or './data', train=True, download=False, transform=train_transform)
            test_set = CIFAR100(root=data_path or './data', train=False, download=False, transform=test_transform)
        except RuntimeError:
            raise RuntimeError(
                f"CIFAR-100 not found at {data_path}. Pre-download on the login node first."
            )

    elif data_set == 'imagenet':
        imagenet_root = '/datashare/imagenet/ILSVRC2012'
        _, val_set = get_imagenet_val_dataset(imagenet_root, batch_size=batch_size)
        # Split the 50k validation set into 25k train / 25k test with a fixed seed
        generator = torch.Generator().manual_seed(42)
        train_set, test_set = random_split(val_set, [25000, 25000], generator=generator)
    else:
        print(f"Dataset {data_set} not supported...")
        exit(1)

    if data_loader:
        train_loader = torch.utils.data.DataLoader(
            dataset = train_set, 
            batch_size = batch_size, 
            shuffle = True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset = test_set, 
            batch_size = batch_size, 
            shuffle = False
        )
        return train_loader, test_loader
    else:
        return train_set, test_set


def find_matrices(base_dir: str, num_classes: int = 10):
    """
        Finds the matrices for the given base directory.

        Args:
            base_dir: The base directory to search for matrices.
            num_classes: The number of classes (e.g., 10 for CIFAR-10, 100 for CIFAR-100).
        Returns:
            A dictionary with the keys being the class indices and the values
            being the paths to the matrices.
    """
    matrix_paths = {}
    for j in range(num_classes):
        matrices_path = os.path.join(base_dir, str(j))  # Only use training data
        if os.path.exists(matrices_path):
            for i in os.listdir(matrices_path):  # Iterating through each 'i' subdirectory
                matrix_file_path = os.path.join(matrices_path, i, 'matrix.pt')
                if os.path.isfile(matrix_file_path):  # Check if matrix.pt exists
                    if j not in matrix_paths:
                        matrix_paths[j] = [matrix_file_path]
                    else:
                        matrix_paths[j].append(matrix_file_path)
    return matrix_paths


def compute_statistics(
        matrix_paths
    ):
    """
        Computes the statistics for the given matrix paths.

        Args:
            matrix_paths: A dictionary with the keys being the class indices and 
            the values being the paths to the matrices.
        Returns:
            A dictionary with the keys being the class indices and the values 
            being the statistics (mean and std).
    """
    statistics = {}
    for j, paths in matrix_paths.items():
        print(f"idx: {j}, paths: {paths}", flush=True)
        matrices = [torch.load(path, map_location=torch.device('cpu'), weights_only=False) for path in paths]
        print(f'Num of matrices: {len(matrices)}', flush=True)
        #matrices = [torch.load(path).cpu() for path in paths]
        # Stack all matrices to compute statistics across all matrices in a subfolder
        stacked_matrices = torch.stack(matrices)
        print(f'Matrices shape: {stacked_matrices.shape}', flush=True)
        # Compute mean and std across the stacked matrices
        mean_matrix = torch.mean(stacked_matrices, dim=0)
        std_matrix = torch.std(stacked_matrices, dim=0)
        # Store the computed statistics
        statistics[j] = {'mean': mean_matrix, 'std': std_matrix}

    return statistics


def compute_train_statistics(
        experiment_name:str = None,
        path = None,
        num_classes: int = 10
    ) -> None:
    """
        Computes the statistics for the given path.

        Args:
            experiment_name: The name of the experiment.
            path: The path to the matrices.
            num_classes: The number of classes in the dataset.
    """
    if path is not None:
        original_matrices_path = f'{path}/experiments/{experiment_name}/matrices/'
    else:
        original_matrices_path = f'experiments/{experiment_name}/matrices/'

    print(f'Path to matrices: {original_matrices_path}', flush=True)
    original_matrices_paths = find_matrices(original_matrices_path, num_classes=num_classes)
    print(f'Matrices paths: {original_matrices_paths}', flush=True)
    statistics = compute_statistics(original_matrices_paths)

    # Convert tensors to lists (or numbers) for JSON serialization
    for subfolder, stats in statistics.items():
        for key, tensor in stats.items():
            if tensor.numel() == 1:  # If the tensor has only one element, convert to a Python scalar
                stats[key] = tensor.item()
            else:  # Otherwise, convert to a list
                stats[key] = tensor.tolist()

    print(f'statistics after to list and item: {statistics}', flush=True)

    os.makedirs(f'experiments/{experiment_name}/matrices/', exist_ok=True)
    with open(f'experiments/{experiment_name}/matrices/matrix_statistics.json', 'w') as json_file:
        json.dump(statistics, json_file, indent=4)


def get_ellipsoid_data(
        ellipsoids: dict, 
        result: torch.Tensor, 
        param: str
    ) -> torch.Tensor:
    """
        Args:
            ellipsoids: matrix statistics dictionary with keys the classes and mean and std
            result: predicted class by the model
            param: the parameter to get from the ellipsoid statistics
        Returns:
            The boundary of the ellipsoid.
    """
    return torch.Tensor(ellipsoids[str(result.item())][param])


def is_in_ellipsoid(
        matrix: torch.Tensor,
        ellipsoid_mean: torch.Tensor,
        ellipsoid_std: torch.Tensor,
        std: float = 2
    ) -> torch.LongTensor:
    """
        Args:
            matrix: the matrix to check.
            ellipsoid_mean: the mean of the ellipsoid.
            ellipsoid_std: the std of the ellipsoid.
            std: increase the size of the ellipsoid by this factor.
        Returns:
            The number of elements in the ellipsoid.
    """
    low_bound = torch.le(ellipsoid_mean-std*ellipsoid_std, matrix)
    up_bound = torch.le(matrix, ellipsoid_mean+std*ellipsoid_std)
    return torch.count_nonzero(torch.logical_and(low_bound, up_bound))


def zero_std(
        matrix: torch.Tensor,
        ellipsoid_std: torch.Tensor,
        epsilon: float = 0
    ) -> torch.LongTensor:
    """
        Args:
            matrix: the matrix to check.
            ellipsoid_std: the std of the ellipsoid.
            epsilon: the threshold.
        Returns:
            The number of elements in the ellipsoid.
    """
    return torch.count_nonzero(torch.logical_and((ellipsoid_std.detach().cpu() <= epsilon), (matrix.detach().cpu() > epsilon)))

def subset(
        train_set,
        length: int,
        input_shape = (1, 28, 28),
        seed: int = 42
    ):
    """
        Make a random subset of the training set of the given length.
        If the length is greater or equal to the length of the training set,
        this function will shuffle the training set.
        Args:
            train_set: the training set (MNIST, CIFAR-10, etc.).
            length: the length of the subset.
            input_shape: the shape of the input.
            seed: random seed for reproducibility (default 42).
        Returns:
            A random subset of the training set of the given length.
    """
    if length > len(train_set):
        length = len(train_set)
    rng = random.Random(seed)
    idx = rng.sample(range(len(train_set)), length)
    exp_dataset = torch.zeros([length, input_shape[0], input_shape[1], input_shape[2]])
    exp_labels = torch.zeros([length], dtype=torch.long)
    for i, j in enumerate(idx):
        exp_dataset[i] = train_set[j][0]
        exp_labels[i] = train_set[j][1]
    return exp_dataset, exp_labels


def zip_and_cleanup(
        src_directory: str,
        zip_filename: str,
        clean:bool = True
    ) -> None:
    """
        Args:
            src_directory: the source directory.
            zip_filename: the filename of the zip file.
            clean: whether to clean the source directory.
    """
    from utils.data_integrity import zip_and_verify

    print("Zipping and verifying...", flush=True)
    result = zip_and_verify(src_directory, zip_filename, cleanup=clean)
    if not result["success"]:
        print(f"WARNING: zip_and_verify failed: {result['errors']}", flush=True)
        # Fallback to old behavior
        print("Falling back to shutil.make_archive...", flush=True)
        shutil.make_archive(zip_filename, 'zip', src_directory)
        if clean:
            for root, dirs, files in os.walk(src_directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
    else:
        print(f"Zip verified: {result['file_count']} files in {result['zip_path']}", flush=True)

def get_parameters_baseline(dataset):
    param_sets = {
        'mnist': {
            'knn': [3, 5, 7],
            'kde': [0.5, 1, 1.5],
            'gmm': [10, 20, 30],
            'ocsvm': [0.01, 0.05, 0.1],
            'iforest': [50, 100, 150],
            'softmax': [0.9, 0.95, 0.99],
            'mahalanobis': [0.9, 0.95, 0.99]
        },
        'cifar10': {
            'knn': [5, 10, 15],
            'kde': [1, 2, 3],
            'gmm': [10, 20, 30],
            'ocsvm': [0.01, 0.05, 0.1],
            'iforest': [100, 150, 200],
            'softmax': [0.5, 0.7, 0.9],
            'mahalanobis': [0.9, 0.95, 0.99]
        },
        'cifar100': {
            'knn': [10, 15, 20],
            'kde': [1.5, 2.5, 3.5],
            'gmm': [50, 100, 150],
            'ocsvm': [0.05, 0.1, 0.2],
            'iforest': [150, 200, 250],
            'softmax': [0.3, 0.5, 0.7],
            'mahalanobis': [0.9, 0.95, 0.99]
        },
        'imagenet': {
            'knn': [10, 15, 20],
            'kde': [1.5, 2.5, 3.5],
            'gmm': [50, 100, 150],
            'ocsvm': [0.05, 0.1, 0.2],
            'iforest': [150, 200, 250],
            'softmax': [0.3, 0.5, 0.7],
            'mahalanobis': [0.9, 0.95, 0.99]
        }
    }

    return param_sets[dataset]
