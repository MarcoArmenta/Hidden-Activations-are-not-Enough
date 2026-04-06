import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Union
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR, MultiStepLR, CyclicLR
from utils.atomic_io import atomic_torch_save

from knowledgematrix.models.alexnet import AlexNet
from knowledgematrix.models.resnet18 import ResNet18
from knowledgematrix.models.vgg11 import VGG11
from model_zoo.cnn import CNN_2D
from model_zoo.mlp import MLP
from constants.constants import DEFAULT_EXPERIMENTS
from utils.utils import get_dataset, get_device, get_input_shape, get_num_classes, get_architecture


def parse_args() -> Namespace:
    """
        Return:
            The parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_name", "--experiment",
        type = str,
        default = None,
        dest = "experiment_name",
        help = "resnet_cifar100, resnet_cifar10, alexnet_cifar10, alexnet_imagenet, mlp_mnist, ..."
    )
    parser.add_argument(
        "--from_checkpoint", 
        action = 'store_true', 
        help = "Resume training from the last checkpoint."
    )
    parser.add_argument(
        "--temp_dir", 
        type = str, 
        default = None, 
        help = "Temporary path on compute node where the dataset is saved."
    )
    return parser.parse_args()


def train_one_epoch(
        model: Union[AlexNet, CNN_2D, MLP, ResNet18, VGG11],
        train_loader, 
        criterion: nn.CrossEntropyLoss, 
        optimizer: Union[optim.SGD, optim.Adam], 
        device: torch.device,
        scheduler: str
    ) -> None:
    """
        Args:
            model: the model to train.
            train_loader: the training loader.
            criterion: the criterion.
            optimizer: the optimizer.
            device: the device to train on.
    """
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        if isinstance(scheduler, CyclicLR):
            scheduler.step()


def evaluate_model(
        model: Union[AlexNet, CNN_2D, MLP, ResNet18, VGG11],
        data_loader, 
        criterion: nn.CrossEntropyLoss, 
        device: torch.device
    ):
    """
        Args:
            model: the model to evaluate.
            data_loader: the data loader.
            criterion: the criterion.
            device: the device to evaluate on.
        Returns:
            The loss and the accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total


def main() -> None:
    """
        Main function to train the model.
    """
    print("Start main")
    args = parse_args()
    if args.experiment_name is not None:
        experiment = args.experiment_name
        dataset = DEFAULT_EXPERIMENTS[experiment]['dataset']
        opt = DEFAULT_EXPERIMENTS[experiment]['optimizer']
        lr = DEFAULT_EXPERIMENTS[experiment]['lr']
        batch_size = DEFAULT_EXPERIMENTS[experiment]['batch_size']
        epochs = DEFAULT_EXPERIMENTS[experiment]['epochs']
        mom = DEFAULT_EXPERIMENTS[experiment]['momentum']
        wd = DEFAULT_EXPERIMENTS[experiment]['weight_decay']
        sched = DEFAULT_EXPERIMENTS[experiment].get('scheduler', None)
        architecture_index = DEFAULT_EXPERIMENTS[experiment]['architecture_index']
        save_every_epochs = 10

    else:
        raise ValueError("Experiment not specified in constants/constants.py")

    print(f"Training: {experiment}", flush=True)
    device = get_device()
    train_loader, test_loader = get_dataset(
        data_set = dataset,
        batch_size = batch_size,
        data_loader = True,
        data_path = args.temp_dir
    )

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)
    model = get_architecture(input_shape, num_classes, architecture_index, pretrained=True, freeze_features=True).to(device)

    # Track whether pretrained weights were loaded and conv layers frozen.
    # Only architectures -3 (AlexNet), -2 (ResNet18), -1 (VGG11) accept
    # pretrained=True / freeze_features=True; others train from scratch.
    pretrained_loaded = architecture_index in (-3, -2, -1)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if opt == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                               weight_decay=wd)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                              weight_decay=wd, momentum=mom)

    if sched == 'step':
        scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)
    elif sched == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=120)
    elif sched == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif sched == 'multi':
        scheduler = MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)
    elif sched == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=1)
    else:
        scheduler = None

    model.train()
    start_epoch = 0
    if args.from_checkpoint:
        checkpoints_path = Path(f'experiments/{experiment}/weights/')
        if checkpoints_path.exists():
            checkpoints = list(checkpoints_path.glob('epoch_*.pth'))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda cp: int(cp.stem.split('_')[1]))
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                else:
                    # Legacy checkpoint format (just state_dict)
                    model.load_state_dict(checkpoint)
                    start_epoch = int(latest_checkpoint.stem.split('_')[1]) + 1
                print(f"Resuming training from epoch {start_epoch}")

    history = {'train_acc':[],
               'test_acc':[],
               'train_loss':[],
               'test_loss':[]}

    print("Training...", flush=True)
    for epoch in range(start_epoch, epochs+1):
        if epoch == 60 and pretrained_loaded:
            # Unfreeze the feature extractor layers
            for layer in model.layers:
                if isinstance(layer, nn.Conv2d):
                    for param in layer.parameters():
                        param.requires_grad = True
            # Recreate optimizer with small fixed LR on all trainable params
            small_lr = 1e-5
            if opt == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=small_lr, weight_decay=wd)
            else:
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=small_lr, weight_decay=wd, momentum=mom)
            # Recreate scheduler with new optimizer
            if sched == 'step':
                scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)
            elif sched == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=60)  # Remaining epochs
            elif sched == 'exp':
                scheduler = ExponentialLR(optimizer, gamma=0.95)
            elif sched == 'multi':
                scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)  # Adjusted for remaining
            elif sched == 'cyclic':
                scheduler = CyclicLR(optimizer, base_lr=small_lr/10, max_lr=small_lr, step_size_up=2000)
            else:
                scheduler = None

        train_one_epoch(
            model = model, 
            train_loader = train_loader, 
            criterion = criterion, 
            optimizer = optimizer, 
            device = device,
            scheduler = scheduler
        )
        train_loss, train_accuracy = evaluate_model(
            model = model, 
            data_loader = train_loader, 
            criterion = criterion, 
            device = device
        )
        test_loss, test_accuracy = evaluate_model(
            model = model, 
            data_loader = test_loader, 
            criterion = criterion, 
            device = device
        )

        history['train_acc'].append(train_accuracy)
        history['test_acc'].append(test_accuracy)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}", flush=True)

        if scheduler is not None and not isinstance(scheduler, CyclicLR):
            scheduler.step()

        if epoch % save_every_epochs == 0 or epoch == epochs:
            os.makedirs(f'experiments/{experiment}/weights/', exist_ok=True)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'epoch': epoch,
            }
            atomic_torch_save(checkpoint, f'experiments/{experiment}/weights/epoch_{epoch}.pth')
        os.makedirs(f'experiments/{experiment}/weights/', exist_ok=True)
        with open(f'experiments/{experiment}/weights/history.json', 'w') as json_file:
            json.dump(history, json_file, indent=4)


if __name__ == "__main__":
    main()
