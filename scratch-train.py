import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import get_dataset, get_num_classes, get_input_shape
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR, MultiStepLR, CyclicLR
import joblib
import os
from argparse import ArgumentParser, Namespace

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from training import *

from knowledgematrix.models.alexnet import AlexNet
from knowledgematrix.models.resnet18 import ResNet18
from knowledgematrix.models.vgg11 import VGG11


def parse_args() -> Namespace:
    """
        Return:
            The parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--temp_dir",
        type = str,
        default = '/',
        help = "Temporary path on compute node where the dataset is saved."
    )
    parser.add_argument(
        "--model",
        type = str,
        default = 'resnet',
        help = "Neural net to train"
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default = 'cifar10',
        help = "Dataset nameL cifar10,100 or imagenet."
    )
    parser.add_argument(
        "--version",
        type = str,
        default = '02',
        help = "Version of hypersearch."
    )
    return parser.parse_args()


def get_device() -> torch.device:
    """Assign a specific GPU to each trial based on trial number."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device("cuda")


def train_model(trial):
    device = get_device()
    args = parse_args()
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    opt = trial.suggest_categorical("optimizer", ['adam', 'sgd'])
    mom = trial.suggest_float('momentum', 0.0, 0.99) if opt == 'sgd' else 0.0
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    sched = trial.suggest_categorical("scheduler", ['step', 'cosine', 'exp', 'multi', 'cyclic'])

    data_dir = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_loader, test_loader = get_dataset(
        data_set=args.dataset,
        batch_size=batch_size,
        data_loader=True,
        data_path=None
    )

    num_classes = get_num_classes(args.dataset)
    input_shape = get_input_shape(args.dataset)

    if args.model == 'alexnet':
        model = AlexNet(input_shape, num_classes, pretrained=True, freeze_features=True).to(device)

    elif args.model == 'resnet':
        model = ResNet18(input_shape, num_classes, pretrained=True, freeze_features=True).to(device)

    elif args.model == 'vgg':
        model = VGG11(input_shape, num_classes, pretrained=True, freeze_features=True).to(device)

    else:
        raise ValueError(f'Model {args.model} is not supported.')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if opt == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=wd)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=wd, momentum=mom)

    if sched == 'step':
        scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

    elif sched == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=120)

    elif sched == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=0.95)

    elif sched == 'multi':
        scheduler = MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)

    else:
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=1)


    model.train()
    # Train the model
    for epoch in range(100):
        if epoch == 60:
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
            else:
                scheduler = CyclicLR(optimizer, base_lr=small_lr/10, max_lr=small_lr, step_size_up=2000)

        train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler
        )
        train_loss, train_accuracy = evaluate_model(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device
        )
        test_loss, test_accuracy = evaluate_model(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device
        )

        if not isinstance(scheduler, CyclicLR):
            scheduler.step()

        print(f"Epoch {epoch}/{100}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}", flush=True)

    test_loss, _ = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    return test_loss


def save_study(study, trial):
    # /lustre07/ -> narval
    # /lustre04/ -> beluga
    # /          -> graham & nibi
    # /lustre10/ -> rorqual
    study_dir = "/lustre07/scratch/armenta/"
    if not os.path.exists(study_dir):
        os.makedirs(study_dir)
    args = parse_args()
    joblib.dump(study, f"{args.model}_{args.dataset}_{args.version}_.pkl")


if __name__ == '__main__':
    # Create a study object and specify the direction
    args = parse_args()
    print(f"Hyper-training: {args.model} on {args.dataset}", flush=True)
    storage = JournalStorage(JournalFileBackend(f'/lustre07/scratch/armenta/{args.model}_{args.dataset}_{args.version}_.log'))
    study = optuna.create_study(direction="minimize",
                                study_name=f"{args.model}_{args.dataset}_{args.version}_",
                                storage=storage,
                                load_if_exists=True)

    study.optimize(train_model,
                   n_trials=1000,
                   #n_jobs=4,
                   callbacks=[save_study])

    # Print the best hyperparameters and accuracy
    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)
