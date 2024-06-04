import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import get_architecture, get_dataset
from __init__ import DEFAULT_TRAININGS


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
        help="Optimizer to train the model with.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="The learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="The batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="The number of epochs to train.",
    )
    parser.add_argument(
        "--reduce_lr_each",
        type=int,
        default=20,
        help="The number of epochs to train.",
    )
    parser.add_argument(
        "--save_every_epochs",
        type=int,
        default=2,
        help="Weights are saved every this amount of epochs during training.",
    )
    parser.add_argument(
        "--default_hyper_parameters",
        action='store_true',
        help="If not called, trains one of the default models."
             f"{DEFAULT_TRAININGS}",
    )
    parser.add_argument(
        "--default_index",
        type=int,
        default=0,
        help="The index of the default model to train."
    )
    return parser.parse_args()


if __name__ == '__main__':
    device = 'cpu'
    args = parse_args()

    if args.default_hyper_parameters:
        index = args.default_index
        print(f'Training default model {index}')

        experiment = DEFAULT_TRAININGS[f'experiment_{index}']

        optimizer_name = experiment['optimizer']
        dataset = experiment['dataset']
        lr = experiment['lr']
        batch_size = experiment['batch_size']
        epochs = experiment['epoch']
        save_every = experiment['save_every']
        reduce_lr = experiment['reduce_lr_each']

    else:
        print(f'Training custom model')
        optimizer_name = args.optimizer
        dataset = args.dataset
        lr = args.lr
        batch_size = args.batch_size
        epochs = args.epochs
        save_every = args.save_every_epochs
        reduce_lr = args.reduce_lr_each

    dir = f"experiments/weights/{dataset}/{optimizer_name}/{lr}/{batch_size}/"
    print(f"Training: {dir}")
    if os.path.exists(dir + 'results.json'):
        print(f"Experiment '{dir}' exists.")
        exit(1)

    os.makedirs(dir, exist_ok=True)

    if optimizer_name == 'sgd':
        optimizer_name = 'sgd'
        learning_rate = lr
        momentum = 0
    elif optimizer_name == 'momentum':
        optimizer_name = 'sgd'
        momentum = 0.9
    else:
        optimizer_name = 'adam'
        learning_rate = lr

    model = get_architecture()

    train_dataset, test_dataset = get_dataset(dataset)
    os.makedirs(f'experiments/datasets/{dataset}/', exist_ok=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(),
                              momentum=momentum,
                              lr=lr)

    # Lists to store metrics
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=reduce_lr,
                                          gamma=0.1)

    print("Training...", flush=True)
    for epoch in range(epochs):
        if epoch % save_every == 0:
            torch.save(model.state_dict(), dir + f'epoch_{epoch}.pth')

        # Calculate metrics with current weights
        with torch.no_grad():
            test_loss = 0
            total_test = 0
            correct_test = 0
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs).to(device)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

            train_loss = 0
            total_train = 0
            correct_train = 0

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs).to(device)
                loss = criterion(outputs, labels)
                train_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_loss = train_loss / len(train_loader)
            test_loss = test_loss / len(test_loader)

            train_accuracy = 100 * correct_train / total_train
            test_accuracy = 100 * correct_test / total_test

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

        # Optimization steps per batch
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f"Epoch {epoch}, Training Accuracy: {train_accuracy:.4f}%, Test Accuracy: {test_accuracy:.4f}%")
        print(f"Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f'Current learning rate: {current_lr}')

    print("Training finished")

    results = {'train_acc': train_accuracies,
               'test_acc': test_accuracies,
               'train_loss': train_losses,
               'test_loss': test_losses,
               'lr': lr,
               'batch_size': batch_size,
               'dataset': dataset,
               'architecture': 'small-mlp',
               'optimizer': optimizer_name}

    with open(dir + 'results.json', "w") as json_file:
        json.dump(results, json_file)
