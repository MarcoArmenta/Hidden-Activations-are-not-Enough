import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from constants.constants import DEFAULT_EXPERIMENTS
from utils.utils import get_architecture, get_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_index", type=int, default=0, help="The index for default experiment")
    parser.add_argument("--architecture_index", type=int, help="The index of the architecture to train.")
    parser.add_argument("--residual", type=int, help="Residual connections in the architecture every 4 layers.")
    parser.add_argument("--dataset", type=str, help="The dataset to train the model on.")
    parser.add_argument("--optimizer", type=str, help="Optimizer to train the model with.")
    parser.add_argument("--lr", type=float, help="The learning rate.")
    parser.add_argument("--batch_size", type=int, help="The batch size.")
    parser.add_argument("--epochs", type=int, help="The number of epochs to train.")
    parser.add_argument("--reduce_lr_each", type=int, help="Reduce learning rate every this number of epochs.")
    parser.add_argument("--save_every_epochs", type=int, help="Save weights every this number of epochs.")
    return parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        print("DEVICE: cuda")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("DEVICE: mps")
        return torch.device("mps")
    else:
        print("DEVICE: cpu")
        return torch.device("cpu")

def train_one_epoch(model, train_loader, criterion, optimizer, device):
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

def evaluate_model(model, data_loader, criterion, device):
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
    return running_loss / len(data_loader), correct / total

def main():
    args = parse_args()
    if args.default_index is not None:
        try:
            experiment = DEFAULT_EXPERIMENTS[f'experiment_{args.default_index}']
            architecture_index = experiment['architecture_index']
            dataset = experiment['dataset']
            optimizer_name = experiment['optimizer']
            lr = experiment['lr']
            batch_size = experiment['batch_size']
            epochs = experiment['epoch']
            reduce_lr_each = experiment['reduce_lr_each']
            save_every_epochs = experiment['save_every_epochs']
            residual = experiment['residual']

        except KeyError:
            print(f"Error: Default index {args.default_index} does not exist.")
            return
    else:
        architecture_index = args.architecture_index
        residual = args.residual
        dataset = args.dataset
        optimizer_name = args.optimizer
        lr = args.lr
        batch_size = args.batch_size
        epochs = args.epochs
        reduce_lr_each = args.reduce_lr_each
        save_every_epochs = args.save_every_epochs

    device = get_device()
    train_loader, test_loader = get_dataset(dataset, batch_size, data_loader=True)
    model = get_architecture(architecture_index=architecture_index, residual=residual).to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

    print("Moving model to device", flush=True)
    model = model.to(device)

    print("Training...", flush=True)
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

        train_loss, train_accuracy = evaluate_model(model, train_loader, criterion, device)
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        if epoch % reduce_lr_each == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        if epoch % save_every_epochs == 0:
            experiment_path = f'{dataset}/{architecture_index}/{optimizer_name}/{lr}/{batch_size}'
            os.makedirs(f'experiments/weights/{experiment_path}', exist_ok=True)
            torch.save(model.state_dict(), f'experiments/weights/{experiment_path}/epoch_{epoch}.pth')

if __name__ == "__main__":
    main()
