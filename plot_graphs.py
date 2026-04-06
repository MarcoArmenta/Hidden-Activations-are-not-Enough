"""
    Reads history.json and plot losses and accuracies
"""

import json
import matplotlib.pyplot as plt


def plot_graphs(path: str, save: bool=True):
    with open(f"{path}history.json", "r") as json_file:
        training_results = json.load(json_file)
        train_acc = training_results["train_acc"]
        test_acc = training_results["test_acc"]
        val_acc = training_results["val_acc"]

        train_loss = training_results["train_loss"]
        test_loss = training_results["test_loss"]
        val_loss = training_results["val_loss"]

        learning_rate = training_results["lr"]
        batch_size = training_results["batch_size"]
        data_set = training_results["data_set"].upper()
        architecture = training_results["architecture"].upper()
        optimizer = training_results["type"].upper()

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Train Loss')
        if val_loss[0] != 0:
            plt.plot(val_loss, label='Validation Loss')
        plt.plot(test_loss, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 15)

        plt.legend()
        plt.title(f'{data_set} {architecture} {optimizer} - Loss, BS={batch_size}, LR={learning_rate}')

        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label='Train Accuracy')
        if val_acc[-1] != 0:
            plt.plot(val_acc, label='Validation Accuracy')
        plt.plot(test_acc, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.legend()
        plt.title(f'Accuracies, BS={batch_size}, LR={learning_rate}')

        plt.axhline(y=100, color='r', linestyle='--')

        plt.tight_layout()
        if save:
            plt.savefig(f'{path}performance.png')
            plt.close()
        else:
            plt.plot()

def plot_acc(path: str, save: bool=True) -> None:
    with open(f"{path}history.json", "r") as json_file:
        training_results = json.load(json_file)
        train_acc = training_results["train_acc"]
        train_acc = [i/100 for i in train_acc]
        test_acc = training_results["test_acc"]
        test_acc = [i/100 for i in test_acc]

        plt.plot(train_acc, label='Train Acc')
        plt.plot(test_acc, label='Test Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        if save:
            plt.savefig(f'{path}accuracy.png')
            plt.close()
        else:
            plt.plot()

def plot_loss(path: str, save: bool=True):
    with open(f"{path}history.json", "r") as json_file:
        training_results = json.load(json_file)
        train_loss = training_results["train_loss"]
        test_loss = training_results["test_loss"]
    
        plt.plot(train_loss, label='Train Loss')
        plt.plot(test_loss, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        if save:
            plt.savefig(f'{path}loss.png')
            plt.close()
        else:
            plt.plot()

plot_acc("experiments/22/weights/")
plot_loss("experiments/22/weights/")