import os
import json
import torch
import matplotlib.pyplot as plt

# Specify the directory where your files are located
data_set = 'mnist'

optimizers = ['momentum']
networks = ['small-mlp']
lrs = [0.01] #, 1e-06, 1e-7, 5e-05, 5e-06, 5e-07]
bs = [32] #8, 16, 32, 64] # [16, 32, 64]

for optimiz in optimizers:
    for network in networks:
        the_path = f'experiments/weights/{data_set}/{optimiz}'
        for lr in lrs:
            for b in bs:
                directory_path = f"{the_path}/{lr}/{b}/"
                print(directory_path)

                # Specify the prefix you want to filter by
                file_prefix = 'epoch_'

                # Get a list of all files in the directory
                try:
                    all_files = os.listdir(directory_path)
                except:
                    continue

                # Filter files that start with the specified prefix
                matching_files = sorted([filename for filename in all_files if filename.startswith(file_prefix)])

                n = [0]
                for _ in range(1, 22):
                    n.append(n[-1] + 5)

                matching_files = [f'epoch_{n[j]}.pth' for j in range(20)]

                if not os.path.exists(directory_path + '/results.json'):
                    print(f'Experiment {lr} with {b} did not finish...')
                    continue

                with open(directory_path + '/results.json', 'r') as json_file:
                    data = json.load(json_file)
                    train_losses = data['train_loss']
                    test_losses = data['test_loss']

                    learning_rate = data['lr']
                    batch_size = data['batch_size']

                    train_accuracies = data['train_acc']
                    test_accuracies = data['test_acc']

                    # Plot Losses
                    plt.figure(figsize=(12, 5))

                    plt.subplot(1, 2, 1)
                    train_losses.pop()
                    plt.plot([test_losses[0]] + train_losses, label='Train Loss')
                    plt.plot(test_losses, label='Test Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.ylim(0, 5)

                    plt.legend()
                    plt.title(f'{data_set} {network} {optimiz} - Loss, BS={batch_size}, LR={learning_rate}')
                    #plt.title(f'Model performance')

                    plt.subplot(1, 2, 2)
                    train_accuracies.pop()
                    plt.plot([test_accuracies[0]] + train_accuracies, label='Train Accuracy')
                    plt.plot(test_accuracies, label='Test Accuracy')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy (%)')
                    plt.ylim(0, 100)
                    plt.legend()
                    #plt.title(f'Model performance')
                    plt.title(f'Accuracies, BS={batch_size}, LR={learning_rate}')

                    plt.axhline(y=100, color='r', linestyle='--')

                    plt.tight_layout()
                    plt.show()
                    #plt.savefig(f'data/MLP_weights/images/{data_set}/{exp}/BS_{batch_size}_LR_{learning_rate}.png')
                    plt.close()



def show_adv_img(img: torch.Tensor) -> None:
    _, axes = plt.subplots(1, 2, figsize=(15, 15))

    axes[0].imshow(img[0], cmap='gray')
    # axes[0].set_title(f"Label: {label}")
    # axes[1].imshow(attacked_img[0], cmap='gray')
    # axes[1].set_title(f"Attack: {attack}. Output: {output}")
    for i in range(2): axes[i].axis("off")
    plt.show()

# adv_success = 'experiments/adversarial_examples/' + experiment_path + f'/adv_success_{args.subset_size//2}_{args.std}_{args.d1}_{args.d2}.pth'
# if os.path.exists(adv_success):
#    im = torch.load(adv_success)
#    for i in range(len(im)):
#        show_adv_img(im[i].detach().numpy())