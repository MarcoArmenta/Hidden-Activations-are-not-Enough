import os
import json
import torch
import numpy as np


EXPERIMENT = "true labels"

# Function to accurately locate matrix.pt files
def find_matrices(base_dir):
    matrix_paths = {}
    for j in range(10):  # Considering subfolders '0' to '9'
        matrices_path = os.path.join(base_dir, str(j), 'train')
        if os.path.exists(matrices_path):
            for i in os.listdir(matrices_path):  # Iterating through each 'i' subdirectory
                matrix_file_path = os.path.join(matrices_path, i, 'matrix.pt')
                if os.path.isfile(matrix_file_path):  # Check if matrix.pt exists
                    if j not in matrix_paths:
                        matrix_paths[j] = [matrix_file_path]
                    else:
                        matrix_paths[j].append(matrix_file_path)
    return matrix_paths


# Function to load matrices and compute statistics
def compute_statistics(matrix_paths):
    statistics = {}
    for j, paths in matrix_paths.items():
        matrices = [torch.load(path) for path in paths]
        # Stack all matrices to compute statistics across all matrices in a subfolder
        stacked_matrices = torch.stack(matrices)
        # Compute mean and std across the stacked matrices
        mean_matrix = torch.mean(stacked_matrices, dim=0)
        std_matrix = torch.std(stacked_matrices, dim=0)
        # Store the computed statistics
        statistics[j] = {'mean': mean_matrix, 'std': std_matrix}
    return statistics


def compute_train_statistics(original_data, optimizer, lr, bs, epoch, verbose=False):
    #for i in range(0,max_epoch+1,5):
    #    print(f"Epoch {i}")
    # original_matrices_path = f'out_of_distribution/{original_data}_vs_{original_data}/{original_data}_{optimizer}_{model}_vs_{original_data}/100/'
    #original_matrices_path = f'data/out_of_distribution/test_sam/{original_data}_vs_{target_data}/{original_data}_{optimizer}_{model}_vs_{original_data}/{EXPERIMENT}_norm/{train_or_test}/{i}/'
    original_matrices_path = f'experiments/matrices/{original_data}/{optimizer}/{lr}/{bs}/{epoch}/'
    original_matrices_paths = find_matrices(original_matrices_path)

    # For demonstration, let's print the count of matrix files found for each of the first 10 subfolders
    #if verbose:
    #    for j in range(10):
    #        if j in original_matrices_paths:
    #            print(f"Subfolder {j} has {len(original_matrices_paths[j])} matrix files.")
    #        else:
    #            print(f"Subfolder {j} has no matrix files.")
    # Compute statistics for each subfolder
    statistics = compute_statistics(original_matrices_paths)

    # Display the computed statistics for each subfolder
    #for j in range(10):
    #    if j in statistics:
    #        mean_matrix = statistics[j]['mean']
    #        std_matrix = statistics[j]['std']
    #        if verbose:
    #            print(f"Subfolder {j} - Mean Matrix Shape: {mean_matrix.shape}, Std Matrix Shape: {std_matrix.shape}")
    #    else:
    #        if verbose:
    #            print(f"Subfolder {j} has no computed statistics.")

    # Convert tensors to lists (or numbers) for JSON serialization
    for subfolder, stats in statistics.items():
        for key, tensor in stats.items():
            if tensor.numel() == 1:  # If the tensor has only one element, convert to a Python scalar
                stats[key] = tensor.item()
            else:  # Otherwise, convert to a list
                stats[key] = tensor.tolist()

    # Define the file path where you want to save the JSON data
    # file_path = os.path.join(f'out_of_distribution/{original_data}_vs_{target_data}/',
    #                          f'{original_data}_{optimizer}_{model}_vs_{target_data}',
    #                          'matrix_statistics.json')

    # Write the converted dictionary to a JSON file
    with open(original_matrices_path + 'matrix_statistics.json', 'w') as json_file:
        json.dump(statistics, json_file, indent=4)


def predict_with_matrix_statistics(statistics, x, elipse=True):
    """
    Returns True if x belongs to the ellipse given by statistics, which has a mean and a std matrices.
    """
    if elipse:
        a = np.array(statistics['mean']).flatten()
        b = np.array(statistics['std']).flatten()
        d = (x.numpy().flatten() - a) ** 2
        sigma_sq = b ** 2

        mask = (sigma_sq != 0)
        non_zero_d = d[mask]
        non_zero_sigma = sigma_sq[mask]
        vec = non_zero_d / non_zero_sigma

        f = non_zero_d.shape[0]
        g = a.shape[0]

        #print(f"There are {g-f} dimensions of a total of {g} with variance equal to ZERO", flush=True)
        #print(f"Percentage of useless dimensions: {(g-f)/g} %", flush=True)
        # TODO: 150 needs to be tunned per NN and dataset. May go from 50 to 300...
        return np.sum(vec) <= 150**2, (g-f)/g
    else:
        a = np.array(statistics['mean']).flatten()
        b = np.array(statistics['std']).flatten()

        mask = (b != 0)
        non_zero_b = b[mask]
        m = np.linalg.norm(non_zero_b)

        f = non_zero_b.shape[0]
        g = a.shape[0]

        d = np.linalg.norm(x.numpy().flatten()[mask] - a[mask])
        return d < 1.7*m, (g-f)/g


def test_ellipse(original_data, target_data, optimizer, model, train_or_test, elipse):
    #statistics_path = f'out_of_distribution/{original_data}_vs_{original_data}/{original_data}_{optimizer}_{model}_vs_{original_data}/matrix_statistics.json'
    #statistics_path = 'out_of_distribution/mnist_vs_mnist/mnist_adam_conv_0.001_64_vs_mnist/matrix_statistics.json'
    statistics_path = f'data/out_of_distribution/test_sam/{original_data}_vs_{target_data}/{original_data}_{optimizer}_{model}_vs_{original_data}/{EXPERIMENT}_norm/{train_or_test}/100/matrix_statistics.json'

    target_matrices_path = f'data/out_of_distribution/test_sam/{original_data}_vs_{target_data}/{original_data}_{optimizer}_{model}_vs_{original_data}/{EXPERIMENT}_norm/train_or_test/100/'
    target_matrices_paths = find_matrices(target_matrices_path)

    file = open(statistics_path)
    statistics = json.load(file)

    percentages = []

    correct = 0.0
    total = 0.0

    matrices = []

    for j, paths in target_matrices_paths.items():
        # TODO: could be parallelized
        for path in paths:
            total += 1.0
            current_matrix = torch.load(path)
            #if current_matrix in matrices:
            #    input('Repeated matrixs!!!')
            #else:
            #    matrices.append(current_matrix)

            prediction_from_matrices, perc = predict_with_matrix_statistics(statistics[str(j)], current_matrix, elipse=elipse)
            # TODO: check if the perc repeats per class.
            if perc not in percentages:
                percentages.append(perc)

            if prediction_from_matrices:
                correct += 1.0

    print('Percentages of USELESS dimensions: ', percentages)
    print('Different percentages: ', len(percentages))
    return correct/total


if __name__ == '__main__':
    original_data = 'mnist'
    target_data = 'mnist'
    optimizer = 'sgd'
    train_or_test = 'test'

    if original_data == 'mnist': # 100*2 train, 8000 test
        model = 'small-mlp_0.01_8' # generalize
        max_epoch = 100
        #model = 'small-mlp_0.1_32' # underfit
    elif original_data == 'fashion':
        #model = 'small-mlp_1e-05_16'
        model = 'small-mlp_0.01_16'
        max_epoch = 100
    elif original_data == 'cifar10':
        model = 'small-mlp_0.001_8'
        max_epoch = 60

    #model = 'conv_0.001_64' # 140**2 train,
    #model = 'res_0.0001_32' # 67**2 train,

    #compute_train_statistics(model, original_data, target_data, optimizer, epoch=9, train_or_test)
    matrix_acc = test_ellipse(original_data, target_data, optimizer, model, train_or_test, elipse=True)
    print('Accuracy for ellipse: ', matrix_acc)
