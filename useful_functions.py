import os
import torch
import json


def compute_chunk_of_matrices(data, representation, epoch: int, clas, train=True, chunk_size=10, root=None, chunk_id=0) -> None:
    if root is not None:
        directory = root + '/' + str(epoch) + '/' + str(clas) + '/'

    else:
        directory = str(epoch) + '/' + str(clas) + '/'

    directory += 'train/' if train else 'test/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    data = data[chunk_id*chunk_size:(chunk_id+1)*chunk_size]

    for i, d in enumerate(data):
        idx = chunk_id*chunk_size+i
        # if matrix was already computed, pass to next sample of data
        if os.path.exists(directory+str(idx)+'/'+'matrix.pt'):
            continue
        # if the path has not been created, then no one is working on this sample
        if not os.path.exists(directory+str(idx)+'/'):
            os.makedirs(directory+str(idx)+'/')
        # if the path has been created, someone else is already computing the matrix
        else:
            continue

        rep = representation.forward(d)
        torch.save(rep, directory+str(idx)+'/'+'matrix.pt')


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


def compute_train_statistics(original_data, optimizer, lr, bs, epoch):
    original_matrices_path = f'experiments/matrices/{original_data}/{optimizer}/{lr}/{bs}/{epoch}/'
    original_matrices_paths = find_matrices(original_matrices_path)

    statistics = compute_statistics(original_matrices_paths)

    # Convert tensors to lists (or numbers) for JSON serialization
    for subfolder, stats in statistics.items():
        for key, tensor in stats.items():
            if tensor.numel() == 1:  # If the tensor has only one element, convert to a Python scalar
                stats[key] = tensor.item()
            else:  # Otherwise, convert to a list
                stats[key] = tensor.tolist()

    with open(original_matrices_path + 'matrix_statistics.json', 'w') as json_file:
        json.dump(statistics, json_file, indent=4)
