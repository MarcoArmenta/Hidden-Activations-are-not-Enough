import sys
import argparse
from mlp_study_exp import Experiment
from test_ellipsoids import compute_train_statistics
from manual_training import DEFAULT_TRAININGS

#  We compute num_samples per class
NUM_SAMPLES = 200
# The number of cpus, which is the same as job arrays, is equal to num_samples / size_of_chunk_id
SIZE_OF_CHUNK_ID = 100


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default="mnist",
        help="The datasets to train the model on.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        nargs="+",
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
        type=str,
        default=8,
        help="The batch size.",
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default=0,
        help="The number of epochs to train.",
    )
    parser.add_argument(
        "--default_training",
        type=bool,
        default=False,
        help="Wether to use a default trained network.",
    )
    parser.add_argument(
        "--default_index",
        type=int,
        default=0,
        help="Index of default trained networks.",
    )
    parser.add_argument(
        "--num_samples_per_class",
        type=int,
        default=20,
        help="Index of default trained networks.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Index of default trained networks.",
    )

    return parser.parse_args()


def run_experiment_values(exp, chunk_id, chunk_size, root, epoch=100, train=True):
    exp.values_on_epoch(root=root, chunk_id=chunk_id, chunk_size=chunk_size, epoch=epoch, train=train)


if __name__ == '__main__':
    device = "cpu"

    args = parse_args()
    num_samples = args.num_samples_per_class
    chunk_size = args.chunk_size

    if args.default_training:
        index = args.default_index
        experiment = DEFAULT_TRAININGS[f'experiment_{index}']

        optimizer_name = experiment['optimizer']
        dataset = experiment['dataset']
        lr = experiment['lr']
        batch_size = experiment['batch_size']
        epoch = experiment['epoch']

    else:
        optimizer_name = args.optimizer
        dataset = args.dataset
        lr = args.lr
        batch_size = args.batch_size
        epoch = args.epoch

    target_data = 'mnist'

    weights_path = f'experiments/weights/{dataset}/{optimizer_name}/{lr}/{batch_size}/'
    save_path = f'experiments/matrices/{dataset}/{optimizer_name}/{lr}/{batch_size}/'

    dict_exp = {"epochs": epoch,
                "weights_path": weights_path,
                "save_path": save_path,
                "device": device,
                "data name": dataset,
                'num_samples': num_samples}

    exp = Experiment(dict_exp)

    def compute_matrices(chunk_id, train=True):
        run_experiment_values(exp,
                              chunk_id=chunk_id,
                              root=save_path,
                              chunk_size=chunk_size,
                              epoch=epoch,
                              train=train)

    chunks = list(range(num_samples//chunk_size))

    for chunk in chunks:
        compute_matrices(chunk)
        compute_matrices(chunk, False)

    print(f"Matrices constructed.", flush=True)

    print('Computing matrix statistics.', flush=True)
    compute_train_statistics(dataset, optimizer_name, lr, batch_size, epoch)
