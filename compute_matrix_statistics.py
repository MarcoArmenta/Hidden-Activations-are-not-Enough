import argparse

from utils.utils import compute_train_statistics


def parse_args():
    """
        Args:
            parser: the parser to use.
        Returns:
            The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type = str,
        default = None,
        help = "Experiment name"
    )
    parser.add_argument(
        "--temp_dir", 
        type = str, 
        default = None,
        help = "Temporary directory to save and read data. Useful when using clusters."
    )
    return parser.parse_args()


def main():
    """
        Main function to compute the matrix statistics.
    """
    args = parse_args()
    print('Computing matrix statistics', flush=True)
    compute_train_statistics(
        experiment_name = args.experiment_name,
        path = args.temp_dir
    )


if __name__ == '__main__':
    main()
