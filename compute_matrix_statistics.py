import argparse

from utils.utils import compute_train_statistics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_index", type=int, default=0, help="Default index experiment")
    parser.add_argument("--temp_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print('Computing matrix statistics', flush=True)
    compute_train_statistics(args.default_index, args.temp_dir)


if __name__ == '__main__':
    main()
