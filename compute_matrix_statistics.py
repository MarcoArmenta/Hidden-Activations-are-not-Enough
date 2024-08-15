from multiprocessing import Pool
import argparse

from utils.utils import compute_train_statistics, zip_and_cleanup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_index", type=int, default=0, help="Default index experiment")
    parser.add_argument("--temp_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print('Computing matrix statistics', flush=True)
    compute_train_statistics(args.default_index, args.temp_dir)
    #zip_and_cleanup(f'{args.temp_dir}/experiments/{args.default_index}/matrices/',f'experiments/{args.default_index}/matrices/matrices')
    #arguments = (i for i in range(18))
    #with Pool(processes=args.nb_workers) as pool:
    #    pool.map(compute_train_statistics, arguments)


if __name__ == '__main__':
    main()
